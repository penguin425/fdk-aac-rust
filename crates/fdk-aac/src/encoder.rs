//! Stateful public encoder configuration corresponding to `aacEncoder_SetParam`.
//!
//! FDK deliberately separates parameter acceptance from configuration
//! resolution: setters update user parameters and initialization flags, while
//! the next initialize operation checks cross-parameter constraints.  Keeping
//! the same split lets callers reconfigure an encoder in any order.

use std::{collections::VecDeque, fmt};

use crate::aac_encoder::{
    AacLcEncoderError, EldMpsEncoderError, PureRustAacEldMonoEncoder, PureRustAacEldMpsEncoder,
    PureRustAacEldMultichannelEncoder, PureRustAacEldStereoEncoder, PureRustAacLcMonoEncoder,
    PureRustAacLcMultichannelEncoder, PureRustAacLcStereoEncoder, PureRustAacLdMonoEncoder,
    PureRustAacLdMpsEncoder, PureRustAacLdMultichannelEncoder, PureRustAacLdStereoEncoder,
    PureRustHeAacMonoEncoder, PureRustHeAacMultichannelEncoder, PureRustHeAacPsEncoder,
    PureRustHeAacStereoEncoder,
};
use crate::adif::AdifHeader;
use crate::adts::{
    adts_crc16, adts_crc16_padded_bit_regions, sample_rate_index, AdtsHeader, MpegVersion,
};
use crate::asc::{
    AudioSpecificConfig, AudioSpecificConfigExtension, GaSpecificConfig, LdSbrHeader,
    ProgramConfig, ProgramElement,
};
use crate::bits::BitWriter;
use crate::decoder::AacLcDecoder;
use crate::encoder_metadata::{EncoderMetadata, MetadataCompressor, MetadataDrcProfile};
use crate::latm::LatmWriter;
use crate::ld_sbr::LdSbrFrequencyTables;
use crate::loas::write_loas_frame;
use crate::raw::ElementId;

pub const AACENC_INIT_NONE: u32 = 0x0000;
pub const AACENC_INIT_CONFIG: u32 = 0x0001;
pub const AACENC_INIT_STATES: u32 = 0x0002;
pub const AACENC_INIT_TRANSPORT: u32 = 0x1000;
pub const AACENC_RESET_INBUFFER: u32 = 0x2000;
pub const AACENC_INIT_ALL: u32 = 0xffff;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderParameter {
    AudioObjectType,
    Bitrate,
    BitrateMode,
    SampleRate,
    SbrMode,
    GranuleLength,
    ChannelMode,
    ChannelOrder,
    SbrRatio,
    Afterburner,
    Bandwidth,
    PeakBitrate,
    TransportMux,
    HeaderPeriod,
    SignalingMode,
    TransportSubframes,
    AudioMuxVersion,
    Protection,
    AncillaryBitrate,
    MetadataMode,
    ControlState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderParameterError {
    InvalidValue {
        parameter: EncoderParameter,
        value: u32,
    },
}

impl fmt::Display for EncoderParameterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidValue { parameter, value } => {
                write!(
                    f,
                    "invalid value {value} for AAC encoder parameter {parameter:?}"
                )
            }
        }
    }
}

impl std::error::Error for EncoderParameterError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncoderConfigurationError {
    MissingSampleRate,
    MissingChannelMode,
    InvalidFrameLength {
        audio_object_type: u32,
        frame_length: u32,
    },
    DownscaleRequiresEld,
    DownscaleWithSbr,
    DownscaleWithEldV2,
    BackwardSignalingRequiresAudioMuxVersion1,
    SingleRateSbrRequiresExplicitSignaling,
    UnsupportedTransportForAudioObjectType {
        transport_mux: u32,
        audio_object_type: u32,
    },
    InvalidBitrateMode(u32),
    InvalidBitrate(u32),
    InvalidAncillaryBitrate(u32),
    InvalidChannelBitrate,
    PeakBitrateTooLow,
    UnsupportedEldMpsSampleRate(u32),
    InvalidEldMpsFrameGeometry {
        sample_rate: u32,
        frame_length: u32,
        sbr_ratio: u32,
    },
    InvalidEldMpsSbrRatio {
        sample_rate: u32,
        sbr_ratio: u32,
    },
}

impl fmt::Display for EncoderConfigurationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingSampleRate => f.write_str("AAC encoder sample rate is not configured"),
            Self::MissingChannelMode => f.write_str("AAC encoder channel mode is not configured"),
            Self::InvalidFrameLength {
                audio_object_type,
                frame_length,
            } => write!(
                f,
                "frame length {frame_length} is invalid for audio object type {audio_object_type}"
            ),
            Self::DownscaleRequiresEld => f.write_str("downscaling requires ER AAC-ELD"),
            Self::DownscaleWithSbr => f.write_str("ELD downscaling cannot be combined with SBR"),
            Self::DownscaleWithEldV2 => {
                f.write_str("ELD downscaling cannot be combined with ELDv2")
            }
            Self::BackwardSignalingRequiresAudioMuxVersion1 => {
                f.write_str("backward-compatible LATM/LOAS signaling requires AudioMuxVersion 1")
            }
            Self::SingleRateSbrRequiresExplicitSignaling => {
                f.write_str("single-rate SBR requires explicit signaling")
            }
            Self::UnsupportedTransportForAudioObjectType {
                transport_mux,
                audio_object_type,
            } => write!(
                f,
                "transport {transport_mux} does not support audio object type {audio_object_type}"
            ),
            Self::InvalidBitrateMode(mode) => write!(f, "invalid bitrate mode {mode}"),
            Self::InvalidBitrate(rate) => write!(f, "invalid bitrate {rate}"),
            Self::InvalidAncillaryBitrate(rate) => {
                write!(f, "invalid ancillary bitrate {rate}")
            }
            Self::InvalidChannelBitrate => {
                f.write_str("bitrate has no valid encoder bandwidth configuration")
            }
            Self::PeakBitrateTooLow => f.write_str("peak bitrate is too low for the VBR mode"),
            Self::UnsupportedEldMpsSampleRate(rate) => {
                write!(f, "ELDv2/MPS does not support sample rate {rate}")
            }
            Self::InvalidEldMpsFrameGeometry {
                sample_rate,
                frame_length,
                sbr_ratio,
            } => write!(
                f,
                "ELDv2/MPS frame geometry {sample_rate} Hz/{frame_length} samples/SBR ratio {sbr_ratio} is invalid"
            ),
            Self::InvalidEldMpsSbrRatio {
                sample_rate,
                sbr_ratio,
            } => write!(
                f,
                "ELDv2/MPS SBR ratio {sbr_ratio} is invalid at {sample_rate} Hz"
            ),
        }
    }
}

impl std::error::Error for EncoderConfigurationError {}

fn aac_block_switch_lookahead(frame_length: usize) -> usize {
    let short = frame_length / 8;
    4 * short + short / 2
}

/// Reproduce `FDK_MetadataEnc_Init`: whole input frames delay the metadata,
/// while the remaining complement delays the audio presented to the core.
fn split_metadata_delay(audio_delay: usize, input_frame_length: usize) -> (usize, usize) {
    let mut metadata_frames = 0;
    let mut remainder = audio_delay as isize - input_frame_length as isize;
    while remainder > 0 {
        remainder -= input_frame_length as isize;
        metadata_frames += 1;
    }
    (metadata_frames, (-remainder) as usize)
}

fn fdk_metadata_delays(config: &ResolvedEncoderConfig) -> (usize, usize) {
    if !matches!(config.audio_object_type, 2 | 5 | 29 | 129 | 132) {
        return (0, 0);
    }

    let core_frame = config.frame_length as usize;
    let aac_delay = core_frame + aac_block_switch_lookahead(core_frame);
    if !config.sbr_active {
        return split_metadata_delay(aac_delay, core_frame);
    }

    let ratio = config.sbr_ratio as usize;
    let sbr_input_delay = if config.audio_object_type == 29 {
        // Dual-rate PS uses QMF downsampling. sbrEncoder_Init_delay balances
        // the paths with a 463-sample (1024 frame) or 435-sample (960 frame)
        // core offset: 576 + 32 + 352 + 2*offset + 1.
        debug_assert_eq!(ratio, 2);
        match core_frame {
            960 => 1_831,
            1024 => 1_887,
            _ => 0,
        }
    } else if ratio == 2 {
        // The ordinary dual-rate SBR time-domain downsampler selects the
        // Wc=480 filter, whose source-rate delay is four samples.
        4
    } else {
        0
    };
    split_metadata_delay(ratio * aac_delay + sbr_input_delay, ratio * core_frame)
}

fn fdk_encoder_delays(
    audio_object_type: u32,
    channel_mode: u32,
    sample_rate: u32,
    frame_length: u32,
    sbr_active: bool,
    sbr_ratio: u32,
) -> (u32, u32) {
    let frame = frame_length as usize;
    if audio_object_type == 39 && channel_mode == 128 {
        // SACENC_212 uses the time-domain downmix path. Its delay balancer
        // needs neither an output nor a bitstream frame buffer. The core
        // delay is F/2 in input-sample units (plus the four-sample dual-rate
        // downsampler delay); the public delay additionally includes the
        // decoder-side LD-QMF analysis and synthesis, (5/2 + 3/2) * Q.
        let qmf_bands = if sample_rate < 27_713 { 32 } else { 64 };
        let ratio = if sbr_active { sbr_ratio as usize } else { 1 };
        let core_delay = frame * ratio / 2 + usize::from(ratio == 2) * 4;
        return ((core_delay + 4 * qmf_bands) as u32, core_delay as u32);
    }
    if !sbr_active {
        let base = match audio_object_type {
            23 => frame,
            39 => frame / 2,
            _ => frame + aac_block_switch_lookahead(frame),
        };
        let metadata_delay = if matches!(audio_object_type, 2 | 129) {
            split_metadata_delay(base, frame).1
        } else {
            0
        };
        let delay = (base + metadata_delay) as u32;
        return (delay, delay);
    }

    let ratio = sbr_ratio as usize;
    if !(1..=2).contains(&ratio) {
        return (0, 0);
    }
    let sfb = 32usize << (ratio - 1);
    let slots = if frame == 1024 { 32 } else { 30 };
    let qmf_analysis = (320usize << (ratio - 1)) - sfb;
    let decoder_qmf = 6 * sfb;
    let qmf_synthesis = 1usize << (ratio - 1);
    let sbr_decoder_delay = qmf_analysis + decoder_qmf + qmf_synthesis;
    let core_coder_delay = frame + aac_block_switch_lookahead(frame);
    let is_ps = audio_object_type == 29;

    let (core_path, sbr_path, input_to_core) = if is_ps {
        let core_path = qmf_analysis
            + 32
            + qmf_analysis
            + decoder_qmf
            + 352
            + qmf_synthesis
            + core_coder_delay * ratio;
        let mut sbr_path =
            qmf_analysis + 640 + decoder_qmf + (sfb * slots - 1) + 352 + qmf_synthesis;
        while core_path > sbr_path {
            sbr_path += frame * ratio;
        }
        let core_offset = ((sbr_path - core_path) >> (ratio - 1)) as usize;
        let input_to_core = qmf_analysis + 32 + 352 + ratio * core_offset + 1;
        (core_path, sbr_path, input_to_core)
    } else {
        let downsampler_delay = usize::from(ratio == 2) * 4;
        let core_path = sbr_decoder_delay + core_coder_delay * ratio + downsampler_delay;
        let mut sbr_path = qmf_analysis + (sfb * slots - 1) + qmf_synthesis;
        while core_path > sbr_path + frame * ratio {
            sbr_path += frame * ratio;
        }
        (core_path, sbr_path, downsampler_delay)
    };
    let sbr_delay = core_path.max(sbr_path);
    let input_frame = frame * ratio;
    let metadata_audio_delay = if matches!(audio_object_type, 5 | 29 | 132) {
        split_metadata_delay(ratio * core_coder_delay + input_to_core, input_frame).1
    } else {
        0
    };
    let delay = (sbr_delay + metadata_audio_delay) as u32;
    (delay, delay.saturating_sub(sbr_decoder_delay as u32))
}

/// CBR reservoir sizing used by FDK for AAC-LD/AAC-ELD. The interpolation is
/// performed from 500 to 4000 bits over 12..70 kbit/s per input channel and is
/// then byte-aligned and capped by the AAC access-unit buffer limit.
fn low_delay_cbr_reservoir_capacity(
    bitrate: u32,
    channels: usize,
    effective_channels: usize,
    nominal_frame_bits: usize,
) -> usize {
    const MIN_BITRATE: u64 = 12_000;
    const MAX_BITRATE: u64 = 70_000;
    const MIN_RESERVOIR: u64 = 500;
    const MAX_RESERVOIR: u64 = 4_000;

    let channel_bitrate = u64::from(bitrate) / channels.max(1) as u64;
    let channel_bitrate = channel_bitrate.clamp(MIN_BITRATE, MAX_BITRATE);
    let interpolated = MIN_RESERVOIR
        + (channel_bitrate - MIN_BITRATE) * (MAX_RESERVOIR - MIN_RESERVOIR)
            / (MAX_BITRATE - MIN_BITRATE);
    let byte_aligned = interpolated as usize & !7;
    let average_bits = (nominal_frame_bits + 7) & !7;
    let maximum = 6_144usize
        .saturating_mul(effective_channels)
        .saturating_sub(average_bits);
    byte_aligned.min(maximum)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedEncoderConfig {
    pub audio_object_type: u32,
    pub sample_rate: u32,
    /// User-facing input channel mode.
    pub channel_mode: u32,
    /// Input PCM ordering: 0 MPEG, 1 WAV, 2 WG4.
    pub channel_order: u32,
    /// Core AAC channel mode after SBR/PS reconfiguration.
    pub core_channel_mode: u32,
    pub channels: usize,
    pub effective_channels: usize,
    pub frame_length: u32,
    pub downscale_factor: u32,
    pub bitrate_mode: u32,
    pub bitrate: u32,
    pub nominal_frame_bits: usize,
    pub max_bits_per_frame: Option<usize>,
    pub sbr_active: bool,
    pub sbr_ratio: u32,
    pub transport_mux: u32,
    pub signaling_mode: u32,
    pub audio_mux_version: Option<u32>,
    pub transport_subframes: u32,
    pub protection: bool,
    pub header_period: u32,
    pub bandwidth: u32,
    pub afterburner: bool,
    pub ancillary_bitrate: u32,
    pub metadata_mode: u32,
    /// Encoder delay in input-sample units, matching `AACENC_InfoStruct::nDelay`.
    pub encoder_delay: u32,
    /// Encoder delay without the decoder-side SBR delay.
    pub encoder_core_delay: u32,
}

#[derive(Debug)]
pub enum PureRustEncoderError {
    Configuration(EncoderConfigurationError),
    ConfigurationSyntax(crate::asc::AscError),
    Codec(AacLcEncoderError),
    EldMps(EldMpsEncoderError),
    UnsupportedConfiguration {
        audio_object_type: u32,
        channel_mode: u32,
        frame_length: u32,
        sbr_active: bool,
    },
    InterleavedInputLength {
        expected: usize,
        actual: usize,
    },
    InvalidAccessUnitEndMarker,
    CrcSyntax(crate::decoder::DecodeError),
    UnsupportedTransportProtection,
}

impl fmt::Display for PureRustEncoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Configuration(error) => error.fmt(f),
            Self::ConfigurationSyntax(error) => error.fmt(f),
            Self::Codec(error) => error.fmt(f),
            Self::EldMps(error) => error.fmt(f),
            Self::UnsupportedConfiguration {
                audio_object_type,
                channel_mode,
                frame_length,
                sbr_active,
            } => write!(
                f,
                "unsupported pure-Rust encoder configuration: AOT {audio_object_type}, channel mode {channel_mode}, frame length {frame_length}, SBR {sbr_active}"
            ),
            Self::InterleavedInputLength { expected, actual } => {
                write!(f, "expected {expected} interleaved samples, got {actual}")
            }
            Self::InvalidAccessUnitEndMarker => {
                f.write_str("encoded AAC access unit has no terminal ID_END marker")
            }
            Self::CrcSyntax(error) => write!(f, "cannot derive ADTS CRC syntax regions: {error}"),
            Self::UnsupportedTransportProtection => {
                f.write_str("protected ADTS output is not yet available from the unified encoder")
            }
        }
    }
}

impl std::error::Error for PureRustEncoderError {}

impl From<EncoderConfigurationError> for PureRustEncoderError {
    fn from(value: EncoderConfigurationError) -> Self {
        Self::Configuration(value)
    }
}

impl From<AacLcEncoderError> for PureRustEncoderError {
    fn from(value: AacLcEncoderError) -> Self {
        Self::Codec(value)
    }
}

impl From<EldMpsEncoderError> for PureRustEncoderError {
    fn from(value: EldMpsEncoderError) -> Self {
        Self::EldMps(value)
    }
}

impl From<crate::asc::AscError> for PureRustEncoderError {
    fn from(value: crate::asc::AscError) -> Self {
        Self::ConfigurationSyntax(value)
    }
}

impl From<crate::decoder::DecodeError> for PureRustEncoderError {
    fn from(value: crate::decoder::DecodeError) -> Self {
        Self::CrcSyntax(value)
    }
}

impl From<crate::adif::AdifError> for PureRustEncoderError {
    fn from(value: crate::adif::AdifError) -> Self {
        Self::Codec(AacLcEncoderError::from(value))
    }
}

impl From<crate::adts::AdtsError> for PureRustEncoderError {
    fn from(value: crate::adts::AdtsError) -> Self {
        Self::Codec(AacLcEncoderError::from(value))
    }
}

impl From<crate::latm::LatmError> for PureRustEncoderError {
    fn from(value: crate::latm::LatmError) -> Self {
        Self::Codec(AacLcEncoderError::from(value))
    }
}

impl From<crate::loas::LoasError> for PureRustEncoderError {
    fn from(value: crate::loas::LoasError) -> Self {
        Self::Codec(AacLcEncoderError::from(value))
    }
}

#[derive(Debug, Clone)]
enum PureRustEncoderBackend {
    LcMono(PureRustAacLcMonoEncoder),
    LcStereo(PureRustAacLcStereoEncoder),
    LcMultichannel(PureRustAacLcMultichannelEncoder),
    HeMono(PureRustHeAacMonoEncoder),
    HeStereo(PureRustHeAacStereoEncoder),
    HeMultichannel(PureRustHeAacMultichannelEncoder),
    HePs(PureRustHeAacPsEncoder),
    LdMono(PureRustAacLdMonoEncoder),
    LdStereo(PureRustAacLdStereoEncoder),
    LdMultichannel(PureRustAacLdMultichannelEncoder),
    LdMps(PureRustAacLdMpsEncoder),
    EldMono(PureRustAacEldMonoEncoder),
    EldStereo(PureRustAacEldStereoEncoder),
    EldMultichannel(PureRustAacEldMultichannelEncoder),
    EldMps(PureRustAacEldMpsEncoder),
}

impl PureRustEncoderBackend {
    fn set_bandwidth(&mut self, bandwidth: u32) {
        match self {
            Self::LcMono(encoder) => encoder.set_bandwidth(bandwidth),
            Self::LcStereo(encoder) => encoder.set_bandwidth(bandwidth),
            Self::LcMultichannel(encoder) => encoder.set_bandwidth(bandwidth),
            Self::HeMono(encoder) => encoder.set_bandwidth(bandwidth),
            Self::HeStereo(encoder) => encoder.set_bandwidth(bandwidth),
            Self::HeMultichannel(encoder) => encoder.set_bandwidth(bandwidth),
            Self::HePs(encoder) => encoder.set_bandwidth(bandwidth),
            Self::LdMono(encoder) => encoder.set_bandwidth(bandwidth),
            Self::LdStereo(encoder) => encoder.set_bandwidth(bandwidth),
            Self::LdMultichannel(encoder) => encoder.set_bandwidth(bandwidth),
            Self::LdMps(encoder) => encoder.set_bandwidth(bandwidth),
            Self::EldMono(encoder) => encoder.set_bandwidth(bandwidth),
            Self::EldStereo(encoder) => encoder.set_bandwidth(bandwidth),
            Self::EldMultichannel(encoder) => encoder.set_bandwidth(bandwidth),
            Self::EldMps(encoder) => encoder.set_bandwidth(bandwidth),
        }
    }
}

/// Unified stateful encoder constructed from the public parameter state.
#[derive(Debug, Clone)]
pub struct ConfiguredPureRustEncoder {
    config: ResolvedEncoderConfig,
    backend: PureRustEncoderBackend,
    latm_writer: Option<LatmWriter>,
    pending_access_units: Vec<Vec<u8>>,
    adif_header: Option<Vec<u8>>,
    adif_header_written: bool,
    metadata: EncoderMetadata,
    metadata_dynamic_range_gain_q16: i32,
    metadata_compression_gain_q16: Option<i32>,
    metadata_frame_gains_override: bool,
    metadata_compressor: MetadataCompressor,
    metadata_gain_delay_frames: usize,
    metadata_gain_delay: VecDeque<(i32, Option<i32>)>,
    metadata_setup_delay: VecDeque<(u32, EncoderMetadata)>,
    metadata_finalize_mode: Option<u32>,
    metadata_audio_delay: VecDeque<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PureRustEncoderParameters {
    max_channels: usize,
    audio_object_type: u32,
    bitrate: u32,
    bitrate_mode: u32,
    sample_rate: u32,
    sbr_mode: u32,
    granule_length: u32,
    downscale_factor: u32,
    channel_mode: u32,
    channel_order: u32,
    sbr_ratio: u32,
    afterburner: u32,
    bandwidth: u32,
    peak_bitrate: u32,
    transport_mux: u32,
    header_period: u32,
    signaling_mode: u32,
    transport_subframes: u32,
    audio_mux_version: u32,
    protection: u32,
    ancillary_bitrate: u32,
    metadata_mode: u32,
    control_state: u32,
    input_buffer_fill: usize,
}

impl PureRustEncoderParameters {
    pub fn new(max_channels: usize) -> Self {
        Self {
            max_channels,
            audio_object_type: 2,
            bitrate: u32::MAX,
            bitrate_mode: 0,
            sample_rate: 0,
            sbr_mode: 0xff,
            granule_length: u32::MAX,
            downscale_factor: 1,
            channel_mode: 0,
            channel_order: 0,
            sbr_ratio: 0,
            afterburner: 0,
            bandwidth: 0,
            peak_bitrate: u32::MAX,
            transport_mux: u32::MAX,
            header_period: 0xff,
            signaling_mode: 0xff,
            transport_subframes: 1,
            audio_mux_version: 0,
            protection: 0,
            ancillary_bitrate: 0,
            metadata_mode: 0,
            control_state: AACENC_INIT_ALL,
            input_buffer_fill: 0,
        }
    }

    pub fn max_channels(&self) -> usize {
        self.max_channels
    }

    pub fn initialization_flags(&self) -> u32 {
        self.control_state
    }

    pub fn input_buffer_fill(&self) -> usize {
        self.input_buffer_fill
    }

    pub fn set_input_buffer_fill(&mut self, samples_per_channel: usize) {
        self.input_buffer_fill = samples_per_channel;
    }

    pub fn clear_initialization_flags(&mut self) {
        self.control_state = AACENC_INIT_NONE;
    }

    pub fn get_parameter(&self, parameter: EncoderParameter) -> u32 {
        match parameter {
            EncoderParameter::AudioObjectType => self.audio_object_type,
            EncoderParameter::Bitrate => self.bitrate,
            EncoderParameter::BitrateMode => self.bitrate_mode,
            EncoderParameter::SampleRate => self.sample_rate,
            EncoderParameter::SbrMode => self.sbr_mode,
            EncoderParameter::GranuleLength => self.granule_length,
            EncoderParameter::ChannelMode => self.channel_mode,
            EncoderParameter::ChannelOrder => self.channel_order,
            EncoderParameter::SbrRatio => self.sbr_ratio,
            EncoderParameter::Afterburner => self.afterburner,
            EncoderParameter::Bandwidth => self.bandwidth,
            EncoderParameter::PeakBitrate => self.peak_bitrate,
            EncoderParameter::TransportMux => self.transport_mux,
            EncoderParameter::HeaderPeriod => self.header_period,
            EncoderParameter::SignalingMode => self.signaling_mode,
            EncoderParameter::TransportSubframes => self.transport_subframes,
            EncoderParameter::AudioMuxVersion => self.audio_mux_version,
            EncoderParameter::Protection => self.protection,
            EncoderParameter::AncillaryBitrate => self.ancillary_bitrate,
            EncoderParameter::MetadataMode => self.metadata_mode,
            EncoderParameter::ControlState => self.control_state,
        }
    }

    pub fn set_parameter(
        &mut self,
        parameter: EncoderParameter,
        value: u32,
    ) -> Result<(), EncoderParameterError> {
        let config = AACENC_INIT_CONFIG;
        let states = AACENC_INIT_STATES;
        let transport = AACENC_INIT_TRANSPORT;
        match parameter {
            EncoderParameter::AudioObjectType => {
                require(
                    parameter,
                    value,
                    matches!(value, 2 | 5 | 23 | 29 | 39 | 129 | 132),
                )?;
                self.update(parameter, value, config | states | transport);
            }
            EncoderParameter::Bitrate => self.update(parameter, value, config | transport),
            EncoderParameter::BitrateMode => {
                require(parameter, value, value <= 5)?;
                self.update(parameter, value, config | transport);
            }
            EncoderParameter::SampleRate => {
                require(
                    parameter,
                    value,
                    matches!(
                        value,
                        8_000
                            | 11_025
                            | 12_000
                            | 16_000
                            | 22_050
                            | 24_000
                            | 32_000
                            | 44_100
                            | 48_000
                            | 64_000
                            | 88_200
                            | 96_000
                    ),
                )?;
                if self.sample_rate != value {
                    self.input_buffer_fill = 0;
                }
                self.update(parameter, value, config | states | transport);
            }
            EncoderParameter::SbrMode => {
                // FDK stores this field as UCHAR and does not range-check it.
                if self.sbr_mode != value {
                    self.sbr_mode = value & 0xff;
                    self.control_state |= config | states | transport;
                }
            }
            EncoderParameter::GranuleLength => {
                require(
                    parameter,
                    value,
                    matches!(value, 1024 | 512 | 480 | 256 | 240 | 128 | 120),
                )?;
                if self.granule_length != value {
                    if matches!(value, 240 | 256) {
                        self.downscale_factor = 2;
                    } else if matches!(value, 120 | 128) {
                        self.downscale_factor = 4;
                    }
                    // USER_PARAM keeps this factor when a later 480/512/1024
                    // granule is selected; retain that observable behaviour.
                    self.granule_length = value;
                    self.control_state |= config | transport;
                }
            }
            EncoderParameter::ChannelMode => {
                let channels = channel_count(value);
                require(
                    parameter,
                    value,
                    channels.is_some_and(|channels| channels <= self.max_channels),
                )?;
                if self.channel_mode != value {
                    self.input_buffer_fill = 0;
                }
                let reset_states = !matches!(value, 1..=6);
                self.update(
                    parameter,
                    value,
                    config | transport | if reset_states { states } else { 0 },
                );
            }
            EncoderParameter::ChannelOrder => {
                require(parameter, value, value <= 2)?;
                if self.channel_order != value {
                    self.input_buffer_fill = 0;
                }
                self.update(parameter, value, config | states | transport);
            }
            EncoderParameter::SbrRatio => {
                require(parameter, value, value <= 2)?;
                self.update(parameter, value, config | states | transport);
            }
            EncoderParameter::Afterburner => {
                require(parameter, value, value <= 1)?;
                self.update(parameter, value, config);
            }
            EncoderParameter::Bandwidth => self.update(parameter, value, config),
            EncoderParameter::PeakBitrate => self.update(parameter, value, config | transport),
            EncoderParameter::TransportMux => {
                require(parameter, value, matches!(value, 0 | 1 | 2 | 6 | 7 | 10))?;
                self.update(parameter, value, transport);
            }
            EncoderParameter::HeaderPeriod => {
                require(parameter, value, value <= 0xff)?;
                self.update(parameter, value, transport);
            }
            EncoderParameter::SignalingMode => {
                require(parameter, value, value <= 2)?;
                self.update(parameter, value, transport);
            }
            EncoderParameter::TransportSubframes => {
                require(parameter, value, (1..=4).contains(&value))?;
                self.update(parameter, value, transport);
            }
            EncoderParameter::AudioMuxVersion => {
                require(parameter, value, value <= 2)?;
                self.update(parameter, value, transport);
            }
            EncoderParameter::Protection => {
                require(parameter, value, value <= 1)?;
                self.update(parameter, value, transport);
            }
            EncoderParameter::AncillaryBitrate => self.update(parameter, value, AACENC_INIT_NONE),
            EncoderParameter::MetadataMode => {
                require(parameter, value, value <= 3)?;
                self.update(parameter, value, config);
            }
            EncoderParameter::ControlState => {
                if value & AACENC_RESET_INBUFFER != 0 {
                    self.input_buffer_fill = 0;
                }
                self.control_state = value;
            }
        }
        Ok(())
    }

    /// Perform the cross-parameter checks deferred by `aacEncoder_SetParam`.
    pub fn resolve(&self) -> Result<ResolvedEncoderConfig, EncoderConfigurationError> {
        if self.sample_rate == 0 {
            return Err(EncoderConfigurationError::MissingSampleRate);
        }
        let channels = channel_count(self.channel_mode)
            .ok_or(EncoderConfigurationError::MissingChannelMode)?;
        let effective_channels = effective_channel_count(self.channel_mode)
            .ok_or(EncoderConfigurationError::MissingChannelMode)?;
        if self.audio_object_type == 39
            && self.channel_mode == 128
            && !(16_000..=48_000).contains(&self.sample_rate)
        {
            return Err(EncoderConfigurationError::UnsupportedEldMpsSampleRate(
                self.sample_rate,
            ));
        }
        let frame_length = if self.granule_length == u32::MAX {
            if matches!(self.audio_object_type, 23 | 39) {
                512
            } else {
                1024
            }
        } else {
            self.granule_length
        };
        let valid_frame = match self.audio_object_type {
            2 | 129 => matches!(frame_length, 960 | 1024),
            5 | 29 | 132 => matches!(frame_length, 960 | 1024),
            23 => matches!(frame_length, 480 | 512),
            39 => matches!(frame_length, 120 | 128 | 240 | 256 | 480 | 512),
            _ => false,
        };
        if !valid_frame {
            return Err(EncoderConfigurationError::InvalidFrameLength {
                audio_object_type: self.audio_object_type,
                frame_length,
            });
        }
        if self.downscale_factor > 1 && self.audio_object_type != 39 {
            return Err(EncoderConfigurationError::DownscaleRequiresEld);
        }
        if self.downscale_factor > 1 && self.sbr_mode == 1 {
            return Err(EncoderConfigurationError::DownscaleWithSbr);
        }
        if self.downscale_factor > 1 && self.channel_mode == 128 {
            return Err(EncoderConfigurationError::DownscaleWithEldV2);
        }

        let transport_mux = if self.transport_mux == u32::MAX {
            if matches!(self.audio_object_type, 23 | 39) {
                10 // TT_MP4_LOAS
            } else {
                2 // TT_MP4_ADTS
            }
        } else {
            self.transport_mux
        };
        if matches!(transport_mux, 1 | 2)
            && !matches!(self.audio_object_type, 2 | 5 | 29 | 129 | 132)
        {
            return Err(
                EncoderConfigurationError::UnsupportedTransportForAudioObjectType {
                    transport_mux,
                    audio_object_type: self.audio_object_type,
                },
            );
        }
        let mut sbr_active = matches!(self.audio_object_type, 5 | 29 | 132)
            || (self.audio_object_type == 39 && self.sbr_mode == 1);
        let mut sbr_ratio = if !sbr_active {
            0
        } else if self.sbr_ratio != 0 {
            self.sbr_ratio
        } else if self.audio_object_type == 39 {
            if self.channel_mode == 128 && self.sample_rate >= 27_713 {
                2
            } else {
                1
            }
        } else {
            2
        };
        let mut bitrate_mode = self.bitrate_mode;
        if bitrate_mode != 0 && self.peak_bitrate != u32::MAX {
            bitrate_mode = adjusted_vbr_mode(
                bitrate_mode,
                self.peak_bitrate,
                effective_channels,
                channels > 1,
            )
            .ok_or(EncoderConfigurationError::PeakBitrateTooLow)?;
        }
        let bitrate = if bitrate_mode == 0 {
            if self.bitrate == u32::MAX {
                default_cbr_bitrate(
                    effective_channels,
                    self.sample_rate,
                    self.audio_object_type == 29,
                    sbr_active,
                    self.sbr_ratio,
                    self.audio_object_type == 39,
                )
            } else {
                self.bitrate
            }
        } else {
            vbr_bitrate(bitrate_mode, effective_channels, channels > 1)
                .ok_or(EncoderConfigurationError::InvalidBitrateMode(bitrate_mode))?
        };
        if bitrate == 0 {
            return Err(EncoderConfigurationError::InvalidBitrate(0));
        }
        let ancillary_bitrate = if self.ancillary_bitrate == u32::MAX {
            if bitrate >= 192_000 {
                19_199
            } else {
                bitrate / 10
            }
        } else {
            self.ancillary_bitrate
        };
        if ancillary_bitrate >= 19_200 || u64::from(ancillary_bitrate) * 20 > u64::from(bitrate) * 3
        {
            return Err(EncoderConfigurationError::InvalidAncillaryBitrate(
                ancillary_bitrate,
            ));
        }
        let ancillary_bits = ((u64::from(ancillary_bitrate) * u64::from(frame_length))
            / u64::from(self.sample_rate)) as u32
            & !7;
        let consumed_ancillary_bitrate = ((u64::from(ancillary_bits) * u64::from(self.sample_rate))
            / u64::from(frame_length)) as u32;
        let psychoacoustic_bitrate = bitrate.saturating_sub(consumed_ancillary_bitrate);
        let mut bandwidth = resolve_bandwidth(
            self.bandwidth,
            psychoacoustic_bitrate,
            bitrate_mode,
            self.sample_rate,
            frame_length,
            effective_channels,
            channels == 1 || self.channel_mode == 128,
        )?;
        if self.audio_object_type == 39
            && self.downscale_factor == 1
            && self.sbr_mode == 0xff
            && self.sbr_ratio == 0
            && self.channel_mode != 128
        {
            if let Some(mode) = eld_auto_sbr_mode(channels, self.sample_rate, bitrate) {
                sbr_active = mode != 0;
                sbr_ratio = mode;
            }
        }
        if self.audio_object_type == 39 && self.channel_mode == 128 {
            let expected_sbr_ratio = if self.sample_rate < 27_713 { 1 } else { 2 };
            if sbr_active && sbr_ratio != expected_sbr_ratio {
                return Err(EncoderConfigurationError::InvalidEldMpsSbrRatio {
                    sample_rate: self.sample_rate,
                    sbr_ratio,
                });
            }
            let qmf_bands = if self.sample_rate < 27_713 { 32 } else { 64 };
            let spatial_frame_length = frame_length * sbr_ratio.max(1);
            if spatial_frame_length % qmf_bands != 0 {
                return Err(EncoderConfigurationError::InvalidEldMpsFrameGeometry {
                    sample_rate: self.sample_rate,
                    frame_length,
                    sbr_ratio,
                });
            }
        }
        if matches!(self.audio_object_type, 5 | 29) && sbr_active && sbr_ratio == 2 {
            bandwidth = he_aac_crossover_bandwidth(self.sample_rate, bitrate)
                .map(|(bandwidth, _)| bandwidth)
                .ok_or(EncoderConfigurationError::InvalidChannelBitrate)?;
        }
        let signaling_mode = if sbr_ratio == 0 {
            // SIG_UNKNOWN is an enum value of -1, returned through UINT by
            // aacEncoder_GetParam after initialization.
            u32::MAX
        } else if matches!(transport_mux, 0 | 2) {
            0
        } else if self.signaling_mode == 0xff {
            2
        } else {
            self.signaling_mode
        };
        if matches!(self.audio_object_type, 2 | 5 | 29 | 129 | 132)
            && matches!(transport_mux, 6 | 7 | 10)
            && signaling_mode == 1
            && self.audio_mux_version == 0
        {
            return Err(EncoderConfigurationError::BackwardSignalingRequiresAudioMuxVersion1);
        }
        if matches!(self.audio_object_type, 2 | 5 | 29 | 129 | 132)
            && signaling_mode == 0
            && sbr_ratio == 1
        {
            return Err(EncoderConfigurationError::SingleRateSbrRequiresExplicitSignaling);
        }
        let frame_bits = |rate: u32| {
            let input_samples = u64::from(frame_length)
                * u64::from(if self.audio_object_type == 39 {
                    sbr_ratio.max(1)
                } else {
                    1
                });
            ((u64::from(rate) * input_samples + u64::from(self.sample_rate) - 1)
                / u64::from(self.sample_rate)) as usize
        };
        let nominal_frame_bits = frame_bits(bitrate);
        let max_bits_per_frame = (self.peak_bitrate != u32::MAX).then(|| {
            let bits = frame_bits(self.peak_bitrate.max(bitrate));
            (bits + 7) & !7
        });

        let metadata_mode = if matches!(self.audio_object_type, 2 | 5 | 29 | 129 | 132) {
            self.metadata_mode
        } else {
            0
        };
        let (encoder_delay, encoder_core_delay) = fdk_encoder_delays(
            self.audio_object_type,
            self.channel_mode,
            self.sample_rate,
            frame_length,
            sbr_active,
            sbr_ratio,
        );

        Ok(ResolvedEncoderConfig {
            audio_object_type: self.audio_object_type,
            sample_rate: self.sample_rate,
            channel_mode: self.channel_mode,
            channel_order: self.channel_order,
            core_channel_mode: if self.audio_object_type == 29 {
                1
            } else {
                self.channel_mode
            },
            channels,
            effective_channels,
            frame_length,
            downscale_factor: self.downscale_factor,
            bitrate_mode,
            bitrate,
            nominal_frame_bits,
            max_bits_per_frame,
            sbr_active,
            sbr_ratio,
            transport_mux,
            signaling_mode,
            audio_mux_version: matches!(transport_mux, 6 | 7 | 10)
                .then_some(self.audio_mux_version),
            transport_subframes: self.transport_subframes,
            protection: self.protection != 0,
            header_period: if self.header_period != 0xff {
                self.header_period
            } else if matches!(transport_mux, 2 | 6 | 10) {
                10
            } else {
                0
            },
            bandwidth,
            afterburner: self.afterburner != 0,
            ancillary_bitrate,
            metadata_mode,
            encoder_delay,
            encoder_core_delay,
        })
    }

    fn update(&mut self, parameter: EncoderParameter, value: u32, flags: u32) {
        let target = match parameter {
            EncoderParameter::AudioObjectType => &mut self.audio_object_type,
            EncoderParameter::Bitrate => &mut self.bitrate,
            EncoderParameter::BitrateMode => &mut self.bitrate_mode,
            EncoderParameter::SampleRate => &mut self.sample_rate,
            EncoderParameter::SbrMode => &mut self.sbr_mode,
            EncoderParameter::GranuleLength => &mut self.granule_length,
            EncoderParameter::ChannelMode => &mut self.channel_mode,
            EncoderParameter::ChannelOrder => &mut self.channel_order,
            EncoderParameter::SbrRatio => &mut self.sbr_ratio,
            EncoderParameter::Afterburner => &mut self.afterburner,
            EncoderParameter::Bandwidth => &mut self.bandwidth,
            EncoderParameter::PeakBitrate => &mut self.peak_bitrate,
            EncoderParameter::TransportMux => &mut self.transport_mux,
            EncoderParameter::HeaderPeriod => &mut self.header_period,
            EncoderParameter::SignalingMode => &mut self.signaling_mode,
            EncoderParameter::TransportSubframes => &mut self.transport_subframes,
            EncoderParameter::AudioMuxVersion => &mut self.audio_mux_version,
            EncoderParameter::Protection => &mut self.protection,
            EncoderParameter::AncillaryBitrate => &mut self.ancillary_bitrate,
            EncoderParameter::MetadataMode => &mut self.metadata_mode,
            EncoderParameter::ControlState => unreachable!(),
        };
        if *target != value {
            *target = value;
            self.control_state |= flags;
        }
    }
}

impl Default for PureRustEncoderParameters {
    fn default() -> Self {
        Self::new(2)
    }
}

impl ConfiguredPureRustEncoder {
    pub fn from_parameters(
        parameters: &PureRustEncoderParameters,
    ) -> Result<Self, PureRustEncoderError> {
        let config = parameters.resolve()?;
        let index = sample_rate_index(config.sample_rate)
            .ok_or(EncoderConfigurationError::MissingSampleRate)?;
        let default_core_capacity = 6144usize
            .saturating_mul(if config.audio_object_type == 29 {
                1
            } else {
                config.effective_channels
            })
            .saturating_sub(config.nominal_frame_bits);
        let core_capacity =
            if matches!(config.audio_object_type, 23 | 39) && config.bitrate_mode == 0 {
                low_delay_cbr_reservoir_capacity(
                    config.bitrate,
                    config.channels,
                    config.effective_channels,
                    config.nominal_frame_bits,
                )
            } else {
                default_core_capacity
            };
        let mut backend = match (
            config.audio_object_type,
            config.channel_mode,
            config.frame_length,
            config.sbr_active,
        ) {
            (2 | 129, 1, 960 | 1024, false) | (2 | 129, 128, 1024, false) => {
                PureRustEncoderBackend::LcMono(PureRustAacLcMonoEncoder::new_with_frame_length(
                    index,
                    config.frame_length as usize,
                    config.nominal_frame_bits,
                    core_capacity,
                )?)
            }
            (2 | 129, 2, 960 | 1024, false) => {
                PureRustEncoderBackend::LcStereo(PureRustAacLcStereoEncoder::new_with_frame_length(
                    index,
                    config.frame_length as usize,
                    config.nominal_frame_bits,
                    core_capacity,
                )?)
            }
            (2 | 129, 3..=7 | 11 | 12 | 14 | 33 | 34, 960 | 1024, false) => {
                PureRustEncoderBackend::LcMultichannel(
                    PureRustAacLcMultichannelEncoder::new_with_channel_mode(
                        index,
                        config.channels,
                        config.channel_mode,
                        config.frame_length as usize,
                        config.nominal_frame_bits,
                        core_capacity,
                    )?,
                )
            }
            (5 | 132, 1, 960 | 1024, true) | (5 | 132, 128, 1024, true)
                if config.sbr_ratio == 2 =>
            {
                let core_index = sample_rate_index(config.sample_rate / 2)
                    .ok_or(EncoderConfigurationError::MissingSampleRate)?;
                let sbr_header = he_aac_sbr_header(config.sample_rate / 2, config.bitrate)
                    .ok_or(EncoderConfigurationError::InvalidChannelBitrate)?;
                PureRustEncoderBackend::HeMono(PureRustHeAacMonoEncoder::new_with_frame_length(
                    core_index,
                    config.sample_rate,
                    config.frame_length as usize,
                    config.nominal_frame_bits,
                    core_capacity,
                    sbr_header,
                )?)
            }
            (5 | 132, 2, 960 | 1024, true) if config.sbr_ratio == 2 => {
                let core_index = sample_rate_index(config.sample_rate / 2)
                    .ok_or(EncoderConfigurationError::MissingSampleRate)?;
                let sbr_header = he_aac_sbr_header(config.sample_rate / 2, config.bitrate)
                    .ok_or(EncoderConfigurationError::InvalidChannelBitrate)?;
                PureRustEncoderBackend::HeStereo(PureRustHeAacStereoEncoder::new_with_frame_length(
                    core_index,
                    config.sample_rate,
                    config.frame_length as usize,
                    config.nominal_frame_bits,
                    core_capacity,
                    sbr_header,
                )?)
            }
            (5 | 132, 3..=7 | 11 | 12 | 14 | 33 | 34, 960 | 1024, true)
                if config.sbr_ratio == 2 =>
            {
                let core_index = sample_rate_index(config.sample_rate / 2)
                    .ok_or(EncoderConfigurationError::MissingSampleRate)?;
                let sbr_header = he_aac_sbr_header(config.sample_rate / 2, config.bitrate)
                    .ok_or(EncoderConfigurationError::InvalidChannelBitrate)?;
                PureRustEncoderBackend::HeMultichannel(
                    PureRustHeAacMultichannelEncoder::new_with_channel_mode(
                        core_index,
                        config.sample_rate,
                        config.channels,
                        config.channel_mode,
                        config.frame_length as usize,
                        config.nominal_frame_bits,
                        core_capacity,
                        sbr_header,
                    )?,
                )
            }
            (29, 2, 960 | 1024, true) if config.sbr_ratio == 2 => {
                let core_index = sample_rate_index(config.sample_rate / 2)
                    .ok_or(EncoderConfigurationError::MissingSampleRate)?;
                let sbr_header = he_aac_sbr_header(config.sample_rate / 2, config.bitrate)
                    .ok_or(EncoderConfigurationError::InvalidChannelBitrate)?;
                PureRustEncoderBackend::HePs(PureRustHeAacPsEncoder::new_with_frame_length(
                    core_index,
                    config.sample_rate,
                    config.frame_length as usize,
                    config.nominal_frame_bits,
                    core_capacity,
                    sbr_header,
                )?)
            }
            (23, 1, 480 | 512, false) => {
                PureRustEncoderBackend::LdMono(PureRustAacLdMonoEncoder::new(
                    index,
                    config.frame_length as usize,
                    config.nominal_frame_bits,
                    core_capacity,
                )?)
            }
            (23, 2, 480 | 512, false) => {
                PureRustEncoderBackend::LdStereo(PureRustAacLdStereoEncoder::new(
                    index,
                    config.frame_length as usize,
                    config.nominal_frame_bits,
                    core_capacity,
                )?)
            }
            (23, 128, 480 | 512, false) => {
                PureRustEncoderBackend::LdMps(PureRustAacLdMpsEncoder::new(
                    index,
                    config.frame_length as usize,
                    config.nominal_frame_bits,
                    core_capacity,
                )?)
            }
            (23, 3..=7 | 11 | 12 | 14, 480 | 512, false) => PureRustEncoderBackend::LdMultichannel(
                PureRustAacLdMultichannelEncoder::new_with_channel_mode(
                    index,
                    config.channels,
                    config.channel_mode,
                    config.frame_length as usize,
                    config.nominal_frame_bits,
                    core_capacity,
                )?,
            ),
            (39, 1, 480 | 512, false) if config.downscale_factor == 1 => {
                PureRustEncoderBackend::EldMono(PureRustAacEldMonoEncoder::new(
                    index,
                    config.frame_length as usize,
                    config.nominal_frame_bits,
                    core_capacity,
                )?)
            }
            (39, 1, 480 | 512, true) if config.downscale_factor == 1 => {
                let dual_rate = config.sbr_ratio == 2;
                let core_rate = config.sample_rate / config.sbr_ratio.max(1);
                let core_index = sample_rate_index(core_rate)
                    .ok_or(EncoderConfigurationError::MissingSampleRate)?;
                let header = eld_mono_sbr_header(core_rate, config.bitrate, dual_rate)
                    .ok_or(EncoderConfigurationError::InvalidChannelBitrate)?;
                let mut encoder = PureRustAacEldMonoEncoder::new(
                    core_index,
                    config.frame_length as usize,
                    config.nominal_frame_bits,
                    core_capacity,
                )?;
                encoder.enable_sbr(header, dual_rate)?;
                encoder
                    .set_sbr_noise_max_level(eld_mono_noise_max_level(core_rate, config.bitrate));
                PureRustEncoderBackend::EldMono(encoder)
            }
            (39, 2, 480 | 512, false) if config.downscale_factor == 1 => {
                PureRustEncoderBackend::EldStereo(PureRustAacEldStereoEncoder::new(
                    index,
                    config.frame_length as usize,
                    config.nominal_frame_bits,
                    core_capacity,
                )?)
            }
            (39, 2, 480 | 512, true) if config.downscale_factor == 1 => {
                let dual_rate = config.sbr_ratio == 2;
                let core_rate = config.sample_rate / config.sbr_ratio.max(1);
                let core_index = sample_rate_index(core_rate)
                    .ok_or(EncoderConfigurationError::MissingSampleRate)?;
                let header = eld_stereo_sbr_header(core_rate, config.bitrate, dual_rate)
                    .ok_or(EncoderConfigurationError::InvalidChannelBitrate)?;
                let mut encoder = PureRustAacEldStereoEncoder::new(
                    core_index,
                    config.frame_length as usize,
                    config.nominal_frame_bits,
                    core_capacity,
                )?;
                encoder.enable_sbr(header, dual_rate)?;
                PureRustEncoderBackend::EldStereo(encoder)
            }
            (39, 3..=7 | 11 | 12 | 14, 480 | 512, false) if config.downscale_factor == 1 => {
                PureRustEncoderBackend::EldMultichannel(
                    PureRustAacEldMultichannelEncoder::new_with_channel_mode(
                        index,
                        config.channels,
                        config.channel_mode,
                        config.frame_length as usize,
                        config.nominal_frame_bits,
                        core_capacity,
                    )?,
                )
            }
            (39, 3..=7 | 11 | 12 | 14, 480 | 512, true) if config.downscale_factor == 1 => {
                let dual_rate = config.sbr_ratio == 2;
                let core_rate = config.sample_rate / config.sbr_ratio.max(1);
                let core_index = sample_rate_index(core_rate)
                    .ok_or(EncoderConfigurationError::MissingSampleRate)?;
                let (multichannel_headers, element_bitrates, noise_max_levels) =
                    eld_multichannel_sbr_headers(
                        core_rate,
                        config.bitrate,
                        config.channels,
                        config.channel_mode,
                        dual_rate,
                    )
                    .ok_or(EncoderConfigurationError::InvalidChannelBitrate)?;
                let mut encoder = PureRustAacEldMultichannelEncoder::new_with_channel_mode(
                    core_index,
                    config.channels,
                    config.channel_mode,
                    config.frame_length as usize,
                    config.nominal_frame_bits,
                    core_capacity,
                )?;
                encoder.enable_sbr_headers_with_bitrates(
                    multichannel_headers,
                    element_bitrates,
                    dual_rate,
                )?;
                encoder.set_sbr_noise_max_levels(&noise_max_levels)?;
                PureRustEncoderBackend::EldMultichannel(encoder)
            }
            (39, 128, 480 | 512, false) if config.downscale_factor == 1 => {
                PureRustEncoderBackend::EldMps(PureRustAacEldMpsEncoder::new(
                    index,
                    config.frame_length as usize,
                    config.nominal_frame_bits,
                    core_capacity,
                )?)
            }
            (39, 128, 480 | 512, true) if config.downscale_factor == 1 => {
                let dual_rate = config.sbr_ratio == 2;
                let core_rate = config.sample_rate / config.sbr_ratio.max(1);
                let core_index = sample_rate_index(core_rate)
                    .ok_or(EncoderConfigurationError::MissingSampleRate)?;
                let mut header = eld_mono_sbr_header(core_rate, config.bitrate, dual_rate)
                    .ok_or(EncoderConfigurationError::InvalidChannelBitrate)?;
                // The ELDv2/SAC path forces 1.5 dB envelope resolution in
                // FDKsbrEnc_EnvInit, unlike the ordinary mono ELD default.
                header.amp_resolution = true;
                let mut encoder = PureRustAacEldMpsEncoder::new_with_spatial_geometry(
                    core_index,
                    config.frame_length as usize,
                    config.nominal_frame_bits,
                    core_capacity,
                    config.sample_rate,
                    config.frame_length as usize * config.sbr_ratio as usize,
                )?;
                encoder.enable_sbr(header, dual_rate)?;
                encoder
                    .set_sbr_noise_max_level(eld_mono_noise_max_level(core_rate, config.bitrate));
                PureRustEncoderBackend::EldMps(encoder)
            }
            _ => {
                return Err(PureRustEncoderError::UnsupportedConfiguration {
                    audio_object_type: config.audio_object_type,
                    channel_mode: config.channel_mode,
                    frame_length: config.frame_length,
                    sbr_active: config.sbr_active,
                });
            }
        };
        let cbr_fill_enabled = config.bitrate_mode == 0;
        match &mut backend {
            PureRustEncoderBackend::LcMono(encoder) => {
                encoder.set_afterburner(config.afterburner);
                encoder.set_bitrate_mode(config.bitrate_mode);
            }
            PureRustEncoderBackend::LcStereo(encoder) => {
                encoder.set_afterburner(config.afterburner);
                encoder.set_bitrate_mode(config.bitrate_mode);
            }
            PureRustEncoderBackend::LcMultichannel(encoder) => {
                encoder.set_afterburner(config.afterburner);
                encoder.set_bitrate_mode(config.bitrate_mode);
            }
            PureRustEncoderBackend::HeMono(encoder) => {
                encoder.set_afterburner(config.afterburner);
                encoder.set_bitrate_mode(config.bitrate_mode);
            }
            PureRustEncoderBackend::HeStereo(encoder) => {
                encoder.set_afterburner(config.afterburner);
                encoder.set_bitrate_mode(config.bitrate_mode);
            }
            PureRustEncoderBackend::HeMultichannel(encoder) => {
                encoder.set_afterburner(config.afterburner);
                encoder.set_bitrate_mode(config.bitrate_mode);
            }
            PureRustEncoderBackend::HePs(encoder) => {
                encoder.set_afterburner(config.afterburner);
                encoder.set_bitrate_mode(config.bitrate_mode);
            }
            PureRustEncoderBackend::LdMono(encoder) => {
                encoder.set_cbr_fill_enabled(cbr_fill_enabled);
                encoder.set_afterburner(config.afterburner);
                encoder.set_bitrate_mode(config.bitrate_mode);
            }
            PureRustEncoderBackend::LdStereo(encoder) => {
                encoder.set_cbr_fill_enabled(cbr_fill_enabled);
                encoder.set_afterburner(config.afterburner);
                encoder.set_bitrate_mode(config.bitrate_mode);
            }
            PureRustEncoderBackend::LdMultichannel(encoder) => {
                encoder.set_cbr_fill_enabled(cbr_fill_enabled);
                encoder.set_afterburner(config.afterburner);
                encoder.set_bitrate_mode(config.bitrate_mode);
            }
            PureRustEncoderBackend::LdMps(encoder) => {
                encoder.set_cbr_fill_enabled(cbr_fill_enabled);
                encoder.set_afterburner(config.afterburner);
                encoder.set_bitrate_mode(config.bitrate_mode);
            }
            PureRustEncoderBackend::EldMono(encoder) => {
                encoder.set_cbr_fill_enabled(cbr_fill_enabled);
                encoder.set_afterburner(config.afterburner);
                encoder.set_bitrate_mode(config.bitrate_mode);
            }
            PureRustEncoderBackend::EldStereo(encoder) => {
                encoder.set_cbr_fill_enabled(cbr_fill_enabled);
                encoder.set_afterburner(config.afterburner);
                encoder.set_bitrate_mode(config.bitrate_mode);
            }
            PureRustEncoderBackend::EldMultichannel(encoder) => {
                encoder.set_cbr_fill_enabled(cbr_fill_enabled);
                encoder.set_afterburner(config.afterburner);
            }
            PureRustEncoderBackend::EldMps(encoder) => {
                encoder.set_cbr_fill_enabled(cbr_fill_enabled);
                encoder.set_afterburner(config.afterburner);
                encoder.set_bitrate_mode(config.bitrate_mode);
            }
        }
        backend.set_bandwidth(config.bandwidth);
        let asc = backend_audio_specific_config(&config, &backend)?;
        let latm_writer = if matches!(config.transport_mux, 6 | 7 | 10) {
            Some(LatmWriter::new_with_sbr_signaling(
                asc,
                u8::from(config.audio_mux_version.unwrap_or(0) != 0),
                config.transport_subframes as u8,
                config.header_period as u8,
                config.signaling_mode as u8,
            )?)
        } else {
            None
        };
        let adif_header = if config.transport_mux == 1 {
            Some(build_adif_header(&config)?.to_bytes()?)
        } else {
            None
        };
        let metadata_frame_length = config.frame_length as usize * config.sbr_ratio.max(1) as usize;
        let metadata_compressor =
            MetadataCompressor::new(config.sample_rate, metadata_frame_length, config.channels);
        let (metadata_gain_delay_frames, metadata_audio_delay_samples) =
            fdk_metadata_delays(&config);
        let metadata_audio_delay_values = metadata_audio_delay_samples * config.channels;
        Ok(Self {
            config,
            backend,
            latm_writer,
            pending_access_units: Vec::new(),
            adif_header,
            adif_header_written: false,
            metadata: EncoderMetadata::default(),
            metadata_dynamic_range_gain_q16: 0,
            metadata_compression_gain_q16: None,
            metadata_frame_gains_override: false,
            metadata_compressor,
            metadata_gain_delay_frames,
            metadata_gain_delay: VecDeque::new(),
            metadata_setup_delay: VecDeque::new(),
            metadata_finalize_mode: None,
            metadata_audio_delay: std::iter::repeat_n(0.0, metadata_audio_delay_values).collect(),
        })
    }

    pub fn config(&self) -> &ResolvedEncoderConfig {
        &self.config
    }

    /// Total input-to-output encoder delay in input-sample units.
    pub fn encoder_delay(&self) -> u32 {
        self.config.encoder_delay
    }

    /// Encoder delay excluding decoder-side SBR/MPS processing.
    pub fn encoder_core_delay(&self) -> u32 {
        self.config.encoder_core_delay
    }

    pub fn input_samples_per_channel(&self) -> usize {
        self.config.frame_length as usize
            * if matches!(
                &self.backend,
                PureRustEncoderBackend::HeMono(_)
                    | PureRustEncoderBackend::HeStereo(_)
                    | PureRustEncoderBackend::HeMultichannel(_)
                    | PureRustEncoderBackend::HePs(_)
            ) {
                2
            } else if matches!(
                &self.backend,
                PureRustEncoderBackend::EldMono(_)
                    | PureRustEncoderBackend::EldStereo(_)
                    | PureRustEncoderBackend::EldMultichannel(_)
                    | PureRustEncoderBackend::EldMps(_)
            ) {
                self.config.sbr_ratio.max(1) as usize
            } else {
                1
            }
    }

    /// Return the sticky metadata setup used for subsequent access units.
    pub fn metadata(&self) -> &EncoderMetadata {
        &self.metadata
    }

    /// Replace the sticky metadata setup, equivalent to supplying an
    /// `IN_METADATA_SETUP` buffer to the C encoder.
    pub fn set_metadata(&mut self, metadata: EncoderMetadata) {
        self.metadata = metadata;
    }

    /// Change metadata serialization while preserving the running codec and
    /// audio-delay state.  This mirrors a live `AACENC_METADATA_MODE`
    /// reinitialization: enabling starts with disabled delay-line entries,
    /// while disabling queues one final default metadata frame in the old
    /// format so decoders receive an explicit reset.
    pub fn set_metadata_mode(&mut self, mode: u32) -> Result<(), EncoderParameterError> {
        if mode > 3 || mode != 0 && !matches!(self.config.audio_object_type, 2 | 5 | 29 | 129 | 132)
        {
            return Err(EncoderParameterError::InvalidValue {
                parameter: EncoderParameter::MetadataMode,
                value: mode,
            });
        }
        let previous = self.config.metadata_mode;
        if previous == mode {
            return Ok(());
        }
        if previous == 0 && mode != 0 {
            self.metadata_setup_delay.clear();
            self.metadata_setup_delay.resize(
                self.metadata_gain_delay_frames,
                (0, EncoderMetadata::default()),
            );
            self.metadata_gain_delay.clear();
        } else if previous != 0 && mode == 0 {
            self.metadata_finalize_mode = Some(previous);
        }
        if matches!(mode, 1 | 2) {
            let frame_length =
                self.config.frame_length as usize * self.config.sbr_ratio.max(1) as usize;
            self.metadata_compressor = MetadataCompressor::new(
                self.config.sample_rate,
                frame_length,
                self.config.channels,
            );
        }
        self.config.metadata_mode = mode;
        Ok(())
    }

    /// Set already-computed per-frame compressor gains in signed Q16 dB.
    /// This is primarily useful for bitstream-level interoperability while
    /// the stateful compressor consumes the same sticky metadata setup.
    pub fn set_metadata_frame_gains(
        &mut self,
        dynamic_range_gain_q16: i32,
        compression_gain_q16: Option<i32>,
    ) {
        self.metadata_dynamic_range_gain_q16 = dynamic_range_gain_q16;
        self.metadata_compression_gain_q16 = compression_gain_q16;
        self.metadata_frame_gains_override = true;
    }

    /// Resume PCM-derived metadata gain generation after an explicit frame
    /// gain override.
    pub fn clear_metadata_frame_gains_override(&mut self) {
        self.metadata_frame_gains_override = false;
    }

    fn delay_metadata_gains(&mut self, current: (i32, Option<i32>)) -> (i32, Option<i32>) {
        if self.metadata_gain_delay_frames == 0 {
            return current;
        }
        if self.metadata_gain_delay.is_empty() {
            self.metadata_gain_delay
                .resize(self.metadata_gain_delay_frames, current);
            return current;
        }
        let delayed = self.metadata_gain_delay.pop_front().unwrap_or(current);
        self.metadata_gain_delay.push_back(current);
        delayed
    }

    fn delay_metadata_setup(&mut self, current: (u32, EncoderMetadata)) -> (u32, EncoderMetadata) {
        if self.metadata_gain_delay_frames == 0 {
            return current;
        }
        if self.metadata_setup_delay.is_empty() {
            self.metadata_setup_delay
                .resize(self.metadata_gain_delay_frames, current.clone());
            return current;
        }
        let delayed = self
            .metadata_setup_delay
            .pop_front()
            .unwrap_or_else(|| current.clone());
        self.metadata_setup_delay.push_back(current);
        delayed
    }

    fn delay_metadata_audio(&mut self, input: &[f32]) -> Vec<f32> {
        input
            .iter()
            .map(|&sample| {
                let delayed = self.metadata_audio_delay.pop_front().unwrap_or(sample);
                self.metadata_audio_delay.push_back(sample);
                delayed
            })
            .collect()
    }

    /// Maximum number of ancillary bytes that can be consumed for one access
    /// unit under the configured FDK ancillary-rate policy.
    pub fn max_ancillary_bytes_per_access_unit(&self) -> usize {
        if self.config.ancillary_bitrate != 0 {
            let bits = (u64::from(self.config.ancillary_bitrate)
                * u64::from(self.config.frame_length)
                / u64::from(self.config.sample_rate)) as usize;
            return (bits & !7) >> 3;
        }
        let available_rate = self
            .config
            .bitrate
            .saturating_sub(self.config.channels as u32 * 8_000);
        (((u64::from(available_rate) * u64::from(self.config.frame_length)
            / u64::from(self.config.sample_rate)) as usize)
            >> 3)
            .min(256)
    }

    /// Encode one frame of channel-interleaved PCM into a raw AAC access unit.
    pub fn encode_interleaved_f32(
        &mut self,
        input: &[f32],
    ) -> Result<Vec<u8>, PureRustEncoderError> {
        Ok(self.encode_interleaved_f32_with_ancillary(input, &[])?.0)
    }

    fn encode_core_interleaved_f32(
        &mut self,
        input: &[f32],
    ) -> Result<Vec<u8>, PureRustEncoderError> {
        let per_channel = self.input_samples_per_channel();
        let expected = per_channel * self.config.channels;
        if input.len() != expected {
            return Err(PureRustEncoderError::InterleavedInputLength {
                expected,
                actual: input.len(),
            });
        }
        match &mut self.backend {
            PureRustEncoderBackend::LcMono(encoder) => {
                if self.config.channel_mode == 128 {
                    Ok(encoder.encode_raw_data_block(&downmix_stereo_to_mono(input))?)
                } else {
                    Ok(encoder.encode_raw_data_block(input)?)
                }
            }
            PureRustEncoderBackend::HeMono(encoder) => {
                if self.config.channel_mode == 128 {
                    Ok(encoder.encode_raw_data_block(&downmix_stereo_to_mono(input))?)
                } else {
                    Ok(encoder.encode_raw_data_block(input)?)
                }
            }
            PureRustEncoderBackend::LdMono(encoder) => Ok(encoder.encode_pcm(input)?),
            PureRustEncoderBackend::EldMono(encoder) => Ok(encoder.encode_pcm(input)?),
            PureRustEncoderBackend::LcStereo(encoder) => {
                let (left, right) = deinterleave_stereo(input);
                Ok(encoder.encode_raw_data_block(&left, &right)?)
            }
            PureRustEncoderBackend::LcMultichannel(encoder) => {
                let channels = deinterleave_channels(
                    input,
                    self.config.channels,
                    self.config.channel_mode,
                    self.config.channel_order,
                );
                Ok(encoder.encode_raw_data_block(&channels)?)
            }
            PureRustEncoderBackend::HePs(encoder) => {
                let (left, right) = deinterleave_stereo(input);
                Ok(encoder.encode_raw_data_block(&left, &right)?)
            }
            PureRustEncoderBackend::HeStereo(encoder) => {
                let (left, right) = deinterleave_stereo(input);
                Ok(encoder.encode_raw_data_block(&left, &right)?)
            }
            PureRustEncoderBackend::HeMultichannel(encoder) => {
                let channels = deinterleave_channels(
                    input,
                    self.config.channels,
                    self.config.channel_mode,
                    self.config.channel_order,
                );
                Ok(encoder.encode_raw_data_block(&channels)?)
            }
            PureRustEncoderBackend::LdStereo(encoder) => {
                let (left, right) = deinterleave_stereo(input);
                Ok(encoder.encode_pcm(&left, &right)?)
            }
            PureRustEncoderBackend::LdMultichannel(encoder) => {
                let channels = deinterleave_channels(
                    input,
                    self.config.channels,
                    self.config.channel_mode,
                    self.config.channel_order,
                );
                Ok(encoder.encode_pcm(&channels)?)
            }
            PureRustEncoderBackend::LdMps(encoder) => {
                let (left, right) = deinterleave_stereo(input);
                Ok(encoder.encode_pcm_with_extensions(&left, &right, None, &[])?)
            }
            PureRustEncoderBackend::EldStereo(encoder) => {
                let (left, right) = deinterleave_stereo(input);
                Ok(encoder.encode_pcm(&left, &right)?)
            }
            PureRustEncoderBackend::EldMultichannel(encoder) => {
                let channels = deinterleave_channels(
                    input,
                    self.config.channels,
                    self.config.channel_mode,
                    self.config.channel_order,
                );
                Ok(encoder.encode_pcm(&channels)?)
            }
            PureRustEncoderBackend::EldMps(encoder) => {
                let (left, right) = deinterleave_stereo(input);
                Ok(encoder.encode_pcm(&left, &right)?)
            }
        }
    }

    /// Encode one raw access unit and embed as much ancillary input as FDK's
    /// per-frame policy permits. The returned count is the consumed prefix of
    /// `ancillary` and mirrors `AACENC_OutArgs::numAncBytes`.
    pub fn encode_interleaved_f32_with_ancillary(
        &mut self,
        input: &[f32],
        ancillary: &[u8],
    ) -> Result<(Vec<u8>, usize), PureRustEncoderError> {
        let per_channel = self.input_samples_per_channel();
        let expected = per_channel * self.config.channels;
        if input.len() != expected {
            return Err(PureRustEncoderError::InterleavedInputLength {
                expected,
                actual: input.len(),
            });
        }
        let limit = self.max_ancillary_bytes_per_access_unit();
        let consumed = if self.config.ancillary_bitrate == 0 {
            usize::from(ancillary.len() <= limit) * ancillary.len()
        } else {
            ancillary.len().min(limit)
        };
        let ancillary = &ancillary[..consumed];
        let mut frame_metadata = self.metadata.clone();
        let mut submitted_metadata_mode = self.config.metadata_mode;
        if submitted_metadata_mode == 0 {
            if let Some(finalize_mode) = self.metadata_finalize_mode.take() {
                submitted_metadata_mode = finalize_mode;
                frame_metadata = EncoderMetadata::default();
            }
        }
        if self.config.channels != 2 {
            frame_metadata.dolby_surround_mode = 0;
        }
        let current_gains = if self.metadata_frame_gains_override {
            (
                self.metadata_dynamic_range_gain_q16,
                self.metadata_compression_gain_q16,
            )
        } else if matches!(submitted_metadata_mode, 1 | 2) {
            let (dynamic, compression) = self.metadata_compressor.process(input, &frame_metadata);
            (
                dynamic,
                (frame_metadata.compression_profile != MetadataDrcProfile::NotPresent)
                    .then_some(compression),
            )
        } else {
            (
                0,
                (frame_metadata.compression_profile != MetadataDrcProfile::NotPresent).then_some(0),
            )
        };
        let (dynamic_gain, compression_gain) = self.delay_metadata_gains(current_gains);
        let (metadata_mode, frame_metadata) =
            self.delay_metadata_setup((submitted_metadata_mode, frame_metadata));
        let dynamic_range = matches!(metadata_mode, 1 | 2)
            .then(|| frame_metadata.dynamic_range_payload(dynamic_gain));
        let etsi = matches!(metadata_mode, 2 | 3)
            .then(|| frame_metadata.etsi_ancillary_payload(compression_gain));
        let dynamic_range_ref = dynamic_range
            .as_ref()
            .map(|(payload, bits)| (payload.as_slice(), *bits));
        let mut ancillary_elements = Vec::with_capacity(2);
        if let Some(payload) = etsi.as_deref() {
            ancillary_elements.push(payload);
        }
        if !ancillary.is_empty() {
            ancillary_elements.push(ancillary);
        }
        // FDK runs the metadata module's audio-alignment delay even when
        // metadata mode is zero. The delay is part of the encoder timing, not
        // conditional on emitting a metadata payload.
        let delayed_input =
            (!self.metadata_audio_delay.is_empty()).then(|| self.delay_metadata_audio(input));
        let input = delayed_input.as_deref().unwrap_or(input);
        let access_unit = match &mut self.backend {
            PureRustEncoderBackend::LdMono(encoder) => {
                encoder.encode_pcm_with_extensions(input, dynamic_range_ref, &ancillary_elements)?
            }
            PureRustEncoderBackend::EldMono(encoder) => {
                encoder.encode_pcm_with_extensions(input, dynamic_range_ref, &ancillary_elements)?
            }
            PureRustEncoderBackend::LdStereo(encoder) => {
                let (left, right) = deinterleave_stereo(input);
                encoder.encode_pcm_with_extensions(
                    &left,
                    &right,
                    dynamic_range_ref,
                    &ancillary_elements,
                )?
            }
            PureRustEncoderBackend::LdMultichannel(encoder) => {
                let channels = deinterleave_channels(
                    input,
                    self.config.channels,
                    self.config.channel_mode,
                    self.config.channel_order,
                );
                encoder.encode_pcm_with_extensions(
                    &channels,
                    dynamic_range_ref,
                    &ancillary_elements,
                )?
            }
            PureRustEncoderBackend::LdMps(encoder) => {
                let (left, right) = deinterleave_stereo(input);
                encoder.encode_pcm_with_extensions(
                    &left,
                    &right,
                    dynamic_range_ref,
                    &ancillary_elements,
                )?
            }
            PureRustEncoderBackend::EldStereo(encoder) => {
                let (left, right) = deinterleave_stereo(input);
                encoder.encode_pcm_with_extensions(
                    &left,
                    &right,
                    dynamic_range_ref,
                    &ancillary_elements,
                )?
            }
            PureRustEncoderBackend::EldMps(encoder) => {
                let (left, right) = deinterleave_stereo(input);
                encoder.encode_pcm_with_extensions(
                    &left,
                    &right,
                    dynamic_range_ref,
                    &ancillary_elements,
                )?
            }
            _ => {
                let access_unit = self.encode_core_interleaved_f32(input)?;
                append_ga_extensions(&access_unit, dynamic_range_ref, &ancillary_elements)?
            }
        };
        Ok((access_unit, consumed))
    }

    /// Encode PCM and apply the configured FDK transport framing.
    ///
    /// An empty vector means that `AACENC_TPSUBFRAMES` is still collecting
    /// access units. Once the configured count is reached, one complete RAW,
    /// ADTS, ADIF, LATM or LOAS transport unit is returned.
    pub fn encode_transport_f32(&mut self, input: &[f32]) -> Result<Vec<u8>, PureRustEncoderError> {
        let access_unit = self.encode_interleaved_f32(input)?;
        let access_unit = self.add_adts_program_config(access_unit)?;
        self.pending_access_units.push(access_unit);
        if self.pending_access_units.len() < self.config.transport_subframes as usize {
            return Ok(Vec::new());
        }
        let access_units = std::mem::take(&mut self.pending_access_units);
        match self.config.transport_mux {
            0 => Ok(access_units.concat()),
            1 => {
                let mut output = Vec::new();
                if !self.adif_header_written {
                    output.extend_from_slice(
                        self.adif_header
                            .as_deref()
                            .expect("ADIF header is created with the backend"),
                    );
                    self.adif_header_written = true;
                }
                output.extend(access_units.concat());
                Ok(output)
            }
            2 => self.write_adts_transport(&access_units),
            6 | 7 => Ok(self
                .latm_writer
                .as_mut()
                .expect("LATM writer is created with the backend")
                .write_audio_mux_element(&access_units)?),
            10 => {
                let latm = self
                    .latm_writer
                    .as_mut()
                    .expect("LATM writer is created with the backend")
                    .write_audio_mux_element(&access_units)?;
                Ok(write_loas_frame(&latm)?)
            }
            _ => unreachable!("set_parameter validates transport mux values"),
        }
    }

    /// Encode PCM plus ancillary data and apply the configured transport.
    /// Ancillary consumption occurs per access unit even while transport
    /// subframes are being accumulated.
    pub fn encode_transport_f32_with_ancillary(
        &mut self,
        input: &[f32],
        ancillary: &[u8],
    ) -> Result<(Vec<u8>, usize), PureRustEncoderError> {
        let (access_unit, consumed) =
            self.encode_interleaved_f32_with_ancillary(input, ancillary)?;
        let access_unit = self.add_adts_program_config(access_unit)?;
        self.pending_access_units.push(access_unit);
        if self.pending_access_units.len() < self.config.transport_subframes as usize {
            return Ok((Vec::new(), consumed));
        }
        let access_units = std::mem::take(&mut self.pending_access_units);
        let output = match self.config.transport_mux {
            0 => access_units.concat(),
            1 => {
                let mut output = Vec::new();
                if !self.adif_header_written {
                    output.extend_from_slice(
                        self.adif_header
                            .as_deref()
                            .expect("ADIF header is created with the backend"),
                    );
                    self.adif_header_written = true;
                }
                output.extend(access_units.concat());
                output
            }
            2 => self.write_adts_transport(&access_units)?,
            6 | 7 => self
                .latm_writer
                .as_mut()
                .expect("LATM writer is created with the backend")
                .write_audio_mux_element(&access_units)?,
            10 => {
                let latm = self
                    .latm_writer
                    .as_mut()
                    .expect("LATM writer is created with the backend")
                    .write_audio_mux_element(&access_units)?;
                write_loas_frame(&latm)?
            }
            _ => unreachable!("set_parameter validates transport mux values"),
        };
        Ok((output, consumed))
    }

    fn write_adts_transport(
        &self,
        access_units: &[Vec<u8>],
    ) -> Result<Vec<u8>, PureRustEncoderError> {
        let protected = self.config.protection;
        let raw_block_count = access_units.len();
        let protection_overhead = if protected {
            if raw_block_count == 1 {
                2
            } else {
                (raw_block_count - 1) * 2 + 2 + raw_block_count * 2
            }
        } else {
            0
        };
        let payload_len: usize =
            access_units.iter().map(Vec::len).sum::<usize>() + protection_overhead;
        let adts_channel_configuration = match self.config.channel_mode {
            1..=7 => self.config.channel_mode as u8,
            11 | 12 | 14 | 33 | 34 => 0,
            _ => self.config.channels as u8,
        };
        let adts_sampling_frequency =
            if self.config.sbr_active && matches!(self.config.audio_object_type, 5 | 29 | 132) {
                self.config.sample_rate / self.config.sbr_ratio.max(1)
            } else {
                self.config.sample_rate
            };
        let mut header = AdtsHeader::new(
            if matches!(self.config.audio_object_type, 129 | 132) {
                MpegVersion::Mpeg2
            } else {
                MpegVersion::Mpeg4
            },
            1,
            sample_rate_index(adts_sampling_frequency)
                .ok_or(EncoderConfigurationError::MissingSampleRate)?,
            adts_channel_configuration,
            payload_len,
        )?;
        header.number_of_raw_data_blocks_in_frame = (raw_block_count - 1) as u8;
        if !protected {
            let mut output = vec![0; header.frame_length];
            let header_len = header.write(&mut output)?;
            let mut offset = header_len;
            for access_unit in access_units {
                output[offset..offset + access_unit.len()].copy_from_slice(access_unit);
                offset += access_unit.len();
            }
            return Ok(output);
        }

        header.protection_absent = false;
        // AdtsHeader::new accounts for the unprotected seven-byte header. Its
        // payload_len above includes every protection field, so frame_length
        // is already the final transmitted size.
        let mut standard_header = vec![0; 9];
        header.write(&mut standard_header)?;
        let mut output = standard_header[..7].to_vec();

        let block_crcs = access_units
            .iter()
            .map(|access_unit| self.adts_raw_block_crc(access_unit))
            .collect::<Result<Vec<_>, _>>()?;
        if raw_block_count == 1 {
            let crc = adts_crc16_padded_bit_regions(
                std::iter::once((output.as_slice(), 0..56, 56)).chain(
                    block_crcs[0].1.iter().cloned().map(|(range, padded_bits)| {
                        (access_units[0].as_slice(), range, padded_bits)
                    }),
                ),
            )?;
            output.extend_from_slice(&crc.to_be_bytes());
            output.extend_from_slice(&access_units[0]);
            return Ok(output);
        }

        let mut cumulative = 0usize;
        for access_unit in &access_units[..raw_block_count - 1] {
            cumulative += access_unit.len() + 2;
            output.extend_from_slice(&(cumulative as u16).to_be_bytes());
        }
        let header_crc = adts_crc16(&output);
        output.extend_from_slice(&header_crc.to_be_bytes());
        for (access_unit, (crc, _)) in access_units.iter().zip(block_crcs) {
            output.extend_from_slice(access_unit);
            output.extend_from_slice(&crc.to_be_bytes());
        }
        debug_assert_eq!(output.len(), header.frame_length);
        Ok(output)
    }

    fn add_adts_program_config(
        &self,
        access_unit: Vec<u8>,
    ) -> Result<Vec<u8>, PureRustEncoderError> {
        if self.config.transport_mux != 2
            || !matches!(self.config.channel_mode, 11 | 12 | 14 | 33 | 34)
        {
            return Ok(access_unit);
        }
        let mut program_config = build_adif_header(&self.config)?
            .program_configs
            .into_iter()
            .next()
            .expect("ADIF construction always creates one program config");
        if self.config.sbr_active && matches!(self.config.audio_object_type, 5 | 29 | 132) {
            let core_rate = self.config.sample_rate / self.config.sbr_ratio.max(1);
            program_config.sampling_frequency_index =
                sample_rate_index(core_rate).ok_or(EncoderConfigurationError::MissingSampleRate)?;
        }
        let mut writer = BitWriter::new();
        writer.write(ElementId::ProgramConfig.bits() as u32, 3);
        program_config.write_to_writer(&mut writer)?;
        let mut output = writer.finish();
        output.extend_from_slice(&access_unit);
        Ok(output)
    }

    fn adts_raw_block_crc(
        &self,
        access_unit: &[u8],
    ) -> Result<(u16, Vec<(std::ops::Range<usize>, usize)>), PureRustEncoderError> {
        let core_sample_rate = if matches!(self.config.audio_object_type, 5 | 29 | 129 | 132) {
            self.config.sample_rate / self.config.sbr_ratio.max(1)
        } else {
            self.config.sample_rate
        };
        let sampling_frequency_index = sample_rate_index(core_sample_rate)
            .ok_or(EncoderConfigurationError::MissingSampleRate)?;
        let core_channels = if matches!(self.config.audio_object_type, 5 | 29 | 129 | 132) {
            1
        } else {
            self.config.channels as u8
        };
        let mut decoder = AacLcDecoder::new(sampling_frequency_index, core_channels)?;
        decoder.decode_raw_data_block_f32(access_unit)?;
        let regions = decoder.last_adts_crc_regions();
        let crc = adts_crc16_padded_bit_regions(
            regions
                .iter()
                .cloned()
                .map(|(range, padded_bits)| (access_unit, range, padded_bits)),
        )?;
        Ok((crc, regions))
    }
}

fn append_ga_extensions(
    access_unit: &[u8],
    dynamic_range: Option<(&[u8], usize)>,
    ancillary_elements: &[&[u8]],
) -> Result<Vec<u8>, PureRustEncoderError> {
    if dynamic_range.is_none() && ancillary_elements.iter().all(|data| data.is_empty()) {
        return Ok(access_unit.to_vec());
    }
    let Some((last_byte_index, &last_byte)) = access_unit
        .iter()
        .enumerate()
        .rev()
        .find(|(_, byte)| **byte != 0)
    else {
        return Err(PureRustEncoderError::InvalidAccessUnitEndMarker);
    };
    let last_one = last_byte_index * 8 + (7 - last_byte.trailing_zeros() as usize);
    if last_one < 2 {
        return Err(PureRustEncoderError::InvalidAccessUnitEndMarker);
    }
    let end_start = last_one - 2;
    if (end_start..=last_one).any(|bit| ((access_unit[bit / 8] >> (7 - bit % 8)) & 1) == 0) {
        return Err(PureRustEncoderError::InvalidAccessUnitEndMarker);
    }

    let mut writer = BitWriter::new();
    for bit in 0..end_start {
        writer.write(u32::from((access_unit[bit / 8] >> (7 - bit % 8)) & 1), 1);
    }

    if let Some((payload, payload_bits)) = dynamic_range {
        let extension_bits = 4usize.saturating_add(payload_bits);
        let count = extension_bits.div_ceil(8);
        writer.write(ElementId::Fill.bits() as u32, 3);
        if count < 15 {
            writer.write(count as u32, 4);
        } else {
            writer.write(15, 4);
            writer.write((count - 14) as u32, 8);
        }
        writer.write(0x0b, 4); // EXT_DYNAMIC_RANGE
        write_bits_from_slice(&mut writer, payload, payload_bits);
        for _ in extension_bits..count * 8 {
            writer.write_bool(false);
        }
    }

    for ancillary in ancillary_elements {
        for chunk in ancillary.chunks(510).filter(|chunk| !chunk.is_empty()) {
            writer.write(ElementId::DataStream.bits() as u32, 3);
            writer.write(0, 4); // element_instance_tag
            writer.write_bool(false); // data_byte_align_flag
            if chunk.len() >= 255 {
                writer.write(255, 8);
                writer.write((chunk.len() - 255) as u32, 8);
            } else {
                writer.write(chunk.len() as u32, 8);
            }
            for &byte in chunk {
                writer.write(byte as u32, 8);
            }
        }
    }
    writer.write(ElementId::End.bits() as u32, 3);
    writer.byte_align();
    Ok(writer.finish())
}

fn write_bits_from_slice(writer: &mut BitWriter, payload: &[u8], bits: usize) {
    debug_assert!(bits <= payload.len() * 8);
    for bit in 0..bits {
        writer.write(u32::from((payload[bit / 8] >> (7 - bit % 8)) & 1), 1);
    }
}

fn backend_audio_specific_config(
    config: &ResolvedEncoderConfig,
    backend: &PureRustEncoderBackend,
) -> Result<AudioSpecificConfig, PureRustEncoderError> {
    Ok(match backend {
        PureRustEncoderBackend::LcMono(_)
        | PureRustEncoderBackend::LcStereo(_)
        | PureRustEncoderBackend::LcMultichannel(_) => {
            let sampling_frequency_index = sample_rate_index(config.sample_rate)
                .ok_or(EncoderConfigurationError::MissingSampleRate)?;
            let (channel_configuration, program_config) =
                encoder_channel_config(config, sampling_frequency_index)?;
            let mut asc = AudioSpecificConfig::aac_lc(config.sample_rate, channel_configuration)?;
            if let Some(ga) = &mut asc.ga_specific {
                ga.frame_length_flag = config.frame_length == 960;
            }
            asc.program_config = program_config;
            asc
        }
        PureRustEncoderBackend::HeMono(_)
        | PureRustEncoderBackend::HeStereo(_)
        | PureRustEncoderBackend::HeMultichannel(_)
        | PureRustEncoderBackend::HePs(_) => {
            let core_rate = config.sample_rate / 2;
            let core_sampling_frequency_index =
                sample_rate_index(core_rate).ok_or(EncoderConfigurationError::MissingSampleRate)?;
            let (channel_configuration, program_config) = match backend {
                PureRustEncoderBackend::HeMono(_) | PureRustEncoderBackend::HePs(_) => (1, None),
                PureRustEncoderBackend::HeStereo(_) => (2, None),
                PureRustEncoderBackend::HeMultichannel(_) => {
                    encoder_channel_config(config, core_sampling_frequency_index)?
                }
                _ => unreachable!(),
            };
            AudioSpecificConfig {
                audio_object_type: 2,
                sampling_frequency_index: core_sampling_frequency_index,
                sampling_frequency: core_rate,
                channel_configuration,
                extension: Some(AudioSpecificConfigExtension {
                    audio_object_type: if config.audio_object_type == 132 {
                        5
                    } else {
                        config.audio_object_type as u8
                    },
                    sampling_frequency_index: sample_rate_index(config.sample_rate)
                        .ok_or(EncoderConfigurationError::MissingSampleRate)?,
                    sampling_frequency: config.sample_rate,
                    ps_present: config.audio_object_type == 29,
                }),
                ga_specific: Some(GaSpecificConfig {
                    frame_length_flag: config.frame_length == 960,
                    ..GaSpecificConfig::default()
                }),
                eld_specific: None,
                usac_config: None,
                error_protection_config: None,
                program_config,
                bits_read: 0,
            }
        }
        PureRustEncoderBackend::LdMono(encoder) => encoder.audio_specific_config(),
        PureRustEncoderBackend::LdStereo(encoder) => encoder.audio_specific_config(),
        PureRustEncoderBackend::LdMultichannel(encoder) => encoder.audio_specific_config(),
        PureRustEncoderBackend::LdMps(encoder) => encoder.audio_specific_config(),
        PureRustEncoderBackend::EldMono(encoder) => encoder.audio_specific_config(),
        PureRustEncoderBackend::EldStereo(encoder) => encoder.audio_specific_config(),
        PureRustEncoderBackend::EldMultichannel(encoder) => encoder.audio_specific_config(),
        PureRustEncoderBackend::EldMps(encoder) => encoder.audio_specific_config()?,
    })
}

fn encoder_channel_config(
    config: &ResolvedEncoderConfig,
    sampling_frequency_index: u8,
) -> Result<(u8, Option<ProgramConfig>), PureRustEncoderError> {
    let channel_configuration = match config.channel_mode {
        1..=7 | 11 | 12 | 14 => config.channel_mode as u8,
        128 => 1,
        33 | 34 => 0,
        _ => {
            return Err(PureRustEncoderError::UnsupportedConfiguration {
                audio_object_type: config.audio_object_type,
                channel_mode: config.channel_mode,
                frame_length: config.frame_length,
                sbr_active: config.sbr_active,
            })
        }
    };
    let program_config = match config.channel_mode {
        33 => Some(ProgramConfig {
            element_instance_tag: 0,
            profile: 1,
            sampling_frequency_index,
            front: vec![
                ProgramElement {
                    is_cpe: false,
                    tag_select: 0,
                },
                ProgramElement {
                    is_cpe: true,
                    tag_select: 0,
                },
            ],
            side: vec![],
            back: vec![
                ProgramElement {
                    is_cpe: true,
                    tag_select: 1,
                },
                ProgramElement {
                    is_cpe: true,
                    tag_select: 2,
                },
            ],
            lfe: vec![0],
            associated_data: vec![],
            valid_cc: vec![],
            mono_mixdown_element_number: None,
            stereo_mixdown_element_number: None,
            matrix_mixdown: None,
            comment: vec![],
            num_channels: 8,
            num_effective_channels: 7,
        }),
        34 => Some(ProgramConfig {
            element_instance_tag: 0,
            profile: 1,
            sampling_frequency_index,
            front: vec![
                ProgramElement {
                    is_cpe: false,
                    tag_select: 0,
                },
                ProgramElement {
                    is_cpe: true,
                    tag_select: 0,
                },
                ProgramElement {
                    is_cpe: true,
                    tag_select: 1,
                },
            ],
            side: vec![],
            back: vec![ProgramElement {
                is_cpe: true,
                tag_select: 2,
            }],
            lfe: vec![0],
            associated_data: vec![],
            valid_cc: vec![],
            mono_mixdown_element_number: None,
            stereo_mixdown_element_number: None,
            matrix_mixdown: None,
            comment: vec![],
            num_channels: 8,
            num_effective_channels: 7,
        }),
        _ => None,
    };
    Ok((channel_configuration, program_config))
}

fn build_adif_header(config: &ResolvedEncoderConfig) -> Result<AdifHeader, PureRustEncoderError> {
    let sampling_frequency_index = sample_rate_index(config.sample_rate)
        .ok_or(EncoderConfigurationError::MissingSampleRate)?;
    let (front, side, back, lfe) = match config.channels {
        1 => (
            vec![ProgramElement {
                is_cpe: false,
                tag_select: 0,
            }],
            vec![],
            vec![],
            vec![],
        ),
        2 => (
            vec![ProgramElement {
                is_cpe: true,
                tag_select: 0,
            }],
            vec![],
            vec![],
            vec![],
        ),
        3 => (
            vec![
                ProgramElement {
                    is_cpe: false,
                    tag_select: 0,
                },
                ProgramElement {
                    is_cpe: true,
                    tag_select: 0,
                },
            ],
            vec![],
            vec![],
            vec![],
        ),
        4 => (
            vec![
                ProgramElement {
                    is_cpe: false,
                    tag_select: 0,
                },
                ProgramElement {
                    is_cpe: true,
                    tag_select: 0,
                },
            ],
            vec![],
            vec![ProgramElement {
                is_cpe: false,
                tag_select: 1,
            }],
            vec![],
        ),
        5 | 6 => (
            vec![
                ProgramElement {
                    is_cpe: false,
                    tag_select: 0,
                },
                ProgramElement {
                    is_cpe: true,
                    tag_select: 0,
                },
            ],
            vec![],
            vec![ProgramElement {
                is_cpe: true,
                tag_select: 1,
            }],
            if config.channels == 6 {
                vec![0]
            } else {
                vec![]
            },
        ),
        7 | 8 => match config.channel_mode {
            7 => (
                vec![
                    ProgramElement {
                        is_cpe: false,
                        tag_select: 0,
                    },
                    ProgramElement {
                        is_cpe: true,
                        tag_select: 0,
                    },
                    ProgramElement {
                        is_cpe: true,
                        tag_select: 1,
                    },
                ],
                vec![],
                vec![ProgramElement {
                    is_cpe: true,
                    tag_select: 2,
                }],
                vec![0],
            ),
            11 => (
                vec![
                    ProgramElement {
                        is_cpe: false,
                        tag_select: 0,
                    },
                    ProgramElement {
                        is_cpe: true,
                        tag_select: 0,
                    },
                ],
                vec![],
                vec![
                    ProgramElement {
                        is_cpe: true,
                        tag_select: 1,
                    },
                    ProgramElement {
                        is_cpe: false,
                        tag_select: 1,
                    },
                ],
                vec![0],
            ),
            12 | 33 => (
                vec![
                    ProgramElement {
                        is_cpe: false,
                        tag_select: 0,
                    },
                    ProgramElement {
                        is_cpe: true,
                        tag_select: 0,
                    },
                ],
                vec![],
                vec![
                    ProgramElement {
                        is_cpe: true,
                        tag_select: 1,
                    },
                    ProgramElement {
                        is_cpe: true,
                        tag_select: 2,
                    },
                ],
                vec![0],
            ),
            14 => (
                vec![
                    ProgramElement {
                        is_cpe: false,
                        tag_select: 0,
                    },
                    ProgramElement {
                        is_cpe: true,
                        tag_select: 0,
                    },
                    ProgramElement {
                        is_cpe: true,
                        tag_select: 2,
                    },
                ],
                vec![],
                vec![ProgramElement {
                    is_cpe: true,
                    tag_select: 1,
                }],
                vec![0],
            ),
            34 => (
                vec![
                    ProgramElement {
                        is_cpe: false,
                        tag_select: 0,
                    },
                    ProgramElement {
                        is_cpe: true,
                        tag_select: 0,
                    },
                    ProgramElement {
                        is_cpe: true,
                        tag_select: 1,
                    },
                ],
                vec![],
                vec![ProgramElement {
                    is_cpe: true,
                    tag_select: 2,
                }],
                vec![0],
            ),
            _ => {
                return Err(PureRustEncoderError::UnsupportedConfiguration {
                    audio_object_type: config.audio_object_type,
                    channel_mode: config.channel_mode,
                    frame_length: config.frame_length,
                    sbr_active: config.sbr_active,
                })
            }
        },
        _ => {
            return Err(PureRustEncoderError::UnsupportedConfiguration {
                audio_object_type: config.audio_object_type,
                channel_mode: config.channel_mode,
                frame_length: config.frame_length,
                sbr_active: config.sbr_active,
            });
        }
    };
    Ok(AdifHeader {
        copyright_id: None,
        original_copy: false,
        home: false,
        variable_bit_rate: config.bitrate_mode != 0,
        bitrate: config.bitrate.min((1 << 23) - 1),
        program_configs: vec![ProgramConfig {
            element_instance_tag: 0,
            profile: 1,
            sampling_frequency_index,
            front,
            side,
            back,
            lfe,
            num_channels: config.channels as u8,
            num_effective_channels: config.effective_channels as u8,
            ..ProgramConfig::default()
        }],
        bits_read: 0,
    })
}

fn deinterleave_stereo(input: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut left = Vec::with_capacity(input.len() / 2);
    let mut right = Vec::with_capacity(input.len() / 2);
    for pair in input.chunks_exact(2) {
        left.push(pair[0]);
        right.push(pair[1]);
    }
    (left, right)
}

fn downmix_stereo_to_mono(input: &[f32]) -> Vec<f32> {
    input
        .chunks_exact(2)
        .map(|sample| (sample[0] + sample[1]) * 0.5)
        .collect()
}

fn deinterleave_channels(
    input: &[f32],
    channels: usize,
    channel_mode: u32,
    channel_order: u32,
) -> Vec<Vec<f32>> {
    let map = encoder_channel_input_map(channel_mode, channels, channel_order);
    let mut planar = vec![Vec::with_capacity(input.len() / channels); channels];
    for frame in input.chunks_exact(channels) {
        for (mpeg_index, output) in planar.iter_mut().enumerate() {
            output.push(frame[map.get(mpeg_index).copied().unwrap_or(mpeg_index)]);
        }
    }
    planar
}

fn encoder_channel_input_map(
    channel_mode: u32,
    channels: usize,
    channel_order: u32,
) -> &'static [usize] {
    match (channel_order, channel_mode, channels) {
        (0, _, _) | (_, _, 1 | 2) => &[],
        (1, 3, 3) => &[2, 0, 1],
        (1, 4, 4) => &[2, 0, 1, 3],
        (1, 5, 5) => &[2, 0, 1, 3, 4],
        (1, 6, 6) => &[2, 0, 1, 4, 5, 3],
        (1, 7, 8) => &[2, 6, 7, 0, 1, 4, 5, 3],
        (1, 11, 7) => &[2, 0, 1, 4, 5, 6, 3],
        (1, 12 | 33, 8) => &[2, 0, 1, 6, 7, 4, 5, 3],
        (1, 14, 8) => &[2, 0, 1, 4, 5, 3, 6, 7],
        (1, 34, 8) => &[2, 6, 7, 0, 1, 4, 5, 3],
        // libAACenc exposes CH_ORDER_WG4 but initializes its channel-map
        // descriptor solely from `co == CH_ORDER_MPEG`; values 1 and 2 both
        // select the default WAV table in the executable implementation.
        (2, 3, 3) => &[2, 0, 1],
        (2, 4, 4) => &[2, 0, 1, 3],
        (2, 5, 5) => &[2, 0, 1, 3, 4],
        (2, 6, 6) => &[2, 0, 1, 4, 5, 3],
        (2, 7, 8) => &[2, 6, 7, 0, 1, 4, 5, 3],
        (2, 11, 7) => &[2, 0, 1, 4, 5, 6, 3],
        (2, 12 | 33, 8) => &[2, 0, 1, 6, 7, 4, 5, 3],
        (2, 14, 8) => &[2, 0, 1, 4, 5, 3, 6, 7],
        (2, 34, 8) => &[2, 6, 7, 0, 1, 4, 5, 3],
        _ => &[],
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SbrTuning {
    bitrate_from: u32,
    bitrate_to: u32,
    core_sample_rate: u32,
    start_frequency: u8,
    stop_frequency: u8,
    noise_bands: u8,
    frequency_scale: u8,
}

macro_rules! sbr_tuning {
    ($from:literal, $to:literal, $rate:literal, $start:literal, $stop:literal, $noise:literal, $scale:literal) => {
        SbrTuning {
            bitrate_from: $from,
            bitrate_to: $to,
            core_sample_rate: $rate,
            start_frequency: $start,
            stop_frequency: $stop,
            noise_bands: $noise,
            frequency_scale: $scale,
        }
    };
}

// Mono/core-mono HE-AAC entries from libSBRenc's sbrTuningTable.  Parametric
// stereo also uses this part of the table because its core coder has one
// channel.  Speech alternatives and psychoacoustic-only fields are retained
// by their respective encoder models and do not affect the SBR header here.
const HE_AAC_MONO_SBR_TUNING: &[SbrTuning] = &[
    sbr_tuning!(8000, 10000, 8000, 7, 11, 1, 3),
    sbr_tuning!(10000, 12000, 8000, 11, 13, 1, 3),
    sbr_tuning!(12000, 16001, 8000, 14, 13, 1, 3),
    sbr_tuning!(16000, 24000, 8000, 14, 14, 2, 2),
    sbr_tuning!(24000, 32000, 8000, 14, 14, 2, 2),
    sbr_tuning!(32000, 48001, 8000, 14, 15, 2, 2),
    sbr_tuning!(8000, 10000, 11025, 5, 6, 1, 3),
    sbr_tuning!(10000, 12000, 11025, 8, 12, 1, 3),
    sbr_tuning!(12000, 16000, 11025, 12, 13, 1, 3),
    sbr_tuning!(16000, 20000, 11025, 12, 13, 1, 3),
    sbr_tuning!(20000, 24001, 11025, 13, 13, 1, 3),
    sbr_tuning!(24000, 32000, 11025, 14, 14, 2, 2),
    sbr_tuning!(32000, 48000, 11025, 15, 15, 2, 2),
    sbr_tuning!(48000, 64001, 11025, 15, 15, 2, 1),
    sbr_tuning!(8000, 10000, 12000, 4, 6, 1, 3),
    sbr_tuning!(10000, 12000, 12000, 7, 11, 1, 3),
    sbr_tuning!(12000, 16000, 12000, 11, 12, 1, 3),
    sbr_tuning!(16000, 20000, 12000, 11, 12, 1, 3),
    sbr_tuning!(20000, 24001, 12000, 12, 12, 1, 3),
    sbr_tuning!(24000, 32000, 12000, 13, 13, 2, 2),
    sbr_tuning!(32000, 48000, 12000, 14, 14, 2, 2),
    sbr_tuning!(48000, 64001, 12000, 14, 15, 2, 1),
    sbr_tuning!(8000, 10000, 16000, 1, 0, 1, 3),
    sbr_tuning!(10000, 12000, 16000, 2, 6, 1, 3),
    sbr_tuning!(12000, 16000, 16000, 4, 6, 1, 3),
    sbr_tuning!(16000, 18000, 16000, 4, 8, 1, 3),
    sbr_tuning!(18000, 22000, 16000, 6, 11, 2, 2),
    sbr_tuning!(22000, 28000, 16000, 10, 12, 2, 2),
    sbr_tuning!(28000, 36000, 16000, 12, 13, 2, 2),
    sbr_tuning!(36000, 44000, 16000, 14, 13, 2, 1),
    sbr_tuning!(44000, 64001, 16000, 14, 13, 2, 1),
    sbr_tuning!(11369, 16000, 22050, 3, 4, 1, 3),
    sbr_tuning!(16000, 18000, 22050, 3, 5, 1, 3),
    sbr_tuning!(18000, 22000, 22050, 4, 8, 2, 2),
    sbr_tuning!(22000, 28000, 22050, 7, 8, 2, 2),
    sbr_tuning!(28000, 36000, 22050, 10, 9, 2, 2),
    sbr_tuning!(36000, 44000, 22050, 11, 10, 2, 1),
    sbr_tuning!(44000, 64001, 22050, 13, 12, 2, 1),
    sbr_tuning!(12000, 16000, 24000, 3, 4, 1, 3),
    sbr_tuning!(16000, 18000, 24000, 3, 5, 1, 3),
    sbr_tuning!(18000, 22000, 24000, 4, 8, 2, 2),
    sbr_tuning!(22000, 28000, 24000, 7, 8, 2, 2),
    sbr_tuning!(28000, 36000, 24000, 10, 9, 2, 2),
    sbr_tuning!(36000, 44000, 24000, 11, 10, 2, 1),
    sbr_tuning!(44000, 64001, 24000, 13, 11, 2, 1),
    sbr_tuning!(24000, 36000, 32000, 4, 4, 2, 3),
    sbr_tuning!(36000, 60000, 32000, 7, 6, 2, 2),
    sbr_tuning!(60000, 72000, 32000, 9, 8, 2, 1),
    sbr_tuning!(72000, 100000, 32000, 11, 10, 2, 1),
    sbr_tuning!(100000, 160001, 32000, 13, 11, 2, 1),
    sbr_tuning!(24000, 36000, 44100, 4, 4, 2, 3),
    sbr_tuning!(36000, 60000, 44100, 7, 6, 2, 2),
    sbr_tuning!(60000, 72000, 44100, 9, 8, 2, 1),
    sbr_tuning!(72000, 100000, 44100, 11, 10, 2, 1),
    sbr_tuning!(100000, 160001, 44100, 13, 11, 2, 1),
    sbr_tuning!(32000, 36000, 48000, 4, 9, 2, 3),
    sbr_tuning!(36000, 60000, 48000, 7, 10, 2, 2),
    sbr_tuning!(60000, 72000, 48000, 9, 10, 2, 1),
    sbr_tuning!(72000, 100000, 48000, 11, 11, 2, 1),
    sbr_tuning!(100000, 160001, 48000, 13, 11, 2, 1),
];

fn select_he_aac_mono_sbr_tuning(core_sample_rate: u32, bitrate: u32) -> Option<SbrTuning> {
    let mut closest_below = None;
    for &entry in HE_AAC_MONO_SBR_TUNING
        .iter()
        .filter(|entry| entry.core_sample_rate == core_sample_rate)
    {
        if (entry.bitrate_from..entry.bitrate_to).contains(&bitrate) {
            return Some(entry);
        }
        if entry.bitrate_to <= bitrate
            && closest_below.is_none_or(|old: SbrTuning| entry.bitrate_to > old.bitrate_to)
        {
            closest_below = Some(entry);
        }
    }
    closest_below
}

fn he_aac_sbr_header(core_sample_rate: u32, bitrate: u32) -> Option<LdSbrHeader> {
    let tuning = select_he_aac_mono_sbr_tuning(core_sample_rate, bitrate)?;
    let non_default_extra_1 = tuning.frequency_scale != 2 || tuning.noise_bands != 2;
    Some(LdSbrHeader {
        start_frequency: tuning.start_frequency,
        stop_frequency: tuning.stop_frequency,
        frequency_scale: non_default_extra_1.then_some(tuning.frequency_scale),
        alter_scale: non_default_extra_1.then_some(true),
        noise_bands: non_default_extra_1.then_some(tuning.noise_bands),
        ..LdSbrHeader::default()
    })
}

#[derive(Clone, Copy)]
struct EldMonoSbrTuning {
    bitrate_from: u32,
    bitrate_to: u32,
    core_sample_rate: u32,
    start_frequency: u8,
    stop_frequency: u8,
    noise_bands: u8,
    frequency_scale: u8,
}

macro_rules! eld_mono_tuning {
    ($from:literal, $to:literal, $rate:literal, $start:literal, $stop:literal, $noise:literal, $scale:literal) => {
        EldMonoSbrTuning {
            bitrate_from: $from,
            bitrate_to: $to,
            core_sample_rate: $rate,
            start_frequency: $start,
            stop_frequency: $stop,
            noise_bands: $noise,
            frequency_scale: $scale,
        }
    };
}

// CODEC_AACLD mono portion of libSBRenc's sbrTuningTable.
const ELD_MONO_SBR_TUNING: &[EldMonoSbrTuning] = &[
    eld_mono_tuning!(8000, 32000, 12000, 1, 0, 1, 3),
    eld_mono_tuning!(16000, 18000, 16000, 4, 9, 1, 3),
    eld_mono_tuning!(18000, 22000, 16000, 7, 12, 1, 3),
    eld_mono_tuning!(22000, 28000, 16000, 6, 9, 2, 3),
    eld_mono_tuning!(28000, 36000, 16000, 8, 12, 2, 3),
    eld_mono_tuning!(36000, 44000, 16000, 10, 12, 2, 1),
    eld_mono_tuning!(44000, 64001, 16000, 11, 13, 2, 1),
    eld_mono_tuning!(18000, 22000, 22050, 4, 5, 2, 3),
    eld_mono_tuning!(22000, 28000, 22050, 5, 6, 2, 2),
    eld_mono_tuning!(28000, 36000, 22050, 7, 8, 2, 2),
    eld_mono_tuning!(36000, 44000, 22050, 9, 9, 2, 1),
    eld_mono_tuning!(44000, 52000, 22050, 12, 11, 2, 1),
    eld_mono_tuning!(52000, 64001, 22050, 13, 11, 2, 1),
    eld_mono_tuning!(20000, 22000, 24000, 3, 8, 2, 2),
    eld_mono_tuning!(22000, 28000, 24000, 3, 8, 2, 2),
    eld_mono_tuning!(28000, 36000, 24000, 4, 8, 2, 2),
    eld_mono_tuning!(36000, 56000, 24000, 8, 9, 2, 1),
    eld_mono_tuning!(56000, 64001, 24000, 13, 11, 2, 1),
    eld_mono_tuning!(24000, 36000, 32000, 4, 4, 2, 3),
    eld_mono_tuning!(36000, 60000, 32000, 7, 6, 2, 2),
    eld_mono_tuning!(60000, 72000, 32000, 9, 8, 2, 1),
    eld_mono_tuning!(72000, 100000, 32000, 11, 10, 2, 1),
    eld_mono_tuning!(100000, 160001, 32000, 13, 11, 2, 1),
    eld_mono_tuning!(36000, 60000, 44100, 8, 6, 2, 2),
    eld_mono_tuning!(60000, 72000, 44100, 9, 10, 2, 1),
    eld_mono_tuning!(72000, 100000, 44100, 11, 11, 2, 1),
    eld_mono_tuning!(100000, 160001, 44100, 13, 11, 2, 1),
    eld_mono_tuning!(36000, 60000, 48000, 4, 4, 2, 3),
    eld_mono_tuning!(60000, 72000, 48000, 9, 10, 2, 1),
    eld_mono_tuning!(72000, 100000, 48000, 11, 11, 2, 1),
    eld_mono_tuning!(100000, 160001, 48000, 13, 11, 2, 1),
];

// CODEC_AACLD stereo portion of the same table. Only the fields transmitted
// by the ELD SBR header are retained here.
const ELD_STEREO_SBR_TUNING: &[EldMonoSbrTuning] = &[
    eld_mono_tuning!(32000, 36000, 16000, 10, 12, 2, 2),
    eld_mono_tuning!(36000, 44000, 16000, 13, 13, 2, 2),
    eld_mono_tuning!(44000, 52000, 16000, 10, 11, 2, 2),
    eld_mono_tuning!(52000, 60000, 16000, 14, 13, 3, 1),
    eld_mono_tuning!(60000, 76000, 16000, 14, 13, 3, 1),
    eld_mono_tuning!(76000, 128001, 16000, 14, 13, 3, 1),
    eld_mono_tuning!(32000, 36000, 22050, 5, 7, 2, 2),
    eld_mono_tuning!(36000, 44000, 22050, 5, 8, 2, 2),
    eld_mono_tuning!(44000, 52000, 22050, 7, 8, 3, 2),
    eld_mono_tuning!(52000, 60000, 22050, 9, 9, 3, 1),
    eld_mono_tuning!(60000, 76000, 22050, 10, 10, 3, 1),
    eld_mono_tuning!(76000, 82000, 22050, 12, 11, 3, 1),
    eld_mono_tuning!(82000, 128001, 22050, 13, 11, 3, 1),
    eld_mono_tuning!(32000, 36000, 24000, 5, 7, 2, 2),
    eld_mono_tuning!(36000, 44000, 24000, 4, 8, 2, 2),
    eld_mono_tuning!(44000, 52000, 24000, 6, 8, 3, 2),
    eld_mono_tuning!(52000, 60000, 24000, 9, 9, 3, 1),
    eld_mono_tuning!(60000, 76000, 24000, 11, 10, 3, 1),
    eld_mono_tuning!(76000, 88000, 24000, 12, 11, 3, 1),
    eld_mono_tuning!(88000, 128001, 24000, 13, 11, 3, 1),
    eld_mono_tuning!(60000, 80000, 32000, 7, 6, 3, 2),
    eld_mono_tuning!(80000, 112000, 32000, 9, 8, 3, 1),
    eld_mono_tuning!(112000, 144000, 32000, 11, 10, 3, 1),
    eld_mono_tuning!(144000, 256001, 32000, 13, 11, 3, 1),
    eld_mono_tuning!(60000, 80000, 44100, 7, 6, 3, 2),
    eld_mono_tuning!(80000, 112000, 44100, 10, 8, 3, 1),
    eld_mono_tuning!(112000, 144000, 44100, 12, 10, 3, 1),
    eld_mono_tuning!(144000, 256001, 44100, 13, 11, 3, 1),
    eld_mono_tuning!(60000, 80000, 48000, 7, 10, 2, 2),
    eld_mono_tuning!(80000, 112000, 48000, 9, 10, 3, 1),
    eld_mono_tuning!(112000, 144000, 48000, 11, 11, 3, 1),
    eld_mono_tuning!(144000, 176000, 48000, 12, 11, 3, 1),
    eld_mono_tuning!(176000, 256001, 48000, 13, 11, 3, 1),
];

fn eld_mono_sbr_header(
    core_sample_rate: u32,
    bitrate: u32,
    dual_rate: bool,
) -> Option<LdSbrHeader> {
    // These ROM entries cannot be instantiated by the bundled executable C
    // encoder: frequency-table initialization fails for 12 kHz single-rate,
    // and for the 72--100 kbit/s 44.1/88.2 kHz dual-rate interval.
    if (!dual_rate && core_sample_rate == 12_000)
        || (dual_rate && core_sample_rate == 44_100 && (72_000..100_000).contains(&bitrate))
    {
        return None;
    }
    let tuning = ELD_MONO_SBR_TUNING
        .iter()
        .filter(|entry| entry.core_sample_rate == core_sample_rate)
        .find(|entry| (entry.bitrate_from..entry.bitrate_to).contains(&bitrate))
        .or_else(|| {
            ELD_MONO_SBR_TUNING
                .iter()
                .filter(|entry| {
                    entry.core_sample_rate == core_sample_rate && entry.bitrate_to <= bitrate
                })
                .max_by_key(|entry| entry.bitrate_to)
        })
        .or_else(|| {
            // FDK clamps rates below the first tuning interval to that
            // interval instead of rejecting the configuration.
            ELD_MONO_SBR_TUNING
                .iter()
                .filter(|entry| entry.core_sample_rate == core_sample_rate)
                .min_by_key(|entry| entry.bitrate_from)
        })?;
    let non_default_extra_1 = tuning.frequency_scale != 2 || tuning.noise_bands != 2;
    let mut header = LdSbrHeader {
        amp_resolution: true,
        start_frequency: tuning.start_frequency,
        stop_frequency: tuning.stop_frequency,
        frequency_scale: non_default_extra_1.then_some(tuning.frequency_scale),
        alter_scale: non_default_extra_1.then_some(true),
        noise_bands: non_default_extra_1.then_some(tuning.noise_bands),
        ..LdSbrHeader::default()
    };
    if !dual_rate {
        while header.stop_frequency > 0 {
            let fits = LdSbrFrequencyTables::from_header(&header, core_sample_rate * 2)
                .ok()
                .and_then(|tables| tables.high.last().copied())
                .is_some_and(|band| band <= 32);
            if fits {
                break;
            }
            header.stop_frequency -= 1;
        }
    }
    LdSbrFrequencyTables::from_header(&header, core_sample_rate * 2)
        .ok()
        .filter(|tables| dual_rate || tables.high.last().is_some_and(|&band| band <= 32))
        .map(|_| header)
}

// `ana_max_level` from the CODEC_AACLD mono rows of sbrTuningTable.  The
// selected row follows the same exact-range/closest-lower rule as the header.
fn eld_mono_noise_max_level(core_sample_rate: u32, bitrate: u32) -> i8 {
    match core_sample_rate {
        12_000 => 6,
        16_000 if (18_000..22_000).contains(&bitrate) => 9,
        16_000 if (28_000..36_000).contains(&bitrate) => 12,
        16_000 if bitrate < 36_000 => 6,
        22_050 if bitrate < 28_000 => 6,
        24_000 if bitrate < 22_000 => 6,
        _ => 3,
    }
}

fn eld_stereo_sbr_header(
    core_sample_rate: u32,
    bitrate: u32,
    dual_rate: bool,
) -> Option<LdSbrHeader> {
    let tuning = ELD_STEREO_SBR_TUNING
        .iter()
        .filter(|entry| entry.core_sample_rate == core_sample_rate)
        .find(|entry| (entry.bitrate_from..entry.bitrate_to).contains(&bitrate))
        .or_else(|| {
            ELD_STEREO_SBR_TUNING
                .iter()
                .filter(|entry| {
                    entry.core_sample_rate == core_sample_rate && entry.bitrate_to <= bitrate
                })
                .max_by_key(|entry| entry.bitrate_to)
        })
        .or_else(|| {
            ELD_STEREO_SBR_TUNING
                .iter()
                .filter(|entry| entry.core_sample_rate == core_sample_rate)
                .min_by_key(|entry| entry.bitrate_from)
        })?;
    let extra = tuning.frequency_scale != 2 || tuning.noise_bands != 2;
    let mut header = LdSbrHeader {
        amp_resolution: true,
        start_frequency: tuning.start_frequency,
        stop_frequency: tuning.stop_frequency,
        frequency_scale: extra.then_some(tuning.frequency_scale),
        alter_scale: extra.then_some(true),
        noise_bands: extra.then_some(tuning.noise_bands),
        ..LdSbrHeader::default()
    };
    if !dual_rate {
        while header.stop_frequency > 0
            && !LdSbrFrequencyTables::from_header(&header, core_sample_rate * 2)
                .ok()
                .and_then(|tables| tables.high.last().copied())
                .is_some_and(|band| band <= 32)
        {
            header.stop_frequency -= 1;
        }
    }
    LdSbrFrequencyTables::from_header(&header, core_sample_rate * 2)
        .ok()
        .filter(|tables| dual_rate || tables.high.last().is_some_and(|&band| band <= 32))
        .map(|_| header)
}

fn eld_multichannel_sbr_headers(
    core_sample_rate: u32,
    bitrate: u32,
    channels: usize,
    channel_mode: u32,
    dual_rate: bool,
) -> Option<(Vec<LdSbrHeader>, Vec<u32>, Vec<i8>)> {
    // Q1.31 values from channel_map.cpp. The LFE share remains reserved but
    // is not passed to SBR, matching aacenc_lib.cpp's SCE/CPE-only loop.
    // FL2FXCONST_DBL receives `float` literals in channel_map.cpp. Preserve
    // their binary32 rounding before converting to Q1.31; using the exact
    // decimal fractions changes element rates by one bit/s at ROM boundaries.
    const Q31_024: u32 = 515_396_064;
    const Q31_026: u32 = 558_345_728;
    const Q31_030: u32 = 644_245_120;
    const Q31_035: u32 = 751_619_264;
    const Q31_037: u32 = 794_568_960;
    const Q31_040: u32 = 858_993_472;
    const Q31_060: u32 = 1_288_490_240;
    const Q31_006: u32 = 128_849_016;
    const Q31_018: u32 = 386_547_072;
    const Q31_020: u32 = 429_496_736;
    const Q31_0275: u32 = 590_558_016;
    const Q31_004: u32 = 85_899_344;
    const Q31_005: u32 = 107_374_184;

    let elements: &[(Option<bool>, u32)] = match (channel_mode, channels) {
        (3, 3) => &[(Some(false), Q31_040), (Some(true), Q31_060)],
        (4, 4) => &[
            (Some(false), Q31_030),
            (Some(true), Q31_040),
            (Some(false), Q31_030),
        ],
        (5, 5) => &[
            (Some(false), Q31_026),
            (Some(true), Q31_037),
            (Some(true), Q31_037),
        ],
        (6, 6) => &[
            (Some(false), Q31_024),
            (Some(true), Q31_035),
            (Some(true), Q31_035),
            (None, Q31_006),
        ],
        (11, 7) => &[
            (Some(false), Q31_020),
            (Some(true), Q31_0275),
            (Some(true), Q31_0275),
            (Some(false), Q31_020),
            (None, Q31_005),
        ],
        (7 | 12, 8) => &[
            (Some(false), Q31_018),
            (Some(true), Q31_026),
            (Some(true), Q31_026),
            (Some(true), Q31_026),
            (None, Q31_004),
        ],
        (14, 8) => &[
            (Some(false), Q31_018),
            (Some(true), Q31_026),
            (Some(true), Q31_026),
            (None, Q31_004),
            (Some(true), Q31_026),
        ],
        _ => return None,
    };
    let multiply_floor =
        |relative: u32, total: u32| ((u64::from(relative) * u64::from(total)) >> 31) as u32;
    let minimum_bitrate = |stereo: bool| {
        let table = if stereo {
            ELD_STEREO_SBR_TUNING
        } else {
            ELD_MONO_SBR_TUNING
        };
        table
            .iter()
            .filter(|entry| entry.core_sample_rate == core_sample_rate)
            .map(|entry| entry.bitrate_from)
            .min()
    };

    let mut limited_bitrate = bitrate;
    for _ in 0..=elements.len() {
        let mut distributed = elements
            .iter()
            .map(|&(_, relative)| multiply_floor(relative, limited_bitrate))
            .collect::<Vec<_>>();
        let remainder = limited_bitrate.saturating_sub(distributed.iter().sum());
        distributed[0] = distributed[0].saturating_add(remainder);
        let mut next = limited_bitrate;
        for ((kind, relative), element_bitrate) in elements.iter().zip(distributed.iter().copied())
        {
            let Some(stereo) = kind else { continue };
            let minimum = minimum_bitrate(*stereo)?;
            if element_bitrate < minimum {
                let numerator = u64::from(minimum + 8) << 31;
                let required = numerator.div_ceil(u64::from(*relative)) as u32;
                next = next.max(required);
                break;
            }
        }
        if next == limited_bitrate {
            break;
        }
        limited_bitrate = next;
    }
    let mut distributed = elements
        .iter()
        .map(|&(_, relative)| multiply_floor(relative, limited_bitrate))
        .collect::<Vec<_>>();
    let remainder = limited_bitrate.saturating_sub(distributed.iter().sum());
    distributed[0] = distributed[0].saturating_add(remainder);

    let element_bitrates = elements
        .iter()
        .zip(distributed.iter().copied())
        .filter_map(|(&(kind, _), bitrate)| kind.map(|_| bitrate))
        .collect::<Vec<_>>();
    let noise_max_levels = elements
        .iter()
        .zip(distributed.iter().copied())
        .filter_map(|(&(kind, _), bitrate)| {
            kind.map(|stereo| {
                if stereo {
                    -3
                } else {
                    eld_mono_noise_max_level(core_sample_rate, bitrate)
                }
            })
        })
        .collect::<Vec<_>>();
    let mut headers = elements
        .iter()
        .zip(distributed.iter().copied())
        .filter_map(|(&(kind, _), element_bitrate)| {
            kind.map(|stereo| {
                if stereo {
                    eld_stereo_sbr_header(core_sample_rate, element_bitrate, dual_rate)
                } else {
                    eld_mono_sbr_header(core_sample_rate, element_bitrate, dual_rate)
                }
            })
        })
        .collect::<Option<Vec<_>>>()?;
    let common_start = headers.iter().map(|header| header.start_frequency).max()?;
    let common_stop = headers.iter().map(|header| header.stop_frequency).max()?;
    for header in &mut headers {
        header.start_frequency = common_start;
        header.stop_frequency = common_stop;
    }
    Some((headers, element_bitrates, noise_max_levels))
}

fn he_aac_crossover_bandwidth(output_sample_rate: u32, bitrate: u32) -> Option<(u32, LdSbrHeader)> {
    let header = he_aac_sbr_header(output_sample_rate / 2, bitrate)?;
    let tables = LdSbrFrequencyTables::from_header(&header, output_sample_rate).ok()?;
    let start_band = u32::from(*tables.high.first()?);
    // updateFreqBandTable: ((band * sampleFreq / noQmfBands) + 1) >> 1,
    // with 64 QMF bands for ordinary dual-rate HE-AAC.
    let bandwidth = (start_band * output_sample_rate / 64 + 1) >> 1;
    Some((bandwidth, header))
}

fn require(
    parameter: EncoderParameter,
    value: u32,
    valid: bool,
) -> Result<(), EncoderParameterError> {
    if valid {
        Ok(())
    } else {
        Err(EncoderParameterError::InvalidValue { parameter, value })
    }
}

pub fn channel_count(mode: u32) -> Option<usize> {
    Some(match mode {
        1 => 1,
        2 => 2,
        3 => 3,
        4 => 4,
        5 => 5,
        6 => 6,
        7 => 8,
        11 => 7,
        12 | 14 | 33 | 34 => 8,
        128 => 2,
        _ => return None,
    })
}

pub fn effective_channel_count(mode: u32) -> Option<usize> {
    Some(match mode {
        1 => 1,
        2 => 2,
        3 => 3,
        4 => 4,
        5 => 5,
        6 => 5,
        7 => 7,
        11 => 6,
        12 | 14 | 33 | 34 => 7,
        128 => 1,
        _ => return None,
    })
}

fn vbr_bitrate(mode: u32, effective_channels: usize, stereo: bool) -> Option<u32> {
    let per_channel = match (mode, stereo) {
        (1, false) => 32_000,
        (2, false) => 40_000,
        (3, false) => 56_000,
        (4, false) => 72_000,
        (5, false) => 112_000,
        (1, true) => 20_000,
        (2, true) => 32_000,
        (3, true) => 48_000,
        (4, true) => 64_000,
        (5, true) => 96_000,
        _ => return None,
    };
    Some(per_channel * effective_channels as u32)
}

fn adjusted_vbr_mode(
    requested_mode: u32,
    peak_bitrate: u32,
    effective_channels: usize,
    stereo: bool,
) -> Option<u32> {
    let requested = vbr_bitrate(requested_mode, effective_channels, stereo)?;
    (1..=5)
        .rev()
        .find(|&candidate| {
            let candidate_rate = vbr_bitrate(candidate, effective_channels, stereo).unwrap_or(0);
            peak_bitrate >= candidate_rate && candidate_rate >= requested.min(candidate_rate)
        })
        .map(|candidate| {
            if vbr_bitrate(candidate, effective_channels, stereo).unwrap_or(0) < requested {
                candidate
            } else {
                requested_mode
            }
        })
}

fn default_cbr_bitrate(
    effective_channels: usize,
    sample_rate: u32,
    ps_active: bool,
    sbr_active: bool,
    requested_sbr_ratio: u32,
    eld: bool,
) -> u32 {
    let base = effective_channels as u32 * sample_rate;
    if ps_active {
        // PS maps a stereo input to a mono AAC core before the default CBR
        // calculation (`GetCoreChannelMode` in libAACenc).
        sample_rate
    } else if sbr_active && (requested_sbr_ratio == 2 || (requested_sbr_ratio == 0 && !eld)) {
        (base + base / 4) / 2
    } else if sbr_active {
        base + base / 8
    } else {
        base + base / 2
    }
}

fn eld_auto_sbr_mode(channels: usize, sample_rate: u32, bitrate: u32) -> Option<u32> {
    const MONO: &[(u32, u32, u32)] = &[
        (48_000, 0, 2),
        (48_000, 64_000, 0),
        (44_100, 0, 2),
        (44_100, 64_000, 0),
        (32_000, 0, 2),
        (32_000, 28_000, 1),
        (32_000, 56_000, 0),
        (24_000, 0, 1),
        (24_000, 40_000, 0),
        (16_000, 0, 1),
        (16_000, 28_000, 0),
        (15_999, 0, 0),
    ];
    const STEREO: &[(u32, u32, u32)] = &[
        (48_000, 0, 2),
        (48_000, 44_000, 2),
        (48_000, 128_000, 0),
        (44_100, 0, 2),
        (44_100, 44_000, 2),
        (44_100, 128_000, 0),
        (32_000, 0, 2),
        (32_000, 32_000, 2),
        (32_000, 68_000, 1),
        (32_000, 96_000, 0),
        (24_000, 0, 1),
        (24_000, 48_000, 1),
        (24_000, 80_000, 0),
        (16_000, 0, 1),
        (16_000, 32_000, 1),
        (16_000, 64_000, 0),
        (15_999, 0, 0),
    ];
    let table = match channels {
        1 => MONO,
        2 => STEREO,
        _ => return None,
    };
    let mut selected = None;
    for &(maximum_sample_rate, minimum_bitrate, mode) in table {
        if sample_rate <= maximum_sample_rate && bitrate >= minimum_bitrate {
            selected = Some(mode);
        }
    }
    selected
}

fn resolve_bandwidth(
    proposed: u32,
    bitrate: u32,
    bitrate_mode: u32,
    sample_rate: u32,
    frame_length: u32,
    effective_channels: usize,
    mono: bool,
) -> Result<u32, EncoderConfigurationError> {
    let bandwidth = if bitrate_mode != 0 {
        if proposed != 0 {
            proposed
        } else {
            match bitrate_mode {
                1 | 2 => 13_000,
                3 => 15_750,
                4 => 16_500,
                5 => 19_293,
                _ => return Err(EncoderConfigurationError::InvalidBitrateMode(bitrate_mode)),
            }
        }
    } else if proposed != 0 {
        proposed.min(20_000).min(sample_rate / 2)
    } else {
        let channel_bitrate = bitrate / effective_channels as u32;
        automatic_cbr_bandwidth(frame_length, sample_rate, channel_bitrate, mono)
            .ok_or(EncoderConfigurationError::InvalidChannelBitrate)?
    };
    Ok(bandwidth.min(sample_rate / 2))
}

fn automatic_cbr_bandwidth(
    frame_length: u32,
    sample_rate: u32,
    channel_bitrate: u32,
    mono: bool,
) -> Option<u32> {
    const LC: &[(u32, u32, u32)] = &[
        (0, 3_700, 5_000),
        (12_000, 5_000, 6_400),
        (20_000, 6_900, 9_640),
        (28_000, 9_600, 13_050),
        (40_000, 12_060, 14_260),
        (56_000, 13_950, 15_500),
        (72_000, 14_200, 16_120),
        (96_000, 17_000, 17_000),
        (576_001, 17_000, 17_000),
    ];
    const LD_22050: &[(u32, u32, u32)] = &[
        (8_000, 2_000, 2_400),
        (12_000, 2_500, 2_700),
        (16_000, 3_300, 3_100),
        (24_000, 6_250, 7_200),
        (32_000, 9_200, 10_500),
        (40_000, 16_000, 16_000),
        (48_000, 16_000, 16_000),
        (282_241, 16_000, 16_000),
    ];
    const LD_24000: &[(u32, u32, u32)] = &[
        (8_000, 2_000, 2_000),
        (12_000, 2_000, 2_300),
        (16_000, 2_200, 2_500),
        (24_000, 5_650, 7_200),
        (32_000, 11_600, 12_000),
        (40_000, 12_000, 16_000),
        (48_000, 16_000, 16_000),
        (64_000, 16_000, 16_000),
        (307_201, 16_000, 16_000),
    ];
    const LD_32000: &[(u32, u32, u32)] = &[
        (8_000, 2_000, 2_000),
        (12_000, 2_000, 2_000),
        (24_000, 4_250, 7_200),
        (32_000, 8_400, 9_000),
        (40_000, 9_400, 11_300),
        (48_000, 11_900, 14_700),
        (64_000, 14_800, 16_000),
        (76_000, 16_000, 16_000),
        (409_601, 16_000, 16_000),
    ];
    const LD_44100: &[(u32, u32, u32)] = &[
        (8_000, 2_000, 2_000),
        (24_000, 2_000, 2_000),
        (32_000, 4_400, 5_700),
        (40_000, 7_400, 8_800),
        (48_000, 9_000, 10_700),
        (56_000, 11_000, 12_900),
        (64_000, 14_400, 15_500),
        (80_000, 16_000, 16_200),
        (96_000, 16_500, 16_000),
        (128_000, 16_000, 16_000),
        (564_481, 16_000, 16_000),
    ];
    const LD_48000: &[(u32, u32, u32)] = &[
        (8_000, 2_000, 2_000),
        (24_000, 2_000, 2_000),
        (32_000, 4_400, 5_700),
        (40_000, 7_400, 8_800),
        (48_000, 9_000, 10_700),
        (56_000, 11_000, 12_800),
        (64_000, 14_300, 15_400),
        (80_000, 16_000, 16_200),
        (96_000, 16_500, 16_000),
        (128_000, 16_000, 16_000),
        (614_401, 16_000, 16_000),
    ];

    let (table, interpolate): (&[(u32, u32, u32)], bool) = match frame_length {
        960 | 1024 => (LC, false),
        120 | 128 | 240 | 256 | 480 | 512 => (
            match sample_rate {
                8_000 | 11_025 | 12_000 | 16_000 | 22_050 => LD_22050,
                24_000 => LD_24000,
                32_000 => LD_32000,
                44_100 => LD_44100,
                48_000 | 64_000 | 88_200 | 96_000 => LD_48000,
                _ => return None,
            },
            true,
        ),
        _ => return None,
    };
    for pair in table.windows(2) {
        let (start_rate, start_mono, start_multi) = pair[0];
        let (end_rate, end_mono, end_multi) = pair[1];
        if channel_bitrate >= start_rate && channel_bitrate < end_rate {
            let start = if mono { start_mono } else { start_multi };
            if !interpolate {
                return Some(start);
            }
            let end = if mono { end_mono } else { end_multi };
            let delta = i64::from(end) - i64::from(start);
            let offset = i64::from(channel_bitrate - start_rate);
            let span = i64::from(end_rate - start_rate);
            return Some((i64::from(start) + delta * offset / span) as u32);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_delay_cbr_reservoir_matches_fdk_interpolation_and_caps() {
        assert_eq!(low_delay_cbr_reservoir_capacity(12_000, 1, 1, 128), 496);
        assert_eq!(low_delay_cbr_reservoir_capacity(41_000, 1, 1, 448), 2_248);
        assert_eq!(low_delay_cbr_reservoir_capacity(70_000, 1, 1, 752), 4_000);
        assert_eq!(
            low_delay_cbr_reservoir_capacity(140_000, 2, 2, 1_504),
            4_000
        );
        assert_eq!(low_delay_cbr_reservoir_capacity(70_000, 1, 1, 6_140), 0);
    }

    #[test]
    fn defaults_match_the_uninitialized_c_encoder_user_parameters() {
        let parameters = PureRustEncoderParameters::new(8);
        assert_eq!(
            parameters.get_parameter(EncoderParameter::AudioObjectType),
            2
        );
        assert_eq!(
            parameters.get_parameter(EncoderParameter::Bitrate),
            u32::MAX
        );
        assert_eq!(
            parameters.get_parameter(EncoderParameter::GranuleLength),
            u32::MAX
        );
        assert_eq!(
            parameters.get_parameter(EncoderParameter::TransportMux),
            u32::MAX
        );
        assert_eq!(
            parameters.get_parameter(EncoderParameter::HeaderPeriod),
            0xff
        );
        assert_eq!(parameters.initialization_flags(), AACENC_INIT_ALL);
    }

    #[test]
    fn setters_validate_every_bounded_parameter_and_track_init_flags() {
        let mut parameters = PureRustEncoderParameters::new(8);
        parameters.clear_initialization_flags();
        parameters
            .set_parameter(EncoderParameter::SampleRate, 44_100)
            .unwrap();
        assert_eq!(
            parameters.initialization_flags(),
            AACENC_INIT_CONFIG | AACENC_INIT_STATES | AACENC_INIT_TRANSPORT
        );
        parameters.clear_initialization_flags();
        parameters
            .set_parameter(EncoderParameter::Afterburner, 1)
            .unwrap();
        assert_eq!(parameters.initialization_flags(), AACENC_INIT_CONFIG);
        assert!(parameters
            .set_parameter(EncoderParameter::Afterburner, 2)
            .is_err());
        assert!(parameters
            .set_parameter(EncoderParameter::BitrateMode, 6)
            .is_err());
        assert!(parameters
            .set_parameter(EncoderParameter::TransportSubframes, 0)
            .is_err());
        assert!(parameters
            .set_parameter(EncoderParameter::AudioMuxVersion, 3)
            .is_err());
    }

    #[test]
    fn input_layout_changes_and_control_reset_clear_buffer_fill() {
        let mut parameters = PureRustEncoderParameters::new(8);
        parameters.clear_initialization_flags();
        parameters.set_input_buffer_fill(17);
        parameters
            .set_parameter(EncoderParameter::ChannelOrder, 1)
            .unwrap();
        assert_eq!(parameters.input_buffer_fill(), 0);
        parameters.set_input_buffer_fill(19);
        parameters
            .set_parameter(EncoderParameter::ControlState, AACENC_RESET_INBUFFER)
            .unwrap();
        assert_eq!(parameters.input_buffer_fill(), 0);
        assert_eq!(parameters.initialization_flags(), AACENC_RESET_INBUFFER);
    }

    fn configured(aot: u32, channels: u32, sample_rate: u32) -> PureRustEncoderParameters {
        let mut parameters = PureRustEncoderParameters::new(8);
        parameters
            .set_parameter(EncoderParameter::AudioObjectType, aot)
            .unwrap();
        parameters
            .set_parameter(EncoderParameter::ChannelMode, channels)
            .unwrap();
        parameters
            .set_parameter(EncoderParameter::SampleRate, sample_rate)
            .unwrap();
        parameters
    }

    #[test]
    fn configured_low_delay_encoder_starts_with_fdk_cbr_reservoir() {
        let mut parameters = configured(23, 1, 48_000);
        parameters
            .set_parameter(EncoderParameter::Bitrate, 48_000)
            .unwrap();
        let encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
        let expected = low_delay_cbr_reservoir_capacity(
            encoder.config.bitrate,
            encoder.config.channels,
            encoder.config.effective_channels,
            encoder.config.nominal_frame_bits,
        );
        let PureRustEncoderBackend::LdMono(codec) = &encoder.backend else {
            panic!("expected AAC-LD mono backend");
        };
        assert_eq!(codec.bit_reservoir().capacity_bits(), expected);
        assert_eq!(codec.bit_reservoir().fullness_bits(), expected);
    }

    #[test]
    fn configured_low_delay_cbr_writes_fill_data_without_losing_reservoir_bits() {
        for aot in [23, 39] {
            let mut parameters = configured(aot, 1, 48_000);
            if aot == 39 {
                parameters
                    .set_parameter(EncoderParameter::SbrMode, 0)
                    .unwrap();
            }
            parameters
                .set_parameter(EncoderParameter::Bitrate, 48_000)
                .unwrap();
            parameters
                .set_parameter(EncoderParameter::TransportMux, 0)
                .unwrap();
            let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
            let asc = backend_audio_specific_config(&encoder.config, &encoder.backend).unwrap();
            let raw = encoder.encode_interleaved_f32(&vec![0.0; 512]).unwrap();
            assert_eq!(raw.len(), 64);

            let fullness = match &encoder.backend {
                PureRustEncoderBackend::LdMono(codec) => codec.bit_reservoir().fullness_bits(),
                PureRustEncoderBackend::EldMono(codec) => codec.bit_reservoir().fullness_bits(),
                _ => unreachable!(),
            };
            let capacity = match &encoder.backend {
                PureRustEncoderBackend::LdMono(codec) => codec.bit_reservoir().capacity_bits(),
                PureRustEncoderBackend::EldMono(codec) => codec.bit_reservoir().capacity_bits(),
                _ => unreachable!(),
            };
            assert_eq!(fullness, capacity);

            let mut pure = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
            assert_eq!(pure.decode_raw_data_block_f32(&raw).unwrap().channels(), 1);

            #[cfg(feature = "ffi")]
            {
                let mut config = asc.to_bytes().unwrap();
                let mut c = crate::Decoder::open(crate::TransportType::Raw).unwrap();
                c.configure_raw(&mut config).unwrap();
                let mut pcm = vec![0i16; 1024];
                assert_eq!(c.decode_access_unit_i16(&raw, &mut pcm).unwrap(), 512);

                let config = crate::EncoderConfig {
                    channels: 1,
                    sample_rate: 48_000,
                    bitrate: 48_000,
                    channel_mode: crate::ChannelMode::Mono,
                    audio_object_type: if aot == 23 {
                        crate::AudioObjectType::Other(23)
                    } else {
                        crate::AudioObjectType::AacEld
                    },
                    transport: crate::TransportType::Raw,
                    afterburner: true,
                    sbr_mode: (aot == 39).then_some(0),
                };
                let mut c_encoder = crate::Encoder::configured(&config).unwrap();
                let mut output = vec![0u8; 2_048];
                let input = vec![0i16; 512];
                let mut observed = 0;
                for _ in 0..8 {
                    let bytes = c_encoder
                        .encode_interleaved_i16(&input, &mut output)
                        .unwrap();
                    if bytes != 0 {
                        assert_eq!(bytes, raw.len());
                        observed += 1;
                    }
                }
                assert!(observed >= 4);
            }
        }
    }

    #[test]
    fn configured_low_delay_vbr_uses_quality_thresholds_without_cbr_fill() {
        let input = (0..512)
            .map(|sample| {
                let t = sample as f32 / 48_000.0;
                10_000.0 * (2.0 * std::f32::consts::PI * 997.0 * t).sin()
                    + 4_000.0 * (2.0 * std::f32::consts::PI * 7_013.0 * t).sin()
            })
            .collect::<Vec<_>>();
        let mut sizes = Vec::new();
        for mode in [1, 5] {
            let mut parameters = configured(23, 1, 48_000);
            parameters
                .set_parameter(EncoderParameter::BitrateMode, mode)
                .unwrap();
            parameters
                .set_parameter(EncoderParameter::TransportMux, 0)
                .unwrap();
            let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
            let asc = backend_audio_specific_config(&encoder.config, &encoder.backend).unwrap();
            let raw = encoder.encode_interleaved_f32(&input).unwrap();
            assert_ne!(raw.len(), encoder.config.nominal_frame_bits.div_ceil(8));
            let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
            assert_eq!(
                decoder.decode_raw_data_block_f32(&raw).unwrap().channels(),
                1
            );
            sizes.push(raw.len());
        }
        assert!(sizes[1] >= sizes[0], "VBR1/VBR5 sizes {sizes:?}");
    }

    #[test]
    fn metadata_modes_embed_separate_ga_drc_etsi_and_user_elements() {
        use crate::decoder::STREAM_FLAG_DRC_PRESENT;

        for mode in 1..=3 {
            let mut parameters = configured(2, 1, 48_000);
            parameters
                .set_parameter(EncoderParameter::MetadataMode, mode)
                .unwrap();
            let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
            let user = [0x12, 0x34, 0x56];
            let (raw, consumed) = encoder
                .encode_interleaved_f32_with_ancillary(&vec![0.0; 1024], &user)
                .unwrap();
            assert_eq!(consumed, user.len());

            let mut decoder = AacLcDecoder::new(3, 1).unwrap();
            decoder.init_ancillary_data(64);
            decoder.decode_raw_data_block_f32(&raw).unwrap();
            assert_eq!(
                decoder.stream_info().flags & STREAM_FLAG_DRC_PRESENT != 0,
                matches!(mode, 1 | 2)
            );
            let ancillary = decoder.ancillary_data();
            if matches!(mode, 2 | 3) {
                assert_eq!(ancillary.len(), 2);
                assert_eq!(ancillary[0].data, [0xbc, 0xc0, 0x00]);
                assert_eq!(ancillary[1].data, user);
            } else {
                assert_eq!(ancillary.len(), 1);
                assert_eq!(ancillary[0].data, user);
            }
        }
    }

    #[test]
    fn live_metadata_mode_enable_delay_and_disable_finalization() {
        use crate::decoder::STREAM_FLAG_DRC_PRESENT;

        fn metadata_presence(raw: &[u8]) -> (bool, bool) {
            let mut decoder = AacLcDecoder::new(3, 1).unwrap();
            decoder.init_ancillary_data(64);
            decoder.decode_raw_data_block_f32(raw).unwrap();
            (
                decoder.stream_info().flags & STREAM_FLAG_DRC_PRESENT != 0,
                decoder
                    .ancillary_data()
                    .iter()
                    .any(|element| element.data.starts_with(&[0xbc])),
            )
        }
        let parameters = configured(2, 1, 48_000);
        let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
        assert_eq!(encoder.metadata_gain_delay_frames, 1);
        let silence = vec![0.0; 1024];
        assert_eq!(
            metadata_presence(&encoder.encode_interleaved_f32(&silence).unwrap()),
            (false, false)
        );

        encoder.set_metadata_mode(2).unwrap();
        assert_eq!(encoder.config().metadata_mode, 2);
        // Live enable clears the pending metadata line, so the first new setup
        // appears only after the configured one-frame metadata delay.
        assert_eq!(
            metadata_presence(&encoder.encode_interleaved_f32(&silence).unwrap()),
            (false, false)
        );
        assert_eq!(
            metadata_presence(&encoder.encode_interleaved_f32(&silence).unwrap()),
            (true, true)
        );

        encoder.set_metadata_mode(0).unwrap();
        assert_eq!(encoder.config().metadata_mode, 0);
        // First drain the already queued setup, then emit FDK's one explicit
        // default-configuration terminator before metadata disappears.
        assert_eq!(
            metadata_presence(&encoder.encode_interleaved_f32(&silence).unwrap()),
            (true, true)
        );
        assert_eq!(
            metadata_presence(&encoder.encode_interleaved_f32(&silence).unwrap()),
            (true, true)
        );
        assert_eq!(
            metadata_presence(&encoder.encode_interleaved_f32(&silence).unwrap()),
            (false, false)
        );

        assert!(matches!(
            encoder.set_metadata_mode(4),
            Err(EncoderParameterError::InvalidValue {
                parameter: EncoderParameter::MetadataMode,
                value: 4,
            })
        ));

        let low_delay = configured(23, 1, 48_000);
        let mut low_delay = ConfiguredPureRustEncoder::from_parameters(&low_delay).unwrap();
        assert!(low_delay.set_metadata_mode(1).is_err());
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn live_metadata_transition_sequence_matches_c_encoder() {
        use crate::decoder::STREAM_FLAG_DRC_PRESENT;
        use crate::{sys, Encoder};

        fn metadata_presence(raw: &[u8]) -> (bool, bool) {
            let mut decoder = AacLcDecoder::new(3, 1).unwrap();
            decoder.init_ancillary_data(64);
            decoder.decode_raw_data_block_f32(raw).unwrap();
            (
                decoder.stream_info().flags & STREAM_FLAG_DRC_PRESENT != 0,
                decoder
                    .ancillary_data()
                    .iter()
                    .any(|element| element.data.starts_with(&[0xbc])),
            )
        }

        let mut c = Encoder::open(1).unwrap();
        c.set_param(sys::AACENC_AOT, 2).unwrap();
        c.set_param(sys::AACENC_CHANNELMODE, 1).unwrap();
        c.set_param(sys::AACENC_SAMPLERATE, 48_000).unwrap();
        c.set_param(sys::AACENC_BITRATE, 96_000).unwrap();
        c.set_param(sys::AACENC_TRANSMUX, 0).unwrap();
        c.initialize().unwrap();
        let info = c.info().unwrap();
        let input = vec![0i16; info.frame_length as usize];
        let mut output = vec![0u8; info.max_output_bytes as usize];
        let mut encode = |encoder: &mut Encoder| {
            for _ in 0..4 {
                let bytes = encoder.encode_interleaved_i16(&input, &mut output).unwrap();
                if bytes != 0 {
                    return metadata_presence(&output[..bytes]);
                }
            }
            panic!("C encoder did not emit an access unit");
        };

        let mut observed = vec![encode(&mut c)];
        c.set_param(sys::AACENC_METADATA_MODE, 2).unwrap();
        observed.push(encode(&mut c));
        observed.push(encode(&mut c));
        c.set_param(sys::AACENC_METADATA_MODE, 0).unwrap();
        observed.push(encode(&mut c));
        observed.push(encode(&mut c));
        observed.push(encode(&mut c));

        assert_eq!(
            observed,
            [
                (false, false),
                (false, false),
                (true, true),
                (true, true),
                (true, true),
                (false, false),
            ]
        );
    }

    #[test]
    fn low_delay_profiles_disable_public_metadata_but_preserve_user_ancillary() {
        use crate::decoder::STREAM_FLAG_DRC_PRESENT;
        use crate::encoder_metadata::MetadataDrcProfile;

        let mut parameters = configured(23, 1, 48_000);
        parameters
            .set_parameter(EncoderParameter::MetadataMode, 2)
            .unwrap();
        let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
        assert_eq!(encoder.config().metadata_mode, 0);
        let asc = backend_audio_specific_config(encoder.config(), &encoder.backend).unwrap();
        let metadata = EncoderMetadata {
            compression_profile: MetadataDrcProfile::FilmLight,
            program_reference_level: Some(-(23 << 16)),
            drc_presentation_mode: 1,
            ..EncoderMetadata::default()
        };
        encoder.set_metadata(metadata.clone());
        encoder.set_metadata_frame_gains(-(3 << 16), Some(-(6 << 16)));
        assert_eq!(encoder.metadata(), &metadata);

        let user = [0xaa, 0x55];
        let (raw, consumed) = encoder
            .encode_interleaved_f32_with_ancillary(&vec![0.0; 512], &user)
            .unwrap();
        assert_eq!(consumed, user.len());

        let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
        decoder.init_ancillary_data(64);
        decoder.decode_raw_data_block_f32(&raw).unwrap();
        assert_eq!(decoder.stream_info().flags & STREAM_FLAG_DRC_PRESENT, 0);
        assert_eq!(decoder.ancillary_data().len(), 1);
        assert_eq!(decoder.ancillary_data()[0].data, user);
    }

    #[test]
    fn metadata_delay_matches_fdk_frame_and_audio_alignment() {
        let mut parameters = configured(2, 1, 48_000);
        parameters
            .set_parameter(EncoderParameter::MetadataMode, 1)
            .unwrap();
        let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
        assert_eq!(
            (encoder.encoder_delay(), encoder.encoder_core_delay()),
            (2048, 2048)
        );
        assert_eq!(encoder.metadata_gain_delay_frames, 1);
        assert_eq!(encoder.delay_metadata_gains((1, Some(2))), (1, Some(2)));
        assert_eq!(encoder.delay_metadata_gains((3, Some(4))), (1, Some(2)));
        assert_eq!(encoder.delay_metadata_gains((5, Some(6))), (3, Some(4)));

        let mut impulse = vec![0.0; 1024];
        impulse[0] = 1.0;
        let delayed = encoder.delay_metadata_audio(&impulse);
        assert_eq!(delayed[..448], [0.0; 448]);
        assert_eq!(delayed[448], 1.0);
        assert!(delayed[449..].iter().all(|sample| *sample == 0.0));

        // Delay compensation is active even in metadata mode zero, exactly as
        // in FDK's always-running metadata module for supported GA profiles.
        let he = ConfiguredPureRustEncoder::from_parameters(&configured(5, 1, 48_000)).unwrap();
        assert_eq!(he.config().metadata_mode, 0);
        assert_eq!(he.metadata_gain_delay_frames, 1);
        assert_eq!(he.metadata_audio_delay.len(), 892);
        assert_eq!((he.encoder_delay(), he.encoder_core_delay()), (5058, 4096));

        let ps = ConfiguredPureRustEncoder::from_parameters(&configured(29, 2, 48_000)).unwrap();
        assert_eq!(ps.metadata_gain_delay_frames, 2);
        assert_eq!(ps.metadata_audio_delay.len(), 1057 * 2);
        assert_eq!((ps.encoder_delay(), ps.encoder_core_delay()), (7106, 6144));
    }

    #[test]
    fn eld_mps_delay_includes_decoder_low_delay_qmf_banks() {
        assert_eq!(
            fdk_encoder_delays(39, 128, 24_000, 512, false, 0),
            (384, 256)
        );
        assert_eq!(
            fdk_encoder_delays(39, 128, 48_000, 512, false, 0),
            (512, 256)
        );
        assert_eq!(fdk_encoder_delays(39, 2, 48_000, 512, false, 0), (256, 256));
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn eld_mps_delays_match_c_across_every_supported_sample_rate() {
        use crate::{sys, Encoder};

        for sample_rate in [16_000, 22_050, 24_000, 32_000, 44_100, 48_000] {
            let resolved = configured(39, 128, sample_rate).resolve().unwrap();

            let mut c = Encoder::open(2).unwrap();
            c.set_param(sys::AACENC_AOT, 39).unwrap();
            c.set_param(sys::AACENC_CHANNELMODE, 128).unwrap();
            c.set_param(sys::AACENC_SAMPLERATE, sample_rate).unwrap();
            c.set_param(sys::AACENC_BITRATE, 32_000).unwrap();
            c.set_param(sys::AACENC_TRANSMUX, 0).unwrap();
            c.initialize()
                .unwrap_or_else(|error| panic!("C ELDv2 init failed at {sample_rate} Hz: {error}"));
            let info = c.info().unwrap();

            assert_eq!(
                (resolved.encoder_delay, resolved.encoder_core_delay),
                (info.delay, info.core_delay),
                "ELDv2 delay differs at {sample_rate} Hz"
            );
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn eld_multichannel_high_mode_acceptance_matches_c() {
        for (mode, sbr_ratio) in [7u32, 11, 12, 14, 33, 34]
            .into_iter()
            .flat_map(|mode| [0u32, 1, 2].into_iter().map(move |ratio| (mode, ratio)))
        {
            let mut encoder = crate::Encoder::open(8).unwrap();
            encoder.set_param(crate::sys::AACENC_AOT, 39).unwrap();
            encoder
                .set_param(crate::sys::AACENC_CHANNELMODE, mode)
                .unwrap();
            encoder
                .set_param(crate::sys::AACENC_SAMPLERATE, 48_000)
                .unwrap();
            encoder
                .set_param(
                    crate::sys::AACENC_BITRATE,
                    32_000 * if mode == 11 { 7 } else { 8 },
                )
                .unwrap();
            encoder.set_param(crate::sys::AACENC_TRANSMUX, 0).unwrap();
            encoder
                .set_param(crate::sys::AACENC_SBR_MODE, u32::from(sbr_ratio != 0))
                .unwrap();
            if sbr_ratio != 0 {
                encoder
                    .set_param(crate::sys::AACENC_SBR_RATIO, sbr_ratio)
                    .unwrap();
            }
            let c_accepts = encoder.initialize().is_ok();
            let mut parameters = configured(39, mode, 48_000);
            parameters
                .set_parameter(
                    EncoderParameter::Bitrate,
                    32_000 * if mode == 11 { 7 } else { 8 },
                )
                .unwrap();
            parameters
                .set_parameter(EncoderParameter::TransportMux, 0)
                .unwrap();
            parameters
                .set_parameter(EncoderParameter::SbrMode, u32::from(sbr_ratio != 0))
                .unwrap();
            if sbr_ratio != 0 {
                parameters
                    .set_parameter(EncoderParameter::SbrRatio, sbr_ratio)
                    .unwrap();
            }
            let rust_accepts = ConfiguredPureRustEncoder::from_parameters(&parameters).is_ok();
            assert_eq!(
                rust_accepts, c_accepts,
                "ELD channel mode {mode}, SBR ratio {sbr_ratio}"
            );
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn aac_ld_channel_mode_acceptance_matches_c() {
        let mut c_accepted = Vec::new();
        let mut rust_accepted = Vec::new();
        for mode in [1u32, 2, 3, 4, 5, 6, 7, 11, 12, 14, 33, 34, 128] {
            let channels = channel_count(mode).unwrap();
            let mut encoder = crate::Encoder::open(channels as u32).unwrap();
            encoder.set_param(crate::sys::AACENC_AOT, 23).unwrap();
            encoder
                .set_param(crate::sys::AACENC_CHANNELMODE, mode)
                .unwrap();
            encoder
                .set_param(crate::sys::AACENC_SAMPLERATE, 48_000)
                .unwrap();
            encoder
                .set_param(crate::sys::AACENC_BITRATE, 32_000 * channels as u32)
                .unwrap();
            encoder.set_param(crate::sys::AACENC_TRANSMUX, 0).unwrap();
            let c_accepts = encoder.initialize().is_ok();

            let mut parameters = configured(23, mode, 48_000);
            parameters
                .set_parameter(EncoderParameter::Bitrate, 32_000 * channels as u32)
                .unwrap();
            parameters
                .set_parameter(EncoderParameter::TransportMux, 0)
                .unwrap();
            let rust_accepts = ConfiguredPureRustEncoder::from_parameters(&parameters).is_ok();

            if c_accepts {
                c_accepted.push(mode);
            }
            if rust_accepts {
                rust_accepted.push(mode);
            }
        }
        assert_eq!(c_accepted, [1, 2, 3, 4, 5, 6, 7, 11, 12, 14, 128]);
        assert_eq!(rust_accepted, c_accepted);
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn public_encoder_profile_configuration_matrix_matches_c() {
        use crate::{sys, Encoder};

        let mut mismatches = Vec::new();
        for aot in [2u32, 5, 29, 23, 39] {
            for mode in [1u32, 2, 3, 4, 5, 6, 7, 11, 12, 14, 33, 34, 128] {
                let Some(channels) = channel_count(mode) else {
                    continue;
                };
                for sample_rate in [
                    8_000u32, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000, 64_000, 96_000,
                ] {
                    for frame_length in [480u32, 512, 960, 1024] {
                        let sbr_ratios: &[u32] = match aot {
                            5 | 29 => &[2],
                            39 => &[0, 1, 2],
                            _ => &[0],
                        };
                        for &sbr_ratio in sbr_ratios {
                            let bitrate = 32_000 * channels as u32;
                            let c_accepts = (|| {
                                let mut encoder = Encoder::open(channels as u32).ok()?;
                                encoder.set_param(sys::AACENC_AOT, aot).ok()?;
                                encoder.set_param(sys::AACENC_CHANNELMODE, mode).ok()?;
                                encoder
                                    .set_param(sys::AACENC_SAMPLERATE, sample_rate)
                                    .ok()?;
                                encoder.set_param(sys::AACENC_BITRATE, bitrate).ok()?;
                                encoder
                                    .set_param(sys::AACENC_GRANULE_LENGTH, frame_length)
                                    .ok()?;
                                encoder.set_param(sys::AACENC_TRANSMUX, 0).ok()?;
                                encoder
                                    .set_param(sys::AACENC_SBR_MODE, u32::from(sbr_ratio != 0))
                                    .ok()?;
                                if sbr_ratio != 0 {
                                    encoder.set_param(sys::AACENC_SBR_RATIO, sbr_ratio).ok()?;
                                }
                                encoder.initialize().ok()
                            })()
                            .is_some();

                            let rust_accepts = (|| {
                                let mut parameters = configured(aot, mode, sample_rate);
                                parameters
                                    .set_parameter(EncoderParameter::Bitrate, bitrate)
                                    .ok()?;
                                parameters
                                    .set_parameter(EncoderParameter::GranuleLength, frame_length)
                                    .ok()?;
                                parameters
                                    .set_parameter(EncoderParameter::TransportMux, 0)
                                    .ok()?;
                                parameters
                                    .set_parameter(
                                        EncoderParameter::SbrMode,
                                        u32::from(sbr_ratio != 0),
                                    )
                                    .ok()?;
                                if sbr_ratio != 0 {
                                    parameters
                                        .set_parameter(EncoderParameter::SbrRatio, sbr_ratio)
                                        .ok()?;
                                }
                                ConfiguredPureRustEncoder::from_parameters(&parameters).ok()
                            })()
                            .is_some();
                            if c_accepts != rust_accepts {
                                mismatches.push((
                                    aot,
                                    mode,
                                    sample_rate,
                                    frame_length,
                                    sbr_ratio,
                                    c_accepts,
                                    rust_accepts,
                                ));
                            }
                        }
                    }
                }
            }
        }
        assert!(
            mismatches.is_empty(),
            "public encoder configuration mismatches: {mismatches:#?}"
        );
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn fdk_accepts_pure_rust_aac_ld_multichannel_access_units() {
        use crate::{sys, Decoder, Encoder, TransportType};

        for mode in [3u32, 4, 5, 6, 7, 11, 12, 14, 128] {
            let channels = channel_count(mode).unwrap();
            let bitrate = 32_000 * channels as u32;
            let mut parameters = configured(23, mode, 48_000);
            parameters
                .set_parameter(EncoderParameter::Bitrate, bitrate)
                .unwrap();
            parameters
                .set_parameter(EncoderParameter::TransportMux, 0)
                .unwrap();
            let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
            let mut asc = backend_audio_specific_config(&encoder.config, &encoder.backend)
                .unwrap()
                .to_bytes()
                .unwrap();

            let mut c_encoder = Encoder::open(channels as u32).unwrap();
            c_encoder.set_param(sys::AACENC_AOT, 23).unwrap();
            c_encoder.set_param(sys::AACENC_CHANNELMODE, mode).unwrap();
            c_encoder.set_param(sys::AACENC_SAMPLERATE, 48_000).unwrap();
            c_encoder.set_param(sys::AACENC_BITRATE, bitrate).unwrap();
            c_encoder.set_param(sys::AACENC_TRANSMUX, 0).unwrap();
            c_encoder.initialize().unwrap();
            let c_asc = c_encoder.audio_specific_config().unwrap();
            assert_eq!(asc, c_asc, "mode {mode}: Rust={asc:02x?}, C={c_asc:02x?}");

            let input = (0..512)
                .flat_map(|sample| {
                    (0..channels).map(move |channel| {
                        (sample as f32 * (0.027 + channel as f32 * 0.003)).sin() * 8_000.0
                            / (channel + 1) as f32
                    })
                })
                .collect::<Vec<_>>();
            let mut raw = encoder.encode_interleaved_f32(&input).unwrap();
            let mut decoder = Decoder::open(TransportType::Raw).unwrap();
            decoder.configure_raw(&mut asc).unwrap();
            assert_eq!(decoder.fill(&mut raw).unwrap(), raw.len());
            let mut pcm = vec![0i16; 512 * channels];
            decoder
                .decode_frame(&mut pcm)
                .unwrap_or_else(|error| panic!("mode {mode}: {error}"));
            // The bundled decoder's default PCM renderer caps output at 5.1,
            // while successful parsing still validates every source element.
            assert_eq!(
                decoder.stream_info().unwrap().channels,
                if mode == 128 {
                    1
                } else {
                    channels.min(6) as i32
                },
                "mode {mode}"
            );
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn eld_mps_sample_rate_acceptance_matches_c() {
        use crate::{sys, Encoder};

        for sample_rate in [8_000, 11_025, 12_000, 16_000, 48_000, 64_000, 96_000] {
            let rust_accepts = configured(39, 128, sample_rate).resolve().is_ok();

            let mut c = Encoder::open(2).unwrap();
            c.set_param(sys::AACENC_AOT, 39).unwrap();
            c.set_param(sys::AACENC_CHANNELMODE, 128).unwrap();
            c.set_param(sys::AACENC_SAMPLERATE, sample_rate).unwrap();
            c.set_param(sys::AACENC_BITRATE, 32_000).unwrap();
            c.set_param(sys::AACENC_TRANSMUX, 0).unwrap();

            assert_eq!(
                rust_accepts,
                c.initialize().is_ok(),
                "ELDv2 sample-rate acceptance differs at {sample_rate} Hz"
            );
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn eld_mps_delays_match_c_for_480_and_512_core_frames_with_sbr() {
        use crate::{sys, Encoder};

        for (sample_rate, frame_length, bitrate, sbr_mode, sbr_ratio) in [
            (16_000, 480, 24_000, 0, 0),
            (24_000, 480, 32_000, 0, 0),
            (32_000, 512, 32_000, 0, 0),
            (24_000, 480, 32_000, 1, 1),
            (24_000, 512, 32_000, 1, 1),
            (48_000, 480, 48_000, 1, 2),
            (48_000, 512, 48_000, 1, 2),
        ] {
            let mut rust = configured(39, 128, sample_rate);
            rust.set_parameter(EncoderParameter::GranuleLength, frame_length)
                .unwrap();
            rust.set_parameter(EncoderParameter::Bitrate, bitrate)
                .unwrap();
            rust.set_parameter(EncoderParameter::SbrMode, sbr_mode)
                .unwrap();
            if sbr_ratio != 0 {
                rust.set_parameter(EncoderParameter::SbrRatio, sbr_ratio)
                    .unwrap();
            }
            rust.set_parameter(EncoderParameter::TransportMux, 0)
                .unwrap();
            let resolved = rust.resolve().unwrap();

            let mut c = Encoder::open(2).unwrap();
            c.set_param(sys::AACENC_AOT, 39).unwrap();
            c.set_param(sys::AACENC_CHANNELMODE, 128).unwrap();
            c.set_param(sys::AACENC_SAMPLERATE, sample_rate).unwrap();
            c.set_param(sys::AACENC_GRANULE_LENGTH, frame_length)
                .unwrap();
            c.set_param(sys::AACENC_BITRATE, bitrate).unwrap();
            c.set_param(sys::AACENC_SBR_MODE, sbr_mode).unwrap();
            if sbr_ratio != 0 {
                c.set_param(sys::AACENC_SBR_RATIO, sbr_ratio).unwrap();
            }
            c.set_param(sys::AACENC_TRANSMUX, 0).unwrap();
            c.initialize().unwrap_or_else(|error| {
                panic!(
                    "C ELDv2 init failed for {sample_rate}/{frame_length}, SBR {sbr_ratio}: {error}"
                )
            });
            let info = c.info().unwrap();

            assert_eq!(
                (resolved.encoder_delay, resolved.encoder_core_delay),
                (info.delay, info.core_delay),
                "ELDv2 delay differs for {sample_rate}/{frame_length}, SBR {sbr_ratio}"
            );
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn eld_mps_frame_geometry_acceptance_matches_c() {
        use crate::{sys, Encoder};

        for (sample_rate, frame_length, sbr_ratio) in [
            (24_000, 480, 0),
            (32_000, 480, 0),
            (44_100, 480, 0),
            (48_000, 480, 0),
            (48_000, 480, 2),
            (48_000, 512, 0),
        ] {
            let mut rust = configured(39, 128, sample_rate);
            rust.set_parameter(EncoderParameter::GranuleLength, frame_length)
                .unwrap();
            rust.set_parameter(EncoderParameter::Bitrate, 32_000)
                .unwrap();
            rust.set_parameter(EncoderParameter::SbrMode, u32::from(sbr_ratio != 0))
                .unwrap();
            if sbr_ratio != 0 {
                rust.set_parameter(EncoderParameter::SbrRatio, sbr_ratio)
                    .unwrap();
            }
            rust.set_parameter(EncoderParameter::TransportMux, 0)
                .unwrap();
            let rust_accepts = rust.resolve().is_ok();

            let mut c = Encoder::open(2).unwrap();
            c.set_param(sys::AACENC_AOT, 39).unwrap();
            c.set_param(sys::AACENC_CHANNELMODE, 128).unwrap();
            c.set_param(sys::AACENC_SAMPLERATE, sample_rate).unwrap();
            c.set_param(sys::AACENC_GRANULE_LENGTH, frame_length)
                .unwrap();
            c.set_param(sys::AACENC_BITRATE, 32_000).unwrap();
            c.set_param(sys::AACENC_SBR_MODE, u32::from(sbr_ratio != 0))
                .unwrap();
            if sbr_ratio != 0 {
                c.set_param(sys::AACENC_SBR_RATIO, sbr_ratio).unwrap();
            }
            c.set_param(sys::AACENC_TRANSMUX, 0).unwrap();

            assert_eq!(
                rust_accepts,
                c.initialize().is_ok(),
                "ELDv2 frame-geometry acceptance differs for {sample_rate}/{frame_length}, SBR {sbr_ratio}"
            );
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn eld_mps_configuration_matrix_acceptance_matches_c() {
        use crate::{sys, Encoder};

        let mut mismatches = Vec::new();
        for sample_rate in [16_000, 22_050, 24_000, 32_000, 44_100, 48_000] {
            for frame_length in [480, 512] {
                for sbr_ratio in [0, 1, 2] {
                    for bitrate in [12_000, 16_000, 24_000, 32_000, 48_000, 64_000] {
                        let mut rust = configured(39, 128, sample_rate);
                        rust.set_parameter(EncoderParameter::GranuleLength, frame_length)
                            .unwrap();
                        rust.set_parameter(EncoderParameter::Bitrate, bitrate)
                            .unwrap();
                        rust.set_parameter(EncoderParameter::SbrMode, u32::from(sbr_ratio != 0))
                            .unwrap();
                        if sbr_ratio != 0 {
                            rust.set_parameter(EncoderParameter::SbrRatio, sbr_ratio)
                                .unwrap();
                        }
                        rust.set_parameter(EncoderParameter::TransportMux, 0)
                            .unwrap();
                        let rust_accepts = rust.resolve().is_ok();

                        let mut c = Encoder::open(2).unwrap();
                        c.set_param(sys::AACENC_AOT, 39).unwrap();
                        c.set_param(sys::AACENC_CHANNELMODE, 128).unwrap();
                        c.set_param(sys::AACENC_SAMPLERATE, sample_rate).unwrap();
                        c.set_param(sys::AACENC_GRANULE_LENGTH, frame_length)
                            .unwrap();
                        c.set_param(sys::AACENC_BITRATE, bitrate).unwrap();
                        c.set_param(sys::AACENC_SBR_MODE, u32::from(sbr_ratio != 0))
                            .unwrap();
                        if sbr_ratio != 0 {
                            c.set_param(sys::AACENC_SBR_RATIO, sbr_ratio).unwrap();
                        }
                        c.set_param(sys::AACENC_TRANSMUX, 0).unwrap();
                        let c_accepts = c.initialize().is_ok();

                        if rust_accepts != c_accepts {
                            mismatches.push((
                                sample_rate,
                                frame_length,
                                sbr_ratio,
                                bitrate,
                                rust_accepts,
                                c_accepts,
                            ));
                        }
                    }
                }
            }
        }
        assert!(
            mismatches.is_empty(),
            "ELDv2 matrix mismatches: {mismatches:?}"
        );
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn eld_sbr_mono_stereo_tuning_boundaries_match_c_asc() {
        use crate::{sys, Encoder};

        let mut mismatches = Vec::new();
        for (channels, table) in [(1_u32, ELD_MONO_SBR_TUNING), (2_u32, ELD_STEREO_SBR_TUNING)] {
            for tuning in table {
                for ratio in [1_u32, 2] {
                    let sample_rate = tuning.core_sample_rate * ratio;
                    if sample_rate > 96_000 {
                        continue;
                    }
                    for bitrate in [tuning.bitrate_from, tuning.bitrate_to - 1] {
                        let mut parameters = configured(39, channels, sample_rate);
                        parameters
                            .set_parameter(EncoderParameter::Bitrate, bitrate)
                            .unwrap();
                        parameters
                            .set_parameter(EncoderParameter::SbrMode, 1)
                            .unwrap();
                        parameters
                            .set_parameter(EncoderParameter::SbrRatio, ratio)
                            .unwrap();
                        parameters
                            .set_parameter(EncoderParameter::TransportMux, 0)
                            .unwrap();
                        let rust = ConfiguredPureRustEncoder::from_parameters(&parameters)
                            .and_then(|encoder| {
                                backend_audio_specific_config(&encoder.config, &encoder.backend)
                            })
                            .and_then(|asc| asc.to_bytes().map_err(Into::into));

                        let mut c = Encoder::open(channels).unwrap();
                        c.set_param(sys::AACENC_AOT, 39).unwrap();
                        c.set_param(sys::AACENC_CHANNELMODE, channels).unwrap();
                        c.set_param(sys::AACENC_SAMPLERATE, sample_rate).unwrap();
                        c.set_param(sys::AACENC_BITRATE, bitrate).unwrap();
                        c.set_param(sys::AACENC_SBR_MODE, 1).unwrap();
                        c.set_param(sys::AACENC_SBR_RATIO, ratio).unwrap();
                        c.set_param(sys::AACENC_TRANSMUX, 0).unwrap();
                        let c_asc = c.initialize().and_then(|_| c.audio_specific_config());

                        match (rust, c_asc) {
                            (Ok(rust), Ok(c)) if rust == c => {}
                            (Err(_), Err(_)) => {}
                            (rust, c) => mismatches.push((
                                channels,
                                sample_rate,
                                ratio,
                                bitrate,
                                rust.ok(),
                                c.ok(),
                            )),
                        }
                    }
                }
            }
        }
        assert!(
            mismatches.is_empty(),
            "ELD SBR tuning/ASC mismatches: {mismatches:?}"
        );
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn eld_stereo_sbr_full_frame_parses_from_c_and_rust_payloads() {
        use crate::{sys, Encoder};

        let mut parameters = configured(39, 2, 48_000);
        parameters
            .set_parameter(EncoderParameter::Bitrate, 64_000)
            .unwrap();
        parameters
            .set_parameter(EncoderParameter::SbrMode, 1)
            .unwrap();
        parameters
            .set_parameter(EncoderParameter::SbrRatio, 2)
            .unwrap();
        parameters
            .set_parameter(EncoderParameter::TransportMux, 0)
            .unwrap();
        let mut rust_encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
        let rust_asc =
            backend_audio_specific_config(&rust_encoder.config, &rust_encoder.backend).unwrap();

        let mut c_encoder = Encoder::open(2).unwrap();
        c_encoder.set_param(sys::AACENC_AOT, 39).unwrap();
        c_encoder.set_param(sys::AACENC_CHANNELMODE, 2).unwrap();
        c_encoder.set_param(sys::AACENC_SAMPLERATE, 48_000).unwrap();
        c_encoder.set_param(sys::AACENC_BITRATE, 64_000).unwrap();
        c_encoder.set_param(sys::AACENC_SBR_MODE, 1).unwrap();
        c_encoder.set_param(sys::AACENC_SBR_RATIO, 2).unwrap();
        c_encoder.set_param(sys::AACENC_TRANSMUX, 0).unwrap();
        c_encoder.initialize().unwrap();
        let c_asc =
            AudioSpecificConfig::parse(&c_encoder.audio_specific_config().unwrap()).unwrap();
        assert_eq!(rust_asc.to_bytes().unwrap(), c_asc.to_bytes().unwrap());

        let input = (0..1024)
            .flat_map(|sample| {
                let phase = sample as f32 * 0.071;
                [phase.sin() * 12_000.0, (phase * 1.07 + 0.4).sin() * 9_000.0]
            })
            .collect::<Vec<_>>();
        let rust_input = input
            .iter()
            .map(|&value| f32::from(value as i16))
            .collect::<Vec<_>>();
        let rust_raw = rust_encoder.encode_interleaved_f32(&rust_input).unwrap();
        let mut rust_decoder = AacLcDecoder::from_audio_specific_config(&rust_asc).unwrap();
        rust_decoder
            .decode_raw_data_block_multichannel_f32(&rust_raw)
            .unwrap();
        assert_eq!(rust_decoder.last_ld_sbr_frames().len(), 1);

        let c_info = c_encoder.info().unwrap();
        let c_input = input.iter().map(|&value| value as i16).collect::<Vec<_>>();
        let mut c_output = vec![0_u8; c_info.max_output_bytes as usize];
        let mut c_bytes = 0;
        let mut c_calls = 0;
        unsafe {
            fdk_aac_sys::fdk_sbr_envelope_capture_enable(1);
            fdk_aac_sys::fdk_sbr_qmf_input_capture_enable(1);
        }
        for _ in 0..12 {
            c_calls += 1;
            c_bytes = c_encoder
                .encode_interleaved_i16(&c_input, &mut c_output)
                .unwrap();
            if c_bytes != 0 {
                break;
            }
        }
        assert_ne!(c_bytes, 0);
        let mut captured = [vec![0_i8; 128], vec![0_i8; 128]];
        let mut captured_scales = [(0, 0, 0); 2];
        let mut prequant = [vec![0_i32; 128], vec![0_i32; 128]];
        let mut prequant_counts = [vec![0_i32; 128], vec![0_i32; 128]];
        let mut common_scales = [0_i32; 2];
        let mut c_qmf_input = [vec![0_i16; 2048], vec![0_i16; 2048]];
        let mut c_qmf_real = [vec![0_i32; 2048], vec![0_i32; 2048]];
        let mut c_qmf_imaginary = [vec![0_i32; 2048], vec![0_i32; 2048]];
        for channel in 0..2 {
            let mut scale0 = 0;
            let mut scale1 = 0;
            let mut qmf_scale = 0;
            let count = unsafe {
                fdk_aac_sys::fdk_sbr_envelope_capture_get(
                    channel as i32,
                    captured[channel].as_mut_ptr(),
                    captured[channel].len() as i32,
                    &mut scale0,
                    &mut scale1,
                    &mut qmf_scale,
                )
            };
            assert!(count >= 0);
            captured[channel].truncate(count as usize);
            captured_scales[channel] = (scale0, scale1, qmf_scale);
            let count = unsafe {
                fdk_aac_sys::fdk_sbr_prequant_capture_get(
                    channel as i32,
                    prequant[channel].as_mut_ptr(),
                    prequant_counts[channel].as_mut_ptr(),
                    prequant[channel].len() as i32,
                    &mut common_scales[channel],
                )
            };
            assert!(count >= 0);
            prequant[channel].truncate(count as usize);
            prequant_counts[channel].truncate(count as usize);
            let count = unsafe {
                fdk_aac_sys::fdk_sbr_qmf_input_capture_get(
                    channel as i32,
                    c_qmf_input[channel].as_mut_ptr(),
                    c_qmf_input[channel].len() as i32,
                )
            };
            assert!(count >= 0);
            c_qmf_input[channel].truncate(count as usize);
            let count = unsafe {
                fdk_aac_sys::fdk_sbr_qmf_output_capture_get(
                    channel as i32,
                    c_qmf_real[channel].as_mut_ptr(),
                    c_qmf_imaginary[channel].as_mut_ptr(),
                    c_qmf_real[channel].len() as i32,
                )
            };
            assert!(count >= 0);
            c_qmf_real[channel].truncate(count as usize);
            c_qmf_imaginary[channel].truncate(count as usize);
        }
        unsafe {
            fdk_aac_sys::fdk_sbr_envelope_capture_enable(0);
            fdk_aac_sys::fdk_sbr_qmf_input_capture_enable(0);
        }
        for channel in 0..2 {
            let rust_channel = rust_input
                .iter()
                .skip(channel)
                .step_by(2)
                .map(|&sample| sample as i16)
                .collect::<Vec<_>>();
            let mut expected = vec![0_i16; 5];
            expected.extend_from_slice(&rust_channel[..rust_channel.len() - 5]);
            assert_eq!(c_qmf_input[channel], expected);
            let mut qmf = crate::ld_sbr_qmf::LdSbrQmfAnalysis::new_cldfb(64).unwrap();
            let rust_slots = qmf
                .process_frame(
                    &expected
                        .iter()
                        .map(|&sample| f64::from(sample))
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let rust_real = rust_slots
                .iter()
                .flat_map(|slot| slot.real.iter())
                .map(|&value| (value * 16_777_216.0) as i32)
                .collect::<Vec<_>>();
            let rust_imaginary = rust_slots
                .iter()
                .flat_map(|slot| slot.imaginary.iter())
                .map(|&value| (value * 16_777_216.0) as i32)
                .collect::<Vec<_>>();
            assert_eq!(c_qmf_real[channel], rust_real);
            assert_eq!(c_qmf_imaginary[channel], rust_imaginary);
        }
        for _ in 1..c_calls {
            let rust_raw = rust_encoder.encode_interleaved_f32(&rust_input).unwrap();
            rust_decoder
                .decode_raw_data_block_multichannel_f32(&rust_raw)
                .unwrap();
        }
        let rust_prequant = match &rust_encoder.backend {
            PureRustEncoderBackend::EldStereo(encoder) => {
                encoder.last_sbr_prequant_debug.clone().unwrap()
            }
            _ => unreachable!(),
        };
        let mut c_decoder = AacLcDecoder::from_audio_specific_config(&c_asc).unwrap();
        c_decoder
            .decode_raw_data_block_multichannel_f32(&c_output[..c_bytes])
            .unwrap();
        assert_eq!(c_decoder.last_ld_sbr_frames().len(), 1);

        let rust_frame = &rust_decoder.last_ld_sbr_frames()[0];
        let c_frame = &c_decoder.last_ld_sbr_frames()[0];
        assert_eq!(captured_scales, [(15, 5, 1), (15, 6, 2)]);
        assert_eq!(common_scales, [-2, -1]);
        assert_eq!(
            captured[0]
                .iter()
                .map(|&value| i16::from(value))
                .collect::<Vec<_>>(),
            c_frame
                .left
                .envelopes
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>()
        );
        assert_eq!(
            captured[1]
                .iter()
                .map(|&value| i16::from(value))
                .collect::<Vec<_>>(),
            c_frame
                .right
                .as_ref()
                .unwrap()
                .envelopes
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>()
        );
        assert_eq!(
            prequant[0],
            [38, 23, 26, 16, 30, 23, 19, 14, 12, 11, 8, 7, 8, 6]
        );
        assert_eq!(
            prequant_counts[0],
            [16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 32, 64, 80]
        );
        assert_eq!(
            rust_prequant.0.ybuffer_scales,
            (captured_scales[0].0, captured_scales[0].1)
        );
        assert_eq!(rust_prequant.0.qmf_scale, captured_scales[0].2);
        assert_eq!(rust_prequant.0.common_scale, common_scales[0]);
        assert_eq!(
            rust_prequant
                .0
                .counts
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>(),
            prequant_counts[0]
        );
        assert_eq!(
            rust_prequant.1.ybuffer_scales,
            (captured_scales[1].0, captured_scales[1].1)
        );
        assert_eq!(rust_prequant.1.qmf_scale, captured_scales[1].2);
        assert_eq!(rust_prequant.1.common_scale, common_scales[1]);
        assert_eq!(
            rust_prequant
                .1
                .counts
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>(),
            prequant_counts[1]
        );
        let maximum_prequant_error = rust_prequant
            .0
            .energies
            .iter()
            .chain(&rust_prequant.1.energies)
            .flatten()
            .copied()
            .zip(prequant[0].iter().chain(&prequant[1]).copied())
            .map(|(rust, c)| rust.abs_diff(c))
            .max()
            .unwrap_or(0);
        assert!(
            maximum_prequant_error == 0,
            "maximum prequant SFB energy error {maximum_prequant_error}"
        );
        assert_eq!(rust_frame.active_header, c_frame.active_header);
        assert_eq!(rust_frame.prefix.coupling, c_frame.prefix.coupling);
        assert_eq!(rust_frame.prefix.left.grid, c_frame.prefix.left.grid);
        assert_eq!(
            rust_frame.prefix.right.as_ref().map(|right| &right.grid),
            c_frame.prefix.right.as_ref().map(|right| &right.grid)
        );
        assert_eq!(
            rust_frame.prefix.left.envelope_time_domain,
            c_frame.prefix.left.envelope_time_domain
        );
        assert_eq!(
            rust_frame.prefix.left.noise_time_domain,
            c_frame.prefix.left.noise_time_domain
        );
        assert_eq!(
            rust_frame
                .prefix
                .right
                .as_ref()
                .map(|right| (&right.envelope_time_domain, &right.noise_time_domain)),
            c_frame
                .prefix
                .right
                .as_ref()
                .map(|right| (&right.envelope_time_domain, &right.noise_time_domain))
        );
        assert_eq!(
            rust_frame.left.inverse_filtering_modes,
            c_frame.left.inverse_filtering_modes
        );
        assert_eq!(
            rust_frame
                .right
                .as_ref()
                .map(|right| &right.inverse_filtering_modes),
            c_frame
                .right
                .as_ref()
                .map(|right| &right.inverse_filtering_modes)
        );
        assert_eq!(rust_frame.left.envelopes, c_frame.left.envelopes);
        assert_eq!(
            rust_frame.right.as_ref().map(|right| &right.envelopes),
            c_frame.right.as_ref().map(|right| &right.envelopes)
        );
        assert_eq!(rust_frame.left.noise, c_frame.left.noise);
        assert_eq!(
            rust_frame.right.as_ref().map(|right| &right.noise),
            c_frame.right.as_ref().map(|right| &right.noise)
        );
        assert_eq!(rust_frame.left_harmonics, c_frame.left_harmonics);
        assert_eq!(rust_frame.right_harmonics, c_frame.right_harmonics);
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn eld_sbr_multichannel_tuning_matrix_matches_c_asc() {
        use crate::{sys, Encoder};
        use std::collections::BTreeSet;

        let mut mismatches = Vec::new();
        for channels in 3_u32..=6 {
            for core_rate in [16_000, 22_050, 24_000, 32_000] {
                for ratio in [1_u32, 2] {
                    let sample_rate = core_rate * ratio;
                    let element_weights: &[(bool, u32)] = match channels {
                        3 => &[(false, 858_993_472), (true, 1_288_490_240)],
                        4 => &[
                            (false, 644_245_120),
                            (true, 858_993_472),
                            (false, 644_245_120),
                        ],
                        5 => &[
                            (false, 558_345_728),
                            (true, 794_568_960),
                            (true, 794_568_960),
                        ],
                        6 => &[
                            (false, 515_396_064),
                            (true, 751_619_264),
                            (true, 751_619_264),
                        ],
                        _ => unreachable!(),
                    };
                    let mut bitrates = BTreeSet::from([
                        60_000, 72_000, 90_000, 120_000, 150_000, 180_000, 240_000,
                    ]);
                    for &(stereo, relative) in element_weights {
                        let table = if stereo {
                            ELD_STEREO_SBR_TUNING
                        } else {
                            ELD_MONO_SBR_TUNING
                        };
                        for boundary in table
                            .iter()
                            .filter(|entry| entry.core_sample_rate == core_rate)
                            .flat_map(|entry| [entry.bitrate_from, entry.bitrate_to])
                        {
                            let total =
                                ((u64::from(boundary) << 31).div_ceil(u64::from(relative))) as u32;
                            for offset in [-16_i64, -1, 0, 1, 16] {
                                let candidate = i64::from(total) + offset;
                                if (8_000..=512_000).contains(&candidate) {
                                    bitrates.insert(candidate as u32);
                                }
                            }
                        }
                    }
                    for bitrate in bitrates {
                        let mut parameters = configured(39, channels, sample_rate);
                        parameters
                            .set_parameter(EncoderParameter::Bitrate, bitrate)
                            .unwrap();
                        parameters
                            .set_parameter(EncoderParameter::SbrMode, 1)
                            .unwrap();
                        parameters
                            .set_parameter(EncoderParameter::SbrRatio, ratio)
                            .unwrap();
                        parameters
                            .set_parameter(EncoderParameter::TransportMux, 0)
                            .unwrap();
                        let rust = ConfiguredPureRustEncoder::from_parameters(&parameters)
                            .and_then(|encoder| {
                                backend_audio_specific_config(&encoder.config, &encoder.backend)
                            })
                            .and_then(|asc| asc.to_bytes().map_err(Into::into));

                        let mut c = Encoder::open(channels).unwrap();
                        c.set_param(sys::AACENC_AOT, 39).unwrap();
                        c.set_param(sys::AACENC_CHANNELMODE, channels).unwrap();
                        c.set_param(sys::AACENC_SAMPLERATE, sample_rate).unwrap();
                        c.set_param(sys::AACENC_BITRATE, bitrate).unwrap();
                        c.set_param(sys::AACENC_SBR_MODE, 1).unwrap();
                        c.set_param(sys::AACENC_SBR_RATIO, ratio).unwrap();
                        c.set_param(sys::AACENC_TRANSMUX, 0).unwrap();
                        let c_asc = c.initialize().and_then(|_| c.audio_specific_config());

                        match (rust, c_asc) {
                            (Ok(rust), Ok(c)) if rust == c => {}
                            (Err(_), Err(_)) => {}
                            (rust, c) => mismatches.push((
                                channels,
                                sample_rate,
                                ratio,
                                bitrate,
                                rust.ok(),
                                c.ok(),
                            )),
                        }
                    }
                }
            }
        }
        assert!(
            mismatches.is_empty(),
            "ELD multichannel SBR tuning/ASC mismatches: {mismatches:?}"
        );
    }

    #[test]
    fn initialization_resolves_c_defaults_for_lc_ld_he_and_vbr() {
        let lc = configured(2, 2, 48_000).resolve().unwrap();
        assert_eq!((lc.frame_length, lc.transport_mux), (1024, 2));
        assert_eq!(lc.bitrate, 144_000);
        assert_eq!(lc.nominal_frame_bits, 3072);

        let ld = configured(23, 1, 48_000).resolve().unwrap();
        assert_eq!((ld.frame_length, ld.transport_mux), (512, 10));

        let he = configured(5, 1, 48_000).resolve().unwrap();
        assert!(he.sbr_active);
        assert_eq!((he.sbr_ratio, he.signaling_mode), (2, 0));
        assert_eq!(he.bitrate, 30_000);

        let mut vbr = configured(2, 2, 48_000);
        vbr.set_parameter(EncoderParameter::BitrateMode, 4).unwrap();
        assert_eq!(vbr.resolve().unwrap().bitrate, 128_000);
        vbr.set_parameter(EncoderParameter::PeakBitrate, 90_000)
            .unwrap();
        let adjusted = vbr.resolve().unwrap();
        assert_eq!((adjusted.bitrate_mode, adjusted.bitrate), (2, 64_000));
    }

    #[test]
    fn initialization_rejects_deferred_cross_parameter_conflicts() {
        let mut lc = configured(2, 1, 48_000);
        lc.set_parameter(EncoderParameter::GranuleLength, 512)
            .unwrap();
        assert!(matches!(
            lc.resolve(),
            Err(EncoderConfigurationError::InvalidFrameLength { .. })
        ));

        let mut he = configured(5, 1, 48_000);
        he.set_parameter(EncoderParameter::SbrRatio, 1).unwrap();
        assert_eq!(
            he.resolve(),
            Err(EncoderConfigurationError::SingleRateSbrRequiresExplicitSignaling)
        );

        let mut loas = configured(5, 1, 48_000);
        loas.set_parameter(EncoderParameter::TransportMux, 10)
            .unwrap();
        loas.set_parameter(EncoderParameter::SignalingMode, 1)
            .unwrap();
        assert_eq!(
            loas.resolve(),
            Err(EncoderConfigurationError::BackwardSignalingRequiresAudioMuxVersion1)
        );
    }

    #[test]
    fn granule_downscale_factor_has_the_same_sticky_setter_semantics_as_c() {
        let mut parameters = configured(39, 1, 48_000);
        parameters
            .set_parameter(EncoderParameter::GranuleLength, 256)
            .unwrap();
        assert_eq!(parameters.resolve().unwrap().downscale_factor, 2);
        parameters
            .set_parameter(EncoderParameter::GranuleLength, 512)
            .unwrap();
        assert_eq!(parameters.resolve().unwrap().downscale_factor, 2);
    }

    #[test]
    fn unified_factory_connects_supported_parameter_sets_to_codec_backends() {
        for &(aot, mode) in &[
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
            (2, 6),
            (5, 1),
            (29, 2),
            (23, 1),
            (23, 2),
            (39, 1),
            (39, 2),
            (39, 128),
        ] {
            let parameters = configured(aot, mode, 48_000);
            let encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
            assert_eq!(encoder.config().audio_object_type, aot);
        }

        let parameters = configured(2, 1, 48_000);
        let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
        let raw = encoder
            .encode_interleaved_f32(&vec![0.0; encoder.input_samples_per_channel()])
            .unwrap();
        assert!(!raw.is_empty());
        assert!(matches!(
            encoder.encode_interleaved_f32(&[]),
            Err(PureRustEncoderError::InterleavedInputLength { .. })
        ));
    }

    #[test]
    fn aac_lc_multichannel_factory_writes_decodable_standard_layouts() {
        for channels in 3..=6 {
            let mut parameters = configured(2, channels as u32, 48_000);
            parameters
                .set_parameter(EncoderParameter::Bitrate, 48_000 * channels as u32)
                .unwrap();
            parameters
                .set_parameter(EncoderParameter::TransportMux, 0)
                .unwrap();
            let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
            let input = vec![0.0; 1024 * channels];
            let raw = encoder.encode_interleaved_f32(&input).unwrap();
            let asc = backend_audio_specific_config(&encoder.config, &encoder.backend).unwrap();
            assert_eq!(asc.channel_configuration, channels as u8);
            let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
            let decoded = decoder
                .decode_raw_data_block_multichannel_f32(&raw)
                .unwrap();
            assert_eq!(decoded.channels.len(), channels);
            assert!(decoded.channels.iter().all(|channel| channel.len() == 1024));

            let mut transient = vec![0.0; 1024 * channels];
            for sample in 512..1024 {
                for channel in 0..channels {
                    transient[sample * channels + channel] =
                        ((sample - 512) as f32 * (0.04 + channel as f32 * 0.003)).sin() * 0.5;
                }
            }
            let silence = vec![0.0; 1024 * channels];
            for input in [&transient, &silence] {
                let raw = encoder.encode_interleaved_f32(input).unwrap();
                let decoded = decoder
                    .decode_raw_data_block_multichannel_f32(&raw)
                    .unwrap();
                assert_eq!(decoded.channels.len(), channels);
            }
        }
    }

    #[test]
    fn aac_lc_multichannel_channel_order_matches_fdk_mapping_tables() {
        for channels in 3..=6 {
            let canonical = (0..1024)
                .flat_map(|sample| {
                    (0..channels).map(move |channel| {
                        (sample as f32 * (0.011 + channel as f32 * 0.002)).sin()
                            * (0.1 + channel as f32 * 0.03)
                    })
                })
                .collect::<Vec<_>>();
            let mut mpeg_parameters = configured(2, channels as u32, 48_000);
            mpeg_parameters
                .set_parameter(EncoderParameter::Bitrate, 48_000 * channels as u32)
                .unwrap();
            mpeg_parameters
                .set_parameter(EncoderParameter::TransportMux, 0)
                .unwrap();
            let mut mpeg = ConfiguredPureRustEncoder::from_parameters(&mpeg_parameters).unwrap();
            let expected = mpeg.encode_interleaved_f32(&canonical).unwrap();

            for order in [1, 2] {
                let map = encoder_channel_input_map(channels as u32, channels, order);
                let mut ordered = vec![0.0; canonical.len()];
                for (source, destination) in canonical
                    .chunks_exact(channels)
                    .zip(ordered.chunks_exact_mut(channels))
                {
                    for mpeg_channel in 0..channels {
                        destination[map[mpeg_channel]] = source[mpeg_channel];
                    }
                }
                let mut parameters = mpeg_parameters.clone();
                parameters
                    .set_parameter(EncoderParameter::ChannelOrder, order)
                    .unwrap();
                let resolved = parameters.resolve().unwrap();
                assert_eq!(resolved.channel_order, order);
                let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
                assert_eq!(encoder.encode_interleaved_f32(&ordered).unwrap(), expected);
            }
        }
    }

    #[test]
    fn aac_lc_five_one_keeps_every_distinct_input_channel_audible() {
        const CHANNELS: usize = 6;
        for order in [0, 1, 2] {
            let mut parameters = configured(2, 6, 48_000);
            parameters
                .set_parameter(EncoderParameter::Bitrate, 288_000)
                .unwrap();
            parameters
                .set_parameter(EncoderParameter::TransportMux, 0)
                .unwrap();
            parameters
                .set_parameter(EncoderParameter::ChannelOrder, order)
                .unwrap();
            let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
            let asc = backend_audio_specific_config(&encoder.config, &encoder.backend).unwrap();
            let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
            let input_map = match encoder_channel_input_map(6, CHANNELS, order) {
                map if map.is_empty() => (0..CHANNELS).collect::<Vec<_>>(),
                map => map.to_vec(),
            };
            let mut energy = [0.0f64; CHANNELS];

            for frame in 0..5 {
                let mut input = vec![0.0; 1024 * CHANNELS];
                for sample in 0..1024 {
                    for canonical_channel in 0..CHANNELS {
                        let position = frame * 1024 + sample;
                        let value = (position as f32 * (0.019 + canonical_channel as f32 * 0.006))
                            .sin()
                            * (0.18 + canonical_channel as f32 * 0.035);
                        input[sample * CHANNELS + input_map[canonical_channel]] = value;
                    }
                }
                let access_unit = encoder.encode_interleaved_f32(&input).unwrap();
                let decoded = decoder
                    .decode_raw_data_block_multichannel_f32(&access_unit)
                    .unwrap();
                if frame >= 2 {
                    for (channel_energy, samples) in energy.iter_mut().zip(decoded.channels) {
                        *channel_energy += samples
                            .iter()
                            .map(|sample| f64::from(*sample) * f64::from(*sample))
                            .sum::<f64>();
                    }
                }
            }

            for (channel, channel_energy) in energy.into_iter().enumerate() {
                assert!(
                    channel_energy > 1.0e-6,
                    "channel order {order}, channel {channel} was silent"
                );
            }
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn aac_lc_supports_all_fdk_seven_and_eight_channel_modes() {
        for (mode, channels, expected_configuration) in [
            (7, 8usize, 7),
            (11, 7, 11),
            (12, 8, 12),
            (14, 8, 14),
            (33, 8, 0),
            (34, 8, 0),
        ] {
            let mut parameters = configured(2, mode, 48_000);
            parameters
                .set_parameter(EncoderParameter::Bitrate, 320_000)
                .unwrap();
            parameters
                .set_parameter(EncoderParameter::TransportMux, 0)
                .unwrap();
            let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
            let asc = backend_audio_specific_config(&encoder.config, &encoder.backend).unwrap();
            assert_eq!(asc.channel_configuration, expected_configuration);
            assert_eq!(asc.program_config.is_some(), matches!(mode, 33 | 34));
            let raw = encoder
                .encode_interleaved_f32(&vec![0.0; 1024 * channels])
                .unwrap();
            let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
            let decoded = decoder
                .decode_raw_data_block_multichannel_f32(&raw)
                .unwrap();
            assert_eq!(decoded.channels.len(), channels, "channel mode {mode}");
            assert!(decoded.channels.iter().all(|channel| channel.len() == 1024));

            let mut asc_bytes = asc.to_bytes().unwrap();
            let mut c_decoder = crate::Decoder::open(crate::TransportType::Raw).unwrap();
            c_decoder.configure_raw(&mut asc_bytes).unwrap();
            let mut c_raw = raw.clone();
            c_decoder.fill(&mut c_raw).unwrap();
            c_decoder
                .decode_frame(&mut vec![0; 2048 * channels])
                .unwrap();
        }
    }

    #[test]
    fn aac_lc_afterburner_changes_mono_stereo_and_multichannel_quantization() {
        for channels in [1usize, 2, 6] {
            let input = (0..1024)
                .flat_map(|sample| {
                    (0..channels).map(move |channel| {
                        let phase = 2.0
                            * std::f32::consts::PI
                            * (23.0 + channel as f32 * 7.0)
                            * sample as f32
                            / 1024.0;
                        0.6 * phase.sin() + 0.2 * (phase * 5.0).sin()
                    })
                })
                .collect::<Vec<_>>();
            let mut off_parameters = configured(2, channels as u32, 48_000);
            off_parameters
                .set_parameter(EncoderParameter::Bitrate, 192_000 * channels as u32)
                .unwrap();
            let mut on_parameters = off_parameters.clone();
            on_parameters
                .set_parameter(EncoderParameter::Afterburner, 1)
                .unwrap();
            let mut off = ConfiguredPureRustEncoder::from_parameters(&off_parameters).unwrap();
            let mut on = ConfiguredPureRustEncoder::from_parameters(&on_parameters).unwrap();
            let off_raw = off.encode_interleaved_f32(&input).unwrap();
            let on_raw = on.encode_interleaved_f32(&input).unwrap();
            assert_ne!(
                off_raw, on_raw,
                "afterburner had no effect for {channels} channels"
            );
        }
    }

    #[test]
    fn aac_lc_vbr_modes_apply_quality_control_and_emit_variable_decodable_aus() {
        use crate::decoder::AacLcDecoder;

        let mut average_sizes = Vec::new();
        for mode in 1..=5 {
            let mut parameters = configured(2, 1, 48_000);
            parameters
                .set_parameter(EncoderParameter::BitrateMode, mode)
                .unwrap();
            parameters
                .set_parameter(EncoderParameter::Afterburner, 1)
                .unwrap();
            let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
            let mut decoder = AacLcDecoder::new(3, 1).unwrap();
            let mut sizes = Vec::new();
            for frame in 0..8 {
                let amplitude = 0.08 + frame as f32 * 0.09;
                let input = (0..1024)
                    .map(|sample| {
                        let phase = 2.0 * std::f32::consts::PI * 31.0 * sample as f32 / 1024.0;
                        amplitude * phase.sin()
                            + amplitude * 0.31 * (phase * (3.0 + frame as f32)).sin()
                    })
                    .collect::<Vec<_>>();
                let raw = encoder.encode_interleaved_f32(&input).unwrap();
                decoder.decode_raw_data_block_f32(&raw).unwrap();
                sizes.push(raw.len());
            }
            assert!(
                sizes.windows(2).any(|pair| pair[0] != pair[1]),
                "VBR mode {mode} emitted fixed-size AUs: {sizes:?}"
            );
            average_sizes.push(sizes.iter().sum::<usize>() as f32 / sizes.len() as f32);
        }
        assert!(average_sizes[0] < average_sizes[2] && average_sizes[1] < average_sizes[2]);
        assert!(average_sizes[2] < average_sizes[3] && average_sizes[3] < average_sizes[4]);
    }

    #[test]
    fn he_aac_and_he_aac_v2_propagate_vbr_and_afterburner_to_the_lc_core() {
        for (aot, channels) in [(5, 1usize), (29, 2usize)] {
            let mut off_parameters = configured(aot, channels as u32, 48_000);
            off_parameters
                .set_parameter(EncoderParameter::BitrateMode, 3)
                .unwrap();
            off_parameters
                .set_parameter(EncoderParameter::Afterburner, 0)
                .unwrap();
            let mut on_parameters = off_parameters.clone();
            on_parameters
                .set_parameter(EncoderParameter::Afterburner, 1)
                .unwrap();
            let mut off = ConfiguredPureRustEncoder::from_parameters(&off_parameters).unwrap();
            let mut on = ConfiguredPureRustEncoder::from_parameters(&on_parameters).unwrap();
            let input = (0..2048)
                .flat_map(|sample| {
                    (0..channels).map(move |channel| {
                        let phase = 2.0
                            * std::f32::consts::PI
                            * (29.0 + channel as f32 * 5.0)
                            * sample as f32
                            / 2048.0;
                        0.42 * phase.sin() + 0.13 * (phase * 4.0).sin()
                    })
                })
                .collect::<Vec<_>>();
            let off_raw = off.encode_interleaved_f32(&input).unwrap();
            let on_raw = on.encode_interleaved_f32(&input).unwrap();
            assert_ne!(off_raw, on_raw, "AOT {aot} ignored afterburner");

            let varied = (0..2048)
                .flat_map(|sample| {
                    (0..channels).map(move |channel| {
                        let phase = sample as f32 * (0.019 + channel as f32 * 0.003);
                        0.19 * phase.sin() + 0.07 * (phase * 3.7).cos()
                    })
                })
                .collect::<Vec<_>>();
            let next = on.encode_interleaved_f32(&varied).unwrap();
            assert_ne!(
                on_raw.len(),
                next.len(),
                "AOT {aot} VBR emitted fixed sizes"
            );
        }
    }

    #[test]
    fn latm_he_aac_emits_the_selected_sbr_signaling_syntax() {
        use crate::latm::LatmAudioMuxElement;

        for mode in 0..=2 {
            let mut parameters = configured(5, 1, 48_000);
            parameters
                .set_parameter(EncoderParameter::Bitrate, 48_000)
                .unwrap();
            parameters
                .set_parameter(EncoderParameter::TransportMux, 6)
                .unwrap();
            parameters
                .set_parameter(EncoderParameter::SignalingMode, mode)
                .unwrap();
            if mode == 1 {
                parameters
                    .set_parameter(EncoderParameter::AudioMuxVersion, 1)
                    .unwrap();
            }
            let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
            let latm = encoder.encode_transport_f32(&vec![0.0; 2048]).unwrap();
            let parsed = LatmAudioMuxElement::parse_aac_lc(&latm).unwrap();
            let asc = parsed.config.unwrap();
            assert_eq!(asc.extension.is_some(), mode != 0);
            if mode != 0 {
                assert_eq!(asc.extension.unwrap().sampling_frequency, 48_000);
                assert_eq!(asc.sampling_frequency, 24_000);
            }
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn c_fdk_aac_lc_afterburner_also_changes_access_units() {
        use crate::{sys, Encoder};

        fn make(afterburner: u32) -> Encoder {
            let mut encoder = Encoder::open(1).unwrap();
            encoder.set_param(sys::AACENC_AOT, 2).unwrap();
            encoder.set_param(sys::AACENC_CHANNELMODE, 1).unwrap();
            encoder.set_param(sys::AACENC_SAMPLERATE, 48_000).unwrap();
            encoder.set_param(sys::AACENC_BITRATE, 64_000).unwrap();
            encoder.set_param(sys::AACENC_TRANSMUX, 0).unwrap();
            encoder
                .set_param(sys::AACENC_AFTERBURNER, afterburner)
                .unwrap();
            encoder.initialize().unwrap();
            encoder
        }

        let input = (0..1024)
            .map(|sample| {
                let phase = sample as f32 * 0.071;
                ((0.37 * phase.sin() + 0.19 * (phase * 2.37).cos()) * 20_000.0) as i16
            })
            .collect::<Vec<_>>();
        let mut off = make(0);
        let mut on = make(1);
        let mut observed_difference = false;
        for _ in 0..6 {
            let mut off_raw = vec![0; 8192];
            let mut on_raw = vec![0; 8192];
            let off_bytes = off.encode_interleaved_i16(&input, &mut off_raw).unwrap();
            let on_bytes = on.encode_interleaved_i16(&input, &mut on_raw).unwrap();
            observed_difference |= off_raw[..off_bytes] != on_raw[..on_bytes];
        }
        assert!(
            observed_difference,
            "C encoder afterburner produced no changed AU"
        );
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn c_fdk_aac_lc_vbr_modes_have_matching_variable_quality_order() {
        use crate::{sys, Encoder};

        let mut average_sizes = Vec::new();
        for mode in 1..=5 {
            let mut encoder = Encoder::open(1).unwrap();
            encoder.set_param(sys::AACENC_AOT, 2).unwrap();
            encoder.set_param(sys::AACENC_CHANNELMODE, 1).unwrap();
            encoder.set_param(sys::AACENC_SAMPLERATE, 48_000).unwrap();
            encoder.set_param(sys::AACENC_BITRATEMODE, mode).unwrap();
            encoder.set_param(sys::AACENC_AFTERBURNER, 1).unwrap();
            encoder.set_param(sys::AACENC_TRANSMUX, 0).unwrap();
            encoder.initialize().unwrap();
            let mut sizes = Vec::new();
            for frame in 0..10 {
                let amplitude = 1_600.0 + frame as f32 * 1_800.0;
                let input = (0..1024)
                    .map(|sample| {
                        let phase = 2.0 * std::f32::consts::PI * 31.0 * sample as f32 / 1024.0;
                        (amplitude * phase.sin()
                            + amplitude * 0.31 * (phase * (3.0 + frame as f32)).sin())
                            as i16
                    })
                    .collect::<Vec<_>>();
                let mut output = vec![0; 8192];
                let bytes = encoder.encode_interleaved_i16(&input, &mut output).unwrap();
                if bytes != 0 {
                    sizes.push(bytes);
                }
            }
            assert!(sizes.len() >= 6);
            assert!(
                sizes.windows(2).any(|pair| pair[0] != pair[1]),
                "C VBR mode {mode} emitted fixed-size AUs: {sizes:?}"
            );
            average_sizes.push(sizes.iter().sum::<usize>() as f32 / sizes.len() as f32);
        }
        // The FDK quality-factor ROM is intentionally non-monotonic for
        // modes 1/2, so individual material can make either one slightly
        // larger. Both remain below mode 3; modes 3--5 increase strictly.
        assert!(average_sizes[0] < average_sizes[2] && average_sizes[1] < average_sizes[2]);
        assert!(average_sizes[2] < average_sizes[3] && average_sizes[3] < average_sizes[4]);
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn he_aac_sbr_signaling_modes_match_c_audio_specific_config() {
        use crate::{sys, Encoder};

        for (aot, channels) in [(5u32, 1u32), (29, 2)] {
            for signaling_mode in 0..=2 {
                let mut c = Encoder::open(channels).unwrap();
                c.set_param(sys::AACENC_AOT, aot).unwrap();
                c.set_param(sys::AACENC_CHANNELMODE, channels).unwrap();
                c.set_param(sys::AACENC_SAMPLERATE, 48_000).unwrap();
                c.set_param(sys::AACENC_BITRATE, if aot == 29 { 32_000 } else { 48_000 })
                    .unwrap();
                c.set_param(sys::AACENC_TRANSMUX, 0).unwrap();
                c.set_param(sys::AACENC_SIGNALING_MODE, signaling_mode)
                    .unwrap();
                c.initialize().unwrap();

                let mut parameters = configured(aot, channels, 48_000);
                parameters
                    .set_parameter(
                        EncoderParameter::Bitrate,
                        if aot == 29 { 32_000 } else { 48_000 },
                    )
                    .unwrap();
                parameters
                    .set_parameter(EncoderParameter::SignalingMode, signaling_mode)
                    .unwrap();
                let pure = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
                let asc = backend_audio_specific_config(pure.config(), &pure.backend).unwrap();
                let (rust_bytes, rust_bits) = asc
                    .to_bytes_with_sbr_signaling(signaling_mode as u8)
                    .unwrap();
                let c_bytes = c.audio_specific_config().unwrap();
                assert_eq!(
                    &rust_bytes[..rust_bits.div_ceil(8)],
                    &c_bytes[..rust_bits.div_ceil(8)],
                    "AOT={aot}, signaling={signaling_mode}"
                );
                let parsed = AudioSpecificConfig::parse(&rust_bytes).unwrap();
                assert_eq!(parsed.extension.is_some(), signaling_mode != 0);
                if signaling_mode != 0 {
                    assert_eq!(parsed.extension.unwrap().ps_present, aot == 29);
                }
            }
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn c_fdk_channel_order_tables_produce_identical_access_units() {
        fn c_encoder(channels: usize, order: u32) -> crate::Encoder {
            let mut encoder = crate::Encoder::open(channels as u32).unwrap();
            encoder.set_param(crate::sys::AACENC_AOT, 2).unwrap();
            encoder
                .set_param(crate::sys::AACENC_SAMPLERATE, 48_000)
                .unwrap();
            encoder
                .set_param(crate::sys::AACENC_CHANNELMODE, channels as u32)
                .unwrap();
            encoder
                .set_param(crate::sys::AACENC_CHANNELORDER, order)
                .unwrap();
            encoder
                .set_param(crate::sys::AACENC_BITRATE, 48_000 * channels as u32)
                .unwrap();
            encoder.set_param(crate::sys::AACENC_TRANSMUX, 0).unwrap();
            encoder
                .set_param(crate::sys::AACENC_AFTERBURNER, 0)
                .unwrap();
            encoder.initialize().unwrap();
            encoder
        }

        for channels in 3..=6 {
            let canonical = (0..1024)
                .flat_map(|sample| {
                    (0..channels).map(move |channel| {
                        ((sample as f32 * (0.011 + channel as f32 * 0.002)).sin()
                            * (2000.0 + channel as f32 * 300.0)) as i16
                    })
                })
                .collect::<Vec<_>>();
            for order in [1, 2] {
                let map = encoder_channel_input_map(channels as u32, channels, order);
                let mut ordered = vec![0i16; canonical.len()];
                for (source, destination) in canonical
                    .chunks_exact(channels)
                    .zip(ordered.chunks_exact_mut(channels))
                {
                    for mpeg_channel in 0..channels {
                        destination[map[mpeg_channel]] = source[mpeg_channel];
                    }
                }
                let mut mpeg = c_encoder(channels, 0);
                let mut mapped = c_encoder(channels, order);
                for _ in 0..3 {
                    let mut mpeg_output = vec![0u8; 16_384];
                    let mut mapped_output = vec![0u8; 16_384];
                    let mpeg_bytes = mpeg
                        .encode_interleaved_i16(&canonical, &mut mpeg_output)
                        .unwrap();
                    let mapped_bytes = mapped
                        .encode_interleaved_i16(&ordered, &mut mapped_output)
                        .unwrap();
                    assert_eq!(
                        &mpeg_output[..mpeg_bytes],
                        &mapped_output[..mapped_bytes],
                        "channels={channels}, order={order}"
                    );
                }
            }
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn c_fdk_accepts_pure_rust_aac_lc_multichannel_access_units() {
        for channels in 3..=6 {
            let mut parameters = configured(2, channels as u32, 48_000);
            parameters
                .set_parameter(EncoderParameter::Bitrate, 48_000 * channels as u32)
                .unwrap();
            parameters
                .set_parameter(EncoderParameter::TransportMux, 0)
                .unwrap();
            let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
            let mut raw = encoder
                .encode_interleaved_f32(&vec![0.0; 1024 * channels])
                .unwrap();
            let mut asc = backend_audio_specific_config(&encoder.config, &encoder.backend)
                .unwrap()
                .to_bytes()
                .unwrap();
            let mut decoder = crate::Decoder::open(crate::TransportType::Raw).unwrap();
            decoder.configure_raw(&mut asc).unwrap();
            assert_eq!(decoder.fill(&mut raw).unwrap(), raw.len());
            let mut output = vec![0i16; 1024 * channels];
            decoder.decode_frame(&mut output).unwrap();
            assert_eq!(decoder.stream_info().unwrap().channels, channels as i32);

            let mut transient = vec![0.0; 1024 * channels];
            for sample in 512..1024 {
                for channel in 0..channels.saturating_sub(1) {
                    transient[sample * channels + channel] =
                        ((sample - 512) as f32 * (0.04 + channel as f32 * 0.003)).sin() * 0.5;
                }
            }
            let silence = vec![0.0; 1024 * channels];
            for input in [&transient, &silence] {
                let mut raw = encoder.encode_interleaved_f32(input).unwrap();
                decoder.fill(&mut raw).unwrap();
                decoder.decode_frame(&mut output).unwrap();
            }
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn c_fdk_accepts_pure_rust_multichannel_adif_pce() {
        for (mode, channels) in [
            (3, 3usize),
            (4, 4),
            (5, 5),
            (6, 6),
            (7, 8),
            (11, 7),
            (12, 8),
            (14, 8),
            (33, 8),
            (34, 8),
        ] {
            let mut parameters = configured(2, mode, 48_000);
            parameters
                .set_parameter(EncoderParameter::Bitrate, 48_000 * channels as u32)
                .unwrap();
            parameters
                .set_parameter(EncoderParameter::TransportMux, 1)
                .unwrap();
            let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
            let mut adif = encoder
                .encode_transport_f32(&vec![0.0; 1024 * channels])
                .unwrap();
            let mut decoder = crate::Decoder::open(crate::TransportType::Adif).unwrap();
            assert_eq!(decoder.fill(&mut adif).unwrap(), adif.len());
            decoder
                .decode_frame(&mut vec![0i16; 1024 * channels])
                .unwrap();
            assert_eq!(
                decoder.stream_info().unwrap().channels,
                channels.min(6) as i32
            );
        }
    }

    #[test]
    fn unified_factory_writes_eldv2_asc_and_spatial_access_unit() {
        let mut parameters = configured(39, 128, 48_000);
        parameters
            .set_parameter(EncoderParameter::Bitrate, 64_000)
            .unwrap();
        parameters
            .set_parameter(EncoderParameter::TransportMux, 0)
            .unwrap();
        let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
        let asc = backend_audio_specific_config(&encoder.config, &encoder.backend).unwrap();
        let eld = asc.eld_specific.as_ref().unwrap();
        assert_eq!(asc.channel_configuration, 1);
        assert_eq!(eld.extensions.len(), 1);
        assert_eq!(eld.extensions[0].extension_type, 2);
        let spatial = crate::sac::SpatialSpecificConfig::parse(&eld.extensions[0].data).unwrap();
        assert_eq!(spatial.sampling_frequency, 48_000);
        assert_eq!(spatial.time_slots, 8);
        assert_eq!(spatial.frequency_resolution, 15);

        let input = (0..512)
            .flat_map(|sample| {
                let phase = sample as f32 * 0.09;
                [phase.sin() * 0.8, (phase + 0.7).sin() * 0.25]
            })
            .collect::<Vec<_>>();
        let raw = encoder.encode_interleaved_f32(&input).unwrap();
        assert!(!raw.is_empty());
        let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
        let decoded = decoder
            .decode_raw_data_block_multichannel_f32(&raw)
            .unwrap();
        assert_eq!(decoded.channels.len(), 2);
        assert!(decoded.channels.iter().all(|channel| channel.len() == 512));
        assert!(decoded
            .channels
            .iter()
            .flatten()
            .all(|sample| sample.is_finite()));
        let mut fixed_decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
        let fixed = fixed_decoder
            .decode_raw_data_block_fixed_interleaved_i16(&raw)
            .unwrap();
        assert_eq!(fixed.len(), 1024);

        let dependent_raw = encoder.encode_interleaved_f32(&input).unwrap();
        let dependent = decoder
            .decode_raw_data_block_multichannel_f32(&dependent_raw)
            .unwrap();
        assert_eq!(dependent.channels.len(), 2);
        assert!(dependent
            .channels
            .iter()
            .flatten()
            .any(|sample| *sample != 0.0));
    }

    #[test]
    fn unified_factory_writes_non_sbr_eld_multichannel_access_units() {
        for (mode, channels, channel_configuration) in [
            (3, 3usize, 3),
            (4, 4, 4),
            (5, 5, 5),
            (6, 6, 6),
            (7, 8, 7),
            (11, 7, 11),
            (12, 8, 12),
            (14, 8, 14),
        ] {
            let mut parameters = configured(39, mode, 48_000);
            parameters
                .set_parameter(EncoderParameter::Bitrate, 48_000 * channels as u32)
                .unwrap();
            parameters
                .set_parameter(EncoderParameter::SbrMode, 0)
                .unwrap();
            parameters
                .set_parameter(EncoderParameter::TransportMux, 0)
                .unwrap();
            let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
            let asc = backend_audio_specific_config(&encoder.config, &encoder.backend).unwrap();
            assert_eq!(asc.channel_configuration, channel_configuration);
            let input = (0..512)
                .flat_map(|sample| {
                    (0..channels).map(move |channel| {
                        (sample as f32 * (0.031 + channel as f32 * 0.004)).sin()
                            * (0.7 / (channel + 1) as f32)
                    })
                })
                .collect::<Vec<_>>();
            let raw = encoder.encode_interleaved_f32(&input).unwrap();
            let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
            let decoded = decoder
                .decode_raw_data_block_multichannel_f32(&raw)
                .unwrap();
            assert_eq!(decoded.channels.len(), channels);
            assert!(decoded.channels.iter().all(|channel| channel.len() == 512));
            assert!(decoded
                .channels
                .iter()
                .flatten()
                .all(|value| value.is_finite()));
            #[cfg(feature = "ffi")]
            {
                let mut asc_bytes = asc.to_bytes().unwrap();
                let mut c_raw = raw.clone();
                let mut c_decoder = crate::Decoder::open(crate::TransportType::Raw).unwrap();
                c_decoder.configure_raw(&mut asc_bytes).unwrap();
                c_decoder.fill(&mut c_raw).unwrap();
                c_decoder
                    .decode_frame(&mut vec![0; 1024 * channels])
                    .unwrap_or_else(|error| {
                        panic!("C decoder rejected ELD channel mode {mode}: {error:?}")
                    });
            }
        }
    }

    #[test]
    fn unified_factory_writes_dual_rate_eld_sbr_multichannel_access_unit() {
        for (mode, channels, bitrate, header_count) in
            [(3, 3usize, 120_000, 2usize), (14, 8, 256_000, 4)]
        {
            let mut parameters = configured(39, mode, 48_000);
            parameters
                .set_parameter(EncoderParameter::Bitrate, bitrate)
                .unwrap();
            parameters
                .set_parameter(EncoderParameter::SbrMode, 1)
                .unwrap();
            parameters
                .set_parameter(EncoderParameter::SbrRatio, 2)
                .unwrap();
            parameters
                .set_parameter(EncoderParameter::TransportMux, 0)
                .unwrap();
            let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
            assert_eq!(encoder.input_samples_per_channel(), 1024);
            let asc = backend_audio_specific_config(&encoder.config, &encoder.backend).unwrap();
            let eld = asc.eld_specific.as_ref().unwrap();
            assert!(eld.sbr_present);
            assert!(eld.sbr_sampling_rate);
            assert_eq!(eld.sbr_headers.len(), header_count);
            let input = (0..1024)
                .flat_map(|sample| {
                    (0..channels).map(move |channel| {
                        (sample as f32 * (0.027 + channel as f32 * 0.004)).sin()
                            * (0.7 / (channel + 1) as f32)
                    })
                })
                .collect::<Vec<_>>();
            let raw = encoder.encode_interleaved_f32(&input).unwrap();
            let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
            let decoded = decoder
                .decode_raw_data_block_multichannel_f32(&raw)
                .unwrap();
            assert_eq!(decoded.channels.len(), channels);
            assert!(
                decoded.channels.iter().all(|channel| channel.len() == 1024),
                "ELD-SBR mode {mode} lengths {:?}",
                decoded.channels.iter().map(Vec::len).collect::<Vec<_>>()
            );
            let mut fixed_decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
            let fixed = fixed_decoder
                .decode_raw_data_block_fixed_interleaved_i16(&raw)
                .unwrap();
            assert_eq!(fixed.len(), 1024 * channels);
        }
    }

    #[test]
    fn unified_factory_combines_single_rate_eld_sbr_and_mps() {
        let mut parameters = configured(39, 128, 24_000);
        parameters
            .set_parameter(EncoderParameter::Bitrate, 32_000)
            .unwrap();
        parameters
            .set_parameter(EncoderParameter::SbrMode, 1)
            .unwrap();
        parameters
            .set_parameter(EncoderParameter::SbrRatio, 1)
            .unwrap();
        parameters
            .set_parameter(EncoderParameter::TransportMux, 0)
            .unwrap();
        let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
        let asc = backend_audio_specific_config(&encoder.config, &encoder.backend).unwrap();
        let eld = asc.eld_specific.as_ref().unwrap();
        assert!(eld.sbr_present);
        assert!(!eld.sbr_sampling_rate);
        assert_eq!(eld.sbr_headers.len(), 1);
        assert_eq!(eld.extensions.len(), 1);
        assert_eq!(eld.extensions[0].extension_type, 2);

        let input = (0..512)
            .flat_map(|sample| {
                let phase = sample as f32 * 0.07;
                [phase.sin() * 0.7, (phase + 0.8).sin() * 0.3]
            })
            .collect::<Vec<_>>();
        let raw = encoder.encode_interleaved_f32(&input).unwrap();
        assert!(!raw.is_empty());
        let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
        let decoded = decoder
            .decode_raw_data_block_multichannel_f32(&raw)
            .unwrap();
        assert_eq!(decoded.channels.len(), 2);
        assert!(decoded.channels.iter().all(|channel| channel.len() == 512));
        assert!(decoded
            .channels
            .iter()
            .flatten()
            .all(|value| value.is_finite()));
    }

    #[test]
    fn unified_factory_combines_dual_rate_eld_sbr_and_mps() {
        let mut parameters = configured(39, 128, 48_000);
        parameters
            .set_parameter(EncoderParameter::Bitrate, 48_000)
            .unwrap();
        parameters
            .set_parameter(EncoderParameter::SbrMode, 1)
            .unwrap();
        parameters
            .set_parameter(EncoderParameter::SbrRatio, 2)
            .unwrap();
        parameters
            .set_parameter(EncoderParameter::TransportMux, 0)
            .unwrap();
        let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
        assert_eq!(encoder.input_samples_per_channel(), 1024);
        let asc = backend_audio_specific_config(&encoder.config, &encoder.backend).unwrap();
        let eld = asc.eld_specific.as_ref().unwrap();
        assert!(eld.sbr_present);
        assert!(eld.sbr_sampling_rate);
        let spatial = crate::sac::SpatialSpecificConfig::parse(&eld.extensions[0].data).unwrap();
        assert_eq!(spatial.sampling_frequency, 48_000);
        assert_eq!(spatial.time_slots, 16);

        let input = (0..1024)
            .flat_map(|sample| {
                let phase = sample as f32 * 0.05;
                [phase.sin() * 0.7, (phase + 0.6).sin() * 0.3]
            })
            .collect::<Vec<_>>();
        let raw = encoder.encode_interleaved_f32(&input).unwrap();
        let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
        let decoded = decoder
            .decode_raw_data_block_multichannel_f32(&raw)
            .unwrap();
        assert_eq!(decoded.channels.len(), 2);
        assert!(decoded.channels.iter().all(|channel| channel.len() == 1024));
        assert!(decoded
            .channels
            .iter()
            .flatten()
            .all(|value| value.is_finite()));
    }

    #[test]
    fn unified_factory_encodes_auto_selected_dual_rate_eld_sbr_mono() {
        let mut parameters = configured(39, 1, 44_100);
        parameters
            .set_parameter(EncoderParameter::Bitrate, 32_000)
            .unwrap();
        parameters
            .set_parameter(EncoderParameter::SbrMode, 0xff)
            .unwrap();
        parameters
            .set_parameter(EncoderParameter::TransportMux, 0)
            .unwrap();
        let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
        assert!(encoder.config().sbr_active);
        assert_eq!(encoder.config().sbr_ratio, 2);
        assert_eq!(encoder.input_samples_per_channel(), 1024);
        let config = backend_audio_specific_config(&encoder.config, &encoder.backend).unwrap();
        assert!(config.eld_specific.as_ref().unwrap().sbr_present);
        let input = (0..1024)
            .map(|sample| {
                (2.0 * std::f32::consts::PI * 997.0 * sample as f32 / 44_100.0).sin() * 12_000.0
            })
            .collect::<Vec<_>>();
        let raw = encoder.encode_interleaved_f32(&input).unwrap();
        let mut decoder = AacLcDecoder::from_audio_specific_config(&config).unwrap();
        let pcm = decoder
            .decode_raw_data_block_fixed_interleaved_i16(&raw)
            .unwrap();
        assert_eq!(pcm.len(), 1024);

        let mut parameters = configured(39, 2, 48_000);
        parameters
            .set_parameter(EncoderParameter::Bitrate, 64_000)
            .unwrap();
        parameters
            .set_parameter(EncoderParameter::SbrMode, 0xff)
            .unwrap();
        parameters
            .set_parameter(EncoderParameter::TransportMux, 0)
            .unwrap();
        let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
        assert_eq!(encoder.config().sbr_ratio, 2);
        assert_eq!(encoder.input_samples_per_channel(), 1024);
        let config = backend_audio_specific_config(&encoder.config, &encoder.backend).unwrap();
        let input = vec![0.0; 2048];
        let raw = encoder.encode_interleaved_f32(&input).unwrap();
        let mut decoder = AacLcDecoder::from_audio_specific_config(&config).unwrap();
        let pcm = decoder
            .decode_raw_data_block_fixed_interleaved_i16(&raw)
            .unwrap();
        assert_eq!(pcm.len(), 2048);
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn fdk_decoder_accepts_configured_pure_rust_dual_rate_eld_sbr_mono() {
        use crate::{Decoder, TransportType};

        let mut parameters = configured(39, 1, 44_100);
        parameters
            .set_parameter(EncoderParameter::Bitrate, 32_000)
            .unwrap();
        parameters
            .set_parameter(EncoderParameter::SbrMode, 0xff)
            .unwrap();
        parameters
            .set_parameter(EncoderParameter::TransportMux, 0)
            .unwrap();
        let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
        let asc = backend_audio_specific_config(&encoder.config, &encoder.backend).unwrap();
        let mut asc = asc.to_bytes().unwrap();
        let mut decoder = Decoder::open(TransportType::Raw).unwrap();
        decoder.configure_raw(&mut asc).unwrap();
        let input = (0..encoder.input_samples_per_channel())
            .map(|sample| {
                (2.0 * std::f32::consts::PI * 997.0 * sample as f32 / 44_100.0).sin() * 12_000.0
            })
            .collect::<Vec<_>>();
        let raw = encoder.encode_interleaved_f32(&input).unwrap();
        let mut pcm = vec![0i16; 2048];
        assert_eq!(
            decoder.decode_access_unit_i16(&raw, &mut pcm).unwrap(),
            1024
        );
        let dependent = encoder.encode_interleaved_f32(&input).unwrap();
        assert_eq!(
            decoder
                .decode_access_unit_i16(&dependent, &mut pcm)
                .unwrap(),
            1024
        );
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn fdk_decoder_accepts_configured_pure_rust_eldv2_mps() {
        use crate::{sys, Decoder, Encoder, TransportType};

        let mut parameters = configured(39, 128, 48_000);
        parameters
            .set_parameter(EncoderParameter::Bitrate, 64_000)
            .unwrap();
        parameters
            .set_parameter(EncoderParameter::TransportMux, 0)
            .unwrap();
        let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
        let mut asc = backend_audio_specific_config(&encoder.config, &encoder.backend)
            .unwrap()
            .to_bytes()
            .unwrap();
        let mut decoder = Decoder::open(TransportType::Raw).unwrap();
        let mut c_encoder = Encoder::open(2).unwrap();
        c_encoder.set_param(sys::AACENC_AOT, 39).unwrap();
        c_encoder.set_param(sys::AACENC_CHANNELMODE, 128).unwrap();
        c_encoder.set_param(sys::AACENC_SAMPLERATE, 48_000).unwrap();
        c_encoder.set_param(sys::AACENC_BITRATE, 64_000).unwrap();
        c_encoder.set_param(sys::AACENC_TRANSMUX, 0).unwrap();
        c_encoder.initialize().unwrap();
        assert_eq!(asc, c_encoder.audio_specific_config().unwrap());
        decoder.configure_raw(&mut asc).unwrap();
        let input = (0..512)
            .flat_map(|sample| {
                let phase = sample as f32 * 0.083;
                [phase.sin() * 12_000.0, (phase + 0.5).sin() * 4_000.0]
            })
            .collect::<Vec<_>>();
        let raw = encoder.encode_interleaved_f32(&input).unwrap();
        let mut pcm = vec![0i16; 2048];
        assert_eq!(
            decoder.decode_access_unit_i16(&raw, &mut pcm).unwrap(),
            1024
        );

        let c_config =
            AudioSpecificConfig::parse(&c_encoder.audio_specific_config().unwrap()).unwrap();
        let c_info = c_encoder.info().unwrap();
        let c_input = input
            .iter()
            .map(|sample| sample.clamp(i16::MIN as f32, i16::MAX as f32) as i16)
            .collect::<Vec<_>>();
        let mut c_raw = vec![0u8; c_info.max_output_bytes as usize];
        let mut c_bytes = 0;
        for _ in 0..12 {
            c_bytes = c_encoder
                .encode_interleaved_i16(&c_input, &mut c_raw)
                .unwrap();
            if c_bytes != 0 {
                break;
            }
        }
        assert_ne!(c_bytes, 0);
        c_raw.truncate(c_bytes);
        let mut rust_decoder = AacLcDecoder::from_audio_specific_config(&c_config).unwrap();
        let c_frame = rust_decoder
            .decode_raw_data_block_multichannel_f32(&c_raw)
            .unwrap();
        assert_eq!(c_frame.channels.len(), 2);
        assert!(c_frame.channels.iter().all(|channel| channel.len() == 512));
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn fdk_decoder_accepts_pure_rust_single_rate_eld_sbr_mps() {
        use crate::{sys, Decoder, Encoder, TransportType};

        let mut parameters = configured(39, 128, 24_000);
        parameters
            .set_parameter(EncoderParameter::Bitrate, 32_000)
            .unwrap();
        parameters
            .set_parameter(EncoderParameter::SbrMode, 1)
            .unwrap();
        parameters
            .set_parameter(EncoderParameter::SbrRatio, 1)
            .unwrap();
        parameters
            .set_parameter(EncoderParameter::TransportMux, 0)
            .unwrap();
        let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
        let mut asc = backend_audio_specific_config(&encoder.config, &encoder.backend)
            .unwrap()
            .to_bytes()
            .unwrap();
        let mut c_encoder = Encoder::open(2).unwrap();
        c_encoder.set_param(sys::AACENC_AOT, 39).unwrap();
        c_encoder.set_param(sys::AACENC_CHANNELMODE, 128).unwrap();
        c_encoder.set_param(sys::AACENC_SAMPLERATE, 24_000).unwrap();
        c_encoder.set_param(sys::AACENC_BITRATE, 32_000).unwrap();
        c_encoder.set_param(sys::AACENC_SBR_MODE, 1).unwrap();
        c_encoder.set_param(sys::AACENC_SBR_RATIO, 1).unwrap();
        c_encoder.set_param(sys::AACENC_TRANSMUX, 0).unwrap();
        c_encoder.initialize().unwrap();
        assert_eq!(asc, c_encoder.audio_specific_config().unwrap());
        let mut decoder = Decoder::open(TransportType::Raw).unwrap();
        decoder.configure_raw(&mut asc).unwrap();
        let input = (0..512)
            .flat_map(|sample| {
                let phase = sample as f32 * 0.07;
                [phase.sin() * 12_000.0, (phase + 0.8).sin() * 5_000.0]
            })
            .collect::<Vec<_>>();
        let raw = encoder.encode_interleaved_f32(&input).unwrap();
        let mut pcm = vec![0i16; 2048];
        assert_eq!(
            decoder.decode_access_unit_i16(&raw, &mut pcm).unwrap(),
            1024
        );
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn fdk_decoder_accepts_pure_rust_dual_rate_eld_sbr_mps() {
        use crate::{sys, Decoder, Encoder, TransportType};

        let mut parameters = configured(39, 128, 48_000);
        parameters
            .set_parameter(EncoderParameter::Bitrate, 48_000)
            .unwrap();
        parameters
            .set_parameter(EncoderParameter::SbrMode, 1)
            .unwrap();
        parameters
            .set_parameter(EncoderParameter::SbrRatio, 2)
            .unwrap();
        parameters
            .set_parameter(EncoderParameter::TransportMux, 0)
            .unwrap();
        let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
        let mut asc = backend_audio_specific_config(&encoder.config, &encoder.backend)
            .unwrap()
            .to_bytes()
            .unwrap();
        let mut c_encoder = Encoder::open(2).unwrap();
        c_encoder.set_param(sys::AACENC_AOT, 39).unwrap();
        c_encoder.set_param(sys::AACENC_CHANNELMODE, 128).unwrap();
        c_encoder.set_param(sys::AACENC_SAMPLERATE, 48_000).unwrap();
        c_encoder.set_param(sys::AACENC_BITRATE, 48_000).unwrap();
        c_encoder.set_param(sys::AACENC_SBR_MODE, 1).unwrap();
        c_encoder.set_param(sys::AACENC_SBR_RATIO, 2).unwrap();
        c_encoder.set_param(sys::AACENC_TRANSMUX, 0).unwrap();
        c_encoder.initialize().unwrap();
        let c_asc = c_encoder.audio_specific_config().unwrap();
        assert_eq!(asc, c_asc);
        let mut decoder = Decoder::open(TransportType::Raw).unwrap();
        decoder.configure_raw(&mut asc).unwrap();
        let input = (0..1024)
            .flat_map(|sample| {
                let phase = sample as f32 * 0.05;
                [phase.sin() * 12_000.0, (phase + 0.6).sin() * 5_000.0]
            })
            .collect::<Vec<_>>();
        let raw = encoder.encode_interleaved_f32(&input).unwrap();
        let mut pcm = vec![0i16; 4096];
        assert_eq!(
            decoder.decode_access_unit_i16(&raw, &mut pcm).unwrap(),
            2048
        );
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn fdk_accepts_pure_rust_non_sbr_eld_multichannel() {
        use crate::{sys, Decoder, Encoder, TransportType};

        for channels in 3..=6 {
            let bitrate = 48_000 * channels as u32;
            let mut parameters = configured(39, channels as u32, 48_000);
            parameters
                .set_parameter(EncoderParameter::Bitrate, bitrate)
                .unwrap();
            parameters
                .set_parameter(EncoderParameter::SbrMode, 0)
                .unwrap();
            parameters
                .set_parameter(EncoderParameter::TransportMux, 0)
                .unwrap();
            let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
            let mut asc = backend_audio_specific_config(&encoder.config, &encoder.backend)
                .unwrap()
                .to_bytes()
                .unwrap();

            let mut c_encoder = Encoder::open(channels as u32).unwrap();
            c_encoder.set_param(sys::AACENC_AOT, 39).unwrap();
            c_encoder
                .set_param(sys::AACENC_CHANNELMODE, channels as u32)
                .unwrap();
            c_encoder.set_param(sys::AACENC_SAMPLERATE, 48_000).unwrap();
            c_encoder.set_param(sys::AACENC_BITRATE, bitrate).unwrap();
            c_encoder.set_param(sys::AACENC_SBR_MODE, 0).unwrap();
            c_encoder.set_param(sys::AACENC_TRANSMUX, 0).unwrap();
            c_encoder.initialize().unwrap();
            assert_eq!(asc, c_encoder.audio_specific_config().unwrap());

            let input = (0..512)
                .flat_map(|sample| {
                    (0..channels).map(move |channel| {
                        (sample as f32 * (0.031 + channel as f32 * 0.004)).sin() * 12_000.0
                            / (channel + 1) as f32
                    })
                })
                .collect::<Vec<_>>();
            let raw = encoder.encode_interleaved_f32(&input).unwrap();
            let mut decoder = Decoder::open(TransportType::Raw).unwrap();
            decoder.configure_raw(&mut asc).unwrap();
            let mut pcm = vec![0i16; 512 * channels];
            assert_eq!(
                decoder.decode_access_unit_i16(&raw, &mut pcm).unwrap(),
                512 * channels
            );
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn fdk_accepts_pure_rust_dual_rate_eld_sbr_three_channel() {
        use crate::{sys, Decoder, Encoder, TransportType};

        let channels = 3usize;
        let mut parameters = configured(39, channels as u32, 48_000);
        parameters
            .set_parameter(EncoderParameter::Bitrate, 120_000)
            .unwrap();
        parameters
            .set_parameter(EncoderParameter::SbrMode, 1)
            .unwrap();
        parameters
            .set_parameter(EncoderParameter::SbrRatio, 2)
            .unwrap();
        parameters
            .set_parameter(EncoderParameter::TransportMux, 0)
            .unwrap();
        let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
        let mut asc = backend_audio_specific_config(&encoder.config, &encoder.backend)
            .unwrap()
            .to_bytes()
            .unwrap();

        let mut c_encoder = Encoder::open(channels as u32).unwrap();
        c_encoder.set_param(sys::AACENC_AOT, 39).unwrap();
        c_encoder
            .set_param(sys::AACENC_CHANNELMODE, channels as u32)
            .unwrap();
        c_encoder.set_param(sys::AACENC_SAMPLERATE, 48_000).unwrap();
        c_encoder.set_param(sys::AACENC_BITRATE, 120_000).unwrap();
        c_encoder.set_param(sys::AACENC_SBR_MODE, 1).unwrap();
        c_encoder.set_param(sys::AACENC_SBR_RATIO, 2).unwrap();
        c_encoder.set_param(sys::AACENC_TRANSMUX, 0).unwrap();
        c_encoder.initialize().unwrap();
        let c_asc_bytes = c_encoder.audio_specific_config().unwrap();
        assert_eq!(asc, c_asc_bytes);
        let c_asc = AudioSpecificConfig::parse(&c_asc_bytes).unwrap();
        let rust_asc = AudioSpecificConfig::parse(&asc).unwrap();
        assert_eq!(rust_asc.channel_configuration, c_asc.channel_configuration);
        assert_eq!(
            rust_asc.eld_specific.as_ref().unwrap().sbr_headers.len(),
            c_asc.eld_specific.as_ref().unwrap().sbr_headers.len()
        );

        let input = (0..1024)
            .flat_map(|sample| {
                (0..channels).map(move |channel| {
                    (sample as f32 * (0.027 + channel as f32 * 0.004)).sin() * 12_000.0
                        / (channel + 1) as f32
                })
            })
            .collect::<Vec<_>>();
        let raw = encoder.encode_interleaved_f32(&input).unwrap();
        let mut decoder = Decoder::open(TransportType::Raw).unwrap();
        decoder.configure_raw(&mut asc).unwrap();
        let mut pcm = vec![0i16; 1024 * channels];
        assert_eq!(
            decoder.decode_access_unit_i16(&raw, &mut pcm).unwrap(),
            1024 * channels
        );
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn fdk_accepts_pure_rust_dual_rate_eld_sbr_four_to_six_channel() {
        use crate::{sys, Decoder, Encoder, TransportType};

        for channels in 4..=6usize {
            let bitrate = 40_000 * channels as u32;
            let mut parameters = configured(39, channels as u32, 48_000);
            parameters
                .set_parameter(EncoderParameter::Bitrate, bitrate)
                .unwrap();
            parameters
                .set_parameter(EncoderParameter::SbrMode, 1)
                .unwrap();
            parameters
                .set_parameter(EncoderParameter::SbrRatio, 2)
                .unwrap();
            parameters
                .set_parameter(EncoderParameter::TransportMux, 0)
                .unwrap();
            let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
            let mut asc = backend_audio_specific_config(&encoder.config, &encoder.backend)
                .unwrap()
                .to_bytes()
                .unwrap();

            let mut c_encoder = Encoder::open(channels as u32).unwrap();
            c_encoder.set_param(sys::AACENC_AOT, 39).unwrap();
            c_encoder
                .set_param(sys::AACENC_CHANNELMODE, channels as u32)
                .unwrap();
            c_encoder.set_param(sys::AACENC_SAMPLERATE, 48_000).unwrap();
            c_encoder.set_param(sys::AACENC_BITRATE, bitrate).unwrap();
            c_encoder.set_param(sys::AACENC_SBR_MODE, 1).unwrap();
            c_encoder.set_param(sys::AACENC_SBR_RATIO, 2).unwrap();
            c_encoder.set_param(sys::AACENC_TRANSMUX, 0).unwrap();
            c_encoder.initialize().unwrap();
            assert_eq!(asc, c_encoder.audio_specific_config().unwrap());

            let input = (0..1024)
                .flat_map(|sample| {
                    (0..channels).map(move |channel| {
                        (sample as f32 * (0.023 + channel as f32 * 0.003)).sin() * 12_000.0
                            / (channel + 1) as f32
                    })
                })
                .collect::<Vec<_>>();
            let raw = encoder.encode_interleaved_f32(&input).unwrap();
            let mut decoder = Decoder::open(TransportType::Raw).unwrap();
            decoder.configure_raw(&mut asc).unwrap();
            let mut pcm = vec![0i16; 1024 * channels];
            assert_eq!(
                decoder.decode_access_unit_i16(&raw, &mut pcm).unwrap(),
                1024 * channels
            );
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn fdk_accepts_pure_rust_single_rate_eld_sbr_multichannel() {
        use crate::{sys, Decoder, Encoder, TransportType};
        for channels in 3..=6usize {
            let bitrate = 30_000 * channels as u32;
            let mut parameters = configured(39, channels as u32, 24_000);
            parameters
                .set_parameter(EncoderParameter::Bitrate, bitrate)
                .unwrap();
            parameters
                .set_parameter(EncoderParameter::SbrMode, 1)
                .unwrap();
            parameters
                .set_parameter(EncoderParameter::SbrRatio, 1)
                .unwrap();
            parameters
                .set_parameter(EncoderParameter::TransportMux, 0)
                .unwrap();
            let mut rust = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
            let mut asc = backend_audio_specific_config(&rust.config, &rust.backend)
                .unwrap()
                .to_bytes()
                .unwrap();

            let mut c = Encoder::open(channels as u32).unwrap();
            c.set_param(sys::AACENC_AOT, 39).unwrap();
            c.set_param(sys::AACENC_CHANNELMODE, channels as u32)
                .unwrap();
            c.set_param(sys::AACENC_SAMPLERATE, 24_000).unwrap();
            c.set_param(sys::AACENC_BITRATE, bitrate).unwrap();
            c.set_param(sys::AACENC_SBR_MODE, 1).unwrap();
            c.set_param(sys::AACENC_SBR_RATIO, 1).unwrap();
            c.set_param(sys::AACENC_TRANSMUX, 0).unwrap();
            c.initialize().unwrap();
            assert_eq!(asc, c.audio_specific_config().unwrap());

            let input = (0..512)
                .flat_map(|sample| {
                    (0..channels).map(move |channel| {
                        (sample as f32 * (0.029 + channel as f32 * 0.004)).sin() * 12_000.0
                            / (channel + 1) as f32
                    })
                })
                .collect::<Vec<_>>();
            let raw = rust.encode_interleaved_f32(&input).unwrap();
            let mut decoder = Decoder::open(TransportType::Raw).unwrap();
            decoder.configure_raw(&mut asc).unwrap();
            let mut pcm = vec![0i16; 512 * channels];
            assert_eq!(
                decoder.decode_access_unit_i16(&raw, &mut pcm).unwrap(),
                512 * channels
            );
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn eld_mps_high_rate_rejection_matches_bundled_c_encoder() {
        use crate::{sys, Encoder};

        // mps_main.cpp mentions a 128-QMF geometry, but the SAC library used
        // by this build returns zero QMF bands at and above 55426 Hz.  Treat
        // the executable C behavior, rather than that stale comment, as the
        // parity target.
        assert!(PureRustAacEldMpsEncoder::new(0, 512, 1_024, 4_000).is_err());

        let mut c = Encoder::open(2).unwrap();
        c.set_param(sys::AACENC_AOT, 39).unwrap();
        c.set_param(sys::AACENC_CHANNELMODE, 128).unwrap();
        c.set_param(sys::AACENC_SAMPLERATE, 96_000).unwrap();
        c.set_param(sys::AACENC_BITRATE, 128_000).unwrap();
        c.set_param(sys::AACENC_TRANSMUX, 0).unwrap();
        assert!(c.initialize().is_err());
    }

    #[test]
    fn unified_factory_embeds_and_reports_ga_ancillary_data() {
        let mut parameters = configured(2, 1, 48_000);
        parameters
            .set_parameter(EncoderParameter::Bitrate, 320_000)
            .unwrap();
        parameters
            .set_parameter(EncoderParameter::TransportMux, 0)
            .unwrap();
        let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
        assert_eq!(encoder.max_ancillary_bytes_per_access_unit(), 256);

        let oversized = (0..300).map(|value| value as u8).collect::<Vec<_>>();
        let (raw_without_ancillary, consumed) = encoder
            .encode_interleaved_f32_with_ancillary(&vec![0.0; 1024], &oversized)
            .unwrap();
        assert_eq!(consumed, 0);
        let mut no_ancillary_decoder = AacLcDecoder::new(3, 1).unwrap();
        no_ancillary_decoder.init_ancillary_data(300);
        no_ancillary_decoder
            .decode_raw_data_block_f32(&raw_without_ancillary)
            .unwrap();
        assert!(no_ancillary_decoder.ancillary_data().is_empty());

        let ancillary = oversized[..256].to_vec();
        let (raw, consumed) = encoder
            .encode_interleaved_f32_with_ancillary(&vec![0.0; 1024], &ancillary)
            .unwrap();
        assert_eq!(consumed, 256);

        let mut decoder = AacLcDecoder::new(3, 1).unwrap();
        decoder.init_ancillary_data(300);
        decoder.decode_raw_data_block_f32(&raw).unwrap();
        assert_eq!(decoder.ancillary_data().len(), 1);
        assert_eq!(decoder.ancillary_data()[0].element_instance_tag, 0);
        assert_eq!(decoder.ancillary_data()[0].data, ancillary);
    }

    #[test]
    fn fixed_ancillary_bitrate_limits_each_protected_adts_access_unit() {
        let mut parameters = configured(2, 1, 48_000);
        parameters
            .set_parameter(EncoderParameter::AncillaryBitrate, 8_000)
            .unwrap();
        parameters
            .set_parameter(EncoderParameter::Protection, 1)
            .unwrap();
        let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
        assert_eq!(encoder.max_ancillary_bytes_per_access_unit(), 21);
        let ancillary = vec![0xa5; 40];
        let (adts, consumed) = encoder
            .encode_transport_f32_with_ancillary(&vec![0.0; 1024], &ancillary)
            .unwrap();
        assert_eq!(consumed, 21);

        let mut decoder = AacLcDecoder::new(3, 1).unwrap();
        decoder.init_ancillary_data(40);
        decoder.decode_adts_frame_f32(&adts).unwrap();
        assert_eq!(decoder.ancillary_data()[0].data, ancillary[..21]);
    }

    #[test]
    fn unified_factory_embeds_er_ancillary_extensions_for_ld_and_eld() {
        let ancillary = [0x12, 0x34, 0x56];
        for (aot, channels, frame_length) in
            [(23, 1, 512), (23, 2, 512), (39, 1, 512), (39, 2, 512)]
        {
            let mut parameters = configured(aot, channels, 48_000);
            parameters
                .set_parameter(EncoderParameter::TransportMux, 0)
                .unwrap();
            let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
            let asc = backend_audio_specific_config(encoder.config(), &encoder.backend).unwrap();
            let (raw, consumed) = encoder
                .encode_interleaved_f32_with_ancillary(
                    &vec![0.0; frame_length * channels as usize],
                    &ancillary,
                )
                .unwrap();
            assert_eq!(consumed, ancillary.len());

            let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
            decoder.init_ancillary_data(ancillary.len());
            decoder.decode_raw_data_block_f32(&raw).unwrap();
            assert_eq!(decoder.ancillary_data().len(), 1);
            assert_eq!(decoder.ancillary_data()[0].data, ancillary);
        }
    }

    #[test]
    fn unified_factory_emits_each_supported_transport_and_collects_subframes() {
        use crate::adif::AdifHeader;
        use crate::adts::AdtsFrame;
        use crate::latm::LatmAudioMuxElement;
        use crate::loas::LoasFrame;

        let silence = vec![0.0; 1024];

        let mut adts_parameters = configured(2, 1, 48_000);
        adts_parameters
            .set_parameter(EncoderParameter::TransportSubframes, 2)
            .unwrap();
        let mut adts = ConfiguredPureRustEncoder::from_parameters(&adts_parameters).unwrap();
        assert!(adts.encode_transport_f32(&silence).unwrap().is_empty());
        let framed = adts.encode_transport_f32(&silence).unwrap();
        let parsed = AdtsFrame::parse(&framed).unwrap();
        assert_eq!(parsed.header.number_of_raw_data_blocks_in_frame, 1);

        let mut raw_parameters = configured(2, 1, 48_000);
        raw_parameters
            .set_parameter(EncoderParameter::TransportMux, 0)
            .unwrap();
        let mut raw = ConfiguredPureRustEncoder::from_parameters(&raw_parameters).unwrap();
        assert!(!raw.encode_transport_f32(&silence).unwrap().is_empty());

        let mut adif_parameters = configured(2, 1, 48_000);
        adif_parameters
            .set_parameter(EncoderParameter::TransportMux, 1)
            .unwrap();
        let mut adif = ConfiguredPureRustEncoder::from_parameters(&adif_parameters).unwrap();
        let first = adif.encode_transport_f32(&silence).unwrap();
        let header = AdifHeader::parse(&first).unwrap();
        assert_eq!(header.last_program_config().unwrap().num_channels, 1);
        let second = adif.encode_transport_f32(&silence).unwrap();
        assert!(!second.starts_with(b"ADIF"));

        for channels in 3..=6 {
            let mut parameters = configured(2, channels, 48_000);
            parameters
                .set_parameter(EncoderParameter::Bitrate, 48_000 * channels)
                .unwrap();
            parameters
                .set_parameter(EncoderParameter::TransportMux, 1)
                .unwrap();
            let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
            let first = encoder
                .encode_transport_f32(&vec![0.0; 1024 * channels as usize])
                .unwrap();
            let header = AdifHeader::parse(&first).unwrap();
            let pce = header.last_program_config().unwrap();
            assert_eq!(pce.num_channels, channels as u8);
            assert_eq!(pce.front.len(), 2);
            assert_eq!(pce.back.len(), usize::from(channels >= 4));
            assert_eq!(pce.lfe.len(), usize::from(channels == 6));
        }

        let mut latm_parameters = configured(2, 1, 48_000);
        latm_parameters
            .set_parameter(EncoderParameter::TransportMux, 6)
            .unwrap();
        let mut latm = ConfiguredPureRustEncoder::from_parameters(&latm_parameters).unwrap();
        let element = latm.encode_transport_f32(&silence).unwrap();
        assert_eq!(
            LatmAudioMuxElement::parse_aac_lc(&element)
                .unwrap()
                .config
                .unwrap()
                .audio_object_type,
            2
        );

        for channels in 3..=6u32 {
            let mut parameters = configured(2, channels, 48_000);
            parameters
                .set_parameter(EncoderParameter::TransportMux, 6)
                .unwrap();
            let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
            let element = encoder
                .encode_transport_f32(&vec![0.0; 1024 * channels as usize])
                .unwrap();
            let mut decoder =
                crate::transport::PureRustTransportDecoder::from_latm_audio_mux_element(&element)
                    .unwrap();
            assert_eq!(
                decoder.decode_latm_interleaved_f32(&element).unwrap().len(),
                1024 * channels as usize
            );
            let mut fixed_decoder =
                crate::transport::PureRustTransportDecoder::from_latm_audio_mux_element(&element)
                    .unwrap();
            assert_eq!(
                fixed_decoder
                    .decode_latm_interleaved_i16(&element)
                    .unwrap()
                    .len(),
                1024 * channels as usize
            );
        }

        let ld_parameters = configured(23, 1, 48_000);
        let mut loas = ConfiguredPureRustEncoder::from_parameters(&ld_parameters).unwrap();
        let loas_bytes = loas.encode_transport_f32(&vec![0.0; 512]).unwrap();
        let loas_frame = LoasFrame::parse(&loas_bytes).unwrap();
        assert_eq!(
            LatmAudioMuxElement::parse_aac_lc(loas_frame.audio_mux_element)
                .unwrap()
                .config
                .unwrap()
                .audio_object_type,
            23
        );
    }

    #[test]
    fn unified_factory_writes_valid_protected_single_and_multi_block_adts() {
        use crate::adts::AdtsFrame;
        use crate::decoder::AacLcDecoder;

        let silence = vec![0.0; 1024];
        for subframes in [1, 2, 4] {
            let mut parameters = configured(2, 1, 48_000);
            parameters
                .set_parameter(EncoderParameter::Protection, 1)
                .unwrap();
            parameters
                .set_parameter(EncoderParameter::TransportSubframes, subframes)
                .unwrap();
            let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
            for _ in 1..subframes {
                assert!(encoder.encode_transport_f32(&silence).unwrap().is_empty());
            }
            let frame_bytes = encoder.encode_transport_f32(&silence).unwrap();
            let frame = AdtsFrame::parse(&frame_bytes).unwrap();
            assert!(!frame.header.protection_absent);
            assert_eq!(
                frame.header.number_of_raw_data_blocks_in_frame,
                (subframes - 1) as u8
            );
            if subframes > 1 {
                frame.validate_multi_block_header_crc().unwrap();
                assert_eq!(frame.raw_data_blocks().unwrap().len(), subframes as usize);
            }
            let mut decoder = AacLcDecoder::new(3, 1).unwrap();
            assert_eq!(
                decoder
                    .decode_adts_frame_blocks_f32(&frame_bytes)
                    .unwrap()
                    .len(),
                subframes as usize
            );
        }
    }

    #[test]
    fn initialization_rejects_adts_and_adif_for_low_delay_profiles() {
        for transport in [1, 2] {
            let mut parameters = configured(23, 1, 48_000);
            parameters
                .set_parameter(EncoderParameter::TransportMux, transport)
                .unwrap();
            assert_eq!(
                parameters.resolve(),
                Err(
                    EncoderConfigurationError::UnsupportedTransportForAudioObjectType {
                        transport_mux: transport,
                        audio_object_type: 23,
                    }
                )
            );
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn setter_acceptance_and_initialization_flags_match_c() {
        use crate::{sys, Encoder};

        fn raw(parameter: EncoderParameter) -> sys::AACENC_PARAM {
            match parameter {
                EncoderParameter::AudioObjectType => sys::AACENC_AOT,
                EncoderParameter::Bitrate => sys::AACENC_BITRATE,
                EncoderParameter::BitrateMode => sys::AACENC_BITRATEMODE,
                EncoderParameter::SampleRate => sys::AACENC_SAMPLERATE,
                EncoderParameter::SbrMode => sys::AACENC_SBR_MODE,
                EncoderParameter::GranuleLength => sys::AACENC_GRANULE_LENGTH,
                EncoderParameter::ChannelMode => sys::AACENC_CHANNELMODE,
                EncoderParameter::ChannelOrder => sys::AACENC_CHANNELORDER,
                EncoderParameter::SbrRatio => sys::AACENC_SBR_RATIO,
                EncoderParameter::Afterburner => sys::AACENC_AFTERBURNER,
                EncoderParameter::Bandwidth => sys::AACENC_BANDWIDTH,
                EncoderParameter::PeakBitrate => sys::AACENC_PEAK_BITRATE,
                EncoderParameter::TransportMux => sys::AACENC_TRANSMUX,
                EncoderParameter::HeaderPeriod => sys::AACENC_HEADER_PERIOD,
                EncoderParameter::SignalingMode => sys::AACENC_SIGNALING_MODE,
                EncoderParameter::TransportSubframes => sys::AACENC_TPSUBFRAMES,
                EncoderParameter::AudioMuxVersion => sys::AACENC_AUDIOMUXVER,
                EncoderParameter::Protection => sys::AACENC_PROTECTION,
                EncoderParameter::AncillaryBitrate => sys::AACENC_ANCILLARY_BITRATE,
                EncoderParameter::MetadataMode => sys::AACENC_METADATA_MODE,
                EncoderParameter::ControlState => sys::AACENC_CONTROL_STATE,
            }
        }

        let cases = [
            (EncoderParameter::AudioObjectType, 23),
            (EncoderParameter::AudioObjectType, 99),
            (EncoderParameter::Bitrate, 96_000),
            (EncoderParameter::BitrateMode, 5),
            (EncoderParameter::BitrateMode, 6),
            (EncoderParameter::SampleRate, 44_100),
            (EncoderParameter::SampleRate, 44_101),
            (EncoderParameter::SbrMode, 511),
            (EncoderParameter::GranuleLength, 120),
            (EncoderParameter::GranuleLength, 960),
            (EncoderParameter::ChannelMode, 6),
            (EncoderParameter::ChannelMode, 10),
            (EncoderParameter::ChannelOrder, 2),
            (EncoderParameter::ChannelOrder, 3),
            (EncoderParameter::SbrRatio, 2),
            (EncoderParameter::SbrRatio, 3),
            (EncoderParameter::Afterburner, 1),
            (EncoderParameter::Afterburner, 2),
            (EncoderParameter::Bandwidth, u32::MAX),
            (EncoderParameter::PeakBitrate, 80_000),
            (EncoderParameter::TransportMux, 10),
            (EncoderParameter::TransportMux, 12),
            (EncoderParameter::HeaderPeriod, 255),
            (EncoderParameter::HeaderPeriod, 256),
            (EncoderParameter::SignalingMode, 2),
            (EncoderParameter::SignalingMode, 3),
            (EncoderParameter::TransportSubframes, 4),
            (EncoderParameter::TransportSubframes, 0),
            (EncoderParameter::AudioMuxVersion, 2),
            (EncoderParameter::AudioMuxVersion, 3),
            (EncoderParameter::Protection, 1),
            (EncoderParameter::Protection, 2),
            (EncoderParameter::AncillaryBitrate, u32::MAX),
            (EncoderParameter::MetadataMode, 3),
            (EncoderParameter::MetadataMode, 4),
        ];

        for (parameter, value) in cases {
            let mut rust = PureRustEncoderParameters::new(8);
            rust.clear_initialization_flags();
            let rust_result = rust.set_parameter(parameter, value);

            let mut c = Encoder::open(8).unwrap();
            c.set_param(sys::AACENC_CONTROL_STATE, 0).unwrap();
            let c_result = c.set_param(raw(parameter), value);
            assert_eq!(
                rust_result.is_ok(),
                c_result.is_ok(),
                "setter acceptance differs for {parameter:?}={value}"
            );
            if rust_result.is_ok() {
                assert_eq!(
                    rust.initialization_flags(),
                    c.get_param(sys::AACENC_CONTROL_STATE),
                    "initialization flags differ for {parameter:?}={value}"
                );
            }
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn representative_resolved_configurations_match_initialized_c_encoder() {
        use crate::{sys, Encoder};

        let cases = [
            (2, 2, 48_000, 0, u32::MAX, 2),
            (5, 1, 48_000, 0, u32::MAX, 2),
            (29, 2, 48_000, 0, u32::MAX, 2),
            (23, 1, 48_000, 0, u32::MAX, 2),
            (39, 1, 48_000, 0, u32::MAX, 2),
            (39, 128, 48_000, 0, u32::MAX, 0),
            (2, 2, 48_000, 4, 90_000, 0),
            (23, 1, 48_000, 1, u32::MAX, 0),
            (23, 2, 48_000, 5, u32::MAX, 0),
        ];
        for (aot, channel_mode, sample_rate, bitrate_mode, peak_bitrate, metadata_mode) in cases {
            let mut rust = configured(aot, channel_mode, sample_rate);
            rust.set_parameter(EncoderParameter::BitrateMode, bitrate_mode)
                .unwrap();
            rust.set_parameter(EncoderParameter::MetadataMode, metadata_mode)
                .unwrap();
            if peak_bitrate != u32::MAX {
                rust.set_parameter(EncoderParameter::PeakBitrate, peak_bitrate)
                    .unwrap();
            }
            let resolved = rust.resolve().unwrap();

            let mut c = Encoder::open(8).unwrap();
            c.set_param(sys::AACENC_AOT, aot).unwrap();
            c.set_param(sys::AACENC_CHANNELMODE, channel_mode).unwrap();
            c.set_param(sys::AACENC_SAMPLERATE, sample_rate).unwrap();
            c.set_param(sys::AACENC_BITRATEMODE, bitrate_mode).unwrap();
            c.set_param(sys::AACENC_METADATA_MODE, metadata_mode)
                .unwrap();
            if peak_bitrate != u32::MAX {
                c.set_param(sys::AACENC_PEAK_BITRATE, peak_bitrate).unwrap();
            }
            c.initialize().unwrap();

            assert_eq!(c.get_param(sys::AACENC_AOT), resolved.audio_object_type);
            assert_eq!(c.get_param(sys::AACENC_SAMPLERATE), resolved.sample_rate);
            assert_eq!(
                c.get_param(sys::AACENC_CHANNELMODE),
                resolved.core_channel_mode
            );
            assert_eq!(
                c.get_param(sys::AACENC_GRANULE_LENGTH),
                resolved.frame_length
            );
            assert_eq!(c.get_param(sys::AACENC_BITRATEMODE), resolved.bitrate_mode);
            assert_eq!(
                c.get_param(sys::AACENC_SBR_MODE),
                u32::from(resolved.audio_object_type == 39 && resolved.sbr_active)
            );
            assert_eq!(c.get_param(sys::AACENC_SBR_RATIO), resolved.sbr_ratio);
            assert_eq!(c.get_param(sys::AACENC_TRANSMUX), resolved.transport_mux);
            assert_eq!(
                c.get_param(sys::AACENC_SIGNALING_MODE),
                resolved.signaling_mode
            );
            if resolved.bitrate_mode == 0 {
                assert_eq!(c.get_param(sys::AACENC_BITRATE), resolved.bitrate);
            }
            assert_eq!(c.get_param(sys::AACENC_BANDWIDTH), resolved.bandwidth);
            assert_eq!(
                c.get_param(sys::AACENC_ANCILLARY_BITRATE),
                resolved.ancillary_bitrate
            );
            assert_eq!(
                c.get_param(sys::AACENC_METADATA_MODE),
                resolved.metadata_mode
            );
            let info = c.info().unwrap();
            assert_eq!(info.delay, resolved.encoder_delay, "AOT {aot}");
            assert_eq!(
                info.core_delay, resolved.encoder_core_delay,
                "core delay for AOT {aot}"
            );
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn ga_ancillary_capacity_consumption_and_bitstreams_match_c_encoder() {
        use crate::{sys, Decoder, Encoder, TransportType};

        let ancillary = (0..256).map(|value| value as u8).collect::<Vec<_>>();
        let mut c = Encoder::open(1).unwrap();
        c.set_param(sys::AACENC_AOT, 2).unwrap();
        c.set_param(sys::AACENC_CHANNELMODE, 1).unwrap();
        c.set_param(sys::AACENC_SAMPLERATE, 48_000).unwrap();
        c.set_param(sys::AACENC_BITRATE, 192_000).unwrap();
        c.set_param(sys::AACENC_TRANSMUX, 0).unwrap();
        c.initialize().unwrap();
        let c_info = c.info().unwrap();
        assert_eq!(c_info.max_ancillary_bytes, 256);
        let mut c_raw = vec![0; c_info.max_output_bytes as usize];
        let mut c_result = (0, 0);
        for _ in 0..4 {
            c_result = c
                .encode_interleaved_i16_with_ancillary(
                    &vec![0; c_info.frame_length as usize],
                    &ancillary,
                    &mut c_raw,
                )
                .unwrap();
            if c_result.0 != 0 {
                break;
            }
        }
        let (c_bytes, c_consumed) = c_result;
        assert_ne!(c_bytes, 0);
        assert_eq!(c_consumed, 256);
        c_raw.truncate(c_bytes);

        let mut pure_from_c = AacLcDecoder::new(3, 1).unwrap();
        pure_from_c.init_ancillary_data(300);
        pure_from_c.decode_raw_data_block_f32(&c_raw).unwrap();
        assert_eq!(pure_from_c.ancillary_data()[0].data, ancillary[..256]);

        let mut parameters = configured(2, 1, 48_000);
        parameters
            .set_parameter(EncoderParameter::Bitrate, 192_000)
            .unwrap();
        parameters
            .set_parameter(EncoderParameter::TransportMux, 0)
            .unwrap();
        let mut pure = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
        assert_eq!(pure.max_ancillary_bytes_per_access_unit(), 256);
        let (mut pure_raw, pure_consumed) = pure
            .encode_interleaved_f32_with_ancillary(&vec![0.0; 1024], &ancillary)
            .unwrap();
        assert_eq!(pure_consumed, c_consumed);

        let mut c_decoder = Decoder::open(TransportType::Raw).unwrap();
        let mut asc = AudioSpecificConfig::aac_lc(48_000, 1)
            .unwrap()
            .to_bytes()
            .unwrap();
        c_decoder.configure_raw(&mut asc).unwrap();
        c_decoder.fill(&mut pure_raw).unwrap();
        c_decoder.decode_frame(&mut vec![0; 2048]).unwrap();
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn er_ancillary_extensions_are_bidirectionally_interoperable_with_c() {
        use crate::{sys, Decoder, Encoder, TransportType};

        let ancillary = [0x12, 0x34, 0x56];
        for aot in [23, 39] {
            let mut c = Encoder::open(1).unwrap();
            c.set_param(sys::AACENC_AOT, aot).unwrap();
            c.set_param(sys::AACENC_CHANNELMODE, 1).unwrap();
            c.set_param(sys::AACENC_SAMPLERATE, 48_000).unwrap();
            c.set_param(sys::AACENC_BITRATE, 64_000).unwrap();
            c.set_param(sys::AACENC_TRANSMUX, 0).unwrap();
            c.initialize().unwrap();
            let info = c.info().unwrap();
            let mut c_raw = vec![0; info.max_output_bytes as usize];
            let mut c_result = (0, 0);
            for _ in 0..8 {
                c_result = c
                    .encode_interleaved_i16_with_ancillary(
                        &vec![0; info.frame_length as usize],
                        &ancillary,
                        &mut c_raw,
                    )
                    .unwrap();
                if c_result.0 != 0 {
                    break;
                }
            }
            assert_eq!(c_result.1, ancillary.len());
            c_raw.truncate(c_result.0);
            let c_asc = AudioSpecificConfig::parse(&c.audio_specific_config().unwrap()).unwrap();
            let mut pure_decoder = AacLcDecoder::from_audio_specific_config(&c_asc).unwrap();
            pure_decoder.init_ancillary_data(ancillary.len());
            pure_decoder.decode_raw_data_block_f32(&c_raw).unwrap();
            assert_eq!(pure_decoder.ancillary_data()[0].data, ancillary);

            let mut parameters = configured(aot, 1, 48_000);
            parameters
                .set_parameter(EncoderParameter::Bitrate, 64_000)
                .unwrap();
            parameters
                .set_parameter(EncoderParameter::TransportMux, 0)
                .unwrap();
            let mut pure = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
            let pure_asc = backend_audio_specific_config(pure.config(), &pure.backend).unwrap();
            let (mut pure_raw, consumed) = pure
                .encode_interleaved_f32_with_ancillary(
                    &vec![0.0; pure.input_samples_per_channel()],
                    &ancillary,
                )
                .unwrap();
            assert_eq!(consumed, ancillary.len());
            let mut asc_bytes = pure_asc.to_bytes().unwrap();
            let mut c_decoder = Decoder::open(TransportType::Raw).unwrap();
            c_decoder.configure_raw(&mut asc_bytes).unwrap();
            c_decoder.fill(&mut pure_raw).unwrap();
            c_decoder.decode_frame(&mut vec![0; 2048]).unwrap();
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn default_metadata_modes_are_bidirectionally_interoperable_with_c() {
        use crate::decoder::STREAM_FLAG_DRC_PRESENT;
        use crate::{sys, Decoder, Encoder, TransportType};

        for mode in 1..=3 {
            let mut c = Encoder::open(1).unwrap();
            c.set_param(sys::AACENC_AOT, 2).unwrap();
            c.set_param(sys::AACENC_CHANNELMODE, 1).unwrap();
            c.set_param(sys::AACENC_SAMPLERATE, 48_000).unwrap();
            c.set_param(sys::AACENC_BITRATE, 96_000).unwrap();
            c.set_param(sys::AACENC_TRANSMUX, 0).unwrap();
            c.set_param(sys::AACENC_METADATA_MODE, mode).unwrap();
            c.initialize().unwrap();
            let info = c.info().unwrap();
            let mut c_raw = vec![0; info.max_output_bytes as usize];
            let mut bytes = 0;
            for _ in 0..8 {
                bytes = c
                    .encode_interleaved_i16(&vec![0; info.frame_length as usize], &mut c_raw)
                    .unwrap();
                if bytes != 0 {
                    break;
                }
            }
            assert_ne!(bytes, 0);
            c_raw.truncate(bytes);

            let mut rust_decoder = AacLcDecoder::new(3, 1).unwrap();
            rust_decoder.init_ancillary_data(32);
            rust_decoder.decode_raw_data_block_f32(&c_raw).unwrap();
            assert_eq!(
                rust_decoder.stream_info().flags & STREAM_FLAG_DRC_PRESENT != 0,
                matches!(mode, 1 | 2)
            );
            assert_eq!(
                rust_decoder
                    .ancillary_data()
                    .first()
                    .map(|data| data.data.as_slice()),
                matches!(mode, 2 | 3).then_some(&[0xbc, 0xc0, 0x00][..])
            );

            let mut parameters = configured(2, 1, 48_000);
            parameters
                .set_parameter(EncoderParameter::Bitrate, 96_000)
                .unwrap();
            parameters
                .set_parameter(EncoderParameter::TransportMux, 0)
                .unwrap();
            parameters
                .set_parameter(EncoderParameter::MetadataMode, mode)
                .unwrap();
            let mut rust_encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
            let mut rust_raw = rust_encoder
                .encode_interleaved_f32(&vec![0.0; 1024])
                .unwrap();
            let mut asc = AudioSpecificConfig::aac_lc(48_000, 1)
                .unwrap()
                .to_bytes()
                .unwrap();
            let mut c_decoder = Decoder::open(TransportType::Raw).unwrap();
            c_decoder.configure_raw(&mut asc).unwrap();
            c_decoder.fill(&mut rust_raw).unwrap();
            c_decoder.decode_frame(&mut vec![0; 2048]).unwrap();
        }
    }

    #[test]
    fn he_aac_tuning_selects_c_table_ranges_and_crossover() {
        let tuning = select_he_aac_mono_sbr_tuning(24_000, 30_000).unwrap();
        assert_eq!((tuning.start_frequency, tuning.stop_frequency), (10, 9));
        let (bandwidth, header) = he_aac_crossover_bandwidth(48_000, 30_000).unwrap();
        assert_eq!(bandwidth, 6_750);
        assert_eq!((header.start_frequency, header.stop_frequency), (10, 9));

        // As in getSbrTuningTableIndex, rates above the table select the
        // highest range below them while rates below the first range fail.
        assert_eq!(
            select_he_aac_mono_sbr_tuning(24_000, 100_000)
                .unwrap()
                .bitrate_to,
            64_001
        );
        assert!(select_he_aac_mono_sbr_tuning(24_000, 8_000).is_none());
    }

    #[test]
    fn eld_noise_max_level_tracks_c_tuning_boundaries() {
        assert_eq!(eld_mono_noise_max_level(12_000, 20_000), 6);
        assert_eq!(eld_mono_noise_max_level(16_000, 17_999), 6);
        assert_eq!(eld_mono_noise_max_level(16_000, 18_000), 9);
        assert_eq!(eld_mono_noise_max_level(16_000, 22_000), 6);
        assert_eq!(eld_mono_noise_max_level(16_000, 28_000), 12);
        assert_eq!(eld_mono_noise_max_level(16_000, 36_000), 3);
        assert_eq!(eld_mono_noise_max_level(22_050, 27_999), 6);
        assert_eq!(eld_mono_noise_max_level(22_050, 28_000), 3);
        assert_eq!(eld_mono_noise_max_level(24_000, 21_999), 6);
        assert_eq!(eld_mono_noise_max_level(24_000, 22_000), 3);
        assert_eq!(eld_mono_noise_max_level(48_000, 60_000), 3);
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn he_aac_bandwidth_matrix_matches_initialized_c_encoder() {
        use crate::{sys, Encoder};

        for (sample_rate, bitrate) in [
            (16_000, 10_000),
            (22_050, 14_000),
            (24_000, 20_000),
            (32_000, 30_000),
            (44_100, 30_000),
            (48_000, 30_000),
            (64_000, 40_000),
            (88_200, 50_000),
            (96_000, 50_000),
        ] {
            let mut rust = configured(5, 1, sample_rate);
            rust.set_parameter(EncoderParameter::Bitrate, bitrate)
                .unwrap();
            let resolved = rust.resolve().unwrap();

            let mut c = Encoder::open(1).unwrap();
            c.set_param(sys::AACENC_AOT, 5).unwrap();
            c.set_param(sys::AACENC_CHANNELMODE, 1).unwrap();
            c.set_param(sys::AACENC_SAMPLERATE, sample_rate).unwrap();
            c.set_param(sys::AACENC_BITRATE, bitrate).unwrap();
            c.initialize().unwrap();
            assert_eq!(
                c.get_param(sys::AACENC_BANDWIDTH),
                resolved.bandwidth,
                "HE-AAC bandwidth differs at {sample_rate} Hz / {bitrate} bit/s"
            );
        }
    }

    #[test]
    fn unified_factory_encodes_he_aac_stereo_with_stereo_asc() {
        let mut parameters = configured(5, 2, 48_000);
        parameters
            .set_parameter(EncoderParameter::Bitrate, 64_000)
            .unwrap();
        parameters
            .set_parameter(EncoderParameter::TransportMux, 0)
            .unwrap();
        let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
        assert_eq!(encoder.input_samples_per_channel(), 2048);
        let asc = backend_audio_specific_config(&encoder.config, &encoder.backend).unwrap();
        assert_eq!(asc.channel_configuration, 2);
        assert_eq!(asc.extension.as_ref().unwrap().audio_object_type, 5);

        let input = (0..2048)
            .flat_map(|sample| {
                let phase = sample as f32 * 0.031;
                [phase.sin() * 0.6, (phase * 1.37 + 0.4).sin() * 0.4]
            })
            .collect::<Vec<_>>();
        let raw = encoder.encode_interleaved_f32(&input).unwrap();
        assert!(!raw.is_empty());
        let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
        let decoded = decoder
            .decode_raw_data_block_multichannel_f32(&raw)
            .unwrap();
        assert_eq!(decoded.channels.len(), 2);
        assert!(decoded.channels.iter().all(|channel| channel.len() == 2048));
        assert!(decoded
            .channels
            .iter()
            .flatten()
            .all(|sample| sample.is_finite()));
    }

    #[test]
    fn unified_factory_encodes_he_aac_multichannel_layouts() {
        for (mode, channels, channel_configuration) in [
            (3, 3usize, 3),
            (4, 4, 4),
            (5, 5, 5),
            (6, 6, 6),
            (7, 8, 7),
            (11, 7, 11),
            (12, 8, 12),
            (14, 8, 14),
            (33, 8, 0),
            (34, 8, 0),
        ] {
            let mut parameters = configured(5, mode, 48_000);
            parameters
                .set_parameter(EncoderParameter::Bitrate, channels as u32 * 32_000)
                .unwrap();
            parameters
                .set_parameter(EncoderParameter::TransportMux, 0)
                .unwrap();
            let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
            assert_eq!(encoder.input_samples_per_channel(), 2048);
            let asc = backend_audio_specific_config(&encoder.config, &encoder.backend).unwrap();
            assert_eq!(asc.channel_configuration, channel_configuration);
            assert_eq!(asc.program_config.is_some(), matches!(mode, 33 | 34));
            assert_eq!(asc.extension.as_ref().unwrap().audio_object_type, 5);

            let input = (0..2048)
                .flat_map(|sample| {
                    (0..channels).map(move |channel| {
                        let phase = sample as f32 * (0.017 + channel as f32 * 0.003);
                        phase.sin() * (0.5 / (channel + 1) as f32)
                    })
                })
                .collect::<Vec<_>>();
            let raw = encoder.encode_interleaved_f32(&input).unwrap();
            let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
            let decoded = decoder
                .decode_raw_data_block_multichannel_f32(&raw)
                .unwrap();
            assert_eq!(decoded.channels.len(), channels);
            assert_eq!(
                decoded.channels.iter().map(Vec::len).collect::<Vec<_>>(),
                vec![2048; channels]
            );
            assert!(decoded
                .channels
                .iter()
                .flatten()
                .all(|sample| sample.is_finite()));
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn fdk_decoder_accepts_pure_rust_he_aac_stereo_and_multichannel() {
        use crate::{Decoder, TransportType};

        for channels in 2usize..=6 {
            let mut parameters = configured(5, channels as u32, 48_000);
            parameters
                .set_parameter(EncoderParameter::Bitrate, channels as u32 * 32_000)
                .unwrap();
            parameters
                .set_parameter(EncoderParameter::TransportMux, 0)
                .unwrap();
            let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
            let mut asc = backend_audio_specific_config(&encoder.config, &encoder.backend)
                .unwrap()
                .to_bytes()
                .unwrap();
            let mut raw = encoder
                .encode_interleaved_f32(&vec![0.0; 2048 * channels])
                .unwrap();
            let mut decoder = Decoder::open(TransportType::Raw).unwrap();
            decoder.configure_raw(&mut asc).unwrap();
            decoder.fill(&mut raw).unwrap();
            decoder.decode_frame(&mut vec![0; 2048 * channels]).unwrap();
            assert_eq!(decoder.stream_info().unwrap().channels, channels as i32);
        }
    }

    #[test]
    fn unified_factory_maps_mpeg2_virtual_aots_to_lc_and_sbr_backends() {
        for &(aot, transport_aot, samples) in &[(129, 2u8, 1024usize), (132, 5, 2048)] {
            for channels in 1usize..=6 {
                let mut parameters = configured(aot, channels as u32, 48_000);
                parameters
                    .set_parameter(EncoderParameter::Bitrate, channels as u32 * 32_000)
                    .unwrap();
                parameters
                    .set_parameter(EncoderParameter::TransportMux, 0)
                    .unwrap();
                let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
                assert_eq!(encoder.input_samples_per_channel(), samples);
                let asc = backend_audio_specific_config(&encoder.config, &encoder.backend).unwrap();
                if aot == 129 {
                    assert_eq!(asc.audio_object_type, transport_aot);
                    assert!(asc.extension.is_none());
                } else {
                    assert_eq!(asc.audio_object_type, 2);
                    assert_eq!(
                        asc.extension.as_ref().unwrap().audio_object_type,
                        transport_aot
                    );
                }
                let raw = encoder
                    .encode_interleaved_f32(&vec![0.0; samples * channels])
                    .unwrap();
                let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
                let decoded = decoder
                    .decode_raw_data_block_multichannel_f32(&raw)
                    .unwrap();
                assert_eq!(decoded.channels.len(), channels);
                assert!(decoded
                    .channels
                    .iter()
                    .all(|channel| channel.len() == samples));
            }
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn fdk_decoder_accepts_pure_rust_mpeg2_virtual_aot_output() {
        use crate::{Decoder, TransportType};

        for &(aot, samples) in &[(129, 1024usize), (5, 2048), (132, 2048)] {
            for channels in [1usize, 2, 6] {
                let mut parameters = configured(aot, channels as u32, 48_000);
                parameters
                    .set_parameter(EncoderParameter::Bitrate, channels as u32 * 32_000)
                    .unwrap();
                parameters
                    .set_parameter(EncoderParameter::TransportMux, 0)
                    .unwrap();
                let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
                let mut asc = backend_audio_specific_config(&encoder.config, &encoder.backend)
                    .unwrap()
                    .to_bytes()
                    .unwrap();
                let mut raw = encoder
                    .encode_interleaved_f32(&vec![0.0; samples * channels])
                    .unwrap();
                let mut decoder = Decoder::open(TransportType::Raw).unwrap();
                decoder.configure_raw(&mut asc).unwrap();
                decoder.fill(&mut raw).unwrap();
                decoder
                    .decode_frame(&mut vec![0; samples * channels.max(2)])
                    .unwrap_or_else(|error| {
                        panic!("C decoder rejected virtual AOT {aot}, {channels}ch: {error:?}")
                    });
                assert_eq!(
                    decoder.stream_info().unwrap().channels,
                    if samples == 2048 {
                        channels.max(2) as i32
                    } else {
                        channels as i32
                    },
                    "virtual AOT {aot}, {channels}ch"
                );
            }
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn c_decoder_accepts_unified_adts_and_low_delay_loas_output() {
        use crate::{Decoder, TransportType};

        let mut lc = ConfiguredPureRustEncoder::from_parameters(&configured(2, 1, 48_000)).unwrap();
        let mut adts = lc.encode_transport_f32(&vec![0.0; 1024]).unwrap();
        let mut decoder = Decoder::open(TransportType::Adts).unwrap();
        decoder.fill(&mut adts).unwrap();
        decoder.decode_frame(&mut vec![0; 4096]).unwrap();

        for (mode, channels) in [(11, 7usize), (12, 8), (14, 8), (33, 8), (34, 8)] {
            let mut parameters = configured(2, mode, 48_000);
            parameters
                .set_parameter(EncoderParameter::Bitrate, channels as u32 * 40_000)
                .unwrap();
            parameters
                .set_parameter(EncoderParameter::TransportMux, 2)
                .unwrap();
            let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
            let mut adts = encoder
                .encode_transport_f32(&vec![0.0; 1024 * channels])
                .unwrap();
            let parsed = crate::adts::AdtsFrame::parse(&adts).unwrap();
            assert_eq!(parsed.header.channel_configuration, 0);
            let mut decoder = Decoder::open(TransportType::Adts).unwrap();
            decoder.fill(&mut adts).unwrap();
            decoder
                .decode_frame(&mut vec![0; 2048 * channels])
                .unwrap_or_else(|error| {
                    panic!("C decoder rejected ADTS channel mode {mode}: {error:?}")
                });
        }

        for (mode, channels) in [(1, 1usize), (11, 7)] {
            let mut parameters = configured(5, mode, 48_000);
            parameters
                .set_parameter(EncoderParameter::Bitrate, channels as u32 * 32_000)
                .unwrap();
            parameters
                .set_parameter(EncoderParameter::TransportMux, 2)
                .unwrap();
            let mut encoder = ConfiguredPureRustEncoder::from_parameters(&parameters).unwrap();
            let mut adts = encoder
                .encode_transport_f32(&vec![0.0; 2048 * channels])
                .unwrap();
            let parsed = crate::adts::AdtsFrame::parse(&adts).unwrap();
            assert_eq!(
                parsed.header.sampling_frequency_index,
                sample_rate_index(24_000).unwrap()
            );
            assert_eq!(
                parsed.header.channel_configuration,
                if mode == 1 { 1 } else { 0 }
            );
            let mut decoder = Decoder::open(TransportType::Adts).unwrap();
            decoder.fill(&mut adts).unwrap();
            decoder
                .decode_frame(&mut vec![0; 4096 * channels.max(2)])
                .unwrap_or_else(|error| {
                    panic!("C decoder rejected HE-AAC ADTS channel mode {mode}: {error:?}")
                });
            assert_eq!(decoder.stream_info().unwrap().sample_rate, 48_000);
        }

        let mut protected_parameters = configured(2, 1, 48_000);
        protected_parameters
            .set_parameter(EncoderParameter::Protection, 1)
            .unwrap();
        protected_parameters
            .set_parameter(EncoderParameter::TransportSubframes, 2)
            .unwrap();
        let mut protected =
            ConfiguredPureRustEncoder::from_parameters(&protected_parameters).unwrap();
        assert!(protected
            .encode_transport_f32(&vec![0.0; 1024])
            .unwrap()
            .is_empty());
        let mut protected_adts = protected.encode_transport_f32(&vec![0.0; 1024]).unwrap();
        let mut decoder = Decoder::open(TransportType::Adts).unwrap();
        decoder.fill(&mut protected_adts).unwrap();
        decoder.decode_frame(&mut vec![0; 4096]).unwrap();

        let mut ld =
            ConfiguredPureRustEncoder::from_parameters(&configured(23, 1, 48_000)).unwrap();
        let mut loas = ld.encode_transport_f32(&vec![0.0; 512]).unwrap();
        let mut decoder = Decoder::open(TransportType::Loas).unwrap();
        decoder.fill(&mut loas).unwrap();
        decoder.decode_frame(&mut vec![0; 4096]).unwrap();
        let info = decoder.stream_info().unwrap();
        assert_eq!(
            (info.sample_rate, info.frame_size, info.channels),
            (48_000, 512, 1)
        );
    }
}
