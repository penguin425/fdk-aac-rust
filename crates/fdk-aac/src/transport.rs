//! Unified Pure Rust AAC-LC transport decoder entry points.
//!
//! This facade binds an AAC-LC decoder to either raw access units configured by
//! an AudioSpecificConfig or ADTS frames. Other FDK transport formats are
//! explicitly rejected instead of being interpreted as raw AAC data.

use std::fmt;

use crate::adif::{AdifError, AdifHeader};
use crate::adts::{
    sample_rate_from_index, AccessUnitLossEstimator, AdtsError, AdtsFrame, AdtsHeader,
    AdtsIncrementalStream,
};
use crate::asc::{AscError, AudioSpecificConfig};
use crate::bits::{BitError, BitReader};
use crate::concealment::PcmConcealment;
use crate::decoder::{
    f32_to_i16, interleave_multichannel_f32, interleave_multichannel_i16, AacLcDecoder,
    ChannelLabel, DecodeError, DecoderStreamInfo, STREAM_FLAG_MPS_PRESENT, STREAM_FLAG_SBR_PRESENT,
};
use crate::drm::{
    DrmAacDecodeError, DrmAacDecoder, DrmAudioConfig, DrmXheDecodeError, DrmXheDecoder,
};
use crate::latm::{LatmAudioMuxElement, LatmError, LatmMuxConfig};
use crate::limiter::TimeDomainLimiter;
use crate::loas::{LoasError, LoasFrame, LoasIncrementalStream};
use crate::usac_decoder::UsacDecodeError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AacTransport {
    Raw,
    Adif,
    Adts,
    LatmMuxConfigPresent,
    LatmOutOfBandConfig,
    Loas,
    Drm,
}

/// Flags corresponding to `AACDEC_CONCEAL`, `AACDEC_FLUSH`, `AACDEC_INTR`
/// and `AACDEC_CLRHIST`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DecodeFrameFlags(u32);

impl DecodeFrameFlags {
    pub const NONE: Self = Self(0);
    pub const CONCEAL: Self = Self(1);
    pub const FLUSH: Self = Self(2);
    pub const INTERRUPTION: Self = Self(4);
    pub const CLEAR_HISTORY: Self = Self(8);
    pub const ALL: Self = Self(15);

    pub const fn bits(self) -> u32 {
        self.0
    }

    pub const fn from_bits(bits: u32) -> Option<Self> {
        if bits & !Self::ALL.0 == 0 {
            Some(Self(bits))
        } else {
            None
        }
    }

    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

impl std::ops::BitOr for DecodeFrameFlags {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for DecodeFrameFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DualChannelOutputMode {
    #[default]
    Stereo,
    Channel1,
    Channel2,
    Mix,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PcmChannelOrder {
    Mpeg,
    #[default]
    Wav,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConcealmentMethod {
    SpectralMute,
    NoiseSubstitution,
    EnergyInterpolation,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MetadataProfile {
    #[default]
    MpegStandard,
    MpegLegacy,
    MpegLegacyPriority,
    AribJapan,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum QmfProcessingMode {
    #[default]
    Automatic,
    Complex,
    LowPower,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderParameter {
    PcmDualChannelOutputMode,
    PcmOutputChannelMapping,
    PcmLimiterEnable,
    PcmLimiterAttackTime,
    PcmLimiterReleaseTime,
    PcmMinOutputChannels,
    PcmMaxOutputChannels,
    MetadataProfile,
    MetadataExpiryTime,
    DrcBoostFactor,
    DrcAttenuationFactor,
    DrcReferenceLevel,
    DrcHeavyCompression,
    DrcDefaultPresentationMode,
    DrcEncoderTargetLevel,
    UniDrcSetEffect,
    UniDrcAlbumMode,
    QmfLowPower,
    ConcealMethod,
    TransportClearBuffer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PcmOutputConfig {
    dual_channel_mode: DualChannelOutputMode,
    channel_order: PcmChannelOrder,
    min_channels: Option<usize>,
    max_channels: Option<usize>,
    metadata_profile: MetadataProfile,
    advanced_downmix: Option<crate::drc::DvbAncillaryDownmixMetadata>,
    matrix_mixdown: Option<crate::asc::MatrixMixdown>,
}

impl Default for PcmOutputConfig {
    fn default() -> Self {
        Self {
            dual_channel_mode: DualChannelOutputMode::Stereo,
            channel_order: PcmChannelOrder::Wav,
            min_channels: None,
            max_channels: None,
            metadata_profile: MetadataProfile::MpegStandard,
            advanced_downmix: None,
            matrix_mixdown: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct PcmFrameDelay {
    delay_samples_per_channel: usize,
    channels: usize,
    buffer: Vec<f64>,
    index: usize,
}

impl PcmFrameDelay {
    fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.index = 0;
    }

    fn process_f32(&mut self, samples: &mut [f32], channels: usize, delay: usize) {
        let mut normalized = samples
            .iter()
            .map(|&sample| sample as f64)
            .collect::<Vec<_>>();
        self.process(&mut normalized, channels, delay);
        for (sample, delayed) in samples.iter_mut().zip(normalized) {
            *sample = delayed as f32;
        }
    }

    fn process_i16(&mut self, samples: &mut [i16], channels: usize, delay: usize) {
        let mut normalized = samples
            .iter()
            .map(|&sample| sample as f64 / 32768.0)
            .collect::<Vec<_>>();
        self.process(&mut normalized, channels, delay);
        for (sample, delayed) in samples.iter_mut().zip(normalized) {
            *sample = (delayed * 32768.0)
                .round()
                .clamp(i16::MIN as f64, i16::MAX as f64) as i16;
        }
    }

    fn process(&mut self, samples: &mut [f64], channels: usize, delay: usize) {
        if delay == 0 || channels == 0 || !samples.len().is_multiple_of(channels) {
            if self.delay_samples_per_channel != 0 {
                self.delay_samples_per_channel = 0;
                self.channels = channels;
                self.buffer.clear();
                self.index = 0;
            }
            return;
        }
        let buffer_len = delay.saturating_mul(channels);
        if self.delay_samples_per_channel != delay || self.channels != channels {
            self.delay_samples_per_channel = delay;
            self.channels = channels;
            self.buffer = vec![0.0; buffer_len];
            self.index = 0;
        }
        for sample in samples {
            std::mem::swap(sample, &mut self.buffer[self.index]);
            self.index = (self.index + 1) % self.buffer.len();
        }
    }

    fn replace_pending_f32(&mut self, samples: &[f32], channels: usize, delay: usize) -> bool {
        let normalized = samples
            .iter()
            .map(|&sample| f64::from(sample))
            .collect::<Vec<_>>();
        self.replace_pending(&normalized, channels, delay)
    }

    fn replace_pending_i16(&mut self, samples: &[i16], channels: usize, delay: usize) -> bool {
        let normalized = samples
            .iter()
            .map(|&sample| f64::from(sample) / 32768.0)
            .collect::<Vec<_>>();
        self.replace_pending(&normalized, channels, delay)
    }

    fn replace_pending(&mut self, samples: &[f64], channels: usize, delay: usize) -> bool {
        if delay == 0
            || self.delay_samples_per_channel != delay
            || self.channels != channels
            || samples.len() != self.buffer.len()
            || self.buffer.is_empty()
        {
            return false;
        }
        for (offset, &sample) in samples.iter().enumerate() {
            let index = (self.index + offset) % self.buffer.len();
            self.buffer[index] = sample;
        }
        true
    }
}

impl AacTransport {
    pub fn is_supported_by_pure_rust(self) -> bool {
        matches!(
            self,
            Self::Raw
                | Self::Adif
                | Self::Adts
                | Self::LatmMuxConfigPresent
                | Self::LatmOutOfBandConfig
                | Self::Loas
                | Self::Drm
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TransportDecodeError {
    Adts(AdtsError),
    Adif(AdifError),
    Latm(LatmError),
    Loas(LoasError),
    Drm(DrmAacDecodeError),
    DrmXhe(DrmXheDecodeError),
    Asc(AscError),
    Decode(DecodeError),
    Usac(UsacDecodeError),
    InvalidParameterValue {
        parameter: DecoderParameter,
        value: i32,
    },
    TransportMismatch {
        configured: AacTransport,
        requested: AacTransport,
    },
    UnsupportedTransport(AacTransport),
}

impl From<AdtsError> for TransportDecodeError {
    fn from(value: AdtsError) -> Self {
        Self::Adts(value)
    }
}

impl From<AdifError> for TransportDecodeError {
    fn from(value: AdifError) -> Self {
        Self::Adif(value)
    }
}
impl From<LatmError> for TransportDecodeError {
    fn from(value: LatmError) -> Self {
        Self::Latm(value)
    }
}
impl From<LoasError> for TransportDecodeError {
    fn from(value: LoasError) -> Self {
        Self::Loas(value)
    }
}
impl From<DrmAacDecodeError> for TransportDecodeError {
    fn from(value: DrmAacDecodeError) -> Self {
        Self::Drm(value)
    }
}
impl From<DrmXheDecodeError> for TransportDecodeError {
    fn from(value: DrmXheDecodeError) -> Self {
        Self::DrmXhe(value)
    }
}

impl From<AscError> for TransportDecodeError {
    fn from(value: AscError) -> Self {
        Self::Asc(value)
    }
}

impl From<DecodeError> for TransportDecodeError {
    fn from(value: DecodeError) -> Self {
        Self::Decode(value)
    }
}

impl From<UsacDecodeError> for TransportDecodeError {
    fn from(value: UsacDecodeError) -> Self {
        Self::Usac(value)
    }
}

impl fmt::Display for TransportDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Adts(err) => err.fmt(f),
            Self::Adif(err) => err.fmt(f),
            Self::Latm(err) => err.fmt(f),
            Self::Loas(err) => err.fmt(f),
            Self::Drm(err) => err.fmt(f),
            Self::DrmXhe(err) => err.fmt(f),
            Self::Asc(err) => err.fmt(f),
            Self::Decode(err) => err.fmt(f),
            Self::Usac(err) => write!(f, "USAC decode error: {err:?}"),
            Self::InvalidParameterValue { parameter, value } => {
                write!(
                    f,
                    "invalid value {value} for AAC decoder parameter {parameter:?}"
                )
            }
            Self::TransportMismatch {
                configured,
                requested,
            } => write!(
                f,
                "AAC transport decoder is configured for {configured:?}, not {requested:?}"
            ),
            Self::UnsupportedTransport(transport) => {
                write!(f, "Pure Rust AAC-LC transport {transport:?} is unsupported")
            }
        }
    }
}

impl std::error::Error for TransportDecodeError {}

/// Incremental ADIF decoder.
///
/// ADIF has no per-frame byte length. This decoder therefore parses each
/// `raw_data_block` through its `ID_END` element and commits decoder state only
/// after the complete block is available. A truncated block remains buffered
/// and is retried transactionally when more bytes arrive.
#[derive(Debug, Default)]
pub struct AdifIncrementalDecoder {
    buffered: Vec<u8>,
    header: Option<AdifHeader>,
    decoder: Option<AacLcDecoder>,
}

impl AdifIncrementalDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, input: &[u8]) {
        self.buffered.extend_from_slice(input);
    }

    pub fn buffered_len(&self) -> usize {
        self.buffered.len()
    }

    pub fn header(&self) -> Option<&AdifHeader> {
        self.header.as_ref()
    }

    pub fn drain_interleaved_f32(&mut self) -> Result<Vec<Vec<f32>>, TransportDecodeError> {
        if !self.ensure_header()? {
            return Ok(Vec::new());
        }
        let mut output = Vec::new();
        loop {
            let mut trial = self
                .decoder
                .as_ref()
                .expect("ADIF decoder configured")
                .clone();
            let mut reader = BitReader::new(&self.buffered);
            match trial.decode_raw_data_block_f32_terminated_from_reader(&mut reader) {
                Ok(frame) => {
                    reader.byte_align();
                    let consumed = reader.bits_read() / 8;
                    self.buffered.drain(..consumed);
                    self.decoder = Some(trial);
                    output.push(frame.interleaved_f32());
                }
                Err(error) if error.is_unexpected_eof() => return Ok(output),
                Err(error) => return Err(error.into()),
            }
        }
    }

    pub fn drain_interleaved_i16(&mut self) -> Result<Vec<Vec<i16>>, TransportDecodeError> {
        if !self.ensure_header()? {
            return Ok(Vec::new());
        }
        let mut output = Vec::new();
        loop {
            let mut trial = self
                .decoder
                .as_ref()
                .expect("ADIF decoder configured")
                .clone();
            let mut reader = BitReader::new(&self.buffered);
            match trial
                .decode_raw_data_block_fixed_interleaved_i16_terminated_from_reader(&mut reader)
            {
                Ok(samples) => {
                    reader.byte_align();
                    let consumed = reader.bits_read() / 8;
                    self.buffered.drain(..consumed);
                    self.decoder = Some(trial);
                    output.push(samples);
                }
                Err(error) if error.is_unexpected_eof() => return Ok(output),
                Err(error) => return Err(error.into()),
            }
        }
    }

    fn ensure_header(&mut self) -> Result<bool, TransportDecodeError> {
        if self.decoder.is_some() {
            return Ok(true);
        }
        let header = match AdifHeader::parse(&self.buffered) {
            Ok(header) => header,
            Err(error) if adif_error_is_unexpected_eof(&error) => return Ok(false),
            Err(error) => return Err(error.into()),
        };
        let header_len = header.bits_read / 8;
        let decoder = AacLcDecoder::from_adif_header(&header)?;
        self.buffered.drain(..header_len);
        self.header = Some(header);
        self.decoder = Some(decoder);
        Ok(true)
    }
}

fn adif_error_is_unexpected_eof(error: &AdifError) -> bool {
    matches!(
        error,
        AdifError::Bit(BitError::UnexpectedEof { .. })
            | AdifError::Asc(AscError::UnexpectedEof { .. })
    )
}

/// Stateful transport facade for Pure Rust AAC-LC decoding.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct TransportStatistics {
    bit_rate: u32,
    total_bytes: u64,
    bad_bytes: u64,
    total_access_units: u64,
    bad_access_units: u64,
}

#[derive(Debug, Clone)]
pub struct PureRustTransportDecoder {
    transport: AacTransport,
    decoder: AacLcDecoder,
    adts_input: AdtsIncrementalStream,
    adts_loss_estimator: Option<AccessUnitLossEstimator>,
    adts_observed_discarded_bytes: usize,
    adts_pending_discarded_bits: usize,
    estimated_lost_access_units: u64,
    statistics: TransportStatistics,
    pcm_output: PcmOutputConfig,
    pcm_limiter_enable: i8,
    pcm_limiter: TimeDomainLimiter,
    metadata_profile: MetadataProfile,
    metadata_expiry_ms: u32,
    qmf_processing_mode: QmfProcessingMode,
    concealment_method_user: Option<ConcealmentMethod>,
    concealment_delay: PcmFrameDelay,
    pending_energy_f32_losses: usize,
    pending_energy_i16_losses: usize,
    adts_pcm_concealment: Option<PcmConcealment>,
    adts_spectral_concealment: bool,
    loas_input: LoasIncrementalStream,
    loas_observed_discarded_bytes: usize,
    latm_mux_config: Option<LatmMuxConfig>,
    adif_header_len: Option<usize>,
    drm_decoder: Option<DrmAacDecoder>,
    drm_xhe_decoder: Option<DrmXheDecoder>,
}

impl PureRustTransportDecoder {
    pub fn from_audio_specific_config(
        config: &AudioSpecificConfig,
    ) -> Result<Self, TransportDecodeError> {
        let mut transport = Self {
            transport: AacTransport::Raw,
            decoder: AacLcDecoder::from_audio_specific_config(config)?,
            adts_input: AdtsIncrementalStream::new(),
            adts_loss_estimator: None,
            adts_observed_discarded_bytes: 0,
            adts_pending_discarded_bits: 0,
            estimated_lost_access_units: 0,
            statistics: TransportStatistics::default(),
            pcm_output: PcmOutputConfig::default(),
            pcm_limiter_enable: -1,
            pcm_limiter: TimeDomainLimiter::default(),
            metadata_profile: MetadataProfile::default(),
            metadata_expiry_ms: 0,
            qmf_processing_mode: QmfProcessingMode::default(),
            concealment_method_user: None,
            concealment_delay: PcmFrameDelay::default(),
            pending_energy_f32_losses: 0,
            pending_energy_i16_losses: 0,
            adts_pcm_concealment: None,
            adts_spectral_concealment: false,
            loas_input: LoasIncrementalStream::new(),
            loas_observed_discarded_bytes: 0,
            latm_mux_config: None,
            adif_header_len: None,
            drm_decoder: None,
            drm_xhe_decoder: None,
        };
        transport.sync_qmf_processing_mode();
        Ok(transport)
    }

    pub fn from_asc_bytes(input: &[u8]) -> Result<Self, TransportDecodeError> {
        Self::from_audio_specific_config(&AudioSpecificConfig::parse(input)?)
    }

    pub fn from_adts_header(header: AdtsHeader) -> Result<Self, TransportDecodeError> {
        let mut transport = Self {
            transport: AacTransport::Adts,
            decoder: AacLcDecoder::from_adts_header(header)?,
            adts_input: AdtsIncrementalStream::new(),
            adts_loss_estimator: None,
            adts_observed_discarded_bytes: 0,
            adts_pending_discarded_bits: 0,
            estimated_lost_access_units: 0,
            statistics: TransportStatistics::default(),
            pcm_output: PcmOutputConfig::default(),
            pcm_limiter_enable: -1,
            pcm_limiter: TimeDomainLimiter::default(),
            metadata_profile: MetadataProfile::default(),
            metadata_expiry_ms: 0,
            qmf_processing_mode: QmfProcessingMode::default(),
            concealment_method_user: None,
            concealment_delay: PcmFrameDelay::default(),
            pending_energy_f32_losses: 0,
            pending_energy_i16_losses: 0,
            adts_pcm_concealment: None,
            adts_spectral_concealment: false,
            loas_input: LoasIncrementalStream::new(),
            loas_observed_discarded_bytes: 0,
            latm_mux_config: None,
            adif_header_len: None,
            drm_decoder: None,
            drm_xhe_decoder: None,
        };
        transport.sync_qmf_processing_mode();
        Ok(transport)
    }

    pub fn from_adts_frame(input: &[u8]) -> Result<Self, TransportDecodeError> {
        Self::from_adts_header(AdtsFrame::parse(input)?.header)
    }

    pub fn from_adif_header(header: &AdifHeader) -> Result<Self, TransportDecodeError> {
        let mut transport = Self {
            transport: AacTransport::Adif,
            decoder: AacLcDecoder::from_adif_header(header)?,
            adts_input: AdtsIncrementalStream::new(),
            adts_loss_estimator: None,
            adts_observed_discarded_bytes: 0,
            adts_pending_discarded_bits: 0,
            estimated_lost_access_units: 0,
            statistics: TransportStatistics::default(),
            pcm_output: PcmOutputConfig::default(),
            pcm_limiter_enable: -1,
            pcm_limiter: TimeDomainLimiter::default(),
            metadata_profile: MetadataProfile::default(),
            metadata_expiry_ms: 0,
            qmf_processing_mode: QmfProcessingMode::default(),
            concealment_method_user: None,
            concealment_delay: PcmFrameDelay::default(),
            pending_energy_f32_losses: 0,
            pending_energy_i16_losses: 0,
            adts_pcm_concealment: None,
            adts_spectral_concealment: false,
            loas_input: LoasIncrementalStream::new(),
            loas_observed_discarded_bytes: 0,
            latm_mux_config: None,
            adif_header_len: Some(header.bits_read / 8),
            drm_decoder: None,
            drm_xhe_decoder: None,
        };
        transport.sync_qmf_processing_mode();
        Ok(transport)
    }

    pub fn from_adif_bytes(input: &[u8]) -> Result<Self, TransportDecodeError> {
        Self::from_adif_header(&AdifHeader::parse(input)?)
    }

    /// Configure `TT_DRM` from its 16-bit SDC type-9 AAC configuration.
    /// Decode inputs include the leading DRM CRC byte.
    pub fn from_drm_sdc_config(input: &[u8]) -> Result<Self, TransportDecodeError> {
        let drm_decoder = DrmAacDecoder::from_sdc_config(input)?;
        let decoder = drm_decoder.decoder.clone();
        let asc = AudioSpecificConfig::aac_lc(
            drm_decoder.drm_config.sampling_frequency,
            drm_decoder.drm_config.channel_configuration,
        )?;
        let mut transport = Self::from_audio_specific_config(&asc)?;
        transport.transport = AacTransport::Drm;
        transport.decoder = decoder;
        transport.drm_decoder = Some(drm_decoder);
        transport.sync_qmf_processing_mode();
        Ok(transport)
    }

    pub fn from_drm_config(config: &DrmAudioConfig) -> Result<Self, TransportDecodeError> {
        let bytes = config.to_bytes().map_err(DrmAacDecodeError::from)?;
        Self::from_drm_sdc_config(&bytes)
    }

    /// Configure DRM xHE-AAC from the SDC type-9 fields followed by the
    /// DRM-specific USAC static decoder configuration. xHE access units are
    /// passed directly; unlike legacy DRM AAC, their USAC syntax does not use
    /// the leading transport CRC byte.
    pub fn from_drm_xhe_static_config(input: &[u8]) -> Result<Self, TransportDecodeError> {
        let drm_xhe_decoder = DrmXheDecoder::from_static_config(input)?;
        let decoder = drm_xhe_decoder.decoder.clone();
        let asc = AudioSpecificConfig::aac_lc(
            drm_xhe_decoder.drm_config.sampling_frequency,
            drm_xhe_decoder.drm_config.channel_configuration,
        )?;
        let mut transport = Self::from_audio_specific_config(&asc)?;
        transport.transport = AacTransport::Drm;
        transport.decoder = decoder;
        transport.drm_xhe_decoder = Some(drm_xhe_decoder);
        transport.sync_qmf_processing_mode();
        Ok(transport)
    }

    /// Configure a LOAS/LATM AAC-LC decoder from an AudioMuxElement carrying a
    /// StreamMuxConfig (`useSameStreamMux == 0`).
    pub fn from_loas_frame(input: &[u8]) -> Result<Self, TransportDecodeError> {
        let loas = LoasFrame::parse(input)?;
        let latm = LatmAudioMuxElement::parse_aac_lc(loas.audio_mux_element)?;
        let latm_mux_config = latm.mux_config.clone();
        let config = latm
            .config
            .ok_or(LatmError::UnsupportedProgramOrLayerLayout)?;
        let mut transport = Self {
            transport: AacTransport::Loas,
            decoder: AacLcDecoder::from_audio_specific_config(&config)?,
            adts_input: AdtsIncrementalStream::new(),
            adts_loss_estimator: None,
            adts_observed_discarded_bytes: 0,
            adts_pending_discarded_bits: 0,
            estimated_lost_access_units: 0,
            statistics: TransportStatistics::default(),
            pcm_output: PcmOutputConfig::default(),
            pcm_limiter_enable: -1,
            pcm_limiter: TimeDomainLimiter::default(),
            metadata_profile: MetadataProfile::default(),
            metadata_expiry_ms: 0,
            qmf_processing_mode: QmfProcessingMode::default(),
            concealment_method_user: None,
            concealment_delay: PcmFrameDelay::default(),
            pending_energy_f32_losses: 0,
            pending_energy_i16_losses: 0,
            adts_pcm_concealment: None,
            adts_spectral_concealment: false,
            loas_input: LoasIncrementalStream::new(),
            loas_observed_discarded_bytes: 0,
            latm_mux_config,
            adif_header_len: None,
            drm_decoder: None,
            drm_xhe_decoder: None,
        };
        transport.sync_qmf_processing_mode();
        Ok(transport)
    }

    /// Configure the direct LATM transport from an AudioMuxElement containing
    /// an in-band StreamMuxConfig.
    pub fn from_latm_audio_mux_element(input: &[u8]) -> Result<Self, TransportDecodeError> {
        let latm = LatmAudioMuxElement::parse_aac_lc(input)?;
        let latm_mux_config = latm.mux_config.clone();
        let config = latm
            .config
            .ok_or(LatmError::UnsupportedProgramOrLayerLayout)?;
        let mut decoder = Self::from_audio_specific_config(&config)?;
        decoder.transport = AacTransport::LatmMuxConfigPresent;
        decoder.latm_mux_config = latm_mux_config;
        Ok(decoder)
    }

    /// Configure the direct LATM transport from an out-of-band ASC. Incoming
    /// AudioMuxElements may immediately use `useSameStreamMux`.
    pub fn from_latm_out_of_band_config(
        config: &AudioSpecificConfig,
    ) -> Result<Self, TransportDecodeError> {
        let mut decoder = Self::from_audio_specific_config(config)?;
        decoder.transport = AacTransport::LatmOutOfBandConfig;
        decoder.latm_mux_config = Some(LatmMuxConfig {
            audio_mux_version: 0,
            all_streams_same_time_framing: true,
            subframe_count: 1,
            streams: vec![crate::latm::LatmStreamLayer {
                program: 0,
                layer: 0,
                config: Some(config.clone()),
                frame_length_type: 0,
                fixed_frame_length_bits: None,
            }],
            other_data_bits: 0,
            crc_check_sum: None,
        });
        Ok(decoder)
    }

    pub fn new_unsupported(transport: AacTransport) -> Result<Self, TransportDecodeError> {
        Err(TransportDecodeError::UnsupportedTransport(transport))
    }

    pub fn transport(&self) -> AacTransport {
        self.transport
    }

    pub fn decoder(&self) -> &AacLcDecoder {
        &self.decoder
    }

    pub fn decoder_mut(&mut self) -> &mut AacLcDecoder {
        &mut self.decoder
    }

    /// Return `CStreamInfo`-equivalent configuration plus transport counters.
    pub fn stream_info(&self) -> DecoderStreamInfo {
        let mut info = self.decoder.stream_info();
        info.bit_rate = self.statistics.bit_rate;
        info.num_lost_access_units =
            i32::try_from(self.estimated_lost_access_units).unwrap_or(i32::MAX);
        info.num_total_bytes = self.statistics.total_bytes;
        info.num_bad_bytes = self.statistics.bad_bytes;
        info.num_total_access_units = self.statistics.total_access_units;
        info.num_bad_access_units = self.statistics.bad_access_units;
        let labels = rendered_channel_labels(&info.channel_labels, self.pcm_output);
        info.num_channels = labels.len();
        info.channel_indices = channel_indices_for_rendered_labels(&labels);
        info.channel_labels = labels;
        info.output_delay = info
            .output_delay
            .saturating_add(self.concealment_delay_samples(&info));
        info.output_delay = info
            .output_delay
            .saturating_add(decoder_processing_delay_samples(&info));
        if self.pcm_limiter_is_enabled() {
            info.output_delay = info
                .output_delay
                .saturating_add(self.pcm_limiter.delay_samples(info.sample_rate));
        }
        info
    }

    /// Set a decoder parameter using the numeric values of the C API.
    pub fn set_parameter(
        &mut self,
        parameter: DecoderParameter,
        value: i32,
    ) -> Result<(), TransportDecodeError> {
        match parameter {
            DecoderParameter::PcmDualChannelOutputMode => {
                self.pcm_output.dual_channel_mode = match value {
                    0 => DualChannelOutputMode::Stereo,
                    1 => DualChannelOutputMode::Channel1,
                    2 => DualChannelOutputMode::Channel2,
                    3 => DualChannelOutputMode::Mix,
                    _ => return Err(invalid_parameter(parameter, value)),
                };
            }
            DecoderParameter::PcmOutputChannelMapping => {
                self.pcm_output.channel_order = match value {
                    0 => PcmChannelOrder::Mpeg,
                    1 => PcmChannelOrder::Wav,
                    _ => return Err(invalid_parameter(parameter, value)),
                };
            }
            DecoderParameter::PcmLimiterEnable => {
                if !(-2..=1).contains(&value) {
                    return Err(invalid_parameter(parameter, value));
                }
                self.pcm_limiter_enable = value as i8;
            }
            DecoderParameter::PcmLimiterAttackTime => {
                let Ok(value) = u32::try_from(value) else {
                    return Err(invalid_parameter(parameter, value));
                };
                if !self.pcm_limiter.set_attack_ms(value) {
                    return Err(invalid_parameter(parameter, value as i32));
                }
            }
            DecoderParameter::PcmLimiterReleaseTime => {
                let Ok(value) = u32::try_from(value) else {
                    return Err(invalid_parameter(parameter, value));
                };
                if !self.pcm_limiter.set_release_ms(value) {
                    return Err(invalid_parameter(parameter, value as i32));
                }
            }
            DecoderParameter::PcmMinOutputChannels => {
                let channels = parse_output_channel_parameter(parameter, value)?;
                self.pcm_output.min_channels = channels;
                if let (Some(minimum), Some(maximum)) = (channels, self.pcm_output.max_channels) {
                    if minimum > maximum {
                        self.pcm_output.max_channels = Some(minimum);
                    }
                }
            }
            DecoderParameter::PcmMaxOutputChannels => {
                let channels = parse_output_channel_parameter(parameter, value)?;
                self.pcm_output.max_channels = channels;
                if let (Some(minimum), Some(maximum)) = (self.pcm_output.min_channels, channels) {
                    if maximum < minimum {
                        self.pcm_output.min_channels = Some(maximum);
                    }
                }
            }
            DecoderParameter::MetadataProfile => {
                self.metadata_profile = match value {
                    0 => MetadataProfile::MpegStandard,
                    1 => MetadataProfile::MpegLegacy,
                    2 => MetadataProfile::MpegLegacyPriority,
                    3 => {
                        self.metadata_expiry_ms = 550;
                        self.decoder.set_metadata_expiry_ms(550);
                        MetadataProfile::AribJapan
                    }
                    _ => return Err(invalid_parameter(parameter, value)),
                };
                self.pcm_output.metadata_profile = self.metadata_profile;
            }
            DecoderParameter::MetadataExpiryTime => {
                self.metadata_expiry_ms =
                    u32::try_from(value).map_err(|_| invalid_parameter(parameter, value))?;
                self.decoder.set_metadata_expiry_ms(self.metadata_expiry_ms);
            }
            DecoderParameter::DrcBoostFactor => {
                let Ok(value) = u8::try_from(value) else {
                    return Err(invalid_parameter(parameter, value));
                };
                if value > 127 {
                    return Err(invalid_parameter(parameter, i32::from(value)));
                }
                self.decoder.set_drc_boost_factor(value);
            }
            DecoderParameter::DrcAttenuationFactor => {
                let Ok(value) = u8::try_from(value) else {
                    return Err(invalid_parameter(parameter, value));
                };
                if value > 127 {
                    return Err(invalid_parameter(parameter, i32::from(value)));
                }
                self.decoder.set_drc_attenuation_factor(value);
            }
            DecoderParameter::DrcReferenceLevel => {
                if value >= 0 && !(40..=127).contains(&value) || value < -127 {
                    return Err(invalid_parameter(parameter, value));
                }
                self.decoder
                    .set_drc_reference_level((value >= 0).then_some(value as u8));
            }
            DecoderParameter::DrcHeavyCompression => {
                let enabled = match value {
                    0 => false,
                    1 => true,
                    _ => return Err(invalid_parameter(parameter, value)),
                };
                self.decoder.set_drc_heavy_compression(enabled);
            }
            DecoderParameter::DrcDefaultPresentationMode => {
                if !(-1..=2).contains(&value) {
                    return Err(invalid_parameter(parameter, value));
                }
                self.decoder.set_drc_default_presentation_mode(value as i8);
            }
            DecoderParameter::DrcEncoderTargetLevel => {
                let level = u8::try_from(value).map_err(|_| invalid_parameter(parameter, value))?;
                if level > 127 {
                    return Err(invalid_parameter(parameter, value));
                }
                self.decoder.set_drc_encoder_target_level(level);
            }
            DecoderParameter::UniDrcSetEffect => {
                if !(-1..=6).contains(&value) {
                    return Err(invalid_parameter(parameter, value));
                }
                self.decoder.set_uni_drc_effect(value as i8);
            }
            DecoderParameter::UniDrcAlbumMode => {
                let album_mode = match value {
                    0 => false,
                    1 => true,
                    _ => return Err(invalid_parameter(parameter, value)),
                };
                self.decoder.set_uni_drc_album_mode(album_mode);
            }
            DecoderParameter::QmfLowPower => {
                self.qmf_processing_mode = match value {
                    -1 => QmfProcessingMode::Automatic,
                    0 => QmfProcessingMode::Complex,
                    1 => QmfProcessingMode::LowPower,
                    _ => return Err(invalid_parameter(parameter, value)),
                };
                self.sync_qmf_processing_mode();
            }
            DecoderParameter::ConcealMethod => {
                let method = match value {
                    0 => ConcealmentMethod::SpectralMute,
                    1 => ConcealmentMethod::NoiseSubstitution,
                    2 => ConcealmentMethod::EnergyInterpolation,
                    _ => return Err(invalid_parameter(parameter, value)),
                };
                if method == ConcealmentMethod::EnergyInterpolation
                    && self.decoder.stream_info().audio_object_type == 42
                {
                    return Err(invalid_parameter(parameter, value));
                }
                self.concealment_method_user = Some(method);
                self.concealment_delay.reset();
                self.pending_energy_f32_losses = 0;
                self.pending_energy_i16_losses = 0;
            }
            DecoderParameter::TransportClearBuffer => self.clear_transport_buffer(),
        }
        let output_channels = {
            let info = self.decoder.stream_info();
            rendered_channel_labels(&info.channel_labels, self.pcm_output).len()
        };
        self.decoder.set_legacy_drc_output_channels(output_channels);
        Ok(())
    }

    pub fn metadata_profile(&self) -> MetadataProfile {
        self.metadata_profile
    }

    pub fn metadata_expiry_ms(&self) -> u32 {
        self.metadata_expiry_ms
    }

    pub fn qmf_processing_mode(&self) -> QmfProcessingMode {
        self.qmf_processing_mode
    }

    fn sync_qmf_processing_mode(&mut self) {
        let low_power = match self.qmf_processing_mode {
            QmfProcessingMode::Automatic => self.decoder.automatic_qmf_low_power(),
            QmfProcessingMode::Complex => false,
            QmfProcessingMode::LowPower => {
                // USAC and MPEG Surround always require complex QMF. The
                // public parameter remains stored, as in libAACdec, while the
                // operational mode is constrained by the active codec tools.
                let info = self.decoder.stream_info();
                info.audio_object_type != 42 && info.flags & STREAM_FLAG_MPS_PRESENT == 0
            }
        };
        self.decoder.set_qmf_low_power(low_power);
    }

    /// Reset all transport statistics. Codec overlap state is deliberately kept.
    pub fn clear_transport_statistics(&mut self) {
        self.statistics = TransportStatistics::default();
        self.estimated_lost_access_units = 0;
    }

    /// Discard buffered transport input, corresponding to
    /// `AAC_TPDEC_CLEAR_BUFFER`. Like libAACdec, byte/loss counters are reset
    /// while lifetime access-unit counters are retained.
    pub fn clear_transport_buffer(&mut self) {
        self.adts_input = AdtsIncrementalStream::new();
        self.adts_observed_discarded_bytes = 0;
        self.adts_pending_discarded_bits = 0;
        self.loas_input = LoasIncrementalStream::new();
        self.loas_observed_discarded_bytes = 0;
        self.statistics.total_bytes = 0;
        self.statistics.bad_bytes = 0;
        self.estimated_lost_access_units = 0;
    }

    fn pcm_limiter_is_enabled(&self) -> bool {
        match self.pcm_limiter_enable {
            1 => true,
            0 | -2 => false,
            _ => !matches!(self.decoder.stream_info().audio_object_type, 23 | 39),
        }
    }

    fn effective_concealment_method(&self) -> ConcealmentMethod {
        self.concealment_method_user.unwrap_or_else(|| {
            if matches!(self.decoder.stream_info().audio_object_type, 23 | 39 | 42) {
                ConcealmentMethod::NoiseSubstitution
            } else {
                ConcealmentMethod::EnergyInterpolation
            }
        })
    }

    fn concealment_delay_samples(&self, info: &DecoderStreamInfo) -> usize {
        if self.effective_concealment_method() == ConcealmentMethod::EnergyInterpolation {
            info.frame_size
        } else {
            0
        }
    }

    fn conceal_f32_selected(&mut self) -> Result<Vec<f32>, DecodeError> {
        match self.effective_concealment_method() {
            ConcealmentMethod::SpectralMute => self.decoder.conceal_f32_muted(),
            ConcealmentMethod::NoiseSubstitution | ConcealmentMethod::EnergyInterpolation => {
                self.decoder.conceal_f32_interleaved()
            }
        }
    }

    fn conceal_i16_selected(&mut self) -> Result<Vec<i16>, DecodeError> {
        match self.effective_concealment_method() {
            ConcealmentMethod::SpectralMute => self.decoder.conceal_fixed_muted_i16(),
            ConcealmentMethod::NoiseSubstitution | ConcealmentMethod::EnergyInterpolation => {
                self.decoder.conceal_fixed_interleaved_i16()
            }
        }
    }

    fn delayed_energy_conceal_f32(&mut self) -> Result<Vec<f32>, TransportDecodeError> {
        self.pending_energy_i16_losses = 0;
        if self.pending_energy_f32_losses != 0 {
            match self.decoder.conceal_f32_interleaved() {
                Ok(concealed) => self.replace_pending_energy_f32(concealed),
                Err(DecodeError::NoConcealmentReference) => {}
                Err(error) => return Err(error.into()),
            }
        }
        self.pending_energy_f32_losses = self.pending_energy_f32_losses.saturating_add(1);
        let info = self.decoder.stream_info();
        Ok(self.render_pcm_f32(vec![0.0; info.frame_size * info.num_channels]))
    }

    fn delayed_energy_conceal_i16(&mut self) -> Result<Vec<i16>, TransportDecodeError> {
        self.pending_energy_f32_losses = 0;
        if self.pending_energy_i16_losses != 0 {
            match self.decoder.conceal_fixed_interleaved_i16() {
                Ok(concealed) => self.replace_pending_energy_i16(concealed),
                Err(DecodeError::NoConcealmentReference) => {}
                Err(error) => return Err(error.into()),
            }
        }
        self.pending_energy_i16_losses = self.pending_energy_i16_losses.saturating_add(1);
        let info = self.decoder.stream_info();
        Ok(self.render_pcm_i16(vec![0; info.frame_size * info.num_channels]))
    }

    fn replace_pending_energy_f32(&mut self, concealed: Vec<f32>) {
        let (prepared, channels, delay) = self.prepare_pcm_f32(concealed);
        self.concealment_delay
            .replace_pending_f32(&prepared, channels, delay);
    }

    fn replace_pending_energy_i16(&mut self, concealed: Vec<i16>) {
        let (prepared, channels, delay) = self.prepare_pcm_i16(concealed);
        self.concealment_delay
            .replace_pending_i16(&prepared, channels, delay);
    }

    fn record_access_units(&mut self, bytes: usize, access_units: usize, bad: bool) {
        let bytes = bytes as u64;
        let access_units = access_units.max(1) as u64;
        self.statistics.total_bytes = self.statistics.total_bytes.saturating_add(bytes);
        if bad {
            self.statistics.bad_bytes = self.statistics.bad_bytes.saturating_add(bytes);
            self.statistics.bad_access_units = self
                .statistics
                .bad_access_units
                .saturating_add(access_units);
        } else {
            self.statistics.total_access_units = self
                .statistics
                .total_access_units
                .saturating_add(access_units);
        }

        let info = self.decoder.stream_info();
        let samples = info.frame_size.saturating_mul(access_units as usize);
        if samples != 0 {
            let rate = (bytes as u128)
                .saturating_mul(8)
                .saturating_mul(info.sample_rate as u128)
                / samples as u128;
            self.statistics.bit_rate = u32::try_from(rate).unwrap_or(u32::MAX);
        }
    }

    fn record_concealed_access_units(&mut self, bytes: usize, access_units: usize) {
        self.record_access_units(bytes, access_units, true);
        self.statistics.total_access_units = self
            .statistics
            .total_access_units
            .saturating_add(access_units.max(1) as u64);
    }

    fn record_recovered_bad_access_unit(&mut self) {
        // The failed decode already accounted its bytes and bad-AU count. A
        // successfully synthesized replacement also counts as a processed AU.
        self.statistics.total_access_units = self.statistics.total_access_units.saturating_add(1);
    }

    fn finish_access_units<T>(
        &mut self,
        bytes: usize,
        access_units: usize,
        result: Result<T, TransportDecodeError>,
    ) -> Result<T, TransportDecodeError> {
        self.record_access_units(bytes, access_units, result.is_err());
        result
    }

    fn finish_counted_access_units<T>(
        &mut self,
        bytes: usize,
        result: Result<(T, usize), TransportDecodeError>,
    ) -> Result<T, TransportDecodeError> {
        match result {
            Ok((value, access_units)) => {
                self.record_access_units(bytes, access_units, false);
                Ok(value)
            }
            Err(error) => {
                self.record_access_units(bytes, 1, true);
                Err(error)
            }
        }
    }

    fn consumed_transport_bytes(&self, input: &[u8]) -> usize {
        match self.transport {
            AacTransport::Adts => AdtsFrame::parse(input)
                .map(|frame| frame.bytes.len())
                .unwrap_or(input.len()),
            AacTransport::Loas => LoasFrame::parse(input)
                .map(|frame| frame.bytes.len())
                .unwrap_or(input.len()),
            _ => input.len(),
        }
    }

    fn is_concealable_access_unit_error(&self, input: &[u8], error: &TransportDecodeError) -> bool {
        let complete_framed_access_unit = match self.transport {
            AacTransport::Adts => AdtsFrame::parse(input).is_ok(),
            AacTransport::Loas => LoasFrame::parse(input).is_ok(),
            // The direct RAW and ADIF entry points consume exactly one access
            // unit per call.  Once any bytes were supplied, an EOF raised by
            // the codec parser therefore describes a truncated/corrupt AU,
            // not a transport buffer that may still receive a suffix.
            AacTransport::Raw | AacTransport::Adif => !input.is_empty(),
            // LATM has no outer sync frame, but a successfully parsed complete
            // AudioMuxElement proves that its embedded payload length was
            // present.  A later decoder EOF is consequently codec corruption.
            AacTransport::LatmMuxConfigPresent | AacTransport::LatmOutOfBandConfig => {
                LatmAudioMuxElement::parse_aac_lc_with_state(input, self.latm_mux_config.as_ref())
                    .is_ok()
            }
            _ => false,
        };
        matches!(error, TransportDecodeError::Decode(decode_error)
            if !decode_error.is_unexpected_eof() || complete_framed_access_unit)
    }

    fn prepare_pcm_f32(&self, input: Vec<f32>) -> (Vec<f32>, usize, usize) {
        let info = self.decoder.stream_info();
        let mut output_config = self.pcm_output;
        output_config.advanced_downmix = self.decoder.legacy_downmix_metadata();
        output_config.matrix_mixdown = self.decoder.legacy_matrix_mixdown();
        let channels = rendered_channel_labels(&info.channel_labels, output_config).len();
        let concealment_delay = self.concealment_delay_samples(&info);
        let output = render_interleaved(
            input.into_iter().map(f64::from).collect(),
            info.channel_labels,
            output_config,
        )
        .into_iter()
        .map(|sample| sample as f32)
        .collect::<Vec<_>>();
        (output, channels, concealment_delay)
    }

    fn render_pcm_f32(&mut self, input: Vec<f32>) -> Vec<f32> {
        let info = self.decoder.stream_info();
        let (mut output, channels, concealment_delay) = self.prepare_pcm_f32(input);
        self.concealment_delay
            .process_f32(&mut output, channels, concealment_delay);
        if self.pcm_limiter_is_enabled() {
            self.pcm_limiter
                .process_f32(&mut output, channels, info.sample_rate);
        }
        output
    }

    fn prepare_pcm_i16(&self, input: Vec<i16>) -> (Vec<i16>, usize, usize) {
        let info = self.decoder.stream_info();
        let mut output_config = self.pcm_output;
        output_config.advanced_downmix = self.decoder.legacy_downmix_metadata();
        output_config.matrix_mixdown = self.decoder.legacy_matrix_mixdown();
        let channels = rendered_channel_labels(&info.channel_labels, output_config).len();
        let concealment_delay = self.concealment_delay_samples(&info);
        let output = render_interleaved(
            input.into_iter().map(f64::from).collect(),
            info.channel_labels,
            output_config,
        )
        .into_iter()
        .map(|sample| sample.round().clamp(i16::MIN as f64, i16::MAX as f64) as i16)
        .collect::<Vec<_>>();
        (output, channels, concealment_delay)
    }

    fn render_pcm_i16(&mut self, input: Vec<i16>) -> Vec<i16> {
        let info = self.decoder.stream_info();
        let (mut output, channels, concealment_delay) = self.prepare_pcm_i16(input);
        self.concealment_delay
            .process_i16(&mut output, channels, concealment_delay);
        if self.pcm_limiter_is_enabled() {
            self.pcm_limiter
                .process_i16(&mut output, channels, info.sample_rate);
        }
        output
    }

    pub fn decode_interleaved_f32(
        &mut self,
        input: &[u8],
    ) -> Result<Vec<f32>, TransportDecodeError> {
        self.decode_interleaved_f32_inner(input, false)
    }

    pub fn decode_interleaved_f32_strict(
        &mut self,
        input: &[u8],
    ) -> Result<Vec<f32>, TransportDecodeError> {
        self.decode_interleaved_f32_inner(input, true)
    }

    pub fn decode_interleaved_i16(
        &mut self,
        input: &[u8],
    ) -> Result<Vec<i16>, TransportDecodeError> {
        self.decode_interleaved_i16_inner(input, false)
    }

    pub fn decode_interleaved_i16_strict(
        &mut self,
        input: &[u8],
    ) -> Result<Vec<i16>, TransportDecodeError> {
        self.decode_interleaved_i16_inner(input, true)
    }

    /// Decode or synthesize one frame with libAACdec-style operational flags.
    /// Input is ignored for `CONCEAL` and `FLUSH`, as in the C API.
    pub fn decode_interleaved_f32_with_flags(
        &mut self,
        input: &[u8],
        flags: DecodeFrameFlags,
    ) -> Result<Vec<f32>, TransportDecodeError> {
        self.prepare_decode_flags(flags)?;
        if flags.contains(DecodeFrameFlags::CONCEAL) {
            if self.effective_concealment_method() == ConcealmentMethod::EnergyInterpolation {
                let result = self.delayed_energy_conceal_f32();
                if result.is_ok() {
                    self.record_concealed_access_units(0, 1);
                }
                return result;
            }
            let result = match self.conceal_f32_selected() {
                Ok(samples) => Ok(samples),
                Err(DecodeError::NoConcealmentReference) => {
                    let info = self.decoder.stream_info();
                    Ok(vec![0.0; info.frame_size * info.num_channels])
                }
                Err(error) => Err(error.into()),
            };
            if result.is_ok() {
                self.record_concealed_access_units(0, 1);
            }
            return result.map(|samples| self.render_pcm_f32(samples));
        }
        if flags.contains(DecodeFrameFlags::FLUSH) {
            let result = self.decoder.flush_interleaved_f32().map_err(Into::into);
            if result.is_ok() {
                self.record_access_units(0, 1, false);
            }
            return result.map(|samples| self.render_pcm_f32(samples));
        }
        let decoder_checkpoint = self.decoder.clone();
        match self.decode_interleaved_f32(input) {
            Ok(samples) => Ok(samples),
            Err(error) if self.is_concealable_access_unit_error(input, &error) => {
                self.decoder = decoder_checkpoint;
                let concealed = if self.effective_concealment_method()
                    == ConcealmentMethod::EnergyInterpolation
                {
                    self.delayed_energy_conceal_f32()
                } else {
                    match self.conceal_f32_selected() {
                        Ok(samples) => Ok(self.render_pcm_f32(samples)),
                        Err(DecodeError::NoConcealmentReference) => Err(error),
                        Err(conceal_error) => Err(conceal_error.into()),
                    }
                };
                if concealed.is_ok() {
                    self.record_recovered_bad_access_unit();
                }
                concealed
            }
            Err(error) => Err(error),
        }
    }

    /// Fixed-point counterpart of [`Self::decode_interleaved_f32_with_flags`].
    pub fn decode_interleaved_i16_with_flags(
        &mut self,
        input: &[u8],
        flags: DecodeFrameFlags,
    ) -> Result<Vec<i16>, TransportDecodeError> {
        self.prepare_decode_flags(flags)?;
        if flags.contains(DecodeFrameFlags::CONCEAL) {
            if self.effective_concealment_method() == ConcealmentMethod::EnergyInterpolation {
                let result = self.delayed_energy_conceal_i16();
                if result.is_ok() {
                    self.record_concealed_access_units(0, 1);
                }
                return result;
            }
            let result = match self.conceal_i16_selected() {
                Ok(samples) => Ok(samples),
                Err(DecodeError::NoConcealmentReference) => {
                    let info = self.decoder.stream_info();
                    Ok(vec![0; info.frame_size * info.num_channels])
                }
                Err(error) => Err(error.into()),
            };
            if result.is_ok() {
                self.record_concealed_access_units(0, 1);
            }
            return result.map(|samples| self.render_pcm_i16(samples));
        }
        if flags.contains(DecodeFrameFlags::FLUSH) {
            let result = self.decoder.flush_interleaved_i16().map_err(Into::into);
            if result.is_ok() {
                self.record_access_units(0, 1, false);
            }
            return result.map(|samples| self.render_pcm_i16(samples));
        }
        let decoder_checkpoint = self.decoder.clone();
        match self.decode_interleaved_i16(input) {
            Ok(samples) => Ok(samples),
            Err(error) if self.is_concealable_access_unit_error(input, &error) => {
                self.decoder = decoder_checkpoint;
                let concealed = if self.effective_concealment_method()
                    == ConcealmentMethod::EnergyInterpolation
                {
                    self.delayed_energy_conceal_i16()
                } else {
                    match self.conceal_i16_selected() {
                        Ok(samples) => Ok(self.render_pcm_i16(samples)),
                        Err(DecodeError::NoConcealmentReference) => Err(error),
                        Err(conceal_error) => Err(conceal_error.into()),
                    }
                };
                if concealed.is_ok() {
                    self.record_recovered_bad_access_unit();
                }
                concealed
            }
            Err(error) => Err(error),
        }
    }

    fn prepare_decode_flags(
        &mut self,
        flags: DecodeFrameFlags,
    ) -> Result<(), TransportDecodeError> {
        if flags.contains(DecodeFrameFlags::INTERRUPTION) {
            self.estimated_lost_access_units = 0;
            self.pending_energy_f32_losses = 0;
            self.pending_energy_i16_losses = 0;
            self.decoder.signal_interruption()?;
            self.concealment_delay.reset();
            self.pcm_limiter.reset();
        }
        if flags.contains(DecodeFrameFlags::CLEAR_HISTORY) {
            self.decoder.clear_history()?;
            self.concealment_delay.reset();
            self.pending_energy_f32_losses = 0;
            self.pending_energy_i16_losses = 0;
            if self.decoder.stream_info().audio_object_type != 42 {
                self.pcm_limiter.reset();
            }
        }
        Ok(())
    }

    pub fn decode_raw_interleaved_f32(
        &mut self,
        input: &[u8],
    ) -> Result<Vec<f32>, TransportDecodeError> {
        self.require_transport(AacTransport::Raw)?;
        let result = self
            .decoder
            .decode_raw_data_block_interleaved_f32(input)
            .map_err(Into::into);
        let decoded = self.finish_access_units(input.len(), 1, result)?;
        Ok(self.render_pcm_f32(decoded))
    }

    pub fn decode_adts_interleaved_f32(
        &mut self,
        input: &[u8],
    ) -> Result<Vec<f32>, TransportDecodeError> {
        self.require_transport(AacTransport::Adts)?;
        let result = (|| {
            self.apply_adts_frame_config(input)?;
            Ok(self.decoder.decode_adts_frame_interleaved_f32(input)?)
        })();
        let bytes = self.consumed_transport_bytes(input);
        let decoded = self.finish_access_units(bytes, 1, result)?;
        Ok(self.render_pcm_f32(decoded))
    }

    /// Decode one byte-aligned raw AAC access unit following the ADIF header.
    pub fn decode_adif_interleaved_f32(
        &mut self,
        input: &[u8],
    ) -> Result<Vec<f32>, TransportDecodeError> {
        self.require_transport(AacTransport::Adif)?;
        let result = self.decode_adif_interleaved_f32_inner(input);
        let decoded = self.finish_access_units(input.len(), 1, result)?;
        Ok(self.render_pcm_f32(decoded))
    }

    fn decode_adif_interleaved_f32_inner(
        &mut self,
        input: &[u8],
    ) -> Result<Vec<f32>, TransportDecodeError> {
        let header_len = self.adif_header_len.take().unwrap_or(0);
        let raw = input
            .get(header_len..)
            .ok_or(TransportDecodeError::Adif(AdifError::InvalidSignature))?;
        Ok(self.decoder.decode_raw_data_block_interleaved_f32(raw)?)
    }

    /// Decode one byte-aligned raw AAC access unit following the ADIF header
    /// through the fixed-point synthesis path.
    pub fn decode_adif_interleaved_i16(
        &mut self,
        input: &[u8],
    ) -> Result<Vec<i16>, TransportDecodeError> {
        self.require_transport(AacTransport::Adif)?;
        let result = self.decode_adif_interleaved_i16_inner(input);
        let decoded = self.finish_access_units(input.len(), 1, result)?;
        Ok(self.render_pcm_i16(decoded))
    }

    fn decode_adif_interleaved_i16_inner(
        &mut self,
        input: &[u8],
    ) -> Result<Vec<i16>, TransportDecodeError> {
        let header_len = self.adif_header_len.take().unwrap_or(0);
        let raw = input
            .get(header_len..)
            .ok_or(TransportDecodeError::Adif(AdifError::InvalidSignature))?;
        Ok(self
            .decoder
            .decode_raw_data_block_fixed_interleaved_i16(raw)?)
    }

    /// Decode a LOAS frame carrying the supported LATM AAC-LC single-stream
    /// AudioMuxElement subset.
    pub fn decode_loas_interleaved_f32(
        &mut self,
        input: &[u8],
    ) -> Result<Vec<f32>, TransportDecodeError> {
        let result = self.decode_loas_interleaved_f32_inner(input, false);
        let bytes = self.consumed_transport_bytes(input);
        let decoded = self.finish_counted_access_units(bytes, result)?;
        Ok(self.render_pcm_f32(decoded))
    }

    pub fn decode_latm_interleaved_f32(
        &mut self,
        input: &[u8],
    ) -> Result<Vec<f32>, TransportDecodeError> {
        let result = self.decode_latm_interleaved_f32_inner(input, false);
        let decoded = self.finish_counted_access_units(input.len(), result)?;
        Ok(self.render_pcm_f32(decoded))
    }

    pub fn decode_latm_interleaved_f32_strict(
        &mut self,
        input: &[u8],
    ) -> Result<Vec<f32>, TransportDecodeError> {
        let result = self.decode_latm_interleaved_f32_inner(input, true);
        let decoded = self.finish_counted_access_units(input.len(), result)?;
        Ok(self.render_pcm_f32(decoded))
    }

    fn decode_latm_interleaved_f32_inner(
        &mut self,
        input: &[u8],
        strict: bool,
    ) -> Result<(Vec<f32>, usize), TransportDecodeError> {
        self.require_latm_transport()?;
        let latm = self.parse_latm(input)?;
        let mut output = Vec::new();
        let mut access_units = 0;
        for payload in latm
            .payloads
            .iter()
            .filter(|payload| payload.program == 0 && payload.layer == 0)
        {
            output.append(&mut self.decode_latm_payload_f32(&payload.data, strict)?);
            access_units += 1;
        }
        Ok((output, access_units))
    }

    pub fn decode_latm_interleaved_i16(
        &mut self,
        input: &[u8],
    ) -> Result<Vec<i16>, TransportDecodeError> {
        let result = self.decode_latm_interleaved_i16_inner(input, false);
        let decoded = self.finish_counted_access_units(input.len(), result)?;
        Ok(self.render_pcm_i16(decoded))
    }

    pub fn decode_latm_interleaved_i16_strict(
        &mut self,
        input: &[u8],
    ) -> Result<Vec<i16>, TransportDecodeError> {
        let result = self.decode_latm_interleaved_i16_inner(input, true);
        let decoded = self.finish_counted_access_units(input.len(), result)?;
        Ok(self.render_pcm_i16(decoded))
    }

    fn decode_latm_interleaved_i16_inner(
        &mut self,
        input: &[u8],
        strict: bool,
    ) -> Result<(Vec<i16>, usize), TransportDecodeError> {
        self.require_latm_transport()?;
        let latm = self.parse_latm(input)?;
        let mut output = Vec::new();
        let mut access_units = 0;
        for payload in latm
            .payloads
            .iter()
            .filter(|payload| payload.program == 0 && payload.layer == 0)
        {
            output.append(&mut self.decode_latm_payload_i16(&payload.data, strict)?);
            access_units += 1;
        }
        Ok((output, access_units))
    }

    pub fn decode_loas_interleaved_f32_strict(
        &mut self,
        input: &[u8],
    ) -> Result<Vec<f32>, TransportDecodeError> {
        let result = self.decode_loas_interleaved_f32_inner(input, true);
        let bytes = self.consumed_transport_bytes(input);
        let decoded = self.finish_counted_access_units(bytes, result)?;
        Ok(self.render_pcm_f32(decoded))
    }

    fn decode_loas_interleaved_f32_inner(
        &mut self,
        input: &[u8],
        strict: bool,
    ) -> Result<(Vec<f32>, usize), TransportDecodeError> {
        self.require_transport(AacTransport::Loas)?;
        let loas = LoasFrame::parse(input)?;
        let latm = self.parse_latm(loas.audio_mux_element)?;
        let mut output = Vec::new();
        let mut access_units = 0;
        for payload in latm
            .payloads
            .iter()
            .filter(|payload| payload.program == 0 && payload.layer == 0)
        {
            let mut pcm = self.decode_latm_payload_f32(&payload.data, strict)?;
            output.append(&mut pcm);
            access_units += 1;
        }
        Ok((output, access_units))
    }

    /// Fixed-point LOAS/LATM AAC-LC decoding.
    pub fn decode_loas_interleaved_i16(
        &mut self,
        input: &[u8],
    ) -> Result<Vec<i16>, TransportDecodeError> {
        let result = self.decode_loas_interleaved_i16_inner(input, false);
        let bytes = self.consumed_transport_bytes(input);
        let decoded = self.finish_counted_access_units(bytes, result)?;
        Ok(self.render_pcm_i16(decoded))
    }

    pub fn decode_loas_interleaved_i16_strict(
        &mut self,
        input: &[u8],
    ) -> Result<Vec<i16>, TransportDecodeError> {
        let result = self.decode_loas_interleaved_i16_inner(input, true);
        let bytes = self.consumed_transport_bytes(input);
        let decoded = self.finish_counted_access_units(bytes, result)?;
        Ok(self.render_pcm_i16(decoded))
    }

    fn decode_loas_interleaved_i16_inner(
        &mut self,
        input: &[u8],
        strict: bool,
    ) -> Result<(Vec<i16>, usize), TransportDecodeError> {
        self.require_transport(AacTransport::Loas)?;
        let loas = LoasFrame::parse(input)?;
        let latm = self.parse_latm(loas.audio_mux_element)?;
        let mut output = Vec::new();
        let mut access_units = 0;
        for payload in latm
            .payloads
            .iter()
            .filter(|payload| payload.program == 0 && payload.layer == 0)
        {
            let mut pcm = self.decode_latm_payload_i16(&payload.data, strict)?;
            output.append(&mut pcm);
            access_units += 1;
        }
        Ok((output, access_units))
    }

    /// Append a chunk of an incremental LOAS byte stream.
    pub fn push_loas_bytes(&mut self, input: &[u8]) -> Result<(), TransportDecodeError> {
        self.require_transport(AacTransport::Loas)?;
        self.loas_input.push(input);
        Ok(())
    }

    pub fn buffered_loas_bytes(&self) -> Result<usize, TransportDecodeError> {
        self.require_transport(AacTransport::Loas)?;
        Ok(self.loas_input.buffered_len())
    }

    pub fn discarded_loas_bytes(&self) -> Result<usize, TransportDecodeError> {
        self.require_transport(AacTransport::Loas)?;
        Ok(self.loas_input.discarded_bytes())
    }

    /// Decode all complete LOAS/LATM AAC-LC frames currently buffered.
    pub fn drain_loas_interleaved_f32(&mut self) -> Result<Vec<Vec<f32>>, TransportDecodeError> {
        self.require_transport(AacTransport::Loas)?;
        let mut output = Vec::new();
        while let Some(frame) = self.loas_input.next_frame() {
            self.observe_loas_discarded_bytes();
            let frame_len = frame.bytes.len();
            let result = (|| {
                let latm = self.parse_latm(frame.audio_mux_element())?;
                let mut decoded = Vec::new();
                for payload in latm
                    .payloads
                    .iter()
                    .filter(|payload| payload.program == 0 && payload.layer == 0)
                {
                    decoded.push(self.decode_latm_payload_f32(&payload.data, false)?);
                }
                let count = decoded.len();
                Ok((decoded, count))
            })();
            output.extend(self.finish_counted_access_units(frame_len, result)?);
        }
        self.observe_loas_discarded_bytes();
        Ok(output
            .into_iter()
            .map(|frame| self.render_pcm_f32(frame))
            .collect())
    }

    /// Fixed-point counterpart of [`Self::drain_loas_interleaved_f32`].
    pub fn drain_loas_interleaved_i16(&mut self) -> Result<Vec<Vec<i16>>, TransportDecodeError> {
        self.require_transport(AacTransport::Loas)?;
        let mut output = Vec::new();
        while let Some(frame) = self.loas_input.next_frame() {
            self.observe_loas_discarded_bytes();
            let frame_len = frame.bytes.len();
            let result = (|| {
                let latm = self.parse_latm(frame.audio_mux_element())?;
                let mut decoded = Vec::new();
                for payload in latm
                    .payloads
                    .iter()
                    .filter(|payload| payload.program == 0 && payload.layer == 0)
                {
                    decoded.push(self.decode_latm_payload_i16(&payload.data, false)?);
                }
                let count = decoded.len();
                Ok((decoded, count))
            })();
            output.extend(self.finish_counted_access_units(frame_len, result)?);
        }
        self.observe_loas_discarded_bytes();
        Ok(output
            .into_iter()
            .map(|frame| self.render_pcm_i16(frame))
            .collect())
    }

    fn parse_latm(&mut self, input: &[u8]) -> Result<LatmAudioMuxElement, TransportDecodeError> {
        let latm =
            LatmAudioMuxElement::parse_aac_lc_with_state(input, self.latm_mux_config.as_ref())?;
        if let Some(config) = latm.config.clone() {
            self.apply_latm_config(config)?;
        }
        if let Some(mux_config) = latm.mux_config.clone() {
            self.latm_mux_config = Some(mux_config);
        }
        Ok(latm)
    }

    fn decode_latm_payload_f32(
        &mut self,
        payload: &[u8],
        strict: bool,
    ) -> Result<Vec<f32>, TransportDecodeError> {
        if self.decoder.audio_object_type() == 42 {
            return Ok(interleave_multichannel_f32(
                &self
                    .decoder
                    .decode_usac_access_unit_multichannel_f32(payload)?,
            ));
        }
        if self.decoder.stream_info().aac_num_channels > 2 {
            let frame = if strict {
                self.decoder
                    .decode_raw_data_block_multichannel_f32_strict(payload)?
            } else {
                self.decoder
                    .decode_raw_data_block_multichannel_f32(payload)?
            };
            return Ok(frame.interleaved_f32());
        }
        if strict {
            Ok(self
                .decoder
                .decode_raw_data_block_f32_strict(payload)?
                .interleaved_f32())
        } else {
            Ok(self
                .decoder
                .decode_raw_data_block_interleaved_f32(payload)?)
        }
    }

    fn decode_latm_payload_i16(
        &mut self,
        payload: &[u8],
        strict: bool,
    ) -> Result<Vec<i16>, TransportDecodeError> {
        if self.decoder.audio_object_type() == 42 {
            return Ok(interleave_multichannel_i16(
                &self
                    .decoder
                    .decode_usac_access_unit_multichannel_f32(payload)?,
            ));
        }
        if self.decoder.stream_info().aac_num_channels > 2 {
            return if strict {
                Ok(self
                    .decoder
                    .decode_raw_data_block_multichannel_fixed_interleaved_i16_strict(payload)?)
            } else {
                Ok(self
                    .decoder
                    .decode_raw_data_block_multichannel_fixed_interleaved_i16(payload)?)
            };
        }
        if strict {
            Ok(self
                .decoder
                .decode_raw_data_block_fixed_interleaved_i16_strict(payload)?)
        } else {
            Ok(self
                .decoder
                .decode_raw_data_block_fixed_interleaved_i16(payload)?)
        }
    }

    /// Decode every raw_data_block in one CRC-protected ADTS multi-block frame.
    pub fn decode_adts_blocks_interleaved_f32(
        &mut self,
        input: &[u8],
    ) -> Result<Vec<Vec<f32>>, TransportDecodeError> {
        self.require_transport(AacTransport::Adts)?;
        let result = (|| {
            self.apply_adts_frame_config(input)?;
            let frames = self
                .decoder
                .decode_adts_frame_blocks_f32(input)?
                .into_iter()
                .map(|frame| frame.interleaved_f32())
                .collect::<Vec<_>>();
            let count = frames.len();
            Ok((frames, count))
        })();
        let bytes = self.consumed_transport_bytes(input);
        let frames = self.finish_counted_access_units(bytes, result)?;
        Ok(frames
            .into_iter()
            .map(|frame| self.render_pcm_f32(frame))
            .collect())
    }

    /// Fixed-point counterpart of [`Self::decode_adts_blocks_interleaved_f32`].
    pub fn decode_adts_blocks_interleaved_i16(
        &mut self,
        input: &[u8],
    ) -> Result<Vec<Vec<i16>>, TransportDecodeError> {
        self.require_transport(AacTransport::Adts)?;
        let result = (|| {
            self.apply_adts_frame_config(input)?;
            let frames = self
                .decoder
                .decode_adts_frame_blocks_fixed_interleaved_i16(input)?;
            let count = frames.len();
            Ok((frames, count))
        })();
        let bytes = self.consumed_transport_bytes(input);
        let frames = self.finish_counted_access_units(bytes, result)?;
        Ok(frames
            .into_iter()
            .map(|frame| self.render_pcm_i16(frame))
            .collect())
    }

    /// Append a chunk of an incremental ADTS byte stream.
    pub fn push_adts_bytes(&mut self, input: &[u8]) -> Result<(), TransportDecodeError> {
        self.require_transport(AacTransport::Adts)?;
        self.adts_input.push(input);
        Ok(())
    }

    pub fn buffered_adts_bytes(&self) -> Result<usize, TransportDecodeError> {
        self.require_transport(AacTransport::Adts)?;
        Ok(self.adts_input.buffered_len())
    }

    pub fn discarded_adts_bytes(&self) -> Result<usize, TransportDecodeError> {
        self.require_transport(AacTransport::Adts)?;
        Ok(self.adts_input.discarded_bytes())
    }

    /// Enable FDK-style CBR loss estimation for ADTS synchronization recovery.
    pub fn set_adts_average_bitrate(
        &mut self,
        average_bitrate: u32,
    ) -> Result<(), TransportDecodeError> {
        self.require_transport(AacTransport::Adts)?;
        let sample_rate = sample_rate_from_index(self.decoder.sampling_frequency_index())
            .expect("decoder construction validates the sampling-frequency index");
        self.adts_loss_estimator = Some(AccessUnitLossEstimator::new(
            average_bitrate,
            sample_rate,
            1024,
        )?);
        self.adts_pending_discarded_bits = 0;
        self.adts_observed_discarded_bytes = self.adts_input.discarded_bytes();
        self.estimated_lost_access_units = 0;
        Ok(())
    }

    pub fn estimated_lost_access_units(&self) -> Result<u64, TransportDecodeError> {
        self.require_transport(AacTransport::Adts)?;
        Ok(self.estimated_lost_access_units)
    }

    pub fn take_estimated_lost_access_units(&mut self) -> Result<u64, TransportDecodeError> {
        self.require_transport(AacTransport::Adts)?;
        Ok(std::mem::take(&mut self.estimated_lost_access_units))
    }

    pub fn enable_adts_pcm_concealment(&mut self) -> Result<(), TransportDecodeError> {
        self.require_transport(AacTransport::Adts)?;
        self.adts_pcm_concealment = Some(PcmConcealment::new());
        Ok(())
    }

    pub fn disable_adts_pcm_concealment(&mut self) -> Result<(), TransportDecodeError> {
        self.require_transport(AacTransport::Adts)?;
        self.adts_pcm_concealment = None;
        Ok(())
    }

    /// Enable fixed-spectrum concealment before IMDCT/overlap-add. This takes
    /// precedence over PCM concealment for the incremental i16 path.
    pub fn enable_adts_spectral_concealment(&mut self) -> Result<(), TransportDecodeError> {
        self.require_transport(AacTransport::Adts)?;
        self.adts_spectral_concealment = true;
        if self.adts_pcm_concealment.is_none() {
            self.adts_pcm_concealment = Some(PcmConcealment::new());
        }
        Ok(())
    }

    pub fn disable_adts_spectral_concealment(&mut self) -> Result<(), TransportDecodeError> {
        self.require_transport(AacTransport::Adts)?;
        self.adts_spectral_concealment = false;
        Ok(())
    }

    /// Decode all complete ADTS frames currently available in the incremental
    /// buffer. A multi-block ADTS frame contributes one PCM vector per raw block.
    pub fn drain_adts_interleaved_f32(&mut self) -> Result<Vec<Vec<f32>>, TransportDecodeError> {
        self.require_transport(AacTransport::Adts)?;
        let mut output = Vec::new();
        while let Some(frame) = self.adts_input.next_frame() {
            self.observe_adts_discarded_bytes();
            let lost = self.estimate_adts_loss_on_recovery(frame.bytes.len() * 8);
            let frame_bytes = frame.bytes.len();
            let blocks = frame.header.number_of_raw_data_blocks_in_frame as usize + 1;
            if let Err(error) = self.apply_adts_frame_config(&frame.bytes) {
                self.record_access_units(frame_bytes, blocks, true);
                return Err(error);
            }
            let mut spectral_concealed = Vec::new();
            if self.adts_spectral_concealment
                && self.effective_concealment_method() == ConcealmentMethod::EnergyInterpolation
                && lost == 1
                && frame.header.number_of_raw_data_blocks_in_frame == 0
            {
                let mut lookahead = self.decoder.clone();
                let next = match lookahead.decode_adts_frame_blocks_f32(&frame.bytes) {
                    Ok(_) => lookahead.f32_concealment_spectral_frame(),
                    Err(_) => None,
                };
                match next {
                    Some(next) => match self.decoder.conceal_f32_interpolated(&next) {
                        Ok(concealed) => spectral_concealed.push(concealed),
                        Err(DecodeError::NoConcealmentReference)
                        | Err(DecodeError::ConcealmentInterpolation(_)) => {}
                        Err(error) => return Err(error.into()),
                    },
                    None => {}
                }
            }
            if self.adts_spectral_concealment && spectral_concealed.is_empty() && lost != 0 {
                for _ in 0..lost {
                    match self.conceal_f32_selected() {
                        Ok(concealed) => spectral_concealed.push(concealed),
                        Err(DecodeError::NoConcealmentReference) => {
                            spectral_concealed.clear();
                            break;
                        }
                        Err(error) => return Err(error.into()),
                    }
                }
            }
            let mut candidate = self.decoder.clone();
            let decoded = match candidate.decode_adts_frame_blocks_f32(&frame.bytes) {
                Ok(decoded) => {
                    self.decoder = candidate;
                    self.record_access_units(frame_bytes, blocks, false);
                    decoded
                        .into_iter()
                        .map(|decoded| decoded.interleaved_f32())
                        .collect::<Vec<_>>()
                }
                Err(decode_error) if self.adts_spectral_concealment => {
                    self.record_concealed_access_units(frame_bytes, blocks);
                    for _ in 0..blocks {
                        match self.conceal_f32_selected() {
                            Ok(concealed) => output.push(concealed),
                            Err(DecodeError::NoConcealmentReference) => {
                                return Err(decode_error.into());
                            }
                            Err(error) => return Err(error.into()),
                        }
                    }
                    continue;
                }
                Err(error) => {
                    self.record_access_units(frame_bytes, blocks, true);
                    return Err(error.into());
                }
            };
            for (index, good) in decoded.into_iter().enumerate() {
                if index == 0 && !spectral_concealed.is_empty() {
                    output.append(&mut spectral_concealed);
                    output.push(good.clone());
                    if let Some(concealment) = &mut self.adts_pcm_concealment {
                        concealment.process_f32(good, 0);
                    }
                    continue;
                }
                if let Some(concealment) = &mut self.adts_pcm_concealment {
                    output.extend(concealment.process_f32(good, if index == 0 { lost } else { 0 }));
                } else {
                    output.push(good);
                }
            }
        }
        self.observe_adts_discarded_bytes();
        Ok(output
            .into_iter()
            .map(|frame| self.render_pcm_f32(frame))
            .collect())
    }

    /// Fixed-point counterpart of [`Self::drain_adts_interleaved_f32`].
    pub fn drain_adts_interleaved_i16(&mut self) -> Result<Vec<Vec<i16>>, TransportDecodeError> {
        self.require_transport(AacTransport::Adts)?;
        let mut output = Vec::new();
        while let Some(frame) = self.adts_input.next_frame() {
            self.observe_adts_discarded_bytes();
            let lost = self.estimate_adts_loss_on_recovery(frame.bytes.len() * 8);
            let frame_bytes = frame.bytes.len();
            let blocks = frame.header.number_of_raw_data_blocks_in_frame as usize + 1;
            if let Err(error) = self.apply_adts_frame_config(&frame.bytes) {
                self.record_access_units(frame_bytes, blocks, true);
                return Err(error);
            }
            let mut missing_without_reference = 0usize;
            if self.adts_spectral_concealment {
                let interpolated = if self.effective_concealment_method()
                    == ConcealmentMethod::EnergyInterpolation
                    && lost == 1
                    && frame.header.number_of_raw_data_blocks_in_frame == 0
                {
                    let mut lookahead = self.decoder.clone();
                    match lookahead.decode_adts_frame_blocks_fixed_interleaved_i16(&frame.bytes) {
                        Ok(_) => lookahead.fixed_concealment_spectral_frame(),
                        Err(_) => None,
                    }
                } else {
                    None
                };
                for loss_index in 0..lost {
                    let concealed = if loss_index == 0 {
                        if let Some(next) = interpolated.as_ref() {
                            self.decoder.conceal_fixed_interpolated_i16(next)
                        } else {
                            self.conceal_i16_selected()
                        }
                    } else {
                        self.conceal_i16_selected()
                    };
                    match concealed {
                        Ok(concealed) => output.push(concealed),
                        Err(DecodeError::NoConcealmentReference) => {
                            missing_without_reference += 1;
                        }
                        Err(DecodeError::ConcealmentInterpolation(_)) => {
                            match self.conceal_i16_selected() {
                                Ok(concealed) => output.push(concealed),
                                Err(error) => return Err(error.into()),
                            }
                        }
                        Err(error) => return Err(error.into()),
                    }
                }
            }
            let mut candidate = self.decoder.clone();
            let decoded =
                match candidate.decode_adts_frame_blocks_fixed_interleaved_i16(&frame.bytes) {
                    Ok(decoded) => {
                        self.decoder = candidate;
                        self.record_access_units(frame_bytes, blocks, false);
                        decoded
                    }
                    Err(decode_error) if self.adts_spectral_concealment => {
                        self.record_concealed_access_units(frame_bytes, blocks);
                        for _ in 0..blocks {
                            match self.conceal_i16_selected() {
                                Ok(concealed) => output.push(concealed),
                                Err(DecodeError::NoConcealmentReference) => {
                                    return Err(decode_error.into());
                                }
                                Err(error) => return Err(error.into()),
                            }
                        }
                        continue;
                    }
                    Err(error) => {
                        self.record_access_units(frame_bytes, blocks, true);
                        return Err(error.into());
                    }
                };
            for (index, good) in decoded.into_iter().enumerate() {
                if self.adts_spectral_concealment {
                    if index == 0 && missing_without_reference != 0 {
                        output.extend((0..missing_without_reference).map(|_| vec![0; good.len()]));
                    }
                    output.push(good);
                } else if let Some(concealment) = &mut self.adts_pcm_concealment {
                    output.extend(concealment.process_i16(good, if index == 0 { lost } else { 0 }));
                } else {
                    output.push(good);
                }
            }
        }
        self.observe_adts_discarded_bytes();
        Ok(output
            .into_iter()
            .map(|frame| self.render_pcm_i16(frame))
            .collect())
    }

    fn decode_interleaved_f32_inner(
        &mut self,
        input: &[u8],
        strict: bool,
    ) -> Result<Vec<f32>, TransportDecodeError> {
        self.pending_energy_i16_losses = 0;
        if self.effective_concealment_method() == ConcealmentMethod::EnergyInterpolation
            && self.pending_energy_f32_losses != 0
        {
            return self.recover_pending_energy_f32(input, strict);
        }
        self.decode_interleaved_f32_inner_plain(input, strict)
    }

    fn recover_pending_energy_f32(
        &mut self,
        input: &[u8],
        strict: bool,
    ) -> Result<Vec<f32>, TransportDecodeError> {
        let losses = self.pending_energy_f32_losses;
        let next = if losses == 1 {
            let mut lookahead = self.clone();
            lookahead.pending_energy_f32_losses = 0;
            lookahead
                .decode_interleaved_f32_inner_plain(input, strict)
                .ok()
                .and_then(|_| lookahead.decoder.f32_concealment_spectral_frame())
        } else {
            None
        };
        let concealed = if let Some(next) = next.as_ref() {
            match self.decoder.conceal_f32_interpolated(next) {
                Ok(concealed) => Some(concealed),
                Err(DecodeError::NoConcealmentReference)
                | Err(DecodeError::ConcealmentInterpolation(_)) => {
                    match self.decoder.conceal_f32_interleaved() {
                        Ok(concealed) => Some(concealed),
                        Err(DecodeError::NoConcealmentReference) => None,
                        Err(error) => return Err(error.into()),
                    }
                }
                Err(error) => return Err(error.into()),
            }
        } else {
            match self.decoder.conceal_f32_interleaved() {
                Ok(concealed) => Some(concealed),
                Err(DecodeError::NoConcealmentReference) => None,
                Err(error) => return Err(error.into()),
            }
        };
        if let Some(concealed) = concealed {
            self.replace_pending_energy_f32(concealed);
        }
        self.pending_energy_f32_losses = 0;
        self.decode_interleaved_f32_inner_plain(input, strict)
    }

    fn decode_interleaved_f32_inner_plain(
        &mut self,
        input: &[u8],
        strict: bool,
    ) -> Result<Vec<f32>, TransportDecodeError> {
        let result = (|| {
            if self.transport == AacTransport::Adts {
                self.apply_adts_frame_config(input)?;
            }
            Ok(match self.transport {
                AacTransport::Raw if strict => (
                    self.decoder
                        .decode_raw_data_block_f32_strict(input)?
                        .interleaved_f32(),
                    1,
                ),
                AacTransport::Raw => (
                    self.decoder
                        .decode_raw_data_block_f32(input)?
                        .interleaved_f32(),
                    1,
                ),
                AacTransport::Adts if strict => (
                    self.decoder
                        .decode_adts_frame_f32_strict(input)?
                        .interleaved_f32(),
                    1,
                ),
                AacTransport::Adts => (
                    self.decoder.decode_adts_frame_f32(input)?.interleaved_f32(),
                    1,
                ),
                AacTransport::Adif => (self.decode_adif_interleaved_f32_inner(input)?, 1),
                AacTransport::LatmMuxConfigPresent | AacTransport::LatmOutOfBandConfig => {
                    self.decode_latm_interleaved_f32_inner(input, strict)?
                }
                AacTransport::Loas => self.decode_loas_interleaved_f32_inner(input, strict)?,
                AacTransport::Drm => (self.decode_drm_interleaved_f32_inner(input)?, 1),
            })
        })();
        let bytes = self.consumed_transport_bytes(input);
        let decoded = self.finish_counted_access_units(bytes, result)?;
        Ok(self.render_pcm_f32(decoded))
    }

    fn decode_interleaved_i16_inner(
        &mut self,
        input: &[u8],
        strict: bool,
    ) -> Result<Vec<i16>, TransportDecodeError> {
        self.pending_energy_f32_losses = 0;
        if self.effective_concealment_method() == ConcealmentMethod::EnergyInterpolation
            && self.pending_energy_i16_losses != 0
        {
            return self.recover_pending_energy_i16(input, strict);
        }
        self.decode_interleaved_i16_inner_plain(input, strict)
    }

    fn recover_pending_energy_i16(
        &mut self,
        input: &[u8],
        strict: bool,
    ) -> Result<Vec<i16>, TransportDecodeError> {
        let losses = self.pending_energy_i16_losses;
        let next = if losses == 1 {
            let mut lookahead = self.clone();
            lookahead.pending_energy_i16_losses = 0;
            lookahead
                .decode_interleaved_i16_inner_plain(input, strict)
                .ok()
                .and_then(|_| lookahead.decoder.fixed_concealment_spectral_frame())
        } else {
            None
        };
        let concealed = if let Some(next) = next.as_ref() {
            match self.decoder.conceal_fixed_interpolated_i16(next) {
                Ok(concealed) => Some(concealed),
                Err(DecodeError::NoConcealmentReference)
                | Err(DecodeError::ConcealmentInterpolation(_)) => {
                    match self.decoder.conceal_fixed_interleaved_i16() {
                        Ok(concealed) => Some(concealed),
                        Err(DecodeError::NoConcealmentReference) => None,
                        Err(error) => return Err(error.into()),
                    }
                }
                Err(error) => return Err(error.into()),
            }
        } else {
            match self.decoder.conceal_fixed_interleaved_i16() {
                Ok(concealed) => Some(concealed),
                Err(DecodeError::NoConcealmentReference) => None,
                Err(error) => return Err(error.into()),
            }
        };
        if let Some(concealed) = concealed {
            self.replace_pending_energy_i16(concealed);
        }
        self.pending_energy_i16_losses = 0;
        self.decode_interleaved_i16_inner_plain(input, strict)
    }

    fn decode_interleaved_i16_inner_plain(
        &mut self,
        input: &[u8],
        strict: bool,
    ) -> Result<Vec<i16>, TransportDecodeError> {
        let result = (|| {
            if self.transport == AacTransport::Adts {
                self.apply_adts_frame_config(input)?;
            }
            Ok(match self.transport {
                AacTransport::Raw if strict => (
                    self.decoder
                        .decode_raw_data_block_fixed_interleaved_i16_strict(input)?,
                    1,
                ),
                AacTransport::Raw => (
                    self.decoder
                        .decode_raw_data_block_fixed_interleaved_i16(input)?,
                    1,
                ),
                AacTransport::Adts if strict => (
                    self.decoder
                        .decode_adts_frame_fixed_interleaved_i16_strict(input)?,
                    1,
                ),
                AacTransport::Adts => (
                    self.decoder
                        .decode_adts_frame_fixed_interleaved_i16(input)?,
                    1,
                ),
                AacTransport::Adif => (self.decode_adif_interleaved_i16_inner(input)?, 1),
                AacTransport::LatmMuxConfigPresent | AacTransport::LatmOutOfBandConfig => {
                    self.decode_latm_interleaved_i16_inner(input, strict)?
                }
                AacTransport::Loas => self.decode_loas_interleaved_i16_inner(input, strict)?,
                AacTransport::Drm => (self.decode_drm_interleaved_i16_inner(input)?, 1),
            })
        })();
        let bytes = self.consumed_transport_bytes(input);
        let decoded = self.finish_counted_access_units(bytes, result)?;
        Ok(self.render_pcm_i16(decoded))
    }

    fn decode_drm_interleaved_f32_inner(
        &mut self,
        input: &[u8],
    ) -> Result<Vec<f32>, TransportDecodeError> {
        if let Some(drm) = self.drm_xhe_decoder.as_mut() {
            let samples = drm.decode_interleaved_f32(input)?;
            self.decoder = drm.decoder.clone();
            return Ok(samples);
        }
        let drm = self
            .drm_decoder
            .as_mut()
            .ok_or(TransportDecodeError::UnsupportedTransport(
                AacTransport::Drm,
            ))?;
        let samples = if drm.drm_config.sbr {
            drm.decode_crc_protected_interleaved_f32_rendering_sbr(input, input.len() * 8)?
                .samples
        } else {
            drm.decode_crc_protected_interleaved_f32(input)?
        };
        self.decoder = drm.decoder.clone();
        Ok(samples)
    }

    fn decode_drm_interleaved_i16_inner(
        &mut self,
        input: &[u8],
    ) -> Result<Vec<i16>, TransportDecodeError> {
        if let Some(drm) = self.drm_xhe_decoder.as_mut() {
            let samples = drm.decode_interleaved_i16(input)?;
            self.decoder = drm.decoder.clone();
            return Ok(samples);
        }
        let drm = self
            .drm_decoder
            .as_mut()
            .ok_or(TransportDecodeError::UnsupportedTransport(
                AacTransport::Drm,
            ))?;
        let samples = if drm.drm_config.sbr {
            drm.decode_crc_protected_interleaved_f32_rendering_sbr(input, input.len() * 8)?
                .samples
                .into_iter()
                .map(f32_to_i16)
                .collect()
        } else {
            drm.decode_crc_protected_interleaved_i16(input)?
        };
        self.decoder = drm.decoder.clone();
        Ok(samples)
    }

    fn require_transport(&self, requested: AacTransport) -> Result<(), TransportDecodeError> {
        if self.transport == requested {
            Ok(())
        } else {
            Err(TransportDecodeError::TransportMismatch {
                configured: self.transport,
                requested,
            })
        }
    }

    fn require_latm_transport(&self) -> Result<(), TransportDecodeError> {
        if matches!(
            self.transport,
            AacTransport::LatmMuxConfigPresent | AacTransport::LatmOutOfBandConfig
        ) {
            Ok(())
        } else {
            Err(TransportDecodeError::TransportMismatch {
                configured: self.transport,
                requested: AacTransport::LatmMuxConfigPresent,
            })
        }
    }

    fn apply_latm_config(
        &mut self,
        config: AudioSpecificConfig,
    ) -> Result<(), TransportDecodeError> {
        if config.sampling_frequency_index != self.decoder.sampling_frequency_index()
            || config.channel_configuration != self.decoder.channel_configuration()
            || config.audio_object_type != self.decoder.audio_object_type()
        {
            self.decoder = AacLcDecoder::from_audio_specific_config(&config)?;
            self.sync_qmf_processing_mode();
        }
        Ok(())
    }

    fn apply_adts_frame_config(&mut self, input: &[u8]) -> Result<(), TransportDecodeError> {
        let header = AdtsHeader::parse(input)?;
        if header.sampling_frequency_index != self.decoder.sampling_frequency_index()
            || header.channel_configuration != self.decoder.channel_configuration()
            || header.profile + 1 != self.decoder.audio_object_type()
        {
            let average_bitrate = self
                .adts_loss_estimator
                .as_ref()
                .map(AccessUnitLossEstimator::average_bitrate);
            self.decoder = AacLcDecoder::from_adts_header(header)?;
            self.sync_qmf_processing_mode();
            if self.adts_pcm_concealment.is_some() {
                self.adts_pcm_concealment = Some(PcmConcealment::new());
            }
            if let Some(average_bitrate) = average_bitrate {
                let sample_rate = header
                    .sample_rate()
                    .expect("ADTS parsing validates the sampling-frequency index");
                self.adts_loss_estimator = Some(
                    AccessUnitLossEstimator::new(average_bitrate, sample_rate, 1024)
                        .expect("an existing estimator has valid non-zero parameters"),
                );
            }
        }
        Ok(())
    }

    fn observe_adts_discarded_bytes(&mut self) {
        let discarded = self.adts_input.discarded_bytes();
        let delta = discarded.saturating_sub(self.adts_observed_discarded_bytes);
        self.adts_pending_discarded_bits = self
            .adts_pending_discarded_bits
            .saturating_add(delta.saturating_mul(8));
        self.statistics.total_bytes = self.statistics.total_bytes.saturating_add(delta as u64);
        self.statistics.bad_bytes = self.statistics.bad_bytes.saturating_add(delta as u64);
        self.adts_observed_discarded_bytes = discarded;
    }

    fn observe_loas_discarded_bytes(&mut self) {
        let discarded = self.loas_input.discarded_bytes();
        let delta = discarded.saturating_sub(self.loas_observed_discarded_bytes);
        self.statistics.total_bytes = self.statistics.total_bytes.saturating_add(delta as u64);
        self.statistics.bad_bytes = self.statistics.bad_bytes.saturating_add(delta as u64);
        self.loas_observed_discarded_bytes = discarded;
    }

    fn estimate_adts_loss_on_recovery(&mut self, current_frame_bits: usize) -> usize {
        if self.adts_pending_discarded_bits == 0 {
            return 0;
        }
        let mut lost = 0usize;
        if let Some(estimator) = &mut self.adts_loss_estimator {
            lost = estimator.recovered(self.adts_pending_discarded_bits, current_frame_bits, 0)
                as usize;
            self.estimated_lost_access_units =
                self.estimated_lost_access_units.saturating_add(lost as u64);
        }
        self.adts_pending_discarded_bits = 0;
        lost
    }
}

fn invalid_parameter(parameter: DecoderParameter, value: i32) -> TransportDecodeError {
    TransportDecodeError::InvalidParameterValue { parameter, value }
}

fn decoder_processing_delay_samples(info: &DecoderStreamInfo) -> usize {
    let sbr = info.flags & STREAM_FLAG_SBR_PRESENT != 0;
    let mps = info.flags & STREAM_FLAG_MPS_PRESENT != 0;
    let low_delay = matches!(info.audio_object_type, 23 | 39);
    let usac = info.audio_object_type == 42;
    let dual_rate = info.sample_rate != info.aac_sample_rate;
    let mut delay = 0usize;

    if sbr {
        if low_delay {
            delay += if dual_rate { 64 } else { 32 };
            if mps {
                delay += 32;
            }
        } else if !usac {
            delay += if dual_rate { 962 } else { 481 };
            if mps {
                delay = delay.saturating_sub(257);
            }
        }
    }
    if mps {
        if low_delay {
            delay += 256;
        } else if !usac {
            delay += 320 + 257;
            if !sbr {
                delay += 320 + 384;
            }
        }
    }
    delay
}

fn parse_output_channel_parameter(
    parameter: DecoderParameter,
    value: i32,
) -> Result<Option<usize>, TransportDecodeError> {
    match value {
        -1 | 0 => Ok(None),
        1 | 2 | 6 | 8 => Ok(Some(value as usize)),
        _ => Err(invalid_parameter(parameter, value)),
    }
}

#[derive(Debug)]
struct PcmPlanes {
    labels: Vec<ChannelLabel>,
    samples: Vec<Vec<f64>>,
}

fn render_interleaved(
    input: Vec<f64>,
    labels: Vec<ChannelLabel>,
    config: PcmOutputConfig,
) -> Vec<f64> {
    if labels.is_empty() || !input.len().is_multiple_of(labels.len()) {
        return input;
    }
    let frames = input.len() / labels.len();
    let mut planes = vec![vec![0.0; frames]; labels.len()];
    for (frame, samples) in input.chunks_exact(labels.len()).enumerate() {
        for (channel, &sample) in samples.iter().enumerate() {
            planes[channel][frame] = sample;
        }
    }
    let rendered = render_planes(
        PcmPlanes {
            labels,
            samples: planes,
        },
        config,
    );
    let mut output = Vec::with_capacity(frames * rendered.labels.len());
    for frame in 0..frames {
        for channel in &rendered.samples {
            output.push(channel[frame]);
        }
    }
    output
}

fn rendered_channel_labels(labels: &[ChannelLabel], config: PcmOutputConfig) -> Vec<ChannelLabel> {
    let mut active_config = config;
    active_config.min_channels = None;
    active_config.channel_order = PcmChannelOrder::Mpeg;
    let active = render_planes(
        PcmPlanes {
            labels: labels.to_vec(),
            samples: vec![Vec::new(); labels.len()],
        },
        active_config,
    )
    .labels;
    let rendered = render_planes(
        PcmPlanes {
            labels: labels.to_vec(),
            samples: vec![Vec::new(); labels.len()],
        },
        config,
    )
    .labels;

    if active == [ChannelLabel::FrontCenter]
        && rendered.len() == 2
        && rendered.contains(&ChannelLabel::FrontLeft)
        && rendered.contains(&ChannelLabel::FrontRight)
    {
        return rendered;
    }

    rendered
        .into_iter()
        .map(|label| {
            if active.contains(&label) {
                label
            } else {
                ChannelLabel::Empty
            }
        })
        .collect()
}

fn render_planes(mut planes: PcmPlanes, config: PcmOutputConfig) -> PcmPlanes {
    if planes.labels.len() == 2 {
        match config.dual_channel_mode {
            DualChannelOutputMode::Stereo => {}
            DualChannelOutputMode::Channel1 => planes.samples[1] = planes.samples[0].clone(),
            DualChannelOutputMode::Channel2 => planes.samples[0] = planes.samples[1].clone(),
            DualChannelOutputMode::Mix => {
                let mixed = mix_vectors(&planes.samples[0], 0.5, &planes.samples[1], 0.5);
                planes.samples = vec![mixed.clone(), mixed];
            }
        }
    }

    if let Some(maximum) = config.max_channels {
        if planes.labels.len() > maximum {
            planes = downmix_planes(planes, maximum, config);
        }
    }
    if let Some(minimum) = config.min_channels {
        if planes.labels.len() < minimum {
            planes = extend_planes(planes, minimum);
        }
    }
    if config.channel_order == PcmChannelOrder::Wav {
        reorder_wav(&mut planes);
    }
    planes
}

fn downmix_planes(mut planes: PcmPlanes, target: usize, config: PcmOutputConfig) -> PcmPlanes {
    // libPCMutils always reduces configurations above six channels to the
    // canonical 3/2+LFE layout first.  The second-stage mono/stereo equations
    // must consume that intermediate result so the A/B and gain-5 metadata is
    // not bypassed.
    if planes.labels.len() > 6 && target <= 6 {
        planes = downmix_six(planes, config);
    }
    match target {
        1 => {
            let mono = legacy_matrix_mono(&planes, config).unwrap_or_else(|| {
                let stereo = downmix_stereo(&planes, config);
                mix_vectors(&stereo[0], 0.5, &stereo[1], 0.5)
            });
            PcmPlanes {
                labels: vec![ChannelLabel::FrontCenter],
                samples: vec![mono],
            }
        }
        2 => PcmPlanes {
            labels: vec![ChannelLabel::FrontLeft, ChannelLabel::FrontRight],
            samples: downmix_stereo(&planes, config),
        },
        6 => planes,
        _ => planes,
    }
}

fn legacy_matrix_mono(planes: &PcmPlanes, config: PcmOutputConfig) -> Option<Vec<f64>> {
    let matrix = config.matrix_mixdown?;
    let advanced_levels_present = config.advanced_downmix.is_some_and(|metadata| {
        metadata.center_mix_level_index.is_some() || metadata.surround_mix_level_index.is_some()
    });
    let selected = config.metadata_profile == MetadataProfile::MpegLegacyPriority
        || (!advanced_levels_present && config.metadata_profile == MetadataProfile::MpegLegacy);
    if !selected
        || !planes.labels.contains(&ChannelLabel::FrontCenter)
        || !planes
            .labels
            .iter()
            .any(|label| matches!(label, ChannelLabel::SideLeft | ChannelLabel::BackLeft))
        || !planes
            .labels
            .iter()
            .any(|label| matches!(label, ChannelLabel::SideRight | ChannelLabel::BackRight))
    {
        return None;
    }
    let index = usize::from(matrix.index.min(3));
    let common = [0.226_540_920, 0.25, 0.269_752_143, 0.333_333_333][index];
    let surround = common * [std::f64::consts::FRAC_1_SQRT_2, 0.5, 0.353_553_39, 0.0][index];
    let frames = planes.samples.first().map_or(0, Vec::len);
    let mut mono = vec![0.0; frames];
    for (channel, label) in planes.samples.iter().zip(&planes.labels) {
        let gain = match label {
            ChannelLabel::FrontCenter | ChannelLabel::FrontLeft | ChannelLabel::FrontRight => {
                common
            }
            ChannelLabel::SideLeft
            | ChannelLabel::SideRight
            | ChannelLabel::BackLeft
            | ChannelLabel::BackRight => surround,
            _ => 0.0,
        };
        add_scaled(&mut mono, channel, gain);
    }
    Some(mono)
}

fn downmix_stereo(planes: &PcmPlanes, config: PcmOutputConfig) -> Vec<Vec<f64>> {
    let frames = planes.samples.first().map_or(0, Vec::len);
    let mut left = vec![0.0; frames];
    let mut right = vec![0.0; frames];
    let root_half = std::f64::consts::FRAC_1_SQRT_2;
    let is_three_two = planes.labels.contains(&ChannelLabel::FrontCenter)
        && planes
            .labels
            .iter()
            .any(|label| matches!(label, ChannelLabel::SideLeft | ChannelLabel::BackLeft))
        && planes
            .labels
            .iter()
            .any(|label| matches!(label, ChannelLabel::SideRight | ChannelLabel::BackRight));
    let advanced_levels_present = config.advanced_downmix.is_some_and(|metadata| {
        metadata.center_mix_level_index.is_some() || metadata.surround_mix_level_index.is_some()
    });
    let matrix_mixdown_selected = config.matrix_mixdown.is_some()
        && (config.metadata_profile == MetadataProfile::MpegLegacyPriority
            || (!advanced_levels_present
                && config.metadata_profile == MetadataProfile::MpegLegacy));
    let arib = config.metadata_profile == MetadataProfile::AribJapan && !advanced_levels_present;
    let metadata = config.advanced_downmix.unwrap_or_default();
    let mut cross_surround = false;
    let (front_gain, center_gain, surround_gain, lfe_gain) = if is_three_two {
        if let Some(matrix) = config.matrix_mixdown.filter(|_| matrix_mixdown_selected) {
            let index = usize::from(matrix.index.min(3));
            let coefficient = [root_half, 0.5, 0.353_553_39, 0.0][index];
            cross_surround = matrix.pseudo_surround_enable;
            let front = if cross_surround {
                [0.320_377_241, 0.369_398_062, 0.414_213_562, 0.585_786_438][index]
            } else {
                [0.414_213_562, 0.453_081_839, 0.485_281_374, 0.585_786_438][index]
            };
            (front, front * root_half, front * coefficient, 0.0)
        } else if arib {
            cross_surround = metadata.pseudo_surround;
            (root_half, 0.5, 0.5, 0.0)
        } else {
            cross_surround = metadata.pseudo_surround;
            (
                1.0,
                advanced_ab_mix_level(metadata.center_mix_level_index.unwrap_or(2)),
                advanced_ab_mix_level(metadata.surround_mix_level_index.unwrap_or(2)),
                advanced_lfe_mix_level(metadata.lfe_mix_level_index.unwrap_or(15)),
            )
        }
    } else {
        (1.0, root_half, root_half, 0.5)
    };
    let pseudo_surround = is_three_two && cross_surround;
    for (channel, label) in planes.samples.iter().zip(&planes.labels) {
        let (left_gain, right_gain) = match label {
            ChannelLabel::Empty => (0.0, 0.0),
            ChannelLabel::FrontLeft => (front_gain, 0.0),
            ChannelLabel::FrontRight => (0.0, front_gain),
            ChannelLabel::FrontLeftCenter => (root_half, 0.0),
            ChannelLabel::FrontRightCenter => (0.0, root_half),
            ChannelLabel::FrontCenter => (center_gain, center_gain),
            ChannelLabel::SideLeft | ChannelLabel::BackLeft if pseudo_surround => {
                (-surround_gain, surround_gain)
            }
            ChannelLabel::SideRight | ChannelLabel::BackRight if pseudo_surround => {
                (-surround_gain, surround_gain)
            }
            ChannelLabel::SideLeft | ChannelLabel::BackLeft => (surround_gain, 0.0),
            ChannelLabel::SideRight | ChannelLabel::BackRight => (0.0, surround_gain),
            ChannelLabel::BackCenter => (0.5, 0.5),
            ChannelLabel::Lfe => (lfe_gain, lfe_gain),
            ChannelLabel::Unknown(index) if index % 2 == 0 => (root_half, 0.0),
            ChannelLabel::Unknown(_) => (0.0, root_half),
        };
        add_scaled(&mut left, channel, left_gain);
        add_scaled(&mut right, channel, right_gain);
    }
    if is_three_two && !arib && !matrix_mixdown_selected {
        let gain = advanced_downmix_gain(metadata.stereo_downmix_gain_index.unwrap_or(0));
        if gain != 1.0 {
            for sample in left.iter_mut().chain(&mut right) {
                *sample *= gain;
            }
        }
    }
    vec![left, right]
}

fn advanced_ab_mix_level(index: u8) -> f64 {
    const LEVELS: [f64; 8] = [1.0, 0.841, 0.707, 0.596, 0.5, 0.422, 0.355, 0.0];
    LEVELS[usize::from(index.min(7))]
}

fn advanced_lfe_mix_level(index: u8) -> f64 {
    const LEVELS: [f64; 16] = [
        3.162, 2.0, 1.679, 1.413, 1.189, 1.0, 0.841, 0.707, 0.596, 0.5, 0.316, 0.178, 0.1, 0.032,
        0.01, 0.0,
    ];
    LEVELS[usize::from(index.min(15))]
}

fn advanced_downmix_gain(index: u8) -> f64 {
    let magnitude = f64::from(index & 0x3f);
    let signed = if index & 0x40 == 0 {
        magnitude
    } else {
        -magnitude
    };
    10.0f64.powf(signed / 80.0)
}

fn downmix_six(planes: PcmPlanes, config: PcmOutputConfig) -> PcmPlanes {
    let frames = planes.samples.first().map_or(0, Vec::len);
    let targets = [
        ChannelLabel::FrontCenter,
        ChannelLabel::FrontLeft,
        ChannelLabel::FrontRight,
        ChannelLabel::BackLeft,
        ChannelLabel::BackRight,
        ChannelLabel::Lfe,
    ];
    let mut output = vec![vec![0.0; frames]; targets.len()];
    let metadata = config.advanced_downmix.unwrap_or_default();
    let mix_a = advanced_ab_mix_level(metadata.downmix_a_index.unwrap_or(2));
    let mix_b = advanced_ab_mix_level(metadata.downmix_b_index.unwrap_or(2));
    let gain = advanced_downmix_gain(metadata.five_channel_downmix_gain_index.unwrap_or(0));
    let five_front = planes.labels.contains(&ChannelLabel::FrontLeftCenter)
        && planes.labels.contains(&ChannelLabel::FrontRightCenter);
    for (source, label) in planes.samples.iter().zip(&planes.labels) {
        match label {
            // MPEG-4 channelConfiguration 7 is 5/0/2.1.  Lc/Rc contribute
            // to both the corresponding main front and the center channel.
            ChannelLabel::FrontLeftCenter => {
                add_scaled(&mut output[0], source, mix_a);
                add_scaled(&mut output[1], source, mix_b);
            }
            ChannelLabel::FrontRightCenter => {
                add_scaled(&mut output[0], source, mix_a);
                add_scaled(&mut output[2], source, mix_b);
            }
            // PCE/configuration-12 style 3/0/4.1: fold side and surround-back
            // pairs with the same A/B metadata into the 3/2 intermediate.
            ChannelLabel::SideLeft if !five_front => add_scaled(&mut output[3], source, mix_a),
            ChannelLabel::SideRight if !five_front => add_scaled(&mut output[4], source, mix_a),
            ChannelLabel::BackLeft if !five_front => add_scaled(&mut output[3], source, mix_b),
            ChannelLabel::BackRight if !five_front => add_scaled(&mut output[4], source, mix_b),
            ChannelLabel::BackCenter => {
                add_scaled(&mut output[3], source, 0.5);
                add_scaled(&mut output[4], source, 0.5);
            }
            _ => {
                if let Some(target) = targets.iter().position(|target| target == label) {
                    add_scaled(&mut output[target], source, 1.0);
                }
            }
        }
    }
    if gain != 1.0 {
        for sample in output.iter_mut().flatten() {
            *sample *= gain;
        }
    }
    PcmPlanes {
        labels: targets.to_vec(),
        samples: output,
    }
}

fn extend_planes(planes: PcmPlanes, target: usize) -> PcmPlanes {
    let targets = target_labels(target);
    if targets.is_empty() {
        return planes;
    }
    let frames = planes.samples.first().map_or(0, Vec::len);
    let mut output = vec![vec![0.0; frames]; targets.len()];
    if planes.labels == [ChannelLabel::FrontCenter] && target == 2 {
        output[0] = planes.samples[0].clone();
        output[1] = planes.samples[0].clone();
    } else {
        for (source, label) in planes.samples.into_iter().zip(planes.labels) {
            if let Some(target) = targets.iter().position(|target| *target == label) {
                output[target] = source;
            }
        }
    }
    PcmPlanes {
        labels: targets,
        samples: output,
    }
}

fn target_labels(channels: usize) -> Vec<ChannelLabel> {
    match channels {
        1 => vec![ChannelLabel::FrontCenter],
        2 => vec![ChannelLabel::FrontLeft, ChannelLabel::FrontRight],
        6 => vec![
            ChannelLabel::FrontCenter,
            ChannelLabel::FrontLeft,
            ChannelLabel::FrontRight,
            ChannelLabel::BackLeft,
            ChannelLabel::BackRight,
            ChannelLabel::Lfe,
        ],
        8 => vec![
            ChannelLabel::FrontCenter,
            ChannelLabel::FrontLeft,
            ChannelLabel::FrontRight,
            ChannelLabel::SideLeft,
            ChannelLabel::SideRight,
            ChannelLabel::BackLeft,
            ChannelLabel::BackRight,
            ChannelLabel::Lfe,
        ],
        _ => Vec::new(),
    }
}

fn reorder_wav(planes: &mut PcmPlanes) {
    let mut order = (0..planes.labels.len()).collect::<Vec<_>>();
    order.sort_by_key(|&index| wav_channel_rank(planes.labels[index]));
    planes.labels = order.iter().map(|&index| planes.labels[index]).collect();
    planes.samples = order
        .into_iter()
        .map(|index| planes.samples[index].clone())
        .collect();
}

fn wav_channel_rank(label: ChannelLabel) -> (u8, usize) {
    match label {
        ChannelLabel::FrontLeft => (0, 0),
        ChannelLabel::FrontRight => (1, 0),
        ChannelLabel::FrontCenter => (2, 0),
        ChannelLabel::Lfe => (3, 0),
        ChannelLabel::BackLeft => (4, 0),
        ChannelLabel::BackRight => (5, 0),
        ChannelLabel::BackCenter => (6, 0),
        ChannelLabel::FrontLeftCenter => (7, 0),
        ChannelLabel::FrontRightCenter => (8, 0),
        ChannelLabel::SideLeft => (9, 0),
        ChannelLabel::SideRight => (10, 0),
        ChannelLabel::Empty => (11, 0),
        ChannelLabel::Unknown(index) => (11, index),
    }
}

fn channel_indices_for_rendered_labels(labels: &[ChannelLabel]) -> Vec<u8> {
    let five_front = labels.contains(&ChannelLabel::FrontLeftCenter)
        && labels.contains(&ChannelLabel::FrontRightCenter);
    labels
        .iter()
        .map(|label| match label {
            ChannelLabel::Empty => 0,
            ChannelLabel::FrontCenter => 0,
            ChannelLabel::FrontLeft => {
                if five_front {
                    3
                } else {
                    1
                }
            }
            ChannelLabel::FrontRight => {
                if five_front {
                    4
                } else {
                    2
                }
            }
            ChannelLabel::FrontLeftCenter => 1,
            ChannelLabel::FrontRightCenter => 2,
            ChannelLabel::SideLeft | ChannelLabel::BackLeft | ChannelLabel::Lfe => 0,
            ChannelLabel::SideRight | ChannelLabel::BackRight => 1,
            ChannelLabel::BackCenter => 2,
            ChannelLabel::Unknown(index) => u8::try_from(*index).unwrap_or(u8::MAX),
        })
        .collect()
}

fn add_scaled(output: &mut [f64], input: &[f64], gain: f64) {
    for (output, input) in output.iter_mut().zip(input) {
        *output += *input * gain;
    }
}

fn mix_vectors(left: &[f64], left_gain: f64, right: &[f64], right_gain: f64) -> Vec<f64> {
    left.iter()
        .zip(right)
        .map(|(&left, &right)| left * left_gain + right * right_gain)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aac_encoder::PureRustAacLcMonoEncoder;
    use crate::adts::AdtsHeader;
    use crate::asc::{
        GaSpecificConfig, ProgramConfig, ProgramElement, UsacConfig, UsacElementConfig,
    };
    use crate::bits::BitWriter;
    use crate::decoder::ConcealmentState;
    use crate::drc::{
        ChannelLayout, DrcInstruction, DrcSelectionRequest, UniDrcConfig, UniDrcGain,
    };
    use crate::raw::ElementId;
    use crate::section::ZERO_HCB;

    fn zero_sce_payload() -> Vec<u8> {
        let mut writer = BitWriter::new();
        writer.write(ElementId::SingleChannel.bits() as u32, 3);
        writer.write(0, 4);
        writer.write(100, 8);
        writer.write_bool(false);
        writer.write(0, 2);
        writer.write_bool(false);
        writer.write(1, 6);
        writer.write_bool(false);
        writer.write(ZERO_HCB as u32, 4);
        writer.write(1, 5);
        writer.write_bool(false);
        writer.write_bool(false);
        writer.write_bool(false);
        writer.finish()
    }

    fn configure_missing_coefficient_drc(decoder: &mut PureRustTransportDecoder) {
        decoder.decoder_mut().configure_drc(
            UniDrcConfig {
                sample_rate: None,
                channel_layout: ChannelLayout {
                    base_channel_count: 1,
                    defined_layout: None,
                    speaker_positions: Vec::new(),
                },
                downmix_instructions: Vec::new(),
                coefficients: Vec::new(),
                instructions: vec![DrcInstruction {
                    drc_set_id: 1,
                    complexity_level: 0,
                    drc_location: 1,
                    downmix_ids: vec![0],
                    apply_to_downmix: false,
                    effect: 0,
                    limiter_peak_target_db: None,
                    target_loudness_upper: None,
                    target_loudness_lower: None,
                    depends_on_drc_set: None,
                    no_independent_use: false,
                    requires_eq: false,
                    channel_count: 1,
                    gain_set_index_per_channel: vec![0],
                    gain_modifications: Vec::new(),
                    gain_modifications_per_band: Vec::new(),
                    ducking_modifications: Vec::new(),
                }],
                extension_present: false,
                extensions: Vec::new(),
                bits_read: 0,
            },
            DrcSelectionRequest::default(),
        );
        decoder.decoder_mut().update_drc_gain(UniDrcGain {
            sequences: Vec::new(),
            extension_present: false,
            extensions: Vec::new(),
            bits_read: 0,
        });
    }

    fn zero_sce_terminated_payload() -> Vec<u8> {
        let payload = zero_sce_payload();
        let mut reader = BitReader::new(&payload);
        let mut writer = BitWriter::new();
        // zero_sce_payload contains 38 syntax bits followed by byte padding.
        for _ in 0..38 {
            writer.write_bool(reader.read_bool().unwrap());
        }
        writer.write(ElementId::End.bits() as u32, 3);
        writer.finish()
    }

    fn adif_header_with_mono_pce() -> Vec<u8> {
        let pce = ProgramConfig {
            element_instance_tag: 0,
            profile: 1,
            sampling_frequency_index: 4,
            front: vec![ProgramElement {
                is_cpe: false,
                tag_select: 0,
            }],
            num_channels: 1,
            num_effective_channels: 1,
            ..ProgramConfig::default()
        };
        let mut writer = BitWriter::new();
        for byte in b"ADIF" {
            writer.write(*byte as u32, 8);
        }
        writer.write_bool(false);
        writer.write_bool(false);
        writer.write_bool(false);
        writer.write_bool(false); // constant bitrate
        writer.write(128_000, 23);
        writer.write(0, 4); // one PCE
        writer.write(0, 20); // adif_buffer_fullness
        pce.write_to_writer(&mut writer).unwrap();
        writer.finish()
    }

    fn loas_frame_with_latm_payload(payload: &[u8], include_config: bool) -> Vec<u8> {
        loas_frame_with_latm_config(payload, include_config, 4, 1)
    }

    fn loas_frame_with_latm_config(
        payload: &[u8],
        include_config: bool,
        sampling_frequency_index: u8,
        channel_configuration: u8,
    ) -> Vec<u8> {
        let mut writer = BitWriter::new();
        writer.write_bool(!include_config); // useSameStreamMux
        if include_config {
            writer.write_bool(false); // audioMuxVersion
            writer.write_bool(true); // allStreamsSameTimeFraming
            writer.write(0, 6); // numSubFrames - 1
            writer.write(0, 4); // numProgram
            writer.write(0, 3); // numLayer
            writer.write(2, 5); // AAC-LC
            writer.write(sampling_frequency_index as u32, 4);
            writer.write(channel_configuration as u32, 4);
            writer.write_bool(false); // frameLengthFlag
            writer.write_bool(false); // dependsOnCoreCoder
            writer.write_bool(false); // extensionFlag
            writer.write(0, 3); // frameLengthType
            writer.write(0xff, 8); // latmBufferFullness
            writer.write_bool(false); // otherDataPresent
            writer.write_bool(false); // crcCheckPresent
        }
        writer.write(payload.len() as u32, 8); // PayloadLengthInfo
        for byte in payload {
            writer.write(*byte as u32, 8);
        }
        let audio_mux_element = writer.finish();
        let length = audio_mux_element.len();
        let mut loas = vec![0x56, 0xe0 | ((length >> 8) as u8 & 0x1f), length as u8];
        loas.extend_from_slice(&audio_mux_element);
        loas
    }

    fn loas_frame_with_two_latm_subframes(payload: &[u8], include_config: bool) -> Vec<u8> {
        let mut writer = BitWriter::new();
        writer.write_bool(!include_config);
        if include_config {
            writer.write_bool(false);
            writer.write_bool(true);
            writer.write(1, 6); // two subframes
            writer.write(0, 4);
            writer.write(0, 3);
            writer.write(2, 5);
            writer.write(4, 4);
            writer.write(1, 4);
            writer.write_bool(false);
            writer.write_bool(false);
            writer.write_bool(false);
            writer.write(0, 3);
            writer.write(0xff, 8);
            writer.write_bool(false);
            writer.write_bool(false);
        }
        writer.write(payload.len() as u32, 8);
        for byte in payload {
            writer.write(*byte as u32, 8);
        }
        writer.write(payload.len() as u32, 8);
        for byte in payload {
            writer.write(*byte as u32, 8);
        }
        let audio_mux_element = writer.finish();
        let length = audio_mux_element.len();
        let mut loas = vec![0x56, 0xe0 | ((length >> 8) as u8 & 0x1f), length as u8];
        loas.extend_from_slice(&audio_mux_element);
        loas
    }

    #[test]
    fn dispatches_raw_asc_configured_access_units() {
        let asc = AudioSpecificConfig::aac_lc(44_100, 1).unwrap();
        let mut decoder = PureRustTransportDecoder::from_audio_specific_config(&asc).unwrap();
        let samples = decoder
            .decode_interleaved_f32_strict(&zero_sce_payload())
            .unwrap();
        assert_eq!(decoder.transport(), AacTransport::Raw);
        assert_eq!(samples.len(), 1024);
        assert!(samples.iter().all(|sample| *sample == 0.0));
    }

    #[test]
    fn stream_info_tracks_success_failure_bitrate_and_counter_reset() {
        let asc = AudioSpecificConfig::aac_lc(44_100, 1).unwrap();
        let payload = zero_sce_payload();
        let mut decoder = PureRustTransportDecoder::from_audio_specific_config(&asc).unwrap();

        decoder.decode_interleaved_f32_strict(&payload).unwrap();
        let first = decoder.stream_info();
        assert_eq!(first.num_total_bytes, payload.len() as u64);
        assert_eq!(first.num_bad_bytes, 0);
        assert_eq!(first.num_total_access_units, 1);
        assert_eq!(first.num_bad_access_units, 0);
        assert_eq!(first.bit_rate, (payload.len() as u32 * 8 * 44_100) / 1024);

        assert!(decoder.decode_interleaved_f32_strict(&[0xff]).is_err());
        let failed = decoder.stream_info();
        assert_eq!(failed.num_total_bytes, payload.len() as u64 + 1);
        assert_eq!(failed.num_bad_bytes, 1);
        assert_eq!(failed.num_total_access_units, 1);
        assert_eq!(failed.num_bad_access_units, 1);

        decoder.clear_transport_statistics();
        let cleared = decoder.stream_info();
        assert_eq!(cleared.bit_rate, 0);
        assert_eq!(cleared.num_total_bytes, 0);
        assert_eq!(cleared.num_bad_bytes, 0);
        assert_eq!(cleared.num_total_access_units, 0);
        assert_eq!(cleared.num_bad_access_units, 0);
    }

    #[test]
    fn decode_frame_flags_conceal_flush_interrupt_and_clear_history() {
        assert_eq!(DecodeFrameFlags::from_bits(16), None);
        let combined = DecodeFrameFlags::CONCEAL | DecodeFrameFlags::CLEAR_HISTORY;
        assert!(combined.contains(DecodeFrameFlags::CONCEAL));
        assert!(combined.contains(DecodeFrameFlags::CLEAR_HISTORY));

        let asc = AudioSpecificConfig::aac_lc(44_100, 1).unwrap();
        let mut decoder = PureRustTransportDecoder::from_audio_specific_config(&asc).unwrap();
        let concealed = decoder
            .decode_interleaved_f32_with_flags(&[0xff], combined)
            .unwrap();
        assert_eq!(concealed, vec![0.0; 1024]);
        let info = decoder.stream_info();
        assert_eq!(info.num_total_bytes, 0);
        assert_eq!(info.num_total_access_units, 1);
        assert_eq!(info.num_bad_access_units, 1);

        let flushed = decoder
            .decode_interleaved_i16_with_flags(&[], DecodeFrameFlags::FLUSH)
            .unwrap();
        assert_eq!(flushed, vec![0; 1024]);
        assert_eq!(decoder.stream_info().num_total_access_units, 2);

        decoder.estimated_lost_access_units = 9;
        decoder
            .decode_interleaved_i16_with_flags(&zero_sce_payload(), DecodeFrameFlags::INTERRUPTION)
            .unwrap();
        assert_eq!(decoder.stream_info().num_lost_access_units, 0);
        assert_eq!(decoder.stream_info().num_total_access_units, 3);
    }

    #[test]
    fn interruption_discards_pre_seek_overlap_and_matches_a_fresh_decoder() {
        let input = (0..1024)
            .map(|index| {
                (2.0 * std::f32::consts::PI * 997.0 * index as f32 / 44_100.0).sin() * 12_000.0
            })
            .collect::<Vec<_>>();
        let mut encoder = PureRustAacLcMonoEncoder::new(4, 4000, 2000).unwrap();
        let access_unit = encoder.encode_raw_data_block(&input).unwrap();
        let asc = AudioSpecificConfig::aac_lc(44_100, 1).unwrap();

        let mut continuous = PureRustTransportDecoder::from_audio_specific_config(&asc).unwrap();
        continuous
            .set_parameter(DecoderParameter::PcmLimiterEnable, -2)
            .unwrap();
        continuous.decode_raw_interleaved_f32(&access_unit).unwrap();
        let without_interruption = continuous.decode_raw_interleaved_f32(&access_unit).unwrap();

        let mut seeked = PureRustTransportDecoder::from_audio_specific_config(&asc).unwrap();
        seeked
            .set_parameter(DecoderParameter::PcmLimiterEnable, -2)
            .unwrap();
        seeked.decode_raw_interleaved_f32(&access_unit).unwrap();
        let after_interruption = seeked
            .decode_interleaved_f32_with_flags(&access_unit, DecodeFrameFlags::INTERRUPTION)
            .unwrap();

        let mut fresh = PureRustTransportDecoder::from_audio_specific_config(&asc).unwrap();
        fresh
            .set_parameter(DecoderParameter::PcmLimiterEnable, -2)
            .unwrap();
        let fresh_output = fresh.decode_raw_interleaved_f32(&access_unit).unwrap();

        assert_eq!(after_interruption, fresh_output);
        assert_ne!(without_interruption, fresh_output);
    }

    #[test]
    fn operational_decode_routes_complete_corrupt_access_units_to_each_method() {
        let payload = zero_sce_payload();
        let header = AdtsHeader::aac_lc(44_100, 1, payload.len()).unwrap();
        let mut good = vec![0; header.header_len()];
        header.write(&mut good).unwrap();
        good.extend_from_slice(&payload);
        let mut corrupt = good.clone();
        corrupt[header.header_len()..].fill(0xff);

        for method in 0..=2 {
            let mut floating = PureRustTransportDecoder::from_adts_frame(&good).unwrap();
            floating
                .set_parameter(DecoderParameter::PcmLimiterEnable, -2)
                .unwrap();
            floating
                .set_parameter(DecoderParameter::ConcealMethod, method)
                .unwrap();
            floating
                .decode_interleaved_f32_with_flags(&good, DecodeFrameFlags::NONE)
                .unwrap();
            let concealed = floating
                .decode_interleaved_f32_with_flags(&corrupt, DecodeFrameFlags::NONE)
                .unwrap();
            assert_eq!(concealed.len(), 1024);
            floating
                .decode_interleaved_f32_with_flags(&good, DecodeFrameFlags::NONE)
                .unwrap();
            let info = floating.stream_info();
            assert_eq!(info.num_total_access_units, 3);
            assert_eq!(info.num_bad_access_units, 1);

            let mut fixed = PureRustTransportDecoder::from_adts_frame(&good).unwrap();
            fixed
                .set_parameter(DecoderParameter::PcmLimiterEnable, -2)
                .unwrap();
            fixed
                .set_parameter(DecoderParameter::ConcealMethod, method)
                .unwrap();
            fixed
                .decode_interleaved_i16_with_flags(&good, DecodeFrameFlags::NONE)
                .unwrap();
            let concealed = fixed
                .decode_interleaved_i16_with_flags(&corrupt, DecodeFrameFlags::NONE)
                .unwrap();
            assert_eq!(concealed.len(), 1024);
            fixed
                .decode_interleaved_i16_with_flags(&good, DecodeFrameFlags::NONE)
                .unwrap();
            let info = fixed.stream_info();
            assert_eq!(info.num_total_access_units, 3);
            assert_eq!(info.num_bad_access_units, 1);
        }

        let mut strict = PureRustTransportDecoder::from_adts_frame(&good).unwrap();
        strict.decode_interleaved_i16(&good).unwrap();
        assert!(strict.decode_interleaved_i16_strict(&corrupt).is_err());
    }

    #[test]
    fn operational_decode_conceals_complete_raw_adif_and_direct_latm_payloads() {
        let good = zero_sce_payload();
        let corrupt = vec![0xff; good.len()];
        let asc = AudioSpecificConfig::aac_lc(44_100, 1).unwrap();

        let mut raw_f32 = PureRustTransportDecoder::from_audio_specific_config(&asc).unwrap();
        raw_f32
            .set_parameter(DecoderParameter::ConcealMethod, 1)
            .unwrap();
        raw_f32
            .decode_interleaved_f32_with_flags(&good, DecodeFrameFlags::NONE)
            .unwrap();
        assert_eq!(
            raw_f32
                .decode_interleaved_f32_with_flags(&corrupt, DecodeFrameFlags::NONE)
                .unwrap()
                .len(),
            1024
        );
        assert_eq!(raw_f32.stream_info().num_bad_access_units, 1);

        let mut raw_i16 = PureRustTransportDecoder::from_audio_specific_config(&asc).unwrap();
        raw_i16
            .set_parameter(DecoderParameter::ConcealMethod, 1)
            .unwrap();
        raw_i16
            .decode_interleaved_i16_with_flags(&good, DecodeFrameFlags::NONE)
            .unwrap();
        assert_eq!(
            raw_i16
                .decode_interleaved_i16_with_flags(&corrupt, DecodeFrameFlags::NONE)
                .unwrap()
                .len(),
            1024
        );

        let header = adif_header_with_mono_pce();
        let mut first_adif = header.clone();
        first_adif.extend_from_slice(&good);
        let mut adif_f32 = PureRustTransportDecoder::from_adif_bytes(&first_adif).unwrap();
        adif_f32
            .set_parameter(DecoderParameter::ConcealMethod, 1)
            .unwrap();
        adif_f32
            .decode_interleaved_f32_with_flags(&first_adif, DecodeFrameFlags::NONE)
            .unwrap();
        assert_eq!(
            adif_f32
                .decode_interleaved_f32_with_flags(&corrupt, DecodeFrameFlags::NONE)
                .unwrap()
                .len(),
            1024
        );

        let mut adif_i16 = PureRustTransportDecoder::from_adif_bytes(&first_adif).unwrap();
        adif_i16
            .set_parameter(DecoderParameter::ConcealMethod, 1)
            .unwrap();
        adif_i16
            .decode_interleaved_i16_with_flags(&first_adif, DecodeFrameFlags::NONE)
            .unwrap();
        assert_eq!(
            adif_i16
                .decode_interleaved_i16_with_flags(&corrupt, DecodeFrameFlags::NONE)
                .unwrap()
                .len(),
            1024
        );

        let first_latm_frame = loas_frame_with_latm_payload(&good, true);
        let corrupt_latm_frame = loas_frame_with_latm_payload(&corrupt, false);
        let first_latm = &first_latm_frame[3..];
        let corrupt_latm = &corrupt_latm_frame[3..];
        let mut latm_f32 =
            PureRustTransportDecoder::from_latm_audio_mux_element(first_latm).unwrap();
        latm_f32
            .set_parameter(DecoderParameter::ConcealMethod, 1)
            .unwrap();
        latm_f32
            .decode_interleaved_f32_with_flags(first_latm, DecodeFrameFlags::NONE)
            .unwrap();
        assert_eq!(
            latm_f32
                .decode_interleaved_f32_with_flags(corrupt_latm, DecodeFrameFlags::NONE)
                .unwrap()
                .len(),
            1024
        );

        let mut latm_i16 =
            PureRustTransportDecoder::from_latm_audio_mux_element(first_latm).unwrap();
        latm_i16
            .set_parameter(DecoderParameter::ConcealMethod, 1)
            .unwrap();
        latm_i16
            .decode_interleaved_i16_with_flags(first_latm, DecodeFrameFlags::NONE)
            .unwrap();
        assert_eq!(
            latm_i16
                .decode_interleaved_i16_with_flags(corrupt_latm, DecodeFrameFlags::NONE)
                .unwrap()
                .len(),
            1024
        );

        let mut strict = PureRustTransportDecoder::from_audio_specific_config(&asc).unwrap();
        assert!(strict.decode_interleaved_i16_strict(&corrupt).is_err());
    }

    #[test]
    fn conceal_parameter_routes_spectral_mute_and_noise_substitution() {
        let mut encoder = PureRustAacLcMonoEncoder::new(4, 32_000, 16_000).unwrap();
        let pcm = (0..1024)
            .map(|sample| (sample as f32 * 0.071).sin() * 0.4)
            .collect::<Vec<_>>();
        let payload = encoder.encode_raw_data_block(&pcm).unwrap();
        let asc = AudioSpecificConfig::aac_lc(44_100, 1).unwrap();

        let mut muted = PureRustTransportDecoder::from_audio_specific_config(&asc).unwrap();
        muted
            .set_parameter(DecoderParameter::PcmLimiterEnable, -2)
            .unwrap();
        muted
            .set_parameter(DecoderParameter::ConcealMethod, 0)
            .unwrap();
        muted.decode_interleaved_f32(&payload).unwrap();
        let muted_pcm = muted
            .decode_interleaved_f32_with_flags(&[], DecodeFrameFlags::CONCEAL)
            .unwrap();
        assert_eq!(
            muted.decoder().f32_concealment_state(),
            ConcealmentState::Mute
        );

        let mut noise = PureRustTransportDecoder::from_audio_specific_config(&asc).unwrap();
        noise
            .set_parameter(DecoderParameter::PcmLimiterEnable, -2)
            .unwrap();
        noise
            .set_parameter(DecoderParameter::ConcealMethod, 1)
            .unwrap();
        noise.decode_interleaved_f32(&payload).unwrap();
        let noise_pcm = noise
            .decode_interleaved_f32_with_flags(&[], DecodeFrameFlags::CONCEAL)
            .unwrap();
        assert_eq!(
            noise.decoder().f32_concealment_state(),
            ConcealmentState::Single
        );
        assert_ne!(muted_pcm, noise_pcm);
    }

    #[test]
    fn explicit_energy_concealment_delays_and_interpolates_the_missing_frame() {
        let mut encoder = PureRustAacLcMonoEncoder::new(4, 32_000, 16_000).unwrap();
        let first_input = (0..1024)
            .map(|sample| (sample as f32 * 0.041).sin() * 0.35)
            .collect::<Vec<_>>();
        let next_input = (0..1024)
            .map(|sample| (sample as f32 * 0.073).sin() * 0.22)
            .collect::<Vec<_>>();
        let first = encoder.encode_raw_data_block(&first_input).unwrap();
        let next = encoder.encode_raw_data_block(&next_input).unwrap();
        let asc = AudioSpecificConfig::aac_lc(44_100, 1).unwrap();

        let mut direct_f32 = PureRustTransportDecoder::from_audio_specific_config(&asc).unwrap();
        direct_f32
            .set_parameter(DecoderParameter::PcmLimiterEnable, -2)
            .unwrap();
        direct_f32
            .set_parameter(DecoderParameter::ConcealMethod, 1)
            .unwrap();
        let first_direct_f32 = direct_f32.decode_interleaved_f32(&first).unwrap();

        let mut expected_f32 = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
        expected_f32.decode_raw_data_block_f32(&first).unwrap();
        let mut next_f32 = expected_f32.clone();
        next_f32.decode_raw_data_block_f32(&next).unwrap();
        let next_spectrum = next_f32.f32_concealment_spectral_frame().unwrap();
        let interpolated_f32 = expected_f32
            .conceal_f32_interpolated(&next_spectrum)
            .unwrap();

        let mut energy_f32 = PureRustTransportDecoder::from_audio_specific_config(&asc).unwrap();
        energy_f32
            .set_parameter(DecoderParameter::PcmLimiterEnable, -2)
            .unwrap();
        energy_f32
            .set_parameter(DecoderParameter::ConcealMethod, 2)
            .unwrap();
        assert!(energy_f32
            .decode_interleaved_f32(&first)
            .unwrap()
            .iter()
            .all(|sample| *sample == 0.0));
        let delayed_first = energy_f32
            .decode_interleaved_f32_with_flags(&[], DecodeFrameFlags::CONCEAL)
            .unwrap();
        assert_eq!(delayed_first, first_direct_f32);
        let recovered = energy_f32.decode_interleaved_f32(&next).unwrap();
        assert_eq!(recovered, interpolated_f32);

        let mut direct_i16 = PureRustTransportDecoder::from_audio_specific_config(&asc).unwrap();
        direct_i16
            .set_parameter(DecoderParameter::PcmLimiterEnable, -2)
            .unwrap();
        direct_i16
            .set_parameter(DecoderParameter::ConcealMethod, 1)
            .unwrap();
        let first_direct_i16 = direct_i16.decode_interleaved_i16(&first).unwrap();

        let mut expected_i16 = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
        expected_i16
            .decode_raw_data_block_fixed_interleaved_i16(&first)
            .unwrap();
        let mut next_i16 = expected_i16.clone();
        next_i16
            .decode_raw_data_block_fixed_interleaved_i16(&next)
            .unwrap();
        let next_spectrum = next_i16.fixed_concealment_spectral_frame().unwrap();
        let interpolated_i16 = expected_i16
            .conceal_fixed_interpolated_i16(&next_spectrum)
            .unwrap();

        let mut energy_i16 = PureRustTransportDecoder::from_audio_specific_config(&asc).unwrap();
        energy_i16
            .set_parameter(DecoderParameter::PcmLimiterEnable, -2)
            .unwrap();
        energy_i16
            .set_parameter(DecoderParameter::ConcealMethod, 2)
            .unwrap();
        assert!(energy_i16
            .decode_interleaved_i16(&first)
            .unwrap()
            .iter()
            .all(|sample| *sample == 0));
        let delayed_first = energy_i16
            .decode_interleaved_i16_with_flags(&[], DecodeFrameFlags::CONCEAL)
            .unwrap();
        assert_eq!(delayed_first, first_direct_i16);
        let recovered = energy_i16.decode_interleaved_i16(&next).unwrap();
        assert_eq!(recovered, interpolated_i16);
    }

    #[test]
    fn clear_transport_buffer_discards_input_but_retains_au_lifetime_counts() {
        let payload = zero_sce_payload();
        let header = AdtsHeader::aac_lc(44_100, 1, payload.len()).unwrap();
        let mut frame = vec![0; header.header_len()];
        header.write(&mut frame).unwrap();
        frame.extend_from_slice(&payload);
        let mut decoder = PureRustTransportDecoder::from_adts_frame(&frame).unwrap();
        decoder.decode_interleaved_i16(&frame).unwrap();
        decoder.push_adts_bytes(&frame[..3]).unwrap();
        assert_eq!(decoder.buffered_adts_bytes().unwrap(), 3);

        decoder.clear_transport_buffer();
        assert_eq!(decoder.buffered_adts_bytes().unwrap(), 0);
        let info = decoder.stream_info();
        assert_eq!(info.num_total_bytes, 0);
        assert_eq!(info.num_bad_bytes, 0);
        assert_eq!(info.num_total_access_units, 1);
    }

    #[test]
    fn pcm_output_parameters_reorder_mix_extend_and_update_stream_info() {
        let config = PcmOutputConfig::default();
        let mpeg_labels = vec![
            ChannelLabel::FrontCenter,
            ChannelLabel::FrontLeft,
            ChannelLabel::FrontRight,
            ChannelLabel::BackLeft,
            ChannelLabel::BackRight,
            ChannelLabel::Lfe,
        ];
        assert_eq!(
            render_interleaved(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                mpeg_labels.clone(),
                config,
            ),
            vec![2.0, 3.0, 1.0, 6.0, 4.0, 5.0]
        );

        let mut dual = config;
        dual.dual_channel_mode = DualChannelOutputMode::Mix;
        assert_eq!(
            render_interleaved(
                vec![10.0, 20.0, 30.0, 50.0],
                vec![ChannelLabel::FrontLeft, ChannelLabel::FrontRight],
                dual,
            ),
            vec![15.0, 15.0, 40.0, 40.0]
        );

        let asc = AudioSpecificConfig::aac_lc(44_100, 1).unwrap();
        let mut decoder = PureRustTransportDecoder::from_audio_specific_config(&asc).unwrap();
        decoder
            .set_parameter(DecoderParameter::PcmMinOutputChannels, 2)
            .unwrap();
        assert_eq!(decoder.stream_info().num_channels, 2);
        assert_eq!(
            decoder
                .decode_interleaved_i16_with_flags(&[], DecodeFrameFlags::CONCEAL)
                .unwrap(),
            vec![0; 2048]
        );

        let stereo = AudioSpecificConfig::aac_lc(44_100, 2).unwrap();
        let mut extended = PureRustTransportDecoder::from_audio_specific_config(&stereo).unwrap();
        extended
            .set_parameter(DecoderParameter::PcmMinOutputChannels, 6)
            .unwrap();
        let extended_info = extended.stream_info();
        assert_eq!(extended_info.num_channels, 6);
        assert_eq!(
            extended_info.channel_labels,
            vec![
                ChannelLabel::FrontLeft,
                ChannelLabel::FrontRight,
                ChannelLabel::Empty,
                ChannelLabel::Empty,
                ChannelLabel::Empty,
                ChannelLabel::Empty,
            ]
        );
        assert_eq!(extended_info.channel_indices, vec![1, 2, 0, 0, 0, 0]);

        decoder
            .set_parameter(DecoderParameter::PcmMinOutputChannels, 6)
            .unwrap();
        decoder
            .set_parameter(DecoderParameter::PcmMaxOutputChannels, 2)
            .unwrap();
        assert_eq!(decoder.pcm_output.min_channels, Some(2));
        assert_eq!(decoder.pcm_output.max_channels, Some(2));
        assert_eq!(decoder.stream_info().num_channels, 2);
        assert!(matches!(
            decoder.set_parameter(DecoderParameter::PcmMaxOutputChannels, 3),
            Err(TransportDecodeError::InvalidParameterValue { .. })
        ));

        let six = AudioSpecificConfig::aac_lc(44_100, 6).unwrap();
        let mut decoder = PureRustTransportDecoder::from_audio_specific_config(&six).unwrap();
        let wav = decoder.stream_info();
        assert_eq!(
            wav.channel_labels,
            vec![
                ChannelLabel::FrontLeft,
                ChannelLabel::FrontRight,
                ChannelLabel::FrontCenter,
                ChannelLabel::Lfe,
                ChannelLabel::BackLeft,
                ChannelLabel::BackRight,
            ]
        );
        assert_eq!(wav.channel_indices, vec![1, 2, 0, 0, 0, 1]);
        decoder
            .set_parameter(DecoderParameter::PcmOutputChannelMapping, 0)
            .unwrap();
        assert_eq!(decoder.stream_info().channel_labels, mpeg_labels);
        decoder
            .set_parameter(DecoderParameter::PcmMaxOutputChannels, 2)
            .unwrap();
        assert_eq!(
            decoder.stream_info().channel_labels,
            vec![ChannelLabel::FrontLeft, ChannelLabel::FrontRight]
        );
    }

    #[test]
    fn pcm_downmix_uses_mpeg4_advanced_and_arib_metadata_profiles() {
        let labels = vec![
            ChannelLabel::FrontCenter,
            ChannelLabel::FrontLeft,
            ChannelLabel::FrontRight,
            ChannelLabel::BackLeft,
            ChannelLabel::BackRight,
            ChannelLabel::Lfe,
        ];
        let mut config = PcmOutputConfig {
            max_channels: Some(2),
            channel_order: PcmChannelOrder::Mpeg,
            ..PcmOutputConfig::default()
        };
        let default =
            render_interleaved(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], labels.clone(), config);
        assert!((default[0] - (2.0 + 5.0 * 0.707)).abs() < 1.0e-12);
        assert!((default[1] - (3.0 + 6.0 * 0.707)).abs() < 1.0e-12);

        config.advanced_downmix = Some(crate::drc::DvbAncillaryDownmixMetadata {
            center_mix_level_index: Some(4),
            surround_mix_level_index: Some(7),
            lfe_mix_level_index: Some(9),
            ..crate::drc::DvbAncillaryDownmixMetadata::default()
        });
        assert_eq!(
            render_interleaved(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], labels.clone(), config,),
            vec![5.5, 6.5]
        );

        config.metadata_profile = MetadataProfile::AribJapan;
        config.advanced_downmix = None;
        let arib = render_interleaved(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], labels.clone(), config);
        assert!((arib[0] - (2.0 * std::f64::consts::FRAC_1_SQRT_2 + 2.5)).abs() < 1.0e-12);
        assert!((arib[1] - (3.0 * std::f64::consts::FRAC_1_SQRT_2 + 3.0)).abs() < 1.0e-12);

        config.metadata_profile = MetadataProfile::MpegLegacy;
        config.matrix_mixdown = Some(crate::asc::MatrixMixdown {
            index: 1,
            pseudo_surround_enable: false,
        });
        let matrix = render_interleaved(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], labels.clone(), config);
        let front = 0.453_081_839;
        assert!((matrix[0] - front * (2.0 + std::f64::consts::FRAC_1_SQRT_2 + 2.0)).abs() < 1.0e-9);
        assert!((matrix[1] - front * (3.0 + std::f64::consts::FRAC_1_SQRT_2 + 2.5)).abs() < 1.0e-9);
        config.max_channels = Some(1);
        assert_eq!(
            render_interleaved(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], labels, config,),
            vec![2.625]
        );
    }

    #[test]
    fn pcm_downmix_applies_seven_one_first_stage_metadata() {
        let labels = vec![
            ChannelLabel::FrontCenter,
            ChannelLabel::FrontLeftCenter,
            ChannelLabel::FrontRightCenter,
            ChannelLabel::FrontLeft,
            ChannelLabel::FrontRight,
            ChannelLabel::BackLeft,
            ChannelLabel::BackRight,
            ChannelLabel::Lfe,
        ];
        let metadata = crate::drc::DvbAncillaryDownmixMetadata {
            downmix_a_index: Some(4),
            downmix_b_index: Some(7),
            center_mix_level_index: Some(2),
            surround_mix_level_index: Some(2),
            five_channel_downmix_gain_index: Some(8),
            ..crate::drc::DvbAncillaryDownmixMetadata::default()
        };
        let gain = advanced_downmix_gain(8);
        let six_config = PcmOutputConfig {
            max_channels: Some(6),
            advanced_downmix: Some(metadata),
            channel_order: PcmChannelOrder::Mpeg,
            ..PcmOutputConfig::default()
        };
        let six = render_interleaved(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            labels.clone(),
            six_config,
        );
        let expected_six = [3.5, 4.0, 5.0, 6.0, 7.0, 8.0].map(|value| value * gain);
        for (actual, expected) in six.iter().zip(expected_six) {
            assert!((actual - expected).abs() < 1.0e-12);
        }

        let stereo = render_interleaved(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            labels,
            PcmOutputConfig {
                max_channels: Some(2),
                advanced_downmix: Some(metadata),
                channel_order: PcmChannelOrder::Mpeg,
                ..PcmOutputConfig::default()
            },
        );
        assert!((stereo[0] - gain * (4.0 + 0.707 * 3.5 + 0.707 * 6.0)).abs() < 1.0e-12);
        assert!((stereo[1] - gain * (5.0 + 0.707 * 3.5 + 0.707 * 7.0)).abs() < 1.0e-12);
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn fdk_and_rust_seven_one_metadata_downmix_match() {
        const SCALE: f64 = 16_777_216.0;
        let input = [1, 2, 3, 4, 5, 6, 7, 8].map(|value| value * (1 << 24));
        // MPEG-4 DSE: C/S levels 2, A=4, B=7, gain-5=+1 dB.
        let ancillary = [0xbc, 0xc0, 0x18, 0xaa, 0x60, 0x9c, 0x10, 0x00];
        let labels = vec![
            ChannelLabel::FrontCenter,
            ChannelLabel::FrontLeftCenter,
            ChannelLabel::FrontRightCenter,
            ChannelLabel::FrontLeft,
            ChannelLabel::FrontRight,
            ChannelLabel::BackLeft,
            ChannelLabel::BackRight,
            ChannelLabel::Lfe,
        ];
        let metadata = crate::drc::parse_dvb_ancillary_downmix(&ancillary).unwrap();

        for target in [6, 2] {
            let mut c_output = [0i32; 8];
            let mut c_channels = 0;
            let result = unsafe {
                fdk_aac_sys::fdk_pcm_downmix_7_1_test(
                    input.as_ptr(),
                    target,
                    ancillary.as_ptr(),
                    ancillary.len() as i32,
                    c_output.as_mut_ptr(),
                    &mut c_channels,
                )
            };
            assert_eq!(result, 0);
            assert_eq!(c_channels, target);
            let rust = render_interleaved(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                labels.clone(),
                PcmOutputConfig {
                    max_channels: Some(target as usize),
                    advanced_downmix: Some(metadata),
                    channel_order: PcmChannelOrder::Mpeg,
                    ..PcmOutputConfig::default()
                },
            );
            for (c, rust) in c_output[..target as usize].iter().zip(rust) {
                let c = f64::from(*c) / SCALE;
                assert!(
                    (c - rust).abs() < 2.0e-3,
                    "target={target} C={c} Rust={rust}"
                );
            }
        }
    }

    #[test]
    fn pcm_limiter_parameters_follow_fdk_defaults_and_delay_reporting() {
        let asc = AudioSpecificConfig::aac_lc(44_100, 2).unwrap();
        let mut decoder = PureRustTransportDecoder::from_audio_specific_config(&asc).unwrap();
        assert_eq!(decoder.stream_info().output_delay, 1685);

        decoder
            .set_parameter(DecoderParameter::PcmLimiterEnable, 0)
            .unwrap();
        assert_eq!(decoder.stream_info().output_delay, 1024);
        decoder
            .set_parameter(DecoderParameter::PcmLimiterEnable, 1)
            .unwrap();
        decoder
            .set_parameter(DecoderParameter::PcmLimiterAttackTime, 1)
            .unwrap();
        assert_eq!(decoder.stream_info().output_delay, 1068);
        decoder
            .set_parameter(DecoderParameter::PcmLimiterReleaseTime, 75)
            .unwrap();

        for (parameter, value) in [
            (DecoderParameter::PcmLimiterEnable, 2),
            (DecoderParameter::PcmLimiterAttackTime, 0),
            (DecoderParameter::PcmLimiterAttackTime, 16),
            (DecoderParameter::PcmLimiterReleaseTime, 0),
        ] {
            assert!(matches!(
                decoder.set_parameter(parameter, value),
                Err(TransportDecodeError::InvalidParameterValue { .. })
            ));
        }

        let ld = AudioSpecificConfig {
            audio_object_type: 23,
            sampling_frequency_index: 4,
            sampling_frequency: 44_100,
            channel_configuration: 1,
            extension: None,
            ga_specific: Some(GaSpecificConfig::default()),
            eld_specific: None,
            usac_config: None,
            error_protection_config: Some(0),
            program_config: None,
            bits_read: 0,
        };
        let mut low_delay = PureRustTransportDecoder::from_audio_specific_config(&ld).unwrap();
        assert_eq!(low_delay.stream_info().output_delay, 0);
        low_delay
            .set_parameter(DecoderParameter::PcmLimiterEnable, 1)
            .unwrap();
        assert_eq!(low_delay.stream_info().output_delay, 661);
        low_delay
            .set_parameter(DecoderParameter::PcmLimiterEnable, -2)
            .unwrap();
        assert_eq!(low_delay.stream_info().output_delay, 0);

        decoder
            .set_parameter(DecoderParameter::ConcealMethod, 1)
            .unwrap();
        assert_eq!(decoder.stream_info().output_delay, 44);
        decoder
            .set_parameter(DecoderParameter::ConcealMethod, 2)
            .unwrap();
        assert_eq!(decoder.stream_info().output_delay, 1068);
        assert!(matches!(
            decoder.set_parameter(DecoderParameter::ConcealMethod, 3),
            Err(TransportDecodeError::InvalidParameterValue { .. })
        ));
    }

    #[test]
    fn stream_info_adds_fdk_sbr_processing_delays() {
        let mut he = AudioSpecificConfig::aac_lc(24_000, 1).unwrap();
        he.extension = Some(crate::asc::AudioSpecificConfigExtension {
            audio_object_type: 5,
            sampling_frequency_index: 3,
            sampling_frequency: 48_000,
            ps_present: false,
        });
        let he = PureRustTransportDecoder::from_audio_specific_config(&he).unwrap();
        assert_eq!(he.stream_info().output_delay, 2048 + 962 + 720);
        assert!(!he.decoder().qmf_low_power());

        let mut he_stereo = AudioSpecificConfig::aac_lc(24_000, 2).unwrap();
        he_stereo.extension = Some(crate::asc::AudioSpecificConfigExtension {
            audio_object_type: 5,
            sampling_frequency_index: 3,
            sampling_frequency: 48_000,
            ps_present: false,
        });
        let he_stereo = PureRustTransportDecoder::from_audio_specific_config(&he_stereo).unwrap();
        assert!(he_stereo.decoder().qmf_low_power());

        // A signalled PS extension forces complex QMF in automatic mode.
        let mut he_ps = AudioSpecificConfig::aac_lc(24_000, 1).unwrap();
        he_ps.extension = Some(crate::asc::AudioSpecificConfigExtension {
            audio_object_type: 29,
            sampling_frequency_index: 3,
            sampling_frequency: 48_000,
            ps_present: true,
        });
        let mut he_ps = PureRustTransportDecoder::from_audio_specific_config(&he_ps).unwrap();
        assert!(!he_ps.decoder().qmf_low_power());
        assert_eq!(he_ps.stream_info().num_channels, 2);
        he_ps
            .set_parameter(DecoderParameter::QmfLowPower, 1)
            .unwrap();
        assert!(he_ps.decoder().qmf_low_power());
        assert_eq!(he_ps.stream_info().num_channels, 1);

        let eld = AudioSpecificConfig {
            audio_object_type: 39,
            sampling_frequency_index: 6,
            sampling_frequency: 24_000,
            channel_configuration: 1,
            extension: None,
            ga_specific: None,
            eld_specific: Some(crate::asc::EldSpecificConfig {
                sbr_present: true,
                sbr_sampling_rate: true,
                sbr_headers: vec![crate::asc::LdSbrHeader {
                    amp_resolution: true,
                    start_frequency: 5,
                    stop_frequency: 3,
                    crossover_band: 2,
                    frequency_scale: Some(0),
                    alter_scale: Some(false),
                    noise_bands: Some(2),
                    ..crate::asc::LdSbrHeader::default()
                }],
                ..crate::asc::EldSpecificConfig::default()
            }),
            usac_config: None,
            error_protection_config: Some(0),
            program_config: None,
            bits_read: 0,
        };
        let eld = PureRustTransportDecoder::from_audio_specific_config(&eld).unwrap();
        assert!(eld.decoder().qmf_low_power());
        assert_eq!(eld.stream_info().output_delay, 64);
    }

    #[test]
    fn uni_drc_parameters_validate_and_update_the_selection_request() {
        let asc = AudioSpecificConfig::aac_lc(48_000, 2).unwrap();
        let mut decoder = PureRustTransportDecoder::from_audio_specific_config(&asc).unwrap();
        decoder
            .set_parameter(DecoderParameter::DrcBoostFactor, 63)
            .unwrap();
        decoder
            .set_parameter(DecoderParameter::DrcAttenuationFactor, 31)
            .unwrap();
        decoder
            .set_parameter(DecoderParameter::DrcReferenceLevel, 96)
            .unwrap();
        decoder
            .set_parameter(DecoderParameter::UniDrcSetEffect, 3)
            .unwrap();
        decoder
            .set_parameter(DecoderParameter::UniDrcAlbumMode, 1)
            .unwrap();
        let request = decoder.decoder().drc_selection_request();
        assert!((request.boost_scale - 63.0 / 127.0).abs() < f32::EPSILON);
        assert!((request.attenuation_scale - 31.0 / 127.0).abs() < f32::EPSILON);
        assert_eq!(request.target_loudness, Some(-24.0));
        assert_eq!(request.preferred_effect_mask, 0b100);
        assert!(request.enabled);
        assert!(request.album_mode);
        decoder
            .set_parameter(DecoderParameter::DrcHeavyCompression, 1)
            .unwrap();
        decoder
            .set_parameter(DecoderParameter::DrcDefaultPresentationMode, 2)
            .unwrap();
        decoder
            .set_parameter(DecoderParameter::DrcEncoderTargetLevel, 100)
            .unwrap();
        let legacy = decoder.decoder().legacy_drc_parameters();
        assert!(legacy.heavy_compression);
        assert_eq!(legacy.default_presentation_mode, 2);
        assert_eq!(legacy.encoder_target_level, 100);

        decoder
            .set_parameter(DecoderParameter::MetadataProfile, 3)
            .unwrap();
        assert_eq!(decoder.metadata_profile(), MetadataProfile::AribJapan);
        assert_eq!(decoder.metadata_expiry_ms(), 550);
        decoder
            .set_parameter(DecoderParameter::MetadataExpiryTime, 900)
            .unwrap();
        assert_eq!(decoder.metadata_expiry_ms(), 900);
        decoder
            .set_parameter(DecoderParameter::QmfLowPower, 1)
            .unwrap();
        assert_eq!(decoder.qmf_processing_mode(), QmfProcessingMode::LowPower);
        assert!(decoder.decoder().qmf_low_power());
        decoder
            .set_parameter(DecoderParameter::QmfLowPower, 0)
            .unwrap();
        assert!(!decoder.decoder().qmf_low_power());
        decoder
            .set_parameter(DecoderParameter::QmfLowPower, -1)
            .unwrap();

        decoder
            .set_parameter(DecoderParameter::UniDrcSetEffect, -1)
            .unwrap();
        decoder
            .set_parameter(DecoderParameter::DrcReferenceLevel, -1)
            .unwrap();
        let request = decoder.decoder().drc_selection_request();
        assert!(!request.enabled);
        assert_eq!(request.target_loudness, None);

        for (parameter, value) in [
            (DecoderParameter::DrcBoostFactor, 128),
            (DecoderParameter::DrcAttenuationFactor, -1),
            (DecoderParameter::DrcReferenceLevel, 39),
            (DecoderParameter::DrcReferenceLevel, 128),
            (DecoderParameter::DrcReferenceLevel, -128),
            (DecoderParameter::UniDrcSetEffect, 7),
            (DecoderParameter::UniDrcAlbumMode, 2),
            (DecoderParameter::DrcHeavyCompression, 2),
            (DecoderParameter::DrcDefaultPresentationMode, 3),
            (DecoderParameter::DrcEncoderTargetLevel, 128),
            (DecoderParameter::MetadataProfile, 4),
            (DecoderParameter::MetadataExpiryTime, -1),
            (DecoderParameter::QmfLowPower, 2),
        ] {
            assert!(matches!(
                decoder.set_parameter(parameter, value),
                Err(TransportDecodeError::InvalidParameterValue { .. })
            ));
        }
    }

    #[test]
    fn dispatches_adts_frames_and_rejects_mismatched_api() {
        let payload = zero_sce_payload();
        let header = AdtsHeader::aac_lc(44_100, 1, payload.len()).unwrap();
        let mut frame = vec![0; header.header_len()];
        header.write(&mut frame).unwrap();
        frame.extend_from_slice(&payload);

        let mut decoder = PureRustTransportDecoder::from_adts_frame(&frame).unwrap();
        let samples = decoder.decode_interleaved_i16_strict(&frame).unwrap();
        assert_eq!(decoder.transport(), AacTransport::Adts);
        assert_eq!(samples.len(), 1024);
        assert!(samples.iter().all(|sample| *sample == 0));
        assert_eq!(
            decoder.decode_raw_interleaved_f32(&payload).unwrap_err(),
            TransportDecodeError::TransportMismatch {
                configured: AacTransport::Adts,
                requested: AacTransport::Raw,
            }
        );
    }

    #[test]
    fn rejects_unimplemented_transports_explicitly() {
        assert_eq!(
            PureRustTransportDecoder::new_unsupported(AacTransport::Loas).unwrap_err(),
            TransportDecodeError::UnsupportedTransport(AacTransport::Loas)
        );
    }

    #[test]
    fn dispatches_crc_protected_adts_multi_block_frames() {
        let payload = zero_sce_payload();
        let block_len = payload.len() + 2;
        let mut header = AdtsHeader::aac_lc(44_100, 1, 0).unwrap();
        header.protection_absent = false;
        header.number_of_raw_data_blocks_in_frame = 1;
        header.frame_length = 7 + 2 + 2 + block_len * 2;
        header.crc_check = Some(0);

        let mut standard_header = vec![0; header.header_len()];
        header.write(&mut standard_header).unwrap();
        let mut input = standard_header[..7].to_vec();
        input.extend_from_slice(&(block_len as u16).to_be_bytes());
        let header_crc = crate::adts::adts_crc16(&input);
        input.extend_from_slice(&header_crc.to_be_bytes());
        let block_crc =
            crate::adts::adts_crc16_padded_bit_regions([(payload.as_slice(), 3..38, 192)]).unwrap();
        for _ in 0..2 {
            input.extend_from_slice(&payload);
            input.extend_from_slice(&block_crc.to_be_bytes());
        }

        let mut decoder = PureRustTransportDecoder::from_adts_frame(&input).unwrap();
        let blocks = decoder.decode_adts_blocks_interleaved_i16(&input).unwrap();
        assert_eq!(blocks.len(), 2);
        assert!(blocks
            .iter()
            .all(|samples| samples.len() == 1024 && samples.iter().all(|sample| *sample == 0)));
        let info = decoder.stream_info();
        assert_eq!(info.num_total_bytes, input.len() as u64);
        assert_eq!(info.num_total_access_units, 2);
        assert_eq!(info.num_bad_access_units, 0);
    }

    #[test]
    fn incrementally_decodes_chunked_adts_frames_through_transport_facade() {
        let payload = zero_sce_payload();
        let header = AdtsHeader::aac_lc(44_100, 1, payload.len()).unwrap();
        let mut frame = vec![0; header.header_len()];
        header.write(&mut frame).unwrap();
        frame.extend_from_slice(&payload);

        let mut decoder = PureRustTransportDecoder::from_adts_header(header).unwrap();
        decoder.push_adts_bytes(&frame[..5]).unwrap();
        assert!(decoder.drain_adts_interleaved_i16().unwrap().is_empty());
        assert_eq!(decoder.buffered_adts_bytes().unwrap(), 5);

        decoder.push_adts_bytes(&frame[5..]).unwrap();
        let decoded = decoder.drain_adts_interleaved_i16().unwrap();
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].len(), 1024);
        assert!(decoded[0].iter().all(|sample| *sample == 0));
        assert_eq!(decoder.buffered_adts_bytes().unwrap(), 0);

        let asc = AudioSpecificConfig::aac_lc(44_100, 1).unwrap();
        let mut raw_decoder = PureRustTransportDecoder::from_audio_specific_config(&asc).unwrap();
        assert_eq!(
            raw_decoder.push_adts_bytes(&frame).unwrap_err(),
            TransportDecodeError::TransportMismatch {
                configured: AacTransport::Raw,
                requested: AacTransport::Adts,
            }
        );
    }

    #[test]
    fn recovers_from_adts_configuration_change() {
        let payload = zero_sce_payload();
        let first_header = AdtsHeader::aac_lc(44_100, 1, payload.len()).unwrap();
        let changed_header = AdtsHeader::aac_lc(48_000, 1, payload.len()).unwrap();
        let mut first = vec![0; first_header.header_len()];
        first_header.write(&mut first).unwrap();
        first.extend_from_slice(&payload);
        let mut changed = vec![0; changed_header.header_len()];
        changed_header.write(&mut changed).unwrap();
        changed.extend_from_slice(&payload);

        let mut decoder = PureRustTransportDecoder::from_adts_frame(&first).unwrap();
        decoder.enable_adts_pcm_concealment().unwrap();
        decoder.set_adts_average_bitrate(128_000).unwrap();
        assert_eq!(decoder.decode_interleaved_i16(&first).unwrap().len(), 1024);
        assert_eq!(decoder.decoder().sampling_frequency_index(), 4);
        let samples = decoder.decode_interleaved_i16(&changed).unwrap();
        assert_eq!(samples.len(), 1024);
        assert!(samples.iter().all(|sample| *sample == 0));
        assert_eq!(decoder.decoder().sampling_frequency_index(), 3);
        assert!(decoder.adts_pcm_concealment.is_some());
        assert_eq!(
            decoder
                .adts_loss_estimator
                .as_ref()
                .unwrap()
                .average_bitrate(),
            128_000
        );
    }

    #[test]
    fn estimates_lost_adts_access_units_after_sync_recovery() {
        let payload = zero_sce_payload();
        let header = AdtsHeader::aac_lc(48_000, 1, payload.len()).unwrap();
        let mut frame = vec![0; header.header_len()];
        header.write(&mut frame).unwrap();
        frame.extend_from_slice(&payload);
        let average_bitrate = (frame.len() * 8 * 48_000 / 1024) as u32;

        let mut decoder = PureRustTransportDecoder::from_adts_frame(&frame).unwrap();
        decoder.set_adts_average_bitrate(average_bitrate).unwrap();
        decoder.enable_adts_pcm_concealment().unwrap();
        decoder.push_adts_bytes(&vec![0; frame.len()]).unwrap();
        decoder.push_adts_bytes(&frame).unwrap();
        let decoded = decoder.drain_adts_interleaved_i16().unwrap();
        assert_eq!(decoded.len(), 2);
        assert!(decoded
            .iter()
            .all(|samples| samples.len() == 1024 && samples.iter().all(|sample| *sample == 0)));
        assert_eq!(decoder.estimated_lost_access_units().unwrap(), 1);
        assert_eq!(decoder.take_estimated_lost_access_units().unwrap(), 1);
        assert_eq!(decoder.estimated_lost_access_units().unwrap(), 0);
    }

    #[test]
    fn applies_fixed_spectral_concealment_before_recovered_adts_frame() {
        let payload = zero_sce_payload();
        let header = AdtsHeader::aac_lc(48_000, 1, payload.len()).unwrap();
        let mut frame = vec![0; header.header_len()];
        header.write(&mut frame).unwrap();
        frame.extend_from_slice(&payload);
        let average_bitrate = (frame.len() * 8 * 48_000 / 1024) as u32;

        let mut decoder = PureRustTransportDecoder::from_adts_frame(&frame).unwrap();
        decoder.set_adts_average_bitrate(average_bitrate).unwrap();
        decoder.enable_adts_spectral_concealment().unwrap();
        decoder.push_adts_bytes(&frame).unwrap();
        assert_eq!(decoder.drain_adts_interleaved_i16().unwrap().len(), 1);

        decoder.push_adts_bytes(&vec![0; frame.len()]).unwrap();
        decoder.push_adts_bytes(&frame).unwrap();
        let decoded = decoder.drain_adts_interleaved_i16().unwrap();
        assert_eq!(decoded.len(), 2);
        assert!(decoded
            .iter()
            .all(|samples| samples.len() == 1024 && samples.iter().all(|sample| *sample == 0)));
    }

    #[test]
    fn applies_f32_surrounding_frame_spectral_interpolation() {
        let payload = zero_sce_payload();
        let header = AdtsHeader::aac_lc(48_000, 1, payload.len()).unwrap();
        let mut frame = vec![0; header.header_len()];
        header.write(&mut frame).unwrap();
        frame.extend_from_slice(&payload);
        let average_bitrate = (frame.len() * 8 * 48_000 / 1024) as u32;

        let mut decoder = PureRustTransportDecoder::from_adts_frame(&frame).unwrap();
        decoder.set_adts_average_bitrate(average_bitrate).unwrap();
        decoder.enable_adts_spectral_concealment().unwrap();
        decoder.push_adts_bytes(&frame).unwrap();
        assert_eq!(decoder.drain_adts_interleaved_f32().unwrap().len(), 1);

        decoder.push_adts_bytes(&vec![0; frame.len()]).unwrap();
        decoder.push_adts_bytes(&frame).unwrap();
        let decoded = decoder.drain_adts_interleaved_f32().unwrap();
        assert_eq!(decoded.len(), 2);
        assert!(decoded
            .iter()
            .all(|samples| samples.len() == 1024 && samples.iter().all(|sample| *sample == 0.0)));
    }

    #[test]
    fn applies_f32_spectral_fade_for_multiple_lost_adts_frames() {
        let payload = zero_sce_payload();
        let header = AdtsHeader::aac_lc(48_000, 1, payload.len()).unwrap();
        let mut frame = vec![0; header.header_len()];
        header.write(&mut frame).unwrap();
        frame.extend_from_slice(&payload);
        let average_bitrate = (frame.len() * 8 * 48_000 / 1024) as u32;

        let mut decoder = PureRustTransportDecoder::from_adts_frame(&frame).unwrap();
        decoder.set_adts_average_bitrate(average_bitrate).unwrap();
        decoder.enable_adts_spectral_concealment().unwrap();
        decoder.push_adts_bytes(&frame).unwrap();
        assert_eq!(decoder.drain_adts_interleaved_f32().unwrap().len(), 1);

        decoder.push_adts_bytes(&vec![0; frame.len() * 2]).unwrap();
        decoder.push_adts_bytes(&frame).unwrap();
        let decoded = decoder.drain_adts_interleaved_f32().unwrap();
        assert_eq!(decoded.len(), 3);
        assert!(decoded
            .iter()
            .all(|samples| samples.len() == 1024 && samples.iter().all(|sample| *sample == 0.0)));
        assert_eq!(
            decoder.decoder().f32_concealment_state(),
            crate::decoder::ConcealmentState::FadeIn
        );
    }

    #[test]
    fn conceals_syntactically_corrupt_adts_access_unit_transactionally() {
        let payload = zero_sce_payload();
        let header = AdtsHeader::aac_lc(48_000, 1, payload.len()).unwrap();
        let mut good = vec![0; header.header_len()];
        header.write(&mut good).unwrap();
        good.extend_from_slice(&payload);
        let mut corrupt = good.clone();
        corrupt[header.header_len()..].fill(0xff);

        let mut f32_decoder = PureRustTransportDecoder::from_adts_frame(&good).unwrap();
        f32_decoder.enable_adts_spectral_concealment().unwrap();
        f32_decoder.push_adts_bytes(&good).unwrap();
        assert_eq!(f32_decoder.drain_adts_interleaved_f32().unwrap().len(), 1);
        f32_decoder.push_adts_bytes(&corrupt).unwrap();
        let concealed = f32_decoder.drain_adts_interleaved_f32().unwrap();
        assert_eq!(concealed.len(), 1);
        assert!(concealed[0].iter().all(|sample| *sample == 0.0));
        assert_eq!(
            f32_decoder.decoder().f32_concealment_state(),
            crate::decoder::ConcealmentState::Single
        );

        let mut fixed_decoder = PureRustTransportDecoder::from_adts_frame(&good).unwrap();
        fixed_decoder.enable_adts_spectral_concealment().unwrap();
        fixed_decoder.push_adts_bytes(&good).unwrap();
        assert_eq!(fixed_decoder.drain_adts_interleaved_i16().unwrap().len(), 1);
        fixed_decoder.push_adts_bytes(&corrupt).unwrap();
        let concealed = fixed_decoder.drain_adts_interleaved_i16().unwrap();
        assert_eq!(concealed.len(), 1);
        assert!(concealed[0].iter().all(|sample| *sample == 0));
        assert_eq!(
            fixed_decoder.decoder().fixed_concealment_state(),
            crate::decoder::ConcealmentState::Single
        );
    }

    #[test]
    fn spectral_concealment_propagates_post_processing_errors() {
        let payload = zero_sce_payload();
        let header = AdtsHeader::aac_lc(48_000, 1, payload.len()).unwrap();
        let mut good = vec![0; header.header_len()];
        header.write(&mut good).unwrap();
        good.extend_from_slice(&payload);
        let mut corrupt = good.clone();
        corrupt[header.header_len()..].fill(0xff);
        let average_bitrate = (good.len() * 8 * 48_000 / 1024) as u32;

        let mut f32_loss = PureRustTransportDecoder::from_adts_frame(&good).unwrap();
        f32_loss.set_adts_average_bitrate(average_bitrate).unwrap();
        f32_loss.enable_adts_spectral_concealment().unwrap();
        f32_loss.push_adts_bytes(&good).unwrap();
        f32_loss.drain_adts_interleaved_f32().unwrap();
        configure_missing_coefficient_drc(&mut f32_loss);
        f32_loss.push_adts_bytes(&vec![0; good.len()]).unwrap();
        f32_loss.push_adts_bytes(&good).unwrap();
        assert!(matches!(
            f32_loss.drain_adts_interleaved_f32(),
            Err(TransportDecodeError::Decode(DecodeError::Drc(_)))
        ));

        let mut f32_corrupt = PureRustTransportDecoder::from_adts_frame(&good).unwrap();
        f32_corrupt.enable_adts_spectral_concealment().unwrap();
        f32_corrupt.push_adts_bytes(&good).unwrap();
        f32_corrupt.drain_adts_interleaved_f32().unwrap();
        configure_missing_coefficient_drc(&mut f32_corrupt);
        f32_corrupt.push_adts_bytes(&corrupt).unwrap();
        assert!(matches!(
            f32_corrupt.drain_adts_interleaved_f32(),
            Err(TransportDecodeError::Decode(DecodeError::Drc(_)))
        ));

        let mut fixed_loss = PureRustTransportDecoder::from_adts_frame(&good).unwrap();
        fixed_loss
            .set_adts_average_bitrate(average_bitrate)
            .unwrap();
        fixed_loss.enable_adts_spectral_concealment().unwrap();
        fixed_loss.push_adts_bytes(&good).unwrap();
        fixed_loss.drain_adts_interleaved_i16().unwrap();
        configure_missing_coefficient_drc(&mut fixed_loss);
        fixed_loss.push_adts_bytes(&vec![0; good.len()]).unwrap();
        fixed_loss.push_adts_bytes(&good).unwrap();
        assert!(matches!(
            fixed_loss.drain_adts_interleaved_i16(),
            Err(TransportDecodeError::Decode(DecodeError::Drc(_)))
        ));

        let mut fixed_corrupt = PureRustTransportDecoder::from_adts_frame(&good).unwrap();
        fixed_corrupt.enable_adts_spectral_concealment().unwrap();
        fixed_corrupt.push_adts_bytes(&good).unwrap();
        fixed_corrupt.drain_adts_interleaved_i16().unwrap();
        configure_missing_coefficient_drc(&mut fixed_corrupt);
        fixed_corrupt.push_adts_bytes(&corrupt).unwrap();
        assert!(matches!(
            fixed_corrupt.drain_adts_interleaved_i16(),
            Err(TransportDecodeError::Decode(DecodeError::Drc(_)))
        ));
    }

    #[test]
    fn arbitrary_incremental_transport_bytes_never_panic() {
        let mut state = 0xa341_316cu32;
        for case in 0..128usize {
            let length = case * 43 % 257;
            let mut payload = vec![0; length];
            for byte in &mut payload {
                state = state.wrapping_mul(22_695_477).wrapping_add(1);
                *byte = (state >> 16) as u8;
            }

            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut adts = AdtsIncrementalStream::new();
                let mut loas = LoasIncrementalStream::new();
                for chunk in payload.chunks(7) {
                    adts.push(chunk);
                    loas.push(chunk);
                    while adts.next_frame().is_some() {}
                    while loas.next_frame().is_some() {}
                }
                let _ = adts.next_frame();
                let _ = loas.next_frame();
            }));
            assert!(
                result.is_ok(),
                "incremental transport parser panicked for deterministic random case {case}, length {length}"
            );
        }
    }

    #[test]
    fn spectral_concealment_handles_missing_initial_reference() {
        let payload = zero_sce_payload();
        let header = AdtsHeader::aac_lc(48_000, 1, payload.len()).unwrap();
        let mut good = vec![0; header.header_len()];
        header.write(&mut good).unwrap();
        good.extend_from_slice(&payload);
        let mut corrupt = good.clone();
        corrupt[header.header_len()..].fill(0xff);

        let mut f32_corrupt = PureRustTransportDecoder::from_adts_frame(&good).unwrap();
        f32_corrupt.enable_adts_spectral_concealment().unwrap();
        f32_corrupt.push_adts_bytes(&corrupt).unwrap();
        assert!(matches!(
            f32_corrupt.drain_adts_interleaved_f32(),
            Err(TransportDecodeError::Decode(_))
        ));

        let mut fixed_corrupt = PureRustTransportDecoder::from_adts_frame(&good).unwrap();
        fixed_corrupt.enable_adts_spectral_concealment().unwrap();
        fixed_corrupt.push_adts_bytes(&corrupt).unwrap();
        assert!(matches!(
            fixed_corrupt.drain_adts_interleaved_i16(),
            Err(TransportDecodeError::Decode(_))
        ));

        let average_bitrate = (good.len() * 8 * 48_000 / 1024) as u32;
        let mut f32_recovery = PureRustTransportDecoder::from_adts_frame(&good).unwrap();
        f32_recovery
            .set_adts_average_bitrate(average_bitrate)
            .unwrap();
        f32_recovery.enable_adts_spectral_concealment().unwrap();
        f32_recovery
            .push_adts_bytes(&vec![0; good.len() * 2])
            .unwrap();
        f32_recovery.push_adts_bytes(&good).unwrap();
        let recovered = f32_recovery.drain_adts_interleaved_f32().unwrap();
        assert_eq!(recovered.len(), 3);
        assert!(recovered.iter().flatten().all(|&sample| sample == 0.0));

        let mut f32_single_loss = PureRustTransportDecoder::from_adts_frame(&good).unwrap();
        f32_single_loss
            .set_adts_average_bitrate(average_bitrate)
            .unwrap();
        f32_single_loss.enable_adts_spectral_concealment().unwrap();
        f32_single_loss
            .push_adts_bytes(&vec![0; good.len()])
            .unwrap();
        f32_single_loss.push_adts_bytes(&good).unwrap();
        let recovered = f32_single_loss.drain_adts_interleaved_f32().unwrap();
        assert_eq!(recovered.len(), 2);
        assert!(recovered.iter().flatten().all(|&sample| sample == 0.0));

        let mut fixed_recovery = PureRustTransportDecoder::from_adts_frame(&good).unwrap();
        fixed_recovery
            .set_adts_average_bitrate(average_bitrate)
            .unwrap();
        fixed_recovery.enable_adts_spectral_concealment().unwrap();
        fixed_recovery
            .push_adts_bytes(&vec![0; good.len() * 2])
            .unwrap();
        fixed_recovery.push_adts_bytes(&good).unwrap();
        let recovered = fixed_recovery.drain_adts_interleaved_i16().unwrap();
        assert_eq!(recovered.len(), 3);
        assert!(recovered.iter().flatten().all(|&sample| sample == 0));
    }

    #[test]
    fn fixed_spectral_concealment_falls_back_when_pce_layout_changes() {
        fn pce(is_cpe: bool, channels: u8) -> ProgramConfig {
            ProgramConfig {
                element_instance_tag: 0,
                profile: 1,
                sampling_frequency_index: 4,
                front: vec![ProgramElement {
                    is_cpe,
                    tag_select: 0,
                }],
                num_channels: channels,
                num_effective_channels: channels,
                ..ProgramConfig::default()
            }
        }

        let mono_payload = {
            let mut writer = BitWriter::new();
            writer.write(ElementId::ProgramConfig.bits() as u32, 3);
            pce(false, 1).write_to_writer(&mut writer).unwrap();
            let sce = zero_sce_payload();
            let mut reader = BitReader::new(&sce);
            for _ in 0..38 {
                writer.write_bool(reader.read_bool().unwrap());
            }
            writer.write(ElementId::End.bits() as u32, 3);
            writer.finish()
        };
        let stereo_payload = {
            let mut writer = BitWriter::new();
            writer.write(ElementId::ProgramConfig.bits() as u32, 3);
            pce(true, 2).write_to_writer(&mut writer).unwrap();
            writer.write(ElementId::ChannelPair.bits() as u32, 3);
            writer.write(0, 4); // element tag
            writer.write_bool(true); // common window
            writer.write_bool(false); // ICS reserved
            writer.write(0, 2); // only-long window
            writer.write_bool(false); // sine window
            writer.write(1, 6); // max_sfb
            writer.write_bool(false); // prediction absent
            writer.write(0, 2); // no MS stereo
            for _ in 0..2 {
                writer.write(100, 8); // global gain
                writer.write(ZERO_HCB as u32, 4);
                writer.write(1, 5);
                writer.write_bool(false); // pulse absent
                writer.write_bool(false); // TNS absent
                writer.write_bool(false); // gain control absent
            }
            writer.write(ElementId::End.bits() as u32, 3);
            writer.finish()
        };
        let frame = |payload: &[u8]| {
            let header = AdtsHeader::aac_lc(44_100, 0, payload.len()).unwrap();
            let mut frame = vec![0; header.header_len()];
            header.write(&mut frame).unwrap();
            frame.extend_from_slice(payload);
            frame
        };
        let mono = frame(&mono_payload);
        let stereo = frame(&stereo_payload);
        let average_bitrate = (mono.len() * 8 * 44_100 / 1024) as u32;

        let mut decoder = PureRustTransportDecoder::from_adts_frame(&mono).unwrap();
        decoder.set_adts_average_bitrate(average_bitrate).unwrap();
        decoder.enable_adts_spectral_concealment().unwrap();
        decoder.push_adts_bytes(&mono).unwrap();
        assert_eq!(decoder.drain_adts_interleaved_i16().unwrap().len(), 1);

        decoder.push_adts_bytes(&vec![0; mono.len()]).unwrap();
        decoder.push_adts_bytes(&stereo).unwrap();
        let recovered = decoder.drain_adts_interleaved_i16().unwrap();
        assert_eq!(recovered.len(), 2);
        assert_eq!(recovered[0].len(), 1024);
        assert_eq!(recovered[1].len(), 2048);
        assert!(recovered.iter().flatten().all(|&sample| sample == 0));
    }

    #[test]
    fn incremental_adts_propagates_plain_corruption_and_conceals_bad_recovery() {
        let payload = zero_sce_payload();
        let header = AdtsHeader::aac_lc(48_000, 1, payload.len()).unwrap();
        let mut good = vec![0; header.header_len()];
        header.write(&mut good).unwrap();
        good.extend_from_slice(&payload);
        let mut corrupt = good.clone();
        corrupt[header.header_len()..].fill(0xff);

        let mut plain = PureRustTransportDecoder::from_adts_frame(&good).unwrap();
        plain.push_adts_bytes(&good).unwrap();
        assert_eq!(plain.drain_adts_interleaved_f32().unwrap().len(), 1);
        plain.push_adts_bytes(&corrupt).unwrap();
        assert!(matches!(
            plain.drain_adts_interleaved_f32(),
            Err(TransportDecodeError::Decode(_))
        ));

        let mut plain_fixed = PureRustTransportDecoder::from_adts_frame(&good).unwrap();
        plain_fixed.push_adts_bytes(&corrupt).unwrap();
        assert!(matches!(
            plain_fixed.drain_adts_interleaved_i16(),
            Err(TransportDecodeError::Decode(_))
        ));

        let average_bitrate = (good.len() * 8 * 48_000 / 1024) as u32;
        let mut f32_recovery = PureRustTransportDecoder::from_adts_frame(&good).unwrap();
        f32_recovery
            .set_adts_average_bitrate(average_bitrate)
            .unwrap();
        f32_recovery.enable_adts_spectral_concealment().unwrap();
        f32_recovery.push_adts_bytes(&good).unwrap();
        assert_eq!(f32_recovery.drain_adts_interleaved_f32().unwrap().len(), 1);
        f32_recovery.push_adts_bytes(&vec![0; good.len()]).unwrap();
        f32_recovery.push_adts_bytes(&corrupt).unwrap();
        let concealed = f32_recovery.drain_adts_interleaved_f32().unwrap();
        assert_eq!(concealed.len(), 1);
        assert!(concealed.iter().flatten().all(|sample| sample.is_finite()));

        let mut fixed_recovery = PureRustTransportDecoder::from_adts_frame(&good).unwrap();
        fixed_recovery
            .set_adts_average_bitrate(average_bitrate)
            .unwrap();
        fixed_recovery.enable_adts_spectral_concealment().unwrap();
        fixed_recovery.push_adts_bytes(&good).unwrap();
        assert_eq!(
            fixed_recovery.drain_adts_interleaved_i16().unwrap().len(),
            1
        );
        fixed_recovery
            .push_adts_bytes(&vec![0; good.len()])
            .unwrap();
        fixed_recovery.push_adts_bytes(&corrupt).unwrap();
        let concealed = fixed_recovery.drain_adts_interleaved_i16().unwrap();
        assert_eq!(concealed.len(), 2);
        assert!(concealed.iter().flatten().all(|&sample| sample == 0));
    }

    #[test]
    fn dispatches_adif_header_followed_by_raw_aac_lc_access_unit() {
        let header = adif_header_with_mono_pce();
        let payload = zero_sce_payload();
        let mut input = header.clone();
        input.extend_from_slice(&payload);

        let mut decoder = PureRustTransportDecoder::from_adif_bytes(&input).unwrap();
        let samples = decoder.decode_adif_interleaved_f32(&input).unwrap();
        assert_eq!(samples.len(), 1024);
        assert!(samples.iter().all(|sample| *sample == 0.0));

        let samples = decoder.decode_adif_interleaved_f32(&payload).unwrap();
        assert_eq!(samples.len(), 1024);

        let mut generic = PureRustTransportDecoder::from_adif_bytes(&input).unwrap();
        assert_eq!(generic.decode_interleaved_f32(&input).unwrap().len(), 1024);

        let mut fixed = PureRustTransportDecoder::from_adif_bytes(&input).unwrap();
        let samples = fixed.decode_interleaved_i16(&input).unwrap();
        assert_eq!(samples.len(), 1024);
        assert!(samples.iter().all(|sample| *sample == 0));

        let samples = fixed.decode_adif_interleaved_i16(&payload).unwrap();
        assert_eq!(samples.len(), 1024);
        assert!(samples.iter().all(|sample| *sample == 0));
    }

    #[test]
    fn decodes_usac_latm_payload_to_f32_and_i16() {
        let usac = UsacConfig {
            sampling_frequency_index: 3,
            sampling_frequency: 48_000,
            core_sbr_frame_length_index: 1,
            core_frame_length: 1024,
            output_frame_length: 1024,
            sbr_ratio_index: 0,
            channel_configuration_index: 1,
            elements: vec![UsacElementConfig::SingleChannel {
                noise_filling: false,
                sbr: None,
            }],
            extensions: Vec::new(),
        };
        let config = AudioSpecificConfig {
            audio_object_type: 42,
            sampling_frequency_index: 3,
            sampling_frequency: 48_000,
            channel_configuration: 1,
            extension: None,
            ga_specific: None,
            eld_specific: None,
            usac_config: Some(usac),
            error_protection_config: None,
            program_config: None,
            bits_read: 0,
        };
        let mut payload = BitWriter::new();
        payload.write_bool(true); // independent
        payload.write_bool(false); // FD core
        payload.write_bool(false); // TNS absent
        payload.write(0, 8); // global gain
        payload.write(0, 2); // ONLY_LONG
        payload.write_bool(false); // window shape
        payload.write(0, 6); // max_sfb
        payload.write_bool(false); // no FAC
        let payload = payload.finish();

        let mut f32_decoder =
            PureRustTransportDecoder::from_audio_specific_config(&config).unwrap();
        f32_decoder.transport = AacTransport::LatmOutOfBandConfig;
        let f32_samples = f32_decoder
            .decode_latm_payload_f32(&payload, false)
            .unwrap();
        assert_eq!(f32_samples.len(), 1024);
        assert!(f32_samples.iter().all(|sample| *sample == 0.0));

        let mut i16_decoder =
            PureRustTransportDecoder::from_audio_specific_config(&config).unwrap();
        i16_decoder.transport = AacTransport::LatmOutOfBandConfig;
        let i16_samples = i16_decoder.decode_latm_payload_i16(&payload, true).unwrap();
        assert_eq!(i16_samples.len(), 1024);
        assert!(i16_samples.iter().all(|sample| *sample == 0));
    }

    #[test]
    fn incremental_adif_propagates_complete_invalid_raw_blocks() {
        let header = adif_header_with_mono_pce();
        let mut invalid = BitWriter::new();
        invalid.write(ElementId::SingleChannel.bits() as u32, 3);
        invalid.write(0, 4); // element tag
        invalid.write(100, 8); // global gain
        invalid.write_bool(false); // reserved
        invalid.write(0, 2); // ONLY_LONG
        invalid.write_bool(false); // window shape
        invalid.write(1, 6); // max_sfb
        invalid.write_bool(false); // prediction absent
        invalid.write(ZERO_HCB as u32, 4);
        invalid.write(1, 5); // one zero-codebook band
        invalid.write_bool(false); // pulse absent
        invalid.write_bool(false); // TNS absent
        invalid.write_bool(true); // unsupported gain control
        let invalid = invalid.finish();

        let mut f32_stream = AdifIncrementalDecoder::new();
        f32_stream.push(&header);
        f32_stream.push(&invalid);
        assert_eq!(
            f32_stream.drain_interleaved_f32(),
            Err(TransportDecodeError::Decode(
                DecodeError::GainControlUnsupported
            ))
        );

        let mut i16_stream = AdifIncrementalDecoder::new();
        i16_stream.push(&header);
        i16_stream.push(&invalid);
        assert_eq!(
            i16_stream.drain_interleaved_i16(),
            Err(TransportDecodeError::Decode(
                DecodeError::GainControlUnsupported
            ))
        );
    }

    #[test]
    fn incrementally_decodes_adif_header_and_raw_data_blocks() {
        let header = adif_header_with_mono_pce();
        let access_unit = zero_sce_terminated_payload();
        let mut stream = AdifIncrementalDecoder::new();

        stream.push(&header[..5]);
        assert!(stream.drain_interleaved_i16().unwrap().is_empty());
        assert_eq!(stream.buffered_len(), 5);

        stream.push(&header[5..]);
        stream.push(&access_unit[..3]);
        assert!(stream.drain_interleaved_i16().unwrap().is_empty());
        assert!(stream.header().is_some());
        assert_eq!(stream.buffered_len(), 3);

        stream.push(&access_unit[3..]);
        stream.push(&access_unit);
        let decoded = stream.drain_interleaved_i16().unwrap();
        assert_eq!(decoded.len(), 2);
        assert!(decoded
            .iter()
            .all(|samples| { samples.len() == 1024 && samples.iter().all(|sample| *sample == 0) }));
        assert_eq!(stream.buffered_len(), 0);
    }

    #[test]
    fn dispatches_loas_latm_aac_lc_and_reuses_stream_mux_config() {
        let payload = zero_sce_payload();
        let first = loas_frame_with_latm_payload(&payload, true);
        let second = loas_frame_with_latm_payload(&payload, false);
        let mut decoder = PureRustTransportDecoder::from_loas_frame(&first).unwrap();
        let first_pcm = decoder.decode_loas_interleaved_f32(&first).unwrap();
        let second_pcm = decoder.decode_loas_interleaved_f32(&second).unwrap();
        assert_eq!(first_pcm.len(), 1024);
        assert_eq!(second_pcm.len(), 1024);
        assert!(first_pcm.iter().all(|sample| *sample == 0.0));
        assert!(second_pcm.iter().all(|sample| *sample == 0.0));
        let info = decoder.stream_info();
        assert_eq!(info.num_total_bytes, (first.len() + second.len()) as u64);
        assert_eq!(info.num_total_access_units, 2);
        assert_eq!(info.num_bad_access_units, 0);

        let mut fixed_decoder = PureRustTransportDecoder::from_loas_frame(&first).unwrap();
        let fixed = fixed_decoder.decode_interleaved_i16(&first).unwrap();
        assert_eq!(fixed.len(), 1024);
        assert!(fixed.iter().all(|sample| *sample == 0));
    }

    #[test]
    fn dispatches_direct_latm_with_in_band_and_out_of_band_configuration() {
        let payload = zero_sce_payload();
        let in_band_loas = loas_frame_with_latm_payload(&payload, true);
        let same_mux_loas = loas_frame_with_latm_payload(&payload, false);
        let in_band = &in_band_loas[3..];
        let same_mux = &same_mux_loas[3..];

        let mut in_band_decoder =
            PureRustTransportDecoder::from_latm_audio_mux_element(in_band).unwrap();
        assert_eq!(
            in_band_decoder.transport(),
            AacTransport::LatmMuxConfigPresent
        );
        assert_eq!(
            in_band_decoder
                .decode_latm_interleaved_i16(in_band)
                .unwrap()
                .len(),
            1024
        );
        assert_eq!(
            in_band_decoder
                .decode_interleaved_i16(same_mux)
                .unwrap()
                .len(),
            1024
        );

        let asc = AudioSpecificConfig::parse(&[0x12, 0x08]).unwrap();
        let mut out_of_band = PureRustTransportDecoder::from_latm_out_of_band_config(&asc).unwrap();
        assert_eq!(out_of_band.transport(), AacTransport::LatmOutOfBandConfig);
        assert_eq!(
            out_of_band
                .decode_latm_interleaved_f32(same_mux)
                .unwrap()
                .len(),
            1024
        );
    }

    #[test]
    fn decodes_latm_aac_lc_960_frame_length_flag() {
        let mut config = AudioSpecificConfig::aac_lc(44_100, 1).unwrap();
        config.ga_specific.as_mut().unwrap().frame_length_flag = true;
        let mut writer = crate::latm::LatmWriter::new(config, 0, 1, 0).unwrap();
        let element = writer
            .write_audio_mux_element(&[zero_sce_payload()])
            .unwrap();

        let mut floating = PureRustTransportDecoder::from_latm_audio_mux_element(&element).unwrap();
        assert_eq!(floating.stream_info().aac_samples_per_frame, 960);
        assert_eq!(
            floating.decode_latm_interleaved_f32(&element).unwrap(),
            vec![0.0; 960]
        );

        let mut fixed = PureRustTransportDecoder::from_latm_audio_mux_element(&element).unwrap();
        assert_eq!(
            fixed.decode_latm_interleaved_i16(&element).unwrap(),
            vec![0; 960]
        );
    }

    #[test]
    fn decodes_all_latm_subframes_and_reuses_their_mux_state() {
        let payload = zero_sce_payload();
        let first = loas_frame_with_two_latm_subframes(&payload, true);
        let second = loas_frame_with_two_latm_subframes(&payload, false);
        let mut decoder = PureRustTransportDecoder::from_loas_frame(&first).unwrap();
        let first_pcm = decoder.decode_loas_interleaved_f32(&first).unwrap();
        let second_pcm = decoder.decode_loas_interleaved_f32(&second).unwrap();
        assert_eq!(first_pcm.len(), 2048);
        assert_eq!(second_pcm.len(), 2048);
        assert!(first_pcm.iter().all(|sample| *sample == 0.0));
        assert!(second_pcm.iter().all(|sample| *sample == 0.0));
        let info = decoder.stream_info();
        assert_eq!(info.num_total_bytes, (first.len() + second.len()) as u64);
        assert_eq!(info.num_total_access_units, 4);
        assert_eq!(info.num_bad_access_units, 0);
    }

    #[test]
    fn incrementally_decodes_and_resynchronizes_loas_frames() {
        let payload = zero_sce_payload();
        let first = loas_frame_with_latm_payload(&payload, true);
        let second = loas_frame_with_latm_payload(&payload, false);
        let mut decoder = PureRustTransportDecoder::from_loas_frame(&first).unwrap();

        decoder.push_loas_bytes(&[0x00, 0x12]).unwrap();
        decoder.push_loas_bytes(&first[..2]).unwrap();
        assert!(decoder.drain_loas_interleaved_f32().unwrap().is_empty());
        assert_eq!(decoder.buffered_loas_bytes().unwrap(), 2);
        assert_eq!(decoder.discarded_loas_bytes().unwrap(), 2);
        assert_eq!(decoder.stream_info().num_bad_bytes, 2);

        decoder.push_loas_bytes(&first[2..]).unwrap();
        decoder.push_loas_bytes(&second).unwrap();
        let decoded = decoder.drain_loas_interleaved_f32().unwrap();
        assert_eq!(decoded.len(), 2);
        assert!(decoded.iter().all(|samples| {
            samples.len() == 1024 && samples.iter().all(|sample| *sample == 0.0)
        }));
        assert_eq!(decoder.buffered_loas_bytes().unwrap(), 0);
        let info = decoder.stream_info();
        assert_eq!(
            info.num_total_bytes,
            (2 + first.len() + second.len()) as u64
        );
        assert_eq!(info.num_bad_bytes, 2);
        assert_eq!(info.num_total_access_units, 2);

        decoder.push_loas_bytes(&first).unwrap();
        decoder.push_loas_bytes(&second).unwrap();
        let fixed = decoder.drain_loas_interleaved_i16().unwrap();
        assert_eq!(fixed.len(), 2);
        assert!(fixed
            .iter()
            .all(|samples| { samples.len() == 1024 && samples.iter().all(|sample| *sample == 0) }));
        assert_eq!(decoder.stream_info().num_total_access_units, 4);

        let asc = AudioSpecificConfig::aac_lc(44_100, 1).unwrap();
        let mut raw_decoder = PureRustTransportDecoder::from_audio_specific_config(&asc).unwrap();
        assert_eq!(
            raw_decoder.push_loas_bytes(&first).unwrap_err(),
            TransportDecodeError::TransportMismatch {
                configured: AacTransport::Raw,
                requested: AacTransport::Loas,
            }
        );
    }

    #[test]
    fn recovers_from_in_band_latm_configuration_change() {
        let payload = zero_sce_payload();
        let first = loas_frame_with_latm_config(&payload, true, 4, 1); // 44100 mono
        let changed = loas_frame_with_latm_config(&payload, true, 3, 1); // 48000 mono
        let mut decoder = PureRustTransportDecoder::from_loas_frame(&first).unwrap();

        assert_eq!(decoder.decoder().sampling_frequency_index(), 4);
        assert_eq!(decoder.decode_interleaved_i16(&first).unwrap().len(), 1024);
        let samples = decoder.decode_interleaved_i16(&changed).unwrap();
        assert_eq!(samples.len(), 1024);
        assert!(samples.iter().all(|sample| *sample == 0));
        assert_eq!(decoder.decoder().sampling_frequency_index(), 3);
    }

    #[test]
    fn reports_transport_support_and_formats_all_error_kinds() {
        for transport in [
            AacTransport::Raw,
            AacTransport::Adif,
            AacTransport::Adts,
            AacTransport::LatmMuxConfigPresent,
            AacTransport::LatmOutOfBandConfig,
            AacTransport::Loas,
            AacTransport::Drm,
        ] {
            assert!(transport.is_supported_by_pure_rust());
        }

        let errors = [
            TransportDecodeError::from(AdtsError::InvalidSyncword(0)),
            TransportDecodeError::from(AdifError::InvalidSignature),
            TransportDecodeError::from(LatmError::MissingStreamMuxConfig),
            TransportDecodeError::from(LoasError::InvalidSyncword(0)),
            TransportDecodeError::from(AscError::InvalidAudioObjectType(0)),
            TransportDecodeError::from(DecodeError::UnsupportedSamplingFrequencyIndex(15)),
            TransportDecodeError::from(UsacDecodeError::UnsupportedConfiguration),
            TransportDecodeError::TransportMismatch {
                configured: AacTransport::Raw,
                requested: AacTransport::Adts,
            },
            TransportDecodeError::UnsupportedTransport(AacTransport::Drm),
        ];
        for error in errors {
            assert!(!error.to_string().is_empty());
        }
    }

    #[test]
    fn decodes_crc_protected_drm_transport_in_float_and_fixed_facades() {
        let config = DrmAudioConfig::aac(48_000, crate::drm::DrmAudioMode::Mono, false).unwrap();
        let mut payload = BitWriter::new();
        payload.write_bool(false); // ICS reserved
        payload.write(0, 2); // ONLY_LONG
        payload.write_bool(false); // sine window
        payload.write(0, 6); // max_sfb
        payload.write_bool(false); // predictor absent
        payload.write_bool(false); // TNS absent
        payload.write_bool(false); // LTP absent
        payload.write(0, 8); // global gain
        payload.write(0, 14); // reordered spectral length
        payload.write(0, 6); // longest codeword
        let payload = payload.finish();
        let mut packet = vec![crate::drm::drm_crc8_bits(&payload, 0, 41).unwrap()];
        packet.extend_from_slice(&payload);

        let mut floating = PureRustTransportDecoder::from_drm_config(&config).unwrap();
        assert_eq!(floating.transport(), AacTransport::Drm);
        assert_eq!(
            floating.decode_interleaved_f32(&packet).unwrap(),
            vec![0.0; 960]
        );
        assert_eq!(floating.stream_info().num_total_access_units, 1);

        let mut fixed = PureRustTransportDecoder::from_drm_config(&config).unwrap();
        assert_eq!(fixed.decode_interleaved_i16(&packet).unwrap(), vec![0; 960]);

        packet[1] ^= 0x80;
        assert!(matches!(
            floating.decode_interleaved_f32(&packet),
            Err(TransportDecodeError::Drm(DrmAacDecodeError::Config(
                crate::drm::DrmError::CrcMismatch { .. }
            )))
        ));
    }

    #[test]
    fn decodes_drm_xhe_transport_from_static_config() {
        let mut config = BitWriter::new();
        config.write(3, 2); // xHE-AAC
        config.write_bool(false); // reserved legacy SBR flag
        config.write(0, 2); // mono
        config.write(7, 3); // 48 kHz
        config.write_bool(false); // text
        config.write_bool(false); // enhancement
        config.write(0, 5); // coder field
        config.write_bool(false); // reserved
        config.write(0, 2); // coreSbrFrameLengthIndex 1
        config.write_bool(true); // noise filling
        let config = config.finish();

        let mut payload = BitWriter::new();
        payload.write_bool(true); // independency flag
        payload.write_bool(false); // FD core
        payload.write_bool(false); // no TNS
        payload.write(0, 8); // global gain
        payload.write(0, 8); // noise level and offset
        payload.write(0, 2); // ONLY_LONG
        payload.write_bool(false); // window shape
        payload.write(0, 6); // max_sfb
        payload.write_bool(false); // no FAC
        let payload = payload.finish();

        let mut floating = PureRustTransportDecoder::from_drm_xhe_static_config(&config).unwrap();
        assert_eq!(floating.transport(), AacTransport::Drm);
        assert_eq!(floating.decoder().audio_object_type(), 42);
        assert_eq!(
            floating.decode_interleaved_f32(&payload).unwrap(),
            vec![0.0; 1024]
        );

        let mut fixed = PureRustTransportDecoder::from_drm_xhe_static_config(&config).unwrap();
        assert_eq!(
            fixed.decode_interleaved_i16(&payload).unwrap(),
            vec![0; 1024]
        );
    }

    #[test]
    fn exercises_remaining_raw_and_adts_facade_entry_points() {
        let payload = zero_sce_payload();
        let mut raw = PureRustTransportDecoder::from_asc_bytes(&[0x12, 0x08]).unwrap();
        assert_eq!(raw.decoder_mut().sampling_frequency_index(), 4);
        assert_eq!(
            raw.decode_raw_interleaved_f32(&payload).unwrap().len(),
            1024
        );
        assert_eq!(raw.decode_interleaved_f32(&payload).unwrap().len(), 1024);
        assert_eq!(raw.decode_interleaved_i16(&payload).unwrap().len(), 1024);
        assert_eq!(
            raw.decode_interleaved_i16_strict(&payload).unwrap().len(),
            1024
        );

        let header = AdtsHeader::aac_lc(44_100, 1, payload.len()).unwrap();
        let mut frame = vec![0; header.header_len()];
        header.write(&mut frame).unwrap();
        frame.extend_from_slice(&payload);
        let mut adts = PureRustTransportDecoder::from_adts_frame(&frame).unwrap();
        assert_eq!(
            adts.decode_adts_interleaved_f32(&frame).unwrap().len(),
            1024
        );
        assert_eq!(adts.decode_interleaved_f32(&frame).unwrap().len(), 1024);
        assert_eq!(
            adts.decode_interleaved_f32_strict(&frame).unwrap().len(),
            1024
        );
        assert_eq!(
            adts.decode_adts_blocks_interleaved_f32(&frame).unwrap()[0].len(),
            1024
        );
        assert_eq!(adts.discarded_adts_bytes().unwrap(), 0);
        assert_eq!(
            adts.set_adts_average_bitrate(0),
            Err(TransportDecodeError::Adts(
                AdtsError::InvalidLossEstimatorConfiguration
            ))
        );
        adts.enable_adts_pcm_concealment().unwrap();
        adts.disable_adts_pcm_concealment().unwrap();
        adts.enable_adts_spectral_concealment().unwrap();
        adts.disable_adts_spectral_concealment().unwrap();

        raw.transport = AacTransport::Drm;
        assert_eq!(
            raw.decode_interleaved_f32(&payload).unwrap_err(),
            TransportDecodeError::UnsupportedTransport(AacTransport::Drm)
        );
        assert_eq!(
            raw.decode_interleaved_i16(&payload).unwrap_err(),
            TransportDecodeError::UnsupportedTransport(AacTransport::Drm)
        );
    }

    #[test]
    fn exercises_strict_latm_and_loas_facade_entry_points() {
        let payload = zero_sce_payload();
        let configured_loas = loas_frame_with_latm_payload(&payload, true);
        let reused_loas = loas_frame_with_latm_payload(&payload, false);
        let configured_latm = &configured_loas[3..];
        let reused_latm = &reused_loas[3..];

        let mut latm =
            PureRustTransportDecoder::from_latm_audio_mux_element(configured_latm).unwrap();
        assert_eq!(
            latm.decode_interleaved_f32(reused_latm).unwrap().len(),
            1024
        );
        assert_eq!(
            latm.decode_latm_interleaved_f32_strict(reused_latm)
                .unwrap()
                .len(),
            1024
        );
        assert_eq!(
            latm.decode_latm_interleaved_i16_strict(reused_latm)
                .unwrap()
                .len(),
            1024
        );

        let mut loas = PureRustTransportDecoder::from_loas_frame(&configured_loas).unwrap();
        assert_eq!(
            loas.decode_interleaved_f32(&reused_loas).unwrap().len(),
            1024
        );
        assert_eq!(
            loas.decode_loas_interleaved_f32_strict(&reused_loas)
                .unwrap()
                .len(),
            1024
        );
        assert_eq!(
            loas.decode_loas_interleaved_i16(&reused_loas)
                .unwrap()
                .len(),
            1024
        );
        assert_eq!(
            loas.decode_loas_interleaved_i16_strict(&reused_loas)
                .unwrap()
                .len(),
            1024
        );
    }

    #[test]
    fn incrementally_decodes_adif_through_f32_path_and_reports_bad_headers() {
        let header = adif_header_with_mono_pce();
        let access_unit = zero_sce_terminated_payload();
        let mut stream = AdifIncrementalDecoder::new();
        assert!(stream.drain_interleaved_f32().unwrap().is_empty());
        stream.push(&header);
        stream.push(&access_unit[..2]);
        assert!(stream.drain_interleaved_f32().unwrap().is_empty());
        stream.push(&access_unit[2..]);
        let decoded = stream.drain_interleaved_f32().unwrap();
        assert_eq!(decoded.len(), 1);
        assert!(decoded[0].iter().all(|sample| *sample == 0.0));

        let mut invalid = AdifIncrementalDecoder::new();
        invalid.push(b"NOPE");
        assert_eq!(
            invalid.drain_interleaved_f32().unwrap_err(),
            TransportDecodeError::Adif(AdifError::InvalidSignature)
        );

        let asc = AudioSpecificConfig::aac_lc(44_100, 1).unwrap();
        let mut raw = PureRustTransportDecoder::from_audio_specific_config(&asc).unwrap();
        assert_eq!(
            raw.decode_latm_interleaved_f32(&[0]).unwrap_err(),
            TransportDecodeError::TransportMismatch {
                configured: AacTransport::Raw,
                requested: AacTransport::LatmMuxConfigPresent,
            }
        );
    }
}
