//! Incremental Pure Rust AAC-LC decode orchestration helpers.

use std::fmt;

use crate::adif::AdifHeader;
use crate::adts::{
    adts_crc16_padded_bit_regions, sample_rate_from_index, AdtsError, AdtsFrame, AdtsStream,
};
use crate::asc::{AscError, AudioSpecificConfig, MatrixMixdown, ProgramConfig};
use crate::bits::{BitError, BitReader, BitWriter};
use crate::concealment::{
    interpolate_f32_spectra_mixed, interpolate_fixed_spectra_mixed, SpectralInterpolationError,
};
use crate::drc::{
    parse_dvb_ancillary_downmix, parse_dvb_ancillary_drc, parse_mpeg4_drc_fill_element,
    parse_mpeg4_drc_payload, DrcError, DrcSelectionRequest, DvbAncillaryDownmixMetadata,
    DvbAncillaryDrcPayload, LoudnessInfoSet, Mpeg4DrcPayload, UniDrcConfig, UniDrcGain,
};
use crate::filterbank::{
    synthesize_aac_lc_frame, synthesize_aac_lc_frame_from_fixed_inverse_q31,
    synthesize_aac_lc_frame_from_inverse_q31, FilterbankError, FixedLongBlockFilterbank,
    LongBlockFilterbank,
};
use crate::fixed::{dbl_to_pcm16, mul_q31, FixpDbl};
use crate::hcr::{
    codewords_to_spectral_data as hcr_codewords_to_spectral,
    decode_reordered_codewords as decode_hcr_codewords, sections_from_ics as hcr_sections_from_ics,
    HcrElementType, HcrError, HcrSideInfo,
};
use crate::huffman::{decode_fdk_2bit, HuffmanError, HUFFMAN_CODEBOOK_SCL};
use crate::ics::{IcsError, IcsInfo, IcsLimits, WindowSequence, WindowShape};
use crate::inverse::{
    inverse_quantize_spectrum_f32, inverse_quantize_spectrum_fixed,
    inverse_quantize_spectrum_fixed_block_scaled, FixedInverseQuantizedSpectrum, InverseQuantError,
    InverseQuantizedSpectrum,
};
use crate::ld_filterbank::{LdFilterbankError, LowDelayFilterbankF32, LowDelayFilterbankQ31};
use crate::ld_sbr::{LdSbrError, LdSbrFrame, LdSbrFrameParser};
use crate::ld_sbr_qmf::{LdSbrChannelProcessor, LdSbrProcessingError, LdSbrQmfAnalysis, QmfSlot};
use crate::pns::{
    apply_pns_f32, apply_pns_fixed, apply_pns_pair_f32, apply_pns_pair_fixed, PnsError,
    PnsRandomState,
};
use crate::ps::{PsError, PsFrame, PsParser, PsQmfProcessor};
use crate::pulse::{PulseData, PulseError};
use crate::raw::{
    ChannelPairElementSideInfoPrefix, CouplingChannelElementPrefix, CouplingPoint, ElementId,
    RawError, SingleChannelElementSideInfo,
};
use crate::rvlc::{
    conceal_scalefactors as conceal_rvlc_scalefactors, decode_forward as decode_rvlc_forward,
    RvlcError, RvlcSideInfo,
};
use crate::sac::{Sac212Decoder, SpatialSpecificConfig};
use crate::sbr::{
    parse_sbr_fill_element, SbrError, SbrFillPayload, SbrMonoFrame, SbrMonoFrameParser,
    SbrStereoFrame, SbrStereoFrameParser,
};
use crate::scalefactor::{ScalefactorData, ScalefactorError, ScalefactorPlan};
use crate::section::{
    SectionData, SectionError, INTENSITY_HCB, INTENSITY_HCB2, NOISE_HCB, ZERO_HCB,
};
use crate::sfb::{
    aac_band_offsets_for_ics, aac_lc_band_offsets_for_ics, aac_lc_sfb_info, SfbError,
};
use crate::spectral::{decode_spectral_data, SpectralData, SpectralError};
use crate::stereo::{
    apply_intensity_stereo_f32, apply_ms_stereo_f32, intensity_scale_f32, MsMaskPresent,
    MsStereoData, StereoError,
};
use crate::tns::{TnsData, TnsError};
use crate::usac_decoder::{
    UsacDecodeError, UsacDecodedFrame, UsacMonoDecoder, UsacMps212AccessUnit, UsacMps212Decoder,
    UsacMultichannelDecoder, UsacStereoDecoder,
};

#[derive(Debug, Clone, PartialEq)]
pub struct DecodedSingleChannelFrame {
    pub side_info: SingleChannelElementSideInfo,
    pub section_data: SectionData,
    pub scalefactors: ScalefactorData,
    pub pulse_data: PulseData,
    pub tns_data: TnsData,
    pub spectral: SpectralData,
    pub spectrum: InverseQuantizedSpectrum,
    pub samples: Vec<f32>,
    pub bits_read: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodedChannelPairSpectra {
    pub prefix: ChannelPairElementSideInfoPrefix,
    pub ms_stereo: Option<MsStereoData>,
    pub left: DecodedChannelStream,
    pub right: DecodedChannelStream,
    /// Start of the second channel, relative to the first bit after ID_CPE.
    pub right_channel_start_bit: usize,
    pub bits_read: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodedChannelPairSpectraFixed {
    pub prefix: ChannelPairElementSideInfoPrefix,
    pub ms_stereo: Option<MsStereoData>,
    pub left: DecodedChannelStreamFixed,
    pub right: DecodedChannelStreamFixed,
    pub right_channel_start_bit: usize,
    pub bits_read: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodedChannelPairFrame {
    pub spectra: DecodedChannelPairSpectra,
    pub left_samples: Vec<f32>,
    pub right_samples: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodedSingleChannelSpectra {
    pub side_info: SingleChannelElementSideInfo,
    pub stream: DecodedChannelStream,
    pub bits_read: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodedChannelStream {
    pub global_gain: u8,
    pub ics: IcsInfo,
    pub section_data: SectionData,
    pub scalefactors: ScalefactorData,
    pub pulse_data: PulseData,
    pub tns_data: TnsData,
    pub spectral: SpectralData,
    pub spectrum: InverseQuantizedSpectrum,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodedSingleChannelSpectraFixed {
    pub side_info: SingleChannelElementSideInfo,
    pub stream: DecodedChannelStreamFixed,
    pub bits_read: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodedChannelStreamFixed {
    pub global_gain: u8,
    pub ics: IcsInfo,
    pub section_data: SectionData,
    pub scalefactors: ScalefactorData,
    pub pulse_data: PulseData,
    pub tns_data: TnsData,
    pub spectral: SpectralData,
    pub spectrum: FixedInverseQuantizedSpectrum,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodedCouplingChannelElement {
    pub prefix: CouplingChannelElementPrefix,
    pub stream: DecodedChannelStream,
    pub gain_lists: CouplingGainElementLists,
    pub bits_read: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodedCouplingChannelElementFixed {
    pub prefix: CouplingChannelElementPrefix,
    pub stream: DecodedChannelStreamFixed,
    pub gain_lists: CouplingGainElementLists,
    pub bits_read: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CouplingGainElementLists {
    pub lists: Vec<CouplingGainElementList>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CouplingGainElementList {
    pub common_gain_element_present: bool,
    pub words: Vec<i16>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CouplingTargetSpectrum {
    pub element_id: ElementId,
    pub element_instance_tag: u8,
    pub channel: usize,
    pub spectrum: InverseQuantizedSpectrum,
}

#[derive(Debug, Clone, PartialEq)]
enum StagedAacLcElement {
    Single {
        element_id: ElementId,
        element_instance_tag: u8,
        spectra: DecodedSingleChannelSpectra,
        labels: Vec<ChannelLabel>,
    },
    Pair {
        element_instance_tag: u8,
        spectra: DecodedChannelPairSpectra,
        labels: Vec<ChannelLabel>,
    },
}

#[derive(Debug, Clone, PartialEq)]
enum StagedAacLcElementFixed {
    Single {
        element_id: ElementId,
        element_instance_tag: u8,
        spectra: DecodedSingleChannelSpectraFixed,
        labels: Vec<ChannelLabel>,
    },
    Pair {
        element_instance_tag: u8,
        spectra: DecodedChannelPairSpectraFixed,
        labels: Vec<ChannelLabel>,
    },
}

#[derive(Debug, Clone)]
enum OrdinarySbrParser {
    Mono(SbrMonoFrameParser),
    Stereo(SbrStereoFrameParser),
}

#[derive(Debug, Clone)]
enum OrdinarySbrFrame {
    Mono(SbrMonoFrame),
    Stereo(SbrStereoFrame),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StagedChannelMapEntry {
    element_id: ElementId,
    element_instance_tag: u8,
    channel: usize,
    output_channel: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DecodedAacLcFrame {
    Mono(DecodedSingleChannelFrame),
    Stereo(DecodedChannelPairFrame),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelLabel {
    Empty,
    FrontCenter,
    FrontLeft,
    FrontRight,
    FrontLeftCenter,
    FrontRightCenter,
    SideLeft,
    SideRight,
    BackLeft,
    BackRight,
    BackCenter,
    Lfe,
    Unknown(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConcealmentState {
    Ok,
    Single,
    FadeOut,
    Mute,
    FadeIn,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedConcealmentSpectralFrame {
    pub channels: Vec<FixedConcealmentChannel>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedConcealmentChannel {
    pub spectrum: FixedInverseQuantizedSpectrum,
    pub ics: IcsInfo,
}

#[derive(Debug, Clone, PartialEq)]
pub struct F32ConcealmentSpectralFrame {
    pub channels: Vec<F32ConcealmentChannel>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct F32ConcealmentChannel {
    pub spectrum: InverseQuantizedSpectrum,
    pub ics: IcsInfo,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LegacyDrcParameters {
    pub attenuation_scale: f32,
    pub boost_scale: f32,
    pub target_reference_level: Option<u8>,
    pub heavy_compression: bool,
    pub default_presentation_mode: i8,
    pub encoder_target_level: u8,
}

#[derive(Debug, Clone)]
struct LegacyQmfDrcFrame {
    band_top: Vec<u8>,
    gains: Vec<f64>,
    interpolation_scheme: u8,
    window_sequence: WindowSequence,
}

impl LegacyQmfDrcFrame {
    fn unity() -> Self {
        Self {
            band_top: vec![255],
            gains: vec![1.0],
            interpolation_scheme: 0,
            window_sequence: WindowSequence::OnlyLong,
        }
    }

    fn gain_for_band(&self, band: usize) -> f64 {
        self.gains.get(band).copied().unwrap_or(1.0)
    }
}

#[derive(Debug, Clone)]
struct LegacyQmfDrcState {
    current: LegacyQmfDrcFrame,
    previous_gains: Vec<f64>,
    enabled: bool,
}

impl Default for LegacyQmfDrcState {
    fn default() -> Self {
        Self {
            current: LegacyQmfDrcFrame::unity(),
            previous_gains: vec![1.0; 64],
            enabled: false,
        }
    }
}

impl Default for LegacyDrcParameters {
    fn default() -> Self {
        Self {
            // libAACdec deliberately defaults legacy MPEG-4 boost/cut to zero.
            attenuation_scale: 0.0,
            boost_scale: 0.0,
            target_reference_level: Some(96),
            heavy_compression: false,
            default_presentation_mode: -1,
            encoder_target_level: 127,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AncillaryDataElement {
    pub element_instance_tag: u8,
    pub data: Vec<u8>,
}

/// Snapshot of the stream properties exposed by the Pure Rust decoder.
///
/// The fields intentionally mirror libAACdec's `CStreamInfo` surface while
/// using owned Rust channel arrays. The direct raw decoder leaves transport
/// counters at zero; [`crate::transport::PureRustTransportDecoder`] overlays
/// the counters collected by its transport parser.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecoderStreamInfo {
    pub sample_rate: u32,
    pub frame_size: usize,
    pub num_channels: usize,
    pub channel_labels: Vec<ChannelLabel>,
    pub channel_indices: Vec<u8>,
    pub aac_sample_rate: u32,
    pub profile: i32,
    pub audio_object_type: u8,
    pub channel_configuration: u8,
    pub bit_rate: u32,
    pub aac_samples_per_frame: usize,
    pub aac_num_channels: usize,
    pub extension_audio_object_type: Option<u8>,
    pub extension_sampling_rate: Option<u32>,
    pub output_delay: usize,
    pub flags: u32,
    pub error_protection_config: i8,
    pub num_lost_access_units: i32,
    pub num_total_bytes: u64,
    pub num_bad_bytes: u64,
    pub num_total_access_units: u64,
    pub num_bad_access_units: u64,
    pub drc_program_reference_level: i8,
    pub drc_presentation_mode: i8,
    pub output_loudness: i32,
}

pub const STREAM_FLAG_ER_VCB11: u32 = 0x000001;
pub const STREAM_FLAG_ER_RVLC: u32 = 0x000002;
pub const STREAM_FLAG_ER_HCR: u32 = 0x000004;
pub const STREAM_FLAG_ELD: u32 = 0x000010;
pub const STREAM_FLAG_LD: u32 = 0x000020;
pub const STREAM_FLAG_ER: u32 = 0x000040;
pub const STREAM_FLAG_USAC: u32 = 0x000100;
pub const STREAM_FLAG_SBR_PRESENT: u32 = 0x008000;
pub const STREAM_FLAG_SBR_CRC: u32 = 0x010000;
pub const STREAM_FLAG_PS_PRESENT: u32 = 0x020000;
pub const STREAM_FLAG_MPS_PRESENT: u32 = 0x040000;
pub const STREAM_FLAG_DRC_PRESENT: u32 = 0x0400000;

#[derive(Debug, Clone)]
enum DecoderInitialization {
    General {
        audio_object_type: u8,
        sampling_frequency_index: u8,
        channel_configuration: u8,
        frame_length: usize,
    },
    AudioSpecificConfig(Box<AudioSpecificConfig>),
    Drm {
        sampling_frequency_index: u8,
        channel_configuration: u8,
    },
}

#[derive(Debug, Clone)]
pub struct AacLcDecoder {
    initialization: DecoderInitialization,
    audio_object_type: u8,
    error_protection_config: Option<u8>,
    er_resilience_flags: [bool; 3],
    frame_length: usize,
    sampling_frequency_index: u8,
    channel_configuration: u8,
    channel_filterbanks: Vec<LongBlockFilterbank>,
    eld_channel_filterbanks: Vec<LowDelayFilterbankF32>,
    eld_fixed_channel_filterbanks: Vec<LowDelayFilterbankQ31>,
    ld_sbr_parsers: Vec<LdSbrFrameParser>,
    ld_sbr_channel_indices: Vec<usize>,
    ld_sbr_processors: Vec<LdSbrChannelProcessor>,
    ld_sbr_fixed_processors: Vec<LdSbrChannelProcessor>,
    last_ld_sbr_frames: Vec<LdSbrFrame>,
    ordinary_sbr_output_frequency: Option<u32>,
    extension_audio_object_type: Option<u8>,
    eld_sbr_dual_rate: bool,
    eld_sbr_crc: bool,
    ordinary_sbr_parsers: Vec<Option<OrdinarySbrParser>>,
    ordinary_sbr_processors: Vec<LdSbrChannelProcessor>,
    ordinary_sbr_fixed_parsers: Vec<Option<OrdinarySbrParser>>,
    ordinary_sbr_fixed_processors: Vec<LdSbrChannelProcessor>,
    last_ordinary_sbr_frames: Vec<OrdinarySbrFrame>,
    last_ordinary_sbr_fixed_frames: Vec<OrdinarySbrFrame>,
    ps_signaled: bool,
    ps_parsers: Vec<PsParser>,
    ps_processors: Vec<PsQmfProcessor>,
    ps_fixed_parsers: Vec<PsParser>,
    ps_fixed_processors: Vec<PsQmfProcessor>,
    last_ps_frames: Vec<Option<PsFrame>>,
    last_ps_fixed_frames: Vec<Option<PsFrame>>,
    qmf_low_power: bool,
    drc_config: Option<UniDrcConfig>,
    drc_gain: Option<UniDrcGain>,
    drc_loudness_info: Option<LoudnessInfoSet>,
    drc_selection: DrcSelectionRequest,
    legacy_drc_payload: Option<Mpeg4DrcPayload>,
    legacy_dvb_drc_payload: Option<DvbAncillaryDrcPayload>,
    legacy_drc_presentation_mode: i8,
    legacy_dvb_downmix_metadata: Option<DvbAncillaryDownmixMetadata>,
    legacy_matrix_mixdown: Option<MatrixMixdown>,
    legacy_drc_parameters: LegacyDrcParameters,
    legacy_drc_output_channels: usize,
    legacy_qmf_drc_states: Vec<LegacyQmfDrcState>,
    legacy_qmf_drc_fixed_states: Vec<LegacyQmfDrcState>,
    legacy_drc_window_sequences: Vec<WindowSequence>,
    legacy_drc_expiry_frames: usize,
    legacy_drc_age_frames: usize,
    legacy_norm_gain_previous: f32,
    legacy_norm_filter_state: f32,
    legacy_norm_filter_input_previous: f32,
    legacy_drc_control_applied: bool,
    coupling_filterbanks: Vec<LongBlockFilterbank>,
    fixed_channel_filterbanks: Vec<FixedLongBlockFilterbank>,
    fixed_coupling_filterbanks: Vec<FixedLongBlockFilterbank>,
    fixed_concealment_spectra: Vec<(FixedInverseQuantizedSpectrum, IcsInfo)>,
    fixed_concealment_losses: usize,
    fixed_concealment_phase: u32,
    fixed_concealment_state: ConcealmentState,
    fixed_concealment_fade_in_remaining: usize,
    f32_concealment_spectra: Vec<(InverseQuantizedSpectrum, IcsInfo)>,
    f32_concealment_phase: u32,
    f32_concealment_losses: usize,
    f32_concealment_state: ConcealmentState,
    f32_concealment_fade_in_remaining: usize,
    pns_random: PnsRandomState,
    adts_crc_regions: Vec<std::ops::Range<usize>>,
    adts_crc_padded_bits: Vec<usize>,
    ancillary_data_capacity: Option<usize>,
    ancillary_data: Vec<AncillaryDataElement>,
    usac_decoder: Option<UsacMonoDecoder>,
    usac_stereo_decoder: Option<UsacStereoDecoder>,
    usac_mps212_decoder: Option<UsacMps212Decoder>,
    usac_multichannel_decoder: Option<UsacMultichannelDecoder>,
    usac_extension_elements: Vec<crate::asc::UsacExtElementConfig>,
    usac_leading_extension_count: usize,
    usac_multichannel_extension_boundaries: Vec<usize>,
    pending_usac_audio_preroll: Option<crate::audio_preroll::AudioPreRoll>,
    usac_preroll_depth: usize,
    usac_last_output: Option<Vec<Vec<f32>>>,
    usac_crossfade_source: Option<Vec<Vec<f32>>>,
    eld_sac_decoder: Option<Sac212Decoder>,
    eld_sac_analysis: Option<LdSbrQmfAnalysis>,
    eld_sac_payload: Option<(Vec<u8>, usize)>,
}

impl AacLcDecoder {
    #[cfg(test)]
    pub(crate) fn last_ld_sbr_frames(&self) -> &[LdSbrFrame] {
        &self.last_ld_sbr_frames
    }

    pub(crate) fn last_adts_crc_regions(&self) -> Vec<(std::ops::Range<usize>, usize)> {
        self.adts_crc_regions
            .iter()
            .cloned()
            .zip(self.adts_crc_padded_bits.iter().copied())
            .collect()
    }

    fn push_adts_crc_region(&mut self, range: std::ops::Range<usize>, padded_bits: usize) {
        self.adts_crc_regions.push(range);
        self.adts_crc_padded_bits.push(padded_bits);
    }

    fn clear_adts_crc_regions(&mut self) {
        self.adts_crc_regions.clear();
        self.adts_crc_padded_bits.clear();
    }

    pub fn new(
        sampling_frequency_index: u8,
        channel_configuration: u8,
    ) -> Result<Self, DecodeError> {
        Self::new_ga(2, sampling_frequency_index, channel_configuration)
    }

    pub fn new_ga(
        audio_object_type: u8,
        sampling_frequency_index: u8,
        channel_configuration: u8,
    ) -> Result<Self, DecodeError> {
        let frame_length = if matches!(audio_object_type, 23 | 39) {
            512
        } else {
            1024
        };
        Self::new_ga_with_frame_length(
            audio_object_type,
            sampling_frequency_index,
            channel_configuration,
            frame_length,
        )
    }

    fn new_ga_with_frame_length(
        audio_object_type: u8,
        sampling_frequency_index: u8,
        channel_configuration: u8,
        frame_length: usize,
    ) -> Result<Self, DecodeError> {
        if !matches!(audio_object_type, 2 | 17 | 20 | 23 | 39 | 42) {
            return Err(DecodeError::UnsupportedAudioObjectType(audio_object_type));
        }
        if !matches!(
            (audio_object_type, frame_length),
            (2 | 17 | 20, 960 | 1024) | (23 | 39, 480 | 512) | (42, 768 | 1024)
        ) {
            return Err(DecodeError::UnsupportedFrameLength(frame_length));
        }
        if sampling_frequency_index >= 13 {
            return Err(DecodeError::UnsupportedSamplingFrequencyIndex(
                sampling_frequency_index,
            ));
        }
        let valid_channel_configuration = if audio_object_type == 42 {
            crate::asc::usac_channel_count(channel_configuration).is_some()
        } else {
            matches!(channel_configuration, 0..=7 | 11 | 12 | 14)
        };
        if !valid_channel_configuration {
            return Err(DecodeError::UnsupportedChannelConfiguration(
                channel_configuration,
            ));
        }
        let channels = (audio_object_type == 42)
            .then(|| crate::asc::usac_channel_count(channel_configuration))
            .flatten()
            .or_else(|| expected_channels_for_config(channel_configuration))
            .unwrap_or(2)
            .max(2);

        Ok(Self {
            initialization: DecoderInitialization::General {
                audio_object_type,
                sampling_frequency_index,
                channel_configuration,
                frame_length,
            },
            audio_object_type,
            error_protection_config: matches!(audio_object_type, 17 | 20 | 23 | 39).then_some(0),
            er_resilience_flags: [false; 3],
            frame_length,
            sampling_frequency_index,
            channel_configuration,
            channel_filterbanks: (0..channels)
                .map(|_| LongBlockFilterbank::new(frame_length))
                .collect::<Result<Vec<_>, _>>()?,
            eld_channel_filterbanks: if audio_object_type == 39 {
                (0..channels)
                    .map(|_| LowDelayFilterbankF32::new(frame_length))
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                Vec::new()
            },
            eld_fixed_channel_filterbanks: if audio_object_type == 39 {
                (0..channels)
                    .map(|_| LowDelayFilterbankQ31::new(frame_length))
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                Vec::new()
            },
            ld_sbr_parsers: Vec::new(),
            ld_sbr_channel_indices: Vec::new(),
            ld_sbr_processors: Vec::new(),
            ld_sbr_fixed_processors: Vec::new(),
            last_ld_sbr_frames: Vec::new(),
            ordinary_sbr_output_frequency: None,
            extension_audio_object_type: None,
            eld_sbr_dual_rate: false,
            eld_sbr_crc: false,
            ordinary_sbr_parsers: Vec::new(),
            ordinary_sbr_processors: Vec::new(),
            ordinary_sbr_fixed_parsers: Vec::new(),
            ordinary_sbr_fixed_processors: Vec::new(),
            last_ordinary_sbr_frames: Vec::new(),
            last_ordinary_sbr_fixed_frames: Vec::new(),
            ps_signaled: false,
            ps_parsers: Vec::new(),
            ps_processors: Vec::new(),
            ps_fixed_parsers: Vec::new(),
            ps_fixed_processors: Vec::new(),
            last_ps_frames: Vec::new(),
            last_ps_fixed_frames: Vec::new(),
            qmf_low_power: false,
            drc_config: None,
            drc_gain: None,
            drc_loudness_info: None,
            drc_selection: DrcSelectionRequest::default(),
            legacy_drc_payload: None,
            legacy_dvb_drc_payload: None,
            legacy_drc_presentation_mode: -1,
            legacy_dvb_downmix_metadata: None,
            legacy_matrix_mixdown: None,
            legacy_drc_parameters: LegacyDrcParameters::default(),
            legacy_drc_output_channels: channels,
            legacy_qmf_drc_states: Vec::new(),
            legacy_qmf_drc_fixed_states: Vec::new(),
            legacy_drc_window_sequences: Vec::new(),
            legacy_drc_expiry_frames: 0,
            legacy_drc_age_frames: 0,
            legacy_norm_gain_previous: 1.0,
            legacy_norm_filter_state: 1.0,
            legacy_norm_filter_input_previous: 1.0,
            legacy_drc_control_applied: false,
            coupling_filterbanks: Vec::new(),
            fixed_channel_filterbanks: (0..channels)
                .map(|_| FixedLongBlockFilterbank::new(frame_length))
                .collect::<Result<Vec<_>, _>>()?,
            fixed_coupling_filterbanks: Vec::new(),
            fixed_concealment_spectra: Vec::new(),
            fixed_concealment_losses: 0,
            fixed_concealment_phase: 0,
            fixed_concealment_state: ConcealmentState::Ok,
            fixed_concealment_fade_in_remaining: 0,
            f32_concealment_spectra: Vec::new(),
            f32_concealment_phase: 0,
            f32_concealment_losses: 0,
            f32_concealment_state: ConcealmentState::Ok,
            f32_concealment_fade_in_remaining: 0,
            pns_random: PnsRandomState::new(0x1f2e_3d4c),
            adts_crc_regions: Vec::new(),
            adts_crc_padded_bits: Vec::new(),
            ancillary_data_capacity: None,
            ancillary_data: Vec::new(),
            usac_decoder: None,
            usac_stereo_decoder: None,
            usac_mps212_decoder: None,
            usac_multichannel_decoder: None,
            usac_extension_elements: Vec::new(),
            usac_leading_extension_count: 0,
            usac_multichannel_extension_boundaries: Vec::new(),
            pending_usac_audio_preroll: None,
            usac_preroll_depth: 0,
            usac_last_output: None,
            usac_crossfade_source: None,
            eld_sac_decoder: None,
            eld_sac_analysis: None,
            eld_sac_payload: None,
        })
    }

    pub fn from_adts_header(header: crate::adts::AdtsHeader) -> Result<Self, DecodeError> {
        if header.profile != 1 {
            return Err(DecodeError::UnsupportedAudioObjectType(header.profile + 1));
        }
        Self::new_ga(
            header.profile + 1,
            header.sampling_frequency_index,
            header.channel_configuration,
        )
    }

    pub fn from_audio_specific_config(config: &AudioSpecificConfig) -> Result<Self, DecodeError> {
        if config.audio_object_type == 42 {
            let usac = config
                .usac_config
                .clone()
                .ok_or(DecodeError::UnsupportedAudioObjectType(42))?;
            if !crate::asc::usac_element_layout_matches(
                usac.channel_configuration_index,
                &usac.elements,
            ) {
                return Err(DecodeError::UnsupportedChannelConfiguration(
                    config.channel_configuration,
                ));
            }
            let mut decoder = Self::new_ga_with_frame_length(
                42,
                usac.sampling_frequency_index,
                config.channel_configuration,
                usize::from(usac.core_frame_length),
            )?;
            let mut saw_core_element = false;
            let mut extension_count = 0usize;
            let mut extension_boundaries = vec![0usize];
            for element in &usac.elements {
                if let crate::asc::UsacElementConfig::Extension(extension) = element {
                    if extension.extension_type == 4 {
                        decoder.drc_config =
                            Some(UniDrcConfig::parse_foundation(&extension.config)?);
                    }
                    decoder.usac_extension_elements.push(extension.clone());
                    extension_count += 1;
                    if !saw_core_element {
                        decoder.usac_leading_extension_count += 1;
                    }
                } else {
                    saw_core_element = true;
                    extension_boundaries.push(extension_count);
                }
            }
            extension_boundaries.push(extension_count);
            decoder.usac_multichannel_extension_boundaries = extension_boundaries;
            for extension in &usac.extensions {
                if extension.extension_type == 2 {
                    decoder.drc_loudness_info = Some(LoudnessInfoSet::parse_v0(&extension.data)?);
                }
            }
            let mut core_usac = usac.clone();
            core_usac
                .elements
                .retain(|element| !matches!(element, crate::asc::UsacElementConfig::Extension(_)));
            if core_usac.elements.len() > 1 {
                decoder.usac_multichannel_decoder =
                    Some(UsacMultichannelDecoder::new(core_usac).map_err(|_| {
                        DecodeError::UnsupportedChannelConfiguration(config.channel_configuration)
                    })?);
            } else if usac.channel_configuration_index == 1 {
                decoder.usac_decoder = Some(UsacMonoDecoder::new(core_usac).map_err(|_| {
                    DecodeError::UnsupportedChannelConfiguration(config.channel_configuration)
                })?);
            } else if usac.elements.iter().any(|element| {
                matches!(
                    element,
                    crate::asc::UsacElementConfig::ChannelPair {
                        stereo_config_index: 1 | 2 | 3,
                        mps212: Some(_),
                        ..
                    }
                )
            }) {
                decoder.usac_mps212_decoder =
                    Some(UsacMps212Decoder::new(core_usac).map_err(|_| {
                        DecodeError::UnsupportedChannelConfiguration(config.channel_configuration)
                    })?);
            } else {
                decoder.usac_stereo_decoder =
                    Some(UsacStereoDecoder::new(core_usac).map_err(|_| {
                        DecodeError::UnsupportedChannelConfiguration(config.channel_configuration)
                    })?);
            }
            decoder.initialization =
                DecoderInitialization::AudioSpecificConfig(Box::new(config.clone()));
            let low_power = decoder.automatic_qmf_low_power();
            decoder.set_qmf_low_power(low_power);
            return Ok(decoder);
        }
        if !matches!(config.audio_object_type, 2 | 17 | 20 | 23 | 39) {
            return Err(DecodeError::UnsupportedAudioObjectType(
                config.audio_object_type,
            ));
        }
        if config
            .extension
            .is_some_and(|extension| !matches!(extension.audio_object_type, 5 | 29))
        {
            return Err(DecodeError::UnsupportedAudioObjectType(
                config.extension.unwrap().audio_object_type,
            ));
        }
        if matches!(config.audio_object_type, 17 | 20 | 23 | 39) {
            if config.channel_configuration == 0 {
                return Err(DecodeError::UnsupportedChannelConfiguration(0));
            }
            if config.audio_object_type == 20
                && config.ga_specific.and_then(|ga| ga.layer).unwrap_or(1) != 0
            {
                return Err(DecodeError::UnsupportedAudioObjectType(20));
            }
        }
        let frame_length = if matches!(config.audio_object_type, 2 | 17 | 20) {
            if config.ga_specific.is_some_and(|ga| ga.frame_length_flag) {
                960
            } else {
                1024
            }
        } else if matches!(config.audio_object_type, 23 | 39) {
            let frame_length_flag = if config.audio_object_type == 39 {
                config
                    .eld_specific
                    .as_ref()
                    .ok_or(DecodeError::UnsupportedAudioObjectType(39))?
                    .frame_length_flag
            } else {
                config.ga_specific.is_some_and(|ga| ga.frame_length_flag)
            };
            if frame_length_flag {
                480
            } else {
                512
            }
        } else {
            1024
        };
        let mut decoder = Self::new_ga_with_frame_length(
            config.audio_object_type,
            config.sampling_frequency_index,
            config.channel_configuration,
            frame_length,
        )?;
        if config.channel_configuration == 0 {
            let channels = config
                .program_config
                .as_ref()
                .map(|pce| pce.num_channels as usize)
                .ok_or(DecodeError::UnsupportedChannelConfiguration(0))?;
            decoder.ensure_channel_filterbanks(channels)?;
            decoder.ensure_fixed_channel_filterbanks(channels)?;
            decoder.legacy_drc_output_channels = channels;
        }
        decoder.error_protection_config = config.error_protection_config;
        if let Some(extension) = config.extension {
            let channel_count = expected_channels_for_config(config.channel_configuration)
                .or_else(|| {
                    config
                        .program_config
                        .as_ref()
                        .map(|pce| pce.num_channels as usize)
                })
                .ok_or(DecodeError::UnsupportedChannelConfiguration(
                    config.channel_configuration,
                ))?;
            let element_count = er_channel_elements(config.channel_configuration)
                .map(<[ElementId]>::len)
                .or_else(|| {
                    config.program_config.as_ref().map(|pce| {
                        pce.front.len() + pce.side.len() + pce.back.len() + pce.lfe.len()
                    })
                })
                .ok_or(DecodeError::UnsupportedChannelConfiguration(
                    config.channel_configuration,
                ))?;
            decoder.ordinary_sbr_output_frequency = Some(extension.sampling_frequency);
            decoder.extension_audio_object_type = Some(extension.audio_object_type);
            decoder.ordinary_sbr_parsers = vec![None; element_count];
            decoder.ordinary_sbr_processors = (0..channel_count)
                .map(|channel| {
                    LdSbrChannelProcessor::new(
                        extension.sampling_frequency,
                        true,
                        0x2468_ace0 ^ channel as u32,
                    )
                })
                .collect();
            decoder.ordinary_sbr_fixed_parsers = vec![None; element_count];
            decoder.ordinary_sbr_fixed_processors = (0..channel_count)
                .map(|channel| {
                    LdSbrChannelProcessor::new(
                        extension.sampling_frequency,
                        true,
                        0x2468_ace0 ^ channel as u32,
                    )
                })
                .collect();
            decoder.ps_signaled = extension.ps_present;
            decoder.ps_parsers = (0..element_count).map(|_| PsParser::new()).collect();
            decoder.ps_processors = (0..element_count).map(|_| PsQmfProcessor::new()).collect();
            decoder.ps_fixed_parsers = (0..element_count).map(|_| PsParser::new()).collect();
            decoder.ps_fixed_processors =
                (0..element_count).map(|_| PsQmfProcessor::new()).collect();
            decoder.last_ps_frames = vec![None; element_count];
            decoder.last_ps_fixed_frames = vec![None; element_count];
        }
        if let Some(ga) = config.ga_specific {
            decoder.er_resilience_flags = [
                ga.section_data_resilience,
                ga.scalefactor_data_resilience,
                ga.spectral_data_resilience,
            ];
        } else if let Some(eld) = &config.eld_specific {
            decoder.er_resilience_flags = [
                eld.section_data_resilience,
                eld.scalefactor_data_resilience,
                eld.spectral_data_resilience,
            ];
        }
        if let Some(eld) = config.eld_specific.as_ref().filter(|eld| eld.sbr_present) {
            decoder.extension_audio_object_type = Some(5);
            decoder.eld_sbr_dual_rate = eld.sbr_sampling_rate;
            decoder.eld_sbr_crc = eld.sbr_crc;
            let elements = er_channel_elements(config.channel_configuration)
                .expect("ER configuration validation requires channelConfiguration 1..=7");
            decoder.ld_sbr_channel_indices = er_sbr_channel_indices(elements);
            // FDK uses twice the core rate for SBR frequency-table derivation
            // in both single-rate (32-band synthesis) and dual-rate modes.
            let processing_frequency = config.sampling_frequency * 2;
            decoder.ld_sbr_parsers = eld
                .sbr_headers
                .iter()
                .zip(
                    elements
                        .iter()
                        .filter(|&&element| element != ElementId::Lfe),
                )
                .map(|(header, element)| {
                    LdSbrFrameParser::new(
                        header.clone(),
                        processing_frequency,
                        frame_length,
                        *element == ElementId::ChannelPair,
                        eld.sbr_crc,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let channel_count = expected_channels_for_config(config.channel_configuration)
                .expect("ER configuration validation requires channelConfiguration 1..=7");
            decoder.ld_sbr_processors = (0..channel_count)
                .map(|channel| {
                    LdSbrChannelProcessor::new_eld(
                        processing_frequency,
                        eld.sbr_sampling_rate,
                        0x1357_9bdf ^ channel as u32,
                    )
                })
                .collect();
            decoder.ld_sbr_fixed_processors = (0..channel_count)
                .map(|channel| {
                    LdSbrChannelProcessor::new_eld(
                        processing_frequency,
                        eld.sbr_sampling_rate,
                        0x1357_9bdf ^ channel as u32,
                    )
                })
                .collect();
        }
        if let Some(extension) = config.eld_specific.as_ref().and_then(|eld| {
            eld.extensions
                .iter()
                .find(|extension| extension.extension_type == 2)
        }) {
            let spatial = SpatialSpecificConfig::parse(&extension.data)
                .map_err(|_| DecodeError::MpsSpatialConfiguration)?;
            let sbr_rate_multiplier = config
                .eld_specific
                .as_ref()
                .filter(|eld| eld.sbr_present && eld.sbr_sampling_rate)
                .map_or(1usize, |_| 2);
            let spatial_sampling_frequency = config
                .sampling_frequency
                .saturating_mul(sbr_rate_multiplier as u32);
            let spatial_frame_length = frame_length.saturating_mul(sbr_rate_multiplier);
            if spatial.sampling_frequency != spatial_sampling_frequency
                || usize::from(spatial.time_slots)
                    * if spatial.sampling_frequency < 27_713 {
                        32
                    } else {
                        64
                    }
                    != spatial_frame_length
            {
                return Err(DecodeError::MpsSpatialConfiguration);
            }
            let qmf_bands = if spatial.sampling_frequency < 27_713 {
                32
            } else {
                64
            };
            decoder.eld_sac_analysis = Some(
                LdSbrQmfAnalysis::new_with_channels(qmf_bands)
                    .map_err(|_| DecodeError::MpsSpatialConfiguration)?,
            );
            decoder.eld_sac_decoder = Some(
                Sac212Decoder::new(spatial).map_err(|_| DecodeError::MpsSpatialConfiguration)?,
            );
        }
        decoder.initialization =
            DecoderInitialization::AudioSpecificConfig(Box::new(config.clone()));
        decoder.legacy_matrix_mixdown = config
            .program_config
            .as_ref()
            .and_then(|program| program.matrix_mixdown);
        let low_power = decoder.automatic_qmf_low_power();
        decoder.set_qmf_low_power(low_power);
        Ok(decoder)
    }

    pub fn from_adif_header(header: &AdifHeader) -> Result<Self, DecodeError> {
        let pce = header
            .last_program_config()
            .ok_or(DecodeError::NoAudioElement)?;
        Self::new_ga(pce.profile + 1, pce.sampling_frequency_index, 0)
    }

    pub(crate) fn new_drm_aac(
        sampling_frequency_index: u8,
        channel_configuration: u8,
    ) -> Result<Self, DecodeError> {
        let mut decoder = Self::new_ga_with_frame_length(
            17,
            sampling_frequency_index,
            channel_configuration,
            960,
        )?;
        decoder.error_protection_config = Some(1);
        decoder.er_resilience_flags = [true, false, true];
        decoder.initialization = DecoderInitialization::Drm {
            sampling_frequency_index,
            channel_configuration,
        };
        Ok(decoder)
    }

    pub(crate) fn decode_drm_aac_mono_f32(
        &mut self,
        payload: &[u8],
    ) -> Result<(Vec<f32>, usize, usize), DecodeError> {
        if self.frame_length != 960 || self.channel_configuration != 1 {
            return Err(DecodeError::UnsupportedChannelConfiguration(
                self.channel_configuration,
            ));
        }
        let mut reader = BitReader::new(payload);
        let (spectra, protected_bits) = decode_drm_aac_single_channel_spectra_from_reader(
            &mut reader,
            self.sampling_frequency_index,
            &mut self.pns_random,
        )?;
        let core_bits = spectra.bits_read;
        let samples = synthesize_aac_lc_frame(
            &spectra.stream.spectrum,
            &spectra.stream.ics,
            &mut self.channel_filterbanks[0],
        )?;
        Ok((samples, protected_bits, core_bits))
    }

    pub(crate) fn decode_drm_aac_stereo_f32(
        &mut self,
        payload: &[u8],
    ) -> Result<([Vec<f32>; 2], usize, usize), DecodeError> {
        if self.frame_length != 960 || self.channel_configuration != 2 {
            return Err(DecodeError::UnsupportedChannelConfiguration(
                self.channel_configuration,
            ));
        }
        let mut reader = BitReader::new(payload);
        let (mut spectra, protected_bits) = decode_drm_aac_channel_pair_spectra_from_reader(
            &mut reader,
            self.sampling_frequency_index,
            &mut self.pns_random,
        )?;
        let core_bits = spectra.bits_read;
        apply_aac_lc_channel_pair_stereo_tools_fixed_bridge(
            &mut spectra,
            self.sampling_frequency_index,
        )?;
        let (left_banks, right_banks) = self.channel_filterbanks.split_at_mut(1);
        let left = synthesize_aac_lc_frame(
            &spectra.left.spectrum,
            &spectra.left.ics,
            &mut left_banks[0],
        )?;
        let right = synthesize_aac_lc_frame(
            &spectra.right.spectrum,
            &spectra.right.ics,
            &mut right_banks[0],
        )?;
        Ok(([left, right], protected_bits, core_bits))
    }

    pub(crate) fn decode_drm_aac_mono_i16(
        &mut self,
        payload: &[u8],
    ) -> Result<(Vec<i16>, usize, usize), DecodeError> {
        if self.frame_length != 960 || self.channel_configuration != 1 {
            return Err(DecodeError::UnsupportedChannelConfiguration(
                self.channel_configuration,
            ));
        }
        let pns_before = self.pns_random;
        let mut reader = BitReader::new(payload);
        let (spectra, protected_bits) = decode_drm_aac_single_channel_spectra_from_reader(
            &mut reader,
            self.sampling_frequency_index,
            &mut self.pns_random,
        )?;
        let core_bits = spectra.bits_read;
        self.pns_random = pns_before;
        let sfb =
            aac_band_offsets_for_ics(self.sampling_frequency_index, &spectra.stream.ics, 960)?;
        let mut fixed = inverse_quantize_spectrum_fixed(
            &spectra.stream.spectral,
            &spectra.stream.scalefactors,
            &spectra.stream.ics,
            sfb,
        )?;
        apply_pns_fixed(
            &mut fixed,
            &spectra.stream.ics,
            sfb.offsets,
            &spectra.stream.section_data,
            &spectra.stream.scalefactors,
            &mut self.pns_random,
        )?;
        spectra.stream.tns_data.apply_fixed(
            &mut fixed,
            sfb.offsets,
            spectra.stream.ics.max_sfb as usize,
        )?;
        let samples = synthesize_aac_lc_frame_from_fixed_inverse_q31(
            &fixed,
            &spectra.stream.ics,
            &mut self.fixed_channel_filterbanks[0],
        )?
        .into_iter()
        .map(dbl_to_pcm16)
        .collect();
        Ok((samples, protected_bits, core_bits))
    }

    pub(crate) fn decode_drm_aac_stereo_i16(
        &mut self,
        payload: &[u8],
    ) -> Result<([Vec<i16>; 2], usize, usize), DecodeError> {
        if self.frame_length != 960 || self.channel_configuration != 2 {
            return Err(DecodeError::UnsupportedChannelConfiguration(
                self.channel_configuration,
            ));
        }
        let pns_before = self.pns_random;
        let mut reader = BitReader::new(payload);
        let (spectra, protected_bits) = decode_drm_aac_channel_pair_spectra_from_reader(
            &mut reader,
            self.sampling_frequency_index,
            &mut self.pns_random,
        )?;
        let core_bits = spectra.bits_read;
        self.pns_random = pns_before;
        let sfb = aac_band_offsets_for_ics(self.sampling_frequency_index, &spectra.left.ics, 960)?;
        let mut fixed = DecodedChannelPairSpectraFixed {
            prefix: spectra.prefix.clone(),
            ms_stereo: spectra.ms_stereo.clone(),
            left: DecodedChannelStreamFixed {
                global_gain: spectra.left.global_gain,
                ics: spectra.left.ics.clone(),
                section_data: spectra.left.section_data.clone(),
                scalefactors: spectra.left.scalefactors.clone(),
                pulse_data: PulseData::absent(),
                tns_data: spectra.left.tns_data.clone(),
                spectral: spectra.left.spectral.clone(),
                spectrum: inverse_quantize_spectrum_fixed(
                    &spectra.left.spectral,
                    &spectra.left.scalefactors,
                    &spectra.left.ics,
                    sfb,
                )?,
            },
            right: DecodedChannelStreamFixed {
                global_gain: spectra.right.global_gain,
                ics: spectra.right.ics.clone(),
                section_data: spectra.right.section_data.clone(),
                scalefactors: spectra.right.scalefactors.clone(),
                pulse_data: PulseData::absent(),
                tns_data: spectra.right.tns_data.clone(),
                spectral: spectra.right.spectral.clone(),
                spectrum: inverse_quantize_spectrum_fixed(
                    &spectra.right.spectral,
                    &spectra.right.scalefactors,
                    &spectra.right.ics,
                    sfb,
                )?,
            },
            right_channel_start_bit: spectra.right_channel_start_bit,
            bits_read: spectra.bits_read,
        };
        apply_pns_pair_fixed(
            &mut fixed.left.spectrum,
            &mut fixed.right.spectrum,
            &fixed.left.ics,
            sfb.offsets,
            &fixed.left.section_data,
            &fixed.right.section_data,
            &fixed.left.scalefactors,
            &fixed.right.scalefactors,
            fixed.ms_stereo.as_ref(),
            &mut self.pns_random,
        )?;
        fixed.left.tns_data.apply_fixed(
            &mut fixed.left.spectrum,
            sfb.offsets,
            fixed.left.ics.max_sfb as usize,
        )?;
        fixed.right.tns_data.apply_fixed(
            &mut fixed.right.spectrum,
            sfb.offsets,
            fixed.right.ics.max_sfb as usize,
        )?;
        apply_aac_lc_channel_pair_fixed_spectrum_stereo_tools_bridge(
            &mut fixed,
            self.sampling_frequency_index,
        )?;
        let (left_banks, right_banks) = self.fixed_channel_filterbanks.split_at_mut(1);
        let left = synthesize_aac_lc_frame_from_fixed_inverse_q31(
            &fixed.left.spectrum,
            &fixed.left.ics,
            &mut left_banks[0],
        )?
        .into_iter()
        .map(dbl_to_pcm16)
        .collect();
        let right = synthesize_aac_lc_frame_from_fixed_inverse_q31(
            &fixed.right.spectrum,
            &fixed.right.ics,
            &mut right_banks[0],
        )?
        .into_iter()
        .map(dbl_to_pcm16)
        .collect();
        Ok(([left, right], protected_bits, core_bits))
    }

    pub fn audio_object_type(&self) -> u8 {
        self.audio_object_type
    }

    /// Enable frame-local Data Stream Element capture with a total byte limit.
    /// Up to seven non-empty elements are retained, matching libAACdec.
    pub fn init_ancillary_data(&mut self, capacity: usize) {
        self.ancillary_data_capacity = Some(capacity);
        self.ancillary_data.clear();
    }

    pub fn disable_ancillary_data(&mut self) {
        self.ancillary_data_capacity = None;
        self.ancillary_data.clear();
    }

    pub fn ancillary_data(&self) -> &[AncillaryDataElement] {
        &self.ancillary_data
    }

    pub fn legacy_downmix_metadata(&self) -> Option<DvbAncillaryDownmixMetadata> {
        self.legacy_dvb_downmix_metadata
    }

    pub fn legacy_matrix_mixdown(&self) -> Option<MatrixMixdown> {
        self.legacy_matrix_mixdown
    }

    fn read_data_stream_element(&mut self, reader: &mut BitReader<'_>) -> Result<(), DecodeError> {
        let element_instance_tag = reader.read_u8(4)?;
        let byte_align = reader.read_bool()?;
        let mut count = reader.read_u8(8)? as usize;
        if count == 255 {
            count += reader.read_u8(8)? as usize;
        }
        if byte_align {
            reader.byte_align();
        }
        let mut data = Vec::with_capacity(count);
        for _ in 0..count {
            data.push(reader.read_u8(8)?);
        }
        self.register_ancillary_data(element_instance_tag, data)
    }

    fn register_ancillary_data(
        &mut self,
        element_instance_tag: u8,
        data: Vec<u8>,
    ) -> Result<(), DecodeError> {
        if let Some(payload) = parse_dvb_ancillary_drc(&data) {
            self.legacy_drc_presentation_mode = payload.presentation_mode as i8;
            self.legacy_dvb_drc_payload = Some(payload);
            self.legacy_drc_age_frames = 0;
        }
        if let Some(update) = parse_dvb_ancillary_downmix(&data) {
            if let Some(metadata) = &mut self.legacy_dvb_downmix_metadata {
                metadata.merge(update);
            } else {
                self.legacy_dvb_downmix_metadata = Some(update);
            }
            self.legacy_drc_age_frames = 0;
        }
        let Some(capacity) = self.ancillary_data_capacity else {
            return Ok(());
        };
        if data.is_empty() {
            return Ok(());
        }
        let used = self
            .ancillary_data
            .iter()
            .map(|element| element.data.len())
            .sum::<usize>();
        if self.ancillary_data.len() >= 7 {
            return Err(DecodeError::TooManyAncillaryElements);
        }
        if used.saturating_add(data.len()) > capacity {
            return Err(DecodeError::AncillaryBufferTooSmall {
                capacity,
                required: used.saturating_add(data.len()),
            });
        }
        self.ancillary_data.push(AncillaryDataElement {
            element_instance_tag,
            data,
        });
        Ok(())
    }

    fn parse_er_extension_payloads(
        &mut self,
        reader: &mut BitReader<'_>,
    ) -> Result<(), DecodeError> {
        while reader.remaining_bits() > 7 {
            let extension_type = reader.read_u8(4)?;
            if extension_type == 0x0b {
                self.legacy_drc_payload = Some(parse_mpeg4_drc_payload(reader)?);
                self.legacy_drc_age_frames = 0;
                continue;
            }
            if extension_type == 0x09 && self.eld_sac_decoder.is_some() {
                if reader.read_u8(4)? != 0x03 {
                    return Err(DecodeError::MpsSpatialFrame);
                }
                let payload_bits = reader.remaining_bits();
                let mut writer = BitWriter::new();
                for _ in 0..payload_bits {
                    writer.write_bool(reader.read_bool()?);
                }
                self.eld_sac_payload = Some((writer.finish(), payload_bits));
                return Ok(());
            }
            if extension_type != 0x02 {
                // Without EXT_DATA_LENGTH, unknown ER payloads consume the
                // remainder of the access unit in FDK. Known SBR payloads have
                // already been consumed by `parse_ld_sbr_frames`.
                while reader.remaining_bits() != 0 {
                    reader.read_bool()?;
                }
                return Ok(());
            }
            let version = reader.read_u8(4)?;
            if version != 0 {
                return Err(DecodeError::UnsupportedAncillaryDataElementVersion(version));
            }
            let mut length = 0usize;
            loop {
                let part = reader.read_u8(8)? as usize;
                length = length.saturating_add(part);
                if part != 255 {
                    break;
                }
            }
            let mut data = Vec::with_capacity(length);
            for _ in 0..length {
                data.push(reader.read_u8(8)?);
            }
            self.register_ancillary_data(0, data)?;
        }
        Ok(())
    }

    fn process_eld_sac_f32(&mut self, channels: &mut Vec<Vec<f32>>) -> Result<(), DecodeError> {
        if self.eld_sac_decoder.is_none() {
            return Ok(());
        }
        if channels.len() != 1 {
            return Err(DecodeError::ChannelConfigurationMismatch {
                expected: 1,
                actual: channels.len(),
            });
        }
        let (payload, payload_bits) = self
            .eld_sac_payload
            .take()
            .ok_or(DecodeError::MpsSpatialFrame)?;
        let mono = channels[0]
            .iter()
            .map(|&sample| f64::from(sample))
            .collect::<Vec<_>>();
        let qmf = self
            .eld_sac_analysis
            .as_mut()
            .ok_or(DecodeError::MpsSpatialConfiguration)?
            .process_frame(&mono)
            .map_err(|_| DecodeError::MpsSpatialFrame)?;
        let (left, right) = self
            .eld_sac_decoder
            .as_mut()
            .ok_or(DecodeError::MpsSpatialConfiguration)?
            .decode_qmf(&qmf, &payload, payload_bits)
            .map_err(|_| DecodeError::MpsSpatialFrame)?;
        *channels = vec![
            left.into_iter().map(|sample| sample as f32).collect(),
            right.into_iter().map(|sample| sample as f32).collect(),
        ];
        Ok(())
    }

    fn process_eld_sac_i16(&mut self, channels: &mut Vec<Vec<i16>>) -> Result<(), DecodeError> {
        if self.eld_sac_decoder.is_none() {
            return Ok(());
        }
        let mut floating = channels
            .iter()
            .map(|channel| {
                channel
                    .iter()
                    .map(|&sample| f32::from(sample) / 32_768.0)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        self.process_eld_sac_f32(&mut floating)?;
        *channels = floating
            .into_iter()
            .map(|channel| channel.into_iter().map(f32_to_i16).collect())
            .collect();
        Ok(())
    }

    pub fn decode_usac_access_unit_f32(
        &mut self,
        input: &[u8],
    ) -> Result<UsacDecodedFrame, UsacDecodeError> {
        if self.usac_decoder.is_none() {
            return Err(UsacDecodeError::UnsupportedConfiguration);
        }
        let mut reader = BitReader::new(input);
        let independent = reader.read_bool()?;
        self.parse_usac_extension_elements(&mut reader, 0, self.usac_leading_extension_count)?;
        self.decode_pending_usac_audio_preroll()?;
        let mut frame = self
            .usac_decoder
            .as_mut()
            .ok_or(UsacDecodeError::UnsupportedConfiguration)?
            .decode_after_independent(&mut reader, independent)?;
        self.parse_usac_extension_elements(
            &mut reader,
            self.usac_leading_extension_count,
            self.usac_extension_elements.len(),
        )?;
        let mut channels = vec![frame.samples];
        self.apply_in_band_uni_drc_f32(&mut channels)?;
        self.finish_usac_output(&mut channels);
        frame.samples = channels.remove(0);
        Ok(frame)
    }

    pub fn decode_usac_access_unit_multichannel_f32(
        &mut self,
        input: &[u8],
    ) -> Result<Vec<Vec<f32>>, UsacDecodeError> {
        if self.usac_decoder.is_none()
            && self.usac_stereo_decoder.is_none()
            && self.usac_mps212_decoder.is_none()
            && self.usac_multichannel_decoder.is_none()
        {
            return Err(UsacDecodeError::UnsupportedConfiguration);
        }
        let mut reader = BitReader::new(input);
        let independent = reader.read_bool()?;
        self.parse_usac_extension_elements(&mut reader, 0, self.usac_leading_extension_count)?;
        self.decode_pending_usac_audio_preroll()?;

        let mut channels = if self.usac_decoder.is_some() {
            let frame = self
                .usac_decoder
                .as_mut()
                .expect("checked above")
                .decode_after_independent(&mut reader, independent)?;
            self.parse_usac_extension_elements(
                &mut reader,
                self.usac_leading_extension_count,
                self.usac_extension_elements.len(),
            )?;
            vec![frame.samples]
        } else if self.usac_stereo_decoder.is_some() {
            let decoded = self
                .usac_stereo_decoder
                .as_mut()
                .expect("checked above")
                .decode_after_independent(&mut reader, independent)?;
            self.parse_usac_extension_elements(
                &mut reader,
                self.usac_leading_extension_count,
                self.usac_extension_elements.len(),
            )?;
            decoded.into_iter().collect()
        } else if self.usac_mps212_decoder.is_some() {
            let access_unit = self
                .usac_mps212_decoder
                .as_mut()
                .expect("checked above")
                .decode_after_independent(&mut reader, independent)?;
            self.parse_usac_extension_elements(
                &mut reader,
                self.usac_leading_extension_count,
                self.usac_extension_elements.len(),
            )?;
            self.usac_mps212_decoder
                .as_mut()
                .expect("checked above")
                .render_access_unit(access_unit)?
                .into_iter()
                .collect()
        } else if self.usac_multichannel_decoder.is_some() {
            let element_count = self
                .usac_multichannel_decoder
                .as_ref()
                .expect("checked above")
                .element_count();
            let boundaries = self.usac_multichannel_extension_boundaries.clone();
            if boundaries.len() != element_count + 2 {
                return Err(UsacDecodeError::UnsupportedConfiguration);
            }
            let mut decoded = Vec::new();
            for element in 0..element_count {
                self.parse_usac_extension_elements(
                    &mut reader,
                    if element == 0 {
                        self.usac_leading_extension_count
                    } else {
                        boundaries[element]
                    },
                    boundaries[element + 1],
                )?;
                decoded.extend(
                    self.usac_multichannel_decoder
                        .as_mut()
                        .expect("checked above")
                        .decode_element_after_independent(element, &mut reader, independent)?,
                );
            }
            self.parse_usac_extension_elements(
                &mut reader,
                boundaries[element_count],
                boundaries[element_count + 1],
            )?;
            decoded
        } else {
            return Err(UsacDecodeError::UnsupportedConfiguration);
        };
        self.apply_in_band_uni_drc_f32(&mut channels)?;
        self.finish_usac_output(&mut channels);
        Ok(channels)
    }

    fn apply_in_band_uni_drc_f32(&self, channels: &mut [Vec<f32>]) -> Result<(), DrcError> {
        let (Some(config), Some(gain)) = (&self.drc_config, &self.drc_gain) else {
            return Ok(());
        };
        let Some(instruction) = config.select_instruction(self.drc_selection) else {
            return Ok(());
        };
        if channels.is_empty() {
            return Ok(());
        }
        let mut interleaved = interleave_multichannel_f32(channels);
        config.apply_instruction_f32_scaled(
            instruction,
            gain,
            &mut interleaved,
            channels.len(),
            self.drc_selection.attenuation_scale,
            self.drc_selection.boost_scale,
        )?;
        for (frame, samples) in interleaved.chunks_exact(channels.len()).enumerate() {
            for (channel, sample) in samples.iter().enumerate() {
                channels[channel][frame] = *sample;
            }
        }
        Ok(())
    }

    fn parse_usac_extension_elements(
        &mut self,
        reader: &mut BitReader<'_>,
        start: usize,
        end: usize,
    ) -> Result<(), UsacDecodeError> {
        for extension in self.usac_extension_elements[start..end].to_vec() {
            if !reader.read_bool()? {
                continue;
            }
            let payload_length = if reader.read_bool()? {
                extension.default_length.unwrap_or(0) as usize
            } else {
                let base = reader.read_u8(8)? as usize;
                if base == 255 {
                    base.checked_add(reader.read_u16(16)? as usize)
                        .and_then(|length| length.checked_sub(2))
                        .ok_or(UsacDecodeError::UnsupportedConfiguration)?
                } else {
                    base
                }
            };
            if payload_length == 0 {
                continue;
            }
            if extension.payload_fragmentation {
                reader.read_bool()?; // usacExtElementStart
                reader.read_bool()?; // usacExtElementStop
            }
            let payload = (0..payload_length)
                .map(|_| reader.read_u8(8))
                .collect::<Result<Vec<_>, _>>()?;
            if extension.extension_type == 3 {
                if self.pending_usac_audio_preroll.is_some() {
                    return Err(UsacDecodeError::UnsupportedConfiguration);
                }
                self.pending_usac_audio_preroll =
                    Some(crate::audio_preroll::AudioPreRoll::parse(&payload)?);
                continue;
            }
            if extension.extension_type != 4 {
                continue;
            }
            let config = self
                .drc_config
                .as_ref()
                .ok_or(UsacDecodeError::UnsupportedConfiguration)?;
            let coefficients = config
                .coefficients
                .iter()
                .find(|coefficient| coefficient.drc_location == 1)
                .or_else(|| config.coefficients.first())
                .ok_or(UsacDecodeError::UnsupportedConfiguration)?;
            let sample_rate = sample_rate_from_index(self.sampling_frequency_index)
                .ok_or(UsacDecodeError::UnsupportedConfiguration)?
                as usize;
            let half_ms = (sample_rate + 1000) / 2000;
            let mut delta_t_min = 1usize;
            while delta_t_min <= half_ms {
                delta_t_min <<= 1;
            }
            self.drc_gain = Some(coefficients.parse_gain_payload(
                &payload,
                self.frame_length,
                delta_t_min as u16,
            )?);
        }
        Ok(())
    }

    fn decode_pending_usac_audio_preroll(&mut self) -> Result<(), UsacDecodeError> {
        let Some(preroll) = self.pending_usac_audio_preroll.take() else {
            return Ok(());
        };
        if preroll.apply_crossfade {
            let channel_count = self.configured_usac_channels().max(1);
            self.usac_crossfade_source = Some(match self.usac_last_output.as_ref() {
                Some(previous) => previous
                    .iter()
                    .take(channel_count)
                    .map(|channel| {
                        let start = channel.len().saturating_sub(128);
                        let mut tail = channel[start..].to_vec();
                        if tail.len() < 128 {
                            tail.resize(128, 0.0);
                        }
                        tail
                    })
                    .collect(),
                None => vec![vec![0.0; 128]; channel_count],
            });
        }
        if !preroll.config.is_empty() {
            let usac = crate::asc::UsacConfig::parse_bytes(&preroll.config)?;
            let mut asc = match &self.initialization {
                DecoderInitialization::AudioSpecificConfig(config) => (**config).clone(),
                _ => return Err(UsacDecodeError::UnsupportedConfiguration),
            };
            asc.sampling_frequency_index = usac.sampling_frequency_index;
            asc.sampling_frequency = usac.sampling_frequency;
            asc.channel_configuration = usac.channel_configuration_index;
            asc.usac_config = Some(usac);
            let mut replacement = Self::from_audio_specific_config(&asc)
                .map_err(|_| UsacDecodeError::UnsupportedConfiguration)?;
            replacement.usac_crossfade_source = self.usac_crossfade_source.take();
            replacement.usac_last_output = self.usac_last_output.take();
            replacement.usac_preroll_depth = self.usac_preroll_depth;
            *self = replacement;
        }
        self.usac_preroll_depth += 1;
        for access_unit in preroll.access_units {
            if let Err(error) = self.decode_usac_access_unit_multichannel_f32(&access_unit) {
                self.usac_preroll_depth -= 1;
                return Err(error);
            }
        }
        self.usac_preroll_depth -= 1;
        Ok(())
    }

    fn finish_usac_output(&mut self, channels: &mut [Vec<f32>]) {
        if self.usac_preroll_depth != 0 {
            return;
        }
        if let Some(source) = self.usac_crossfade_source.take() {
            for (output, old) in channels.iter_mut().zip(source) {
                for (index, (sample, previous)) in output.iter_mut().zip(old).take(128).enumerate()
                {
                    let alpha = index as f32 / 128.0;
                    *sample = previous * (1.0 - alpha) + *sample * alpha;
                }
            }
        }
        self.usac_last_output = Some(channels.to_vec());
    }

    pub fn decode_usac_mps212_access_unit(
        &mut self,
        input: &[u8],
    ) -> Result<UsacMps212AccessUnit, UsacDecodeError> {
        if self.usac_mps212_decoder.is_none() {
            return Err(UsacDecodeError::UnsupportedConfiguration);
        }
        let mut reader = BitReader::new(input);
        let independent = reader.read_bool()?;
        self.parse_usac_extension_elements(&mut reader, 0, self.usac_leading_extension_count)?;
        self.decode_pending_usac_audio_preroll()?;
        let mut access_unit = self
            .usac_mps212_decoder
            .as_mut()
            .expect("checked above")
            .decode_after_independent(&mut reader, independent)?;
        self.parse_usac_extension_elements(
            &mut reader,
            self.usac_leading_extension_count,
            self.usac_extension_elements.len(),
        )?;
        access_unit.bits_read = reader.bits_read();
        Ok(access_unit)
    }

    pub fn frame_length(&self) -> usize {
        self.frame_length
    }

    pub fn sampling_frequency_index(&self) -> u8 {
        self.sampling_frequency_index
    }

    pub fn channel_configuration(&self) -> u8 {
        self.channel_configuration
    }

    /// Return the decoder's current stream configuration and output shape.
    ///
    /// Unlike libAACdec's borrowed pointer, this is an owned snapshot and
    /// remains valid after the decoder is used again.
    pub fn stream_info(&self) -> DecoderStreamInfo {
        let aac_sample_rate = sample_rate_from_index(self.sampling_frequency_index)
            .expect("decoder construction validates the sampling-frequency index");
        let aac_num_channels = expected_channels_for_config(self.channel_configuration)
            .unwrap_or_else(|| self.configured_usac_channels());

        let extension_sampling_rate = if let Some(rate) = self.ordinary_sbr_output_frequency {
            Some(rate)
        } else if self.eld_sbr_dual_rate {
            Some(aac_sample_rate.saturating_mul(2))
        } else if !self.ld_sbr_processors.is_empty() {
            Some(aac_sample_rate)
        } else {
            None
        };
        let sample_rate = extension_sampling_rate.unwrap_or(aac_sample_rate);
        let frame_size = self
            .frame_length
            .saturating_mul(usize::try_from(sample_rate / aac_sample_rate).unwrap_or(1));
        let ps_rendered = self.ps_signaled && !self.qmf_low_power;
        let num_channels = if ps_rendered && aac_num_channels == 1 {
            2
        } else {
            aac_num_channels
        };
        let channel_labels = if ps_rendered && aac_num_channels == 1 {
            vec![ChannelLabel::FrontLeft, ChannelLabel::FrontRight]
        } else {
            channel_labels_for_config(self.channel_configuration)
                .map(<[ChannelLabel]>::to_vec)
                .unwrap_or_else(|| unknown_channel_labels(num_channels))
        };
        let channel_indices = channel_indices_for_labels(&channel_labels);

        let mut flags = 0;
        match self.audio_object_type {
            17 | 20 => flags |= STREAM_FLAG_ER,
            23 => flags |= STREAM_FLAG_ER | STREAM_FLAG_LD,
            39 => flags |= STREAM_FLAG_ER | STREAM_FLAG_ELD,
            42 => flags |= STREAM_FLAG_USAC,
            _ => {}
        }
        if self.er_resilience_flags[0] {
            flags |= STREAM_FLAG_ER_VCB11;
        }
        if self.er_resilience_flags[1] {
            flags |= STREAM_FLAG_ER_RVLC;
        }
        if self.er_resilience_flags[2] {
            flags |= STREAM_FLAG_ER_HCR;
        }
        if matches!(self.extension_audio_object_type, Some(5 | 29))
            || !self.ld_sbr_processors.is_empty()
        {
            flags |= STREAM_FLAG_SBR_PRESENT;
        }
        if self.eld_sbr_crc {
            flags |= STREAM_FLAG_SBR_CRC;
        }
        if self.ps_signaled {
            flags |= STREAM_FLAG_PS_PRESENT;
        }
        if self.usac_mps212_decoder.is_some() {
            flags |= STREAM_FLAG_MPS_PRESENT;
        }
        if self.drc_config.is_some()
            || self.legacy_drc_payload.is_some()
            || self.legacy_dvb_drc_payload.is_some()
        {
            flags |= STREAM_FLAG_DRC_PRESENT;
        }

        DecoderStreamInfo {
            sample_rate,
            frame_size,
            num_channels,
            channel_labels,
            channel_indices,
            aac_sample_rate,
            profile: -1,
            audio_object_type: self.audio_object_type,
            channel_configuration: self.channel_configuration,
            bit_rate: 0,
            aac_samples_per_frame: self.frame_length,
            aac_num_channels,
            extension_audio_object_type: self.extension_audio_object_type,
            extension_sampling_rate,
            output_delay: 0,
            flags,
            error_protection_config: self.error_protection_config.map_or(-1, |value| value as i8),
            num_lost_access_units: 0,
            num_total_bytes: 0,
            num_bad_bytes: 0,
            num_total_access_units: 0,
            num_bad_access_units: 0,
            drc_program_reference_level: self
                .legacy_drc_payload
                .as_ref()
                .and_then(|payload| payload.program_reference_level)
                .map_or(-1, |value| value as i8),
            drc_presentation_mode: self.legacy_drc_presentation_mode,
            output_loudness: self.reported_output_loudness(),
        }
    }

    /// Select the real-valued low-power or complex QMF path for every SBR
    /// channel. Parametric stereo needs complex QMF samples and is therefore
    /// intentionally not rendered while low-power mode is active, matching
    /// libAACdec's QMF-mode synchronization.
    pub fn set_qmf_low_power(&mut self, enabled: bool) {
        self.qmf_low_power = enabled;
        for processor in self
            .ld_sbr_processors
            .iter_mut()
            .chain(&mut self.ld_sbr_fixed_processors)
            .chain(&mut self.ordinary_sbr_processors)
            .chain(&mut self.ordinary_sbr_fixed_processors)
        {
            processor.set_low_power(enabled);
        }
    }

    pub fn qmf_low_power(&self) -> bool {
        self.qmf_low_power
    }

    pub fn automatic_qmf_low_power(&self) -> bool {
        let channels = expected_channels_for_config(self.channel_configuration)
            .unwrap_or_else(|| self.configured_usac_channels());
        let ps_capable_mono = channels == 1 && matches!(self.audio_object_type, 2 | 5 | 22 | 29);
        self.audio_object_type != 42
            && self.usac_mps212_decoder.is_none()
            && !ps_capable_mono
            && (!self.ld_sbr_processors.is_empty() || !self.ordinary_sbr_processors.is_empty())
    }

    fn configured_usac_channels(&self) -> usize {
        if self.usac_decoder.is_some() {
            1
        } else if self.usac_stereo_decoder.is_some() || self.usac_mps212_decoder.is_some() {
            2
        } else if let Some(decoder) = &self.usac_multichannel_decoder {
            decoder.channels()
        } else {
            0
        }
    }

    pub fn fixed_concealment_state(&self) -> ConcealmentState {
        self.fixed_concealment_state
    }

    pub fn f32_concealment_state(&self) -> ConcealmentState {
        self.f32_concealment_state
    }

    /// Clear synthesis, prediction, concealment, SBR and PS histories while
    /// retaining the active stream configuration and user-facing settings.
    pub fn clear_history(&mut self) -> Result<(), DecodeError> {
        if self.audio_object_type == 42 {
            let ancillary_capacity = self.ancillary_data_capacity;
            let drc_config = self.drc_config.clone();
            let drc_loudness_info = self.drc_loudness_info.clone();
            let drc_selection = self.drc_selection;
            let legacy_drc_parameters = self.legacy_drc_parameters;
            let legacy_drc_presentation_mode = self.legacy_drc_presentation_mode;
            let legacy_drc_expiry_frames = self.legacy_drc_expiry_frames;
            let qmf_low_power = self.qmf_low_power;
            let mut replacement = self.rebuild_from_initialization()?;
            replacement.ancillary_data_capacity = ancillary_capacity;
            replacement.drc_config = drc_config;
            replacement.drc_gain = None;
            replacement.drc_loudness_info = drc_loudness_info;
            replacement.drc_selection = drc_selection;
            replacement.legacy_drc_parameters = legacy_drc_parameters;
            replacement.legacy_drc_presentation_mode = legacy_drc_presentation_mode;
            replacement.legacy_drc_expiry_frames = legacy_drc_expiry_frames;
            replacement.set_qmf_low_power(qmf_low_power);
            *self = replacement;
            return Ok(());
        }

        for filterbank in &mut self.channel_filterbanks {
            filterbank.clear_history();
        }
        for filterbank in &mut self.fixed_channel_filterbanks {
            filterbank.clear_history();
        }
        for filterbank in &mut self.coupling_filterbanks {
            filterbank.clear_history();
        }
        for filterbank in &mut self.fixed_coupling_filterbanks {
            filterbank.clear_history();
        }
        for filterbank in &mut self.eld_channel_filterbanks {
            filterbank.clear_history();
        }
        for filterbank in &mut self.eld_fixed_channel_filterbanks {
            filterbank.clear_history();
        }
        for parser in &mut self.ld_sbr_parsers {
            parser.clear_history();
        }
        for processor in self
            .ld_sbr_processors
            .iter_mut()
            .chain(&mut self.ld_sbr_fixed_processors)
            .chain(&mut self.ordinary_sbr_processors)
            .chain(&mut self.ordinary_sbr_fixed_processors)
        {
            processor.clear_history();
        }
        for parser in self
            .ordinary_sbr_parsers
            .iter_mut()
            .chain(&mut self.ordinary_sbr_fixed_parsers)
            .flatten()
        {
            match parser {
                OrdinarySbrParser::Mono(parser) => parser.clear_history(),
                OrdinarySbrParser::Stereo(parser) => parser.clear_history(),
            }
        }
        for parser in self.ps_parsers.iter_mut().chain(&mut self.ps_fixed_parsers) {
            parser.clear_history();
        }
        for processor in self
            .ps_processors
            .iter_mut()
            .chain(&mut self.ps_fixed_processors)
        {
            processor.clear_history();
        }

        self.last_ld_sbr_frames.clear();
        self.last_ordinary_sbr_frames.clear();
        self.last_ordinary_sbr_fixed_frames.clear();
        self.last_ps_frames.fill(None);
        self.last_ps_fixed_frames.fill(None);
        self.legacy_qmf_drc_states.clear();
        self.legacy_qmf_drc_fixed_states.clear();
        self.legacy_drc_window_sequences.clear();
        self.fixed_concealment_spectra.clear();
        self.fixed_concealment_losses = 0;
        self.fixed_concealment_phase = 0;
        self.fixed_concealment_state = ConcealmentState::Ok;
        self.fixed_concealment_fade_in_remaining = 0;
        self.f32_concealment_spectra.clear();
        self.f32_concealment_losses = 0;
        self.f32_concealment_phase = 0;
        self.f32_concealment_state = ConcealmentState::Ok;
        self.f32_concealment_fade_in_remaining = 0;
        self.pns_random = PnsRandomState::new(0x1f2e_3d4c);
        self.clear_adts_crc_regions();
        self.ancillary_data.clear();
        self.drc_gain = None;
        Ok(())
    }

    /// Signal a discontinuity. Pure Rust resets all predictive and synthesis
    /// histories so that no samples or delta state cross the interruption.
    pub fn signal_interruption(&mut self) -> Result<(), DecodeError> {
        self.clear_history()
    }

    /// Drain one frame of filterbank delay without consuming an access unit.
    pub fn flush_interleaved_f32(&mut self) -> Result<Vec<f32>, DecodeError> {
        let channels = expected_channels_for_config(self.channel_configuration)
            .unwrap_or_else(|| self.f32_concealment_spectra.len().max(1));
        self.ensure_channel_filterbanks(channels)?;
        let mut output: Vec<Vec<f32>> = Vec::with_capacity(channels);
        for channel in 0..channels {
            output.push(if self.audio_object_type == 39 {
                self.eld_channel_filterbanks[channel].flush()?
            } else {
                self.channel_filterbanks[channel].flush()
            });
        }
        let processors = if !self.ld_sbr_processors.is_empty() {
            &mut self.ld_sbr_processors
        } else {
            &mut self.ordinary_sbr_processors
        };
        if processors.len() >= output.len() {
            for (channel, samples) in output.iter_mut().enumerate() {
                *samples = processors[channel]
                    .flush(self.frame_length)?
                    .into_iter()
                    .map(|sample| sample as f32)
                    .collect();
            }
        }
        if self.ps_signaled && output.len() == 1 {
            output.push(output[0].clone());
        }
        Ok(interleave_multichannel_f32(&output))
    }

    /// Fixed-point counterpart of [`Self::flush_interleaved_f32`].
    pub fn flush_interleaved_i16(&mut self) -> Result<Vec<i16>, DecodeError> {
        let channels = expected_channels_for_config(self.channel_configuration)
            .unwrap_or_else(|| self.fixed_concealment_spectra.len().max(1));
        self.ensure_fixed_channel_filterbanks(channels)?;
        let mut output: Vec<Vec<i16>> = Vec::with_capacity(channels);
        for channel in 0..channels {
            output.push(if self.audio_object_type == 39 {
                self.eld_fixed_channel_filterbanks[channel]
                    .flush()?
                    .into_iter()
                    .map(dbl_to_pcm16)
                    .collect()
            } else {
                self.fixed_channel_filterbanks[channel]
                    .flush()
                    .into_iter()
                    .map(dbl_to_pcm16)
                    .collect()
            });
        }
        let processors = if !self.ld_sbr_fixed_processors.is_empty() {
            &mut self.ld_sbr_fixed_processors
        } else {
            &mut self.ordinary_sbr_fixed_processors
        };
        if processors.len() >= output.len() {
            for (channel, samples) in output.iter_mut().enumerate() {
                *samples = processors[channel]
                    .flush(self.frame_length)?
                    .into_iter()
                    .map(|sample| f32_to_i16(sample as f32))
                    .collect();
            }
        }
        if self.ps_signaled && output.len() == 1 {
            output.push(output[0].clone());
        }
        Ok(interleave_multichannel_i16_samples(&output))
    }

    fn rebuild_from_initialization(&self) -> Result<Self, DecodeError> {
        match &self.initialization {
            DecoderInitialization::General {
                audio_object_type,
                sampling_frequency_index,
                channel_configuration,
                frame_length,
            } => Self::new_ga_with_frame_length(
                *audio_object_type,
                *sampling_frequency_index,
                *channel_configuration,
                *frame_length,
            ),
            DecoderInitialization::AudioSpecificConfig(config) => {
                Self::from_audio_specific_config(config)
            }
            DecoderInitialization::Drm {
                sampling_frequency_index,
                channel_configuration,
            } => Self::new_drm_aac(*sampling_frequency_index, *channel_configuration),
        }
    }

    pub fn configure_drc(&mut self, config: UniDrcConfig, request: DrcSelectionRequest) {
        self.drc_config = Some(config);
        self.drc_selection = request;
    }

    pub fn update_drc_gain(&mut self, gain: UniDrcGain) {
        self.drc_gain = Some(gain);
    }

    /// Update MPEG-D loudness metadata associated with the active Unified DRC
    /// configuration. This is the information reported through CStreamInfo's
    /// `outputLoudness` field after instruction selection.
    pub fn update_drc_loudness_info(&mut self, loudness: LoudnessInfoSet) {
        self.drc_loudness_info = Some(loudness);
    }

    pub fn clear_drc_loudness_info(&mut self) {
        self.drc_loudness_info = None;
    }

    pub fn set_drc_boost_factor(&mut self, value: u8) {
        self.drc_selection.boost_scale = value as f32 / 127.0;
        self.legacy_drc_parameters.boost_scale = value as f32 / 127.0;
    }

    pub fn set_drc_attenuation_factor(&mut self, value: u8) {
        self.drc_selection.attenuation_scale = value as f32 / 127.0;
        self.legacy_drc_parameters.attenuation_scale = value as f32 / 127.0;
    }

    pub fn set_drc_reference_level(&mut self, value: Option<u8>) {
        self.drc_selection.target_loudness = value.map(|value| -(value as f32) * 0.25);
        self.legacy_drc_parameters.target_reference_level = value;
    }

    pub fn set_drc_heavy_compression(&mut self, enabled: bool) {
        self.legacy_drc_parameters.heavy_compression = enabled;
    }

    pub fn set_drc_default_presentation_mode(&mut self, mode: i8) {
        self.legacy_drc_parameters.default_presentation_mode = mode;
    }

    pub fn set_drc_encoder_target_level(&mut self, level: u8) {
        self.legacy_drc_parameters.encoder_target_level = level;
    }

    pub fn legacy_drc_parameters(&self) -> LegacyDrcParameters {
        self.legacy_drc_parameters
    }

    pub fn set_legacy_drc_output_channels(&mut self, channels: usize) {
        self.legacy_drc_output_channels = channels;
    }

    fn effective_legacy_parameters(&self) -> (f32, bool) {
        let presentation_mode = self
            .legacy_dvb_drc_payload
            .map(|payload| payload.presentation_mode as i8)
            .filter(|mode| matches!(mode, 1 | 2))
            .unwrap_or(self.legacy_drc_parameters.default_presentation_mode);
        let source_channels = expected_channels_for_config(self.channel_configuration)
            .unwrap_or(self.legacy_drc_output_channels.max(1));
        let output_channels = self.legacy_drc_output_channels;
        let is_downmix = output_channels > 0 && source_channels > output_channels;
        let is_mono_downmix = is_downmix && output_channels == 1;
        let is_stereo_downmix = is_downmix && output_channels == 2;
        let mut attenuation = self.legacy_drc_parameters.attenuation_scale;
        let mut heavy = self.legacy_drc_parameters.heavy_compression;
        match presentation_mode {
            0 => {
                let downmix_headroom = if is_downmix {
                    (-80.0 * (source_channels as f32 / output_channels as f32).log10()).floor()
                        as i32
                } else {
                    0
                };
                let program_level = self
                    .legacy_drc_payload
                    .as_ref()
                    .and_then(|payload| payload.program_reference_level)
                    .or(self.legacy_drc_parameters.target_reference_level)
                    .map(i32::from)
                    .unwrap_or(0);
                let headroom = self
                    .legacy_drc_parameters
                    .target_reference_level
                    .map(|target| i32::from(target) + downmix_headroom - program_level)
                    .unwrap_or(downmix_headroom);
                if headroom < 0 {
                    let encoder_headroom =
                        (i32::from(self.legacy_drc_parameters.encoder_target_level)
                            - program_level)
                            .min(0);
                    if encoder_headroom < headroom {
                        let required = (-headroom) as f32 / (-encoder_headroom) as f32;
                        attenuation = attenuation.max((required * 127.0).round() / 127.0);
                    } else {
                        attenuation = 1.0;
                        if headroom - encoder_headroom <= -40 {
                            heavy = true;
                        }
                    }
                }
            }
            1 => {
                if self
                    .legacy_drc_parameters
                    .target_reference_level
                    .is_some_and(|level| level < 124)
                {
                    heavy = true;
                } else if is_mono_downmix || is_stereo_downmix {
                    attenuation = 1.0;
                }
            }
            2 => {
                heavy = false;
                if self
                    .legacy_drc_parameters
                    .target_reference_level
                    .is_some_and(|level| level < 124)
                {
                    if is_mono_downmix {
                        heavy = true;
                    } else {
                        attenuation = 1.0;
                    }
                } else if is_mono_downmix || is_stereo_downmix {
                    attenuation = 1.0;
                }
            }
            _ => {}
        }
        if heavy {
            attenuation = 1.0;
        }
        (attenuation.clamp(0.0, 1.0), heavy)
    }

    fn legacy_control_bands(&self, channel: usize) -> Option<(Vec<u8>, Vec<f32>)> {
        self.legacy_drc_parameters.target_reference_level?;
        let (attenuation_scale, heavy_compression) = self.effective_legacy_parameters();
        if heavy_compression {
            if let Some(payload) = self.legacy_dvb_drc_payload {
                let top = self.frame_length.div_ceil(4).saturating_sub(1).min(255) as u8;
                return Some((vec![top], vec![payload.gain()]));
            }
        }
        let payload = self.legacy_drc_payload.as_ref()?;
        let gains = payload.control_gains(
            channel,
            attenuation_scale,
            self.legacy_drc_parameters.boost_scale,
        )?;
        Some((payload.band_top.clone(), gains))
    }

    fn legacy_one_band_control_gain(&self, channel: usize) -> Option<f32> {
        let (bands, gains) = self.legacy_control_bands(channel)?;
        (bands.len() == 1 && gains.len() == 1).then_some(gains[0])
    }

    fn legacy_qmf_drc_frame(&self, channel: usize) -> Option<LegacyQmfDrcFrame> {
        let (band_top, gains) = self.legacy_control_bands(channel)?;
        let interpolation_scheme =
            if self.effective_legacy_parameters().1 && self.legacy_dvb_drc_payload.is_some() {
                0
            } else {
                self.legacy_drc_payload
                    .as_ref()
                    .map_or(0, |payload| payload.interpolation_scheme)
            };
        Some(LegacyQmfDrcFrame {
            band_top,
            gains: gains.into_iter().map(f64::from).collect(),
            interpolation_scheme,
            window_sequence: self
                .legacy_drc_window_sequences
                .get(channel)
                .copied()
                .unwrap_or(WindowSequence::OnlyLong),
        })
    }

    fn apply_legacy_drc_to_qmf_slots(
        &mut self,
        processor_channel: usize,
        channel: usize,
        slots: &mut [QmfSlot],
        fixed: bool,
    ) {
        let next = self.legacy_qmf_drc_frame(channel);
        let states = if fixed {
            &mut self.legacy_qmf_drc_fixed_states
        } else {
            &mut self.legacy_qmf_drc_states
        };
        states.resize_with(processor_channel + 1, LegacyQmfDrcState::default);
        self.legacy_drc_control_applied |= apply_legacy_qmf_drc(
            &mut states[processor_channel],
            slots,
            next,
            self.frame_length,
        );
    }

    fn current_legacy_normalization_gain(&self) -> f32 {
        self.legacy_drc_payload.as_ref().map_or(1.0, |payload| {
            payload.normalization_gain(self.legacy_drc_parameters.target_reference_level)
        })
    }

    pub fn set_metadata_expiry_ms(&mut self, milliseconds: u32) {
        let sample_rate = sample_rate_from_index(self.sampling_frequency_index).unwrap_or(0);
        self.legacy_drc_expiry_frames = if milliseconds == 0 || sample_rate == 0 {
            0
        } else {
            (milliseconds as usize * sample_rate as usize)
                .div_ceil(self.frame_length.saturating_mul(1000))
        };
    }

    fn age_legacy_drc(&mut self) {
        if (self.legacy_drc_payload.is_none()
            && self.legacy_dvb_drc_payload.is_none()
            && self.legacy_dvb_downmix_metadata.is_none()
            && self.legacy_matrix_mixdown.is_none())
            || self.legacy_drc_expiry_frames == 0
        {
            return;
        }
        self.legacy_drc_age_frames = self.legacy_drc_age_frames.saturating_add(1);
        if self.legacy_drc_age_frames > self.legacy_drc_expiry_frames {
            self.legacy_drc_payload = None;
            self.legacy_dvb_drc_payload = None;
            self.legacy_dvb_downmix_metadata = None;
            self.legacy_matrix_mixdown = None;
            self.legacy_drc_age_frames = 0;
        }
    }

    fn read_mpeg4_drc_fill(&mut self, reader: &mut BitReader<'_>) -> Result<(), DecodeError> {
        self.legacy_drc_payload = parse_mpeg4_drc_fill_element(reader)?;
        self.legacy_drc_age_frames = 0;
        Ok(())
    }

    pub fn set_uni_drc_effect(&mut self, value: i8) {
        self.drc_selection.enabled = value >= 0;
        self.drc_selection.preferred_effect_mask = if value > 0 {
            1u16 << (value as u32 - 1)
        } else {
            0
        };
    }

    pub fn set_uni_drc_album_mode(&mut self, album_mode: bool) {
        self.drc_selection.album_mode = album_mode;
    }

    pub fn drc_selection_request(&self) -> DrcSelectionRequest {
        self.drc_selection
    }

    pub fn disable_drc(&mut self) {
        self.drc_config = None;
        self.drc_gain = None;
        self.drc_loudness_info = None;
    }

    fn reported_output_loudness(&self) -> i32 {
        if let (Some(config), Some(_gain)) = (&self.drc_config, &self.drc_gain) {
            if self.drc_selection.enabled {
                let loudness =
                    config
                        .select_instruction(self.drc_selection)
                        .and_then(|instruction| {
                            self.drc_loudness_info.as_ref().and_then(|set| {
                                set.select_program_loudness(
                                    instruction.drc_set_id,
                                    self.drc_selection.downmix_id,
                                    self.drc_selection.album_mode,
                                )
                            })
                        });
                return loudness.map_or(-1, |measured| {
                    let output = self.drc_selection.target_loudness.unwrap_or(measured);
                    (-output * 4.0).round().clamp(0.0, 231.0) as i32
                });
            }
        }

        let Some(program_level) = self
            .legacy_drc_payload
            .as_ref()
            .and_then(|payload| payload.program_reference_level)
        else {
            return -1;
        };
        self.legacy_drc_parameters
            .target_reference_level
            .map_or(i32::from(program_level), i32::from)
    }

    fn apply_legacy_drc_to_f32_spectrum(
        &mut self,
        spectrum: &mut InverseQuantizedSpectrum,
        channel: usize,
    ) {
        let Some((band_top, gains)) = self.legacy_control_bands(channel) else {
            return;
        };
        apply_legacy_band_gains_f32(spectrum, &band_top, &gains);
        self.legacy_drc_control_applied = true;
    }

    fn apply_legacy_drc_to_fixed_spectrum(
        &mut self,
        spectrum: &mut FixedInverseQuantizedSpectrum,
        channel: usize,
    ) {
        let Some((band_top, gains)) = self.legacy_control_bands(channel) else {
            return;
        };
        apply_legacy_band_gains_fixed(spectrum, &band_top, &gains);
        self.legacy_drc_control_applied = true;
    }

    fn apply_configured_drc_f32(&mut self, channels: &mut [Vec<f32>]) -> Result<(), DecodeError> {
        self.apply_legacy_drc_f32(channels);
        let (Some(config), Some(gain)) = (&self.drc_config, &self.drc_gain) else {
            return Ok(());
        };
        let Some(instruction) = config.select_instruction(self.drc_selection) else {
            return Ok(());
        };
        if channels.is_empty() {
            return Ok(());
        }
        let mut interleaved = interleave_multichannel_f32(channels);
        config.apply_instruction_f32_scaled(
            instruction,
            gain,
            &mut interleaved,
            channels.len(),
            self.drc_selection.attenuation_scale,
            self.drc_selection.boost_scale,
        )?;
        for (frame, samples) in interleaved.chunks_exact(channels.len()).enumerate() {
            for (channel, sample) in samples.iter().enumerate() {
                channels[channel][frame] = *sample;
            }
        }
        Ok(())
    }

    fn apply_configured_drc_i16(&mut self, channels: &mut [Vec<i16>]) -> Result<(), DecodeError> {
        self.apply_legacy_drc_i16(channels);
        let (Some(config), Some(gain)) = (&self.drc_config, &self.drc_gain) else {
            return Ok(());
        };
        let Some(instruction) = config.select_instruction(self.drc_selection) else {
            return Ok(());
        };
        if channels.is_empty() {
            return Ok(());
        }
        let mut interleaved = interleave_multichannel_i16_samples(channels);
        config.apply_instruction_i16_scaled(
            instruction,
            gain,
            &mut interleaved,
            channels.len(),
            self.drc_selection.attenuation_scale,
            self.drc_selection.boost_scale,
        )?;
        for (frame, samples) in interleaved.chunks_exact(channels.len()).enumerate() {
            for (channel, sample) in samples.iter().enumerate() {
                channels[channel][frame] = *sample;
            }
        }
        Ok(())
    }

    fn apply_legacy_drc_f32(&mut self, channels: &mut [Vec<f32>]) {
        if self.legacy_drc_payload.is_none() && self.legacy_dvb_drc_payload.is_none() {
            return;
        }
        let apply_control = !self.legacy_drc_control_applied;
        let channel_gains: Vec<_> = (0..channels.len())
            .map(|index| {
                if apply_control {
                    self.legacy_one_band_control_gain(index).unwrap_or(1.0)
                } else {
                    1.0
                }
            })
            .collect();
        let current_normalization = self.current_legacy_normalization_gain();
        let sample_count = channels.first().map_or(0, Vec::len);
        for frame in 0..sample_count {
            let normalization = self.next_legacy_normalization_gain();
            for (channel, &control_gain) in channels.iter_mut().zip(&channel_gains) {
                channel[frame] *= control_gain * normalization;
            }
        }
        self.legacy_norm_gain_previous = current_normalization;
    }

    fn apply_legacy_drc_i16(&mut self, channels: &mut [Vec<i16>]) {
        if self.legacy_drc_payload.is_none() && self.legacy_dvb_drc_payload.is_none() {
            return;
        }
        let apply_control = !self.legacy_drc_control_applied;
        let channel_gains: Vec<_> = (0..channels.len())
            .map(|index| {
                if apply_control {
                    self.legacy_one_band_control_gain(index).unwrap_or(1.0)
                } else {
                    1.0
                }
            })
            .collect();
        let current_normalization = self.current_legacy_normalization_gain();
        let sample_count = channels.first().map_or(0, Vec::len);
        for frame in 0..sample_count {
            let normalization = self.next_legacy_normalization_gain();
            for (channel, &control_gain) in channels.iter_mut().zip(&channel_gains) {
                channel[frame] = (f32::from(channel[frame]) * control_gain * normalization)
                    .round()
                    .clamp(f32::from(i16::MIN), f32::from(i16::MAX))
                    as i16;
            }
        }
        self.legacy_norm_gain_previous = current_normalization;
    }

    fn next_legacy_normalization_gain(&mut self) -> f32 {
        let input = self.legacy_norm_gain_previous;
        if input == 1.0
            && self.legacy_norm_filter_state == 1.0
            && self.legacy_norm_filter_input_previous == 1.0
        {
            return 1.0;
        }
        let output = 0.96907 * self.legacy_norm_filter_state
            + 0.015466 * input
            + 0.015466 * self.legacy_norm_filter_input_previous;
        self.legacy_norm_filter_input_previous = input;
        self.legacy_norm_filter_state = output;
        output
    }

    pub fn fixed_concealment_spectral_frame(&self) -> Option<FixedConcealmentSpectralFrame> {
        (!self.fixed_concealment_spectra.is_empty()).then(|| FixedConcealmentSpectralFrame {
            channels: self
                .fixed_concealment_spectra
                .iter()
                .map(|(spectrum, ics)| FixedConcealmentChannel {
                    spectrum: spectrum.clone(),
                    ics: ics.clone(),
                })
                .collect(),
        })
    }

    pub fn f32_concealment_spectral_frame(&self) -> Option<F32ConcealmentSpectralFrame> {
        (!self.f32_concealment_spectra.is_empty()).then(|| F32ConcealmentSpectralFrame {
            channels: self
                .f32_concealment_spectra
                .iter()
                .map(|(spectrum, ics)| F32ConcealmentChannel {
                    spectrum: spectrum.clone(),
                    ics: ics.clone(),
                })
                .collect(),
        })
    }

    pub fn decode_raw_data_block_f32(
        &mut self,
        input: &[u8],
    ) -> Result<DecodedAacLcFrame, DecodeError> {
        let mut reader = BitReader::new(input);
        let frame = self.decode_raw_data_block_f32_from_reader(&mut reader)?;
        self.validate_frame_channel_configuration(&frame)?;
        Ok(frame)
    }

    pub fn decode_raw_data_block_f32_strict(
        &mut self,
        input: &[u8],
    ) -> Result<DecodedAacLcFrame, DecodeError> {
        let mut reader = BitReader::new(input);
        let frame = self.decode_raw_data_block_f32_from_reader(&mut reader)?;
        self.validate_frame_channel_configuration(&frame)?;
        validate_zero_trailing_bits(&reader)?;
        Ok(frame)
    }

    pub fn decode_raw_data_block_f32_from_reader(
        &mut self,
        reader: &mut BitReader<'_>,
    ) -> Result<DecodedAacLcFrame, DecodeError> {
        self.age_legacy_drc();
        self.legacy_drc_control_applied = false;
        self.ancillary_data.clear();
        while !matches!(self.audio_object_type, 17 | 20 | 23 | 39) && reader.remaining_bits() >= 3 {
            let element_id = ElementId::from_bits(reader.read_u8(3)?);
            match element_id {
                ElementId::SingleChannel | ElementId::Lfe | ElementId::ChannelPair => {
                    reader.push_back(3)?;
                    break;
                }
                ElementId::CouplingChannel => {
                    reader.push_back(3)?;
                    let prefix = CouplingChannelElementPrefix::parse_aac_lc_from_reader(reader)?;
                    return Err(DecodeError::UnsupportedCouplingChannelElement(prefix));
                }
                ElementId::DataStream => self.read_data_stream_element(reader)?,
                ElementId::ProgramConfig => {
                    let _ = ProgramConfig::parse_from_reader(reader)?;
                }
                ElementId::Fill => {
                    if fill_extension_type(reader)? == Some(0x0b) {
                        self.read_mpeg4_drc_fill(reader)?;
                    } else {
                        skip_fill_element(reader)?;
                    }
                }
                ElementId::End => return Err(DecodeError::NoAudioElement),
            }
        }
        let frame = self.decode_raw_data_block_multichannel_f32_inner(reader)?;
        match frame.channels.as_slice() {
            [mono] => Ok(DecodedAacLcFrame::Mono(DecodedSingleChannelFrame {
                side_info: synthetic_single_channel_side_info(),
                section_data: SectionData {
                    sections: Vec::new(),
                    codebooks: vec![Vec::new()],
                    bits_read: 0,
                },
                scalefactors: ScalefactorData {
                    values: vec![Vec::new()],
                },
                pulse_data: PulseData::absent(),
                tns_data: TnsData::absent(1),
                spectral: SpectralData {
                    windows: Vec::new(),
                },
                spectrum: InverseQuantizedSpectrum {
                    windows: Vec::new(),
                },
                samples: mono.clone(),
                bits_read: reader.bits_read(),
            })),
            [left, right] => Ok(DecodedAacLcFrame::Stereo(DecodedChannelPairFrame {
                spectra: synthetic_channel_pair_spectra(),
                left_samples: left.clone(),
                right_samples: right.clone(),
            })),
            channels => Err(DecodeError::UnsupportedChannelConfiguration(
                channels.len() as u8
            )),
        }
    }

    /// Decode one raw_data_block and consume its required `ID_END` terminator.
    ///
    /// Unlike the compatibility-oriented reader API, this is suitable for a
    /// transport parser that must locate the following raw data block.
    pub fn decode_raw_data_block_f32_terminated_from_reader(
        &mut self,
        reader: &mut BitReader<'_>,
    ) -> Result<DecodedAacLcFrame, DecodeError> {
        let frame = self.decode_raw_data_block_f32_from_reader(reader)?;
        consume_raw_data_block_terminator(reader)?;
        Ok(frame)
    }

    pub fn decode_adts_frame_f32(
        &mut self,
        input: &[u8],
    ) -> Result<DecodedAacLcFrame, DecodeError> {
        let frame = AdtsFrame::parse(input)?;
        if frame.header.profile + 1 != self.audio_object_type {
            return Err(DecodeError::UnsupportedAudioObjectType(
                frame.header.profile + 1,
            ));
        }
        if frame.header.number_of_raw_data_blocks_in_frame != 0 {
            return Err(DecodeError::UnsupportedRawBlocksInAdtsFrame(
                frame.header.number_of_raw_data_blocks_in_frame,
            ));
        }
        if frame.header.sampling_frequency_index != self.sampling_frequency_index {
            return Err(DecodeError::AdtsConfigChanged);
        }
        let decoded = self.decode_raw_data_block_f32(frame.payload)?;
        self.validate_frame_channel_configuration(&decoded)?;
        if !frame.header.protection_absent {
            self.validate_adts_syntax_crc(
                frame,
                frame.payload,
                frame
                    .header
                    .crc_check
                    .ok_or(AdtsError::SyntaxRegionsRequiredForCrc)?,
                true,
            )?;
        }
        Ok(decoded)
    }

    /// Decode every raw_data_block carried by one ADTS frame.
    ///
    /// Multi-block ADTS requires CRC-protected block-position fields; CRC-less
    /// multi-block input is rejected by [`AdtsFrame::raw_data_blocks`].
    pub fn decode_adts_frame_blocks_f32(
        &mut self,
        input: &[u8],
    ) -> Result<Vec<DecodedAacLcFrame>, DecodeError> {
        let frame = AdtsFrame::parse(input)?;
        validate_adts_aac_lc_configuration(self, frame.header)?;
        if frame.header.protection_absent && frame.header.number_of_raw_data_blocks_in_frame != 0 {
            let mut reader = BitReader::new(frame.payload);
            let frames = (0..=frame.header.number_of_raw_data_blocks_in_frame)
                .map(|_| self.decode_raw_data_block_f32_terminated_from_reader(&mut reader))
                .collect::<Result<Vec<_>, _>>()?;
            if !reader.remaining_bits_are_zero() {
                return Err(DecodeError::NonZeroTrailingBits(reader.remaining_bits()));
            }
            return Ok(frames);
        }
        if !frame.header.protection_absent && frame.header.number_of_raw_data_blocks_in_frame != 0 {
            frame.validate_multi_block_header_crc()?;
        }
        frame
            .raw_data_blocks()?
            .into_iter()
            .map(|block| {
                let decoded = self.decode_raw_data_block_f32(block.payload)?;
                self.validate_frame_channel_configuration(&decoded)?;
                if let Some(expected) = block.crc_check {
                    self.validate_adts_syntax_crc(frame, block.payload, expected, false)?;
                }
                Ok(decoded)
            })
            .collect()
    }

    /// Fixed-point counterpart of [`Self::decode_adts_frame_blocks_f32`].
    pub fn decode_adts_frame_blocks_fixed_interleaved_i16(
        &mut self,
        input: &[u8],
    ) -> Result<Vec<Vec<i16>>, DecodeError> {
        let frame = AdtsFrame::parse(input)?;
        validate_adts_aac_lc_configuration(self, frame.header)?;
        if frame.header.protection_absent && frame.header.number_of_raw_data_blocks_in_frame != 0 {
            let mut reader = BitReader::new(frame.payload);
            let frames = (0..=frame.header.number_of_raw_data_blocks_in_frame)
                .map(|_| {
                    self.decode_raw_data_block_fixed_interleaved_i16_terminated_from_reader(
                        &mut reader,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            if !reader.remaining_bits_are_zero() {
                return Err(DecodeError::NonZeroTrailingBits(reader.remaining_bits()));
            }
            return Ok(frames);
        }
        if !frame.header.protection_absent && frame.header.number_of_raw_data_blocks_in_frame != 0 {
            frame.validate_multi_block_header_crc()?;
        }
        frame
            .raw_data_blocks()?
            .into_iter()
            .map(|block| {
                let decoded = self.decode_raw_data_block_fixed_interleaved_i16(block.payload)?;
                if let Some(expected) = block.crc_check {
                    self.validate_adts_syntax_crc(frame, block.payload, expected, false)?;
                }
                Ok(decoded)
            })
            .collect()
    }

    pub fn decode_adts_frame_f32_strict(
        &mut self,
        input: &[u8],
    ) -> Result<DecodedAacLcFrame, DecodeError> {
        let frame = AdtsFrame::parse(input)?;
        if frame.header.profile + 1 != self.audio_object_type {
            return Err(DecodeError::UnsupportedAudioObjectType(
                frame.header.profile + 1,
            ));
        }
        if frame.header.number_of_raw_data_blocks_in_frame != 0 {
            return Err(DecodeError::UnsupportedRawBlocksInAdtsFrame(
                frame.header.number_of_raw_data_blocks_in_frame,
            ));
        }
        if frame.header.sampling_frequency_index != self.sampling_frequency_index {
            return Err(DecodeError::AdtsConfigChanged);
        }
        let decoded = self.decode_raw_data_block_f32_strict(frame.payload)?;
        if !frame.header.protection_absent {
            self.validate_adts_syntax_crc(
                frame,
                frame.payload,
                frame
                    .header
                    .crc_check
                    .ok_or(AdtsError::SyntaxRegionsRequiredForCrc)?,
                true,
            )?;
        }
        Ok(decoded)
    }

    pub fn decode_raw_data_block_interleaved_f32(
        &mut self,
        input: &[u8],
    ) -> Result<Vec<f32>, DecodeError> {
        Ok(self.decode_raw_data_block_f32(input)?.interleaved_f32())
    }

    pub fn decode_adts_frame_interleaved_f32(
        &mut self,
        input: &[u8],
    ) -> Result<Vec<f32>, DecodeError> {
        Ok(self.decode_adts_frame_f32(input)?.interleaved_f32())
    }

    pub fn decode_raw_data_block_interleaved_i16(
        &mut self,
        input: &[u8],
    ) -> Result<Vec<i16>, DecodeError> {
        Ok(self.decode_raw_data_block_f32(input)?.interleaved_i16())
    }

    pub fn decode_adts_frame_interleaved_i16(
        &mut self,
        input: &[u8],
    ) -> Result<Vec<i16>, DecodeError> {
        Ok(self.decode_adts_frame_f32(input)?.interleaved_i16())
    }

    pub fn decode_raw_data_block_fixed_interleaved_i16(
        &mut self,
        input: &[u8],
    ) -> Result<Vec<i16>, DecodeError> {
        let mut reader = BitReader::new(input);
        self.decode_raw_data_block_fixed_interleaved_i16_from_reader(&mut reader)
    }

    pub fn decode_raw_data_block_fixed_interleaved_i16_strict(
        &mut self,
        input: &[u8],
    ) -> Result<Vec<i16>, DecodeError> {
        let mut reader = BitReader::new(input);
        let pcm = self.decode_raw_data_block_fixed_interleaved_i16_from_reader(&mut reader)?;
        validate_zero_trailing_bits(&reader)?;
        Ok(pcm)
    }

    pub fn decode_raw_data_block_fixed_interleaved_i16_from_reader(
        &mut self,
        reader: &mut BitReader<'_>,
    ) -> Result<Vec<i16>, DecodeError> {
        self.decode_raw_data_block_multichannel_fixed_interleaved_i16_from_reader(reader)
    }

    /// Fixed-point counterpart of
    /// [`Self::decode_raw_data_block_f32_terminated_from_reader`].
    pub fn decode_raw_data_block_fixed_interleaved_i16_terminated_from_reader(
        &mut self,
        reader: &mut BitReader<'_>,
    ) -> Result<Vec<i16>, DecodeError> {
        let pcm = self.decode_raw_data_block_fixed_interleaved_i16_from_reader(reader)?;
        consume_raw_data_block_terminator(reader)?;
        Ok(pcm)
    }

    pub fn decode_adts_frame_fixed_interleaved_i16(
        &mut self,
        input: &[u8],
    ) -> Result<Vec<i16>, DecodeError> {
        let frame = AdtsFrame::parse(input)?;
        if frame.header.profile + 1 != self.audio_object_type {
            return Err(DecodeError::UnsupportedAudioObjectType(
                frame.header.profile + 1,
            ));
        }
        if frame.header.number_of_raw_data_blocks_in_frame != 0 {
            return Err(DecodeError::UnsupportedRawBlocksInAdtsFrame(
                frame.header.number_of_raw_data_blocks_in_frame,
            ));
        }
        if frame.header.sampling_frequency_index != self.sampling_frequency_index {
            return Err(DecodeError::AdtsConfigChanged);
        }
        let pcm = self.decode_raw_data_block_fixed_interleaved_i16(frame.payload)?;
        if !frame.header.protection_absent {
            self.validate_adts_syntax_crc(
                frame,
                frame.payload,
                frame
                    .header
                    .crc_check
                    .ok_or(AdtsError::SyntaxRegionsRequiredForCrc)?,
                true,
            )?;
        }
        Ok(pcm)
    }

    pub fn decode_adts_frame_fixed_interleaved_i16_strict(
        &mut self,
        input: &[u8],
    ) -> Result<Vec<i16>, DecodeError> {
        let frame = AdtsFrame::parse(input)?;
        if frame.header.profile + 1 != self.audio_object_type {
            return Err(DecodeError::UnsupportedAudioObjectType(
                frame.header.profile + 1,
            ));
        }
        if frame.header.number_of_raw_data_blocks_in_frame != 0 {
            return Err(DecodeError::UnsupportedRawBlocksInAdtsFrame(
                frame.header.number_of_raw_data_blocks_in_frame,
            ));
        }
        if frame.header.sampling_frequency_index != self.sampling_frequency_index {
            return Err(DecodeError::AdtsConfigChanged);
        }
        let pcm = self.decode_raw_data_block_fixed_interleaved_i16_strict(frame.payload)?;
        if !frame.header.protection_absent {
            self.validate_adts_syntax_crc(
                frame,
                frame.payload,
                frame
                    .header
                    .crc_check
                    .ok_or(AdtsError::SyntaxRegionsRequiredForCrc)?,
                true,
            )?;
        }
        Ok(pcm)
    }

    pub fn decode_adts_stream_interleaved_f32<'a>(
        &'a mut self,
        input: &'a [u8],
    ) -> DecodedAdtsStreamF32<'a> {
        DecodedAdtsStreamF32 {
            decoder: self,
            frames: AdtsStream::new(input),
            strict: false,
        }
    }

    pub fn decode_adts_stream_interleaved_f32_strict<'a>(
        &'a mut self,
        input: &'a [u8],
    ) -> DecodedAdtsStreamF32<'a> {
        DecodedAdtsStreamF32 {
            decoder: self,
            frames: AdtsStream::new(input),
            strict: true,
        }
    }

    pub fn decode_adts_stream_interleaved_i16<'a>(
        &'a mut self,
        input: &'a [u8],
    ) -> DecodedAdtsStreamI16<'a> {
        DecodedAdtsStreamI16 {
            decoder: self,
            frames: AdtsStream::new(input),
            strict: false,
        }
    }

    pub fn decode_adts_stream_interleaved_i16_strict<'a>(
        &'a mut self,
        input: &'a [u8],
    ) -> DecodedAdtsStreamI16<'a> {
        DecodedAdtsStreamI16 {
            decoder: self,
            frames: AdtsStream::new(input),
            strict: true,
        }
    }

    pub fn decode_adts_stream_fixed_interleaved_i16<'a>(
        &'a mut self,
        input: &'a [u8],
    ) -> DecodedAdtsStreamFixedI16<'a> {
        DecodedAdtsStreamFixedI16 {
            decoder: self,
            frames: AdtsStream::new(input),
            strict: false,
        }
    }

    pub fn decode_adts_stream_fixed_interleaved_i16_strict<'a>(
        &'a mut self,
        input: &'a [u8],
    ) -> DecodedAdtsStreamFixedI16<'a> {
        DecodedAdtsStreamFixedI16 {
            decoder: self,
            frames: AdtsStream::new(input),
            strict: true,
        }
    }

    pub fn decode_adts_stream_multichannel_f32<'a>(
        &'a mut self,
        input: &'a [u8],
    ) -> DecodedAdtsMultichannelStreamF32<'a> {
        DecodedAdtsMultichannelStreamF32 {
            decoder: self,
            frames: AdtsStream::new(input),
            strict: false,
        }
    }

    pub fn decode_adts_stream_multichannel_f32_strict<'a>(
        &'a mut self,
        input: &'a [u8],
    ) -> DecodedAdtsMultichannelStreamF32<'a> {
        DecodedAdtsMultichannelStreamF32 {
            decoder: self,
            frames: AdtsStream::new(input),
            strict: true,
        }
    }

    pub fn decode_adts_stream_multichannel_interleaved_f32<'a>(
        &'a mut self,
        input: &'a [u8],
    ) -> DecodedAdtsMultichannelInterleavedStreamF32<'a> {
        DecodedAdtsMultichannelInterleavedStreamF32 {
            decoder: self,
            frames: AdtsStream::new(input),
            strict: false,
        }
    }

    pub fn decode_adts_stream_multichannel_interleaved_f32_strict<'a>(
        &'a mut self,
        input: &'a [u8],
    ) -> DecodedAdtsMultichannelInterleavedStreamF32<'a> {
        DecodedAdtsMultichannelInterleavedStreamF32 {
            decoder: self,
            frames: AdtsStream::new(input),
            strict: true,
        }
    }

    pub fn decode_adts_stream_multichannel_interleaved_i16<'a>(
        &'a mut self,
        input: &'a [u8],
    ) -> DecodedAdtsMultichannelInterleavedStreamI16<'a> {
        DecodedAdtsMultichannelInterleavedStreamI16 {
            decoder: self,
            frames: AdtsStream::new(input),
            strict: false,
        }
    }

    pub fn decode_adts_stream_multichannel_interleaved_i16_strict<'a>(
        &'a mut self,
        input: &'a [u8],
    ) -> DecodedAdtsMultichannelInterleavedStreamI16<'a> {
        DecodedAdtsMultichannelInterleavedStreamI16 {
            decoder: self,
            frames: AdtsStream::new(input),
            strict: true,
        }
    }

    pub fn decode_adts_stream_multichannel_fixed_interleaved_i16<'a>(
        &'a mut self,
        input: &'a [u8],
    ) -> DecodedAdtsMultichannelFixedInterleavedStreamI16<'a> {
        DecodedAdtsMultichannelFixedInterleavedStreamI16 {
            decoder: self,
            frames: AdtsStream::new(input),
            strict: false,
        }
    }

    pub fn decode_adts_stream_multichannel_fixed_interleaved_i16_strict<'a>(
        &'a mut self,
        input: &'a [u8],
    ) -> DecodedAdtsMultichannelFixedInterleavedStreamI16<'a> {
        DecodedAdtsMultichannelFixedInterleavedStreamI16 {
            decoder: self,
            frames: AdtsStream::new(input),
            strict: true,
        }
    }

    pub fn decode_raw_data_block_multichannel_f32(
        &mut self,
        input: &[u8],
    ) -> Result<DecodedAacLcMultichannelFrame, DecodeError> {
        let mut reader = BitReader::new(input);
        self.decode_raw_data_block_multichannel_f32_from_reader(&mut reader)
    }

    pub fn decode_raw_data_block_multichannel_f32_strict(
        &mut self,
        input: &[u8],
    ) -> Result<DecodedAacLcMultichannelFrame, DecodeError> {
        let mut reader = BitReader::new(input);
        let frame = self.decode_raw_data_block_multichannel_f32_from_reader(&mut reader)?;
        validate_zero_trailing_bits(&reader)?;
        Ok(frame)
    }

    pub fn decode_raw_data_block_multichannel_f32_from_reader(
        &mut self,
        reader: &mut BitReader<'_>,
    ) -> Result<DecodedAacLcMultichannelFrame, DecodeError> {
        self.age_legacy_drc();
        self.legacy_drc_control_applied = false;
        self.ancillary_data.clear();
        self.decode_raw_data_block_multichannel_f32_inner(reader)
    }

    fn decode_raw_data_block_multichannel_f32_inner(
        &mut self,
        reader: &mut BitReader<'_>,
    ) -> Result<DecodedAacLcMultichannelFrame, DecodeError> {
        if matches!(self.audio_object_type, 17 | 20 | 23 | 39) {
            return self.decode_er_aac_lc_multichannel_f32_from_reader(reader);
        }
        self.clear_adts_crc_regions();
        let mut staged = Vec::new();
        let mut coupled = Vec::new();
        let mut program_config = match &self.initialization {
            DecoderInitialization::AudioSpecificConfig(config) => config.program_config.clone(),
            _ => None,
        };
        let mut sbr_payloads = Vec::new();
        while reader.remaining_bits() >= 3 {
            let expected_so_far = expected_channels_for_config(self.channel_configuration)
                .or_else(|| program_config.as_ref().map(|pce| pce.num_channels as usize));
            let audio_complete =
                expected_so_far.is_some_and(|expected| staged_channel_count(&staged) >= expected);
            let element_id = ElementId::from_bits(reader.read_u8(3)?);
            if audio_complete
                && matches!(
                    element_id,
                    ElementId::SingleChannel | ElementId::ChannelPair | ElementId::Lfe
                )
            {
                reader.push_back(3)?;
                break;
            }
            let crc_region_start = reader.bits_read();
            match element_id {
                ElementId::SingleChannel | ElementId::Lfe => {
                    let element_instance_tag = reader.read_u8(4)?;
                    reader.push_back(7)?;
                    let pce_labels = program_config.as_ref().map(|pce| {
                        program_config_labels_for_element(
                            pce,
                            element_id,
                            element_instance_tag,
                            staged_channel_count(&staged),
                        )
                    });
                    let spectra = decode_aac_lc_single_channel_spectra_staged_from_reader(
                        reader,
                        self.sampling_frequency_index,
                        self.frame_length,
                        &mut self.pns_random,
                        false,
                    )?;
                    self.push_adts_crc_region(
                        crc_region_start..reader.bits_read().min(crc_region_start + 192),
                        192,
                    );
                    staged.push(StagedAacLcElement::Single {
                        element_id,
                        element_instance_tag,
                        spectra,
                        labels: pce_labels.unwrap_or_default(),
                    });
                }
                ElementId::ChannelPair => {
                    let element_instance_tag = reader.read_u8(4)?;
                    reader.push_back(7)?;
                    let pce_labels = program_config.as_ref().map(|pce| {
                        program_config_labels_for_element(
                            pce,
                            element_id,
                            element_instance_tag,
                            staged_channel_count(&staged),
                        )
                    });
                    let spectra = decode_aac_lc_channel_pair_spectra_staged_from_reader(
                        reader,
                        self.sampling_frequency_index,
                        self.frame_length,
                        &mut self.pns_random,
                        false,
                    )?;
                    self.push_adts_crc_region(
                        crc_region_start..reader.bits_read().min(crc_region_start + 192),
                        192,
                    );
                    let right_start = crc_region_start + spectra.right_channel_start_bit;
                    self.push_adts_crc_region(
                        right_start..reader.bits_read().min(right_start + 128),
                        128,
                    );
                    staged.push(StagedAacLcElement::Pair {
                        element_instance_tag,
                        spectra,
                        labels: pce_labels.unwrap_or_default(),
                    });
                }
                ElementId::CouplingChannel => {
                    reader.push_back(3)?;
                    coupled.push(decode_aac_lc_coupling_channel_element_from_reader(
                        reader,
                        self.sampling_frequency_index,
                    )?);
                    self.push_adts_crc_region(
                        crc_region_start..reader.bits_read().min(crc_region_start + 192),
                        192,
                    );
                }
                ElementId::DataStream => {
                    self.read_data_stream_element(reader)?;
                    let end = reader.bits_read();
                    self.push_adts_crc_region(crc_region_start..end, end - crc_region_start);
                }
                ElementId::ProgramConfig => {
                    let parsed = ProgramConfig::parse_from_reader(reader)?;
                    if let Some(matrix) = parsed.matrix_mixdown {
                        self.legacy_matrix_mixdown = Some(matrix);
                        self.legacy_drc_age_frames = 0;
                    }
                    program_config = Some(parsed);
                    let end = reader.bits_read();
                    self.push_adts_crc_region(crc_region_start..end, end - crc_region_start);
                }
                ElementId::Fill => {
                    if fill_extension_type(reader)? == Some(0x0b) {
                        self.read_mpeg4_drc_fill(reader)?;
                    } else if let Some(payload) = parse_sbr_fill_element(reader)? {
                        sbr_payloads.push(payload);
                    }
                }
                ElementId::End => {
                    // Leave ID_END for transport parsers that must locate the
                    // next raw_data_block. Configured layouts normally stop
                    // before peeking it; channelConfiguration=0 reaches it
                    // here when its PCE was supplied out of band (e.g. ADIF).
                    reader.push_back(3)?;
                    break;
                }
            }
        }
        if staged.is_empty() {
            return Err(DecodeError::NoAudioElement);
        }
        apply_staged_frequency_couplings(
            &mut staged,
            &coupled,
            CouplingPoint::BeforeTns,
            self.sampling_frequency_index,
        )?;
        apply_tns_to_staged_spectra(&mut staged, self.sampling_frequency_index)?;
        apply_staged_frequency_couplings(
            &mut staged,
            &coupled,
            CouplingPoint::BetweenTnsAndImdct,
            self.sampling_frequency_index,
        )?;
        let mut channels = Vec::new();
        let mut matched_labels = Vec::new();
        let channel_map = staged_channel_map(&staged);
        let sbr_stereo_elements = staged
            .iter()
            .map(|element| matches!(element, StagedAacLcElement::Pair { .. }))
            .collect::<Vec<_>>();
        let legacy_drc_in_core_domain =
            sbr_payloads.is_empty() && self.ordinary_sbr_output_frequency.is_none();
        self.ensure_channel_filterbanks(staged_channel_count(&staged))?;
        self.legacy_drc_window_sequences.clear();
        let mut concealment_spectra = Vec::new();
        let mut channel_index = 0usize;
        for element in staged {
            match element {
                StagedAacLcElement::Single {
                    mut spectra,
                    labels,
                    ..
                } => {
                    self.legacy_drc_window_sequences
                        .push(spectra.stream.ics.window_sequence);
                    concealment_spectra
                        .push((spectra.stream.spectrum.clone(), spectra.stream.ics.clone()));
                    if legacy_drc_in_core_domain {
                        self.apply_legacy_drc_to_f32_spectrum(
                            &mut spectra.stream.spectrum,
                            channel_index,
                        );
                    }
                    let samples = synthesize_aac_lc_frame(
                        &spectra.stream.spectrum,
                        &spectra.stream.ics,
                        &mut self.channel_filterbanks[channel_index],
                    )?;
                    channels.push(samples);
                    matched_labels.extend(labels);
                    channel_index += 1;
                }
                StagedAacLcElement::Pair {
                    mut spectra,
                    labels,
                    ..
                } => {
                    apply_aac_lc_channel_pair_stereo_tools_fixed_bridge(
                        &mut spectra,
                        self.sampling_frequency_index,
                    )?;
                    self.legacy_drc_window_sequences
                        .push(spectra.left.ics.window_sequence);
                    self.legacy_drc_window_sequences
                        .push(spectra.right.ics.window_sequence);
                    concealment_spectra
                        .push((spectra.left.spectrum.clone(), spectra.left.ics.clone()));
                    concealment_spectra
                        .push((spectra.right.spectrum.clone(), spectra.right.ics.clone()));
                    if legacy_drc_in_core_domain {
                        self.apply_legacy_drc_to_f32_spectrum(
                            &mut spectra.left.spectrum,
                            channel_index,
                        );
                        self.apply_legacy_drc_to_f32_spectrum(
                            &mut spectra.right.spectrum,
                            channel_index + 1,
                        );
                    }
                    let (left_banks, right_banks) =
                        self.channel_filterbanks.split_at_mut(channel_index + 1);
                    let left_fb = &mut left_banks[channel_index];
                    let right_fb = &mut right_banks[0];
                    channels.push(synthesize_aac_lc_frame(
                        &spectra.left.spectrum,
                        &spectra.left.ics,
                        left_fb,
                    )?);
                    channels.push(synthesize_aac_lc_frame(
                        &spectra.right.spectrum,
                        &spectra.right.ics,
                        right_fb,
                    )?);
                    matched_labels.extend(labels);
                    channel_index += 2;
                }
            }
        }
        let time_couplings = coupled
            .iter()
            .enumerate()
            .filter(|(_, cce)| cce.prefix.uses_time_coupling())
            .collect::<Vec<_>>();
        if !time_couplings.is_empty() {
            self.ensure_coupling_filterbanks(time_couplings.len())?;
            for (bank_index, (_, cce)) in time_couplings.into_iter().enumerate() {
                let coupling_samples = synthesize_aac_lc_frame(
                    &cce.stream.spectrum,
                    &cce.stream.ics,
                    &mut self.coupling_filterbanks[bank_index],
                )?;
                apply_time_domain_cce_to_channels(
                    &mut channels,
                    &channel_map,
                    cce,
                    &coupling_samples,
                )?;
            }
        }
        self.process_ordinary_sbr_f32(&mut channels, &sbr_payloads, &sbr_stereo_elements)?;
        let expected_channels = expected_channels_for_config(self.channel_configuration)
            .or_else(|| program_config.as_ref().map(|pce| pce.num_channels as usize));
        let expected_channels = expected_channels.map(|count| {
            if self.ps_signaled && count == 1 {
                2
            } else {
                count
            }
        });
        if let Some(expected) = expected_channels {
            if channels.len() != expected {
                return Err(DecodeError::ChannelConfigurationMismatch {
                    expected,
                    actual: channels.len(),
                });
            }
        }
        let labels = if self.ps_signaled && channels.len() == 2 {
            vec![ChannelLabel::FrontLeft, ChannelLabel::FrontRight]
        } else if !matched_labels.is_empty() && matched_labels.len() == channels.len() {
            matched_labels
        } else {
            channel_labels_for_config(self.channel_configuration)
                .map(|labels| labels.to_vec())
                .or_else(|| program_config.as_ref().map(program_config_channel_labels))
                .unwrap_or_else(|| unknown_channel_labels(channels.len()))
        };
        if self.f32_concealment_losses != 0 {
            self.f32_concealment_fade_in_remaining = self.f32_concealment_losses.min(5);
        }
        if self.f32_concealment_fade_in_remaining != 0 {
            self.f32_concealment_state = ConcealmentState::FadeIn;
            apply_f32_concealment_recovery_fade(
                &mut channels,
                self.f32_concealment_fade_in_remaining,
            );
            self.f32_concealment_fade_in_remaining -= 1;
        } else {
            self.f32_concealment_state = ConcealmentState::Ok;
        }
        self.apply_configured_drc_f32(&mut channels)?;
        self.f32_concealment_spectra = concealment_spectra;
        self.f32_concealment_losses = 0;
        self.f32_concealment_phase = 0;
        Ok(DecodedAacLcMultichannelFrame { channels, labels })
    }

    fn decode_er_aac_lc_multichannel_f32_from_reader(
        &mut self,
        reader: &mut BitReader<'_>,
    ) -> Result<DecodedAacLcMultichannelFrame, DecodeError> {
        if !matches!(self.error_protection_config, Some(0 | 1)) {
            return Err(DecodeError::ErrorResilienceUnsupported);
        }
        let elements: &[ElementId] = if self.eld_sac_decoder.is_some() {
            &[ElementId::SingleChannel]
        } else {
            er_channel_elements(self.channel_configuration).ok_or(
                DecodeError::UnsupportedChannelConfiguration(self.channel_configuration),
            )?
        };
        let mut staged = Vec::with_capacity(elements.len());
        for &element_id in elements {
            if element_id == ElementId::ChannelPair {
                let spectra = decode_er_channel_pair_spectra_from_reader(
                    reader,
                    self.sampling_frequency_index,
                    self.frame_length,
                    self.audio_object_type == 39,
                    self.error_protection_config == Some(1),
                    self.er_resilience_flags[0],
                    self.er_resilience_flags[1],
                    self.er_resilience_flags[2],
                    &mut self.pns_random,
                )?;
                staged.push(StagedAacLcElement::Pair {
                    element_instance_tag: spectra.prefix.element_instance_tag,
                    spectra,
                    labels: Vec::new(),
                });
            } else {
                let spectra = decode_er_single_channel_spectra_from_reader(
                    reader,
                    element_id,
                    self.sampling_frequency_index,
                    self.frame_length,
                    self.audio_object_type == 39,
                    self.er_resilience_flags[0],
                    self.er_resilience_flags[1],
                    self.er_resilience_flags[2],
                    &mut self.pns_random,
                )?;
                staged.push(StagedAacLcElement::Single {
                    element_id,
                    element_instance_tag: spectra.side_info.element_instance_tag,
                    spectra,
                    labels: Vec::new(),
                });
            }
        }
        let ld_sbr_frames = self.parse_ld_sbr_frames(reader)?;
        self.parse_er_extension_payloads(reader)?;
        let channel_count = staged_channel_count(&staged);
        self.ensure_channel_filterbanks(channel_count)?;
        let mut channels = Vec::with_capacity(channel_count);
        let mut concealment_spectra = Vec::with_capacity(channel_count);
        let mut channel_index = 0;
        for element in staged {
            match element {
                StagedAacLcElement::Single { spectra, .. } => {
                    concealment_spectra
                        .push((spectra.stream.spectrum.clone(), spectra.stream.ics.clone()));
                    channels.push(if self.audio_object_type == 39 {
                        synthesize_aac_eld_frame_f32(
                            &spectra.stream.spectrum,
                            &mut self.eld_channel_filterbanks[channel_index],
                        )?
                    } else {
                        let mut pcm = synthesize_aac_lc_frame(
                            &spectra.stream.spectrum,
                            &spectra.stream.ics,
                            &mut self.channel_filterbanks[channel_index],
                        )?;
                        if self.audio_object_type == 23 {
                            let scale = 1.0 / self.frame_length as f32;
                            pcm.iter_mut().for_each(|sample| *sample *= scale);
                        }
                        pcm
                    });
                    channel_index += 1;
                }
                StagedAacLcElement::Pair { mut spectra, .. } => {
                    apply_aac_lc_channel_pair_stereo_tools_fixed_bridge(
                        &mut spectra,
                        self.sampling_frequency_index,
                    )?;
                    concealment_spectra
                        .push((spectra.left.spectrum.clone(), spectra.left.ics.clone()));
                    concealment_spectra
                        .push((spectra.right.spectrum.clone(), spectra.right.ics.clone()));
                    if self.audio_object_type == 39 {
                        let (left_banks, right_banks) =
                            self.eld_channel_filterbanks.split_at_mut(channel_index + 1);
                        channels.push(synthesize_aac_eld_frame_f32(
                            &spectra.left.spectrum,
                            &mut left_banks[channel_index],
                        )?);
                        channels.push(synthesize_aac_eld_frame_f32(
                            &spectra.right.spectrum,
                            &mut right_banks[0],
                        )?);
                    } else {
                        let (left_banks, right_banks) =
                            self.channel_filterbanks.split_at_mut(channel_index + 1);
                        let mut left = synthesize_aac_lc_frame(
                            &spectra.left.spectrum,
                            &spectra.left.ics,
                            &mut left_banks[channel_index],
                        )?;
                        let mut right = synthesize_aac_lc_frame(
                            &spectra.right.spectrum,
                            &spectra.right.ics,
                            &mut right_banks[0],
                        )?;
                        if self.audio_object_type == 23 {
                            let scale = 1.0 / self.frame_length as f32;
                            left.iter_mut().for_each(|sample| *sample *= scale);
                            right.iter_mut().for_each(|sample| *sample *= scale);
                        }
                        channels.push(left);
                        channels.push(right);
                    }
                    channel_index += 2;
                }
            }
        }
        self.process_ld_sbr_f32(&mut channels, &ld_sbr_frames)?;
        self.process_eld_sac_f32(&mut channels)?;
        if !ld_sbr_frames.is_empty() {
            self.last_ld_sbr_frames = ld_sbr_frames.clone();
        }
        if self.f32_concealment_losses != 0 {
            self.f32_concealment_fade_in_remaining = self.f32_concealment_losses.min(5);
        }
        if self.f32_concealment_fade_in_remaining != 0 {
            self.f32_concealment_state = ConcealmentState::FadeIn;
            apply_f32_concealment_recovery_fade(
                &mut channels,
                self.f32_concealment_fade_in_remaining,
            );
            self.f32_concealment_fade_in_remaining -= 1;
        } else {
            self.f32_concealment_state = ConcealmentState::Ok;
        }
        self.apply_configured_drc_f32(&mut channels)?;
        self.f32_concealment_spectra = concealment_spectra;
        self.f32_concealment_losses = 0;
        self.f32_concealment_phase = 0;
        let labels = if self.eld_sac_decoder.is_some() {
            vec![ChannelLabel::FrontLeft, ChannelLabel::FrontRight]
        } else {
            channel_labels_for_config(self.channel_configuration)
                .ok_or(DecodeError::UnsupportedChannelConfiguration(
                    self.channel_configuration,
                ))?
                .to_vec()
        };
        Ok(DecodedAacLcMultichannelFrame { channels, labels })
    }

    fn decode_er_aac_lc_multichannel_fixed_i16_from_reader(
        &mut self,
        reader: &mut BitReader<'_>,
    ) -> Result<Vec<i16>, DecodeError> {
        if !matches!(self.error_protection_config, Some(0 | 1)) {
            return Err(DecodeError::ErrorResilienceUnsupported);
        }
        let elements: &[ElementId] = if self.eld_sac_decoder.is_some() {
            &[ElementId::SingleChannel]
        } else {
            er_channel_elements(self.channel_configuration).ok_or(
                DecodeError::UnsupportedChannelConfiguration(self.channel_configuration),
            )?
        };
        let mut staged = Vec::with_capacity(elements.len());
        for &element_id in elements {
            if element_id == ElementId::ChannelPair {
                let spectra = decode_er_channel_pair_spectra_fixed_from_reader(
                    reader,
                    self.sampling_frequency_index,
                    self.frame_length,
                    self.audio_object_type == 39,
                    self.error_protection_config == Some(1),
                    self.er_resilience_flags[0],
                    self.er_resilience_flags[1],
                    self.er_resilience_flags[2],
                    &mut self.pns_random,
                )?;
                staged.push(StagedAacLcElementFixed::Pair {
                    element_instance_tag: spectra.prefix.element_instance_tag,
                    spectra,
                    labels: Vec::new(),
                });
            } else {
                let spectra = decode_er_single_channel_spectra_fixed_from_reader(
                    reader,
                    element_id,
                    self.sampling_frequency_index,
                    self.frame_length,
                    self.audio_object_type == 39,
                    self.er_resilience_flags[0],
                    self.er_resilience_flags[1],
                    self.er_resilience_flags[2],
                    &mut self.pns_random,
                )?;
                staged.push(StagedAacLcElementFixed::Single {
                    element_id,
                    element_instance_tag: spectra.side_info.element_instance_tag,
                    spectra,
                    labels: Vec::new(),
                });
            }
        }
        let ld_sbr_frames = self.parse_ld_sbr_frames(reader)?;
        self.parse_er_extension_payloads(reader)?;
        let channel_count = staged_fixed_channel_count(&staged);
        self.ensure_fixed_channel_filterbanks(channel_count)?;
        let mut channels = Vec::with_capacity(channel_count);
        let mut eld_q31_channels = Vec::with_capacity(channel_count);
        let mut concealment_spectra = Vec::with_capacity(channel_count);
        let mut channel_index = 0;
        for element in staged {
            match element {
                StagedAacLcElementFixed::Single { spectra, .. } => {
                    concealment_spectra
                        .push((spectra.stream.spectrum.clone(), spectra.stream.ics.clone()));
                    if self.audio_object_type == 39 {
                        eld_q31_channels.push(synthesize_aac_eld_frame_fixed_q31(
                            &spectra.stream.spectrum,
                            &mut self.eld_fixed_channel_filterbanks[channel_index],
                        )?);
                    } else {
                        channels.push(
                            synthesize_aac_lc_frame_from_fixed_inverse_q31(
                                &spectra.stream.spectrum,
                                &spectra.stream.ics,
                                &mut self.fixed_channel_filterbanks[channel_index],
                            )?
                            .into_iter()
                            .map(|sample| {
                                dbl_to_pcm16(if self.audio_object_type == 23 {
                                    sample >> 4
                                } else {
                                    sample
                                })
                            })
                            .collect::<Vec<_>>(),
                        );
                    }
                    channel_index += 1;
                }
                StagedAacLcElementFixed::Pair { mut spectra, .. } => {
                    apply_aac_lc_channel_pair_fixed_spectrum_stereo_tools_bridge(
                        &mut spectra,
                        self.sampling_frequency_index,
                    )?;
                    concealment_spectra
                        .push((spectra.left.spectrum.clone(), spectra.left.ics.clone()));
                    concealment_spectra
                        .push((spectra.right.spectrum.clone(), spectra.right.ics.clone()));
                    if self.audio_object_type == 39 {
                        let (left_banks, right_banks) = self
                            .eld_fixed_channel_filterbanks
                            .split_at_mut(channel_index + 1);
                        eld_q31_channels.push(synthesize_aac_eld_frame_fixed_q31(
                            &spectra.left.spectrum,
                            &mut left_banks[channel_index],
                        )?);
                        eld_q31_channels.push(synthesize_aac_eld_frame_fixed_q31(
                            &spectra.right.spectrum,
                            &mut right_banks[0],
                        )?);
                    } else {
                        let (left_banks, right_banks) = self
                            .fixed_channel_filterbanks
                            .split_at_mut(channel_index + 1);
                        channels.push(
                            synthesize_aac_lc_frame_from_fixed_inverse_q31(
                                &spectra.left.spectrum,
                                &spectra.left.ics,
                                &mut left_banks[channel_index],
                            )?
                            .into_iter()
                            .map(|sample| {
                                dbl_to_pcm16(if self.audio_object_type == 23 {
                                    sample >> 4
                                } else {
                                    sample
                                })
                            })
                            .collect(),
                        );
                        channels.push(
                            synthesize_aac_lc_frame_from_fixed_inverse_q31(
                                &spectra.right.spectrum,
                                &spectra.right.ics,
                                &mut right_banks[0],
                            )?
                            .into_iter()
                            .map(|sample| {
                                dbl_to_pcm16(if self.audio_object_type == 23 {
                                    sample >> 4
                                } else {
                                    sample
                                })
                            })
                            .collect(),
                        );
                    }
                    channel_index += 2;
                }
            }
        }
        if self.audio_object_type == 39 {
            channels = if ld_sbr_frames.is_empty() {
                eld_q31_channels
                    .into_iter()
                    .map(|channel| channel.into_iter().map(dbl_to_pcm16).collect())
                    .collect()
            } else {
                self.process_ld_sbr_fixed_q31(&eld_q31_channels, &ld_sbr_frames)?
            };
        } else {
            self.process_ld_sbr_fixed(&mut channels, &ld_sbr_frames)?;
        }
        self.process_eld_sac_i16(&mut channels)?;
        if !ld_sbr_frames.is_empty() {
            self.last_ld_sbr_frames = ld_sbr_frames.clone();
        }
        self.apply_configured_drc_i16(&mut channels)?;
        self.fixed_concealment_spectra = concealment_spectra;
        self.fixed_concealment_losses = 0;
        Ok(interleave_multichannel_i16_samples(&channels))
    }

    pub fn decode_raw_data_block_multichannel_fixed_interleaved_i16(
        &mut self,
        input: &[u8],
    ) -> Result<Vec<i16>, DecodeError> {
        let mut reader = BitReader::new(input);
        self.decode_raw_data_block_multichannel_fixed_interleaved_i16_from_reader(&mut reader)
    }

    pub fn decode_raw_data_block_multichannel_fixed_interleaved_i16_strict(
        &mut self,
        input: &[u8],
    ) -> Result<Vec<i16>, DecodeError> {
        let mut reader = BitReader::new(input);
        let pcm =
            self.decode_raw_data_block_multichannel_fixed_interleaved_i16_from_reader(&mut reader)?;
        validate_zero_trailing_bits(&reader)?;
        Ok(pcm)
    }

    pub fn decode_raw_data_block_multichannel_fixed_interleaved_i16_from_reader(
        &mut self,
        reader: &mut BitReader<'_>,
    ) -> Result<Vec<i16>, DecodeError> {
        self.age_legacy_drc();
        self.legacy_drc_control_applied = false;
        self.ancillary_data.clear();
        self.decode_raw_data_block_multichannel_fixed_interleaved_i16_inner(reader)
    }

    fn decode_raw_data_block_multichannel_fixed_interleaved_i16_inner(
        &mut self,
        reader: &mut BitReader<'_>,
    ) -> Result<Vec<i16>, DecodeError> {
        if matches!(self.audio_object_type, 17 | 20 | 23 | 39) {
            return self.decode_er_aac_lc_multichannel_fixed_i16_from_reader(reader);
        }
        self.clear_adts_crc_regions();
        let recovery_losses = self.fixed_concealment_losses;
        let mut staged: Vec<StagedAacLcElementFixed> = Vec::new();
        let mut coupled = Vec::new();
        let mut program_config = match &self.initialization {
            DecoderInitialization::AudioSpecificConfig(config) => config.program_config.clone(),
            _ => None,
        };
        let mut sbr_payloads = Vec::new();
        while reader.remaining_bits() >= 3 {
            let expected_so_far = expected_channels_for_config(self.channel_configuration)
                .or_else(|| program_config.as_ref().map(|pce| pce.num_channels as usize));
            let audio_complete = expected_so_far
                .is_some_and(|expected| staged_fixed_channel_count(&staged) >= expected);
            let element_id = ElementId::from_bits(reader.read_u8(3)?);
            if audio_complete
                && matches!(
                    element_id,
                    ElementId::SingleChannel | ElementId::ChannelPair | ElementId::Lfe
                )
            {
                reader.push_back(3)?;
                break;
            }
            let crc_region_start = reader.bits_read();
            match element_id {
                ElementId::SingleChannel | ElementId::Lfe => {
                    let element_instance_tag = reader.read_u8(4)?;
                    reader.push_back(7)?;
                    let spectra = decode_aac_lc_single_channel_spectra_fixed_staged_from_reader(
                        reader,
                        self.sampling_frequency_index,
                        self.frame_length,
                        &mut self.pns_random,
                        false,
                    )?;
                    self.push_adts_crc_region(
                        crc_region_start..reader.bits_read().min(crc_region_start + 192),
                        192,
                    );
                    staged.push(StagedAacLcElementFixed::Single {
                        element_id,
                        element_instance_tag,
                        spectra,
                        labels: Vec::new(),
                    });
                }
                ElementId::ChannelPair => {
                    let element_instance_tag = reader.read_u8(4)?;
                    reader.push_back(7)?;
                    let spectra = decode_aac_lc_channel_pair_spectra_fixed_staged_from_reader(
                        reader,
                        self.sampling_frequency_index,
                        self.frame_length,
                        &mut self.pns_random,
                        false,
                    )?;
                    self.push_adts_crc_region(
                        crc_region_start..reader.bits_read().min(crc_region_start + 192),
                        192,
                    );
                    let right_start = crc_region_start + spectra.right_channel_start_bit;
                    self.push_adts_crc_region(
                        right_start..reader.bits_read().min(right_start + 128),
                        128,
                    );
                    staged.push(StagedAacLcElementFixed::Pair {
                        element_instance_tag,
                        spectra,
                        labels: Vec::new(),
                    });
                }
                ElementId::CouplingChannel => {
                    reader.push_back(3)?;
                    coupled.push(
                        decode_aac_lc_coupling_channel_element_fixed_bridge_from_reader(
                            reader,
                            self.sampling_frequency_index,
                            &mut self.pns_random,
                        )?,
                    );
                    self.push_adts_crc_region(
                        crc_region_start..reader.bits_read().min(crc_region_start + 192),
                        192,
                    );
                }
                ElementId::DataStream => {
                    self.read_data_stream_element(reader)?;
                    let end = reader.bits_read();
                    self.push_adts_crc_region(crc_region_start..end, end - crc_region_start);
                }
                ElementId::ProgramConfig => {
                    let parsed = ProgramConfig::parse_from_reader(reader)?;
                    if let Some(matrix) = parsed.matrix_mixdown {
                        self.legacy_matrix_mixdown = Some(matrix);
                        self.legacy_drc_age_frames = 0;
                    }
                    program_config = Some(parsed);
                    let end = reader.bits_read();
                    self.push_adts_crc_region(crc_region_start..end, end - crc_region_start);
                }
                ElementId::Fill => {
                    if fill_extension_type(reader)? == Some(0x0b) {
                        self.read_mpeg4_drc_fill(reader)?;
                    } else if let Some(payload) = parse_sbr_fill_element(reader)? {
                        sbr_payloads.push(payload);
                    }
                }
                ElementId::End => {
                    reader.push_back(3)?;
                    break;
                }
            }
        }
        if staged.is_empty() {
            return Err(DecodeError::NoAudioElement);
        }
        apply_staged_fixed_frequency_couplings(
            &mut staged,
            &coupled,
            CouplingPoint::BeforeTns,
            self.sampling_frequency_index,
        )?;
        apply_tns_to_staged_fixed_spectra(&mut staged, self.sampling_frequency_index)?;
        apply_staged_fixed_frequency_couplings(
            &mut staged,
            &coupled,
            CouplingPoint::BetweenTnsAndImdct,
            self.sampling_frequency_index,
        )?;

        let channel_count = staged_fixed_channel_count(&staged);
        let channel_map = staged_fixed_channel_map(&staged);
        let sbr_stereo_elements = staged
            .iter()
            .map(|element| matches!(element, StagedAacLcElementFixed::Pair { .. }))
            .collect::<Vec<_>>();
        let legacy_drc_in_core_domain =
            sbr_payloads.is_empty() && self.ordinary_sbr_output_frequency.is_none();
        self.ensure_fixed_channel_filterbanks(channel_count)?;
        self.legacy_drc_window_sequences.clear();
        let mut channels = Vec::new();
        let mut concealment_spectra = Vec::with_capacity(channel_count);
        let mut channel_index = 0usize;
        for element in staged {
            match element {
                StagedAacLcElementFixed::Single { mut spectra, .. } => {
                    self.legacy_drc_window_sequences
                        .push(spectra.stream.ics.window_sequence);
                    concealment_spectra
                        .push((spectra.stream.spectrum.clone(), spectra.stream.ics.clone()));
                    if legacy_drc_in_core_domain {
                        self.apply_legacy_drc_to_fixed_spectrum(
                            &mut spectra.stream.spectrum,
                            channel_index,
                        );
                    }
                    let samples = synthesize_aac_lc_frame_from_fixed_inverse_q31(
                        &spectra.stream.spectrum,
                        &spectra.stream.ics,
                        &mut self.fixed_channel_filterbanks[channel_index],
                    )?;
                    channels.push(samples);
                    channel_index += 1;
                }
                StagedAacLcElementFixed::Pair { mut spectra, .. } => {
                    apply_aac_lc_channel_pair_fixed_spectrum_stereo_tools_bridge(
                        &mut spectra,
                        self.sampling_frequency_index,
                    )?;
                    self.legacy_drc_window_sequences
                        .push(spectra.left.ics.window_sequence);
                    self.legacy_drc_window_sequences
                        .push(spectra.right.ics.window_sequence);
                    concealment_spectra
                        .push((spectra.left.spectrum.clone(), spectra.left.ics.clone()));
                    concealment_spectra
                        .push((spectra.right.spectrum.clone(), spectra.right.ics.clone()));
                    if legacy_drc_in_core_domain {
                        self.apply_legacy_drc_to_fixed_spectrum(
                            &mut spectra.left.spectrum,
                            channel_index,
                        );
                        self.apply_legacy_drc_to_fixed_spectrum(
                            &mut spectra.right.spectrum,
                            channel_index + 1,
                        );
                    }
                    let (left_banks, right_banks) = self
                        .fixed_channel_filterbanks
                        .split_at_mut(channel_index + 1);
                    let left_fb = &mut left_banks[channel_index];
                    let right_fb = &mut right_banks[0];
                    channels.push(synthesize_aac_lc_frame_from_fixed_inverse_q31(
                        &spectra.left.spectrum,
                        &spectra.left.ics,
                        left_fb,
                    )?);
                    channels.push(synthesize_aac_lc_frame_from_fixed_inverse_q31(
                        &spectra.right.spectrum,
                        &spectra.right.ics,
                        right_fb,
                    )?);
                    channel_index += 2;
                }
            }
        }
        let time_couplings = coupled
            .iter()
            .filter(|cce| cce.prefix.uses_time_coupling())
            .collect::<Vec<_>>();
        if !time_couplings.is_empty() {
            self.ensure_fixed_coupling_filterbanks(time_couplings.len())?;
            for (bank_index, cce) in time_couplings.into_iter().enumerate() {
                let coupling_samples = synthesize_aac_lc_frame_from_fixed_inverse_q31(
                    &cce.stream.spectrum,
                    &cce.stream.ics,
                    &mut self.fixed_coupling_filterbanks[bank_index],
                )?;
                apply_time_domain_cce_to_fixed_channels_fixed_cce(
                    &mut channels,
                    &channel_map,
                    cce,
                    &coupling_samples,
                )?;
            }
        }
        let expected_channels = expected_channels_for_config(self.channel_configuration)
            .or_else(|| program_config.as_ref().map(|pce| pce.num_channels as usize));
        if let Some(expected) = expected_channels {
            if channels.len() != expected {
                return Err(DecodeError::ChannelConfigurationMismatch {
                    expected,
                    actual: channels.len(),
                });
            }
        }
        if recovery_losses != 0 {
            self.fixed_concealment_fade_in_remaining = recovery_losses.min(5);
        }
        if self.fixed_concealment_fade_in_remaining != 0 {
            self.fixed_concealment_state = ConcealmentState::FadeIn;
            apply_fixed_concealment_recovery_fade(
                &mut channels,
                self.fixed_concealment_fade_in_remaining,
            );
            self.fixed_concealment_fade_in_remaining -= 1;
        } else {
            self.fixed_concealment_state = ConcealmentState::Ok;
        }
        self.fixed_concealment_spectra = concealment_spectra;
        self.fixed_concealment_losses = 0;
        self.fixed_concealment_phase = 0;
        let mut channels = channels
            .into_iter()
            .map(|channel| channel.into_iter().map(dbl_to_pcm16).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        self.process_ordinary_sbr_fixed(&mut channels, &sbr_payloads, &sbr_stereo_elements)?;
        self.apply_configured_drc_i16(&mut channels)?;
        Ok(interleave_multichannel_i16_samples(&channels))
    }

    /// Synthesize one missing AAC-LC frame from the most recently decoded
    /// fixed-point spectra. This mirrors FDK's interpolation concealment entry
    /// point: preserve filterbank overlap, transition short/start windows to a
    /// stop window, randomize spectral signs, and fade repeated losses.
    pub fn conceal_fixed_interleaved_i16(&mut self) -> Result<Vec<i16>, DecodeError> {
        self.conceal_fixed_interleaved_i16_mode(false)
    }

    /// Spectral-muting concealment. The zero spectrum is still passed through
    /// the active synthesis filterbank so overlap and low-delay state decay in
    /// the same way as FDK's `ConcealMethodMute`.
    pub fn conceal_fixed_muted_i16(&mut self) -> Result<Vec<i16>, DecodeError> {
        self.conceal_fixed_interleaved_i16_mode(true)
    }

    fn conceal_fixed_interleaved_i16_mode(
        &mut self,
        spectral_mute: bool,
    ) -> Result<Vec<i16>, DecodeError> {
        self.legacy_drc_control_applied = false;
        if self.fixed_concealment_spectra.is_empty() {
            return Err(DecodeError::NoConcealmentReference);
        }
        let legacy_drc_in_core_domain = self.last_ordinary_sbr_fixed_frames.is_empty()
            && self.last_ld_sbr_frames.is_empty()
            && self.ordinary_sbr_output_frequency.is_none();
        let channel_count = self.fixed_concealment_spectra.len();
        self.fixed_concealment_state = if spectral_mute {
            ConcealmentState::Mute
        } else {
            match self.fixed_concealment_losses {
                0 => ConcealmentState::Single,
                1..=6 => ConcealmentState::FadeOut,
                _ => ConcealmentState::Mute,
            }
        };
        self.ensure_fixed_channel_filterbanks(channel_count)?;
        let mut channels = Vec::with_capacity(channel_count);
        for channel_index in 0..channel_count {
            let (mut spectrum, mut ics) = self.fixed_concealment_spectra[channel_index].clone();
            if spectral_mute {
                for window in &mut spectrum.windows {
                    window.fill(0);
                }
            } else {
                prepare_fixed_concealment_spectrum(
                    &mut spectrum,
                    self.fixed_concealment_losses,
                    &mut self.fixed_concealment_phase,
                );
            }
            if matches!(
                ics.window_sequence,
                WindowSequence::LongStart | WindowSequence::EightShort
            ) {
                ics.window_sequence = WindowSequence::LongStop;
            }
            if legacy_drc_in_core_domain {
                self.apply_legacy_drc_to_fixed_spectrum(&mut spectrum, channel_index);
            }
            if self.audio_object_type == 39 {
                channels.push(synthesize_aac_eld_frame_fixed_i16(
                    &spectrum,
                    &mut self.eld_fixed_channel_filterbanks[channel_index],
                )?);
            } else {
                let samples = synthesize_aac_lc_frame_from_fixed_inverse_q31(
                    &spectrum,
                    &ics,
                    &mut self.fixed_channel_filterbanks[channel_index],
                )?;
                channels.push(samples.into_iter().map(dbl_to_pcm16).collect::<Vec<_>>());
            }
        }
        if self.audio_object_type == 39 && !self.last_ld_sbr_frames.is_empty() {
            let frames = self.last_ld_sbr_frames.clone();
            self.process_ld_sbr_fixed(&mut channels, &frames)?;
        }
        if !self.last_ordinary_sbr_fixed_frames.is_empty() {
            self.conceal_ordinary_sbr_fixed(&mut channels)?;
        }
        self.apply_configured_drc_i16(&mut channels)?;
        self.fixed_concealment_losses = self.fixed_concealment_losses.saturating_add(1);
        Ok(interleave_multichannel_i16_samples(&channels))
    }

    /// Conceal one missing frame between the stored previous good spectrum and
    /// a look-ahead next good spectrum.
    pub fn conceal_fixed_interpolated_i16(
        &mut self,
        next: &FixedConcealmentSpectralFrame,
    ) -> Result<Vec<i16>, DecodeError> {
        self.legacy_drc_control_applied = false;
        if self.fixed_concealment_spectra.is_empty() {
            return Err(DecodeError::NoConcealmentReference);
        }
        if self.fixed_concealment_spectra.len() != next.channels.len() {
            return Err(DecodeError::ConcealmentInterpolation(
                SpectralInterpolationError::LayoutMismatch,
            ));
        }
        if matches!(self.audio_object_type, 23 | 39) {
            return self.conceal_eld_fixed_interpolated_i16(next);
        }
        let legacy_drc_in_core_domain = self.last_ordinary_sbr_fixed_frames.is_empty()
            && self.ordinary_sbr_output_frequency.is_none();
        self.ensure_fixed_channel_filterbanks(next.channels.len())?;
        let mut channels = Vec::with_capacity(next.channels.len());
        for (index, ((previous, previous_ics), next_channel)) in self
            .fixed_concealment_spectra
            .clone()
            .into_iter()
            .zip(&next.channels)
            .enumerate()
        {
            let long_bands =
                aac_lc_sfb_info(self.sampling_frequency_index, WindowSequence::OnlyLong)?;
            let short_bands =
                aac_lc_sfb_info(self.sampling_frequency_index, WindowSequence::EightShort)?;
            let (mut interpolated, interpolated_sequence) = interpolate_fixed_spectra_mixed(
                &previous,
                previous_ics.window_sequence,
                &next_channel.spectrum,
                next_channel.ics.window_sequence,
                long_bands.offsets,
                short_bands.offsets,
            )?;
            let mut interpolated_ics = previous_ics;
            interpolated_ics.window_sequence = interpolated_sequence;
            interpolated_ics.window_shape = match interpolated_sequence {
                WindowSequence::LongStart | WindowSequence::EightShort => {
                    next_channel.ics.window_shape
                }
                WindowSequence::OnlyLong | WindowSequence::LongStop => {
                    interpolated_ics.window_shape
                }
            };
            randomize_fixed_spectrum_signs(&mut interpolated, &mut self.fixed_concealment_phase);
            if legacy_drc_in_core_domain {
                self.apply_legacy_drc_to_fixed_spectrum(&mut interpolated, index);
            }
            let samples = synthesize_aac_lc_frame_from_fixed_inverse_q31(
                &interpolated,
                &interpolated_ics,
                &mut self.fixed_channel_filterbanks[index],
            )?;
            channels.push(samples.into_iter().map(dbl_to_pcm16).collect::<Vec<_>>());
        }
        // A successfully interpolated isolated loss transitions from SINGLE
        // straight back to OK in FDK; the following good frame is not faded.
        self.fixed_concealment_losses = 0;
        self.fixed_concealment_state = ConcealmentState::Single;
        if !self.last_ordinary_sbr_fixed_frames.is_empty() {
            self.conceal_ordinary_sbr_fixed(&mut channels)?;
        }
        self.apply_configured_drc_i16(&mut channels)?;
        Ok(interleave_multichannel_i16_samples(&channels))
    }

    fn conceal_eld_fixed_interpolated_i16(
        &mut self,
        next: &FixedConcealmentSpectralFrame,
    ) -> Result<Vec<i16>, DecodeError> {
        let legacy_drc_in_core_domain = self.last_ld_sbr_frames.is_empty();
        let mut channels = Vec::with_capacity(next.channels.len());
        for (index, ((previous, _), next_channel)) in self
            .fixed_concealment_spectra
            .clone()
            .into_iter()
            .zip(&next.channels)
            .enumerate()
        {
            if previous.windows.len() != 1
                || next_channel.spectrum.windows.len() != 1
                || previous.windows[0].len() != next_channel.spectrum.windows[0].len()
            {
                return Err(DecodeError::ConcealmentInterpolation(
                    SpectralInterpolationError::LayoutMismatch,
                ));
            }
            let mut interpolated = FixedInverseQuantizedSpectrum {
                windows: vec![previous.windows[0]
                    .iter()
                    .zip(&next_channel.spectrum.windows[0])
                    .map(|(&left, &right)| {
                        (((left as f64).powi(2) + (right as f64).powi(2)) * 0.5)
                            .sqrt()
                            .min(i32::MAX as f64) as i32
                    })
                    .collect()],
                window_exponents: vec![previous.window_exponents.first().copied().unwrap_or(0)],
            };
            randomize_fixed_spectrum_signs(&mut interpolated, &mut self.fixed_concealment_phase);
            if legacy_drc_in_core_domain {
                self.apply_legacy_drc_to_fixed_spectrum(&mut interpolated, index);
            }
            if self.audio_object_type == 39 {
                channels.push(synthesize_aac_eld_frame_fixed_i16(
                    &interpolated,
                    &mut self.eld_fixed_channel_filterbanks[index],
                )?);
            } else {
                let samples = synthesize_aac_lc_frame_from_fixed_inverse_q31(
                    &interpolated,
                    &self.fixed_concealment_spectra[index].1,
                    &mut self.fixed_channel_filterbanks[index],
                )?;
                channels.push(samples.into_iter().map(dbl_to_pcm16).collect());
            }
        }
        if !self.last_ld_sbr_frames.is_empty() {
            let frames = self.last_ld_sbr_frames.clone();
            self.process_ld_sbr_fixed(&mut channels, &frames)?;
        }
        self.fixed_concealment_losses = 0;
        self.fixed_concealment_state = ConcealmentState::Single;
        self.apply_configured_drc_i16(&mut channels)?;
        Ok(interleave_multichannel_i16_samples(&channels))
    }

    /// Synthesize one missing AAC-LC frame from the previous floating-point
    /// spectrum while preserving IMDCT overlap and FDK fade state.
    pub fn conceal_f32_interleaved(&mut self) -> Result<Vec<f32>, DecodeError> {
        self.conceal_f32_interleaved_mode(false)
    }

    /// Floating-point counterpart of [`Self::conceal_fixed_muted_i16`].
    pub fn conceal_f32_muted(&mut self) -> Result<Vec<f32>, DecodeError> {
        self.conceal_f32_interleaved_mode(true)
    }

    fn conceal_f32_interleaved_mode(
        &mut self,
        spectral_mute: bool,
    ) -> Result<Vec<f32>, DecodeError> {
        self.legacy_drc_control_applied = false;
        if self.f32_concealment_spectra.is_empty() {
            return Err(DecodeError::NoConcealmentReference);
        }
        let legacy_drc_in_core_domain = self.last_ordinary_sbr_frames.is_empty()
            && self.last_ld_sbr_frames.is_empty()
            && self.ordinary_sbr_output_frequency.is_none();
        let channel_count = self.f32_concealment_spectra.len();
        self.f32_concealment_state = if spectral_mute {
            ConcealmentState::Mute
        } else {
            match self.f32_concealment_losses {
                0 => ConcealmentState::Single,
                1..=6 => ConcealmentState::FadeOut,
                _ => ConcealmentState::Mute,
            }
        };
        self.ensure_channel_filterbanks(channel_count)?;
        let mut channels = Vec::with_capacity(channel_count);
        for channel_index in 0..channel_count {
            let (mut spectrum, mut ics) = self.f32_concealment_spectra[channel_index].clone();
            if spectral_mute {
                for window in &mut spectrum.windows {
                    window.fill(0.0);
                }
            } else {
                prepare_f32_concealment_spectrum(
                    &mut spectrum,
                    self.f32_concealment_losses,
                    &mut self.f32_concealment_phase,
                );
            }
            if matches!(
                ics.window_sequence,
                WindowSequence::LongStart | WindowSequence::EightShort
            ) {
                ics.window_sequence = WindowSequence::LongStop;
            }
            if legacy_drc_in_core_domain {
                self.apply_legacy_drc_to_f32_spectrum(&mut spectrum, channel_index);
            }
            if self.audio_object_type == 39 {
                channels.push(synthesize_aac_eld_frame_f32(
                    &spectrum,
                    &mut self.eld_channel_filterbanks[channel_index],
                )?);
            } else {
                channels.push(synthesize_aac_lc_frame(
                    &spectrum,
                    &ics,
                    &mut self.channel_filterbanks[channel_index],
                )?);
            }
        }
        if self.audio_object_type == 39 && !self.last_ld_sbr_frames.is_empty() {
            let frames = self.last_ld_sbr_frames.clone();
            self.process_ld_sbr_f32(&mut channels, &frames)?;
        }
        if !self.last_ordinary_sbr_frames.is_empty() {
            self.conceal_ordinary_sbr_f32(&mut channels)?;
        }
        self.apply_configured_drc_f32(&mut channels)?;
        self.f32_concealment_losses = self.f32_concealment_losses.saturating_add(1);
        Ok(interleave_multichannel_f32(&channels))
    }

    pub fn conceal_f32_interpolated(
        &mut self,
        next: &F32ConcealmentSpectralFrame,
    ) -> Result<Vec<f32>, DecodeError> {
        self.legacy_drc_control_applied = false;
        if self.f32_concealment_spectra.is_empty() {
            return Err(DecodeError::NoConcealmentReference);
        }
        if self.f32_concealment_spectra.len() != next.channels.len() {
            return Err(DecodeError::ConcealmentInterpolation(
                SpectralInterpolationError::LayoutMismatch,
            ));
        }
        if matches!(self.audio_object_type, 23 | 39) {
            return self.conceal_eld_f32_interpolated(next);
        }
        let legacy_drc_in_core_domain = self.last_ordinary_sbr_frames.is_empty()
            && self.ordinary_sbr_output_frequency.is_none();
        self.ensure_channel_filterbanks(next.channels.len())?;
        let long_bands = aac_lc_sfb_info(self.sampling_frequency_index, WindowSequence::OnlyLong)?;
        let short_bands =
            aac_lc_sfb_info(self.sampling_frequency_index, WindowSequence::EightShort)?;
        let mut channels = Vec::with_capacity(next.channels.len());
        for (index, ((previous, previous_ics), next_channel)) in self
            .f32_concealment_spectra
            .clone()
            .into_iter()
            .zip(&next.channels)
            .enumerate()
        {
            let (mut interpolated, interpolated_sequence) = interpolate_f32_spectra_mixed(
                &previous,
                previous_ics.window_sequence,
                &next_channel.spectrum,
                next_channel.ics.window_sequence,
                long_bands.offsets,
                short_bands.offsets,
            )?;
            let mut interpolated_ics = previous_ics;
            interpolated_ics.window_sequence = interpolated_sequence;
            if matches!(
                interpolated_sequence,
                WindowSequence::LongStart | WindowSequence::EightShort
            ) {
                interpolated_ics.window_shape = next_channel.ics.window_shape;
            }
            randomize_f32_spectrum_signs(&mut interpolated, &mut self.f32_concealment_phase);
            if legacy_drc_in_core_domain {
                self.apply_legacy_drc_to_f32_spectrum(&mut interpolated, index);
            }
            channels.push(synthesize_aac_lc_frame(
                &interpolated,
                &interpolated_ics,
                &mut self.channel_filterbanks[index],
            )?);
        }
        self.f32_concealment_losses = 0;
        self.f32_concealment_state = ConcealmentState::Single;
        if !self.last_ordinary_sbr_frames.is_empty() {
            self.conceal_ordinary_sbr_f32(&mut channels)?;
        }
        self.apply_configured_drc_f32(&mut channels)?;
        Ok(interleave_multichannel_f32(&channels))
    }

    fn conceal_eld_f32_interpolated(
        &mut self,
        next: &F32ConcealmentSpectralFrame,
    ) -> Result<Vec<f32>, DecodeError> {
        let legacy_drc_in_core_domain = self.last_ld_sbr_frames.is_empty();
        let mut channels = Vec::with_capacity(next.channels.len());
        for (index, ((previous, _), next_channel)) in self
            .f32_concealment_spectra
            .clone()
            .into_iter()
            .zip(&next.channels)
            .enumerate()
        {
            if previous.windows.len() != 1
                || next_channel.spectrum.windows.len() != 1
                || previous.windows[0].len() != next_channel.spectrum.windows[0].len()
            {
                return Err(DecodeError::ConcealmentInterpolation(
                    SpectralInterpolationError::LayoutMismatch,
                ));
            }
            let mut interpolated = InverseQuantizedSpectrum {
                windows: vec![previous.windows[0]
                    .iter()
                    .zip(&next_channel.spectrum.windows[0])
                    .map(|(&left, &right)| ((left * left + right * right) * 0.5).sqrt())
                    .collect()],
            };
            randomize_f32_spectrum_signs(&mut interpolated, &mut self.f32_concealment_phase);
            if legacy_drc_in_core_domain {
                self.apply_legacy_drc_to_f32_spectrum(&mut interpolated, index);
            }
            if self.audio_object_type == 39 {
                channels.push(synthesize_aac_eld_frame_f32(
                    &interpolated,
                    &mut self.eld_channel_filterbanks[index],
                )?);
            } else {
                channels.push(synthesize_aac_lc_frame(
                    &interpolated,
                    &self.f32_concealment_spectra[index].1,
                    &mut self.channel_filterbanks[index],
                )?);
            }
        }
        if !self.last_ld_sbr_frames.is_empty() {
            let frames = self.last_ld_sbr_frames.clone();
            self.process_ld_sbr_f32(&mut channels, &frames)?;
        }
        self.f32_concealment_losses = 0;
        self.f32_concealment_state = ConcealmentState::Single;
        self.apply_configured_drc_f32(&mut channels)?;
        Ok(interleave_multichannel_f32(&channels))
    }

    pub fn decode_adts_frame_multichannel_f32(
        &mut self,
        input: &[u8],
    ) -> Result<DecodedAacLcMultichannelFrame, DecodeError> {
        let frame = AdtsFrame::parse(input)?;
        if frame.header.profile + 1 != self.audio_object_type {
            return Err(DecodeError::UnsupportedAudioObjectType(
                frame.header.profile + 1,
            ));
        }
        if frame.header.number_of_raw_data_blocks_in_frame != 0 {
            return Err(DecodeError::UnsupportedRawBlocksInAdtsFrame(
                frame.header.number_of_raw_data_blocks_in_frame,
            ));
        }
        if frame.header.sampling_frequency_index != self.sampling_frequency_index {
            return Err(DecodeError::AdtsConfigChanged);
        }
        let decoded = self.decode_raw_data_block_multichannel_f32(frame.payload)?;
        if !frame.header.protection_absent {
            self.validate_adts_syntax_crc(
                frame,
                frame.payload,
                frame
                    .header
                    .crc_check
                    .ok_or(AdtsError::SyntaxRegionsRequiredForCrc)?,
                true,
            )?;
        }
        Ok(decoded)
    }

    pub fn decode_adts_frame_multichannel_f32_strict(
        &mut self,
        input: &[u8],
    ) -> Result<DecodedAacLcMultichannelFrame, DecodeError> {
        let frame = AdtsFrame::parse(input)?;
        if frame.header.profile + 1 != self.audio_object_type {
            return Err(DecodeError::UnsupportedAudioObjectType(
                frame.header.profile + 1,
            ));
        }
        if frame.header.number_of_raw_data_blocks_in_frame != 0 {
            return Err(DecodeError::UnsupportedRawBlocksInAdtsFrame(
                frame.header.number_of_raw_data_blocks_in_frame,
            ));
        }
        if frame.header.sampling_frequency_index != self.sampling_frequency_index {
            return Err(DecodeError::AdtsConfigChanged);
        }
        let decoded = self.decode_raw_data_block_multichannel_f32_strict(frame.payload)?;
        if !frame.header.protection_absent {
            self.validate_adts_syntax_crc(
                frame,
                frame.payload,
                frame
                    .header
                    .crc_check
                    .ok_or(AdtsError::SyntaxRegionsRequiredForCrc)?,
                true,
            )?;
        }
        Ok(decoded)
    }

    pub fn decode_adts_frame_multichannel_fixed_interleaved_i16(
        &mut self,
        input: &[u8],
    ) -> Result<Vec<i16>, DecodeError> {
        let frame = AdtsFrame::parse(input)?;
        if frame.header.profile + 1 != self.audio_object_type {
            return Err(DecodeError::UnsupportedAudioObjectType(
                frame.header.profile + 1,
            ));
        }
        if frame.header.number_of_raw_data_blocks_in_frame != 0 {
            return Err(DecodeError::UnsupportedRawBlocksInAdtsFrame(
                frame.header.number_of_raw_data_blocks_in_frame,
            ));
        }
        if frame.header.sampling_frequency_index != self.sampling_frequency_index {
            return Err(DecodeError::AdtsConfigChanged);
        }
        let pcm = self.decode_raw_data_block_multichannel_fixed_interleaved_i16(frame.payload)?;
        if !frame.header.protection_absent {
            self.validate_adts_syntax_crc(
                frame,
                frame.payload,
                frame
                    .header
                    .crc_check
                    .ok_or(AdtsError::SyntaxRegionsRequiredForCrc)?,
                true,
            )?;
        }
        Ok(pcm)
    }

    pub fn decode_adts_frame_multichannel_fixed_interleaved_i16_strict(
        &mut self,
        input: &[u8],
    ) -> Result<Vec<i16>, DecodeError> {
        let frame = AdtsFrame::parse(input)?;
        if frame.header.profile + 1 != self.audio_object_type {
            return Err(DecodeError::UnsupportedAudioObjectType(
                frame.header.profile + 1,
            ));
        }
        if frame.header.number_of_raw_data_blocks_in_frame != 0 {
            return Err(DecodeError::UnsupportedRawBlocksInAdtsFrame(
                frame.header.number_of_raw_data_blocks_in_frame,
            ));
        }
        if frame.header.sampling_frequency_index != self.sampling_frequency_index {
            return Err(DecodeError::AdtsConfigChanged);
        }
        let pcm =
            self.decode_raw_data_block_multichannel_fixed_interleaved_i16_strict(frame.payload)?;
        if !frame.header.protection_absent {
            self.validate_adts_syntax_crc(
                frame,
                frame.payload,
                frame
                    .header
                    .crc_check
                    .ok_or(AdtsError::SyntaxRegionsRequiredForCrc)?,
                true,
            )?;
        }
        Ok(pcm)
    }

    pub fn decode_adts_frame_multichannel_interleaved_f32(
        &mut self,
        input: &[u8],
    ) -> Result<Vec<f32>, DecodeError> {
        Ok(self
            .decode_adts_frame_multichannel_f32(input)?
            .interleaved_f32())
    }

    pub fn decode_adts_frame_multichannel_interleaved_i16(
        &mut self,
        input: &[u8],
    ) -> Result<Vec<i16>, DecodeError> {
        Ok(self
            .decode_adts_frame_multichannel_f32(input)?
            .interleaved_i16())
    }

    fn validate_frame_channel_configuration(
        &self,
        frame: &DecodedAacLcFrame,
    ) -> Result<(), DecodeError> {
        match self.channel_configuration {
            0 => Ok(()),
            1 if frame.channels() == 1 => Ok(()),
            1 if self.ps_signaled && frame.channels() == 2 => Ok(()),
            2 if frame.channels() == 2 => Ok(()),
            1 | 2 => Err(DecodeError::ChannelConfigurationMismatch {
                expected: self.channel_configuration as usize,
                actual: frame.channels(),
            }),
            other => Err(DecodeError::UnsupportedChannelConfiguration(other)),
        }
    }

    fn validate_adts_syntax_crc(
        &self,
        frame: AdtsFrame<'_>,
        payload: &[u8],
        expected: u16,
        include_header: bool,
    ) -> Result<(), DecodeError> {
        let mut regions = Vec::with_capacity(self.adts_crc_regions.len() + 1);
        if include_header {
            regions.push((frame.bytes, 0..56, 56));
        }
        regions.extend(
            self.adts_crc_regions
                .iter()
                .cloned()
                .zip(self.adts_crc_padded_bits.iter().copied())
                .map(|(range, padded_bits)| (payload, range, padded_bits)),
        );
        let calculated = adts_crc16_padded_bit_regions(regions)?;
        if calculated == expected {
            Ok(())
        } else {
            Err(AdtsError::CrcMismatch {
                expected,
                calculated,
            }
            .into())
        }
    }

    fn ensure_channel_filterbanks(&mut self, channels: usize) -> Result<(), DecodeError> {
        while self.channel_filterbanks.len() < channels {
            self.channel_filterbanks
                .push(LongBlockFilterbank::new(self.frame_length)?);
        }
        if self.audio_object_type == 39 {
            while self.eld_channel_filterbanks.len() < channels {
                self.eld_channel_filterbanks
                    .push(LowDelayFilterbankF32::new(self.frame_length)?);
            }
        }
        Ok(())
    }

    fn parse_ld_sbr_frames(
        &mut self,
        reader: &mut BitReader<'_>,
    ) -> Result<Vec<LdSbrFrame>, DecodeError> {
        self.ld_sbr_parsers
            .iter_mut()
            .map(|parser| Ok(parser.parse(reader)?))
            .collect()
    }

    fn process_ld_sbr_f32(
        &mut self,
        channels: &mut [Vec<f32>],
        frames: &[LdSbrFrame],
    ) -> Result<(), DecodeError> {
        let mut sbr_channel = 0usize;
        for frame in frames {
            let channel = *self
                .ld_sbr_channel_indices
                .get(sbr_channel)
                .ok_or(DecodeError::SbrPayloadLayoutMismatch)?;
            let core = channels[channel]
                .iter()
                .map(|&sample| sample as f64)
                .collect::<Vec<_>>();
            channels[channel] = self.ld_sbr_processors[channel]
                .process(&core, frame, false)?
                .into_iter()
                .map(|sample| sample as f32)
                .collect();
            sbr_channel += 1;
            if frame.right.is_some() {
                let channel = *self
                    .ld_sbr_channel_indices
                    .get(sbr_channel)
                    .ok_or(DecodeError::SbrPayloadLayoutMismatch)?;
                let core = channels[channel]
                    .iter()
                    .map(|&sample| sample as f64)
                    .collect::<Vec<_>>();
                channels[channel] = self.ld_sbr_processors[channel]
                    .process(&core, frame, true)?
                    .into_iter()
                    .map(|sample| sample as f32)
                    .collect();
                sbr_channel += 1;
            }
        }
        if self.eld_sbr_dual_rate {
            for channel in 0..channels.len() {
                if self.ld_sbr_channel_indices.contains(&channel) {
                    continue;
                }
                let core = channels[channel]
                    .iter()
                    .map(|&sample| sample as f64)
                    .collect::<Vec<_>>();
                channels[channel] = self.ld_sbr_processors[channel]
                    .upsample_only(&core)?
                    .into_iter()
                    .map(|sample| sample as f32)
                    .collect();
            }
        }
        Ok(())
    }

    fn process_ld_sbr_fixed(
        &mut self,
        channels: &mut [Vec<i16>],
        frames: &[LdSbrFrame],
    ) -> Result<(), DecodeError> {
        let mut sbr_channel = 0usize;
        for frame in frames {
            let channel = *self
                .ld_sbr_channel_indices
                .get(sbr_channel)
                .ok_or(DecodeError::SbrPayloadLayoutMismatch)?;
            let core = channels[channel]
                .iter()
                .map(|&sample| sample as f64 / 32768.0)
                .collect::<Vec<_>>();
            channels[channel] = self.ld_sbr_fixed_processors[channel]
                .process(&core, frame, false)?
                .into_iter()
                .map(|sample| f32_to_i16(sample as f32))
                .collect();
            sbr_channel += 1;
            if frame.right.is_some() {
                let channel = *self
                    .ld_sbr_channel_indices
                    .get(sbr_channel)
                    .ok_or(DecodeError::SbrPayloadLayoutMismatch)?;
                let core = channels[channel]
                    .iter()
                    .map(|&sample| sample as f64 / 32768.0)
                    .collect::<Vec<_>>();
                channels[channel] = self.ld_sbr_fixed_processors[channel]
                    .process(&core, frame, true)?
                    .into_iter()
                    .map(|sample| f32_to_i16(sample as f32))
                    .collect();
                sbr_channel += 1;
            }
        }
        if self.eld_sbr_dual_rate {
            for channel in 0..channels.len() {
                if self.ld_sbr_channel_indices.contains(&channel) {
                    continue;
                }
                let core = channels[channel]
                    .iter()
                    .map(|&sample| sample as f64 / 32768.0)
                    .collect::<Vec<_>>();
                channels[channel] = self.ld_sbr_fixed_processors[channel]
                    .upsample_only(&core)?
                    .into_iter()
                    .map(|sample| f32_to_i16(sample as f32))
                    .collect();
            }
        }
        Ok(())
    }

    fn process_ld_sbr_fixed_q31(
        &mut self,
        channels: &[Vec<FixpDbl>],
        frames: &[LdSbrFrame],
    ) -> Result<Vec<Vec<i16>>, DecodeError> {
        if self.ld_sbr_channel_indices.is_empty() && !channels.is_empty() {
            return Err(DecodeError::SbrPayloadLayoutMismatch);
        }
        let mut output = vec![Vec::new(); channels.len()];
        let mut sbr_channel = 0usize;
        for frame in frames {
            let channel = *self
                .ld_sbr_channel_indices
                .get(sbr_channel)
                .ok_or(DecodeError::SbrPayloadLayoutMismatch)?;
            let core = channels[channel]
                .iter()
                .map(|&sample| sample as f64 / 2048.0)
                .collect::<Vec<_>>();
            output[channel] = self.ld_sbr_fixed_processors[channel]
                .process(&core, frame, false)?
                .into_iter()
                .map(|sample| eld_raw_pcm_to_i16(sample as f32))
                .collect();
            sbr_channel += 1;
            if frame.right.is_some() {
                let channel = *self
                    .ld_sbr_channel_indices
                    .get(sbr_channel)
                    .ok_or(DecodeError::SbrPayloadLayoutMismatch)?;
                let core = channels[channel]
                    .iter()
                    .map(|&sample| sample as f64 / 2048.0)
                    .collect::<Vec<_>>();
                output[channel] = self.ld_sbr_fixed_processors[channel]
                    .process(&core, frame, true)?
                    .into_iter()
                    .map(|sample| eld_raw_pcm_to_i16(sample as f32))
                    .collect();
                sbr_channel += 1;
            }
        }
        if sbr_channel != self.ld_sbr_channel_indices.len() {
            return Err(DecodeError::SbrPayloadLayoutMismatch);
        }
        for channel in 0..channels.len() {
            if !output[channel].is_empty() {
                continue;
            }
            let core = channels[channel]
                .iter()
                .map(|&sample| sample as f64 / 2048.0)
                .collect::<Vec<_>>();
            output[channel] = if self.eld_sbr_dual_rate {
                self.ld_sbr_fixed_processors[channel]
                    .upsample_only(&core)?
                    .into_iter()
                    .map(|sample| eld_raw_pcm_to_i16(sample as f32))
                    .collect()
            } else {
                core.into_iter()
                    .map(|sample| eld_raw_pcm_to_i16(sample as f32))
                    .collect()
            };
        }
        Ok(output)
    }

    fn process_ordinary_sbr_f32(
        &mut self,
        channels: &mut Vec<Vec<f32>>,
        payloads: &[SbrFillPayload],
        stereo_elements: &[bool],
    ) -> Result<(), DecodeError> {
        let Some(output_frequency) = self.ordinary_sbr_output_frequency else {
            return Ok(());
        };
        if payloads.is_empty() {
            return if self.last_ordinary_sbr_frames.is_empty() {
                Ok(())
            } else {
                self.conceal_ordinary_sbr_f32(channels)
            };
        }
        if payloads.len() > stereo_elements.len() {
            return Err(DecodeError::SbrPayloadLayoutMismatch);
        }
        let mut channel = 0usize;
        let mut processor_channel = 0usize;
        let mut parsed_frames = Vec::with_capacity(payloads.len());
        for (element, (&stereo, payload)) in stereo_elements.iter().zip(payloads).enumerate() {
            if self.ordinary_sbr_parsers[element].is_none() {
                let header = payload
                    .header
                    .clone()
                    .ok_or(SbrError::MissingInitialHeader)?;
                self.ordinary_sbr_parsers[element] = Some(if stereo {
                    OrdinarySbrParser::Stereo(SbrStereoFrameParser::new(
                        header,
                        output_frequency,
                        self.frame_length,
                    )?)
                } else {
                    OrdinarySbrParser::Mono(SbrMonoFrameParser::new(
                        header,
                        output_frequency,
                        self.frame_length,
                    )?)
                });
            }
            match self.ordinary_sbr_parsers[element].as_mut().unwrap() {
                OrdinarySbrParser::Mono(parser) if !stereo => {
                    let frame = parser.parse(payload)?;
                    let core = channels[channel]
                        .iter()
                        .map(|&sample| sample as f64)
                        .collect::<Vec<_>>();
                    let ps_frame = self.ps_parsers[element].parse_sbr_extension(
                        &frame.extended_data,
                        (self.frame_length / 32) as u8,
                    )?;
                    if let Some(ps_frame) = ps_frame.filter(|_| !self.qmf_low_power) {
                        let mut slots = self.ordinary_sbr_processors[processor_channel]
                            .process_channel_to_qmf(
                                &core,
                                &frame.active_header,
                                &frame.frequency_tables,
                                &frame.control,
                                &frame.values,
                                &frame.dequantized,
                                &frame.harmonics,
                                2,
                            )?;
                        self.apply_legacy_drc_to_qmf_slots(
                            processor_channel,
                            channel,
                            &mut slots,
                            false,
                        );
                        let (left, right) =
                            self.ps_processors[element].process_qmf(&slots, &ps_frame)?;
                        channels[channel] = left.into_iter().map(|sample| sample as f32).collect();
                        channels.insert(
                            channel + 1,
                            right.into_iter().map(|sample| sample as f32).collect(),
                        );
                        self.last_ps_frames[element] = Some(ps_frame);
                        self.ps_signaled = true;
                        channel += 2;
                    } else {
                        let mut slots = self.ordinary_sbr_processors[processor_channel]
                            .process_channel_to_qmf(
                                &core,
                                &frame.active_header,
                                &frame.frequency_tables,
                                &frame.control,
                                &frame.values,
                                &frame.dequantized,
                                &frame.harmonics,
                                2,
                            )?;
                        self.apply_legacy_drc_to_qmf_slots(
                            processor_channel,
                            channel,
                            &mut slots,
                            false,
                        );
                        channels[channel] = self.ordinary_sbr_processors[processor_channel]
                            .synthesize_qmf(&slots)?
                            .into_iter()
                            .map(|sample| sample as f32)
                            .collect();
                        channel += 1;
                    }
                    processor_channel += 1;
                    parsed_frames.push(OrdinarySbrFrame::Mono(frame));
                }
                OrdinarySbrParser::Stereo(parser) if stereo => {
                    let frame = parser.parse(payload)?;
                    for right in [false, true] {
                        let (control, raw, dequantized, harmonics) = if right {
                            (
                                &frame.right_control,
                                &frame.right,
                                &frame.right_dequantized,
                                &frame.right_harmonics,
                            )
                        } else {
                            (
                                &frame.left_control,
                                &frame.left,
                                &frame.left_dequantized,
                                &frame.left_harmonics,
                            )
                        };
                        let core = channels[channel]
                            .iter()
                            .map(|&sample| sample as f64)
                            .collect::<Vec<_>>();
                        let mut slots = self.ordinary_sbr_processors[processor_channel]
                            .process_channel_to_qmf(
                                &core,
                                &frame.active_header,
                                &frame.frequency_tables,
                                control,
                                raw,
                                dequantized,
                                harmonics,
                                2,
                            )?;
                        self.apply_legacy_drc_to_qmf_slots(
                            processor_channel,
                            channel,
                            &mut slots,
                            false,
                        );
                        channels[channel] = self.ordinary_sbr_processors[processor_channel]
                            .synthesize_qmf(&slots)?
                            .into_iter()
                            .map(|sample| sample as f32)
                            .collect();
                        channel += 1;
                        processor_channel += 1;
                    }
                    parsed_frames.push(OrdinarySbrFrame::Stereo(frame));
                }
                _ => return Err(DecodeError::SbrPayloadLayoutMismatch),
            }
        }
        while channel < channels.len() {
            let core = channels[channel]
                .iter()
                .map(|&sample| sample as f64)
                .collect::<Vec<_>>();
            channels[channel] = self.ordinary_sbr_processors[processor_channel]
                .upsample_only(&core)?
                .into_iter()
                .map(|sample| sample as f32)
                .collect();
            channel += 1;
            processor_channel += 1;
        }
        self.last_ordinary_sbr_frames = parsed_frames;
        Ok(())
    }

    fn process_ordinary_sbr_fixed(
        &mut self,
        channels: &mut Vec<Vec<i16>>,
        payloads: &[SbrFillPayload],
        stereo_elements: &[bool],
    ) -> Result<(), DecodeError> {
        let Some(output_frequency) = self.ordinary_sbr_output_frequency else {
            return Ok(());
        };
        if payloads.is_empty() {
            return if self.last_ordinary_sbr_fixed_frames.is_empty() {
                Ok(())
            } else {
                self.conceal_ordinary_sbr_fixed(channels)
            };
        }
        if payloads.len() > stereo_elements.len() {
            return Err(DecodeError::SbrPayloadLayoutMismatch);
        }
        let mut channel = 0usize;
        let mut processor_channel = 0usize;
        let mut parsed_frames = Vec::with_capacity(payloads.len());
        for (element, (&stereo, payload)) in stereo_elements.iter().zip(payloads).enumerate() {
            if self.ordinary_sbr_fixed_parsers[element].is_none() {
                let header = payload
                    .header
                    .clone()
                    .ok_or(SbrError::MissingInitialHeader)?;
                self.ordinary_sbr_fixed_parsers[element] = Some(if stereo {
                    OrdinarySbrParser::Stereo(SbrStereoFrameParser::new(
                        header,
                        output_frequency,
                        self.frame_length,
                    )?)
                } else {
                    OrdinarySbrParser::Mono(SbrMonoFrameParser::new(
                        header,
                        output_frequency,
                        self.frame_length,
                    )?)
                });
            }
            match self.ordinary_sbr_fixed_parsers[element].as_mut().unwrap() {
                OrdinarySbrParser::Mono(parser) if !stereo => {
                    let frame = parser.parse(payload)?;
                    let core = channels[channel]
                        .iter()
                        .map(|&sample| sample as f64 / 32768.0)
                        .collect::<Vec<_>>();
                    let ps_frame = self.ps_fixed_parsers[element].parse_sbr_extension(
                        &frame.extended_data,
                        (self.frame_length / 32) as u8,
                    )?;
                    if let Some(ps_frame) = ps_frame.filter(|_| !self.qmf_low_power) {
                        let mut slots = self.ordinary_sbr_fixed_processors[processor_channel]
                            .process_channel_to_qmf(
                                &core,
                                &frame.active_header,
                                &frame.frequency_tables,
                                &frame.control,
                                &frame.values,
                                &frame.dequantized,
                                &frame.harmonics,
                                2,
                            )?;
                        self.apply_legacy_drc_to_qmf_slots(
                            processor_channel,
                            channel,
                            &mut slots,
                            true,
                        );
                        let (left, right) =
                            self.ps_fixed_processors[element].process_qmf(&slots, &ps_frame)?;
                        channels[channel] = left
                            .into_iter()
                            .map(|sample| f32_to_i16(sample as f32))
                            .collect();
                        channels.insert(
                            channel + 1,
                            right
                                .into_iter()
                                .map(|sample| f32_to_i16(sample as f32))
                                .collect(),
                        );
                        self.last_ps_fixed_frames[element] = Some(ps_frame);
                        self.ps_signaled = true;
                        channel += 2;
                    } else {
                        let mut slots = self.ordinary_sbr_fixed_processors[processor_channel]
                            .process_channel_to_qmf(
                                &core,
                                &frame.active_header,
                                &frame.frequency_tables,
                                &frame.control,
                                &frame.values,
                                &frame.dequantized,
                                &frame.harmonics,
                                2,
                            )?;
                        self.apply_legacy_drc_to_qmf_slots(
                            processor_channel,
                            channel,
                            &mut slots,
                            true,
                        );
                        channels[channel] = self.ordinary_sbr_fixed_processors[processor_channel]
                            .synthesize_qmf(&slots)?
                            .into_iter()
                            .map(|sample| f32_to_i16(sample as f32))
                            .collect();
                        channel += 1;
                    }
                    processor_channel += 1;
                    parsed_frames.push(OrdinarySbrFrame::Mono(frame));
                }
                OrdinarySbrParser::Stereo(parser) if stereo => {
                    let frame = parser.parse(payload)?;
                    for right in [false, true] {
                        let (control, raw, dequantized, harmonics) = if right {
                            (
                                &frame.right_control,
                                &frame.right,
                                &frame.right_dequantized,
                                &frame.right_harmonics,
                            )
                        } else {
                            (
                                &frame.left_control,
                                &frame.left,
                                &frame.left_dequantized,
                                &frame.left_harmonics,
                            )
                        };
                        let core = channels[channel]
                            .iter()
                            .map(|&sample| sample as f64 / 32768.0)
                            .collect::<Vec<_>>();
                        let mut slots = self.ordinary_sbr_fixed_processors[processor_channel]
                            .process_channel_to_qmf(
                                &core,
                                &frame.active_header,
                                &frame.frequency_tables,
                                control,
                                raw,
                                dequantized,
                                harmonics,
                                2,
                            )?;
                        self.apply_legacy_drc_to_qmf_slots(
                            processor_channel,
                            channel,
                            &mut slots,
                            true,
                        );
                        channels[channel] = self.ordinary_sbr_fixed_processors[processor_channel]
                            .synthesize_qmf(&slots)?
                            .into_iter()
                            .map(|sample| f32_to_i16(sample as f32))
                            .collect();
                        channel += 1;
                        processor_channel += 1;
                    }
                    parsed_frames.push(OrdinarySbrFrame::Stereo(frame));
                }
                _ => return Err(DecodeError::SbrPayloadLayoutMismatch),
            }
        }
        while channel < channels.len() {
            let core = channels[channel]
                .iter()
                .map(|&sample| sample as f64 / 32768.0)
                .collect::<Vec<_>>();
            channels[channel] = self.ordinary_sbr_fixed_processors[processor_channel]
                .upsample_only(&core)?
                .into_iter()
                .map(|sample| f32_to_i16(sample as f32))
                .collect();
            channel += 1;
            processor_channel += 1;
        }
        self.last_ordinary_sbr_fixed_frames = parsed_frames;
        Ok(())
    }

    fn conceal_ordinary_sbr_f32(
        &mut self,
        channels: &mut Vec<Vec<f32>>,
    ) -> Result<(), DecodeError> {
        let frames = self.last_ordinary_sbr_frames.clone();
        let mut channel = 0usize;
        let mut processor_channel = 0usize;
        for (element, frame) in frames.into_iter().enumerate() {
            match frame {
                OrdinarySbrFrame::Mono(frame) => {
                    let core = channels[channel]
                        .iter()
                        .map(|&v| v as f64)
                        .collect::<Vec<_>>();
                    if let Some(ps_frame) = self.last_ps_frames[element]
                        .clone()
                        .filter(|_| !self.qmf_low_power)
                    {
                        let mut slots = self.ordinary_sbr_processors[processor_channel]
                            .process_channel_to_qmf(
                                &core,
                                &frame.active_header,
                                &frame.frequency_tables,
                                &frame.control,
                                &frame.values,
                                &frame.dequantized,
                                &frame.harmonics,
                                2,
                            )?;
                        self.apply_legacy_drc_to_qmf_slots(
                            processor_channel,
                            channel,
                            &mut slots,
                            false,
                        );
                        let (left, right) =
                            self.ps_processors[element].process_qmf(&slots, &ps_frame)?;
                        channels[channel] = left.into_iter().map(|v| v as f32).collect();
                        channels.insert(channel + 1, right.into_iter().map(|v| v as f32).collect());
                        channel += 2;
                    } else {
                        let mut slots = self.ordinary_sbr_processors[processor_channel]
                            .process_channel_to_qmf(
                                &core,
                                &frame.active_header,
                                &frame.frequency_tables,
                                &frame.control,
                                &frame.values,
                                &frame.dequantized,
                                &frame.harmonics,
                                2,
                            )?;
                        self.apply_legacy_drc_to_qmf_slots(
                            processor_channel,
                            channel,
                            &mut slots,
                            false,
                        );
                        channels[channel] = self.ordinary_sbr_processors[processor_channel]
                            .synthesize_qmf(&slots)?
                            .into_iter()
                            .map(|v| v as f32)
                            .collect();
                        channel += 1;
                    }
                    processor_channel += 1;
                }
                OrdinarySbrFrame::Stereo(frame) => {
                    for right in [false, true] {
                        let (control, raw, values, harmonics) = if right {
                            (
                                &frame.right_control,
                                &frame.right,
                                &frame.right_dequantized,
                                &frame.right_harmonics,
                            )
                        } else {
                            (
                                &frame.left_control,
                                &frame.left,
                                &frame.left_dequantized,
                                &frame.left_harmonics,
                            )
                        };
                        let core = channels[channel]
                            .iter()
                            .map(|&v| v as f64)
                            .collect::<Vec<_>>();
                        let mut slots = self.ordinary_sbr_processors[processor_channel]
                            .process_channel_to_qmf(
                                &core,
                                &frame.active_header,
                                &frame.frequency_tables,
                                control,
                                raw,
                                values,
                                harmonics,
                                2,
                            )?;
                        self.apply_legacy_drc_to_qmf_slots(
                            processor_channel,
                            channel,
                            &mut slots,
                            false,
                        );
                        channels[channel] = self.ordinary_sbr_processors[processor_channel]
                            .synthesize_qmf(&slots)?
                            .into_iter()
                            .map(|v| v as f32)
                            .collect();
                        channel += 1;
                        processor_channel += 1;
                    }
                }
            }
        }
        while channel < channels.len() {
            let core = channels[channel]
                .iter()
                .map(|&sample| sample as f64)
                .collect::<Vec<_>>();
            channels[channel] = self.ordinary_sbr_processors[processor_channel]
                .upsample_only(&core)?
                .into_iter()
                .map(|sample| sample as f32)
                .collect();
            channel += 1;
            processor_channel += 1;
        }
        Ok(())
    }

    fn conceal_ordinary_sbr_fixed(
        &mut self,
        channels: &mut Vec<Vec<i16>>,
    ) -> Result<(), DecodeError> {
        let frames = self.last_ordinary_sbr_fixed_frames.clone();
        let mut channel = 0usize;
        let mut processor_channel = 0usize;
        for (element, frame) in frames.into_iter().enumerate() {
            match frame {
                OrdinarySbrFrame::Mono(frame) => {
                    let core = channels[channel]
                        .iter()
                        .map(|&v| v as f64 / 32768.0)
                        .collect::<Vec<_>>();
                    if let Some(ps_frame) = self.last_ps_fixed_frames[element]
                        .clone()
                        .filter(|_| !self.qmf_low_power)
                    {
                        let mut slots = self.ordinary_sbr_fixed_processors[processor_channel]
                            .process_channel_to_qmf(
                                &core,
                                &frame.active_header,
                                &frame.frequency_tables,
                                &frame.control,
                                &frame.values,
                                &frame.dequantized,
                                &frame.harmonics,
                                2,
                            )?;
                        self.apply_legacy_drc_to_qmf_slots(
                            processor_channel,
                            channel,
                            &mut slots,
                            true,
                        );
                        let (left, right) =
                            self.ps_fixed_processors[element].process_qmf(&slots, &ps_frame)?;
                        channels[channel] =
                            left.into_iter().map(|v| f32_to_i16(v as f32)).collect();
                        channels.insert(
                            channel + 1,
                            right.into_iter().map(|v| f32_to_i16(v as f32)).collect(),
                        );
                        channel += 2;
                    } else {
                        let mut slots = self.ordinary_sbr_fixed_processors[processor_channel]
                            .process_channel_to_qmf(
                                &core,
                                &frame.active_header,
                                &frame.frequency_tables,
                                &frame.control,
                                &frame.values,
                                &frame.dequantized,
                                &frame.harmonics,
                                2,
                            )?;
                        self.apply_legacy_drc_to_qmf_slots(
                            processor_channel,
                            channel,
                            &mut slots,
                            true,
                        );
                        channels[channel] = self.ordinary_sbr_fixed_processors[processor_channel]
                            .synthesize_qmf(&slots)?
                            .into_iter()
                            .map(|v| f32_to_i16(v as f32))
                            .collect();
                        channel += 1;
                    }
                    processor_channel += 1;
                }
                OrdinarySbrFrame::Stereo(frame) => {
                    for right in [false, true] {
                        let (control, raw, values, harmonics) = if right {
                            (
                                &frame.right_control,
                                &frame.right,
                                &frame.right_dequantized,
                                &frame.right_harmonics,
                            )
                        } else {
                            (
                                &frame.left_control,
                                &frame.left,
                                &frame.left_dequantized,
                                &frame.left_harmonics,
                            )
                        };
                        let core = channels[channel]
                            .iter()
                            .map(|&v| v as f64 / 32768.0)
                            .collect::<Vec<_>>();
                        let mut slots = self.ordinary_sbr_fixed_processors[processor_channel]
                            .process_channel_to_qmf(
                                &core,
                                &frame.active_header,
                                &frame.frequency_tables,
                                control,
                                raw,
                                values,
                                harmonics,
                                2,
                            )?;
                        self.apply_legacy_drc_to_qmf_slots(
                            processor_channel,
                            channel,
                            &mut slots,
                            true,
                        );
                        channels[channel] = self.ordinary_sbr_fixed_processors[processor_channel]
                            .synthesize_qmf(&slots)?
                            .into_iter()
                            .map(|v| f32_to_i16(v as f32))
                            .collect();
                        channel += 1;
                        processor_channel += 1;
                    }
                }
            }
        }
        while channel < channels.len() {
            let core = channels[channel]
                .iter()
                .map(|&sample| sample as f64 / 32768.0)
                .collect::<Vec<_>>();
            channels[channel] = self.ordinary_sbr_fixed_processors[processor_channel]
                .upsample_only(&core)?
                .into_iter()
                .map(|sample| f32_to_i16(sample as f32))
                .collect();
            channel += 1;
            processor_channel += 1;
        }
        Ok(())
    }

    fn ensure_coupling_filterbanks(&mut self, channels: usize) -> Result<(), DecodeError> {
        while self.coupling_filterbanks.len() < channels {
            self.coupling_filterbanks
                .push(LongBlockFilterbank::new(self.frame_length)?);
        }
        Ok(())
    }

    fn ensure_fixed_channel_filterbanks(&mut self, channels: usize) -> Result<(), DecodeError> {
        while self.fixed_channel_filterbanks.len() < channels {
            self.fixed_channel_filterbanks
                .push(FixedLongBlockFilterbank::new(self.frame_length)?);
        }
        if self.audio_object_type == 39 {
            while self.eld_fixed_channel_filterbanks.len() < channels {
                self.eld_fixed_channel_filterbanks
                    .push(LowDelayFilterbankQ31::new(self.frame_length)?);
            }
        }
        Ok(())
    }

    fn ensure_fixed_coupling_filterbanks(&mut self, channels: usize) -> Result<(), DecodeError> {
        while self.fixed_coupling_filterbanks.len() < channels {
            self.fixed_coupling_filterbanks
                .push(FixedLongBlockFilterbank::new(self.frame_length)?);
        }
        Ok(())
    }

    pub fn synthesize_channel_stream_fixed_q31(
        &mut self,
        stream: &DecodedChannelStream,
        channel_index: usize,
    ) -> Result<Vec<FixpDbl>, DecodeError> {
        self.ensure_fixed_channel_filterbanks(channel_index + 1)?;
        Ok(synthesize_aac_lc_frame_from_inverse_q31(
            &stream.spectrum,
            &stream.ics,
            &mut self.fixed_channel_filterbanks[channel_index],
        )?)
    }

    pub fn synthesize_channel_stream_fixed_i16(
        &mut self,
        stream: &DecodedChannelStream,
        channel_index: usize,
    ) -> Result<Vec<i16>, DecodeError> {
        Ok(self
            .synthesize_channel_stream_fixed_q31(stream, channel_index)?
            .into_iter()
            .map(dbl_to_pcm16)
            .collect())
    }

    pub fn synthesize_coupling_channel_stream_fixed_q31(
        &mut self,
        stream: &DecodedChannelStream,
        coupling_index: usize,
    ) -> Result<Vec<FixpDbl>, DecodeError> {
        self.ensure_fixed_coupling_filterbanks(coupling_index + 1)?;
        Ok(synthesize_aac_lc_frame_from_inverse_q31(
            &stream.spectrum,
            &stream.ics,
            &mut self.fixed_coupling_filterbanks[coupling_index],
        )?)
    }

    pub fn synthesize_coupling_channel_stream_fixed_i16(
        &mut self,
        stream: &DecodedChannelStream,
        coupling_index: usize,
    ) -> Result<Vec<i16>, DecodeError> {
        Ok(self
            .synthesize_coupling_channel_stream_fixed_q31(stream, coupling_index)?
            .into_iter()
            .map(dbl_to_pcm16)
            .collect())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodedAacLcMultichannelFrame {
    pub channels: Vec<Vec<f32>>,
    pub labels: Vec<ChannelLabel>,
}

impl DecodedAacLcMultichannelFrame {
    pub fn channels(&self) -> usize {
        self.channels.len()
    }

    pub fn labels(&self) -> &[ChannelLabel] {
        &self.labels
    }

    pub fn samples_per_channel(&self) -> usize {
        self.channels.iter().map(Vec::len).min().unwrap_or(0)
    }

    pub fn interleaved_f32(&self) -> Vec<f32> {
        interleave_multichannel_f32(&self.channels)
    }

    pub fn interleaved_i16(&self) -> Vec<i16> {
        interleave_multichannel_i16(&self.channels)
    }
}

pub struct DecodedAdtsStreamF32<'a> {
    decoder: &'a mut AacLcDecoder,
    frames: AdtsStream<'a>,
    strict: bool,
}

impl Iterator for DecodedAdtsStreamF32<'_> {
    type Item = Result<Vec<f32>, DecodeError>;

    fn next(&mut self) -> Option<Self::Item> {
        let frame = match self.frames.next()? {
            Ok(frame) => frame,
            Err(err) => return Some(Err(err.into())),
        };
        let decoded = if self.strict {
            self.decoder.decode_adts_frame_f32_strict(frame.bytes)
        } else {
            self.decoder.decode_adts_frame_f32(frame.bytes)
        };
        Some(decoded.map(|frame| frame.interleaved_f32()))
    }
}

pub struct DecodedAdtsStreamI16<'a> {
    decoder: &'a mut AacLcDecoder,
    frames: AdtsStream<'a>,
    strict: bool,
}

impl Iterator for DecodedAdtsStreamI16<'_> {
    type Item = Result<Vec<i16>, DecodeError>;

    fn next(&mut self) -> Option<Self::Item> {
        let frame = match self.frames.next()? {
            Ok(frame) => frame,
            Err(err) => return Some(Err(err.into())),
        };
        let decoded = if self.strict {
            self.decoder.decode_adts_frame_f32_strict(frame.bytes)
        } else {
            self.decoder.decode_adts_frame_f32(frame.bytes)
        };
        Some(decoded.map(|frame| frame.interleaved_i16()))
    }
}

pub struct DecodedAdtsStreamFixedI16<'a> {
    decoder: &'a mut AacLcDecoder,
    frames: AdtsStream<'a>,
    strict: bool,
}

impl Iterator for DecodedAdtsStreamFixedI16<'_> {
    type Item = Result<Vec<i16>, DecodeError>;

    fn next(&mut self) -> Option<Self::Item> {
        let frame = match self.frames.next()? {
            Ok(frame) => frame,
            Err(err) => return Some(Err(err.into())),
        };
        Some(if self.strict {
            self.decoder
                .decode_adts_frame_fixed_interleaved_i16_strict(frame.bytes)
        } else {
            self.decoder
                .decode_adts_frame_fixed_interleaved_i16(frame.bytes)
        })
    }
}

pub struct DecodedAdtsMultichannelStreamF32<'a> {
    decoder: &'a mut AacLcDecoder,
    frames: AdtsStream<'a>,
    strict: bool,
}

impl Iterator for DecodedAdtsMultichannelStreamF32<'_> {
    type Item = Result<DecodedAacLcMultichannelFrame, DecodeError>;

    fn next(&mut self) -> Option<Self::Item> {
        let frame = match self.frames.next()? {
            Ok(frame) => frame,
            Err(err) => return Some(Err(err.into())),
        };
        Some(if self.strict {
            self.decoder
                .decode_adts_frame_multichannel_f32_strict(frame.bytes)
        } else {
            self.decoder.decode_adts_frame_multichannel_f32(frame.bytes)
        })
    }
}

pub struct DecodedAdtsMultichannelInterleavedStreamF32<'a> {
    decoder: &'a mut AacLcDecoder,
    frames: AdtsStream<'a>,
    strict: bool,
}

impl Iterator for DecodedAdtsMultichannelInterleavedStreamF32<'_> {
    type Item = Result<Vec<f32>, DecodeError>;

    fn next(&mut self) -> Option<Self::Item> {
        let frame = match self.frames.next()? {
            Ok(frame) => frame,
            Err(err) => return Some(Err(err.into())),
        };
        let decoded = if self.strict {
            self.decoder
                .decode_adts_frame_multichannel_f32_strict(frame.bytes)
        } else {
            self.decoder.decode_adts_frame_multichannel_f32(frame.bytes)
        };
        Some(decoded.map(|frame| frame.interleaved_f32()))
    }
}

pub struct DecodedAdtsMultichannelInterleavedStreamI16<'a> {
    decoder: &'a mut AacLcDecoder,
    frames: AdtsStream<'a>,
    strict: bool,
}

impl Iterator for DecodedAdtsMultichannelInterleavedStreamI16<'_> {
    type Item = Result<Vec<i16>, DecodeError>;

    fn next(&mut self) -> Option<Self::Item> {
        let frame = match self.frames.next()? {
            Ok(frame) => frame,
            Err(err) => return Some(Err(err.into())),
        };
        let decoded = if self.strict {
            self.decoder
                .decode_adts_frame_multichannel_f32_strict(frame.bytes)
        } else {
            self.decoder.decode_adts_frame_multichannel_f32(frame.bytes)
        };
        Some(decoded.map(|frame| frame.interleaved_i16()))
    }
}

pub struct DecodedAdtsMultichannelFixedInterleavedStreamI16<'a> {
    decoder: &'a mut AacLcDecoder,
    frames: AdtsStream<'a>,
    strict: bool,
}

impl Iterator for DecodedAdtsMultichannelFixedInterleavedStreamI16<'_> {
    type Item = Result<Vec<i16>, DecodeError>;

    fn next(&mut self) -> Option<Self::Item> {
        let frame = match self.frames.next()? {
            Ok(frame) => frame,
            Err(err) => return Some(Err(err.into())),
        };
        Some(if self.strict {
            self.decoder
                .decode_adts_frame_multichannel_fixed_interleaved_i16_strict(frame.bytes)
        } else {
            self.decoder
                .decode_adts_frame_multichannel_fixed_interleaved_i16(frame.bytes)
        })
    }
}

impl DecodedSingleChannelFrame {
    pub fn channels(&self) -> usize {
        1
    }

    pub fn samples_per_channel(&self) -> usize {
        self.samples.len()
    }

    pub fn interleaved_f32(&self) -> Vec<f32> {
        self.samples.clone()
    }

    pub fn interleaved_i16(&self) -> Vec<i16> {
        self.samples.iter().copied().map(f32_to_i16).collect()
    }
}

impl DecodedChannelPairFrame {
    pub fn channels(&self) -> usize {
        2
    }

    pub fn samples_per_channel(&self) -> usize {
        self.left_samples.len().min(self.right_samples.len())
    }

    pub fn interleaved_f32(&self) -> Vec<f32> {
        interleave_stereo_f32(&self.left_samples, &self.right_samples)
    }

    pub fn interleaved_i16(&self) -> Vec<i16> {
        interleave_stereo_i16(&self.left_samples, &self.right_samples)
    }
}

impl DecodedAacLcFrame {
    pub fn channels(&self) -> usize {
        match self {
            Self::Mono(frame) => frame.channels(),
            Self::Stereo(frame) => frame.channels(),
        }
    }

    pub fn samples_per_channel(&self) -> usize {
        match self {
            Self::Mono(frame) => frame.samples_per_channel(),
            Self::Stereo(frame) => frame.samples_per_channel(),
        }
    }

    pub fn interleaved_f32(&self) -> Vec<f32> {
        match self {
            Self::Mono(frame) => frame.interleaved_f32(),
            Self::Stereo(frame) => frame.interleaved_f32(),
        }
    }

    pub fn interleaved_i16(&self) -> Vec<i16> {
        match self {
            Self::Mono(frame) => frame.interleaved_i16(),
            Self::Stereo(frame) => frame.interleaved_i16(),
        }
    }
}

pub fn interleave_stereo_f32(left: &[f32], right: &[f32]) -> Vec<f32> {
    let frames = left.len().min(right.len());
    let mut out = Vec::with_capacity(frames * 2);
    for index in 0..frames {
        out.push(left[index]);
        out.push(right[index]);
    }
    out
}

pub fn interleave_stereo_i16(left: &[f32], right: &[f32]) -> Vec<i16> {
    let frames = left.len().min(right.len());
    let mut out = Vec::with_capacity(frames * 2);
    for index in 0..frames {
        out.push(f32_to_i16(left[index]));
        out.push(f32_to_i16(right[index]));
    }
    out
}

pub fn interleave_stereo_i16_samples(left: &[i16], right: &[i16]) -> Vec<i16> {
    let frames = left.len().min(right.len());
    let mut out = Vec::with_capacity(frames * 2);
    for index in 0..frames {
        out.push(left[index]);
        out.push(right[index]);
    }
    out
}

pub fn interleave_multichannel_f32(channels: &[Vec<f32>]) -> Vec<f32> {
    let frames = channels.iter().map(Vec::len).min().unwrap_or(0);
    let mut out = Vec::with_capacity(frames * channels.len());
    for frame in 0..frames {
        for channel in channels {
            out.push(channel[frame]);
        }
    }
    out
}

pub fn interleave_multichannel_i16(channels: &[Vec<f32>]) -> Vec<i16> {
    let frames = channels.iter().map(Vec::len).min().unwrap_or(0);
    let mut out = Vec::with_capacity(frames * channels.len());
    for frame in 0..frames {
        for channel in channels {
            out.push(f32_to_i16(channel[frame]));
        }
    }
    out
}

pub fn interleave_multichannel_i16_samples(channels: &[Vec<i16>]) -> Vec<i16> {
    let frames = channels.iter().map(Vec::len).min().unwrap_or(0);
    let mut out = Vec::with_capacity(frames * channels.len());
    for frame in 0..frames {
        for channel in channels {
            out.push(channel[frame]);
        }
    }
    out
}

pub fn f32_to_i16(sample: f32) -> i16 {
    if !sample.is_finite() {
        return 0;
    }
    let scaled = (sample.clamp(-1.0, 1.0) * 32768.0).round();
    if scaled >= i16::MAX as f32 {
        i16::MAX
    } else if scaled <= i16::MIN as f32 {
        i16::MIN
    } else {
        scaled as i16
    }
}

fn eld_raw_pcm_to_i16(sample: f32) -> i16 {
    if !sample.is_finite() {
        return 0;
    }
    (sample * 8.0)
        .round()
        .clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

fn skip_fill_element(reader: &mut BitReader<'_>) -> Result<(), DecodeError> {
    let mut count = reader.read_u8(4)? as usize;
    if count == 15 {
        count += reader.read_u8(8)? as usize - 1;
    }
    skip_bytes(reader, count)
}

fn fill_extension_type(reader: &BitReader<'_>) -> Result<Option<u8>, BitError> {
    let mut probe = reader.clone();
    let mut count = probe.read_u8(4)? as usize;
    if count == 15 {
        count += probe.read_u8(8)? as usize;
        count = count.saturating_sub(1);
    }
    if count == 0 {
        return Ok(None);
    }
    Ok(Some(probe.read_u8(4)?))
}

fn apply_legacy_band_gains_f32(
    spectrum: &mut InverseQuantizedSpectrum,
    band_top: &[u8],
    gains: &[f32],
) {
    let mut position = 0usize;
    let mut band = 0usize;
    for coefficient in spectrum.windows.iter_mut().flatten() {
        while band < band_top.len() && position >= (usize::from(band_top[band]) + 1) * 4 {
            band += 1;
        }
        if let Some(&gain) = gains.get(band) {
            *coefficient *= gain;
        }
        position += 1;
    }
}

fn apply_legacy_band_gains_fixed(
    spectrum: &mut FixedInverseQuantizedSpectrum,
    band_top: &[u8],
    gains: &[f32],
) {
    let coefficient_count = spectrum.windows.iter().map(Vec::len).sum::<usize>();
    let covered = band_top
        .last()
        .is_some_and(|top| (usize::from(*top) + 1) * 4 >= coefficient_count);
    let maximum_gain = gains
        .iter()
        .copied()
        .fold(if covered { 0.0f32 } else { 1.0f32 }, f32::max)
        .max(1.0);
    let exponent_shift = maximum_gain.log2().ceil().max(0.0) as i16;
    let mantissa_scale = 2.0f64.powi(-i32::from(exponent_shift));
    let mut position = 0usize;
    let mut band = 0usize;
    for coefficient in spectrum.windows.iter_mut().flatten() {
        while band < band_top.len() && position >= (usize::from(band_top[band]) + 1) * 4 {
            band += 1;
        }
        let gain = gains.get(band).copied().unwrap_or(1.0) as f64 * mantissa_scale;
        *coefficient = (f64::from(*coefficient) * gain)
            .round()
            .clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32;
        position += 1;
    }
    for exponent in &mut spectrum.window_exponents {
        *exponent = exponent.saturating_add(exponent_shift);
    }
}

fn apply_legacy_qmf_drc(
    state: &mut LegacyQmfDrcState,
    slots: &mut [QmfSlot],
    next: Option<LegacyQmfDrcFrame>,
    frame_length: usize,
) -> bool {
    if next.is_none() && !state.enabled {
        return false;
    }
    let next = next.unwrap_or_else(LegacyQmfDrcFrame::unity);
    let next_non_unity = next.gains.iter().any(|gain| (*gain - 1.0).abs() > 1.0e-12);
    state.enabled |= next_non_unity;
    if !state.enabled || slots.is_empty() {
        state.current = next;
        return false;
    }

    let slot_count = slots.len();
    let border_map: &[usize; 16] = if slot_count == 30 {
        &[0, 0, 4, 8, 11, 15, 19, 23, 26, 30, 30, 30, 30, 30, 30, 30]
    } else {
        &[0, 0, 4, 8, 12, 16, 20, 24, 28, 32, 32, 32, 32, 32, 32, 32]
    };
    let offset = slot_count.saturating_sub(slot_count / 2).saturating_sub(10);
    for (slot_index, slot) in slots.iter_mut().enumerate() {
        let original_column = slot_index + offset;
        let (frame, column, alpha) = if original_column < slot_count / 2 {
            let frame = state.current.clone();
            let j = original_column + slot_count / 2;
            let alpha = legacy_qmf_interpolation(&frame, j, border_map, slot_count);
            (frame, original_column, alpha)
        } else if original_column < slot_count {
            let j = original_column - slot_count / 2;
            let frame = next.clone();
            let alpha = legacy_qmf_interpolation(&frame, j, border_map, slot_count);
            (frame, original_column, alpha)
        } else {
            let column = original_column - slot_count;
            let j = column + slot_count / 2;
            let frame = next.clone();
            let alpha = legacy_qmf_interpolation(&frame, j, border_map, slot_count);
            (frame, column, alpha)
        };

        if frame.window_sequence == WindowSequence::EightShort {
            apply_legacy_qmf_short_slot(
                state,
                slot,
                &frame,
                column,
                slot_count,
                frame_length,
                border_map,
            );
        } else {
            let ranges = legacy_qmf_long_band_ranges(&frame, slot_count, frame_length);
            for (band, (bottom, top)) in ranges.into_iter().enumerate() {
                let next_gain = frame.gain_for_band(band);
                let top = top.min(slot.real.len());
                for bin in bottom.min(top)..top {
                    let previous = state.previous_gains.get(bin).copied().unwrap_or(1.0);
                    let gain = previous + (next_gain - previous) * alpha;
                    slot.real[bin] *= gain;
                    if let Some(value) = slot.imaginary.get_mut(bin) {
                        *value *= gain;
                    }
                    if original_column == slot_count / 2 - 1 {
                        state.previous_gains[bin] = next_gain;
                    }
                }
            }
        }
    }
    let current_non_unity = state
        .current
        .gains
        .iter()
        .any(|gain| (*gain - 1.0).abs() > 1.0e-12);
    state.current = next;
    state.enabled = next_non_unity || current_non_unity;
    true
}

fn legacy_qmf_interpolation(
    frame: &LegacyQmfDrcFrame,
    position: usize,
    border_map: &[usize; 16],
    slot_count: usize,
) -> f64 {
    if position >= border_map[15] {
        return 1.0;
    }
    if frame.interpolation_scheme == 0 {
        position as f64 / slot_count as f64
    } else if position
        >= border_map[usize::from(frame.interpolation_scheme).min(border_map.len() - 1)]
    {
        1.0
    } else {
        0.0
    }
}

fn legacy_qmf_long_band_ranges(
    frame: &LegacyQmfDrcFrame,
    slot_count: usize,
    frame_length: usize,
) -> Vec<(usize, usize)> {
    let mut bottom = 0usize;
    frame
        .band_top
        .iter()
        .enumerate()
        .map(|(band, &top)| {
            let top_mdct = (usize::from(top) + 1) * 4;
            let mut top_qmf = if slot_count == 30 || frame_length == 960 {
                top_mdct / 30
            } else {
                (top_mdct & !31) / 32
            };
            if band + 1 == frame.band_top.len() {
                top_qmf = 64;
            }
            let range = (bottom, top_qmf);
            bottom = top_qmf;
            range
        })
        .collect()
}

fn apply_legacy_qmf_short_slot(
    state: &mut LegacyQmfDrcState,
    slot: &mut QmfSlot,
    frame: &LegacyQmfDrcFrame,
    column: usize,
    slot_count: usize,
    frame_length: usize,
    border_map: &[usize; 16],
) {
    let frame_size = if slot_count == 30 { 960 } else { frame_length };
    let short_length = frame_size / 8;
    let mut bottom_mdct = 0usize;
    for (band, &band_top) in frame.band_top.iter().enumerate() {
        let mut top_mdct = ((usize::from(band_top) + 1) * 4).min(frame_size.saturating_sub(1));
        if slot_count != 30 {
            top_mdct &= !3;
        }
        let start_window = (bottom_mdct / short_length + 1).min(14);
        let stop_window = (top_mdct.div_ceil(short_length) + 1).clamp(1, 15);
        let start_column = border_map[start_window];
        let mut stop_column = border_map[stop_window];
        let period = slot_count * 4;
        let mut bottom_qmf = (bottom_mdct % period) * 32 / short_length;
        let mut top_qmf = (top_mdct % period) * 32 / short_length;
        if band + 1 == frame.band_top.len() {
            top_qmf = 64;
            stop_column = slot_count;
        } else if top_qmf == 0 {
            top_qmf = 64;
        }
        if stop_column == slot_count {
            let save_bottom = if border_map[8] > start_column {
                0
            } else {
                bottom_qmf
            };
            for bin in save_bottom..top_qmf.min(state.previous_gains.len()) {
                state.previous_gains[bin] = frame.gain_for_band(band);
            }
        }
        if column >= start_column && column < stop_column {
            if start_window + 1 < border_map.len() && column >= border_map[start_window + 1] {
                bottom_qmf = 0;
            }
            if stop_window > 0 && column < border_map[stop_window - 1] {
                top_qmf = 64;
            }
            let gain = frame.gain_for_band(band);
            let top = top_qmf.min(slot.real.len());
            for bin in bottom_qmf.min(top)..top {
                slot.real[bin] *= gain;
                if let Some(value) = slot.imaginary.get_mut(bin) {
                    *value *= gain;
                }
            }
        }
        bottom_mdct = top_mdct;
    }
}

fn skip_bytes(reader: &mut BitReader<'_>, bytes: usize) -> Result<(), DecodeError> {
    for _ in 0..bytes {
        reader.read_u8(8)?;
    }
    Ok(())
}

pub fn expected_channels_for_config(channel_configuration: u8) -> Option<usize> {
    match channel_configuration {
        0 => None,
        1 => Some(1),
        2 => Some(2),
        3 => Some(3),
        4 => Some(4),
        5 => Some(5),
        6 => Some(6),
        7 => Some(8),
        11 => Some(7),
        12 | 14 => Some(8),
        _ => None,
    }
}

pub fn channel_labels_for_config(channel_configuration: u8) -> Option<&'static [ChannelLabel]> {
    match channel_configuration {
        0 => None,
        1 => Some(&[ChannelLabel::FrontCenter]),
        2 => Some(&[ChannelLabel::FrontLeft, ChannelLabel::FrontRight]),
        3 => Some(&[
            ChannelLabel::FrontCenter,
            ChannelLabel::FrontLeft,
            ChannelLabel::FrontRight,
        ]),
        4 => Some(&[
            ChannelLabel::FrontCenter,
            ChannelLabel::FrontLeft,
            ChannelLabel::FrontRight,
            ChannelLabel::BackCenter,
        ]),
        5 => Some(&[
            ChannelLabel::FrontCenter,
            ChannelLabel::FrontLeft,
            ChannelLabel::FrontRight,
            ChannelLabel::BackLeft,
            ChannelLabel::BackRight,
        ]),
        6 => Some(&[
            ChannelLabel::FrontCenter,
            ChannelLabel::FrontLeft,
            ChannelLabel::FrontRight,
            ChannelLabel::BackLeft,
            ChannelLabel::BackRight,
            ChannelLabel::Lfe,
        ]),
        7 => Some(&[
            ChannelLabel::FrontCenter,
            ChannelLabel::FrontLeftCenter,
            ChannelLabel::FrontRightCenter,
            ChannelLabel::FrontLeft,
            ChannelLabel::FrontRight,
            ChannelLabel::BackLeft,
            ChannelLabel::BackRight,
            ChannelLabel::Lfe,
        ]),
        11 => Some(&[
            ChannelLabel::FrontCenter,
            ChannelLabel::FrontLeft,
            ChannelLabel::FrontRight,
            ChannelLabel::BackLeft,
            ChannelLabel::BackRight,
            ChannelLabel::BackCenter,
            ChannelLabel::Lfe,
        ]),
        12 => Some(&[
            ChannelLabel::FrontCenter,
            ChannelLabel::FrontLeft,
            ChannelLabel::FrontRight,
            ChannelLabel::SideLeft,
            ChannelLabel::SideRight,
            ChannelLabel::BackLeft,
            ChannelLabel::BackRight,
            ChannelLabel::Lfe,
        ]),
        14 => Some(&[
            ChannelLabel::FrontCenter,
            ChannelLabel::FrontLeft,
            ChannelLabel::FrontRight,
            ChannelLabel::BackLeft,
            ChannelLabel::BackRight,
            ChannelLabel::Lfe,
            ChannelLabel::Unknown(6),
            ChannelLabel::Unknown(7),
        ]),
        _ => None,
    }
}

fn channel_indices_for_labels(labels: &[ChannelLabel]) -> Vec<u8> {
    labels
        .iter()
        .map(|label| match label {
            ChannelLabel::Empty => 0,
            ChannelLabel::FrontCenter => 0,
            ChannelLabel::FrontLeft => 1,
            ChannelLabel::FrontRight => 2,
            ChannelLabel::SideLeft
            | ChannelLabel::BackLeft
            | ChannelLabel::FrontLeftCenter
            | ChannelLabel::Lfe => 0,
            ChannelLabel::SideRight | ChannelLabel::BackRight | ChannelLabel::FrontRightCenter => 1,
            ChannelLabel::BackCenter => 2,
            ChannelLabel::Unknown(index) => u8::try_from(*index).unwrap_or(u8::MAX),
        })
        .collect()
}

fn unknown_channel_labels(channels: usize) -> Vec<ChannelLabel> {
    (0..channels).map(ChannelLabel::Unknown).collect()
}

fn er_channel_elements(channel_configuration: u8) -> Option<&'static [ElementId]> {
    Some(match channel_configuration {
        1 => &[ElementId::SingleChannel],
        2 => &[ElementId::ChannelPair],
        3 => &[ElementId::SingleChannel, ElementId::ChannelPair],
        4 => &[
            ElementId::SingleChannel,
            ElementId::ChannelPair,
            ElementId::SingleChannel,
        ],
        5 => &[
            ElementId::SingleChannel,
            ElementId::ChannelPair,
            ElementId::ChannelPair,
        ],
        6 => &[
            ElementId::SingleChannel,
            ElementId::ChannelPair,
            ElementId::ChannelPair,
            ElementId::Lfe,
        ],
        7 => &[
            ElementId::SingleChannel,
            ElementId::ChannelPair,
            ElementId::ChannelPair,
            ElementId::ChannelPair,
            ElementId::Lfe,
        ],
        11 => &[
            ElementId::SingleChannel,
            ElementId::ChannelPair,
            ElementId::ChannelPair,
            ElementId::SingleChannel,
            ElementId::Lfe,
        ],
        12 => &[
            ElementId::SingleChannel,
            ElementId::ChannelPair,
            ElementId::ChannelPair,
            ElementId::ChannelPair,
            ElementId::Lfe,
        ],
        14 => &[
            ElementId::SingleChannel,
            ElementId::ChannelPair,
            ElementId::ChannelPair,
            ElementId::Lfe,
            ElementId::ChannelPair,
        ],
        _ => return None,
    })
}

fn er_sbr_channel_indices(elements: &[ElementId]) -> Vec<usize> {
    let mut indices = Vec::new();
    let mut channel = 0usize;
    for &element in elements {
        match element {
            ElementId::ChannelPair => {
                indices.extend([channel, channel + 1]);
                channel += 2;
            }
            ElementId::Lfe => channel += 1,
            _ => {
                indices.push(channel);
                channel += 1;
            }
        }
    }
    indices
}

fn parse_er_ics(
    reader: &mut BitReader<'_>,
    sampling_frequency_index: u8,
    frame_length: usize,
    eld_enabled: bool,
) -> Result<IcsInfo, DecodeError> {
    if eld_enabled {
        let total_sfb = er_long_sfb_count(sampling_frequency_index, frame_length)?;
        Ok(IcsInfo::parse_eld(reader, total_sfb)?)
    } else if frame_length <= 512 {
        let total_sfb = er_long_sfb_count(sampling_frequency_index, frame_length)?;
        Ok(IcsInfo::parse_aac_ld(reader, total_sfb)?)
    } else {
        Ok(IcsInfo::parse_aac_lc(reader, IcsLimits::AAC_LC_MAX)?)
    }
}

fn decode_er_channel_stream_from_reader(
    reader: &mut BitReader<'_>,
    sampling_frequency_index: u8,
    frame_length: usize,
    eld_enabled: bool,
    shared_ics: Option<&IcsInfo>,
    vcb11_enabled: bool,
    rvlc_enabled: bool,
    hcr_enabled: bool,
    hcr_element_type: HcrElementType,
) -> Result<DecodedChannelStream, DecodeError> {
    let global_gain = reader.read_u8(8)?;
    let ics = match shared_ics {
        Some(ics) => ics.clone(),
        None => parse_er_ics(reader, sampling_frequency_index, frame_length, eld_enabled)?,
    };
    if frame_length <= 512 && !ics.window_sequence.is_long() {
        return Err(DecodeError::UnsupportedFrameLength(frame_length));
    }
    let sfb = aac_band_offsets_for_ics(sampling_frequency_index, &ics, frame_length)?;
    let section_data = SectionData::parse_aac_lc_with_vcb11(reader, &ics, vcb11_enabled)?;
    let scalefactor_plan = ScalefactorPlan::from_section_data(&section_data)?;
    let rvlc_side = rvlc_enabled
        .then(|| RvlcSideInfo::parse(reader, &ics, &section_data))
        .transpose()?;
    let mut scalefactors = if rvlc_enabled {
        None
    } else {
        Some(scalefactor_plan.decode_from_bitstream(reader, global_gain)?)
    };
    let pulse_data = if eld_enabled {
        PulseData::absent()
    } else {
        PulseData::parse_aac_lc(reader, &ics, sfb.offsets, sfb.granule_length)?
    };
    let tns_present = reader.read_bool()?;
    if !eld_enabled && reader.read_bool()? {
        return Err(DecodeError::GainControlUnsupported);
    }
    let eld_tns_data = if eld_enabled && tns_present {
        Some(TnsData::parse_present_aac_lc(reader, &ics)?)
    } else {
        None
    };
    let hcr_side = hcr_enabled
        .then(|| HcrSideInfo::parse(reader, hcr_element_type))
        .transpose()?;
    if let Some(side) = &rvlc_side {
        scalefactors = Some(decode_rvlc_or_conceal(
            reader,
            side,
            &ics,
            &section_data,
            global_gain,
        )?);
    }
    let tns_data = if let Some(tns) = eld_tns_data {
        tns
    } else if tns_present {
        TnsData::parse_present_aac_lc(reader, &ics)?
    } else {
        TnsData::absent(ics.window_group_lengths.iter().map(|&v| v as usize).sum())
    };
    let mut spectral = if let Some(side) = hcr_side {
        let payload = side.read_payload(reader)?;
        decode_hcr_spectral_or_mute(
            &payload,
            &side,
            &ics,
            &section_data,
            sfb.offsets,
            sfb.granule_length,
        )?
    } else {
        decode_spectral_data(reader, &ics, &section_data, sfb.offsets, sfb.granule_length)?
    };
    pulse_data.apply_to_spectral(&mut spectral, sfb.offsets)?;
    let scalefactors = scalefactors.expect("normal or RVLC scalefactors decoded");
    let spectrum = inverse_quantize_spectrum_f32(&spectral, &scalefactors, &ics, sfb)?;
    Ok(DecodedChannelStream {
        global_gain,
        ics,
        section_data,
        scalefactors,
        pulse_data,
        tns_data,
        spectral,
        spectrum,
    })
}

fn decode_drm_aac_single_channel_spectra_from_reader(
    reader: &mut BitReader<'_>,
    sampling_frequency_index: u8,
    pns_random: &mut PnsRandomState,
) -> Result<(DecodedSingleChannelSpectra, usize), DecodeError> {
    let start = reader.bits_read();
    let ics = IcsInfo::parse_aac_lc(reader, IcsLimits::AAC_LC_MAX)?;
    let sfb = aac_band_offsets_for_ics(sampling_frequency_index, &ics, 960)?;
    let tns_present = reader.read_bool()?;
    if reader.read_bool()? {
        return Err(DecodeError::LtpUnsupported);
    }
    let global_gain = reader.read_u8(8)?;
    let section_data = SectionData::parse_aac_lc_with_vcb11(reader, &ics, true)?;
    let scalefactor_plan = ScalefactorPlan::from_section_data(&section_data)?;
    let scalefactors = scalefactor_plan.decode_from_bitstream(reader, global_gain)?;
    let hcr_side = HcrSideInfo::parse(reader, HcrElementType::SingleChannel)?;
    let tns_data = if tns_present {
        TnsData::parse_present_aac_lc(reader, &ics)?
    } else {
        TnsData::absent(ics.window_group_lengths.iter().map(|&v| v as usize).sum())
    };
    let protected_bits = reader.bits_read() - start;
    let hcr_payload = hcr_side.read_payload(reader)?;
    let spectral = decode_hcr_spectral_or_mute(
        &hcr_payload,
        &hcr_side,
        &ics,
        &section_data,
        sfb.offsets,
        sfb.granule_length,
    )?;
    let mut spectrum = inverse_quantize_spectrum_f32(&spectral, &scalefactors, &ics, sfb)?;
    apply_pns_f32(
        &mut spectrum,
        &ics,
        sfb.offsets,
        &section_data,
        &scalefactors,
        pns_random,
    )?;
    tns_data.apply_f32(&mut spectrum, sfb.offsets, ics.max_sfb as usize)?;
    let stream = DecodedChannelStream {
        global_gain,
        ics: ics.clone(),
        section_data: section_data.clone(),
        scalefactors,
        pulse_data: PulseData::absent(),
        tns_data,
        spectral,
        spectrum,
    };
    Ok((
        DecodedSingleChannelSpectra {
            side_info: crate::raw::SingleChannelElementSideInfo {
                id: ElementId::SingleChannel,
                element_instance_tag: 0,
                global_gain,
                ics,
                bits_read: protected_bits,
            },
            stream,
            bits_read: reader.bits_read() - start,
        },
        protected_bits,
    ))
}

fn decode_drm_aac_channel_pair_spectra_from_reader(
    reader: &mut BitReader<'_>,
    sampling_frequency_index: u8,
    pns_random: &mut PnsRandomState,
) -> Result<(DecodedChannelPairSpectra, usize), DecodeError> {
    let start = reader.bits_read();
    let ics = IcsInfo::parse_aac_lc(reader, IcsLimits::AAC_LC_MAX)?;
    let sfb = aac_band_offsets_for_ics(sampling_frequency_index, &ics, 960)?;
    let ms_stereo = MsStereoData::parse_aac_lc(reader, &ics)?;

    let parse_side = |reader: &mut BitReader<'_>| -> Result<_, DecodeError> {
        let tns_present = reader.read_bool()?;
        if reader.read_bool()? {
            return Err(DecodeError::LtpUnsupported);
        }
        let global_gain = reader.read_u8(8)?;
        let section_data = SectionData::parse_aac_lc_with_vcb11(reader, &ics, true)?;
        let scalefactor_plan = ScalefactorPlan::from_section_data(&section_data)?;
        let scalefactors = scalefactor_plan.decode_from_bitstream(reader, global_gain)?;
        let hcr = HcrSideInfo::parse(reader, HcrElementType::ChannelPair)?;
        Ok((tns_present, global_gain, section_data, scalefactors, hcr))
    };
    let left_side = parse_side(reader)?;
    let right_start = reader.bits_read() - start;
    let right_side = parse_side(reader)?;
    let left_tns = if left_side.0 {
        TnsData::parse_present_aac_lc(reader, &ics)?
    } else {
        TnsData::absent(ics.window_group_lengths.iter().map(|&v| v as usize).sum())
    };
    let right_tns = if right_side.0 {
        TnsData::parse_present_aac_lc(reader, &ics)?
    } else {
        TnsData::absent(ics.window_group_lengths.iter().map(|&v| v as usize).sum())
    };
    let protected_bits = reader.bits_read() - start;

    let decode_stream = |reader: &mut BitReader<'_>,
                         side: (bool, u8, SectionData, ScalefactorData, HcrSideInfo),
                         tns_data: TnsData|
     -> Result<DecodedChannelStream, DecodeError> {
        let (_, global_gain, section_data, scalefactors, hcr) = side;
        let payload = hcr.read_payload(reader)?;
        let spectral = decode_hcr_spectral_or_mute(
            &payload,
            &hcr,
            &ics,
            &section_data,
            sfb.offsets,
            sfb.granule_length,
        )?;
        let spectrum = inverse_quantize_spectrum_f32(&spectral, &scalefactors, &ics, sfb)?;
        Ok(DecodedChannelStream {
            global_gain,
            ics: ics.clone(),
            section_data,
            scalefactors,
            pulse_data: PulseData::absent(),
            tns_data,
            spectral,
            spectrum,
        })
    };
    let mut left = decode_stream(reader, left_side, left_tns)?;
    let mut right = decode_stream(reader, right_side, right_tns)?;
    apply_pns_pair_f32(
        &mut left.spectrum,
        &mut right.spectrum,
        &ics,
        sfb.offsets,
        &left.section_data,
        &right.section_data,
        &left.scalefactors,
        &right.scalefactors,
        Some(&ms_stereo),
        pns_random,
    )?;
    left.tns_data
        .apply_f32(&mut left.spectrum, sfb.offsets, ics.max_sfb as usize)?;
    right
        .tns_data
        .apply_f32(&mut right.spectrum, sfb.offsets, ics.max_sfb as usize)?;
    Ok((
        DecodedChannelPairSpectra {
            prefix: ChannelPairElementSideInfoPrefix {
                element_instance_tag: 0,
                common_window: true,
                shared_ics: Some(ics),
                bits_read: right_start,
            },
            ms_stereo: Some(ms_stereo),
            left,
            right,
            right_channel_start_bit: right_start,
            bits_read: reader.bits_read() - start,
        },
        protected_bits,
    ))
}

fn er_long_sfb_count(sampling_frequency_index: u8, frame_length: usize) -> Result<u8, DecodeError> {
    Ok(match (frame_length, sampling_frequency_index) {
        (512, 0..=4) => 36,
        (512, 5) => 37,
        (512, 6..=12) => 31,
        (480, 0..=4) => 35,
        (480, 5) => 37,
        (480, 6..=12) => 30,
        (_, _) => return Err(SfbError::UnsupportedFrameLength(frame_length).into()),
    })
}

fn decode_er_single_channel_spectra_from_reader(
    reader: &mut BitReader<'_>,
    element_id: ElementId,
    sampling_frequency_index: u8,
    frame_length: usize,
    eld_enabled: bool,
    vcb11_enabled: bool,
    rvlc_enabled: bool,
    hcr_enabled: bool,
    pns_random: &mut PnsRandomState,
) -> Result<DecodedSingleChannelSpectra, DecodeError> {
    let start = reader.bits_read();
    let element_instance_tag = if eld_enabled { 0 } else { reader.read_u8(4)? };
    let stream = decode_er_channel_stream_from_reader(
        reader,
        sampling_frequency_index,
        frame_length,
        eld_enabled,
        None,
        vcb11_enabled,
        rvlc_enabled,
        hcr_enabled,
        match element_id {
            ElementId::Lfe => HcrElementType::LowFrequencyEffects,
            _ => HcrElementType::SingleChannel,
        },
    )?;
    let sfb = aac_band_offsets_for_ics(sampling_frequency_index, &stream.ics, frame_length)?;
    let mut stream = stream;
    apply_pns_f32(
        &mut stream.spectrum,
        &stream.ics,
        sfb.offsets,
        &stream.section_data,
        &stream.scalefactors,
        pns_random,
    )?;
    stream.tns_data.apply_f32(
        &mut stream.spectrum,
        sfb.offsets,
        stream.ics.max_sfb as usize,
    )?;
    Ok(DecodedSingleChannelSpectra {
        side_info: SingleChannelElementSideInfo {
            id: element_id,
            element_instance_tag,
            global_gain: stream.global_gain,
            ics: stream.ics.clone(),
            bits_read: reader.bits_read() - start,
        },
        stream,
        bits_read: reader.bits_read() - start,
    })
}

struct EldEp1ChannelPrefix {
    global_gain: u8,
    ics: IcsInfo,
    section_data: SectionData,
    scalefactors: Option<ScalefactorData>,
    rvlc_side: Option<RvlcSideInfo>,
    tns_present: bool,
}

fn parse_eld_ep1_channel_prefix(
    reader: &mut BitReader<'_>,
    shared_ics: &IcsInfo,
    vcb11_enabled: bool,
    rvlc_enabled: bool,
) -> Result<EldEp1ChannelPrefix, DecodeError> {
    let global_gain = reader.read_u8(8)?;
    let section_data = SectionData::parse_aac_lc_with_vcb11(reader, shared_ics, vcb11_enabled)?;
    let scalefactor_plan = ScalefactorPlan::from_section_data(&section_data)?;
    let rvlc_side = rvlc_enabled
        .then(|| RvlcSideInfo::parse(reader, shared_ics, &section_data))
        .transpose()?;
    let scalefactors = if rvlc_enabled {
        None
    } else {
        Some(scalefactor_plan.decode_from_bitstream(reader, global_gain)?)
    };
    let tns_present = reader.read_bool()?;
    Ok(EldEp1ChannelPrefix {
        global_gain,
        ics: shared_ics.clone(),
        section_data,
        scalefactors,
        rvlc_side,
        tns_present,
    })
}

fn read_eld_ep1_tns(
    reader: &mut BitReader<'_>,
    prefix: &EldEp1ChannelPrefix,
) -> Result<TnsData, DecodeError> {
    if prefix.tns_present {
        Ok(TnsData::parse_present_aac_lc(reader, &prefix.ics)?)
    } else {
        Ok(TnsData::absent(1))
    }
}

fn finish_eld_ep1_channel_f32(
    reader: &mut BitReader<'_>,
    prefix: EldEp1ChannelPrefix,
    tns_data: TnsData,
    sampling_frequency_index: u8,
    frame_length: usize,
    hcr_enabled: bool,
) -> Result<DecodedChannelStream, DecodeError> {
    let sfb = aac_band_offsets_for_ics(sampling_frequency_index, &prefix.ics, frame_length)?;
    let hcr_side = hcr_enabled
        .then(|| HcrSideInfo::parse(reader, HcrElementType::ChannelPair))
        .transpose()?;
    let scalefactors = match &prefix.rvlc_side {
        Some(side) => decode_rvlc_or_conceal(
            reader,
            side,
            &prefix.ics,
            &prefix.section_data,
            prefix.global_gain,
        )?,
        None => prefix
            .scalefactors
            .expect("normal ELD scalefactors decoded in side-info stage"),
    };
    let spectral = if let Some(side) = hcr_side {
        let payload = side.read_payload(reader)?;
        decode_hcr_spectral_or_mute(
            &payload,
            &side,
            &prefix.ics,
            &prefix.section_data,
            sfb.offsets,
            sfb.granule_length,
        )?
    } else {
        decode_spectral_data(
            reader,
            &prefix.ics,
            &prefix.section_data,
            sfb.offsets,
            sfb.granule_length,
        )?
    };
    let spectrum = inverse_quantize_spectrum_f32(&spectral, &scalefactors, &prefix.ics, sfb)?;
    Ok(DecodedChannelStream {
        global_gain: prefix.global_gain,
        ics: prefix.ics,
        section_data: prefix.section_data,
        scalefactors,
        pulse_data: PulseData::absent(),
        tns_data,
        spectral,
        spectrum,
    })
}

fn finish_eld_ep1_channel_fixed(
    reader: &mut BitReader<'_>,
    prefix: EldEp1ChannelPrefix,
    tns_data: TnsData,
    sampling_frequency_index: u8,
    frame_length: usize,
    hcr_enabled: bool,
) -> Result<DecodedChannelStreamFixed, DecodeError> {
    let sfb = aac_band_offsets_for_ics(sampling_frequency_index, &prefix.ics, frame_length)?;
    let hcr_side = hcr_enabled
        .then(|| HcrSideInfo::parse(reader, HcrElementType::ChannelPair))
        .transpose()?;
    let scalefactors = match &prefix.rvlc_side {
        Some(side) => decode_rvlc_or_conceal(
            reader,
            side,
            &prefix.ics,
            &prefix.section_data,
            prefix.global_gain,
        )?,
        None => prefix
            .scalefactors
            .expect("normal ELD scalefactors decoded in side-info stage"),
    };
    let spectral = if let Some(side) = hcr_side {
        let payload = side.read_payload(reader)?;
        decode_hcr_spectral_or_mute(
            &payload,
            &side,
            &prefix.ics,
            &prefix.section_data,
            sfb.offsets,
            sfb.granule_length,
        )?
    } else {
        decode_spectral_data(
            reader,
            &prefix.ics,
            &prefix.section_data,
            sfb.offsets,
            sfb.granule_length,
        )?
    };
    let spectrum =
        inverse_quantize_spectrum_fixed_block_scaled(&spectral, &scalefactors, &prefix.ics, sfb)?;
    Ok(DecodedChannelStreamFixed {
        global_gain: prefix.global_gain,
        ics: prefix.ics,
        section_data: prefix.section_data,
        scalefactors,
        pulse_data: PulseData::absent(),
        tns_data,
        spectral,
        spectrum,
    })
}

fn decode_eld_ep1_channel_pair_prefix(
    reader: &mut BitReader<'_>,
    sampling_frequency_index: u8,
    frame_length: usize,
    vcb11_enabled: bool,
    rvlc_enabled: bool,
) -> Result<
    (
        IcsInfo,
        MsStereoData,
        EldEp1ChannelPrefix,
        EldEp1ChannelPrefix,
        TnsData,
        TnsData,
    ),
    DecodeError,
> {
    let shared_ics = IcsInfo::parse_eld(
        reader,
        er_long_sfb_count(sampling_frequency_index, frame_length)?,
    )?;
    let ms_stereo = MsStereoData::parse_aac_lc(reader, &shared_ics)?;
    let left = parse_eld_ep1_channel_prefix(reader, &shared_ics, vcb11_enabled, rvlc_enabled)?;
    let right = parse_eld_ep1_channel_prefix(reader, &shared_ics, vcb11_enabled, rvlc_enabled)?;
    let left_tns = read_eld_ep1_tns(reader, &left)?;
    let right_tns = read_eld_ep1_tns(reader, &right)?;
    Ok((shared_ics, ms_stereo, left, right, left_tns, right_tns))
}

fn decode_eld_ep1_channel_pair_spectra_from_reader(
    reader: &mut BitReader<'_>,
    sampling_frequency_index: u8,
    frame_length: usize,
    vcb11_enabled: bool,
    rvlc_enabled: bool,
    hcr_enabled: bool,
    pns_random: &mut PnsRandomState,
) -> Result<DecodedChannelPairSpectra, DecodeError> {
    let start = reader.bits_read();
    let (shared_ics, ms_stereo, left_prefix, right_prefix, left_tns, right_tns) =
        decode_eld_ep1_channel_pair_prefix(
            reader,
            sampling_frequency_index,
            frame_length,
            vcb11_enabled,
            rvlc_enabled,
        )?;
    let left = finish_eld_ep1_channel_f32(
        reader,
        left_prefix,
        left_tns,
        sampling_frequency_index,
        frame_length,
        hcr_enabled,
    )?;
    let right_channel_start_bit = reader.bits_read() - start;
    let right = finish_eld_ep1_channel_f32(
        reader,
        right_prefix,
        right_tns,
        sampling_frequency_index,
        frame_length,
        hcr_enabled,
    )?;
    let mut decoded = DecodedChannelPairSpectra {
        prefix: ChannelPairElementSideInfoPrefix {
            element_instance_tag: 0,
            common_window: true,
            shared_ics: Some(shared_ics),
            bits_read: right_channel_start_bit,
        },
        ms_stereo: Some(ms_stereo),
        left,
        right,
        right_channel_start_bit,
        bits_read: reader.bits_read() - start,
    };
    apply_channel_pair_pns_and_tns(
        &mut decoded,
        sampling_frequency_index,
        frame_length,
        pns_random,
    )?;
    Ok(decoded)
}

fn decode_er_channel_pair_spectra_from_reader(
    reader: &mut BitReader<'_>,
    sampling_frequency_index: u8,
    frame_length: usize,
    eld_enabled: bool,
    eld_ep_config_1: bool,
    vcb11_enabled: bool,
    rvlc_enabled: bool,
    hcr_enabled: bool,
    pns_random: &mut PnsRandomState,
) -> Result<DecodedChannelPairSpectra, DecodeError> {
    if eld_enabled && eld_ep_config_1 {
        return decode_eld_ep1_channel_pair_spectra_from_reader(
            reader,
            sampling_frequency_index,
            frame_length,
            vcb11_enabled,
            rvlc_enabled,
            hcr_enabled,
            pns_random,
        );
    }
    let start = reader.bits_read();
    let element_instance_tag = if eld_enabled { 0 } else { reader.read_u8(4)? };
    let common_window = if eld_enabled {
        true
    } else {
        reader.read_bool()?
    };
    let shared_ics = common_window
        .then(|| parse_er_ics(reader, sampling_frequency_index, frame_length, eld_enabled))
        .transpose()?;
    let ms_stereo = shared_ics
        .as_ref()
        .map(|ics| MsStereoData::parse_aac_lc(reader, ics))
        .transpose()?;
    let left = decode_er_channel_stream_from_reader(
        reader,
        sampling_frequency_index,
        frame_length,
        eld_enabled,
        shared_ics.as_ref(),
        vcb11_enabled,
        rvlc_enabled,
        hcr_enabled,
        HcrElementType::ChannelPair,
    )?;
    let right_channel_start_bit = reader.bits_read() - start;
    let right = decode_er_channel_stream_from_reader(
        reader,
        sampling_frequency_index,
        frame_length,
        eld_enabled,
        shared_ics.as_ref(),
        vcb11_enabled,
        rvlc_enabled,
        hcr_enabled,
        HcrElementType::ChannelPair,
    )?;
    let mut decoded = DecodedChannelPairSpectra {
        prefix: ChannelPairElementSideInfoPrefix {
            element_instance_tag,
            common_window,
            shared_ics,
            bits_read: right_channel_start_bit,
        },
        ms_stereo,
        left,
        right,
        right_channel_start_bit,
        bits_read: reader.bits_read() - start,
    };
    apply_channel_pair_pns_and_tns(
        &mut decoded,
        sampling_frequency_index,
        frame_length,
        pns_random,
    )?;
    Ok(decoded)
}

fn decode_er_channel_stream_fixed_from_reader(
    reader: &mut BitReader<'_>,
    sampling_frequency_index: u8,
    frame_length: usize,
    eld_enabled: bool,
    shared_ics: Option<&IcsInfo>,
    vcb11_enabled: bool,
    rvlc_enabled: bool,
    hcr_enabled: bool,
    hcr_element_type: HcrElementType,
) -> Result<DecodedChannelStreamFixed, DecodeError> {
    let global_gain = reader.read_u8(8)?;
    let ics = match shared_ics {
        Some(ics) => ics.clone(),
        None => parse_er_ics(reader, sampling_frequency_index, frame_length, eld_enabled)?,
    };
    if frame_length <= 512 && !ics.window_sequence.is_long() {
        return Err(DecodeError::UnsupportedFrameLength(frame_length));
    }
    let sfb = aac_band_offsets_for_ics(sampling_frequency_index, &ics, frame_length)?;
    let section_data = SectionData::parse_aac_lc_with_vcb11(reader, &ics, vcb11_enabled)?;
    let scalefactor_plan = ScalefactorPlan::from_section_data(&section_data)?;
    let rvlc_side = rvlc_enabled
        .then(|| RvlcSideInfo::parse(reader, &ics, &section_data))
        .transpose()?;
    let mut scalefactors = if rvlc_enabled {
        None
    } else {
        Some(scalefactor_plan.decode_from_bitstream(reader, global_gain)?)
    };
    let pulse_data = if eld_enabled {
        PulseData::absent()
    } else {
        PulseData::parse_aac_lc(reader, &ics, sfb.offsets, sfb.granule_length)?
    };
    let tns_present = reader.read_bool()?;
    if !eld_enabled && reader.read_bool()? {
        return Err(DecodeError::GainControlUnsupported);
    }
    let eld_tns_data = if eld_enabled && tns_present {
        Some(TnsData::parse_present_aac_lc(reader, &ics)?)
    } else {
        None
    };
    let hcr_side = hcr_enabled
        .then(|| HcrSideInfo::parse(reader, hcr_element_type))
        .transpose()?;
    if let Some(side) = &rvlc_side {
        scalefactors = Some(decode_rvlc_or_conceal(
            reader,
            side,
            &ics,
            &section_data,
            global_gain,
        )?);
    }
    let tns_data = if let Some(tns) = eld_tns_data {
        tns
    } else if tns_present {
        TnsData::parse_present_aac_lc(reader, &ics)?
    } else {
        TnsData::absent(ics.window_group_lengths.iter().map(|&v| v as usize).sum())
    };
    let mut spectral = if let Some(side) = hcr_side {
        let payload = side.read_payload(reader)?;
        decode_hcr_spectral_or_mute(
            &payload,
            &side,
            &ics,
            &section_data,
            sfb.offsets,
            sfb.granule_length,
        )?
    } else {
        decode_spectral_data(reader, &ics, &section_data, sfb.offsets, sfb.granule_length)?
    };
    pulse_data.apply_to_spectral(&mut spectral, sfb.offsets)?;
    let scalefactors = scalefactors.expect("normal or RVLC scalefactors decoded");
    let spectrum = if eld_enabled {
        inverse_quantize_spectrum_fixed_block_scaled(&spectral, &scalefactors, &ics, sfb)?
    } else {
        inverse_quantize_spectrum_fixed(&spectral, &scalefactors, &ics, sfb)?
    };
    Ok(DecodedChannelStreamFixed {
        global_gain,
        ics,
        section_data,
        scalefactors,
        pulse_data,
        tns_data,
        spectral,
        spectrum,
    })
}

fn decode_hcr_spectral_or_mute(
    payload: &[u8],
    side: &HcrSideInfo,
    ics: &IcsInfo,
    section_data: &SectionData,
    band_offsets: &[usize],
    granule_length: usize,
) -> Result<SpectralData, DecodeError> {
    let sections = hcr_sections_from_ics(ics, section_data, band_offsets)?;
    match decode_hcr_codewords(payload, side, &sections)
        .and_then(|words| hcr_codewords_to_spectral(ics, &sections, &words, granule_length))
    {
        Ok(spectral) => Ok(spectral),
        Err(_) => Ok(SpectralData {
            windows: vec![
                vec![0; granule_length];
                ics.window_group_lengths.iter().sum::<u8>() as usize
            ],
        }),
    }
}

fn synthesize_aac_eld_frame_f32(
    spectrum: &InverseQuantizedSpectrum,
    filterbank: &mut LowDelayFilterbankF32,
) -> Result<Vec<f32>, DecodeError> {
    if spectrum.windows.len() != 1 {
        return Err(FilterbankError::ExpectedOneLongWindow {
            actual: spectrum.windows.len(),
        }
        .into());
    }
    Ok(filterbank.process(&spectrum.windows[0])?)
}

fn synthesize_aac_eld_frame_fixed_i16(
    spectrum: &FixedInverseQuantizedSpectrum,
    filterbank: &mut LowDelayFilterbankQ31,
) -> Result<Vec<i16>, DecodeError> {
    if spectrum.windows.len() != 1 {
        return Err(FilterbankError::ExpectedOneLongWindow {
            actual: spectrum.windows.len(),
        }
        .into());
    }
    Ok(synthesize_aac_eld_frame_fixed_q31(spectrum, filterbank)?
        .into_iter()
        .map(dbl_to_pcm16)
        .collect())
}

fn synthesize_aac_eld_frame_fixed_q31(
    spectrum: &FixedInverseQuantizedSpectrum,
    filterbank: &mut LowDelayFilterbankQ31,
) -> Result<Vec<FixpDbl>, DecodeError> {
    if spectrum.windows.len() != 1 {
        return Err(FilterbankError::ExpectedOneLongWindow {
            actual: spectrum.windows.len(),
        }
        .into());
    }
    Ok(filterbank.process_with_exponent(
        &spectrum.windows[0],
        spectrum
            .window_exponents
            .first()
            .copied()
            .unwrap_or(0)
            .saturating_add(3),
    )?)
}

fn decode_rvlc_or_conceal(
    reader: &mut BitReader<'_>,
    side: &RvlcSideInfo,
    ics: &IcsInfo,
    section_data: &SectionData,
    global_gain: u8,
) -> Result<ScalefactorData, DecodeError> {
    let declared_bits = side.scalefactor_bits.saturating_add(side.escape_bits);
    if reader.remaining_bits() < declared_bits {
        return Ok(decode_rvlc_forward(reader, side, ics, section_data, global_gain)?.scalefactors);
    }
    match decode_rvlc_forward(reader, side, ics, section_data, global_gain) {
        Ok(decoded) => Ok(decoded.scalefactors),
        Err(_) => Ok(conceal_rvlc_scalefactors(
            side,
            ics,
            section_data,
            global_gain,
        )?),
    }
}

fn decode_er_single_channel_spectra_fixed_from_reader(
    reader: &mut BitReader<'_>,
    element_id: ElementId,
    sampling_frequency_index: u8,
    frame_length: usize,
    eld_enabled: bool,
    vcb11_enabled: bool,
    rvlc_enabled: bool,
    hcr_enabled: bool,
    pns_random: &mut PnsRandomState,
) -> Result<DecodedSingleChannelSpectraFixed, DecodeError> {
    let start = reader.bits_read();
    let element_instance_tag = if eld_enabled { 0 } else { reader.read_u8(4)? };
    let mut stream = decode_er_channel_stream_fixed_from_reader(
        reader,
        sampling_frequency_index,
        frame_length,
        eld_enabled,
        None,
        vcb11_enabled,
        rvlc_enabled,
        hcr_enabled,
        match element_id {
            ElementId::Lfe => HcrElementType::LowFrequencyEffects,
            _ => HcrElementType::SingleChannel,
        },
    )?;
    let sfb = aac_band_offsets_for_ics(sampling_frequency_index, &stream.ics, frame_length)?;
    apply_pns_fixed(
        &mut stream.spectrum,
        &stream.ics,
        sfb.offsets,
        &stream.section_data,
        &stream.scalefactors,
        pns_random,
    )?;
    stream.tns_data.apply_fixed(
        &mut stream.spectrum,
        sfb.offsets,
        stream.ics.max_sfb as usize,
    )?;
    Ok(DecodedSingleChannelSpectraFixed {
        side_info: SingleChannelElementSideInfo {
            id: element_id,
            element_instance_tag,
            global_gain: stream.global_gain,
            ics: stream.ics.clone(),
            bits_read: reader.bits_read() - start,
        },
        stream,
        bits_read: reader.bits_read() - start,
    })
}

fn decode_er_channel_pair_spectra_fixed_from_reader(
    reader: &mut BitReader<'_>,
    sampling_frequency_index: u8,
    frame_length: usize,
    eld_enabled: bool,
    eld_ep_config_1: bool,
    vcb11_enabled: bool,
    rvlc_enabled: bool,
    hcr_enabled: bool,
    pns_random: &mut PnsRandomState,
) -> Result<DecodedChannelPairSpectraFixed, DecodeError> {
    if eld_enabled && eld_ep_config_1 {
        return decode_eld_ep1_channel_pair_spectra_fixed_from_reader(
            reader,
            sampling_frequency_index,
            frame_length,
            vcb11_enabled,
            rvlc_enabled,
            hcr_enabled,
            pns_random,
        );
    }
    let start = reader.bits_read();
    let element_instance_tag = if eld_enabled { 0 } else { reader.read_u8(4)? };
    let common_window = if eld_enabled {
        true
    } else {
        reader.read_bool()?
    };
    let shared_ics = common_window
        .then(|| parse_er_ics(reader, sampling_frequency_index, frame_length, eld_enabled))
        .transpose()?;
    let ms_stereo = shared_ics
        .as_ref()
        .map(|ics| MsStereoData::parse_aac_lc(reader, ics))
        .transpose()?;
    let left = decode_er_channel_stream_fixed_from_reader(
        reader,
        sampling_frequency_index,
        frame_length,
        eld_enabled,
        shared_ics.as_ref(),
        vcb11_enabled,
        rvlc_enabled,
        hcr_enabled,
        HcrElementType::ChannelPair,
    )?;
    let right_channel_start_bit = reader.bits_read() - start;
    let right = decode_er_channel_stream_fixed_from_reader(
        reader,
        sampling_frequency_index,
        frame_length,
        eld_enabled,
        shared_ics.as_ref(),
        vcb11_enabled,
        rvlc_enabled,
        hcr_enabled,
        HcrElementType::ChannelPair,
    )?;
    let mut decoded = DecodedChannelPairSpectraFixed {
        prefix: ChannelPairElementSideInfoPrefix {
            element_instance_tag,
            common_window,
            shared_ics,
            bits_read: right_channel_start_bit,
        },
        ms_stereo,
        left,
        right,
        right_channel_start_bit,
        bits_read: reader.bits_read() - start,
    };
    apply_channel_pair_pns_and_tns_fixed_bridge(
        &mut decoded,
        sampling_frequency_index,
        frame_length,
        pns_random,
    )?;
    Ok(decoded)
}

fn decode_eld_ep1_channel_pair_spectra_fixed_from_reader(
    reader: &mut BitReader<'_>,
    sampling_frequency_index: u8,
    frame_length: usize,
    vcb11_enabled: bool,
    rvlc_enabled: bool,
    hcr_enabled: bool,
    pns_random: &mut PnsRandomState,
) -> Result<DecodedChannelPairSpectraFixed, DecodeError> {
    let start = reader.bits_read();
    let (shared_ics, ms_stereo, left_prefix, right_prefix, left_tns, right_tns) =
        decode_eld_ep1_channel_pair_prefix(
            reader,
            sampling_frequency_index,
            frame_length,
            vcb11_enabled,
            rvlc_enabled,
        )?;
    let left = finish_eld_ep1_channel_fixed(
        reader,
        left_prefix,
        left_tns,
        sampling_frequency_index,
        frame_length,
        hcr_enabled,
    )?;
    let right_channel_start_bit = reader.bits_read() - start;
    let right = finish_eld_ep1_channel_fixed(
        reader,
        right_prefix,
        right_tns,
        sampling_frequency_index,
        frame_length,
        hcr_enabled,
    )?;
    let mut decoded = DecodedChannelPairSpectraFixed {
        prefix: ChannelPairElementSideInfoPrefix {
            element_instance_tag: 0,
            common_window: true,
            shared_ics: Some(shared_ics),
            bits_read: right_channel_start_bit,
        },
        ms_stereo: Some(ms_stereo),
        left,
        right,
        right_channel_start_bit,
        bits_read: reader.bits_read() - start,
    };
    apply_channel_pair_pns_and_tns_fixed_bridge(
        &mut decoded,
        sampling_frequency_index,
        frame_length,
        pns_random,
    )?;
    Ok(decoded)
}

pub fn program_config_channel_labels(program_config: &ProgramConfig) -> Vec<ChannelLabel> {
    let mut labels = Vec::with_capacity(program_config.num_channels as usize);
    let mut front_center_used = false;
    for element in &program_config.front {
        if element.is_cpe {
            labels.push(ChannelLabel::FrontLeft);
            labels.push(ChannelLabel::FrontRight);
        } else if !front_center_used {
            labels.push(ChannelLabel::FrontCenter);
            front_center_used = true;
        } else {
            labels.push(ChannelLabel::Unknown(labels.len()));
        }
    }
    for element in &program_config.side {
        if element.is_cpe {
            labels.push(ChannelLabel::SideLeft);
            labels.push(ChannelLabel::SideRight);
        } else {
            labels.push(ChannelLabel::Unknown(labels.len()));
        }
    }
    for element in &program_config.back {
        if element.is_cpe {
            labels.push(ChannelLabel::BackLeft);
            labels.push(ChannelLabel::BackRight);
        } else {
            labels.push(ChannelLabel::BackCenter);
        }
    }
    for _ in &program_config.lfe {
        labels.push(ChannelLabel::Lfe);
    }
    labels
}

pub fn program_config_labels_for_element(
    program_config: &ProgramConfig,
    element_id: ElementId,
    element_instance_tag: u8,
    unknown_base: usize,
) -> Vec<ChannelLabel> {
    match element_id {
        ElementId::SingleChannel => {
            program_config_single_channel_label(program_config, element_instance_tag, unknown_base)
                .map(|label| vec![label])
                .unwrap_or_else(|| vec![ChannelLabel::Unknown(unknown_base)])
        }
        ElementId::Lfe => {
            if program_config.lfe.contains(&element_instance_tag) {
                vec![ChannelLabel::Lfe]
            } else {
                vec![ChannelLabel::Unknown(unknown_base)]
            }
        }
        ElementId::ChannelPair => {
            program_config_channel_pair_labels(program_config, element_instance_tag, unknown_base)
                .unwrap_or_else(|| {
                    vec![
                        ChannelLabel::Unknown(unknown_base),
                        ChannelLabel::Unknown(unknown_base + 1),
                    ]
                })
        }
        _ => Vec::new(),
    }
}

fn program_config_single_channel_label(
    program_config: &ProgramConfig,
    element_instance_tag: u8,
    unknown_base: usize,
) -> Option<ChannelLabel> {
    let mut front_sce_index = 0usize;
    for element in &program_config.front {
        if !element.is_cpe {
            if element.tag_select == element_instance_tag {
                return Some(if front_sce_index == 0 {
                    ChannelLabel::FrontCenter
                } else {
                    ChannelLabel::Unknown(unknown_base)
                });
            }
            front_sce_index += 1;
        }
    }
    for element in &program_config.side {
        if !element.is_cpe && element.tag_select == element_instance_tag {
            return Some(ChannelLabel::Unknown(unknown_base));
        }
    }
    for element in &program_config.back {
        if !element.is_cpe && element.tag_select == element_instance_tag {
            return Some(ChannelLabel::BackCenter);
        }
    }
    None
}

fn program_config_channel_pair_labels(
    program_config: &ProgramConfig,
    element_instance_tag: u8,
    _unknown_base: usize,
) -> Option<Vec<ChannelLabel>> {
    for element in &program_config.front {
        if element.is_cpe && element.tag_select == element_instance_tag {
            return Some(vec![ChannelLabel::FrontLeft, ChannelLabel::FrontRight]);
        }
    }
    for element in &program_config.side {
        if element.is_cpe && element.tag_select == element_instance_tag {
            return Some(vec![ChannelLabel::SideLeft, ChannelLabel::SideRight]);
        }
    }
    for element in &program_config.back {
        if element.is_cpe && element.tag_select == element_instance_tag {
            return Some(vec![ChannelLabel::BackLeft, ChannelLabel::BackRight]);
        }
    }
    None
}

pub fn decode_aac_lc_single_channel_f32(
    input: &[u8],
    sampling_frequency_index: u8,
    filterbank: &mut LongBlockFilterbank,
    pns_random: &mut PnsRandomState,
) -> Result<DecodedSingleChannelFrame, DecodeError> {
    let mut reader = BitReader::new(input);
    decode_aac_lc_single_channel_f32_from_reader(
        &mut reader,
        sampling_frequency_index,
        filterbank,
        pns_random,
    )
}

pub fn decode_aac_lc_single_channel_f32_from_reader(
    reader: &mut BitReader<'_>,
    sampling_frequency_index: u8,
    filterbank: &mut LongBlockFilterbank,
    pns_random: &mut PnsRandomState,
) -> Result<DecodedSingleChannelFrame, DecodeError> {
    let spectra = decode_aac_lc_single_channel_spectra_from_reader(
        reader,
        sampling_frequency_index,
        pns_random,
    )?;
    let samples =
        synthesize_aac_lc_frame(&spectra.stream.spectrum, &spectra.stream.ics, filterbank)?;

    Ok(DecodedSingleChannelFrame {
        side_info: spectra.side_info,
        section_data: spectra.stream.section_data,
        scalefactors: spectra.stream.scalefactors,
        pulse_data: spectra.stream.pulse_data,
        tns_data: spectra.stream.tns_data,
        spectral: spectra.stream.spectral,
        spectrum: spectra.stream.spectrum,
        samples,
        bits_read: spectra.bits_read,
    })
}

pub fn decode_aac_lc_single_channel_fixed_i16(
    input: &[u8],
    sampling_frequency_index: u8,
    filterbank: &mut FixedLongBlockFilterbank,
    pns_random: &mut PnsRandomState,
) -> Result<Vec<i16>, DecodeError> {
    let mut reader = BitReader::new(input);
    decode_aac_lc_single_channel_fixed_i16_from_reader(
        &mut reader,
        sampling_frequency_index,
        filterbank,
        pns_random,
    )
}

pub fn decode_aac_lc_single_channel_fixed_i16_from_reader(
    reader: &mut BitReader<'_>,
    sampling_frequency_index: u8,
    filterbank: &mut FixedLongBlockFilterbank,
    pns_random: &mut PnsRandomState,
) -> Result<Vec<i16>, DecodeError> {
    let spectra = decode_aac_lc_single_channel_spectra_from_reader(
        reader,
        sampling_frequency_index,
        pns_random,
    )?;
    Ok(synthesize_aac_lc_frame_from_inverse_q31(
        &spectra.stream.spectrum,
        &spectra.stream.ics,
        filterbank,
    )?
    .into_iter()
    .map(dbl_to_pcm16)
    .collect())
}

pub fn decode_aac_lc_single_channel_spectra_from_reader(
    reader: &mut BitReader<'_>,
    sampling_frequency_index: u8,
    pns_random: &mut PnsRandomState,
) -> Result<DecodedSingleChannelSpectra, DecodeError> {
    decode_aac_lc_single_channel_spectra_staged_from_reader(
        reader,
        sampling_frequency_index,
        1024,
        pns_random,
        true,
    )
}

fn decode_aac_lc_single_channel_spectra_staged_from_reader(
    reader: &mut BitReader<'_>,
    sampling_frequency_index: u8,
    frame_length: usize,
    pns_random: &mut PnsRandomState,
    apply_tns: bool,
) -> Result<DecodedSingleChannelSpectra, DecodeError> {
    let start = reader.bits_read();
    let side_info =
        SingleChannelElementSideInfo::parse_aac_lc_from_reader(reader, IcsLimits::AAC_LC_MAX)?;
    let mut channel = decode_channel_stream_after_global_gain(
        reader,
        sampling_frequency_index,
        frame_length,
        side_info.global_gain,
        side_info.ics.clone(),
    )?;
    let sfb = aac_band_offsets_for_ics(sampling_frequency_index, &channel.ics, frame_length)?;
    apply_pns_f32(
        &mut channel.spectrum,
        &channel.ics,
        sfb.offsets,
        &channel.section_data,
        &channel.scalefactors,
        pns_random,
    )?;
    if apply_tns {
        channel.tns_data.apply_f32(
            &mut channel.spectrum,
            sfb.offsets,
            channel.ics.max_sfb as usize,
        )?;
    }
    Ok(DecodedSingleChannelSpectra {
        side_info,
        stream: channel,
        bits_read: reader.bits_read() - start,
    })
}

pub fn decode_aac_lc_single_channel_spectra_fixed_bridge(
    input: &[u8],
    sampling_frequency_index: u8,
    pns_random: &mut PnsRandomState,
) -> Result<DecodedSingleChannelSpectraFixed, DecodeError> {
    let mut reader = BitReader::new(input);
    decode_aac_lc_single_channel_spectra_fixed_bridge_from_reader(
        &mut reader,
        sampling_frequency_index,
        pns_random,
    )
}

pub fn decode_aac_lc_single_channel_spectra_fixed_bridge_from_reader(
    reader: &mut BitReader<'_>,
    sampling_frequency_index: u8,
    pns_random: &mut PnsRandomState,
) -> Result<DecodedSingleChannelSpectraFixed, DecodeError> {
    decode_aac_lc_single_channel_spectra_fixed_staged_from_reader(
        reader,
        sampling_frequency_index,
        1024,
        pns_random,
        true,
    )
}

fn decode_aac_lc_single_channel_spectra_fixed_staged_from_reader(
    reader: &mut BitReader<'_>,
    sampling_frequency_index: u8,
    frame_length: usize,
    pns_random: &mut PnsRandomState,
    apply_tns: bool,
) -> Result<DecodedSingleChannelSpectraFixed, DecodeError> {
    let start = reader.bits_read();
    let side_info =
        SingleChannelElementSideInfo::parse_aac_lc_from_reader(reader, IcsLimits::AAC_LC_MAX)?;
    let stream = decode_channel_stream_fixed_bridge_after_global_gain(
        reader,
        sampling_frequency_index,
        frame_length,
        side_info.global_gain,
        side_info.ics.clone(),
        pns_random,
        false,
    )?;
    let sfb = aac_band_offsets_for_ics(sampling_frequency_index, &stream.ics, frame_length)?;
    let mut stream = stream;
    apply_pns_fixed(
        &mut stream.spectrum,
        &stream.ics,
        sfb.offsets,
        &stream.section_data,
        &stream.scalefactors,
        pns_random,
    )?;
    if apply_tns {
        stream.tns_data.apply_fixed(
            &mut stream.spectrum,
            sfb.offsets,
            stream.ics.max_sfb as usize,
        )?;
    }
    Ok(DecodedSingleChannelSpectraFixed {
        side_info,
        stream,
        bits_read: reader.bits_read() - start,
    })
}

pub fn decode_aac_lc_channel_pair_spectra(
    input: &[u8],
    sampling_frequency_index: u8,
    pns_random: &mut PnsRandomState,
) -> Result<DecodedChannelPairSpectra, DecodeError> {
    let mut reader = BitReader::new(input);
    decode_aac_lc_channel_pair_spectra_from_reader(
        &mut reader,
        sampling_frequency_index,
        pns_random,
    )
}

pub fn decode_aac_lc_channel_pair_spectra_from_reader(
    reader: &mut BitReader<'_>,
    sampling_frequency_index: u8,
    pns_random: &mut PnsRandomState,
) -> Result<DecodedChannelPairSpectra, DecodeError> {
    decode_aac_lc_channel_pair_spectra_staged_from_reader(
        reader,
        sampling_frequency_index,
        1024,
        pns_random,
        true,
    )
}

fn decode_aac_lc_channel_pair_spectra_staged_from_reader(
    reader: &mut BitReader<'_>,
    sampling_frequency_index: u8,
    frame_length: usize,
    pns_random: &mut PnsRandomState,
    apply_tns: bool,
) -> Result<DecodedChannelPairSpectra, DecodeError> {
    let start = reader.bits_read();
    let prefix =
        ChannelPairElementSideInfoPrefix::parse_aac_lc_from_reader(reader, IcsLimits::AAC_LC_MAX)?;
    let ms_stereo = if let Some(shared_ics) = &prefix.shared_ics {
        Some(MsStereoData::parse_aac_lc(reader, shared_ics)?)
    } else {
        None
    };

    let left = decode_channel_stream_from_reader(
        reader,
        sampling_frequency_index,
        frame_length,
        prefix.shared_ics.as_ref(),
    )?;
    let right_channel_start_bit = reader.bits_read() - start - 3;
    let right = decode_channel_stream_from_reader(
        reader,
        sampling_frequency_index,
        frame_length,
        prefix.shared_ics.as_ref(),
    )?;

    let mut decoded = DecodedChannelPairSpectra {
        prefix,
        ms_stereo,
        left,
        right,
        right_channel_start_bit,
        bits_read: 0,
    };
    apply_channel_pair_pns(
        &mut decoded,
        sampling_frequency_index,
        frame_length,
        pns_random,
    )?;
    if apply_tns {
        apply_channel_pair_tns(&mut decoded, sampling_frequency_index, frame_length)?;
    }
    decoded.bits_read = reader.bits_read() - start;
    Ok(decoded)
}

pub fn decode_aac_lc_channel_pair_spectra_fixed_bridge(
    input: &[u8],
    sampling_frequency_index: u8,
    pns_random: &mut PnsRandomState,
) -> Result<DecodedChannelPairSpectraFixed, DecodeError> {
    let mut reader = BitReader::new(input);
    decode_aac_lc_channel_pair_spectra_fixed_bridge_from_reader(
        &mut reader,
        sampling_frequency_index,
        pns_random,
    )
}

pub fn decode_aac_lc_channel_pair_spectra_fixed_bridge_from_reader(
    reader: &mut BitReader<'_>,
    sampling_frequency_index: u8,
    pns_random: &mut PnsRandomState,
) -> Result<DecodedChannelPairSpectraFixed, DecodeError> {
    decode_aac_lc_channel_pair_spectra_fixed_staged_from_reader(
        reader,
        sampling_frequency_index,
        1024,
        pns_random,
        true,
    )
}

fn decode_aac_lc_channel_pair_spectra_fixed_staged_from_reader(
    reader: &mut BitReader<'_>,
    sampling_frequency_index: u8,
    frame_length: usize,
    pns_random: &mut PnsRandomState,
    apply_tns: bool,
) -> Result<DecodedChannelPairSpectraFixed, DecodeError> {
    let start = reader.bits_read();
    let prefix =
        ChannelPairElementSideInfoPrefix::parse_aac_lc_from_reader(reader, IcsLimits::AAC_LC_MAX)?;
    let ms_stereo = if let Some(shared_ics) = &prefix.shared_ics {
        Some(MsStereoData::parse_aac_lc(reader, shared_ics)?)
    } else {
        None
    };

    let left = decode_channel_stream_fixed_bridge_from_reader(
        reader,
        sampling_frequency_index,
        frame_length,
        prefix.shared_ics.as_ref(),
        pns_random,
        false,
    )?;
    let right_channel_start_bit = reader.bits_read() - start - 3;
    let right = decode_channel_stream_fixed_bridge_from_reader(
        reader,
        sampling_frequency_index,
        frame_length,
        prefix.shared_ics.as_ref(),
        pns_random,
        false,
    )?;

    let mut decoded = DecodedChannelPairSpectraFixed {
        prefix,
        ms_stereo,
        left,
        right,
        right_channel_start_bit,
        bits_read: 0,
    };
    apply_channel_pair_pns_fixed_bridge(
        &mut decoded,
        sampling_frequency_index,
        frame_length,
        pns_random,
    )?;
    if apply_tns {
        apply_channel_pair_tns_fixed_bridge(&mut decoded, sampling_frequency_index, frame_length)?;
    }
    decoded.bits_read = reader.bits_read() - start;
    Ok(decoded)
}

pub fn decode_aac_lc_channel_pair_f32(
    input: &[u8],
    sampling_frequency_index: u8,
    left_filterbank: &mut LongBlockFilterbank,
    right_filterbank: &mut LongBlockFilterbank,
    pns_random: &mut PnsRandomState,
) -> Result<DecodedChannelPairFrame, DecodeError> {
    let mut reader = BitReader::new(input);
    decode_aac_lc_channel_pair_f32_from_reader(
        &mut reader,
        sampling_frequency_index,
        left_filterbank,
        right_filterbank,
        pns_random,
    )
}

pub fn decode_aac_lc_channel_pair_f32_from_reader(
    reader: &mut BitReader<'_>,
    sampling_frequency_index: u8,
    left_filterbank: &mut LongBlockFilterbank,
    right_filterbank: &mut LongBlockFilterbank,
    pns_random: &mut PnsRandomState,
) -> Result<DecodedChannelPairFrame, DecodeError> {
    let mut spectra = decode_aac_lc_channel_pair_spectra_from_reader(
        reader,
        sampling_frequency_index,
        pns_random,
    )?;
    apply_aac_lc_channel_pair_stereo_tools_fixed_bridge(&mut spectra, sampling_frequency_index)?;
    let left_samples =
        synthesize_aac_lc_frame(&spectra.left.spectrum, &spectra.left.ics, left_filterbank)?;
    let right_samples = synthesize_aac_lc_frame(
        &spectra.right.spectrum,
        &spectra.right.ics,
        right_filterbank,
    )?;
    Ok(DecodedChannelPairFrame {
        spectra,
        left_samples,
        right_samples,
    })
}

pub fn decode_aac_lc_channel_pair_fixed_interleaved_i16(
    input: &[u8],
    sampling_frequency_index: u8,
    left_filterbank: &mut FixedLongBlockFilterbank,
    right_filterbank: &mut FixedLongBlockFilterbank,
    pns_random: &mut PnsRandomState,
) -> Result<Vec<i16>, DecodeError> {
    let mut reader = BitReader::new(input);
    decode_aac_lc_channel_pair_fixed_interleaved_i16_from_reader(
        &mut reader,
        sampling_frequency_index,
        left_filterbank,
        right_filterbank,
        pns_random,
    )
}

pub fn decode_aac_lc_channel_pair_fixed_interleaved_i16_from_reader(
    reader: &mut BitReader<'_>,
    sampling_frequency_index: u8,
    left_filterbank: &mut FixedLongBlockFilterbank,
    right_filterbank: &mut FixedLongBlockFilterbank,
    pns_random: &mut PnsRandomState,
) -> Result<Vec<i16>, DecodeError> {
    let mut spectra = decode_aac_lc_channel_pair_spectra_from_reader(
        reader,
        sampling_frequency_index,
        pns_random,
    )?;
    apply_aac_lc_channel_pair_stereo_tools_f32(&mut spectra, sampling_frequency_index)?;
    let left = synthesize_aac_lc_frame_from_inverse_q31(
        &spectra.left.spectrum,
        &spectra.left.ics,
        left_filterbank,
    )?
    .into_iter()
    .map(dbl_to_pcm16)
    .collect::<Vec<_>>();
    let right = synthesize_aac_lc_frame_from_inverse_q31(
        &spectra.right.spectrum,
        &spectra.right.ics,
        right_filterbank,
    )?
    .into_iter()
    .map(dbl_to_pcm16)
    .collect::<Vec<_>>();
    Ok(interleave_stereo_i16_samples(&left, &right))
}

pub fn decode_aac_lc_channel_pair_fixed_spectrum_interleaved_i16_bridge(
    input: &[u8],
    sampling_frequency_index: u8,
    left_filterbank: &mut FixedLongBlockFilterbank,
    right_filterbank: &mut FixedLongBlockFilterbank,
    pns_random: &mut PnsRandomState,
) -> Result<Vec<i16>, DecodeError> {
    let mut reader = BitReader::new(input);
    decode_aac_lc_channel_pair_fixed_spectrum_interleaved_i16_bridge_from_reader(
        &mut reader,
        sampling_frequency_index,
        left_filterbank,
        right_filterbank,
        pns_random,
    )
}

pub fn decode_aac_lc_channel_pair_fixed_spectrum_interleaved_i16_bridge_from_reader(
    reader: &mut BitReader<'_>,
    sampling_frequency_index: u8,
    left_filterbank: &mut FixedLongBlockFilterbank,
    right_filterbank: &mut FixedLongBlockFilterbank,
    pns_random: &mut PnsRandomState,
) -> Result<Vec<i16>, DecodeError> {
    let mut spectra = decode_aac_lc_channel_pair_spectra_fixed_bridge_from_reader(
        reader,
        sampling_frequency_index,
        pns_random,
    )?;
    apply_aac_lc_channel_pair_fixed_spectrum_stereo_tools_bridge(
        &mut spectra,
        sampling_frequency_index,
    )?;
    let left = synthesize_aac_lc_frame_from_fixed_inverse_q31(
        &spectra.left.spectrum,
        &spectra.left.ics,
        left_filterbank,
    )?
    .into_iter()
    .map(dbl_to_pcm16)
    .collect::<Vec<_>>();
    let right = synthesize_aac_lc_frame_from_fixed_inverse_q31(
        &spectra.right.spectrum,
        &spectra.right.ics,
        right_filterbank,
    )?
    .into_iter()
    .map(dbl_to_pcm16)
    .collect::<Vec<_>>();
    Ok(interleave_stereo_i16_samples(&left, &right))
}

pub fn apply_aac_lc_channel_pair_stereo_tools_f32(
    decoded: &mut DecodedChannelPairSpectra,
    sampling_frequency_index: u8,
) -> Result<(), DecodeError> {
    let sfb = aac_lc_band_offsets_for_ics(sampling_frequency_index, &decoded.left.ics)?;
    if let Some(ms) = &decoded.ms_stereo {
        apply_ms_stereo_f32(
            ms,
            &mut decoded.left.spectrum,
            &mut decoded.right.spectrum,
            &decoded.left.ics,
            sfb.offsets,
            &decoded.left.section_data,
            &decoded.right.section_data,
        )?;
    }
    apply_intensity_stereo_f32(
        decoded.ms_stereo.as_ref(),
        &decoded.left.spectrum,
        &mut decoded.right.spectrum,
        &decoded.left.ics,
        sfb.offsets,
        &decoded.right.section_data,
        &decoded.right.scalefactors,
    )?;
    Ok(())
}

pub fn apply_aac_lc_channel_pair_stereo_tools_fixed_bridge(
    decoded: &mut DecodedChannelPairSpectra,
    sampling_frequency_index: u8,
) -> Result<(), DecodeError> {
    let sfb = aac_lc_band_offsets_for_ics(sampling_frequency_index, &decoded.left.ics)?;
    if let Some(ms) = &decoded.ms_stereo {
        apply_ms_stereo_fixed_bridge(
            ms,
            &mut decoded.left.spectrum,
            &mut decoded.right.spectrum,
            &decoded.left.ics,
            sfb.offsets,
            &decoded.left.section_data,
            &decoded.right.section_data,
        )?;
    }
    apply_intensity_stereo_fixed_bridge(
        decoded.ms_stereo.as_ref(),
        &decoded.left.spectrum,
        &mut decoded.right.spectrum,
        &decoded.left.ics,
        sfb.offsets,
        &decoded.right.section_data,
        &decoded.right.scalefactors,
    )?;
    Ok(())
}

pub fn apply_aac_lc_channel_pair_fixed_spectrum_stereo_tools_bridge(
    decoded: &mut DecodedChannelPairSpectraFixed,
    sampling_frequency_index: u8,
) -> Result<(), DecodeError> {
    let sfb = aac_lc_band_offsets_for_ics(sampling_frequency_index, &decoded.left.ics)?;
    if let Some(ms) = &decoded.ms_stereo {
        apply_ms_stereo_fixed_spectrum_bridge(
            ms,
            &mut decoded.left.spectrum,
            &mut decoded.right.spectrum,
            &decoded.left.ics,
            sfb.offsets,
            &decoded.left.section_data,
            &decoded.right.section_data,
        )?;
    }
    apply_intensity_stereo_fixed_spectrum_bridge(
        decoded.ms_stereo.as_ref(),
        &decoded.left.spectrum,
        &mut decoded.right.spectrum,
        &decoded.left.ics,
        sfb.offsets,
        &decoded.right.section_data,
        &decoded.right.scalefactors,
    )?;
    Ok(())
}

pub fn apply_intensity_stereo_fixed_spectrum_bridge(
    ms: Option<&MsStereoData>,
    left: &FixedInverseQuantizedSpectrum,
    right: &mut FixedInverseQuantizedSpectrum,
    ics: &IcsInfo,
    band_offsets: &[usize],
    right_sections: &SectionData,
    right_scalefactors: &ScalefactorData,
) -> Result<(), DecodeError> {
    let groups = ics.window_group_lengths.len();
    let max_sfb = ics.max_sfb as usize;
    let total_windows = ics
        .window_group_lengths
        .iter()
        .map(|&len| len as usize)
        .sum::<usize>();
    if left.windows.len() != total_windows
        || right.windows.len() != total_windows
        || band_offsets.len() <= max_sfb
        || right_sections.codebooks.len() != groups
        || right_sections
            .codebooks
            .iter()
            .any(|group| group.len() < max_sfb)
        || right_scalefactors.values.len() != groups
        || right_scalefactors
            .values
            .iter()
            .any(|group| group.len() < max_sfb)
    {
        return Err(DecodeError::Stereo(StereoError::LayoutMismatch));
    }

    let mut window_offset = 0usize;
    for (group, &group_len) in ics.window_group_lengths.iter().enumerate() {
        for band in 0..ics.max_sfb as usize {
            let codebook = right_sections.codebooks[group][band];
            if codebook != INTENSITY_HCB && codebook != INTENSITY_HCB2 {
                continue;
            }
            let scale_q15 = (intensity_scale_f32(
                right_scalefactors.values[group][band],
                codebook,
                ms.is_some_and(|ms| ms.is_used(group, band)),
            ) * 32768.0)
                .round() as i64;
            let start = band_offsets[band];
            let end = band_offsets[band + 1];
            for group_window in 0..group_len as usize {
                let window = window_offset + group_window;
                if left.windows[window].len() < end || right.windows[window].len() < end {
                    return Err(DecodeError::Stereo(StereoError::LayoutMismatch));
                }
                for index in start..end {
                    let sample = (left.windows[window][index] as i64 * scale_q15 + (1 << 14)) >> 15;
                    right.windows[window][index] =
                        sample.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
                }
            }
        }
        window_offset += group_len as usize;
    }
    Ok(())
}

pub fn apply_ms_stereo_fixed_spectrum_bridge(
    ms: &MsStereoData,
    left: &mut FixedInverseQuantizedSpectrum,
    right: &mut FixedInverseQuantizedSpectrum,
    ics: &IcsInfo,
    band_offsets: &[usize],
    left_sections: &SectionData,
    right_sections: &SectionData,
) -> Result<(), DecodeError> {
    let groups = ics.window_group_lengths.len();
    let max_sfb = ics.max_sfb as usize;
    let total_windows = ics
        .window_group_lengths
        .iter()
        .map(|&len| len as usize)
        .sum::<usize>();
    if ms.used.len() != groups
        || ms.used.iter().any(|group| group.len() < max_sfb)
        || left.windows.len() != total_windows
        || right.windows.len() != total_windows
        || band_offsets.len() <= max_sfb
    {
        return Err(DecodeError::Stereo(StereoError::LayoutMismatch));
    }

    const INV_SQRT_2_Q15: i64 = 23170;
    let mut window_offset = 0usize;
    for (group, &group_len) in ics.window_group_lengths.iter().enumerate() {
        for band in 0..ics.max_sfb as usize {
            if !ms.is_used(group, band)
                || !is_ms_applicable_for_fixed_bridge(left_sections, right_sections, group, band)
            {
                continue;
            }
            let start = band_offsets[band];
            let end = band_offsets[band + 1];
            for group_window in 0..group_len as usize {
                let window = window_offset + group_window;
                if left.windows[window].len() < end || right.windows[window].len() < end {
                    return Err(DecodeError::Stereo(StereoError::LayoutMismatch));
                }
                for index in start..end {
                    let mid = left.windows[window][index] as i64;
                    let side = right.windows[window][index] as i64;
                    let new_left = ((mid + side) * INV_SQRT_2_Q15 + (1 << 14)) >> 15;
                    let new_right = ((mid - side) * INV_SQRT_2_Q15 + (1 << 14)) >> 15;
                    left.windows[window][index] =
                        new_left.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
                    right.windows[window][index] =
                        new_right.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
                }
            }
        }
        window_offset += group_len as usize;
    }
    Ok(())
}

pub fn apply_intensity_stereo_fixed_bridge(
    ms: Option<&MsStereoData>,
    left: &InverseQuantizedSpectrum,
    right: &mut InverseQuantizedSpectrum,
    ics: &IcsInfo,
    band_offsets: &[usize],
    right_sections: &SectionData,
    right_scalefactors: &ScalefactorData,
) -> Result<(), DecodeError> {
    let groups = ics.window_group_lengths.len();
    let max_sfb = ics.max_sfb as usize;
    let total_windows = ics
        .window_group_lengths
        .iter()
        .map(|&len| len as usize)
        .sum::<usize>();
    if left.windows.len() != total_windows
        || right.windows.len() != total_windows
        || band_offsets.len() <= max_sfb
        || right_sections.codebooks.len() != groups
        || right_sections
            .codebooks
            .iter()
            .any(|group| group.len() < max_sfb)
        || right_scalefactors.values.len() != groups
        || right_scalefactors
            .values
            .iter()
            .any(|group| group.len() < max_sfb)
    {
        return Err(DecodeError::Stereo(StereoError::LayoutMismatch));
    }

    const SPECTRAL_BRIDGE_SCALE: f32 = 32768.0;
    const SCALE_Q15: f32 = 32768.0;
    let mut window_offset = 0usize;
    for (group, &group_len) in ics.window_group_lengths.iter().enumerate() {
        for band in 0..ics.max_sfb as usize {
            let codebook = right_sections.codebooks[group][band];
            if codebook != INTENSITY_HCB && codebook != INTENSITY_HCB2 {
                continue;
            }
            let scale_q15 = (intensity_scale_f32(
                right_scalefactors.values[group][band],
                codebook,
                ms.is_some_and(|ms| ms.is_used(group, band)),
            ) * SCALE_Q15)
                .round() as i64;
            let start = band_offsets[band];
            let end = band_offsets[band + 1];
            for group_window in 0..group_len as usize {
                let window = window_offset + group_window;
                if left.windows[window].len() < end || right.windows[window].len() < end {
                    return Err(DecodeError::Stereo(StereoError::LayoutMismatch));
                }
                for index in start..end {
                    let source =
                        (left.windows[window][index] * SPECTRAL_BRIDGE_SCALE).round() as i64;
                    let sample = (source * scale_q15 + (1 << 14)) >> 15;
                    right.windows[window][index] = sample as f32 / SPECTRAL_BRIDGE_SCALE;
                }
            }
        }
        window_offset += group_len as usize;
    }
    Ok(())
}

pub fn apply_ms_stereo_fixed_bridge(
    ms: &MsStereoData,
    left: &mut InverseQuantizedSpectrum,
    right: &mut InverseQuantizedSpectrum,
    ics: &IcsInfo,
    band_offsets: &[usize],
    left_sections: &SectionData,
    right_sections: &SectionData,
) -> Result<(), DecodeError> {
    let groups = ics.window_group_lengths.len();
    let max_sfb = ics.max_sfb as usize;
    let total_windows = ics
        .window_group_lengths
        .iter()
        .map(|&len| len as usize)
        .sum::<usize>();
    if ms.used.len() != groups
        || ms.used.iter().any(|group| group.len() < max_sfb)
        || left.windows.len() != total_windows
        || right.windows.len() != total_windows
        || band_offsets.len() <= max_sfb
    {
        return Err(DecodeError::Stereo(StereoError::LayoutMismatch));
    }

    const SPECTRAL_BRIDGE_SCALE: f32 = 32768.0;
    const INV_SQRT_2_Q15: i64 = 23170;
    let mut window_offset = 0usize;
    for (group, &group_len) in ics.window_group_lengths.iter().enumerate() {
        for band in 0..ics.max_sfb as usize {
            if !ms.is_used(group, band)
                || !is_ms_applicable_for_fixed_bridge(left_sections, right_sections, group, band)
            {
                continue;
            }
            let start = band_offsets[band];
            let end = band_offsets[band + 1];
            for group_window in 0..group_len as usize {
                let window = window_offset + group_window;
                if left.windows[window].len() < end || right.windows[window].len() < end {
                    return Err(DecodeError::Stereo(StereoError::LayoutMismatch));
                }
                for index in start..end {
                    let mid = (left.windows[window][index] * SPECTRAL_BRIDGE_SCALE).round() as i64;
                    let side =
                        (right.windows[window][index] * SPECTRAL_BRIDGE_SCALE).round() as i64;
                    // Malformed or extreme escape-coded spectra can saturate
                    // the f32-to-i64 bridge. Keep the fixed-point MS operation
                    // total by widening before the add and multiply.
                    let new_left = (((i128::from(mid) + i128::from(side))
                        * i128::from(INV_SQRT_2_Q15)
                        + (1 << 14))
                        >> 15) as f64;
                    let new_right = (((i128::from(mid) - i128::from(side))
                        * i128::from(INV_SQRT_2_Q15)
                        + (1 << 14))
                        >> 15) as f64;
                    left.windows[window][index] =
                        (new_left / f64::from(SPECTRAL_BRIDGE_SCALE)) as f32;
                    right.windows[window][index] =
                        (new_right / f64::from(SPECTRAL_BRIDGE_SCALE)) as f32;
                }
            }
        }
        window_offset += group_len as usize;
    }
    Ok(())
}

fn is_ms_applicable_for_fixed_bridge(
    left_sections: &SectionData,
    right_sections: &SectionData,
    group: usize,
    band: usize,
) -> bool {
    let left = left_sections.codebooks[group][band];
    let right = right_sections.codebooks[group][band];
    is_fixed_bridge_spectral_or_zero(left) && is_fixed_bridge_spectral_or_zero(right)
}

fn is_fixed_bridge_spectral_or_zero(codebook: u8) -> bool {
    !matches!(codebook, NOISE_HCB | INTENSITY_HCB | INTENSITY_HCB2) && codebook != ZERO_HCB
}

pub fn decode_aac_lc_coupling_channel_element(
    input: &[u8],
    sampling_frequency_index: u8,
) -> Result<DecodedCouplingChannelElement, DecodeError> {
    let mut reader = BitReader::new(input);
    decode_aac_lc_coupling_channel_element_from_reader(&mut reader, sampling_frequency_index)
}

pub fn decode_aac_lc_coupling_channel_element_from_reader(
    reader: &mut BitReader<'_>,
    sampling_frequency_index: u8,
) -> Result<DecodedCouplingChannelElement, DecodeError> {
    let start = reader.bits_read();
    let prefix = CouplingChannelElementPrefix::parse_aac_lc_from_reader(reader)?;
    let stream = decode_channel_stream_from_reader(reader, sampling_frequency_index, 1024, None)?;
    let gain_lists = decode_coupling_gain_element_lists(reader, &prefix, &stream)?;
    Ok(DecodedCouplingChannelElement {
        prefix,
        stream,
        gain_lists,
        bits_read: reader.bits_read() - start,
    })
}

pub fn decode_aac_lc_coupling_channel_element_fixed_bridge(
    input: &[u8],
    sampling_frequency_index: u8,
    pns_random: &mut PnsRandomState,
) -> Result<DecodedCouplingChannelElementFixed, DecodeError> {
    let mut reader = BitReader::new(input);
    decode_aac_lc_coupling_channel_element_fixed_bridge_from_reader(
        &mut reader,
        sampling_frequency_index,
        pns_random,
    )
}

pub fn decode_aac_lc_coupling_channel_element_fixed_bridge_from_reader(
    reader: &mut BitReader<'_>,
    sampling_frequency_index: u8,
    pns_random: &mut PnsRandomState,
) -> Result<DecodedCouplingChannelElementFixed, DecodeError> {
    let start = reader.bits_read();
    let prefix = CouplingChannelElementPrefix::parse_aac_lc_from_reader(reader)?;
    let stream = decode_channel_stream_fixed_bridge_from_reader(
        reader,
        sampling_frequency_index,
        1024,
        None,
        pns_random,
        true,
    )?;
    let gain_lists = decode_coupling_gain_element_lists_for_layout(
        reader,
        &prefix,
        &stream.ics,
        &stream.section_data,
    )?;
    Ok(DecodedCouplingChannelElementFixed {
        prefix,
        stream,
        gain_lists,
        bits_read: reader.bits_read() - start,
    })
}

pub fn decode_coupling_gain_element_lists(
    reader: &mut BitReader<'_>,
    prefix: &CouplingChannelElementPrefix,
    stream: &DecodedChannelStream,
) -> Result<CouplingGainElementLists, DecodeError> {
    decode_coupling_gain_element_lists_for_layout(reader, prefix, &stream.ics, &stream.section_data)
}

fn decode_coupling_gain_element_lists_for_layout(
    reader: &mut BitReader<'_>,
    prefix: &CouplingChannelElementPrefix,
    ics: &IcsInfo,
    section_data: &SectionData,
) -> Result<CouplingGainElementLists, DecodeError> {
    let mut lists = Vec::with_capacity(prefix.gain_element_lists);
    if prefix.gain_element_lists != 0 {
        // gain_element_list[0] is not transmitted. ISO/IEC 14496-3 defines
        // it as a common unity gain for the first coupled target.
        lists.push(CouplingGainElementList {
            common_gain_element_present: true,
            words: vec![60],
        });
    }
    for _ in 1..prefix.gain_element_lists {
        let common_gain_element_present = prefix.independently_switched || reader.read_bool()?;
        let mut words = Vec::new();
        if common_gain_element_present {
            words.push(decode_fdk_2bit(reader, &HUFFMAN_CODEBOOK_SCL)? as i16);
        } else {
            for group in 0..ics.window_group_lengths.len() {
                for sfb in 0..ics.max_sfb as usize {
                    if section_data.codebooks[group][sfb] != crate::section::ZERO_HCB {
                        words.push(decode_fdk_2bit(reader, &HUFFMAN_CODEBOOK_SCL)? as i16);
                    }
                }
            }
        }
        lists.push(CouplingGainElementList {
            common_gain_element_present,
            words,
        });
    }
    Ok(CouplingGainElementLists { lists })
}

pub fn apply_coupling_channel_element_noop_if_zero_gain(
    cce: &DecodedCouplingChannelElement,
) -> Result<(), DecodeError> {
    let has_gain_words = cce
        .gain_lists
        .lists
        .iter()
        .any(|list| !list.words.is_empty());
    if has_gain_words {
        return Err(DecodeError::CouplingGainApplicationUnsupported);
    }
    Ok(())
}

pub fn coupling_gain_word_to_scale(word: i16, gain_element_sign: bool) -> f32 {
    coupling_gain_word_to_scale_with_scale(word, gain_element_sign, 0)
}

pub fn coupling_gain_word_to_scale_with_scale(
    word: i16,
    gain_element_sign: bool,
    gain_element_scale: u8,
) -> f32 {
    coupling_gain_accumulator_to_scale(word - 60, gain_element_sign, gain_element_scale)
}

fn coupling_gain_accumulator_to_scale(
    accumulator: i16,
    gain_element_sign: bool,
    gain_element_scale: u8,
) -> f32 {
    let (signed_exponent, sign) = if gain_element_sign {
        (
            accumulator >> 1,
            if accumulator & 1 != 0 { -1.0 } else { 1.0 },
        )
    } else {
        (accumulator, 1.0)
    };
    let exponent_step = 2.0f32.powi(gain_element_scale.min(3) as i32 - 3);
    sign * 2.0f32.powf(-(signed_exponent as f32) * exponent_step)
}

pub fn apply_frequency_coupling_to_spectrum(
    target: &mut InverseQuantizedSpectrum,
    cce: &DecodedCouplingChannelElement,
    gain_list_index: usize,
) -> Result<(), DecodeError> {
    apply_frequency_coupling_to_spectrum_at_rate(target, cce, gain_list_index, 4)
}

fn apply_frequency_coupling_to_spectrum_at_rate(
    target: &mut InverseQuantizedSpectrum,
    cce: &DecodedCouplingChannelElement,
    gain_list_index: usize,
    sampling_frequency_index: u8,
) -> Result<(), DecodeError> {
    if !cce.prefix.uses_frequency_coupling() {
        return Err(DecodeError::TimeDomainCouplingUnsupported);
    }
    let Some(gain_list) = cce.gain_lists.lists.get(gain_list_index) else {
        return Ok(());
    };
    let Some(&word) = gain_list.words.first() else {
        return Ok(());
    };
    if !gain_list.common_gain_element_present {
        return apply_frequency_coupling_bandwise_to_spectrum_at_rate(
            target,
            cce,
            gain_list_index,
            sampling_frequency_index,
        );
    }
    if target.windows.len() != cce.stream.spectrum.windows.len() {
        return Err(DecodeError::CouplingLayoutMismatch);
    }
    let scale = coupling_gain_word_to_scale_with_scale(
        word,
        cce.prefix.gain_element_sign,
        cce.prefix.gain_element_scale,
    );
    for (target_window, coupling_window) in
        target.windows.iter_mut().zip(&cce.stream.spectrum.windows)
    {
        if target_window.len() != coupling_window.len() {
            return Err(DecodeError::CouplingLayoutMismatch);
        }
        for (target_line, &coupling_line) in target_window.iter_mut().zip(coupling_window) {
            *target_line += coupling_line * scale;
        }
    }
    Ok(())
}

pub fn apply_frequency_coupling_to_fixed_spectrum_bridge(
    target: &mut FixedInverseQuantizedSpectrum,
    cce: &DecodedCouplingChannelElementFixed,
    gain_list_index: usize,
) -> Result<(), DecodeError> {
    apply_frequency_coupling_to_fixed_spectrum_at_rate(target, cce, gain_list_index, 4)
}

fn apply_frequency_coupling_to_fixed_spectrum_at_rate(
    target: &mut FixedInverseQuantizedSpectrum,
    cce: &DecodedCouplingChannelElementFixed,
    gain_list_index: usize,
    sampling_frequency_index: u8,
) -> Result<(), DecodeError> {
    if !cce.prefix.uses_frequency_coupling() {
        return Err(DecodeError::TimeDomainCouplingUnsupported);
    }
    let Some(gain_list) = cce.gain_lists.lists.get(gain_list_index) else {
        return Ok(());
    };
    let Some(&word) = gain_list.words.first() else {
        return Ok(());
    };
    if !gain_list.common_gain_element_present {
        return apply_frequency_coupling_bandwise_to_fixed_spectrum_at_rate(
            target,
            cce,
            gain_list_index,
            sampling_frequency_index,
        );
    }
    if target.windows.len() != cce.stream.spectrum.windows.len() {
        return Err(DecodeError::CouplingLayoutMismatch);
    }
    let scale_q15 = (coupling_gain_word_to_scale_with_scale(
        word,
        cce.prefix.gain_element_sign,
        cce.prefix.gain_element_scale,
    ) * 32768.0)
        .round() as i64;
    for (target_window, coupling_window) in
        target.windows.iter_mut().zip(&cce.stream.spectrum.windows)
    {
        if target_window.len() != coupling_window.len() {
            return Err(DecodeError::CouplingLayoutMismatch);
        }
        for (target_line, &coupling_line) in target_window.iter_mut().zip(coupling_window) {
            let add = (coupling_line as i64 * scale_q15 + (1 << 14)) >> 15;
            let mixed = *target_line as i64 + add;
            *target_line = mixed.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        }
    }
    Ok(())
}

pub fn apply_frequency_coupling_bandwise_to_spectrum(
    target: &mut InverseQuantizedSpectrum,
    cce: &DecodedCouplingChannelElement,
    gain_list_index: usize,
) -> Result<(), DecodeError> {
    apply_frequency_coupling_bandwise_to_spectrum_at_rate(target, cce, gain_list_index, 4)
}

fn apply_frequency_coupling_bandwise_to_spectrum_at_rate(
    target: &mut InverseQuantizedSpectrum,
    cce: &DecodedCouplingChannelElement,
    gain_list_index: usize,
    sampling_frequency_index: u8,
) -> Result<(), DecodeError> {
    if !cce.prefix.uses_frequency_coupling() {
        return Err(DecodeError::TimeDomainCouplingUnsupported);
    }
    let Some(gain_list) = cce.gain_lists.lists.get(gain_list_index) else {
        return Ok(());
    };
    if gain_list.common_gain_element_present || gain_list.words.is_empty() {
        return apply_frequency_coupling_to_spectrum_at_rate(
            target,
            cce,
            gain_list_index,
            sampling_frequency_index,
        );
    }
    if target.windows.len() != cce.stream.spectrum.windows.len() {
        return Err(DecodeError::CouplingLayoutMismatch);
    }
    let sfb = aac_lc_band_offsets_for_ics(sampling_frequency_index, &cce.stream.ics)?;
    let mut word_index = 0usize;
    let mut gain_accumulator = 0i16;
    for (group, &group_len) in cce.stream.ics.window_group_lengths.iter().enumerate() {
        for band in 0..cce.stream.ics.max_sfb as usize {
            if cce.stream.section_data.codebooks[group][band] == crate::section::ZERO_HCB {
                continue;
            }
            let Some(&word) = gain_list.words.get(word_index) else {
                return Err(DecodeError::CouplingLayoutMismatch);
            };
            word_index += 1;
            gain_accumulator = gain_accumulator.saturating_add(word - 60);
            let scale = coupling_gain_accumulator_to_scale(
                gain_accumulator,
                cce.prefix.gain_element_sign,
                cce.prefix.gain_element_scale,
            );
            let start = sfb.offsets[band];
            let end = sfb.offsets[band + 1];
            let group_start: usize = cce.stream.ics.window_group_lengths[..group]
                .iter()
                .map(|&len| len as usize)
                .sum();
            for window in group_start..group_start + group_len as usize {
                if target.windows[window].len() != cce.stream.spectrum.windows[window].len() {
                    return Err(DecodeError::CouplingLayoutMismatch);
                }
                for line in start..end.min(target.windows[window].len()) {
                    target.windows[window][line] +=
                        cce.stream.spectrum.windows[window][line] * scale;
                }
            }
        }
    }
    Ok(())
}

pub fn apply_frequency_coupling_bandwise_to_fixed_spectrum_bridge(
    target: &mut FixedInverseQuantizedSpectrum,
    cce: &DecodedCouplingChannelElementFixed,
    gain_list_index: usize,
) -> Result<(), DecodeError> {
    apply_frequency_coupling_bandwise_to_fixed_spectrum_at_rate(target, cce, gain_list_index, 4)
}

fn apply_frequency_coupling_bandwise_to_fixed_spectrum_at_rate(
    target: &mut FixedInverseQuantizedSpectrum,
    cce: &DecodedCouplingChannelElementFixed,
    gain_list_index: usize,
    sampling_frequency_index: u8,
) -> Result<(), DecodeError> {
    if !cce.prefix.uses_frequency_coupling() {
        return Err(DecodeError::TimeDomainCouplingUnsupported);
    }
    let Some(gain_list) = cce.gain_lists.lists.get(gain_list_index) else {
        return Ok(());
    };
    if gain_list.common_gain_element_present || gain_list.words.is_empty() {
        return apply_frequency_coupling_to_fixed_spectrum_at_rate(
            target,
            cce,
            gain_list_index,
            sampling_frequency_index,
        );
    }
    if target.windows.len() != cce.stream.spectrum.windows.len() {
        return Err(DecodeError::CouplingLayoutMismatch);
    }
    let sfb = aac_lc_band_offsets_for_ics(sampling_frequency_index, &cce.stream.ics)?;
    let mut word_index = 0usize;
    let mut gain_accumulator = 0i16;
    for (group, &group_len) in cce.stream.ics.window_group_lengths.iter().enumerate() {
        for band in 0..cce.stream.ics.max_sfb as usize {
            if cce.stream.section_data.codebooks[group][band] == crate::section::ZERO_HCB {
                continue;
            }
            let Some(&word) = gain_list.words.get(word_index) else {
                return Err(DecodeError::CouplingLayoutMismatch);
            };
            word_index += 1;
            gain_accumulator = gain_accumulator.saturating_add(word - 60);
            let scale_q15 = (coupling_gain_accumulator_to_scale(
                gain_accumulator,
                cce.prefix.gain_element_sign,
                cce.prefix.gain_element_scale,
            ) * 32768.0)
                .round() as i64;
            let start = sfb.offsets[band];
            let end = sfb.offsets[band + 1];
            let group_start: usize = cce.stream.ics.window_group_lengths[..group]
                .iter()
                .map(|&len| len as usize)
                .sum();
            for window in group_start..group_start + group_len as usize {
                if target.windows[window].len() != cce.stream.spectrum.windows[window].len() {
                    return Err(DecodeError::CouplingLayoutMismatch);
                }
                for line in start..end.min(target.windows[window].len()) {
                    let add = (cce.stream.spectrum.windows[window][line] as i64 * scale_q15
                        + (1 << 14))
                        >> 15;
                    let mixed = target.windows[window][line] as i64 + add;
                    target.windows[window][line] =
                        mixed.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
                }
            }
        }
    }
    Ok(())
}

pub fn apply_time_domain_coupling_to_samples(
    target: &mut [f32],
    coupling: &[f32],
    cce: &DecodedCouplingChannelElement,
    gain_list_index: usize,
) -> Result<(), DecodeError> {
    if !cce.prefix.uses_time_coupling() {
        return Err(DecodeError::CouplingLayoutMismatch);
    }
    let Some(gain_list) = cce.gain_lists.lists.get(gain_list_index) else {
        return Ok(());
    };
    let Some(&word) = gain_list.words.first() else {
        return Ok(());
    };
    if !gain_list.common_gain_element_present {
        return Err(DecodeError::BandwiseCouplingGainUnsupported);
    }
    if target.len() != coupling.len() {
        return Err(DecodeError::CouplingLayoutMismatch);
    }
    let scale = coupling_gain_word_to_scale_with_scale(
        word,
        cce.prefix.gain_element_sign,
        cce.prefix.gain_element_scale,
    );
    for (target_sample, &coupling_sample) in target.iter_mut().zip(coupling) {
        *target_sample += coupling_sample * scale;
    }
    Ok(())
}

pub fn apply_coupling_channel_element_to_matching_spectra(
    targets: &mut [CouplingTargetSpectrum],
    cce: &DecodedCouplingChannelElement,
) -> Result<(), DecodeError> {
    let mut gain_index = 0usize;
    for target in &cce.prefix.targets {
        for channel in target_channel_indices(target) {
            if let Some(target_spectrum) = targets.iter_mut().find(|candidate| {
                candidate.element_instance_tag == target.tag_select
                    && candidate.channel == channel
                    && ((target.is_cpe && candidate.element_id == ElementId::ChannelPair)
                        || (!target.is_cpe && candidate.element_id != ElementId::ChannelPair))
            }) {
                apply_frequency_coupling_to_spectrum(
                    &mut target_spectrum.spectrum,
                    cce,
                    gain_index,
                )?;
            }
            gain_index += 1;
        }
    }
    Ok(())
}

fn target_channel_indices(target: &crate::raw::CouplingTarget) -> Vec<usize> {
    if !target.is_cpe {
        return vec![0];
    }
    let mut channels = Vec::new();
    if target.left {
        channels.push(0);
    }
    if target.right {
        channels.push(1);
    }
    channels
}

fn staged_channel_count(staged: &[StagedAacLcElement]) -> usize {
    staged
        .iter()
        .map(|element| match element {
            StagedAacLcElement::Single { .. } => 1,
            StagedAacLcElement::Pair { .. } => 2,
        })
        .sum()
}

fn staged_fixed_channel_count(staged: &[StagedAacLcElementFixed]) -> usize {
    staged
        .iter()
        .map(|element| match element {
            StagedAacLcElementFixed::Single { .. } => 1,
            StagedAacLcElementFixed::Pair { .. } => 2,
        })
        .sum()
}

fn staged_channel_map(staged: &[StagedAacLcElement]) -> Vec<StagedChannelMapEntry> {
    let mut map = Vec::new();
    let mut output_channel = 0usize;
    for element in staged {
        match element {
            StagedAacLcElement::Single {
                element_id,
                element_instance_tag,
                ..
            } => {
                map.push(StagedChannelMapEntry {
                    element_id: *element_id,
                    element_instance_tag: *element_instance_tag,
                    channel: 0,
                    output_channel,
                });
                output_channel += 1;
            }
            StagedAacLcElement::Pair {
                element_instance_tag,
                ..
            } => {
                map.push(StagedChannelMapEntry {
                    element_id: ElementId::ChannelPair,
                    element_instance_tag: *element_instance_tag,
                    channel: 0,
                    output_channel,
                });
                map.push(StagedChannelMapEntry {
                    element_id: ElementId::ChannelPair,
                    element_instance_tag: *element_instance_tag,
                    channel: 1,
                    output_channel: output_channel + 1,
                });
                output_channel += 2;
            }
        }
    }
    map
}

fn staged_fixed_channel_map(staged: &[StagedAacLcElementFixed]) -> Vec<StagedChannelMapEntry> {
    let mut map = Vec::new();
    let mut output_channel = 0usize;
    for element in staged {
        match element {
            StagedAacLcElementFixed::Single {
                element_id,
                element_instance_tag,
                ..
            } => {
                map.push(StagedChannelMapEntry {
                    element_id: *element_id,
                    element_instance_tag: *element_instance_tag,
                    channel: 0,
                    output_channel,
                });
                output_channel += 1;
            }
            StagedAacLcElementFixed::Pair {
                element_instance_tag,
                ..
            } => {
                map.push(StagedChannelMapEntry {
                    element_id: ElementId::ChannelPair,
                    element_instance_tag: *element_instance_tag,
                    channel: 0,
                    output_channel,
                });
                map.push(StagedChannelMapEntry {
                    element_id: ElementId::ChannelPair,
                    element_instance_tag: *element_instance_tag,
                    channel: 1,
                    output_channel: output_channel + 1,
                });
                output_channel += 2;
            }
        }
    }
    map
}

fn apply_time_domain_cce_to_channels(
    channels: &mut [Vec<f32>],
    channel_map: &[StagedChannelMapEntry],
    cce: &DecodedCouplingChannelElement,
    coupling_samples: &[f32],
) -> Result<(), DecodeError> {
    let mut gain_index = 0usize;
    for target in &cce.prefix.targets {
        for channel in target_channel_indices(target) {
            for mapped in channel_map {
                let matches = mapped.element_instance_tag == target.tag_select
                    && mapped.channel == channel
                    && ((target.is_cpe && mapped.element_id == ElementId::ChannelPair)
                        || (!target.is_cpe && mapped.element_id != ElementId::ChannelPair));
                if matches {
                    apply_time_domain_coupling_to_samples(
                        &mut channels[mapped.output_channel],
                        coupling_samples,
                        cce,
                        gain_index,
                    )?;
                }
            }
            gain_index += 1;
        }
    }
    Ok(())
}

fn apply_time_domain_cce_to_fixed_channels_fixed_cce(
    channels: &mut [Vec<FixpDbl>],
    channel_map: &[StagedChannelMapEntry],
    cce: &DecodedCouplingChannelElementFixed,
    coupling_samples: &[FixpDbl],
) -> Result<(), DecodeError> {
    let mut gain_index = 0usize;
    for target in &cce.prefix.targets {
        for channel in target_channel_indices(target) {
            for mapped in channel_map {
                let matches = mapped.element_instance_tag == target.tag_select
                    && mapped.channel == channel
                    && ((target.is_cpe && mapped.element_id == ElementId::ChannelPair)
                        || (!target.is_cpe && mapped.element_id != ElementId::ChannelPair));
                if matches {
                    apply_time_domain_coupling_to_fixed_samples_fixed_cce(
                        &mut channels[mapped.output_channel],
                        coupling_samples,
                        cce,
                        gain_index,
                    )?;
                }
            }
            gain_index += 1;
        }
    }
    Ok(())
}

pub fn apply_time_domain_coupling_to_fixed_samples(
    target: &mut [FixpDbl],
    coupling: &[FixpDbl],
    cce: &DecodedCouplingChannelElement,
    gain_list_index: usize,
) -> Result<(), DecodeError> {
    if !cce.prefix.uses_time_coupling() {
        return Err(DecodeError::CouplingLayoutMismatch);
    }
    let Some(gain_list) = cce.gain_lists.lists.get(gain_list_index) else {
        return Ok(());
    };
    let Some(&word) = gain_list.words.first() else {
        return Ok(());
    };
    if !gain_list.common_gain_element_present {
        return Err(DecodeError::BandwiseCouplingGainUnsupported);
    }
    if target.len() != coupling.len() {
        return Err(DecodeError::CouplingLayoutMismatch);
    }
    let scale = coupling_gain_word_to_scale_with_scale(
        word,
        cce.prefix.gain_element_sign,
        cce.prefix.gain_element_scale,
    );
    for (target_sample, &coupling_sample) in target.iter_mut().zip(coupling) {
        let scaled = (coupling_sample as f32 * scale).round() as i64;
        let mixed = *target_sample as i64 + scaled;
        *target_sample = mixed.clamp(i32::MIN as i64 + 1, i32::MAX as i64) as FixpDbl;
    }
    Ok(())
}

pub fn apply_time_domain_coupling_to_fixed_samples_fixed_cce(
    target: &mut [FixpDbl],
    coupling: &[FixpDbl],
    cce: &DecodedCouplingChannelElementFixed,
    gain_list_index: usize,
) -> Result<(), DecodeError> {
    if !cce.prefix.uses_time_coupling() {
        return Err(DecodeError::CouplingLayoutMismatch);
    }
    let Some(gain_list) = cce.gain_lists.lists.get(gain_list_index) else {
        return Ok(());
    };
    let Some(&word) = gain_list.words.first() else {
        return Ok(());
    };
    if !gain_list.common_gain_element_present {
        return Err(DecodeError::BandwiseCouplingGainUnsupported);
    }
    if target.len() != coupling.len() {
        return Err(DecodeError::CouplingLayoutMismatch);
    }
    let scale = coupling_gain_word_to_scale_with_scale(
        word,
        cce.prefix.gain_element_sign,
        cce.prefix.gain_element_scale,
    );
    for (target_sample, &coupling_sample) in target.iter_mut().zip(coupling) {
        let scaled = (coupling_sample as f32 * scale).round() as i64;
        let mixed = *target_sample as i64 + scaled;
        *target_sample = mixed.clamp(i32::MIN as i64 + 1, i32::MAX as i64) as FixpDbl;
    }
    Ok(())
}

fn apply_tns_to_staged_spectra(
    staged: &mut [StagedAacLcElement],
    sampling_frequency_index: u8,
) -> Result<(), DecodeError> {
    let apply = |stream: &mut DecodedChannelStream| -> Result<(), DecodeError> {
        let sfb = aac_lc_band_offsets_for_ics(sampling_frequency_index, &stream.ics)?;
        stream.tns_data.apply_f32(
            &mut stream.spectrum,
            sfb.offsets,
            stream.ics.max_sfb as usize,
        )?;
        Ok(())
    };
    for element in staged {
        match element {
            StagedAacLcElement::Single { spectra, .. } => apply(&mut spectra.stream)?,
            StagedAacLcElement::Pair { spectra, .. } => {
                apply(&mut spectra.left)?;
                apply(&mut spectra.right)?;
            }
        }
    }
    Ok(())
}

fn apply_tns_to_staged_fixed_spectra(
    staged: &mut [StagedAacLcElementFixed],
    sampling_frequency_index: u8,
) -> Result<(), DecodeError> {
    let apply = |stream: &mut DecodedChannelStreamFixed| -> Result<(), DecodeError> {
        let sfb = aac_lc_band_offsets_for_ics(sampling_frequency_index, &stream.ics)?;
        stream.tns_data.apply_fixed(
            &mut stream.spectrum,
            sfb.offsets,
            stream.ics.max_sfb as usize,
        )?;
        Ok(())
    };
    for element in staged {
        match element {
            StagedAacLcElementFixed::Single { spectra, .. } => apply(&mut spectra.stream)?,
            StagedAacLcElementFixed::Pair { spectra, .. } => {
                apply(&mut spectra.left)?;
                apply(&mut spectra.right)?;
            }
        }
    }
    Ok(())
}

fn apply_cce_to_staged_frequency_spectra(
    staged: &mut [StagedAacLcElement],
    cce: &DecodedCouplingChannelElement,
    sampling_frequency_index: u8,
) -> Result<(), DecodeError> {
    if !cce.prefix.uses_frequency_coupling() {
        return Err(DecodeError::TimeDomainCouplingUnsupported);
    }
    let mut gain_index = 0usize;
    for target in &cce.prefix.targets {
        for channel in target_channel_indices(target) {
            for element in staged.iter_mut() {
                match element {
                    StagedAacLcElement::Single {
                        element_id,
                        element_instance_tag,
                        spectra,
                        ..
                    } if !target.is_cpe
                        && *element_instance_tag == target.tag_select
                        && channel == 0
                        && *element_id != ElementId::ChannelPair =>
                    {
                        apply_frequency_coupling_to_spectrum_at_rate(
                            &mut spectra.stream.spectrum,
                            cce,
                            gain_index,
                            sampling_frequency_index,
                        )?;
                    }
                    StagedAacLcElement::Pair {
                        element_instance_tag,
                        spectra,
                        ..
                    } if target.is_cpe && *element_instance_tag == target.tag_select => {
                        if channel == 0 {
                            apply_frequency_coupling_to_spectrum_at_rate(
                                &mut spectra.left.spectrum,
                                cce,
                                gain_index,
                                sampling_frequency_index,
                            )?;
                        } else {
                            apply_frequency_coupling_to_spectrum_at_rate(
                                &mut spectra.right.spectrum,
                                cce,
                                gain_index,
                                sampling_frequency_index,
                            )?;
                        }
                    }
                    _ => {}
                }
            }
            gain_index += 1;
        }
    }
    Ok(())
}

fn apply_staged_frequency_couplings(
    staged: &mut [StagedAacLcElement],
    coupled: &[DecodedCouplingChannelElement],
    point: CouplingPoint,
    sampling_frequency_index: u8,
) -> Result<(), DecodeError> {
    for cce in coupled
        .iter()
        .filter(|cce| cce.prefix.coupling_point() == point)
    {
        apply_cce_to_staged_frequency_spectra(staged, cce, sampling_frequency_index)?;
    }
    Ok(())
}

fn apply_cce_to_staged_fixed_frequency_spectra(
    staged: &mut [StagedAacLcElementFixed],
    cce: &DecodedCouplingChannelElementFixed,
    sampling_frequency_index: u8,
) -> Result<(), DecodeError> {
    if !cce.prefix.uses_frequency_coupling() {
        return Err(DecodeError::TimeDomainCouplingUnsupported);
    }
    let mut gain_index = 0usize;
    for target in &cce.prefix.targets {
        for channel in target_channel_indices(target) {
            for element in staged.iter_mut() {
                match element {
                    StagedAacLcElementFixed::Single {
                        element_id,
                        element_instance_tag,
                        spectra,
                        ..
                    } if !target.is_cpe
                        && *element_instance_tag == target.tag_select
                        && channel == 0
                        && *element_id != ElementId::ChannelPair =>
                    {
                        apply_frequency_coupling_to_fixed_spectrum_at_rate(
                            &mut spectra.stream.spectrum,
                            cce,
                            gain_index,
                            sampling_frequency_index,
                        )?;
                    }
                    StagedAacLcElementFixed::Pair {
                        element_instance_tag,
                        spectra,
                        ..
                    } if target.is_cpe && *element_instance_tag == target.tag_select => {
                        if channel == 0 {
                            apply_frequency_coupling_to_fixed_spectrum_at_rate(
                                &mut spectra.left.spectrum,
                                cce,
                                gain_index,
                                sampling_frequency_index,
                            )?;
                        } else {
                            apply_frequency_coupling_to_fixed_spectrum_at_rate(
                                &mut spectra.right.spectrum,
                                cce,
                                gain_index,
                                sampling_frequency_index,
                            )?;
                        }
                    }
                    _ => {}
                }
            }
            gain_index += 1;
        }
    }
    Ok(())
}

fn apply_staged_fixed_frequency_couplings(
    staged: &mut [StagedAacLcElementFixed],
    coupled: &[DecodedCouplingChannelElementFixed],
    point: CouplingPoint,
    sampling_frequency_index: u8,
) -> Result<(), DecodeError> {
    for cce in coupled
        .iter()
        .filter(|cce| cce.prefix.coupling_point() == point)
    {
        apply_cce_to_staged_fixed_frequency_spectra(staged, cce, sampling_frequency_index)?;
    }
    Ok(())
}

fn synthetic_long_ics() -> IcsInfo {
    IcsInfo {
        window_sequence: WindowSequence::OnlyLong,
        window_shape: WindowShape::Sine,
        max_sfb: 0,
        total_sfb: IcsLimits::AAC_LC_MAX.long_sfb,
        predictor_data_present: false,
        scale_factor_grouping: 0,
        window_group_lengths: vec![1],
        bits_read: 0,
    }
}

fn synthetic_single_channel_side_info() -> SingleChannelElementSideInfo {
    SingleChannelElementSideInfo {
        id: ElementId::SingleChannel,
        element_instance_tag: 0,
        global_gain: 0,
        ics: synthetic_long_ics(),
        bits_read: 0,
    }
}

fn synthetic_channel_stream() -> DecodedChannelStream {
    DecodedChannelStream {
        global_gain: 0,
        ics: synthetic_long_ics(),
        section_data: SectionData {
            sections: Vec::new(),
            codebooks: vec![Vec::new()],
            bits_read: 0,
        },
        scalefactors: ScalefactorData {
            values: vec![Vec::new()],
        },
        pulse_data: PulseData::absent(),
        tns_data: TnsData::absent(1),
        spectral: SpectralData {
            windows: Vec::new(),
        },
        spectrum: InverseQuantizedSpectrum {
            windows: Vec::new(),
        },
    }
}

fn synthetic_channel_pair_spectra() -> DecodedChannelPairSpectra {
    DecodedChannelPairSpectra {
        prefix: ChannelPairElementSideInfoPrefix {
            element_instance_tag: 0,
            common_window: false,
            shared_ics: None,
            bits_read: 0,
        },
        ms_stereo: Some(MsStereoData {
            mask_present: MsMaskPresent::None,
            used: Vec::new(),
        }),
        left: synthetic_channel_stream(),
        right: synthetic_channel_stream(),
        right_channel_start_bit: 0,
        bits_read: 0,
    }
}

fn decode_channel_stream_from_reader(
    reader: &mut BitReader<'_>,
    sampling_frequency_index: u8,
    frame_length: usize,
    shared_ics: Option<&IcsInfo>,
) -> Result<DecodedChannelStream, DecodeError> {
    let global_gain = reader.read_u8(8)?;
    let ics = match shared_ics {
        Some(ics) => ics.clone(),
        None => IcsInfo::parse_aac_lc(reader, IcsLimits::AAC_LC_MAX)?,
    };
    decode_channel_stream_after_global_gain(
        reader,
        sampling_frequency_index,
        frame_length,
        global_gain,
        ics,
    )
}

fn decode_channel_stream_fixed_bridge_from_reader(
    reader: &mut BitReader<'_>,
    sampling_frequency_index: u8,
    frame_length: usize,
    shared_ics: Option<&IcsInfo>,
    pns_random: &mut PnsRandomState,
    apply_noise_and_tns: bool,
) -> Result<DecodedChannelStreamFixed, DecodeError> {
    let global_gain = reader.read_u8(8)?;
    let ics = match shared_ics {
        Some(ics) => ics.clone(),
        None => IcsInfo::parse_aac_lc(reader, IcsLimits::AAC_LC_MAX)?,
    };
    decode_channel_stream_fixed_bridge_after_global_gain(
        reader,
        sampling_frequency_index,
        frame_length,
        global_gain,
        ics,
        pns_random,
        apply_noise_and_tns,
    )
}

fn decode_channel_stream_after_global_gain(
    reader: &mut BitReader<'_>,
    sampling_frequency_index: u8,
    frame_length: usize,
    global_gain: u8,
    ics: IcsInfo,
) -> Result<DecodedChannelStream, DecodeError> {
    let sfb = aac_band_offsets_for_ics(sampling_frequency_index, &ics, frame_length)?;
    let section_data = SectionData::parse_aac_lc(reader, &ics)?;
    let scalefactor_plan = ScalefactorPlan::from_section_data(&section_data)?;
    let scalefactors = scalefactor_plan.decode_from_bitstream(reader, global_gain)?;
    let pulse_data = PulseData::parse_aac_lc(reader, &ics, sfb.offsets, sfb.granule_length)?;
    let tns_data = TnsData::parse_aac_lc(reader, &ics)?;
    if reader.read_bool()? {
        return Err(DecodeError::GainControlUnsupported);
    }
    let mut spectral =
        decode_spectral_data(reader, &ics, &section_data, sfb.offsets, sfb.granule_length)?;
    pulse_data.apply_to_spectral(&mut spectral, sfb.offsets)?;
    let spectrum = inverse_quantize_spectrum_f32(&spectral, &scalefactors, &ics, sfb)?;

    Ok(DecodedChannelStream {
        global_gain,
        ics,
        section_data,
        scalefactors,
        pulse_data,
        tns_data,
        spectral,
        spectrum,
    })
}

fn decode_channel_stream_fixed_bridge_after_global_gain(
    reader: &mut BitReader<'_>,
    sampling_frequency_index: u8,
    frame_length: usize,
    global_gain: u8,
    ics: IcsInfo,
    pns_random: &mut PnsRandomState,
    apply_noise_and_tns: bool,
) -> Result<DecodedChannelStreamFixed, DecodeError> {
    let sfb = aac_band_offsets_for_ics(sampling_frequency_index, &ics, frame_length)?;
    let section_data = SectionData::parse_aac_lc(reader, &ics)?;
    let scalefactor_plan = ScalefactorPlan::from_section_data(&section_data)?;
    let scalefactors = scalefactor_plan.decode_from_bitstream(reader, global_gain)?;
    let pulse_data = PulseData::parse_aac_lc(reader, &ics, sfb.offsets, sfb.granule_length)?;
    let tns_data = TnsData::parse_aac_lc(reader, &ics)?;
    if reader.read_bool()? {
        return Err(DecodeError::GainControlUnsupported);
    }
    let mut spectral =
        decode_spectral_data(reader, &ics, &section_data, sfb.offsets, sfb.granule_length)?;
    pulse_data.apply_to_spectral(&mut spectral, sfb.offsets)?;
    let mut spectrum =
        inverse_quantize_spectrum_fixed_block_scaled(&spectral, &scalefactors, &ics, sfb)?;
    if apply_noise_and_tns {
        apply_pns_fixed(
            &mut spectrum,
            &ics,
            sfb.offsets,
            &section_data,
            &scalefactors,
            pns_random,
        )?;
        tns_data.apply_fixed(&mut spectrum, sfb.offsets, ics.max_sfb as usize)?;
    }

    Ok(DecodedChannelStreamFixed {
        global_gain,
        ics,
        section_data,
        scalefactors,
        pulse_data,
        tns_data,
        spectral,
        spectrum,
    })
}

fn apply_channel_pair_pns_and_tns(
    decoded: &mut DecodedChannelPairSpectra,
    sampling_frequency_index: u8,
    frame_length: usize,
    pns_random: &mut PnsRandomState,
) -> Result<(), DecodeError> {
    apply_channel_pair_pns(decoded, sampling_frequency_index, frame_length, pns_random)?;
    apply_channel_pair_tns(decoded, sampling_frequency_index, frame_length)
}

fn apply_channel_pair_pns(
    decoded: &mut DecodedChannelPairSpectra,
    sampling_frequency_index: u8,
    frame_length: usize,
    pns_random: &mut PnsRandomState,
) -> Result<(), DecodeError> {
    let sfb = aac_band_offsets_for_ics(sampling_frequency_index, &decoded.left.ics, frame_length)?;
    apply_pns_pair_f32(
        &mut decoded.left.spectrum,
        &mut decoded.right.spectrum,
        &decoded.left.ics,
        sfb.offsets,
        &decoded.left.section_data,
        &decoded.right.section_data,
        &decoded.left.scalefactors,
        &decoded.right.scalefactors,
        decoded.ms_stereo.as_ref(),
        pns_random,
    )?;
    Ok(())
}

fn apply_channel_pair_tns(
    decoded: &mut DecodedChannelPairSpectra,
    sampling_frequency_index: u8,
    frame_length: usize,
) -> Result<(), DecodeError> {
    let sfb = aac_band_offsets_for_ics(sampling_frequency_index, &decoded.left.ics, frame_length)?;
    decoded.left.tns_data.apply_f32(
        &mut decoded.left.spectrum,
        sfb.offsets,
        decoded.left.ics.max_sfb as usize,
    )?;
    decoded.right.tns_data.apply_f32(
        &mut decoded.right.spectrum,
        sfb.offsets,
        decoded.right.ics.max_sfb as usize,
    )?;
    Ok(())
}

fn apply_channel_pair_pns_and_tns_fixed_bridge(
    decoded: &mut DecodedChannelPairSpectraFixed,
    sampling_frequency_index: u8,
    frame_length: usize,
    pns_random: &mut PnsRandomState,
) -> Result<(), DecodeError> {
    apply_channel_pair_pns_fixed_bridge(
        decoded,
        sampling_frequency_index,
        frame_length,
        pns_random,
    )?;
    apply_channel_pair_tns_fixed_bridge(decoded, sampling_frequency_index, frame_length)
}

fn apply_channel_pair_pns_fixed_bridge(
    decoded: &mut DecodedChannelPairSpectraFixed,
    sampling_frequency_index: u8,
    frame_length: usize,
    pns_random: &mut PnsRandomState,
) -> Result<(), DecodeError> {
    let sfb = aac_band_offsets_for_ics(sampling_frequency_index, &decoded.left.ics, frame_length)?;
    apply_pns_pair_fixed(
        &mut decoded.left.spectrum,
        &mut decoded.right.spectrum,
        &decoded.left.ics,
        sfb.offsets,
        &decoded.left.section_data,
        &decoded.right.section_data,
        &decoded.left.scalefactors,
        &decoded.right.scalefactors,
        decoded.ms_stereo.as_ref(),
        pns_random,
    )?;
    Ok(())
}

fn apply_channel_pair_tns_fixed_bridge(
    decoded: &mut DecodedChannelPairSpectraFixed,
    sampling_frequency_index: u8,
    frame_length: usize,
) -> Result<(), DecodeError> {
    let sfb = aac_band_offsets_for_ics(sampling_frequency_index, &decoded.left.ics, frame_length)?;
    decoded.left.tns_data.apply_fixed(
        &mut decoded.left.spectrum,
        sfb.offsets,
        decoded.left.ics.max_sfb as usize,
    )?;
    decoded.right.tns_data.apply_fixed(
        &mut decoded.right.spectrum,
        sfb.offsets,
        decoded.right.ics.max_sfb as usize,
    )?;
    Ok(())
}

fn validate_zero_trailing_bits(reader: &BitReader<'_>) -> Result<(), DecodeError> {
    if reader.remaining_bits_are_zero() {
        Ok(())
    } else {
        Err(DecodeError::NonZeroTrailingBits(reader.remaining_bits()))
    }
}

fn prepare_fixed_concealment_spectrum(
    spectrum: &mut FixedInverseQuantizedSpectrum,
    consecutive_losses: usize,
    phase: &mut u32,
) {
    let attenuation = match consecutive_losses {
        0 => i32::MAX,
        1..=6 => fixed_concealment_factor(consecutive_losses),
        _ => 0,
    };
    for window in &mut spectrum.windows {
        for coefficient in window {
            *phase = phase.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let value = mul_q31(*coefficient, attenuation);
            *coefficient = if (*phase & 0x8000_0000) != 0 {
                value.saturating_neg()
            } else {
                value
            };
        }
    }
}

fn prepare_f32_concealment_spectrum(
    spectrum: &mut InverseQuantizedSpectrum,
    consecutive_losses: usize,
    phase: &mut u32,
) {
    let attenuation = match consecutive_losses {
        0 => 1.0,
        1..=6 => 2.0f32.powf(-(consecutive_losses as f32) * 0.5),
        _ => 0.0,
    };
    for window in &mut spectrum.windows {
        for coefficient in window {
            *phase = phase.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *coefficient *= attenuation;
            if (*phase & 0x8000_0000) != 0 {
                *coefficient = -*coefficient;
            }
        }
    }
}

fn randomize_fixed_spectrum_signs(spectrum: &mut FixedInverseQuantizedSpectrum, phase: &mut u32) {
    for window in &mut spectrum.windows {
        for coefficient in window {
            *phase = phase.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            if (*phase & 0x8000_0000) != 0 {
                *coefficient = coefficient.saturating_neg();
            }
        }
    }
}

fn randomize_f32_spectrum_signs(spectrum: &mut InverseQuantizedSpectrum, phase: &mut u32) {
    for window in &mut spectrum.windows {
        for coefficient in window {
            *phase = phase.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            if (*phase & 0x8000_0000) != 0 {
                *coefficient = -*coefficient;
            }
        }
    }
}

fn fixed_concealment_factor(exponent: usize) -> i32 {
    const FADE_FACTOR_Q31: i32 = 1_518_500_250; // round((1/sqrt(2)) * 2^31)
    let mut value = i32::MAX;
    for _ in 0..exponent {
        value = mul_q31(value, FADE_FACTOR_Q31);
    }
    value
}

fn apply_fixed_concealment_recovery_fade(channels: &mut [Vec<FixpDbl>], fade_in_remaining: usize) {
    let start = fixed_concealment_factor(fade_in_remaining);
    let stop = fixed_concealment_factor(fade_in_remaining.saturating_sub(1));
    for channel in channels {
        let denominator = channel.len().saturating_sub(1).max(1) as i64;
        for (index, sample) in channel.iter_mut().enumerate() {
            let factor = start as i64 + (stop as i64 - start as i64) * index as i64 / denominator;
            *sample = mul_q31(*sample, factor as i32);
        }
    }
}

fn apply_f32_concealment_recovery_fade(channels: &mut [Vec<f32>], fade_in_remaining: usize) {
    let start = 2.0f32.powf(-(fade_in_remaining as f32) * 0.5);
    let stop = 2.0f32.powf(-(fade_in_remaining.saturating_sub(1) as f32) * 0.5);
    for channel in channels {
        let denominator = channel.len().saturating_sub(1).max(1) as f32;
        for (index, sample) in channel.iter_mut().enumerate() {
            let ratio = index as f32 / denominator;
            *sample *= start + (stop - start) * ratio;
        }
    }
}

fn consume_raw_data_block_terminator(reader: &mut BitReader<'_>) -> Result<(), DecodeError> {
    if reader.remaining_bits() < 3 {
        return Err(DecodeError::RawDataBlockTerminatorMissing);
    }
    if ElementId::from_bits(reader.read_u8(3)?) != ElementId::End {
        return Err(DecodeError::RawDataBlockTerminatorMissing);
    }
    Ok(())
}

fn validate_adts_aac_lc_configuration(
    decoder: &AacLcDecoder,
    header: crate::adts::AdtsHeader,
) -> Result<(), DecodeError> {
    if header.profile + 1 != decoder.audio_object_type {
        return Err(DecodeError::UnsupportedAudioObjectType(header.profile + 1));
    }
    if header.sampling_frequency_index != decoder.sampling_frequency_index {
        return Err(DecodeError::AdtsConfigChanged);
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    Adts(AdtsError),
    AdtsConfigChanged,
    AncillaryBufferTooSmall { capacity: usize, required: usize },
    Asc(AscError),
    Bit(BitError),
    BandwiseCouplingGainUnsupported,
    ChannelConfigurationMismatch { expected: usize, actual: usize },
    CouplingGainApplicationUnsupported,
    CouplingLayoutMismatch,
    Drc(DrcError),
    Filterbank(FilterbankError),
    ErrorResilienceUnsupported,
    GainControlUnsupported,
    LtpUnsupported,
    Huffman(HuffmanError),
    Ics(IcsError),
    Inverse(DecodeInverseError),
    LdFilterbank(LdFilterbankError),
    LdSbr(LdSbrError),
    LdSbrProcessing(LdSbrProcessingError),
    MpsSpatialConfiguration,
    MpsSpatialFrame,
    NoAudioElement,
    NoConcealmentReference,
    ConcealmentInterpolation(SpectralInterpolationError),
    NonZeroTrailingBits(usize),
    Pns(PnsError),
    Ps(PsError),
    Pulse(PulseError),
    Raw(RawError),
    Hcr(HcrError),
    Rvlc(RvlcError),
    Sbr(SbrError),
    SbrPayloadLayoutMismatch,
    RawDataBlockTerminatorMissing,
    Scalefactor(ScalefactorError),
    Section(SectionError),
    Sfb(SfbError),
    Spectral(SpectralError),
    Stereo(StereoError),
    Tns(TnsError),
    UnsupportedAudioObjectType(u8),
    UnsupportedAncillaryDataElementVersion(u8),
    UnsupportedChannelConfiguration(u8),
    UnsupportedCouplingChannelElement(CouplingChannelElementPrefix),
    UnsupportedFirstElement(ElementId),
    UnsupportedFrameLength(usize),
    UnsupportedRawBlocksInAdtsFrame(u8),
    UnsupportedSamplingFrequencyIndex(u8),
    TimeDomainCouplingUnsupported,
    TooManyAncillaryElements,
}

impl DecodeError {
    /// Whether decoding can succeed after appending more bits to the same
    /// input. Transport buffers use this to distinguish an incomplete access
    /// unit from malformed syntax without committing partially mutated DSP
    /// state.
    pub fn is_unexpected_eof(&self) -> bool {
        fn bit(error: &BitError) -> bool {
            matches!(error, BitError::UnexpectedEof { .. })
        }
        fn huffman(error: &HuffmanError) -> bool {
            matches!(error, HuffmanError::Bit(error) if bit(error))
        }
        fn ics(error: &IcsError) -> bool {
            matches!(error, IcsError::Bit(error) if bit(error))
        }
        fn section(error: &SectionError) -> bool {
            matches!(error, SectionError::Bit(error) if bit(error))
        }
        fn scalefactor(error: &ScalefactorError) -> bool {
            matches!(error, ScalefactorError::Huffman(error) if huffman(error))
        }
        fn raw(error: &RawError) -> bool {
            match error {
                RawError::Bit(error) => bit(error),
                RawError::Asc(AscError::UnexpectedEof { .. }) => true,
                RawError::Ics(error) => ics(error),
                RawError::Section(error) => section(error),
                RawError::Scalefactor(error) => scalefactor(error),
                _ => false,
            }
        }
        fn spectral(error: &SpectralError) -> bool {
            match error {
                SpectralError::Bit(error) => bit(error),
                SpectralError::Huffman(error) => huffman(error),
                _ => false,
            }
        }

        match self {
            Self::Asc(AscError::UnexpectedEof { .. }) => true,
            Self::Bit(error) => bit(error),
            Self::Huffman(error) => huffman(error),
            Self::Ics(error) => ics(error),
            Self::Pulse(PulseError::Bit(error)) => bit(error),
            Self::Raw(error) => raw(error),
            Self::Hcr(HcrError::Bit(error)) => bit(error),
            Self::Hcr(HcrError::Spectral(error)) => spectral(error),
            Self::Rvlc(RvlcError::Bit(error)) => bit(error),
            Self::Scalefactor(error) => scalefactor(error),
            Self::Section(error) => section(error),
            Self::Spectral(error) => spectral(error),
            Self::Stereo(StereoError::Bit(error)) => bit(error),
            Self::Tns(TnsError::Bit(error)) => bit(error),
            Self::NoAudioElement => true,
            Self::RawDataBlockTerminatorMissing => true,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeInverseError {
    Sfb(SfbError),
    LayoutMismatch,
}

impl From<BitError> for DecodeError {
    fn from(value: BitError) -> Self {
        Self::Bit(value)
    }
}

impl From<HuffmanError> for DecodeError {
    fn from(value: HuffmanError) -> Self {
        Self::Huffman(value)
    }
}

impl From<AdtsError> for DecodeError {
    fn from(value: AdtsError) -> Self {
        Self::Adts(value)
    }
}

impl From<AscError> for DecodeError {
    fn from(value: AscError) -> Self {
        Self::Asc(value)
    }
}

impl From<FilterbankError> for DecodeError {
    fn from(value: FilterbankError) -> Self {
        Self::Filterbank(value)
    }
}

impl From<DrcError> for DecodeError {
    fn from(value: DrcError) -> Self {
        Self::Drc(value)
    }
}

impl From<SpectralInterpolationError> for DecodeError {
    fn from(value: SpectralInterpolationError) -> Self {
        Self::ConcealmentInterpolation(value)
    }
}

impl From<IcsError> for DecodeError {
    fn from(value: IcsError) -> Self {
        Self::Ics(value)
    }
}

impl From<InverseQuantError> for DecodeError {
    fn from(value: InverseQuantError) -> Self {
        Self::Inverse(match value {
            InverseQuantError::Sfb(err) => DecodeInverseError::Sfb(err),
            InverseQuantError::LayoutMismatch => DecodeInverseError::LayoutMismatch,
        })
    }
}

impl From<LdFilterbankError> for DecodeError {
    fn from(value: LdFilterbankError) -> Self {
        Self::LdFilterbank(value)
    }
}

impl From<LdSbrError> for DecodeError {
    fn from(value: LdSbrError) -> Self {
        Self::LdSbr(value)
    }
}

impl From<LdSbrProcessingError> for DecodeError {
    fn from(value: LdSbrProcessingError) -> Self {
        Self::LdSbrProcessing(value)
    }
}

impl From<SbrError> for DecodeError {
    fn from(value: SbrError) -> Self {
        Self::Sbr(value)
    }
}

impl From<PnsError> for DecodeError {
    fn from(value: PnsError) -> Self {
        Self::Pns(value)
    }
}

impl From<PsError> for DecodeError {
    fn from(value: PsError) -> Self {
        Self::Ps(value)
    }
}

impl From<PulseError> for DecodeError {
    fn from(value: PulseError) -> Self {
        Self::Pulse(value)
    }
}

impl From<RawError> for DecodeError {
    fn from(value: RawError) -> Self {
        Self::Raw(value)
    }
}

impl From<HcrError> for DecodeError {
    fn from(value: HcrError) -> Self {
        Self::Hcr(value)
    }
}

impl From<RvlcError> for DecodeError {
    fn from(value: RvlcError) -> Self {
        Self::Rvlc(value)
    }
}

impl From<ScalefactorError> for DecodeError {
    fn from(value: ScalefactorError) -> Self {
        Self::Scalefactor(value)
    }
}

impl From<SectionError> for DecodeError {
    fn from(value: SectionError) -> Self {
        Self::Section(value)
    }
}

impl From<SfbError> for DecodeError {
    fn from(value: SfbError) -> Self {
        Self::Sfb(value)
    }
}

impl From<SpectralError> for DecodeError {
    fn from(value: SpectralError) -> Self {
        Self::Spectral(value)
    }
}

impl From<StereoError> for DecodeError {
    fn from(value: StereoError) -> Self {
        Self::Stereo(value)
    }
}

impl From<TnsError> for DecodeError {
    fn from(value: TnsError) -> Self {
        Self::Tns(value)
    }
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Adts(err) => err.fmt(f),
            Self::AdtsConfigChanged => write!(f, "ADTS frame configuration changed"),
            Self::AncillaryBufferTooSmall { capacity, required } => write!(
                f,
                "AAC ancillary data needs {required} bytes, configured capacity is {capacity}"
            ),
            Self::Asc(err) => err.fmt(f),
            Self::BandwiseCouplingGainUnsupported => {
                write!(
                    f,
                    "AAC CCE bandwise coupling gain application is unsupported"
                )
            }
            Self::Bit(err) => err.fmt(f),
            Self::ChannelConfigurationMismatch { expected, actual } => write!(
                f,
                "AAC channel configuration expects {expected} channel(s), decoded {actual}"
            ),
            Self::CouplingGainApplicationUnsupported => {
                write!(
                    f,
                    "AAC CCE non-zero coupling gain application is unsupported"
                )
            }
            Self::CouplingLayoutMismatch => write!(f, "AAC CCE spectrum layout mismatch"),
            Self::ConcealmentInterpolation(err) => err.fmt(f),
            Self::Filterbank(err) => err.fmt(f),
            Self::Drc(err) => err.fmt(f),
            Self::ErrorResilienceUnsupported => {
                write!(f, "AAC error-resilience tool payload is unsupported")
            }
            Self::GainControlUnsupported => write!(f, "AAC gain_control_data is unsupported"),
            Self::LtpUnsupported => write!(f, "AAC LTP data is unsupported"),
            Self::Huffman(err) => err.fmt(f),
            Self::Ics(err) => err.fmt(f),
            Self::Inverse(err) => write!(f, "inverse quantization error: {err:?}"),
            Self::LdFilterbank(err) => err.fmt(f),
            Self::LdSbr(err) => err.fmt(f),
            Self::LdSbrProcessing(err) => write!(f, "LD-SBR processing error: {err:?}"),
            Self::MpsSpatialConfiguration => write!(f, "invalid AAC-ELD MPEG Surround configuration"),
            Self::MpsSpatialFrame => write!(f, "invalid AAC-ELD MPEG Surround frame"),
            Self::Sbr(err) => err.fmt(f),
            Self::SbrPayloadLayoutMismatch => write!(f, "ordinary SBR payload layout mismatch"),
            Self::NoAudioElement => {
                write!(f, "AAC raw_data_block contains no decodable audio element")
            }
            Self::NoConcealmentReference => {
                write!(f, "AAC concealment has no previously decoded spectral frame")
            }
            Self::NonZeroTrailingBits(bits) => write!(
                f,
                "AAC raw_data_block has non-zero data in {bits} trailing bit(s) after decoded payload"
            ),
            Self::Pns(err) => err.fmt(f),
            Self::Ps(err) => err.fmt(f),
            Self::Pulse(err) => err.fmt(f),
            Self::Raw(err) => err.fmt(f),
            Self::Hcr(err) => err.fmt(f),
            Self::Rvlc(err) => err.fmt(f),
            Self::RawDataBlockTerminatorMissing => {
                write!(f, "AAC raw_data_block is missing its ID_END terminator")
            }
            Self::Scalefactor(err) => err.fmt(f),
            Self::Section(err) => err.fmt(f),
            Self::Sfb(err) => err.fmt(f),
            Self::Spectral(err) => err.fmt(f),
            Self::Stereo(err) => err.fmt(f),
            Self::Tns(err) => err.fmt(f),
            Self::UnsupportedAudioObjectType(aot) => {
                write!(f, "unsupported AAC audio object type {aot}")
            }
            Self::UnsupportedAncillaryDataElementVersion(version) => write!(
                f,
                "unsupported AAC ancillary data-element version {version}"
            ),
            Self::UnsupportedChannelConfiguration(config) => {
                write!(f, "unsupported AAC channel configuration {config}")
            }
            Self::UnsupportedCouplingChannelElement(prefix) => write!(
                f,
                "unsupported AAC coupling channel element tag {} targeting {} element(s)",
                prefix.element_instance_tag,
                prefix.targets.len()
            ),
            Self::UnsupportedFirstElement(id) => {
                write!(f, "unsupported AAC raw_data_block first element {id:?}")
            }
            Self::UnsupportedFrameLength(length) => {
                write!(f, "unsupported AAC frame length {length}")
            }
            Self::UnsupportedRawBlocksInAdtsFrame(count) => write!(
                f,
                "unsupported ADTS raw_data_block count {}, expected 0",
                *count as usize + 1
            ),
            Self::UnsupportedSamplingFrequencyIndex(index) => {
                write!(f, "unsupported AAC sampling frequency index {index}")
            }
            Self::TimeDomainCouplingUnsupported => {
                write!(f, "AAC CCE time-domain coupling application is unsupported")
            }
            Self::TooManyAncillaryElements => {
                write!(f, "AAC frame contains more than seven ancillary data elements")
            }
        }
    }
}

impl std::error::Error for DecodeError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adts::AdtsHeader;
    use crate::asc::{AudioSpecificConfigExtension, LdSbrHeader, ProgramElement};
    use crate::bits::BitWriter;
    use crate::drc::{
        ChannelLayout, DrcCharacteristic, DrcCoefficients, DrcInstruction, GainBand,
        GainCodingProfile, GainInterpolationType, GainModification, GainNode, GainSequence,
        GainSet,
    };
    use crate::filterbank::LongBlockFilterbank;
    use crate::ics::{WindowSequence, WindowShape};
    use crate::ld_sbr::{encode_sbr_huffman, LdSbrFrequencyTables, SbrHuffmanBook};
    use crate::raw::ElementId;
    use crate::section::{INTENSITY_HCB, NOISE_HCB, ZERO_HCB};
    use crate::stereo::MsMaskPresent;
    use crate::tns::{TnsDirection, TnsFilter};

    #[test]
    fn decode_error_conversions_and_nested_eof_classification_are_total() {
        macro_rules! conversion {
            ($value:expr, $pattern:pat) => {
                assert!(matches!(DecodeError::from($value), $pattern));
            };
        }

        conversion!(
            BitError::UnexpectedEof {
                needed_bits: 1,
                remaining_bits: 0,
            },
            DecodeError::Bit(_)
        );
        conversion!(HuffmanError::InvalidCodebook(12), DecodeError::Huffman(_));
        conversion!(AdtsError::InvalidProfile(4), DecodeError::Adts(_));
        conversion!(AscError::InvalidAudioObjectType(0), DecodeError::Asc(_));
        conversion!(
            FilterbankError::InvalidFrameLength(0),
            DecodeError::Filterbank(_)
        );
        conversion!(DrcError::InvalidBandCount(0), DecodeError::Drc(_));
        conversion!(
            SpectralInterpolationError::LayoutMismatch,
            DecodeError::ConcealmentInterpolation(_)
        );
        conversion!(IcsError::PredictionUnsupported, DecodeError::Ics(_));
        conversion!(
            InverseQuantError::Sfb(SfbError::UnsupportedFrameLength(1)),
            DecodeError::Inverse(DecodeInverseError::Sfb(_))
        );
        conversion!(
            InverseQuantError::LayoutMismatch,
            DecodeError::Inverse(DecodeInverseError::LayoutMismatch)
        );
        conversion!(
            LdFilterbankError::UnsupportedFrameLength(1),
            DecodeError::LdFilterbank(_)
        );
        conversion!(LdSbrError::UnexpectedEof, DecodeError::LdSbr(_));
        conversion!(
            LdSbrProcessingError::MissingRightChannel,
            DecodeError::LdSbrProcessing(_)
        );
        conversion!(SbrError::InvalidGrid, DecodeError::Sbr(_));
        conversion!(PnsError::LayoutMismatch, DecodeError::Pns(_));
        conversion!(PsError::InvalidHuffmanCodeword, DecodeError::Ps(_));
        conversion!(PulseError::PulseOnShortWindow, DecodeError::Pulse(_));
        conversion!(RawError::LfeMayNotUseShortWindow, DecodeError::Raw(_));
        conversion!(HcrError::EmptySegment, DecodeError::Hcr(_));
        conversion!(RvlcError::CodewordTooLong, DecodeError::Rvlc(_));
        conversion!(
            ScalefactorError::RaggedCodebookGrid,
            DecodeError::Scalefactor(_)
        );
        conversion!(SectionError::InvalidCodebook(12), DecodeError::Section(_));
        conversion!(
            SfbError::UnsupportedSamplingFrequencyIndex(15),
            DecodeError::Sfb(_)
        );
        conversion!(SpectralError::InvalidBandOffsets, DecodeError::Spectral(_));
        conversion!(StereoError::LayoutMismatch, DecodeError::Stereo(_));
        conversion!(TnsError::LayoutMismatch, DecodeError::Tns(_));

        let eof = || BitError::UnexpectedEof {
            needed_bits: 1,
            remaining_bits: 0,
        };
        let nested_eof = [
            DecodeError::Asc(AscError::UnexpectedEof {
                needed_bits: 1,
                remaining_bits: 0,
            }),
            DecodeError::Bit(eof()),
            DecodeError::Huffman(HuffmanError::Bit(eof())),
            DecodeError::Ics(IcsError::Bit(eof())),
            DecodeError::Pulse(PulseError::Bit(eof())),
            DecodeError::Raw(RawError::Bit(eof())),
            DecodeError::Raw(RawError::Asc(AscError::UnexpectedEof {
                needed_bits: 1,
                remaining_bits: 0,
            })),
            DecodeError::Raw(RawError::Ics(IcsError::Bit(eof()))),
            DecodeError::Raw(RawError::Section(SectionError::Bit(eof()))),
            DecodeError::Raw(RawError::Scalefactor(ScalefactorError::Huffman(
                HuffmanError::Bit(eof()),
            ))),
            DecodeError::Hcr(HcrError::Bit(eof())),
            DecodeError::Hcr(HcrError::Spectral(SpectralError::Bit(eof()))),
            DecodeError::Hcr(HcrError::Spectral(SpectralError::Huffman(
                HuffmanError::Bit(eof()),
            ))),
            DecodeError::Rvlc(RvlcError::Bit(eof())),
            DecodeError::Scalefactor(ScalefactorError::Huffman(HuffmanError::Bit(eof()))),
            DecodeError::Section(SectionError::Bit(eof())),
            DecodeError::Spectral(SpectralError::Bit(eof())),
            DecodeError::Spectral(SpectralError::Huffman(HuffmanError::Bit(eof()))),
            DecodeError::Stereo(StereoError::Bit(eof())),
            DecodeError::Tns(TnsError::Bit(eof())),
            DecodeError::NoAudioElement,
            DecodeError::RawDataBlockTerminatorMissing,
        ];
        assert!(nested_eof.iter().all(DecodeError::is_unexpected_eof));
        assert!(!DecodeError::Raw(RawError::LfeMayNotUseShortWindow).is_unexpected_eof());
        assert!(!DecodeError::Spectral(SpectralError::InvalidBandOffsets).is_unexpected_eof());
        assert!(!DecodeError::UnsupportedFrameLength(0).is_unexpected_eof());
    }

    #[test]
    fn formats_every_decode_error_variant() {
        let bit = || BitError::UnexpectedEof {
            needed_bits: 1,
            remaining_bits: 0,
        };
        let errors = vec![
            DecodeError::Adts(AdtsError::InvalidProfile(4)),
            DecodeError::AdtsConfigChanged,
            DecodeError::Asc(AscError::InvalidAudioObjectType(0)),
            DecodeError::Bit(bit()),
            DecodeError::BandwiseCouplingGainUnsupported,
            DecodeError::ChannelConfigurationMismatch {
                expected: 2,
                actual: 1,
            },
            DecodeError::CouplingGainApplicationUnsupported,
            DecodeError::CouplingLayoutMismatch,
            DecodeError::Drc(DrcError::InvalidBandCount(0)),
            DecodeError::Filterbank(FilterbankError::InvalidFrameLength(0)),
            DecodeError::ErrorResilienceUnsupported,
            DecodeError::GainControlUnsupported,
            DecodeError::LtpUnsupported,
            DecodeError::Huffman(HuffmanError::InvalidCodebook(12)),
            DecodeError::Ics(IcsError::PredictionUnsupported),
            DecodeError::Inverse(DecodeInverseError::LayoutMismatch),
            DecodeError::LdFilterbank(LdFilterbankError::UnsupportedFrameLength(1)),
            DecodeError::LdSbr(LdSbrError::UnexpectedEof),
            DecodeError::LdSbrProcessing(LdSbrProcessingError::MissingRightChannel),
            DecodeError::NoAudioElement,
            DecodeError::NoConcealmentReference,
            DecodeError::ConcealmentInterpolation(SpectralInterpolationError::LayoutMismatch),
            DecodeError::NonZeroTrailingBits(1),
            DecodeError::Pns(PnsError::LayoutMismatch),
            DecodeError::Ps(PsError::InvalidHuffmanCodeword),
            DecodeError::Pulse(PulseError::PulseOnShortWindow),
            DecodeError::Raw(RawError::LfeMayNotUseShortWindow),
            DecodeError::Hcr(HcrError::EmptySegment),
            DecodeError::Rvlc(RvlcError::CodewordTooLong),
            DecodeError::Sbr(SbrError::InvalidGrid),
            DecodeError::SbrPayloadLayoutMismatch,
            DecodeError::RawDataBlockTerminatorMissing,
            DecodeError::Scalefactor(ScalefactorError::RaggedCodebookGrid),
            DecodeError::Section(SectionError::InvalidCodebook(12)),
            DecodeError::Sfb(SfbError::UnsupportedSamplingFrequencyIndex(15)),
            DecodeError::Spectral(SpectralError::InvalidBandOffsets),
            DecodeError::Stereo(StereoError::LayoutMismatch),
            DecodeError::Tns(TnsError::LayoutMismatch),
            DecodeError::UnsupportedAudioObjectType(1),
            DecodeError::UnsupportedAncillaryDataElementVersion(1),
            DecodeError::UnsupportedChannelConfiguration(0),
            DecodeError::UnsupportedCouplingChannelElement(CouplingChannelElementPrefix {
                element_instance_tag: 0,
                independently_switched: false,
                targets: Vec::new(),
                coupling_domain: false,
                gain_element_sign: false,
                gain_element_scale: 0,
                gain_element_lists: 0,
                bits_read: 0,
            }),
            DecodeError::UnsupportedFirstElement(ElementId::Fill),
            DecodeError::UnsupportedFrameLength(1),
            DecodeError::UnsupportedRawBlocksInAdtsFrame(1),
            DecodeError::UnsupportedSamplingFrequencyIndex(15),
            DecodeError::TimeDomainCouplingUnsupported,
        ];
        assert!(errors.iter().all(|error| !error.to_string().is_empty()));
    }

    fn zero_sce_payload(gain_control_data_present: bool) -> Vec<u8> {
        let mut writer = BitWriter::new();
        write_zero_sce_payload_bits(&mut writer, gain_control_data_present);
        writer.finish()
    }

    fn nonzero_spectral_sce_payload() -> Vec<u8> {
        let mut writer = BitWriter::new();
        writer.write(ElementId::SingleChannel.bits() as u32, 3);
        writer.write(0, 4); // element_instance_tag
        writer.write(100, 8); // global_gain
        writer.write_bool(false); // reserved
        writer.write(WindowSequence::OnlyLong.bits() as u32, 2);
        writer.write_bool(false); // sine
        writer.write(2, 6); // max_sfb
        writer.write_bool(false); // predictor_data_present
        writer.write(1, 4); // section codebook: spectral codebook 1
        writer.write(2, 5); // section length
        writer.write_bool(false); // SCL Huffman delta 60, consumes one bit via pushback
        writer.write_bool(false); // second SCL delta 60
        writer.write(0, 4); // first codebook 1 tuple may be forced to zero by SCL pushback
        writer.write(0b10000, 5); // second codebook 1 tuple carries non-zero data
        writer.write(0, 16); // zero guard bits keep following tool flags absent
        writer.write_bool(false); // pulse_data_present
        writer.write_bool(false); // tns_data_present
        writer.write_bool(false); // gain_control_data_present
        writer.finish()
    }

    fn pce_plus_zero_sce_payload() -> Vec<u8> {
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
        writer.write(ElementId::ProgramConfig.bits() as u32, 3);
        pce.write_to_writer(&mut writer).unwrap();
        write_zero_sce_payload_bits(&mut writer, false);
        writer.finish()
    }

    fn write_zero_sce_payload_bits(writer: &mut BitWriter, gain_control_data_present: bool) {
        write_zero_sce_payload_bits_with_tag(writer, 0, gain_control_data_present);
    }

    fn sbr_huffman_code(book: SbrHuffmanBook, symbol: i8) -> Vec<bool> {
        encode_sbr_huffman(book, symbol).expect("test symbol must exist in the SBR Huffman ROM")
    }

    fn write_zero_sce_payload_bits_with_tag(
        writer: &mut BitWriter,
        element_instance_tag: u8,
        gain_control_data_present: bool,
    ) {
        writer.write(ElementId::SingleChannel.bits() as u32, 3);
        writer.write(element_instance_tag as u32, 4);
        writer.write(100, 8); // global_gain
        writer.write_bool(false); // reserved
        writer.write(0, 2); // ONLY_LONG_SEQUENCE
        writer.write_bool(false); // sine
        writer.write(1, 6); // max_sfb
        writer.write_bool(false); // predictor_data_present
        writer.write(ZERO_HCB as u32, 4); // section codebook
        writer.write(1, 5); // section length
                            // no scalefactor bits for ZERO_HCB
        writer.write_bool(false); // pulse_data_present
        writer.write_bool(false); // tns_data_present
        writer.write_bool(gain_control_data_present);
    }

    fn write_shared_long_ics(writer: &mut BitWriter, max_sfb: u8) {
        writer.write_bool(false); // reserved
        writer.write(0, 2); // ONLY_LONG_SEQUENCE
        writer.write_bool(false); // sine
        writer.write(max_sfb as u32, 6);
        writer.write_bool(false); // predictor_data_present
    }

    fn write_zero_channel_stream(writer: &mut BitWriter, max_sfb: u8) {
        writer.write(100, 8); // global_gain
        writer.write(ZERO_HCB as u32, 4);
        writer.write(max_sfb as u32, 5);
        writer.write_bool(false); // pulse_data_present
        writer.write_bool(false); // tns_data_present
        writer.write_bool(false); // gain_control_data_present
    }

    fn write_zero_independent_channel_stream(writer: &mut BitWriter, max_sfb: u8) {
        writer.write(100, 8); // global_gain
        write_shared_long_ics(writer, max_sfb);
        writer.write(ZERO_HCB as u32, 4);
        writer.write(max_sfb as u32, 5);
        writer.write_bool(false); // pulse_data_present
        writer.write_bool(false); // tns_data_present
        writer.write_bool(false); // gain_control_data_present
    }

    fn zero_cpe_payload(ms_mask_present: u8) -> Vec<u8> {
        let mut writer = BitWriter::new();
        writer.write(ElementId::ChannelPair.bits() as u32, 3);
        writer.write(0, 4); // element_instance_tag
        writer.write_bool(true); // common_window
        write_shared_long_ics(&mut writer, 1);
        writer.write(ms_mask_present as u32, 2);
        write_zero_channel_stream(&mut writer, 1);
        write_zero_channel_stream(&mut writer, 1);
        writer.finish()
    }

    fn correlated_pns_cpe_payload() -> Vec<u8> {
        let mut writer = BitWriter::new();
        writer.write(ElementId::ChannelPair.bits() as u32, 3);
        writer.write(0, 4);
        writer.write_bool(true);
        write_shared_long_ics(&mut writer, 1);
        writer.write(2, 2); // ms_mask_present == all, marks PNS correlation.

        for _ in 0..2 {
            writer.write(100, 8); // global_gain
            writer.write(NOISE_HCB as u32, 4);
            writer.write(1, 5); // one noisy section
                                // First PNS/noise scalefactor uses SCL Huffman in this incremental
                                // model; code 0 decodes to a deterministic table entry and keeps
                                // the test focused on orchestration/correlation.
            writer.write(0, 1);
            writer.write_bool(false); // pulse_data_present
            writer.write_bool(false); // tns_data_present
            writer.write_bool(false); // gain_control_data_present
        }

        writer.finish()
    }

    fn test_ics(max_sfb: u8) -> IcsInfo {
        IcsInfo {
            window_sequence: WindowSequence::OnlyLong,
            window_shape: WindowShape::Sine,
            max_sfb,
            total_sfb: max_sfb,
            predictor_data_present: false,
            scale_factor_grouping: 0,
            window_group_lengths: vec![1],
            bits_read: 0,
        }
    }

    fn test_sections(codebooks: Vec<u8>) -> SectionData {
        SectionData {
            sections: Vec::new(),
            codebooks: vec![codebooks],
            bits_read: 0,
        }
    }

    fn test_stream(
        ics: &IcsInfo,
        section_data: SectionData,
        scalefactors: Vec<i16>,
        spectrum: Vec<f32>,
    ) -> DecodedChannelStream {
        DecodedChannelStream {
            global_gain: 100,
            ics: ics.clone(),
            section_data,
            scalefactors: ScalefactorData {
                values: vec![scalefactors],
            },
            pulse_data: PulseData::absent(),
            tns_data: TnsData::absent(
                ics.window_group_lengths
                    .iter()
                    .map(|&len| len as usize)
                    .sum(),
            ),
            spectral: SpectralData {
                windows: vec![vec![0; spectrum.len()]],
            },
            spectrum: InverseQuantizedSpectrum {
                windows: vec![spectrum],
            },
        }
    }

    #[test]
    fn decodes_zero_single_channel_frame_to_silence() {
        let payload = zero_sce_payload(false);
        let mut filterbank = LongBlockFilterbank::new(1024).unwrap();
        let mut pns_random = PnsRandomState::new(1);
        let decoded =
            decode_aac_lc_single_channel_f32(&payload, 4, &mut filterbank, &mut pns_random)
                .unwrap();

        assert_eq!(decoded.side_info.global_gain, 100);
        assert_eq!(decoded.section_data.codebooks, vec![vec![ZERO_HCB]]);
        assert_eq!(decoded.samples.len(), 1024);
        assert!(decoded.samples.iter().all(|sample| *sample == 0.0));
        assert!(decoded.spectrum.windows[0]
            .iter()
            .all(|sample| *sample == 0.0));
    }

    #[test]
    fn decodes_nonzero_spectral_single_channel_fixture() {
        let payload = nonzero_spectral_sce_payload();
        let mut filterbank = LongBlockFilterbank::new(1024).unwrap();
        let mut pns_random = PnsRandomState::new(1);
        let decoded =
            decode_aac_lc_single_channel_f32(&payload, 4, &mut filterbank, &mut pns_random)
                .unwrap();

        assert_eq!(decoded.section_data.codebooks, vec![vec![1, 1]]);
        assert_eq!(decoded.scalefactors.values.len(), 1);
        assert_eq!(decoded.scalefactors.values[0].len(), 2);
        assert!(decoded.spectral.windows[0]
            .iter()
            .any(|sample| *sample != 0));
        assert!(decoded.spectrum.windows[0]
            .iter()
            .any(|sample| *sample != 0.0));
        assert!(decoded.samples.iter().any(|sample| *sample != 0.0));
    }

    #[test]
    fn configured_uni_drc_is_applied_automatically_to_f32_and_fixed_output() {
        let config = UniDrcConfig {
            sample_rate: Some(44_100),
            channel_layout: ChannelLayout {
                base_channel_count: 1,
                defined_layout: None,
                speaker_positions: Vec::new(),
            },
            downmix_instructions: Vec::new(),
            coefficients: vec![DrcCoefficients {
                drc_location: 1,
                drc_frame_size: Some(1024),
                gain_sequence_count: 1,
                gain_sets: vec![GainSet {
                    coding_profile: GainCodingProfile::Regular,
                    interpolation_type: GainInterpolationType::Linear,
                    full_frame: true,
                    time_alignment: false,
                    time_delta_min: None,
                    drc_band_type: false,
                    bands: vec![GainBand {
                        sequence_index: 0,
                        cicp_characteristic_index: Some(1),
                        characteristic: Some(DrcCharacteristic::Cicp(1)),
                        border: None,
                    }],
                }],
                custom_characteristics_left: Vec::new(),
                custom_characteristics_right: Vec::new(),
                shape_filters: Vec::new(),
            }],
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
                gain_modifications: vec![GainModification {
                    target_characteristic_left: None,
                    target_characteristic_right: None,
                    attenuation_scaling: 1.0,
                    amplification_scaling: 1.0,
                    gain_offset_db: 0.0,
                    shape_filter_index: None,
                }],
                gain_modifications_per_band: Vec::new(),
                ducking_modifications: Vec::new(),
            }],
            extension_present: false,
            extensions: Vec::new(),
            bits_read: 0,
        };
        let gain = UniDrcGain {
            sequences: vec![GainSequence {
                interpolation_type: GainInterpolationType::Linear,
                nodes: vec![GainNode {
                    time: 0,
                    gain_db: -6.0,
                    slope: 0.0,
                }],
            }],
            extension_present: false,
            extensions: Vec::new(),
            bits_read: 0,
        };
        let payload = nonzero_spectral_sce_payload();

        let mut baseline = AacLcDecoder::new(4, 1).unwrap();
        let baseline_f32 = baseline
            .decode_raw_data_block_f32(&payload)
            .unwrap()
            .interleaved_f32();
        let mut processed = AacLcDecoder::new(4, 1).unwrap();
        processed.configure_drc(config.clone(), DrcSelectionRequest::default());
        processed.update_drc_gain(gain.clone());
        processed.apply_configured_drc_f32(&mut []).unwrap();
        let processed_f32 = processed
            .decode_raw_data_block_f32(&payload)
            .unwrap()
            .interleaved_f32();
        let baseline_energy = baseline_f32
            .iter()
            .map(|sample| sample * sample)
            .sum::<f32>();
        let processed_energy = processed_f32
            .iter()
            .map(|sample| sample * sample)
            .sum::<f32>();
        let rms_ratio = (processed_energy / baseline_energy).sqrt();
        assert!((rms_ratio - 10.0f32.powf(-6.0 / 20.0)).abs() < 1.0e-5);
        let baseline_concealed = baseline.conceal_f32_interleaved().unwrap();
        let processed_concealed = processed.conceal_f32_interleaved().unwrap();
        let baseline_energy = baseline_concealed
            .iter()
            .map(|sample| sample * sample)
            .sum::<f32>();
        let processed_energy = processed_concealed
            .iter()
            .map(|sample| sample * sample)
            .sum::<f32>();
        let rms_ratio = (processed_energy / baseline_energy).sqrt();
        assert!((rms_ratio - 10.0f32.powf(-6.0 / 20.0)).abs() < 1.0e-5);

        let mut baseline = AacLcDecoder::new(4, 1).unwrap();
        let baseline_i16 = baseline
            .decode_raw_data_block_fixed_interleaved_i16(&payload)
            .unwrap();
        let mut processed = AacLcDecoder::new(4, 1).unwrap();
        processed.configure_drc(config, DrcSelectionRequest::default());
        processed.update_drc_gain(gain);
        processed.apply_configured_drc_i16(&mut []).unwrap();
        let processed_i16 = processed
            .decode_raw_data_block_fixed_interleaved_i16(&payload)
            .unwrap();
        assert!(baseline_i16.iter().any(|sample| *sample != 0));
        let baseline_energy = baseline_i16
            .iter()
            .map(|&sample| (sample as f64).powi(2))
            .sum::<f64>();
        let processed_energy = processed_i16
            .iter()
            .map(|&sample| (sample as f64).powi(2))
            .sum::<f64>();
        let rms_ratio = (processed_energy / baseline_energy).sqrt();
        assert!((rms_ratio - 10.0f64.powf(-6.0 / 20.0)).abs() < 0.01);
        let baseline_concealed = baseline.conceal_fixed_interleaved_i16().unwrap();
        let processed_concealed = processed.conceal_fixed_interleaved_i16().unwrap();
        let baseline_energy = baseline_concealed
            .iter()
            .map(|&sample| (sample as f64).powi(2))
            .sum::<f64>();
        let processed_energy = processed_concealed
            .iter()
            .map(|&sample| (sample as f64).powi(2))
            .sum::<f64>();
        let rms_ratio = (processed_energy / baseline_energy).sqrt();
        assert!((rms_ratio - 10.0f64.powf(-6.0 / 20.0)).abs() < 0.02);
    }

    #[test]
    fn terminated_raw_block_reader_consumes_end_before_next_block() {
        let mut writer = BitWriter::new();
        write_zero_sce_payload_bits(&mut writer, false);
        writer.write(ElementId::End.bits() as u32, 3);
        write_zero_sce_payload_bits(&mut writer, false);
        writer.write(ElementId::End.bits() as u32, 3);
        let input = writer.finish();

        let mut reader = BitReader::new(&input);
        let mut decoder = AacLcDecoder::new(4, 1).unwrap();
        for _ in 0..2 {
            let decoded = decoder
                .decode_raw_data_block_f32_terminated_from_reader(&mut reader)
                .unwrap();
            assert!(matches!(decoded, DecodedAacLcFrame::Mono(_)));
        }
        assert!(reader.remaining_bits_are_zero());

        let unterminated = zero_sce_payload(false);
        let mut reader = BitReader::new(&unterminated);
        assert_eq!(
            decoder
                .decode_raw_data_block_f32_terminated_from_reader(&mut reader)
                .unwrap_err(),
            DecodeError::RawDataBlockTerminatorMissing
        );

        assert_eq!(
            consume_raw_data_block_terminator(&mut BitReader::with_bit_len(&[0], 3).unwrap(),),
            Err(DecodeError::RawDataBlockTerminatorMissing)
        );

        let mut header = AdtsHeader::aac_lc(44_100, 1, 0).unwrap();
        header.profile = 0;
        assert_eq!(
            validate_adts_aac_lc_configuration(&decoder, header),
            Err(DecodeError::UnsupportedAudioObjectType(1))
        );
        header.profile = 1;
        header.sampling_frequency_index = 3;
        assert_eq!(
            validate_adts_aac_lc_configuration(&decoder, header),
            Err(DecodeError::AdtsConfigChanged)
        );
    }

    #[test]
    fn decodes_zero_single_channel_frame_to_fixed_i16_silence() {
        let payload = zero_sce_payload(false);
        let mut filterbank = FixedLongBlockFilterbank::new(1024).unwrap();
        let mut pns_random = PnsRandomState::new(1);
        let pcm =
            decode_aac_lc_single_channel_fixed_i16(&payload, 4, &mut filterbank, &mut pns_random)
                .unwrap();

        assert_eq!(pcm.len(), 1024);
        assert!(pcm.iter().all(|sample| *sample == 0));
    }

    #[test]
    fn decodes_zero_single_channel_spectra_to_fixed_spectrum_bridge() {
        let payload = zero_sce_payload(false);
        let mut pns_random = PnsRandomState::new(1);
        let decoded =
            decode_aac_lc_single_channel_spectra_fixed_bridge(&payload, 4, &mut pns_random)
                .unwrap();

        assert_eq!(decoded.side_info.global_gain, 100);
        assert_eq!(decoded.stream.section_data.codebooks, vec![vec![ZERO_HCB]]);
        assert_eq!(decoded.stream.spectrum.windows.len(), 1);
        assert_eq!(decoded.stream.spectrum.windows[0].len(), 1024);
        assert!(decoded.stream.spectrum.windows[0]
            .iter()
            .all(|sample| *sample == 0));
    }

    #[test]
    fn rejects_gain_control_data() {
        let payload = zero_sce_payload(true);
        let mut filterbank = LongBlockFilterbank::new(1024).unwrap();
        let mut pns_random = PnsRandomState::new(1);
        assert_eq!(
            decode_aac_lc_single_channel_f32(&payload, 4, &mut filterbank, &mut pns_random)
                .unwrap_err()
                .to_string(),
            "AAC gain_control_data is unsupported"
        );
    }

    #[test]
    fn decodes_cpe_common_window_zero_channel_streams_to_spectra() {
        let payload = zero_cpe_payload(0);
        let mut pns_random = PnsRandomState::new(1);
        let decoded = decode_aac_lc_channel_pair_spectra(&payload, 4, &mut pns_random).unwrap();

        assert!(decoded.prefix.common_window);
        assert_eq!(decoded.ms_stereo.as_ref().unwrap().used, vec![vec![false]]);
        assert_eq!(decoded.left.section_data.codebooks, vec![vec![ZERO_HCB]]);
        assert_eq!(decoded.right.section_data.codebooks, vec![vec![ZERO_HCB]]);
        assert!(decoded.left.spectrum.windows[0]
            .iter()
            .all(|value| *value == 0.0));
        assert!(decoded.right.spectrum.windows[0]
            .iter()
            .all(|value| *value == 0.0));
    }

    #[test]
    fn decodes_cpe_common_window_zero_channel_streams_to_fixed_spectra_bridge() {
        let payload = zero_cpe_payload(0);
        let mut pns_random = PnsRandomState::new(1);
        let mut decoded =
            decode_aac_lc_channel_pair_spectra_fixed_bridge(&payload, 4, &mut pns_random).unwrap();
        apply_aac_lc_channel_pair_fixed_spectrum_stereo_tools_bridge(&mut decoded, 4).unwrap();

        assert!(decoded.prefix.common_window);
        assert_eq!(decoded.left.section_data.codebooks, vec![vec![ZERO_HCB]]);
        assert_eq!(decoded.right.section_data.codebooks, vec![vec![ZERO_HCB]]);
        assert!(decoded.left.spectrum.windows[0]
            .iter()
            .all(|sample| *sample == 0));
        assert!(decoded.right.spectrum.windows[0]
            .iter()
            .all(|sample| *sample == 0));
    }

    #[test]
    fn decodes_cpe_independent_windows_without_ms_stereo_in_both_formats() {
        let mut writer = BitWriter::new();
        writer.write(ElementId::ChannelPair.bits() as u32, 3);
        writer.write(0, 4); // element_instance_tag
        writer.write_bool(false); // no common window
        write_zero_independent_channel_stream(&mut writer, 1);
        write_zero_independent_channel_stream(&mut writer, 1);
        let payload = writer.finish();

        let mut pns_random = PnsRandomState::new(1);
        let decoded = decode_aac_lc_channel_pair_spectra(&payload, 4, &mut pns_random).unwrap();
        assert!(!decoded.prefix.common_window);
        assert!(decoded.ms_stereo.is_none());
        assert!(decoded
            .left
            .spectrum
            .windows
            .iter()
            .chain(&decoded.right.spectrum.windows)
            .flatten()
            .all(|sample| *sample == 0.0));

        let decoded =
            decode_aac_lc_channel_pair_spectra_fixed_bridge(&payload, 4, &mut pns_random).unwrap();
        assert!(!decoded.prefix.common_window);
        assert!(decoded.ms_stereo.is_none());
        assert!(decoded
            .left
            .spectrum
            .windows
            .iter()
            .chain(&decoded.right.spectrum.windows)
            .flatten()
            .all(|sample| *sample == 0));
    }

    #[test]
    fn decodes_cpe_pns_with_ms_correlation_before_stereo_tools() {
        let payload = correlated_pns_cpe_payload();
        let mut pns_random = PnsRandomState::new(7);
        let decoded = decode_aac_lc_channel_pair_spectra(&payload, 4, &mut pns_random).unwrap();

        assert!(decoded.ms_stereo.as_ref().unwrap().is_used(0, 0));
        assert_eq!(decoded.left.section_data.codebooks, vec![vec![NOISE_HCB]]);
        assert_eq!(decoded.right.section_data.codebooks, vec![vec![NOISE_HCB]]);
        assert_eq!(
            decoded.left.spectrum.windows[0],
            decoded.right.spectrum.windows[0]
        );
        assert!(decoded.left.spectrum.windows[0]
            .iter()
            .any(|value| *value != 0.0));
    }

    #[test]
    fn applies_ms_and_intensity_tools_to_decoded_cpe_spectra() {
        let ics = test_ics(2);
        let prefix = ChannelPairElementSideInfoPrefix {
            element_instance_tag: 0,
            common_window: true,
            shared_ics: Some(ics.clone()),
            bits_read: 0,
        };
        let ms = MsStereoData {
            mask_present: MsMaskPresent::Some,
            used: vec![vec![true, false]],
        };
        let left = test_stream(
            &ics,
            test_sections(vec![1, 1]),
            vec![0, 0],
            vec![3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0],
        );
        let right = test_stream(
            &ics,
            test_sections(vec![1, INTENSITY_HCB]),
            vec![0, -100],
            vec![1.0, 2.0, 4.0, 8.0, 0.0, 0.0, 0.0, 0.0],
        );
        let mut decoded = DecodedChannelPairSpectra {
            prefix,
            ms_stereo: Some(ms),
            left,
            right,
            right_channel_start_bit: 0,
            bits_read: 0,
        };

        apply_aac_lc_channel_pair_stereo_tools_f32(&mut decoded, 4).unwrap();

        assert!(
            (decoded.left.spectrum.windows[0][0] - 4.0 * std::f32::consts::FRAC_1_SQRT_2).abs()
                < 1.0e-6
        );
        assert!(
            (decoded.right.spectrum.windows[0][0] - 2.0 * std::f32::consts::FRAC_1_SQRT_2).abs()
                < 1.0e-6
        );
        assert_eq!(decoded.left.spectrum.windows[0][4], 13.0);
        assert_eq!(decoded.right.spectrum.windows[0][4], 13.0);
        assert_eq!(decoded.right.spectrum.windows[0][7], 23.0);
    }

    #[test]
    fn fixed_bridge_ms_stereo_matches_f32_reference_with_quantized_tolerance() {
        let ics = test_ics(1);
        let ms = MsStereoData {
            mask_present: MsMaskPresent::All,
            used: vec![vec![true]],
        };
        let left = test_stream(
            &ics,
            test_sections(vec![1]),
            vec![0],
            vec![3.0, 5.0, 7.0, 11.0],
        );
        let right = test_stream(
            &ics,
            test_sections(vec![1]),
            vec![0],
            vec![1.0, 2.0, 4.0, 8.0],
        );
        let prefix = ChannelPairElementSideInfoPrefix {
            element_instance_tag: 0,
            common_window: true,
            shared_ics: Some(ics.clone()),
            bits_read: 0,
        };
        let mut reference = DecodedChannelPairSpectra {
            prefix: prefix.clone(),
            ms_stereo: Some(ms.clone()),
            left: left.clone(),
            right: right.clone(),
            right_channel_start_bit: 0,
            bits_read: 0,
        };
        let mut bridged = DecodedChannelPairSpectra {
            prefix,
            ms_stereo: Some(ms),
            left,
            right,
            right_channel_start_bit: 0,
            bits_read: 0,
        };

        apply_aac_lc_channel_pair_stereo_tools_f32(&mut reference, 4).unwrap();
        apply_aac_lc_channel_pair_stereo_tools_fixed_bridge(&mut bridged, 4).unwrap();

        for (reference, bridged) in reference.left.spectrum.windows[0]
            .iter()
            .zip(&bridged.left.spectrum.windows[0])
        {
            assert!((reference - bridged).abs() <= 1.0e-3);
        }
        for (reference, bridged) in reference.right.spectrum.windows[0]
            .iter()
            .zip(&bridged.right.spectrum.windows[0])
        {
            assert!((reference - bridged).abs() <= 1.0e-3);
        }
    }

    #[test]
    fn stereo_tool_wrappers_propagate_ms_and_intensity_layout_errors() {
        let ics = test_ics(1);
        let prefix = ChannelPairElementSideInfoPrefix {
            element_instance_tag: 0,
            common_window: true,
            shared_ics: Some(ics.clone()),
            bits_read: 0,
        };
        let invalid_ms = MsStereoData {
            mask_present: MsMaskPresent::Some,
            used: Vec::new(),
        };
        let left = test_stream(
            &ics,
            test_sections(vec![1]),
            vec![0],
            vec![1.0, 2.0, 3.0, 4.0],
        );
        let right = test_stream(
            &ics,
            test_sections(vec![1]),
            vec![0],
            vec![4.0, 3.0, 2.0, 1.0],
        );
        let invalid_ms_pair = DecodedChannelPairSpectra {
            prefix: prefix.clone(),
            ms_stereo: Some(invalid_ms.clone()),
            left: left.clone(),
            right: right.clone(),
            right_channel_start_bit: 0,
            bits_read: 0,
        };
        let mut decoded = invalid_ms_pair.clone();
        assert!(apply_aac_lc_channel_pair_stereo_tools_f32(&mut decoded, 4).is_err());
        let mut decoded = invalid_ms_pair;
        assert!(apply_aac_lc_channel_pair_stereo_tools_fixed_bridge(&mut decoded, 4).is_err());

        let mut intensity_right = test_stream(
            &ics,
            test_sections(vec![INTENSITY_HCB]),
            vec![0],
            vec![0.0; 4],
        );
        intensity_right.scalefactors.values[0].clear();
        let invalid_intensity_pair = DecodedChannelPairSpectra {
            prefix: prefix.clone(),
            ms_stereo: None,
            left,
            right: intensity_right,
            right_channel_start_bit: 0,
            bits_read: 0,
        };
        let mut decoded = invalid_intensity_pair.clone();
        assert!(apply_aac_lc_channel_pair_stereo_tools_f32(&mut decoded, 4).is_err());
        let mut decoded = invalid_intensity_pair;
        assert!(apply_aac_lc_channel_pair_stereo_tools_fixed_bridge(&mut decoded, 4).is_err());

        let fixed_stream = |section, scalefactors: Vec<i16>| DecodedChannelStreamFixed {
            global_gain: 100,
            ics: ics.clone(),
            section_data: test_sections(vec![section]),
            scalefactors: ScalefactorData {
                values: vec![scalefactors],
            },
            pulse_data: PulseData::absent(),
            tns_data: TnsData::absent(1),
            spectral: SpectralData {
                windows: vec![vec![0; 4]],
            },
            spectrum: FixedInverseQuantizedSpectrum {
                windows: vec![vec![0; 4]],
                window_exponents: vec![0],
            },
        };
        let mut fixed = DecodedChannelPairSpectraFixed {
            prefix: prefix.clone(),
            ms_stereo: Some(invalid_ms),
            left: fixed_stream(1, vec![0]),
            right: fixed_stream(1, vec![0]),
            right_channel_start_bit: 0,
            bits_read: 0,
        };
        assert!(
            apply_aac_lc_channel_pair_fixed_spectrum_stereo_tools_bridge(&mut fixed, 4).is_err()
        );
        let mut fixed = DecodedChannelPairSpectraFixed {
            prefix,
            ms_stereo: None,
            left: fixed_stream(1, vec![0]),
            right: fixed_stream(INTENSITY_HCB, Vec::new()),
            right_channel_start_bit: 0,
            bits_read: 0,
        };
        assert!(
            apply_aac_lc_channel_pair_fixed_spectrum_stereo_tools_bridge(&mut fixed, 4).is_err()
        );
    }

    #[test]
    fn staged_and_pair_pns_tns_wrappers_propagate_layout_errors() {
        let ics = test_ics(1);
        let sections = test_sections(vec![ZERO_HCB]);
        let present_tns = TnsData {
            present: true,
            filters: vec![vec![TnsFilter {
                start_band: 0,
                stop_band: 1,
                direction: TnsDirection::Forward,
                resolution: 3,
                coefficients: vec![1],
            }]],
        };
        let mut stream = test_stream(&ics, sections.clone(), vec![0], Vec::new());
        stream.tns_data = present_tns.clone();
        stream.spectrum.windows.clear();
        let mut staged = [StagedAacLcElement::Single {
            element_id: ElementId::SingleChannel,
            element_instance_tag: 0,
            spectra: DecodedSingleChannelSpectra {
                side_info: SingleChannelElementSideInfo {
                    id: ElementId::SingleChannel,
                    element_instance_tag: 0,
                    global_gain: 100,
                    ics: ics.clone(),
                    bits_read: 0,
                },
                stream,
                bits_read: 0,
            },
            labels: Vec::new(),
        }];
        assert!(apply_tns_to_staged_spectra(&mut staged, 4).is_err());

        let fixed_stream = |values: Vec<i32>, tns_data: TnsData| DecodedChannelStreamFixed {
            global_gain: 100,
            ics: ics.clone(),
            section_data: sections.clone(),
            scalefactors: ScalefactorData {
                values: vec![vec![0]],
            },
            pulse_data: PulseData::absent(),
            tns_data,
            spectral: SpectralData {
                windows: vec![vec![0; 4]],
            },
            spectrum: FixedInverseQuantizedSpectrum {
                windows: vec![values],
                window_exponents: vec![0],
            },
        };
        let mut invalid_fixed_stream = fixed_stream(Vec::new(), present_tns.clone());
        invalid_fixed_stream.spectrum.windows.clear();
        let mut staged = [StagedAacLcElementFixed::Single {
            element_id: ElementId::SingleChannel,
            element_instance_tag: 0,
            spectra: DecodedSingleChannelSpectraFixed {
                side_info: SingleChannelElementSideInfo {
                    id: ElementId::SingleChannel,
                    element_instance_tag: 0,
                    global_gain: 100,
                    ics: ics.clone(),
                    bits_read: 0,
                },
                stream: invalid_fixed_stream,
                bits_read: 0,
            },
            labels: Vec::new(),
        }];
        assert!(apply_tns_to_staged_fixed_spectra(&mut staged, 4).is_err());

        let prefix = ChannelPairElementSideInfoPrefix {
            element_instance_tag: 0,
            common_window: true,
            shared_ics: Some(ics.clone()),
            bits_read: 0,
        };
        let valid = test_stream(&ics, sections.clone(), vec![0], vec![0.0; 4]);
        let mut invalid_pns = DecodedChannelPairSpectra {
            prefix: prefix.clone(),
            ms_stereo: None,
            left: valid.clone(),
            right: valid.clone(),
            right_channel_start_bit: 0,
            bits_read: 0,
        };
        invalid_pns.left.spectrum.windows.clear();
        assert!(
            apply_channel_pair_pns(&mut invalid_pns, 4, 1024, &mut PnsRandomState::new(1)).is_err()
        );

        let mut invalid_left = DecodedChannelPairSpectra {
            prefix: prefix.clone(),
            ms_stereo: None,
            left: valid.clone(),
            right: valid.clone(),
            right_channel_start_bit: 0,
            bits_read: 0,
        };
        invalid_left.left.tns_data = present_tns.clone();
        invalid_left.left.spectrum.windows.clear();
        assert!(apply_channel_pair_tns(&mut invalid_left, 4, 1024).is_err());
        let mut invalid_right = DecodedChannelPairSpectra {
            prefix,
            ms_stereo: None,
            left: valid.clone(),
            right: valid,
            right_channel_start_bit: 0,
            bits_read: 0,
        };
        invalid_right.right.tns_data = present_tns.clone();
        invalid_right.right.spectrum.windows.clear();
        assert!(apply_channel_pair_tns(&mut invalid_right, 4, 1024).is_err());

        let fixed_prefix = ChannelPairElementSideInfoPrefix {
            element_instance_tag: 0,
            common_window: true,
            shared_ics: Some(ics.clone()),
            bits_read: 0,
        };
        let valid_fixed = fixed_stream(vec![0; 4], TnsData::absent(1));
        let mut invalid_pns = DecodedChannelPairSpectraFixed {
            prefix: fixed_prefix.clone(),
            ms_stereo: None,
            left: valid_fixed.clone(),
            right: valid_fixed.clone(),
            right_channel_start_bit: 0,
            bits_read: 0,
        };
        invalid_pns.left.spectrum.windows.clear();
        assert!(apply_channel_pair_pns_and_tns_fixed_bridge(
            &mut invalid_pns.clone(),
            4,
            1024,
            &mut PnsRandomState::new(1),
        )
        .is_err());
        assert!(apply_channel_pair_pns_fixed_bridge(
            &mut invalid_pns,
            4,
            1024,
            &mut PnsRandomState::new(1),
        )
        .is_err());
        let mut invalid_left_stream = fixed_stream(Vec::new(), present_tns.clone());
        invalid_left_stream.spectrum.windows.clear();
        let mut invalid_left = DecodedChannelPairSpectraFixed {
            prefix: fixed_prefix.clone(),
            ms_stereo: None,
            left: invalid_left_stream,
            right: valid_fixed.clone(),
            right_channel_start_bit: 0,
            bits_read: 0,
        };
        assert!(apply_channel_pair_tns_fixed_bridge(&mut invalid_left, 4, 1024).is_err());
        let mut invalid_right_stream = fixed_stream(Vec::new(), present_tns);
        invalid_right_stream.spectrum.windows.clear();
        let mut invalid_right = DecodedChannelPairSpectraFixed {
            prefix: fixed_prefix,
            ms_stereo: None,
            left: valid_fixed,
            right: invalid_right_stream,
            right_channel_start_bit: 0,
            bits_read: 0,
        };
        assert!(apply_channel_pair_tns_fixed_bridge(&mut invalid_right, 4, 1024).is_err());
    }

    #[test]
    fn fixed_spectrum_ms_stereo_transforms_samples_and_validates_window_size() {
        let ics = test_ics(1);
        let ms = MsStereoData {
            mask_present: MsMaskPresent::All,
            used: vec![vec![true]],
        };
        let sections = test_sections(vec![1]);
        let mut left = FixedInverseQuantizedSpectrum {
            windows: vec![vec![32_768, 16_384, -32_768, -16_384]],
            window_exponents: vec![0],
        };
        let mut right = FixedInverseQuantizedSpectrum {
            windows: vec![vec![16_384, -16_384, 8_192, -8_192]],
            window_exponents: vec![0],
        };
        apply_ms_stereo_fixed_spectrum_bridge(
            &ms,
            &mut left,
            &mut right,
            &ics,
            &[0, 4],
            &sections,
            &sections,
        )
        .unwrap();
        assert_eq!(left.windows[0], [34_755, 0, -17_377, -17_377]);
        assert_eq!(right.windows[0], [11_585, 23_170, -28_962, -5_792]);

        let invalid_ms = MsStereoData {
            mask_present: MsMaskPresent::Some,
            used: Vec::new(),
        };
        assert_eq!(
            apply_ms_stereo_fixed_spectrum_bridge(
                &invalid_ms,
                &mut left.clone(),
                &mut right.clone(),
                &ics,
                &[0, 4],
                &sections,
                &sections,
            ),
            Err(DecodeError::Stereo(StereoError::LayoutMismatch))
        );

        let mut short_left = FixedInverseQuantizedSpectrum {
            windows: vec![Vec::new()],
            window_exponents: vec![0],
        };
        let mut short_right = short_left.clone();
        assert_eq!(
            apply_ms_stereo_fixed_spectrum_bridge(
                &ms,
                &mut short_left,
                &mut short_right,
                &ics,
                &[0, 4],
                &sections,
                &sections,
            ),
            Err(DecodeError::Stereo(StereoError::LayoutMismatch))
        );

        let mut floating_left = InverseQuantizedSpectrum {
            windows: vec![vec![1.0; 4]],
        };
        let mut floating_right = InverseQuantizedSpectrum {
            windows: vec![vec![0.5; 4]],
        };
        assert_eq!(
            apply_ms_stereo_fixed_bridge(
                &invalid_ms,
                &mut floating_left,
                &mut floating_right,
                &ics,
                &[0, 4],
                &sections,
                &sections,
            ),
            Err(DecodeError::Stereo(StereoError::LayoutMismatch))
        );
        floating_left.windows[0].clear();
        floating_right.windows[0].clear();
        assert_eq!(
            apply_ms_stereo_fixed_bridge(
                &ms,
                &mut floating_left,
                &mut floating_right,
                &ics,
                &[0, 4],
                &sections,
                &sections,
            ),
            Err(DecodeError::Stereo(StereoError::LayoutMismatch))
        );
    }

    #[test]
    fn fixed_spectrum_intensity_stereo_transforms_samples_and_validates_window_size() {
        let ics = test_ics(1);
        let sections = test_sections(vec![INTENSITY_HCB]);
        let scalefactors = ScalefactorData {
            values: vec![vec![-100]],
        };
        let left = FixedInverseQuantizedSpectrum {
            windows: vec![vec![32_768, 16_384, -32_768, -16_384]],
            window_exponents: vec![0],
        };
        let mut right = FixedInverseQuantizedSpectrum {
            windows: vec![vec![0; 4]],
            window_exponents: vec![0],
        };
        apply_intensity_stereo_fixed_spectrum_bridge(
            None,
            &left,
            &mut right,
            &ics,
            &[0, 4],
            &sections,
            &scalefactors,
        )
        .unwrap();
        assert!(right.windows[0].iter().any(|&sample| sample != 0));

        let ms = MsStereoData {
            mask_present: MsMaskPresent::All,
            used: vec![vec![true]],
        };
        apply_intensity_stereo_fixed_spectrum_bridge(
            Some(&ms),
            &left,
            &mut right,
            &ics,
            &[0, 4],
            &sections,
            &scalefactors,
        )
        .unwrap();

        let missing_scalefactor = ScalefactorData {
            values: vec![Vec::new()],
        };
        assert_eq!(
            apply_intensity_stereo_fixed_spectrum_bridge(
                None,
                &left,
                &mut right.clone(),
                &ics,
                &[0, 4],
                &sections,
                &missing_scalefactor,
            ),
            Err(DecodeError::Stereo(StereoError::LayoutMismatch))
        );

        let floating_left = InverseQuantizedSpectrum {
            windows: vec![vec![1.0; 4]],
        };
        let mut floating_right = InverseQuantizedSpectrum {
            windows: vec![vec![0.0; 4]],
        };
        apply_intensity_stereo_fixed_bridge(
            Some(&ms),
            &floating_left,
            &mut floating_right,
            &ics,
            &[0, 4],
            &sections,
            &scalefactors,
        )
        .unwrap();
        floating_right.windows[0].clear();
        assert_eq!(
            apply_intensity_stereo_fixed_bridge(
                Some(&ms),
                &floating_left,
                &mut floating_right,
                &ics,
                &[0, 4],
                &sections,
                &scalefactors,
            ),
            Err(DecodeError::Stereo(StereoError::LayoutMismatch))
        );
        assert_eq!(
            apply_intensity_stereo_fixed_bridge(
                None,
                &floating_left,
                &mut floating_right,
                &ics,
                &[0, 4],
                &sections,
                &missing_scalefactor,
            ),
            Err(DecodeError::Stereo(StereoError::LayoutMismatch))
        );

        let mut short_right = FixedInverseQuantizedSpectrum {
            windows: vec![Vec::new()],
            window_exponents: vec![0],
        };
        assert_eq!(
            apply_intensity_stereo_fixed_spectrum_bridge(
                None,
                &left,
                &mut short_right,
                &ics,
                &[0, 4],
                &sections,
                &scalefactors,
            ),
            Err(DecodeError::Stereo(StereoError::LayoutMismatch))
        );
    }

    #[test]
    fn fixed_bridge_intensity_stereo_matches_f32_reference_with_quantized_tolerance() {
        let ics = test_ics(2);
        let left = test_stream(
            &ics,
            test_sections(vec![1, 1]),
            vec![0, 0],
            vec![3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0],
        );
        let right = test_stream(
            &ics,
            test_sections(vec![1, INTENSITY_HCB]),
            vec![0, -100],
            vec![1.0, 2.0, 4.0, 8.0, 0.0, 0.0, 0.0, 0.0],
        );
        let prefix = ChannelPairElementSideInfoPrefix {
            element_instance_tag: 0,
            common_window: true,
            shared_ics: Some(ics),
            bits_read: 0,
        };
        let mut reference = DecodedChannelPairSpectra {
            prefix: prefix.clone(),
            ms_stereo: None,
            left: left.clone(),
            right: right.clone(),
            right_channel_start_bit: 0,
            bits_read: 0,
        };
        let mut bridged = DecodedChannelPairSpectra {
            prefix,
            ms_stereo: None,
            left,
            right,
            right_channel_start_bit: 0,
            bits_read: 0,
        };

        apply_aac_lc_channel_pair_stereo_tools_f32(&mut reference, 4).unwrap();
        apply_aac_lc_channel_pair_stereo_tools_fixed_bridge(&mut bridged, 4).unwrap();

        for (reference, bridged) in reference.right.spectrum.windows[0]
            .iter()
            .zip(&bridged.right.spectrum.windows[0])
        {
            assert!((reference - bridged).abs() <= 1.0e-3);
        }
    }

    #[test]
    fn decodes_cpe_zero_frame_through_stereo_filterbanks() {
        let payload = zero_cpe_payload(0);
        let mut left_filterbank = LongBlockFilterbank::new(1024).unwrap();
        let mut right_filterbank = LongBlockFilterbank::new(1024).unwrap();
        let mut pns_random = PnsRandomState::new(1);
        let decoded = decode_aac_lc_channel_pair_f32(
            &payload,
            4,
            &mut left_filterbank,
            &mut right_filterbank,
            &mut pns_random,
        )
        .unwrap();

        assert_eq!(decoded.left_samples.len(), 1024);
        assert_eq!(decoded.right_samples.len(), 1024);
        assert!(decoded.left_samples.iter().all(|sample| *sample == 0.0));
        assert!(decoded.right_samples.iter().all(|sample| *sample == 0.0));
    }

    #[test]
    fn decodes_cpe_zero_frame_through_fixed_stereo_filterbanks() {
        let payload = zero_cpe_payload(0);
        let mut left_filterbank = FixedLongBlockFilterbank::new(1024).unwrap();
        let mut right_filterbank = FixedLongBlockFilterbank::new(1024).unwrap();
        let mut pns_random = PnsRandomState::new(1);
        let pcm = decode_aac_lc_channel_pair_fixed_interleaved_i16(
            &payload,
            4,
            &mut left_filterbank,
            &mut right_filterbank,
            &mut pns_random,
        )
        .unwrap();

        assert_eq!(pcm.len(), 2048);
        assert!(pcm.iter().all(|sample| *sample == 0));
    }

    #[test]
    fn decodes_cpe_zero_frame_through_fixed_spectrum_bridge_filterbanks() {
        let payload = zero_cpe_payload(0);
        let mut left_filterbank = FixedLongBlockFilterbank::new(1024).unwrap();
        let mut right_filterbank = FixedLongBlockFilterbank::new(1024).unwrap();
        let mut pns_random = PnsRandomState::new(1);
        let pcm = decode_aac_lc_channel_pair_fixed_spectrum_interleaved_i16_bridge(
            &payload,
            4,
            &mut left_filterbank,
            &mut right_filterbank,
            &mut pns_random,
        )
        .unwrap();

        assert_eq!(pcm.len(), 2048);
        assert!(pcm.iter().all(|sample| *sample == 0));
    }

    #[test]
    fn cpe_decode_facades_propagate_every_byte_prefix_truncation() {
        let mono = zero_sce_payload(false);
        let payload = zero_cpe_payload(0);
        let mut left_f32 = LongBlockFilterbank::new(1024).unwrap();
        let mut right_f32 = LongBlockFilterbank::new(1024).unwrap();
        let mut left_fixed = FixedLongBlockFilterbank::new(1024).unwrap();
        let mut right_fixed = FixedLongBlockFilterbank::new(1024).unwrap();
        let mut left_bridge = FixedLongBlockFilterbank::new(1024).unwrap();
        let mut right_bridge = FixedLongBlockFilterbank::new(1024).unwrap();
        let mut mono_fixed = FixedLongBlockFilterbank::new(1024).unwrap();
        let mut random = PnsRandomState::new(1);

        for end in 0..mono.len() {
            assert!(decode_aac_lc_single_channel_fixed_i16_from_reader(
                &mut BitReader::new(&mono[..end]),
                4,
                &mut mono_fixed,
                &mut random,
            )
            .is_err());
        }

        for end in 0..payload.len() {
            let truncated = &payload[..end];
            assert!(decode_aac_lc_channel_pair_spectra_from_reader(
                &mut BitReader::new(truncated),
                4,
                &mut random,
            )
            .is_err());
            assert!(decode_aac_lc_channel_pair_spectra_fixed_bridge_from_reader(
                &mut BitReader::new(truncated),
                4,
                &mut random,
            )
            .is_err());
            assert!(decode_aac_lc_channel_pair_f32_from_reader(
                &mut BitReader::new(truncated),
                4,
                &mut left_f32,
                &mut right_f32,
                &mut random,
            )
            .is_err());
            assert!(
                decode_aac_lc_channel_pair_fixed_interleaved_i16_from_reader(
                    &mut BitReader::new(truncated),
                    4,
                    &mut left_fixed,
                    &mut right_fixed,
                    &mut random,
                )
                .is_err()
            );
            assert!(
                decode_aac_lc_channel_pair_fixed_spectrum_interleaved_i16_bridge_from_reader(
                    &mut BitReader::new(truncated),
                    4,
                    &mut left_bridge,
                    &mut right_bridge,
                    &mut random,
                )
                .is_err()
            );
        }

        let mono_bits = decode_aac_lc_single_channel_spectra_from_reader(
            &mut BitReader::new(&mono),
            4,
            &mut PnsRandomState::new(1),
        )
        .unwrap()
        .bits_read;
        for bit_len in 0..mono_bits {
            assert!(decode_aac_lc_single_channel_spectra_from_reader(
                &mut BitReader::with_bit_len(&mono, bit_len).unwrap(),
                4,
                &mut random,
            )
            .is_err());
            assert!(
                decode_aac_lc_single_channel_spectra_fixed_bridge_from_reader(
                    &mut BitReader::with_bit_len(&mono, bit_len).unwrap(),
                    4,
                    &mut random,
                )
                .is_err()
            );
            assert!(decode_aac_lc_single_channel_f32_from_reader(
                &mut BitReader::with_bit_len(&mono, bit_len).unwrap(),
                4,
                &mut left_f32,
                &mut random,
            )
            .is_err());
            assert!(decode_aac_lc_single_channel_fixed_i16_from_reader(
                &mut BitReader::with_bit_len(&mono, bit_len).unwrap(),
                4,
                &mut mono_fixed,
                &mut random,
            )
            .is_err());
        }

        let payload_bits = decode_aac_lc_channel_pair_spectra_from_reader(
            &mut BitReader::new(&payload),
            4,
            &mut PnsRandomState::new(1),
        )
        .unwrap()
        .bits_read;
        for bit_len in 0..payload_bits {
            assert!(decode_aac_lc_channel_pair_spectra_from_reader(
                &mut BitReader::with_bit_len(&payload, bit_len).unwrap(),
                4,
                &mut random,
            )
            .is_err());
            assert!(decode_aac_lc_channel_pair_spectra_fixed_bridge_from_reader(
                &mut BitReader::with_bit_len(&payload, bit_len).unwrap(),
                4,
                &mut random,
            )
            .is_err());
            assert!(decode_aac_lc_channel_pair_f32_from_reader(
                &mut BitReader::with_bit_len(&payload, bit_len).unwrap(),
                4,
                &mut left_f32,
                &mut right_f32,
                &mut random,
            )
            .is_err());
            assert!(
                decode_aac_lc_channel_pair_fixed_interleaved_i16_from_reader(
                    &mut BitReader::with_bit_len(&payload, bit_len).unwrap(),
                    4,
                    &mut left_fixed,
                    &mut right_fixed,
                    &mut random,
                )
                .is_err()
            );
            assert!(
                decode_aac_lc_channel_pair_fixed_spectrum_interleaved_i16_bridge_from_reader(
                    &mut BitReader::with_bit_len(&payload, bit_len).unwrap(),
                    4,
                    &mut left_bridge,
                    &mut right_bridge,
                    &mut random,
                )
                .is_err()
            );
        }
    }

    #[test]
    fn arbitrary_raw_aac_payloads_never_panic() {
        let mut state = 0x6d2b_79f5u32;
        for case in 0..8usize {
            let length = case * 31 % 65;
            let mut payload = vec![0; length];
            for byte in &mut payload {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *byte = (state >> 24) as u8;
            }

            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut mono = AacLcDecoder::new(4, 1).unwrap();
                let _ = mono.decode_raw_data_block_f32_strict(&payload);

                let mut stereo = AacLcDecoder::new(4, 2).unwrap();
                let _ = stereo.decode_raw_data_block_fixed_interleaved_i16_strict(&payload);

                let mut drm_mono = AacLcDecoder::new_drm_aac(3, 1).unwrap();
                let _ = drm_mono.decode_drm_aac_mono_f32(&payload);

                let mut drm_stereo = AacLcDecoder::new_drm_aac(3, 2).unwrap();
                let _ = drm_stereo.decode_drm_aac_stereo_i16(&payload);
            }));
            assert!(
                result.is_ok(),
                "AAC decoder panicked for deterministic random case {case}, length {length}"
            );
        }
    }

    #[test]
    fn sce_and_cpe_decode_facades_propagate_filterbank_mismatches() {
        let mono = zero_sce_payload(false);
        assert!(decode_aac_lc_single_channel_fixed_i16(
            &mono,
            4,
            &mut FixedLongBlockFilterbank::new(960).unwrap(),
            &mut PnsRandomState::new(1),
        )
        .is_err());

        let mut decoder = AacLcDecoder::new(4, 1).unwrap();
        decoder.channel_filterbanks[0] = LongBlockFilterbank::new(960).unwrap();
        assert!(decoder.decode_raw_data_block_f32(&mono).is_err());

        let mut decoder = AacLcDecoder::new(4, 1).unwrap();
        decoder.fixed_channel_filterbanks[0] = FixedLongBlockFilterbank::new(960).unwrap();
        assert!(decoder
            .decode_raw_data_block_fixed_interleaved_i16(&mono)
            .is_err());

        let payload = zero_cpe_payload(0);
        assert!(decode_aac_lc_channel_pair_f32(
            &payload,
            4,
            &mut LongBlockFilterbank::new(1024).unwrap(),
            &mut LongBlockFilterbank::new(960).unwrap(),
            &mut PnsRandomState::new(1),
        )
        .is_err());
        for (left_length, right_length) in [(960, 1024), (1024, 960)] {
            let mut decoder = AacLcDecoder::new(4, 2).unwrap();
            decoder.channel_filterbanks[0] = LongBlockFilterbank::new(left_length).unwrap();
            decoder.channel_filterbanks[1] = LongBlockFilterbank::new(right_length).unwrap();
            assert!(decoder.decode_raw_data_block_f32(&payload).is_err());

            let mut decoder = AacLcDecoder::new(4, 2).unwrap();
            decoder.fixed_channel_filterbanks[0] =
                FixedLongBlockFilterbank::new(left_length).unwrap();
            decoder.fixed_channel_filterbanks[1] =
                FixedLongBlockFilterbank::new(right_length).unwrap();
            assert!(decoder
                .decode_raw_data_block_fixed_interleaved_i16(&payload)
                .is_err());

            assert!(decode_aac_lc_channel_pair_fixed_interleaved_i16(
                &payload,
                4,
                &mut FixedLongBlockFilterbank::new(left_length).unwrap(),
                &mut FixedLongBlockFilterbank::new(right_length).unwrap(),
                &mut PnsRandomState::new(1),
            )
            .is_err());
            assert!(
                decode_aac_lc_channel_pair_fixed_spectrum_interleaved_i16_bridge(
                    &payload,
                    4,
                    &mut FixedLongBlockFilterbank::new(left_length).unwrap(),
                    &mut FixedLongBlockFilterbank::new(right_length).unwrap(),
                    &mut PnsRandomState::new(1),
                )
                .is_err()
            );
        }
    }

    #[test]
    fn stateful_decoder_dispatches_raw_sce_to_mono_frame() {
        let payload = zero_sce_payload(false);
        let mut decoder = AacLcDecoder::new(4, 1).unwrap();
        let decoded = decoder.decode_raw_data_block_f32(&payload).unwrap();

        assert_eq!(decoded.channels(), 1);
        assert_eq!(decoded.samples_per_channel(), 1024);
        assert!(decoded
            .interleaved_f32()
            .iter()
            .all(|sample| *sample == 0.0));
    }

    #[test]
    fn stateful_decoder_dispatches_raw_cpe_to_stereo_frame() {
        let payload = zero_cpe_payload(0);
        let mut decoder = AacLcDecoder::new(4, 2).unwrap();
        let decoded = decoder.decode_raw_data_block_f32(&payload).unwrap();

        assert_eq!(decoded.channels(), 2);
        assert_eq!(decoded.samples_per_channel(), 1024);
        assert!(decoded
            .interleaved_f32()
            .iter()
            .all(|sample| *sample == 0.0));
    }

    #[test]
    fn stateful_decoder_decodes_adts_frame_payload() {
        let payload = zero_cpe_payload(0);
        let header = AdtsHeader::aac_lc(44_100, 2, payload.len()).unwrap();
        let mut frame = vec![0; header.header_len()];
        header.write(&mut frame).unwrap();
        frame.extend_from_slice(&payload);

        let mut decoder = AacLcDecoder::from_adts_header(header).unwrap();
        let decoded = decoder.decode_adts_frame_f32(&frame).unwrap();

        assert!(matches!(decoded, DecodedAacLcFrame::Stereo(_)));
    }

    #[test]
    fn interleaves_stereo_f32_and_i16_samples() {
        assert_eq!(
            interleave_stereo_f32(&[1.0, 2.0, 3.0], &[4.0, 5.0]),
            vec![1.0, 4.0, 2.0, 5.0]
        );
        assert_eq!(f32_to_i16(1.0), i16::MAX);
        assert_eq!(f32_to_i16(-1.0), i16::MIN);
        assert_eq!(f32_to_i16(0.5), 16384);
        assert_eq!(f32_to_i16(f32::NAN), 0);
        assert_eq!(eld_raw_pcm_to_i16(f32::NAN), 0);
        assert_eq!(eld_raw_pcm_to_i16(100_000.0), i16::MAX);
        assert_eq!(
            interleave_stereo_i16(&[0.0, 1.0], &[-1.0, 0.5]),
            vec![0, i16::MIN, i16::MAX, 16384]
        );
    }

    #[test]
    fn skips_extended_data_stream_and_fill_lengths() {
        let mut data_stream = BitWriter::new();
        data_stream.write(3, 4);
        data_stream.write_bool(false);
        data_stream.write(255, 8);
        data_stream.write(1, 8);
        for _ in 0..256 {
            data_stream.write(0xa5, 8);
        }
        let bits = data_stream.bits_written();
        let bytes = data_stream.finish();
        let mut reader = BitReader::with_bit_len(&bytes, bits).unwrap();
        AacLcDecoder::new(4, 1)
            .unwrap()
            .read_data_stream_element(&mut reader)
            .unwrap();
        assert_eq!(reader.bits_read(), bits);

        let mut fill = BitWriter::new();
        fill.write(15, 4);
        fill.write(1, 8);
        for _ in 0..15 {
            fill.write(0x5a, 8);
        }
        let bits = fill.bits_written();
        let bytes = fill.finish();
        let mut reader = BitReader::with_bit_len(&bytes, bits).unwrap();
        skip_fill_element(&mut reader).unwrap();
        assert_eq!(reader.bits_read(), bits);
    }

    #[test]
    fn interleaves_multichannel_f32_and_i16_samples() {
        let channels = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![-1.0, 0.5]];
        assert_eq!(
            interleave_multichannel_f32(&channels),
            vec![1.0, 3.0, -1.0, 2.0, 4.0, 0.5]
        );
        assert_eq!(
            interleave_multichannel_i16(&channels),
            vec![i16::MAX, i16::MAX, i16::MIN, i16::MAX, i16::MAX, 16384]
        );
    }

    #[test]
    fn synthesizes_decoded_channel_stream_with_fixed_filterbank_to_i16() {
        let ics = test_ics(1);
        let stream = test_stream(
            &ics,
            test_sections(vec![ZERO_HCB]),
            vec![0],
            vec![0.0; 1024],
        );
        let mut decoder = AacLcDecoder::new(4, 1).unwrap();

        let pcm = decoder
            .synthesize_channel_stream_fixed_i16(&stream, 0)
            .unwrap();

        assert_eq!(pcm.len(), 1024);
        assert!(pcm.iter().all(|sample| *sample == 0));
        let mut invalid = stream.clone();
        invalid.spectrum.windows[0].clear();
        assert!(decoder
            .synthesize_channel_stream_fixed_q31(&invalid, 0)
            .is_err());
    }

    #[test]
    fn synthesizes_coupling_channel_stream_with_fixed_filterbank_to_i16() {
        let ics = test_ics(1);
        let stream = test_stream(
            &ics,
            test_sections(vec![ZERO_HCB]),
            vec![0],
            vec![0.0; 1024],
        );
        let mut decoder = AacLcDecoder::new(4, 1).unwrap();

        let pcm = decoder
            .synthesize_coupling_channel_stream_fixed_i16(&stream, 0)
            .unwrap();

        assert_eq!(pcm.len(), 1024);
        assert!(pcm.iter().all(|sample| *sample == 0));
        let mut invalid = stream.clone();
        invalid.spectrum.windows[0].clear();
        assert!(decoder
            .synthesize_coupling_channel_stream_fixed_q31(&invalid, 0)
            .is_err());
    }

    #[test]
    fn decoded_frame_exposes_interleaved_output() {
        let payload = zero_cpe_payload(0);
        let mut decoder = AacLcDecoder::new(4, 2).unwrap();
        let decoded = decoder.decode_raw_data_block_f32(&payload).unwrap();

        assert_eq!(decoded.channels(), 2);
        assert_eq!(decoded.samples_per_channel(), 1024);
        assert_eq!(decoded.interleaved_f32().len(), 2048);
        assert_eq!(decoded.interleaved_i16().len(), 2048);
        assert!(decoded.interleaved_i16().iter().all(|sample| *sample == 0));
    }

    #[test]
    fn stateful_decoder_decodes_interleaved_adts_helpers() {
        let payload = zero_cpe_payload(0);
        let header = AdtsHeader::aac_lc(44_100, 2, payload.len()).unwrap();
        let mut frame = vec![0; header.header_len()];
        header.write(&mut frame).unwrap();
        frame.extend_from_slice(&payload);

        let mut f32_decoder = AacLcDecoder::from_adts_header(header).unwrap();
        let mut i16_decoder = AacLcDecoder::from_adts_header(header).unwrap();
        let f32_samples = f32_decoder
            .decode_adts_frame_interleaved_f32(&frame)
            .unwrap();
        let i16_samples = i16_decoder
            .decode_adts_frame_interleaved_i16(&frame)
            .unwrap();

        assert_eq!(f32_samples.len(), 2048);
        assert_eq!(i16_samples.len(), 2048);
        assert!(f32_samples.iter().all(|sample| *sample == 0.0));
        assert!(i16_samples.iter().all(|sample| *sample == 0));
    }

    #[test]
    fn strict_raw_decode_accepts_zero_padding_and_rejects_nonzero_trailing_bits() {
        let payload = zero_sce_payload(false);
        let mut decoder = AacLcDecoder::new(4, 1).unwrap();
        decoder.decode_raw_data_block_f32_strict(&payload).unwrap();

        let mut with_trailing = payload.clone();
        with_trailing.push(0x80);
        let err = decoder
            .decode_raw_data_block_f32_strict(&with_trailing)
            .unwrap_err();

        assert_eq!(err, DecodeError::NonZeroTrailingBits(10));
        assert_eq!(
            err.to_string(),
            "AAC raw_data_block has non-zero data in 10 trailing bit(s) after decoded payload"
        );
    }

    #[test]
    fn decodes_aac_lc_960_frame_length_flag_in_float_and_fixed_paths() {
        let mut config = AudioSpecificConfig::aac_lc(44_100, 1).unwrap();
        config.ga_specific.as_mut().unwrap().frame_length_flag = true;
        let payload = zero_sce_payload(false);

        let floating = AacLcDecoder::from_audio_specific_config(&config)
            .unwrap()
            .decode_raw_data_block_multichannel_f32(&payload)
            .unwrap();
        assert_eq!(floating.channels, vec![vec![0.0; 960]]);

        let fixed = AacLcDecoder::from_audio_specific_config(&config)
            .unwrap()
            .decode_raw_data_block_multichannel_fixed_interleaved_i16(&payload)
            .unwrap();
        assert_eq!(fixed, vec![0; 960]);
    }

    #[test]
    fn strict_adts_and_fixed_decode_reject_nonzero_trailing_bits() {
        let mut payload = zero_sce_payload(false);
        payload.push(0x80);
        let header = AdtsHeader::aac_lc(44_100, 1, payload.len()).unwrap();
        let mut frame = vec![0; header.header_len()];
        header.write(&mut frame).unwrap();
        frame.extend_from_slice(&payload);

        let mut f32_decoder = AacLcDecoder::from_adts_header(header).unwrap();
        assert_eq!(
            f32_decoder
                .decode_adts_frame_f32_strict(&frame)
                .unwrap_err(),
            DecodeError::NonZeroTrailingBits(10)
        );

        let mut fixed_decoder = AacLcDecoder::from_adts_header(header).unwrap();
        assert_eq!(
            fixed_decoder
                .decode_adts_frame_fixed_interleaved_i16_strict(&frame)
                .unwrap_err(),
            DecodeError::NonZeroTrailingBits(10)
        );
    }

    #[test]
    fn decodes_crc_protected_adts_multi_raw_data_block_frame() {
        let payload = zero_sce_payload(false);
        let block_len = payload.len() + 2; // raw payload plus per-block CRC
        let mut header = AdtsHeader::aac_lc(44_100, 1, 0).unwrap();
        header.protection_absent = false;
        header.number_of_raw_data_blocks_in_frame = 1;
        header.frame_length = 7 + 2 + 2 + block_len * 2;
        header.crc_check = Some(0);

        let mut standard_header = vec![0; header.header_len()];
        header.write(&mut standard_header).unwrap();
        let mut frame = standard_header[..7].to_vec();
        frame.extend_from_slice(&(block_len as u16).to_be_bytes());
        let header_crc = crate::adts::adts_crc16(&frame);
        frame.extend_from_slice(&header_crc.to_be_bytes());
        let mut probe = AacLcDecoder::new(4, 1).unwrap();
        probe.decode_raw_data_block_f32(&payload).unwrap();
        let block_crc = adts_crc16_padded_bit_regions(
            probe
                .adts_crc_regions
                .iter()
                .cloned()
                .zip(probe.adts_crc_padded_bits.iter().copied())
                .map(|(range, padded_bits)| (payload.as_slice(), range, padded_bits)),
        )
        .unwrap();
        for _ in 0..2 {
            frame.extend_from_slice(&payload);
            frame.extend_from_slice(&block_crc.to_be_bytes());
        }

        let mut decoder = AacLcDecoder::new(4, 1).unwrap();
        let frames = decoder.decode_adts_frame_blocks_f32(&frame).unwrap();
        assert_eq!(frames.len(), 2);
        assert!(frames.iter().all(|decoded| {
            decoded.channels() == 1
                && decoded
                    .interleaved_f32()
                    .iter()
                    .all(|sample| *sample == 0.0)
        }));

        let mut fixed_decoder = AacLcDecoder::new(4, 1).unwrap();
        let fixed = fixed_decoder
            .decode_adts_frame_blocks_fixed_interleaved_i16(&frame)
            .unwrap();
        assert_eq!(fixed.len(), 2);
        assert!(fixed.iter().flatten().all(|&sample| sample == 0));
    }

    #[test]
    fn validates_single_block_adts_crc_over_cpe_syntax_regions() {
        let payload = zero_cpe_payload(0);
        let mut header = AdtsHeader::aac_lc(44_100, 2, payload.len()).unwrap();
        header.protection_absent = false;
        header.frame_length = 9 + payload.len();
        header.crc_check = Some(0);
        let mut frame = vec![0; 9];
        header.write(&mut frame).unwrap();
        frame.extend_from_slice(&payload);

        let mut probe = AacLcDecoder::new(4, 2).unwrap();
        probe.decode_raw_data_block_f32(&payload).unwrap();
        assert_eq!(probe.adts_crc_regions.len(), 2);
        let mut regions = vec![(frame.as_slice(), 0..56, 56)];
        regions.extend(
            probe
                .adts_crc_regions
                .iter()
                .cloned()
                .zip(probe.adts_crc_padded_bits.iter().copied())
                .map(|(range, padded_bits)| (payload.as_slice(), range, padded_bits)),
        );
        let crc = adts_crc16_padded_bit_regions(regions).unwrap();
        frame[7..9].copy_from_slice(&crc.to_be_bytes());

        let mut decoder = AacLcDecoder::new(4, 2).unwrap();
        assert!(matches!(
            decoder.decode_adts_frame_f32(&frame).unwrap(),
            DecodedAacLcFrame::Stereo(_)
        ));

        let mut decoder = AacLcDecoder::new(4, 2).unwrap();
        assert!(matches!(
            decoder.decode_adts_frame_f32_strict(&frame).unwrap(),
            DecodedAacLcFrame::Stereo(_)
        ));
        let mut decoder = AacLcDecoder::new(4, 2).unwrap();
        assert_eq!(
            decoder
                .decode_adts_frame_fixed_interleaved_i16(&frame)
                .unwrap()
                .len(),
            2048
        );
        let mut decoder = AacLcDecoder::new(4, 2).unwrap();
        assert_eq!(
            decoder
                .decode_adts_frame_fixed_interleaved_i16_strict(&frame)
                .unwrap()
                .len(),
            2048
        );
        let mut decoder = AacLcDecoder::new(4, 2).unwrap();
        assert_eq!(
            decoder
                .decode_adts_frame_multichannel_f32(&frame)
                .unwrap()
                .channels(),
            2
        );
        let mut decoder = AacLcDecoder::new(4, 2).unwrap();
        assert_eq!(
            decoder
                .decode_adts_frame_multichannel_f32_strict(&frame)
                .unwrap()
                .channels(),
            2
        );
        let mut decoder = AacLcDecoder::new(4, 2).unwrap();
        assert_eq!(
            decoder
                .decode_adts_frame_multichannel_fixed_interleaved_i16(&frame)
                .unwrap()
                .len(),
            2048
        );
        let mut decoder = AacLcDecoder::new(4, 2).unwrap();
        assert_eq!(
            decoder
                .decode_adts_frame_multichannel_fixed_interleaved_i16_strict(&frame)
                .unwrap()
                .len(),
            2048
        );

        frame[7] ^= 1;
        macro_rules! assert_crc_mismatch {
            ($method:ident) => {{
                let mut decoder = AacLcDecoder::new(4, 2).unwrap();
                assert!(matches!(
                    decoder.$method(&frame),
                    Err(DecodeError::Adts(AdtsError::CrcMismatch { .. }))
                ));
            }};
        }
        assert_crc_mismatch!(decode_adts_frame_f32);
        assert_crc_mismatch!(decode_adts_frame_f32_strict);
        assert_crc_mismatch!(decode_adts_frame_fixed_interleaved_i16);
        assert_crc_mismatch!(decode_adts_frame_fixed_interleaved_i16_strict);
        assert_crc_mismatch!(decode_adts_frame_multichannel_f32);
        assert_crc_mismatch!(decode_adts_frame_multichannel_f32_strict);
        assert_crc_mismatch!(decode_adts_frame_multichannel_fixed_interleaved_i16);
        assert_crc_mismatch!(decode_adts_frame_multichannel_fixed_interleaved_i16_strict);
    }

    #[test]
    fn constructs_public_decoder_for_mono_usac_aot42() {
        let usac = crate::asc::UsacConfig {
            sampling_frequency_index: 3,
            sampling_frequency: 48_000,
            core_sbr_frame_length_index: 1,
            core_frame_length: 1024,
            output_frame_length: 1024,
            sbr_ratio_index: 0,
            channel_configuration_index: 1,
            elements: vec![crate::asc::UsacElementConfig::SingleChannel {
                noise_filling: false,
                sbr: None,
            }],
            extensions: Vec::new(),
        };
        let asc = AudioSpecificConfig {
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
        let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
        assert_eq!(decoder.audio_object_type(), 42);
        assert_eq!(decoder.frame_length(), 1024);
        assert!(decoder.usac_decoder.is_some());
        decoder.clear_history().unwrap();
        assert!(decoder.usac_decoder.is_some());
        assert_eq!(decoder.stream_info().sample_rate, 48_000);

        let mut payload = BitWriter::new();
        payload.write_bool(true); // independent
        payload.write_bool(false); // FD core
        payload.write_bool(false); // TNS inactive
        payload.write(0, 8); // global gain
        payload.write(0, 2); // ONLY_LONG
        payload.write_bool(false); // window shape
        payload.write(0, 6); // max_sfb
        payload.write_bool(false); // no FAC
        let payload = payload.finish();
        let frame = decoder.decode_usac_access_unit_f32(&payload).unwrap();
        assert_eq!(frame.samples, vec![0.0; 1024]);

        let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
        assert_eq!(
            decoder
                .decode_usac_access_unit_multichannel_f32(&payload)
                .unwrap(),
            vec![vec![0.0; 1024]]
        );
    }

    #[test]
    fn usac_audio_preroll_decodes_embedded_access_units_before_current_frame() {
        let usac = crate::asc::UsacConfig {
            sampling_frequency_index: 3,
            sampling_frequency: 48_000,
            core_sbr_frame_length_index: 1,
            core_frame_length: 1024,
            output_frame_length: 1024,
            sbr_ratio_index: 0,
            channel_configuration_index: 1,
            elements: vec![
                crate::asc::UsacElementConfig::Extension(crate::asc::UsacExtElementConfig {
                    extension_type: 3,
                    default_length: None,
                    payload_fragmentation: false,
                    config: Vec::new(),
                }),
                crate::asc::UsacElementConfig::SingleChannel {
                    noise_filling: false,
                    sbr: None,
                },
            ],
            extensions: Vec::new(),
        };
        let in_band_config = usac.to_bytes().unwrap();
        assert!(in_band_config.len() < 15);
        let asc = AudioSpecificConfig {
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

        fn write_silent_core(writer: &mut BitWriter) {
            writer.write_bool(false); // FD core
            writer.write_bool(false); // TNS inactive
            writer.write(0, 8); // global gain
            writer.write(0, 2); // ONLY_LONG
            writer.write_bool(false); // window shape
            writer.write(0, 6); // max_sfb
            writer.write_bool(false); // no FAC
        }

        let mut embedded = BitWriter::new();
        embedded.write_bool(true); // independent
        embedded.write_bool(false); // no nested AudioPreRoll payload
        write_silent_core(&mut embedded);
        let embedded = embedded.finish();

        let mut preroll_payload = BitWriter::new();
        preroll_payload.write(in_band_config.len() as u32, 4); // configLength
        for byte in in_band_config {
            preroll_payload.write(byte.into(), 8);
        }
        preroll_payload.write_bool(false); // applyCrossfade
        preroll_payload.write_bool(false); // reserved
        preroll_payload.write(1, 2); // numPreRollAU
        preroll_payload.write(embedded.len() as u32, 16);
        for byte in embedded {
            preroll_payload.write(byte.into(), 8);
        }
        let preroll_payload = preroll_payload.finish();

        let mut outer = BitWriter::new();
        outer.write_bool(true); // independent
        outer.write_bool(true); // AudioPreRoll payload present
        outer.write_bool(false); // explicit payload length
        outer.write(preroll_payload.len() as u32, 8);
        for byte in preroll_payload {
            outer.write(byte.into(), 8);
        }
        write_silent_core(&mut outer);

        let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
        assert_eq!(
            decoder
                .decode_usac_access_unit_multichannel_f32(&outer.finish())
                .unwrap(),
            vec![vec![0.0; 1024]]
        );
        assert!(decoder.pending_usac_audio_preroll.is_none());
    }

    #[test]
    fn usac_audio_preroll_reconfigures_mono_to_stereo_before_current_frame() {
        let preroll_element =
            crate::asc::UsacElementConfig::Extension(crate::asc::UsacExtElementConfig {
                extension_type: 3,
                default_length: None,
                payload_fragmentation: false,
                config: Vec::new(),
            });
        let mono = crate::asc::UsacConfig {
            sampling_frequency_index: 3,
            sampling_frequency: 48_000,
            core_sbr_frame_length_index: 1,
            core_frame_length: 1024,
            output_frame_length: 1024,
            sbr_ratio_index: 0,
            channel_configuration_index: 1,
            elements: vec![
                preroll_element.clone(),
                crate::asc::UsacElementConfig::SingleChannel {
                    noise_filling: false,
                    sbr: None,
                },
            ],
            extensions: Vec::new(),
        };
        let stereo = crate::asc::UsacConfig {
            channel_configuration_index: 2,
            elements: vec![
                preroll_element,
                crate::asc::UsacElementConfig::ChannelPair {
                    noise_filling: false,
                    sbr: None,
                    stereo_config_index: 0,
                    mps212: None,
                },
            ],
            ..mono.clone()
        };
        let config = stereo.to_bytes().unwrap();
        assert!(config.len() < 15);
        let asc = AudioSpecificConfig {
            audio_object_type: 42,
            sampling_frequency_index: 3,
            sampling_frequency: 48_000,
            channel_configuration: 1,
            extension: None,
            ga_specific: None,
            eld_specific: None,
            usac_config: Some(mono),
            error_protection_config: None,
            program_config: None,
            bits_read: 0,
        };

        let mut payload = BitWriter::new();
        payload.write(config.len() as u32, 4);
        for byte in config {
            payload.write(byte.into(), 8);
        }
        payload.write_bool(false); // applyCrossfade
        payload.write_bool(false); // reserved
        payload.write(0, 2); // no embedded AUs
        let payload = payload.finish();

        let mut access_unit = BitWriter::new();
        access_unit.write_bool(true); // independent
        access_unit.write_bool(true); // AudioPreRoll present
        access_unit.write_bool(false); // explicit length
        access_unit.write(payload.len() as u32, 8);
        for byte in payload {
            access_unit.write(byte.into(), 8);
        }
        access_unit.write_bool(false); // left FD
        access_unit.write_bool(false); // right FD
        access_unit.write_bool(false); // TNS inactive
        access_unit.write_bool(false); // separate windows
        access_unit.write_bool(false); // common_tw
        for _ in 0..2 {
            access_unit.write(0, 8); // global gain
            access_unit.write(0, 2); // ONLY_LONG
            access_unit.write_bool(false); // window shape
            access_unit.write(0, 6); // max_sfb
            access_unit.write_bool(false); // no FAC
        }

        let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
        assert_eq!(
            decoder
                .decode_usac_access_unit_multichannel_f32(&access_unit.finish())
                .unwrap(),
            vec![vec![0.0; 1024]; 2]
        );
        assert!(decoder.usac_stereo_decoder.is_some());
        assert_eq!(decoder.channel_configuration(), 2);
    }

    #[test]
    fn usac_audio_preroll_crossfade_uses_128_sample_linear_ramp() {
        let mut decoder = AacLcDecoder::new_ga(42, 3, 1).unwrap();
        decoder.usac_last_output = Some(vec![vec![1.0; 1024]]);
        decoder.pending_usac_audio_preroll = Some(crate::audio_preroll::AudioPreRoll {
            config: Vec::new(),
            apply_crossfade: true,
            access_units: Vec::new(),
            bits_read: 0,
        });
        decoder.decode_pending_usac_audio_preroll().unwrap();
        let mut channels = vec![vec![0.0; 1024]];
        decoder.finish_usac_output(&mut channels);
        assert_eq!(channels[0][0], 1.0);
        assert_eq!(channels[0][64], 0.5);
        assert_eq!(channels[0][127], 1.0 / 128.0);
        assert_eq!(channels[0][128], 0.0);
    }

    #[test]
    fn usac_in_band_uni_drc_config_loudness_and_gain_update_stream_info() {
        let mut drc_config = BitWriter::new();
        drc_config.write_bool(false); // sample rate absent
        drc_config.write(0, 7); // no downmix instructions
        drc_config.write_bool(false); // no basic DRC
        drc_config.write(1, 3); // one coefficient set
        drc_config.write(1, 6); // one instruction
        drc_config.write(1, 7); // mono channel layout
        drc_config.write_bool(false); // no explicit speaker positions
        drc_config.write(1, 4); // coefficient location
        drc_config.write_bool(false); // frame size absent
        drc_config.write(1, 6); // one gain set
        drc_config.write(3, 2); // constant coding profile
        drc_config.write_bool(true); // linear interpolation
        drc_config.write_bool(true); // full frame
        drc_config.write_bool(false); // no time alignment
        drc_config.write_bool(false); // default timeDeltaMin
        drc_config.write(1, 7); // CICP characteristic
        drc_config.write(3, 6); // drcSetId
        drc_config.write(1, 4); // drcLocation
        drc_config.write(0, 7); // base-layout downmix
        drc_config.write_bool(false); // no additional downmix IDs
        drc_config.write(0, 16); // no requested effect
        drc_config.write_bool(false); // no limiter target
        drc_config.write_bool(false); // no target loudness range
        drc_config.write_bool(false); // no dependency
        drc_config.write_bool(false); // independent use allowed
        drc_config.write(1, 6); // gain-set index zero
        drc_config.write_bool(false); // no channel repeat
        drc_config.write_bool(false); // default gain scaling
        drc_config.write_bool(false); // no gain offset
        drc_config.write_bool(false); // no config extension

        let mut loudness = BitWriter::new();
        loudness.write(0, 6); // no album entries
        loudness.write(1, 6); // one track entry
        loudness.write(3, 6); // matching drcSetId
        loudness.write(0, 7); // base-layout downmix
        loudness.write_bool(false); // no sample peak
        loudness.write_bool(false); // no true peak
        loudness.write(1, 4); // one measurement
        loudness.write(1, 4); // program loudness
        loudness.write(127, 8); // -26 dB
        loudness.write(1, 4); // measurement system
        loudness.write(3, 2); // reliability
        loudness.write_bool(false); // no loudness extension

        let usac = crate::asc::UsacConfig {
            sampling_frequency_index: 3,
            sampling_frequency: 48_000,
            core_sbr_frame_length_index: 1,
            core_frame_length: 1024,
            output_frame_length: 1024,
            sbr_ratio_index: 0,
            channel_configuration_index: 1,
            elements: vec![
                crate::asc::UsacElementConfig::SingleChannel {
                    noise_filling: false,
                    sbr: None,
                },
                crate::asc::UsacElementConfig::Extension(crate::asc::UsacExtElementConfig {
                    extension_type: 4,
                    default_length: Some(1),
                    payload_fragmentation: false,
                    config: drc_config.finish(),
                }),
            ],
            extensions: vec![crate::asc::UsacConfigExtension {
                extension_type: 2,
                data: loudness.finish(),
            }],
        };
        let asc = AudioSpecificConfig {
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
        let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
        assert!(decoder.drc_config.is_some());
        assert!(decoder.drc_loudness_info.is_some());
        assert_eq!(decoder.stream_info().output_loudness, -1);

        let mut access_unit = BitWriter::new();
        access_unit.write_bool(true); // independent
        access_unit.write_bool(false); // FD core
        access_unit.write_bool(false); // TNS inactive
        access_unit.write(0, 8); // global gain
        access_unit.write(0, 2); // ONLY_LONG
        access_unit.write_bool(false); // window shape
        access_unit.write(0, 6); // max_sfb
        access_unit.write_bool(false); // no FAC
        access_unit.write_bool(true); // extension present
        access_unit.write_bool(true); // use one-byte default length
        access_unit.write_bool(false); // no gain extension (constant sequence)
        access_unit.write(0, 7); // extension byte padding
        decoder
            .decode_usac_access_unit_f32(&access_unit.finish())
            .unwrap();
        assert!(decoder.drc_gain.is_some());
        assert_eq!(decoder.stream_info().output_loudness, 104);

        let mut leading_asc = asc.clone();
        leading_asc
            .usac_config
            .as_mut()
            .unwrap()
            .elements
            .swap(0, 1);
        let mut leading = AacLcDecoder::from_audio_specific_config(&leading_asc).unwrap();
        let mut access_unit = BitWriter::new();
        access_unit.write_bool(true); // independent
        access_unit.write_bool(true); // leading extension present
        access_unit.write_bool(true); // use default length
        access_unit.write_bool(false); // no gain extension
        access_unit.write(0, 7); // extension byte padding
        access_unit.write_bool(false); // FD core
        access_unit.write_bool(false); // TNS inactive
        access_unit.write(0, 8); // global gain
        access_unit.write(0, 2); // ONLY_LONG
        access_unit.write_bool(false); // window shape
        access_unit.write(0, 6); // max_sfb
        access_unit.write_bool(false); // no FAC
        leading
            .decode_usac_access_unit_f32(&access_unit.finish())
            .unwrap();
        assert_eq!(leading.stream_info().output_loudness, 104);

        let mut mps_asc = asc.clone();
        mps_asc.channel_configuration = 2;
        let mps_config = mps_asc.usac_config.as_mut().unwrap();
        mps_config.channel_configuration_index = 2;
        let drc_extension = mps_config.elements[1].clone();
        mps_config.elements = vec![
            crate::asc::UsacElementConfig::ChannelPair {
                noise_filling: false,
                sbr: None,
                stereo_config_index: 1,
                mps212: Some(crate::asc::Mps212Config {
                    frequency_resolution_index: 1,
                    frequency_resolution_bands: 28,
                    fixed_gain_downmix: 0,
                    temporal_shape_config: 0,
                    decorrelation_config: 0,
                    high_rate_mode: false,
                    phase_coding: false,
                    ott_bands_phase: None,
                    residual_bands: None,
                    pseudo_lr: false,
                    environment_quantization_mode: None,
                }),
            },
            drc_extension,
        ];
        let mut mps = AacLcDecoder::from_audio_specific_config(&mps_asc).unwrap();
        let mut access_unit = BitWriter::new();
        access_unit.write_bool(true); // independent
        access_unit.write_bool(false); // FD downmix
        access_unit.write_bool(false); // TNS inactive
        access_unit.write(0, 8); // global gain
        access_unit.write(0, 2); // ONLY_LONG
        access_unit.write_bool(false); // window shape
        access_unit.write(0, 6); // max_sfb
        access_unit.write_bool(false); // no FAC
        access_unit.write(0, 2); // default CLD
        access_unit.write(0, 2); // default ICC
        access_unit.write_bool(true); // trailing DRC extension present
        access_unit.write_bool(true); // use default length
        access_unit.write_bool(false); // no gain extension
        access_unit.write(0, 7); // extension byte padding
        mps.decode_usac_mps212_access_unit(&access_unit.finish())
            .unwrap();
        assert_eq!(mps.stream_info().output_loudness, 104);
    }

    #[test]
    fn public_usac_multichannel_dispatches_stereo_and_mps212() {
        let base = crate::asc::UsacConfig {
            sampling_frequency_index: 3,
            sampling_frequency: 48_000,
            core_sbr_frame_length_index: 1,
            core_frame_length: 1024,
            output_frame_length: 1024,
            sbr_ratio_index: 0,
            channel_configuration_index: 2,
            elements: Vec::new(),
            extensions: Vec::new(),
        };
        let asc = |usac_config| AudioSpecificConfig {
            audio_object_type: 42,
            sampling_frequency_index: 3,
            sampling_frequency: 48_000,
            channel_configuration: 2,
            extension: None,
            ga_specific: None,
            eld_specific: None,
            usac_config: Some(usac_config),
            error_protection_config: None,
            program_config: None,
            bits_read: 0,
        };

        let mut stereo = base.clone();
        stereo.elements = vec![crate::asc::UsacElementConfig::ChannelPair {
            noise_filling: false,
            sbr: None,
            stereo_config_index: 0,
            mps212: None,
        }];
        let mut stereo_payload = BitWriter::new();
        stereo_payload.write_bool(true); // independent
        stereo_payload.write_bool(false); // left FD
        stereo_payload.write_bool(false); // right FD
        stereo_payload.write_bool(false); // TNS inactive
        stereo_payload.write_bool(false); // separate windows
        stereo_payload.write_bool(false); // common_tw
        for _ in 0..2 {
            stereo_payload.write(0, 8); // global gain
            stereo_payload.write(0, 2); // ONLY_LONG
            stereo_payload.write_bool(false); // window shape
            stereo_payload.write(0, 6); // max_sfb
            stereo_payload.write_bool(false); // no FAC
        }
        let mut decoder = AacLcDecoder::from_audio_specific_config(&asc(stereo)).unwrap();
        let channels = decoder
            .decode_usac_access_unit_multichannel_f32(&stereo_payload.finish())
            .unwrap();
        assert_eq!(channels, vec![vec![0.0; 1024], vec![0.0; 1024]]);

        let mut mps = base.clone();
        mps.elements = vec![crate::asc::UsacElementConfig::ChannelPair {
            noise_filling: false,
            sbr: None,
            stereo_config_index: 1,
            mps212: Some(crate::asc::Mps212Config {
                frequency_resolution_index: 1,
                frequency_resolution_bands: 28,
                fixed_gain_downmix: 0,
                temporal_shape_config: 0,
                decorrelation_config: 0,
                high_rate_mode: false,
                phase_coding: false,
                ott_bands_phase: None,
                residual_bands: None,
                pseudo_lr: false,
                environment_quantization_mode: None,
            }),
        }];
        let mps_asc = asc(mps);
        let mut mps_payload = BitWriter::new();
        mps_payload.write_bool(true); // independent
        mps_payload.write_bool(false); // FD downmix
        mps_payload.write_bool(false); // TNS inactive
        mps_payload.write(0, 8); // global gain
        mps_payload.write(0, 2); // ONLY_LONG
        mps_payload.write_bool(false); // window shape
        mps_payload.write(0, 6); // max_sfb
        mps_payload.write_bool(false); // no FAC
        mps_payload.write(0, 2); // default CLD
        mps_payload.write(0, 2); // default ICC
        let mps_payload = mps_payload.finish();
        let mut decoder = AacLcDecoder::from_audio_specific_config(&mps_asc).unwrap();
        let access_unit = decoder
            .decode_usac_mps212_access_unit(&mps_payload)
            .unwrap();
        assert_eq!(access_unit.downmix.samples, vec![0.0; 1024]);

        let mut decoder = AacLcDecoder::from_audio_specific_config(&mps_asc).unwrap();
        assert!(matches!(
            decoder.decode_usac_access_unit_multichannel_f32(&mps_payload),
            Err(UsacDecodeError::Mps(_))
        ));

        let sbr_header = LdSbrHeader {
            amp_resolution: true,
            start_frequency: 5,
            stop_frequency: 8,
            crossover_band: 2,
            frequency_scale: Some(1),
            alter_scale: Some(false),
            noise_bands: Some(2),
            limiter_bands: Some(2),
            limiter_gains: Some(2),
            interpol_frequency: Some(true),
            smoothing_mode: Some(true),
            ..LdSbrHeader::default()
        };
        let tables = LdSbrFrequencyTables::from_header(&sbr_header, 44_100).unwrap();
        let zero = sbr_huffman_code(SbrHuffmanBook::EnvelopeLevel30Frequency, 0);
        let mut sbr_mps = base;
        sbr_mps.sampling_frequency_index = 4;
        sbr_mps.sampling_frequency = 44_100;
        sbr_mps.core_sbr_frame_length_index = 3;
        sbr_mps.sbr_ratio_index = 3;
        sbr_mps.output_frame_length = 2048;
        sbr_mps.elements = vec![crate::asc::UsacElementConfig::ChannelPair {
            noise_filling: false,
            sbr: Some(crate::asc::UsacSbrConfig {
                harmonic_sbr: false,
                inter_tes: false,
                pvc: false,
                start_frequency: 5,
                stop_frequency: 8,
                frequency_scale: Some(1),
                alter_scale: Some(false),
                noise_bands: Some(2),
                limiter_bands: Some(2),
                limiter_gains: Some(2),
                interpol_frequency: Some(true),
                smoothing_mode: Some(true),
            }),
            stereo_config_index: 1,
            mps212: Some(crate::asc::Mps212Config {
                frequency_resolution_index: 1,
                frequency_resolution_bands: 28,
                fixed_gain_downmix: 0,
                temporal_shape_config: 0,
                decorrelation_config: 0,
                high_rate_mode: false,
                phase_coding: false,
                ott_bands_phase: None,
                residual_bands: None,
                pseudo_lr: false,
                environment_quantization_mode: None,
            }),
        }];
        let mut sbr_asc = asc(sbr_mps);
        sbr_asc.sampling_frequency_index = 4;
        sbr_asc.sampling_frequency = 44_100;

        let mut payload = BitWriter::new();
        payload.write_bool(true); // independent core frame
        payload.write_bool(false); // FD core
        payload.write_bool(false); // TNS absent
        payload.write(0, 8); // global gain
        payload.write(0, 2); // ONLY_LONG
        payload.write_bool(false); // window shape
        payload.write(0, 6); // max_sfb
        payload.write_bool(false); // no FAC
        payload.write_bool(true); // SBR amp resolution
        payload.write(2, 4); // crossover
        payload.write_bool(false); // preprocessing
        payload.write(0, 2); // ordinary SBR
        payload.write_bool(true); // use default SBR header
        payload.write(0, 2); // FIXFIX
        payload.write(0, 2); // one envelope
        payload.write_bool(true); // high frequency resolution
        for _ in 0..tables.noise_band_count() {
            payload.write(0, 2); // inverse filtering off
        }
        payload.write(8, 6); // absolute envelope
        for _ in 1..tables.high_band_count() {
            for &bit in &zero {
                payload.write_bool(bit);
            }
        }
        payload.write_bool(false); // inter-TES inactive
        payload.write(4, 5); // absolute noise
        for _ in 1..tables.noise_band_count() {
            for &bit in &zero {
                payload.write_bool(bit);
            }
        }
        payload.write_bool(false); // no harmonics
        payload.write(0, 2); // default CLD
        payload.write(0, 2); // default ICC
        let mut decoder = AacLcDecoder::from_audio_specific_config(&sbr_asc).unwrap();
        let channels = decoder
            .decode_usac_access_unit_multichannel_f32(&payload.finish())
            .unwrap();
        assert_eq!(channels.len(), 2);
        assert!(channels.iter().all(|channel| channel.len() == 2048));
    }

    #[test]
    fn audio_specific_config_constructor_covers_usac_and_er_selection_edges() {
        let usac = crate::asc::UsacConfig {
            sampling_frequency_index: 3,
            sampling_frequency: 48_000,
            core_sbr_frame_length_index: 1,
            core_frame_length: 1024,
            output_frame_length: 1024,
            sbr_ratio_index: 0,
            channel_configuration_index: 1,
            elements: vec![crate::asc::UsacElementConfig::SingleChannel {
                noise_filling: false,
                sbr: None,
            }],
            extensions: Vec::new(),
        };
        let usac_asc = |usac_config| AudioSpecificConfig {
            audio_object_type: 42,
            sampling_frequency_index: 3,
            sampling_frequency: 48_000,
            channel_configuration: 1,
            extension: None,
            ga_specific: None,
            eld_specific: None,
            usac_config,
            error_protection_config: None,
            program_config: None,
            bits_read: 0,
        };
        assert_eq!(
            AacLcDecoder::from_audio_specific_config(&usac_asc(None)).unwrap_err(),
            DecodeError::UnsupportedAudioObjectType(42)
        );

        let mut invalid_usac_frame = usac.clone();
        invalid_usac_frame.core_frame_length = 960;
        assert_eq!(
            AacLcDecoder::from_audio_specific_config(&usac_asc(Some(invalid_usac_frame)))
                .unwrap_err(),
            DecodeError::UnsupportedFrameLength(960)
        );

        let mut invalid_mono = usac.clone();
        invalid_mono.elements = vec![crate::asc::UsacElementConfig::ChannelPair {
            noise_filling: false,
            sbr: None,
            stereo_config_index: 0,
            mps212: None,
        }];
        assert_eq!(
            AacLcDecoder::from_audio_specific_config(&usac_asc(Some(invalid_mono))).unwrap_err(),
            DecodeError::UnsupportedChannelConfiguration(1)
        );

        let mut dual_sce = usac.clone();
        dual_sce.channel_configuration_index = 8;
        dual_sce
            .elements
            .push(crate::asc::UsacElementConfig::SingleChannel {
                noise_filling: false,
                sbr: None,
            });
        let mut dual_sce_asc = usac_asc(Some(dual_sce));
        dual_sce_asc.channel_configuration = 8;
        let mut dual_sce_decoder = AacLcDecoder::from_audio_specific_config(&dual_sce_asc).unwrap();
        assert_eq!(dual_sce_decoder.configured_usac_channels(), 2);
        assert!(dual_sce_decoder.usac_multichannel_decoder.is_some());
        let mut dual_sce_frame = BitWriter::new();
        dual_sce_frame.write_bool(true); // usacIndependencyFlag
        for _ in 0..2 {
            dual_sce_frame.write_bool(false); // SCE FD core mode
            dual_sce_frame.write_bool(false); // SCE tns_data_present
            dual_sce_frame.write(0, 8); // SCE global_gain
            dual_sce_frame.write(0, 2); // ONLY_LONG
            dual_sce_frame.write_bool(false);
            dual_sce_frame.write(0, 6); // max_sfb
            dual_sce_frame.write_bool(false); // no FAC
        }
        assert_eq!(
            dual_sce_decoder
                .decode_usac_access_unit_multichannel_f32(&dual_sce_frame.finish())
                .unwrap(),
            vec![vec![0.0; 1024]; 2]
        );

        let mut interleaved_extension_asc = dual_sce_asc.clone();
        interleaved_extension_asc
            .usac_config
            .as_mut()
            .unwrap()
            .elements
            .insert(
                1,
                crate::asc::UsacElementConfig::Extension(crate::asc::UsacExtElementConfig {
                    extension_type: 7,
                    default_length: Some(1),
                    payload_fragmentation: false,
                    config: Vec::new(),
                }),
            );
        let mut interleaved_decoder =
            AacLcDecoder::from_audio_specific_config(&interleaved_extension_asc).unwrap();
        let mut interleaved_frame = BitWriter::new();
        interleaved_frame.write_bool(true); // usacIndependencyFlag
        interleaved_frame.write_bool(false); // SCE FD core mode
        interleaved_frame.write_bool(false); // no SCE TNS
        interleaved_frame.write(0, 8); // SCE global_gain
        interleaved_frame.write(0, 2); // ONLY_LONG
        interleaved_frame.write_bool(false);
        interleaved_frame.write(0, 6); // max_sfb
        interleaved_frame.write_bool(false); // no FAC
        interleaved_frame.write_bool(true); // extension present
        interleaved_frame.write_bool(true); // use default length (one byte)
        interleaved_frame.write(0xaa, 8); // ignored extension payload
        interleaved_frame.write_bool(false); // second SCE FD core mode
        interleaved_frame.write_bool(false); // no SCE TNS
        interleaved_frame.write(0, 8); // SCE global_gain
        interleaved_frame.write(0, 2); // ONLY_LONG
        interleaved_frame.write_bool(false);
        interleaved_frame.write(0, 6); // max_sfb
        interleaved_frame.write_bool(false); // no FAC
        assert_eq!(
            interleaved_decoder
                .decode_usac_access_unit_multichannel_f32(&interleaved_frame.finish())
                .unwrap(),
            vec![vec![0.0; 1024]; 2]
        );

        let mut stereo = usac.clone();
        stereo.channel_configuration_index = 2;
        stereo.elements = vec![crate::asc::UsacElementConfig::ChannelPair {
            noise_filling: false,
            sbr: None,
            stereo_config_index: 0,
            mps212: None,
        }];
        let mut stereo_asc = usac_asc(Some(stereo));
        stereo_asc.channel_configuration = 2;
        let stereo_decoder = AacLcDecoder::from_audio_specific_config(&stereo_asc).unwrap();
        assert!(stereo_decoder.usac_stereo_decoder.is_some());

        let mut invalid_stereo_asc = stereo_asc.clone();
        invalid_stereo_asc.usac_config.as_mut().unwrap().elements =
            vec![crate::asc::UsacElementConfig::SingleChannel {
                noise_filling: false,
                sbr: None,
            }];
        assert_eq!(
            AacLcDecoder::from_audio_specific_config(&invalid_stereo_asc).unwrap_err(),
            DecodeError::UnsupportedChannelConfiguration(2)
        );

        let mut invalid_mps = usac;
        invalid_mps.channel_configuration_index = 2;
        invalid_mps.elements = vec![crate::asc::UsacElementConfig::ChannelPair {
            noise_filling: false,
            sbr: None,
            stereo_config_index: 1,
            mps212: Some(crate::asc::Mps212Config {
                frequency_resolution_index: 1,
                frequency_resolution_bands: 28,
                fixed_gain_downmix: 0,
                temporal_shape_config: 0,
                decorrelation_config: 3,
                high_rate_mode: false,
                phase_coding: false,
                ott_bands_phase: None,
                residual_bands: None,
                pseudo_lr: false,
                environment_quantization_mode: None,
            }),
        }];
        let mut invalid_mps_asc = usac_asc(Some(invalid_mps));
        invalid_mps_asc.channel_configuration = 2;
        assert_eq!(
            AacLcDecoder::from_audio_specific_config(&invalid_mps_asc).unwrap_err(),
            DecodeError::UnsupportedChannelConfiguration(2)
        );

        let mut invalid_extension = AudioSpecificConfig::aac_lc(44_100, 1).unwrap();
        invalid_extension.extension = Some(crate::asc::AudioSpecificConfigExtension {
            audio_object_type: 7,
            sampling_frequency_index: 4,
            sampling_frequency: 44_100,
            ps_present: false,
        });
        assert_eq!(
            AacLcDecoder::from_audio_specific_config(&invalid_extension).unwrap_err(),
            DecodeError::UnsupportedAudioObjectType(7)
        );

        invalid_extension
            .extension
            .as_mut()
            .unwrap()
            .audio_object_type = 5;
        invalid_extension.channel_configuration = 0;
        assert_eq!(
            AacLcDecoder::from_audio_specific_config(&invalid_extension).unwrap_err(),
            DecodeError::UnsupportedChannelConfiguration(0)
        );

        let mut er_without_channels = AudioSpecificConfig::aac_lc(44_100, 1).unwrap();
        er_without_channels.audio_object_type = 17;
        er_without_channels.channel_configuration = 0;
        assert_eq!(
            AacLcDecoder::from_audio_specific_config(&er_without_channels).unwrap_err(),
            DecodeError::UnsupportedChannelConfiguration(0)
        );

        let mut missing_eld = AudioSpecificConfig::aac_lc(44_100, 1).unwrap();
        missing_eld.audio_object_type = 39;
        missing_eld.ga_specific = None;
        missing_eld.eld_specific = None;
        assert_eq!(
            AacLcDecoder::from_audio_specific_config(&missing_eld).unwrap_err(),
            DecodeError::UnsupportedAudioObjectType(39)
        );

        let mut er_960 = AudioSpecificConfig::aac_lc(44_100, 1).unwrap();
        er_960.audio_object_type = 17;
        er_960.ga_specific.as_mut().unwrap().frame_length_flag = true;
        assert_eq!(
            AacLcDecoder::from_audio_specific_config(&er_960)
                .unwrap()
                .frame_length(),
            960
        );

        let mut invalid_frequency = AudioSpecificConfig::aac_lc(44_100, 1).unwrap();
        invalid_frequency.sampling_frequency_index = 13;
        assert_eq!(
            AacLcDecoder::from_audio_specific_config(&invalid_frequency).unwrap_err(),
            DecodeError::UnsupportedSamplingFrequencyIndex(13)
        );

        assert_eq!(
            AacLcDecoder::new_drm_aac(13, 1).unwrap_err(),
            DecodeError::UnsupportedSamplingFrequencyIndex(13)
        );
        assert_eq!(
            AacLcDecoder::new_drm_aac(3, 8).unwrap_err(),
            DecodeError::UnsupportedChannelConfiguration(8)
        );
    }

    #[test]
    fn multichannel_orchestrators_propagate_truncated_cpe_and_cce_elements() {
        let asc = AudioSpecificConfig::aac_lc(44_100, 1).unwrap();
        for element_id in [ElementId::ChannelPair, ElementId::CouplingChannel] {
            let mut writer = BitWriter::new();
            writer.write(element_id.bits() as u32, 3);
            if element_id == ElementId::ChannelPair {
                writer.write(0, 4); // element_instance_tag
            }
            let payload = writer.finish();

            let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
            assert!(decoder
                .decode_raw_data_block_multichannel_f32(&payload)
                .unwrap_err()
                .is_unexpected_eof());

            let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
            assert!(decoder
                .decode_raw_data_block_fixed_interleaved_i16(&payload)
                .unwrap_err()
                .is_unexpected_eof());
        }

        for channels in [1, 2] {
            let mut er_asc = AudioSpecificConfig::aac_lc(44_100, channels).unwrap();
            er_asc.audio_object_type = 17;
            er_asc.error_protection_config = Some(0);

            let mut decoder = AacLcDecoder::from_audio_specific_config(&er_asc).unwrap();
            assert!(decoder
                .decode_raw_data_block_multichannel_f32(&[])
                .unwrap_err()
                .is_unexpected_eof());

            let mut decoder = AacLcDecoder::from_audio_specific_config(&er_asc).unwrap();
            assert!(decoder
                .decode_raw_data_block_fixed_interleaved_i16(&[])
                .unwrap_err()
                .is_unexpected_eof());
        }
    }

    #[test]
    fn rejects_aac_main_at_configuration_like_fdk() {
        let mut header = AdtsHeader::aac_lc(44_100, 1, 0).unwrap();
        header.profile = 0;
        assert_eq!(
            AacLcDecoder::from_adts_header(header).unwrap_err(),
            DecodeError::UnsupportedAudioObjectType(1)
        );
        let mut asc = AudioSpecificConfig::aac_lc(44_100, 1).unwrap();
        asc.audio_object_type = 1;
        assert_eq!(
            AacLcDecoder::from_audio_specific_config(&asc).unwrap_err(),
            DecodeError::UnsupportedAudioObjectType(1)
        );
        let pce = ProgramConfig {
            profile: 0,
            sampling_frequency_index: 4,
            front: vec![crate::asc::ProgramElement {
                is_cpe: false,
                tag_select: 0,
            }],
            num_channels: 1,
            num_effective_channels: 1,
            ..ProgramConfig::default()
        };
        let adif = AdifHeader {
            copyright_id: None,
            original_copy: false,
            home: false,
            variable_bit_rate: true,
            bitrate: 128_000,
            program_configs: vec![pce],
            bits_read: 0,
        };
        assert_eq!(
            AacLcDecoder::from_adif_header(&adif).unwrap_err(),
            DecodeError::UnsupportedAudioObjectType(1)
        );
    }

    #[test]
    fn decodes_er_aac_lc_mono_without_resilience_tools() {
        let lc_payload = zero_sce_payload(false);
        let mut source = BitReader::new(&lc_payload);
        source.read_u8(3).unwrap(); // ER raw_data_block omits ID_SCE
        let mut writer = BitWriter::new();
        for _ in 3..38 {
            writer.write_bool(source.read_bool().unwrap());
        }
        let payload = writer.finish();
        let asc = AudioSpecificConfig {
            audio_object_type: 17,
            sampling_frequency_index: 4,
            sampling_frequency: 44_100,
            channel_configuration: 1,
            extension: None,
            ga_specific: Some(crate::asc::GaSpecificConfig::default()),
            eld_specific: None,
            usac_config: None,
            error_protection_config: Some(0),
            program_config: None,
            bits_read: 0,
        };
        let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
        let decoded = decoder.decode_raw_data_block_f32(&payload).unwrap();
        assert_eq!(decoder.audio_object_type(), 17);
        assert!(matches!(decoded, DecodedAacLcFrame::Mono(_)));
        assert!(decoded
            .interleaved_f32()
            .iter()
            .all(|sample| *sample == 0.0));

        let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
        decoder.channel_filterbanks[0] = LongBlockFilterbank::new(960).unwrap();
        assert!(decoder.decode_raw_data_block_f32(&payload).is_err());
        let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
        decoder.fixed_channel_filterbanks[0] = FixedLongBlockFilterbank::new(960).unwrap();
        assert!(decoder
            .decode_raw_data_block_fixed_interleaved_i16(&payload)
            .is_err());
    }

    #[test]
    fn decodes_er_aac_scalable_layer_zero_as_er_aac_lc() {
        let lc_payload = zero_sce_payload(false);
        let mut source = BitReader::new(&lc_payload);
        source.read_u8(3).unwrap();
        let mut writer = BitWriter::new();
        for _ in 3..38 {
            writer.write_bool(source.read_bool().unwrap());
        }
        let payload = writer.finish();
        let asc = AudioSpecificConfig {
            audio_object_type: 20,
            sampling_frequency_index: 4,
            sampling_frequency: 44_100,
            channel_configuration: 1,
            extension: None,
            ga_specific: Some(crate::asc::GaSpecificConfig {
                layer: Some(0),
                ..crate::asc::GaSpecificConfig::default()
            }),
            eld_specific: None,
            usac_config: None,
            error_protection_config: Some(0),
            program_config: None,
            bits_read: 0,
        };
        let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
        assert_eq!(decoder.audio_object_type(), 20);
        assert_eq!(decoder.frame_length(), 1024);
        assert!(decoder
            .decode_raw_data_block_f32(&payload)
            .unwrap()
            .interleaved_f32()
            .iter()
            .all(|&sample| sample == 0.0));

        let mut enhanced = asc;
        enhanced.ga_specific.as_mut().unwrap().layer = Some(1);
        assert_eq!(
            AacLcDecoder::from_audio_specific_config(&enhanced).unwrap_err(),
            DecodeError::UnsupportedAudioObjectType(20)
        );
    }

    #[test]
    fn constructs_er_drm_aac_960_filterbanks() {
        let decoder = AacLcDecoder::new_ga_with_frame_length(17, 3, 2, 960).unwrap();
        assert_eq!(decoder.frame_length(), 960);
        assert_eq!(decoder.audio_object_type(), 17);
    }

    #[test]
    fn decodes_drm_aac_mono_empty_hcr_frame_and_reports_crc_region() {
        let mut decoder = AacLcDecoder::new_drm_aac(3, 1).unwrap();
        let mut bits = BitWriter::new();
        bits.write_bool(false); // ICS reserved
        bits.write(0, 2); // ONLY_LONG
        bits.write_bool(false); // sine window
        bits.write(0, 6); // max_sfb
        bits.write_bool(false); // predictor absent
        bits.write_bool(false); // TNS absent
        bits.write_bool(false); // LTP absent
        bits.write(0, 8); // global gain
        bits.write(0, 14); // reordered spectral length
        bits.write(0, 6); // longest codeword
        let (samples, protected_bits, core_bits) =
            decoder.decode_drm_aac_mono_f32(&bits.finish()).unwrap();
        assert_eq!(protected_bits, 41);
        assert_eq!(core_bits, 41);
        assert_eq!(samples, vec![0.0; 960]);
    }

    #[test]
    fn decodes_drm_aac_stereo_empty_hcr_frame_and_reports_crc_region() {
        let mut decoder = AacLcDecoder::new_drm_aac(3, 2).unwrap();
        let mut bits = BitWriter::new();
        bits.write_bool(false); // ICS reserved
        bits.write(0, 2); // ONLY_LONG
        bits.write_bool(false); // sine window
        bits.write(0, 6); // max_sfb
        bits.write_bool(false); // predictor absent
        bits.write(0, 2); // ms_mask_present = none
        for _ in 0..2 {
            bits.write_bool(false); // TNS absent
            bits.write_bool(false); // LTP absent
            bits.write(0, 8); // global gain
            bits.write(0, 14); // reordered spectral length
            bits.write(0, 6); // longest codeword
        }
        let (channels, protected_bits, core_bits) =
            decoder.decode_drm_aac_stereo_f32(&bits.finish()).unwrap();
        assert_eq!(protected_bits, 73);
        assert_eq!(core_bits, 73);
        assert_eq!(channels[0], vec![0.0; 960]);
        assert_eq!(channels[1], vec![0.0; 960]);
    }

    #[test]
    fn drm_decode_facades_propagate_filterbank_mismatches() {
        let mut mono = BitWriter::new();
        write_shared_long_ics(&mut mono, 0);
        mono.write_bool(false); // TNS absent
        mono.write_bool(false); // LTP absent
        mono.write(0, 8); // global gain
        mono.write(0, 14); // reordered spectral length
        mono.write(0, 6); // longest codeword
        let mono = mono.finish();

        let mut stereo = BitWriter::new();
        write_shared_long_ics(&mut stereo, 0);
        stereo.write(0, 2); // no MS stereo
        for _ in 0..2 {
            stereo.write_bool(false); // TNS absent
            stereo.write_bool(false); // LTP absent
            stereo.write(0, 8); // global gain
            stereo.write(0, 14); // reordered spectral length
            stereo.write(0, 6); // longest codeword
        }
        let stereo = stereo.finish();

        let mut decoder = AacLcDecoder::new_drm_aac(3, 1).unwrap();
        decoder.channel_filterbanks[0] = LongBlockFilterbank::new(1024).unwrap();
        assert!(decoder.decode_drm_aac_mono_f32(&mono).is_err());

        let mut decoder = AacLcDecoder::new_drm_aac(3, 1).unwrap();
        decoder.fixed_channel_filterbanks[0] = FixedLongBlockFilterbank::new(1024).unwrap();
        assert!(decoder.decode_drm_aac_mono_i16(&mono).is_err());

        for mismatched_channel in 0..2 {
            let mut decoder = AacLcDecoder::new_drm_aac(3, 2).unwrap();
            decoder.channel_filterbanks[mismatched_channel] =
                LongBlockFilterbank::new(1024).unwrap();
            assert!(decoder.decode_drm_aac_stereo_f32(&stereo).is_err());

            let mut decoder = AacLcDecoder::new_drm_aac(3, 2).unwrap();
            decoder.fixed_channel_filterbanks[mismatched_channel] =
                FixedLongBlockFilterbank::new(1024).unwrap();
            assert!(decoder.decode_drm_aac_stereo_i16(&stereo).is_err());
        }
    }

    #[test]
    fn decodes_drm_aac_empty_tns_filters_in_mono_and_stereo() {
        let mut mono = BitWriter::new();
        write_shared_long_ics(&mut mono, 0);
        mono.write_bool(true); // TNS present
        mono.write_bool(false); // LTP absent
        mono.write(0, 8); // global gain
        mono.write(0, 14); // reordered spectral length
        mono.write(0, 6); // longest codeword
        mono.write(0, 2); // zero TNS filters
        let (samples, protected_bits, core_bits) = AacLcDecoder::new_drm_aac(3, 1)
            .unwrap()
            .decode_drm_aac_mono_f32(&mono.finish())
            .unwrap();
        assert_eq!(protected_bits, 43);
        assert_eq!(core_bits, 43);
        assert_eq!(samples, vec![0.0; 960]);

        let mut stereo = BitWriter::new();
        write_shared_long_ics(&mut stereo, 0);
        stereo.write(0, 2); // no MS stereo
        for _ in 0..2 {
            stereo.write_bool(true); // TNS present
            stereo.write_bool(false); // LTP absent
            stereo.write(0, 8); // global gain
            stereo.write(0, 14); // reordered spectral length
            stereo.write(0, 6); // longest codeword
        }
        stereo.write(0, 2); // zero left TNS filters
        stereo.write(0, 2); // zero right TNS filters
        let (channels, protected_bits, core_bits) = AacLcDecoder::new_drm_aac(3, 2)
            .unwrap()
            .decode_drm_aac_stereo_f32(&stereo.finish())
            .unwrap();
        assert_eq!(protected_bits, 77);
        assert_eq!(core_bits, 77);
        assert_eq!(channels, [vec![0.0; 960], vec![0.0; 960]]);
    }

    #[test]
    fn drm_cores_reject_mono_stereo_configuration_mismatches() {
        assert_eq!(
            AacLcDecoder::new_drm_aac(3, 2)
                .unwrap()
                .decode_drm_aac_mono_f32(&[]),
            Err(DecodeError::UnsupportedChannelConfiguration(2))
        );
        assert_eq!(
            AacLcDecoder::new_drm_aac(3, 1)
                .unwrap()
                .decode_drm_aac_stereo_f32(&[]),
            Err(DecodeError::UnsupportedChannelConfiguration(1))
        );
        assert_eq!(
            AacLcDecoder::new_drm_aac(3, 2)
                .unwrap()
                .decode_drm_aac_mono_i16(&[]),
            Err(DecodeError::UnsupportedChannelConfiguration(2))
        );
        assert_eq!(
            AacLcDecoder::new_drm_aac(3, 1)
                .unwrap()
                .decode_drm_aac_stereo_i16(&[]),
            Err(DecodeError::UnsupportedChannelConfiguration(1))
        );

        assert!(AacLcDecoder::new_drm_aac(3, 1)
            .unwrap()
            .decode_drm_aac_mono_f32(&[])
            .is_err());
        assert!(AacLcDecoder::new_drm_aac(3, 2)
            .unwrap()
            .decode_drm_aac_stereo_f32(&[])
            .is_err());
        assert!(AacLcDecoder::new_drm_aac(3, 1)
            .unwrap()
            .decode_drm_aac_mono_i16(&[])
            .is_err());
        assert!(AacLcDecoder::new_drm_aac(3, 2)
            .unwrap()
            .decode_drm_aac_stereo_i16(&[])
            .is_err());

        let mut decoder = AacLcDecoder::new_drm_aac(3, 1).unwrap();
        decoder.channel_configuration = 8;
        assert_eq!(
            decoder.decode_raw_data_block_f32(&[]),
            Err(DecodeError::UnsupportedChannelConfiguration(8))
        );
        assert_eq!(
            decoder.decode_raw_data_block_fixed_interleaved_i16(&[]),
            Err(DecodeError::UnsupportedChannelConfiguration(8))
        );
    }

    #[test]
    fn decodes_er_aac_lc_stereo_mapping() {
        let lc_payload = zero_cpe_payload(0);
        let mut source = BitReader::new(&lc_payload);
        source.read_u8(3).unwrap();
        let mut writer = BitWriter::new();
        while source.remaining_bits() != 0 {
            writer.write_bool(source.read_bool().unwrap());
        }
        let payload = writer.finish();
        let asc = AudioSpecificConfig {
            audio_object_type: 17,
            sampling_frequency_index: 4,
            sampling_frequency: 44_100,
            channel_configuration: 2,
            extension: None,
            ga_specific: Some(crate::asc::GaSpecificConfig::default()),
            eld_specific: None,
            usac_config: None,
            error_protection_config: Some(1),
            program_config: None,
            bits_read: 0,
        };
        let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
        let decoded = decoder.decode_raw_data_block_f32(&payload).unwrap();
        assert!(matches!(decoded, DecodedAacLcFrame::Stereo(_)));
        assert_eq!(decoded.interleaved_f32().len(), 2048);
        assert_eq!(decoder.conceal_f32_interleaved().unwrap().len(), 2048);
        decoder.decode_raw_data_block_f32(&payload).unwrap();
        assert_eq!(decoder.f32_concealment_state(), ConcealmentState::FadeIn);
        let mut fixed_decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
        let fixed = fixed_decoder
            .decode_raw_data_block_fixed_interleaved_i16(&payload)
            .unwrap();
        assert_eq!(fixed.len(), 2048);
        assert!(fixed.iter().all(|sample| *sample == 0));

        for mismatched_channel in 0..2 {
            let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
            decoder.channel_filterbanks[mismatched_channel] =
                LongBlockFilterbank::new(960).unwrap();
            assert!(decoder.decode_raw_data_block_f32(&payload).is_err());

            let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
            decoder.fixed_channel_filterbanks[mismatched_channel] =
                FixedLongBlockFilterbank::new(960).unwrap();
            assert!(decoder
                .decode_raw_data_block_fixed_interleaved_i16(&payload)
                .is_err());
        }

        let mut ld_asc = asc.clone();
        ld_asc.audio_object_type = 23;
        let pcm = AacLcDecoder::from_audio_specific_config(&ld_asc)
            .unwrap()
            .decode_raw_data_block_f32(&payload)
            .unwrap()
            .interleaved_f32();
        assert_eq!(pcm.len(), 1024);
        assert!(pcm.iter().all(|&sample| sample == 0.0));
    }

    #[test]
    fn decodes_er_aac_lc_rvlc_zero_band_payload() {
        let mut writer = BitWriter::new();
        writer.write(0, 4); // element instance tag (no ID_SCE)
        writer.write(100, 8); // global gain
        writer.write_bool(false);
        writer.write(0, 2); // long window
        writer.write_bool(false);
        writer.write(1, 6); // maxSfb
        writer.write_bool(false); // prediction absent
        writer.write(ZERO_HCB as u32, 4);
        writer.write(1, 5); // one zero-codebook section
        writer.write_bool(false); // sf_concealment
        writer.write(100, 8); // reverse global gain
        writer.write(0, 9); // no RVLC scalefactor codewords
        writer.write_bool(false); // no escapes
        writer.write_bool(false); // pulse absent
        writer.write_bool(false); // TNS absent
        writer.write_bool(false); // gain control absent
        let payload = writer.finish();
        let asc = AudioSpecificConfig {
            audio_object_type: 17,
            sampling_frequency_index: 4,
            sampling_frequency: 44_100,
            channel_configuration: 1,
            extension: None,
            ga_specific: Some(crate::asc::GaSpecificConfig {
                extension_flag: true,
                scalefactor_data_resilience: true,
                extension_flag3: Some(false),
                ..crate::asc::GaSpecificConfig::default()
            }),
            eld_specific: None,
            usac_config: None,
            error_protection_config: Some(1),
            program_config: None,
            bits_read: 0,
        };
        let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
        let pcm = decoder
            .decode_raw_data_block_f32(&payload)
            .unwrap()
            .interleaved_f32();
        assert_eq!(pcm.len(), 1024);
        assert!(pcm.iter().all(|sample| *sample == 0.0));
        let mut fixed = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
        assert!(fixed
            .decode_raw_data_block_fixed_interleaved_i16(&payload)
            .unwrap()
            .iter()
            .all(|sample| *sample == 0));
    }

    #[test]
    fn rvlc_forward_failure_falls_back_to_concealed_scalefactors() {
        let ics = test_ics(1);
        let sections = test_sections(vec![12]); // invalid for forward RVLC decoding
        let side = RvlcSideInfo {
            scalefactor_concealment: true,
            reverse_global_gain: 100,
            scalefactor_bits: 0,
            noise_energy: None,
            escapes_present: false,
            escape_bits: 0,
            noise_last_position: None,
            bits_read: 0,
        };
        assert_eq!(
            decode_rvlc_or_conceal(&mut BitReader::new(&[]), &side, &ics, &sections, 100,).unwrap(),
            ScalefactorData {
                values: vec![vec![0]],
            }
        );

        let mut truncated = side;
        truncated.scalefactor_bits = 1;
        assert!(matches!(
            decode_rvlc_or_conceal(
                &mut BitReader::new(&[]),
                &truncated,
                &ics,
                &test_sections(vec![ZERO_HCB]),
                100,
            ),
            Err(DecodeError::Rvlc(RvlcError::Bit(
                BitError::UnexpectedEof { .. }
            )))
        ));

        let prefix = EldEp1ChannelPrefix {
            global_gain: 100,
            ics: test_ics(0),
            section_data: test_sections(Vec::new()),
            scalefactors: Some(ScalefactorData { values: Vec::new() }),
            rvlc_side: None,
            tns_present: false,
        };
        assert_eq!(
            read_eld_ep1_tns(&mut BitReader::new(&[]), &prefix).unwrap(),
            TnsData::absent(1)
        );

        let rvlc_prefix = EldEp1ChannelPrefix {
            global_gain: 100,
            ics: test_ics(0),
            section_data: test_sections(Vec::new()),
            scalefactors: None,
            rvlc_side: Some(truncated),
            tns_present: false,
        };
        assert!(finish_eld_ep1_channel_f32(
            &mut BitReader::new(&[]),
            rvlc_prefix,
            TnsData::absent(1),
            4,
            512,
            false,
        )
        .is_err());
        let rvlc_prefix = EldEp1ChannelPrefix {
            global_gain: 100,
            ics: test_ics(0),
            section_data: test_sections(Vec::new()),
            scalefactors: None,
            rvlc_side: Some(truncated),
            tns_present: false,
        };
        assert!(finish_eld_ep1_channel_fixed(
            &mut BitReader::new(&[]),
            rvlc_prefix,
            TnsData::absent(1),
            4,
            512,
            false,
        )
        .is_err());

        let invalid_sections = EldEp1ChannelPrefix {
            global_gain: 100,
            ics: test_ics(1),
            section_data: test_sections(Vec::new()),
            scalefactors: Some(ScalefactorData {
                values: vec![vec![0]],
            }),
            rvlc_side: None,
            tns_present: false,
        };
        assert!(finish_eld_ep1_channel_f32(
            &mut BitReader::new(&[0, 0, 0]),
            invalid_sections,
            TnsData::absent(1),
            4,
            512,
            true,
        )
        .is_err());
        let invalid_sections = EldEp1ChannelPrefix {
            global_gain: 100,
            ics: test_ics(1),
            section_data: test_sections(Vec::new()),
            scalefactors: Some(ScalefactorData {
                values: vec![vec![0]],
            }),
            rvlc_side: None,
            tns_present: false,
        };
        assert!(finish_eld_ep1_channel_fixed(
            &mut BitReader::new(&[0, 0, 0]),
            invalid_sections,
            TnsData::absent(1),
            4,
            512,
            true,
        )
        .is_err());

        let invalid_sections = EldEp1ChannelPrefix {
            global_gain: 100,
            ics: test_ics(1),
            section_data: test_sections(Vec::new()),
            scalefactors: Some(ScalefactorData {
                values: vec![vec![0]],
            }),
            rvlc_side: None,
            tns_present: false,
        };
        assert!(finish_eld_ep1_channel_f32(
            &mut BitReader::new(&[]),
            invalid_sections,
            TnsData::absent(1),
            4,
            512,
            false,
        )
        .is_err());
        let invalid_sections = EldEp1ChannelPrefix {
            global_gain: 100,
            ics: test_ics(1),
            section_data: test_sections(Vec::new()),
            scalefactors: Some(ScalefactorData {
                values: vec![vec![0]],
            }),
            rvlc_side: None,
            tns_present: false,
        };
        assert!(finish_eld_ep1_channel_fixed(
            &mut BitReader::new(&[]),
            invalid_sections,
            TnsData::absent(1),
            4,
            512,
            false,
        )
        .is_err());
    }

    #[test]
    fn decodes_er_aac_lc_hcr_zero_band_payload() {
        let mut writer = BitWriter::new();
        writer.write(0, 4); // element instance tag (no ID_SCE)
        writer.write(100, 8); // global gain
        writer.write_bool(false);
        writer.write(0, 2); // long window
        writer.write_bool(false);
        writer.write(1, 6); // maxSfb
        writer.write_bool(false); // prediction absent
        writer.write(ZERO_HCB as u32, 4);
        writer.write(1, 5); // one zero-codebook section
        writer.write_bool(false); // pulse absent
        writer.write_bool(false); // TNS absent
        writer.write_bool(false); // gain control absent
        writer.write(0, 14); // no reordered spectral payload
        writer.write(0, 6); // no longest codeword
        let payload = writer.finish();
        let mut pns_random = PnsRandomState::new(1);
        let lfe = decode_er_single_channel_spectra_from_reader(
            &mut BitReader::new(&payload),
            ElementId::Lfe,
            4,
            1024,
            false,
            false,
            false,
            true,
            &mut pns_random,
        )
        .unwrap();
        assert_eq!(lfe.side_info.id, ElementId::Lfe);
        let fixed_lfe = decode_er_single_channel_spectra_fixed_from_reader(
            &mut BitReader::new(&payload),
            ElementId::Lfe,
            4,
            1024,
            false,
            false,
            false,
            true,
            &mut pns_random,
        )
        .unwrap();
        assert_eq!(fixed_lfe.side_info.id, ElementId::Lfe);
        let asc = AudioSpecificConfig {
            audio_object_type: 17,
            sampling_frequency_index: 4,
            sampling_frequency: 44_100,
            channel_configuration: 1,
            extension: None,
            ga_specific: Some(crate::asc::GaSpecificConfig {
                extension_flag: true,
                spectral_data_resilience: true,
                extension_flag3: Some(false),
                ..crate::asc::GaSpecificConfig::default()
            }),
            eld_specific: None,
            usac_config: None,
            error_protection_config: Some(1),
            program_config: None,
            bits_read: 0,
        };
        let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
        assert!(decoder
            .decode_raw_data_block_f32(&payload)
            .unwrap()
            .interleaved_f32()
            .iter()
            .all(|sample| *sample == 0.0));
        let mut fixed = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
        assert!(fixed
            .decode_raw_data_block_fixed_interleaved_i16(&payload)
            .unwrap()
            .iter()
            .all(|sample| *sample == 0));
    }

    #[test]
    fn decodes_er_aac_lc_vcb11_virtual_codebook() {
        let mut writer = BitWriter::new();
        writer.write(0, 4); // element instance tag
        writer.write(100, 8); // global gain
        writer.write_bool(false);
        writer.write(0, 2); // long window
        writer.write_bool(false);
        writer.write(1, 6); // maxSfb
        writer.write_bool(false);
        writer.write(16, 5); // virtual codebook, implicit one-band section
        writer.write_bool(false); // zero scalefactor delta
        writer.write_bool(false); // pulse absent
        writer.write_bool(false); // TNS absent
        writer.write_bool(false); // gain control absent
        writer.write(0, 16); // zero tuple plus byte padding
        let payload = writer.finish();
        let asc = AudioSpecificConfig {
            audio_object_type: 17,
            sampling_frequency_index: 4,
            sampling_frequency: 44_100,
            channel_configuration: 1,
            extension: None,
            ga_specific: Some(crate::asc::GaSpecificConfig {
                extension_flag: true,
                section_data_resilience: true,
                extension_flag3: Some(false),
                ..crate::asc::GaSpecificConfig::default()
            }),
            eld_specific: None,
            usac_config: None,
            error_protection_config: Some(1),
            program_config: None,
            bits_read: 0,
        };
        let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
        assert!(decoder
            .decode_raw_data_block_f32(&payload)
            .unwrap()
            .interleaved_f32()
            .iter()
            .all(|&sample| sample == 0.0));
        let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
        assert!(decoder
            .decode_raw_data_block_fixed_interleaved_i16(&payload)
            .unwrap()
            .iter()
            .all(|&sample| sample == 0));
    }

    #[test]
    fn decodes_er_aac_ld_512_and_480_zero_frames() {
        let mut writer = BitWriter::new();
        writer.write(0, 4); // element instance tag
        writer.write(100, 8); // global gain
        writer.write_bool(false); // ICS reserved
        writer.write(0, 2); // only-long window
        writer.write_bool(false); // sine shape
        writer.write(1, 6); // maxSfb
        writer.write_bool(false); // prediction absent
        writer.write(ZERO_HCB as u32, 4);
        writer.write(1, 5);
        writer.write_bool(false); // pulse absent
        writer.write_bool(false); // TNS absent
        writer.write_bool(false); // gain control absent
        let payload = writer.finish();
        let stereo_payload = {
            let encoded = zero_cpe_payload(0);
            let mut reader = BitReader::new(&encoded);
            assert_eq!(reader.read_u8(3).unwrap(), ElementId::ChannelPair.bits());
            let mut writer = BitWriter::new();
            while reader.remaining_bits() != 0 {
                writer.write_bool(reader.read_bool().unwrap());
            }
            writer.finish()
        };

        for (frame_length_flag, expected) in [(false, 512), (true, 480)] {
            let asc = AudioSpecificConfig {
                audio_object_type: 23,
                sampling_frequency_index: 4,
                sampling_frequency: 44_100,
                channel_configuration: 1,
                extension: None,
                ga_specific: Some(crate::asc::GaSpecificConfig {
                    frame_length_flag,
                    ..crate::asc::GaSpecificConfig::default()
                }),
                eld_specific: None,
                usac_config: None,
                error_protection_config: Some(0),
                program_config: None,
                bits_read: 0,
            };
            let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
            assert_eq!(decoder.frame_length(), expected);
            let pcm = decoder
                .decode_raw_data_block_f32(&payload)
                .unwrap()
                .interleaved_f32();
            assert_eq!(pcm.len(), expected);
            assert!(pcm.iter().all(|&sample| sample == 0.0));

            let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
            let pcm = decoder
                .decode_raw_data_block_fixed_interleaved_i16(&payload)
                .unwrap();
            assert_eq!(pcm.len(), expected);
            assert!(pcm.iter().all(|&sample| sample == 0));

            let mut stereo_asc = asc.clone();
            stereo_asc.channel_configuration = 2;
            let mut decoder = AacLcDecoder::from_audio_specific_config(&stereo_asc).unwrap();
            let pcm = decoder
                .decode_raw_data_block_f32(&stereo_payload)
                .unwrap()
                .interleaved_f32();
            assert_eq!(pcm.len(), expected * 2);
            assert!(pcm.iter().all(|&sample| sample == 0.0));

            let mut decoder = AacLcDecoder::from_audio_specific_config(&stereo_asc).unwrap();
            let pcm = decoder
                .decode_raw_data_block_fixed_interleaved_i16(&stereo_payload)
                .unwrap();
            assert_eq!(pcm.len(), expected * 2);
            assert!(pcm.iter().all(|&sample| sample == 0));
        }
    }

    #[test]
    fn maps_er_long_window_scalefactor_band_counts() {
        assert_eq!(er_long_sfb_count(0, 512).unwrap(), 36);
        assert_eq!(er_long_sfb_count(5, 512).unwrap(), 37);
        assert_eq!(er_long_sfb_count(12, 512).unwrap(), 31);
        assert_eq!(er_long_sfb_count(0, 480).unwrap(), 35);
        assert_eq!(er_long_sfb_count(5, 480).unwrap(), 37);
        assert_eq!(er_long_sfb_count(12, 480).unwrap(), 30);
        assert!(matches!(
            er_long_sfb_count(4, 1024),
            Err(DecodeError::Sfb(SfbError::UnsupportedFrameLength(1024)))
        ));
    }

    #[test]
    fn rejects_er_short_frames_gain_control_and_drm_ltp() {
        assert!(decode_er_channel_stream_from_reader(
            &mut BitReader::new(&[100]),
            4,
            512,
            true,
            None,
            false,
            false,
            false,
            HcrElementType::SingleChannel,
        )
        .is_err());
        assert!(decode_er_channel_stream_fixed_from_reader(
            &mut BitReader::new(&[100]),
            4,
            512,
            true,
            None,
            false,
            false,
            false,
            HcrElementType::SingleChannel,
        )
        .is_err());
        assert!(decode_er_single_channel_spectra_from_reader(
            &mut BitReader::new(&[0]),
            ElementId::SingleChannel,
            4,
            1024,
            false,
            false,
            false,
            false,
            &mut PnsRandomState::new(1),
        )
        .is_err());

        let mut short = test_ics(0);
        short.window_sequence = WindowSequence::EightShort;
        short.window_group_lengths = vec![8];
        assert_eq!(
            decode_er_channel_stream_from_reader(
                &mut BitReader::new(&[100]),
                4,
                512,
                false,
                Some(&short),
                false,
                false,
                false,
                HcrElementType::SingleChannel,
            ),
            Err(DecodeError::UnsupportedFrameLength(512))
        );
        assert_eq!(
            decode_er_channel_stream_fixed_from_reader(
                &mut BitReader::new(&[100]),
                4,
                512,
                false,
                Some(&short),
                false,
                false,
                false,
                HcrElementType::SingleChannel,
            ),
            Err(DecodeError::UnsupportedFrameLength(512))
        );

        let long = test_ics(0);
        let mut writer = BitWriter::new();
        writer.write(100, 8);
        writer.write_bool(false); // pulse absent
        writer.write_bool(false); // TNS absent
        writer.write_bool(true); // unsupported gain control
        let payload = writer.finish();
        assert_eq!(
            decode_er_channel_stream_from_reader(
                &mut BitReader::new(&payload),
                4,
                1024,
                false,
                Some(&long),
                false,
                false,
                false,
                HcrElementType::SingleChannel,
            ),
            Err(DecodeError::GainControlUnsupported)
        );
        assert_eq!(
            decode_er_channel_stream_fixed_from_reader(
                &mut BitReader::new(&payload),
                4,
                1024,
                false,
                Some(&long),
                false,
                false,
                false,
                HcrElementType::SingleChannel,
            ),
            Err(DecodeError::GainControlUnsupported)
        );

        let mut mono = BitWriter::new();
        write_shared_long_ics(&mut mono, 0);
        mono.write_bool(false); // TNS absent
        mono.write_bool(true); // unsupported LTP
        assert_eq!(
            decode_drm_aac_single_channel_spectra_from_reader(
                &mut BitReader::new(&mono.finish()),
                4,
                &mut PnsRandomState::new(1),
            ),
            Err(DecodeError::LtpUnsupported)
        );

        let mut pair = BitWriter::new();
        write_shared_long_ics(&mut pair, 0);
        pair.write(0, 2); // no MS stereo
        pair.write_bool(false); // left TNS absent
        pair.write_bool(true); // unsupported left LTP
        assert_eq!(
            decode_drm_aac_channel_pair_spectra_from_reader(
                &mut BitReader::new(&pair.finish()),
                4,
                &mut PnsRandomState::new(1),
            ),
            Err(DecodeError::LtpUnsupported)
        );
    }

    #[test]
    fn decodes_empty_tns_filters_in_er_and_eld_channel_streams() {
        let ics = test_ics(1);
        let mut er = BitWriter::new();
        er.write(100, 8);
        er.write(ZERO_HCB as u32, 4);
        er.write(1, 5);
        er.write_bool(false); // pulse absent
        er.write_bool(true); // TNS present
        er.write_bool(false); // gain control absent
        er.write(0, 2); // zero long-window TNS filters
        let er = er.finish();

        let decoded = decode_er_channel_stream_from_reader(
            &mut BitReader::new(&er),
            4,
            1024,
            false,
            Some(&ics),
            false,
            false,
            false,
            HcrElementType::SingleChannel,
        )
        .unwrap();
        assert!(decoded.tns_data.present);
        assert_eq!(decoded.tns_data.filters, vec![Vec::new()]);
        let decoded = decode_er_channel_stream_fixed_from_reader(
            &mut BitReader::new(&er),
            4,
            1024,
            false,
            Some(&ics),
            false,
            false,
            false,
            HcrElementType::SingleChannel,
        )
        .unwrap();
        assert!(decoded.tns_data.present);
        assert_eq!(decoded.tns_data.filters, vec![Vec::new()]);

        let mut eld = BitWriter::new();
        eld.write(100, 8);
        eld.write(ZERO_HCB as u32, 4);
        eld.write(1, 5);
        eld.write_bool(true); // TNS present
        eld.write(0, 2); // zero long-window TNS filters
        let eld = eld.finish();

        let decoded = decode_er_channel_stream_from_reader(
            &mut BitReader::new(&eld),
            4,
            512,
            true,
            Some(&ics),
            false,
            false,
            false,
            HcrElementType::SingleChannel,
        )
        .unwrap();
        assert!(decoded.tns_data.present);
        assert_eq!(decoded.tns_data.filters, vec![Vec::new()]);
        let decoded = decode_er_channel_stream_fixed_from_reader(
            &mut BitReader::new(&eld),
            4,
            512,
            true,
            Some(&ics),
            false,
            false,
            false,
            HcrElementType::SingleChannel,
        )
        .unwrap();
        assert!(decoded.tns_data.present);
        assert_eq!(decoded.tns_data.filters, vec![Vec::new()]);
    }

    #[test]
    fn decodes_er_aac_eld_implicit_ics_zero_frames() {
        let mut writer = BitWriter::new();
        writer.write(100, 8); // global gain, no element tag
        writer.write(1, 6); // implicit long-window ICS maxSfb
        writer.write(ZERO_HCB as u32, 4);
        writer.write(1, 5);
        writer.write_bool(false); // TNS absent
        let payload = writer.finish();

        for (frame_length_flag, expected) in [(false, 512), (true, 480)] {
            let asc = AudioSpecificConfig {
                audio_object_type: 39,
                sampling_frequency_index: 4,
                sampling_frequency: 44_100,
                channel_configuration: 1,
                extension: None,
                ga_specific: None,
                eld_specific: Some(crate::asc::EldSpecificConfig {
                    frame_length_flag,
                    ..crate::asc::EldSpecificConfig::default()
                }),
                usac_config: None,
                error_protection_config: Some(0),
                program_config: None,
                bits_read: 0,
            };
            let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
            let pcm = decoder
                .decode_raw_data_block_f32(&payload)
                .unwrap()
                .interleaved_f32();
            assert_eq!(pcm.len(), expected);
            assert!(pcm.iter().all(|&sample| sample == 0.0));
            let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
            let pcm = decoder
                .decode_raw_data_block_fixed_interleaved_i16(&payload)
                .unwrap();
            assert_eq!(pcm.len(), expected);
            assert!(pcm.iter().all(|&sample| sample == 0));

            let wrong_length = if expected == 512 { 480 } else { 512 };
            let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
            decoder.eld_channel_filterbanks[0] = LowDelayFilterbankF32::new(wrong_length).unwrap();
            assert!(decoder.decode_raw_data_block_f32(&payload).is_err());
            let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
            decoder.eld_fixed_channel_filterbanks[0] =
                LowDelayFilterbankQ31::new(wrong_length).unwrap();
            assert!(decoder
                .decode_raw_data_block_fixed_interleaved_i16(&payload)
                .is_err());
        }
    }

    #[test]
    fn decodes_er_aac_eld_channel_pair_with_shared_implicit_ics() {
        let mut writer = BitWriter::new();
        writer.write(1, 6); // shared ELD maxSfb
        writer.write(0, 2); // no MS stereo
        for _ in 0..2 {
            writer.write(100, 8); // global gain
            writer.write(ZERO_HCB as u32, 4);
            writer.write(1, 5);
            writer.write_bool(false); // TNS absent
        }
        let payload = writer.finish();

        let mut pns_random = PnsRandomState::new(1);
        let decoded = decode_er_channel_pair_spectra_from_reader(
            &mut BitReader::new(&payload),
            4,
            512,
            true,
            false,
            false,
            false,
            false,
            &mut pns_random,
        )
        .unwrap();
        assert!(decoded.prefix.common_window);
        assert_eq!(decoded.prefix.shared_ics.as_ref().unwrap().max_sfb, 1);
        assert!(decoded
            .left
            .spectrum
            .windows
            .iter()
            .chain(&decoded.right.spectrum.windows)
            .flatten()
            .all(|sample| *sample == 0.0));

        let decoded = decode_er_channel_pair_spectra_fixed_from_reader(
            &mut BitReader::new(&payload),
            4,
            512,
            true,
            false,
            false,
            false,
            false,
            &mut pns_random,
        )
        .unwrap();
        assert!(decoded.prefix.common_window);
        assert_eq!(decoded.prefix.shared_ics.as_ref().unwrap().max_sfb, 1);
        assert!(decoded
            .left
            .spectrum
            .windows
            .iter()
            .chain(&decoded.right.spectrum.windows)
            .flatten()
            .all(|sample| *sample == 0));

        let asc = AudioSpecificConfig {
            audio_object_type: 39,
            sampling_frequency_index: 4,
            sampling_frequency: 44_100,
            channel_configuration: 2,
            extension: None,
            ga_specific: None,
            eld_specific: Some(crate::asc::EldSpecificConfig::default()),
            usac_config: None,
            error_protection_config: Some(0),
            program_config: None,
            bits_read: 0,
        };
        for mismatched_channel in 0..2 {
            let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
            decoder.eld_channel_filterbanks[mismatched_channel] =
                LowDelayFilterbankF32::new(480).unwrap();
            assert!(decoder.decode_raw_data_block_f32(&payload).is_err());

            let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
            decoder.eld_fixed_channel_filterbanks[mismatched_channel] =
                LowDelayFilterbankQ31::new(480).unwrap();
            assert!(decoder
                .decode_raw_data_block_fixed_interleaved_i16(&payload)
                .is_err());
        }

        let decoded = decode_er_channel_pair_spectra_from_reader(
            &mut BitReader::new(&payload),
            4,
            512,
            true,
            true,
            false,
            false,
            false,
            &mut pns_random,
        )
        .unwrap();
        assert!(decoded.prefix.common_window);
        assert!(decoded.ms_stereo.is_some());
        assert!(decoded
            .left
            .spectrum
            .windows
            .iter()
            .chain(&decoded.right.spectrum.windows)
            .flatten()
            .all(|sample| *sample == 0.0));

        let decoded = decode_er_channel_pair_spectra_fixed_from_reader(
            &mut BitReader::new(&payload),
            4,
            512,
            true,
            true,
            false,
            false,
            false,
            &mut pns_random,
        )
        .unwrap();
        assert!(decoded.prefix.common_window);
        assert!(decoded.ms_stereo.is_some());
        assert!(decoded
            .left
            .spectrum
            .windows
            .iter()
            .chain(&decoded.right.spectrum.windows)
            .flatten()
            .all(|sample| *sample == 0));
        let mut tns = BitWriter::new();
        tns.write(1, 6); // shared ELD maxSfb
        tns.write(0, 2); // no MS stereo
        for _ in 0..2 {
            tns.write(100, 8);
            tns.write(ZERO_HCB as u32, 4);
            tns.write(1, 5);
            tns.write_bool(true); // TNS present in each prefix
        }
        tns.write(0, 2); // left has zero TNS filters
        tns.write(0, 2); // right has zero TNS filters
        let tns = tns.finish();

        let decoded = decode_er_channel_pair_spectra_from_reader(
            &mut BitReader::new(&tns),
            4,
            512,
            true,
            true,
            false,
            false,
            false,
            &mut pns_random,
        )
        .unwrap();
        assert!(decoded.left.tns_data.present);
        assert!(decoded.right.tns_data.present);

        let decoded = decode_er_channel_pair_spectra_fixed_from_reader(
            &mut BitReader::new(&tns),
            4,
            512,
            true,
            true,
            false,
            false,
            false,
            &mut pns_random,
        )
        .unwrap();
        assert!(decoded.left.tns_data.present);
        assert!(decoded.right.tns_data.present);

        for end in 0..payload.len() {
            let truncated = &payload[..end];
            assert!(decode_er_channel_pair_spectra_from_reader(
                &mut BitReader::new(truncated),
                4,
                512,
                true,
                false,
                false,
                false,
                false,
                &mut pns_random,
            )
            .is_err());
            assert!(decode_er_channel_pair_spectra_fixed_from_reader(
                &mut BitReader::new(truncated),
                4,
                512,
                true,
                false,
                false,
                false,
                false,
                &mut pns_random,
            )
            .is_err());
            assert!(decode_er_channel_pair_spectra_from_reader(
                &mut BitReader::new(truncated),
                4,
                512,
                true,
                true,
                false,
                false,
                false,
                &mut pns_random,
            )
            .is_err());
            assert!(decode_er_channel_pair_spectra_fixed_from_reader(
                &mut BitReader::new(truncated),
                4,
                512,
                true,
                true,
                false,
                false,
                false,
                &mut pns_random,
            )
            .is_err());
        }
    }

    #[test]
    fn decodes_er_aac_eld_core_and_consumes_explicit_ld_sbr_payload() {
        let (spectral_body, spectral_bits) = (0u16..=u16::MAX)
            .find_map(|candidate| {
                let bytes = candidate.to_be_bytes();
                let mut reader = BitReader::new(&bytes);
                let tuple = crate::spectral::decode_spectral_tuple(&mut reader, 1).ok()?;
                tuple.iter().any(|&value| value != 0).then_some((
                    (candidate as u32) >> (16 - reader.bits_read()),
                    reader.bits_read(),
                ))
            })
            .unwrap();
        let sbr_header = crate::asc::LdSbrHeader {
            amp_resolution: true,
            start_frequency: 5,
            stop_frequency: 3,
            crossover_band: 2,
            frequency_scale: Some(0),
            alter_scale: Some(false),
            noise_bands: Some(2),
            limiter_bands: Some(2),
            limiter_gains: Some(2),
            interpol_frequency: Some(true),
            smoothing_mode: Some(false),
            ..crate::asc::LdSbrHeader::default()
        };
        let tables = crate::ld_sbr::LdSbrFrequencyTables::from_header(&sbr_header, 88_200).unwrap();
        let mut writer = BitWriter::new();
        writer.write(180, 8); // ELD core global gain
        writer.write(1, 6); // maxSfb
        writer.write(1, 4); // spectral codebook 1
        writer.write(1, 5);
        writer.write_bool(false); // scalefactor delta zero
        writer.write_bool(false); // core TNS absent
        writer.write(spectral_body, spectral_bits);
        writer.write_bool(false); // no SBR frame header
        writer.write_bool(false); // no SBR data_extra
        writer.write_bool(false); // FIXFIX grid
        writer.write(0, 2); // one envelope
        writer.write_bool(true); // current amp resolution
        writer.write_bool(true); // high frequency resolution
        writer.write_bool(false); // envelope frequency direction
        writer.write_bool(false); // noise frequency direction
        for _ in 0..tables.noise_band_count() {
            writer.write(0, 2); // inverse filtering off
        }
        writer.write(0, 6);
        for _ in 1..tables.high_band_count() {
            writer.write_bool(false); // zero envelope delta
        }
        writer.write(31, 5);
        for _ in 1..tables.noise_band_count() {
            writer.write_bool(false); // zero noise delta
        }
        writer.write_bool(false); // no added harmonics
        writer.write_bool(false); // no extended data
        let payload = writer.finish();
        let asc = AudioSpecificConfig {
            audio_object_type: 39,
            sampling_frequency_index: 4,
            sampling_frequency: 44_100,
            channel_configuration: 1,
            extension: None,
            ga_specific: None,
            eld_specific: Some(crate::asc::EldSpecificConfig {
                sbr_present: true,
                sbr_sampling_rate: true,
                sbr_headers: vec![sbr_header],
                ..crate::asc::EldSpecificConfig::default()
            }),
            usac_config: None,
            error_protection_config: Some(0),
            program_config: None,
            bits_read: 0,
        };
        let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
        let pcm = decoder
            .decode_raw_data_block_multichannel_f32_strict(&payload)
            .unwrap()
            .interleaved_f32();
        assert_eq!(pcm.len(), 1024);
        assert!(pcm.iter().all(|sample| sample.is_finite()));
        assert!(pcm.iter().any(|&sample| sample != 0.0));
        let mono_sbr_frame = decoder.last_ld_sbr_frames[0].clone();
        let next = decoder.f32_concealment_spectral_frame().unwrap();
        let interpolated = decoder.conceal_f32_interpolated(&next).unwrap();
        assert_eq!(interpolated.len(), 1024);
        assert!(interpolated.iter().all(|sample| sample.is_finite()));
        let concealed = decoder.conceal_f32_interleaved().unwrap();
        assert_eq!(concealed.len(), 1024);
        assert!(concealed.iter().all(|sample| sample.is_finite()));
        assert!(concealed.iter().any(|&sample| sample != 0.0));
        let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
        let pcm = decoder
            .decode_raw_data_block_fixed_interleaved_i16_strict(&payload)
            .unwrap();
        assert_eq!(pcm.len(), 1024);
        assert!(pcm.iter().any(|&sample| sample != 0));
        let next = decoder.fixed_concealment_spectral_frame().unwrap();
        let interpolated = decoder.conceal_fixed_interpolated_i16(&next).unwrap();
        assert_eq!(interpolated.len(), 1024);
        let concealed = decoder.conceal_fixed_interleaved_i16().unwrap();
        assert_eq!(concealed.len(), 1024);
        assert!(concealed.iter().any(|&sample| sample != 0));

        let mut stereo_frame = mono_sbr_frame.clone();
        stereo_frame.prefix.right = Some(stereo_frame.prefix.left.clone());
        stereo_frame.right = Some(stereo_frame.left.clone());
        stereo_frame.right_dequantized = Some(stereo_frame.left_dequantized.clone());
        stereo_frame.right_harmonics = Some(stereo_frame.left_harmonics.clone());
        let mut stereo_asc = asc.clone();
        stereo_asc.channel_configuration = 2;

        let mut decoder = AacLcDecoder::from_audio_specific_config(&stereo_asc).unwrap();
        let mut floating = vec![vec![0.0; 512], vec![0.0; 512]];
        decoder
            .process_ld_sbr_f32(&mut floating, &[stereo_frame.clone()])
            .unwrap();
        assert!(floating.iter().all(|channel| channel.len() == 1024));
        assert!(floating.iter().flatten().all(|sample| sample.is_finite()));

        let mut decoder = AacLcDecoder::from_audio_specific_config(&stereo_asc).unwrap();
        let mut fixed = vec![vec![0; 512], vec![0; 512]];
        decoder
            .process_ld_sbr_fixed(&mut fixed, &[stereo_frame.clone()])
            .unwrap();
        assert!(fixed.iter().all(|channel| channel.len() == 1024));

        let mut decoder = AacLcDecoder::from_audio_specific_config(&stereo_asc).unwrap();
        let q31 = decoder
            .process_ld_sbr_fixed_q31(&[vec![0; 512], vec![0; 512]], &[stereo_frame])
            .unwrap();
        assert!(q31.iter().all(|channel| channel.len() == 1024));

        #[cfg(feature = "ffi")]
        {
            let mut config = asc.to_bytes().unwrap();
            let mut fdk = crate::Decoder::open(crate::TransportType::Raw).unwrap();
            fdk.configure_raw(&mut config).unwrap();
            let mut fdk_pcm = vec![0i16; 2048];
            let mut samples = 0;
            for _ in 0..8 {
                samples = fdk.decode_access_unit_i16(&payload, &mut fdk_pcm).unwrap();
            }
            assert_eq!(samples, 1024);
            assert!(fdk_pcm[..samples].iter().any(|&sample| sample != 0));

            let mut pure = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
            let mut pure_pcm = Vec::new();
            for _ in 0..8 {
                pure_pcm = pure
                    .decode_raw_data_block_fixed_interleaved_i16_strict(&payload)
                    .unwrap();
            }
            assert_eq!(pure_pcm.len(), samples);
            assert!(pure_pcm.iter().any(|&sample| sample != 0));
            let fdk_energy = fdk_pcm[..samples]
                .iter()
                .map(|&sample| (sample as f64).powi(2))
                .sum::<f64>();
            let pure_energy = pure_pcm
                .iter()
                .map(|&sample| (sample as f64).powi(2))
                .sum::<f64>();
            let rms_ratio = (pure_energy / fdk_energy).sqrt();
            let pure_peak = pure_pcm
                .iter()
                .map(|sample| sample.unsigned_abs())
                .max()
                .unwrap();
            let fdk_peak = fdk_pcm[..samples]
                .iter()
                .map(|sample| sample.unsigned_abs())
                .max()
                .unwrap();
            assert!(
                (0.75..=1.5).contains(&rms_ratio),
                "LD-SBR RMS ratio {rms_ratio}, peaks pure={pure_peak} FDK={fdk_peak}"
            );
        }
    }

    #[test]
    fn decodes_er_aac_eld_ep_config_1_staged_cpe() {
        let mut writer = BitWriter::new();
        writer.write(1, 6); // shared implicit-long maxSfb
        writer.write(0, 2); // no MS stereo
        writer.write(100, 8); // left global gain
        writer.write(ZERO_HCB as u32, 4);
        writer.write(1, 5);
        writer.write_bool(true); // left TNS payload is staged after both headers
        writer.write(100, 8); // right global gain before either spectrum
        writer.write(ZERO_HCB as u32, 4);
        writer.write(1, 5);
        writer.write_bool(true); // right TNS payload is staged after both headers
        writer.write(0, 2); // left n_filt = 0
        writer.write(0, 2); // right n_filt = 0
                            // Both spectral payloads are empty for ZERO_HCB.
        let payload = writer.finish();
        let asc = AudioSpecificConfig {
            audio_object_type: 39,
            sampling_frequency_index: 4,
            sampling_frequency: 44_100,
            channel_configuration: 2,
            extension: None,
            ga_specific: None,
            eld_specific: Some(crate::asc::EldSpecificConfig::default()),
            usac_config: None,
            error_protection_config: Some(1),
            program_config: None,
            bits_read: 0,
        };
        let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
        let pcm = decoder
            .decode_raw_data_block_f32(&payload)
            .unwrap()
            .interleaved_f32();
        assert_eq!(pcm.len(), 1024);
        assert!(pcm.iter().all(|&sample| sample == 0.0));

        let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
        let pcm = decoder
            .decode_raw_data_block_fixed_interleaved_i16(&payload)
            .unwrap();
        assert_eq!(pcm.len(), 1024);
        assert!(pcm.iter().all(|&sample| sample == 0));
    }

    #[test]
    fn decodes_er_aac_eld_ep_config_1_staged_rvlc_hcr_cpe() {
        let mut writer = BitWriter::new();
        writer.write(1, 6); // shared implicit-long maxSfb
        writer.write(0, 2); // no MS stereo
        for _ in 0..2 {
            writer.write(100, 8); // global gain
            writer.write(ZERO_HCB as u32, 4);
            writer.write(1, 5);
            writer.write_bool(false); // sf_concealment
            writer.write(100, 8); // reverse global gain
            writer.write(0, 9); // zero RVLC payload bits
            writer.write_bool(false); // no RVLC escapes
            writer.write_bool(true); // TNS payload follows both channel headers
        }
        writer.write(0, 2); // left n_filt = 0
        writer.write(0, 2); // right n_filt = 0
        for _ in 0..2 {
            writer.write(0, 14); // no reordered HCR payload
            writer.write(0, 6); // no longest HCR codeword
                                // RVLC and spectral payloads are empty for this channel.
        }
        let payload = writer.finish();
        let asc = AudioSpecificConfig {
            audio_object_type: 39,
            sampling_frequency_index: 4,
            sampling_frequency: 44_100,
            channel_configuration: 2,
            extension: None,
            ga_specific: None,
            eld_specific: Some(crate::asc::EldSpecificConfig {
                scalefactor_data_resilience: true,
                spectral_data_resilience: true,
                ..crate::asc::EldSpecificConfig::default()
            }),
            usac_config: None,
            error_protection_config: Some(1),
            program_config: None,
            bits_read: 0,
        };
        let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
        let pcm = decoder
            .decode_raw_data_block_f32(&payload)
            .unwrap()
            .interleaved_f32();
        assert_eq!(pcm.len(), 1024);
        assert!(pcm.iter().all(|&sample| sample == 0.0));

        let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
        let pcm = decoder
            .decode_raw_data_block_fixed_interleaved_i16(&payload)
            .unwrap();
        assert_eq!(pcm.len(), 1024);
        assert!(pcm.iter().all(|&sample| sample == 0));

        let mut pns_random = PnsRandomState::new(1);
        for end in 0..payload.len() {
            let truncated = &payload[..end];
            assert!(decode_er_channel_pair_spectra_from_reader(
                &mut BitReader::new(truncated),
                4,
                512,
                true,
                true,
                false,
                true,
                true,
                &mut pns_random,
            )
            .is_err());
            assert!(decode_er_channel_pair_spectra_fixed_from_reader(
                &mut BitReader::new(truncated),
                4,
                512,
                true,
                true,
                false,
                true,
                true,
                &mut pns_random,
            )
            .is_err());
        }
    }

    #[test]
    fn conceals_malformed_complete_hcr_payload_but_not_truncation() {
        let mut writer = BitWriter::new();
        writer.write(0, 4); // element instance tag
        writer.write(100, 8); // global gain
        writer.write_bool(false);
        writer.write(0, 2); // long window
        writer.write_bool(false);
        writer.write(1, 6); // maxSfb
        writer.write_bool(false);
        writer.write(1, 4); // spectral codebook 1
        writer.write(1, 5); // one band
        writer.write_bool(false); // zero scalefactor delta
        writer.write_bool(false); // pulse absent
        writer.write_bool(false); // TNS absent
        writer.write_bool(false); // gain control absent
        writer.write(1, 14); // complete but too-short reordered payload
        writer.write(1, 6);
        writer.write_bool(true);
        let payload = writer.finish();
        let asc = AudioSpecificConfig {
            audio_object_type: 17,
            sampling_frequency_index: 4,
            sampling_frequency: 44_100,
            channel_configuration: 1,
            extension: None,
            ga_specific: Some(crate::asc::GaSpecificConfig {
                extension_flag: true,
                spectral_data_resilience: true,
                extension_flag3: Some(false),
                ..crate::asc::GaSpecificConfig::default()
            }),
            eld_specific: None,
            usac_config: None,
            error_protection_config: Some(1),
            program_config: None,
            bits_read: 0,
        };
        let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
        assert!(decoder
            .decode_raw_data_block_fixed_interleaved_i16(&payload)
            .unwrap()
            .iter()
            .all(|&sample| sample == 0));

        let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
        let truncated = &payload[..payload.len() - 1];
        assert!(decoder
            .decode_raw_data_block_fixed_interleaved_i16(truncated)
            .unwrap_err()
            .is_unexpected_eof());
    }

    #[test]
    fn registers_data_stream_crc_region_after_audio_elements() {
        let mut writer = BitWriter::new();
        write_zero_sce_payload_bits(&mut writer, false);
        writer.write(ElementId::DataStream.bits() as u32, 3);
        writer.write(0, 4); // element instance tag
        writer.write_bool(false); // no byte alignment
        writer.write(1, 8); // one data byte
        writer.write(0xa5, 8);
        let payload = writer.finish();
        let mut decoder = AacLcDecoder::new(4, 1).unwrap();
        decoder.decode_raw_data_block_f32(&payload).unwrap();
        assert_eq!(decoder.adts_crc_regions.len(), 2);
        assert_eq!(decoder.adts_crc_regions[1].len(), 21);
    }

    #[test]
    fn decodes_crc_less_adts_multi_raw_data_block_frame_from_end_markers() {
        let mut writer = BitWriter::new();
        for _ in 0..2 {
            write_zero_sce_payload_bits(&mut writer, false);
            writer.write(ElementId::End.bits() as u32, 3);
        }
        let payload = writer.finish();
        let mut header = AdtsHeader::aac_lc(44_100, 1, payload.len()).unwrap();
        header.number_of_raw_data_blocks_in_frame = 1;
        let mut frame = vec![0; header.header_len()];
        header.write(&mut frame).unwrap();
        frame.extend_from_slice(&payload);

        let mut decoder = AacLcDecoder::new(4, 1).unwrap();
        let frames = decoder.decode_adts_frame_blocks_f32(&frame).unwrap();
        assert_eq!(frames.len(), 2);
        assert!(frames.iter().all(|decoded| {
            decoded.channels() == 1
                && decoded
                    .interleaved_f32()
                    .iter()
                    .all(|sample| *sample == 0.0)
        }));

        let mut fixed_decoder = AacLcDecoder::new(4, 1).unwrap();
        let fixed = fixed_decoder
            .decode_adts_frame_blocks_fixed_interleaved_i16(&frame)
            .unwrap();
        assert_eq!(fixed.len(), 2);
        assert!(fixed.iter().flatten().all(|&sample| sample == 0));

        let mut trailing_payload = payload;
        trailing_payload.push(0x80);
        let mut trailing_header = AdtsHeader::aac_lc(44_100, 1, trailing_payload.len()).unwrap();
        trailing_header.number_of_raw_data_blocks_in_frame = 1;
        let mut trailing_frame = vec![0; trailing_header.header_len()];
        trailing_header.write(&mut trailing_frame).unwrap();
        trailing_frame.extend_from_slice(&trailing_payload);
        assert!(matches!(
            AacLcDecoder::new(4, 1)
                .unwrap()
                .decode_adts_frame_blocks_f32(&trailing_frame),
            Err(DecodeError::NonZeroTrailingBits(_))
        ));
        assert!(matches!(
            AacLcDecoder::new(4, 1)
                .unwrap()
                .decode_adts_frame_blocks_fixed_interleaved_i16(&trailing_frame),
            Err(DecodeError::NonZeroTrailingBits(_))
        ));
    }

    #[test]
    fn adts_stream_facades_forward_transport_parse_errors() {
        macro_rules! assert_stream_error {
            ($method:ident) => {{
                let mut decoder = AacLcDecoder::new(4, 1).unwrap();
                assert!(matches!(
                    decoder.$method(&[0]).next(),
                    Some(Err(DecodeError::Adts(_)))
                ));
            }};
        }

        assert_stream_error!(decode_adts_stream_interleaved_f32);
        assert_stream_error!(decode_adts_stream_interleaved_i16);
        assert_stream_error!(decode_adts_stream_fixed_interleaved_i16);
        assert_stream_error!(decode_adts_stream_multichannel_f32);
        assert_stream_error!(decode_adts_stream_multichannel_interleaved_f32);
        assert_stream_error!(decode_adts_stream_multichannel_interleaved_i16);
        assert_stream_error!(decode_adts_stream_multichannel_fixed_interleaved_i16);
    }

    #[test]
    fn strict_adts_stream_iterators_reject_nonzero_trailing_bits_per_frame() {
        let good_payload = zero_sce_payload(false);
        let good_header = AdtsHeader::aac_lc(44_100, 1, good_payload.len()).unwrap();
        let mut good_frame = vec![0; good_header.header_len()];
        good_header.write(&mut good_frame).unwrap();
        good_frame.extend_from_slice(&good_payload);

        let mut bad_payload = zero_sce_payload(false);
        bad_payload.push(0x80);
        let bad_header = AdtsHeader::aac_lc(44_100, 1, bad_payload.len()).unwrap();
        let mut bad_frame = vec![0; bad_header.header_len()];
        bad_header.write(&mut bad_frame).unwrap();
        bad_frame.extend_from_slice(&bad_payload);

        let mut stream = good_frame.clone();
        stream.extend_from_slice(&bad_frame);

        let mut decoder = AacLcDecoder::from_adts_header(good_header).unwrap();
        let mut iter = decoder.decode_adts_stream_interleaved_i16_strict(&stream);
        assert!(iter
            .next()
            .unwrap()
            .unwrap()
            .iter()
            .all(|sample| *sample == 0));
        assert_eq!(
            iter.next().unwrap().unwrap_err(),
            DecodeError::NonZeroTrailingBits(10)
        );
        assert!(iter.next().is_none());
    }

    #[test]
    fn strict_multichannel_fixed_stream_iterator_rejects_nonzero_trailing_bits() {
        let mut payload = zero_cpe_payload(0);
        payload.push(0x80);
        let header = AdtsHeader::aac_lc(44_100, 2, payload.len()).unwrap();
        let mut frame = vec![0; header.header_len()];
        header.write(&mut frame).unwrap();
        frame.extend_from_slice(&payload);

        let mut decoder = AacLcDecoder::from_adts_header(header).unwrap();
        let err = decoder
            .decode_adts_stream_multichannel_fixed_interleaved_i16_strict(&frame)
            .next()
            .unwrap()
            .unwrap_err();

        assert!(matches!(err, DecodeError::NonZeroTrailingBits(_)));
    }

    #[test]
    fn decoder_rejects_aac_lc_prediction_data_in_raw_sce() {
        let mut writer = BitWriter::new();
        writer.write(ElementId::SingleChannel.bits() as u32, 3);
        writer.write(0, 4); // element_instance_tag
        writer.write(100, 8); // global_gain
        writer.write_bool(false); // ics reserved bit
        writer.write(WindowSequence::OnlyLong.bits() as u32, 2);
        writer.write_bool(false); // sine
        writer.write(1, 6); // max_sfb
        writer.write_bool(true); // predictor_data_present => unsupported in AAC-LC
        let payload = writer.finish();

        let mut decoder = AacLcDecoder::new(4, 1).unwrap();
        assert_eq!(
            decoder.decode_raw_data_block_f32(&payload).unwrap_err(),
            DecodeError::Raw(RawError::Ics(IcsError::PredictionUnsupported))
        );
    }

    #[test]
    fn decoder_rejects_unsupported_adts_aot_and_multiple_raw_blocks() {
        let payload = zero_sce_payload(false);
        let unsupported_aot_header = AdtsHeader::new(
            crate::adts::MpegVersion::Mpeg4,
            2, // ADTS profile 2 => audioObjectType 3, not AAC-LC
            4,
            1,
            payload.len(),
        )
        .unwrap();
        let mut unsupported_aot_frame = vec![0; unsupported_aot_header.header_len()];
        unsupported_aot_header
            .write(&mut unsupported_aot_frame)
            .unwrap();
        unsupported_aot_frame.extend_from_slice(&payload);

        let mut decoder = AacLcDecoder::new(4, 1).unwrap();
        assert_eq!(
            decoder
                .decode_adts_frame_f32(&unsupported_aot_frame)
                .unwrap_err(),
            DecodeError::UnsupportedAudioObjectType(3)
        );

        let mut multi_block_header = AdtsHeader::aac_lc(44_100, 1, payload.len()).unwrap();
        multi_block_header.number_of_raw_data_blocks_in_frame = 1;
        let mut multi_block_frame = vec![0; multi_block_header.header_len()];
        multi_block_header.write(&mut multi_block_frame).unwrap();
        multi_block_frame.extend_from_slice(&payload);

        assert_eq!(
            decoder
                .decode_adts_frame_f32(&multi_block_frame)
                .unwrap_err(),
            DecodeError::UnsupportedRawBlocksInAdtsFrame(1)
        );
    }

    #[test]
    fn every_adts_frame_facade_rejects_changed_configuration() {
        let payload = zero_sce_payload(false);
        let make_frame = |header: AdtsHeader| {
            let mut frame = vec![0; header.header_len()];
            header.write(&mut frame).unwrap();
            frame.extend_from_slice(&payload);
            frame
        };
        let unsupported_aot = make_frame(
            AdtsHeader::new(crate::adts::MpegVersion::Mpeg4, 2, 4, 1, payload.len()).unwrap(),
        );
        let mut multi_block_header = AdtsHeader::aac_lc(44_100, 1, payload.len()).unwrap();
        multi_block_header.number_of_raw_data_blocks_in_frame = 1;
        let multiple_blocks = make_frame(multi_block_header);
        let changed_frequency = make_frame(AdtsHeader::aac_lc(48_000, 1, payload.len()).unwrap());

        macro_rules! assert_all_facades {
            ($frame:expr, $expected:expr) => {{
                let mut decoder = AacLcDecoder::new(4, 1).unwrap();
                assert_eq!(
                    decoder.decode_adts_frame_f32($frame).unwrap_err(),
                    $expected
                );
                assert_eq!(
                    decoder.decode_adts_frame_f32_strict($frame).unwrap_err(),
                    $expected
                );
                assert_eq!(
                    decoder
                        .decode_adts_frame_fixed_interleaved_i16($frame)
                        .unwrap_err(),
                    $expected
                );
                assert_eq!(
                    decoder
                        .decode_adts_frame_fixed_interleaved_i16_strict($frame)
                        .unwrap_err(),
                    $expected
                );
                assert_eq!(
                    decoder
                        .decode_adts_frame_multichannel_f32($frame)
                        .unwrap_err(),
                    $expected
                );
                assert_eq!(
                    decoder
                        .decode_adts_frame_multichannel_f32_strict($frame)
                        .unwrap_err(),
                    $expected
                );
                assert_eq!(
                    decoder
                        .decode_adts_frame_multichannel_fixed_interleaved_i16($frame)
                        .unwrap_err(),
                    $expected
                );
                assert_eq!(
                    decoder
                        .decode_adts_frame_multichannel_fixed_interleaved_i16_strict($frame)
                        .unwrap_err(),
                    $expected
                );
            }};
        }

        assert_all_facades!(&unsupported_aot, DecodeError::UnsupportedAudioObjectType(3));
        assert_all_facades!(
            &multiple_blocks,
            DecodeError::UnsupportedRawBlocksInAdtsFrame(1)
        );
        assert_all_facades!(&changed_frequency, DecodeError::AdtsConfigChanged);
    }

    #[test]
    fn stateful_decoder_validates_channel_configuration() {
        let payload = zero_cpe_payload(0);
        let mut decoder = AacLcDecoder::new(4, 1).unwrap();

        assert_eq!(
            decoder
                .decode_raw_data_block_f32(&payload)
                .unwrap_err()
                .to_string(),
            "AAC channel configuration expects 1 channel(s), decoded 2"
        );
        assert_eq!(
            AacLcDecoder::new(4, 1)
                .unwrap()
                .decode_raw_data_block_multichannel_fixed_interleaved_i16(&payload),
            Err(DecodeError::ChannelConfigurationMismatch {
                expected: 1,
                actual: 2,
            })
        );

        let mono = AacLcDecoder::new(4, 1)
            .unwrap()
            .decode_raw_data_block_f32(&zero_sce_payload(false))
            .unwrap();
        decoder.channel_configuration = 2;
        assert_eq!(
            decoder.validate_frame_channel_configuration(&mono),
            Err(DecodeError::ChannelConfigurationMismatch {
                expected: 2,
                actual: 1,
            })
        );
        decoder.channel_configuration = 7;
        assert_eq!(
            decoder.validate_frame_channel_configuration(&mono),
            Err(DecodeError::UnsupportedChannelConfiguration(7))
        );
    }

    #[test]
    fn sbr_processing_rejects_payload_and_channel_layout_mismatches() {
        let payload = SbrFillPayload {
            extension_type: crate::sbr::EXT_SBR_DATA,
            transmitted_crc: None,
            header_present: false,
            header: None,
            frame_data: Vec::new(),
            frame_data_bits: 0,
        };
        let mut decoder = AacLcDecoder::new(4, 1).unwrap();
        decoder.ordinary_sbr_output_frequency = Some(44_100);

        let mut f32_channels = vec![vec![0.0; 1024]];
        assert_eq!(
            decoder.process_ordinary_sbr_f32(&mut f32_channels, &[], &[]),
            Ok(())
        );
        assert_eq!(
            decoder.process_ordinary_sbr_f32(&mut f32_channels, &[payload.clone()], &[]),
            Err(DecodeError::SbrPayloadLayoutMismatch)
        );
        let mut fixed_channels = vec![vec![0; 1024]];
        assert_eq!(
            decoder.process_ordinary_sbr_fixed(&mut fixed_channels, &[], &[]),
            Ok(())
        );
        assert_eq!(
            decoder.process_ordinary_sbr_fixed(&mut fixed_channels, &[payload], &[]),
            Err(DecodeError::SbrPayloadLayoutMismatch)
        );
        assert_eq!(
            decoder.process_ld_sbr_fixed_q31(&[Vec::new()], &[]),
            Err(DecodeError::SbrPayloadLayoutMismatch)
        );

        let invalid_frequency_payload = SbrFillPayload {
            extension_type: crate::sbr::EXT_SBR_DATA,
            transmitted_crc: None,
            header_present: true,
            header: Some(LdSbrHeader {
                start_frequency: 5,
                stop_frequency: 8,
                ..LdSbrHeader::default()
            }),
            frame_data: Vec::new(),
            frame_data_bits: 0,
        };
        for stereo in [false, true] {
            let mut decoder = AacLcDecoder::new(4, if stereo { 2 } else { 1 }).unwrap();
            decoder.ordinary_sbr_output_frequency = Some(1);
            decoder.ordinary_sbr_parsers = vec![None];
            decoder.ordinary_sbr_fixed_parsers = vec![None];
            let channel_count = if stereo { 2 } else { 1 };
            let mut floating = vec![vec![0.0; 1024]; channel_count];
            assert!(decoder
                .process_ordinary_sbr_f32(
                    &mut floating,
                    std::slice::from_ref(&invalid_frequency_payload),
                    &[stereo],
                )
                .is_err());
            let mut fixed = vec![vec![0; 1024]; channel_count];
            assert!(decoder
                .process_ordinary_sbr_fixed(
                    &mut fixed,
                    std::slice::from_ref(&invalid_frequency_payload),
                    &[stereo],
                )
                .is_err());
        }
    }

    #[test]
    fn stateful_decoder_iterates_adts_stream_to_interleaved_frames() {
        let payload = zero_cpe_payload(0);
        let header = AdtsHeader::aac_lc(44_100, 2, payload.len()).unwrap();
        let mut one_frame = vec![0; header.header_len()];
        header.write(&mut one_frame).unwrap();
        one_frame.extend_from_slice(&payload);
        let mut stream = one_frame.clone();
        stream.extend_from_slice(&one_frame);

        let mut decoder = AacLcDecoder::from_adts_header(header).unwrap();
        let frames = decoder
            .decode_adts_stream_interleaved_i16(&stream)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].len(), 2048);
        assert_eq!(frames[1].len(), 2048);
        assert!(frames.iter().flatten().all(|sample| *sample == 0));
    }

    #[test]
    fn stateful_decoder_decodes_raw_sce_fixed_interleaved_i16() {
        let payload = zero_sce_payload(false);
        let mut decoder = AacLcDecoder::new(4, 1).unwrap();
        let pcm = decoder
            .decode_raw_data_block_fixed_interleaved_i16(&payload)
            .unwrap();

        assert_eq!(pcm.len(), 1024);
        assert!(pcm.iter().all(|sample| *sample == 0));
    }

    #[test]
    fn fixed_decoder_conceals_from_last_spectral_frame() {
        let payload = zero_sce_payload(false);
        let mut decoder = AacLcDecoder::new(4, 1).unwrap();
        assert!(decoder.fixed_concealment_spectral_frame().is_none());
        assert_eq!(
            decoder.conceal_fixed_interleaved_i16().unwrap_err(),
            DecodeError::NoConcealmentReference
        );
        assert_eq!(
            decoder
                .conceal_fixed_interpolated_i16(&FixedConcealmentSpectralFrame {
                    channels: Vec::new(),
                })
                .unwrap_err(),
            DecodeError::NoConcealmentReference
        );
        let decoded = decoder
            .decode_raw_data_block_fixed_interleaved_i16(&payload)
            .unwrap();
        let spectral = decoder.fixed_concealment_spectral_frame().unwrap();
        assert_eq!(spectral.channels.len(), 1);
        decoder.fixed_concealment_spectra[0].1.window_sequence = WindowSequence::LongStart;
        let concealed = decoder.conceal_fixed_interleaved_i16().unwrap();
        assert_eq!(decoded.len(), 1024);
        assert_eq!(concealed.len(), 1024);
        assert!(concealed.iter().all(|sample| *sample == 0));
        assert_eq!(decoder.fixed_concealment_state(), ConcealmentState::Single);
    }

    #[test]
    fn fixed_concealment_runs_fade_out_mute_and_fade_in_states() {
        let payload = zero_sce_payload(false);
        let mut decoder = AacLcDecoder::new(4, 1).unwrap();
        decoder
            .decode_raw_data_block_fixed_interleaved_i16(&payload)
            .unwrap();
        decoder.conceal_fixed_interleaved_i16().unwrap();
        assert_eq!(decoder.fixed_concealment_state(), ConcealmentState::Single);
        decoder.conceal_fixed_interleaved_i16().unwrap();
        assert_eq!(decoder.fixed_concealment_state(), ConcealmentState::FadeOut);
        for _ in 0..6 {
            decoder.conceal_fixed_interleaved_i16().unwrap();
        }
        assert_eq!(decoder.fixed_concealment_state(), ConcealmentState::Mute);

        decoder
            .decode_raw_data_block_fixed_interleaved_i16(&payload)
            .unwrap();
        assert_eq!(decoder.fixed_concealment_state(), ConcealmentState::FadeIn);
        for _ in 0..5 {
            decoder
                .decode_raw_data_block_fixed_interleaved_i16(&payload)
                .unwrap();
        }
        assert_eq!(decoder.fixed_concealment_state(), ConcealmentState::Ok);
    }

    #[test]
    fn f32_concealment_runs_fade_out_mute_and_fade_in_states() {
        let payload = nonzero_spectral_sce_payload();
        let mut decoder = AacLcDecoder::new(4, 1).unwrap();
        assert!(decoder.f32_concealment_spectral_frame().is_none());
        assert_eq!(
            decoder.conceal_f32_interleaved().unwrap_err(),
            DecodeError::NoConcealmentReference
        );
        assert_eq!(
            decoder
                .conceal_f32_interpolated(&F32ConcealmentSpectralFrame {
                    channels: Vec::new(),
                })
                .unwrap_err(),
            DecodeError::NoConcealmentReference
        );
        decoder.decode_raw_data_block_f32(&payload).unwrap();
        let spectral = decoder.f32_concealment_spectral_frame().unwrap();
        assert_eq!(spectral.channels.len(), 1);
        decoder.f32_concealment_spectra[0].1.window_sequence = WindowSequence::LongStart;
        let first = decoder.conceal_f32_interleaved().unwrap();
        assert!(first.iter().any(|sample| *sample != 0.0));
        assert_eq!(decoder.f32_concealment_state(), ConcealmentState::Single);
        decoder.conceal_f32_interleaved().unwrap();
        assert_eq!(decoder.f32_concealment_state(), ConcealmentState::FadeOut);
        for _ in 0..6 {
            decoder.conceal_f32_interleaved().unwrap();
        }
        assert_eq!(decoder.f32_concealment_state(), ConcealmentState::Mute);

        decoder.decode_raw_data_block_f32(&payload).unwrap();
        assert_eq!(decoder.f32_concealment_state(), ConcealmentState::FadeIn);
        for _ in 0..5 {
            decoder.decode_raw_data_block_f32(&payload).unwrap();
        }
        assert_eq!(decoder.f32_concealment_state(), ConcealmentState::Ok);
    }

    #[test]
    fn aac_eld_concealment_uses_low_delay_filterbanks_in_f32_and_fixed_paths() {
        let ics = test_ics(1);
        let floating = InverseQuantizedSpectrum {
            windows: vec![(0..512)
                .map(|index| (index as f32 * 0.071).sin() * 0.01)
                .collect()],
        };
        let next_floating = F32ConcealmentSpectralFrame {
            channels: vec![F32ConcealmentChannel {
                spectrum: InverseQuantizedSpectrum {
                    windows: vec![floating.windows[0]
                        .iter()
                        .map(|sample| sample * 2.0)
                        .collect()],
                },
                ics: ics.clone(),
            }],
        };
        let fixed = FixedInverseQuantizedSpectrum {
            windows: vec![floating.windows[0]
                .iter()
                .map(|sample| (*sample * 2_147_483_648.0) as i32)
                .collect()],
            window_exponents: vec![0],
        };
        let next_fixed = FixedConcealmentSpectralFrame {
            channels: vec![FixedConcealmentChannel {
                spectrum: FixedInverseQuantizedSpectrum {
                    windows: vec![fixed.windows[0]
                        .iter()
                        .map(|sample| sample.saturating_mul(2))
                        .collect()],
                    window_exponents: vec![0],
                },
                ics: ics.clone(),
            }],
        };

        let mut f32_decoder = AacLcDecoder::new_ga(39, 4, 1).unwrap();
        f32_decoder.f32_concealment_spectra = vec![(floating, ics.clone())];
        let concealed = f32_decoder.conceal_f32_interleaved().unwrap();
        assert_eq!(concealed.len(), 512);
        assert!(concealed.iter().any(|sample| *sample != 0.0));
        let interpolated = f32_decoder
            .conceal_f32_interpolated(&next_floating)
            .unwrap();
        assert_eq!(interpolated.len(), 512);
        assert!(interpolated.iter().any(|sample| *sample != 0.0));

        let mut fixed_decoder = AacLcDecoder::new_ga(39, 4, 1).unwrap();
        fixed_decoder.fixed_concealment_spectra = vec![(fixed, ics)];
        let concealed = fixed_decoder.conceal_fixed_interleaved_i16().unwrap();
        assert_eq!(concealed.len(), 512);
        assert!(concealed.iter().any(|sample| *sample != 0));
        let interpolated = fixed_decoder
            .conceal_fixed_interpolated_i16(&next_fixed)
            .unwrap();
        assert_eq!(interpolated.len(), 512);
        assert!(interpolated.iter().any(|sample| *sample != 0));

        let mut ld_f32_decoder = AacLcDecoder::new_ga(23, 4, 1).unwrap();
        ld_f32_decoder.f32_concealment_spectra = vec![(
            next_floating.channels[0].spectrum.clone(),
            next_floating.channels[0].ics.clone(),
        )];
        let interpolated = ld_f32_decoder
            .conceal_f32_interpolated(&next_floating)
            .unwrap();
        assert_eq!(interpolated.len(), 512);
        assert!(interpolated.iter().any(|sample| *sample != 0.0));

        let mut ld_fixed_decoder = AacLcDecoder::new_ga(23, 4, 1).unwrap();
        ld_fixed_decoder.fixed_concealment_spectra = vec![(
            next_fixed.channels[0].spectrum.clone(),
            next_fixed.channels[0].ics.clone(),
        )];
        let interpolated = ld_fixed_decoder
            .conceal_fixed_interpolated_i16(&next_fixed)
            .unwrap();
        assert_eq!(interpolated.len(), 512);
        assert!(interpolated.iter().any(|sample| *sample != 0));
    }

    #[test]
    fn eld_synthesis_rejects_multiple_spectral_windows_in_all_formats() {
        let floating = InverseQuantizedSpectrum {
            windows: vec![vec![0.0; 512], vec![0.0; 512]],
        };
        let mut floating_filterbank = LowDelayFilterbankF32::new(512).unwrap();
        assert_eq!(
            synthesize_aac_eld_frame_f32(&floating, &mut floating_filterbank),
            Err(DecodeError::Filterbank(
                FilterbankError::ExpectedOneLongWindow { actual: 2 }
            ))
        );

        let fixed = FixedInverseQuantizedSpectrum {
            windows: vec![vec![0; 512], vec![0; 512]],
            window_exponents: vec![0, 0],
        };
        let mut fixed_filterbank = LowDelayFilterbankQ31::new(512).unwrap();
        assert_eq!(
            synthesize_aac_eld_frame_fixed_i16(&fixed, &mut fixed_filterbank),
            Err(DecodeError::Filterbank(
                FilterbankError::ExpectedOneLongWindow { actual: 2 }
            ))
        );
        assert_eq!(
            synthesize_aac_eld_frame_fixed_q31(&fixed, &mut fixed_filterbank),
            Err(DecodeError::Filterbank(
                FilterbankError::ExpectedOneLongWindow { actual: 2 }
            ))
        );
    }

    #[test]
    fn interpolated_concealment_rejects_channel_and_eld_spectrum_layout_mismatches() {
        let ics = synthetic_long_ics();
        let floating = InverseQuantizedSpectrum {
            windows: vec![vec![0.0; 1024]],
        };
        let fixed = FixedInverseQuantizedSpectrum {
            windows: vec![vec![0; 1024]],
            window_exponents: vec![0],
        };
        let expected =
            DecodeError::ConcealmentInterpolation(SpectralInterpolationError::LayoutMismatch);

        let mut decoder = AacLcDecoder::new(4, 1).unwrap();
        decoder.f32_concealment_spectra = vec![(floating, ics.clone())];
        assert_eq!(
            decoder.conceal_f32_interpolated(&F32ConcealmentSpectralFrame {
                channels: Vec::new(),
            }),
            Err(expected.clone())
        );

        let mut decoder = AacLcDecoder::new(4, 1).unwrap();
        decoder.fixed_concealment_spectra = vec![(fixed, ics.clone())];
        assert_eq!(
            decoder.conceal_fixed_interpolated_i16(&FixedConcealmentSpectralFrame {
                channels: Vec::new(),
            }),
            Err(expected.clone())
        );

        let mut decoder = AacLcDecoder::new(4, 1).unwrap();
        decoder.f32_concealment_spectra = vec![(
            InverseQuantizedSpectrum {
                windows: vec![vec![0.0; 1024]],
            },
            ics.clone(),
        )];
        assert_eq!(
            decoder.conceal_f32_interpolated(&F32ConcealmentSpectralFrame {
                channels: vec![F32ConcealmentChannel {
                    spectrum: InverseQuantizedSpectrum {
                        windows: vec![vec![0.0; 1023]],
                    },
                    ics: ics.clone(),
                }],
            }),
            Err(expected.clone())
        );

        let mut decoder = AacLcDecoder::new(4, 1).unwrap();
        decoder.fixed_concealment_spectra = vec![(
            FixedInverseQuantizedSpectrum {
                windows: vec![vec![0; 1024]],
                window_exponents: vec![0],
            },
            ics.clone(),
        )];
        assert_eq!(
            decoder.conceal_fixed_interpolated_i16(&FixedConcealmentSpectralFrame {
                channels: vec![FixedConcealmentChannel {
                    spectrum: FixedInverseQuantizedSpectrum {
                        windows: vec![vec![0; 1023]],
                        window_exponents: vec![0],
                    },
                    ics: ics.clone(),
                }],
            }),
            Err(expected.clone())
        );

        let mut decoder = AacLcDecoder::new_ga(39, 4, 1).unwrap();
        decoder.f32_concealment_spectra = vec![(
            InverseQuantizedSpectrum {
                windows: vec![vec![0.0; 512]],
            },
            ics.clone(),
        )];
        assert_eq!(
            decoder.conceal_f32_interpolated(&F32ConcealmentSpectralFrame {
                channels: vec![F32ConcealmentChannel {
                    spectrum: InverseQuantizedSpectrum {
                        windows: vec![vec![0.0; 511]],
                    },
                    ics: ics.clone(),
                }],
            }),
            Err(expected.clone())
        );

        let mut decoder = AacLcDecoder::new_ga(39, 4, 1).unwrap();
        decoder.fixed_concealment_spectra = vec![(
            FixedInverseQuantizedSpectrum {
                windows: vec![vec![0; 512]],
                window_exponents: vec![0],
            },
            ics.clone(),
        )];
        assert_eq!(
            decoder.conceal_fixed_interpolated_i16(&FixedConcealmentSpectralFrame {
                channels: vec![FixedConcealmentChannel {
                    spectrum: FixedInverseQuantizedSpectrum {
                        windows: vec![vec![0; 511]],
                        window_exponents: vec![0],
                    },
                    ics,
                }],
            }),
            Err(expected)
        );
    }

    #[test]
    fn concealment_propagates_filterbank_length_mismatches_for_all_core_paths() {
        let long_ics = synthetic_long_ics();
        let long_f32 = InverseQuantizedSpectrum {
            windows: vec![vec![0.0; 1024]],
        };
        let long_fixed = FixedInverseQuantizedSpectrum {
            windows: vec![vec![0; 1024]],
            window_exponents: vec![0],
        };
        let long_next_f32 = F32ConcealmentSpectralFrame {
            channels: vec![F32ConcealmentChannel {
                spectrum: long_f32.clone(),
                ics: long_ics.clone(),
            }],
        };
        let long_next_fixed = FixedConcealmentSpectralFrame {
            channels: vec![FixedConcealmentChannel {
                spectrum: long_fixed.clone(),
                ics: long_ics.clone(),
            }],
        };

        let mut normal_f32 = AacLcDecoder::new(4, 1).unwrap();
        normal_f32.f32_concealment_spectra = vec![(long_f32, long_ics.clone())];
        normal_f32.channel_filterbanks[0] = LongBlockFilterbank::new(960).unwrap();
        assert!(normal_f32.conceal_f32_interleaved().is_err());
        assert!(normal_f32.conceal_f32_interpolated(&long_next_f32).is_err());

        let mut normal_fixed = AacLcDecoder::new(4, 1).unwrap();
        normal_fixed.fixed_concealment_spectra = vec![(long_fixed, long_ics.clone())];
        normal_fixed.fixed_channel_filterbanks[0] = FixedLongBlockFilterbank::new(960).unwrap();
        assert!(normal_fixed.conceal_fixed_interleaved_i16().is_err());
        assert!(normal_fixed
            .conceal_fixed_interpolated_i16(&long_next_fixed)
            .is_err());

        let low_delay_ics = test_ics(1);
        let low_delay_f32 = InverseQuantizedSpectrum {
            windows: vec![vec![0.0; 512]],
        };
        let low_delay_fixed = FixedInverseQuantizedSpectrum {
            windows: vec![vec![0; 512]],
            window_exponents: vec![0],
        };
        let low_delay_next_f32 = F32ConcealmentSpectralFrame {
            channels: vec![F32ConcealmentChannel {
                spectrum: low_delay_f32.clone(),
                ics: low_delay_ics.clone(),
            }],
        };
        let low_delay_next_fixed = FixedConcealmentSpectralFrame {
            channels: vec![FixedConcealmentChannel {
                spectrum: low_delay_fixed.clone(),
                ics: low_delay_ics.clone(),
            }],
        };

        let mut eld_f32 = AacLcDecoder::new_ga(39, 4, 1).unwrap();
        eld_f32.f32_concealment_spectra = vec![(low_delay_f32.clone(), low_delay_ics.clone())];
        eld_f32.eld_channel_filterbanks[0] = LowDelayFilterbankF32::new(480).unwrap();
        assert!(eld_f32.conceal_f32_interleaved().is_err());
        assert!(eld_f32
            .conceal_f32_interpolated(&low_delay_next_f32)
            .is_err());

        let mut eld_fixed = AacLcDecoder::new_ga(39, 4, 1).unwrap();
        eld_fixed.fixed_concealment_spectra =
            vec![(low_delay_fixed.clone(), low_delay_ics.clone())];
        eld_fixed.eld_fixed_channel_filterbanks[0] = LowDelayFilterbankQ31::new(480).unwrap();
        assert!(eld_fixed.conceal_fixed_interleaved_i16().is_err());
        assert!(eld_fixed
            .conceal_fixed_interpolated_i16(&low_delay_next_fixed)
            .is_err());

        let mut ld_f32 = AacLcDecoder::new_ga(23, 4, 1).unwrap();
        ld_f32.f32_concealment_spectra = vec![(low_delay_f32.clone(), low_delay_ics.clone())];
        ld_f32.channel_filterbanks[0] = LongBlockFilterbank::new(480).unwrap();
        assert!(ld_f32
            .conceal_f32_interpolated(&low_delay_next_f32)
            .is_err());

        let mut ld_fixed = AacLcDecoder::new_ga(23, 4, 1).unwrap();
        ld_fixed.fixed_concealment_spectra = vec![(low_delay_fixed, low_delay_ics)];
        ld_fixed.fixed_channel_filterbanks[0] = FixedLongBlockFilterbank::new(480).unwrap();
        assert!(ld_fixed
            .conceal_fixed_interpolated_i16(&low_delay_next_fixed)
            .is_err());
    }

    #[test]
    fn fixed_spectral_concealment_randomizes_single_loss_then_attenuates() {
        let mut spectrum = FixedInverseQuantizedSpectrum {
            windows: vec![vec![0x4000_0000; 8]],
            window_exponents: vec![0],
        };
        let mut phase = 0;
        prepare_fixed_concealment_spectrum(&mut spectrum, 0, &mut phase);
        assert!(spectrum.windows[0]
            .iter()
            .all(|value| (value.abs() as i64 - 0x4000_0000).abs() <= 1));
        assert!(spectrum.windows[0].iter().any(|value| *value < 0));
        assert!(spectrum.windows[0].iter().any(|value| *value > 0));

        prepare_fixed_concealment_spectrum(&mut spectrum, 1, &mut phase);
        assert!(spectrum.windows[0]
            .iter()
            .all(|value| value.abs() < 0x4000_0000));
    }

    #[test]
    fn spectral_mute_routes_zero_spectra_through_both_filterbanks() {
        let ics = test_ics(1);
        let mut floating_spectrum = vec![0.0; 1024];
        floating_spectrum[0] = 1.0;
        let mut floating = AacLcDecoder::new(4, 1).unwrap();
        floating.f32_concealment_spectra = vec![(
            InverseQuantizedSpectrum {
                windows: vec![floating_spectrum],
            },
            ics.clone(),
        )];
        let mut floating_noise = floating.clone();
        let muted = floating.conceal_f32_muted().unwrap();
        let noise = floating_noise.conceal_f32_interleaved().unwrap();
        assert!(muted.iter().all(|sample| *sample == 0.0));
        assert!(noise.iter().any(|sample| *sample != 0.0));
        assert_eq!(floating.f32_concealment_state(), ConcealmentState::Mute);

        let mut fixed_spectrum = vec![0; 1024];
        fixed_spectrum[0] = 0x4000_0000;
        let mut fixed = AacLcDecoder::new(4, 1).unwrap();
        fixed.fixed_concealment_spectra = vec![(
            FixedInverseQuantizedSpectrum {
                windows: vec![fixed_spectrum],
                window_exponents: vec![0],
            },
            ics,
        )];
        let mut fixed_noise = fixed.clone();
        let muted = fixed.conceal_fixed_muted_i16().unwrap();
        let noise = fixed_noise.conceal_fixed_interleaved_i16().unwrap();
        assert!(muted.iter().all(|sample| *sample == 0));
        assert!(noise.iter().any(|sample| *sample != 0));
        assert_eq!(fixed.fixed_concealment_state(), ConcealmentState::Mute);
    }

    #[test]
    fn fixed_interpolated_concealment_handles_long_short_transitions() {
        let long_channel = FixedConcealmentChannel {
            spectrum: FixedInverseQuantizedSpectrum {
                windows: vec![vec![0; 1024]],
                window_exponents: vec![0],
            },
            ics: synthetic_long_ics(),
        };
        let mut short_ics = synthetic_long_ics();
        short_ics.window_sequence = WindowSequence::EightShort;
        short_ics.total_sfb = IcsLimits::AAC_LC_MAX.short_sfb;
        short_ics.window_group_lengths = vec![1; 8];
        let short_channel = FixedConcealmentChannel {
            spectrum: FixedInverseQuantizedSpectrum {
                windows: vec![vec![0; 128]; 8],
                window_exponents: vec![0; 8],
            },
            ics: short_ics,
        };

        let mut long_to_short = AacLcDecoder::new(4, 1).unwrap();
        long_to_short.fixed_concealment_spectra =
            vec![(long_channel.spectrum.clone(), long_channel.ics.clone())];
        let pcm = long_to_short
            .conceal_fixed_interpolated_i16(&FixedConcealmentSpectralFrame {
                channels: vec![short_channel.clone()],
            })
            .unwrap();
        assert_eq!(pcm.len(), 1024);

        let mut short_to_long = AacLcDecoder::new(4, 1).unwrap();
        short_to_long.fixed_concealment_spectra =
            vec![(short_channel.spectrum.clone(), short_channel.ics.clone())];
        let pcm = short_to_long
            .conceal_fixed_interpolated_i16(&FixedConcealmentSpectralFrame {
                channels: vec![long_channel],
            })
            .unwrap();
        assert_eq!(pcm.len(), 1024);
    }

    #[test]
    fn f32_interpolated_concealment_handles_long_to_short_transition() {
        let long_ics = synthetic_long_ics();
        let mut short_ics = synthetic_long_ics();
        short_ics.window_sequence = WindowSequence::EightShort;
        short_ics.window_shape = WindowShape::Kbd;
        short_ics.total_sfb = IcsLimits::AAC_LC_MAX.short_sfb;
        short_ics.window_group_lengths = vec![1; 8];

        let mut decoder = AacLcDecoder::new(4, 1).unwrap();
        decoder.f32_concealment_spectra = vec![(
            InverseQuantizedSpectrum {
                windows: vec![vec![0.0; 1024]],
            },
            long_ics,
        )];
        let pcm = decoder
            .conceal_f32_interpolated(&F32ConcealmentSpectralFrame {
                channels: vec![F32ConcealmentChannel {
                    spectrum: InverseQuantizedSpectrum {
                        windows: vec![vec![0.0; 128]; 8],
                    },
                    ics: short_ics,
                }],
            })
            .unwrap();

        assert_eq!(pcm.len(), 1024);
        assert_eq!(decoder.f32_concealment_state(), ConcealmentState::Single);
    }

    #[test]
    fn stateful_decoder_decodes_pce_channel_config_zero_fixed_interleaved_i16() {
        let payload = pce_plus_zero_sce_payload();
        let decoded = AacLcDecoder::new(4, 0)
            .unwrap()
            .decode_raw_data_block_f32(&payload)
            .unwrap();
        assert!(matches!(decoded, DecodedAacLcFrame::Mono(_)));

        let header = AdtsHeader::aac_lc(44_100, 0, payload.len()).unwrap();
        let mut frame = vec![0; header.header_len()];
        header.write(&mut frame).unwrap();
        frame.extend_from_slice(&payload);

        let mut decoder = AacLcDecoder::from_adts_header(header).unwrap();
        let pcm = decoder
            .decode_adts_frame_fixed_interleaved_i16(&frame)
            .unwrap();

        assert_eq!(pcm.len(), 1024);
        assert!(pcm.iter().all(|sample| *sample == 0));
    }

    #[test]
    fn stateful_decoder_decodes_adts_cpe_fixed_interleaved_i16() {
        let payload = zero_cpe_payload(0);
        let header = AdtsHeader::aac_lc(44_100, 2, payload.len()).unwrap();
        let mut frame = vec![0; header.header_len()];
        header.write(&mut frame).unwrap();
        frame.extend_from_slice(&payload);

        let mut decoder = AacLcDecoder::from_adts_header(header).unwrap();
        let pcm = decoder
            .decode_adts_frame_fixed_interleaved_i16(&frame)
            .unwrap();

        assert_eq!(pcm.len(), 2048);
        assert!(pcm.iter().all(|sample| *sample == 0));
    }

    #[test]
    fn stateful_decoder_iterates_adts_stream_fixed_interleaved_i16() {
        let payload = zero_cpe_payload(0);
        let header = AdtsHeader::aac_lc(44_100, 2, payload.len()).unwrap();
        let mut frame = vec![0; header.header_len()];
        header.write(&mut frame).unwrap();
        frame.extend_from_slice(&payload);
        let mut stream = Vec::new();
        stream.extend_from_slice(&frame);
        stream.extend_from_slice(&frame);

        let mut decoder = AacLcDecoder::from_adts_header(header).unwrap();
        let frames = decoder
            .decode_adts_stream_fixed_interleaved_i16(&stream)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(frames.len(), 2);
        assert!(frames.iter().flatten().all(|sample| *sample == 0));
    }

    #[test]
    fn every_adts_stream_facade_decodes_strict_and_lenient_mono() {
        let payload = zero_sce_payload(false);
        let header = AdtsHeader::aac_lc(44_100, 1, payload.len()).unwrap();
        let mut frame = vec![0; header.header_len()];
        header.write(&mut frame).unwrap();
        frame.extend_from_slice(&payload);

        let mut decoder = AacLcDecoder::from_adts_header(header).unwrap();
        assert_eq!(
            decoder
                .decode_adts_stream_interleaved_f32(&frame)
                .next()
                .unwrap()
                .unwrap()
                .len(),
            1024
        );
        let mut decoder = AacLcDecoder::from_adts_header(header).unwrap();
        assert_eq!(
            decoder
                .decode_adts_stream_interleaved_f32_strict(&frame)
                .next()
                .unwrap()
                .unwrap()
                .len(),
            1024
        );
        let mut decoder = AacLcDecoder::from_adts_header(header).unwrap();
        assert_eq!(
            decoder
                .decode_adts_stream_interleaved_i16_strict(&frame)
                .next()
                .unwrap()
                .unwrap()
                .len(),
            1024
        );
        let mut decoder = AacLcDecoder::from_adts_header(header).unwrap();
        assert_eq!(
            decoder
                .decode_adts_stream_fixed_interleaved_i16_strict(&frame)
                .next()
                .unwrap()
                .unwrap()
                .len(),
            1024
        );

        let mut decoder = AacLcDecoder::from_adts_header(header).unwrap();
        let decoded = decoder
            .decode_adts_stream_multichannel_f32_strict(&frame)
            .next()
            .unwrap()
            .unwrap();
        assert_eq!(
            (decoded.channels(), decoded.samples_per_channel()),
            (1, 1024)
        );
        let mut decoder = AacLcDecoder::from_adts_header(header).unwrap();
        assert_eq!(
            decoder
                .decode_adts_stream_multichannel_interleaved_f32(&frame)
                .next()
                .unwrap()
                .unwrap()
                .len(),
            1024
        );
        let mut decoder = AacLcDecoder::from_adts_header(header).unwrap();
        assert_eq!(
            decoder
                .decode_adts_stream_multichannel_interleaved_f32_strict(&frame)
                .next()
                .unwrap()
                .unwrap()
                .len(),
            1024
        );
        let mut decoder = AacLcDecoder::from_adts_header(header).unwrap();
        assert_eq!(
            decoder
                .decode_adts_stream_multichannel_interleaved_i16_strict(&frame)
                .next()
                .unwrap()
                .unwrap()
                .len(),
            1024
        );
        let mut decoder = AacLcDecoder::from_adts_header(header).unwrap();
        assert_eq!(
            decoder
                .decode_adts_stream_multichannel_fixed_interleaved_i16_strict(&frame)
                .next()
                .unwrap()
                .unwrap()
                .len(),
            1024
        );

        let mut decoder = AacLcDecoder::from_adts_header(header).unwrap();
        assert_eq!(
            decoder
                .decode_adts_frame_multichannel_f32(&frame)
                .unwrap()
                .samples_per_channel(),
            1024
        );
        let mut decoder = AacLcDecoder::from_adts_header(header).unwrap();
        assert_eq!(
            decoder
                .decode_adts_frame_multichannel_f32_strict(&frame)
                .unwrap()
                .channels(),
            1
        );
        let mut decoder = AacLcDecoder::from_adts_header(header).unwrap();
        assert_eq!(
            decoder
                .decode_adts_frame_multichannel_fixed_interleaved_i16(&frame)
                .unwrap()
                .len(),
            1024
        );
        let mut decoder = AacLcDecoder::from_adts_header(header).unwrap();
        assert_eq!(
            decoder
                .decode_adts_frame_multichannel_fixed_interleaved_i16_strict(&frame)
                .unwrap()
                .len(),
            1024
        );
        let mut decoder = AacLcDecoder::from_adts_header(header).unwrap();
        assert_eq!(
            decoder
                .decode_adts_frame_multichannel_interleaved_f32(&frame)
                .unwrap()
                .len(),
            1024
        );
        let mut decoder = AacLcDecoder::from_adts_header(header).unwrap();
        assert_eq!(
            decoder
                .decode_adts_frame_multichannel_interleaved_i16(&frame)
                .unwrap()
                .len(),
            1024
        );

        let mut decoder = AacLcDecoder::from_adts_header(header).unwrap();
        assert_eq!(
            decoder
                .decode_raw_data_block_interleaved_i16(&payload)
                .unwrap()
                .len(),
            1024
        );
        assert!(matches!(
            decoder.decode_usac_access_unit_f32(&[]),
            Err(UsacDecodeError::UnsupportedConfiguration)
        ));
        assert!(matches!(
            decoder.decode_usac_access_unit_multichannel_f32(&[]),
            Err(UsacDecodeError::UnsupportedConfiguration)
        ));
        assert!(matches!(
            decoder.decode_usac_mps212_access_unit(&[]),
            Err(UsacDecodeError::UnsupportedConfiguration)
        ));
        assert_eq!(decoder.sampling_frequency_index(), 4);
        assert_eq!(decoder.channel_configuration(), 1);
        decoder.disable_drc();
    }

    #[test]
    fn unconfigured_drc_application_is_a_noop_for_both_sample_formats() {
        let mut decoder = AacLcDecoder::new(4, 1).unwrap();
        let mut floating = vec![vec![0.25, -0.5]];
        decoder.apply_configured_drc_f32(&mut floating).unwrap();
        assert_eq!(floating, vec![vec![0.25, -0.5]]);

        let mut fixed = vec![vec![123, -456]];
        decoder.apply_configured_drc_i16(&mut fixed).unwrap();
        assert_eq!(fixed, vec![vec![123, -456]]);

        let empty_config = UniDrcConfig {
            sample_rate: None,
            channel_layout: ChannelLayout {
                base_channel_count: 1,
                defined_layout: None,
                speaker_positions: Vec::new(),
            },
            downmix_instructions: Vec::new(),
            coefficients: Vec::new(),
            instructions: Vec::new(),
            extension_present: false,
            extensions: Vec::new(),
            bits_read: 0,
        };
        let empty_gain = UniDrcGain {
            sequences: Vec::new(),
            extension_present: false,
            extensions: Vec::new(),
            bits_read: 0,
        };
        decoder.configure_drc(empty_config, DrcSelectionRequest::default());
        decoder.update_drc_gain(empty_gain);
        decoder.apply_configured_drc_f32(&mut floating).unwrap();
        decoder.apply_configured_drc_i16(&mut fixed).unwrap();
    }

    #[test]
    fn legacy_one_band_drc_is_applied_and_reported_in_stream_info() {
        let mut decoder = AacLcDecoder::new(4, 2).unwrap();
        assert_eq!(decoder.stream_info().output_loudness, -1);
        decoder.legacy_drc_payload = Some(Mpeg4DrcPayload {
            pce_instance_tag: None,
            excluded_channels: vec![false, true],
            interpolation_scheme: 0,
            band_top: vec![255],
            program_reference_level: Some(96),
            dynamic_range: vec![0x98],
        });
        decoder.set_drc_attenuation_factor(127);
        let mut floating = vec![vec![0.5], vec![0.5]];
        decoder.apply_configured_drc_f32(&mut floating).unwrap();
        assert!((floating[0][0] - 0.25).abs() < 1e-6);
        assert_eq!(floating[1][0], 0.5);

        let mut fixed = vec![vec![10_000], vec![10_000]];
        decoder.apply_configured_drc_i16(&mut fixed).unwrap();
        assert_eq!(fixed, vec![vec![5_000], vec![10_000]]);
        let info = decoder.stream_info();
        assert_ne!(info.flags & STREAM_FLAG_DRC_PRESENT, 0);
        assert_eq!(info.drc_program_reference_level, 96);
        assert_eq!(info.output_loudness, 96);
        decoder.set_drc_reference_level(None);
        assert_eq!(decoder.stream_info().output_loudness, 96);

        decoder.set_metadata_expiry_ms(1); // ceil(44.1 / 1024) = one frame
        decoder.age_legacy_drc();
        assert!(decoder.legacy_drc_payload.is_some());
        decoder.age_legacy_drc();
        assert!(decoder.legacy_drc_payload.is_none());

        let mut raw = BitWriter::new();
        write_zero_sce_payload_bits(&mut raw, false);
        raw.write(ElementId::Fill.bits() as u32, 3);
        raw.write(3, 4); // three-byte EXT_DYNAMIC_RANGE payload
        raw.write(0x0b, 4);
        raw.write_bool(false); // PCE tag absent
        raw.write_bool(false); // exclusions absent
        raw.write_bool(false); // one band
        raw.write_bool(true); // program reference present
        raw.write(88, 7);
        raw.write_bool(false);
        raw.write(0x8c, 8); // -3 dB control
        let mut parsed = AacLcDecoder::new(4, 1).unwrap();
        parsed
            .decode_raw_data_block_multichannel_f32(&raw.finish())
            .unwrap();
        assert_eq!(parsed.stream_info().drc_program_reference_level, 88);
        assert_eq!(parsed.stream_info().output_loudness, 96);
        parsed.set_drc_reference_level(None);
        assert_eq!(parsed.stream_info().output_loudness, 88);
        assert!(parsed.legacy_drc_payload.is_some());
    }

    #[test]
    fn stream_info_reports_selected_mpeg_d_output_loudness() {
        use crate::drc::{
            ChannelLayout, DrcInstruction, LoudnessInfo, LoudnessMeasurement, LoudnessMethod,
        };

        let instruction = DrcInstruction {
            drc_set_id: 3,
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
            gain_set_index_per_channel: vec![-1],
            gain_modifications: Vec::new(),
            gain_modifications_per_band: Vec::new(),
            ducking_modifications: Vec::new(),
        };
        let mut decoder = AacLcDecoder::new(4, 1).unwrap();
        decoder.configure_drc(
            UniDrcConfig {
                sample_rate: Some(44_100),
                channel_layout: ChannelLayout {
                    base_channel_count: 1,
                    defined_layout: None,
                    speaker_positions: Vec::new(),
                },
                downmix_instructions: Vec::new(),
                coefficients: Vec::new(),
                instructions: vec![instruction],
                extension_present: false,
                extensions: Vec::new(),
                bits_read: 0,
            },
            DrcSelectionRequest::default(),
        );
        decoder.update_drc_gain(UniDrcGain {
            sequences: Vec::new(),
            extension_present: false,
            extensions: Vec::new(),
            bits_read: 0,
        });
        let info = |value| LoudnessInfo {
            drc_set_id: 3,
            downmix_id: 0,
            sample_peak_level: None,
            true_peak_level: None,
            true_peak_measurement_system: None,
            true_peak_reliability: None,
            measurements: vec![LoudnessMeasurement {
                method: LoudnessMethod::ProgramLoudness,
                value,
                measurement_system: 1,
                reliability: 3,
            }],
        };
        decoder.update_drc_loudness_info(LoudnessInfoSet {
            album: vec![info(-24.0)],
            track: vec![info(-26.0)],
            extension_present: false,
            bits_read: 0,
        });
        assert_eq!(decoder.stream_info().output_loudness, 104);

        decoder.set_drc_reference_level(Some(92)); // normalize to -23 dB
        assert_eq!(decoder.stream_info().output_loudness, 92);
        decoder.set_drc_reference_level(None);
        decoder.set_uni_drc_album_mode(true);
        assert_eq!(decoder.stream_info().output_loudness, 96);

        decoder.clear_drc_loudness_info();
        assert_eq!(decoder.stream_info().output_loudness, -1);
    }

    #[test]
    fn legacy_multiband_drc_maps_four_line_band_top_units_in_both_formats() {
        let mut floating = InverseQuantizedSpectrum {
            windows: vec![vec![1.0; 12]],
        };
        apply_legacy_band_gains_f32(&mut floating, &[0, 1], &[2.0, 0.5]);
        assert_eq!(&floating.windows[0][..4], &[2.0; 4]);
        assert_eq!(&floating.windows[0][4..8], &[0.5; 4]);
        assert_eq!(&floating.windows[0][8..], &[1.0; 4]);

        let mut fixed = FixedInverseQuantizedSpectrum {
            windows: vec![vec![100; 12]],
            window_exponents: vec![0],
        };
        apply_legacy_band_gains_fixed(&mut fixed, &[0, 1], &[2.0, 0.5]);
        assert_eq!(fixed.window_exponents, vec![1]);
        assert_eq!(&fixed.windows[0][..4], &[100; 4]);
        assert_eq!(&fixed.windows[0][4..8], &[25; 4]);
        assert_eq!(&fixed.windows[0][8..], &[50; 4]);
    }

    #[test]
    fn legacy_multiband_drc_is_applied_to_concealed_core_spectra() {
        let payload = Mpeg4DrcPayload {
            pce_instance_tag: None,
            excluded_channels: vec![false],
            interpolation_scheme: 0,
            band_top: vec![31, 255],
            program_reference_level: None,
            dynamic_range: vec![0x98, 0],
        };
        let ics = synthetic_long_ics();

        let mut floating = InverseQuantizedSpectrum {
            windows: vec![vec![0.0; 1024]],
        };
        floating.windows[0][0] = 0.25;
        let mut baseline = AacLcDecoder::new(4, 1).unwrap();
        baseline.f32_concealment_spectra = vec![(floating, ics.clone())];
        let mut controlled = baseline.clone();
        controlled.legacy_drc_payload = Some(payload.clone());
        controlled.set_drc_attenuation_factor(127);
        let baseline_pcm = baseline.conceal_f32_interleaved().unwrap();
        let controlled_pcm = controlled.conceal_f32_interleaved().unwrap();
        let baseline_energy = baseline_pcm
            .iter()
            .map(|sample| sample * sample)
            .sum::<f32>();
        let controlled_energy = controlled_pcm
            .iter()
            .map(|sample| sample * sample)
            .sum::<f32>();
        assert!(((controlled_energy / baseline_energy).sqrt() - 0.5).abs() < 1.0e-5);
        assert!(controlled.legacy_drc_control_applied);

        let mut fixed = FixedInverseQuantizedSpectrum {
            windows: vec![vec![0; 1024]],
            window_exponents: vec![0],
        };
        fixed.windows[0][0] = 0x2000_0000;
        let mut baseline = AacLcDecoder::new(4, 1).unwrap();
        baseline.fixed_concealment_spectra = vec![(fixed, ics)];
        let mut controlled = baseline.clone();
        controlled.legacy_drc_payload = Some(payload);
        controlled.set_drc_attenuation_factor(127);
        let baseline_pcm = baseline.conceal_fixed_interleaved_i16().unwrap();
        let controlled_pcm = controlled.conceal_fixed_interleaved_i16().unwrap();
        let baseline_energy = baseline_pcm
            .iter()
            .map(|&sample| (sample as f64).powi(2))
            .sum::<f64>();
        let controlled_energy = controlled_pcm
            .iter()
            .map(|&sample| (sample as f64).powi(2))
            .sum::<f64>();
        assert!(((controlled_energy / baseline_energy).sqrt() - 0.5).abs() < 0.01);
        assert!(controlled.legacy_drc_control_applied);
    }

    #[test]
    fn legacy_multiband_drc_is_interpolated_in_the_sbr_qmf_domain() {
        let controlled = LegacyQmfDrcFrame {
            band_top: vec![31, 255],
            gains: vec![0.5, 1.0],
            interpolation_scheme: 0,
            window_sequence: WindowSequence::OnlyLong,
        };
        let mut state = LegacyQmfDrcState::default();
        let make_slots = || {
            vec![
                QmfSlot {
                    real: vec![1.0; 64],
                    imaginary: vec![1.0; 64],
                };
                32
            ]
        };

        let mut transition = make_slots();
        assert!(apply_legacy_qmf_drc(
            &mut state,
            &mut transition,
            Some(controlled.clone()),
            1024,
        ));
        assert!(transition.iter().any(|slot| slot.real[0] > 0.5));
        assert!(transition.iter().any(|slot| slot.real[0] < 1.0));
        assert!(transition
            .iter()
            .all(|slot| (slot.real[4] - 1.0).abs() < 1.0e-12));

        let mut overlap = make_slots();
        assert!(apply_legacy_qmf_drc(
            &mut state,
            &mut overlap,
            Some(controlled.clone()),
            1024,
        ));
        assert!(overlap.iter().any(|slot| slot.real[0] > 0.5));
        assert!(overlap
            .iter()
            .any(|slot| (slot.real[0] - 0.5).abs() < 1.0e-12));

        let mut settled = make_slots();
        assert!(apply_legacy_qmf_drc(
            &mut state,
            &mut settled,
            Some(controlled),
            1024,
        ));
        assert!(settled
            .iter()
            .all(|slot| (slot.real[0] - 0.5).abs() < 1.0e-12));
        assert!(settled
            .iter()
            .all(|slot| (slot.imaginary[4] - 1.0).abs() < 1.0e-12));
    }

    #[test]
    fn dvb_ancillary_drc_is_parsed_without_ancillary_capture_and_selects_heavy_gain() {
        let mut no_metadata = AacLcDecoder::new(4, 1).unwrap();
        no_metadata.set_drc_default_presentation_mode(1);
        assert_eq!(no_metadata.stream_info().drc_presentation_mode, -1);

        let mut raw = BitWriter::new();
        write_zero_sce_payload_bits(&mut raw, false);
        raw.write(ElementId::DataStream.bits() as u32, 3);
        raw.write(0, 4); // element instance tag
        raw.write_bool(true); // byte alignment flag
        raw.write(6, 8);
        raw.byte_align();
        for byte in [0xbc, 0xc2, 0x14, 0xdd, 0x01, 0x90] {
            raw.write(byte, 8);
        }
        raw.write(ElementId::End.bits() as u32, 3);

        let mut decoder = AacLcDecoder::new(4, 1).unwrap();
        decoder.set_drc_heavy_compression(true);
        decoder
            .decode_raw_data_block_multichannel_f32(&raw.finish())
            .unwrap();
        let payload = decoder.legacy_dvb_drc_payload.unwrap();
        assert_eq!(payload.presentation_mode, 0);
        assert!((decoder.legacy_one_band_control_gain(0).unwrap() - 0.5).abs() < 0.001);
        assert!(decoder.ancillary_data().is_empty());
        assert_ne!(decoder.stream_info().flags & STREAM_FLAG_DRC_PRESENT, 0);
        assert_eq!(decoder.stream_info().drc_presentation_mode, 0);
        let downmix = decoder.legacy_downmix_metadata().unwrap();
        assert!(downmix.pseudo_surround);
        assert_eq!(downmix.center_mix_level_index, Some(5));
        assert_eq!(downmix.surround_mix_level_index, Some(5));

        decoder.set_drc_heavy_compression(false);
        assert_eq!(decoder.legacy_one_band_control_gain(0), None);
        decoder.set_drc_default_presentation_mode(1);
        assert!(decoder.legacy_one_band_control_gain(0).is_some());
        decoder.set_drc_reference_level(None);
        assert_eq!(decoder.legacy_one_band_control_gain(0), None);
        decoder.set_metadata_expiry_ms(1);
        decoder.age_legacy_drc();
        decoder.age_legacy_drc();
        assert_eq!(decoder.legacy_downmix_metadata(), None);
        assert_eq!(decoder.stream_info().drc_presentation_mode, 0);
    }

    #[test]
    fn legacy_presentation_parameter_handling_matches_downmix_headroom_rules() {
        let mut decoder = AacLcDecoder::new(4, 6).unwrap();
        decoder.set_drc_default_presentation_mode(0);
        decoder.set_drc_reference_level(Some(96));
        decoder.set_drc_encoder_target_level(127);
        decoder.set_legacy_drc_output_channels(2);
        let (attenuation, heavy) = decoder.effective_legacy_parameters();
        assert_eq!(attenuation, 1.0);
        assert!(!heavy); // 5.1 -> stereo is just below the 10 dB threshold

        decoder.set_legacy_drc_output_channels(1);
        let (attenuation, heavy) = decoder.effective_legacy_parameters();
        assert_eq!(attenuation, 1.0);
        assert!(heavy);

        decoder.set_legacy_drc_output_channels(6);
        decoder.legacy_drc_payload = Some(Mpeg4DrcPayload {
            pce_instance_tag: None,
            excluded_channels: Vec::new(),
            interpolation_scheme: 0,
            band_top: vec![255],
            program_reference_level: Some(100),
            dynamic_range: vec![0x98],
        });
        decoder.set_drc_encoder_target_level(80);
        let (attenuation, heavy) = decoder.effective_legacy_parameters();
        assert!((attenuation - 25.0 / 127.0).abs() < 1.0e-6);
        assert!(!heavy);

        decoder.set_drc_default_presentation_mode(2);
        decoder.set_drc_attenuation_factor(0);
        let (attenuation, heavy) = decoder.effective_legacy_parameters();
        assert_eq!(attenuation, 1.0);
        assert!(!heavy);
    }

    #[test]
    fn ga_constructor_rejects_each_independent_configuration_limit() {
        assert_eq!(
            AacLcDecoder::new_ga_with_frame_length(2, 4, 1, 512).unwrap_err(),
            DecodeError::UnsupportedFrameLength(512)
        );
        assert_eq!(
            AacLcDecoder::new_ga_with_frame_length(2, 13, 1, 1024).unwrap_err(),
            DecodeError::UnsupportedSamplingFrequencyIndex(13)
        );
        assert_eq!(
            AacLcDecoder::new_ga_with_frame_length(2, 4, 8, 1024).unwrap_err(),
            DecodeError::UnsupportedChannelConfiguration(8)
        );

        let mut decoder = AacLcDecoder::new(4, 1).unwrap();
        decoder.ensure_coupling_filterbanks(2).unwrap();

        let mut invalid_er = AacLcDecoder::new_ga(17, 4, 1).unwrap();
        invalid_er.error_protection_config = None;
        assert_eq!(
            invalid_er.decode_er_aac_lc_multichannel_f32_from_reader(&mut BitReader::new(&[])),
            Err(DecodeError::ErrorResilienceUnsupported)
        );
        assert_eq!(
            invalid_er
                .decode_er_aac_lc_multichannel_fixed_i16_from_reader(&mut BitReader::new(&[])),
            Err(DecodeError::ErrorResilienceUnsupported)
        );

        let mut eld = AacLcDecoder::new_ga_with_frame_length(39, 4, 1, 512).unwrap();
        eld.ensure_channel_filterbanks(3).unwrap();
        eld.ensure_fixed_channel_filterbanks(3).unwrap();
        assert_eq!(eld.channel_filterbanks.len(), 3);
        assert_eq!(eld.eld_channel_filterbanks.len(), 3);
        assert_eq!(eld.fixed_channel_filterbanks.len(), 3);
        assert_eq!(eld.eld_fixed_channel_filterbanks.len(), 3);
        assert_eq!(decoder.coupling_filterbanks.len(), 2);
        decoder.ensure_coupling_filterbanks(1).unwrap();
        assert_eq!(decoder.coupling_filterbanks.len(), 2);
    }

    #[test]
    fn stateful_decoder_assembles_multiple_raw_audio_elements() {
        let mut writer = BitWriter::new();
        write_zero_sce_payload_bits_with_tag(&mut writer, 0, false);
        write_zero_sce_payload_bits_with_tag(&mut writer, 1, false);
        writer.write(ElementId::End.bits() as u32, 3);
        let payload = writer.finish();

        let mut decoder = AacLcDecoder::new(4, 2).unwrap();
        let decoded = decoder
            .decode_raw_data_block_multichannel_f32(&payload)
            .unwrap();

        assert_eq!(decoded.channels(), 2);
        assert_eq!(
            decoded.labels(),
            &[ChannelLabel::FrontLeft, ChannelLabel::FrontRight]
        );
        assert_eq!(decoded.samples_per_channel(), 1024);
        assert_eq!(decoded.interleaved_f32().len(), 2048);
        assert!(decoded.interleaved_i16().iter().all(|sample| *sample == 0));
    }

    #[test]
    fn stateful_decoder_assembles_multiple_raw_audio_elements_fixed_i16() {
        let mut writer = BitWriter::new();
        write_zero_sce_payload_bits_with_tag(&mut writer, 0, false);
        write_zero_sce_payload_bits_with_tag(&mut writer, 1, false);
        writer.write(ElementId::End.bits() as u32, 3);
        let payload = writer.finish();

        let mut decoder = AacLcDecoder::new(4, 2).unwrap();
        let pcm = decoder
            .decode_raw_data_block_multichannel_fixed_interleaved_i16(&payload)
            .unwrap();

        assert_eq!(pcm.len(), 2048);
        assert!(pcm.iter().all(|sample| *sample == 0));
    }

    #[test]
    fn multichannel_decoder_stages_and_applies_frequency_cce() {
        let mut writer = BitWriter::new();
        write_zero_sce_payload_bits_with_tag(&mut writer, 0, false);
        writer.write(ElementId::CouplingChannel.bits() as u32, 3);
        writer.write(0, 4); // cce element_instance_tag
        writer.write_bool(false); // ind_sw_cce_flag
        writer.write(0, 3); // one coupled element
        writer.write_bool(false); // target SCE
        writer.write(0, 4); // target tag
        writer.write_bool(true); // cc_domain frequency
        writer.write_bool(false); // gain_element_sign
        writer.write(0, 2); // gain_element_scale
        write_zero_independent_channel_stream(&mut writer, 1);
        writer.write(ElementId::End.bits() as u32, 3);
        let payload = writer.finish();

        let mut decoder = AacLcDecoder::new(4, 1).unwrap();
        let decoded = decoder
            .decode_raw_data_block_multichannel_f32(&payload)
            .unwrap();

        assert_eq!(decoded.channels(), 1);
        assert_eq!(decoded.samples_per_channel(), 1024);
        assert!(decoded.channels[0].iter().all(|sample| *sample == 0.0));
    }

    #[test]
    fn multichannel_fixed_decoder_stages_and_applies_frequency_cce() {
        let mut writer = BitWriter::new();
        write_zero_sce_payload_bits_with_tag(&mut writer, 0, false);
        writer.write(ElementId::CouplingChannel.bits() as u32, 3);
        writer.write(0, 4); // cce element_instance_tag
        writer.write_bool(false); // ind_sw_cce_flag
        writer.write(0, 3); // one coupled element
        writer.write_bool(false); // target SCE
        writer.write(0, 4); // target tag
        writer.write_bool(true); // cc_domain frequency
        writer.write_bool(false); // gain_element_sign
        writer.write(0, 2); // gain_element_scale
        write_zero_independent_channel_stream(&mut writer, 1);
        writer.write(ElementId::End.bits() as u32, 3);
        let payload = writer.finish();

        let mut decoder = AacLcDecoder::new(4, 1).unwrap();
        let pcm = decoder
            .decode_raw_data_block_multichannel_fixed_interleaved_i16(&payload)
            .unwrap();

        assert_eq!(pcm.len(), 1024);
        assert!(pcm.iter().all(|sample| *sample == 0));
    }

    #[test]
    fn multiple_frequency_cces_accumulate_on_one_sce_in_f32_and_fixed_paths() {
        let coupling_template = nonzero_spectral_sce_payload();
        let mut template_reader = BitReader::new(&coupling_template);
        let mut template_pns = PnsRandomState::new(1);
        let coupling_template_bits = decode_aac_lc_single_channel_spectra_from_reader(
            &mut template_reader,
            4,
            &mut template_pns,
        )
        .unwrap()
        .bits_read;
        let make_payload = |cce_count: usize| {
            let mut writer = BitWriter::new();
            write_zero_sce_payload_bits_with_tag(&mut writer, 0, false);
            for cce_tag in 0..cce_count {
                writer.write(ElementId::CouplingChannel.bits() as u32, 3);
                writer.write(cce_tag as u32, 4);
                writer.write_bool(false); // ind_sw_cce_flag
                writer.write(0, 3); // one coupled element
                writer.write_bool(false); // target SCE
                writer.write(0, 4); // target tag
                writer.write_bool(true); // frequency domain
                writer.write_bool(false); // unsigned gain
                writer.write(0, 2); // 1/8-octave exponent step
                                    // Reuse the exact independently-decoded channel-stream bits,
                                    // excluding the SCE id/tag prefix and byte padding.
                writer.write(80, 8); // lower coupling gain avoids Q31 saturation
                for bit in 15..coupling_template_bits {
                    let value = coupling_template[bit / 8] & (1 << (7 - bit % 8)) != 0;
                    writer.write_bool(value);
                }
            }
            writer.write(ElementId::End.bits() as u32, 3);
            writer.finish()
        };
        let one_payload = make_payload(1);
        let two_payload = make_payload(2);

        let mut one_decoder = AacLcDecoder::new(4, 1).unwrap();
        let one = one_decoder
            .decode_raw_data_block_multichannel_f32(&one_payload)
            .unwrap()
            .channels
            .remove(0);
        let mut two_decoder = AacLcDecoder::new(4, 1).unwrap();
        let two = two_decoder
            .decode_raw_data_block_multichannel_f32(&two_payload)
            .unwrap()
            .channels
            .remove(0);
        assert!(one.iter().any(|sample| *sample != 0.0));
        let dot = one.iter().zip(&two).map(|(a, b)| a * b).sum::<f32>();
        let one_energy = one.iter().map(|sample| sample * sample).sum::<f32>();
        let two_energy = two.iter().map(|sample| sample * sample).sum::<f32>();
        let correlation = dot / (one_energy * two_energy).sqrt();
        let rms_ratio = (two_energy / one_energy).sqrt();
        assert!(correlation > 0.999_999 && (1.999..=2.001).contains(&rms_ratio));

        let mut one_decoder = AacLcDecoder::new(4, 1).unwrap();
        let one = one_decoder
            .decode_raw_data_block_multichannel_fixed_interleaved_i16(&one_payload)
            .unwrap();
        let mut two_decoder = AacLcDecoder::new(4, 1).unwrap();
        let two = two_decoder
            .decode_raw_data_block_multichannel_fixed_interleaved_i16(&two_payload)
            .unwrap();
        assert!(one.iter().any(|sample| *sample != 0));
        let dot = one
            .iter()
            .zip(&two)
            .map(|(&a, &b)| a as f64 * b as f64)
            .sum::<f64>();
        let one_energy = one
            .iter()
            .map(|&sample| (sample as f64).powi(2))
            .sum::<f64>();
        let two_energy = two
            .iter()
            .map(|&sample| (sample as f64).powi(2))
            .sum::<f64>();
        let correlation = dot / (one_energy * two_energy).sqrt();
        let rms_ratio = (two_energy / one_energy).sqrt();
        assert!(correlation > 0.999 && (1.95..=2.05).contains(&rms_ratio));
    }

    #[test]
    fn pce_routes_nonzero_frequency_cce_by_element_tag_in_f32_and_fixed_paths() {
        let pce = ProgramConfig {
            front: vec![ProgramElement {
                is_cpe: false,
                tag_select: 5,
            }],
            num_channels: 1,
            num_effective_channels: 1,
            ..ProgramConfig::default()
        };
        let coupling_template = nonzero_spectral_sce_payload();
        let mut template_reader = BitReader::new(&coupling_template);
        let mut template_pns = PnsRandomState::new(1);
        let coupling_template_bits = decode_aac_lc_single_channel_spectra_from_reader(
            &mut template_reader,
            4,
            &mut template_pns,
        )
        .unwrap()
        .bits_read;
        let mut writer = BitWriter::new();
        writer.write(ElementId::ProgramConfig.bits() as u32, 3);
        pce.write_to_writer(&mut writer).unwrap();
        write_zero_sce_payload_bits_with_tag(&mut writer, 5, false);
        writer.write(ElementId::CouplingChannel.bits() as u32, 3);
        writer.write(0, 4); // CCE tag
        writer.write_bool(false); // ind_sw_cce_flag
        writer.write(0, 3); // one target
        writer.write_bool(false); // target is SCE
        writer.write(5, 4); // target the PCE's SCE tag
        writer.write_bool(true); // frequency domain
        writer.write_bool(false); // unsigned gain
        writer.write(0, 2); // gain scale
        writer.write(80, 8); // coupling global gain
        for bit in 15..coupling_template_bits {
            let value = coupling_template[bit / 8] & (1 << (7 - bit % 8)) != 0;
            writer.write_bool(value);
        }
        let payload = writer.finish();

        let mut decoder = AacLcDecoder::new(4, 0).unwrap();
        let decoded = decoder
            .decode_raw_data_block_multichannel_f32_strict(&payload)
            .unwrap();
        assert_eq!(decoded.labels(), &[ChannelLabel::FrontCenter]);
        assert!(decoded.channels[0].iter().any(|sample| *sample != 0.0));

        let mut decoder = AacLcDecoder::new(4, 0).unwrap();
        let pcm = decoder
            .decode_raw_data_block_multichannel_fixed_interleaved_i16_strict(&payload)
            .unwrap();
        assert_eq!(pcm.len(), 1024);
        assert!(pcm.iter().any(|sample| *sample != 0));
    }

    #[test]
    fn legacy_decoder_uses_staging_path_for_trailing_frequency_cce() {
        let mut writer = BitWriter::new();
        write_zero_sce_payload_bits_with_tag(&mut writer, 0, false);
        writer.write(ElementId::CouplingChannel.bits() as u32, 3);
        writer.write(0, 4); // cce element_instance_tag
        writer.write_bool(false); // ind_sw_cce_flag
        writer.write(0, 3); // one coupled element
        writer.write_bool(false); // target SCE
        writer.write(0, 4); // target tag
        writer.write_bool(true); // frequency domain
        writer.write_bool(false); // gain_element_sign
        writer.write(0, 2); // gain_element_scale
        write_zero_independent_channel_stream(&mut writer, 1);
        writer.write(ElementId::End.bits() as u32, 3);
        let payload = writer.finish();

        let mut decoder = AacLcDecoder::new(4, 0).unwrap();
        let decoded = decoder.decode_raw_data_block_f32(&payload).unwrap();

        assert!(matches!(decoded, DecodedAacLcFrame::Mono(_)));
        assert_eq!(decoded.samples_per_channel(), 1024);
    }

    #[test]
    fn multichannel_decoder_integrates_time_domain_cce() {
        let mut writer = BitWriter::new();
        write_zero_sce_payload_bits_with_tag(&mut writer, 0, false);
        writer.write(ElementId::CouplingChannel.bits() as u32, 3);
        writer.write(0, 4); // cce element_instance_tag
        writer.write_bool(true); // independently switched => after-IMDCT coupling
        writer.write(0, 3); // one coupled element
        writer.write_bool(false); // target SCE
        writer.write(0, 4); // target tag
        writer.write_bool(false); // time domain
        writer.write_bool(false); // gain_element_sign
        writer.write(0, 2); // gain_element_scale
        write_zero_independent_channel_stream(&mut writer, 1);
        writer.write(ElementId::End.bits() as u32, 3);
        let payload = writer.finish();

        let mut decoder = AacLcDecoder::new(4, 1).unwrap();
        let decoded = decoder
            .decode_raw_data_block_multichannel_f32(&payload)
            .unwrap();

        assert_eq!(decoded.channels(), 1);
        assert_eq!(decoded.samples_per_channel(), 1024);
        assert!(decoded.channels[0].iter().all(|sample| *sample == 0.0));

        let mut decoder = AacLcDecoder::new(4, 1).unwrap();
        decoder
            .coupling_filterbanks
            .push(LongBlockFilterbank::new(960).unwrap());
        assert!(decoder
            .decode_raw_data_block_multichannel_f32(&payload)
            .is_err());
    }

    #[test]
    fn multichannel_fixed_decoder_integrates_time_domain_cce() {
        let mut writer = BitWriter::new();
        write_zero_sce_payload_bits_with_tag(&mut writer, 0, false);
        writer.write(ElementId::CouplingChannel.bits() as u32, 3);
        writer.write(0, 4); // cce element_instance_tag
        writer.write_bool(true); // independently switched => after-IMDCT coupling
        writer.write(0, 3); // one coupled element
        writer.write_bool(false); // target SCE
        writer.write(0, 4); // target tag
        writer.write_bool(false); // time domain
        writer.write_bool(false); // gain_element_sign
        writer.write(0, 2); // gain_element_scale
        write_zero_independent_channel_stream(&mut writer, 1);
        writer.write(ElementId::End.bits() as u32, 3);
        let payload = writer.finish();

        let mut decoder = AacLcDecoder::new(4, 1).unwrap();
        let pcm = decoder
            .decode_raw_data_block_multichannel_fixed_interleaved_i16(&payload)
            .unwrap();

        assert_eq!(pcm.len(), 1024);
        assert!(pcm.iter().all(|sample| *sample == 0));

        let mut decoder = AacLcDecoder::new(4, 1).unwrap();
        decoder
            .fixed_coupling_filterbanks
            .push(FixedLongBlockFilterbank::new(960).unwrap());
        assert!(decoder
            .decode_raw_data_block_multichannel_fixed_interleaved_i16(&payload)
            .is_err());
    }

    #[test]
    fn maps_channel_configuration_to_labels() {
        let expected_labels: &[&[ChannelLabel]] = &[
            &[ChannelLabel::FrontCenter],
            &[ChannelLabel::FrontLeft, ChannelLabel::FrontRight],
            &[
                ChannelLabel::FrontCenter,
                ChannelLabel::FrontLeft,
                ChannelLabel::FrontRight,
            ],
            &[
                ChannelLabel::FrontCenter,
                ChannelLabel::FrontLeft,
                ChannelLabel::FrontRight,
                ChannelLabel::BackCenter,
            ],
            &[
                ChannelLabel::FrontCenter,
                ChannelLabel::FrontLeft,
                ChannelLabel::FrontRight,
                ChannelLabel::BackLeft,
                ChannelLabel::BackRight,
            ],
            &[
                ChannelLabel::FrontCenter,
                ChannelLabel::FrontLeft,
                ChannelLabel::FrontRight,
                ChannelLabel::BackLeft,
                ChannelLabel::BackRight,
                ChannelLabel::Lfe,
            ],
            &[
                ChannelLabel::FrontCenter,
                ChannelLabel::FrontLeftCenter,
                ChannelLabel::FrontRightCenter,
                ChannelLabel::FrontLeft,
                ChannelLabel::FrontRight,
                ChannelLabel::BackLeft,
                ChannelLabel::BackRight,
                ChannelLabel::Lfe,
            ],
        ];
        for (index, labels) in expected_labels.iter().enumerate() {
            let configuration = index as u8 + 1;
            assert_eq!(
                expected_channels_for_config(configuration),
                Some(labels.len())
            );
            assert_eq!(channel_labels_for_config(configuration), Some(*labels));
        }

        let expected_er_elements: &[&[ElementId]] = &[
            &[ElementId::SingleChannel],
            &[ElementId::ChannelPair],
            &[ElementId::SingleChannel, ElementId::ChannelPair],
            &[
                ElementId::SingleChannel,
                ElementId::ChannelPair,
                ElementId::SingleChannel,
            ],
            &[
                ElementId::SingleChannel,
                ElementId::ChannelPair,
                ElementId::ChannelPair,
            ],
            &[
                ElementId::SingleChannel,
                ElementId::ChannelPair,
                ElementId::ChannelPair,
                ElementId::Lfe,
            ],
            &[
                ElementId::SingleChannel,
                ElementId::ChannelPair,
                ElementId::ChannelPair,
                ElementId::ChannelPair,
                ElementId::Lfe,
            ],
        ];
        for (index, elements) in expected_er_elements.iter().enumerate() {
            assert_eq!(er_channel_elements(index as u8 + 1), Some(*elements));
        }

        assert_eq!(
            channel_labels_for_config(6).unwrap(),
            &[
                ChannelLabel::FrontCenter,
                ChannelLabel::FrontLeft,
                ChannelLabel::FrontRight,
                ChannelLabel::BackLeft,
                ChannelLabel::BackRight,
                ChannelLabel::Lfe,
            ]
        );
        assert_eq!(expected_channels_for_config(7), Some(8));
        assert_eq!(expected_channels_for_config(8), None);
        assert_eq!(channel_labels_for_config(0), None);
        assert_eq!(channel_labels_for_config(8), None);
        assert_eq!(er_channel_elements(0), None);
        assert_eq!(er_channel_elements(8), None);
    }

    #[test]
    fn maps_program_config_to_channel_labels() {
        let pce = ProgramConfig {
            front: vec![
                ProgramElement {
                    is_cpe: false,
                    tag_select: 0,
                },
                ProgramElement {
                    is_cpe: true,
                    tag_select: 1,
                },
                ProgramElement {
                    is_cpe: false,
                    tag_select: 6,
                },
            ],
            side: vec![
                ProgramElement {
                    is_cpe: true,
                    tag_select: 2,
                },
                ProgramElement {
                    is_cpe: false,
                    tag_select: 7,
                },
            ],
            back: vec![
                ProgramElement {
                    is_cpe: false,
                    tag_select: 3,
                },
                ProgramElement {
                    is_cpe: true,
                    tag_select: 4,
                },
            ],
            lfe: vec![0],
            num_channels: 12,
            num_effective_channels: 11,
            ..ProgramConfig::default()
        };

        assert_eq!(
            program_config_channel_labels(&pce),
            vec![
                ChannelLabel::FrontCenter,
                ChannelLabel::FrontLeft,
                ChannelLabel::FrontRight,
                ChannelLabel::Unknown(3),
                ChannelLabel::SideLeft,
                ChannelLabel::SideRight,
                ChannelLabel::Unknown(6),
                ChannelLabel::BackCenter,
                ChannelLabel::BackLeft,
                ChannelLabel::BackRight,
                ChannelLabel::Lfe,
            ]
        );
    }

    #[test]
    fn maps_each_program_config_element_and_unknown_fallback() {
        let pce = ProgramConfig {
            front: vec![
                ProgramElement {
                    is_cpe: false,
                    tag_select: 0,
                },
                ProgramElement {
                    is_cpe: false,
                    tag_select: 6,
                },
                ProgramElement {
                    is_cpe: true,
                    tag_select: 1,
                },
            ],
            side: vec![
                ProgramElement {
                    is_cpe: false,
                    tag_select: 4,
                },
                ProgramElement {
                    is_cpe: true,
                    tag_select: 2,
                },
            ],
            back: vec![
                ProgramElement {
                    is_cpe: false,
                    tag_select: 5,
                },
                ProgramElement {
                    is_cpe: true,
                    tag_select: 3,
                },
            ],
            lfe: vec![0],
            ..ProgramConfig::default()
        };

        assert_eq!(
            program_config_labels_for_element(&pce, ElementId::SingleChannel, 0, 9),
            [ChannelLabel::FrontCenter]
        );
        assert_eq!(
            program_config_labels_for_element(&pce, ElementId::SingleChannel, 6, 9),
            [ChannelLabel::Unknown(9)]
        );
        assert_eq!(
            program_config_labels_for_element(&pce, ElementId::SingleChannel, 4, 9),
            [ChannelLabel::Unknown(9)]
        );
        assert_eq!(
            program_config_labels_for_element(&pce, ElementId::SingleChannel, 5, 9),
            [ChannelLabel::BackCenter]
        );
        assert_eq!(
            program_config_labels_for_element(&pce, ElementId::ChannelPair, 1, 9),
            [ChannelLabel::FrontLeft, ChannelLabel::FrontRight]
        );
        assert_eq!(
            program_config_labels_for_element(&pce, ElementId::ChannelPair, 2, 9),
            [ChannelLabel::SideLeft, ChannelLabel::SideRight]
        );
        assert_eq!(
            program_config_labels_for_element(&pce, ElementId::ChannelPair, 3, 9),
            [ChannelLabel::BackLeft, ChannelLabel::BackRight]
        );
        assert_eq!(
            program_config_labels_for_element(&pce, ElementId::Lfe, 0, 9),
            [ChannelLabel::Lfe]
        );
        assert_eq!(
            program_config_labels_for_element(&pce, ElementId::Lfe, 7, 9),
            [ChannelLabel::Unknown(9)]
        );
        assert_eq!(
            program_config_labels_for_element(&pce, ElementId::SingleChannel, 7, 9),
            [ChannelLabel::Unknown(9)]
        );
        assert_eq!(
            program_config_labels_for_element(&pce, ElementId::ChannelPair, 7, 9),
            [ChannelLabel::Unknown(9), ChannelLabel::Unknown(10)]
        );
        assert!(program_config_labels_for_element(&pce, ElementId::Fill, 0, 9).is_empty());
    }

    #[test]
    fn stateful_decoder_uses_pce_labels_for_channel_config_zero() {
        let pce = ProgramConfig {
            front: vec![
                ProgramElement {
                    is_cpe: false,
                    tag_select: 0,
                },
                ProgramElement {
                    is_cpe: false,
                    tag_select: 1,
                },
            ],
            num_channels: 2,
            num_effective_channels: 2,
            ..ProgramConfig::default()
        };
        let mut writer = BitWriter::new();
        writer.write(ElementId::ProgramConfig.bits() as u32, 3);
        pce.write_to_writer(&mut writer).unwrap();
        write_zero_sce_payload_bits_with_tag(&mut writer, 0, false);
        write_zero_sce_payload_bits_with_tag(&mut writer, 1, false);
        writer.write(ElementId::End.bits() as u32, 3);
        let payload = writer.finish();

        let mut decoder = AacLcDecoder::new(4, 0).unwrap();
        let decoded = decoder
            .decode_raw_data_block_multichannel_f32(&payload)
            .unwrap();

        assert_eq!(decoded.channels(), 2);
        assert_eq!(
            decoded.labels(),
            &[ChannelLabel::FrontCenter, ChannelLabel::Unknown(1)]
        );
    }

    #[test]
    fn stateful_decoder_uses_pce_labels_for_channel_pair() {
        let pce = ProgramConfig {
            front: vec![ProgramElement {
                is_cpe: true,
                tag_select: 0,
            }],
            num_channels: 2,
            num_effective_channels: 2,
            ..ProgramConfig::default()
        };
        let mut writer = BitWriter::new();
        writer.write(ElementId::ProgramConfig.bits() as u32, 3);
        pce.write_to_writer(&mut writer).unwrap();
        writer.write(ElementId::ChannelPair.bits() as u32, 3);
        writer.write(0, 4); // element_instance_tag
        writer.write_bool(true); // common window
        write_shared_long_ics(&mut writer, 1);
        writer.write(0, 2); // no M/S
        write_zero_channel_stream(&mut writer, 1);
        write_zero_channel_stream(&mut writer, 1);
        writer.write(ElementId::End.bits() as u32, 3);

        let decoded = AacLcDecoder::new(4, 0)
            .unwrap()
            .decode_raw_data_block_multichannel_f32(&writer.finish())
            .unwrap();
        assert_eq!(
            decoded.labels(),
            &[ChannelLabel::FrontLeft, ChannelLabel::FrontRight]
        );
    }

    #[test]
    fn stateful_decoder_matches_pce_labels_by_element_tag() {
        let pce = ProgramConfig {
            front: vec![ProgramElement {
                is_cpe: false,
                tag_select: 0,
            }],
            back: vec![ProgramElement {
                is_cpe: false,
                tag_select: 1,
            }],
            num_channels: 2,
            num_effective_channels: 2,
            ..ProgramConfig::default()
        };
        let mut writer = BitWriter::new();
        writer.write(ElementId::ProgramConfig.bits() as u32, 3);
        pce.write_to_writer(&mut writer).unwrap();
        write_zero_sce_payload_bits_with_tag(&mut writer, 1, false);
        write_zero_sce_payload_bits_with_tag(&mut writer, 0, false);
        writer.write(ElementId::End.bits() as u32, 3);
        let payload = writer.finish();

        let mut decoder = AacLcDecoder::new(4, 0).unwrap();
        let decoded = decoder
            .decode_raw_data_block_multichannel_f32(&payload)
            .unwrap();

        assert_eq!(decoded.channels(), 2);
        assert_eq!(
            decoded.labels(),
            &[ChannelLabel::BackCenter, ChannelLabel::FrontCenter]
        );
    }

    #[test]
    fn stateful_decoder_decodes_adts_multichannel_helpers() {
        let mut writer = BitWriter::new();
        write_zero_sce_payload_bits(&mut writer, false);
        write_zero_sce_payload_bits(&mut writer, false);
        writer.write(ElementId::End.bits() as u32, 3);
        let payload = writer.finish();
        let header = AdtsHeader::aac_lc(44_100, 2, payload.len()).unwrap();
        let mut frame = vec![0; header.header_len()];
        header.write(&mut frame).unwrap();
        frame.extend_from_slice(&payload);

        let mut decoder = AacLcDecoder::from_adts_header(header).unwrap();
        let pcm = decoder
            .decode_adts_frame_multichannel_interleaved_i16(&frame)
            .unwrap();

        assert_eq!(pcm.len(), 2048);
        assert!(pcm.iter().all(|sample| *sample == 0));
    }

    #[test]
    fn audio_specific_config_he_aac_fill_drives_dual_rate_qmf_output() {
        let header = LdSbrHeader {
            amp_resolution: true,
            start_frequency: 5,
            stop_frequency: 8,
            crossover_band: 2,
            frequency_scale: Some(1),
            alter_scale: Some(false),
            noise_bands: Some(2),
            ..LdSbrHeader::default()
        };
        let tables = LdSbrFrequencyTables::from_header(&header, 44_100).unwrap();
        let zero = sbr_huffman_code(SbrHuffmanBook::EnvelopeLevel30Frequency, 0);
        let mut sbr = BitWriter::new();
        sbr.write(crate::sbr::EXT_SBR_DATA as u32, 4);
        sbr.write_bool(true);
        header.write(&mut sbr).unwrap();
        sbr.write_bool(false); // bs_data_extra
        sbr.write(0, 2); // FIXFIX
        sbr.write(0, 2); // one envelope
        sbr.write_bool(true); // high frequency resolution
        sbr.write_bool(false); // envelope frequency direction
        sbr.write_bool(false); // noise frequency direction
        for _ in 0..tables.noise_band_count() {
            sbr.write(0, 2);
        }
        sbr.write(0, 6);
        for _ in 1..tables.high_band_count() {
            for &bit in &zero {
                sbr.write_bool(bit);
            }
        }
        sbr.write(31, 5); // low injected noise energy
        for _ in 1..tables.noise_band_count() {
            for &bit in &zero {
                sbr.write_bool(bit);
            }
        }
        sbr.write_bool(false); // harmonics
        sbr.write_bool(true); // extended data
        sbr.write(2, 4); // two extension bytes
        sbr.write(2, 2); // EXTENSION_ID_PS_CODING
        sbr.write_bool(true); // PS header
        sbr.write_bool(false); // IID disabled
        sbr.write_bool(false); // ICC disabled
        sbr.write_bool(false); // PS extension disabled
        sbr.write_bool(false); // FIX borders
        sbr.write(0, 2); // zero envelopes, retain neutral parameters
        sbr.write(0, 7); // byte padding
        let sbr = sbr.finish();

        let mut fill = BitWriter::new();
        assert!(sbr.len() < 15);
        fill.write(sbr.len() as u32, 4);
        for &byte in &sbr {
            fill.write(byte as u32, 8);
        }
        let payload = parse_sbr_fill_element(&mut BitReader::new(&fill.finish()))
            .unwrap()
            .unwrap();

        let build_raw = |sbr: &[u8]| {
            let mut raw = BitWriter::new();
            write_zero_sce_payload_bits(&mut raw, false);
            raw.write(ElementId::Fill.bits() as u32, 3);
            if sbr.len() < 15 {
                raw.write(sbr.len() as u32, 4);
            } else {
                raw.write(15, 4);
                raw.write((sbr.len() - 14) as u32, 8);
            }
            for &byte in sbr {
                raw.write(byte as u32, 8);
            }
            raw.write(ElementId::End.bits() as u32, 3);
            raw.finish()
        };
        let raw = build_raw(&sbr);
        let mut padded_sbr = sbr.clone();
        padded_sbr.resize(15, 0);
        let padded_raw = build_raw(&padded_sbr);

        let config = AudioSpecificConfig {
            audio_object_type: 2,
            sampling_frequency_index: 7,
            sampling_frequency: 22_050,
            channel_configuration: 1,
            extension: Some(AudioSpecificConfigExtension {
                audio_object_type: 5,
                sampling_frequency_index: 4,
                sampling_frequency: 44_100,
                ps_present: false,
            }),
            ga_specific: Some(crate::asc::GaSpecificConfig::default()),
            eld_specific: None,
            usac_config: None,
            error_protection_config: None,
            program_config: None,
            bits_read: 0,
        };
        let mut invalid_floating_decoder =
            AacLcDecoder::from_audio_specific_config(&config).unwrap();
        let mut invalid_floating_core = vec![Vec::new()];
        assert!(invalid_floating_decoder
            .process_ordinary_sbr_f32(
                &mut invalid_floating_core,
                std::slice::from_ref(&payload),
                &[false],
            )
            .is_err());
        let mut invalid_fixed_decoder = AacLcDecoder::from_audio_specific_config(&config).unwrap();
        let mut invalid_fixed_core = vec![Vec::new()];
        assert!(invalid_fixed_decoder
            .process_ordinary_sbr_fixed(
                &mut invalid_fixed_core,
                std::slice::from_ref(&payload),
                &[false],
            )
            .is_err());
        let mut decoder = AacLcDecoder::from_audio_specific_config(&config).unwrap();
        let decoded = decoder
            .decode_raw_data_block_multichannel_f32(&raw)
            .unwrap();
        assert_eq!(decoded.channels(), 2);
        assert_eq!(
            decoded.labels(),
            &[ChannelLabel::FrontLeft, ChannelLabel::FrontRight]
        );
        assert_eq!(decoded.channels[0].len(), 2048);
        assert_eq!(decoded.channels[1].len(), 2048);
        assert!(decoded.channels[0].iter().all(|sample| sample.is_finite()));
        let padded = AacLcDecoder::from_audio_specific_config(&config)
            .unwrap()
            .decode_raw_data_block_multichannel_f32(&padded_raw)
            .unwrap();
        assert_eq!(padded.channels(), 2);
        assert!(padded
            .channels
            .iter()
            .flatten()
            .all(|sample| sample.is_finite()));
        let mut invalid_sbr_concealment = decoder.clone();
        let mut invalid_core = vec![Vec::new()];
        assert!(invalid_sbr_concealment
            .conceal_ordinary_sbr_f32(&mut invalid_core)
            .is_err());
        let next = decoder.f32_concealment_spectral_frame().unwrap();
        let interpolated = decoder.conceal_f32_interpolated(&next).unwrap();
        assert_eq!(interpolated.len(), 4096);
        assert!(interpolated.iter().all(|sample| sample.is_finite()));
        let concealed = decoder.conceal_f32_interleaved().unwrap();
        assert_eq!(concealed.len(), 4096);
        assert!(concealed.iter().all(|sample| sample.is_finite()));
        let mut core_only = BitWriter::new();
        write_zero_sce_payload_bits(&mut core_only, false);
        core_only.write(ElementId::End.bits() as u32, 3);
        let core_only = core_only.finish();
        assert_eq!(
            decoder
                .decode_raw_data_block_multichannel_f32(&core_only)
                .unwrap()
                .samples_per_channel(),
            2048
        );

        let mut fixed_decoder = AacLcDecoder::from_audio_specific_config(&config).unwrap();
        let fixed = fixed_decoder
            .decode_raw_data_block_multichannel_fixed_interleaved_i16(&raw)
            .unwrap();
        assert_eq!(fixed.len(), 4096);
        let mut invalid_sbr_concealment = fixed_decoder.clone();
        let mut invalid_core = vec![Vec::new()];
        assert!(invalid_sbr_concealment
            .conceal_ordinary_sbr_fixed(&mut invalid_core)
            .is_err());
        let next = fixed_decoder.fixed_concealment_spectral_frame().unwrap();
        assert_eq!(
            fixed_decoder
                .conceal_fixed_interpolated_i16(&next)
                .unwrap()
                .len(),
            4096
        );
        assert_eq!(
            fixed_decoder.conceal_fixed_interleaved_i16().unwrap().len(),
            4096
        );
        assert_eq!(
            fixed_decoder
                .decode_raw_data_block_multichannel_fixed_interleaved_i16(&core_only)
                .unwrap()
                .len(),
            4096
        );

        let mut short_config = config.clone();
        short_config.ga_specific.as_mut().unwrap().frame_length_flag = true;
        let mut short_floating = AacLcDecoder::from_audio_specific_config(&short_config).unwrap();
        let short = short_floating
            .decode_raw_data_block_multichannel_f32(&raw)
            .unwrap();
        assert_eq!(short.channels.len(), 2);
        assert!(short.channels.iter().all(|channel| channel.len() == 1920));
        assert_eq!(
            short_floating.last_ps_frames[0]
                .as_ref()
                .unwrap()
                .borders
                .last(),
            Some(&30)
        );
        let mut short_fixed = AacLcDecoder::from_audio_specific_config(&short_config).unwrap();
        assert_eq!(
            short_fixed
                .decode_raw_data_block_multichannel_fixed_interleaved_i16(&raw)
                .unwrap()
                .len(),
            3840
        );
        assert_eq!(
            short_fixed.last_ps_fixed_frames[0]
                .as_ref()
                .unwrap()
                .borders
                .last(),
            Some(&30)
        );
    }

    #[test]
    fn ordinary_stereo_sbr_processes_and_conceals_f32_and_fixed_channels() {
        let header = LdSbrHeader {
            amp_resolution: true,
            start_frequency: 5,
            stop_frequency: 8,
            crossover_band: 2,
            frequency_scale: Some(1),
            alter_scale: Some(false),
            noise_bands: Some(2),
            ..LdSbrHeader::default()
        };
        let tables = LdSbrFrequencyTables::from_header(&header, 44_100).unwrap();
        let level_zero = sbr_huffman_code(SbrHuffmanBook::EnvelopeLevel30Frequency, 0);
        let balance_zero = sbr_huffman_code(SbrHuffmanBook::EnvelopeBalance30Frequency, 0);
        let mut frame = BitWriter::new();
        frame.write_bool(false); // no data extra
        frame.write_bool(true); // coupling
        frame.write(0, 2); // FIXFIX
        frame.write(0, 2); // one envelope
        frame.write_bool(true); // high frequency resolution
        for _ in 0..4 {
            frame.write_bool(false); // frequency-domain envelope/noise coding
        }
        for _ in 0..tables.noise_band_count() {
            frame.write(0, 2); // inverse filtering off
        }
        frame.write(0, 6); // left absolute envelope
        for _ in 1..tables.high_band_count() {
            for &bit in &level_zero {
                frame.write_bool(bit);
            }
        }
        frame.write(0, 5); // left absolute noise floor
        for _ in 1..tables.noise_band_count() {
            for &bit in &level_zero {
                frame.write_bool(bit);
            }
        }
        frame.write(6, 5); // centered envelope balance
        for _ in 1..tables.high_band_count() {
            for &bit in &balance_zero {
                frame.write_bool(bit);
            }
        }
        frame.write(6, 5); // centered noise balance
        for _ in 1..tables.noise_band_count() {
            for &bit in &balance_zero {
                frame.write_bool(bit);
            }
        }
        frame.write_bool(false); // no left harmonics
        frame.write_bool(false); // no right harmonics
        frame.write_bool(false); // no extended data
        let frame_data_bits = frame.bits_written();
        let payload = SbrFillPayload {
            extension_type: crate::sbr::EXT_SBR_DATA,
            transmitted_crc: None,
            header_present: true,
            header: Some(header),
            frame_data: frame.finish(),
            frame_data_bits,
        };

        let config = AudioSpecificConfig {
            audio_object_type: 2,
            sampling_frequency_index: 7,
            sampling_frequency: 22_050,
            channel_configuration: 2,
            extension: Some(AudioSpecificConfigExtension {
                audio_object_type: 5,
                sampling_frequency_index: 4,
                sampling_frequency: 44_100,
                ps_present: false,
            }),
            ga_specific: Some(crate::asc::GaSpecificConfig::default()),
            eld_specific: None,
            usac_config: None,
            error_protection_config: None,
            program_config: None,
            bits_read: 0,
        };

        let mut floating_decoder = AacLcDecoder::from_audio_specific_config(&config).unwrap();
        let mut floating = vec![vec![0.0; 1024], vec![0.0; 1024]];
        floating_decoder
            .process_ordinary_sbr_f32(&mut floating, &[payload.clone()], &[true])
            .unwrap();
        assert!(floating.iter().all(|channel| channel.len() == 2048));
        assert!(floating.iter().flatten().all(|sample| sample.is_finite()));
        let mut invalid_floating_decoder = floating_decoder.clone();
        let mut invalid_floating = vec![Vec::new(), Vec::new()];
        assert!(invalid_floating_decoder
            .process_ordinary_sbr_f32(
                &mut invalid_floating,
                std::slice::from_ref(&payload),
                &[true],
            )
            .is_err());
        assert_eq!(
            floating_decoder.process_ordinary_sbr_f32(&mut floating, &[payload.clone()], &[false],),
            Err(DecodeError::SbrPayloadLayoutMismatch)
        );
        let mut invalid_conceal_decoder = floating_decoder.clone();
        let mut invalid_concealed = vec![Vec::new(), Vec::new()];
        assert!(invalid_conceal_decoder
            .conceal_ordinary_sbr_f32(&mut invalid_concealed)
            .is_err());
        let mut concealed = vec![vec![0.0; 1024], vec![0.0; 1024]];
        floating_decoder
            .conceal_ordinary_sbr_f32(&mut concealed)
            .unwrap();
        assert!(concealed.iter().all(|channel| channel.len() == 2048));

        let mut fixed_decoder = AacLcDecoder::from_audio_specific_config(&config).unwrap();
        let mut fixed = vec![vec![0; 1024], vec![0; 1024]];
        fixed_decoder
            .process_ordinary_sbr_fixed(&mut fixed, &[payload.clone()], &[true])
            .unwrap();
        assert!(fixed.iter().all(|channel| channel.len() == 2048));
        let mut invalid_fixed_decoder = fixed_decoder.clone();
        let mut invalid_fixed = vec![Vec::new(), Vec::new()];
        assert!(
            invalid_fixed_decoder
                .process_ordinary_sbr_fixed(
                    &mut invalid_fixed,
                    std::slice::from_ref(&payload),
                    &[true],
                )
                .is_err()
        );
        assert_eq!(
            fixed_decoder.process_ordinary_sbr_fixed(&mut fixed, &[payload], &[false]),
            Err(DecodeError::SbrPayloadLayoutMismatch)
        );
        let mut invalid_conceal_decoder = fixed_decoder.clone();
        let mut invalid_concealed = vec![Vec::new(), Vec::new()];
        assert!(invalid_conceal_decoder
            .conceal_ordinary_sbr_fixed(&mut invalid_concealed)
            .is_err());
        let mut concealed = vec![vec![0; 1024], vec![0; 1024]];
        fixed_decoder
            .conceal_ordinary_sbr_fixed(&mut concealed)
            .unwrap();
        assert!(concealed.iter().all(|channel| channel.len() == 2048));
    }

    #[test]
    fn ordinary_mono_sbr_without_ps_processes_and_conceals_both_formats() {
        let header = LdSbrHeader {
            amp_resolution: true,
            start_frequency: 5,
            stop_frequency: 8,
            crossover_band: 2,
            frequency_scale: Some(1),
            alter_scale: Some(false),
            noise_bands: Some(2),
            ..LdSbrHeader::default()
        };
        let tables = LdSbrFrequencyTables::from_header(&header, 44_100).unwrap();
        let zero = sbr_huffman_code(SbrHuffmanBook::EnvelopeLevel30Frequency, 0);
        let mut frame = BitWriter::new();
        frame.write_bool(false); // no data extra
        frame.write(0, 2); // FIXFIX
        frame.write(0, 2); // one envelope
        frame.write_bool(true); // high frequency resolution
        frame.write_bool(false); // envelope frequency direction
        frame.write_bool(false); // noise frequency direction
        for _ in 0..tables.noise_band_count() {
            frame.write(0, 2); // inverse filtering off
        }
        frame.write(0, 6); // absolute envelope
        for _ in 1..tables.high_band_count() {
            for &bit in &zero {
                frame.write_bool(bit);
            }
        }
        frame.write(0, 5); // absolute noise floor
        for _ in 1..tables.noise_band_count() {
            for &bit in &zero {
                frame.write_bool(bit);
            }
        }
        frame.write_bool(false); // no harmonics
        frame.write_bool(false); // no extended data
        let frame_data_bits = frame.bits_written();
        let payload = SbrFillPayload {
            extension_type: crate::sbr::EXT_SBR_DATA,
            transmitted_crc: None,
            header_present: true,
            header: Some(header),
            frame_data: frame.finish(),
            frame_data_bits,
        };
        let config = AudioSpecificConfig {
            audio_object_type: 2,
            sampling_frequency_index: 7,
            sampling_frequency: 22_050,
            channel_configuration: 1,
            extension: Some(AudioSpecificConfigExtension {
                audio_object_type: 5,
                sampling_frequency_index: 4,
                sampling_frequency: 44_100,
                ps_present: false,
            }),
            ga_specific: Some(crate::asc::GaSpecificConfig::default()),
            eld_specific: None,
            usac_config: None,
            error_protection_config: None,
            program_config: None,
            bits_read: 0,
        };

        let mut floating_decoder = AacLcDecoder::from_audio_specific_config(&config).unwrap();
        let mut floating = vec![vec![0.0; 1024]];
        floating_decoder
            .process_ordinary_sbr_f32(&mut floating, &[payload.clone()], &[false])
            .unwrap();
        assert_eq!(floating[0].len(), 2048);
        let mut invalid_floating_decoder = floating_decoder.clone();
        let mut invalid_floating = vec![Vec::new()];
        assert!(invalid_floating_decoder
            .process_ordinary_sbr_f32(
                &mut invalid_floating,
                std::slice::from_ref(&payload),
                &[false],
            )
            .is_err());
        let mut invalid_conceal_decoder = floating_decoder.clone();
        let mut invalid_concealed = vec![Vec::new()];
        assert!(invalid_conceal_decoder
            .conceal_ordinary_sbr_f32(&mut invalid_concealed)
            .is_err());
        let mut concealed = vec![vec![0.0; 1024]];
        floating_decoder
            .conceal_ordinary_sbr_f32(&mut concealed)
            .unwrap();
        assert_eq!(concealed[0].len(), 2048);

        let mut fixed_decoder = AacLcDecoder::from_audio_specific_config(&config).unwrap();
        let mut fixed = vec![vec![0; 1024]];
        fixed_decoder
            .process_ordinary_sbr_fixed(&mut fixed, &[payload.clone()], &[false])
            .unwrap();
        assert_eq!(fixed[0].len(), 2048);
        let mut invalid_fixed_decoder = fixed_decoder.clone();
        let mut invalid_fixed = vec![Vec::new()];
        assert!(invalid_fixed_decoder
            .process_ordinary_sbr_fixed(
                &mut invalid_fixed,
                std::slice::from_ref(&payload),
                &[false],
            )
            .is_err());
        let mut invalid_conceal_decoder = fixed_decoder.clone();
        let mut invalid_concealed = vec![Vec::new()];
        assert!(invalid_conceal_decoder
            .conceal_ordinary_sbr_fixed(&mut invalid_concealed)
            .is_err());
        let mut concealed = vec![vec![0; 1024]];
        fixed_decoder
            .conceal_ordinary_sbr_fixed(&mut concealed)
            .unwrap();
        assert_eq!(concealed[0].len(), 2048);

        let mut short_config = config.clone();
        short_config.ga_specific.as_mut().unwrap().frame_length_flag = true;
        let mut short_floating = vec![vec![0.0; 960]];
        AacLcDecoder::from_audio_specific_config(&short_config)
            .unwrap()
            .process_ordinary_sbr_f32(
                &mut short_floating,
                std::slice::from_ref(&payload),
                &[false],
            )
            .unwrap();
        assert_eq!(short_floating[0].len(), 1920);
        let mut short_fixed = vec![vec![0; 960]];
        AacLcDecoder::from_audio_specific_config(&short_config)
            .unwrap()
            .process_ordinary_sbr_fixed(&mut short_fixed, &[payload], &[false])
            .unwrap();
        assert_eq!(short_fixed[0].len(), 1920);
    }

    #[test]
    fn stateful_decoder_iterates_adts_stream_to_multichannel_frames() {
        let mut writer = BitWriter::new();
        write_zero_sce_payload_bits(&mut writer, false);
        write_zero_sce_payload_bits(&mut writer, false);
        writer.write(ElementId::End.bits() as u32, 3);
        let payload = writer.finish();
        let header = AdtsHeader::aac_lc(44_100, 2, payload.len()).unwrap();
        let mut one_frame = vec![0; header.header_len()];
        header.write(&mut one_frame).unwrap();
        one_frame.extend_from_slice(&payload);
        let mut stream = one_frame.clone();
        stream.extend_from_slice(&one_frame);

        let mut frame_decoder = AacLcDecoder::from_adts_header(header).unwrap();
        let decoded_frames = frame_decoder
            .decode_adts_stream_multichannel_f32(&stream)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(decoded_frames.len(), 2);
        assert_eq!(decoded_frames[0].channels(), 2);
        assert_eq!(
            decoded_frames[0].labels(),
            &[ChannelLabel::FrontLeft, ChannelLabel::FrontRight]
        );

        let mut pcm_decoder = AacLcDecoder::from_adts_header(header).unwrap();
        let pcm_frames = pcm_decoder
            .decode_adts_stream_multichannel_interleaved_i16(&stream)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(pcm_frames.len(), 2);
        assert_eq!(pcm_frames[0].len(), 2048);
        assert!(pcm_frames.iter().flatten().all(|sample| *sample == 0));

        let mut fixed_decoder = AacLcDecoder::from_adts_header(header).unwrap();
        let fixed_frames = fixed_decoder
            .decode_adts_stream_multichannel_fixed_interleaved_i16(&stream)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(fixed_frames.len(), 2);
        assert_eq!(fixed_frames[0].len(), 2048);
        assert!(fixed_frames.iter().flatten().all(|sample| *sample == 0));
    }

    #[test]
    fn stateful_decoder_skips_fill_before_first_audio_element() {
        let mut writer = BitWriter::new();
        writer.write(ElementId::Fill.bits() as u32, 3);
        writer.write(1, 4);
        writer.write(0xa5, 8);
        write_zero_sce_payload_bits(&mut writer, false);
        let payload = writer.finish();

        let mut decoder = AacLcDecoder::new(4, 1).unwrap();
        let decoded = decoder.decode_raw_data_block_f32(&payload).unwrap();

        assert!(matches!(decoded, DecodedAacLcFrame::Mono(_)));
    }

    #[test]
    fn stateful_decoder_skips_data_stream_before_first_audio_element() {
        let mut writer = BitWriter::new();
        writer.write(ElementId::DataStream.bits() as u32, 3);
        writer.write(0, 4); // element_instance_tag
        writer.write_bool(true); // byte_align
        writer.write(1, 8); // count
        writer.byte_align();
        writer.write(0, 8);
        write_zero_sce_payload_bits(&mut writer, false);
        let payload = writer.finish();

        let mut decoder = AacLcDecoder::new(4, 1).unwrap();
        let decoded = decoder.decode_raw_data_block_f32(&payload).unwrap();

        assert!(matches!(decoded, DecodedAacLcFrame::Mono(_)));
        let fixed = AacLcDecoder::new(4, 1)
            .unwrap()
            .decode_raw_data_block_multichannel_fixed_interleaved_i16(&payload)
            .unwrap();
        assert_eq!(fixed.len(), 1024);
    }

    #[test]
    fn stateful_decoder_exposes_frame_local_ancillary_data() {
        let mut writer = BitWriter::new();
        for (tag, data) in [(3, &[0x12, 0x34][..]), (7, &[0xab][..])] {
            writer.write(ElementId::DataStream.bits() as u32, 3);
            writer.write(tag, 4);
            writer.write_bool(true);
            writer.write(data.len() as u32, 8);
            writer.byte_align();
            for &byte in data {
                writer.write(byte as u32, 8);
            }
        }
        write_zero_sce_payload_bits(&mut writer, false);
        let payload = writer.finish();

        let expected = [
            AncillaryDataElement {
                element_instance_tag: 3,
                data: vec![0x12, 0x34],
            },
            AncillaryDataElement {
                element_instance_tag: 7,
                data: vec![0xab],
            },
        ];
        let mut decoder = AacLcDecoder::new(4, 1).unwrap();
        decoder.init_ancillary_data(3);
        decoder.decode_raw_data_block_f32(&payload).unwrap();
        assert_eq!(decoder.ancillary_data(), expected);
        decoder
            .decode_raw_data_block_f32(&zero_sce_payload(false))
            .unwrap();
        assert!(decoder.ancillary_data().is_empty());

        let mut fixed = AacLcDecoder::new(4, 1).unwrap();
        fixed.init_ancillary_data(3);
        fixed
            .decode_raw_data_block_multichannel_fixed_interleaved_i16(&payload)
            .unwrap();
        assert_eq!(fixed.ancillary_data(), expected);
        fixed.disable_ancillary_data();
        assert!(fixed.ancillary_data().is_empty());
    }

    #[test]
    fn ancillary_capture_enforces_fdk_capacity_and_element_limits() {
        let mut oversized = BitWriter::new();
        oversized.write(ElementId::DataStream.bits() as u32, 3);
        oversized.write(0, 4);
        oversized.write_bool(false);
        oversized.write(2, 8);
        oversized.write(0x1234, 16);
        write_zero_sce_payload_bits(&mut oversized, false);
        let mut decoder = AacLcDecoder::new(4, 1).unwrap();
        decoder.init_ancillary_data(1);
        assert_eq!(
            decoder.decode_raw_data_block_f32(&oversized.finish()),
            Err(DecodeError::AncillaryBufferTooSmall {
                capacity: 1,
                required: 2,
            })
        );

        let mut excessive = BitWriter::new();
        for tag in 0..8 {
            excessive.write(ElementId::DataStream.bits() as u32, 3);
            excessive.write(tag, 4);
            excessive.write_bool(false);
            excessive.write(1, 8);
            excessive.write(tag, 8);
        }
        write_zero_sce_payload_bits(&mut excessive, false);
        let mut decoder = AacLcDecoder::new(4, 1).unwrap();
        decoder.init_ancillary_data(8);
        assert_eq!(
            decoder.decode_raw_data_block_f32(&excessive.finish()),
            Err(DecodeError::TooManyAncillaryElements)
        );
        assert_eq!(decoder.ancillary_data().len(), 7);
    }

    #[test]
    fn stream_info_reports_core_layout_and_fdk_channel_indices() {
        let decoder = AacLcDecoder::new(4, 6).unwrap();
        let info = decoder.stream_info();

        assert_eq!(info.sample_rate, 44_100);
        assert_eq!(info.aac_sample_rate, 44_100);
        assert_eq!(info.frame_size, 1024);
        assert_eq!(info.aac_samples_per_frame, 1024);
        assert_eq!(info.num_channels, 6);
        assert_eq!(info.aac_num_channels, 6);
        assert_eq!(
            info.channel_labels,
            vec![
                ChannelLabel::FrontCenter,
                ChannelLabel::FrontLeft,
                ChannelLabel::FrontRight,
                ChannelLabel::BackLeft,
                ChannelLabel::BackRight,
                ChannelLabel::Lfe,
            ]
        );
        assert_eq!(info.channel_indices, vec![0, 1, 2, 0, 1, 0]);
        assert_eq!(info.profile, -1);
        assert_eq!(info.audio_object_type, 2);
        assert_eq!(info.error_protection_config, -1);
        assert_eq!(info.flags, 0);
    }

    #[test]
    fn clear_history_resets_core_and_eld_filterbanks_and_concealment() {
        let mut decoder = AacLcDecoder::new(4, 1).unwrap();
        let mut spectrum = vec![0.0; 1024];
        spectrum[0] = 1.0;
        decoder.channel_filterbanks[0]
            .process_only_long_sine(&spectrum)
            .unwrap();
        assert!(decoder.channel_filterbanks[0]
            .overlap()
            .iter()
            .any(|sample| *sample != 0.0));
        decoder.f32_concealment_state = ConcealmentState::Mute;
        decoder.f32_concealment_losses = 9;
        decoder.ancillary_data.push(AncillaryDataElement {
            element_instance_tag: 1,
            data: vec![2],
        });

        decoder.clear_history().unwrap();
        assert!(decoder.channel_filterbanks[0]
            .overlap()
            .iter()
            .all(|sample| *sample == 0.0));
        assert_eq!(decoder.f32_concealment_state, ConcealmentState::Ok);
        assert_eq!(decoder.f32_concealment_losses, 0);
        assert!(decoder.ancillary_data().is_empty());

        let mut eld = AacLcDecoder::new_ga(39, 6, 1).unwrap();
        let mut eld_spectrum = vec![0.0; 512];
        eld_spectrum[0] = 1.0;
        eld.eld_channel_filterbanks[0]
            .process(&eld_spectrum)
            .unwrap();
        assert!(eld.eld_channel_filterbanks[0]
            .state()
            .iter()
            .any(|sample| *sample != 0.0));
        eld.clear_history().unwrap();
        assert!(eld.eld_channel_filterbanks[0]
            .state()
            .iter()
            .all(|sample| *sample == 0.0));
    }

    #[test]
    fn stream_info_reflects_ps_and_dual_rate_eld_sbr_output() {
        let mut he = AudioSpecificConfig::aac_lc(24_000, 1).unwrap();
        he.extension = Some(crate::asc::AudioSpecificConfigExtension {
            audio_object_type: 5,
            sampling_frequency_index: 3,
            sampling_frequency: 48_000,
            ps_present: true,
        });
        let he = AacLcDecoder::from_audio_specific_config(&he)
            .unwrap()
            .stream_info();
        assert_eq!((he.sample_rate, he.frame_size), (48_000, 2048));
        assert_eq!(
            (he.aac_sample_rate, he.aac_samples_per_frame),
            (24_000, 1024)
        );
        assert_eq!((he.num_channels, he.aac_num_channels), (2, 1));
        assert_eq!(
            he.channel_labels,
            vec![ChannelLabel::FrontLeft, ChannelLabel::FrontRight]
        );
        assert_eq!(he.channel_indices, vec![1, 2]);
        assert_eq!(he.extension_audio_object_type, Some(5));
        assert_eq!(he.extension_sampling_rate, Some(48_000));
        assert_eq!(
            he.flags & (STREAM_FLAG_SBR_PRESENT | STREAM_FLAG_PS_PRESENT),
            STREAM_FLAG_SBR_PRESENT | STREAM_FLAG_PS_PRESENT
        );

        let header = crate::asc::LdSbrHeader {
            amp_resolution: true,
            start_frequency: 5,
            stop_frequency: 3,
            crossover_band: 2,
            frequency_scale: Some(0),
            alter_scale: Some(false),
            noise_bands: Some(2),
            ..crate::asc::LdSbrHeader::default()
        };
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
                sbr_crc: true,
                sbr_headers: vec![header],
                ..crate::asc::EldSpecificConfig::default()
            }),
            usac_config: None,
            error_protection_config: Some(0),
            program_config: None,
            bits_read: 0,
        };
        let eld = AacLcDecoder::from_audio_specific_config(&eld)
            .unwrap()
            .stream_info();
        assert_eq!((eld.sample_rate, eld.frame_size), (48_000, 1024));
        assert_eq!(
            (eld.aac_sample_rate, eld.aac_samples_per_frame),
            (24_000, 512)
        );
        assert_eq!(eld.extension_sampling_rate, Some(48_000));
        assert_eq!(eld.extension_audio_object_type, Some(5));
        assert_eq!(eld.error_protection_config, 0);
        assert_eq!(
            eld.flags
                & (STREAM_FLAG_ER
                    | STREAM_FLAG_ELD
                    | STREAM_FLAG_SBR_PRESENT
                    | STREAM_FLAG_SBR_CRC),
            STREAM_FLAG_ER | STREAM_FLAG_ELD | STREAM_FLAG_SBR_PRESENT | STREAM_FLAG_SBR_CRC
        );
    }

    #[test]
    fn legacy_raw_facade_rejects_more_than_stereo_output() {
        let mut writer = BitWriter::new();
        for tag in 0..3 {
            write_zero_sce_payload_bits_with_tag(&mut writer, tag, false);
        }
        writer.write(ElementId::End.bits() as u32, 3);
        let mut decoder = AacLcDecoder::new(4, 0).unwrap();
        assert_eq!(
            decoder.decode_raw_data_block_f32(&writer.finish()),
            Err(DecodeError::UnsupportedChannelConfiguration(3))
        );
    }

    #[test]
    fn stateful_decoder_reports_no_audio_element_at_end() {
        let mut writer = BitWriter::new();
        writer.write(ElementId::End.bits() as u32, 3);
        let payload = writer.finish();
        let mut decoder = AacLcDecoder::new(4, 2).unwrap();

        assert_eq!(
            decoder
                .decode_raw_data_block_f32(&payload)
                .unwrap_err()
                .to_string(),
            "AAC raw_data_block contains no decodable audio element"
        );
    }

    #[test]
    fn stateful_decoder_rejects_unsupported_coupling_first_element() {
        let mut writer = BitWriter::new();
        writer.write(ElementId::CouplingChannel.bits() as u32, 3);
        writer.write(2, 4); // cce element_instance_tag
        writer.write_bool(false); // ind_sw_cce_flag
        writer.write(0, 3); // one coupled element
        writer.write_bool(false); // target SCE
        writer.write(1, 4); // target tag
        writer.write_bool(true); // cc_domain
        writer.write_bool(false); // gain_element_sign
        writer.write(0, 2); // gain_element_scale
        let payload = writer.finish();
        let mut decoder = AacLcDecoder::new(4, 2).unwrap();

        let err = decoder.decode_raw_data_block_f32(&payload).unwrap_err();
        assert_eq!(
            err.to_string(),
            "unsupported AAC coupling channel element tag 2 targeting 1 element(s)"
        );
        let expected_prefix = {
            let mut reader = BitReader::new(&payload);
            CouplingChannelElementPrefix::parse_aac_lc_from_reader(&mut reader).unwrap()
        };
        assert_eq!(
            err,
            DecodeError::UnsupportedCouplingChannelElement(expected_prefix)
        );
    }

    #[test]
    fn decodes_cce_coupled_channel_stream_and_implicit_unity_gain() {
        let mut writer = BitWriter::new();
        writer.write(ElementId::CouplingChannel.bits() as u32, 3);
        writer.write(2, 4); // cce element_instance_tag
        writer.write_bool(false); // ind_sw_cce_flag
        writer.write(0, 3); // one coupled element
        writer.write_bool(false); // target SCE
        writer.write(1, 4); // target tag
        writer.write_bool(true); // cc_domain
        writer.write_bool(false); // gain_element_sign
        writer.write(0, 2); // gain_element_scale
        write_zero_independent_channel_stream(&mut writer, 1);

        let decoded = decode_aac_lc_coupling_channel_element(&writer.finish(), 4).unwrap();

        assert_eq!(decoded.prefix.element_instance_tag, 2);
        assert_eq!(decoded.stream.ics.max_sfb, 1);
        assert_eq!(decoded.gain_lists.lists.len(), 1);
        assert_eq!(decoded.gain_lists.lists[0].words, vec![60]);
        assert!(apply_coupling_channel_element_noop_if_zero_gain(&decoded).is_err());
        let mut zero_gain = decoded;
        zero_gain.gain_lists.lists.clear();
        assert_eq!(
            apply_coupling_channel_element_noop_if_zero_gain(&zero_gain),
            Ok(())
        );
    }

    #[test]
    fn decodes_cce_coupled_channel_stream_to_fixed_spectrum_bridge() {
        let mut writer = BitWriter::new();
        writer.write(ElementId::CouplingChannel.bits() as u32, 3);
        writer.write(2, 4); // cce element_instance_tag
        writer.write_bool(false); // ind_sw_cce_flag
        writer.write(0, 3); // one coupled element
        writer.write_bool(false); // target SCE
        writer.write(1, 4); // target tag
        writer.write_bool(true); // cc_domain
        writer.write_bool(false); // gain_element_sign
        writer.write(0, 2); // gain_element_scale
        write_zero_independent_channel_stream(&mut writer, 1);

        let mut pns_random = PnsRandomState::new(1);
        let decoded = decode_aac_lc_coupling_channel_element_fixed_bridge(
            &writer.finish(),
            4,
            &mut pns_random,
        )
        .unwrap();

        assert_eq!(decoded.prefix.element_instance_tag, 2);
        assert_eq!(decoded.stream.ics.max_sfb, 1);
        assert_eq!(decoded.gain_lists.lists.len(), 1);
        assert_eq!(decoded.gain_lists.lists[0].words, vec![60]);
        assert!(decoded.stream.spectrum.windows[0]
            .iter()
            .all(|sample| *sample == 0));
    }

    #[test]
    fn decodes_cce_common_gain_list_and_rejects_application() {
        let mut writer = BitWriter::new();
        writer.write(ElementId::CouplingChannel.bits() as u32, 3);
        writer.write(1, 4); // cce element_instance_tag
        writer.write_bool(false); // ind_sw_cce_flag
        writer.write(0, 3); // one coupled element
        writer.write_bool(true); // target CPE
        writer.write(0, 4); // target tag
        writer.write_bool(true); // left
        writer.write_bool(true); // right, creates one extra gain list for coupled pair
        writer.write_bool(true); // cc_domain
        writer.write_bool(false); // gain_element_sign
        writer.write(0, 2); // gain_element_scale
        write_zero_independent_channel_stream(&mut writer, 1);
        writer.write_bool(true); // common_gain_element_present
        writer.write(0, 2); // SCL Huffman code -> 60

        let payload_bits = writer.bits_written();
        let payload = writer.finish();
        let decoded = decode_aac_lc_coupling_channel_element(&payload, 4).unwrap();

        let mut pns_random = PnsRandomState::new(1);
        // The final SCL value 60 is the one-bit codeword `0`; the second zero
        // written above is only look-ahead/padding and is not required by the
        // bounded decoder.
        for bit_len in 0..payload_bits - 1 {
            assert!(
                decode_aac_lc_coupling_channel_element_fixed_bridge_from_reader(
                    &mut BitReader::with_bit_len(&payload, bit_len).unwrap(),
                    4,
                    &mut pns_random,
                )
                .is_err()
            );
        }
        assert!(
            decode_aac_lc_coupling_channel_element_fixed_bridge_from_reader(
                &mut BitReader::with_bit_len(&payload, payload_bits - 1).unwrap(),
                4,
                &mut pns_random,
            )
            .is_ok()
        );

        assert_eq!(decoded.prefix.gain_element_lists, 2);
        assert_eq!(decoded.gain_lists.lists.len(), 2);
        assert!(decoded.gain_lists.lists[0].common_gain_element_present);
        assert_eq!(decoded.gain_lists.lists[0].words, vec![60]);
        assert!(decoded.gain_lists.lists[1].common_gain_element_present);
        assert_eq!(decoded.gain_lists.lists[1].words, vec![60]);
        assert_eq!(
            apply_coupling_channel_element_noop_if_zero_gain(&decoded)
                .unwrap_err()
                .to_string(),
            "AAC CCE non-zero coupling gain application is unsupported"
        );

        let mut prefix = decoded.prefix.clone();
        prefix.independently_switched = false;
        let ics = test_ics(2);
        let sections = test_sections(vec![ZERO_HCB, 1]);
        let mut gain_bits = BitWriter::new();
        gain_bits.write_bool(false); // bandwise gain list
        gain_bits.write(0, 2); // SCL Huffman code -> 60
        let gain_bytes = gain_bits.finish();
        let lists = decode_coupling_gain_element_lists_for_layout(
            &mut BitReader::new(&gain_bytes),
            &prefix,
            &ics,
            &sections,
        )
        .unwrap();
        assert!(!lists.lists[1].common_gain_element_present);
        assert_eq!(lists.lists[1].words, vec![60]);
        assert!(decode_coupling_gain_element_lists_for_layout(
            &mut BitReader::with_bit_len(&[0], 1).unwrap(),
            &prefix,
            &ics,
            &sections,
        )
        .is_err());
    }

    #[test]
    fn applies_frequency_coupling_common_gain_to_target_spectrum() {
        let ics = test_ics(1);
        let cce = DecodedCouplingChannelElement {
            prefix: CouplingChannelElementPrefix {
                element_instance_tag: 0,
                independently_switched: false,
                targets: Vec::new(),
                coupling_domain: true,
                gain_element_sign: false,
                gain_element_scale: 0,
                gain_element_lists: 2,
                bits_read: 0,
            },
            stream: test_stream(
                &ics,
                test_sections(vec![ZERO_HCB]),
                vec![0],
                vec![2.0, -4.0, 6.0, -8.0],
            ),
            gain_lists: CouplingGainElementLists {
                lists: vec![CouplingGainElementList {
                    common_gain_element_present: true,
                    words: vec![60],
                }],
            },
            bits_read: 0,
        };
        let mut target = InverseQuantizedSpectrum {
            windows: vec![vec![1.0, 1.0, 1.0, 1.0]],
        };

        apply_frequency_coupling_to_spectrum(&mut target, &cce, 0).unwrap();

        assert_eq!(target.windows[0], vec![3.0, -3.0, 7.0, -7.0]);

        let mut delegated = InverseQuantizedSpectrum {
            windows: vec![vec![1.0; 4]],
        };
        apply_frequency_coupling_bandwise_to_spectrum(&mut delegated, &cce, 0).unwrap();
        assert_eq!(delegated.windows[0], target.windows[0]);

        assert!(apply_frequency_coupling_to_spectrum(&mut target.clone(), &cce, 1).is_ok());
        let mut empty_gain = cce.clone();
        empty_gain.gain_lists.lists[0].words.clear();
        assert!(apply_frequency_coupling_to_spectrum(&mut target.clone(), &empty_gain, 0).is_ok());
        assert_eq!(
            apply_frequency_coupling_to_spectrum(
                &mut InverseQuantizedSpectrum { windows: vec![] },
                &cce,
                0,
            ),
            Err(DecodeError::CouplingLayoutMismatch)
        );
        assert_eq!(
            apply_frequency_coupling_to_spectrum(
                &mut InverseQuantizedSpectrum {
                    windows: vec![vec![0.0; 2]],
                },
                &cce,
                0,
            ),
            Err(DecodeError::CouplingLayoutMismatch)
        );
    }

    #[test]
    fn rejects_frequency_application_for_time_domain_coupling() {
        let ics = test_ics(1);
        let cce = DecodedCouplingChannelElement {
            prefix: CouplingChannelElementPrefix {
                element_instance_tag: 0,
                independently_switched: true,
                targets: Vec::new(),
                coupling_domain: false,
                gain_element_sign: false,
                gain_element_scale: 0,
                gain_element_lists: 2,
                bits_read: 0,
            },
            stream: test_stream(
                &ics,
                test_sections(vec![ZERO_HCB]),
                vec![0],
                vec![1.0, 1.0, 1.0, 1.0],
            ),
            gain_lists: CouplingGainElementLists {
                lists: vec![CouplingGainElementList {
                    common_gain_element_present: false,
                    words: vec![60],
                }],
            },
            bits_read: 0,
        };
        let mut target = InverseQuantizedSpectrum {
            windows: vec![vec![0.0; 4]],
        };

        assert_eq!(
            apply_frequency_coupling_to_spectrum(&mut target, &cce, 0)
                .unwrap_err()
                .to_string(),
            "AAC CCE time-domain coupling application is unsupported"
        );
        assert_eq!(
            apply_frequency_coupling_bandwise_to_spectrum(&mut target, &cce, 0),
            Err(DecodeError::TimeDomainCouplingUnsupported)
        );
    }

    #[test]
    fn applies_bandwise_frequency_coupling_gain() {
        let ics = test_ics(1);
        let cce = DecodedCouplingChannelElement {
            prefix: CouplingChannelElementPrefix {
                element_instance_tag: 0,
                independently_switched: false,
                targets: Vec::new(),
                coupling_domain: true,
                gain_element_sign: false,
                gain_element_scale: 0,
                gain_element_lists: 2,
                bits_read: 0,
            },
            stream: test_stream(
                &ics,
                test_sections(vec![1]),
                vec![0],
                vec![2.0, 4.0, 0.0, 0.0],
            ),
            gain_lists: CouplingGainElementLists {
                lists: vec![CouplingGainElementList {
                    common_gain_element_present: false,
                    words: vec![60],
                }],
            },
            bits_read: 0,
        };
        let mut target = InverseQuantizedSpectrum {
            windows: vec![vec![1.0, 1.0, 1.0, 1.0]],
        };

        apply_frequency_coupling_bandwise_to_spectrum(&mut target, &cce, 0).unwrap();

        assert_eq!(target.windows[0], vec![3.0, 5.0, 1.0, 1.0]);

        let mut delegated = InverseQuantizedSpectrum {
            windows: vec![vec![1.0; 4]],
        };
        apply_frequency_coupling_to_spectrum(&mut delegated, &cce, 0).unwrap();
        assert_eq!(delegated.windows[0], target.windows[0]);

        assert!(
            apply_frequency_coupling_bandwise_to_spectrum(&mut target.clone(), &cce, 1).is_ok()
        );
        assert_eq!(
            apply_frequency_coupling_bandwise_to_spectrum(
                &mut InverseQuantizedSpectrum { windows: vec![] },
                &cce,
                0,
            ),
            Err(DecodeError::CouplingLayoutMismatch)
        );
        assert_eq!(
            apply_frequency_coupling_bandwise_to_spectrum(
                &mut InverseQuantizedSpectrum {
                    windows: vec![vec![0.0; 2]],
                },
                &cce,
                0,
            ),
            Err(DecodeError::CouplingLayoutMismatch)
        );

        let two_bands = test_ics(2);
        let mut missing_gain = cce.clone();
        missing_gain.stream = test_stream(
            &two_bands,
            test_sections(vec![1, 1]),
            vec![0, 0],
            vec![1.0; 8],
        );
        assert_eq!(
            apply_frequency_coupling_bandwise_to_spectrum(
                &mut InverseQuantizedSpectrum {
                    windows: vec![vec![0.0; 8]],
                },
                &missing_gain,
                0,
            ),
            Err(DecodeError::CouplingLayoutMismatch)
        );
        let mut skips_zero = missing_gain;
        skips_zero.stream.section_data = test_sections(vec![ZERO_HCB, 1]);
        apply_frequency_coupling_bandwise_to_spectrum(
            &mut InverseQuantizedSpectrum {
                windows: vec![vec![0.0; 8]],
            },
            &skips_zero,
            0,
        )
        .unwrap();

        let mut grouped = cce.clone();
        grouped.stream.ics.window_sequence = WindowSequence::EightShort;
        grouped.stream.ics.window_group_lengths = vec![1, 1];
        grouped.stream.section_data.codebooks = vec![vec![1], vec![1]];
        grouped.stream.spectrum.windows = vec![vec![2.0; 16], vec![4.0; 16]];
        grouped.gain_lists.lists[0].words = vec![60, 60];
        let mut grouped_target = InverseQuantizedSpectrum {
            windows: vec![vec![0.0; 16], vec![0.0; 16]],
        };
        apply_frequency_coupling_bandwise_to_spectrum(&mut grouped_target, &grouped, 0).unwrap();
        assert!(grouped_target.windows[1]
            .iter()
            .any(|&sample| sample != 0.0));
    }

    #[test]
    fn accumulates_bandwise_cce_dpcm_gain_with_selected_exponent_step() {
        let ics = test_ics(2);
        let cce = DecodedCouplingChannelElement {
            prefix: CouplingChannelElementPrefix {
                element_instance_tag: 0,
                independently_switched: false,
                targets: Vec::new(),
                coupling_domain: true,
                gain_element_sign: false,
                gain_element_scale: 3,
                gain_element_lists: 2,
                bits_read: 0,
            },
            stream: test_stream(&ics, test_sections(vec![1, 1]), vec![0, 0], vec![1.0; 8]),
            gain_lists: CouplingGainElementLists {
                lists: vec![CouplingGainElementList {
                    common_gain_element_present: false,
                    words: vec![61, 61],
                }],
            },
            bits_read: 0,
        };
        let mut target = InverseQuantizedSpectrum {
            windows: vec![vec![0.0; 8]],
        };

        apply_frequency_coupling_bandwise_to_spectrum(&mut target, &cce, 0).unwrap();

        assert_eq!(&target.windows[0][..4], &[0.5; 4]);
        assert_eq!(&target.windows[0][4..8], &[0.25; 4]);
    }

    #[test]
    fn applies_time_domain_coupling_common_gain_to_samples() {
        let ics = test_ics(1);
        let cce = DecodedCouplingChannelElement {
            prefix: CouplingChannelElementPrefix {
                element_instance_tag: 0,
                independently_switched: true,
                targets: Vec::new(),
                coupling_domain: false,
                gain_element_sign: true,
                gain_element_scale: 0,
                gain_element_lists: 2,
                bits_read: 0,
            },
            stream: test_stream(&ics, test_sections(vec![ZERO_HCB]), vec![0], vec![0.0; 4]),
            gain_lists: CouplingGainElementLists {
                lists: vec![CouplingGainElementList {
                    common_gain_element_present: true,
                    words: vec![60],
                }],
            },
            bits_read: 0,
        };
        let mut target = vec![1.0, 2.0, 3.0];

        assert_eq!(
            apply_cce_to_staged_frequency_spectra(&mut [], &cce, 4),
            Err(DecodeError::TimeDomainCouplingUnsupported)
        );

        apply_time_domain_coupling_to_samples(&mut target, &[0.5, 1.0, 1.5], &cce, 0).unwrap();

        // With gain_element_sign set, the accumulator's low bit carries the
        // sign. A zero delta is therefore still positive unity gain.
        assert_eq!(target, vec![1.5, 3.0, 4.5]);

        let mut frequency_domain = cce.clone();
        frequency_domain.prefix.independently_switched = false;
        assert_eq!(
            apply_time_domain_coupling_to_samples(&mut [0.0], &[0.0], &frequency_domain, 0),
            Err(DecodeError::CouplingLayoutMismatch)
        );
        assert!(apply_time_domain_coupling_to_samples(&mut [0.0], &[0.0], &cce, 1).is_ok());
        let mut no_word = cce.clone();
        no_word.gain_lists.lists[0].words.clear();
        assert!(apply_time_domain_coupling_to_samples(&mut [0.0], &[0.0], &no_word, 0).is_ok());
        let mut bandwise_f32 = cce.clone();
        bandwise_f32.gain_lists.lists[0].common_gain_element_present = false;
        assert_eq!(
            apply_time_domain_coupling_to_samples(&mut [0.0], &[0.0], &bandwise_f32, 0),
            Err(DecodeError::BandwiseCouplingGainUnsupported)
        );
        assert_eq!(
            apply_time_domain_coupling_to_samples(&mut [0.0], &[0.0, 1.0], &cce, 0),
            Err(DecodeError::CouplingLayoutMismatch)
        );

        let mut fixed_target = vec![10, 20, 30];
        apply_time_domain_coupling_to_fixed_samples(&mut fixed_target, &[5, 10, 15], &cce, 0)
            .unwrap();
        assert_eq!(fixed_target, [15, 30, 45]);

        let fixed_cce = DecodedCouplingChannelElementFixed {
            prefix: cce.prefix.clone(),
            stream: DecodedChannelStreamFixed {
                global_gain: 100,
                ics: ics.clone(),
                section_data: test_sections(vec![ZERO_HCB]),
                scalefactors: ScalefactorData {
                    values: vec![vec![0]],
                },
                pulse_data: PulseData::absent(),
                tns_data: TnsData::absent(1),
                spectral: SpectralData {
                    windows: vec![vec![0; 4]],
                },
                spectrum: FixedInverseQuantizedSpectrum {
                    windows: vec![vec![0; 4]],
                    window_exponents: vec![0],
                },
            },
            gain_lists: cce.gain_lists.clone(),
            bits_read: 0,
        };
        assert_eq!(
            apply_cce_to_staged_fixed_frequency_spectra(&mut [], &fixed_cce, 4),
            Err(DecodeError::TimeDomainCouplingUnsupported)
        );
        let mut fixed_target = vec![10, 20, 30];
        apply_time_domain_coupling_to_fixed_samples_fixed_cce(
            &mut fixed_target,
            &[5, 10, 15],
            &fixed_cce,
            0,
        )
        .unwrap();
        assert_eq!(fixed_target, [15, 30, 45]);

        let mut invalid_fixed_cce = fixed_cce.clone();
        invalid_fixed_cce.prefix.independently_switched = false;
        assert_eq!(
            apply_time_domain_coupling_to_fixed_samples_fixed_cce(
                &mut [0],
                &[0],
                &invalid_fixed_cce,
                0,
            ),
            Err(DecodeError::CouplingLayoutMismatch)
        );
        assert!(apply_time_domain_coupling_to_fixed_samples_fixed_cce(
            &mut [0],
            &[0],
            &fixed_cce,
            1,
        )
        .is_ok());
        let mut empty_fixed_cce = fixed_cce.clone();
        empty_fixed_cce.gain_lists.lists[0].words.clear();
        assert!(apply_time_domain_coupling_to_fixed_samples_fixed_cce(
            &mut [0],
            &[0],
            &empty_fixed_cce,
            0,
        )
        .is_ok());
        let mut bandwise_fixed_cce = fixed_cce.clone();
        bandwise_fixed_cce.gain_lists.lists[0].common_gain_element_present = false;
        assert_eq!(
            apply_time_domain_coupling_to_fixed_samples_fixed_cce(
                &mut [0],
                &[0],
                &bandwise_fixed_cce,
                0,
            ),
            Err(DecodeError::BandwiseCouplingGainUnsupported)
        );
        assert_eq!(
            apply_time_domain_coupling_to_fixed_samples_fixed_cce(&mut [0], &[0, 1], &fixed_cce, 0,),
            Err(DecodeError::CouplingLayoutMismatch)
        );

        let targets = vec![
            crate::raw::CouplingTarget {
                is_cpe: false,
                tag_select: 2,
                left: true,
                right: false,
            },
            crate::raw::CouplingTarget {
                is_cpe: true,
                tag_select: 3,
                left: true,
                right: true,
            },
        ];
        let gain_lists = CouplingGainElementLists {
            lists: vec![
                CouplingGainElementList {
                    common_gain_element_present: true,
                    words: vec![60],
                };
                3
            ],
        };
        let channel_map = vec![
            StagedChannelMapEntry {
                element_id: ElementId::SingleChannel,
                element_instance_tag: 2,
                channel: 0,
                output_channel: 0,
            },
            StagedChannelMapEntry {
                element_id: ElementId::ChannelPair,
                element_instance_tag: 3,
                channel: 0,
                output_channel: 1,
            },
            StagedChannelMapEntry {
                element_id: ElementId::ChannelPair,
                element_instance_tag: 3,
                channel: 1,
                output_channel: 2,
            },
            StagedChannelMapEntry {
                element_id: ElementId::Lfe,
                element_instance_tag: 9,
                channel: 0,
                output_channel: 3,
            },
        ];

        let mut routed_cce = cce.clone();
        routed_cce.prefix.targets = targets.clone();
        routed_cce.prefix.gain_element_lists = 3;
        routed_cce.gain_lists = gain_lists.clone();
        let mut channels = vec![vec![1.0, 2.0]; 4];
        apply_time_domain_cce_to_channels(&mut channels, &channel_map, &routed_cce, &[0.5, 1.0])
            .unwrap();
        assert_eq!(channels[0], [1.5, 3.0]);
        assert_eq!(channels[1], [1.5, 3.0]);
        assert_eq!(channels[2], [1.5, 3.0]);
        assert_eq!(channels[3], [1.0, 2.0]);
        assert_eq!(
            apply_time_domain_cce_to_channels(&mut channels, &channel_map, &routed_cce, &[0.5]),
            Err(DecodeError::CouplingLayoutMismatch)
        );

        let mut routed_fixed_cce = fixed_cce.clone();
        routed_fixed_cce.prefix.targets = targets;
        routed_fixed_cce.prefix.gain_element_lists = 3;
        routed_fixed_cce.gain_lists = gain_lists;
        let mut fixed_channels = vec![vec![10, 20]; 4];
        apply_time_domain_cce_to_fixed_channels_fixed_cce(
            &mut fixed_channels,
            &channel_map,
            &routed_fixed_cce,
            &[5, 10],
        )
        .unwrap();
        assert_eq!(fixed_channels[0], [15, 30]);
        assert_eq!(fixed_channels[1], [15, 30]);
        assert_eq!(fixed_channels[2], [15, 30]);
        assert_eq!(fixed_channels[3], [10, 20]);
        assert_eq!(
            apply_time_domain_cce_to_fixed_channels_fixed_cce(
                &mut fixed_channels,
                &channel_map,
                &routed_fixed_cce,
                &[5],
            ),
            Err(DecodeError::CouplingLayoutMismatch)
        );

        let mut invalid_layout = cce.clone();
        invalid_layout.prefix.independently_switched = false;
        assert_eq!(
            apply_time_domain_coupling_to_fixed_samples(&mut [0], &[0], &invalid_layout, 0),
            Err(DecodeError::CouplingLayoutMismatch)
        );
        assert!(apply_time_domain_coupling_to_fixed_samples(&mut [0], &[0], &cce, 1).is_ok());
        let mut empty_word = cce.clone();
        empty_word.gain_lists.lists[0].words.clear();
        assert!(
            apply_time_domain_coupling_to_fixed_samples(&mut [0], &[0], &empty_word, 0).is_ok()
        );
        let mut bandwise = cce.clone();
        bandwise.gain_lists.lists[0].common_gain_element_present = false;
        assert_eq!(
            apply_time_domain_coupling_to_fixed_samples(&mut [0], &[0], &bandwise, 0),
            Err(DecodeError::BandwiseCouplingGainUnsupported)
        );
        assert_eq!(
            apply_time_domain_coupling_to_fixed_samples(&mut [0], &[0, 1], &cce, 0),
            Err(DecodeError::CouplingLayoutMismatch)
        );
    }

    #[test]
    fn applies_coupling_point_zero_before_target_tns() {
        let ics = test_ics(1);
        let tns = TnsData {
            present: true,
            filters: vec![vec![TnsFilter {
                start_band: 0,
                stop_band: 1,
                direction: TnsDirection::Forward,
                resolution: 3,
                coefficients: vec![2],
            }]],
        };
        let mut pair = synthetic_channel_pair_spectra();
        pair.left = test_stream(&ics, test_sections(vec![ZERO_HCB]), vec![0], vec![0.0; 4]);
        pair.left.tns_data = tns;
        pair.right = test_stream(&ics, test_sections(vec![ZERO_HCB]), vec![0], vec![0.0; 4]);
        let unrelated_stream = pair.right.clone();
        let mut staged = vec![
            StagedAacLcElement::Single {
                element_id: ElementId::SingleChannel,
                element_instance_tag: 9,
                spectra: DecodedSingleChannelSpectra {
                    side_info: SingleChannelElementSideInfo {
                        id: ElementId::SingleChannel,
                        element_instance_tag: 9,
                        global_gain: 100,
                        ics: ics.clone(),
                        bits_read: 0,
                    },
                    stream: unrelated_stream,
                    bits_read: 0,
                },
                labels: Vec::new(),
            },
            StagedAacLcElement::Pair {
                element_instance_tag: 2,
                spectra: pair,
                labels: Vec::new(),
            },
        ];
        let cce = DecodedCouplingChannelElement {
            prefix: CouplingChannelElementPrefix {
                element_instance_tag: 0,
                independently_switched: false,
                targets: vec![crate::raw::CouplingTarget {
                    is_cpe: true,
                    tag_select: 2,
                    left: true,
                    right: false,
                }],
                coupling_domain: false,
                gain_element_sign: false,
                gain_element_scale: 0,
                gain_element_lists: 1,
                bits_read: 0,
            },
            stream: test_stream(
                &ics,
                test_sections(vec![ZERO_HCB]),
                vec![0],
                vec![1.0, 0.0, 0.0, 0.0],
            ),
            gain_lists: CouplingGainElementLists {
                lists: vec![CouplingGainElementList {
                    common_gain_element_present: true,
                    words: vec![60],
                }],
            },
            bits_read: 0,
        };

        apply_staged_frequency_couplings(
            &mut staged,
            std::slice::from_ref(&cce),
            CouplingPoint::BeforeTns,
            4,
        )
        .unwrap();
        apply_tns_to_staged_spectra(&mut staged, 4).unwrap();

        let spectra = staged
            .iter()
            .find_map(|element| match element {
                StagedAacLcElement::Pair { spectra, .. } => Some(spectra),
                _ => None,
            })
            .unwrap();
        assert_eq!(cce.prefix.coupling_point(), CouplingPoint::BeforeTns);
        assert_ne!(spectra.left.spectrum.windows[0], vec![1.0, 0.0, 0.0, 0.0]);
        assert_eq!(spectra.right.spectrum.windows[0], vec![0.0; 4]);
    }

    #[test]
    fn staged_pair_frequency_coupling_targets_both_f32_and_fixed_channels() {
        let ics = test_ics(1);
        let prefix = CouplingChannelElementPrefix {
            element_instance_tag: 0,
            independently_switched: false,
            targets: vec![crate::raw::CouplingTarget {
                is_cpe: true,
                tag_select: 3,
                left: true,
                right: true,
            }],
            coupling_domain: true,
            gain_element_sign: false,
            gain_element_scale: 0,
            gain_element_lists: 2,
            bits_read: 0,
        };
        let gain_lists = CouplingGainElementLists {
            lists: vec![
                CouplingGainElementList {
                    common_gain_element_present: true,
                    words: vec![60],
                },
                CouplingGainElementList {
                    common_gain_element_present: true,
                    words: vec![60],
                },
            ],
        };
        let sections = test_sections(vec![ZERO_HCB]);
        let cce = DecodedCouplingChannelElement {
            prefix: prefix.clone(),
            stream: test_stream(&ics, sections.clone(), vec![0], vec![1.0, 0.5, -0.5, -1.0]),
            gain_lists: gain_lists.clone(),
            bits_read: 0,
        };
        let pair = DecodedChannelPairSpectra {
            prefix: ChannelPairElementSideInfoPrefix {
                element_instance_tag: 3,
                common_window: true,
                shared_ics: Some(ics.clone()),
                bits_read: 0,
            },
            ms_stereo: None,
            left: test_stream(&ics, sections.clone(), vec![0], vec![0.0; 4]),
            right: test_stream(&ics, sections.clone(), vec![0], vec![0.0; 4]),
            right_channel_start_bit: 0,
            bits_read: 0,
        };
        let mut staged = vec![
            StagedAacLcElement::Single {
                element_id: ElementId::SingleChannel,
                element_instance_tag: 9,
                spectra: DecodedSingleChannelSpectra {
                    side_info: SingleChannelElementSideInfo {
                        id: ElementId::SingleChannel,
                        element_instance_tag: 9,
                        global_gain: 100,
                        ics: ics.clone(),
                        bits_read: 0,
                    },
                    stream: pair.left.clone(),
                    bits_read: 0,
                },
                labels: Vec::new(),
            },
            StagedAacLcElement::Pair {
                element_instance_tag: 3,
                spectra: pair.clone(),
                labels: Vec::new(),
            },
        ];
        apply_cce_to_staged_frequency_spectra(&mut staged, &cce, 4).unwrap();
        let spectra = staged
            .iter()
            .find_map(|element| match element {
                StagedAacLcElement::Pair { spectra, .. } => Some(spectra),
                _ => None,
            })
            .unwrap();
        assert!(spectra.left.spectrum.windows[0]
            .iter()
            .any(|&value| value != 0.0));
        assert!(spectra.right.spectrum.windows[0]
            .iter()
            .any(|&value| value != 0.0));
        let before = spectra.clone();
        let mut unmatched = cce.clone();
        unmatched.prefix.targets[0].tag_select = 4;
        apply_cce_to_staged_frequency_spectra(&mut staged, &unmatched, 4).unwrap();
        let spectra = staged
            .iter()
            .find_map(|element| match element {
                StagedAacLcElement::Pair { spectra, .. } => Some(spectra),
                _ => None,
            })
            .unwrap();
        assert_eq!(spectra, &before);

        let fixed_stream = |values: Vec<i32>| DecodedChannelStreamFixed {
            global_gain: 100,
            ics: ics.clone(),
            section_data: sections.clone(),
            scalefactors: ScalefactorData {
                values: vec![vec![0]],
            },
            pulse_data: PulseData::absent(),
            tns_data: TnsData::absent(1),
            spectral: SpectralData {
                windows: vec![vec![0; 4]],
            },
            spectrum: FixedInverseQuantizedSpectrum {
                windows: vec![values],
                window_exponents: vec![0],
            },
        };
        let fixed_cce = DecodedCouplingChannelElementFixed {
            prefix,
            stream: fixed_stream(vec![1 << 20, 1 << 19, -(1 << 19), -(1 << 20)]),
            gain_lists,
            bits_read: 0,
        };
        let fixed_pair = DecodedChannelPairSpectraFixed {
            prefix: pair.prefix,
            ms_stereo: None,
            left: fixed_stream(vec![0; 4]),
            right: fixed_stream(vec![0; 4]),
            right_channel_start_bit: 0,
            bits_read: 0,
        };
        let mut fixed_staged = vec![
            StagedAacLcElementFixed::Single {
                element_id: ElementId::SingleChannel,
                element_instance_tag: 9,
                spectra: DecodedSingleChannelSpectraFixed {
                    side_info: SingleChannelElementSideInfo {
                        id: ElementId::SingleChannel,
                        element_instance_tag: 9,
                        global_gain: 100,
                        ics: ics.clone(),
                        bits_read: 0,
                    },
                    stream: fixed_pair.left.clone(),
                    bits_read: 0,
                },
                labels: Vec::new(),
            },
            StagedAacLcElementFixed::Pair {
                element_instance_tag: 3,
                spectra: fixed_pair,
                labels: Vec::new(),
            },
        ];
        apply_staged_fixed_frequency_couplings(
            &mut fixed_staged,
            std::slice::from_ref(&fixed_cce),
            CouplingPoint::BetweenTnsAndImdct,
            4,
        )
        .unwrap();
        let spectra = fixed_staged
            .iter()
            .find_map(|element| match element {
                StagedAacLcElementFixed::Pair { spectra, .. } => Some(spectra),
                _ => None,
            })
            .unwrap();
        assert!(spectra.left.spectrum.windows[0]
            .iter()
            .any(|&value| value != 0));
        assert!(spectra.right.spectrum.windows[0]
            .iter()
            .any(|&value| value != 0));
        let before = spectra.clone();
        let mut unmatched = fixed_cce.clone();
        unmatched.prefix.targets[0].tag_select = 4;
        apply_cce_to_staged_fixed_frequency_spectra(&mut fixed_staged, &unmatched, 4).unwrap();
        let spectra = fixed_staged
            .iter()
            .find_map(|element| match element {
                StagedAacLcElementFixed::Pair { spectra, .. } => Some(spectra),
                _ => None,
            })
            .unwrap();
        assert_eq!(spectra, &before);

        let mut left_only = cce.clone();
        left_only.prefix.targets[0].right = false;
        let mut invalid_left = staged.clone();
        if let StagedAacLcElement::Pair { spectra, .. } = &mut invalid_left[1] {
            spectra.left.spectrum.windows.clear();
        }
        assert!(apply_cce_to_staged_frequency_spectra(&mut invalid_left, &left_only, 4).is_err());
        let mut right_only = cce.clone();
        right_only.prefix.targets[0].left = false;
        let mut invalid_right = staged.clone();
        if let StagedAacLcElement::Pair { spectra, .. } = &mut invalid_right[1] {
            spectra.right.spectrum.windows.clear();
        }
        assert!(apply_cce_to_staged_frequency_spectra(&mut invalid_right, &right_only, 4).is_err());

        let mut left_only = fixed_cce.clone();
        left_only.prefix.targets[0].right = false;
        let mut invalid_left = fixed_staged.clone();
        if let StagedAacLcElementFixed::Pair { spectra, .. } = &mut invalid_left[1] {
            spectra.left.spectrum.windows.clear();
        }
        assert!(
            apply_cce_to_staged_fixed_frequency_spectra(&mut invalid_left, &left_only, 4).is_err()
        );
        let mut right_only = fixed_cce.clone();
        right_only.prefix.targets[0].left = false;
        let mut invalid_right = fixed_staged.clone();
        if let StagedAacLcElementFixed::Pair { spectra, .. } = &mut invalid_right[1] {
            spectra.right.spectrum.windows.clear();
        }
        assert!(
            apply_cce_to_staged_fixed_frequency_spectra(&mut invalid_right, &right_only, 4)
                .is_err()
        );

        let single_target = crate::raw::CouplingTarget {
            is_cpe: false,
            tag_select: 3,
            left: true,
            right: false,
        };
        let mut single_cce = cce;
        single_cce.prefix.targets = vec![single_target.clone()];
        let mut single = vec![StagedAacLcElement::Single {
            element_id: ElementId::SingleChannel,
            element_instance_tag: 3,
            spectra: DecodedSingleChannelSpectra {
                side_info: SingleChannelElementSideInfo {
                    id: ElementId::SingleChannel,
                    element_instance_tag: 3,
                    global_gain: 100,
                    ics: ics.clone(),
                    bits_read: 0,
                },
                stream: test_stream(&ics, sections.clone(), vec![0], Vec::new()),
                bits_read: 0,
            },
            labels: Vec::new(),
        }];
        assert!(apply_cce_to_staged_frequency_spectra(&mut single, &single_cce, 4).is_err());

        let mut single_fixed_cce = fixed_cce;
        single_fixed_cce.prefix.targets = vec![single_target];
        let mut single = vec![StagedAacLcElementFixed::Single {
            element_id: ElementId::SingleChannel,
            element_instance_tag: 3,
            spectra: DecodedSingleChannelSpectraFixed {
                side_info: SingleChannelElementSideInfo {
                    id: ElementId::SingleChannel,
                    element_instance_tag: 3,
                    global_gain: 100,
                    ics: ics.clone(),
                    bits_read: 0,
                },
                stream: fixed_stream(Vec::new()),
                bits_read: 0,
            },
            labels: Vec::new(),
        }];
        assert!(
            apply_cce_to_staged_fixed_frequency_spectra(&mut single, &single_fixed_cce, 4).is_err()
        );
    }

    #[test]
    fn applies_coupling_to_matching_target_spectra() {
        let ics = test_ics(1);
        let cce = DecodedCouplingChannelElement {
            prefix: CouplingChannelElementPrefix {
                element_instance_tag: 0,
                independently_switched: false,
                targets: vec![crate::raw::CouplingTarget {
                    is_cpe: true,
                    tag_select: 3,
                    left: true,
                    right: false,
                }],
                coupling_domain: true,
                gain_element_sign: false,
                gain_element_scale: 0,
                gain_element_lists: 1,
                bits_read: 0,
            },
            stream: test_stream(
                &ics,
                test_sections(vec![ZERO_HCB]),
                vec![0],
                vec![1.0, 2.0, 3.0, 4.0],
            ),
            gain_lists: CouplingGainElementLists {
                lists: vec![CouplingGainElementList {
                    common_gain_element_present: true,
                    words: vec![60],
                }],
            },
            bits_read: 0,
        };
        let mut targets = vec![CouplingTargetSpectrum {
            element_id: ElementId::ChannelPair,
            element_instance_tag: 3,
            channel: 0,
            spectrum: InverseQuantizedSpectrum {
                windows: vec![vec![0.0; 4]],
            },
        }];

        apply_coupling_channel_element_to_matching_spectra(&mut targets, &cce).unwrap();

        assert_eq!(targets[0].spectrum.windows[0], vec![1.0, 2.0, 3.0, 4.0]);

        targets[0].spectrum.windows.clear();
        assert_eq!(
            apply_coupling_channel_element_to_matching_spectra(&mut targets, &cce),
            Err(DecodeError::CouplingLayoutMismatch)
        );
        let mut unmatched = cce;
        unmatched.prefix.targets[0].tag_select = 4;
        assert_eq!(
            apply_coupling_channel_element_to_matching_spectra(&mut targets, &unmatched),
            Ok(())
        );
    }

    #[test]
    fn maps_multiple_cce_gain_lists_to_sce_and_cpe_channels_in_syntax_order() {
        let ics = test_ics(1);
        let cce = DecodedCouplingChannelElement {
            prefix: CouplingChannelElementPrefix {
                element_instance_tag: 7,
                independently_switched: false,
                targets: vec![
                    crate::raw::CouplingTarget {
                        is_cpe: false,
                        tag_select: 2,
                        left: true,
                        right: false,
                    },
                    crate::raw::CouplingTarget {
                        is_cpe: true,
                        tag_select: 3,
                        left: true,
                        right: true,
                    },
                ],
                coupling_domain: true,
                gain_element_sign: false,
                gain_element_scale: 3,
                gain_element_lists: 3,
                bits_read: 0,
            },
            stream: test_stream(&ics, test_sections(vec![ZERO_HCB]), vec![0], vec![1.0; 4]),
            gain_lists: CouplingGainElementLists {
                lists: vec![
                    CouplingGainElementList {
                        common_gain_element_present: true,
                        words: vec![60],
                    },
                    CouplingGainElementList {
                        common_gain_element_present: true,
                        words: vec![61],
                    },
                    CouplingGainElementList {
                        common_gain_element_present: true,
                        words: vec![62],
                    },
                ],
            },
            bits_read: 0,
        };
        let mut targets = vec![
            CouplingTargetSpectrum {
                element_id: ElementId::SingleChannel,
                element_instance_tag: 2,
                channel: 0,
                spectrum: InverseQuantizedSpectrum {
                    windows: vec![vec![0.0; 4]],
                },
            },
            CouplingTargetSpectrum {
                element_id: ElementId::ChannelPair,
                element_instance_tag: 3,
                channel: 0,
                spectrum: InverseQuantizedSpectrum {
                    windows: vec![vec![0.0; 4]],
                },
            },
            CouplingTargetSpectrum {
                element_id: ElementId::ChannelPair,
                element_instance_tag: 3,
                channel: 1,
                spectrum: InverseQuantizedSpectrum {
                    windows: vec![vec![0.0; 4]],
                },
            },
        ];

        apply_coupling_channel_element_to_matching_spectra(&mut targets, &cce).unwrap();

        assert_eq!(targets[0].spectrum.windows[0], vec![1.0; 4]);
        assert_eq!(targets[1].spectrum.windows[0], vec![0.5; 4]);
        assert_eq!(targets[2].spectrum.windows[0], vec![0.25; 4]);
    }

    #[test]
    fn applies_frequency_coupling_to_fixed_spectrum_bridge() {
        let ics = test_ics(1);
        let cce = DecodedCouplingChannelElementFixed {
            prefix: CouplingChannelElementPrefix {
                element_instance_tag: 0,
                independently_switched: false,
                targets: Vec::new(),
                coupling_domain: true,
                gain_element_sign: false,
                gain_element_scale: 0,
                gain_element_lists: 1,
                bits_read: 0,
            },
            stream: DecodedChannelStreamFixed {
                global_gain: 100,
                ics: ics.clone(),
                section_data: test_sections(vec![ZERO_HCB]),
                scalefactors: ScalefactorData {
                    values: vec![vec![0]],
                },
                pulse_data: PulseData::absent(),
                tns_data: TnsData::absent(1),
                spectral: SpectralData {
                    windows: vec![vec![0; 4]],
                },
                spectrum: FixedInverseQuantizedSpectrum {
                    windows: vec![vec![1, 2, 3, 4]],
                    window_exponents: vec![0],
                },
            },
            gain_lists: CouplingGainElementLists {
                lists: vec![CouplingGainElementList {
                    common_gain_element_present: true,
                    words: vec![60],
                }],
            },
            bits_read: 0,
        };
        let mut target = FixedInverseQuantizedSpectrum {
            windows: vec![vec![0; 4]],
            window_exponents: vec![0],
        };

        apply_frequency_coupling_to_fixed_spectrum_bridge(&mut target, &cce, 0).unwrap();

        assert_eq!(target.windows[0], vec![1, 2, 3, 4]);

        let mut delegated = FixedInverseQuantizedSpectrum {
            windows: vec![vec![0; 4]],
            window_exponents: vec![0],
        };
        apply_frequency_coupling_bandwise_to_fixed_spectrum_bridge(&mut delegated, &cce, 0)
            .unwrap();
        assert_eq!(delegated.windows[0], target.windows[0]);

        let mut time_domain = cce.clone();
        time_domain.prefix.independently_switched = true;
        time_domain.prefix.coupling_domain = false;
        assert_eq!(
            apply_frequency_coupling_to_fixed_spectrum_bridge(&mut target.clone(), &time_domain, 0,),
            Err(DecodeError::TimeDomainCouplingUnsupported)
        );
        assert!(
            apply_frequency_coupling_to_fixed_spectrum_bridge(&mut target.clone(), &cce, 1,)
                .is_ok()
        );
        let mut empty_gain = cce.clone();
        empty_gain.gain_lists.lists[0].words.clear();
        assert!(apply_frequency_coupling_to_fixed_spectrum_bridge(
            &mut target.clone(),
            &empty_gain,
            0,
        )
        .is_ok());
        assert_eq!(
            apply_frequency_coupling_to_fixed_spectrum_bridge(
                &mut FixedInverseQuantizedSpectrum {
                    windows: vec![],
                    window_exponents: vec![],
                },
                &cce,
                0,
            ),
            Err(DecodeError::CouplingLayoutMismatch)
        );
        assert_eq!(
            apply_frequency_coupling_to_fixed_spectrum_bridge(
                &mut FixedInverseQuantizedSpectrum {
                    windows: vec![vec![0; 2]],
                    window_exponents: vec![0],
                },
                &cce,
                0,
            ),
            Err(DecodeError::CouplingLayoutMismatch)
        );

        let mut bandwise = cce.clone();
        bandwise.stream.section_data = test_sections(vec![1]);
        bandwise.stream.spectrum = FixedInverseQuantizedSpectrum {
            windows: vec![vec![2, 4, 0, 0]],
            window_exponents: vec![0],
        };
        bandwise.gain_lists.lists[0] = CouplingGainElementList {
            common_gain_element_present: false,
            words: vec![60],
        };
        let mut target = FixedInverseQuantizedSpectrum {
            windows: vec![vec![1; 4]],
            window_exponents: vec![0],
        };
        apply_frequency_coupling_bandwise_to_fixed_spectrum_bridge(&mut target, &bandwise, 0)
            .unwrap();
        assert_eq!(target.windows[0], vec![3, 5, 1, 1]);

        assert_eq!(
            apply_frequency_coupling_bandwise_to_fixed_spectrum_bridge(
                &mut target.clone(),
                &time_domain,
                0,
            ),
            Err(DecodeError::TimeDomainCouplingUnsupported)
        );
        assert!(apply_frequency_coupling_bandwise_to_fixed_spectrum_bridge(
            &mut target.clone(),
            &bandwise,
            1,
        )
        .is_ok());
        assert_eq!(
            apply_frequency_coupling_bandwise_to_fixed_spectrum_bridge(
                &mut FixedInverseQuantizedSpectrum {
                    windows: vec![],
                    window_exponents: vec![],
                },
                &bandwise,
                0,
            ),
            Err(DecodeError::CouplingLayoutMismatch)
        );
        assert_eq!(
            apply_frequency_coupling_bandwise_to_fixed_spectrum_bridge(
                &mut FixedInverseQuantizedSpectrum {
                    windows: vec![vec![0; 2]],
                    window_exponents: vec![0],
                },
                &bandwise,
                0,
            ),
            Err(DecodeError::CouplingLayoutMismatch)
        );

        let two_bands = test_ics(2);
        let mut missing_gain = bandwise.clone();
        missing_gain.stream.ics = two_bands;
        missing_gain.stream.section_data = test_sections(vec![1, 1]);
        missing_gain.stream.spectrum = FixedInverseQuantizedSpectrum {
            windows: vec![vec![1; 8]],
            window_exponents: vec![0],
        };
        assert_eq!(
            apply_frequency_coupling_bandwise_to_fixed_spectrum_bridge(
                &mut FixedInverseQuantizedSpectrum {
                    windows: vec![vec![0; 8]],
                    window_exponents: vec![0],
                },
                &missing_gain,
                0,
            ),
            Err(DecodeError::CouplingLayoutMismatch)
        );
        let mut skips_zero = missing_gain;
        skips_zero.stream.section_data = test_sections(vec![ZERO_HCB, 1]);
        apply_frequency_coupling_bandwise_to_fixed_spectrum_bridge(
            &mut FixedInverseQuantizedSpectrum {
                windows: vec![vec![0; 8]],
                window_exponents: vec![0],
            },
            &skips_zero,
            0,
        )
        .unwrap();

        let mut grouped = bandwise.clone();
        grouped.stream.ics.window_sequence = WindowSequence::EightShort;
        grouped.stream.ics.window_group_lengths = vec![1, 1];
        grouped.stream.section_data.codebooks = vec![vec![1], vec![1]];
        grouped.stream.spectrum.windows = vec![vec![2; 16], vec![4; 16]];
        grouped.gain_lists.lists[0].words = vec![60, 60];
        let mut grouped_target = FixedInverseQuantizedSpectrum {
            windows: vec![vec![0; 16], vec![0; 16]],
            window_exponents: vec![0, 0],
        };
        apply_frequency_coupling_bandwise_to_fixed_spectrum_bridge(
            &mut grouped_target,
            &grouped,
            0,
        )
        .unwrap();
        assert!(grouped_target.windows[1].iter().any(|&sample| sample != 0));

        let mut delegated = FixedInverseQuantizedSpectrum {
            windows: vec![vec![1; 4]],
            window_exponents: vec![0],
        };
        apply_frequency_coupling_to_fixed_spectrum_bridge(&mut delegated, &bandwise, 0).unwrap();
        assert_eq!(delegated.windows[0], target.windows[0]);
    }

    #[test]
    fn gain_element_scale_changes_coupling_scale() {
        assert_eq!(coupling_gain_word_to_scale(60, false), 1.0);
        assert_eq!(coupling_gain_word_to_scale_with_scale(60, false, 0), 1.0);
        assert_eq!(coupling_gain_word_to_scale_with_scale(60, false, 3), 1.0);
        assert_eq!(coupling_gain_word_to_scale_with_scale(61, false, 3), 0.5);
        assert!(
            (coupling_gain_word_to_scale_with_scale(61, false, 0) - 2.0f32.powf(-0.125)).abs()
                < 1.0e-7
        );
        assert_eq!(coupling_gain_word_to_scale_with_scale(61, true, 3), -1.0);
        assert_eq!(coupling_gain_word_to_scale_with_scale(62, true, 3), 0.5);
    }

    #[test]
    fn decodes_independently_switched_cce_gain_list_without_presence_bit() {
        let mut writer = BitWriter::new();
        writer.write(ElementId::CouplingChannel.bits() as u32, 3);
        writer.write(0, 4); // cce element_instance_tag
        writer.write_bool(true); // ind_sw_cce_flag forces common gain elements
        writer.write(0, 3); // one coupled element
        writer.write_bool(true); // target CPE
        writer.write(0, 4); // target tag
        writer.write_bool(true); // left
        writer.write_bool(true); // right, one additional gain element list
        writer.write_bool(true); // cc_domain
        writer.write_bool(false); // gain_element_sign
        writer.write(0, 2); // gain_element_scale
        write_zero_independent_channel_stream(&mut writer, 1);
        writer.write(0, 2); // no common_gain_element_present bit here; SCL Huffman code -> 60

        let decoded = decode_aac_lc_coupling_channel_element(&writer.finish(), 4).unwrap();

        assert!(decoded.prefix.independently_switched);
        assert_eq!(decoded.gain_lists.lists.len(), 2);
        assert!(decoded.gain_lists.lists[0].common_gain_element_present);
        assert_eq!(decoded.gain_lists.lists[0].words, vec![60]);
        assert!(decoded.gain_lists.lists[1].common_gain_element_present);
        assert_eq!(decoded.gain_lists.lists[1].words, vec![60]);
    }
}
