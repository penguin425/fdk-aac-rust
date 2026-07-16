#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use std::ffi::c_void;

pub type INT = i32;
pub type UINT = u32;
pub type UCHAR = u8;
pub type SCHAR = i8;
pub type INT64 = i64;
pub type INT_PCM = i16;

unsafe extern "C" {
    pub fn fdk_ld_block_switch_test(
        input: *const i16,
        frames: i32,
        frame_length: i32,
        low_overlap: *mut u8,
    ) -> i32;
    pub fn fdk_sbr_fast_transient_test(
        slot_band_energy: *const i32,
        frames: i32,
        time_slots: i32,
        qmf_bands: i32,
        bandwidth_qmf_slot: i32,
        first_sbr_band: i32,
        transient_info: *mut u8,
    ) -> i32;
    pub fn fdk_sbr_code_envelope_test(
        absolute_values: *const i8,
        frames: i32,
        bands: i32,
        coded_values: *mut i8,
        directions: *mut i32,
    ) -> i32;
    pub fn fdk_sbr_coupled_delta_bits_test(
        current: *const i8,
        previous: *const i8,
        bands: i32,
        channel: i32,
        amp_res_1_5: i32,
        frequency_bits: *mut i32,
        time_bits: *mut i32,
    ) -> i32;
    pub fn fdk_sbr_first_env_threshold_test(time_bits: i32, streak: i32) -> i32;
    pub fn fdk_sbr_code_envelope_coupled_test(
        absolute_values: *const i8,
        frames: i32,
        bands: i32,
        channel: i32,
        amp_res_1_5: i32,
        coded_values: *mut i8,
        directions: *mut i32,
    ) -> i32;
    pub fn fdk_sbr_invf_regions_test(
        original_q16: *const i32,
        sbr_q16: *const i32,
        energy_q16: *const i32,
        transient: *const u8,
        frames: i32,
        modes: *mut u8,
    ) -> i32;
    pub fn fdk_sbr_invf_capture_enable(enable: i32);
    pub fn fdk_sbr_invf_capture_get(
        channel: i32,
        modes: *mut u8,
        original_regions: *mut i32,
        sbr_regions: *mut i32,
        capacity: i32,
    ) -> i32;
    pub fn fdk_sbr_invf_quota_capture_get(channel: i32, quotas: *mut i32, capacity: i32) -> i32;
    pub fn fdk_sbr_invf_detector_test(
        quotas: *const i32,
        energies: *const i32,
        index_vector: *const i8,
        band_table: *const i32,
        transient_flags: *const u8,
        frames: i32,
        estimates: i32,
        channels: i32,
        bands: i32,
        modes: *mut u8,
        original_regions: *mut i32,
        sbr_regions: *mut i32,
    ) -> i32;
    pub fn fdk_sbr_patch_map_test(
        master: *const u8,
        master_count: i32,
        high_start: i32,
        sample_rate: i32,
        qmf_channels: i32,
        index_vector: *mut i8,
    ) -> i32;
    pub fn fdk_sbr_tonality_quota_test(
        real: *const i32,
        imaginary: *const i32,
        samples: i32,
        quota: *mut i32,
        energy: *mut i32,
    ) -> i32;
    pub fn fdk_sbr_global_tonality_test(
        quotas: *const i32,
        energies: *const i32,
        estimates: i32,
        slots: i32,
        bands: i32,
        start_band: i32,
        previous_tonality: i32,
        current_tonality: *mut i32,
        global_tonality: *mut i32,
        amp_res_3db: *mut i32,
    ) -> i32;
    pub fn fdk_sbr_complex_energy_test(
        real: *const i32,
        imaginary: *const i32,
        slots: i32,
        bands: i32,
        qmf_scale: i32,
        energies: *mut i32,
        energy_scale: *mut i32,
    ) -> i32;
    pub fn fdk_sbr_envelope_capture_enable(enable: i32);
    pub fn fdk_sbr_qmf_input_capture_enable(enable: i32);
    pub fn fdk_sbr_qmf_input_capture_get(channel: i32, output: *mut i16, capacity: i32) -> i32;
    pub fn fdk_sbr_qmf_output_capture_get(
        channel: i32,
        real: *mut i32,
        imaginary: *mut i32,
        capacity: i32,
    ) -> i32;
    pub fn fdk_sbr_envelope_capture_get(
        channel: i32,
        values: *mut i8,
        capacity: i32,
        scale0: *mut i32,
        scale1: *mut i32,
        qmf_scale: *mut i32,
    ) -> i32;
    pub fn fdk_sbr_prequant_capture_get(
        channel: i32,
        energies: *mut i32,
        sample_counts: *mut i32,
        capacity: i32,
        common_scale: *mut i32,
    ) -> i32;
    pub fn fdk_sbr_transient_capture_get(channel: i32, position: *mut i32, flag: *mut i32) -> i32;
    pub fn fdk_sbr_sfb_energy_test(
        energies: *const i32,
        slots: i32,
        bands: i32,
        lower_band: i32,
        upper_band: i32,
        start_slot: i32,
        stop_slot: i32,
        scale0: i32,
        scale1: i32,
    ) -> i32;
    pub fn fdk_sbr_sfb_energy_split_test(
        energies: *const i32,
        slots: i32,
        bands: i32,
        lower_band: i32,
        upper_band: i32,
        start_slot: i32,
        stop_slot: i32,
        border_slot: i32,
        scale0: i32,
        scale1: i32,
    ) -> i32;
    pub fn fdk_sbr_log2_ld64_test(values: *const i32, count: i32, output: *mut i32) -> i32;
    pub fn fdk_log2_coefficients_test(
        coefficients: *mut i32,
        count: i32,
        conversion: *mut i32,
    ) -> i32;
    pub fn fdk_sbr_quantize_energy_test(
        energy: i32,
        sample_count: i32,
        common_scale: i32,
        amp_res_3db: i32,
    ) -> i32;
    pub fn fdk_imlt_long_test(
        input: *const i32,
        shape: *const u8,
        frames: i32,
        frame_length: i32,
        output: *mut i32,
    ) -> i32;
    pub fn fdk_psy_configuration_test(
        bitrate: i32,
        sample_rate: i32,
        bandwidth: i32,
        frame_length: i32,
        filterbank: i32,
        sfb_count: *mut i32,
        sfb_active: *mut i32,
        offsets: *mut i32,
        min_snr_q31: *mut i32,
        mask_low_q31: *mut i32,
        mask_high_q31: *mut i32,
    ) -> i32;
    pub fn fdk_adjust_pe_min_max_test(
        current_pe: *const i32,
        frames: i32,
        initial_mean_pe: i32,
        minimum_pe: *mut i32,
        maximum_pe: *mut i32,
    ) -> i32;
    pub fn fdk_reduce_thresholds_cbr_test(
        energy_ld64: *const i32,
        threshold_ld64: *const i32,
        minimum_snr_ld64: *const i32,
        avoid_hole: *const u8,
        sfb_count: i32,
        reduction_mantissa_q31: i32,
        reduction_exponent: i32,
        reduced_threshold_ld64: *mut i32,
        reduced_avoid_hole: *mut u8,
    ) -> i32;
    pub fn fdk_reduce_thresholds_vbr_test(
        energy_q31: *const i32,
        energy_ld64: *const i32,
        threshold_ld64: *const i32,
        minimum_snr_ld64: *const i32,
        form_factor_ld64: *const i32,
        sfb_offsets: *const i32,
        avoid_hole: *const u8,
        sfb_count: i32,
        quality_factor_q31: i32,
        chaos_measure_q31: *mut i32,
        instant_chaos_measure_q31: *mut i32,
        reduced_threshold_ld64: *mut i32,
        reduced_avoid_hole: *mut u8,
    ) -> i32;
    pub fn fdk_bitres_factor_test(
        current_pe: *const i32,
        reservoir_bits: *const i32,
        frames: i32,
        average_dynamic_bits: i32,
        maximum_reservoir_bits: i32,
        factor_q16: *mut i32,
    ) -> i32;
    pub fn fdk_aacenc_metadata_size_test() -> i32;
    pub fn fdk_metadata_compressor_test(
        input: *const i16,
        frames: i32,
        frame_length: i32,
        sample_rate: u32,
        channels: i32,
        channel_mode: i32,
        drc_profile: i32,
        compression_profile: i32,
        dialnorm_q16: i32,
        drc_target_q16: i32,
        compression_target_q16: i32,
        dynamic_range_q16: *mut i32,
        compression_q16: *mut i32,
    ) -> i32;
    pub fn fdk_pcm_downmix_7_1_test(
        input: *const i32,
        target_channels: i32,
        metadata: *const u8,
        metadata_bytes: i32,
        output: *mut i32,
        output_channels: *mut i32,
    ) -> i32;
    pub fn fdk_dct_iv_test(
        input: *const i32,
        length: i32,
        output: *mut i32,
        scale: *mut i32,
    ) -> i32;
    pub fn fdk_dst_iv_test(
        input: *const i32,
        length: i32,
        output: *mut i32,
        scale: *mut i32,
    ) -> i32;
    pub fn fdk_fft32_test(input: *const i32, output: *mut i32, scale: *mut i32) -> i32;
    pub fn fdk_fft32_capture_enable(enabled: i32);
    pub fn fdk_fft32_capture_get(stage: i32, output: *mut i32) -> i32;
    pub fn fdk_eld_filterbank_test(
        spectrum: *const i32,
        length: i32,
        spectrum_exp: i32,
        output: *mut i32,
    ) -> i32;
    pub fn fdk_sbr_inverse_filtered_patch_test(
        real: *const i32,
        imag: *const i32,
        total_samples: i32,
        mode: u8,
        previous_mode: u8,
        previous_bandwidth: i32,
        real_output: *mut i32,
        imag_output: *mut i32,
        high_band_scale: *mut i32,
    ) -> i32;
    pub fn fdk_sbr_autocorrelation2_test(
        real: *const i32,
        imag: *const i32,
        total_samples: i32,
        coefficients: *mut i32,
        det_scale: *mut i32,
    ) -> i32;
    pub fn fdk_fixed_mul_div2_test(
        left: *const i32,
        right: *const i32,
        count: i32,
        output: *mut i32,
    ) -> i32;
    pub fn fdk_sbr_inverse_filter_levels_test(
        modes: *const u8,
        previous_modes: *const u8,
        previous_bandwidths: *const i32,
        count: i32,
        bandwidths: *mut i32,
    ) -> i32;
    pub fn fdk_sbr_component_energies_test(
        reference_m: i32,
        reference_e: i8,
        estimated_m: i32,
        estimated_e: i8,
        noise_m: i32,
        noise_e: i8,
        sine_present: u8,
        sine_mapped: u8,
        no_noise: i32,
        gain_m: *mut i32,
        gain_e: *mut i8,
        noise_level_m: *mut i32,
        noise_level_e: *mut i8,
        sine_m: *mut i32,
        sine_e: *mut i8,
    ) -> i32;
    pub fn fdk_sbr_limited_components_test(
        reference_m: *const i32,
        reference_e: *const i8,
        estimated_m: *const i32,
        estimated_e: *const i8,
        noise_m: *const i32,
        noise_e: *const i8,
        sine_present: *const u8,
        sine_mapped: *const u8,
        count: i32,
        limiter_gains: u8,
        no_noise: i32,
        gain_m: *mut i32,
        gain_e: *mut i8,
        noise_level_m: *mut i32,
        noise_level_e: *mut i8,
        sine_m: *mut i32,
        sine_e: *mut i8,
    ) -> i32;
    pub fn fdk_sbr_limiter_bands_test(
        low_table: *const u8,
        low_bands: u8,
        source_starts: *const u8,
        target_starts: *const u8,
        band_counts: *const u8,
        patch_count: u8,
        limiter_bands: u8,
        result_count: *mut u8,
        result_table: *mut u8,
    ) -> i32;
    pub fn fdk_qmf_analysis32_test(
        input: *const i32,
        samples: i32,
        real_out: *mut i32,
        imag_out: *mut i32,
        lb_scale: *mut i32,
    ) -> i32;
    pub fn fdk_qmf_analysis_usac_test(
        input: *const i32,
        samples: i32,
        channels: i32,
        real_out: *mut i32,
        imag_out: *mut i32,
        lb_scale: *mut i32,
    ) -> i32;
    pub fn fdk_qmf_analysis64_test(
        input: *const i32,
        samples: i32,
        real_out: *mut i32,
        imag_out: *mut i32,
        lb_scale: *mut i32,
    ) -> i32;
    pub fn fdk_qmf_analysis64_cldfb_test(
        input: *const i32,
        samples: i32,
        real_out: *mut i32,
        imag_out: *mut i32,
        lb_scale: *mut i32,
    ) -> i32;
    pub fn fdk_qmf_analysis64_cldfb_pcm_test(
        input: *const i16,
        samples: i32,
        real_out: *mut i32,
        imag_out: *mut i32,
        lb_scale: *mut i32,
    ) -> i32;
    pub fn fdk_qmf_analysis32_lp_test(
        input: *const i32,
        samples: i32,
        real_out: *mut i32,
        lb_scale: *mut i32,
    ) -> i32;
    pub fn fdk_qmf_analysis32_cldfb_test(
        input: *const i32,
        samples: i32,
        real_out: *mut i32,
        imag_out: *mut i32,
        lb_scale: *mut i32,
    ) -> i32;
    pub fn fdk_qmf_analysis32_cldfb_lp_test(
        input: *const i32,
        samples: i32,
        real_out: *mut i32,
        lb_scale: *mut i32,
    ) -> i32;
    pub fn fdk_qmf_synthesis64_test(
        real_in: *const i32,
        imag_in: *const i32,
        slots: i32,
        output: *mut i32,
    ) -> i32;
    pub fn fdk_qmf_synthesis64_lp_test(real_in: *const i32, slots: i32, output: *mut i32) -> i32;
    pub fn fdk_qmf_synthesis32_cldfb_test(
        real_in: *const i32,
        imag_in: *const i32,
        slots: i32,
        output: *mut i32,
    ) -> i32;
    pub fn fdk_qmf_synthesis32_cldfb_lp_test(
        real_in: *const i32,
        slots: i32,
        output: *mut i32,
    ) -> i32;
    pub fn fdk_qmf_synthesis64_cldfb_test(
        real_in: *const i32,
        imag_in: *const i32,
        slots: i32,
        output: *mut i32,
    ) -> i32;
    pub fn fdk_qmf_roundtrip32_64_test(input: *const i32, samples: i32, output: *mut i32) -> i32;
    pub fn fdk_qmf_roundtrip32_cldfb_test(input: *const i32, samples: i32, output: *mut i32)
        -> i32;
}

pub type HANDLE_AACENCODER = *mut AACENCODER;
pub type HANDLE_AACDECODER = *mut AAC_DECODER_INSTANCE;

#[repr(C)]
pub struct AACENCODER {
    _private: [u8; 0],
}

#[repr(C)]
pub struct AAC_DECODER_INSTANCE {
    _private: [u8; 0],
}

pub type AACENC_ERROR = i32;
pub const AACENC_OK: AACENC_ERROR = 0x0000;
pub const AACENC_INVALID_HANDLE: AACENC_ERROR = 0x0020;
pub const AACENC_MEMORY_ERROR: AACENC_ERROR = 0x0021;
pub const AACENC_UNSUPPORTED_PARAMETER: AACENC_ERROR = 0x0022;
pub const AACENC_INVALID_CONFIG: AACENC_ERROR = 0x0023;
pub const AACENC_INIT_ERROR: AACENC_ERROR = 0x0040;
pub const AACENC_INIT_AAC_ERROR: AACENC_ERROR = 0x0041;
pub const AACENC_INIT_SBR_ERROR: AACENC_ERROR = 0x0042;
pub const AACENC_INIT_TP_ERROR: AACENC_ERROR = 0x0043;
pub const AACENC_INIT_META_ERROR: AACENC_ERROR = 0x0044;
pub const AACENC_INIT_MPS_ERROR: AACENC_ERROR = 0x0045;
pub const AACENC_ENCODE_ERROR: AACENC_ERROR = 0x0060;
pub const AACENC_ENCODE_EOF: AACENC_ERROR = 0x0080;

pub type AAC_DECODER_ERROR = i32;
pub const AAC_DEC_OK: AAC_DECODER_ERROR = 0x0000;
pub const AAC_DEC_OUT_OF_MEMORY: AAC_DECODER_ERROR = 0x0002;
pub const AAC_DEC_UNKNOWN: AAC_DECODER_ERROR = 0x0005;
pub const AAC_DEC_INVALID_HANDLE: AAC_DECODER_ERROR = 0x2001;
pub const AAC_DEC_UNSUPPORTED_FORMAT: AAC_DECODER_ERROR = 0x2003;
pub const AAC_DEC_OUTPUT_BUFFER_TOO_SMALL: AAC_DECODER_ERROR = 0x200C;
pub const AAC_DEC_NOT_ENOUGH_BITS: AAC_DECODER_ERROR = 0x1002;

pub type TRANSPORT_TYPE = i32;
pub const TT_UNKNOWN: TRANSPORT_TYPE = -1;
pub const TT_MP4_RAW: TRANSPORT_TYPE = 0;
pub const TT_MP4_ADIF: TRANSPORT_TYPE = 1;
pub const TT_MP4_ADTS: TRANSPORT_TYPE = 2;
pub const TT_MP4_LATM_MCP1: TRANSPORT_TYPE = 6;
pub const TT_MP4_LATM_MCP0: TRANSPORT_TYPE = 7;
pub const TT_MP4_LOAS: TRANSPORT_TYPE = 10;
pub const TT_DRM: TRANSPORT_TYPE = 12;

pub type AUDIO_OBJECT_TYPE = i32;
pub const AOT_AAC_LC: AUDIO_OBJECT_TYPE = 2;
pub const AOT_SBR: AUDIO_OBJECT_TYPE = 5;
pub const AOT_PS: AUDIO_OBJECT_TYPE = 29;
pub const AOT_ER_AAC_ELD: AUDIO_OBJECT_TYPE = 39;

pub type CHANNEL_MODE = i32;
pub const MODE_1: CHANNEL_MODE = 1;
pub const MODE_2: CHANNEL_MODE = 2;
pub const MODE_1_2: CHANNEL_MODE = 3;
pub const MODE_1_2_1: CHANNEL_MODE = 4;
pub const MODE_1_2_2: CHANNEL_MODE = 5;
pub const MODE_1_2_2_1: CHANNEL_MODE = 6;

pub type AACENC_PARAM = i32;
pub const AACENC_AOT: AACENC_PARAM = 0x0100;
pub const AACENC_BITRATE: AACENC_PARAM = 0x0101;
pub const AACENC_BITRATEMODE: AACENC_PARAM = 0x0102;
pub const AACENC_SAMPLERATE: AACENC_PARAM = 0x0103;
pub const AACENC_SBR_MODE: AACENC_PARAM = 0x0104;
pub const AACENC_GRANULE_LENGTH: AACENC_PARAM = 0x0105;
pub const AACENC_CHANNELMODE: AACENC_PARAM = 0x0106;
pub const AACENC_CHANNELORDER: AACENC_PARAM = 0x0107;
pub const AACENC_SBR_RATIO: AACENC_PARAM = 0x0108;
pub const AACENC_AFTERBURNER: AACENC_PARAM = 0x0200;
pub const AACENC_BANDWIDTH: AACENC_PARAM = 0x0203;
pub const AACENC_PEAK_BITRATE: AACENC_PARAM = 0x0207;
pub const AACENC_TRANSMUX: AACENC_PARAM = 0x0300;
pub const AACENC_HEADER_PERIOD: AACENC_PARAM = 0x0301;
pub const AACENC_SIGNALING_MODE: AACENC_PARAM = 0x0302;
pub const AACENC_TPSUBFRAMES: AACENC_PARAM = 0x0303;
pub const AACENC_AUDIOMUXVER: AACENC_PARAM = 0x0304;
pub const AACENC_PROTECTION: AACENC_PARAM = 0x0306;
pub const AACENC_ANCILLARY_BITRATE: AACENC_PARAM = 0x0500;
pub const AACENC_METADATA_MODE: AACENC_PARAM = 0x0600;
pub const AACENC_CONTROL_STATE: AACENC_PARAM = 0xff00;

pub const IN_AUDIO_DATA: i32 = 0;
pub const IN_ANCILLRY_DATA: i32 = 1;
pub const IN_METADATA_SETUP: i32 = 2;
pub const OUT_BITSTREAM_DATA: i32 = 3;

pub type AACENC_METADATA_DRC_PROFILE = i32;
pub const AACENC_METADATA_DRC_NONE: AACENC_METADATA_DRC_PROFILE = 0;
pub const AACENC_METADATA_DRC_FILMSTANDARD: AACENC_METADATA_DRC_PROFILE = 1;
pub const AACENC_METADATA_DRC_FILMLIGHT: AACENC_METADATA_DRC_PROFILE = 2;
pub const AACENC_METADATA_DRC_MUSICSTANDARD: AACENC_METADATA_DRC_PROFILE = 3;
pub const AACENC_METADATA_DRC_MUSICLIGHT: AACENC_METADATA_DRC_PROFILE = 4;
pub const AACENC_METADATA_DRC_SPEECH: AACENC_METADATA_DRC_PROFILE = 5;
pub const AACENC_METADATA_DRC_NOT_PRESENT: AACENC_METADATA_DRC_PROFILE = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AACENC_ExtMetaData {
    pub extAncDataEnable: UCHAR,
    pub extDownmixLevelEnable: UCHAR,
    pub extDownmixLevel_A: UCHAR,
    pub extDownmixLevel_B: UCHAR,
    pub dmxGainEnable: UCHAR,
    pub dmxGain5: INT,
    pub dmxGain2: INT,
    pub lfeDmxEnable: UCHAR,
    pub lfeDmxLevel: UCHAR,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AACENC_MetaData {
    pub drc_profile: AACENC_METADATA_DRC_PROFILE,
    pub comp_profile: AACENC_METADATA_DRC_PROFILE,
    pub drc_TargetRefLevel: INT,
    pub comp_TargetRefLevel: INT,
    pub prog_ref_level_present: INT,
    pub prog_ref_level: INT,
    pub PCE_mixdown_idx_present: UCHAR,
    pub ETSI_DmxLvl_present: UCHAR,
    pub centerMixLevel: SCHAR,
    pub surroundMixLevel: SCHAR,
    pub dolbySurroundMode: UCHAR,
    pub drcPresentationMode: UCHAR,
    pub ExtMetaData: AACENC_ExtMetaData,
}

impl Default for AACENC_MetaData {
    fn default() -> Self {
        Self {
            drc_profile: AACENC_METADATA_DRC_NONE,
            comp_profile: AACENC_METADATA_DRC_NOT_PRESENT,
            drc_TargetRefLevel: -(31 << 16),
            comp_TargetRefLevel: -(23 << 16),
            prog_ref_level_present: 0,
            prog_ref_level: -(23 << 16),
            PCE_mixdown_idx_present: 0,
            ETSI_DmxLvl_present: 0,
            centerMixLevel: 0,
            surroundMixLevel: 0,
            dolbySurroundMode: 0,
            drcPresentationMode: 0,
            ExtMetaData: AACENC_ExtMetaData::default(),
        }
    }
}

pub const AACDEC_CONCEAL: UINT = 1;
pub const AACDEC_FLUSH: UINT = 2;
pub const AACDEC_INTR: UINT = 4;
pub const AACDEC_CLRHIST: UINT = 8;

pub type AACDEC_PARAM = i32;
pub const AAC_PCM_DUAL_CHANNEL_OUTPUT_MODE: AACDEC_PARAM = 0x0002;
pub const AAC_PCM_OUTPUT_CHANNEL_MAPPING: AACDEC_PARAM = 0x0003;
pub const AAC_PCM_LIMITER_ENABLE: AACDEC_PARAM = 0x0004;
pub const AAC_PCM_LIMITER_ATTACK_TIME: AACDEC_PARAM = 0x0005;
pub const AAC_PCM_LIMITER_RELEAS_TIME: AACDEC_PARAM = 0x0006;
pub const AAC_PCM_MIN_OUTPUT_CHANNELS: AACDEC_PARAM = 0x0011;
pub const AAC_PCM_MAX_OUTPUT_CHANNELS: AACDEC_PARAM = 0x0012;
pub const AAC_METADATA_PROFILE: AACDEC_PARAM = 0x0020;
pub const AAC_METADATA_EXPIRY_TIME: AACDEC_PARAM = 0x0021;
pub const AAC_CONCEAL_METHOD: AACDEC_PARAM = 0x0100;
pub const AAC_DRC_BOOST_FACTOR: AACDEC_PARAM = 0x0200;
pub const AAC_DRC_ATTENUATION_FACTOR: AACDEC_PARAM = 0x0201;
pub const AAC_DRC_REFERENCE_LEVEL: AACDEC_PARAM = 0x0202;
pub const AAC_DRC_HEAVY_COMPRESSION: AACDEC_PARAM = 0x0203;
pub const AAC_DRC_DEFAULT_PRESENTATION_MODE: AACDEC_PARAM = 0x0204;
pub const AAC_DRC_ENC_TARGET_LEVEL: AACDEC_PARAM = 0x0205;
pub const AAC_UNIDRC_SET_EFFECT: AACDEC_PARAM = 0x0206;
pub const AAC_UNIDRC_ALBUM_MODE: AACDEC_PARAM = 0x0207;
pub const AAC_QMF_LOWPOWER: AACDEC_PARAM = 0x0300;
pub const AAC_TPDEC_CLEAR_BUFFER: AACDEC_PARAM = 0x0603;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AACENC_InfoStruct {
    pub maxOutBufBytes: UINT,
    pub maxAncBytes: UINT,
    pub inBufFillLevel: UINT,
    pub inputChannels: UINT,
    pub frameLength: UINT,
    pub nDelay: UINT,
    pub nDelayCore: UINT,
    pub confBuf: [UCHAR; 64],
    pub confSize: UINT,
}

impl Default for AACENC_InfoStruct {
    fn default() -> Self {
        Self {
            maxOutBufBytes: 0,
            maxAncBytes: 0,
            inBufFillLevel: 0,
            inputChannels: 0,
            frameLength: 0,
            nDelay: 0,
            nDelayCore: 0,
            confBuf: [0; 64],
            confSize: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AACENC_BufDesc {
    pub numBufs: INT,
    pub bufs: *mut *mut c_void,
    pub bufferIdentifiers: *mut INT,
    pub bufSizes: *mut INT,
    pub bufElSizes: *mut INT,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AACENC_InArgs {
    pub numInSamples: INT,
    pub numAncBytes: INT,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AACENC_OutArgs {
    pub numOutBytes: INT,
    pub numInSamples: INT,
    pub numAncBytes: INT,
    pub bitResState: INT,
}

pub type AUDIO_CHANNEL_TYPE = i32;

#[repr(C)]
#[derive(Debug)]
pub struct CStreamInfo {
    pub sampleRate: INT,
    pub frameSize: INT,
    pub numChannels: INT,
    pub pChannelType: *mut AUDIO_CHANNEL_TYPE,
    pub pChannelIndices: *mut UCHAR,
    pub aacSampleRate: INT,
    pub profile: INT,
    pub aot: AUDIO_OBJECT_TYPE,
    pub channelConfig: INT,
    pub bitRate: INT,
    pub aacSamplesPerFrame: INT,
    pub aacNumChannels: INT,
    pub extAot: AUDIO_OBJECT_TYPE,
    pub extSamplingRate: INT,
    pub outputDelay: UINT,
    pub flags: UINT,
    pub epConfig: SCHAR,
    pub numLostAccessUnits: INT,
    pub numTotalBytes: INT64,
    pub numBadBytes: INT64,
    pub numTotalAccessUnits: INT64,
    pub numBadAccessUnits: INT64,
    pub drcProgRefLev: SCHAR,
    pub drcPresMode: SCHAR,
    pub outputLoudness: INT,
}

extern "C" {
    pub fn fdk_mps_encode_frame_test(
        sampling_rate: UINT,
        frame_length: UINT,
        left: *const INT_PCM,
        right: *const INT_PCM,
        payload: *mut UCHAR,
        payload_capacity: UINT,
        payload_bits: *mut UINT,
        downmix: *mut INT_PCM,
    ) -> INT;
    pub fn fdk_mps_encode_two_frames_test(
        sampling_rate: UINT,
        frame_length: UINT,
        left: *const INT_PCM,
        right: *const INT_PCM,
        payload: *mut UCHAR,
        payload_stride: UINT,
        payload_bits: *mut UINT,
    ) -> INT;
    pub fn fdk_mps_last_parameters_test(cld: *mut SCHAR, icc: *mut SCHAR, capacity: UINT) -> INT;
    pub fn fdk_sbr_frequency_tables_test(
        sampling_rate: UINT,
        start_frequency: UCHAR,
        stop_frequency: UCHAR,
        crossover_band: UCHAR,
        frequency_scale: UCHAR,
        alter_scale: UCHAR,
        noise_bands: UCHAR,
        low_bands: *mut UCHAR,
        high_bands: *mut UCHAR,
        noise_band_count: *mut UCHAR,
        low_table: *mut UCHAR,
        high_table: *mut UCHAR,
    ) -> INT;
    pub fn aacEncOpen(
        phAacEncoder: *mut HANDLE_AACENCODER,
        encModules: UINT,
        maxChannels: UINT,
    ) -> AACENC_ERROR;
    pub fn aacEncClose(phAacEncoder: *mut HANDLE_AACENCODER) -> AACENC_ERROR;
    pub fn aacEncoder_SetParam(
        hAacEncoder: HANDLE_AACENCODER,
        param: AACENC_PARAM,
        value: UINT,
    ) -> AACENC_ERROR;
    pub fn aacEncoder_GetParam(hAacEncoder: HANDLE_AACENCODER, param: AACENC_PARAM) -> UINT;
    pub fn aacEncEncode(
        hAacEncoder: HANDLE_AACENCODER,
        inBufDesc: *const AACENC_BufDesc,
        outBufDesc: *const AACENC_BufDesc,
        inargs: *const AACENC_InArgs,
        outargs: *mut AACENC_OutArgs,
    ) -> AACENC_ERROR;
    pub fn aacEncInfo(
        hAacEncoder: HANDLE_AACENCODER,
        pInfo: *mut AACENC_InfoStruct,
    ) -> AACENC_ERROR;

    pub fn aacDecoder_Open(transportFmt: TRANSPORT_TYPE, nrOfLayers: UINT) -> HANDLE_AACDECODER;
    pub fn aacDecoder_ConfigRaw(
        self_: HANDLE_AACDECODER,
        conf: *mut *mut UCHAR,
        length: *const UINT,
    ) -> AAC_DECODER_ERROR;
    pub fn aacDecoder_Fill(
        self_: HANDLE_AACDECODER,
        pBuffer: *mut *mut UCHAR,
        bufferSize: *const UINT,
        bytesValid: *mut UINT,
    ) -> AAC_DECODER_ERROR;
    pub fn aacDecoder_DecodeFrame(
        self_: HANDLE_AACDECODER,
        pTimeData: *mut INT_PCM,
        timeDataSize: INT,
        flags: UINT,
    ) -> AAC_DECODER_ERROR;
    pub fn aacDecoder_SetParam(
        self_: HANDLE_AACDECODER,
        param: AACDEC_PARAM,
        value: INT,
    ) -> AAC_DECODER_ERROR;
    pub fn aacDecoder_Close(self_: HANDLE_AACDECODER);
    pub fn aacDecoder_GetStreamInfo(self_: HANDLE_AACDECODER) -> *mut CStreamInfo;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opens_and_closes_encoder() {
        let mut handle: HANDLE_AACENCODER = std::ptr::null_mut();
        let err = unsafe { aacEncOpen(&mut handle, 0, 2) };
        assert_eq!(err, AACENC_OK);
        assert!(!handle.is_null());
        let err = unsafe { aacEncClose(&mut handle) };
        assert_eq!(err, AACENC_OK);
        assert!(handle.is_null());
    }

    #[test]
    fn opens_and_closes_decoder() {
        let handle = unsafe { aacDecoder_Open(TT_MP4_ADTS, 1) };
        assert!(!handle.is_null());
        unsafe { aacDecoder_Close(handle) };
    }
}
