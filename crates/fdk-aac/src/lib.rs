//! Safe Rust entry points for the Fraunhofer FDK AAC codec.
//!
//! Pure Rust AAC codec and transport implementation, with optional legacy FDK
//! C/C++ API wrappers behind the `ffi` feature.

#![allow(clippy::cast_possible_truncation)]

pub mod aac_encoder;
pub mod adif;
pub mod adts;
pub mod asc;
pub mod audio_preroll;
pub mod bits;
pub mod concealment;
pub mod decoder;
pub mod drc;
pub mod drm;
pub mod eld_analysis;
pub mod encoder;
pub mod encoder_metadata;
pub mod filterbank;
pub mod fixed;
mod fixed_fft;
pub mod hcr;
pub mod huffman;
pub mod huffman_tables;
pub mod ics;
pub mod inverse;
pub mod latm;
pub mod ld_filterbank;
pub mod ld_sbr;
pub mod ld_sbr_qmf;
pub mod limiter;
pub mod loas;
pub mod pns;
pub mod ps;
pub mod ps_encoder;
pub mod pulse;
pub mod raw;
pub mod rvlc;
pub mod sac;
pub mod sbr;
pub mod sbr_encoder;
pub mod scalefactor;
pub mod section;
pub mod sfb;
pub mod spectral;
pub mod stereo;
pub mod tns;
pub mod transport;
pub mod usac;
pub mod usac_acelp;
pub mod usac_arith;
pub mod usac_decoder;
pub mod usac_fac;
pub mod usac_fd;
pub mod usac_lpc;
pub mod usac_lpd;
pub mod usac_mps;
pub mod usac_sbr;
pub mod usac_stereo;
pub mod usac_tcx;

#[cfg(feature = "ffi")]
#[deny(clippy::cast_possible_truncation)]
mod ffi {
    use crate::adts::AdtsHeader;
    use std::{ffi::c_void, fmt, ptr};

    pub use fdk_aac_sys as sys;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct EncoderError(pub sys::AACENC_ERROR);

    impl fmt::Display for EncoderError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "FDK AAC encoder error 0x{:04x}", self.0)
        }
    }

    impl std::error::Error for EncoderError {}

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct DecoderError(pub sys::AAC_DECODER_ERROR);

    impl fmt::Display for DecoderError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "FDK AAC decoder error 0x{:04x}", self.0)
        }
    }

    impl std::error::Error for DecoderError {}

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum AudioObjectType {
        AacLc,
        HeAac,
        HeAacV2,
        AacEld,
        Other(u32),
    }

    impl AudioObjectType {
        fn as_raw(self) -> u32 {
            match self {
                Self::AacLc => sys::AOT_AAC_LC as u32,
                Self::HeAac => sys::AOT_SBR as u32,
                Self::HeAacV2 => sys::AOT_PS as u32,
                Self::AacEld => sys::AOT_ER_AAC_ELD as u32,
                Self::Other(value) => value,
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ChannelMode {
        Mono,
        Stereo,
        Three,
        Four,
        Five,
        FiveOne,
        Other(u32),
    }

    impl ChannelMode {
        fn as_raw(self) -> u32 {
            match self {
                Self::Mono => sys::MODE_1 as u32,
                Self::Stereo => sys::MODE_2 as u32,
                Self::Three => sys::MODE_1_2 as u32,
                Self::Four => sys::MODE_1_2_1 as u32,
                Self::Five => sys::MODE_1_2_2 as u32,
                Self::FiveOne => sys::MODE_1_2_2_1 as u32,
                Self::Other(value) => value,
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum TransportType {
        Raw,
        Adif,
        Adts,
        LatmMuxConfigPresent,
        LatmOutOfBandConfig,
        Loas,
        Drm,
    }

    impl TransportType {
        fn as_raw(self) -> sys::TRANSPORT_TYPE {
            match self {
                Self::Raw => sys::TT_MP4_RAW,
                Self::Adif => sys::TT_MP4_ADIF,
                Self::Adts => sys::TT_MP4_ADTS,
                Self::LatmMuxConfigPresent => sys::TT_MP4_LATM_MCP1,
                Self::LatmOutOfBandConfig => sys::TT_MP4_LATM_MCP0,
                Self::Loas => sys::TT_MP4_LOAS,
                Self::Drm => sys::TT_DRM,
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct EncoderConfig {
        pub channels: u32,
        pub sample_rate: u32,
        pub bitrate: u32,
        pub channel_mode: ChannelMode,
        pub audio_object_type: AudioObjectType,
        pub transport: TransportType,
        pub afterburner: bool,
        pub sbr_mode: Option<u32>,
    }

    impl EncoderConfig {
        pub fn aac_lc_stereo(sample_rate: u32, bitrate: u32) -> Self {
            Self {
                channels: 2,
                sample_rate,
                bitrate,
                channel_mode: ChannelMode::Stereo,
                audio_object_type: AudioObjectType::AacLc,
                transport: TransportType::Adts,
                afterburner: true,
                sbr_mode: None,
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct EncoderInfo {
        pub max_output_bytes: u32,
        pub max_ancillary_bytes: u32,
        pub input_channels: u32,
        pub frame_length: u32,
        pub delay: u32,
        pub core_delay: u32,
    }

    pub struct Encoder {
        handle: sys::HANDLE_AACENCODER,
        he_aac_v2_frame_samples: Option<usize>,
        pending_he_aac_v2_input: Vec<i16>,
        pending_he_aac_v2_metadata: Option<sys::AACENC_MetaData>,
    }

    impl Encoder {
        pub fn open(max_channels: u32) -> Result<Self, EncoderError> {
            let mut handle = ptr::null_mut();
            check_encoder(unsafe { sys::aacEncOpen(&mut handle, 0, max_channels) })?;
            Ok(Self {
                handle,
                he_aac_v2_frame_samples: None,
                pending_he_aac_v2_input: Vec::new(),
                pending_he_aac_v2_metadata: None,
            })
        }

        pub fn configured(config: &EncoderConfig) -> Result<Self, EncoderError> {
            let mut encoder = Self::open(config.channels)?;
            encoder.set_param(sys::AACENC_AOT, config.audio_object_type.as_raw())?;
            encoder.set_param(sys::AACENC_SAMPLERATE, config.sample_rate)?;
            encoder.set_param(sys::AACENC_CHANNELMODE, config.channel_mode.as_raw())?;
            encoder.set_param(sys::AACENC_CHANNELORDER, 1)?;
            encoder.set_param(sys::AACENC_BITRATE, config.bitrate)?;
            encoder.set_param(sys::AACENC_TRANSMUX, config.transport.as_raw() as u32)?;
            encoder.set_param(sys::AACENC_AFTERBURNER, u32::from(config.afterburner))?;
            if let Some(mode) = config.sbr_mode {
                encoder.set_param(sys::AACENC_SBR_MODE, mode)?;
            }
            encoder.initialize()?;
            Ok(encoder)
        }

        pub fn set_param(
            &mut self,
            param: sys::AACENC_PARAM,
            value: u32,
        ) -> Result<(), EncoderError> {
            check_encoder(unsafe { sys::aacEncoder_SetParam(self.handle, param, value) })?;
            self.he_aac_v2_frame_samples = None;
            self.pending_he_aac_v2_input.clear();
            self.pending_he_aac_v2_metadata = None;
            Ok(())
        }

        pub fn get_param(&self, param: sys::AACENC_PARAM) -> u32 {
            unsafe { sys::aacEncoder_GetParam(self.handle, param) }
        }

        pub fn initialize(&mut self) -> Result<(), EncoderError> {
            check_encoder(unsafe {
                sys::aacEncEncode(
                    self.handle,
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null_mut(),
                )
            })?;
            self.pending_he_aac_v2_input.clear();
            self.pending_he_aac_v2_metadata = None;
            self.he_aac_v2_frame_samples = if self.get_param(sys::AACENC_AOT) == sys::AOT_PS as u32
            {
                let info = self.info()?;
                Some(
                    usize::try_from(
                        info.frame_length
                            .checked_mul(info.input_channels)
                            .ok_or_else(invalid_encoder_input)?,
                    )
                    .map_err(|_| invalid_encoder_input())?,
                )
            } else {
                None
            };
            Ok(())
        }

        pub fn info(&self) -> Result<EncoderInfo, EncoderError> {
            let mut raw = sys::AACENC_InfoStruct::default();
            check_encoder(unsafe { sys::aacEncInfo(self.handle, &mut raw) })?;
            Ok(EncoderInfo {
                max_output_bytes: raw.maxOutBufBytes,
                max_ancillary_bytes: raw.maxAncBytes,
                input_channels: raw.inputChannels,
                frame_length: raw.frameLength,
                delay: raw.nDelay,
                core_delay: raw.nDelayCore,
            })
        }

        pub fn audio_specific_config(&self) -> Result<Vec<u8>, EncoderError> {
            let mut raw = sys::AACENC_InfoStruct::default();
            check_encoder(unsafe { sys::aacEncInfo(self.handle, &mut raw) })?;
            let size = usize::try_from(raw.confSize).map_err(|_| invalid_encoder_input())?;
            raw.confBuf
                .get(..size)
                .map(|bytes| bytes.to_vec())
                .ok_or_else(invalid_encoder_input)
        }

        pub fn encode_interleaved_i16(
            &mut self,
            input: &[i16],
            output: &mut [u8],
        ) -> Result<usize, EncoderError> {
            self.encode_interleaved_i16_with_ancillary(input, &[], output)
                .map(|(bytes, _)| bytes)
        }

        pub fn encode_interleaved_i16_with_ancillary(
            &mut self,
            input: &[i16],
            ancillary: &[u8],
            output: &mut [u8],
        ) -> Result<(usize, usize), EncoderError> {
            self.encode_interleaved_i16_with_ancillary_and_metadata(input, ancillary, None, output)
        }

        pub fn encode_interleaved_i16_with_ancillary_and_metadata(
            &mut self,
            input: &[i16],
            ancillary: &[u8],
            metadata: Option<&sys::AACENC_MetaData>,
            output: &mut [u8],
        ) -> Result<(usize, usize), EncoderError> {
            // Upstream issue #129 documents an out-of-bounds deinterleave when
            // HE-AACv2 receives less than one complete input frame. Keep the
            // safe wrapper's incremental-input behavior, but do not expose the
            // vulnerable C path to a short slice.
            if let Some(frame_samples) = self.he_aac_v2_frame_samples {
                self.pending_he_aac_v2_input.extend_from_slice(input);
                if self.pending_he_aac_v2_input.len() < frame_samples {
                    if let Some(metadata) = metadata {
                        self.pending_he_aac_v2_metadata = Some(*metadata);
                    }
                    return Ok((0, 0));
                }
                let complete_frame = self
                    .pending_he_aac_v2_input
                    .drain(..frame_samples)
                    .collect::<Vec<_>>();
                let pending_metadata = self.pending_he_aac_v2_metadata.take();
                return self.encode_complete_interleaved_i16_with_ancillary_and_metadata(
                    &complete_frame,
                    ancillary,
                    metadata.or(pending_metadata.as_ref()),
                    output,
                );
            }
            self.encode_complete_interleaved_i16_with_ancillary_and_metadata(
                input, ancillary, metadata, output,
            )
        }

        fn encode_complete_interleaved_i16_with_ancillary_and_metadata(
            &mut self,
            input: &[i16],
            ancillary: &[u8],
            metadata: Option<&sys::AACENC_MetaData>,
            output: &mut [u8],
        ) -> Result<(usize, usize), EncoderError> {
            let in_ptr = input.as_ptr() as *mut c_void;
            let ancillary_ptr = ancillary.as_ptr() as *mut c_void;
            let metadata_ptr = metadata
                .map(|value| value as *const sys::AACENC_MetaData as *mut c_void)
                .unwrap_or(ptr::null_mut());
            let mut out_ptr = output.as_mut_ptr() as *mut c_void;

            let mut in_ptrs = [in_ptr, ancillary_ptr, metadata_ptr];
            let mut in_ids = [
                sys::IN_AUDIO_DATA,
                sys::IN_ANCILLRY_DATA,
                sys::IN_METADATA_SETUP,
            ];
            let mut out_id = sys::OUT_BITSTREAM_DATA;
            let mut in_sizes = [
                encoder_i32_len(std::mem::size_of_val(input))?,
                encoder_i32_len(std::mem::size_of_val(ancillary))?,
                encoder_i32_len(std::mem::size_of::<sys::AACENC_MetaData>())?,
            ];
            let mut out_size = encoder_i32_len(output.len())?;
            let mut in_element_sizes = [
                encoder_i32_len(std::mem::size_of::<i16>())?,
                encoder_i32_len(std::mem::size_of::<u8>())?,
                encoder_i32_len(std::mem::size_of::<sys::AACENC_MetaData>())?,
            ];
            let mut out_element_size = encoder_i32_len(std::mem::size_of::<u8>())?;

            let in_desc = sys::AACENC_BufDesc {
                numBufs: if metadata.is_some() { 3 } else { 2 },
                bufs: in_ptrs.as_mut_ptr(),
                bufferIdentifiers: in_ids.as_mut_ptr(),
                bufSizes: in_sizes.as_mut_ptr(),
                bufElSizes: in_element_sizes.as_mut_ptr(),
            };
            let out_desc = sys::AACENC_BufDesc {
                numBufs: 1,
                bufs: &mut out_ptr,
                bufferIdentifiers: &mut out_id,
                bufSizes: &mut out_size,
                bufElSizes: &mut out_element_size,
            };
            let in_args = sys::AACENC_InArgs {
                numInSamples: encoder_i32_len(input.len())?,
                numAncBytes: encoder_i32_len(ancillary.len())?,
            };
            let mut out_args = sys::AACENC_OutArgs::default();

            check_encoder(unsafe {
                sys::aacEncEncode(self.handle, &in_desc, &out_desc, &in_args, &mut out_args)
            })?;
            Ok((
                checked_encoder_count(out_args.numOutBytes, output.len())?,
                checked_encoder_count(out_args.numAncBytes, ancillary.len())?,
            ))
        }
    }

    impl Drop for Encoder {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                let _ = unsafe { sys::aacEncClose(&mut self.handle) };
            }
        }
    }

    pub struct Decoder {
        handle: sys::HANDLE_AACDECODER,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct StreamInfo {
        pub sample_rate: i32,
        pub frame_size: i32,
        pub channels: i32,
        pub output_delay: u32,
        pub flags: u32,
        pub drc_program_reference_level: i8,
        pub drc_presentation_mode: i8,
        pub output_loudness: i32,
    }

    impl Decoder {
        pub fn open(transport: TransportType) -> Result<Self, DecoderError> {
            let handle = unsafe { sys::aacDecoder_Open(transport.as_raw(), 1) };
            if handle.is_null() {
                Err(DecoderError(sys::AAC_DEC_OUT_OF_MEMORY))
            } else {
                Ok(Self { handle })
            }
        }

        /// Open a raw AAC decoder configured from a Pure Rust parsed ADTS header.
        ///
        /// This avoids FDK's C/C++ ADTS transport parser for configuration. The AAC
        /// core is still the upstream FDK decoder; callers should feed raw ADTS
        /// payloads, for example from [`adts::AdtsStream`].
        pub fn open_raw_from_adts(header: AdtsHeader) -> Result<Self, DecoderError> {
            let mut decoder = Self::open(TransportType::Raw)?;
            let mut asc = header
                .audio_specific_config()
                .map_err(|_| DecoderError(sys::AAC_DEC_UNSUPPORTED_FORMAT))?
                .to_bytes()
                .map_err(|_| DecoderError(sys::AAC_DEC_UNSUPPORTED_FORMAT))?;
            decoder.configure_raw(&mut asc)?;
            Ok(decoder)
        }

        pub fn configure_raw(&mut self, config: &mut [u8]) -> Result<(), DecoderError> {
            let mut ptr = config.as_mut_ptr();
            let len = decoder_u32_len(config.len())?;
            check_decoder(unsafe { sys::aacDecoder_ConfigRaw(self.handle, &mut ptr, &len) })
        }

        pub fn fill(&mut self, input: &mut [u8]) -> Result<usize, DecoderError> {
            let mut ptr = input.as_mut_ptr();
            let size = decoder_u32_len(input.len())?;
            let mut valid = size;
            check_decoder(unsafe {
                sys::aacDecoder_Fill(self.handle, &mut ptr, &size, &mut valid)
            })?;
            let consumed = size
                .checked_sub(valid)
                .ok_or(DecoderError(sys::AAC_DEC_UNKNOWN))?;
            Ok(usize::try_from(consumed).expect("u32 fits usize on supported Rust targets"))
        }

        pub fn decode_frame(&mut self, output: &mut [i16]) -> Result<(), DecoderError> {
            self.decode_frame_with_flags(output, 0)
        }

        pub fn decode_frame_with_flags(
            &mut self,
            output: &mut [i16],
            flags: u32,
        ) -> Result<(), DecoderError> {
            check_decoder(unsafe {
                sys::aacDecoder_DecodeFrame(
                    self.handle,
                    output.as_mut_ptr(),
                    decoder_i32_len(output.len())?,
                    flags,
                )
            })
        }

        pub fn set_parameter(
            &mut self,
            parameter: sys::AACDEC_PARAM,
            value: i32,
        ) -> Result<(), DecoderError> {
            check_decoder(unsafe { sys::aacDecoder_SetParam(self.handle, parameter, value) })
        }

        pub fn decode_access_unit_i16(
            &mut self,
            input: &[u8],
            output: &mut [i16],
        ) -> Result<usize, DecoderError> {
            let mut owned = input.to_vec();
            self.fill(&mut owned)?;
            self.decode_frame(output)?;
            let samples = self.stream_info().map_or(Ok(output.len()), |info| {
                info.frame_size
                    .checked_mul(info.channels)
                    .filter(|samples| *samples >= 0)
                    .and_then(|samples| usize::try_from(samples).ok())
                    .ok_or(DecoderError(sys::AAC_DEC_UNKNOWN))
            })?;
            Ok(samples.min(output.len()))
        }

        pub fn stream_info(&self) -> Option<StreamInfo> {
            let raw = unsafe { sys::aacDecoder_GetStreamInfo(self.handle).as_ref()? };
            Some(StreamInfo {
                sample_rate: raw.sampleRate,
                frame_size: raw.frameSize,
                channels: raw.numChannels,
                output_delay: raw.outputDelay,
                flags: raw.flags,
                drc_program_reference_level: raw.drcProgRefLev,
                drc_presentation_mode: raw.drcPresMode,
                output_loudness: raw.outputLoudness,
            })
        }
    }

    impl Drop for Decoder {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                unsafe { sys::aacDecoder_Close(self.handle) };
            }
        }
    }

    fn check_encoder(err: sys::AACENC_ERROR) -> Result<(), EncoderError> {
        if err == sys::AACENC_OK {
            Ok(())
        } else {
            Err(EncoderError(err))
        }
    }

    fn check_decoder(err: sys::AAC_DECODER_ERROR) -> Result<(), DecoderError> {
        if err == sys::AAC_DEC_OK {
            Ok(())
        } else {
            Err(DecoderError(err))
        }
    }

    fn invalid_encoder_input() -> EncoderError {
        EncoderError(sys::AACENC_INVALID_CONFIG)
    }

    fn encoder_i32_len(len: usize) -> Result<i32, EncoderError> {
        i32::try_from(len).map_err(|_| invalid_encoder_input())
    }

    fn checked_encoder_count(value: i32, capacity: usize) -> Result<usize, EncoderError> {
        let value = usize::try_from(value).map_err(|_| EncoderError(sys::AACENC_ENCODE_ERROR))?;
        if value > capacity {
            Err(EncoderError(sys::AACENC_ENCODE_ERROR))
        } else {
            Ok(value)
        }
    }

    fn decoder_u32_len(len: usize) -> Result<u32, DecoderError> {
        u32::try_from(len).map_err(|_| DecoderError(sys::AAC_DEC_UNSUPPORTED_FORMAT))
    }

    fn decoder_i32_len(len: usize) -> Result<i32, DecoderError> {
        i32::try_from(len).map_err(|_| DecoderError(sys::AAC_DEC_OUTPUT_BUFFER_TOO_SMALL))
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::aac_encoder::PureRustAacLcMonoEncoder;
        use crate::adts;
        use crate::asc::{
            AudioSpecificConfig, EldSpecificConfig, GaSpecificConfig, ProgramConfig, ProgramElement,
        };
        use crate::bits::{BitReader, BitWriter};
        use crate::decoder::AacLcDecoder;
        use crate::raw::ElementId;
        use crate::section::ZERO_HCB;

        #[test]
        fn ffi_lengths_reject_values_that_do_not_fit_c_types() {
            assert_eq!(
                encoder_i32_len(usize::MAX),
                Err(EncoderError(sys::AACENC_INVALID_CONFIG))
            );
            assert_eq!(
                decoder_i32_len(usize::MAX),
                Err(DecoderError(sys::AAC_DEC_OUTPUT_BUFFER_TOO_SMALL))
            );
            if usize::BITS > u32::BITS {
                assert_eq!(
                    decoder_u32_len(usize::MAX),
                    Err(DecoderError(sys::AAC_DEC_UNSUPPORTED_FORMAT))
                );
            }
        }

        #[test]
        fn ffi_counts_reject_negative_and_out_of_capacity_results() {
            assert_eq!(
                checked_encoder_count(-1, 32),
                Err(EncoderError(sys::AACENC_ENCODE_ERROR))
            );
            assert_eq!(
                checked_encoder_count(33, 32),
                Err(EncoderError(sys::AACENC_ENCODE_ERROR))
            );
            assert_eq!(checked_encoder_count(32, 32), Ok(32));
        }

        #[test]
        fn ffi_he_aac_v2_buffers_short_input_before_calling_c() {
            let mut config = EncoderConfig::aac_lc_stereo(48_000, 32_000);
            config.audio_object_type = AudioObjectType::HeAacV2;
            config.transport = TransportType::Raw;
            let mut encoder = Encoder::configured(&config).unwrap();
            let info = encoder.info().unwrap();
            let frame_samples = usize::try_from(info.frame_length * info.input_channels).unwrap();
            assert_eq!(encoder.he_aac_v2_frame_samples, Some(frame_samples));

            let mut output = vec![0; info.max_output_bytes as usize];
            let split = frame_samples / 3;
            assert_eq!(
                encoder
                    .encode_interleaved_i16(&vec![0; split], &mut output)
                    .unwrap(),
                0
            );
            assert_eq!(encoder.pending_he_aac_v2_input.len(), split);

            encoder
                .encode_interleaved_i16(&vec![0; frame_samples - split], &mut output)
                .unwrap();
            assert!(encoder.pending_he_aac_v2_input.is_empty());
        }
        use crate::spectral::decode_spectral_tuple;
        use crate::transport::{DecodeFrameFlags, DecoderParameter, PureRustTransportDecoder};

        #[test]
        fn configures_encoder() {
            let config = EncoderConfig::aac_lc_stereo(44_100, 128_000);
            let encoder = Encoder::configured(&config).unwrap();
            let info = encoder.info().unwrap();
            assert_eq!(info.input_channels, 2);
            assert!(info.frame_length > 0);
            assert!(info.max_output_bytes > 0);
        }

        #[test]
        fn configures_eld_sbr_encoder_and_exposes_asc() {
            let mut config = EncoderConfig::aac_lc_stereo(44_100, 32_000);
            config.channels = 1;
            config.channel_mode = ChannelMode::Mono;
            config.audio_object_type = AudioObjectType::AacEld;
            config.transport = TransportType::Raw;
            config.sbr_mode = Some(1);
            let encoder = Encoder::configured(&config).unwrap();
            let asc_bytes = encoder.audio_specific_config().unwrap();
            let asc = AudioSpecificConfig::parse(&asc_bytes).unwrap();
            assert_eq!(asc.audio_object_type, 39);
            assert!(asc.eld_specific.unwrap().sbr_present);
        }

        #[test]
        fn decodes_encoder_generated_eld_sbr_access_units_in_pure_rust() {
            let mut config = EncoderConfig::aac_lc_stereo(44_100, 32_000);
            config.channels = 1;
            config.channel_mode = ChannelMode::Mono;
            config.audio_object_type = AudioObjectType::AacEld;
            config.transport = TransportType::Raw;
            config.sbr_mode = Some(1);
            let mut encoder = Encoder::configured(&config).unwrap();
            let info = encoder.info().unwrap();
            let mut asc_bytes = encoder.audio_specific_config().unwrap();
            let asc = AudioSpecificConfig::parse(&asc_bytes).unwrap();
            let mut fdk = Decoder::open(TransportType::Raw).unwrap();
            fdk.configure_raw(&mut asc_bytes).unwrap();
            fdk.set_parameter(sys::AAC_QMF_LOWPOWER, 1).unwrap();
            let mut pure = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
            let mut pure_float = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
            pure.set_qmf_low_power(true);
            pure_float.set_qmf_low_power(true);
            let mut encoded = vec![0u8; info.max_output_bytes as usize];
            let mut fdk_pcm = vec![0i16; 4096];
            let mut decoded_access_units = 0;
            let mut fdk_sequence = Vec::new();
            let mut rust_sequence = Vec::new();
            let mut rust_float_sequence = Vec::new();
            for frame in 0..20 {
                let input = (0..info.frame_length as usize)
                    .map(|sample| {
                        let position = frame as f64 * info.frame_length as f64 + sample as f64;
                        (position * 2.0 * std::f64::consts::PI * 997.0 / 44_100.0).sin() * 2_000.0
                    })
                    .map(|sample| sample.round() as i16)
                    .collect::<Vec<_>>();
                let bytes = encoder
                    .encode_interleaved_i16(&input, &mut encoded)
                    .unwrap();
                if bytes == 0 {
                    continue;
                }
                let fdk_samples = fdk
                    .decode_access_unit_i16(&encoded[..bytes], &mut fdk_pcm)
                    .unwrap();
                let rust = pure
                    .decode_raw_data_block_fixed_interleaved_i16(&encoded[..bytes])
                    .unwrap_or_else(|error| panic!("encoded ELD-SBR frame {frame}: {error:?}"));
                let rust_float = pure_float
                    .decode_raw_data_block_interleaved_f32(&encoded[..bytes])
                    .unwrap_or_else(|error| {
                        panic!("float encoded ELD-SBR frame {frame}: {error:?}")
                    });
                assert_eq!(rust.len(), fdk_samples);
                assert!(rust
                    .iter()
                    .all(|sample| sample.unsigned_abs() <= i16::MAX as u16 + 1));
                if decoded_access_units >= 8 {
                    fdk_sequence.extend_from_slice(&fdk_pcm[..fdk_samples]);
                    rust_sequence.extend_from_slice(&rust);
                    rust_float_sequence.extend(rust_float.iter().map(|sample| f64::from(*sample)));
                }
                decoded_access_units += 1;
            }
            assert!(decoded_access_units >= 8);
            assert_eq!(
                fdk.stream_info().unwrap().output_delay as usize,
                PureRustTransportDecoder::from_audio_specific_config(&asc)
                    .unwrap()
                    .stream_info()
                    .output_delay
            );
            let mut best = (0isize, 0.0f64, 0.0f64);
            for lag in -2048isize..=2048 {
                let mut dot = 0.0;
                let mut rust_energy = 0.0;
                let mut fdk_energy = 0.0;
                for (index, &rust) in rust_sequence.iter().enumerate() {
                    let fdk_index = index as isize + lag;
                    if (0..fdk_sequence.len() as isize).contains(&fdk_index) {
                        let rust = f64::from(rust);
                        let fdk = f64::from(fdk_sequence[fdk_index as usize]);
                        dot += rust * fdk;
                        rust_energy += rust * rust;
                        fdk_energy += fdk * fdk;
                    }
                }
                let correlation = dot / (rust_energy * fdk_energy).sqrt();
                if correlation.abs() > best.1.abs() {
                    best = (lag, correlation, (rust_energy / fdk_energy).sqrt());
                }
            }
            assert!(best.1.is_finite());
            assert!(best.0.abs() <= 1, "fixed ELD-SBR lag {}", best.0);
            assert!(best.1 > 0.99, "fixed ELD-SBR correlation {}", best.1);
            let mut float_best = (0isize, 0.0f64, 0.0f64);
            for lag in -2048isize..=2048 {
                let mut dot = 0.0;
                let mut rust_energy = 0.0;
                let mut fdk_energy = 0.0;
                for (index, &rust) in rust_float_sequence.iter().enumerate() {
                    let fdk_index = index as isize + lag;
                    if (0..fdk_sequence.len() as isize).contains(&fdk_index) {
                        let fdk = f64::from(fdk_sequence[fdk_index as usize]);
                        dot += rust * fdk;
                        rust_energy += rust * rust;
                        fdk_energy += fdk * fdk;
                    }
                }
                let correlation = dot / (rust_energy * fdk_energy).sqrt();
                if correlation.abs() > float_best.1.abs() {
                    float_best = (lag, correlation, (rust_energy / fdk_energy).sqrt());
                }
            }
            assert!(
                float_best.0.abs() <= 1,
                "float ELD-SBR lag {}",
                float_best.0
            );
            assert!(
                float_best.1 > 0.99,
                "float ELD-SBR correlation {}",
                float_best.1
            );
            assert!(
                (0.10..=0.15).contains(&float_best.2),
                "float ELD-SBR RMS ratio {}",
                float_best.2
            );
            assert!(
                (0.95..=1.05).contains(&best.2),
                "encoder ELD-SBR best-lag RMS ratio {}, lag {}, correlation {}",
                best.2,
                best.0,
                best.1
            );
        }

        #[test]
        fn opens_decoder() {
            let decoder = Decoder::open(TransportType::Adts).unwrap();
            assert!(decoder.stream_info().is_some());
        }

        #[test]
        fn opens_raw_decoder_from_pure_rust_adts_header() {
            let header = adts::AdtsHeader::aac_lc(44_100, 2, 0).unwrap();
            let decoder = Decoder::open_raw_from_adts(header).unwrap();
            assert!(decoder.stream_info().is_some());
        }

        #[test]
        fn fdk_decoder_access_unit_helper_matches_pure_rust_zero_cpe_silence() {
            let payload = zero_cpe_payload_for_parity();
            let header = adts::AdtsHeader::aac_lc(44_100, 2, payload.len()).unwrap();
            let mut frame = vec![0; header.header_len()];
            header.write(&mut frame).unwrap();
            frame.extend_from_slice(&payload);

            let mut fdk = Decoder::open(TransportType::Adts).unwrap();
            let mut fdk_pcm = vec![123i16; 4096];
            let fdk_samples = fdk.decode_access_unit_i16(&frame, &mut fdk_pcm).unwrap();
            let fdk_pcm = &fdk_pcm[..fdk_samples];

            let mut pure = AacLcDecoder::from_adts_header(header).unwrap();
            let pure_pcm = pure.decode_adts_frame_interleaved_i16(&frame).unwrap();

            assert_eq!(fdk_pcm.len(), pure_pcm.len());
            assert!(fdk_pcm.iter().all(|sample| *sample == 0));
            assert_eq!(fdk_pcm, pure_pcm.as_slice());
        }

        #[test]
        fn raw_fdk_decoder_configured_from_pure_rust_adts_matches_pure_rust_payload_decode() {
            let payload = zero_cpe_payload_for_parity();
            let header = adts::AdtsHeader::aac_lc(44_100, 2, payload.len()).unwrap();

            let mut fdk = Decoder::open_raw_from_adts(header).unwrap();
            let mut fdk_pcm = vec![123i16; 4096];
            let fdk_samples = fdk.decode_access_unit_i16(&payload, &mut fdk_pcm).unwrap();
            let fdk_pcm = &fdk_pcm[..fdk_samples];

            let mut pure = AacLcDecoder::new(
                header.sampling_frequency_index,
                header.channel_configuration,
            )
            .unwrap();
            let pure_pcm = pure
                .decode_raw_data_block_multichannel_f32(&payload)
                .unwrap()
                .interleaved_i16();

            assert_eq!(fdk_pcm.len(), pure_pcm.len());
            assert!(fdk_pcm.iter().all(|sample| *sample == 0));
            assert_eq!(fdk_pcm, pure_pcm.as_slice());
        }

        #[test]
        fn fdk_and_pure_rust_limiter_parameters_report_the_same_delay() {
            let payload = zero_cpe_payload_for_parity();
            let header = adts::AdtsHeader::aac_lc(44_100, 2, payload.len()).unwrap();
            let mut frame = vec![0; header.header_len()];
            header.write(&mut frame).unwrap();
            frame.extend_from_slice(&payload);

            for (fdk_parameter, rust_parameter, value, expected_delay) in [
                (
                    sys::AAC_PCM_LIMITER_ENABLE,
                    DecoderParameter::PcmLimiterEnable,
                    0,
                    1024,
                ),
                (
                    sys::AAC_PCM_LIMITER_ATTACK_TIME,
                    DecoderParameter::PcmLimiterAttackTime,
                    1,
                    1068,
                ),
            ] {
                let mut fdk = Decoder::open(TransportType::Adts).unwrap();
                fdk.set_parameter(fdk_parameter, value).unwrap();
                let mut fdk_pcm = vec![0i16; 4096];
                fdk.decode_access_unit_i16(&frame, &mut fdk_pcm).unwrap();

                let mut pure = PureRustTransportDecoder::from_adts_header(header).unwrap();
                pure.set_parameter(rust_parameter, value).unwrap();
                pure.decode_interleaved_i16(&frame).unwrap();

                assert_eq!(fdk.stream_info().unwrap().output_delay, expected_delay);
                assert_eq!(pure.stream_info().output_delay, expected_delay as usize);
            }
        }

        #[test]
        fn fdk_and_pure_rust_explicit_energy_concealment_have_matching_timing() {
            fn correlation(left: &[i16], right: &[i16]) -> f64 {
                let (mut dot, mut left_energy, mut right_energy) = (0.0, 0.0, 0.0);
                for (&left, &right) in left.iter().zip(right) {
                    let left = f64::from(left);
                    let right = f64::from(right);
                    dot += left * right;
                    left_energy += left * left;
                    right_energy += right * right;
                }
                dot / (left_energy * right_energy).sqrt().max(f64::MIN_POSITIVE)
            }

            let mut encoder = PureRustAacLcMonoEncoder::new(4, 32_000, 16_000).unwrap();
            let access_units = (0..5)
                .map(|frame| {
                    let pcm = (0..1024)
                        .map(|sample| {
                            let position = frame * 1024 + sample;
                            (position as f32 * 0.041).sin() * 12_000.0
                        })
                        .collect::<Vec<_>>();
                    encoder.encode_raw_data_block(&pcm).unwrap()
                })
                .collect::<Vec<_>>();
            let asc = AudioSpecificConfig::aac_lc(44_100, 1).unwrap();
            let mut asc_bytes = asc.to_bytes().unwrap();

            let mut fdk = Decoder::open(TransportType::Raw).unwrap();
            fdk.configure_raw(&mut asc_bytes).unwrap();
            fdk.set_parameter(sys::AAC_PCM_LIMITER_ENABLE, 0).unwrap();
            fdk.set_parameter(sys::AAC_CONCEAL_METHOD, 2).unwrap();
            let mut pure = PureRustTransportDecoder::from_audio_specific_config(&asc).unwrap();
            pure.set_parameter(DecoderParameter::PcmLimiterEnable, 0)
                .unwrap();
            pure.set_parameter(DecoderParameter::ConcealMethod, 2)
                .unwrap();
            for access_unit in &access_units[..3] {
                fdk.fill(&mut access_unit.clone()).unwrap();
                fdk.decode_frame(&mut vec![0i16; 1024]).unwrap();
                pure.decode_interleaved_i16(access_unit).unwrap();
            }

            let mut fdk_delayed = vec![0i16; 1024];
            fdk.decode_frame_with_flags(&mut fdk_delayed, sys::AACDEC_CONCEAL)
                .unwrap();
            let mut fdk_interpolated = vec![0i16; 1024];
            fdk.fill(&mut access_units[3].clone()).unwrap();
            fdk.decode_frame(&mut fdk_interpolated).unwrap();

            let pure_delayed = pure
                .decode_interleaved_i16_with_flags(&[], DecodeFrameFlags::CONCEAL)
                .unwrap();
            let pure_interpolated = pure.decode_interleaved_i16(&access_units[3]).unwrap();

            assert!(fdk_delayed.iter().any(|sample| *sample != 0));
            assert!(pure_delayed.iter().any(|sample| *sample != 0));
            assert!(fdk_interpolated.iter().any(|sample| *sample != 0));
            assert!(pure_interpolated.iter().any(|sample| *sample != 0));
            assert!(correlation(&fdk_delayed, &pure_delayed).abs() > 0.75);
            assert!(correlation(&fdk_interpolated, &pure_interpolated).abs() > 0.65);
        }

        #[test]
        fn fdk_and_pure_rust_remaining_decoder_parameters_accept_the_same_ranges() {
            let header = adts::AdtsHeader::aac_lc(44_100, 2, 0).unwrap();
            for (fdk_parameter, rust_parameter, value, accepted) in [
                (
                    sys::AAC_METADATA_PROFILE,
                    DecoderParameter::MetadataProfile,
                    3,
                    true,
                ),
                (
                    sys::AAC_METADATA_PROFILE,
                    DecoderParameter::MetadataProfile,
                    4,
                    false,
                ),
                (
                    sys::AAC_METADATA_EXPIRY_TIME,
                    DecoderParameter::MetadataExpiryTime,
                    550,
                    true,
                ),
                (
                    sys::AAC_METADATA_EXPIRY_TIME,
                    DecoderParameter::MetadataExpiryTime,
                    -1,
                    false,
                ),
                (
                    sys::AAC_DRC_HEAVY_COMPRESSION,
                    DecoderParameter::DrcHeavyCompression,
                    1,
                    true,
                ),
                (
                    sys::AAC_DRC_HEAVY_COMPRESSION,
                    DecoderParameter::DrcHeavyCompression,
                    2,
                    false,
                ),
                (
                    sys::AAC_DRC_DEFAULT_PRESENTATION_MODE,
                    DecoderParameter::DrcDefaultPresentationMode,
                    -1,
                    true,
                ),
                (
                    sys::AAC_DRC_DEFAULT_PRESENTATION_MODE,
                    DecoderParameter::DrcDefaultPresentationMode,
                    3,
                    false,
                ),
                (
                    sys::AAC_DRC_ENC_TARGET_LEVEL,
                    DecoderParameter::DrcEncoderTargetLevel,
                    127,
                    true,
                ),
                (
                    sys::AAC_DRC_ENC_TARGET_LEVEL,
                    DecoderParameter::DrcEncoderTargetLevel,
                    128,
                    false,
                ),
                (
                    sys::AAC_QMF_LOWPOWER,
                    DecoderParameter::QmfLowPower,
                    -1,
                    true,
                ),
                (
                    sys::AAC_QMF_LOWPOWER,
                    DecoderParameter::QmfLowPower,
                    2,
                    false,
                ),
            ] {
                let mut fdk = Decoder::open_raw_from_adts(header).unwrap();
                let mut pure = PureRustTransportDecoder::from_adts_header(header).unwrap();
                assert_eq!(
                    fdk.set_parameter(fdk_parameter, value).is_ok(),
                    accepted,
                    "FDK parameter {fdk_parameter:#x} value {value}"
                );
                assert_eq!(
                    pure.set_parameter(rust_parameter, value).is_ok(),
                    accepted,
                    "Rust parameter {rust_parameter:?} value {value}"
                );
            }
        }

        #[test]
        fn fdk_and_pure_rust_er_aac_ld_nonzero_frame_have_matching_shape() {
            let (body, body_bits) = (0u16..=u16::MAX)
                .find_map(|candidate| {
                    let bytes = candidate.to_be_bytes();
                    let mut reader = BitReader::new(&bytes);
                    let tuple = decode_spectral_tuple(&mut reader, 1).ok()?;
                    tuple.iter().any(|&value| value != 0).then_some((
                        (candidate as u32) >> (16 - reader.bits_read()),
                        reader.bits_read(),
                    ))
                })
                .unwrap();
            let mut writer = BitWriter::new();
            writer.write(0, 4);
            writer.write(180, 8);
            writer.write_bool(false);
            writer.write(0, 2);
            writer.write_bool(false);
            writer.write(1, 6);
            writer.write_bool(false);
            writer.write(1, 4);
            writer.write(1, 5);
            writer.write_bool(false); // scalefactor delta zero
            writer.write_bool(false); // pulse
            writer.write_bool(false); // TNS
            writer.write_bool(false); // gain control
            writer.write(body, body_bits);
            let payload = writer.finish();
            let asc = AudioSpecificConfig {
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

            let mut config = asc.to_bytes().unwrap();
            let mut fdk = Decoder::open(TransportType::Raw).unwrap();
            fdk.configure_raw(&mut config).unwrap();
            let mut fdk_pcm = vec![0i16; 1024];
            let mut fdk_samples = 0;
            for _ in 0..6 {
                fdk_samples = fdk.decode_access_unit_i16(&payload, &mut fdk_pcm).unwrap();
            }
            fdk_pcm.truncate(fdk_samples);

            let mut pure = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
            let mut pure_pcm = Vec::new();
            for _ in 0..6 {
                pure_pcm = pure
                    .decode_raw_data_block_fixed_interleaved_i16(&payload)
                    .unwrap();
            }
            assert_eq!(fdk_pcm.len(), 512);
            assert_eq!(pure_pcm.len(), fdk_pcm.len());
            assert!(fdk_pcm.iter().any(|&sample| sample != 0));
            assert!(pure_pcm.iter().any(|&sample| sample != 0));
            let dot = fdk_pcm
                .iter()
                .zip(&pure_pcm)
                .map(|(&left, &right)| left as f64 * right as f64)
                .sum::<f64>();
            let fdk_energy = fdk_pcm
                .iter()
                .map(|&sample| (sample as f64).powi(2))
                .sum::<f64>();
            let pure_energy = pure_pcm
                .iter()
                .map(|&sample| (sample as f64).powi(2))
                .sum::<f64>();
            let correlation = dot / (fdk_energy * pure_energy).sqrt();
            assert!(correlation > 0.95, "LD synthesis correlation {correlation}");
            let rms_ratio = (pure_energy / fdk_energy).sqrt();
            assert!(
                (0.99..=1.01).contains(&rms_ratio),
                "LD synthesis RMS ratio {rms_ratio}"
            );
        }

        #[test]
        fn fdk_and_pure_rust_er_aac_eld_nonzero_frame_have_matching_shape() {
            let (body, body_bits) = (0u16..=u16::MAX)
                .find_map(|candidate| {
                    let bytes = candidate.to_be_bytes();
                    let mut reader = BitReader::new(&bytes);
                    let tuple = decode_spectral_tuple(&mut reader, 1).ok()?;
                    tuple.iter().any(|&value| value != 0).then_some((
                        (candidate as u32) >> (16 - reader.bits_read()),
                        reader.bits_read(),
                    ))
                })
                .unwrap();
            let mut writer = BitWriter::new();
            writer.write(180, 8); // global_gain
            writer.write(1, 6); // implicit-long ELD max_sfb
            writer.write(1, 4); // codebook 1
            writer.write(1, 5); // one long section
            writer.write_bool(false); // scalefactor delta zero
            writer.write_bool(false); // TNS absent
            writer.write(body, body_bits);
            let payload = writer.finish();
            let asc = AudioSpecificConfig {
                audio_object_type: 39,
                sampling_frequency_index: 4,
                sampling_frequency: 44_100,
                channel_configuration: 1,
                extension: None,
                ga_specific: None,
                eld_specific: Some(EldSpecificConfig {
                    frame_length_flag: false,
                    section_data_resilience: false,
                    scalefactor_data_resilience: false,
                    spectral_data_resilience: false,
                    sbr_present: false,
                    sbr_sampling_rate: false,
                    sbr_crc: false,
                    sbr_headers: Vec::new(),
                    extensions: Vec::new(),
                }),
                usac_config: None,
                error_protection_config: Some(0),
                program_config: None,
                bits_read: 0,
            };

            let mut config = asc.to_bytes().unwrap();
            let mut fdk = Decoder::open(TransportType::Raw).unwrap();
            fdk.configure_raw(&mut config).unwrap();
            let mut fdk_pcm = vec![0i16; 1024];
            let mut fdk_samples = 0;
            for _ in 0..6 {
                fdk_samples = fdk.decode_access_unit_i16(&payload, &mut fdk_pcm).unwrap();
            }
            fdk_pcm.truncate(fdk_samples);

            let mut pure = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
            let mut pure_pcm = Vec::new();
            for _ in 0..6 {
                pure_pcm = pure
                    .decode_raw_data_block_fixed_interleaved_i16(&payload)
                    .unwrap();
            }
            assert_eq!(fdk_pcm.len(), 512);
            assert_eq!(pure_pcm.len(), fdk_pcm.len());
            assert!(fdk_pcm.iter().any(|&sample| sample != 0));
            assert!(pure_pcm.iter().any(|&sample| sample != 0));
            let dot = fdk_pcm
                .iter()
                .zip(&pure_pcm)
                .map(|(&left, &right)| left as f64 * right as f64)
                .sum::<f64>();
            let fdk_energy = fdk_pcm
                .iter()
                .map(|&sample| (sample as f64).powi(2))
                .sum::<f64>();
            let pure_energy = pure_pcm
                .iter()
                .map(|&sample| (sample as f64).powi(2))
                .sum::<f64>();
            let correlation = dot / (fdk_energy * pure_energy).sqrt();
            assert!(
                correlation > 0.95,
                "ELD synthesis correlation {correlation}"
            );
            let rms_ratio = (pure_energy / fdk_energy).sqrt();
            assert!(
                (0.75..=1.25).contains(&rms_ratio),
                "ELD synthesis RMS ratio {rms_ratio}"
            );
        }

        #[test]
        fn fdk_and_pure_rust_adts_stream_match_two_zero_cpe_frames() {
            let payload = zero_cpe_payload_for_parity();
            let header = adts::AdtsHeader::aac_lc(44_100, 2, payload.len()).unwrap();
            let mut frame = vec![0; header.header_len()];
            header.write(&mut frame).unwrap();
            frame.extend_from_slice(&payload);
            let mut stream = frame.clone();
            stream.extend_from_slice(&frame);

            let mut fdk = Decoder::open(TransportType::Adts).unwrap();
            let mut fdk_frames = Vec::new();
            for parsed in adts::AdtsStream::new(&stream) {
                let parsed = parsed.unwrap();
                let mut pcm = vec![123i16; 4096];
                let samples = fdk.decode_access_unit_i16(parsed.bytes, &mut pcm).unwrap();
                pcm.truncate(samples);
                fdk_frames.push(pcm);
            }

            let mut pure = AacLcDecoder::from_adts_header(header).unwrap();
            let pure_frames = pure
                .decode_adts_stream_multichannel_interleaved_i16(&stream)
                .collect::<Result<Vec<_>, _>>()
                .unwrap();

            assert_eq!(fdk_frames.len(), 2);
            assert_eq!(fdk_frames, pure_frames);
            assert!(fdk_frames.iter().flatten().all(|sample| *sample == 0));
        }

        #[test]
        fn fdk_and_pure_rust_match_zero_sce_mono_silence() {
            let payload = zero_sce_payload_for_parity(0);
            let header = adts::AdtsHeader::aac_lc(44_100, 1, payload.len()).unwrap();
            let frame = adts_frame_bytes(header, &payload);

            let mut fdk = Decoder::open(TransportType::Adts).unwrap();
            let mut fdk_pcm = vec![123i16; 2048];
            let fdk_samples = fdk.decode_access_unit_i16(&frame, &mut fdk_pcm).unwrap();
            fdk_pcm.truncate(fdk_samples);

            let mut pure = AacLcDecoder::from_adts_header(header).unwrap();
            let pure_pcm = pure.decode_adts_frame_interleaved_i16(&frame).unwrap();

            assert_eq!(fdk_pcm, pure_pcm);
            assert!(fdk_pcm.iter().all(|sample| *sample == 0));
        }

        #[test]
        fn fdk_and_pure_rust_match_zero_sce_with_inband_pce_channel_config_zero() {
            let payload = pce_plus_zero_sce_payload_for_parity();
            let header = adts::AdtsHeader::aac_lc(44_100, 0, payload.len()).unwrap();
            let frame = adts_frame_bytes(header, &payload);

            let mut fdk = Decoder::open(TransportType::Adts).unwrap();
            let mut fdk_pcm = vec![123i16; 2048];
            let fdk_samples = fdk.decode_access_unit_i16(&frame, &mut fdk_pcm).unwrap();
            fdk_pcm.truncate(fdk_samples);

            let mut pure = AacLcDecoder::from_adts_header(header).unwrap();
            let pure_pcm = pure
                .decode_adts_frame_multichannel_interleaved_i16(&frame)
                .unwrap();

            assert_eq!(fdk_pcm, pure_pcm);
            assert!(fdk_pcm.iter().all(|sample| *sample == 0));
        }

        #[test]
        fn fdk_and_pure_rust_match_zero_sce_with_zero_frequency_cce() {
            let payload = zero_sce_with_zero_frequency_cce_payload_for_parity();
            let header = adts::AdtsHeader::aac_lc(44_100, 1, payload.len()).unwrap();
            let frame = adts_frame_bytes(header, &payload);

            let mut fdk = Decoder::open(TransportType::Adts).unwrap();
            let mut fdk_pcm = vec![123i16; 2048];
            let fdk_samples = fdk.decode_access_unit_i16(&frame, &mut fdk_pcm).unwrap();
            fdk_pcm.truncate(fdk_samples);

            let mut pure = AacLcDecoder::from_adts_header(header).unwrap();
            let pure_pcm = pure
                .decode_adts_frame_multichannel_interleaved_i16(&frame)
                .unwrap();

            assert_eq!(fdk_pcm, pure_pcm);
            assert!(fdk_pcm.iter().all(|sample| *sample == 0));
        }

        #[test]
        fn fdk_and_pure_rust_accept_nonzero_pulse_sce_fixture() {
            let payload = pulse_sce_payload_for_parity_smoke();
            let header = adts::AdtsHeader::aac_lc(44_100, 1, payload.len()).unwrap();
            let frame = adts_frame_bytes(header, &payload);

            let mut fdk = Decoder::open(TransportType::Adts).unwrap();
            let mut fdk_pcm = vec![0i16; 2048];
            let fdk_samples = fdk.decode_access_unit_i16(&frame, &mut fdk_pcm).unwrap();
            fdk_pcm.truncate(fdk_samples);

            let mut pure = AacLcDecoder::from_adts_header(header).unwrap();
            let pure_pcm = pure.decode_adts_frame_interleaved_i16(&frame).unwrap();

            assert_eq!(fdk_pcm.len(), pure_pcm.len());
            assert!(pure_pcm.iter().any(|sample| *sample != 0));
            let report = pcm_delta_report(&fdk_pcm, &pure_pcm);
            assert_eq!(report.samples, pure_pcm.len());
            assert!(report.max_abs_delta > 0);
        }

        #[test]
        fn fdk_and_pure_rust_apply_legacy_one_band_normalization_equally() {
            let mut encoder = PureRustAacLcMonoEncoder::new(4, 32_000, 16_000).unwrap();
            let input: Vec<_> = (0..1024)
                .map(|index| (index as f32 * 2.0 * std::f32::consts::PI / 31.0).sin() * 2_000.0)
                .collect();
            let plain = encoder.encode_raw_data_block(&input).unwrap();
            let normalized = insert_legacy_drc_before_end(&plain, 88);
            let decode_fdk = |payload: &[u8]| {
                let header = adts::AdtsHeader::aac_lc(44_100, 1, 0).unwrap();
                let mut decoder = Decoder::open_raw_from_adts(header).unwrap();
                decoder
                    .set_parameter(sys::AAC_PCM_LIMITER_ENABLE, 0)
                    .unwrap();
                decoder.set_parameter(sys::AAC_CONCEAL_METHOD, 1).unwrap();
                let mut pcm = vec![0i16; 2048];
                decoder.decode_access_unit_i16(payload, &mut pcm).unwrap();
                decoder.decode_access_unit_i16(&plain, &mut pcm).unwrap();
                let samples = decoder.decode_access_unit_i16(&plain, &mut pcm).unwrap();
                pcm.truncate(samples);
                pcm
            };
            let decode_pure = |payload: &[u8]| {
                let mut decoder = AacLcDecoder::new(4, 1).unwrap();
                decoder
                    .decode_raw_data_block_multichannel_fixed_interleaved_i16(payload)
                    .unwrap();
                decoder
                    .decode_raw_data_block_multichannel_fixed_interleaved_i16(&plain)
                    .unwrap();
                decoder
                    .decode_raw_data_block_multichannel_fixed_interleaved_i16(&plain)
                    .unwrap()
            };
            let rms = |samples: &[i16]| {
                (samples
                    .iter()
                    .map(|sample| f64::from(*sample).powi(2))
                    .sum::<f64>()
                    / samples.len() as f64)
                    .sqrt()
            };
            let fdk_ratio = rms(&decode_fdk(&normalized)) / rms(&decode_fdk(&plain));
            let pure_ratio = rms(&decode_pure(&normalized)) / rms(&decode_pure(&plain));
            let expected = 2.0f64.powf(-8.0 / 24.0); // -24 dB target vs -22 dB program
                                                     // FDK delays normalization by one AAC frame and smooths the step,
                                                     // so this frame lies between the old and requested gains.
            assert!(
                (expected..1.0).contains(&fdk_ratio),
                "FDK ratio {fdk_ratio}"
            );
            assert!(
                (expected..1.0).contains(&pure_ratio),
                "Rust ratio {pure_ratio}"
            );
            assert!(
                (fdk_ratio - pure_ratio).abs() < 0.01,
                "FDK ratio {fdk_ratio}, Rust ratio {pure_ratio}"
            );
        }

        #[test]
        fn fdk_and_pure_rust_report_legacy_output_loudness_equally() {
            let mut encoder = PureRustAacLcMonoEncoder::new(4, 32_000, 16_000).unwrap();
            let plain = encoder.encode_raw_data_block(&vec![0.0; 1024]).unwrap();
            let metadata = insert_legacy_drc_before_end(&plain, 88);
            let header = adts::AdtsHeader::aac_lc(44_100, 1, 0).unwrap();
            let mut fdk = Decoder::open_raw_from_adts(header).unwrap();
            let mut pure = AacLcDecoder::new(4, 1).unwrap();
            let mut pcm = vec![0i16; 2048];

            assert_eq!(fdk.stream_info().unwrap().output_loudness, -1);
            assert_eq!(pure.stream_info().output_loudness, -1);
            for payload in [&metadata[..], &plain[..], &plain[..]] {
                fdk.decode_access_unit_i16(payload, &mut pcm).unwrap();
                pure.decode_raw_data_block_multichannel_fixed_interleaved_i16(payload)
                    .unwrap();
            }
            let fdk_info = fdk.stream_info().unwrap();
            let pure_info = pure.stream_info();
            assert_eq!(fdk_info.drc_program_reference_level, 88);
            assert_eq!(pure_info.drc_program_reference_level, 88);
            assert_eq!(fdk_info.output_loudness, 96);
            assert_eq!(pure_info.output_loudness, 96);

            fdk.set_parameter(sys::AAC_DRC_REFERENCE_LEVEL, -1).unwrap();
            pure.set_drc_reference_level(None);
            fdk.decode_access_unit_i16(&plain, &mut pcm).unwrap();
            pure.decode_raw_data_block_multichannel_fixed_interleaved_i16(&plain)
                .unwrap();
            assert_eq!(fdk.stream_info().unwrap().output_loudness, 88);
            assert_eq!(pure.stream_info().output_loudness, 88);
        }

        #[test]
        fn fdk_and_pure_rust_report_dvb_presentation_modes_equally() {
            let mut encoder = PureRustAacLcMonoEncoder::new(4, 32_000, 16_000).unwrap();
            let plain = encoder.encode_raw_data_block(&vec![0.0; 1024]).unwrap();
            let header = adts::AdtsHeader::aac_lc(44_100, 1, 0).unwrap();

            for mode in 0..=2 {
                let payload = insert_dvb_ancillary_drc_before_end(&plain, mode, 0x90);
                let mut fdk = Decoder::open_raw_from_adts(header).unwrap();
                let mut pure = AacLcDecoder::new(4, 1).unwrap();
                let mut pcm = vec![0i16; 2048];
                fdk.set_parameter(sys::AAC_DRC_DEFAULT_PRESENTATION_MODE, 1)
                    .unwrap();
                pure.set_drc_default_presentation_mode(1);
                assert_eq!(fdk.stream_info().unwrap().drc_presentation_mode, -1);
                assert_eq!(pure.stream_info().drc_presentation_mode, -1);

                for access_unit in [&payload[..], &plain[..]] {
                    fdk.decode_access_unit_i16(access_unit, &mut pcm).unwrap();
                    pure.decode_raw_data_block_multichannel_fixed_interleaved_i16(access_unit)
                        .unwrap();
                }
                assert_eq!(fdk.stream_info().unwrap().drc_presentation_mode, mode as i8);
                assert_eq!(pure.stream_info().drc_presentation_mode, mode as i8);
            }
        }

        #[test]
        fn fdk_and_pure_rust_apply_legacy_multiband_spectral_drc_equally() {
            let mut encoder = PureRustAacLcMonoEncoder::new(4, 32_000, 16_000).unwrap();
            let input: Vec<_> = (0..1024)
                .map(|index| (index as f32 * 2.0 * std::f32::consts::PI / 31.0).sin() * 2_000.0)
                .collect();
            let plain = encoder.encode_raw_data_block(&input).unwrap();
            let compressed = insert_legacy_multiband_drc_before_end(&plain);
            let decode_fdk = |payload: &[u8]| {
                let header = adts::AdtsHeader::aac_lc(44_100, 1, 0).unwrap();
                let mut decoder = Decoder::open_raw_from_adts(header).unwrap();
                decoder
                    .set_parameter(sys::AAC_PCM_LIMITER_ENABLE, 0)
                    .unwrap();
                decoder.set_parameter(sys::AAC_CONCEAL_METHOD, 1).unwrap();
                decoder
                    .set_parameter(sys::AAC_DRC_ATTENUATION_FACTOR, 127)
                    .unwrap();
                let mut pcm = vec![0i16; 2048];
                let samples = decoder.decode_access_unit_i16(payload, &mut pcm).unwrap();
                pcm.truncate(samples);
                pcm
            };
            let decode_pure = |payload: &[u8]| {
                let mut decoder = AacLcDecoder::new(4, 1).unwrap();
                decoder.set_drc_attenuation_factor(127);
                decoder
                    .decode_raw_data_block_multichannel_fixed_interleaved_i16(payload)
                    .unwrap()
            };
            let decode_pure_f32 = |payload: &[u8]| {
                let mut decoder = AacLcDecoder::new(4, 1).unwrap();
                decoder.set_drc_attenuation_factor(127);
                decoder
                    .decode_raw_data_block_multichannel_f32(payload)
                    .unwrap()
                    .interleaved_f32()
            };
            let rms = |samples: &[i16]| {
                (samples
                    .iter()
                    .map(|sample| f64::from(*sample).powi(2))
                    .sum::<f64>()
                    / samples.len() as f64)
                    .sqrt()
            };
            let fdk_ratio = rms(&decode_fdk(&compressed)) / rms(&decode_fdk(&plain));
            let pure_ratio = rms(&decode_pure(&compressed)) / rms(&decode_pure(&plain));
            let rms_f32 = |samples: &[f32]| {
                (samples
                    .iter()
                    .map(|sample| f64::from(*sample).powi(2))
                    .sum::<f64>()
                    / samples.len() as f64)
                    .sqrt()
            };
            let pure_f32_ratio =
                rms_f32(&decode_pure_f32(&compressed)) / rms_f32(&decode_pure_f32(&plain));
            assert!((0.45..0.65).contains(&fdk_ratio), "FDK ratio {fdk_ratio}");
            assert!(
                (fdk_ratio - pure_ratio).abs() < 0.03,
                "FDK ratio {fdk_ratio}, Rust ratio {pure_ratio}, Rust f32 ratio {pure_f32_ratio}"
            );
        }

        #[test]
        fn fdk_and_pure_rust_apply_dvb_ancillary_heavy_compression_equally() {
            let mut encoder = PureRustAacLcMonoEncoder::new(4, 32_000, 16_000).unwrap();
            let input: Vec<_> = (0..1024)
                .map(|index| (index as f32 * 2.0 * std::f32::consts::PI / 31.0).sin() * 2_000.0)
                .collect();
            let plain = encoder.encode_raw_data_block(&input).unwrap();
            let compressed = insert_dvb_ancillary_drc_before_end(&plain, 0, 0x90);
            let decode_fdk = |payload: &[u8]| {
                let header = adts::AdtsHeader::aac_lc(44_100, 1, 0).unwrap();
                let mut decoder = Decoder::open_raw_from_adts(header).unwrap();
                decoder
                    .set_parameter(sys::AAC_PCM_LIMITER_ENABLE, 0)
                    .unwrap();
                decoder.set_parameter(sys::AAC_CONCEAL_METHOD, 1).unwrap();
                decoder
                    .set_parameter(sys::AAC_DRC_HEAVY_COMPRESSION, 1)
                    .unwrap();
                let mut pcm = vec![0i16; 2048];
                decoder.decode_access_unit_i16(&plain, &mut pcm).unwrap();
                decoder.decode_access_unit_i16(payload, &mut pcm).unwrap();
                let samples = decoder.decode_access_unit_i16(payload, &mut pcm).unwrap();
                pcm.truncate(samples);
                pcm
            };
            let decode_pure = |payload: &[u8]| {
                let mut decoder = AacLcDecoder::new(4, 1).unwrap();
                decoder.set_drc_heavy_compression(true);
                decoder
                    .decode_raw_data_block_multichannel_fixed_interleaved_i16(&plain)
                    .unwrap();
                decoder
                    .decode_raw_data_block_multichannel_fixed_interleaved_i16(payload)
                    .unwrap();
                decoder
                    .decode_raw_data_block_multichannel_fixed_interleaved_i16(payload)
                    .unwrap()
            };
            let rms = |samples: &[i16]| {
                (samples
                    .iter()
                    .map(|sample| f64::from(*sample).powi(2))
                    .sum::<f64>()
                    / samples.len() as f64)
                    .sqrt()
            };
            let fdk_ratio = rms(&decode_fdk(&compressed)) / rms(&decode_fdk(&plain));
            let pure_ratio = rms(&decode_pure(&compressed)) / rms(&decode_pure(&plain));
            assert!((0.45..0.55).contains(&fdk_ratio), "FDK ratio {fdk_ratio}");
            assert!(
                (fdk_ratio - pure_ratio).abs() < 0.03,
                "FDK ratio {fdk_ratio}, Rust ratio {pure_ratio}"
            );
        }

        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        struct PcmDeltaReport {
            samples: usize,
            max_abs_delta: i32,
            sum_abs_delta: i64,
        }

        fn pcm_delta_report(left: &[i16], right: &[i16]) -> PcmDeltaReport {
            let samples = left.len().min(right.len());
            let mut max_abs_delta = 0i32;
            let mut sum_abs_delta = 0i64;
            for index in 0..samples {
                let delta = (left[index] as i32 - right[index] as i32).abs();
                max_abs_delta = max_abs_delta.max(delta);
                sum_abs_delta += delta as i64;
            }
            PcmDeltaReport {
                samples,
                max_abs_delta,
                sum_abs_delta,
            }
        }

        fn zero_cpe_payload_for_parity() -> Vec<u8> {
            let mut writer = BitWriter::new();
            writer.write(ElementId::ChannelPair.bits() as u32, 3);
            writer.write(0, 4); // element_instance_tag
            writer.write_bool(true); // common_window
            write_shared_long_ics(&mut writer, 1);
            writer.write(0, 2); // ms_mask_present none
            write_zero_channel_stream(&mut writer, 1);
            write_zero_channel_stream(&mut writer, 1);
            writer.write(ElementId::End.bits() as u32, 3);
            writer.finish()
        }

        fn zero_sce_payload_for_parity(tag: u8) -> Vec<u8> {
            let mut writer = BitWriter::new();
            write_zero_sce_payload_bits(&mut writer, tag);
            writer.write(ElementId::End.bits() as u32, 3);
            writer.finish()
        }

        fn pce_plus_zero_sce_payload_for_parity() -> Vec<u8> {
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
            write_zero_sce_payload_bits(&mut writer, 0);
            writer.write(ElementId::End.bits() as u32, 3);
            writer.finish()
        }

        fn zero_sce_with_zero_frequency_cce_payload_for_parity() -> Vec<u8> {
            let mut writer = BitWriter::new();
            write_zero_sce_payload_bits(&mut writer, 0);
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
            writer.finish()
        }

        fn pulse_sce_payload_for_parity_smoke() -> Vec<u8> {
            let mut writer = BitWriter::new();
            writer.write(ElementId::SingleChannel.bits() as u32, 3);
            writer.write(0, 4); // element_instance_tag
            writer.write(100, 8); // global_gain
            write_shared_long_ics(&mut writer, 1);
            writer.write(ZERO_HCB as u32, 4);
            writer.write(1, 5);
            writer.write_bool(true); // pulse_data_present
            writer.write(0, 2); // number_pulse = one pulse
            writer.write(0, 6); // pulse_start_sfb
            writer.write(0, 5); // pulse_offset at first line
            writer.write(8, 4); // pulse_amp
            writer.write_bool(false); // tns_data_present
            writer.write_bool(false); // gain_control_data_present
            writer.write(ElementId::End.bits() as u32, 3);
            writer.finish()
        }

        fn insert_legacy_drc_before_end(raw: &[u8], program_reference_level: u8) -> Vec<u8> {
            let mut position = BitReader::new(raw);
            AacLcDecoder::new(4, 1)
                .unwrap()
                .decode_raw_data_block_multichannel_f32_from_reader(&mut position)
                .unwrap();
            let end_bit = position.bits_read();
            let mut source = BitReader::new(raw);
            let mut writer = BitWriter::new();
            for _ in 0..end_bit {
                writer.write_bool(source.read_bool().unwrap());
            }
            writer.write(ElementId::Fill.bits() as u32, 3);
            writer.write(3, 4);
            writer.write(0x0b, 4); // EXT_DYNAMIC_RANGE
            writer.write_bool(false); // PCE tag absent
            writer.write_bool(false); // excluded channels absent
            writer.write_bool(false); // one band
            writer.write_bool(true);
            writer.write(program_reference_level as u32, 7);
            writer.write_bool(false);
            writer.write(0, 8); // normalization only
            writer.write(ElementId::End.bits() as u32, 3);
            writer.byte_align();
            writer.finish()
        }

        fn insert_legacy_multiband_drc_before_end(raw: &[u8]) -> Vec<u8> {
            let mut position = BitReader::new(raw);
            AacLcDecoder::new(4, 1)
                .unwrap()
                .decode_raw_data_block_multichannel_f32_from_reader(&mut position)
                .unwrap();
            let mut source = BitReader::new(raw);
            let mut writer = BitWriter::new();
            for _ in 0..position.bits_read() {
                writer.write_bool(source.read_bool().unwrap());
            }
            writer.write(ElementId::Fill.bits() as u32, 3);
            writer.write(6, 4);
            writer.write(0x0b, 4); // EXT_DYNAMIC_RANGE
            writer.write_bool(false); // PCE tag absent
            writer.write_bool(false); // excluded channels absent
            writer.write_bool(true); // DRC bands present
            writer.write(1, 4); // two bands
            writer.write(0, 4); // interpolation scheme
            writer.write(31, 8); // bins 0..127
            writer.write(255, 8); // bins 128..1023
            writer.write_bool(false); // program reference absent
            writer.write(0x98, 8); // -6.02 dB in the low band
            writer.write(0, 8); // unity in the remaining spectrum
            writer.write(ElementId::End.bits() as u32, 3);
            writer.byte_align();
            writer.finish()
        }

        fn insert_dvb_ancillary_drc_before_end(
            raw: &[u8],
            presentation_mode: u8,
            compression_value: u8,
        ) -> Vec<u8> {
            let mut position = BitReader::new(raw);
            AacLcDecoder::new(4, 1)
                .unwrap()
                .decode_raw_data_block_multichannel_f32_from_reader(&mut position)
                .unwrap();
            let mut source = BitReader::new(raw);
            let mut writer = BitWriter::new();
            writer.write(ElementId::DataStream.bits() as u32, 3);
            writer.write(0, 4); // element_instance_tag
            writer.write_bool(true); // data_byte_align_flag
            writer.write(5, 8);
            writer.byte_align();
            writer.write(0xbc, 8); // DVB ancillary sync
            writer.write((0xc0 | ((presentation_mode & 3) << 2)) as u32, 8);
            writer.write(0x04, 8); // compression field present
            writer.write(0x01, 8); // reserved audio mode 0, compression on
            writer.write(compression_value as u32, 8);
            for _ in 0..position.bits_read() {
                writer.write_bool(source.read_bool().unwrap());
            }
            writer.write(ElementId::End.bits() as u32, 3);
            writer.byte_align();
            writer.finish()
        }

        fn adts_frame_bytes(header: adts::AdtsHeader, payload: &[u8]) -> Vec<u8> {
            let mut frame = vec![0; header.header_len()];
            header.write(&mut frame).unwrap();
            frame.extend_from_slice(payload);
            frame
        }

        fn write_zero_sce_payload_bits(writer: &mut BitWriter, tag: u8) {
            writer.write(ElementId::SingleChannel.bits() as u32, 3);
            writer.write(tag as u32, 4);
            writer.write(100, 8); // global_gain
            write_shared_long_ics(writer, 1);
            writer.write(ZERO_HCB as u32, 4);
            writer.write(1, 5);
            writer.write_bool(false); // pulse_data_present
            writer.write_bool(false); // tns_data_present
            writer.write_bool(false); // gain_control_data_present
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
    }
}

#[cfg(feature = "ffi")]
pub use ffi::*;
