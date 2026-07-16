//! Digital Radio Mondiale transport configuration and CRC primitives.

use std::fmt;

use crate::asc::{
    AscError, AudioSpecificConfig, Mps212Config, UsacConfig, UsacElementConfig, UsacSbrConfig,
};
use crate::bits::{BitError, BitReader, BitWriter};
use crate::decoder::{
    interleave_multichannel_f32, interleave_multichannel_i16, interleave_stereo_f32,
    interleave_stereo_i16_samples, AacLcDecoder, DecodeError,
};
use crate::ld_sbr_qmf::{LdSbrChannelProcessor, LdSbrProcessingError};
use crate::ps::{PsError, PsFrame, PsParser, PsQmfProcessor};
use crate::sac::{Sac212Decoder, SacDecodeError, SacError, SpatialSpecificConfig};
use crate::sbr::{
    SbrError, SbrFillPayload, SbrMonoFrame, SbrMonoFrameParser, SbrStereoFrame,
    SbrStereoFrameParser, EXT_SBR_DATA_CRC,
};
use crate::usac_decoder::UsacDecodeError;

const AAC_SAMPLE_RATES: [Option<(u8, u32)>; 8] = [
    Some((11, 8_000)),
    Some((9, 12_000)),
    Some((8, 16_000)),
    Some((6, 24_000)),
    None,
    Some((3, 48_000)),
    None,
    None,
];

const XHE_SAMPLE_RATES: [(u8, u32); 8] = [
    (0x1b, 9_600),
    (0x09, 12_000),
    (0x08, 16_000),
    (0x17, 19_200),
    (0x06, 24_000),
    (0x05, 32_000),
    (0x12, 38_400),
    (0x03, 48_000),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrmAudioCoding {
    Aac,
    Celp,
    Hvxc,
    XheAac,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrmAudioMode {
    Mono,
    ParametricStereo,
    Stereo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrmSurroundMode {
    FiveOne,
    SevenOne,
    StreamDefined,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DrmAudioConfig {
    pub audio_coding: DrmAudioCoding,
    pub sbr: bool,
    pub audio_mode: DrmAudioMode,
    pub sampling_frequency_index: u8,
    pub sampling_frequency: u32,
    pub text_flag: bool,
    pub enhancement_flag: bool,
    pub coder_field: u8,
    pub reserved: bool,
    pub channel_configuration: u8,
    pub parametric_stereo: bool,
    pub drm_surround: bool,
    pub surround_mode: Option<DrmSurroundMode>,
    pub samples_per_frame: Option<u16>,
    pub error_protection_config: Option<u8>,
    pub bits_read: usize,
}

impl DrmAudioConfig {
    pub fn aac(
        sampling_frequency: u32,
        audio_mode: DrmAudioMode,
        sbr: bool,
    ) -> Result<Self, DrmError> {
        let rate = AAC_SAMPLE_RATES
            .iter()
            .position(|entry| entry.is_some_and(|(_, frequency)| frequency == sampling_frequency))
            .ok_or(DrmError::ReservedSamplingRate(7))? as u8;
        let mode = match audio_mode {
            DrmAudioMode::Mono => 0,
            DrmAudioMode::ParametricStereo if sbr => 1,
            DrmAudioMode::Stereo => 2,
            DrmAudioMode::ParametricStereo => return Err(DrmError::ParametricStereoRequiresSbr),
        };
        let mut writer = BitWriter::new();
        writer.write(0, 2);
        writer.write_bool(sbr);
        writer.write(mode, 2);
        writer.write(rate as u32, 3);
        writer.write_bool(false);
        writer.write_bool(false);
        writer.write(0, 5);
        writer.write_bool(false);
        Self::parse(&writer.finish())
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, DrmError> {
        if self.audio_coding != DrmAudioCoding::Aac {
            return Err(DrmError::StaticConfigRequiresDrmAac);
        }
        let rate = AAC_SAMPLE_RATES
            .iter()
            .position(|entry| {
                entry.is_some_and(|(_, frequency)| frequency == self.sampling_frequency)
            })
            .ok_or(DrmError::ReservedSamplingRate(7))?;
        let mode = match self.audio_mode {
            DrmAudioMode::Mono => 0,
            DrmAudioMode::ParametricStereo if self.sbr => 1,
            DrmAudioMode::Stereo => 2,
            DrmAudioMode::ParametricStereo => return Err(DrmError::ParametricStereoRequiresSbr),
        };
        let mut writer = BitWriter::new();
        writer.write(0, 2);
        writer.write_bool(self.sbr);
        writer.write(mode, 2);
        writer.write(rate as u32, 3);
        writer.write_bool(self.text_flag);
        writer.write_bool(self.enhancement_flag);
        writer.write(self.coder_field as u32, 5);
        writer.write_bool(self.reserved);
        Ok(writer.finish())
    }

    /// Parse the 16-bit DRM SDC type-9 audio configuration following its
    /// Short Id and Stream Id fields.
    pub fn parse(input: &[u8]) -> Result<Self, DrmError> {
        let mut reader = BitReader::new(input);
        Self::parse_from_reader(&mut reader)
    }

    fn parse_from_reader(reader: &mut BitReader<'_>) -> Result<Self, DrmError> {
        let audio_coding = [
            DrmAudioCoding::Aac,
            DrmAudioCoding::Celp,
            DrmAudioCoding::Hvxc,
            DrmAudioCoding::XheAac,
        ][reader.read_u8(2)? as usize];
        let mut sbr = reader.read_bool()?;
        let audio_mode_bits = reader.read_u8(2)?;
        let sampling_rate_bits = reader.read_u8(3)?;
        let text_flag = reader.read_bool()?;
        let enhancement_flag = reader.read_bool()?;
        let coder_field = reader.read_u8(5)?;
        let reserved = reader.read_bool()?;

        let (sampling_frequency_index, sampling_frequency) = match audio_coding {
            DrmAudioCoding::XheAac => {
                sbr = false;
                XHE_SAMPLE_RATES[sampling_rate_bits as usize]
            }
            _ => AAC_SAMPLE_RATES[sampling_rate_bits as usize]
                .ok_or(DrmError::ReservedSamplingRate(sampling_rate_bits))?,
        };

        let (audio_mode, channel_configuration, parametric_stereo) = match audio_coding {
            DrmAudioCoding::Aac => match audio_mode_bits {
                0 => (DrmAudioMode::Mono, 1, false),
                1 => (DrmAudioMode::ParametricStereo, 1, true),
                2 => (DrmAudioMode::Stereo, 2, false),
                value => return Err(DrmError::InvalidAudioMode(value)),
            },
            DrmAudioCoding::XheAac => match audio_mode_bits {
                0 => (DrmAudioMode::Mono, 1, false),
                2 => (DrmAudioMode::Stereo, 2, false),
                value => return Err(DrmError::InvalidAudioMode(value)),
            },
            DrmAudioCoding::Celp | DrmAudioCoding::Hvxc => (DrmAudioMode::Mono, 1, false),
        };
        if parametric_stereo && !sbr {
            return Err(DrmError::ParametricStereoRequiresSbr);
        }
        let surround_mode =
            if audio_coding == DrmAudioCoding::Aac || audio_coding == DrmAudioCoding::XheAac {
                match coder_field >> 2 {
                    0 => None,
                    2 => Some(DrmSurroundMode::FiveOne),
                    3 => Some(DrmSurroundMode::SevenOne),
                    7 => Some(DrmSurroundMode::StreamDefined),
                    value => return Err(DrmError::ReservedMpegSurroundMode(value)),
                }
            } else {
                None
            };
        let drm_surround =
            audio_coding == DrmAudioCoding::Aac && surround_mode.is_some() && audio_mode_bits != 1;

        Ok(Self {
            audio_coding,
            sbr,
            audio_mode,
            sampling_frequency_index,
            sampling_frequency,
            text_flag,
            enhancement_flag,
            coder_field,
            reserved,
            channel_configuration,
            parametric_stereo,
            drm_surround,
            surround_mode,
            samples_per_frame: (audio_coding == DrmAudioCoding::Aac).then_some(960),
            error_protection_config: (audio_coding == DrmAudioCoding::Aac).then_some(1),
            bits_read: reader.bits_read(),
        })
    }

    /// Parse a type-9 xHE-AAC SDC entity followed by its DRM-specific static
    /// decoder configuration and map it to the common USAC configuration.
    pub fn parse_xhe_with_static_config(
        input: &[u8],
    ) -> Result<(Self, AudioSpecificConfig), DrmError> {
        let mut reader = BitReader::new(input);
        let base = Self::parse_from_reader(&mut reader)?;
        if base.audio_coding != DrmAudioCoding::XheAac {
            return Err(DrmError::StaticConfigRequiresXheAac);
        }
        let core_sbr_frame_length_index = reader.read_u8(2)? + 1;
        const FRAME_LENGTHS: [u16; 5] = [768, 1024, 2048, 2048, 4096];
        const SBR_RATIOS: [u8; 5] = [0, 0, 2, 3, 1];
        let index = core_sbr_frame_length_index as usize;
        let output_frame_length = FRAME_LENGTHS[index];
        let sbr_ratio_index = SBR_RATIOS[index];
        let core_frame_length = match sbr_ratio_index {
            1 => output_frame_length / 4,
            2 => output_frame_length * 3 / 8,
            3 => output_frame_length / 2,
            _ => output_frame_length,
        };
        let noise_filling = reader.read_bool()?;
        let sbr = (sbr_ratio_index != 0)
            .then(|| UsacSbrConfig::parse(&mut reader))
            .transpose()?;
        let element = if base.audio_mode == DrmAudioMode::Mono {
            UsacElementConfig::SingleChannel { noise_filling, sbr }
        } else {
            let stereo_config_index = if sbr.is_some() { reader.read_u8(2)? } else { 0 };
            let mps212 = (stereo_config_index != 0)
                .then(|| Mps212Config::parse_drm(&mut reader, stereo_config_index))
                .transpose()?;
            UsacElementConfig::ChannelPair {
                noise_filling,
                sbr,
                stereo_config_index,
                mps212,
            }
        };
        let usac = UsacConfig {
            sampling_frequency_index: base.sampling_frequency_index,
            sampling_frequency: base.sampling_frequency,
            core_sbr_frame_length_index,
            core_frame_length,
            output_frame_length,
            sbr_ratio_index,
            channel_configuration_index: base.channel_configuration,
            elements: vec![element],
            extensions: Vec::new(),
        };
        let asc = AudioSpecificConfig {
            audio_object_type: 42,
            sampling_frequency_index: base.sampling_frequency_index,
            sampling_frequency: base.sampling_frequency,
            channel_configuration: base.channel_configuration,
            extension: None,
            ga_specific: None,
            eld_specific: None,
            usac_config: Some(usac),
            error_protection_config: None,
            program_config: None,
            bits_read: reader.bits_read(),
        };
        Ok((base, asc))
    }

    /// Construct the implicit standard 1-to-2 MPEG Surround configuration
    /// used by FDK for legacy DRM Surround signalling.
    pub fn surround_specific_config(&self) -> Result<Option<SpatialSpecificConfig>, DrmError> {
        if !self.drm_surround {
            return Ok(None);
        }
        let sampling_frequency = if self.sbr {
            self.sampling_frequency
                .checked_mul(2)
                .ok_or(DrmError::SamplingFrequencyOverflow)?
        } else {
            self.sampling_frequency
        };
        Ok(Some(SpatialSpecificConfig::default_212(
            sampling_frequency,
            30,
        )?))
    }
}

/// Stateful decoder for MPEG-conformant USAC access units carried by DRM
/// xHE-AAC. The 16-bit SDC and following static configuration are consumed at
/// construction; audio access units are passed directly to the USAC core.
#[derive(Debug, Clone)]
pub struct DrmXheDecoder {
    pub drm_config: DrmAudioConfig,
    pub(crate) decoder: AacLcDecoder,
}

impl DrmXheDecoder {
    pub fn from_static_config(input: &[u8]) -> Result<Self, DrmXheDecodeError> {
        let (drm_config, asc) = DrmAudioConfig::parse_xhe_with_static_config(input)?;
        let decoder = AacLcDecoder::from_audio_specific_config(&asc)?;
        Ok(Self {
            drm_config,
            decoder,
        })
    }

    pub fn decode_access_unit_f32(
        &mut self,
        payload: &[u8],
    ) -> Result<Vec<Vec<f32>>, DrmXheDecodeError> {
        Ok(self
            .decoder
            .decode_usac_access_unit_multichannel_f32(payload)?)
    }

    pub fn decode_interleaved_f32(
        &mut self,
        payload: &[u8],
    ) -> Result<Vec<f32>, DrmXheDecodeError> {
        Ok(interleave_multichannel_f32(
            &self.decode_access_unit_f32(payload)?,
        ))
    }

    pub fn decode_interleaved_i16(
        &mut self,
        payload: &[u8],
    ) -> Result<Vec<i16>, DrmXheDecodeError> {
        Ok(interleave_multichannel_i16(
            &self.decode_access_unit_f32(payload)?,
        ))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum DrmXheDecodeError {
    Config(DrmError),
    Decode(DecodeError),
    Usac(UsacDecodeError),
}

impl From<DrmError> for DrmXheDecodeError {
    fn from(value: DrmError) -> Self {
        Self::Config(value)
    }
}

impl From<DecodeError> for DrmXheDecodeError {
    fn from(value: DecodeError) -> Self {
        Self::Decode(value)
    }
}

impl From<UsacDecodeError> for DrmXheDecodeError {
    fn from(value: UsacDecodeError) -> Self {
        Self::Usac(value)
    }
}

impl fmt::Display for DrmXheDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(error) => error.fmt(f),
            Self::Decode(error) => error.fmt(f),
            Self::Usac(error) => write!(f, "DRM xHE-AAC decode error: {error:?}"),
        }
    }
}

impl std::error::Error for DrmXheDecodeError {}

#[derive(Debug, Clone)]
pub struct DrmAacDecoder {
    pub drm_config: DrmAudioConfig,
    pub(crate) decoder: AacLcDecoder,
    sbr_mono_parser: Option<DrmSbrMonoParser>,
    sbr_mono_processor: Option<LdSbrChannelProcessor>,
    sbr_stereo_parser: Option<DrmSbrStereoParser>,
    sbr_stereo_processors: Option<[LdSbrChannelProcessor; 2]>,
    ps_parser: Option<PsParser>,
    ps_processor: Option<PsQmfProcessor>,
    last_ps_frame: Option<PsFrame>,
    surround_decoder: Option<Sac212Decoder>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DrmAacDecodedFrame {
    pub samples: Vec<f32>,
    pub reversed_sbr_payload: Vec<u8>,
    pub sbr_payload_bits: usize,
}

impl DrmAacDecoder {
    pub fn from_sdc_config(input: &[u8]) -> Result<Self, DrmAacDecodeError> {
        let drm_config = DrmAudioConfig::parse(input)?;
        if drm_config.audio_coding != DrmAudioCoding::Aac {
            return Err(DrmAacDecodeError::Config(
                DrmError::StaticConfigRequiresDrmAac,
            ));
        }
        let decoder = AacLcDecoder::new_drm_aac(
            drm_config.sampling_frequency_index,
            drm_config.channel_configuration,
        )?;
        let (sbr_mono_parser, sbr_mono_processor) = if drm_config.sbr
            && (drm_config.audio_mode == DrmAudioMode::Mono
                || drm_config.audio_mode == DrmAudioMode::ParametricStereo)
        {
            let output_sampling_frequency = drm_config
                .sampling_frequency
                .checked_mul(2)
                .ok_or(DrmSbrError::SamplingFrequencyOverflow)?;
            (
                Some(DrmSbrMonoParser::new(drm_config.sampling_frequency)?),
                Some(LdSbrChannelProcessor::new(
                    output_sampling_frequency,
                    true,
                    0x2468_ace0,
                )),
            )
        } else {
            (None, None)
        };
        let (sbr_stereo_parser, sbr_stereo_processors) =
            if drm_config.sbr && drm_config.audio_mode == DrmAudioMode::Stereo {
                let output_sampling_frequency = drm_config
                    .sampling_frequency
                    .checked_mul(2)
                    .ok_or(DrmSbrError::SamplingFrequencyOverflow)?;
                (
                    Some(DrmSbrStereoParser::new(drm_config.sampling_frequency)?),
                    Some([
                        LdSbrChannelProcessor::new(output_sampling_frequency, true, 0x2468_ace0),
                        LdSbrChannelProcessor::new(output_sampling_frequency, true, 0x1357_9bdf),
                    ]),
                )
            } else {
                (None, None)
            };
        let parametric_stereo = drm_config.parametric_stereo;
        Ok(Self {
            drm_config,
            decoder,
            sbr_mono_parser,
            sbr_mono_processor,
            sbr_stereo_parser,
            sbr_stereo_processors,
            ps_parser: parametric_stereo.then(PsParser::new),
            ps_processor: parametric_stereo.then(PsQmfProcessor::new),
            last_ps_frame: None,
            surround_decoder: None,
        })
    }

    pub fn configure_surround(
        &mut self,
        config: SpatialSpecificConfig,
    ) -> Result<(), DrmAacDecodeError> {
        if !self.drm_config.drm_surround {
            return Err(DrmError::SurroundNotSignaled.into());
        }
        self.surround_decoder = Some(Sac212Decoder::new(config)?);
        Ok(())
    }

    pub fn decode_crc_protected_mono_f32(
        &mut self,
        packet: &[u8],
    ) -> Result<Vec<f32>, DrmAacDecodeError> {
        let (&read_crc, payload) =
            packet
                .split_first()
                .ok_or(DrmAacDecodeError::Config(DrmError::Bit(
                    BitError::UnexpectedEof {
                        needed_bits: 8,
                        remaining_bits: 0,
                    },
                )))?;
        let mut trial = self.decoder.clone();
        let (samples, protected_bits, _) = trial.decode_drm_aac_mono_f32(payload)?;
        check_drm_crc_bits(read_crc, payload, 0, protected_bits)?;
        self.decoder = trial;
        Ok(samples)
    }

    pub fn decode_crc_protected_stereo_f32(
        &mut self,
        packet: &[u8],
    ) -> Result<Vec<f32>, DrmAacDecodeError> {
        let (&read_crc, payload) =
            packet
                .split_first()
                .ok_or(DrmAacDecodeError::Config(DrmError::Bit(
                    BitError::UnexpectedEof {
                        needed_bits: 8,
                        remaining_bits: 0,
                    },
                )))?;
        let mut trial = self.decoder.clone();
        let ([left, right], protected_bits, _) = trial.decode_drm_aac_stereo_f32(payload)?;
        check_drm_crc_bits(read_crc, payload, 0, protected_bits)?;
        self.decoder = trial;
        Ok(interleave_stereo_f32(&left, &right))
    }

    pub fn decode_crc_protected_interleaved_f32(
        &mut self,
        packet: &[u8],
    ) -> Result<Vec<f32>, DrmAacDecodeError> {
        match self.drm_config.audio_mode {
            DrmAudioMode::Mono => self.decode_crc_protected_mono_f32(packet),
            DrmAudioMode::Stereo => self.decode_crc_protected_stereo_f32(packet),
            DrmAudioMode::ParametricStereo => Err(DrmAacDecodeError::Config(
                DrmError::ParametricStereoRequiresSbr,
            )),
        }
    }

    pub fn decode_crc_protected_interleaved_f32_with_sbr(
        &mut self,
        packet: &[u8],
        packet_bits: usize,
    ) -> Result<DrmAacDecodedFrame, DrmAacDecodeError> {
        self.decode_crc_protected_interleaved_f32_with_sbr_impl(packet, packet_bits, false, None)
    }

    /// Decode the DRM AAC core and render a trailing mono scalable-SBR frame.
    /// Unlike the extraction-only `with_sbr` entry point, this validates the
    /// inner SBR CRC and advances the SBR parser/QMF state transactionally.
    pub fn decode_crc_protected_interleaved_f32_rendering_sbr(
        &mut self,
        packet: &[u8],
        packet_bits: usize,
    ) -> Result<DrmAacDecodedFrame, DrmAacDecodeError> {
        self.decode_crc_protected_interleaved_f32_with_sbr_impl(packet, packet_bits, true, None)
    }

    pub fn decode_crc_protected_interleaved_f32_rendering_sbr_and_surround(
        &mut self,
        packet: &[u8],
        packet_bits: usize,
        surround_payload: &[u8],
        surround_payload_bits: usize,
    ) -> Result<DrmAacDecodedFrame, DrmAacDecodeError> {
        self.decode_crc_protected_interleaved_f32_with_sbr_impl(
            packet,
            packet_bits,
            true,
            Some((surround_payload, surround_payload_bits)),
        )
    }

    fn decode_crc_protected_interleaved_f32_with_sbr_impl(
        &mut self,
        packet: &[u8],
        packet_bits: usize,
        render_sbr: bool,
        surround_payload: Option<(&[u8], usize)>,
    ) -> Result<DrmAacDecodedFrame, DrmAacDecodeError> {
        if packet_bits < 8 || packet_bits > packet.len() * 8 {
            return Err(DrmAacDecodeError::Config(
                DrmError::InvalidAudioPacketBits {
                    declared_bits: packet_bits,
                    available_bits: packet.len() * 8,
                },
            ));
        }
        let (&read_crc, payload) = packet
            .split_first()
            .expect("packet_bits guarantees CRC byte");
        let payload_bits = packet_bits - 8;
        let mut trial = self.decoder.clone();
        let (mut samples, protected_bits, core_bits) = match self.drm_config.audio_mode {
            DrmAudioMode::Mono => trial.decode_drm_aac_mono_f32(payload)?,
            DrmAudioMode::Stereo => {
                let ([left, right], protected, core) = trial.decode_drm_aac_stereo_f32(payload)?;
                (interleave_stereo_f32(&left, &right), protected, core)
            }
            DrmAudioMode::ParametricStereo => trial.decode_drm_aac_mono_f32(payload)?,
        };
        if core_bits > payload_bits {
            return Err(DrmAacDecodeError::Config(
                DrmError::InvalidAudioPacketBits {
                    declared_bits: packet_bits,
                    available_bits: 8 + core_bits,
                },
            ));
        }
        check_drm_crc_bits(read_crc, payload, 0, protected_bits)?;
        let sbr_payload_bits = if self.drm_config.sbr {
            payload_bits - core_bits
        } else {
            0
        };
        let reversed_sbr_payload = if sbr_payload_bits == 0 {
            Vec::new()
        } else {
            reverse_drm_sbr_payload_bits(payload, core_bits, sbr_payload_bits)?
        };
        let mut trial_sbr_parser = self.sbr_mono_parser.clone();
        let mut trial_sbr_processor = self.sbr_mono_processor.clone();
        let mut trial_sbr_stereo_parser = self.sbr_stereo_parser.clone();
        let mut trial_sbr_stereo_processors = self.sbr_stereo_processors.clone();
        let mut trial_ps_parser = self.ps_parser.clone();
        let mut trial_ps_processor = self.ps_processor.clone();
        let mut trial_last_ps_frame = self.last_ps_frame.clone();
        let mut trial_surround_decoder = self.surround_decoder.clone();
        if render_sbr && sbr_payload_bits != 0 {
            if let (Some(parser), Some(processor)) =
                (trial_sbr_parser.as_mut(), trial_sbr_processor.as_mut())
            {
                let frame = parser.parse(&reversed_sbr_payload, sbr_payload_bits)?;
                let core = samples
                    .iter()
                    .map(|&sample| f64::from(sample))
                    .collect::<Vec<_>>();
                if self.drm_config.audio_mode == DrmAudioMode::ParametricStereo {
                    let ps_parser = trial_ps_parser
                        .as_mut()
                        .ok_or(DrmError::ParametricStereoRequiresSbr)?;
                    let ps_frame = ps_parser
                        .parse_sbr_extension(&frame.extended_data, 30)?
                        .or_else(|| trial_last_ps_frame.clone())
                        .ok_or(DrmError::MissingParametricStereoPayload)?;
                    let slots = processor.process_channel_to_qmf(
                        &core,
                        &frame.active_header,
                        &frame.frequency_tables,
                        &frame.control,
                        &frame.values,
                        &frame.dequantized,
                        &frame.harmonics,
                        2,
                    )?;
                    let (left, right) = trial_ps_processor
                        .as_mut()
                        .ok_or(DrmError::ParametricStereoRequiresSbr)?
                        .process_qmf(&slots, &ps_frame)?;
                    samples = interleave_stereo_f32(
                        &left
                            .into_iter()
                            .map(|sample| sample as f32)
                            .collect::<Vec<_>>(),
                        &right
                            .into_iter()
                            .map(|sample| sample as f32)
                            .collect::<Vec<_>>(),
                    );
                    trial_last_ps_frame = Some(ps_frame);
                } else {
                    if let Some((payload, payload_bits)) = surround_payload {
                        let slots = processor.process_channel_to_qmf(
                            &core,
                            &frame.active_header,
                            &frame.frequency_tables,
                            &frame.control,
                            &frame.values,
                            &frame.dequantized,
                            &frame.harmonics,
                            2,
                        )?;
                        let surround = trial_surround_decoder
                            .as_mut()
                            .ok_or(DrmError::MissingSurroundConfiguration)?;
                        let (left, right) = surround.decode_qmf(&slots, payload, payload_bits)?;
                        samples = interleave_stereo_f32(
                            &left
                                .into_iter()
                                .map(|sample| sample as f32)
                                .collect::<Vec<_>>(),
                            &right
                                .into_iter()
                                .map(|sample| sample as f32)
                                .collect::<Vec<_>>(),
                        );
                    } else {
                        samples = processor
                            .process_channel(
                                &core,
                                &frame.active_header,
                                &frame.frequency_tables,
                                &frame.control,
                                &frame.values,
                                &frame.dequantized,
                                &frame.harmonics,
                                2,
                            )?
                            .into_iter()
                            .map(|sample| sample as f32)
                            .collect();
                    }
                }
            } else if let (Some(parser), Some(processors)) = (
                trial_sbr_stereo_parser.as_mut(),
                trial_sbr_stereo_processors.as_mut(),
            ) {
                let frame = parser.parse(&reversed_sbr_payload, sbr_payload_bits)?;
                let mut left_core = Vec::with_capacity(samples.len() / 2);
                let mut right_core = Vec::with_capacity(samples.len() / 2);
                for stereo in samples.chunks_exact(2) {
                    left_core.push(f64::from(stereo[0]));
                    right_core.push(f64::from(stereo[1]));
                }
                let left = processors[0].process_channel(
                    &left_core,
                    &frame.active_header,
                    &frame.frequency_tables,
                    &frame.left_control,
                    &frame.left,
                    &frame.left_dequantized,
                    &frame.left_harmonics,
                    2,
                )?;
                let right = processors[1].process_channel(
                    &right_core,
                    &frame.active_header,
                    &frame.frequency_tables,
                    &frame.right_control,
                    &frame.right,
                    &frame.right_dequantized,
                    &frame.right_harmonics,
                    2,
                )?;
                samples = interleave_stereo_f32(
                    &left
                        .into_iter()
                        .map(|sample| sample as f32)
                        .collect::<Vec<_>>(),
                    &right
                        .into_iter()
                        .map(|sample| sample as f32)
                        .collect::<Vec<_>>(),
                );
            }
        }
        self.decoder = trial;
        self.sbr_mono_parser = trial_sbr_parser;
        self.sbr_mono_processor = trial_sbr_processor;
        self.sbr_stereo_parser = trial_sbr_stereo_parser;
        self.sbr_stereo_processors = trial_sbr_stereo_processors;
        self.ps_parser = trial_ps_parser;
        self.ps_processor = trial_ps_processor;
        self.last_ps_frame = trial_last_ps_frame;
        self.surround_decoder = trial_surround_decoder;
        Ok(DrmAacDecodedFrame {
            samples,
            reversed_sbr_payload,
            sbr_payload_bits,
        })
    }

    pub fn decode_crc_protected_interleaved_i16(
        &mut self,
        packet: &[u8],
    ) -> Result<Vec<i16>, DrmAacDecodeError> {
        let (&read_crc, payload) =
            packet
                .split_first()
                .ok_or(DrmAacDecodeError::Config(DrmError::Bit(
                    BitError::UnexpectedEof {
                        needed_bits: 8,
                        remaining_bits: 0,
                    },
                )))?;
        let mut trial = self.decoder.clone();
        let (samples, protected_bits) = match self.drm_config.audio_mode {
            DrmAudioMode::Mono => trial
                .decode_drm_aac_mono_i16(payload)
                .map(|(samples, bits, _)| (samples, bits))?,
            DrmAudioMode::Stereo => {
                trial
                    .decode_drm_aac_stereo_i16(payload)
                    .map(|([left, right], bits, _)| {
                        (interleave_stereo_i16_samples(&left, &right), bits)
                    })?
            }
            DrmAudioMode::ParametricStereo => {
                return Err(DrmAacDecodeError::Config(
                    DrmError::ParametricStereoRequiresSbr,
                ))
            }
        };
        check_drm_crc_bits(read_crc, payload, 0, protected_bits)?;
        self.decoder = trial;
        Ok(samples)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum DrmAacDecodeError {
    Config(DrmError),
    Decode(DecodeError),
    Sbr(DrmSbrError),
    SbrProcessing(LdSbrProcessingError),
    Ps(PsError),
    Sac(SacDecodeError),
}

impl From<DrmError> for DrmAacDecodeError {
    fn from(value: DrmError) -> Self {
        Self::Config(value)
    }
}

impl From<DecodeError> for DrmAacDecodeError {
    fn from(value: DecodeError) -> Self {
        Self::Decode(value)
    }
}

impl From<DrmSbrError> for DrmAacDecodeError {
    fn from(value: DrmSbrError) -> Self {
        Self::Sbr(value)
    }
}

impl From<LdSbrProcessingError> for DrmAacDecodeError {
    fn from(value: LdSbrProcessingError) -> Self {
        Self::SbrProcessing(value)
    }
}

impl From<PsError> for DrmAacDecodeError {
    fn from(value: PsError) -> Self {
        Self::Ps(value)
    }
}

impl From<SacDecodeError> for DrmAacDecodeError {
    fn from(value: SacDecodeError) -> Self {
        Self::Sac(value)
    }
}

impl fmt::Display for DrmAacDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(error) => error.fmt(f),
            Self::Decode(error) => error.fmt(f),
            Self::Sbr(error) => error.fmt(f),
            Self::SbrProcessing(error) => write!(f, "DRM SBR processing error: {error:?}"),
            Self::Ps(error) => error.fmt(f),
            Self::Sac(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for DrmAacDecodeError {}

/// FDK DRM CRC-8: MSB-first polynomial 0x1d, initial 0xff and final xor 0xff.
pub fn drm_crc8(bytes: &[u8]) -> u8 {
    drm_crc8_bits(bytes, 0, bytes.len() * 8).expect("full byte slice is in range")
}

pub fn drm_crc8_bits(bytes: &[u8], bit_offset: usize, bit_len: usize) -> Result<u8, DrmError> {
    if bit_offset
        .checked_add(bit_len)
        .is_none_or(|end| end > bytes.len() * 8)
    {
        return Err(DrmError::InvalidCrcRegion {
            bit_offset,
            bit_len,
            available_bits: bytes.len() * 8,
        });
    }
    let mut crc = 0xffu8;
    for position in bit_offset..bit_offset + bit_len {
        let input_bit = (bytes[position / 8] >> (7 - position % 8)) & 1;
        let feedback = input_bit ^ u8::from((crc & 0x80) != 0);
        crc <<= 1;
        if feedback != 0 {
            crc ^= 0x1d;
        }
    }
    Ok(crc ^ 0xff)
}

pub fn check_drm_crc(read_value: u8, protected_bytes: &[u8]) -> Result<(), DrmError> {
    let calculated = drm_crc8(protected_bytes);
    if calculated == read_value {
        Ok(())
    } else {
        Err(DrmError::CrcMismatch {
            expected: read_value,
            calculated,
        })
    }
}

pub fn check_drm_crc_bits(
    read_value: u8,
    bytes: &[u8],
    bit_offset: usize,
    bit_len: usize,
) -> Result<(), DrmError> {
    let calculated = drm_crc8_bits(bytes, bit_offset, bit_len)?;
    if calculated == read_value {
        Ok(())
    } else {
        Err(DrmError::CrcMismatch {
            expected: read_value,
            calculated,
        })
    }
}

/// Reverse the complete DRM SBR payload bit sequence. The DRM syntax stores
/// SBR back-to-front; FDK reads bytes from the end and reverses each byte,
/// which is equivalent to reversing all significant bits here.
pub fn reverse_drm_sbr_payload_bits(
    bytes: &[u8],
    bit_offset: usize,
    bit_len: usize,
) -> Result<Vec<u8>, DrmError> {
    if bit_offset
        .checked_add(bit_len)
        .is_none_or(|end| end > bytes.len() * 8)
    {
        return Err(DrmError::InvalidSbrRegion {
            bit_offset,
            bit_len,
            available_bits: bytes.len() * 8,
        });
    }
    let mut output = vec![0u8; bit_len.div_ceil(8)];
    for output_bit in 0..bit_len {
        let source = bit_offset + bit_len - 1 - output_bit;
        if (bytes[source / 8] >> (7 - source % 8)) & 1 != 0 {
            output[output_bit / 8] |= 1 << (7 - output_bit % 8);
        }
    }
    Ok(output)
}

pub fn parse_reversed_drm_sbr_payload(
    bytes: &[u8],
    bit_len: usize,
) -> Result<SbrFillPayload, DrmError> {
    if bit_len < 9 || bit_len > bytes.len() * 8 {
        return Err(DrmError::InvalidSbrRegion {
            bit_offset: 0,
            bit_len,
            available_bits: bytes.len() * 8,
        });
    }
    let mut reader = BitReader::new(bytes);
    let transmitted_crc = reader.read_u8(8)?;
    check_drm_crc_bits(transmitted_crc, bytes, 8, bit_len - 8)?;
    let header_present = reader.read_bool()?;
    let header = header_present
        .then(|| crate::asc::LdSbrHeader::parse(&mut reader))
        .transpose()?;
    if reader.bits_read() > bit_len {
        return Err(DrmError::InvalidSbrRegion {
            bit_offset: reader.bits_read(),
            bit_len: 0,
            available_bits: bit_len,
        });
    }
    let frame_data_bits = bit_len - reader.bits_read();
    let mut frame_data = vec![0u8; frame_data_bits.div_ceil(8)];
    for bit in 0..frame_data_bits {
        if reader.read_bool()? {
            frame_data[bit / 8] |= 1 << (7 - bit % 8);
        }
    }
    Ok(SbrFillPayload {
        extension_type: EXT_SBR_DATA_CRC,
        transmitted_crc: Some(u16::from(transmitted_crc)),
        header_present,
        header,
        frame_data,
        frame_data_bits,
    })
}

pub fn drm_default_sbr_header(output_sampling_frequency: u32) -> crate::asc::LdSbrHeader {
    let (start_frequency, stop_frequency) = if output_sampling_frequency >= 96_000 {
        (4, 3)
    } else if output_sampling_frequency > 24_000 {
        (7, 3)
    } else {
        (5, 0)
    };
    crate::asc::LdSbrHeader {
        amp_resolution: true,
        crossover_band: 0,
        reserved: 0,
        start_frequency,
        stop_frequency,
        frequency_scale: Some(0),
        alter_scale: Some(true),
        noise_bands: Some(2),
        limiter_bands: Some(2),
        limiter_gains: Some(2),
        interpol_frequency: Some(true),
        smoothing_mode: Some(true),
    }
}

#[derive(Debug, Clone)]
pub struct DrmSbrMonoParser {
    parser: SbrMonoFrameParser,
}

impl DrmSbrMonoParser {
    pub fn new(core_sampling_frequency: u32) -> Result<Self, DrmSbrError> {
        let output_sampling_frequency = core_sampling_frequency
            .checked_mul(2)
            .ok_or(DrmSbrError::SamplingFrequencyOverflow)?;
        Ok(Self {
            parser: SbrMonoFrameParser::new(
                drm_default_sbr_header(output_sampling_frequency),
                output_sampling_frequency,
                960,
            )?,
        })
    }

    pub fn parse(
        &mut self,
        reversed_payload: &[u8],
        payload_bits: usize,
    ) -> Result<SbrMonoFrame, DrmSbrError> {
        let payload = parse_reversed_drm_sbr_payload(reversed_payload, payload_bits)?;
        Ok(self.parser.parse(&payload)?)
    }
}

#[derive(Debug, Clone)]
pub struct DrmSbrStereoParser {
    parser: SbrStereoFrameParser,
}

impl DrmSbrStereoParser {
    pub fn new(core_sampling_frequency: u32) -> Result<Self, DrmSbrError> {
        let output_sampling_frequency = core_sampling_frequency
            .checked_mul(2)
            .ok_or(DrmSbrError::SamplingFrequencyOverflow)?;
        Ok(Self {
            parser: SbrStereoFrameParser::new(
                drm_default_sbr_header(output_sampling_frequency),
                output_sampling_frequency,
                960,
            )?,
        })
    }

    pub fn parse(
        &mut self,
        reversed_payload: &[u8],
        payload_bits: usize,
    ) -> Result<SbrStereoFrame, DrmSbrError> {
        let payload = parse_reversed_drm_sbr_payload(reversed_payload, payload_bits)?;
        Ok(self.parser.parse(&payload)?)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum DrmSbrError {
    Drm(DrmError),
    Sbr(SbrError),
    SamplingFrequencyOverflow,
}

impl From<DrmError> for DrmSbrError {
    fn from(value: DrmError) -> Self {
        Self::Drm(value)
    }
}

impl From<SbrError> for DrmSbrError {
    fn from(value: SbrError) -> Self {
        Self::Sbr(value)
    }
}

impl fmt::Display for DrmSbrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Drm(error) => error.fmt(f),
            Self::Sbr(error) => error.fmt(f),
            Self::SamplingFrequencyOverflow => write!(f, "DRM SBR output frequency overflow"),
        }
    }
}

impl std::error::Error for DrmSbrError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DrmError {
    Asc(AscError),
    Bit(BitError),
    Sac(SacError),
    CrcMismatch {
        expected: u8,
        calculated: u8,
    },
    InvalidAudioMode(u8),
    InvalidAudioPacketBits {
        declared_bits: usize,
        available_bits: usize,
    },
    InvalidCrcRegion {
        bit_offset: usize,
        bit_len: usize,
        available_bits: usize,
    },
    InvalidSbrRegion {
        bit_offset: usize,
        bit_len: usize,
        available_bits: usize,
    },
    MissingParametricStereoPayload,
    MissingSurroundConfiguration,
    ParametricStereoRequiresSbr,
    ReservedSamplingRate(u8),
    ReservedMpegSurroundMode(u8),
    SamplingFrequencyOverflow,
    SurroundNotSignaled,
    StaticConfigRequiresXheAac,
    StaticConfigRequiresDrmAac,
}

impl From<BitError> for DrmError {
    fn from(value: BitError) -> Self {
        Self::Bit(value)
    }
}

impl From<AscError> for DrmError {
    fn from(value: AscError) -> Self {
        Self::Asc(value)
    }
}

impl From<SacError> for DrmError {
    fn from(value: SacError) -> Self {
        Self::Sac(value)
    }
}

impl fmt::Display for DrmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Asc(error) => error.fmt(f),
            Self::Bit(error) => error.fmt(f),
            Self::Sac(error) => error.fmt(f),
            Self::CrcMismatch {
                expected,
                calculated,
            } => write!(
                f,
                "DRM CRC mismatch: read 0x{expected:02x}, calculated 0x{calculated:02x}"
            ),
            Self::InvalidAudioMode(value) => write!(f, "invalid DRM audio mode {value}"),
            Self::InvalidAudioPacketBits {
                declared_bits,
                available_bits,
            } => write!(
                f,
                "invalid DRM audio packet length {declared_bits} bits for {available_bits} available bits"
            ),
            Self::InvalidCrcRegion {
                bit_offset,
                bit_len,
                available_bits,
            } => write!(
                f,
                "invalid DRM CRC bit region {bit_offset}+{bit_len}, only {available_bits} bits available"
            ),
            Self::InvalidSbrRegion {
                bit_offset,
                bit_len,
                available_bits,
            } => write!(
                f,
                "invalid DRM SBR bit region {bit_offset}+{bit_len}, only {available_bits} bits available"
            ),
            Self::MissingParametricStereoPayload => {
                write!(f, "initial DRM Parametric Stereo payload is missing")
            }
            Self::MissingSurroundConfiguration => {
                write!(f, "DRM MPEG Surround configuration is missing")
            }
            Self::ParametricStereoRequiresSbr => {
                write!(f, "DRM parametric stereo requires SBR")
            }
            Self::ReservedSamplingRate(value) => {
                write!(f, "reserved DRM sampling-rate index {value}")
            }
            Self::ReservedMpegSurroundMode(value) => {
                write!(f, "reserved DRM MPEG Surround mode {value:03b}")
            }
            Self::SamplingFrequencyOverflow => write!(f, "DRM sampling frequency overflow"),
            Self::SurroundNotSignaled => write!(f, "DRM MPEG Surround is not signaled"),
            Self::StaticConfigRequiresXheAac => {
                write!(f, "DRM static USAC configuration requires xHE-AAC coding")
            }
            Self::StaticConfigRequiresDrmAac => {
                write!(f, "DRM AAC decoder configuration requires AAC coding")
            }
        }
    }
}

impl std::error::Error for DrmError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitWriter;
    use crate::ld_sbr::{decode_sbr_huffman, LdSbrFrequencyTables, SbrHuffmanBook};

    #[test]
    fn writes_drm_aac_sdc_configuration() {
        for (mode, sbr) in [
            (DrmAudioMode::Mono, false),
            (DrmAudioMode::Stereo, false),
            (DrmAudioMode::ParametricStereo, true),
        ] {
            let config = DrmAudioConfig::aac(48_000, mode, sbr).unwrap();
            let parsed = DrmAudioConfig::parse(&config.to_bytes().unwrap()).unwrap();
            assert_eq!(parsed.audio_coding, DrmAudioCoding::Aac);
            assert_eq!(parsed.audio_mode, mode);
            assert_eq!(parsed.sbr, sbr);
            assert_eq!(parsed.sampling_frequency, 48_000);
        }
        assert_eq!(huffman_code(SbrHuffmanBook::EnvelopeLevel15Time, 127), None);
    }

    fn huffman_code(book: SbrHuffmanBook, symbol: i8) -> Option<Vec<bool>> {
        for length in 1..=16 {
            for value in 0..(1u32 << length) {
                let mut writer = BitWriter::new();
                writer.write(value, length);
                let bytes = writer.finish();
                let mut reader = BitReader::new(&bytes);
                if decode_sbr_huffman(&mut reader, book) == Ok(symbol)
                    && reader.bits_read() == length
                {
                    return Some(
                        (0..length)
                            .rev()
                            .map(|bit| value & (1 << bit) != 0)
                            .collect(),
                    );
                }
            }
        }
        None
    }

    fn write_code(writer: &mut BitWriter, code: &[bool]) {
        for &bit in code {
            writer.write_bool(bit);
        }
    }

    fn drm_sbr_packet_with_extension(extension: &[u8]) -> (Vec<u8>, usize, usize) {
        let header = drm_default_sbr_header(96_000);
        let tables = LdSbrFrequencyTables::from_header(&header, 96_000).unwrap();
        let zero = huffman_code(SbrHuffmanBook::EnvelopeLevel30Frequency, 0).unwrap();
        let mut frame_data = BitWriter::new();
        frame_data.write_bool(false); // no data extra
        frame_data.write(0, 2); // FIXFIX
        frame_data.write(0, 2); // one envelope
        frame_data.write_bool(true); // high frequency resolution
        frame_data.write_bool(false); // envelope frequency direction
        frame_data.write_bool(false); // noise frequency direction
        for _ in 0..tables.noise_band_count() {
            frame_data.write(0, 2);
        }
        frame_data.write(0, 6);
        for _ in 1..tables.high_band_count() {
            write_code(&mut frame_data, &zero);
        }
        frame_data.write(0, 5);
        for _ in 1..tables.noise_band_count() {
            write_code(&mut frame_data, &zero);
        }
        frame_data.write_bool(false); // no harmonics
        frame_data.write_bool(!extension.is_empty());
        if !extension.is_empty() {
            frame_data.write(extension.len() as u32, 4);
            for &byte in extension {
                frame_data.write(byte.into(), 8);
            }
        }
        let frame_data_bits = frame_data.bits_written();
        let frame_data = frame_data.finish();

        let mut protected = BitWriter::new();
        protected.write_bool(false); // use DRM default header
        for bit in 0..frame_data_bits {
            protected.write_bool((frame_data[bit / 8] >> (7 - bit % 8)) & 1 != 0);
        }
        let protected_bits = protected.bits_written();
        let protected = protected.finish();
        let inner_crc = drm_crc8_bits(&protected, 0, protected_bits).unwrap();
        let mut normal_sbr = BitWriter::new();
        normal_sbr.write(inner_crc.into(), 8);
        for bit in 0..protected_bits {
            normal_sbr.write_bool((protected[bit / 8] >> (7 - bit % 8)) & 1 != 0);
        }
        let sbr_bits = normal_sbr.bits_written();
        let normal_sbr = normal_sbr.finish();
        if !extension.is_empty() {
            let parsed = DrmSbrMonoParser::new(48_000)
                .unwrap()
                .parse(&normal_sbr, sbr_bits)
                .unwrap();
            assert_eq!(parsed.extended_data, extension);
        }

        let mut payload = BitWriter::new();
        payload.write_bool(false);
        payload.write(0, 2);
        payload.write_bool(false);
        payload.write(0, 6);
        payload.write_bool(false);
        payload.write_bool(false);
        payload.write_bool(false);
        payload.write(0, 8);
        payload.write(0, 14);
        payload.write(0, 6); // 41 core bits
        for bit in (0..sbr_bits).rev() {
            payload.write_bool((normal_sbr[bit / 8] >> (7 - bit % 8)) & 1 != 0);
        }
        let payload = payload.finish();
        let mut packet = vec![drm_crc8_bits(&payload, 0, 41).unwrap()];
        packet.extend_from_slice(&payload);
        (packet, 8 + 41 + sbr_bits, sbr_bits)
    }

    fn drm_stereo_sbr_packet() -> (Vec<u8>, usize) {
        let header = drm_default_sbr_header(96_000);
        let tables = LdSbrFrequencyTables::from_header(&header, 96_000).unwrap();
        let level_zero = huffman_code(SbrHuffmanBook::EnvelopeLevel30Frequency, 0).unwrap();
        let balance_zero = huffman_code(SbrHuffmanBook::EnvelopeBalance30Frequency, 0).unwrap();
        let mut frame = BitWriter::new();
        frame.write_bool(false); // no data extra
        frame.write_bool(true); // coupling
        frame.write(0, 2); // FIXFIX
        frame.write(0, 2); // one envelope
        frame.write_bool(true);
        for _ in 0..4 {
            frame.write_bool(false); // level/noise frequency directions, both channels
        }
        for _ in 0..tables.noise_band_count() {
            frame.write(0, 2);
        }
        frame.write(0, 6);
        for _ in 1..tables.high_band_count() {
            write_code(&mut frame, &level_zero);
        }
        frame.write(0, 5);
        for _ in 1..tables.noise_band_count() {
            write_code(&mut frame, &level_zero);
        }
        frame.write(6, 5); // centered envelope balance
        for _ in 1..tables.high_band_count() {
            write_code(&mut frame, &balance_zero);
        }
        frame.write(6, 5); // centered noise balance
        for _ in 1..tables.noise_band_count() {
            write_code(&mut frame, &balance_zero);
        }
        frame.write_bool(false);
        frame.write_bool(false);
        frame.write_bool(false);
        let frame_bits = frame.bits_written();
        let frame = frame.finish();
        let mut protected = BitWriter::new();
        protected.write_bool(false);
        for bit in 0..frame_bits {
            protected.write_bool((frame[bit / 8] >> (7 - bit % 8)) & 1 != 0);
        }
        let protected_bits = protected.bits_written();
        let protected = protected.finish();
        let mut normal = BitWriter::new();
        normal.write(
            drm_crc8_bits(&protected, 0, protected_bits).unwrap().into(),
            8,
        );
        for bit in 0..protected_bits {
            normal.write_bool((protected[bit / 8] >> (7 - bit % 8)) & 1 != 0);
        }
        let sbr_bits = normal.bits_written();
        let normal = normal.finish();

        let mut payload = BitWriter::new();
        payload.write_bool(false);
        payload.write(0, 2);
        payload.write_bool(false);
        payload.write(0, 6);
        payload.write_bool(false);
        payload.write(0, 2);
        for _ in 0..2 {
            payload.write_bool(false);
            payload.write_bool(false);
            payload.write(0, 8);
            payload.write(0, 14);
            payload.write(0, 6);
        }
        for bit in (0..sbr_bits).rev() {
            payload.write_bool((normal[bit / 8] >> (7 - bit % 8)) & 1 != 0);
        }
        let payload = payload.finish();
        let mut packet = vec![drm_crc8_bits(&payload, 0, 73).unwrap()];
        packet.extend_from_slice(&payload);
        (packet, 8 + 73 + sbr_bits)
    }

    fn config(coding: u8, sbr: bool, mode: u8, sample_rate: u8, coder: u8) -> Vec<u8> {
        let mut writer = BitWriter::new();
        writer.write(coding as u32, 2);
        writer.write_bool(sbr);
        writer.write(mode as u32, 2);
        writer.write(sample_rate as u32, 3);
        writer.write_bool(true);
        writer.write_bool(false);
        writer.write(coder as u32, 5);
        writer.write_bool(false);
        writer.finish()
    }

    #[test]
    fn parses_drm_aac_stereo_configuration() {
        let parsed = DrmAudioConfig::parse(&config(0, false, 2, 5, 0)).unwrap();
        assert_eq!(parsed.audio_coding, DrmAudioCoding::Aac);
        assert_eq!(parsed.audio_mode, DrmAudioMode::Stereo);
        assert_eq!(parsed.sampling_frequency, 48_000);
        assert_eq!(parsed.channel_configuration, 2);
        assert_eq!(parsed.samples_per_frame, Some(960));
        assert_eq!(parsed.error_protection_config, Some(1));
        assert_eq!(parsed.bits_read, 16);
    }

    #[test]
    fn derives_implicit_drm_surround_212_configuration() {
        let parsed = DrmAudioConfig::parse(&config(0, true, 0, 5, 8)).unwrap();
        assert!(parsed.drm_surround);
        assert_eq!(parsed.surround_mode, Some(DrmSurroundMode::FiveOne));
        let surround = parsed.surround_specific_config().unwrap().unwrap();
        assert_eq!(surround.sampling_frequency, 96_000);
        assert_eq!(surround.time_slots, 30);
        assert_eq!(surround.frequency_resolution, 28);
        assert_eq!((surround.input_channels, surround.output_channels), (1, 2));
    }

    #[test]
    fn rejects_reserved_drm_surround_mode() {
        assert!(matches!(
            DrmAudioConfig::parse(&config(0, true, 0, 5, 4)),
            Err(DrmError::ReservedMpegSurroundMode(1))
        ));
    }

    #[test]
    fn parses_xhe_sampling_rate_mapping() {
        let parsed = DrmAudioConfig::parse(&config(3, true, 0, 0, 0)).unwrap();
        assert_eq!(parsed.audio_coding, DrmAudioCoding::XheAac);
        assert_eq!(parsed.sampling_frequency_index, 0x1b);
        assert_eq!(parsed.sampling_frequency, 9_600);
        assert!(!parsed.sbr);
    }

    #[test]
    fn maps_drm_xhe_static_mono_config_to_usac() {
        let mut writer = BitWriter::new();
        writer.write(3, 2); // xHE-AAC
        writer.write_bool(false); // reserved SBR flag
        writer.write(0, 2); // mono
        writer.write(7, 3); // 48 kHz
        writer.write_bool(false); // text
        writer.write_bool(false); // enhancement
        writer.write(0, 5); // coder field
        writer.write_bool(false); // reserved
        writer.write(0, 2); // DRM index 0 -> USAC core index 1
        writer.write_bool(true); // noise filling

        let static_config = writer.finish();
        let (drm, asc) = DrmAudioConfig::parse_xhe_with_static_config(&static_config).unwrap();
        assert_eq!(drm.sampling_frequency, 48_000);
        assert_eq!(asc.audio_object_type, 42);
        let usac = asc.usac_config.as_ref().unwrap();
        assert_eq!(usac.core_sbr_frame_length_index, 1);
        assert_eq!(usac.core_frame_length, 1024);
        assert_eq!(usac.output_frame_length, 1024);
        assert_eq!(
            usac.elements,
            vec![UsacElementConfig::SingleChannel {
                noise_filling: true,
                sbr: None,
            }]
        );
        let mut decoder = DrmXheDecoder::from_static_config(&static_config).unwrap();
        let mut payload = BitWriter::new();
        payload.write_bool(true); // independency flag
        payload.write_bool(false); // FD core
        payload.write_bool(false); // no TNS
        payload.write(0, 8); // global gain
        payload.write(0, 8); // noise level/offset
        payload.write(0, 2); // ONLY_LONG
        payload.write_bool(false); // window shape
        payload.write(0, 6); // max_sfb
        payload.write_bool(false); // no FAC
        let pcm = decoder.decode_interleaved_f32(&payload.finish()).unwrap();
        assert_eq!(pcm, vec![0.0; 1024]);
    }

    #[test]
    fn maps_drm_xhe_stereo_mps_static_config() {
        let mut writer = BitWriter::new();
        writer.write(3, 2); // xHE-AAC
        writer.write_bool(false);
        writer.write(2, 2); // stereo
        writer.write(7, 3); // 48 kHz
        writer.write_bool(false);
        writer.write_bool(false);
        writer.write(0, 5);
        writer.write_bool(false);
        writer.write(1, 2); // DRM index 1 -> USAC index 2, 8:3 SBR
        writer.write_bool(false); // noise filling
        writer.write_bool(false); // harmonic SBR
        writer.write_bool(false); // inter-TES
        writer.write_bool(false); // PVC
        writer.write(5, 4); // SBR start
        writer.write(8, 4); // SBR stop
        writer.write_bool(false); // no extra 1
        writer.write_bool(false); // no extra 2
        writer.write(2, 2); // stereoConfigIndex with residual
        writer.write(1, 3); // 28 MPS parameter bands
        writer.write(0, 3); // fixed downmix gain
        writer.write_bool(true); // DRM temp shaping -> config 3
        writer.write_bool(true); // high-rate mode
        writer.write_bool(true); // phase coding
        writer.write_bool(true); // OTT phase bands present
        writer.write(8, 5);
        writer.write(6, 5); // residual bands
        writer.write_bool(true); // pseudo LR

        let (_, asc) = DrmAudioConfig::parse_xhe_with_static_config(&writer.finish()).unwrap();
        let usac = asc.usac_config.as_ref().unwrap();
        assert_eq!(usac.core_sbr_frame_length_index, 2);
        assert_eq!(usac.core_frame_length, 768);
        assert_eq!(usac.output_frame_length, 2048);
        let mixed_elements = [
            UsacElementConfig::SingleChannel {
                noise_filling: false,
                sbr: None,
            },
            usac.elements[0].clone(),
        ];
        for element in &mixed_elements {
            if let UsacElementConfig::ChannelPair {
                stereo_config_index,
                mps212: Some(mps),
                ..
            } = element
            {
                assert_eq!(*stereo_config_index, 2);
                assert_eq!(mps.frequency_resolution_bands, 28);
                assert_eq!(mps.temporal_shape_config, 3);
                assert_eq!(mps.decorrelation_config, 0);
                assert_eq!(mps.residual_bands, Some(6));
                assert!(mps.pseudo_lr);
            }
        }
        assert!(matches!(
            usac.elements[0],
            UsacElementConfig::ChannelPair {
                mps212: Some(_),
                ..
            }
        ));
        crate::decoder::AacLcDecoder::from_audio_specific_config(&asc).unwrap();
    }

    #[test]
    fn validates_drm_crc() {
        let protected = b"123456789";
        let crc = drm_crc8(protected);
        assert_eq!(crc, 0x4b);
        assert_eq!(check_drm_crc(crc, protected), Ok(()));
        assert!(matches!(
            check_drm_crc(crc ^ 1, protected),
            Err(DrmError::CrcMismatch { .. })
        ));
        assert_eq!(drm_crc8_bits(&[0xd3, 0x6a, 0xf0], 3, 13), Ok(0x7c));
        assert_eq!(check_drm_crc_bits(0x7c, &[0xd3, 0x6a, 0xf0], 3, 13), Ok(()));
        assert!(matches!(
            drm_crc8_bits(&[0], 4, 5),
            Err(DrmError::InvalidCrcRegion { .. })
        ));
        assert_eq!(
            reverse_drm_sbr_payload_bits(&[0xb6, 0x69], 3, 11),
            Ok(vec![0x59, 0xa0])
        );
    }

    #[test]
    fn decodes_crc_protected_drm_aac_mono_transactionally() {
        let mut decoder = DrmAacDecoder::from_sdc_config(&config(0, false, 0, 5, 0)).unwrap();
        let mut writer = BitWriter::new();
        writer.write_bool(false); // ICS reserved
        writer.write(0, 2); // ONLY_LONG
        writer.write_bool(false); // sine
        writer.write(0, 6); // max_sfb
        writer.write_bool(false); // prediction absent
        writer.write_bool(false); // TNS absent
        writer.write_bool(false); // LTP absent
        writer.write(0, 8); // global gain
        writer.write(0, 14); // HCR payload length
        writer.write(0, 6); // longest codeword
        let payload = writer.finish();
        let crc = drm_crc8_bits(&payload, 0, 41).unwrap();
        let mut packet = vec![crc];
        packet.extend_from_slice(&payload);
        assert_eq!(
            decoder.decode_crc_protected_mono_f32(&packet).unwrap(),
            vec![0.0; 960]
        );
        assert_eq!(
            decoder
                .decode_crc_protected_interleaved_i16(&packet)
                .unwrap(),
            vec![0; 960]
        );

        packet[1] ^= 0x80;
        assert!(matches!(
            decoder.decode_crc_protected_mono_f32(&packet),
            Err(DrmAacDecodeError::Config(DrmError::CrcMismatch { .. }))
        ));
    }

    #[test]
    fn decodes_crc_protected_drm_aac_stereo() {
        let mut decoder = DrmAacDecoder::from_sdc_config(&config(0, false, 2, 5, 0)).unwrap();
        let mut writer = BitWriter::new();
        writer.write_bool(false); // ICS reserved
        writer.write(0, 2); // ONLY_LONG
        writer.write_bool(false); // sine
        writer.write(0, 6); // max_sfb
        writer.write_bool(false); // prediction absent
        writer.write(0, 2); // no M/S
        for _ in 0..2 {
            writer.write_bool(false); // TNS absent
            writer.write_bool(false); // LTP absent
            writer.write(0, 8); // global gain
            writer.write(0, 14); // HCR payload length
            writer.write(0, 6); // longest codeword
        }
        let payload = writer.finish();
        let mut packet = vec![drm_crc8_bits(&payload, 0, 73).unwrap()];
        packet.extend_from_slice(&payload);
        let pcm = decoder
            .decode_crc_protected_interleaved_f32(&packet)
            .unwrap();
        assert_eq!(pcm, vec![0.0; 1920]);
        let pcm = decoder
            .decode_crc_protected_interleaved_i16(&packet)
            .unwrap();
        assert_eq!(pcm, vec![0; 1920]);
    }

    #[test]
    fn extracts_and_reverses_trailing_drm_sbr_bits() {
        let mut decoder = DrmAacDecoder::from_sdc_config(&config(0, true, 0, 5, 0)).unwrap();
        let mut writer = BitWriter::new();
        writer.write_bool(false);
        writer.write(0, 2);
        writer.write_bool(false);
        writer.write(0, 6);
        writer.write_bool(false);
        writer.write_bool(false);
        writer.write_bool(false);
        writer.write(0, 8);
        writer.write(0, 14);
        writer.write(0, 6); // 41 core bits
        writer.write(0b010_1100_1101, 11); // reversed DRM storage order
        let payload = writer.finish();
        let mut packet = vec![drm_crc8_bits(&payload, 0, 41).unwrap()];
        packet.extend_from_slice(&payload);

        let frame = decoder
            .decode_crc_protected_interleaved_f32_with_sbr(&packet, 8 + 41 + 11)
            .unwrap();
        assert_eq!(frame.samples, vec![0.0; 960]);
        assert_eq!(frame.sbr_payload_bits, 11);
        assert_eq!(frame.reversed_sbr_payload, [0xb3, 0x40]);
    }

    #[test]
    fn renders_crc_protected_drm_mono_sbr_transactionally() {
        let mut decoder = DrmAacDecoder::from_sdc_config(&config(0, true, 0, 5, 0)).unwrap();
        let header = drm_default_sbr_header(96_000);
        let tables = LdSbrFrequencyTables::from_header(&header, 96_000).unwrap();
        let zero = huffman_code(SbrHuffmanBook::EnvelopeLevel30Frequency, 0).unwrap();

        let mut frame_data = BitWriter::new();
        frame_data.write_bool(false); // no data extra
        frame_data.write(0, 2); // FIXFIX
        frame_data.write(0, 2); // one envelope
        frame_data.write_bool(true); // high frequency resolution
        frame_data.write_bool(false); // envelope frequency direction
        frame_data.write_bool(false); // noise frequency direction
        for _ in 0..tables.noise_band_count() {
            frame_data.write(0, 2); // inverse filtering off
        }
        frame_data.write(0, 6); // absolute envelope
        for _ in 1..tables.high_band_count() {
            write_code(&mut frame_data, &zero);
        }
        frame_data.write(0, 5); // absolute noise floor
        for _ in 1..tables.noise_band_count() {
            write_code(&mut frame_data, &zero);
        }
        frame_data.write_bool(false); // no harmonics
        frame_data.write_bool(false); // no extended data
        let frame_data_bits = frame_data.bits_written();
        let frame_data = frame_data.finish();

        let mut protected = BitWriter::new();
        protected.write_bool(false); // use DRM default header
        for bit in 0..frame_data_bits {
            protected.write_bool((frame_data[bit / 8] >> (7 - bit % 8)) & 1 != 0);
        }
        let protected_bits = protected.bits_written();
        let protected = protected.finish();
        let inner_crc = drm_crc8_bits(&protected, 0, protected_bits).unwrap();
        let mut normal_sbr = BitWriter::new();
        normal_sbr.write(inner_crc.into(), 8);
        for bit in 0..protected_bits {
            normal_sbr.write_bool((protected[bit / 8] >> (7 - bit % 8)) & 1 != 0);
        }
        let sbr_bits = normal_sbr.bits_written();
        let normal_sbr = normal_sbr.finish();

        let mut payload = BitWriter::new();
        payload.write_bool(false);
        payload.write(0, 2);
        payload.write_bool(false);
        payload.write(0, 6);
        payload.write_bool(false);
        payload.write_bool(false);
        payload.write_bool(false);
        payload.write(0, 8);
        payload.write(0, 14);
        payload.write(0, 6); // 41 core bits
        for bit in (0..sbr_bits).rev() {
            payload.write_bool((normal_sbr[bit / 8] >> (7 - bit % 8)) & 1 != 0);
        }
        let payload = payload.finish();
        let mut packet = vec![drm_crc8_bits(&payload, 0, 41).unwrap()];
        packet.extend_from_slice(&payload);

        let decoded = decoder
            .decode_crc_protected_interleaved_f32_rendering_sbr(&packet, 8 + 41 + sbr_bits)
            .unwrap();
        assert_eq!(decoded.samples.len(), 1920);
        assert!(decoded.samples.iter().all(|sample| sample.is_finite()));
        assert_eq!(decoded.sbr_payload_bits, sbr_bits);

        decoder.sbr_mono_parser = None;
        decoder.sbr_mono_processor = None;
        let core_only = decoder
            .decode_crc_protected_interleaved_f32_rendering_sbr(&packet, 8 + 41 + sbr_bits)
            .unwrap();
        assert_eq!(core_only.samples.len(), 960);
        assert_eq!(core_only.sbr_payload_bits, sbr_bits);
    }

    #[test]
    fn renders_drm_parametric_stereo_from_sbr_extension() {
        let mut ps = BitWriter::new();
        ps.write(2, 2); // EXTENSION_ID_PS_CODING
        ps.write_bool(true); // PS header present
        ps.write_bool(false); // IID disabled
        ps.write_bool(false); // ICC disabled
        ps.write_bool(false); // PS extension disabled
        ps.write_bool(false); // fixed borders
        ps.write(0, 2); // implicit single zero-parameter envelope
        let ps = ps.finish();
        let (packet, packet_bits, sbr_bits) = drm_sbr_packet_with_extension(&ps);
        let mut decoder = DrmAacDecoder::from_sdc_config(&config(0, true, 1, 5, 0)).unwrap();

        let decoded = decoder
            .decode_crc_protected_interleaved_f32_rendering_sbr(&packet, packet_bits)
            .unwrap();
        assert_eq!(decoded.samples.len(), 3840);
        assert!(decoded.samples.iter().all(|sample| sample.is_finite()));
        assert_eq!(decoded.sbr_payload_bits, sbr_bits);
        for stereo in decoded.samples.chunks_exact(2) {
            assert_eq!(stereo[0], stereo[1]);
        }

        let (packet, packet_bits, _) = drm_sbr_packet_with_extension(&[]);
        let repeated = decoder
            .decode_crc_protected_interleaved_f32_rendering_sbr(&packet, packet_bits)
            .unwrap();
        assert_eq!(repeated.samples.len(), 3840);
        assert!(repeated.samples.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn renders_coupled_drm_stereo_sbr() {
        let (packet, packet_bits) = drm_stereo_sbr_packet();
        let mut decoder = DrmAacDecoder::from_sdc_config(&config(0, true, 2, 5, 0)).unwrap();
        let decoded = decoder
            .decode_crc_protected_interleaved_f32_rendering_sbr(&packet, packet_bits)
            .unwrap();
        assert_eq!(decoded.samples.len(), 3840);
        assert!(decoded.samples.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn renders_attached_drm_surround_payload_from_sbr_qmf() {
        let (packet, packet_bits, _) = drm_sbr_packet_with_extension(&[]);
        let mut decoder = DrmAacDecoder::from_sdc_config(&config(0, true, 0, 5, 8)).unwrap();
        decoder
            .configure_surround(SpatialSpecificConfig::default_212(96_000, 30).unwrap())
            .unwrap();
        let mut spatial = BitWriter::new();
        spatial.write_bool(false); // fixed framing
        spatial.write(0, 3); // one parameter set
        spatial.write_bool(true); // independent
        spatial.write(0, 2); // default CLD
        spatial.write(0, 2); // default ICC
        spatial.write(0, 2); // no smoothing
        let spatial_bits = spatial.bits_written();
        let spatial = spatial.finish();

        let mut missing = DrmAacDecoder::from_sdc_config(&config(0, true, 0, 5, 8)).unwrap();
        assert_eq!(
            missing
                .decode_crc_protected_interleaved_f32_rendering_sbr_and_surround(
                    &packet,
                    packet_bits,
                    &spatial,
                    spatial_bits,
                )
                .unwrap_err(),
            DrmAacDecodeError::Config(DrmError::MissingSurroundConfiguration)
        );

        let decoded = decoder
            .decode_crc_protected_interleaved_f32_rendering_sbr_and_surround(
                &packet,
                packet_bits,
                &spatial,
                spatial_bits,
            )
            .unwrap();
        assert_eq!(decoded.samples.len(), 3840);
        assert!(decoded.samples.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn parses_reversed_drm_sbr_crc_and_frame_payload() {
        let mut body = BitWriter::new();
        body.write_bool(false); // no SBR header
        body.write(0b10110, 5); // frame data
        let body_bytes = body.finish();
        let crc = drm_crc8_bits(&body_bytes, 0, 6).unwrap();
        let mut payload = BitWriter::new();
        payload.write(crc.into(), 8);
        payload.write_bool(false);
        payload.write(0b10110, 5);

        let parsed = parse_reversed_drm_sbr_payload(&payload.finish(), 14).unwrap();
        assert_eq!(parsed.transmitted_crc, Some(u16::from(crc)));
        assert!(!parsed.header_present);
        assert!(parsed.header.is_none());
        assert_eq!(parsed.frame_data_bits, 5);
        assert_eq!(parsed.frame_data, [0b1011_0000]);
    }

    #[test]
    fn constructs_drm_sbr_parser_from_fdk_default_header() {
        let high = drm_default_sbr_header(96_000);
        assert_eq!((high.start_frequency, high.stop_frequency), (4, 3));
        assert_eq!(high.frequency_scale, Some(0));
        DrmSbrMonoParser::new(48_000).unwrap();

        let mid = drm_default_sbr_header(48_000);
        assert_eq!((mid.start_frequency, mid.stop_frequency), (7, 3));
        DrmSbrMonoParser::new(24_000).unwrap();
        assert!(matches!(DrmSbrMonoParser::new(1), Err(DrmSbrError::Sbr(_))));
        assert!(matches!(
            DrmSbrStereoParser::new(1),
            Err(DrmSbrError::Sbr(_))
        ));
    }

    #[test]
    fn validates_audio_config_modes_rates_and_static_coding() {
        assert_eq!(
            DrmAudioConfig::aac(44_100, DrmAudioMode::Mono, false).unwrap_err(),
            DrmError::ReservedSamplingRate(7)
        );
        assert_eq!(
            DrmAudioConfig::aac(48_000, DrmAudioMode::ParametricStereo, false).unwrap_err(),
            DrmError::ParametricStereoRequiresSbr
        );
        assert_eq!(
            DrmAudioConfig::parse(&config(0, false, 1, 5, 0)).unwrap_err(),
            DrmError::ParametricStereoRequiresSbr
        );
        assert_eq!(
            DrmAudioConfig::parse(&config(0, false, 3, 5, 0)).unwrap_err(),
            DrmError::InvalidAudioMode(3)
        );
        assert_eq!(
            DrmAudioConfig::parse(&config(3, false, 1, 7, 0)).unwrap_err(),
            DrmError::InvalidAudioMode(1)
        );
        let mut parametric_stereo_static = config(3, false, 1, 7, 0);
        parametric_stereo_static.push(0);
        assert_eq!(
            DrmAudioConfig::parse_xhe_with_static_config(&parametric_stereo_static).unwrap_err(),
            DrmError::InvalidAudioMode(1)
        );

        for coding in [1, 2] {
            let parsed = DrmAudioConfig::parse(&config(coding, true, 3, 0, 31)).unwrap();
            assert_eq!(parsed.audio_mode, DrmAudioMode::Mono);
            assert_eq!(parsed.surround_mode, None);
            assert_eq!(parsed.surround_specific_config().unwrap(), None);
        }
        for (coder, expected) in [
            (12, DrmSurroundMode::SevenOne),
            (28, DrmSurroundMode::StreamDefined),
        ] {
            let parsed = DrmAudioConfig::parse(&config(0, false, 0, 5, coder)).unwrap();
            assert_eq!(parsed.surround_mode, Some(expected));
        }

        let xhe = DrmAudioConfig::parse(&config(3, false, 0, 7, 0)).unwrap();
        assert_eq!(
            xhe.to_bytes().unwrap_err(),
            DrmError::StaticConfigRequiresDrmAac
        );
        assert_eq!(
            DrmAudioConfig::parse_xhe_with_static_config(&config(0, false, 0, 5, 0)).unwrap_err(),
            DrmError::StaticConfigRequiresXheAac
        );

        let mut invalid_ps = DrmAudioConfig::aac(48_000, DrmAudioMode::Mono, false).unwrap();
        invalid_ps.audio_mode = DrmAudioMode::ParametricStereo;
        assert_eq!(
            invalid_ps.to_bytes().unwrap_err(),
            DrmError::ParametricStereoRequiresSbr
        );
    }

    #[test]
    fn rejects_invalid_decoder_and_sbr_facade_inputs() {
        assert!(matches!(
            DrmAacDecoder::from_sdc_config(&config(3, false, 0, 7, 0)),
            Err(DrmAacDecodeError::Config(
                DrmError::StaticConfigRequiresDrmAac
            ))
        ));

        let mut mono = DrmAacDecoder::from_sdc_config(&config(0, false, 0, 5, 0)).unwrap();
        let spatial = SpatialSpecificConfig::default_212(48_000, 30).unwrap();
        assert!(matches!(
            mono.configure_surround(spatial),
            Err(DrmAacDecodeError::Config(DrmError::SurroundNotSignaled))
        ));
        for packet_bits in [7, 9] {
            assert!(matches!(
                mono.decode_crc_protected_interleaved_f32_with_sbr(&[0], packet_bits),
                Err(DrmAacDecodeError::Config(
                    DrmError::InvalidAudioPacketBits { .. }
                ))
            ));
        }
        assert!(matches!(
            mono.decode_crc_protected_interleaved_f32(&[]),
            Err(DrmAacDecodeError::Config(DrmError::Bit(_)))
        ));

        let mut core = BitWriter::new();
        core.write_bool(false);
        core.write(0, 2);
        core.write_bool(false);
        core.write(0, 6);
        core.write_bool(false);
        core.write_bool(false);
        core.write_bool(false);
        core.write(0, 8);
        core.write(0, 14);
        core.write(0, 6);
        let payload = core.finish();
        let mut packet = vec![drm_crc8_bits(&payload, 0, 41).unwrap()];
        packet.extend_from_slice(&payload);
        assert!(matches!(
            mono.decode_crc_protected_interleaved_f32_with_sbr(&packet, 48),
            Err(DrmAacDecodeError::Config(
                DrmError::InvalidAudioPacketBits { .. }
            ))
        ));
        let decoded = mono
            .decode_crc_protected_interleaved_f32_with_sbr(&packet, 49)
            .unwrap();
        assert_eq!(decoded.samples, vec![0.0; 960]);
        assert_eq!(decoded.sbr_payload_bits, 0);
        assert!(decoded.reversed_sbr_payload.is_empty());

        let mut ps = DrmAacDecoder::from_sdc_config(&config(0, true, 1, 5, 0)).unwrap();
        assert_eq!(
            ps.decode_crc_protected_interleaved_f32(&[]).unwrap_err(),
            DrmAacDecodeError::Config(DrmError::ParametricStereoRequiresSbr)
        );
        assert_eq!(
            ps.decode_crc_protected_interleaved_i16(&[]).unwrap_err(),
            DrmAacDecodeError::Config(DrmError::Bit(BitError::UnexpectedEof {
                needed_bits: 8,
                remaining_bits: 0,
            }))
        );
        assert_eq!(
            ps.decode_crc_protected_interleaved_i16(&[0]).unwrap_err(),
            DrmAacDecodeError::Config(DrmError::ParametricStereoRequiresSbr)
        );

        assert!(matches!(
            reverse_drm_sbr_payload_bits(&[0], usize::MAX, 1),
            Err(DrmError::InvalidSbrRegion { .. })
        ));
        for bit_len in [8, 9] {
            assert!(matches!(
                parse_reversed_drm_sbr_payload(&[], bit_len),
                Err(DrmError::InvalidSbrRegion { .. })
            ));
        }
        let header = crate::asc::LdSbrHeader {
            start_frequency: 5,
            stop_frequency: 8,
            ..crate::asc::LdSbrHeader::default()
        };
        let mut truncated_region = BitWriter::new();
        truncated_region.write(0, 8); // patched CRC
        truncated_region.write_bool(true);
        header.write(&mut truncated_region).unwrap();
        let mut truncated_region = truncated_region.finish();
        truncated_region[0] = drm_crc8_bits(&truncated_region, 8, 1).unwrap();
        assert!(matches!(
            parse_reversed_drm_sbr_payload(&truncated_region, 9),
            Err(DrmError::InvalidSbrRegion {
                bit_len: 0,
                available_bits: 9,
                ..
            })
        ));

        let low = drm_default_sbr_header(24_000);
        assert_eq!((low.start_frequency, low.stop_frequency), (5, 0));
        assert_eq!(
            DrmSbrMonoParser::new(u32::MAX).unwrap_err(),
            DrmSbrError::SamplingFrequencyOverflow
        );
        assert_eq!(
            DrmSbrStereoParser::new(u32::MAX).unwrap_err(),
            DrmSbrError::SamplingFrequencyOverflow
        );

        let surround = DrmAudioConfig::parse(&config(0, false, 0, 5, 8)).unwrap();
        assert_eq!(
            surround
                .surround_specific_config()
                .unwrap()
                .unwrap()
                .sampling_frequency,
            48_000
        );
        let mut invalid_surround = surround;
        invalid_surround.sampling_frequency = 0;
        assert_eq!(
            invalid_surround.surround_specific_config(),
            Err(DrmError::Sac(SacError::InvalidSamplingFrequency))
        );
    }

    #[test]
    fn maps_remaining_xhe_frame_length_ratios_and_i16_output() {
        fn mono_static_config(signaled_index: u8) -> Vec<u8> {
            let mut writer = BitWriter::new();
            writer.write(3, 2);
            writer.write_bool(false);
            writer.write(0, 2);
            writer.write(7, 3);
            writer.write_bool(false);
            writer.write_bool(false);
            writer.write(0, 5);
            writer.write_bool(false);
            writer.write(signaled_index.into(), 2);
            writer.write_bool(false);
            if signaled_index >= 1 {
                writer.write_bool(false);
                writer.write_bool(false);
                writer.write_bool(false);
                writer.write(5, 4);
                writer.write(8, 4);
                writer.write_bool(false);
                writer.write_bool(false);
            }
            writer.finish()
        }

        for (signaled, core, output, ratio) in [(2, 1024, 2048, 3), (3, 1024, 4096, 1)] {
            let (_, asc) =
                DrmAudioConfig::parse_xhe_with_static_config(&mono_static_config(signaled))
                    .unwrap();
            let usac = asc.usac_config.unwrap();
            assert_eq!(usac.core_frame_length, core);
            assert_eq!(usac.output_frame_length, output);
            assert_eq!(usac.sbr_ratio_index, ratio);
        }

        let static_config = mono_static_config(0);
        let mut decoder = DrmXheDecoder::from_static_config(&static_config).unwrap();
        let mut payload = BitWriter::new();
        payload.write_bool(true);
        payload.write_bool(false);
        payload.write_bool(false);
        payload.write(0, 8);
        payload.write(0, 2);
        payload.write_bool(false);
        payload.write(0, 6);
        payload.write_bool(false);
        assert_eq!(
            decoder.decode_interleaved_i16(&payload.finish()).unwrap(),
            vec![0; 1024]
        );
    }

    #[test]
    fn converts_and_formats_all_drm_error_layers() {
        let bit = BitError::UnexpectedEof {
            needed_bits: 1,
            remaining_bits: 0,
        };
        let drm_errors = [
            DrmError::from(bit.clone()),
            DrmError::from(AscError::InvalidAudioObjectType(0)),
            DrmError::from(SacError::InvalidSamplingFrequency),
            DrmError::CrcMismatch {
                expected: 0,
                calculated: 1,
            },
            DrmError::InvalidAudioMode(3),
            DrmError::InvalidAudioPacketBits {
                declared_bits: 9,
                available_bits: 8,
            },
            DrmError::InvalidCrcRegion {
                bit_offset: 8,
                bit_len: 1,
                available_bits: 8,
            },
            DrmError::InvalidSbrRegion {
                bit_offset: 8,
                bit_len: 1,
                available_bits: 8,
            },
            DrmError::MissingParametricStereoPayload,
            DrmError::MissingSurroundConfiguration,
            DrmError::ParametricStereoRequiresSbr,
            DrmError::ReservedSamplingRate(4),
            DrmError::ReservedMpegSurroundMode(1),
            DrmError::SamplingFrequencyOverflow,
            DrmError::SurroundNotSignaled,
            DrmError::StaticConfigRequiresXheAac,
            DrmError::StaticConfigRequiresDrmAac,
        ];
        for error in drm_errors {
            assert!(!error.to_string().is_empty());
        }

        let xhe_errors = [
            DrmXheDecodeError::from(DrmError::StaticConfigRequiresXheAac),
            DrmXheDecodeError::from(DecodeError::UnsupportedSamplingFrequencyIndex(15)),
            DrmXheDecodeError::from(UsacDecodeError::UnsupportedConfiguration),
        ];
        for error in xhe_errors {
            assert!(!error.to_string().is_empty());
        }

        let sbr_errors = [
            DrmSbrError::from(DrmError::InvalidSbrRegion {
                bit_offset: 0,
                bit_len: 1,
                available_bits: 0,
            }),
            DrmSbrError::from(SbrError::InvalidGrid),
            DrmSbrError::SamplingFrequencyOverflow,
        ];
        for error in sbr_errors {
            assert!(!error.to_string().is_empty());
        }

        let aac_errors = [
            DrmAacDecodeError::from(DrmError::StaticConfigRequiresDrmAac),
            DrmAacDecodeError::from(DecodeError::UnsupportedSamplingFrequencyIndex(15)),
            DrmAacDecodeError::from(DrmSbrError::SamplingFrequencyOverflow),
            DrmAacDecodeError::from(LdSbrProcessingError::MissingRightChannel),
            DrmAacDecodeError::from(PsError::MissingInitialHeader),
            DrmAacDecodeError::from(SacDecodeError::UnsupportedLayout),
        ];
        for error in aac_errors {
            assert!(!error.to_string().is_empty());
        }
    }
}
