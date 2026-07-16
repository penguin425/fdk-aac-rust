//! Pure Rust MPEG-4 AudioSpecificConfig support.
//!
//! This currently implements the common AAC-LC / HE-AAC configuration subset:
//! Audio Object Type, sampling frequency, channel configuration, optional SBR/PS
//! extension signalling, and the first GASpecificConfig flags.

use std::fmt;

use crate::adts::{sample_rate_from_index, sample_rate_index};
use crate::bits::{BitError, BitReader, BitWriter};

const AOT_ESCAPE: u8 = 31;
const AOT_AAC_LC: u8 = 2;
const AOT_SBR: u8 = 5;
const AOT_PS: u8 = 29;
const AOT_USAC: u8 = 42;

/// MPEG-4 AudioSpecificConfig.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioSpecificConfig {
    pub audio_object_type: u8,
    pub sampling_frequency_index: u8,
    pub sampling_frequency: u32,
    pub channel_configuration: u8,
    pub extension: Option<AudioSpecificConfigExtension>,
    pub ga_specific: Option<GaSpecificConfig>,
    pub eld_specific: Option<EldSpecificConfig>,
    pub usac_config: Option<UsacConfig>,
    pub error_protection_config: Option<u8>,
    pub program_config: Option<ProgramConfig>,
    pub bits_read: usize,
}

impl AudioSpecificConfig {
    pub fn aac_lc(sample_rate: u32, channel_configuration: u8) -> Result<Self, AscError> {
        let sampling_frequency_index =
            sample_rate_index(sample_rate).ok_or(AscError::UnsupportedSampleRate(sample_rate))?;
        Ok(Self {
            audio_object_type: AOT_AAC_LC,
            sampling_frequency_index,
            sampling_frequency: sample_rate,
            channel_configuration,
            extension: None,
            ga_specific: Some(GaSpecificConfig::default()),
            eld_specific: None,
            usac_config: None,
            error_protection_config: None,
            program_config: None,
            bits_read: 0,
        })
    }

    pub fn parse(input: &[u8]) -> Result<Self, AscError> {
        let mut reader = BitReader::new(input);
        Self::parse_from_reader(&mut reader)
    }

    pub(crate) fn parse_from_reader(reader: &mut BitReader<'_>) -> Result<Self, AscError> {
        let start = reader.bits_read();
        let audio_object_type = read_audio_object_type(reader)?;
        let (sampling_frequency_index, sampling_frequency) = read_sampling_frequency(reader)?;
        let channel_configuration = reader.read_u8(4)?;
        if !matches!(channel_configuration, 0..=7 | 11 | 12 | 14) {
            return Err(AscError::InvalidChannelConfiguration(channel_configuration));
        }

        let mut core_audio_object_type = audio_object_type;
        let core_sampling_frequency_index = sampling_frequency_index;
        let core_sampling_frequency = sampling_frequency;
        let mut extension = None;

        if audio_object_type == AOT_SBR || audio_object_type == AOT_PS {
            let extension_audio_object_type = audio_object_type;
            let (extension_sampling_frequency_index, extension_sampling_frequency) =
                read_sampling_frequency(reader)?;
            core_audio_object_type = read_audio_object_type(reader)?;
            extension = Some(AudioSpecificConfigExtension {
                audio_object_type: extension_audio_object_type,
                sampling_frequency_index: extension_sampling_frequency_index,
                sampling_frequency: extension_sampling_frequency,
                ps_present: audio_object_type == AOT_PS,
            });
        }

        let mut program_config = None;
        let usac_config = if core_audio_object_type == AOT_USAC {
            Some(UsacConfig::parse(reader)?)
        } else {
            None
        };
        let ga_specific = if is_ga_specific(core_audio_object_type) {
            let mut ga = GaSpecificConfig::parse(reader)?;
            if channel_configuration == 0 {
                program_config = Some(ProgramConfig::parse_from_reader(reader)?);
            }
            ga.parse_tail(reader, core_audio_object_type)?;
            Some(ga)
        } else {
            None
        };
        let eld_specific = if core_audio_object_type == 39 {
            Some(EldSpecificConfig::parse(reader, channel_configuration)?)
        } else {
            None
        };

        let error_protection_config = if is_error_resilient(core_audio_object_type) {
            let value = reader.read_u8(2)?;
            if value > 1 {
                return Err(AscError::UnsupportedErrorProtectionConfig(value));
            }
            Some(value)
        } else {
            None
        };

        // MPEG-4 backward-compatible explicit SBR/PS signaling follows the
        // core GASpecificConfig as a sync extension (0x2b7 / 0x548).
        if extension.is_none() && reader.remaining_bits() >= 16 {
            let mut probe = reader.clone();
            if probe.read_u16(11)? == 0x2b7 && read_audio_object_type(&mut probe)? == AOT_SBR {
                let sbr_present = probe.read_bool()?;
                if sbr_present {
                    let (extension_sampling_frequency_index, extension_sampling_frequency) =
                        read_sampling_frequency(&mut probe)?;
                    let mut ps_present = false;
                    if probe.remaining_bits() >= 12 {
                        let mut ps_probe = probe.clone();
                        if ps_probe.read_u16(11)? == 0x548 {
                            ps_present = ps_probe.read_bool()?;
                            probe = ps_probe;
                        }
                    }
                    extension = Some(AudioSpecificConfigExtension {
                        audio_object_type: AOT_SBR,
                        sampling_frequency_index: extension_sampling_frequency_index,
                        sampling_frequency: extension_sampling_frequency,
                        ps_present,
                    });
                    *reader = probe;
                }
            }
        }

        Ok(Self {
            audio_object_type: core_audio_object_type,
            sampling_frequency_index: core_sampling_frequency_index,
            sampling_frequency: core_sampling_frequency,
            channel_configuration,
            extension,
            ga_specific,
            eld_specific,
            usac_config,
            error_protection_config,
            program_config,
            bits_read: reader.bits_read() - start,
        })
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, AscError> {
        let mut writer = BitWriter::new();

        if let Some(extension) = self.extension {
            write_audio_object_type(&mut writer, extension.audio_object_type)?;
            write_sampling_frequency(
                &mut writer,
                self.sampling_frequency_index,
                self.sampling_frequency,
            )?;
        } else {
            write_audio_object_type(&mut writer, self.audio_object_type)?;
            write_sampling_frequency(
                &mut writer,
                self.sampling_frequency_index,
                self.sampling_frequency,
            )?;
        }

        if !matches!(self.channel_configuration, 0..=7 | 11 | 12 | 14) {
            return Err(AscError::InvalidChannelConfiguration(
                self.channel_configuration,
            ));
        }
        writer.write(self.channel_configuration as u32, 4);

        if self.extension.is_some() {
            write_sampling_frequency(
                &mut writer,
                self.extension.unwrap().sampling_frequency_index,
                self.extension.unwrap().sampling_frequency,
            )?;
            write_audio_object_type(&mut writer, self.audio_object_type)?;
        }

        if let Some(ga) = self.ga_specific {
            ga.write_base(&mut writer)?;
            if self.channel_configuration == 0 {
                self.program_config
                    .as_ref()
                    .ok_or(AscError::MissingProgramConfigElement)?
                    .write_to_writer(&mut writer)?;
            }
            ga.write_tail(&mut writer, self.audio_object_type)?;
        }
        if self.audio_object_type == 39 {
            self.eld_specific
                .as_ref()
                .ok_or(AscError::MissingEldSpecificConfig)?
                .write(&mut writer, self.channel_configuration)?;
        }
        if self.audio_object_type == AOT_USAC {
            self.usac_config
                .as_ref()
                .ok_or(AscError::MissingUsacConfig)?
                .write(&mut writer)?;
        }
        if is_error_resilient(self.audio_object_type) {
            let value = self
                .error_protection_config
                .ok_or(AscError::MissingErrorProtectionConfig)?;
            if value > 1 {
                return Err(AscError::UnsupportedErrorProtectionConfig(value));
            }
            writer.write(value as u32, 2);
        }

        Ok(writer.finish())
    }

    /// Serialize HE-AAC/PS using implicit (0), backward-compatible explicit
    /// sync-extension (1), or explicit hierarchical (2) signaling.
    pub(crate) fn to_bytes_with_sbr_signaling(
        &self,
        signaling_mode: u8,
    ) -> Result<(Vec<u8>, usize), AscError> {
        if self.extension.is_none() || signaling_mode == 2 {
            let bytes = self.to_bytes()?;
            let bits = Self::parse(&bytes)?.bits_read;
            return Ok((bytes, bits));
        }
        let extension = self.extension.expect("checked above");
        let mut core = self.clone();
        core.extension = None;
        let core_bytes = core.to_bytes()?;
        let core_bits = Self::parse(&core_bytes)?.bits_read;
        if signaling_mode == 0 {
            return Ok((core_bytes, core_bits));
        }
        if signaling_mode != 1 {
            return Err(AscError::InvalidSbrSignalingMode(signaling_mode));
        }
        let mut writer = BitWriter::new();
        for bit in 0..core_bits {
            writer.write(u32::from((core_bytes[bit / 8] >> (7 - bit % 8)) & 1), 1);
        }
        writer.write(0x2b7, 11);
        write_audio_object_type(&mut writer, AOT_SBR)?;
        writer.write_bool(true);
        write_sampling_frequency(
            &mut writer,
            extension.sampling_frequency_index,
            extension.sampling_frequency,
        )?;
        if extension.ps_present {
            writer.write(0x548, 11);
            writer.write_bool(true);
        }
        let bits = writer.bits_written();
        Ok((writer.finish(), bits))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UsacConfig {
    pub sampling_frequency_index: u8,
    pub sampling_frequency: u32,
    pub core_sbr_frame_length_index: u8,
    pub core_frame_length: u16,
    pub output_frame_length: u16,
    pub sbr_ratio_index: u8,
    pub channel_configuration_index: u8,
    pub elements: Vec<UsacElementConfig>,
    pub extensions: Vec<UsacConfigExtension>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UsacElementConfig {
    SingleChannel {
        noise_filling: bool,
        sbr: Option<UsacSbrConfig>,
    },
    ChannelPair {
        noise_filling: bool,
        sbr: Option<UsacSbrConfig>,
        stereo_config_index: u8,
        mps212: Option<Mps212Config>,
    },
    Lfe {
        sbr: Option<UsacSbrConfig>,
    },
    Extension(UsacExtElementConfig),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mps212Config {
    pub frequency_resolution_index: u8,
    pub frequency_resolution_bands: u8,
    pub fixed_gain_downmix: u8,
    pub temporal_shape_config: u8,
    pub decorrelation_config: u8,
    pub high_rate_mode: bool,
    pub phase_coding: bool,
    pub ott_bands_phase: Option<u8>,
    pub residual_bands: Option<u8>,
    pub pseudo_lr: bool,
    pub environment_quantization_mode: Option<bool>,
}

impl Mps212Config {
    fn parse(reader: &mut BitReader<'_>, stereo_config_index: u8) -> Result<Self, AscError> {
        const FREQUENCY_BANDS: [u8; 8] = [0, 28, 20, 14, 10, 7, 5, 4];
        let frequency_resolution_index = reader.read_u8(3)?;
        let frequency_resolution_bands = FREQUENCY_BANDS[usize::from(frequency_resolution_index)];
        if frequency_resolution_bands == 0 {
            return Err(AscError::InvalidUsacMpsConfig);
        }
        let fixed_gain_downmix = reader.read_u8(3)?;
        let temporal_shape_config = reader.read_u8(2)?;
        let decorrelation_config = reader.read_u8(2)?;
        if decorrelation_config > 2 {
            return Err(AscError::InvalidUsacMpsConfig);
        }
        let high_rate_mode = reader.read_bool()?;
        let phase_coding = reader.read_bool()?;
        let ott_bands_phase = reader.read_bool()?.then(|| reader.read_u8(5)).transpose()?;
        if ott_bands_phase.is_some_and(|bands| bands > 28) {
            return Err(AscError::InvalidUsacMpsConfig);
        }
        let (residual_bands, pseudo_lr) = if stereo_config_index > 1 {
            let bands = reader.read_u8(5)?;
            if bands > frequency_resolution_bands {
                return Err(AscError::InvalidUsacMpsConfig);
            }
            (Some(bands), reader.read_bool()?)
        } else {
            (None, false)
        };
        let environment_quantization_mode = (temporal_shape_config == 2)
            .then(|| reader.read_bool())
            .transpose()?;
        Ok(Self {
            frequency_resolution_index,
            frequency_resolution_bands,
            fixed_gain_downmix,
            temporal_shape_config,
            decorrelation_config,
            high_rate_mode,
            phase_coding,
            ott_bands_phase,
            residual_bands,
            pseudo_lr,
            environment_quantization_mode,
        })
    }

    pub(crate) fn parse_drm(
        reader: &mut BitReader<'_>,
        stereo_config_index: u8,
    ) -> Result<Self, AscError> {
        const FREQUENCY_BANDS: [u8; 8] = [0, 28, 20, 14, 10, 7, 5, 4];
        let frequency_resolution_index = reader.read_u8(3)?;
        let frequency_resolution_bands = FREQUENCY_BANDS[frequency_resolution_index as usize];
        if frequency_resolution_bands == 0 {
            return Err(AscError::InvalidUsacMpsConfig);
        }
        let fixed_gain_downmix = reader.read_u8(3)?;
        let temporal_shape_config = if reader.read_bool()? { 3 } else { 0 };
        let high_rate_mode = reader.read_bool()?;
        let phase_coding = reader.read_bool()?;
        let ott_bands_phase = reader.read_bool()?.then(|| reader.read_u8(5)).transpose()?;
        if ott_bands_phase.is_some_and(|bands| bands > 28) {
            return Err(AscError::InvalidUsacMpsConfig);
        }
        let (residual_bands, pseudo_lr) = if stereo_config_index > 1 {
            let bands = reader.read_u8(5)?;
            if bands > frequency_resolution_bands {
                return Err(AscError::InvalidUsacMpsConfig);
            }
            (Some(bands), reader.read_bool()?)
        } else {
            (None, false)
        };
        Ok(Self {
            frequency_resolution_index,
            frequency_resolution_bands,
            fixed_gain_downmix,
            temporal_shape_config,
            decorrelation_config: 0,
            high_rate_mode,
            phase_coding,
            ott_bands_phase,
            residual_bands,
            pseudo_lr,
            environment_quantization_mode: None,
        })
    }

    fn write(&self, writer: &mut BitWriter, stereo_config_index: u8) {
        writer.write(self.frequency_resolution_index.into(), 3);
        writer.write(self.fixed_gain_downmix.into(), 3);
        writer.write(self.temporal_shape_config.into(), 2);
        writer.write(self.decorrelation_config.into(), 2);
        writer.write_bool(self.high_rate_mode);
        writer.write_bool(self.phase_coding);
        writer.write_bool(self.ott_bands_phase.is_some());
        if let Some(bands) = self.ott_bands_phase {
            writer.write(bands.into(), 5);
        }
        if stereo_config_index > 1 {
            writer.write(self.residual_bands.unwrap_or(0).into(), 5);
            writer.write_bool(self.pseudo_lr);
        }
        if self.temporal_shape_config == 2 {
            writer.write_bool(self.environment_quantization_mode.unwrap_or(false));
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UsacSbrConfig {
    pub harmonic_sbr: bool,
    pub inter_tes: bool,
    pub pvc: bool,
    pub start_frequency: u8,
    pub stop_frequency: u8,
    pub frequency_scale: Option<u8>,
    pub alter_scale: Option<bool>,
    pub noise_bands: Option<u8>,
    pub limiter_bands: Option<u8>,
    pub limiter_gains: Option<u8>,
    pub interpol_frequency: Option<bool>,
    pub smoothing_mode: Option<bool>,
}

impl UsacSbrConfig {
    pub(crate) fn parse(reader: &mut BitReader<'_>) -> Result<Self, AscError> {
        let harmonic_sbr = reader.read_bool()?;
        let inter_tes = reader.read_bool()?;
        let pvc = reader.read_bool()?;
        let start_frequency = reader.read_u8(4)?;
        let stop_frequency = reader.read_u8(4)?;
        let extra_1 = reader.read_bool()?;
        let extra_2 = reader.read_bool()?;
        let (frequency_scale, alter_scale, noise_bands) = if extra_1 {
            (
                Some(reader.read_u8(2)?),
                Some(reader.read_bool()?),
                Some(reader.read_u8(2)?),
            )
        } else {
            (None, None, None)
        };
        let (limiter_bands, limiter_gains, interpol_frequency, smoothing_mode) = if extra_2 {
            (
                Some(reader.read_u8(2)?),
                Some(reader.read_u8(2)?),
                Some(reader.read_bool()?),
                Some(reader.read_bool()?),
            )
        } else {
            (None, None, None, None)
        };
        Ok(Self {
            harmonic_sbr,
            inter_tes,
            pvc,
            start_frequency,
            stop_frequency,
            frequency_scale,
            alter_scale,
            noise_bands,
            limiter_bands,
            limiter_gains,
            interpol_frequency,
            smoothing_mode,
        })
    }

    fn write(&self, writer: &mut BitWriter) {
        writer.write_bool(self.harmonic_sbr);
        writer.write_bool(self.inter_tes);
        writer.write_bool(self.pvc);
        writer.write(self.start_frequency as u32, 4);
        writer.write(self.stop_frequency as u32, 4);
        writer.write_bool(self.frequency_scale.is_some());
        writer.write_bool(self.limiter_bands.is_some());
        if let Some(value) = self.frequency_scale {
            writer.write(value as u32, 2);
            writer.write_bool(self.alter_scale.unwrap_or(true));
            writer.write(self.noise_bands.unwrap_or(2) as u32, 2);
        }
        if let Some(value) = self.limiter_bands {
            writer.write(value as u32, 2);
            writer.write(self.limiter_gains.unwrap_or(2) as u32, 2);
            writer.write_bool(self.interpol_frequency.unwrap_or(true));
            writer.write_bool(self.smoothing_mode.unwrap_or(true));
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UsacExtElementConfig {
    pub extension_type: u32,
    pub default_length: Option<u32>,
    pub payload_fragmentation: bool,
    pub config: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UsacConfigExtension {
    pub extension_type: u32,
    pub data: Vec<u8>,
}

impl UsacConfig {
    pub(crate) fn parse_bytes(input: &[u8]) -> Result<Self, AscError> {
        Self::parse(&mut BitReader::new(input))
    }

    pub(crate) fn to_bytes(&self) -> Result<Vec<u8>, AscError> {
        let mut writer = BitWriter::new();
        self.write(&mut writer)?;
        Ok(writer.finish())
    }

    fn parse(reader: &mut BitReader<'_>) -> Result<Self, AscError> {
        let sampling_frequency_index = reader.read_u8(5)?;
        let sampling_frequency = if sampling_frequency_index == 31 {
            reader.read(24)?
        } else {
            sample_rate_from_index(sampling_frequency_index).ok_or(
                AscError::InvalidSamplingFrequencyIndex(sampling_frequency_index),
            )?
        };
        if sampling_frequency == 0 || sampling_frequency > 96_000 {
            return Err(AscError::UnsupportedSampleRate(sampling_frequency));
        }
        let core_sbr_frame_length_index = reader.read_u8(3)?;
        const FRAME_LENGTHS: [u16; 5] = [768, 1024, 2048, 2048, 4096];
        const SBR_RATIOS: [u8; 5] = [0, 0, 2, 3, 1];
        let index = core_sbr_frame_length_index as usize;
        let output_frame_length =
            *FRAME_LENGTHS
                .get(index)
                .ok_or(AscError::InvalidUsacFrameLengthIndex(
                    core_sbr_frame_length_index,
                ))?;
        let sbr_ratio_index = SBR_RATIOS[index];
        let core_frame_length = match sbr_ratio_index {
            1 => output_frame_length / 4,
            2 => output_frame_length * 3 / 8,
            3 => output_frame_length / 2,
            _ => output_frame_length,
        };
        let channel_configuration_index = reader.read_u8(5)?;
        let (expected_channels, expected_sce, expected_cpe, expected_lfe) =
            usac_channel_layout(channel_configuration_index).ok_or(
                AscError::InvalidUsacChannelConfiguration(channel_configuration_index),
            )?;
        let element_count = read_escaped_value(reader, 4, 8, 16)? as usize + 1;
        let mut elements = Vec::with_capacity(element_count);
        for _ in 0..element_count {
            let element_type = reader.read_u8(2)?;
            let element = match element_type {
                0 | 1 => {
                    if reader.read_bool()? {
                        return Err(AscError::UnsupportedUsacTwMdct);
                    }
                    let noise_filling = reader.read_bool()?;
                    let sbr = (sbr_ratio_index != 0)
                        .then(|| UsacSbrConfig::parse(reader))
                        .transpose()?;
                    if element_type == 0 {
                        UsacElementConfig::SingleChannel { noise_filling, sbr }
                    } else {
                        let stereo_config_index =
                            if sbr.is_some() { reader.read_u8(2)? } else { 0 };
                        let mps212 = (stereo_config_index != 0)
                            .then(|| Mps212Config::parse(reader, stereo_config_index))
                            .transpose()?;
                        UsacElementConfig::ChannelPair {
                            noise_filling,
                            sbr,
                            stereo_config_index,
                            mps212,
                        }
                    }
                }
                2 => UsacElementConfig::Lfe {
                    sbr: (sbr_ratio_index != 0)
                        .then(|| UsacSbrConfig::parse(reader))
                        .transpose()?,
                },
                _ => {
                    let extension_type = read_escaped_value(reader, 4, 8, 16)?;
                    let length = read_escaped_value(reader, 4, 8, 16)? as usize;
                    let default_length = if reader.read_bool()? {
                        Some(read_escaped_value(reader, 8, 16, 0)? + 1)
                    } else {
                        None
                    };
                    let payload_fragmentation = reader.read_bool()?;
                    let config = (0..length)
                        .map(|_| reader.read_u8(8).map_err(AscError::from))
                        .collect::<Result<Vec<_>, _>>()?;
                    UsacElementConfig::Extension(UsacExtElementConfig {
                        extension_type,
                        default_length,
                        payload_fragmentation,
                        config,
                    })
                }
            };
            elements.push(element);
        }
        let (actual_channels, actual_sce, actual_cpe, actual_lfe) = usac_element_layout(&elements);
        if actual_channels != expected_channels {
            return Err(AscError::UsacChannelCountMismatch {
                expected: expected_channels,
                actual: actual_channels,
            });
        }
        if (actual_sce, actual_cpe, actual_lfe) != (expected_sce, expected_cpe, expected_lfe) {
            return Err(AscError::UsacElementLayoutMismatch {
                expected_sce,
                expected_cpe,
                expected_lfe,
                actual_sce,
                actual_cpe,
                actual_lfe,
            });
        }
        let extensions = if reader.read_bool()? {
            let count = read_escaped_value(reader, 2, 4, 8)? as usize + 1;
            (0..count)
                .map(|_| {
                    let extension_type = read_escaped_value(reader, 4, 8, 16)?;
                    let length = read_escaped_value(reader, 4, 8, 16)? as usize;
                    let data = (0..length)
                        .map(|_| reader.read_u8(8).map_err(AscError::from))
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(UsacConfigExtension {
                        extension_type,
                        data,
                    })
                })
                .collect::<Result<Vec<_>, AscError>>()?
        } else {
            Vec::new()
        };
        Ok(Self {
            sampling_frequency_index,
            sampling_frequency,
            core_sbr_frame_length_index,
            core_frame_length,
            output_frame_length,
            sbr_ratio_index,
            channel_configuration_index,
            elements,
            extensions,
        })
    }

    fn write(&self, writer: &mut BitWriter) -> Result<(), AscError> {
        let (expected_channels, expected_sce, expected_cpe, expected_lfe) =
            usac_channel_layout(self.channel_configuration_index).ok_or(
                AscError::InvalidUsacChannelConfiguration(self.channel_configuration_index),
            )?;
        let (actual_channels, actual_sce, actual_cpe, actual_lfe) =
            usac_element_layout(&self.elements);
        if actual_channels != expected_channels {
            return Err(AscError::UsacChannelCountMismatch {
                expected: expected_channels,
                actual: actual_channels,
            });
        }
        if (actual_sce, actual_cpe, actual_lfe) != (expected_sce, expected_cpe, expected_lfe) {
            return Err(AscError::UsacElementLayoutMismatch {
                expected_sce,
                expected_cpe,
                expected_lfe,
                actual_sce,
                actual_cpe,
                actual_lfe,
            });
        }
        writer.write(self.sampling_frequency_index as u32, 5);
        if self.sampling_frequency_index == 31 {
            writer.write(self.sampling_frequency, 24);
        }
        writer.write(self.core_sbr_frame_length_index as u32, 3);
        writer.write(self.channel_configuration_index as u32, 5);
        write_escaped_value(writer, self.elements.len() as u32 - 1, 4, 8, 16)?;
        for element in &self.elements {
            match element {
                UsacElementConfig::SingleChannel { noise_filling, sbr } => {
                    writer.write(0, 2);
                    writer.write_bool(false);
                    writer.write_bool(*noise_filling);
                    if let Some(sbr) = sbr {
                        sbr.write(writer);
                    }
                }
                UsacElementConfig::ChannelPair {
                    noise_filling,
                    sbr,
                    stereo_config_index,
                    mps212,
                } => {
                    writer.write(1, 2);
                    writer.write_bool(false);
                    writer.write_bool(*noise_filling);
                    if let Some(sbr) = sbr {
                        sbr.write(writer);
                        writer.write(*stereo_config_index as u32, 2);
                        if let Some(config) = mps212 {
                            config.write(writer, *stereo_config_index);
                        }
                    }
                }
                UsacElementConfig::Lfe { sbr } => {
                    writer.write(2, 2);
                    if let Some(sbr) = sbr {
                        sbr.write(writer);
                    }
                }
                UsacElementConfig::Extension(extension) => {
                    writer.write(3, 2);
                    write_escaped_value(writer, extension.extension_type, 4, 8, 16)?;
                    write_escaped_value(writer, extension.config.len() as u32, 4, 8, 16)?;
                    writer.write_bool(extension.default_length.is_some());
                    if let Some(length) = extension.default_length {
                        write_escaped_value(writer, length - 1, 8, 16, 0)?;
                    }
                    writer.write_bool(extension.payload_fragmentation);
                    for &byte in &extension.config {
                        writer.write(byte as u32, 8);
                    }
                }
            }
        }
        writer.write_bool(!self.extensions.is_empty());
        if !self.extensions.is_empty() {
            write_escaped_value(writer, self.extensions.len() as u32 - 1, 2, 4, 8)?;
            for extension in &self.extensions {
                write_escaped_value(writer, extension.extension_type, 4, 8, 16)?;
                write_escaped_value(writer, extension.data.len() as u32, 4, 8, 16)?;
                for &byte in &extension.data {
                    writer.write(byte as u32, 8);
                }
            }
        }
        Ok(())
    }
}

pub(crate) fn usac_channel_count(index: u8) -> Option<usize> {
    usac_channel_layout(index).map(|layout| layout.0)
}

pub(crate) fn usac_element_layout_matches(index: u8, elements: &[UsacElementConfig]) -> bool {
    usac_channel_layout(index).is_some_and(|expected| usac_element_layout(elements) == expected)
}

fn usac_channel_layout(index: u8) -> Option<(usize, usize, usize, usize)> {
    const LAYOUTS: [(usize, usize, usize, usize); 14] = [
        (0, 0, 0, 0),
        (1, 1, 0, 0),
        (2, 0, 1, 0),
        (3, 1, 1, 0),
        (4, 2, 1, 0),
        (5, 1, 2, 0),
        (6, 1, 2, 1),
        (8, 1, 3, 1),
        (2, 2, 0, 0),
        (3, 1, 1, 0),
        (4, 0, 2, 0),
        (7, 2, 2, 1),
        (8, 1, 3, 1),
        (24, 6, 8, 2),
    ];
    (index != 0)
        .then(|| LAYOUTS.get(usize::from(index)).copied())
        .flatten()
}

fn usac_element_layout(elements: &[UsacElementConfig]) -> (usize, usize, usize, usize) {
    let mut sce = 0usize;
    let mut cpe = 0usize;
    let mut lfe = 0usize;
    for element in elements {
        match element {
            UsacElementConfig::SingleChannel { .. } => sce += 1,
            UsacElementConfig::ChannelPair { .. } => cpe += 1,
            UsacElementConfig::Lfe { .. } => lfe += 1,
            UsacElementConfig::Extension(_) => {}
        }
    }
    (sce + 2 * cpe + lfe, sce, cpe, lfe)
}

fn read_escaped_value(
    reader: &mut BitReader<'_>,
    first: usize,
    second: usize,
    third: usize,
) -> Result<u32, AscError> {
    let first_max = (1u32 << first) - 1;
    let mut value = reader.read(first)?;
    if value == first_max && second != 0 {
        let second_max = (1u32 << second) - 1;
        let next = reader.read(second)?;
        value += next;
        if next == second_max && third != 0 {
            value += reader.read(third)?;
        }
    }
    Ok(value)
}

fn write_escaped_value(
    writer: &mut BitWriter,
    mut value: u32,
    first: usize,
    second: usize,
    third: usize,
) -> Result<(), AscError> {
    let first_max = (1u32 << first) - 1;
    let first_value = value.min(first_max);
    writer.write(first_value, first);
    value -= first_value;
    if first_value == first_max && second != 0 {
        let second_max = (1u32 << second) - 1;
        let second_value = value.min(second_max);
        writer.write(second_value, second);
        value -= second_value;
        if second_value == second_max && third != 0 {
            if value >= 1u32 << third {
                return Err(AscError::EscapedValueTooLarge);
            }
            writer.write(value, third);
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EldSpecificConfig {
    pub frame_length_flag: bool,
    pub section_data_resilience: bool,
    pub scalefactor_data_resilience: bool,
    pub spectral_data_resilience: bool,
    pub sbr_present: bool,
    pub sbr_sampling_rate: bool,
    pub sbr_crc: bool,
    pub sbr_headers: Vec<LdSbrHeader>,
    pub extensions: Vec<EldExtension>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LdSbrHeader {
    pub amp_resolution: bool,
    pub crossover_band: u8,
    pub reserved: u8,
    pub start_frequency: u8,
    pub stop_frequency: u8,
    pub frequency_scale: Option<u8>,
    pub alter_scale: Option<bool>,
    pub noise_bands: Option<u8>,
    pub limiter_bands: Option<u8>,
    pub limiter_gains: Option<u8>,
    pub interpol_frequency: Option<bool>,
    pub smoothing_mode: Option<bool>,
}

impl LdSbrHeader {
    pub(crate) fn parse(reader: &mut BitReader<'_>) -> Result<Self, AscError> {
        let amp_resolution = reader.read_bool()?;
        let start_frequency = reader.read_u8(4)?;
        let stop_frequency = reader.read_u8(4)?;
        let crossover_band = reader.read_u8(3)?;
        let reserved = reader.read_u8(2)?;
        let extra_1 = reader.read_bool()?;
        let extra_2 = reader.read_bool()?;
        let (frequency_scale, alter_scale, noise_bands) = if extra_1 {
            (
                Some(reader.read_u8(2)?),
                Some(reader.read_bool()?),
                Some(reader.read_u8(2)?),
            )
        } else {
            (None, None, None)
        };
        let (limiter_bands, limiter_gains, interpol_frequency, smoothing_mode) = if extra_2 {
            (
                Some(reader.read_u8(2)?),
                Some(reader.read_u8(2)?),
                Some(reader.read_bool()?),
                Some(reader.read_bool()?),
            )
        } else {
            (None, None, None, None)
        };
        Ok(Self {
            amp_resolution,
            crossover_band,
            reserved,
            start_frequency,
            stop_frequency,
            frequency_scale,
            alter_scale,
            noise_bands,
            limiter_bands,
            limiter_gains,
            interpol_frequency,
            smoothing_mode,
        })
    }

    pub(crate) fn write(&self, writer: &mut BitWriter) -> Result<(), AscError> {
        if self.crossover_band > 7
            || self.reserved > 3
            || self.start_frequency > 15
            || self.stop_frequency > 15
            || self.frequency_scale.is_some_and(|value| value > 3)
            || self.noise_bands.is_some_and(|value| value > 3)
            || self.limiter_bands.is_some_and(|value| value > 3)
            || self.limiter_gains.is_some_and(|value| value > 3)
        {
            return Err(AscError::InvalidLdSbrHeader);
        }
        let extra_1 = self.frequency_scale.is_some()
            || self.alter_scale.is_some()
            || self.noise_bands.is_some();
        let extra_2 = self.limiter_bands.is_some()
            || self.limiter_gains.is_some()
            || self.interpol_frequency.is_some()
            || self.smoothing_mode.is_some();
        if extra_1
            && (self.frequency_scale.is_none()
                || self.alter_scale.is_none()
                || self.noise_bands.is_none())
            || extra_2
                && (self.limiter_bands.is_none()
                    || self.limiter_gains.is_none()
                    || self.interpol_frequency.is_none()
                    || self.smoothing_mode.is_none())
        {
            return Err(AscError::InvalidLdSbrHeader);
        }
        writer.write_bool(self.amp_resolution);
        writer.write(self.start_frequency as u32, 4);
        writer.write(self.stop_frequency as u32, 4);
        writer.write(self.crossover_band as u32, 3);
        writer.write(self.reserved as u32, 2);
        writer.write_bool(extra_1);
        writer.write_bool(extra_2);
        if extra_1 {
            writer.write(self.frequency_scale.unwrap() as u32, 2);
            writer.write_bool(self.alter_scale.unwrap());
            writer.write(self.noise_bands.unwrap() as u32, 2);
        }
        if extra_2 {
            writer.write(self.limiter_bands.unwrap() as u32, 2);
            writer.write(self.limiter_gains.unwrap() as u32, 2);
            writer.write_bool(self.interpol_frequency.unwrap());
            writer.write_bool(self.smoothing_mode.unwrap());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EldExtension {
    pub extension_type: u8,
    pub data: Vec<u8>,
}

impl EldSpecificConfig {
    fn parse(reader: &mut BitReader<'_>, channel_configuration: u8) -> Result<Self, AscError> {
        let frame_length_flag = reader.read_bool()?;
        let section_data_resilience = reader.read_bool()?;
        let scalefactor_data_resilience = reader.read_bool()?;
        let spectral_data_resilience = reader.read_bool()?;
        let sbr_present = reader.read_bool()?;
        let (sbr_sampling_rate, sbr_crc) = if sbr_present {
            (reader.read_bool()?, reader.read_bool()?)
        } else {
            (false, false)
        };
        let mut sbr_headers = Vec::new();
        if sbr_present {
            let count = ld_sbr_header_count(channel_configuration).ok_or(
                AscError::UnsupportedLdSbrChannelConfiguration(channel_configuration),
            )?;
            for _ in 0..count {
                sbr_headers.push(LdSbrHeader::parse(reader)?);
            }
        }
        let mut extensions = Vec::new();
        loop {
            let extension_type = reader.read_u8(4)?;
            if extension_type == 0 {
                break;
            }
            let first = reader.read_u8(4)? as usize;
            let mut length = first;
            if first == 15 {
                let second = reader.read_u8(8)? as usize;
                length += second;
                if second == 255 {
                    length += reader.read_u16(16)? as usize;
                }
            }
            let mut data = Vec::with_capacity(length);
            for _ in 0..length {
                data.push(reader.read_u8(8)?);
            }
            extensions.push(EldExtension {
                extension_type,
                data,
            });
            if extensions.len() > 15 {
                return Err(AscError::TooManyEldExtensions);
            }
        }
        Ok(Self {
            frame_length_flag,
            section_data_resilience,
            scalefactor_data_resilience,
            spectral_data_resilience,
            sbr_present,
            sbr_sampling_rate,
            sbr_crc,
            sbr_headers,
            extensions,
        })
    }

    fn write(&self, writer: &mut BitWriter, channel_configuration: u8) -> Result<(), AscError> {
        writer.write_bool(self.frame_length_flag);
        writer.write_bool(self.section_data_resilience);
        writer.write_bool(self.scalefactor_data_resilience);
        writer.write_bool(self.spectral_data_resilience);
        writer.write_bool(self.sbr_present);
        if self.sbr_present {
            writer.write_bool(self.sbr_sampling_rate);
            writer.write_bool(self.sbr_crc);
            let count = ld_sbr_header_count(channel_configuration).ok_or(
                AscError::UnsupportedLdSbrChannelConfiguration(channel_configuration),
            )?;
            if self.sbr_headers.len() != count {
                return Err(AscError::LdSbrHeaderCount {
                    expected: count,
                    actual: self.sbr_headers.len(),
                });
            }
            for header in &self.sbr_headers {
                header.write(writer)?;
            }
        } else if !self.sbr_headers.is_empty() {
            return Err(AscError::LdSbrHeaderCount {
                expected: 0,
                actual: self.sbr_headers.len(),
            });
        }
        if self.extensions.len() > 15 {
            return Err(AscError::TooManyEldExtensions);
        }
        for extension in &self.extensions {
            if extension.extension_type == 0 || extension.extension_type > 15 {
                return Err(AscError::InvalidEldExtensionType(extension.extension_type));
            }
            writer.write(extension.extension_type as u32, 4);
            let length = extension.data.len();
            if length < 15 {
                writer.write(length as u32, 4);
            } else {
                writer.write(15, 4);
                let remainder = length - 15;
                if remainder < 255 {
                    writer.write(remainder as u32, 8);
                } else {
                    let tail = remainder - 255;
                    if tail > u16::MAX as usize {
                        return Err(AscError::EldExtensionTooLong(length));
                    }
                    writer.write(255, 8);
                    writer.write(tail as u32, 16);
                }
            }
            for &byte in &extension.data {
                writer.write(byte as u32, 8);
            }
        }
        writer.write(0, 4);
        Ok(())
    }
}

fn ld_sbr_header_count(channel_configuration: u8) -> Option<usize> {
    Some(match channel_configuration {
        1 | 2 => 1,
        3 => 2,
        4..=6 => 3,
        7 | 11 | 12 | 14 => 4,
        _ => return None,
    })
}

/// Extension information for implicit first-field SBR/PS signalling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioSpecificConfigExtension {
    pub audio_object_type: u8,
    pub sampling_frequency_index: u8,
    pub sampling_frequency: u32,
    pub ps_present: bool,
}

/// Generic Audio specific config fields shared by common AAC profiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GaSpecificConfig {
    pub frame_length_flag: bool,
    pub depends_on_core_coder: bool,
    pub core_coder_delay: Option<u16>,
    pub extension_flag: bool,
    pub layer: Option<u8>,
    pub num_of_subframes: Option<u8>,
    pub layer_length: Option<u16>,
    pub section_data_resilience: bool,
    pub scalefactor_data_resilience: bool,
    pub spectral_data_resilience: bool,
    pub extension_flag3: Option<bool>,
}

impl GaSpecificConfig {
    fn parse(reader: &mut BitReader<'_>) -> Result<Self, AscError> {
        let frame_length_flag = reader.read_bool()?;
        let depends_on_core_coder = reader.read_bool()?;
        let core_coder_delay = if depends_on_core_coder {
            Some(reader.read_u16(14)?)
        } else {
            None
        };
        let extension_flag = reader.read_bool()?;

        Ok(Self {
            frame_length_flag,
            depends_on_core_coder,
            core_coder_delay,
            extension_flag,
            ..Self::default()
        })
    }

    fn write_base(self, writer: &mut BitWriter) -> Result<(), AscError> {
        writer.write_bool(self.frame_length_flag);
        writer.write_bool(self.depends_on_core_coder);
        if let Some(delay) = self.core_coder_delay {
            if delay > 0x3fff {
                return Err(AscError::InvalidCoreCoderDelay(delay));
            }
            writer.write(delay as u32, 14);
        } else if self.depends_on_core_coder {
            return Err(AscError::MissingCoreCoderDelay);
        }
        writer.write_bool(self.extension_flag);
        Ok(())
    }

    fn parse_tail(
        &mut self,
        reader: &mut BitReader<'_>,
        audio_object_type: u8,
    ) -> Result<(), AscError> {
        if matches!(audio_object_type, 6 | 20) {
            self.layer = Some(reader.read_u8(3)?);
        }
        if self.extension_flag {
            if audio_object_type == 22 {
                self.num_of_subframes = Some(reader.read_u8(5)?);
                self.layer_length = Some(reader.read_u16(11)?);
            }
            if matches!(audio_object_type, 17 | 19 | 20 | 23) {
                self.section_data_resilience = reader.read_bool()?;
                self.scalefactor_data_resilience = reader.read_bool()?;
                self.spectral_data_resilience = reader.read_bool()?;
            }
            self.extension_flag3 = Some(reader.read_bool()?);
        }
        Ok(())
    }

    fn write_tail(self, writer: &mut BitWriter, audio_object_type: u8) -> Result<(), AscError> {
        if matches!(audio_object_type, 6 | 20) {
            writer.write(self.layer.ok_or(AscError::MissingScalableLayer)? as u32, 3);
        }
        if self.extension_flag {
            if audio_object_type == 22 {
                writer.write(
                    self.num_of_subframes
                        .ok_or(AscError::MissingBsacExtension)? as u32,
                    5,
                );
                writer.write(
                    self.layer_length.ok_or(AscError::MissingBsacExtension)? as u32,
                    11,
                );
            }
            if matches!(audio_object_type, 17 | 19 | 20 | 23) {
                writer.write_bool(self.section_data_resilience);
                writer.write_bool(self.scalefactor_data_resilience);
                writer.write_bool(self.spectral_data_resilience);
            }
            writer.write_bool(self.extension_flag3.unwrap_or(false));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ProgramConfig {
    pub element_instance_tag: u8,
    pub profile: u8,
    pub sampling_frequency_index: u8,
    pub front: Vec<ProgramElement>,
    pub side: Vec<ProgramElement>,
    pub back: Vec<ProgramElement>,
    pub lfe: Vec<u8>,
    pub associated_data: Vec<u8>,
    pub valid_cc: Vec<ProgramCcElement>,
    pub mono_mixdown_element_number: Option<u8>,
    pub stereo_mixdown_element_number: Option<u8>,
    pub matrix_mixdown: Option<MatrixMixdown>,
    pub comment: Vec<u8>,
    pub num_channels: u8,
    pub num_effective_channels: u8,
}

impl ProgramConfig {
    pub fn parse_from_bytes(input: &[u8]) -> Result<Self, AscError> {
        let mut reader = BitReader::new(input);
        Self::parse_from_reader(&mut reader)
    }

    pub(crate) fn parse_from_reader(reader: &mut BitReader<'_>) -> Result<Self, AscError> {
        let element_instance_tag = reader.read_u8(4)?;
        let profile = reader.read_u8(2)?;
        let sampling_frequency_index = reader.read_u8(4)?;
        if sampling_frequency_index >= 13 && sampling_frequency_index != 0x0f {
            return Err(AscError::InvalidSamplingFrequencyIndex(
                sampling_frequency_index,
            ));
        }

        let num_front = reader.read_u8(4)?;
        let num_side = reader.read_u8(4)?;
        let num_back = reader.read_u8(4)?;
        let num_lfe = reader.read_u8(2)?;
        let num_assoc = reader.read_u8(3)?;
        let num_valid_cc = reader.read_u8(4)?;

        let mono_mixdown_element_number = if reader.read_bool()? {
            Some(reader.read_u8(4)?)
        } else {
            None
        };
        let stereo_mixdown_element_number = if reader.read_bool()? {
            Some(reader.read_u8(4)?)
        } else {
            None
        };
        let matrix_mixdown = if reader.read_bool()? {
            Some(MatrixMixdown {
                index: reader.read_u8(2)?,
                pseudo_surround_enable: reader.read_bool()?,
            })
        } else {
            None
        };

        let front = read_program_elements(reader, num_front)?;
        let side = read_program_elements(reader, num_side)?;
        let back = read_program_elements(reader, num_back)?;
        let num_effective_channels = count_channels(&front, &side, &back)?;

        let mut lfe = Vec::with_capacity(num_lfe as usize);
        for _ in 0..num_lfe {
            lfe.push(reader.read_u8(4)?);
        }

        let mut associated_data = Vec::with_capacity(num_assoc as usize);
        for _ in 0..num_assoc {
            associated_data.push(reader.read_u8(4)?);
        }

        let mut valid_cc = Vec::with_capacity(num_valid_cc as usize);
        for _ in 0..num_valid_cc {
            valid_cc.push(ProgramCcElement {
                is_ind_sw: reader.read_bool()?,
                tag_select: reader.read_u8(4)?,
            });
        }

        reader.byte_align();
        let comment_len = reader.read_u8(8)? as usize;
        let mut comment = Vec::with_capacity(comment_len);
        for _ in 0..comment_len {
            comment.push(reader.read_u8(8)?);
        }

        let num_channels = num_effective_channels
            .checked_add(num_lfe)
            .ok_or(AscError::ProgramConfigTooLarge)?;

        Ok(Self {
            element_instance_tag,
            profile,
            sampling_frequency_index,
            front,
            side,
            back,
            lfe,
            associated_data,
            valid_cc,
            mono_mixdown_element_number,
            stereo_mixdown_element_number,
            matrix_mixdown,
            comment,
            num_channels,
            num_effective_channels,
        })
    }

    pub(crate) fn write_to_writer(&self, writer: &mut BitWriter) -> Result<(), AscError> {
        if self.front.len() > 15 || self.side.len() > 15 || self.back.len() > 15 {
            return Err(AscError::ProgramConfigTooLarge);
        }
        if self.lfe.len() > 3 || self.associated_data.len() > 7 || self.valid_cc.len() > 15 {
            return Err(AscError::ProgramConfigTooLarge);
        }
        if self.comment.len() > u8::MAX as usize {
            return Err(AscError::ProgramConfigTooLarge);
        }

        writer.write(self.element_instance_tag as u32, 4);
        writer.write(self.profile as u32, 2);
        writer.write(self.sampling_frequency_index as u32, 4);
        writer.write(self.front.len() as u32, 4);
        writer.write(self.side.len() as u32, 4);
        writer.write(self.back.len() as u32, 4);
        writer.write(self.lfe.len() as u32, 2);
        writer.write(self.associated_data.len() as u32, 3);
        writer.write(self.valid_cc.len() as u32, 4);

        writer.write_bool(self.mono_mixdown_element_number.is_some());
        if let Some(value) = self.mono_mixdown_element_number {
            writer.write(value as u32, 4);
        }
        writer.write_bool(self.stereo_mixdown_element_number.is_some());
        if let Some(value) = self.stereo_mixdown_element_number {
            writer.write(value as u32, 4);
        }
        writer.write_bool(self.matrix_mixdown.is_some());
        if let Some(value) = self.matrix_mixdown {
            writer.write(value.index as u32, 2);
            writer.write_bool(value.pseudo_surround_enable);
        }

        for element in self.front.iter().chain(&self.side).chain(&self.back) {
            writer.write_bool(element.is_cpe);
            writer.write(element.tag_select as u32, 4);
        }
        for &tag in &self.lfe {
            writer.write(tag as u32, 4);
        }
        for &tag in &self.associated_data {
            writer.write(tag as u32, 4);
        }
        for cc in &self.valid_cc {
            writer.write_bool(cc.is_ind_sw);
            writer.write(cc.tag_select as u32, 4);
        }

        writer.byte_align();
        writer.write(self.comment.len() as u32, 8);
        for &byte in &self.comment {
            writer.write(byte as u32, 8);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProgramElement {
    pub is_cpe: bool,
    pub tag_select: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProgramCcElement {
    pub is_ind_sw: bool,
    pub tag_select: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixMixdown {
    pub index: u8,
    pub pseudo_surround_enable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AscError {
    UnexpectedEof {
        needed_bits: usize,
        remaining_bits: usize,
    },
    InvalidAudioObjectType(u8),
    InvalidSamplingFrequencyIndex(u8),
    InvalidChannelConfiguration(u8),
    InvalidSbrSignalingMode(u8),
    UnsupportedSampleRate(u32),
    MissingProgramConfigElement,
    MissingErrorProtectionConfig,
    MissingScalableLayer,
    MissingBsacExtension,
    UnsupportedErrorProtectionConfig(u8),
    ProgramConfigTooLarge,
    InvalidCoreCoderDelay(u16),
    MissingCoreCoderDelay,
    MissingEldSpecificConfig,
    UnsupportedEldSbr,
    UnsupportedLdSbrChannelConfiguration(u8),
    LdSbrHeaderCount {
        expected: usize,
        actual: usize,
    },
    InvalidLdSbrHeader,
    TooManyEldExtensions,
    InvalidEldExtensionType(u8),
    EldExtensionTooLong(usize),
    MissingUsacConfig,
    InvalidUsacFrameLengthIndex(u8),
    InvalidUsacChannelConfiguration(u8),
    UnsupportedUsacTwMdct,
    UnsupportedUsacSbrConfig,
    UnsupportedUsacMpsConfig(u8),
    InvalidUsacMpsConfig,
    UsacChannelCountMismatch {
        expected: usize,
        actual: usize,
    },
    UsacElementLayoutMismatch {
        expected_sce: usize,
        expected_cpe: usize,
        expected_lfe: usize,
        actual_sce: usize,
        actual_cpe: usize,
        actual_lfe: usize,
    },
    EscapedValueTooLarge,
}

impl From<BitError> for AscError {
    fn from(value: BitError) -> Self {
        match value {
            BitError::UnexpectedEof {
                needed_bits,
                remaining_bits,
            } => Self::UnexpectedEof {
                needed_bits,
                remaining_bits,
            },
            BitError::TooManyBitsRequested { .. } => Self::UnexpectedEof {
                needed_bits: usize::MAX,
                remaining_bits: 0,
            },
            BitError::InvalidPushBack { .. } => Self::UnexpectedEof {
                needed_bits: 0,
                remaining_bits: 0,
            },
        }
    }
}

impl fmt::Display for AscError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::UnexpectedEof {
                needed_bits,
                remaining_bits,
            } => write!(
                f,
                "ASC input too short: need {needed_bits} bits, only {remaining_bits} bits remain"
            ),
            Self::InvalidAudioObjectType(value) => {
                write!(f, "invalid ASC audio object type {value}")
            }
            Self::InvalidSamplingFrequencyIndex(value) => {
                write!(f, "invalid ASC sampling frequency index {value}")
            }
            Self::InvalidChannelConfiguration(value) => {
                write!(f, "invalid ASC channel configuration {value}")
            }
            Self::InvalidSbrSignalingMode(value) => {
                write!(f, "invalid ASC SBR signaling mode {value}")
            }
            Self::UnsupportedSampleRate(value) => write!(f, "unsupported ASC sample rate {value}"),
            Self::MissingProgramConfigElement => {
                write!(f, "ASC channel config 0 requires a program_config_element")
            }
            Self::MissingErrorProtectionConfig => {
                write!(f, "ER ASC requires epConfig")
            }
            Self::MissingScalableLayer => write!(f, "scalable ASC requires layer"),
            Self::MissingBsacExtension => write!(f, "BSAC ASC extension fields are missing"),
            Self::UnsupportedErrorProtectionConfig(value) => {
                write!(f, "unsupported ER ASC epConfig {value}")
            }
            Self::ProgramConfigTooLarge => {
                write!(f, "ASC program_config_element exceeds supported limits")
            }
            Self::InvalidCoreCoderDelay(value) => write!(f, "invalid ASC core coder delay {value}"),
            Self::MissingCoreCoderDelay => write!(
                f,
                "ASC core coder delay is required when depends_on_core_coder is set"
            ),
            Self::MissingEldSpecificConfig => {
                write!(f, "ER AAC-ELD ASC requires ELDSpecificConfig")
            }
            Self::UnsupportedEldSbr => write!(f, "ER AAC-ELD SBR configuration is unsupported"),
            Self::UnsupportedLdSbrChannelConfiguration(value) => write!(
                f,
                "unsupported ER AAC-ELD LD-SBR channel configuration {value}"
            ),
            Self::LdSbrHeaderCount { expected, actual } => write!(
                f,
                "ER AAC-ELD LD-SBR requires {expected} headers, got {actual}"
            ),
            Self::InvalidLdSbrHeader => write!(f, "invalid ER AAC-ELD LD-SBR header"),
            Self::TooManyEldExtensions => write!(f, "too many ER AAC-ELD extensions"),
            Self::InvalidEldExtensionType(value) => {
                write!(f, "invalid ER AAC-ELD extension type {value}")
            }
            Self::EldExtensionTooLong(length) => {
                write!(f, "ER AAC-ELD extension is too long: {length} bytes")
            }
            Self::MissingUsacConfig => write!(f, "USAC ASC requires UsacConfig"),
            Self::InvalidUsacFrameLengthIndex(value) => {
                write!(f, "invalid USAC coreSbrFrameLengthIndex {value}")
            }
            Self::InvalidUsacChannelConfiguration(value) => {
                write!(f, "unsupported USAC channelConfigurationIndex {value}")
            }
            Self::UnsupportedUsacTwMdct => write!(f, "USAC tw_mdct is unsupported"),
            Self::UnsupportedUsacSbrConfig => {
                write!(f, "USAC SBR configuration parsing is not implemented")
            }
            Self::UnsupportedUsacMpsConfig(value) => {
                write!(
                    f,
                    "USAC Mps212Config stereoConfigIndex {value} is unsupported"
                )
            }
            Self::InvalidUsacMpsConfig => write!(f, "invalid USAC Mps212Config"),
            Self::UsacChannelCountMismatch { expected, actual } => write!(
                f,
                "USAC channel configuration requires {expected} channels, got {actual}"
            ),
            Self::UsacElementLayoutMismatch {
                expected_sce,
                expected_cpe,
                expected_lfe,
                actual_sce,
                actual_cpe,
                actual_lfe,
            } => write!(
                f,
                "USAC channel configuration requires SCE/CPE/LFE {expected_sce}/{expected_cpe}/{expected_lfe}, got {actual_sce}/{actual_cpe}/{actual_lfe}"
            ),
            Self::EscapedValueTooLarge => write!(f, "escapedValue exceeds supported width"),
        }
    }
}

fn read_program_elements(
    reader: &mut BitReader<'_>,
    count: u8,
) -> Result<Vec<ProgramElement>, AscError> {
    let mut elements = Vec::with_capacity(count as usize);
    for _ in 0..count {
        elements.push(ProgramElement {
            is_cpe: reader.read_bool()?,
            tag_select: reader.read_u8(4)?,
        });
    }
    Ok(elements)
}

fn count_channels(
    front: &[ProgramElement],
    side: &[ProgramElement],
    back: &[ProgramElement],
) -> Result<u8, AscError> {
    let mut count: u8 = 0;
    for element in front.iter().chain(side).chain(back) {
        count = count
            .checked_add(if element.is_cpe { 2 } else { 1 })
            .ok_or(AscError::ProgramConfigTooLarge)?;
    }
    Ok(count)
}

impl std::error::Error for AscError {}

fn read_audio_object_type(reader: &mut BitReader<'_>) -> Result<u8, AscError> {
    let value = reader.read_u8(5)?;
    if value == AOT_ESCAPE {
        Ok(32 + reader.read_u8(6)?)
    } else if value == 0 {
        Err(AscError::InvalidAudioObjectType(value))
    } else {
        Ok(value)
    }
}

fn write_audio_object_type(writer: &mut BitWriter, value: u8) -> Result<(), AscError> {
    if value == 0 || value == AOT_ESCAPE || value > 95 {
        return Err(AscError::InvalidAudioObjectType(value));
    }
    if value < AOT_ESCAPE {
        writer.write(value as u32, 5);
    } else {
        writer.write(AOT_ESCAPE as u32, 5);
        writer.write((value - 32) as u32, 6);
    }
    Ok(())
}

fn read_sampling_frequency(reader: &mut BitReader<'_>) -> Result<(u8, u32), AscError> {
    let index = reader.read_u8(4)?;
    if index == 0x0f {
        Ok((index, reader.read(24)?))
    } else {
        let sample_rate =
            sample_rate_from_index(index).ok_or(AscError::InvalidSamplingFrequencyIndex(index))?;
        Ok((index, sample_rate))
    }
}

fn write_sampling_frequency(
    writer: &mut BitWriter,
    index: u8,
    sample_rate: u32,
) -> Result<(), AscError> {
    if index == 0x0f {
        writer.write(index as u32, 4);
        writer.write(sample_rate, 24);
        return Ok(());
    }

    let expected =
        sample_rate_from_index(index).ok_or(AscError::InvalidSamplingFrequencyIndex(index))?;
    if expected != sample_rate {
        return Err(AscError::UnsupportedSampleRate(sample_rate));
    }
    writer.write(index as u32, 4);
    Ok(())
}

fn is_ga_specific(audio_object_type: u8) -> bool {
    matches!(
        audio_object_type,
        1 | 2 | 3 | 4 | 6 | 7 | 17 | 19 | 20 | 21 | 22 | 23
    )
}

fn is_error_resilient(audio_object_type: u8) -> bool {
    matches!(audio_object_type, 17 | 19 | 20 | 21 | 22 | 23 | 39)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_aac_lc_stereo_44100() {
        let asc = AudioSpecificConfig::parse(&[0x12, 0x10]).unwrap();
        assert_eq!(asc.audio_object_type, 2);
        assert_eq!(asc.sampling_frequency_index, 4);
        assert_eq!(asc.sampling_frequency, 44_100);
        assert_eq!(asc.channel_configuration, 2);
        assert_eq!(asc.ga_specific, Some(GaSpecificConfig::default()));
        assert_eq!(asc.bits_read, 16);
    }

    #[test]
    fn writes_aac_lc_stereo_44100() {
        let asc = AudioSpecificConfig::aac_lc(44_100, 2).unwrap();
        assert_eq!(asc.to_bytes().unwrap(), vec![0x12, 0x10]);
    }

    #[test]
    fn parses_explicit_sample_rate() {
        // AOT=2, samplingFrequencyIndex=15, samplingFrequency=12345,
        // channelConfiguration=1, GASpecificConfig flags=000.
        let bytes = [0x17, 0x80, 0x18, 0x1c, 0x88];
        let asc = AudioSpecificConfig::parse(&bytes).unwrap();
        assert_eq!(asc.audio_object_type, 2);
        assert_eq!(asc.sampling_frequency_index, 15);
        assert_eq!(asc.sampling_frequency, 12_345);
        assert_eq!(asc.channel_configuration, 1);
    }

    #[test]
    fn parses_sbr_extension_prefix() {
        // AOT=SBR, extension sampling frequency index=4, core AOT=AAC-LC,
        // channel configuration=2, GASpecificConfig flags=000.
        let bytes = [0x2a, 0x12, 0x08, 0x00];
        let asc = AudioSpecificConfig::parse(&bytes).unwrap();
        assert_eq!(asc.audio_object_type, AOT_AAC_LC);
        assert_eq!(asc.sampling_frequency, 44_100);
        assert_eq!(asc.channel_configuration, 2);
        assert_eq!(asc.extension.unwrap().audio_object_type, AOT_SBR);
        assert_eq!(asc.extension.unwrap().sampling_frequency, 44_100);
    }

    #[test]
    fn parses_program_config_element() {
        let pce = ProgramConfig {
            element_instance_tag: 0,
            profile: 1,
            sampling_frequency_index: 4,
            front: vec![ProgramElement {
                is_cpe: false,
                tag_select: 0,
            }],
            comment: b"rust".to_vec(),
            num_channels: 1,
            num_effective_channels: 1,
            ..ProgramConfig::default()
        };
        let asc = AudioSpecificConfig {
            audio_object_type: AOT_AAC_LC,
            sampling_frequency_index: 4,
            sampling_frequency: 44_100,
            channel_configuration: 0,
            extension: None,
            ga_specific: Some(GaSpecificConfig::default()),
            eld_specific: None,
            usac_config: None,
            error_protection_config: None,
            program_config: Some(pce.clone()),
            bits_read: 0,
        };

        let bytes = asc.to_bytes().unwrap();
        let parsed = AudioSpecificConfig::parse(&bytes).unwrap();
        assert_eq!(parsed.channel_configuration, 0);
        assert_eq!(parsed.program_config.as_ref().unwrap().front, pce.front);
        assert_eq!(parsed.program_config.as_ref().unwrap().comment, b"rust");
        assert_eq!(parsed.program_config.as_ref().unwrap().num_channels, 1);
        assert_eq!(
            parsed
                .program_config
                .as_ref()
                .unwrap()
                .num_effective_channels,
            1
        );
    }

    #[test]
    fn roundtrips_er_aac_lc_ep_config_and_rejects_unsupported_values() {
        let asc = AudioSpecificConfig {
            audio_object_type: 17,
            sampling_frequency_index: 4,
            sampling_frequency: 44_100,
            channel_configuration: 2,
            extension: None,
            ga_specific: Some(GaSpecificConfig::default()),
            eld_specific: None,
            usac_config: None,
            error_protection_config: Some(1),
            program_config: None,
            bits_read: 0,
        };
        let bytes = asc.to_bytes().unwrap();
        let parsed = AudioSpecificConfig::parse(&bytes).unwrap();
        assert_eq!(parsed.audio_object_type, 17);
        assert_eq!(parsed.error_protection_config, Some(1));
        assert_eq!(parsed.bits_read, 18);

        let mut invalid = asc;
        invalid.error_protection_config = Some(2);
        assert_eq!(
            invalid.to_bytes().unwrap_err(),
            AscError::UnsupportedErrorProtectionConfig(2)
        );

        let mut writer = BitWriter::new();
        writer.write(17, 5);
        writer.write(4, 4);
        writer.write(2, 4);
        writer.write(0, 3);
        writer.write(2, 2);
        assert_eq!(
            AudioSpecificConfig::parse(&writer.finish()).unwrap_err(),
            AscError::UnsupportedErrorProtectionConfig(2)
        );
    }

    #[test]
    fn roundtrips_er_resilience_flags_and_extension_flag3() {
        let asc = AudioSpecificConfig {
            audio_object_type: 17,
            sampling_frequency_index: 3,
            sampling_frequency: 48_000,
            channel_configuration: 1,
            extension: None,
            ga_specific: Some(GaSpecificConfig {
                extension_flag: true,
                section_data_resilience: true,
                scalefactor_data_resilience: false,
                spectral_data_resilience: true,
                extension_flag3: Some(true),
                ..GaSpecificConfig::default()
            }),
            eld_specific: None,
            usac_config: None,
            error_protection_config: Some(0),
            program_config: None,
            bits_read: 0,
        };
        let parsed = AudioSpecificConfig::parse(&asc.to_bytes().unwrap()).unwrap();
        let ga = parsed.ga_specific.unwrap();
        assert!(ga.extension_flag);
        assert!(ga.section_data_resilience);
        assert!(!ga.scalefactor_data_resilience);
        assert!(ga.spectral_data_resilience);
        assert_eq!(ga.extension_flag3, Some(true));
        assert_eq!(parsed.error_protection_config, Some(0));
    }

    #[test]
    fn roundtrips_er_aac_eld_specific_config_and_extensions() {
        let asc = AudioSpecificConfig {
            audio_object_type: 39,
            sampling_frequency_index: 4,
            sampling_frequency: 44_100,
            channel_configuration: 1,
            extension: None,
            ga_specific: None,
            eld_specific: Some(EldSpecificConfig {
                frame_length_flag: true,
                section_data_resilience: true,
                scalefactor_data_resilience: false,
                spectral_data_resilience: true,
                sbr_present: false,
                sbr_sampling_rate: false,
                sbr_crc: false,
                sbr_headers: Vec::new(),
                extensions: vec![
                    EldExtension {
                        extension_type: 2,
                        data: vec![1, 2, 3],
                    },
                    EldExtension {
                        extension_type: 3,
                        data: (0..20).collect(),
                    },
                    EldExtension {
                        extension_type: 7,
                        data: (0..300).map(|value| value as u8).collect(),
                    },
                ],
            }),
            usac_config: None,
            error_protection_config: Some(1),
            program_config: None,
            bits_read: 0,
        };
        let bytes = asc.to_bytes().unwrap();
        let parsed = AudioSpecificConfig::parse(&bytes).unwrap();
        assert_eq!(parsed.audio_object_type, 39);
        assert_eq!(parsed.eld_specific, asc.eld_specific);
        assert_eq!(parsed.error_protection_config, Some(1));
    }

    #[test]
    fn roundtrips_eld_ld_sbr_default_header() {
        let asc = AudioSpecificConfig {
            audio_object_type: 39,
            sampling_frequency_index: 4,
            sampling_frequency: 44_100,
            channel_configuration: 1,
            extension: None,
            ga_specific: None,
            eld_specific: Some(EldSpecificConfig {
                sbr_present: true,
                sbr_sampling_rate: true,
                sbr_crc: true,
                sbr_headers: vec![LdSbrHeader {
                    amp_resolution: true,
                    crossover_band: 3,
                    start_frequency: 5,
                    stop_frequency: 10,
                    frequency_scale: Some(2),
                    alter_scale: Some(true),
                    noise_bands: Some(1),
                    limiter_bands: Some(2),
                    limiter_gains: Some(1),
                    interpol_frequency: Some(true),
                    smoothing_mode: Some(false),
                    ..LdSbrHeader::default()
                }],
                ..EldSpecificConfig::default()
            }),
            usac_config: None,
            error_protection_config: Some(0),
            program_config: None,
            bits_read: 0,
        };
        let bytes = asc.to_bytes().unwrap();
        let parsed = AudioSpecificConfig::parse(&bytes).unwrap();
        assert_eq!(parsed.eld_specific, asc.eld_specific);
    }

    #[test]
    fn roundtrips_usac_decoder_and_extension_configuration() {
        let usac = UsacConfig {
            sampling_frequency_index: 3,
            sampling_frequency: 48_000,
            core_sbr_frame_length_index: 1,
            core_frame_length: 1024,
            output_frame_length: 1024,
            sbr_ratio_index: 0,
            channel_configuration_index: 2,
            elements: vec![
                UsacElementConfig::ChannelPair {
                    noise_filling: true,
                    sbr: None,
                    stereo_config_index: 0,
                    mps212: None,
                },
                UsacElementConfig::Extension(UsacExtElementConfig {
                    extension_type: 3,
                    default_length: Some(4),
                    payload_fragmentation: true,
                    config: vec![0xaa, 0x55],
                }),
                UsacElementConfig::Extension(UsacExtElementConfig {
                    extension_type: 4,
                    default_length: None,
                    payload_fragmentation: false,
                    config: Vec::new(),
                }),
            ],
            extensions: vec![UsacConfigExtension {
                extension_type: 7,
                data: vec![1, 2, 3],
            }],
        };
        let asc = AudioSpecificConfig {
            audio_object_type: AOT_USAC,
            sampling_frequency_index: 3,
            sampling_frequency: 48_000,
            channel_configuration: 2,
            extension: None,
            ga_specific: None,
            eld_specific: None,
            usac_config: Some(usac.clone()),
            error_protection_config: None,
            program_config: None,
            bits_read: 0,
        };
        let parsed = AudioSpecificConfig::parse(&asc.to_bytes().unwrap()).unwrap();
        assert_eq!(parsed.audio_object_type, AOT_USAC);
        assert_eq!(parsed.usac_config, Some(usac));
    }

    #[test]
    fn roundtrips_usac_two_to_one_sbr_configuration() {
        let sbr = UsacSbrConfig {
            harmonic_sbr: true,
            inter_tes: true,
            pvc: false,
            start_frequency: 5,
            stop_frequency: 9,
            frequency_scale: Some(1),
            alter_scale: Some(false),
            noise_bands: Some(3),
            limiter_bands: Some(2),
            limiter_gains: Some(1),
            interpol_frequency: Some(true),
            smoothing_mode: Some(false),
        };
        let usac = UsacConfig {
            sampling_frequency_index: 3,
            sampling_frequency: 48_000,
            core_sbr_frame_length_index: 3,
            core_frame_length: 1024,
            output_frame_length: 2048,
            sbr_ratio_index: 3,
            channel_configuration_index: 1,
            elements: vec![UsacElementConfig::SingleChannel {
                noise_filling: true,
                sbr: Some(sbr),
            }],
            extensions: Vec::new(),
        };
        let asc = AudioSpecificConfig {
            audio_object_type: AOT_USAC,
            sampling_frequency_index: 3,
            sampling_frequency: 48_000,
            channel_configuration: 1,
            extension: None,
            ga_specific: None,
            eld_specific: None,
            usac_config: Some(usac.clone()),
            error_protection_config: None,
            program_config: None,
            bits_read: 0,
        };
        let parsed = AudioSpecificConfig::parse(&asc.to_bytes().unwrap()).unwrap();
        assert_eq!(parsed.usac_config, Some(usac));
    }

    #[test]
    fn roundtrips_usac_mps212_residual_configuration() {
        let sbr = UsacSbrConfig {
            harmonic_sbr: false,
            inter_tes: false,
            pvc: false,
            start_frequency: 4,
            stop_frequency: 10,
            frequency_scale: None,
            alter_scale: None,
            noise_bands: None,
            limiter_bands: None,
            limiter_gains: None,
            interpol_frequency: None,
            smoothing_mode: None,
        };
        let mps = Mps212Config {
            frequency_resolution_index: 3,
            frequency_resolution_bands: 14,
            fixed_gain_downmix: 2,
            temporal_shape_config: 2,
            decorrelation_config: 1,
            high_rate_mode: true,
            phase_coding: true,
            ott_bands_phase: Some(12),
            residual_bands: Some(8),
            pseudo_lr: true,
            environment_quantization_mode: Some(true),
        };
        let usac = UsacConfig {
            sampling_frequency_index: 3,
            sampling_frequency: 48_000,
            core_sbr_frame_length_index: 3,
            core_frame_length: 1024,
            output_frame_length: 2048,
            sbr_ratio_index: 3,
            channel_configuration_index: 2,
            elements: vec![UsacElementConfig::ChannelPair {
                noise_filling: true,
                sbr: Some(sbr),
                stereo_config_index: 2,
                mps212: Some(mps.clone()),
            }],
            extensions: Vec::new(),
        };
        let asc = AudioSpecificConfig {
            audio_object_type: AOT_USAC,
            sampling_frequency_index: 3,
            sampling_frequency: 48_000,
            channel_configuration: 2,
            extension: None,
            ga_specific: None,
            eld_specific: None,
            usac_config: Some(usac),
            error_protection_config: None,
            program_config: None,
            bits_read: 0,
        };
        let parsed = AudioSpecificConfig::parse(&asc.to_bytes().unwrap()).unwrap();
        assert_eq!(
            parsed.usac_config.unwrap().elements,
            asc.usac_config.unwrap().elements
        );
    }

    fn parse_mps_test_bits(
        writer: BitWriter,
        stereo_config_index: u8,
        drm: bool,
    ) -> Result<Mps212Config, AscError> {
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        if drm {
            Mps212Config::parse_drm(&mut reader, stereo_config_index)
        } else {
            Mps212Config::parse(&mut reader, stereo_config_index)
        }
    }

    #[test]
    fn mps212_parsers_validate_frequency_phase_decorrelation_and_residual_bands() {
        let mut invalid_frequency = BitWriter::new();
        invalid_frequency.write(0, 3);
        assert_eq!(
            parse_mps_test_bits(invalid_frequency, 0, false),
            Err(AscError::InvalidUsacMpsConfig)
        );

        let mut invalid_decorrelation = BitWriter::new();
        invalid_decorrelation.write(1, 3);
        invalid_decorrelation.write(0, 3);
        invalid_decorrelation.write(0, 2);
        invalid_decorrelation.write(3, 2);
        assert_eq!(
            parse_mps_test_bits(invalid_decorrelation, 0, false),
            Err(AscError::InvalidUsacMpsConfig)
        );

        let mut invalid_phase = BitWriter::new();
        invalid_phase.write(1, 3);
        invalid_phase.write(0, 3);
        invalid_phase.write(0, 2);
        invalid_phase.write(0, 2);
        invalid_phase.write_bool(false);
        invalid_phase.write_bool(false);
        invalid_phase.write_bool(true);
        invalid_phase.write(29, 5);
        assert_eq!(
            parse_mps_test_bits(invalid_phase, 0, false),
            Err(AscError::InvalidUsacMpsConfig)
        );

        let mut invalid_residual = BitWriter::new();
        invalid_residual.write(7, 3); // four frequency bands
        invalid_residual.write(0, 3);
        invalid_residual.write(0, 2);
        invalid_residual.write(0, 2);
        invalid_residual.write_bool(false);
        invalid_residual.write_bool(false);
        invalid_residual.write_bool(false);
        invalid_residual.write(5, 5);
        assert_eq!(
            parse_mps_test_bits(invalid_residual, 2, false),
            Err(AscError::InvalidUsacMpsConfig)
        );

        let mut no_residual = BitWriter::new();
        no_residual.write(1, 3);
        no_residual.write(0, 3);
        no_residual.write(0, 2);
        no_residual.write(0, 2);
        no_residual.write_bool(false);
        no_residual.write_bool(false);
        no_residual.write_bool(false);
        let parsed = parse_mps_test_bits(no_residual, 1, false).unwrap();
        assert_eq!(parsed.residual_bands, None);
        assert!(!parsed.pseudo_lr);

        for (bits, stereo_index) in [(vec![(0, 3)], 0), (vec![(1, 3), (0, 3)], 0)] {
            let mut writer = BitWriter::new();
            for (value, width) in bits {
                writer.write(value, width);
            }
            if stereo_index == 0 && writer.bits_written() > 3 {
                writer.write_bool(false);
                writer.write_bool(false);
                writer.write_bool(false);
                writer.write_bool(true);
                writer.write(29, 5);
            }
            assert_eq!(
                parse_mps_test_bits(writer, stereo_index, true),
                Err(AscError::InvalidUsacMpsConfig)
            );
        }

        let mut drm_residual = BitWriter::new();
        drm_residual.write(7, 3);
        drm_residual.write(0, 3);
        drm_residual.write_bool(false);
        drm_residual.write_bool(false);
        drm_residual.write_bool(false);
        drm_residual.write_bool(false);
        drm_residual.write(5, 5);
        assert_eq!(
            parse_mps_test_bits(drm_residual, 2, true),
            Err(AscError::InvalidUsacMpsConfig)
        );

        let mut drm_no_residual = BitWriter::new();
        drm_no_residual.write(1, 3);
        drm_no_residual.write(0, 3);
        drm_no_residual.write_bool(false);
        drm_no_residual.write_bool(false);
        drm_no_residual.write_bool(false);
        drm_no_residual.write_bool(false);
        let parsed = parse_mps_test_bits(drm_no_residual, 1, true).unwrap();
        assert_eq!(parsed.residual_bands, None);
        assert!(!parsed.pseudo_lr);
    }

    #[test]
    fn usac_parser_covers_explicit_rates_remaining_ratios_lfe_and_validation() {
        let sbr = UsacSbrConfig {
            harmonic_sbr: false,
            inter_tes: false,
            pvc: false,
            start_frequency: 5,
            stop_frequency: 9,
            frequency_scale: None,
            alter_scale: None,
            noise_bands: None,
            limiter_bands: None,
            limiter_gains: None,
            interpol_frequency: None,
            smoothing_mode: None,
        };
        for (frame_index, expected_core, expected_output, expected_ratio) in
            [(2, 768, 2048, 2), (4, 1024, 4096, 1)]
        {
            let elements = vec![
                UsacElementConfig::SingleChannel {
                    noise_filling: false,
                    sbr: Some(sbr.clone()),
                },
                UsacElementConfig::ChannelPair {
                    noise_filling: false,
                    sbr: Some(sbr.clone()),
                    stereo_config_index: 0,
                    mps212: None,
                },
                UsacElementConfig::ChannelPair {
                    noise_filling: false,
                    sbr: Some(sbr.clone()),
                    stereo_config_index: 0,
                    mps212: None,
                },
                UsacElementConfig::Lfe {
                    sbr: Some(sbr.clone()),
                },
            ];
            let config = UsacConfig {
                sampling_frequency_index: 3,
                sampling_frequency: 48_000,
                core_sbr_frame_length_index: frame_index,
                core_frame_length: expected_core,
                output_frame_length: expected_output,
                sbr_ratio_index: expected_ratio,
                channel_configuration_index: 6,
                elements: elements.clone(),
                extensions: Vec::new(),
            };
            let mut writer = BitWriter::new();
            config.write(&mut writer).unwrap();
            let parsed = UsacConfig::parse(&mut BitReader::new(&writer.finish())).unwrap();
            assert_eq!(parsed.core_frame_length, expected_core);
            assert_eq!(parsed.output_frame_length, expected_output);
            assert_eq!(parsed.sbr_ratio_index, expected_ratio);
            assert_eq!(parsed.elements, elements);
        }

        let explicit = UsacConfig {
            sampling_frequency_index: 31,
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
        let mut writer = BitWriter::new();
        explicit.write(&mut writer).unwrap();
        assert_eq!(
            UsacConfig::parse(&mut BitReader::new(&writer.finish()))
                .unwrap()
                .sampling_frequency,
            48_000
        );

        let mut invalid_rate = BitWriter::new();
        invalid_rate.write(31, 5);
        invalid_rate.write(0, 24);
        assert_eq!(
            UsacConfig::parse(&mut BitReader::new(&invalid_rate.finish())),
            Err(AscError::UnsupportedSampleRate(0))
        );
        let mut invalid_index = BitWriter::new();
        invalid_index.write(13, 5);
        assert_eq!(
            UsacConfig::parse(&mut BitReader::new(&invalid_index.finish())),
            Err(AscError::InvalidSamplingFrequencyIndex(13))
        );

        let mut invalid_channel = BitWriter::new();
        invalid_channel.write(3, 5);
        invalid_channel.write(1, 3);
        invalid_channel.write(0, 5);
        assert_eq!(
            UsacConfig::parse(&mut BitReader::new(&invalid_channel.finish())),
            Err(AscError::InvalidUsacChannelConfiguration(0))
        );

        let mut tw_mdct = BitWriter::new();
        tw_mdct.write(3, 5);
        tw_mdct.write(1, 3);
        tw_mdct.write(1, 5);
        tw_mdct.write(0, 4); // one element
        tw_mdct.write(0, 2); // single channel
        tw_mdct.write_bool(true);
        assert_eq!(
            UsacConfig::parse(&mut BitReader::new(&tw_mdct.finish())),
            Err(AscError::UnsupportedUsacTwMdct)
        );

        let mut mismatch = BitWriter::new();
        mismatch.write(3, 5);
        mismatch.write(1, 3);
        mismatch.write(2, 5);
        mismatch.write(0, 4); // one element
        mismatch.write(2, 2); // one-channel LFE for a two-channel config
        assert_eq!(
            UsacConfig::parse(&mut BitReader::new(&mismatch.finish())),
            Err(AscError::UsacChannelCountMismatch {
                expected: 2,
                actual: 1,
            })
        );
    }

    #[test]
    fn usac_channel_configuration_table_matches_fdk_for_all_indices() {
        for index in 1..=13 {
            let (_, sce_count, cpe_count, lfe_count) = usac_channel_layout(index).unwrap();
            let mut elements = Vec::new();
            elements.extend((0..sce_count).map(|_| UsacElementConfig::SingleChannel {
                noise_filling: false,
                sbr: None,
            }));
            elements.extend((0..cpe_count).map(|_| UsacElementConfig::ChannelPair {
                noise_filling: false,
                sbr: None,
                stereo_config_index: 0,
                mps212: None,
            }));
            elements.extend((0..lfe_count).map(|_| UsacElementConfig::Lfe { sbr: None }));
            let config = UsacConfig {
                sampling_frequency_index: 3,
                sampling_frequency: 48_000,
                core_sbr_frame_length_index: 1,
                core_frame_length: 1024,
                output_frame_length: 1024,
                sbr_ratio_index: 0,
                channel_configuration_index: index,
                elements,
                extensions: Vec::new(),
            };
            let mut writer = BitWriter::new();
            config.write(&mut writer).unwrap();
            assert_eq!(
                UsacConfig::parse(&mut BitReader::new(&writer.finish())).unwrap(),
                config
            );
        }

        let wrong_two_channel_layout = UsacConfig {
            sampling_frequency_index: 3,
            sampling_frequency: 48_000,
            core_sbr_frame_length_index: 1,
            core_frame_length: 1024,
            output_frame_length: 1024,
            sbr_ratio_index: 0,
            channel_configuration_index: 8,
            elements: vec![UsacElementConfig::ChannelPair {
                noise_filling: false,
                sbr: None,
                stereo_config_index: 0,
                mps212: None,
            }],
            extensions: Vec::new(),
        };
        assert!(matches!(
            wrong_two_channel_layout.write(&mut BitWriter::new()),
            Err(AscError::UsacElementLayoutMismatch { .. })
        ));
    }

    #[test]
    fn audio_object_sampling_and_escaped_value_helpers_cover_boundaries() {
        for invalid in [0, AOT_ESCAPE, 96, u8::MAX] {
            let mut writer = BitWriter::new();
            assert_eq!(
                write_audio_object_type(&mut writer, invalid),
                Err(AscError::InvalidAudioObjectType(invalid))
            );
        }
        for value in [1, 30, 32, 95] {
            let mut writer = BitWriter::new();
            write_audio_object_type(&mut writer, value).unwrap();
            let bytes = writer.finish();
            assert_eq!(
                read_audio_object_type(&mut BitReader::new(&bytes)).unwrap(),
                value
            );
        }

        let mut writer = BitWriter::new();
        write_sampling_frequency(&mut writer, 15, 12_345).unwrap();
        let bytes = writer.finish();
        assert_eq!(
            read_sampling_frequency(&mut BitReader::new(&bytes)).unwrap(),
            (15, 12_345)
        );
        let mut writer = BitWriter::new();
        assert_eq!(
            write_sampling_frequency(&mut writer, 4, 48_000),
            Err(AscError::UnsupportedSampleRate(48_000))
        );
        let mut writer = BitWriter::new();
        assert_eq!(
            write_sampling_frequency(&mut writer, 13, 1),
            Err(AscError::InvalidSamplingFrequencyIndex(13))
        );

        for value in [0, 7, 8, 22, 23, 277] {
            let mut writer = BitWriter::new();
            write_escaped_value(&mut writer, value, 3, 4, 8).unwrap();
            let bytes = writer.finish();
            assert_eq!(
                read_escaped_value(&mut BitReader::new(&bytes), 3, 4, 8).unwrap(),
                value
            );
        }
        let mut writer = BitWriter::new();
        assert_eq!(
            write_escaped_value(&mut writer, 278, 3, 4, 8),
            Err(AscError::EscapedValueTooLarge)
        );
    }

    #[test]
    fn converts_and_formats_every_asc_error_variant() {
        assert_eq!(
            AscError::from(BitError::UnexpectedEof {
                needed_bits: 3,
                remaining_bits: 1,
            }),
            AscError::UnexpectedEof {
                needed_bits: 3,
                remaining_bits: 1,
            }
        );
        assert_eq!(
            AscError::from(BitError::TooManyBitsRequested {
                requested_bits: 33,
                max_bits: 32,
            }),
            AscError::UnexpectedEof {
                needed_bits: usize::MAX,
                remaining_bits: 0,
            }
        );
        assert_eq!(
            AscError::from(BitError::InvalidPushBack {
                requested_bits: 1,
                bits_read: 0,
            }),
            AscError::UnexpectedEof {
                needed_bits: 0,
                remaining_bits: 0,
            }
        );

        let errors = [
            AscError::UnexpectedEof {
                needed_bits: 1,
                remaining_bits: 0,
            },
            AscError::InvalidAudioObjectType(0),
            AscError::InvalidSamplingFrequencyIndex(13),
            AscError::InvalidChannelConfiguration(8),
            AscError::UnsupportedSampleRate(1),
            AscError::MissingProgramConfigElement,
            AscError::MissingErrorProtectionConfig,
            AscError::MissingScalableLayer,
            AscError::MissingBsacExtension,
            AscError::UnsupportedErrorProtectionConfig(2),
            AscError::ProgramConfigTooLarge,
            AscError::InvalidCoreCoderDelay(0xffff),
            AscError::MissingCoreCoderDelay,
            AscError::MissingEldSpecificConfig,
            AscError::UnsupportedEldSbr,
            AscError::UnsupportedLdSbrChannelConfiguration(0),
            AscError::LdSbrHeaderCount {
                expected: 1,
                actual: 0,
            },
            AscError::InvalidLdSbrHeader,
            AscError::TooManyEldExtensions,
            AscError::InvalidEldExtensionType(0),
            AscError::EldExtensionTooLong(1),
            AscError::MissingUsacConfig,
            AscError::InvalidUsacFrameLengthIndex(0),
            AscError::InvalidUsacChannelConfiguration(0),
            AscError::UnsupportedUsacTwMdct,
            AscError::UnsupportedUsacSbrConfig,
            AscError::UnsupportedUsacMpsConfig(3),
            AscError::InvalidUsacMpsConfig,
            AscError::UsacChannelCountMismatch {
                expected: 2,
                actual: 1,
            },
            AscError::UsacElementLayoutMismatch {
                expected_sce: 0,
                expected_cpe: 1,
                expected_lfe: 0,
                actual_sce: 2,
                actual_cpe: 0,
                actual_lfe: 0,
            },
            AscError::EscapedValueTooLarge,
        ];
        for error in errors {
            assert!(!error.to_string().is_empty());
        }
    }

    #[test]
    fn roundtrips_scalable_bsac_and_full_program_configuration() {
        for (aot, ga, ep) in [
            (
                6,
                GaSpecificConfig {
                    depends_on_core_coder: true,
                    core_coder_delay: Some(0x123),
                    layer: Some(5),
                    ..GaSpecificConfig::default()
                },
                None,
            ),
            (
                22,
                GaSpecificConfig {
                    extension_flag: true,
                    num_of_subframes: Some(17),
                    layer_length: Some(1234),
                    extension_flag3: Some(true),
                    ..GaSpecificConfig::default()
                },
                Some(1),
            ),
        ] {
            let asc = AudioSpecificConfig {
                audio_object_type: aot,
                sampling_frequency_index: 4,
                sampling_frequency: 44_100,
                channel_configuration: 1,
                extension: None,
                ga_specific: Some(ga),
                eld_specific: None,
                usac_config: None,
                error_protection_config: ep,
                program_config: None,
                bits_read: 0,
            };
            let parsed = AudioSpecificConfig::parse(&asc.to_bytes().unwrap()).unwrap();
            assert_eq!(parsed.ga_specific, Some(ga));
            assert_eq!(parsed.error_protection_config, ep);
        }

        let pce = ProgramConfig {
            element_instance_tag: 3,
            profile: 1,
            sampling_frequency_index: 4,
            front: vec![ProgramElement {
                is_cpe: true,
                tag_select: 1,
            }],
            side: vec![ProgramElement {
                is_cpe: false,
                tag_select: 2,
            }],
            back: vec![ProgramElement {
                is_cpe: true,
                tag_select: 3,
            }],
            lfe: vec![4],
            associated_data: vec![5],
            valid_cc: vec![ProgramCcElement {
                is_ind_sw: true,
                tag_select: 6,
            }],
            mono_mixdown_element_number: Some(7),
            stereo_mixdown_element_number: Some(8),
            matrix_mixdown: Some(MatrixMixdown {
                index: 2,
                pseudo_surround_enable: true,
            }),
            comment: b"all fields".to_vec(),
            num_channels: 6,
            num_effective_channels: 5,
        };
        let mut writer = BitWriter::new();
        pce.write_to_writer(&mut writer).unwrap();
        let parsed = ProgramConfig::parse_from_bytes(&writer.finish()).unwrap();
        assert_eq!(parsed, pce);
    }

    #[test]
    fn ga_and_top_level_writers_reject_missing_or_invalid_fields() {
        let base = AudioSpecificConfig::aac_lc(44_100, 1).unwrap();
        let mut invalid = base.clone();
        invalid.channel_configuration = 8;
        assert_eq!(
            invalid.to_bytes(),
            Err(AscError::InvalidChannelConfiguration(8))
        );

        let mut scalable = base.clone();
        scalable.audio_object_type = 6;
        scalable.ga_specific = Some(GaSpecificConfig::default());
        assert_eq!(scalable.to_bytes(), Err(AscError::MissingScalableLayer));

        let mut delayed = base.clone();
        delayed.ga_specific = Some(GaSpecificConfig {
            depends_on_core_coder: true,
            ..GaSpecificConfig::default()
        });
        assert_eq!(delayed.to_bytes(), Err(AscError::MissingCoreCoderDelay));

        let mut bsac = base.clone();
        bsac.audio_object_type = 22;
        bsac.ga_specific = Some(GaSpecificConfig {
            extension_flag: true,
            ..GaSpecificConfig::default()
        });
        assert_eq!(bsac.to_bytes(), Err(AscError::MissingBsacExtension));

        let mut er = base;
        er.audio_object_type = 17;
        assert_eq!(er.to_bytes(), Err(AscError::MissingErrorProtectionConfig));

        let base = AudioSpecificConfig::aac_lc(44_100, 1).unwrap();
        let mut invalid_base_rate = base.clone();
        invalid_base_rate.sampling_frequency_index = 13;
        assert_eq!(
            invalid_base_rate.to_bytes(),
            Err(AscError::InvalidSamplingFrequencyIndex(13))
        );

        let mut invalid_extension_rate = base.clone();
        invalid_extension_rate.extension = Some(AudioSpecificConfigExtension {
            audio_object_type: 5,
            sampling_frequency_index: 13,
            sampling_frequency: 44_100,
            ps_present: false,
        });
        assert_eq!(
            invalid_extension_rate.to_bytes(),
            Err(AscError::InvalidSamplingFrequencyIndex(13))
        );

        let mut invalid_core_rate = base;
        invalid_core_rate.extension = Some(AudioSpecificConfigExtension {
            audio_object_type: 5,
            sampling_frequency_index: 4,
            sampling_frequency: 44_100,
            ps_present: false,
        });
        invalid_core_rate.sampling_frequency_index = 13;
        assert_eq!(
            invalid_core_rate.to_bytes(),
            Err(AscError::InvalidSamplingFrequencyIndex(13))
        );
    }

    #[test]
    fn extension_writer_and_parser_channel_validation_roundtrip() {
        let parsed = AudioSpecificConfig::parse(&[0x2a, 0x12, 0x08, 0x00]).unwrap();
        let reparsed = AudioSpecificConfig::parse(&parsed.to_bytes().unwrap()).unwrap();
        assert_eq!(reparsed.audio_object_type, AOT_AAC_LC);
        assert_eq!(reparsed.extension, parsed.extension);

        let mut writer = BitWriter::new();
        writer.write(2, 5);
        writer.write(4, 4);
        writer.write(8, 4);
        assert_eq!(
            AudioSpecificConfig::parse(&writer.finish()),
            Err(AscError::InvalidChannelConfiguration(8))
        );
    }

    #[test]
    fn ld_sbr_eld_and_program_writers_validate_all_size_classes() {
        for header in [
            LdSbrHeader {
                crossover_band: 8,
                ..LdSbrHeader::default()
            },
            LdSbrHeader {
                frequency_scale: Some(4),
                ..LdSbrHeader::default()
            },
        ] {
            let mut writer = BitWriter::new();
            assert_eq!(header.write(&mut writer), Err(AscError::InvalidLdSbrHeader));
        }
        let mut writer = BitWriter::new();
        assert_eq!(
            LdSbrHeader {
                frequency_scale: Some(1),
                ..LdSbrHeader::default()
            }
            .write(&mut writer),
            Err(AscError::InvalidLdSbrHeader)
        );

        let mut asc = AudioSpecificConfig {
            audio_object_type: 39,
            sampling_frequency_index: 4,
            sampling_frequency: 44_100,
            channel_configuration: 1,
            extension: None,
            ga_specific: None,
            eld_specific: Some(EldSpecificConfig {
                sbr_present: true,
                sbr_headers: Vec::new(),
                ..EldSpecificConfig::default()
            }),
            usac_config: None,
            error_protection_config: Some(0),
            program_config: None,
            bits_read: 0,
        };
        assert!(matches!(
            asc.to_bytes(),
            Err(AscError::LdSbrHeaderCount {
                expected: 1,
                actual: 0
            })
        ));
        asc.eld_specific = Some(EldSpecificConfig {
            sbr_present: false,
            sbr_headers: vec![LdSbrHeader::default()],
            ..EldSpecificConfig::default()
        });
        assert!(matches!(
            asc.to_bytes(),
            Err(AscError::LdSbrHeaderCount {
                expected: 0,
                actual: 1
            })
        ));
        asc.eld_specific = Some(EldSpecificConfig {
            extensions: vec![EldExtension {
                extension_type: 0,
                data: Vec::new(),
            }],
            ..EldSpecificConfig::default()
        });
        assert_eq!(asc.to_bytes(), Err(AscError::InvalidEldExtensionType(0)));
        asc.eld_specific = Some(EldSpecificConfig {
            extensions: vec![EldExtension {
                extension_type: 3,
                data: vec![0; 65_806],
            }],
            ..EldSpecificConfig::default()
        });
        assert_eq!(asc.to_bytes(), Err(AscError::EldExtensionTooLong(65_806)));

        for (channel_configuration, header_count) in [(3, 2), (4, 3), (7, 4)] {
            let config = EldSpecificConfig {
                sbr_present: true,
                sbr_headers: vec![LdSbrHeader::default(); header_count],
                ..EldSpecificConfig::default()
            };
            config
                .write(&mut BitWriter::new(), channel_configuration)
                .unwrap();
        }
        let config = EldSpecificConfig {
            sbr_present: true,
            ..EldSpecificConfig::default()
        };
        assert_eq!(
            config.write(&mut BitWriter::new(), 0),
            Err(AscError::UnsupportedLdSbrChannelConfiguration(0))
        );
        let mut unsupported_sbr_channel = BitWriter::new();
        for _ in 0..4 {
            unsupported_sbr_channel.write_bool(false);
        }
        unsupported_sbr_channel.write_bool(true);
        unsupported_sbr_channel.write_bool(false);
        unsupported_sbr_channel.write_bool(false);
        assert_eq!(
            EldSpecificConfig::parse(&mut BitReader::new(&unsupported_sbr_channel.finish()), 0),
            Err(AscError::UnsupportedLdSbrChannelConfiguration(0))
        );
        let config = EldSpecificConfig {
            extensions: vec![
                EldExtension {
                    extension_type: 1,
                    data: Vec::new(),
                };
                16
            ],
            ..EldSpecificConfig::default()
        };
        assert_eq!(
            config.write(&mut BitWriter::new(), 1),
            Err(AscError::TooManyEldExtensions)
        );

        let mut excessive_extensions = BitWriter::new();
        for _ in 0..5 {
            excessive_extensions.write_bool(false);
        }
        for _ in 0..16 {
            excessive_extensions.write(1, 4);
            excessive_extensions.write(0, 4);
        }
        assert_eq!(
            EldSpecificConfig::parse(&mut BitReader::new(&excessive_extensions.finish()), 1),
            Err(AscError::TooManyEldExtensions)
        );

        let mut pce = ProgramConfig::default();
        pce.front = vec![
            ProgramElement {
                is_cpe: false,
                tag_select: 0,
            };
            16
        ];
        let mut writer = BitWriter::new();
        assert_eq!(
            pce.write_to_writer(&mut writer),
            Err(AscError::ProgramConfigTooLarge)
        );
        pce.front.clear();
        pce.lfe = vec![0; 4];
        let mut writer = BitWriter::new();
        assert_eq!(
            pce.write_to_writer(&mut writer),
            Err(AscError::ProgramConfigTooLarge)
        );
        pce.lfe.clear();
        pce.comment = vec![0; 256];
        let mut writer = BitWriter::new();
        assert_eq!(
            pce.write_to_writer(&mut writer),
            Err(AscError::ProgramConfigTooLarge)
        );

        let mut invalid_delay = BitWriter::new();
        assert_eq!(
            GaSpecificConfig {
                core_coder_delay: Some(u16::MAX),
                ..GaSpecificConfig::default()
            }
            .write_base(&mut invalid_delay),
            Err(AscError::InvalidCoreCoderDelay(u16::MAX))
        );

        let mut invalid_pce_rate = BitWriter::new();
        invalid_pce_rate.write(0, 4);
        invalid_pce_rate.write(0, 2);
        invalid_pce_rate.write(13, 4);
        assert_eq!(
            ProgramConfig::parse_from_bytes(&invalid_pce_rate.finish()),
            Err(AscError::InvalidSamplingFrequencyIndex(13))
        );
        assert_eq!(
            AudioSpecificConfig::parse(&[0]),
            Err(AscError::InvalidAudioObjectType(0))
        );
    }
}
