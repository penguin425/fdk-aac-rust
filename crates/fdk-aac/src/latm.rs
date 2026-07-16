//! Pure Rust LATM AudioMuxElement parsing for AAC-LC streams.

use std::fmt;

use crate::asc::{AscError, AudioSpecificConfig};
use crate::bits::{BitError, BitReader, BitWriter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LatmAudioMuxElement {
    pub use_same_stream_mux: bool,
    pub config: Option<AudioSpecificConfig>,
    pub mux_config: Option<LatmMuxConfig>,
    /// StreamMuxConfig CRC byte. FDK transports and exposes this field but do
    /// not treat it as a payload CRC.
    pub crc_check_sum: Option<u8>,
    pub raw_data_block: Vec<u8>,
    pub raw_data_block_bits: usize,
    pub payloads: Vec<LatmPayload>,
    pub other_data: Vec<u8>,
    pub other_data_bits: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LatmPayload {
    pub subframe: u8,
    pub program: u8,
    pub layer: u8,
    pub data: Vec<u8>,
    pub bits: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LatmStreamLayer {
    pub program: u8,
    pub layer: u8,
    pub config: Option<AudioSpecificConfig>,
    pub frame_length_type: u8,
    pub fixed_frame_length_bits: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LatmMuxConfig {
    pub audio_mux_version: u8,
    pub all_streams_same_time_framing: bool,
    pub subframe_count: u8,
    pub streams: Vec<LatmStreamLayer>,
    pub other_data_bits: usize,
    pub crc_check_sum: Option<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LatmAacLcWriter {
    sampling_frequency_index: u8,
    channel_configuration: u8,
    config_written: bool,
}

/// Stateful LATM writer for any AudioSpecificConfig supported by this crate.
///
/// Unlike `LatmAacLcWriter`, this preserves the exact ASC syntax for ER AAC-LD,
/// AAC-ELD, HE-AAC and PS, supports AudioMuxVersion 0/1, multiple subframes and
/// periodic StreamMuxConfig repetition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LatmWriter {
    config: AudioSpecificConfig,
    sbr_signaling_mode: u8,
    audio_mux_version: u8,
    subframe_count: u8,
    header_period: u8,
    frames_since_config: u8,
    config_written: bool,
    crc_check_sum: Option<u8>,
    other_data: Vec<u8>,
    other_data_bits: usize,
    fixed_frame_length_bits: Option<usize>,
}

impl LatmWriter {
    pub fn new(
        config: AudioSpecificConfig,
        audio_mux_version: u8,
        subframe_count: u8,
        header_period: u8,
    ) -> Result<Self, LatmError> {
        Self::new_with_sbr_signaling(config, audio_mux_version, subframe_count, header_period, 2)
    }

    pub fn new_with_sbr_signaling(
        config: AudioSpecificConfig,
        audio_mux_version: u8,
        subframe_count: u8,
        header_period: u8,
        sbr_signaling_mode: u8,
    ) -> Result<Self, LatmError> {
        validate_audio_specific_config(&config)?;
        if config.extension.is_some() && sbr_signaling_mode > 2 {
            return Err(LatmError::Asc(AscError::InvalidSbrSignalingMode(
                sbr_signaling_mode,
            )));
        }
        if audio_mux_version > 1 {
            return Err(LatmError::UnsupportedAudioMuxVersion(audio_mux_version));
        }
        if !(1..=4).contains(&subframe_count) {
            return Err(LatmError::InvalidSubframeCount(subframe_count));
        }
        Ok(Self {
            config,
            sbr_signaling_mode,
            audio_mux_version,
            subframe_count,
            header_period,
            frames_since_config: 0,
            config_written: false,
            crc_check_sum: None,
            other_data: Vec::new(),
            other_data_bits: 0,
            fixed_frame_length_bits: None,
        })
    }

    /// Set the optional eight-bit StreamMuxConfig CRC field.  LATM carries
    /// this value as configuration metadata; it is not a CRC over each
    /// AudioMuxElement payload.
    pub fn set_crc_check_sum(&mut self, crc_check_sum: Option<u8>) {
        if self.crc_check_sum != crc_check_sum {
            self.crc_check_sum = crc_check_sum;
            // A changed StreamMuxConfig field must be signalled before a
            // useSameStreamMux frame can refer to it.
            self.config_written = false;
        }
    }

    /// Set the `otherData` bits appended after all payload mux slots.
    pub fn set_other_data(&mut self, data: Vec<u8>, bits: usize) -> Result<(), LatmError> {
        if bits > data.len().saturating_mul(8) {
            return Err(LatmError::InvalidOtherDataLength {
                bits,
                available_bits: data.len().saturating_mul(8),
            });
        }
        if bits > u32::MAX as usize {
            return Err(LatmError::OtherDataTooLarge);
        }
        if self.other_data_bits != bits {
            self.config_written = false;
        }
        self.other_data = data;
        self.other_data_bits = bits;
        Ok(())
    }

    /// Select LATM `frameLengthType == 1` and its fixed payload size. `None`
    /// restores byte-length-prefixed `frameLengthType == 0`.
    pub fn set_fixed_frame_length_bits(&mut self, bits: Option<usize>) -> Result<(), LatmError> {
        if bits.is_some_and(|bits| bits == 0 || bits > 0x1ff) {
            return Err(LatmError::InvalidFixedFrameLength(bits.unwrap()));
        }
        if self.fixed_frame_length_bits != bits {
            self.fixed_frame_length_bits = bits;
            self.config_written = false;
        }
        Ok(())
    }

    pub fn write_audio_mux_element(&mut self, payloads: &[Vec<u8>]) -> Result<Vec<u8>, LatmError> {
        if payloads.len() != self.subframe_count as usize {
            return Err(LatmError::PayloadCountMismatch {
                expected: self.subframe_count as usize,
                actual: payloads.len(),
            });
        }
        if let Some(bits) = self.fixed_frame_length_bits {
            if let Some(payload) = payloads
                .iter()
                .find(|payload| bits > payload.len().saturating_mul(8))
            {
                return Err(LatmError::InvalidPayloadBitLength {
                    bits,
                    available_bits: payload.len().saturating_mul(8),
                });
            }
        }
        let repeat_config = !self.config_written
            || (self.header_period != 0 && self.frames_since_config >= self.header_period);
        let mut writer = BitWriter::new();
        writer.write_bool(!repeat_config);
        if repeat_config {
            self.write_stream_mux_config(&mut writer)?;
            self.config_written = true;
            self.frames_since_config = 0;
        }
        for payload in payloads {
            if let Some(bits) = self.fixed_frame_length_bits {
                write_packed_prefix(&mut writer, payload, bits);
            } else {
                write_payload_length(&mut writer, payload.len());
                for &byte in payload {
                    writer.write(byte as u32, 8);
                }
            }
        }
        write_packed_prefix(&mut writer, &self.other_data, self.other_data_bits);
        self.frames_since_config = self.frames_since_config.saturating_add(1);
        Ok(writer.finish())
    }

    fn write_stream_mux_config(&self, writer: &mut BitWriter) -> Result<(), LatmError> {
        writer.write_bool(self.audio_mux_version != 0);
        if self.audio_mux_version != 0 {
            writer.write_bool(false); // audioMuxVersionA
            write_latm_value(writer, 0xff); // taraBufferFullness
        }
        writer.write_bool(true); // allStreamsSameTimeFraming
        writer.write((self.subframe_count - 1) as u32, 6);
        writer.write(0, 4); // numProgram
        writer.write(0, 3); // numLayer

        let (bytes, bits) = self
            .config
            .to_bytes_with_sbr_signaling(self.sbr_signaling_mode)?;
        if self.audio_mux_version == 0 {
            write_packed_prefix(writer, &bytes, bits);
        } else {
            write_latm_value(writer, bits as u32);
            write_packed_prefix(writer, &bytes, bits);
        }
        if let Some(bits) = self.fixed_frame_length_bits {
            writer.write(1, 3); // frameLengthType
            writer.write(bits as u32, 9); // frameLength
        } else {
            writer.write(0, 3); // frameLengthType
            writer.write(0xff, 8); // latmBufferFullness
        }
        writer.write_bool(self.other_data_bits != 0);
        if self.other_data_bits != 0 {
            if self.audio_mux_version == 1 {
                write_latm_value(writer, self.other_data_bits as u32);
            } else {
                write_latm_other_data_length(writer, self.other_data_bits);
            }
        }
        writer.write_bool(self.crc_check_sum.is_some());
        if let Some(crc_check_sum) = self.crc_check_sum {
            writer.write(crc_check_sum as u32, 8);
        }
        Ok(())
    }
}

fn write_payload_length(writer: &mut BitWriter, mut length: usize) {
    while length >= 255 {
        writer.write(255, 8);
        length -= 255;
    }
    writer.write(length as u32, 8);
}

fn write_packed_prefix(writer: &mut BitWriter, bytes: &[u8], bits: usize) {
    for bit in 0..bits {
        writer.write_bool(((bytes[bit / 8] >> (7 - bit % 8)) & 1) != 0);
    }
}

fn write_latm_value(writer: &mut BitWriter, value: u32) {
    let bytes = if value <= 0xff {
        1
    } else if value <= 0xffff {
        2
    } else if value <= 0xff_ffff {
        3
    } else {
        4
    };
    writer.write((bytes - 1) as u32, 2);
    for shift in (0..bytes).rev() {
        writer.write((value >> (shift * 8)) & 0xff, 8);
    }
}

fn write_latm_other_data_length(writer: &mut BitWriter, value: usize) {
    let bytes = ((usize::BITS - value.leading_zeros()).max(1) as usize).div_ceil(8);
    for index in (0..bytes).rev() {
        writer.write_bool(index != 0);
        writer.write(((value >> (index * 8)) & 0xff) as u32, 8);
    }
}

impl LatmAacLcWriter {
    pub fn new(sampling_frequency_index: u8, channel_configuration: u8) -> Result<Self, LatmError> {
        if sampling_frequency_index >= 13 {
            return Err(LatmError::InvalidSamplingFrequencyIndex(
                sampling_frequency_index,
            ));
        }
        if !matches!(channel_configuration, 1..=7 | 11 | 12 | 14) {
            return Err(LatmError::InvalidChannelConfiguration(
                channel_configuration,
            ));
        }
        Ok(Self {
            sampling_frequency_index,
            channel_configuration,
            config_written: false,
        })
    }

    pub fn write_audio_mux_element(&mut self, raw_data_block: &[u8]) -> Vec<u8> {
        let mut writer = BitWriter::new();
        writer.write_bool(self.config_written);
        if !self.config_written {
            writer.write_bool(false); // audioMuxVersion
            writer.write_bool(true); // allStreamsSameTimeFraming
            writer.write(0, 6); // numSubFrames
            writer.write(0, 4); // numProgram
            writer.write(0, 3); // numLayer
            writer.write(2, 5); // AAC-LC
            writer.write(self.sampling_frequency_index as u32, 4);
            writer.write(self.channel_configuration as u32, 4);
            writer.write_bool(false); // frameLengthFlag
            writer.write_bool(false); // dependsOnCoreCoder
            writer.write_bool(false); // extensionFlag
            writer.write(0, 3); // frameLengthType
            writer.write(0xff, 8); // latmBufferFullness
            writer.write_bool(false); // otherDataPresent
            writer.write_bool(false); // crcCheckPresent
            self.config_written = true;
        }
        let mut length = raw_data_block.len();
        while length >= 255 {
            writer.write(255, 8);
            length -= 255;
        }
        writer.write(length as u32, 8);
        for &byte in raw_data_block {
            writer.write(byte as u32, 8);
        }
        writer.finish()
    }
}

impl LatmAudioMuxElement {
    /// Parse the interoperable AAC-LC subset: AudioMuxVersion 0, one program,
    /// one layer and frameLengthType 0.
    pub fn parse_aac_lc(input: &[u8]) -> Result<Self, LatmError> {
        Self::parse_aac_lc_with_state(input, None)
    }

    pub fn parse_aac_lc_with_state(
        input: &[u8],
        previous_mux_config: Option<&LatmMuxConfig>,
    ) -> Result<Self, LatmError> {
        let mut reader = BitReader::new(input);
        let use_same_stream_mux = reader.read_bool()?;
        let mux_config = if use_same_stream_mux {
            previous_mux_config
                .cloned()
                .ok_or(LatmError::MissingStreamMuxConfig)?
        } else {
            parse_stream_mux_config_aac_lc(&mut reader)?
        };
        if !mux_config.all_streams_same_time_framing {
            return Err(LatmError::StreamsNotSameTimeFramed);
        }
        let mut payloads =
            Vec::with_capacity(mux_config.subframe_count as usize * mux_config.streams.len());
        for subframe in 0..mux_config.subframe_count {
            let mut lengths = Vec::with_capacity(mux_config.streams.len());
            for stream in &mux_config.streams {
                let bits = match stream.frame_length_type {
                    0 => read_payload_length_bytes(&mut reader)? * 8,
                    1 => stream
                        .fixed_frame_length_bits
                        .ok_or(LatmError::UnsupportedFrameLengthType(1))?,
                    value => return Err(LatmError::UnsupportedFrameLengthType(value)),
                };
                lengths.push(bits);
            }
            for (stream, bits) in mux_config.streams.iter().zip(lengths) {
                payloads.push(LatmPayload {
                    subframe,
                    program: stream.program,
                    layer: stream.layer,
                    data: read_packed_bits(&mut reader, bits)?,
                    bits,
                });
            }
        }
        let other_data = read_packed_bits(&mut reader, mux_config.other_data_bits)?;
        let first_payload = payloads.first();
        Ok(Self {
            use_same_stream_mux,
            config: (!use_same_stream_mux)
                .then(|| mux_config.streams.first()?.config.clone())
                .flatten(),
            mux_config: (!use_same_stream_mux).then_some(mux_config.clone()),
            crc_check_sum: mux_config.crc_check_sum,
            raw_data_block: first_payload.map_or_else(Vec::new, |payload| payload.data.clone()),
            raw_data_block_bits: first_payload.map_or(0, |payload| payload.bits),
            payloads,
            other_data,
            other_data_bits: mux_config.other_data_bits,
        })
    }

    /// Parse with StreamMuxConfig state retained from an earlier
    /// `useSameStreamMux == 0` AudioMuxElement.
    pub fn parse_aac_lc_with_mux_state(
        input: &[u8],
        previous_other_data_bits: usize,
        previous_crc_check_sum: Option<u8>,
    ) -> Result<Self, LatmError> {
        let previous = LatmMuxConfig {
            audio_mux_version: 0,
            all_streams_same_time_framing: true,
            subframe_count: 1,
            streams: vec![LatmStreamLayer {
                program: 0,
                layer: 0,
                config: None,
                frame_length_type: 0,
                fixed_frame_length_bits: None,
            }],
            other_data_bits: previous_other_data_bits,
            crc_check_sum: previous_crc_check_sum,
        };
        Self::parse_aac_lc_with_state(input, Some(&previous))
    }
}

fn parse_stream_mux_config_aac_lc(reader: &mut BitReader<'_>) -> Result<LatmMuxConfig, LatmError> {
    let audio_mux_version = u8::from(reader.read_bool()?);
    if audio_mux_version == 1 {
        let audio_mux_version_a = u8::from(reader.read_bool()?);
        if audio_mux_version_a != 0 {
            return Err(LatmError::UnsupportedAudioMuxVersionA(audio_mux_version_a));
        }
        let _tara_buffer_fullness = read_latm_value(reader)?;
    }
    let all_streams_same_time_framing = reader.read_bool()?;
    let subframe_count = reader.read_u8(6)? + 1;
    let program_count = reader.read_u8(4)? + 1;
    let mut streams = Vec::new();
    for program in 0..program_count {
        let layer_count = reader.read_u8(3)? + 1;
        for layer in 0..layer_count {
            let use_same_config = (program != 0 || layer != 0) && reader.read_bool()?;
            let config = if use_same_config {
                if layer == 0 {
                    return Err(LatmError::InvalidUseSameConfig);
                }
                streams
                    .last()
                    .and_then(|stream: &LatmStreamLayer| stream.config.clone())
                    .ok_or(LatmError::InvalidUseSameConfig)?
            } else {
                parse_latm_audio_specific_config(reader, audio_mux_version)?
            };
            let frame_length_type = reader.read_u8(3)?;
            let fixed_frame_length_bits = match frame_length_type {
                0 => {
                    reader.read_u8(8)?; // latmBufferFullness
                    None
                }
                1 => Some(reader.read_u16(9)? as usize),
                value => return Err(LatmError::UnsupportedFrameLengthType(value)),
            };
            streams.push(LatmStreamLayer {
                program,
                layer,
                config: Some(config),
                frame_length_type,
                fixed_frame_length_bits,
            });
        }
    }
    let other_data_bits = if reader.read_bool()? {
        if audio_mux_version == 1 {
            read_latm_value(reader)? as usize
        } else {
            let mut length = 0usize;
            loop {
                let escaped = reader.read_bool()?;
                let byte = reader.read_u8(8)? as usize;
                length = length
                    .checked_mul(256)
                    .and_then(|value| value.checked_add(byte))
                    .ok_or(LatmError::OtherDataTooLarge)?;
                if !escaped {
                    break;
                }
            }
            length
        }
    } else {
        0
    };
    let crc_check_sum = if reader.read_bool()? {
        Some(reader.read_u8(8)?)
    } else {
        None
    };
    Ok(LatmMuxConfig {
        audio_mux_version,
        all_streams_same_time_framing,
        subframe_count,
        streams,
        other_data_bits,
        crc_check_sum,
    })
}

fn parse_latm_audio_specific_config(
    reader: &mut BitReader<'_>,
    audio_mux_version: u8,
) -> Result<AudioSpecificConfig, LatmError> {
    if audio_mux_version == 0 {
        let config = AudioSpecificConfig::parse_from_reader(reader)?;
        validate_audio_specific_config(&config)?;
        return Ok(config);
    }
    let asc_bits = read_latm_value(reader)? as usize;
    if asc_bits > reader.remaining_bits() {
        return Err(LatmError::InvalidAudioSpecificConfigLength {
            declared_bits: asc_bits,
            remaining_bits: reader.remaining_bits(),
        });
    }
    let config = AudioSpecificConfig::parse(&read_packed_bits(reader, asc_bits)?)?;
    validate_audio_specific_config(&config)?;
    Ok(config)
}

fn read_payload_length_bytes(reader: &mut BitReader<'_>) -> Result<usize, LatmError> {
    let mut length = 0usize;
    loop {
        let chunk = reader.read_u8(8)? as usize;
        length = length
            .checked_add(chunk)
            .ok_or(LatmError::PayloadTooLarge)?;
        if chunk != 255 {
            return Ok(length);
        }
    }
}

fn read_packed_bits(reader: &mut BitReader<'_>, bits: usize) -> Result<Vec<u8>, LatmError> {
    if bits > reader.remaining_bits() {
        return Err(BitError::UnexpectedEof {
            needed_bits: bits,
            remaining_bits: reader.remaining_bits(),
        }
        .into());
    }
    let mut writer = BitWriter::new();
    for _ in 0..bits {
        writer.write_bool(reader.read_bool()?);
    }
    Ok(writer.finish())
}

fn read_latm_value(reader: &mut BitReader<'_>) -> Result<u32, LatmError> {
    let bytes_for_value = reader.read_u8(2)? as usize + 1;
    let mut value = 0u32;
    for _ in 0..bytes_for_value {
        value = (value << 8) | reader.read_u8(8)? as u32;
    }
    Ok(value)
}

fn validate_audio_specific_config(config: &AudioSpecificConfig) -> Result<(), LatmError> {
    if !matches!(config.audio_object_type, 2 | 17 | 20 | 23 | 39 | 42) {
        return Err(LatmError::UnsupportedAudioObjectType(
            config.audio_object_type,
        ));
    }
    if let Some(ga) = config.ga_specific {
        if ga.frame_length_flag && !matches!(config.audio_object_type, 2 | 17 | 20 | 23) {
            return Err(LatmError::UnsupportedFrameLengthFlag);
        }
        if ga.depends_on_core_coder {
            return Err(LatmError::UnsupportedCoreCoderDependency);
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LatmError {
    Bit(BitError),
    Asc(crate::asc::AscError),
    InvalidChannelConfiguration(u8),
    InvalidAudioSpecificConfigLength {
        declared_bits: usize,
        remaining_bits: usize,
    },
    InvalidSamplingFrequencyIndex(u8),
    InvalidSubframeCount(u8),
    InvalidOtherDataLength {
        bits: usize,
        available_bits: usize,
    },
    InvalidFixedFrameLength(usize),
    InvalidPayloadBitLength {
        bits: usize,
        available_bits: usize,
    },
    InvalidUseSameConfig,
    MissingStreamMuxConfig,
    OtherDataTooLarge,
    PayloadTooLarge,
    PayloadCountMismatch {
        expected: usize,
        actual: usize,
    },
    StreamsNotSameTimeFramed,
    UnsupportedAudioMuxVersion(u8),
    UnsupportedAudioMuxVersionA(u8),
    UnsupportedAudioObjectType(u8),
    UnsupportedCoreCoderDependency,
    UnsupportedCrc,
    UnsupportedFrameLengthFlag,
    UnsupportedFrameLengthType(u8),
    UnsupportedErrorProtectionConfig(u8),
    UnsupportedGaExtension,
    UnsupportedOtherData,
    UnsupportedProgramOrLayerLayout,
}

impl From<BitError> for LatmError {
    fn from(value: BitError) -> Self {
        Self::Bit(value)
    }
}
impl From<crate::asc::AscError> for LatmError {
    fn from(value: crate::asc::AscError) -> Self {
        Self::Asc(value)
    }
}
impl fmt::Display for LatmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bit(error) => error.fmt(f),
            Self::Asc(error) => error.fmt(f),
            Self::InvalidChannelConfiguration(value) => {
                write!(f, "invalid LATM channel configuration {value}")
            }
            Self::InvalidAudioSpecificConfigLength {
                declared_bits,
                remaining_bits,
            } => write!(
                f,
                "LATM AudioSpecificConfig declares {declared_bits} bits, only {remaining_bits} remain"
            ),
            Self::InvalidSamplingFrequencyIndex(value) => {
                write!(f, "invalid LATM sampling frequency index {value}")
            }
            Self::InvalidSubframeCount(value) => {
                write!(f, "invalid LATM subframe count {value}")
            }
            Self::InvalidOtherDataLength {
                bits,
                available_bits,
            } => write!(
                f,
                "LATM otherData declares {bits} bits, only {available_bits} are available"
            ),
            Self::InvalidFixedFrameLength(bits) => {
                write!(f, "invalid LATM fixed frame length {bits} bits")
            }
            Self::InvalidPayloadBitLength {
                bits,
                available_bits,
            } => write!(
                f,
                "LATM payload requires {bits} bits, only {available_bits} are available"
            ),
            Self::InvalidUseSameConfig => write!(f, "invalid LATM useSameConfig reference"),
            Self::MissingStreamMuxConfig => write!(f, "LATM useSameStreamMux has no prior config"),
            Self::OtherDataTooLarge => write!(f, "LATM otherData length is too large"),
            Self::PayloadTooLarge => write!(f, "LATM payload length is too large"),
            Self::PayloadCountMismatch { expected, actual } => write!(
                f,
                "LATM expected {expected} subframe payloads, got {actual}"
            ),
            Self::StreamsNotSameTimeFramed => write!(f, "LATM streams must use same time framing"),
            Self::UnsupportedAudioMuxVersion(value) => {
                write!(f, "unsupported LATM AudioMuxVersion {value}")
            }
            Self::UnsupportedAudioMuxVersionA(value) => {
                write!(f, "unsupported LATM AudioMuxVersionA {value}")
            }
            Self::UnsupportedAudioObjectType(value) => {
                write!(f, "unsupported LATM audio object type {value}")
            }
            Self::UnsupportedCoreCoderDependency => {
                write!(f, "LATM core coder dependency is unsupported")
            }
            Self::UnsupportedCrc => write!(f, "LATM CRC is unsupported"),
            Self::UnsupportedFrameLengthFlag => write!(f, "LATM frameLengthFlag is unsupported"),
            Self::UnsupportedFrameLengthType(value) => {
                write!(f, "LATM frameLengthType {value} is unsupported")
            }
            Self::UnsupportedErrorProtectionConfig(value) => {
                write!(f, "unsupported LATM ER epConfig {value}")
            }
            Self::UnsupportedGaExtension => write!(f, "LATM GA extension is unsupported"),
            Self::UnsupportedOtherData => write!(f, "LATM otherData is unsupported"),
            Self::UnsupportedProgramOrLayerLayout => {
                write!(f, "LATM only supports one program and one layer")
            }
        }
    }
}
impl std::error::Error for LatmError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_aac_lc_asc(writer: &mut BitWriter, channel_configuration: u32) {
        writer.write(2, 5);
        writer.write(4, 4);
        writer.write(channel_configuration, 4);
        writer.write_bool(false);
        writer.write_bool(false);
        writer.write_bool(false);
    }

    #[test]
    fn writer_emits_initial_and_same_mux_config_elements() {
        let mut writer = LatmAacLcWriter::new(4, 1).unwrap();
        let first_bytes = writer.write_audio_mux_element(&[0xaa, 0xbb]);
        let first = LatmAudioMuxElement::parse_aac_lc(&first_bytes).unwrap();
        assert!(!first.use_same_stream_mux);
        assert_eq!(first.raw_data_block, [0xaa, 0xbb]);
        let mux = first.mux_config.unwrap();

        let second_bytes = writer.write_audio_mux_element(&vec![0x55; 300]);
        let second =
            LatmAudioMuxElement::parse_aac_lc_with_state(&second_bytes, Some(&mux)).unwrap();
        assert!(second.use_same_stream_mux);
        assert_eq!(second.raw_data_block, vec![0x55; 300]);
    }

    #[test]
    fn generic_writer_roundtrips_subframes_amv_and_periodic_config() {
        let config = AudioSpecificConfig::aac_lc(44_100, 2).unwrap();
        let mut writer = LatmWriter::new(config.clone(), 1, 2, 1).unwrap();
        let first_bytes = writer
            .write_audio_mux_element(&[vec![0xaa], vec![0xbb, 0xcc]])
            .unwrap();
        let first = LatmAudioMuxElement::parse_aac_lc(&first_bytes).unwrap();
        assert!(!first.use_same_stream_mux);
        let parsed_config = first.config.as_ref().unwrap();
        assert_eq!(parsed_config.audio_object_type, config.audio_object_type);
        assert_eq!(parsed_config.sampling_frequency, config.sampling_frequency);
        assert_eq!(
            parsed_config.channel_configuration,
            config.channel_configuration
        );
        assert_eq!(first.mux_config.as_ref().unwrap().audio_mux_version, 1);
        assert_eq!(first.payloads.len(), 2);
        assert_eq!(first.payloads[1].data, [0xbb, 0xcc]);

        let second = writer.write_audio_mux_element(&[vec![1], vec![2]]).unwrap();
        // A period of one repeats StreamMuxConfig on every LATM frame.
        assert!(
            !LatmAudioMuxElement::parse_aac_lc(&second)
                .unwrap()
                .use_same_stream_mux
        );
        assert_eq!(
            writer.write_audio_mux_element(&[vec![1]]),
            Err(LatmError::PayloadCountMismatch {
                expected: 2,
                actual: 1
            })
        );
    }

    #[test]
    fn generic_writer_signals_crc_and_repeats_config_after_change() {
        let config = AudioSpecificConfig::aac_lc(48_000, 1).unwrap();
        let mut writer = LatmWriter::new(config, 0, 1, 0).unwrap();
        writer.set_crc_check_sum(Some(0x5a));
        let first = writer.write_audio_mux_element(&[vec![0xaa]]).unwrap();
        let parsed = LatmAudioMuxElement::parse_aac_lc(&first).unwrap();
        assert!(!parsed.use_same_stream_mux);
        assert_eq!(parsed.crc_check_sum, Some(0x5a));
        let first_config = parsed.mux_config.unwrap();

        let same = writer.write_audio_mux_element(&[vec![0xbb]]).unwrap();
        let parsed =
            LatmAudioMuxElement::parse_aac_lc_with_state(&same, Some(&first_config)).unwrap();
        assert!(parsed.use_same_stream_mux);
        assert_eq!(parsed.crc_check_sum, Some(0x5a));

        writer.set_crc_check_sum(Some(0xa5));
        let changed = writer.write_audio_mux_element(&[vec![0xcc]]).unwrap();
        let parsed = LatmAudioMuxElement::parse_aac_lc(&changed).unwrap();
        assert!(!parsed.use_same_stream_mux);
        assert_eq!(parsed.crc_check_sum, Some(0xa5));

        writer.set_crc_check_sum(None);
        let removed = writer.write_audio_mux_element(&[vec![0xdd]]).unwrap();
        let parsed = LatmAudioMuxElement::parse_aac_lc(&removed).unwrap();
        assert_eq!(parsed.crc_check_sum, None);
    }

    #[test]
    fn generic_writer_roundtrips_other_data_for_both_mux_versions() {
        for audio_mux_version in [0, 1] {
            let config = AudioSpecificConfig::aac_lc(48_000, 1).unwrap();
            let mut writer = LatmWriter::new(config, audio_mux_version, 1, 0).unwrap();
            writer
                .set_other_data(vec![0b1010_1100, 0b1100_0000], 10)
                .unwrap();
            let first = writer.write_audio_mux_element(&[vec![0xaa]]).unwrap();
            let parsed = LatmAudioMuxElement::parse_aac_lc(&first).unwrap();
            assert_eq!(parsed.other_data_bits, 10);
            assert_eq!(parsed.other_data, [0b1010_1100, 0b1100_0000]);
            let mux = parsed.mux_config.unwrap();

            // Changing only the contents keeps useSameStreamMux valid.
            writer
                .set_other_data(vec![0b0101_0000, 0b1100_0000], 10)
                .unwrap();
            let same = writer.write_audio_mux_element(&[vec![0xbb]]).unwrap();
            let parsed = LatmAudioMuxElement::parse_aac_lc_with_state(&same, Some(&mux)).unwrap();
            assert!(parsed.use_same_stream_mux);
            assert_eq!(parsed.other_data, [0b0101_0000, 0b1100_0000]);

            // A length change is part of StreamMuxConfig and forces a repeat.
            writer.set_other_data(vec![0x80], 1).unwrap();
            let changed = writer.write_audio_mux_element(&[vec![0xcc]]).unwrap();
            let parsed = LatmAudioMuxElement::parse_aac_lc(&changed).unwrap();
            assert!(!parsed.use_same_stream_mux);
            assert_eq!(parsed.other_data_bits, 1);
            assert_eq!(parsed.other_data, [0x80]);
        }
        let mut writer =
            LatmWriter::new(AudioSpecificConfig::aac_lc(48_000, 1).unwrap(), 0, 1, 0).unwrap();
        assert_eq!(
            writer.set_other_data(vec![0], 9),
            Err(LatmError::InvalidOtherDataLength {
                bits: 9,
                available_bits: 8
            })
        );
    }

    #[test]
    fn generic_writer_roundtrips_fixed_non_byte_payloads() {
        let config = AudioSpecificConfig::aac_lc(48_000, 1).unwrap();
        let mut writer = LatmWriter::new(config, 0, 2, 0).unwrap();
        writer.set_fixed_frame_length_bits(Some(13)).unwrap();
        let encoded = writer
            .write_audio_mux_element(&[
                vec![0b1010_1010, 0b1111_1000],
                vec![0b0101_0101, 0b0000_0000],
            ])
            .unwrap();
        let parsed = LatmAudioMuxElement::parse_aac_lc(&encoded).unwrap();
        assert_eq!(
            parsed.mux_config.as_ref().unwrap().streams[0].frame_length_type,
            1
        );
        assert_eq!(
            parsed.mux_config.as_ref().unwrap().streams[0].fixed_frame_length_bits,
            Some(13)
        );
        assert_eq!(parsed.payloads[0].bits, 13);
        assert_eq!(parsed.payloads[0].data, [0b1010_1010, 0b1111_1000]);
        assert_eq!(parsed.payloads[1].data, [0b0101_0101, 0]);

        let mux = parsed.mux_config.unwrap();
        let same = writer
            .write_audio_mux_element(&[vec![0xff, 0xf8], vec![0, 0]])
            .unwrap();
        let parsed = LatmAudioMuxElement::parse_aac_lc_with_state(&same, Some(&mux)).unwrap();
        assert!(parsed.use_same_stream_mux);
        assert_eq!(parsed.payloads[0].data, [0xff, 0xf8]);

        writer.set_fixed_frame_length_bits(None).unwrap();
        let variable = writer.write_audio_mux_element(&[vec![1], vec![2]]).unwrap();
        let parsed = LatmAudioMuxElement::parse_aac_lc(&variable).unwrap();
        assert_eq!(parsed.mux_config.unwrap().streams[0].frame_length_type, 0);

        assert_eq!(
            writer.set_fixed_frame_length_bits(Some(0)),
            Err(LatmError::InvalidFixedFrameLength(0))
        );
        assert_eq!(
            writer.set_fixed_frame_length_bits(Some(512)),
            Err(LatmError::InvalidFixedFrameLength(512))
        );
        writer.set_fixed_frame_length_bits(Some(9)).unwrap();
        assert_eq!(
            writer.write_audio_mux_element(&[vec![0], vec![0, 0]]),
            Err(LatmError::InvalidPayloadBitLength {
                bits: 9,
                available_bits: 8
            })
        );
    }

    #[test]
    fn parses_single_stream_aac_lc_audio_mux_element() {
        let payload = [0xaa, 0xbb];
        let mut writer = BitWriter::new();
        writer.write_bool(false); // useSameStreamMux
        writer.write_bool(false); // audioMuxVersion
        writer.write_bool(true); // allStreamsSameTimeFraming
        writer.write(0, 6); // numSubFrames - 1
        writer.write(0, 4); // numProgram
        writer.write(0, 3); // numLayer
        writer.write(2, 5); // AAC-LC
        writer.write(4, 4); // 44100
        writer.write(1, 4); // mono
        writer.write_bool(false); // frameLengthFlag
        writer.write_bool(false); // dependsOnCoreCoder
        writer.write_bool(false); // extensionFlag
        writer.write(0, 3); // frameLengthType
        writer.write(0xff, 8); // latmBufferFullness
        writer.write_bool(false); // otherDataPresent
        writer.write_bool(false); // crcCheckPresent
        writer.write(payload.len() as u32, 8); // PayloadLengthInfo
        for byte in payload {
            writer.write(byte as u32, 8);
        }
        let parsed = LatmAudioMuxElement::parse_aac_lc(&writer.finish()).unwrap();
        assert_eq!(parsed.config.unwrap().audio_object_type, 2);
        assert_eq!(parsed.raw_data_block, payload);
    }

    #[test]
    fn parses_er_aac_lc_inline_config_and_ep_config() {
        let mut writer = BitWriter::new();
        writer.write_bool(false);
        writer.write_bool(false); // audioMuxVersion 0
        writer.write_bool(true);
        writer.write(0, 6);
        writer.write(0, 4);
        writer.write(0, 3);
        writer.write(17, 5); // ER AAC-LC
        writer.write(4, 4);
        writer.write(1, 4);
        writer.write_bool(false); // 1024 samples
        writer.write_bool(false); // no core coder
        writer.write_bool(true); // extension flags present
        writer.write_bool(false); // section resilience
        writer.write_bool(false); // scalefactor resilience
        writer.write_bool(false); // spectral resilience
        writer.write_bool(false); // extensionFlag3
        writer.write(1, 2); // epConfig
        writer.write(0, 3); // frameLengthType
        writer.write(0xff, 8);
        writer.write_bool(false);
        writer.write_bool(false);
        writer.write(1, 8);
        writer.write(0xaa, 8);
        let element = LatmAudioMuxElement::parse_aac_lc(&writer.finish()).unwrap();
        let config = element.config.unwrap();
        assert_eq!(config.audio_object_type, 17);
        assert_eq!(config.error_protection_config, Some(1));
        assert_eq!(element.raw_data_block, [0xaa]);
    }

    #[test]
    fn parses_er_aac_ld_480_inline_config() {
        let mut writer = BitWriter::new();
        writer.write_bool(false); // useSameStreamMux
        writer.write_bool(false); // audioMuxVersion 0
        writer.write_bool(true);
        writer.write(0, 6);
        writer.write(0, 4);
        writer.write(0, 3);
        writer.write(23, 5); // ER AAC-LD
        writer.write(4, 4);
        writer.write(1, 4);
        writer.write_bool(true); // 480 samples
        writer.write_bool(false);
        writer.write_bool(false); // no GA extension
        writer.write(0, 2); // epConfig
        writer.write(0, 3); // frameLengthType
        writer.write(0xff, 8);
        writer.write_bool(false);
        writer.write_bool(false);
        writer.write(1, 8);
        writer.write(0x55, 8);
        let element = LatmAudioMuxElement::parse_aac_lc(&writer.finish()).unwrap();
        let config = element.config.unwrap();
        assert_eq!(config.audio_object_type, 23);
        assert!(config.ga_specific.unwrap().frame_length_flag);
        assert_eq!(element.raw_data_block, [0x55]);
    }

    #[test]
    fn parses_er_aac_eld_inline_config() {
        let mut writer = BitWriter::new();
        writer.write_bool(false); // useSameStreamMux
        writer.write_bool(false); // audioMuxVersion 0
        writer.write_bool(true); // allStreamsSameTimeFraming
        writer.write(0, 6); // one subframe
        writer.write(0, 4); // one program
        writer.write(0, 3); // one layer
        writer.write(31, 5); // escaped AOT
        writer.write(7, 6); // 32 + 7 = ER AAC-ELD (39)
        writer.write(4, 4); // 44100 Hz
        writer.write(1, 4); // mono
        writer.write_bool(false); // 512-sample ELD frame
        writer.write_bool(false); // section resilience
        writer.write_bool(false); // scalefactor resilience
        writer.write_bool(false); // spectral resilience
        writer.write_bool(false); // LD-SBR absent
        writer.write(0, 4); // ELD extension terminator
        writer.write(0, 2); // epConfig
        writer.write(0, 3); // frameLengthType
        writer.write(0xff, 8); // latmBufferFullness
        writer.write_bool(false); // otherDataPresent
        writer.write_bool(false); // crcCheckPresent
        writer.write(1, 8); // one payload byte
        writer.write(0x55, 8);

        let element = LatmAudioMuxElement::parse_aac_lc(&writer.finish()).unwrap();
        let config = element.config.unwrap();
        assert_eq!(config.audio_object_type, 39);
        assert_eq!(config.error_protection_config, Some(0));
        assert_eq!(config.eld_specific.unwrap().frame_length_flag, false);
        assert_eq!(element.raw_data_block, [0x55]);
    }

    #[test]
    fn parses_multiple_programs_layers_subframes_and_fixed_lengths() {
        let mut writer = BitWriter::new();
        writer.write_bool(false); // useSameStreamMux
        writer.write_bool(false); // audioMuxVersion
        writer.write_bool(true); // allStreamsSameTimeFraming
        writer.write(1, 6); // two subframes
        writer.write(1, 4); // two programs
        writer.write(1, 3); // program 0: two layers
        write_aac_lc_asc(&mut writer, 1);
        writer.write(0, 3); // frameLengthType 0
        writer.write(0xff, 8);
        writer.write_bool(true); // layer 1 reuses layer 0 config
        writer.write(0, 3);
        writer.write(0xff, 8);
        writer.write(0, 3); // program 1: one layer
        writer.write_bool(false); // explicit config
        write_aac_lc_asc(&mut writer, 2);
        writer.write(1, 3); // fixed frame length
        writer.write(12, 9);
        writer.write_bool(false); // otherDataPresent
        writer.write_bool(false); // crcCheckPresent
        for subframe in 0..2u32 {
            writer.write(1, 8);
            writer.write(2, 8);
            writer.write(0xa0 + subframe, 8);
            writer.write(0xb0 + subframe, 8);
            writer.write(0xc0 + subframe, 8);
            writer.write(0xd00 + subframe, 12);
        }
        let element = LatmAudioMuxElement::parse_aac_lc(&writer.finish()).unwrap();
        let mux = element.mux_config.as_ref().unwrap();
        assert_eq!(mux.subframe_count, 2);
        assert_eq!(mux.streams.len(), 3);
        assert_eq!(mux.streams[1].config, mux.streams[0].config);
        assert_eq!(mux.streams[2].fixed_frame_length_bits, Some(12));
        assert_eq!(element.payloads.len(), 6);
        assert_eq!(element.payloads[0].data, [0xa0]);
        assert_eq!(element.payloads[2].bits, 12);
        assert_eq!(element.payloads[5].subframe, 1);
        assert_eq!(element.payloads[5].program, 1);
    }

    #[test]
    fn reuses_complete_multiple_stream_mux_config() {
        let mut config_writer = BitWriter::new();
        config_writer.write_bool(false);
        config_writer.write_bool(false);
        config_writer.write_bool(true);
        config_writer.write(0, 6);
        config_writer.write(0, 4);
        config_writer.write(1, 3); // two layers
        write_aac_lc_asc(&mut config_writer, 1);
        config_writer.write(0, 3);
        config_writer.write(0, 8);
        config_writer.write_bool(true);
        config_writer.write(1, 3);
        config_writer.write(8, 9);
        config_writer.write_bool(false);
        config_writer.write_bool(false);
        config_writer.write(1, 8);
        config_writer.write(0x11, 8);
        config_writer.write(0x22, 8);
        let first = LatmAudioMuxElement::parse_aac_lc(&config_writer.finish()).unwrap();
        let mux = first.mux_config.as_ref().unwrap();

        let mut same_writer = BitWriter::new();
        same_writer.write_bool(true);
        same_writer.write(1, 8);
        same_writer.write(0x33, 8);
        same_writer.write(0x44, 8);
        let second =
            LatmAudioMuxElement::parse_aac_lc_with_state(&same_writer.finish(), Some(mux)).unwrap();
        assert!(second.use_same_stream_mux);
        assert!(second.mux_config.is_none());
        assert_eq!(second.payloads[0].data, [0x33]);
        assert_eq!(second.payloads[1].data, [0x44]);
    }

    #[test]
    fn parses_other_data_and_stream_mux_crc() {
        let payload = [0xaa];
        let other_data = 0b1010_1100u8;
        let mut writer = BitWriter::new();
        writer.write_bool(false); // useSameStreamMux
        writer.write_bool(false); // audioMuxVersion
        writer.write_bool(true); // allStreamsSameTimeFraming
        writer.write(0, 6);
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
        writer.write_bool(true); // otherDataPresent
        writer.write_bool(false); // otherDataLenEsc
        writer.write(8, 8); // otherDataLenBits
        writer.write_bool(true); // crcCheckPresent
        writer.write(0x5a, 8);
        writer.write(payload.len() as u32, 8);
        writer.write(payload[0] as u32, 8);
        writer.write(other_data as u32, 8);

        let parsed = LatmAudioMuxElement::parse_aac_lc(&writer.finish()).unwrap();
        assert_eq!(parsed.raw_data_block, payload);
        assert_eq!(parsed.other_data_bits, 8);
        assert_eq!(parsed.other_data, [other_data]);
        assert_eq!(parsed.crc_check_sum, Some(0x5a));
    }

    #[test]
    fn reuses_other_data_length_for_same_stream_mux() {
        let mut writer = BitWriter::new();
        writer.write_bool(true); // useSameStreamMux
        writer.write(1, 8); // payload length
        writer.write(0x11, 8);
        writer.write(0xab, 8); // retained otherData
        let parsed =
            LatmAudioMuxElement::parse_aac_lc_with_mux_state(&writer.finish(), 8, Some(0x5a))
                .unwrap();
        assert_eq!(parsed.raw_data_block, [0x11]);
        assert_eq!(parsed.other_data, [0xab]);
        assert_eq!(parsed.crc_check_sum, Some(0x5a));
    }

    #[test]
    fn parses_audio_mux_version_one_length_delimited_asc() {
        let asc = AudioSpecificConfig::aac_lc(44_100, 1)
            .unwrap()
            .to_bytes()
            .unwrap();
        let mut writer = BitWriter::new();
        writer.write_bool(false); // useSameStreamMux
        writer.write_bool(true); // audioMuxVersion
        writer.write_bool(false); // audioMuxVersionA
        writer.write(0, 2); // taraBufferFullness uses one byte
        writer.write(0xff, 8);
        writer.write_bool(true); // allStreamsSameTimeFraming
        writer.write(0, 6);
        writer.write(0, 4);
        writer.write(0, 3);
        writer.write(0, 2); // ascLen uses one byte
        writer.write((asc.len() * 8) as u32, 8);
        for byte in &asc {
            writer.write(*byte as u32, 8);
        }
        writer.write(0, 3); // frameLengthType
        writer.write(0xff, 8);
        writer.write_bool(true); // otherDataPresent
        writer.write(0, 2); // otherDataLenBits uses one byte
        writer.write(8, 8);
        writer.write_bool(false); // crcCheckPresent
        writer.write(1, 8); // payload length
        writer.write(0x11, 8);
        writer.write(0xab, 8);

        let parsed = LatmAudioMuxElement::parse_aac_lc(&writer.finish()).unwrap();
        let config = parsed.config.unwrap();
        assert_eq!(config.audio_object_type, 2);
        assert_eq!(config.sampling_frequency, 44_100);
        assert_eq!(config.channel_configuration, 1);
        assert_eq!(parsed.raw_data_block, [0x11]);
        assert_eq!(parsed.other_data, [0xab]);
    }

    #[test]
    fn parses_audio_mux_version_one_usac_config() {
        let asc = AudioSpecificConfig {
            audio_object_type: 42,
            sampling_frequency_index: 3,
            sampling_frequency: 48_000,
            channel_configuration: 1,
            extension: None,
            ga_specific: None,
            eld_specific: None,
            usac_config: Some(crate::asc::UsacConfig {
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
            }),
            error_protection_config: None,
            program_config: None,
            bits_read: 0,
        }
        .to_bytes()
        .unwrap();
        let mut writer = BitWriter::new();
        writer.write_bool(false); // useSameStreamMux
        writer.write_bool(true); // audioMuxVersion
        writer.write_bool(false); // audioMuxVersionA
        writer.write(0, 2); // taraBufferFullness length
        writer.write(0, 8);
        writer.write_bool(true); // allStreamsSameTimeFraming
        writer.write(0, 6); // one subframe
        writer.write(0, 4); // one program
        writer.write(0, 3); // one layer
        writer.write(0, 2); // ascLen length
        writer.write((asc.len() * 8) as u32, 8);
        for byte in asc {
            writer.write(byte as u32, 8);
        }
        writer.write(0, 3); // frameLengthType
        writer.write(0xff, 8);
        writer.write_bool(false); // otherDataPresent
        writer.write_bool(false); // crcCheckPresent
        writer.write(1, 8); // payload bytes
        writer.write(0, 8);

        let parsed = LatmAudioMuxElement::parse_aac_lc(&writer.finish()).unwrap();
        let config = parsed.config.unwrap();
        assert_eq!(config.audio_object_type, 42);
        assert_eq!(config.usac_config.unwrap().core_frame_length, 1024);
        assert_eq!(parsed.raw_data_block, [0]);
    }

    #[test]
    fn writer_validates_sampling_and_channel_configuration() {
        assert_eq!(
            LatmAacLcWriter::new(13, 1),
            Err(LatmError::InvalidSamplingFrequencyIndex(13))
        );
        for channels in [0, 8] {
            assert_eq!(
                LatmAacLcWriter::new(4, channels),
                Err(LatmError::InvalidChannelConfiguration(channels))
            );
        }
        for configuration in [11, 12, 14] {
            let mut writer = LatmAacLcWriter::new(4, configuration).unwrap();
            let bytes = writer.write_audio_mux_element(&[0]);
            let parsed = LatmAudioMuxElement::parse_aac_lc(&bytes).unwrap();
            assert_eq!(parsed.config.unwrap().channel_configuration, configuration);
        }
    }

    #[test]
    fn stateful_parser_validates_mux_state_and_frame_length_types() {
        assert_eq!(
            LatmAudioMuxElement::parse_aac_lc(&[0x80]),
            Err(LatmError::MissingStreamMuxConfig)
        );
        let base = LatmMuxConfig {
            audio_mux_version: 0,
            all_streams_same_time_framing: false,
            subframe_count: 1,
            streams: vec![LatmStreamLayer {
                program: 0,
                layer: 0,
                config: None,
                frame_length_type: 0,
                fixed_frame_length_bits: None,
            }],
            other_data_bits: 0,
            crc_check_sum: None,
        };
        assert_eq!(
            LatmAudioMuxElement::parse_aac_lc_with_state(&[0x80], Some(&base)),
            Err(LatmError::StreamsNotSameTimeFramed)
        );
        let mut unsupported = base.clone();
        unsupported.all_streams_same_time_framing = true;
        unsupported.streams[0].frame_length_type = 2;
        assert_eq!(
            LatmAudioMuxElement::parse_aac_lc_with_state(&[0x80], Some(&unsupported)),
            Err(LatmError::UnsupportedFrameLengthType(2))
        );
        unsupported.streams[0].frame_length_type = 1;
        assert_eq!(
            LatmAudioMuxElement::parse_aac_lc_with_state(&[0x80], Some(&unsupported)),
            Err(LatmError::UnsupportedFrameLengthType(1))
        );
    }

    #[test]
    fn rejects_audio_mux_version_a_and_length_delimited_asc_overrun() {
        let mut writer = BitWriter::new();
        writer.write_bool(false);
        writer.write_bool(true);
        writer.write_bool(true);
        assert_eq!(
            LatmAudioMuxElement::parse_aac_lc(&writer.finish()),
            Err(LatmError::UnsupportedAudioMuxVersionA(1))
        );

        let mut writer = BitWriter::new();
        writer.write_bool(false);
        writer.write_bool(true);
        writer.write_bool(false);
        writer.write(0, 2);
        writer.write(0, 8);
        writer.write_bool(true);
        writer.write(0, 6);
        writer.write(0, 4);
        writer.write(0, 3);
        writer.write(0, 2);
        writer.write(100, 8);
        assert!(matches!(
            LatmAudioMuxElement::parse_aac_lc(&writer.finish()),
            Err(LatmError::InvalidAudioSpecificConfigLength { .. })
        ));
    }

    #[test]
    fn packed_bit_and_latm_value_helpers_cover_boundaries() {
        assert_eq!(
            read_packed_bits(&mut BitReader::new(&[0b1010_0000]), 4).unwrap(),
            [0b1010_0000]
        );
        assert!(matches!(
            read_packed_bits(&mut BitReader::new(&[]), 1),
            Err(LatmError::Bit(BitError::UnexpectedEof { .. }))
        ));
        let mut writer = BitWriter::new();
        writer.write(3, 2);
        writer.write(0x1234_5678, 32);
        assert_eq!(
            read_latm_value(&mut BitReader::new(&writer.finish())).unwrap(),
            0x1234_5678
        );
        let mut writer = BitWriter::new();
        writer.write(255, 8);
        writer.write(255, 8);
        writer.write(2, 8);
        assert_eq!(
            read_payload_length_bytes(&mut BitReader::new(&writer.finish())).unwrap(),
            512
        );
    }

    #[test]
    fn validates_audio_specific_config_profile_constraints() {
        let mut config = AudioSpecificConfig::aac_lc(44_100, 1).unwrap();
        config.audio_object_type = 1;
        assert_eq!(
            validate_audio_specific_config(&config),
            Err(LatmError::UnsupportedAudioObjectType(1))
        );
        config = AudioSpecificConfig::aac_lc(44_100, 1).unwrap();
        config.ga_specific.as_mut().unwrap().frame_length_flag = true;
        assert_eq!(validate_audio_specific_config(&config), Ok(()));
        config.audio_object_type = 39;
        assert_eq!(
            validate_audio_specific_config(&config),
            Err(LatmError::UnsupportedFrameLengthFlag)
        );
        config.audio_object_type = 2;
        config.ga_specific.as_mut().unwrap().frame_length_flag = false;
        config.ga_specific.as_mut().unwrap().depends_on_core_coder = true;
        assert_eq!(
            validate_audio_specific_config(&config),
            Err(LatmError::UnsupportedCoreCoderDependency)
        );
    }

    #[test]
    fn error_conversions_and_all_messages_are_nonempty() {
        let bit = BitError::UnexpectedEof {
            needed_bits: 1,
            remaining_bits: 0,
        };
        assert_eq!(LatmError::from(bit.clone()), LatmError::Bit(bit.clone()));
        let asc = crate::asc::AscError::InvalidSamplingFrequencyIndex(15);
        assert_eq!(LatmError::from(asc.clone()), LatmError::Asc(asc.clone()));
        assert_eq!(LatmError::Bit(bit.clone()).to_string(), bit.to_string());
        assert_eq!(LatmError::Asc(asc.clone()).to_string(), asc.to_string());
        let errors = [
            LatmError::InvalidChannelConfiguration(0),
            LatmError::InvalidAudioSpecificConfigLength {
                declared_bits: 8,
                remaining_bits: 0,
            },
            LatmError::InvalidSamplingFrequencyIndex(13),
            LatmError::InvalidUseSameConfig,
            LatmError::MissingStreamMuxConfig,
            LatmError::OtherDataTooLarge,
            LatmError::PayloadTooLarge,
            LatmError::StreamsNotSameTimeFramed,
            LatmError::UnsupportedAudioMuxVersion(2),
            LatmError::UnsupportedAudioMuxVersionA(1),
            LatmError::UnsupportedAudioObjectType(1),
            LatmError::UnsupportedCoreCoderDependency,
            LatmError::UnsupportedCrc,
            LatmError::UnsupportedFrameLengthFlag,
            LatmError::UnsupportedFrameLengthType(2),
            LatmError::UnsupportedErrorProtectionConfig(2),
            LatmError::UnsupportedGaExtension,
            LatmError::UnsupportedOtherData,
            LatmError::UnsupportedProgramOrLayerLayout,
        ];
        assert!(errors.iter().all(|error| !error.to_string().is_empty()));
    }

    #[test]
    fn stream_mux_config_rejects_invalid_reuse_and_frame_length_type() {
        let mut invalid_reuse = BitWriter::new();
        invalid_reuse.write_bool(false); // audioMuxVersion
        invalid_reuse.write_bool(true); // allStreamsSameTimeFraming
        invalid_reuse.write(0, 6); // one subframe
        invalid_reuse.write(1, 4); // two programs
        invalid_reuse.write(0, 3); // program 0: one layer
        write_aac_lc_asc(&mut invalid_reuse, 1);
        invalid_reuse.write(0, 3); // frameLengthType 0
        invalid_reuse.write(0, 8); // latmBufferFullness
        invalid_reuse.write(0, 3); // program 1: one layer
        invalid_reuse.write_bool(true); // illegal reuse at layer zero
        assert_eq!(
            parse_stream_mux_config_aac_lc(&mut BitReader::new(&invalid_reuse.finish())),
            Err(LatmError::InvalidUseSameConfig)
        );

        let mut unsupported_length = BitWriter::new();
        unsupported_length.write_bool(false); // audioMuxVersion
        unsupported_length.write_bool(true); // allStreamsSameTimeFraming
        unsupported_length.write(0, 6); // one subframe
        unsupported_length.write(0, 4); // one program
        unsupported_length.write(0, 3); // one layer
        write_aac_lc_asc(&mut unsupported_length, 1);
        unsupported_length.write(2, 3); // unsupported frameLengthType
        assert_eq!(
            parse_stream_mux_config_aac_lc(&mut BitReader::new(&unsupported_length.finish())),
            Err(LatmError::UnsupportedFrameLengthType(2))
        );

        let mut oversized_other_data = BitWriter::new();
        oversized_other_data.write_bool(false); // audioMuxVersion
        oversized_other_data.write_bool(true); // allStreamsSameTimeFraming
        oversized_other_data.write(0, 6);
        oversized_other_data.write(0, 4);
        oversized_other_data.write(0, 3);
        write_aac_lc_asc(&mut oversized_other_data, 1);
        oversized_other_data.write(0, 3);
        oversized_other_data.write(0, 8);
        oversized_other_data.write_bool(true); // otherDataPresent
        for _ in 0..8 {
            oversized_other_data.write_bool(true);
            oversized_other_data.write(0xff, 8);
        }
        oversized_other_data.write_bool(false);
        oversized_other_data.write(0xff, 8);
        assert_eq!(
            parse_stream_mux_config_aac_lc(&mut BitReader::new(&oversized_other_data.finish())),
            Err(LatmError::OtherDataTooLarge)
        );
    }
}
