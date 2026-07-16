//! Pure Rust ADIF transport-header parsing.

use std::fmt;

use crate::asc::{AscError, ProgramConfig, ProgramElement};
use crate::bits::{BitError, BitReader, BitWriter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdifHeader {
    pub copyright_id: Option<[u8; 9]>,
    pub original_copy: bool,
    pub home: bool,
    pub variable_bit_rate: bool,
    pub bitrate: u32,
    pub program_configs: Vec<ProgramConfig>,
    pub bits_read: usize,
}

impl AdifHeader {
    pub fn aac_lc_mono(sampling_frequency_index: u8, bitrate: u32) -> Result<Self, AdifError> {
        if sampling_frequency_index >= 13 || bitrate >= (1 << 23) {
            return Err(AdifError::InvalidConfiguration);
        }
        Ok(Self {
            copyright_id: None,
            original_copy: false,
            home: false,
            variable_bit_rate: false,
            bitrate,
            program_configs: vec![ProgramConfig {
                element_instance_tag: 0,
                profile: 1,
                sampling_frequency_index,
                front: vec![ProgramElement {
                    is_cpe: false,
                    tag_select: 0,
                }],
                num_channels: 1,
                num_effective_channels: 1,
                ..ProgramConfig::default()
            }],
            bits_read: 0,
        })
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, AdifError> {
        if self.bitrate >= (1 << 23)
            || self.program_configs.is_empty()
            || self.program_configs.len() > 16
        {
            return Err(AdifError::InvalidConfiguration);
        }
        let mut writer = BitWriter::new();
        for &byte in b"ADIF" {
            writer.write(byte as u32, 8);
        }
        writer.write_bool(self.copyright_id.is_some());
        if let Some(id) = self.copyright_id {
            for byte in id {
                writer.write(byte as u32, 8);
            }
        }
        writer.write_bool(self.original_copy);
        writer.write_bool(self.home);
        writer.write_bool(self.variable_bit_rate);
        writer.write(self.bitrate, 23);
        writer.write((self.program_configs.len() - 1) as u32, 4);
        for config in &self.program_configs {
            if !self.variable_bit_rate {
                writer.write(0, 20);
            }
            config.write_to_writer(&mut writer)?;
        }
        writer.byte_align();
        Ok(writer.finish())
    }

    pub fn parse(input: &[u8]) -> Result<Self, AdifError> {
        let mut reader = BitReader::new(input);
        let start = reader.bits_read();
        if [
            reader.read_u8(8)?,
            reader.read_u8(8)?,
            reader.read_u8(8)?,
            reader.read_u8(8)?,
        ] != *b"ADIF"
        {
            return Err(AdifError::InvalidSignature);
        }
        let copyright_id = if reader.read_bool()? {
            let mut id = [0; 9];
            for byte in &mut id {
                *byte = reader.read_u8(8)?;
            }
            Some(id)
        } else {
            None
        };
        let original_copy = reader.read_bool()?;
        let home = reader.read_bool()?;
        let variable_bit_rate = reader.read_bool()?;
        let bitrate = reader.read(23)?;
        let count = reader.read_u8(4)? as usize + 1;
        let mut program_configs = Vec::with_capacity(count);
        for _ in 0..count {
            if !variable_bit_rate {
                let _adif_buffer_fullness = reader.read(20)?;
            }
            program_configs.push(ProgramConfig::parse_from_reader(&mut reader)?);
        }
        reader.byte_align();
        Ok(Self {
            copyright_id,
            original_copy,
            home,
            variable_bit_rate,
            bitrate,
            program_configs,
            bits_read: reader.bits_read() - start,
        })
    }

    pub fn last_program_config(&self) -> Option<&ProgramConfig> {
        self.program_configs.last()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdifError {
    Bit(BitError),
    Asc(AscError),
    InvalidSignature,
    InvalidConfiguration,
}

impl From<BitError> for AdifError {
    fn from(value: BitError) -> Self {
        Self::Bit(value)
    }
}

impl From<AscError> for AdifError {
    fn from(value: AscError) -> Self {
        Self::Asc(value)
    }
}

impl fmt::Display for AdifError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bit(error) => error.fmt(f),
            Self::Asc(error) => error.fmt(f),
            Self::InvalidSignature => write!(f, "invalid ADIF signature"),
            Self::InvalidConfiguration => write!(f, "invalid ADIF encoder configuration"),
        }
    }
}

impl std::error::Error for AdifError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asc::{ProgramConfig, ProgramElement};
    use crate::bits::BitWriter;

    #[test]
    fn parses_constant_rate_adif_header_and_pce() {
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
        writer.write_bool(false); // copyright_id_present
        writer.write_bool(true); // original_copy
        writer.write_bool(false); // home
        writer.write_bool(false); // constant bitrate
        writer.write(128_000, 23);
        writer.write(0, 4); // one PCE
        writer.write(0, 20); // adif_buffer_fullness
        pce.write_to_writer(&mut writer).unwrap();
        let bytes = writer.finish();

        let header = AdifHeader::parse(&bytes).unwrap();
        assert_eq!(header.bitrate, 128_000);
        assert!(!header.variable_bit_rate);
        assert!(header.original_copy);
        assert_eq!(header.last_program_config(), Some(&pce));
        assert_eq!(header.bits_read % 8, 0);
    }

    #[test]
    fn writes_mono_aac_lc_adif_header() {
        let header = AdifHeader::aac_lc_mono(4, 128_000).unwrap();
        let bytes = header.to_bytes().unwrap();
        let parsed = AdifHeader::parse(&bytes).unwrap();
        assert_eq!(parsed.bitrate, 128_000);
        assert_eq!(parsed.program_configs.len(), 1);
        let pce = parsed.last_program_config().unwrap();
        assert_eq!(pce.profile, 1);
        assert_eq!(pce.sampling_frequency_index, 4);
        assert_eq!(pce.num_channels, 1);
        assert_eq!(pce.front[0].tag_select, 0);
    }

    #[test]
    fn roundtrips_variable_rate_copyright_and_multiple_programs() {
        let mut header = AdifHeader::aac_lc_mono(4, 192_000).unwrap();
        header.copyright_id = Some(*b"123456789");
        header.original_copy = true;
        header.home = true;
        header.variable_bit_rate = true;
        header
            .program_configs
            .push(header.program_configs[0].clone());
        header.program_configs[1].element_instance_tag = 1;
        let parsed = AdifHeader::parse(&header.to_bytes().unwrap()).unwrap();
        assert_eq!(parsed.copyright_id, Some(*b"123456789"));
        assert!(parsed.original_copy && parsed.home && parsed.variable_bit_rate);
        assert_eq!(parsed.program_configs.len(), 2);
        assert_eq!(
            parsed.last_program_config().unwrap().element_instance_tag,
            1
        );
    }

    #[test]
    fn rejects_invalid_constructor_and_writer_configurations() {
        assert_eq!(
            AdifHeader::aac_lc_mono(13, 1),
            Err(AdifError::InvalidConfiguration)
        );
        assert_eq!(
            AdifHeader::aac_lc_mono(4, 1 << 23),
            Err(AdifError::InvalidConfiguration)
        );
        let mut header = AdifHeader::aac_lc_mono(4, 1).unwrap();
        header.program_configs.clear();
        assert_eq!(header.to_bytes(), Err(AdifError::InvalidConfiguration));
        header = AdifHeader::aac_lc_mono(4, 1).unwrap();
        header.program_configs = vec![header.program_configs[0].clone(); 17];
        assert_eq!(header.to_bytes(), Err(AdifError::InvalidConfiguration));
        header = AdifHeader::aac_lc_mono(4, 1).unwrap();
        header.bitrate = 1 << 23;
        assert_eq!(header.to_bytes(), Err(AdifError::InvalidConfiguration));
    }

    #[test]
    fn parser_reports_signature_and_truncation_errors() {
        assert!(matches!(
            AdifHeader::parse(b"ADI"),
            Err(AdifError::Bit(BitError::UnexpectedEof { .. }))
        ));
        assert_eq!(AdifHeader::parse(b"NOPE"), Err(AdifError::InvalidSignature));
        assert!(matches!(
            AdifHeader::parse(b"ADIF"),
            Err(AdifError::Bit(BitError::UnexpectedEof { .. }))
        ));
    }

    #[test]
    fn error_conversions_and_messages_preserve_the_cause() {
        let bit = BitError::UnexpectedEof {
            needed_bits: 1,
            remaining_bits: 0,
        };
        assert_eq!(AdifError::from(bit.clone()), AdifError::Bit(bit));
        let asc = AscError::InvalidSamplingFrequencyIndex(15);
        assert_eq!(AdifError::from(asc.clone()), AdifError::Asc(asc));
        assert_eq!(
            AdifError::InvalidSignature.to_string(),
            "invalid ADIF signature"
        );
        assert_eq!(
            AdifError::InvalidConfiguration.to_string(),
            "invalid ADIF encoder configuration"
        );
        let bit = BitError::UnexpectedEof {
            needed_bits: 1,
            remaining_bits: 0,
        };
        assert_eq!(AdifError::Bit(bit.clone()).to_string(), bit.to_string());
        let asc = AscError::InvalidSamplingFrequencyIndex(15);
        assert_eq!(AdifError::Asc(asc.clone()).to_string(), asc.to_string());
    }
}
