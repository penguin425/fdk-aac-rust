//! Pure Rust AAC section_data parsing for AAC-LC.

use std::fmt;

use crate::bits::{BitError, BitReader};
use crate::ics::IcsInfo;

pub const ZERO_HCB: u8 = 0;
pub const ESCBOOK: u8 = 11;
pub const BOOKSCL: u8 = 12;
pub const NOISE_HCB: u8 = 13;
pub const INTENSITY_HCB2: u8 = 14;
pub const INTENSITY_HCB: u8 = 15;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SectionData {
    pub sections: Vec<Section>,
    /// Codebook per group and scalefactor band. Outer index is group, inner is sfb.
    pub codebooks: Vec<Vec<u8>>,
    pub bits_read: usize,
}

impl SectionData {
    pub fn parse_aac_lc(reader: &mut BitReader<'_>, ics: &IcsInfo) -> Result<Self, SectionError> {
        Self::parse_aac_lc_with_vcb11(reader, ics, false)
    }

    pub fn parse_aac_lc_with_vcb11(
        reader: &mut BitReader<'_>,
        ics: &IcsInfo,
        vcb11_enabled: bool,
    ) -> Result<Self, SectionError> {
        let start = reader.bits_read();
        let num_groups = ics.window_group_lengths.len();
        let max_sfb = ics.max_sfb as usize;
        let sect_len_bits = if ics.window_sequence.is_long() { 5 } else { 3 };
        let sect_esc_val = (1usize << sect_len_bits) - 1;

        let mut sections = Vec::new();
        let mut codebooks = vec![vec![ZERO_HCB; max_sfb]; num_groups];

        for (group, group_codebooks) in codebooks.iter_mut().enumerate() {
            let mut band = 0usize;
            while band < max_sfb {
                let codebook = reader.read_u8(if vcb11_enabled { 5 } else { 4 })?;
                validate_codebook(codebook, vcb11_enabled)?;

                let mut sect_len = 0usize;
                if vcb11_enabled && (codebook == ESCBOOK || codebook >= 16) {
                    sect_len = 1;
                } else {
                    loop {
                        let incr = reader.read_u8(sect_len_bits)? as usize;
                        sect_len += incr;
                        if incr != sect_esc_val {
                            break;
                        }
                    }
                }
                if sect_len == 0 {
                    return Err(SectionError::ZeroLengthSection { group, band });
                }

                let top = band + sect_len;
                if top > max_sfb {
                    return Err(SectionError::SectionExceedsMaxSfb {
                        group,
                        band,
                        top,
                        max_sfb,
                    });
                }

                for slot in &mut group_codebooks[band..top] {
                    *slot = codebook;
                }
                sections.push(Section {
                    group,
                    start_sfb: band as u8,
                    end_sfb: top as u8,
                    codebook,
                });
                band = top;
            }
        }

        Ok(Self {
            sections,
            codebooks,
            bits_read: reader.bits_read() - start,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Section {
    pub group: usize,
    pub start_sfb: u8,
    pub end_sfb: u8,
    pub codebook: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SectionError {
    Bit(BitError),
    InvalidCodebook(u8),
    ZeroLengthSection {
        group: usize,
        band: usize,
    },
    SectionExceedsMaxSfb {
        group: usize,
        band: usize,
        top: usize,
        max_sfb: usize,
    },
}

impl From<BitError> for SectionError {
    fn from(value: BitError) -> Self {
        Self::Bit(value)
    }
}

impl fmt::Display for SectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bit(err) => err.fmt(f),
            Self::InvalidCodebook(cb) => write!(f, "invalid AAC section codebook {cb}"),
            Self::ZeroLengthSection { group, band } => {
                write!(f, "zero-length AAC section at group {group}, band {band}")
            }
            Self::SectionExceedsMaxSfb {
                group,
                band,
                top,
                max_sfb,
            } => write!(
                f,
                "AAC section at group {group}, band {band} ends at {top}, exceeding max_sfb {max_sfb}"
            ),
        }
    }
}

impl std::error::Error for SectionError {}

fn validate_codebook(codebook: u8, vcb11_enabled: bool) -> Result<(), SectionError> {
    if codebook == BOOKSCL || (!vcb11_enabled && codebook > INTENSITY_HCB) {
        Err(SectionError::InvalidCodebook(codebook))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitWriter;
    use crate::ics::{IcsInfo, WindowSequence, WindowShape};

    fn long_ics(max_sfb: u8) -> IcsInfo {
        IcsInfo {
            window_sequence: WindowSequence::OnlyLong,
            window_shape: WindowShape::Sine,
            max_sfb,
            total_sfb: 51,
            predictor_data_present: false,
            scale_factor_grouping: 0,
            window_group_lengths: vec![1],
            bits_read: 0,
        }
    }

    fn short_ics(max_sfb: u8) -> IcsInfo {
        IcsInfo {
            window_sequence: WindowSequence::EightShort,
            window_shape: WindowShape::Sine,
            max_sfb,
            total_sfb: 15,
            predictor_data_present: false,
            scale_factor_grouping: 0,
            window_group_lengths: vec![1, 1],
            bits_read: 0,
        }
    }

    #[test]
    fn parses_long_sections() {
        let mut writer = BitWriter::new();
        writer.write(1, 4); // cb
        writer.write(2, 5); // len
        writer.write(0, 4); // cb
        writer.write(2, 5); // len

        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        let data = SectionData::parse_aac_lc(&mut reader, &long_ics(4)).unwrap();
        assert_eq!(data.codebooks, vec![vec![1, 1, 0, 0]]);
        assert_eq!(data.sections.len(), 2);
        assert_eq!(data.bits_read, 18);
    }

    #[test]
    fn parses_short_sections_for_each_group() {
        let mut writer = BitWriter::new();
        for codebook in [1u32, 2] {
            writer.write(codebook, 4);
            writer.write(3, 3);
        }

        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        let data = SectionData::parse_aac_lc(&mut reader, &short_ics(3)).unwrap();
        assert_eq!(data.codebooks, vec![vec![1, 1, 1], vec![2, 2, 2]]);
        assert_eq!(data.sections.len(), 2);
        assert_eq!(data.bits_read, 14);
    }

    #[test]
    fn rejects_invalid_and_oversized_sections() {
        let mut writer = BitWriter::new();
        writer.write(BOOKSCL as u32, 4);
        writer.write(1, 5);
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        assert_eq!(
            SectionData::parse_aac_lc(&mut reader, &long_ics(1)).unwrap_err(),
            SectionError::InvalidCodebook(BOOKSCL)
        );

        let mut writer = BitWriter::new();
        writer.write(1, 4);
        writer.write(3, 5);
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        assert!(matches!(
            SectionData::parse_aac_lc(&mut reader, &long_ics(2)).unwrap_err(),
            SectionError::SectionExceedsMaxSfb { .. }
        ));
    }

    #[test]
    fn parses_vcb11_virtual_codebooks_with_implicit_unit_sections() {
        let mut writer = BitWriter::new();
        writer.write(16, 5);
        writer.write(31, 5);
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        let data = SectionData::parse_aac_lc_with_vcb11(&mut reader, &long_ics(2), true).unwrap();
        assert_eq!(data.codebooks, [vec![16, 31]]);
        assert_eq!(data.bits_read, 10);
    }

    #[test]
    fn parses_escaped_long_length_and_empty_layout() {
        let mut writer = BitWriter::new();
        writer.write(1, 4);
        writer.write(31, 5);
        writer.write(2, 5);
        let data = SectionData::parse_aac_lc(&mut BitReader::new(&writer.finish()), &long_ics(33))
            .unwrap();
        assert_eq!(data.sections[0].end_sfb, 33);
        assert_eq!(data.codebooks[0], vec![1; 33]);

        let empty = SectionData::parse_aac_lc(&mut BitReader::new(&[]), &long_ics(0)).unwrap();
        assert!(empty.sections.is_empty());
        assert_eq!(empty.codebooks, vec![Vec::<u8>::new()]);
        assert_eq!(empty.bits_read, 0);
    }

    #[test]
    fn rejects_zero_length_truncation_and_all_invalid_codebooks() {
        let mut writer = BitWriter::new();
        writer.write(1, 4);
        writer.write(0, 5);
        assert!(matches!(
            SectionData::parse_aac_lc(&mut BitReader::new(&writer.finish()), &long_ics(1)),
            Err(SectionError::ZeroLengthSection { .. })
        ));
        assert!(matches!(
            SectionData::parse_aac_lc(&mut BitReader::new(&[]), &long_ics(1)),
            Err(SectionError::Bit(BitError::UnexpectedEof { .. }))
        ));
        assert_eq!(
            validate_codebook(16, false),
            Err(SectionError::InvalidCodebook(16))
        );
        assert_eq!(
            validate_codebook(BOOKSCL, true),
            Err(SectionError::InvalidCodebook(BOOKSCL))
        );
        assert!(validate_codebook(31, true).is_ok());
    }

    #[test]
    fn vcb11_escbook_uses_implicit_single_band_section() {
        let mut writer = BitWriter::new();
        writer.write(ESCBOOK as u32, 5);
        let data = SectionData::parse_aac_lc_with_vcb11(
            &mut BitReader::new(&writer.finish()),
            &long_ics(1),
            true,
        )
        .unwrap();
        assert_eq!(data.codebooks, [vec![ESCBOOK]]);
        assert_eq!(data.bits_read, 5);
    }

    #[test]
    fn formats_all_section_errors() {
        let bit = BitError::UnexpectedEof {
            needed_bits: 1,
            remaining_bits: 0,
        };
        assert_eq!(
            SectionError::from(bit.clone()),
            SectionError::Bit(bit.clone())
        );
        assert_eq!(SectionError::Bit(bit.clone()).to_string(), bit.to_string());
        for error in [
            SectionError::InvalidCodebook(12),
            SectionError::ZeroLengthSection { group: 0, band: 1 },
            SectionError::SectionExceedsMaxSfb {
                group: 0,
                band: 1,
                top: 4,
                max_sfb: 3,
            },
        ] {
            assert!(!error.to_string().is_empty());
        }
    }
}
