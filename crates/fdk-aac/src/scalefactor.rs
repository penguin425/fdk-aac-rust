//! AAC-LC scale_factor_data planning helpers.
//!
//! Full scalefactor decoding requires porting the AAC scalefactor Huffman table.
//! This module implements the traversal semantics from FDK's
//! `CBlock_ReadScaleFactorData`: given section codebooks, decide which SFBs need
//! spectral scalefactor deltas, intensity deltas, noise/PNS data, or no value.

use std::fmt;

use crate::bits::BitReader;
use crate::huffman::{decode_fdk_2bit, HuffmanError, HUFFMAN_CODEBOOK_SCL};
use crate::section::{SectionData, INTENSITY_HCB, INTENSITY_HCB2, NOISE_HCB, ZERO_HCB};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScalefactorPlan {
    pub entries: Vec<ScalefactorEntry>,
    pub groups: usize,
    pub max_sfb: usize,
}

impl ScalefactorPlan {
    pub fn from_section_data(section_data: &SectionData) -> Result<Self, ScalefactorError> {
        let groups = section_data.codebooks.len();
        let max_sfb = section_data
            .codebooks
            .first()
            .map_or(0, |group| group.len());

        let mut entries = Vec::with_capacity(groups * max_sfb);
        for (group, codebooks) in section_data.codebooks.iter().enumerate() {
            if codebooks.len() != max_sfb {
                return Err(ScalefactorError::RaggedCodebookGrid);
            }
            for (sfb, &codebook) in codebooks.iter().enumerate() {
                let kind = match codebook {
                    ZERO_HCB => ScalefactorKind::Zero,
                    INTENSITY_HCB | INTENSITY_HCB2 => ScalefactorKind::Intensity,
                    NOISE_HCB => ScalefactorKind::Noise,
                    1..=11 | 16..=31 => ScalefactorKind::Spectral,
                    other => return Err(ScalefactorError::InvalidCodebook(other)),
                };
                entries.push(ScalefactorEntry {
                    group,
                    sfb,
                    codebook,
                    kind,
                });
            }
        }

        Ok(Self {
            entries,
            groups,
            max_sfb,
        })
    }

    pub fn huffman_delta_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.kind.requires_huffman_delta())
            .count()
    }

    /// Apply scalefactor Huffman output words to this plan.
    ///
    /// FDK's scalefactor Huffman decoder returns values centered around 60. This
    /// method takes that decoded word stream and applies the AAC-LC accumulators:
    /// spectral scalefactors are relative to `global_gain`, intensity uses a
    /// separate `position` accumulator, and zero bands are forced to zero.
    pub fn apply_decoded_words(
        &self,
        global_gain: u8,
        decoded_words: &[i16],
    ) -> Result<ScalefactorData, ScalefactorError> {
        let expected = self.huffman_delta_count();
        if decoded_words.len() != expected {
            return Err(ScalefactorError::DecodedWordCountMismatch {
                expected,
                actual: decoded_words.len(),
            });
        }

        let mut words = decoded_words.iter().copied();
        let mut factor = global_gain as i16;
        let mut position = 0i16;
        let mut values = vec![vec![0i16; self.max_sfb]; self.groups];

        for entry in &self.entries {
            let value = match entry.kind {
                ScalefactorKind::Zero => 0,
                ScalefactorKind::Spectral => {
                    let word = words.next().expect("count checked");
                    factor += word - 60;
                    factor - 100
                }
                ScalefactorKind::Intensity => {
                    let word = words.next().expect("count checked");
                    position += word - 60;
                    position - 100
                }
                ScalefactorKind::Noise => {
                    let word = words.next().expect("count checked");
                    global_gain as i16 + word - 60 - 100
                }
            };
            values[entry.group][entry.sfb] = value;
        }

        Ok(ScalefactorData { values })
    }

    pub fn decode_from_bitstream(
        &self,
        reader: &mut BitReader<'_>,
        global_gain: u8,
    ) -> Result<ScalefactorData, ScalefactorError> {
        let mut decoded_words = Vec::with_capacity(self.huffman_delta_count());
        for entry in &self.entries {
            if entry.kind.requires_huffman_delta() {
                decoded_words.push(decode_fdk_2bit(reader, &HUFFMAN_CODEBOOK_SCL)? as i16);
            }
        }
        self.apply_decoded_words(global_gain, &decoded_words)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScalefactorData {
    pub values: Vec<Vec<i16>>,
}

impl ScalefactorData {
    /// Decode USAC `scale_factor_data()`. Unlike AAC section data, every SFB is
    /// active and the first value is implied directly by `global_gain`.
    pub fn decode_usac(
        reader: &mut BitReader<'_>,
        global_gain: u8,
        window_groups: usize,
        max_sfb: usize,
    ) -> Result<Self, ScalefactorError> {
        let mut factor = i16::from(global_gain);
        let mut values = vec![vec![0; max_sfb]; window_groups];
        for (group, group_values) in values.iter_mut().enumerate() {
            for (band, value) in group_values.iter_mut().enumerate() {
                if group != 0 || band != 0 {
                    factor += decode_fdk_2bit(reader, &HUFFMAN_CODEBOOK_SCL)? as i16 - 60;
                }
                *value = factor - 100;
            }
        }
        Ok(Self { values })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalefactorEntry {
    pub group: usize,
    pub sfb: usize,
    pub codebook: u8,
    pub kind: ScalefactorKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalefactorKind {
    Zero,
    Spectral,
    Intensity,
    Noise,
}

impl ScalefactorKind {
    pub fn requires_huffman_delta(self) -> bool {
        match self {
            Self::Zero => false,
            Self::Spectral | Self::Intensity | Self::Noise => true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScalefactorError {
    Huffman(HuffmanError),
    RaggedCodebookGrid,
    InvalidCodebook(u8),
    DecodedWordCountMismatch { expected: usize, actual: usize },
}

impl From<HuffmanError> for ScalefactorError {
    fn from(value: HuffmanError) -> Self {
        Self::Huffman(value)
    }
}

impl fmt::Display for ScalefactorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Huffman(err) => err.fmt(f),
            Self::RaggedCodebookGrid => write!(f, "ragged AAC scalefactor codebook grid"),
            Self::InvalidCodebook(codebook) => {
                write!(f, "invalid codebook {codebook} in AAC scalefactor grid")
            }
            Self::DecodedWordCountMismatch { expected, actual } => write!(
                f,
                "AAC scalefactor decoded word count mismatch: expected {expected}, got {actual}"
            ),
        }
    }
}

impl std::error::Error for ScalefactorError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitReader;
    use crate::section::{Section, SectionData};

    #[test]
    fn creates_plan_in_fdks_traversal_order() {
        let section_data = SectionData {
            sections: vec![Section {
                group: 0,
                start_sfb: 0,
                end_sfb: 4,
                codebook: 1,
            }],
            codebooks: vec![
                vec![ZERO_HCB, 1, INTENSITY_HCB, NOISE_HCB],
                vec![2, ZERO_HCB, INTENSITY_HCB2, 11],
            ],
            bits_read: 0,
        };

        let plan = ScalefactorPlan::from_section_data(&section_data).unwrap();
        assert_eq!(plan.groups, 2);
        assert_eq!(plan.max_sfb, 4);
        assert_eq!(plan.huffman_delta_count(), 6);
        assert_eq!(plan.entries[0].kind, ScalefactorKind::Zero);
        assert_eq!(plan.entries[1].kind, ScalefactorKind::Spectral);
        assert_eq!(plan.entries[2].kind, ScalefactorKind::Intensity);
        assert_eq!(plan.entries[3].kind, ScalefactorKind::Noise);
        assert_eq!(plan.entries[4].group, 1);
        assert_eq!(plan.entries[4].sfb, 0);
    }

    #[test]
    fn rejects_ragged_or_invalid_codebook_grid() {
        let ragged = SectionData {
            sections: Vec::new(),
            codebooks: vec![vec![1, 2], vec![1]],
            bits_read: 0,
        };
        assert_eq!(
            ScalefactorPlan::from_section_data(&ragged).unwrap_err(),
            ScalefactorError::RaggedCodebookGrid
        );

        let invalid = SectionData {
            sections: Vec::new(),
            codebooks: vec![vec![32]],
            bits_read: 0,
        };
        assert_eq!(
            ScalefactorPlan::from_section_data(&invalid).unwrap_err(),
            ScalefactorError::InvalidCodebook(32)
        );
    }

    #[test]
    fn applies_decoded_words_like_fdk_accumulators() {
        let section_data = SectionData {
            sections: Vec::new(),
            codebooks: vec![vec![
                ZERO_HCB,
                1,
                1,
                INTENSITY_HCB,
                INTENSITY_HCB2,
                NOISE_HCB,
            ]],
            bits_read: 0,
        };
        let plan = ScalefactorPlan::from_section_data(&section_data).unwrap();
        let data = plan
            .apply_decoded_words(100, &[61, 59, 62, 58, 60])
            .unwrap();

        // spectral factor starts at global_gain=100 and stores factor-100.
        // intensity position starts at 0 and stores position-100.
        assert_eq!(data.values, vec![vec![0, 1, 0, -98, -100, 0]]);
    }

    #[test]
    fn rejects_wrong_decoded_word_count() {
        let section_data = SectionData {
            sections: Vec::new(),
            codebooks: vec![vec![1, 1]],
            bits_read: 0,
        };
        let plan = ScalefactorPlan::from_section_data(&section_data).unwrap();
        assert_eq!(
            plan.apply_decoded_words(100, &[60]).unwrap_err(),
            ScalefactorError::DecodedWordCountMismatch {
                expected: 2,
                actual: 1
            }
        );
    }

    #[test]
    fn decodes_scalefactors_from_bitstream() {
        let section_data = SectionData {
            sections: Vec::new(),
            codebooks: vec![vec![ZERO_HCB, 1, 1]],
            bits_read: 0,
        };
        let plan = ScalefactorPlan::from_section_data(&section_data).unwrap();

        // SCL Huffman: 0 -> 60 (1 bit effective via pushback), then 100 -> 59.
        let mut reader = BitReader::new(&[0b0100_0000]);
        let data = plan.decode_from_bitstream(&mut reader, 100).unwrap();
        assert_eq!(data.values, vec![vec![0, 0, -1]]);
        assert_eq!(reader.bits_read(), 4);
    }

    #[test]
    fn decodes_usac_with_implied_first_scalefactor() {
        // First SFB consumes no bits. The next Huffman words are 60 and 59.
        let mut reader = BitReader::new(&[0b0100_0000]);
        let data = ScalefactorData::decode_usac(&mut reader, 100, 1, 3).unwrap();
        assert_eq!(data.values, vec![vec![0, 0, -1]]);
        assert_eq!(reader.bits_read(), 4);
    }

    #[test]
    fn empty_and_virtual_codebook_plans_are_supported() {
        let empty = SectionData {
            sections: Vec::new(),
            codebooks: Vec::new(),
            bits_read: 0,
        };
        let plan = ScalefactorPlan::from_section_data(&empty).unwrap();
        assert_eq!(plan.groups, 0);
        assert_eq!(plan.max_sfb, 0);
        assert_eq!(plan.huffman_delta_count(), 0);
        assert_eq!(
            plan.apply_decoded_words(100, &[]).unwrap().values,
            Vec::<Vec<i16>>::new()
        );

        let virtual_books = SectionData {
            sections: Vec::new(),
            codebooks: vec![vec![16, 31]],
            bits_read: 0,
        };
        let plan = ScalefactorPlan::from_section_data(&virtual_books).unwrap();
        assert!(plan
            .entries
            .iter()
            .all(|entry| entry.kind == ScalefactorKind::Spectral));
    }

    #[test]
    fn zero_sized_usac_grid_consumes_no_bits() {
        let mut reader = BitReader::new(&[]);
        assert_eq!(
            ScalefactorData::decode_usac(&mut reader, 100, 2, 0)
                .unwrap()
                .values,
            vec![vec![], vec![]]
        );
        assert_eq!(reader.bits_read(), 0);
    }

    #[test]
    fn bitstream_decoders_propagate_huffman_eof() {
        let section_data = SectionData {
            sections: Vec::new(),
            codebooks: vec![vec![1]],
            bits_read: 0,
        };
        let plan = ScalefactorPlan::from_section_data(&section_data).unwrap();
        assert!(matches!(
            plan.decode_from_bitstream(&mut BitReader::new(&[]), 100),
            Err(ScalefactorError::Huffman(_))
        ));
        assert!(matches!(
            ScalefactorData::decode_usac(&mut BitReader::new(&[]), 100, 1, 2),
            Err(ScalefactorError::Huffman(_))
        ));
    }

    #[test]
    fn formats_every_scalefactor_error() {
        assert!(!ScalefactorError::RaggedCodebookGrid.to_string().is_empty());
        assert!(!ScalefactorError::InvalidCodebook(32).to_string().is_empty());
        assert!(!ScalefactorError::DecodedWordCountMismatch {
            expected: 2,
            actual: 1
        }
        .to_string()
        .is_empty());
        let huffman = HuffmanError::InvalidCodebook(12);
        assert_eq!(
            ScalefactorError::from(huffman.clone()),
            ScalefactorError::Huffman(huffman.clone())
        );
        assert_eq!(
            ScalefactorError::Huffman(huffman.clone()).to_string(),
            huffman.to_string()
        );
    }
}
