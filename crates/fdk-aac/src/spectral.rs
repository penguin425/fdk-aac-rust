//! AAC-LC spectral Huffman word expansion helpers.

use std::fmt;

use crate::bits::{BitError, BitReader};
use crate::huffman::{decode_fdk_2bit, spectral_codebook, CodeBookDescription, HuffmanError};
use crate::ics::IcsInfo;
use crate::section::{SectionData, ESCBOOK, INTENSITY_HCB, INTENSITY_HCB2, NOISE_HCB, ZERO_HCB};
use crate::sfb::{aac_lc_band_offsets_for_ics, SfbError};

pub const MAX_QUANTIZED_VALUE: i32 = 8191;
const VCB11_LARGEST_ABSOLUTE_VALUE: [i32; 16] = [
    15, 31, 47, 63, 95, 127, 159, 191, 223, 255, 319, 383, 511, 767, 1023, 2047,
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpectralData {
    /// Per-window quantized MDCT coefficients before inverse quantization.
    pub windows: Vec<Vec<i32>>,
}

pub fn decode_spectral_data(
    reader: &mut BitReader<'_>,
    ics: &IcsInfo,
    section_data: &SectionData,
    band_offsets: &[usize],
    granule_length: usize,
) -> Result<SpectralData, SpectralError> {
    validate_layout(ics, section_data, band_offsets, granule_length)?;

    let mut windows = vec![vec![0i32; granule_length]; total_windows(ics)];
    let mut group_offset = 0usize;

    for (group, &group_len) in ics.window_group_lengths.iter().enumerate() {
        for band in 0..ics.max_sfb as usize {
            let codebook = section_data.codebooks[group][band];
            if !is_spectral_codebook(codebook) {
                continue;
            }

            let description = spectral_codebook(if (16..=31).contains(&codebook) {
                ESCBOOK
            } else {
                codebook
            })?;
            let step = description.dimension as usize;
            let band_start = band_offsets[band];
            let band_end = band_offsets[band + 1];
            if (band_end - band_start) % step != 0 {
                return Err(SpectralError::BandWidthNotMultipleOfStep {
                    band,
                    width: band_end - band_start,
                    step,
                });
            }

            for group_window in 0..group_len as usize {
                let window = group_offset + group_window;
                for index in (band_start..band_end).step_by(step) {
                    let coeffs = decode_spectral_tuple(reader, codebook)?;
                    windows[window][index..index + step].copy_from_slice(&coeffs);
                }
            }
        }
        group_offset += group_len as usize;
    }

    Ok(SpectralData { windows })
}

pub fn decode_aac_lc_spectral_data(
    reader: &mut BitReader<'_>,
    sampling_frequency_index: u8,
    ics: &IcsInfo,
    section_data: &SectionData,
) -> Result<SpectralData, SpectralError> {
    let sfb = aac_lc_band_offsets_for_ics(sampling_frequency_index, ics)?;
    decode_spectral_data(reader, ics, section_data, sfb.offsets, sfb.granule_length)
}

fn is_spectral_codebook(codebook: u8) -> bool {
    !matches!(
        codebook,
        ZERO_HCB | NOISE_HCB | INTENSITY_HCB | INTENSITY_HCB2
    )
}

fn total_windows(ics: &IcsInfo) -> usize {
    ics.window_group_lengths
        .iter()
        .map(|&length| length as usize)
        .sum()
}

fn validate_layout(
    ics: &IcsInfo,
    section_data: &SectionData,
    band_offsets: &[usize],
    granule_length: usize,
) -> Result<(), SpectralError> {
    let groups = ics.window_group_lengths.len();
    let max_sfb = ics.max_sfb as usize;
    if section_data.codebooks.len() != groups {
        return Err(SpectralError::LayoutMismatch);
    }
    if section_data
        .codebooks
        .iter()
        .any(|group| group.len() < max_sfb)
    {
        return Err(SpectralError::LayoutMismatch);
    }
    if band_offsets.len() <= max_sfb || band_offsets[max_sfb] > granule_length {
        return Err(SpectralError::InvalidBandOffsets);
    }
    if band_offsets.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(SpectralError::InvalidBandOffsets);
    }
    Ok(())
}

pub fn decode_spectral_tuple(
    reader: &mut BitReader<'_>,
    codebook: u8,
) -> Result<Vec<i32>, SpectralError> {
    let effective_codebook = if (16..=31).contains(&codebook) {
        ESCBOOK
    } else {
        codebook
    };
    let description = spectral_codebook(effective_codebook)?;
    let word = decode_fdk_2bit(reader, description.table)?;
    let coefficients = expand_spectral_word(reader, effective_codebook, description, word)?;
    if let Some(&maximum) = codebook
        .checked_sub(16)
        .and_then(|index| VCB11_LARGEST_ABSOLUTE_VALUE.get(index as usize))
    {
        if let Some(&value) = coefficients.iter().find(|&&value| value.abs() > maximum) {
            return Err(SpectralError::LargestAbsoluteValueExceeded {
                codebook,
                value,
                maximum,
            });
        }
    }
    Ok(coefficients)
}

pub fn expand_spectral_word(
    reader: &mut BitReader<'_>,
    codebook: u8,
    description: CodeBookDescription,
    mut word: u16,
) -> Result<Vec<i32>, SpectralError> {
    let mask = description.mask();
    let mut coeffs = Vec::with_capacity(description.dimension as usize);

    for _ in 0..description.dimension {
        let mut value = ((word & mask) as i32) - description.offset as i32;
        word >>= description.num_bits;

        // FDK reads sign bits only for unsigned codebooks (offset == 0) and only
        // for non-zero coefficients.
        if description.offset == 0 && value != 0 && reader.read_bool()? {
            value = -value;
        }

        coeffs.push(value);
    }

    if codebook == ESCBOOK {
        for value in coeffs.iter_mut().take(2) {
            *value = read_escape(reader, *value)?;
        }
    }

    Ok(coeffs)
}

pub fn read_escape(reader: &mut BitReader<'_>, q: i32) -> Result<i32, SpectralError> {
    if q.abs() != 16 {
        return Ok(q);
    }

    let mut i = 4usize;
    while i < 13 {
        if !reader.read_bool()? {
            break;
        }
        i += 1;
    }

    if i == 13 {
        return Ok(MAX_QUANTIZED_VALUE + 1);
    }

    let off = reader.read(i)? as i32;
    let mut value = off + (1i32 << i);
    if q < 0 {
        value = -value;
    }
    Ok(value)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpectralError {
    Bit(BitError),
    Huffman(HuffmanError),
    Sfb(SfbError),
    BandWidthNotMultipleOfStep {
        band: usize,
        width: usize,
        step: usize,
    },
    InvalidBandOffsets,
    LayoutMismatch,
    LargestAbsoluteValueExceeded {
        codebook: u8,
        value: i32,
        maximum: i32,
    },
}

impl From<BitError> for SpectralError {
    fn from(value: BitError) -> Self {
        Self::Bit(value)
    }
}

impl From<HuffmanError> for SpectralError {
    fn from(value: HuffmanError) -> Self {
        Self::Huffman(value)
    }
}

impl From<SfbError> for SpectralError {
    fn from(value: SfbError) -> Self {
        Self::Sfb(value)
    }
}

impl fmt::Display for SpectralError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bit(err) => err.fmt(f),
            Self::Huffman(err) => err.fmt(f),
            Self::Sfb(err) => err.fmt(f),
            Self::BandWidthNotMultipleOfStep { band, width, step } => write!(
                f,
                "AAC spectral band {band} width {width} is not a multiple of Huffman step {step}"
            ),
            Self::InvalidBandOffsets => write!(f, "invalid AAC spectral band offsets"),
            Self::LayoutMismatch => {
                write!(f, "AAC spectral layout does not match ICS/section data")
            }
            Self::LargestAbsoluteValueExceeded {
                codebook,
                value,
                maximum,
            } => write!(
                f,
                "AAC virtual codebook {codebook} decoded value {value} beyond largest absolute value {maximum}"
            ),
        }
    }
}

impl std::error::Error for SpectralError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitWriter;
    use crate::huffman::spectral_codebook;
    use crate::huffman::write_spectral_tuple;
    use crate::ics::{IcsInfo, WindowSequence, WindowShape};
    use crate::section::SectionData;

    fn test_ics(window_group_lengths: Vec<u8>, max_sfb: u8) -> IcsInfo {
        IcsInfo {
            window_sequence: if window_group_lengths.iter().sum::<u8>() == 1 {
                WindowSequence::OnlyLong
            } else {
                WindowSequence::EightShort
            },
            window_shape: WindowShape::Sine,
            max_sfb,
            total_sfb: max_sfb,
            predictor_data_present: false,
            scale_factor_grouping: 0,
            window_group_lengths,
            bits_read: 0,
        }
    }

    #[test]
    fn expands_signed_offset_codebook_without_extra_sign_bits() {
        let description = spectral_codebook(1).unwrap();
        let mut reader = BitReader::new(&[0xff]);
        let coeffs = expand_spectral_word(&mut reader, 1, description, 0b11_10_01_00).unwrap();
        assert_eq!(coeffs, vec![-1, 0, 1, 2]);
        assert_eq!(reader.bits_read(), 0);
    }

    #[test]
    fn expands_unsigned_codebook_with_sign_bits_for_non_zero_coefficients() {
        let description = spectral_codebook(7).unwrap();
        let mut reader = BitReader::new(&[0b10_000000]);
        let coeffs = expand_spectral_word(&mut reader, 7, description, 0x0021).unwrap();
        assert_eq!(coeffs, vec![-1, 2]);
        assert_eq!(reader.bits_read(), 2);
    }

    #[test]
    fn skips_sign_bits_for_zero_coefficients() {
        let description = spectral_codebook(7).unwrap();
        let mut reader = BitReader::new(&[0xff]);
        let coeffs = expand_spectral_word(&mut reader, 7, description, 0x0000).unwrap();
        assert_eq!(coeffs, vec![0, 0]);
        assert_eq!(reader.bits_read(), 0);
    }

    #[test]
    fn applies_escape_sequence_for_codebook_11() {
        let description = spectral_codebook(11).unwrap();
        let mut writer = BitWriter::new();
        writer.write_bool(false); // sign for first coefficient: positive
        writer.write_bool(true); // sign for second coefficient: negative
        writer.write_bool(false); // first escape prefix ends at i=4
        writer.write(0b0011, 4); // 16 + 3 = 19
        writer.write_bool(true); // second escape prefix: i=5
        writer.write_bool(false);
        writer.write(0b00010, 5); // -(32 + 2) = -34
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);

        let coeffs = expand_spectral_word(&mut reader, 11, description, 0b1_0000_1_0000).unwrap();
        assert_eq!(coeffs, vec![19, -34]);
    }

    #[test]
    fn virtual_codebook_16_uses_codebook_11_tree() {
        let mut virtual_reader = BitReader::new(&[0; 8]);
        let mut escape_reader = BitReader::new(&[0; 8]);
        assert_eq!(
            decode_spectral_tuple(&mut virtual_reader, 16).unwrap(),
            decode_spectral_tuple(&mut escape_reader, ESCBOOK).unwrap()
        );
        assert_eq!(virtual_reader.bits_read(), escape_reader.bits_read());
    }

    #[test]
    fn escape_overflow_matches_fdk_sentinel() {
        let mut writer = BitWriter::new();
        for _ in 0..9 {
            writer.write_bool(true);
        }
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        assert_eq!(
            read_escape(&mut reader, 16).unwrap(),
            MAX_QUANTIZED_VALUE + 1
        );
    }

    #[test]
    fn decodes_spectral_data_into_grouped_windows() {
        let ics = test_ics(vec![2], 2);
        let section_data = SectionData {
            sections: Vec::new(),
            codebooks: vec![vec![ZERO_HCB, 1]],
            bits_read: 0,
        };
        let band_offsets = [0usize, 4, 8];

        // Codebook 1 all-zero prefix decodes a zero tuple. The single spectral
        // band still has one tuple per grouped window and consumes Huffman bits.
        let mut reader = BitReader::new(&[0; 4]);
        let spectral =
            decode_spectral_data(&mut reader, &ics, &section_data, &band_offsets, 8).unwrap();

        assert_eq!(spectral.windows.len(), 2);
        assert_eq!(spectral.windows[0], vec![0; 8]);
        assert_eq!(spectral.windows[1], vec![0; 8]);
        assert!(reader.bits_read() > 0);
    }

    #[test]
    fn rejects_band_width_that_does_not_match_codebook_step() {
        let ics = test_ics(vec![1], 1);
        let section_data = SectionData {
            sections: Vec::new(),
            codebooks: vec![vec![1]],
            bits_read: 0,
        };
        let mut reader = BitReader::new(&[0; 4]);
        assert!(matches!(
            decode_spectral_data(&mut reader, &ics, &section_data, &[0, 3], 3).unwrap_err(),
            SpectralError::BandWidthNotMultipleOfStep { .. }
        ));

        let reserved = SectionData {
            sections: Vec::new(),
            codebooks: vec![vec![12]],
            bits_read: 0,
        };
        assert_eq!(
            decode_spectral_data(&mut BitReader::new(&[]), &ics, &reserved, &[0, 4], 4),
            Err(SpectralError::Huffman(HuffmanError::InvalidCodebook(12)))
        );
    }

    #[test]
    fn decodes_with_aac_lc_sfb_lookup() {
        let ics = test_ics(vec![1], 1);
        let section_data = SectionData {
            sections: Vec::new(),
            codebooks: vec![vec![1]],
            bits_read: 0,
        };

        let mut reader = BitReader::new(&[0; 4]);
        let spectral = decode_aac_lc_spectral_data(&mut reader, 4, &ics, &section_data).unwrap();
        assert_eq!(spectral.windows.len(), 1);
        assert_eq!(spectral.windows[0].len(), 1024);
        assert!(reader.bits_read() > 0);
    }

    #[test]
    fn validates_every_spectral_layout_dimension() {
        let ics = test_ics(vec![1, 1], 1);
        let good = SectionData {
            sections: Vec::new(),
            codebooks: vec![vec![0], vec![0]],
            bits_read: 0,
        };
        assert_eq!(
            decode_spectral_data(
                &mut BitReader::new(&[]),
                &ics,
                &SectionData {
                    codebooks: vec![vec![0]],
                    ..good.clone()
                },
                &[0, 4],
                4
            ),
            Err(SpectralError::LayoutMismatch)
        );
        assert_eq!(
            decode_spectral_data(
                &mut BitReader::new(&[]),
                &ics,
                &SectionData {
                    codebooks: vec![vec![], vec![0]],
                    ..good.clone()
                },
                &[0, 4],
                4
            ),
            Err(SpectralError::LayoutMismatch)
        );
        for offsets in [&[0usize][..], &[0, 5][..], &[4, 3][..]] {
            assert_eq!(
                decode_spectral_data(&mut BitReader::new(&[]), &ics, &good, offsets, 4),
                Err(SpectralError::InvalidBandOffsets)
            );
        }
    }

    #[test]
    fn escape_handles_passthrough_negative_and_truncation() {
        assert_eq!(read_escape(&mut BitReader::new(&[]), 15).unwrap(), 15);
        let mut reader = BitReader::new(&[0b0001_0000]);
        assert_eq!(read_escape(&mut reader, -16).unwrap(), -18);
        assert!(matches!(
            read_escape(&mut BitReader::new(&[]), 16),
            Err(SpectralError::Bit(BitError::UnexpectedEof { .. }))
        ));
    }

    #[test]
    fn virtual_codebook_enforces_its_largest_absolute_value() {
        let mut writer = BitWriter::new();
        write_spectral_tuple(&mut writer, ESCBOOK, &[16, 0]).unwrap();
        let bytes = writer.finish();
        assert!(matches!(
            decode_spectral_tuple(&mut BitReader::new(&bytes), 16),
            Err(SpectralError::LargestAbsoluteValueExceeded {
                codebook: 16,
                maximum: 15,
                ..
            })
        ));
    }

    #[test]
    fn converts_and_formats_all_spectral_errors() {
        let bit = BitError::UnexpectedEof {
            needed_bits: 1,
            remaining_bits: 0,
        };
        assert_eq!(
            SpectralError::from(bit.clone()),
            SpectralError::Bit(bit.clone())
        );
        assert_eq!(SpectralError::Bit(bit.clone()).to_string(), bit.to_string());
        let huffman = HuffmanError::InvalidCodebook(12);
        assert_eq!(
            SpectralError::from(huffman.clone()),
            SpectralError::Huffman(huffman.clone())
        );
        let sfb = SfbError::UnsupportedSamplingFrequencyIndex(13);
        assert_eq!(
            SpectralError::from(sfb.clone()),
            SpectralError::Sfb(sfb.clone())
        );
        assert_eq!(
            SpectralError::Huffman(huffman.clone()).to_string(),
            huffman.to_string()
        );
        assert_eq!(SpectralError::Sfb(sfb.clone()).to_string(), sfb.to_string());
        for error in [
            SpectralError::BandWidthNotMultipleOfStep {
                band: 0,
                width: 3,
                step: 4,
            },
            SpectralError::InvalidBandOffsets,
            SpectralError::LayoutMismatch,
            SpectralError::LargestAbsoluteValueExceeded {
                codebook: 16,
                value: 16,
                maximum: 15,
            },
        ] {
            assert!(!error.to_string().is_empty());
        }
    }
}
