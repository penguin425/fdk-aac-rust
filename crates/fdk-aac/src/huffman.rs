//! FDK-style Huffman table decoding helpers.

use std::fmt;

use crate::bits::{BitError, BitReader, BitWriter};
use crate::huffman_tables::*;

pub type FdkHuffmanTable = &'static [[u16; 4]];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CodeBookDescription {
    pub table: FdkHuffmanTable,
    pub dimension: u8,
    pub num_bits: u8,
    pub offset: u8,
}

impl CodeBookDescription {
    pub const fn mask(self) -> u16 {
        (1u16 << self.num_bits) - 1
    }
}

pub const AAC_CODEBOOK_DESCRIPTIONS: [Option<CodeBookDescription>; 12] = [
    None,
    Some(CodeBookDescription {
        table: &HUFFMAN_CODEBOOK_1,
        dimension: 4,
        num_bits: 2,
        offset: 1,
    }),
    Some(CodeBookDescription {
        table: &HUFFMAN_CODEBOOK_2,
        dimension: 4,
        num_bits: 2,
        offset: 1,
    }),
    Some(CodeBookDescription {
        table: &HUFFMAN_CODEBOOK_3,
        dimension: 4,
        num_bits: 2,
        offset: 0,
    }),
    Some(CodeBookDescription {
        table: &HUFFMAN_CODEBOOK_4,
        dimension: 4,
        num_bits: 2,
        offset: 0,
    }),
    Some(CodeBookDescription {
        table: &HUFFMAN_CODEBOOK_5,
        dimension: 2,
        num_bits: 4,
        offset: 4,
    }),
    Some(CodeBookDescription {
        table: &HUFFMAN_CODEBOOK_6,
        dimension: 2,
        num_bits: 4,
        offset: 4,
    }),
    Some(CodeBookDescription {
        table: &HUFFMAN_CODEBOOK_7,
        dimension: 2,
        num_bits: 4,
        offset: 0,
    }),
    Some(CodeBookDescription {
        table: &HUFFMAN_CODEBOOK_8,
        dimension: 2,
        num_bits: 4,
        offset: 0,
    }),
    Some(CodeBookDescription {
        table: &HUFFMAN_CODEBOOK_9,
        dimension: 2,
        num_bits: 4,
        offset: 0,
    }),
    Some(CodeBookDescription {
        table: &HUFFMAN_CODEBOOK_10,
        dimension: 2,
        num_bits: 4,
        offset: 0,
    }),
    Some(CodeBookDescription {
        table: &HUFFMAN_CODEBOOK_11,
        dimension: 2,
        num_bits: 5,
        offset: 0,
    }),
];

pub fn spectral_codebook(codebook: u8) -> Result<CodeBookDescription, HuffmanError> {
    AAC_CODEBOOK_DESCRIPTIONS
        .get(codebook as usize)
        .and_then(|description| *description)
        .ok_or(HuffmanError::InvalidCodebook(codebook))
}

/// Return the exact number of bits used to encode one spectral tuple.
///
/// The length is recovered directly from FDK's decoder table, so this cannot
/// drift from the decoder when a codebook table is corrected.  Sign and
/// ESCBOOK extension bits are included.
pub fn spectral_tuple_bit_cost(codebook: u8, coefficients: &[i32]) -> Option<usize> {
    let description = spectral_codebook(codebook).ok()?;
    if coefficients.len() != usize::from(description.dimension) {
        return None;
    }

    let mut word = 0u16;
    let mut side_bits = 0usize;
    for (index, &coefficient) in coefficients.iter().enumerate() {
        let magnitude = coefficient.unsigned_abs();
        let stored = if description.offset == 0 {
            if codebook == 11 {
                magnitude.min(16)
            } else {
                magnitude
            }
        } else {
            let limit = (1i32 << description.num_bits) - 1 - i32::from(description.offset);
            if coefficient < -i32::from(description.offset) || coefficient > limit {
                return None;
            }
            (coefficient + i32::from(description.offset)) as u32
        };
        if stored >= (1u32 << description.num_bits) {
            return None;
        }
        word |= (stored as u16) << (usize::from(description.num_bits) * index);

        if description.offset == 0 && magnitude != 0 {
            side_bits += 1;
        }
        if codebook == 11 && magnitude >= 16 {
            let exponent = 31 - magnitude.leading_zeros();
            if exponent > 12 {
                return None;
            }
            side_bits += 2 * (exponent as usize - 4) + 5;
        }
    }

    fdk_huffman_word_length(description.table, word).map(|length| length + side_bits)
}

fn fdk_huffman_word_length(table: FdkHuffmanTable, word: u16) -> Option<usize> {
    fdk_huffman_word_code(table, word).map(|(_, length)| length)
}

fn fdk_huffman_word_code(table: FdkHuffmanTable, word: u16) -> Option<(u32, usize)> {
    fn visit(
        table: FdkHuffmanTable,
        row: usize,
        word: u16,
        code: u32,
        consumed: usize,
        path: &mut Vec<usize>,
    ) -> Option<(u32, usize)> {
        if row >= table.len() || path.contains(&row) {
            return None;
        }
        path.push(row);
        let mut best = None;
        for (branch, &entry) in table[row].iter().enumerate() {
            let candidate = if (entry & 1) != 0 {
                (entry >> 2 == word).then(|| {
                    if (entry & 2) != 0 {
                        ((code << 1) | (branch as u32 >> 1), consumed + 1)
                    } else {
                        ((code << 2) | branch as u32, consumed + 2)
                    }
                })
            } else {
                visit(
                    table,
                    (entry >> 2) as usize,
                    word,
                    (code << 2) | branch as u32,
                    consumed + 2,
                    path,
                )
            };
            if let Some(candidate) = candidate {
                best =
                    Some(best.map_or(
                        candidate,
                        |old: (u32, usize)| {
                            if candidate.1 < old.1 {
                                candidate
                            } else {
                                old
                            }
                        },
                    ));
            }
        }
        path.pop();
        best
    }

    visit(table, 0, word, 0, 0, &mut Vec::new())
}

pub fn write_fdk_huffman_word(
    writer: &mut BitWriter,
    table: FdkHuffmanTable,
    word: u16,
) -> Result<usize, HuffmanError> {
    let (code, length) =
        fdk_huffman_word_code(table, word).ok_or(HuffmanError::UnrepresentableWord(word))?;
    writer.write(code, length);
    Ok(length)
}

pub fn write_spectral_tuple(
    writer: &mut BitWriter,
    codebook: u8,
    coefficients: &[i32],
) -> Result<usize, HuffmanError> {
    let description = spectral_codebook(codebook)?;
    if coefficients.len() != usize::from(description.dimension) {
        return Err(HuffmanError::InvalidTupleDimension {
            expected: description.dimension,
            actual: coefficients.len(),
        });
    }
    let expected = spectral_tuple_bit_cost(codebook, coefficients)
        .ok_or(HuffmanError::UnrepresentableTuple(codebook))?;
    let mut word = 0u16;
    for (index, &coefficient) in coefficients.iter().enumerate() {
        let stored = if description.offset == 0 {
            coefficient
                .unsigned_abs()
                .min(if codebook == 11 { 16 } else { u32::MAX })
        } else {
            (coefficient + i32::from(description.offset)) as u32
        };
        word |= (stored as u16) << (usize::from(description.num_bits) * index);
    }
    let start = writer.bits_written();
    write_fdk_huffman_word(writer, description.table, word)?;
    if description.offset == 0 {
        for &coefficient in coefficients {
            if coefficient != 0 {
                writer.write_bool(coefficient < 0);
            }
        }
    }
    if codebook == 11 {
        for &coefficient in coefficients {
            let magnitude = coefficient.unsigned_abs();
            if magnitude >= 16 {
                let exponent = 31 - magnitude.leading_zeros();
                for _ in 4..exponent {
                    writer.write_bool(true);
                }
                writer.write_bool(false);
                writer.write(magnitude - (1 << exponent), exponent as usize);
            }
        }
    }
    debug_assert_eq!(writer.bits_written() - start, expected);
    Ok(expected)
}

/// Decode one word from an FDK 2-bit stepping Huffman table.
///
/// This mirrors `CBlock_DecodeHuffmanWordCB`: each table entry either points to
/// another row (`entry & 1 == 0`) or contains a leaf value (`entry & 1 != 0`).
/// Leaf entries with bit 1 set push one bit back into the bitstream.
pub fn decode_fdk_2bit(
    reader: &mut BitReader<'_>,
    table: FdkHuffmanTable,
) -> Result<u16, HuffmanError> {
    let mut index = 0usize;
    loop {
        if index >= table.len() {
            return Err(HuffmanError::InvalidTableIndex(index));
        }
        // FDK's tables encode a one-bit leaf by duplicating it in both entries
        // selected by the second look-ahead bit and marking the leaf so that
        // the decoder pushes that bit back.  At the exact end of a bounded
        // access unit there need not be a physical look-ahead bit.  Accept the
        // single available bit only when both possible continuations are the
        // same one-bit leaf; otherwise the input is genuinely truncated.
        if reader.remaining_bits() == 1 {
            let first = reader.read_u8(1)? as usize;
            let low = table[index][first << 1];
            let high = table[index][(first << 1) | 1];
            if low == high && (low & 3) == 3 {
                return Ok(low >> 2);
            }
            return Err(BitError::UnexpectedEof {
                needed_bits: 2,
                remaining_bits: 1,
            }
            .into());
        }
        let bits = reader.read_u8(2)? as usize;
        let entry = table[index][bits];
        if (entry & 1) != 0 {
            if (entry & 2) != 0 {
                reader.push_back(1)?;
            }
            return Ok(entry >> 2);
        }
        index = (entry >> 2) as usize;
    }
}

pub const HUFFMAN_CODEBOOK_SCL: [[u16; 4]; 65] = [
    [0x00f3, 0x00f3, 0x0004, 0x0008],
    [0x00ef, 0x00ef, 0x00f5, 0x00e9],
    [0x00f9, 0x000c, 0x0010, 0x0014],
    [0x00e7, 0x00e7, 0x00ff, 0x00ff],
    [0x00e1, 0x0101, 0x00dd, 0x0105],
    [0x0018, 0x001c, 0x0020, 0x0028],
    [0x010b, 0x010b, 0x00db, 0x00db],
    [0x010f, 0x010f, 0x00d5, 0x0111],
    [0x00d1, 0x0115, 0x00cd, 0x0024],
    [0x011b, 0x011b, 0x00cb, 0x00cb],
    [0x002c, 0x0030, 0x0034, 0x0040],
    [0x00c7, 0x00c7, 0x011f, 0x011f],
    [0x0121, 0x00c1, 0x0125, 0x00bd],
    [0x0129, 0x00b9, 0x0038, 0x003c],
    [0x0133, 0x0133, 0x012f, 0x012f],
    [0x0137, 0x0137, 0x013b, 0x013b],
    [0x0044, 0x0048, 0x004c, 0x0058],
    [0x00b7, 0x00b7, 0x00af, 0x00af],
    [0x00b1, 0x013d, 0x00a9, 0x00a5],
    [0x0141, 0x00a1, 0x0050, 0x0054],
    [0x0147, 0x0147, 0x009f, 0x009f],
    [0x014b, 0x014b, 0x009b, 0x009b],
    [0x005c, 0x0060, 0x0064, 0x0070],
    [0x014f, 0x014f, 0x0095, 0x008d],
    [0x0155, 0x0085, 0x0091, 0x0089],
    [0x0151, 0x0081, 0x0068, 0x006c],
    [0x015f, 0x015f, 0x0167, 0x0167],
    [0x007b, 0x007b, 0x007f, 0x007f],
    [0x0074, 0x0078, 0x0080, 0x00b0],
    [0x0159, 0x0075, 0x0069, 0x006d],
    [0x0071, 0x0061, 0x0161, 0x007c],
    [0x0067, 0x0067, 0x005b, 0x005b],
    [0x0084, 0x0088, 0x008c, 0x009c],
    [0x005f, 0x005f, 0x0169, 0x0055],
    [0x004d, 0x000d, 0x0005, 0x0009],
    [0x0001, 0x0090, 0x0094, 0x0098],
    [0x018b, 0x018b, 0x018f, 0x018f],
    [0x0193, 0x0193, 0x0197, 0x0197],
    [0x019b, 0x019b, 0x01d7, 0x01d7],
    [0x00a0, 0x00a4, 0x00a8, 0x00ac],
    [0x0187, 0x0187, 0x016f, 0x016f],
    [0x0173, 0x0173, 0x0177, 0x0177],
    [0x017b, 0x017b, 0x017f, 0x017f],
    [0x0183, 0x0183, 0x01a3, 0x01a3],
    [0x00b4, 0x00c8, 0x00dc, 0x00f0],
    [0x00b8, 0x00bc, 0x00c0, 0x00c4],
    [0x01bf, 0x01bf, 0x01c3, 0x01c3],
    [0x01c7, 0x01c7, 0x01cb, 0x01cb],
    [0x01cf, 0x01cf, 0x01d3, 0x01d3],
    [0x01bb, 0x01bb, 0x01a7, 0x01a7],
    [0x00cc, 0x00d0, 0x00d4, 0x00d8],
    [0x01ab, 0x01ab, 0x01af, 0x01af],
    [0x01b3, 0x01b3, 0x01b7, 0x01b7],
    [0x01db, 0x01db, 0x001b, 0x001b],
    [0x0023, 0x0023, 0x0027, 0x0027],
    [0x00e0, 0x00e4, 0x00e8, 0x00ec],
    [0x002b, 0x002b, 0x0017, 0x0017],
    [0x019f, 0x019f, 0x01e3, 0x01e3],
    [0x01df, 0x01df, 0x0013, 0x0013],
    [0x001f, 0x001f, 0x003f, 0x003f],
    [0x00f4, 0x00f8, 0x00fc, 0x0100],
    [0x0043, 0x0043, 0x004b, 0x004b],
    [0x0053, 0x0053, 0x0047, 0x0047],
    [0x002f, 0x002f, 0x0033, 0x0033],
    [0x003b, 0x003b, 0x0037, 0x0037],
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HuffmanError {
    Bit(BitError),
    InvalidCodebook(u8),
    InvalidTableIndex(usize),
    UnrepresentableWord(u16),
    UnrepresentableTuple(u8),
    InvalidTupleDimension { expected: u8, actual: usize },
}

impl From<BitError> for HuffmanError {
    fn from(value: BitError) -> Self {
        Self::Bit(value)
    }
}

impl fmt::Display for HuffmanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bit(err) => err.fmt(f),
            Self::InvalidCodebook(codebook) => write!(f, "invalid AAC Huffman codebook {codebook}"),
            Self::InvalidTableIndex(index) => write!(f, "invalid Huffman table index {index}"),
            Self::UnrepresentableWord(word) => {
                write!(f, "Huffman word {word:#x} is absent from the table")
            }
            Self::UnrepresentableTuple(codebook) => write!(
                f,
                "spectral tuple is not representable by codebook {codebook}"
            ),
            Self::InvalidTupleDimension { expected, actual } => write!(
                f,
                "spectral codebook expects {expected} coefficients, got {actual}"
            ),
        }
    }
}

impl std::error::Error for HuffmanError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_scalefactor_mid_value_with_pushback() {
        let mut reader = BitReader::new(&[0b0000_0000]);
        assert_eq!(
            decode_fdk_2bit(&mut reader, &HUFFMAN_CODEBOOK_SCL).unwrap(),
            60
        );
        assert_eq!(reader.bits_read(), 1);
    }

    #[test]
    fn decodes_scalefactor_neighbor_values() {
        let mut reader = BitReader::new(&[0b1000_0000]);
        assert_eq!(
            decode_fdk_2bit(&mut reader, &HUFFMAN_CODEBOOK_SCL).unwrap(),
            59
        );
        assert_eq!(reader.bits_read(), 3);

        let mut reader = BitReader::new(&[0b1100_0000]);
        assert_eq!(
            decode_fdk_2bit(&mut reader, &HUFFMAN_CODEBOOK_SCL).unwrap(),
            62
        );
        assert_eq!(reader.bits_read(), 4);
    }

    #[test]
    fn exposes_spectral_codebook_metadata_like_fdk() {
        let expected = [
            (1, 51, 4, 2, 1),
            (2, 39, 4, 2, 1),
            (3, 39, 4, 2, 0),
            (4, 38, 4, 2, 0),
            (5, 41, 2, 4, 4),
            (6, 40, 2, 4, 4),
            (7, 31, 2, 4, 0),
            (8, 31, 2, 4, 0),
            (9, 84, 2, 4, 0),
            (10, 82, 2, 4, 0),
            (11, 152, 2, 5, 0),
        ];

        for (codebook, rows, dimension, num_bits, offset) in expected {
            let description = spectral_codebook(codebook).unwrap();
            assert_eq!(description.table.len(), rows);
            assert_eq!(description.dimension, dimension);
            assert_eq!(description.num_bits, num_bits);
            assert_eq!(description.offset, offset);
        }

        assert_eq!(
            spectral_codebook(0).unwrap_err(),
            HuffmanError::InvalidCodebook(0)
        );
        assert_eq!(
            spectral_codebook(12).unwrap_err(),
            HuffmanError::InvalidCodebook(12)
        );
    }

    #[test]
    fn spectral_codebooks_decode_zero_prefix_without_table_errors() {
        for codebook in 1..=11 {
            let description = spectral_codebook(codebook).unwrap();
            let mut reader = BitReader::new(&[0; 16]);
            let word = decode_fdk_2bit(&mut reader, description.table).unwrap();
            assert!(
                reader.bits_read() > 0,
                "codebook {codebook} consumed no bits"
            );
            assert!(
                reader.bits_read() <= 32,
                "codebook {codebook} consumed too many bits"
            );
            assert!(
                word <= 0x03ff || codebook == 11,
                "codebook {codebook} word {word:#x}"
            );
        }
    }

    #[test]
    fn two_bit_decoder_accepts_a_one_bit_leaf_at_bounded_end() {
        // Values 7 and 9, both encoded in one bit.  Bit 1 in a leaf entry
        // means that the second look-ahead bit is not part of the codeword.
        static ONE_BIT: [[u16; 4]; 1] = [[(7 << 2) | 3, (7 << 2) | 3, (9 << 2) | 3, (9 << 2) | 3]];
        let bytes = [0b1000_0000];
        let mut reader = BitReader::with_bit_len(&bytes, 1).unwrap();
        assert_eq!(decode_fdk_2bit(&mut reader, &ONE_BIT).unwrap(), 9);
        assert_eq!(reader.remaining_bits(), 0);

        static AMBIGUOUS: [[u16; 4]; 1] =
            [[(7 << 2) | 3, (8 << 2) | 1, (9 << 2) | 3, (10 << 2) | 1]];
        let mut reader = BitReader::with_bit_len(&bytes, 1).unwrap();
        assert!(matches!(
            decode_fdk_2bit(&mut reader, &AMBIGUOUS),
            Err(HuffmanError::Bit(BitError::UnexpectedEof { .. }))
        ));
    }

    #[test]
    fn spectral_tuple_cost_includes_sign_and_escape_extensions() {
        let positive = spectral_tuple_bit_cost(7, &[3, 0]).unwrap();
        let negative = spectral_tuple_bit_cost(7, &[-3, 0]).unwrap();
        assert_eq!(positive, negative);
        assert!(spectral_tuple_bit_cost(1, &[2, 0, 0, 0]).is_none());
        assert!(spectral_tuple_bit_cost(7, &[8, 0]).is_none());

        let escape_16 = spectral_tuple_bit_cost(11, &[16, 0]).unwrap();
        assert_eq!(spectral_tuple_bit_cost(11, &[17, 0]), Some(escape_16));
        assert_eq!(spectral_tuple_bit_cost(11, &[32, 0]), Some(escape_16 + 2));
        assert!(spectral_tuple_bit_cost(11, &[8192, 0]).is_none());
    }

    #[test]
    fn spectral_tuple_writer_roundtrips_decoder_table_sign_and_escape_bits() {
        use crate::spectral::decode_spectral_tuple;

        for (codebook, tuple) in [
            (1, vec![-1, 0, 1, -1]),
            (4, vec![-2, 1, 0, 2]),
            (7, vec![-3, 0]),
            (10, vec![12, -7]),
            (11, vec![-32, 17]),
        ] {
            let mut writer = BitWriter::new();
            let expected = spectral_tuple_bit_cost(codebook, &tuple).unwrap();
            assert_eq!(
                write_spectral_tuple(&mut writer, codebook, &tuple).unwrap(),
                expected
            );
            let bytes = writer.finish();
            let mut reader = BitReader::new(&bytes);
            assert_eq!(decode_spectral_tuple(&mut reader, codebook).unwrap(), tuple);
            assert_eq!(reader.bits_read(), expected);
        }
    }

    #[test]
    fn rejects_invalid_tuple_shapes_and_unrepresentable_values() {
        assert_eq!(spectral_tuple_bit_cost(0, &[0]), None);
        assert_eq!(spectral_tuple_bit_cost(1, &[0, 0]), None);
        assert_eq!(spectral_tuple_bit_cost(1, &[-2, 0, 0, 0]), None);
        assert_eq!(spectral_tuple_bit_cost(1, &[1, 1, 1, 2]), None);
        assert_eq!(spectral_tuple_bit_cost(7, &[16, 0]), None);

        let mut writer = BitWriter::new();
        assert_eq!(
            write_spectral_tuple(&mut writer, 7, &[1]),
            Err(HuffmanError::InvalidTupleDimension {
                expected: 2,
                actual: 1
            })
        );
        assert_eq!(
            write_spectral_tuple(&mut writer, 7, &[16, 0]),
            Err(HuffmanError::UnrepresentableTuple(7))
        );
        assert_eq!(
            write_spectral_tuple(&mut writer, 12, &[0, 0]),
            Err(HuffmanError::InvalidCodebook(12))
        );
    }

    #[test]
    fn table_search_handles_cycles_duplicates_and_missing_words() {
        static CYCLE: [[u16; 4]; 1] = [[0, 0, 0, 0]];
        assert_eq!(fdk_huffman_word_code(&CYCLE, 0), None);

        // The same value is reachable as both a two-bit and pushback one-bit leaf.
        static DUPLICATE: [[u16; 4]; 1] = [[0x15, 0x17, 0x15, 0x15]];
        assert_eq!(fdk_huffman_word_code(&DUPLICATE, 5), Some((0, 1)));
        assert_eq!(fdk_huffman_word_length(&DUPLICATE, 5), Some(1));

        let mut writer = BitWriter::new();
        assert_eq!(
            write_fdk_huffman_word(&mut writer, &DUPLICATE, 6),
            Err(HuffmanError::UnrepresentableWord(6))
        );
    }

    #[test]
    fn decoder_rejects_out_of_range_table_links() {
        static INVALID_LINK: [[u16; 4]; 1] = [[4, 4, 4, 4]];
        assert_eq!(
            decode_fdk_2bit(&mut BitReader::new(&[0]), &INVALID_LINK),
            Err(HuffmanError::InvalidTableIndex(1))
        );
        assert!(matches!(
            decode_fdk_2bit(&mut BitReader::new(&[]), &HUFFMAN_CODEBOOK_SCL),
            Err(HuffmanError::Bit(BitError::UnexpectedEof { .. }))
        ));
    }

    #[test]
    fn huffman_errors_have_diagnostics() {
        let errors = [
            HuffmanError::InvalidCodebook(12),
            HuffmanError::InvalidTableIndex(9),
            HuffmanError::UnrepresentableWord(0x123),
            HuffmanError::UnrepresentableTuple(7),
            HuffmanError::InvalidTupleDimension {
                expected: 4,
                actual: 2,
            },
            HuffmanError::from(BitError::UnexpectedEof {
                needed_bits: 2,
                remaining_bits: 0,
            }),
        ];
        for error in errors {
            assert!(!error.to_string().is_empty());
        }
    }
}
