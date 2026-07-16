//! Safe bitstream reader/writer primitives for MPEG audio syntax parsing.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BitError {
    UnexpectedEof {
        needed_bits: usize,
        remaining_bits: usize,
    },
    TooManyBitsRequested {
        requested_bits: usize,
        max_bits: usize,
    },
    InvalidPushBack {
        requested_bits: usize,
        bits_read: usize,
    },
}

impl fmt::Display for BitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::UnexpectedEof {
                needed_bits,
                remaining_bits,
            } => write!(
                f,
                "bitstream too short: need {needed_bits} bits, only {remaining_bits} bits remain"
            ),
            Self::TooManyBitsRequested {
                requested_bits,
                max_bits,
            } => write!(
                f,
                "requested {requested_bits} bits from bitstream, max supported is {max_bits}"
            ),
            Self::InvalidPushBack {
                requested_bits,
                bits_read,
            } => write!(
                f,
                "cannot push back {requested_bits} bits after reading only {bits_read} bits"
            ),
        }
    }
}

impl std::error::Error for BitError {}

#[derive(Debug, Clone)]
pub struct BitReader<'a> {
    bytes: &'a [u8],
    bit_pos: usize,
    bit_len: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            bit_pos: 0,
            bit_len: bytes.len() * 8,
        }
    }

    pub fn with_bit_len(bytes: &'a [u8], bit_len: usize) -> Result<Self, BitError> {
        let available = bytes.len().saturating_mul(8);
        if bit_len > available {
            return Err(BitError::UnexpectedEof {
                needed_bits: bit_len,
                remaining_bits: available,
            });
        }
        Ok(Self {
            bytes,
            bit_pos: 0,
            bit_len,
        })
    }

    pub fn bits_read(&self) -> usize {
        self.bit_pos
    }

    pub fn remaining_bits(&self) -> usize {
        self.bit_len - self.bit_pos
    }

    pub fn remaining_bits_are_zero(&self) -> bool {
        let mut bit_pos = self.bit_pos;
        while bit_pos < self.bit_len {
            let byte = self.bytes[bit_pos / 8];
            let shift = 7 - (bit_pos % 8);
            if ((byte >> shift) & 1) != 0 {
                return false;
            }
            bit_pos += 1;
        }
        true
    }

    pub fn crc_msb(&self, start_bit: usize, end_bit: usize, width: u8, polynomial: u32) -> u32 {
        debug_assert!(width > 0 && width <= 32);
        debug_assert!(start_bit <= end_bit && end_bit <= self.bit_len);
        let mask = if width == 32 {
            u32::MAX
        } else {
            (1u32 << width) - 1
        };
        let top = 1u32 << (width - 1);
        let mut crc = 0u32;
        for bit_pos in start_bit..end_bit {
            let input = ((self.bytes[bit_pos / 8] >> (7 - bit_pos % 8)) & 1) != 0;
            let feedback = (crc & top != 0) ^ input;
            crc = (crc << 1) & mask;
            if feedback {
                crc ^= polynomial & mask;
            }
        }
        crc
    }

    pub fn byte_align(&mut self) {
        let misalignment = self.bit_pos % 8;
        if misalignment != 0 {
            self.bit_pos = (self.bit_pos + 8 - misalignment).min(self.bit_len);
        }
    }

    pub fn push_back(&mut self, bits: usize) -> Result<(), BitError> {
        if bits > self.bit_pos {
            return Err(BitError::InvalidPushBack {
                requested_bits: bits,
                bits_read: self.bit_pos,
            });
        }
        self.bit_pos -= bits;
        Ok(())
    }

    pub fn read_bool(&mut self) -> Result<bool, BitError> {
        Ok(self.read(1)? != 0)
    }

    pub fn read_u8(&mut self, bits: usize) -> Result<u8, BitError> {
        if bits > 8 {
            return Err(BitError::TooManyBitsRequested {
                requested_bits: bits,
                max_bits: 8,
            });
        }
        Ok(self.read(bits)? as u8)
    }

    pub fn read_u16(&mut self, bits: usize) -> Result<u16, BitError> {
        if bits > 16 {
            return Err(BitError::TooManyBitsRequested {
                requested_bits: bits,
                max_bits: 16,
            });
        }
        Ok(self.read(bits)? as u16)
    }

    pub fn read(&mut self, bits: usize) -> Result<u32, BitError> {
        if bits > 32 {
            return Err(BitError::TooManyBitsRequested {
                requested_bits: bits,
                max_bits: 32,
            });
        }
        if self.remaining_bits() < bits {
            return Err(BitError::UnexpectedEof {
                needed_bits: bits,
                remaining_bits: self.remaining_bits(),
            });
        }

        let mut value = 0;
        for _ in 0..bits {
            let byte = self.bytes[self.bit_pos / 8];
            let shift = 7 - (self.bit_pos % 8);
            value = (value << 1) | (((byte >> shift) & 1) as u32);
            self.bit_pos += 1;
        }
        Ok(value)
    }
}

#[derive(Debug, Clone, Default)]
pub struct BitWriter {
    bytes: Vec<u8>,
    bit_pos: usize,
}

impl BitWriter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn bits_written(&self) -> usize {
        self.bit_pos
    }

    pub fn write_bool(&mut self, value: bool) {
        self.write(u32::from(value), 1);
    }

    pub fn write(&mut self, value: u32, bits: usize) {
        debug_assert!(bits <= 32);
        for bit in (0..bits).rev() {
            if self.bit_pos % 8 == 0 {
                self.bytes.push(0);
            }
            if ((value >> bit) & 1) != 0 {
                let index = self.bytes.len() - 1;
                self.bytes[index] |= 1 << (7 - (self.bit_pos % 8));
            }
            self.bit_pos += 1;
        }
    }

    pub fn byte_align(&mut self) {
        let misalignment = self.bit_pos % 8;
        if misalignment != 0 {
            self.write(0, 8 - misalignment);
        }
    }

    pub fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_across_byte_boundaries() {
        let mut reader = BitReader::new(&[0b1010_1100, 0b0110_0000]);
        assert_eq!(reader.read_u8(3).unwrap(), 0b101);
        assert_eq!(reader.read_u8(5).unwrap(), 0b0_1100);
        assert_eq!(reader.read_u8(4).unwrap(), 0b0110);
        assert_eq!(reader.bits_read(), 12);
    }

    #[test]
    fn writes_and_reads_roundtrip() {
        let mut writer = BitWriter::new();
        writer.write(0b101, 3);
        writer.write(0b0_1100, 5);
        writer.write(0b0110, 4);
        let bytes = writer.finish();

        let mut reader = BitReader::new(&bytes);
        assert_eq!(reader.read_u8(3).unwrap(), 0b101);
        assert_eq!(reader.read_u8(5).unwrap(), 0b0_1100);
        assert_eq!(reader.read_u8(4).unwrap(), 0b0110);
    }

    #[test]
    fn byte_align_skips_padding() {
        let mut writer = BitWriter::new();
        writer.write(0b101, 3);
        writer.byte_align();
        writer.write(0xff, 8);

        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        assert_eq!(reader.read_u8(3).unwrap(), 0b101);
        reader.byte_align();
        assert_eq!(reader.read_u8(8).unwrap(), 0xff);
    }

    #[test]
    fn push_back_rewinds_reader() {
        let mut reader = BitReader::new(&[0b1010_0000]);
        assert_eq!(reader.read_u8(3).unwrap(), 0b101);
        reader.push_back(2).unwrap();
        assert_eq!(reader.read_u8(4).unwrap(), 0b0100);
        assert_eq!(reader.bits_read(), 5);
    }

    #[test]
    fn checks_remaining_bits_are_zero() {
        let mut reader = BitReader::new(&[0b1010_0000]);
        reader.read_u8(4).unwrap();
        assert!(reader.remaining_bits_are_zero());

        let mut reader = BitReader::new(&[0b1010_1000]);
        reader.read_u8(4).unwrap();
        assert!(!reader.remaining_bits_are_zero());
    }

    #[test]
    fn exact_bit_length_hides_byte_padding() {
        let mut reader = BitReader::with_bit_len(&[0b1010_0000], 3).unwrap();
        assert_eq!(reader.read_u8(3).unwrap(), 0b101);
        assert!(matches!(
            reader.read_bool(),
            Err(BitError::UnexpectedEof {
                remaining_bits: 0,
                ..
            })
        ));
    }

    #[test]
    fn validates_reader_lengths_widths_and_pushback() {
        assert_eq!(
            BitReader::with_bit_len(&[0], 9).unwrap_err(),
            BitError::UnexpectedEof {
                needed_bits: 9,
                remaining_bits: 8
            }
        );
        let mut reader = BitReader::new(&[0xff, 0x00]);
        assert!(matches!(
            reader.read_u8(9),
            Err(BitError::TooManyBitsRequested { max_bits: 8, .. })
        ));
        assert!(matches!(
            reader.read_u16(17),
            Err(BitError::TooManyBitsRequested { max_bits: 16, .. })
        ));
        assert!(matches!(
            reader.read(33),
            Err(BitError::TooManyBitsRequested { max_bits: 32, .. })
        ));
        assert_eq!(reader.read_u16(16).unwrap(), 0xff00);
        assert!(matches!(
            reader.read_bool(),
            Err(BitError::UnexpectedEof {
                remaining_bits: 0,
                ..
            })
        ));
        assert!(matches!(
            reader.push_back(17),
            Err(BitError::InvalidPushBack { bits_read: 16, .. })
        ));
        reader.push_back(0).unwrap();
    }

    #[test]
    fn zero_width_operations_do_not_advance() {
        let mut reader = BitReader::new(&[]);
        assert_eq!(reader.read(0).unwrap(), 0);
        assert_eq!(reader.read_u8(0).unwrap(), 0);
        assert_eq!(reader.read_u16(0).unwrap(), 0);
        assert_eq!(reader.bits_read(), 0);
        assert_eq!(reader.remaining_bits(), 0);
        assert!(reader.remaining_bits_are_zero());

        let mut writer = BitWriter::new();
        writer.write(u32::MAX, 0);
        writer.byte_align();
        assert_eq!(writer.bits_written(), 0);
        assert!(writer.finish().is_empty());
    }

    #[test]
    fn reader_alignment_clamps_to_an_exact_bit_length() {
        let mut reader = BitReader::with_bit_len(&[0xff], 5).unwrap();
        reader.read_bool().unwrap();
        reader.byte_align();
        assert_eq!(reader.bits_read(), 5);
        reader.byte_align();
        assert_eq!(reader.bits_read(), 5);
    }

    #[test]
    fn computes_msb_crc_for_small_and_full_width_registers() {
        let reader = BitReader::new(&[0b1011_0000]);
        assert_eq!(reader.crc_msb(0, 4, 4, 0b0011), 0b1110);
        let full = reader.crc_msb(0, 8, 32, 0x04c1_1db7);
        assert_ne!(full, 0);
        assert_eq!(reader.crc_msb(2, 2, 8, 0x07), 0);
    }

    #[test]
    fn writes_booleans_and_complete_words() {
        let mut writer = BitWriter::new();
        writer.write_bool(true);
        writer.write_bool(false);
        writer.write(0x3fff_ffff, 30);
        assert_eq!(writer.bits_written(), 32);
        assert_eq!(writer.finish(), [0xbf, 0xff, 0xff, 0xff]);
    }

    #[test]
    fn formats_every_bit_error_variant() {
        assert_eq!(
            BitError::UnexpectedEof {
                needed_bits: 8,
                remaining_bits: 3
            }
            .to_string(),
            "bitstream too short: need 8 bits, only 3 bits remain"
        );
        assert_eq!(
            BitError::TooManyBitsRequested {
                requested_bits: 33,
                max_bits: 32
            }
            .to_string(),
            "requested 33 bits from bitstream, max supported is 32"
        );
        assert_eq!(
            BitError::InvalidPushBack {
                requested_bits: 2,
                bits_read: 1
            }
            .to_string(),
            "cannot push back 2 bits after reading only 1 bits"
        );
    }
}
