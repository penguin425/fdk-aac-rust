//! Pure Rust AAC Individual Channel Stream side-info parsing.
//!
//! This module ports the AAC-LC subset of FDK's `IcsRead` logic: window
//! sequence/shape, `max_sfb`, predictor flag rejection, and short-window grouping.
//! Section/scalefactor/spectral parsing is built on top of this information.

use std::fmt;

use crate::bits::{BitError, BitReader};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowSequence {
    OnlyLong,
    LongStart,
    EightShort,
    LongStop,
}

impl WindowSequence {
    pub fn from_bits(bits: u8) -> Self {
        match bits & 0x03 {
            0 => Self::OnlyLong,
            1 => Self::LongStart,
            2 => Self::EightShort,
            _ => Self::LongStop,
        }
    }

    pub fn bits(self) -> u8 {
        match self {
            Self::OnlyLong => 0,
            Self::LongStart => 1,
            Self::EightShort => 2,
            Self::LongStop => 3,
        }
    }

    pub fn is_long(self) -> bool {
        self != Self::EightShort
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowShape {
    Sine,
    Kbd,
    /// AAC-LD low-overlap sine slope. It is signalled by the same one-bit
    /// value as KBD and selected from the audio object type.
    LowOverlap,
}

impl WindowShape {
    pub fn from_bit(bit: bool) -> Self {
        if bit {
            Self::Kbd
        } else {
            Self::Sine
        }
    }

    pub fn bit(self) -> bool {
        match self {
            Self::Sine => false,
            Self::Kbd | Self::LowOverlap => true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IcsLimits {
    pub long_sfb: u8,
    pub short_sfb: u8,
}

impl IcsLimits {
    /// Conservative AAC-LC limits used until full sampling-rate SFB tables are
    /// ported. FDK's common AAC-LC 1024/128 tables stay within these maxima.
    pub const AAC_LC_MAX: Self = Self {
        long_sfb: 51,
        short_sfb: 15,
    };
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IcsInfo {
    pub window_sequence: WindowSequence,
    pub window_shape: WindowShape,
    pub max_sfb: u8,
    pub total_sfb: u8,
    pub predictor_data_present: bool,
    pub scale_factor_grouping: u8,
    pub window_group_lengths: Vec<u8>,
    pub bits_read: usize,
}

impl IcsInfo {
    pub fn parse_aac_lc(reader: &mut BitReader<'_>, limits: IcsLimits) -> Result<Self, IcsError> {
        let start = reader.bits_read();

        // reserved_bit in GA AAC syntax. FDK consumes and ignores it for AAC-LC.
        let _reserved = reader.read_bool()?;
        let window_sequence = WindowSequence::from_bits(reader.read_u8(2)?);
        let window_shape = WindowShape::from_bit(reader.read_bool()?);

        let total_sfb = if window_sequence.is_long() {
            limits.long_sfb
        } else {
            limits.short_sfb
        };
        let max_sfb_bits = if window_sequence.is_long() { 6 } else { 4 };
        let max_sfb = reader.read_u8(max_sfb_bits)?;
        if max_sfb > total_sfb {
            return Err(IcsError::MaxSfbOutOfRange { max_sfb, total_sfb });
        }

        let mut predictor_data_present = false;
        let mut scale_factor_grouping = 0;
        let window_group_lengths;

        if window_sequence.is_long() {
            predictor_data_present = reader.read_bool()?;
            if predictor_data_present {
                return Err(IcsError::PredictionUnsupported);
            }
            window_group_lengths = vec![1];
        } else {
            scale_factor_grouping = reader.read_u8(7)?;
            window_group_lengths = grouping_to_lengths(scale_factor_grouping);
        }

        Ok(Self {
            window_sequence,
            window_shape,
            max_sfb,
            total_sfb,
            predictor_data_present,
            scale_factor_grouping,
            window_group_lengths,
            bits_read: reader.bits_read() - start,
        })
    }

    pub fn parse_eld(reader: &mut BitReader<'_>, total_sfb: u8) -> Result<Self, IcsError> {
        let start = reader.bits_read();
        let max_sfb = reader.read_u8(6)?;
        if max_sfb > total_sfb {
            return Err(IcsError::MaxSfbOutOfRange { max_sfb, total_sfb });
        }
        Ok(Self {
            window_sequence: WindowSequence::OnlyLong,
            window_shape: WindowShape::Sine,
            max_sfb,
            total_sfb,
            predictor_data_present: false,
            scale_factor_grouping: 0,
            window_group_lengths: vec![1],
            bits_read: reader.bits_read() - start,
        })
    }

    pub fn parse_aac_ld(reader: &mut BitReader<'_>, total_sfb: u8) -> Result<Self, IcsError> {
        let mut ics = Self::parse_aac_lc(
            reader,
            IcsLimits {
                long_sfb: total_sfb,
                short_sfb: 0,
            },
        )?;
        if ics.window_sequence != WindowSequence::OnlyLong {
            return Err(IcsError::LowDelayWindowSequence(ics.window_sequence));
        }
        if ics.window_shape == WindowShape::Kbd {
            ics.window_shape = WindowShape::LowOverlap;
        }
        Ok(ics)
    }
}

pub fn grouping_to_lengths(scale_factor_grouping: u8) -> Vec<u8> {
    let mut lengths = Vec::with_capacity(8);
    lengths.push(1);
    for i in 0..7 {
        let mask = 1 << (6 - i);
        if (scale_factor_grouping & mask) != 0 {
            let last = lengths.last_mut().expect("at least one group");
            *last += 1;
        } else {
            lengths.push(1);
        }
    }
    lengths
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IcsError {
    Bit(BitError),
    MaxSfbOutOfRange { max_sfb: u8, total_sfb: u8 },
    PredictionUnsupported,
    LowDelayWindowSequence(WindowSequence),
}

impl From<BitError> for IcsError {
    fn from(value: BitError) -> Self {
        Self::Bit(value)
    }
}

impl fmt::Display for IcsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bit(err) => err.fmt(f),
            Self::MaxSfbOutOfRange { max_sfb, total_sfb } => {
                write!(f, "ICS max_sfb {max_sfb} exceeds total_sfb {total_sfb}")
            }
            Self::PredictionUnsupported => write!(f, "AAC prediction data is not supported"),
            Self::LowDelayWindowSequence(sequence) => {
                write!(f, "AAC-LD requires ONLY_LONG, found {sequence:?}")
            }
        }
    }
}

impl std::error::Error for IcsError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitWriter;

    #[test]
    fn parses_long_ics_info() {
        let mut writer = BitWriter::new();
        writer.write_bool(false); // reserved
        writer.write(WindowSequence::OnlyLong.bits() as u32, 2);
        writer.write_bool(false); // sine
        writer.write(42, 6);
        writer.write_bool(false); // predictor data absent

        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        let ics = IcsInfo::parse_aac_lc(&mut reader, IcsLimits::AAC_LC_MAX).unwrap();
        assert_eq!(ics.window_sequence, WindowSequence::OnlyLong);
        assert_eq!(ics.window_shape, WindowShape::Sine);
        assert_eq!(ics.max_sfb, 42);
        assert_eq!(ics.total_sfb, 51);
        assert_eq!(ics.window_group_lengths, vec![1]);
        assert_eq!(ics.bits_read, 11);
    }

    #[test]
    fn aac_ld_maps_kbd_signal_to_low_overlap_and_rejects_other_sequences() {
        let mut writer = BitWriter::new();
        writer.write_bool(false);
        writer.write(WindowSequence::OnlyLong.bits() as u32, 2);
        writer.write_bool(true);
        writer.write(12, 6);
        writer.write_bool(false);
        let bytes = writer.finish();
        let ics = IcsInfo::parse_aac_ld(&mut BitReader::new(&bytes), 40).unwrap();
        assert_eq!(ics.window_shape, WindowShape::LowOverlap);
        assert!(ics.window_shape.bit());

        let mut writer = BitWriter::new();
        writer.write_bool(false);
        writer.write(WindowSequence::LongStart.bits() as u32, 2);
        writer.write_bool(false);
        writer.write(0, 6);
        writer.write_bool(false);
        assert!(matches!(
            IcsInfo::parse_aac_ld(&mut BitReader::new(&writer.finish()), 40),
            Err(IcsError::LowDelayWindowSequence(WindowSequence::LongStart))
        ));
    }

    #[test]
    fn parses_short_ics_grouping() {
        let mut writer = BitWriter::new();
        writer.write_bool(false);
        writer.write(WindowSequence::EightShort.bits() as u32, 2);
        writer.write_bool(true);
        writer.write(12, 4);
        writer.write(0b110_0101, 7);

        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        let ics = IcsInfo::parse_aac_lc(&mut reader, IcsLimits::AAC_LC_MAX).unwrap();
        assert_eq!(ics.window_sequence, WindowSequence::EightShort);
        assert_eq!(ics.window_shape, WindowShape::Kbd);
        assert_eq!(ics.max_sfb, 12);
        assert_eq!(ics.window_group_lengths, vec![3, 1, 2, 2]);
        assert_eq!(ics.bits_read, 15);
    }

    #[test]
    fn parses_eld_implicit_long_window() {
        let mut writer = BitWriter::new();
        writer.write(30, 6);
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        let ics = IcsInfo::parse_eld(&mut reader, 35).unwrap();
        assert_eq!(ics.window_sequence, WindowSequence::OnlyLong);
        assert_eq!(ics.window_shape, WindowShape::Sine);
        assert_eq!(ics.max_sfb, 30);
        assert_eq!(ics.bits_read, 6);
    }

    #[test]
    fn rejects_prediction_and_too_large_max_sfb() {
        let mut writer = BitWriter::new();
        writer.write_bool(false);
        writer.write(WindowSequence::OnlyLong.bits() as u32, 2);
        writer.write_bool(false);
        writer.write(52, 6);
        writer.write_bool(false);
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        assert_eq!(
            IcsInfo::parse_aac_lc(&mut reader, IcsLimits::AAC_LC_MAX).unwrap_err(),
            IcsError::MaxSfbOutOfRange {
                max_sfb: 52,
                total_sfb: 51
            }
        );

        let mut writer = BitWriter::new();
        writer.write_bool(false);
        writer.write(WindowSequence::OnlyLong.bits() as u32, 2);
        writer.write_bool(false);
        writer.write(10, 6);
        writer.write_bool(true);
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        assert_eq!(
            IcsInfo::parse_aac_lc(&mut reader, IcsLimits::AAC_LC_MAX).unwrap_err(),
            IcsError::PredictionUnsupported
        );
    }

    #[test]
    fn window_sequence_and_shape_roundtrip_all_values() {
        for (bits, sequence) in [
            (0, WindowSequence::OnlyLong),
            (1, WindowSequence::LongStart),
            (2, WindowSequence::EightShort),
            (3, WindowSequence::LongStop),
            (7, WindowSequence::LongStop),
        ] {
            assert_eq!(WindowSequence::from_bits(bits), sequence);
            assert_eq!(WindowSequence::from_bits(sequence.bits()), sequence);
            assert_eq!(sequence.is_long(), sequence != WindowSequence::EightShort);
        }
        assert_eq!(WindowShape::from_bit(false), WindowShape::Sine);
        assert_eq!(WindowShape::from_bit(true), WindowShape::Kbd);
        assert!(!WindowShape::Sine.bit());
        assert!(WindowShape::Kbd.bit());
    }

    #[test]
    fn grouping_handles_fully_split_and_fully_joined_windows() {
        assert_eq!(grouping_to_lengths(0), vec![1; 8]);
        assert_eq!(grouping_to_lengths(0x7f), vec![8]);
        assert_eq!(grouping_to_lengths(0b1010101), vec![2, 2, 2, 2]);
    }

    #[test]
    fn eld_rejects_max_sfb_and_parsers_propagate_eof() {
        let mut writer = BitWriter::new();
        writer.write(36, 6);
        assert_eq!(
            IcsInfo::parse_eld(&mut BitReader::new(&writer.finish()), 35),
            Err(IcsError::MaxSfbOutOfRange {
                max_sfb: 36,
                total_sfb: 35
            })
        );
        assert!(matches!(
            IcsInfo::parse_eld(&mut BitReader::new(&[]), 35),
            Err(IcsError::Bit(BitError::UnexpectedEof { .. }))
        ));
        assert!(matches!(
            IcsInfo::parse_aac_lc(&mut BitReader::new(&[]), IcsLimits::AAC_LC_MAX),
            Err(IcsError::Bit(BitError::UnexpectedEof { .. }))
        ));
    }

    #[test]
    fn formats_all_ics_errors() {
        let bit = BitError::UnexpectedEof {
            needed_bits: 1,
            remaining_bits: 0,
        };
        assert_eq!(IcsError::from(bit.clone()), IcsError::Bit(bit.clone()));
        assert_eq!(IcsError::Bit(bit.clone()).to_string(), bit.to_string());
        assert_eq!(
            IcsError::MaxSfbOutOfRange {
                max_sfb: 16,
                total_sfb: 15
            }
            .to_string(),
            "ICS max_sfb 16 exceeds total_sfb 15"
        );
        assert_eq!(
            IcsError::PredictionUnsupported.to_string(),
            "AAC prediction data is not supported"
        );
    }
}
