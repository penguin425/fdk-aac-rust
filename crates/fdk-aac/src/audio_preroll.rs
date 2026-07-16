//! MPEG-H/USAC AudioPreRoll extension payload parsing.

use std::fmt;

use crate::bits::{BitError, BitReader};

pub const MAX_USAC_PREROLL_ACCESS_UNITS: usize = 3;
pub const MAX_USAC_PREROLL_CONFIG_BYTES: usize = 512;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioPreRoll {
    pub config: Vec<u8>,
    pub apply_crossfade: bool,
    pub access_units: Vec<Vec<u8>>,
    pub bits_read: usize,
}

impl AudioPreRoll {
    pub fn parse(payload: &[u8]) -> Result<Self, AudioPreRollError> {
        let mut reader = BitReader::new(payload);
        let config_length = read_escaped_value(&mut reader, 4, 4, 8)? as usize;
        if config_length > MAX_USAC_PREROLL_CONFIG_BYTES {
            return Err(AudioPreRollError::ConfigTooLarge(config_length));
        }
        let config = read_bytes(&mut reader, config_length)?;
        let apply_crossfade = reader.read_bool()?;
        reader.read_bool()?; // reserved
        let access_unit_count = read_escaped_value(&mut reader, 2, 4, 0)? as usize;
        if access_unit_count > MAX_USAC_PREROLL_ACCESS_UNITS {
            return Err(AudioPreRollError::TooManyAccessUnits(access_unit_count));
        }
        let mut access_units = Vec::with_capacity(access_unit_count);
        for index in 0..access_unit_count {
            let length = read_escaped_value(&mut reader, 16, 16, 0)? as usize;
            if length == 0 {
                return Err(AudioPreRollError::EmptyAccessUnit(index));
            }
            let access_unit = read_bytes(&mut reader, length)?;
            if index == 0 && access_unit.first().is_some_and(|byte| byte & 0x80 == 0) {
                return Err(AudioPreRollError::FirstAccessUnitNotIndependent(index));
            }
            access_units.push(access_unit);
        }
        if !reader.remaining_bits_are_zero() {
            return Err(AudioPreRollError::NonZeroTrailingBits(
                reader.remaining_bits(),
            ));
        }
        Ok(Self {
            config,
            apply_crossfade,
            access_units,
            bits_read: reader.bits_read(),
        })
    }
}

fn read_bytes(reader: &mut BitReader<'_>, length: usize) -> Result<Vec<u8>, BitError> {
    (0..length).map(|_| reader.read_u8(8)).collect()
}

fn read_escaped_value(
    reader: &mut BitReader<'_>,
    first_bits: u8,
    second_bits: u8,
    third_bits: u8,
) -> Result<u32, BitError> {
    let first_max = (1u32 << first_bits) - 1;
    let mut value = reader.read(first_bits.into())?;
    if value != first_max || second_bits == 0 {
        return Ok(value);
    }
    let second_max = (1u32 << second_bits) - 1;
    let second = reader.read(second_bits.into())?;
    value += second;
    if second == second_max && third_bits != 0 {
        value += reader.read(third_bits.into())?;
    }
    Ok(value)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioPreRollError {
    Bit(BitError),
    ConfigTooLarge(usize),
    TooManyAccessUnits(usize),
    EmptyAccessUnit(usize),
    FirstAccessUnitNotIndependent(usize),
    NonZeroTrailingBits(usize),
}

impl From<BitError> for AudioPreRollError {
    fn from(value: BitError) -> Self {
        Self::Bit(value)
    }
}

impl fmt::Display for AudioPreRollError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bit(error) => error.fmt(formatter),
            Self::ConfigTooLarge(length) => {
                write!(formatter, "USAC AudioPreRoll config is {length} bytes")
            }
            Self::TooManyAccessUnits(count) => {
                write!(formatter, "USAC AudioPreRoll carries {count} access units")
            }
            Self::EmptyAccessUnit(index) => {
                write!(formatter, "USAC AudioPreRoll access unit {index} is empty")
            }
            Self::FirstAccessUnitNotIndependent(index) => write!(
                formatter,
                "USAC AudioPreRoll access unit {index} is not independent"
            ),
            Self::NonZeroTrailingBits(bits) => write!(
                formatter,
                "USAC AudioPreRoll has {bits} non-zero trailing bit(s)"
            ),
        }
    }
}

impl std::error::Error for AudioPreRollError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitWriter;

    fn write_escaped(writer: &mut BitWriter, value: u32, first: u8, second: u8) {
        let first_max = (1u32 << first) - 1;
        if value < first_max {
            writer.write(value, first.into());
        } else {
            writer.write(first_max, first.into());
            writer.write(value - first_max, second.into());
        }
    }

    #[test]
    fn parses_config_crossfade_and_independent_access_units() {
        let mut writer = BitWriter::new();
        write_escaped(&mut writer, 2, 4, 4);
        writer.write(0x12, 8);
        writer.write(0x34, 8);
        writer.write_bool(true);
        writer.write_bool(false);
        write_escaped(&mut writer, 2, 2, 4);
        write_escaped(&mut writer, 2, 16, 16);
        writer.write(0x80, 8);
        writer.write(0x01, 8);
        write_escaped(&mut writer, 1, 16, 16);
        writer.write(0xff, 8);
        let payload = writer.finish();

        let parsed = AudioPreRoll::parse(&payload).unwrap();
        assert_eq!(parsed.config, [0x12, 0x34]);
        assert!(parsed.apply_crossfade);
        assert_eq!(parsed.access_units, [vec![0x80, 0x01], vec![0xff]]);
    }

    #[test]
    fn rejects_truncation_limits_empty_and_dependent_access_units() {
        assert!(matches!(
            AudioPreRoll::parse(&[]),
            Err(AudioPreRollError::Bit(BitError::UnexpectedEof { .. }))
        ));

        let mut too_many = BitWriter::new();
        too_many.write(0, 4);
        too_many.write(0, 2);
        too_many.write(3, 2);
        too_many.write(1, 4);
        assert_eq!(
            AudioPreRoll::parse(&too_many.finish()),
            Err(AudioPreRollError::TooManyAccessUnits(4))
        );

        let mut empty = BitWriter::new();
        empty.write(0, 4);
        empty.write(0, 2);
        empty.write(1, 2);
        empty.write(0, 16);
        assert_eq!(
            AudioPreRoll::parse(&empty.finish()),
            Err(AudioPreRollError::EmptyAccessUnit(0))
        );

        let mut dependent = BitWriter::new();
        dependent.write(0, 4);
        dependent.write(0, 2);
        dependent.write(1, 2);
        dependent.write(1, 16);
        dependent.write(0, 8);
        assert_eq!(
            AudioPreRoll::parse(&dependent.finish()),
            Err(AudioPreRollError::FirstAccessUnitNotIndependent(0))
        );
    }
}
