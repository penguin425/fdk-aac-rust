//! AAC-LC pulse_data parsing and application.

use std::fmt;

use crate::bits::{BitError, BitReader};
use crate::ics::{IcsInfo, WindowSequence};
use crate::spectral::SpectralData;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PulseData {
    pub present: bool,
    /// Raw `number_pulse` value. The number of pulses is `number_pulse + 1`.
    pub number_pulse: u8,
    pub pulse_start_sfb: u8,
    pub offsets: Vec<u8>,
    pub amplitudes: Vec<u8>,
}

impl PulseData {
    pub fn absent() -> Self {
        Self {
            present: false,
            number_pulse: 0,
            pulse_start_sfb: 0,
            offsets: Vec::new(),
            amplitudes: Vec::new(),
        }
    }

    pub fn parse_aac_lc(
        reader: &mut BitReader<'_>,
        ics: &IcsInfo,
        band_offsets: &[usize],
        frame_len: usize,
    ) -> Result<Self, PulseError> {
        if !reader.read_bool()? {
            return Ok(Self::absent());
        }
        if ics.window_sequence == WindowSequence::EightShort {
            return Err(PulseError::PulseOnShortWindow);
        }

        let number_pulse = reader.read_u8(2)?;
        let pulse_start_sfb = reader.read_u8(6)?;
        if pulse_start_sfb as usize >= ics.max_sfb as usize
            || pulse_start_sfb as usize >= band_offsets.len()
        {
            return Err(PulseError::InvalidPulseStartSfb {
                pulse_start_sfb,
                max_sfb: ics.max_sfb,
            });
        }

        let mut offsets = Vec::with_capacity(number_pulse as usize + 1);
        let mut amplitudes = Vec::with_capacity(number_pulse as usize + 1);
        let mut line = band_offsets[pulse_start_sfb as usize];
        for _ in 0..=number_pulse {
            let offset = reader.read_u8(5)?;
            let amplitude = reader.read_u8(4)?;
            line += offset as usize;
            offsets.push(offset);
            amplitudes.push(amplitude);
        }
        if line >= frame_len {
            return Err(PulseError::PulseLineOutOfRange { line, frame_len });
        }

        Ok(Self {
            present: true,
            number_pulse,
            pulse_start_sfb,
            offsets,
            amplitudes,
        })
    }

    pub fn apply_to_spectral(
        &self,
        spectral: &mut SpectralData,
        band_offsets: &[usize],
    ) -> Result<(), PulseError> {
        if !self.present {
            return Ok(());
        }
        if spectral.windows.len() != 1 {
            return Err(PulseError::PulseOnShortWindow);
        }
        let mut line = *band_offsets.get(self.pulse_start_sfb as usize).ok_or(
            PulseError::InvalidPulseStartSfb {
                pulse_start_sfb: self.pulse_start_sfb,
                max_sfb: band_offsets.len().saturating_sub(1) as u8,
            },
        )?;

        for (&offset, &amplitude) in self.offsets.iter().zip(&self.amplitudes) {
            let frame_len = spectral.windows[0].len();
            line += offset as usize;
            let coef = spectral.windows[0]
                .get_mut(line)
                .ok_or(PulseError::PulseLineOutOfRange { line, frame_len })?;
            if *coef > 0 {
                *coef += amplitude as i32;
            } else {
                *coef -= amplitude as i32;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PulseError {
    Bit(BitError),
    InvalidPulseStartSfb { pulse_start_sfb: u8, max_sfb: u8 },
    PulseLineOutOfRange { line: usize, frame_len: usize },
    PulseOnShortWindow,
}

impl From<BitError> for PulseError {
    fn from(value: BitError) -> Self {
        Self::Bit(value)
    }
}

impl fmt::Display for PulseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bit(err) => write!(f, "pulse bitstream error: {err}"),
            Self::InvalidPulseStartSfb {
                pulse_start_sfb,
                max_sfb,
            } => write!(
                f,
                "invalid pulse start scalefactor band {pulse_start_sfb}, max_sfb={max_sfb}"
            ),
            Self::PulseLineOutOfRange { line, frame_len } => {
                write!(
                    f,
                    "pulse spectral line {line} outside frame length {frame_len}"
                )
            }
            Self::PulseOnShortWindow => {
                write!(f, "AAC-LC pulse_data is only valid for long windows")
            }
        }
    }
}

impl std::error::Error for PulseError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitWriter;
    use crate::ics::{IcsInfo, WindowShape};

    fn long_ics() -> IcsInfo {
        IcsInfo {
            window_sequence: WindowSequence::OnlyLong,
            window_shape: WindowShape::Sine,
            max_sfb: 3,
            total_sfb: 3,
            predictor_data_present: false,
            scale_factor_grouping: 0,
            window_group_lengths: vec![1],
            bits_read: 0,
        }
    }

    #[test]
    fn parses_absent_pulse_data() {
        let mut writer = BitWriter::new();
        writer.write_bool(false);
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        assert_eq!(
            PulseData::parse_aac_lc(&mut reader, &long_ics(), &[0, 4, 8, 12], 16).unwrap(),
            PulseData::absent()
        );
    }

    #[test]
    fn parses_and_applies_pulse_data_like_fdk() {
        let mut writer = BitWriter::new();
        writer.write_bool(true);
        writer.write(1, 2); // two pulses
        writer.write(1, 6); // start at sfb offset 4
        writer.write(2, 5);
        writer.write(3, 4);
        writer.write(1, 5);
        writer.write(2, 4);
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);

        let pulse = PulseData::parse_aac_lc(&mut reader, &long_ics(), &[0, 4, 8, 12], 16).unwrap();
        assert_eq!(pulse.offsets, vec![2, 1]);
        assert_eq!(pulse.amplitudes, vec![3, 2]);

        let mut spectral = SpectralData {
            windows: vec![vec![0; 16]],
        };
        spectral.windows[0][6] = 5;
        spectral.windows[0][7] = -1;
        pulse
            .apply_to_spectral(&mut spectral, &[0, 4, 8, 12])
            .unwrap();
        assert_eq!(spectral.windows[0][6], 8);
        assert_eq!(spectral.windows[0][7], -3);
    }

    #[test]
    fn rejects_pulses_on_short_windows() {
        let mut short = long_ics();
        short.window_sequence = WindowSequence::EightShort;
        let mut reader = BitReader::new(&[0x80]);
        assert_eq!(
            PulseData::parse_aac_lc(&mut reader, &short, &[0, 4], 128),
            Err(PulseError::PulseOnShortWindow)
        );
        let pulse = PulseData {
            present: true,
            number_pulse: 0,
            pulse_start_sfb: 0,
            offsets: vec![0],
            amplitudes: vec![1],
        };
        assert_eq!(
            pulse.apply_to_spectral(
                &mut SpectralData {
                    windows: vec![vec![0; 4]; 8]
                },
                &[0, 4]
            ),
            Err(PulseError::PulseOnShortWindow)
        );
    }

    #[test]
    fn validates_pulse_start_and_accumulated_line() {
        let mut writer = BitWriter::new();
        writer.write_bool(true);
        writer.write(0, 2);
        writer.write(3, 6);
        assert!(matches!(
            PulseData::parse_aac_lc(
                &mut BitReader::new(&writer.finish()),
                &long_ics(),
                &[0, 4, 8, 12],
                16
            ),
            Err(PulseError::InvalidPulseStartSfb { .. })
        ));

        let mut writer = BitWriter::new();
        writer.write_bool(true);
        writer.write(0, 2);
        writer.write(2, 6);
        writer.write(8, 5);
        writer.write(1, 4);
        assert_eq!(
            PulseData::parse_aac_lc(
                &mut BitReader::new(&writer.finish()),
                &long_ics(),
                &[0, 4, 8, 12],
                16
            ),
            Err(PulseError::PulseLineOutOfRange {
                line: 16,
                frame_len: 16
            })
        );
    }

    #[test]
    fn apply_validates_offsets_and_handles_absent_and_zero_coefficients() {
        let mut spectral = SpectralData {
            windows: vec![vec![0; 4]],
        };
        PulseData::absent()
            .apply_to_spectral(&mut spectral, &[])
            .unwrap();
        let pulse = PulseData {
            present: true,
            number_pulse: 0,
            pulse_start_sfb: 1,
            offsets: vec![0],
            amplitudes: vec![2],
        };
        assert!(matches!(
            pulse.apply_to_spectral(&mut spectral, &[0]),
            Err(PulseError::InvalidPulseStartSfb { .. })
        ));
        let pulse = PulseData {
            pulse_start_sfb: 0,
            offsets: vec![4],
            ..pulse.clone()
        };
        assert!(matches!(
            pulse.apply_to_spectral(&mut spectral, &[0]),
            Err(PulseError::PulseLineOutOfRange { .. })
        ));
        let pulse = PulseData {
            offsets: vec![0],
            ..pulse
        };
        pulse.apply_to_spectral(&mut spectral, &[0]).unwrap();
        assert_eq!(spectral.windows[0][0], -2);
    }

    #[test]
    fn propagates_truncated_bits_and_formats_all_errors() {
        assert!(matches!(
            PulseData::parse_aac_lc(&mut BitReader::new(&[]), &long_ics(), &[0], 16),
            Err(PulseError::Bit(BitError::UnexpectedEof { .. }))
        ));
        let bit = BitError::UnexpectedEof {
            needed_bits: 1,
            remaining_bits: 0,
        };
        assert_eq!(PulseError::from(bit.clone()), PulseError::Bit(bit));
        for error in [
            PulseError::InvalidPulseStartSfb {
                pulse_start_sfb: 3,
                max_sfb: 3,
            },
            PulseError::PulseLineOutOfRange {
                line: 16,
                frame_len: 16,
            },
            PulseError::PulseOnShortWindow,
        ] {
            assert!(!error.to_string().is_empty());
        }
        assert!(PulseError::Bit(BitError::UnexpectedEof {
            needed_bits: 1,
            remaining_bits: 0
        })
        .to_string()
        .starts_with("pulse bitstream error:"));
    }
}
