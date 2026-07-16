//! Unified Speech and Audio Coding frame-side-information helpers.

use std::fmt;

use crate::bits::{BitError, BitReader};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UsacCoreMode {
    FrequencyDomain,
    LinearPredictionDomain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LpdDivisionMode {
    Acelp20,
    Tcx20,
    Tcx40,
    Tcx80,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LpdChannelSideInfo {
    pub acelp_core_mode: u8,
    pub lpd_mode: u8,
    pub divisions: [LpdDivisionMode; 4],
    pub bpf_control_info: bool,
    pub previous_frame_was_lpd: bool,
    pub fac_data_present: bool,
    pub bits_read: usize,
}

impl LpdChannelSideInfo {
    pub fn parse(reader: &mut BitReader<'_>) -> Result<Self, UsacError> {
        let start = reader.bits_read();
        let acelp_core_mode = reader.read_u8(3)?;
        let lpd_mode = reader.read_u8(5)?;
        let divisions = map_lpd_mode(lpd_mode)?;
        let bpf_control_info = reader.read_bool()?;
        let previous_frame_was_lpd = reader.read_bool()?;
        let fac_data_present = reader.read_bool()?;
        Ok(Self {
            acelp_core_mode,
            lpd_mode,
            divisions,
            bpf_control_info,
            previous_frame_was_lpd,
            fac_data_present,
            bits_read: reader.bits_read() - start,
        })
    }
}

pub fn map_lpd_mode(mode: u8) -> Result<[LpdDivisionMode; 4], UsacError> {
    use LpdDivisionMode::{Acelp20, Tcx20, Tcx40, Tcx80};
    if mode > 25 {
        return Err(UsacError::InvalidLpdMode(mode));
    }
    Ok(match mode {
        25 => [Tcx80; 4],
        24 => [Tcx40; 4],
        16..=19 => [
            Tcx40,
            Tcx40,
            if mode & 1 != 0 { Tcx20 } else { Acelp20 },
            if mode & 2 != 0 { Tcx20 } else { Acelp20 },
        ],
        20..=23 => [
            if mode & 1 != 0 { Tcx20 } else { Acelp20 },
            if mode & 2 != 0 { Tcx20 } else { Acelp20 },
            Tcx40,
            Tcx40,
        ],
        _ => [
            if mode & 1 != 0 { Tcx20 } else { Acelp20 },
            if mode & 2 != 0 { Tcx20 } else { Acelp20 },
            if mode & 4 != 0 { Tcx20 } else { Acelp20 },
            if mode & 8 != 0 { Tcx20 } else { Acelp20 },
        ],
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UsacFrameElementModes {
    pub independency_flag: bool,
    pub core_modes: Vec<UsacCoreMode>,
    pub bits_read: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UsacWindowSequence {
    OnlyLong,
    LongStart,
    EightShort,
    LongStop,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UsacIcsInfo {
    pub window_sequence: UsacWindowSequence,
    pub window_shape: bool,
    pub max_sfb: u8,
    pub scale_factor_grouping: Option<u8>,
    pub window_group_lengths: Vec<u8>,
}

impl UsacIcsInfo {
    pub fn parse(
        reader: &mut BitReader<'_>,
        long_sfb_count: u8,
        short_sfb_count: u8,
    ) -> Result<Self, UsacError> {
        let window_sequence = match reader.read_u8(2)? {
            0 => UsacWindowSequence::OnlyLong,
            1 => UsacWindowSequence::LongStart,
            2 => UsacWindowSequence::EightShort,
            _ => UsacWindowSequence::LongStop,
        };
        let window_shape = reader.read_bool()?;
        let short = window_sequence == UsacWindowSequence::EightShort;
        let max_sfb = reader.read_u8(if short { 4 } else { 6 })?;
        let maximum = if short {
            short_sfb_count
        } else {
            long_sfb_count
        };
        if max_sfb > maximum {
            return Err(UsacError::MaxSfbOutOfRange { max_sfb, maximum });
        }
        let (scale_factor_grouping, window_group_lengths) = if short {
            let grouping = reader.read_u8(7)?;
            let mut lengths = vec![1u8];
            for bit in (0..7).rev() {
                if grouping & (1 << bit) != 0 {
                    *lengths.last_mut().unwrap() += 1;
                } else {
                    lengths.push(1);
                }
            }
            (Some(grouping), lengths)
        } else {
            (None, vec![1])
        };
        Ok(Self {
            window_sequence,
            window_shape,
            max_sfb,
            scale_factor_grouping,
            window_group_lengths,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UsacFdChannelSideInfo {
    pub tns_data_present: bool,
    pub global_gain: u8,
    pub noise_level_and_offset: Option<u8>,
    pub ics: UsacIcsInfo,
    pub bits_read: usize,
}

impl UsacFdChannelSideInfo {
    pub fn parse(
        reader: &mut BitReader<'_>,
        noise_filling: bool,
        long_sfb_count: u8,
        short_sfb_count: u8,
    ) -> Result<Self, UsacError> {
        let start = reader.bits_read();
        let tns_data_present = reader.read_bool()?;
        let global_gain = reader.read_u8(8)?;
        let noise_level_and_offset = noise_filling.then(|| reader.read_u8(8)).transpose()?;
        let ics = UsacIcsInfo::parse(reader, long_sfb_count, short_sfb_count)?;
        Ok(Self {
            tns_data_present,
            global_gain,
            noise_level_and_offset,
            ics,
            bits_read: reader.bits_read() - start,
        })
    }
}

impl UsacFrameElementModes {
    pub fn parse(
        reader: &mut BitReader<'_>,
        channel_elements: &[usize],
    ) -> Result<Self, UsacError> {
        let start = reader.bits_read();
        let independency_flag = reader.read_bool()?;
        let mut core_modes = Vec::new();
        for &channels in channel_elements {
            if !matches!(channels, 1 | 2) {
                return Err(UsacError::InvalidElementChannelCount(channels));
            }
            for _ in 0..channels {
                core_modes.push(if reader.read_bool()? {
                    UsacCoreMode::LinearPredictionDomain
                } else {
                    UsacCoreMode::FrequencyDomain
                });
            }
        }
        Ok(Self {
            independency_flag,
            core_modes,
            bits_read: reader.bits_read() - start,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UsacError {
    Bit(BitError),
    InvalidLpdMode(u8),
    InvalidElementChannelCount(usize),
    MaxSfbOutOfRange { max_sfb: u8, maximum: u8 },
}

impl From<BitError> for UsacError {
    fn from(value: BitError) -> Self {
        Self::Bit(value)
    }
}

impl fmt::Display for UsacError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bit(error) => error.fmt(f),
            Self::InvalidLpdMode(mode) => write!(f, "invalid USAC lpd_mode {mode}"),
            Self::InvalidElementChannelCount(count) => {
                write!(f, "invalid USAC element channel count {count}")
            }
            Self::MaxSfbOutOfRange { max_sfb, maximum } => {
                write!(f, "USAC max_sfb {max_sfb} exceeds {maximum}")
            }
        }
    }
}

impl std::error::Error for UsacError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitWriter;

    #[test]
    fn maps_all_valid_fdk_lpd_modes() {
        for mode in 0..=25 {
            assert_eq!(map_lpd_mode(mode).unwrap().len(), 4);
        }
        assert_eq!(map_lpd_mode(24).unwrap(), [LpdDivisionMode::Tcx40; 4]);
        assert_eq!(map_lpd_mode(25).unwrap(), [LpdDivisionMode::Tcx80; 4]);
        assert_eq!(map_lpd_mode(19).unwrap()[..2], [LpdDivisionMode::Tcx40; 2]);
        assert_eq!(map_lpd_mode(23).unwrap()[2..], [LpdDivisionMode::Tcx40; 2]);
        assert_eq!(map_lpd_mode(26), Err(UsacError::InvalidLpdMode(26)));
    }

    #[test]
    fn parses_lpd_channel_side_information() {
        let mut writer = BitWriter::new();
        writer.write(5, 3);
        writer.write(25, 5);
        writer.write_bool(true);
        writer.write_bool(false);
        writer.write_bool(true);
        let info = LpdChannelSideInfo::parse(&mut BitReader::new(&writer.finish())).unwrap();
        assert_eq!(info.acelp_core_mode, 5);
        assert_eq!(info.divisions, [LpdDivisionMode::Tcx80; 4]);
        assert!(info.bpf_control_info);
        assert!(!info.previous_frame_was_lpd);
        assert!(info.fac_data_present);
        assert_eq!(info.bits_read, 11);
    }

    #[test]
    fn parses_independency_and_per_channel_core_modes() {
        let mut writer = BitWriter::new();
        writer.write_bool(true);
        writer.write_bool(false);
        writer.write_bool(true);
        writer.write_bool(false);
        let modes =
            UsacFrameElementModes::parse(&mut BitReader::new(&writer.finish()), &[1, 2]).unwrap();
        assert!(modes.independency_flag);
        assert_eq!(
            modes.core_modes,
            vec![
                UsacCoreMode::FrequencyDomain,
                UsacCoreMode::LinearPredictionDomain,
                UsacCoreMode::FrequencyDomain,
            ]
        );
    }

    #[test]
    fn parses_usac_short_fd_channel_side_information() {
        let mut writer = BitWriter::new();
        writer.write_bool(true); // TNS present
        writer.write(180, 8);
        writer.write(0x9a, 8); // noise level and offset
        writer.write(2, 2); // EIGHT_SHORT
        writer.write_bool(true);
        writer.write(8, 4);
        writer.write(0b110_1010, 7);
        let info =
            UsacFdChannelSideInfo::parse(&mut BitReader::new(&writer.finish()), true, 49, 14)
                .unwrap();
        assert!(info.tns_data_present);
        assert_eq!(info.global_gain, 180);
        assert_eq!(info.noise_level_and_offset, Some(0x9a));
        assert_eq!(info.ics.window_sequence, UsacWindowSequence::EightShort);
        assert_eq!(info.ics.window_group_lengths.iter().sum::<u8>(), 8);
        assert_eq!(info.bits_read, 31);
    }

    #[test]
    fn parses_all_long_window_sequences_and_short_grouping_extremes() {
        for (bits, expected) in [
            (0, UsacWindowSequence::OnlyLong),
            (1, UsacWindowSequence::LongStart),
            (3, UsacWindowSequence::LongStop),
        ] {
            let mut writer = BitWriter::new();
            writer.write(bits, 2);
            writer.write_bool(false);
            writer.write(10, 6);
            let info = UsacIcsInfo::parse(&mut BitReader::new(&writer.finish()), 49, 14).unwrap();
            assert_eq!(info.window_sequence, expected);
            assert_eq!(info.scale_factor_grouping, None);
            assert_eq!(info.window_group_lengths, [1]);
        }
        for (grouping, expected) in [(0, vec![1; 8]), (0x7f, vec![8])] {
            let mut writer = BitWriter::new();
            writer.write(2, 2);
            writer.write_bool(true);
            writer.write(0, 4);
            writer.write(grouping, 7);
            let info = UsacIcsInfo::parse(&mut BitReader::new(&writer.finish()), 49, 14).unwrap();
            assert_eq!(info.window_group_lengths, expected);
            assert_eq!(info.scale_factor_grouping, Some(grouping as u8));
        }
    }

    #[test]
    fn ics_rejects_long_and_short_max_sfb_and_eof() {
        for (sequence, max_sfb, maximum) in [(0, 50, 49), (2, 15, 14)] {
            let mut writer = BitWriter::new();
            writer.write(sequence, 2);
            writer.write_bool(false);
            writer.write(max_sfb, if sequence == 2 { 4 } else { 6 });
            assert_eq!(
                UsacIcsInfo::parse(&mut BitReader::new(&writer.finish()), 49, 14),
                Err(UsacError::MaxSfbOutOfRange {
                    max_sfb: max_sfb as u8,
                    maximum
                })
            );
        }
        assert!(matches!(
            UsacIcsInfo::parse(&mut BitReader::new(&[]), 49, 14),
            Err(UsacError::Bit(BitError::UnexpectedEof { .. }))
        ));
        assert!(matches!(
            LpdChannelSideInfo::parse(&mut BitReader::new(&[])),
            Err(UsacError::Bit(BitError::UnexpectedEof { .. }))
        ));
    }

    #[test]
    fn fd_side_info_parses_without_noise_filling() {
        let mut writer = BitWriter::new();
        writer.write_bool(false);
        writer.write(100, 8);
        writer.write(0, 2);
        writer.write_bool(false);
        writer.write(0, 6);
        let info =
            UsacFdChannelSideInfo::parse(&mut BitReader::new(&writer.finish()), false, 49, 14)
                .unwrap();
        assert!(!info.tns_data_present);
        assert_eq!(info.global_gain, 100);
        assert_eq!(info.noise_level_and_offset, None);
        assert_eq!(info.bits_read, 18);
    }

    #[test]
    fn frame_modes_support_empty_layout_and_reject_invalid_channel_counts() {
        let empty = UsacFrameElementModes::parse(&mut BitReader::new(&[0]), &[]).unwrap();
        assert!(!empty.independency_flag);
        assert!(empty.core_modes.is_empty());
        assert_eq!(empty.bits_read, 1);
        for count in [0, 3] {
            assert_eq!(
                UsacFrameElementModes::parse(&mut BitReader::new(&[0]), &[count]),
                Err(UsacError::InvalidElementChannelCount(count))
            );
        }
        assert!(matches!(
            UsacFrameElementModes::parse(&mut BitReader::new(&[]), &[1]),
            Err(UsacError::Bit(BitError::UnexpectedEof { .. }))
        ));
    }

    #[test]
    fn error_conversion_and_messages_cover_all_variants() {
        let bit = BitError::UnexpectedEof {
            needed_bits: 1,
            remaining_bits: 0,
        };
        assert_eq!(UsacError::from(bit.clone()), UsacError::Bit(bit.clone()));
        assert_eq!(UsacError::Bit(bit.clone()).to_string(), bit.to_string());
        let errors = [
            UsacError::InvalidLpdMode(26),
            UsacError::InvalidElementChannelCount(3),
            UsacError::MaxSfbOutOfRange {
                max_sfb: 15,
                maximum: 14,
            },
        ];
        assert!(errors.iter().all(|error| !error.to_string().is_empty()));
    }
}
