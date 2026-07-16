//! AAC-LC scalefactor band offset tables for 1024/128 frame decoding.

use std::fmt;

use crate::ics::{IcsInfo, WindowSequence};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScaleFactorBandInfo {
    pub offsets: &'static [usize],
    pub num_bands: u8,
    pub granule_length: usize,
}

impl ScaleFactorBandInfo {
    pub fn offsets_for_ics(self, ics: &IcsInfo) -> Result<&'static [usize], SfbError> {
        if ics.max_sfb > self.num_bands {
            return Err(SfbError::MaxSfbOutOfRange {
                max_sfb: ics.max_sfb,
                total_sfb: self.num_bands,
            });
        }
        Ok(self.offsets)
    }
}

pub fn aac_lc_sfb_info(
    sampling_frequency_index: u8,
    window_sequence: WindowSequence,
) -> Result<ScaleFactorBandInfo, SfbError> {
    let long = window_sequence.is_long();
    let (offsets, num_bands): (&'static [usize], u8) = match (sampling_frequency_index, long) {
        (0 | 1, true) => (&SFB_96_1024, 41),
        (0 | 1, false) => (&SFB_96_128, 12),
        (2, true) => (&SFB_64_1024, 47),
        (2, false) => (&SFB_64_128, 12),
        (3 | 4, true) => (&SFB_48_1024, 49),
        (3 | 4, false) => (&SFB_48_128, 14),
        (5, true) => (&SFB_32_1024, 51),
        (5, false) => (&SFB_48_128, 14),
        (6 | 7, true) => (&SFB_24_1024, 47),
        (6 | 7, false) => (&SFB_24_128, 15),
        (8 | 9 | 10, true) => (&SFB_16_1024, 43),
        (8 | 9 | 10, false) => (&SFB_16_128, 15),
        (11 | 12, true) => (&SFB_8_1024, 40),
        (11 | 12, false) => (&SFB_8_128, 15),
        (other, _) => return Err(SfbError::UnsupportedSamplingFrequencyIndex(other)),
    };

    Ok(ScaleFactorBandInfo {
        offsets,
        num_bands,
        granule_length: if long { 1024 } else { 128 },
    })
}

/// Select AAC scalefactor-band geometry for either the regular 1024/128 or
/// reduced 960/120 transform family.
pub fn aac_sfb_info_for_frame(
    sampling_frequency_index: u8,
    window_sequence: WindowSequence,
    frame_length: usize,
) -> Result<ScaleFactorBandInfo, SfbError> {
    if frame_length == 1024 {
        return aac_lc_sfb_info(sampling_frequency_index, window_sequence);
    }
    let ics = IcsInfo {
        window_sequence,
        window_shape: crate::ics::WindowShape::Sine,
        max_sfb: 0,
        total_sfb: 0,
        predictor_data_present: false,
        scale_factor_grouping: 0,
        window_group_lengths: if window_sequence.is_long() {
            vec![1]
        } else {
            vec![1; 8]
        },
        bits_read: 0,
    };
    aac_band_offsets_for_ics(sampling_frequency_index, &ics, frame_length)
}

pub fn aac_lc_band_offsets_for_ics(
    sampling_frequency_index: u8,
    ics: &IcsInfo,
) -> Result<ScaleFactorBandInfo, SfbError> {
    let info = aac_lc_sfb_info(sampling_frequency_index, ics.window_sequence)?;
    info.offsets_for_ics(ics)?;
    Ok(info)
}

pub fn aac_band_offsets_for_ics(
    sampling_frequency_index: u8,
    ics: &IcsInfo,
    frame_length: usize,
) -> Result<ScaleFactorBandInfo, SfbError> {
    if frame_length == 1024 {
        return aac_lc_band_offsets_for_ics(sampling_frequency_index, ics);
    }
    if frame_length == 960 {
        let long = ics.window_sequence.is_long();
        let (offsets, num_bands): (&'static [usize], u8) = match (sampling_frequency_index, long) {
            (0 | 1, true) => (&SFB_96_960, 40),
            (0 | 1, false) => (&SFB_96_120, 12),
            (2, true) => (&SFB_64_960, 46),
            (2, false) => (&SFB_64_120, 12),
            (3 | 4, true) => (&SFB_48_960, 49),
            (3 | 4, false) => (&SFB_48_120, 14),
            (5, true) => (&SFB_32_960, 49),
            (5, false) => (&SFB_48_120, 14),
            (6 | 7, true) => (&SFB_24_960, 46),
            (6 | 7, false) => (&SFB_24_120, 15),
            (8 | 9 | 10, true) => (&SFB_16_960, 42),
            (8 | 9 | 10, false) => (&SFB_16_120, 15),
            (11 | 12, true) => (&SFB_8_960, 40),
            (11 | 12, false) => (&SFB_8_120, 15),
            (other, _) => return Err(SfbError::UnsupportedSamplingFrequencyIndex(other)),
        };
        if ics.max_sfb > num_bands {
            return Err(SfbError::MaxSfbOutOfRange {
                max_sfb: ics.max_sfb,
                total_sfb: num_bands,
            });
        }
        return Ok(ScaleFactorBandInfo {
            offsets,
            num_bands,
            granule_length: if long { 960 } else { 120 },
        });
    }
    if !ics.window_sequence.is_long() {
        return Err(SfbError::UnsupportedFrameLength(frame_length));
    }
    let (offsets, num_bands): (&'static [usize], u8) =
        match (frame_length, sampling_frequency_index) {
            (512, 0..=4) => (&SFB_48_512, 36),
            (512, 5) => (&SFB_32_512, 37),
            (512, 6..=12) => (&SFB_24_512, 31),
            (480, 0..=4) => (&SFB_48_480, 35),
            (480, 5) => (&SFB_32_480, 37),
            (480, 6..=12) => (&SFB_24_480, 30),
            (other, _) => return Err(SfbError::UnsupportedFrameLength(other)),
        };
    if ics.max_sfb > num_bands {
        return Err(SfbError::MaxSfbOutOfRange {
            max_sfb: ics.max_sfb,
            total_sfb: num_bands,
        });
    }
    Ok(ScaleFactorBandInfo {
        offsets,
        num_bands,
        granule_length: frame_length,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SfbError {
    UnsupportedSamplingFrequencyIndex(u8),
    MaxSfbOutOfRange { max_sfb: u8, total_sfb: u8 },
    UnsupportedFrameLength(usize),
}

impl fmt::Display for SfbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSamplingFrequencyIndex(index) => {
                write!(f, "unsupported AAC-LC sampling frequency index {index}")
            }
            Self::MaxSfbOutOfRange { max_sfb, total_sfb } => write!(
                f,
                "AAC ICS max_sfb {max_sfb} exceeds scalefactor band table total {total_sfb}"
            ),
            Self::UnsupportedFrameLength(length) => {
                write!(f, "unsupported AAC scalefactor-band frame length {length}")
            }
        }
    }
}

impl std::error::Error for SfbError {}

pub const SFB_96_1024: [usize; 42] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 64, 72, 80, 88, 96, 108, 120, 132,
    144, 156, 172, 188, 212, 240, 276, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024,
];

pub const SFB_96_128: [usize; 13] = [0, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 92, 128];

pub const SFB_64_1024: [usize; 48] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 64, 72, 80, 88, 100, 112, 124, 140,
    156, 172, 192, 216, 240, 268, 304, 344, 384, 424, 464, 504, 544, 584, 624, 664, 704, 744, 784,
    824, 864, 904, 944, 984, 1024,
];

pub const SFB_64_128: [usize; 13] = [0, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 92, 128];

pub const SFB_48_1024: [usize; 50] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 88, 96, 108, 120, 132, 144, 160,
    176, 196, 216, 240, 264, 292, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704,
    736, 768, 800, 832, 864, 896, 928, 1024,
];

pub const SFB_48_128: [usize; 15] = [0, 4, 8, 12, 16, 20, 28, 36, 44, 56, 68, 80, 96, 112, 128];

pub const SFB_32_1024: [usize; 52] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 88, 96, 108, 120, 132, 144, 160,
    176, 196, 216, 240, 264, 292, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704,
    736, 768, 800, 832, 864, 896, 928, 960, 992, 1024,
];

pub const SFB_24_1024: [usize; 48] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 124, 136,
    148, 160, 172, 188, 204, 220, 240, 260, 284, 308, 336, 364, 396, 432, 468, 508, 552, 600, 652,
    704, 768, 832, 896, 960, 1024,
];

pub const SFB_24_128: [usize; 16] = [
    0, 4, 8, 12, 16, 20, 24, 28, 36, 44, 52, 64, 76, 92, 108, 128,
];

pub const SFB_48_512: [usize; 37] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 68, 76, 84, 92, 100, 112, 124,
    136, 148, 164, 184, 208, 236, 268, 300, 332, 364, 396, 428, 460, 512,
];
pub const SFB_32_512: [usize; 38] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 64, 72, 80, 88, 96, 108, 120, 132,
    144, 160, 176, 192, 212, 236, 260, 288, 320, 352, 384, 416, 448, 480, 512,
];
pub const SFB_24_512: [usize; 32] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 52, 60, 68, 80, 92, 104, 120, 140, 164, 192, 224,
    256, 288, 320, 352, 384, 416, 448, 480, 512,
];
pub const SFB_48_480: [usize; 36] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 64, 72, 80, 88, 96, 108, 120, 132,
    144, 156, 172, 188, 212, 240, 272, 304, 336, 368, 400, 432, 480,
];
pub const SFB_32_480: [usize; 38] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96, 104, 112, 124,
    136, 148, 164, 180, 200, 224, 256, 288, 320, 352, 384, 416, 448, 480,
];
pub const SFB_24_480: [usize; 31] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 52, 60, 68, 80, 92, 104, 120, 140, 164, 192, 224,
    256, 288, 320, 352, 384, 416, 448, 480,
];

pub const SFB_16_1024: [usize; 44] = [
    0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 100, 112, 124, 136, 148, 160, 172, 184, 196, 212,
    228, 244, 260, 280, 300, 320, 344, 368, 396, 424, 456, 492, 532, 572, 616, 664, 716, 772, 832,
    896, 960, 1024,
];

pub const SFB_16_128: [usize; 16] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 60, 72, 88, 108, 128,
];

pub const SFB_8_1024: [usize; 41] = [
    0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 172, 188, 204, 220, 236, 252, 268,
    288, 308, 328, 348, 372, 396, 420, 448, 476, 508, 544, 580, 620, 664, 712, 764, 820, 880, 944,
    1024,
];

pub const SFB_8_128: [usize; 16] = [
    0, 4, 8, 12, 16, 20, 24, 28, 36, 44, 52, 60, 72, 88, 108, 128,
];

pub const SFB_96_960: [usize; 41] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 64, 72, 80, 88, 96, 108, 120, 132,
    144, 156, 172, 188, 212, 240, 276, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960,
];
pub const SFB_96_120: [usize; 13] = [0, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 92, 120];
pub const SFB_64_960: [usize; 47] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 64, 72, 80, 88, 100, 112, 124, 140,
    156, 172, 192, 216, 240, 268, 304, 344, 384, 424, 464, 504, 544, 584, 624, 664, 704, 744, 784,
    824, 864, 904, 944, 960,
];
pub const SFB_64_120: [usize; 13] = SFB_96_120;
pub const SFB_48_960: [usize; 50] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 88, 96, 108, 120, 132, 144, 160,
    176, 196, 216, 240, 264, 292, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704,
    736, 768, 800, 832, 864, 896, 928, 960,
];
pub const SFB_32_960: [usize; 50] = SFB_48_960;
pub const SFB_48_120: [usize; 15] = [0, 4, 8, 12, 16, 20, 28, 36, 44, 56, 68, 80, 96, 112, 120];
pub const SFB_24_960: [usize; 47] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 124, 136,
    148, 160, 172, 188, 204, 220, 240, 260, 284, 308, 336, 364, 396, 432, 468, 508, 552, 600, 652,
    704, 768, 832, 896, 960,
];
pub const SFB_24_120: [usize; 16] = [
    0, 4, 8, 12, 16, 20, 24, 28, 36, 44, 52, 64, 76, 92, 108, 120,
];
pub const SFB_16_960: [usize; 43] = [
    0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 100, 112, 124, 136, 148, 160, 172, 184, 196, 212,
    228, 244, 260, 280, 300, 320, 344, 368, 396, 424, 456, 492, 532, 572, 616, 664, 716, 772, 832,
    896, 960,
];
pub const SFB_16_120: [usize; 16] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 60, 72, 88, 108, 120,
];
pub const SFB_8_960: [usize; 41] = [
    0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 172, 188, 204, 220, 236, 252, 268,
    288, 308, 328, 348, 372, 396, 420, 448, 476, 508, 544, 580, 620, 664, 712, 764, 820, 880, 944,
    960,
];
pub const SFB_8_120: [usize; 16] = SFB_24_120;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ics::{IcsInfo, WindowShape};

    fn ics(window_sequence: WindowSequence, max_sfb: u8) -> IcsInfo {
        IcsInfo {
            window_sequence,
            window_shape: WindowShape::Sine,
            max_sfb,
            total_sfb: max_sfb,
            predictor_data_present: false,
            scale_factor_grouping: 0,
            window_group_lengths: if window_sequence.is_long() {
                vec![1]
            } else {
                vec![1, 1, 1, 1, 1, 1, 1, 1]
            },
            bits_read: 0,
        }
    }

    #[test]
    fn exposes_44100_long_and_short_offsets_like_fdk() {
        let long = aac_lc_sfb_info(4, WindowSequence::OnlyLong).unwrap();
        assert_eq!(long.num_bands, 49);
        assert_eq!(long.granule_length, 1024);
        assert_eq!(
            &long.offsets[..12],
            &[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48]
        );
        assert_eq!(long.offsets[long.num_bands as usize], 1024);

        let short = aac_lc_sfb_info(4, WindowSequence::EightShort).unwrap();
        assert_eq!(short.num_bands, 14);
        assert_eq!(short.granule_length, 128);
        assert_eq!(short.offsets, &SFB_48_128);
        assert_eq!(short.offsets[short.num_bands as usize], 128);
    }

    #[test]
    fn maps_all_aac_lc_sampling_frequency_indices_for_1024_128() {
        let expected_long = [41, 41, 47, 49, 49, 51, 47, 47, 43, 43, 43, 40, 40];
        let expected_short = [12, 12, 12, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15];
        for index in 0u8..=12 {
            assert_eq!(
                aac_lc_sfb_info(index, WindowSequence::OnlyLong)
                    .unwrap()
                    .num_bands,
                expected_long[index as usize]
            );
            assert_eq!(
                aac_lc_sfb_info(index, WindowSequence::EightShort)
                    .unwrap()
                    .num_bands,
                expected_short[index as usize]
            );
        }
    }

    #[test]
    fn rejects_unsupported_index_and_too_large_max_sfb() {
        assert_eq!(
            aac_lc_sfb_info(13, WindowSequence::OnlyLong).unwrap_err(),
            SfbError::UnsupportedSamplingFrequencyIndex(13)
        );

        assert_eq!(
            aac_lc_band_offsets_for_ics(4, &ics(WindowSequence::OnlyLong, 50)).unwrap_err(),
            SfbError::MaxSfbOutOfRange {
                max_sfb: 50,
                total_sfb: 49
            }
        );
    }

    #[test]
    fn exposes_er_aac_ld_512_and_480_offsets() {
        let ld512 = aac_band_offsets_for_ics(4, &ics(WindowSequence::OnlyLong, 36), 512).unwrap();
        assert_eq!(ld512.offsets, &SFB_48_512);
        assert_eq!(ld512.granule_length, 512);
        let ld480 = aac_band_offsets_for_ics(5, &ics(WindowSequence::OnlyLong, 37), 480).unwrap();
        assert_eq!(ld480.offsets, &SFB_32_480);
        assert_eq!(ld480.granule_length, 480);
        assert_eq!(
            aac_band_offsets_for_ics(4, &ics(WindowSequence::EightShort, 1), 512).unwrap_err(),
            SfbError::UnsupportedFrameLength(512)
        );
    }

    #[test]
    fn exposes_drm_aac_960_and_120_offsets() {
        let long = aac_band_offsets_for_ics(3, &ics(WindowSequence::OnlyLong, 49), 960).unwrap();
        assert_eq!(long.offsets, &SFB_48_960);
        assert_eq!(long.granule_length, 960);
        let short = aac_band_offsets_for_ics(3, &ics(WindowSequence::EightShort, 14), 960).unwrap();
        assert_eq!(short.offsets, &SFB_48_120);
        assert_eq!(short.granule_length, 120);
    }

    #[test]
    fn maps_every_960_long_and_short_sampling_table() {
        let expected_long = [40, 40, 46, 49, 49, 49, 46, 46, 42, 42, 42, 40, 40];
        let expected_short = [12, 12, 12, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15];
        for index in 0u8..=12 {
            assert_eq!(
                aac_band_offsets_for_ics(index, &ics(WindowSequence::OnlyLong, 0), 960)
                    .unwrap()
                    .num_bands,
                expected_long[index as usize]
            );
            assert_eq!(
                aac_band_offsets_for_ics(index, &ics(WindowSequence::EightShort, 0), 960)
                    .unwrap()
                    .num_bands,
                expected_short[index as usize]
            );
        }
    }

    #[test]
    fn maps_all_low_delay_frame_and_rate_categories() {
        for (length, index, bands) in [
            (512, 0, 36),
            (512, 5, 37),
            (512, 12, 31),
            (480, 0, 35),
            (480, 5, 37),
            (480, 12, 30),
        ] {
            let info =
                aac_band_offsets_for_ics(index, &ics(WindowSequence::OnlyLong, 0), length).unwrap();
            assert_eq!(info.num_bands, bands);
            assert_eq!(info.granule_length, length);
            assert_eq!(*info.offsets.last().unwrap(), length);
        }
    }

    #[test]
    fn rejects_every_frame_specific_out_of_range_case() {
        assert_eq!(
            aac_band_offsets_for_ics(13, &ics(WindowSequence::OnlyLong, 0), 960),
            Err(SfbError::UnsupportedSamplingFrequencyIndex(13))
        );
        assert!(matches!(
            aac_band_offsets_for_ics(0, &ics(WindowSequence::OnlyLong, 41), 960),
            Err(SfbError::MaxSfbOutOfRange { .. })
        ));
        assert!(matches!(
            aac_band_offsets_for_ics(0, &ics(WindowSequence::OnlyLong, 37), 512),
            Err(SfbError::MaxSfbOutOfRange { .. })
        ));
        assert_eq!(
            aac_band_offsets_for_ics(0, &ics(WindowSequence::OnlyLong, 0), 256),
            Err(SfbError::UnsupportedFrameLength(256))
        );
    }

    #[test]
    fn formats_all_sfb_errors() {
        assert_eq!(
            SfbError::UnsupportedSamplingFrequencyIndex(13).to_string(),
            "unsupported AAC-LC sampling frequency index 13"
        );
        assert_eq!(
            SfbError::MaxSfbOutOfRange {
                max_sfb: 50,
                total_sfb: 49
            }
            .to_string(),
            "AAC ICS max_sfb 50 exceeds scalefactor band table total 49"
        );
        assert_eq!(
            SfbError::UnsupportedFrameLength(256).to_string(),
            "unsupported AAC scalefactor-band frame length 256"
        );
    }
}
