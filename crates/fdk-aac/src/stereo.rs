//! AAC-LC channel-pair stereo tools.

use std::fmt;

use crate::bits::{BitError, BitReader};
use crate::ics::IcsInfo;
use crate::inverse::InverseQuantizedSpectrum;
use crate::scalefactor::ScalefactorData;
use crate::section::{SectionData, INTENSITY_HCB, INTENSITY_HCB2, NOISE_HCB, ZERO_HCB};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MsMaskPresent {
    None,
    Some,
    All,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MsStereoData {
    pub mask_present: MsMaskPresent,
    /// Per-window-group, per-scalefactor-band MS enable flags.
    pub used: Vec<Vec<bool>>,
}

impl MsStereoData {
    pub fn parse_aac_lc(reader: &mut BitReader<'_>, ics: &IcsInfo) -> Result<Self, StereoError> {
        let raw = reader.read_u8(2)?;
        let groups = ics.window_group_lengths.len();
        let max_sfb = ics.max_sfb as usize;
        let mask_present = match raw {
            0 => MsMaskPresent::None,
            1 => MsMaskPresent::Some,
            2 => MsMaskPresent::All,
            _ => return Err(StereoError::ReservedMsMaskPresent(raw)),
        };

        let mut used = vec![vec![false; max_sfb]; groups];
        match mask_present {
            MsMaskPresent::None => {}
            MsMaskPresent::All => {
                for group in &mut used {
                    group.fill(true);
                }
            }
            MsMaskPresent::Some => {
                for group in &mut used {
                    for band in group {
                        *band = reader.read_bool()?;
                    }
                }
            }
        }

        Ok(Self { mask_present, used })
    }

    pub fn is_used(&self, group: usize, band: usize) -> bool {
        self.used
            .get(group)
            .and_then(|bands| bands.get(band))
            .copied()
            .unwrap_or(false)
    }
}

pub fn apply_ms_stereo_f32(
    ms: &MsStereoData,
    left: &mut InverseQuantizedSpectrum,
    right: &mut InverseQuantizedSpectrum,
    ics: &IcsInfo,
    band_offsets: &[usize],
    left_sections: &SectionData,
    right_sections: &SectionData,
) -> Result<(), StereoError> {
    validate_ms_layout(
        ms,
        left,
        right,
        ics,
        band_offsets,
        left_sections,
        right_sections,
    )?;

    let mut window_offset = 0usize;
    for (group, &group_len) in ics.window_group_lengths.iter().enumerate() {
        for band in 0..ics.max_sfb as usize {
            if !ms.is_used(group, band)
                || !is_ms_applicable(left_sections, right_sections, group, band)
            {
                continue;
            }
            let start = band_offsets[band];
            let end = band_offsets[band + 1];
            for group_window in 0..group_len as usize {
                let window = window_offset + group_window;
                for index in start..end {
                    let mid = left.windows[window][index];
                    let side = right.windows[window][index];
                    left.windows[window][index] = (mid + side) * std::f32::consts::FRAC_1_SQRT_2;
                    right.windows[window][index] = (mid - side) * std::f32::consts::FRAC_1_SQRT_2;
                }
            }
        }
        window_offset += group_len as usize;
    }

    Ok(())
}

pub fn apply_intensity_stereo_f32(
    ms: Option<&MsStereoData>,
    left: &InverseQuantizedSpectrum,
    right: &mut InverseQuantizedSpectrum,
    ics: &IcsInfo,
    band_offsets: &[usize],
    right_sections: &SectionData,
    right_scalefactors: &ScalefactorData,
) -> Result<(), StereoError> {
    validate_intensity_layout(
        left,
        right,
        ics,
        band_offsets,
        right_sections,
        right_scalefactors,
    )?;

    let mut window_offset = 0usize;
    for (group, &group_len) in ics.window_group_lengths.iter().enumerate() {
        for band in 0..ics.max_sfb as usize {
            let codebook = right_sections.codebooks[group][band];
            if codebook != INTENSITY_HCB && codebook != INTENSITY_HCB2 {
                continue;
            }

            let scale = intensity_scale_f32(
                right_scalefactors.values[group][band],
                codebook,
                ms.is_some_and(|ms| ms.is_used(group, band)),
            );
            let start = band_offsets[band];
            let end = band_offsets[band + 1];
            for group_window in 0..group_len as usize {
                let window = window_offset + group_window;
                for index in start..end {
                    right.windows[window][index] = left.windows[window][index] * scale;
                }
            }
        }
        window_offset += group_len as usize;
    }

    Ok(())
}

pub fn intensity_scale_f32(position_minus_100: i16, codebook: u8, ms_used: bool) -> f32 {
    let magnitude = 2.0f32.powf(-0.25 * (position_minus_100 as f32 + 100.0));
    let negative = if ms_used {
        codebook == INTENSITY_HCB
    } else {
        codebook == INTENSITY_HCB2
    };
    if negative {
        -magnitude
    } else {
        magnitude
    }
}

fn is_ms_applicable(
    left_sections: &SectionData,
    right_sections: &SectionData,
    group: usize,
    band: usize,
) -> bool {
    let left = left_sections.codebooks[group][band];
    let right = right_sections.codebooks[group][band];
    is_spectral_or_zero(left) && is_spectral_or_zero(right)
}

fn is_spectral_or_zero(codebook: u8) -> bool {
    !matches!(codebook, NOISE_HCB | INTENSITY_HCB | INTENSITY_HCB2) && codebook != ZERO_HCB
}

fn validate_ms_layout(
    ms: &MsStereoData,
    left: &InverseQuantizedSpectrum,
    right: &InverseQuantizedSpectrum,
    ics: &IcsInfo,
    band_offsets: &[usize],
    left_sections: &SectionData,
    right_sections: &SectionData,
) -> Result<(), StereoError> {
    let groups = ics.window_group_lengths.len();
    let max_sfb = ics.max_sfb as usize;
    let total_windows = ics
        .window_group_lengths
        .iter()
        .map(|&len| len as usize)
        .sum::<usize>();

    if ms.used.len() != groups || ms.used.iter().any(|group| group.len() < max_sfb) {
        return Err(StereoError::LayoutMismatch);
    }
    if left.windows.len() != total_windows || right.windows.len() != total_windows {
        return Err(StereoError::LayoutMismatch);
    }
    if band_offsets.len() <= max_sfb {
        return Err(StereoError::LayoutMismatch);
    }
    let granule_len = band_offsets[max_sfb];
    if left
        .windows
        .iter()
        .chain(&right.windows)
        .any(|window| window.len() < granule_len)
    {
        return Err(StereoError::LayoutMismatch);
    }
    if left_sections.codebooks.len() != groups
        || right_sections.codebooks.len() != groups
        || left_sections
            .codebooks
            .iter()
            .chain(&right_sections.codebooks)
            .any(|group| group.len() < max_sfb)
    {
        return Err(StereoError::LayoutMismatch);
    }
    Ok(())
}

fn validate_intensity_layout(
    left: &InverseQuantizedSpectrum,
    right: &InverseQuantizedSpectrum,
    ics: &IcsInfo,
    band_offsets: &[usize],
    right_sections: &SectionData,
    right_scalefactors: &ScalefactorData,
) -> Result<(), StereoError> {
    let groups = ics.window_group_lengths.len();
    let max_sfb = ics.max_sfb as usize;
    let total_windows = ics
        .window_group_lengths
        .iter()
        .map(|&len| len as usize)
        .sum::<usize>();

    if left.windows.len() != total_windows || right.windows.len() != total_windows {
        return Err(StereoError::LayoutMismatch);
    }
    if band_offsets.len() <= max_sfb {
        return Err(StereoError::LayoutMismatch);
    }
    let granule_len = band_offsets[max_sfb];
    if left
        .windows
        .iter()
        .chain(&right.windows)
        .any(|window| window.len() < granule_len)
    {
        return Err(StereoError::LayoutMismatch);
    }
    if right_sections.codebooks.len() != groups
        || right_sections
            .codebooks
            .iter()
            .any(|group| group.len() < max_sfb)
        || right_scalefactors.values.len() != groups
        || right_scalefactors
            .values
            .iter()
            .any(|group| group.len() < max_sfb)
    {
        return Err(StereoError::LayoutMismatch);
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StereoError {
    Bit(BitError),
    LayoutMismatch,
    ReservedMsMaskPresent(u8),
}

impl From<BitError> for StereoError {
    fn from(value: BitError) -> Self {
        Self::Bit(value)
    }
}

impl fmt::Display for StereoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bit(err) => write!(f, "stereo bitstream error: {err}"),
            Self::LayoutMismatch => write!(f, "stereo tool layout mismatch"),
            Self::ReservedMsMaskPresent(value) => {
                write!(f, "reserved ms_mask_present value {value}")
            }
        }
    }
}

impl std::error::Error for StereoError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitWriter;
    use crate::ics::{IcsInfo, WindowSequence, WindowShape};
    use crate::scalefactor::ScalefactorData;

    fn ics(groups: Vec<u8>, max_sfb: u8) -> IcsInfo {
        IcsInfo {
            window_sequence: if groups.len() == 1 {
                WindowSequence::OnlyLong
            } else {
                WindowSequence::EightShort
            },
            window_shape: WindowShape::Sine,
            max_sfb,
            total_sfb: max_sfb,
            predictor_data_present: false,
            scale_factor_grouping: 0,
            window_group_lengths: groups,
            bits_read: 0,
        }
    }

    fn sections(codebooks: Vec<Vec<u8>>) -> SectionData {
        SectionData {
            sections: Vec::new(),
            codebooks,
            bits_read: 0,
        }
    }

    #[test]
    fn parses_ms_mask_none_some_all() {
        let ics = ics(vec![1, 1], 3);
        let mut writer = BitWriter::new();
        writer.write(1, 2);
        for bit in [true, false, true, false, true, false] {
            writer.write_bool(bit);
        }
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        let ms = MsStereoData::parse_aac_lc(&mut reader, &ics).unwrap();
        assert_eq!(ms.mask_present, MsMaskPresent::Some);
        assert_eq!(
            ms.used,
            vec![vec![true, false, true], vec![false, true, false]]
        );

        let mut reader = BitReader::new(&[0b1000_0000]);
        let ms = MsStereoData::parse_aac_lc(&mut reader, &ics).unwrap();
        assert_eq!(ms.mask_present, MsMaskPresent::All);
        assert!(ms.used.iter().flatten().all(|used| *used));

        let mut reader = BitReader::new(&[0b1100_0000]);
        assert_eq!(
            MsStereoData::parse_aac_lc(&mut reader, &ics).unwrap_err(),
            StereoError::ReservedMsMaskPresent(3)
        );
    }

    #[test]
    fn applies_ms_stereo_to_enabled_spectral_bands() {
        let ics = ics(vec![1], 2);
        let ms = MsStereoData {
            mask_present: MsMaskPresent::Some,
            used: vec![vec![true, false]],
        };
        let mut left = InverseQuantizedSpectrum {
            windows: vec![vec![3.0, 5.0, 7.0, 11.0]],
        };
        let mut right = InverseQuantizedSpectrum {
            windows: vec![vec![1.0, 2.0, 13.0, 17.0]],
        };
        let left_sections = sections(vec![vec![1, 1]]);
        let right_sections = sections(vec![vec![1, 1]]);

        apply_ms_stereo_f32(
            &ms,
            &mut left,
            &mut right,
            &ics,
            &[0, 2, 4],
            &left_sections,
            &right_sections,
        )
        .unwrap();

        assert!((left.windows[0][0] - 4.0 * std::f32::consts::FRAC_1_SQRT_2).abs() < 1.0e-6);
        assert!((right.windows[0][0] - 2.0 * std::f32::consts::FRAC_1_SQRT_2).abs() < 1.0e-6);
        assert_eq!(left.windows[0][2], 7.0);
        assert_eq!(right.windows[0][3], 17.0);
    }

    #[test]
    fn skips_intensity_and_noise_bands() {
        let ics = ics(vec![1], 2);
        let ms = MsStereoData {
            mask_present: MsMaskPresent::All,
            used: vec![vec![true, true]],
        };
        let mut left = InverseQuantizedSpectrum {
            windows: vec![vec![3.0, 5.0, 7.0, 11.0]],
        };
        let mut right = InverseQuantizedSpectrum {
            windows: vec![vec![1.0, 2.0, 13.0, 17.0]],
        };
        let left_sections = sections(vec![vec![INTENSITY_HCB, 1]]);
        let right_sections = sections(vec![vec![1, NOISE_HCB]]);

        apply_ms_stereo_f32(
            &ms,
            &mut left,
            &mut right,
            &ics,
            &[0, 2, 4],
            &left_sections,
            &right_sections,
        )
        .unwrap();

        assert_eq!(left.windows[0], vec![3.0, 5.0, 7.0, 11.0]);
        assert_eq!(right.windows[0], vec![1.0, 2.0, 13.0, 17.0]);
    }

    #[test]
    fn computes_intensity_scale_sign_like_fdk() {
        assert!((intensity_scale_f32(-100, INTENSITY_HCB, false) - 1.0).abs() < 1.0e-6);
        assert!((intensity_scale_f32(-96, INTENSITY_HCB, false) - 0.5).abs() < 1.0e-6);
        assert!((intensity_scale_f32(-100, INTENSITY_HCB2, false) + 1.0).abs() < 1.0e-6);
        assert!((intensity_scale_f32(-100, INTENSITY_HCB, true) + 1.0).abs() < 1.0e-6);
        assert!((intensity_scale_f32(-100, INTENSITY_HCB2, true) - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn applies_intensity_stereo_to_right_channel() {
        let ics = ics(vec![1], 3);
        let left = InverseQuantizedSpectrum {
            windows: vec![vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]],
        };
        let mut right = InverseQuantizedSpectrum {
            windows: vec![vec![0.0; 6]],
        };
        let right_sections = sections(vec![vec![1, INTENSITY_HCB, INTENSITY_HCB2]]);
        let right_scalefactors = ScalefactorData {
            values: vec![vec![0, -100, -96]],
        };
        let ms = MsStereoData {
            mask_present: MsMaskPresent::Some,
            used: vec![vec![false, false, true]],
        };

        apply_intensity_stereo_f32(
            Some(&ms),
            &left,
            &mut right,
            &ics,
            &[0, 2, 4, 6],
            &right_sections,
            &right_scalefactors,
        )
        .unwrap();

        assert_eq!(right.windows[0][0], 0.0);
        assert_eq!(right.windows[0][1], 0.0);
        assert!((right.windows[0][2] - 6.0).abs() < 1.0e-6);
        assert!((right.windows[0][3] - 8.0).abs() < 1.0e-6);
        assert!((right.windows[0][4] - 5.0).abs() < 1.0e-6);
        assert!((right.windows[0][5] - 6.0).abs() < 1.0e-6);
    }

    #[test]
    fn applies_intensity_stereo_across_grouped_short_windows() {
        let ics = ics(vec![2], 1);
        let left = InverseQuantizedSpectrum {
            windows: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        };
        let mut right = InverseQuantizedSpectrum {
            windows: vec![vec![0.0; 2], vec![0.0; 2]],
        };
        let right_sections = sections(vec![vec![INTENSITY_HCB2]]);
        let right_scalefactors = ScalefactorData {
            values: vec![vec![-100]],
        };

        apply_intensity_stereo_f32(
            None,
            &left,
            &mut right,
            &ics,
            &[0, 2],
            &right_sections,
            &right_scalefactors,
        )
        .unwrap();

        assert_eq!(right.windows[0], vec![-1.0, -2.0]);
        assert_eq!(right.windows[1], vec![-3.0, -4.0]);
    }

    #[test]
    fn parses_none_mask_and_handles_out_of_range_lookup() {
        let info = ics(vec![1], 2);
        let ms = MsStereoData::parse_aac_lc(&mut BitReader::new(&[0]), &info).unwrap();
        assert_eq!(ms.mask_present, MsMaskPresent::None);
        assert_eq!(ms.used, vec![vec![false, false]]);
        assert!(!ms.is_used(1, 0));
        assert!(!ms.is_used(0, 2));
        assert!(matches!(
            MsStereoData::parse_aac_lc(&mut BitReader::new(&[]), &info),
            Err(StereoError::Bit(BitError::UnexpectedEof { .. }))
        ));
    }

    #[test]
    fn validates_every_ms_layout_component() {
        let info = ics(vec![1], 1);
        let valid_ms = MsStereoData {
            mask_present: MsMaskPresent::All,
            used: vec![vec![true]],
        };
        let valid_spectrum = InverseQuantizedSpectrum {
            windows: vec![vec![1.0, 2.0]],
        };
        let valid_sections = sections(vec![vec![1]]);
        let run = |ms: &MsStereoData,
                   left: InverseQuantizedSpectrum,
                   right: InverseQuantizedSpectrum,
                   offsets: &[usize],
                   left_sections: SectionData,
                   right_sections: SectionData| {
            let mut left = left;
            let mut right = right;
            apply_ms_stereo_f32(
                ms,
                &mut left,
                &mut right,
                &info,
                offsets,
                &left_sections,
                &right_sections,
            )
        };
        let empty = InverseQuantizedSpectrum {
            windows: Vec::new(),
        };
        assert_eq!(
            run(
                &MsStereoData {
                    mask_present: MsMaskPresent::All,
                    used: Vec::new()
                },
                valid_spectrum.clone(),
                valid_spectrum.clone(),
                &[0, 2],
                valid_sections.clone(),
                valid_sections.clone(),
            ),
            Err(StereoError::LayoutMismatch)
        );
        assert_eq!(
            run(
                &valid_ms,
                empty,
                valid_spectrum.clone(),
                &[0, 2],
                valid_sections.clone(),
                valid_sections.clone()
            ),
            Err(StereoError::LayoutMismatch)
        );
        assert_eq!(
            run(
                &valid_ms,
                valid_spectrum.clone(),
                valid_spectrum.clone(),
                &[0],
                valid_sections.clone(),
                valid_sections.clone()
            ),
            Err(StereoError::LayoutMismatch)
        );
        assert_eq!(
            run(
                &valid_ms,
                InverseQuantizedSpectrum {
                    windows: vec![vec![1.0]]
                },
                valid_spectrum.clone(),
                &[0, 2],
                valid_sections.clone(),
                valid_sections.clone(),
            ),
            Err(StereoError::LayoutMismatch)
        );
        assert_eq!(
            run(
                &valid_ms,
                valid_spectrum.clone(),
                valid_spectrum,
                &[0, 2],
                sections(Vec::new()),
                valid_sections
            ),
            Err(StereoError::LayoutMismatch)
        );
    }

    #[test]
    fn validates_intensity_layout_and_stereo_errors() {
        let info = ics(vec![1], 1);
        let valid = InverseQuantizedSpectrum {
            windows: vec![vec![1.0, 2.0]],
        };
        let section = sections(vec![vec![INTENSITY_HCB]]);
        let factors = ScalefactorData {
            values: vec![vec![-100]],
        };
        let mut right = valid.clone();
        assert_eq!(
            apply_intensity_stereo_f32(
                None,
                &InverseQuantizedSpectrum {
                    windows: Vec::new()
                },
                &mut right,
                &info,
                &[0, 2],
                &section,
                &factors
            ),
            Err(StereoError::LayoutMismatch)
        );
        let mut right = valid.clone();
        assert_eq!(
            apply_intensity_stereo_f32(None, &valid, &mut right, &info, &[0], &section, &factors),
            Err(StereoError::LayoutMismatch)
        );
        let mut short = InverseQuantizedSpectrum {
            windows: vec![vec![1.0]],
        };
        assert_eq!(
            apply_intensity_stereo_f32(
                None,
                &valid,
                &mut short,
                &info,
                &[0, 2],
                &section,
                &factors
            ),
            Err(StereoError::LayoutMismatch)
        );
        let mut right = valid.clone();
        assert_eq!(
            apply_intensity_stereo_f32(
                None,
                &valid,
                &mut right,
                &info,
                &[0, 2],
                &sections(Vec::new()),
                &factors
            ),
            Err(StereoError::LayoutMismatch)
        );
        let mut right = valid.clone();
        assert_eq!(
            apply_intensity_stereo_f32(
                None,
                &valid,
                &mut right,
                &info,
                &[0, 2],
                &section,
                &ScalefactorData { values: Vec::new() }
            ),
            Err(StereoError::LayoutMismatch)
        );

        let bit = BitError::UnexpectedEof {
            needed_bits: 2,
            remaining_bits: 0,
        };
        assert_eq!(
            StereoError::from(bit.clone()),
            StereoError::Bit(bit.clone())
        );
        assert!(StereoError::Bit(bit)
            .to_string()
            .starts_with("stereo bitstream error:"));
        for error in [
            StereoError::LayoutMismatch,
            StereoError::ReservedMsMaskPresent(3),
        ] {
            assert!(!error.to_string().is_empty());
        }
    }

    #[test]
    fn zero_codebook_is_not_ms_applicable() {
        let info = ics(vec![1], 1);
        let ms = MsStereoData {
            mask_present: MsMaskPresent::All,
            used: vec![vec![true]],
        };
        let mut left = InverseQuantizedSpectrum {
            windows: vec![vec![3.0]],
        };
        let mut right = InverseQuantizedSpectrum {
            windows: vec![vec![1.0]],
        };
        apply_ms_stereo_f32(
            &ms,
            &mut left,
            &mut right,
            &info,
            &[0, 1],
            &sections(vec![vec![ZERO_HCB]]),
            &sections(vec![vec![1]]),
        )
        .unwrap();
        assert_eq!(left.windows[0][0], 3.0);
        assert_eq!(right.windows[0][0], 1.0);
    }
}
