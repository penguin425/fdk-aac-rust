//! AAC-LC Perceptual Noise Substitution (PNS) f32 reference helpers.

use std::fmt;

use crate::ics::IcsInfo;
use crate::inverse::{
    inverse_quantize_value_fixed, FixedInverseQuantizedSpectrum, InverseQuantizedSpectrum,
};
use crate::scalefactor::ScalefactorData;
use crate::section::{SectionData, NOISE_HCB};
use crate::stereo::MsStereoData;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PnsRandomState {
    seed: u32,
}

impl PnsRandomState {
    pub fn new(seed: u32) -> Self {
        Self { seed }
    }

    pub fn seed(self) -> u32 {
        self.seed
    }

    fn next_f32(&mut self) -> f32 {
        self.seed = self
            .seed
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        let signed = self.seed as i32 as f32;
        signed / 2_147_483_648.0
    }

    fn next_i32(&mut self) -> i32 {
        self.seed = self
            .seed
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.seed as i32
    }
}

pub fn apply_pns_f32(
    spectrum: &mut InverseQuantizedSpectrum,
    ics: &IcsInfo,
    band_offsets: &[usize],
    sections: &SectionData,
    scalefactors: &ScalefactorData,
    random: &mut PnsRandomState,
) -> Result<(), PnsError> {
    validate_layout(spectrum, ics, band_offsets, sections, scalefactors)?;

    let mut window_offset = 0usize;
    for (group, &group_len) in ics.window_group_lengths.iter().enumerate() {
        for band in 0..ics.max_sfb as usize {
            if sections.codebooks[group][band] != NOISE_HCB {
                continue;
            }
            let start = band_offsets[band];
            let end = band_offsets[band + 1];
            let gain = pns_gain_f32(scalefactors.values[group][band]);
            for group_window in 0..group_len as usize {
                let window = window_offset + group_window;
                let mut noise = normalized_noise(end - start, random);
                for value in &mut noise {
                    *value *= gain;
                }
                spectrum.windows[window][start..end].copy_from_slice(&noise);
            }
        }
        window_offset += group_len as usize;
    }

    Ok(())
}

pub fn apply_pns_fixed_bridge(
    spectrum: &mut FixedInverseQuantizedSpectrum,
    ics: &IcsInfo,
    band_offsets: &[usize],
    sections: &SectionData,
    scalefactors: &ScalefactorData,
    random: &mut PnsRandomState,
) -> Result<(), PnsError> {
    apply_pns_fixed(spectrum, ics, band_offsets, sections, scalefactors, random)
}

pub fn apply_pns_fixed(
    spectrum: &mut FixedInverseQuantizedSpectrum,
    ics: &IcsInfo,
    band_offsets: &[usize],
    sections: &SectionData,
    scalefactors: &ScalefactorData,
    random: &mut PnsRandomState,
) -> Result<(), PnsError> {
    validate_fixed_layout(spectrum, ics, band_offsets, sections, scalefactors)?;

    let mut window_offset = 0usize;
    for (group, &group_len) in ics.window_group_lengths.iter().enumerate() {
        for band in 0..ics.max_sfb as usize {
            if sections.codebooks[group][band] != NOISE_HCB {
                continue;
            }
            let start = band_offsets[band];
            let end = band_offsets[band + 1];
            for group_window in 0..group_len as usize {
                let window = window_offset + group_window;
                let exponent = spectrum.window_exponents.get(window).copied().unwrap_or(0);
                let gain = crate::inverse::inverse_value_f32_to_q31(
                    pns_gain_f32(scalefactors.values[group][band])
                        * 2.0f32.powi(-(exponent as i32)),
                );
                let noise = normalized_noise_fixed_q31(end - start, random);
                for (dst, value) in spectrum.windows[window][start..end]
                    .iter_mut()
                    .zip(noise.into_iter())
                {
                    *dst = mul_q31_saturate(value, gain);
                }
            }
        }
        window_offset += group_len as usize;
    }

    Ok(())
}

pub fn apply_pns_pair_f32(
    left: &mut InverseQuantizedSpectrum,
    right: &mut InverseQuantizedSpectrum,
    ics: &IcsInfo,
    band_offsets: &[usize],
    left_sections: &SectionData,
    right_sections: &SectionData,
    left_scalefactors: &ScalefactorData,
    right_scalefactors: &ScalefactorData,
    ms: Option<&MsStereoData>,
    random: &mut PnsRandomState,
) -> Result<(), PnsError> {
    validate_layout(left, ics, band_offsets, left_sections, left_scalefactors)?;
    validate_layout(right, ics, band_offsets, right_sections, right_scalefactors)?;

    let mut window_offset = 0usize;
    for (group, &group_len) in ics.window_group_lengths.iter().enumerate() {
        for band in 0..ics.max_sfb as usize {
            let left_noise = left_sections.codebooks[group][band] == NOISE_HCB;
            let right_noise = right_sections.codebooks[group][band] == NOISE_HCB;
            if !left_noise && !right_noise {
                continue;
            }
            let start = band_offsets[band];
            let end = band_offsets[band + 1];
            let width = end - start;
            let correlated =
                left_noise && right_noise && ms.is_some_and(|ms| ms.is_used(group, band));
            let left_gain = pns_gain_f32(left_scalefactors.values[group][band]);
            let right_gain = pns_gain_f32(right_scalefactors.values[group][band]);

            for group_window in 0..group_len as usize {
                let window = window_offset + group_window;
                let shared = if correlated {
                    Some(normalized_noise(width, random))
                } else {
                    None
                };

                if left_noise {
                    let mut noise = shared
                        .clone()
                        .unwrap_or_else(|| normalized_noise(width, random));
                    for value in &mut noise {
                        *value *= left_gain;
                    }
                    left.windows[window][start..end].copy_from_slice(&noise);
                }
                if right_noise {
                    let mut noise = shared.unwrap_or_else(|| normalized_noise(width, random));
                    for value in &mut noise {
                        *value *= right_gain;
                    }
                    right.windows[window][start..end].copy_from_slice(&noise);
                }
            }
        }
        window_offset += group_len as usize;
    }

    Ok(())
}

pub fn apply_pns_pair_fixed_bridge(
    left: &mut FixedInverseQuantizedSpectrum,
    right: &mut FixedInverseQuantizedSpectrum,
    ics: &IcsInfo,
    band_offsets: &[usize],
    left_sections: &SectionData,
    right_sections: &SectionData,
    left_scalefactors: &ScalefactorData,
    right_scalefactors: &ScalefactorData,
    ms: Option<&MsStereoData>,
    random: &mut PnsRandomState,
) -> Result<(), PnsError> {
    apply_pns_pair_fixed(
        left,
        right,
        ics,
        band_offsets,
        left_sections,
        right_sections,
        left_scalefactors,
        right_scalefactors,
        ms,
        random,
    )
}

pub fn apply_pns_pair_fixed(
    left: &mut FixedInverseQuantizedSpectrum,
    right: &mut FixedInverseQuantizedSpectrum,
    ics: &IcsInfo,
    band_offsets: &[usize],
    left_sections: &SectionData,
    right_sections: &SectionData,
    left_scalefactors: &ScalefactorData,
    right_scalefactors: &ScalefactorData,
    ms: Option<&MsStereoData>,
    random: &mut PnsRandomState,
) -> Result<(), PnsError> {
    validate_fixed_layout(left, ics, band_offsets, left_sections, left_scalefactors)?;
    validate_fixed_layout(right, ics, band_offsets, right_sections, right_scalefactors)?;

    let mut window_offset = 0usize;
    for (group, &group_len) in ics.window_group_lengths.iter().enumerate() {
        for band in 0..ics.max_sfb as usize {
            let left_noise = left_sections.codebooks[group][band] == NOISE_HCB;
            let right_noise = right_sections.codebooks[group][band] == NOISE_HCB;
            if !left_noise && !right_noise {
                continue;
            }
            let start = band_offsets[band];
            let end = band_offsets[band + 1];
            let width = end - start;
            let correlated =
                left_noise && right_noise && ms.is_some_and(|ms| ms.is_used(group, band));
            let left_gain = pns_gain_fixed_q31(left_scalefactors.values[group][band]);
            let right_gain = pns_gain_fixed_q31(right_scalefactors.values[group][band]);

            for group_window in 0..group_len as usize {
                let window = window_offset + group_window;
                let shared = if correlated {
                    Some(normalized_noise_fixed_q31(width, random))
                } else {
                    None
                };

                if left_noise {
                    let noise = shared
                        .clone()
                        .unwrap_or_else(|| normalized_noise_fixed_q31(width, random));
                    for (dst, value) in left.windows[window][start..end]
                        .iter_mut()
                        .zip(noise.into_iter())
                    {
                        *dst = mul_q31_saturate(value, left_gain);
                    }
                }
                if right_noise {
                    let noise = shared.unwrap_or_else(|| normalized_noise_fixed_q31(width, random));
                    for (dst, value) in right.windows[window][start..end]
                        .iter_mut()
                        .zip(noise.into_iter())
                    {
                        *dst = mul_q31_saturate(value, right_gain);
                    }
                }
            }
        }
        window_offset += group_len as usize;
    }

    Ok(())
}

pub fn pns_gain_f32(noise_energy_minus_100: i16) -> f32 {
    2.0f32.powf(0.25 * noise_energy_minus_100 as f32)
}

pub fn normalized_noise(width: usize, random: &mut PnsRandomState) -> Vec<f32> {
    if width == 0 {
        return Vec::new();
    }
    let mut noise = (0..width).map(|_| random.next_f32()).collect::<Vec<_>>();
    let energy = noise.iter().map(|value| value * value).sum::<f32>();
    if energy > 0.0 {
        let norm = energy.sqrt().recip();
        for value in &mut noise {
            *value *= norm;
        }
    }
    noise
}

pub fn pns_gain_fixed_q31(noise_energy_minus_100: i16) -> i32 {
    inverse_quantize_value_fixed(1, noise_energy_minus_100)
}

pub fn normalized_noise_fixed_q31(width: usize, random: &mut PnsRandomState) -> Vec<i32> {
    if width == 0 {
        return Vec::new();
    }
    let raw = (0..width)
        .map(|_| random.next_i32() as i64)
        .collect::<Vec<_>>();
    let energy = raw
        .iter()
        .map(|&value| (value as i128 * value as i128) as u128)
        .sum::<u128>();
    if energy == 0 {
        return vec![0; width];
    }
    let norm = integer_sqrt_u128(energy).max(1);
    raw.into_iter()
        .map(|value| {
            let scaled = (value as i128 * (1i128 << 31)) / norm as i128;
            scaled.clamp(i32::MIN as i128 + 1, i32::MAX as i128) as i32
        })
        .collect()
}

fn mul_q31_saturate(a: i32, b: i32) -> i32 {
    let product = a as i64 * b as i64;
    ((product + (1 << 30)) >> 31).clamp(i32::MIN as i64 + 1, i32::MAX as i64) as i32
}

fn integer_sqrt_u128(value: u128) -> u128 {
    if value < 2 {
        return value;
    }
    let mut low = 1u128;
    let mut high = 1u128 << ((128 - value.leading_zeros() as usize + 1) / 2).min(64);
    while low <= high {
        let mid = (low + high) >> 1;
        let square = mid.saturating_mul(mid);
        if square == value {
            return mid;
        } else if square < value {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    high
}

fn validate_layout(
    spectrum: &InverseQuantizedSpectrum,
    ics: &IcsInfo,
    band_offsets: &[usize],
    sections: &SectionData,
    scalefactors: &ScalefactorData,
) -> Result<(), PnsError> {
    let groups = ics.window_group_lengths.len();
    let max_sfb = ics.max_sfb as usize;
    let total_windows = ics
        .window_group_lengths
        .iter()
        .map(|&len| len as usize)
        .sum::<usize>();
    if spectrum.windows.len() != total_windows || band_offsets.len() <= max_sfb {
        return Err(PnsError::LayoutMismatch);
    }
    let granule_len = band_offsets[max_sfb];
    if spectrum
        .windows
        .iter()
        .any(|window| window.len() < granule_len)
    {
        return Err(PnsError::LayoutMismatch);
    }
    if sections.codebooks.len() != groups
        || sections.codebooks.iter().any(|group| group.len() < max_sfb)
        || scalefactors.values.len() != groups
        || scalefactors
            .values
            .iter()
            .any(|group| group.len() < max_sfb)
    {
        return Err(PnsError::LayoutMismatch);
    }
    Ok(())
}

fn validate_fixed_layout(
    spectrum: &FixedInverseQuantizedSpectrum,
    ics: &IcsInfo,
    band_offsets: &[usize],
    sections: &SectionData,
    scalefactors: &ScalefactorData,
) -> Result<(), PnsError> {
    let groups = ics.window_group_lengths.len();
    let max_sfb = ics.max_sfb as usize;
    let total_windows = ics
        .window_group_lengths
        .iter()
        .map(|&len| len as usize)
        .sum::<usize>();
    if spectrum.windows.len() != total_windows || band_offsets.len() <= max_sfb {
        return Err(PnsError::LayoutMismatch);
    }
    let granule_len = band_offsets[max_sfb];
    if spectrum
        .windows
        .iter()
        .any(|window| window.len() < granule_len)
    {
        return Err(PnsError::LayoutMismatch);
    }
    if sections.codebooks.len() != groups
        || sections.codebooks.iter().any(|group| group.len() < max_sfb)
        || scalefactors.values.len() != groups
        || scalefactors
            .values
            .iter()
            .any(|group| group.len() < max_sfb)
    {
        return Err(PnsError::LayoutMismatch);
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PnsError {
    LayoutMismatch,
}

impl fmt::Display for PnsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LayoutMismatch => write!(f, "PNS layout mismatch"),
        }
    }
}

impl std::error::Error for PnsError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ics::{IcsInfo, WindowSequence, WindowShape};
    use crate::section::ZERO_HCB;
    use crate::stereo::{MsMaskPresent, MsStereoData};

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
    fn generates_unit_energy_noise_deterministically() {
        let mut a = PnsRandomState::new(0x1234_5678);
        let mut b = PnsRandomState::new(0x1234_5678);
        let noise_a = normalized_noise(8, &mut a);
        let noise_b = normalized_noise(8, &mut b);
        assert_eq!(noise_a, noise_b);
        let energy = noise_a.iter().map(|value| value * value).sum::<f32>();
        assert!((energy - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn generates_fixed_unit_energy_noise_deterministically() {
        let mut a = PnsRandomState::new(0x1234_5678);
        let mut b = PnsRandomState::new(0x1234_5678);
        let noise_a = normalized_noise_fixed_q31(8, &mut a);
        let noise_b = normalized_noise_fixed_q31(8, &mut b);
        assert_eq!(noise_a, noise_b);
        let energy_q62 = noise_a
            .iter()
            .map(|&value| value as i128 * value as i128)
            .sum::<i128>();
        let one_q62 = 1i128 << 62;
        assert!((energy_q62 - one_q62).abs() < (1i128 << 33));

        // This predecessor maps to zero on the next LCG step.
        let mut zero = PnsRandomState::new(634_785_765);
        assert_eq!(normalized_noise_fixed_q31(1, &mut zero), [0]);
        let mut zero = PnsRandomState::new(634_785_765);
        assert_eq!(normalized_noise(1, &mut zero), [0.0]);
    }

    #[test]
    fn fixed_pns_gain_tracks_inverse_quantizer_gain() {
        assert_eq!(pns_gain_fixed_q31(-4), 0x4000_0000);
        assert_eq!(pns_gain_fixed_q31(0), i32::MAX);
    }

    #[test]
    fn applies_pns_to_noise_bands_only() {
        let ics = ics(vec![1], 2);
        let mut spectrum = InverseQuantizedSpectrum {
            windows: vec![vec![9.0, 9.0, 5.0, 6.0]],
        };
        let sections = sections(vec![vec![NOISE_HCB, ZERO_HCB]]);
        let scalefactors = ScalefactorData {
            values: vec![vec![0, 0]],
        };
        let mut random = PnsRandomState::new(1);
        apply_pns_f32(
            &mut spectrum,
            &ics,
            &[0, 2, 4],
            &sections,
            &scalefactors,
            &mut random,
        )
        .unwrap();

        let energy = spectrum.windows[0][0..2]
            .iter()
            .map(|value| value * value)
            .sum::<f32>();
        assert!((energy - 1.0).abs() < 1.0e-6);
        assert_eq!(&spectrum.windows[0][2..4], &[5.0, 6.0]);
    }

    #[test]
    fn applies_pns_to_fixed_noise_bands_only() {
        let ics = ics(vec![1], 2);
        let mut spectrum = FixedInverseQuantizedSpectrum {
            windows: vec![vec![9, 9, 5, 6]],
            window_exponents: vec![0],
        };
        let sections = sections(vec![vec![NOISE_HCB, ZERO_HCB]]);
        let scalefactors = ScalefactorData {
            values: vec![vec![0, 0]],
        };
        let mut random = PnsRandomState::new(1);
        apply_pns_fixed(
            &mut spectrum,
            &ics,
            &[0, 2, 4],
            &sections,
            &scalefactors,
            &mut random,
        )
        .unwrap();

        assert!(spectrum.windows[0][0..2].iter().any(|value| *value != 0));
        assert_eq!(&spectrum.windows[0][2..4], &[5, 6]);
    }

    #[test]
    fn applies_correlated_pair_pns_when_ms_mask_is_set() {
        let ics = ics(vec![1], 1);
        let mut left = InverseQuantizedSpectrum {
            windows: vec![vec![0.0; 4]],
        };
        let mut right = InverseQuantizedSpectrum {
            windows: vec![vec![0.0; 4]],
        };
        let sections = sections(vec![vec![NOISE_HCB]]);
        let scalefactors = ScalefactorData {
            values: vec![vec![0]],
        };
        let ms = MsStereoData {
            mask_present: MsMaskPresent::Some,
            used: vec![vec![true]],
        };
        let mut random = PnsRandomState::new(7);

        apply_pns_pair_f32(
            &mut left,
            &mut right,
            &ics,
            &[0, 4],
            &sections,
            &sections,
            &scalefactors,
            &scalefactors,
            Some(&ms),
            &mut random,
        )
        .unwrap();

        assert_eq!(left.windows[0], right.windows[0]);
    }

    #[test]
    fn applies_uncorrelated_pair_pns_without_ms_mask() {
        let ics = ics(vec![1], 1);
        let mut left = InverseQuantizedSpectrum {
            windows: vec![vec![0.0; 4]],
        };
        let mut right = InverseQuantizedSpectrum {
            windows: vec![vec![0.0; 4]],
        };
        let sections = sections(vec![vec![NOISE_HCB]]);
        let scalefactors = ScalefactorData {
            values: vec![vec![0]],
        };
        let mut random = PnsRandomState::new(7);

        apply_pns_pair_f32(
            &mut left,
            &mut right,
            &ics,
            &[0, 4],
            &sections,
            &sections,
            &scalefactors,
            &scalefactors,
            None,
            &mut random,
        )
        .unwrap();

        assert_ne!(left.windows[0], right.windows[0]);
    }

    #[test]
    fn empty_noise_and_integer_square_root_boundaries() {
        let mut random = PnsRandomState::new(9);
        assert_eq!(random.seed(), 9);
        assert!(normalized_noise(0, &mut random).is_empty());
        assert!(normalized_noise_fixed_q31(0, &mut random).is_empty());
        assert_eq!(random.seed(), 9);
        assert_eq!(integer_sqrt_u128(0), 0);
        assert_eq!(integer_sqrt_u128(1), 1);
        assert_eq!(integer_sqrt_u128(4), 2);
        assert_eq!(integer_sqrt_u128(8), 2);
        assert_eq!(integer_sqrt_u128(9), 3);
        assert_eq!(mul_q31_saturate(i32::MAX, i32::MAX), i32::MAX - 1);
    }

    #[test]
    fn fixed_pair_pns_supports_correlated_and_single_sided_noise() {
        let ics = ics(vec![1], 2);
        let noise = sections(vec![vec![NOISE_HCB, ZERO_HCB]]);
        let right_sections = sections(vec![vec![NOISE_HCB, NOISE_HCB]]);
        let factors = ScalefactorData {
            values: vec![vec![-4, -4]],
        };
        let ms = MsStereoData {
            mask_present: MsMaskPresent::Some,
            used: vec![vec![true, false]],
        };
        let make = || FixedInverseQuantizedSpectrum {
            windows: vec![vec![0; 4]],
            window_exponents: vec![0],
        };
        let mut left = make();
        let mut right = make();
        let mut random = PnsRandomState::new(7);
        apply_pns_pair_fixed_bridge(
            &mut left,
            &mut right,
            &ics,
            &[0, 2, 4],
            &noise,
            &right_sections,
            &factors,
            &factors,
            Some(&ms),
            &mut random,
        )
        .unwrap();
        assert_eq!(&left.windows[0][..2], &right.windows[0][..2]);
        assert_eq!(&left.windows[0][2..], &[0, 0]);
        assert!(right.windows[0][2..].iter().any(|&value| value != 0));

        let right_zero = sections(vec![vec![ZERO_HCB, ZERO_HCB]]);
        let mut left = make();
        let mut right = make();
        apply_pns_pair_fixed(
            &mut left,
            &mut right,
            &ics,
            &[0, 2, 4],
            &noise,
            &right_zero,
            &factors,
            &factors,
            None,
            &mut random,
        )
        .unwrap();
        assert!(left.windows[0][..2].iter().any(|&value| value != 0));
        assert_eq!(right.windows[0], [0; 4]);
    }

    #[test]
    fn pair_pns_handles_right_only_and_grouped_windows() {
        let ics = ics(vec![2, 1], 1);
        let zero = sections(vec![vec![ZERO_HCB], vec![ZERO_HCB]]);
        let noise = sections(vec![vec![NOISE_HCB], vec![NOISE_HCB]]);
        let factors = ScalefactorData {
            values: vec![vec![0], vec![0]],
        };
        let mut left = InverseQuantizedSpectrum {
            windows: vec![vec![3.0; 2]; 3],
        };
        let mut right = InverseQuantizedSpectrum {
            windows: vec![vec![0.0; 2]; 3],
        };
        apply_pns_pair_f32(
            &mut left,
            &mut right,
            &ics,
            &[0, 2],
            &zero,
            &noise,
            &factors,
            &factors,
            None,
            &mut PnsRandomState::new(1),
        )
        .unwrap();
        assert!(left.windows.iter().flatten().all(|&value| value == 3.0));
        assert!(right
            .windows
            .iter()
            .all(|window| window.iter().any(|&value| value != 0.0)));

        let mut left = InverseQuantizedSpectrum {
            windows: vec![vec![0.0; 2]; 3],
        };
        let mut right = InverseQuantizedSpectrum {
            windows: vec![vec![3.0; 2]; 3],
        };
        apply_pns_pair_f32(
            &mut left,
            &mut right,
            &ics,
            &[0, 2],
            &noise,
            &zero,
            &factors,
            &factors,
            None,
            &mut PnsRandomState::new(1),
        )
        .unwrap();
        assert!(left
            .windows
            .iter()
            .all(|window| window.iter().any(|&value| value != 0.0)));
        assert!(right.windows.iter().flatten().all(|&value| value == 3.0));
    }

    #[test]
    fn fixed_single_channel_bridge_respects_window_exponent() {
        let ics = ics(vec![1], 1);
        let mut spectrum = FixedInverseQuantizedSpectrum {
            windows: vec![vec![0; 4]],
            window_exponents: vec![2],
        };
        apply_pns_fixed_bridge(
            &mut spectrum,
            &ics,
            &[0, 4],
            &sections(vec![vec![NOISE_HCB]]),
            &ScalefactorData {
                values: vec![vec![0]],
            },
            &mut PnsRandomState::new(2),
        )
        .unwrap();
        assert!(spectrum.windows[0].iter().any(|&value| value != 0));
        assert!(spectrum.windows[0]
            .iter()
            .all(|&value| value != i32::MAX && value != i32::MIN + 1));
    }

    #[test]
    fn layout_validation_rejects_each_mismatched_component() {
        let ics = ics(vec![1], 1);
        let good_sections = sections(vec![vec![NOISE_HCB]]);
        let good_factors = ScalefactorData {
            values: vec![vec![0]],
        };
        let mut random = PnsRandomState::new(1);
        for (windows, offsets) in [(vec![], vec![0, 1]), (vec![vec![0.0]], vec![0])] {
            assert_eq!(
                apply_pns_f32(
                    &mut InverseQuantizedSpectrum { windows },
                    &ics,
                    &offsets,
                    &good_sections,
                    &good_factors,
                    &mut random
                ),
                Err(PnsError::LayoutMismatch)
            );
        }
        assert_eq!(
            apply_pns_f32(
                &mut InverseQuantizedSpectrum {
                    windows: vec![vec![]]
                },
                &ics,
                &[0, 1],
                &good_sections,
                &good_factors,
                &mut random
            ),
            Err(PnsError::LayoutMismatch)
        );
        assert_eq!(
            apply_pns_f32(
                &mut InverseQuantizedSpectrum {
                    windows: vec![vec![0.0]]
                },
                &ics,
                &[0, 1],
                &sections(vec![vec![]]),
                &good_factors,
                &mut random
            ),
            Err(PnsError::LayoutMismatch)
        );
        assert_eq!(
            apply_pns_fixed(
                &mut FixedInverseQuantizedSpectrum {
                    windows: vec![vec![0]],
                    window_exponents: vec![0]
                },
                &ics,
                &[0, 1],
                &good_sections,
                &ScalefactorData { values: vec![] },
                &mut random
            ),
            Err(PnsError::LayoutMismatch)
        );
        for windows in [vec![], vec![vec![]]] {
            assert_eq!(
                apply_pns_fixed(
                    &mut FixedInverseQuantizedSpectrum {
                        windows,
                        window_exponents: vec![0],
                    },
                    &ics,
                    &[0, 1],
                    &good_sections,
                    &good_factors,
                    &mut random,
                ),
                Err(PnsError::LayoutMismatch)
            );
        }
        assert_eq!(PnsError::LayoutMismatch.to_string(), "PNS layout mismatch");
    }
}
