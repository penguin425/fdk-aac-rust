//! Reference f32 inverse quantization for AAC-LC spectral coefficients.

use std::fmt;

use crate::fixed::{FixpDbl, MAXVAL_DBL, MINVAL_DBL_PLUS_ONE};
use crate::ics::IcsInfo;
use crate::scalefactor::ScalefactorData;
use crate::sfb::{aac_lc_band_offsets_for_ics, ScaleFactorBandInfo, SfbError};
use crate::spectral::SpectralData;

#[derive(Debug, Clone, PartialEq)]
pub struct InverseQuantizedSpectrum {
    /// Per-window inverse-quantized MDCT coefficients in f32 reference form.
    pub windows: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedInverseQuantizedSpectrum {
    /// Per-window inverse-quantized MDCT coefficients in Q31-like fixed form.
    pub windows: Vec<Vec<FixpDbl>>,
    /// Binary exponent for each window. A coefficient represents
    /// `windows[w][i] / 2^31 * 2^window_exponents[w]`.
    pub window_exponents: Vec<i16>,
}

impl FixedInverseQuantizedSpectrum {
    pub fn from_f32_bridge(spectrum: &InverseQuantizedSpectrum) -> Self {
        Self {
            windows: spectrum
                .windows
                .iter()
                .map(|window| {
                    window
                        .iter()
                        .map(|&value| inverse_value_f32_to_q31(value))
                        .collect()
                })
                .collect(),
            window_exponents: vec![0; spectrum.windows.len()],
        }
    }
}

pub fn inverse_value_f32_to_q31(value: f32) -> FixpDbl {
    if !value.is_finite() {
        return 0;
    }
    if value >= 1.0 {
        MAXVAL_DBL
    } else if value <= -1.0 {
        MINVAL_DBL_PLUS_ONE
    } else {
        (value * 2_147_483_648.0).round() as FixpDbl
    }
}

pub fn inverse_quantize_value_f32(quantized: i32, scalefactor: i16) -> f32 {
    if quantized == 0 {
        return 0.0;
    }
    let sign = if quantized < 0 { -1.0 } else { 1.0 };
    let magnitude = (quantized.abs() as f32).powf(4.0 / 3.0);
    let gain = 2.0f32.powf(0.25 * scalefactor as f32);
    sign * magnitude * gain
}

pub fn inverse_quantize_value_fixed_bridge(quantized: i32, scalefactor: i16) -> FixpDbl {
    inverse_quantize_value_fixed(quantized, scalefactor)
}

pub fn inverse_quantize_value_fixed(quantized: i32, scalefactor: i16) -> FixpDbl {
    inverse_quantize_value_fixed_int(quantized, scalefactor)
        .expect("fixed inverse quantization supports all scalefactors")
}

pub fn inverse_quantize_value_fixed_int(quantized: i32, scalefactor: i16) -> Option<FixpDbl> {
    if quantized == 0 {
        return Some(0);
    }

    let magnitude = quantized.unsigned_abs() as u128;
    let pow43_q16 = pow43_u16_fractional(magnitude);
    let octave = (scalefactor as i32).div_euclid(4);
    let gain_q16 = scalefactor_remainder_gain_q16(scalefactor);
    let scaled_q32 = pow43_q16.saturating_mul(gain_q16 as u128);
    let shift = octave - 1;
    let scaled = if shift >= 0 {
        let shift = shift.min(127) as u32;
        let shifted = if scaled_q32 > (u128::MAX >> shift) {
            u128::MAX
        } else {
            scaled_q32 << shift
        };
        shifted.min(i32::MAX as u128) as i64
    } else {
        let rshift = (-shift).min(127) as u32;
        ((scaled_q32 + (1u128 << rshift.saturating_sub(1))) >> rshift) as i64
    };

    let signed = if quantized < 0 { -scaled } else { scaled };
    Some(signed.clamp(MINVAL_DBL_PLUS_ONE as i64, MAXVAL_DBL as i64) as FixpDbl)
}

fn scalefactor_remainder_gain_q16(scalefactor: i16) -> u32 {
    const GAIN_Q16: [u32; 4] = [65_536, 77_936, 92_682, 110_218];
    GAIN_Q16[(scalefactor as i32).rem_euclid(4) as usize]
}

fn pow43_u16_fractional(magnitude: u128) -> u128 {
    if magnitude == 0 {
        return 0;
    }
    let magnitude_fourth = magnitude
        .saturating_mul(magnitude)
        .saturating_mul(magnitude)
        .saturating_mul(magnitude);
    let radicand = if magnitude_fourth > (u128::MAX >> 48) {
        u128::MAX
    } else {
        magnitude_fourth << 48
    };
    integer_cuberoot_u128(radicand)
}

fn integer_cuberoot_u128(value: u128) -> u128 {
    if value < 2 {
        return value;
    }
    let mut low = 1u128;
    let mut high = 1u128 << ((128 - value.leading_zeros() as usize + 2) / 3).min(43);
    while low <= high {
        let mid = (low + high) >> 1;
        let cube = mid.saturating_mul(mid).saturating_mul(mid);
        if cube == value {
            return mid;
        } else if cube < value {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    high
}

pub fn inverse_quantize_spectrum_f32(
    spectral: &SpectralData,
    scalefactors: &ScalefactorData,
    ics: &IcsInfo,
    sfb: ScaleFactorBandInfo,
) -> Result<InverseQuantizedSpectrum, InverseQuantError> {
    validate_layout(spectral, scalefactors, ics, sfb)?;

    let mut output = spectral
        .windows
        .iter()
        .map(|window| vec![0.0f32; window.len()])
        .collect::<Vec<_>>();

    let mut window_offset = 0usize;
    for (group, &group_len) in ics.window_group_lengths.iter().enumerate() {
        for band in 0..ics.max_sfb as usize {
            let scale = scalefactors.values[group][band];
            let band_start = sfb.offsets[band];
            let band_end = sfb.offsets[band + 1];
            for group_window in 0..group_len as usize {
                let window = window_offset + group_window;
                for index in band_start..band_end {
                    output[window][index] =
                        inverse_quantize_value_f32(spectral.windows[window][index], scale);
                }
            }
        }
        window_offset += group_len as usize;
    }

    Ok(InverseQuantizedSpectrum { windows: output })
}

pub fn inverse_quantize_spectrum_fixed_bridge(
    spectral: &SpectralData,
    scalefactors: &ScalefactorData,
    ics: &IcsInfo,
    sfb: ScaleFactorBandInfo,
) -> Result<FixedInverseQuantizedSpectrum, InverseQuantError> {
    inverse_quantize_spectrum_fixed(spectral, scalefactors, ics, sfb)
}

pub fn inverse_quantize_spectrum_fixed(
    spectral: &SpectralData,
    scalefactors: &ScalefactorData,
    ics: &IcsInfo,
    sfb: ScaleFactorBandInfo,
) -> Result<FixedInverseQuantizedSpectrum, InverseQuantError> {
    validate_layout(spectral, scalefactors, ics, sfb)?;

    let mut output = spectral
        .windows
        .iter()
        .map(|window| vec![0; window.len()])
        .collect::<Vec<_>>();

    let mut window_offset = 0usize;
    for (group, &group_len) in ics.window_group_lengths.iter().enumerate() {
        for band in 0..ics.max_sfb as usize {
            let scale = scalefactors.values[group][band];
            let band_start = sfb.offsets[band];
            let band_end = sfb.offsets[band + 1];
            for group_window in 0..group_len as usize {
                let window = window_offset + group_window;
                for index in band_start..band_end {
                    output[window][index] =
                        inverse_quantize_value_fixed(spectral.windows[window][index], scale);
                }
            }
        }
        window_offset += group_len as usize;
    }

    Ok(FixedInverseQuantizedSpectrum {
        window_exponents: vec![0; output.len()],
        windows: output,
    })
}

pub fn inverse_quantize_spectrum_fixed_block_scaled(
    spectral: &SpectralData,
    scalefactors: &ScalefactorData,
    ics: &IcsInfo,
    sfb: ScaleFactorBandInfo,
) -> Result<FixedInverseQuantizedSpectrum, InverseQuantError> {
    let float = inverse_quantize_spectrum_f32(spectral, scalefactors, ics, sfb)?;
    let mut windows = Vec::with_capacity(float.windows.len());
    let mut window_exponents = Vec::with_capacity(float.windows.len());
    for window in float.windows {
        let maximum = window.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
        let exponent = if maximum > 0.0 {
            // Match CBlock_ScaleSpectralData's TNS headroom reservation. The
            // lattice filter can grow a band after inverse quantization, so
            // leave four mantissa bits rather than saturating that growth.
            maximum.log2().floor() as i16 + 5
        } else {
            0
        };
        let scale = 2.0f32.powi(-(exponent as i32));
        windows.push(
            window
                .into_iter()
                .map(|value| inverse_value_f32_to_q31(value * scale))
                .collect(),
        );
        window_exponents.push(exponent);
    }
    Ok(FixedInverseQuantizedSpectrum {
        windows,
        window_exponents,
    })
}

pub fn inverse_quantize_aac_lc_fixed_bridge(
    spectral: &SpectralData,
    scalefactors: &ScalefactorData,
    sampling_frequency_index: u8,
    ics: &IcsInfo,
) -> Result<FixedInverseQuantizedSpectrum, InverseQuantError> {
    inverse_quantize_aac_lc_fixed(spectral, scalefactors, sampling_frequency_index, ics)
}

pub fn inverse_quantize_aac_lc_fixed(
    spectral: &SpectralData,
    scalefactors: &ScalefactorData,
    sampling_frequency_index: u8,
    ics: &IcsInfo,
) -> Result<FixedInverseQuantizedSpectrum, InverseQuantError> {
    let sfb = aac_lc_band_offsets_for_ics(sampling_frequency_index, ics)?;
    inverse_quantize_spectrum_fixed(spectral, scalefactors, ics, sfb)
}

pub fn inverse_quantize_aac_lc_f32(
    spectral: &SpectralData,
    scalefactors: &ScalefactorData,
    sampling_frequency_index: u8,
    ics: &IcsInfo,
) -> Result<InverseQuantizedSpectrum, InverseQuantError> {
    let sfb = aac_lc_band_offsets_for_ics(sampling_frequency_index, ics)?;
    inverse_quantize_spectrum_f32(spectral, scalefactors, ics, sfb)
}

fn validate_layout(
    spectral: &SpectralData,
    scalefactors: &ScalefactorData,
    ics: &IcsInfo,
    sfb: ScaleFactorBandInfo,
) -> Result<(), InverseQuantError> {
    let total_windows = ics
        .window_group_lengths
        .iter()
        .map(|&len| len as usize)
        .sum::<usize>();
    if spectral.windows.len() != total_windows {
        return Err(InverseQuantError::LayoutMismatch);
    }
    if spectral
        .windows
        .iter()
        .any(|window| window.len() != sfb.granule_length)
    {
        return Err(InverseQuantError::LayoutMismatch);
    }
    if scalefactors.values.len() != ics.window_group_lengths.len()
        || scalefactors
            .values
            .iter()
            .any(|group| group.len() < ics.max_sfb as usize)
    {
        return Err(InverseQuantError::LayoutMismatch);
    }
    if ics.max_sfb > sfb.num_bands {
        return Err(InverseQuantError::Sfb(SfbError::MaxSfbOutOfRange {
            max_sfb: ics.max_sfb,
            total_sfb: sfb.num_bands,
        }));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InverseQuantError {
    Sfb(SfbError),
    LayoutMismatch,
}

impl From<SfbError> for InverseQuantError {
    fn from(value: SfbError) -> Self {
        Self::Sfb(value)
    }
}

impl fmt::Display for InverseQuantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sfb(err) => err.fmt(f),
            Self::LayoutMismatch => write!(f, "AAC inverse quantization layout mismatch"),
        }
    }
}

impl std::error::Error for InverseQuantError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ics::{IcsInfo, WindowSequence, WindowShape};
    use crate::sfb::SFB_48_1024;

    fn ics(window_group_lengths: Vec<u8>, max_sfb: u8) -> IcsInfo {
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

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1.0e-5,
            "actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn inverse_quantizes_single_values_using_aac_formula() {
        assert_eq!(inverse_quantize_value_f32(0, 100), 0.0);
        assert_close(inverse_quantize_value_f32(1, 0), 1.0);
        assert_close(inverse_quantize_value_f32(-8, 0), -16.0);
        assert_close(inverse_quantize_value_f32(1, 4), 2.0);
    }

    #[test]
    fn fixed_bridge_inverse_quantizes_single_values() {
        assert_eq!(inverse_quantize_value_fixed_bridge(0, 100), 0);
        assert_eq!(inverse_quantize_value_fixed_bridge(1, -4), 0x4000_0000);
        assert_eq!(inverse_quantize_value_fixed_bridge(-1, -4), -0x4000_0000);
        assert_eq!(inverse_quantize_value_fixed_bridge(1, 0), MAXVAL_DBL);
        assert_eq!(
            inverse_quantize_value_fixed_bridge(-8, 0),
            MINVAL_DBL_PLUS_ONE
        );
    }

    #[test]
    fn fixed_inverse_quantizes_single_values_without_f32_fallback() {
        assert_eq!(inverse_quantize_value_fixed(0, 100), 0);
        assert_eq!(inverse_quantize_value_fixed(1, -4), 0x4000_0000);
        assert_eq!(inverse_quantize_value_fixed(-1, -4), -0x4000_0000);
        assert_eq!(inverse_quantize_value_fixed(1, 0), MAXVAL_DBL);
    }

    #[test]
    fn fixed_integer_inverse_quantizes_scalefactor_multiples_of_four() {
        assert_eq!(inverse_quantize_value_fixed_int(0, 0), Some(0));
        assert_eq!(inverse_quantize_value_fixed_int(1, -4), Some(0x4000_0000));
        assert_eq!(inverse_quantize_value_fixed_int(-1, -4), Some(-0x4000_0000));
        assert_eq!(inverse_quantize_value_fixed_int(1, -8), Some(0x2000_0000));
    }

    #[test]
    fn fixed_integer_inverse_quantizes_all_scalefactor_remainders() {
        for scalefactor in -8..=8 {
            let fixed = inverse_quantize_value_fixed_int(1, scalefactor).unwrap();
            let reference = inverse_value_f32_to_q31(inverse_quantize_value_f32(1, scalefactor));
            assert!(
                (fixed as i64 - reference as i64).abs() <= 16_384,
                "scalefactor={scalefactor}, fixed={fixed}, reference={reference}"
            );
        }
    }

    #[test]
    fn integer_pow43_q16_tracks_exact_cubes() {
        assert_eq!(pow43_u16_fractional(1), 1 << 16);
        assert_eq!(pow43_u16_fractional(8), 16 << 16);
        assert_eq!(pow43_u16_fractional(27), 81 << 16);
    }

    #[test]
    fn inverse_quantizes_bands_per_grouped_window() {
        let ics = ics(vec![2], 2);
        let spectral = SpectralData {
            windows: vec![vec![1, -8, 0, 0, 1, 1, 0, 0], vec![0, 0, 0, 0, -8, 1, 0, 0]],
        };
        let scalefactors = ScalefactorData {
            values: vec![vec![0, 4]],
        };
        let sfb = ScaleFactorBandInfo {
            offsets: &[0, 4, 8],
            num_bands: 2,
            granule_length: 8,
        };

        let inverse = inverse_quantize_spectrum_f32(&spectral, &scalefactors, &ics, sfb).unwrap();
        assert_close(inverse.windows[0][0], 1.0);
        assert_close(inverse.windows[0][1], -16.0);
        assert_close(inverse.windows[0][4], 2.0);
        assert_close(inverse.windows[1][4], -32.0);
    }

    #[test]
    fn fixed_bridge_inverse_quantizes_bands_per_grouped_window() {
        let ics = ics(vec![2], 2);
        let spectral = SpectralData {
            windows: vec![vec![1, -1, 0, 0, 1, 1, 0, 0], vec![0, 0, 0, 0, -1, 1, 0, 0]],
        };
        let scalefactors = ScalefactorData {
            values: vec![vec![-4, -4]],
        };
        let sfb = ScaleFactorBandInfo {
            offsets: &[0, 4, 8],
            num_bands: 2,
            granule_length: 8,
        };

        let fixed =
            inverse_quantize_spectrum_fixed_bridge(&spectral, &scalefactors, &ics, sfb).unwrap();

        assert_eq!(fixed.windows[0][0], 0x4000_0000);
        assert_eq!(fixed.windows[0][1], -0x4000_0000);
        assert_eq!(fixed.windows[0][4], 0x4000_0000);
        assert_eq!(fixed.windows[1][4], -0x4000_0000);
    }

    #[test]
    fn inverse_quantizes_with_aac_lc_sfb_lookup() {
        let ics = ics(vec![1], 1);
        let mut window = vec![0; 1024];
        window[0] = 1;
        window[SFB_48_1024[1] - 1] = -8;
        let spectral = SpectralData {
            windows: vec![window],
        };
        let scalefactors = ScalefactorData {
            values: vec![vec![0]],
        };

        let inverse = inverse_quantize_aac_lc_f32(&spectral, &scalefactors, 4, &ics).unwrap();
        assert_eq!(inverse.windows[0].len(), 1024);
        assert_close(inverse.windows[0][0], 1.0);
        assert_close(inverse.windows[0][SFB_48_1024[1] - 1], -16.0);
    }

    #[test]
    fn rejects_layout_mismatch() {
        let ics = ics(vec![1], 1);
        let spectral = SpectralData {
            windows: vec![vec![0; 4]],
        };
        let scalefactors = ScalefactorData { values: vec![] };
        let sfb = ScaleFactorBandInfo {
            offsets: &[0, 4],
            num_bands: 1,
            granule_length: 4,
        };

        assert_eq!(
            inverse_quantize_spectrum_f32(&spectral, &scalefactors, &ics, sfb).unwrap_err(),
            InverseQuantError::LayoutMismatch
        );
    }

    #[test]
    fn bridges_inverse_quantized_spectrum_to_fixed_q31() {
        let spectrum = InverseQuantizedSpectrum {
            windows: vec![vec![0.5, -0.5, 0.0, 1.5, -1.5, f32::NAN]],
        };
        let fixed = FixedInverseQuantizedSpectrum::from_f32_bridge(&spectrum);

        assert_eq!(fixed.windows.len(), 1);
        assert_eq!(fixed.windows[0][0], 0x4000_0000);
        assert_eq!(fixed.windows[0][1], -0x4000_0000);
        assert_eq!(fixed.windows[0][2], 0);
        assert_eq!(fixed.windows[0][3], MAXVAL_DBL);
        assert_eq!(fixed.windows[0][4], MINVAL_DBL_PLUS_ONE);
        assert_eq!(fixed.windows[0][5], 0);
    }

    #[test]
    fn block_scaled_inverse_preserves_coefficients_above_q31_range() {
        let ics = ics(vec![1], 1);
        let spectral = SpectralData {
            windows: vec![vec![8, -27, 1, 0]],
        };
        let scalefactors = ScalefactorData {
            values: vec![vec![0]],
        };
        let sfb = ScaleFactorBandInfo {
            offsets: &[0, 4],
            num_bands: 1,
            granule_length: 4,
        };
        let fixed =
            inverse_quantize_spectrum_fixed_block_scaled(&spectral, &scalefactors, &ics, sfb)
                .unwrap();
        let exponent = fixed.window_exponents[0] as i32;
        let reconstructed = fixed.windows[0]
            .iter()
            .map(|&value| value as f64 / 2_147_483_648.0 * 2.0f64.powi(exponent))
            .collect::<Vec<_>>();
        for (actual, expected) in reconstructed.iter().zip([16.0, -81.0, 1.0, 0.0]) {
            assert!((actual - expected).abs() < 1.0e-4);
        }
        assert!(fixed.windows[0]
            .iter()
            .all(|&value| value != MAXVAL_DBL && value != MINVAL_DBL_PLUS_ONE));
    }

    #[test]
    fn q31_bridge_handles_finite_boundaries_and_non_finite_values() {
        assert_eq!(inverse_value_f32_to_q31(f32::INFINITY), 0);
        assert_eq!(inverse_value_f32_to_q31(f32::NEG_INFINITY), 0);
        assert_eq!(inverse_value_f32_to_q31(1.0), MAXVAL_DBL);
        assert_eq!(inverse_value_f32_to_q31(-1.0), MINVAL_DBL_PLUS_ONE);
        assert_eq!(inverse_value_f32_to_q31(0.25), 0x2000_0000);
    }

    #[test]
    fn integer_inverse_saturates_extreme_scalefactors_and_signs() {
        assert_eq!(
            inverse_quantize_value_fixed_int(i32::MAX, i16::MAX),
            Some(MAXVAL_DBL)
        );
        assert_eq!(
            inverse_quantize_value_fixed_int(i32::MIN, i16::MAX),
            Some(MINVAL_DBL_PLUS_ONE)
        );
        assert_eq!(inverse_quantize_value_fixed_int(1, i16::MIN), Some(0));
        assert_eq!(inverse_quantize_value_fixed_bridge(1, -4), 0x4000_0000);
    }

    #[test]
    fn integer_cube_root_covers_zero_exact_and_floor_results() {
        assert_eq!(pow43_u16_fractional(0), 0);
        assert_eq!(integer_cuberoot_u128(0), 0);
        assert_eq!(integer_cuberoot_u128(1), 1);
        assert_eq!(integer_cuberoot_u128(8), 2);
        assert_eq!(integer_cuberoot_u128(26), 2);
        assert_eq!(integer_cuberoot_u128(27), 3);
    }

    #[test]
    fn fixed_aac_wrappers_match_direct_table_path() {
        let ics = ics(vec![1], 1);
        let spectral = SpectralData {
            windows: vec![vec![0; 1024]],
        };
        let scalefactors = ScalefactorData {
            values: vec![vec![-4]],
        };
        let direct = inverse_quantize_aac_lc_fixed(&spectral, &scalefactors, 4, &ics).unwrap();
        assert_eq!(
            inverse_quantize_aac_lc_fixed_bridge(&spectral, &scalefactors, 4, &ics).unwrap(),
            direct
        );
        assert_eq!(direct.windows, vec![vec![0; 1024]]);
    }

    #[test]
    fn layout_validator_reports_each_dimension_mismatch() {
        let ics = ics(vec![1], 1);
        let sfb = ScaleFactorBandInfo {
            offsets: &[0, 4],
            num_bands: 1,
            granule_length: 4,
        };
        let factors = ScalefactorData {
            values: vec![vec![0]],
        };
        assert_eq!(
            inverse_quantize_spectrum_fixed(
                &SpectralData {
                    windows: Vec::new()
                },
                &factors,
                &ics,
                sfb
            ),
            Err(InverseQuantError::LayoutMismatch)
        );
        assert_eq!(
            inverse_quantize_spectrum_fixed(
                &SpectralData {
                    windows: vec![vec![0; 3]]
                },
                &factors,
                &ics,
                sfb
            ),
            Err(InverseQuantError::LayoutMismatch)
        );
        assert_eq!(
            inverse_quantize_spectrum_fixed(
                &SpectralData {
                    windows: vec![vec![0; 4]]
                },
                &ScalefactorData {
                    values: vec![vec![]]
                },
                &ics,
                sfb
            ),
            Err(InverseQuantError::LayoutMismatch)
        );
        let too_many_bands = ScaleFactorBandInfo {
            num_bands: 0,
            ..sfb
        };
        assert!(matches!(
            inverse_quantize_spectrum_fixed(
                &SpectralData {
                    windows: vec![vec![0; 4]]
                },
                &factors,
                &ics,
                too_many_bands
            ),
            Err(InverseQuantError::Sfb(SfbError::MaxSfbOutOfRange { .. }))
        ));
    }

    #[test]
    fn block_scaled_zero_window_uses_zero_exponent() {
        let ics = ics(vec![1], 1);
        let result = inverse_quantize_spectrum_fixed_block_scaled(
            &SpectralData {
                windows: vec![vec![0; 4]],
            },
            &ScalefactorData {
                values: vec![vec![0]],
            },
            &ics,
            ScaleFactorBandInfo {
                offsets: &[0, 4],
                num_bands: 1,
                granule_length: 4,
            },
        )
        .unwrap();
        assert_eq!(result.window_exponents, [0]);
        assert_eq!(result.windows, [vec![0; 4]]);
    }

    #[test]
    fn formats_and_converts_inverse_errors() {
        let sfb = SfbError::UnsupportedSamplingFrequencyIndex(13);
        assert_eq!(
            InverseQuantError::from(sfb.clone()),
            InverseQuantError::Sfb(sfb.clone())
        );
        assert_eq!(
            InverseQuantError::Sfb(sfb.clone()).to_string(),
            sfb.to_string()
        );
        assert_eq!(
            InverseQuantError::LayoutMismatch.to_string(),
            "AAC inverse quantization layout mismatch"
        );
    }
}
