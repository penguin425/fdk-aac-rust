//! Stateful PCM concealment used while the spectral concealment path is being
//! migrated from `libAACdec/conceal.cpp`.

use std::fmt;

use crate::fixed::{mul_q31, FixpDbl};
use crate::ics::WindowSequence;
use crate::inverse::{FixedInverseQuantizedSpectrum, InverseQuantizedSpectrum};

const FDK_DEFAULT_FADE_FACTOR: f32 = 0.707_106_77; // 1/sqrt(2)
const FDK_DEFAULT_FADE_OUT_FRAMES: usize = 6;
const FDK_DEFAULT_FADE_IN_FRAMES: usize = 5;

#[derive(Debug, Clone, PartialEq)]
pub struct PcmConcealment {
    last_good: Vec<f32>,
    consecutive_losses: usize,
    fade_factor: f32,
    fade_out_frames: usize,
    fade_in_frames: usize,
}

impl Default for PcmConcealment {
    fn default() -> Self {
        Self {
            last_good: Vec::new(),
            consecutive_losses: 0,
            fade_factor: FDK_DEFAULT_FADE_FACTOR,
            fade_out_frames: FDK_DEFAULT_FADE_OUT_FRAMES,
            fade_in_frames: FDK_DEFAULT_FADE_IN_FRAMES,
        }
    }
}

impl PcmConcealment {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn consecutive_losses(&self) -> usize {
        self.consecutive_losses
    }

    /// Insert `lost_frames` before a valid decoded frame and apply the FDK
    /// default frame-level fade factors. If no earlier good frame exists,
    /// silence with the recovered frame's layout is emitted.
    pub fn process_f32(&mut self, good: Vec<f32>, lost_frames: usize) -> Vec<Vec<f32>> {
        let mut output = Vec::with_capacity(lost_frames + 1);
        for _ in 0..lost_frames {
            let factor = self.loss_factor();
            let concealed = if self.last_good.is_empty() {
                vec![0.0; good.len()]
            } else {
                self.last_good
                    .iter()
                    .map(|sample| sample * factor)
                    .collect()
            };
            output.push(concealed);
            self.consecutive_losses = self.consecutive_losses.saturating_add(1);
        }

        let recovered = if self.consecutive_losses == 0 {
            good.clone()
        } else {
            let exponent = self.consecutive_losses.min(self.fade_in_frames) as i32;
            let start = self.fade_factor.powi(exponent);
            fade_frame_f32(&good, start, 1.0)
        };
        self.last_good = good;
        self.consecutive_losses = 0;
        output.push(recovered);
        output
    }

    pub fn process_i16(&mut self, good: Vec<i16>, lost_frames: usize) -> Vec<Vec<i16>> {
        let normalized = good
            .iter()
            .map(|&sample| sample as f32 / 32768.0)
            .collect::<Vec<_>>();
        self.process_f32(normalized, lost_frames)
            .into_iter()
            .map(|frame| frame.into_iter().map(f32_to_i16).collect())
            .collect()
    }

    fn loss_factor(&self) -> f32 {
        if self.consecutive_losses >= self.fade_out_frames {
            0.0
        } else {
            self.fade_factor.powi((self.consecutive_losses + 1) as i32)
        }
    }
}

fn fade_frame_f32(samples: &[f32], start: f32, stop: f32) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }
    let denominator = samples.len().saturating_sub(1).max(1) as f32;
    samples
        .iter()
        .enumerate()
        .map(|(index, &sample)| {
            let factor = start + (stop - start) * index as f32 / denominator;
            sample * factor
        })
        .collect()
}

fn f32_to_i16(sample: f32) -> i16 {
    (sample * 32768.0)
        .round()
        .clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

/// FDK `ConcealMethodInter` band-energy interpolation for spectra with their
/// scale already folded into Q31 coefficients. The previous spectrum supplies
/// phase/shape; each band is scaled to the geometric mean energy of the two
/// surrounding good frames.
pub fn interpolate_fixed_spectra(
    previous: &FixedInverseQuantizedSpectrum,
    next: &FixedInverseQuantizedSpectrum,
    band_offsets: &[usize],
) -> Result<FixedInverseQuantizedSpectrum, SpectralInterpolationError> {
    if previous.windows.len() != next.windows.len()
        || band_offsets.len() < 2
        || band_offsets[0] != 0
    {
        return Err(SpectralInterpolationError::LayoutMismatch);
    }
    let mut output = previous.clone();
    for ((out, prev), next) in output
        .windows
        .iter_mut()
        .zip(&previous.windows)
        .zip(&next.windows)
    {
        if prev.len() != next.len()
            || band_offsets.last().copied().unwrap_or(0) > prev.len()
            || band_offsets.windows(2).any(|pair| pair[0] > pair[1])
        {
            return Err(SpectralInterpolationError::LayoutMismatch);
        }
        for band in band_offsets.windows(2) {
            let start = band[0];
            let stop = band[1];
            let previous_exponent = band_energy_exponent(&prev[start..stop]);
            let next_exponent = band_energy_exponent(&next[start..stop]);
            interpolate_fixed_band(&mut out[start..stop], previous_exponent, next_exponent);
        }
    }
    Ok(output)
}

/// Window-sequence-aware counterpart of [`interpolate_fixed_spectra`], matching
/// FDK's long/short transition branches.
pub fn interpolate_fixed_spectra_mixed(
    previous: &FixedInverseQuantizedSpectrum,
    previous_sequence: WindowSequence,
    next: &FixedInverseQuantizedSpectrum,
    next_sequence: WindowSequence,
    long_band_offsets: &[usize],
    short_band_offsets: &[usize],
) -> Result<(FixedInverseQuantizedSpectrum, WindowSequence), SpectralInterpolationError> {
    let previous_short = previous_sequence == WindowSequence::EightShort;
    let next_short = next_sequence == WindowSequence::EightShort;
    match (previous_short, next_short) {
        (false, false) => Ok((
            interpolate_fixed_spectra(previous, next, long_band_offsets)?,
            WindowSequence::OnlyLong,
        )),
        (true, true) => Ok((
            interpolate_fixed_spectra(previous, next, short_band_offsets)?,
            WindowSequence::EightShort,
        )),
        (false, true) => {
            let next_first = next
                .windows
                .first()
                .ok_or(SpectralInterpolationError::LayoutMismatch)?;
            let long_len = previous
                .windows
                .first()
                .ok_or(SpectralInterpolationError::LayoutMismatch)?
                .len();
            let expanded_next = expand_short_window(next_first, long_len)?;
            let expanded = FixedInverseQuantizedSpectrum {
                windows: vec![expanded_next],
                window_exponents: vec![next.window_exponents.first().copied().unwrap_or(0)],
            };
            Ok((
                interpolate_fixed_spectra(previous, &expanded, long_band_offsets)?,
                WindowSequence::LongStart,
            ))
        }
        (true, false) => {
            let previous_last = previous
                .windows
                .last()
                .ok_or(SpectralInterpolationError::LayoutMismatch)?;
            let long_len = next
                .windows
                .first()
                .ok_or(SpectralInterpolationError::LayoutMismatch)?
                .len();
            let expanded_previous = FixedInverseQuantizedSpectrum {
                windows: vec![expand_short_window(previous_last, long_len)?],
                window_exponents: vec![previous.window_exponents.last().copied().unwrap_or(0)],
            };
            Ok((
                interpolate_fixed_spectra(next, &expanded_previous, long_band_offsets)?,
                WindowSequence::LongStop,
            ))
        }
    }
}

fn expand_short_window(
    short: &[FixpDbl],
    long_len: usize,
) -> Result<Vec<FixpDbl>, SpectralInterpolationError> {
    if short.is_empty() || long_len != short.len() * 8 {
        return Err(SpectralInterpolationError::LayoutMismatch);
    }
    Ok((0..long_len).map(|line| short[line >> 3]).collect())
}

pub fn interpolate_f32_spectra_mixed(
    previous: &InverseQuantizedSpectrum,
    previous_sequence: WindowSequence,
    next: &InverseQuantizedSpectrum,
    next_sequence: WindowSequence,
    long_band_offsets: &[usize],
    short_band_offsets: &[usize],
) -> Result<(InverseQuantizedSpectrum, WindowSequence), SpectralInterpolationError> {
    let previous_short = previous_sequence == WindowSequence::EightShort;
    let next_short = next_sequence == WindowSequence::EightShort;
    match (previous_short, next_short) {
        (false, false) => Ok((
            interpolate_f32_spectra(previous, next, long_band_offsets)?,
            WindowSequence::OnlyLong,
        )),
        (true, true) => Ok((
            interpolate_f32_spectra(previous, next, short_band_offsets)?,
            WindowSequence::EightShort,
        )),
        (false, true) => {
            let next_first = next
                .windows
                .first()
                .ok_or(SpectralInterpolationError::LayoutMismatch)?;
            let long_len = previous
                .windows
                .first()
                .ok_or(SpectralInterpolationError::LayoutMismatch)?
                .len();
            let expanded = InverseQuantizedSpectrum {
                windows: vec![expand_short_window_f32(next_first, long_len)?],
            };
            Ok((
                interpolate_f32_spectra(previous, &expanded, long_band_offsets)?,
                WindowSequence::LongStart,
            ))
        }
        (true, false) => {
            let previous_last = previous
                .windows
                .last()
                .ok_or(SpectralInterpolationError::LayoutMismatch)?;
            let long_len = next
                .windows
                .first()
                .ok_or(SpectralInterpolationError::LayoutMismatch)?
                .len();
            let expanded = InverseQuantizedSpectrum {
                windows: vec![expand_short_window_f32(previous_last, long_len)?],
            };
            Ok((
                interpolate_f32_spectra(next, &expanded, long_band_offsets)?,
                WindowSequence::LongStop,
            ))
        }
    }
}

fn interpolate_f32_spectra(
    previous: &InverseQuantizedSpectrum,
    next: &InverseQuantizedSpectrum,
    band_offsets: &[usize],
) -> Result<InverseQuantizedSpectrum, SpectralInterpolationError> {
    if previous.windows.len() != next.windows.len()
        || band_offsets.len() < 2
        || band_offsets[0] != 0
    {
        return Err(SpectralInterpolationError::LayoutMismatch);
    }
    let mut output = previous.clone();
    for ((out, previous), next) in output
        .windows
        .iter_mut()
        .zip(&previous.windows)
        .zip(&next.windows)
    {
        if previous.len() != next.len()
            || band_offsets.last().copied().unwrap_or(0) > previous.len()
            || band_offsets.windows(2).any(|pair| pair[0] > pair[1])
        {
            return Err(SpectralInterpolationError::LayoutMismatch);
        }
        for band in band_offsets.windows(2) {
            let start = band[0];
            let stop = band[1];
            let previous_energy = f32_band_energy(&previous[start..stop]);
            let next_energy = f32_band_energy(&next[start..stop]);
            // FDK derives this factor from bounded energy exponents and clamps
            // the resulting shift to the fixed-point word width. Computing the
            // ratio directly can overflow for a silent previous band and turn
            // `0 * inf` into NaN, so retain the same bounded-exponent model.
            let max_log_gain = 31.0 * std::f64::consts::LN_2;
            let log_gain = 0.25 * (next_energy.ln() - previous_energy.ln());
            let gain = log_gain.clamp(-max_log_gain, max_log_gain).exp() as f32;
            for line in &mut out[start..stop] {
                *line *= gain;
            }
        }
    }
    Ok(output)
}

fn f32_band_energy(lines: &[f32]) -> f64 {
    lines.iter().fold(f64::MIN_POSITIVE, |energy, &line| {
        energy + line as f64 * line as f64
    })
}

fn expand_short_window_f32(
    short: &[f32],
    long_len: usize,
) -> Result<Vec<f32>, SpectralInterpolationError> {
    if short.is_empty() || long_len != short.len() * 8 {
        return Err(SpectralInterpolationError::LayoutMismatch);
    }
    Ok((0..long_len).map(|line| short[line >> 3]).collect())
}

fn band_energy_exponent(lines: &[FixpDbl]) -> i32 {
    let width_scale = if lines.is_empty() {
        0
    } else {
        usize::BITS - 1 - lines.len().leading_zeros()
    };
    let mut energy = 1u64;
    for &line in lines {
        let square = (line as i64 * line as i64) as u64;
        energy = energy.saturating_add((square >> 32) >> width_scale);
    }
    let energy = energy.min(u32::MAX as u64) as u32;
    energy.leading_zeros() as i32 - 1
}

fn interpolate_fixed_band(lines: &mut [FixpDbl], previous_energy: i32, next_energy: i32) {
    const FAC_MOD_4_Q31: [i32; 4] = [
        1_073_741_824, // 0.5
        1_276_900_426, // 2^-0.75
        1_518_500_250, // 2^-0.5
        1_805_818_847, // 2^-0.25
    ];
    let delta = previous_energy - next_energy;
    let factor = FAC_MOD_4_Q31[(delta & 3) as usize];
    let shift = ((delta >> 2) + 1).clamp(-31, 31);
    for line in lines {
        let value = mul_q31(*line, factor);
        *line = if shift >= 0 {
            let scaled = (value as i64) << shift;
            scaled.clamp(i32::MIN as i64, i32::MAX as i64) as i32
        } else {
            value >> -shift
        };
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectralInterpolationError {
    LayoutMismatch,
}

impl fmt::Display for SpectralInterpolationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "concealment spectral layouts do not match")
    }
}

impl std::error::Error for SpectralInterpolationError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeats_and_fades_last_good_frame_then_fades_in_recovery() {
        let mut concealment = PcmConcealment::new();
        assert_eq!(
            concealment.process_f32(vec![1.0, 1.0], 0),
            vec![vec![1.0, 1.0]]
        );
        let output = concealment.process_f32(vec![0.5, 0.5], 2);
        assert_eq!(output.len(), 3);
        assert!((output[0][0] - FDK_DEFAULT_FADE_FACTOR).abs() < 1e-6);
        assert!((output[1][0] - 0.5).abs() < 1e-6);
        assert!(output[2][0] < 0.5);
        assert!((output[2][1] - 0.5).abs() < 1e-6);
        assert_eq!(concealment.consecutive_losses(), 0);
    }

    #[test]
    fn emits_silence_without_a_previous_good_frame() {
        let mut concealment = PcmConcealment::new();
        let output = concealment.process_i16(vec![1000, -1000], 1);
        assert_eq!(output[0], [0, 0]);
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn interpolates_equal_band_energy_without_changing_magnitude() {
        let previous = FixedInverseQuantizedSpectrum {
            windows: vec![vec![0x1000_0000; 4]],
            window_exponents: vec![0],
        };
        let output = interpolate_fixed_spectra(&previous, &previous, &[0, 4]).unwrap();
        for (&actual, &expected) in output.windows[0].iter().zip(&previous.windows[0]) {
            assert!((actual as i64 - expected as i64).abs() <= 2);
        }
    }

    #[test]
    fn interpolates_to_geometric_mean_band_energy() {
        let previous = FixedInverseQuantizedSpectrum {
            windows: vec![vec![0x0800_0000; 4]],
            window_exponents: vec![0],
        };
        let next = FixedInverseQuantizedSpectrum {
            windows: vec![vec![0x2000_0000; 4]],
            window_exponents: vec![0],
        };
        let output = interpolate_fixed_spectra(&previous, &next, &[0, 4]).unwrap();
        // next amplitude is 4x previous; geometric-mean amplitude is 2x.
        for &actual in &output.windows[0] {
            assert!((actual as i64 - 0x1000_0000).abs() <= 4);
        }
    }

    #[test]
    fn interpolates_long_short_transitions_with_start_stop_windows() {
        let long = FixedInverseQuantizedSpectrum {
            windows: vec![vec![0x1000_0000; 16]],
            window_exponents: vec![0],
        };
        let short = FixedInverseQuantizedSpectrum {
            windows: vec![vec![0x1000_0000; 2]; 8],
            window_exponents: vec![0; 8],
        };
        let (start, start_sequence) = interpolate_fixed_spectra_mixed(
            &long,
            WindowSequence::OnlyLong,
            &short,
            WindowSequence::EightShort,
            &[0, 8, 16],
            &[0, 1, 2],
        )
        .unwrap();
        assert_eq!(start_sequence, WindowSequence::LongStart);
        assert_eq!(start.windows.len(), 1);
        assert_eq!(start.windows[0].len(), 16);

        let (stop, stop_sequence) = interpolate_fixed_spectra_mixed(
            &short,
            WindowSequence::EightShort,
            &long,
            WindowSequence::OnlyLong,
            &[0, 8, 16],
            &[0, 1, 2],
        )
        .unwrap();
        assert_eq!(stop_sequence, WindowSequence::LongStop);
        assert_eq!(stop.windows.len(), 1);
        assert_eq!(stop.windows[0].len(), 16);
    }

    #[test]
    fn f32_interpolation_matches_geometric_mean_and_mixed_windows() {
        let previous = InverseQuantizedSpectrum {
            windows: vec![vec![0.125; 16]],
        };
        let next = InverseQuantizedSpectrum {
            windows: vec![vec![0.5; 16]],
        };
        let (output, sequence) = interpolate_f32_spectra_mixed(
            &previous,
            WindowSequence::OnlyLong,
            &next,
            WindowSequence::OnlyLong,
            &[0, 8, 16],
            &[0, 1, 2],
        )
        .unwrap();
        assert_eq!(sequence, WindowSequence::OnlyLong);
        assert!(output.windows[0]
            .iter()
            .all(|value| (*value - 0.25).abs() < 1e-6));

        let short = InverseQuantizedSpectrum {
            windows: vec![vec![0.25; 2]; 8],
        };
        let (_, sequence) = interpolate_f32_spectra_mixed(
            &previous,
            WindowSequence::OnlyLong,
            &short,
            WindowSequence::EightShort,
            &[0, 8, 16],
            &[0, 1, 2],
        )
        .unwrap();
        assert_eq!(sequence, WindowSequence::LongStart);
    }

    #[test]
    fn pcm_concealment_reaches_silence_and_handles_empty_recovery() {
        let mut concealment = PcmConcealment::new();
        concealment.process_f32(vec![1.0], 0);
        let output = concealment.process_f32(vec![0.0], FDK_DEFAULT_FADE_OUT_FRAMES + 1);
        assert_eq!(output[FDK_DEFAULT_FADE_OUT_FRAMES], vec![0.0]);

        concealment.process_f32(vec![1.0], 0);
        let empty = concealment.process_f32(Vec::new(), 1);
        assert_eq!(empty.last(), Some(&Vec::new()));

        let converted = concealment.process_i16(vec![i16::MAX, i16::MIN], 0);
        assert_eq!(converted[0], [i16::MAX, i16::MIN]);
    }

    #[test]
    fn validates_fixed_interpolation_layouts_and_same_window_modes() {
        let long = FixedInverseQuantizedSpectrum {
            windows: vec![vec![0x2000_0000; 16]],
            window_exponents: vec![0],
        };
        let low = FixedInverseQuantizedSpectrum {
            windows: vec![vec![0x0100_0000; 16]],
            window_exponents: vec![0],
        };
        let reduced = interpolate_fixed_spectra(&long, &low, &[0, 16]).unwrap();
        assert!(reduced.windows[0][0].abs() < long.windows[0][0].abs());
        assert!(interpolate_fixed_spectra(&long, &low, &[0, 0, 16]).is_ok());
        assert_eq!(
            interpolate_fixed_spectra(&long, &low, &[1, 16]),
            Err(SpectralInterpolationError::LayoutMismatch)
        );
        assert_eq!(
            interpolate_fixed_spectra(&long, &low, &[0, 17]),
            Err(SpectralInterpolationError::LayoutMismatch)
        );
        assert_eq!(
            interpolate_fixed_spectra(&long, &low, &[0, 8, 7]),
            Err(SpectralInterpolationError::LayoutMismatch)
        );

        let (_, sequence) = interpolate_fixed_spectra_mixed(
            &long,
            WindowSequence::OnlyLong,
            &low,
            WindowSequence::LongStop,
            &[0, 16],
            &[0, 2],
        )
        .unwrap();
        assert_eq!(sequence, WindowSequence::OnlyLong);

        let short = FixedInverseQuantizedSpectrum {
            windows: vec![vec![0x1000_0000; 2]; 8],
            window_exponents: vec![0; 8],
        };
        let (_, sequence) = interpolate_fixed_spectra_mixed(
            &short,
            WindowSequence::EightShort,
            &short,
            WindowSequence::EightShort,
            &[0, 16],
            &[0, 2],
        )
        .unwrap();
        assert_eq!(sequence, WindowSequence::EightShort);

        let bad_short = FixedInverseQuantizedSpectrum {
            windows: vec![vec![1; 3]],
            window_exponents: vec![0],
        };
        assert_eq!(
            interpolate_fixed_spectra_mixed(
                &long,
                WindowSequence::OnlyLong,
                &bad_short,
                WindowSequence::EightShort,
                &[0, 16],
                &[0, 3],
            ),
            Err(SpectralInterpolationError::LayoutMismatch)
        );
    }

    #[test]
    fn covers_f32_short_stop_and_layout_errors() {
        let long = InverseQuantizedSpectrum {
            windows: vec![vec![0.5; 16]],
        };
        let short = InverseQuantizedSpectrum {
            windows: vec![vec![0.25; 2]; 8],
        };
        let (_, same_short) = interpolate_f32_spectra_mixed(
            &short,
            WindowSequence::EightShort,
            &short,
            WindowSequence::EightShort,
            &[0, 16],
            &[0, 2],
        )
        .unwrap();
        assert_eq!(same_short, WindowSequence::EightShort);
        let (_, stop) = interpolate_f32_spectra_mixed(
            &short,
            WindowSequence::EightShort,
            &long,
            WindowSequence::OnlyLong,
            &[0, 16],
            &[0, 2],
        )
        .unwrap();
        assert_eq!(stop, WindowSequence::LongStop);

        assert_eq!(
            interpolate_f32_spectra(&long, &long, &[1, 16]),
            Err(SpectralInterpolationError::LayoutMismatch)
        );
        assert_eq!(
            interpolate_f32_spectra(&long, &long, &[0, 17]),
            Err(SpectralInterpolationError::LayoutMismatch)
        );
        let bad_short = InverseQuantizedSpectrum {
            windows: vec![vec![1.0; 3]],
        };
        assert_eq!(
            interpolate_f32_spectra_mixed(
                &bad_short,
                WindowSequence::EightShort,
                &long,
                WindowSequence::OnlyLong,
                &[0, 16],
                &[0, 3],
            ),
            Err(SpectralInterpolationError::LayoutMismatch)
        );
        assert!(!SpectralInterpolationError::LayoutMismatch
            .to_string()
            .is_empty());
    }
}
