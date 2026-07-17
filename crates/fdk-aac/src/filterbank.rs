//! Reference f32 AAC-LC filterbank building blocks.
//!
//! This module starts with a deliberately simple O(N^2) IMDCT and ONLY_LONG
//! sine-window overlap-add path. It is intended as a correctness/reference
//! bridge before replacing it with a fast MDCT and adding short/start/stop/KBD.

use std::fmt;

use crate::fixed::{clamp_i64_to_fixp_dbl_fractional, FixpDbl, MAXVAL_DBL, MINVAL_DBL_PLUS_ONE};
use crate::ics::{IcsInfo, WindowSequence, WindowShape};
use crate::inverse::{FixedInverseQuantizedSpectrum, InverseQuantizedSpectrum};

#[derive(Debug, Clone, PartialEq)]
pub struct LongBlockFilterbank {
    frame_len: usize,
    overlap: Vec<f32>,
    previous_window_shape: WindowShape,
    imdct_plan: ImdctPlan,
    short_imdct_plan: Option<ImdctPlan>,
}

impl LongBlockFilterbank {
    pub fn new(frame_len: usize) -> Result<Self, FilterbankError> {
        if frame_len == 0 || frame_len % 2 != 0 {
            return Err(FilterbankError::InvalidFrameLength(frame_len));
        }
        Ok(Self {
            frame_len,
            overlap: vec![0.0; frame_len],
            previous_window_shape: WindowShape::Sine,
            imdct_plan: ImdctPlan::new(frame_len),
            short_imdct_plan: if frame_len == 1024 {
                Some(ImdctPlan::new(128))
            } else {
                None
            },
        })
    }

    pub fn overlap(&self) -> &[f32] {
        &self.overlap
    }

    pub fn previous_window_shape(&self) -> WindowShape {
        self.previous_window_shape
    }

    pub fn clear_history(&mut self) {
        self.overlap.fill(0.0);
        self.previous_window_shape = WindowShape::Sine;
    }

    pub fn flush(&mut self) -> Vec<f32> {
        self.previous_window_shape = WindowShape::Sine;
        std::mem::replace(&mut self.overlap, vec![0.0; self.frame_len])
    }

    pub fn process_only_long_sine(
        &mut self,
        spectrum: &[f32],
    ) -> Result<Vec<f32>, FilterbankError> {
        if spectrum.len() != self.frame_len {
            return Err(FilterbankError::SpectrumLengthMismatch {
                expected: self.frame_len,
                actual: spectrum.len(),
            });
        }

        let mut time = self.imdct_plan.process(spectrum);
        apply_sine_window(&mut time);

        let mut output = vec![0.0f32; self.frame_len];
        for i in 0..self.frame_len {
            output[i] = time[i] + self.overlap[i];
        }
        self.overlap.copy_from_slice(&time[self.frame_len..]);
        self.previous_window_shape = WindowShape::Sine;

        Ok(output)
    }

    pub fn process_long_window(
        &mut self,
        spectrum: &[f32],
        sequence: WindowSequence,
        current_shape: WindowShape,
    ) -> Result<Vec<f32>, FilterbankError> {
        if spectrum.len() != self.frame_len {
            return Err(FilterbankError::SpectrumLengthMismatch {
                expected: self.frame_len,
                actual: spectrum.len(),
            });
        }
        if !matches!(
            sequence,
            WindowSequence::OnlyLong | WindowSequence::LongStart | WindowSequence::LongStop
        ) {
            return Err(FilterbankError::UnsupportedWindowSequence(sequence));
        }

        let mut time = self.imdct_plan.process(spectrum);
        let window = block_switching_window(
            self.frame_len,
            sequence,
            self.previous_window_shape,
            current_shape,
        )?;
        for (sample, window) in time.iter_mut().zip(window) {
            *sample *= window;
        }

        let mut output = vec![0.0f32; self.frame_len];
        for i in 0..self.frame_len {
            output[i] = time[i] + self.overlap[i];
        }
        self.overlap.copy_from_slice(&time[self.frame_len..]);
        self.previous_window_shape = current_shape;

        Ok(output)
    }

    pub fn process_eight_short_sine(
        &mut self,
        spectra: &[Vec<f32>],
    ) -> Result<Vec<f32>, FilterbankError> {
        if self.frame_len != 1024 {
            return Err(FilterbankError::InvalidFrameLength(self.frame_len));
        }
        if spectra.len() != 8 {
            return Err(FilterbankError::ExpectedEightShortWindows {
                actual: spectra.len(),
            });
        }
        if spectra.iter().any(|spectrum| spectrum.len() != 128) {
            return Err(FilterbankError::SpectrumLengthMismatch {
                expected: 128,
                actual: spectra
                    .iter()
                    .find(|spectrum| spectrum.len() != 128)
                    .map_or(0, Vec::len),
            });
        }

        let short_plan = self
            .short_imdct_plan
            .as_ref()
            .ok_or(FilterbankError::InvalidFrameLength(self.frame_len))?;
        let mut time = vec![0.0f32; 2 * self.frame_len];
        for (short_index, spectrum) in spectra.iter().enumerate() {
            let mut short_time = short_plan.process(spectrum);
            apply_sine_window(&mut short_time);
            let offset = 448 + short_index * 128;
            for (i, &sample) in short_time.iter().enumerate() {
                time[offset + i] += sample;
            }
        }

        let mut output = vec![0.0f32; self.frame_len];
        for i in 0..self.frame_len {
            output[i] = time[i] + self.overlap[i];
        }
        self.overlap.copy_from_slice(&time[self.frame_len..]);
        self.previous_window_shape = WindowShape::Sine;

        Ok(output)
    }

    pub fn process_eight_short_window(
        &mut self,
        spectra: &[Vec<f32>],
        current_shape: WindowShape,
    ) -> Result<Vec<f32>, FilterbankError> {
        if self.frame_len != 1024 {
            return Err(FilterbankError::InvalidFrameLength(self.frame_len));
        }
        if spectra.len() != 8 {
            return Err(FilterbankError::ExpectedEightShortWindows {
                actual: spectra.len(),
            });
        }
        if spectra.iter().any(|spectrum| spectrum.len() != 128) {
            return Err(FilterbankError::SpectrumLengthMismatch {
                expected: 128,
                actual: spectra
                    .iter()
                    .find(|spectrum| spectrum.len() != 128)
                    .map_or(0, Vec::len),
            });
        }

        let short_plan = self
            .short_imdct_plan
            .as_ref()
            .ok_or(FilterbankError::InvalidFrameLength(self.frame_len))?;
        let mut time = vec![0.0f32; 2 * self.frame_len];
        let short_window = window_for_shape(current_shape, 256)?;
        for (short_index, spectrum) in spectra.iter().enumerate() {
            let mut short_time = short_plan.process(spectrum);
            for (sample, window) in short_time.iter_mut().zip(short_window.iter().copied()) {
                *sample *= window;
            }
            let offset = 448 + short_index * 128;
            for (i, &sample) in short_time.iter().enumerate() {
                time[offset + i] += sample;
            }
        }

        let mut output = vec![0.0f32; self.frame_len];
        for i in 0..self.frame_len {
            output[i] = time[i] + self.overlap[i];
        }
        self.overlap.copy_from_slice(&time[self.frame_len..]);
        self.previous_window_shape = current_shape;

        Ok(output)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImdctPlan {
    n: usize,
    kernel: Vec<f32>,
}

impl ImdctPlan {
    pub fn new(n: usize) -> Self {
        let scale = std::f32::consts::PI / n as f32;
        let mut kernel = Vec::with_capacity(2 * n * n);
        for sample in 0..2 * n {
            let phase_base = (sample as f32 + 0.5 + n as f32 / 2.0) * scale;
            for k in 0..n {
                kernel.push((phase_base * (k as f32 + 0.5)).cos());
            }
        }
        Self { n, kernel }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    pub fn process(&self, spectrum: &[f32]) -> Vec<f32> {
        assert_eq!(spectrum.len(), self.n);
        let mut out = vec![0.0f32; 2 * self.n];
        for (sample, slot) in out.iter_mut().enumerate() {
            let row = &self.kernel[sample * self.n..(sample + 1) * self.n];
            *slot = row
                .iter()
                .zip(spectrum)
                .map(|(&kernel, &value)| kernel * value)
                .sum();
        }
        out
    }
}

pub fn imdct_planned_f32(spectrum: &[f32]) -> Vec<f32> {
    ImdctPlan::new(spectrum.len()).process(spectrum)
}

#[derive(Debug, Clone, PartialEq)]
pub struct FixedImdctPlan {
    n: usize,
    kernel_q31: Vec<FixpDbl>,
}

impl FixedImdctPlan {
    pub fn new(n: usize) -> Self {
        let scale = std::f32::consts::PI / n as f32;
        let mut kernel_q31 = Vec::with_capacity(2 * n * n);
        for sample in 0..2 * n {
            let phase_base = (sample as f32 + 0.5 + n as f32 / 2.0) * scale;
            for k in 0..n {
                kernel_q31.push(f32_to_q31((phase_base * (k as f32 + 0.5)).cos()));
            }
        }
        Self { n, kernel_q31 }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Reference Q31 inverse MDCT from N spectral lines to 2N time samples.
    ///
    /// This intentionally mirrors `ImdctPlan`'s direct O(N^2) kernel and is a
    /// correctness bridge toward a libFDK-style fixed-point filterbank.  The
    /// output is saturated Q31; callers that need FDK bit-exact AAC output must
    /// still apply the same per-stage scaling/headroom as libFDK.
    pub fn process_q31(&self, spectrum: &[FixpDbl]) -> Vec<FixpDbl> {
        assert_eq!(spectrum.len(), self.n);
        let mut out = vec![0; 2 * self.n];
        for (sample, slot) in out.iter_mut().enumerate() {
            let row = &self.kernel_q31[sample * self.n..(sample + 1) * self.n];
            let mut acc = 0i64;
            for (&kernel, &value) in row.iter().zip(spectrum) {
                acc += ((kernel as i64) * (value as i64)) >> 31;
            }
            *slot = clamp_i64_to_fixp_dbl_fractional(acc);
        }
        out
    }
}

pub fn imdct_planned_q31(spectrum: &[FixpDbl]) -> Vec<FixpDbl> {
    FixedImdctPlan::new(spectrum.len()).process_q31(spectrum)
}

#[derive(Debug, Clone, PartialEq)]
pub struct FixedLongBlockFilterbank {
    frame_len: usize,
    overlap: Vec<FixpDbl>,
    previous_window_shape: WindowShape,
    imdct_plan: FixedImdctPlan,
    short_imdct_plan: Option<FixedImdctPlan>,
}

impl FixedLongBlockFilterbank {
    pub fn new(frame_len: usize) -> Result<Self, FilterbankError> {
        if frame_len == 0 || frame_len % 2 != 0 {
            return Err(FilterbankError::InvalidFrameLength(frame_len));
        }
        Ok(Self {
            frame_len,
            overlap: vec![0; frame_len],
            previous_window_shape: WindowShape::Sine,
            imdct_plan: FixedImdctPlan::new(frame_len),
            short_imdct_plan: if frame_len == 1024 {
                Some(FixedImdctPlan::new(128))
            } else {
                None
            },
        })
    }

    pub fn overlap(&self) -> &[FixpDbl] {
        &self.overlap
    }

    pub fn previous_window_shape(&self) -> WindowShape {
        self.previous_window_shape
    }

    pub fn clear_history(&mut self) {
        self.overlap.fill(0);
        self.previous_window_shape = WindowShape::Sine;
    }

    pub fn flush(&mut self) -> Vec<FixpDbl> {
        self.previous_window_shape = WindowShape::Sine;
        std::mem::replace(&mut self.overlap, vec![0; self.frame_len])
    }

    pub fn process_only_long_sine_q31(
        &mut self,
        spectrum: &[FixpDbl],
    ) -> Result<Vec<FixpDbl>, FilterbankError> {
        if spectrum.len() != self.frame_len {
            return Err(FilterbankError::SpectrumLengthMismatch {
                expected: self.frame_len,
                actual: spectrum.len(),
            });
        }

        let mut time = self.imdct_plan.process_q31(spectrum);
        apply_sine_window_q31(&mut time);

        let mut output = vec![0; self.frame_len];
        for i in 0..self.frame_len {
            output[i] = clamp_i64_to_fixp_dbl_fractional(time[i] as i64 + self.overlap[i] as i64);
        }
        self.overlap.copy_from_slice(&time[self.frame_len..]);
        self.previous_window_shape = WindowShape::Sine;

        Ok(output)
    }

    pub fn process_long_window_q31(
        &mut self,
        spectrum: &[FixpDbl],
        sequence: WindowSequence,
        current_shape: WindowShape,
    ) -> Result<Vec<FixpDbl>, FilterbankError> {
        if spectrum.len() != self.frame_len {
            return Err(FilterbankError::SpectrumLengthMismatch {
                expected: self.frame_len,
                actual: spectrum.len(),
            });
        }
        if !matches!(
            sequence,
            WindowSequence::OnlyLong | WindowSequence::LongStart | WindowSequence::LongStop
        ) {
            return Err(FilterbankError::UnsupportedWindowSequence(sequence));
        }

        let mut time = self.imdct_plan.process_q31(spectrum);
        let window = block_switching_window_q31(
            self.frame_len,
            sequence,
            self.previous_window_shape,
            current_shape,
        )?;
        for (sample, window) in time.iter_mut().zip(window) {
            *sample = clamp_i64_to_fixp_dbl_fractional(((*sample as i64) * (window as i64)) >> 31);
        }

        let mut output = vec![0; self.frame_len];
        for i in 0..self.frame_len {
            output[i] = clamp_i64_to_fixp_dbl_fractional(time[i] as i64 + self.overlap[i] as i64);
        }
        self.overlap.copy_from_slice(&time[self.frame_len..]);
        self.previous_window_shape = current_shape;

        Ok(output)
    }

    pub fn process_eight_short_window_q31(
        &mut self,
        spectra: &[Vec<FixpDbl>],
        current_shape: WindowShape,
    ) -> Result<Vec<FixpDbl>, FilterbankError> {
        if self.frame_len != 1024 {
            return Err(FilterbankError::InvalidFrameLength(self.frame_len));
        }
        if spectra.len() != 8 {
            return Err(FilterbankError::ExpectedEightShortWindows {
                actual: spectra.len(),
            });
        }
        if spectra.iter().any(|spectrum| spectrum.len() != 128) {
            return Err(FilterbankError::SpectrumLengthMismatch {
                expected: 128,
                actual: spectra
                    .iter()
                    .find(|spectrum| spectrum.len() != 128)
                    .map_or(0, Vec::len),
            });
        }

        let short_plan = self
            .short_imdct_plan
            .as_ref()
            .ok_or(FilterbankError::InvalidFrameLength(self.frame_len))?;
        let short_window = window_for_shape_q31(current_shape, 256)?;
        let mut time = vec![0; 2 * self.frame_len];
        for (short_index, spectrum) in spectra.iter().enumerate() {
            let mut short_time = short_plan.process_q31(spectrum);
            for (sample, window) in short_time.iter_mut().zip(short_window.iter().copied()) {
                *sample =
                    clamp_i64_to_fixp_dbl_fractional(((*sample as i64) * (window as i64)) >> 31);
            }
            let offset = 448 + short_index * 128;
            for (i, &sample) in short_time.iter().enumerate() {
                time[offset + i] =
                    clamp_i64_to_fixp_dbl_fractional(time[offset + i] as i64 + sample as i64);
            }
        }

        let mut output = vec![0; self.frame_len];
        for i in 0..self.frame_len {
            output[i] = clamp_i64_to_fixp_dbl_fractional(time[i] as i64 + self.overlap[i] as i64);
        }
        self.overlap.copy_from_slice(&time[self.frame_len..]);
        self.previous_window_shape = current_shape;

        Ok(output)
    }
}

fn f32_to_q31(value: f32) -> FixpDbl {
    if value >= 1.0 {
        MAXVAL_DBL
    } else if value <= -1.0 {
        MINVAL_DBL_PLUS_ONE
    } else {
        (value * 2_147_483_648.0).round() as FixpDbl
    }
}

pub fn sine_window_q31(len: usize) -> Vec<FixpDbl> {
    sine_window(len).into_iter().map(f32_to_q31).collect()
}

pub fn apply_sine_window_q31(samples: &mut [FixpDbl]) {
    let window = sine_window_q31(samples.len());
    for (sample, window) in samples.iter_mut().zip(window) {
        *sample = clamp_i64_to_fixp_dbl_fractional(((*sample as i64) * (window as i64)) >> 31);
    }
}

pub fn window_for_shape_q31(
    shape: WindowShape,
    len: usize,
) -> Result<Vec<FixpDbl>, FilterbankError> {
    Ok(window_for_shape(shape, len)?
        .into_iter()
        .map(f32_to_q31)
        .collect())
}

pub fn block_switching_window_q31(
    frame_len: usize,
    sequence: WindowSequence,
    previous_shape: WindowShape,
    current_shape: WindowShape,
) -> Result<Vec<FixpDbl>, FilterbankError> {
    Ok(
        block_switching_window(frame_len, sequence, previous_shape, current_shape)?
            .into_iter()
            .map(f32_to_q31)
            .collect(),
    )
}

pub fn synthesize_aac_lc_frame_q31(
    spectra: &[Vec<FixpDbl>],
    ics: &IcsInfo,
    filterbank: &mut FixedLongBlockFilterbank,
) -> Result<Vec<FixpDbl>, FilterbankError> {
    match ics.window_sequence {
        WindowSequence::OnlyLong | WindowSequence::LongStart | WindowSequence::LongStop => {
            if spectra.len() != 1 {
                return Err(FilterbankError::ExpectedOneLongWindow {
                    actual: spectra.len(),
                });
            }
            filterbank.process_long_window_q31(&spectra[0], ics.window_sequence, ics.window_shape)
        }
        WindowSequence::EightShort => {
            filterbank.process_eight_short_window_q31(spectra, ics.window_shape)
        }
    }
}

pub fn inverse_quantized_spectrum_to_q31(spectrum: &InverseQuantizedSpectrum) -> Vec<Vec<FixpDbl>> {
    FixedInverseQuantizedSpectrum::from_f32_bridge(spectrum).windows
}

pub fn synthesize_aac_lc_frame_from_fixed_inverse_q31(
    spectrum: &FixedInverseQuantizedSpectrum,
    ics: &IcsInfo,
    filterbank: &mut FixedLongBlockFilterbank,
) -> Result<Vec<FixpDbl>, FilterbankError> {
    synthesize_aac_lc_frame_q31(&spectrum.windows, ics, filterbank)
}

pub fn synthesize_aac_lc_frame_from_inverse_q31(
    spectrum: &InverseQuantizedSpectrum,
    ics: &IcsInfo,
    filterbank: &mut FixedLongBlockFilterbank,
) -> Result<Vec<FixpDbl>, FilterbankError> {
    let spectrum = FixedInverseQuantizedSpectrum::from_f32_bridge(spectrum);
    synthesize_aac_lc_frame_from_fixed_inverse_q31(&spectrum, ics, filterbank)
}

pub fn synthesize_aac_lc_frame(
    spectrum: &InverseQuantizedSpectrum,
    ics: &IcsInfo,
    filterbank: &mut LongBlockFilterbank,
) -> Result<Vec<f32>, FilterbankError> {
    match ics.window_sequence {
        WindowSequence::OnlyLong | WindowSequence::LongStart | WindowSequence::LongStop => {
            if spectrum.windows.len() != 1 {
                return Err(FilterbankError::ExpectedOneLongWindow {
                    actual: spectrum.windows.len(),
                });
            }
            filterbank.process_long_window(
                &spectrum.windows[0],
                ics.window_sequence,
                ics.window_shape,
            )
        }
        WindowSequence::EightShort => {
            filterbank.process_eight_short_window(&spectrum.windows, ics.window_shape)
        }
    }
}

pub fn synthesize_aac_lc_sine_frame(
    spectrum: &InverseQuantizedSpectrum,
    ics: &IcsInfo,
    filterbank: &mut LongBlockFilterbank,
) -> Result<Vec<f32>, FilterbankError> {
    if ics.window_shape != WindowShape::Sine {
        return Err(FilterbankError::UnsupportedWindowShape(ics.window_shape));
    }

    match ics.window_sequence {
        WindowSequence::OnlyLong => {
            if spectrum.windows.len() != 1 {
                return Err(FilterbankError::ExpectedOneLongWindow {
                    actual: spectrum.windows.len(),
                });
            }
            filterbank.process_long_window(
                &spectrum.windows[0],
                WindowSequence::OnlyLong,
                WindowShape::Sine,
            )
        }
        WindowSequence::EightShort => filterbank.process_eight_short_sine(&spectrum.windows),
        WindowSequence::LongStart | WindowSequence::LongStop => Err(
            FilterbankError::UnsupportedWindowSequence(ics.window_sequence),
        ),
    }
}

pub fn synthesize_aac_lc_long_sine_frame(
    spectrum: &InverseQuantizedSpectrum,
    ics: &IcsInfo,
    filterbank: &mut LongBlockFilterbank,
) -> Result<Vec<f32>, FilterbankError> {
    if ics.window_sequence != WindowSequence::OnlyLong {
        return Err(FilterbankError::UnsupportedWindowSequence(
            ics.window_sequence,
        ));
    }
    synthesize_aac_lc_sine_frame(spectrum, ics, filterbank)
}

pub fn block_switching_window(
    frame_len: usize,
    sequence: WindowSequence,
    previous_shape: WindowShape,
    current_shape: WindowShape,
) -> Result<Vec<f32>, FilterbankError> {
    if frame_len == 0 || frame_len % 2 != 0 {
        return Err(FilterbankError::InvalidFrameLength(frame_len));
    }

    let long_prev = window_for_shape(previous_shape, 2 * frame_len)?;
    let long_curr = window_for_shape(current_shape, 2 * frame_len)?;
    if sequence == WindowSequence::OnlyLong {
        let mut window = vec![0.0f32; 2 * frame_len];
        window[..frame_len].copy_from_slice(&long_prev[..frame_len]);
        window[frame_len..].copy_from_slice(&long_curr[frame_len..]);
        return Ok(window);
    }
    if frame_len != 1024 {
        return Err(FilterbankError::InvalidFrameLength(frame_len));
    }
    let short_curr = window_for_shape(current_shape, 256)?;
    let short_prev = window_for_shape(previous_shape, 256)?;
    let mut window = vec![0.0f32; 2048];

    if sequence == WindowSequence::LongStart {
        window[..1024].copy_from_slice(&long_prev[..1024]);
        window[1024..1472].fill(1.0);
        window[1472..1600].copy_from_slice(&short_curr[128..]);
        // Remaining tail stays zero.
    } else if sequence == WindowSequence::LongStop {
        // Leading head stays zero.
        window[448..576].copy_from_slice(&short_prev[..128]);
        window[576..1024].fill(1.0);
        window[1024..].copy_from_slice(&long_curr[1024..]);
    } else {
        return Err(FilterbankError::UnsupportedWindowSequence(sequence));
    }

    Ok(window)
}

pub fn window_for_shape(shape: WindowShape, len: usize) -> Result<Vec<f32>, FilterbankError> {
    match shape {
        WindowShape::Sine => Ok(sine_window(len)),
        WindowShape::Kbd => kbd_window(len),
        WindowShape::LowOverlap => low_overlap_window(len),
    }
}

/// AAC-LD uses a 25%-overlap window: the outer 3/8 frame on each side is
/// zero, quarter-frame sine slopes surround a 3/4-frame flat section.
pub fn low_overlap_window(len: usize) -> Result<Vec<f32>, FilterbankError> {
    if len == 0 || len % 8 != 0 {
        return Err(FilterbankError::InvalidFrameLength(len));
    }
    let frame = len / 2;
    let slope = frame / 4;
    let zero = (frame - slope) / 2;
    let sine = sine_window(2 * slope);
    let mut window = vec![0.0; len];
    window[zero..zero + slope].copy_from_slice(&sine[..slope]);
    window[zero + slope..len - zero - slope].fill(1.0);
    window[len - zero - slope..len - zero].copy_from_slice(&sine[slope..]);
    Ok(window)
}

/// Naive inverse MDCT from N spectral lines to 2N time samples.
///
/// Formula uses the common MDCT-IV inverse kernel:
/// `x[n] = sum_k X[k] cos(pi/N * (n + 0.5 + N/2) * (k + 0.5))`.
pub fn imdct_naive_f32(spectrum: &[f32]) -> Vec<f32> {
    let n = spectrum.len();
    let scale = std::f32::consts::PI / n as f32;
    let mut out = vec![0.0f32; 2 * n];
    for (sample, slot) in out.iter_mut().enumerate() {
        let phase_base = (sample as f32 + 0.5 + n as f32 / 2.0) * scale;
        *slot = spectrum
            .iter()
            .enumerate()
            .map(|(k, &value)| value * (phase_base * (k as f32 + 0.5)).cos())
            .sum();
    }
    out
}

pub fn sine_window(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (std::f32::consts::PI / len as f32 * (i as f32 + 0.5)).sin())
        .collect()
}

pub fn kbd_window(len: usize) -> Result<Vec<f32>, FilterbankError> {
    if len == 0 || len % 2 != 0 {
        return Err(FilterbankError::InvalidFrameLength(len));
    }
    let alpha = match len {
        2048 => 4.0,
        256 => 6.0,
        _ => return Err(FilterbankError::InvalidFrameLength(len)),
    };
    let half = len / 2;
    let mut kaiser = Vec::with_capacity(half);
    for i in 0..half {
        let ratio = (i as f32 + 0.5) / half as f32;
        let arg = std::f32::consts::PI * alpha * (1.0 - (2.0 * ratio - 1.0).powi(2)).sqrt();
        kaiser.push(bessel_i0(arg));
    }
    let norm: f32 = kaiser.iter().sum();
    let mut cumulative = 0.0f32;
    let mut left = Vec::with_capacity(half);
    for value in kaiser {
        cumulative += value;
        left.push((cumulative / norm).sqrt());
    }

    let mut window = vec![0.0f32; len];
    window[..half].copy_from_slice(&left);
    for i in 0..half {
        window[len - 1 - i] = left[i];
    }
    Ok(window)
}

fn bessel_i0(x: f32) -> f32 {
    let mut sum = 1.0f32;
    let mut term = 1.0f32;
    let y = (x * x) / 4.0;
    for k in 1..=32 {
        term *= y / ((k * k) as f32);
        sum += term;
        if term.abs() < 1.0e-8 * sum.abs() {
            break;
        }
    }
    sum
}

pub fn apply_sine_window(samples: &mut [f32]) {
    let len = samples.len();
    for (i, sample) in samples.iter_mut().enumerate() {
        *sample *= (std::f32::consts::PI / len as f32 * (i as f32 + 0.5)).sin();
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterbankError {
    ExpectedEightShortWindows { actual: usize },
    ExpectedOneLongWindow { actual: usize },
    InvalidFrameLength(usize),
    SpectrumLengthMismatch { expected: usize, actual: usize },
    UnsupportedWindowSequence(WindowSequence),
    UnsupportedWindowShape(WindowShape),
}

impl fmt::Display for FilterbankError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExpectedEightShortWindows { actual } => {
                write!(f, "expected eight short-window spectra, got {actual}")
            }
            Self::ExpectedOneLongWindow { actual } => {
                write!(f, "expected one long-window spectrum, got {actual}")
            }
            Self::InvalidFrameLength(length) => {
                write!(f, "invalid filterbank frame length {length}")
            }
            Self::SpectrumLengthMismatch { expected, actual } => write!(
                f,
                "spectrum length mismatch: expected {expected}, got {actual}"
            ),
            Self::UnsupportedWindowSequence(sequence) => {
                write!(f, "unsupported AAC window sequence {sequence:?}")
            }
            Self::UnsupportedWindowShape(shape) => {
                write!(f, "unsupported AAC window shape {shape:?}")
            }
        }
    }
}

impl std::error::Error for FilterbankError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ics::WindowShape;

    fn only_long_ics(shape: WindowShape) -> IcsInfo {
        ics_with_sequence(WindowSequence::OnlyLong, shape)
    }

    fn ics_with_sequence(window_sequence: WindowSequence, shape: WindowShape) -> IcsInfo {
        IcsInfo {
            window_sequence,
            window_shape: shape,
            max_sfb: 1,
            total_sfb: 1,
            predictor_data_present: false,
            scale_factor_grouping: 0,
            window_group_lengths: if window_sequence == WindowSequence::EightShort {
                vec![1, 1, 1, 1, 1, 1, 1, 1]
            } else {
                vec![1]
            },
            bits_read: 0,
        }
    }

    fn eight_short_ics(shape: WindowShape) -> IcsInfo {
        ics_with_sequence(WindowSequence::EightShort, shape)
    }

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1.0e-5,
            "actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn small_filterbanks_reject_long_start_and_sine_dispatch_rejects_transitions() {
        let mut float = LongBlockFilterbank::new(2).unwrap();
        assert_eq!(
            float.process_long_window(&[0.0; 2], WindowSequence::LongStart, WindowShape::Sine),
            Err(FilterbankError::InvalidFrameLength(2))
        );
        let mut fixed = FixedLongBlockFilterbank::new(2).unwrap();
        assert_eq!(
            fixed.process_long_window_q31(&[0; 2], WindowSequence::LongStop, WindowShape::Sine),
            Err(FilterbankError::InvalidFrameLength(2))
        );

        let spectrum = InverseQuantizedSpectrum {
            windows: vec![vec![0.0; 1024]],
        };
        let mut filterbank = LongBlockFilterbank::new(1024).unwrap();
        assert_eq!(
            synthesize_aac_lc_sine_frame(
                &spectrum,
                &ics_with_sequence(WindowSequence::LongStart, WindowShape::Sine),
                &mut filterbank,
            ),
            Err(FilterbankError::UnsupportedWindowSequence(
                WindowSequence::LongStart
            ))
        );
    }

    #[test]
    fn creates_sine_window() {
        let window = sine_window(4);
        assert_close(window[0], (std::f32::consts::PI * 0.125).sin());
        assert_close(window[3], (std::f32::consts::PI * 0.875).sin());
        assert_close(window[0], window[3]);
    }

    #[test]
    fn creates_kbd_window() {
        let window = kbd_window(256).unwrap();
        assert_eq!(window.len(), 256);
        assert!(window[0] > 0.0);
        assert!(window[127] > window[0]);
        assert_close(window[0], window[255]);
        assert_close(window[127], window[128]);
    }

    #[test]
    fn block_switching_windows_have_expected_regions() {
        let start = block_switching_window(
            1024,
            WindowSequence::LongStart,
            WindowShape::Sine,
            WindowShape::Kbd,
        )
        .unwrap();
        assert_eq!(start.len(), 2048);
        assert!(start[..1024].iter().any(|value| *value > 0.0));
        assert!(start[1024..1472]
            .iter()
            .all(|value| (*value - 1.0).abs() < 1.0e-6));
        assert!(start[1600..].iter().all(|value| value.abs() < 1.0e-6));

        let stop = block_switching_window(
            1024,
            WindowSequence::LongStop,
            WindowShape::Kbd,
            WindowShape::Sine,
        )
        .unwrap();
        assert!(stop[..448].iter().all(|value| value.abs() < 1.0e-6));
        assert!(stop[576..1024]
            .iter()
            .all(|value| (*value - 1.0).abs() < 1.0e-6));
        assert!(stop[1024..].iter().any(|value| *value > 0.0));
    }

    #[test]
    fn imdct_naive_produces_expected_impulse_response_for_small_n() {
        let out = imdct_naive_f32(&[1.0, 0.0]);
        let expected = [
            (std::f32::consts::PI / 2.0 * 1.5 * 0.5).cos(),
            (std::f32::consts::PI / 2.0 * 2.5 * 0.5).cos(),
            (std::f32::consts::PI / 2.0 * 3.5 * 0.5).cos(),
            (std::f32::consts::PI / 2.0 * 4.5 * 0.5).cos(),
        ];
        for (actual, expected) in out.iter().zip(expected) {
            assert_close(*actual, expected);
        }
    }

    #[test]
    fn planned_imdct_matches_naive_reference() {
        let spectrum = [1.0, -0.5, 0.25, 0.125];
        let naive = imdct_naive_f32(&spectrum);
        let planned = imdct_planned_f32(&spectrum);
        assert_eq!(planned.len(), naive.len());
        for (actual, expected) in planned.iter().zip(naive) {
            assert_close(*actual, expected);
        }
    }

    #[test]
    fn fixed_imdct_plan_matches_f32_reference_for_small_impulse() {
        let spectrum_q31 = [0x4000_0000, 0, 0, 0];
        let spectrum_f32 = [0.5, 0.0, 0.0, 0.0];
        let fixed = imdct_planned_q31(&spectrum_q31);
        let float = imdct_planned_f32(&spectrum_f32);

        assert_eq!(fixed.len(), float.len());
        for (actual, expected) in fixed.iter().zip(float) {
            let actual_f32 = *actual as f32 / 2_147_483_648.0;
            assert!(
                (actual_f32 - expected).abs() < 2.0e-6,
                "actual={actual_f32}, expected={expected}"
            );
        }
    }

    #[test]
    fn fixed_imdct_saturates_large_accumulators() {
        let spectrum_q31 = [MAXVAL_DBL; 4];
        let out = imdct_planned_q31(&spectrum_q31);
        assert_eq!(out.len(), 8);
        assert!(out
            .iter()
            .all(|value| { (MINVAL_DBL_PLUS_ONE..=MAXVAL_DBL).contains(value) }));
    }

    #[test]
    fn fixed_sine_window_matches_f32_reference() {
        let fixed = sine_window_q31(4);
        let float = sine_window(4);
        for (actual, expected) in fixed.iter().zip(float) {
            let actual_f32 = *actual as f32 / 2_147_483_648.0;
            assert!(
                (actual_f32 - expected).abs() < 2.0e-6,
                "actual={actual_f32}, expected={expected}"
            );
        }
    }

    #[test]
    fn fixed_long_sine_filterbank_overlap_adds() {
        let mut fb = FixedLongBlockFilterbank::new(2).unwrap();
        let first = fb.process_only_long_sine_q31(&[0x4000_0000, 0]).unwrap();
        let raw = imdct_planned_q31(&[0x4000_0000, 0]);
        let mut windowed = raw.clone();
        apply_sine_window_q31(&mut windowed);

        assert_eq!(first.len(), 2);
        assert_eq!(first[0], windowed[0]);
        assert_eq!(first[1], windowed[1]);
        assert_eq!(fb.overlap().len(), 2);

        let second = fb.process_only_long_sine_q31(&[0, 0]).unwrap();
        assert_eq!(second[0], windowed[2]);
        assert_eq!(second[1], windowed[3]);
    }

    #[test]
    fn fixed_block_switching_windows_match_f32_reference() {
        let fixed = block_switching_window_q31(
            1024,
            WindowSequence::LongStart,
            WindowShape::Sine,
            WindowShape::Kbd,
        )
        .unwrap();
        let float = block_switching_window(
            1024,
            WindowSequence::LongStart,
            WindowShape::Sine,
            WindowShape::Kbd,
        )
        .unwrap();
        assert_eq!(fixed.len(), float.len());
        for (actual, expected) in fixed.iter().zip(float) {
            let actual_f32 = *actual as f32 / 2_147_483_648.0;
            assert!(
                (actual_f32 - expected).abs() < 2.0e-6,
                "actual={actual_f32}, expected={expected}"
            );
        }
    }

    #[test]
    fn aac_ld_low_overlap_window_uses_quarter_frame_slopes() {
        for frame in [480, 512] {
            let window = low_overlap_window(2 * frame).unwrap();
            let slope = frame / 4;
            let zero = (frame - slope) / 2;
            assert!(window[..zero].iter().all(|&value| value == 0.0));
            assert!(window[2 * frame - zero..].iter().all(|&value| value == 0.0));
            assert!(window[zero..zero + slope]
                .windows(2)
                .all(|pair| pair[0] < pair[1]));
            assert!(window[zero + slope..2 * frame - zero - slope]
                .iter()
                .all(|&value| value == 1.0));
            assert!(window[2 * frame - zero - slope..2 * frame - zero]
                .windows(2)
                .all(|pair| pair[0] > pair[1]));
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn fixed_long_imdct_tracks_fdk_adaptive_window_state() {
        let frame = 512usize;
        let shapes = [0u8, 1, 0];
        let mut spectra = vec![0i32; shapes.len() * frame];
        for index in 0..frame {
            spectra[frame + index] = if index % 17 == 0 {
                ((index as i32 % 9) - 4) * (1 << 24)
            } else {
                0
            };
        }
        let mut c = vec![0i32; spectra.len()];
        assert_eq!(
            unsafe {
                fdk_aac_sys::fdk_imlt_long_test(
                    spectra.as_ptr(),
                    shapes.as_ptr(),
                    shapes.len() as i32,
                    frame as i32,
                    c.as_mut_ptr(),
                )
            },
            0
        );
        let mut bank = FixedLongBlockFilterbank::new(frame).unwrap();
        let mut rust = Vec::with_capacity(spectra.len());
        for (spectrum, &shape) in spectra.chunks_exact(frame).zip(&shapes) {
            rust.extend(
                bank.process_long_window_q31(
                    spectrum,
                    WindowSequence::OnlyLong,
                    if shape == 0 {
                        WindowShape::Sine
                    } else {
                        WindowShape::LowOverlap
                    },
                )
                .unwrap(),
            );
        }
        let dot = c
            .iter()
            .zip(&rust)
            .map(|(&left, &right)| f64::from(left) * f64::from(right))
            .sum::<f64>();
        let c_energy = c.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
        let rust_energy = rust.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
        let correlation = dot.abs() / (c_energy * rust_energy).sqrt();
        assert!(
            correlation > 0.99,
            "FDK/Rust IMDCT correlation {correlation}, energies {c_energy}/{rust_energy}"
        );
    }

    #[test]
    fn fixed_long_window_tracks_current_shape() {
        let mut fb = FixedLongBlockFilterbank::new(1024).unwrap();
        let zero = vec![0; 1024];
        let pcm = fb
            .process_long_window_q31(&zero, WindowSequence::LongStart, WindowShape::Kbd)
            .unwrap();
        assert_eq!(pcm.len(), 1024);
        assert!(pcm.iter().all(|sample| *sample == 0));
        assert_eq!(fb.previous_window_shape(), WindowShape::Kbd);

        let pcm = fb
            .process_long_window_q31(&zero, WindowSequence::LongStop, WindowShape::Sine)
            .unwrap();
        assert_eq!(pcm.len(), 1024);
        assert!(pcm.iter().all(|sample| *sample == 0));
        assert_eq!(fb.previous_window_shape(), WindowShape::Sine);
    }

    #[test]
    fn fixed_eight_short_window_places_short_windows() {
        let mut fb = FixedLongBlockFilterbank::new(1024).unwrap();
        let mut spectra = vec![vec![0; 128]; 8];
        spectra[0][0] = 0x4000_0000;
        spectra[7][0] = 0x4000_0000;

        let pcm = fb
            .process_eight_short_window_q31(&spectra, WindowShape::Sine)
            .unwrap();
        assert_eq!(pcm.len(), 1024);
        assert!(pcm[..448].iter().all(|sample| *sample == 0));
        assert!(pcm[448..704].iter().any(|sample| *sample != 0));
        assert!(fb.overlap().iter().any(|sample| *sample != 0));
        assert_eq!(fb.previous_window_shape(), WindowShape::Sine);
    }

    #[test]
    fn fixed_synthesize_dispatches_long_and_short_frames() {
        let mut fb = FixedLongBlockFilterbank::new(1024).unwrap();
        let long = vec![vec![0; 1024]];
        let pcm =
            synthesize_aac_lc_frame_q31(&long, &only_long_ics(WindowShape::Sine), &mut fb).unwrap();
        assert_eq!(pcm.len(), 1024);
        assert!(pcm.iter().all(|sample| *sample == 0));

        let short = vec![vec![0; 128]; 8];
        let pcm = synthesize_aac_lc_frame_q31(&short, &eight_short_ics(WindowShape::Kbd), &mut fb)
            .unwrap();
        assert_eq!(pcm.len(), 1024);
        assert_eq!(fb.previous_window_shape(), WindowShape::Kbd);
    }

    #[test]
    fn converts_inverse_quantized_spectrum_to_q31() {
        let spectrum = InverseQuantizedSpectrum {
            windows: vec![vec![0.5, -0.5, 0.0, 2.0, -2.0]],
        };
        let q31 = inverse_quantized_spectrum_to_q31(&spectrum);
        assert_eq!(q31.len(), 1);
        assert_eq!(q31[0][0], 0x4000_0000);
        assert_eq!(q31[0][1], -0x4000_0000);
        assert_eq!(q31[0][2], 0);
        assert_eq!(q31[0][3], MAXVAL_DBL);
        assert_eq!(q31[0][4], MINVAL_DBL_PLUS_ONE);
    }

    #[test]
    fn fixed_synthesize_accepts_inverse_quantized_spectrum_bridge() {
        let spectrum = InverseQuantizedSpectrum {
            windows: vec![vec![0.0; 1024]],
        };
        let mut fb = FixedLongBlockFilterbank::new(1024).unwrap();
        let pcm = synthesize_aac_lc_frame_from_inverse_q31(
            &spectrum,
            &only_long_ics(WindowShape::Sine),
            &mut fb,
        )
        .unwrap();
        assert_eq!(pcm.len(), 1024);
        assert!(pcm.iter().all(|sample| *sample == 0));
    }

    #[test]
    fn long_sine_filterbank_overlap_adds() {
        let mut fb = LongBlockFilterbank::new(2).unwrap();
        let first = fb.process_only_long_sine(&[1.0, 0.0]).unwrap();
        let raw = imdct_naive_f32(&[1.0, 0.0]);
        let mut windowed = raw.clone();
        apply_sine_window(&mut windowed);

        assert_eq!(first.len(), 2);
        assert_close(first[0], windowed[0]);
        assert_close(first[1], windowed[1]);
        assert_eq!(fb.overlap().len(), 2);

        let second = fb.process_only_long_sine(&[0.0, 0.0]).unwrap();
        assert_close(second[0], windowed[2]);
        assert_close(second[1], windowed[3]);
    }

    #[test]
    fn synthesizes_only_long_sine_frame() {
        let spectrum = InverseQuantizedSpectrum {
            windows: vec![vec![0.0; 1024]],
        };
        let mut fb = LongBlockFilterbank::new(1024).unwrap();
        let pcm = synthesize_aac_lc_long_sine_frame(
            &spectrum,
            &only_long_ics(WindowShape::Sine),
            &mut fb,
        )
        .unwrap();
        assert_eq!(pcm.len(), 1024);
        assert!(pcm.iter().all(|sample| *sample == 0.0));
    }

    #[test]
    fn eight_short_sine_filterbank_places_short_windows() {
        let mut fb = LongBlockFilterbank::new(1024).unwrap();
        let mut spectra = vec![vec![0.0; 128]; 8];
        spectra[0][0] = 1.0;
        spectra[7][0] = 1.0;

        let pcm = fb.process_eight_short_sine(&spectra).unwrap();
        assert_eq!(pcm.len(), 1024);
        assert!(pcm[..448].iter().all(|sample| sample.abs() < 1.0e-6));
        assert!(pcm[448..704].iter().any(|sample| sample.abs() > 1.0e-6));

        // The last short window extends past the current 1024-sample output and
        // therefore leaves data in the overlap buffer for the next frame.
        assert!(fb.overlap().iter().any(|sample| sample.abs() > 1.0e-6));
    }

    #[test]
    fn synthesize_dispatches_eight_short_sine_frame() {
        let spectrum = InverseQuantizedSpectrum {
            windows: vec![vec![0.0; 128]; 8],
        };
        let mut fb = LongBlockFilterbank::new(1024).unwrap();
        let pcm =
            synthesize_aac_lc_sine_frame(&spectrum, &eight_short_ics(WindowShape::Sine), &mut fb)
                .unwrap();
        assert_eq!(pcm.len(), 1024);
        assert!(pcm.iter().all(|sample| *sample == 0.0));
    }

    #[test]
    fn synthesizes_start_stop_and_kbd_windows() {
        let spectrum = InverseQuantizedSpectrum {
            windows: vec![vec![0.0; 1024]],
        };
        let mut fb = LongBlockFilterbank::new(1024).unwrap();

        let pcm = synthesize_aac_lc_frame(
            &spectrum,
            &ics_with_sequence(WindowSequence::LongStart, WindowShape::Kbd),
            &mut fb,
        )
        .unwrap();
        assert_eq!(pcm.len(), 1024);
        assert_eq!(fb.previous_window_shape(), WindowShape::Kbd);

        let pcm = synthesize_aac_lc_frame(
            &spectrum,
            &ics_with_sequence(WindowSequence::LongStop, WindowShape::Sine),
            &mut fb,
        )
        .unwrap();
        assert_eq!(pcm.len(), 1024);
        assert_eq!(fb.previous_window_shape(), WindowShape::Sine);
    }

    #[test]
    fn synthesizes_eight_short_kbd_frame() {
        let spectrum = InverseQuantizedSpectrum {
            windows: vec![vec![0.0; 128]; 8],
        };
        let mut fb = LongBlockFilterbank::new(1024).unwrap();
        let pcm = synthesize_aac_lc_frame(&spectrum, &eight_short_ics(WindowShape::Kbd), &mut fb)
            .unwrap();
        assert_eq!(pcm.len(), 1024);
        assert_eq!(fb.previous_window_shape(), WindowShape::Kbd);
    }

    #[test]
    fn rejects_wrong_number_of_short_windows() {
        let mut fb = LongBlockFilterbank::new(1024).unwrap();
        let err = fb
            .process_eight_short_sine(&vec![vec![0.0; 128]; 7])
            .unwrap_err();
        assert_eq!(
            err,
            FilterbankError::ExpectedEightShortWindows { actual: 7 }
        );
    }

    #[test]
    fn rejects_unsupported_shape_or_sequence() {
        let spectrum = InverseQuantizedSpectrum {
            windows: vec![vec![0.0; 1024]],
        };
        let mut fb = LongBlockFilterbank::new(1024).unwrap();
        assert_eq!(
            synthesize_aac_lc_sine_frame(&spectrum, &only_long_ics(WindowShape::Kbd), &mut fb)
                .unwrap_err(),
            FilterbankError::UnsupportedWindowShape(WindowShape::Kbd)
        );

        assert_eq!(
            synthesize_aac_lc_long_sine_frame(
                &spectrum,
                &ics_with_sequence(WindowSequence::LongStart, WindowShape::Sine),
                &mut fb,
            )
            .unwrap_err(),
            FilterbankError::UnsupportedWindowSequence(WindowSequence::LongStart)
        );
    }

    #[test]
    fn plans_expose_empty_and_nonempty_lengths() {
        let empty = ImdctPlan::new(0);
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
        assert!(empty.process(&[]).is_empty());
        let plan = ImdctPlan::new(2);
        assert_eq!(plan.len(), 2);
        assert!(!plan.is_empty());

        let empty = FixedImdctPlan::new(0);
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
        assert!(empty.process_q31(&[]).is_empty());
        let plan = FixedImdctPlan::new(2);
        assert_eq!(plan.len(), 2);
        assert!(!plan.is_empty());
    }

    #[test]
    fn constructors_and_long_paths_reject_invalid_lengths_and_sequences() {
        for length in [0, 3] {
            assert_eq!(
                LongBlockFilterbank::new(length),
                Err(FilterbankError::InvalidFrameLength(length))
            );
            assert_eq!(
                FixedLongBlockFilterbank::new(length),
                Err(FilterbankError::InvalidFrameLength(length))
            );
        }
        let mut float = LongBlockFilterbank::new(2).unwrap();
        assert!(matches!(
            float.process_only_long_sine(&[0.0]),
            Err(FilterbankError::SpectrumLengthMismatch { .. })
        ));
        assert!(matches!(
            float.process_long_window(&[0.0], WindowSequence::OnlyLong, WindowShape::Sine),
            Err(FilterbankError::SpectrumLengthMismatch { .. })
        ));
        assert_eq!(
            float.process_long_window(&[0.0; 2], WindowSequence::EightShort, WindowShape::Sine),
            Err(FilterbankError::UnsupportedWindowSequence(
                WindowSequence::EightShort
            ))
        );
        let mut fixed = FixedLongBlockFilterbank::new(2).unwrap();
        assert!(matches!(
            fixed.process_only_long_sine_q31(&[0]),
            Err(FilterbankError::SpectrumLengthMismatch { .. })
        ));
        assert!(matches!(
            fixed.process_long_window_q31(&[0], WindowSequence::OnlyLong, WindowShape::Sine),
            Err(FilterbankError::SpectrumLengthMismatch { .. })
        ));
        assert_eq!(
            fixed.process_long_window_q31(&[0; 2], WindowSequence::EightShort, WindowShape::Sine),
            Err(FilterbankError::UnsupportedWindowSequence(
                WindowSequence::EightShort
            ))
        );
    }

    #[test]
    fn all_short_window_apis_validate_frame_count_and_spectrum_width() {
        let mut small = LongBlockFilterbank::new(2).unwrap();
        assert_eq!(
            small.process_eight_short_sine(&[]),
            Err(FilterbankError::InvalidFrameLength(2))
        );
        assert_eq!(
            small.process_eight_short_window(&[], WindowShape::Sine),
            Err(FilterbankError::InvalidFrameLength(2))
        );
        let mut float = LongBlockFilterbank::new(1024).unwrap();
        assert_eq!(
            float.process_eight_short_window(&vec![vec![0.0; 128]; 7], WindowShape::Sine),
            Err(FilterbankError::ExpectedEightShortWindows { actual: 7 })
        );
        let mut wrong = vec![vec![0.0; 128]; 8];
        wrong[3].truncate(127);
        assert!(matches!(
            float.process_eight_short_sine(&wrong),
            Err(FilterbankError::SpectrumLengthMismatch { actual: 127, .. })
        ));
        assert!(matches!(
            float.process_eight_short_window(&wrong, WindowShape::Kbd),
            Err(FilterbankError::SpectrumLengthMismatch { actual: 127, .. })
        ));

        let mut small = FixedLongBlockFilterbank::new(2).unwrap();
        assert_eq!(
            small.process_eight_short_window_q31(&[], WindowShape::Sine),
            Err(FilterbankError::InvalidFrameLength(2))
        );
        let mut fixed = FixedLongBlockFilterbank::new(1024).unwrap();
        assert_eq!(
            fixed.process_eight_short_window_q31(&vec![vec![0; 128]; 7], WindowShape::Sine),
            Err(FilterbankError::ExpectedEightShortWindows { actual: 7 })
        );
        let mut wrong = vec![vec![0; 128]; 8];
        wrong[0].clear();
        assert!(matches!(
            fixed.process_eight_short_window_q31(&wrong, WindowShape::Sine),
            Err(FilterbankError::SpectrumLengthMismatch { actual: 0, .. })
        ));
    }

    #[test]
    fn synthesis_dispatch_rejects_wrong_long_window_counts() {
        let ics = only_long_ics(WindowShape::Sine);
        let mut float = LongBlockFilterbank::new(1024).unwrap();
        let spectrum = InverseQuantizedSpectrum {
            windows: vec![vec![0.0; 1024]; 2],
        };
        assert_eq!(
            synthesize_aac_lc_frame(&spectrum, &ics, &mut float),
            Err(FilterbankError::ExpectedOneLongWindow { actual: 2 })
        );
        assert_eq!(
            synthesize_aac_lc_sine_frame(&spectrum, &ics, &mut float),
            Err(FilterbankError::ExpectedOneLongWindow { actual: 2 })
        );
        let mut fixed = FixedLongBlockFilterbank::new(1024).unwrap();
        assert_eq!(
            synthesize_aac_lc_frame_q31(&vec![vec![0; 1024]; 2], &ics, &mut fixed),
            Err(FilterbankError::ExpectedOneLongWindow { actual: 2 })
        );
        let fixed_spectrum = FixedInverseQuantizedSpectrum {
            windows: vec![vec![0; 1024]],
            window_exponents: vec![0],
        };
        assert_eq!(
            synthesize_aac_lc_frame_from_fixed_inverse_q31(&fixed_spectrum, &ics, &mut fixed)
                .unwrap()
                .len(),
            1024
        );
    }

    #[test]
    fn window_builders_reject_invalid_lengths_and_sequences() {
        for length in [0, 3] {
            assert_eq!(
                block_switching_window(
                    length,
                    WindowSequence::OnlyLong,
                    WindowShape::Sine,
                    WindowShape::Sine
                ),
                Err(FilterbankError::InvalidFrameLength(length))
            );
        }
        assert_eq!(
            block_switching_window(
                2,
                WindowSequence::LongStart,
                WindowShape::Sine,
                WindowShape::Sine
            ),
            Err(FilterbankError::InvalidFrameLength(2))
        );
        assert_eq!(
            block_switching_window(
                1024,
                WindowSequence::EightShort,
                WindowShape::Sine,
                WindowShape::Sine
            ),
            Err(FilterbankError::UnsupportedWindowSequence(
                WindowSequence::EightShort
            ))
        );
        for length in [0, 3, 128] {
            assert_eq!(
                kbd_window(length),
                Err(FilterbankError::InvalidFrameLength(length))
            );
        }
        assert_eq!(window_for_shape(WindowShape::Sine, 0), Ok(Vec::new()));
    }

    #[test]
    fn formats_every_filterbank_error() {
        for error in [
            FilterbankError::ExpectedEightShortWindows { actual: 7 },
            FilterbankError::ExpectedOneLongWindow { actual: 2 },
            FilterbankError::InvalidFrameLength(3),
            FilterbankError::SpectrumLengthMismatch {
                expected: 4,
                actual: 3,
            },
            FilterbankError::UnsupportedWindowSequence(WindowSequence::EightShort),
            FilterbankError::UnsupportedWindowShape(WindowShape::Kbd),
        ] {
            assert!(!error.to_string().is_empty());
        }
    }

    #[test]
    fn flush_drains_synthesis_overlap_exactly_once() {
        let mut float = LongBlockFilterbank::new(32).unwrap();
        let mut spectrum = vec![0.0; 32];
        spectrum[3] = 1.0;
        float.process_only_long_sine(&spectrum).unwrap();
        let delayed = float.flush();
        assert!(delayed.iter().any(|sample| *sample != 0.0));
        assert_eq!(float.flush(), vec![0.0; 32]);

        let mut fixed = FixedLongBlockFilterbank::new(32).unwrap();
        let mut spectrum = vec![0; 32];
        spectrum[3] = 1 << 24;
        fixed.process_only_long_sine_q31(&spectrum).unwrap();
        let delayed = fixed.flush();
        assert!(delayed.iter().any(|sample| *sample != 0));
        assert_eq!(fixed.flush(), vec![0; 32]);
    }
}
