//! AAC-ELD low-overlap analysis filterbank.

use std::fmt;
use std::sync::OnceLock;

const AAC_ENCODER_ROM: &str = include_str!(concat!(
    env!("FDK_AAC_UPSTREAM_DIR"),
    "/libAACenc/src/aacEnc_rom.cpp"
));

static COEFFICIENTS_512: OnceLock<Vec<f32>> = OnceLock::new();
static COEFFICIENTS_480: OnceLock<Vec<f32>> = OnceLock::new();

fn analysis_coefficients(frame_length: usize) -> Result<&'static [f32], EldAnalysisError> {
    let (name, expected, storage) = match frame_length {
        512 => ("ELDAnalysis512", 1536, &COEFFICIENTS_512),
        480 => ("ELDAnalysis480", 1440, &COEFFICIENTS_480),
        other => return Err(EldAnalysisError::UnsupportedFrameLength(other)),
    };
    let coefficients = storage.get_or_init(|| parse_wtc_array(name).unwrap_or_default());
    if coefficients.len() != expected {
        return Err(EldAnalysisError::InvalidCoefficientCount {
            frame_length,
            expected,
            actual: coefficients.len(),
        });
    }
    Ok(coefficients)
}

fn parse_wtc_array(name: &str) -> Option<Vec<f32>> {
    let start = AAC_ENCODER_ROM.find(name)?;
    let body_start = AAC_ENCODER_ROM[start..].find('{')? + start + 1;
    let body_end = AAC_ENCODER_ROM[body_start..].find("};")? + body_start;
    let mut body = &AAC_ENCODER_ROM[body_start..body_end];
    let mut result = Vec::new();
    while let Some(start) = body.find("WTC") {
        body = &body[start + 3..];
        let digits = body.find("0x")? + 2;
        if body.len() < digits + 8 {
            return None;
        }
        let bits = u32::from_str_radix(&body[digits..digits + 8], 16).ok()?;
        result.push(bits as i32 as f32 / 2_147_483_648.0);
        body = &body[digits + 8..];
    }
    Some(result)
}

#[derive(Debug, Clone)]
pub struct EldAnalysisFilterbank {
    frame_length: usize,
    previous: Vec<f32>,
    overlap: Vec<f32>,
    dct_iv_kernel: Vec<f32>,
}

impl EldAnalysisFilterbank {
    pub fn new(frame_length: usize) -> Result<Self, EldAnalysisError> {
        analysis_coefficients(frame_length)?;
        let phase = std::f32::consts::PI / frame_length as f32;
        let normalization = (2.0 / frame_length as f32).sqrt();
        let mut dct_iv_kernel = Vec::with_capacity(frame_length * frame_length);
        for output in 0..frame_length {
            for input in 0..frame_length {
                dct_iv_kernel.push(
                    (phase * (input as f32 + 0.5) * (output as f32 + 0.5)).cos() * normalization,
                );
            }
        }
        Ok(Self {
            frame_length,
            previous: vec![0.0; frame_length],
            overlap: vec![0.0; 2 * frame_length],
            dct_iv_kernel,
        })
    }

    pub fn frame_length(&self) -> usize {
        self.frame_length
    }

    pub fn reset(&mut self) {
        self.previous.fill(0.0);
        self.overlap.fill(0.0);
    }

    pub fn analyze(&mut self, input: &[f32]) -> Result<Vec<f32>, EldAnalysisError> {
        if input.len() != self.frame_length {
            return Err(EldAnalysisError::InputLengthMismatch {
                expected: self.frame_length,
                actual: input.len(),
            });
        }
        if input.iter().any(|sample| !sample.is_finite()) {
            return Err(EldAnalysisError::NonFiniteInput);
        }
        let n = self.frame_length;
        let mut time = Vec::with_capacity(2 * n);
        time.extend_from_slice(&self.previous);
        time.extend_from_slice(input);
        let window = analysis_coefficients(n)?;
        let mut folded = vec![0.0f32; n];
        for i in 0..n / 4 {
            let upper_left = time[n + 3 * n / 4 - 1 - i];
            let upper_right = time[n + 3 * n / 4 + i];
            let z0 = upper_left * window[n / 2 - 1 - i] + upper_right * window[n / 2 + i];
            let out = 0.5
                * (upper_left * window[n + n / 2 - 1 - i] + upper_right * window[n + n / 2 + i])
                + 0.25 * self.overlap[n / 2 + i] * window[2 * n + i];
            self.overlap[n / 2 + i] = self.overlap[i];
            self.overlap[i] = z0;
            folded[i] = self.overlap[n / 2 + i]
                + 0.25 * self.overlap[n + n / 2 - 1 - i] * window[2 * n + n / 2 + i];
            folded[n - 1 - i] = out;
            self.overlap[n + n / 2 - 1 - i] = out;
        }
        for i in n / 4..n / 2 {
            let upper_left = time[n + 3 * n / 4 - 1 - i];
            let z0 = upper_left * window[n / 2 - 1 - i];
            let out = 0.5 * upper_left * window[n + n / 2 - 1 - i]
                + 0.25 * self.overlap[n / 2 + i] * window[2 * n + i];
            self.overlap[n / 2 + i] = self.overlap[i] + time[n - n / 4 + i] * window[n / 2 + i];
            self.overlap[i] = z0;
            folded[i] = self.overlap[n / 2 + i]
                + 0.25 * self.overlap[n + n / 2 - 1 - i] * window[2 * n + n / 2 + i];
            folded[n - 1 - i] = out;
            self.overlap[n + n / 2 - 1 - i] = out;
        }
        let mut spectrum = vec![0.0f32; n];
        for (output, value) in spectrum.iter_mut().enumerate() {
            let row = &self.dct_iv_kernel[output * n..(output + 1) * n];
            *value = folded
                .iter()
                .zip(row)
                .map(|(&sample, &coefficient)| sample * coefficient)
                .sum();
        }
        self.previous.copy_from_slice(input);
        Ok(spectrum)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EldAnalysisError {
    UnsupportedFrameLength(usize),
    InvalidCoefficientCount {
        frame_length: usize,
        expected: usize,
        actual: usize,
    },
    InputLengthMismatch {
        expected: usize,
        actual: usize,
    },
    NonFiniteInput,
}

impl fmt::Display for EldAnalysisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedFrameLength(length) => {
                write!(f, "unsupported AAC-ELD analysis length {length}")
            }
            Self::InvalidCoefficientCount {
                frame_length,
                expected,
                actual,
            } => write!(
                f,
                "AAC-ELD {frame_length} analysis window has {actual} coefficients, expected {expected}"
            ),
            Self::InputLengthMismatch { expected, actual } => {
                write!(f, "AAC-ELD analysis expected {expected} samples, got {actual}")
            }
            Self::NonFiniteInput => write!(f, "AAC-ELD analysis input contains NaN or infinity"),
        }
    }
}

impl std::error::Error for EldAnalysisError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_eld_analysis_rom_and_processes_both_frame_lengths() {
        for length in [480, 512] {
            assert_eq!(analysis_coefficients(length).unwrap().len(), 3 * length);
            let mut filterbank = EldAnalysisFilterbank::new(length).unwrap();
            assert!(filterbank
                .analyze(&vec![0.0; length])
                .unwrap()
                .iter()
                .all(|sample| *sample == 0.0));
            let mut impulse = vec![0.0; length];
            impulse[length / 2] = 1.0;
            assert!(filterbank
                .analyze(&impulse)
                .unwrap()
                .iter()
                .any(|sample| *sample != 0.0));
        }
    }
}
