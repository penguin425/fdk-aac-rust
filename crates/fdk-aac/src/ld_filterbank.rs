//! AAC-LD low-delay synthesis filterbank support.

use std::fmt;
use std::sync::OnceLock;

const AAC_ROM_SOURCE: &str = include_str!(concat!(
    env!("FDK_AAC_UPSTREAM_DIR"),
    "/libAACdec/src/aac_rom.cpp"
));

static COEFFICIENTS_512: OnceLock<Vec<f32>> = OnceLock::new();
static COEFFICIENTS_480: OnceLock<Vec<f32>> = OnceLock::new();
static COEFFICIENTS_Q31_512: OnceLock<Vec<i32>> = OnceLock::new();
static COEFFICIENTS_Q31_480: OnceLock<Vec<i32>> = OnceLock::new();

pub fn synthesis_coefficients(frame_length: usize) -> Result<&'static [f32], LdFilterbankError> {
    let (name, expected, storage) = match frame_length {
        512 => ("LowDelaySynthesis512", 1536, &COEFFICIENTS_512),
        480 => ("LowDelaySynthesis480", 1440, &COEFFICIENTS_480),
        other => return Err(LdFilterbankError::UnsupportedFrameLength(other)),
    };
    let coefficients = storage.get_or_init(|| parse_wtc_array(name).unwrap_or_default());
    validate_coefficient_count(frame_length, expected, coefficients)
}

pub fn synthesis_coefficients_q31(
    frame_length: usize,
) -> Result<&'static [i32], LdFilterbankError> {
    let (name, expected, storage) = match frame_length {
        512 => ("LowDelaySynthesis512", 1536, &COEFFICIENTS_Q31_512),
        480 => ("LowDelaySynthesis480", 1440, &COEFFICIENTS_Q31_480),
        other => return Err(LdFilterbankError::UnsupportedFrameLength(other)),
    };
    let coefficients = storage.get_or_init(|| parse_wtc_array_q31(name).unwrap_or_default());
    validate_coefficient_count(frame_length, expected, coefficients)
}

fn validate_coefficient_count<T>(
    frame_length: usize,
    expected: usize,
    coefficients: &[T],
) -> Result<&[T], LdFilterbankError> {
    if coefficients.len() != expected {
        return Err(LdFilterbankError::InvalidCoefficientCount {
            frame_length,
            expected,
            actual: coefficients.len(),
        });
    }
    Ok(coefficients)
}

#[derive(Debug, Clone)]
pub struct LowDelayFilterbankF32 {
    frame_length: usize,
    state: Vec<f32>,
    dct_iv_kernel: Vec<f32>,
}

impl LowDelayFilterbankF32 {
    pub fn new(frame_length: usize) -> Result<Self, LdFilterbankError> {
        synthesis_coefficients(frame_length)?;
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
            state: vec![0.0; 2 * frame_length],
            dct_iv_kernel,
        })
    }

    pub fn state(&self) -> &[f32] {
        &self.state
    }

    pub fn clear_history(&mut self) {
        self.state.fill(0.0);
    }

    pub fn flush(&mut self) -> Result<Vec<f32>, LdFilterbankError> {
        self.process(&vec![0.0; self.frame_length])
    }

    pub fn process(&mut self, spectrum: &[f32]) -> Result<Vec<f32>, LdFilterbankError> {
        if spectrum.len() != self.frame_length {
            return Err(LdFilterbankError::SpectrumLengthMismatch {
                expected: self.frame_length,
                actual: spectrum.len(),
            });
        }
        let n = self.frame_length;
        let mut transformed = vec![0.0f32; n];
        for (output, value) in transformed.iter_mut().enumerate() {
            let row = &self.dct_iv_kernel[output * n..(output + 1) * n];
            *value = spectrum
                .iter()
                .zip(row)
                .map(|(&sample, &coefficient)| sample * coefficient)
                .sum();
        }
        let stored = synthesis_coefficients(n)?;
        let coefficient = |index: usize| {
            let exponent = if index < n {
                1
            } else if index < 2 * n {
                0
            } else {
                -2
            };
            stored[index] * 2.0f32.powi(exponent)
        };
        let mut output = vec![0.0f32; n];
        for i in 0..n / 4 {
            let z2 = transformed[n / 2 + i];
            let z0 = z2 + self.state[n / 2 + i] * coefficient(2 * n + i);
            self.state[n / 2 + i] =
                transformed[n / 2 - 1 - i] + self.state[n + i] * coefficient(2 * n + n / 2 + i);
            output[n * 3 / 4 - 1 - i] = self.state[n / 2 + i] * coefficient(n + n / 2 - 1 - i)
                + self.state[i] * coefficient(n + n / 2 + i);
            self.state[i] = z0;
            self.state[n + i] = z2;
        }
        for i in n / 4..n / 2 {
            let z2 = transformed[n / 2 + i];
            let z0 = z2 + self.state[n / 2 + i] * coefficient(2 * n + i);
            self.state[n / 2 + i] =
                transformed[n / 2 - 1 - i] + self.state[n + i] * coefficient(2 * n + n / 2 + i);
            output[i - n / 4] = self.state[n / 2 + i] * coefficient(n / 2 - 1 - i)
                + self.state[i] * coefficient(n / 2 + i);
            output[n * 3 / 4 - 1 - i] = self.state[n / 2 + i] * coefficient(n + n / 2 - 1 - i)
                + self.state[i] * coefficient(n + n / 2 + i);
            self.state[i] = z0;
            self.state[n + i] = z2;
        }
        for i in 0..n / 4 {
            output[n * 3 / 4 + i] = self.state[i] * coefficient(n / 2 + i);
        }
        Ok(output)
    }
}

#[derive(Debug, Clone)]
pub struct LowDelayFilterbankQ31 {
    frame_length: usize,
    state: Vec<i32>,
    dct_iv_kernel: Vec<i32>,
}

impl LowDelayFilterbankQ31 {
    pub fn new(frame_length: usize) -> Result<Self, LdFilterbankError> {
        synthesis_coefficients_q31(frame_length)?;
        let phase = std::f64::consts::PI / frame_length as f64;
        let normalization = (2.0 / frame_length as f64).sqrt();
        let mut dct_iv_kernel = Vec::with_capacity(frame_length * frame_length);
        for output in 0..frame_length {
            for input in 0..frame_length {
                dct_iv_kernel.push(f64_to_q31(
                    (phase * (input as f64 + 0.5) * (output as f64 + 0.5)).cos() * normalization,
                ));
            }
        }
        Ok(Self {
            frame_length,
            state: vec![0; 2 * frame_length],
            dct_iv_kernel,
        })
    }

    pub fn state(&self) -> &[i32] {
        &self.state
    }

    pub fn clear_history(&mut self) {
        self.state.fill(0);
    }

    pub fn flush(&mut self) -> Result<Vec<i32>, LdFilterbankError> {
        self.process(&vec![0; self.frame_length])
    }

    pub fn process(&mut self, spectrum: &[i32]) -> Result<Vec<i32>, LdFilterbankError> {
        self.process_with_exponent(spectrum, 23)
    }

    pub fn process_with_exponent(
        &mut self,
        spectrum: &[i32],
        spectrum_exponent: i16,
    ) -> Result<Vec<i32>, LdFilterbankError> {
        if spectrum.len() != self.frame_length {
            return Err(LdFilterbankError::SpectrumLengthMismatch {
                expected: self.frame_length,
                actual: spectrum.len(),
            });
        }
        let n = self.frame_length;
        let mut transformed = vec![0i32; n];
        for (output, value) in transformed.iter_mut().enumerate() {
            let row = &self.dct_iv_kernel[output * n..(output + 1) * n];
            let sum = spectrum
                .iter()
                .zip(row)
                .map(|(&sample, &coefficient)| sample as i128 * coefficient as i128)
                .sum::<i128>();
            *value = clamp_i128_to_i32(round_shift_i128(sum, 31));
        }
        let exponent_shift = spectrum_exponent as i32 - 23;
        for value in &mut transformed {
            *value = if exponent_shift >= 0 {
                clamp_i128_to_i32((*value as i128) << exponent_shift.min(63))
            } else {
                clamp_i128_to_i32(round_shift_i128(
                    *value as i128,
                    (-exponent_shift).min(63) as u32,
                ))
            };
        }
        let coefficients = synthesis_coefficients_q31(n)?;
        let multiply = |value: i32, index: usize| {
            let exponent = if index < n {
                1
            } else if index < 2 * n {
                0
            } else {
                -2
            };
            let product = round_shift_i128(value as i128 * coefficients[index] as i128, 31);
            let scaled = if exponent >= 0 {
                product << exponent
            } else {
                round_shift_i128(product, (-exponent) as u32)
            };
            clamp_i128_to_i32(scaled)
        };
        let add = |left: i32, right: i32| clamp_i128_to_i32(left as i128 + right as i128);
        let mut output = vec![0i32; n];
        for i in 0..n / 4 {
            let z2 = transformed[n / 2 + i];
            let z0 = add(z2, multiply(self.state[n / 2 + i], 2 * n + i));
            self.state[n / 2 + i] = add(
                transformed[n / 2 - 1 - i],
                multiply(self.state[n + i], 2 * n + n / 2 + i),
            );
            output[n * 3 / 4 - 1 - i] = add(
                multiply(self.state[n / 2 + i], n + n / 2 - 1 - i),
                multiply(self.state[i], n + n / 2 + i),
            );
            self.state[i] = z0;
            self.state[n + i] = z2;
        }
        for i in n / 4..n / 2 {
            let z2 = transformed[n / 2 + i];
            let z0 = add(z2, multiply(self.state[n / 2 + i], 2 * n + i));
            self.state[n / 2 + i] = add(
                transformed[n / 2 - 1 - i],
                multiply(self.state[n + i], 2 * n + n / 2 + i),
            );
            output[i - n / 4] = add(
                multiply(self.state[n / 2 + i], n / 2 - 1 - i),
                multiply(self.state[i], n / 2 + i),
            );
            output[n * 3 / 4 - 1 - i] = add(
                multiply(self.state[n / 2 + i], n + n / 2 - 1 - i),
                multiply(self.state[i], n + n / 2 + i),
            );
            self.state[i] = z0;
            self.state[n + i] = z2;
        }
        for i in 0..n / 4 {
            output[n * 3 / 4 + i] = multiply(self.state[i], n / 2 + i);
        }
        Ok(output)
    }
}

fn f64_to_q31(value: f64) -> i32 {
    (value * 2_147_483_648.0)
        .round()
        .clamp(i32::MIN as f64, i32::MAX as f64) as i32
}

fn round_shift_i128(value: i128, bits: u32) -> i128 {
    if bits == 0 {
        value
    } else if value >= 0 {
        (value + (1i128 << (bits - 1))) >> bits
    } else {
        -((-value + (1i128 << (bits - 1))) >> bits)
    }
}

fn clamp_i128_to_i32(value: i128) -> i32 {
    value.clamp(i32::MIN as i128, i32::MAX as i128) as i32
}

fn parse_wtc_array(name: &str) -> Option<Vec<f32>> {
    Some(
        parse_wtc_array_q31(name)?
            .into_iter()
            .map(|bits| bits as f32 / 2_147_483_648.0)
            .collect(),
    )
}

fn parse_wtc_array_q31(name: &str) -> Option<Vec<i32>> {
    parse_wtc_array_q31_from(AAC_ROM_SOURCE, name)
}

fn parse_wtc_array_q31_from(source: &str, name: &str) -> Option<Vec<i32>> {
    let start = source.find(name)?;
    let body_start = source[start..].find('{')? + start + 1;
    let body_end = source[body_start..].find("};")? + body_start;
    let body = &source[body_start..body_end];
    let mut result = Vec::new();
    let mut remaining = body;
    while let Some(start) = remaining.find("WTC(0x") {
        let digits_start = start + 6;
        if remaining.len() < digits_start + 8 {
            return None;
        }
        let bits = u32::from_str_radix(&remaining[digits_start..digits_start + 8], 16).ok()?;
        result.push(bits as i32);
        remaining = &remaining[digits_start + 8..];
    }
    Some(result)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LdFilterbankError {
    UnsupportedFrameLength(usize),
    InvalidCoefficientCount {
        frame_length: usize,
        expected: usize,
        actual: usize,
    },
    SpectrumLengthMismatch {
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for LdFilterbankError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedFrameLength(length) => {
                write!(f, "unsupported AAC-LD filterbank length {length}")
            }
            Self::InvalidCoefficientCount {
                frame_length,
                expected,
                actual,
            } => write!(
                f,
                "AAC-LD {frame_length} filterbank has {actual} coefficients, expected {expected}"
            ),
            Self::SpectrumLengthMismatch { expected, actual } => write!(
                f,
                "AAC-LD filterbank expected {expected} spectral lines, got {actual}"
            ),
        }
    }
}

impl std::error::Error for LdFilterbankError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embeds_both_fdk_low_delay_coefficient_tables() {
        assert_eq!(synthesis_coefficients(512).unwrap().len(), 1536);
        assert_eq!(synthesis_coefficients(480).unwrap().len(), 1440);
        assert_eq!(synthesis_coefficients_q31(512).unwrap().len(), 1536);
        assert_eq!(synthesis_coefficients_q31(480).unwrap().len(), 1440);
    }

    #[test]
    fn preserves_signed_q31_values() {
        let coefficients = synthesis_coefficients(512).unwrap();
        assert!(coefficients.iter().any(|&value| value < 0.0));
        assert!(coefficients.iter().any(|&value| value > 0.0));
        assert!(coefficients
            .iter()
            .all(|&value| (-1.0..1.0).contains(&value)));
    }

    #[test]
    fn zero_input_preserves_zero_state_and_output() {
        for length in [480, 512] {
            let mut filterbank = LowDelayFilterbankF32::new(length).unwrap();
            assert!(filterbank
                .process(&vec![0.0; length])
                .unwrap()
                .iter()
                .all(|&sample| sample == 0.0));
            assert!(filterbank.state().iter().all(|&sample| sample == 0.0));
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn eld_filterbank_waveform_matches_fdk() {
        for length in [480, 512] {
            let spectrum = (0..length)
                .map(|index| {
                    let value = (index as f64 * 0.071).sin() * 0.0005
                        + (index as f64 * 0.193).cos() * 0.00025;
                    (value * 2_147_483_648.0).round() as i32
                })
                .collect::<Vec<_>>();
            let mut c = vec![0i32; length];
            assert_eq!(
                unsafe {
                    fdk_aac_sys::fdk_eld_filterbank_test(
                        spectrum.as_ptr(),
                        length as i32,
                        23,
                        c.as_mut_ptr(),
                    )
                },
                1
            );
            let rust = LowDelayFilterbankQ31::new(length)
                .unwrap()
                .process(&spectrum)
                .unwrap();
            let dot = rust
                .iter()
                .zip(&c)
                .map(|(&left, &right)| left as f64 * right as f64)
                .sum::<f64>();
            let rust_energy = rust
                .iter()
                .map(|&value| (value as f64).powi(2))
                .sum::<f64>();
            let c_energy = c.iter().map(|&value| (value as f64).powi(2)).sum::<f64>();
            let correlation = dot / (rust_energy * c_energy).sqrt();
            let rms_ratio = (rust_energy / c_energy).sqrt();
            assert!(
                correlation > 0.999,
                "ELD-{length} correlation {correlation}, RMS ratio {rms_ratio}"
            );
            assert!(
                (0.95..=1.05).contains(&rms_ratio),
                "ELD-{length} RMS ratio {rms_ratio}"
            );
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn eld_dct_iv_matches_fdk_order_and_sign() {
        let length = 480;
        let input = (0..length)
            .map(|index| {
                let value = (index as f64 * 0.071).sin() * 0.0005;
                (value * 2_147_483_648.0).round() as i32
            })
            .collect::<Vec<_>>();
        let mut c = vec![0i32; length];
        let mut c_scale = 0;
        assert_eq!(
            unsafe {
                fdk_aac_sys::fdk_dct_iv_test(
                    input.as_ptr(),
                    length as i32,
                    c.as_mut_ptr(),
                    &mut c_scale,
                )
            },
            0
        );
        let filterbank = LowDelayFilterbankQ31::new(length).unwrap();
        let rust = filterbank
            .dct_iv_kernel
            .chunks_exact(length)
            .map(|row| {
                let sum = input
                    .iter()
                    .zip(row)
                    .map(|(&sample, &coefficient)| sample as i128 * coefficient as i128)
                    .sum::<i128>();
                clamp_i128_to_i32(round_shift_i128(sum, 31))
            })
            .collect::<Vec<_>>();
        let dot = rust
            .iter()
            .zip(&c)
            .map(|(&left, &right)| left as f64 * right as f64)
            .sum::<f64>();
        let rust_energy = rust
            .iter()
            .map(|&value| (value as f64).powi(2))
            .sum::<f64>();
        let c_energy = c.iter().map(|&value| (value as f64).powi(2)).sum::<f64>();
        let correlation = dot / (rust_energy * c_energy).sqrt();
        let rms_ratio = (rust_energy / c_energy).sqrt();
        assert!(
            correlation > 0.9999,
            "DCT-IV correlation {correlation}, RMS ratio {rms_ratio}, C scale {c_scale}"
        );
    }

    #[test]
    fn impulse_updates_state_and_produces_finite_output() {
        let mut filterbank = LowDelayFilterbankF32::new(512).unwrap();
        let mut spectrum = vec![0.0; 512];
        spectrum[0] = 1.0;
        let output = filterbank.process(&spectrum).unwrap();
        assert!(output.iter().all(|sample| sample.is_finite()));
        assert!(output.iter().any(|&sample| sample != 0.0));
        assert!(filterbank.state().iter().any(|&sample| sample != 0.0));
    }

    #[test]
    fn q31_state_machine_tracks_f32_reference() {
        let mut float = LowDelayFilterbankF32::new(512).unwrap();
        let mut fixed = LowDelayFilterbankQ31::new(512).unwrap();
        let mut float_spectrum = vec![0.0; 512];
        let mut fixed_spectrum = vec![0; 512];
        float_spectrum[0] = 0.25;
        fixed_spectrum[0] = 0x2000_0000;
        let float_output = float.process(&float_spectrum).unwrap();
        let fixed_output = fixed.process(&fixed_spectrum).unwrap();
        let maximum_error = float_output
            .iter()
            .zip(fixed_output)
            .map(|(&reference, actual)| (reference - actual as f32 / 2_147_483_648.0).abs())
            .fold(0.0f32, f32::max);
        assert!(maximum_error < 2.0e-6, "maximum Q31 error {maximum_error}");
        assert!(fixed.state().iter().any(|&sample| sample != 0));
    }

    #[test]
    fn rejects_unsupported_lengths_and_spectrum_mismatches() {
        for length in [0, 256, 1024] {
            assert_eq!(
                synthesis_coefficients(length),
                Err(LdFilterbankError::UnsupportedFrameLength(length))
            );
            assert_eq!(
                synthesis_coefficients_q31(length),
                Err(LdFilterbankError::UnsupportedFrameLength(length))
            );
            assert!(matches!(
                LowDelayFilterbankF32::new(length),
                Err(LdFilterbankError::UnsupportedFrameLength(value)) if value == length
            ));
            assert!(matches!(
                LowDelayFilterbankQ31::new(length),
                Err(LdFilterbankError::UnsupportedFrameLength(value)) if value == length
            ));
        }
        let mut float = LowDelayFilterbankF32::new(480).unwrap();
        assert_eq!(
            float.process(&[0.0]),
            Err(LdFilterbankError::SpectrumLengthMismatch {
                expected: 480,
                actual: 1
            })
        );
        let mut fixed = LowDelayFilterbankQ31::new(480).unwrap();
        assert_eq!(
            fixed.process(&[0]),
            Err(LdFilterbankError::SpectrumLengthMismatch {
                expected: 480,
                actual: 1
            })
        );
    }

    #[test]
    fn q31_exponent_scaling_covers_positive_negative_and_extreme_shifts() {
        for exponent in [24, 22, i16::MAX, i16::MIN] {
            let mut filterbank = LowDelayFilterbankQ31::new(480).unwrap();
            let mut spectrum = vec![0; 480];
            spectrum[0] = 0x1000_0000;
            let output = filterbank
                .process_with_exponent(&spectrum, exponent)
                .unwrap();
            assert_eq!(output.len(), 480);
            assert_eq!(filterbank.state().len(), 960);
        }
    }

    #[test]
    fn numeric_helpers_round_and_clamp_both_signs() {
        assert_eq!(f64_to_q31(2.0), i32::MAX);
        assert_eq!(f64_to_q31(-2.0), i32::MIN);
        assert_eq!(f64_to_q31(0.5), 0x4000_0000);
        assert_eq!(round_shift_i128(7, 0), 7);
        assert_eq!(round_shift_i128(7, 2), 2);
        assert_eq!(round_shift_i128(-7, 2), -2);
        assert_eq!(clamp_i128_to_i32(i128::MAX), i32::MAX);
        assert_eq!(clamp_i128_to_i32(i128::MIN), i32::MIN);
        assert_eq!(clamp_i128_to_i32(7), 7);
    }

    #[test]
    fn rom_parser_reports_missing_names_and_preserves_q31_conversion() {
        assert_eq!(parse_wtc_array_q31("LowDelaySynthesis999"), None);
        assert_eq!(parse_wtc_array("LowDelaySynthesis999"), None);
        let q31 = parse_wtc_array_q31("LowDelaySynthesis480").unwrap();
        let float = parse_wtc_array("LowDelaySynthesis480").unwrap();
        assert_eq!(q31.len(), float.len());
        for (&fixed, &value) in q31.iter().zip(&float).take(16) {
            assert_eq!(value, fixed as f32 / 2_147_483_648.0);
        }

        assert_eq!(
            parse_wtc_array_q31_from(
                "LowDelaySynthesisTest { WTC(0x12345678) };",
                "LowDelaySynthesisTest",
            ),
            Some(vec![0x1234_5678])
        );
        assert_eq!(
            parse_wtc_array_q31_from(
                "LowDelaySynthesisTest { WTC(0x1234 };",
                "LowDelaySynthesisTest",
            ),
            None
        );
        assert_eq!(
            parse_wtc_array_q31_from(
                "LowDelaySynthesisTest { WTC(0xzzzzzzzz) };",
                "LowDelaySynthesisTest",
            ),
            None
        );
        assert_eq!(
            parse_wtc_array_q31_from("LowDelaySynthesisTest", "LowDelaySynthesisTest"),
            None
        );
        assert_eq!(
            parse_wtc_array_q31_from("LowDelaySynthesisTest {", "LowDelaySynthesisTest"),
            None
        );
    }

    #[test]
    fn coefficient_count_validator_reports_actual_length() {
        assert_eq!(validate_coefficient_count(480, 2, &[1, 2]), Ok(&[1, 2][..]));
        assert_eq!(
            validate_coefficient_count::<i32>(480, 1440, &[]),
            Err(LdFilterbankError::InvalidCoefficientCount {
                frame_length: 480,
                expected: 1440,
                actual: 0,
            })
        );
    }

    #[test]
    fn formats_every_low_delay_filterbank_error() {
        for error in [
            LdFilterbankError::UnsupportedFrameLength(256),
            LdFilterbankError::InvalidCoefficientCount {
                frame_length: 480,
                expected: 1440,
                actual: 0,
            },
            LdFilterbankError::SpectrumLengthMismatch {
                expected: 480,
                actual: 1,
            },
        ] {
            assert!(!error.to_string().is_empty());
        }
    }
}
