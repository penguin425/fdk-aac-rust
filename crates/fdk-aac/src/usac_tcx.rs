//! USAC transform-coded excitation (TCX) payload and spectral reconstruction.

use crate::bits::{BitError, BitReader};
use crate::usac_arith::{UsacArithmeticDecoder, UsacArithmeticError};
use crate::usac_lpc::decode_gain_f32;

#[derive(Debug, Clone, PartialEq)]
pub struct TcxFrame {
    pub noise_factor: u8,
    pub global_gain: u8,
    pub arithmetic_reset: bool,
    pub quantized_spectrum: Vec<i32>,
    pub spectrum: Vec<f32>,
    pub bits_read: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TcxError {
    Bit(BitError),
    Arithmetic(UsacArithmeticError),
    InvalidLength(usize),
}

impl From<BitError> for TcxError {
    fn from(value: BitError) -> Self {
        Self::Bit(value)
    }
}

impl From<UsacArithmeticError> for TcxError {
    fn from(value: UsacArithmeticError) -> Self {
        Self::Arithmetic(value)
    }
}

impl TcxFrame {
    pub fn parse(
        reader: &mut BitReader<'_>,
        arithmetic: &mut UsacArithmeticDecoder,
        length: usize,
        first_tcx: bool,
        independent: bool,
        noise_seed: &mut u32,
    ) -> Result<Self, TcxError> {
        if length == 0 || length > 1024 || length & 1 != 0 {
            return Err(TcxError::InvalidLength(length));
        }
        let start = reader.bits_read();
        let noise_factor = reader.read_u8(3)?;
        let global_gain = reader.read_u8(7)?;
        let arithmetic_reset = first_tcx && (independent || reader.read_bool()?);
        let quantized_spectrum = arithmetic.decode(reader, length, length, arithmetic_reset)?;
        let mut spectrum: Vec<_> = quantized_spectrum
            .iter()
            .map(|&value| value as f32)
            .collect();
        fill_noise(&mut spectrum, noise_factor, noise_seed);
        normalize_gain(&mut spectrum, global_gain);
        Ok(Self {
            noise_factor,
            global_gain,
            arithmetic_reset,
            quantized_spectrum,
            spectrum,
            bits_read: reader.bits_read() - start,
        })
    }
}

pub fn fill_noise(spectrum: &mut [f32], noise_factor: u8, seed: &mut u32) {
    let level = 0.0625 * f32::from(8u8.saturating_sub(noise_factor.min(7)));
    let start = spectrum.len() / 6;
    for chunk in spectrum[start..].chunks_mut(8) {
        if chunk.iter().all(|&value| value == 0.0) {
            for value in chunk {
                *seed = seed.wrapping_mul(69069).wrapping_add(5);
                *value = if *seed & 0x1_0000 != 0 { -level } else { level };
            }
        }
    }
}

pub fn normalize_gain(spectrum: &mut [f32], gain_code: u8) {
    let energy = spectrum
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .max(0.01);
    let factor = decode_gain_f32(gain_code) * spectrum.len() as f32 / energy.sqrt();
    for value in spectrum {
        *value *= factor;
    }
}

/// Adaptive low-frequency deemphasis. Returns one gain per eight-line block
/// for subsequent FAC deshaping.
pub fn adaptive_low_frequency_deemphasis(spectrum: &mut [f32]) -> Vec<f32> {
    let end = spectrum.len() / 4;
    let energies: Vec<_> = spectrum[..end]
        .chunks(8)
        .map(|chunk| {
            chunk
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .max(0.01)
        })
        .collect();
    let peak = energies.iter().copied().fold(0.01f32, f32::max);
    let mut previous_gain = 0.1;
    let mut gains = Vec::with_capacity(energies.len());
    for (chunk, energy) in spectrum[..end].chunks_mut(8).zip(energies) {
        let gain = (energy / peak).sqrt().max(previous_gain);
        for value in chunk {
            *value *= gain;
        }
        gains.push(gain);
        previous_gain = gain;
    }
    gains
}

/// Interpolated frequency-domain noise shaping between the old and new LPC
/// envelopes. FDK uses `1/|A(e^jw)|` at 48/64 control points and a first-order
/// recurrence between both endpoint gains.
pub fn apply_fdns(spectrum: &mut [f32], old_lpc: &[f32; 16], new_lpc: &[f32; 16]) {
    let points = if spectrum.len() % 64 == 0 { 64 } else { 48 };
    let step = spectrum.len() / points;
    let envelope = |lpc: &[f32; 16], point: usize| {
        let omega = std::f32::consts::PI * point as f32 / points as f32;
        let mut real = 1.0;
        let mut imaginary = 0.0;
        let mut weight = 0.92;
        for (i, &coefficient) in lpc.iter().enumerate() {
            let phase = omega * (i + 1) as f32;
            real += weight * coefficient * phase.cos();
            imaginary -= weight * coefficient * phase.sin();
            weight *= 0.92;
        }
        1.0 / (real * real + imaginary * imaginary).sqrt().max(1e-9)
    };
    let mut previous = 0.0;
    for point in 0..points {
        let old = envelope(old_lpc, point);
        let new = envelope(new_lpc, point);
        let sum = (old + new).max(1e-9);
        let feedforward = 2.0 * old * new / sum;
        let feedback = (new - old) / sum;
        for value in &mut spectrum[point * step..(point + 1) * step] {
            let shaped = feedforward * *value + feedback * previous;
            *value = shaped;
            previous = shaped;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pseudo_random_payload(seed: u32) -> Vec<u8> {
        let mut state = seed;
        let mut payload = vec![0u8; 256];
        for byte in &mut payload {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *byte = (state >> 24) as u8;
        }
        payload
    }

    #[test]
    fn fills_only_all_zero_noise_blocks() {
        let mut spectrum = vec![0.0; 48];
        spectrum[16] = 1.0;
        let mut seed = 0;
        fill_noise(&mut spectrum, 7, &mut seed);
        assert_eq!(spectrum[16], 1.0);
        assert!(spectrum[8..16].iter().all(|value| value.abs() == 0.0625));
        assert!(
            spectrum[16..24]
                .iter()
                .filter(|&&value| value != 0.0)
                .count()
                == 1
        );
        assert!(spectrum[24..].iter().all(|value| value.abs() == 0.0625));
    }

    #[test]
    fn tcx_gain_matches_reference_equation() {
        let mut spectrum = vec![1.0; 4];
        normalize_gain(&mut spectrum, 0);
        assert!(spectrum.iter().all(|value| (*value - 2.0).abs() < 1e-6));
    }

    #[test]
    fn validates_tcx_length_before_reading() {
        assert_eq!(
            TcxFrame::parse(
                &mut BitReader::new(&[]),
                &mut UsacArithmeticDecoder::new(),
                3,
                true,
                true,
                &mut 0,
            ),
            Err(TcxError::InvalidLength(3))
        );
    }

    #[test]
    fn low_frequency_deemphasis_returns_fac_gains() {
        let mut spectrum = vec![1.0; 256];
        let gains = adaptive_low_frequency_deemphasis(&mut spectrum);
        assert_eq!(gains.len(), 8);
        assert!(gains.iter().all(|gain| *gain > 0.0 && *gain <= 1.0));
    }

    #[test]
    fn flat_lpc_fdns_preserves_spectrum() {
        let mut spectrum = vec![1.0; 256];
        apply_fdns(&mut spectrum, &[0.0; 16], &[0.0; 16]);
        assert!(spectrum.iter().all(|value| (*value - 1.0).abs() < 1e-6));
    }

    #[test]
    fn parses_deterministic_tcx_payload_and_reconstructs_spectrum() {
        let payload = pseudo_random_payload(1);
        let mut arithmetic = UsacArithmeticDecoder::new();
        let mut noise_seed = 0x1234_5678;
        let frame = TcxFrame::parse(
            &mut BitReader::new(&payload),
            &mut arithmetic,
            2,
            true,
            true,
            &mut noise_seed,
        )
        .unwrap();
        assert!(frame.arithmetic_reset);
        assert_eq!(frame.quantized_spectrum.len(), 2);
        assert_eq!(frame.spectrum.len(), 2);
        assert!(frame.spectrum.iter().all(|value| value.is_finite()));
        assert!(frame.bits_read >= 10);
    }

    #[test]
    fn parse_covers_signalled_and_missing_arithmetic_context() {
        // first_tcx + dependent frame reads the explicit reset flag.
        let mut payload = vec![0u8; 16];
        payload[1] = 0; // reset flag is false after the ten header bits
        assert!(matches!(
            TcxFrame::parse(
                &mut BitReader::new(&payload),
                &mut UsacArithmeticDecoder::new(),
                2,
                true,
                false,
                &mut 0,
            ),
            Err(TcxError::Arithmetic(_))
        ));
        // A non-first TCX block does not consume a reset bit and requires a
        // previously established arithmetic context.
        assert_eq!(
            TcxFrame::parse(
                &mut BitReader::new(&payload),
                &mut UsacArithmeticDecoder::new(),
                2,
                false,
                true,
                &mut 0,
            ),
            Err(TcxError::Arithmetic(
                UsacArithmeticError::MissingPreviousContext
            ))
        );
    }

    #[test]
    fn validates_all_lengths_and_converts_nested_errors() {
        for length in [0, 1025, 5] {
            assert_eq!(
                TcxFrame::parse(
                    &mut BitReader::new(&[]),
                    &mut UsacArithmeticDecoder::new(),
                    length,
                    true,
                    true,
                    &mut 0,
                ),
                Err(TcxError::InvalidLength(length))
            );
        }
        let bit = BitError::UnexpectedEof {
            needed_bits: 3,
            remaining_bits: 0,
        };
        assert_eq!(TcxError::from(bit.clone()), TcxError::Bit(bit));
        assert_eq!(
            TcxError::from(UsacArithmeticError::EscapeOverflow),
            TcxError::Arithmetic(UsacArithmeticError::EscapeOverflow)
        );
        assert!(matches!(
            TcxFrame::parse(
                &mut BitReader::new(&[]),
                &mut UsacArithmeticDecoder::new(),
                2,
                true,
                true,
                &mut 0,
            ),
            Err(TcxError::Bit(BitError::UnexpectedEof { .. }))
        ));
    }

    #[test]
    fn noise_deemphasis_and_fdns_cover_extreme_shapes() {
        let mut spectrum = vec![0.0; 8];
        let mut seed = 1;
        fill_noise(&mut spectrum, 0, &mut seed);
        assert!(spectrum[1..].iter().all(|value| value.abs() == 0.5));

        let mut empty = Vec::new();
        normalize_gain(&mut empty, 127);
        assert!(adaptive_low_frequency_deemphasis(&mut empty).is_empty());

        let mut spectrum = vec![1.0; 240];
        let mut old = [0.0; 16];
        let mut new = [0.0; 16];
        old[0] = 0.2;
        new[0] = -0.2;
        apply_fdns(&mut spectrum, &old, &new);
        assert!(spectrum.iter().all(|value| value.is_finite()));
        assert!(spectrum.iter().any(|value| (*value - 1.0).abs() > 1e-4));
    }
}
