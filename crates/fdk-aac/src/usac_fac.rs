//! Frequency-domain aliasing cancellation (FAC) payload decoding.

use crate::bits::{BitError, BitReader};
use crate::usac_lpc::{decode_avq, decode_gain_f32, UsacLpcError};

#[derive(Debug, Clone, PartialEq)]
pub struct FacData {
    pub gain_code: Option<u8>,
    pub coefficients: Vec<f32>,
    pub bits_read: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FacError {
    Bit(BitError),
    Lpc(UsacLpcError),
    InvalidLength(usize),
    InvalidMode(u8),
}

impl From<BitError> for FacError {
    fn from(value: BitError) -> Self {
        Self::Bit(value)
    }
}

impl From<UsacLpcError> for FacError {
    fn from(value: UsacLpcError) -> Self {
        Self::Lpc(value)
    }
}

impl FacData {
    /// Port of `CLpd_FAC_Read`. FAC uses AVQ `nk_mode=1`, one RE8 vector per
    /// eight coefficients, and an optional 7-bit logarithmic gain.
    pub fn parse(
        reader: &mut BitReader<'_>,
        length: usize,
        use_gain: bool,
    ) -> Result<Self, FacError> {
        if length == 0 || length % 8 != 0 {
            return Err(FacError::InvalidLength(length));
        }
        let start = reader.bits_read();
        let gain_code = use_gain.then(|| reader.read_u8(7)).transpose()?;
        let gain = gain_code.map_or(1.0, decode_gain_f32);
        let coefficients = decode_avq(reader, 1, length / 8)?
            .into_iter()
            .map(|value| value as f32 * gain)
            .collect();
        Ok(Self {
            gain_code,
            coefficients,
            bits_read: reader.bits_read() - start,
        })
    }

    pub fn apply_tcx_gains(
        &mut self,
        tcx_gain: f32,
        alfd_gains: &[f32],
        mode: u8,
    ) -> Result<(), FacError> {
        let factor = match mode {
            0 => 0.5,
            1 => std::f32::consts::FRAC_1_SQRT_2 / 2.0,
            2 => 0.25,
            3 => std::f32::consts::FRAC_1_SQRT_2 / 4.0,
            _ => return Err(FacError::InvalidMode(mode)),
        } * tcx_gain;
        for value in &mut self.coefficients {
            *value *= factor;
        }
        for i in 0..self.coefficients.len() / 4 {
            let gain_index = i >> (3usize.saturating_sub(mode as usize));
            if let Some(&gain) = alfd_gains.get(gain_index) {
                self.coefficients[i] *= gain;
            }
        }
        Ok(())
    }

    pub fn synthesize_dct4(&self) -> Vec<f32> {
        inverse_dct4(&self.coefficients)
    }

    /// Build the 2*FAC transition signal: DCT-IV input followed by the zero
    /// input response of the 0.92-weighted LPC synthesis filter.
    pub fn synthesize_transition(&self, lpc: &[f32; 16], previous: &[f32]) -> Vec<f32> {
        let length = self.coefficients.len();
        let mut signal = vec![0.0; 2 * length];
        signal[..length].copy_from_slice(&self.synthesize_dct4());
        let weighted: [f32; 16] = std::array::from_fn(|i| lpc[i] * 0.92f32.powi((i + 1) as i32));
        for i in 0..2 * length {
            let prediction = (0..16)
                .map(|tap| {
                    let past_index = i as isize - tap as isize - 1;
                    let past = if past_index >= 0 {
                        signal[past_index as usize]
                    } else {
                        previous
                            .len()
                            .checked_sub((-past_index) as usize)
                            .and_then(|index| previous.get(index))
                            .copied()
                            .unwrap_or(0.0)
                    };
                    weighted[tap] * past
                })
                .sum::<f32>();
            signal[i] -= prediction;
        }
        for i in 0..length {
            let phase = std::f32::consts::FRAC_PI_2 * (i as f32 + 0.5) / length as f32;
            let old = previous
                .len()
                .checked_sub(length)
                .and_then(|start| previous.get(start + i))
                .copied()
                .unwrap_or(0.0);
            signal[i] = old * phase.cos() + signal[i] * phase.sin();
        }
        signal
    }
}

pub fn inverse_dct4(input: &[f32]) -> Vec<f32> {
    let length = input.len();
    let scale = (2.0 / length as f32).sqrt();
    (0..length)
        .map(|sample| {
            input
                .iter()
                .enumerate()
                .map(|(bin, &value)| {
                    let angle = std::f32::consts::PI / length as f32
                        * (sample as f32 + 0.5)
                        * (bin as f32 + 0.5);
                    value * angle.cos()
                })
                .sum::<f32>()
                * scale
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitWriter;

    #[test]
    fn parses_zero_fac_vector_without_gain() {
        let mut bits = BitWriter::new();
        bits.write_bool(false); // Q0
        let data = FacData::parse(&mut BitReader::new(&bits.finish()), 8, false).unwrap();
        assert_eq!(data.coefficients, vec![0.0; 8]);
        assert_eq!(data.bits_read, 1);
    }

    #[test]
    fn applies_fac_gain_to_re8_vector() {
        let mut bits = BitWriter::new();
        bits.write(28, 7); // gain 10
        bits.write_bool(true);
        bits.write_bool(false); // Q2
        bits.write(0, 8); // [1; 8]
        let data = FacData::parse(&mut BitReader::new(&bits.finish()), 8, true).unwrap();
        assert_eq!(data.gain_code, Some(28));
        assert!(data
            .coefficients
            .iter()
            .all(|value| (*value - 10.0).abs() < 1e-5));
    }

    #[test]
    fn validates_fac_length() {
        assert_eq!(
            FacData::parse(&mut BitReader::new(&[]), 7, false),
            Err(FacError::InvalidLength(7))
        );
    }

    #[test]
    fn dct4_is_self_inverse() {
        let input = vec![1.0, -2.0, 0.5, 3.0, -1.0, 0.0, 2.0, -0.5];
        let reconstructed = inverse_dct4(&inverse_dct4(&input));
        for (actual, expected) in reconstructed.iter().zip(input) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn applies_mode_dependent_tcx_gain() {
        let mut data = FacData {
            gain_code: None,
            coefficients: vec![1.0; 8],
            bits_read: 0,
        };
        data.apply_tcx_gains(2.0, &[0.5], 0).unwrap();
        assert_eq!(data.coefficients[0], 0.5);
        assert_eq!(data.coefficients[2], 1.0);
    }

    #[test]
    fn applies_all_high_tcx_modes_and_rejects_invalid_mode() {
        for (mode, expected) in [
            (1, std::f32::consts::FRAC_1_SQRT_2),
            (2, 0.5),
            (3, std::f32::consts::FRAC_1_SQRT_2 / 2.0),
        ] {
            let mut data = FacData {
                gain_code: None,
                coefficients: vec![1.0; 8],
                bits_read: 0,
            };
            data.apply_tcx_gains(2.0, &[], mode).unwrap();
            assert!((data.coefficients[0] - expected).abs() < 1e-6);
        }
        let mut data = FacData {
            gain_code: None,
            coefficients: vec![1.0; 8],
            bits_read: 0,
        };
        assert_eq!(
            data.apply_tcx_gains(1.0, &[], 4),
            Err(FacError::InvalidMode(4))
        );
    }

    #[test]
    fn converts_fac_bit_errors() {
        let bit = BitError::UnexpectedEof {
            needed_bits: 7,
            remaining_bits: 0,
        };
        assert_eq!(FacError::from(bit.clone()), FacError::Bit(bit));
        assert!(matches!(
            FacData::parse(&mut BitReader::new(&[]), 8, true),
            Err(FacError::Bit(BitError::UnexpectedEof { .. }))
        ));
    }

    #[test]
    fn fac_transition_extends_with_weighted_lpc_zir() {
        let data = FacData {
            gain_code: None,
            coefficients: vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            bits_read: 0,
        };
        let signal = data.synthesize_transition(&[0.0; 16], &[0.0; 8]);
        assert_eq!(signal.len(), 16);
        assert!(signal[..8].iter().any(|value| *value != 0.0));
        assert_eq!(signal[8..], [0.0; 8]);
    }
}
