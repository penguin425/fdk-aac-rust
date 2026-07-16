//! Parametric Stereo encoder analysis and payload writing.

use std::fmt;

use crate::bits::BitWriter;
use crate::ld_sbr_qmf::QmfSlot;
use crate::ps::{encode_ps_huffman, PsHuffmanBook};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PsEncoderFrame {
    pub iid: Vec<i8>,
    pub icc: Vec<i8>,
}

impl PsEncoderFrame {
    /// Write an SBR extended-data block beginning with extension id 2 (PS).
    pub fn write_sbr_extension(&self, header_present: bool) -> Result<Vec<u8>, PsEncoderError> {
        if self.iid.len() != 20 || self.icc.len() != 20 {
            return Err(PsEncoderError::InvalidParameterCount);
        }
        let mut writer = BitWriter::new();
        writer.write(2, 2); // EXTENSION_ID_PS_CODING
        writer.write_bool(header_present);
        if header_present {
            writer.write_bool(true); // IID enabled
            writer.write(1, 3); // 20-band coarse IID
            writer.write_bool(true); // ICC enabled
            writer.write(1, 3); // 20-band ICC
            writer.write_bool(false); // no PS extension data
        }
        writer.write_bool(false); // fixed borders
        writer.write(1, 2); // one envelope
        writer.write_bool(false); // IID frequency delta
        write_frequency_deltas(&mut writer, &self.iid, PsHuffmanBook::IidFrequency)?;
        writer.write_bool(false); // ICC frequency delta
        write_frequency_deltas(&mut writer, &self.icc, PsHuffmanBook::IccFrequency)?;
        writer.byte_align();
        Ok(writer.finish())
    }
}

fn write_frequency_deltas(
    writer: &mut BitWriter,
    values: &[i8],
    book: PsHuffmanBook,
) -> Result<(), PsEncoderError> {
    let mut previous = 0i8;
    for &value in values {
        let delta = value - previous;
        let code = encode_ps_huffman(book, delta)
            .ok_or(PsEncoderError::UnrepresentableHuffmanSymbol(delta))?;
        for bit in code {
            writer.write_bool(bit);
        }
        previous = value;
    }
    Ok(())
}

pub fn analyze_ps_qmf(
    left: &[QmfSlot],
    right: &[QmfSlot],
) -> Result<PsEncoderFrame, PsEncoderError> {
    if left.is_empty() || left.len() != right.len() {
        return Err(PsEncoderError::QmfLayoutMismatch);
    }
    if left
        .iter()
        .chain(right)
        .any(|slot| slot.real.len() < 64 || slot.imaginary.len() < 64)
    {
        return Err(PsEncoderError::QmfLayoutMismatch);
    }
    let mut iid = Vec::with_capacity(20);
    let mut icc = Vec::with_capacity(20);
    for parameter_band in 0..20 {
        let start = parameter_band * 64 / 20;
        let end = (parameter_band + 1) * 64 / 20;
        let mut left_energy = 0.0;
        let mut right_energy = 0.0;
        let mut cross_real = 0.0;
        let mut cross_imaginary = 0.0;
        for (left, right) in left.iter().zip(right) {
            for band in start..end {
                let (lr, li) = (left.real[band], left.imaginary[band]);
                let (rr, ri) = (right.real[band], right.imaginary[band]);
                left_energy += lr * lr + li * li;
                right_energy += rr * rr + ri * ri;
                cross_real += lr * rr + li * ri;
                cross_imaginary += li * rr - lr * ri;
            }
        }
        let level = if left_energy + right_energy <= f64::EPSILON {
            0
        } else {
            (1.5 * ((left_energy + 1.0e-20) / (right_energy + 1.0e-20)).log2())
                .round()
                .clamp(-7.0, 7.0) as i8
        };
        let denominator = (left_energy * right_energy).sqrt();
        let correlation = if denominator <= f64::EPSILON {
            1.0
        } else {
            (cross_real.hypot(cross_imaginary) / denominator).clamp(0.0, 1.0)
        };
        iid.push(level);
        icc.push(((1.0 - correlation) * 7.0).round() as i8);
    }
    Ok(PsEncoderFrame { iid, icc })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PsEncoderError {
    QmfLayoutMismatch,
    InvalidParameterCount,
    UnrepresentableHuffmanSymbol(i8),
}

impl fmt::Display for PsEncoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::QmfLayoutMismatch => write!(f, "PS encoder QMF layout mismatch"),
            Self::InvalidParameterCount => write!(f, "PS encoder requires 20 IID/ICC parameters"),
            Self::UnrepresentableHuffmanSymbol(symbol) => {
                write!(f, "unrepresentable PS Huffman symbol {symbol}")
            }
        }
    }
}

impl std::error::Error for PsEncoderError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ps::PsParser;

    #[test]
    fn analyzes_level_correlation_and_roundtrips_ps_payload() {
        let mut left = vec![
            QmfSlot {
                real: vec![0.0; 64],
                imaginary: vec![0.0; 64]
            };
            32
        ];
        let mut right = left.clone();
        for slot in 0..32 {
            for band in 0..64 {
                left[slot].real[band] = (slot + band + 1) as f64 * 0.01;
                right[slot].real[band] = left[slot].real[band] * 0.5;
            }
        }
        let encoded = analyze_ps_qmf(&left, &right).unwrap();
        assert!(encoded.iid.iter().all(|&value| value > 0));
        assert!(encoded.icc.iter().all(|&value| value == 0));
        let payload = encoded.write_sbr_extension(true).unwrap();
        let decoded = PsParser::new()
            .parse_sbr_extension(&payload, 32)
            .unwrap()
            .unwrap();
        assert_eq!(decoded.iid_mapped_20[0], encoded.iid);
        assert_eq!(decoded.icc_mapped_20[0], encoded.icc);
    }

    #[test]
    fn validates_qmf_layouts_and_parameter_counts() {
        assert_eq!(
            analyze_ps_qmf(&[], &[]),
            Err(PsEncoderError::QmfLayoutMismatch)
        );
        let slot = QmfSlot {
            real: vec![0.0; 64],
            imaginary: vec![0.0; 64],
        };
        assert_eq!(
            analyze_ps_qmf(std::slice::from_ref(&slot), &[]),
            Err(PsEncoderError::QmfLayoutMismatch)
        );
        let short = QmfSlot {
            real: vec![0.0; 63],
            imaginary: vec![0.0; 64],
        };
        assert_eq!(
            analyze_ps_qmf(std::slice::from_ref(&short), std::slice::from_ref(&short)),
            Err(PsEncoderError::QmfLayoutMismatch)
        );
        assert_eq!(
            PsEncoderFrame {
                iid: vec![0; 19],
                icc: vec![0; 20]
            }
            .write_sbr_extension(false),
            Err(PsEncoderError::InvalidParameterCount)
        );
    }

    #[test]
    fn analyzes_silence_and_one_sided_energy() {
        let silence = QmfSlot {
            real: vec![0.0; 64],
            imaginary: vec![0.0; 64],
        };
        let silent = analyze_ps_qmf(
            std::slice::from_ref(&silence),
            std::slice::from_ref(&silence),
        )
        .unwrap();
        assert_eq!(silent.iid, vec![0; 20]);
        assert_eq!(silent.icc, vec![0; 20]);

        let left = QmfSlot {
            real: vec![1.0; 64],
            imaginary: vec![0.0; 64],
        };
        let one_sided =
            analyze_ps_qmf(std::slice::from_ref(&left), std::slice::from_ref(&silence)).unwrap();
        assert_eq!(one_sided.iid, vec![7; 20]);
        assert_eq!(one_sided.icc, vec![0; 20]);
    }

    #[test]
    fn rejects_unrepresentable_deltas_and_formats_errors() {
        let frame = PsEncoderFrame {
            iid: vec![100; 20],
            icc: vec![0; 20],
        };
        assert_eq!(
            frame.write_sbr_extension(false),
            Err(PsEncoderError::UnrepresentableHuffmanSymbol(100))
        );
        for error in [
            PsEncoderError::QmfLayoutMismatch,
            PsEncoderError::InvalidParameterCount,
            PsEncoderError::UnrepresentableHuffmanSymbol(42),
        ] {
            assert!(!error.to_string().is_empty());
        }
    }
}
