//! USAC common-window M/S and complex-prediction side information.

use crate::bits::{BitError, BitReader};
use crate::huffman::{decode_fdk_2bit, HuffmanError, HUFFMAN_CODEBOOK_SCL};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UsacStereoData {
    pub mask_present: u8,
    pub used: Vec<Vec<bool>>,
    pub complex_prediction: Option<ComplexPredictionData>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComplexPredictionData {
    pub prediction_direction_side_to_mid: bool,
    pub complex_coefficients: bool,
    pub use_previous_frame: bool,
    pub delta_code_time: bool,
    pub alpha_real: Vec<Vec<i16>>,
    pub alpha_imaginary: Vec<Vec<i16>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UsacStereoError {
    Bit(BitError),
    Huffman(HuffmanError),
}

impl From<BitError> for UsacStereoError {
    fn from(v: BitError) -> Self {
        Self::Bit(v)
    }
}
impl From<HuffmanError> for UsacStereoError {
    fn from(v: HuffmanError) -> Self {
        Self::Huffman(v)
    }
}

impl UsacStereoData {
    pub fn parse(
        reader: &mut BitReader<'_>,
        groups: usize,
        max_sfb: usize,
        independent: bool,
    ) -> Result<Self, UsacStereoError> {
        let mask_present = reader.read_u8(2)?;
        let mut used = vec![vec![false; max_sfb]; groups];
        match mask_present {
            1 => {
                for group in &mut used {
                    for flag in group {
                        *flag = reader.read_bool()?;
                    }
                }
            }
            2 => {
                for group in &mut used {
                    group.fill(true);
                }
            }
            3 => {
                let all = reader.read_bool()?;
                if all {
                    for group in &mut used {
                        group.fill(true);
                    }
                } else {
                    for group in &mut used {
                        for band in (0..max_sfb).step_by(2) {
                            let value = reader.read_bool()?;
                            group[band] = value;
                            if band + 1 < max_sfb {
                                group[band + 1] = value;
                            }
                        }
                    }
                }
            }
            _ => {}
        }
        let complex_prediction = if mask_present == 3 {
            let prediction_direction_side_to_mid = reader.read_bool()?;
            let complex_coefficients = reader.read_bool()?;
            let use_previous_frame = complex_coefficients && !independent && reader.read_bool()?;
            let delta_code_time = !independent && reader.read_bool()?;
            let mut real = vec![vec![0i16; max_sfb]; groups];
            let mut imaginary = vec![vec![0i16; max_sfb]; groups];
            for group in 0..groups {
                for band in (0..max_sfb).step_by(2) {
                    if used[group][band] {
                        let previous = if delta_code_time && group > 0 {
                            real[group - 1][band]
                        } else if band > 0 {
                            real[group][band - 1]
                        } else {
                            0
                        };
                        real[group][band] = previous
                            - (decode_fdk_2bit(reader, &HUFFMAN_CODEBOOK_SCL)? as i16 - 60);
                        if complex_coefficients {
                            let previous = if delta_code_time && group > 0 {
                                imaginary[group - 1][band]
                            } else if band > 0 {
                                imaginary[group][band - 1]
                            } else {
                                0
                            };
                            imaginary[group][band] = previous
                                - (decode_fdk_2bit(reader, &HUFFMAN_CODEBOOK_SCL)? as i16 - 60);
                        }
                        if band + 1 < max_sfb {
                            real[group][band + 1] = real[group][band];
                            imaginary[group][band + 1] = imaginary[group][band];
                        }
                    }
                }
            }
            Some(ComplexPredictionData {
                prediction_direction_side_to_mid,
                complex_coefficients,
                use_previous_frame,
                delta_code_time,
                alpha_real: real,
                alpha_imaginary: imaginary,
            })
        } else {
            None
        };
        Ok(Self {
            mask_present,
            used,
            complex_prediction,
        })
    }

    pub fn apply_ms(
        &self,
        left: &mut [Vec<f32>],
        right: &mut [Vec<f32>],
        offsets: &[usize],
        group_lengths: &[u8],
    ) {
        let mut window = 0;
        for (group, &length) in group_lengths.iter().enumerate() {
            for relative in 0..usize::from(length) {
                for band in 0..self.used[group].len() {
                    if self.used[group][band] {
                        for line in offsets[band]..offsets[band + 1] {
                            let mid = left[window + relative][line];
                            let side = right[window + relative][line];
                            left[window + relative][line] = mid + side;
                            right[window + relative][line] = mid - side;
                        }
                    }
                }
            }
            window += usize::from(length);
        }
    }

    /// Apply USAC complex prediction. The imaginary downmix is reconstructed
    /// from the MDCT using the odd-symmetric half-bin finite-difference form;
    /// alpha quantization is 0.1 per integer step in the reference decoder.
    pub fn apply_complex_prediction(
        &self,
        left: &mut [Vec<f32>],
        right: &mut [Vec<f32>],
        offsets: &[usize],
        group_lengths: &[u8],
        previous_downmix: Option<&[f32]>,
    ) -> Option<Vec<f32>> {
        let prediction = self.complex_prediction.as_ref()?;
        let mut window = 0;
        let mut last_downmix = Vec::new();
        for (group, &length) in group_lengths.iter().enumerate() {
            for relative in 0..usize::from(length) {
                let index = window + relative;
                let mut downmix = vec![0.0; left[index].len()];
                for line in 0..downmix.len() {
                    downmix[line] = if prediction.prediction_direction_side_to_mid {
                        0.5 * (left[index][line] - right[index][line])
                    } else {
                        0.5 * (left[index][line] + right[index][line])
                    };
                }
                let mut imaginary = vec![0.0; downmix.len()];
                for line in 0..downmix.len() {
                    let before = if line == 0 {
                        if prediction.use_previous_frame {
                            previous_downmix
                                .and_then(|values| values.last())
                                .copied()
                                .unwrap_or(0.0)
                        } else {
                            0.0
                        }
                    } else {
                        downmix[line - 1]
                    };
                    let after = downmix.get(line + 1).copied().unwrap_or(0.0);
                    imaginary[line] = 0.5 * (after - before);
                }
                for band in 0..self.used[group].len() {
                    if !self.used[group][band] {
                        continue;
                    }
                    let alpha_real = 0.1 * prediction.alpha_real[group][band] as f32;
                    let alpha_imaginary = 0.1 * prediction.alpha_imaginary[group][band] as f32;
                    for line in offsets[band]..offsets[band + 1] {
                        let side = right[index][line]
                            - alpha_real * downmix[line]
                            - alpha_imaginary * imaginary[line];
                        let base = left[index][line];
                        left[index][line] = base + side;
                        right[index][line] = if prediction.prediction_direction_side_to_mid {
                            side - base
                        } else {
                            base - side
                        };
                    }
                }
                last_downmix = downmix;
            }
            window += usize::from(length);
        }
        Some(last_downmix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitWriter;
    use crate::huffman::write_fdk_huffman_word;

    #[test]
    fn parses_full_ms_mask_and_applies_matrix() {
        let data = UsacStereoData::parse(&mut BitReader::new(&[0x80]), 1, 1, true).unwrap();
        let mut left = vec![vec![2.0, 0.0]];
        let mut right = vec![vec![0.5, 0.0]];
        data.apply_ms(&mut left, &mut right, &[0, 2], &[1]);
        assert_eq!(left[0][0], 2.5);
        assert_eq!(right[0][0], 1.5);
    }

    #[test]
    fn parses_selective_ms_flags() {
        // mask=1 followed by flags 1,0,1
        let data = UsacStereoData::parse(&mut BitReader::new(&[0b0110_1000]), 1, 3, true).unwrap();
        assert_eq!(data.used, vec![vec![true, false, true]]);
    }

    #[test]
    fn applies_real_complex_prediction_alpha() {
        let data = UsacStereoData {
            mask_present: 3,
            used: vec![vec![true]],
            complex_prediction: Some(ComplexPredictionData {
                prediction_direction_side_to_mid: false,
                complex_coefficients: false,
                use_previous_frame: false,
                delta_code_time: false,
                alpha_real: vec![vec![2]],
                alpha_imaginary: vec![vec![0]],
            }),
        };
        let mut left = vec![vec![2.0, 2.0]];
        let mut right = vec![vec![1.0, 1.0]];
        data.apply_complex_prediction(&mut left, &mut right, &[0, 2], &[1], None);
        assert!(left[0].iter().all(|value| value.is_finite()));
        assert_ne!(left[0], vec![2.0, 2.0]);
    }

    #[test]
    fn parses_absent_and_paired_selective_masks() {
        let absent = UsacStereoData::parse(&mut BitReader::new(&[0]), 2, 3, true).unwrap();
        assert_eq!(absent.mask_present, 0);
        assert_eq!(absent.used, vec![vec![false; 3]; 2]);
        assert!(absent.complex_prediction.is_none());

        let mut writer = BitWriter::new();
        writer.write(3, 2);
        writer.write_bool(false);
        writer.write_bool(true);
        writer.write_bool(false);
        writer.write_bool(false);
        writer.write_bool(false);
        writer.write_bool(false);
        write_fdk_huffman_word(&mut writer, &HUFFMAN_CODEBOOK_SCL, 60).unwrap();
        let bytes = writer.finish();
        let paired = UsacStereoData::parse(&mut BitReader::new(&bytes), 1, 4, true).unwrap();
        assert_eq!(paired.used, vec![vec![true, true, false, false]]);
        assert!(paired.complex_prediction.is_some());
    }

    #[test]
    fn parses_complex_coefficients_with_time_deltas() {
        let mut writer = BitWriter::new();
        writer.write(3, 2);
        writer.write_bool(true); // all bands
        writer.write_bool(true); // side predicts mid
        writer.write_bool(true); // complex coefficients
        writer.write_bool(true); // previous frame
        writer.write_bool(true); // delta in time
        for _ in 0..4 {
            write_fdk_huffman_word(&mut writer, &HUFFMAN_CODEBOOK_SCL, 59).unwrap();
            write_fdk_huffman_word(&mut writer, &HUFFMAN_CODEBOOK_SCL, 61).unwrap();
        }
        let bytes = writer.finish();
        let parsed = UsacStereoData::parse(&mut BitReader::new(&bytes), 2, 3, false).unwrap();
        let prediction = parsed.complex_prediction.unwrap();
        assert!(prediction.prediction_direction_side_to_mid);
        assert!(prediction.complex_coefficients);
        assert!(prediction.use_previous_frame);
        assert!(prediction.delta_code_time);
        assert_eq!(prediction.alpha_real[0][0], 1);
        assert_eq!(prediction.alpha_real[0][1], 1);
        assert_eq!(prediction.alpha_real[1][0], 2);
        assert_eq!(prediction.alpha_imaginary[1][0], -2);
    }

    #[test]
    fn parser_propagates_bit_errors_for_masks_and_coefficients() {
        assert!(matches!(
            UsacStereoData::parse(&mut BitReader::new(&[]), 1, 1, true),
            Err(UsacStereoError::Bit(BitError::UnexpectedEof { .. }))
        ));
        assert!(UsacStereoData::parse(&mut BitReader::new(&[0b0100_0000]), 1, 8, true).is_err());

        let mut writer = BitWriter::new();
        writer.write(3, 2);
        writer.write_bool(true);
        writer.write_bool(false);
        writer.write_bool(false);
        let bytes = writer.finish();
        assert!(UsacStereoData::parse(&mut BitReader::new(&bytes), 1, 16, true).is_err());
    }

    #[test]
    fn ms_matrix_honors_groups_windows_and_unused_bands() {
        let data = UsacStereoData {
            mask_present: 1,
            used: vec![vec![true, false], vec![false, true]],
            complex_prediction: None,
        };
        let mut left = vec![vec![2.0, 3.0]; 3];
        let mut right = vec![vec![0.5, 1.0]; 3];
        data.apply_ms(&mut left, &mut right, &[0, 1, 2], &[2, 1]);
        assert_eq!(left[0], vec![2.5, 3.0]);
        assert_eq!(left[1], vec![2.5, 3.0]);
        assert_eq!(left[2], vec![2.0, 4.0]);
        assert_eq!(right[2], vec![0.5, 2.0]);
    }

    #[test]
    fn complex_prediction_uses_imaginary_previous_and_direction_paths() {
        let without_prediction = UsacStereoData {
            mask_present: 0,
            used: vec![vec![false]],
            complex_prediction: None,
        };
        assert!(without_prediction
            .apply_complex_prediction(&mut [vec![0.0]], &mut [vec![0.0]], &[0, 1], &[1], None)
            .is_none());

        let data = UsacStereoData {
            mask_present: 3,
            used: vec![vec![true, false]],
            complex_prediction: Some(ComplexPredictionData {
                prediction_direction_side_to_mid: true,
                complex_coefficients: true,
                use_previous_frame: true,
                delta_code_time: false,
                alpha_real: vec![vec![1, 0]],
                alpha_imaginary: vec![vec![2, 0]],
            }),
        };
        let mut left = vec![vec![4.0, 8.0, 12.0]];
        let mut right = vec![vec![2.0, 4.0, 6.0]];
        let downmix = data
            .apply_complex_prediction(&mut left, &mut right, &[0, 2, 3], &[1], Some(&[10.0]))
            .unwrap();
        assert_eq!(downmix, vec![1.0, 2.0, 3.0]);
        assert!(left[0].iter().all(|value| value.is_finite()));
        assert_eq!(left[0][2], 12.0);
        assert_eq!(right[0][2], 6.0);
    }
}
