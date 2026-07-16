//! USAC algebraic code-excited linear prediction (ACELP) payload parsing.

use crate::bits::{BitError, BitReader};
use std::sync::LazyLock;

const CORE_MODE_BITS: [u8; 8] = [20, 28, 36, 44, 52, 64, 12, 16];
const PITCH_BITS_1024: [u8; 4] = [9, 6, 9, 6];
const PITCH_BITS_768: [u8; 3] = [9, 6, 6];
const FDK_USAC_ROM: &str = include_str!(concat!(
    env!("FDK_AAC_UPSTREAM_DIR"),
    "/libAACdec/src/usacdec_rom.cpp"
));
static GAIN_TABLE: LazyLock<Vec<u16>> = LazyLock::new(|| {
    let marker = "fdk_t_qua_gain7b[";
    let start = FDK_USAC_ROM.find(marker).expect("ACELP gain ROM");
    let source = &FDK_USAC_ROM[start..];
    let body = &source[source.find('{').unwrap() + 1..source.find("};").unwrap()];
    let values: Vec<_> = body
        .split(|c: char| !c.is_ascii_digit())
        .filter(|word| !word.is_empty())
        .map(|word| word.parse().unwrap())
        .collect();
    assert_eq!(values.len(), 256);
    values
});
const FDK_LTP_SOURCE: &str = include_str!(concat!(
    env!("FDK_AAC_UPSTREAM_DIR"),
    "/libAACdec/src/usacdec_ace_ltp.cpp"
));
static PITCH_FILTER: LazyLock<Vec<f32>> = LazyLock::new(|| {
    let marker = "Pred_lt4_inter4_2[";
    let start = FDK_LTP_SOURCE.find(marker).expect("ACELP pitch filter ROM");
    let source = &FDK_LTP_SOURCE[start..];
    let body = &source[source.find('{').unwrap() + 1..source.find("};").unwrap()];
    let packed: Vec<_> = body
        .split("0x")
        .skip(1)
        .map(|entry| u32::from_str_radix(&entry[..8], 16).unwrap())
        .collect();
    assert_eq!(packed.len(), 64);
    packed
        .into_iter()
        .flat_map(|value| {
            [
                (value >> 16) as i16 as f32 / 32768.0,
                value as i16 as f32 / 32768.0,
            ]
        })
        .collect()
});

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcelpSubframe {
    pub pitch_lag: u16,
    pub pitch_fraction_quarters: u8,
    pub ltp_filtering: bool,
    pub innovative_indices: Vec<u16>,
    pub gain_index: u8,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcelpGains {
    pub pitch: f32,
    pub innovative: f32,
    pub innovation_energy: f32,
}

impl AcelpSubframe {
    pub fn innovative_code(&self, core_mode: u8) -> Result<[f32; 64], AcelpError> {
        let bits = *CORE_MODE_BITS
            .get(usize::from(core_mode))
            .ok_or(AcelpError::InvalidCoreMode(core_mode))?;
        decode_4t64(&self.innovative_indices, bits)
    }

    pub fn decode_gains(&self, mean_energy_index: u8, code: &[f32]) -> AcelpGains {
        let energy = code.iter().map(|value| value * value).sum::<f32>();
        let normalized = (code.len() as f32 / (energy + 0.01)).sqrt();
        let mean_db = [18.0, 30.0, 42.0, 54.0][usize::from(mean_energy_index & 3)];
        let base = 10.0f32.powf(mean_db / 20.0) * normalized;
        let index = usize::from(self.gain_index) * 2;
        AcelpGains {
            pitch: f32::from(GAIN_TABLE[index]) / 16384.0,
            innovative: base * f32::from(GAIN_TABLE[index + 1]) / 2048.0,
            innovation_energy: energy,
        }
    }
}

fn pulse_1(index: i32, position_bits: i32, offset: i32) -> Vec<i32> {
    let mask = (1 << position_bits) - 1;
    vec![(index & mask) + offset + (((index >> position_bits) & 1) * 16)]
}

fn pulse_2(index: i32, position_bits: i32, offset: i32) -> Vec<i32> {
    let mask = (1 << position_bits) - 1;
    let mut first = ((index >> position_bits) & mask) + offset;
    let mut second = (index & mask) + offset;
    let sign = (index >> (2 * position_bits)) & 1;
    if second < first {
        if sign == 1 {
            first += 16
        } else {
            second += 16
        }
    } else if sign == 1 {
        first += 16;
        second += 16;
    }
    vec![first, second]
}

fn pulse_3(index: i32, position_bits: i32, offset: i32) -> Vec<i32> {
    let mask = (1 << (2 * position_bits - 1)) - 1;
    let half = offset + (((index >> (2 * position_bits - 1)) & 1) << (position_bits - 1));
    let mut output = pulse_2(index & mask, position_bits - 1, half);
    output.extend(pulse_1(
        (index >> (2 * position_bits)) & ((1 << (position_bits + 1)) - 1),
        position_bits,
        offset,
    ));
    output
}

fn pulse_4_plus(index: i32, position_bits: i32, offset: i32) -> Vec<i32> {
    let mask = (1 << (2 * position_bits - 1)) - 1;
    let half = offset + (((index >> (2 * position_bits - 1)) & 1) << (position_bits - 1));
    let mut output = pulse_2(index & mask, position_bits - 1, half);
    output.extend(pulse_2(
        (index >> (2 * position_bits)) & ((1 << (2 * position_bits + 1)) - 1),
        position_bits,
        offset,
    ));
    output
}

fn pulse_4(index: i32, position_bits: i32, offset: i32) -> Vec<i32> {
    let reduced = position_bits - 1;
    let upper = offset + (1 << reduced);
    match (index >> (4 * position_bits - 2)) & 3 {
        0 => pulse_4_plus(
            index,
            reduced,
            if (index >> (4 * reduced + 1)) & 1 == 0 {
                offset
            } else {
                upper
            },
        ),
        1 => {
            let mut out = pulse_1(index >> (3 * reduced + 1), reduced, offset);
            out.extend(pulse_3(index, reduced, upper));
            out
        }
        2 => {
            let mut out = pulse_2(index >> (2 * reduced + 1), reduced, offset);
            out.extend(pulse_2(index, reduced, upper));
            out
        }
        _ => {
            let mut out = pulse_3(index >> (reduced + 1), reduced, offset);
            out.extend(pulse_1(index, reduced, upper));
            out
        }
    }
}

fn add_pulses(code: &mut [f32; 64], track: usize, pulses: &[i32]) {
    for &pulse in pulses {
        let position = ((pulse & 15) as usize) * 4 + track;
        code[position] += if pulse & 16 == 0 { 1.0 } else { -1.0 };
    }
}

pub fn decode_4t64(indices: &[u16], bits: u8) -> Result<[f32; 64], AcelpError> {
    let mut code = [0.0; 64];
    match bits {
        12 => {
            for pair in 0..2 {
                add_pulses(
                    &mut code,
                    usize::from(indices[pair * 2]) * 2 + pair,
                    &pulse_1(i32::from(indices[pair * 2 + 1]), 4, 0),
                );
            }
        }
        16 => {
            let omitted = if indices[0] == 0 { 1 } else { 3 };
            let mut source = 1;
            for track in 0..4 {
                if track != omitted {
                    add_pulses(&mut code, track, &pulse_1(i32::from(indices[source]), 4, 0));
                    source += 1;
                }
            }
        }
        20 => {
            for track in 0..4 {
                add_pulses(&mut code, track, &pulse_1(i32::from(indices[track]), 4, 0));
            }
        }
        28 => {
            for track in 0..4 {
                let pulses = if track < 2 {
                    pulse_2(i32::from(indices[track]), 4, 0)
                } else {
                    pulse_1(i32::from(indices[track]), 4, 0)
                };
                add_pulses(&mut code, track, &pulses);
            }
        }
        36 => {
            for track in 0..4 {
                add_pulses(&mut code, track, &pulse_2(i32::from(indices[track]), 4, 0));
            }
        }
        44 => {
            for track in 0..4 {
                let pulses = if track < 2 {
                    pulse_3(i32::from(indices[track]), 4, 0)
                } else {
                    pulse_2(i32::from(indices[track]), 4, 0)
                };
                add_pulses(&mut code, track, &pulses);
            }
        }
        52 => {
            for track in 0..4 {
                add_pulses(&mut code, track, &pulse_3(i32::from(indices[track]), 4, 0));
            }
        }
        64 => {
            for track in 0..4 {
                add_pulses(
                    &mut code,
                    track,
                    &pulse_4(
                        (i32::from(indices[track]) << 14) + i32::from(indices[track + 4]),
                        4,
                        0,
                    ),
                );
            }
        }
        _ => return Err(AcelpError::InvalidInnovativeCodebookBits(bits)),
    }
    Ok(code)
}

/// Generate 64 adaptive-codebook samples using FDK's 32-tap quarter-sample
/// pitch interpolation. Generated samples feed subsequent samples for short
/// pitch lags, matching the reference in-place implementation.
pub fn interpolate_pitch(
    history: &[f32],
    lag: usize,
    fraction_quarters: u8,
) -> Result<[f32; 64], AcelpError> {
    if lag < 16 || history.len() < lag + 16 || fraction_quarters > 3 {
        return Err(AcelpError::InvalidPitchHistory {
            lag,
            history: history.len(),
        });
    }
    let (phase, extra) = if fraction_quarters == 0 {
        (3usize, 0usize)
    } else {
        (usize::from(fraction_quarters - 1), 1usize)
    };
    let filter = &PITCH_FILTER[phase * 32..phase * 32 + 32];
    let mut work = history.to_vec();
    let mut output = [0.0; 64];
    for sample in 0..64 {
        let center = work.len() - lag - 15 - extra;
        let value = filter
            .iter()
            .enumerate()
            .map(|(tap, &coefficient)| work[center + tap] * coefficient)
            .sum();
        output[sample] = value;
        work.push(value);
    }
    Ok(output)
}

pub fn pitch_postfilter(excitation: &mut [f32; 64], previous: f32, following: f32) {
    let original = *excitation;
    for i in 0..64 {
        let left = if i == 0 { previous } else { original[i - 1] };
        let right = if i == 63 { following } else { original[i + 1] };
        excitation[i] = 0.18 * left + 0.64 * original[i] + 0.18 * right;
    }
}

#[derive(Debug, Clone)]
pub struct AcelpDecoder {
    excitation_history: Vec<f32>,
    synthesis_history: [f32; 16],
    deemphasis_memory: f32,
}

impl Default for AcelpDecoder {
    fn default() -> Self {
        Self {
            excitation_history: vec![0.0; 428],
            synthesis_history: [0.0; 16],
            deemphasis_memory: 0.0,
        }
    }
}

impl AcelpDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn decode_subframe(
        &mut self,
        subframe: &AcelpSubframe,
        core_mode: u8,
        mean_energy_index: u8,
        lpc: &[f32; 16],
    ) -> Result<[f32; 64], AcelpError> {
        let mut adaptive = interpolate_pitch(
            &self.excitation_history,
            usize::from(subframe.pitch_lag),
            subframe.pitch_fraction_quarters,
        )?;
        if !subframe.ltp_filtering {
            let previous = *self.excitation_history.last().unwrap();
            let following = adaptive[63];
            pitch_postfilter(&mut adaptive, previous, following);
        }
        let mut code = subframe.innovative_code(core_mode)?;
        for i in (1..64).rev() {
            code[i] -= 0.3 * code[i - 1];
        }
        let sharpened_lag =
            usize::from(subframe.pitch_lag) + usize::from(subframe.pitch_fraction_quarters > 2);
        if sharpened_lag < 64 {
            for i in sharpened_lag..64 {
                code[i] += 0.85 * code[i - sharpened_lag];
            }
        }
        let gains = subframe.decode_gains(mean_energy_index, &code);
        let excitation: [f32; 64] =
            std::array::from_fn(|i| gains.pitch * adaptive[i] + gains.innovative * code[i]);
        self.excitation_history.extend_from_slice(&excitation);
        let drain = self.excitation_history.len().saturating_sub(428);
        self.excitation_history.drain(..drain);

        let mut synthesis = [0.0; 64];
        for i in 0..64 {
            let prediction = (0..16)
                .map(|tap| {
                    let past = if i > tap {
                        synthesis[i - tap - 1]
                    } else {
                        self.synthesis_history[16 + i - tap - 1]
                    };
                    lpc[tap] * past
                })
                .sum::<f32>();
            synthesis[i] = excitation[i] - prediction;
        }
        self.synthesis_history.copy_from_slice(&synthesis[48..]);
        Ok(synthesis)
    }

    pub fn decode_frame(
        &mut self,
        frame: &AcelpFrame,
        old_lsp: &[f32; 16],
        new_lsp: &[f32; 16],
    ) -> Result<Vec<f32>, AcelpError> {
        let count = frame.subframes.len();
        let factors: &[f32] = match count {
            4 => &[0.125, 0.375, 0.625, 0.875],
            3 => &[1.0 / 6.0, 0.5, 5.0 / 6.0],
            _ => {
                return Err(AcelpError::SubframeCountMismatch {
                    expected: if count < 4 { 3 } else { 4 },
                    actual: count,
                });
            }
        };
        let mut output = Vec::with_capacity(count * 64);
        for (subframe, &factor) in frame.subframes.iter().zip(factors) {
            let interpolated_lsp =
                std::array::from_fn(|i| old_lsp[i] * (1.0 - factor) + new_lsp[i] * factor);
            let lpc = crate::usac_lpc::lsp_to_lpc(&interpolated_lsp);
            let synthesis =
                self.decode_subframe(subframe, frame.core_mode, frame.mean_energy_index, &lpc)?;
            for sample in synthesis {
                let value = sample + 0.68 * self.deemphasis_memory;
                self.deemphasis_memory = value;
                output.push(value);
            }
        }
        Ok(output)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcelpFrame {
    pub core_mode: u8,
    pub mean_energy_index: u8,
    pub subframes: Vec<AcelpSubframe>,
    pub bits_read: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AcelpError {
    Bit(BitError),
    InvalidCoreMode(u8),
    InvalidInnovativeCodebookBits(u8),
    InvalidFrameLength(usize),
    PitchOutOfRange {
        lag: i32,
        minimum: i32,
        maximum: i32,
    },
    InvalidPitchHistory {
        lag: usize,
        history: usize,
    },
    SubframeCountMismatch {
        expected: usize,
        actual: usize,
    },
}

impl From<BitError> for AcelpError {
    fn from(value: BitError) -> Self {
        Self::Bit(value)
    }
}

impl AcelpFrame {
    pub fn parse(
        reader: &mut BitReader<'_>,
        core_mode: u8,
        core_frame_length: usize,
        pitch_offset: i32,
    ) -> Result<Self, AcelpError> {
        let innovative_bits = *CORE_MODE_BITS
            .get(usize::from(core_mode))
            .ok_or(AcelpError::InvalidCoreMode(core_mode))?;
        let pitch_bits: &[u8] = match core_frame_length {
            1024 => &PITCH_BITS_1024,
            768 => &PITCH_BITS_768,
            _ => return Err(AcelpError::InvalidFrameLength(core_frame_length)),
        };
        let start = reader.bits_read();
        let mean_energy_index = reader.read_u8(2)?;
        let minimum = 34 + pitch_offset;
        let fractional_half_start = 128 - pitch_offset;
        let integer_start = 160;
        let maximum = 231 + 6 * pitch_offset;
        let mut relative_minimum = 0;
        let mut subframes = Vec::with_capacity(pitch_bits.len());
        for &bits in pitch_bits {
            let index = reader.read_u16(usize::from(bits))? as i32;
            let (lag, fraction) = if bits == 6 {
                (relative_minimum + index / 4, (index & 3) as u8)
            } else if index < (fractional_half_start - minimum) * 4 {
                (minimum + index / 4, (index & 3) as u8)
            } else if index
                < (fractional_half_start - minimum) * 4
                    + (integer_start - fractional_half_start) * 2
            {
                let adjusted = index - (fractional_half_start - minimum) * 4;
                (
                    fractional_half_start + adjusted / 2,
                    ((adjusted & 1) * 2) as u8,
                )
            } else {
                (
                    index + integer_start
                        - (fractional_half_start - minimum) * 4
                        - (integer_start - fractional_half_start) * 2,
                    0,
                )
            };
            if lag < minimum || lag > maximum {
                return Err(AcelpError::PitchOutOfRange {
                    lag,
                    minimum,
                    maximum,
                });
            }
            if bits == 9 {
                relative_minimum = (lag - 8).max(minimum);
                let relative_maximum = (relative_minimum + 15).min(maximum);
                relative_minimum = relative_maximum - 15;
            }
            let ltp_filtering = reader.read_bool()?;
            let innovative_indices = read_innovative_indices(reader, innovative_bits)?;
            let gain_index = reader.read_u8(7)?;
            subframes.push(AcelpSubframe {
                pitch_lag: lag as u16,
                pitch_fraction_quarters: fraction,
                ltp_filtering,
                innovative_indices,
                gain_index,
            });
        }
        Ok(Self {
            core_mode,
            mean_energy_index,
            subframes,
            bits_read: reader.bits_read() - start,
        })
    }
}

fn read_innovative_indices(
    reader: &mut BitReader<'_>,
    total_bits: u8,
) -> Result<Vec<u16>, AcelpError> {
    let widths: &[u8] = match total_bits {
        12 => &[1, 5, 1, 5],
        16 => &[1, 5, 5, 5],
        20 => &[5, 5, 5, 5],
        28 => &[9, 9, 5, 5],
        36 => &[9, 9, 9, 9],
        44 => &[13, 13, 9, 9],
        52 => &[13, 13, 13, 13],
        64 => &[2, 2, 2, 2, 14, 14, 14, 14],
        _ => return Err(AcelpError::InvalidInnovativeCodebookBits(total_bits)),
    };
    widths
        .iter()
        .map(|&width| reader.read_u16(usize::from(width)).map_err(AcelpError::Bit))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitWriter;

    #[test]
    fn parses_four_subframe_low_rate_acelp() {
        let mut bits = BitWriter::new();
        bits.write(2, 2);
        for (subframe, pitch_bits) in PITCH_BITS_1024.into_iter().enumerate() {
            bits.write(0, pitch_bits as usize);
            bits.write_bool(subframe & 1 != 0);
            for width in [5, 5, 5, 5] {
                bits.write(0, width);
            }
            bits.write(10, 7);
        }
        let frame = AcelpFrame::parse(&mut BitReader::new(&bits.finish()), 0, 1024, 0).unwrap();
        assert_eq!(frame.mean_energy_index, 2);
        assert_eq!(frame.subframes.len(), 4);
        assert_eq!(frame.subframes[0].pitch_lag, 34);
        assert_eq!(frame.subframes[1].pitch_lag, 34);
        assert!(frame.subframes[1].ltp_filtering);
        assert_eq!(frame.subframes[0].innovative_indices.len(), 4);
    }

    #[test]
    fn parses_three_subframe_64_bit_codebook() {
        let mut bits = BitWriter::new();
        bits.write(0, 2);
        for pitch_bits in PITCH_BITS_768 {
            bits.write(0, pitch_bits as usize);
            bits.write_bool(false);
            bits.write(0, 32);
            bits.write(0, 32);
            bits.write(0, 7);
        }
        let frame = AcelpFrame::parse(&mut BitReader::new(&bits.finish()), 5, 768, 0).unwrap();
        assert_eq!(frame.subframes.len(), 3);
        assert_eq!(frame.subframes[0].innovative_indices.len(), 8);
    }

    #[test]
    fn rejects_invalid_mode_and_length() {
        assert_eq!(
            AcelpFrame::parse(&mut BitReader::new(&[]), 8, 1024, 0),
            Err(AcelpError::InvalidCoreMode(8))
        );
        assert_eq!(
            AcelpFrame::parse(&mut BitReader::new(&[]), 0, 512, 0),
            Err(AcelpError::InvalidFrameLength(512))
        );
    }

    #[test]
    fn decodes_pitch_and_innovative_gain_rom() {
        let subframe = AcelpSubframe {
            pitch_lag: 34,
            pitch_fraction_quarters: 0,
            ltp_filtering: false,
            innovative_indices: vec![0; 4],
            gain_index: 0,
        };
        let gains = subframe.decode_gains(0, &[1.0; 64]);
        assert!((gains.pitch - 204.0 / 16384.0).abs() < 1e-7);
        assert!(gains.innovative > 0.0);
        assert_eq!(gains.innovation_energy, 64.0);
    }

    #[test]
    fn decodes_all_4t64_codebook_rates() {
        for (bits, index_count, pulses) in [
            (12, 4, 2),
            (16, 4, 3),
            (20, 4, 4),
            (28, 4, 6),
            (36, 4, 8),
            (44, 4, 10),
            (52, 4, 12),
            (64, 8, 16),
        ] {
            let code = decode_4t64(&vec![0; index_count], bits).unwrap();
            assert_eq!(
                code.iter().map(|value| value.abs()).sum::<f32>(),
                pulses as f32
            );
            assert!(code
                .iter()
                .enumerate()
                .all(|(i, value)| *value == 0.0 || i < 4));
        }
    }

    #[test]
    fn pulse_sign_bit_creates_negative_innovation() {
        let code = decode_4t64(&[16, 16, 16, 16], 20).unwrap();
        assert_eq!(code[0], -1.0);
        assert_eq!(code[1], -1.0);
        assert_eq!(code[2], -1.0);
        assert_eq!(code[3], -1.0);
    }

    #[test]
    fn quarter_sample_pitch_interpolation_preserves_constant_signal() {
        for fraction in 0..4 {
            let output = interpolate_pitch(&vec![1.0; 300], 80, fraction).unwrap();
            assert!(output.iter().all(|value| (*value - 1.0).abs() < 2e-3));
        }
    }

    #[test]
    fn pitch_postfilter_has_unity_dc_gain() {
        let mut excitation = [1.0; 64];
        pitch_postfilter(&mut excitation, 1.0, 1.0);
        assert!(excitation.iter().all(|value| (*value - 1.0).abs() < 1e-6));
    }

    #[test]
    fn stateful_acelp_decoder_synthesizes_subframe() {
        let subframe = AcelpSubframe {
            pitch_lag: 64,
            pitch_fraction_quarters: 0,
            ltp_filtering: true,
            innovative_indices: vec![0; 4],
            gain_index: 20,
        };
        let output = AcelpDecoder::new()
            .decode_subframe(&subframe, 0, 0, &[0.0; 16])
            .unwrap();
        assert!(output.iter().all(|value| value.is_finite()));
        assert!(output.iter().any(|value| *value != 0.0));
    }

    #[test]
    fn decodes_complete_frame_with_lsp_interpolation_and_deemphasis() {
        let subframe = AcelpSubframe {
            pitch_lag: 64,
            pitch_fraction_quarters: 0,
            ltp_filtering: true,
            innovative_indices: vec![0; 4],
            gain_index: 20,
        };
        let frame = AcelpFrame {
            core_mode: 0,
            mean_energy_index: 0,
            subframes: vec![subframe; 4],
            bits_read: 0,
        };
        let old = crate::usac_lpc::lsf_to_lsp(&std::array::from_fn(|i| 300.0 + i as f32 * 350.0));
        let new = crate::usac_lpc::lsf_to_lsp(&std::array::from_fn(|i| 350.0 + i as f32 * 350.0));
        let output = AcelpDecoder::new()
            .decode_frame(&frame, &old, &new)
            .unwrap();
        assert_eq!(output.len(), 256);
        assert!(output.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn pulse_decomposition_covers_all_sign_and_partition_branches() {
        assert_eq!(pulse_2(337, 4, 0), vec![21, 1]); // second < first, first negative
        assert_eq!(pulse_2(81, 4, 0), vec![5, 17]); // second < first, second negative
        assert_eq!(pulse_2(257, 4, 0), vec![16, 17]); // ordered, both negative

        for selector in 0..4i32 {
            let pulses = pulse_4(selector << 14, 4, 0);
            assert_eq!(pulses.len(), 4);
        }
        let upper_half = pulse_4(1 << 13, 4, 0);
        assert_eq!(upper_half.len(), 4);

        let indices = [0, 1, 2, 3, 0, 0, 0, 0];
        let code = decode_4t64(&indices, 64).unwrap();
        assert_eq!(code.iter().map(|value| value.abs()).sum::<f32>(), 16.0);
    }

    #[test]
    fn validates_pitch_history_and_core_mode() {
        for (history, lag, fraction) in [(31usize, 16usize, 0u8), (64, 15, 0), (64, 16, 4)] {
            assert!(matches!(
                interpolate_pitch(&vec![0.0; history], lag, fraction),
                Err(AcelpError::InvalidPitchHistory { .. })
            ));
        }
        let subframe = AcelpSubframe {
            pitch_lag: 64,
            pitch_fraction_quarters: 0,
            ltp_filtering: true,
            innovative_indices: vec![0; 4],
            gain_index: 0,
        };
        assert_eq!(
            subframe.innovative_code(8),
            Err(AcelpError::InvalidCoreMode(8))
        );
        let invalid_pitch = AcelpSubframe {
            pitch_lag: 15,
            ..subframe.clone()
        };
        assert!(matches!(
            AcelpDecoder::new().decode_subframe(&invalid_pitch, 0, 0, &[0.0; 16]),
            Err(AcelpError::InvalidPitchHistory { .. })
        ));
        let bit = BitError::UnexpectedEof {
            needed_bits: 1,
            remaining_bits: 0,
        };
        assert_eq!(AcelpError::from(bit.clone()), AcelpError::Bit(bit));
    }

    #[test]
    fn parses_all_innovative_index_width_layouts() {
        for (bits, count) in [
            (12, 4),
            (16, 4),
            (20, 4),
            (28, 4),
            (36, 4),
            (44, 4),
            (52, 4),
            (64, 8),
        ] {
            let values = read_innovative_indices(&mut BitReader::new(&[0; 8]), bits).unwrap();
            assert_eq!(values, vec![0; count]);
        }
        assert_eq!(
            read_innovative_indices(&mut BitReader::new(&[]), 13),
            Err(AcelpError::InvalidInnovativeCodebookBits(13))
        );
        assert_eq!(
            decode_4t64(&[], 13),
            Err(AcelpError::InvalidInnovativeCodebookBits(13))
        );
    }

    #[test]
    fn parses_half_sample_and_integer_pitch_regions() {
        let mut bits = BitWriter::new();
        bits.write(0, 2);
        for (pitch_bits, index) in PITCH_BITS_1024.into_iter().zip([376u32, 0, 440, 0]) {
            bits.write(index, pitch_bits as usize);
            bits.write_bool(false);
            bits.write(0, 20);
            bits.write(0, 7);
        }
        let frame = AcelpFrame::parse(&mut BitReader::new(&bits.finish()), 0, 1024, 0).unwrap();
        assert_eq!(frame.subframes[0].pitch_lag, 128);
        assert_eq!(frame.subframes[0].pitch_fraction_quarters, 0);
        assert_eq!(frame.subframes[2].pitch_lag, 160);
        assert_eq!(frame.subframes[2].pitch_fraction_quarters, 0);

        let mut invalid = BitWriter::new();
        invalid.write(0, 2);
        invalid.write(511, 9);
        assert!(matches!(
            AcelpFrame::parse(&mut BitReader::new(&invalid.finish()), 0, 1024, -20),
            Err(AcelpError::PitchOutOfRange { .. })
        ));
    }

    #[test]
    fn decoder_covers_postfilter_sharpening_and_three_subframes() {
        let subframe = AcelpSubframe {
            pitch_lag: 34,
            pitch_fraction_quarters: 3,
            ltp_filtering: false,
            innovative_indices: vec![0; 4],
            gain_index: 20,
        };
        let output = AcelpDecoder::new()
            .decode_subframe(&subframe, 0, 0, &[0.0; 16])
            .unwrap();
        assert!(output.iter().all(|value| value.is_finite()));

        let frame = AcelpFrame {
            core_mode: 0,
            mean_energy_index: 0,
            subframes: vec![subframe.clone(); 3],
            bits_read: 0,
        };
        let lsp = crate::usac_lpc::lsf_to_lsp(&std::array::from_fn(|i| 300.0 + i as f32 * 350.0));
        assert_eq!(
            AcelpDecoder::new()
                .decode_frame(&frame, &lsp, &lsp)
                .unwrap()
                .len(),
            192
        );
        for count in [0, 2, 5] {
            let mut invalid = frame.clone();
            invalid.subframes.resize(count, subframe.clone());
            assert!(matches!(
                AcelpDecoder::new().decode_frame(&invalid, &lsp, &lsp),
                Err(AcelpError::SubframeCountMismatch { actual, .. }) if actual == count
            ));
        }
    }
}
