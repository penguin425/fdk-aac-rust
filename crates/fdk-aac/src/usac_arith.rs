//! USAC context-adaptive arithmetic spectral decoder.
//!
//! The probability model is read from the bundled FDK source so the Rust port
//! cannot silently drift from the reference ROM while the migration is in
//! progress.

use std::sync::LazyLock;

use crate::bits::{BitError, BitReader};

const FDK_ARITH_SOURCE: &str = include_str!(concat!(
    env!("FDK_AAC_UPSTREAM_DIR"),
    "/libArithCoding/src/ac_arith_coder.cpp"
));
const MAX_LINES: usize = 1024;
const VALUE_ESCAPE: usize = 16;

static HASH: LazyLock<Vec<u32>> = LazyLock::new(|| parse_table("ari_merged_hash_ps", 742));
static PROBABILITIES: LazyLock<Vec<u16>> = LazyLock::new(|| {
    parse_table("ari_pk", 64 * 17)
        .into_iter()
        .map(|value| value as u16)
        .collect()
});

fn parse_table(name: &str, expected: usize) -> Vec<u32> {
    let marker = format!("{name}[");
    let start = FDK_ARITH_SOURCE.find(&marker).expect("FDK arithmetic ROM");
    let body = &FDK_ARITH_SOURCE[start..];
    let body = &body[body.find('{').unwrap() + 1..body.find("};").unwrap()];
    let values: Vec<_> = body
        .split(|c: char| !(c.is_ascii_hexdigit() || c == 'x' || c == 'X'))
        .filter(|word| !word.is_empty())
        .map(|word| {
            if let Some(hex) = word.strip_prefix("0x").or_else(|| word.strip_prefix("0X")) {
                u32::from_str_radix(hex, 16).unwrap()
            } else {
                word.parse().unwrap()
            }
        })
        .collect();
    assert_eq!(values.len(), expected, "unexpected {name} ROM size");
    values
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UsacArithmeticError {
    Bit(BitError),
    InvalidLineCount { lines: usize, maximum: usize },
    MissingPreviousContext,
    EscapeOverflow,
}

impl From<BitError> for UsacArithmeticError {
    fn from(value: BitError) -> Self {
        Self::Bit(value)
    }
}

#[derive(Debug, Clone)]
pub struct UsacArithmeticDecoder {
    previous_lines: usize,
    previous_context: [u8; MAX_LINES / 2 + 4],
}

impl Default for UsacArithmeticDecoder {
    fn default() -> Self {
        Self {
            previous_lines: 0,
            previous_context: [0; MAX_LINES / 2 + 4],
        }
    }
}

impl UsacArithmeticDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Decodes `lines` coefficients and preserves the model for the next frame.
    pub fn decode(
        &mut self,
        reader: &mut BitReader<'_>,
        lines: usize,
        maximum_lines: usize,
        reset: bool,
    ) -> Result<Vec<i32>, UsacArithmeticError> {
        if lines > maximum_lines
            || maximum_lines > MAX_LINES
            || maximum_lines & 1 != 0
            || lines & 1 != 0
        {
            return Err(UsacArithmeticError::InvalidLineCount {
                lines,
                maximum: maximum_lines,
            });
        }
        let pairs = maximum_lines / 2;
        if reset {
            self.previous_context[..pairs + 4].fill(0);
        } else if maximum_lines != self.previous_lines {
            if self.previous_lines == 0 {
                return Err(UsacArithmeticError::MissingPreviousContext);
            }
            resize_context(&mut self.previous_context, self.previous_lines / 2, pairs);
        }
        self.previous_lines = maximum_lines;
        let mut spectrum = vec![0; maximum_lines];
        if lines == 0 {
            self.previous_context[2..2 + pairs].fill(1);
        } else {
            decode_pairs(
                reader,
                &mut self.previous_context[2..],
                &mut spectrum,
                lines / 2,
                pairs,
            )?;
        }
        Ok(spectrum)
    }

    /// Decode USAC `ac_spectral_data()` for one long window or eight short
    /// windows. Reset is implicit for independent frames and is signalled by
    /// one bit otherwise; it applies only to the first window.
    pub fn decode_windows(
        &mut self,
        reader: &mut BitReader<'_>,
        transmitted_lines: usize,
        frame_length: usize,
        short_windows: bool,
        independent: bool,
    ) -> Result<Vec<Vec<i32>>, UsacArithmeticError> {
        let reset = independent || reader.read_bool()?;
        let window_count = if short_windows { 8 } else { 1 };
        if frame_length % window_count != 0 {
            return Err(UsacArithmeticError::InvalidLineCount {
                lines: transmitted_lines,
                maximum: frame_length,
            });
        }
        let window_length = frame_length / window_count;
        let mut windows = Vec::with_capacity(window_count);
        for window in 0..window_count {
            windows.push(self.decode(
                reader,
                transmitted_lines,
                window_length,
                reset && window == 0,
            )?);
        }
        Ok(windows)
    }
}

fn resize_context(context: &mut [u8], input: usize, output: usize) {
    let table = &mut context[2..];
    let factor = if input < output {
        if input < output / 4 {
            8
        } else if input == output / 4 {
            4
        } else {
            2
        }
    } else if output < input / 4 {
        8
    } else if output == input / 4 {
        4
    } else {
        2
    };
    if input < output {
        table[output] = table[input];
        table[output + 1] = table[input + 1];
        let mut out = output;
        for source in (0..input).rev() {
            for _ in 0..factor {
                out -= 1;
                table[out] = table[source];
            }
        }
    } else {
        for out in 0..output {
            table[out] = table[out * factor];
        }
        table[output] = table[input];
        table[output + 1] = table[input + 1];
    }
}

#[derive(Clone, Copy)]
struct RangeState {
    low: u16,
    high: u16,
    value: u16,
}

fn decode_symbol(
    reader: &mut BitReader<'_>,
    state: &mut RangeState,
    frequencies: &[u16],
) -> Result<usize, BitError> {
    let range = u32::from(state.high) - u32::from(state.low) + 1;
    let scaled = ((u32::from(state.value) - u32::from(state.low) + 1) << 14) - 1;
    let mut symbol = 0;
    while symbol + 1 < frequencies.len() && u32::from(frequencies[symbol]) * range > scaled {
        symbol += 1;
    }
    let mut high = u32::from(state.high);
    if symbol != 0 {
        high = u32::from(state.low) + ((range * u32::from(frequencies[symbol - 1])) >> 14) - 1;
    }
    let mut low = u32::from(state.low) + ((range * u32::from(frequencies[symbol])) >> 14);
    let mut value = u32::from(state.value);
    loop {
        if high & 0x8000 != 0 {
            if low & 0x8000 == 0 {
                if low & 0x4000 != 0 && high & 0x4000 == 0 {
                    low -= 0x4000;
                    high -= 0x4000;
                    value -= 0x4000;
                } else {
                    break;
                }
            }
        }
        low = (low << 1) & 0xffff;
        high = ((high << 1) | 1) & 0xffff;
        value = ((value << 1) | u32::from(reader.read_bool()?)) & 0xffff;
    }
    state.low = low as u16;
    state.high = high as u16;
    state.value = value as u16;
    Ok(symbol)
}

fn probability_index(context: u32) -> usize {
    let table = &*HASH;
    let key = context.max(1).wrapping_shl(12).wrapping_sub(1);
    let mut base = if key > table[485] {
        486
    } else if key > table[255] {
        256
    } else {
        0
    };
    for step in [128, 64, 32, 16, 8, 4, 2] {
        if key > table[base + step - 1] {
            base += step;
        }
    }
    let mut packed = table[base];
    if key > packed {
        packed = table[base + 1];
    }
    if context != packed >> 12 {
        packed >>= 6;
    }
    (packed & 0x3f) as usize
}

fn decode_pairs(
    reader: &mut BitReader<'_>,
    previous: &mut [u8],
    spectrum: &mut [i32],
    pair_count: usize,
    maximum_pairs: usize,
) -> Result<(), UsacArithmeticError> {
    let mut state = RangeState {
        low: 0,
        high: 0xffff,
        value: reader.read_u16(16)?,
    };
    let mut histories = [0i32; 3];
    let mut state_increment = u32::from(previous[0]) << 12;
    let mut decoded = 0;
    while decoded < pair_count {
        let mut context = (state_increment >> 8) + (u32::from(previous[decoded + 1]) << 8);
        context = (context << 4) + histories[0] as u32;
        state_increment = context;
        if decoded > 3 && histories.iter().sum::<i32>() < 5 {
            context += 0x10000;
        }
        let mut level = 0;
        let mut escapes = 0usize;
        let symbol = loop {
            let index = probability_index(context + ((escapes as u32) << 17));
            let row = &PROBABILITIES[index * 17..index * 17 + 17];
            let symbol = decode_symbol(reader, &mut state, row)?;
            if symbol < VALUE_ESCAPE {
                break symbol;
            }
            level += 1;
            if level > 23 {
                return Err(UsacArithmeticError::EscapeOverflow);
            }
            escapes = (escapes + 1).min(7);
        };
        if symbol == 0 && escapes != 0 {
            break;
        }
        let (mut a, mut b) = if symbol == 0 {
            (0, 0)
        } else {
            ((symbol & 3) as i32, (symbol >> 2) as i32)
        };
        for _ in 0..level {
            let model = if a == 0 {
                [12661, 5700, 3751, 0]
            } else if b == 0 {
                [12571, 10569, 3696, 0]
            } else {
                [10827, 6884, 2929, 0]
            };
            let low = decode_symbol(reader, &mut state, &model)?;
            a = (a << 1) | (low as i32 & 1);
            b = (b << 1) | (low as i32 >> 1);
        }
        spectrum[decoded * 2] = a;
        spectrum[decoded * 2 + 1] = b;
        let amplitude = (a + b + 1).min(15);
        histories = [amplitude, histories[0], histories[1]];
        previous[decoded] = amplitude as u8;
        decoded += 1;
    }
    reader.push_back(14)?;
    for pair in 0..decoded {
        for coefficient in &mut spectrum[pair * 2..pair * 2 + 2] {
            if *coefficient != 0 && !reader.read_bool()? {
                *coefficient = -*coefficient;
            }
        }
    }
    previous[decoded..maximum_pairs].fill(1);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_reference_probability_rom() {
        assert_eq!(HASH.len(), 742);
        assert_eq!(PROBABILITIES.len(), 1088);
        assert_eq!(PROBABILITIES[16], 0);
    }

    #[test]
    fn reset_with_no_transmitted_lines_initializes_context() {
        let mut decoder = UsacArithmeticDecoder::new();
        assert_eq!(
            decoder
                .decode(&mut BitReader::new(&[]), 0, 1024, true)
                .unwrap(),
            vec![0; 1024]
        );
        assert!(decoder.previous_context[2..514]
            .iter()
            .all(|&value| value == 1));
    }

    #[test]
    fn rejects_invalid_sizes_and_missing_context() {
        let mut decoder = UsacArithmeticDecoder::new();
        assert!(matches!(
            decoder.decode(&mut BitReader::new(&[]), 3, 4, true),
            Err(UsacArithmeticError::InvalidLineCount { .. })
        ));
        assert_eq!(
            decoder.decode(&mut BitReader::new(&[]), 0, 256, false),
            Err(UsacArithmeticError::MissingPreviousContext)
        );
        for (lines, maximum) in [(6, 4), (0, 1025), (0, 3)] {
            assert!(matches!(
                decoder.decode(&mut BitReader::new(&[]), lines, maximum, true),
                Err(UsacArithmeticError::InvalidLineCount { .. })
            ));
        }
        assert!(matches!(
            decoder.decode(&mut BitReader::new(&[]), 2, 2, true),
            Err(UsacArithmeticError::Bit(BitError::UnexpectedEof { .. }))
        ));
        assert_eq!(
            decoder.decode(&mut BitReader::new(&vec![0; 4096]), 256, 256, true),
            Err(UsacArithmeticError::EscapeOverflow)
        );
    }

    #[test]
    fn window_wrapper_uses_implicit_or_explicit_reset_flag() {
        let mut independent = UsacArithmeticDecoder::new();
        assert_eq!(
            independent
                .decode_windows(&mut BitReader::new(&[]), 0, 1024, false, true)
                .unwrap()
                .len(),
            1
        );
        let mut dependent = UsacArithmeticDecoder::new();
        assert_eq!(
            dependent
                .decode_windows(&mut BitReader::new(&[0x80]), 0, 1024, false, false)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn decodes_eight_empty_short_windows_with_shared_context() {
        let mut decoder = UsacArithmeticDecoder::new();
        let windows = decoder
            .decode_windows(&mut BitReader::new(&[]), 0, 1024, true, true)
            .unwrap();
        assert_eq!(windows, vec![vec![0; 128]; 8]);
        assert_eq!(decoder.previous_lines, 128);
    }

    #[test]
    fn window_wrapper_reports_reset_and_layout_errors() {
        let mut decoder = UsacArithmeticDecoder::new();
        assert!(matches!(
            decoder.decode_windows(&mut BitReader::new(&[]), 0, 1024, false, false),
            Err(UsacArithmeticError::Bit(BitError::UnexpectedEof { .. }))
        ));
        assert!(matches!(
            decoder.decode_windows(&mut BitReader::new(&[]), 0, 1023, true, true),
            Err(UsacArithmeticError::InvalidLineCount { .. })
        ));
    }

    #[test]
    fn context_resize_covers_every_expansion_and_reduction_factor() {
        for (input, output) in [(4, 40), (4, 16), (6, 16), (40, 4), (16, 4), (16, 6)] {
            let mut context = [0u8; MAX_LINES / 2 + 4];
            for (index, value) in context[2..2 + input + 2].iter_mut().enumerate() {
                *value = index as u8 + 1;
            }
            let tail = [context[2 + input], context[3 + input]];
            resize_context(&mut context, input, output);
            assert_eq!([context[2 + output], context[3 + output]], tail);
            assert!(context[2..2 + output].iter().any(|&value| value != 0));
        }
    }

    #[test]
    fn probability_hash_lookup_stays_within_the_model_table() {
        for context in [0, 1, 2, 0x100, 0xffff, 0x1_0000, 0x20_0000, u32::MAX] {
            assert!(probability_index(context) < 64);
        }
    }

    #[test]
    fn deterministic_arithmetic_payloads_preserve_output_bounds() {
        let mut successes = 0;
        let mut bit_errors = 0;
        let mut nonzero = 0;
        for seed in 0..128u32 {
            let mut state = seed.wrapping_add(1);
            let mut bytes = vec![0u8; 96];
            for byte in &mut bytes {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *byte = (state >> 24) as u8;
            }
            if seed == 0 {
                bytes.clear();
            }
            let mut decoder = UsacArithmeticDecoder::new();
            let result = decoder.decode(&mut BitReader::new(&bytes), 32, 32, true);
            assert!(matches!(
                &result,
                Ok(_) | Err(UsacArithmeticError::Bit(_)) | Err(UsacArithmeticError::EscapeOverflow)
            ));
            if let Ok(spectrum) = result {
                successes += 1;
                assert_eq!(spectrum.len(), 32);
                assert!(spectrum
                    .iter()
                    .all(|value| value.unsigned_abs() < (1 << 27)));
                nonzero += usize::from(spectrum.iter().any(|&value| value != 0));
            } else {
                bit_errors += usize::from(matches!(result, Err(UsacArithmeticError::Bit(_))));
            }
        }
        assert!(successes > 0);
        assert!(nonzero > 0);
        assert!(successes + bit_errors <= 128);
    }

    #[test]
    fn dependent_frame_resizes_saved_context_in_both_directions() {
        let mut decoder = UsacArithmeticDecoder::new();
        decoder
            .decode(&mut BitReader::new(&[]), 0, 128, true)
            .unwrap();
        assert_eq!(
            decoder
                .decode(&mut BitReader::new(&[]), 0, 512, false)
                .unwrap()
                .len(),
            512
        );
        assert_eq!(
            decoder
                .decode(&mut BitReader::new(&[]), 0, 128, false)
                .unwrap()
                .len(),
            128
        );
    }
}
