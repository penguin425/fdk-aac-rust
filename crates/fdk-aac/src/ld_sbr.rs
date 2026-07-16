//! AAC-ELD low-delay SBR frame-grid syntax.
//!
//! This is the timing layer used before envelope/noise Huffman decoding.  It
//! follows `extractFrameInfo` and `extractLowDelayGrid` from FDK.

use std::{fmt, sync::LazyLock};

use crate::asc::LdSbrHeader;

const SBR_ROM_SOURCE: &str = include_str!(concat!(
    env!("FDK_AAC_UPSTREAM_DIR"),
    "/libSBRdec/src/sbr_rom.cpp"
));

// Convert FDK's pseudo-float reference energy into the normalized dual-rate
// QMF domain. Single-rate synthesis applies another four energy-scale bits in
// the channel processor.
const ENVELOPE_ENERGY_SCALE: f64 = 1.0 / 1_048_576.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SbrHuffmanBook {
    EnvelopeLevel15Time,
    EnvelopeLevel15Frequency,
    EnvelopeLevel30Time,
    EnvelopeLevel30Frequency,
    EnvelopeBalance15Time,
    EnvelopeBalance15Frequency,
    EnvelopeBalance30Time,
    EnvelopeBalance30Frequency,
    NoiseLevelTime,
    NoiseBalanceTime,
}

impl SbrHuffmanBook {
    fn table(self) -> &'static [[i8; 2]] {
        match self {
            Self::EnvelopeLevel15Time => &ENV_LEVEL_10_T,
            Self::EnvelopeLevel15Frequency => &ENV_LEVEL_10_F,
            Self::EnvelopeLevel30Time => &ENV_LEVEL_11_T,
            Self::EnvelopeLevel30Frequency => &ENV_LEVEL_11_F,
            Self::EnvelopeBalance15Time => &ENV_BALANCE_10_T,
            Self::EnvelopeBalance15Frequency => &ENV_BALANCE_10_F,
            Self::EnvelopeBalance30Time => &ENV_BALANCE_11_T,
            Self::EnvelopeBalance30Frequency => &ENV_BALANCE_11_F,
            Self::NoiseLevelTime => &NOISE_LEVEL_11_T,
            Self::NoiseBalanceTime => &NOISE_BALANCE_11_T,
        }
    }
}

pub fn decode_sbr_huffman(
    reader: &mut BitReader<'_>,
    book: SbrHuffmanBook,
) -> Result<i8, LdSbrError> {
    let table = book.table();
    let mut index = 0i8;
    while index >= 0 {
        let row = table
            .get(index as usize)
            .ok_or(LdSbrError::InvalidHuffmanCodeword)?;
        index = row[reader.read_bool()? as usize];
    }
    Ok(index + 64)
}

pub fn encode_sbr_huffman(book: SbrHuffmanBook, symbol: i8) -> Option<Vec<bool>> {
    fn find(table: &[[i8; 2]], node: i8, target: i8, bits: &mut Vec<bool>) -> bool {
        if node < 0 {
            return node + 64 == target;
        }
        let row = &table[node as usize];
        for (bit, &next) in row.iter().enumerate() {
            bits.push(bit != 0);
            if find(table, next, target, bits) {
                return true;
            }
            bits.pop();
        }
        false
    }

    let mut bits = Vec::new();
    find(book.table(), 0, symbol, &mut bits).then_some(bits)
}

fn parse_sbr_huffman_table(name: &str) -> Vec<[i8; 2]> {
    let declaration = format!("const SCHAR {name}");
    let start = SBR_ROM_SOURCE
        .find(&declaration)
        .unwrap_or_else(|| panic!("missing FDK SBR Huffman table {name}"));
    let body_start = SBR_ROM_SOURCE[start..].find('{').unwrap() + start + 1;
    let body_end = SBR_ROM_SOURCE[body_start..].find("};").unwrap() + body_start;
    let body = &SBR_ROM_SOURCE[body_start..body_end];
    let mut values = Vec::new();
    for pair in body.split('{').skip(1) {
        let Some(end) = pair.find('}') else { continue };
        let numbers = pair[..end]
            .split(',')
            .map(|value| value.trim().parse::<i8>().unwrap())
            .collect::<Vec<_>>();
        if numbers.len() == 2 {
            values.push([numbers[0], numbers[1]]);
        }
    }
    values
}

macro_rules! sbr_book {
    ($static_name:ident, $c_name:literal) => {
        static $static_name: LazyLock<Vec<[i8; 2]>> =
            LazyLock::new(|| parse_sbr_huffman_table($c_name));
    };
}

sbr_book!(ENV_LEVEL_10_T, "FDK_sbrDecoder_sbr_huffBook_EnvLevel10T");
sbr_book!(ENV_LEVEL_10_F, "FDK_sbrDecoder_sbr_huffBook_EnvLevel10F");
sbr_book!(ENV_LEVEL_11_T, "FDK_sbrDecoder_sbr_huffBook_EnvLevel11T");
sbr_book!(ENV_LEVEL_11_F, "FDK_sbrDecoder_sbr_huffBook_EnvLevel11F");
sbr_book!(
    ENV_BALANCE_10_T,
    "FDK_sbrDecoder_sbr_huffBook_EnvBalance10T"
);
sbr_book!(
    ENV_BALANCE_10_F,
    "FDK_sbrDecoder_sbr_huffBook_EnvBalance10F"
);
sbr_book!(
    ENV_BALANCE_11_T,
    "FDK_sbrDecoder_sbr_huffBook_EnvBalance11T"
);
sbr_book!(
    ENV_BALANCE_11_F,
    "FDK_sbrDecoder_sbr_huffBook_EnvBalance11F"
);
sbr_book!(
    NOISE_LEVEL_11_T,
    "FDK_sbrDecoder_sbr_huffBook_NoiseLevel11T"
);
sbr_book!(
    NOISE_BALANCE_11_T,
    "FDK_sbrDecoder_sbr_huffBook_NoiseBalance11T"
);
use crate::bits::{BitError, BitReader};
use crate::usac_sbr::InterTesEnvelope;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LdSbrFrequencyTables {
    pub master: Vec<u8>,
    pub high: Vec<u8>,
    pub low: Vec<u8>,
    pub noise: Vec<u8>,
}

impl LdSbrFrequencyTables {
    pub fn from_header(header: &LdSbrHeader, sampling_frequency: u32) -> Result<Self, LdSbrError> {
        let k0 = start_band(sampling_frequency, header.start_frequency)
            .ok_or(LdSbrError::UnsupportedSamplingFrequency(sampling_frequency))?;
        let k2 = stop_band(sampling_frequency, header.stop_frequency, k0)
            .ok_or(LdSbrError::InvalidFrequencyRange)?;
        let frequency_scale = header.frequency_scale.unwrap_or(2);
        let alter_scale = header.alter_scale.unwrap_or(true);
        let master = make_master(k0, k2, frequency_scale, alter_scale)?;
        let crossover = header.crossover_band as usize;
        if crossover >= master.len() {
            return Err(LdSbrError::InvalidCrossoverBand(header.crossover_band));
        }
        let high = master[crossover..].to_vec();
        let high_bands = high.len() - 1;
        let low = if high_bands % 2 == 0 {
            (0..=high_bands / 2).map(|index| high[index * 2]).collect()
        } else {
            let mut result = vec![high[0]];
            result.extend((1..=(high_bands + 1) / 2).map(|index| high[index * 2 - 1]));
            result
        };
        let noise_bands_per_octave = header.noise_bands.unwrap_or(2) as f64;
        let octaves = (k2 as f64 / high[0] as f64).log2();
        let noise_count = ((noise_bands_per_octave * octaves).round() as usize)
            .clamp(1, 5)
            .min(low.len() - 1);
        let noise = downsample_borders(&low, noise_count);
        Ok(Self {
            master,
            high,
            low,
            noise,
        })
    }

    pub fn high_band_count(&self) -> usize {
        self.high.len() - 1
    }

    pub fn low_band_count(&self) -> usize {
        self.low.len() - 1
    }

    pub fn noise_band_count(&self) -> usize {
        self.noise.len() - 1
    }
}

fn start_band(sampling_frequency: u32, index: u8) -> Option<u8> {
    let table: &[u8; 16] = match sampling_frequency {
        16_000 => &[
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        ],
        22_050 => &[
            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 28, 30,
        ],
        24_000 => &[
            11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 27, 29, 32,
        ],
        32_000 => &[
            10, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 27, 29, 32,
        ],
        40_000 => &[
            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 26, 28, 30, 32,
        ],
        44_100 => &[
            8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 25, 28, 32,
        ],
        48_000 => &[7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 24, 27, 31],
        64_000 => &[6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 21, 23, 26, 30],
        88_200 | 96_000 => &[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 23, 27, 31],
        _ => return None,
    };
    table.get(index as usize).copied()
}

fn stop_band(sampling_frequency: u32, index: u8, k0: u8) -> Option<u8> {
    let value = match index {
        0..=13 => {
            let minimum_hz = if sampling_frequency < 32_000 {
                6_000
            } else if sampling_frequency < 64_000 {
                8_000
            } else {
                10_000
            };
            let minimum =
                ((minimum_hz * 128 + sampling_frequency / 2) / sampling_frequency).min(64) as u8;
            let widths = logarithmic_widths(minimum, 64, 13).ok()?;
            let mut sorted = widths;
            sorted.sort_unstable();
            minimum + sorted[..index as usize].iter().copied().sum::<u8>()
        }
        14 => k0.saturating_mul(2).min(64),
        15 => k0.saturating_mul(3).min(64),
        _ => return None,
    };
    (value > k0 && value - k0 <= 56).then_some(value)
}

fn make_master(k0: u8, k2: u8, scale: u8, alter: bool) -> Result<Vec<u8>, LdSbrError> {
    if scale == 0 {
        let width = if alter { 2 } else { 1 };
        let count = if alter {
            ((((k2 - k0) as usize / 2) + 1) & !1).max(2)
        } else {
            ((k2 - k0) as usize & !1).max(2)
        };
        let mut widths = vec![width; count];
        let difference = k2 as isize - (k0 as isize + count as isize * width as isize);
        if difference < 0 {
            for item in widths.iter_mut().take((-difference) as usize) {
                *item -= 1;
            }
        } else {
            for item in widths.iter_mut().rev().take(difference as usize) {
                *item += 1;
            }
        }
        return Ok(cumulative(k0, &widths));
    }
    let bands_per_octave = match scale {
        1 => 12.0,
        2 => 10.0,
        _ => 8.0,
    };
    let split = 1000 * k2 as u32 > 2245 * k0 as u32;
    let k1 = if split { k0 * 2 } else { k2 };
    let mut first = logarithmic_widths_for_density(k0, k1, bands_per_octave, false)?;
    if !split {
        // FDK shell-sorts CalcBands' widths even when the master table has a
        // single logarithmic region.
        first.sort_unstable();
        return Ok(cumulative(k0, &first));
    }
    let mut second = logarithmic_widths_for_density(k1, k2, bands_per_octave, alter)?;
    first.sort_unstable();
    second.sort_unstable();
    if first.last().copied().unwrap() > second[0] {
        let change = (first.last().copied().unwrap() - second[0])
            .min((second.last().copied().unwrap() - second[0]) / 2);
        second[0] += change;
        *second.last_mut().unwrap() -= change;
        second.sort_unstable();
    }
    first.extend(second);
    Ok(cumulative(k0, &first))
}

fn logarithmic_widths_for_density(
    start: u8,
    stop: u8,
    density: f64,
    warp: bool,
) -> Result<Vec<u8>, LdSbrError> {
    // Bit-exact port of FDK's numberOfBands(). CalcLdInt stores log2(i) in
    // Q25, FDK_getNumOctavesDiv8 converts its difference to Q15, and both
    // density and warp multiplications truncate rather than round.
    let calc_ld_int = |value: u8| ((value as f64).log2() * (1u64 << 25) as f64).round() as i64;
    let octaves_div8 = (calc_ld_int(stop) - calc_ld_int(start)) >> 13;
    let bpo_div16 = (density / 16.0 * 32768.0).round() as i64;
    let mut bands_div128 = (octaves_div8 * bpo_div16 * 2) >> 16;
    if warp {
        bands_div128 = (bands_div128 * 25_200 * 2) >> 16;
    }
    bands_div128 += 256; // Q15 1/128, for rounding to an even band count.
    let count = (2 * (bands_div128 >> 9)) as usize;
    logarithmic_widths(start, stop, count)
}

fn logarithmic_widths(start: u8, stop: u8, count: usize) -> Result<Vec<u8>, LdSbrError> {
    if count == 0 || stop <= start {
        return Err(LdSbrError::InvalidFrequencyRange);
    }
    // Bit-exact port of FDK's Q31 `calcFactorPerBand` followed by its Q15
    // `CalcBands`. The fixed-point root search intentionally differs by one
    // QMF band from a direct floating-point geometric progression in several
    // valid ELD-SBR configurations.
    let mut factor = 1i64 << 29; // Q31 0.25
    let mut step = 1i64 << 28; // Q31 0.125
    let start_q31 = i64::from(start) << 24;
    let stop_q31 = i64::from(stop) << 24;
    let mut direction = true;
    let mut iterations = 0;
    while step > 0 && iterations <= 100 {
        iterations += 1;
        let mut value = stop_q31;
        for _ in 0..count {
            value = ((value * factor) >> 32) << 2;
        }
        if value < start_q31 {
            if !direction {
                step >>= 1;
            }
            direction = true;
            factor += step;
        } else {
            if direction {
                step >>= 1;
            }
            direction = false;
            factor -= step;
        }
    }
    let factor_q15 = ((factor << 1) >> 16).min(i64::from(i16::MAX));
    let mut widths = vec![0u8; count];
    let mut previous = i64::from(stop);
    let mut exact = i64::from(stop) << 8;
    for index in (0..count).rev() {
        exact = (exact * factor_q15 * 2) >> 16;
        let current = (exact + 128) >> 8;
        widths[index] = (previous - current) as u8;
        previous = current;
    }
    widths
        .iter()
        .all(|&width| width > 0)
        .then_some(widths)
        .ok_or(LdSbrError::InvalidFrequencyRange)
}

fn cumulative(start: u8, widths: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(widths.len() + 1);
    result.push(start);
    for &width in widths {
        result.push(result.last().copied().unwrap() + width);
    }
    result
}

fn downsample_borders(source: &[u8], count: usize) -> Vec<u8> {
    let mut result = vec![source[0]];
    let mut remaining_source = source.len() - 1;
    let mut remaining_result = count;
    let mut index = 0;
    while remaining_source > 0 {
        let step = remaining_source / remaining_result;
        index += step;
        result.push(source[index]);
        remaining_source -= step;
        remaining_result -= 1;
    }
    result
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LdSbrGrid {
    pub transient: bool,
    pub amp_resolution: Option<bool>,
    pub borders: Vec<u8>,
    pub frequency_resolution: Vec<bool>,
    pub transient_envelope: Option<usize>,
    pub noise_borders: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LdSbrChannelControl {
    pub grid: LdSbrGrid,
    pub envelope_time_domain: Vec<bool>,
    pub noise_time_domain: Vec<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LdSbrChannelElementPrefix {
    pub data_extra: Option<(u8, Option<u8>)>,
    pub coupling: bool,
    pub left: LdSbrChannelControl,
    pub right: Option<LdSbrChannelControl>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LdSbrChannelValues {
    pub inverse_filtering_modes: Vec<u8>,
    pub envelopes: Vec<Vec<i16>>,
    pub noise: Vec<Vec<i16>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LdSbrDequantizedChannel {
    pub envelope_energy: Vec<Vec<f64>>,
    pub noise_energy: Vec<Vec<f64>>,
}

impl LdSbrChannelValues {
    pub fn parse_mono_after_prefix(
        reader: &mut BitReader<'_>,
        control: &LdSbrChannelControl,
        tables: &LdSbrFrequencyTables,
        default_amp_resolution: bool,
    ) -> Result<Self, LdSbrError> {
        let inverse_filtering_modes = read_invf(reader, tables.noise_band_count())?;
        let envelopes = read_envelopes(reader, control, tables, default_amp_resolution, false)?;
        let noise = read_noise(reader, control, tables, false)?;
        Ok(Self {
            inverse_filtering_modes,
            envelopes,
            noise,
        })
    }

    pub fn parse_stereo_after_prefix(
        reader: &mut BitReader<'_>,
        prefix: &LdSbrChannelElementPrefix,
        tables: &LdSbrFrequencyTables,
        default_amp_resolution: bool,
    ) -> Result<(Self, Self), LdSbrError> {
        let right_control = prefix.right.as_ref().ok_or(LdSbrError::ExpectedStereo)?;
        let left_invf = read_invf(reader, tables.noise_band_count())?;
        let right_invf = if prefix.coupling {
            left_invf.clone()
        } else {
            read_invf(reader, tables.noise_band_count())?
        };
        let left_envelopes =
            read_envelopes(reader, &prefix.left, tables, default_amp_resolution, false)?;
        let (left_noise, right_envelopes, right_noise) = if prefix.coupling {
            let left_noise = read_noise(reader, &prefix.left, tables, false)?;
            let right_envelopes =
                read_envelopes(reader, right_control, tables, default_amp_resolution, true)?;
            let right_noise = read_noise(reader, right_control, tables, true)?;
            (left_noise, right_envelopes, right_noise)
        } else {
            let right_envelopes =
                read_envelopes(reader, right_control, tables, default_amp_resolution, false)?;
            let left_noise = read_noise(reader, &prefix.left, tables, false)?;
            let right_noise = read_noise(reader, right_control, tables, false)?;
            (left_noise, right_envelopes, right_noise)
        };
        Ok((
            Self {
                inverse_filtering_modes: left_invf,
                envelopes: left_envelopes,
                noise: left_noise,
            },
            Self {
                inverse_filtering_modes: right_invf,
                envelopes: right_envelopes,
                noise: right_noise,
            },
        ))
    }

    pub fn reconstruct_deltas(
        &mut self,
        control: &LdSbrChannelControl,
        tables: &LdSbrFrequencyTables,
        previous: &mut LdSbrPreviousValues,
    ) -> Result<(), LdSbrError> {
        let high_count = tables.high_band_count();
        if previous.envelope_high.len() != high_count {
            previous.envelope_high.resize(high_count, 0);
        }
        let offset = 2 * tables.low_band_count() as isize - high_count as isize;
        for (envelope_index, values) in self.envelopes.iter_mut().enumerate() {
            let high_resolution = control.grid.frequency_resolution[envelope_index];
            if control.envelope_time_domain[envelope_index] {
                for (band, value) in values.iter_mut().enumerate() {
                    *value += previous.envelope_high[low_to_high(offset, band, high_resolution)];
                    map_to_high(
                        *value,
                        &mut previous.envelope_high,
                        offset,
                        band,
                        high_resolution,
                    );
                }
            } else {
                for band in 1..values.len() {
                    values[band] += values[band - 1];
                }
                for (band, &value) in values.iter().enumerate() {
                    map_to_high(
                        value,
                        &mut previous.envelope_high,
                        offset,
                        band,
                        high_resolution,
                    );
                }
            }
        }
        let noise_count = tables.noise_band_count();
        if previous.noise.len() != noise_count {
            previous.noise.resize(noise_count, 0);
        }
        for index in 0..self.noise.len() {
            let (before, current_and_after) = self.noise.split_at_mut(index);
            let values = &mut current_and_after[0];
            if control.noise_time_domain[index] {
                let reference = if index == 0 {
                    &previous.noise
                } else {
                    &before[index - 1]
                };
                for (value, &prior) in values.iter_mut().zip(reference) {
                    *value += prior;
                }
            } else {
                for band in 1..values.len() {
                    values[band] += values[band - 1];
                }
            }
        }
        if let Some(last) = self.noise.last() {
            previous.noise.clone_from(last);
        }
        Ok(())
    }

    pub fn dequantize_uncoupled(
        &self,
        control: &LdSbrChannelControl,
        default_amp_resolution: bool,
    ) -> LdSbrDequantizedChannel {
        let envelope_energy = self
            .envelopes
            .iter()
            .map(|values| {
                let amp_resolution = control
                    .grid
                    .amp_resolution
                    .unwrap_or(default_amp_resolution);
                let divisor = if amp_resolution { 1.0 } else { 2.0 };
                values
                    .iter()
                    .map(|&value| {
                        64.0 * 2.0f64.powf(value as f64 / divisor) * ENVELOPE_ENERGY_SCALE
                    })
                    .collect()
            })
            .collect();
        let noise_energy = self
            .noise
            .iter()
            .map(|values| {
                values
                    .iter()
                    .map(|&value| 2.0f64.powi(6 - value as i32))
                    .collect()
            })
            .collect();
        LdSbrDequantizedChannel {
            envelope_energy,
            noise_energy,
        }
    }

    pub fn dequantize_coupled_pair(
        level: &Self,
        balance: &Self,
        control: &LdSbrChannelControl,
        default_amp_resolution: bool,
    ) -> Result<(LdSbrDequantizedChannel, LdSbrDequantizedChannel), LdSbrError> {
        if level.envelopes.len() != balance.envelopes.len()
            || level.noise.len() != balance.noise.len()
        {
            return Err(LdSbrError::CoupledLayoutMismatch);
        }
        let mut left_envelopes = Vec::with_capacity(level.envelopes.len());
        let mut right_envelopes = Vec::with_capacity(level.envelopes.len());
        for (level_values, balance_values) in level.envelopes.iter().zip(&balance.envelopes) {
            if level_values.len() != balance_values.len() {
                return Err(LdSbrError::CoupledLayoutMismatch);
            }
            let amp_resolution = control
                .grid
                .amp_resolution
                .unwrap_or(default_amp_resolution);
            let divisor = if amp_resolution { 1.0 } else { 2.0 };
            let mut left = Vec::with_capacity(level_values.len());
            let mut right = Vec::with_capacity(level_values.len());
            for (&level_value, &balance_value) in level_values.iter().zip(balance_values) {
                let total =
                    64.0 * 2.0f64.powf(level_value as f64 / divisor) * ENVELOPE_ENERGY_SCALE;
                let ratio = 2.0f64.powf(balance_value as f64 / divisor - 12.0);
                let right_value = 2.0 * total / (ratio + 1.0);
                left.push(ratio * right_value);
                right.push(right_value);
            }
            left_envelopes.push(left);
            right_envelopes.push(right);
        }
        let mut left_noise = Vec::with_capacity(level.noise.len());
        let mut right_noise = Vec::with_capacity(level.noise.len());
        for (level_values, balance_values) in level.noise.iter().zip(&balance.noise) {
            if level_values.len() != balance_values.len() {
                return Err(LdSbrError::CoupledLayoutMismatch);
            }
            let mut left = Vec::with_capacity(level_values.len());
            let mut right = Vec::with_capacity(level_values.len());
            for (&level_value, &balance_value) in level_values.iter().zip(balance_values) {
                let total = 2.0f64.powi(6 - level_value as i32);
                let ratio = 2.0f64.powi(balance_value as i32 - 12);
                let right_value = 2.0 * total / (ratio + 1.0);
                left.push(ratio * right_value);
                right.push(right_value);
            }
            left_noise.push(left);
            right_noise.push(right);
        }
        Ok((
            LdSbrDequantizedChannel {
                envelope_energy: left_envelopes,
                noise_energy: left_noise,
            },
            LdSbrDequantizedChannel {
                envelope_energy: right_envelopes,
                noise_energy: right_noise,
            },
        ))
    }

    pub fn parse_mono_after_prefix_usac(
        reader: &mut BitReader<'_>,
        control: &LdSbrChannelControl,
        tables: &LdSbrFrequencyTables,
        default_amp_resolution: bool,
        inter_tes: bool,
    ) -> Result<(Self, Vec<InterTesEnvelope>), LdSbrError> {
        let inverse_filtering_modes = read_invf(reader, tables.noise_band_count())?;
        let (envelopes, inter_tes_envelopes) = read_envelopes_usac(
            reader,
            control,
            tables,
            default_amp_resolution,
            false,
            inter_tes,
        )?;
        let noise = read_noise(reader, control, tables, false)?;
        Ok((
            Self {
                inverse_filtering_modes,
                envelopes,
                noise,
            },
            inter_tes_envelopes,
        ))
    }

    pub fn parse_stereo_after_prefix_usac(
        reader: &mut BitReader<'_>,
        prefix: &LdSbrChannelElementPrefix,
        tables: &LdSbrFrequencyTables,
        default_amp_resolution: bool,
        inter_tes: bool,
    ) -> Result<((Self, Vec<InterTesEnvelope>), (Self, Vec<InterTesEnvelope>)), LdSbrError> {
        let right_control = prefix.right.as_ref().ok_or(LdSbrError::ExpectedStereo)?;
        let left_invf = read_invf(reader, tables.noise_band_count())?;
        let right_invf = if prefix.coupling {
            left_invf.clone()
        } else {
            read_invf(reader, tables.noise_band_count())?
        };
        let (left_envelopes, left_tes) = read_envelopes_usac(
            reader,
            &prefix.left,
            tables,
            default_amp_resolution,
            false,
            inter_tes,
        )?;
        let (left_noise, right_envelopes, right_tes, right_noise) = if prefix.coupling {
            let left_noise = read_noise(reader, &prefix.left, tables, false)?;
            let (right_envelopes, right_tes) = read_envelopes_usac(
                reader,
                right_control,
                tables,
                default_amp_resolution,
                true,
                inter_tes,
            )?;
            let right_noise = read_noise(reader, right_control, tables, true)?;
            (left_noise, right_envelopes, right_tes, right_noise)
        } else {
            let (right_envelopes, right_tes) = read_envelopes_usac(
                reader,
                right_control,
                tables,
                default_amp_resolution,
                false,
                inter_tes,
            )?;
            let left_noise = read_noise(reader, &prefix.left, tables, false)?;
            let right_noise = read_noise(reader, right_control, tables, false)?;
            (left_noise, right_envelopes, right_tes, right_noise)
        };
        Ok((
            (
                Self {
                    inverse_filtering_modes: left_invf,
                    envelopes: left_envelopes,
                    noise: left_noise,
                },
                left_tes,
            ),
            (
                Self {
                    inverse_filtering_modes: right_invf,
                    envelopes: right_envelopes,
                    noise: right_noise,
                },
                right_tes,
            ),
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LdSbrPreviousValues {
    pub envelope_high: Vec<i16>,
    pub noise: Vec<i16>,
}

pub fn read_add_harmonics(
    reader: &mut BitReader<'_>,
    high_band_count: usize,
) -> Result<Vec<bool>, LdSbrError> {
    if !reader.read_bool()? {
        return Ok(vec![false; high_band_count]);
    }
    (0..high_band_count)
        .map(|_| Ok(reader.read_bool()?))
        .collect::<Result<Vec<_>, LdSbrError>>()
}

pub fn read_extended_data(reader: &mut BitReader<'_>) -> Result<Vec<u8>, LdSbrError> {
    if reader.remaining_bits() == 0 {
        return Ok(Vec::new());
    }
    if !reader.read_bool()? {
        return Ok(Vec::new());
    }
    let mut count = reader.read_u8(4)? as usize;
    if count == 15 {
        count += reader.read_u8(8)? as usize;
    }
    if reader.remaining_bits() < count * 8 {
        return Err(LdSbrError::UnexpectedEof);
    }
    (0..count)
        .map(|_| Ok(reader.read_u8(8)?))
        .collect::<Result<Vec<_>, LdSbrError>>()
}

#[derive(Debug, Clone, PartialEq)]
pub struct LdSbrFrame {
    pub transmitted_crc: Option<u16>,
    pub header_present: bool,
    pub active_header: LdSbrHeader,
    pub frequency_tables: LdSbrFrequencyTables,
    pub prefix: LdSbrChannelElementPrefix,
    pub left: LdSbrChannelValues,
    pub right: Option<LdSbrChannelValues>,
    pub left_dequantized: LdSbrDequantizedChannel,
    pub right_dequantized: Option<LdSbrDequantizedChannel>,
    pub left_harmonics: Vec<bool>,
    pub right_harmonics: Option<Vec<bool>>,
    pub extended_data: Vec<u8>,
    pub bits_read: usize,
}

#[derive(Debug, Clone)]
pub struct LdSbrFrameParser {
    header: LdSbrHeader,
    sampling_frequency: u32,
    time_slots: u8,
    stereo: bool,
    crc_present: bool,
    previous_left: LdSbrPreviousValues,
    previous_right: LdSbrPreviousValues,
}

impl LdSbrFrameParser {
    pub fn new(
        header: LdSbrHeader,
        sampling_frequency: u32,
        frame_length: usize,
        stereo: bool,
        crc_present: bool,
    ) -> Result<Self, LdSbrError> {
        let time_slots = match frame_length {
            480 => 15,
            512 => 16,
            _ => return Err(LdSbrError::UnsupportedFrameLength(frame_length)),
        };
        LdSbrFrequencyTables::from_header(&header, sampling_frequency)?;
        Ok(Self {
            header,
            sampling_frequency,
            time_slots,
            stereo,
            crc_present,
            previous_left: LdSbrPreviousValues::default(),
            previous_right: LdSbrPreviousValues::default(),
        })
    }

    pub fn clear_history(&mut self) {
        self.previous_left = LdSbrPreviousValues::default();
        self.previous_right = LdSbrPreviousValues::default();
    }

    pub fn parse(&mut self, reader: &mut BitReader<'_>) -> Result<LdSbrFrame, LdSbrError> {
        let start = reader.bits_read();
        let transmitted_crc = self.crc_present.then(|| reader.read_u16(10)).transpose()?;
        let crc_region_start = reader.bits_read();
        let header_present = reader.read_bool()?;
        let next_header = if header_present {
            parse_frame_header(reader)?
        } else {
            self.header.clone()
        };
        let tables = LdSbrFrequencyTables::from_header(&next_header, self.sampling_frequency)?;
        let prefix = LdSbrChannelElementPrefix::parse(reader, self.time_slots, self.stereo)?;
        let (mut left, mut right) = if self.stereo {
            let (left, right) = LdSbrChannelValues::parse_stereo_after_prefix(
                reader,
                &prefix,
                &tables,
                next_header.amp_resolution,
            )?;
            (left, Some(right))
        } else {
            (
                LdSbrChannelValues::parse_mono_after_prefix(
                    reader,
                    &prefix.left,
                    &tables,
                    next_header.amp_resolution,
                )?,
                None,
            )
        };
        let mut next_previous_left = self.previous_left.clone();
        let mut next_previous_right = self.previous_right.clone();
        left.reconstruct_deltas(&prefix.left, &tables, &mut next_previous_left)?;
        if let (Some(values), Some(control)) = (right.as_mut(), prefix.right.as_ref()) {
            values.reconstruct_deltas(control, &tables, &mut next_previous_right)?;
        }
        let (left_dequantized, right_dequantized) = if prefix.coupling {
            let right_values = right.as_ref().ok_or(LdSbrError::ExpectedStereo)?;
            let (left, right) = LdSbrChannelValues::dequantize_coupled_pair(
                &left,
                right_values,
                &prefix.left,
                next_header.amp_resolution,
            )?;
            (left, Some(right))
        } else {
            (
                left.dequantize_uncoupled(&prefix.left, next_header.amp_resolution),
                right
                    .as_ref()
                    .zip(prefix.right.as_ref())
                    .map(|(values, control)| {
                        values.dequantize_uncoupled(control, next_header.amp_resolution)
                    }),
            )
        };
        let left_harmonics = read_add_harmonics(reader, tables.high_band_count())?;
        let right_harmonics = if self.stereo {
            Some(read_add_harmonics(reader, tables.high_band_count())?)
        } else {
            None
        };
        let extended_data = read_extended_data(reader)?;
        if let Some(expected) = transmitted_crc {
            let calculated = reader.crc_msb(crc_region_start, reader.bits_read(), 10, 0x0633);
            if calculated != expected as u32 {
                return Err(LdSbrError::CrcMismatch {
                    expected,
                    calculated: calculated as u16,
                });
            }
        }
        self.previous_left = next_previous_left;
        self.previous_right = next_previous_right;
        self.header = next_header.clone();
        Ok(LdSbrFrame {
            transmitted_crc,
            header_present,
            active_header: next_header.clone(),
            frequency_tables: tables,
            prefix,
            left,
            right,
            left_dequantized,
            right_dequantized,
            left_harmonics,
            right_harmonics,
            extended_data,
            bits_read: reader.bits_read() - start,
        })
    }

    pub fn header(&self) -> &LdSbrHeader {
        &self.header
    }
}

fn parse_frame_header(reader: &mut BitReader<'_>) -> Result<LdSbrHeader, LdSbrError> {
    let amp_resolution = reader.read_bool()?;
    let start_frequency = reader.read_u8(4)?;
    let stop_frequency = reader.read_u8(4)?;
    let crossover_band = reader.read_u8(3)?;
    let reserved = reader.read_u8(2)?;
    let extra_1 = reader.read_bool()?;
    let extra_2 = reader.read_bool()?;
    let (frequency_scale, alter_scale, noise_bands) = if extra_1 {
        (
            Some(reader.read_u8(2)?),
            Some(reader.read_bool()?),
            Some(reader.read_u8(2)?),
        )
    } else {
        (None, None, None)
    };
    let (limiter_bands, limiter_gains, interpol_frequency, smoothing_mode) = if extra_2 {
        (
            Some(reader.read_u8(2)?),
            Some(reader.read_u8(2)?),
            Some(reader.read_bool()?),
            Some(reader.read_bool()?),
        )
    } else {
        (None, None, None, None)
    };
    Ok(LdSbrHeader {
        amp_resolution,
        crossover_band,
        reserved,
        start_frequency,
        stop_frequency,
        frequency_scale,
        alter_scale,
        noise_bands,
        limiter_bands,
        limiter_gains,
        interpol_frequency,
        smoothing_mode,
    })
}

fn low_to_high(offset: isize, index: usize, high_resolution: bool) -> usize {
    if high_resolution {
        index
    } else if offset >= 0 {
        if index < offset as usize {
            index
        } else {
            2 * index - offset as usize
        }
    } else {
        let magnitude = (-offset) as usize;
        if index < magnitude {
            3 * index
        } else {
            2 * index + magnitude
        }
    }
}

fn map_to_high(
    value: i16,
    previous: &mut [i16],
    offset: isize,
    index: usize,
    high_resolution: bool,
) {
    if high_resolution {
        previous[index] = value;
    } else if offset >= 0 {
        let offset = offset as usize;
        if index < offset {
            previous[index] = value;
        } else {
            previous[2 * index - offset] = value;
            previous[2 * index + 1 - offset] = value;
        }
    } else {
        let offset = (-offset) as usize;
        if index < offset {
            previous[3 * index..3 * index + 3].fill(value);
        } else {
            previous[2 * index + offset] = value;
            previous[2 * index + 1 + offset] = value;
        }
    }
}

pub(crate) fn read_invf(reader: &mut BitReader<'_>, count: usize) -> Result<Vec<u8>, LdSbrError> {
    (0..count)
        .map(|_| Ok(reader.read_u8(2)?))
        .collect::<Result<Vec<_>, LdSbrError>>()
}

fn read_envelopes(
    reader: &mut BitReader<'_>,
    control: &LdSbrChannelControl,
    tables: &LdSbrFrequencyTables,
    default_amp_resolution: bool,
    balance: bool,
) -> Result<Vec<Vec<i16>>, LdSbrError> {
    let amp_resolution = control
        .grid
        .amp_resolution
        .unwrap_or(default_amp_resolution);
    let (time_book, frequency_book) = match (balance, amp_resolution) {
        (false, false) => (
            SbrHuffmanBook::EnvelopeLevel15Time,
            SbrHuffmanBook::EnvelopeLevel15Frequency,
        ),
        (false, true) => (
            SbrHuffmanBook::EnvelopeLevel30Time,
            SbrHuffmanBook::EnvelopeLevel30Frequency,
        ),
        (true, false) => (
            SbrHuffmanBook::EnvelopeBalance15Time,
            SbrHuffmanBook::EnvelopeBalance15Frequency,
        ),
        (true, true) => (
            SbrHuffmanBook::EnvelopeBalance30Time,
            SbrHuffmanBook::EnvelopeBalance30Frequency,
        ),
    };
    let start_bits = match (balance, amp_resolution) {
        (false, false) => 7,
        (false, true) => 6,
        (true, false) => 6,
        (true, true) => 5,
    };
    let scale = if balance { 2 } else { 1 };
    let mut envelopes = Vec::with_capacity(control.grid.envelope_count());
    for (index, &high_resolution) in control.grid.frequency_resolution.iter().enumerate() {
        let count = if high_resolution {
            tables.high_band_count()
        } else {
            tables.low_band_count()
        };
        let time_domain = control.envelope_time_domain[index];
        let mut values = Vec::with_capacity(count);
        if !time_domain {
            values.push((reader.read_u8(start_bits)? as i16) * scale);
        }
        while values.len() < count {
            let book = if time_domain {
                time_book
            } else {
                frequency_book
            };
            values.push(decode_sbr_huffman(reader, book)? as i16 * scale);
        }
        envelopes.push(values);
    }
    Ok(envelopes)
}

fn read_envelopes_usac(
    reader: &mut BitReader<'_>,
    control: &LdSbrChannelControl,
    tables: &LdSbrFrequencyTables,
    default_amp_resolution: bool,
    balance: bool,
    inter_tes: bool,
) -> Result<(Vec<Vec<i16>>, Vec<InterTesEnvelope>), LdSbrError> {
    let amp_resolution = control
        .grid
        .amp_resolution
        .unwrap_or(default_amp_resolution);
    let (time_book, frequency_book) = match (balance, amp_resolution) {
        (false, false) => (
            SbrHuffmanBook::EnvelopeLevel15Time,
            SbrHuffmanBook::EnvelopeLevel15Frequency,
        ),
        (false, true) => (
            SbrHuffmanBook::EnvelopeLevel30Time,
            SbrHuffmanBook::EnvelopeLevel30Frequency,
        ),
        (true, false) => (
            SbrHuffmanBook::EnvelopeBalance15Time,
            SbrHuffmanBook::EnvelopeBalance15Frequency,
        ),
        (true, true) => (
            SbrHuffmanBook::EnvelopeBalance30Time,
            SbrHuffmanBook::EnvelopeBalance30Frequency,
        ),
    };
    let start_bits = match (balance, amp_resolution) {
        (false, false) => 7,
        (false, true) => 6,
        (true, false) => 6,
        (true, true) => 5,
    };
    let scale = if balance { 2 } else { 1 };
    let mut envelopes = Vec::with_capacity(control.grid.envelope_count());
    let mut shaping = Vec::with_capacity(control.grid.envelope_count());
    for (index, &high_resolution) in control.grid.frequency_resolution.iter().enumerate() {
        let count = if high_resolution {
            tables.high_band_count()
        } else {
            tables.low_band_count()
        };
        let time_domain = control.envelope_time_domain[index];
        let mut values = Vec::with_capacity(count);
        if !time_domain {
            values.push(reader.read_u8(start_bits)? as i16 * scale);
        }
        while values.len() < count {
            values.push(
                decode_sbr_huffman(
                    reader,
                    if time_domain {
                        time_book
                    } else {
                        frequency_book
                    },
                )? as i16
                    * scale,
            );
        }
        envelopes.push(values);
        let active = inter_tes && reader.read_bool()?;
        let mode = if active { reader.read_u8(2)? } else { 0 };
        shaping.push(InterTesEnvelope { active, mode });
    }
    Ok((envelopes, shaping))
}

pub(crate) fn read_noise(
    reader: &mut BitReader<'_>,
    control: &LdSbrChannelControl,
    tables: &LdSbrFrequencyTables,
    balance: bool,
) -> Result<Vec<Vec<i16>>, LdSbrError> {
    let count = tables.noise_band_count();
    let scale = if balance { 2 } else { 1 };
    let mut result = Vec::with_capacity(control.grid.noise_envelope_count());
    for &time_domain in &control.noise_time_domain {
        let mut values = Vec::with_capacity(count);
        if !time_domain {
            values.push(reader.read_u8(5)? as i16 * scale);
        }
        while values.len() < count {
            let book = if time_domain {
                if balance {
                    SbrHuffmanBook::NoiseBalanceTime
                } else {
                    SbrHuffmanBook::NoiseLevelTime
                }
            } else if balance {
                SbrHuffmanBook::EnvelopeBalance30Frequency
            } else {
                SbrHuffmanBook::EnvelopeLevel30Frequency
            };
            values.push(decode_sbr_huffman(reader, book)? as i16 * scale);
        }
        result.push(values);
    }
    Ok(result)
}

impl LdSbrChannelElementPrefix {
    /// Parse an ELD SBR channel element up to (but not including) `sbr_invf`.
    /// The remaining syntax depends on frequency tables derived from the ASC
    /// default header and is decoded by the next payload stage.
    pub fn parse(
        reader: &mut BitReader<'_>,
        time_slots: u8,
        stereo: bool,
    ) -> Result<Self, LdSbrError> {
        let data_extra = if reader.read_bool()? {
            let left = reader.read_u8(4)?;
            let right = stereo.then(|| reader.read_u8(4)).transpose()?;
            Some((left, right))
        } else {
            None
        };
        let coupling = stereo && reader.read_bool()?;
        let left_grid = LdSbrGrid::parse(reader, time_slots)?;
        let right_grid = if stereo {
            if coupling {
                left_grid.clone()
            } else {
                LdSbrGrid::parse(reader, time_slots)?
            }
        } else {
            left_grid.clone()
        };
        let left = read_channel_control(reader, left_grid)?;
        let right = stereo
            .then(|| read_channel_control(reader, right_grid))
            .transpose()?;
        Ok(Self {
            data_extra,
            coupling,
            left,
            right,
        })
    }
}

fn read_channel_control(
    reader: &mut BitReader<'_>,
    grid: LdSbrGrid,
) -> Result<LdSbrChannelControl, LdSbrError> {
    let envelope_time_domain = (0..grid.envelope_count())
        .map(|_| reader.read_bool())
        .collect::<Result<Vec<_>, _>>()?;
    let noise_time_domain = (0..grid.noise_envelope_count())
        .map(|_| reader.read_bool())
        .collect::<Result<Vec<_>, _>>()?;
    Ok(LdSbrChannelControl {
        grid,
        envelope_time_domain,
        noise_time_domain,
    })
}

impl LdSbrGrid {
    pub fn parse(reader: &mut BitReader<'_>, time_slots: u8) -> Result<Self, LdSbrError> {
        if !matches!(time_slots, 15 | 16) {
            return Err(LdSbrError::UnsupportedTimeSlots(time_slots));
        }
        let transient = reader.read_bool()?;
        if transient {
            let position = reader.read_u8(4)?;
            if position >= time_slots {
                return Err(LdSbrError::InvalidTransientPosition {
                    position,
                    time_slots,
                });
            }
            let row = transient_row(time_slots, position);
            let envelope_count = row[0] as usize;
            let transient_envelope = row[1] as usize;
            let mut borders = Vec::with_capacity(envelope_count + 1);
            borders.push(0);
            for &border in &row[3..3 + envelope_count.saturating_sub(1)] {
                borders.push(border as u8);
            }
            borders.push(time_slots);
            let frequency_resolution = (0..envelope_count)
                .map(|_| reader.read_bool())
                .collect::<Result<Vec<_>, _>>()?;
            let middle = borders[if transient_envelope == 0 {
                1
            } else {
                transient_envelope
            }];
            Ok(Self {
                transient,
                amp_resolution: None,
                borders,
                frequency_resolution,
                transient_envelope: Some(transient_envelope),
                noise_borders: vec![0, middle, time_slots],
            })
        } else {
            let exponent = reader.read_u8(2)?;
            let envelope_count = 1usize << exponent;
            if envelope_count > 4 {
                return Err(LdSbrError::TooManyEnvelopes(envelope_count));
            }
            let amp_resolution = (envelope_count == 1)
                .then(|| reader.read_bool())
                .transpose()?;
            let resolution = reader.read_bool()?;
            let borders = fixed_borders(time_slots, envelope_count);
            let noise_borders = if envelope_count == 1 {
                vec![0, time_slots]
            } else {
                vec![0, 8, time_slots]
            };
            Ok(Self {
                transient,
                amp_resolution,
                borders,
                frequency_resolution: vec![resolution; envelope_count],
                transient_envelope: None,
                noise_borders,
            })
        }
    }

    pub fn envelope_count(&self) -> usize {
        self.frequency_resolution.len()
    }

    pub fn noise_envelope_count(&self) -> usize {
        self.noise_borders.len() - 1
    }
}

fn fixed_borders(time_slots: u8, count: usize) -> Vec<u8> {
    let step = 16 / count as u8;
    (0..count)
        .map(|index| index as u8 * step)
        .chain(std::iter::once(time_slots))
        .collect()
}

fn transient_row(time_slots: u8, position: u8) -> [i8; 6] {
    let p = position as i8;
    let last_three_envelope_position = if time_slots == 15 { 9 } else { 10 };
    if position < 2 {
        [2, 0, 0, p + 4, -1, -1]
    } else if position <= last_three_envelope_position {
        [3, 1, 1, p, p + 4, -1]
    } else {
        [2, 1, 1, p, -1, -1]
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LdSbrError {
    UnexpectedEof,
    UnsupportedTimeSlots(u8),
    InvalidTransientPosition { position: u8, time_slots: u8 },
    TooManyEnvelopes(usize),
    UnsupportedSamplingFrequency(u32),
    InvalidFrequencyRange,
    InvalidCrossoverBand(u8),
    InvalidHuffmanCodeword,
    ExpectedStereo,
    UnsupportedFrameLength(usize),
    CrcMismatch { expected: u16, calculated: u16 },
    CoupledLayoutMismatch,
}

impl From<BitError> for LdSbrError {
    fn from(_: BitError) -> Self {
        Self::UnexpectedEof
    }
}

impl fmt::Display for LdSbrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::UnexpectedEof => write!(f, "truncated LD-SBR grid"),
            Self::UnsupportedTimeSlots(value) => {
                write!(f, "unsupported LD-SBR time-slot count {value}")
            }
            Self::InvalidTransientPosition {
                position,
                time_slots,
            } => write!(
                f,
                "LD-SBR transient position {position} exceeds {time_slots} time slots"
            ),
            Self::TooManyEnvelopes(value) => {
                write!(f, "unsupported LD-SBR envelope count {value}")
            }
            Self::UnsupportedSamplingFrequency(value) => {
                write!(f, "unsupported LD-SBR sampling frequency {value}")
            }
            Self::InvalidFrequencyRange => write!(f, "invalid LD-SBR frequency range"),
            Self::InvalidCrossoverBand(value) => {
                write!(f, "invalid LD-SBR crossover band {value}")
            }
            Self::InvalidHuffmanCodeword => write!(f, "invalid LD-SBR Huffman codeword"),
            Self::ExpectedStereo => write!(f, "LD-SBR stereo payload requires a right channel"),
            Self::UnsupportedFrameLength(value) => {
                write!(f, "unsupported LD-SBR frame length {value}")
            }
            Self::CrcMismatch {
                expected,
                calculated,
            } => write!(
                f,
                "LD-SBR CRC mismatch: expected {expected:#05x}, calculated {calculated:#05x}"
            ),
            Self::CoupledLayoutMismatch => write!(f, "LD-SBR coupled channel layout mismatch"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitWriter;

    #[test]
    fn formats_every_ld_sbr_error_variant() {
        let errors = [
            LdSbrError::UnexpectedEof,
            LdSbrError::UnsupportedTimeSlots(17),
            LdSbrError::InvalidTransientPosition {
                position: 17,
                time_slots: 16,
            },
            LdSbrError::TooManyEnvelopes(8),
            LdSbrError::UnsupportedSamplingFrequency(12_345),
            LdSbrError::InvalidFrequencyRange,
            LdSbrError::InvalidCrossoverBand(15),
            LdSbrError::InvalidHuffmanCodeword,
            LdSbrError::ExpectedStereo,
            LdSbrError::UnsupportedFrameLength(1024),
            LdSbrError::CrcMismatch {
                expected: 0x155,
                calculated: 0x2aa,
            },
            LdSbrError::CoupledLayoutMismatch,
        ];
        for error in errors {
            assert!(!error.to_string().is_empty());
        }

        assert!(std::panic::catch_unwind(|| parse_sbr_huffman_table("missing_table")).is_err());

        let header = LdSbrHeader {
            start_frequency: 5,
            stop_frequency: 8,
            crossover_band: 2,
            ..LdSbrHeader::default()
        };
        assert!(LdSbrFrameParser::new(header.clone(), 44_100, 480, false, false).is_ok());
        assert!(matches!(
            LdSbrFrameParser::new(header.clone(), 44_100, 1024, false, false),
            Err(LdSbrError::UnsupportedFrameLength(1024))
        ));
        let mut crc_parser = LdSbrFrameParser::new(header, 44_100, 512, false, true).unwrap();
        assert!(matches!(
            crc_parser.parse(&mut BitReader::new(&[])),
            Err(LdSbrError::UnexpectedEof)
        ));
    }

    #[test]
    fn parses_complete_frame_header_and_converts_bit_errors() {
        assert_eq!(
            LdSbrError::from(BitError::UnexpectedEof {
                needed_bits: 1,
                remaining_bits: 0,
            }),
            LdSbrError::UnexpectedEof
        );
        let mut writer = BitWriter::new();
        writer.write_bool(true);
        writer.write(5, 4);
        writer.write(8, 4);
        writer.write(2, 3);
        writer.write(1, 2);
        writer.write_bool(true);
        writer.write_bool(true);
        writer.write(2, 2);
        writer.write_bool(false);
        writer.write(3, 2);
        writer.write(1, 2);
        writer.write(2, 2);
        writer.write_bool(true);
        writer.write_bool(false);
        let header = parse_frame_header(&mut BitReader::new(&writer.finish())).unwrap();
        assert_eq!(
            header,
            LdSbrHeader {
                amp_resolution: true,
                crossover_band: 2,
                reserved: 1,
                start_frequency: 5,
                stop_frequency: 8,
                frequency_scale: Some(2),
                alter_scale: Some(false),
                noise_bands: Some(3),
                limiter_bands: Some(1),
                limiter_gains: Some(2),
                interpol_frequency: Some(true),
                smoothing_mode: Some(false),
            }
        );

        let mut minimal = BitWriter::new();
        minimal.write_bool(false);
        minimal.write(5, 4);
        minimal.write(8, 4);
        minimal.write(2, 3);
        minimal.write(0, 2);
        minimal.write_bool(false);
        minimal.write_bool(false);
        let header = parse_frame_header(&mut BitReader::new(&minimal.finish())).unwrap();
        assert_eq!(header.frequency_scale, None);
        assert_eq!(header.alter_scale, None);
        assert_eq!(header.noise_bands, None);
        assert_eq!(header.limiter_bands, None);
        assert_eq!(header.limiter_gains, None);
        assert_eq!(header.interpol_frequency, None);
        assert_eq!(header.smoothing_mode, None);

        let initial_header = LdSbrHeader {
            start_frequency: 5,
            stop_frequency: 8,
            crossover_band: 2,
            ..LdSbrHeader::default()
        };
        let mut parser = LdSbrFrameParser::new(initial_header, 44_100, 512, false, false).unwrap();
        let mut writer = BitWriter::new();
        writer.write_bool(true); // frame header present
        writer.write_bool(true); // amp resolution
        writer.write(5, 4);
        writer.write(8, 4);
        writer.write(2, 3);
        writer.write(1, 2);
        writer.write_bool(true);
        writer.write_bool(true);
        writer.write(2, 2);
        writer.write_bool(false);
        writer.write(3, 2);
        writer.write(1, 2);
        writer.write(2, 2);
        writer.write_bool(true);
        writer.write_bool(false);
        assert_eq!(
            parser.parse(&mut BitReader::new(&writer.finish())),
            Err(LdSbrError::UnexpectedEof)
        );
    }

    #[test]
    fn parses_fixfix_grids_for_480_and_512_frames() {
        for slots in [15, 16] {
            for (exponent, envelope_count) in [(0, 1), (1, 2), (2, 4)] {
                let mut writer = BitWriter::new();
                writer.write_bool(false);
                writer.write(exponent, 2);
                if envelope_count == 1 {
                    writer.write_bool(true);
                }
                writer.write_bool(true);
                let bytes = writer.finish();
                let grid = LdSbrGrid::parse(&mut BitReader::new(&bytes), slots).unwrap();
                assert_eq!(grid.borders, fixed_borders(slots, envelope_count));
                assert_eq!(grid.frequency_resolution, vec![true; envelope_count]);
                assert_eq!(
                    grid.noise_borders,
                    if envelope_count == 1 {
                        vec![0, slots]
                    } else {
                        vec![0, 8, slots]
                    }
                );
            }
        }

        assert_eq!(
            LdSbrGrid::parse(&mut BitReader::new(&[]), 14),
            Err(LdSbrError::UnsupportedTimeSlots(14))
        );
        let mut writer = BitWriter::new();
        writer.write_bool(true);
        writer.write(15, 4);
        assert_eq!(
            LdSbrGrid::parse(&mut BitReader::new(&writer.finish()), 15),
            Err(LdSbrError::InvalidTransientPosition {
                position: 15,
                time_slots: 15,
            })
        );
    }

    #[test]
    fn parses_transient_grid_using_fdk_envelope_table_layout() {
        let mut writer = BitWriter::new();
        writer.write_bool(true);
        writer.write(10, 4);
        writer.write_bool(true);
        writer.write_bool(false);
        writer.write_bool(true);
        let bytes = writer.finish();
        let grid = LdSbrGrid::parse(&mut BitReader::new(&bytes), 16).unwrap();
        assert_eq!(grid.borders, vec![0, 10, 14, 16]);
        assert_eq!(grid.transient_envelope, Some(1));
        assert_eq!(grid.noise_borders, vec![0, 10, 16]);
        assert_eq!(grid.frequency_resolution, vec![true, false, true]);

        for (position, expected_borders, transient_envelope, noise_middle) in [
            (1u8, vec![0u8, 5, 16], 0usize, 5u8),
            (15u8, vec![0u8, 15, 16], 1usize, 15u8),
        ] {
            let mut writer = BitWriter::new();
            writer.write_bool(true);
            writer.write(position as u32, 4);
            writer.write_bool(false);
            writer.write_bool(true);
            let grid = LdSbrGrid::parse(&mut BitReader::new(&writer.finish()), 16).unwrap();
            assert_eq!(grid.borders, expected_borders);
            assert_eq!(grid.transient_envelope, Some(transient_envelope));
            assert_eq!(grid.noise_borders, vec![0, noise_middle, 16]);
        }
    }

    #[test]
    fn rejects_reserved_eight_envelope_fixfix_grid() {
        let mut writer = BitWriter::new();
        writer.write_bool(false);
        writer.write(3, 2);
        let bytes = writer.finish();
        assert_eq!(
            LdSbrGrid::parse(&mut BitReader::new(&bytes), 16).unwrap_err(),
            LdSbrError::TooManyEnvelopes(8)
        );
    }

    #[test]
    fn parses_uncoupled_stereo_prefix_through_direction_vectors() {
        let mut writer = BitWriter::new();
        writer.write_bool(true); // data extra
        writer.write(3, 4);
        writer.write(9, 4);
        writer.write_bool(false); // uncoupled
        for resolution in [false, true] {
            writer.write_bool(false); // FIXFIX
            writer.write(0, 2); // one envelope
            writer.write_bool(true); // current amp resolution
            writer.write_bool(resolution);
        }
        writer.write_bool(false); // left envelope frequency direction
        writer.write_bool(true); // left noise time direction
        writer.write_bool(true); // right envelope time direction
        writer.write_bool(false); // right noise frequency direction
        let bytes = writer.finish();
        let prefix =
            LdSbrChannelElementPrefix::parse(&mut BitReader::new(&bytes), 16, true).unwrap();
        assert_eq!(prefix.data_extra, Some((3, Some(9))));
        assert!(!prefix.coupling);
        assert_eq!(prefix.left.envelope_time_domain, vec![false]);
        assert_eq!(prefix.left.noise_time_domain, vec![true]);
        let right = prefix.right.unwrap();
        assert_eq!(right.grid.frequency_resolution, vec![true]);
        assert_eq!(right.envelope_time_domain, vec![true]);
        assert_eq!(right.noise_time_domain, vec![false]);
    }

    #[test]
    fn derives_monotonic_frequency_tables_from_default_header() {
        let header = LdSbrHeader {
            start_frequency: 5,
            stop_frequency: 8,
            crossover_band: 2,
            ..LdSbrHeader::default()
        };
        let tables = LdSbrFrequencyTables::from_header(&header, 44_100).unwrap();
        for table in [&tables.master, &tables.high, &tables.low, &tables.noise] {
            assert!(table.windows(2).all(|pair| pair[0] < pair[1]));
        }
        assert_eq!(tables.high[0], tables.master[2]);
        assert_eq!(tables.high.last(), tables.master.last());
        assert_eq!(tables.low.first(), tables.high.first());
        assert_eq!(tables.low.last(), tables.high.last());
        assert_eq!(tables.noise.first(), tables.low.first());
        assert_eq!(tables.noise.last(), tables.low.last());
        assert_eq!(tables.noise_band_count(), tables.noise.len() - 1);

        assert!(matches!(
            LdSbrFrequencyTables::from_header(
                &LdSbrHeader {
                    crossover_band: u8::MAX,
                    ..header.clone()
                },
                44_100,
            ),
            Err(LdSbrError::InvalidCrossoverBand(u8::MAX))
        ));

        let mut odd_band_table = None;
        'find_odd_layout: for scale in 0..=3 {
            for alter_scale in [false, true] {
                for start_frequency in 0..=15 {
                    for stop_frequency in 0..=15 {
                        for crossover_band in 0..=15 {
                            let candidate = LdSbrFrequencyTables::from_header(
                                &LdSbrHeader {
                                    start_frequency,
                                    stop_frequency,
                                    crossover_band,
                                    frequency_scale: Some(scale),
                                    alter_scale: Some(alter_scale),
                                    ..LdSbrHeader::default()
                                },
                                44_100,
                            );
                            if candidate
                                .as_ref()
                                .is_ok_and(|candidate| candidate.high_band_count() % 2 == 1)
                            {
                                odd_band_table = candidate.ok();
                                break 'find_odd_layout;
                            }
                        }
                    }
                }
            }
        }
        let odd_band_table = odd_band_table.expect("an odd master-band layout exists");
        assert_eq!(odd_band_table.low.first(), odd_band_table.high.first());
        assert_eq!(odd_band_table.low.last(), odd_band_table.high.last());

        assert!(make_master(10, 20, 0, true).is_ok());
        assert!(make_master(10, 21, 0, false).is_ok());
        assert!(make_master(10, 20, 3, false).is_ok());
        assert_eq!(
            logarithmic_widths(10, 20, 0),
            Err(LdSbrError::InvalidFrequencyRange)
        );
        assert_eq!(logarithmic_widths(63, 64, 1), Ok(vec![1]));
        assert_eq!(
            logarithmic_widths(20, 10, 2),
            Err(LdSbrError::InvalidFrequencyRange)
        );
    }

    #[test]
    fn loads_and_decodes_fdk_sbr_huffman_books() {
        assert_eq!(ENV_LEVEL_10_T.len(), 120);
        assert_eq!(ENV_LEVEL_11_T.len(), 62);
        assert_eq!(ENV_BALANCE_11_T.len(), 24);
        assert_eq!(NOISE_LEVEL_11_T.len(), 62);
        for book in [
            SbrHuffmanBook::EnvelopeLevel15Time,
            SbrHuffmanBook::EnvelopeLevel15Frequency,
            SbrHuffmanBook::EnvelopeLevel30Time,
            SbrHuffmanBook::EnvelopeLevel30Frequency,
            SbrHuffmanBook::EnvelopeBalance15Time,
            SbrHuffmanBook::EnvelopeBalance15Frequency,
            SbrHuffmanBook::EnvelopeBalance30Time,
            SbrHuffmanBook::EnvelopeBalance30Frequency,
            SbrHuffmanBook::NoiseLevelTime,
            SbrHuffmanBook::NoiseBalanceTime,
        ] {
            let encoded = encode_sbr_huffman(book, 0).unwrap();
            let mut writer = BitWriter::new();
            for bit in encoded {
                writer.write_bool(bit);
            }
            assert_eq!(
                decode_sbr_huffman(&mut BitReader::new(&writer.finish()), book).unwrap(),
                0
            );
        }
    }

    #[test]
    fn frequency_lookup_and_resolution_mapping_cover_all_layouts() {
        for sampling_frequency in [
            16_000, 22_050, 24_000, 32_000, 40_000, 44_100, 48_000, 64_000, 88_200, 96_000,
        ] {
            let k0 = start_band(sampling_frequency, 0).unwrap();
            assert!(start_band(sampling_frequency, 15).unwrap() > k0);
            for stop_index in 0..=15 {
                let _ = stop_band(sampling_frequency, stop_index, k0);
            }
        }
        assert_eq!(start_band(12_345, 0), None);
        assert_eq!(start_band(44_100, 16), None);
        assert_eq!(stop_band(44_100, 16, 8), None);
        assert_eq!(stop_band(44_100, 0, 64), None);

        assert_eq!(low_to_high(4, 3, true), 3);
        assert_eq!(low_to_high(2, 1, false), 1);
        assert_eq!(low_to_high(2, 3, false), 4);
        assert_eq!(low_to_high(-2, 1, false), 3);
        assert_eq!(low_to_high(-2, 3, false), 8);

        let mut high = vec![0; 10];
        map_to_high(1, &mut high, 4, 2, true);
        assert_eq!(high[2], 1);

        let mut positive = vec![0; 10];
        map_to_high(2, &mut positive, 2, 1, false);
        map_to_high(3, &mut positive, 2, 3, false);
        assert_eq!(positive, [0, 2, 0, 0, 3, 3, 0, 0, 0, 0]);

        let mut negative = vec![0; 10];
        map_to_high(4, &mut negative, -2, 1, false);
        map_to_high(5, &mut negative, -2, 3, false);
        assert_eq!(negative, [0, 0, 0, 4, 4, 4, 0, 0, 5, 5]);
    }

    #[test]
    fn reads_every_usac_envelope_and_noise_coding_mode() {
        fn write_huffman(writer: &mut BitWriter, book: SbrHuffmanBook) {
            for bit in encode_sbr_huffman(book, 0).unwrap() {
                writer.write_bool(bit);
            }
        }

        let tables = LdSbrFrequencyTables {
            master: vec![10, 11, 12],
            high: vec![10, 11],
            low: vec![10, 11],
            noise: vec![10, 11, 12],
        };
        let control =
            |amp_resolution, envelope_time_domain, noise_time_domain| LdSbrChannelControl {
                grid: LdSbrGrid {
                    transient: false,
                    amp_resolution: Some(amp_resolution),
                    borders: vec![0, 16],
                    frequency_resolution: vec![true],
                    transient_envelope: None,
                    noise_borders: vec![0, 16],
                },
                envelope_time_domain: vec![envelope_time_domain],
                noise_time_domain: vec![noise_time_domain],
            };

        for (balance, amp_resolution, time_book) in [
            (false, false, SbrHuffmanBook::EnvelopeLevel15Time),
            (false, true, SbrHuffmanBook::EnvelopeLevel30Time),
            (true, false, SbrHuffmanBook::EnvelopeBalance15Time),
            (true, true, SbrHuffmanBook::EnvelopeBalance30Time),
        ] {
            let start_bits = match (balance, amp_resolution) {
                (false, false) => 7,
                (false, true) | (true, false) => 6,
                (true, true) => 5,
            };
            let mut writer = BitWriter::new();
            writer.write(0, start_bits);
            writer.write_bool(true);
            writer.write(2, 2);
            let (values, shaping) = read_envelopes_usac(
                &mut BitReader::new(&writer.finish()),
                &control(amp_resolution, false, false),
                &tables,
                false,
                balance,
                true,
            )
            .unwrap();
            assert_eq!(values, vec![vec![0]]);
            assert_eq!(
                shaping,
                vec![InterTesEnvelope {
                    active: true,
                    mode: 2
                }]
            );

            let mut writer = BitWriter::new();
            for bit in encode_sbr_huffman(time_book, 0).unwrap() {
                writer.write_bool(bit);
            }
            writer.write_bool(false);
            let (values, shaping) = read_envelopes_usac(
                &mut BitReader::new(&writer.finish()),
                &control(amp_resolution, true, false),
                &tables,
                false,
                balance,
                true,
            )
            .unwrap();
            assert_eq!(values, vec![vec![0]]);
            assert_eq!(
                shaping,
                vec![InterTesEnvelope {
                    active: false,
                    mode: 0
                }]
            );
        }

        let mut low_resolution = control(false, false, false);
        low_resolution.grid.frequency_resolution[0] = false;
        let mut writer = BitWriter::new();
        writer.write(0, 7);
        let (values, shaping) = read_envelopes_usac(
            &mut BitReader::new(&writer.finish()),
            &low_resolution,
            &tables,
            false,
            false,
            false,
        )
        .unwrap();
        assert_eq!(values, vec![vec![0]]);
        assert_eq!(
            shaping,
            vec![InterTesEnvelope {
                active: false,
                mode: 0,
            }]
        );

        let wider_tables = LdSbrFrequencyTables {
            master: vec![10, 11, 12],
            high: vec![10, 11, 12],
            low: vec![10, 11, 12],
            noise: vec![10, 12],
        };
        let mut absolute_only = BitWriter::new();
        absolute_only.write(0, 6);
        let bits = absolute_only.bits_written();
        let bytes = absolute_only.finish();
        assert!(read_envelopes_usac(
            &mut BitReader::with_bit_len(&bytes, bits).unwrap(),
            &control(true, false, false),
            &wider_tables,
            false,
            false,
            false,
        )
        .is_err());

        for (balance, time_domain, book) in [
            (false, false, SbrHuffmanBook::EnvelopeLevel30Frequency),
            (true, false, SbrHuffmanBook::EnvelopeBalance30Frequency),
            (false, true, SbrHuffmanBook::NoiseLevelTime),
            (true, true, SbrHuffmanBook::NoiseBalanceTime),
        ] {
            let mut writer = BitWriter::new();
            if !time_domain {
                writer.write(0, 5);
            }
            let words = if time_domain { 2 } else { 1 };
            for _ in 0..words {
                for bit in encode_sbr_huffman(book, 0).unwrap() {
                    writer.write_bool(bit);
                }
            }
            let values = read_noise(
                &mut BitReader::new(&writer.finish()),
                &control(true, false, time_domain),
                &tables,
                balance,
            )
            .unwrap();
            assert_eq!(values, vec![vec![0, 0]]);
        }

        let channel_control = control(true, false, false);
        let mut writer = BitWriter::new();
        writer.write(0, 4); // two inverse-filtering modes
        writer.write(0, 6); // level envelope absolute value
        writer.write(0, 5); // level noise absolute value
        write_huffman(&mut writer, SbrHuffmanBook::EnvelopeLevel30Frequency);
        let (mono, shaping) = LdSbrChannelValues::parse_mono_after_prefix_usac(
            &mut BitReader::new(&writer.finish()),
            &channel_control,
            &tables,
            true,
            false,
        )
        .unwrap();
        assert_eq!(mono.inverse_filtering_modes, [0, 0]);
        assert_eq!(mono.envelopes, [vec![0]]);
        assert_eq!(mono.noise, [vec![0, 0]]);
        assert_eq!(
            shaping,
            [InterTesEnvelope {
                active: false,
                mode: 0
            }]
        );

        for coupling in [false, true] {
            let prefix = LdSbrChannelElementPrefix {
                data_extra: None,
                coupling,
                left: channel_control.clone(),
                right: Some(channel_control.clone()),
            };
            let mut writer = BitWriter::new();
            writer.write(0, 4); // left inverse-filtering modes
            if !coupling {
                writer.write(0, 4); // right inverse-filtering modes
            }
            writer.write(0, 6); // left level envelope
            if coupling {
                writer.write(0, 5); // left level noise
                write_huffman(&mut writer, SbrHuffmanBook::EnvelopeLevel30Frequency);
                writer.write(0, 5); // right balance envelope
                writer.write(0, 5); // right balance noise
                write_huffman(&mut writer, SbrHuffmanBook::EnvelopeBalance30Frequency);
            } else {
                writer.write(0, 6); // right level envelope
                for _ in 0..2 {
                    writer.write(0, 5); // level noise
                    write_huffman(&mut writer, SbrHuffmanBook::EnvelopeLevel30Frequency);
                }
            }
            let ((left, _), (right, _)) = LdSbrChannelValues::parse_stereo_after_prefix_usac(
                &mut BitReader::new(&writer.finish()),
                &prefix,
                &tables,
                true,
                false,
            )
            .unwrap();
            assert_eq!(left.inverse_filtering_modes, [0, 0]);
            assert_eq!(right.inverse_filtering_modes, [0, 0]);
            assert_eq!(left.envelopes, [vec![0]]);
            assert_eq!(right.envelopes, [vec![0]]);
            assert_eq!(left.noise, [vec![0, 0]]);
            assert_eq!(right.noise, [vec![0, 0]]);
        }

        let mut invf_only = BitWriter::new();
        invf_only.write(0, 4);
        let bits = invf_only.bits_written();
        let bytes = invf_only.finish();
        assert!(LdSbrChannelValues::parse_mono_after_prefix_usac(
            &mut BitReader::with_bit_len(&bytes, bits).unwrap(),
            &channel_control,
            &tables,
            true,
            false,
        )
        .is_err());

        let uncoupled = LdSbrChannelElementPrefix {
            data_extra: None,
            coupling: false,
            left: channel_control.clone(),
            right: Some(channel_control.clone()),
        };
        let mut invf_only = BitWriter::new();
        invf_only.write(0, 8);
        let bits = invf_only.bits_written();
        let bytes = invf_only.finish();
        assert!(LdSbrChannelValues::parse_stereo_after_prefix_usac(
            &mut BitReader::with_bit_len(&bytes, bits).unwrap(),
            &uncoupled,
            &tables,
            true,
            false,
        )
        .is_err());

        let mut left_only = BitWriter::new();
        left_only.write(0, 8); // both inverse-filtering modes
        left_only.write(0, 6); // complete left envelope
        let bits = left_only.bits_written();
        let bytes = left_only.finish();
        assert!(LdSbrChannelValues::parse_stereo_after_prefix_usac(
            &mut BitReader::with_bit_len(&bytes, bits).unwrap(),
            &uncoupled,
            &tables,
            true,
            false,
        )
        .is_err());

        let coupled = LdSbrChannelElementPrefix {
            data_extra: None,
            coupling: true,
            left: channel_control.clone(),
            right: Some(channel_control.clone()),
        };
        let mut left_only = BitWriter::new();
        left_only.write(0, 4); // shared inverse-filtering modes
        left_only.write(0, 6); // complete left envelope
        left_only.write(0, 5); // first left noise value
        write_huffman(&mut left_only, SbrHuffmanBook::EnvelopeLevel30Frequency);
        let bits = left_only.bits_written();
        let bytes = left_only.finish();
        assert!(LdSbrChannelValues::parse_stereo_after_prefix_usac(
            &mut BitReader::with_bit_len(&bytes, bits).unwrap(),
            &coupled,
            &tables,
            true,
            false,
        )
        .is_err());

        let missing_right = LdSbrChannelElementPrefix {
            data_extra: None,
            coupling: false,
            left: channel_control,
            right: None,
        };
        assert_eq!(
            LdSbrChannelValues::parse_stereo_after_prefix_usac(
                &mut BitReader::new(&[]),
                &missing_right,
                &tables,
                true,
                false,
            ),
            Err(LdSbrError::ExpectedStereo)
        );
    }

    #[test]
    fn parses_mono_invf_envelope_and_noise_values() {
        let header = LdSbrHeader {
            start_frequency: 5,
            stop_frequency: 8,
            crossover_band: 2,
            ..LdSbrHeader::default()
        };
        let tables = LdSbrFrequencyTables::from_header(&header, 44_100).unwrap();
        let control = LdSbrChannelControl {
            grid: LdSbrGrid {
                transient: false,
                amp_resolution: Some(true),
                borders: vec![0, 16],
                frequency_resolution: vec![true],
                transient_envelope: None,
                noise_borders: vec![0, 16],
            },
            envelope_time_domain: vec![false],
            noise_time_domain: vec![false],
        };
        let mut writer = BitWriter::new();
        for mode in 0..tables.noise_band_count() {
            writer.write((mode & 3) as u32, 2);
        }
        writer.write(17, 6); // absolute envelope value
        for _ in 1..tables.high_band_count() {
            writer.write_bool(false); // zero delta in EnvLevel11F
        }
        writer.write(7, 5); // absolute noise value
        for _ in 1..tables.noise_band_count() {
            writer.write_bool(false); // zero delta in EnvLevel11F
        }
        let bytes = writer.finish();
        let values = LdSbrChannelValues::parse_mono_after_prefix(
            &mut BitReader::new(&bytes),
            &control,
            &tables,
            false,
        )
        .unwrap();
        assert_eq!(values.envelopes[0][0], 17);
        assert!(values.envelopes[0][1..].iter().all(|&value| value == 0));
        assert_eq!(values.noise[0][0], 7);
        assert!(values.noise[0][1..].iter().all(|&value| value == 0));
    }

    #[test]
    fn reconstructs_frequency_and_time_deltas_with_low_high_mapping() {
        let tables = LdSbrFrequencyTables {
            master: vec![10, 11, 12, 13, 14],
            high: vec![10, 11, 12, 13, 14],
            low: vec![10, 12, 14],
            noise: vec![10, 12, 14],
        };
        let control = LdSbrChannelControl {
            grid: LdSbrGrid {
                transient: false,
                amp_resolution: None,
                borders: vec![0, 8, 16],
                frequency_resolution: vec![false, true],
                transient_envelope: None,
                noise_borders: vec![0, 8, 16],
            },
            envelope_time_domain: vec![false, true],
            noise_time_domain: vec![false, true],
        };
        let mut values = LdSbrChannelValues {
            inverse_filtering_modes: vec![0, 0],
            envelopes: vec![vec![10, 2], vec![1, 2, 3, 4]],
            noise: vec![vec![5, 1], vec![1, -1]],
        };
        let mut previous = LdSbrPreviousValues::default();
        values
            .reconstruct_deltas(&control, &tables, &mut previous)
            .unwrap();
        assert_eq!(values.envelopes, vec![vec![10, 12], vec![11, 12, 15, 16]]);
        assert_eq!(previous.envelope_high, vec![11, 12, 15, 16]);
        assert_eq!(values.noise, vec![vec![5, 6], vec![6, 5]]);
        assert_eq!(previous.noise, vec![6, 5]);
    }

    #[test]
    fn parses_harmonic_flags_and_extended_data() {
        let mut writer = BitWriter::new();
        writer.write_bool(true);
        for value in [true, false, true, true] {
            writer.write_bool(value);
        }
        writer.write_bool(true);
        writer.write(2, 4);
        writer.write(0xa5, 8);
        writer.write(0x5a, 8);
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        assert_eq!(
            read_add_harmonics(&mut reader, 4).unwrap(),
            vec![true, false, true, true]
        );
        assert_eq!(read_extended_data(&mut reader).unwrap(), vec![0xa5, 0x5a]);

        assert!(read_extended_data(&mut BitReader::new(&[]))
            .unwrap()
            .is_empty());
        assert!(read_extended_data(&mut BitReader::new(&[0]))
            .unwrap()
            .is_empty());

        let mut writer = BitWriter::new();
        writer.write_bool(true);
        writer.write(15, 4);
        writer.write(1, 8);
        for value in 0u8..16 {
            writer.write(value as u32, 8);
        }
        assert_eq!(
            read_extended_data(&mut BitReader::new(&writer.finish())).unwrap(),
            (0u8..16).collect::<Vec<_>>()
        );

        let mut writer = BitWriter::new();
        writer.write_bool(true);
        writer.write(1, 4);
        assert_eq!(
            read_extended_data(&mut BitReader::new(&writer.finish())),
            Err(LdSbrError::UnexpectedEof)
        );
    }

    #[test]
    fn parses_uncoupled_stereo_values_in_fdk_payload_order() {
        let tables = LdSbrFrequencyTables {
            master: vec![10, 11],
            high: vec![10, 11],
            low: vec![10, 11],
            noise: vec![10, 11],
        };
        let control = |resolution| LdSbrChannelControl {
            grid: LdSbrGrid {
                transient: false,
                amp_resolution: Some(true),
                borders: vec![0, 16],
                frequency_resolution: vec![resolution],
                transient_envelope: None,
                noise_borders: vec![0, 16],
            },
            envelope_time_domain: vec![false],
            noise_time_domain: vec![false],
        };
        let prefix = LdSbrChannelElementPrefix {
            data_extra: None,
            coupling: false,
            left: control(false),
            right: Some(control(true)),
        };
        let mut writer = BitWriter::new();
        writer.write(1, 2); // left invf
        writer.write(2, 2); // right invf
        writer.write(11, 6); // left envelope
        writer.write(22, 6); // right envelope
        writer.write(3, 5); // left noise
        writer.write(7, 5); // right noise
        let bytes = writer.finish();
        let (left, right) = LdSbrChannelValues::parse_stereo_after_prefix(
            &mut BitReader::new(&bytes),
            &prefix,
            &tables,
            false,
        )
        .unwrap();
        assert_eq!(left.inverse_filtering_modes, vec![1]);
        assert_eq!(right.inverse_filtering_modes, vec![2]);
        assert_eq!(left.envelopes, vec![vec![11]]);
        assert_eq!(right.envelopes, vec![vec![22]]);
        assert_eq!(left.noise, vec![vec![3]]);
        assert_eq!(right.noise, vec![vec![7]]);
    }

    #[test]
    fn stateful_frame_parser_carries_time_deltas_across_frames() {
        let header = LdSbrHeader {
            amp_resolution: true,
            start_frequency: 5,
            stop_frequency: 8,
            crossover_band: 2,
            ..LdSbrHeader::default()
        };
        let tables = LdSbrFrequencyTables::from_header(&header, 44_100).unwrap();
        let write_frame = |time_domain: bool| {
            let mut writer = BitWriter::new();
            writer.write_bool(false); // no frame header
            writer.write_bool(false); // no data_extra
            writer.write_bool(false); // FIXFIX
            writer.write(0, 2); // one envelope
            writer.write_bool(true); // current amp resolution
            writer.write_bool(true); // high frequency resolution
            writer.write_bool(time_domain); // envelope direction
            writer.write_bool(time_domain); // noise direction
            for _ in 0..tables.noise_band_count() {
                writer.write(0, 2); // invf mode
            }
            if !time_domain {
                writer.write(12, 6);
            }
            for _ in (if time_domain { 0 } else { 1 })..tables.high_band_count() {
                writer.write_bool(false); // zero Huffman delta
            }
            if !time_domain {
                writer.write(4, 5);
            }
            for _ in (if time_domain { 0 } else { 1 })..tables.noise_band_count() {
                writer.write_bool(false); // zero Huffman delta
            }
            writer.write_bool(false); // no harmonics
            writer.write_bool(false); // no extended data
            writer.finish()
        };
        let first = write_frame(false);
        let second = write_frame(true);
        for bit_len in 0..first.len() * 8 {
            let mut reader = BitReader::with_bit_len(&first, bit_len).unwrap();
            let _ = LdSbrFrameParser::new(header.clone(), 44_100, 512, false, false)
                .unwrap()
                .parse(&mut reader);
        }
        let mut parser = LdSbrFrameParser::new(header, 44_100, 512, false, false).unwrap();
        assert_eq!(parser.header().crossover_band, 2);
        let first = parser.parse(&mut BitReader::new(&first)).unwrap();
        let second = parser.parse(&mut BitReader::new(&second)).unwrap();
        assert_eq!(first.left.envelopes[0], second.left.envelopes[0]);
        assert_eq!(first.left.noise[0], second.left.noise[0]);
        assert!(!first.header_present);
        assert!(first.left_harmonics.iter().all(|&value| !value));
    }

    #[test]
    fn stateful_frame_parser_decodes_coupled_stereo_payload() {
        let header = LdSbrHeader {
            amp_resolution: true,
            start_frequency: 5,
            stop_frequency: 8,
            crossover_band: 2,
            ..LdSbrHeader::default()
        };
        let tables = LdSbrFrequencyTables::from_header(&header, 44_100).unwrap();
        let mut writer = BitWriter::new();
        writer.write_bool(false); // no frame header
        writer.write_bool(false); // no data extra
        writer.write_bool(true); // coupling
        writer.write_bool(false); // FIXFIX
        writer.write(0, 2); // one envelope
        writer.write_bool(true); // current amplitude resolution
        writer.write_bool(true); // high frequency resolution
        for _ in 0..2 {
            writer.write_bool(false); // envelope frequency direction
            writer.write_bool(false); // noise frequency direction
        }
        for _ in 0..tables.noise_band_count() {
            writer.write(1, 2); // shared inverse filtering
        }
        writer.write(9, 6); // coupled level envelope
        for _ in 1..tables.high_band_count() {
            for bit in encode_sbr_huffman(SbrHuffmanBook::EnvelopeLevel30Frequency, 0).unwrap() {
                writer.write_bool(bit);
            }
        }
        writer.write(6, 5); // coupled level noise
        for _ in 1..tables.noise_band_count() {
            for bit in encode_sbr_huffman(SbrHuffmanBook::EnvelopeLevel30Frequency, 0).unwrap() {
                writer.write_bool(bit);
            }
        }
        writer.write(6, 5); // centered envelope balance
        for _ in 1..tables.high_band_count() {
            for bit in encode_sbr_huffman(SbrHuffmanBook::EnvelopeBalance30Frequency, 0).unwrap() {
                writer.write_bool(bit);
            }
        }
        writer.write(6, 5); // centered noise balance
        for _ in 1..tables.noise_band_count() {
            for bit in encode_sbr_huffman(SbrHuffmanBook::EnvelopeBalance30Frequency, 0).unwrap() {
                writer.write_bool(bit);
            }
        }
        writer.write_bool(false); // no left harmonics
        writer.write_bool(false); // no right harmonics
        writer.write_bool(false); // no extended data

        let bits = writer.bits_written();
        let bytes = writer.finish();
        let mut reader = BitReader::with_bit_len(&bytes, bits).unwrap();
        let frame = LdSbrFrameParser::new(header.clone(), 44_100, 512, true, false)
            .unwrap()
            .parse(&mut reader)
            .unwrap();
        assert_eq!(reader.bits_read(), bits);
        assert!(frame.prefix.coupling);
        assert!(frame.right.is_some());
        assert!(frame.right_dequantized.is_some());
        assert!(frame.right_harmonics.unwrap().iter().all(|value| !value));
        for (left, right) in frame.left_dequantized.envelope_energy[0]
            .iter()
            .zip(&frame.right_dequantized.unwrap().envelope_energy[0])
        {
            assert!((left - right).abs() < 1.0e-12);
        }
        for bit_len in 0..bits {
            let mut reader = BitReader::with_bit_len(&bytes, bit_len).unwrap();
            let _ = LdSbrFrameParser::new(header.clone(), 44_100, 512, true, false)
                .unwrap()
                .parse(&mut reader);
        }

        let mut writer = BitWriter::new();
        writer.write_bool(false); // no frame header
        writer.write_bool(false); // no data extra
        writer.write_bool(false); // uncoupled stereo
        for _ in 0..2 {
            writer.write_bool(false); // FIXFIX
            writer.write(0, 2); // one envelope
            writer.write_bool(true); // current amplitude resolution
            writer.write_bool(true); // high frequency resolution
        }
        for _ in 0..2 {
            writer.write_bool(false); // envelope frequency direction
            writer.write_bool(false); // noise frequency direction
        }
        for _ in 0..2 * tables.noise_band_count() {
            writer.write(1, 2);
        }
        for absolute in [9, 10] {
            writer.write(absolute, 6);
            for _ in 1..tables.high_band_count() {
                for bit in encode_sbr_huffman(SbrHuffmanBook::EnvelopeLevel30Frequency, 0).unwrap()
                {
                    writer.write_bool(bit);
                }
            }
        }
        for absolute in [6, 7] {
            writer.write(absolute, 5);
            for _ in 1..tables.noise_band_count() {
                for bit in encode_sbr_huffman(SbrHuffmanBook::EnvelopeLevel30Frequency, 0).unwrap()
                {
                    writer.write_bool(bit);
                }
            }
        }
        writer.write_bool(false); // no left harmonics
        writer.write_bool(false); // no right harmonics
        writer.write_bool(false); // no extended data
        let bits = writer.bits_written();
        let bytes = writer.finish();
        let mut reader = BitReader::with_bit_len(&bytes, bits).unwrap();
        let frame = LdSbrFrameParser::new(header, 44_100, 512, true, false)
            .unwrap()
            .parse(&mut reader)
            .unwrap();
        assert!(!frame.prefix.coupling);
        assert!(frame.right_dequantized.is_some());
    }

    #[test]
    fn validates_eld_sbr_crc10() {
        let header = LdSbrHeader {
            amp_resolution: true,
            start_frequency: 5,
            stop_frequency: 8,
            crossover_band: 2,
            ..LdSbrHeader::default()
        };
        let tables = LdSbrFrequencyTables::from_header(&header, 44_100).unwrap();
        let mut writer = BitWriter::new();
        writer.write(0, 10); // patched CRC
        writer.write_bool(false); // no frame header
        writer.write_bool(false); // no data_extra
        writer.write_bool(false); // FIXFIX
        writer.write(0, 2);
        writer.write_bool(true);
        writer.write_bool(true);
        writer.write_bool(false); // envelope frequency direction
        writer.write_bool(false); // noise frequency direction
        for _ in 0..tables.noise_band_count() {
            writer.write(0, 2);
        }
        writer.write(12, 6);
        for _ in 1..tables.high_band_count() {
            writer.write_bool(false);
        }
        writer.write(4, 5);
        for _ in 1..tables.noise_band_count() {
            writer.write_bool(false);
        }
        writer.write_bool(false); // harmonics
        writer.write_bool(false); // extension
        let region_bits = writer.bits_written() - 10;
        let mut bytes = writer.finish();
        let crc = BitReader::new(&bytes).crc_msb(10, 10 + region_bits, 10, 0x0633) as u16;
        bytes[0] = (crc >> 2) as u8;
        bytes[1] = (bytes[1] & 0x3f) | ((crc as u8 & 3) << 6);
        let mut parser = LdSbrFrameParser::new(header.clone(), 44_100, 512, false, true).unwrap();
        assert_eq!(
            parser
                .parse(&mut BitReader::new(&bytes))
                .unwrap()
                .transmitted_crc,
            Some(crc)
        );
        let invf_payload_bit = 19;
        bytes[invf_payload_bit / 8] ^= 1 << (7 - invf_payload_bit % 8);
        let mut parser = LdSbrFrameParser::new(header, 44_100, 512, false, true).unwrap();
        assert!(matches!(
            parser.parse(&mut BitReader::new(&bytes)),
            Err(LdSbrError::CrcMismatch { .. })
        ));
    }

    #[test]
    fn dequantizes_uncoupled_and_unmaps_centered_coupling() {
        let control = LdSbrChannelControl {
            grid: LdSbrGrid {
                transient: false,
                amp_resolution: Some(true),
                borders: vec![0, 16],
                frequency_resolution: vec![true],
                transient_envelope: None,
                noise_borders: vec![0, 16],
            },
            envelope_time_domain: vec![false],
            noise_time_domain: vec![false],
        };
        let level = LdSbrChannelValues {
            inverse_filtering_modes: vec![0],
            envelopes: vec![vec![0, 1]],
            noise: vec![vec![6, 5]],
        };
        let uncoupled = level.dequantize_uncoupled(&control, false);
        assert_eq!(
            uncoupled.envelope_energy,
            vec![vec![64.0 / 1_048_576.0, 128.0 / 1_048_576.0]]
        );
        assert_eq!(uncoupled.noise_energy, vec![vec![1.0, 2.0]]);

        let balance = LdSbrChannelValues {
            inverse_filtering_modes: vec![0],
            envelopes: vec![vec![12, 12]],
            noise: vec![vec![12, 12]],
        };
        let (left, right) =
            LdSbrChannelValues::dequantize_coupled_pair(&level, &balance, &control, false).unwrap();
        assert_eq!(left.envelope_energy, right.envelope_energy);
        assert_eq!(left.noise_energy, right.noise_energy);
        assert_eq!(left.envelope_energy, uncoupled.envelope_energy);
        assert_eq!(left.noise_energy, uncoupled.noise_energy);

        let mut mismatched = balance.clone();
        mismatched.envelopes.clear();
        assert_eq!(
            LdSbrChannelValues::dequantize_coupled_pair(&level, &mismatched, &control, false,),
            Err(LdSbrError::CoupledLayoutMismatch)
        );
        mismatched = balance.clone();
        mismatched.envelopes[0].pop();
        assert_eq!(
            LdSbrChannelValues::dequantize_coupled_pair(&level, &mismatched, &control, false,),
            Err(LdSbrError::CoupledLayoutMismatch)
        );
        mismatched = balance;
        mismatched.noise[0].pop();
        assert_eq!(
            LdSbrChannelValues::dequantize_coupled_pair(&level, &mismatched, &control, false,),
            Err(LdSbrError::CoupledLayoutMismatch)
        );
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn frequency_band_counts_match_fdk_for_generated_eld_sbr_header() {
        let header = LdSbrHeader {
            amp_resolution: true,
            start_frequency: 8,
            stop_frequency: 6,
            crossover_band: 0,
            ..LdSbrHeader::default()
        };
        let rust = LdSbrFrequencyTables::from_header(&header, 88_200).unwrap();
        let mut low = 0;
        let mut high = 0;
        let mut noise = 0;
        let mut low_table = [0u8; 65];
        let mut high_table = [0u8; 65];
        assert_eq!(
            unsafe {
                fdk_aac_sys::fdk_sbr_frequency_tables_test(
                    88_200,
                    8,
                    6,
                    0,
                    2,
                    1,
                    2,
                    &mut low,
                    &mut high,
                    &mut noise,
                    low_table.as_mut_ptr(),
                    high_table.as_mut_ptr(),
                )
            },
            0
        );
        assert_eq!(rust.high_band_count(), high as usize);
        assert_eq!(rust.low_band_count(), low as usize);
        assert_eq!(rust.noise_band_count(), noise as usize);
        assert_eq!(rust.low, low_table[..=low as usize]);
        assert_eq!(rust.high, high_table[..=high as usize]);
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn eld_frequency_tables_match_fdk_across_supported_rates() {
        for sampling_rate in [
            16_000, 22_050, 24_000, 32_000, 40_000, 44_100, 48_000, 64_000, 88_200, 96_000,
        ] {
            for start_frequency in [0u8, 8, 15] {
                for stop_frequency in 0u8..=15 {
                    let mut low = 0;
                    let mut high = 0;
                    let mut noise = 0;
                    let mut low_table = [0u8; 65];
                    let mut high_table = [0u8; 65];
                    let status = unsafe {
                        fdk_aac_sys::fdk_sbr_frequency_tables_test(
                            sampling_rate,
                            start_frequency,
                            stop_frequency,
                            0,
                            2,
                            1,
                            2,
                            &mut low,
                            &mut high,
                            &mut noise,
                            low_table.as_mut_ptr(),
                            high_table.as_mut_ptr(),
                        )
                    };
                    if status != 0 {
                        continue;
                    }
                    let header = LdSbrHeader {
                        start_frequency,
                        stop_frequency,
                        crossover_band: 0,
                        ..LdSbrHeader::default()
                    };
                    let rust = LdSbrFrequencyTables::from_header(&header, sampling_rate)
                        .unwrap_or_else(|error| {
                            panic!("rate {sampling_rate}, start {start_frequency}, stop {stop_frequency}, C range {}..{}: {error:?}", high_table[0], high_table[high as usize])
                        });
                    assert_eq!(
                        rust.low,
                        low_table[..=low as usize],
                        "low rate {sampling_rate}, start {start_frequency}, stop {stop_frequency}"
                    );
                    assert_eq!(
                        rust.high,
                        high_table[..=high as usize],
                        "high rate {sampling_rate}, start {start_frequency}, stop {stop_frequency}"
                    );
                    assert_eq!(rust.noise_band_count(), noise as usize);
                }
            }
        }
    }
}
