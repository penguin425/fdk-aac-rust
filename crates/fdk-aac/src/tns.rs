//! AAC-LC TNS syntax parsing and f32 reference filtering.

use std::fmt;

use crate::bits::{BitError, BitReader};
use crate::ics::{IcsInfo, WindowSequence};
use crate::inverse::{FixedInverseQuantizedSpectrum, InverseQuantizedSpectrum};

#[derive(Debug, Clone, PartialEq)]
pub struct TnsData {
    pub present: bool,
    pub filters: Vec<Vec<TnsFilter>>,
}

impl TnsData {
    pub fn absent(windows: usize) -> Self {
        Self {
            present: false,
            filters: vec![Vec::new(); windows],
        }
    }

    pub fn parse_present_aac_lc(
        reader: &mut BitReader<'_>,
        ics: &IcsInfo,
    ) -> Result<Self, TnsError> {
        let windows = total_windows(ics);
        let is_long = ics.window_sequence != WindowSequence::EightShort;
        let mut filters = vec![Vec::new(); windows];

        for window_filters in &mut filters {
            let n_filt = reader.read_u8(if is_long { 2 } else { 1 })?;
            if n_filt == 0 {
                continue;
            }
            let coef_res = reader.read_u8(1)?;
            let mut next_stop_band = ics.total_sfb;

            for _ in 0..n_filt {
                let length = reader.read_u8(if is_long { 6 } else { 4 })?;
                let length = length.min(next_stop_band);
                let start_band = next_stop_band - length;
                let stop_band = next_stop_band;
                next_stop_band = start_band;

                let order = reader.read_u8(if is_long { 5 } else { 3 })?;
                if order > 20 {
                    return Err(TnsError::OrderTooLarge(order));
                }

                let mut direction = TnsDirection::Forward;
                let mut resolution = coef_res + 3;
                let mut coefficients = Vec::new();
                if order > 0 {
                    direction = if reader.read_bool()? {
                        TnsDirection::Backward
                    } else {
                        TnsDirection::Forward
                    };
                    let coef_compress = reader.read_u8(1)?;
                    resolution = coef_res + 3;
                    let bits = resolution - coef_compress;
                    for _ in 0..order {
                        coefficients.push(read_signed_tns_coef(reader, bits)?);
                    }
                }

                window_filters.push(TnsFilter {
                    start_band,
                    stop_band,
                    direction,
                    resolution,
                    coefficients,
                });
            }
        }

        Ok(Self {
            present: true,
            filters,
        })
    }

    pub fn parse_aac_lc(reader: &mut BitReader<'_>, ics: &IcsInfo) -> Result<Self, TnsError> {
        if reader.read_bool()? {
            Self::parse_present_aac_lc(reader, ics)
        } else {
            Ok(Self::absent(total_windows(ics)))
        }
    }

    pub fn parse_present_usac(
        reader: &mut BitReader<'_>,
        short: bool,
        total_sfb: u8,
    ) -> Result<Self, TnsError> {
        let windows = if short { 8 } else { 1 };
        let mut filters = vec![Vec::new(); windows];
        for window_filters in &mut filters {
            let count = reader.read_u8(if short { 1 } else { 2 })?;
            if count == 0 {
                continue;
            }
            let coefficient_resolution = reader.read_u8(1)?;
            let mut next_stop = total_sfb;
            for _ in 0..count {
                let length = reader.read_u8(if short { 4 } else { 6 })?.min(next_stop);
                let start = next_stop - length;
                let order = reader.read_u8(if short { 3 } else { 4 })?;
                let mut direction = TnsDirection::Forward;
                let mut coefficients = Vec::new();
                if order != 0 {
                    direction = if reader.read_bool()? {
                        TnsDirection::Backward
                    } else {
                        TnsDirection::Forward
                    };
                    let compress = reader.read_u8(1)?;
                    let bits = coefficient_resolution + 3 - compress;
                    for _ in 0..order {
                        coefficients.push(read_signed_tns_coef(reader, bits)?);
                    }
                }
                window_filters.push(TnsFilter {
                    start_band: start,
                    stop_band: next_stop,
                    direction,
                    resolution: coefficient_resolution + 3,
                    coefficients,
                });
                next_stop = start;
            }
        }
        Ok(Self {
            present: true,
            filters,
        })
    }

    pub fn apply_to_windows_f32(
        &self,
        windows: &mut [Vec<f32>],
        band_offsets: &[usize],
    ) -> Result<(), TnsError> {
        if windows.len() != self.filters.len() {
            return Err(TnsError::LayoutMismatch);
        }
        for (window, filters) in windows.iter_mut().zip(&self.filters) {
            for filter in filters {
                let start =
                    band_offsets[usize::from(filter.start_band).min(band_offsets.len() - 1)];
                let stop = band_offsets[usize::from(filter.stop_band).min(band_offsets.len() - 1)]
                    .min(window.len());
                if start < stop {
                    apply_lattice_synthesis_f32(&mut window[start..stop], filter);
                }
            }
        }
        Ok(())
    }

    pub fn apply_f32(
        &self,
        spectrum: &mut InverseQuantizedSpectrum,
        band_offsets: &[usize],
        max_tns_bands: usize,
    ) -> Result<(), TnsError> {
        if !self.present {
            return Ok(());
        }
        if spectrum.windows.len() != self.filters.len() {
            return Err(TnsError::LayoutMismatch);
        }
        for (window, filters) in spectrum.windows.iter_mut().zip(&self.filters) {
            for filter in filters {
                if filter.coefficients.is_empty() {
                    continue;
                }
                let start_band = (filter.start_band as usize)
                    .min(max_tns_bands)
                    .min(band_offsets.len() - 1);
                let stop_band = (filter.stop_band as usize)
                    .min(max_tns_bands)
                    .min(band_offsets.len() - 1);
                let start = band_offsets[start_band];
                let stop = band_offsets[stop_band].min(window.len());
                if stop > start {
                    apply_lattice_synthesis_f32(&mut window[start..stop], filter);
                }
            }
        }
        Ok(())
    }

    pub fn apply_fixed_bridge(
        &self,
        spectrum: &mut FixedInverseQuantizedSpectrum,
        band_offsets: &[usize],
        max_tns_bands: usize,
    ) -> Result<(), TnsError> {
        self.apply_fixed(spectrum, band_offsets, max_tns_bands)
    }

    pub fn apply_fixed(
        &self,
        spectrum: &mut FixedInverseQuantizedSpectrum,
        band_offsets: &[usize],
        max_tns_bands: usize,
    ) -> Result<(), TnsError> {
        if !self.present {
            return Ok(());
        }
        if spectrum.windows.len() != self.filters.len() {
            return Err(TnsError::LayoutMismatch);
        }
        for (window, filters) in spectrum.windows.iter_mut().zip(&self.filters) {
            for filter in filters {
                if filter.coefficients.is_empty() {
                    continue;
                }
                let start_band = (filter.start_band as usize)
                    .min(max_tns_bands)
                    .min(band_offsets.len() - 1);
                let stop_band = (filter.stop_band as usize)
                    .min(max_tns_bands)
                    .min(band_offsets.len() - 1);
                let start = band_offsets[start_band];
                let stop = band_offsets[stop_band].min(window.len());
                if stop > start {
                    apply_lattice_synthesis_fixed(&mut window[start..stop], filter);
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TnsFilter {
    pub start_band: u8,
    pub stop_band: u8,
    pub direction: TnsDirection,
    pub resolution: u8,
    pub coefficients: Vec<i8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TnsDirection {
    Forward,
    Backward,
}

pub fn apply_lattice_synthesis_f32(lines: &mut [f32], filter: &TnsFilter) {
    let coeffs = filter
        .coefficients
        .iter()
        .map(|&coef| tns_coefficient_to_f32(filter.resolution, coef))
        .collect::<Vec<_>>();
    let order = coeffs.len();
    if order == 0 || lines.is_empty() {
        return;
    }
    let mut state = vec![0.0f32; order];
    let indices: Box<dyn Iterator<Item = usize>> = match filter.direction {
        TnsDirection::Forward => Box::new(0..lines.len()),
        TnsDirection::Backward => Box::new((0..lines.len()).rev()),
    };
    for index in indices {
        let mut y = lines[index];
        for i in 0..order {
            y -= coeffs[i] * state[i];
        }
        for i in (1..order).rev() {
            state[i] = state[i - 1] + coeffs[i - 1] * y;
        }
        state[0] = y;
        lines[index] = y;
    }
}

pub fn apply_lattice_synthesis_fixed_bridge(lines: &mut [i32], filter: &TnsFilter) {
    apply_lattice_synthesis_fixed(lines, filter);
}

pub fn apply_lattice_synthesis_fixed(lines: &mut [i32], filter: &TnsFilter) {
    let coeffs = filter
        .coefficients
        .iter()
        .map(|&coef| tns_coefficient_to_q31(filter.resolution, coef))
        .collect::<Vec<_>>();
    let order = coeffs.len();
    if order == 0 || lines.is_empty() {
        return;
    }
    let mut state = vec![0i32; order];
    let indices: Box<dyn Iterator<Item = usize>> = match filter.direction {
        TnsDirection::Forward => Box::new(0..lines.len()),
        TnsDirection::Backward => Box::new((0..lines.len()).rev()),
    };
    for index in indices {
        let mut y = lines[index];
        for i in 0..order {
            y = sub_q31_saturate(y, mul_q31_saturate(coeffs[i], state[i]));
        }
        for i in (1..order).rev() {
            state[i] = add_q31_saturate(state[i - 1], mul_q31_saturate(coeffs[i - 1], y));
        }
        state[0] = y;
        lines[index] = y;
    }
}

pub fn tns_coefficient_to_f32(resolution: u8, coefficient: i8) -> f32 {
    match resolution {
        3 => TNS_COEFF_3[(coefficient + 4) as usize],
        4 => TNS_COEFF_4[(coefficient + 8) as usize],
        _ => 0.0,
    }
}

pub fn tns_coefficient_to_q31(resolution: u8, coefficient: i8) -> i32 {
    match resolution {
        3 => TNS_COEFF_3_Q31[(coefficient + 4) as usize],
        4 => TNS_COEFF_4_Q31[(coefficient + 8) as usize],
        _ => 0,
    }
}

fn mul_q31_saturate(a: i32, b: i32) -> i32 {
    let product = a as i64 * b as i64;
    ((product + (1 << 30)) >> 31).clamp(i32::MIN as i64 + 1, i32::MAX as i64) as i32
}

fn add_q31_saturate(a: i32, b: i32) -> i32 {
    (a as i64 + b as i64).clamp(i32::MIN as i64 + 1, i32::MAX as i64) as i32
}

fn sub_q31_saturate(a: i32, b: i32) -> i32 {
    (a as i64 - b as i64).clamp(i32::MIN as i64 + 1, i32::MAX as i64) as i32
}

fn read_signed_tns_coef(reader: &mut BitReader<'_>, bits: u8) -> Result<i8, TnsError> {
    let raw = reader.read_u8(bits as usize)?;
    let sign = 1u8 << (bits - 1);
    let extend = !((1u16 << bits) - 1) as i16;
    Ok(if raw & sign != 0 {
        (raw as i16 | extend) as i8
    } else {
        raw as i8
    })
}

fn total_windows(ics: &IcsInfo) -> usize {
    ics.window_group_lengths
        .iter()
        .map(|&len| len as usize)
        .sum()
}

const fn q31_to_f32(value: i32) -> f32 {
    value as f32 / 2_147_483_648.0
}

const TNS_COEFF_3: [f32; 8] = [
    q31_to_f32(0x81f1d1d4u32 as i32),
    q31_to_f32(0x9126146cu32 as i32),
    q31_to_f32(0xadb922c4u32 as i32),
    q31_to_f32(0xd438af1fu32 as i32),
    q31_to_f32(0x00000000),
    q31_to_f32(0x3789809bu32 as i32),
    q31_to_f32(0x64130dd4u32 as i32),
    q31_to_f32(0x7cca7016u32 as i32),
];

const TNS_COEFF_3_Q31: [i32; 8] = [
    0x81f1d1d4u32 as i32,
    0x9126146cu32 as i32,
    0xadb922c4u32 as i32,
    0xd438af1fu32 as i32,
    0x00000000,
    0x3789809bu32 as i32,
    0x64130dd4u32 as i32,
    0x7cca7016u32 as i32,
];

const TNS_COEFF_4: [f32; 16] = [
    q31_to_f32(0x808bc842u32 as i32),
    q31_to_f32(0x84e2e58cu32 as i32),
    q31_to_f32(0x8d6b49d1u32 as i32),
    q31_to_f32(0x99da920au32 as i32),
    q31_to_f32(0xa9c45713u32 as i32),
    q31_to_f32(0xbc9ddeb9u32 as i32),
    q31_to_f32(0xd1c2d51bu32 as i32),
    q31_to_f32(0xe87ae53du32 as i32),
    q31_to_f32(0x00000000),
    q31_to_f32(0x1b6d0060),
    q31_to_f32(0x3413a09a),
    q31_to_f32(0x4a5018b8),
    q31_to_f32(0x5f1f5ebbu32 as i32),
    q31_to_f32(0x6ed9ebbau32 as i32),
    q31_to_f32(0x79bc385fu32 as i32),
    q31_to_f32(0x7f4c7e5bu32 as i32),
];

const TNS_COEFF_4_Q31: [i32; 16] = [
    0x808bc842u32 as i32,
    0x84e2e58cu32 as i32,
    0x8d6b49d1u32 as i32,
    0x99da920au32 as i32,
    0xa9c45713u32 as i32,
    0xbc9ddeb9u32 as i32,
    0xd1c2d51bu32 as i32,
    0xe87ae53du32 as i32,
    0x00000000,
    0x1b6d0060,
    0x3413a09a,
    0x4a5018b8,
    0x5f1f5ebbu32 as i32,
    0x6ed9ebbau32 as i32,
    0x79bc385fu32 as i32,
    0x7f4c7e5bu32 as i32,
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TnsError {
    Bit(BitError),
    LayoutMismatch,
    OrderTooLarge(u8),
    TooManyFilters(u8),
}

impl From<BitError> for TnsError {
    fn from(value: BitError) -> Self {
        Self::Bit(value)
    }
}

impl fmt::Display for TnsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bit(err) => write!(f, "TNS bitstream error: {err}"),
            Self::LayoutMismatch => write!(f, "TNS layout mismatch"),
            Self::OrderTooLarge(order) => write!(f, "TNS order {order} exceeds AAC-LC limit"),
            Self::TooManyFilters(count) => {
                write!(f, "TNS filter count {count} exceeds AAC-LC limit")
            }
        }
    }
}

impl std::error::Error for TnsError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitWriter;
    use crate::ics::{IcsInfo, WindowShape};

    fn long_ics() -> IcsInfo {
        IcsInfo {
            window_sequence: WindowSequence::OnlyLong,
            window_shape: WindowShape::Sine,
            max_sfb: 4,
            total_sfb: 4,
            predictor_data_present: false,
            scale_factor_grouping: 0,
            window_group_lengths: vec![1],
            bits_read: 0,
        }
    }

    #[test]
    fn parses_absent_tns_data() {
        let mut writer = BitWriter::new();
        writer.write_bool(false);
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        let tns = TnsData::parse_aac_lc(&mut reader, &long_ics()).unwrap();
        assert!(!tns.present);
        assert_eq!(tns.filters.len(), 1);
    }

    #[test]
    fn parses_long_tns_filter() {
        let mut writer = BitWriter::new();
        writer.write_bool(true);
        writer.write(1, 2); // n_filt
        writer.write(1, 1); // coef_res => 4-bit table
        writer.write(2, 6); // length
        writer.write(2, 5); // order
        writer.write(1, 1); // backward
        writer.write(0, 1); // no compression, 4-bit coefficients
        writer.write(0b1111, 4); // -1
        writer.write(0b0001, 4); // +1
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);

        let tns = TnsData::parse_aac_lc(&mut reader, &long_ics()).unwrap();
        assert!(tns.present);
        assert_eq!(tns.filters[0][0].start_band, 2);
        assert_eq!(tns.filters[0][0].stop_band, 4);
        assert_eq!(tns.filters[0][0].direction, TnsDirection::Backward);
        assert_eq!(tns.filters[0][0].resolution, 4);
        assert_eq!(tns.filters[0][0].coefficients, vec![-1, 1]);

        let mut writer = BitWriter::new();
        writer.write(1, 2); // n_filt
        writer.write(0, 1); // 3-bit coefficient resolution
        writer.write(1, 6); // length
        writer.write(1, 5); // order
        writer.write_bool(false); // forward
        writer.write_bool(false); // no compression
        writer.write(0, 3); // coefficient
        let parsed =
            TnsData::parse_present_aac_lc(&mut BitReader::new(&writer.finish()), &long_ics())
                .unwrap();
        assert_eq!(parsed.filters[0][0].direction, TnsDirection::Forward);
    }

    #[test]
    fn applies_tns_filter_to_band_region() {
        let tns = TnsData {
            present: true,
            filters: vec![vec![TnsFilter {
                start_band: 1,
                stop_band: 2,
                direction: TnsDirection::Forward,
                resolution: 4,
                coefficients: vec![1],
            }]],
        };
        let mut spectrum = InverseQuantizedSpectrum {
            windows: vec![vec![0.0, 1.0, 1.0, 1.0, 0.0]],
        };
        tns.apply_f32(&mut spectrum, &[0, 1, 4, 5], 3).unwrap();
        assert_eq!(spectrum.windows[0][0], 0.0);
        assert_ne!(spectrum.windows[0][2], 1.0);
        assert_eq!(spectrum.windows[0][4], 0.0);
    }

    #[test]
    fn applies_tns_filter_to_fixed_band_region() {
        let tns = TnsData {
            present: true,
            filters: vec![vec![TnsFilter {
                start_band: 1,
                stop_band: 2,
                direction: TnsDirection::Forward,
                resolution: 4,
                coefficients: vec![1],
            }]],
        };
        let one = 0x2000_0000;
        let mut spectrum = FixedInverseQuantizedSpectrum {
            windows: vec![vec![0, one, one, one, 0]],
            window_exponents: vec![0],
        };
        tns.apply_fixed(&mut spectrum, &[0, 1, 4, 5], 3).unwrap();
        assert_eq!(spectrum.windows[0][0], 0);
        assert_ne!(spectrum.windows[0][2], one);
        assert_eq!(spectrum.windows[0][4], 0);
    }

    #[test]
    fn fixed_lattice_synthesis_tracks_f32_reference() {
        let filter = TnsFilter {
            start_band: 0,
            stop_band: 1,
            direction: TnsDirection::Forward,
            resolution: 4,
            coefficients: vec![1, -1],
        };
        let mut fixed = vec![0x2000_0000, 0x1000_0000, -0x1800_0000, 0x0800_0000];
        let mut float = fixed
            .iter()
            .map(|&value| value as f32 / 2_147_483_648.0)
            .collect::<Vec<_>>();

        apply_lattice_synthesis_fixed(&mut fixed, &filter);
        apply_lattice_synthesis_f32(&mut float, &filter);

        for (actual, expected) in fixed.iter().zip(float) {
            let expected = (expected * 2_147_483_648.0).round() as i32;
            assert!(
                (*actual as i64 - expected as i64).abs() <= 2_048,
                "actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn parses_zero_order_long_filter_and_rejects_large_order() {
        let mut writer = BitWriter::new();
        writer.write(1, 2);
        writer.write(0, 1);
        writer.write(63, 6);
        writer.write(0, 5);
        let parsed =
            TnsData::parse_present_aac_lc(&mut BitReader::new(&writer.finish()), &long_ics())
                .unwrap();
        assert_eq!(parsed.filters[0][0].start_band, 0);
        assert_eq!(parsed.filters[0][0].stop_band, 4);
        assert!(parsed.filters[0][0].coefficients.is_empty());

        let mut writer = BitWriter::new();
        writer.write(1, 2);
        writer.write(0, 1);
        writer.write(1, 6);
        writer.write(21, 5);
        assert_eq!(
            TnsData::parse_present_aac_lc(&mut BitReader::new(&writer.finish()), &long_ics()),
            Err(TnsError::OrderTooLarge(21))
        );
    }

    #[test]
    fn parses_empty_short_aac_and_usac_windows() {
        let mut short = long_ics();
        short.window_sequence = WindowSequence::EightShort;
        short.window_group_lengths = vec![3, 5];
        let aac = TnsData::parse_present_aac_lc(&mut BitReader::new(&[0]), &short).unwrap();
        assert_eq!(aac.filters, vec![Vec::new(); 8]);
        let usac = TnsData::parse_present_usac(&mut BitReader::new(&[0]), true, 15).unwrap();
        assert_eq!(usac.filters, vec![Vec::new(); 8]);

        let mut writer = BitWriter::new();
        writer.write(1, 2); // one filter
        writer.write(0, 1); // resolution 3
        writer.write(1, 6); // length
        writer.write(0, 4); // zero order
        let usac =
            TnsData::parse_present_usac(&mut BitReader::new(&writer.finish()), false, 15).unwrap();
        assert_eq!(usac.filters[0][0].start_band, 14);
        assert!(usac.filters[0][0].coefficients.is_empty());
    }

    #[test]
    fn parses_usac_compressed_forward_and_backward_coefficients() {
        let mut writer = BitWriter::new();
        writer.write(2, 2); // two filters
        writer.write(1, 1); // resolution 4
        writer.write(2, 6);
        writer.write(1, 4);
        writer.write_bool(false);
        writer.write(1, 1); // compressed to 3 bits
        writer.write(0b111, 3); // -1
        writer.write(1, 6);
        writer.write(1, 4);
        writer.write_bool(true);
        writer.write(0, 1);
        writer.write(1, 4);
        let parsed =
            TnsData::parse_present_usac(&mut BitReader::new(&writer.finish()), false, 4).unwrap();
        assert_eq!(parsed.filters[0].len(), 2);
        assert_eq!(parsed.filters[0][0].direction, TnsDirection::Forward);
        assert_eq!(parsed.filters[0][0].coefficients, [-1]);
        assert_eq!(parsed.filters[0][1].direction, TnsDirection::Backward);
        assert_eq!(parsed.filters[0][1].coefficients, [1]);
    }

    #[test]
    fn applies_window_api_in_both_directions_and_validates_layout() {
        let filters = vec![
            TnsFilter {
                start_band: 0,
                stop_band: 1,
                direction: TnsDirection::Forward,
                resolution: 3,
                coefficients: vec![1],
            },
            TnsFilter {
                start_band: 1,
                stop_band: 2,
                direction: TnsDirection::Backward,
                resolution: 4,
                coefficients: vec![-1],
            },
        ];
        let tns = TnsData {
            present: true,
            filters: vec![filters],
        };
        assert_eq!(
            tns.apply_to_windows_f32(&mut [], &[0, 2, 4]),
            Err(TnsError::LayoutMismatch)
        );
        let mut windows = [vec![1.0; 4]];
        tns.apply_to_windows_f32(&mut windows, &[0, 2, 4]).unwrap();
        assert!(windows[0].iter().all(|value| value.is_finite()));
        assert_ne!(windows[0], vec![1.0; 4]);
    }

    #[test]
    fn absent_empty_and_outside_filters_are_noops() {
        let mut float = InverseQuantizedSpectrum {
            windows: vec![vec![1.0; 4]],
        };
        TnsData::absent(1)
            .apply_f32(&mut float, &[0, 4], 1)
            .unwrap();
        assert_eq!(float.windows[0], vec![1.0; 4]);
        let mut fixed = FixedInverseQuantizedSpectrum {
            windows: vec![vec![1; 4]],
            window_exponents: vec![0],
        };
        TnsData::absent(1)
            .apply_fixed_bridge(&mut fixed, &[0, 4], 1)
            .unwrap();
        assert_eq!(fixed.windows[0], vec![1; 4]);

        let tns = TnsData {
            present: true,
            filters: vec![vec![TnsFilter {
                start_band: 1,
                stop_band: 1,
                direction: TnsDirection::Forward,
                resolution: 4,
                coefficients: Vec::new(),
            }]],
        };
        tns.apply_f32(&mut float, &[0, 4], 1).unwrap();
        tns.apply_fixed(&mut fixed, &[0, 4], 1).unwrap();
        assert_eq!(float.windows[0], vec![1.0; 4]);
        assert_eq!(fixed.windows[0], vec![1; 4]);
    }

    #[test]
    fn coefficient_tables_and_lattice_boundaries_are_total() {
        assert_eq!(q31_to_f32(std::hint::black_box(0)), 0.0);
        assert_eq!(q31_to_f32(std::hint::black_box(1 << 30)), 0.5);
        assert_eq!(q31_to_f32(std::hint::black_box(i32::MIN)), -1.0);
        for coefficient in -4..=3 {
            assert_eq!(
                tns_coefficient_to_f32(3, coefficient),
                tns_coefficient_to_q31(3, coefficient) as f32 / 2_147_483_648.0
            );
        }
        for coefficient in -8..=7 {
            assert_eq!(
                tns_coefficient_to_f32(4, coefficient),
                tns_coefficient_to_q31(4, coefficient) as f32 / 2_147_483_648.0
            );
        }
        assert_eq!(tns_coefficient_to_f32(2, 0), 0.0);
        assert_eq!(tns_coefficient_to_q31(5, 0), 0);
        let empty_filter = TnsFilter {
            start_band: 0,
            stop_band: 0,
            direction: TnsDirection::Forward,
            resolution: 4,
            coefficients: Vec::new(),
        };
        apply_lattice_synthesis_f32(&mut [1.0], &empty_filter);
        apply_lattice_synthesis_fixed_bridge(&mut [1], &empty_filter);
        apply_lattice_synthesis_fixed(
            &mut [],
            &TnsFilter {
                coefficients: vec![1],
                ..empty_filter
            },
        );
        let mut backward = [1 << 20, 1 << 20, 1 << 20];
        apply_lattice_synthesis_fixed(
            &mut backward,
            &TnsFilter {
                direction: TnsDirection::Backward,
                coefficients: vec![1],
                ..empty_filter
            },
        );
        assert_ne!(backward, [1 << 20; 3]);
    }

    #[test]
    fn q31_helpers_saturate_and_signed_reader_extends() {
        assert_eq!(add_q31_saturate(i32::MAX, 1), i32::MAX);
        assert_eq!(add_q31_saturate(i32::MIN + 1, -1), i32::MIN + 1);
        assert_eq!(sub_q31_saturate(i32::MAX, -1), i32::MAX);
        assert_eq!(sub_q31_saturate(i32::MIN + 1, 1), i32::MIN + 1);
        assert!(mul_q31_saturate(i32::MAX, i32::MAX) > 0);
        assert_eq!(
            read_signed_tns_coef(&mut BitReader::new(&[0b1110_0000]), 3).unwrap(),
            -1
        );
        assert_eq!(
            read_signed_tns_coef(&mut BitReader::new(&[0b0110_0000]), 3).unwrap(),
            3
        );
    }

    #[test]
    fn parser_eof_layout_and_error_messages_are_covered() {
        assert!(matches!(
            TnsData::parse_aac_lc(&mut BitReader::new(&[]), &long_ics()),
            Err(TnsError::Bit(BitError::UnexpectedEof { .. }))
        ));
        let present = TnsData {
            present: true,
            filters: vec![Vec::new()],
        };
        assert_eq!(
            present.apply_f32(
                &mut InverseQuantizedSpectrum {
                    windows: Vec::new()
                },
                &[0],
                0
            ),
            Err(TnsError::LayoutMismatch)
        );
        assert_eq!(
            present.apply_fixed(
                &mut FixedInverseQuantizedSpectrum {
                    windows: Vec::new(),
                    window_exponents: Vec::new()
                },
                &[0],
                0
            ),
            Err(TnsError::LayoutMismatch)
        );
        let bit = BitError::UnexpectedEof {
            needed_bits: 1,
            remaining_bits: 0,
        };
        assert_eq!(TnsError::from(bit.clone()), TnsError::Bit(bit));
        for error in [
            TnsError::LayoutMismatch,
            TnsError::OrderTooLarge(21),
            TnsError::TooManyFilters(4),
        ] {
            assert!(!error.to_string().is_empty());
        }
        assert!(TnsError::Bit(BitError::UnexpectedEof {
            needed_bits: 1,
            remaining_bits: 0
        })
        .to_string()
        .starts_with("TNS bitstream error:"));
    }
}
