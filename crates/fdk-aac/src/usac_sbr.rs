//! USAC-specific SBR frame controls: info header, harmonic patching, PVC and inter-TES.

use std::sync::LazyLock;

use crate::bits::{BitError, BitReader};
use crate::ld_sbr_qmf::QmfSlot;

const PVC_SOURCE: &str = include_str!(concat!(
    env!("FDK_AAC_UPSTREAM_DIR"),
    "/libSBRdec/src/pvc_dec.cpp"
));

fn parse_hex_table(name: &str, expected: usize) -> Vec<i8> {
    let start = PVC_SOURCE.find(name).expect("PVC ROM");
    let source = &PVC_SOURCE[start..];
    let body = &source[source.find('{').unwrap() + 1..source.find("};").unwrap()];
    let values: Vec<_> = body
        .split("0x")
        .skip(1)
        .map(|entry| u8::from_str_radix(&entry[..2], 16).unwrap() as i8)
        .collect();
    assert_eq!(values.len(), expected);
    values
}

static PVC_TAB1_MODE1: LazyLock<Vec<i8>> =
    LazyLock::new(|| parse_hex_table("g_3a_pvcTab1_mode1", 72));
static PVC_TAB2_MODE1: LazyLock<Vec<i8>> =
    LazyLock::new(|| parse_hex_table("g_2a_pvcTab2_mode1", 1024));
static PVC_TAB1_MODE2: LazyLock<Vec<i8>> =
    LazyLock::new(|| parse_hex_table("g_3a_pvcTab1_mode2", 54));
static PVC_TAB2_MODE2: LazyLock<Vec<i8>> =
    LazyLock::new(|| parse_hex_table("g_2a_pvcTab2_mode2", 768));

fn smoothing_coefficients(slots: usize) -> Vec<f32> {
    let name = format!("pvc_SC_{slots}");
    let start = PVC_SOURCE.find(&name).expect("PVC smoothing ROM");
    let source = &PVC_SOURCE[start..];
    let body = &source[source.find('{').unwrap() + 1..source.find("};").unwrap()];
    body.split("0x")
        .skip(1)
        .map(|entry| u32::from_str_radix(&entry[..8], 16).unwrap() as f32 / 2_147_483_648.0)
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UsacSbrFrameInfo {
    pub info_present: bool,
    pub header_present: bool,
    pub amplitude_resolution: Option<bool>,
    pub crossover_band: Option<u8>,
    pub preprocessing: Option<bool>,
    pub pvc_mode: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HarmonicSbrControl {
    pub patching_mode: bool,
    pub oversampling: bool,
    pub pitch_in_bins: Option<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PvcEnvelope {
    pub division_mode: u8,
    pub noise_shaping_mode: bool,
    pub slots_per_group: u8,
    pub ids: [u8; 16],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UsacPvcGrid {
    pub noise_position: u8,
    pub variable_length: u8,
    pub borders: Vec<i8>,
    pub pvc_borders: Vec<i8>,
    pub noise_borders: Vec<i8>,
}

impl UsacPvcGrid {
    pub fn parse(
        reader: &mut BitReader<'_>,
        previous_right_border: Option<i8>,
        previous_pvc: bool,
    ) -> Result<Self, UsacSbrError> {
        let noise_position = reader.read_u8(4)?;
        let variable_length = if reader.read_bool()? {
            let value = reader.read_u8(2)? + 1;
            if value > 3 {
                return Err(UsacSbrError::InvalidPvcGrid);
            }
            value
        } else {
            0
        };
        let left = previous_right_border.map_or(0, |border| border - 16);
        if left > 3 {
            return Err(UsacSbrError::InvalidPvcGrid);
        }
        let right = 16 + variable_length as i8;
        let borders = if noise_position == 0 {
            vec![left, right]
        } else {
            if i8::try_from(noise_position).unwrap() <= left
                || i8::try_from(noise_position).unwrap() >= right
            {
                return Err(UsacSbrError::InvalidPvcGrid);
            }
            vec![left, noise_position as i8, right]
        };
        let mut pvc_borders = borders.clone();
        pvc_borders[0] = if previous_pvc { 0 } else { left };
        *pvc_borders.last_mut().unwrap() = 16;
        Ok(Self {
            noise_position,
            variable_length,
            noise_borders: borders.clone(),
            borders,
            pvc_borders,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InterTesEnvelope {
    pub active: bool,
    pub mode: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UsacSbrError {
    Bit(BitError),
    ReservedPvcMode,
    InvalidPvcGrid,
}

impl From<BitError> for UsacSbrError {
    fn from(v: BitError) -> Self {
        Self::Bit(v)
    }
}

impl UsacSbrFrameInfo {
    pub fn parse(
        reader: &mut BitReader<'_>,
        independent: bool,
        pvc_enabled: bool,
        stereo: bool,
    ) -> Result<Self, UsacSbrError> {
        let info_present = independent || reader.read_bool()?;
        let header_present = if independent {
            true
        } else if info_present {
            reader.read_bool()?
        } else {
            false
        };
        if !info_present {
            return Ok(Self {
                info_present,
                header_present,
                amplitude_resolution: None,
                crossover_band: None,
                preprocessing: None,
                pvc_mode: 0,
            });
        }
        let amplitude_resolution = Some(reader.read_bool()?);
        let crossover_band = Some(reader.read_u8(4)?);
        let preprocessing = Some(reader.read_bool()?);
        let mut pvc_mode = if pvc_enabled { reader.read_u8(2)? } else { 0 };
        if pvc_mode > 2 {
            return Err(UsacSbrError::ReservedPvcMode);
        }
        if stereo {
            pvc_mode = 0;
        }
        Ok(Self {
            info_present,
            header_present,
            amplitude_resolution,
            crossover_band,
            preprocessing,
            pvc_mode,
        })
    }
}

impl HarmonicSbrControl {
    pub fn parse(reader: &mut BitReader<'_>) -> Result<Self, BitError> {
        let patching_mode = reader.read_bool()?;
        if patching_mode {
            Ok(Self {
                patching_mode,
                oversampling: false,
                pitch_in_bins: None,
            })
        } else {
            let oversampling = reader.read_bool()?;
            let pitch_in_bins = reader.read_bool()?.then(|| reader.read_u8(7)).transpose()?;
            Ok(Self {
                patching_mode,
                oversampling,
                pitch_in_bins,
            })
        }
    }
}

impl PvcEnvelope {
    pub fn parse(
        reader: &mut BitReader<'_>,
        pvc_mode: u8,
        independent: bool,
        previous_id: u8,
    ) -> Result<Self, UsacSbrError> {
        if !matches!(pvc_mode, 1 | 2) {
            return Err(UsacSbrError::ReservedPvcMode);
        }
        let division_mode = reader.read_u8(3)?;
        let noise_shaping_mode = reader.read_bool()?;
        let slots_per_group = match (pvc_mode, noise_shaping_mode) {
            (1, false) => 16,
            (1, true) => 4,
            (2, false) => 12,
            _ => 3,
        };
        let mut ids = [previous_id; 16];
        if division_mode <= 3 {
            let reuse = !independent && reader.read_bool()?;
            ids[0] = if reuse {
                previous_id
            } else {
                reader.read_u8(7)?
            };
            let mut slot = 1usize;
            let mut sum_length = 0;
            for _ in 0..division_mode {
                let bits = if sum_length >= 13 {
                    1
                } else if sum_length >= 11 {
                    2
                } else if sum_length >= 7 {
                    3
                } else {
                    4
                };
                let length = usize::from(reader.read_u8(bits)?);
                sum_length += length + 1;
                if sum_length >= 16 {
                    return Err(UsacSbrError::InvalidPvcGrid);
                }
                for _ in 0..length {
                    ids[slot] = ids[slot - 1];
                    slot += 1;
                }
                ids[slot] = reader.read_u8(7)?;
                slot += 1;
            }
            while slot < 16 {
                ids[slot] = ids[slot - 1];
                slot += 1;
            }
        } else {
            let exponent = division_mode - 4;
            let groups = 2usize << exponent;
            let length = 8usize >> exponent;
            let first_new = independent || reader.read_bool()?;
            ids[0] = if first_new {
                reader.read_u8(7)?
            } else {
                previous_id
            };
            for slot in 1..length {
                ids[slot] = ids[0];
            }
            for group in 1..groups {
                let start = group * length;
                let changed = reader.read_bool()?;
                ids[start] = if changed {
                    reader.read_u8(7)?
                } else {
                    ids[start - 1]
                };
                for slot in start + 1..start + length {
                    ids[slot] = ids[start];
                }
            }
        }
        Ok(Self {
            division_mode,
            noise_shaping_mode,
            slots_per_group,
            ids,
        })
    }
}

pub fn parse_inter_tes_envelopes(
    reader: &mut BitReader<'_>,
    count: usize,
) -> Result<Vec<InterTesEnvelope>, BitError> {
    (0..count)
        .map(|_| {
            let active = reader.read_bool()?;
            let mode = if active { reader.read_u8(2)? } else { 0 };
            Ok(InterTesEnvelope { active, mode })
        })
        .collect()
}

#[derive(Debug, Clone)]
pub struct PvcPredictor {
    history_db: [[f32; 3]; 16],
    history_index: usize,
    available: usize,
    previous_mode: u8,
    previous_kx: usize,
}

impl Default for PvcPredictor {
    fn default() -> Self {
        Self {
            history_db: [[-10.0; 3]; 16],
            history_index: 0,
            available: 0,
            previous_mode: 0,
            previous_kx: 0,
        }
    }
}

impl PvcPredictor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Predict grouped high-band linear energies for one 16-slot PVC frame.
    /// `qmf[slot][band]` contains complex QMF samples at the SBR rate.
    pub fn predict_frame(
        &mut self,
        mode: u8,
        slots_per_group: usize,
        rate: usize,
        kx: usize,
        ids: &[u8; 16],
        qmf: &[Vec<(f32, f32)>],
    ) -> Result<Vec<Vec<f32>>, UsacSbrError> {
        if !matches!(mode, 1 | 2) || !matches!(slots_per_group, 3 | 4 | 12 | 16) {
            return Err(UsacSbrError::ReservedPvcMode);
        }
        if self.previous_mode == 0 || self.previous_kx != kx {
            self.available = 0;
        }
        let low_width = 8 / rate;
        let high_bands = if mode == 1 { 8 } else { 6 };
        let coefficients = smoothing_coefficients(slots_per_group);
        let (tab1, tab2, boundaries, scaling): (&[i8], &[i8], [u8; 2], [i32; 4]) = if mode == 1 {
            (&PVC_TAB1_MODE1, &PVC_TAB2_MODE1, [17, 68], [8, 8, 7, 1])
        } else {
            (&PVC_TAB1_MODE2, &PVC_TAB2_MODE2, [16, 52], [7, 7, 6, 0])
        };
        let mut output = Vec::with_capacity(16);
        for time in 0..16 {
            let mut energy_db = [-10.0; 3];
            for group in 0..3 {
                let start = kx.saturating_sub((3 - group) * low_width);
                let stop = kx.saturating_sub((2 - group) * low_width);
                let mut energy = 0.0;
                for slot in time * rate..(time + 1) * rate {
                    if let Some(bands) = qmf.get(slot) {
                        for &(real, imaginary) in
                            &bands[start.min(bands.len())..stop.min(bands.len())]
                        {
                            energy += (real * real + imaginary * imaginary) / 8.0;
                        }
                    }
                }
                energy_db[group] = 10.0 * energy.max(0.1).log10();
            }
            self.history_db[self.history_index] = energy_db;
            let mut smooth = [0.0; 3];
            let usable = coefficients.len().min(self.available + 1);
            for (delay, &coefficient) in coefficients.iter().take(usable).enumerate() {
                let index = (self.history_index + 16 - delay) & 15;
                for group in 0..3 {
                    smooth[group] += coefficient * self.history_db[index][group];
                }
            }
            let id = usize::from(ids[time]);
            let class = if id < usize::from(boundaries[0]) {
                0
            } else if id < usize::from(boundaries[1]) {
                1
            } else {
                2
            };
            let mut predicted = vec![0.0; high_bands];
            for high in 0..high_bands {
                let mut db = tab2[id * high_bands + high] as f32 / 2.0f32.powi(scaling[3]);
                for low in 0..3 {
                    let coefficient = tab1[(class * 3 + low) * high_bands + high] as f32
                        / 2.0f32.powi(scaling[low]);
                    db += coefficient * smooth[low];
                }
                predicted[high] = 10.0f32.powf(db / 10.0);
            }
            output.push(predicted);
            self.history_index = (self.history_index + 1) & 15;
            self.available = (self.available + 1).min(15);
        }
        self.previous_mode = mode;
        self.previous_kx = kx;
        Ok(output)
    }

    pub fn predict_qmf_slots(
        &mut self,
        mode: u8,
        slots_per_group: usize,
        rate: usize,
        kx: usize,
        ids: &[u8; 16],
        qmf: &[QmfSlot],
    ) -> Result<Vec<Vec<f64>>, UsacSbrError> {
        let converted: Vec<_> = qmf
            .iter()
            .map(|slot| {
                slot.real
                    .iter()
                    .zip(&slot.imaginary)
                    .map(|(&real, &imaginary)| (real as f32, imaginary as f32))
                    .collect()
            })
            .collect();
        Ok(self
            .predict_frame(mode, slots_per_group, rate, kx, ids, &converted)?
            .into_iter()
            .map(|values| values.into_iter().map(f64::from).collect())
            .collect())
    }
}

pub fn apply_pvc_predicted_energies(
    qmf: &mut [QmfSlot],
    low_subband: usize,
    high_subband: usize,
    predicted: &[Vec<f64>],
    rate: usize,
) {
    if low_subband >= high_subband || predicted.is_empty() || rate == 0 {
        return;
    }
    for (time, groups) in predicted.iter().enumerate() {
        for slot_index in time * rate..((time + 1) * rate).min(qmf.len()) {
            let slot = &mut qmf[slot_index];
            for (group, &target) in groups.iter().enumerate() {
                let start = low_subband + (high_subband - low_subband) * group / groups.len();
                let stop = low_subband + (high_subband - low_subband) * (group + 1) / groups.len();
                let power = (start..stop)
                    .map(|band| {
                        slot.real[band] * slot.real[band]
                            + slot.imaginary[band] * slot.imaginary[band]
                    })
                    .sum::<f64>()
                    / (stop - start).max(1) as f64;
                let gain = (target.max(0.0) / power.max(1e-20)).sqrt().min(1e4);
                for band in start..stop {
                    slot.real[band] *= gain;
                    slot.imaginary[band] *= gain;
                }
            }
        }
    }
}

/// Apply inter-temporal envelope shaping to complex QMF samples while
/// preserving total high-band energy over the envelope.
pub fn apply_inter_tes_qmf(
    qmf: &mut [Vec<(f32, f32)>],
    start: usize,
    stop: usize,
    low_subband: usize,
    high_subband_count: usize,
    mode: u8,
) {
    if start >= stop || stop > qmf.len() || mode == 0 {
        return;
    }
    let gamma = [0.0, 1.0, 2.0, 4.0][usize::from(mode.min(3))];
    let high_stop = low_subband + high_subband_count;
    let count = stop - start;
    let mut low_power = vec![0.0; count];
    let mut high_power = vec![0.0; count];
    for (slot, bands) in qmf[start..stop].iter().enumerate() {
        low_power[slot] = bands[..low_subband.min(bands.len())]
            .iter()
            .map(|&(r, i)| r * r + i * i)
            .sum();
        high_power[slot] = bands[low_subband.min(bands.len())..high_stop.min(bands.len())]
            .iter()
            .map(|&(r, i)| r * r + i * i)
            .sum();
    }
    let total_low = low_power.iter().sum::<f32>();
    let total_high = high_power.iter().sum::<f32>();
    let mut gains: Vec<_> = low_power
        .iter()
        .map(|&power| {
            let normalized = if total_low > 0.0 {
                (power * count as f32 / total_low).sqrt()
            } else {
                1.0
            };
            (1.0 + gamma * (normalized - 1.0)).max(0.2)
        })
        .collect();
    let high_after = high_power
        .iter()
        .zip(&gains)
        .map(|(&power, &gain)| power * gain * gain)
        .sum::<f32>();
    let compensation = if total_high > 0.0 && high_after > 0.0 {
        (total_high / high_after).sqrt()
    } else {
        1.0
    };
    for (slot, gain) in qmf[start..stop].iter_mut().zip(gains.drain(..)) {
        let gain = gain * compensation;
        let band_start = low_subband.min(slot.len());
        let band_stop = high_stop.min(slot.len());
        for (real, imaginary) in &mut slot[band_start..band_stop] {
            *real *= gain;
            *imaginary *= gain;
        }
    }
}

pub fn apply_inter_tes_qmf_f64(
    qmf: &mut [QmfSlot],
    start: usize,
    stop: usize,
    low_subband: usize,
    high_subband_count: usize,
    mode: u8,
) {
    if start >= stop || stop > qmf.len() || mode == 0 {
        return;
    }
    let gamma = [0.0, 1.0, 2.0, 4.0][usize::from(mode.min(3))];
    let high_stop = low_subband + high_subband_count;
    let mut low_power = Vec::with_capacity(stop - start);
    let mut high_power = Vec::with_capacity(stop - start);
    for slot in &qmf[start..stop] {
        low_power.push(
            slot.real[..low_subband.min(slot.real.len())]
                .iter()
                .zip(&slot.imaginary)
                .map(|(&real, &imaginary)| real * real + imaginary * imaginary)
                .sum::<f64>(),
        );
        high_power.push(
            slot.real[low_subband.min(slot.real.len())..high_stop.min(slot.real.len())]
                .iter()
                .zip(&slot.imaginary[low_subband.min(slot.imaginary.len())..])
                .map(|(&real, &imaginary)| real * real + imaginary * imaginary)
                .sum::<f64>(),
        );
    }
    let total_low = low_power.iter().sum::<f64>();
    let total_high = high_power.iter().sum::<f64>();
    let mean_low = total_low / low_power.len().max(1) as f64;
    let mut gains: Vec<_> = low_power
        .iter()
        .map(|&energy| {
            (energy.max(1e-20) / mean_low.max(1e-20))
                .powf(gamma * 0.5)
                .max(0.2)
        })
        .collect();
    let shaped = gains
        .iter()
        .zip(&high_power)
        .map(|(&gain, &energy)| gain * gain * energy)
        .sum::<f64>();
    let compensation = (total_high / shaped.max(1e-20)).sqrt();
    for (slot, gain) in qmf[start..stop].iter_mut().zip(gains.drain(..)) {
        let gain = gain * compensation;
        for band in low_subband..high_stop.min(slot.real.len()).min(slot.imaginary.len()) {
            slot.real[band] *= gain;
            slot.imaginary[band] *= gain;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitWriter;

    #[test]
    fn parses_independent_sbr_info_with_pvc() {
        let mut bits = BitWriter::new();
        bits.write_bool(true);
        bits.write(5, 4);
        bits.write_bool(false);
        bits.write(2, 2);
        let info = UsacSbrFrameInfo::parse(&mut BitReader::new(&bits.finish()), true, true, false)
            .unwrap();
        assert_eq!(info.pvc_mode, 2);
        assert!(info.header_present);
    }

    #[test]
    fn parses_fixed_pvc_grid() {
        let mut bits = BitWriter::new();
        bits.write(4, 3);
        bits.write_bool(false);
        bits.write(3, 7);
        bits.write_bool(true);
        bits.write(9, 7);
        let pvc = PvcEnvelope::parse(&mut BitReader::new(&bits.finish()), 1, true, 0).unwrap();
        assert_eq!(&pvc.ids[..8], &[3; 8]);
        assert_eq!(&pvc.ids[8..], &[9; 8]);
    }

    #[test]
    fn parses_inter_tes_modes() {
        let mut bits = BitWriter::new();
        bits.write_bool(true);
        bits.write(3, 2);
        bits.write_bool(false);
        let values = parse_inter_tes_envelopes(&mut BitReader::new(&bits.finish()), 2).unwrap();
        assert_eq!(values[0].mode, 3);
        assert!(!values[1].active);
    }

    #[test]
    fn parses_pvc_noise_and_variable_hf_grid() {
        let mut bits = BitWriter::new();
        bits.write(7, 4);
        bits.write_bool(true);
        bits.write(1, 2);
        let grid = UsacPvcGrid::parse(&mut BitReader::new(&bits.finish()), Some(17), true).unwrap();
        assert_eq!(grid.borders, [1, 7, 18]);
        assert_eq!(grid.pvc_borders, [0, 7, 16]);
        assert_eq!(grid.noise_borders, [1, 7, 18]);
    }

    #[test]
    fn rejects_reserved_pvc_variable_length() {
        let mut bits = BitWriter::new();
        bits.write(0, 4);
        bits.write_bool(true);
        bits.write(3, 2);
        assert_eq!(
            UsacPvcGrid::parse(&mut BitReader::new(&bits.finish()), None, false),
            Err(UsacSbrError::InvalidPvcGrid)
        );
    }

    #[test]
    fn predicts_finite_pvc_energies_from_rom() {
        let qmf = vec![vec![(0.25, -0.25); 64]; 16];
        let predicted = PvcPredictor::new()
            .predict_frame(1, 16, 1, 32, &[0; 16], &qmf)
            .unwrap();
        assert_eq!(predicted.len(), 16);
        assert!(predicted
            .iter()
            .flatten()
            .all(|value| value.is_finite() && *value > 0.0));
    }

    #[test]
    fn inter_tes_preserves_total_high_band_energy() {
        let mut qmf = vec![vec![(1.0, 0.0); 8]; 4];
        for band in 0..4 {
            qmf[0][band].0 = 4.0;
        }
        let before = qmf
            .iter()
            .flat_map(|slot| &slot[4..8])
            .map(|&(r, i)| r * r + i * i)
            .sum::<f32>();
        apply_inter_tes_qmf(&mut qmf, 0, 4, 4, 4, 3);
        let after = qmf
            .iter()
            .flat_map(|slot| &slot[4..8])
            .map(|&(r, i)| r * r + i * i)
            .sum::<f32>();
        assert!((before - after).abs() < 1e-4);
        assert_ne!(qmf[0][4].0, qmf[1][4].0);
    }

    #[test]
    fn f64_inter_tes_preserves_high_band_energy() {
        let mut qmf = vec![
            QmfSlot {
                real: vec![1.0; 8],
                imaginary: vec![0.0; 8]
            };
            4
        ];
        qmf[0].real[..4].fill(4.0);
        let power = |slots: &[QmfSlot]| {
            slots
                .iter()
                .flat_map(|slot| slot.real[4..8].iter().zip(&slot.imaginary[4..8]))
                .map(|(&real, &imaginary)| real * real + imaginary * imaginary)
                .sum::<f64>()
        };
        let before = power(&qmf);
        apply_inter_tes_qmf_f64(&mut qmf, 0, 4, 4, 4, 3);
        assert!((before - power(&qmf)).abs() < 1e-10);
        assert_ne!(qmf[0].real[4], qmf[1].real[4]);
    }

    #[test]
    fn pvc_energy_adjustment_realizes_group_targets() {
        let mut qmf = vec![
            QmfSlot {
                real: vec![1.0; 12],
                imaginary: vec![0.0; 12]
            };
            2
        ];
        apply_pvc_predicted_energies(&mut qmf, 4, 12, &[vec![4.0, 9.0]], 2);
        for slot in &qmf {
            let first = slot.real[4..8]
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                / 4.0;
            let second = slot.real[8..12]
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                / 4.0;
            assert!((first - 4.0).abs() < 1e-10);
            assert!((second - 9.0).abs() < 1e-10);
        }
    }

    #[test]
    fn parses_dependent_frame_info_absence_headers_stereo_and_errors() {
        let info = UsacSbrFrameInfo::parse(&mut BitReader::new(&[0]), false, true, false).unwrap();
        assert!(!info.info_present);
        assert!(!info.header_present);
        assert_eq!(info.amplitude_resolution, None);

        let mut writer = BitWriter::new();
        writer.write_bool(true); // info present
        writer.write_bool(false); // header absent
        writer.write_bool(false); // amplitude resolution
        writer.write(7, 4);
        writer.write_bool(true); // preprocessing
        writer.write(2, 2); // PVC, suppressed for stereo
        let bytes = writer.finish();
        let info = UsacSbrFrameInfo::parse(&mut BitReader::new(&bytes), false, true, true).unwrap();
        assert!(info.info_present);
        assert!(!info.header_present);
        assert_eq!(info.crossover_band, Some(7));
        assert_eq!(info.pvc_mode, 0);

        let mut writer = BitWriter::new();
        writer.write_bool(false);
        writer.write(0, 4);
        writer.write_bool(false);
        writer.write(3, 2);
        assert_eq!(
            UsacSbrFrameInfo::parse(&mut BitReader::new(&writer.finish()), true, true, false,),
            Err(UsacSbrError::ReservedPvcMode)
        );
        assert!(matches!(
            UsacSbrError::from(BitError::UnexpectedEof {
                needed_bits: 1,
                remaining_bits: 0,
            }),
            UsacSbrError::Bit(_)
        ));
    }

    #[test]
    fn parses_both_harmonic_sbr_control_layouts() {
        let patched = HarmonicSbrControl::parse(&mut BitReader::new(&[0x80])).unwrap();
        assert_eq!(
            patched,
            HarmonicSbrControl {
                patching_mode: true,
                oversampling: false,
                pitch_in_bins: None,
            }
        );

        let mut writer = BitWriter::new();
        writer.write_bool(false);
        writer.write_bool(true);
        writer.write_bool(true);
        writer.write(85, 7);
        let parsed = HarmonicSbrControl::parse(&mut BitReader::new(&writer.finish())).unwrap();
        assert!(!parsed.patching_mode);
        assert!(parsed.oversampling);
        assert_eq!(parsed.pitch_in_bins, Some(85));
    }

    #[test]
    fn parses_all_pvc_division_and_noise_shaping_layouts() {
        assert_eq!(
            PvcEnvelope::parse(&mut BitReader::new(&[0]), 0, true, 0),
            Err(UsacSbrError::ReservedPvcMode)
        );

        let mut writer = BitWriter::new();
        writer.write(3, 3);
        writer.write_bool(false);
        writer.write(1, 7);
        writer.write(10, 4);
        writer.write(2, 7);
        writer.write(1, 2);
        writer.write(3, 7);
        writer.write(0, 1);
        writer.write(4, 7);
        let parsed = PvcEnvelope::parse(&mut BitReader::new(&writer.finish()), 1, true, 0).unwrap();
        assert_eq!(parsed.slots_per_group, 16);
        assert_eq!(parsed.ids[0], 1);
        assert_eq!(parsed.ids[11], 2);
        assert_eq!(parsed.ids[13], 3);
        assert_eq!(parsed.ids[14], 4);

        let mut invalid = BitWriter::new();
        invalid.write(1, 3);
        invalid.write_bool(false);
        invalid.write(1, 7);
        invalid.write(15, 4);
        assert_eq!(
            PvcEnvelope::parse(&mut BitReader::new(&invalid.finish()), 1, true, 0),
            Err(UsacSbrError::InvalidPvcGrid)
        );

        for (division, mode, noise, expected_slots) in [
            (4, 1, true, 4),
            (5, 2, false, 12),
            (6, 2, true, 3),
            (7, 1, false, 16),
        ] {
            let exponent = division - 4;
            let groups = 2usize << exponent;
            let mut writer = BitWriter::new();
            writer.write(division, 3);
            writer.write_bool(noise);
            writer.write(5, 7); // independent first ID
            for group in 1..groups {
                let changed = group % 2 == 0;
                writer.write_bool(changed);
                if changed {
                    writer.write((5 + group) as u32, 7);
                }
            }
            let parsed =
                PvcEnvelope::parse(&mut BitReader::new(&writer.finish()), mode, true, 9).unwrap();
            assert_eq!(parsed.division_mode, division as u8);
            assert_eq!(parsed.slots_per_group, expected_slots);
            assert_eq!(parsed.ids[0], 5);
        }

        let mut reused = BitWriter::new();
        reused.write(0, 3);
        reused.write_bool(true);
        reused.write_bool(true);
        let parsed =
            PvcEnvelope::parse(&mut BitReader::new(&reused.finish()), 1, false, 12).unwrap();
        assert_eq!(parsed.ids, [12; 16]);
        assert_eq!(parsed.slots_per_group, 4);
    }

    #[test]
    fn predicts_mode_two_through_qmf_slot_facade_and_validates_modes() {
        let qmf = vec![
            QmfSlot {
                real: vec![0.25; 64],
                imaginary: vec![-0.25; 64],
            };
            16
        ];
        let mut ids = [0; 16];
        ids[5] = 16;
        ids[10] = 52;
        let mut predictor = PvcPredictor::new();
        let predicted = predictor
            .predict_qmf_slots(2, 12, 1, 32, &ids, &qmf)
            .unwrap();
        assert_eq!(predicted.len(), 16);
        assert!(predicted.iter().all(|bands| bands.len() == 6));
        assert!(predicted.iter().flatten().all(|value| value.is_finite()));

        let converted = vec![vec![(0.0, 0.0); 64]; 16];
        assert_eq!(
            predictor.predict_frame(0, 16, 1, 32, &ids, &converted),
            Err(UsacSbrError::ReservedPvcMode)
        );
        assert_eq!(
            predictor.predict_frame(1, 5, 1, 32, &ids, &converted),
            Err(UsacSbrError::ReservedPvcMode)
        );
    }

    #[test]
    fn pvc_grid_and_energy_helpers_cover_noop_and_invalid_borders() {
        let mut writer = BitWriter::new();
        writer.write(0, 4);
        writer.write_bool(false);
        assert_eq!(
            UsacPvcGrid::parse(&mut BitReader::new(&writer.finish()), Some(20), false),
            Err(UsacSbrError::InvalidPvcGrid)
        );

        let mut writer = BitWriter::new();
        writer.write(1, 4);
        writer.write_bool(false);
        assert_eq!(
            UsacPvcGrid::parse(&mut BitReader::new(&writer.finish()), Some(18), false),
            Err(UsacSbrError::InvalidPvcGrid)
        );

        let original = vec![QmfSlot {
            real: vec![1.0; 4],
            imaginary: vec![0.0; 4],
        }];
        for (low, high, predicted, rate) in [
            (2, 2, vec![vec![1.0]], 1),
            (0, 2, Vec::new(), 1),
            (0, 2, vec![vec![1.0]], 0),
        ] {
            let mut qmf = original.clone();
            apply_pvc_predicted_energies(&mut qmf, low, high, &predicted, rate);
            assert_eq!(qmf, original);
        }
    }

    #[test]
    fn inter_tes_noop_and_zero_energy_paths_are_bounded() {
        let original = vec![vec![(1.0, 0.0); 4]; 2];
        for (start, stop, mode) in [(1, 1, 1), (0, 3, 1), (0, 2, 0)] {
            let mut qmf = original.clone();
            apply_inter_tes_qmf(&mut qmf, start, stop, 2, 2, mode);
            assert_eq!(qmf, original);
        }
        let mut zero = vec![vec![(0.0, 0.0); 4]; 2];
        apply_inter_tes_qmf(&mut zero, 0, 2, 2, 2, 2);
        assert!(zero.iter().flatten().all(|&(r, i)| r == 0.0 && i == 0.0));

        let original = vec![
            QmfSlot {
                real: vec![1.0; 4],
                imaginary: vec![0.0; 4],
            };
            2
        ];
        for (start, stop, mode) in [(1, 1, 1), (0, 3, 1), (0, 2, 0)] {
            let mut qmf = original.clone();
            apply_inter_tes_qmf_f64(&mut qmf, start, stop, 2, 2, mode);
            assert_eq!(qmf, original);
        }
    }

    #[test]
    fn covers_remaining_pvc_border_length_and_history_fallbacks() {
        let mut grid = BitWriter::new();
        grid.write(0, 4);
        grid.write_bool(false);
        let parsed = UsacPvcGrid::parse(&mut BitReader::new(&grid.finish()), None, false).unwrap();
        assert_eq!(parsed.borders, [0, 16]);

        let mut lengths = BitWriter::new();
        lengths.write(2, 3);
        lengths.write_bool(false);
        lengths.write(1, 7);
        lengths.write(7, 4); // cumulative length becomes 8
        lengths.write(2, 7);
        lengths.write(0, 3); // uses the 3-bit length class
        lengths.write(3, 7);
        let parsed =
            PvcEnvelope::parse(&mut BitReader::new(&lengths.finish()), 1, true, 0).unwrap();
        assert_eq!(parsed.ids[8], 2);
        assert_eq!(parsed.ids[9], 3);

        let mut reuse = BitWriter::new();
        reuse.write(4, 3);
        reuse.write_bool(false);
        reuse.write_bool(false); // reuse previous first ID
        reuse.write_bool(false); // second group unchanged
        let parsed =
            PvcEnvelope::parse(&mut BitReader::new(&reuse.finish()), 1, false, 37).unwrap();
        assert_eq!(parsed.ids, [37; 16]);

        let predicted = PvcPredictor::new()
            .predict_frame(1, 16, 1, 32, &[0; 16], &[])
            .unwrap();
        assert!(predicted.iter().flatten().all(|value| value.is_finite()));
    }
}
