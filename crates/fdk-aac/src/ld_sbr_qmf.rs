//! Floating-point reference port of FDK's 32-band SBR QMF analysis bank.

use std::sync::LazyLock;

use crate::fixed_fft::{fixed_dct_iv_64, fixed_dst_iv_64};

use crate::ld_sbr::{
    LdSbrChannelControl, LdSbrDequantizedChannel, LdSbrError, LdSbrFrame, LdSbrFrequencyTables,
};
use crate::sbr::{UsacPvcSbrFrame, UsacSbrMonoFrame, UsacSbrStereoFrame};
use crate::usac_sbr::{apply_inter_tes_qmf_f64, apply_pvc_predicted_energies, PvcPredictor};

const ROM: &str = include_str!(concat!(
    env!("FDK_AAC_UPSTREAM_DIR"),
    "/libFDK/src/FDK_tools_rom.cpp"
));
const SBR_ROM: &str = include_str!(concat!(
    env!("FDK_AAC_UPSTREAM_DIR"),
    "/libSBRdec/src/sbr_rom.cpp"
));
const CHANNELS: usize = 32;
const POLYPHASE: usize = 5;
const PCM_ENERGY_FLOOR: f64 = 1.0 / 18_014_398_509_481_984.0;
const SMOOTHING_RATIOS: [f64; 4] = [
    0.666_666_666_666_66,
    0.365_163_834_270_84,
    0.146_994_335_208_35,
    0.031_830_500_937_51,
];

static PROTOTYPE: LazyLock<Vec<f64>> =
    LazyLock::new(|| parse_fixed_array("const FIXP_PFT qmf_pfilt640[]"));
static PROTOTYPE_24: LazyLock<Vec<f64>> =
    LazyLock::new(|| parse_fixed_array("const FIXP_PFT qmf_pfilt240[]"));
static PHASE_COS: LazyLock<Vec<f64>> =
    LazyLock::new(|| parse_fixed_array("const FIXP_QTW qmf_phaseshift_cos32[]"));
static PHASE_SIN: LazyLock<Vec<f64>> =
    LazyLock::new(|| parse_fixed_array("const FIXP_QTW qmf_phaseshift_sin32[]"));
static PHASE_COS_16: LazyLock<Vec<f64>> =
    LazyLock::new(|| parse_fixed_array("const FIXP_QTW qmf_phaseshift_cos16[]"));
static PHASE_SIN_16: LazyLock<Vec<f64>> =
    LazyLock::new(|| parse_fixed_array("const FIXP_QTW qmf_phaseshift_sin16[]"));
static PHASE_COS_24: LazyLock<Vec<f64>> =
    LazyLock::new(|| parse_fixed_array("const FIXP_QTW qmf_phaseshift_cos24[]"));
static PHASE_SIN_24: LazyLock<Vec<f64>> =
    LazyLock::new(|| parse_fixed_array("const FIXP_QTW qmf_phaseshift_sin24[]"));
#[cfg(test)]
static PHASE_COS_64: LazyLock<Vec<f64>> =
    LazyLock::new(|| parse_fixed_array("const FIXP_QTW qmf_phaseshift_cos64[]"));
#[cfg(test)]
static PHASE_SIN_64: LazyLock<Vec<f64>> =
    LazyLock::new(|| parse_fixed_array("const FIXP_QTW qmf_phaseshift_sin64[]"));
static CLDFB_PROTOTYPE_32: LazyLock<Vec<f64>> = LazyLock::new(|| {
    parse_float_macro_array("const FIXP_PFT qmf_cldfb_320", "QTCFLLD(")
        .into_iter()
        // FDK stores the CLDFB prototype with one coefficient headroom bit;
        // its fixed-point FIR and analysis scaling contribute another /2.
        .map(|coefficient| quantize_q15_f32(coefficient as f32 * 0.5) * 0.5)
        .collect()
});
static CLDFB_PROTOTYPE_64: LazyLock<Vec<f64>> = LazyLock::new(|| {
    parse_float_macro_array("const FIXP_PFT qmf_cldfb_640", "QTCFLLD(")
        .into_iter()
        .map(|coefficient| quantize_q15_f32(coefficient as f32 * 0.5) * 0.5)
        .collect()
});
static CLDFB_PHASE_COS_32_ANALYSIS: LazyLock<Vec<f64>> = LazyLock::new(|| {
    parse_float_macro_array("const FIXP_QTW qmf_phaseshift_cos32_cldfb_ana", "QTCFLLDT(")
        .into_iter()
        .map(|coefficient| quantize_q15_f32(coefficient as f32))
        .collect()
});
static CLDFB_PHASE_COS_32_SYNTHESIS: LazyLock<Vec<f64>> = LazyLock::new(|| {
    parse_float_macro_array("const FIXP_QTW qmf_phaseshift_cos32_cldfb_syn", "QTCFLLDT(")
        .into_iter()
        .map(|coefficient| quantize_q15_f32(coefficient as f32))
        .collect()
});
static CLDFB_PHASE_COS_64: LazyLock<Vec<f64>> = LazyLock::new(|| {
    parse_float_macro_array("const FIXP_QTW qmf_phaseshift_cos64_cldfb", "QTCFLLDT(")
        .into_iter()
        .map(|coefficient| quantize_q15_f32(coefficient as f32))
        .collect()
});
static CLDFB_PHASE_SIN_64: LazyLock<Vec<f64>> = LazyLock::new(|| {
    parse_float_macro_array("const FIXP_QTW qmf_phaseshift_sin64_cldfb", "QTCFLLDT(")
        .into_iter()
        .map(|coefficient| quantize_q15_f32(coefficient as f32))
        .collect()
});
static CLDFB_PHASE_SIN_32: LazyLock<Vec<f64>> = LazyLock::new(|| {
    parse_float_macro_array("const FIXP_QTW qmf_phaseshift_sin32_cldfb", "QTCFLLDT(")
        .into_iter()
        .map(|coefficient| quantize_q15_f32(coefficient as f32))
        .collect()
});
static RANDOM_PHASE: LazyLock<Vec<[f64; 2]>> = LazyLock::new(|| {
    let declaration = "const FIXP_SGL FDK_sbrDecoder_sbr_randomPhase";
    let start = SBR_ROM.find(declaration).unwrap();
    let body_start = SBR_ROM[start..].find('{').unwrap() + start;
    let body_end = SBR_ROM[body_start..].find("};").unwrap() + body_start;
    SBR_ROM[body_start..body_end]
        .lines()
        .filter(|line| line.contains("{FL2FXCONST_SGL"))
        .map(|line| {
            let mut fields = line
                .trim()
                .trim_start_matches('{')
                .trim_end_matches(|ch| matches!(ch, '}' | ',' | ';'))
                .split(',');
            let parse = |field: &str| {
                let field = field.trim();
                if field == "MAXVAL_SGL" {
                    32_767.0 / 32_768.0
                } else {
                    field
                        .trim_start_matches("FL2FXCONST_SGL(")
                        .split('f')
                        .next()
                        .unwrap()
                        .parse::<f64>()
                        .unwrap()
                }
            };
            [parse(fields.next().unwrap()), parse(fields.next().unwrap())]
        })
        .collect()
});

fn parse_fixed_array(declaration: &str) -> Vec<f64> {
    let start = ROM.find(declaration).unwrap();
    let body_start = ROM[start..].find('{').unwrap() + start;
    let body_end = ROM[body_start..].find("};").unwrap() + body_start;
    ROM[body_start..body_end]
        .split("0x")
        .skip(1)
        .filter_map(|token| {
            let hex = token
                .chars()
                .take_while(|ch| ch.is_ascii_hexdigit())
                .collect::<String>();
            (hex.len() == 8).then(|| {
                let raw = u32::from_str_radix(&hex, 16).unwrap() as i32;
                raw as f64 / 2_147_483_648.0
            })
        })
        .collect()
}

fn parse_float_macro_array(declaration: &str, macro_name: &str) -> Vec<f64> {
    let start = ROM.find(declaration).unwrap();
    let body_start = ROM[start..].find('{').unwrap() + start;
    let body_end = ROM[body_start..].find("};").unwrap() + body_start;
    ROM[body_start..body_end]
        .split(macro_name)
        .skip(1)
        .map(|token| token.split(')').next().unwrap().parse::<f64>().unwrap())
        .collect()
}

fn quantize_q15_f32(value: f32) -> f64 {
    ((value * 32_768.0).round().clamp(-32_768.0, 32_767.0) / 32_768.0) as f64
}

#[derive(Debug, Clone, PartialEq)]
pub struct QmfSlot {
    pub real: Vec<f64>,
    pub imaginary: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SbrPatch {
    pub source_start_band: u8,
    pub target_start_band: u8,
    pub band_count: u8,
}

pub fn derive_patches(
    tables: &LdSbrFrequencyTables,
    sampling_frequency: u32,
) -> Result<Vec<SbrPatch>, LdSbrError> {
    let master = &tables.master;
    let lsb = master[0] as i32;
    let high_start = tables.high[0] as i32;
    let xover_offset = high_start - lsb;
    let usb = *tables.high.last().unwrap() as i32;
    if lsb - 1 < 4 {
        return Err(LdSbrError::InvalidFrequencyRange);
    }
    let desired = ((2_048_000u64 * 2 / sampling_frequency as u64 + 1) >> 1) as i32;
    let mut desired_border = closest_master(master, desired, true);
    let mut source_start = 1 + xover_offset;
    let mut target_stop = lsb + xover_offset;
    let mut patches = Vec::new();
    while target_stop < usb {
        if patches.len() > 6 {
            return Err(LdSbrError::InvalidFrequencyRange);
        }
        let mut count = desired_border - target_stop;
        if count >= lsb - source_start {
            let distance = (target_stop - source_start) & !1;
            count = lsb - (target_stop - distance);
            count = closest_master(master, target_stop + count, false) - target_stop;
        }
        let distance = (count + target_stop - lsb + 1) & !1;
        if count > 0 {
            patches.push(SbrPatch {
                source_start_band: (target_stop - distance) as u8,
                target_start_band: target_stop as u8,
                band_count: count as u8,
            });
            target_stop += count;
        }
        source_start = 1;
        if desired_border - target_stop < 3 {
            desired_border = usb;
        }
    }
    if patches.len() > 1 && patches.last().unwrap().band_count < 3 {
        patches.pop();
    }
    if patches.len() > 6 {
        return Err(LdSbrError::InvalidFrequencyRange);
    }
    Ok(patches)
}

fn closest_master(master: &[u8], goal: i32, upward: bool) -> i32 {
    if upward {
        master
            .iter()
            .copied()
            .find(|&value| value as i32 >= goal)
            .unwrap_or(*master.last().unwrap()) as i32
    } else {
        master
            .iter()
            .copied()
            .rev()
            .find(|&value| value as i32 <= goal)
            .unwrap_or(master[0]) as i32
    }
}

pub fn apply_patches(slots: &mut [QmfSlot], patches: &[SbrPatch]) -> Result<(), QmfError> {
    for slot in slots {
        if slot.real.len() < 64 {
            slot.real.resize(64, 0.0);
        }
        if slot.imaginary.len() < 64 {
            slot.imaginary.resize(64, 0.0);
        }
        for patch in patches {
            for offset in 0..patch.band_count as usize {
                let source = patch.source_start_band as usize + offset;
                let target = patch.target_start_band as usize + offset;
                if source >= CHANNELS || target >= 64 {
                    return Err(QmfError::InvalidPatchBand { source, target });
                }
                slot.real[target] = slot.real[source];
                slot.imaginary[target] = slot.imaginary[source];
            }
        }
    }
    Ok(())
}

pub fn apply_inverse_filtered_patches(
    slots: &mut [QmfSlot],
    patches: &[SbrPatch],
    noise_borders: &[u8],
    modes: &[u8],
    previous_modes: &mut Vec<u8>,
    previous_bandwidths: &mut Vec<f64>,
    history: &mut [[(f64, f64); 2]],
) -> Result<(), QmfError> {
    let mut degree_alias = [0.0; 64];
    apply_inverse_filtered_patches_mode(
        slots,
        patches,
        noise_borders,
        modes,
        previous_modes,
        previous_bandwidths,
        history,
        false,
        &mut degree_alias,
    )
}

#[allow(clippy::too_many_arguments)]
fn apply_inverse_filtered_patches_mode(
    slots: &mut [QmfSlot],
    patches: &[SbrPatch],
    noise_borders: &[u8],
    modes: &[u8],
    previous_modes: &mut Vec<u8>,
    previous_bandwidths: &mut Vec<f64>,
    history: &mut [[(f64, f64); 2]],
    low_power: bool,
    degree_alias: &mut [f64; 64],
) -> Result<(), QmfError> {
    if modes.len() + 1 != noise_borders.len() {
        return Err(QmfError::InverseFilteringLayoutMismatch);
    }
    previous_modes.resize(modes.len(), 0);
    previous_bandwidths.resize(modes.len(), 0.0);
    let bandwidths = smoothed_inverse_filter_bandwidths(modes, previous_modes, previous_bandwidths);
    for slot in slots.iter_mut() {
        slot.real.resize(64, 0.0);
        slot.imaginary.resize(64, 0.0);
    }
    let low_band_slots = slots
        .iter()
        .map(|slot| {
            (
                slot.real[..CHANNELS].to_vec(),
                slot.imaginary[..CHANNELS].to_vec(),
            )
        })
        .collect::<Vec<_>>();

    degree_alias.fill(0.0);
    if low_power {
        let mut reflection = [0.0f64; CHANNELS];
        for source in 0..CHANNELS {
            let mut series = Vec::with_capacity(slots.len() + 2);
            series.extend(history[source].map(|sample| sample.0));
            series.extend(low_band_slots.iter().map(|slot| slot.0[source]));
            let mut r01 = 0.0;
            let mut r11 = 0.0;
            for samples in series.windows(2) {
                r01 += samples[1] * samples[0];
                r11 += samples[0] * samples[0];
            }
            reflection[source] = if r11 > 1.0e-30 {
                (-r01 / r11).clamp(-1.0 + f64::EPSILON, 1.0)
            } else {
                0.0
            };
        }
        for band in 2..CHANNELS {
            let current = reflection[band];
            let below = reflection[band - 1];
            let below_two = reflection[band - 2];
            let partial = (1.0 - below * below).clamp(0.0, 1.0);
            if band & 1 == 0 && current < 0.0 {
                if below < 0.0 {
                    degree_alias[band] = 1.0;
                    if below_two > 0.0 {
                        degree_alias[band - 1] = partial;
                    }
                } else if below_two > 0.0 {
                    degree_alias[band] = partial;
                }
            } else if band & 1 == 1 && current > 0.0 {
                if below > 0.0 {
                    degree_alias[band] = 1.0;
                    if below_two < 0.0 {
                        degree_alias[band - 1] = partial;
                    }
                } else if below_two < 0.0 {
                    degree_alias[band] = partial;
                }
            }
        }
    }
    for patch in patches {
        for offset in 0..patch.band_count as usize {
            let source = patch.source_start_band as usize + offset;
            let target = patch.target_start_band as usize + offset;
            if source >= CHANNELS || target >= 64 {
                return Err(QmfError::InvalidPatchBand { source, target });
            }
            let noise_band = noise_borders
                .windows(2)
                .position(|border| target >= border[0] as usize && target < border[1] as usize)
                .ok_or(QmfError::InverseFilteringLayoutMismatch)?;
            let bandwidth = bandwidths[noise_band];
            let mut series = Vec::with_capacity(slots.len() + 2);
            series.extend(history[source]);
            series.extend(
                low_band_slots
                    .iter()
                    .map(|slot| (slot.0[source], slot.1[source])),
            );
            let (a0, a1) = if low_power {
                let real = series.iter().map(|sample| sample.0).collect::<Vec<_>>();
                let (a0, a1) = real_lpc2(&real);
                ((a0, 0.0), (a1, 0.0))
            } else {
                complex_lpc2(&series)
            };
            for (index, slot) in slots.iter_mut().enumerate() {
                let current = series[index + 2];
                let prior_1 = complex_mul(a0, series[index + 1]);
                let prior_2 = complex_mul(a1, series[index]);
                slot.real[target] =
                    current.0 + bandwidth * prior_1.0 + bandwidth * bandwidth * prior_2.0;
                slot.imaginary[target] = if low_power {
                    0.0
                } else {
                    current.1 + bandwidth * prior_1.1 + bandwidth * bandwidth * prior_2.1
                };
            }
            if low_power && offset != 0 {
                degree_alias[target] = degree_alias[source];
            }
        }
    }
    for source in 0..CHANNELS {
        if slots.len() >= 2 {
            history[source] = [
                (
                    low_band_slots[slots.len() - 2].0[source],
                    low_band_slots[slots.len() - 2].1[source],
                ),
                (
                    low_band_slots[slots.len() - 1].0[source],
                    low_band_slots[slots.len() - 1].1[source],
                ),
            ];
        }
    }
    previous_modes.clone_from(&modes.to_vec());
    previous_bandwidths.clone_from(&bandwidths);
    Ok(())
}

fn real_lpc2(series: &[f64]) -> (f64, f64) {
    let mut r11 = 0.0;
    let mut r22 = 0.0;
    let mut r12 = 0.0;
    let mut p1 = 0.0;
    let mut p2 = 0.0;
    for n in 2..series.len() {
        let x = series[n];
        let x1 = series[n - 1];
        let x2 = series[n - 2];
        r11 += x1 * x1;
        r22 += x2 * x2;
        r12 += x1 * x2;
        p1 += x * x1;
        p2 += x * x2;
    }
    let determinant = r11 * r22 - r12 * r12;
    if determinant <= 1.0e-30 {
        return (0.0, 0.0);
    }
    let a0 = (r12 * p2 - p1 * r22) / determinant;
    let a1 = (r12 * p1 - p2 * r11) / determinant;
    if a0.abs() >= 4.0 || a1.abs() >= 4.0 {
        (0.0, 0.0)
    } else {
        (a0, a1)
    }
}

fn smoothed_inverse_filter_bandwidths(
    modes: &[u8],
    previous_modes: &[u8],
    previous_bandwidths: &[f64],
) -> Vec<f64> {
    let mut bandwidths = Vec::with_capacity(modes.len());
    for index in 0..modes.len() {
        let target = match modes[index] {
            1 if previous_modes[index] == 0 => 0.60,
            1 => 0.75,
            2 => 0.90,
            3 => 0.98,
            _ if previous_modes[index] == 1 => 0.60,
            _ => 0.0,
        };
        let old = previous_bandwidths[index];
        let smoothed = if target < old {
            0.75 * target + 0.25 * old
        } else {
            0.90625 * target + 0.09375 * old
        };
        bandwidths.push(if smoothed < 0.015625 {
            0.0
        } else {
            smoothed.min(0.99609375)
        });
    }
    bandwidths
}

fn complex_lpc2(series: &[(f64, f64)]) -> ((f64, f64), (f64, f64)) {
    let mut r11 = 0.0;
    let mut r22 = 0.0;
    let mut r12 = (0.0, 0.0);
    let mut p1 = (0.0, 0.0);
    let mut p2 = (0.0, 0.0);
    for n in 2..series.len() {
        let x = series[n];
        let x1 = series[n - 1];
        let x2 = series[n - 2];
        r11 += complex_norm(x1);
        r22 += complex_norm(x2);
        r12 = complex_add(r12, complex_mul(x1, complex_conj(x2)));
        p1 = complex_add(p1, complex_mul(x, complex_conj(x1)));
        p2 = complex_add(p2, complex_mul(x, complex_conj(x2)));
    }
    let determinant = r11 * r22 - complex_norm(r12);
    if determinant <= 1.0e-20 {
        return ((0.0, 0.0), (0.0, 0.0));
    }
    let a0_num = complex_sub(complex_mul(complex_conj(r12), p2), complex_scale(p1, r22));
    let a1_num = complex_sub(complex_mul(r12, p1), complex_scale(p2, r11));
    let a0 = complex_scale(a0_num, 1.0 / determinant);
    let a1 = complex_scale(a1_num, 1.0 / determinant);
    if complex_norm(a0) >= 16.0 || complex_norm(a1) >= 16.0 {
        ((0.0, 0.0), (0.0, 0.0))
    } else {
        (a0, a1)
    }
}

fn complex_add(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 + b.0, a.1 + b.1)
}
fn complex_sub(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 - b.0, a.1 - b.1)
}
fn complex_scale(a: (f64, f64), scale: f64) -> (f64, f64) {
    (a.0 * scale, a.1 * scale)
}
fn complex_conj(a: (f64, f64)) -> (f64, f64) {
    (a.0, -a.1)
}
fn complex_mul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}
fn complex_norm(a: (f64, f64)) -> f64 {
    a.0 * a.0 + a.1 * a.1
}

pub fn apply_envelope_gains(
    slots: &mut [QmfSlot],
    control: &LdSbrChannelControl,
    tables: &LdSbrFrequencyTables,
    values: &LdSbrDequantizedChannel,
) -> Result<(), QmfError> {
    let mut state = vec![1.0; 64];
    apply_envelope_gains_limited(slots, control, tables, values, 3, false, &mut state)
}

pub fn apply_envelope_gains_limited(
    slots: &mut [QmfSlot],
    control: &LdSbrChannelControl,
    tables: &LdSbrFrequencyTables,
    values: &LdSbrDequantizedChannel,
    limiter_gains: u8,
    smoothing: bool,
    previous_gains: &mut Vec<f64>,
) -> Result<(), QmfError> {
    let low = tables.high[0] as usize;
    let high = *tables.high.last().unwrap() as usize;
    apply_envelope_gains_with_limiter_borders(
        slots,
        control,
        tables,
        values,
        limiter_gains,
        smoothing,
        previous_gains,
        &[low, high],
        false,
        None,
    )
}

fn apply_envelope_gains_with_limiter_borders(
    slots: &mut [QmfSlot],
    control: &LdSbrChannelControl,
    tables: &LdSbrFrequencyTables,
    values: &LdSbrDequantizedChannel,
    limiter_gains: u8,
    smoothing: bool,
    previous_gains: &mut Vec<f64>,
    limiter_borders: &[usize],
    previous_attack_first: bool,
    clip_ratios: Option<&mut Vec<[f64; 64]>>,
) -> Result<(), QmfError> {
    apply_envelope_gains_with_limiter_borders_mode(
        slots,
        control,
        tables,
        values,
        limiter_gains,
        smoothing,
        previous_gains,
        limiter_borders,
        previous_attack_first,
        clip_ratios,
        false,
        &[0.0; 64],
        &[],
    )
}

#[allow(clippy::too_many_arguments)]
fn apply_envelope_gains_with_limiter_borders_mode(
    slots: &mut [QmfSlot],
    control: &LdSbrChannelControl,
    tables: &LdSbrFrequencyTables,
    values: &LdSbrDequantizedChannel,
    limiter_gains: u8,
    smoothing: bool,
    previous_gains: &mut Vec<f64>,
    limiter_borders: &[usize],
    previous_attack_first: bool,
    mut clip_ratios: Option<&mut Vec<[f64; 64]>>,
    low_power: bool,
    degree_alias: &[f64; 64],
    harmonics: &[bool],
) -> Result<(), QmfError> {
    if values.envelope_energy.len() != control.grid.envelope_count() {
        return Err(QmfError::EnvelopeLayoutMismatch);
    }
    if limiter_gains > 3 {
        return Err(QmfError::EnvelopeLayoutMismatch);
    }
    let startup = previous_gains.is_empty();
    previous_gains.resize(64, 1.0);
    let limiter_factor =
        [0.501_193_202_5, 1.0, 1.995_262_315, f64::INFINITY][limiter_gains as usize];
    if let Some(ratios) = clip_ratios.as_deref_mut() {
        ratios.resize(control.grid.envelope_count(), [1.0; 64]);
    }
    for envelope in 0..control.grid.envelope_count() {
        let start_slot = control.grid.borders[envelope] as usize;
        let stop_slot = control.grid.borders[envelope + 1] as usize;
        if stop_slot > slots.len() || start_slot >= stop_slot {
            return Err(QmfError::EnvelopeLayoutMismatch);
        }
        let borders = if control.grid.frequency_resolution[envelope] {
            &tables.high
        } else {
            &tables.low
        };
        if values.envelope_energy[envelope].len() + 1 != borders.len() {
            return Err(QmfError::EnvelopeLayoutMismatch);
        }
        let mut estimated_energy = [0.0; 64];
        let mut target_energy = [0.0; 64];
        for band in 0..borders.len() - 1 {
            let start_band = borders[band] as usize;
            let stop_band = borders[band + 1] as usize;
            let mut energy = 0.0;
            for slot in &slots[start_slot..stop_slot] {
                for qmf_band in start_band..stop_band {
                    energy += slot.real[qmf_band] * slot.real[qmf_band];
                    if !low_power {
                        energy += slot.imaginary[qmf_band] * slot.imaginary[qmf_band];
                    }
                }
            }
            let target = values.envelope_energy[envelope][band].max(0.0);
            let slot_count = (stop_slot - start_slot) as f64;
            let band_width = (stop_band - start_band) as f64;
            // FDK carries a +1 energy exponent when the imaginary QMF part is
            // absent, preserving the energy represented by a conjugate pair.
            let mean = energy * if low_power { 2.0 } else { 1.0 } / (slot_count * band_width);
            for qmf_band in start_band..stop_band {
                estimated_energy[qmf_band] = mean;
                target_energy[qmf_band] = target;
            }
        }
        for limiter in limiter_borders.windows(2) {
            let start_band = limiter[0];
            let stop_band = limiter[1];
            let reference_sum = target_energy[start_band..stop_band].iter().sum::<f64>();
            let estimated_sum = estimated_energy[start_band..stop_band].iter().sum::<f64>();
            let maximum_power_gain = if limiter_factor.is_infinite() {
                f64::INFINITY
            } else if estimated_sum > 1.0e-30 {
                reference_sum / estimated_sum * limiter_factor
            } else {
                0.0
            };
            let mut power_gains = Vec::with_capacity(stop_band - start_band);
            for qmf_band in start_band..stop_band {
                // calcSubbandGain adds the energy of one integer PCM LSB.
                // Core PCM is normalized by 32768 and the analysis QMF by
                // another 4096 in the Rust path: (32768 * 4096)^-2.
                let requested =
                    target_energy[qmf_band] / (estimated_energy[qmf_band] + PCM_ENERGY_FLOOR);
                let limited = requested.min(maximum_power_gain);
                if let Some(ratios) = clip_ratios.as_deref_mut() {
                    ratios[envelope][qmf_band] = if requested > 1.0e-30 {
                        limited / requested
                    } else {
                        1.0
                    };
                }
                power_gains.push(limited);
            }
            if low_power {
                let use_alias_reduction = (start_band..stop_band)
                    .map(|qmf_band| {
                        tables
                            .high
                            .windows(2)
                            .position(|border| {
                                qmf_band >= border[0] as usize && qmf_band < border[1] as usize
                            })
                            .and_then(|sfb| harmonics.get(sfb))
                            .is_none_or(|enabled| !*enabled)
                    })
                    .collect::<Vec<_>>();
                reduce_aliasing_power_gains(
                    start_band,
                    &mut power_gains,
                    &estimated_energy,
                    degree_alias,
                    &use_alias_reduction,
                );
            }
            let adjusted_sum = power_gains
                .iter()
                .enumerate()
                .map(|(offset, gain)| gain * estimated_energy[start_band + offset])
                .sum::<f64>();
            // FDK compensates energy lost by limiting, but caps the boost at
            // +4 dB so a sparse/empty transposed band cannot explode.
            let boost = if adjusted_sum > 1.0e-30 {
                (reference_sum / adjusted_sum).min(2.511_886_432)
            } else {
                1.0
            };
            for (relative_slot, slot) in slots[start_slot..stop_slot].iter_mut().enumerate() {
                for (offset, qmf_band) in (start_band..stop_band).enumerate() {
                    let gain = (power_gains[offset] * boost).sqrt();
                    let attack = control.grid.transient_envelope == Some(envelope)
                        || (previous_attack_first && envelope == 0);
                    let applied_gain = if !low_power && smoothing && !attack && !startup {
                        let ratio = SMOOTHING_RATIOS.get(relative_slot).copied().unwrap_or(0.0);
                        ratio * previous_gains[qmf_band] + (1.0 - ratio) * gain
                    } else {
                        gain
                    };
                    slot.real[qmf_band] *= applied_gain;
                    if low_power {
                        slot.imaginary[qmf_band] = 0.0;
                    } else {
                        slot.imaginary[qmf_band] *= applied_gain;
                    }
                }
            }
            for (offset, qmf_band) in (start_band..stop_band).enumerate() {
                previous_gains[qmf_band] = (power_gains[offset] * boost).sqrt();
            }
        }
    }
    Ok(())
}

fn reduce_aliasing_power_gains(
    start_band: usize,
    gains: &mut [f64],
    estimated_energy: &[f64; 64],
    degree_alias: &[f64; 64],
    use_alias_reduction: &[bool],
) {
    let mut groups = Vec::new();
    let mut open = None;
    for relative in 0..gains.len().saturating_sub(1) {
        let band = start_band + relative;
        if degree_alias[band + 1] != 0.0 && use_alias_reduction[relative] {
            if open.is_none() {
                open = Some(relative);
            } else if open.is_some_and(|start| start + 3 == relative) {
                groups.push((open.take().unwrap(), relative + 1));
            }
        } else if let Some(start) = open.take() {
            let stop = if use_alias_reduction[relative] {
                relative + 1
            } else {
                relative
            };
            if stop > start {
                groups.push((start, stop));
            }
        }
    }
    if let Some(start) = open {
        groups.push((start, gains.len()));
    }

    for (start, stop) in groups {
        let original = (start..stop)
            .map(|relative| estimated_energy[start_band + relative])
            .sum::<f64>();
        let amplified = (start..stop)
            .map(|relative| gains[relative] * estimated_energy[start_band + relative])
            .sum::<f64>();
        if original <= 1.0e-30 || amplified <= 1.0e-30 {
            continue;
        }
        let group_gain = amplified / original;
        for relative in start..stop {
            let band = start_band + relative;
            let alpha = degree_alias[band]
                .max(degree_alias.get(band + 1).copied().unwrap_or(0.0))
                .clamp(0.0, 1.0);
            gains[relative] = alpha * group_gain + (1.0 - alpha) * gains[relative];
        }
        let modified = (start..stop)
            .map(|relative| gains[relative] * estimated_energy[start_band + relative])
            .sum::<f64>();
        if modified > 1.0e-30 {
            let compensation = amplified / modified;
            for gain in &mut gains[start..stop] {
                *gain *= compensation;
            }
        }
    }
}

fn derive_limiter_borders(
    tables: &LdSbrFrequencyTables,
    patches: &[SbrPatch],
    limiter_bands: u8,
) -> Result<Vec<usize>, QmfError> {
    if limiter_bands > 3 || tables.low.len() < 2 {
        return Err(QmfError::EnvelopeLayoutMismatch);
    }
    let low = tables.low[0] as usize;
    let high = *tables.low.last().unwrap() as usize;
    if limiter_bands == 0 {
        return Ok(vec![low, high]);
    }
    let patch_borders = patches
        .iter()
        .skip(1)
        .map(|patch| patch.target_start_band as usize)
        .chain(std::iter::once(high))
        .collect::<Vec<_>>();
    let mut borders = tables
        .low
        .iter()
        .map(|&band| band as usize)
        .collect::<Vec<_>>();
    borders.extend(patch_borders.iter().copied());
    borders.sort_unstable();
    borders.dedup();
    let density = [1.0, 1.2, 2.0, 3.0][limiter_bands as usize];
    let mut index = 1;
    while index < borders.len() {
        let too_close = density * (borders[index] as f64 / borders[index - 1] as f64).log2() < 0.49;
        if too_close {
            let upper_is_patch = patch_borders.contains(&borders[index]);
            let lower_is_patch = patch_borders.contains(&borders[index - 1]);
            if !upper_is_patch {
                borders.remove(index);
                continue;
            }
            if !lower_is_patch && index > 1 {
                borders.remove(index - 1);
                continue;
            }
        }
        index += 1;
    }
    Ok(borders)
}

pub fn apply_noise_and_harmonics(
    slots: &mut [QmfSlot],
    control: &LdSbrChannelControl,
    tables: &LdSbrFrequencyTables,
    values: &LdSbrDequantizedChannel,
    harmonics: &[bool],
    random_state: &mut u32,
    harmonic_phase: &mut u8,
    previous_harmonic_bands: &mut Vec<bool>,
) -> Result<(), QmfError> {
    let low = tables.high[0] as usize;
    let high = *tables.high.last().unwrap() as usize;
    apply_noise_and_harmonics_with_limiter_borders(
        slots,
        control,
        tables,
        values,
        harmonics,
        random_state,
        harmonic_phase,
        previous_harmonic_bands,
        &[low, high],
        None,
        false,
        false,
        None,
    )
}

fn apply_noise_and_harmonics_with_limiter_borders(
    slots: &mut [QmfSlot],
    control: &LdSbrChannelControl,
    tables: &LdSbrFrequencyTables,
    values: &LdSbrDequantizedChannel,
    harmonics: &[bool],
    random_state: &mut u32,
    harmonic_phase: &mut u8,
    previous_harmonic_bands: &mut Vec<bool>,
    limiter_borders: &[usize],
    clip_ratios: Option<&[[f64; 64]]>,
    smoothing: bool,
    previous_attack_first: bool,
    previous_noise_levels: Option<&mut Vec<f64>>,
) -> Result<(), QmfError> {
    apply_noise_and_harmonics_with_limiter_borders_mode(
        slots,
        control,
        tables,
        values,
        harmonics,
        random_state,
        harmonic_phase,
        previous_harmonic_bands,
        limiter_borders,
        clip_ratios,
        smoothing,
        previous_attack_first,
        previous_noise_levels,
        false,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn apply_noise_and_harmonics_with_limiter_borders_mode(
    slots: &mut [QmfSlot],
    control: &LdSbrChannelControl,
    tables: &LdSbrFrequencyTables,
    values: &LdSbrDequantizedChannel,
    harmonics: &[bool],
    random_state: &mut u32,
    harmonic_phase: &mut u8,
    previous_harmonic_bands: &mut Vec<bool>,
    limiter_borders: &[usize],
    clip_ratios: Option<&[[f64; 64]]>,
    smoothing: bool,
    previous_attack_first: bool,
    mut previous_noise_levels: Option<&mut Vec<f64>>,
    low_power: bool,
    eld_grid: bool,
) -> Result<(), QmfError> {
    if values.noise_energy.len() != control.grid.noise_envelope_count()
        || harmonics.len() != tables.high_band_count()
    {
        return Err(QmfError::EnvelopeLayoutMismatch);
    }
    previous_harmonic_bands.resize(64, false);
    let mut harmonic_start = [usize::MAX; 64];
    let mut harmonic_sfb_start = [usize::MAX; 64];
    let mut current_harmonic_bands = vec![false; 64];
    for (sfb, &enabled) in harmonics.iter().enumerate() {
        if enabled {
            let band = (tables.high[sfb] as usize + tables.high[sfb + 1] as usize) / 2;
            let start = if previous_harmonic_bands[band] {
                0
            } else {
                control.grid.transient_envelope.unwrap_or(0)
            };
            harmonic_start[band] = start;
            harmonic_sfb_start[tables.high[sfb] as usize..tables.high[sfb + 1] as usize]
                .fill(start);
            current_harmonic_bands[band] = true;
        }
    }
    let low_band = tables.noise[0] as usize;
    let high_band = *tables.noise.last().unwrap() as usize;
    let noise_startup = previous_noise_levels
        .as_ref()
        .is_none_or(|levels| levels.is_empty());
    if let Some(levels) = previous_noise_levels.as_deref_mut() {
        levels.resize(64, 0.0);
    }
    let mut component_boosts = vec![[1.0f64; 64]; control.grid.envelope_count()];
    for envelope in 0..control.grid.envelope_count() {
        let start_slot = control.grid.borders[envelope] as usize;
        let stop_slot = control.grid.borders[envelope + 1] as usize;
        let noise_envelope = control
            .grid
            .noise_borders
            .windows(2)
            .position(|border| start_slot >= border[0] as usize && start_slot < border[1] as usize)
            .ok_or(QmfError::EnvelopeLayoutMismatch)?;
        let attack = control.grid.transient_envelope == Some(envelope);
        for limiter in limiter_borders.windows(2) {
            let mut reference = 0.0;
            let mut adjusted = 0.0;
            for qmf_band in limiter[0]..limiter[1] {
                let target = target_envelope_energy(control, tables, values, start_slot, qmf_band)?;
                reference += target;
                let signal = slots[start_slot..stop_slot]
                    .iter()
                    .map(|slot| {
                        (if low_power { 2.0 } else { 1.0 }) * slot.real[qmf_band].powi(2)
                            + if low_power {
                                0.0
                            } else {
                                slot.imaginary[qmf_band].powi(2)
                            }
                    })
                    .sum::<f64>()
                    / (stop_slot - start_slot) as f64;
                let noise_band = tables
                    .noise
                    .windows(2)
                    .position(|border| {
                        qmf_band >= border[0] as usize && qmf_band < border[1] as usize
                    })
                    .ok_or(QmfError::InverseFilteringLayoutMismatch)?;
                let quotient = values.noise_energy[noise_envelope][noise_band].max(0.0);
                let noise_clip = clip_ratios
                    .and_then(|ratios| ratios.get(envelope))
                    .map(|ratios| ratios[qmf_band])
                    .unwrap_or(1.0);
                let harmonic_present = envelope >= harmonic_sfb_start[qmf_band];
                let signal_ratio = if harmonic_present {
                    quotient / (1.0 + quotient)
                } else if attack {
                    1.0
                } else {
                    1.0 / (1.0 + quotient)
                };
                adjusted += signal * signal_ratio;
                if envelope >= harmonic_start[qmf_band] {
                    adjusted += target / (1.0 + quotient);
                } else if !attack {
                    adjusted += target * quotient / (1.0 + quotient) * noise_clip;
                }
            }
            let boost = if adjusted > 1.0e-30 {
                (reference / adjusted).min(2.511_886_432)
            } else {
                2.511_886_432
            };
            component_boosts[envelope][limiter[0]..limiter[1]].fill(boost);
        }
    }
    for (slot_index, slot) in slots.iter_mut().enumerate() {
        let noise_envelope = control
            .grid
            .noise_borders
            .windows(2)
            .position(|border| slot_index >= border[0] as usize && slot_index < border[1] as usize)
            .ok_or(QmfError::EnvelopeLayoutMismatch)?;
        let envelope = control
            .grid
            .borders
            .windows(2)
            .position(|border| slot_index >= border[0] as usize && slot_index < border[1] as usize)
            .ok_or(QmfError::EnvelopeLayoutMismatch)?;
        let suppress_for_attack = control.grid.transient_envelope == Some(envelope);
        let smooth_attack = suppress_for_attack || (previous_attack_first && envelope == 0);
        let relative_slot = slot_index - control.grid.borders[envelope] as usize;
        for qmf_band in low_band..high_band {
            let noise_band = tables
                .noise
                .windows(2)
                .position(|border| qmf_band >= border[0] as usize && qmf_band < border[1] as usize)
                .ok_or(QmfError::EnvelopeLayoutMismatch)?;
            let quotient = values.noise_energy[noise_envelope][noise_band].max(0.0);
            let target = target_envelope_energy(control, tables, values, slot_index, qmf_band)?;
            let amplitude_boost = component_boosts[envelope][qmf_band].sqrt();
            let noise_clip = clip_ratios
                .and_then(|ratios| ratios.get(envelope))
                .map(|ratios| ratios[qmf_band])
                .unwrap_or(1.0);
            let harmonic_present = envelope >= harmonic_sfb_start[qmf_band];
            let signal_gain = if harmonic_present {
                (quotient / (1.0 + quotient)).sqrt()
            } else if suppress_for_attack {
                1.0
            } else {
                (1.0 / (1.0 + quotient)).sqrt()
            } * amplitude_boost;
            let current_noise_gain =
                (target * quotient / (1.0 + quotient) * noise_clip).sqrt() * amplitude_boost;
            let noise_gain = if !low_power && smoothing && !smooth_attack && !noise_startup {
                let ratio = SMOOTHING_RATIOS.get(relative_slot).copied().unwrap_or(0.0);
                let previous = previous_noise_levels
                    .as_ref()
                    .map(|levels| levels[qmf_band])
                    .unwrap_or(0.0);
                ratio * previous + (1.0 - ratio) * current_noise_gain
            } else {
                current_noise_gain
            };
            *random_state = (*random_state + 1) & 511;
            let phase = RANDOM_PHASE[*random_state as usize];
            slot.real[qmf_band] *= signal_gain;
            if low_power {
                slot.imaginary[qmf_band] = 0.0;
            } else {
                slot.imaginary[qmf_band] *= signal_gain;
            }
            let suppress_for_harmonic = envelope >= harmonic_start[qmf_band];
            if !suppress_for_attack && !suppress_for_harmonic {
                slot.real[qmf_band] += phase[0] * noise_gain;
                if !low_power {
                    slot.imaginary[qmf_band] += phase[1] * noise_gain;
                }
            }
            if slot_index + 1 == control.grid.borders[envelope + 1] as usize {
                if let Some(levels) = previous_noise_levels.as_deref_mut() {
                    levels[qmf_band] = current_noise_gain;
                }
            }
        }
    }
    for (slot_index, slot) in slots.iter_mut().enumerate() {
        let envelope = control
            .grid
            .borders
            .windows(2)
            .position(|border| slot_index >= border[0] as usize && slot_index < border[1] as usize)
            .ok_or(QmfError::EnvelopeLayoutMismatch)?;
        let phase = *harmonic_phase & 3;
        *harmonic_phase = (phase + 1) & 3;
        let mut amplitudes = [0.0f64; 64];
        for qmf_band in low_band..high_band {
            if envelope < harmonic_start[qmf_band] {
                continue;
            }
            let noise_envelope = control
                .grid
                .noise_borders
                .windows(2)
                .position(|border| {
                    slot_index >= border[0] as usize && slot_index < border[1] as usize
                })
                .ok_or(QmfError::EnvelopeLayoutMismatch)?;
            let noise_band = tables
                .noise
                .windows(2)
                .position(|border| qmf_band >= border[0] as usize && qmf_band < border[1] as usize)
                .ok_or(QmfError::EnvelopeLayoutMismatch)?;
            let quotient = values.noise_energy[noise_envelope][noise_band].max(0.0);
            let target = target_envelope_energy(control, tables, values, slot_index, qmf_band)?;
            let amplitude =
                (target / (1.0 + quotient)).sqrt() * component_boosts[envelope][qmf_band].sqrt();
            amplitudes[qmf_band] = amplitude;
            if !low_power {
                match phase {
                    0 => slot.real[qmf_band] += amplitude,
                    1 if qmf_band & 1 == 0 => slot.imaginary[qmf_band] += amplitude,
                    1 => slot.imaginary[qmf_band] -= amplitude,
                    2 => slot.real[qmf_band] -= amplitude,
                    3 if qmf_band & 1 == 0 => slot.imaginary[qmf_band] -= amplitude,
                    _ => slot.imaginary[qmf_band] += amplitude,
                }
            }
        }
        if low_power {
            if phase & 1 == 0 {
                let sign = if phase == 0 { 1.0 } else { -1.0 };
                for qmf_band in low_band..high_band {
                    slot.real[qmf_band] += sign * amplitudes[qmf_band];
                }
            } else if eld_grid {
                // The ELD real CLDFB rotates an imaginary sinusoid into both
                // adjacent real bands with the prototype-specific phase.
                let phase_sign = if phase == 1 { 1.0 } else { -1.0 };
                for qmf_band in low_band..high_band {
                    let amplitude = amplitudes[qmf_band];
                    if amplitude == 0.0 {
                        continue;
                    }
                    let coefficient = if qmf_band & 1 == 0 {
                        0.124_518_315_453_913_9
                    } else {
                        0.112_376_785_932_502_8
                    };
                    if qmf_band > 0 {
                        slot.real[qmf_band - 1] += phase_sign * coefficient * amplitude;
                    }
                    if qmf_band + 1 < 64 {
                        slot.real[qmf_band + 1] -= phase_sign * coefficient * amplitude;
                    }
                }
            } else {
                // In the real QMF, the quadrature phase is represented by the
                // same small neighbouring-band leakage as adjustTimeSlotLC.
                let phase_sign = if phase == 1 { 1.0 } else { -1.0 };
                for qmf_band in low_band..high_band {
                    let amplitude = amplitudes[qmf_band];
                    if amplitude == 0.0 {
                        continue;
                    }
                    if qmf_band > 0 {
                        let parity = if (qmf_band - 1) & 1 == 0 { 1.0 } else { -1.0 };
                        slot.real[qmf_band - 1] += phase_sign * parity * 0.008_15 * amplitude;
                    }
                    if qmf_band + 1 < 64 {
                        let parity = if (qmf_band + 1) & 1 == 0 { 1.0 } else { -1.0 };
                        slot.real[qmf_band + 1] -= phase_sign * parity * 0.008_15 * amplitude;
                    }
                }
            }
        }
    }
    *previous_harmonic_bands = current_harmonic_bands;
    Ok(())
}

fn target_envelope_energy(
    control: &LdSbrChannelControl,
    tables: &LdSbrFrequencyTables,
    values: &LdSbrDequantizedChannel,
    slot: usize,
    qmf_band: usize,
) -> Result<f64, QmfError> {
    let envelope = control
        .grid
        .borders
        .windows(2)
        .position(|border| slot >= border[0] as usize && slot < border[1] as usize)
        .ok_or(QmfError::EnvelopeLayoutMismatch)?;
    let borders = if control.grid.frequency_resolution[envelope] {
        &tables.high
    } else {
        &tables.low
    };
    let sfb = borders
        .windows(2)
        .position(|border| qmf_band >= border[0] as usize && qmf_band < border[1] as usize)
        .ok_or(QmfError::EnvelopeLayoutMismatch)?;
    Ok(values.envelope_energy[envelope][sfb])
}

#[derive(Debug, Clone)]
pub struct LdSbrQmfAnalysis {
    channels: usize,
    cldfb: bool,
    low_power: bool,
    states: Vec<f64>,
}

impl Default for LdSbrQmfAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

impl LdSbrQmfAnalysis {
    pub fn new() -> Self {
        Self::new_with_channels(CHANNELS).unwrap()
    }

    pub fn new_with_channels(channels: usize) -> Result<Self, QmfError> {
        if !matches!(channels, 16 | 24 | 32 | 64) {
            return Err(QmfError::UnsupportedChannelCount(channels));
        }
        Ok(Self {
            channels,
            cldfb: false,
            low_power: false,
            states: vec![0.0; 2 * POLYPHASE * channels],
        })
    }

    pub fn new_cldfb_32() -> Self {
        Self::new_cldfb(32).unwrap()
    }

    pub fn new_cldfb(channels: usize) -> Result<Self, QmfError> {
        if !matches!(channels, 32 | 64) {
            return Err(QmfError::UnsupportedChannelCount(channels));
        }
        Ok(Self {
            channels,
            cldfb: true,
            low_power: false,
            states: vec![0.0; 2 * POLYPHASE * channels],
        })
    }

    pub fn set_low_power(&mut self, enabled: bool) {
        self.low_power = enabled;
    }

    pub fn process_frame(&mut self, samples: &[f64]) -> Result<Vec<QmfSlot>, QmfError> {
        if samples.len() % self.channels != 0 {
            return Err(QmfError::InvalidSampleCount(samples.len()));
        }
        samples
            .chunks_exact(self.channels)
            .map(|slot| self.process_slot(slot))
            .collect()
    }

    pub fn process_slot(&mut self, samples: &[f64]) -> Result<QmfSlot, QmfError> {
        let channels = self.channels;
        if samples.len() != channels {
            return Err(QmfError::InvalidSampleCount(samples.len()));
        }
        self.states[9 * channels..10 * channels].copy_from_slice(samples);
        let mut time = vec![0.0; 2 * channels];
        let normalized_pcm = self.states.iter().all(|sample| sample.abs() <= 1.0);
        let mut fixed_cldfb64_time = None;
        let mut filter_index = 0usize;
        let filter_stride = if channels == 24 { 5 } else { 320 / channels };
        if self.cldfb {
            let prototype: &[f64] = if channels == 64 {
                &CLDFB_PROTOTYPE_64
            } else {
                &CLDFB_PROTOTYPE_32
            };
            let mut fixed_time = [0i32; 128];
            for k in 0..2 * channels {
                let mut sum = 0.0;
                let mut fixed_accumulator = 0i32;
                for p in 0..POLYPHASE {
                    sum += prototype[filter_index + p] * self.states[k + 2 * channels * p];
                    if channels == 64 {
                        let coefficient = (prototype[filter_index + p] * 65_536.0).round() as i32;
                        let state = if normalized_pcm {
                            (self.states[k + 2 * channels * p] * 32_768.0).round() as i32
                        } else {
                            self.states[k + 2 * channels * p].round() as i32
                        };
                        fixed_accumulator =
                            fixed_accumulator.wrapping_add(coefficient.wrapping_mul(state));
                    }
                }
                // The C path accumulates five Q15 coefficient × 16-bit PCM
                // products in FIXP_DBL and left-shifts the result once. In
                // this normalized domain its exact quantum is 2^-17.
                sum = (sum * 131_072.0).round() / 131_072.0;
                time[2 * channels - 1 - k] = sum;
                if channels == 64 {
                    fixed_time[2 * channels - 1 - k] = fixed_accumulator.wrapping_shl(1);
                }
                filter_index += POLYPHASE;
            }
            if channels == 64 {
                fixed_cldfb64_time = Some(fixed_time);
            }
        } else {
            let prototype = if channels == 24 {
                &PROTOTYPE_24
            } else {
                &PROTOTYPE
            };
            for k in 0..channels {
                let mut state_1 = 10 * channels - 1 - k;
                let mut sum = 0.0;
                for p in 0..POLYPHASE {
                    sum += prototype[filter_index + p] * self.states[state_1];
                    if p + 1 < POLYPHASE {
                        state_1 -= 2 * channels;
                    }
                }
                time[k] = sum;
                filter_index += filter_stride;
                let mut state_0 = k;
                let mut sum = 0.0;
                for p in 0..POLYPHASE {
                    sum += prototype[filter_index + p] * self.states[state_0];
                    if p + 1 < POLYPHASE {
                        state_0 += 2 * channels;
                    }
                }
                time[2 * channels - 1 - k] = sum;
            }
        }
        let mut real = vec![0.0; channels];
        let mut imaginary = vec![0.0; channels];
        if self.low_power {
            let half = channels / 2;
            if self.cldfb {
                for i in 0..half {
                    real[half + i] = (time[channels - 1 - i] - time[i]) * 0.5;
                    real[half - 1 - i] = (time[channels + i] + time[2 * channels - 1 - i]) * 0.5;
                }
                real = dct_iv(&real);
            } else {
                real[0] = time[3 * half] * 0.5;
                for i in 1..half {
                    real[i] = (time[3 * half - i] + time[3 * half + i]) * 0.5;
                }
                for i in 0..half {
                    real[half + i] = (time[2 * half - i] - time[i]) * 0.5;
                }
                real = dct_iii(&real);
            }
            for value in &mut real {
                *value *= 1.0 / 4096.0;
            }
            self.states.copy_within(channels..10 * channels, 0);
            return Ok(QmfSlot { real, imaginary });
        }
        if channels == 64 && !self.cldfb {
            let x = time[1] * 0.5;
            let y = time[0];
            real[0] = x + y * 0.5;
            imaginary[0] = x - y * 0.5;
            for i in 1..channels {
                let x = time[i + 1] * 0.5;
                let y = time[2 * channels - i];
                real[i] = x - y * 0.5;
                imaginary[i] = x + y * 0.5;
            }
        } else {
            for i in (0..channels).step_by(2) {
                let x0 = time[i] * 0.5;
                let x1 = time[i + 1] * 0.5;
                let y0 = time[2 * channels - 1 - i];
                let y1 = time[2 * channels - 2 - i];
                real[i] = x0 - y0 * 0.5;
                real[i + 1] = x1 - y1 * 0.5;
                imaginary[i] = x0 + y0 * 0.5;
                imaginary[i + 1] = x1 + y1 * 0.5;
            }
        }
        let fixed_cldfb64_modulation = fixed_cldfb64_time.is_some();
        if fixed_cldfb64_modulation {
            let fixed_time = fixed_cldfb64_time.unwrap();
            let mut fixed_real = [0i32; 64];
            let mut fixed_imaginary = [0i32; 64];
            for band in (0..64).step_by(2) {
                let x0 = fixed_time[band] >> 1;
                let x1 = fixed_time[band + 1] >> 1;
                let y0 = fixed_time[127 - band];
                let y1 = fixed_time[126 - band];
                fixed_real[band] = x0 - (y0 >> 1);
                fixed_real[band + 1] = x1 - (y1 >> 1);
                fixed_imaginary[band] = x0 + (y0 >> 1);
                fixed_imaginary[band + 1] = x1 + (y1 >> 1);
            }
            fixed_dct_iv_64(&mut fixed_real);
            fixed_dst_iv_64(&mut fixed_imaginary);
            for band in 0..64 {
                let cosine = (CLDFB_PHASE_COS_64[band] * 32_768.0).round() as i32;
                let sine = (CLDFB_PHASE_SIN_64[band] * 32_768.0).round() as i32;
                let multiply = |left: i32, right: i32| {
                    (((i64::from(left) * i64::from(right)) >> 16) as i32).wrapping_shl(1)
                };
                let rotated_imaginary = multiply(fixed_imaginary[band], cosine)
                    .wrapping_sub(multiply(fixed_real[band], sine));
                let rotated_real = multiply(fixed_imaginary[band], sine)
                    .wrapping_add(multiply(fixed_real[band], cosine));
                let output_factor = if normalized_pcm {
                    2.0_f64.powi(-39)
                } else {
                    2.0_f64.powi(-24)
                };
                real[band] = rotated_real as f64 * output_factor;
                imaginary[band] = rotated_imaginary as f64 * output_factor;
            }
        } else {
            real = dct_iv(&real);
            imaginary = dst_iv(&imaginary);
        }
        let output_scale = match channels {
            16 => 1.0 / 2048.0,
            24 | 64 => 1.0 / 8192.0,
            _ => 1.0 / 4096.0,
        };
        for band in 0..channels {
            if self.cldfb && !fixed_cldfb64_modulation {
                let r = real[band];
                let i = imaginary[band];
                let (phase_cos, phase_sin): (&[f64], &[f64]) = if channels == 64 {
                    (&CLDFB_PHASE_COS_64, &CLDFB_PHASE_SIN_64)
                } else {
                    (&CLDFB_PHASE_COS_32_ANALYSIS, &CLDFB_PHASE_SIN_32)
                };
                real[band] = r * phase_cos[band] + i * phase_sin[band];
                imaginary[band] = i * phase_cos[band] - r * phase_sin[band];
            } else if channels != 64 {
                let r = real[band];
                let i = imaginary[band];
                let (phase_cos, phase_sin): (&[f64], &[f64]) = match channels {
                    16 => (&PHASE_COS_16, &PHASE_SIN_16),
                    24 => (&PHASE_COS_24, &PHASE_SIN_24),
                    _ => (&PHASE_COS, &PHASE_SIN),
                };
                real[band] = r * phase_cos[band] - i * phase_sin[band];
                imaginary[band] = i * phase_cos[band] + r * phase_sin[band];
            }
            if !fixed_cldfb64_modulation {
                real[band] *= output_scale;
                imaginary[band] *= output_scale;
            }
        }
        self.states.copy_within(channels..10 * channels, 0);
        Ok(QmfSlot { real, imaginary })
    }
}

#[derive(Debug, Clone)]
pub struct LdSbrQmfSynthesis {
    channels: usize,
    stride: usize,
    cldfb: bool,
    low_power: bool,
    states: Vec<f64>,
}

impl LdSbrQmfSynthesis {
    pub fn new(channels: usize) -> Result<Self, QmfError> {
        let stride = match channels {
            64 => 1,
            32 => 2,
            _ => return Err(QmfError::UnsupportedChannelCount(channels)),
        };
        Ok(Self {
            channels,
            stride,
            cldfb: false,
            low_power: false,
            states: vec![0.0; 9 * channels],
        })
    }

    pub fn new_cldfb_32() -> Self {
        Self::new_cldfb(32).unwrap()
    }

    pub fn new_cldfb(channels: usize) -> Result<Self, QmfError> {
        if !matches!(channels, 32 | 64) {
            return Err(QmfError::UnsupportedChannelCount(channels));
        }
        Ok(Self {
            channels,
            stride: 1,
            cldfb: true,
            low_power: false,
            states: vec![0.0; 9 * channels],
        })
    }

    pub fn set_low_power(&mut self, enabled: bool) {
        self.low_power = enabled;
    }

    pub fn process_frame(&mut self, slots: &[QmfSlot]) -> Result<Vec<f64>, QmfError> {
        let mut output = Vec::with_capacity(slots.len() * self.channels);
        for slot in slots {
            output.extend(self.process_slot(slot)?);
        }
        Ok(output)
    }

    pub fn process_slot(&mut self, slot: &QmfSlot) -> Result<Vec<f64>, QmfError> {
        if slot.real.len() < self.channels
            || (!self.low_power && slot.imaginary.len() < self.channels)
        {
            return Err(QmfError::InvalidSubbandCount {
                expected: self.channels,
                actual: if self.low_power {
                    slot.real.len()
                } else {
                    slot.real.len().min(slot.imaginary.len())
                },
            });
        }
        let l = self.channels;
        if self.low_power {
            let mut real = dct_ii(&slot.real[..l]);
            let mut imaginary = vec![0.0; l];
            let half = l / 2;
            if self.cldfb {
                // CLDFB synthesis uses the same quarter-scale input
                // convention in both complex and real-only modulation.
                let transformed = dct_iv(&slot.real[..l])
                    .into_iter()
                    .map(|value| value * 0.25)
                    .collect::<Vec<_>>();
                let mut work = vec![0.0; 2 * l];
                work[half..half + l].copy_from_slice(&transformed);
                for i in 0..half {
                    work[i] = work[l - 1 - i];
                    work[2 * l - 1 - i] = -work[l + i];
                }
                real.copy_from_slice(&work[..l]);
                imaginary.copy_from_slice(&work[l..]);
            } else {
                imaginary[0] = real[half];
                imaginary[half] = 0.0;
                real.swap(0, half);
                for i in 1..half / 2 {
                    imaginary[half - i] = real[l - i];
                    imaginary[half + i] = -real[l - i];
                    imaginary[i] = real[half + i];
                    imaginary[l - i] = -real[half + i];
                    real[half + i] = real[i];
                    real[l - i] = real[half - i];
                    real.swap(i, half - i);
                }
                imaginary[half / 2] = real[half + half / 2];
                imaginary[half + half / 2] = -real[half + half / 2];
                real[half + half / 2] = real[half / 2];
            }
            return self.synthesize_modulated(&real, &imaginary);
        }
        let (mut real_input, mut imaginary_input) =
            (slot.real[..l].to_vec(), slot.imaginary[..l].to_vec());
        if self.cldfb {
            let (phase_cos, phase_sin): (&[f64], &[f64]) = if l == 64 {
                (&CLDFB_PHASE_COS_64, &CLDFB_PHASE_SIN_64)
            } else {
                (&CLDFB_PHASE_COS_32_SYNTHESIS, &CLDFB_PHASE_SIN_32)
            };
            for band in 0..l {
                let real = slot.real[band];
                let imaginary = slot.imaginary[band];
                real_input[band] = (imaginary * phase_sin[band] + real * phase_cos[band]) * 0.25;
                imaginary_input[band] =
                    (imaginary * phase_cos[band] - real * phase_sin[band]) * 0.25;
            }
        }
        let mut real = dct_iv(&real_input);
        let mut imaginary = dst_iv(&imaginary_input);
        for i in 0..l / 2 {
            let sign = if self.cldfb { 1.0 } else { -1.0 };
            let r1 = sign * real[i];
            let i2 = sign * imaginary[l - 1 - i];
            let r2 = sign * real[l - 1 - i];
            let i1 = sign * imaginary[i];
            real[i] = (r1 - i1) * 0.5;
            imaginary[l - 1 - i] = -(r1 + i1) * 0.5;
            real[l - 1 - i] = (r2 - i2) * 0.5;
            imaginary[i] = -(r2 + i2) * 0.5;
        }
        self.synthesize_modulated(&real, &imaginary)
    }

    fn synthesize_modulated(
        &mut self,
        real: &[f64],
        imaginary: &[f64],
    ) -> Result<Vec<f64>, QmfError> {
        let l = self.channels;
        let mut output = vec![0.0; l];
        if self.cldfb {
            let prototype: &[f64] = if l == 64 {
                &CLDFB_PROTOTYPE_64
            } else {
                &CLDFB_PROTOTYPE_32
            };
            let mut filter_forward = 0;
            let mut filter_reverse = prototype.len() / 2;
            for j in (0..l).rev() {
                let state = j * 9;
                let r = real[j];
                let i = imaginary[j];
                output[j] = (self.states[state] + prototype[filter_reverse + 4] * r * 0.5) * 256.0;
                self.states[state] =
                    self.states[state + 1] + prototype[filter_forward + 4] * i * 0.5;
                self.states[state + 1] =
                    self.states[state + 2] + prototype[filter_reverse + 3] * r * 0.5;
                self.states[state + 2] =
                    self.states[state + 3] + prototype[filter_forward + 3] * i * 0.5;
                self.states[state + 3] =
                    self.states[state + 4] + prototype[filter_reverse + 2] * r * 0.5;
                self.states[state + 4] =
                    self.states[state + 5] + prototype[filter_forward + 2] * i * 0.5;
                self.states[state + 5] =
                    self.states[state + 6] + prototype[filter_reverse + 1] * r * 0.5;
                self.states[state + 6] =
                    self.states[state + 7] + prototype[filter_forward + 1] * i * 0.5;
                self.states[state + 7] =
                    self.states[state + 8] + prototype[filter_reverse] * r * 0.5;
                self.states[state + 8] = prototype[filter_forward] * i * 0.5;
                filter_forward += POLYPHASE;
                filter_reverse += POLYPHASE;
            }
            return Ok(output);
        }
        let mut filter_forward = self.stride * POLYPHASE;
        let mut filter_reverse = 320 - self.stride * POLYPHASE;
        for j in (0..l).rev() {
            let state = j * 9;
            let r = real[j];
            let i = imaginary[j];
            output[j] = (self.states[state] + PROTOTYPE[filter_reverse] * r * 0.5) * 8.0;
            self.states[state] = self.states[state + 1] + PROTOTYPE[filter_forward + 4] * i * 0.5;
            self.states[state + 1] =
                self.states[state + 2] + PROTOTYPE[filter_reverse + 1] * r * 0.5;
            self.states[state + 2] =
                self.states[state + 3] + PROTOTYPE[filter_forward + 3] * i * 0.5;
            self.states[state + 3] =
                self.states[state + 4] + PROTOTYPE[filter_reverse + 2] * r * 0.5;
            self.states[state + 4] =
                self.states[state + 5] + PROTOTYPE[filter_forward + 2] * i * 0.5;
            self.states[state + 5] =
                self.states[state + 6] + PROTOTYPE[filter_reverse + 3] * r * 0.5;
            self.states[state + 6] =
                self.states[state + 7] + PROTOTYPE[filter_forward + 1] * i * 0.5;
            self.states[state + 7] =
                self.states[state + 8] + PROTOTYPE[filter_reverse + 4] * r * 0.5;
            self.states[state + 8] = PROTOTYPE[filter_forward] * i * 0.5;
            filter_forward += self.stride * POLYPHASE;
            if j > 0 {
                filter_reverse -= self.stride * POLYPHASE;
            }
        }
        Ok(output)
    }
}

#[derive(Debug, Clone)]
pub struct LdSbrChannelProcessor {
    analysis: LdSbrQmfAnalysis,
    synthesis: LdSbrQmfSynthesis,
    sampling_frequency: u32,
    dual_rate: bool,
    eld: bool,
    random_state: u32,
    harmonic_phase: u8,
    previous_harmonic_bands: Vec<bool>,
    previous_invf_modes: Vec<u8>,
    previous_bandwidths: Vec<f64>,
    patch_history: Vec<[(f64, f64); 2]>,
    previous_gains: Vec<f64>,
    previous_noise_levels: Vec<f64>,
    previous_attack_first: bool,
    pvc_predictor: PvcPredictor,
}

impl LdSbrChannelProcessor {
    pub fn new(sampling_frequency: u32, dual_rate: bool, _random_seed: u32) -> Self {
        Self {
            analysis: LdSbrQmfAnalysis::new(),
            synthesis: LdSbrQmfSynthesis::new(if dual_rate { 64 } else { 32 }).unwrap(),
            sampling_frequency,
            dual_rate,
            eld: false,
            random_state: 0,
            harmonic_phase: 0,
            previous_harmonic_bands: vec![false; 64],
            previous_invf_modes: Vec::new(),
            previous_bandwidths: Vec::new(),
            patch_history: vec![[(0.0, 0.0); 2]; CHANNELS],
            previous_gains: Vec::new(),
            previous_noise_levels: Vec::new(),
            previous_attack_first: false,
            pvc_predictor: PvcPredictor::new(),
        }
    }

    pub fn new_usac(
        sampling_frequency: u32,
        sbr_ratio_index: u8,
        random_seed: u32,
    ) -> Result<Self, LdSbrProcessingError> {
        let analysis_channels = match sbr_ratio_index {
            1 => 16,
            2 => 24,
            3 => 32,
            _ => return Err(QmfError::InvalidTimeStep(0).into()),
        };
        let mut processor = Self::new(sampling_frequency, true, random_seed);
        processor.analysis = LdSbrQmfAnalysis::new_with_channels(analysis_channels)?;
        Ok(processor)
    }

    pub fn new_eld(sampling_frequency: u32, dual_rate: bool, random_seed: u32) -> Self {
        let mut processor = Self::new(sampling_frequency, dual_rate, random_seed);
        processor.eld = true;
        processor.analysis = LdSbrQmfAnalysis::new_cldfb_32();
        processor.synthesis =
            LdSbrQmfSynthesis::new_cldfb(if dual_rate { 64 } else { 32 }).unwrap();
        processor
    }

    pub fn clear_history(&mut self) {
        let low_power = self.analysis.low_power;
        let analysis_channels = self.analysis.channels;
        *self = if self.eld {
            Self::new_eld(self.sampling_frequency, self.dual_rate, 0)
        } else if analysis_channels == 16 {
            Self::new_usac(self.sampling_frequency, 1, 0).expect("valid USAC QMF configuration")
        } else if analysis_channels == 24 {
            Self::new_usac(self.sampling_frequency, 2, 0).expect("valid USAC QMF configuration")
        } else {
            Self::new(self.sampling_frequency, self.dual_rate, 0)
        };
        self.set_low_power(low_power);
    }

    pub fn set_low_power(&mut self, enabled: bool) {
        self.analysis.set_low_power(enabled);
        self.synthesis.set_low_power(enabled);
    }

    pub fn process(
        &mut self,
        core_samples: &[f64],
        frame: &LdSbrFrame,
        right_channel: bool,
    ) -> Result<Vec<f64>, LdSbrProcessingError> {
        let slots = self.process_frame_to_qmf(core_samples, frame, right_channel)?;
        self.synthesize_qmf(&slots)
    }

    pub fn process_frame_to_qmf(
        &mut self,
        core_samples: &[f64],
        frame: &LdSbrFrame,
        right_channel: bool,
    ) -> Result<Vec<QmfSlot>, LdSbrProcessingError> {
        let (control, values, harmonics) = if right_channel {
            (
                frame
                    .prefix
                    .right
                    .as_ref()
                    .ok_or(LdSbrProcessingError::MissingRightChannel)?,
                frame
                    .right_dequantized
                    .as_ref()
                    .ok_or(LdSbrProcessingError::MissingRightChannel)?,
                frame
                    .right_harmonics
                    .as_ref()
                    .ok_or(LdSbrProcessingError::MissingRightChannel)?,
            )
        } else {
            (
                &frame.prefix.left,
                &frame.left_dequantized,
                &frame.left_harmonics,
            )
        };
        let raw_values = if right_channel {
            frame
                .right
                .as_ref()
                .ok_or(LdSbrProcessingError::MissingRightChannel)?
        } else {
            &frame.left
        };
        self.process_channel_to_qmf(
            core_samples,
            &frame.active_header,
            &frame.frequency_tables,
            control,
            raw_values,
            values,
            harmonics,
            1,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn process_channel(
        &mut self,
        core_samples: &[f64],
        header: &crate::asc::LdSbrHeader,
        tables: &LdSbrFrequencyTables,
        control: &LdSbrChannelControl,
        raw_values: &crate::ld_sbr::LdSbrChannelValues,
        values: &LdSbrDequantizedChannel,
        harmonics: &[bool],
        time_step: u8,
    ) -> Result<Vec<f64>, LdSbrProcessingError> {
        let slots = self.process_channel_to_qmf(
            core_samples,
            header,
            tables,
            control,
            raw_values,
            values,
            harmonics,
            time_step,
        )?;
        self.synthesize_qmf(&slots)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn process_channel_to_qmf(
        &mut self,
        core_samples: &[f64],
        header: &crate::asc::LdSbrHeader,
        tables: &LdSbrFrequencyTables,
        control: &LdSbrChannelControl,
        raw_values: &crate::ld_sbr::LdSbrChannelValues,
        values: &LdSbrDequantizedChannel,
        harmonics: &[bool],
        time_step: u8,
    ) -> Result<Vec<QmfSlot>, LdSbrProcessingError> {
        if !matches!(time_step, 1 | 2 | 4) {
            return Err(QmfError::InvalidTimeStep(time_step).into());
        }
        let scaled_control;
        let control = if time_step == 1 {
            control
        } else {
            scaled_control = LdSbrChannelControl {
                grid: crate::ld_sbr::LdSbrGrid {
                    transient: control.grid.transient,
                    amp_resolution: control.grid.amp_resolution,
                    borders: control
                        .grid
                        .borders
                        .iter()
                        .map(|value| value * time_step)
                        .collect(),
                    frequency_resolution: control.grid.frequency_resolution.clone(),
                    transient_envelope: control.grid.transient_envelope,
                    noise_borders: control
                        .grid
                        .noise_borders
                        .iter()
                        .map(|value| value * time_step)
                        .collect(),
                },
                envelope_time_domain: control.envelope_time_domain.clone(),
                noise_time_domain: control.noise_time_domain.clone(),
            };
            &scaled_control
        };
        let scaled_values;
        let values = if self.dual_rate {
            values
        } else {
            // FDK carries fewer energy exponent bits into a single-rate
            // adjuster: six for ELD CLDFB and four for the standard QMF.
            let divisor = if self.synthesis.cldfb { 64.0 } else { 16.0 };
            scaled_values = LdSbrDequantizedChannel {
                envelope_energy: values
                    .envelope_energy
                    .iter()
                    .map(|envelope| envelope.iter().map(|energy| energy / divisor).collect())
                    .collect(),
                noise_energy: values.noise_energy.clone(),
            };
            &scaled_values
        };
        let mut slots = self.analysis.process_frame(core_samples)?;
        let patches = derive_patches(tables, self.sampling_frequency)?;
        let low_power = self.analysis.low_power;
        let mut degree_alias = [0.0; 64];
        apply_inverse_filtered_patches_mode(
            &mut slots,
            &patches,
            &tables.noise,
            &raw_values.inverse_filtering_modes,
            &mut self.previous_invf_modes,
            &mut self.previous_bandwidths,
            &mut self.patch_history,
            low_power,
            &mut degree_alias,
        )?;
        let limiter_borders =
            derive_limiter_borders(tables, &patches, header.limiter_bands.unwrap_or(2))?;
        let mut clip_ratios = Vec::new();
        apply_envelope_gains_with_limiter_borders_mode(
            &mut slots,
            control,
            tables,
            values,
            header.limiter_gains.unwrap_or(2),
            !header.smoothing_mode.unwrap_or(true),
            &mut self.previous_gains,
            &limiter_borders,
            self.previous_attack_first,
            Some(&mut clip_ratios),
            low_power,
            &degree_alias,
            harmonics,
        )?;
        apply_noise_and_harmonics_with_limiter_borders_mode(
            &mut slots,
            control,
            tables,
            values,
            harmonics,
            &mut self.random_state,
            &mut self.harmonic_phase,
            &mut self.previous_harmonic_bands,
            &limiter_borders,
            Some(&clip_ratios),
            !header.smoothing_mode.unwrap_or(true),
            self.previous_attack_first,
            Some(&mut self.previous_noise_levels),
            low_power,
            self.eld,
        )?;
        self.previous_attack_first =
            control.grid.transient_envelope == Some(control.grid.envelope_count());
        Ok(slots)
    }

    pub fn synthesize_qmf(&mut self, slots: &[QmfSlot]) -> Result<Vec<f64>, LdSbrProcessingError> {
        Ok(self.synthesis.process_frame(slots)?)
    }

    /// Run the QMF analysis/synthesis pair without high-frequency
    /// reconstruction. FDK uses this path for LFE elements in an SBR stream:
    /// they carry no SBR payload but still have to match the output rate.
    pub fn upsample_only(
        &mut self,
        core_samples: &[f64],
    ) -> Result<Vec<f64>, LdSbrProcessingError> {
        let mut slots = self.analysis.process_frame(core_samples)?;
        for slot in &mut slots {
            slot.real.resize(self.synthesis.channels, 0.0);
            slot.imaginary.resize(self.synthesis.channels, 0.0);
        }
        self.synthesize_qmf(&slots)
    }

    pub fn process_usac_mono_to_qmf(
        &mut self,
        core_samples: &[f64],
        frame: &UsacSbrMonoFrame,
        time_step: u8,
    ) -> Result<Vec<QmfSlot>, LdSbrProcessingError> {
        let mut slots = self.process_channel_to_qmf(
            core_samples,
            &frame.frame.active_header,
            &frame.frame.frequency_tables,
            &frame.frame.control,
            &frame.frame.values,
            &frame.frame.dequantized,
            &frame.frame.harmonics,
            time_step,
        )?;
        let low_subband = usize::from(frame.frame.frequency_tables.high[0]);
        let high_subband_count = usize::from(
            *frame.frame.frequency_tables.high.last().unwrap()
                - frame.frame.frequency_tables.high[0],
        );
        for (envelope, shaping) in frame.inter_tes.iter().enumerate() {
            if shaping.active {
                let start = usize::from(frame.frame.control.grid.borders[envelope] * time_step);
                let stop = usize::from(frame.frame.control.grid.borders[envelope + 1] * time_step);
                apply_inter_tes_qmf_f64(
                    &mut slots,
                    start,
                    stop,
                    low_subband,
                    high_subband_count,
                    shaping.mode,
                );
            }
        }
        Ok(slots)
    }

    pub fn process_usac_mono(
        &mut self,
        core_samples: &[f64],
        frame: &UsacSbrMonoFrame,
        time_step: u8,
    ) -> Result<Vec<f64>, LdSbrProcessingError> {
        let slots = self.process_usac_mono_to_qmf(core_samples, frame, time_step)?;
        self.synthesize_qmf(&slots)
    }

    pub fn process_usac_stereo_channel_to_qmf(
        &mut self,
        core_samples: &[f64],
        frame: &UsacSbrStereoFrame,
        right_channel: bool,
        time_step: u8,
    ) -> Result<Vec<QmfSlot>, LdSbrProcessingError> {
        let (control, raw, values, harmonics, shaping) = if right_channel {
            (
                &frame.frame.right_control,
                &frame.frame.right,
                &frame.frame.right_dequantized,
                &frame.frame.right_harmonics,
                &frame.inter_tes[1],
            )
        } else {
            (
                &frame.frame.left_control,
                &frame.frame.left,
                &frame.frame.left_dequantized,
                &frame.frame.left_harmonics,
                &frame.inter_tes[0],
            )
        };
        let mut slots = self.process_channel_to_qmf(
            core_samples,
            &frame.frame.active_header,
            &frame.frame.frequency_tables,
            control,
            raw,
            values,
            harmonics,
            time_step,
        )?;
        let low = usize::from(frame.frame.frequency_tables.high[0]);
        let high_count = usize::from(
            *frame.frame.frequency_tables.high.last().unwrap()
                - frame.frame.frequency_tables.high[0],
        );
        for (envelope, tes) in shaping.iter().enumerate() {
            if tes.active {
                apply_inter_tes_qmf_f64(
                    &mut slots,
                    usize::from(control.grid.borders[envelope] * time_step),
                    usize::from(control.grid.borders[envelope + 1] * time_step),
                    low,
                    high_count,
                    tes.mode,
                );
            }
        }
        Ok(slots)
    }

    pub fn process_usac_pvc_to_qmf(
        &mut self,
        core_samples: &[f64],
        header: &crate::asc::LdSbrHeader,
        tables: &LdSbrFrequencyTables,
        frame: &UsacPvcSbrFrame,
        pvc_mode: u8,
    ) -> Result<Vec<QmfSlot>, LdSbrProcessingError> {
        let mut slots = self.analysis.process_frame(core_samples)?;
        let patches = derive_patches(tables, self.sampling_frequency)?;
        apply_inverse_filtered_patches(
            &mut slots,
            &patches,
            &tables.noise,
            &frame.inverse_filtering_modes,
            &mut self.previous_invf_modes,
            &mut self.previous_bandwidths,
            &mut self.patch_history,
        )?;
        let rate = (slots.len() / 16).max(1);
        let low = usize::from(tables.high[0]);
        let high = usize::from(*tables.high.last().unwrap());
        let predicted = self
            .pvc_predictor
            .predict_qmf_slots(
                pvc_mode,
                usize::from(frame.envelope.slots_per_group),
                rate,
                low,
                &frame.envelope.ids,
                &slots,
            )
            .map_err(|_| LdSbrProcessingError::Qmf(QmfError::EnvelopeLayoutMismatch))?;
        apply_pvc_predicted_energies(&mut slots, low, high, &predicted, rate);
        let scale_borders = |values: &[i8]| {
            values
                .iter()
                .map(|&value| (value.max(0) as usize * rate).min(slots.len()) as u8)
                .collect::<Vec<_>>()
        };
        let borders = scale_borders(&frame.grid.borders);
        let noise_borders = scale_borders(&frame.grid.noise_borders);
        let control = LdSbrChannelControl {
            grid: crate::ld_sbr::LdSbrGrid {
                transient: false,
                amp_resolution: Some(header.amp_resolution),
                frequency_resolution: vec![true; borders.len() - 1],
                transient_envelope: None,
                borders,
                noise_borders,
            },
            envelope_time_domain: vec![false; frame.grid.borders.len() - 1],
            noise_time_domain: vec![false; frame.grid.noise_borders.len() - 1],
        };
        let envelope_energy = control
            .grid
            .borders
            .windows(2)
            .map(|window| {
                tables
                    .high
                    .windows(2)
                    .map(|bands| {
                        let mut energy = 0.0;
                        let mut count = 0usize;
                        for slot in &slots[usize::from(window[0])..usize::from(window[1])] {
                            for band in usize::from(bands[0])..usize::from(bands[1]) {
                                energy += slot.real[band] * slot.real[band]
                                    + slot.imaginary[band] * slot.imaginary[band];
                                count += 1;
                            }
                        }
                        energy / count.max(1) as f64
                    })
                    .collect()
            })
            .collect();
        let noise_energy = frame
            .noise
            .iter()
            .map(|values| {
                values
                    .iter()
                    .map(|&value| 2.0f64.powi(6 - value as i32))
                    .collect()
            })
            .collect();
        let dequantized = LdSbrDequantizedChannel {
            envelope_energy,
            noise_energy,
        };
        apply_noise_and_harmonics(
            &mut slots,
            &control,
            tables,
            &dequantized,
            &frame.harmonics,
            &mut self.random_state,
            &mut self.harmonic_phase,
            &mut self.previous_harmonic_bands,
        )?;
        Ok(slots)
    }

    pub fn process_usac_pvc(
        &mut self,
        core_samples: &[f64],
        header: &crate::asc::LdSbrHeader,
        tables: &LdSbrFrequencyTables,
        frame: &UsacPvcSbrFrame,
        pvc_mode: u8,
    ) -> Result<Vec<f64>, LdSbrProcessingError> {
        let slots = self.process_usac_pvc_to_qmf(core_samples, header, tables, frame, pvc_mode)?;
        self.synthesize_qmf(&slots)
    }

    /// Drain the analysis/synthesis filter memories with a zero core frame.
    pub fn flush(&mut self, core_sample_count: usize) -> Result<Vec<f64>, LdSbrProcessingError> {
        let mut slots = self.analysis.process_frame(&vec![0.0; core_sample_count])?;
        for slot in &mut slots {
            slot.real.resize(self.synthesis.channels, 0.0);
            slot.imaginary.resize(self.synthesis.channels, 0.0);
        }
        Ok(self.synthesis.process_frame(&slots)?)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LdSbrProcessingError {
    Syntax(LdSbrError),
    Qmf(QmfError),
    MissingRightChannel,
}

impl From<LdSbrError> for LdSbrProcessingError {
    fn from(value: LdSbrError) -> Self {
        Self::Syntax(value)
    }
}

impl From<QmfError> for LdSbrProcessingError {
    fn from(value: QmfError) -> Self {
        Self::Qmf(value)
    }
}

fn dct_iv(input: &[f64]) -> Vec<f64> {
    let n = input.len() as f64;
    (0..input.len())
        .map(|k| {
            input
                .iter()
                .enumerate()
                .map(|(index, &value)| {
                    value
                        * (std::f64::consts::PI / n * (index as f64 + 0.5) * (k as f64 + 0.5)).cos()
                })
                .sum()
        })
        .collect()
}

fn dct_ii(input: &[f64]) -> Vec<f64> {
    let n = input.len() as f64;
    (0..input.len())
        .map(|k| {
            input
                .iter()
                .enumerate()
                .map(|(index, &value)| {
                    value * (std::f64::consts::PI / n * (index as f64 + 0.5) * k as f64).cos()
                })
                .sum()
        })
        .collect()
}

fn dct_iii(input: &[f64]) -> Vec<f64> {
    let n = input.len() as f64;
    (0..input.len())
        .map(|k| {
            input[0] * 0.5
                + input
                    .iter()
                    .enumerate()
                    .skip(1)
                    .map(|(index, &value)| {
                        value * (std::f64::consts::PI / n * index as f64 * (k as f64 + 0.5)).cos()
                    })
                    .sum::<f64>()
        })
        .collect()
}

fn dst_iv(input: &[f64]) -> Vec<f64> {
    let n = input.len() as f64;
    (0..input.len())
        .map(|k| {
            input
                .iter()
                .enumerate()
                .map(|(index, &value)| {
                    value
                        * (std::f64::consts::PI / n * (index as f64 + 0.5) * (k as f64 + 0.5)).sin()
                })
                .sum()
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QmfError {
    InvalidSampleCount(usize),
    InvalidPatchBand { source: usize, target: usize },
    EnvelopeLayoutMismatch,
    UnsupportedChannelCount(usize),
    InvalidSubbandCount { expected: usize, actual: usize },
    InverseFilteringLayoutMismatch,
    InvalidTimeStep(u8),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asc::LdSbrHeader;
    use crate::bits::{BitReader, BitWriter};
    use crate::fixed_fft::{
        fft32_radix4_stage1, fft32_radix4_stage2, fft32_radix4_stage3, fixed_dct_iv_64,
        fixed_dst_iv_64, fixed_fft32,
    };
    use crate::ld_sbr::{encode_sbr_huffman, SbrHuffmanBook};
    use crate::sbr::{SbrMonoFrameParser, SbrStereoFrame};

    fn fft32_radix2_probe(values: &mut [i32; 64]) {
        for index in 0usize..32 {
            let reversed = index.reverse_bits() >> (usize::BITS - 5);
            if reversed > index {
                values.swap(2 * index, 2 * reversed);
                values.swap(2 * index + 1, 2 * reversed + 1);
            }
        }
        for stage in 0..5 {
            let width = 1usize << (stage + 1);
            let half = width / 2;
            for start in (0..32).step_by(width) {
                for offset in 0..half {
                    let angle = -2.0 * std::f64::consts::PI * offset as f64 / width as f64;
                    let wr = (angle.cos() * 2_147_483_648.0)
                        .round()
                        .clamp(i32::MIN as f64, i32::MAX as f64)
                        as i32;
                    let wi = (angle.sin() * 2_147_483_648.0)
                        .round()
                        .clamp(i32::MIN as f64, i32::MAX as f64)
                        as i32;
                    let a = 2 * (start + offset);
                    let b = 2 * (start + offset + half);
                    let br = values[b];
                    let bi = values[b + 1];
                    let ar = values[a];
                    let ai = values[a + 1];
                    if stage == 0 {
                        values[a] = ar.wrapping_add(br);
                        values[a + 1] = ai.wrapping_add(bi);
                        values[b] = ar.wrapping_sub(br);
                        values[b + 1] = ai.wrapping_sub(bi);
                    } else {
                        let mul_div2 =
                            |left: i32, right: i32| ((left as i64 * right as i64) >> 32) as i32;
                        let tr = mul_div2(br, wr).wrapping_sub(mul_div2(bi, wi));
                        let ti = mul_div2(br, wi).wrapping_add(mul_div2(bi, wr));
                        values[a] = (ar >> 1).wrapping_add(tr);
                        values[a + 1] = (ai >> 1).wrapping_add(ti);
                        values[b] = (ar >> 1).wrapping_sub(tr);
                        values[b + 1] = (ai >> 1).wrapping_sub(ti);
                    }
                }
            }
        }
    }

    fn sbr_huffman_code(book: SbrHuffmanBook, symbol: i8) -> Vec<bool> {
        encode_sbr_huffman(book, symbol).expect("requested symbol exists in the SBR Huffman book")
    }

    fn write_sbr_code(writer: &mut BitWriter, code: &[bool]) {
        for &bit in code {
            writer.write_bool(bit);
        }
    }

    fn usac_test_header() -> LdSbrHeader {
        LdSbrHeader {
            amp_resolution: true,
            start_frequency: 5,
            stop_frequency: 8,
            crossover_band: 2,
            frequency_scale: Some(1),
            alter_scale: Some(false),
            noise_bands: Some(2),
            ..LdSbrHeader::default()
        }
    }

    fn parsed_usac_mono_frame() -> UsacSbrMonoFrame {
        let header = usac_test_header();
        let tables = LdSbrFrequencyTables::from_header(&header, 44_100).unwrap();
        let zero = sbr_huffman_code(SbrHuffmanBook::EnvelopeLevel30Frequency, 0);
        let mut writer = BitWriter::new();
        writer.write(0, 2); // FIXFIX
        writer.write(0, 2); // one envelope
        writer.write_bool(true);
        for _ in 0..tables.noise_band_count() {
            writer.write(1, 2);
        }
        writer.write(9, 6);
        for _ in 1..tables.high_band_count() {
            write_sbr_code(&mut writer, &zero);
        }
        writer.write_bool(true); // inter-TES active
        writer.write(2, 2);
        writer.write(6, 5);
        for _ in 1..tables.noise_band_count() {
            write_sbr_code(&mut writer, &zero);
        }
        writer.write_bool(false);
        let bits = writer.bits_written();
        let bytes = writer.finish();
        let mut reader = BitReader::with_bit_len(&bytes, bits).unwrap();
        SbrMonoFrameParser::new_usac(header, 44_100)
            .unwrap()
            .parse_usac(&mut reader, true, false, true)
            .unwrap()
    }

    fn parsed_usac_pvc_frame() -> (LdSbrHeader, LdSbrFrequencyTables, UsacPvcSbrFrame) {
        let header = usac_test_header();
        let tables = LdSbrFrequencyTables::from_header(&header, 44_100).unwrap();
        let zero = sbr_huffman_code(SbrHuffmanBook::EnvelopeLevel30Frequency, 0);
        let mut writer = BitWriter::new();
        writer.write(0, 4);
        writer.write_bool(false);
        for _ in 0..tables.noise_band_count() {
            writer.write(2, 2);
        }
        writer.write(0, 3);
        writer.write_bool(false);
        writer.write(37, 7);
        writer.write(5, 5);
        for _ in 1..tables.noise_band_count() {
            write_sbr_code(&mut writer, &zero);
        }
        writer.write_bool(false);
        let bits = writer.bits_written();
        let bytes = writer.finish();
        let mut reader = BitReader::with_bit_len(&bytes, bits).unwrap();
        let frame = SbrMonoFrameParser::new_usac(header.clone(), 44_100)
            .unwrap()
            .parse_usac_pvc(&mut reader, true, 1, false)
            .unwrap();
        (header, tables, frame)
    }

    #[test]
    fn loads_fdk_qmf_rom() {
        assert!(PROTOTYPE.len() >= 325);
        assert_eq!(PHASE_COS.len(), 32);
        assert_eq!(PHASE_SIN.len(), 32);
        assert_eq!(PHASE_COS_64.len(), 64);
        assert_eq!(PHASE_SIN_64.len(), 64);

        assert_eq!(
            complex_lpc2(&[(1.0, 0.0), (0.0, 0.0), (1.0, 0.0), (100.0, 0.0)]),
            ((0.0, 0.0), (0.0, 0.0))
        );
    }

    #[test]
    fn supports_64_band_analysis_state_layout() {
        let mut analysis = LdSbrQmfAnalysis::new_with_channels(64).unwrap();
        let slots = analysis.process_frame(&vec![0.0; 128]).unwrap();
        assert_eq!(slots.len(), 2);
        assert!(slots
            .iter()
            .all(|slot| slot.real.len() == 64 && slot.imaginary.len() == 64));
    }

    #[test]
    fn processes_usac_mono_and_pvc_through_qmf_and_pcm_facades() {
        let mono = parsed_usac_mono_frame();
        assert_eq!(
            LdSbrChannelProcessor::new(44_100, true, 1)
                .process_usac_mono_to_qmf(&[], &mono, 0)
                .unwrap_err(),
            LdSbrProcessingError::Qmf(QmfError::InvalidTimeStep(0))
        );
        assert!(LdSbrChannelProcessor::new(44_100, true, 1)
            .process_usac_mono_to_qmf(&[], &mono, 2)
            .is_err());
        let mut invalid_values = mono.frame.values.clone();
        invalid_values.inverse_filtering_modes.clear();
        assert!(LdSbrChannelProcessor::new(44_100, true, 1)
            .process_channel_to_qmf(
                &vec![0.0; 1024],
                &mono.frame.active_header,
                &mono.frame.frequency_tables,
                &mono.frame.control,
                &invalid_values,
                &mono.frame.dequantized,
                &mono.frame.harmonics,
                2,
            )
            .is_err());
        assert!(LdSbrChannelProcessor::new(44_100, true, 1)
            .process_channel_to_qmf(
                &vec![0.0; 1024],
                &mono.frame.active_header,
                &mono.frame.frequency_tables,
                &mono.frame.control,
                &mono.frame.values,
                &mono.frame.dequantized,
                &[],
                2,
            )
            .is_err());
        let mut processor = LdSbrChannelProcessor::new(44_100, true, 1);
        let slots = processor
            .process_usac_mono_to_qmf(&vec![0.0; 1024], &mono, 2)
            .unwrap();
        assert_eq!(slots.len(), 32);
        assert!(slots.iter().all(|slot| slot
            .real
            .iter()
            .chain(&slot.imaginary)
            .all(|value| value.is_finite())));
        let pcm = LdSbrChannelProcessor::new(44_100, true, 1)
            .process_usac_mono(&vec![0.0; 1024], &mono, 2)
            .unwrap();
        assert_eq!(pcm.len(), 2048);
        assert!(pcm.iter().all(|value| value.is_finite()));

        let single_rate = LdSbrChannelProcessor::new(44_100, false, 1)
            .process_usac_mono_to_qmf(&vec![0.0; 1024], &mono, 2)
            .unwrap();
        assert_eq!(single_rate.len(), 32);
        assert!(single_rate.iter().all(|slot| slot
            .real
            .iter()
            .chain(&slot.imaginary)
            .all(|value| value.is_finite())));

        let stereo = UsacSbrStereoFrame {
            frame: SbrStereoFrame {
                active_header: mono.frame.active_header.clone(),
                frequency_tables: mono.frame.frequency_tables.clone(),
                data_extra: None,
                coupling: false,
                left_control: mono.frame.control.clone(),
                right_control: mono.frame.control.clone(),
                left: mono.frame.values.clone(),
                right: mono.frame.values.clone(),
                left_dequantized: mono.frame.dequantized.clone(),
                right_dequantized: mono.frame.dequantized.clone(),
                left_harmonics: mono.frame.harmonics.clone(),
                right_harmonics: mono.frame.harmonics.clone(),
                extended_data: Vec::new(),
                bits_read: mono.frame.bits_read,
            },
            harmonic_controls: [None, None],
            inter_tes: [mono.inter_tes.clone(), mono.inter_tes.clone()],
        };
        assert!(LdSbrChannelProcessor::new(44_100, true, 1)
            .process_usac_stereo_channel_to_qmf(&[], &stereo, false, 2)
            .is_err());
        for right_channel in [false, true] {
            let slots = LdSbrChannelProcessor::new(44_100, true, 1)
                .process_usac_stereo_channel_to_qmf(&vec![0.0; 1024], &stereo, right_channel, 2)
                .unwrap();
            assert_eq!(slots.len(), 32);
            assert!(slots.iter().all(|slot| slot
                .real
                .iter()
                .chain(&slot.imaginary)
                .all(|value| value.is_finite())));
        }

        let (header, tables, pvc) = parsed_usac_pvc_frame();
        let mut processor = LdSbrChannelProcessor::new(44_100, true, 1);
        let slots = processor
            .process_usac_pvc_to_qmf(&vec![0.0; 512], &header, &tables, &pvc, 1)
            .unwrap();
        assert_eq!(slots.len(), 16);
        assert!(slots.iter().all(|slot| slot
            .real
            .iter()
            .chain(&slot.imaginary)
            .all(|value| value.is_finite())));
        assert!(matches!(
            processor.process_usac_pvc_to_qmf(&vec![0.0; 512], &header, &tables, &pvc, 0,),
            Err(LdSbrProcessingError::Qmf(QmfError::EnvelopeLayoutMismatch))
        ));
        let mut invalid_invf = pvc.clone();
        invalid_invf.inverse_filtering_modes.clear();
        assert!(LdSbrChannelProcessor::new(44_100, true, 1)
            .process_usac_pvc_to_qmf(&vec![0.0; 512], &header, &tables, &invalid_invf, 1)
            .is_err());
        let mut invalid_harmonics = pvc.clone();
        invalid_harmonics.harmonics.clear();
        assert!(LdSbrChannelProcessor::new(44_100, true, 1)
            .process_usac_pvc_to_qmf(&vec![0.0; 512], &header, &tables, &invalid_harmonics, 1,)
            .is_err());
        let pcm = LdSbrChannelProcessor::new(44_100, true, 1)
            .process_usac_pvc(&vec![0.0; 512], &header, &tables, &pvc, 1)
            .unwrap();
        assert_eq!(pcm.len(), 1024);
        assert!(pcm.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn zero_and_impulse_analysis_are_stateful_and_finite() {
        let mut qmf = LdSbrQmfAnalysis::new();
        let zero = qmf.process_frame(&vec![0.0; 512]).unwrap();
        assert_eq!(zero.len(), 16);
        assert!(zero
            .iter()
            .all(|slot| slot.real.iter().chain(&slot.imaginary).all(|v| *v == 0.0)));
        let mut impulse = vec![0.0; 512];
        impulse[0] = 1.0;
        let first = qmf.process_frame(&impulse).unwrap();
        let tail = qmf.process_frame(&vec![0.0; 512]).unwrap();
        assert!(first
            .iter()
            .chain(&tail)
            .flat_map(|slot| slot.real.iter().chain(&slot.imaginary))
            .all(|value| value.is_finite()));
        assert!(first.iter().chain(&tail).any(|slot| slot
            .real
            .iter()
            .chain(&slot.imaginary)
            .any(|value| *value != 0.0)));
    }

    #[test]
    fn derives_and_applies_fdk_style_patches() {
        let header = crate::asc::LdSbrHeader {
            start_frequency: 5,
            stop_frequency: 8,
            crossover_band: 2,
            ..crate::asc::LdSbrHeader::default()
        };
        let tables = LdSbrFrequencyTables::from_header(&header, 44_100).unwrap();
        let patches = derive_patches(&tables, 44_100).unwrap();
        assert!(!patches.is_empty());
        assert!(patches.windows(2).all(|pair| {
            pair[0].target_start_band + pair[0].band_count == pair[1].target_start_band
        }));
        let mut slot = QmfSlot {
            real: (0..32).map(|value| value as f64).collect(),
            imaginary: (0..32).map(|value| -(value as f64)).collect(),
        };
        apply_patches(std::slice::from_mut(&mut slot), &patches).unwrap();
        for patch in patches {
            for offset in 0..patch.band_count as usize {
                assert_eq!(
                    slot.real[patch.target_start_band as usize + offset],
                    slot.real[patch.source_start_band as usize + offset]
                );
            }
        }
    }

    #[test]
    fn limiter_bands_preserve_patch_boundaries_and_remove_dense_sfb_borders() {
        let tables = LdSbrFrequencyTables {
            master: vec![32, 34, 36, 40, 48],
            high: vec![32, 34, 36, 40, 48],
            low: vec![32, 34, 36, 40, 48],
            noise: vec![32, 48],
        };
        let patches = vec![
            SbrPatch {
                source_start_band: 8,
                target_start_band: 32,
                band_count: 8,
            },
            SbrPatch {
                source_start_band: 12,
                target_start_band: 40,
                band_count: 8,
            },
        ];
        assert_eq!(
            derive_limiter_borders(&tables, &patches, 0).unwrap(),
            vec![32, 48]
        );
        assert_eq!(
            derive_limiter_borders(&tables, &patches, 3).unwrap(),
            vec![32, 40, 48]
        );
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn limiter_borders_match_fdk() {
        let header = crate::asc::LdSbrHeader {
            start_frequency: 8,
            stop_frequency: 6,
            crossover_band: 0,
            ..crate::asc::LdSbrHeader::default()
        };
        let tables = LdSbrFrequencyTables::from_header(&header, 88_200).unwrap();
        let patches = derive_patches(&tables, 44_100).unwrap();
        let source_starts = patches
            .iter()
            .map(|patch| patch.source_start_band)
            .collect::<Vec<_>>();
        let target_starts = patches
            .iter()
            .map(|patch| patch.target_start_band)
            .collect::<Vec<_>>();
        let band_counts = patches
            .iter()
            .map(|patch| patch.band_count)
            .collect::<Vec<_>>();
        for limiter_bands in 0..=3 {
            let mut count = 0;
            let mut c_table = [0u8; 64];
            assert_eq!(
                unsafe {
                    fdk_aac_sys::fdk_sbr_limiter_bands_test(
                        tables.low.as_ptr(),
                        tables.low_band_count() as u8,
                        source_starts.as_ptr(),
                        target_starts.as_ptr(),
                        band_counts.as_ptr(),
                        patches.len() as u8,
                        limiter_bands,
                        &mut count,
                        c_table.as_mut_ptr(),
                    )
                },
                0
            );
            let low = tables.low[0] as usize;
            let rust = derive_limiter_borders(&tables, &patches, limiter_bands)
                .unwrap()
                .into_iter()
                .map(|border| (border - low) as u8)
                .collect::<Vec<_>>();
            assert_eq!(rust, c_table[..=count as usize], "mode {limiter_bands}");
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn fdk_component_gain_noise_and_sine_energies_follow_reference_equations() {
        let fixed = |mantissa: i32, exponent: i8| {
            mantissa as f64 / 2_147_483_648.0 * 2.0f64.powi(exponent as i32)
        };
        for (sine_present, sine_mapped, no_noise, expected_gain, expected_sine) in [
            (0, 0, 0, 1.0, 0.0),
            (1, 1, 0, 1.0, 2.0),
            (0, 0, 1, 2.0, 0.0),
        ] {
            let mut gain_m = 0;
            let mut gain_e = 0;
            let mut noise_m = 0;
            let mut noise_e = 0;
            let mut sine_m = 0;
            let mut sine_e = 0;
            assert_eq!(
                unsafe {
                    fdk_aac_sys::fdk_sbr_component_energies_test(
                        1 << 30,
                        3,
                        1 << 30,
                        1,
                        1 << 30,
                        1,
                        sine_present,
                        sine_mapped,
                        no_noise,
                        &mut gain_m,
                        &mut gain_e,
                        &mut noise_m,
                        &mut noise_e,
                        &mut sine_m,
                        &mut sine_e,
                    )
                },
                0
            );
            let gain = fixed(gain_m, gain_e);
            let noise = fixed(noise_m, noise_e);
            let sine = fixed(sine_m, sine_e);
            assert!(
                (gain - expected_gain).abs() < 1.0e-6,
                "gain {gain} raw ({gain_m},{gain_e}), noise {noise} raw ({noise_m},{noise_e}), sine {sine} raw ({sine_m},{sine_e})"
            );
            assert!((noise - 2.0).abs() < 1.0e-6, "noise {noise}");
            assert!((sine - expected_sine).abs() < 1.0e-6, "sine {sine}");
        }
    }

    #[test]
    fn inverse_filter_modes_whiten_patched_qmf_series() {
        let patch = SbrPatch {
            source_start_band: 5,
            target_start_band: 32,
            band_count: 1,
        };
        let input = (0..16)
            .map(|slot| {
                let mut real = vec![0.0; 32];
                let mut imaginary = vec![0.0; 32];
                real[5] = ((slot * 17 + 5) % 23) as f64 / 11.5 - 1.0;
                imaginary[5] = ((slot * 11 + 3) % 19) as f64 / 9.5 - 1.0;
                QmfSlot { real, imaginary }
            })
            .collect::<Vec<_>>();
        let mut off = input.clone();
        let mut high = input;
        let mut off_modes = Vec::new();
        let mut off_bw = Vec::new();
        let mut off_history = vec![[(0.0, 0.0); 2]; 32];
        apply_inverse_filtered_patches(
            &mut off,
            &[patch],
            &[32, 33],
            &[0],
            &mut off_modes,
            &mut off_bw,
            &mut off_history,
        )
        .unwrap();
        let mut high_modes = Vec::new();
        let mut high_bw = Vec::new();
        let mut high_history = vec![[(0.0, 0.0); 2]; 32];
        apply_inverse_filtered_patches(
            &mut high,
            &[patch],
            &[32, 33],
            &[3],
            &mut high_modes,
            &mut high_bw,
            &mut high_history,
        )
        .unwrap();
        assert!(off.iter().zip(&high).any(|(a, b)| {
            (a.real[32] - b.real[32]).abs() > 1.0e-9
                || (a.imaginary[32] - b.imaginary[32]).abs() > 1.0e-9
        }));
        assert!(high
            .iter()
            .all(|slot| slot.real[32].is_finite() && slot.imaginary[32].is_finite()));
        assert_eq!(high_modes, vec![3]);
        assert!(high_bw[0] > 0.8);
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn inverse_filter_bandwidth_smoothing_matches_fdk() {
        let mut previous_modes = vec![1u8, 0, 2, 3];
        let mut previous = vec![0.20f64, 0.40, 0.80, 0.99];
        for modes in [vec![0u8, 1, 2, 3], vec![1u8, 3, 0, 2]] {
            let previous_fixed = previous
                .iter()
                .map(|value| (value * 2_147_483_648.0).round() as i32)
                .collect::<Vec<_>>();
            let mut c = vec![0i32; modes.len()];
            assert_eq!(
                unsafe {
                    fdk_aac_sys::fdk_sbr_inverse_filter_levels_test(
                        modes.as_ptr(),
                        previous_modes.as_ptr(),
                        previous_fixed.as_ptr(),
                        modes.len() as i32,
                        c.as_mut_ptr(),
                    )
                },
                0
            );
            let rust = smoothed_inverse_filter_bandwidths(&modes, &previous_modes, &previous);
            for (index, (&rust, &fixed)) in rust.iter().zip(&c).enumerate() {
                let expected = fixed as f64 / 2_147_483_648.0;
                assert!(
                    (rust - expected).abs() < 1.0e-7,
                    "band {index}: Rust {rust}, FDK {expected}"
                );
            }
            previous = rust;
            previous_modes = modes;
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn complex_autocorrelation_terms_match_fdk() {
        let series = (0..18)
            .map(|index| {
                (
                    (index as f64 * 0.37).sin() * 0.08,
                    (index as f64 * 0.23).cos() * 0.06,
                )
            })
            .collect::<Vec<_>>();
        let real = series
            .iter()
            .map(|sample| (sample.0 * 2_147_483_648.0).round() as i32)
            .collect::<Vec<_>>();
        let imaginary = series
            .iter()
            .map(|sample| (sample.1 * 2_147_483_648.0).round() as i32)
            .collect::<Vec<_>>();
        let mut c = [0i32; 9];
        let mut det_scale = 0;
        let scaling = unsafe {
            fdk_aac_sys::fdk_sbr_autocorrelation2_test(
                real.as_ptr(),
                imaginary.as_ptr(),
                series.len() as i32,
                c.as_mut_ptr(),
                &mut det_scale,
            )
        };
        assert!(scaling >= 0);
        assert!(det_scale >= 0);
        let mut r11 = 0.0;
        let mut r22 = 0.0;
        let mut r12 = (0.0, 0.0);
        let mut p1 = (0.0, 0.0);
        let mut p2 = (0.0, 0.0);
        for n in 2..series.len() {
            let x = series[n];
            let x1 = series[n - 1];
            let x2 = series[n - 2];
            r11 += complex_norm(x1);
            r22 += complex_norm(x2);
            r12 = complex_add(r12, complex_mul(x1, complex_conj(x2)));
            p1 = complex_add(p1, complex_mul(x, complex_conj(x1)));
            p2 = complex_add(p2, complex_mul(x, complex_conj(x2)));
        }
        let rust = [r11, r22, r12.0, r12.1, p1.0, p1.1, p2.0, p2.1];
        for index in 0..rust.len() {
            let rust_ratio = rust[index] / r11;
            let c_ratio = c[index] as f64 / c[0] as f64;
            assert!(
                (rust_ratio - c_ratio).abs() < 2.0e-5,
                "term {index}: Rust {rust_ratio}, FDK {c_ratio}"
            );
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn inverse_filtered_patch_output_matches_fdk() {
        let series = (0..18)
            .map(|index| {
                (
                    (index as f64 * 0.37).sin() * 0.04,
                    (index as f64 * 0.23).cos() * 0.03,
                )
            })
            .collect::<Vec<_>>();
        let real = series
            .iter()
            .map(|sample| (sample.0 * 2_147_483_648.0).round() as i32)
            .collect::<Vec<_>>();
        let imaginary = series
            .iter()
            .map(|sample| (sample.1 * 2_147_483_648.0).round() as i32)
            .collect::<Vec<_>>();
        let mut c_real = vec![0i32; 16];
        let mut c_imaginary = vec![0i32; 16];
        let mut high_band_scale = 0;
        assert_eq!(
            unsafe {
                fdk_aac_sys::fdk_sbr_inverse_filtered_patch_test(
                    real.as_ptr(),
                    imaginary.as_ptr(),
                    series.len() as i32,
                    3,
                    3,
                    (0.98 * 2_147_483_648.0) as i32,
                    c_real.as_mut_ptr(),
                    c_imaginary.as_mut_ptr(),
                    &mut high_band_scale,
                )
            },
            0
        );
        let mut slots = series[2..]
            .iter()
            .map(|sample| {
                let mut slot = QmfSlot {
                    real: vec![0.0; 64],
                    imaginary: vec![0.0; 64],
                };
                slot.real[5] = sample.0;
                slot.imaginary[5] = sample.1;
                slot
            })
            .collect::<Vec<_>>();
        let mut previous_modes = vec![3];
        let mut previous_bandwidths = vec![0.98];
        let mut history = vec![[(0.0, 0.0); 2]; 32];
        history[5] = [series[0], series[1]];
        apply_inverse_filtered_patches(
            &mut slots,
            &[SbrPatch {
                source_start_band: 5,
                target_start_band: 32,
                band_count: 1,
            }],
            &[32, 33],
            &[3],
            &mut previous_modes,
            &mut previous_bandwidths,
            &mut history,
        )
        .unwrap();
        assert_eq!(high_band_scale, -2);
        // Undo the LPC headroom stored in hb_scale to compare with Rust's
        // exponent-free floating-point QMF values.
        let scale = 2.0f64.powi(-high_band_scale);
        let c = c_real
            .iter()
            .zip(&c_imaginary)
            .map(|(&real, &imaginary)| {
                (
                    real as f64 / 2_147_483_648.0 * scale,
                    imaginary as f64 / 2_147_483_648.0 * scale,
                )
            })
            .collect::<Vec<_>>();
        let mut dot = 0.0;
        let mut rust_energy = 0.0;
        let mut c_energy = 0.0;
        for (slot, &(real, imaginary)) in slots.iter().zip(&c) {
            dot += slot.real[32] * real + slot.imaginary[32] * imaginary;
            rust_energy += slot.real[32].powi(2) + slot.imaginary[32].powi(2);
            c_energy += real.powi(2) + imaginary.powi(2);
        }
        let correlation = dot / (rust_energy * c_energy).sqrt();
        let rms_ratio = (rust_energy / c_energy).sqrt();
        assert!(correlation > 0.999, "correlation {correlation}");
        assert!((0.98..=1.02).contains(&rms_ratio), "RMS ratio {rms_ratio}");
    }

    #[test]
    fn adjusts_qmf_band_energy_to_envelope_targets() {
        let tables = LdSbrFrequencyTables {
            master: vec![32, 34, 36],
            high: vec![32, 34, 36],
            low: vec![32, 36],
            noise: vec![32, 36],
        };
        let control = LdSbrChannelControl {
            grid: crate::ld_sbr::LdSbrGrid {
                transient: false,
                amp_resolution: None,
                borders: vec![0, 2],
                frequency_resolution: vec![true],
                transient_envelope: None,
                noise_borders: vec![0, 2],
            },
            envelope_time_domain: vec![false],
            noise_time_domain: vec![false],
        };
        let values = LdSbrDequantizedChannel {
            envelope_energy: vec![vec![4.0, 9.0]],
            noise_energy: vec![vec![1.0]],
        };
        let mut slots = vec![
            QmfSlot {
                real: vec![1.0; 64],
                imaginary: vec![0.0; 64],
            };
            2
        ];
        apply_envelope_gains(&mut slots, &control, &tables, &values).unwrap();
        for (range, expected) in [(32..34, 4.0), (34..36, 9.0)] {
            let mean = slots
                .iter()
                .flat_map(|slot| range.clone().map(|band| slot.real[band].powi(2)))
                .sum::<f64>()
                / 4.0;
            assert!((mean - expected).abs() < 1.0e-12);
        }
    }

    #[test]
    fn envelope_adjustment_rejects_every_malformed_layout() {
        let tables = LdSbrFrequencyTables {
            master: vec![32, 34, 36],
            high: vec![32, 34, 36],
            low: vec![32, 36],
            noise: vec![32, 36],
        };
        let control = LdSbrChannelControl {
            grid: crate::ld_sbr::LdSbrGrid {
                transient: false,
                amp_resolution: None,
                borders: vec![0, 2],
                frequency_resolution: vec![true],
                transient_envelope: None,
                noise_borders: vec![0, 2],
            },
            envelope_time_domain: vec![false],
            noise_time_domain: vec![false],
        };
        let values = LdSbrDequantizedChannel {
            envelope_energy: vec![vec![1.0, 1.0]],
            noise_energy: vec![vec![1.0]],
        };
        let slots = vec![
            QmfSlot {
                real: vec![1.0; 64],
                imaginary: vec![0.0; 64],
            };
            2
        ];

        let mut gains = Vec::new();
        assert_eq!(
            apply_envelope_gains_limited(
                &mut slots.clone(),
                &control,
                &tables,
                &LdSbrDequantizedChannel {
                    envelope_energy: vec![],
                    noise_energy: vec![],
                },
                0,
                false,
                &mut gains,
            ),
            Err(QmfError::EnvelopeLayoutMismatch)
        );
        assert_eq!(
            apply_envelope_gains_limited(
                &mut slots.clone(),
                &control,
                &tables,
                &values,
                4,
                false,
                &mut gains,
            ),
            Err(QmfError::EnvelopeLayoutMismatch)
        );
        let mut invalid_slots = control.clone();
        invalid_slots.grid.borders = vec![0, 3];
        assert_eq!(
            apply_envelope_gains(&mut slots.clone(), &invalid_slots, &tables, &values),
            Err(QmfError::EnvelopeLayoutMismatch)
        );
        let mut low_resolution = control.clone();
        low_resolution.grid.frequency_resolution = vec![false];
        assert_eq!(
            apply_envelope_gains(&mut slots.clone(), &low_resolution, &tables, &values),
            Err(QmfError::EnvelopeLayoutMismatch)
        );
        assert_eq!(
            derive_limiter_borders(&tables, &[], 4),
            Err(QmfError::EnvelopeLayoutMismatch)
        );
        assert_eq!(
            apply_noise_and_harmonics(
                &mut slots.clone(),
                &control,
                &tables,
                &values,
                &[],
                &mut 1,
                &mut 0,
                &mut Vec::new(),
            ),
            Err(QmfError::EnvelopeLayoutMismatch)
        );
        assert_eq!(
            target_envelope_energy(&control, &tables, &values, 3, 32),
            Err(QmfError::EnvelopeLayoutMismatch)
        );
        assert_eq!(
            target_envelope_energy(&low_resolution, &tables, &values, 0, 32),
            Ok(1.0)
        );

        let smoothed = smoothed_inverse_filter_bandwidths(&[1], &[1], &[1.0]);
        assert_eq!(smoothed, vec![0.8125]);
    }

    #[test]
    fn limiter_caps_envelope_power_gain_and_smoothing_tracks_state() {
        let tables = LdSbrFrequencyTables {
            master: vec![32, 33, 34],
            high: vec![32, 33, 34],
            low: vec![32, 33, 34],
            noise: vec![32, 34],
        };
        let control = LdSbrChannelControl {
            grid: crate::ld_sbr::LdSbrGrid {
                transient: false,
                amp_resolution: None,
                borders: vec![0, 2],
                frequency_resolution: vec![true],
                transient_envelope: None,
                noise_borders: vec![0, 2],
            },
            envelope_time_domain: vec![false],
            noise_time_domain: vec![false],
        };
        let values = LdSbrDequantizedChannel {
            envelope_energy: vec![vec![100.0, 1.0]],
            noise_energy: vec![vec![1.0]],
        };
        let mut slots = vec![
            QmfSlot {
                real: vec![1.0; 64],
                imaginary: vec![0.0; 64],
            };
            2
        ];
        for slot in &mut slots {
            slot.real[33] = 10.0;
        }
        let mut state = vec![1.0; 64];
        let mut clip_ratios = Vec::new();
        apply_envelope_gains_with_limiter_borders(
            &mut slots,
            &control,
            &tables,
            &values,
            1,
            false,
            &mut state,
            &[32, 34],
            false,
            Some(&mut clip_ratios),
        )
        .unwrap();
        let boost = 2.511_886_432;
        let boosted_high = boost;
        let boosted_low = boost;
        assert!((slots[0].real[32].powi(2) - boosted_high).abs() < 1.0e-12);
        assert!((slots[0].real[33].powi(2) - boosted_low).abs() < 1.0e-12);
        assert!((clip_ratios[0][32] - 0.01).abs() < 1.0e-12);
        assert_eq!(clip_ratios[0][33], 1.0);

        let zero_values = LdSbrDequantizedChannel {
            envelope_energy: vec![vec![0.0, 0.0]],
            noise_energy: vec![vec![0.0]],
        };
        let mut zero_slots = vec![
            QmfSlot {
                real: vec![1.0; 64],
                imaginary: vec![0.0; 64],
            };
            2
        ];
        let mut zero_clip_ratios = Vec::new();
        apply_envelope_gains_with_limiter_borders(
            &mut zero_slots,
            &control,
            &tables,
            &zero_values,
            1,
            false,
            &mut Vec::new(),
            &[32, 34],
            false,
            Some(&mut zero_clip_ratios),
        )
        .unwrap();
        assert_eq!(zero_clip_ratios[0][32], 1.0);

        let mut slots = vec![
            QmfSlot {
                real: vec![1.0; 64],
                imaginary: vec![0.0; 64],
            };
            2
        ];
        let mut state = vec![1.0; 64];
        apply_envelope_gains_limited(&mut slots, &control, &tables, &values, 2, true, &mut state)
            .unwrap();
        assert!((slots[0].real[32] - 4.0).abs() < 1.0e-12);
        let second = SMOOTHING_RATIOS[1] + (1.0 - SMOOTHING_RATIOS[1]) * 10.0;
        assert!((slots[1].real[32] - second).abs() < 1.0e-12);
        assert!((state[32] - 10.0).abs() < 1.0e-12);

        let mut startup_slots = vec![
            QmfSlot {
                real: vec![1.0; 64],
                imaginary: vec![0.0; 64],
            };
            2
        ];
        let mut startup_state = Vec::new();
        apply_envelope_gains_limited(
            &mut startup_slots,
            &control,
            &tables,
            &values,
            2,
            true,
            &mut startup_state,
        )
        .unwrap();
        assert!((startup_slots[0].real[32] - 10.0).abs() < 1.0e-12);
        assert!((startup_slots[1].real[32] - 10.0).abs() < 1.0e-12);

        let mut attack_control = control.clone();
        attack_control.grid.transient_envelope = Some(0);
        let mut attack_slots = vec![
            QmfSlot {
                real: vec![1.0; 64],
                imaginary: vec![0.0; 64],
            };
            2
        ];
        let mut state = vec![1.0; 64];
        apply_envelope_gains_limited(
            &mut attack_slots,
            &attack_control,
            &tables,
            &values,
            2,
            true,
            &mut state,
        )
        .unwrap();
        assert!((attack_slots[0].real[32] - 10.0).abs() < 1.0e-12);

        let mut carried_attack = control.clone();
        carried_attack.grid.transient_envelope = None;
        let mut carried_slots = vec![
            QmfSlot {
                real: vec![1.0; 64],
                imaginary: vec![0.0; 64],
            };
            2
        ];
        let mut state = vec![1.0; 64];
        apply_envelope_gains_with_limiter_borders(
            &mut carried_slots,
            &carried_attack,
            &tables,
            &values,
            2,
            true,
            &mut state,
            &[32, 34],
            true,
            None,
        )
        .unwrap();
        assert!((carried_slots[0].real[32] - 10.0).abs() < 1.0e-12);
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn limiter_gain_clip_and_boost_match_fdk_full_component_chain() {
        fn mantissa_exponent(value: f64) -> (i32, i8) {
            if value == 0.0 {
                return (0, 0);
            }
            let exponent = value.abs().log2().floor() as i32 + 1;
            let mantissa = (value * 2.0f64.powi(31 - exponent)).round();
            (
                mantissa.clamp(i32::MIN as f64, i32::MAX as f64) as i32,
                exponent as i8,
            )
        }
        fn value(mantissa: i32, exponent: i8) -> f64 {
            mantissa as f64 / 2_147_483_648.0 * 2.0f64.powi(exponent as i32)
        }

        let reference = [100.0, 1.0];
        let estimated = [1.0, 100.0];
        let reference_parts = reference.map(mantissa_exponent);
        let estimated_parts = estimated.map(mantissa_exponent);
        let reference_m = reference_parts.map(|part| part.0);
        let reference_e = reference_parts.map(|part| part.1);
        let estimated_m = estimated_parts.map(|part| part.0);
        let estimated_e = estimated_parts.map(|part| part.1);
        let noise_m = [0; 2];
        let noise_e = [0; 2];
        let sine_present = [0; 2];
        let sine_mapped = [0; 2];
        let mut gain_m = [0; 2];
        let mut gain_e = [0; 2];
        let mut noise_level_m = [0; 2];
        let mut noise_level_e = [0; 2];
        let mut sine_m = [0; 2];
        let mut sine_e = [0; 2];
        assert_eq!(
            unsafe {
                fdk_aac_sys::fdk_sbr_limited_components_test(
                    reference_m.as_ptr(),
                    reference_e.as_ptr(),
                    estimated_m.as_ptr(),
                    estimated_e.as_ptr(),
                    noise_m.as_ptr(),
                    noise_e.as_ptr(),
                    sine_present.as_ptr(),
                    sine_mapped.as_ptr(),
                    2,
                    1,
                    1,
                    gain_m.as_mut_ptr(),
                    gain_e.as_mut_ptr(),
                    noise_level_m.as_mut_ptr(),
                    noise_level_e.as_mut_ptr(),
                    sine_m.as_mut_ptr(),
                    sine_e.as_mut_ptr(),
                )
            },
            0
        );

        let tables = LdSbrFrequencyTables {
            master: vec![32, 33, 34],
            high: vec![32, 33, 34],
            low: vec![32, 33, 34],
            noise: vec![32, 34],
        };
        let control = LdSbrChannelControl {
            grid: crate::ld_sbr::LdSbrGrid {
                transient: false,
                amp_resolution: None,
                borders: vec![0, 2],
                frequency_resolution: vec![true],
                transient_envelope: None,
                noise_borders: vec![0, 2],
            },
            envelope_time_domain: vec![false],
            noise_time_domain: vec![false],
        };
        let values = LdSbrDequantizedChannel {
            envelope_energy: vec![reference.to_vec()],
            noise_energy: vec![vec![0.0]],
        };
        let mut slots = vec![
            QmfSlot {
                real: vec![0.0; 64],
                imaginary: vec![0.0; 64],
            };
            2
        ];
        for slot in &mut slots {
            slot.real[32] = 1.0;
            slot.real[33] = 10.0;
        }
        apply_envelope_gains_with_limiter_borders(
            &mut slots,
            &control,
            &tables,
            &values,
            1,
            false,
            &mut Vec::new(),
            &[32, 34],
            false,
            None,
        )
        .unwrap();
        let rust_power_gains = [slots[0].real[32].powi(2), slots[0].real[33].powi(2) / 100.0];
        for band in 0..2 {
            let fdk_gain = value(gain_m[band], gain_e[band]);
            let relative_error = (rust_power_gains[band] - fdk_gain).abs() / fdk_gain.max(1.0e-30);
            assert!(
                // This synthetic fixture expresses energies in unnormalised
                // units, so FDK's explicit +1 floor is visible as at most a
                // one-percent difference. Real QMF energy uses PCM_ENERGY_FLOOR.
                relative_error < 0.011,
                "band {band}: Rust gain {}, FDK gain {}, relative error {}",
                rust_power_gains[band],
                fdk_gain,
                relative_error
            );
            assert_eq!(noise_level_m[band], 0);
            assert_eq!(sine_m[band], 0);
        }
    }

    #[test]
    fn adds_deterministic_noise_and_harmonics() {
        let tables = LdSbrFrequencyTables {
            master: vec![32, 34],
            high: vec![32, 34],
            low: vec![32, 34],
            noise: vec![32, 34],
        };
        let control = LdSbrChannelControl {
            grid: crate::ld_sbr::LdSbrGrid {
                transient: false,
                amp_resolution: None,
                borders: vec![0, 2],
                frequency_resolution: vec![true],
                transient_envelope: None,
                noise_borders: vec![0, 2],
            },
            envelope_time_domain: vec![false],
            noise_time_domain: vec![false],
        };
        let values = LdSbrDequantizedChannel {
            envelope_energy: vec![vec![4.0]],
            noise_energy: vec![vec![1.0]],
        };
        let blank = vec![
            QmfSlot {
                real: vec![0.0; 64],
                imaginary: vec![0.0; 64],
            };
            2
        ];
        let mut first = blank.clone();
        let mut second = blank;
        let mut seed_a = 7;
        let mut seed_b = 7;
        let mut harmonic_phase_a = 0;
        let mut harmonic_phase_b = 0;
        let mut previous_a = Vec::new();
        let mut previous_b = Vec::new();
        apply_noise_and_harmonics(
            &mut first,
            &control,
            &tables,
            &values,
            &[true],
            &mut seed_a,
            &mut harmonic_phase_a,
            &mut previous_a,
        )
        .unwrap();
        apply_noise_and_harmonics(
            &mut second,
            &control,
            &tables,
            &values,
            &[true],
            &mut seed_b,
            &mut harmonic_phase_b,
            &mut previous_b,
        )
        .unwrap();
        assert_eq!(first, second);
        assert!(first
            .iter()
            .flat_map(|slot| slot.real.iter().chain(&slot.imaginary))
            .all(|value| value.is_finite()));
        assert!(first
            .iter()
            .any(|slot| slot.real[33] != 0.0 || slot.imaginary[33] != 0.0));
        let sine_amplitude = 2.0;
        assert!((first[0].real[33] - sine_amplitude).abs() < 1.0e-12);
        assert!((first[1].imaginary[33] + sine_amplitude).abs() < 1.0e-12);
        assert_eq!(harmonic_phase_a, 2);
        assert!(previous_a[33]);

        let mut continued = vec![
            QmfSlot {
                real: vec![0.0; 64],
                imaginary: vec![0.0; 64],
            };
            2
        ];
        apply_noise_and_harmonics(
            &mut continued,
            &control,
            &tables,
            &values,
            &[true],
            &mut seed_a,
            &mut harmonic_phase_a,
            &mut previous_a,
        )
        .unwrap();
        assert!(continued
            .iter()
            .flat_map(|slot| slot.real.iter().chain(&slot.imaginary))
            .all(|value| value.is_finite()));

        let mut noise_only = vec![
            QmfSlot {
                real: vec![0.0; 64],
                imaginary: vec![0.0; 64],
            };
            2
        ];
        let mut phase_index = 7;
        let mut harmonic_phase = 0;
        let mut previous = Vec::new();
        apply_noise_and_harmonics(
            &mut noise_only,
            &control,
            &tables,
            &values,
            &[false],
            &mut phase_index,
            &mut harmonic_phase,
            &mut previous,
        )
        .unwrap();
        let amplitude = 2.0;
        assert!((noise_only[0].real[32] - RANDOM_PHASE[8][0] * amplitude).abs() < 1.0e-12);
        assert!((noise_only[0].imaginary[32] - RANDOM_PHASE[8][1] * amplitude).abs() < 1.0e-12);
        assert_eq!(phase_index, 11);

        let mut attack_control = control.clone();
        attack_control.grid.transient_envelope = Some(0);
        let mut attack = vec![
            QmfSlot {
                real: vec![1.0; 64],
                imaginary: vec![0.0; 64],
            };
            2
        ];
        apply_noise_and_harmonics(
            &mut attack,
            &attack_control,
            &tables,
            &values,
            &[false],
            &mut 0,
            &mut 0,
            &mut Vec::new(),
        )
        .unwrap();
        assert!((attack[0].real[32] - 2.511_886_432f64.sqrt()).abs() < 1.0e-12);

        let mut harmonic_attack = vec![
            QmfSlot {
                real: vec![1.0; 64],
                imaginary: vec![0.0; 64],
            };
            2
        ];
        apply_noise_and_harmonics(
            &mut harmonic_attack,
            &attack_control,
            &tables,
            &values,
            &[true],
            &mut 0,
            &mut 0,
            &mut Vec::new(),
        )
        .unwrap();
        let limited_boost = 2.511_886_432f64;
        assert!((harmonic_attack[0].real[32] - (0.5 * limited_boost).sqrt()).abs() < 1.0e-12);

        let mut smoothed_noise = vec![
            QmfSlot {
                real: vec![0.0; 64],
                imaginary: vec![0.0; 64],
            };
            2
        ];
        let mut previous_noise = vec![4.0; 64];
        let mut seed = 7;
        apply_noise_and_harmonics_with_limiter_borders(
            &mut smoothed_noise,
            &control,
            &tables,
            &values,
            &[false],
            &mut seed,
            &mut 0,
            &mut Vec::new(),
            &[32, 34],
            None,
            true,
            false,
            Some(&mut previous_noise),
        )
        .unwrap();
        let expected_noise = SMOOTHING_RATIOS[0] * 4.0 + (1.0 - SMOOTHING_RATIOS[0]) * 2.0;
        assert!((smoothed_noise[0].real[32] - RANDOM_PHASE[8][0] * expected_noise).abs() < 1.0e-12);
        assert!((previous_noise[32] - 2.0).abs() < 1.0e-12);
    }

    #[test]
    fn low_power_alias_reduction_preserves_group_energy_and_real_rendering() {
        let mut gains = [0.25, 4.0, 1.0];
        let mut estimated = [0.0; 64];
        estimated[32..35].copy_from_slice(&[2.0, 1.0, 3.0]);
        let mut alias = [0.0; 64];
        alias[33] = 1.0;
        alias[34] = 0.5;
        let before = gains
            .iter()
            .enumerate()
            .map(|(index, gain)| gain * estimated[32 + index])
            .sum::<f64>();
        reduce_aliasing_power_gains(32, &mut gains, &estimated, &alias, &[true; 3]);
        let after = gains
            .iter()
            .enumerate()
            .map(|(index, gain)| gain * estimated[32 + index])
            .sum::<f64>();
        assert!((after - before).abs() < 1.0e-12);
        assert!(gains.iter().all(|gain| gain.is_finite() && *gain >= 0.0));

        let tables = LdSbrFrequencyTables {
            master: vec![32, 34],
            high: vec![32, 34],
            low: vec![32, 34],
            noise: vec![32, 34],
        };
        let control = LdSbrChannelControl {
            grid: crate::ld_sbr::LdSbrGrid {
                transient: false,
                amp_resolution: None,
                borders: vec![0, 2],
                frequency_resolution: vec![true],
                transient_envelope: None,
                noise_borders: vec![0, 2],
            },
            envelope_time_domain: vec![false],
            noise_time_domain: vec![false],
        };
        let values = LdSbrDequantizedChannel {
            envelope_energy: vec![vec![4.0]],
            noise_energy: vec![vec![1.0]],
        };
        let mut slots = vec![
            QmfSlot {
                real: vec![0.0; 64],
                imaginary: vec![7.0; 64],
            };
            2
        ];
        let mut seed = 7;
        let mut phase = 1;
        apply_noise_and_harmonics_with_limiter_borders_mode(
            &mut slots,
            &control,
            &tables,
            &values,
            &[true],
            &mut seed,
            &mut phase,
            &mut Vec::new(),
            &[32, 34],
            None,
            true,
            false,
            None,
            true,
            false,
        )
        .unwrap();
        assert!(slots
            .iter()
            .all(|slot| slot.imaginary[32..34].iter().all(|value| *value == 0.0)));
        assert!(slots
            .iter()
            .any(|slot| { slot.real[32] != 0.0 || slot.real[33] != 0.0 || slot.real[34] != 0.0 }));
    }

    #[test]
    fn qmf_synthesis_produces_expected_frame_sizes_and_state_tail() {
        for channels in [32, 64] {
            let mut synthesis = LdSbrQmfSynthesis::new(channels).unwrap();
            let zero_slots = vec![
                QmfSlot {
                    real: vec![0.0; 64],
                    imaginary: vec![0.0; 64],
                };
                16
            ];
            let zero = synthesis.process_frame(&zero_slots).unwrap();
            assert_eq!(zero.len(), channels * 16);
            assert!(zero.iter().all(|value| *value == 0.0));
            let mut impulse_slots = zero_slots.clone();
            impulse_slots[0].real[0] = 1.0;
            let impulse = synthesis.process_frame(&impulse_slots).unwrap();
            let tail = synthesis.process_frame(&zero_slots).unwrap();
            assert!(impulse.iter().chain(&tail).all(|value| value.is_finite()));
            assert!(impulse.iter().chain(&tail).any(|value| *value != 0.0));
        }
    }

    #[test]
    fn cldfb_analysis_synthesis_and_constructor_errors_are_total() {
        assert_eq!(
            LdSbrProcessingError::from(LdSbrError::UnexpectedEof),
            LdSbrProcessingError::Syntax(LdSbrError::UnexpectedEof)
        );
        assert_eq!(
            LdSbrProcessingError::from(QmfError::InvalidTimeStep(0)),
            LdSbrProcessingError::Qmf(QmfError::InvalidTimeStep(0))
        );
        assert_eq!(
            LdSbrQmfAnalysis::new_with_channels(20).unwrap_err(),
            QmfError::UnsupportedChannelCount(20)
        );
        assert_eq!(
            LdSbrQmfSynthesis::new(16).unwrap_err(),
            QmfError::UnsupportedChannelCount(16)
        );
        assert_eq!(
            LdSbrQmfSynthesis::new_cldfb(16).unwrap_err(),
            QmfError::UnsupportedChannelCount(16)
        );

        let mut analysis = LdSbrQmfAnalysis::default();
        assert_eq!(
            analysis.process_frame(&[0.0]).unwrap_err(),
            QmfError::InvalidSampleCount(1)
        );
        assert_eq!(
            analysis.process_slot(&[0.0]).unwrap_err(),
            QmfError::InvalidSampleCount(1)
        );
        let mut synthesis = LdSbrQmfSynthesis::new(32).unwrap();
        assert_eq!(
            synthesis
                .process_slot(&QmfSlot {
                    real: vec![0.0; 31],
                    imaginary: vec![0.0; 32],
                })
                .unwrap_err(),
            QmfError::InvalidSubbandCount {
                expected: 32,
                actual: 31,
            }
        );

        let input = (0..64)
            .map(|index| (index as f64 * 0.17).sin() * 0.01)
            .collect::<Vec<_>>();
        let mut analysis = LdSbrQmfAnalysis::new_cldfb_32();
        let slots = analysis.process_frame(&input).unwrap();
        assert_eq!(slots.len(), 2);
        assert!(slots
            .iter()
            .flat_map(|slot| slot.real.iter().chain(&slot.imaginary))
            .all(|value| value.is_finite()));
        let output = LdSbrQmfSynthesis::new_cldfb_32()
            .process_frame(&slots)
            .unwrap();
        assert_eq!(output.len(), input.len());
        assert!(output.iter().all(|value| value.is_finite()));

        let slots64 = vec![QmfSlot {
            real: vec![0.0; 64],
            imaginary: vec![0.0; 64],
        }];
        let output64 = LdSbrQmfSynthesis::new_cldfb(64)
            .unwrap()
            .process_frame(&slots64)
            .unwrap();
        assert_eq!(output64, vec![0.0; 64]);
    }

    #[test]
    fn usac_analysis_supports_sixteen_and_twenty_four_band_qmf() {
        for channels in [16, 24] {
            let mut analysis = LdSbrQmfAnalysis::new_with_channels(channels).unwrap();
            let mut impulse = vec![0.0; channels * 2];
            impulse[0] = 1.0;
            let slots = analysis.process_frame(&impulse).unwrap();
            assert_eq!(slots.len(), 2);
            assert!(slots.iter().all(|slot| {
                slot.real.len() == channels
                    && slot.imaginary.len() == channels
                    && slot
                        .real
                        .iter()
                        .chain(&slot.imaginary)
                        .all(|v| v.is_finite())
            }));
        }
        // FDK stores the symmetric 240-tap prototype in its 130-value
        // polyphase representation.
        assert_eq!(PROTOTYPE_24.len(), 130);
        assert_eq!(PHASE_COS_16.len(), 16);
        assert_eq!(PHASE_SIN_16.len(), 16);
        assert_eq!(PHASE_COS_24.len(), 24);
        assert_eq!(PHASE_SIN_24.len(), 24);
    }

    #[test]
    fn patching_rejects_invalid_frequency_and_inverse_filter_layouts() {
        let invalid_tables = LdSbrFrequencyTables {
            master: vec![4, 8],
            high: vec![4, 8],
            low: vec![4, 8],
            noise: vec![4, 8],
        };
        assert_eq!(
            derive_patches(&invalid_tables, 44_100),
            Err(LdSbrError::InvalidFrequencyRange)
        );

        let too_many_during_iteration = LdSbrFrequencyTables {
            master: (5..=64).collect(),
            high: (5..=64).collect(),
            low: vec![5, 64],
            noise: vec![5, 64],
        };
        assert_eq!(
            derive_patches(&too_many_during_iteration, 44_100),
            Err(LdSbrError::InvalidFrequencyRange)
        );

        let too_many_at_upper_border = LdSbrFrequencyTables {
            master: (5..=33).collect(),
            high: (5..=33).collect(),
            low: vec![5, 33],
            noise: vec![5, 33],
        };
        assert_eq!(
            derive_patches(&too_many_at_upper_border, 44_100),
            Err(LdSbrError::InvalidFrequencyRange)
        );

        let invalid_patch = SbrPatch {
            source_start_band: 32,
            target_start_band: 63,
            band_count: 2,
        };
        let mut slots = vec![QmfSlot {
            real: vec![0.0; 32],
            imaginary: vec![0.0; 32],
        }];
        assert_eq!(
            apply_patches(&mut slots, &[invalid_patch]),
            Err(QmfError::InvalidPatchBand {
                source: 32,
                target: 63,
            })
        );

        let mut previous_modes = Vec::new();
        let mut previous_bandwidths = Vec::new();
        let mut history = vec![[(0.0, 0.0); 2]; 32];
        assert_eq!(
            apply_inverse_filtered_patches(
                &mut slots,
                &[],
                &[32, 64],
                &[],
                &mut previous_modes,
                &mut previous_bandwidths,
                &mut history,
            ),
            Err(QmfError::InverseFilteringLayoutMismatch)
        );
        assert_eq!(
            apply_inverse_filtered_patches(
                &mut slots,
                &[invalid_patch],
                &[32, 64],
                &[0],
                &mut previous_modes,
                &mut previous_bandwidths,
                &mut history,
            ),
            Err(QmfError::InvalidPatchBand {
                source: 32,
                target: 63,
            })
        );
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn fdk_and_rust_qmf_analysis_subbands_correlate() {
        let input = (0..512)
            .map(|index| {
                let value =
                    (index as f64 * 0.071).sin() * 0.25 + (index as f64 * 0.193).cos() * 0.125;
                (value * 2_147_483_648.0) as i32
            })
            .collect::<Vec<_>>();
        let mut fdk_real = vec![0i32; 16 * 32];
        let mut fdk_imag = vec![0i32; 16 * 32];
        let mut scale = 0;
        let result = unsafe {
            fdk_aac_sys::fdk_qmf_analysis32_test(
                input.as_ptr(),
                input.len() as i32,
                fdk_real.as_mut_ptr(),
                fdk_imag.as_mut_ptr(),
                &mut scale,
            )
        };
        assert_eq!(result, 0);
        let normalized = input
            .iter()
            .map(|&sample| sample as f64 / 2_147_483_648.0)
            .collect::<Vec<_>>();
        let mut rust = LdSbrQmfAnalysis::new();
        let slots = rust.process_frame(&normalized).unwrap();
        let rust_values = slots
            .iter()
            .flat_map(|slot| slot.real.iter().zip(&slot.imaginary))
            .flat_map(|(&real, &imaginary)| [real, imaginary])
            .collect::<Vec<_>>();
        let fdk_values = fdk_real
            .iter()
            .zip(&fdk_imag)
            .flat_map(|(&real, &imaginary)| [real as f64, imaginary as f64])
            .collect::<Vec<_>>();
        let dot = rust_values
            .iter()
            .zip(&fdk_values)
            .map(|(&left, &right)| left * right)
            .sum::<f64>();
        let rust_energy = rust_values.iter().map(|value| value * value).sum::<f64>();
        let fdk_energy = fdk_values.iter().map(|value| value * value).sum::<f64>();
        let correlation = dot / (rust_energy * fdk_energy).sqrt();
        let normalized_rms_ratio =
            (rust_energy / fdk_energy).sqrt() * 2_147_483_648.0 * 2.0f64.powi(-scale);
        assert!(
            correlation > 0.985,
            "QMF analysis correlation {correlation}, scale {scale}"
        );
        assert!(
            (0.999..=1.001).contains(&normalized_rms_ratio),
            "QMF analysis normalized RMS ratio {normalized_rms_ratio}, scale {scale}"
        );
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn fdk_and_rust_usac_qmf_analysis_subbands_correlate() {
        for channels in [24usize, 16] {
            let input = (0..channels * 16)
                .map(|index| {
                    let value =
                        (index as f64 * 0.071).sin() * 0.25 + (index as f64 * 0.193).cos() * 0.125;
                    (value * 2_147_483_648.0) as i32
                })
                .collect::<Vec<_>>();
            let mut fdk_real = vec![0i32; input.len()];
            let mut fdk_imag = vec![0i32; input.len()];
            let mut scale = 0;
            assert_eq!(
                unsafe {
                    fdk_aac_sys::fdk_qmf_analysis_usac_test(
                        input.as_ptr(),
                        input.len() as i32,
                        channels as i32,
                        fdk_real.as_mut_ptr(),
                        fdk_imag.as_mut_ptr(),
                        &mut scale,
                    )
                },
                0
            );
            let normalized = input
                .iter()
                .map(|&sample| sample as f64 / 2_147_483_648.0)
                .collect::<Vec<_>>();
            let slots = LdSbrQmfAnalysis::new_with_channels(channels)
                .unwrap()
                .process_frame(&normalized)
                .unwrap();
            let rust_values = slots
                .iter()
                .flat_map(|slot| slot.real.iter().zip(&slot.imaginary))
                .flat_map(|(&real, &imaginary)| [real, imaginary])
                .collect::<Vec<_>>();
            let fdk_values = fdk_real
                .iter()
                .zip(&fdk_imag)
                .flat_map(|(&real, &imaginary)| [real as f64, imaginary as f64])
                .collect::<Vec<_>>();
            let dot = rust_values
                .iter()
                .zip(&fdk_values)
                .map(|(&left, &right)| left * right)
                .sum::<f64>();
            let rust_energy = rust_values.iter().map(|value| value * value).sum::<f64>();
            let fdk_energy = fdk_values.iter().map(|value| value * value).sum::<f64>();
            let correlation = dot / (rust_energy * fdk_energy).sqrt();
            let normalized_rms_ratio =
                (rust_energy / fdk_energy).sqrt() * 2_147_483_648.0 * 2.0f64.powi(-scale);
            assert!(
                correlation > 0.985,
                "{channels}-band QMF analysis correlation {correlation}, scale {scale}"
            );
            assert!(
                (0.999..=1.001).contains(&normalized_rms_ratio),
                "{channels}-band QMF normalized RMS ratio {normalized_rms_ratio}, scale {scale}"
            );
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn fdk_and_rust_qmf64_analysis_subbands_correlate() {
        let input = (0..1024)
            .map(|index| {
                let value =
                    (index as f64 * 0.071).sin() * 0.25 + (index as f64 * 0.193).cos() * 0.125;
                (value * 2_147_483_648.0) as i32
            })
            .collect::<Vec<_>>();
        let mut fdk_real = vec![0i32; 16 * 64];
        let mut fdk_imag = vec![0i32; 16 * 64];
        let mut scale = 0;
        assert_eq!(
            unsafe {
                fdk_aac_sys::fdk_qmf_analysis64_test(
                    input.as_ptr(),
                    input.len() as i32,
                    fdk_real.as_mut_ptr(),
                    fdk_imag.as_mut_ptr(),
                    &mut scale,
                )
            },
            0
        );
        let normalized = input
            .iter()
            .map(|&sample| sample as f64 / 2_147_483_648.0)
            .collect::<Vec<_>>();
        let slots = LdSbrQmfAnalysis::new_with_channels(64)
            .unwrap()
            .process_frame(&normalized)
            .unwrap();
        let rust_values = slots
            .iter()
            .flat_map(|slot| slot.real.iter().zip(&slot.imaginary))
            .flat_map(|(&real, &imaginary)| [real, imaginary])
            .collect::<Vec<_>>();
        let fdk_values = fdk_real
            .iter()
            .zip(&fdk_imag)
            .flat_map(|(&real, &imaginary)| [real as f64, imaginary as f64])
            .collect::<Vec<_>>();
        let dot = rust_values
            .iter()
            .zip(&fdk_values)
            .map(|(&left, &right)| left * right)
            .sum::<f64>();
        let rust_energy = rust_values.iter().map(|value| value * value).sum::<f64>();
        let fdk_energy = fdk_values.iter().map(|value| value * value).sum::<f64>();
        let correlation = dot / (rust_energy * fdk_energy).sqrt();
        let normalized_rms_ratio =
            (rust_energy / fdk_energy).sqrt() * 2_147_483_648.0 * 2.0f64.powi(-scale);
        assert!(
            correlation > 0.999_999,
            "64-band QMF analysis correlation {correlation}, scale {scale}"
        );
        assert!(
            (0.999..=1.001).contains(&normalized_rms_ratio),
            "64-band QMF analysis normalized RMS ratio {normalized_rms_ratio}, scale {scale}"
        );
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn fdk_and_rust_qmf64_cldfb_analysis_subbands_correlate() {
        let input = (0..1024)
            .map(|index| {
                let value =
                    (index as f64 * 0.071).sin() * 0.25 + (index as f64 * 0.193).cos() * 0.125;
                (value * 2_147_483_648.0) as i32
            })
            .collect::<Vec<_>>();
        let mut fdk_real = vec![0i32; 16 * 64];
        let mut fdk_imag = vec![0i32; 16 * 64];
        let mut scale = 0;
        assert_eq!(
            unsafe {
                fdk_aac_sys::fdk_qmf_analysis64_cldfb_test(
                    input.as_ptr(),
                    input.len() as i32,
                    fdk_real.as_mut_ptr(),
                    fdk_imag.as_mut_ptr(),
                    &mut scale,
                )
            },
            0
        );
        let normalized = input
            .iter()
            .map(|&sample| sample as f64 / 2_147_483_648.0)
            .collect::<Vec<_>>();
        let slots = LdSbrQmfAnalysis::new_cldfb(64)
            .unwrap()
            .process_frame(&normalized)
            .unwrap();
        let rust_values = slots
            .iter()
            .flat_map(|slot| slot.real.iter().zip(&slot.imaginary))
            .flat_map(|(&real, &imaginary)| [real, imaginary])
            .collect::<Vec<_>>();
        let fdk_values = fdk_real
            .iter()
            .zip(&fdk_imag)
            .flat_map(|(&real, &imaginary)| [real as f64, imaginary as f64])
            .collect::<Vec<_>>();
        let dot = rust_values
            .iter()
            .zip(&fdk_values)
            .map(|(&left, &right)| left * right)
            .sum::<f64>();
        let rust_energy = rust_values.iter().map(|value| value * value).sum::<f64>();
        let fdk_energy = fdk_values.iter().map(|value| value * value).sum::<f64>();
        let correlation = dot / (rust_energy * fdk_energy).sqrt();
        let normalized_rms_ratio =
            (rust_energy / fdk_energy).sqrt() * 2_147_483_648.0 * 2.0f64.powi(-scale);
        assert!(
            correlation > 0.999_999,
            "64-band CLDFB analysis correlation {correlation}, scale {scale}"
        );
        assert!(
            (0.999..=1.001).contains(&normalized_rms_ratio),
            "64-band CLDFB normalized RMS ratio {normalized_rms_ratio}, scale {scale}"
        );
        assert_eq!(scale, -8);
        let mantissa_factor = 2.0_f64.powi(31 - scale);
        let maximum_mantissa_error = rust_values
            .iter()
            .zip(&fdk_values)
            .map(|(&rust, &fdk)| (rust.mul_add(mantissa_factor, -fdk)).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            maximum_mantissa_error <= 4_096.0,
            "64-band CLDFB maximum mantissa error {maximum_mantissa_error}"
        );
        for band in 0..64 {
            let rust_band = slots
                .iter()
                .map(|slot| {
                    slot.real[band] * slot.real[band] + slot.imaginary[band] * slot.imaginary[band]
                })
                .sum::<f64>();
            let fdk_band = (0..16)
                .map(|slot| {
                    let r = fdk_real[slot * 64 + band] as f64;
                    let i = fdk_imag[slot * 64 + band] as f64;
                    r * r + i * i
                })
                .sum::<f64>();
            if fdk_band > 1.0 {
                let ratio = (rust_band / fdk_band).sqrt() * 2_147_483_648.0 * 2.0f64.powi(-scale);
                assert!(
                    (0.998..=1.002).contains(&ratio),
                    "64-band CLDFB subband {band} normalized RMS ratio {ratio}"
                );
            }
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn raw_pcm_qmf64_cldfb_mantissa_scaling_matches_c() {
        let input = (0..1024)
            .map(|index| {
                let value = (index as f64 * 0.071).sin() * 12_000.0
                    + (index as f64 * 0.193).cos() * 4_000.0;
                value as i16
            })
            .collect::<Vec<_>>();
        let mut c_real = vec![0_i32; 16 * 64];
        let mut c_imaginary = vec![0_i32; 16 * 64];
        let mut scale = 0;
        assert_eq!(
            unsafe {
                fdk_aac_sys::fdk_qmf_analysis64_cldfb_pcm_test(
                    input.as_ptr(),
                    input.len() as i32,
                    c_real.as_mut_ptr(),
                    c_imaginary.as_mut_ptr(),
                    &mut scale,
                )
            },
            0
        );
        assert_eq!(scale, -8);
        let rust = LdSbrQmfAnalysis::new_cldfb(64)
            .unwrap()
            .process_frame(
                &input
                    .iter()
                    .map(|&value| f64::from(value))
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let factor = 2.0_f64.powi(16 - scale);
        let maximum_error = rust
            .iter()
            .flat_map(|slot| slot.real.iter().zip(&slot.imaginary))
            .flat_map(|(&real, &imaginary)| [real, imaginary])
            .zip(
                c_real
                    .iter()
                    .zip(&c_imaginary)
                    .flat_map(|(&real, &imaginary)| [real, imaginary]),
            )
            .map(|(rust, c)| rust.mul_add(factor, -f64::from(c)).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            maximum_error == 0.0,
            "maximum PCM mantissa error {maximum_error}"
        );
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn fdk_and_rust_low_power_qmf_analysis_subbands_correlate() {
        let input = (0..512)
            .map(|index| {
                let value =
                    (index as f64 * 0.071).sin() * 0.25 + (index as f64 * 0.193).cos() * 0.125;
                (value * 2_147_483_648.0) as i32
            })
            .collect::<Vec<_>>();
        let mut fdk_real = vec![0i32; 16 * 32];
        let mut scale = 0;
        assert_eq!(
            unsafe {
                fdk_aac_sys::fdk_qmf_analysis32_lp_test(
                    input.as_ptr(),
                    input.len() as i32,
                    fdk_real.as_mut_ptr(),
                    &mut scale,
                )
            },
            0
        );
        let normalized = input
            .iter()
            .map(|&sample| sample as f64 / 2_147_483_648.0)
            .collect::<Vec<_>>();
        let mut rust = LdSbrQmfAnalysis::new();
        rust.set_low_power(true);
        let slots = rust.process_frame(&normalized).unwrap();
        assert!(slots
            .iter()
            .flat_map(|slot| &slot.imaginary)
            .all(|&value| value == 0.0));
        let rust_values = slots
            .iter()
            .flat_map(|slot| slot.real.iter().copied())
            .collect::<Vec<_>>();
        let fdk_values = fdk_real
            .iter()
            .map(|&value| f64::from(value))
            .collect::<Vec<_>>();
        let dot = rust_values
            .iter()
            .zip(&fdk_values)
            .map(|(&left, &right)| left * right)
            .sum::<f64>();
        let rust_energy = rust_values.iter().map(|value| value * value).sum::<f64>();
        let fdk_energy = fdk_values.iter().map(|value| value * value).sum::<f64>();
        let correlation = dot / (rust_energy * fdk_energy).sqrt();
        let normalized_rms_ratio =
            (rust_energy / fdk_energy).sqrt() * 2_147_483_648.0 * 2.0f64.powi(-scale);
        assert!(
            correlation > 0.995,
            "LP QMF analysis correlation {correlation}, scale {scale}"
        );
        assert!(
            (0.98..=1.02).contains(&normalized_rms_ratio),
            "LP QMF analysis normalized RMS ratio {normalized_rms_ratio}, scale {scale}"
        );
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn fdk_and_rust_cldfb_analysis_subbands_correlate() {
        let input = (0..512)
            .map(|index| {
                let value =
                    (index as f64 * 0.071).sin() * 0.25 + (index as f64 * 0.193).cos() * 0.125;
                (value * 2_147_483_648.0) as i32
            })
            .collect::<Vec<_>>();
        let mut fdk_real = vec![0i32; 16 * 32];
        let mut fdk_imag = vec![0i32; 16 * 32];
        let mut scale = 0;
        assert_eq!(
            unsafe {
                fdk_aac_sys::fdk_qmf_analysis32_cldfb_test(
                    input.as_ptr(),
                    input.len() as i32,
                    fdk_real.as_mut_ptr(),
                    fdk_imag.as_mut_ptr(),
                    &mut scale,
                )
            },
            0
        );
        let normalized = input
            .iter()
            .map(|&sample| sample as f64 / 2_147_483_648.0)
            .collect::<Vec<_>>();
        let slots = LdSbrQmfAnalysis::new_cldfb_32()
            .process_frame(&normalized)
            .unwrap();
        let rust_values = slots
            .iter()
            .flat_map(|slot| slot.real.iter().zip(&slot.imaginary))
            .flat_map(|(&real, &imaginary)| [real, imaginary])
            .collect::<Vec<_>>();
        let fdk_values = fdk_real
            .iter()
            .zip(&fdk_imag)
            .flat_map(|(&real, &imaginary)| [real as f64, imaginary as f64])
            .collect::<Vec<_>>();
        let dot = rust_values
            .iter()
            .zip(&fdk_values)
            .map(|(&left, &right)| left * right)
            .sum::<f64>();
        let rust_energy = rust_values.iter().map(|value| value * value).sum::<f64>();
        let fdk_energy = fdk_values.iter().map(|value| value * value).sum::<f64>();
        let correlation = dot / (rust_energy * fdk_energy).sqrt();
        let normalized_rms_ratio =
            (rust_energy / fdk_energy).sqrt() * 2_147_483_648.0 * 2.0f64.powi(-scale);
        assert!(
            correlation > 0.985,
            "CLDFB analysis correlation {correlation}, scale {scale}"
        );
        assert!(
            (0.999..=1.001).contains(&normalized_rms_ratio),
            "CLDFB analysis normalized RMS ratio {normalized_rms_ratio}, scale {scale}"
        );
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn fdk_and_rust_low_power_cldfb_analysis_subbands_correlate() {
        let input = (0..512)
            .map(|index| {
                let value =
                    (index as f64 * 0.071).sin() * 0.25 + (index as f64 * 0.193).cos() * 0.125;
                (value * 2_147_483_648.0) as i32
            })
            .collect::<Vec<_>>();
        let mut fdk_real = vec![0i32; 16 * 32];
        let mut scale = 0;
        assert_eq!(
            unsafe {
                fdk_aac_sys::fdk_qmf_analysis32_cldfb_lp_test(
                    input.as_ptr(),
                    input.len() as i32,
                    fdk_real.as_mut_ptr(),
                    &mut scale,
                )
            },
            0
        );
        let normalized = input
            .iter()
            .map(|&sample| sample as f64 / 2_147_483_648.0)
            .collect::<Vec<_>>();
        let mut rust = LdSbrQmfAnalysis::new_cldfb_32();
        rust.set_low_power(true);
        let slots = rust.process_frame(&normalized).unwrap();
        let rust_values = slots
            .iter()
            .flat_map(|slot| slot.real.iter().copied())
            .collect::<Vec<_>>();
        let fdk_values = fdk_real
            .iter()
            .map(|&value| f64::from(value))
            .collect::<Vec<_>>();
        let dot = rust_values
            .iter()
            .zip(&fdk_values)
            .map(|(&left, &right)| left * right)
            .sum::<f64>();
        let rust_energy = rust_values.iter().map(|value| value * value).sum::<f64>();
        let fdk_energy = fdk_values.iter().map(|value| value * value).sum::<f64>();
        let correlation = dot / (rust_energy * fdk_energy).sqrt();
        let normalized_rms_ratio =
            (rust_energy / fdk_energy).sqrt() * 2_147_483_648.0 * 2.0f64.powi(-scale);
        assert!(
            correlation > 0.999,
            "LP CLDFB analysis correlation {correlation}"
        );
        assert!(
            (0.99..=1.01).contains(&normalized_rms_ratio),
            "LP CLDFB analysis normalized RMS ratio {normalized_rms_ratio}, scale {scale}"
        );
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn fdk_and_rust_qmf_synthesis_waveforms_correlate() {
        let slots = (0..16)
            .map(|slot| {
                let mut real = vec![0.0; 64];
                let mut imaginary = vec![0.0; 64];
                for band in 0..48 {
                    real[band] = ((slot * 17 + band * 7) as f64 * 0.031).sin() * 0.01;
                    imaginary[band] = ((slot * 11 + band * 5) as f64 * 0.043).cos() * 0.01;
                }
                QmfSlot { real, imaginary }
            })
            .collect::<Vec<_>>();
        let real = slots
            .iter()
            .flat_map(|slot| slot.real.iter())
            .map(|value| (value * 2_147_483_648.0) as i32)
            .collect::<Vec<_>>();
        let imaginary = slots
            .iter()
            .flat_map(|slot| slot.imaginary.iter())
            .map(|value| (value * 2_147_483_648.0) as i32)
            .collect::<Vec<_>>();
        let mut fdk = vec![0i32; 16 * 64];
        let result = unsafe {
            fdk_aac_sys::fdk_qmf_synthesis64_test(
                real.as_ptr(),
                imaginary.as_ptr(),
                16,
                fdk.as_mut_ptr(),
            )
        };
        assert_eq!(result, 0);
        let mut synthesis = LdSbrQmfSynthesis::new(64).unwrap();
        let rust = synthesis.process_frame(&slots).unwrap();
        let dot = rust
            .iter()
            .zip(&fdk)
            .map(|(&left, &right)| left * right as f64)
            .sum::<f64>();
        let rust_energy = rust.iter().map(|value| value * value).sum::<f64>();
        let fdk_energy = fdk.iter().map(|&value| (value as f64).powi(2)).sum::<f64>();
        let correlation = dot / (rust_energy * fdk_energy).sqrt();
        let normalized_rms_ratio = (rust_energy / fdk_energy).sqrt() * 2_147_483_648.0;
        assert!(
            correlation > 0.9999,
            "QMF synthesis correlation {correlation}"
        );
        assert!(
            (0.999..=1.001).contains(&normalized_rms_ratio),
            "QMF synthesis normalized RMS ratio {normalized_rms_ratio}"
        );
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn fdk_and_rust_low_power_qmf_synthesis_waveforms_correlate() {
        let slots = (0..16)
            .map(|slot| {
                let mut real = vec![0.0; 64];
                for band in 0..48 {
                    real[band] = ((slot * 17 + band * 7) as f64 * 0.031).sin() * 0.01;
                }
                QmfSlot {
                    real,
                    imaginary: Vec::new(),
                }
            })
            .collect::<Vec<_>>();
        let real = slots
            .iter()
            .flat_map(|slot| slot.real.iter())
            .map(|value| (value * 2_147_483_648.0) as i32)
            .collect::<Vec<_>>();
        let mut fdk = vec![0i32; 16 * 64];
        assert_eq!(
            unsafe {
                fdk_aac_sys::fdk_qmf_synthesis64_lp_test(real.as_ptr(), 16, fdk.as_mut_ptr())
            },
            0
        );
        let mut synthesis = LdSbrQmfSynthesis::new(64).unwrap();
        synthesis.set_low_power(true);
        let rust = synthesis.process_frame(&slots).unwrap();
        let dot = rust
            .iter()
            .zip(&fdk)
            .map(|(&left, &right)| left * f64::from(right))
            .sum::<f64>();
        let rust_energy = rust.iter().map(|value| value * value).sum::<f64>();
        let fdk_energy = fdk
            .iter()
            .map(|&value| f64::from(value).powi(2))
            .sum::<f64>();
        let correlation = dot / (rust_energy * fdk_energy).sqrt();
        let normalized_rms_ratio = (rust_energy / fdk_energy).sqrt() * 2_147_483_648.0;
        assert!(
            correlation > 0.999,
            "LP QMF synthesis correlation {correlation}"
        );
        assert!(
            (0.98..=1.02).contains(&normalized_rms_ratio),
            "LP QMF synthesis normalized RMS ratio {normalized_rms_ratio}"
        );
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn fdk_and_rust_cldfb_synthesis_waveforms_correlate() {
        let slots = (0..16)
            .map(|slot| {
                let mut real = vec![0.0; 32];
                let mut imaginary = vec![0.0; 32];
                for band in 0..29 {
                    real[band] = ((slot * 17 + band * 7) as f64 * 0.031).sin() * 0.01;
                    imaginary[band] = ((slot * 11 + band * 5) as f64 * 0.043).cos() * 0.01;
                }
                QmfSlot { real, imaginary }
            })
            .collect::<Vec<_>>();
        let real = slots
            .iter()
            .flat_map(|slot| slot.real.iter())
            .map(|value| (value * 2_147_483_648.0) as i32)
            .collect::<Vec<_>>();
        let imaginary = slots
            .iter()
            .flat_map(|slot| slot.imaginary.iter())
            .map(|value| (value * 2_147_483_648.0) as i32)
            .collect::<Vec<_>>();
        let mut fdk = vec![0i32; 16 * 32];
        assert_eq!(
            unsafe {
                fdk_aac_sys::fdk_qmf_synthesis32_cldfb_test(
                    real.as_ptr(),
                    imaginary.as_ptr(),
                    16,
                    fdk.as_mut_ptr(),
                )
            },
            0
        );
        let rust = LdSbrQmfSynthesis::new_cldfb_32()
            .process_frame(&slots)
            .unwrap();
        let dot = rust
            .iter()
            .zip(&fdk)
            .map(|(&left, &right)| left * right as f64)
            .sum::<f64>();
        let rust_energy = rust.iter().map(|value| value * value).sum::<f64>();
        let fdk_energy = fdk.iter().map(|&value| (value as f64).powi(2)).sum::<f64>();
        let correlation = dot / (rust_energy * fdk_energy).sqrt();
        let normalized_rms_ratio = (rust_energy / fdk_energy).sqrt() * 2_147_483_648.0;
        assert!(
            correlation > 0.999,
            "CLDFB synthesis correlation {correlation}"
        );
        assert!(
            (0.999..=1.001).contains(&normalized_rms_ratio),
            "CLDFB synthesis normalized RMS ratio {normalized_rms_ratio}"
        );
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn fdk_and_rust_low_power_cldfb_synthesis_waveforms_correlate() {
        let slots = (0..16)
            .map(|slot| {
                let mut real = vec![0.0; 32];
                for band in 0..29 {
                    real[band] = ((slot * 17 + band * 7) as f64 * 0.031).sin() * 0.01;
                }
                QmfSlot {
                    real,
                    imaginary: Vec::new(),
                }
            })
            .collect::<Vec<_>>();
        let real = slots
            .iter()
            .flat_map(|slot| slot.real.iter())
            .map(|value| (value * 2_147_483_648.0) as i32)
            .collect::<Vec<_>>();
        let mut fdk = vec![0i32; 16 * 32];
        assert_eq!(
            unsafe {
                fdk_aac_sys::fdk_qmf_synthesis32_cldfb_lp_test(real.as_ptr(), 16, fdk.as_mut_ptr())
            },
            0
        );
        let mut synthesis = LdSbrQmfSynthesis::new_cldfb_32();
        synthesis.set_low_power(true);
        let rust = synthesis.process_frame(&slots).unwrap();
        let dot = rust
            .iter()
            .zip(&fdk)
            .map(|(&left, &right)| left * f64::from(right))
            .sum::<f64>();
        let rust_energy = rust.iter().map(|value| value * value).sum::<f64>();
        let fdk_energy = fdk
            .iter()
            .map(|&value| f64::from(value).powi(2))
            .sum::<f64>();
        let correlation = dot / (rust_energy * fdk_energy).sqrt();
        let normalized_rms_ratio = (rust_energy / fdk_energy).sqrt() * 2_147_483_648.0;
        assert!(
            correlation > 0.999,
            "LP CLDFB synthesis correlation {correlation}"
        );
        assert!(
            (0.99..=1.01).contains(&normalized_rms_ratio),
            "LP CLDFB synthesis normalized RMS ratio {normalized_rms_ratio}"
        );
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn fdk_and_rust_cldfb64_synthesis_waveforms_correlate() {
        let slots = (0..16)
            .map(|slot| {
                let mut real = vec![0.0; 64];
                let mut imaginary = vec![0.0; 64];
                for band in 0..58 {
                    real[band] = ((slot * 17 + band * 7) as f64 * 0.031).sin() * 0.001;
                    imaginary[band] = ((slot * 11 + band * 5) as f64 * 0.043).cos() * 0.001;
                }
                QmfSlot { real, imaginary }
            })
            .collect::<Vec<_>>();
        let real = slots
            .iter()
            .flat_map(|slot| slot.real.iter())
            .map(|value| (value * 2_147_483_648.0) as i32)
            .collect::<Vec<_>>();
        let imaginary = slots
            .iter()
            .flat_map(|slot| slot.imaginary.iter())
            .map(|value| (value * 2_147_483_648.0) as i32)
            .collect::<Vec<_>>();
        let mut fdk = vec![0i32; 16 * 64];
        assert_eq!(
            unsafe {
                fdk_aac_sys::fdk_qmf_synthesis64_cldfb_test(
                    real.as_ptr(),
                    imaginary.as_ptr(),
                    16,
                    fdk.as_mut_ptr(),
                )
            },
            0
        );
        let rust = LdSbrQmfSynthesis::new_cldfb(64)
            .unwrap()
            .process_frame(&slots)
            .unwrap();
        let dot = rust
            .iter()
            .zip(&fdk)
            .map(|(&left, &right)| left * right as f64)
            .sum::<f64>();
        let rust_energy = rust.iter().map(|value| value * value).sum::<f64>();
        let fdk_energy = fdk.iter().map(|&value| (value as f64).powi(2)).sum::<f64>();
        let correlation = dot / (rust_energy * fdk_energy).sqrt();
        let normalized_rms_ratio = (rust_energy / fdk_energy).sqrt() * 2_147_483_648.0;
        assert!(
            correlation > 0.999,
            "CLDFB64 synthesis correlation {correlation}"
        );
        assert!(
            (0.999..=1.001).contains(&normalized_rms_ratio),
            "CLDFB64 synthesis normalized RMS ratio {normalized_rms_ratio}"
        );
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn fdk_and_rust_qmf_32_to_64_roundtrip_correlate() {
        let input = (0..512)
            .map(|index| {
                let value =
                    (index as f64 * 0.071).sin() * 0.25 + (index as f64 * 0.193).cos() * 0.125;
                (value * 2_147_483_648.0) as i32
            })
            .collect::<Vec<_>>();
        let mut fdk = vec![0i32; 1024];
        assert_eq!(
            unsafe {
                fdk_aac_sys::fdk_qmf_roundtrip32_64_test(
                    input.as_ptr(),
                    input.len() as i32,
                    fdk.as_mut_ptr(),
                )
            },
            0
        );
        let normalized = input
            .iter()
            .map(|&sample| sample as f64 / 2_147_483_648.0)
            .collect::<Vec<_>>();
        let mut analysis = LdSbrQmfAnalysis::new();
        let mut slots = analysis.process_frame(&normalized).unwrap();
        for slot in &mut slots {
            slot.real.resize(64, 0.0);
            slot.imaginary.resize(64, 0.0);
        }
        let mut synthesis = LdSbrQmfSynthesis::new(64).unwrap();
        let rust = synthesis.process_frame(&slots).unwrap();
        let dot = rust
            .iter()
            .zip(&fdk)
            .map(|(&left, &right)| left * right as f64)
            .sum::<f64>();
        let rust_energy = rust.iter().map(|value| value * value).sum::<f64>();
        let fdk_energy = fdk.iter().map(|&value| (value as f64).powi(2)).sum::<f64>();
        let correlation = dot / (rust_energy * fdk_energy).sqrt();
        let normalized_rms_ratio = (rust_energy / fdk_energy).sqrt() * 2_147_483_648.0;
        assert!(
            correlation > 0.985,
            "QMF roundtrip correlation {correlation}, ratio {normalized_rms_ratio}"
        );
        assert!((0.995..=1.005).contains(&normalized_rms_ratio));
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn fdk_and_rust_cldfb32_roundtrip_correlate() {
        let input = (0..512)
            .map(|index| {
                let value =
                    (index as f64 * 0.071).sin() * 0.1 + (index as f64 * 0.193).cos() * 0.05;
                (value * 2_147_483_648.0) as i32
            })
            .collect::<Vec<_>>();
        let mut fdk = vec![0i32; input.len()];
        assert_eq!(
            unsafe {
                fdk_aac_sys::fdk_qmf_roundtrip32_cldfb_test(
                    input.as_ptr(),
                    input.len() as i32,
                    fdk.as_mut_ptr(),
                )
            },
            0
        );
        let normalized = input
            .iter()
            .map(|&sample| sample as f64 / 2_147_483_648.0)
            .collect::<Vec<_>>();
        let slots = LdSbrQmfAnalysis::new_cldfb_32()
            .process_frame(&normalized)
            .unwrap();
        let rust = LdSbrQmfSynthesis::new_cldfb_32()
            .process_frame(&slots)
            .unwrap();
        let dot = rust
            .iter()
            .zip(&fdk)
            .map(|(&left, &right)| left * right as f64)
            .sum::<f64>();
        let rust_energy = rust.iter().map(|value| value * value).sum::<f64>();
        let fdk_energy = fdk.iter().map(|&value| (value as f64).powi(2)).sum::<f64>();
        let correlation = dot / (rust_energy * fdk_energy).sqrt();
        let normalized_rms_ratio = (rust_energy / fdk_energy).sqrt() * 2_147_483_648.0;
        assert!(
            correlation > 0.999,
            "CLDFB32 roundtrip correlation {correlation}"
        );
        assert!(
            (0.999..=1.001).contains(&normalized_rms_ratio),
            "CLDFB32 roundtrip normalized RMS ratio {normalized_rms_ratio}"
        );
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn fixed_cldfb_transform_stage_bridges_expose_fdk_scaling() {
        let mut fft_input = [0i32; 64];
        fft_input[0] = 1 << 20;
        let mut fft_output = [0i32; 64];
        let mut fft_scale = -1;
        assert_eq!(
            unsafe {
                fdk_aac_sys::fdk_fft32_test(
                    fft_input.as_ptr(),
                    fft_output.as_mut_ptr(),
                    &mut fft_scale,
                )
            },
            0
        );
        assert_eq!(fft_scale, 4);
        for bin in fft_output.chunks_exact(2) {
            assert_eq!(bin, &[1 << 16, 0]);
        }

        let mut fft_probe = [0i32; 64];
        for (index, value) in fft_probe.iter_mut().enumerate() {
            *value = (((index as i64 * 1_103_515_245 + 12_345) & 0x1f_ffff) as i32) - (1 << 20);
        }
        let mut c_fft_probe = [0i32; 64];
        unsafe { fdk_aac_sys::fdk_fft32_capture_enable(1) };
        assert_eq!(
            unsafe {
                fdk_aac_sys::fdk_fft32_test(
                    fft_probe.as_ptr(),
                    c_fft_probe.as_mut_ptr(),
                    &mut fft_scale,
                )
            },
            0
        );
        unsafe { fdk_aac_sys::fdk_fft32_capture_enable(0) };
        let mut c_fft_stages = [[0i32; 64]; 3];
        for (stage, output) in c_fft_stages.iter_mut().enumerate() {
            assert_eq!(
                unsafe { fdk_aac_sys::fdk_fft32_capture_get(stage as i32, output.as_mut_ptr()) },
                0
            );
        }
        assert_eq!(c_fft_stages[2], c_fft_probe);
        assert_ne!(c_fft_stages[0], c_fft_stages[1]);
        let mut rust_stage1 = fft_probe;
        fft32_radix4_stage1(&mut rust_stage1);
        assert_eq!(rust_stage1, c_fft_stages[0]);
        fft32_radix4_stage2(&mut rust_stage1);
        assert_eq!(rust_stage1, c_fft_stages[1]);
        fft32_radix4_stage3(&mut rust_stage1);
        assert_eq!(rust_stage1, c_fft_stages[2]);
        let mut rust_complete = fft_probe;
        fixed_fft32(&mut rust_complete);
        assert_eq!(rust_complete, c_fft_probe);
        fft32_radix2_probe(&mut fft_probe);
        let fft_max_error = fft_probe
            .iter()
            .zip(c_fft_probe)
            .map(|(&rust, c)| rust.abs_diff(c))
            .max()
            .unwrap();
        assert!(
            fft_max_error <= 10,
            "radix-2 FFT32 probe maximum error {fft_max_error}"
        );

        let input = (0..64)
            .map(|index| (((index as i64 * 1_103_515_245 + 12_345) & 0x1f_ffff) as i32) - (1 << 20))
            .collect::<Vec<_>>();
        let mut dct = [0i32; 64];
        let mut dst = [0i32; 64];
        let mut dct_scale = -1;
        let mut dst_scale = -1;
        assert_eq!(
            unsafe {
                fdk_aac_sys::fdk_dct_iv_test(input.as_ptr(), 64, dct.as_mut_ptr(), &mut dct_scale)
            },
            0
        );
        assert_eq!(
            unsafe {
                fdk_aac_sys::fdk_dst_iv_test(input.as_ptr(), 64, dst.as_mut_ptr(), &mut dst_scale)
            },
            0
        );
        assert_eq!((dct_scale, dst_scale), (6, 6));
        let mut rust_dct: [i32; 64] = input.clone().try_into().unwrap();
        assert_eq!(fixed_dct_iv_64(&mut rust_dct), dct_scale);
        assert_eq!(rust_dct, dct);
        let mut rust_dst: [i32; 64] = input.clone().try_into().unwrap();
        assert_eq!(fixed_dst_iv_64(&mut rust_dst), dst_scale);
        assert_eq!(rust_dst, dst);
        assert!(dct.iter().any(|&value| value != 0));
        assert!(dst.iter().any(|&value| value != 0));
        assert_ne!(dct, dst);
    }
}
