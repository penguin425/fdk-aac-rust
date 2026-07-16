//! MPEG Surround 2-1-2 frame parsing and parameter-state handling for USAC.

use std::sync::LazyLock;

use crate::bits::{BitError, BitReader, BitWriter};
use crate::ld_sbr_qmf::{LdSbrQmfSynthesis, QmfError, QmfSlot};
use crate::ps::{hybrid_synthesis, PsHybridAnalysis};

const HUFFMAN_SOURCE: &str = include_str!(concat!(
    env!("FDK_AAC_UPSTREAM_DIR"),
    "/libFDK/src/huff_nodes.cpp"
));
const DECORRELATOR_SOURCE: &str = include_str!(concat!(
    env!("FDK_AAC_UPSTREAM_DIR"),
    "/libFDK/src/FDK_decorrelate.cpp"
));
const SAC_ROM_SOURCE: &str = include_str!(concat!(
    env!("FDK_AAC_UPSTREAM_DIR"),
    "/libSACdec/src/sac_rom.cpp"
));

const STRIDES: [usize; 4] = [1, 2, 5, 28];
const CLD_DB: [f32; 31] = [
    -150.0, -45.0, -40.0, -35.0, -30.0, -25.0, -22.0, -19.0, -16.0, -13.0, -10.0, -8.0, -6.0, -4.0,
    -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 13.0, 16.0, 19.0, 22.0, 25.0, 30.0, 35.0, 40.0, 45.0,
    150.0,
];
const ICC: [f32; 8] = [1.0, 0.937, 0.84118, 0.60092, 0.36764, 0.0, -0.589, -0.99];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MpsParameterKind {
    Cld,
    Icc,
    Ipd,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MpsError {
    Bit(BitError),
    InvalidParameterSets,
    InvalidParameterSlot,
    InvalidDataMode,
    InvalidHuffmanCodeword,
    UnsupportedHuffmanCoding,
    Qmf(QmfError),
}

impl From<BitError> for MpsError {
    fn from(value: BitError) -> Self {
        Self::Bit(value)
    }
}

impl From<QmfError> for MpsError {
    fn from(value: QmfError) -> Self {
        Self::Qmf(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MpsParameterSet {
    pub slot: usize,
    pub cld: Vec<i8>,
    pub icc: Vec<i8>,
    pub ipd: Option<Vec<i8>>,
    pub smoothing: MpsSmoothing,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MpsSmoothing {
    pub mode: u8,
    pub time: Option<u8>,
    pub stride_index: Option<u8>,
    pub bands: Vec<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mps212Frame {
    pub independent: bool,
    pub parameter_sets: Vec<MpsParameterSet>,
    pub transient_shaping: Option<MpsTransientShaping>,
    pub stp_enabled: Vec<bool>,
    pub ges_envelopes: Vec<Option<Vec<u8>>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MpsTransientShaping {
    pub enabled: bool,
    /// `None` is a non-transient slot; transient slots contain the 3-bit phase.
    pub phases: Vec<Option<u8>>,
}

/// Apply one OTT parameter band to a mono/direct and decorrelated complex QMF value.
///
/// The matrix preserves total power for uncorrelated inputs, realizes CLD as the
/// left/right power ratio, ICC as the inter-channel correlation, and rotates the
/// right output by the transmitted IPD step (multiples of pi/8).
pub fn spatial_upmix_band(
    direct: (f32, f32),
    decorrelated: (f32, f32),
    cld_index: i8,
    icc_index: i8,
    ipd_index: Option<i8>,
) -> ((f32, f32), (f32, f32)) {
    let cld = CLD_DB[(i16::from(cld_index).clamp(-15, 15) + 15) as usize];
    let ratio = 10.0f32.powf(cld / 10.0);
    let left_level = (ratio / (1.0 + ratio)).sqrt();
    let right_level = (1.0 / (1.0 + ratio)).sqrt();
    let correlation = ICC[usize::from(icc_index.clamp(0, 7) as u8)];
    let direct_gain = ((1.0 + correlation) * 0.5).sqrt();
    let diffuse_gain = ((1.0 - correlation) * 0.5).sqrt();
    let common = (
        direct_gain * direct.0 + diffuse_gain * decorrelated.0,
        direct_gain * direct.1 + diffuse_gain * decorrelated.1,
    );
    let opposite = (
        direct_gain * direct.0 - diffuse_gain * decorrelated.0,
        direct_gain * direct.1 - diffuse_gain * decorrelated.1,
    );
    let left = (left_level * common.0, left_level * common.1);
    let phase = f32::from(ipd_index.unwrap_or(0) & 15) * std::f32::consts::PI / 8.0;
    let (sin, cos) = phase.sin_cos();
    let right = (
        right_level * (opposite.0 * cos - opposite.1 * sin),
        right_level * (opposite.0 * sin + opposite.1 * cos),
    );
    (left, right)
}

pub fn spatial_prediction_upmix_band(
    direct: (f32, f32),
    residual_or_decorrelated: (f32, f32),
    cld_index: i8,
    icc_index: i8,
    ipd_index: Option<i8>,
    residual_band: bool,
) -> ((f32, f32), (f32, f32)) {
    let icc_index = usize::from(icc_index.clamp(0, 7) as u8);
    let ipd = ipd_index.unwrap_or(0) & 15;
    if cld_index == 0 && icc_index == 0 && ipd == 8 {
        let gain = 0.5 / 1.2;
        let left = complex_add(
            complex_scale(direct, gain),
            complex_scale(
                residual_or_decorrelated,
                if residual_band { gain } else { 0.0 },
            ),
        );
        let right_second = if residual_band { gain } else { 0.0 };
        let right = complex_add(
            complex_scale(direct, if residual_band { gain } else { -gain }),
            complex_scale(residual_or_decorrelated, -right_second),
        );
        return (left, right);
    }
    let cld = CLD_DB[(i16::from(cld_index).clamp(-15, 15) + 15) as usize];
    let ratio = 10.0f32.powf(cld / 10.0);
    let iid = ratio.sqrt();
    let rho = ICC[icc_index];
    let phase = f32::from(ipd) * std::f32::consts::PI / 8.0;
    let (sin, cos) = phase.sin_cos();
    let temp = (ratio + 1.0 + 2.0 * iid * rho * cos).max(1e-12);
    let inverse_weight = (temp / (ratio + 1.0)).sqrt();
    let weight = 0.5 * inverse_weight.max(1.0 / 1.2);
    let alpha_re = (1.0 - ratio) / temp;
    let alpha_im = -2.0 * iid * rho * sin / temp;
    let h11 = (weight * (1.0 - alpha_re), -weight * alpha_im);
    let h21 = (weight * (1.0 + alpha_re), weight * alpha_im);
    let second = if residual_band {
        weight
    } else {
        2.0 * iid * (1.0 - rho * rho).max(0.0).sqrt() * weight / temp
    };
    (
        complex_add(
            complex_multiply(h11, direct),
            complex_scale(residual_or_decorrelated, second),
        ),
        complex_add(
            complex_multiply(h21, direct),
            complex_scale(residual_or_decorrelated, -second),
        ),
    )
}

fn complex_scale(value: (f32, f32), gain: f32) -> (f32, f32) {
    (value.0 * gain, value.1 * gain)
}

fn complex_add(left: (f32, f32), right: (f32, f32)) -> (f32, f32) {
    (left.0 + right.0, left.1 + right.1)
}

fn complex_multiply(left: (f32, f32), right: (f32, f32)) -> (f32, f32) {
    (
        left.0 * right.0 - left.1 * right.1,
        left.0 * right.1 + left.1 * right.0,
    )
}

fn decorrelator_coefficients(name: &str, count: usize) -> Vec<f32> {
    let start = DECORRELATOR_SOURCE
        .find(name)
        .expect("USAC decorrelator ROM");
    let source = &DECORRELATOR_SOURCE[start..];
    let values: Vec<_> = source
        .split("DECORR(0x")
        .skip(1)
        .take(count)
        .map(|entry| {
            let raw = u32::from_str_radix(&entry[..8], 16).unwrap() as i32;
            raw as f32 / 2_147_483_648.0
        })
        .collect();
    assert_eq!(values.len(), count);
    values
}

static DECORR_COEFFICIENTS: LazyLock<[Vec<f32>; 4]> = LazyLock::new(|| {
    [
        decorrelator_coefficients("DecorrNumeratorReal0_USAC", 11),
        decorrelator_coefficients("DecorrNumeratorReal1_USAC", 9),
        decorrelator_coefficients("DecorrNumeratorReal2_USAC", 4),
        decorrelator_coefficients("DecorrNumeratorReal3_USAC", 3),
    ]
});

#[derive(Debug, Clone)]
pub struct MpsUsacDecorrelator {
    config: usize,
    states: Vec<Vec<(f32, f32)>>,
    delays: Vec<Vec<(f32, f32)>>,
    delay_indices: [usize; 4],
}

impl MpsUsacDecorrelator {
    pub fn new(config: u8) -> Result<Self, MpsError> {
        if config > 2 {
            return Err(MpsError::InvalidDataMode);
        }
        let mut states = Vec::with_capacity(71);
        let mut delays = Vec::with_capacity(71);
        for band in 0..71 {
            let reverb = reverb_band(config as usize, band);
            states.push(vec![(0.0, 0.0); [10, 8, 3, 2][reverb]]);
            delays.push(vec![(0.0, 0.0); [11, 10, 5, 2][reverb]]);
        }
        Ok(Self {
            config: config as usize,
            states,
            delays,
            delay_indices: [0; 4],
        })
    }

    pub fn process_slot(&mut self, input: &[(f32, f32)]) -> Result<Vec<(f32, f32)>, MpsError> {
        if input.len() != 71 {
            return Err(MpsError::InvalidParameterSlot);
        }
        let mut output = vec![(0.0, 0.0); 71];
        for band in 0..71 {
            let reverb = reverb_band(self.config, band);
            let delay_index = self.delay_indices[reverb];
            let delayed = self.delays[band][delay_index];
            self.delays[band][delay_index] = input[band];
            output[band] = allpass_real(
                delayed,
                &DECORR_COEFFICIENTS[reverb],
                &mut self.states[band],
            );
        }
        for reverb in 0..4 {
            let delay = [11, 10, 5, 2][reverb];
            self.delay_indices[reverb] = (self.delay_indices[reverb] + 1) % delay;
        }
        Ok(output)
    }
}

fn parameter_band_map(bands: usize) -> Result<Vec<usize>, MpsError> {
    if bands == 15 {
        // Low-delay MPEG Surround uses the libSACenc 64-QMF-band mapping.
        // The renderer operates on the 71-band 8+2+61 hybrid layout, so map
        // every hybrid band back to its originating QMF band first.
        const LD_15: [usize; 64] = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13,
            13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
            14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
        ];
        return Ok((0..71)
            .map(|hybrid_band| {
                let qmf_band = match hybrid_band {
                    0..=7 => 0,
                    8..=9 => 1,
                    _ => hybrid_band - 8,
                };
                LD_15[qmf_band]
            })
            .collect());
    }
    if !matches!(bands, 4 | 5 | 7 | 10 | 14 | 20 | 28) {
        return Err(MpsError::InvalidParameterSets);
    }
    let name = format!("kernels_{bands}_to_71");
    let start = SAC_ROM_SOURCE.find(&name).unwrap();
    let source = &SAC_ROM_SOURCE[start..];
    let body = &source[source.find('{').unwrap() + 1..source.find("};").unwrap()];
    let mut values: Vec<_> = body
        .split(',')
        .filter_map(|value| value.trim().parse::<usize>().ok())
        .collect();
    assert!(
        values.len() >= 71,
        "embedded SAC parameter-band map must contain 71 entries"
    );
    values.truncate(71);
    Ok(values)
}

#[derive(Debug, Clone)]
pub struct Mps212HybridRenderer {
    decorrelator: MpsUsacDecorrelator,
    band_map: Vec<usize>,
    previous_cld: Vec<i8>,
    previous_icc: Vec<i8>,
    previous_ipd: Vec<i8>,
}

impl Mps212HybridRenderer {
    pub fn new(parameter_bands: usize, decorrelation_config: u8) -> Result<Self, MpsError> {
        Ok(Self {
            decorrelator: MpsUsacDecorrelator::new(decorrelation_config)?,
            band_map: parameter_band_map(parameter_bands)?,
            previous_cld: vec![0; parameter_bands],
            previous_icc: vec![0; parameter_bands],
            previous_ipd: vec![0; parameter_bands],
        })
    }

    pub fn process_frame(
        &mut self,
        direct: &[Vec<(f32, f32)>],
        frame: &Mps212Frame,
    ) -> Result<(Vec<Vec<(f32, f32)>>, Vec<Vec<(f32, f32)>>), MpsError> {
        self.process_frame_internal(direct, None, 0, frame)
    }

    pub fn process_frame_with_residual(
        &mut self,
        direct: &[Vec<(f32, f32)>],
        residual: &[Vec<(f32, f32)>],
        residual_bands: usize,
        frame: &Mps212Frame,
    ) -> Result<(Vec<Vec<(f32, f32)>>, Vec<Vec<(f32, f32)>>), MpsError> {
        self.process_frame_internal(direct, Some(residual), residual_bands, frame)
    }

    fn process_frame_internal(
        &mut self,
        direct: &[Vec<(f32, f32)>],
        residual: Option<&[Vec<(f32, f32)>]>,
        residual_bands: usize,
        frame: &Mps212Frame,
    ) -> Result<(Vec<Vec<(f32, f32)>>, Vec<Vec<(f32, f32)>>), MpsError> {
        if direct.iter().any(|slot| slot.len() != 71) || frame.parameter_sets.is_empty() {
            return Err(MpsError::InvalidParameterSlot);
        }
        if residual.is_some_and(|slots| {
            slots.len() != direct.len() || slots.iter().any(|slot| slot.len() != 71)
        }) {
            return Err(MpsError::InvalidParameterSlot);
        }
        let mut left = Vec::with_capacity(direct.len());
        let mut right = Vec::with_capacity(direct.len());
        let mut start_slot = 0usize;
        for parameters in &frame.parameter_sets {
            if parameters.cld.len() != self.previous_cld.len()
                || parameters.icc.len() != self.previous_icc.len()
            {
                return Err(MpsError::InvalidParameterSets);
            }
            let end_slot = parameters.slot + 1;
            if end_slot > direct.len() || end_slot <= start_slot {
                return Err(MpsError::InvalidParameterSlot);
            }
            for slot in start_slot..end_slot {
                let fraction = (slot - start_slot + 1) as f32 / (end_slot - start_slot) as f32;
                let decorrelated = self.decorrelator.process_slot(&direct[slot])?;
                let mut left_slot = vec![(0.0, 0.0); 71];
                let mut right_slot = vec![(0.0, 0.0); 71];
                for hybrid_band in 0..71 {
                    let band = self.band_map[hybrid_band];
                    let interpolate = |previous: i8, target: i8| {
                        (f32::from(previous) + fraction * f32::from(target - previous)).round()
                            as i8
                    };
                    let cld = interpolate(self.previous_cld[band], parameters.cld[band]);
                    let icc = interpolate(self.previous_icc[band], parameters.icc[band]);
                    let ipd = parameters.ipd.as_ref().and_then(|values| {
                        values
                            .get(band)
                            .map(|&value| interpolate(self.previous_ipd[band], value) & 15)
                    });
                    let residual_band = residual.is_some() && band < residual_bands;
                    let second = if residual_band {
                        residual.unwrap()[slot][hybrid_band]
                    } else {
                        decorrelated[hybrid_band]
                    };
                    (left_slot[hybrid_band], right_slot[hybrid_band]) = if residual.is_some() {
                        spatial_prediction_upmix_band(
                            direct[slot][hybrid_band],
                            second,
                            cld,
                            icc,
                            ipd,
                            residual_band,
                        )
                    } else {
                        spatial_upmix_band(direct[slot][hybrid_band], second, cld, icc, ipd)
                    };
                }
                left.push(left_slot);
                right.push(right_slot);
            }
            self.previous_cld.clone_from(&parameters.cld);
            self.previous_icc.clone_from(&parameters.icc);
            if let Some(ipd) = &parameters.ipd {
                if ipd.len() > self.previous_ipd.len() {
                    return Err(MpsError::InvalidParameterSets);
                }
                self.previous_ipd.fill(0);
                self.previous_ipd[..ipd.len()].copy_from_slice(ipd);
            }
            start_slot = end_slot;
        }
        if start_slot != direct.len() {
            return Err(MpsError::InvalidParameterSlot);
        }
        Ok((left, right))
    }
}

#[derive(Debug, Clone)]
pub struct Mps212QmfProcessor {
    qmf_bands: usize,
    hybrid: PsHybridAnalysis,
    residual_hybrid: PsHybridAnalysis,
    renderer: Mps212HybridRenderer,
    left_synthesis: LdSbrQmfSynthesis,
    right_synthesis: LdSbrQmfSynthesis,
}

impl Mps212QmfProcessor {
    pub fn new(parameter_bands: usize, decorrelation_config: u8) -> Result<Self, MpsError> {
        Self::new_with_qmf_bands(parameter_bands, decorrelation_config, 64)
    }

    pub fn new_with_qmf_bands(
        parameter_bands: usize,
        decorrelation_config: u8,
        qmf_bands: usize,
    ) -> Result<Self, MpsError> {
        if !matches!(qmf_bands, 32 | 64) {
            return Err(MpsError::Qmf(QmfError::InvalidSubbandCount {
                expected: 64,
                actual: qmf_bands,
            }));
        }
        Ok(Self {
            qmf_bands,
            hybrid: PsHybridAnalysis::new(),
            residual_hybrid: PsHybridAnalysis::new(),
            renderer: Mps212HybridRenderer::new(parameter_bands, decorrelation_config)?,
            left_synthesis: LdSbrQmfSynthesis::new(qmf_bands)?,
            right_synthesis: LdSbrQmfSynthesis::new(qmf_bands)?,
        })
    }

    pub fn process_qmf(
        &mut self,
        mono: &[QmfSlot],
        frame: &Mps212Frame,
    ) -> Result<(Vec<f64>, Vec<f64>), MpsError> {
        for slot in mono {
            if slot.real.len() < self.qmf_bands || slot.imaginary.len() < self.qmf_bands {
                return Err(MpsError::Qmf(QmfError::InvalidSubbandCount {
                    expected: self.qmf_bands,
                    actual: slot.real.len().min(slot.imaginary.len()),
                }));
            }
        }
        let qmf_bands = self.qmf_bands;
        let hybrid: Vec<Vec<(f32, f32)>> = mono
            .iter()
            .map(|slot| {
                let padded;
                let slot = if qmf_bands == 64 {
                    slot
                } else {
                    padded = QmfSlot {
                        real: slot.real[..qmf_bands]
                            .iter()
                            .copied()
                            .chain(std::iter::repeat(0.0))
                            .take(64)
                            .collect(),
                        imaginary: slot.imaginary[..qmf_bands]
                            .iter()
                            .copied()
                            .chain(std::iter::repeat(0.0))
                            .take(64)
                            .collect(),
                    };
                    &padded
                };
                self.hybrid
                    .process(slot)
                    .into_iter()
                    .map(|(real, imaginary)| (real as f32, imaginary as f32))
                    .collect()
            })
            .collect();
        let (left, right) = self.renderer.process_frame(&hybrid, frame)?;
        let synthesize = |slots: Vec<Vec<(f32, f32)>>| {
            slots
                .into_iter()
                .map(|slot| {
                    let slot: Vec<_> = slot
                        .into_iter()
                        .map(|(real, imaginary)| (f64::from(real), f64::from(imaginary)))
                        .collect();
                    let mut qmf = hybrid_synthesis(&slot);
                    qmf.real.truncate(qmf_bands);
                    qmf.imaginary.truncate(qmf_bands);
                    qmf
                })
                .collect::<Vec<_>>()
        };
        Ok((
            self.left_synthesis.process_frame(&synthesize(left))?,
            self.right_synthesis.process_frame(&synthesize(right))?,
        ))
    }

    pub fn process_qmf_with_residual(
        &mut self,
        downmix: &[QmfSlot],
        residual: &[QmfSlot],
        residual_bands: usize,
        frame: &Mps212Frame,
    ) -> Result<(Vec<f64>, Vec<f64>), MpsError> {
        if downmix.len() != residual.len() {
            return Err(MpsError::InvalidParameterSlot);
        }
        for slot in downmix.iter().chain(residual) {
            if slot.real.len() < self.qmf_bands || slot.imaginary.len() < self.qmf_bands {
                return Err(MpsError::Qmf(QmfError::InvalidSubbandCount {
                    expected: self.qmf_bands,
                    actual: slot.real.len().min(slot.imaginary.len()),
                }));
            }
        }
        let qmf_bands = self.qmf_bands;
        let padded_slot = |slot: &QmfSlot| QmfSlot {
            real: slot.real[..qmf_bands]
                .iter()
                .copied()
                .chain(std::iter::repeat(0.0))
                .take(64)
                .collect(),
            imaginary: slot.imaginary[..qmf_bands]
                .iter()
                .copied()
                .chain(std::iter::repeat(0.0))
                .take(64)
                .collect(),
        };
        let convert = |values: Vec<(f64, f64)>| {
            values
                .into_iter()
                .map(|(real, imaginary)| (real as f32, imaginary as f32))
                .collect::<Vec<_>>()
        };
        let direct: Vec<_> = downmix
            .iter()
            .map(|slot| {
                let padded = padded_slot(slot);
                convert(self.hybrid.process(&padded))
            })
            .collect();
        let residual: Vec<_> = residual
            .iter()
            .map(|slot| {
                let padded = padded_slot(slot);
                convert(self.residual_hybrid.process(&padded))
            })
            .collect();
        let (left, right) =
            self.renderer
                .process_frame_with_residual(&direct, &residual, residual_bands, frame)?;
        let synthesize = |slots: Vec<Vec<(f32, f32)>>| {
            slots
                .into_iter()
                .map(|slot| {
                    let slot: Vec<_> = slot
                        .into_iter()
                        .map(|(real, imaginary)| (f64::from(real), f64::from(imaginary)))
                        .collect();
                    let mut qmf = hybrid_synthesis(&slot);
                    qmf.real.truncate(qmf_bands);
                    qmf.imaginary.truncate(qmf_bands);
                    qmf
                })
                .collect::<Vec<_>>()
        };
        Ok((
            self.left_synthesis.process_frame(&synthesize(left))?,
            self.right_synthesis.process_frame(&synthesize(right))?,
        ))
    }
}

fn reverb_band(config: usize, hybrid_band: usize) -> usize {
    const OFFSETS: [[usize; 4]; 3] = [[8, 21, 30, 71], [8, 56, 71, 71], [0, 21, 71, 71]];
    OFFSETS[config]
        .iter()
        .position(|&end| hybrid_band < end)
        .unwrap_or(3)
}

fn allpass_real(input: (f32, f32), coefficients: &[f32], state: &mut [(f32, f32)]) -> (f32, f32) {
    let order = state.len();
    debug_assert_eq!(coefficients.len(), order + 1);
    let output = (
        4.0 * (state[0].0 + input.0 * coefficients[0] * 0.5),
        4.0 * (state[0].1 + input.1 * coefficients[0] * 0.5),
    );
    for index in 0..order - 1 {
        state[index] = (
            state[index + 1].0 + input.0 * coefficients[index + 1] * 0.5
                - output.0 * coefficients[order - index] * 0.5,
            state[index + 1].1 + input.1 * coefficients[index + 1] * 0.5
                - output.1 * coefficients[order - index] * 0.5,
        );
    }
    state[order - 1] = (
        input.0 * coefficients[order] * 0.5 - output.0 * coefficients[0] * 0.5,
        input.1 * coefficients[order] * 0.5 - output.1 * coefficients[0] * 0.5,
    );
    output
}

#[derive(Debug, Clone)]
struct ParameterHistory {
    values: Vec<i8>,
    coarse: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParameterDataMode {
    Default,
    Keep,
    Interpolate,
    New,
}

impl ParameterHistory {
    fn new(bands: usize, default: i8) -> Self {
        Self {
            values: vec![default; bands],
            coarse: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Mps212FrameDecoder {
    time_slots: usize,
    high_rate: bool,
    phase_coding: bool,
    temporal_shape_config: u8,
    num_parameter_set_bits: usize,
    cld: ParameterHistory,
    icc: ParameterHistory,
    ipd: ParameterHistory,
}

/// Stateful low-delay 2-1-2 frame writer using FDK's minimum-bit selection
/// between grouped PCM and one-/two-dimensional differential Huffman coding.
#[derive(Debug, Clone)]
pub struct Mps212FrameEncoder {
    bands: usize,
    time_slots: usize,
    previous_cld: Vec<i8>,
    previous_icc: Vec<i8>,
    num_parameter_set_bits: usize,
    byte_aligned: bool,
}

impl Mps212FrameEncoder {
    pub fn new(time_slots: usize, bands: usize) -> Result<Self, MpsError> {
        if time_slots == 0 || time_slots > 64 || bands == 0 {
            return Err(MpsError::InvalidParameterSlot);
        }
        Ok(Self {
            bands,
            time_slots,
            previous_cld: vec![0; bands],
            previous_icc: vec![0; bands],
            num_parameter_set_bits: 3,
            byte_aligned: false,
        })
    }

    pub fn with_low_delay_framing(mut self) -> Self {
        self.num_parameter_set_bits = 1;
        // fdk_sacenc_writeSpatialFrame aligns every LD spatial frame before
        // reporting nOutputBits to the AAC extension writer.
        self.byte_aligned = true;
        self
    }

    pub fn encode(
        &mut self,
        cld: &[i8],
        icc: &[i8],
        independent: bool,
    ) -> Result<(Vec<u8>, usize), MpsError> {
        if cld.len() != self.bands
            || icc.len() != self.bands
            || cld.iter().any(|&value| !(-15..=15).contains(&value))
            || icc.iter().any(|&value| !(0..=7).contains(&value))
        {
            return Err(MpsError::InvalidDataMode);
        }
        let mut writer = BitWriter::new();
        writer.write_bool(false); // bsFramingType: one full-frame set
        writer.write(0, self.num_parameter_set_bits); // bsNumParamSets = 0 + 1
        writer.write_bool(independent);
        write_parameter_set(
            &mut writer,
            cld,
            &self.previous_cld,
            independent,
            MpsParameterKind::Cld,
            31,
            15,
        );
        write_parameter_set(
            &mut writer,
            icc,
            &self.previous_icc,
            independent,
            MpsParameterKind::Icc,
            8,
            0,
        );
        writer.write(0, 2); // bsSmoothMode[0]
        if self.byte_aligned {
            writer.byte_align();
        }
        let bits = writer.bits_written();
        self.previous_cld.copy_from_slice(cld);
        self.previous_icc.copy_from_slice(icc);
        Ok((writer.finish(), bits))
    }

    pub fn time_slots(&self) -> usize {
        self.time_slots
    }
}

fn write_parameter_set(
    writer: &mut BitWriter,
    values: &[i8],
    previous: &[i8],
    independent: bool,
    kind: MpsParameterKind,
    levels: u32,
    offset: i8,
) {
    if !independent && values == previous {
        writer.write(1, 2); // KEEP
        return;
    }
    writer.write(3, 2); // FINECOARSE / NEW
    writer.write_bool(false); // bsDataPair
    writer.write_bool(false); // fine quantization
    writer.write(0, 2); // frequency stride 1
    let pcm = encoded_bits(|candidate| {
        candidate.write_bool(true); // grouped PCM
        pcm_encode(candidate, values, levels, offset);
    });
    let frequency = huffman_vector_candidate(values, kind, false, offset);
    let time = (!independent).then(|| {
        let differences = values
            .iter()
            .zip(previous)
            .map(|(&current, &old)| current - old)
            .collect::<Vec<_>>();
        huffman_vector_candidate(&differences, kind, true, 0)
    });
    let huffman = match time {
        Some(time) if time.bits < frequency.bits => encoded_bits(|candidate| {
            candidate.write_bool(false); // differential Huffman
            candidate.write_bool(true); // time difference
            append_encoded(candidate, &time);
        }),
        _ if !independent => encoded_bits(|candidate| {
            candidate.write_bool(false); // differential Huffman
            candidate.write_bool(false); // frequency difference
            append_encoded(candidate, &frequency);
        }),
        _ => encoded_bits(|candidate| {
            candidate.write_bool(false); // differential Huffman
            append_encoded(candidate, &frequency);
        }),
    };
    // FDK deliberately selects grouped PCM on an equal bit count.
    append_encoded(
        writer,
        if pcm.bits <= huffman.bits {
            &pcm
        } else {
            &huffman
        },
    );
}

#[derive(Debug, Clone)]
struct EncodedBits {
    bytes: Vec<u8>,
    bits: usize,
}

fn encoded_bits(write: impl FnOnce(&mut BitWriter)) -> EncodedBits {
    let mut writer = BitWriter::new();
    write(&mut writer);
    let bits = writer.bits_written();
    EncodedBits {
        bytes: writer.finish(),
        bits,
    }
}

fn append_encoded(writer: &mut BitWriter, encoded: &EncodedBits) {
    for bit in 0..encoded.bits {
        writer.write(u32::from((encoded.bytes[bit / 8] >> (7 - bit % 8)) & 1), 1);
    }
}

fn huffman_code(table: &[[i16; 2]], target: i16) -> Option<Vec<bool>> {
    fn visit(table: &[[i16; 2]], node: i16, target: i16, path: &mut Vec<bool>) -> bool {
        for bit in [false, true] {
            path.push(bit);
            let child = table[node as usize][usize::from(bit)];
            if child == target
                || child > 0 && (child as usize) < table.len() && visit(table, child, target, path)
            {
                return true;
            }
            path.pop();
        }
        false
    }
    let mut path = Vec::new();
    visit(table, 0, target, &mut path).then_some(path)
}

fn write_huffman_code(writer: &mut BitWriter, table: &[[i16; 2]], target: i16) -> bool {
    let Some(code) = huffman_code(table, target) else {
        return false;
    };
    for bit in code {
        writer.write_bool(bit);
    }
    true
}

fn huffman_vector_candidate(
    values: &[i8],
    kind: MpsParameterKind,
    time_delta: bool,
    offset: i8,
) -> EncodedBits {
    let data = if time_delta {
        values.to_vec()
    } else {
        let absolute = values
            .iter()
            .map(|&value| value + offset)
            .collect::<Vec<_>>();
        let mut differences = Vec::with_capacity(absolute.len());
        differences.push(absolute[0]);
        differences.extend(absolute.windows(2).map(|pair| pair[1] - pair[0]));
        differences
    };
    let one = huffman_1d_candidate(&data, kind, time_delta);
    let two = huffman_2d_frequency_candidate(&data, kind, time_delta);
    match (one, two) {
        (Some(one), Some(two)) if two.bits < one.bits => two,
        (Some(one), _) => one,
        (None, Some(two)) => two,
        (None, None) => EncodedBits {
            bytes: Vec::new(),
            bits: usize::MAX / 2,
        },
    }
}

fn huffman_1d_candidate(data: &[i8], kind: MpsParameterKind, time: bool) -> Option<EncodedBits> {
    let mut writer = BitWriter::new();
    writer.write_bool(false); // HUFF_1D
    let mut start = 0;
    if !time {
        let part0 = match kind {
            MpsParameterKind::Cld => &CLD_PART0[..],
            MpsParameterKind::Icc => &ICC_PART0[..],
            MpsParameterKind::Ipd => &IPD_PART0[..],
        };
        if !write_huffman_code(&mut writer, part0, -(i16::from(data[0]) + 1)) {
            return None;
        }
        start = 1;
    }
    let table = match (kind, time) {
        (MpsParameterKind::Cld, false) => &CLD_1D_FREQ[..],
        (MpsParameterKind::Cld, true) => &CLD_1D_TIME[..],
        (MpsParameterKind::Icc, _) => &ICC_1D[..],
        (MpsParameterKind::Ipd, false) => &IPD_1D_FREQ[..],
        (MpsParameterKind::Ipd, true) => &IPD_1D_TIME[..],
    };
    for &value in &data[start..] {
        let magnitude = i16::from(value).abs();
        if !write_huffman_code(&mut writer, table, -(magnitude + 1)) {
            return None;
        }
        if value != 0 {
            writer.write_bool(value < 0);
        }
    }
    let bits = writer.bits_written();
    Some(EncodedBits {
        bytes: writer.finish(),
        bits,
    })
}

fn huffman_2d_frequency_candidate(
    data: &[i8],
    kind: MpsParameterKind,
    time: bool,
) -> Option<EncodedBits> {
    let start = usize::from(!time);
    let paired = (data.len() - start) / 2 * 2;
    if paired == 0 {
        return None;
    }
    let max = data[start..start + paired]
        .iter()
        .map(|value| value.unsigned_abs() as usize)
        .max()
        .unwrap_or(0);
    let lav_code = match kind {
        MpsParameterKind::Cld => [3, 5, 7, 9].iter().position(|&lav| max <= lav),
        MpsParameterKind::Icc => [1, 3, 5, 7].iter().position(|&lav| max <= lav),
        MpsParameterKind::Ipd => None,
    }?;
    let lav = match kind {
        MpsParameterKind::Cld => 2 * lav_code + 3,
        MpsParameterKind::Icc => 2 * lav_code + 1,
        MpsParameterKind::Ipd => unreachable!(),
    };
    let table = table_2d(kind, time, false, lav);
    let remainder = (start + paired < data.len()).then(|| data[start + paired]);
    let remainder_table = match (kind, time) {
        (MpsParameterKind::Cld, false) => &CLD_1D_FREQ[..],
        (MpsParameterKind::Cld, true) => &CLD_1D_TIME[..],
        (MpsParameterKind::Icc, _) => &ICC_1D[..],
        _ => return None,
    };
    let remainder_code = match remainder {
        Some(value) => Some(huffman_code(
            remainder_table,
            -(i16::from(value).abs() + 1),
        )?),
        None => None,
    };
    Some(encoded_bits(|writer| {
        writer.write_bool(true); // HUFF_2D / frequency pairs
        assert!(write_huffman_code(
            writer,
            &LAV_TABLE,
            -(lav_code as i16 + 1)
        ));
        if !time {
            let part0 = match kind {
                MpsParameterKind::Cld => &CLD_PART0[..],
                MpsParameterKind::Icc => &ICC_PART0[..],
                MpsParameterKind::Ipd => &IPD_PART0[..],
            };
            assert!(write_huffman_code(writer, part0, -(i16::from(data[0]) + 1)));
        }
        let mut escapes = Vec::new();
        for pair in data[start..start + paired].chunks_exact(2) {
            let original = [pair[0], pair[1]];
            let (mapped, symmetry, symmetry_bits) = map_2d_symmetry(original, lav as i8);
            let packed = (i16::from(mapped[0]) << 4) | i16::from(mapped[1]);
            if write_huffman_code(writer, table, -(packed + 1)) {
                writer.write(symmetry as u32, symmetry_bits);
            } else {
                assert!(write_huffman_code(writer, table, 0));
                escapes.push(original);
            }
        }
        if !escapes.is_empty() {
            let flattened = escapes
                .iter()
                .flat_map(|pair| pair.iter().copied())
                .collect::<Vec<_>>();
            pcm_encode(writer, &flattened, (2 * lav + 1) as u32, lav as i8);
        }
        if let (Some(value), Some(code)) = (remainder, remainder_code.as_ref()) {
            for &bit in code {
                writer.write_bool(bit);
            }
            if value != 0 {
                writer.write_bool(value < 0);
            }
        }
    }))
}

fn map_2d_symmetry(mut pair: [i8; 2], lav: i8) -> ([i8; 2], u8, usize) {
    let mut sum = pair[0] + pair[1];
    let mut difference = pair[0] - pair[1];
    let mut symmetry = 0u8;
    let mut bits = 0usize;
    if sum != 0 {
        let negative = sum < 0;
        if negative {
            sum = -sum;
            difference = -difference;
        }
        symmetry = u8::from(negative);
        bits += 1;
    }
    if difference != 0 {
        let negative = difference < 0;
        difference = difference.abs();
        symmetry = (symmetry << 1) | u8::from(negative);
        bits += 1;
    }
    pair = if sum % 2 != 0 {
        [lav - sum / 2, lav - difference / 2]
    } else {
        [sum / 2, difference / 2]
    };
    (pair, symmetry, bits)
}

fn pcm_encode(writer: &mut BitWriter, values: &[i8], levels: u32, offset: i8) {
    let max_group = match levels {
        3 => 5,
        7 => 6,
        11 => 2,
        13 | 19 | 51 => 4,
        25 => 3,
        4 | 8 | 15 | 16 | 26 | 31 => 1,
        _ => unreachable!("validated MPEG Surround PCM level count"),
    };
    for group in values.chunks(max_group) {
        let mut packed = 0u32;
        for &value in group {
            packed = packed * levels + u32::try_from(i16::from(value) + i16::from(offset)).unwrap();
        }
        writer.write(packed, bit_width(levels.pow(group.len() as u32) as usize));
    }
}

impl Mps212FrameDecoder {
    pub fn new(
        time_slots: usize,
        bands: usize,
        ipd_bands: usize,
        high_rate: bool,
        phase_coding: bool,
    ) -> Self {
        Self {
            time_slots,
            high_rate,
            phase_coding,
            temporal_shape_config: 0,
            num_parameter_set_bits: 3,
            cld: ParameterHistory::new(bands, 0),
            icc: ParameterHistory::new(bands, 0),
            ipd: ParameterHistory::new(ipd_bands, 0),
        }
    }

    pub fn with_temporal_shape_config(mut self, config: u8) -> Self {
        self.temporal_shape_config = config;
        self
    }

    pub fn with_low_delay_framing(mut self) -> Self {
        self.num_parameter_set_bits = 1;
        self
    }

    pub fn parse(
        &mut self,
        reader: &mut BitReader<'_>,
        global_independent: bool,
    ) -> Result<Mps212Frame, MpsError> {
        let (framing, count) = if self.high_rate {
            (
                reader.read_bool()?,
                reader.read(self.num_parameter_set_bits)? as usize + 1,
            )
        } else {
            (false, 1)
        };
        if count >= 8 {
            return Err(MpsError::InvalidParameterSets);
        }
        let slots = if framing {
            let bits = bit_width(self.time_slots);
            let mut slots = Vec::with_capacity(count);
            let mut previous = None;
            for _ in 0..count {
                let slot = reader.read(bits)? as usize;
                if slot >= self.time_slots || previous.is_some_and(|p| slot <= p) {
                    return Err(MpsError::InvalidParameterSlot);
                }
                slots.push(slot);
                previous = Some(slot);
            }
            slots
        } else {
            (0..count)
                .map(|i| self.time_slots * (i + 1) / count - 1)
                .collect()
        };
        let independent = global_independent || reader.read_bool()?;

        let cld = parse_parameter_data_mode(
            reader,
            count,
            independent,
            MpsParameterKind::Cld,
            &mut self.cld,
            self.num_parameter_set_bits == 1,
        )?;
        let icc = parse_parameter_data_mode(
            reader,
            count,
            independent,
            MpsParameterKind::Icc,
            &mut self.icc,
            self.num_parameter_set_bits == 1,
        )?;
        let ipd = if self.phase_coding && reader.read_bool()? {
            let _opd_smoothing = reader.read_bool()?;
            Some(parse_parameter_data_mode(
                reader,
                count,
                independent,
                MpsParameterKind::Ipd,
                &mut self.ipd,
                self.num_parameter_set_bits == 1,
            )?)
        } else {
            None
        };
        let smoothing = if self.high_rate {
            (0..count)
                .map(|_| {
                    if self.num_parameter_set_bits == 1 && reader.remaining_bits() < 2 {
                        // The FDK LD writer may omit a trailing all-zero
                        // smoothing mode. Its bit reader supplies zero beyond
                        // the declared payload, so mirror that default without
                        // weakening EOF checks for other MPS syntax.
                        Ok(MpsSmoothing::default())
                    } else {
                        parse_smoothing(reader, self.bands())
                    }
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            vec![MpsSmoothing::default(); count]
        };
        let transient_shaping = (self.temporal_shape_config == 3)
            .then(|| parse_transient_shaping(reader, self.time_slots))
            .transpose()?;
        let (stp_enabled, ges_envelopes) =
            parse_stp_ges(reader, self.temporal_shape_config, self.time_slots)?;

        Ok(Mps212Frame {
            independent,
            transient_shaping,
            stp_enabled,
            ges_envelopes,
            parameter_sets: (0..count)
                .map(|i| MpsParameterSet {
                    slot: slots[i],
                    cld: cld[i].clone(),
                    icc: icc[i].clone(),
                    ipd: ipd.as_ref().map(|sets| sets[i].clone()),
                    smoothing: smoothing[i].clone(),
                })
                .collect(),
        })
    }

    fn bands(&self) -> usize {
        self.cld.values.len()
    }
}

fn read_wide(reader: &mut BitReader<'_>, bits: usize) -> Result<u128, MpsError> {
    let mut value = 0u128;
    let mut remaining = bits;
    while remaining != 0 {
        let chunk = remaining.min(32);
        value = (value << chunk) | u128::from(reader.read(chunk)?);
        remaining -= chunk;
    }
    Ok(value)
}

fn binomial(n: usize, k: usize) -> u128 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    (1..=k).fold(1u128, |value, i| value * (n - k + i) as u128 / i as u128)
}

fn parse_transient_shaping(
    reader: &mut BitReader<'_>,
    time_slots: usize,
) -> Result<MpsTransientShaping, MpsError> {
    let count_bits = match time_slots {
        32 => 4,
        64 => 5,
        _ => return Err(MpsError::InvalidParameterSlot),
    };
    let enabled = reader.read_bool()?;
    if !enabled {
        return Ok(MpsTransientShaping {
            enabled,
            phases: vec![None; time_slots],
        });
    }
    let count_code = reader.read_u8(count_bits)? as usize;
    let transient_count = count_code + 1;
    let code_bits = bit_width(binomial(time_slots, transient_count) as usize);
    let mut rank = read_wide(reader, code_bits)?;
    let mut phases = vec![None; time_slots];
    let mut remaining = transient_count;
    for slot in (0..time_slots).rev() {
        if remaining > slot {
            for value in &mut phases[..=slot] {
                *value = Some(0);
            }
            break;
        }
        let combinations = binomial(slot, remaining);
        if rank >= combinations {
            rank -= combinations;
            phases[slot] = Some(0);
            remaining -= 1;
            if remaining == 0 {
                break;
            }
        }
    }
    for phase in phases.iter_mut().filter(|phase| phase.is_some()) {
        *phase = Some(reader.read_u8(3)?);
    }
    Ok(MpsTransientShaping { enabled, phases })
}

fn parse_stp_ges(
    reader: &mut BitReader<'_>,
    config: u8,
    time_slots: usize,
) -> Result<(Vec<bool>, Vec<Option<Vec<u8>>>), MpsError> {
    let channels = 2;
    let mut stp = vec![false; channels];
    let mut ges = vec![None; channels];
    if !matches!(config, 1 | 2) || !reader.read_bool()? {
        return Ok((stp, ges));
    }
    if config == 1 {
        for enabled in &mut stp {
            *enabled = reader.read_bool()?;
        }
    } else {
        let enabled = (0..channels)
            .map(|_| reader.read_bool())
            .collect::<Result<Vec<_>, _>>()?;
        for (channel, enabled) in enabled.into_iter().enumerate() {
            if enabled {
                ges[channel] = Some(decode_reshape_envelope(reader, time_slots)?);
            }
        }
    }
    Ok((stp, ges))
}

fn decode_reshape_envelope(
    reader: &mut BitReader<'_>,
    time_slots: usize,
) -> Result<Vec<u8>, MpsError> {
    let mut output = Vec::with_capacity(time_slots);
    while output.len() < time_slots {
        let node = huffman_node(reader, &RESHAPE_2D)?;
        let packed = -(node + 1);
        let value = (packed >> 4) as u8;
        let length = (packed & 15) as usize + 1;
        if value > 4 || output.len() + length > time_slots {
            return Err(MpsError::InvalidHuffmanCodeword);
        }
        output.resize(output.len() + length, value);
    }
    Ok(output)
}

fn parse_smoothing(
    reader: &mut BitReader<'_>,
    frequency_bands: usize,
) -> Result<MpsSmoothing, MpsError> {
    let mode = reader.read_u8(2)?;
    let time = (mode >= 2).then(|| reader.read_u8(2)).transpose()?;
    let (stride_index, bands) = if mode == 3 {
        let stride_index = reader.read_u8(2)?;
        let count = (frequency_bands - 1) / STRIDES[stride_index as usize] + 1;
        let bands = (0..count)
            .map(|_| reader.read_bool())
            .collect::<Result<Vec<_>, _>>()?;
        (Some(stride_index), bands)
    } else {
        (None, Vec::new())
    };
    Ok(MpsSmoothing {
        mode,
        time,
        stride_index,
        bands,
    })
}

fn bit_width(values: usize) -> usize {
    if values <= 1 {
        0
    } else {
        usize::BITS as usize - (values - 1).leading_zeros() as usize
    }
}

fn parse_parameter_data(
    reader: &mut BitReader<'_>,
    count: usize,
    independent: bool,
    kind: MpsParameterKind,
    history: &mut ParameterHistory,
) -> Result<Vec<Vec<i8>>, MpsError> {
    parse_parameter_data_mode(reader, count, independent, kind, history, false)
}

fn parse_parameter_data_mode(
    reader: &mut BitReader<'_>,
    count: usize,
    independent: bool,
    kind: MpsParameterKind,
    history: &mut ParameterHistory,
    low_delay: bool,
) -> Result<Vec<Vec<i8>>, MpsError> {
    let mut modes = Vec::with_capacity(count);
    for i in 0..count {
        let mode = match (reader.read_bool()?, reader.read_bool()?) {
            (false, false) => ParameterDataMode::Default,
            (false, true) => ParameterDataMode::Keep,
            (true, false) => ParameterDataMode::Interpolate,
            (true, true) => ParameterDataMode::New,
        };
        if independent
            && i == 0
            && matches!(
                mode,
                ParameterDataMode::Keep | ParameterDataMode::Interpolate
            )
            || i + 1 == count && mode == ParameterDataMode::Interpolate
        {
            return Err(MpsError::InvalidDataMode);
        }
        modes.push(mode);
    }

    let default = 0;
    let mut sets: Vec<Option<Vec<i8>>> = vec![None; count];
    let mut i = 0;
    while i < count {
        match modes[i] {
            ParameterDataMode::Default => {
                history.values.fill(default);
                history.coarse = false;
                sets[i] = Some(history.values.clone());
                i += 1;
            }
            ParameterDataMode::Keep => {
                sets[i] = Some(history.values.clone());
                i += 1;
            }
            ParameterDataMode::Interpolate => i += 1,
            ParameterDataMode::New => {
                let pair = reader.read_bool()?;
                let coarse = reader.read_bool()?;
                let stride_index = reader.read_u8(2)? as usize;
                convert_history_quantization(history, kind, coarse);
                let map = stride_map(history.values.len(), STRIDES[stride_index]);
                let previous: Vec<_> = map.iter().map(|&band| history.values[band]).collect();
                let decoded = decode_pcm_or_huffman_mode(
                    reader,
                    kind,
                    coarse,
                    pair,
                    map.len(),
                    &previous,
                    !(independent && i == 0),
                    low_delay,
                )?;
                let decoded_count = if pair { 2 } else { 1 };
                for set_offset in 0..decoded_count {
                    if i + set_offset >= count {
                        return Err(MpsError::InvalidDataMode);
                    }
                    let expanded = expand_stride(&decoded[set_offset], &map, history.values.len());
                    sets[i + set_offset] = Some(expanded.clone());
                    history.values = expanded;
                }
                history.coarse = coarse;
                i += decoded_count;
            }
        }
    }

    for i in 0..count {
        if modes[i] == ParameterDataMode::Interpolate {
            let left = if i == 0 {
                history.values.clone()
            } else {
                sets[i - 1].as_ref().unwrap().clone()
            };
            let right_index = (i + 1..count)
                .find(|&j| sets[j].is_some())
                .ok_or(MpsError::InvalidDataMode)?;
            let right = sets[right_index].as_ref().unwrap();
            let distance = (right_index - i + 1) as i16;
            sets[i] = Some(
                left.iter()
                    .zip(right)
                    .map(|(&a, &b)| {
                        let numerator = i16::from(a) * (distance - 1) + i16::from(b);
                        (numerator / distance) as i8
                    })
                    .collect(),
            );
        }
    }
    let result: Vec<_> = sets.into_iter().map(Option::unwrap).collect();
    history.values = result.last().unwrap().clone();
    Ok(result)
}

fn convert_history_quantization(
    history: &mut ParameterHistory,
    kind: MpsParameterKind,
    coarse: bool,
) {
    if coarse == history.coarse {
        return;
    }
    if coarse {
        for value in &mut history.values {
            *value = if kind == MpsParameterKind::Cld {
                *value / 2
            } else {
                *value >> 1
            };
        }
    } else {
        for value in &mut history.values {
            *value <<= 1;
            if kind == MpsParameterKind::Cld && *value == -14 {
                *value = -15;
            } else if kind == MpsParameterKind::Cld && *value == 14 {
                *value = 15;
            }
        }
    }
}

fn stride_map(bands: usize, stride: usize) -> Vec<usize> {
    let data_bands = (bands - 1) / stride + 1;
    let mut edges: Vec<_> = (0..=data_bands).map(|i| i * stride).collect();
    let mut offset = 0;
    while edges[data_bands] > bands {
        if offset < data_bands {
            offset += 1;
        }
        for edge in &mut edges[offset..] {
            *edge -= 1;
        }
    }
    edges.truncate(data_bands);
    edges
}

fn expand_stride(values: &[i8], starts: &[usize], bands: usize) -> Vec<i8> {
    let mut out = vec![0; bands];
    for (i, (&value, &start)) in values.iter().zip(starts).enumerate() {
        let end = starts.get(i + 1).copied().unwrap_or(bands);
        out[start..end].fill(value);
    }
    out
}

fn decode_pcm_or_huffman(
    reader: &mut BitReader<'_>,
    kind: MpsParameterKind,
    coarse: bool,
    pair: bool,
    bands: usize,
    history: &[i8],
    allow_time_backwards: bool,
) -> Result<Vec<Vec<i8>>, MpsError> {
    decode_pcm_or_huffman_mode(
        reader,
        kind,
        coarse,
        pair,
        bands,
        history,
        allow_time_backwards,
        false,
    )
}

fn decode_pcm_or_huffman_mode(
    reader: &mut BitReader<'_>,
    kind: MpsParameterKind,
    coarse: bool,
    pair: bool,
    bands: usize,
    history: &[i8],
    allow_time_backwards: bool,
    low_delay: bool,
) -> Result<Vec<Vec<i8>>, MpsError> {
    let (levels, offset) = match (kind, coarse) {
        (MpsParameterKind::Cld, true) => (15, 7),
        (MpsParameterKind::Cld, false) => (31, 15),
        (MpsParameterKind::Icc, true) => (4, 0),
        (MpsParameterKind::Icc, false) => (8, 0),
        (MpsParameterKind::Ipd, true) => (8, 0),
        (MpsParameterKind::Ipd, false) => (16, 0),
    };
    if !reader.read_bool()? {
        return decode_huffman_1d_mode(
            reader,
            kind,
            pair,
            bands,
            history,
            offset,
            kind == MpsParameterKind::Ipd && !coarse,
            allow_time_backwards,
            low_delay,
        );
    }
    let count = bands * if pair { 2 } else { 1 };
    let values = pcm_decode(reader, count, levels, offset)?;
    if pair {
        Ok(vec![
            values.iter().step_by(2).copied().collect(),
            values.iter().skip(1).step_by(2).copied().collect(),
        ])
    } else {
        Ok(vec![values])
    }
}

fn named_huffman_table(name: &str) -> Vec<[i16; 2]> {
    let start = HUFFMAN_SOURCE.find(name).expect("MPS Huffman ROM table");
    let source = &HUFFMAN_SOURCE[start..];
    let body_start = source.find("{{").unwrap() + 1;
    let body_end = source[body_start..].find("}}").unwrap() + body_start;
    source[body_start..body_end]
        .split('{')
        .skip(1)
        .filter_map(|entry| {
            let end = entry.find('}')?;
            let mut values = entry[..end]
                .split(',')
                .map(|value| value.trim().parse::<i16>().ok());
            Some([values.next()??, values.next()??])
        })
        .collect()
}

static CLD_1D_FREQ: LazyLock<Vec<[i16; 2]>> =
    LazyLock::new(|| named_huffman_table("FDK_huffCLDNodes_h1D_0"));
static CLD_1D_TIME: LazyLock<Vec<[i16; 2]>> =
    LazyLock::new(|| named_huffman_table("FDK_huffCLDNodes_h1D_1"));
static ICC_1D: LazyLock<Vec<[i16; 2]>> =
    LazyLock::new(|| named_huffman_table("FDK_huffICCNodes_h1D_0"));

fn named_huffman_pairs(name: &str) -> Vec<[i16; 2]> {
    let start = HUFFMAN_SOURCE
        .find(name)
        .expect("MPS Huffman ROM structure");
    let source = &HUFFMAN_SOURCE[start..];
    let open = source.find('{').unwrap();
    let mut depth = 0usize;
    let mut close = open;
    for (offset, byte) in source.as_bytes()[open..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    close = open + offset;
                    break;
                }
            }
            _ => {}
        }
    }
    let body = &source[open..=close];
    body.split('{')
        .skip(1)
        .filter_map(|entry| {
            let end = entry.find('}')?;
            let text = &entry[..end];
            let values: Vec<_> = text
                .split(',')
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(|value| value.parse::<i16>().ok())
                .collect::<Option<_>>()?;
            (values.len() == 2).then(|| [values[0], values[1]])
        })
        .collect()
}

static CLD_2D_00: LazyLock<Vec<[i16; 2]>> =
    LazyLock::new(|| named_huffman_pairs("FDK_huffCLDNodes_h2_0_0"));
static CLD_2D_01: LazyLock<Vec<[i16; 2]>> =
    LazyLock::new(|| named_huffman_pairs("FDK_huffCLDNodes_h2_0_1"));
static CLD_2D_10: LazyLock<Vec<[i16; 2]>> =
    LazyLock::new(|| named_huffman_pairs("FDK_huffCLDNodes_h2_1_0"));
static CLD_2D_11: LazyLock<Vec<[i16; 2]>> =
    LazyLock::new(|| named_huffman_pairs("FDK_huffCLDNodes_h2_1_1"));
static ICC_2D_00: LazyLock<Vec<[i16; 2]>> =
    LazyLock::new(|| named_huffman_pairs("FDK_huffICCNodes_h2D_0_0"));
static ICC_2D_01: LazyLock<Vec<[i16; 2]>> =
    LazyLock::new(|| named_huffman_pairs("FDK_huffICCNodes_h2D_0_1"));
static ICC_2D_10: LazyLock<Vec<[i16; 2]>> =
    LazyLock::new(|| named_huffman_pairs("FDK_huffICCNodes_h2D_1_0"));
static ICC_2D_11: LazyLock<Vec<[i16; 2]>> =
    LazyLock::new(|| named_huffman_pairs("FDK_huffICCNodes_h2D_1_1"));
static IPD_ALL: LazyLock<Vec<[i16; 2]>> =
    LazyLock::new(|| named_huffman_pairs("FDK_huffIPDNodes ="));
static RESHAPE_2D: LazyLock<Vec<[i16; 2]>> =
    LazyLock::new(|| named_huffman_pairs("FDK_huffReshapeNodes ="));

const CLD_PART0: [[i16; 2]; 30] = [
    [2, 1],
    [4, 3],
    [6, 5],
    [8, 7],
    [10, 9],
    [12, 11],
    [14, 13],
    [-8, 15],
    [-9, 16],
    [-10, 17],
    [-18, 18],
    [-17, -19],
    [-16, 19],
    [-11, -20],
    [-15, -21],
    [-7, 20],
    [-22, 21],
    [-12, -14],
    [-13, -23],
    [23, 22],
    [-24, -31],
    [-6, 24],
    [-25, -26],
    [26, 25],
    [-5, -27],
    [-28, 27],
    [-4, 28],
    [-29, 29],
    [-1, -30],
    [-2, -3],
];
const ICC_PART0: [[i16; 2]; 7] = [
    [2, 1],
    [-5, 3],
    [-4, -6],
    [-3, 4],
    [-2, 5],
    [-1, 6],
    [-7, -8],
];
const IPD_PART0: [[i16; 2]; 15] = [
    [-1, 1],
    [3, 2],
    [-8, 4],
    [6, 5],
    [-16, 7],
    [9, 8],
    [11, 10],
    [-2, -7],
    [-6, 12],
    [-4, -5],
    [-3, 13],
    [-10, 14],
    [-11, -12],
    [-14, -15],
    [-9, -13],
];
const IPD_1D_FREQ: [[i16; 2]; 7] = [
    [-1, 1],
    [-8, 2],
    [-2, 3],
    [5, 4],
    [-3, -7],
    [-6, 6],
    [-4, -5],
];
const IPD_1D_TIME: [[i16; 2]; 7] = [
    [-1, 1],
    [-2, 2],
    [-8, 3],
    [-3, 4],
    [-7, 5],
    [-4, 6],
    [-5, -6],
];
const LAV_TABLE: [[i16; 2]; 3] = [[-1, 1], [-2, 2], [-3, -4]];

fn table_2d(
    kind: MpsParameterKind,
    time_delta: bool,
    time_pair: bool,
    lav: usize,
) -> &'static [[i16; 2]] {
    let lengths = match kind {
        MpsParameterKind::Cld => [15, 35, 63, 99],
        MpsParameterKind::Icc | MpsParameterKind::Ipd => [3, 15, 35, 63],
    };
    let table: &[[i16; 2]] = match kind {
        MpsParameterKind::Cld => match (time_delta, time_pair) {
            (false, false) => &CLD_2D_00,
            (false, true) => &CLD_2D_01,
            (true, false) => &CLD_2D_10,
            (true, true) => &CLD_2D_11,
        },
        MpsParameterKind::Icc => match (time_delta, time_pair) {
            (false, false) => &ICC_2D_00,
            (false, true) => &ICC_2D_01,
            (true, false) => &ICC_2D_10,
            (true, true) => &ICC_2D_11,
        },
        MpsParameterKind::Ipd => {
            let structure = usize::from(time_delta) * 2 + usize::from(time_pair);
            let base = 21 + structure * lengths.iter().sum::<usize>();
            &IPD_ALL[base..base + lengths.iter().sum::<usize>()]
        }
    };
    let index = match kind {
        MpsParameterKind::Cld => (lav - 3) / 2,
        _ => (lav - 1) / 2,
    };
    let start: usize = lengths[..index].iter().sum();
    &table[start..start + lengths[index]]
}

fn huffman_node(reader: &mut BitReader<'_>, table: &[[i16; 2]]) -> Result<i16, MpsError> {
    let mut node = 0i16;
    loop {
        let row = table
            .get(node as usize)
            .ok_or(MpsError::InvalidHuffmanCodeword)?;
        node = row[usize::from(reader.read_bool()?)];
        if node <= 0 {
            return Ok(node);
        }
    }
}

fn huffman_1d_vector(
    reader: &mut BitReader<'_>,
    kind: MpsParameterKind,
    time_delta: bool,
    bands: usize,
) -> Result<Vec<i8>, MpsError> {
    huffman_1d_vector_mode(reader, kind, time_delta, bands, !time_delta)
}

fn huffman_1d_vector_mode(
    reader: &mut BitReader<'_>,
    kind: MpsParameterKind,
    time_delta: bool,
    bands: usize,
    partition_zero: bool,
) -> Result<Vec<i8>, MpsError> {
    let part0: &[[i16; 2]] = match kind {
        MpsParameterKind::Cld => &CLD_PART0,
        MpsParameterKind::Icc => &ICC_PART0,
        MpsParameterKind::Ipd => &IPD_PART0,
    };
    let table: &[[i16; 2]] = match (kind, time_delta) {
        (MpsParameterKind::Cld, false) => &CLD_1D_FREQ,
        (MpsParameterKind::Cld, true) => &CLD_1D_TIME,
        (MpsParameterKind::Icc, _) => &ICC_1D,
        (MpsParameterKind::Ipd, false) => &IPD_1D_FREQ,
        (MpsParameterKind::Ipd, true) => &IPD_1D_TIME,
    };
    let mut out = Vec::with_capacity(bands);
    if partition_zero {
        out.push((-(huffman_node(reader, part0)? + 1)) as i8);
    }
    while out.len() < bands {
        let mut value = (-(huffman_node(reader, table)? + 1)) as i8;
        if kind != MpsParameterKind::Ipd && value != 0 && reader.read_bool()? {
            value = -value;
        }
        out.push(value);
    }
    Ok(out)
}

fn restore_2d_symmetry(
    reader: &mut BitReader<'_>,
    kind: MpsParameterKind,
    lav: i8,
    mut pair: [i8; 2],
) -> Result<[i8; 2], MpsError> {
    let sum = pair[0] + pair[1];
    let difference = pair[0] - pair[1];
    if sum > lav {
        pair[0] = -sum + 2 * lav + 1;
        pair[1] = -difference;
    } else {
        pair = [sum, difference];
    }
    let sign_test = if kind == MpsParameterKind::Ipd {
        pair[0] - pair[1]
    } else {
        pair[0] + pair[1]
    };
    if sign_test != 0 && reader.read_bool()? {
        if kind == MpsParameterKind::Ipd {
            pair.swap(0, 1);
        } else {
            pair[0] = -pair[0];
            pair[1] = -pair[1];
        }
    }
    if kind != MpsParameterKind::Ipd && pair[0] - pair[1] != 0 && reader.read_bool()? {
        pair.swap(0, 1);
    }
    Ok(pair)
}

fn huffman_2d_pairs(
    reader: &mut BitReader<'_>,
    kind: MpsParameterKind,
    time_delta: bool,
    time_pair: bool,
    count: usize,
    partition_zeros: usize,
) -> Result<(Vec<i8>, Vec<[i8; 2]>), MpsError> {
    let lav_code = (-(huffman_node(reader, &LAV_TABLE)? + 1)) as usize;
    let lav = match kind {
        MpsParameterKind::Cld => 2 * lav_code + 3,
        MpsParameterKind::Icc => 2 * lav_code + 1,
        MpsParameterKind::Ipd => 2 * (if lav_code == 0 { 3 } else { lav_code - 1 }) + 1,
    };
    let part0: &[[i16; 2]] = match kind {
        MpsParameterKind::Cld => &CLD_PART0,
        MpsParameterKind::Icc => &ICC_PART0,
        MpsParameterKind::Ipd => &IPD_PART0,
    };
    // FDK's huff_dec_2D reads the LAV first and only then the frequency
    // partition-zero value(s).  Reversing these happens to decode some CLD
    // vectors but shifts the following ICC syntax.
    let mut partition = Vec::with_capacity(partition_zeros);
    for _ in 0..partition_zeros {
        partition.push((-(huffman_node(reader, part0)? + 1)) as i8);
    }
    let table = table_2d(kind, time_delta, time_pair, lav);
    let mut out = vec![[0; 2]; count];
    let mut escapes = Vec::new();
    for (index, pair) in out.iter_mut().enumerate() {
        let node = huffman_node(reader, table)?;
        if node == 0 {
            escapes.push(index);
        } else {
            let packed = -(node + 1);
            *pair = restore_2d_symmetry(
                reader,
                kind,
                lav as i8,
                [(packed >> 4) as i8, (packed & 15) as i8],
            )?;
        }
    }
    if !escapes.is_empty() {
        let escaped = pcm_decode(reader, escapes.len() * 2, (2 * lav + 1) as u32, 0)?;
        for (position, &index) in escapes.iter().enumerate() {
            out[index] = [
                escaped[position * 2] - lav as i8,
                escaped[position * 2 + 1] - lav as i8,
            ];
        }
    }
    Ok((partition, out))
}

fn decode_huffman_differences(
    reader: &mut BitReader<'_>,
    kind: MpsParameterKind,
    pair: bool,
    bands: usize,
    time0: bool,
    time1: bool,
) -> Result<(Vec<i8>, Option<Vec<i8>>), MpsError> {
    decode_huffman_differences_mode(reader, kind, pair, bands, time0, time1, false)
}

fn decode_huffman_differences_mode(
    reader: &mut BitReader<'_>,
    kind: MpsParameterKind,
    pair: bool,
    bands: usize,
    time0: bool,
    time1: bool,
    low_delay: bool,
) -> Result<(Vec<i8>, Option<Vec<i8>>), MpsError> {
    let two_dimensional = reader.read_bool()?;
    if !two_dimensional {
        return Ok((
            huffman_1d_vector(reader, kind, time0, bands)?,
            pair.then(|| huffman_1d_vector(reader, kind, time1, bands))
                .transpose()?,
        ));
    }
    let time_pair = pair && !low_delay && reader.read_bool()?;
    if time_pair {
        let has_partition_zero = !time0 || !time1;
        let mut first = Vec::with_capacity(bands);
        let mut second = Vec::with_capacity(bands);
        let (partition, pairs) = huffman_2d_pairs(
            reader,
            kind,
            time0 || time1,
            true,
            bands - usize::from(has_partition_zero),
            usize::from(has_partition_zero) * 2,
        )?;
        if has_partition_zero {
            first.push(partition[0]);
            second.push(partition[1]);
        }
        for values in pairs {
            first.push(values[0]);
            second.push(values[1]);
        }
        return Ok((first, Some(second)));
    }

    let decode_frequency = |reader: &mut BitReader<'_>, time: bool| -> Result<Vec<i8>, MpsError> {
        let mut values = Vec::with_capacity(bands);
        let paired = (bands - usize::from(!time)) / 2;
        let (partition, pairs) =
            huffman_2d_pairs(reader, kind, time, false, paired, usize::from(!time))?;
        values.extend(partition);
        for pair in pairs {
            values.extend(pair);
        }
        if values.len() < bands {
            // A frequency-paired 2-D vector with an odd remainder uses the
            // frequency Huffman table, but has no partition-zero value.  The
            // two choices are independent in the MPEG Surround syntax.
            values.extend(huffman_1d_vector_mode(reader, kind, time, 1, false)?);
        }
        Ok(values)
    };
    Ok((
        decode_frequency(reader, time0)?,
        pair.then(|| decode_frequency(reader, time1)).transpose()?,
    ))
}

#[allow(clippy::too_many_arguments)]
fn decode_huffman_1d(
    reader: &mut BitReader<'_>,
    kind: MpsParameterKind,
    pair: bool,
    bands: usize,
    history: &[i8],
    offset: i8,
    attach_lsb: bool,
    allow_time_backwards: bool,
) -> Result<Vec<Vec<i8>>, MpsError> {
    decode_huffman_1d_mode(
        reader,
        kind,
        pair,
        bands,
        history,
        offset,
        attach_lsb,
        allow_time_backwards,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn decode_huffman_1d_mode(
    reader: &mut BitReader<'_>,
    kind: MpsParameterKind,
    pair: bool,
    bands: usize,
    history: &[i8],
    offset: i8,
    attach_lsb: bool,
    allow_time_backwards: bool,
    low_delay: bool,
) -> Result<Vec<Vec<i8>>, MpsError> {
    let time0 = if pair || allow_time_backwards {
        reader.read_bool()?
    } else {
        false
    };
    let time1 = if pair && (!time0 || allow_time_backwards) {
        reader.read_bool()?
    } else {
        false
    };
    let (diff0, diff1) =
        decode_huffman_differences_mode(reader, kind, pair, bands, time0, time1, low_delay)?;
    let backwards = if low_delay {
        true
    } else if pair && time0 && !allow_time_backwards {
        false
    } else if pair && time1 {
        true
    } else if pair && time0 {
        !reader.read_bool()?
    } else {
        true
    };
    let history_msb: Vec<i8> = history
        .iter()
        .map(|&value| {
            let value = value + offset;
            if attach_lsb {
                value >> 1
            } else {
                value
            }
        })
        .collect();
    let frequency = |diff: &[i8]| {
        let mut out = diff.to_vec();
        for i in 1..out.len() {
            out[i] += out[i - 1];
        }
        out
    };
    let time_back =
        |base: &[i8], diff: &[i8]| base.iter().zip(diff).map(|(&a, &b)| a + b).collect();
    let time_forward =
        |base: &[i8], diff: &[i8]| base.iter().zip(diff).map(|(&a, &b)| a - b).collect();
    let (mut first, mut second) = if backwards {
        let first = if time0 {
            time_back(&history_msb, &diff0)
        } else {
            frequency(&diff0)
        };
        let second = diff1.as_ref().map(|diff| {
            if time1 {
                time_back(&first, diff)
            } else {
                frequency(diff)
            }
        });
        (first, second)
    } else {
        let second = frequency(diff1.as_ref().unwrap());
        // `backwards == false` is only selected from a paired time-delta
        // branch, which necessarily has `time0 == true`.
        let first = time_forward(&second, &diff0);
        (first, Some(second))
    };
    let attach = |values: &mut [i8], reader: &mut BitReader<'_>| -> Result<(), MpsError> {
        for value in values {
            *value = if attach_lsb {
                ((*value << 1) | i8::from(reader.read_bool()?)) - offset
            } else {
                *value - offset
            };
        }
        Ok(())
    };
    attach(&mut first, reader)?;
    if let Some(values) = &mut second {
        attach(values, reader)?;
    }
    let mut result = vec![first];
    if let Some(values) = second {
        result.push(values);
    }
    Ok(result)
}

fn pcm_decode(
    reader: &mut BitReader<'_>,
    count: usize,
    levels: u32,
    offset: i8,
) -> Result<Vec<i8>, MpsError> {
    let max_group = match levels {
        3 => 5,
        7 => 6,
        11 => 2,
        13 | 19 | 51 => 4,
        25 => 3,
        4 | 8 | 15 | 16 | 26 | 31 => 1,
        _ => return Err(MpsError::InvalidDataMode),
    };
    let mut out = vec![0; count];
    for base in (0..count).step_by(max_group) {
        let length = max_group.min(count - base);
        let combinations = levels.pow(length as u32);
        let bits = bit_width(combinations as usize);
        let mut packed = reader.read(bits)?;
        for j in 0..length {
            let index = base + length - j - 1;
            out[index] = (packed % levels) as i8 - offset;
            packed /= levels;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitWriter;

    #[test]
    fn grouped_pcm_frame_encoder_roundtrips_new_and_keep_modes() {
        let mut encoder = Mps212FrameEncoder::new(16, 15).unwrap();
        let mut decoder = Mps212FrameDecoder::new(16, 15, 0, true, false);
        let cld = [-15, -10, -6, -4, -2, 0, 2, 4, 6, 8, 10, 13, 15, 3, -3];
        let icc = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6];
        let (payload, bits) = encoder.encode(&cld, &icc, true).unwrap();
        let frame = decoder
            .parse(&mut BitReader::with_bit_len(&payload, bits).unwrap(), false)
            .unwrap();
        assert!(frame.independent);
        assert_eq!(frame.parameter_sets[0].slot, 15);
        assert_eq!(frame.parameter_sets[0].cld, cld);
        assert_eq!(frame.parameter_sets[0].icc, icc);

        let (keep, keep_bits) = encoder.encode(&cld, &icc, false).unwrap();
        assert!(keep_bits < bits);
        let frame = decoder
            .parse(
                &mut BitReader::with_bit_len(&keep, keep_bits).unwrap(),
                false,
            )
            .unwrap();
        assert!(!frame.independent);
        assert_eq!(frame.parameter_sets[0].cld, cld);
        assert_eq!(frame.parameter_sets[0].icc, icc);
    }

    fn code_for(table: &[[i16; 2]], target: i16) -> Vec<bool> {
        fn visit(table: &[[i16; 2]], node: i16, target: i16, path: &mut Vec<bool>) -> bool {
            for bit in [false, true] {
                path.push(bit);
                let child = table[node as usize][usize::from(bit)];
                if child == target || child > 0 && visit(table, child, target, path) {
                    return true;
                }
                path.pop();
            }
            false
        }
        let mut path = Vec::new();
        assert!(visit(table, 0, target, &mut path));
        path
    }

    #[test]
    fn parses_fixed_frame_with_pcm_cld_and_default_icc() {
        let mut w = BitWriter::new();
        w.write(1, 1); // independent
        w.write(3, 2); // CLD new
        w.write(0, 1); // unpaired
        w.write(0, 1); // fine
        w.write(0, 2); // stride 1
        w.write(1, 1); // PCM
        w.write(16, 5);
        w.write(14, 5);
        w.write(18, 5); // +1,-1,+3
        w.write(0, 2); // ICC default
        let bit_len = w.bits_written();
        let bytes = w.finish();
        let mut reader = BitReader::new(&bytes);
        let mut decoder = Mps212FrameDecoder::new(16, 3, 0, false, false);
        let frame = decoder.parse(&mut reader, false).unwrap();
        assert_eq!(frame.parameter_sets[0].slot, 15);
        assert_eq!(frame.parameter_sets[0].cld, [1, -1, 3]);
        assert_eq!(frame.parameter_sets[0].icc, [0, 0, 0]);
        for truncated_len in 0..bit_len {
            let mut reader = BitReader::with_bit_len(&bytes, truncated_len).unwrap();
            assert!(Mps212FrameDecoder::new(16, 3, 0, false, false)
                .parse(&mut reader, false)
                .is_err());
        }
    }

    #[test]
    fn stride_expansion_matches_centered_fdk_map() {
        assert_eq!(stride_map(7, 5), [0, 4]);
        assert_eq!(
            expand_stride(&[2, -3], &[0, 4], 7),
            [2, 2, 2, 2, -3, -3, -3]
        );
    }

    #[test]
    fn spatial_matrix_realizes_cld_and_ipd() {
        let (left, right) = spatial_upmix_band((1.0, 0.0), (0.0, 1.0), 0, 5, Some(4));
        let left_power = left.0 * left.0 + left.1 * left.1;
        let right_power = right.0 * right.0 + right.1 * right.1;
        assert!((left_power - right_power).abs() < 1e-6);
        assert!((left_power + right_power - 1.0).abs() < 1e-6);
        // The pi/2 IPD rotation maps the pre-rotation (+,-) quadrant to (+,+).
        assert!(right.0 > 0.0 && right.1 > 0.0);
    }

    #[test]
    fn spatial_matrix_cld_extremes_select_a_side() {
        let (left, right) = spatial_upmix_band((1.0, 0.0), (0.0, 0.0), 15, 0, None);
        assert!(left.0 > 0.999);
        assert!(right.0 < 1e-6);
    }

    #[test]
    fn prediction_matrix_uses_residual_as_opposite_component() {
        let (left, right) =
            spatial_prediction_upmix_band((1.0, 0.0), (0.25, 0.0), 0, 0, Some(8), true);
        assert!(left.0 > right.0);
        assert!((left.0 - (1.25 / 2.4)).abs() < 1e-6);
        assert!((right.0 - (0.75 / 2.4)).abs() < 1e-6);
    }

    #[test]
    fn decodes_1d_frequency_huffman_and_delta() {
        let mut writer = BitWriter::new();
        writer.write(0, 1); // Huffman instead of PCM
        writer.write(0, 1); // 1D coding scheme
        for bit in code_for(&ICC_PART0, -1) {
            writer.write(bit as u32, 1);
        }
        for _ in 1..4 {
            for bit in code_for(&ICC_1D, -1) {
                writer.write(bit as u32, 1);
            }
        }
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        let values = decode_pcm_or_huffman(
            &mut reader,
            MpsParameterKind::Icc,
            false,
            false,
            4,
            &[0; 4],
            false,
        )
        .unwrap();
        assert_eq!(values, [vec![0, 0, 0, 0]]);
    }

    #[test]
    fn parses_selective_smoothing_stride() {
        let mut writer = BitWriter::new();
        writer.write(3, 2);
        writer.write(2, 2);
        writer.write(1, 2); // stride 2, three transmitted bands for 6 bands
        writer.write(0b101, 3);
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        assert_eq!(
            parse_smoothing(&mut reader, 6).unwrap(),
            MpsSmoothing {
                mode: 3,
                time: Some(2),
                stride_index: Some(1),
                bands: vec![true, false, true],
            }
        );
    }

    #[test]
    fn decodes_2d_frequency_pair_from_fdk_rom() {
        let table = table_2d(MpsParameterKind::Icc, false, false, 1);
        assert_eq!(table.len(), 3);
        let mut writer = BitWriter::new();
        writer.write(1, 1); // 2D coding scheme
        for bit in code_for(&LAV_TABLE, -1) {
            writer.write(bit as u32, 1);
        }
        for bit in code_for(&ICC_PART0, -1) {
            writer.write(bit as u32, 1);
        }
        for bit in code_for(table, -1) {
            writer.write(bit as u32, 1);
        }
        let bit_len = writer.bits_written();
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        let (first, second) =
            decode_huffman_differences(&mut reader, MpsParameterKind::Icc, false, 3, false, false)
                .unwrap();
        assert_eq!(first, [0, 0, 0]);
        assert_eq!(second, None);
        for truncated_len in 0..bit_len {
            let mut reader = BitReader::with_bit_len(&bytes, truncated_len).unwrap();
            assert!(decode_huffman_differences(
                &mut reader,
                MpsParameterKind::Icc,
                false,
                3,
                false,
                false,
            )
            .is_err());
        }

        let mut truncated_symmetry = BitWriter::new();
        truncated_symmetry.write_bool(true); // 2D coding scheme
        for bit in code_for(&LAV_TABLE, -1) {
            truncated_symmetry.write_bool(bit);
        }
        for bit in code_for(&ICC_PART0, -1) {
            truncated_symmetry.write_bool(bit);
        }
        for bit in code_for(table, -2) {
            truncated_symmetry.write_bool(bit);
        }
        let bit_len = truncated_symmetry.bits_written();
        let bytes = truncated_symmetry.finish();
        assert!(decode_huffman_differences(
            &mut BitReader::with_bit_len(&bytes, bit_len).unwrap(),
            MpsParameterKind::Icc,
            false,
            3,
            false,
            false,
        )
        .is_err());

        let mut truncated_time_pair = BitWriter::new();
        truncated_time_pair.write_bool(true); // 2D coding scheme
        truncated_time_pair.write_bool(true); // time-paired parameter sets
        for bit in code_for(&LAV_TABLE, -1) {
            truncated_time_pair.write_bool(bit);
        }
        for _ in 0..2 {
            for bit in code_for(&ICC_PART0, -1) {
                truncated_time_pair.write_bool(bit);
            }
        }
        for bit in code_for(table_2d(MpsParameterKind::Icc, false, true, 1), -2) {
            truncated_time_pair.write_bool(bit);
        }
        let bit_len = truncated_time_pair.bits_written();
        let bytes = truncated_time_pair.finish();
        assert!(decode_huffman_differences(
            &mut BitReader::with_bit_len(&bytes, bit_len).unwrap(),
            MpsParameterKind::Icc,
            true,
            2,
            false,
            false,
        )
        .is_err());

        let mut writer = BitWriter::new();
        writer.write_bool(false); // 1D coding
        for _ in 0..2 {
            for bit in code_for(&CLD_1D_TIME, -1) {
                writer.write_bool(bit);
            }
        }
        let bytes = writer.finish();
        assert_eq!(
            decode_huffman_differences(
                &mut BitReader::new(&bytes),
                MpsParameterKind::Cld,
                true,
                1,
                true,
                true,
            )
            .unwrap(),
            (vec![0], Some(vec![0]))
        );

        let mut writer = BitWriter::new();
        writer.write_bool(true); // 2D coding
        writer.write_bool(false); // separate frequency vectors
        for _ in 0..2 {
            for bit in code_for(&CLD_1D_TIME, -1) {
                writer.write_bool(bit);
            }
        }
        let bytes = writer.finish();
        assert_eq!(
            decode_huffman_differences(
                &mut BitReader::new(&bytes),
                MpsParameterKind::Cld,
                true,
                1,
                true,
                true,
            )
            .unwrap(),
            (vec![0], Some(vec![0]))
        );
    }

    #[test]
    fn huffman_vectors_cover_parameter_kinds_time_tables_and_signed_values() {
        for kind in [
            MpsParameterKind::Cld,
            MpsParameterKind::Icc,
            MpsParameterKind::Ipd,
        ] {
            for time_delta in [false, true] {
                let part0: &[[i16; 2]] = match kind {
                    MpsParameterKind::Cld => &CLD_PART0,
                    MpsParameterKind::Icc => &ICC_PART0,
                    MpsParameterKind::Ipd => &IPD_PART0,
                };
                let table: &[[i16; 2]] = match (kind, time_delta) {
                    (MpsParameterKind::Cld, false) => &CLD_1D_FREQ,
                    (MpsParameterKind::Cld, true) => &CLD_1D_TIME,
                    (MpsParameterKind::Icc, _) => &ICC_1D,
                    (MpsParameterKind::Ipd, false) => &IPD_1D_FREQ,
                    (MpsParameterKind::Ipd, true) => &IPD_1D_TIME,
                };
                let mut writer = BitWriter::new();
                if !time_delta {
                    for bit in code_for(part0, -1) {
                        writer.write_bool(bit);
                    }
                }
                for bit in code_for(table, -1) {
                    writer.write_bool(bit);
                }
                let bytes = writer.finish();
                let mut reader = BitReader::new(&bytes);
                assert_eq!(
                    huffman_1d_vector(
                        &mut reader,
                        kind,
                        time_delta,
                        if time_delta { 1 } else { 2 },
                    )
                    .unwrap(),
                    vec![0; if time_delta { 1 } else { 2 }]
                );
            }
        }

        let mut writer = BitWriter::new();
        for bit in code_for(&CLD_PART0, -1) {
            writer.write_bool(bit);
        }
        for bit in code_for(&CLD_1D_FREQ, -2) {
            writer.write_bool(bit);
        }
        writer.write_bool(true); // negative sign
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        assert_eq!(
            huffman_1d_vector(&mut reader, MpsParameterKind::Cld, false, 2).unwrap(),
            [0, -1]
        );
    }

    #[test]
    fn restores_2d_symmetry_signs_and_order() {
        let mut reader = BitReader::new(&[0]);
        assert_eq!(
            restore_2d_symmetry(&mut reader, MpsParameterKind::Cld, 3, [3, 3]).unwrap(),
            [1, 0]
        );

        let mut reader = BitReader::new(&[0xc0]);
        assert_eq!(
            restore_2d_symmetry(&mut reader, MpsParameterKind::Cld, 3, [1, 1]).unwrap(),
            [0, -2]
        );

        let mut reader = BitReader::new(&[0x80]);
        assert_eq!(
            restore_2d_symmetry(&mut reader, MpsParameterKind::Ipd, 3, [1, 1]).unwrap(),
            [0, 2]
        );
    }

    #[test]
    fn decodes_2d_escape_and_time_paired_parameters() {
        let (lav_code, lav, table) = (0..4)
            .map(|lav_code| {
                let lav = 2 * lav_code + 1;
                (
                    lav_code,
                    lav,
                    table_2d(MpsParameterKind::Icc, false, false, lav),
                )
            })
            .find(|(_, _, table)| table.iter().flatten().any(|&node| node == 0))
            .expect("an ICC 2D ROM table contains PCM escape leaves");
        let mut writer = BitWriter::new();
        for bit in code_for(&LAV_TABLE, -(lav_code as i16 + 1)) {
            writer.write_bool(bit);
        }
        for bit in code_for(table, 0) {
            writer.write_bool(bit);
        }
        writer.write(0, bit_width((2 * lav + 1).pow(2)));
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        assert_eq!(
            huffman_2d_pairs(&mut reader, MpsParameterKind::Icc, false, false, 1, 0).unwrap(),
            (Vec::new(), vec![[-(lav as i8), -(lav as i8)]])
        );

        for kind in [
            MpsParameterKind::Cld,
            MpsParameterKind::Icc,
            MpsParameterKind::Ipd,
        ] {
            let part0: &[[i16; 2]] = match kind {
                MpsParameterKind::Cld => &CLD_PART0,
                MpsParameterKind::Icc => &ICC_PART0,
                MpsParameterKind::Ipd => &IPD_PART0,
            };
            let lav = if kind == MpsParameterKind::Cld { 3 } else { 1 };
            let table = table_2d(kind, false, true, lav);
            let mut writer = BitWriter::new();
            writer.write_bool(true); // 2D coding
            writer.write_bool(true); // pair the two parameter sets in time
            for bit in code_for(&LAV_TABLE, -1) {
                writer.write_bool(bit);
            }
            for _ in 0..2 {
                for bit in code_for(part0, -1) {
                    writer.write_bool(bit);
                }
            }
            for bit in code_for(table, -1) {
                writer.write_bool(bit);
            }
            let bytes = writer.finish();
            let mut reader = BitReader::new(&bytes);
            assert_eq!(
                decode_huffman_differences(&mut reader, kind, true, 2, false, false).unwrap(),
                (vec![0, 0], Some(vec![0, 0]))
            );

            let table = table_2d(kind, true, true, lav);
            let mut writer = BitWriter::new();
            writer.write_bool(true); // 2D coding
            writer.write_bool(true); // pair two time-delta parameter sets
            for bit in code_for(&LAV_TABLE, -1) {
                writer.write_bool(bit);
            }
            for _ in 0..2 {
                for bit in code_for(table, -1) {
                    writer.write_bool(bit);
                }
            }
            let bytes = writer.finish();
            let mut reader = BitReader::new(&bytes);
            assert_eq!(
                decode_huffman_differences(&mut reader, kind, true, 2, true, true).unwrap(),
                (vec![0, 0], Some(vec![0, 0]))
            );
        }
    }

    #[test]
    fn decodes_odd_2d_frequency_vectors_for_cld_and_ipd() {
        for kind in [MpsParameterKind::Cld, MpsParameterKind::Ipd] {
            let part0: &[[i16; 2]] = if kind == MpsParameterKind::Cld {
                &CLD_PART0
            } else {
                &IPD_PART0
            };
            let (lav, time_table): (usize, &[[i16; 2]]) = if kind == MpsParameterKind::Cld {
                (3, &CLD_1D_TIME)
            } else {
                (7, &IPD_1D_TIME)
            };
            let mut writer = BitWriter::new();
            writer.write_bool(true); // 2D frequency coding
            for bit in code_for(&LAV_TABLE, -1) {
                writer.write_bool(bit);
            }
            for bit in code_for(part0, -1) {
                writer.write_bool(bit);
            }
            for bit in code_for(table_2d(kind, false, false, lav), -1) {
                writer.write_bool(bit);
            }
            for bit in code_for(time_table, -1) {
                writer.write_bool(bit);
            }
            let bytes = writer.finish();
            let mut reader = BitReader::new(&bytes);
            assert_eq!(
                decode_huffman_differences(&mut reader, kind, false, 4, false, false).unwrap(),
                (vec![0; 4], None)
            );
        }
    }

    #[test]
    fn paired_huffman_restores_forward_backward_and_ipd_lsb_modes() {
        let mut writer = BitWriter::new();
        writer.write_bool(true); // first set uses a time delta
        writer.write_bool(false); // 1D coding
        for bit in code_for(&CLD_1D_TIME, -1) {
            writer.write_bool(bit);
        }
        for bit in code_for(&CLD_PART0, -1) {
            writer.write_bool(bit);
        }
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        assert_eq!(
            decode_huffman_1d(
                &mut reader,
                MpsParameterKind::Cld,
                true,
                1,
                &[0],
                0,
                false,
                false,
            )
            .unwrap(),
            [vec![0], vec![0]]
        );

        let mut writer = BitWriter::new();
        writer.write_bool(false); // first set uses frequency deltas
        writer.write_bool(true); // second set uses a time delta
        writer.write_bool(false); // 1D coding
        for bit in code_for(&CLD_PART0, -1) {
            writer.write_bool(bit);
        }
        for bit in code_for(&CLD_1D_TIME, -1) {
            writer.write_bool(bit);
        }
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        assert_eq!(
            decode_huffman_1d(
                &mut reader,
                MpsParameterKind::Cld,
                true,
                1,
                &[0],
                0,
                false,
                false,
            )
            .unwrap(),
            [vec![0], vec![0]]
        );

        let mut writer = BitWriter::new();
        writer.write_bool(true); // first time delta
        writer.write_bool(false); // second frequency delta
        writer.write_bool(false); // 1D coding
        for bit in code_for(&CLD_1D_TIME, -1) {
            writer.write_bool(bit);
        }
        for bit in code_for(&CLD_PART0, -1) {
            writer.write_bool(bit);
        }
        writer.write_bool(false); // decode backwards from history
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        assert_eq!(
            decode_huffman_1d(
                &mut reader,
                MpsParameterKind::Cld,
                true,
                1,
                &[0],
                0,
                false,
                true,
            )
            .unwrap(),
            [vec![0], vec![0]]
        );

        let mut writer = BitWriter::new();
        writer.write_bool(true); // first time delta
        writer.write_bool(true); // second time delta
        writer.write_bool(false); // 1D coding
        for _ in 0..2 {
            for bit in code_for(&IPD_1D_TIME, -1) {
                writer.write_bool(bit);
            }
        }
        writer.write_bool(false); // first attached LSB
        writer.write_bool(true); // second attached LSB
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        assert_eq!(
            decode_huffman_1d(
                &mut reader,
                MpsParameterKind::Ipd,
                true,
                1,
                &[2],
                0,
                true,
                true,
            )
            .unwrap(),
            [vec![2], vec![3]]
        );
    }

    #[test]
    fn all_2d_rom_layouts_are_addressable() {
        for &kind in &[
            MpsParameterKind::Cld,
            MpsParameterKind::Icc,
            MpsParameterKind::Ipd,
        ] {
            let lavs: &[usize] = if kind == MpsParameterKind::Cld {
                &[3, 5, 7, 9]
            } else {
                &[1, 3, 5, 7]
            };
            for time_delta in [false, true] {
                for time_pair in [false, true] {
                    for &lav in lavs {
                        let table = table_2d(kind, time_delta, time_pair, lav);
                        assert!(!table.is_empty());
                        assert!(table.iter().any(|row| row[0] < 0 || row[1] < 0));
                    }
                }
            }
        }
    }

    #[test]
    fn decodes_tsd_combinatorial_slot_and_phase() {
        let mut writer = BitWriter::new();
        writer.write(1, 1); // enabled
        writer.write(0, 4); // one transient
        writer.write(31, 5); // last of 32 possible one-slot combinations
        writer.write(6, 3); // phase
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        let shaping = parse_transient_shaping(&mut reader, 32).unwrap();
        assert!(shaping.enabled);
        assert_eq!(
            shaping
                .phases
                .iter()
                .filter(|phase| phase.is_some())
                .count(),
            1
        );
        assert_eq!(shaping.phases[31], Some(6));

        let mut writer = BitWriter::new();
        writer.write(1, 1); // enabled
        writer.write(1, 4); // two transients
        writer.write(495, bit_width(binomial(32, 2) as usize));
        writer.write(3, 3);
        writer.write(5, 3);
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        let shaping = parse_transient_shaping(&mut reader, 32).unwrap();
        assert_eq!(shaping.phases[30], Some(3));
        assert_eq!(shaping.phases[31], Some(5));
        assert_eq!(
            shaping
                .phases
                .iter()
                .filter(|phase| phase.is_some())
                .count(),
            2
        );

        let mut writer = BitWriter::new();
        writer.write(1, 1); // enabled
        writer.write(15, 4); // maximum: 16 transient slots
        writer.write(0, bit_width(binomial(32, 16) as usize));
        for phase in 0..16 {
            writer.write(phase % 8, 3);
        }
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        let shaping = parse_transient_shaping(&mut reader, 32).unwrap();
        assert_eq!(
            shaping.phases,
            (0..16)
                .map(|phase| Some((phase % 8) as u8))
                .chain((16..32).map(|_| None))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn parameter_modes_cover_leading_interpolation_and_reject_overflowing_pair() {
        let mut writer = BitWriter::new();
        writer.write(2, 2); // interpolate from history to the next set
        writer.write(0, 2); // default next set
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        let mut history = ParameterHistory::new(1, 4);
        assert_eq!(
            parse_parameter_data(&mut reader, 2, false, MpsParameterKind::Cld, &mut history,)
                .unwrap(),
            [vec![0], vec![0]]
        );

        let mut writer = BitWriter::new();
        writer.write(3, 2); // new
        writer.write_bool(true); // paired, but only one parameter set remains
        writer.write_bool(false); // fine
        writer.write(0, 2); // stride 1
        writer.write_bool(true); // PCM
        writer.write(15, 5);
        writer.write(15, 5);
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        let mut history = ParameterHistory::new(1, 0);
        assert_eq!(
            parse_parameter_data(&mut reader, 1, false, MpsParameterKind::Cld, &mut history,),
            Err(MpsError::InvalidDataMode)
        );
    }

    #[test]
    fn disabled_tsd_has_no_transient_slots() {
        let mut reader = BitReader::new(&[0]);
        let shaping = parse_transient_shaping(&mut reader, 64).unwrap();
        assert!(!shaping.enabled);
        assert!(shaping.phases.iter().all(Option::is_none));
    }

    #[test]
    fn parses_stp_channel_enable_flags() {
        let mut writer = BitWriter::new();
        writer.write(1, 1);
        writer.write(1, 1);
        writer.write(0, 1);
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        let (stp, ges) = parse_stp_ges(&mut reader, 1, 32).unwrap();
        assert_eq!(stp, [true, false]);
        assert_eq!(ges, [None, None]);
    }

    #[test]
    fn decodes_ges_reshape_run_length_huffman() {
        let zero_one_slot = code_for(&RESHAPE_2D, -1);
        let mut writer = BitWriter::new();
        for _ in 0..4 {
            for &bit in &zero_one_slot {
                writer.write(bit as u32, 1);
            }
        }
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        assert_eq!(decode_reshape_envelope(&mut reader, 4).unwrap(), [0; 4]);
    }

    #[test]
    fn usac_decorrelator_applies_reverb_band_delay() {
        let mut decorrelator = MpsUsacDecorrelator::new(0).unwrap();
        let mut input = vec![(0.0, 0.0); 71];
        input[0] = (1.0, -0.5);
        assert_eq!(decorrelator.process_slot(&input).unwrap()[0], (0.0, 0.0));
        for _ in 1..11 {
            assert_eq!(
                decorrelator.process_slot(&vec![(0.0, 0.0); 71]).unwrap()[0],
                (0.0, 0.0)
            );
        }
        let delayed = decorrelator.process_slot(&vec![(0.0, 0.0); 71]).unwrap()[0];
        assert!(delayed.0 != 0.0 && delayed.1 != 0.0);
        assert!(delayed.0.is_finite() && delayed.1.is_finite());
    }

    #[test]
    fn decorrelator_configs_map_empty_reverb_bands() {
        assert_eq!(reverb_band(0, 7), 0);
        assert_eq!(reverb_band(0, 8), 1);
        assert_eq!(reverb_band(1, 56), 2);
        assert_eq!(reverb_band(2, 0), 1);
    }

    #[test]
    fn fdk_parameter_maps_cover_all_hybrid_bands() {
        for bands in [4, 5, 7, 10, 14, 20, 28] {
            let map = parameter_band_map(bands).unwrap();
            assert_eq!(map.len(), 71);
            assert!(map.iter().all(|&band| band < bands));
            assert_eq!(*map.last().unwrap(), bands - 1);
        }
    }

    #[test]
    fn hybrid_renderer_preserves_centered_direct_energy() {
        let frame = Mps212Frame {
            independent: true,
            transient_shaping: None,
            stp_enabled: vec![false; 2],
            ges_envelopes: vec![None; 2],
            parameter_sets: vec![MpsParameterSet {
                slot: 1,
                cld: vec![0; 28],
                icc: vec![0; 28],
                ipd: None,
                smoothing: MpsSmoothing::default(),
            }],
        };
        let mut direct = vec![vec![(0.0, 0.0); 71]; 2];
        direct[0][20] = (0.75, -0.25);
        let mut renderer = Mps212HybridRenderer::new(28, 0).unwrap();
        let (left, right) = renderer.process_frame(&direct, &frame).unwrap();
        let power = |value: (f32, f32)| value.0 * value.0 + value.1 * value.1;
        assert!((power(left[0][20]) + power(right[0][20]) - power(direct[0][20])).abs() < 1e-6);
    }

    #[test]
    fn qmf_processor_produces_dual_channel_pcm() {
        let frame = Mps212Frame {
            independent: true,
            transient_shaping: None,
            stp_enabled: vec![false; 2],
            ges_envelopes: vec![None; 2],
            parameter_sets: vec![MpsParameterSet {
                slot: 31,
                cld: vec![0; 28],
                icc: vec![0; 28],
                ipd: None,
                smoothing: MpsSmoothing::default(),
            }],
        };
        let mut slots = vec![
            QmfSlot {
                real: vec![0.0; 64],
                imaginary: vec![0.0; 64]
            };
            32
        ];
        slots[0].real[10] = 1.0;
        let mut processor = Mps212QmfProcessor::new(28, 0).unwrap();
        let (left, right) = processor.process_qmf(&slots, &frame).unwrap();
        assert_eq!(left.len(), right.len());
        assert!(!left.is_empty());
        assert!(left.iter().chain(&right).all(|sample| sample.is_finite()));
    }

    fn centered_frame(slot: usize, bands: usize) -> Mps212Frame {
        Mps212Frame {
            independent: true,
            transient_shaping: None,
            stp_enabled: vec![false; 2],
            ges_envelopes: vec![None; 2],
            parameter_sets: vec![MpsParameterSet {
                slot,
                cld: vec![0; bands],
                icc: vec![0; bands],
                ipd: None,
                smoothing: MpsSmoothing::default(),
            }],
        }
    }

    #[test]
    fn decorrelator_and_parameter_map_reject_invalid_layouts() {
        assert!(matches!(
            MpsUsacDecorrelator::new(3),
            Err(MpsError::InvalidDataMode)
        ));
        let mut decorrelator = MpsUsacDecorrelator::new(0).unwrap();
        assert_eq!(
            decorrelator.process_slot(&[(0.0, 0.0); 70]),
            Err(MpsError::InvalidParameterSlot)
        );
        assert_eq!(parameter_band_map(6), Err(MpsError::InvalidParameterSets));
        assert!(matches!(
            Mps212HybridRenderer::new(6, 0),
            Err(MpsError::InvalidParameterSets)
        ));
    }

    #[test]
    fn hybrid_renderer_validates_slots_parameters_and_residuals() {
        let mut renderer = Mps212HybridRenderer::new(4, 0).unwrap();
        assert_eq!(
            renderer.process_frame(&[vec![(0.0, 0.0); 70]], &centered_frame(0, 4)),
            Err(MpsError::InvalidParameterSlot)
        );
        let empty = Mps212Frame {
            parameter_sets: Vec::new(),
            ..centered_frame(0, 4)
        };
        assert_eq!(
            renderer.process_frame(&[vec![(0.0, 0.0); 71]], &empty),
            Err(MpsError::InvalidParameterSlot)
        );
        let direct = vec![vec![(0.0, 0.0); 71]; 2];
        assert_eq!(
            renderer.process_frame_with_residual(
                &direct,
                &[vec![(0.0, 0.0); 71]],
                1,
                &centered_frame(1, 4)
            ),
            Err(MpsError::InvalidParameterSlot)
        );
        assert_eq!(
            renderer.process_frame(&direct, &centered_frame(1, 5)),
            Err(MpsError::InvalidParameterSets)
        );
        assert_eq!(
            renderer.process_frame(&direct, &centered_frame(0, 4)),
            Err(MpsError::InvalidParameterSlot)
        );
        assert_eq!(
            renderer.process_frame(&direct, &centered_frame(2, 4)),
            Err(MpsError::InvalidParameterSlot)
        );
    }

    #[test]
    fn hybrid_renderer_processes_residual_and_rejects_oversized_ipd() {
        let direct = vec![vec![(0.25, -0.1); 71]; 2];
        let residual = vec![vec![(0.05, 0.02); 71]; 2];
        let mut frame = centered_frame(1, 4);
        frame.parameter_sets[0].ipd = Some(vec![2; 4]);
        let mut renderer = Mps212HybridRenderer::new(4, 1).unwrap();
        let (left, right) = renderer
            .process_frame_with_residual(&direct, &residual, 2, &frame)
            .unwrap();
        assert_eq!(left.len(), 2);
        assert!(left
            .iter()
            .flatten()
            .chain(right.iter().flatten())
            .all(|&(real, imaginary)| real.is_finite() && imaginary.is_finite()));

        frame.parameter_sets[0].ipd = Some(vec![0; 5]);
        assert_eq!(
            renderer.process_frame(&direct, &frame),
            Err(MpsError::InvalidParameterSets)
        );
    }

    #[test]
    fn qmf_processor_validates_subbands_and_residual_slot_count() {
        let frame = centered_frame(0, 4);
        let invalid = [QmfSlot {
            real: vec![0.0; 63],
            imaginary: vec![0.0; 64],
        }];
        let mut processor = Mps212QmfProcessor::new(4, 0).unwrap();
        assert!(matches!(
            processor.process_qmf(&invalid, &frame),
            Err(MpsError::Qmf(QmfError::InvalidSubbandCount { .. }))
        ));
        let valid = [QmfSlot {
            real: vec![0.0; 64],
            imaginary: vec![0.0; 64],
        }];
        assert_eq!(
            processor.process_qmf_with_residual(&valid, &[], 1, &frame),
            Err(MpsError::InvalidParameterSlot)
        );
        assert!(matches!(
            processor.process_qmf_with_residual(&valid, &invalid, 1, &frame),
            Err(MpsError::Qmf(QmfError::InvalidSubbandCount { .. }))
        ));
    }

    #[test]
    fn qmf_processor_renders_a_complete_residual_frame() {
        let frame = centered_frame(31, 28);
        let mut downmix = vec![
            QmfSlot {
                real: vec![0.0; 64],
                imaginary: vec![0.0; 64],
            };
            32
        ];
        let mut residual = downmix.clone();
        downmix[0].real[8] = 0.5;
        residual[0].imaginary[8] = 0.25;
        let mut processor = Mps212QmfProcessor::new(28, 2).unwrap();
        let (left, right) = processor
            .process_qmf_with_residual(&downmix, &residual, 10, &frame)
            .unwrap();
        assert_eq!(left.len(), right.len());
        assert!(!left.is_empty());
        assert!(left.iter().chain(&right).all(|sample| sample.is_finite()));
    }

    #[test]
    fn parameter_history_interpolates_keep_default_and_quantization_modes() {
        let mut history = ParameterHistory {
            values: vec![2, 4],
            coarse: false,
        };
        let mut writer = BitWriter::new();
        writer.write(1, 2); // keep history
        writer.write(2, 2); // interpolate
        writer.write(0, 2); // reset to default
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        assert_eq!(
            parse_parameter_data(&mut reader, 3, false, MpsParameterKind::Cld, &mut history,)
                .unwrap(),
            [vec![2, 4], vec![1, 2], vec![0, 0]]
        );

        for (bits, count, independent) in [(0b01_00, 2, true), (0b00_10, 2, false)] {
            let bytes = [(bits << 4) as u8];
            let mut reader = BitReader::new(&bytes);
            let mut history = ParameterHistory::new(1, 0);
            assert_eq!(
                parse_parameter_data(
                    &mut reader,
                    count,
                    independent,
                    MpsParameterKind::Icc,
                    &mut history,
                ),
                Err(MpsError::InvalidDataMode)
            );
        }

        let mut cld = ParameterHistory {
            values: vec![-15, 15, 3],
            coarse: false,
        };
        convert_history_quantization(&mut cld, MpsParameterKind::Cld, true);
        assert_eq!(cld.values, [-7, 7, 1]);
        cld.coarse = true;
        convert_history_quantization(&mut cld, MpsParameterKind::Cld, false);
        assert_eq!(cld.values, [-15, 15, 2]);

        let mut icc = ParameterHistory {
            values: vec![-3, 3],
            coarse: false,
        };
        convert_history_quantization(&mut icc, MpsParameterKind::Icc, true);
        assert_eq!(icc.values, [-2, 1]);
        convert_history_quantization(&mut icc, MpsParameterKind::Icc, true);
    }

    #[test]
    fn pcm_tables_wide_reads_and_combinatorics_cover_boundaries() {
        assert_eq!(bit_width(0), 0);
        assert_eq!(bit_width(1), 0);
        assert_eq!(bit_width(2), 1);
        assert_eq!(binomial(3, 4), 0);
        assert_eq!(binomial(5, 2), 10);

        let mut wide = BitReader::new(&[0x12, 0x34, 0x56, 0x78, 0x9a]);
        assert_eq!(read_wide(&mut wide, 40).unwrap(), 0x1234_5678_9a);

        for (levels, count) in [
            (3, 5),
            (7, 6),
            (11, 2),
            (13, 4),
            (19, 4),
            (51, 4),
            (25, 3),
            (4, 1),
            (8, 1),
            (15, 1),
            (16, 1),
            (26, 1),
            (31, 1),
        ] {
            let mut reader = BitReader::new(&[0; 16]);
            let decoded = pcm_decode(&mut reader, count, levels, 0).unwrap();
            assert_eq!(decoded, vec![0; count]);
        }
        let mut reader = BitReader::new(&[0]);
        assert_eq!(
            pcm_decode(&mut reader, 1, 5, 0),
            Err(MpsError::InvalidDataMode)
        );

        let mut reader = BitReader::new(&[0]);
        assert_eq!(
            parse_transient_shaping(&mut reader, 16),
            Err(MpsError::InvalidParameterSlot)
        );
    }

    #[test]
    fn converts_bit_and_qmf_errors() {
        assert_eq!(
            MpsError::from(BitError::UnexpectedEof {
                needed_bits: 1,
                remaining_bits: 0,
            }),
            MpsError::Bit(BitError::UnexpectedEof {
                needed_bits: 1,
                remaining_bits: 0,
            })
        );
        let qmf = QmfError::InvalidSubbandCount {
            expected: 64,
            actual: 63,
        };
        assert_eq!(MpsError::from(qmf.clone()), MpsError::Qmf(qmf));
    }

    #[test]
    fn parses_high_rate_explicit_slots_ipd_and_smoothing() {
        let mut writer = BitWriter::new();
        writer.write_bool(true); // explicit framing
        writer.write(1, 3); // two parameter sets
        writer.write(3, 4);
        writer.write(15, 4); // increasing explicit slots
        for _ in 0..2 {
            writer.write(0, 2); // CLD default
        }
        for _ in 0..2 {
            writer.write(0, 2); // ICC default
        }
        writer.write_bool(true); // phase coding present
        writer.write_bool(false); // no OPD smoothing
        for _ in 0..2 {
            writer.write(0, 2); // IPD default
        }
        writer.write(0, 2); // smoothing mode 0
        writer.write(2, 2); // smoothing mode 2
        writer.write(1, 2); // smoothing time
        let bit_len = writer.bits_written();
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        let mut decoder =
            Mps212FrameDecoder::new(16, 3, 3, true, true).with_temporal_shape_config(0);
        let frame = decoder.parse(&mut reader, true).unwrap();
        assert!(frame.independent);
        assert_eq!(frame.parameter_sets.len(), 2);
        assert_eq!(frame.parameter_sets[0].slot, 3);
        assert_eq!(frame.parameter_sets[1].slot, 15);
        assert_eq!(frame.parameter_sets[0].ipd, Some(vec![0; 3]));
        assert_eq!(frame.parameter_sets[1].smoothing.mode, 2);
        assert_eq!(frame.parameter_sets[1].smoothing.time, Some(1));
        for truncated_len in 0..bit_len {
            let mut reader = BitReader::with_bit_len(&bytes, truncated_len).unwrap();
            assert!(Mps212FrameDecoder::new(16, 3, 3, true, true)
                .with_temporal_shape_config(0)
                .parse(&mut reader, true)
                .is_err());
        }

        let mut writer = BitWriter::new();
        writer.write(0, 2); // default CLD
        writer.write(0, 2); // default ICC
        let bits = writer.bits_written();
        let bytes = writer.finish();
        let mut decoder =
            Mps212FrameDecoder::new(16, 3, 0, false, false).with_temporal_shape_config(3);
        assert!(decoder
            .parse(&mut BitReader::with_bit_len(&bytes, bits).unwrap(), true,)
            .is_err());
    }

    #[test]
    fn high_rate_framing_rejects_count_and_slot_order() {
        let mut reader = BitReader::new(&[0x70]); // implicit framing, count code 7
        let mut decoder = Mps212FrameDecoder::new(16, 1, 0, true, false);
        assert_eq!(
            decoder.parse(&mut reader, true),
            Err(MpsError::InvalidParameterSets)
        );

        let mut writer = BitWriter::new();
        writer.write_bool(true);
        writer.write(1, 3); // two sets
        writer.write(5, 4);
        writer.write(5, 4); // duplicate slot
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        let mut decoder = Mps212FrameDecoder::new(16, 1, 0, true, false);
        assert_eq!(
            decoder.parse(&mut reader, true),
            Err(MpsError::InvalidParameterSlot)
        );
    }

    #[test]
    fn parses_ges_channel_flags_and_rejects_bad_reshape_runs() {
        let one_zero = code_for(&RESHAPE_2D, -1);
        let mut writer = BitWriter::new();
        writer.write_bool(true); // shaping data present
        writer.write_bool(true); // left GES
        writer.write_bool(false); // right GES
        for _ in 0..4 {
            for &bit in &one_zero {
                writer.write_bool(bit);
            }
        }
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        let (stp, ges) = parse_stp_ges(&mut reader, 2, 4).unwrap();
        assert_eq!(stp, [false, false]);
        assert_eq!(ges, [Some(vec![0; 4]), None]);

        let target = RESHAPE_2D
            .iter()
            .flatten()
            .copied()
            .find(|&node| node < 0 && ((-(node + 1)) & 15) != 0)
            .expect("reshape ROM contains a multi-slot run");
        let code = code_for(&RESHAPE_2D, target);
        let mut writer = BitWriter::new();
        for bit in code {
            writer.write_bool(bit);
        }
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        assert_eq!(
            decode_reshape_envelope(&mut reader, 1),
            Err(MpsError::InvalidHuffmanCodeword)
        );
    }

    #[test]
    fn pcm_parameter_decoder_covers_all_kinds_quantizers_and_pairing() {
        for kind in [
            MpsParameterKind::Cld,
            MpsParameterKind::Icc,
            MpsParameterKind::Ipd,
        ] {
            for coarse in [false, true] {
                for pair in [false, true] {
                    let mut writer = BitWriter::new();
                    writer.write_bool(true); // PCM
                    writer.write(0, 32);
                    let bytes = writer.finish();
                    let mut reader = BitReader::new(&bytes);
                    let decoded =
                        decode_pcm_or_huffman(&mut reader, kind, coarse, pair, 2, &[0; 2], false)
                            .unwrap();
                    assert_eq!(decoded.len(), if pair { 2 } else { 1 });
                    assert!(decoded.iter().all(|set| set.len() == 2));
                }
            }
        }
    }
}
