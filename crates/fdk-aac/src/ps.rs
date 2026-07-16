//! MPEG-4 Parametric Stereo payload parsing and differential reconstruction.

use std::{fmt, sync::LazyLock};

use crate::bits::{BitError, BitReader};
use crate::ld_sbr_qmf::{LdSbrQmfSynthesis, QmfError, QmfSlot};

const ROM: &str = include_str!(concat!(
    env!("FDK_AAC_UPSTREAM_DIR"),
    "/libSBRdec/src/sbr_rom.cpp"
));
const BINS: [usize; 3] = [10, 20, 34];
const FIX_ENVELOPES: [usize; 4] = [0, 1, 2, 4];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PsHuffmanBook {
    IidTime,
    IidFrequency,
    IidFineTime,
    IidFineFrequency,
    IccTime,
    IccFrequency,
}

impl PsHuffmanBook {
    fn table(self) -> &'static [[i8; 2]] {
        match self {
            Self::IidTime => &IID_TIME,
            Self::IidFrequency => &IID_FREQUENCY,
            Self::IidFineTime => &IID_FINE_TIME,
            Self::IidFineFrequency => &IID_FINE_FREQUENCY,
            Self::IccTime => &ICC_TIME,
            Self::IccFrequency => &ICC_FREQUENCY,
        }
    }
}

fn parse_table(name: &str) -> Vec<[i8; 2]> {
    let declaration = format!("const SCHAR {name}");
    let start = ROM.find(&declaration).unwrap();
    let body_start = ROM[start..].find('{').unwrap() + start + 1;
    let body_end = ROM[body_start..].find("};").unwrap() + body_start;
    ROM[body_start..body_end]
        .split('{')
        .skip(1)
        .filter_map(|part| {
            let end = part.find('}')?;
            let values = part[..end]
                .split(',')
                .map(|value| value.trim().parse::<i8>().ok())
                .collect::<Option<Vec<_>>>()?;
            (values.len() == 2).then(|| [values[0], values[1]])
        })
        .collect()
}

macro_rules! table {
    ($name:ident, $source:literal) => {
        static $name: LazyLock<Vec<[i8; 2]>> = LazyLock::new(|| parse_table($source));
    };
}
table!(IID_TIME, "aBookPsIidTimeDecode");
table!(IID_FREQUENCY, "aBookPsIidFreqDecode");
table!(IID_FINE_TIME, "aBookPsIidFineTimeDecode");
table!(IID_FINE_FREQUENCY, "aBookPsIidFineFreqDecode");
table!(ICC_TIME, "aBookPsIccTimeDecode");
table!(ICC_FREQUENCY, "aBookPsIccFreqDecode");

pub fn decode_ps_huffman(reader: &mut BitReader<'_>, book: PsHuffmanBook) -> Result<i8, PsError> {
    let table = book.table();
    let mut index = 0i8;
    while index >= 0 {
        let row = table
            .get(index as usize)
            .ok_or(PsError::InvalidHuffmanCodeword)?;
        index = row[reader.read_bool()? as usize];
    }
    Ok(index + 64)
}

pub fn encode_ps_huffman(book: PsHuffmanBook, symbol: i8) -> Option<Vec<bool>> {
    encode_ps_huffman_from(book.table(), symbol)
}

fn encode_ps_huffman_from(table: &[[i8; 2]], symbol: i8) -> Option<Vec<bool>> {
    fn find(table: &[[i8; 2]], node: i8, target: i8, bits: &mut Vec<bool>) -> bool {
        if node < 0 {
            return node + 64 == target;
        }
        let Some(row) = table.get(node as usize) else {
            return false;
        };
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
    find(table, 0, symbol, &mut bits).then_some(bits)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PsHeader {
    pub iid_enabled: bool,
    pub iid_mode: u8,
    pub icc_enabled: bool,
    pub icc_mode: u8,
    pub extension_enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PsFrame {
    pub header_present: bool,
    pub header: PsHeader,
    pub variable_borders: bool,
    pub borders: Vec<u8>,
    pub iid_time_domain: Vec<bool>,
    pub icc_time_domain: Vec<bool>,
    pub iid: Vec<Vec<i8>>,
    pub icc: Vec<Vec<i8>>,
    pub iid_mapped_20: Vec<Vec<i8>>,
    pub icc_mapped_20: Vec<Vec<i8>>,
    pub extension_data: Vec<u8>,
    pub bits_read: usize,
}

#[derive(Debug, Clone, Default)]
pub struct PsParser {
    header: Option<PsHeader>,
    previous_iid: Vec<i8>,
    previous_icc: Vec<i8>,
}

impl PsParser {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear_history(&mut self) {
        self.previous_iid.clear();
        self.previous_icc.clear();
    }

    pub fn parse(
        &mut self,
        reader: &mut BitReader<'_>,
        time_slots: u8,
    ) -> Result<PsFrame, PsError> {
        if !matches!(time_slots, 30 | 32) {
            return Err(PsError::UnsupportedTimeSlots(time_slots));
        }
        let start = reader.bits_read();
        let header_present = reader.read_bool()?;
        let header = if header_present {
            let iid_enabled = reader.read_bool()?;
            let iid_mode = if iid_enabled { reader.read_u8(3)? } else { 0 };
            let icc_enabled = reader.read_bool()?;
            let icc_mode = if icc_enabled { reader.read_u8(3)? } else { 0 };
            let extension_enabled = reader.read_bool()?;
            PsHeader {
                iid_enabled,
                iid_mode,
                icc_enabled,
                icc_mode,
                extension_enabled,
            }
        } else {
            self.header.ok_or(PsError::MissingInitialHeader)?
        };
        if header.iid_mode > 5 || header.icc_mode > 5 {
            return Err(PsError::UnsupportedMode {
                iid: header.iid_mode,
                icc: header.icc_mode,
            });
        }
        let variable_borders = reader.read_bool()?;
        let mut envelope_count = if variable_borders {
            1 + reader.read_u8(2)? as usize
        } else {
            FIX_ENVELOPES[reader.read_u8(2)? as usize]
        };
        let mut borders = vec![0];
        if variable_borders {
            for _ in 0..envelope_count {
                borders.push(reader.read_u8(5)? + 1);
            }
        } else if envelope_count != 0 {
            borders.extend(
                (1..envelope_count).map(|env| (env * time_slots as usize / envelope_count) as u8),
            );
            borders.push(time_slots);
        }
        let iid_resolution = mode_resolution(header.iid_mode);
        let icc_resolution = mode_resolution(header.icc_mode);
        let fine_iid = header.iid_mode > 2;
        let (mut iid_time_domain, mut iid) = read_parameters(
            reader,
            header.iid_enabled,
            envelope_count,
            iid_resolution,
            |time| match (fine_iid, time) {
                (false, false) => PsHuffmanBook::IidFrequency,
                (false, true) => PsHuffmanBook::IidTime,
                (true, false) => PsHuffmanBook::IidFineFrequency,
                (true, true) => PsHuffmanBook::IidFineTime,
            },
        )?;
        let (mut icc_time_domain, mut icc) = read_parameters(
            reader,
            header.icc_enabled,
            envelope_count,
            icc_resolution,
            |time| {
                if time {
                    PsHuffmanBook::IccTime
                } else {
                    PsHuffmanBook::IccFrequency
                }
            },
        )?;
        let extension_data = if header.extension_enabled {
            let mut count = reader.read_u8(4)? as usize;
            if count == 15 {
                count += reader.read_u8(8)? as usize;
            }
            (0..count)
                .map(|_| reader.read_u8(8).map_err(PsError::from))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            Vec::new()
        };
        if envelope_count == 0 {
            envelope_count = 1;
            iid_time_domain.push(true);
            icc_time_domain.push(true);
            iid.push(Vec::new());
            icc.push(Vec::new());
            borders = vec![0, time_slots];
        } else if variable_borders && *borders.last().unwrap() < time_slots {
            envelope_count += 1;
            borders.push(time_slots);
            iid_time_domain.push(false);
            icc_time_domain.push(false);
            iid.push(iid.last().cloned().unwrap_or_default());
            icc.push(icc.last().cloned().unwrap_or_default());
        }
        normalize_borders(&mut borders, envelope_count, time_slots);
        reconstruct(
            &mut iid,
            &iid_time_domain,
            &self.previous_iid,
            iid_resolution,
            if fine_iid { -15 } else { -7 },
            if fine_iid { 15 } else { 7 },
            header.iid_enabled,
        );
        reconstruct(
            &mut icc,
            &icc_time_domain,
            &self.previous_icc,
            icc_resolution,
            0,
            7,
            header.icc_enabled,
        );
        if let Some(last) = iid.last() {
            self.previous_iid = expand_resolution(last, iid_resolution);
        }
        if let Some(last) = icc.last() {
            self.previous_icc = expand_resolution(last, icc_resolution);
        }
        self.header = Some(header);
        let iid_mapped_20 = iid
            .iter()
            .map(|values| map_to_20(values, iid_resolution))
            .collect();
        let icc_mapped_20 = icc
            .iter()
            .map(|values| map_to_20(values, icc_resolution))
            .collect();
        Ok(PsFrame {
            header_present,
            header,
            variable_borders,
            borders,
            iid_time_domain,
            icc_time_domain,
            iid,
            icc,
            iid_mapped_20,
            icc_mapped_20,
            extension_data,
            bits_read: reader.bits_read() - start,
        })
    }

    /// Parse the first PS element from an SBR `bs_extended_data` byte block.
    /// The two-bit extension id is part of that block; id 2 denotes PS.
    pub fn parse_sbr_extension(
        &mut self,
        data: &[u8],
        time_slots: u8,
    ) -> Result<Option<PsFrame>, PsError> {
        let mut reader = BitReader::new(data);
        while reader.remaining_bits() >= 8 {
            let extension_id = reader.read_u8(2)?;
            if extension_id == 2 {
                return self.parse(&mut reader, time_slots).map(Some);
            }
            // Unknown SBR extension elements do not carry an independently
            // signalled length in baseline syntax, so consume the remainder.
            while reader.remaining_bits() >= 8 {
                reader.read_u8(8)?;
            }
        }
        Ok(None)
    }
}

fn mode_resolution(mode: u8) -> usize {
    (mode % 3) as usize
}

fn read_parameters(
    reader: &mut BitReader<'_>,
    enabled: bool,
    envelopes: usize,
    resolution: usize,
    book: impl Fn(bool) -> PsHuffmanBook,
) -> Result<(Vec<bool>, Vec<Vec<i8>>), PsError> {
    if !enabled {
        return Ok((vec![false; envelopes], vec![Vec::new(); envelopes]));
    }
    let count = BINS[resolution];
    let mut directions = Vec::with_capacity(envelopes);
    let mut values = Vec::with_capacity(envelopes);
    for _ in 0..envelopes {
        let time = reader.read_bool()?;
        directions.push(time);
        values.push(
            (0..count)
                .map(|_| decode_ps_huffman(reader, book(time)))
                .collect::<Result<Vec<_>, _>>()?,
        );
    }
    Ok((directions, values))
}

fn reconstruct(
    frames: &mut [Vec<i8>],
    time_domain: &[bool],
    previous: &[i8],
    resolution: usize,
    min: i8,
    max: i8,
    enabled: bool,
) {
    let stride = if resolution == 0 { 2 } else { 1 };
    for env in 0..frames.len() {
        if !enabled {
            frames[env] = vec![0; BINS[resolution]];
            continue;
        }
        if time_domain[env] {
            for band in 0..frames[env].len() {
                let reference = if env == 0 {
                    previous.get(band * stride).copied().unwrap_or(0)
                } else {
                    frames[env - 1][band]
                };
                frames[env][band] = frames[env][band].saturating_add(reference).clamp(min, max);
            }
        } else {
            let mut prior = 0i8;
            for value in &mut frames[env] {
                *value = value.saturating_add(prior).clamp(min, max);
                prior = *value;
            }
        }
    }
}

fn expand_resolution(values: &[i8], resolution: usize) -> Vec<i8> {
    if resolution == 0 {
        values.iter().flat_map(|&value| [value, value]).collect()
    } else {
        values.to_vec()
    }
}

fn map_to_20(values: &[i8], resolution: usize) -> Vec<i8> {
    if resolution == 0 {
        return expand_resolution(values, resolution);
    }
    if resolution == 1 {
        return values.to_vec();
    }
    let v = values;
    vec![
        (2 * v[0] + v[1]) / 3,
        (v[1] + 2 * v[2]) / 3,
        (2 * v[3] + v[4]) / 3,
        (v[4] + 2 * v[5]) / 3,
        (v[6] + v[7]) / 2,
        (v[8] + v[9]) / 2,
        v[10],
        v[11],
        (v[12] + v[13]) / 2,
        (v[14] + v[15]) / 2,
        v[16],
        v[17],
        v[18],
        v[19],
        (v[20] + v[21]) / 2,
        (v[22] + v[23]) / 2,
        (v[24] + v[25]) / 2,
        (v[26] + v[27]) / 2,
        (v[28] + v[29] + v[30] + v[31]) / 4,
        (v[32] + v[33]) / 2,
    ]
}

fn normalize_borders(borders: &mut [u8], envelopes: usize, time_slots: u8) {
    for env in 1..envelopes {
        let maximum = time_slots - (envelopes - env) as u8;
        borders[env] = borders[env]
            .min(maximum)
            .max(borders[env - 1].saturating_add(1));
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PsError {
    Bit(BitError),
    Qmf(QmfError),
    InvalidHuffmanCodeword,
    MissingInitialHeader,
    UnsupportedMode { iid: u8, icc: u8 },
    UnsupportedTimeSlots(u8),
    QmfSlotLayoutMismatch { expected: usize, actual: usize },
}

impl From<BitError> for PsError {
    fn from(value: BitError) -> Self {
        Self::Bit(value)
    }
}

impl From<QmfError> for PsError {
    fn from(value: QmfError) -> Self {
        Self::Qmf(value)
    }
}

impl fmt::Display for PsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bit(error) => error.fmt(f),
            Self::Qmf(error) => write!(f, "PS QMF error: {error:?}"),
            Self::InvalidHuffmanCodeword => write!(f, "invalid PS Huffman codeword"),
            Self::MissingInitialHeader => write!(f, "PS payload requires an initial header"),
            Self::UnsupportedMode { iid, icc } => {
                write!(f, "unsupported PS IID/ICC modes {iid}/{icc}")
            }
            Self::UnsupportedTimeSlots(value) => {
                write!(f, "unsupported PS time-slot count {value}")
            }
            Self::QmfSlotLayoutMismatch { expected, actual } => {
                write!(f, "PS expected {expected} QMF slots, got {actual}")
            }
        }
    }
}

impl std::error::Error for PsError {}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PsMixMatrix {
    pub h11: f64,
    pub h12: f64,
    pub h21: f64,
    pub h22: f64,
}

impl PsMixMatrix {
    fn identity_mono() -> Self {
        Self {
            h11: 1.0,
            h12: 1.0,
            h21: 0.0,
            h22: 0.0,
        }
    }

    fn interpolate(self, target: Self, fraction: f64) -> Self {
        let mix = |left, right| left + fraction * (right - left);
        Self {
            h11: mix(self.h11, target.h11),
            h12: mix(self.h12, target.h12),
            h21: mix(self.h21, target.h21),
            h22: mix(self.h22, target.h22),
        }
    }
}

static SCALE_FACTORS: LazyLock<Vec<f64>> =
    LazyLock::new(|| parse_fixed_values("const FIXP_DBL ScaleFactors["));
static SCALE_FACTORS_FINE: LazyLock<Vec<f64>> =
    LazyLock::new(|| parse_fixed_values("const FIXP_DBL ScaleFactorsFine["));
static ALPHAS: LazyLock<Vec<f64>> = LazyLock::new(|| parse_fixed_values("const FIXP_DBL Alphas["));

fn parse_fixed_values(declaration: &str) -> Vec<f64> {
    let start = ROM.find(declaration).unwrap();
    let body_start = ROM[start..].find('{').unwrap() + start;
    let body_end = ROM[body_start..].find("};").unwrap() + body_start;
    ROM[body_start..body_end]
        .split("0x")
        .skip(1)
        .filter_map(|part| {
            let digits = part
                .chars()
                .take_while(|character| character.is_ascii_hexdigit())
                .collect::<String>();
            (digits.len() == 8)
                .then(|| u32::from_str_radix(&digits, 16).unwrap() as i32 as f64 / 2_147_483_648.0)
        })
        .collect()
}

pub fn ps_mix_matrix(iid: i8, icc: i8, fine_iid: bool) -> PsMixMatrix {
    let (factors, steps) = if fine_iid {
        (&*SCALE_FACTORS_FINE, 15usize)
    } else {
        (&*SCALE_FACTORS, 7usize)
    };
    let iid = iid.clamp(-(steps as i8), steps as i8) as isize;
    let scale_r = factors[(steps as isize + iid) as usize] * 2.0;
    let scale_l = factors[(steps as isize - iid) as usize] * 2.0;
    let alpha = ALPHAS[icc.clamp(0, 7) as usize];
    let beta = alpha * (scale_r - scale_l) * std::f64::consts::FRAC_1_SQRT_2;
    PsMixMatrix {
        h11: scale_l * (beta + alpha).cos(),
        h12: scale_r * (beta - alpha).cos(),
        h21: scale_l * (beta + alpha).sin(),
        h22: scale_r * (beta - alpha).sin(),
    }
}

#[derive(Debug, Clone)]
pub struct PsQmfProcessor {
    left_synthesis: LdSbrQmfSynthesis,
    right_synthesis: LdSbrQmfSynthesis,
    hybrid: PsHybridAnalysis,
    decorrelator: PsDecorrelator,
    previous_matrices: Vec<PsMixMatrix>,
}

impl PsQmfProcessor {
    pub fn new() -> Self {
        Self {
            left_synthesis: LdSbrQmfSynthesis::new(64).unwrap(),
            right_synthesis: LdSbrQmfSynthesis::new(64).unwrap(),
            hybrid: PsHybridAnalysis::new(),
            decorrelator: PsDecorrelator::new(),
            previous_matrices: vec![PsMixMatrix::identity_mono(); 22],
        }
    }

    pub fn clear_history(&mut self) {
        *self = Self::new();
    }

    pub fn process_qmf(
        &mut self,
        mono: &[QmfSlot],
        frame: &PsFrame,
    ) -> Result<(Vec<f64>, Vec<f64>), PsError> {
        if mono.len() != *frame.borders.last().unwrap_or(&0) as usize {
            return Err(PsError::QmfSlotLayoutMismatch {
                expected: *frame.borders.last().unwrap_or(&0) as usize,
                actual: mono.len(),
            });
        }
        let mut left = Vec::with_capacity(mono.len());
        let mut right = Vec::with_capacity(mono.len());
        for (slot_index, slot) in mono.iter().enumerate() {
            if slot.real.len() < 64 || slot.imaginary.len() < 64 {
                return Err(PsError::Qmf(QmfError::InvalidSubbandCount {
                    expected: 64,
                    actual: slot.real.len().min(slot.imaginary.len()),
                }));
            }
            let envelope = frame
                .borders
                .windows(2)
                .position(|border| {
                    slot_index >= border[0] as usize && slot_index < border[1] as usize
                })
                .unwrap_or(frame.borders.len().saturating_sub(2));
            let start = frame.borders[envelope] as usize;
            let length = (frame.borders[envelope + 1] - frame.borders[envelope]).max(1) as f64;
            let fraction = (slot_index + 1 - start) as f64 / length;
            let hybrid = self.hybrid.process(slot);
            let decorrelated_hybrid = self.decorrelator.process_slot(&hybrid);
            let mut left_hybrid = vec![(0.0, 0.0); 71];
            let mut right_hybrid = vec![(0.0, 0.0); 71];
            for group in 0..22 {
                let bin = PS_GROUP_TO_BIN[group];
                let iid = frame.iid_mapped_20[envelope][bin];
                let icc = frame.icc_mapped_20[envelope][bin];
                let target = ps_mix_matrix(iid, icc, frame.header.iid_mode > 2);
                let matrix = self.previous_matrices[group].interpolate(target, fraction);
                for band in PS_GROUP_BORDERS[group]..PS_GROUP_BORDERS[group + 1] {
                    let source = hybrid[band];
                    let decorrelated = decorrelated_hybrid[band];
                    left_hybrid[band] = (
                        matrix.h11 * source.0 + matrix.h21 * decorrelated.0,
                        matrix.h11 * source.1 + matrix.h21 * decorrelated.1,
                    );
                    right_hybrid[band] = (
                        matrix.h12 * source.0 + matrix.h22 * decorrelated.0,
                        matrix.h12 * source.1 + matrix.h22 * decorrelated.1,
                    );
                }
                if slot_index + 1 == frame.borders[envelope + 1] as usize {
                    self.previous_matrices[group] = target;
                }
            }
            left.push(hybrid_synthesis(&left_hybrid));
            right.push(hybrid_synthesis(&right_hybrid));
        }
        Ok((
            self.left_synthesis.process_frame(&left)?,
            self.right_synthesis.process_frame(&right)?,
        ))
    }
}

impl Default for PsQmfProcessor {
    fn default() -> Self {
        Self::new()
    }
}

const PS_GROUP_BORDERS: [usize; 23] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 21, 25, 30, 42, 71,
];
const PS_GROUP_TO_BIN: [usize; 22] = [
    0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
];

const HYBRID_8: [f64; 13] = [
    0.00746082949812,
    0.02270420949825,
    0.04546865930473,
    0.07266113929591,
    0.09885108575264,
    0.11793710567217,
    0.125,
    0.11793710567217,
    0.09885108575264,
    0.07266113929591,
    0.04546865930473,
    0.02270420949825,
    0.00746082949812,
];
const HYBRID_2: [f64; 3] = [0.01899487526049, -0.07293139167538, 0.30596630545168];

#[derive(Debug, Clone)]
pub(crate) struct PsHybridAnalysis {
    low_history: Vec<Vec<(f64, f64)>>,
    high_delay: Vec<Vec<(f64, f64)>>,
}

impl PsHybridAnalysis {
    pub(crate) fn new() -> Self {
        Self {
            low_history: vec![vec![(0.0, 0.0); 13]; 3],
            high_delay: vec![vec![(0.0, 0.0); 61]; 6],
        }
    }

    pub(crate) fn process(&mut self, slot: &QmfSlot) -> Vec<(f64, f64)> {
        for band in 0..3 {
            self.low_history[band].rotate_left(1);
            self.low_history[band][12] = (slot.real[band], slot.imaginary[band]);
        }
        let eight = modulated_filter(&self.low_history[0], &HYBRID_8, 8);
        let mut output = vec![
            eight[7],
            eight[0],
            eight[6],
            eight[1],
            add_complex(eight[2], eight[5]),
            add_complex(eight[3], eight[4]),
        ];
        output.extend(dual_filter(&self.low_history[1], true));
        output.extend(dual_filter(&self.low_history[2], false));
        let delayed_high = self.high_delay.remove(0);
        self.high_delay.push(
            (3..64)
                .map(|band| (slot.real[band], slot.imaginary[band]))
                .collect(),
        );
        output.extend(delayed_high);
        output
    }
}

fn modulated_filter(history: &[(f64, f64)], prototype: &[f64], channels: usize) -> Vec<(f64, f64)> {
    (0..channels)
        .map(|channel| {
            history.iter().zip(prototype).enumerate().fold(
                (0.0, 0.0),
                |sum, (index, (&sample, &coefficient))| {
                    let angle = -2.0 * std::f64::consts::PI * channel as f64 * (index as f64 - 6.0)
                        / channels as f64;
                    let rotated = (
                        sample.0 * angle.cos() - sample.1 * angle.sin(),
                        sample.1 * angle.cos() + sample.0 * angle.sin(),
                    );
                    (
                        sum.0 + coefficient * rotated.0,
                        sum.1 + coefficient * rotated.1,
                    )
                },
            )
        })
        .collect()
}

fn dual_filter(history: &[(f64, f64)], invert: bool) -> [(f64, f64); 2] {
    let mut side = (0.0, 0.0);
    for (&offset, &coefficient) in [1usize, 3, 5].iter().zip(&HYBRID_2) {
        side.0 += coefficient * (history[6 - offset].0 + history[6 + offset].0);
        side.1 += coefficient * (history[6 - offset].1 + history[6 + offset].1);
    }
    let center = (history[6].0 * 0.5, history[6].1 * 0.5);
    let pair = [
        add_complex(center, side),
        (center.0 - side.0, center.1 - side.1),
    ];
    if invert {
        [pair[1], pair[0]]
    } else {
        pair
    }
}

fn add_complex(left: (f64, f64), right: (f64, f64)) -> (f64, f64) {
    (left.0 + right.0, left.1 + right.1)
}

pub(crate) fn hybrid_synthesis(hybrid: &[(f64, f64)]) -> QmfSlot {
    let sum = |range: std::ops::Range<usize>| {
        range.fold((0.0, 0.0), |sum, index| add_complex(sum, hybrid[index]))
    };
    let mut bands = vec![sum(0..6), sum(6..8), sum(8..10)];
    bands.extend_from_slice(&hybrid[10..]);
    QmfSlot {
        real: bands.iter().map(|v| v.0).collect(),
        imaginary: bands.iter().map(|v| v.1).collect(),
    }
}

static PS_DECORR_COEFFICIENTS: LazyLock<Vec<[(f64, f64); 4]>> = LazyLock::new(|| {
    let declaration = "const FIXP_STP DecorrPsCoeffsCplx";
    let source = include_str!(concat!(
        env!("FDK_AAC_UPSTREAM_DIR"),
        "/libFDK/src/FDK_decorrelate.cpp"
    ));
    let start = source.find(declaration).unwrap();
    let body_start = source[start..].find('{').unwrap() + start;
    let body_end = source[body_start..].find("};").unwrap() + body_start;
    let values = source[body_start..body_end]
        .split("0x")
        .skip(1)
        .filter_map(|part| {
            let digits = part
                .chars()
                .take_while(|character| character.is_ascii_hexdigit())
                .collect::<String>();
            (digits.len() == 8)
                .then(|| u32::from_str_radix(&digits, 16).unwrap() as i32 as f64 / 2_147_483_648.0)
        })
        .collect::<Vec<_>>();
    values
        .chunks_exact(8)
        .take(30)
        .map(|row| {
            [
                (row[0], row[1]),
                (row[2], row[3]),
                (row[4], row[5]),
                (row[6], row[7]),
            ]
        })
        .collect()
});

#[derive(Debug, Clone)]
struct PsAllpassBand {
    input_delay: std::collections::VecDeque<(f64, f64)>,
    stages: [std::collections::VecDeque<(f64, f64)>; 3],
}

impl PsAllpassBand {
    fn new() -> Self {
        Self {
            input_delay: std::collections::VecDeque::from(vec![(0.0, 0.0); 2]),
            stages: std::array::from_fn(|stage| {
                std::collections::VecDeque::from(vec![(0.0, 0.0); [3, 4, 5][stage]])
            }),
        }
    }

    fn process(&mut self, input: (f64, f64), coefficients: &[(f64, f64); 4]) -> (f64, f64) {
        let delayed = self.input_delay.pop_front().unwrap();
        self.input_delay.push_back(input);
        let mut value = complex_mul_f64(delayed, coefficients[0]);
        for stage in 0..3 {
            let state = self.stages[stage].pop_front().unwrap();
            let coefficient = coefficients[stage + 1];
            let output = add_complex(complex_mul_f64(value, coefficient), state);
            let feedback = complex_mul_f64(output, (coefficient.0, -coefficient.1));
            self.stages[stage].push_back((value.0 - feedback.0, value.1 - feedback.1));
            value = output;
        }
        value
    }
}

#[derive(Debug, Clone)]
struct PsDecorrelator {
    allpass: Vec<PsAllpassBand>,
    delay_14: Vec<std::collections::VecDeque<(f64, f64)>>,
    delay_1: Vec<(f64, f64)>,
    peak_decay: Vec<f64>,
    peak_difference: Vec<f64>,
    smooth_energy: Vec<f64>,
}

impl PsDecorrelator {
    fn new() -> Self {
        Self {
            allpass: (0..30).map(|_| PsAllpassBand::new()).collect(),
            delay_14: (0..12)
                .map(|_| std::collections::VecDeque::from(vec![(0.0, 0.0); 14]))
                .collect(),
            delay_1: vec![(0.0, 0.0); 29],
            peak_decay: vec![0.0; 20],
            peak_difference: vec![0.0; 20],
            smooth_energy: vec![0.0; 20],
        }
    }

    fn process_slot(&mut self, input: &[(f64, f64)]) -> Vec<(f64, f64)> {
        let mut output = input
            .iter()
            .copied()
            .enumerate()
            .map(|(band, value)| self.process(band, value))
            .collect::<Vec<_>>();
        for parameter_band in 0..20 {
            let range = DUCKER_BORDERS[parameter_band]..DUCKER_BORDERS[parameter_band + 1];
            let direct_energy = range
                .clone()
                .map(|band| input[band].0.powi(2) + input[band].1.powi(2))
                .sum::<f64>();
            self.peak_decay[parameter_band] =
                direct_energy.max(self.peak_decay[parameter_band] * 0.765_928_338_364_649);
            self.peak_difference[parameter_band] += 0.25
                * (self.peak_decay[parameter_band]
                    - direct_energy
                    - self.peak_difference[parameter_band]);
            self.smooth_energy[parameter_band] = (self.smooth_energy[parameter_band]
                + 0.25 * (direct_energy - self.smooth_energy[parameter_band]))
                .max(0.0);
            if 0.75 * self.peak_difference[parameter_band]
                > 0.5 * self.smooth_energy[parameter_band]
            {
                let gain = ((2.0 / 3.0) * self.smooth_energy[parameter_band]
                    / (self.peak_difference[parameter_band] + 1.0e-30))
                    .clamp(0.0, 1.0);
                for band in range {
                    output[band].0 *= gain;
                    output[band].1 *= gain;
                }
            }
        }
        output
    }

    fn process(&mut self, band: usize, input: (f64, f64)) -> (f64, f64) {
        match band {
            0..=29 => self.allpass[band].process(input, &PS_DECORR_COEFFICIENTS[band]),
            30..=41 => {
                let delay = &mut self.delay_14[band - 30];
                let output = delay.pop_front().unwrap();
                delay.push_back(input);
                output
            }
            _ => {
                let output = self.delay_1[band - 42];
                self.delay_1[band - 42] = input;
                output
            }
        }
    }
}

const DUCKER_BORDERS: [usize; 21] = [
    0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 21, 25, 30, 42, 71,
];

fn complex_mul_f64(left: (f64, f64), right: (f64, f64)) -> (f64, f64) {
    (
        left.0 * right.0 - left.1 * right.1,
        left.1 * right.0 + left.0 * right.1,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitWriter;

    fn code(book: PsHuffmanBook, symbol: i8) -> Vec<bool> {
        encode_ps_huffman(book, symbol).expect("test symbol must exist in the PS Huffman ROM")
    }

    fn write_code(writer: &mut BitWriter, bits: &[bool]) {
        for &bit in bits {
            writer.write_bool(bit);
        }
    }

    #[test]
    fn parses_header_fixed_grid_iid_icc_and_extension() {
        let iid_zero = code(PsHuffmanBook::IidFrequency, 0);
        let icc_zero = code(PsHuffmanBook::IccFrequency, 0);
        let mut writer = BitWriter::new();
        writer.write_bool(true); // header
        writer.write_bool(true); // IID
        writer.write(1, 3); // 20 bands, coarse
        writer.write_bool(true); // ICC
        writer.write(1, 3); // 20 bands
        writer.write_bool(true); // extension enabled
        writer.write_bool(false); // FIX borders
        writer.write(1, 2); // one envelope
        writer.write_bool(false); // IID frequency delta
        for _ in 0..20 {
            write_code(&mut writer, &iid_zero);
        }
        writer.write_bool(false); // ICC frequency delta
        for _ in 0..20 {
            write_code(&mut writer, &icc_zero);
        }
        writer.write(1, 4);
        writer.write(0xa5, 8);
        let bits = writer.bits_written();
        let bytes = writer.finish();
        let mut parser = PsParser::new();
        let frame = parser
            .parse(&mut BitReader::with_bit_len(&bytes, bits).unwrap(), 32)
            .unwrap();
        assert_eq!(frame.borders, vec![0, 32]);
        assert_eq!(frame.iid, vec![vec![0; 20]]);
        assert_eq!(frame.icc, vec![vec![0; 20]]);
        assert_eq!(frame.extension_data, vec![0xa5]);
        assert_eq!(frame.bits_read, bits);
    }

    #[test]
    fn inherits_header_and_reconstructs_time_deltas() {
        let frequency_zero = code(PsHuffmanBook::IidFrequency, 0);
        let time_zero = code(PsHuffmanBook::IidTime, 0);
        let mut first = BitWriter::new();
        first.write_bool(true);
        first.write_bool(true);
        first.write(0, 3); // low-resolution IID
        first.write_bool(false); // ICC disabled
        first.write_bool(false); // extension disabled
        first.write_bool(false);
        first.write(1, 2);
        first.write_bool(false);
        for _ in 0..10 {
            write_code(&mut first, &frequency_zero);
        }
        let first_bits = first.bits_written();
        let first = first.finish();
        let mut parser = PsParser::new();
        parser
            .parse(
                &mut BitReader::with_bit_len(&first, first_bits).unwrap(),
                32,
            )
            .unwrap();

        let mut next = BitWriter::new();
        next.write_bool(false); // inherited header
        next.write_bool(false); // FIX
        next.write(1, 2);
        next.write_bool(true); // time delta
        for _ in 0..10 {
            write_code(&mut next, &time_zero);
        }
        let next_bits = next.bits_written();
        let next = next.finish();
        let frame = parser
            .parse(&mut BitReader::with_bit_len(&next, next_bits).unwrap(), 32)
            .unwrap();
        assert!(!frame.header_present);
        assert_eq!(frame.iid[0], vec![0; 10]);
        assert_eq!(frame.iid_mapped_20[0], vec![0; 20]);
    }

    #[test]
    fn center_iid_and_full_correlation_produce_equal_channel_matrix() {
        let matrix = ps_mix_matrix(0, 0, false);
        assert!((matrix.h11 - matrix.h12).abs() < 1.0e-12);
        assert!(matrix.h21.abs() < 1.0e-12);
        assert!(matrix.h22.abs() < 1.0e-12);
    }

    #[test]
    fn qmf_processor_generates_finite_dual_channel_pcm() {
        let frame = PsFrame {
            header_present: true,
            header: PsHeader {
                iid_enabled: true,
                iid_mode: 1,
                icc_enabled: true,
                icc_mode: 1,
                extension_enabled: false,
            },
            variable_borders: false,
            borders: vec![0, 32],
            iid_time_domain: vec![false],
            icc_time_domain: vec![false],
            iid: vec![vec![0; 20]],
            icc: vec![vec![0; 20]],
            iid_mapped_20: vec![vec![0; 20]],
            icc_mapped_20: vec![vec![0; 20]],
            extension_data: Vec::new(),
            bits_read: 0,
        };
        let mut slots = vec![
            QmfSlot {
                real: vec![0.0; 64],
                imaginary: vec![0.0; 64]
            };
            32
        ];
        slots[0].real[0] = 1.0;
        let (left, right) = PsQmfProcessor::new().process_qmf(&slots, &frame).unwrap();
        assert_eq!(left.len(), 2048);
        assert_eq!(right.len(), 2048);
        assert!(left.iter().chain(&right).all(|value| value.is_finite()));
        assert_eq!(left, right);
    }

    #[test]
    fn extracts_ps_from_sbr_extension_id() {
        let mut writer = BitWriter::new();
        writer.write(2, 2); // EXTENSION_ID_PS_CODING
        writer.write_bool(true); // PS header
        writer.write_bool(false); // IID disabled
        writer.write_bool(false); // ICC disabled
        writer.write_bool(false); // extension disabled
        writer.write_bool(false); // FIX borders
        writer.write(0, 2); // no envelopes -> retain zero parameters
        let mut parser = PsParser::new();
        let frame = parser
            .parse_sbr_extension(&writer.finish(), 32)
            .unwrap()
            .unwrap();
        assert_eq!(frame.borders, vec![0, 32]);
        assert_eq!(frame.iid_mapped_20, vec![vec![0; 20]]);
        assert_eq!(frame.icc_mapped_20, vec![vec![0; 20]]);
    }

    #[test]
    fn hybrid_analysis_synthesis_has_six_slot_delay_and_perfect_band_sum() {
        let mut analysis = PsHybridAnalysis::new();
        let mut reconstructed = Vec::new();
        for slot_index in 0..7 {
            let mut slot = QmfSlot {
                real: vec![0.0; 64],
                imaginary: vec![0.0; 64],
            };
            if slot_index == 0 {
                for band in 0..64 {
                    slot.real[band] = (band + 1) as f64;
                    slot.imaginary[band] = -(band as f64) * 0.25;
                }
            }
            reconstructed.push(hybrid_synthesis(&analysis.process(&slot)));
        }
        assert!(reconstructed[..6]
            .iter()
            .flat_map(|slot| slot.real.iter().chain(&slot.imaginary))
            .all(|value| value.abs() < 1.0e-12));
        for band in 0..64 {
            assert!((reconstructed[6].real[band] - (band + 1) as f64).abs() < 1.0e-10);
            assert!((reconstructed[6].imaginary[band] + band as f64 * 0.25).abs() < 1.0e-10);
        }
    }

    #[test]
    fn loads_and_runs_fdk_ps_complex_allpass_coefficients() {
        assert_eq!(PS_DECORR_COEFFICIENTS.len(), 30);
        let mut decorrelator = PsDecorrelator::new();
        let output = (0..40)
            .map(|slot| decorrelator.process(0, if slot == 0 { (1.0, 0.0) } else { (0.0, 0.0) }))
            .collect::<Vec<_>>();
        assert!(output
            .iter()
            .all(|value| value.0.is_finite() && value.1.is_finite()));
        assert!(output.iter().any(|value| value.0 != 0.0 || value.1 != 0.0));
        let energy = output
            .iter()
            .map(|value| value.0 * value.0 + value.1 * value.1)
            .sum::<f64>();
        assert!(energy > 0.1 && energy < 2.0);
    }

    #[test]
    fn all_ps_huffman_books_roundtrip_and_reject_absent_symbols() {
        for book in [
            PsHuffmanBook::IidTime,
            PsHuffmanBook::IidFrequency,
            PsHuffmanBook::IidFineTime,
            PsHuffmanBook::IidFineFrequency,
            PsHuffmanBook::IccTime,
            PsHuffmanBook::IccFrequency,
        ] {
            let bits = encode_ps_huffman(book, 0).unwrap();
            let mut writer = BitWriter::new();
            write_code(&mut writer, &bits);
            assert_eq!(
                decode_ps_huffman(&mut BitReader::new(&writer.finish()), book),
                Ok(0)
            );
            assert_eq!(encode_ps_huffman(book, 63), None);
        }
        assert!(matches!(
            decode_ps_huffman(&mut BitReader::new(&[]), PsHuffmanBook::IidTime),
            Err(PsError::Bit(BitError::UnexpectedEof { .. }))
        ));
        assert_eq!(encode_ps_huffman_from(&[[1, -64]], 1), None);
    }

    #[test]
    fn parses_variable_fine_grid_time_deltas_and_extended_payload() {
        let iid_frequency = encode_ps_huffman(PsHuffmanBook::IidFineFrequency, 0).unwrap();
        let iid_time = encode_ps_huffman(PsHuffmanBook::IidFineTime, 0).unwrap();
        let icc_frequency = encode_ps_huffman(PsHuffmanBook::IccFrequency, 0).unwrap();
        let icc_time = encode_ps_huffman(PsHuffmanBook::IccTime, 0).unwrap();
        let mut writer = BitWriter::new();
        writer.write_bool(true);
        writer.write_bool(true);
        writer.write(5, 3); // fine IID, 34 bins
        writer.write_bool(true);
        writer.write(1, 3); // ICC, 20 bins
        writer.write_bool(true);
        writer.write_bool(true); // variable borders
        writer.write(1, 2); // two coded envelopes
        writer.write(0, 5); // border 1
        writer.write(9, 5); // border 10; implicit final envelope is appended
        writer.write_bool(false);
        for _ in 0..34 {
            write_code(&mut writer, &iid_frequency);
        }
        writer.write_bool(true);
        for _ in 0..34 {
            write_code(&mut writer, &iid_time);
        }
        writer.write_bool(false);
        for _ in 0..20 {
            write_code(&mut writer, &icc_frequency);
        }
        writer.write_bool(true);
        for _ in 0..20 {
            write_code(&mut writer, &icc_time);
        }
        writer.write(15, 4);
        writer.write(0, 8); // extended count remains 15
        for byte in 0..15 {
            writer.write(byte, 8);
        }
        let bits = writer.bits_written();
        let bytes = writer.finish();
        let frame = PsParser::new()
            .parse(&mut BitReader::with_bit_len(&bytes, bits).unwrap(), 32)
            .unwrap();
        assert_eq!(frame.borders, vec![0, 1, 10, 32]);
        assert_eq!(frame.iid.len(), 3);
        assert_eq!(frame.iid_mapped_20[0], vec![0; 20]);
        assert_eq!(frame.extension_data, (0..15).collect::<Vec<_>>());
    }

    #[test]
    fn parser_rejects_invalid_context_modes_and_skips_unknown_extension() {
        let mut parser = PsParser::new();
        assert_eq!(
            parser.parse(&mut BitReader::new(&[]), 16),
            Err(PsError::UnsupportedTimeSlots(16))
        );
        assert_eq!(
            parser.parse(&mut BitReader::new(&[0]), 32),
            Err(PsError::MissingInitialHeader)
        );

        let mut writer = BitWriter::new();
        writer.write_bool(true);
        writer.write_bool(true);
        writer.write(6, 3);
        writer.write_bool(false);
        writer.write_bool(false);
        let bytes = writer.finish();
        assert_eq!(
            parser.parse(&mut BitReader::new(&bytes), 32),
            Err(PsError::UnsupportedMode { iid: 6, icc: 0 })
        );

        assert_eq!(
            PsParser::new().parse_sbr_extension(&[0, 0xaa], 32),
            Ok(None)
        );

        for (iid_enabled, icc_enabled) in [(true, false), (false, true)] {
            let mut writer = BitWriter::new();
            writer.write_bool(true);
            writer.write_bool(iid_enabled);
            if iid_enabled {
                writer.write(1, 3);
            }
            writer.write_bool(icc_enabled);
            if icc_enabled {
                writer.write(1, 3);
            }
            writer.write_bool(false);
            writer.write_bool(false);
            writer.write(1, 2);
            let bits = writer.bits_written();
            let bytes = writer.finish();
            assert!(matches!(
                PsParser::new().parse(&mut BitReader::with_bit_len(&bytes, bits).unwrap(), 32,),
                Err(PsError::Bit(BitError::UnexpectedEof { .. }))
            ));
        }

        let mut writer = BitWriter::new();
        writer.write_bool(true); // header present
        writer.write_bool(false); // IID disabled
        writer.write_bool(false); // ICC disabled
        writer.write_bool(true); // extension enabled
        writer.write_bool(false); // fixed borders
        writer.write(0, 2); // no envelopes
        writer.write(1, 4); // one extension byte, deliberately absent
        let bits = writer.bits_written();
        let bytes = writer.finish();
        assert!(matches!(
            PsParser::new().parse(&mut BitReader::with_bit_len(&bytes, bits).unwrap(), 32),
            Err(PsError::Bit(BitError::UnexpectedEof { .. }))
        ));

        let mut writer = BitWriter::new();
        writer.write_bool(true); // header present
        writer.write_bool(false); // IID disabled
        writer.write_bool(false); // ICC disabled
        writer.write_bool(false); // extension disabled
        writer.write_bool(false); // fixed borders
        writer.write(2, 2); // two envelopes
        let bits = writer.bits_written();
        let bytes = writer.finish();
        let frame = PsParser::new()
            .parse(&mut BitReader::with_bit_len(&bytes, bits).unwrap(), 32)
            .unwrap();
        assert_eq!(frame.borders, [0, 16, 32]);
    }

    #[test]
    fn maps_34_bins_and_normalizes_crowded_borders() {
        let values = (0..34).map(|value| value as i8).collect::<Vec<_>>();
        let mapped = map_to_20(&values, 2);
        assert_eq!(mapped.len(), 20);
        assert_eq!(mapped[0], 0);
        assert_eq!(mapped[19], 32);

        let mut borders = vec![0, 31, 2, 32];
        normalize_borders(&mut borders, 3, 32);
        assert_eq!(borders, vec![0, 30, 31, 32]);

        let fine = ps_mix_matrix(15, 7, true);
        assert!([fine.h11, fine.h12, fine.h21, fine.h22]
            .iter()
            .all(|value| value.is_finite()));
    }

    #[test]
    fn qmf_processor_validates_slots_and_error_diagnostics() {
        let frame = PsFrame {
            header_present: true,
            header: PsHeader::default(),
            variable_borders: false,
            borders: vec![0, 1],
            iid_time_domain: vec![false],
            icc_time_domain: vec![false],
            iid: vec![vec![0; 20]],
            icc: vec![vec![0; 20]],
            iid_mapped_20: vec![vec![0; 20]],
            icc_mapped_20: vec![vec![0; 20]],
            extension_data: Vec::new(),
            bits_read: 0,
        };
        assert!(matches!(
            PsQmfProcessor::default().process_qmf(&[], &frame),
            Err(PsError::QmfSlotLayoutMismatch {
                expected: 1,
                actual: 0
            })
        ));
        let short = QmfSlot {
            real: vec![0.0; 63],
            imaginary: vec![0.0; 64],
        };
        assert!(matches!(
            PsQmfProcessor::new().process_qmf(&[short], &frame),
            Err(PsError::Qmf(QmfError::InvalidSubbandCount { .. }))
        ));

        let qmf = QmfError::InvalidSubbandCount {
            expected: 64,
            actual: 1,
        };
        assert!(matches!(PsError::from(qmf), PsError::Qmf(_)));
        let errors = [
            PsError::InvalidHuffmanCodeword,
            PsError::MissingInitialHeader,
            PsError::UnsupportedMode { iid: 6, icc: 0 },
            PsError::UnsupportedTimeSlots(16),
            PsError::QmfSlotLayoutMismatch {
                expected: 1,
                actual: 0,
            },
            PsError::from(BitError::UnexpectedEof {
                needed_bits: 1,
                remaining_bits: 0,
            }),
            PsError::from(QmfError::InvalidSubbandCount {
                expected: 64,
                actual: 1,
            }),
        ];
        for error in errors {
            assert!(!error.to_string().is_empty());
        }
    }

    #[test]
    fn decorrelator_ducker_and_delay_bands_execute() {
        let mut decorrelator = PsDecorrelator::new();
        decorrelator.peak_difference.fill(1.0);
        decorrelator.smooth_energy.fill(0.1);
        let mut input = vec![(0.0, 0.0); 71];
        input[30] = (1.0, 0.0);
        input[42] = (1.0, 0.0);
        let first = decorrelator.process_slot(&input);
        assert_eq!(first.len(), 71);
        let mut delayed = Vec::new();
        for _ in 0..14 {
            delayed = decorrelator.process_slot(&vec![(0.0, 0.0); 71]);
        }
        assert!(delayed
            .iter()
            .all(|value| value.0.is_finite() && value.1.is_finite()));
    }
}
