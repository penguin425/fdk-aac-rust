//! USAC LPC/AVQ bitstream primitives shared by LPC, FAC and TCX decoding.

use std::sync::LazyLock;

use crate::bits::{BitError, BitReader};

pub const AVQ_MAX_CODEBOOK: u8 = 36;

const FDK_USAC_ROM: &str = include_str!(concat!(
    env!("FDK_AAC_UPSTREAM_DIR"),
    "/libAACdec/src/usacdec_rom.cpp"
));
static FACTORIAL: LazyLock<Vec<i32>> = LazyLock::new(|| parse_rom("fdk_dec_tab_factorial", 8));
static ABSOLUTE_LEADERS: LazyLock<Vec<i32>> = LazyLock::new(|| parse_rom("fdk_dec_Da", 37 * 8));
static SIGN_CODES: LazyLock<Vec<i32>> = LazyLock::new(|| parse_rom("fdk_dec_Ds", 226));
static SIGN_OFFSETS: LazyLock<Vec<i32>> = LazyLock::new(|| parse_rom("fdk_dec_Is", 226));
static SIGN_COUNTS: LazyLock<Vec<i32>> = LazyLock::new(|| parse_rom("fdk_dec_Ns", 37));
static SIGN_STARTS: LazyLock<Vec<i32>> = LazyLock::new(|| parse_rom("fdk_dec_Ia", 37));
static ABSOLUTE_Q3: LazyLock<Vec<i32>> = LazyLock::new(|| parse_rom("fdk_dec_A3", 9));
static ABSOLUTE_Q4: LazyLock<Vec<i32>> = LazyLock::new(|| parse_rom("fdk_dec_A4", 28));
static OFFSET_Q3: LazyLock<Vec<i32>> = LazyLock::new(|| parse_rom("fdk_dec_I3", 9));
static OFFSET_Q4: LazyLock<Vec<i32>> = LazyLock::new(|| parse_rom("fdk_dec_I4", 28));
static LSF_FIRST_STAGE_HZ: LazyLock<Vec<f32>> = LazyLock::new(|| {
    let marker = "fdk_dec_dico_lsf_abs_8b[]";
    let start = FDK_USAC_ROM.find(marker).expect("USAC LSF ROM");
    let source = &FDK_USAC_ROM[start..];
    let body = &source[source.find('{').unwrap() + 1..source.find("};").unwrap()];
    let values: Vec<_> = body
        .split("DICO(")
        .skip(1)
        .map(|entry| {
            let hex = entry.strip_prefix("0x").unwrap().split(')').next().unwrap();
            // FIXP_DBL -> FIXP_LPC is a 16-bit shift; one stored LSF unit is
            // 1/4 Hz for the 6.4 kHz LSF domain.
            ((u32::from_str_radix(hex, 16).unwrap() >> 16) as f32) / 4.0
        })
        .collect();
    assert_eq!(values.len(), 256 * 16);
    values
});

fn parse_rom(name: &str, expected: usize) -> Vec<i32> {
    let marker = format!("{name}[");
    let start = FDK_USAC_ROM.find(&marker).expect("USAC RE8 ROM");
    let source = &FDK_USAC_ROM[start..];
    let body = &source[source.find('{').unwrap() + 1..source.find("};").unwrap()];
    let values: Vec<_> = body
        .split(|c: char| !(c.is_ascii_digit() || c == '-'))
        .filter(|word| !word.is_empty() && *word != "-")
        .map(|word| word.parse().unwrap())
        .collect();
    assert_eq!(values.len(), expected, "unexpected {name} ROM size");
    values
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UsacLpcError {
    Bit(BitError),
    InvalidNkMode(u8),
    CodebookOutOfRange(u8),
    InvalidBaseIndex { codebook: u8, index: u16 },
    MissingPreviousLsf,
}

impl From<BitError> for UsacLpcError {
    fn from(value: BitError) -> Self {
        Self::Bit(value)
    }
}

/// Decode the `qn` codebook numbers used by USAC AVQ.
pub fn decode_qn(
    reader: &mut BitReader<'_>,
    nk_mode: u8,
    count: usize,
) -> Result<Vec<u8>, UsacLpcError> {
    if nk_mode > 3 {
        return Err(UsacLpcError::InvalidNkMode(nk_mode));
    }
    let mut qn = Vec::with_capacity(count);
    if nk_mode == 1 {
        for _ in 0..count {
            let unary = read_unary(reader, AVQ_MAX_CODEBOOK as usize)? as u8;
            qn.push(if unary == 0 { 0 } else { unary + 1 });
        }
    } else {
        for _ in 0..count {
            qn.push(2 + reader.read_u8(2)?);
        }
        for value in &mut qn {
            if *value <= 4 {
                continue;
            }
            let unary = read_unary(reader, AVQ_MAX_CODEBOOK as usize)? as u8;
            *value = if nk_mode == 2 {
                if unary == 0 {
                    0
                } else {
                    unary + 4
                }
            } else {
                match unary {
                    0 => 5,
                    1 => 6,
                    2 => 0,
                    other => other + 4,
                }
            };
        }
    }
    if let Some(&value) = qn.iter().find(|&&value| value > AVQ_MAX_CODEBOOK) {
        return Err(UsacLpcError::CodebookOutOfRange(value));
    }
    Ok(qn)
}

fn read_unary(reader: &mut BitReader<'_>, maximum: usize) -> Result<usize, BitError> {
    let mut value = 0;
    while value < maximum && reader.read_bool()? {
        value += 1;
    }
    Ok(value)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AvqIndex {
    pub codebook: u8,
    pub base_index: u16,
    pub voronoi: [u32; 8],
    pub extension_order: u8,
}

impl AvqIndex {
    pub fn decode_re8(&self) -> Result<[i32; 8], UsacLpcError> {
        decode_re8(self.codebook, self.base_index, &self.voronoi)
    }
}

/// Read AVQ indices, retaining their exact integer representation for RE8.
pub fn read_avq_indices(
    reader: &mut BitReader<'_>,
    nk_mode: u8,
    count: usize,
) -> Result<Vec<AvqIndex>, UsacLpcError> {
    let qn = decode_qn(reader, nk_mode, count)?;
    let mut result = Vec::with_capacity(count);
    for codebook in qn {
        let (extension_order, base_codebook) = if codebook > 4 {
            let order = (codebook - 3) >> 1;
            (order, codebook - order * 2)
        } else {
            (0, codebook)
        };
        let base_index = reader.read_u16(usize::from(base_codebook) * 4)?;
        let mut voronoi = [0; 8];
        if extension_order != 0 {
            for value in &mut voronoi {
                *value = reader.read(usize::from(extension_order))?;
            }
        }
        result.push(AvqIndex {
            codebook,
            base_index,
            voronoi,
            extension_order,
        });
    }
    Ok(result)
}

/// Decode consecutive RE8 vectors from the AVQ bitstream.
pub fn decode_avq(
    reader: &mut BitReader<'_>,
    nk_mode: u8,
    count: usize,
) -> Result<Vec<i32>, UsacLpcError> {
    let indices = read_avq_indices(reader, nk_mode, count)?;
    let mut output = Vec::with_capacity(count * 8);
    for index in indices {
        output.extend_from_slice(&index.decode_re8()?);
    }
    Ok(output)
}

/// `gain * 2^gain_e = 10^(gain_code/28)` represented directly as f32.
pub fn decode_gain_f32(gain_code: u8) -> f32 {
    10.0f32.powf(f32::from(gain_code & 0x7f) / 28.0)
}

/// Decode an absolute 16th-order LSF vector: 8-bit first-stage codebook plus
/// two interleaved RE8 refinement vectors.
pub fn decode_absolute_lsf(reader: &mut BitReader<'_>) -> Result<[f32; 16], UsacLpcError> {
    let lsf = read_first_stage_lsf(reader)?;
    decode_refined_lsf(reader, lsf, 0)
}

fn read_first_stage_lsf(reader: &mut BitReader<'_>) -> Result<[f32; 16], BitError> {
    let first_stage = usize::from(reader.read_u8(8)?);
    let mut lsf = [0.0; 16];
    lsf.copy_from_slice(&LSF_FIRST_STAGE_HZ[first_stage * 16..first_stage * 16 + 16]);
    Ok(lsf)
}

pub fn decode_refined_lsf(
    reader: &mut BitReader<'_>,
    mut base: [f32; 16],
    nk_mode: u8,
) -> Result<[f32; 16], UsacLpcError> {
    let refinement = decode_avq(reader, nk_mode, 2)?;
    apply_lsf_refinement(&mut base, &refinement, nk_mode);
    Ok(base)
}

/// Apply the distance-adaptive second-stage LSF weighting used by FDK.
pub fn apply_lsf_refinement(lsf: &mut [f32; 16], refinement: &[i32], nk_mode: u8) {
    assert_eq!(refinement.len(), 16);
    let mut distance = [0.0; 17];
    distance[0] = lsf[0];
    distance[16] = 6400.0 - lsf[15];
    for i in 1..16 {
        distance[i] = lsf[i] - lsf[i - 1];
    }
    let numerator = match nk_mode {
        0 => 60.0,
        1 => 65.0,
        2 => 64.0,
        _ => 63.0,
    };
    for i in 0..16 {
        let weight = 2.0 * numerator / 400.0 * (distance[i] * distance[i + 1]).sqrt();
        lsf[i] += weight * refinement[i] as f32;
    }
    reorder_lsf(lsf, 50.0);
}

fn reorder_lsf(lsf: &mut [f32; 16], minimum_distance: f32) {
    let mut minimum = minimum_distance;
    for value in lsf.iter_mut() {
        *value = value.max(minimum);
        minimum = *value + minimum_distance;
    }
    let mut maximum = 6400.0 - minimum_distance;
    for value in lsf.iter_mut().rev() {
        *value = value.min(maximum);
        maximum = *value - minimum_distance;
    }
}

pub fn lsf_to_lsp(lsf: &[f32; 16]) -> [f32; 16] {
    lsf.map(|frequency| (frequency * std::f32::consts::PI / 6400.0).cos())
}

pub fn interpolate_lsf(left: &[f32; 16], right: &[f32; 16]) -> [f32; 16] {
    std::array::from_fn(|i| 0.5 * (left[i] + right[i]))
}

/// Convert 16 cosine-domain LSP values to the predictor coefficients
/// `a[1]..a[16]` (the implicit `a[0]` is one).
pub fn lsp_to_lpc(lsp: &[f32; 16]) -> [f32; 16] {
    fn multiply(left: &[f32], right: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; left.len() + right.len() - 1];
        for (i, &a) in left.iter().enumerate() {
            for (j, &b) in right.iter().enumerate() {
                output[i + j] += a * b;
            }
        }
        output
    }
    let mut first = vec![1.0];
    let mut second = vec![1.0];
    for pair in 0..8 {
        first = multiply(&first, &[1.0, -2.0 * lsp[pair * 2], 1.0]);
        second = multiply(&second, &[1.0, -2.0 * lsp[pair * 2 + 1], 1.0]);
    }
    first = multiply(&first, &[1.0, 1.0]);
    second = multiply(&second, &[1.0, -1.0]);
    std::array::from_fn(|i| 0.5 * (first[i + 1] + second[i + 1]))
}

#[derive(Debug, Clone, PartialEq)]
pub struct LpcFrame {
    pub lsf: [Option<[f32; 16]>; 5],
    pub lsp: [Option<[f32; 16]>; 5],
    pub stability: [Option<f32>; 5],
    pub coefficients: [Option<[f32; 16]>; 5],
    pub adaptive_mean: [f32; 16],
    pub bits_read: usize,
}

impl LpcFrame {
    pub fn parse(
        reader: &mut BitReader<'_>,
        modes: [u8; 4],
        first_lpd: bool,
        previous_lpc4: Option<[f32; 16]>,
        last_lpc_lost: bool,
        last_frame_ok: bool,
    ) -> Result<Self, UsacLpcError> {
        let start = reader.bits_read();
        let mut lsf: [Option<[f32; 16]>; 5] = [None; 5];
        lsf[4] = Some(decode_absolute_lsf(reader)?);
        let mut lpc0_available = true;
        let mut first_even = 0;
        if !first_lpd {
            lsf[0] = Some(previous_lpc4.ok_or(UsacLpcError::MissingPreviousLsf)?);
            lpc0_available = !last_lpc_lost;
            first_even = 2;
        }
        for slot in (first_even..3).step_by(2) {
            if slot == 2 && modes[0] == 3 {
                break;
            }
            let relative = reader.read_bool()?;
            let base = if relative {
                lsf[4].unwrap()
            } else {
                read_first_stage_lsf(reader)?
            };
            lsf[slot] = Some(decode_refined_lsf(
                reader,
                base,
                if relative { 3 } else { 0 },
            )?);
        }
        if modes[0] < 2 {
            let mode = read_limited_unary(reader, 2)?;
            lsf[1] = Some(match mode {
                1 => decode_absolute_lsf(reader)?,
                2 => {
                    if lpc0_available {
                        interpolate_lsf(&lsf[0].unwrap(), &lsf[2].unwrap())
                    } else {
                        lsf[2].unwrap()
                    }
                }
                _ => decode_refined_lsf(reader, lsf[2].unwrap(), 2)?,
            });
        }
        if modes[2] < 2 {
            let mode = read_limited_unary(reader, 3)?;
            let (base, nk_mode) = match mode {
                1 => (read_first_stage_lsf(reader)?, 0),
                0 => (interpolate_lsf(&lsf[2].unwrap(), &lsf[4].unwrap()), 1),
                2 => (lsf[2].unwrap(), 2),
                _ => (lsf[4].unwrap(), 2),
            };
            lsf[3] = Some(decode_refined_lsf(reader, base, nk_mode)?);
        }
        if let (false, false, Some(next)) = (
            lpc0_available,
            last_frame_ok,
            lsf.iter().skip(1).flatten().next().copied(),
        ) {
            let initial = initial_lsf();
            lsf[0] = Some(if modes[0] > 0 {
                std::array::from_fn(|i| 0.75 * next[i] + 0.25 * initial[i])
            } else {
                next
            });
        }
        let present: Vec<_> = lsf.iter().flatten().copied().collect();
        let tail = present.iter().rev().take(3).collect::<Vec<_>>();
        let adaptive_mean = std::array::from_fn(|i| {
            tail.iter().map(|vector| vector[i]).sum::<f32>() / tail.len() as f32
        });
        let mut stability = [None; 5];
        let mut previous_slot = 0;
        for slot in 1..5 {
            if let (Some(previous), Some(current)) = (lsf[previous_slot], lsf[slot]) {
                let distance = (0..16)
                    .map(|i| (current[i] - previous[i]).powi(2))
                    .sum::<f32>();
                stability[previous_slot] =
                    Some(((1.25 - distance / 400_000.0) * 0.5).clamp(0.0, 0.5));
                previous_slot = slot;
            }
        }
        let lsp = lsf.map(|value| value.map(|vector| lsf_to_lsp(&vector)));
        let coefficients = lsp.map(|value| value.map(|vector| lsp_to_lpc(&vector)));
        Ok(Self {
            lsf,
            lsp,
            stability,
            coefficients,
            adaptive_mean,
            bits_read: reader.bits_read() - start,
        })
    }
}

fn read_limited_unary(reader: &mut BitReader<'_>, maximum_bits: usize) -> Result<usize, BitError> {
    let mut value = 0;
    while value < maximum_bits && reader.read_bool()? {
        value += 1;
    }
    Ok(value)
}

fn initial_lsf() -> [f32; 16] {
    std::array::from_fn(|i| 6400.0 * (i + 1) as f32 / 17.0)
}

pub fn decode_re8(
    codebook: u8,
    base_index: u16,
    voronoi: &[u32; 8],
) -> Result<[i32; 8], UsacLpcError> {
    if codebook > AVQ_MAX_CODEBOOK {
        return Err(UsacLpcError::CodebookOutOfRange(codebook));
    }
    if codebook <= 4 {
        return decode_base(codebook, base_index);
    }
    let order = (codebook - 3) >> 1;
    let base_codebook = codebook - order * 2;
    let base = decode_base(base_codebook, base_index)?;
    let extension = decode_voronoi(voronoi, order);
    Ok(std::array::from_fn(|i| (base[i] << order) + extension[i]))
}

fn lookup(table: &[i32], index: u16) -> Option<usize> {
    table.iter().rposition(|&offset| offset <= i32::from(index))
}

fn decode_base(codebook: u8, index: u16) -> Result<[i32; 8], UsacLpcError> {
    if codebook < 2 {
        return Ok([0; 8]);
    }
    let (offsets, absolute_ids) = match codebook {
        2 | 3 => (&*OFFSET_Q3, &*ABSOLUTE_Q3),
        4 => (&*OFFSET_Q4, &*ABSOLUTE_Q4),
        _ => {
            return Err(UsacLpcError::InvalidBaseIndex { codebook, index });
        }
    };
    let absolute_slot =
        lookup(offsets, index).ok_or(UsacLpcError::InvalidBaseIndex { codebook, index })?;
    let absolute_id = absolute_ids[absolute_slot] as usize;
    let mut leader = [0i32; 8];
    leader.copy_from_slice(&ABSOLUTE_LEADERS[absolute_id * 8..absolute_id * 8 + 8]);
    let sign_start = SIGN_STARTS[absolute_id] as usize;
    let sign_count = SIGN_COUNTS[absolute_id] as usize;
    let sign_slot = lookup(&SIGN_OFFSETS[sign_start..sign_start + sign_count], index)
        .ok_or(UsacLpcError::InvalidBaseIndex { codebook, index })?;
    let mut sign_code = SIGN_CODES[sign_start + sign_slot] * 2;
    for value in leader.iter_mut().rev() {
        *value *= 1 - (sign_code & 2);
        sign_code >>= 1;
    }
    let rank = i32::from(index) - SIGN_OFFSETS[sign_start + sign_slot];
    Ok(decode_permutation(rank, leader))
}

fn decode_permutation(rank: i32, leader: [i32; 8]) -> [i32; 8] {
    let mut alphabet = Vec::new();
    let mut weights = Vec::new();
    for value in leader {
        if alphabet.last() == Some(&value) {
            *weights.last_mut().unwrap() += 1;
        } else {
            alphabet.push(value);
            weights.push(1i32);
        }
    }
    if weights[0] == 8 {
        return [alphabet[0]; 8];
    }
    let denominator = weights
        .iter()
        .map(|&weight| (1..=weight).product::<i32>())
        .product::<i32>();
    let mut target = rank * denominator;
    let mut denominator_factor = 1;
    let mut output = [0; 8];
    for (position, output_value) in output.iter_mut().enumerate() {
        let factor = denominator_factor * FACTORIAL[position];
        let mut symbol = 0;
        loop {
            target -= weights[symbol] * factor;
            if target < 0 {
                break;
            }
            symbol += 1;
        }
        *output_value = alphabet[symbol];
        target += weights[symbol] * factor;
        denominator_factor *= weights[symbol];
        weights[symbol] -= 1;
    }
    output
}

fn decode_voronoi(indices: &[u32; 8], order: u8) -> [i32; 8] {
    let mut point = [indices[7] as i32; 8];
    let mut sum = 0;
    for i in (1..=6).rev() {
        let value = 2 * indices[i] as i32;
        sum += value;
        point[i] += value;
    }
    point[0] += 4 * indices[0] as i32 + sum;
    let modulus = 1i32 << order;
    let real = std::array::from_fn(|i| {
        f64::from(point[i] - if i == 0 { 2 } else { 0 }) / f64::from(modulus)
    });
    let nearest = nearest_re8(real);
    std::array::from_fn(|i| point[i] - modulus * nearest[i])
}

fn nearest_2d8(input: [f64; 8]) -> [i32; 8] {
    let mut output = input.map(|value| 2 * (value / 2.0).round() as i32);
    if output.iter().sum::<i32>() % 4 != 0 {
        let mut index = 0;
        let mut maximum = 0.0;
        for i in 0..8 {
            let error = (input[i] - f64::from(output[i])).abs();
            if error > maximum {
                maximum = error;
                index = i;
            }
        }
        output[index] += if input[index] - f64::from(output[index]) < 0.0 {
            -2
        } else {
            2
        };
    }
    output
}

fn nearest_re8(input: [f64; 8]) -> [i32; 8] {
    let even = nearest_2d8(input);
    let shifted_even = nearest_2d8(input.map(|value| value - 1.0));
    let odd = shifted_even.map(|value| value + 1);
    let error = |candidate: &[i32; 8]| {
        (0..8)
            .map(|i| (input[i] - f64::from(candidate[i])).powi(2))
            .sum::<f64>()
    };
    if error(&even) < error(&odd) {
        even
    } else {
        odd
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitWriter;

    #[test]
    fn decodes_all_qn_mode_mappings() {
        let mut bits = BitWriter::new();
        bits.write_bool(false); // mode 1: Q0
        bits.write_bool(true);
        bits.write_bool(false); // mode 1: Q2
        assert_eq!(
            decode_qn(&mut BitReader::new(&bits.finish()), 1, 2).unwrap(),
            [0, 2]
        );

        let mut bits = BitWriter::new();
        bits.write(3, 2); // extension marker
        bits.write_bool(true);
        bits.write_bool(true);
        bits.write_bool(false); // unary 2 => Q0 for modes 0/3
        assert_eq!(
            decode_qn(&mut BitReader::new(&bits.finish()), 0, 1).unwrap(),
            [0]
        );

        let mut bits = BitWriter::new();
        bits.write(3, 2);
        bits.write_bool(true);
        bits.write_bool(false); // unary 1 => Q5 for mode 2
        assert_eq!(
            decode_qn(&mut BitReader::new(&bits.finish()), 2, 1).unwrap(),
            [5]
        );
    }

    #[test]
    fn reads_base_and_voronoi_indices() {
        let mut bits = BitWriter::new();
        bits.write(3, 2); // extension
        bits.write_bool(false); // qn=5
        bits.write(0xabc, 12); // reduced base Q3
        for value in 0..8 {
            bits.write(value & 1, 1);
        }
        let indices = read_avq_indices(&mut BitReader::new(&bits.finish()), 0, 1).unwrap();
        assert_eq!(indices[0].codebook, 5);
        assert_eq!(indices[0].extension_order, 1);
        assert_eq!(indices[0].base_index, 0xabc);
        assert_eq!(indices[0].voronoi, [0, 1, 0, 1, 0, 1, 0, 1]);
    }

    #[test]
    fn decodes_reference_gain_equation() {
        assert_eq!(decode_gain_f32(0), 1.0);
        assert!((decode_gain_f32(28) - 10.0).abs() < 1e-5);
        assert!((decode_gain_f32(56) - 100.0).abs() < 1e-4);
    }

    #[test]
    fn loads_re8_rom_and_decodes_known_base_points() {
        assert_eq!(ABSOLUTE_LEADERS.len(), 296);
        assert_eq!(decode_re8(0, 0, &[0; 8]).unwrap(), [0; 8]);
        assert_eq!(decode_re8(2, 0, &[0; 8]).unwrap(), [1; 8]);
        let q2_last = decode_re8(2, 127, &[0; 8]).unwrap();
        assert!(q2_last.iter().all(|value| value.abs() == 1));
    }

    #[test]
    fn voronoi_extension_produces_re8_point() {
        let point = decode_re8(5, 0, &[0, 1, 0, 1, 0, 1, 0, 1]).unwrap();
        assert_eq!(point.iter().sum::<i32>() & 1, 0);
    }

    #[test]
    fn decodes_avq_bitstream_to_flat_coefficients() {
        let mut bits = BitWriter::new();
        bits.write(0, 2); // Q2
        bits.write(0, 8); // first Q2 point
        let output = decode_avq(&mut BitReader::new(&bits.finish()), 0, 1).unwrap();
        assert_eq!(output, vec![1; 8]);
    }

    #[test]
    fn decodes_absolute_lsf_and_enforces_spacing() {
        let mut bits = BitWriter::new();
        bits.write(0, 8); // first-stage vector 0
        bits.write(2, 2); // Q4 for first half
        bits.write(2, 2); // Q4 for second half
        bits.write(0, 16);
        bits.write(0, 16);
        let lsf = decode_absolute_lsf(&mut BitReader::new(&bits.finish())).unwrap();
        assert!(lsf.windows(2).all(|pair| pair[1] - pair[0] >= 49.999));
        assert!(lsf[0] >= 50.0 && lsf[15] <= 6350.0);
    }

    #[test]
    fn first_stage_lsf_rom_has_expected_endpoints() {
        assert_eq!(LSF_FIRST_STAGE_HZ.len(), 4096);
        assert!((LSF_FIRST_STAGE_HZ[0] - 377.25).abs() < 0.01);
    }

    #[test]
    fn converts_lsf_endpoints_to_cosine_lsp() {
        let mut lsf = [0.0; 16];
        lsf[15] = 6400.0;
        let lsp = lsf_to_lsp(&lsf);
        assert!((lsp[0] - 1.0).abs() < 1e-6);
        assert!((lsp[15] + 1.0).abs() < 1e-6);
    }

    fn write_zero_avq_pair(bits: &mut BitWriter) {
        for _ in 0..2 {
            bits.write(3, 2); // extension marker
        }
        for _ in 0..2 {
            bits.write_bool(true);
            bits.write_bool(true);
            bits.write_bool(false); // unary 2 => Q0 in mode 0/3
        }
    }

    fn write_q2_pair(bits: &mut BitWriter) {
        bits.write(0, 2);
        bits.write(0, 2);
        bits.write(0, 8);
        bits.write(0, 8);
    }

    fn write_absolute_zero(bits: &mut BitWriter) {
        bits.write(0, 8);
        write_q2_pair(bits);
    }

    fn write_limited_unary(bits: &mut BitWriter, value: usize, maximum: usize) {
        for _ in 0..value.min(maximum) {
            bits.write_bool(true);
        }
        if value < maximum {
            bits.write_bool(false);
        }
    }

    fn lpc_payload(lpc1_mode: usize, lpc3_mode: usize) -> Vec<u8> {
        let mut bits = BitWriter::new();
        write_absolute_zero(&mut bits);
        for _ in 0..2 {
            bits.write_bool(true); // LPC0/LPC2 relative to LPC4, nk_mode 3
            write_zero_avq_pair(&mut bits);
        }
        write_limited_unary(&mut bits, lpc1_mode, 2);
        match lpc1_mode {
            0 => write_q2_pair(&mut bits),
            1 => write_absolute_zero(&mut bits),
            _ => {}
        }
        write_limited_unary(&mut bits, lpc3_mode, 3);
        match lpc3_mode {
            0 => {
                bits.write_bool(false);
                bits.write_bool(false);
            }
            1 => {
                bits.write(0, 8);
                write_q2_pair(&mut bits);
            }
            _ => write_q2_pair(&mut bits),
        }
        bits.finish()
    }

    #[test]
    fn parses_lpc_frame_in_tcx80_layout() {
        let mut bits = BitWriter::new();
        bits.write(0, 8); // LPC4 first stage
        write_zero_avq_pair(&mut bits);
        bits.write_bool(true); // LPC0 relative to LPC4
        write_zero_avq_pair(&mut bits);
        let frame = LpcFrame::parse(
            &mut BitReader::new(&bits.finish()),
            [3; 4],
            true,
            None,
            false,
            true,
        )
        .unwrap();
        assert!(frame.lsf[0].is_some());
        assert!(frame.lsf[1..4].iter().all(Option::is_none));
        assert!(frame.lsf[4].is_some());
        assert!(frame.lsp[0].is_some());
        assert!(frame.coefficients[0].is_some());

        let mut truncated = BitWriter::new();
        write_absolute_zero(&mut truncated);
        truncated.write_bool(true); // LPC0 relative to LPC4
        assert!(matches!(
            LpcFrame::parse(
                &mut BitReader::new(&truncated.finish()),
                [3; 4],
                true,
                None,
                false,
                true,
            ),
            Err(UsacLpcError::Bit(BitError::UnexpectedEof { .. }))
        ));
    }

    #[test]
    fn converts_lsp_to_finite_symmetric_predictor() {
        let lsp = lsf_to_lsp(&initial_lsf());
        let lpc = lsp_to_lpc(&lsp);
        assert!(lpc.iter().all(|value| value.is_finite()));
        assert!(lpc.iter().all(|value| value.abs() < 20.0));
    }

    #[test]
    fn covers_qn_mode_edge_mappings_and_validation() {
        assert_eq!(
            decode_qn(&mut BitReader::new(&[]), 4, 1),
            Err(UsacLpcError::InvalidNkMode(4))
        );

        let mut mode2 = BitWriter::new();
        mode2.write(3, 2);
        mode2.write_bool(false); // unary 0 => Q0
        assert_eq!(
            decode_qn(&mut BitReader::new(&mode2.finish()), 2, 1).unwrap(),
            [0]
        );

        for (unary, expected) in [(0usize, 5u8), (1, 6), (2, 0), (3, 7)] {
            let mut bits = BitWriter::new();
            bits.write(3, 2);
            for _ in 0..unary {
                bits.write_bool(true);
            }
            bits.write_bool(false);
            assert_eq!(
                decode_qn(&mut BitReader::new(&bits.finish()), 3, 1).unwrap(),
                [expected]
            );
        }

        let mut overflow = BitWriter::new();
        for _ in 0..AVQ_MAX_CODEBOOK {
            overflow.write_bool(true);
        }
        assert_eq!(
            decode_qn(&mut BitReader::new(&overflow.finish()), 1, 1),
            Err(UsacLpcError::CodebookOutOfRange(37))
        );
    }

    #[test]
    fn refinement_modes_interpolation_and_error_conversion() {
        let base = initial_lsf();
        for mode in 0..=3 {
            let mut lsf = base;
            apply_lsf_refinement(&mut lsf, &[1; 16], mode);
            assert!(lsf.windows(2).all(|pair| pair[1] - pair[0] >= 49.999));
        }
        let midpoint = interpolate_lsf(&[0.0; 16], &[2.0; 16]);
        assert_eq!(midpoint, [1.0; 16]);

        let bit = BitError::UnexpectedEof {
            needed_bits: 1,
            remaining_bits: 0,
        };
        assert_eq!(UsacLpcError::from(bit.clone()), UsacLpcError::Bit(bit));
        assert_eq!(
            decode_re8(37, 0, &[0; 8]),
            Err(UsacLpcError::CodebookOutOfRange(37))
        );
        assert_eq!(
            decode_base(5, 0),
            Err(UsacLpcError::InvalidBaseIndex {
                codebook: 5,
                index: 0
            })
        );
    }

    #[test]
    fn parses_all_zero_acelp_lpc_layout() {
        let payload = [0u8; 128];
        let frame = LpcFrame::parse(
            &mut BitReader::new(&payload),
            [0; 4],
            true,
            None,
            false,
            true,
        )
        .unwrap();
        assert!(frame.lsf.iter().all(Option::is_some));
        assert!(frame.coefficients.iter().all(Option::is_some));
        assert!(frame.bits_read > 0);
    }

    #[test]
    fn requires_previous_lsf_and_recovers_a_lost_lpc0() {
        let mut absolute = BitWriter::new();
        absolute.write(0, 8);
        absolute.write(0, 2);
        absolute.write(0, 2);
        absolute.write(0, 8);
        absolute.write(0, 8);
        let bytes = absolute.finish();
        assert_eq!(
            LpcFrame::parse(
                &mut BitReader::new(&bytes),
                [3; 4],
                false,
                None,
                false,
                true,
            ),
            Err(UsacLpcError::MissingPreviousLsf)
        );

        let previous = initial_lsf();
        let frame = LpcFrame::parse(
            &mut BitReader::new(&bytes),
            [3; 4],
            false,
            Some(previous),
            true,
            false,
        )
        .unwrap();
        assert!(frame.lsf[0].is_some());
        assert_ne!(frame.lsf[0].unwrap(), previous);
        assert!(frame.lsf[4].is_some());

        let mut direct_recovery = BitWriter::new();
        write_absolute_zero(&mut direct_recovery); // LPC4
        direct_recovery.write_bool(true); // LPC2 relative to LPC4
        write_zero_avq_pair(&mut direct_recovery);
        write_limited_unary(&mut direct_recovery, 2, 2); // interpolate LPC1
        write_limited_unary(&mut direct_recovery, 2, 3); // refine LPC3 from LPC2
        write_q2_pair(&mut direct_recovery);
        let frame = LpcFrame::parse(
            &mut BitReader::new(&direct_recovery.finish()),
            [0; 4],
            false,
            Some(previous),
            true,
            false,
        )
        .unwrap();
        assert_eq!(frame.lsf[0], frame.lsf[1]);
    }

    #[test]
    fn parses_absolute_interpolated_and_refined_lpc_modes() {
        for (lpc1_mode, lpc3_mode) in [(0, 0), (1, 1), (2, 2), (2, 3)] {
            let payload = lpc_payload(lpc1_mode, lpc3_mode);
            let frame = LpcFrame::parse(
                &mut BitReader::new(&payload),
                [0; 4],
                true,
                None,
                false,
                true,
            )
            .unwrap();
            assert!(frame.lsf.iter().all(Option::is_some));
        }

        // With LPC0 unavailable, interpolation mode 2 falls back to LPC2;
        // recovery mode 0 then copies the next available vector.
        let mut bits = BitWriter::new();
        write_absolute_zero(&mut bits);
        bits.write_bool(true); // LPC2 relative
        write_zero_avq_pair(&mut bits);
        write_limited_unary(&mut bits, 2, 2);
        write_limited_unary(&mut bits, 3, 3);
        write_q2_pair(&mut bits);
        let frame = LpcFrame::parse(
            &mut BitReader::new(&bits.finish()),
            [0; 4],
            false,
            Some(initial_lsf()),
            true,
            false,
        )
        .unwrap();
        assert_eq!(frame.lsf[0], frame.lsf[1]);
    }

    #[test]
    fn nearest_re8_covers_positive_adjustment_and_odd_lattice() {
        let adjusted = nearest_2d8([-1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(adjusted[0], 0);
        assert_eq!(nearest_re8([1.0; 8]), [1; 8]);
    }
}
