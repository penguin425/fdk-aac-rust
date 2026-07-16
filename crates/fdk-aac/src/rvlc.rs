//! Reversible VLC primitives used by error-resilient AAC scalefactors.

use std::fmt;

use crate::bits::{BitError, BitReader};
use crate::ics::{IcsInfo, WindowSequence};
use crate::scalefactor::ScalefactorData;
use crate::section::{SectionData, NOISE_HCB};
use crate::section::{INTENSITY_HCB, INTENSITY_HCB2, ZERO_HCB};

const RVLC_TREE: [u32; 22] = [
    0x407001, 0x002009, 0x003406, 0x004405, 0x005404, 0x006403, 0x007400, 0x008402, 0x411401,
    0x00a408, 0x00c00b, 0x00e409, 0x01000d, 0x40f40a, 0x41400f, 0x01340b, 0x011015, 0x410012,
    0x41240c, 0x416014, 0x41540d, 0x41340e,
];

const RVLC_ESCAPE_TREE: [u32; 53] = [
    0x002001, 0x400003, 0x401004, 0x402005, 0x403007, 0x404006, 0x00a405, 0x009008, 0x00b406,
    0x00c407, 0x00d408, 0x00e409, 0x40b40a, 0x40c00f, 0x40d010, 0x40e011, 0x40f012, 0x410013,
    0x411014, 0x412015, 0x016413, 0x414415, 0x017416, 0x417018, 0x419019, 0x01a418, 0x01b41a,
    0x01c023, 0x03201d, 0x01e020, 0x43501f, 0x41b41c, 0x021022, 0x41d41e, 0x41f420, 0x02402b,
    0x025028, 0x026027, 0x421422, 0x423424, 0x02902a, 0x425426, 0x427428, 0x02c02f, 0x02d02e,
    0x42942a, 0x42b42c, 0x030031, 0x42d42e, 0x42f430, 0x033034, 0x431432, 0x433434,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RvlcSideInfo {
    pub scalefactor_concealment: bool,
    pub reverse_global_gain: u8,
    pub scalefactor_bits: usize,
    pub noise_energy: Option<u16>,
    pub escapes_present: bool,
    pub escape_bits: usize,
    pub noise_last_position: Option<u16>,
    pub bits_read: usize,
}

impl RvlcSideInfo {
    pub fn parse(
        reader: &mut BitReader<'_>,
        ics: &IcsInfo,
        sections: &SectionData,
    ) -> Result<Self, RvlcError> {
        let start = reader.bits_read();
        let scalefactor_concealment = reader.read_bool()?;
        let reverse_global_gain = reader.read_u8(8)?;
        let mut scalefactor_bits =
            reader.read(if ics.window_sequence == WindowSequence::EightShort {
                11
            } else {
                9
            })? as usize;
        let noise_used = sections.codebooks.iter().any(|group| {
            group
                .iter()
                .take(ics.max_sfb as usize)
                .any(|&cb| cb == NOISE_HCB)
        });
        let noise_energy = noise_used.then(|| reader.read_u16(9)).transpose()?;
        let escapes_present = reader.read_bool()?;
        let escape_bits = if escapes_present {
            reader.read_u8(8)? as usize
        } else {
            0
        };
        let noise_last_position = if noise_used {
            let value = reader.read_u16(9)?;
            scalefactor_bits = scalefactor_bits
                .checked_sub(9)
                .ok_or(RvlcError::InvalidScalefactorLength)?;
            Some(value)
        } else {
            None
        };
        Ok(Self {
            scalefactor_concealment,
            reverse_global_gain,
            scalefactor_bits,
            noise_energy,
            escapes_present,
            escape_bits,
            noise_last_position,
            bits_read: reader.bits_read() - start,
        })
    }
}

pub fn decode_scalefactor_delta(reader: &mut BitReader<'_>) -> Result<i8, RvlcError> {
    let value = decode_tree(reader, &RVLC_TREE, 9)?;
    if value > 14 {
        return Err(RvlcError::ForbiddenCodeword(value));
    }
    Ok(value as i8 - 7)
}

pub fn decode_escape(reader: &mut BitReader<'_>) -> Result<i16, RvlcError> {
    Ok(decode_tree(reader, &RVLC_ESCAPE_TREE, 20)? as u8 as i8 as i16)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RvlcForwardResult {
    pub scalefactors: ScalefactorData,
    pub last_intensity_delta: Option<i16>,
    pub decoded_escapes: Vec<i16>,
    pub bits_read: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RvlcBackwardResult {
    pub scalefactors: ScalefactorData,
    pub bits_read: usize,
}

pub fn conceal_scalefactors(
    side_info: &RvlcSideInfo,
    ics: &IcsInfo,
    sections: &SectionData,
    global_gain: u8,
) -> Result<ScalefactorData, RvlcError> {
    if sections.codebooks.len() != ics.window_group_lengths.len()
        || sections
            .codebooks
            .iter()
            .any(|group| group.len() < ics.max_sfb as usize)
    {
        return Err(RvlcError::LayoutMismatch);
    }
    let spectral = global_gain as i16 - 100;
    let noise = global_gain as i16 - 246 + side_info.noise_energy.unwrap_or(0) as i16;
    let values = sections
        .codebooks
        .iter()
        .map(|codebooks| {
            codebooks
                .iter()
                .take(ics.max_sfb as usize)
                .map(|&codebook| match codebook {
                    ZERO_HCB => 0,
                    INTENSITY_HCB | INTENSITY_HCB2 => -100,
                    NOISE_HCB => noise,
                    1..=11 | 16..=31 => spectral,
                    _ => 0,
                })
                .collect()
        })
        .collect();
    Ok(ScalefactorData { values })
}

pub fn decode_forward(
    reader: &mut BitReader<'_>,
    side_info: &RvlcSideInfo,
    ics: &IcsInfo,
    sections: &SectionData,
    global_gain: u8,
) -> Result<RvlcForwardResult, RvlcError> {
    let start = reader.bits_read();
    if sections.codebooks.len() != ics.window_group_lengths.len()
        || sections
            .codebooks
            .iter()
            .any(|group| group.len() < ics.max_sfb as usize)
    {
        return Err(RvlcError::LayoutMismatch);
    }
    let main_bytes = read_packed_bits(reader, side_info.scalefactor_bits)?;
    let escape_bytes = read_packed_bits(reader, side_info.escape_bits)?;
    let mut escape_reader = BitReader::new(&escape_bytes);
    let mut escapes = Vec::new();
    while escape_reader.bits_read() < side_info.escape_bits {
        escapes.push(decode_escape(&mut escape_reader)?);
        if escape_reader.bits_read() > side_info.escape_bits {
            return Err(RvlcError::BitLengthMismatch {
                expected: side_info.escape_bits,
                consumed: escape_reader.bits_read(),
            });
        }
    }
    let mut escape_values = escapes.iter().copied();
    let mut main_reader = BitReader::new(&main_bytes);
    let mut factor = global_gain as i16 - 100;
    let mut position = -100i16;
    let mut noise_energy = global_gain as i16 - 100 - 90 - 256;
    let mut first_noise = true;
    let mut intensity_used = false;
    let mut values = vec![vec![0i16; ics.max_sfb as usize]; sections.codebooks.len()];
    for (group, codebooks) in sections.codebooks.iter().enumerate() {
        for (band, &codebook) in codebooks.iter().take(ics.max_sfb as usize).enumerate() {
            values[group][band] = match codebook {
                ZERO_HCB => 0,
                INTENSITY_HCB | INTENSITY_HCB2 => {
                    intensity_used = true;
                    position += decode_delta_with_escape(&mut main_reader, &mut escape_values)?;
                    position
                }
                NOISE_HCB if first_noise => {
                    first_noise = false;
                    noise_energy += side_info
                        .noise_energy
                        .ok_or(RvlcError::MissingNoiseEnergy)?
                        as i16;
                    100 + noise_energy
                }
                NOISE_HCB => {
                    noise_energy += decode_delta_with_escape(&mut main_reader, &mut escape_values)?;
                    100 + noise_energy
                }
                1..=11 | 16..=31 => {
                    factor += decode_delta_with_escape(&mut main_reader, &mut escape_values)?;
                    factor
                }
                other => return Err(RvlcError::InvalidCodebook(other)),
            };
        }
    }
    let last_intensity_delta = intensity_used
        .then(|| decode_delta_with_escape(&mut main_reader, &mut escape_values))
        .transpose()?;
    if main_reader.bits_read() != side_info.scalefactor_bits {
        return Err(RvlcError::BitLengthMismatch {
            expected: side_info.scalefactor_bits,
            consumed: main_reader.bits_read(),
        });
    }
    if escape_values.next().is_some() {
        return Err(RvlcError::UnusedEscapes);
    }
    Ok(RvlcForwardResult {
        scalefactors: ScalefactorData { values },
        last_intensity_delta,
        decoded_escapes: escapes,
        bits_read: reader.bits_read() - start,
    })
}

pub fn decode_backward(
    main_data: &[u8],
    side_info: &RvlcSideInfo,
    ics: &IcsInfo,
    sections: &SectionData,
    global_gain: u8,
    last_intensity_delta: Option<i16>,
    escapes: &[i16],
) -> Result<RvlcBackwardResult, RvlcError> {
    if main_data.len() * 8 < side_info.scalefactor_bits
        || sections.codebooks.len() != ics.window_group_lengths.len()
        || sections
            .codebooks
            .iter()
            .any(|group| group.len() < ics.max_sfb as usize)
    {
        return Err(RvlcError::LayoutMismatch);
    }
    let intensity_used = sections.codebooks.iter().any(|group| {
        group
            .iter()
            .take(ics.max_sfb as usize)
            .any(|&cb| matches!(cb, INTENSITY_HCB | INTENSITY_HCB2))
    });
    let first_noise = sections
        .codebooks
        .iter()
        .enumerate()
        .find_map(|(group, codebooks)| {
            codebooks
                .iter()
                .take(ics.max_sfb as usize)
                .position(|&cb| cb == NOISE_HCB)
                .map(|band| (group, band))
        });
    let mut reverse = ReverseBits::new(main_data, side_info.scalefactor_bits);
    let mut escape_values = escapes.iter().rev().copied();
    let mut factor = side_info.reverse_global_gain as i16 - 100;
    let mut position = last_intensity_delta.unwrap_or(0) - 100;
    let mut noise_energy = side_info.reverse_global_gain as i16
        + side_info.noise_last_position.unwrap_or(0) as i16
        - 100
        - 90
        - 256;
    if intensity_used {
        let reverse_anchor = decode_delta_with_escape_reverse(&mut reverse, &mut escape_values)?;
        if Some(reverse_anchor) != last_intensity_delta {
            return Err(RvlcError::ReverseAnchorMismatch {
                forward: last_intensity_delta.unwrap_or(0),
                backward: reverse_anchor,
            });
        }
    }
    let mut values = vec![vec![0i16; ics.max_sfb as usize]; sections.codebooks.len()];
    for group in (0..sections.codebooks.len()).rev() {
        for band in (0..ics.max_sfb as usize).rev() {
            values[group][band] = match sections.codebooks[group][band] {
                ZERO_HCB => 0,
                INTENSITY_HCB | INTENSITY_HCB2 => {
                    let value = position;
                    position -= decode_delta_with_escape_reverse(&mut reverse, &mut escape_values)?;
                    value
                }
                NOISE_HCB if first_noise == Some((group, band)) => {
                    side_info
                        .noise_energy
                        .ok_or(RvlcError::MissingNoiseEnergy)? as i16
                        + global_gain as i16
                        - 100
                        - 90
                        - 256
                }
                NOISE_HCB => {
                    let value = noise_energy;
                    noise_energy -=
                        decode_delta_with_escape_reverse(&mut reverse, &mut escape_values)?;
                    value
                }
                1..=11 | 16..=31 => {
                    let value = factor;
                    factor -= decode_delta_with_escape_reverse(&mut reverse, &mut escape_values)?;
                    value
                }
                other => return Err(RvlcError::InvalidCodebook(other)),
            };
        }
    }
    if reverse.bits_read() != side_info.scalefactor_bits {
        return Err(RvlcError::BitLengthMismatch {
            expected: side_info.scalefactor_bits,
            consumed: reverse.bits_read(),
        });
    }
    if escape_values.next().is_some() {
        return Err(RvlcError::UnusedEscapes);
    }
    Ok(RvlcBackwardResult {
        scalefactors: ScalefactorData { values },
        bits_read: reverse.bits_read(),
    })
}

fn decode_delta_with_escape(
    reader: &mut BitReader<'_>,
    escapes: &mut dyn Iterator<Item = i16>,
) -> Result<i16, RvlcError> {
    let delta = decode_scalefactor_delta(reader)? as i16;
    match delta {
        -7 => Ok(delta - escapes.next().ok_or(RvlcError::MissingEscape)?),
        7 => Ok(delta + escapes.next().ok_or(RvlcError::MissingEscape)?),
        _ => Ok(delta),
    }
}

fn decode_delta_with_escape_reverse(
    reader: &mut ReverseBits<'_>,
    escapes: &mut dyn Iterator<Item = i16>,
) -> Result<i16, RvlcError> {
    let value = decode_tree_reverse(reader, &RVLC_TREE, 9)?;
    if value > 14 {
        return Err(RvlcError::ForbiddenCodeword(value));
    }
    let delta = value as i16 - 7;
    match delta {
        -7 => Ok(delta - escapes.next().ok_or(RvlcError::MissingEscape)?),
        7 => Ok(delta + escapes.next().ok_or(RvlcError::MissingEscape)?),
        _ => Ok(delta),
    }
}

fn decode_tree_reverse(
    reader: &mut ReverseBits<'_>,
    tree: &[u32],
    maximum_length: usize,
) -> Result<u16, RvlcError> {
    let mut node = tree[0];
    for _ in 0..maximum_length {
        let branch = if reader.read_bool()? {
            node & 0x0fff
        } else {
            (node & 0x00fff000) >> 12
        };
        if branch & 0x0400 != 0 {
            return Ok((branch & 0x03ff) as u16);
        }
        node = *tree
            .get(branch as usize)
            .ok_or(RvlcError::InvalidTreeNode(branch as usize))?;
    }
    Err(RvlcError::CodewordTooLong)
}

struct ReverseBits<'a> {
    bytes: &'a [u8],
    cursor: usize,
    end: usize,
}

impl<'a> ReverseBits<'a> {
    fn new(bytes: &'a [u8], bits: usize) -> Self {
        Self {
            bytes,
            cursor: bits,
            end: bits,
        }
    }

    fn read_bool(&mut self) -> Result<bool, RvlcError> {
        if self.cursor == 0 {
            return Err(BitError::UnexpectedEof {
                needed_bits: 1,
                remaining_bits: 0,
            }
            .into());
        }
        self.cursor -= 1;
        Ok(((self.bytes[self.cursor / 8] >> (7 - self.cursor % 8)) & 1) != 0)
    }

    fn bits_read(&self) -> usize {
        self.end - self.cursor
    }
}

fn read_packed_bits(reader: &mut BitReader<'_>, bits: usize) -> Result<Vec<u8>, RvlcError> {
    if bits > reader.remaining_bits() {
        return Err(BitError::UnexpectedEof {
            needed_bits: bits,
            remaining_bits: reader.remaining_bits(),
        }
        .into());
    }
    let mut bytes = vec![0u8; bits.div_ceil(8)];
    for bit in 0..bits {
        if reader.read_bool()? {
            bytes[bit / 8] |= 1 << (7 - bit % 8);
        }
    }
    Ok(bytes)
}

fn decode_tree(
    reader: &mut BitReader<'_>,
    tree: &[u32],
    maximum_length: usize,
) -> Result<u16, RvlcError> {
    let mut node = tree[0];
    for _ in 0..maximum_length {
        let branch = if reader.read_bool()? {
            node & 0x0fff
        } else {
            (node & 0x00fff000) >> 12
        };
        if branch & 0x0400 != 0 {
            return Ok((branch & 0x03ff) as u16);
        }
        node = *tree
            .get(branch as usize)
            .ok_or(RvlcError::InvalidTreeNode(branch as usize))?;
    }
    Err(RvlcError::CodewordTooLong)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RvlcError {
    Bit(BitError),
    CodewordTooLong,
    ForbiddenCodeword(u16),
    InvalidScalefactorLength,
    InvalidTreeNode(usize),
    BitLengthMismatch { expected: usize, consumed: usize },
    InvalidCodebook(u8),
    LayoutMismatch,
    MissingEscape,
    MissingNoiseEnergy,
    UnusedEscapes,
    ReverseAnchorMismatch { forward: i16, backward: i16 },
}

impl From<BitError> for RvlcError {
    fn from(value: BitError) -> Self {
        Self::Bit(value)
    }
}

impl fmt::Display for RvlcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bit(error) => error.fmt(f),
            Self::CodewordTooLong => write!(f, "RVLC codeword is too long"),
            Self::ForbiddenCodeword(value) => write!(f, "forbidden RVLC codeword {value}"),
            Self::InvalidScalefactorLength => write!(f, "invalid RVLC scalefactor bit length"),
            Self::InvalidTreeNode(value) => write!(f, "invalid RVLC tree node {value}"),
            Self::BitLengthMismatch { expected, consumed } => write!(
                f,
                "RVLC region is {expected} bits but decoder consumed {consumed}"
            ),
            Self::InvalidCodebook(value) => write!(f, "invalid RVLC codebook {value}"),
            Self::LayoutMismatch => write!(f, "RVLC scalefactor layout mismatch"),
            Self::MissingEscape => write!(f, "RVLC boundary delta is missing an escape"),
            Self::MissingNoiseEnergy => write!(f, "RVLC noise codebook is missing noise energy"),
            Self::UnusedEscapes => write!(f, "RVLC escape region contains unused values"),
            Self::ReverseAnchorMismatch { forward, backward } => write!(
                f,
                "RVLC intensity anchor mismatch: forward {forward}, backward {backward}"
            ),
        }
    }
}

impl std::error::Error for RvlcError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitWriter;
    use crate::ics::IcsLimits;
    use crate::section::Section;

    fn long_ics(max_sfb: u8) -> IcsInfo {
        IcsInfo {
            window_sequence: WindowSequence::OnlyLong,
            window_shape: crate::ics::WindowShape::Sine,
            max_sfb,
            total_sfb: 51,
            predictor_data_present: false,
            scale_factor_grouping: 0,
            window_group_lengths: vec![1],
            bits_read: 0,
        }
    }

    fn side(scalefactor_bits: usize) -> RvlcSideInfo {
        RvlcSideInfo {
            scalefactor_concealment: false,
            reverse_global_gain: 100,
            scalefactor_bits,
            noise_energy: None,
            escapes_present: false,
            escape_bits: 0,
            noise_last_position: None,
            bits_read: 0,
        }
    }

    fn code_for(tree: &[u32], maximum_length: usize, target: u16) -> Option<Vec<bool>> {
        for length in 1..=maximum_length {
            for pattern in 0..(1usize << length) {
                let mut bytes = vec![0u8; length.div_ceil(8)];
                let mut bits = Vec::with_capacity(length);
                for bit in 0..length {
                    let value = ((pattern >> (length - 1 - bit)) & 1) != 0;
                    bits.push(value);
                    if value {
                        bytes[bit / 8] |= 1 << (7 - bit % 8);
                    }
                }
                let mut reader = BitReader::new(&bytes);
                if decode_tree(&mut reader, tree, maximum_length) == Ok(target)
                    && reader.bits_read() == length
                {
                    return Some(bits);
                }
            }
        }
        None
    }

    fn packed(bits: &[bool]) -> Vec<u8> {
        let mut writer = BitWriter::new();
        for &bit in bits {
            writer.write_bool(bit);
        }
        writer.finish()
    }

    #[test]
    fn decodes_zero_delta_from_single_bit() {
        let mut reader = BitReader::new(&[0]);
        assert_eq!(decode_scalefactor_delta(&mut reader).unwrap(), 0);
        assert_eq!(reader.bits_read(), 1);
        assert_eq!(code_for(&[0], 0, 0), None);
    }

    #[test]
    fn side_info_reports_truncated_scalefactor_length() {
        let mut writer = BitWriter::new();
        writer.write_bool(false);
        writer.write(100, 8);
        writer.write(0, 8); // one bit short of the long-window length field
        let bit_len = writer.bits_written();
        let bytes = writer.finish();
        let mut reader = BitReader::with_bit_len(&bytes, bit_len).unwrap();
        assert_eq!(
            RvlcSideInfo::parse(
                &mut reader,
                &long_ics(0),
                &SectionData {
                    sections: Vec::new(),
                    codebooks: vec![Vec::new()],
                    bits_read: 0,
                },
            ),
            Err(RvlcError::Bit(BitError::UnexpectedEof {
                needed_bits: 9,
                remaining_bits: 8,
            }))
        );
    }

    #[test]
    fn packed_bit_reader_preserves_set_bits_and_rejects_truncation() {
        assert_eq!(
            read_packed_bits(&mut BitReader::new(&[0b1010_0000]), 4).unwrap(),
            [0b1010_0000]
        );
        assert_eq!(
            read_packed_bits(&mut BitReader::new(&[]), 1),
            Err(RvlcError::Bit(BitError::UnexpectedEof {
                needed_bits: 1,
                remaining_bits: 0,
            }))
        );
    }

    #[test]
    fn parses_rvlc_side_info_with_noise_and_escapes() {
        let mut ics_writer = BitWriter::new();
        ics_writer.write_bool(false);
        ics_writer.write(0, 2);
        ics_writer.write_bool(false);
        ics_writer.write(1, 6);
        ics_writer.write_bool(false);
        let bytes = ics_writer.finish();
        let mut ics_reader = BitReader::new(&bytes);
        let ics = IcsInfo::parse_aac_lc(&mut ics_reader, IcsLimits::AAC_LC_MAX).unwrap();
        let sections = SectionData {
            sections: vec![Section {
                group: 0,
                codebook: NOISE_HCB,
                start_sfb: 0,
                end_sfb: 1,
            }],
            codebooks: vec![vec![NOISE_HCB]],
            bits_read: 0,
        };
        let mut writer = BitWriter::new();
        writer.write_bool(true);
        writer.write(100, 8);
        writer.write(20, 9);
        writer.write(123, 9);
        writer.write_bool(true);
        writer.write(7, 8);
        writer.write(321, 9);
        let payload = writer.finish();
        let mut reader = BitReader::new(&payload);
        let side = RvlcSideInfo::parse(&mut reader, &ics, &sections).unwrap();
        assert_eq!(side.scalefactor_bits, 11);
        assert_eq!(side.noise_energy, Some(123));
        assert_eq!(side.escape_bits, 7);
        assert_eq!(side.noise_last_position, Some(321));
    }

    #[test]
    fn reconstructs_forward_spectral_and_intensity_accumulators() {
        let mut ics_writer = BitWriter::new();
        ics_writer.write_bool(false);
        ics_writer.write(0, 2);
        ics_writer.write_bool(false);
        ics_writer.write(2, 6);
        ics_writer.write_bool(false);
        let bytes = ics_writer.finish();
        let mut ics_reader = BitReader::new(&bytes);
        let ics = IcsInfo::parse_aac_lc(&mut ics_reader, IcsLimits::AAC_LC_MAX).unwrap();
        let sections = SectionData {
            sections: vec![
                Section {
                    group: 0,
                    codebook: 1,
                    start_sfb: 0,
                    end_sfb: 1,
                },
                Section {
                    group: 0,
                    codebook: INTENSITY_HCB,
                    start_sfb: 1,
                    end_sfb: 2,
                },
            ],
            codebooks: vec![vec![1, INTENSITY_HCB]],
            bits_read: 0,
        };
        let side = RvlcSideInfo {
            scalefactor_concealment: false,
            reverse_global_gain: 100,
            scalefactor_bits: 3,
            noise_energy: None,
            escapes_present: false,
            escape_bits: 0,
            noise_last_position: None,
            bits_read: 0,
        };
        // One-bit zero deltas for spectral, intensity, and final intensity anchor.
        let mut reader = BitReader::new(&[0]);
        let decoded = decode_forward(&mut reader, &side, &ics, &sections, 100).unwrap();
        assert_eq!(decoded.scalefactors.values, [vec![0, -100]]);
        assert_eq!(decoded.last_intensity_delta, Some(0));
        assert_eq!(decoded.bits_read, 3);
        let backward = decode_backward(
            &[0],
            &side,
            &ics,
            &sections,
            100,
            decoded.last_intensity_delta,
            &decoded.decoded_escapes,
        )
        .unwrap();
        assert_eq!(backward.scalefactors, decoded.scalefactors);
        assert_eq!(backward.bits_read, 3);
    }

    #[test]
    fn concealment_uses_stable_values_for_each_codebook_class() {
        let mut ics_writer = BitWriter::new();
        ics_writer.write_bool(false);
        ics_writer.write(0, 2);
        ics_writer.write_bool(false);
        ics_writer.write(4, 6);
        ics_writer.write_bool(false);
        let bytes = ics_writer.finish();
        let mut reader = BitReader::new(&bytes);
        let ics = IcsInfo::parse_aac_lc(&mut reader, IcsLimits::AAC_LC_MAX).unwrap();
        let sections = SectionData {
            sections: Vec::new(),
            codebooks: vec![vec![ZERO_HCB, 1, INTENSITY_HCB, NOISE_HCB]],
            bits_read: 0,
        };
        let side = RvlcSideInfo {
            scalefactor_concealment: true,
            reverse_global_gain: 100,
            scalefactor_bits: 0,
            noise_energy: Some(146),
            escapes_present: false,
            escape_bits: 0,
            noise_last_position: Some(0),
            bits_read: 0,
        };
        assert_eq!(
            conceal_scalefactors(&side, &ics, &sections, 100)
                .unwrap()
                .values,
            [vec![0, 0, -100, 0]]
        );
    }

    #[test]
    fn parses_short_side_info_and_rejects_impossible_noise_length() {
        let short = IcsInfo {
            window_sequence: WindowSequence::EightShort,
            window_shape: crate::ics::WindowShape::Sine,
            max_sfb: 1,
            total_sfb: 15,
            predictor_data_present: false,
            scale_factor_grouping: 0x7f,
            window_group_lengths: vec![8],
            bits_read: 0,
        };
        let spectral = SectionData {
            sections: Vec::new(),
            codebooks: vec![vec![1]],
            bits_read: 0,
        };
        let mut writer = BitWriter::new();
        writer.write_bool(false);
        writer.write(90, 8);
        writer.write(17, 11);
        writer.write_bool(false);
        let bytes = writer.finish();
        let parsed = RvlcSideInfo::parse(&mut BitReader::new(&bytes), &short, &spectral).unwrap();
        assert_eq!(parsed.scalefactor_bits, 17);
        assert_eq!(parsed.escape_bits, 0);
        assert_eq!(parsed.noise_energy, None);

        let noisy = SectionData {
            sections: Vec::new(),
            codebooks: vec![vec![NOISE_HCB]],
            bits_read: 0,
        };
        let mut writer = BitWriter::new();
        writer.write_bool(false);
        writer.write(90, 8);
        writer.write(8, 11);
        writer.write(0, 9);
        writer.write_bool(false);
        writer.write(0, 9);
        let bytes = writer.finish();
        assert_eq!(
            RvlcSideInfo::parse(&mut BitReader::new(&bytes), &short, &noisy),
            Err(RvlcError::InvalidScalefactorLength)
        );
    }

    #[test]
    fn forward_noise_decoding_and_validation_errors() {
        let ics = long_ics(2);
        let noise = SectionData {
            sections: Vec::new(),
            codebooks: vec![vec![NOISE_HCB, NOISE_HCB]],
            bits_read: 0,
        };
        let mut info = side(1);
        info.noise_energy = Some(146);
        let decoded = decode_forward(&mut BitReader::new(&[0]), &info, &ics, &noise, 100).unwrap();
        assert_eq!(decoded.scalefactors.values, [vec![-100, -100]]);

        info.noise_energy = None;
        assert_eq!(
            decode_forward(&mut BitReader::new(&[0]), &info, &ics, &noise, 100),
            Err(RvlcError::MissingNoiseEnergy)
        );

        let invalid = SectionData {
            sections: Vec::new(),
            codebooks: vec![vec![12]],
            bits_read: 0,
        };
        assert_eq!(
            decode_forward(
                &mut BitReader::new(&[]),
                &side(0),
                &long_ics(1),
                &invalid,
                100
            ),
            Err(RvlcError::InvalidCodebook(12))
        );
        let zero = SectionData {
            sections: Vec::new(),
            codebooks: vec![vec![ZERO_HCB]],
            bits_read: 0,
        };
        assert_eq!(
            decode_forward(
                &mut BitReader::new(&[0]),
                &side(1),
                &long_ics(1),
                &zero,
                100
            ),
            Err(RvlcError::BitLengthMismatch {
                expected: 1,
                consumed: 0
            })
        );
        assert_eq!(
            decode_forward(&mut BitReader::new(&[]), &side(1), &long_ics(1), &zero, 100),
            Err(RvlcError::Bit(BitError::UnexpectedEof {
                needed_bits: 1,
                remaining_bits: 0,
            }))
        );
    }

    #[test]
    fn layout_concealment_and_backward_validation_errors() {
        let malformed = SectionData {
            sections: Vec::new(),
            codebooks: Vec::new(),
            bits_read: 0,
        };
        assert_eq!(
            conceal_scalefactors(&side(0), &long_ics(1), &malformed, 100),
            Err(RvlcError::LayoutMismatch)
        );
        assert_eq!(
            decode_forward(
                &mut BitReader::new(&[]),
                &side(0),
                &long_ics(1),
                &malformed,
                100
            ),
            Err(RvlcError::LayoutMismatch)
        );
        assert_eq!(
            decode_backward(&[], &side(1), &long_ics(1), &malformed, 100, None, &[]),
            Err(RvlcError::LayoutMismatch)
        );

        let zero = SectionData {
            sections: Vec::new(),
            codebooks: vec![vec![ZERO_HCB]],
            bits_read: 0,
        };
        assert_eq!(
            decode_backward(&[0], &side(1), &long_ics(1), &zero, 100, None, &[]),
            Err(RvlcError::BitLengthMismatch {
                expected: 1,
                consumed: 0
            })
        );
        let invalid = SectionData {
            sections: Vec::new(),
            codebooks: vec![vec![12]],
            bits_read: 0,
        };
        assert_eq!(
            decode_backward(&[], &side(0), &long_ics(1), &invalid, 100, None, &[]),
            Err(RvlcError::InvalidCodebook(12))
        );
    }

    #[test]
    fn tree_and_reverse_reader_report_corrupt_inputs() {
        let mut reverse = ReverseBits::new(&[], 0);
        assert_eq!(
            reverse.read_bool(),
            Err(RvlcError::Bit(BitError::UnexpectedEof {
                needed_bits: 1,
                remaining_bits: 0,
            }))
        );

        let mut reader = BitReader::new(&[0]);
        assert_eq!(
            decode_tree(&mut reader, &[0], 1),
            Err(RvlcError::CodewordTooLong)
        );
        let mut reader = BitReader::new(&[0]);
        assert_eq!(
            decode_tree(&mut reader, &[0x0000_2000], 2),
            Err(RvlcError::InvalidTreeNode(2))
        );
        let mut reverse = ReverseBits::new(&[0], 1);
        assert_eq!(
            decode_tree_reverse(&mut reverse, &[0], 1),
            Err(RvlcError::CodewordTooLong)
        );
        let mut reverse = ReverseBits::new(&[0], 1);
        assert_eq!(
            decode_tree_reverse(&mut reverse, &[0x0000_2000], 2),
            Err(RvlcError::InvalidTreeNode(2))
        );
    }

    #[test]
    fn errors_have_stable_diagnostic_text() {
        let errors = [
            RvlcError::CodewordTooLong,
            RvlcError::ForbiddenCodeword(15),
            RvlcError::InvalidScalefactorLength,
            RvlcError::InvalidTreeNode(3),
            RvlcError::BitLengthMismatch {
                expected: 2,
                consumed: 1,
            },
            RvlcError::InvalidCodebook(12),
            RvlcError::LayoutMismatch,
            RvlcError::MissingEscape,
            RvlcError::MissingNoiseEnergy,
            RvlcError::UnusedEscapes,
            RvlcError::ReverseAnchorMismatch {
                forward: 1,
                backward: -1,
            },
        ];
        for error in errors {
            assert!(!error.to_string().is_empty());
        }
        let bit = RvlcError::from(BitError::UnexpectedEof {
            needed_bits: 1,
            remaining_bits: 0,
        });
        assert!(bit.to_string().contains("bit"));
    }

    #[test]
    fn decodes_escape_and_boundary_deltas_in_both_directions() {
        let escape_code = code_for(&RVLC_ESCAPE_TREE, 20, 5).unwrap();
        let escape_bytes = packed(&escape_code);
        assert_eq!(
            decode_escape(&mut BitReader::new(&escape_bytes)).unwrap(),
            5
        );

        for (symbol, escape, expected) in [(0, 3, -10), (14, 3, 10)] {
            let code = code_for(&RVLC_TREE, 9, symbol).unwrap();
            let bytes = packed(&code);
            assert_eq!(
                decode_delta_with_escape(&mut BitReader::new(&bytes), &mut [escape].into_iter()),
                Ok(expected)
            );
            assert_eq!(
                decode_delta_with_escape(&mut BitReader::new(&bytes), &mut [].into_iter()),
                Err(RvlcError::MissingEscape)
            );

            let reverse_bytes = packed(&code.iter().rev().copied().collect::<Vec<_>>());
            let mut reverse = ReverseBits::new(&reverse_bytes, code.len());
            assert_eq!(
                decode_delta_with_escape_reverse(&mut reverse, &mut [escape].into_iter()),
                Ok(expected)
            );
        }
    }

    #[test]
    fn forward_rejects_an_unused_escape_region() {
        let escape_code = code_for(&RVLC_ESCAPE_TREE, 20, 1).unwrap();
        let bytes = packed(&escape_code);
        let mut info = side(0);
        info.escape_bits = escape_code.len();
        info.escapes_present = true;
        let zero = SectionData {
            sections: Vec::new(),
            codebooks: vec![vec![ZERO_HCB]],
            bits_read: 0,
        };
        assert_eq!(
            decode_forward(&mut BitReader::new(&bytes), &info, &long_ics(1), &zero, 100),
            Err(RvlcError::UnusedEscapes)
        );
    }

    #[test]
    fn backward_decodes_noise_and_checks_intensity_anchor() {
        let noise = SectionData {
            sections: Vec::new(),
            codebooks: vec![vec![NOISE_HCB, NOISE_HCB]],
            bits_read: 0,
        };
        let mut info = side(1);
        info.noise_energy = Some(146);
        info.noise_last_position = Some(146);
        let decoded = decode_backward(&[0], &info, &long_ics(2), &noise, 100, None, &[]).unwrap();
        assert_eq!(decoded.scalefactors.values, [vec![-200, -200]]);

        let intensity = SectionData {
            sections: Vec::new(),
            codebooks: vec![vec![INTENSITY_HCB]],
            bits_read: 0,
        };
        assert_eq!(
            decode_backward(&[0], &side(2), &long_ics(1), &intensity, 100, Some(1), &[]),
            Err(RvlcError::ReverseAnchorMismatch {
                forward: 1,
                backward: 0
            })
        );
    }

    #[test]
    fn rejects_forbidden_symbols_forward_and_backward() {
        let code = code_for(&RVLC_TREE, 9, 16).unwrap();
        let bytes = packed(&code);
        assert_eq!(
            decode_scalefactor_delta(&mut BitReader::new(&bytes)),
            Err(RvlcError::ForbiddenCodeword(16))
        );

        let reverse_bytes = packed(&code.iter().rev().copied().collect::<Vec<_>>());
        let mut reverse = ReverseBits::new(&reverse_bytes, code.len());
        assert_eq!(
            decode_delta_with_escape_reverse(&mut reverse, &mut [].into_iter()),
            Err(RvlcError::ForbiddenCodeword(16))
        );
    }

    #[test]
    fn escape_region_rejects_codeword_past_declared_length() {
        let first_bit = [false, true]
            .into_iter()
            .find(|&bit| {
                let bytes = packed(&[bit]);
                let mut reader = BitReader::new(&bytes);
                decode_escape(&mut reader).is_ok() && reader.bits_read() > 1
            })
            .expect("an escape code continues through padding");
        let zero = SectionData {
            sections: Vec::new(),
            codebooks: vec![vec![ZERO_HCB]],
            bits_read: 0,
        };
        let mut info = side(0);
        info.escapes_present = true;
        info.escape_bits = 1;
        assert!(matches!(
            decode_forward(
                &mut BitReader::new(&packed(&[first_bit])),
                &info,
                &long_ics(1),
                &zero,
                100,
            ),
            Err(RvlcError::BitLengthMismatch { expected: 1, consumed }) if consumed > 1
        ));
    }

    #[test]
    fn backward_rejects_unused_escape_and_concealment_accepts_unknown_book() {
        let zero = SectionData {
            sections: Vec::new(),
            codebooks: vec![vec![ZERO_HCB]],
            bits_read: 0,
        };
        assert_eq!(
            decode_backward(&[], &side(0), &long_ics(1), &zero, 100, None, &[1]),
            Err(RvlcError::UnusedEscapes)
        );
        let unknown = SectionData {
            sections: Vec::new(),
            codebooks: vec![vec![12]],
            bits_read: 0,
        };
        assert_eq!(
            conceal_scalefactors(&side(0), &long_ics(1), &unknown, 100)
                .unwrap()
                .values,
            [vec![0]]
        );
    }
}
