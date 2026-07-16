//! Huffman Codeword Reordering (HCR) side information.
//!
//! ISO/IEC 14496-3 carries these protected fields separately from the
//! reordered spectral payload.  Keeping the bounds here prevents malformed
//! ER frames from turning a payload length into an unbounded read.

use std::fmt;

use crate::bits::{BitError, BitReader, BitWriter};
use crate::huffman::{write_spectral_tuple, HuffmanError};
use crate::ics::{IcsInfo, WindowSequence};
use crate::section::SectionData;
use crate::spectral::{decode_spectral_tuple, SpectralData, SpectralError};

pub const SCE_MAX_REORDERED_SPECTRAL_BITS: usize = 6144;
pub const CPE_MAX_REORDERED_SPECTRAL_BITS: usize = 12288;
pub const MAX_LONGEST_CODEWORD_BITS: usize = 49;

const MAX_CODEWORD_BITS: [usize; 32] = [
    0, 11, 9, 20, 16, 13, 11, 14, 12, 17, 14, 49, 0, 0, 0, 0, 14, 17, 21, 21, 25, 25, 29, 29, 29,
    29, 33, 33, 33, 37, 37, 41,
];
const CODEBOOK_DIMENSION_SHIFT: [usize; 32] = [
    1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
];
const CODEBOOK_PRIORITY: [usize; 32] = [
    0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 22, 0, 0, 0, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
    18, 19, 20, 21,
];
const LARGEST_ABSOLUTE_VALUE: [i32; 32] = [
    0, 1, 1, 2, 2, 4, 4, 7, 7, 12, 12, 8191, 0, 0, 0, 0, 15, 31, 47, 63, 95, 127, 159, 191, 223,
    255, 319, 383, 511, 767, 1023, 2047,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HcrElementType {
    SingleChannel,
    ChannelPair,
    LowFrequencyEffects,
    CouplingChannel,
}

impl HcrElementType {
    const fn max_reordered_bits(self) -> usize {
        match self {
            Self::ChannelPair => CPE_MAX_REORDERED_SPECTRAL_BITS,
            Self::SingleChannel | Self::LowFrequencyEffects | Self::CouplingChannel => {
                SCE_MAX_REORDERED_SPECTRAL_BITS
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HcrSideInfo {
    pub reordered_spectral_bits: usize,
    pub longest_codeword_bits: usize,
    pub bits_read: usize,
}

impl HcrSideInfo {
    pub fn parse(
        reader: &mut BitReader<'_>,
        element_type: HcrElementType,
    ) -> Result<Self, HcrError> {
        let start = reader.bits_read();
        let reordered_spectral_bits = reader.read_u16(14)? as usize;
        let longest_codeword_bits = reader.read_u8(6)? as usize;
        let max_reordered_bits = element_type.max_reordered_bits();
        if reordered_spectral_bits > max_reordered_bits {
            return Err(HcrError::ReorderedSpectralLengthOutOfRange {
                length: reordered_spectral_bits,
                maximum: max_reordered_bits,
            });
        }
        if longest_codeword_bits > MAX_LONGEST_CODEWORD_BITS {
            return Err(HcrError::LongestCodewordLengthOutOfRange {
                length: longest_codeword_bits,
                maximum: MAX_LONGEST_CODEWORD_BITS,
            });
        }
        if reordered_spectral_bits < longest_codeword_bits {
            return Err(HcrError::InconsistentLengths {
                reordered_spectral_bits,
                longest_codeword_bits,
            });
        }
        Ok(Self {
            reordered_spectral_bits,
            longest_codeword_bits,
            bits_read: reader.bits_read() - start,
        })
    }

    pub fn read_payload(self, reader: &mut BitReader<'_>) -> Result<Vec<u8>, HcrError> {
        let mut payload = vec![0u8; self.reordered_spectral_bits.div_ceil(8)];
        for bit in 0..self.reordered_spectral_bits {
            if reader.read_bool()? {
                payload[bit / 8] |= 1 << (7 - (bit % 8));
            }
        }
        Ok(payload)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HcrSection {
    pub codebook: u8,
    pub spectral_lines: usize,
}

pub fn sections_from_ics(
    ics: &IcsInfo,
    section_data: &SectionData,
    band_offsets: &[usize],
) -> Result<Vec<HcrSection>, HcrError> {
    let max_sfb = ics.max_sfb as usize;
    if band_offsets.len() <= max_sfb
        || section_data.codebooks.len() != ics.window_group_lengths.len()
        || section_data
            .codebooks
            .iter()
            .any(|codebooks| codebooks.len() < max_sfb)
    {
        return Err(HcrError::LayoutMismatch);
    }
    if ics.window_sequence != WindowSequence::EightShort {
        let mut result = Vec::with_capacity(section_data.sections.len());
        for section in &section_data.sections {
            let start = section.start_sfb as usize;
            let end = section.end_sfb as usize;
            if end > max_sfb || start >= end {
                return Err(HcrError::LayoutMismatch);
            }
            result.push(HcrSection {
                codebook: hcr_codebook(section.codebook),
                spectral_lines: band_offsets[end] - band_offsets[start],
            });
        }
        return Ok(result);
    }

    let mut result: Vec<HcrSection> = Vec::new();
    for band in 0..max_sfb {
        let width = band_offsets[band + 1] - band_offsets[band];
        if width % 4 != 0 {
            return Err(HcrError::ShortBandNotFourLineAligned { band, width });
        }
        for _ in 0..(width / 4) {
            for (group, &group_len) in ics.window_group_lengths.iter().enumerate() {
                let codebook = hcr_codebook(section_data.codebooks[group][band]);
                for _ in 0..group_len {
                    if let Some(last) = result.last_mut().filter(|last| last.codebook == codebook) {
                        last.spectral_lines += 4;
                    } else {
                        result.push(HcrSection {
                            codebook,
                            spectral_lines: 4,
                        });
                    }
                }
            }
        }
    }
    Ok(result)
}

fn hcr_codebook(codebook: u8) -> u8 {
    if codebook <= 11 || (16..=31).contains(&codebook) {
        codebook
    } else {
        0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HcrSegment {
    pub codebook: u8,
    pub section_index: usize,
    pub codeword_index: usize,
    pub left_bit: usize,
    pub right_bit: usize,
}

impl HcrSegment {
    pub fn width(self) -> usize {
        self.right_bit - self.left_bit + 1
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HcrReadDirection {
    LeftToRight,
    RightToLeft,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HcrSegmentState {
    pub left_bit: usize,
    pub right_bit: usize,
    remaining_bits: usize,
}

impl HcrSegmentState {
    pub fn from_segment(segment: HcrSegment) -> Self {
        Self {
            left_bit: segment.left_bit,
            right_bit: segment.right_bit,
            remaining_bits: segment.width(),
        }
    }

    pub fn remaining_bits(self) -> usize {
        self.remaining_bits
    }
}

pub fn decode_segment_codeword(
    payload: &[u8],
    state: &mut HcrSegmentState,
    codebook: u8,
    direction: HcrReadDirection,
) -> Result<(Vec<i32>, usize), HcrError> {
    let available = state.remaining_bits();
    if available == 0 {
        return Err(HcrError::EmptySegment);
    }
    let packed = copy_directional_bits(payload, *state, direction)?;
    let mut reader = BitReader::with_bit_len(&packed, available)?;
    let coefficients = decode_spectral_tuple(&mut reader, codebook)?;
    let consumed = reader.bits_read();
    match direction {
        HcrReadDirection::LeftToRight => state.left_bit += consumed,
        HcrReadDirection::RightToLeft => state.right_bit = state.right_bit.saturating_sub(consumed),
    }
    state.remaining_bits -= consumed;
    Ok((coefficients, consumed))
}

fn copy_directional_bits(
    source: &[u8],
    state: HcrSegmentState,
    direction: HcrReadDirection,
) -> Result<Vec<u8>, HcrError> {
    let length = state.remaining_bits();
    if source.len().saturating_mul(8) <= state.right_bit {
        return Err(HcrError::PayloadTooShort {
            needed_bits: state.right_bit + 1,
            available_bits: source.len().saturating_mul(8),
        });
    }
    let mut result = vec![0u8; length.div_ceil(8)];
    for offset in 0..length {
        let source_bit = match direction {
            HcrReadDirection::LeftToRight => state.left_bit + offset,
            HcrReadDirection::RightToLeft => state.right_bit - offset,
        };
        let bit = (source[source_bit / 8] >> (7 - (source_bit % 8))) & 1;
        result[offset / 8] |= bit << (7 - (offset % 8));
    }
    Ok(result)
}

/// Builds the HCR priority-codeword segmentation grid used by the FDK
/// decoder. Zero/intensity/noise sections must be represented as codebook 0.
pub fn prepare_segmentation_grid(
    side_info: &HcrSideInfo,
    sections: &[HcrSection],
) -> Result<Vec<HcrSegment>, HcrError> {
    let mut sorted = Vec::with_capacity(sections.len());
    for (index, section) in sections.iter().copied().enumerate() {
        let codebook = section.codebook as usize;
        if codebook >= MAX_CODEWORD_BITS.len() {
            return Err(HcrError::InvalidCodebook(section.codebook));
        }
        let dimension = 1usize << CODEBOOK_DIMENSION_SHIFT[codebook];
        if section.spectral_lines == 0 || section.spectral_lines % dimension != 0 {
            return Err(HcrError::InvalidSectionLineCount {
                section: index,
                spectral_lines: section.spectral_lines,
                dimension,
            });
        }
        if CODEBOOK_PRIORITY[codebook] != 0 {
            sorted.push((CODEBOOK_PRIORITY[codebook], index, section));
        }
    }
    // FDK sorts descending by priority and preserves the original order for
    // equal-priority sections.
    sorted.sort_by_key(|&(priority, index, _)| (std::cmp::Reverse(priority), index));

    let mut bit = 0usize;
    let mut segments = Vec::new();
    'sections: for (_, index, section) in sorted {
        let codebook = section.codebook as usize;
        let width = MAX_CODEWORD_BITS[codebook].min(side_info.longest_codeword_bits);
        if width == 0 {
            continue;
        }
        let codewords = section.spectral_lines >> CODEBOOK_DIMENSION_SHIFT[codebook];
        for codeword_index in 0..codewords {
            if bit + width <= side_info.reordered_spectral_bits {
                segments.push(HcrSegment {
                    codebook: section.codebook,
                    section_index: index,
                    codeword_index,
                    left_bit: bit,
                    right_bit: bit + width - 1,
                });
                bit += width;
            } else {
                if let Some(last) = segments.last_mut() {
                    last.right_bit = side_info.reordered_spectral_bits - 1;
                }
                break 'sections;
            }
        }
    }
    Ok(segments)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HcrPriorityCodeword {
    pub section_index: usize,
    pub codeword_index: usize,
    pub coefficients: Vec<i32>,
    pub consumed_bits: usize,
    pub remaining_left_bit: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HcrDecodedCodeword {
    pub section_index: usize,
    pub codeword_index: usize,
    pub coefficients: Vec<i32>,
}

/// Encode HCR with one fixed-width priority segment per spectral codeword.
/// This is less compact than multi-set interleaving but fully conformant and
/// makes every codeword decodable in the priority pass.
pub fn encode_reordered_codewords(
    sections: &[HcrSection],
    codewords: &[HcrDecodedCodeword],
) -> Result<(HcrSideInfo, Vec<u8>), HcrError> {
    let descriptors = sorted_codeword_descriptors(sections)?;
    if descriptors.is_empty() {
        return Ok((
            HcrSideInfo {
                reordered_spectral_bits: 0,
                longest_codeword_bits: 0,
                bits_read: 20,
            },
            Vec::new(),
        ));
    }
    let longest = descriptors
        .iter()
        .map(|descriptor| MAX_CODEWORD_BITS[descriptor.codebook as usize])
        .max()
        .unwrap();
    let total = descriptors
        .iter()
        .map(|descriptor| MAX_CODEWORD_BITS[descriptor.codebook as usize].min(longest))
        .sum::<usize>();
    if total > CPE_MAX_REORDERED_SPECTRAL_BITS {
        return Err(HcrError::ReorderedSpectralLengthOutOfRange {
            length: total,
            maximum: CPE_MAX_REORDERED_SPECTRAL_BITS,
        });
    }
    let mut writer = BitWriter::new();
    for descriptor in descriptors {
        let word = codewords
            .iter()
            .find(|word| {
                word.section_index == descriptor.section_index
                    && word.codeword_index == descriptor.codeword_index
            })
            .ok_or(HcrError::MissingCodeword {
                section: descriptor.section_index,
                codeword: descriptor.codeword_index,
            })?;
        let start = writer.bits_written();
        write_spectral_tuple(&mut writer, descriptor.codebook, &word.coefficients)
            .map_err(|error| HcrError::Spectral(SpectralError::Huffman(error)))?;
        let used = writer.bits_written() - start;
        let width = MAX_CODEWORD_BITS[descriptor.codebook as usize].min(longest);
        validate_segment_width(
            descriptor.section_index,
            descriptor.codeword_index,
            used,
            width,
        )
        .expect("HCR MAX_CODEWORD_BITS must bound every encoder codeword");
        writer.write(0, width - used);
    }
    Ok((
        HcrSideInfo {
            reordered_spectral_bits: total,
            longest_codeword_bits: longest,
            bits_read: 20,
        },
        writer.finish(),
    ))
}

fn validate_segment_width(
    section: usize,
    codeword: usize,
    used: usize,
    width: usize,
) -> Result<(), HcrError> {
    if used > width {
        return Err(HcrError::SegmentOverrun {
            section,
            codeword,
            consumed_bits: used,
            segment_bits: width,
        });
    }
    Ok(())
}

pub fn codewords_to_spectral_data(
    ics: &IcsInfo,
    sections: &[HcrSection],
    codewords: &[HcrDecodedCodeword],
    granule_length: usize,
) -> Result<SpectralData, HcrError> {
    let windows = ics
        .window_group_lengths
        .iter()
        .map(|&length| length as usize)
        .sum::<usize>();
    let total_lines = windows
        .checked_mul(granule_length)
        .ok_or(HcrError::LayoutMismatch)?;
    let mut reordered = vec![0i32; total_lines];
    let mut destination = 0usize;
    for (section_index, section) in sections.iter().copied().enumerate() {
        let end = destination
            .checked_add(section.spectral_lines)
            .ok_or(HcrError::LayoutMismatch)?;
        if end > reordered.len() {
            return Err(HcrError::LayoutMismatch);
        }
        if section.codebook != 0 {
            let dimension = 1usize << CODEBOOK_DIMENSION_SHIFT[section.codebook as usize];
            let expected = section.spectral_lines / dimension;
            for codeword_index in 0..expected {
                let word = codewords
                    .iter()
                    .find(|word| {
                        word.section_index == section_index && word.codeword_index == codeword_index
                    })
                    .ok_or(HcrError::MissingCodeword {
                        section: section_index,
                        codeword: codeword_index,
                    })?;
                if word.coefficients.len() != dimension {
                    return Err(HcrError::CoefficientDimensionMismatch {
                        section: section_index,
                        codeword: codeword_index,
                        expected: dimension,
                        actual: word.coefficients.len(),
                    });
                }
                let maximum = LARGEST_ABSOLUTE_VALUE[section.codebook as usize];
                if let Some(&value) = word
                    .coefficients
                    .iter()
                    .find(|&&value| value.abs() > maximum)
                {
                    return Err(HcrError::LargestAbsoluteValueExceeded {
                        codebook: section.codebook,
                        value,
                        maximum,
                    });
                }
                let offset = destination + codeword_index * dimension;
                reordered[offset..offset + dimension].copy_from_slice(&word.coefficients);
            }
        }
        destination = end;
    }

    if ics.window_sequence != WindowSequence::EightShort {
        return Ok(SpectralData {
            windows: vec![reordered],
        });
    }
    if windows != 8 || granule_length != 128 || reordered.len() != 1024 {
        return Err(HcrError::LayoutMismatch);
    }
    let mut output = vec![vec![0i32; granule_length]; windows];
    for (window, spectrum) in output.iter_mut().enumerate() {
        let mut out = 0usize;
        for unit_group in 0..32 {
            let source = window * 4 + unit_group * 32;
            spectrum[out..out + 4].copy_from_slice(&reordered[source..source + 4]);
            out += 4;
        }
    }
    Ok(SpectralData { windows: output })
}

#[derive(Debug)]
struct PendingCodeword {
    descriptor: HcrDecodedCodeword,
    bits: Vec<bool>,
    complete: bool,
}

/// Decodes all reordered HCR codewords, including non-priority sets whose
/// codewords may continue across multiple segments.
pub fn decode_reordered_codewords(
    payload: &[u8],
    side_info: &HcrSideInfo,
    sections: &[HcrSection],
) -> Result<Vec<HcrDecodedCodeword>, HcrError> {
    if payload.len().saturating_mul(8) < side_info.reordered_spectral_bits {
        return Err(HcrError::PayloadTooShort {
            needed_bits: side_info.reordered_spectral_bits,
            available_bits: payload.len().saturating_mul(8),
        });
    }
    let segments = prepare_segmentation_grid(side_info, sections)?;
    if segments.is_empty() {
        return Ok(Vec::new());
    }
    let descriptors = sorted_codeword_descriptors(sections)?;
    let num_segments = segments.len();
    let mut states: Vec<_> = segments
        .iter()
        .copied()
        .map(HcrSegmentState::from_segment)
        .collect();
    let mut result = Vec::with_capacity(descriptors.len());

    for (index, descriptor) in descriptors.iter().take(num_segments).enumerate() {
        let (coefficients, _) = decode_segment_codeword(
            payload,
            &mut states[index],
            descriptor.codebook,
            HcrReadDirection::LeftToRight,
        )?;
        result.push(HcrDecodedCodeword {
            section_index: descriptor.section_index,
            codeword_index: descriptor.codeword_index,
            coefficients,
        });
    }

    let mut direction = HcrReadDirection::RightToLeft;
    for set in descriptors[num_segments.min(descriptors.len())..].chunks(num_segments) {
        let mut pending: Vec<_> = set
            .iter()
            .map(|descriptor| PendingCodeword {
                descriptor: HcrDecodedCodeword {
                    section_index: descriptor.section_index,
                    codeword_index: descriptor.codeword_index,
                    coefficients: Vec::new(),
                },
                bits: Vec::new(),
                complete: false,
            })
            .collect();
        for trial in 0..num_segments {
            for segment_index in 0..num_segments {
                let codeword_index = (segment_index + num_segments - trial) % num_segments;
                let Some(codeword) = pending.get_mut(codeword_index) else {
                    continue;
                };
                while !codeword.complete && states[segment_index].remaining_bits() != 0 {
                    codeword.bits.push(take_segment_bit(
                        payload,
                        &mut states[segment_index],
                        direction,
                    )?);
                    match try_decode_exact(&codeword.bits, set[codeword_index].codebook)? {
                        Some((coefficients, consumed)) => {
                            let pushed_back = codeword.bits.len() - consumed;
                            restore_segment_bits(
                                &mut states[segment_index],
                                direction,
                                pushed_back,
                            );
                            codeword.bits.truncate(consumed);
                            codeword.descriptor.coefficients = coefficients;
                            codeword.complete = true;
                        }
                        None => continue,
                    }
                }
            }
        }
        for codeword in pending {
            if !codeword.complete {
                return Err(HcrError::IncompleteCodeword {
                    section: codeword.descriptor.section_index,
                    codeword: codeword.descriptor.codeword_index,
                    accumulated_bits: codeword.bits.len(),
                });
            }
            result.push(codeword.descriptor);
        }
        direction = match direction {
            HcrReadDirection::LeftToRight => HcrReadDirection::RightToLeft,
            HcrReadDirection::RightToLeft => HcrReadDirection::LeftToRight,
        };
    }
    result.sort_by_key(|word| (word.section_index, word.codeword_index));
    Ok(result)
}

#[derive(Debug, Clone, Copy)]
struct CodewordDescriptor {
    section_index: usize,
    codeword_index: usize,
    codebook: u8,
}

fn sorted_codeword_descriptors(
    sections: &[HcrSection],
) -> Result<Vec<CodewordDescriptor>, HcrError> {
    let mut sorted = Vec::new();
    for (section_index, section) in sections.iter().copied().enumerate() {
        let codebook = section.codebook as usize;
        if codebook >= CODEBOOK_PRIORITY.len() {
            return Err(HcrError::InvalidCodebook(section.codebook));
        }
        if CODEBOOK_PRIORITY[codebook] == 0 {
            continue;
        }
        let dimension = 1usize << CODEBOOK_DIMENSION_SHIFT[codebook];
        if section.spectral_lines == 0 || section.spectral_lines % dimension != 0 {
            return Err(HcrError::InvalidSectionLineCount {
                section: section_index,
                spectral_lines: section.spectral_lines,
                dimension,
            });
        }
        for codeword_index in 0..section.spectral_lines / dimension {
            sorted.push((
                CODEBOOK_PRIORITY[codebook],
                CodewordDescriptor {
                    section_index,
                    codeword_index,
                    codebook: section.codebook,
                },
            ));
        }
    }
    sorted.sort_by_key(|&(priority, descriptor)| {
        (
            std::cmp::Reverse(priority),
            descriptor.section_index,
            descriptor.codeword_index,
        )
    });
    Ok(sorted
        .into_iter()
        .map(|(_, descriptor)| descriptor)
        .collect())
}

fn take_segment_bit(
    payload: &[u8],
    state: &mut HcrSegmentState,
    direction: HcrReadDirection,
) -> Result<bool, HcrError> {
    if state.remaining_bits() == 0 {
        return Err(HcrError::EmptySegment);
    }
    let index = match direction {
        HcrReadDirection::LeftToRight => state.left_bit,
        HcrReadDirection::RightToLeft => state.right_bit,
    };
    if index >= payload.len().saturating_mul(8) {
        return Err(HcrError::PayloadTooShort {
            needed_bits: index + 1,
            available_bits: payload.len().saturating_mul(8),
        });
    }
    match direction {
        HcrReadDirection::LeftToRight => state.left_bit += 1,
        HcrReadDirection::RightToLeft => state.right_bit = state.right_bit.saturating_sub(1),
    }
    state.remaining_bits -= 1;
    Ok(((payload[index / 8] >> (7 - (index % 8))) & 1) != 0)
}

fn restore_segment_bits(state: &mut HcrSegmentState, direction: HcrReadDirection, count: usize) {
    match direction {
        HcrReadDirection::LeftToRight => state.left_bit -= count,
        HcrReadDirection::RightToLeft => state.right_bit += count,
    }
    state.remaining_bits += count;
}

fn try_decode_exact(bits: &[bool], codebook: u8) -> Result<Option<(Vec<i32>, usize)>, HcrError> {
    let mut packed = vec![0u8; bits.len().div_ceil(8)];
    for (index, &bit) in bits.iter().enumerate() {
        if bit {
            packed[index / 8] |= 1 << (7 - (index % 8));
        }
    }
    let mut reader = BitReader::with_bit_len(&packed, bits.len())?;
    match decode_spectral_tuple(&mut reader, codebook) {
        Ok(coefficients) => Ok(Some((coefficients, reader.bits_read()))),
        Err(error) if spectral_is_unexpected_eof(&error) => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn spectral_is_unexpected_eof(error: &SpectralError) -> bool {
    matches!(
        error,
        SpectralError::Bit(BitError::UnexpectedEof { .. })
            | SpectralError::Huffman(HuffmanError::Bit(BitError::UnexpectedEof { .. }))
    )
}

/// Decodes the priority codeword at the left edge of every HCR segment.
/// Remaining segment bits are retained for the later non-priority passes.
pub fn decode_priority_codewords(
    payload: &[u8],
    side_info: &HcrSideInfo,
    sections: &[HcrSection],
) -> Result<Vec<HcrPriorityCodeword>, HcrError> {
    if payload.len().saturating_mul(8) < side_info.reordered_spectral_bits {
        return Err(HcrError::PayloadTooShort {
            needed_bits: side_info.reordered_spectral_bits,
            available_bits: payload.len().saturating_mul(8),
        });
    }
    let segments = prepare_segmentation_grid(side_info, sections)?;
    let mut decoded = Vec::with_capacity(segments.len());
    for segment in segments {
        let packed = copy_bit_range(payload, segment.left_bit, segment.width());
        let mut reader = BitReader::with_bit_len(&packed, segment.width())?;
        let coefficients = decode_spectral_tuple(&mut reader, segment.codebook)?;
        let consumed_bits = reader.bits_read();
        decoded.push(HcrPriorityCodeword {
            section_index: segment.section_index,
            codeword_index: segment.codeword_index,
            coefficients,
            consumed_bits,
            remaining_left_bit: segment.left_bit + consumed_bits,
        });
    }
    Ok(decoded)
}

fn copy_bit_range(source: &[u8], start: usize, length: usize) -> Vec<u8> {
    let mut result = vec![0u8; length.div_ceil(8)];
    for offset in 0..length {
        let bit = (source[(start + offset) / 8] >> (7 - ((start + offset) % 8))) & 1;
        result[offset / 8] |= bit << (7 - (offset % 8));
    }
    result
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HcrError {
    Bit(BitError),
    Spectral(SpectralError),
    ReorderedSpectralLengthOutOfRange {
        length: usize,
        maximum: usize,
    },
    LongestCodewordLengthOutOfRange {
        length: usize,
        maximum: usize,
    },
    InconsistentLengths {
        reordered_spectral_bits: usize,
        longest_codeword_bits: usize,
    },
    InvalidCodebook(u8),
    InvalidSectionLineCount {
        section: usize,
        spectral_lines: usize,
        dimension: usize,
    },
    PayloadTooShort {
        needed_bits: usize,
        available_bits: usize,
    },
    SegmentOverrun {
        section: usize,
        codeword: usize,
        consumed_bits: usize,
        segment_bits: usize,
    },
    LayoutMismatch,
    ShortBandNotFourLineAligned {
        band: usize,
        width: usize,
    },
    EmptySegment,
    SegmentReadOverrun {
        consumed_bits: usize,
        available_bits: usize,
    },
    IncompleteCodeword {
        section: usize,
        codeword: usize,
        accumulated_bits: usize,
    },
    MissingCodeword {
        section: usize,
        codeword: usize,
    },
    CoefficientDimensionMismatch {
        section: usize,
        codeword: usize,
        expected: usize,
        actual: usize,
    },
    LargestAbsoluteValueExceeded {
        codebook: u8,
        value: i32,
        maximum: i32,
    },
}

impl From<BitError> for HcrError {
    fn from(value: BitError) -> Self {
        Self::Bit(value)
    }
}

impl From<SpectralError> for HcrError {
    fn from(value: SpectralError) -> Self {
        Self::Spectral(value)
    }
}

impl fmt::Display for HcrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bit(error) => error.fmt(f),
            Self::Spectral(error) => error.fmt(f),
            Self::ReorderedSpectralLengthOutOfRange { length, maximum } => write!(
                f,
                "HCR reordered spectral length {length} exceeds {maximum} bits"
            ),
            Self::LongestCodewordLengthOutOfRange { length, maximum } => write!(
                f,
                "HCR longest codeword length {length} exceeds {maximum} bits"
            ),
            Self::InconsistentLengths {
                reordered_spectral_bits,
                longest_codeword_bits,
            } => write!(
                f,
                "HCR longest codeword ({longest_codeword_bits} bits) exceeds reordered payload ({reordered_spectral_bits} bits)"
            ),
            Self::InvalidCodebook(codebook) => write!(f, "invalid HCR codebook {codebook}"),
            Self::InvalidSectionLineCount {
                section,
                spectral_lines,
                dimension,
            } => write!(
                f,
                "HCR section {section} has {spectral_lines} lines, not divisible by codebook dimension {dimension}"
            ),
            Self::PayloadTooShort {
                needed_bits,
                available_bits,
            } => write!(
                f,
                "HCR payload needs {needed_bits} bits, only {available_bits} are available"
            ),
            Self::SegmentOverrun {
                section,
                codeword,
                consumed_bits,
                segment_bits,
            } => write!(
                f,
                "HCR section {section} codeword {codeword} consumed {consumed_bits} bits from a {segment_bits}-bit segment"
            ),
            Self::LayoutMismatch => write!(f, "HCR section layout does not match ICS data"),
            Self::ShortBandNotFourLineAligned { band, width } => write!(
                f,
                "HCR short-window band {band} has {width} lines, not a multiple of four"
            ),
            Self::EmptySegment => write!(f, "cannot decode an empty HCR segment"),
            Self::SegmentReadOverrun {
                consumed_bits,
                available_bits,
            } => write!(
                f,
                "HCR codeword consumed {consumed_bits} bits with only {available_bits} segment bits available"
            ),
            Self::IncompleteCodeword {
                section,
                codeword,
                accumulated_bits,
            } => write!(
                f,
                "incomplete HCR section {section} codeword {codeword} after {accumulated_bits} bits"
            ),
            Self::MissingCodeword { section, codeword } => {
                write!(f, "missing HCR section {section} codeword {codeword}")
            }
            Self::CoefficientDimensionMismatch {
                section,
                codeword,
                expected,
                actual,
            } => write!(
                f,
                "HCR section {section} codeword {codeword} has {actual} coefficients, expected {expected}"
            ),
            Self::LargestAbsoluteValueExceeded {
                codebook,
                value,
                maximum,
            } => write!(
                f,
                "HCR codebook {codebook} decoded value {value} beyond largest absolute value {maximum}"
            ),
        }
    }
}

impl std::error::Error for HcrError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitWriter;
    use crate::ics::WindowShape;
    use crate::section::Section;

    #[test]
    fn converts_and_formats_every_hcr_error_variant() {
        let bit = BitError::UnexpectedEof {
            needed_bits: 1,
            remaining_bits: 0,
        };
        let spectral = SpectralError::InvalidBandOffsets;
        assert_eq!(HcrError::from(bit.clone()), HcrError::Bit(bit));
        assert_eq!(
            HcrError::from(spectral.clone()),
            HcrError::Spectral(spectral)
        );
        let errors = [
            HcrError::Bit(BitError::UnexpectedEof {
                needed_bits: 1,
                remaining_bits: 0,
            }),
            HcrError::Spectral(SpectralError::InvalidBandOffsets),
            HcrError::ReorderedSpectralLengthOutOfRange {
                length: 2,
                maximum: 1,
            },
            HcrError::LongestCodewordLengthOutOfRange {
                length: 2,
                maximum: 1,
            },
            HcrError::InconsistentLengths {
                reordered_spectral_bits: 1,
                longest_codeword_bits: 2,
            },
            HcrError::InvalidCodebook(12),
            HcrError::InvalidSectionLineCount {
                section: 0,
                spectral_lines: 3,
                dimension: 2,
            },
            HcrError::PayloadTooShort {
                needed_bits: 2,
                available_bits: 1,
            },
            HcrError::SegmentOverrun {
                section: 0,
                codeword: 1,
                consumed_bits: 2,
                segment_bits: 1,
            },
            HcrError::LayoutMismatch,
            HcrError::ShortBandNotFourLineAligned { band: 0, width: 3 },
            HcrError::EmptySegment,
            HcrError::SegmentReadOverrun {
                consumed_bits: 2,
                available_bits: 1,
            },
            HcrError::IncompleteCodeword {
                section: 0,
                codeword: 1,
                accumulated_bits: 2,
            },
            HcrError::MissingCodeword {
                section: 0,
                codeword: 1,
            },
            HcrError::CoefficientDimensionMismatch {
                section: 0,
                codeword: 1,
                expected: 4,
                actual: 2,
            },
            HcrError::LargestAbsoluteValueExceeded {
                codebook: 16,
                value: 17,
                maximum: 16,
            },
        ];
        for error in errors {
            assert!(!error.to_string().is_empty());
        }
    }

    #[test]
    fn segment_bit_helpers_cover_both_directions_and_boundaries() {
        let mut state = HcrSegmentState {
            left_bit: 0,
            right_bit: 3,
            remaining_bits: 4,
        };
        assert!(
            take_segment_bit(&[0b1001_0000], &mut state, HcrReadDirection::LeftToRight).unwrap()
        );
        assert_eq!(
            (state.left_bit, state.right_bit, state.remaining_bits()),
            (1, 3, 3)
        );
        restore_segment_bits(&mut state, HcrReadDirection::LeftToRight, 1);
        assert_eq!(
            (state.left_bit, state.right_bit, state.remaining_bits()),
            (0, 3, 4)
        );

        assert!(
            take_segment_bit(&[0b1001_0000], &mut state, HcrReadDirection::RightToLeft).unwrap()
        );
        assert_eq!(
            (state.left_bit, state.right_bit, state.remaining_bits()),
            (0, 2, 3)
        );
        restore_segment_bits(&mut state, HcrReadDirection::RightToLeft, 1);
        assert_eq!(
            (state.left_bit, state.right_bit, state.remaining_bits()),
            (0, 3, 4)
        );

        let mut empty = HcrSegmentState {
            left_bit: 0,
            right_bit: 0,
            remaining_bits: 0,
        };
        assert_eq!(
            take_segment_bit(&[0], &mut empty, HcrReadDirection::LeftToRight),
            Err(HcrError::EmptySegment)
        );
        assert_eq!(
            decode_segment_codeword(&[0], &mut empty, 1, HcrReadDirection::LeftToRight),
            Err(HcrError::EmptySegment)
        );

        let mut beyond_payload = HcrSegmentState {
            left_bit: 8,
            right_bit: 8,
            remaining_bits: 1,
        };
        let too_short = HcrError::PayloadTooShort {
            needed_bits: 9,
            available_bits: 8,
        };
        assert_eq!(
            take_segment_bit(&[0], &mut beyond_payload, HcrReadDirection::LeftToRight),
            Err(too_short.clone())
        );
        assert_eq!(
            copy_directional_bits(&[0], beyond_payload, HcrReadDirection::RightToLeft),
            Err(too_short)
        );
    }

    #[test]
    fn exact_decoder_distinguishes_incomplete_and_invalid_codewords() {
        assert_eq!(try_decode_exact(&[], 1).unwrap(), None);
        assert_eq!(try_decode_exact(&[true], 1).unwrap(), None);
        assert!(spectral_is_unexpected_eof(&SpectralError::Bit(
            BitError::UnexpectedEof {
                needed_bits: 1,
                remaining_bits: 0,
            }
        )));
        assert!(spectral_is_unexpected_eof(&SpectralError::Huffman(
            HuffmanError::Bit(BitError::UnexpectedEof {
                needed_bits: 1,
                remaining_bits: 0,
            })
        )));
        assert!(!spectral_is_unexpected_eof(
            &SpectralError::InvalidBandOffsets
        ));
        assert!(matches!(
            try_decode_exact(&[], 12),
            Err(HcrError::Spectral(SpectralError::Huffman(
                HuffmanError::InvalidCodebook(12)
            )))
        ));
    }

    #[test]
    fn public_hcr_entry_points_reject_invalid_layouts_and_short_payloads() {
        let side = HcrSideInfo {
            reordered_spectral_bits: 9,
            longest_codeword_bits: 9,
            bits_read: 20,
        };
        let valid = [HcrSection {
            codebook: 1,
            spectral_lines: 4,
        }];
        let short = HcrError::PayloadTooShort {
            needed_bits: 9,
            available_bits: 8,
        };
        assert_eq!(
            decode_reordered_codewords(&[0], &side, &valid),
            Err(short.clone())
        );
        assert_eq!(decode_priority_codewords(&[0], &side, &valid), Err(short));

        for sections in [
            vec![HcrSection {
                codebook: 32,
                spectral_lines: 4,
            }],
            vec![HcrSection {
                codebook: 1,
                spectral_lines: 3,
            }],
        ] {
            assert!(prepare_segmentation_grid(&side, &sections).is_err());
            assert!(encode_reordered_codewords(&sections, &[]).is_err());
        }

        let zero_width = prepare_segmentation_grid(
            &HcrSideInfo {
                reordered_spectral_bits: 0,
                longest_codeword_bits: 0,
                bits_read: 20,
            },
            &valid,
        )
        .unwrap();
        assert!(zero_width.is_empty());
        assert!(encode_reordered_codewords(
            &[HcrSection {
                codebook: 0,
                spectral_lines: 4,
            }],
            &[]
        )
        .unwrap()
        .1
        .is_empty());

        assert!(matches!(
            encode_reordered_codewords(
                &[HcrSection {
                    codebook: 11,
                    spectral_lines: 504,
                }],
                &[]
            ),
            Err(HcrError::ReorderedSpectralLengthOutOfRange { .. })
        ));

        assert!(matches!(
            encode_reordered_codewords(
                &[HcrSection {
                    codebook: 1,
                    spectral_lines: 4,
                }],
                &[HcrDecodedCodeword {
                    section_index: 0,
                    codeword_index: 0,
                    coefficients: Vec::new(),
                }],
            ),
            Err(HcrError::Spectral(SpectralError::Huffman(_)))
        ));
    }

    #[test]
    fn segment_codeword_decoder_advances_from_the_right_edge() {
        let mut state = HcrSegmentState {
            left_bit: 0,
            right_bit: 10,
            remaining_bits: 11,
        };
        let (coefficients, consumed) =
            decode_segment_codeword(&[0; 2], &mut state, 1, HcrReadDirection::RightToLeft).unwrap();
        assert_eq!(coefficients, [0; 4]);
        assert!(consumed > 0);
        assert_eq!(state.right_bit, 10 - consumed);
        assert_eq!(state.remaining_bits(), 11 - consumed);
    }

    fn payload(reordered: u16, longest: u8) -> Vec<u8> {
        let mut writer = BitWriter::new();
        writer.write(reordered as u32, 14);
        writer.write(longest as u32, 6);
        writer.finish()
    }

    #[test]
    fn parses_valid_side_info() {
        let bytes = payload(6144, 49);
        let mut reader = BitReader::new(&bytes);
        let side = HcrSideInfo::parse(&mut reader, HcrElementType::SingleChannel).unwrap();
        assert_eq!(side.reordered_spectral_bits, 6144);
        assert_eq!(side.longest_codeword_bits, 49);
        assert_eq!(side.bits_read, 20);
    }

    #[test]
    fn permits_cpe_length_above_single_channel_limit() {
        let bytes = payload(12288, 20);
        let mut reader = BitReader::new(&bytes);
        assert!(HcrSideInfo::parse(&mut reader, HcrElementType::ChannelPair).is_ok());
        let mut reader = BitReader::new(&bytes);
        assert!(matches!(
            HcrSideInfo::parse(&mut reader, HcrElementType::SingleChannel),
            Err(HcrError::ReorderedSpectralLengthOutOfRange { .. })
        ));
    }

    #[test]
    fn rejects_inconsistent_lengths() {
        let bytes = payload(12, 13);
        let mut reader = BitReader::new(&bytes);
        assert!(matches!(
            HcrSideInfo::parse(&mut reader, HcrElementType::SingleChannel),
            Err(HcrError::InconsistentLengths { .. })
        ));
    }

    #[test]
    fn prepares_priority_ordered_segmentation_grid() {
        let side = HcrSideInfo {
            reordered_spectral_bits: 38,
            longest_codeword_bits: 20,
            bits_read: 20,
        };
        let sections = [
            HcrSection {
                codebook: 1,
                spectral_lines: 4,
            },
            HcrSection {
                codebook: 3,
                spectral_lines: 8,
            },
        ];
        let segments = prepare_segmentation_grid(&side, &sections).unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].codebook, 3);
        assert_eq!(segments[0].width(), 38);
    }

    #[test]
    fn skips_zero_codebook_sections() {
        let side = HcrSideInfo {
            reordered_spectral_bits: 11,
            longest_codeword_bits: 11,
            bits_read: 20,
        };
        let sections = [
            HcrSection {
                codebook: 0,
                spectral_lines: 8,
            },
            HcrSection {
                codebook: 1,
                spectral_lines: 4,
            },
        ];
        let segments = prepare_segmentation_grid(&side, &sections).unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].codebook, 1);
    }

    #[test]
    fn decodes_priority_word_from_segment_left_edge() {
        let side = HcrSideInfo {
            reordered_spectral_bits: 11,
            longest_codeword_bits: 11,
            bits_read: 20,
        };
        let sections = [HcrSection {
            codebook: 1,
            spectral_lines: 4,
        }];
        // Codebook 1's all-zero body decodes without sign bits.
        let decoded = decode_priority_codewords(&[0; 2], &side, &sections).unwrap();
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].section_index, 0);
        assert_eq!(decoded[0].coefficients.len(), 4);
        assert!(decoded[0].consumed_bits <= 11);
    }

    #[test]
    fn builds_long_sections_from_sfb_widths() {
        let ics = IcsInfo {
            window_sequence: WindowSequence::OnlyLong,
            window_shape: WindowShape::Sine,
            max_sfb: 2,
            total_sfb: 2,
            predictor_data_present: false,
            scale_factor_grouping: 0,
            window_group_lengths: vec![1],
            bits_read: 0,
        };
        let data = SectionData {
            sections: vec![Section {
                group: 0,
                start_sfb: 0,
                end_sfb: 2,
                codebook: 5,
            }],
            codebooks: vec![vec![5, 5]],
            bits_read: 0,
        };
        assert_eq!(
            sections_from_ics(&ics, &data, &[0, 4, 12]).unwrap(),
            [HcrSection {
                codebook: 5,
                spectral_lines: 12
            }]
        );
    }

    #[test]
    fn builds_short_sections_in_unit_window_order() {
        let ics = IcsInfo {
            window_sequence: WindowSequence::EightShort,
            window_shape: WindowShape::Sine,
            max_sfb: 1,
            total_sfb: 1,
            predictor_data_present: false,
            scale_factor_grouping: 0,
            window_group_lengths: vec![2, 1],
            bits_read: 0,
        };
        let data = SectionData {
            sections: Vec::new(),
            codebooks: vec![vec![1], vec![3]],
            bits_read: 0,
        };
        assert_eq!(
            sections_from_ics(&ics, &data, &[0, 8]).unwrap(),
            [
                HcrSection {
                    codebook: 1,
                    spectral_lines: 8
                },
                HcrSection {
                    codebook: 3,
                    spectral_lines: 4
                },
                HcrSection {
                    codebook: 1,
                    spectral_lines: 8
                },
                HcrSection {
                    codebook: 3,
                    spectral_lines: 4
                }
            ]
        );
    }

    #[test]
    fn exposes_segment_bits_in_both_directions() {
        let state = HcrSegmentState {
            left_bit: 0,
            right_bit: 3,
            remaining_bits: 4,
        };
        assert_eq!(
            copy_directional_bits(&[0b1011_0000], state, HcrReadDirection::LeftToRight).unwrap(),
            [0b1011_0000]
        );
        assert_eq!(
            copy_directional_bits(&[0b1011_0000], state, HcrReadDirection::RightToLeft).unwrap(),
            [0b1101_0000]
        );
    }

    #[test]
    fn decodes_non_priority_word_from_opposite_segment_edge() {
        let side = HcrSideInfo {
            reordered_spectral_bits: 11,
            longest_codeword_bits: 11,
            bits_read: 20,
        };
        let sections = [HcrSection {
            codebook: 1,
            spectral_lines: 8,
        }];
        let decoded = decode_reordered_codewords(&[0; 2], &side, &sections).unwrap();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].codeword_index, 0);
        assert_eq!(decoded[1].codeword_index, 1);
        assert_eq!(decoded[0].coefficients, decoded[1].coefficients);
    }

    #[test]
    fn decodes_multiple_non_priority_sets_and_reports_incomplete_words() {
        let side = HcrSideInfo {
            reordered_spectral_bits: 22,
            longest_codeword_bits: 11,
            bits_read: 20,
        };
        let sections = [HcrSection {
            codebook: 1,
            spectral_lines: 20,
        }];
        let decoded = decode_reordered_codewords(&[0; 3], &side, &sections).unwrap();
        assert_eq!(decoded.len(), 5);
        assert!(decoded.iter().all(|word| word.coefficients == [0; 4]));

        let too_narrow = HcrSideInfo {
            reordered_spectral_bits: 1,
            longest_codeword_bits: 1,
            bits_read: 20,
        };
        // Codebook 1's all-zero tuple is exactly one bit, so this bounded HCR
        // segment is complete even though the table decoder normally performs
        // a two-bit look-ahead.
        let one_bit = decode_reordered_codewords(
            &[0],
            &too_narrow,
            &[HcrSection {
                codebook: 1,
                spectral_lines: 4,
            }],
        )
        .unwrap();
        assert_eq!(one_bit.len(), 1);
        assert_eq!(one_bit[0].coefficients, [0; 4]);

        let narrow = HcrSideInfo {
            reordered_spectral_bits: 2,
            longest_codeword_bits: 2,
            bits_read: 20,
        };
        let two_one_bit = decode_reordered_codewords(
            &[0],
            &narrow,
            &[HcrSection {
                codebook: 1,
                spectral_lines: 8,
            }],
        )
        .unwrap();
        assert_eq!(two_one_bit.len(), 2);

        assert!(matches!(
            decode_reordered_codewords(
                &[0],
                &too_narrow,
                &[HcrSection {
                    codebook: 11,
                    spectral_lines: 2,
                }],
            ),
            Err(HcrError::Spectral(_))
        ));
    }

    #[test]
    fn maps_codewords_back_to_long_spectrum() {
        let ics = IcsInfo {
            window_sequence: WindowSequence::OnlyLong,
            window_shape: WindowShape::Sine,
            max_sfb: 1,
            total_sfb: 1,
            predictor_data_present: false,
            scale_factor_grouping: 0,
            window_group_lengths: vec![1],
            bits_read: 0,
        };
        let sections = [
            HcrSection {
                codebook: 1,
                spectral_lines: 4,
            },
            HcrSection {
                codebook: 0,
                spectral_lines: 4,
            },
        ];
        let words = [HcrDecodedCodeword {
            section_index: 0,
            codeword_index: 0,
            coefficients: vec![1, 0, -1, 1],
        }];
        let spectral = codewords_to_spectral_data(&ics, &sections, &words, 16).unwrap();
        assert_eq!(&spectral.windows[0][..8], &[1, 0, -1, 1, 0, 0, 0, 0]);
        assert!(spectral.windows[0][8..].iter().all(|&value| value == 0));
    }

    #[test]
    fn fixed_priority_segment_encoder_roundtrips_all_codewords() {
        let sections = [
            HcrSection {
                codebook: 1,
                spectral_lines: 8,
            },
            HcrSection {
                codebook: 7,
                spectral_lines: 4,
            },
        ];
        let words = [
            HcrDecodedCodeword {
                section_index: 0,
                codeword_index: 0,
                coefficients: vec![1, 0, -1, 1],
            },
            HcrDecodedCodeword {
                section_index: 0,
                codeword_index: 1,
                coefficients: vec![0, 1, 0, -1],
            },
            HcrDecodedCodeword {
                section_index: 1,
                codeword_index: 0,
                coefficients: vec![3, -2],
            },
            HcrDecodedCodeword {
                section_index: 1,
                codeword_index: 1,
                coefficients: vec![0, 7],
            },
        ];
        let (side, payload) = encode_reordered_codewords(&sections, &words).unwrap();
        assert_eq!(side.longest_codeword_bits, 14);
        assert_eq!(
            decode_reordered_codewords(&payload, &side, &sections).unwrap(),
            words
        );

        assert_eq!(validate_segment_width(2, 3, 49, 49), Ok(()));
        assert_eq!(
            validate_segment_width(2, 3, 50, 49),
            Err(HcrError::SegmentOverrun {
                section: 2,
                codeword: 3,
                consumed_bits: 50,
                segment_bits: 49,
            })
        );
    }

    fn short_ics() -> IcsInfo {
        IcsInfo {
            window_sequence: WindowSequence::EightShort,
            window_shape: WindowShape::Sine,
            max_sfb: 1,
            total_sfb: 1,
            predictor_data_present: false,
            scale_factor_grouping: 0,
            window_group_lengths: vec![8],
            bits_read: 0,
        }
    }

    #[test]
    fn maps_reordered_short_codewords_back_to_eight_windows() {
        let sections = [
            HcrSection {
                codebook: 1,
                spectral_lines: 4,
            },
            HcrSection {
                codebook: 0,
                spectral_lines: 1020,
            },
        ];
        let words = [HcrDecodedCodeword {
            section_index: 0,
            codeword_index: 0,
            coefficients: vec![1, -1, 0, 1],
        }];
        let spectral = codewords_to_spectral_data(&short_ics(), &sections, &words, 128).unwrap();
        assert_eq!(spectral.windows.len(), 8);
        assert_eq!(&spectral.windows[0][..4], &[1, -1, 0, 1]);
        assert!(spectral
            .windows
            .iter()
            .flatten()
            .skip(4)
            .all(|&value| value == 0));
    }

    #[test]
    fn spectral_mapping_rejects_missing_dimension_lav_and_short_layouts() {
        let long = IcsInfo {
            window_sequence: WindowSequence::OnlyLong,
            window_shape: WindowShape::Sine,
            max_sfb: 1,
            total_sfb: 1,
            predictor_data_present: false,
            scale_factor_grouping: 0,
            window_group_lengths: vec![1],
            bits_read: 0,
        };
        let section = [HcrSection {
            codebook: 1,
            spectral_lines: 4,
        }];
        assert!(matches!(
            codewords_to_spectral_data(&long, &section, &[], 4),
            Err(HcrError::MissingCodeword { .. })
        ));
        let wrong_dimension = [HcrDecodedCodeword {
            section_index: 0,
            codeword_index: 0,
            coefficients: vec![0; 2],
        }];
        assert!(matches!(
            codewords_to_spectral_data(&long, &section, &wrong_dimension, 4),
            Err(HcrError::CoefficientDimensionMismatch { .. })
        ));
        let excessive = [HcrDecodedCodeword {
            section_index: 0,
            codeword_index: 0,
            coefficients: vec![2, 0, 0, 0],
        }];
        assert!(matches!(
            codewords_to_spectral_data(&long, &section, &excessive, 4),
            Err(HcrError::LargestAbsoluteValueExceeded { .. })
        ));
        assert_eq!(
            codewords_to_spectral_data(&short_ics(), &[], &[], 64),
            Err(HcrError::LayoutMismatch)
        );
        assert_eq!(
            codewords_to_spectral_data(
                &long,
                &[HcrSection {
                    codebook: 0,
                    spectral_lines: 5
                }],
                &[],
                4
            ),
            Err(HcrError::LayoutMismatch)
        );
    }

    #[test]
    fn section_builder_validates_layout_alignment_and_codebook_mapping() {
        let mut ics = short_ics();
        let data = SectionData {
            sections: Vec::new(),
            codebooks: vec![vec![12]],
            bits_read: 0,
        };
        let sections = sections_from_ics(&ics, &data, &[0, 4]).unwrap();
        assert_eq!(
            sections,
            [HcrSection {
                codebook: 0,
                spectral_lines: 32
            }]
        );
        assert!(matches!(
            sections_from_ics(&ics, &data, &[0, 3]),
            Err(HcrError::ShortBandNotFourLineAligned { .. })
        ));
        ics.max_sfb = 2;
        assert_eq!(
            sections_from_ics(&ics, &data, &[0, 4]),
            Err(HcrError::LayoutMismatch)
        );

        let long = IcsInfo {
            window_sequence: WindowSequence::OnlyLong,
            window_shape: WindowShape::Sine,
            max_sfb: 1,
            total_sfb: 1,
            predictor_data_present: false,
            scale_factor_grouping: 0,
            window_group_lengths: vec![1],
            bits_read: 0,
        };
        let invalid_section = SectionData {
            sections: vec![Section {
                group: 0,
                start_sfb: 1,
                end_sfb: 1,
                codebook: 1,
            }],
            codebooks: vec![vec![1]],
            bits_read: 0,
        };
        assert_eq!(
            sections_from_ics(&long, &invalid_section, &[0, 4]),
            Err(HcrError::LayoutMismatch)
        );
    }

    #[test]
    fn empty_encoder_and_side_info_boundaries_are_explicit() {
        let (side, encoded) = encode_reordered_codewords(&[], &[]).unwrap();
        assert_eq!(side.reordered_spectral_bits, 0);
        assert_eq!(side.longest_codeword_bits, 0);
        assert!(encoded.is_empty());
        assert_eq!(
            decode_reordered_codewords(
                &[],
                &HcrSideInfo {
                    reordered_spectral_bits: 0,
                    longest_codeword_bits: 0,
                    bits_read: 20,
                },
                &[],
            )
            .unwrap(),
            []
        );

        let bytes = payload(64, 63);
        assert!(matches!(
            HcrSideInfo::parse(&mut BitReader::new(&bytes), HcrElementType::SingleChannel),
            Err(HcrError::LongestCodewordLengthOutOfRange { .. })
        ));
        let side = HcrSideInfo {
            reordered_spectral_bits: 9,
            longest_codeword_bits: 9,
            bits_read: 20,
        };
        assert!(matches!(
            side.read_payload(&mut BitReader::new(&[0xff])),
            Err(HcrError::Bit(BitError::UnexpectedEof { .. }))
        ));
        let side = HcrSideInfo {
            reordered_spectral_bits: 5,
            longest_codeword_bits: 5,
            bits_read: 20,
        };
        assert_eq!(
            side.read_payload(&mut BitReader::new(&[0b1010_1000]))
                .unwrap(),
            [0b1010_1000]
        );
    }
}
