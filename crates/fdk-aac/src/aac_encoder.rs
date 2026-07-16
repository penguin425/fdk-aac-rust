//! Pure Rust AAC-LC encoder analysis foundation.
//!
//! This module deliberately contains no FDK FFI. It produces the windowed
//! MDCT spectrum and transient metric consumed by psychoacoustic analysis and
//! quantization in the subsequent encoder stages.

use std::fmt;

use crate::adif::{AdifError, AdifHeader};
use crate::adts::{sample_rate_from_index, AdtsError, AdtsHeader, MpegVersion};
use crate::asc::{
    AudioSpecificConfig, EldExtension, EldSpecificConfig, GaSpecificConfig, LdSbrHeader,
};
use crate::bits::BitWriter;
use crate::drm::drm_crc8_bits;
use crate::eld_analysis::{EldAnalysisError, EldAnalysisFilterbank};
use crate::hcr::{
    encode_reordered_codewords, HcrDecodedCodeword, HcrError, HcrSection, HcrSideInfo,
    SCE_MAX_REORDERED_SPECTRAL_BITS,
};
use crate::huffman::{
    spectral_codebook, spectral_tuple_bit_cost, write_fdk_huffman_word, write_spectral_tuple,
    HuffmanError, HUFFMAN_CODEBOOK_SCL,
};
use crate::ics::{IcsInfo, WindowSequence, WindowShape};
use crate::latm::{LatmAacLcWriter, LatmError};
use crate::ld_sbr::LdSbrFrequencyTables;
use crate::loas::{write_loas_frame, LoasError};
use crate::ps_encoder::{analyze_ps_qmf, PsEncoderError};
use crate::raw::ElementId;
use crate::sac::{Sac212Encoder, SacEncodeError};
#[cfg(test)]
use crate::sbr_encoder::LowDelayPrequantDebug;
use crate::sbr_encoder::{
    LowDelaySbrCodingState, SbrEncoderAnalysis, SbrEncoderAnalysisFrame, SbrEncoderError,
};
use crate::sfb::{aac_band_offsets_for_ics, aac_sfb_info_for_frame, SfbError};

#[derive(Debug, Clone, PartialEq)]
pub struct AacLcAnalysisFrame {
    pub spectrum: Vec<f32>,
    pub short_spectra: Option<Vec<Vec<f32>>>,
    pub short_window_group_lengths: Vec<u8>,
    /// Ratio of the maximum eight-sample energy to the mean eight-sample
    /// energy. This is the block-switching input, not a final window decision.
    pub transient_ratio: f32,
}

#[derive(Debug, Clone)]
pub struct AacLcAnalysisFilterbank {
    frame_length: usize,
    previous: Vec<f32>,
    window: Vec<f64>,
    kernel: Vec<f64>,
    short_window: Vec<f64>,
    short_kernel: Vec<f64>,
    low_overlap_window: Vec<f64>,
    previous_window_shape: WindowShape,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PsychoacousticBand {
    pub energy: f32,
    pub masking_threshold: f32,
    /// 0 is noise-like and 1 is tonal.
    pub tonality: f32,
    /// FDK form factor: sum of square roots of absolute MDCT coefficients.
    pub form_factor: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PsychoacousticAnalysis {
    pub bands: Vec<PsychoacousticBand>,
    pub perceptual_entropy: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedSfb {
    pub scalefactor: i16,
    pub coefficients: Vec<i32>,
    pub noise_energy: f32,
    pub estimated_bits: usize,
    pub codebook: u8,
    pub codebook_bit_costs: [Option<usize>; 12],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AacLcSection {
    pub codebook: u8,
    pub start_sfb: usize,
    pub end_sfb: usize,
    pub spectral_bits: usize,
    pub side_information_bits: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedAacLcFrame {
    pub global_gain: u8,
    pub bands: Vec<QuantizedSfb>,
    pub estimated_spectral_bits: usize,
    pub estimated_section_bits: usize,
    pub sections: Vec<AacLcSection>,
    pub masking_relaxation: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedAacLcShortFrame {
    pub global_gain: u8,
    pub group_lengths: Vec<u8>,
    pub groups: Vec<QuantizedAacLcFrame>,
    pub estimated_spectral_bits: usize,
    pub estimated_section_bits: usize,
    pub masking_relaxation: f32,
}

impl QuantizedAacLcShortFrame {
    pub fn write_sce_raw_data_block(
        &self,
        element_instance_tag: u8,
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        if self.groups.is_empty()
            || self.groups.len() != self.group_lengths.len()
            || self
                .group_lengths
                .iter()
                .map(|&length| usize::from(length))
                .sum::<usize>()
                != 8
            || self.groups.iter().any(|group| group.bands.len() > 15)
            || element_instance_tag > 15
        {
            return Err(AacLcEncoderError::InvalidRawElementLayout);
        }
        let max_sfb = self.groups[0].bands.len();
        if self.groups.iter().any(|group| group.bands.len() != max_sfb) {
            return Err(AacLcEncoderError::InvalidRawElementLayout);
        }
        self.write_sce_raw_data_block_with_sbr_fill_optional(element_instance_tag, None)
    }

    pub fn write_sce_raw_data_block_with_sbr_fill(
        &self,
        element_instance_tag: u8,
        fill_element: &[u8],
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        self.write_sce_raw_data_block_with_sbr_fill_optional(
            element_instance_tag,
            Some(fill_element),
        )
    }

    fn write_sce_raw_data_block_with_sbr_fill_optional(
        &self,
        element_instance_tag: u8,
        fill_element: Option<&[u8]>,
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        if self.groups.is_empty()
            || self.groups.len() != self.group_lengths.len()
            || self
                .group_lengths
                .iter()
                .map(|&value| usize::from(value))
                .sum::<usize>()
                != 8
            || self.groups.iter().any(|group| group.bands.len() > 15)
            || element_instance_tag > 15
        {
            return Err(AacLcEncoderError::InvalidRawElementLayout);
        }
        let max_sfb = self.groups[0].bands.len();
        if self.groups.iter().any(|group| group.bands.len() != max_sfb) {
            return Err(AacLcEncoderError::InvalidRawElementLayout);
        }
        let mut writer = BitWriter::new();
        writer.write(ElementId::SingleChannel.bits() as u32, 3);
        writer.write(element_instance_tag as u32, 4);
        write_short_channel_stream(&mut writer, self, true)?;
        if let Some(fill) = fill_element {
            writer.write(ElementId::Fill.bits() as u32, 3);
            write_packed_fill_element(&mut writer, fill)?;
        }
        writer.write(ElementId::End.bits() as u32, 3);
        writer.byte_align();
        Ok(writer.finish())
    }

    pub fn write_cpe_raw_data_block(
        left: &Self,
        right: &Self,
        element_instance_tag: u8,
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        Self::write_cpe_raw_data_block_with_sbr_fill_optional(
            left,
            right,
            element_instance_tag,
            None,
        )
    }

    fn write_cpe_raw_data_block_with_sbr_fill_optional(
        left: &Self,
        right: &Self,
        element_instance_tag: u8,
        fill_element: Option<&[u8]>,
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        if left.group_lengths != right.group_lengths
            || left.groups.len() != right.groups.len()
            || left.groups.first().map(|group| group.bands.len())
                != right.groups.first().map(|group| group.bands.len())
            || element_instance_tag > 15
        {
            return Err(AacLcEncoderError::InvalidRawElementLayout);
        }
        let max_sfb = left
            .groups
            .first()
            .ok_or(AacLcEncoderError::InvalidRawElementLayout)?
            .bands
            .len();
        let mut writer = BitWriter::new();
        writer.write(ElementId::ChannelPair.bits() as u32, 3);
        writer.write(element_instance_tag as u32, 4);
        writer.write_bool(true);
        write_short_ics(&mut writer, max_sfb, &left.group_lengths);
        writer.write(0, 2);
        write_short_channel_stream(&mut writer, left, false)?;
        write_short_channel_stream(&mut writer, right, false)?;
        if let Some(fill) = fill_element {
            writer.write(ElementId::Fill.bits() as u32, 3);
            write_packed_fill_element(&mut writer, fill)?;
        }
        writer.write(ElementId::End.bits() as u32, 3);
        writer.byte_align();
        Ok(writer.finish())
    }
}

fn write_short_ics(writer: &mut BitWriter, max_sfb: usize, group_lengths: &[u8]) {
    writer.write_bool(false);
    writer.write(WindowSequence::EightShort.bits() as u32, 2);
    writer.write_bool(false);
    writer.write(max_sfb as u32, 4);
    writer.write(short_grouping_bits(group_lengths) as u32, 7);
}

fn write_short_channel_stream(
    writer: &mut BitWriter,
    frame: &QuantizedAacLcShortFrame,
    include_ics: bool,
) -> Result<(), AacLcEncoderError> {
    writer.write(frame.global_gain as u32, 8);
    if include_ics {
        let max_sfb = frame
            .groups
            .first()
            .ok_or(AacLcEncoderError::InvalidRawElementLayout)?
            .bands
            .len();
        write_short_ics(writer, max_sfb, &frame.group_lengths);
    }
    for group in &frame.groups {
        write_sections(writer, &group.sections, true);
    }
    let mut factor = i16::from(frame.global_gain) - 100;
    for group in &frame.groups {
        write_scalefactors(writer, &group.bands, &mut factor)?;
    }
    writer.write_bool(false);
    writer.write_bool(false);
    writer.write_bool(false);
    for group in &frame.groups {
        write_spectral_bands(writer, &group.bands)?;
    }
    Ok(())
}

fn short_grouping_bits(lengths: &[u8]) -> u8 {
    let mut bits = 0u8;
    let mut boundary = 0usize;
    for &length in lengths {
        for _ in 1..length {
            bits |= 1 << (6 - boundary);
            boundary += 1;
        }
        boundary += 1;
    }
    bits
}

impl QuantizedAacLcFrame {
    /// Write one AAC-LC long-window SCE followed by `ID_END` and zero padding.
    pub fn write_sce_raw_data_block(
        &self,
        element_instance_tag: u8,
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        self.write_sce_raw_data_block_with_sequence(element_instance_tag, WindowSequence::OnlyLong)
    }

    pub fn write_sce_raw_data_block_with_sequence(
        &self,
        element_instance_tag: u8,
        sequence: WindowSequence,
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        if self.bands.len() > 63 || element_instance_tag > 15 {
            return Err(AacLcEncoderError::InvalidRawElementLayout);
        }
        if !sequence.is_long() {
            return Err(AacLcEncoderError::InvalidRawElementLayout);
        }
        self.write_sce_raw_data_block_with_sequence_and_fill(element_instance_tag, sequence, None)
    }

    pub fn write_sce_raw_data_block_with_sbr_fill(
        &self,
        element_instance_tag: u8,
        fill_element: &[u8],
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        self.write_sce_raw_data_block_with_sequence_and_fill(
            element_instance_tag,
            WindowSequence::OnlyLong,
            Some(fill_element),
        )
    }

    fn write_sce_raw_data_block_with_sequence_and_fill(
        &self,
        element_instance_tag: u8,
        sequence: WindowSequence,
        fill_element: Option<&[u8]>,
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        if self.bands.len() > 63 || element_instance_tag > 15 || !sequence.is_long() {
            return Err(AacLcEncoderError::InvalidRawElementLayout);
        }
        let mut writer = BitWriter::new();
        writer.write(ElementId::SingleChannel.bits() as u32, 3);
        writer.write(element_instance_tag as u32, 4);
        write_long_channel_stream(&mut writer, self, Some(sequence))?;
        if let Some(fill) = fill_element {
            writer.write(ElementId::Fill.bits() as u32, 3);
            write_packed_fill_element(&mut writer, fill)?;
        }
        writer.write(ElementId::End.bits() as u32, 3);
        writer.byte_align();
        Ok(writer.finish())
    }

    /// Write a common-window long-block AAC-LC CPE with no MS transform.
    pub fn write_cpe_raw_data_block(
        left: &Self,
        right: &Self,
        element_instance_tag: u8,
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        Self::write_cpe_raw_data_block_with_sequence(
            left,
            right,
            element_instance_tag,
            WindowSequence::OnlyLong,
        )
    }

    pub fn write_cpe_raw_data_block_with_sequence(
        left: &Self,
        right: &Self,
        element_instance_tag: u8,
        sequence: WindowSequence,
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        Self::write_cpe_raw_data_block_with_sequence_and_fill(
            left,
            right,
            element_instance_tag,
            sequence,
            None,
        )
    }

    fn write_cpe_raw_data_block_with_sequence_and_fill(
        left: &Self,
        right: &Self,
        element_instance_tag: u8,
        sequence: WindowSequence,
        fill_element: Option<&[u8]>,
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        if left.bands.len() != right.bands.len()
            || left.bands.len() > 63
            || element_instance_tag > 15
            || !sequence.is_long()
        {
            return Err(AacLcEncoderError::InvalidRawElementLayout);
        }
        let mut writer = BitWriter::new();
        writer.write(ElementId::ChannelPair.bits() as u32, 3);
        writer.write(element_instance_tag as u32, 4);
        writer.write_bool(true); // common_window
        write_long_ics(&mut writer, left.bands.len(), sequence);
        writer.write(0, 2); // ms_mask_present = none
        write_long_channel_stream(&mut writer, left, None)?;
        write_long_channel_stream(&mut writer, right, None)?;
        if let Some(fill) = fill_element {
            writer.write(ElementId::Fill.bits() as u32, 3);
            write_packed_fill_element(&mut writer, fill)?;
        }
        writer.write(ElementId::End.bits() as u32, 3);
        writer.byte_align();
        Ok(writer.finish())
    }
}

fn write_packed_fill_element(writer: &mut BitWriter, fill: &[u8]) -> Result<(), AacLcEncoderError> {
    let first = *fill
        .first()
        .ok_or(AacLcEncoderError::InvalidRawElementLayout)?;
    let nibble = first >> 4;
    let bits = if nibble < 15 {
        4 + usize::from(nibble) * 8
    } else {
        let extension = ((u16::from(first & 0x0f)) << 4)
            | u16::from(
                *fill
                    .get(1)
                    .ok_or(AacLcEncoderError::InvalidRawElementLayout)?
                    >> 4,
            );
        12 + (14 + usize::from(extension)) * 8
    };
    if bits > fill.len() * 8 {
        return Err(AacLcEncoderError::InvalidRawElementLayout);
    }
    for bit in 0..bits {
        writer.write_bool(fill[bit / 8] & (1 << (7 - bit % 8)) != 0);
    }
    Ok(())
}

fn write_long_ics_with_shape(
    writer: &mut BitWriter,
    max_sfb: usize,
    sequence: WindowSequence,
    shape: WindowShape,
) {
    writer.write_bool(false);
    writer.write(sequence.bits() as u32, 2);
    writer.write_bool(shape.bit());
    writer.write(max_sfb as u32, 6);
    writer.write_bool(false);
}

fn write_long_ics(writer: &mut BitWriter, max_sfb: usize, sequence: WindowSequence) {
    write_long_ics_with_shape(writer, max_sfb, sequence, WindowShape::Sine);
}

fn write_long_channel_stream(
    writer: &mut BitWriter,
    frame: &QuantizedAacLcFrame,
    sequence: Option<WindowSequence>,
) -> Result<(), AacLcEncoderError> {
    writer.write(frame.global_gain as u32, 8);
    if let Some(sequence) = sequence {
        write_long_ics(writer, frame.bands.len(), sequence);
    }
    write_sections(writer, &frame.sections, false);
    let mut factor = i16::from(frame.global_gain) - 100;
    write_scalefactors(writer, &frame.bands, &mut factor)?;
    writer.write_bool(false);
    writer.write_bool(false);
    writer.write_bool(false);
    write_spectral_bands(writer, &frame.bands)
}

fn write_sections(writer: &mut BitWriter, sections: &[AacLcSection], short: bool) {
    let (escape, bits) = if short { (7, 3) } else { (31, 5) };
    for section in sections {
        writer.write(section.codebook as u32, 4);
        let mut length = section.end_sfb - section.start_sfb;
        while length >= escape {
            writer.write(escape as u32, bits);
            length -= escape;
        }
        writer.write(length as u32, bits);
    }
}

fn write_scalefactors(
    writer: &mut BitWriter,
    bands: &[QuantizedSfb],
    factor: &mut i16,
) -> Result<(), AacLcEncoderError> {
    for band in bands {
        if band.codebook != 0 {
            let delta = band.scalefactor - *factor;
            if !(-60..=60).contains(&delta) {
                return Err(AacLcEncoderError::ScalefactorDeltaOutOfRange(delta));
            }
            write_fdk_huffman_word(writer, &HUFFMAN_CODEBOOK_SCL, (delta + 60) as u16)?;
            *factor = band.scalefactor;
        }
    }
    Ok(())
}

fn write_spectral_bands(
    writer: &mut BitWriter,
    bands: &[QuantizedSfb],
) -> Result<(), AacLcEncoderError> {
    for band in bands {
        if band.codebook == 0 {
            continue;
        }
        let dimension = usize::from(spectral_codebook(band.codebook)?.dimension);
        for tuple in band.coefficients.chunks_exact(dimension) {
            write_spectral_tuple(writer, band.codebook, tuple)?;
        }
    }
    Ok(())
}

/// Wrap one byte-aligned AAC raw_data_block in an MPEG-4 AAC-LC ADTS frame.
pub fn write_adts_frame(
    raw_data_block: &[u8],
    sampling_frequency_index: u8,
    channel_configuration: u8,
) -> Result<Vec<u8>, AacLcEncoderError> {
    let header = AdtsHeader::new(
        MpegVersion::Mpeg4,
        1,
        sampling_frequency_index,
        channel_configuration,
        raw_data_block.len(),
    )?;
    let mut output = vec![0u8; header.frame_length];
    let header_length = header.write(&mut output)?;
    output[header_length..].copy_from_slice(raw_data_block);
    Ok(output)
}

fn write_drm_mono_packet(
    frame: &QuantizedAacLcFrame,
    sampling_frequency_index: u8,
    offsets: &[usize],
) -> Result<Vec<u8>, AacLcEncoderError> {
    if offsets.len() != frame.bands.len() + 1 {
        return Err(AacLcEncoderError::InvalidRawElementLayout);
    }
    let ics = drm_ics(sampling_frequency_index, frame.bands.len() as u8)?;
    let mut emitted = Vec::<(usize, usize, u8)>::new();
    let mut start = 0usize;
    while start < frame.bands.len() {
        let codebook = frame.bands[start].codebook;
        let end = if codebook == 11 {
            start + 1
        } else {
            let mut end = start + 1;
            while end < frame.bands.len() && frame.bands[end].codebook == codebook {
                end += 1;
            }
            end
        };
        emitted.push((start, end, codebook));
        start = end;
    }
    let hcr_sections = emitted
        .iter()
        .map(|&(start, end, codebook)| HcrSection {
            codebook,
            spectral_lines: offsets[end] - offsets[start],
        })
        .collect::<Vec<_>>();
    let mut words = Vec::new();
    for (section_index, &(start, end, codebook)) in emitted.iter().enumerate() {
        if codebook == 0 {
            continue;
        }
        let dimension = usize::from(spectral_codebook(codebook)?.dimension);
        let mut codeword_index = 0usize;
        for band in &frame.bands[start..end] {
            for coefficients in band.coefficients.chunks_exact(dimension) {
                words.push(HcrDecodedCodeword {
                    section_index,
                    codeword_index,
                    coefficients: coefficients.to_vec(),
                });
                codeword_index += 1;
            }
        }
    }
    let (hcr_side, hcr_payload) = encode_reordered_codewords(&hcr_sections, &words)?;
    validate_hcr_length(&hcr_side, SCE_MAX_REORDERED_SPECTRAL_BITS)?;

    let mut writer = BitWriter::new();
    write_long_ics(&mut writer, ics.max_sfb as usize, WindowSequence::OnlyLong);
    writer.write_bool(false); // TNS absent
    writer.write_bool(false); // LTP absent
    writer.write(frame.global_gain as u32, 8);
    for &(start, end, codebook) in &emitted {
        writer.write(codebook as u32, 5);
        if codebook != 11 {
            let mut length = end - start;
            while length >= 31 {
                writer.write(31, 5);
                length -= 31;
            }
            writer.write(length as u32, 5);
        }
    }
    let mut factor = i16::from(frame.global_gain) - 100;
    write_scalefactors(&mut writer, &frame.bands, &mut factor)?;
    writer.write(hcr_side.reordered_spectral_bits as u32, 14);
    writer.write(hcr_side.longest_codeword_bits as u32, 6);
    let protected_bits = writer.bits_written();
    for bit in 0..hcr_side.reordered_spectral_bits {
        writer.write_bool(hcr_payload[bit / 8] & (1 << (7 - bit % 8)) != 0);
    }
    let payload = writer.finish();
    let crc = drm_crc8_bits(&payload, 0, protected_bits)
        .map_err(|_| AacLcEncoderError::InvalidRawElementLayout)?;
    let mut packet = Vec::with_capacity(payload.len() + 1);
    packet.push(crc);
    packet.extend_from_slice(&payload);
    Ok(packet)
}

fn prepare_drm_hcr(
    frame: &QuantizedAacLcFrame,
    offsets: &[usize],
) -> Result<(Vec<(usize, usize, u8)>, HcrSideInfo, Vec<u8>), AacLcEncoderError> {
    let mut emitted = Vec::new();
    let mut start = 0usize;
    while start < frame.bands.len() {
        let codebook = frame.bands[start].codebook;
        let end = if codebook == 11 {
            start + 1
        } else {
            let mut end = start + 1;
            while end < frame.bands.len() && frame.bands[end].codebook == codebook {
                end += 1;
            }
            end
        };
        emitted.push((start, end, codebook));
        start = end;
    }
    let sections = emitted
        .iter()
        .map(|&(start, end, codebook)| HcrSection {
            codebook,
            spectral_lines: offsets[end] - offsets[start],
        })
        .collect::<Vec<_>>();
    let mut words = Vec::new();
    for (section_index, &(start, end, codebook)) in emitted.iter().enumerate() {
        if codebook == 0 {
            continue;
        }
        let dimension = usize::from(spectral_codebook(codebook)?.dimension);
        let mut codeword_index = 0;
        for band in &frame.bands[start..end] {
            for coefficients in band.coefficients.chunks_exact(dimension) {
                words.push(HcrDecodedCodeword {
                    section_index,
                    codeword_index,
                    coefficients: coefficients.to_vec(),
                });
                codeword_index += 1;
            }
        }
    }
    let (side, payload) = encode_reordered_codewords(&sections, &words)?;
    Ok((emitted, side, payload))
}

fn validate_hcr_length(side: &HcrSideInfo, maximum: usize) -> Result<(), AacLcEncoderError> {
    if side.reordered_spectral_bits > maximum {
        return Err(HcrError::ReorderedSpectralLengthOutOfRange {
            length: side.reordered_spectral_bits,
            maximum,
        }
        .into());
    }
    Ok(())
}

fn write_drm_channel_side(
    writer: &mut BitWriter,
    frame: &QuantizedAacLcFrame,
    emitted: &[(usize, usize, u8)],
    hcr: HcrSideInfo,
) -> Result<(), AacLcEncoderError> {
    writer.write_bool(false);
    writer.write_bool(false);
    writer.write(frame.global_gain as u32, 8);
    for &(start, end, codebook) in emitted {
        writer.write(codebook as u32, 5);
        if codebook != 11 {
            let mut length = end - start;
            while length >= 31 {
                writer.write(31, 5);
                length -= 31;
            }
            writer.write(length as u32, 5);
        }
    }
    let mut factor = i16::from(frame.global_gain) - 100;
    write_scalefactors(writer, &frame.bands, &mut factor)?;
    writer.write(hcr.reordered_spectral_bits as u32, 14);
    writer.write(hcr.longest_codeword_bits as u32, 6);
    Ok(())
}

fn write_drm_stereo_packet(
    left: &QuantizedAacLcFrame,
    right: &QuantizedAacLcFrame,
    sampling_frequency_index: u8,
    offsets: &[usize],
) -> Result<Vec<u8>, AacLcEncoderError> {
    let ics = drm_ics(sampling_frequency_index, left.bands.len() as u8)?;
    let (left_sections, left_hcr, left_payload) = prepare_drm_hcr(left, offsets)?;
    let (right_sections, right_hcr, right_payload) = prepare_drm_hcr(right, offsets)?;
    for side in [left_hcr, right_hcr] {
        validate_hcr_length(&side, 12_288)?;
    }
    let mut writer = BitWriter::new();
    write_long_ics(&mut writer, ics.max_sfb as usize, WindowSequence::OnlyLong);
    writer.write(0, 2); // ms_mask_present = none
    write_drm_channel_side(&mut writer, left, &left_sections, left_hcr)?;
    write_drm_channel_side(&mut writer, right, &right_sections, right_hcr)?;
    let protected_bits = writer.bits_written();
    for (side, payload) in [(left_hcr, &left_payload), (right_hcr, &right_payload)] {
        for bit in 0..side.reordered_spectral_bits {
            writer.write_bool(payload[bit / 8] & (1 << (7 - bit % 8)) != 0);
        }
    }
    let payload = writer.finish();
    let crc = drm_crc8_bits(&payload, 0, protected_bits)
        .map_err(|_| AacLcEncoderError::InvalidRawElementLayout)?;
    let mut packet = vec![crc];
    packet.extend_from_slice(&payload);
    Ok(packet)
}

#[derive(Debug, Clone)]
pub struct AacLcQuantizer {
    sampling_frequency_index: u8,
    frame_length: usize,
    afterburner: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AacLcBitReservoir {
    nominal_frame_bits: usize,
    capacity_bits: usize,
    fullness_bits: usize,
}

#[derive(Debug, Clone)]
struct LowDelayPeCorrection {
    factor: f32,
    pe_last: f32,
    dynamic_bits_last: Option<usize>,
    pe_min: Option<f32>,
    pe_max: Option<f32>,
}

impl Default for LowDelayPeCorrection {
    fn default() -> Self {
        Self {
            factor: 1.0,
            pe_last: 0.0,
            dynamic_bits_last: None,
            pe_min: None,
            pe_max: None,
        }
    }
}

impl LowDelayPeCorrection {
    fn reservoir_bit_factor(&mut self, current_pe: f32, mean_pe: f32, fill: f32) -> f32 {
        let pe_min = self.pe_min.get_or_insert(mean_pe * 0.8);
        let pe_max = self.pe_max.get_or_insert(mean_pe * 1.2);
        let clipped_pe = current_pe.clamp(*pe_min, *pe_max);
        let pe_slope = if *pe_max > *pe_min {
            (clipped_pe - *pe_min) / (*pe_max - *pe_min)
        } else {
            0.0
        };
        let clipped_fill = fill.clamp(0.2, 0.95);
        let bit_save = 0.3 - (clipped_fill - 0.2) * 0.466_666_67;
        let bit_spend = -0.1 + (clipped_fill - 0.2) * 0.666_666_7;
        let factor = 1.0 - bit_save + pe_slope * (bit_spend + bit_save);

        adjust_low_delay_pe_min_max(current_pe, pe_min, pe_max);
        factor
    }

    fn corrected_grant(&mut self, current_pe: f32, granted_pe: f32, bits_to_pe_factor: f32) -> f32 {
        if let Some(bits_last) = self.dynamic_bits_last.filter(|&bits| bits > 0) {
            let estimated_last = bits_last as f32 * bits_to_pe_factor;
            if current_pe < 1.5 * self.pe_last
                && current_pe > 0.7 * self.pe_last
                && 1.2 * estimated_last > self.pe_last
                && 0.65 * estimated_last < self.pe_last
            {
                let ratio = self.pe_last / estimated_last.max(f32::MIN_POSITIVE);
                let new_factor = if self.pe_last <= estimated_last {
                    (1.1 * ratio).min(1.0).max(0.85)
                } else {
                    (0.9 * ratio).min(1.15).max(1.0)
                };
                if (new_factor - 1.0) * (self.factor - 1.0) < 0.0 {
                    self.factor = 1.0;
                }
                let moves_away = (self.factor < 1.0 && new_factor < self.factor)
                    || (self.factor > 1.0 && new_factor > self.factor);
                let weight = if moves_away { 0.15 } else { 0.30 };
                self.factor =
                    ((1.0 - weight) * self.factor + weight * new_factor).clamp(0.85, 1.15);
            } else {
                self.factor = 1.0;
            }
        } else {
            self.factor = 1.0;
        }
        self.pe_last = granted_pe;
        self.dynamic_bits_last = None;
        granted_pe * self.factor
    }

    fn commit_dynamic_bits(&mut self, bits: usize) {
        self.dynamic_bits_last = Some(bits);
    }
}

fn adjust_low_delay_pe_min_max(current_pe: f32, pe_min: &mut f32, pe_max: &mut f32) {
    if current_pe > *pe_max {
        let difference = current_pe - *pe_max;
        *pe_min += 0.3 * difference;
        *pe_max += difference;
    } else if current_pe < *pe_min {
        let difference = *pe_min - current_pe;
        *pe_min -= 0.14 * difference;
        *pe_max -= 0.07 * difference;
    } else {
        *pe_min += 0.3 * (current_pe - *pe_min);
        *pe_max -= 0.07 * (*pe_max - current_pe);
    }

    let minimum_difference = current_pe / 6.0;
    if *pe_max - *pe_min < minimum_difference {
        let below = (current_pe - *pe_min).max(0.0);
        let above = (*pe_max - current_pe).max(0.0);
        let total = below + above;
        if total > 0.0 {
            *pe_max = current_pe + above / total * minimum_difference;
            *pe_min = (current_pe - below / total * minimum_difference).max(0.0);
        } else {
            *pe_min = (current_pe - minimum_difference * 0.5).max(0.0);
            *pe_max = current_pe + minimum_difference * 0.5;
        }
    }
}

/// Apply the encoder low-pass in the MDCT domain.  FDK computes the first
/// discarded line as `2 * bandwidth * transform_length / sample_rate`.
fn apply_spectral_bandwidth(spectrum: &mut [f32], sample_rate: u32, bandwidth: u32) {
    if sample_rate == 0 {
        return;
    }
    let first_discarded =
        ((2 * u64::from(bandwidth) * spectrum.len() as u64) / u64::from(sample_rate)) as usize;
    let first_discarded = first_discarded.min(spectrum.len());
    spectrum[first_discarded..].fill(0.0);
}

fn apply_short_spectral_bandwidth(
    spectra: &mut Option<Vec<Vec<f32>>>,
    sample_rate: u32,
    bandwidth: u32,
) {
    if let Some(spectra) = spectra {
        for spectrum in spectra {
            apply_spectral_bandwidth(spectrum, sample_rate, bandwidth);
        }
    }
}

#[derive(Debug, Clone)]
pub struct PureRustAacLcMonoEncoder {
    sampling_frequency_index: u8,
    sampling_frequency: u32,
    bandwidth: u32,
    analysis: AacLcAnalysisFilterbank,
    psychoacoustic: AacLcPsychoacousticModel,
    quantizer: AacLcQuantizer,
    block_switcher: AacLcBlockSwitcher,
    reservoir: AacLcBitReservoir,
    bitrate: u32,
    vbr_quality_factor: Option<f32>,
    chaos_measure_old: f32,
    latm_writer: LatmAacLcWriter,
    adif_header: Vec<u8>,
    adif_header_written: bool,
}

/// ER AAC-LD mono spectral encoder.
///
/// AAC-LD mono PCM/spectral encoder for 480- and 512-sample frames.
#[derive(Debug, Clone)]
pub struct PureRustAacLdMonoEncoder {
    sampling_frequency_index: u8,
    sampling_frequency: u32,
    frame_length: usize,
    bandwidth: u32,
    analysis: AacLcAnalysisFilterbank,
    block_switcher: AacLdBlockSwitcher,
    window_shape: WindowShape,
    reservoir: AacLcBitReservoir,
    cbr_fill_enabled: bool,
    afterburner: bool,
    vbr_quality_factor: Option<f32>,
    chaos_measure_old: f32,
    pe_correction: LowDelayPeCorrection,
}

#[derive(Debug, Clone)]
pub struct PureRustAacLdStereoEncoder {
    sampling_frequency_index: u8,
    sampling_frequency: u32,
    frame_length: usize,
    bandwidth: u32,
    left_analysis: AacLcAnalysisFilterbank,
    right_analysis: AacLcAnalysisFilterbank,
    left_block_switcher: AacLdBlockSwitcher,
    right_block_switcher: AacLdBlockSwitcher,
    window_shape: WindowShape,
    reservoir: AacLcBitReservoir,
    cbr_fill_enabled: bool,
    afterburner: bool,
    vbr_quality_factor: Option<f32>,
    chaos_measure_old: f32,
    pe_correction: LowDelayPeCorrection,
}

/// ER AAC-LD encoder for standardized 3.0 through 7.1 channel layouts.
#[derive(Debug, Clone)]
pub struct PureRustAacLdMultichannelEncoder {
    sampling_frequency_index: u8,
    sampling_frequency: u32,
    frame_length: usize,
    channels: usize,
    channel_mode: u32,
    bandwidth: u32,
    analyses: Vec<AacLcAnalysisFilterbank>,
    block_switchers: Vec<AacLdBlockSwitcher>,
    window_shape: WindowShape,
    reservoir: AacLcBitReservoir,
    cbr_fill_enabled: bool,
    afterburner: bool,
    vbr_quality_factor: Option<f32>,
    chaos_measure_old: f32,
    pe_correction: LowDelayPeCorrection,
}

#[derive(Debug, Clone)]
pub struct PureRustAacEldMonoEncoder {
    sampling_frequency_index: u8,
    sampling_frequency: u32,
    frame_length: usize,
    bandwidth: u32,
    analysis: EldAnalysisFilterbank,
    reservoir: AacLcBitReservoir,
    cbr_fill_enabled: bool,
    afterburner: bool,
    vbr_quality_factor: Option<f32>,
    chaos_measure_old: f32,
    pe_correction: LowDelayPeCorrection,
    sbr_analysis: Option<SbrEncoderAnalysis>,
    sbr_header: Option<LdSbrHeader>,
    sbr_dual_rate: bool,
    sbr_crc: bool,
    sbr_header_written: bool,
    sbr_downsampler: Option<HalfbandDownsampler>,
    sbr_coding_state: LowDelaySbrCodingState,
}

#[derive(Debug, Clone)]
pub struct PureRustAacEldStereoEncoder {
    sampling_frequency_index: u8,
    sampling_frequency: u32,
    frame_length: usize,
    bandwidth: u32,
    left_analysis: EldAnalysisFilterbank,
    right_analysis: EldAnalysisFilterbank,
    reservoir: AacLcBitReservoir,
    cbr_fill_enabled: bool,
    afterburner: bool,
    vbr_quality_factor: Option<f32>,
    chaos_measure_old: f32,
    pe_correction: LowDelayPeCorrection,
    left_sbr_analysis: Option<SbrEncoderAnalysis>,
    right_sbr_analysis: Option<SbrEncoderAnalysis>,
    sbr_header: Option<LdSbrHeader>,
    sbr_dual_rate: bool,
    sbr_crc: bool,
    sbr_header_written: bool,
    left_sbr_downsampler: Option<HalfbandDownsampler>,
    right_sbr_downsampler: Option<HalfbandDownsampler>,
    left_sbr_coding_state: LowDelaySbrCodingState,
    right_sbr_coding_state: LowDelaySbrCodingState,
    left_sbr_input_delay: [f32; 5],
    right_sbr_input_delay: [f32; 5],
    #[cfg(test)]
    pub(crate) last_sbr_prequant_debug: Option<(LowDelayPrequantDebug, LowDelayPrequantDebug)>,
}

/// ER AAC-ELD encoder for standardized 3.0 through 7.1 channel layouts.
#[derive(Debug, Clone)]
pub struct PureRustAacEldMultichannelEncoder {
    sampling_frequency_index: u8,
    sampling_frequency: u32,
    frame_length: usize,
    channels: usize,
    channel_mode: u32,
    bandwidth: u32,
    analyses: Vec<EldAnalysisFilterbank>,
    reservoir: AacLcBitReservoir,
    cbr_fill_enabled: bool,
    afterburner: bool,
    sbr_analyses: Vec<SbrEncoderAnalysis>,
    sbr_headers: Vec<LdSbrHeader>,
    sbr_element_bitrates: Vec<u32>,
    sbr_dual_rate: bool,
    sbr_crc: bool,
    sbr_header_written: bool,
    sbr_downsamplers: Vec<HalfbandDownsampler>,
    sbr_coding_states: Vec<LowDelaySbrCodingState>,
}

#[derive(Debug, Clone)]
pub struct PureRustAacEldMpsEncoder {
    core: PureRustAacEldMonoEncoder,
    spatial: Sac212Encoder,
}

#[derive(Debug, Clone)]
pub struct PureRustAacLdMpsEncoder {
    core: PureRustAacLdMonoEncoder,
    spatial: Sac212Encoder,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EldMpsEncoderError {
    Core(AacLcEncoderError),
    Spatial(SacEncodeError),
    InvalidFrameGeometry,
}

impl From<AacLcEncoderError> for EldMpsEncoderError {
    fn from(value: AacLcEncoderError) -> Self {
        Self::Core(value)
    }
}

impl From<SacEncodeError> for EldMpsEncoderError {
    fn from(value: SacEncodeError) -> Self {
        Self::Spatial(value)
    }
}

impl fmt::Display for EldMpsEncoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AAC-ELD MPEG Surround encoder error: {self:?}")
    }
}

impl std::error::Error for EldMpsEncoderError {}

impl PureRustAacLdMonoEncoder {
    pub fn new(
        sampling_frequency_index: u8,
        frame_length: usize,
        nominal_frame_bits: usize,
        reservoir_capacity_bits: usize,
    ) -> Result<Self, AacLcEncoderError> {
        if !matches!(frame_length, 480 | 512) {
            return Err(AacLcEncoderError::UnsupportedFrameLength(frame_length));
        }
        let sampling_frequency = sample_rate_from_index(sampling_frequency_index)
            .ok_or(AacLcEncoderError::InvalidRawElementLayout)?;
        let ics = er_long_ics(sampling_frequency_index, frame_length)?;
        aac_band_offsets_for_ics(sampling_frequency_index, &ics, frame_length)?;
        Ok(Self {
            sampling_frequency_index,
            sampling_frequency,
            frame_length,
            bandwidth: sampling_frequency / 2,
            analysis: AacLcAnalysisFilterbank::new(frame_length)?,
            block_switcher: AacLdBlockSwitcher::default(),
            window_shape: WindowShape::Sine,
            reservoir: AacLcBitReservoir::new_full(nominal_frame_bits, reservoir_capacity_bits),
            cbr_fill_enabled: false,
            afterburner: false,
            vbr_quality_factor: None,
            chaos_measure_old: 0.3,
            pe_correction: LowDelayPeCorrection::default(),
        })
    }

    pub fn frame_length(&self) -> usize {
        self.frame_length
    }

    pub fn bit_reservoir(&self) -> &AacLcBitReservoir {
        &self.reservoir
    }

    pub fn set_cbr_fill_enabled(&mut self, enabled: bool) {
        self.cbr_fill_enabled = enabled;
    }

    pub fn set_afterburner(&mut self, enabled: bool) {
        self.afterburner = enabled;
    }

    pub fn set_bitrate_mode(&mut self, mode: u32) {
        self.vbr_quality_factor = low_delay_vbr_quality_factor(mode);
    }

    pub fn set_bandwidth(&mut self, bandwidth: u32) {
        self.bandwidth = bandwidth.min(self.sampling_frequency / 2);
    }

    pub fn audio_specific_config(&self) -> AudioSpecificConfig {
        AudioSpecificConfig {
            audio_object_type: 23,
            sampling_frequency_index: self.sampling_frequency_index,
            sampling_frequency: self.sampling_frequency,
            channel_configuration: 1,
            extension: None,
            ga_specific: Some(GaSpecificConfig {
                frame_length_flag: self.frame_length == 480,
                ..GaSpecificConfig::default()
            }),
            eld_specific: None,
            usac_config: None,
            error_protection_config: Some(0),
            program_config: None,
            bits_read: 0,
        }
    }

    /// Quantize one AAC-LD spectrum and write the ER mono raw access unit.
    pub fn encode_spectrum(&mut self, spectrum: &[f32]) -> Result<Vec<u8>, AacLcEncoderError> {
        self.encode_spectrum_with_ancillary(spectrum, &[])
    }

    pub fn encode_spectrum_with_ancillary(
        &mut self,
        spectrum: &[f32],
        ancillary: &[u8],
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        let ancillary_elements = [ancillary];
        self.encode_spectrum_with_extensions(spectrum, None, &ancillary_elements)
    }

    pub(crate) fn encode_spectrum_with_extensions(
        &mut self,
        spectrum: &[f32],
        dynamic_range: Option<(&[u8], usize)>,
        ancillary_elements: &[&[u8]],
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        self.encode_spectrum_with_optional_mps_extensions(
            spectrum,
            None,
            dynamic_range,
            ancillary_elements,
        )
    }

    fn encode_spectrum_with_optional_mps_extensions(
        &mut self,
        spectrum: &[f32],
        mps: Option<(&[u8], usize)>,
        dynamic_range: Option<(&[u8], usize)>,
        ancillary_elements: &[&[u8]],
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        if spectrum.len() != self.frame_length {
            return Err(AacLcEncoderError::InputLengthMismatch {
                expected: self.frame_length,
                actual: spectrum.len(),
            });
        }
        if spectrum.iter().any(|sample| !sample.is_finite()) {
            return Err(AacLcEncoderError::NonFiniteInput);
        }
        let ics = er_long_ics(self.sampling_frequency_index, self.frame_length)?;
        let offsets =
            aac_band_offsets_for_ics(self.sampling_frequency_index, &ics, self.frame_length)?
                .offsets;
        let mut spectrum = spectrum.to_vec();
        apply_spectral_bandwidth(&mut spectrum, self.sampling_frequency, self.bandwidth);
        let mut psycho = analyze_low_delay_psychoacoustic_bands(
            &spectrum,
            offsets,
            self.reservoir.nominal_frame_bits(),
            self.sampling_frequency,
            self.frame_length,
            self.bandwidth,
            1,
        );
        let bitrate = ((self.reservoir.nominal_frame_bits() as u64
            * u64::from(self.sampling_frequency))
            / self.frame_length as u64) as u32;
        if let Some(quality) = self.vbr_quality_factor {
            apply_low_delay_vbr_thresholds(
                &mut [&mut psycho],
                offsets,
                bitrate,
                self.sampling_frequency,
                self.frame_length,
                self.bandwidth,
                quality,
                &mut self.chaos_measure_old,
            );
        }
        let bits_to_pe = low_delay_bits_to_pe_factor(
            bitrate,
            1,
            self.sampling_frequency,
            self.afterburner,
            bitrate < 32_000,
        );
        let static_bits = 24;
        let available = if self.vbr_quality_factor.is_some() {
            self.reservoir.available_frame_bits()
        } else {
            static_bits
                + low_delay_granted_dynamic_bits(
                    &self.reservoir,
                    psycho.perceptual_entropy,
                    bits_to_pe,
                    static_bits,
                    &mut self.pe_correction,
                )
        };
        if self.vbr_quality_factor.is_none() {
            let granted_pe = available.saturating_sub(static_bits) as f32 * bits_to_pe;
            let target_pe = self.pe_correction.corrected_grant(
                psycho.perceptual_entropy,
                granted_pe,
                bits_to_pe,
            );
            apply_low_delay_cbr_thresholds(
                &mut [&mut psycho],
                offsets,
                target_pe,
                bitrate,
                self.sampling_frequency,
                self.frame_length,
                self.bandwidth,
            );
        }
        let (frame, raw) = fit_low_delay_access_unit(available, |relaxation| {
            let frame = quantize_drm_spectrum(&spectrum, &psycho, offsets, relaxation);
            let raw = write_er_aac_ld_mono_access_unit(
                &frame,
                0,
                self.window_shape,
                mps,
                dynamic_range,
                ancillary_elements,
                0,
            )?;
            Ok((frame, raw))
        })?;
        self.pe_correction
            .commit_dynamic_bits(frame.estimated_spectral_bits + frame.estimated_section_bits);
        let raw = add_er_cbr_fill(raw, self.cbr_fill_enabled, &self.reservoir, |fill_bytes| {
            write_er_aac_ld_mono_access_unit(
                &frame,
                0,
                self.window_shape,
                mps,
                dynamic_range,
                ancillary_elements,
                fill_bytes,
            )
        })?;
        self.reservoir.commit_frame(raw.len() * 8)?;
        Ok(raw)
    }

    pub fn encode_pcm(&mut self, input: &[f32]) -> Result<Vec<u8>, AacLcEncoderError> {
        self.encode_pcm_with_ancillary(input, &[])
    }

    pub fn encode_pcm_with_ancillary(
        &mut self,
        input: &[f32],
        ancillary: &[u8],
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        let ancillary_elements = [ancillary];
        self.encode_pcm_with_extensions(input, None, &ancillary_elements)
    }

    pub(crate) fn encode_pcm_with_extensions(
        &mut self,
        input: &[f32],
        dynamic_range: Option<(&[u8], usize)>,
        ancillary_elements: &[&[u8]],
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        self.encode_pcm_with_optional_mps_extensions(input, None, dynamic_range, ancillary_elements)
    }

    pub(crate) fn encode_pcm_with_mps_extensions(
        &mut self,
        input: &[f32],
        mps_payload: &[u8],
        mps_payload_bits: usize,
        dynamic_range: Option<(&[u8], usize)>,
        ancillary_elements: &[&[u8]],
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        self.encode_pcm_with_optional_mps_extensions(
            input,
            Some((mps_payload, mps_payload_bits)),
            dynamic_range,
            ancillary_elements,
        )
    }

    fn encode_pcm_with_optional_mps_extensions(
        &mut self,
        input: &[f32],
        mps: Option<(&[u8], usize)>,
        dynamic_range: Option<(&[u8], usize)>,
        ancillary_elements: &[&[u8]],
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        self.window_shape = if self.block_switcher.detect(input) {
            WindowShape::LowOverlap
        } else {
            WindowShape::Sine
        };
        let spectrum = self
            .analysis
            .analyze_aac_ld(input, self.window_shape)?
            .spectrum;
        self.encode_spectrum_with_optional_mps_extensions(
            &spectrum,
            mps,
            dynamic_range,
            ancillary_elements,
        )
    }
}

impl PureRustAacLdStereoEncoder {
    pub fn new(
        sampling_frequency_index: u8,
        frame_length: usize,
        nominal_frame_bits: usize,
        reservoir_capacity_bits: usize,
    ) -> Result<Self, AacLcEncoderError> {
        if !matches!(frame_length, 480 | 512) {
            return Err(AacLcEncoderError::UnsupportedFrameLength(frame_length));
        }
        let sampling_frequency = sample_rate_from_index(sampling_frequency_index)
            .ok_or(AacLcEncoderError::InvalidRawElementLayout)?;
        let ics = er_long_ics(sampling_frequency_index, frame_length)?;
        aac_band_offsets_for_ics(sampling_frequency_index, &ics, frame_length)?;
        Ok(Self {
            sampling_frequency_index,
            sampling_frequency,
            frame_length,
            bandwidth: sampling_frequency / 2,
            left_analysis: AacLcAnalysisFilterbank::new(frame_length)?,
            right_analysis: AacLcAnalysisFilterbank::new(frame_length)?,
            left_block_switcher: AacLdBlockSwitcher::default(),
            right_block_switcher: AacLdBlockSwitcher::default(),
            window_shape: WindowShape::Sine,
            reservoir: AacLcBitReservoir::new_full(nominal_frame_bits, reservoir_capacity_bits),
            cbr_fill_enabled: false,
            afterburner: false,
            vbr_quality_factor: None,
            chaos_measure_old: 0.3,
            pe_correction: LowDelayPeCorrection::default(),
        })
    }

    pub fn frame_length(&self) -> usize {
        self.frame_length
    }

    pub fn bit_reservoir(&self) -> &AacLcBitReservoir {
        &self.reservoir
    }

    pub fn set_cbr_fill_enabled(&mut self, enabled: bool) {
        self.cbr_fill_enabled = enabled;
    }

    pub fn set_afterburner(&mut self, enabled: bool) {
        self.afterburner = enabled;
    }

    pub fn set_bitrate_mode(&mut self, mode: u32) {
        self.vbr_quality_factor = low_delay_vbr_quality_factor(mode);
    }

    pub fn set_bandwidth(&mut self, bandwidth: u32) {
        self.bandwidth = bandwidth.min(self.sampling_frequency / 2);
    }

    pub fn audio_specific_config(&self) -> AudioSpecificConfig {
        AudioSpecificConfig {
            audio_object_type: 23,
            sampling_frequency_index: self.sampling_frequency_index,
            sampling_frequency: self.sampling_frequency,
            channel_configuration: 2,
            extension: None,
            ga_specific: Some(GaSpecificConfig {
                frame_length_flag: self.frame_length == 480,
                ..GaSpecificConfig::default()
            }),
            eld_specific: None,
            usac_config: None,
            error_protection_config: Some(0),
            program_config: None,
            bits_read: 0,
        }
    }

    pub fn encode_spectra(
        &mut self,
        left: &[f32],
        right: &[f32],
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        self.encode_spectra_with_ancillary(left, right, &[])
    }

    pub fn encode_spectra_with_ancillary(
        &mut self,
        left: &[f32],
        right: &[f32],
        ancillary: &[u8],
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        let ancillary_elements = [ancillary];
        self.encode_spectra_with_extensions(left, right, None, &ancillary_elements)
    }

    pub(crate) fn encode_spectra_with_extensions(
        &mut self,
        left: &[f32],
        right: &[f32],
        dynamic_range: Option<(&[u8], usize)>,
        ancillary_elements: &[&[u8]],
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        if left.len() != self.frame_length || right.len() != self.frame_length {
            return Err(AacLcEncoderError::InputLengthMismatch {
                expected: self.frame_length,
                actual: left.len().min(right.len()),
            });
        }
        if left.iter().chain(right).any(|sample| !sample.is_finite()) {
            return Err(AacLcEncoderError::NonFiniteInput);
        }
        let ics = er_long_ics(self.sampling_frequency_index, self.frame_length)?;
        let offsets =
            aac_band_offsets_for_ics(self.sampling_frequency_index, &ics, self.frame_length)?
                .offsets;
        let mut left = left.to_vec();
        let mut right = right.to_vec();
        apply_spectral_bandwidth(&mut left, self.sampling_frequency, self.bandwidth);
        apply_spectral_bandwidth(&mut right, self.sampling_frequency, self.bandwidth);
        let mut left_psycho = analyze_low_delay_psychoacoustic_bands(
            &left,
            offsets,
            self.reservoir.nominal_frame_bits(),
            self.sampling_frequency,
            self.frame_length,
            self.bandwidth,
            2,
        );
        let mut right_psycho = analyze_low_delay_psychoacoustic_bands(
            &right,
            offsets,
            self.reservoir.nominal_frame_bits(),
            self.sampling_frequency,
            self.frame_length,
            self.bandwidth,
            2,
        );
        let bitrate = ((self.reservoir.nominal_frame_bits() as u64
            * u64::from(self.sampling_frequency))
            / self.frame_length as u64) as u32;
        if let Some(quality) = self.vbr_quality_factor {
            apply_low_delay_vbr_thresholds(
                &mut [&mut left_psycho, &mut right_psycho],
                offsets,
                bitrate,
                self.sampling_frequency,
                self.frame_length,
                self.bandwidth,
                quality,
                &mut self.chaos_measure_old,
            );
        }
        let bits_to_pe = low_delay_bits_to_pe_factor(
            bitrate,
            2,
            self.sampling_frequency,
            self.afterburner,
            false,
        );
        let static_bits = 40;
        let available = if self.vbr_quality_factor.is_some() {
            self.reservoir.available_frame_bits()
        } else {
            static_bits
                + low_delay_granted_dynamic_bits(
                    &self.reservoir,
                    left_psycho.perceptual_entropy + right_psycho.perceptual_entropy,
                    bits_to_pe,
                    static_bits,
                    &mut self.pe_correction,
                )
        };
        if self.vbr_quality_factor.is_none() {
            let current_pe = left_psycho.perceptual_entropy + right_psycho.perceptual_entropy;
            let granted_pe = available.saturating_sub(static_bits) as f32 * bits_to_pe;
            let target_pe = self
                .pe_correction
                .corrected_grant(current_pe, granted_pe, bits_to_pe);
            apply_low_delay_cbr_thresholds(
                &mut [&mut left_psycho, &mut right_psycho],
                offsets,
                target_pe,
                bitrate,
                self.sampling_frequency,
                self.frame_length,
                self.bandwidth,
            );
        }
        let ((left_frame, right_frame), raw) =
            fit_low_delay_access_unit(available, |relaxation| {
                let left = quantize_drm_spectrum(&left, &left_psycho, offsets, relaxation);
                let right = quantize_drm_spectrum(&right, &right_psycho, offsets, relaxation);
                let raw = write_er_aac_ld_stereo_access_unit(
                    &left,
                    &right,
                    0,
                    self.window_shape,
                    dynamic_range,
                    ancillary_elements,
                    0,
                )?;
                Ok(((left, right), raw))
            })?;
        self.pe_correction.commit_dynamic_bits(
            left_frame.estimated_spectral_bits
                + left_frame.estimated_section_bits
                + right_frame.estimated_spectral_bits
                + right_frame.estimated_section_bits,
        );
        let raw = add_er_cbr_fill(raw, self.cbr_fill_enabled, &self.reservoir, |fill_bytes| {
            write_er_aac_ld_stereo_access_unit(
                &left_frame,
                &right_frame,
                0,
                self.window_shape,
                dynamic_range,
                ancillary_elements,
                fill_bytes,
            )
        })?;
        self.reservoir.commit_frame(raw.len() * 8)?;
        Ok(raw)
    }

    pub fn encode_pcm(
        &mut self,
        left: &[f32],
        right: &[f32],
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        self.encode_pcm_with_ancillary(left, right, &[])
    }

    pub fn encode_pcm_with_ancillary(
        &mut self,
        left: &[f32],
        right: &[f32],
        ancillary: &[u8],
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        let ancillary_elements = [ancillary];
        self.encode_pcm_with_extensions(left, right, None, &ancillary_elements)
    }

    pub(crate) fn encode_pcm_with_extensions(
        &mut self,
        left: &[f32],
        right: &[f32],
        dynamic_range: Option<(&[u8], usize)>,
        ancillary_elements: &[&[u8]],
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        let attack =
            self.left_block_switcher.detect(left) | self.right_block_switcher.detect(right);
        self.window_shape = if attack {
            WindowShape::LowOverlap
        } else {
            WindowShape::Sine
        };
        let left = self
            .left_analysis
            .analyze_aac_ld(left, self.window_shape)?
            .spectrum;
        let right = self
            .right_analysis
            .analyze_aac_ld(right, self.window_shape)?
            .spectrum;
        self.encode_spectra_with_extensions(&left, &right, dynamic_range, ancillary_elements)
    }
}

impl PureRustAacLdMultichannelEncoder {
    pub fn new_with_channel_mode(
        sampling_frequency_index: u8,
        channels: usize,
        channel_mode: u32,
        frame_length: usize,
        nominal_frame_bits: usize,
        reservoir_capacity_bits: usize,
    ) -> Result<Self, AacLcEncoderError> {
        let layout = multichannel_layout_for_mode(channel_mode)?;
        let layout_channels = layout
            .iter()
            .flat_map(|(_, first, second)| [Some(*first), *second])
            .flatten()
            .max()
            .map_or(0, |last| last + 1);
        if layout_channels != channels || !matches!(frame_length, 480 | 512) {
            return Err(AacLcEncoderError::InvalidRawElementLayout);
        }
        let sampling_frequency = sample_rate_from_index(sampling_frequency_index)
            .ok_or(AacLcEncoderError::InvalidRawElementLayout)?;
        let ics = er_long_ics(sampling_frequency_index, frame_length)?;
        aac_band_offsets_for_ics(sampling_frequency_index, &ics, frame_length)?;
        let analyses = (0..channels)
            .map(|_| AacLcAnalysisFilterbank::new(frame_length))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            sampling_frequency_index,
            sampling_frequency,
            frame_length,
            channels,
            channel_mode,
            bandwidth: sampling_frequency / 2,
            analyses,
            block_switchers: vec![AacLdBlockSwitcher::default(); channels],
            window_shape: WindowShape::Sine,
            reservoir: AacLcBitReservoir::new_full(nominal_frame_bits, reservoir_capacity_bits),
            cbr_fill_enabled: false,
            afterburner: false,
            vbr_quality_factor: None,
            chaos_measure_old: 0.3,
            pe_correction: LowDelayPeCorrection::default(),
        })
    }

    pub fn bit_reservoir(&self) -> &AacLcBitReservoir {
        &self.reservoir
    }

    pub fn set_bandwidth(&mut self, bandwidth: u32) {
        self.bandwidth = bandwidth.min(self.sampling_frequency / 2);
    }

    pub fn set_cbr_fill_enabled(&mut self, enabled: bool) {
        self.cbr_fill_enabled = enabled;
    }

    pub fn set_afterburner(&mut self, enabled: bool) {
        self.afterburner = enabled;
    }

    pub fn set_bitrate_mode(&mut self, mode: u32) {
        self.vbr_quality_factor = low_delay_vbr_quality_factor(mode);
    }

    pub fn audio_specific_config(&self) -> AudioSpecificConfig {
        AudioSpecificConfig {
            audio_object_type: 23,
            sampling_frequency_index: self.sampling_frequency_index,
            sampling_frequency: self.sampling_frequency,
            channel_configuration: self.channel_mode as u8,
            extension: None,
            ga_specific: Some(GaSpecificConfig {
                frame_length_flag: self.frame_length == 480,
                // FDK signals the ER extension fields for multichannel LD,
                // even when all three resilience tools are disabled.
                extension_flag: true,
                extension_flag3: Some(false),
                ..GaSpecificConfig::default()
            }),
            eld_specific: None,
            usac_config: None,
            error_protection_config: Some(0),
            program_config: None,
            bits_read: 0,
        }
    }

    pub fn encode_pcm(&mut self, pcm: &[Vec<f32>]) -> Result<Vec<u8>, AacLcEncoderError> {
        self.encode_pcm_with_extensions(pcm, None, &[])
    }

    pub(crate) fn encode_pcm_with_extensions(
        &mut self,
        pcm: &[Vec<f32>],
        dynamic_range: Option<(&[u8], usize)>,
        ancillary_elements: &[&[u8]],
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        if pcm.len() != self.channels
            || pcm.iter().any(|channel| channel.len() != self.frame_length)
        {
            return Err(AacLcEncoderError::InputLengthMismatch {
                expected: self.channels * self.frame_length,
                actual: pcm.iter().map(Vec::len).sum(),
            });
        }
        if pcm.iter().flatten().any(|sample| !sample.is_finite()) {
            return Err(AacLcEncoderError::NonFiniteInput);
        }
        let attack = self
            .block_switchers
            .iter_mut()
            .zip(pcm)
            .fold(false, |attack, (switcher, samples)| {
                switcher.detect(samples) | attack
            });
        self.window_shape = if attack {
            WindowShape::LowOverlap
        } else {
            WindowShape::Sine
        };
        let ics = er_long_ics(self.sampling_frequency_index, self.frame_length)?;
        let offsets =
            aac_band_offsets_for_ics(self.sampling_frequency_index, &ics, self.frame_length)?
                .offsets;
        let mut spectra = Vec::with_capacity(self.channels);
        for (analysis, samples) in self.analyses.iter_mut().zip(pcm) {
            let mut spectrum = analysis
                .analyze_aac_ld(samples, self.window_shape)?
                .spectrum;
            apply_spectral_bandwidth(&mut spectrum, self.sampling_frequency, self.bandwidth);
            spectra.push(spectrum);
        }
        let targets = multichannel_channel_bit_targets(
            self.channel_mode,
            self.reservoir.nominal_frame_bits(),
            20,
        )?;
        let mut psycho = spectra
            .iter()
            .zip(targets)
            .map(|(spectrum, target)| {
                analyze_low_delay_psychoacoustic_bands(
                    spectrum,
                    offsets,
                    target,
                    self.sampling_frequency,
                    self.frame_length,
                    self.bandwidth,
                    self.channels,
                )
            })
            .collect::<Vec<_>>();
        let bitrate = ((self.reservoir.nominal_frame_bits() as u64
            * u64::from(self.sampling_frequency))
            / self.frame_length as u64) as u32;
        if let Some(quality) = self.vbr_quality_factor {
            let mut refs = psycho.iter_mut().collect::<Vec<_>>();
            apply_low_delay_vbr_thresholds(
                &mut refs,
                offsets,
                bitrate,
                self.sampling_frequency,
                self.frame_length,
                self.bandwidth,
                quality,
                &mut self.chaos_measure_old,
            );
        }
        let bits_to_pe = low_delay_bits_to_pe_factor(
            bitrate,
            self.channels,
            self.sampling_frequency,
            self.afterburner,
            false,
        );
        let static_bits = 20 * self.channels;
        let total_pe = psycho.iter().map(|value| value.perceptual_entropy).sum();
        let available = if self.vbr_quality_factor.is_some() {
            self.reservoir.available_frame_bits()
        } else {
            static_bits
                + low_delay_granted_dynamic_bits(
                    &self.reservoir,
                    total_pe,
                    bits_to_pe,
                    static_bits,
                    &mut self.pe_correction,
                )
        };
        if self.vbr_quality_factor.is_none() {
            let granted_pe = available.saturating_sub(static_bits) as f32 * bits_to_pe;
            let target_pe = self
                .pe_correction
                .corrected_grant(total_pe, granted_pe, bits_to_pe);
            let mut refs = psycho.iter_mut().collect::<Vec<_>>();
            apply_low_delay_cbr_thresholds(
                &mut refs,
                offsets,
                target_pe,
                bitrate,
                self.sampling_frequency,
                self.frame_length,
                self.bandwidth,
            );
        }
        let (frames, raw) = fit_low_delay_access_unit(available, |relaxation| {
            let frames = spectra
                .iter()
                .zip(&psycho)
                .map(|(spectrum, psycho)| {
                    quantize_drm_spectrum(spectrum, psycho, offsets, relaxation)
                })
                .collect::<Vec<_>>();
            let raw = write_er_aac_ld_multichannel_access_unit(
                &frames,
                self.channel_mode,
                self.window_shape,
                dynamic_range,
                ancillary_elements,
                0,
            )?;
            Ok((frames, raw))
        })?;
        self.pe_correction.commit_dynamic_bits(
            frames
                .iter()
                .map(|frame| frame.estimated_spectral_bits + frame.estimated_section_bits)
                .sum(),
        );
        let raw = add_er_cbr_fill(raw, self.cbr_fill_enabled, &self.reservoir, |fill_bytes| {
            write_er_aac_ld_multichannel_access_unit(
                &frames,
                self.channel_mode,
                self.window_shape,
                dynamic_range,
                ancillary_elements,
                fill_bytes,
            )
        })?;
        self.reservoir.commit_frame(raw.len() * 8)?;
        Ok(raw)
    }
}

fn eld_audio_specific_config(
    sampling_frequency_index: u8,
    sampling_frequency: u32,
    frame_length: usize,
    channel_configuration: u8,
    sbr: Option<(&LdSbrHeader, bool, bool)>,
    extensions: Vec<EldExtension>,
) -> AudioSpecificConfig {
    AudioSpecificConfig {
        audio_object_type: 39,
        sampling_frequency_index,
        sampling_frequency,
        channel_configuration,
        extension: None,
        ga_specific: None,
        eld_specific: Some(EldSpecificConfig {
            frame_length_flag: frame_length == 480,
            section_data_resilience: false,
            scalefactor_data_resilience: false,
            spectral_data_resilience: false,
            sbr_present: sbr.is_some(),
            sbr_sampling_rate: sbr.is_some_and(|(_, dual_rate, _)| dual_rate),
            sbr_crc: sbr.is_some_and(|(_, _, crc)| crc),
            sbr_headers: sbr
                .map(|(header, _, _)| vec![header.clone()])
                .unwrap_or_default(),
            extensions,
        }),
        usac_config: None,
        error_protection_config: Some(0),
        program_config: None,
        bits_read: 0,
    }
}

impl PureRustAacEldMonoEncoder {
    pub(crate) fn set_sbr_noise_max_level(&mut self, level_db: i8) {
        let cap = 2.0_f64.powf(f64::from(level_db) / 3.0) * 0.25;
        self.sbr_coding_state = LowDelaySbrCodingState::default().with_noise_floor_cap(cap);
    }

    pub fn new(
        sampling_frequency_index: u8,
        frame_length: usize,
        nominal_frame_bits: usize,
        reservoir_capacity_bits: usize,
    ) -> Result<Self, AacLcEncoderError> {
        if !matches!(frame_length, 480 | 512) {
            return Err(AacLcEncoderError::UnsupportedFrameLength(frame_length));
        }
        let sampling_frequency = sample_rate_from_index(sampling_frequency_index)
            .ok_or(AacLcEncoderError::InvalidRawElementLayout)?;
        let ics = er_long_ics(sampling_frequency_index, frame_length)?;
        aac_band_offsets_for_ics(sampling_frequency_index, &ics, frame_length)?;
        Ok(Self {
            sampling_frequency_index,
            sampling_frequency,
            frame_length,
            bandwidth: sampling_frequency / 2,
            analysis: EldAnalysisFilterbank::new(frame_length)?,
            reservoir: AacLcBitReservoir::new_full(nominal_frame_bits, reservoir_capacity_bits),
            cbr_fill_enabled: false,
            afterburner: false,
            vbr_quality_factor: None,
            chaos_measure_old: 0.3,
            pe_correction: LowDelayPeCorrection::default(),
            sbr_analysis: None,
            sbr_header: None,
            sbr_dual_rate: false,
            sbr_crc: false,
            sbr_header_written: false,
            sbr_downsampler: None,
            sbr_coding_state: LowDelaySbrCodingState::default(),
        })
    }

    pub fn audio_specific_config(&self) -> AudioSpecificConfig {
        eld_audio_specific_config(
            self.sampling_frequency_index,
            self.sampling_frequency,
            self.frame_length,
            1,
            self.sbr_header
                .as_ref()
                .map(|header| (header, self.sbr_dual_rate, self.sbr_crc)),
            Vec::new(),
        )
    }

    pub(crate) fn audio_specific_config_with_extensions(
        &self,
        extensions: Vec<EldExtension>,
    ) -> AudioSpecificConfig {
        eld_audio_specific_config(
            self.sampling_frequency_index,
            self.sampling_frequency,
            self.frame_length,
            1,
            self.sbr_header
                .as_ref()
                .map(|header| (header, self.sbr_dual_rate, self.sbr_crc)),
            extensions,
        )
    }

    pub fn enable_sbr(
        &mut self,
        header: LdSbrHeader,
        dual_rate: bool,
    ) -> Result<(), AacLcEncoderError> {
        self.enable_sbr_with_crc(header, dual_rate, false)
    }

    pub fn enable_sbr_with_crc(
        &mut self,
        header: LdSbrHeader,
        dual_rate: bool,
        crc: bool,
    ) -> Result<(), AacLcEncoderError> {
        self.sbr_analysis = Some(SbrEncoderAnalysis::new_low_delay(
            &header,
            self.sampling_frequency,
            dual_rate,
        )?);
        self.sbr_header = Some(header);
        self.sbr_dual_rate = dual_rate;
        self.sbr_crc = crc;
        self.sbr_header_written = false;
        self.sbr_coding_state = LowDelaySbrCodingState::default();
        self.sbr_downsampler = dual_rate.then(HalfbandDownsampler::new);
        Ok(())
    }

    pub fn set_bandwidth(&mut self, bandwidth: u32) {
        self.bandwidth = bandwidth.min(self.sampling_frequency / 2);
    }

    pub fn bit_reservoir(&self) -> &AacLcBitReservoir {
        &self.reservoir
    }

    pub fn set_cbr_fill_enabled(&mut self, enabled: bool) {
        self.cbr_fill_enabled = enabled;
    }

    pub fn set_afterburner(&mut self, enabled: bool) {
        self.afterburner = enabled;
    }

    pub fn set_bitrate_mode(&mut self, mode: u32) {
        self.vbr_quality_factor = low_delay_vbr_quality_factor(mode);
    }

    pub fn encode_spectrum(&mut self, spectrum: &[f32]) -> Result<Vec<u8>, AacLcEncoderError> {
        self.encode_spectrum_with_ancillary(spectrum, &[])
    }

    pub fn encode_spectrum_with_ancillary(
        &mut self,
        spectrum: &[f32],
        ancillary: &[u8],
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        let ancillary_elements = [ancillary];
        self.encode_spectrum_with_extensions(spectrum, None, &ancillary_elements)
    }

    pub(crate) fn encode_spectrum_with_extensions(
        &mut self,
        spectrum: &[f32],
        dynamic_range: Option<(&[u8], usize)>,
        ancillary_elements: &[&[u8]],
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        self.encode_spectrum_with_extensions_and_sbr(
            spectrum,
            dynamic_range,
            ancillary_elements,
            None,
            None,
        )
    }

    fn encode_spectrum_with_extensions_and_sbr(
        &mut self,
        spectrum: &[f32],
        dynamic_range: Option<(&[u8], usize)>,
        ancillary_elements: &[&[u8]],
        sbr: Option<EldMonoSbrPayload<'_>>,
        mps: Option<(&[u8], usize)>,
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        if spectrum.len() != self.frame_length {
            return Err(AacLcEncoderError::InputLengthMismatch {
                expected: self.frame_length,
                actual: spectrum.len(),
            });
        }
        if spectrum.iter().any(|sample| !sample.is_finite()) {
            return Err(AacLcEncoderError::NonFiniteInput);
        }
        let ics = er_long_ics(self.sampling_frequency_index, self.frame_length)?;
        let offsets =
            aac_band_offsets_for_ics(self.sampling_frequency_index, &ics, self.frame_length)?
                .offsets;
        let mut spectrum = spectrum.to_vec();
        apply_spectral_bandwidth(&mut spectrum, self.sampling_frequency, self.bandwidth);
        let mut psycho = analyze_low_delay_psychoacoustic_bands(
            &spectrum,
            offsets,
            self.reservoir.nominal_frame_bits(),
            self.sampling_frequency,
            self.frame_length,
            self.bandwidth,
            1,
        );
        let bitrate = ((self.reservoir.nominal_frame_bits() as u64
            * u64::from(self.sampling_frequency))
            / self.frame_length as u64) as u32;
        if let Some(quality) = self.vbr_quality_factor {
            apply_low_delay_vbr_thresholds(
                &mut [&mut psycho],
                offsets,
                bitrate,
                self.sampling_frequency,
                self.frame_length,
                self.bandwidth,
                quality,
                &mut self.chaos_measure_old,
            );
        }
        let bits_to_pe = low_delay_bits_to_pe_factor(
            bitrate,
            1,
            self.sampling_frequency,
            self.afterburner,
            bitrate < 32_000,
        );
        // Spatial/SBR payload rates are accounted for by the ELD wrapper
        // before the core QC stage. Subtract only the core element syntax
        // here; charging extensions again would double-count them.
        let static_bits = 16;
        let available = if self.vbr_quality_factor.is_some() {
            self.reservoir.available_frame_bits()
        } else {
            static_bits
                + low_delay_granted_dynamic_bits(
                    &self.reservoir,
                    psycho.perceptual_entropy,
                    bits_to_pe,
                    static_bits,
                    &mut self.pe_correction,
                )
        };
        if self.vbr_quality_factor.is_none() {
            let granted_pe = available.saturating_sub(static_bits) as f32 * bits_to_pe;
            let target_pe = self.pe_correction.corrected_grant(
                psycho.perceptual_entropy,
                granted_pe,
                bits_to_pe,
            );
            apply_low_delay_cbr_thresholds(
                &mut [&mut psycho],
                offsets,
                target_pe,
                bitrate,
                self.sampling_frequency,
                self.frame_length,
                self.bandwidth,
            );
        }
        let (frame, raw) = fit_low_delay_access_unit(available, |relaxation| {
            let frame = quantize_drm_spectrum(&spectrum, &psycho, offsets, relaxation);
            let raw = write_er_aac_eld_mono_access_unit(
                &frame,
                sbr,
                mps,
                dynamic_range,
                ancillary_elements,
                0,
            )?;
            Ok((frame, raw))
        })?;
        self.pe_correction
            .commit_dynamic_bits(frame.estimated_spectral_bits + frame.estimated_section_bits);
        let raw = add_er_cbr_fill(raw, self.cbr_fill_enabled, &self.reservoir, |fill_bytes| {
            write_er_aac_eld_mono_access_unit(
                &frame,
                sbr,
                mps,
                dynamic_range,
                ancillary_elements,
                fill_bytes,
            )
        })?;
        self.reservoir.commit_frame(raw.len() * 8)?;
        Ok(raw)
    }

    pub fn encode_pcm(&mut self, input: &[f32]) -> Result<Vec<u8>, AacLcEncoderError> {
        self.encode_pcm_with_ancillary(input, &[])
    }

    pub fn encode_pcm_with_ancillary(
        &mut self,
        input: &[f32],
        ancillary: &[u8],
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        let ancillary_elements = [ancillary];
        self.encode_pcm_with_extensions(input, None, &ancillary_elements)
    }

    pub(crate) fn encode_pcm_with_extensions(
        &mut self,
        input: &[f32],
        dynamic_range: Option<(&[u8], usize)>,
        ancillary_elements: &[&[u8]],
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        self.encode_pcm_with_optional_mps_extensions(input, None, dynamic_range, ancillary_elements)
    }

    fn encode_pcm_with_optional_mps_extensions(
        &mut self,
        input: &[f32],
        mps: Option<(&[u8], usize)>,
        dynamic_range: Option<(&[u8], usize)>,
        ancillary_elements: &[&[u8]],
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        let expected = self.frame_length * if self.sbr_dual_rate { 2 } else { 1 };
        if input.len() != expected {
            return Err(AacLcEncoderError::InputLengthMismatch {
                expected,
                actual: input.len(),
            });
        }
        let mut sbr_frame = self
            .sbr_analysis
            .as_mut()
            .map(|analysis| analysis.analyze_low_delay(input, self.frame_length))
            .transpose()?;
        let core = if let Some(downsampler) = &mut self.sbr_downsampler {
            downsampler.process(input)
        } else {
            input.to_vec()
        };
        let spectrum = self.analysis.analyze(&core)?;
        let sbr_header = self.sbr_header.clone();
        let sbr_tables = self
            .sbr_analysis
            .as_ref()
            .map(|analysis| analysis.frequency_tables().clone());
        if let Some(frame) = sbr_frame.as_mut() {
            let bitrate = (self.reservoir.nominal_frame_bits as u64
                * u64::from(self.sampling_frequency)
                / self.frame_length as u64) as u32;
            frame.select_low_delay_amp_resolution(bitrate);
        }
        if let (Some(frame), Some(header), Some(tables)) =
            (sbr_frame.as_mut(), sbr_header.as_ref(), sbr_tables.as_ref())
        {
            frame.prepare_mono_low_delay_coding(
                header,
                tables,
                !self.sbr_header_written,
                &mut self.sbr_coding_state,
            );
        }
        let sbr = sbr_frame
            .as_ref()
            .zip(sbr_header.as_ref())
            .zip(sbr_tables.as_ref())
            .map(|((frame, header), tables)| EldMonoSbrPayload {
                frame,
                header,
                tables,
                header_present: !self.sbr_header_written,
                crc_present: self.sbr_crc,
            });
        let raw = self.encode_spectrum_with_extensions_and_sbr(
            &spectrum,
            dynamic_range,
            ancillary_elements,
            sbr,
            mps,
        )?;
        if sbr_frame.is_some() {
            self.sbr_header_written = true;
        }
        Ok(raw)
    }

    pub(crate) fn encode_pcm_with_mps_extensions(
        &mut self,
        input: &[f32],
        mps_payload: &[u8],
        mps_payload_bits: usize,
        dynamic_range: Option<(&[u8], usize)>,
        ancillary_elements: &[&[u8]],
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        self.encode_pcm_with_optional_mps_extensions(
            input,
            Some((mps_payload, mps_payload_bits)),
            dynamic_range,
            ancillary_elements,
        )
    }
}

impl PureRustAacEldMultichannelEncoder {
    pub(crate) fn set_sbr_noise_max_levels(
        &mut self,
        element_levels_db: &[i8],
    ) -> Result<(), AacLcEncoderError> {
        let layout = multichannel_layout_for_mode(self.channel_mode)?;
        if element_levels_db.len()
            != layout
                .iter()
                .filter(|(id, _, _)| *id != ElementId::Lfe)
                .count()
        {
            return Err(AacLcEncoderError::InvalidRawElementLayout);
        }
        let mut element = 0;
        for (id, first, second) in layout {
            if id == ElementId::Lfe {
                continue;
            }
            let cap = 2.0_f64.powf(f64::from(element_levels_db[element]) / 3.0) * 0.25;
            let first = multichannel_sbr_channel_index(self.channel_mode, first)?;
            self.sbr_coding_states[first] =
                LowDelaySbrCodingState::default().with_noise_floor_cap(cap);
            if let Some(second) = second
                .map(|channel| multichannel_sbr_channel_index(self.channel_mode, channel))
                .transpose()?
            {
                self.sbr_coding_states[second] =
                    LowDelaySbrCodingState::default().with_noise_floor_cap(cap);
            }
            element += 1;
        }
        Ok(())
    }

    pub fn new(
        sampling_frequency_index: u8,
        channels: usize,
        frame_length: usize,
        nominal_frame_bits: usize,
        reservoir_capacity_bits: usize,
    ) -> Result<Self, AacLcEncoderError> {
        Self::new_with_channel_mode(
            sampling_frequency_index,
            channels,
            channels as u32,
            frame_length,
            nominal_frame_bits,
            reservoir_capacity_bits,
        )
    }

    pub fn new_with_channel_mode(
        sampling_frequency_index: u8,
        channels: usize,
        channel_mode: u32,
        frame_length: usize,
        nominal_frame_bits: usize,
        reservoir_capacity_bits: usize,
    ) -> Result<Self, AacLcEncoderError> {
        let layout = multichannel_layout_for_mode(channel_mode)?;
        let layout_channels = layout
            .iter()
            .flat_map(|(_, first, second)| [Some(*first), *second])
            .flatten()
            .max()
            .map_or(0, |last| last + 1);
        if layout_channels != channels || !matches!(frame_length, 480 | 512) {
            return Err(AacLcEncoderError::InvalidRawElementLayout);
        }
        let sampling_frequency = sample_rate_from_index(sampling_frequency_index)
            .ok_or(AacLcEncoderError::InvalidRawElementLayout)?;
        let ics = er_long_ics(sampling_frequency_index, frame_length)?;
        aac_band_offsets_for_ics(sampling_frequency_index, &ics, frame_length)?;
        let analyses = (0..channels)
            .map(|_| EldAnalysisFilterbank::new(frame_length))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            sampling_frequency_index,
            sampling_frequency,
            frame_length,
            channels,
            channel_mode,
            bandwidth: sampling_frequency / 2,
            analyses,
            reservoir: AacLcBitReservoir::new_full(nominal_frame_bits, reservoir_capacity_bits),
            cbr_fill_enabled: false,
            afterburner: false,
            sbr_analyses: Vec::new(),
            sbr_headers: Vec::new(),
            sbr_element_bitrates: Vec::new(),
            sbr_dual_rate: false,
            sbr_crc: false,
            sbr_header_written: false,
            sbr_downsamplers: Vec::new(),
            sbr_coding_states: Vec::new(),
        })
    }

    pub fn audio_specific_config(&self) -> AudioSpecificConfig {
        let mut asc = eld_audio_specific_config(
            self.sampling_frequency_index,
            self.sampling_frequency,
            self.frame_length,
            self.channel_mode as u8,
            self.sbr_headers.first().map(|header| {
                // `eld_audio_specific_config` accepts one header; replace the
                // list below for the per-element ELD multichannel syntax.
                (header, self.sbr_dual_rate, self.sbr_crc)
            }),
            Vec::new(),
        );
        if let Some(eld) = &mut asc.eld_specific {
            eld.sbr_headers = self.sbr_headers.clone();
        }
        asc
    }

    pub fn enable_sbr(
        &mut self,
        header: LdSbrHeader,
        dual_rate: bool,
    ) -> Result<(), AacLcEncoderError> {
        let header_count = multichannel_layout_for_mode(self.channel_mode)?
            .iter()
            .filter(|(id, _, _)| *id != ElementId::Lfe)
            .count();
        self.enable_sbr_headers(vec![header; header_count], dual_rate)
    }

    pub fn enable_sbr_headers(
        &mut self,
        headers: Vec<LdSbrHeader>,
        dual_rate: bool,
    ) -> Result<(), AacLcEncoderError> {
        let total_bitrate = (self.reservoir.nominal_frame_bits as u64
            * u64::from(self.sampling_frequency)
            / self.frame_length as u64) as u32;
        let sbr_channels = multichannel_non_lfe_channels(self.channel_mode)?.len();
        let element_bitrates = multichannel_layout_for_mode(self.channel_mode)?
            .into_iter()
            .filter(|(id, _, _)| *id != ElementId::Lfe)
            .map(|(_, _, second)| {
                let channels = 1 + usize::from(second.is_some());
                ((u64::from(total_bitrate) * channels as u64) / sbr_channels as u64) as u32
            })
            .collect();
        self.enable_sbr_headers_with_bitrates(headers, element_bitrates, dual_rate)
    }

    pub(crate) fn enable_sbr_headers_with_bitrates(
        &mut self,
        headers: Vec<LdSbrHeader>,
        element_bitrates: Vec<u32>,
        dual_rate: bool,
    ) -> Result<(), AacLcEncoderError> {
        let layout = multichannel_layout_for_mode(self.channel_mode)?;
        let expected_headers = layout
            .iter()
            .filter(|(id, _, _)| *id != ElementId::Lfe)
            .count();
        if headers.len() != expected_headers || element_bitrates.len() != expected_headers {
            return Err(AacLcEncoderError::InvalidRawElementLayout);
        }
        let non_lfe_channels = multichannel_non_lfe_channels(self.channel_mode)?;
        let sbr_channels = non_lfe_channels.len();
        let mut channel_headers = vec![None; self.channels];
        let mut header_index = 0;
        for (id, first, second) in layout {
            if id == ElementId::Lfe {
                continue;
            }
            channel_headers[first] = Some(&headers[header_index]);
            if let Some(second) = second {
                channel_headers[second] = Some(&headers[header_index]);
            }
            header_index += 1;
        }
        self.sbr_analyses = non_lfe_channels
            .iter()
            .map(|&channel| channel_headers[channel])
            .map(|header| {
                SbrEncoderAnalysis::new_low_delay(
                    header.expect("every non-LFE channel has an SBR header"),
                    self.sampling_frequency,
                    dual_rate,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.sbr_coding_states = vec![LowDelaySbrCodingState::default(); sbr_channels];
        self.sbr_downsamplers = if dual_rate {
            (0..self.channels)
                .map(|_| HalfbandDownsampler::new())
                .collect()
        } else {
            Vec::new()
        };
        self.sbr_headers = headers;
        self.sbr_element_bitrates = element_bitrates;
        self.sbr_dual_rate = dual_rate;
        self.sbr_header_written = false;
        Ok(())
    }

    pub fn set_bandwidth(&mut self, bandwidth: u32) {
        self.bandwidth = bandwidth.min(self.sampling_frequency / 2);
    }

    pub fn set_cbr_fill_enabled(&mut self, enabled: bool) {
        self.cbr_fill_enabled = enabled;
    }

    pub fn set_afterburner(&mut self, enabled: bool) {
        self.afterburner = enabled;
    }

    pub fn encode_pcm(&mut self, pcm: &[Vec<f32>]) -> Result<Vec<u8>, AacLcEncoderError> {
        let input_frame_length = self.frame_length * if self.sbr_dual_rate { 2 } else { 1 };
        if pcm.len() != self.channels
            || pcm
                .iter()
                .any(|channel| channel.len() != input_frame_length)
        {
            return Err(AacLcEncoderError::InputLengthMismatch {
                expected: self.channels * input_frame_length,
                actual: pcm.iter().map(Vec::len).sum(),
            });
        }
        if pcm.iter().flatten().any(|sample| !sample.is_finite()) {
            return Err(AacLcEncoderError::NonFiniteInput);
        }
        let mut sbr_frames = self
            .sbr_analyses
            .iter_mut()
            .zip(
                multichannel_non_lfe_channels(self.channel_mode)?
                    .into_iter()
                    .map(|index| &pcm[index]),
            )
            .map(|(analysis, samples)| analysis.analyze_low_delay(samples, self.frame_length))
            .collect::<Result<Vec<_>, _>>()?;
        let core_pcm = if self.sbr_dual_rate {
            self.sbr_downsamplers
                .iter_mut()
                .zip(pcm)
                .map(|(downsampler, samples)| downsampler.process(samples))
                .collect::<Vec<_>>()
        } else {
            pcm.to_vec()
        };
        let headers = self.sbr_headers.clone();
        let tables = self
            .sbr_analyses
            .iter()
            .map(|analysis| analysis.frequency_tables().clone())
            .collect::<Vec<_>>();
        if !headers.is_empty() {
            let header_present = !self.sbr_header_written;
            let mut header_index = 0;
            for (id, first, second) in multichannel_layout_for_mode(self.channel_mode)? {
                if id == ElementId::Lfe {
                    continue;
                }
                let first = multichannel_sbr_channel_index(self.channel_mode, first)?;
                let header = &headers[header_index];
                let element_bitrate = self.sbr_element_bitrates[header_index];
                let tables = &tables[first];
                if let Some(second) = second
                    .map(|channel| multichannel_sbr_channel_index(self.channel_mode, channel))
                    .transpose()?
                {
                    let (left_frames, right_frames) = sbr_frames.split_at_mut(second);
                    let left = &mut left_frames[first];
                    let right = &mut right_frames[0];
                    left.select_low_delay_amp_resolution(element_bitrate);
                    right.select_low_delay_amp_resolution(element_bitrate);
                    let (left_states, right_states) = self.sbr_coding_states.split_at_mut(second);
                    if SbrEncoderAnalysisFrame::uses_low_delay_coupling(left, right) {
                        SbrEncoderAnalysisFrame::prepare_coupled_low_delay_coding(
                            left,
                            right,
                            header,
                            tables,
                            header_present,
                            &mut left_states[first],
                            &mut right_states[0],
                        )?;
                    } else {
                        left.prepare_mono_low_delay_coding(
                            header,
                            tables,
                            header_present,
                            &mut left_states[first],
                        );
                        right.prepare_mono_low_delay_coding(
                            header,
                            tables,
                            header_present,
                            &mut right_states[0],
                        );
                        SbrEncoderAnalysisFrame::synchronize_stereo_time_streak(
                            left,
                            right,
                            &mut left_states[first],
                            &mut right_states[0],
                        );
                    }
                } else {
                    sbr_frames[first].select_low_delay_amp_resolution(element_bitrate);
                    sbr_frames[first].prepare_mono_low_delay_coding(
                        header,
                        tables,
                        header_present,
                        &mut self.sbr_coding_states[first],
                    );
                }
                header_index += 1;
            }
        }
        let sbr_payloads = if !headers.is_empty() {
            Some(EldMultichannelSbrPayloads {
                frames: &sbr_frames,
                headers: &headers,
                tables: &tables,
                header_present: !self.sbr_header_written,
                crc_present: self.sbr_crc,
            })
        } else {
            None
        };
        let ics = er_long_ics(self.sampling_frequency_index, self.frame_length)?;
        let offsets =
            aac_band_offsets_for_ics(self.sampling_frequency_index, &ics, self.frame_length)?
                .offsets;
        let mut spectra = Vec::with_capacity(self.channels);
        let mut psycho = Vec::with_capacity(self.channels);
        for (analysis, samples) in self.analyses.iter_mut().zip(&core_pcm) {
            let mut spectrum = analysis.analyze(samples)?;
            apply_spectral_bandwidth(&mut spectrum, self.sampling_frequency, self.bandwidth);
            psycho.push(analyze_low_delay_psychoacoustic_bands(
                &spectrum,
                offsets,
                self.reservoir.nominal_frame_bits(),
                self.sampling_frequency,
                self.frame_length,
                self.bandwidth,
                self.channels,
            ));
            spectra.push(spectrum);
        }
        let available = self.reservoir.available_frame_bits();
        let mut relaxation = if self.afterburner { 0.85 } else { 1.0 };
        let (frames, mut raw) = loop {
            let frames = spectra
                .iter()
                .zip(&psycho)
                .map(|(spectrum, psycho)| {
                    quantize_drm_spectrum(spectrum, psycho, offsets, relaxation)
                })
                .collect::<Vec<_>>();
            let raw = write_er_aac_eld_multichannel_access_unit(
                &frames,
                self.channel_mode,
                sbr_payloads,
                0,
            )?;
            if raw.len() * 8 <= available || relaxation >= 1.0e9 {
                break (frames, raw);
            }
            relaxation *= 1.35;
        };
        if raw.len() * 8 > available {
            return Err(AacLcEncoderError::BitReservoirUnderflow {
                available,
                requested: raw.len() * 8,
            });
        }
        if self.cbr_fill_enabled {
            raw = add_er_cbr_fill(raw, true, &self.reservoir, |fill_bytes| {
                write_er_aac_eld_multichannel_access_unit(
                    &frames,
                    self.channel_mode,
                    sbr_payloads,
                    fill_bytes,
                )
            })?;
        }
        self.reservoir.commit_frame(raw.len() * 8)?;
        if !sbr_frames.is_empty() {
            self.sbr_header_written = true;
        }
        Ok(raw)
    }
}

impl PureRustAacEldMpsEncoder {
    pub fn new(
        sampling_frequency_index: u8,
        frame_length: usize,
        nominal_frame_bits: usize,
        reservoir_capacity_bits: usize,
    ) -> Result<Self, EldMpsEncoderError> {
        let sampling_frequency = sample_rate_from_index(sampling_frequency_index)
            .ok_or(EldMpsEncoderError::InvalidFrameGeometry)?;
        Self::new_with_spatial_geometry(
            sampling_frequency_index,
            frame_length,
            nominal_frame_bits,
            reservoir_capacity_bits,
            sampling_frequency,
            frame_length,
        )
    }

    pub fn new_with_spatial_geometry(
        core_sampling_frequency_index: u8,
        core_frame_length: usize,
        nominal_frame_bits: usize,
        reservoir_capacity_bits: usize,
        spatial_sampling_frequency: u32,
        spatial_frame_length: usize,
    ) -> Result<Self, EldMpsEncoderError> {
        let qmf_bands = if spatial_sampling_frequency < 27_713 {
            32
        } else {
            64
        };
        if spatial_frame_length % qmf_bands != 0 {
            return Err(EldMpsEncoderError::InvalidFrameGeometry);
        }
        let time_slots = u8::try_from(spatial_frame_length / qmf_bands)
            .map_err(|_| EldMpsEncoderError::InvalidFrameGeometry)?;
        Ok(Self {
            core: PureRustAacEldMonoEncoder::new(
                core_sampling_frequency_index,
                core_frame_length,
                nominal_frame_bits,
                reservoir_capacity_bits,
            )?,
            spatial: Sac212Encoder::new(spatial_sampling_frequency, time_slots)?,
        })
    }

    pub fn audio_specific_config(&self) -> Result<AudioSpecificConfig, EldMpsEncoderError> {
        let (ssc, _) = self
            .spatial
            .config()
            .write()
            .map_err(SacEncodeError::from)?;
        Ok(self
            .core
            .audio_specific_config_with_extensions(vec![EldExtension {
                extension_type: 0x02, // ELDEXT_LDSAC
                data: ssc,
            }]))
    }

    pub fn set_bandwidth(&mut self, bandwidth: u32) {
        self.core.set_bandwidth(bandwidth);
    }

    pub fn enable_sbr(
        &mut self,
        header: LdSbrHeader,
        dual_rate: bool,
    ) -> Result<(), EldMpsEncoderError> {
        self.core.enable_sbr(header, dual_rate)?;
        Ok(())
    }

    pub(crate) fn set_sbr_noise_max_level(&mut self, level_db: i8) {
        self.core.set_sbr_noise_max_level(level_db);
    }

    pub fn set_cbr_fill_enabled(&mut self, enabled: bool) {
        self.core.set_cbr_fill_enabled(enabled);
    }

    pub fn set_afterburner(&mut self, enabled: bool) {
        self.core.set_afterburner(enabled);
    }

    pub fn set_bitrate_mode(&mut self, mode: u32) {
        self.core.set_bitrate_mode(mode);
    }

    pub fn encode_pcm(
        &mut self,
        left: &[f32],
        right: &[f32],
    ) -> Result<Vec<u8>, EldMpsEncoderError> {
        self.encode_pcm_with_extensions(left, right, None, &[])
    }

    pub(crate) fn encode_pcm_with_extensions(
        &mut self,
        left: &[f32],
        right: &[f32],
        dynamic_range: Option<(&[u8], usize)>,
        ancillary_elements: &[&[u8]],
    ) -> Result<Vec<u8>, EldMpsEncoderError> {
        let spatial = self.spatial.encode(left, right)?;
        Ok(self.core.encode_pcm_with_mps_extensions(
            &spatial.downmix,
            &spatial.payload,
            spatial.payload_bits,
            dynamic_range,
            ancillary_elements,
        )?)
    }
}

impl PureRustAacLdMpsEncoder {
    pub fn new(
        sampling_frequency_index: u8,
        frame_length: usize,
        nominal_frame_bits: usize,
        reservoir_capacity_bits: usize,
    ) -> Result<Self, EldMpsEncoderError> {
        let sampling_frequency = sample_rate_from_index(sampling_frequency_index)
            .ok_or(EldMpsEncoderError::InvalidFrameGeometry)?;
        // AAC-LD MPS uses the 32-band low-delay analysis bank at every
        // supported rate (15/16 slots for 480/512-sample frames).
        let qmf_bands = 32;
        if frame_length % qmf_bands != 0 {
            return Err(EldMpsEncoderError::InvalidFrameGeometry);
        }
        let time_slots = u8::try_from(frame_length / qmf_bands)
            .map_err(|_| EldMpsEncoderError::InvalidFrameGeometry)?;
        Ok(Self {
            core: PureRustAacLdMonoEncoder::new(
                sampling_frequency_index,
                frame_length,
                nominal_frame_bits,
                reservoir_capacity_bits,
            )?,
            spatial: Sac212Encoder::new_with_qmf_bands(sampling_frequency, time_slots, qmf_bands)?,
        })
    }

    pub fn audio_specific_config(&self) -> AudioSpecificConfig {
        let mut asc = self.core.audio_specific_config();
        if let Some(ga) = &mut asc.ga_specific {
            ga.extension_flag = true;
            ga.extension_flag3 = Some(false);
        }
        asc
    }

    pub fn set_bandwidth(&mut self, bandwidth: u32) {
        self.core.set_bandwidth(bandwidth);
    }

    pub fn set_cbr_fill_enabled(&mut self, enabled: bool) {
        self.core.set_cbr_fill_enabled(enabled);
    }

    pub fn set_afterburner(&mut self, enabled: bool) {
        self.core.set_afterburner(enabled);
    }

    pub fn set_bitrate_mode(&mut self, mode: u32) {
        self.core.set_bitrate_mode(mode);
    }

    pub(crate) fn encode_pcm_with_extensions(
        &mut self,
        left: &[f32],
        right: &[f32],
        dynamic_range: Option<(&[u8], usize)>,
        ancillary_elements: &[&[u8]],
    ) -> Result<Vec<u8>, EldMpsEncoderError> {
        let spatial = self.spatial.encode(left, right)?;
        Ok(self.core.encode_pcm_with_mps_extensions(
            &spatial.downmix,
            &spatial.payload,
            spatial.payload_bits,
            dynamic_range,
            ancillary_elements,
        )?)
    }
}

impl PureRustAacEldStereoEncoder {
    pub fn new(
        sampling_frequency_index: u8,
        frame_length: usize,
        nominal_frame_bits: usize,
        reservoir_capacity_bits: usize,
    ) -> Result<Self, AacLcEncoderError> {
        if !matches!(frame_length, 480 | 512) {
            return Err(AacLcEncoderError::UnsupportedFrameLength(frame_length));
        }
        let sampling_frequency = sample_rate_from_index(sampling_frequency_index)
            .ok_or(AacLcEncoderError::InvalidRawElementLayout)?;
        let ics = er_long_ics(sampling_frequency_index, frame_length)?;
        aac_band_offsets_for_ics(sampling_frequency_index, &ics, frame_length)?;
        Ok(Self {
            sampling_frequency_index,
            sampling_frequency,
            frame_length,
            bandwidth: sampling_frequency / 2,
            left_analysis: EldAnalysisFilterbank::new(frame_length)?,
            right_analysis: EldAnalysisFilterbank::new(frame_length)?,
            reservoir: AacLcBitReservoir::new_full(nominal_frame_bits, reservoir_capacity_bits),
            cbr_fill_enabled: false,
            afterburner: false,
            vbr_quality_factor: None,
            chaos_measure_old: 0.3,
            pe_correction: LowDelayPeCorrection::default(),
            left_sbr_analysis: None,
            right_sbr_analysis: None,
            sbr_header: None,
            sbr_dual_rate: false,
            sbr_crc: false,
            sbr_header_written: false,
            left_sbr_downsampler: None,
            right_sbr_downsampler: None,
            left_sbr_coding_state: LowDelaySbrCodingState::default().with_noise_floor_cap(0.125),
            right_sbr_coding_state: LowDelaySbrCodingState::default().with_noise_floor_cap(0.125),
            left_sbr_input_delay: [0.0; 5],
            right_sbr_input_delay: [0.0; 5],
            #[cfg(test)]
            last_sbr_prequant_debug: None,
        })
    }

    pub fn audio_specific_config(&self) -> AudioSpecificConfig {
        eld_audio_specific_config(
            self.sampling_frequency_index,
            self.sampling_frequency,
            self.frame_length,
            2,
            self.sbr_header
                .as_ref()
                .map(|header| (header, self.sbr_dual_rate, self.sbr_crc)),
            Vec::new(),
        )
    }

    pub fn enable_sbr(
        &mut self,
        header: LdSbrHeader,
        dual_rate: bool,
    ) -> Result<(), AacLcEncoderError> {
        self.enable_sbr_with_crc(header, dual_rate, false)
    }

    pub fn enable_sbr_with_crc(
        &mut self,
        header: LdSbrHeader,
        dual_rate: bool,
        crc: bool,
    ) -> Result<(), AacLcEncoderError> {
        self.left_sbr_analysis = Some(SbrEncoderAnalysis::new_low_delay(
            &header,
            self.sampling_frequency,
            dual_rate,
        )?);
        self.right_sbr_analysis = Some(SbrEncoderAnalysis::new_low_delay(
            &header,
            self.sampling_frequency,
            dual_rate,
        )?);
        self.sbr_header = Some(header);
        self.sbr_dual_rate = dual_rate;
        self.sbr_crc = crc;
        self.sbr_header_written = false;
        self.left_sbr_downsampler = dual_rate.then(HalfbandDownsampler::new);
        self.right_sbr_downsampler = dual_rate.then(HalfbandDownsampler::new);
        self.left_sbr_coding_state = LowDelaySbrCodingState::default().with_noise_floor_cap(0.125);
        self.right_sbr_coding_state = LowDelaySbrCodingState::default().with_noise_floor_cap(0.125);
        Ok(())
    }

    pub fn set_bandwidth(&mut self, bandwidth: u32) {
        self.bandwidth = bandwidth.min(self.sampling_frequency / 2);
    }

    pub fn bit_reservoir(&self) -> &AacLcBitReservoir {
        &self.reservoir
    }

    pub fn set_cbr_fill_enabled(&mut self, enabled: bool) {
        self.cbr_fill_enabled = enabled;
    }

    pub fn set_afterburner(&mut self, enabled: bool) {
        self.afterburner = enabled;
    }

    pub fn set_bitrate_mode(&mut self, mode: u32) {
        self.vbr_quality_factor = low_delay_vbr_quality_factor(mode);
    }

    pub fn encode_spectra(
        &mut self,
        left: &[f32],
        right: &[f32],
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        self.encode_spectra_with_ancillary(left, right, &[])
    }

    pub fn encode_spectra_with_ancillary(
        &mut self,
        left: &[f32],
        right: &[f32],
        ancillary: &[u8],
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        let ancillary_elements = [ancillary];
        self.encode_spectra_with_extensions(left, right, None, &ancillary_elements)
    }

    pub(crate) fn encode_spectra_with_extensions(
        &mut self,
        left: &[f32],
        right: &[f32],
        dynamic_range: Option<(&[u8], usize)>,
        ancillary_elements: &[&[u8]],
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        self.encode_spectra_with_extensions_and_sbr(
            left,
            right,
            dynamic_range,
            ancillary_elements,
            None,
        )
    }

    fn encode_spectra_with_extensions_and_sbr(
        &mut self,
        left: &[f32],
        right: &[f32],
        dynamic_range: Option<(&[u8], usize)>,
        ancillary_elements: &[&[u8]],
        sbr: Option<EldStereoSbrPayload<'_>>,
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        if left.len() != self.frame_length || right.len() != self.frame_length {
            return Err(AacLcEncoderError::InputLengthMismatch {
                expected: self.frame_length,
                actual: left.len().min(right.len()),
            });
        }
        if left.iter().chain(right).any(|sample| !sample.is_finite()) {
            return Err(AacLcEncoderError::NonFiniteInput);
        }
        let ics = er_long_ics(self.sampling_frequency_index, self.frame_length)?;
        let offsets =
            aac_band_offsets_for_ics(self.sampling_frequency_index, &ics, self.frame_length)?
                .offsets;
        let mut left = left.to_vec();
        let mut right = right.to_vec();
        apply_spectral_bandwidth(&mut left, self.sampling_frequency, self.bandwidth);
        apply_spectral_bandwidth(&mut right, self.sampling_frequency, self.bandwidth);
        let mut left_psycho = analyze_low_delay_psychoacoustic_bands(
            &left,
            offsets,
            self.reservoir.nominal_frame_bits(),
            self.sampling_frequency,
            self.frame_length,
            self.bandwidth,
            2,
        );
        let mut right_psycho = analyze_low_delay_psychoacoustic_bands(
            &right,
            offsets,
            self.reservoir.nominal_frame_bits(),
            self.sampling_frequency,
            self.frame_length,
            self.bandwidth,
            2,
        );
        let bitrate = ((self.reservoir.nominal_frame_bits() as u64
            * u64::from(self.sampling_frequency))
            / self.frame_length as u64) as u32;
        if let Some(quality) = self.vbr_quality_factor {
            apply_low_delay_vbr_thresholds(
                &mut [&mut left_psycho, &mut right_psycho],
                offsets,
                bitrate,
                self.sampling_frequency,
                self.frame_length,
                self.bandwidth,
                quality,
                &mut self.chaos_measure_old,
            );
        }
        let bits_to_pe = low_delay_bits_to_pe_factor(
            bitrate,
            2,
            self.sampling_frequency,
            self.afterburner,
            false,
        );
        let static_bits = 32;
        let available = if self.vbr_quality_factor.is_some() {
            self.reservoir.available_frame_bits()
        } else {
            static_bits
                + low_delay_granted_dynamic_bits(
                    &self.reservoir,
                    left_psycho.perceptual_entropy + right_psycho.perceptual_entropy,
                    bits_to_pe,
                    static_bits,
                    &mut self.pe_correction,
                )
        };
        if self.vbr_quality_factor.is_none() {
            let current_pe = left_psycho.perceptual_entropy + right_psycho.perceptual_entropy;
            let granted_pe = available.saturating_sub(static_bits) as f32 * bits_to_pe;
            let target_pe = self
                .pe_correction
                .corrected_grant(current_pe, granted_pe, bits_to_pe);
            apply_low_delay_cbr_thresholds(
                &mut [&mut left_psycho, &mut right_psycho],
                offsets,
                target_pe,
                bitrate,
                self.sampling_frequency,
                self.frame_length,
                self.bandwidth,
            );
        }
        let ((left_frame, right_frame), raw) =
            fit_low_delay_access_unit(available, |relaxation| {
                let left = quantize_drm_spectrum(&left, &left_psycho, offsets, relaxation);
                let right = quantize_drm_spectrum(&right, &right_psycho, offsets, relaxation);
                let raw = write_er_aac_eld_stereo_access_unit(
                    &left,
                    &right,
                    sbr,
                    dynamic_range,
                    ancillary_elements,
                    0,
                )?;
                Ok(((left, right), raw))
            })?;
        self.pe_correction.commit_dynamic_bits(
            left_frame.estimated_spectral_bits
                + left_frame.estimated_section_bits
                + right_frame.estimated_spectral_bits
                + right_frame.estimated_section_bits,
        );
        let raw = add_er_cbr_fill(raw, self.cbr_fill_enabled, &self.reservoir, |fill_bytes| {
            write_er_aac_eld_stereo_access_unit(
                &left_frame,
                &right_frame,
                sbr,
                dynamic_range,
                ancillary_elements,
                fill_bytes,
            )
        })?;
        self.reservoir.commit_frame(raw.len() * 8)?;
        Ok(raw)
    }

    pub fn encode_pcm(
        &mut self,
        left: &[f32],
        right: &[f32],
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        self.encode_pcm_with_ancillary(left, right, &[])
    }

    pub fn encode_pcm_with_ancillary(
        &mut self,
        left: &[f32],
        right: &[f32],
        ancillary: &[u8],
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        let ancillary_elements = [ancillary];
        self.encode_pcm_with_extensions(left, right, None, &ancillary_elements)
    }

    pub(crate) fn encode_pcm_with_extensions(
        &mut self,
        left: &[f32],
        right: &[f32],
        dynamic_range: Option<(&[u8], usize)>,
        ancillary_elements: &[&[u8]],
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        let expected = self.frame_length * if self.sbr_dual_rate { 2 } else { 1 };
        if left.len() != expected || right.len() != expected {
            return Err(AacLcEncoderError::InputLengthMismatch {
                expected,
                actual: left.len().min(right.len()),
            });
        }
        let left_sbr_input = delay_sbr_input(left, &mut self.left_sbr_input_delay);
        let right_sbr_input = delay_sbr_input(right, &mut self.right_sbr_input_delay);
        let mut left_sbr = self
            .left_sbr_analysis
            .as_mut()
            .map(|analysis| analysis.analyze_low_delay(&left_sbr_input, self.frame_length))
            .transpose()?;
        let mut right_sbr = self
            .right_sbr_analysis
            .as_mut()
            .map(|analysis| analysis.analyze_low_delay(&right_sbr_input, self.frame_length))
            .transpose()?;
        #[cfg(test)]
        if let (Some(left), Some(right)) = (&left_sbr, &right_sbr) {
            self.last_sbr_prequant_debug = left
                .low_delay_prequant_debug
                .clone()
                .zip(right.low_delay_prequant_debug.clone());
        }
        let left_core = if let Some(downsampler) = &mut self.left_sbr_downsampler {
            downsampler.process(left)
        } else {
            left.to_vec()
        };
        let right_core = if let Some(downsampler) = &mut self.right_sbr_downsampler {
            downsampler.process(right)
        } else {
            right.to_vec()
        };
        let left = self.left_analysis.analyze(&left_core)?;
        let right = self.right_analysis.analyze(&right_core)?;
        let header = self.sbr_header.clone();
        let tables = self
            .left_sbr_analysis
            .as_ref()
            .map(|analysis| analysis.frequency_tables().clone());
        let bitrate = (self.reservoir.nominal_frame_bits as u64
            * u64::from(self.sampling_frequency)
            / self.frame_length as u64) as u32;
        if let Some(frame) = left_sbr.as_mut() {
            frame.select_low_delay_amp_resolution(bitrate);
        }
        if let Some(frame) = right_sbr.as_mut() {
            frame.select_low_delay_amp_resolution(bitrate);
        }
        if let (Some(left), Some(right), Some(header), Some(tables)) = (
            left_sbr.as_mut(),
            right_sbr.as_mut(),
            header.as_ref(),
            tables.as_ref(),
        ) {
            let header_present = !self.sbr_header_written;
            if SbrEncoderAnalysisFrame::uses_low_delay_coupling(left, right) {
                SbrEncoderAnalysisFrame::prepare_coupled_low_delay_coding(
                    left,
                    right,
                    header,
                    tables,
                    header_present,
                    &mut self.left_sbr_coding_state,
                    &mut self.right_sbr_coding_state,
                )?;
            } else {
                left.prepare_mono_low_delay_coding(
                    header,
                    tables,
                    header_present,
                    &mut self.left_sbr_coding_state,
                );
                right.prepare_mono_low_delay_coding(
                    header,
                    tables,
                    header_present,
                    &mut self.right_sbr_coding_state,
                );
                SbrEncoderAnalysisFrame::synchronize_stereo_time_streak(
                    left,
                    right,
                    &mut self.left_sbr_coding_state,
                    &mut self.right_sbr_coding_state,
                );
            }
        }
        let sbr = left_sbr
            .as_ref()
            .zip(right_sbr.as_ref())
            .zip(header.as_ref())
            .zip(tables.as_ref())
            .map(|(((left, right), header), tables)| EldStereoSbrPayload {
                left,
                right,
                header,
                tables,
                header_present: !self.sbr_header_written,
                crc_present: self.sbr_crc,
            });
        let raw = self.encode_spectra_with_extensions_and_sbr(
            &left,
            &right,
            dynamic_range,
            ancillary_elements,
            sbr,
        )?;
        if left_sbr.is_some() {
            self.sbr_header_written = true;
        }
        Ok(raw)
    }
}

fn delay_sbr_input(samples: &[f32], history: &mut [f32; 5]) -> Vec<f32> {
    let delay = history.len();
    let mut delayed = Vec::with_capacity(samples.len());
    delayed.extend_from_slice(history);
    delayed.extend_from_slice(&samples[..samples.len().saturating_sub(delay)]);
    history.copy_from_slice(&samples[samples.len() - delay..]);
    delayed
}

#[derive(Debug, Clone)]
struct HalfbandDownsampler {
    taps: Vec<f64>,
    history: Vec<f64>,
}

impl HalfbandDownsampler {
    fn new() -> Self {
        let count = 33usize;
        let center = (count - 1) as f64 / 2.0;
        let cutoff = 0.24f64;
        let mut taps = (0..count)
            .map(|index| {
                let x = index as f64 - center;
                let sinc = if x == 0.0 {
                    2.0 * cutoff
                } else {
                    (2.0 * std::f64::consts::PI * cutoff * x).sin() / (std::f64::consts::PI * x)
                };
                let window = 0.54
                    - 0.46 * (2.0 * std::f64::consts::PI * index as f64 / (count - 1) as f64).cos();
                sinc * window
            })
            .collect::<Vec<_>>();
        let sum = taps.iter().sum::<f64>();
        for tap in &mut taps {
            *tap /= sum;
        }
        Self {
            taps,
            history: vec![0.0; count - 1],
        }
    }

    fn process(&mut self, input: &[f32]) -> Vec<f32> {
        let mut extended = Vec::with_capacity(self.history.len() + input.len());
        extended.extend_from_slice(&self.history);
        extended.extend(input.iter().map(|&sample| sample as f64));
        let history = self.history.len();
        let output = (1..input.len())
            .step_by(2)
            .map(|input_index| {
                let end = history + input_index;
                self.taps
                    .iter()
                    .enumerate()
                    .map(|(tap, &coefficient)| extended[end - tap] * coefficient)
                    .sum::<f64>() as f32
            })
            .collect();
        let history_len = self.history.len();
        self.history
            .copy_from_slice(&extended[extended.len() - history_len..]);
        output
    }
}

#[derive(Debug, Clone)]
pub struct PureRustHeAacMonoEncoder {
    core_sampling_frequency_index: u8,
    core_sampling_frequency: u32,
    core_frame_length: usize,
    bandwidth: u32,
    core_analysis: AacLcAnalysisFilterbank,
    psychoacoustic: AacLcPsychoacousticModel,
    quantizer: AacLcQuantizer,
    block_switcher: AacLcBlockSwitcher,
    reservoir: AacLcBitReservoir,
    bitrate: u32,
    vbr_quality_factor: Option<f32>,
    chaos_measure_old: f32,
    downsampler: HalfbandDownsampler,
    sbr_analysis: SbrEncoderAnalysis,
    sbr_header: LdSbrHeader,
    sbr_header_written: bool,
}

impl PureRustHeAacMonoEncoder {
    pub fn new(
        core_sampling_frequency_index: u8,
        output_sampling_frequency: u32,
        nominal_frame_bits: usize,
        reservoir_capacity_bits: usize,
        sbr_header: LdSbrHeader,
    ) -> Result<Self, AacLcEncoderError> {
        Self::new_with_frame_length(
            core_sampling_frequency_index,
            output_sampling_frequency,
            1024,
            nominal_frame_bits,
            reservoir_capacity_bits,
            sbr_header,
        )
    }

    pub fn new_with_frame_length(
        core_sampling_frequency_index: u8,
        output_sampling_frequency: u32,
        core_frame_length: usize,
        nominal_frame_bits: usize,
        reservoir_capacity_bits: usize,
        sbr_header: LdSbrHeader,
    ) -> Result<Self, AacLcEncoderError> {
        let psychoacoustic = AacLcPsychoacousticModel::new_with_frame_length(
            core_sampling_frequency_index,
            core_frame_length,
        )?;
        let quantizer = AacLcQuantizer::new_with_frame_length(
            core_sampling_frequency_index,
            core_frame_length,
        )?;
        let core_sampling_frequency = sample_rate_from_index(core_sampling_frequency_index)
            .ok_or(AacLcEncoderError::InvalidRawElementLayout)?;
        let bitrate = ((nominal_frame_bits as u64 * u64::from(core_sampling_frequency))
            / core_frame_length as u64)
            .min((1 << 23) - 1) as u32;
        Ok(Self {
            core_sampling_frequency_index,
            core_sampling_frequency,
            core_frame_length,
            bandwidth: core_sampling_frequency / 2,
            core_analysis: AacLcAnalysisFilterbank::new(core_frame_length)?,
            psychoacoustic,
            quantizer,
            block_switcher: AacLcBlockSwitcher::default(),
            reservoir: AacLcBitReservoir::new(nominal_frame_bits, reservoir_capacity_bits),
            bitrate,
            vbr_quality_factor: None,
            chaos_measure_old: 0.3,
            downsampler: HalfbandDownsampler::new(),
            sbr_analysis: SbrEncoderAnalysis::new(&sbr_header, output_sampling_frequency)?,
            sbr_header,
            sbr_header_written: false,
        })
    }

    pub fn encode_raw_data_block(&mut self, input: &[f32]) -> Result<Vec<u8>, AacLcEncoderError> {
        self.encode_raw_data_block_with_extension(input, None)
    }

    pub fn set_bandwidth(&mut self, bandwidth: u32) {
        self.bandwidth = bandwidth.min(self.core_sampling_frequency / 2);
    }

    pub fn set_afterburner(&mut self, enabled: bool) {
        self.quantizer.set_afterburner(enabled);
    }

    pub fn set_bitrate_mode(&mut self, mode: u32) {
        self.vbr_quality_factor = low_delay_vbr_quality_factor(mode);
    }

    fn encode_raw_data_block_with_extension(
        &mut self,
        input: &[f32],
        extension: Option<&[u8]>,
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        let expected = self.core_frame_length * 2;
        if input.len() != expected {
            return Err(AacLcEncoderError::InputLengthMismatch {
                expected,
                actual: input.len(),
            });
        }
        if input.iter().any(|sample| !sample.is_finite()) {
            return Err(AacLcEncoderError::NonFiniteInput);
        }
        let sbr = self.sbr_analysis.analyze(input)?;
        let fill = sbr.write_mono_fill_element_with_extension(
            &self.sbr_header,
            self.sbr_analysis.frequency_tables(),
            !self.sbr_header_written,
            extension,
        )?;
        let core = self.downsampler.process(input);
        let sequence = self.block_switcher.update(transient_ratio(&core));
        let mut analysis = self.core_analysis.analyze_with_sequence(&core, sequence)?;
        apply_spectral_bandwidth(
            &mut analysis.spectrum,
            self.core_sampling_frequency,
            self.bandwidth,
        );
        apply_short_spectral_bandwidth(
            &mut analysis.short_spectra,
            self.core_sampling_frequency,
            self.bandwidth,
        );
        let target = if self.vbr_quality_factor.is_some() {
            self.reservoir.nominal_frame_bits() + self.reservoir.capacity_bits()
        } else {
            self.reservoir.available_frame_bits()
        };
        let raw = if sequence == WindowSequence::EightShort {
            let spectra = analysis
                .short_spectra
                .as_ref()
                .expect("short analysis requested");
            let mut psycho = self.psychoacoustic.analyze_short(spectra)?;
            if let Some(quality) = self.vbr_quality_factor {
                let info = aac_sfb_info_for_frame(
                    self.core_sampling_frequency_index,
                    WindowSequence::EightShort,
                    self.core_frame_length,
                )?;
                let mut analyses = psycho.iter_mut().collect::<Vec<_>>();
                apply_low_delay_vbr_thresholds(
                    &mut analyses,
                    info.offsets,
                    self.bitrate,
                    self.core_sampling_frequency,
                    info.granule_length,
                    self.bandwidth,
                    quality,
                    &mut self.chaos_measure_old,
                );
            }
            self.quantizer
                .quantize_short(
                    spectra,
                    &psycho,
                    &analysis.short_window_group_lengths,
                    target,
                )?
                .write_sce_raw_data_block_with_sbr_fill(0, &fill)?
        } else {
            let mut psycho = self.psychoacoustic.analyze(&analysis.spectrum)?;
            if let Some(quality) = self.vbr_quality_factor {
                let info = aac_sfb_info_for_frame(
                    self.core_sampling_frequency_index,
                    WindowSequence::OnlyLong,
                    self.core_frame_length,
                )?;
                apply_low_delay_vbr_thresholds(
                    &mut [&mut psycho],
                    info.offsets,
                    self.bitrate,
                    self.core_sampling_frequency,
                    self.core_frame_length,
                    self.bandwidth,
                    quality,
                    &mut self.chaos_measure_old,
                );
            }
            let quantized = self
                .quantizer
                .quantize_long(&analysis.spectrum, &psycho, target)?;
            quantized.write_sce_raw_data_block_with_sequence_and_fill(0, sequence, Some(&fill))?
        };
        if self.vbr_quality_factor.is_none() {
            self.reservoir.commit_frame(raw.len() * 8)?;
        }
        self.sbr_header_written = true;
        Ok(raw)
    }
}

#[derive(Debug, Clone)]
pub struct PureRustHeAacPsEncoder {
    mono: PureRustHeAacMonoEncoder,
    left_analysis: SbrEncoderAnalysis,
    right_analysis: SbrEncoderAnalysis,
    ps_header_written: bool,
    core_frame_length: usize,
}

#[derive(Debug, Clone)]
pub struct PureRustHeAacStereoEncoder {
    core: PureRustAacLcStereoEncoder,
    left_downsampler: HalfbandDownsampler,
    right_downsampler: HalfbandDownsampler,
    left_sbr_analysis: SbrEncoderAnalysis,
    right_sbr_analysis: SbrEncoderAnalysis,
    sbr_header: LdSbrHeader,
    sbr_header_written: bool,
    core_frame_length: usize,
}

#[derive(Debug, Clone)]
pub struct PureRustHeAacMultichannelEncoder {
    core: PureRustAacLcMultichannelEncoder,
    downsamplers: Vec<HalfbandDownsampler>,
    sbr_analyses: Vec<SbrEncoderAnalysis>,
    sbr_header: LdSbrHeader,
    sbr_header_written: bool,
    channels: usize,
    channel_mode: u32,
    core_frame_length: usize,
}

impl PureRustHeAacMultichannelEncoder {
    pub fn new(
        core_sampling_frequency_index: u8,
        output_sampling_frequency: u32,
        channels: usize,
        nominal_frame_bits: usize,
        reservoir_capacity_bits: usize,
        sbr_header: LdSbrHeader,
    ) -> Result<Self, AacLcEncoderError> {
        Self::new_with_frame_length(
            core_sampling_frequency_index,
            output_sampling_frequency,
            channels,
            1024,
            nominal_frame_bits,
            reservoir_capacity_bits,
            sbr_header,
        )
    }

    pub fn new_with_frame_length(
        core_sampling_frequency_index: u8,
        output_sampling_frequency: u32,
        channels: usize,
        core_frame_length: usize,
        nominal_frame_bits: usize,
        reservoir_capacity_bits: usize,
        sbr_header: LdSbrHeader,
    ) -> Result<Self, AacLcEncoderError> {
        Self::new_with_channel_mode(
            core_sampling_frequency_index,
            output_sampling_frequency,
            channels,
            channels as u32,
            core_frame_length,
            nominal_frame_bits,
            reservoir_capacity_bits,
            sbr_header,
        )
    }

    pub fn new_with_channel_mode(
        core_sampling_frequency_index: u8,
        output_sampling_frequency: u32,
        channels: usize,
        channel_mode: u32,
        core_frame_length: usize,
        nominal_frame_bits: usize,
        reservoir_capacity_bits: usize,
        sbr_header: LdSbrHeader,
    ) -> Result<Self, AacLcEncoderError> {
        multichannel_layout_for_mode(channel_mode)?;
        let sbr_analyses = (0..channels)
            .map(|_| SbrEncoderAnalysis::new(&sbr_header, output_sampling_frequency))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            core: PureRustAacLcMultichannelEncoder::new_with_channel_mode(
                core_sampling_frequency_index,
                channels,
                channel_mode,
                core_frame_length,
                nominal_frame_bits,
                reservoir_capacity_bits,
            )?,
            downsamplers: (0..channels).map(|_| HalfbandDownsampler::new()).collect(),
            sbr_analyses,
            sbr_header,
            sbr_header_written: false,
            channels,
            channel_mode,
            core_frame_length,
        })
    }

    pub fn set_bandwidth(&mut self, bandwidth: u32) {
        self.core.set_bandwidth(bandwidth);
    }

    pub fn set_afterburner(&mut self, enabled: bool) {
        self.core.set_afterburner(enabled);
    }

    pub fn set_bitrate_mode(&mut self, mode: u32) {
        self.core.set_bitrate_mode(mode);
    }

    pub fn encode_raw_data_block(
        &mut self,
        pcm: &[Vec<f32>],
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        let frame_length = self.core_frame_length * 2;
        if pcm.len() != self.channels || pcm.iter().any(|channel| channel.len() != frame_length) {
            return Err(AacLcEncoderError::InputLengthMismatch {
                expected: self.channels * frame_length,
                actual: pcm.iter().map(Vec::len).sum(),
            });
        }
        if pcm.iter().flatten().any(|sample| !sample.is_finite()) {
            return Err(AacLcEncoderError::NonFiniteInput);
        }
        let mut sbr_frames = Vec::with_capacity(self.channels);
        for (analysis, channel) in self.sbr_analyses.iter_mut().zip(pcm) {
            sbr_frames.push(analysis.analyze(channel)?);
        }
        let mut fills = Vec::new();
        for (element, first, second) in multichannel_layout_for_mode(self.channel_mode)? {
            if element == ElementId::Lfe {
                continue;
            }
            let fill = if let Some(second) = second {
                SbrEncoderAnalysisFrame::write_stereo_fill_element(
                    &sbr_frames[first],
                    &sbr_frames[second],
                    &self.sbr_header,
                    self.sbr_analyses[first].frequency_tables(),
                    !self.sbr_header_written,
                )?
            } else {
                sbr_frames[first].write_mono_fill_element(
                    &self.sbr_header,
                    self.sbr_analyses[first].frequency_tables(),
                    !self.sbr_header_written,
                )?
            };
            fills.push(fill);
        }
        let core = self
            .downsamplers
            .iter_mut()
            .zip(pcm)
            .map(|(downsampler, channel)| downsampler.process(channel))
            .collect::<Vec<_>>();
        let raw = self
            .core
            .encode_raw_data_block_with_sbr_fills(&core, &fills)?;
        self.sbr_header_written = true;
        Ok(raw)
    }
}

impl PureRustHeAacStereoEncoder {
    pub fn new(
        core_sampling_frequency_index: u8,
        output_sampling_frequency: u32,
        nominal_frame_bits: usize,
        reservoir_capacity_bits: usize,
        sbr_header: LdSbrHeader,
    ) -> Result<Self, AacLcEncoderError> {
        Self::new_with_frame_length(
            core_sampling_frequency_index,
            output_sampling_frequency,
            1024,
            nominal_frame_bits,
            reservoir_capacity_bits,
            sbr_header,
        )
    }

    pub fn new_with_frame_length(
        core_sampling_frequency_index: u8,
        output_sampling_frequency: u32,
        core_frame_length: usize,
        nominal_frame_bits: usize,
        reservoir_capacity_bits: usize,
        sbr_header: LdSbrHeader,
    ) -> Result<Self, AacLcEncoderError> {
        Ok(Self {
            core: PureRustAacLcStereoEncoder::new_with_frame_length(
                core_sampling_frequency_index,
                core_frame_length,
                nominal_frame_bits,
                reservoir_capacity_bits,
            )?,
            left_downsampler: HalfbandDownsampler::new(),
            right_downsampler: HalfbandDownsampler::new(),
            left_sbr_analysis: SbrEncoderAnalysis::new(&sbr_header, output_sampling_frequency)?,
            right_sbr_analysis: SbrEncoderAnalysis::new(&sbr_header, output_sampling_frequency)?,
            sbr_header,
            sbr_header_written: false,
            core_frame_length,
        })
    }

    pub fn set_bandwidth(&mut self, bandwidth: u32) {
        self.core.set_bandwidth(bandwidth);
    }

    pub fn set_afterburner(&mut self, enabled: bool) {
        self.core.set_afterburner(enabled);
    }

    pub fn set_bitrate_mode(&mut self, mode: u32) {
        self.core.set_bitrate_mode(mode);
    }

    pub fn encode_raw_data_block(
        &mut self,
        left: &[f32],
        right: &[f32],
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        let expected = self.core_frame_length * 2;
        if left.len() != expected || right.len() != expected {
            return Err(AacLcEncoderError::InputLengthMismatch {
                expected,
                actual: left.len().min(right.len()),
            });
        }
        if left.iter().chain(right).any(|sample| !sample.is_finite()) {
            return Err(AacLcEncoderError::NonFiniteInput);
        }
        let left_sbr = self.left_sbr_analysis.analyze(left)?;
        let right_sbr = self.right_sbr_analysis.analyze(right)?;
        let fill = SbrEncoderAnalysisFrame::write_stereo_fill_element(
            &left_sbr,
            &right_sbr,
            &self.sbr_header,
            self.left_sbr_analysis.frequency_tables(),
            !self.sbr_header_written,
        )?;
        let left_core = self.left_downsampler.process(left);
        let right_core = self.right_downsampler.process(right);
        let raw = self
            .core
            .encode_raw_data_block_with_sbr_fill(&left_core, &right_core, &fill)?;
        self.sbr_header_written = true;
        Ok(raw)
    }
}

impl PureRustHeAacPsEncoder {
    pub fn new(
        core_sampling_frequency_index: u8,
        output_sampling_frequency: u32,
        nominal_frame_bits: usize,
        reservoir_capacity_bits: usize,
        sbr_header: LdSbrHeader,
    ) -> Result<Self, AacLcEncoderError> {
        Self::new_with_frame_length(
            core_sampling_frequency_index,
            output_sampling_frequency,
            1024,
            nominal_frame_bits,
            reservoir_capacity_bits,
            sbr_header,
        )
    }

    pub fn new_with_frame_length(
        core_sampling_frequency_index: u8,
        output_sampling_frequency: u32,
        core_frame_length: usize,
        nominal_frame_bits: usize,
        reservoir_capacity_bits: usize,
        sbr_header: LdSbrHeader,
    ) -> Result<Self, AacLcEncoderError> {
        Ok(Self {
            mono: PureRustHeAacMonoEncoder::new_with_frame_length(
                core_sampling_frequency_index,
                output_sampling_frequency,
                core_frame_length,
                nominal_frame_bits,
                reservoir_capacity_bits,
                sbr_header.clone(),
            )?,
            left_analysis: SbrEncoderAnalysis::new(&sbr_header, output_sampling_frequency)?,
            right_analysis: SbrEncoderAnalysis::new(&sbr_header, output_sampling_frequency)?,
            ps_header_written: false,
            core_frame_length,
        })
    }

    pub fn set_bandwidth(&mut self, bandwidth: u32) {
        self.mono.set_bandwidth(bandwidth);
    }

    pub fn set_afterburner(&mut self, enabled: bool) {
        self.mono.set_afterburner(enabled);
    }

    pub fn set_bitrate_mode(&mut self, mode: u32) {
        self.mono.set_bitrate_mode(mode);
    }

    pub fn encode_raw_data_block(
        &mut self,
        left: &[f32],
        right: &[f32],
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        let expected = self.core_frame_length * 2;
        if left.len() != expected || right.len() != expected {
            return Err(AacLcEncoderError::InputLengthMismatch {
                expected,
                actual: left.len().min(right.len()),
            });
        }
        let left_qmf = self.left_analysis.analyze(left)?;
        let right_qmf = self.right_analysis.analyze(right)?;
        let ps = analyze_ps_qmf(&left_qmf.slots, &right_qmf.slots)?
            .write_sbr_extension(!self.ps_header_written)?;
        let downmix = left
            .iter()
            .zip(right)
            .map(|(&left, &right)| 0.5 * (left + right))
            .collect::<Vec<_>>();
        let raw = self
            .mono
            .encode_raw_data_block_with_extension(&downmix, Some(&ps))?;
        self.ps_header_written = true;
        Ok(raw)
    }
}

#[derive(Debug, Clone)]
pub struct PureRustDrmAacMonoEncoder {
    sampling_frequency_index: u8,
    analysis: AacLcAnalysisFilterbank,
}

#[derive(Debug, Clone)]
pub struct PureRustDrmAacStereoEncoder {
    sampling_frequency_index: u8,
    left_analysis: AacLcAnalysisFilterbank,
    right_analysis: AacLcAnalysisFilterbank,
}

fn retry_drm_hcr(
    encode: &mut dyn FnMut(f32) -> Result<Vec<u8>, AacLcEncoderError>,
) -> Result<Vec<u8>, AacLcEncoderError> {
    let mut relaxation = 1.0f32;
    loop {
        match encode(relaxation) {
            Ok(output) => return Ok(output),
            Err(AacLcEncoderError::Hcr(HcrError::ReorderedSpectralLengthOutOfRange { .. }))
                if relaxation < 4096.0 =>
            {
                relaxation *= 2.0;
            }
            Err(error) => return Err(error),
        }
    }
}

impl PureRustDrmAacStereoEncoder {
    pub fn new(sampling_frequency_index: u8) -> Result<Self, AacLcEncoderError> {
        let probe = drm_ics(sampling_frequency_index, 0)?;
        aac_band_offsets_for_ics(sampling_frequency_index, &probe, 960)?;
        Ok(Self {
            sampling_frequency_index,
            left_analysis: AacLcAnalysisFilterbank::new(960)?,
            right_analysis: AacLcAnalysisFilterbank::new(960)?,
        })
    }

    pub fn encode_packet(
        &mut self,
        left: &[f32],
        right: &[f32],
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        let left = self.left_analysis.analyze(left)?;
        let right = self.right_analysis.analyze(right)?;
        let probe = drm_ics(self.sampling_frequency_index, 0)?;
        let info = aac_band_offsets_for_ics(self.sampling_frequency_index, &probe, 960)?;
        let left_psycho = analyze_psychoacoustic_bands(&left.spectrum, info.offsets);
        let right_psycho = analyze_psychoacoustic_bands(&right.spectrum, info.offsets);
        retry_drm_hcr(&mut |relaxation| {
            let left =
                quantize_drm_spectrum(&left.spectrum, &left_psycho, info.offsets, relaxation);
            let right =
                quantize_drm_spectrum(&right.spectrum, &right_psycho, info.offsets, relaxation);
            write_drm_stereo_packet(&left, &right, self.sampling_frequency_index, info.offsets)
        })
    }
}

fn quantize_drm_spectrum(
    spectrum: &[f32],
    psycho: &PsychoacousticAnalysis,
    offsets: &[usize],
    relaxation: f32,
) -> QuantizedAacLcFrame {
    let mut bands = offsets
        .windows(2)
        .zip(&psycho.bands)
        .map(|(border, psycho)| {
            quantize_band(
                &spectrum[border[0]..border[1]],
                psycho.masking_threshold * relaxation,
                false,
            )
        })
        .collect::<Vec<_>>();
    enforce_scalefactor_delta_range(&mut bands);
    let (sections, spectral_bits, section_bits) = select_huffman_sections(&mut bands, false);
    let maximum_scale = bands
        .iter()
        .filter(|band| band.coefficients.iter().any(|&value| value != 0))
        .map(|band| band.scalefactor)
        .max()
        .unwrap_or(-100);
    let mut previous_scale = maximum_scale;
    for band in &mut bands {
        if band.codebook != 0 {
            if band.coefficients.iter().all(|&value| value == 0) {
                band.scalefactor = previous_scale;
            } else {
                previous_scale = band.scalefactor;
            }
        }
    }
    QuantizedAacLcFrame {
        global_gain: (i32::from(maximum_scale) + 100).clamp(0, 255) as u8,
        bands,
        estimated_spectral_bits: spectral_bits,
        estimated_section_bits: section_bits,
        sections,
        masking_relaxation: relaxation,
    }
}

fn enforce_scalefactor_delta_range(bands: &mut [QuantizedSfb]) {
    let Some(mut factor) = bands
        .iter()
        .filter(|band| band.coefficients.iter().any(|&value| value != 0))
        .map(|band| band.scalefactor)
        .max()
    else {
        return;
    };
    enforce_scalefactor_delta_range_from(bands, &mut factor);
}

fn enforce_scalefactor_delta_range_from(bands: &mut [QuantizedSfb], factor: &mut i16) {
    for band in bands {
        if band.coefficients.iter().all(|&value| value == 0) {
            continue;
        }
        let delta = band.scalefactor - *factor;
        if !(-60..=60).contains(&delta) {
            // FDK's assimilate-multiple-holes pass removes isolated bands
            // whose scale cannot be represented relative to the preceding
            // transmitted factor. Keeping them would create an invalid
            // scalefactor Huffman symbol.
            band.coefficients.fill(0);
            band.codebook = 0;
            band.estimated_bits = 0;
            band.codebook_bit_costs = zero_codebook_costs();
        } else {
            *factor = band.scalefactor;
        }
    }
}

fn normalize_section_zero_scalefactors(bands: &mut [QuantizedSfb], factor: &mut i16) {
    for band in bands {
        if band.codebook == 0 {
            continue;
        }
        if band.coefficients.iter().all(|&value| value == 0) {
            // Section merging may assign a spectral codebook to a zero band.
            // Such a band still transmits a scalefactor, so inherit the last
            // representable value just as FDK's hole assimilation does.
            band.scalefactor = *factor;
        } else {
            *factor = band.scalefactor;
        }
    }
}

/// Find the smallest masking-threshold multiplier whose complete access unit
/// fits the granted bits.  FDK's threshold adjustment iterates toward the
/// granted PE; using only powers of two here needlessly discarded as much as
/// half of the available spectral budget.
fn fit_low_delay_access_unit<T, F>(
    available_bits: usize,
    mut candidate: F,
) -> Result<(T, Vec<u8>), AacLcEncoderError>
where
    F: FnMut(f32) -> Result<(T, Vec<u8>), AacLcEncoderError>,
{
    let (initial_frame, initial_raw) = candidate(1.0)?;
    if initial_raw.len() * 8 <= available_bits {
        return Ok((initial_frame, initial_raw));
    }

    let mut lower = 1.0f32;
    let mut upper = 2.0f32;
    let (mut best_frame, mut best_raw) = loop {
        let output = candidate(upper)?;
        if output.1.len() * 8 <= available_bits || upper >= 4096.0 {
            break output;
        }
        lower = upper;
        upper *= 2.0;
    };

    if best_raw.len() * 8 <= available_bits {
        // Ten corrections resolve a power-of-two bracket to better than
        // 0.1%, while retaining deterministic quantizer decisions.
        for _ in 0..10 {
            let middle = (lower + upper) * 0.5;
            let (frame, raw) = candidate(middle)?;
            if raw.len() * 8 <= available_bits {
                upper = middle;
                best_frame = frame;
                best_raw = raw;
            } else {
                lower = middle;
            }
        }
    }
    Ok((best_frame, best_raw))
}

/// Long-window branch of FDK's `FDKaacEnc_bitresCalcBitFac`.  The returned
/// budget is still capped by the physically available reservoir bits.
fn low_delay_granted_dynamic_bits(
    reservoir: &AacLcBitReservoir,
    perceptual_entropy: f32,
    bits_to_pe_factor: f32,
    static_bits: usize,
    state: &mut LowDelayPeCorrection,
) -> usize {
    let nominal_dynamic = reservoir.nominal_frame_bits.saturating_sub(static_bits);
    let nominal = nominal_dynamic as f32;
    if nominal <= 0.0 {
        return 0;
    }
    let fill = if reservoir.capacity_bits == 0 {
        0.0
    } else {
        reservoir.fullness_bits as f32 / reservoir.capacity_bits as f32
    };
    let mean_pe = nominal * bits_to_pe_factor;
    let mut bit_factor = state.reservoir_bit_factor(perceptual_entropy, mean_pe, fill);
    bit_factor = bit_factor.min(0.7 + reservoir.fullness_bits as f32 / nominal);
    ((nominal * bit_factor.max(0.0)).round() as usize)
        .min(reservoir.available_frame_bits().saturating_sub(static_bits))
}

impl PureRustDrmAacMonoEncoder {
    pub fn new(sampling_frequency_index: u8) -> Result<Self, AacLcEncoderError> {
        let probe = drm_ics(sampling_frequency_index, 0)?;
        aac_band_offsets_for_ics(sampling_frequency_index, &probe, 960)?;
        Ok(Self {
            sampling_frequency_index,
            analysis: AacLcAnalysisFilterbank::new(960)?,
        })
    }

    /// Encode one CRC-protected DRM ER AAC mono packet using HCR priority
    /// segments. The packet begins with the DRM CRC-8 byte.
    pub fn encode_packet(&mut self, input: &[f32]) -> Result<Vec<u8>, AacLcEncoderError> {
        let analysis = self.analysis.analyze(input)?;
        let probe = drm_ics(self.sampling_frequency_index, 0)?;
        let info = aac_band_offsets_for_ics(self.sampling_frequency_index, &probe, 960)?;
        let psycho = analyze_psychoacoustic_bands(&analysis.spectrum, info.offsets);
        retry_drm_hcr(&mut |relaxation| {
            let mut bands = info
                .offsets
                .windows(2)
                .zip(&psycho.bands)
                .map(|(border, psycho)| {
                    quantize_band(
                        &analysis.spectrum[border[0]..border[1]],
                        psycho.masking_threshold * relaxation,
                        false,
                    )
                })
                .collect::<Vec<_>>();
            let (sections, spectral_bits, section_bits) =
                select_huffman_sections(&mut bands, false);
            let maximum_scale = bands
                .iter()
                .filter(|band| band.coefficients.iter().any(|&value| value != 0))
                .map(|band| band.scalefactor)
                .max()
                .unwrap_or(-100);
            let frame = QuantizedAacLcFrame {
                global_gain: (i32::from(maximum_scale) + 100).clamp(0, 255) as u8,
                bands,
                estimated_spectral_bits: spectral_bits,
                estimated_section_bits: section_bits,
                sections,
                masking_relaxation: relaxation,
            };
            write_drm_mono_packet(&frame, self.sampling_frequency_index, info.offsets)
        })
    }
}

fn drm_ics(sampling_frequency_index: u8, max_sfb: u8) -> Result<IcsInfo, AacLcEncoderError> {
    let mut ics = IcsInfo {
        window_sequence: WindowSequence::OnlyLong,
        window_shape: WindowShape::Sine,
        max_sfb,
        total_sfb: 0,
        predictor_data_present: false,
        scale_factor_grouping: 0,
        window_group_lengths: vec![1],
        bits_read: 0,
    };
    let info = aac_band_offsets_for_ics(sampling_frequency_index, &ics, 960)?;
    ics.total_sfb = info.num_bands;
    if max_sfb == 0 {
        ics.max_sfb = info.num_bands;
    }
    Ok(ics)
}

fn er_long_ics(
    sampling_frequency_index: u8,
    frame_length: usize,
) -> Result<IcsInfo, AacLcEncoderError> {
    let mut ics = IcsInfo {
        window_sequence: WindowSequence::OnlyLong,
        window_shape: WindowShape::Sine,
        max_sfb: 0,
        total_sfb: 0,
        predictor_data_present: false,
        scale_factor_grouping: 0,
        window_group_lengths: vec![1],
        bits_read: 0,
    };
    let info = aac_band_offsets_for_ics(sampling_frequency_index, &ics, frame_length)?;
    ics.max_sfb = info.num_bands;
    ics.total_sfb = info.num_bands;
    Ok(ics)
}

fn write_er_aac_ld_mono_access_unit(
    frame: &QuantizedAacLcFrame,
    element_instance_tag: u8,
    window_shape: WindowShape,
    mps: Option<(&[u8], usize)>,
    dynamic_range: Option<(&[u8], usize)>,
    ancillary_elements: &[&[u8]],
    fill_bytes: usize,
) -> Result<Vec<u8>, AacLcEncoderError> {
    if frame.bands.len() > 63 || element_instance_tag > 15 {
        return Err(AacLcEncoderError::InvalidRawElementLayout);
    }
    let mut writer = BitWriter::new();
    writer.write(element_instance_tag as u32, 4);
    writer.write(frame.global_gain as u32, 8);
    write_long_ics_with_shape(
        &mut writer,
        frame.bands.len(),
        WindowSequence::OnlyLong,
        window_shape,
    );
    write_sections(&mut writer, &frame.sections, false);
    let mut factor = i16::from(frame.global_gain) - 100;
    write_scalefactors(&mut writer, &frame.bands, &mut factor)?;
    writer.write_bool(false); // pulse absent
    writer.write_bool(false); // TNS absent
    writer.write_bool(false); // gain control absent
    write_spectral_bands(&mut writer, &frame.bands)?;
    if let Some((payload, bits)) = mps {
        writer.write(0x09, 4); // EXT_LDSAC_DATA
        writer.write(0x03, 4); // explicit SSC in payload
        write_payload_bits(&mut writer, payload, bits);
    }
    write_er_extensions(&mut writer, dynamic_range, ancillary_elements, fill_bytes);
    writer.byte_align();
    Ok(writer.finish())
}

fn write_er_aac_ld_stereo_access_unit(
    left: &QuantizedAacLcFrame,
    right: &QuantizedAacLcFrame,
    element_instance_tag: u8,
    window_shape: WindowShape,
    dynamic_range: Option<(&[u8], usize)>,
    ancillary_elements: &[&[u8]],
    fill_bytes: usize,
) -> Result<Vec<u8>, AacLcEncoderError> {
    if left.bands.len() != right.bands.len() || left.bands.len() > 63 || element_instance_tag > 15 {
        return Err(AacLcEncoderError::InvalidRawElementLayout);
    }
    let mut writer = BitWriter::new();
    writer.write(element_instance_tag as u32, 4);
    writer.write_bool(true); // common_window
    write_long_ics_with_shape(
        &mut writer,
        left.bands.len(),
        WindowSequence::OnlyLong,
        window_shape,
    );
    writer.write(0, 2); // ms_mask_present = none
    write_long_channel_stream(&mut writer, left, None)?;
    write_long_channel_stream(&mut writer, right, None)?;
    write_er_extensions(&mut writer, dynamic_range, ancillary_elements, fill_bytes);
    writer.byte_align();
    Ok(writer.finish())
}

fn write_er_aac_ld_multichannel_access_unit(
    frames: &[QuantizedAacLcFrame],
    channel_mode: u32,
    window_shape: WindowShape,
    dynamic_range: Option<(&[u8], usize)>,
    ancillary_elements: &[&[u8]],
    fill_bytes: usize,
) -> Result<Vec<u8>, AacLcEncoderError> {
    let layout = multichannel_layout_for_mode(channel_mode)?;
    let layout_channels = layout
        .iter()
        .flat_map(|(_, first, second)| [Some(*first), *second])
        .flatten()
        .max()
        .map_or(0, |last| last + 1);
    if layout_channels != frames.len() || frames.iter().any(|frame| frame.bands.len() > 63) {
        return Err(AacLcEncoderError::InvalidRawElementLayout);
    }
    let mut writer = BitWriter::new();
    let mut sce_tag = 0u32;
    let mut cpe_tag = 0u32;
    let mut lfe_tag = 0u32;
    for (element, first, second) in layout {
        let tag = match element {
            ElementId::SingleChannel => {
                let tag = sce_tag;
                sce_tag += 1;
                tag
            }
            ElementId::ChannelPair => {
                let tag = cpe_tag;
                cpe_tag += 1;
                tag
            }
            ElementId::Lfe => {
                let tag = lfe_tag;
                lfe_tag += 1;
                tag
            }
            _ => return Err(AacLcEncoderError::InvalidRawElementLayout),
        };
        writer.write(tag, 4);
        if let Some(second) = second {
            if frames[first].bands.len() != frames[second].bands.len() {
                return Err(AacLcEncoderError::InvalidRawElementLayout);
            }
            writer.write_bool(true); // common_window
            write_long_ics_with_shape(
                &mut writer,
                frames[first].bands.len(),
                WindowSequence::OnlyLong,
                window_shape,
            );
            writer.write(0, 2); // ms_mask_present = none
            write_long_channel_stream(&mut writer, &frames[first], None)?;
            write_long_channel_stream(&mut writer, &frames[second], None)?;
        } else {
            write_long_channel_stream_with_shape(&mut writer, &frames[first], window_shape)?;
        }
    }
    write_er_extensions(&mut writer, dynamic_range, ancillary_elements, fill_bytes);
    writer.byte_align();
    Ok(writer.finish())
}

fn write_long_channel_stream_with_shape(
    writer: &mut BitWriter,
    frame: &QuantizedAacLcFrame,
    window_shape: WindowShape,
) -> Result<(), AacLcEncoderError> {
    writer.write(frame.global_gain as u32, 8);
    write_long_ics_with_shape(
        writer,
        frame.bands.len(),
        WindowSequence::OnlyLong,
        window_shape,
    );
    write_sections(writer, &frame.sections, false);
    let mut factor = i16::from(frame.global_gain) - 100;
    write_scalefactors(writer, &frame.bands, &mut factor)?;
    writer.write_bool(false); // pulse absent
    writer.write_bool(false); // TNS absent
    writer.write_bool(false); // gain control absent
    write_spectral_bands(writer, &frame.bands)
}

fn write_eld_channel_stream(
    writer: &mut BitWriter,
    frame: &QuantizedAacLcFrame,
    include_max_sfb: bool,
) -> Result<(), AacLcEncoderError> {
    writer.write(frame.global_gain as u32, 8);
    if include_max_sfb {
        writer.write(frame.bands.len() as u32, 6);
    }
    write_sections(writer, &frame.sections, false);
    let mut factor = i16::from(frame.global_gain) - 100;
    write_scalefactors(writer, &frame.bands, &mut factor)?;
    writer.write_bool(false); // TNS absent
    write_spectral_bands(writer, &frame.bands)
}

#[derive(Clone, Copy)]
struct EldMonoSbrPayload<'a> {
    frame: &'a SbrEncoderAnalysisFrame,
    header: &'a LdSbrHeader,
    tables: &'a LdSbrFrequencyTables,
    header_present: bool,
    crc_present: bool,
}

#[derive(Clone, Copy)]
struct EldStereoSbrPayload<'a> {
    left: &'a SbrEncoderAnalysisFrame,
    right: &'a SbrEncoderAnalysisFrame,
    header: &'a LdSbrHeader,
    tables: &'a LdSbrFrequencyTables,
    header_present: bool,
    crc_present: bool,
}

#[derive(Clone, Copy)]
struct EldMultichannelSbrPayloads<'a> {
    frames: &'a [SbrEncoderAnalysisFrame],
    headers: &'a [LdSbrHeader],
    tables: &'a [LdSbrFrequencyTables],
    header_present: bool,
    crc_present: bool,
}

fn write_er_aac_eld_mono_access_unit(
    frame: &QuantizedAacLcFrame,
    sbr: Option<EldMonoSbrPayload<'_>>,
    mps: Option<(&[u8], usize)>,
    dynamic_range: Option<(&[u8], usize)>,
    ancillary_elements: &[&[u8]],
    fill_bytes: usize,
) -> Result<Vec<u8>, AacLcEncoderError> {
    if frame.bands.len() > 63 {
        return Err(AacLcEncoderError::InvalidRawElementLayout);
    }
    let mut writer = BitWriter::new();
    write_eld_channel_stream(&mut writer, frame, true)?;
    if let Some(sbr) = sbr {
        sbr.frame.write_mono_low_delay_payload_with_crc(
            &mut writer,
            sbr.header,
            sbr.tables,
            sbr.header_present,
            sbr.crc_present,
        )?;
    }
    if let Some((payload, bits)) = mps {
        writer.write(0x09, 4); // EXT_LDSAC_DATA
        writer.write(0x03, 4); // explicit SSC lives in ELDEXT_LDSAC
        write_payload_bits(&mut writer, payload, bits);
    }
    write_er_extensions(&mut writer, dynamic_range, ancillary_elements, fill_bytes);
    writer.byte_align();
    Ok(writer.finish())
}

fn write_er_aac_eld_stereo_access_unit(
    left: &QuantizedAacLcFrame,
    right: &QuantizedAacLcFrame,
    sbr: Option<EldStereoSbrPayload<'_>>,
    dynamic_range: Option<(&[u8], usize)>,
    ancillary_elements: &[&[u8]],
    fill_bytes: usize,
) -> Result<Vec<u8>, AacLcEncoderError> {
    if left.bands.len() != right.bands.len() || left.bands.len() > 63 {
        return Err(AacLcEncoderError::InvalidRawElementLayout);
    }
    let mut writer = BitWriter::new();
    writer.write(left.bands.len() as u32, 6);
    writer.write(0, 2); // ms_mask_present = none
    write_eld_channel_stream(&mut writer, left, false)?;
    write_eld_channel_stream(&mut writer, right, false)?;
    if let Some(sbr) = sbr {
        SbrEncoderAnalysisFrame::write_stereo_low_delay_payload(
            sbr.left,
            sbr.right,
            &mut writer,
            sbr.header,
            sbr.tables,
            sbr.header_present,
            sbr.crc_present,
        )?;
    }
    write_er_extensions(&mut writer, dynamic_range, ancillary_elements, fill_bytes);
    writer.byte_align();
    Ok(writer.finish())
}

fn write_er_aac_eld_multichannel_access_unit(
    frames: &[QuantizedAacLcFrame],
    channel_mode: u32,
    sbr: Option<EldMultichannelSbrPayloads<'_>>,
    fill_bytes: usize,
) -> Result<Vec<u8>, AacLcEncoderError> {
    let layout = multichannel_layout_for_mode(channel_mode)?;
    let layout_channels = layout
        .iter()
        .flat_map(|(_, first, second)| [Some(*first), *second])
        .flatten()
        .max()
        .map_or(0, |last| last + 1);
    if layout_channels != frames.len() || frames.iter().any(|frame| frame.bands.len() > 63) {
        return Err(AacLcEncoderError::InvalidRawElementLayout);
    }
    let mut writer = BitWriter::new();
    for (_, first, second) in layout.iter().copied() {
        if let Some(second) = second {
            if frames[first].bands.len() != frames[second].bands.len() {
                return Err(AacLcEncoderError::InvalidRawElementLayout);
            }
            writer.write(frames[first].bands.len() as u32, 6);
            writer.write(0, 2); // ms_mask_present = none
            write_eld_channel_stream(&mut writer, &frames[first], false)?;
            write_eld_channel_stream(&mut writer, &frames[second], false)?;
        } else {
            write_eld_channel_stream(&mut writer, &frames[first], true)?;
        }
    }
    if let Some(sbr) = sbr {
        let mut header_index = 0;
        for (id, first, second) in layout {
            if id == ElementId::Lfe {
                continue;
            }
            let first = multichannel_sbr_channel_index(channel_mode, first)?;
            let header = &sbr.headers[header_index];
            let tables = &sbr.tables[first];
            if let Some(second) = second
                .map(|channel| multichannel_sbr_channel_index(channel_mode, channel))
                .transpose()?
            {
                SbrEncoderAnalysisFrame::write_stereo_low_delay_payload(
                    &sbr.frames[first],
                    &sbr.frames[second],
                    &mut writer,
                    header,
                    tables,
                    sbr.header_present,
                    sbr.crc_present,
                )?;
            } else {
                sbr.frames[first].write_mono_low_delay_payload_with_crc(
                    &mut writer,
                    header,
                    tables,
                    sbr.header_present,
                    sbr.crc_present,
                )?;
            }
            header_index += 1;
        }
    }
    write_er_extensions(&mut writer, None, &[], fill_bytes);
    writer.byte_align();
    Ok(writer.finish())
}

fn write_er_extensions(
    writer: &mut BitWriter,
    dynamic_range: Option<(&[u8], usize)>,
    ancillary_elements: &[&[u8]],
    fill_bytes: usize,
) {
    if let Some((payload, bits)) = dynamic_range {
        writer.write(0x0b, 4); // EXT_DYNAMIC_RANGE
        write_payload_bits(writer, payload, bits);
    }
    for ancillary in ancillary_elements {
        write_er_ancillary_extension(writer, ancillary);
    }
    if fill_bytes != 0 {
        writer.write(0x01, 4); // EXT_FILL_DATA
        writer.write(0, 4); // required zero fill nibble
        for _ in 1..fill_bytes {
            writer.write(0xa5, 8);
        }
    }
}

fn write_payload_bits(writer: &mut BitWriter, payload: &[u8], bits: usize) {
    debug_assert!(bits <= payload.len() * 8);
    for bit in 0..bits {
        writer.write(u32::from((payload[bit / 8] >> (7 - bit % 8)) & 1), 1);
    }
}

fn write_er_ancillary_extension(writer: &mut BitWriter, ancillary: &[u8]) {
    if ancillary.is_empty() {
        return;
    }
    writer.write(0x02, 4); // EXT_DATA_ELEMENT
    writer.write(0, 4); // data_element_version = ANC_DATA
    let mut remaining = ancillary.len();
    while remaining >= 255 {
        writer.write(255, 8);
        remaining -= 255;
    }
    writer.write(remaining as u32, 8);
    for &byte in ancillary {
        writer.write(byte as u32, 8);
    }
}

impl PureRustAacLcMonoEncoder {
    pub fn new(
        sampling_frequency_index: u8,
        nominal_frame_bits: usize,
        reservoir_capacity_bits: usize,
    ) -> Result<Self, AacLcEncoderError> {
        Self::new_with_frame_length(
            sampling_frequency_index,
            1024,
            nominal_frame_bits,
            reservoir_capacity_bits,
        )
    }

    pub fn new_with_frame_length(
        sampling_frequency_index: u8,
        frame_length: usize,
        nominal_frame_bits: usize,
        reservoir_capacity_bits: usize,
    ) -> Result<Self, AacLcEncoderError> {
        let sample_rate = sample_rate_from_index(sampling_frequency_index)
            .ok_or(AacLcEncoderError::InvalidRawElementLayout)?;
        let bitrate = ((nominal_frame_bits as u64 * u64::from(sample_rate)) / frame_length as u64)
            .min((1 << 23) - 1) as u32;
        Ok(Self {
            sampling_frequency_index,
            sampling_frequency: sample_rate,
            bandwidth: sample_rate / 2,
            analysis: AacLcAnalysisFilterbank::new(frame_length)?,
            psychoacoustic: AacLcPsychoacousticModel::new_with_frame_length(
                sampling_frequency_index,
                frame_length,
            )?,
            quantizer: AacLcQuantizer::new_with_frame_length(
                sampling_frequency_index,
                frame_length,
            )?,
            block_switcher: AacLcBlockSwitcher::default(),
            reservoir: AacLcBitReservoir::new(nominal_frame_bits, reservoir_capacity_bits),
            bitrate,
            vbr_quality_factor: None,
            chaos_measure_old: 0.3,
            latm_writer: LatmAacLcWriter::new(sampling_frequency_index, 1)?,
            adif_header: AdifHeader::aac_lc_mono(sampling_frequency_index, bitrate)?.to_bytes()?,
            adif_header_written: false,
        })
    }

    pub fn bit_reservoir(&self) -> &AacLcBitReservoir {
        &self.reservoir
    }

    pub fn set_bandwidth(&mut self, bandwidth: u32) {
        self.bandwidth = bandwidth.min(self.sampling_frequency / 2);
    }

    pub fn set_afterburner(&mut self, enabled: bool) {
        self.quantizer.set_afterburner(enabled);
    }

    pub fn set_bitrate_mode(&mut self, mode: u32) {
        self.vbr_quality_factor = low_delay_vbr_quality_factor(mode);
    }

    pub fn encode_raw_data_block(&mut self, input: &[f32]) -> Result<Vec<u8>, AacLcEncoderError> {
        if input.len() != self.analysis.frame_length() {
            return Err(AacLcEncoderError::InputLengthMismatch {
                expected: self.analysis.frame_length(),
                actual: input.len(),
            });
        }
        if input.iter().any(|sample| !sample.is_finite()) {
            return Err(AacLcEncoderError::NonFiniteInput);
        }
        let sequence = self.block_switcher.update(transient_ratio(input));
        let mut analysis = self.analysis.analyze_with_sequence(input, sequence)?;
        apply_spectral_bandwidth(
            &mut analysis.spectrum,
            self.sampling_frequency,
            self.bandwidth,
        );
        apply_short_spectral_bandwidth(
            &mut analysis.short_spectra,
            self.sampling_frequency,
            self.bandwidth,
        );
        let target = if self.vbr_quality_factor.is_some() {
            self.reservoir.nominal_frame_bits() + self.reservoir.capacity_bits()
        } else {
            self.reservoir.available_frame_bits()
        };
        let raw = if sequence == WindowSequence::EightShort {
            let spectra = analysis
                .short_spectra
                .as_ref()
                .expect("short analysis requested");
            let mut psycho = self.psychoacoustic.analyze_short(spectra)?;
            if let Some(quality) = self.vbr_quality_factor {
                let info = aac_sfb_info_for_frame(
                    self.sampling_frequency_index,
                    WindowSequence::EightShort,
                    self.analysis.frame_length(),
                )?;
                let mut analyses = psycho.iter_mut().collect::<Vec<_>>();
                apply_low_delay_vbr_thresholds(
                    &mut analyses,
                    info.offsets,
                    self.bitrate,
                    self.sampling_frequency,
                    info.granule_length,
                    self.bandwidth,
                    quality,
                    &mut self.chaos_measure_old,
                );
            }
            self.quantizer
                .quantize_short(
                    spectra,
                    &psycho,
                    &analysis.short_window_group_lengths,
                    target,
                )?
                .write_sce_raw_data_block(0)?
        } else {
            let mut psycho = self.psychoacoustic.analyze(&analysis.spectrum)?;
            if let Some(quality) = self.vbr_quality_factor {
                let info = aac_sfb_info_for_frame(
                    self.sampling_frequency_index,
                    WindowSequence::OnlyLong,
                    self.analysis.frame_length(),
                )?;
                apply_low_delay_vbr_thresholds(
                    &mut [&mut psycho],
                    info.offsets,
                    self.bitrate,
                    self.sampling_frequency,
                    self.analysis.frame_length(),
                    self.bandwidth,
                    quality,
                    &mut self.chaos_measure_old,
                );
            }
            self.quantizer
                .quantize_long(&analysis.spectrum, &psycho, target)?
                .write_sce_raw_data_block_with_sequence(0, sequence)?
        };
        if self.vbr_quality_factor.is_none() {
            self.reservoir.commit_frame(raw.len().saturating_mul(8))?;
        }
        Ok(raw)
    }

    pub fn encode_adts_frame(&mut self, input: &[f32]) -> Result<Vec<u8>, AacLcEncoderError> {
        let raw = self.encode_raw_data_block(input)?;
        write_adts_frame(&raw, self.sampling_frequency_index, 1)
    }

    pub fn encode_loas_frame(&mut self, input: &[f32]) -> Result<Vec<u8>, AacLcEncoderError> {
        let raw = self.encode_raw_data_block(input)?;
        let latm = self.latm_writer.write_audio_mux_element(&raw);
        Ok(write_loas_frame(&latm)?)
    }

    /// Emit the ADIF header on the first access unit and raw AAC blocks after it.
    pub fn encode_adif_access_unit(&mut self, input: &[f32]) -> Result<Vec<u8>, AacLcEncoderError> {
        let raw = self.encode_raw_data_block(input)?;
        if self.adif_header_written {
            return Ok(raw);
        }
        self.adif_header_written = true;
        let mut output = Vec::with_capacity(self.adif_header.len() + raw.len());
        output.extend_from_slice(&self.adif_header);
        output.extend_from_slice(&raw);
        Ok(output)
    }
}

#[derive(Debug, Clone)]
pub struct PureRustAacLcStereoEncoder {
    sampling_frequency_index: u8,
    sampling_frequency: u32,
    bandwidth: u32,
    frame_length: usize,
    left_analysis: AacLcAnalysisFilterbank,
    right_analysis: AacLcAnalysisFilterbank,
    psychoacoustic: AacLcPsychoacousticModel,
    quantizer: AacLcQuantizer,
    block_switcher: AacLcBlockSwitcher,
    reservoir: AacLcBitReservoir,
    bitrate: u32,
    vbr_quality_factor: Option<f32>,
    chaos_measure_old: f32,
}

/// AAC-LC encoder for the standardized 3.0 through 5.1 channel
/// configurations.  Channel elements are emitted in MPEG order and share one
/// block-switch decision, matching the synchronization requirement used by
/// libAACenc for a channel configuration.
#[derive(Debug, Clone)]
pub struct PureRustAacLcMultichannelEncoder {
    sampling_frequency_index: u8,
    sampling_frequency: u32,
    bandwidth: u32,
    channels: usize,
    channel_mode: u32,
    frame_length: usize,
    analyses: Vec<AacLcAnalysisFilterbank>,
    psychoacoustic: AacLcPsychoacousticModel,
    quantizer: AacLcQuantizer,
    block_switcher: AacLcBlockSwitcher,
    reservoir: AacLcBitReservoir,
    bitrate: u32,
    vbr_quality_factor: Option<f32>,
    chaos_measure_old: f32,
}

impl PureRustAacLcMultichannelEncoder {
    pub fn new(
        sampling_frequency_index: u8,
        channels: usize,
        nominal_frame_bits: usize,
        reservoir_capacity_bits: usize,
    ) -> Result<Self, AacLcEncoderError> {
        Self::new_with_frame_length(
            sampling_frequency_index,
            channels,
            1024,
            nominal_frame_bits,
            reservoir_capacity_bits,
        )
    }

    pub fn new_with_frame_length(
        sampling_frequency_index: u8,
        channels: usize,
        frame_length: usize,
        nominal_frame_bits: usize,
        reservoir_capacity_bits: usize,
    ) -> Result<Self, AacLcEncoderError> {
        Self::new_with_channel_mode(
            sampling_frequency_index,
            channels,
            channels as u32,
            frame_length,
            nominal_frame_bits,
            reservoir_capacity_bits,
        )
    }

    pub fn new_with_channel_mode(
        sampling_frequency_index: u8,
        channels: usize,
        channel_mode: u32,
        frame_length: usize,
        nominal_frame_bits: usize,
        reservoir_capacity_bits: usize,
    ) -> Result<Self, AacLcEncoderError> {
        let layout = multichannel_layout_for_mode(channel_mode)?;
        let layout_channels = layout
            .iter()
            .flat_map(|(_, first, second)| [Some(*first), *second])
            .flatten()
            .max()
            .map_or(0, |last| last + 1);
        if layout_channels != channels {
            return Err(AacLcEncoderError::InvalidRawElementLayout);
        }
        let sampling_frequency = sample_rate_from_index(sampling_frequency_index)
            .ok_or(AacLcEncoderError::InvalidRawElementLayout)?;
        let bitrate = ((nominal_frame_bits as u64 * u64::from(sampling_frequency))
            / frame_length as u64)
            .min((1 << 23) - 1) as u32;
        let analyses = (0..channels)
            .map(|_| AacLcAnalysisFilterbank::new(frame_length))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            sampling_frequency_index,
            sampling_frequency,
            bandwidth: sampling_frequency / 2,
            channels,
            channel_mode,
            frame_length,
            analyses,
            psychoacoustic: AacLcPsychoacousticModel::new_with_frame_length(
                sampling_frequency_index,
                frame_length,
            )?,
            quantizer: AacLcQuantizer::new_with_frame_length(
                sampling_frequency_index,
                frame_length,
            )?,
            block_switcher: AacLcBlockSwitcher::default(),
            reservoir: AacLcBitReservoir::new(nominal_frame_bits, reservoir_capacity_bits),
            bitrate,
            vbr_quality_factor: None,
            chaos_measure_old: 0.3,
        })
    }

    pub fn set_bandwidth(&mut self, bandwidth: u32) {
        self.bandwidth = bandwidth.min(self.sampling_frequency / 2);
    }

    pub fn set_afterburner(&mut self, enabled: bool) {
        self.quantizer.set_afterburner(enabled);
    }

    pub fn set_bitrate_mode(&mut self, mode: u32) {
        self.vbr_quality_factor = low_delay_vbr_quality_factor(mode);
    }

    pub fn encode_raw_data_block(
        &mut self,
        pcm: &[Vec<f32>],
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        self.encode_raw_data_block_with_sbr_fills(pcm, &[])
    }

    fn encode_raw_data_block_with_sbr_fills(
        &mut self,
        pcm: &[Vec<f32>],
        sbr_fills: &[Vec<u8>],
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        if pcm.len() != self.channels
            || pcm.iter().any(|channel| channel.len() != self.frame_length)
        {
            return Err(AacLcEncoderError::InputLengthMismatch {
                expected: self.frame_length * self.channels,
                actual: pcm.iter().map(Vec::len).sum(),
            });
        }
        let lfe_index = multichannel_layout_for_mode(self.channel_mode)?
            .into_iter()
            .find_map(|(element, first, _)| (element == ElementId::Lfe).then_some(first));
        let transient = pcm
            .iter()
            .enumerate()
            .filter(|(index, _)| Some(*index) != lfe_index)
            .map(|(_, channel)| channel)
            .map(|channel| transient_ratio(channel))
            .fold(0.0f32, f32::max);
        let sequence = self.block_switcher.update(transient);
        // Reserve element headers, ICS/scalefactor side information and final
        // byte alignment. The per-channel quantizer budget accounts only for
        // sections and spectral codewords.
        let frame_bits = if self.vbr_quality_factor.is_some() {
            self.reservoir.nominal_frame_bits() + self.reservoir.capacity_bits()
        } else {
            self.reservoir.available_frame_bits()
        };
        let channel_targets = multichannel_channel_bit_targets(self.channel_mode, frame_bits, 256)?;
        let mut analyzed_channels = Vec::with_capacity(self.channels);
        for (index, (analysis, samples)) in self.analyses.iter_mut().zip(pcm).enumerate() {
            // LFE has its own window state and may never signal EIGHT_SHORT.
            let channel_sequence = if Some(index) == lfe_index {
                WindowSequence::OnlyLong
            } else {
                sequence
            };
            let mut analyzed = analysis.analyze_with_sequence(samples, channel_sequence)?;
            apply_spectral_bandwidth(
                &mut analyzed.spectrum,
                self.sampling_frequency,
                self.bandwidth,
            );
            apply_short_spectral_bandwidth(
                &mut analyzed.short_spectra,
                self.sampling_frequency,
                self.bandwidth,
            );
            analyzed_channels.push((analyzed, channel_sequence));
        }
        let common_groups = if sequence == WindowSequence::EightShort {
            let first_groups = &analyzed_channels[0].0.short_window_group_lengths;
            if analyzed_channels
                .iter()
                .filter(|(_, channel_sequence)| *channel_sequence == WindowSequence::EightShort)
                .all(|(channel, _)| channel.short_window_group_lengths == *first_groups)
            {
                first_groups.clone()
            } else {
                // A common-window CPE must signal one grouping for both
                // channels. Falling back to eight independent windows retains
                // every spectral coefficient when channel transients disagree.
                vec![1; 8]
            }
        } else {
            Vec::new()
        };
        let mut frames = Vec::with_capacity(self.channels);
        for (channel_index, (analyzed, channel_sequence)) in analyzed_channels.iter().enumerate() {
            let target = channel_targets[channel_index];
            if *channel_sequence == WindowSequence::EightShort {
                let spectra = analyzed
                    .short_spectra
                    .as_ref()
                    .expect("short analysis requested");
                let mut psycho = self.psychoacoustic.analyze_short(spectra)?;
                if let Some(quality) = self.vbr_quality_factor {
                    let info = aac_sfb_info_for_frame(
                        self.sampling_frequency_index,
                        WindowSequence::EightShort,
                        self.frame_length,
                    )?;
                    let mut analyses = psycho.iter_mut().collect::<Vec<_>>();
                    apply_low_delay_vbr_thresholds(
                        &mut analyses,
                        info.offsets,
                        self.bitrate,
                        self.sampling_frequency,
                        info.granule_length,
                        self.bandwidth,
                        quality,
                        &mut self.chaos_measure_old,
                    );
                }
                frames.push(MultichannelQuantizedFrame::Short(
                    self.quantizer
                        .quantize_short(spectra, &psycho, &common_groups, target)?,
                ));
            } else {
                let mut psycho = self.psychoacoustic.analyze(&analyzed.spectrum)?;
                if let Some(quality) = self.vbr_quality_factor {
                    let info = aac_sfb_info_for_frame(
                        self.sampling_frequency_index,
                        WindowSequence::OnlyLong,
                        self.frame_length,
                    )?;
                    apply_low_delay_vbr_thresholds(
                        &mut [&mut psycho],
                        info.offsets,
                        self.bitrate,
                        self.sampling_frequency,
                        self.frame_length,
                        self.bandwidth,
                        quality,
                        &mut self.chaos_measure_old,
                    );
                }
                frames.push(MultichannelQuantizedFrame::Long {
                    frame: self
                        .quantizer
                        .quantize_long(&analyzed.spectrum, &psycho, target)?,
                    sequence: *channel_sequence,
                });
            }
        }
        let mut writer = BitWriter::new();
        write_multichannel_elements(&mut writer, &frames, self.channel_mode, sbr_fills)?;
        writer.write(ElementId::End.bits() as u32, 3);
        writer.byte_align();
        let raw = writer.finish();
        if self.vbr_quality_factor.is_none() {
            self.reservoir.commit_frame(raw.len() * 8)?;
        }
        Ok(raw)
    }

    pub fn sampling_frequency_index(&self) -> u8 {
        self.sampling_frequency_index
    }
}

fn multichannel_layout_for_mode(
    channel_mode: u32,
) -> Result<Vec<(ElementId, usize, Option<usize>)>, AacLcEncoderError> {
    Ok(match channel_mode {
        3 => vec![
            (ElementId::SingleChannel, 0, None),
            (ElementId::ChannelPair, 1, Some(2)),
        ],
        4 => vec![
            (ElementId::SingleChannel, 0, None),
            (ElementId::ChannelPair, 1, Some(2)),
            (ElementId::SingleChannel, 3, None),
        ],
        5 => vec![
            (ElementId::SingleChannel, 0, None),
            (ElementId::ChannelPair, 1, Some(2)),
            (ElementId::ChannelPair, 3, Some(4)),
        ],
        6 => vec![
            (ElementId::SingleChannel, 0, None),
            (ElementId::ChannelPair, 1, Some(2)),
            (ElementId::ChannelPair, 3, Some(4)),
            (ElementId::Lfe, 5, None),
        ],
        7 | 12 | 33 => vec![
            (ElementId::SingleChannel, 0, None),
            (ElementId::ChannelPair, 1, Some(2)),
            (ElementId::ChannelPair, 3, Some(4)),
            (ElementId::ChannelPair, 5, Some(6)),
            (ElementId::Lfe, 7, None),
        ],
        11 => vec![
            (ElementId::SingleChannel, 0, None),
            (ElementId::ChannelPair, 1, Some(2)),
            (ElementId::ChannelPair, 3, Some(4)),
            (ElementId::SingleChannel, 5, None),
            (ElementId::Lfe, 6, None),
        ],
        14 => vec![
            (ElementId::SingleChannel, 0, None),
            (ElementId::ChannelPair, 1, Some(2)),
            (ElementId::ChannelPair, 3, Some(4)),
            (ElementId::Lfe, 5, None),
            (ElementId::ChannelPair, 6, Some(7)),
        ],
        34 => vec![
            (ElementId::SingleChannel, 0, None),
            (ElementId::ChannelPair, 1, Some(2)),
            (ElementId::ChannelPair, 3, Some(4)),
            (ElementId::ChannelPair, 5, Some(6)),
            (ElementId::Lfe, 7, None),
        ],
        _ => return Err(AacLcEncoderError::InvalidRawElementLayout),
    })
}

/// Initial dynamic-bit distribution from `FDKaacEnc_initChannelMapping`.
/// Element shares are C-exact table values; a CPE's share is subsequently
/// divided between its two channels before psychoacoustic fitting.
fn multichannel_channel_bit_targets(
    channel_mode: u32,
    frame_bits: usize,
    channel_overhead: usize,
) -> Result<Vec<usize>, AacLcEncoderError> {
    let weights: &[f64] = match channel_mode {
        3 => &[0.4, 0.6],
        4 => &[0.3, 0.4, 0.3],
        5 => &[0.26, 0.37, 0.37],
        6 => &[0.24, 0.35, 0.35, 0.06],
        11 => &[0.2, 0.275, 0.275, 0.2, 0.05],
        7 | 12 | 33 | 34 => &[0.18, 0.26, 0.26, 0.26, 0.04],
        14 => &[0.18, 0.26, 0.26, 0.04, 0.26],
        _ => return Err(AacLcEncoderError::InvalidRawElementLayout),
    };
    let layout = multichannel_layout_for_mode(channel_mode)?;
    if layout.len() != weights.len() {
        return Err(AacLcEncoderError::InvalidRawElementLayout);
    }
    let channels = layout
        .iter()
        .flat_map(|(_, first, second)| [Some(*first), *second])
        .flatten()
        .max()
        .map_or(0, |last| last + 1);
    let mut targets = vec![0; channels];
    for ((_, first, second), &weight) in layout.into_iter().zip(weights) {
        let element_bits = (frame_bits as f64 * weight).round() as usize;
        if let Some(second) = second {
            let first_bits = element_bits / 2;
            targets[first] = first_bits.saturating_sub(channel_overhead);
            targets[second] = element_bits
                .saturating_sub(first_bits)
                .saturating_sub(channel_overhead);
        } else {
            targets[first] = element_bits.saturating_sub(channel_overhead);
        }
    }
    Ok(targets)
}

fn multichannel_non_lfe_channels(channel_mode: u32) -> Result<Vec<usize>, AacLcEncoderError> {
    let mut channels = Vec::new();
    for (element, first, second) in multichannel_layout_for_mode(channel_mode)? {
        if element == ElementId::Lfe {
            continue;
        }
        channels.push(first);
        if let Some(second) = second {
            channels.push(second);
        }
    }
    Ok(channels)
}

fn multichannel_sbr_channel_index(
    channel_mode: u32,
    channel: usize,
) -> Result<usize, AacLcEncoderError> {
    multichannel_non_lfe_channels(channel_mode)?
        .iter()
        .position(|&candidate| candidate == channel)
        .ok_or(AacLcEncoderError::InvalidRawElementLayout)
}

enum MultichannelQuantizedFrame {
    Long {
        frame: QuantizedAacLcFrame,
        sequence: WindowSequence,
    },
    Short(QuantizedAacLcShortFrame),
}

fn write_multichannel_elements(
    writer: &mut BitWriter,
    frames: &[MultichannelQuantizedFrame],
    channel_mode: u32,
    sbr_fills: &[Vec<u8>],
) -> Result<(), AacLcEncoderError> {
    let expected_fills = multichannel_layout_for_mode(channel_mode)?
        .iter()
        .filter(|(element, _, _)| *element != ElementId::Lfe)
        .count();
    if !sbr_fills.is_empty() && sbr_fills.len() != expected_fills {
        return Err(AacLcEncoderError::InvalidRawElementLayout);
    }
    let mut fill_index = 0;
    let mut sce_tag = 0u32;
    let mut cpe_tag = 0u32;
    let mut lfe_tag = 0u32;
    for (element, first, second) in multichannel_layout_for_mode(channel_mode)? {
        let tag = match element {
            ElementId::SingleChannel => {
                let tag = sce_tag;
                sce_tag += 1;
                tag
            }
            ElementId::ChannelPair => {
                let tag = cpe_tag;
                cpe_tag += 1;
                tag
            }
            ElementId::Lfe => {
                let tag = lfe_tag;
                lfe_tag += 1;
                tag
            }
            _ => return Err(AacLcEncoderError::InvalidRawElementLayout),
        };
        writer.write(element.bits() as u32, 3);
        writer.write(tag, 4);
        match (&frames[first], second.map(|index| &frames[index])) {
            (MultichannelQuantizedFrame::Long { frame, sequence }, None) => {
                write_long_channel_stream(writer, frame, Some(*sequence))?;
            }
            (MultichannelQuantizedFrame::Short(frame), None) => {
                if element == ElementId::Lfe {
                    return Err(AacLcEncoderError::InvalidRawElementLayout);
                }
                write_short_channel_stream(writer, frame, true)?;
            }
            (
                MultichannelQuantizedFrame::Long {
                    frame: left,
                    sequence,
                },
                Some(MultichannelQuantizedFrame::Long {
                    frame: right,
                    sequence: right_sequence,
                }),
            ) if sequence == right_sequence => {
                writer.write_bool(true);
                write_long_ics(writer, left.bands.len(), *sequence);
                writer.write(0, 2);
                write_long_channel_stream(writer, left, None)?;
                write_long_channel_stream(writer, right, None)?;
            }
            (
                MultichannelQuantizedFrame::Short(left),
                Some(MultichannelQuantizedFrame::Short(right)),
            ) if left.group_lengths == right.group_lengths => {
                writer.write_bool(true);
                write_short_ics(writer, left.groups[0].bands.len(), &left.group_lengths);
                writer.write(0, 2);
                write_short_channel_stream(writer, left, false)?;
                write_short_channel_stream(writer, right, false)?;
            }
            _ => return Err(AacLcEncoderError::InvalidRawElementLayout),
        }
        if element != ElementId::Lfe {
            if let Some(fill) = sbr_fills.get(fill_index) {
                writer.write(ElementId::Fill.bits() as u32, 3);
                write_packed_fill_element(writer, fill)?;
            }
            fill_index += 1;
        }
    }
    Ok(())
}

impl PureRustAacLcStereoEncoder {
    pub fn new(
        sampling_frequency_index: u8,
        nominal_frame_bits: usize,
        reservoir_capacity_bits: usize,
    ) -> Result<Self, AacLcEncoderError> {
        Self::new_with_frame_length(
            sampling_frequency_index,
            1024,
            nominal_frame_bits,
            reservoir_capacity_bits,
        )
    }

    pub fn new_with_frame_length(
        sampling_frequency_index: u8,
        frame_length: usize,
        nominal_frame_bits: usize,
        reservoir_capacity_bits: usize,
    ) -> Result<Self, AacLcEncoderError> {
        let sampling_frequency = sample_rate_from_index(sampling_frequency_index)
            .ok_or(AacLcEncoderError::InvalidRawElementLayout)?;
        let bitrate = ((nominal_frame_bits as u64 * u64::from(sampling_frequency))
            / frame_length as u64)
            .min((1 << 23) - 1) as u32;
        Ok(Self {
            sampling_frequency_index,
            sampling_frequency,
            bandwidth: sampling_frequency / 2,
            frame_length,
            left_analysis: AacLcAnalysisFilterbank::new(frame_length)?,
            right_analysis: AacLcAnalysisFilterbank::new(frame_length)?,
            psychoacoustic: AacLcPsychoacousticModel::new_with_frame_length(
                sampling_frequency_index,
                frame_length,
            )?,
            quantizer: AacLcQuantizer::new_with_frame_length(
                sampling_frequency_index,
                frame_length,
            )?,
            block_switcher: AacLcBlockSwitcher::default(),
            reservoir: AacLcBitReservoir::new(nominal_frame_bits, reservoir_capacity_bits),
            bitrate,
            vbr_quality_factor: None,
            chaos_measure_old: 0.3,
        })
    }

    pub fn set_bandwidth(&mut self, bandwidth: u32) {
        self.bandwidth = bandwidth.min(self.sampling_frequency / 2);
    }

    pub fn set_afterburner(&mut self, enabled: bool) {
        self.quantizer.set_afterburner(enabled);
    }

    pub fn set_bitrate_mode(&mut self, mode: u32) {
        self.vbr_quality_factor = low_delay_vbr_quality_factor(mode);
    }

    pub fn encode_raw_data_block(
        &mut self,
        left: &[f32],
        right: &[f32],
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        self.encode_raw_data_block_with_sbr_fill(left, right, &[])
    }

    fn encode_raw_data_block_with_sbr_fill(
        &mut self,
        left: &[f32],
        right: &[f32],
        sbr_fill: &[u8],
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        if left.len() != self.frame_length || right.len() != self.frame_length {
            return Err(AacLcEncoderError::InputLengthMismatch {
                expected: self.frame_length,
                actual: left.len().min(right.len()),
            });
        }
        let sequence = self
            .block_switcher
            .update(transient_ratio(left).max(transient_ratio(right)));
        let mut left = self.left_analysis.analyze_with_sequence(left, sequence)?;
        let mut right = self.right_analysis.analyze_with_sequence(right, sequence)?;
        apply_spectral_bandwidth(&mut left.spectrum, self.sampling_frequency, self.bandwidth);
        apply_short_spectral_bandwidth(
            &mut left.short_spectra,
            self.sampling_frequency,
            self.bandwidth,
        );
        apply_spectral_bandwidth(&mut right.spectrum, self.sampling_frequency, self.bandwidth);
        apply_short_spectral_bandwidth(
            &mut right.short_spectra,
            self.sampling_frequency,
            self.bandwidth,
        );
        let target = if self.vbr_quality_factor.is_some() {
            (self.reservoir.nominal_frame_bits() + self.reservoir.capacity_bits()) / 2
        } else {
            self.reservoir.available_frame_bits() / 2
        };
        let raw = if sequence == WindowSequence::EightShort {
            let left_spectra = left
                .short_spectra
                .as_ref()
                .expect("short analysis requested");
            let right_spectra = right
                .short_spectra
                .as_ref()
                .expect("short analysis requested");
            let mut left_psycho = self.psychoacoustic.analyze_short(left_spectra)?;
            let mut right_psycho = self.psychoacoustic.analyze_short(right_spectra)?;
            if let Some(quality) = self.vbr_quality_factor {
                let info = aac_sfb_info_for_frame(
                    self.sampling_frequency_index,
                    WindowSequence::EightShort,
                    self.frame_length,
                )?;
                let mut analyses = left_psycho
                    .iter_mut()
                    .chain(right_psycho.iter_mut())
                    .collect::<Vec<_>>();
                apply_low_delay_vbr_thresholds(
                    &mut analyses,
                    info.offsets,
                    self.bitrate,
                    self.sampling_frequency,
                    info.granule_length,
                    self.bandwidth,
                    quality,
                    &mut self.chaos_measure_old,
                );
            }
            let groups = if left.short_window_group_lengths == right.short_window_group_lengths {
                left.short_window_group_lengths.clone()
            } else {
                vec![1; 8]
            };
            let left =
                self.quantizer
                    .quantize_short(left_spectra, &left_psycho, &groups, target)?;
            let right =
                self.quantizer
                    .quantize_short(right_spectra, &right_psycho, &groups, target)?;
            QuantizedAacLcShortFrame::write_cpe_raw_data_block_with_sbr_fill_optional(
                &left,
                &right,
                0,
                (!sbr_fill.is_empty()).then_some(sbr_fill),
            )?
        } else {
            let mut left_psycho = self.psychoacoustic.analyze(&left.spectrum)?;
            let mut right_psycho = self.psychoacoustic.analyze(&right.spectrum)?;
            if let Some(quality) = self.vbr_quality_factor {
                let info = aac_sfb_info_for_frame(
                    self.sampling_frequency_index,
                    WindowSequence::OnlyLong,
                    self.frame_length,
                )?;
                apply_low_delay_vbr_thresholds(
                    &mut [&mut left_psycho, &mut right_psycho],
                    info.offsets,
                    self.bitrate,
                    self.sampling_frequency,
                    self.frame_length,
                    self.bandwidth,
                    quality,
                    &mut self.chaos_measure_old,
                );
            }
            let left = self
                .quantizer
                .quantize_long(&left.spectrum, &left_psycho, target)?;
            let right = self
                .quantizer
                .quantize_long(&right.spectrum, &right_psycho, target)?;
            QuantizedAacLcFrame::write_cpe_raw_data_block_with_sequence_and_fill(
                &left,
                &right,
                0,
                sequence,
                (!sbr_fill.is_empty()).then_some(sbr_fill),
            )?
        };
        if self.vbr_quality_factor.is_none() {
            self.reservoir.commit_frame(raw.len() * 8)?;
        }
        Ok(raw)
    }

    pub fn encode_adts_frame(
        &mut self,
        left: &[f32],
        right: &[f32],
    ) -> Result<Vec<u8>, AacLcEncoderError> {
        let raw = self.encode_raw_data_block(left, right)?;
        write_adts_frame(&raw, self.sampling_frequency_index, 2)
    }
}

impl AacLcBitReservoir {
    pub fn new(nominal_frame_bits: usize, capacity_bits: usize) -> Self {
        Self {
            nominal_frame_bits,
            capacity_bits,
            fullness_bits: 0,
        }
    }

    /// FDK initializes the encoder reservoir at its configured maximum so a
    /// difficult first access unit can use the complete frame budget.
    pub fn new_full(nominal_frame_bits: usize, capacity_bits: usize) -> Self {
        Self {
            nominal_frame_bits,
            capacity_bits,
            fullness_bits: capacity_bits,
        }
    }

    pub fn capacity_bits(&self) -> usize {
        self.capacity_bits
    }

    pub fn nominal_frame_bits(&self) -> usize {
        self.nominal_frame_bits
    }

    pub fn fullness_bits(&self) -> usize {
        self.fullness_bits
    }

    pub fn available_frame_bits(&self) -> usize {
        self.nominal_frame_bits.saturating_add(self.fullness_bits)
    }

    /// Minimum coded frame size needed to avoid silently overflowing a full
    /// CBR reservoir. FDK serializes the excess as fill data.
    pub fn minimum_cbr_frame_bits(&self) -> usize {
        self.nominal_frame_bits
            .saturating_add(self.fullness_bits)
            .saturating_sub(self.capacity_bits)
    }

    /// Clamp the psychoacoustic demand to the nominal frame allocation plus
    /// bits saved by preceding frames.
    pub fn allocate_frame(&self, requested_bits: usize) -> usize {
        requested_bits.min(self.available_frame_bits())
    }

    /// Account for the actual coded size after section/codebook selection.
    pub fn commit_frame(&mut self, used_bits: usize) -> Result<(), AacLcEncoderError> {
        let available = self.available_frame_bits();
        if used_bits > available {
            return Err(AacLcEncoderError::BitReservoirUnderflow {
                available,
                requested: used_bits,
            });
        }
        self.fullness_bits = available.saturating_sub(used_bits).min(self.capacity_bits);
        Ok(())
    }
}

fn add_er_cbr_fill<F>(
    raw: Vec<u8>,
    enabled: bool,
    reservoir: &AacLcBitReservoir,
    rewrite: F,
) -> Result<Vec<u8>, AacLcEncoderError>
where
    F: FnOnce(usize) -> Result<Vec<u8>, AacLcEncoderError>,
{
    if !enabled {
        return Ok(raw);
    }
    let target_bytes = reservoir.minimum_cbr_frame_bits().div_ceil(8);
    let fill_bytes = target_bytes.saturating_sub(raw.len());
    if fill_bytes == 0 {
        Ok(raw)
    } else {
        rewrite(fill_bytes)
    }
}

impl AacLcQuantizer {
    pub fn new(sampling_frequency_index: u8) -> Result<Self, AacLcEncoderError> {
        Self::new_with_frame_length(sampling_frequency_index, 1024)
    }

    pub fn new_with_frame_length(
        sampling_frequency_index: u8,
        frame_length: usize,
    ) -> Result<Self, AacLcEncoderError> {
        aac_sfb_info_for_frame(
            sampling_frequency_index,
            WindowSequence::OnlyLong,
            frame_length,
        )?;
        Ok(Self {
            sampling_frequency_index,
            frame_length,
            afterburner: false,
        })
    }

    /// Enable FDK's analysis-by-synthesis scalefactor refinement.
    ///
    /// The C encoder selects `invQuant = 2` for AACENC_AFTERBURNER and uses
    /// the inverse-quantized SFB distortion to refine the initial scale
    /// factors.  Its acceptance boundary is an NMR margin of 1.25.  Our
    /// quantizer already measures the actual inverse-quantized distortion, so
    /// applying the reciprocal margin to the allowed noise gives the same
    /// refinement criterion without an extra fixed-point estimation pass.
    pub fn set_afterburner(&mut self, enabled: bool) {
        self.afterburner = enabled;
    }

    pub fn quantize_long(
        &self,
        spectrum: &[f32],
        psychoacoustic: &PsychoacousticAnalysis,
        target_spectral_bits: usize,
    ) -> Result<QuantizedAacLcFrame, AacLcEncoderError> {
        let info = aac_sfb_info_for_frame(
            self.sampling_frequency_index,
            WindowSequence::OnlyLong,
            self.frame_length,
        )?;
        if spectrum.len() != info.granule_length
            || psychoacoustic.bands.len() + 1 != info.offsets.len()
        {
            return Err(AacLcEncoderError::PsychoacousticLayoutMismatch);
        }
        let mut relaxation = 1.0f32;
        let mut bands;
        loop {
            bands = info
                .offsets
                .windows(2)
                .zip(&psychoacoustic.bands)
                .map(|(border, psycho)| {
                    quantize_band(
                        &spectrum[border[0]..border[1]],
                        psycho.masking_threshold * relaxation,
                        self.afterburner,
                    )
                })
                .collect::<Vec<_>>();
            enforce_scalefactor_delta_range(&mut bands);
            let (sections, spectral_bits, section_bits) =
                select_huffman_sections(&mut bands, false);
            let bits = spectral_bits + section_bits;
            if bits <= target_spectral_bits || relaxation >= 4096.0 {
                let maximum_scale = bands
                    .iter()
                    .filter(|band| band.coefficients.iter().any(|&value| value != 0))
                    .map(|band| band.scalefactor)
                    .max()
                    .unwrap_or(-100);
                let mut factor = maximum_scale;
                normalize_section_zero_scalefactors(&mut bands, &mut factor);
                return Ok(QuantizedAacLcFrame {
                    global_gain: (i32::from(maximum_scale) + 100).clamp(0, 255) as u8,
                    estimated_spectral_bits: spectral_bits,
                    estimated_section_bits: section_bits,
                    sections,
                    bands,
                    masking_relaxation: relaxation,
                });
            }
            relaxation *= 2.0;
        }
    }

    pub fn quantize_long_with_reservoir(
        &self,
        spectrum: &[f32],
        psychoacoustic: &PsychoacousticAnalysis,
        requested_bits: usize,
        reservoir: &mut AacLcBitReservoir,
    ) -> Result<QuantizedAacLcFrame, AacLcEncoderError> {
        let target = reservoir.allocate_frame(requested_bits);
        let frame = self.quantize_long(spectrum, psychoacoustic, target)?;
        reservoir.commit_frame(
            frame
                .estimated_spectral_bits
                .saturating_add(frame.estimated_section_bits),
        )?;
        Ok(frame)
    }

    pub fn quantize_short(
        &self,
        spectra: &[Vec<f32>],
        psychoacoustic: &[PsychoacousticAnalysis],
        group_lengths: &[u8],
        target_bits: usize,
    ) -> Result<QuantizedAacLcShortFrame, AacLcEncoderError> {
        let info = aac_sfb_info_for_frame(
            self.sampling_frequency_index,
            WindowSequence::EightShort,
            self.frame_length,
        )?;
        if spectra.len() != 8
            || psychoacoustic.len() != 8
            || spectra
                .iter()
                .any(|spectrum| spectrum.len() != info.granule_length)
            || psychoacoustic
                .iter()
                .any(|psycho| psycho.bands.len() + 1 != info.offsets.len())
            || group_lengths
                .iter()
                .map(|&length| usize::from(length))
                .sum::<usize>()
                != 8
            || group_lengths.iter().any(|&length| length == 0)
        {
            return Err(AacLcEncoderError::PsychoacousticLayoutMismatch);
        }
        let mut relaxation = 1.0f32;
        loop {
            let mut groups = Vec::with_capacity(group_lengths.len());
            let mut window_start = 0usize;
            for &group_length in group_lengths {
                let window_end = window_start + usize::from(group_length);
                let mut bands = Vec::with_capacity(info.offsets.len() - 1);
                for (sfb, border) in info.offsets.windows(2).enumerate() {
                    let mut coefficients =
                        Vec::with_capacity((border[1] - border[0]) * usize::from(group_length));
                    let mut allowed_noise = 0.0f32;
                    for window in window_start..window_end {
                        coefficients.extend_from_slice(&spectra[window][border[0]..border[1]]);
                        allowed_noise += psychoacoustic[window].bands[sfb].masking_threshold;
                    }
                    bands.push(quantize_band(
                        &coefficients,
                        allowed_noise * relaxation,
                        self.afterburner,
                    ));
                }
                let (sections, spectral_bits, section_bits) =
                    select_huffman_sections(&mut bands, true);
                groups.push(QuantizedAacLcFrame {
                    global_gain: 0,
                    bands,
                    estimated_spectral_bits: spectral_bits,
                    estimated_section_bits: section_bits,
                    sections,
                    masking_relaxation: relaxation,
                });
                window_start = window_end;
            }
            let mut factor = groups
                .iter()
                .flat_map(|group| &group.bands)
                .filter(|band| band.coefficients.iter().any(|&value| value != 0))
                .map(|band| band.scalefactor)
                .max()
                .unwrap_or(-100);
            for group in &mut groups {
                enforce_scalefactor_delta_range_from(&mut group.bands, &mut factor);
                let (sections, spectral_bits, section_bits) =
                    select_huffman_sections(&mut group.bands, true);
                group.sections = sections;
                group.estimated_spectral_bits = spectral_bits;
                group.estimated_section_bits = section_bits;
            }
            let spectral_bits = groups
                .iter()
                .map(|group| group.estimated_spectral_bits)
                .sum();
            let section_bits = groups
                .iter()
                .map(|group| group.estimated_section_bits)
                .sum();
            if spectral_bits + section_bits <= target_bits || relaxation >= 4096.0 {
                let maximum_scale = groups
                    .iter()
                    .flat_map(|group| &group.bands)
                    .filter(|band| band.coefficients.iter().any(|&value| value != 0))
                    .map(|band| band.scalefactor)
                    .max()
                    .unwrap_or(-100);
                let mut factor = maximum_scale;
                for group in &mut groups {
                    normalize_section_zero_scalefactors(&mut group.bands, &mut factor);
                }
                let global_gain = (i32::from(maximum_scale) + 100).clamp(0, 255) as u8;
                for group in &mut groups {
                    group.global_gain = global_gain;
                }
                return Ok(QuantizedAacLcShortFrame {
                    global_gain,
                    group_lengths: group_lengths.to_vec(),
                    groups,
                    estimated_spectral_bits: spectral_bits,
                    estimated_section_bits: section_bits,
                    masking_relaxation: relaxation,
                });
            }
            relaxation *= 2.0;
        }
    }
}

fn quantize_band(spectrum: &[f32], allowed_noise: f32, afterburner: bool) -> QuantizedSfb {
    if spectrum.iter().all(|value| value.abs() <= 1.0e-20) {
        return QuantizedSfb {
            scalefactor: 0,
            coefficients: vec![0; spectrum.len()],
            noise_energy: 0.0,
            estimated_bits: 0,
            codebook: 0,
            codebook_bit_costs: zero_codebook_costs(),
        };
    }
    // FDKaacEnc_improveScf treats an NMR above 1.25 as requiring another
    // inverse-quantization search.  Reserving that margin here makes the
    // afterburner path choose a finer scale factor from measured distortion.
    let allowed_noise = if afterburner {
        allowed_noise / 1.25
    } else {
        allowed_noise
    };
    let mut selected = None;
    for scalefactor in -120i16..=120 {
        let gain = 2.0f32.powf(0.25 * scalefactor as f32);
        let coefficients = spectrum
            .iter()
            .map(|&value| {
                let magnitude = (value.abs() / gain).powf(0.75).round();
                (magnitude.min(8191.0) as i32) * value.signum() as i32
            })
            .collect::<Vec<_>>();
        let noise = spectrum
            .iter()
            .zip(&coefficients)
            .map(|(&original, &quantized)| {
                let reconstructed = if quantized == 0 {
                    0.0
                } else {
                    quantized.signum() as f32
                        * (quantized.unsigned_abs() as f32).powf(4.0 / 3.0)
                        * gain
                };
                (original - reconstructed).powi(2)
            })
            .sum::<f32>();
        if noise <= allowed_noise.max(1.0e-20) {
            selected = Some((scalefactor, coefficients, noise));
        } else if selected.is_some() {
            break;
        }
    }
    let (scalefactor, coefficients, noise_energy) = selected.unwrap_or_else(|| {
        let gain = 2.0f32.powf(-30.0);
        let coefficients = spectrum
            .iter()
            .map(|&value| {
                ((value.abs() / gain).powf(0.75).round().min(8191.0) as i32) * value.signum() as i32
            })
            .collect::<Vec<_>>();
        (-120, coefficients, f32::INFINITY)
    });
    let codebook_bit_costs = spectral_band_codebook_costs(&coefficients);
    let (codebook, estimated_bits) = codebook_bit_costs
        .iter()
        .enumerate()
        .filter_map(|(codebook, &cost)| cost.map(|cost| (codebook as u8, cost)))
        .min_by_key(|&(_, cost)| cost)
        .expect("codebook 11 represents every valid quantized AAC coefficient");
    QuantizedSfb {
        scalefactor,
        coefficients,
        noise_energy,
        estimated_bits,
        codebook,
        codebook_bit_costs,
    }
}

fn zero_codebook_costs() -> [Option<usize>; 12] {
    let mut costs = [None; 12];
    costs[0] = Some(0);
    costs
}

fn spectral_band_codebook_costs(coefficients: &[i32]) -> [Option<usize>; 12] {
    let mut costs = [None; 12];
    if coefficients.iter().all(|&coefficient| coefficient == 0) {
        costs[0] = Some(0);
    }
    for codebook in 1..=11u8 {
        let dimension = usize::from(spectral_codebook(codebook).unwrap().dimension);
        if !coefficients.len().is_multiple_of(dimension) {
            continue;
        }
        costs[usize::from(codebook)] = coefficients
            .chunks_exact(dimension)
            .try_fold(0usize, |total, tuple| {
                spectral_tuple_bit_cost(codebook, tuple).map(|bits| total + bits)
            });
    }
    costs
}

fn section_side_information_bits(length: usize, short_window: bool) -> usize {
    let increment = if short_window { 7 } else { 31 };
    let length_bits = if short_window { 3 } else { 5 };
    4 + length_bits * (length / increment + 1)
}

/// Jointly select spectral codebooks and contiguous AAC sections.
///
/// Dynamic programming includes both Huffman payload and section side-info,
/// allowing adjacent SFBs to share a slightly less efficient codebook when the
/// saved section header is smaller than the payload penalty.
fn select_huffman_sections(
    bands: &mut [QuantizedSfb],
    short_window: bool,
) -> (Vec<AacLcSection>, usize, usize) {
    let count = bands.len();
    let mut best = vec![usize::MAX; count + 1];
    let mut previous = vec![None; count + 1];
    best[0] = 0;
    for start in 0..count {
        if best[start] == usize::MAX {
            continue;
        }
        for codebook in 0..=11usize {
            let mut spectral = 0usize;
            for end in start + 1..=count {
                let Some(cost) = bands[end - 1].codebook_bit_costs[codebook] else {
                    break;
                };
                spectral += cost;
                let side = section_side_information_bits(end - start, short_window);
                let candidate = best[start].saturating_add(spectral).saturating_add(side);
                if candidate < best[end] {
                    best[end] = candidate;
                    previous[end] = Some((start, codebook as u8, spectral, side));
                }
            }
        }
    }

    let mut sections = Vec::new();
    let mut end = count;
    while end != 0 {
        let (start, codebook, spectral_bits, side_information_bits) =
            previous[end].expect("every quantized SFB is representable by AAC codebook 11");
        for band in &mut bands[start..end] {
            band.codebook = codebook;
            band.estimated_bits = band.codebook_bit_costs[usize::from(codebook)].unwrap();
        }
        sections.push(AacLcSection {
            codebook,
            start_sfb: start,
            end_sfb: end,
            spectral_bits,
            side_information_bits,
        });
        end = start;
    }
    sections.reverse();
    let spectral_bits = sections.iter().map(|section| section.spectral_bits).sum();
    let side_bits = sections
        .iter()
        .map(|section| section.side_information_bits)
        .sum();
    (sections, spectral_bits, side_bits)
}

#[derive(Debug, Clone)]
pub struct AacLcPsychoacousticModel {
    sampling_frequency_index: u8,
    frame_length: usize,
}

impl AacLcPsychoacousticModel {
    pub fn new(sampling_frequency_index: u8) -> Result<Self, AacLcEncoderError> {
        Self::new_with_frame_length(sampling_frequency_index, 1024)
    }

    pub fn new_with_frame_length(
        sampling_frequency_index: u8,
        frame_length: usize,
    ) -> Result<Self, AacLcEncoderError> {
        aac_sfb_info_for_frame(
            sampling_frequency_index,
            WindowSequence::OnlyLong,
            frame_length,
        )?;
        Ok(Self {
            sampling_frequency_index,
            frame_length,
        })
    }

    pub fn analyze(&self, spectrum: &[f32]) -> Result<PsychoacousticAnalysis, AacLcEncoderError> {
        if spectrum.len() != self.frame_length {
            return Err(AacLcEncoderError::InputLengthMismatch {
                expected: self.frame_length,
                actual: spectrum.len(),
            });
        }
        let offsets = aac_sfb_info_for_frame(
            self.sampling_frequency_index,
            WindowSequence::OnlyLong,
            self.frame_length,
        )?
        .offsets;
        Ok(analyze_psychoacoustic_bands(spectrum, offsets))
    }

    pub fn analyze_short(
        &self,
        spectra: &[Vec<f32>],
    ) -> Result<Vec<PsychoacousticAnalysis>, AacLcEncoderError> {
        let short_length = self.frame_length / 8;
        if spectra.len() != 8 || spectra.iter().any(|window| window.len() != short_length) {
            return Err(AacLcEncoderError::InputLengthMismatch {
                expected: self.frame_length,
                actual: spectra.iter().map(Vec::len).sum(),
            });
        }
        let offsets = aac_sfb_info_for_frame(
            self.sampling_frequency_index,
            WindowSequence::EightShort,
            self.frame_length,
        )?
        .offsets;
        Ok(spectra
            .iter()
            .map(|spectrum| analyze_psychoacoustic_bands(spectrum, offsets))
            .collect())
    }
}

fn analyze_psychoacoustic_bands(spectrum: &[f32], offsets: &[usize]) -> PsychoacousticAnalysis {
    let mut energies = Vec::with_capacity(offsets.len() - 1);
    let mut tonalities = Vec::with_capacity(offsets.len() - 1);
    for border in offsets.windows(2) {
        let powers = spectrum[border[0]..border[1]]
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value) + 1.0e-30)
            .collect::<Vec<_>>();
        let energy = powers.iter().sum::<f64>() as f32;
        let arithmetic = powers.iter().sum::<f64>() / powers.len() as f64;
        let geometric =
            (powers.iter().map(|power| power.ln()).sum::<f64>() / powers.len() as f64).exp();
        let flatness = (geometric / arithmetic.max(1.0e-30)).clamp(0.0, 1.0);
        energies.push(energy);
        tonalities.push((1.0 - flatness) as f32);
    }
    let mut bands = Vec::with_capacity(energies.len());
    let mut perceptual_entropy = 0.0f32;
    for band in 0..energies.len() {
        // Approximate the spreading function in the Bark domain. Upward
        // masking is weaker than downward masking.
        let spread = energies[band]
            + band
                .checked_sub(1)
                .map_or(0.0, |index| energies[index] * 0.15)
            + energies.get(band + 1).copied().unwrap_or(0.0) * 0.05;
        let masking_factor = 0.16 - 0.13 * tonalities[band];
        let width = (offsets[band + 1] - offsets[band]) as f32;
        let absolute_threshold = width * 1.0e-9 * (1.0 + band as f32 * 0.08).powi(2);
        let threshold = (spread * masking_factor)
            .max(absolute_threshold)
            .min(energies[band].max(absolute_threshold));
        if energies[band] > threshold {
            perceptual_entropy += width * (energies[band] / threshold).log2();
        }
        bands.push(PsychoacousticBand {
            energy: energies[band],
            masking_threshold: threshold,
            tonality: tonalities[band],
            form_factor: spectrum[offsets[band]..offsets[band + 1]]
                .iter()
                .map(|value| value.abs().sqrt())
                .sum(),
        });
    }
    PsychoacousticAnalysis {
        bands,
        perceptual_entropy,
    }
}

fn fdk_bark_line_value(frame_length: usize, line: usize, sample_rate: u32) -> f64 {
    let frequency = line as f64 * sample_rate as f64 / (2 * frame_length) as f64;
    let atan_low = (frequency * (4.0 / 3.0) * 0.0001).atan();
    13.3 * (frequency * 0.00076).atan() + 3.5 * atan_low * atan_low
}

fn low_delay_min_snr(
    bitrate: u32,
    sample_rate: u32,
    frame_length: usize,
    offsets: &[usize],
    active_bands: usize,
) -> Vec<f32> {
    let active_bands = active_bands.min(offsets.len().saturating_sub(1));
    if active_bands == 0 || sample_rate == 0 {
        return Vec::new();
    }
    let active_barks =
        fdk_bark_line_value(frame_length, offsets[active_bands], sample_rate).min(24.0);
    let bark_factor = (active_barks / 25.0).max(f64::MIN_POSITIVE);
    let pe_per_window = bitrate as f64 / sample_rate as f64 * 1.18 * 0.024 * frame_length as f64;
    let pe_per_bark = pe_per_window / bark_factor;
    (0..active_bands)
        .map(|band| {
            let bark_width = fdk_bark_line_value(frame_length, offsets[band + 1], sample_rate)
                - fdk_bark_line_value(frame_length, offsets[band], sample_rate);
            let width = (offsets[band + 1] - offsets[band]).max(1) as f64;
            let pe_per_line = pe_per_bark * bark_width / width;
            let ratio = 1.0 / (2.0f64.powf(pe_per_line) - 1.5).max(1.0);
            ratio.clamp(0.003, 0.8) as f32
        })
        .collect()
}

#[derive(Clone, Copy)]
struct LowDelayBitsToPeEntry {
    bitrate: u32,
    // Tenths: afterburner-off mono/stereo, afterburner-on mono/stereo.
    factor: [u8; 4],
}

const B2PE_16000: &[LowDelayBitsToPeEntry] = &[
    LowDelayBitsToPeEntry {
        bitrate: 10_000,
        factor: [16, 0, 14, 0],
    },
    LowDelayBitsToPeEntry {
        bitrate: 24_000,
        factor: [18, 14, 16, 12],
    },
    LowDelayBitsToPeEntry {
        bitrate: 32_000,
        factor: [18, 16, 16, 14],
    },
    LowDelayBitsToPeEntry {
        bitrate: 48_000,
        factor: [16, 18, 16, 16],
    },
    LowDelayBitsToPeEntry {
        bitrate: 64_000,
        factor: [12, 16, 12, 16],
    },
    LowDelayBitsToPeEntry {
        bitrate: 96_000,
        factor: [14, 18, 14, 16],
    },
    LowDelayBitsToPeEntry {
        bitrate: 128_000,
        factor: [14, 18, 14, 18],
    },
    LowDelayBitsToPeEntry {
        bitrate: 148_000,
        factor: [14, 18, 14, 14],
    },
];
const B2PE_22050: &[LowDelayBitsToPeEntry] = &[
    LowDelayBitsToPeEntry {
        bitrate: 16_000,
        factor: [16, 14, 12, 8],
    },
    LowDelayBitsToPeEntry {
        bitrate: 24_000,
        factor: [16, 14, 14, 10],
    },
    LowDelayBitsToPeEntry {
        bitrate: 32_000,
        factor: [14, 14, 14, 12],
    },
    LowDelayBitsToPeEntry {
        bitrate: 48_000,
        factor: [12, 16, 12, 14],
    },
    LowDelayBitsToPeEntry {
        bitrate: 64_000,
        factor: [16, 16, 16, 14],
    },
    LowDelayBitsToPeEntry {
        bitrate: 96_000,
        factor: [18, 16, 18, 16],
    },
    LowDelayBitsToPeEntry {
        bitrate: 128_000,
        factor: [18, 18, 16, 16],
    },
    LowDelayBitsToPeEntry {
        bitrate: 148_000,
        factor: [14, 18, 14, 16],
    },
];
const B2PE_24000: &[LowDelayBitsToPeEntry] = &[
    LowDelayBitsToPeEntry {
        bitrate: 16_000,
        factor: [14, 14, 12, 8],
    },
    LowDelayBitsToPeEntry {
        bitrate: 24_000,
        factor: [16, 12, 14, 10],
    },
    LowDelayBitsToPeEntry {
        bitrate: 32_000,
        factor: [14, 12, 14, 8],
    },
    LowDelayBitsToPeEntry {
        bitrate: 48_000,
        factor: [14, 16, 14, 14],
    },
    LowDelayBitsToPeEntry {
        bitrate: 64_000,
        factor: [16, 16, 16, 14],
    },
    LowDelayBitsToPeEntry {
        bitrate: 96_000,
        factor: [18, 16, 18, 16],
    },
    LowDelayBitsToPeEntry {
        bitrate: 128_000,
        factor: [14, 16, 18, 18],
    },
    LowDelayBitsToPeEntry {
        bitrate: 148_000,
        factor: [14, 16, 14, 18],
    },
];
const B2PE_32000: &[LowDelayBitsToPeEntry] = &[
    LowDelayBitsToPeEntry {
        bitrate: 16_000,
        factor: [12, 14, 8, 8],
    },
    LowDelayBitsToPeEntry {
        bitrate: 24_000,
        factor: [14, 12, 10, 6],
    },
    LowDelayBitsToPeEntry {
        bitrate: 32_000,
        factor: [12, 12, 10, 8],
    },
    LowDelayBitsToPeEntry {
        bitrate: 48_000,
        factor: [14, 14, 12, 12],
    },
    LowDelayBitsToPeEntry {
        bitrate: 64_000,
        factor: [16, 14, 16, 12],
    },
    LowDelayBitsToPeEntry {
        bitrate: 96_000,
        factor: [16, 14, 16, 14],
    },
    LowDelayBitsToPeEntry {
        bitrate: 128_000,
        factor: [18, 16, 18, 16],
    },
    LowDelayBitsToPeEntry {
        bitrate: 148_000,
        factor: [18, 16, 18, 16],
    },
    LowDelayBitsToPeEntry {
        bitrate: 160_000,
        factor: [18, 16, 18, 16],
    },
    LowDelayBitsToPeEntry {
        bitrate: 200_000,
        factor: [14, 16, 14, 16],
    },
    LowDelayBitsToPeEntry {
        bitrate: 320_000,
        factor: [32, 18, 32, 18],
    },
];
const B2PE_44100: &[LowDelayBitsToPeEntry] = &[
    LowDelayBitsToPeEntry {
        bitrate: 16_000,
        factor: [12, 16, 8, 10],
    },
    LowDelayBitsToPeEntry {
        bitrate: 24_000,
        factor: [10, 12, 10, 8],
    },
    LowDelayBitsToPeEntry {
        bitrate: 32_000,
        factor: [12, 12, 8, 6],
    },
    LowDelayBitsToPeEntry {
        bitrate: 48_000,
        factor: [12, 12, 12, 8],
    },
    LowDelayBitsToPeEntry {
        bitrate: 64_000,
        factor: [14, 12, 12, 10],
    },
    LowDelayBitsToPeEntry {
        bitrate: 96_000,
        factor: [16, 12, 16, 12],
    },
    LowDelayBitsToPeEntry {
        bitrate: 128_000,
        factor: [16, 16, 16, 14],
    },
    LowDelayBitsToPeEntry {
        bitrate: 148_000,
        factor: [16, 16, 16, 16],
    },
    LowDelayBitsToPeEntry {
        bitrate: 160_000,
        factor: [16, 16, 16, 16],
    },
    LowDelayBitsToPeEntry {
        bitrate: 200_000,
        factor: [18, 16, 16, 16],
    },
    LowDelayBitsToPeEntry {
        bitrate: 320_000,
        factor: [32, 16, 32, 16],
    },
];
const B2PE_48000: &[LowDelayBitsToPeEntry] = &[
    LowDelayBitsToPeEntry {
        bitrate: 16_000,
        factor: [14, 0, 8, 0],
    },
    LowDelayBitsToPeEntry {
        bitrate: 24_000,
        factor: [14, 12, 10, 8],
    },
    LowDelayBitsToPeEntry {
        bitrate: 32_000,
        factor: [10, 12, 6, 8],
    },
    LowDelayBitsToPeEntry {
        bitrate: 48_000,
        factor: [12, 10, 8, 8],
    },
    LowDelayBitsToPeEntry {
        bitrate: 64_000,
        factor: [12, 12, 12, 10],
    },
    LowDelayBitsToPeEntry {
        bitrate: 96_000,
        factor: [16, 14, 16, 12],
    },
    LowDelayBitsToPeEntry {
        bitrate: 128_000,
        factor: [16, 16, 16, 14],
    },
    LowDelayBitsToPeEntry {
        bitrate: 148_000,
        factor: [16, 16, 16, 14],
    },
    LowDelayBitsToPeEntry {
        bitrate: 160_000,
        factor: [16, 16, 16, 14],
    },
    LowDelayBitsToPeEntry {
        bitrate: 200_000,
        factor: [12, 16, 16, 14],
    },
    LowDelayBitsToPeEntry {
        bitrate: 320_000,
        factor: [32, 16, 32, 16],
    },
];

/// Port of `FDKaacEnc_InitBits2PeFactor`, returned as an ordinary real factor.
fn low_delay_bits_to_pe_factor(
    bitrate: u32,
    channels: usize,
    sample_rate: u32,
    afterburner: bool,
    dead_zone_quantization: bool,
) -> f32 {
    let mut factor = 1.18f32;
    let table = [
        (16_000, B2PE_16000),
        (22_050, B2PE_22050),
        (24_000, B2PE_24000),
        (32_000, B2PE_32000),
        (44_100, B2PE_44100),
        (48_000, B2PE_48000),
    ]
    .into_iter()
    .rev()
    .find(|(rate, _)| sample_rate >= *rate)
    .map(|(_, table)| table);

    if let (1 | 2, Some(table)) = (channels, table) {
        let index = usize::from(afterburner) * 2 + channels - 1;
        let interpolated = if bitrate >= table.last().unwrap().bitrate {
            Some(table.last().unwrap().factor[index] as f32 * 0.1)
        } else {
            table.windows(2).find_map(|pair| {
                (pair[0].bitrate <= bitrate && bitrate < pair[1].bitrate).then(|| {
                    let position = (bitrate - pair[0].bitrate) as f32
                        / (pair[1].bitrate - pair[0].bitrate) as f32;
                    (pair[0].factor[index] as f32
                        + position * (pair[1].factor[index] as f32 - pair[0].factor[index] as f32))
                        * 0.1
                })
            })
        };
        if let Some(value) = interpolated.filter(|&value| value >= 0.35) {
            factor = value.min(3.0);
        }
    }

    if dead_zone_quantization {
        let per_channel = bitrate / channels.max(1) as u32;
        factor += if per_channel > 32_000 && per_channel <= 40_000 {
            0.4
        } else if per_channel >= 16_000 {
            0.3
        } else {
            0.0
        };
    }
    factor
}

fn low_delay_vbr_quality_factor(mode: u32) -> Option<f32> {
    match mode {
        1 => Some(0.150),
        2 => Some(0.162),
        3 => Some(0.176),
        4 => Some(0.120),
        5 => Some(0.070),
        _ => None,
    }
}

fn low_delay_form_factor_chaos(analysis: &PsychoacousticAnalysis, offsets: &[usize]) -> (f32, f32) {
    let mut energy = 0.0f32;
    let mut form_factor = 0.0f32;
    let mut lines = 0usize;
    for (band, border) in analysis
        .bands
        .iter()
        .zip(offsets.windows(2))
        .filter(|(band, _)| band.energy > band.masking_threshold)
    {
        energy += band.energy.max(0.0);
        form_factor += band.form_factor.max(0.0);
        lines += border[1] - border[0];
    }
    if lines == 0 || energy <= 0.0 {
        (1.0, energy)
    } else {
        let average_quarter_root = (energy / lines as f32).powf(0.25);
        let active_lines = form_factor / average_quarter_root.max(f32::MIN_POSITIVE);
        // FDK accumulates energy with SCALE_NRGS=8 and restores its fourth
        // root, contributing a factor of 2^(8/4)=4. The channel measure is
        // intentionally not clipped here; clipping happens after channels are
        // energy-weighted.
        ((4.0 * active_lines / lines as f32).max(0.0), energy)
    }
}

fn reduce_low_delay_vbr_band(
    energy: f32,
    threshold: f32,
    minimum_snr: f32,
    reduction: f32,
    avoid_hole: u8,
) -> (f32, u8) {
    // FDK stores ld64 values in Q1.31. MIN_LDTHRESH=-0.515625 therefore
    // represents a linear threshold of 2^-33. Bands below it are deliberately
    // left untouched by the VBR reducer.
    const MINIMUM_VBR_THRESHOLD: f32 = 1.0 / 8_589_934_592.0;
    if energy <= threshold
        || energy <= 0.0
        || threshold < MINIMUM_VBR_THRESHOLD
        || !threshold.is_finite()
    {
        return (threshold, avoid_hole);
    }
    let adjusted = (threshold.powf(0.25) + reduction.max(0.0)).powi(4);
    let mut reduced = adjusted;
    let mut next_avoid_hole = avoid_hole;
    if reduced > energy * minimum_snr && avoid_hole != 0 {
        reduced = (energy * minimum_snr).max(threshold);
        next_avoid_hole = 2;
    }
    (reduced.max(MINIMUM_VBR_THRESHOLD), next_avoid_hole)
}

/// Float-domain port of the long-window branch of
/// `FDKaacEnc_reduceThresholdsVBR`.
fn apply_low_delay_vbr_thresholds(
    analyses: &mut [&mut PsychoacousticAnalysis],
    offsets: &[usize],
    bitrate: u32,
    sample_rate: u32,
    frame_length: usize,
    bandwidth: u32,
    quality_factor: f32,
    chaos_measure_old: &mut f32,
) {
    let frame_energy = analyses
        .iter()
        .flat_map(|analysis| analysis.bands.iter())
        .map(|band| band.energy.max(0.0))
        .sum::<f32>()
        .max(1.0e-10);
    let weighted_chaos = (analyses
        .iter()
        .map(|analysis| low_delay_form_factor_chaos(analysis, offsets))
        .map(|(chaos, energy)| chaos * energy)
        .sum::<f32>()
        / frame_energy)
        .min(1.0);
    let averaged = 0.25 * weighted_chaos + 0.75 * *chaos_measure_old;
    let chaos = weighted_chaos.min(averaged);
    *chaos_measure_old = chaos;
    let shaped_chaos = (0.2 + (0.7 / 0.3) * (chaos - 0.2)).clamp(0.1, 1.0);
    let reduction = quality_factor * shaped_chaos * frame_energy.powf(0.25);

    let lowpass_line =
        ((2 * u64::from(bandwidth) * frame_length as u64) / u64::from(sample_rate.max(1))) as usize;
    let active_bands = offsets
        .iter()
        .position(|&offset| offset >= lowpass_line)
        .unwrap_or(offsets.len().saturating_sub(1))
        .max(1)
        .min(offsets.len().saturating_sub(1));
    let minimum_snr = low_delay_min_snr(
        bitrate / analyses.len().max(1) as u32,
        sample_rate,
        frame_length,
        offsets,
        active_bands,
    );
    for analysis in analyses {
        for (band, &min_snr) in analysis.bands.iter_mut().zip(&minimum_snr) {
            band.masking_threshold = reduce_low_delay_vbr_band(
                band.energy,
                band.masking_threshold,
                min_snr,
                reduction,
                1,
            )
            .0;
        }
        analysis.perceptual_entropy = analysis
            .bands
            .iter()
            .zip(offsets.windows(2))
            .filter(|(band, _)| band.energy > band.masking_threshold)
            .map(|(band, border)| {
                (border[1] - border[0]) as f32
                    * (band.energy / band.masking_threshold.max(f32::MIN_POSITIVE)).log2()
            })
            .sum();
    }
}

/// Long-window CBR threshold adaptation corresponding to
/// `FDKaacEnc_adaptThresholdsToPe`/`FDKaacEnc_reduceThresholdsCBR`.
fn reduce_low_delay_cbr_band(
    energy: f32,
    threshold: f32,
    minimum_snr: f32,
    reduction: f32,
    avoid_hole: u8,
) -> (f32, u8) {
    const NO_AH: u8 = 0;
    const AH_ACTIVE: u8 = 2;
    if energy <= threshold || energy <= 0.0 || avoid_hole == AH_ACTIVE {
        return (threshold, avoid_hole);
    }

    let mut reduced = (threshold.max(0.0).powf(0.25) + reduction.max(0.0)).powi(4);
    let mut next_avoid_hole = avoid_hole;
    let avoid_hole_limit = energy * minimum_snr;
    if reduced > avoid_hole_limit && avoid_hole != NO_AH {
        reduced = avoid_hole_limit.max(threshold);
        next_avoid_hole = AH_ACTIVE;
    }
    // `FDKaacEnc_reduceThresholdsCBR` retains at least a 29 dB
    // energy-to-threshold ratio even for bands where avoid-hole is disabled.
    reduced = reduced.max(energy * 10.0f32.powf(-2.9));
    (reduced, next_avoid_hole)
}

fn apply_low_delay_cbr_thresholds(
    analyses: &mut [&mut PsychoacousticAnalysis],
    offsets: &[usize],
    target_pe: f32,
    bitrate: u32,
    sample_rate: u32,
    frame_length: usize,
    bandwidth: u32,
) {
    let current_pe: f32 = analyses
        .iter()
        .map(|analysis| analysis.perceptual_entropy)
        .sum();
    if current_pe <= target_pe.max(0.0) {
        return;
    }
    let originals = analyses
        .iter()
        .map(|analysis| {
            analysis
                .bands
                .iter()
                .map(|band| band.masking_threshold)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let lowpass_line =
        ((2 * u64::from(bandwidth) * frame_length as u64) / u64::from(sample_rate.max(1))) as usize;
    let active_bands = offsets
        .iter()
        .position(|&offset| offset >= lowpass_line)
        .unwrap_or(offsets.len().saturating_sub(1))
        .max(1)
        .min(offsets.len().saturating_sub(1));
    let minimum_snr = low_delay_min_snr(
        bitrate / analyses.len().max(1) as u32,
        sample_rate,
        frame_length,
        offsets,
        active_bands,
    );

    let pe_for_reduction = |reduction: f32| -> f32 {
        analyses
            .iter()
            .enumerate()
            .map(|(channel, analysis)| {
                analysis
                    .bands
                    .iter()
                    .zip(offsets.windows(2))
                    .zip(&minimum_snr)
                    .enumerate()
                    .filter_map(|(band_index, ((band, border), &min_snr))| {
                        let threshold = reduce_low_delay_cbr_band(
                            band.energy,
                            originals[channel][band_index],
                            min_snr,
                            reduction,
                            1,
                        )
                        .0;
                        (band.energy > threshold).then(|| {
                            (border[1] - border[0]) as f32
                                * (band.energy / threshold.max(f32::MIN_POSITIVE)).log2()
                        })
                    })
                    .sum::<f32>()
            })
            .sum()
    };

    let mut lower = 0.0f32;
    let mut upper = originals
        .iter()
        .flatten()
        .copied()
        .fold(0.0f32, f32::max)
        .powf(0.25)
        .max(1.0e-12);
    while pe_for_reduction(upper) > target_pe && upper < 1.0e12 {
        upper *= 2.0;
    }
    // FDK performs one initial estimate and up to three second guesses.  A
    // bounded binary correction gives the same common fourth-root reduction
    // while avoiding fixed-point approximation drift in this float path.
    for _ in 0..12 {
        let middle = (lower + upper) * 0.5;
        if pe_for_reduction(middle) > target_pe {
            lower = middle;
        } else {
            upper = middle;
        }
    }

    for (channel, analysis) in analyses.iter_mut().enumerate() {
        for (band_index, (band, &min_snr)) in
            analysis.bands.iter_mut().zip(&minimum_snr).enumerate()
        {
            band.masking_threshold = reduce_low_delay_cbr_band(
                band.energy,
                originals[channel][band_index],
                min_snr,
                upper,
                1,
            )
            .0;
        }
        analysis.perceptual_entropy = analysis
            .bands
            .iter()
            .zip(offsets.windows(2))
            .filter(|(band, _)| band.energy > band.masking_threshold)
            .map(|(band, border)| {
                (border[1] - border[0]) as f32
                    * (band.energy / band.masking_threshold.max(f32::MIN_POSITIVE)).log2()
            })
            .sum();
    }

    let total_pe = |analyses: &[&mut PsychoacousticAnalysis]| -> f32 {
        analyses
            .iter()
            .map(|analysis| analysis.perceptual_entropy)
            .sum()
    };
    if total_pe(analyses) <= target_pe {
        return;
    }

    // FDK first relaxes min-SNR to a 1 dB energy/threshold ratio, starting at
    // the highest SFB across all channels.
    let one_db_ratio = 10.0f32.powf(-0.1);
    for sfb in (0..minimum_snr.len()).rev() {
        for analysis in analyses.iter_mut() {
            let band = &mut analysis.bands[sfb];
            if band.energy > band.masking_threshold {
                band.masking_threshold = band.masking_threshold.max(band.energy * one_db_ratio);
            }
            analysis.perceptual_entropy = analysis
                .bands
                .iter()
                .zip(offsets.windows(2))
                .filter(|(band, _)| band.energy > band.masking_threshold)
                .map(|(band, border)| {
                    (border[1] - border[0]) as f32
                        * (band.energy / band.masking_threshold.max(f32::MIN_POSITIVE)).log2()
                })
                .sum();
        }
        if total_pe(analyses) <= target_pe {
            return;
        }
    }

    // If 1 dB is still insufficient, mirror `allowMoreHoles`: consider only
    // the configured high-band region, sweep eight logarithmic energy levels,
    // and within each level remove higher SFBs first.
    let start_sfb = if bitrate / analyses.len().max(1) as u32 >= 20_000 {
        15.min(minimum_snr.len())
    } else {
        0
    };
    let energies = analyses
        .iter()
        .flat_map(|analysis| analysis.bands[start_sfb..].iter())
        .filter(|band| band.energy > 0.0 && band.energy > band.masking_threshold)
        .map(|band| band.energy)
        .collect::<Vec<_>>();
    if energies.is_empty() {
        return;
    }
    let minimum_energy = energies.iter().copied().fold(f32::INFINITY, f32::min);
    let average_energy = energies.iter().sum::<f32>() / energies.len() as f32;
    for level in 0..8 {
        let fraction = (2 * level + 1) as f32 / 15.0;
        let border = minimum_energy
            * (average_energy / minimum_energy.max(f32::MIN_POSITIVE)).powf(fraction);
        for sfb in (start_sfb..minimum_snr.len()).rev() {
            for analysis in analyses.iter_mut() {
                let band = &mut analysis.bands[sfb];
                if band.energy < border && band.energy > band.masking_threshold {
                    band.masking_threshold = band.energy;
                    analysis.perceptual_entropy = analysis
                        .bands
                        .iter()
                        .zip(offsets.windows(2))
                        .filter(|(band, _)| band.energy > band.masking_threshold)
                        .map(|(band, border)| {
                            (border[1] - border[0]) as f32
                                * (band.energy / band.masking_threshold.max(f32::MIN_POSITIVE))
                                    .log2()
                        })
                        .sum();
                }
            }
            if total_pe(analyses) <= target_pe {
                return;
            }
        }
    }
}

fn analyze_low_delay_psychoacoustic_bands(
    spectrum: &[f32],
    offsets: &[usize],
    nominal_frame_bits: usize,
    sample_rate: u32,
    frame_length: usize,
    bandwidth: u32,
    channels: usize,
) -> PsychoacousticAnalysis {
    let mut analysis = analyze_psychoacoustic_bands(spectrum, offsets);
    let bitrate = ((nominal_frame_bits as u64 * u64::from(sample_rate)) / frame_length as u64)
        .min(u64::from(u32::MAX)) as u32
        / channels.max(1) as u32;
    let lowpass_line =
        ((2 * u64::from(bandwidth) * frame_length as u64) / u64::from(sample_rate.max(1))) as usize;
    let active_bands = offsets
        .iter()
        .position(|&offset| offset >= lowpass_line)
        .unwrap_or(offsets.len().saturating_sub(1))
        .max(1)
        .min(analysis.bands.len());
    let minimum_snr = low_delay_min_snr(bitrate, sample_rate, frame_length, offsets, active_bands);
    for (band, &min_snr) in analysis.bands.iter_mut().zip(&minimum_snr) {
        let maximum_threshold = band.energy * min_snr;
        if maximum_threshold.is_finite() && maximum_threshold > 0.0 {
            band.masking_threshold = band.masking_threshold.min(maximum_threshold);
        }
    }
    analysis.perceptual_entropy = analysis
        .bands
        .iter()
        .zip(offsets.windows(2))
        .filter(|(band, _)| band.energy > band.masking_threshold)
        .map(|(band, border)| {
            (border[1] - border[0]) as f32
                * (band.energy / band.masking_threshold.max(f32::MIN_POSITIVE)).log2()
        })
        .sum();
    analysis
}

#[derive(Debug, Clone)]
pub struct AacLcBlockSwitcher {
    sequence: WindowSequence,
    attack_threshold: f32,
}

#[derive(Debug, Clone)]
struct AacLdBlockSwitcher {
    previous_filtered_energy: [f64; 4],
    accumulated_energy: f64,
    previous_input: f64,
    previous_filtered: f64,
    last_attack: bool,
    last_attack_index: usize,
}

impl Default for AacLdBlockSwitcher {
    fn default() -> Self {
        Self {
            previous_filtered_energy: [0.0; 4],
            accumulated_energy: 0.0,
            previous_input: 0.0,
            previous_filtered: 0.0,
            last_attack: false,
            last_attack_index: 0,
        }
    }
}

impl AacLdBlockSwitcher {
    fn detect(&mut self, input: &[f32]) -> bool {
        debug_assert_eq!(input.len() % 4, 0);
        let window_length = input.len() / 4;
        let mut filtered_energy = [0.0f64; 4];
        for (window, samples) in input.chunks_exact(window_length).enumerate() {
            for &sample in samples {
                // Float equivalent of FDK's two-state high-pass section. PCM
                // follows this crate's signed-i16-scale convention.
                let value = f64::from(sample) / 65_536.0;
                let filtered =
                    0.7548 * (value - self.previous_input) + 0.5095 * self.previous_filtered;
                self.previous_input = value;
                self.previous_filtered = filtered;
                filtered_energy[window] += filtered * filtered;
            }
        }

        let mut attack = false;
        let mut attack_index = 0;
        let mut previous = self.previous_filtered_energy[3];
        for (index, &energy) in filtered_energy.iter().enumerate() {
            self.accumulated_energy = 0.7 * self.accumulated_energy + 0.3 * previous;
            if 0.1 * energy > self.accumulated_energy {
                attack = true;
                attack_index = index;
            }
            previous = energy;
        }
        // C's 1e6*NORM_PCM_ENERGY threshold, expressed after the input / 2
        // normalization and BLOCK_SWITCH_ENERGY_SHIFT scaling.
        if filtered_energy.iter().copied().fold(0.0, f64::max)
            < 1_000_000.0 / (32_768.0 * 32_768.0 * 128.0)
        {
            attack = false;
        }
        if !attack
            && self.last_attack
            && self.last_attack_index == 3
            && self.previous_filtered_energy[3] > 10.0 * filtered_energy[1]
        {
            attack = true;
            attack_index = 0;
        }
        self.previous_filtered_energy = filtered_energy;
        self.last_attack = attack;
        self.last_attack_index = attack_index;
        attack
    }
}

impl Default for AacLcBlockSwitcher {
    fn default() -> Self {
        Self {
            sequence: WindowSequence::OnlyLong,
            attack_threshold: 10.0,
        }
    }
}

impl AacLcBlockSwitcher {
    pub fn sequence(&self) -> WindowSequence {
        self.sequence
    }

    pub fn update(&mut self, transient_ratio: f32) -> WindowSequence {
        let attack = transient_ratio >= self.attack_threshold;
        self.sequence = match (self.sequence, attack) {
            (WindowSequence::OnlyLong, false) => WindowSequence::OnlyLong,
            (WindowSequence::OnlyLong, true) => WindowSequence::LongStart,
            (WindowSequence::LongStart, _) => WindowSequence::EightShort,
            (WindowSequence::EightShort, true) => WindowSequence::EightShort,
            (WindowSequence::EightShort, false) => WindowSequence::LongStop,
            (WindowSequence::LongStop, false) => WindowSequence::OnlyLong,
            (WindowSequence::LongStop, true) => WindowSequence::LongStart,
        };
        self.sequence
    }
}

impl AacLcAnalysisFilterbank {
    pub fn new(frame_length: usize) -> Result<Self, AacLcEncoderError> {
        if !matches!(frame_length, 480 | 512 | 960 | 1024) {
            return Err(AacLcEncoderError::UnsupportedFrameLength(frame_length));
        }
        let sample_count = 2 * frame_length;
        let window = (0..sample_count)
            .map(|index| (std::f64::consts::PI / sample_count as f64 * (index as f64 + 0.5)).sin())
            .collect::<Vec<_>>();
        let normalization = (2.0 / frame_length as f64).sqrt();
        let mut kernel = Vec::with_capacity(frame_length * sample_count);
        for coefficient in 0..frame_length {
            for sample in 0..sample_count {
                let phase = std::f64::consts::PI / frame_length as f64
                    * (sample as f64 + 0.5 + frame_length as f64 / 2.0)
                    * (coefficient as f64 + 0.5);
                kernel.push(phase.cos() * normalization);
            }
        }
        let short_length = frame_length / 8;
        let short_sample_count = 2 * short_length;
        let short_window = (0..short_sample_count)
            .map(|index| {
                (std::f64::consts::PI / short_sample_count as f64 * (index as f64 + 0.5)).sin()
            })
            .collect::<Vec<_>>();
        let short_normalization = (2.0 / short_length as f64).sqrt();
        let mut short_kernel = Vec::with_capacity(short_length * short_sample_count);
        for coefficient in 0..short_length {
            for sample in 0..short_sample_count {
                let phase = std::f64::consts::PI / short_length as f64
                    * (sample as f64 + 0.5 + short_length as f64 / 2.0)
                    * (coefficient as f64 + 0.5);
                short_kernel.push(phase.cos() * short_normalization);
            }
        }
        let slope = frame_length / 4;
        let zero = (frame_length - slope) / 2;
        let low_sine = (0..2 * slope)
            .map(|index| (std::f64::consts::PI / (2 * slope) as f64 * (index as f64 + 0.5)).sin())
            .collect::<Vec<_>>();
        let mut low_overlap_window = vec![0.0; sample_count];
        low_overlap_window[zero..zero + slope].copy_from_slice(&low_sine[..slope]);
        low_overlap_window[zero + slope..sample_count - zero - slope].fill(1.0);
        low_overlap_window[sample_count - zero - slope..sample_count - zero]
            .copy_from_slice(&low_sine[slope..]);
        Ok(Self {
            frame_length,
            previous: vec![0.0; frame_length],
            window,
            kernel,
            short_window,
            short_kernel,
            low_overlap_window,
            previous_window_shape: WindowShape::Sine,
        })
    }

    pub fn frame_length(&self) -> usize {
        self.frame_length
    }

    pub fn reset(&mut self) {
        self.previous.fill(0.0);
        self.previous_window_shape = WindowShape::Sine;
    }

    pub fn analyze(&mut self, input: &[f32]) -> Result<AacLcAnalysisFrame, AacLcEncoderError> {
        self.analyze_with_sequence(input, WindowSequence::OnlyLong)
    }

    fn analyze_aac_ld(
        &mut self,
        input: &[f32],
        current_shape: WindowShape,
    ) -> Result<AacLcAnalysisFrame, AacLcEncoderError> {
        if input.len() != self.frame_length {
            return Err(AacLcEncoderError::InputLengthMismatch {
                expected: self.frame_length,
                actual: input.len(),
            });
        }
        if input.iter().any(|sample| !sample.is_finite()) {
            return Err(AacLcEncoderError::NonFiniteInput);
        }
        let mut block = self
            .previous
            .iter()
            .chain(input)
            .map(|&sample| sample as f64)
            .collect::<Vec<_>>();
        let previous_window = match self.previous_window_shape {
            WindowShape::LowOverlap => &self.low_overlap_window,
            _ => &self.window,
        };
        let current_window = match current_shape {
            WindowShape::LowOverlap => &self.low_overlap_window,
            _ => &self.window,
        };
        for (sample, &window) in block[..self.frame_length]
            .iter_mut()
            .zip(&previous_window[..self.frame_length])
        {
            *sample *= window;
        }
        for (sample, &window) in block[self.frame_length..]
            .iter_mut()
            .zip(&current_window[self.frame_length..])
        {
            *sample *= window;
        }
        let mut spectrum = vec![0.0f32; self.frame_length];
        for (coefficient, output) in spectrum.iter_mut().enumerate() {
            let row = &self.kernel[coefficient * block.len()..(coefficient + 1) * block.len()];
            *output = block
                .iter()
                .zip(row)
                .map(|(&sample, &basis)| sample * basis)
                .sum::<f64>() as f32;
        }
        self.previous.copy_from_slice(input);
        self.previous_window_shape = current_shape;
        Ok(AacLcAnalysisFrame {
            spectrum,
            short_spectra: None,
            short_window_group_lengths: Vec::new(),
            transient_ratio: transient_ratio(input),
        })
    }

    pub fn analyze_with_sequence(
        &mut self,
        input: &[f32],
        sequence: WindowSequence,
    ) -> Result<AacLcAnalysisFrame, AacLcEncoderError> {
        if input.len() != self.frame_length {
            return Err(AacLcEncoderError::InputLengthMismatch {
                expected: self.frame_length,
                actual: input.len(),
            });
        }
        if input.iter().any(|sample| !sample.is_finite()) {
            return Err(AacLcEncoderError::NonFiniteInput);
        }
        let mut block = Vec::with_capacity(2 * self.frame_length);
        block.extend(self.previous.iter().map(|&sample| sample as f64));
        block.extend(input.iter().map(|&sample| sample as f64));
        let unwindowed = block.clone();
        let transition_window;
        let window = match sequence {
            WindowSequence::OnlyLong | WindowSequence::EightShort => &self.window,
            WindowSequence::LongStart | WindowSequence::LongStop => {
                transition_window = self.transition_window(sequence);
                &transition_window
            }
        };
        for (sample, &window) in block.iter_mut().zip(window) {
            *sample *= window;
        }
        let mut spectrum = vec![0.0f32; self.frame_length];
        for (coefficient, output) in spectrum.iter_mut().enumerate() {
            let row = &self.kernel[coefficient * block.len()..(coefficient + 1) * block.len()];
            *output = block
                .iter()
                .zip(row)
                .map(|(&sample, &basis)| sample * basis)
                .sum::<f64>() as f32;
        }
        let transient_ratio = transient_ratio(input);
        let short_spectra = (sequence == WindowSequence::EightShort)
            .then(|| self.analyze_short_windows(&unwindowed));
        let short_window_group_lengths = short_spectra
            .as_deref()
            .map(group_short_windows)
            .unwrap_or_default();
        self.previous.copy_from_slice(input);
        Ok(AacLcAnalysisFrame {
            spectrum,
            short_spectra,
            short_window_group_lengths,
            transient_ratio,
        })
    }

    fn transition_window(&self, sequence: WindowSequence) -> Vec<f64> {
        let n = self.frame_length;
        let short = n / 8;
        let flat = n / 2 - short / 2;
        let mut window = vec![0.0; 2 * n];
        match sequence {
            WindowSequence::LongStart => {
                window[..n].copy_from_slice(&self.window[..n]);
                window[n..n + flat].fill(1.0);
                window[n + flat..n + flat + short]
                    .copy_from_slice(&self.short_window[short..2 * short]);
            }
            WindowSequence::LongStop => {
                window[flat..flat + short].copy_from_slice(&self.short_window[..short]);
                window[flat + short..n].fill(1.0);
                window[n..].copy_from_slice(&self.window[n..]);
            }
            _ => window.copy_from_slice(&self.window),
        }
        window
    }

    fn analyze_short_windows(&self, block: &[f64]) -> Vec<Vec<f32>> {
        let short_length = self.frame_length / 8;
        let short_sample_count = 2 * short_length;
        let first = self.frame_length / 2 - short_length / 2;
        (0..8)
            .map(|window| {
                let start = first + window * short_length;
                let input = block[start..start + short_sample_count]
                    .iter()
                    .zip(&self.short_window)
                    .map(|(&sample, &slope)| sample * slope)
                    .collect::<Vec<_>>();
                (0..short_length)
                    .map(|coefficient| {
                        let row = &self.short_kernel[coefficient * short_sample_count
                            ..(coefficient + 1) * short_sample_count];
                        input
                            .iter()
                            .zip(row)
                            .map(|(&sample, &basis)| sample * basis)
                            .sum::<f64>() as f32
                    })
                    .collect()
            })
            .collect()
    }
}

pub fn group_short_windows(spectra: &[Vec<f32>]) -> Vec<u8> {
    if spectra.len() != 8 {
        return Vec::new();
    }
    let energies = spectra
        .iter()
        .map(|window| window.iter().map(|value| value * value).sum::<f32>())
        .collect::<Vec<_>>();
    let mut groups = vec![1u8];
    for window in 1..8 {
        let low = energies[window - 1].min(energies[window]);
        let high = energies[window - 1].max(energies[window]);
        let similar = high <= 1.0e-20 || high / low.max(1.0e-20) <= 4.0;
        if similar && groups.last().copied().unwrap_or(0) < 8 {
            *groups.last_mut().unwrap() += 1;
        } else {
            groups.push(1);
        }
    }
    groups
}

fn transient_ratio(input: &[f32]) -> f32 {
    let energies = input
        .chunks(8)
        .map(|chunk| chunk.iter().map(|sample| sample * sample).sum::<f32>())
        .collect::<Vec<_>>();
    let mean = energies.iter().sum::<f32>() / energies.len().max(1) as f32;
    if mean <= f32::EPSILON {
        0.0
    } else {
        energies.into_iter().fold(0.0f32, f32::max) / mean
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AacLcEncoderError {
    UnsupportedFrameLength(usize),
    InputLengthMismatch { expected: usize, actual: usize },
    NonFiniteInput,
    Sfb(SfbError),
    PsychoacousticLayoutMismatch,
    BitReservoirUnderflow { available: usize, requested: usize },
    InvalidRawElementLayout,
    ScalefactorDeltaOutOfRange(i16),
    Huffman(HuffmanError),
    Adts(AdtsError),
    Latm(LatmError),
    Loas(LoasError),
    Adif(AdifError),
    Sbr(SbrEncoderError),
    Ps(PsEncoderError),
    Hcr(HcrError),
    EldAnalysis(EldAnalysisError),
}

impl From<SfbError> for AacLcEncoderError {
    fn from(value: SfbError) -> Self {
        Self::Sfb(value)
    }
}

impl From<HuffmanError> for AacLcEncoderError {
    fn from(value: HuffmanError) -> Self {
        Self::Huffman(value)
    }
}

impl From<AdtsError> for AacLcEncoderError {
    fn from(value: AdtsError) -> Self {
        Self::Adts(value)
    }
}

impl From<LatmError> for AacLcEncoderError {
    fn from(value: LatmError) -> Self {
        Self::Latm(value)
    }
}

impl From<LoasError> for AacLcEncoderError {
    fn from(value: LoasError) -> Self {
        Self::Loas(value)
    }
}

impl From<AdifError> for AacLcEncoderError {
    fn from(value: AdifError) -> Self {
        Self::Adif(value)
    }
}

impl From<SbrEncoderError> for AacLcEncoderError {
    fn from(value: SbrEncoderError) -> Self {
        Self::Sbr(value)
    }
}

impl From<PsEncoderError> for AacLcEncoderError {
    fn from(value: PsEncoderError) -> Self {
        Self::Ps(value)
    }
}

impl From<HcrError> for AacLcEncoderError {
    fn from(value: HcrError) -> Self {
        Self::Hcr(value)
    }
}

impl From<EldAnalysisError> for AacLcEncoderError {
    fn from(value: EldAnalysisError) -> Self {
        Self::EldAnalysis(value)
    }
}

impl fmt::Display for AacLcEncoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedFrameLength(length) => {
                write!(f, "unsupported AAC-LC encoder frame length {length}")
            }
            Self::InputLengthMismatch { expected, actual } => {
                write!(f, "expected {expected} PCM samples, got {actual}")
            }
            Self::NonFiniteInput => write!(f, "AAC-LC encoder input contains NaN or infinity"),
            Self::Sfb(error) => error.fmt(f),
            Self::PsychoacousticLayoutMismatch => {
                write!(f, "AAC-LC encoder psychoacoustic/SFB layout mismatch")
            }
            Self::BitReservoirUnderflow {
                available,
                requested,
            } => write!(
                f,
                "AAC-LC frame needs {requested} bits but only {available} reservoir bits are available"
            ),
            Self::InvalidRawElementLayout => write!(f, "invalid AAC-LC raw element layout"),
            Self::ScalefactorDeltaOutOfRange(delta) => {
                write!(f, "AAC scalefactor delta {delta} is outside Huffman range -60..60")
            }
            Self::Huffman(error) => error.fmt(f),
            Self::Adts(error) => error.fmt(f),
            Self::Latm(error) => error.fmt(f),
            Self::Loas(error) => error.fmt(f),
            Self::Adif(error) => error.fmt(f),
            Self::Sbr(error) => error.fmt(f),
            Self::Ps(error) => error.fmt(f),
            Self::Hcr(error) => error.fmt(f),
            Self::EldAnalysis(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for AacLcEncoderError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sfb::aac_lc_sfb_info;

    #[test]
    fn drm_hcr_retry_relaxes_only_reordered_length_overflow() {
        let mut attempts = Vec::new();
        let relaxation = retry_drm_hcr(&mut |relaxation| {
            attempts.push(relaxation);
            if relaxation < 4.0 {
                Err(HcrError::ReorderedSpectralLengthOutOfRange {
                    length: 2,
                    maximum: 1,
                }
                .into())
            } else {
                Ok(vec![relaxation as u8])
            }
        })
        .unwrap();
        assert_eq!(attempts, [1.0, 2.0, 4.0]);
        assert_eq!(relaxation, [4]);

        let mut attempts = 0;
        assert!(matches!(
            retry_drm_hcr(&mut |_| {
                attempts += 1;
                Err(HcrError::ReorderedSpectralLengthOutOfRange {
                    length: 2,
                    maximum: 1,
                }
                .into())
            }),
            Err(AacLcEncoderError::Hcr(
                HcrError::ReorderedSpectralLengthOutOfRange { .. }
            ))
        ));
        assert_eq!(attempts, 13);

        assert_eq!(
            retry_drm_hcr(&mut |_| Err(AacLcEncoderError::NonFiniteInput)),
            Err(AacLcEncoderError::NonFiniteInput)
        );
    }

    #[test]
    fn converts_and_formats_every_encoder_error_variant() {
        let sfb = SfbError::UnsupportedFrameLength(1);
        let huffman = HuffmanError::InvalidCodebook(12);
        let adts = AdtsError::InvalidProfile(4);
        let latm = LatmError::MissingStreamMuxConfig;
        let loas = LoasError::InvalidSyncword(0);
        let adif = AdifError::InvalidSignature;
        let sbr = SbrEncoderError::NonFiniteInput;
        let ps = PsEncoderError::InvalidParameterCount;
        let hcr = HcrError::InvalidCodebook(0);
        assert_eq!(
            AacLcEncoderError::from(sfb.clone()),
            AacLcEncoderError::Sfb(sfb)
        );
        assert_eq!(
            AacLcEncoderError::from(huffman.clone()),
            AacLcEncoderError::Huffman(huffman)
        );
        assert_eq!(
            AacLcEncoderError::from(adts.clone()),
            AacLcEncoderError::Adts(adts)
        );
        assert_eq!(
            AacLcEncoderError::from(latm.clone()),
            AacLcEncoderError::Latm(latm)
        );
        assert_eq!(
            AacLcEncoderError::from(loas.clone()),
            AacLcEncoderError::Loas(loas)
        );
        assert_eq!(
            AacLcEncoderError::from(adif.clone()),
            AacLcEncoderError::Adif(adif)
        );
        assert_eq!(
            AacLcEncoderError::from(sbr.clone()),
            AacLcEncoderError::Sbr(sbr)
        );
        assert_eq!(
            AacLcEncoderError::from(ps.clone()),
            AacLcEncoderError::Ps(ps)
        );
        assert_eq!(
            AacLcEncoderError::from(hcr.clone()),
            AacLcEncoderError::Hcr(hcr)
        );

        let errors = [
            AacLcEncoderError::UnsupportedFrameLength(1),
            AacLcEncoderError::InputLengthMismatch {
                expected: 1024,
                actual: 1,
            },
            AacLcEncoderError::NonFiniteInput,
            AacLcEncoderError::Sfb(SfbError::UnsupportedFrameLength(1)),
            AacLcEncoderError::PsychoacousticLayoutMismatch,
            AacLcEncoderError::BitReservoirUnderflow {
                available: 1,
                requested: 2,
            },
            AacLcEncoderError::InvalidRawElementLayout,
            AacLcEncoderError::ScalefactorDeltaOutOfRange(61),
            AacLcEncoderError::Huffman(HuffmanError::InvalidCodebook(12)),
            AacLcEncoderError::Adts(AdtsError::InvalidProfile(4)),
            AacLcEncoderError::Latm(LatmError::MissingStreamMuxConfig),
            AacLcEncoderError::Loas(LoasError::InvalidSyncword(0)),
            AacLcEncoderError::Adif(AdifError::InvalidSignature),
            AacLcEncoderError::Sbr(SbrEncoderError::NonFiniteInput),
            AacLcEncoderError::Ps(PsEncoderError::InvalidParameterCount),
            AacLcEncoderError::Hcr(HcrError::InvalidCodebook(0)),
        ];
        for error in errors {
            assert!(!error.to_string().is_empty());
        }
    }

    #[test]
    fn encoder_components_reject_invalid_lengths_nonfinite_values_and_layouts() {
        assert_eq!(
            AacLcAnalysisFilterbank::new(1000).unwrap_err(),
            AacLcEncoderError::UnsupportedFrameLength(1000)
        );
        let mut analysis = AacLcAnalysisFilterbank::new(1024).unwrap();
        assert!(matches!(
            analysis.analyze(&[]),
            Err(AacLcEncoderError::InputLengthMismatch { .. })
        ));
        let mut nonfinite = vec![0.0; 1024];
        nonfinite[0] = f32::NAN;
        assert_eq!(
            analysis.analyze(&nonfinite).unwrap_err(),
            AacLcEncoderError::NonFiniteInput
        );

        let model = AacLcPsychoacousticModel::new(4).unwrap();
        assert!(matches!(
            model.analyze(&[]),
            Err(AacLcEncoderError::InputLengthMismatch { .. })
        ));
        assert!(matches!(
            model.analyze_short(&vec![vec![0.0; 128]; 7]),
            Err(AacLcEncoderError::InputLengthMismatch { .. })
        ));
        let psycho = model.analyze(&vec![0.0; 1024]).unwrap();
        let quantizer = AacLcQuantizer::new(4).unwrap();
        assert_eq!(
            quantizer.quantize_long(&[], &psycho, 1000).unwrap_err(),
            AacLcEncoderError::PsychoacousticLayoutMismatch
        );
        assert_eq!(
            quantizer
                .quantize_short(&vec![vec![0.0; 128]; 7], &[], &[8], 1000)
                .unwrap_err(),
            AacLcEncoderError::PsychoacousticLayoutMismatch
        );
        assert!(group_short_windows(&[]).is_empty());

        let mut mono = PureRustAacLcMonoEncoder::new(4, 4000, 2000).unwrap();
        assert!(matches!(
            mono.encode_raw_data_block(&[]),
            Err(AacLcEncoderError::InputLengthMismatch { .. })
        ));
        assert_eq!(
            mono.encode_raw_data_block(&nonfinite).unwrap_err(),
            AacLcEncoderError::NonFiniteInput
        );
        mono.block_switcher.sequence = WindowSequence::LongStart;
        mono.quantizer.sampling_frequency_index = 13;
        assert!(matches!(
            mono.encode_raw_data_block(&vec![0.0; 1024]),
            Err(AacLcEncoderError::Sfb(_))
        ));
        let mut stereo = PureRustAacLcStereoEncoder::new(4, 8000, 4000).unwrap();
        assert!(matches!(
            stereo.encode_raw_data_block(&[], &[]),
            Err(AacLcEncoderError::InputLengthMismatch { .. })
        ));

        let header = LdSbrHeader {
            start_frequency: 5,
            stop_frequency: 8,
            ..LdSbrHeader::default()
        };
        let mut he =
            PureRustHeAacMonoEncoder::new(6, 48_000, 16_000, 8_000, header.clone()).unwrap();
        assert!(matches!(
            he.encode_raw_data_block(&[]),
            Err(AacLcEncoderError::InputLengthMismatch { .. })
        ));
        let mut he_nonfinite = vec![0.0; 2048];
        he_nonfinite[0] = f32::INFINITY;
        assert_eq!(
            he.encode_raw_data_block(&he_nonfinite).unwrap_err(),
            AacLcEncoderError::NonFiniteInput
        );
        assert_eq!(
            he.encode_raw_data_block_with_extension(&vec![0.0; 2048], Some(&vec![0; 270]))
                .unwrap_err(),
            AacLcEncoderError::Sbr(SbrEncoderError::PayloadTooLarge(270))
        );
        he.block_switcher.sequence = WindowSequence::LongStart;
        he.quantizer.sampling_frequency_index = 13;
        assert!(matches!(
            he.encode_raw_data_block(&vec![0.0; 2048]),
            Err(AacLcEncoderError::Sfb(_))
        ));
        assert!(matches!(
            PureRustHeAacPsEncoder::new(13, 48_000, 20_000, 8_000, header.clone()),
            Err(AacLcEncoderError::Sfb(_))
        ));
        let mut ps = PureRustHeAacPsEncoder::new(6, 48_000, 20_000, 8_000, header).unwrap();
        assert!(matches!(
            ps.encode_raw_data_block(&[], &[]),
            Err(AacLcEncoderError::InputLengthMismatch { .. })
        ));
    }

    #[test]
    fn analysis_is_stateful_and_silence_stays_zero() {
        let mut analysis = AacLcAnalysisFilterbank::new(1024).unwrap();
        let silence = analysis.analyze(&vec![0.0; 1024]).unwrap();
        assert!(silence.spectrum.iter().all(|&value| value == 0.0));
        assert_eq!(silence.transient_ratio, 0.0);

        let mut impulse = vec![0.0; 1024];
        impulse[1023] = 1.0;
        let first = analysis.analyze(&impulse).unwrap();
        let overlap = analysis.analyze(&vec![0.0; 1024]).unwrap();
        assert!(first.spectrum.iter().any(|&value| value != 0.0));
        assert!(overlap.spectrum.iter().any(|&value| value != 0.0));
        assert!(first.transient_ratio > 100.0);
        analysis.reset();
        assert!(analysis
            .analyze(&vec![0.0; 1024])
            .unwrap()
            .spectrum
            .iter()
            .all(|&value| value == 0.0));
    }

    #[test]
    fn sinusoid_energy_is_concentrated_in_the_mdct_spectrum() {
        let mut analysis = AacLcAnalysisFilterbank::new(1024).unwrap();
        let input = (0..1024)
            .map(|index| (2.0 * std::f32::consts::PI * 37.5 * index as f32 / 1024.0).sin())
            .collect::<Vec<_>>();
        analysis.analyze(&input).unwrap();
        let steady = analysis.analyze(&input).unwrap();
        let total = steady
            .spectrum
            .iter()
            .map(|value| value * value)
            .sum::<f32>();
        let mut energies = steady
            .spectrum
            .iter()
            .map(|value| value * value)
            .collect::<Vec<_>>();
        energies.sort_by(|left, right| right.total_cmp(left));
        let dominant = energies.iter().take(8).sum::<f32>();
        assert!(total.is_finite() && total > 0.0);
        let concentration = dominant / total;
        assert!(concentration > 0.75, "MDCT concentration {concentration}");
    }

    #[test]
    fn psychoacoustic_model_uses_decoder_sfb_layout_and_detects_tonality() {
        let model = AacLcPsychoacousticModel::new(4).unwrap();
        let mut spectrum = vec![0.0; 1024];
        spectrum[100] = 10.0;
        let analysis = model.analyze(&spectrum).unwrap();
        assert_eq!(analysis.bands.len(), 49);
        let total_band_energy = analysis.bands.iter().map(|band| band.energy).sum::<f32>();
        assert!((total_band_energy - 100.0).abs() < 1.0e-4);
        let active = analysis
            .bands
            .iter()
            .find(|band| band.energy > 1.0)
            .unwrap();
        assert!(active.tonality > 0.9);
        assert!(active.masking_threshold > 0.0);
        assert!(active.masking_threshold < active.energy);
        assert!(analysis.perceptual_entropy > 0.0);
    }

    #[test]
    fn block_switcher_emits_legal_attack_and_release_sequence() {
        let mut switcher = AacLcBlockSwitcher::default();
        assert_eq!(switcher.update(1.0), WindowSequence::OnlyLong);
        assert_eq!(switcher.update(20.0), WindowSequence::LongStart);
        assert_eq!(switcher.update(20.0), WindowSequence::EightShort);
        assert_eq!(switcher.update(20.0), WindowSequence::EightShort);
        assert_eq!(switcher.update(1.0), WindowSequence::LongStop);
        assert_eq!(switcher.sequence(), WindowSequence::LongStop);
        assert_eq!(switcher.update(20.0), WindowSequence::LongStart);
        assert_eq!(switcher.update(1.0), WindowSequence::EightShort);
        assert_eq!(switcher.update(1.0), WindowSequence::LongStop);
        assert_eq!(switcher.update(1.0), WindowSequence::OnlyLong);
    }

    #[test]
    fn eight_short_analysis_uses_eight_windows_and_short_sfb_tables() {
        let mut filterbank = AacLcAnalysisFilterbank::new(1024).unwrap();
        let mut input = vec![0.0; 1024];
        input[100] = 1.0;
        let frame = filterbank
            .analyze_with_sequence(&input, WindowSequence::EightShort)
            .unwrap();
        let short = frame.short_spectra.as_ref().unwrap();
        assert_eq!(short.len(), 8);
        assert!(short.iter().all(|window| window.len() == 128));
        assert_eq!(
            frame
                .short_window_group_lengths
                .iter()
                .map(|&length| usize::from(length))
                .sum::<usize>(),
            8
        );
        assert!(short
            .iter()
            .flat_map(|window| window.iter())
            .any(|&value| value != 0.0));

        let psycho = AacLcPsychoacousticModel::new(4)
            .unwrap()
            .analyze_short(short)
            .unwrap();
        assert_eq!(psycho.len(), 8);
        assert!(psycho.iter().all(|window| window.bands.len() == 14));
    }

    #[test]
    fn short_window_grouping_splits_large_energy_changes() {
        let spectra = [1.0f32, 1.0, 1.0, 10.0, 10.0, 1.0, 1.0, 1.0]
            .into_iter()
            .map(|amplitude| vec![amplitude; 128])
            .collect::<Vec<_>>();
        assert_eq!(group_short_windows(&spectra), vec![3, 2, 3]);
    }

    #[test]
    fn quantizer_meets_masking_threshold_and_relaxes_for_bit_budget() {
        let mut filterbank = AacLcAnalysisFilterbank::new(1024).unwrap();
        let input = (0..1024)
            .map(|index| {
                0.6 * (2.0 * std::f32::consts::PI * 23.0 * index as f32 / 1024.0).sin()
                    + 0.2 * (2.0 * std::f32::consts::PI * 117.0 * index as f32 / 1024.0).sin()
            })
            .collect::<Vec<_>>();
        filterbank.analyze(&input).unwrap();
        let frame = filterbank.analyze(&input).unwrap();
        let model = AacLcPsychoacousticModel::new(4).unwrap();
        let psycho = model.analyze(&frame.spectrum).unwrap();
        let quantizer = AacLcQuantizer::new(4).unwrap();
        let transparent = quantizer
            .quantize_long(&frame.spectrum, &psycho, usize::MAX)
            .unwrap();
        assert_eq!(transparent.bands.len(), psycho.bands.len());
        for (quantized, band) in transparent.bands.iter().zip(&psycho.bands) {
            assert!(
                quantized.noise_energy <= band.masking_threshold * 1.0001,
                "noise {} threshold {}",
                quantized.noise_energy,
                band.masking_threshold
            );
            assert!(quantized
                .coefficients
                .iter()
                .all(|value| value.unsigned_abs() <= 8191));
        }

        let constrained = quantizer
            .quantize_long(
                &frame.spectrum,
                &psycho,
                transparent.estimated_spectral_bits / 3,
            )
            .unwrap();
        assert!(constrained.masking_relaxation > 1.0);
        assert!(constrained.estimated_spectral_bits <= transparent.estimated_spectral_bits);
        let mut reservoir = AacLcBitReservoir::new(100_000, 100_000);
        let reservoir_frame = quantizer
            .quantize_long_with_reservoir(&frame.spectrum, &psycho, 100_000, &mut reservoir)
            .unwrap();
        assert_eq!(reservoir_frame.bands.len(), transparent.bands.len());
        assert!(reservoir.fullness_bits() <= 100_000);
        let mut empty_reservoir = AacLcBitReservoir::new(0, 0);
        assert!(matches!(
            quantizer.quantize_long_with_reservoir(
                &frame.spectrum,
                &psycho,
                0,
                &mut empty_reservoir,
            ),
            Err(AacLcEncoderError::BitReservoirUnderflow { available: 0, .. })
        ));

        assert_eq!(transparent.sections.first().unwrap().start_sfb, 0);
        assert_eq!(
            transparent.sections.last().unwrap().end_sfb,
            transparent.bands.len()
        );
        assert!(transparent
            .sections
            .windows(2)
            .all(|sections| sections[0].end_sfb == sections[1].start_sfb));
        assert_eq!(
            transparent.estimated_spectral_bits,
            transparent
                .bands
                .iter()
                .map(|band| band.estimated_bits)
                .sum()
        );
        assert_eq!(
            transparent.estimated_section_bits,
            transparent
                .sections
                .iter()
                .map(|section| section.side_information_bits)
                .sum()
        );
    }

    #[test]
    fn section_selector_collapses_silent_bands_into_zero_codebook() {
        let mut bands = (0..49)
            .map(|_| quantize_band(&[0.0; 4], 1.0, false))
            .collect::<Vec<_>>();
        let (sections, spectral_bits, side_bits) = select_huffman_sections(&mut bands, false);
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].codebook, 0);
        assert_eq!(sections[0].start_sfb, 0);
        assert_eq!(sections[0].end_sfb, 49);
        assert_eq!(spectral_bits, 0);
        // 49 SFBs require a 31 escape increment followed by the remainder.
        assert_eq!(side_bits, 14);
    }

    #[test]
    #[should_panic(expected = "every quantized SFB is representable by AAC codebook 11")]
    fn section_selector_rejects_unrepresentable_internal_band() {
        let mut bands = vec![
            QuantizedSfb {
                scalefactor: 0,
                coefficients: vec![0; 4],
                noise_energy: 0.0,
                estimated_bits: 0,
                codebook: 0,
                codebook_bit_costs: [None; 12],
            };
            2
        ];

        let _ = select_huffman_sections(&mut bands, false);
    }

    #[test]
    fn bit_reservoir_saves_and_spends_bits_across_frames() {
        let mut reservoir = AacLcBitReservoir::new(1000, 600);
        assert_eq!(reservoir.allocate_frame(2000), 1000);
        reservoir.commit_frame(600).unwrap();
        assert_eq!(reservoir.fullness_bits(), 400);
        assert_eq!(reservoir.allocate_frame(2000), 1400);
        reservoir.commit_frame(1300).unwrap();
        assert_eq!(reservoir.fullness_bits(), 100);
        reservoir.commit_frame(100).unwrap();
        assert_eq!(reservoir.fullness_bits(), 600);
        assert_eq!(
            reservoir.commit_frame(1601),
            Err(AacLcEncoderError::BitReservoirUnderflow {
                available: 1600,
                requested: 1601,
            })
        );
        assert_eq!(reservoir.fullness_bits(), 600);
    }

    #[test]
    fn full_bit_reservoir_can_spend_capacity_on_the_first_frame() {
        let mut reservoir = AacLcBitReservoir::new_full(1_000, 600);
        assert_eq!(reservoir.capacity_bits(), 600);
        assert_eq!(reservoir.fullness_bits(), 600);
        assert_eq!(reservoir.available_frame_bits(), 1_600);
        reservoir.commit_frame(1_300).unwrap();
        assert_eq!(reservoir.fullness_bits(), 300);
    }

    #[test]
    fn sce_raw_data_block_writer_roundtrips_through_pure_rust_decoder() {
        use crate::decoder::AacLcDecoder;

        let spectrum = vec![0.0; 1024];
        let psycho = AacLcPsychoacousticModel::new(4)
            .unwrap()
            .analyze(&spectrum)
            .unwrap();
        let frame = AacLcQuantizer::new(4)
            .unwrap()
            .quantize_long(&spectrum, &psycho, 1000)
            .unwrap();
        let payload = frame.write_sce_raw_data_block(0).unwrap();
        let mut reader = crate::bits::BitReader::new(&payload);
        let decoded = AacLcDecoder::new(4, 1)
            .unwrap()
            .decode_raw_data_block_f32_terminated_from_reader(&mut reader)
            .unwrap();
        assert_eq!(decoded.channels(), 1);
        assert_eq!(decoded.samples_per_channel(), 1024);
        assert!(decoded
            .interleaved_f32()
            .iter()
            .all(|&sample| sample == 0.0));

        let adts = write_adts_frame(&payload, 4, 1).unwrap();
        let decoded = AacLcDecoder::new(4, 1)
            .unwrap()
            .decode_adts_frame_f32(&adts)
            .unwrap();
        assert_eq!(decoded.channels(), 1);
        assert!(decoded
            .interleaved_f32()
            .iter()
            .all(|&sample| sample == 0.0));
    }

    #[test]
    fn sce_writer_roundtrips_nonzero_section_scalefactor_and_spectrum() {
        use crate::decoder::AacLcDecoder;

        let info = aac_lc_sfb_info(4, WindowSequence::OnlyLong).unwrap();
        let mut bands = info
            .offsets
            .windows(2)
            .map(|border| quantize_band(&vec![0.0; border[1] - border[0]], 1.0, false))
            .collect::<Vec<_>>();
        bands[0].coefficients = vec![1, -1, 0, 0];
        bands[0].scalefactor = 0;
        bands[0].codebook_bit_costs = spectral_band_codebook_costs(&bands[0].coefficients);
        let (sections, estimated_spectral_bits, estimated_section_bits) =
            select_huffman_sections(&mut bands, false);
        let frame = QuantizedAacLcFrame {
            global_gain: 100,
            bands,
            estimated_spectral_bits,
            estimated_section_bits,
            sections,
            masking_relaxation: 1.0,
        };
        let payload = frame.write_sce_raw_data_block(3).unwrap();
        let mut reader = crate::bits::BitReader::new(&payload);
        let decoded = AacLcDecoder::new(4, 1)
            .unwrap()
            .decode_raw_data_block_f32_terminated_from_reader(&mut reader)
            .unwrap();
        assert!(decoded
            .interleaved_f32()
            .iter()
            .any(|&sample| sample != 0.0));
    }

    #[test]
    fn common_window_cpe_writer_roundtrips_two_nonzero_channels() {
        use crate::decoder::AacLcDecoder;

        let make_frame = |first: Vec<i32>| {
            let info = aac_lc_sfb_info(4, WindowSequence::OnlyLong).unwrap();
            let mut bands = info
                .offsets
                .windows(2)
                .map(|border| quantize_band(&vec![0.0; border[1] - border[0]], 1.0, false))
                .collect::<Vec<_>>();
            bands[0].coefficients = first;
            bands[0].scalefactor = 0;
            bands[0].codebook_bit_costs = spectral_band_codebook_costs(&bands[0].coefficients);
            let (sections, estimated_spectral_bits, estimated_section_bits) =
                select_huffman_sections(&mut bands, false);
            QuantizedAacLcFrame {
                global_gain: 100,
                bands,
                estimated_spectral_bits,
                estimated_section_bits,
                sections,
                masking_relaxation: 1.0,
            }
        };
        let left = make_frame(vec![1, -1, 0, 0]);
        let right = make_frame(vec![-1, 0, 1, 0]);
        let payload = QuantizedAacLcFrame::write_cpe_raw_data_block(&left, &right, 2).unwrap();
        let mut reader = crate::bits::BitReader::new(&payload);
        let decoded = AacLcDecoder::new(4, 2)
            .unwrap()
            .decode_raw_data_block_f32_terminated_from_reader(&mut reader)
            .unwrap();
        assert_eq!(decoded.channels(), 2);
        assert_eq!(decoded.samples_per_channel(), 1024);
        let pcm = decoded.interleaved_f32();
        assert!(pcm.chunks_exact(2).any(|sample| sample[0] != 0.0));
        assert!(pcm.chunks_exact(2).any(|sample| sample[1] != 0.0));
    }

    #[test]
    fn eight_short_quantization_and_sce_writer_roundtrip() {
        use crate::decoder::AacLcDecoder;

        let mut spectra = vec![vec![0.0; 128]; 8];
        spectra[2][8] = 4.0;
        spectra[3][9] = -3.0;
        let psycho = AacLcPsychoacousticModel::new(4)
            .unwrap()
            .analyze_short(&spectra)
            .unwrap();
        let frame = AacLcQuantizer::new(4)
            .unwrap()
            .quantize_short(&spectra, &psycho, &[2, 3, 3], usize::MAX)
            .unwrap();
        assert_eq!(frame.groups.len(), 3);
        assert_eq!(frame.groups[0].bands.len(), 14);
        let payload = frame.write_sce_raw_data_block(1).unwrap();
        let mut reader = crate::bits::BitReader::new(&payload);
        let decoded = AacLcDecoder::new(4, 1)
            .unwrap()
            .decode_raw_data_block_f32_terminated_from_reader(&mut reader)
            .unwrap();
        assert_eq!(decoded.channels(), 1);
        assert_eq!(decoded.samples_per_channel(), 1024);
        assert!(decoded
            .interleaved_f32()
            .iter()
            .any(|&sample| sample != 0.0));

        let cpe = QuantizedAacLcShortFrame::write_cpe_raw_data_block(&frame, &frame, 1).unwrap();
        let mut reader = crate::bits::BitReader::new(&cpe);
        let decoded = AacLcDecoder::new(4, 2)
            .unwrap()
            .decode_raw_data_block_f32_terminated_from_reader(&mut reader)
            .unwrap();
        assert_eq!(decoded.channels(), 2);
        assert!(decoded
            .interleaved_f32()
            .chunks_exact(2)
            .any(|sample| sample[0] != 0.0 && sample[1] != 0.0));

        assert!(!frame
            .write_sce_raw_data_block_with_sbr_fill(1, &[0])
            .unwrap()
            .is_empty());
        assert_eq!(
            frame.write_sce_raw_data_block_with_sbr_fill(1, &[]),
            Err(AacLcEncoderError::InvalidRawElementLayout)
        );
        let mut empty = frame.clone();
        empty.groups.clear();
        assert_eq!(
            empty.write_sce_raw_data_block(0),
            Err(AacLcEncoderError::InvalidRawElementLayout)
        );
        let mut ragged = frame.clone();
        ragged.groups[1].bands.pop();
        assert_eq!(
            ragged.write_sce_raw_data_block(0),
            Err(AacLcEncoderError::InvalidRawElementLayout)
        );
        assert_eq!(
            ragged.write_sce_raw_data_block_with_sbr_fill(0, &[0]),
            Err(AacLcEncoderError::InvalidRawElementLayout)
        );

        let constrained = AacLcQuantizer::new(4)
            .unwrap()
            .quantize_short(&spectra, &psycho, &[2, 3, 3], 0)
            .unwrap();
        assert!(constrained.masking_relaxation > 1.0);
    }

    #[test]
    fn raw_writers_reject_each_malformed_layout_and_fill_form() {
        let silent_band = || QuantizedSfb {
            scalefactor: 0,
            coefficients: vec![0; 4],
            noise_energy: 0.0,
            estimated_bits: 0,
            codebook: 0,
            codebook_bit_costs: zero_codebook_costs(),
        };
        let long = QuantizedAacLcFrame {
            global_gain: 100,
            bands: vec![],
            estimated_spectral_bits: 0,
            estimated_section_bits: 0,
            sections: vec![],
            masking_relaxation: 1.0,
        };
        assert_eq!(
            long.write_sce_raw_data_block_with_sequence(0, WindowSequence::EightShort),
            Err(AacLcEncoderError::InvalidRawElementLayout)
        );
        assert_eq!(
            long.write_sce_raw_data_block_with_sequence_and_fill(
                16,
                WindowSequence::OnlyLong,
                None,
            ),
            Err(AacLcEncoderError::InvalidRawElementLayout)
        );
        assert_eq!(
            long.write_sce_raw_data_block_with_sbr_fill(0, &[0x10]),
            Err(AacLcEncoderError::InvalidRawElementLayout)
        );
        assert_eq!(
            long.write_sce_raw_data_block_with_sbr_fill(0, &[0xf0]),
            Err(AacLcEncoderError::InvalidRawElementLayout)
        );

        let mut too_many_bands = long.clone();
        too_many_bands.bands = (0..64).map(|_| silent_band()).collect();
        assert_eq!(
            too_many_bands.write_sce_raw_data_block(0),
            Err(AacLcEncoderError::InvalidRawElementLayout)
        );
        let mut unequal = long.clone();
        unequal.bands.push(silent_band());
        assert_eq!(
            QuantizedAacLcFrame::write_cpe_raw_data_block(&long, &unequal, 0),
            Err(AacLcEncoderError::InvalidRawElementLayout)
        );

        let short = QuantizedAacLcShortFrame {
            global_gain: 100,
            group_lengths: vec![8],
            groups: vec![long.clone()],
            estimated_spectral_bits: 0,
            estimated_section_bits: 0,
            masking_relaxation: 1.0,
        };
        let mut malformed_short = short.clone();
        malformed_short.group_lengths = vec![7];
        assert_eq!(
            malformed_short.write_sce_raw_data_block_with_sbr_fill(0, &[0]),
            Err(AacLcEncoderError::InvalidRawElementLayout)
        );
        let empty_short = QuantizedAacLcShortFrame {
            groups: vec![],
            group_lengths: vec![],
            ..short.clone()
        };
        assert_eq!(
            QuantizedAacLcShortFrame::write_cpe_raw_data_block(&empty_short, &empty_short, 0),
            Err(AacLcEncoderError::InvalidRawElementLayout)
        );
        assert_eq!(
            QuantizedAacLcShortFrame::write_cpe_raw_data_block(&short, &malformed_short, 0),
            Err(AacLcEncoderError::InvalidRawElementLayout)
        );

        let mut bad_scale = long.clone();
        let mut band = silent_band();
        band.codebook = 1;
        band.scalefactor = 61;
        bad_scale.bands.push(band);
        assert_eq!(
            bad_scale.write_sce_raw_data_block(0),
            Err(AacLcEncoderError::ScalefactorDeltaOutOfRange(61))
        );
        assert_eq!(
            write_drm_mono_packet(&long, 4, &[0, 1]),
            Err(AacLcEncoderError::InvalidRawElementLayout)
        );
        assert!(write_adts_frame(&[], 15, 1).is_err());

        let zero_drm = QuantizedAacLcFrame {
            bands: vec![silent_band()],
            ..long.clone()
        };
        assert!(!write_drm_mono_packet(&zero_drm, 4, &[0, 4])
            .unwrap()
            .is_empty());
        let (sections, side, payload) = prepare_drm_hcr(&zero_drm, &[0, 4]).unwrap();
        assert_eq!(sections, vec![(0, 1, 0)]);
        assert_eq!(side.reordered_spectral_bits, 0);
        assert!(payload.is_empty());
        let mut escape_band = silent_band();
        escape_band.codebook = 11;
        escape_band.coefficients = vec![17, -17];
        let escape_drm = QuantizedAacLcFrame {
            bands: vec![escape_band],
            ..long
        };
        assert!(!write_drm_mono_packet(&escape_drm, 4, &[0, 2])
            .unwrap()
            .is_empty());
        let (sections, side, _) = prepare_drm_hcr(&escape_drm, &[0, 2]).unwrap();
        let mut writer = BitWriter::new();
        write_drm_channel_side(&mut writer, &escape_drm, &sections, side).unwrap();

        let excessive = HcrSideInfo {
            reordered_spectral_bits: SCE_MAX_REORDERED_SPECTRAL_BITS + 1,
            longest_codeword_bits: 1,
            bits_read: 0,
        };
        assert_eq!(
            validate_hcr_length(&excessive, SCE_MAX_REORDERED_SPECTRAL_BITS),
            Err(AacLcEncoderError::Hcr(
                HcrError::ReorderedSpectralLengthOutOfRange {
                    length: SCE_MAX_REORDERED_SPECTRAL_BITS + 1,
                    maximum: SCE_MAX_REORDERED_SPECTRAL_BITS,
                }
            ))
        );
    }

    #[test]
    fn encoder_internal_fallbacks_cover_extreme_quantization_and_windows() {
        let fallback = quantize_band(&[f32::NAN], 0.0, false);
        assert_eq!(fallback.scalefactor, -120);
        assert!(fallback.noise_energy.is_infinite());
        assert_eq!(fallback.coefficients.len(), 1);
        assert!(spectral_band_codebook_costs(&[1])
            .iter()
            .any(Option::is_none));

        let filterbank = AacLcAnalysisFilterbank::new(1024).unwrap();
        assert_eq!(
            filterbank.transition_window(WindowSequence::OnlyLong),
            filterbank.window
        );
    }

    #[test]
    fn stateful_mono_encoder_emits_decodable_adts_across_block_switches() {
        use crate::decoder::AacLcDecoder;
        use crate::transport::PureRustTransportDecoder;

        let mut encoder = PureRustAacLcMonoEncoder::new(4, 32_000, 16_000).unwrap();
        let mut decoder = AacLcDecoder::new(4, 1).unwrap();
        let silence = vec![0.0; 1024];
        let mut attack = vec![0.0; 1024];
        attack[511] = 8.0;
        for input in [&silence, &attack, &silence, &silence, &silence] {
            let adts = encoder.encode_adts_frame(input).unwrap();
            let decoded = decoder.decode_adts_frame_f32(&adts).unwrap();
            assert_eq!(decoded.channels(), 1);
            assert_eq!(decoded.samples_per_channel(), 1024);
            assert!(decoded
                .interleaved_f32()
                .iter()
                .all(|sample| sample.is_finite()));
        }
        assert!(encoder.bit_reservoir().fullness_bits() > 0);

        let mut loas_encoder = PureRustAacLcMonoEncoder::new(4, 32_000, 16_000).unwrap();
        let loas_first = loas_encoder.encode_loas_frame(&silence).unwrap();
        let loas_second = loas_encoder.encode_loas_frame(&silence).unwrap();
        let mut loas_decoder = PureRustTransportDecoder::from_loas_frame(&loas_first).unwrap();
        assert_eq!(
            loas_decoder
                .decode_loas_interleaved_f32(&loas_first)
                .unwrap()
                .len(),
            1024
        );
        assert_eq!(
            loas_decoder
                .decode_loas_interleaved_f32(&loas_second)
                .unwrap()
                .len(),
            1024
        );

        let mut adif_encoder = PureRustAacLcMonoEncoder::new(4, 3000, 2000).unwrap();
        let first = adif_encoder.encode_adif_access_unit(&silence).unwrap();
        let second = adif_encoder.encode_adif_access_unit(&silence).unwrap();
        assert!(first.starts_with(b"ADIF"));
        assert!(!second.starts_with(b"ADIF"));
        let mut adif_decoder = PureRustTransportDecoder::from_adif_bytes(&first).unwrap();
        assert_eq!(
            adif_decoder
                .decode_adif_interleaved_f32(&first)
                .unwrap()
                .len(),
            1024
        );
        assert_eq!(
            adif_decoder
                .decode_adif_interleaved_f32(&second)
                .unwrap()
                .len(),
            1024
        );
    }

    #[test]
    fn sce_writer_inserts_valid_sbr_fill_before_id_end() {
        use crate::asc::LdSbrHeader;
        use crate::decoder::AacLcDecoder;
        use crate::sbr_encoder::SbrEncoderAnalysis;

        let spectrum = vec![0.0; 1024];
        let psycho = AacLcPsychoacousticModel::new(4)
            .unwrap()
            .analyze(&spectrum)
            .unwrap();
        let core = AacLcQuantizer::new(4)
            .unwrap()
            .quantize_long(&spectrum, &psycho, 2000)
            .unwrap();
        let header = LdSbrHeader {
            amp_resolution: true,
            start_frequency: 5,
            stop_frequency: 8,
            crossover_band: 0,
            ..LdSbrHeader::default()
        };
        let mut analysis = SbrEncoderAnalysis::new(&header, 48_000).unwrap();
        let fullband = vec![0.0; 2048];
        let frame = analysis.analyze(&fullband).unwrap();
        let fill = frame
            .write_mono_fill_element(&header, analysis.frequency_tables(), true)
            .unwrap();
        let raw = core
            .write_sce_raw_data_block_with_sbr_fill(0, &fill)
            .unwrap();
        let mut reader = crate::bits::BitReader::new(&raw);
        let decoded = AacLcDecoder::new(4, 1)
            .unwrap()
            .decode_raw_data_block_f32_terminated_from_reader(&mut reader)
            .unwrap();
        assert_eq!(decoded.samples_per_channel(), 1024);
    }

    #[test]
    fn halfband_downsampler_rejects_alias_band_and_he_aac_facade_is_decodable() {
        use crate::asc::{AudioSpecificConfig, AudioSpecificConfigExtension};
        use crate::decoder::AacLcDecoder;

        let mut lowpass = HalfbandDownsampler::new();
        let alternating = (0..2048)
            .map(|index| if index & 1 == 0 { 1.0 } else { -1.0 })
            .collect::<Vec<_>>();
        let rejected = lowpass.process(&alternating);
        let tail_rms = (rejected[128..]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            / (rejected.len() - 128) as f32)
            .sqrt();
        assert!(tail_rms < 0.02, "halfband alias RMS {tail_rms}");

        let header = LdSbrHeader {
            amp_resolution: true,
            start_frequency: 5,
            stop_frequency: 8,
            crossover_band: 0,
            ..LdSbrHeader::default()
        };
        let mut encoder = PureRustHeAacMonoEncoder::new(6, 48_000, 16_000, 8_000, header).unwrap();
        let mut decoder = AacLcDecoder::new(6, 1).unwrap();
        let silence = vec![0.0; 2048];
        let mut attack = silence.clone();
        attack[1024] = 8.0;
        for input in [silence.clone(), alternating, attack, silence] {
            let raw = encoder.encode_raw_data_block(&input).unwrap();
            let mut reader = crate::bits::BitReader::new(&raw);
            let decoded = decoder
                .decode_raw_data_block_f32_terminated_from_reader(&mut reader)
                .unwrap();
            assert_eq!(decoded.samples_per_channel(), 1024);
            assert!(decoded
                .interleaved_f32()
                .iter()
                .all(|sample| sample.is_finite()));
        }

        let mut asc = AudioSpecificConfig::aac_lc(24_000, 1).unwrap();
        asc.extension = Some(AudioSpecificConfigExtension {
            audio_object_type: 5,
            sampling_frequency_index: 3,
            sampling_frequency: 48_000,
            ps_present: false,
        });
        let header = LdSbrHeader {
            amp_resolution: true,
            start_frequency: 5,
            stop_frequency: 8,
            crossover_band: 0,
            ..LdSbrHeader::default()
        };
        let mut encoder = PureRustHeAacMonoEncoder::new(6, 48_000, 16_000, 8_000, header).unwrap();
        let raw = encoder.encode_raw_data_block(&vec![0.0; 2048]).unwrap();
        let decoded = AacLcDecoder::from_audio_specific_config(&asc)
            .unwrap()
            .decode_raw_data_block_f32(&raw)
            .unwrap();
        assert_eq!(decoded.samples_per_channel(), 2048);

        asc.extension.as_mut().unwrap().ps_present = true;
        let header = LdSbrHeader {
            amp_resolution: true,
            start_frequency: 5,
            stop_frequency: 8,
            crossover_band: 0,
            ..LdSbrHeader::default()
        };
        let mut encoder = PureRustHeAacPsEncoder::new(6, 48_000, 20_000, 8_000, header).unwrap();
        let left = (0..2048)
            .map(|index| (2.0 * std::f32::consts::PI * 31.0 * index as f32 / 2048.0).sin())
            .collect::<Vec<_>>();
        let right = left.iter().map(|sample| sample * 0.5).collect::<Vec<_>>();
        let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
        for _ in 0..2 {
            let raw = encoder.encode_raw_data_block(&left, &right).unwrap();
            let decoded = decoder.decode_raw_data_block_f32(&raw).unwrap();
            assert_eq!(decoded.channels(), 2);
            assert_eq!(decoded.samples_per_channel(), 2048);
            assert!(decoded
                .interleaved_f32()
                .iter()
                .all(|sample| sample.is_finite()));
        }
    }

    #[test]
    fn drm_er_aac_hcr_encoder_roundtrips_crc_and_nonzero_spectrum() {
        use crate::decoder::AacLcDecoder;
        use crate::drm::check_drm_crc_bits;

        let mut encoder = PureRustDrmAacMonoEncoder::new(3).unwrap();
        let input = (0..960)
            .map(|index| (2.0 * std::f32::consts::PI * 19.0 * index as f32 / 960.0).sin())
            .collect::<Vec<_>>();
        let packet = encoder.encode_packet(&input).unwrap();
        let (&crc, payload) = packet.split_first().unwrap();
        let mut decoder = AacLcDecoder::new_drm_aac(3, 1).unwrap();
        let (samples, protected_bits, _) = decoder.decode_drm_aac_mono_f32(payload).unwrap();
        check_drm_crc_bits(crc, payload, 0, protected_bits).unwrap();
        assert_eq!(samples.len(), 960);
        assert!(samples.iter().all(|sample| sample.is_finite()));
        assert!(samples.iter().any(|&sample| sample != 0.0));

        let silence = encoder.encode_packet(&vec![0.0; 960]).unwrap();
        let decoded = decoder.decode_drm_aac_mono_f32(&silence[1..]).unwrap();
        check_drm_crc_bits(silence[0], &silence[1..], 0, decoded.1).unwrap();

        let mut stereo_encoder = PureRustDrmAacStereoEncoder::new(3).unwrap();
        let right = input.iter().map(|sample| sample * -0.5).collect::<Vec<_>>();
        let packet = stereo_encoder.encode_packet(&input, &right).unwrap();
        let mut stereo_decoder = AacLcDecoder::new_drm_aac(3, 2).unwrap();
        let decoded = stereo_decoder
            .decode_drm_aac_stereo_f32(&packet[1..])
            .unwrap();
        check_drm_crc_bits(packet[0], &packet[1..], 0, decoded.1).unwrap();
        assert!(decoded.0[0].iter().any(|&sample| sample != 0.0));
        assert!(decoded.0[1].iter().any(|&sample| sample != 0.0));

        let mut state = 1u32;
        let broadband = (0..960)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state as i32 as f32) / i32::MAX as f32
            })
            .collect::<Vec<_>>();
        let mono_packet = PureRustDrmAacMonoEncoder::new(3)
            .unwrap()
            .encode_packet(&broadband)
            .unwrap();
        assert!(!mono_packet.is_empty());
        let stereo_packet = PureRustDrmAacStereoEncoder::new(3)
            .unwrap()
            .encode_packet(&broadband, &broadband)
            .unwrap();
        assert!(!stereo_packet.is_empty());
    }

    #[test]
    fn stateful_stereo_encoder_keeps_common_window_across_switches() {
        use crate::decoder::AacLcDecoder;

        let mut encoder = PureRustAacLcStereoEncoder::new(4, 40_000, 16_000).unwrap();
        let mut decoder = AacLcDecoder::new(4, 2).unwrap();
        let silence = vec![0.0; 1024];
        let mut left_attack = silence.clone();
        left_attack[400] = 8.0;
        let mut right_attack = silence.clone();
        right_attack[600] = -4.0;
        for (left, right) in [
            (&silence, &silence),
            (&left_attack, &right_attack),
            (&silence, &silence),
            (&silence, &silence),
            (&silence, &silence),
        ] {
            let adts = encoder.encode_adts_frame(left, right).unwrap();
            let decoded = decoder.decode_adts_frame_f32(&adts).unwrap();
            assert_eq!(decoded.channels(), 2);
            assert_eq!(decoded.samples_per_channel(), 1024);
        }

        let mut identical = PureRustAacLcStereoEncoder::new(4, 40_000, 16_000).unwrap();
        identical.encode_raw_data_block(&silence, &silence).unwrap();
        identical
            .encode_raw_data_block(&left_attack, &left_attack)
            .unwrap();
        let payload = identical.encode_raw_data_block(&silence, &silence).unwrap();
        let decoded = decoder
            .decode_raw_data_block_multichannel_f32(&payload)
            .unwrap();
        assert_eq!(decoded.channels(), 2);
        assert_eq!(decoded.samples_per_channel(), 1024);
    }

    #[test]
    fn er_aac_ld_spectral_encoder_writes_480_and_512_access_units() {
        use crate::decoder::AacLcDecoder;

        for frame_length in [480, 512] {
            let mut encoder =
                PureRustAacLdMonoEncoder::new(4, frame_length, 12_000, 4_000).unwrap();
            assert_eq!(encoder.frame_length(), frame_length);
            let asc = encoder.audio_specific_config();
            assert_eq!(asc.audio_object_type, 23);
            assert_eq!(
                asc.ga_specific.unwrap().frame_length_flag,
                frame_length == 480
            );
            let mut spectrum = vec![0.0; frame_length];
            spectrum[0] = 10_000.0;
            spectrum[1] = -5_000.0;
            let access_unit = encoder.encode_spectrum(&spectrum).unwrap();
            let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
            let pcm = decoder
                .decode_raw_data_block_fixed_interleaved_i16(&access_unit)
                .unwrap_or_else(|error| panic!("AAC-LD stereo {frame_length}: {error:?}"));
            assert_eq!(pcm.len(), frame_length);
            assert!(pcm.iter().any(|&sample| sample != 0));

            let mut stereo =
                PureRustAacLdStereoEncoder::new(4, frame_length, 24_000, 8_000).unwrap();
            let access_unit = stereo
                .encode_spectra(
                    &spectrum,
                    &spectrum.iter().map(|value| -*value).collect::<Vec<_>>(),
                )
                .unwrap();
            let mut decoder =
                AacLcDecoder::from_audio_specific_config(&stereo.audio_specific_config()).unwrap();
            let pcm = decoder
                .decode_raw_data_block_fixed_interleaved_i16(&access_unit)
                .unwrap_or_else(|error| panic!("AAC-LD stereo {frame_length}: {error:?}"));
            assert_eq!(pcm.len(), 2 * frame_length);
            assert!(pcm.iter().any(|&sample| sample != 0));

            let input = (0..frame_length)
                .map(|sample| {
                    (2.0 * std::f32::consts::PI * 997.0 * sample as f32 / 44_100.0).sin() * 12_000.0
                })
                .collect::<Vec<_>>();
            let mut encoder =
                PureRustAacLdMonoEncoder::new(4, frame_length, 12_000, 4_000).unwrap();
            let access_unit = encoder.encode_pcm(&input).unwrap();
            let mut decoder =
                AacLcDecoder::from_audio_specific_config(&encoder.audio_specific_config()).unwrap();
            let pcm = decoder
                .decode_raw_data_block_fixed_interleaved_i16(&access_unit)
                .unwrap();
            assert_eq!(pcm.len(), frame_length);
            assert!(pcm.iter().any(|&sample| sample != 0));
        }

        assert!(matches!(
            PureRustAacLdMonoEncoder::new(4, 1024, 12_000, 4_000),
            Err(AacLcEncoderError::UnsupportedFrameLength(1024))
        ));
    }

    #[test]
    fn aac_lc_afterburner_refines_inverse_quantized_band_distortion() {
        let spectrum = (0..1024)
            .map(|line| {
                let x = line as f32;
                0.31 * (x * 0.173).sin() + 0.17 * (x * 0.071).cos()
            })
            .collect::<Vec<_>>();
        let psycho = AacLcPsychoacousticModel::new(4)
            .unwrap()
            .analyze(&spectrum)
            .unwrap();
        let off = AacLcQuantizer::new(4)
            .unwrap()
            .quantize_long(&spectrum, &psycho, usize::MAX)
            .unwrap();
        let mut refined_quantizer = AacLcQuantizer::new(4).unwrap();
        refined_quantizer.set_afterburner(true);
        let on = refined_quantizer
            .quantize_long(&spectrum, &psycho, usize::MAX)
            .unwrap();

        let off_noise = off.bands.iter().map(|band| band.noise_energy).sum::<f32>();
        let on_noise = on.bands.iter().map(|band| band.noise_energy).sum::<f32>();
        assert!(
            on_noise < off_noise,
            "afterburner noise {on_noise} >= {off_noise}"
        );
        assert!(on
            .bands
            .iter()
            .zip(&psycho.bands)
            .all(|(band, psycho)| band.noise_energy <= psycho.masking_threshold / 1.25 * 1.0001));
        assert_ne!(
            off.bands
                .iter()
                .map(|band| (&band.coefficients, band.scalefactor))
                .collect::<Vec<_>>(),
            on.bands
                .iter()
                .map(|band| (&band.coefficients, band.scalefactor))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn low_delay_threshold_search_uses_the_tightest_bit_budget_bracket() {
        let (relaxation, raw) = fit_low_delay_access_unit(800, |relaxation| {
            let bytes = (180.0 / relaxation).ceil() as usize;
            Ok((relaxation, vec![0; bytes]))
        })
        .unwrap();
        assert!(raw.len() * 8 <= 800);
        assert!((1.79..1.82).contains(&relaxation), "{relaxation}");
        assert!(relaxation < 2.0);
    }

    #[test]
    fn low_delay_bits_to_pe_uses_fdk_tables_interpolation_and_compensation() {
        assert!((low_delay_bits_to_pe_factor(48_000, 1, 48_000, false, false) - 1.2).abs() < 1e-6);
        assert!((low_delay_bits_to_pe_factor(48_000, 1, 48_000, true, false) - 0.8).abs() < 1e-6);
        assert!((low_delay_bits_to_pe_factor(40_000, 2, 48_000, false, false) - 1.1).abs() < 1e-6);
        assert!((low_delay_bits_to_pe_factor(48_000, 2, 44_100, true, false) - 0.8).abs() < 1e-6);
        assert!((low_delay_bits_to_pe_factor(400_000, 1, 48_000, false, false) - 3.0).abs() < 1e-6);
        assert!((low_delay_bits_to_pe_factor(20_000, 1, 48_000, false, true) - 1.7).abs() < 1e-6);
        assert!((low_delay_bits_to_pe_factor(8_000, 1, 8_000, false, false) - 1.18).abs() < 1e-6);
        assert!((low_delay_bits_to_pe_factor(48_000, 3, 48_000, false, false) - 1.18).abs() < 1e-6);
    }

    #[test]
    fn low_delay_bit_reservoir_spends_and_saves_with_fdk_long_window_curve() {
        let grant = |reservoir: &AacLcBitReservoir, pe| {
            low_delay_granted_dynamic_bits(
                reservoir,
                pe,
                1.0,
                0,
                &mut LowDelayPeCorrection::default(),
            )
        };
        let mut reservoir = AacLcBitReservoir::new_full(1_000, 1_000);
        assert_eq!(grant(&reservoir, 0.0), 1_050);
        assert_eq!(grant(&reservoir, 10_000.0), 1_400);

        reservoir.fullness_bits = 500;
        assert_eq!(grant(&reservoir, 0.0), 840);
        assert_eq!(grant(&reservoir, 10_000.0), 1_100);

        reservoir.fullness_bits = 0;
        assert_eq!(grant(&reservoir, 0.0), 700);
        assert_eq!(grant(&reservoir, 10_000.0), 700);
    }

    #[test]
    fn low_delay_distribution_excludes_static_bits_and_tracks_pe_range() {
        let reservoir = AacLcBitReservoir::new_full(1_000, 1_000);
        let mut state = LowDelayPeCorrection::default();
        let dynamic = low_delay_granted_dynamic_bits(&reservoir, 0.0, 1.0, 200, &mut state);
        assert_eq!(dynamic, 840);
        assert!((state.pe_min.unwrap() - 550.4).abs() < 1e-3);
        assert!((state.pe_max.unwrap() - 915.2).abs() < 1e-3);

        let previous_min = state.pe_min.unwrap();
        let _ = low_delay_granted_dynamic_bits(&reservoir, 2_000.0, 1.0, 200, &mut state);
        assert!(state.pe_min.unwrap() > previous_min);
        assert!(state.pe_max.unwrap() >= 2_000.0);
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn low_delay_pe_min_max_state_matches_c_adjust_threshold() {
        let current = [0, 400, 800, 1_000, 1_600, 900, 100, 750, 3_000];
        let mut c_minimum = [0i32; 9];
        let mut c_maximum = [0i32; 9];
        assert_eq!(
            unsafe {
                fdk_aac_sys::fdk_adjust_pe_min_max_test(
                    current.as_ptr(),
                    current.len() as i32,
                    800,
                    c_minimum.as_mut_ptr(),
                    c_maximum.as_mut_ptr(),
                )
            },
            0
        );

        let mut rust_minimum = 800.0 * 0.8;
        let mut rust_maximum = 800.0 * 1.2;
        for (frame, &pe) in current.iter().enumerate() {
            adjust_low_delay_pe_min_max(pe as f32, &mut rust_minimum, &mut rust_maximum);
            assert!(
                (rust_minimum - c_minimum[frame] as f32).abs() <= 2.0,
                "frame {frame}: minimum Rust={rust_minimum}, C={}",
                c_minimum[frame]
            );
            assert!(
                (rust_maximum - c_maximum[frame] as f32).abs() <= 2.0,
                "frame {frame}: maximum Rust={rust_maximum}, C={}",
                c_maximum[frame]
            );
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn low_delay_reservoir_factor_matches_c_long_window_state() {
        let current = [0, 400, 800, 1_000, 1_600, 900, 100, 750, 3_000];
        let reservoir = [0, 100, 250, 500, 750, 1_000, 900, 300, 650];
        let mut c_factor_q16 = [0i32; 9];
        assert_eq!(
            unsafe {
                fdk_aac_sys::fdk_bitres_factor_test(
                    current.as_ptr(),
                    reservoir.as_ptr(),
                    current.len() as i32,
                    800,
                    1_000,
                    c_factor_q16.as_mut_ptr(),
                )
            },
            0
        );

        let mut state = LowDelayPeCorrection::default();
        for frame in 0..current.len() {
            let fill = reservoir[frame] as f32 / 1_000.0;
            let rust_factor = state
                .reservoir_bit_factor(current[frame] as f32, 800.0, fill)
                .min(0.7 + reservoir[frame] as f32 / 800.0);
            let c_factor = c_factor_q16[frame] as f32 / 65_536.0;
            assert!(
                (rust_factor - c_factor).abs() <= 3.0e-3,
                "frame {frame}: Rust={rust_factor}, C={c_factor}"
            );
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn low_delay_per_band_cbr_reduction_matches_c() {
        fn to_ld64(value: f64) -> i32 {
            ((value.log2() / 64.0) * 2_147_483_648.0).round() as i32
        }
        fn from_ld64(value: i32) -> f32 {
            2.0f64.powf((value as f64 / 2_147_483_648.0) * 64.0) as f32
        }

        let energy = [0.8f32, 0.5, 0.25, 0.125, 0.0625, 0.02];
        let threshold = [0.001f32, 0.02, 0.02, 0.1, 0.0625, 0.00001];
        let minimum_snr = [0.20f32, 0.10, 0.40, 0.50, 0.25, 0.05];
        // NO_AH, AH_INACTIVE, and AH_ACTIVE all need distinct coverage.
        let avoid_hole = [0u8, 1, 1, 1, 2, 0];
        let energy_ld = energy.map(|value| to_ld64(value as f64));
        let threshold_ld = threshold.map(|value| to_ld64(value as f64));
        let minimum_snr_ld = minimum_snr.map(|value| to_ld64(value as f64));
        let reduction = 0.05f32;
        let reduction_exponent = -3;
        let reduction_mantissa = ((reduction as f64 * 2.0f64.powi(-reduction_exponent))
            * 2_147_483_648.0)
            .round() as i32;
        let mut c_threshold = [0i32; 6];
        let mut c_avoid_hole = [0u8; 6];
        assert_eq!(
            unsafe {
                fdk_aac_sys::fdk_reduce_thresholds_cbr_test(
                    energy_ld.as_ptr(),
                    threshold_ld.as_ptr(),
                    minimum_snr_ld.as_ptr(),
                    avoid_hole.as_ptr(),
                    energy.len() as i32,
                    reduction_mantissa,
                    reduction_exponent,
                    c_threshold.as_mut_ptr(),
                    c_avoid_hole.as_mut_ptr(),
                )
            },
            0
        );

        for band in 0..energy.len() {
            let (rust_threshold, rust_avoid_hole) = reduce_low_delay_cbr_band(
                energy[band],
                threshold[band],
                minimum_snr[band],
                reduction,
                avoid_hole[band],
            );
            let reference = from_ld64(c_threshold[band]);
            let relative_error = (rust_threshold - reference).abs() / reference.max(1.0e-20);
            assert!(
                relative_error <= 2.0e-4,
                "band {band}: Rust={rust_threshold}, C={reference}, error={relative_error}"
            );
            assert_eq!(
                rust_avoid_hole, c_avoid_hole[band],
                "avoid-hole state differs in band {band}"
            );
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn low_delay_long_window_vbr_reduction_matches_c() {
        fn q31(value: f64) -> i32 {
            (value * 2_147_483_648.0).round() as i32
        }
        fn to_ld64(value: f64) -> i32 {
            ((value.log2() / 64.0) * 2_147_483_648.0).round() as i32
        }
        fn from_ld64(value: i32) -> f32 {
            2.0f64.powf((value as f64 / 2_147_483_648.0) * 64.0) as f32
        }

        let energy = [0.30f32, 0.10, 0.05];
        let threshold = [0.001f32, 0.002, 0.01];
        let minimum_snr = [0.20f32, 0.10, 0.40];
        let form_factor = [0.50f32, 0.25, 0.15];
        let offsets = [0i32, 4, 8, 12];
        let avoid_hole = [1u8; 3];
        let energy_q31 = energy.map(|value| q31(value as f64));
        let energy_ld = energy.map(|value| to_ld64(value as f64));
        let threshold_ld = threshold.map(|value| to_ld64(value as f64));
        let minimum_snr_ld = minimum_snr.map(|value| to_ld64(value as f64));
        // C stores the form factor after FORM_FAC_SHIFT=4; the Rust analysis
        // keeps the unscaled sum of square roots.
        let form_factor_ld = form_factor.map(|value| to_ld64(value as f64 / 16.0));
        let quality = 0.176f32;
        let old_chaos = 0.3f32;
        let mut c_chaos = q31(old_chaos as f64);
        let mut c_instant_chaos = 0;
        let mut c_threshold = [0i32; 3];
        let mut c_avoid_hole = [0u8; 3];
        assert_eq!(
            unsafe {
                fdk_aac_sys::fdk_reduce_thresholds_vbr_test(
                    energy_q31.as_ptr(),
                    energy_ld.as_ptr(),
                    threshold_ld.as_ptr(),
                    minimum_snr_ld.as_ptr(),
                    form_factor_ld.as_ptr(),
                    offsets.as_ptr(),
                    avoid_hole.as_ptr(),
                    energy.len() as i32,
                    q31(quality as f64),
                    &mut c_chaos,
                    &mut c_instant_chaos,
                    c_threshold.as_mut_ptr(),
                    c_avoid_hole.as_mut_ptr(),
                )
            },
            0
        );

        let analysis = PsychoacousticAnalysis {
            bands: (0..energy.len())
                .map(|band| PsychoacousticBand {
                    energy: energy[band],
                    masking_threshold: threshold[band],
                    tonality: 0.0,
                    form_factor: form_factor[band],
                })
                .collect(),
            perceptual_entropy: 0.0,
        };
        let rust_offsets = offsets.map(|value| value as usize);
        let (instant_chaos, frame_energy) = low_delay_form_factor_chaos(&analysis, &rust_offsets);
        let instant_chaos = instant_chaos.min(1.0);
        let rust_chaos = instant_chaos.min(0.25 * instant_chaos + 0.75 * old_chaos);
        let c_chaos = c_chaos as f32 / 2_147_483_648.0;
        assert!(
            (rust_chaos - c_chaos).abs() <= 2.0e-3,
            "instant Rust={instant_chaos}, C={}, smoothed Rust={rust_chaos}, C={c_chaos}, energy={frame_energy}",
            c_instant_chaos as f32 / 2_147_483_648.0
        );
        let shaped_chaos = (0.2 + (0.7 / 0.3) * (rust_chaos - 0.2)).clamp(0.1, 1.0);
        let reduction = quality * shaped_chaos * frame_energy.powf(0.25);
        for band in 0..energy.len() {
            let (rust_threshold, rust_avoid_hole) = reduce_low_delay_vbr_band(
                energy[band],
                threshold[band],
                minimum_snr[band],
                reduction,
                avoid_hole[band],
            );
            let reference = from_ld64(c_threshold[band]);
            let relative_error = (rust_threshold - reference).abs() / reference;
            assert!(
                relative_error <= 6.0e-3,
                "band {band}: Rust={rust_threshold}, C={reference}, error={relative_error}"
            );
            assert_eq!(rust_avoid_hole, c_avoid_hole[band]);
        }
    }

    #[test]
    fn low_delay_vbr_quality_uses_fdk_threshold_curve_and_stateful_chaos() {
        let offsets = [0, 4, 8];
        let base = PsychoacousticAnalysis {
            bands: vec![
                PsychoacousticBand {
                    energy: 0.3,
                    masking_threshold: 0.001,
                    tonality: 0.2,
                    form_factor: 0.5,
                },
                PsychoacousticBand {
                    energy: 0.1,
                    masking_threshold: 0.002,
                    tonality: 0.8,
                    form_factor: 0.25,
                },
            ],
            perceptual_entropy: 0.0,
        };
        let mut medium = base.clone();
        let mut highest = base;
        let mut medium_history = 0.3;
        let mut highest_history = 0.3;
        apply_low_delay_vbr_thresholds(
            &mut [&mut medium],
            &offsets,
            56_000,
            48_000,
            512,
            24_000,
            low_delay_vbr_quality_factor(3).unwrap(),
            &mut medium_history,
        );
        apply_low_delay_vbr_thresholds(
            &mut [&mut highest],
            &offsets,
            56_000,
            48_000,
            512,
            24_000,
            low_delay_vbr_quality_factor(5).unwrap(),
            &mut highest_history,
        );
        // Avoid-hole limiting may make two modes converge to the same
        // threshold. With avoid-hole disabled, the larger mode-3 quality
        // factor must still produce the larger common-root reduction.
        let mode3 = reduce_low_delay_vbr_band(0.3, 0.001, 0.2, 0.10, 0).0;
        let mode5 = reduce_low_delay_vbr_band(0.3, 0.001, 0.2, 0.04, 0).0;
        assert!(mode3 > mode5);
        assert!(medium.perceptual_entropy.is_finite());
        assert!(highest.perceptual_entropy.is_finite());
        assert!(medium_history != 0.3 && (0.0..=1.0).contains(&medium_history));
        assert_eq!(low_delay_vbr_quality_factor(0), None);
        assert_eq!(low_delay_vbr_quality_factor(7), None);
    }

    #[test]
    fn form_factor_chaos_counts_flat_and_concentrated_active_lines() {
        let offsets = [0, 4];
        let flat = analyze_psychoacoustic_bands(&[1.0; 4], &offsets);
        let concentrated = analyze_psychoacoustic_bands(&[2.0, 0.0, 0.0, 0.0], &offsets);
        let (flat_chaos, flat_energy) = low_delay_form_factor_chaos(&flat, &offsets);
        let (concentrated_chaos, concentrated_energy) =
            low_delay_form_factor_chaos(&concentrated, &offsets);
        assert!((flat_energy - 4.0).abs() < 1e-6);
        assert!((concentrated_energy - 4.0).abs() < 1e-6);
        assert!((flat_chaos - 4.0).abs() < 1e-6);
        assert!((concentrated_chaos - 2.0f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn low_delay_pe_correction_tracks_previous_dynamic_bit_error() {
        let mut correction = LowDelayPeCorrection::default();
        assert_eq!(correction.corrected_grant(1_000.0, 1_000.0, 1.0), 1_000.0);
        correction.commit_dynamic_bits(850);
        let raised = correction.corrected_grant(1_000.0, 1_000.0, 1.0);
        assert!(raised > 1_000.0 && raised <= 1_150.0, "{raised}");

        correction.commit_dynamic_bits(1_200);
        let lowered = correction.corrected_grant(1_000.0, 1_000.0, 1.0);
        assert!(lowered < 1_000.0 && lowered >= 850.0, "{lowered}");

        correction.commit_dynamic_bits(1_000);
        assert_eq!(correction.corrected_grant(2_000.0, 1_000.0, 1.0), 1_000.0);
    }

    #[test]
    fn low_delay_cbr_bandwise_thresholds_converge_to_granted_pe() {
        let offsets = [0, 8, 16, 32];
        let mut analysis = PsychoacousticAnalysis {
            bands: vec![
                PsychoacousticBand {
                    energy: 100.0,
                    masking_threshold: 0.01,
                    tonality: 0.5,
                    form_factor: 8.0,
                },
                PsychoacousticBand {
                    energy: 40.0,
                    masking_threshold: 0.01,
                    tonality: 0.5,
                    form_factor: 6.0,
                },
                PsychoacousticBand {
                    energy: 10.0,
                    masking_threshold: 0.005,
                    tonality: 0.5,
                    form_factor: 4.0,
                },
            ],
            perceptual_entropy: 0.0,
        };
        analysis.perceptual_entropy = analysis
            .bands
            .iter()
            .zip(offsets.windows(2))
            .map(|(band, border)| {
                (border[1] - border[0]) as f32 * (band.energy / band.masking_threshold).log2()
            })
            .sum();
        let target = analysis.perceptual_entropy * 0.75;
        apply_low_delay_cbr_thresholds(
            &mut [&mut analysis],
            &offsets,
            target,
            64_000,
            48_000,
            512,
            24_000,
        );
        assert!(analysis.perceptual_entropy <= target * 1.01);
        assert!(analysis.perceptual_entropy >= target * 0.95);
        assert!(analysis
            .bands
            .iter()
            .all(|band| band.masking_threshold <= band.energy * 0.8));
    }

    #[test]
    fn low_delay_cbr_fallback_relaxes_one_db_then_opens_high_band_holes() {
        let offsets = (0..=20).collect::<Vec<_>>();
        let mut analysis = PsychoacousticAnalysis {
            bands: (0..20)
                .map(|index| {
                    let energy = 100.0 / ((index + 1) * (index + 1)) as f32;
                    PsychoacousticBand {
                        energy,
                        masking_threshold: energy * 1.0e-5,
                        tonality: 0.5,
                        form_factor: energy.powf(0.25),
                    }
                })
                .collect(),
            perceptual_entropy: 20.0 * 1.0e5f32.log2(),
        };
        apply_low_delay_cbr_thresholds(
            &mut [&mut analysis],
            &offsets,
            0.0,
            64_000,
            48_000,
            512,
            24_000,
        );
        assert!(analysis.bands[..15]
            .iter()
            .all(|band| band.masking_threshold < band.energy));
        assert!(analysis.bands[15..]
            .iter()
            .any(|band| band.masking_threshold == band.energy));
        assert!(analysis.bands[..15]
            .iter()
            .all(|band| { band.masking_threshold >= band.energy * 10.0f32.powf(-0.1) * 0.999 }));
    }

    #[test]
    fn aac_ld_transient_selects_signalled_low_overlap_and_roundtrips() {
        use crate::bits::BitReader;
        use crate::decoder::AacLcDecoder;

        let mut encoder = PureRustAacLdMonoEncoder::new(4, 512, 12_000, 4_000).unwrap();
        encoder.encode_pcm(&vec![0.0; 512]).unwrap();
        let mut impulse = vec![0.0; 512];
        impulse[128] = 32_000.0;
        let access_unit = encoder.encode_pcm(&impulse).unwrap();
        assert_eq!(encoder.window_shape, WindowShape::LowOverlap);

        let mut reader = BitReader::new(&access_unit);
        reader.read(4 + 8).unwrap();
        let ics = IcsInfo::parse_aac_ld(&mut reader, 40).unwrap();
        assert_eq!(ics.window_sequence, WindowSequence::OnlyLong);
        assert_eq!(ics.window_shape, WindowShape::LowOverlap);

        let mut decoder =
            AacLcDecoder::from_audio_specific_config(&encoder.audio_specific_config()).unwrap();
        let pcm = decoder
            .decode_raw_data_block_fixed_interleaved_i16(&access_unit)
            .unwrap();
        assert_eq!(pcm.len(), 512);
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn aac_ld_block_switch_decisions_match_c_state_machine() {
        let frame_length = 512;
        let mut frames = vec![vec![0i16; frame_length]; 9];
        for frame in [1usize, 2, 7, 8] {
            for (index, sample) in frames[frame].iter_mut().enumerate() {
                *sample = ((2.0 * std::f64::consts::PI * 997.0 * index as f64 / 44_100.0).sin()
                    * if frame < 7 { 1_000.0 } else { 12_000.0 }) as i16;
            }
        }
        frames[3][32] = 32_000;
        frames[5][511] = 32_000;
        let input = frames.iter().flatten().copied().collect::<Vec<_>>();
        let mut c = vec![0u8; frames.len()];
        assert_eq!(
            unsafe {
                crate::sys::fdk_ld_block_switch_test(
                    input.as_ptr(),
                    frames.len() as i32,
                    frame_length as i32,
                    c.as_mut_ptr(),
                )
            },
            0
        );
        let mut rust_switcher = AacLdBlockSwitcher::default();
        let rust = frames
            .iter()
            .map(|frame| {
                u8::from(
                    rust_switcher.detect(&frame.iter().map(|&v| f32::from(v)).collect::<Vec<_>>()),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(rust, c);
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn low_delay_min_snr_matches_fdk_psychoacoustic_configuration() {
        let mut count = 0;
        let mut active = 0;
        let mut offsets = [0i32; 65];
        let mut min_snr = [0i32; 64];
        let mut mask_low = [0i32; 64];
        let mut mask_high = [0i32; 64];
        assert_eq!(
            unsafe {
                fdk_aac_sys::fdk_psy_configuration_test(
                    64_000,
                    48_000,
                    20_000,
                    512,
                    1,
                    &mut count,
                    &mut active,
                    offsets.as_mut_ptr(),
                    min_snr.as_mut_ptr(),
                    mask_low.as_mut_ptr(),
                    mask_high.as_mut_ptr(),
                )
            },
            0
        );
        let ics = er_long_ics(3, 512).unwrap();
        let rust_offsets = aac_band_offsets_for_ics(3, &ics, 512).unwrap().offsets;
        assert_eq!(
            &offsets[..=count as usize],
            rust_offsets
                .iter()
                .map(|&value| value as i32)
                .collect::<Vec<_>>()
        );
        let rust = low_delay_min_snr(64_000, 48_000, 512, rust_offsets, active as usize);
        for (band, (&actual, &reference)) in
            rust.iter().zip(&min_snr[..active as usize]).enumerate()
        {
            let reference = reference as f64 / 2_147_483_648.0;
            assert!(
                (f64::from(actual) - reference).abs() <= 1.5e-2,
                "band {band}: Rust={actual}, FDK={reference}"
            );
        }
    }

    #[test]
    fn er_aac_eld_pcm_encoder_writes_mono_and_stereo_access_units() {
        use crate::decoder::AacLcDecoder;

        for frame_length in [480, 512] {
            let input = (0..frame_length)
                .map(|sample| {
                    (2.0 * std::f32::consts::PI * 997.0 * sample as f32 / 44_100.0).sin() * 12_000.0
                })
                .collect::<Vec<_>>();
            let mut mono = PureRustAacEldMonoEncoder::new(4, frame_length, 12_000, 4_000).unwrap();
            let asc = mono.audio_specific_config();
            assert_eq!(asc.audio_object_type, 39);
            let access_unit = mono.encode_pcm(&input).unwrap();
            let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
            let pcm = decoder
                .decode_raw_data_block_fixed_interleaved_i16(&access_unit)
                .unwrap();
            assert_eq!(pcm.len(), frame_length);
            assert!(pcm.iter().any(|sample| *sample != 0));

            let mut stereo =
                PureRustAacEldStereoEncoder::new(4, frame_length, 24_000, 8_000).unwrap();
            let right = input.iter().map(|sample| -*sample).collect::<Vec<_>>();
            let access_unit = stereo.encode_pcm(&input, &right).unwrap();
            let mut decoder =
                AacLcDecoder::from_audio_specific_config(&stereo.audio_specific_config()).unwrap();
            let pcm = decoder
                .decode_raw_data_block_fixed_interleaved_i16(&access_unit)
                .unwrap();
            assert_eq!(pcm.len(), 2 * frame_length);
            assert!(pcm.iter().any(|sample| *sample != 0));
        }
    }

    #[test]
    fn aac_eld_mono_sbr_is_written_between_core_and_er_extensions() {
        use crate::decoder::AacLcDecoder;

        let base = LdSbrHeader {
            amp_resolution: true,
            start_frequency: 5,
            crossover_band: 0,
            ..LdSbrHeader::default()
        };
        let single_header = (0..16)
            .find_map(|stop_frequency| {
                let header = LdSbrHeader {
                    stop_frequency,
                    ..base.clone()
                };
                let tables = LdSbrFrequencyTables::from_header(&header, 48_000).ok()?;
                (tables.high.last().copied()? <= 32).then_some(header)
            })
            .unwrap();

        for (dual_rate, crc, header) in [
            (false, false, single_header),
            (true, false, base.clone()),
            (true, true, base),
        ] {
            let mut encoder = PureRustAacEldMonoEncoder::new(6, 512, 12_000, 4_000).unwrap();
            encoder.enable_sbr_with_crc(header, dual_rate, crc).unwrap();
            let asc = encoder.audio_specific_config();
            let eld = asc.eld_specific.as_ref().unwrap();
            assert!(eld.sbr_present);
            assert_eq!(eld.sbr_sampling_rate, dual_rate);
            assert_eq!(eld.sbr_crc, crc);
            assert_eq!(eld.sbr_headers.len(), 1);
            let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
            let input_length = 512 * if dual_rate { 2 } else { 1 };
            let input_rate = if dual_rate { 48_000.0 } else { 24_000.0 };
            let input = (0..input_length)
                .map(|sample| {
                    (2.0 * std::f32::consts::PI * 997.0 * sample as f32 / input_rate).sin()
                        * 12_000.0
                })
                .collect::<Vec<_>>();
            let access_unit = encoder.encode_pcm(&input).unwrap();
            let pcm = decoder
                .decode_raw_data_block_fixed_interleaved_i16(&access_unit)
                .unwrap();
            assert_eq!(pcm.len(), input_length);
            assert!(pcm.iter().any(|&sample| sample != 0));
        }

        let header = LdSbrHeader {
            amp_resolution: true,
            start_frequency: 5,
            stop_frequency: 8,
            ..LdSbrHeader::default()
        };
        let mut encoder = PureRustAacEldStereoEncoder::new(6, 512, 24_000, 8_000).unwrap();
        encoder.enable_sbr_with_crc(header, true, true).unwrap();
        let asc = encoder.audio_specific_config();
        let mut decoder = AacLcDecoder::from_audio_specific_config(&asc).unwrap();
        let left = (0..1024)
            .map(|sample| {
                (2.0 * std::f32::consts::PI * 997.0 * sample as f32 / 48_000.0).sin() * 12_000.0
            })
            .collect::<Vec<_>>();
        let right = (0..1024)
            .map(|sample| {
                (2.0 * std::f32::consts::PI * 1499.0 * sample as f32 / 48_000.0).sin() * 8_000.0
            })
            .collect::<Vec<_>>();
        let raw = encoder.encode_pcm(&left, &right).unwrap();
        let pcm = decoder
            .decode_raw_data_block_fixed_interleaved_i16(&raw)
            .unwrap();
        assert_eq!(pcm.len(), 2048);
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn fdk_decodes_pure_rust_er_aac_eld_pcm_access_units() {
        use crate::{Decoder, TransportType};

        let mut encoder = PureRustAacEldMonoEncoder::new(4, 512, 12_000, 4_000).unwrap();
        let mut config = encoder.audio_specific_config().to_bytes().unwrap();
        let mut decoder = Decoder::open(TransportType::Raw).unwrap();
        decoder.configure_raw(&mut config).unwrap();
        let mut observed_nonzero = false;
        for frame in 0..8 {
            let input = (0..512)
                .map(|sample| {
                    let sample = frame * 512 + sample;
                    (2.0 * std::f32::consts::PI * 997.0 * sample as f32 / 44_100.0).sin() * 12_000.0
                })
                .collect::<Vec<_>>();
            let access_unit = encoder.encode_pcm(&input).unwrap();
            let mut pcm = vec![0i16; 1024];
            let samples = decoder
                .decode_access_unit_i16(&access_unit, &mut pcm)
                .unwrap();
            assert_eq!(samples, 512);
            observed_nonzero |= pcm[..samples].iter().any(|sample| *sample != 0);
        }
        assert!(observed_nonzero);

        let mut encoder = PureRustAacEldStereoEncoder::new(4, 512, 24_000, 8_000).unwrap();
        let mut config = encoder.audio_specific_config().to_bytes().unwrap();
        let mut decoder = Decoder::open(TransportType::Raw).unwrap();
        decoder.configure_raw(&mut config).unwrap();
        let left = (0..512)
            .map(|sample| {
                (2.0 * std::f32::consts::PI * 997.0 * sample as f32 / 44_100.0).sin() * 12_000.0
            })
            .collect::<Vec<_>>();
        let right = left.iter().map(|sample| -*sample).collect::<Vec<_>>();
        let access_unit = encoder.encode_pcm(&left, &right).unwrap();
        let mut pcm = vec![0i16; 2048];
        let samples = decoder
            .decode_access_unit_i16(&access_unit, &mut pcm)
            .unwrap();
        assert_eq!(samples, 1024);
        assert!(pcm[..samples].iter().any(|sample| *sample != 0));
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn fdk_decodes_eld_mono_core_at_mps_bandwidth() {
        use crate::{Decoder, TransportType};

        let mut encoder = PureRustAacEldMonoEncoder::new(3, 512, 683, 4_000).unwrap();
        encoder.set_bandwidth(15_150);
        let mut config = encoder.audio_specific_config().to_bytes().unwrap();
        let mut decoder = Decoder::open(TransportType::Raw).unwrap();
        decoder.configure_raw(&mut config).unwrap();
        let input = (0..512)
            .map(|sample| {
                let phase = sample as f32 * 0.083;
                (phase.sin() * 12_000.0 + (phase + 0.5).sin() * 4_000.0)
                    * std::f32::consts::FRAC_1_SQRT_2
            })
            .collect::<Vec<_>>();
        let access_unit = encoder.encode_pcm(&input).unwrap();
        let mut pcm = vec![0i16; 1_024];
        assert_eq!(
            decoder
                .decode_access_unit_i16(&access_unit, &mut pcm)
                .unwrap(),
            512
        );
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn fdk_decodes_pure_rust_er_aac_ld_pcm_with_matching_waveform() {
        use crate::{Decoder, TransportType};

        let mut encoder = PureRustAacLdMonoEncoder::new(4, 512, 12_000, 4_000).unwrap();
        let mut config = encoder.audio_specific_config().to_bytes().unwrap();
        let mut decoder = Decoder::open(TransportType::Raw).unwrap();
        decoder.configure_raw(&mut config).unwrap();
        let mut reference = Vec::new();
        let mut decoded = Vec::new();
        for frame in 0..12 {
            let input = (0..512)
                .map(|sample| {
                    let sample = frame * 512 + sample;
                    (2.0 * std::f32::consts::PI * 997.0 * sample as f32 / 44_100.0).sin() * 12_000.0
                })
                .collect::<Vec<_>>();
            let access_unit = encoder.encode_pcm(&input).unwrap();
            let mut pcm = vec![0i16; 1024];
            let samples = decoder
                .decode_access_unit_i16(&access_unit, &mut pcm)
                .unwrap();
            assert_eq!(samples, 512);
            reference.extend(input.iter().map(|sample| f64::from(*sample)));
            decoded.extend(pcm[..samples].iter().map(|sample| f64::from(*sample)));
        }
        let mut best = 0.0f64;
        for lag in -1024isize..=1024 {
            let reference_start = lag.unsigned_abs() * usize::from(lag < 0);
            let decoded_start = lag.unsigned_abs() * usize::from(lag > 0);
            let count = (reference.len() - reference_start).min(decoded.len() - decoded_start);
            let reference = &reference[reference_start..reference_start + count];
            let decoded = &decoded[decoded_start..decoded_start + count];
            let dot = reference
                .iter()
                .zip(decoded)
                .map(|(&left, &right)| left * right)
                .sum::<f64>();
            let reference_energy = reference.iter().map(|value| value * value).sum::<f64>();
            let decoded_energy = decoded.iter().map(|value| value * value).sum::<f64>();
            if reference_energy > 0.0 && decoded_energy > 0.0 {
                best = best.max(dot.abs() / (reference_energy * decoded_energy).sqrt());
            }
        }
        assert!(best > 0.95, "AAC-LD PCM waveform correlation {best}");
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn c_and_rust_decode_transient_aac_ld_low_overlap_with_matching_shape() {
        use crate::decoder::AacLcDecoder;
        use crate::{Decoder, TransportType};

        let mut encoder = PureRustAacLdMonoEncoder::new(4, 512, 12_000, 4_000).unwrap();
        let asc = encoder.audio_specific_config();
        let mut config = asc.to_bytes().unwrap();
        let mut c = Decoder::open(TransportType::Raw).unwrap();
        c.configure_raw(&mut config).unwrap();
        let mut rust = AacLcDecoder::from_audio_specific_config(&asc).unwrap();

        let silence = encoder.encode_pcm(&vec![0.0; 512]).unwrap();
        let mut c_pcm = vec![0i16; 1024];
        assert_eq!(c.decode_access_unit_i16(&silence, &mut c_pcm).unwrap(), 512);
        rust.decode_raw_data_block_interleaved_f32(&silence)
            .unwrap();

        let mut transient = (0..512)
            .map(|index| {
                (2.0 * std::f32::consts::PI * 997.0 * index as f32 / 48_000.0).sin() * 12_000.0
            })
            .collect::<Vec<_>>();
        transient[128] = 24_000.0;
        transient[129] = -24_000.0;
        let access_unit = encoder.encode_pcm(&transient).unwrap();
        assert_eq!(encoder.window_shape, WindowShape::LowOverlap);
        assert_eq!(
            c.decode_access_unit_i16(&access_unit, &mut c_pcm).unwrap(),
            512
        );
        let mut c_output = c_pcm[..512].to_vec();
        let mut rust_output = rust
            .decode_raw_data_block_interleaved_f32(&access_unit)
            .unwrap()
            .into_iter()
            .map(|sample| sample.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16)
            .collect::<Vec<_>>();
        encoder.window_shape = WindowShape::Sine;
        for _ in 0..4 {
            let tail = encoder.encode_spectrum(&vec![0.0; 512]).unwrap();
            assert_eq!(c.decode_access_unit_i16(&tail, &mut c_pcm).unwrap(), 512);
            c_output.extend_from_slice(&c_pcm[..512]);
            rust_output.extend(
                rust.decode_raw_data_block_interleaved_f32(&tail)
                    .unwrap()
                    .into_iter()
                    .map(|sample| sample.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16),
            );
        }
        let mut best = (0.0f64, 0isize, 0.0f64, 0.0f64);
        for lag in -512isize..=512 {
            let c_start = lag.unsigned_abs() * usize::from(lag < 0);
            let rust_start = lag.unsigned_abs() * usize::from(lag > 0);
            let count = (c_output.len() - c_start).min(rust_output.len() - rust_start);
            let c_slice = &c_output[c_start..c_start + count];
            let rust_slice = &rust_output[rust_start..rust_start + count];
            let dot = c_slice
                .iter()
                .zip(rust_slice)
                .map(|(&left, &right)| f64::from(left) * f64::from(right))
                .sum::<f64>();
            let c_energy = c_slice
                .iter()
                .map(|&sample| f64::from(sample).powi(2))
                .sum::<f64>();
            let rust_energy = rust_slice
                .iter()
                .map(|&sample| f64::from(sample).powi(2))
                .sum::<f64>();
            if c_energy > 0.0 && rust_energy > 0.0 {
                let correlation = dot.abs() / (c_energy * rust_energy).sqrt();
                if correlation > best.0 {
                    best = (correlation, lag, c_energy, rust_energy);
                }
            }
        }
        let (correlation, lag, c_energy, rust_energy) = best;
        assert!(
            c_energy > 0.0 && rust_energy > 0.0,
            "low-overlap energies C={c_energy}, Rust={rust_energy}"
        );
        assert!(
            correlation > 0.99,
            "AAC-LD low-overlap correlation {correlation} at lag {lag}, RMS Rust/C={}",
            (rust_energy / c_energy).sqrt()
        );
        let rms_ratio = (rust_energy / c_energy).sqrt();
        assert!(
            (0.98..=1.02).contains(&rms_ratio),
            "AAC-LD low-overlap RMS ratio Rust/C={rms_ratio}"
        );
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn pure_and_fdk_aac_lc_encoders_are_bidirectionally_interoperable() {
        use crate::asc::AudioSpecificConfig;
        use crate::decoder::AacLcDecoder;
        use crate::{AudioObjectType, ChannelMode, Decoder, Encoder, EncoderConfig, TransportType};

        let mut config = EncoderConfig::aac_lc_stereo(44_100, 64_000);
        config.channels = 1;
        config.channel_mode = ChannelMode::Mono;
        config.audio_object_type = AudioObjectType::AacLc;
        config.transport = TransportType::Raw;
        let mut fdk_encoder = Encoder::configured(&config).unwrap();
        let info = fdk_encoder.info().unwrap();
        assert_eq!(info.frame_length, 1024);
        let mut encoded = vec![0u8; info.max_output_bytes as usize];
        let input_i16 = (0..1024)
            .map(|index| {
                ((2.0 * std::f64::consts::PI * 997.0 * index as f64 / 44_100.0).sin() * 12_000.0)
                    as i16
            })
            .collect::<Vec<_>>();
        let mut pure_decoder = AacLcDecoder::new(4, 1).unwrap();
        let mut decoded_fdk_units = 0;
        let mut observed_nonzero = false;
        for _ in 0..12 {
            let bytes = fdk_encoder
                .encode_interleaved_i16(&input_i16, &mut encoded)
                .unwrap();
            if bytes != 0 {
                let decoded = pure_decoder
                    .decode_raw_data_block_interleaved_f32(&encoded[..bytes])
                    .unwrap();
                assert_eq!(decoded.len(), 1024);
                observed_nonzero |= decoded.iter().any(|&sample| sample != 0.0);
                decoded_fdk_units += 1;
            }
        }
        assert!(decoded_fdk_units >= 8);
        assert!(observed_nonzero);

        let input_f32 = input_i16
            .iter()
            .map(|&sample| f32::from(sample))
            .collect::<Vec<_>>();
        let mut pure_encoder = PureRustAacLcMonoEncoder::new(4, 4000, 2000).unwrap();
        let mut asc = AudioSpecificConfig::aac_lc(44_100, 1)
            .unwrap()
            .to_bytes()
            .unwrap();
        let mut fdk_decoder = Decoder::open(TransportType::Raw).unwrap();
        fdk_decoder.configure_raw(&mut asc).unwrap();
        let mut pcm = vec![0i16; 2048];
        let mut observed_nonzero = false;
        for _ in 0..4 {
            let access_unit = pure_encoder.encode_raw_data_block(&input_f32).unwrap();
            let samples = fdk_decoder
                .decode_access_unit_i16(&access_unit, &mut pcm)
                .unwrap();
            assert_eq!(samples, 1024);
            observed_nonzero |= pcm[..samples].iter().any(|&sample| sample != 0);
        }
        assert!(observed_nonzero);
    }

    #[test]
    fn spectral_bandwidth_uses_the_fdk_mdct_line_formula() {
        let mut spectrum = vec![1.0; 512];
        apply_spectral_bandwidth(&mut spectrum, 48_000, 6_000);
        assert!(spectrum[..128].iter().all(|&value| value == 1.0));
        assert!(spectrum[128..].iter().all(|&value| value == 0.0));

        apply_spectral_bandwidth(&mut spectrum, 48_000, 24_000);
        assert!(spectrum[128..].iter().all(|&value| value == 0.0));
    }

    #[test]
    fn aac_ld_spectral_entry_point_applies_configured_bandwidth() {
        let mut filtered = PureRustAacLdMonoEncoder::new(3, 512, 12_000, 4_000).unwrap();
        filtered.set_bandwidth(6_000);
        let mut high_only = vec![0.0; 512];
        high_only[300] = 20_000.0;

        let mut silence = PureRustAacLdMonoEncoder::new(3, 512, 12_000, 4_000).unwrap();
        silence.set_bandwidth(6_000);
        assert_eq!(
            filtered.encode_spectrum(&high_only).unwrap(),
            silence.encode_spectrum(&vec![0.0; 512]).unwrap()
        );
    }

    #[test]
    fn multichannel_dynamic_bits_follow_fdk_element_weights() {
        assert_eq!(
            multichannel_channel_bit_targets(3, 10_000, 0).unwrap(),
            [4_000, 3_000, 3_000]
        );
        assert_eq!(
            multichannel_channel_bit_targets(6, 10_000, 0).unwrap(),
            [2_400, 1_750, 1_750, 1_750, 1_750, 600]
        );
        assert_eq!(
            multichannel_channel_bit_targets(11, 10_000, 0).unwrap(),
            [2_000, 1_375, 1_375, 1_375, 1_375, 2_000, 500]
        );
        assert_eq!(
            multichannel_channel_bit_targets(14, 10_000, 0).unwrap(),
            [1_800, 1_300, 1_300, 1_300, 1_300, 400, 1_300, 1_300]
        );
    }
}
