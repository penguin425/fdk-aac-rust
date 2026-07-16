//! Pure Rust ADTS transport header support.
//!
//! This module is independent from the C/C++ FDK implementation.  It is the
//! first small transport-layer component migrated to safe Rust and can be used
//! to split or create ADTS-framed AAC streams before handing raw access units to
//! a decoder/encoder.

use std::fmt;

use crate::asc::{AudioSpecificConfig, GaSpecificConfig};

const SYNCWORD: u16 = 0x0fff;
const HEADER_LEN_WITHOUT_CRC: usize = 7;
const HEADER_LEN_WITH_CRC: usize = 9;
const MAX_FRAME_LENGTH: usize = 0x1fff;

/// FDK's ADTS CRC primitive: MSB-first CRC-16 with polynomial `0x8005` and
/// initial value `0xffff`.
///
/// FDK applies this primitive to syntax-specific CRC regions, not necessarily a
/// complete raw data block. It is exposed here for transport parsers that track
/// those regions while consuming AAC syntax.
pub fn adts_crc16(bytes: &[u8]) -> u16 {
    let mut crc = 0xffffu16;
    update_adts_crc16(&mut crc, bytes);
    crc
}

/// Calculate one ADTS CRC over multiple syntax regions without resetting the
/// CRC state between regions.
pub fn adts_crc16_regions<'a>(regions: impl IntoIterator<Item = &'a [u8]>) -> u16 {
    let mut crc = 0xffffu16;
    for region in regions {
        update_adts_crc16(&mut crc, region);
    }
    crc
}

/// Calculate the ADTS CRC over arbitrary MSB-first bit ranges. Regions are
/// accumulated in order, matching FDK's registered syntax regions.
pub fn adts_crc16_bit_regions<'a>(
    regions: impl IntoIterator<Item = (&'a [u8], std::ops::Range<usize>)>,
) -> Result<u16, AdtsError> {
    let mut crc = 0xffffu16;
    for (bytes, range) in regions {
        if range.start > range.end || range.end > bytes.len() * 8 {
            return Err(AdtsError::InvalidCrcBitRange {
                start: range.start,
                end: range.end,
                available_bits: bytes.len() * 8,
            });
        }
        for bit_index in range {
            let input_bit = (bytes[bit_index / 8] >> (7 - bit_index % 8)) & 1;
            let feedback = input_bit ^ u8::from((crc & 0x8000) != 0);
            crc <<= 1;
            if feedback != 0 {
                crc ^= 0x8005;
            }
        }
    }
    Ok(crc)
}

/// Calculate ADTS CRC regions with FDK's `mBits > 0` semantics. Each region is
/// extended with zero bits to `padded_bits` when the syntax region ends early.
pub fn adts_crc16_padded_bit_regions<'a>(
    regions: impl IntoIterator<Item = (&'a [u8], std::ops::Range<usize>, usize)>,
) -> Result<u16, AdtsError> {
    let mut crc = 0xffffu16;
    for (bytes, range, padded_bits) in regions {
        if range.start > range.end || range.end > bytes.len() * 8 {
            return Err(AdtsError::InvalidCrcBitRange {
                start: range.start,
                end: range.end,
                available_bits: bytes.len() * 8,
            });
        }
        let region_bits = range.len();
        if padded_bits < region_bits {
            return Err(AdtsError::InvalidCrcPadding {
                region_bits,
                padded_bits,
            });
        }
        for bit_index in range {
            update_adts_crc16_bit(&mut crc, (bytes[bit_index / 8] >> (7 - bit_index % 8)) & 1);
        }
        for _ in region_bits..padded_bits {
            update_adts_crc16_bit(&mut crc, 0);
        }
    }
    Ok(crc)
}

fn update_adts_crc16_bit(crc: &mut u16, input_bit: u8) {
    let feedback = input_bit ^ u8::from((*crc & 0x8000) != 0);
    *crc <<= 1;
    if feedback != 0 {
        *crc ^= 0x8005;
    }
}

fn update_adts_crc16(crc: &mut u16, bytes: &[u8]) {
    for &byte in bytes {
        for bit in (0..8).rev() {
            let feedback = ((byte >> bit) & 1) ^ u8::from((*crc & 0x8000) != 0);
            *crc <<= 1;
            if feedback != 0 {
                *crc ^= 0x8005;
            }
        }
    }
}

const SAMPLE_RATES: [u32; 16] = [
    96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025, 8_000,
    7_350, 0, 0, 0,
];

/// MPEG version bit stored in an ADTS header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MpegVersion {
    /// MPEG-4 AAC (`ID == 0`).
    Mpeg4,
    /// MPEG-2 AAC (`ID == 1`).
    Mpeg2,
}

impl MpegVersion {
    fn from_bit(bit: u8) -> Self {
        if bit == 0 {
            Self::Mpeg4
        } else {
            Self::Mpeg2
        }
    }

    fn bit(self) -> u8 {
        match self {
            Self::Mpeg4 => 0,
            Self::Mpeg2 => 1,
        }
    }
}

/// Parsed ADTS fixed and variable header fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdtsHeader {
    pub mpeg_version: MpegVersion,
    pub protection_absent: bool,
    /// ADTS profile field, not the MPEG Audio Object Type. AAC-LC is `1`.
    pub profile: u8,
    pub sampling_frequency_index: u8,
    pub private_bit: bool,
    pub channel_configuration: u8,
    pub original_copy: bool,
    pub home: bool,
    pub copyright_identification_bit: bool,
    pub copyright_identification_start: bool,
    /// Full ADTS frame size in bytes, including this header and optional CRC.
    pub frame_length: usize,
    pub buffer_fullness: u16,
    /// Number of raw data blocks minus one. `0` means one raw data block.
    pub number_of_raw_data_blocks_in_frame: u8,
    /// Optional 16-bit CRC value when `protection_absent == false`.
    pub crc_check: Option<u16>,
}

impl AdtsHeader {
    /// Create a common AAC-LC ADTS header description.
    ///
    /// `payload_len` is the raw AAC access-unit length, excluding the ADTS
    /// header. The resulting `frame_length` includes the 7-byte ADTS header.
    pub fn aac_lc(
        sample_rate: u32,
        channel_configuration: u8,
        payload_len: usize,
    ) -> Result<Self, AdtsError> {
        let sampling_frequency_index =
            sample_rate_index(sample_rate).ok_or(AdtsError::UnsupportedSampleRate(sample_rate))?;
        Self::new(
            MpegVersion::Mpeg4,
            1,
            sampling_frequency_index,
            channel_configuration,
            payload_len,
        )
    }

    /// Create an ADTS header description without CRC.
    pub fn new(
        mpeg_version: MpegVersion,
        profile: u8,
        sampling_frequency_index: u8,
        channel_configuration: u8,
        payload_len: usize,
    ) -> Result<Self, AdtsError> {
        if profile > 3 {
            return Err(AdtsError::InvalidProfile(profile));
        }
        if sampling_frequency_index >= 13 {
            return Err(AdtsError::InvalidSamplingFrequencyIndex(
                sampling_frequency_index,
            ));
        }
        if channel_configuration > 7 {
            return Err(AdtsError::InvalidChannelConfiguration(
                channel_configuration,
            ));
        }

        let frame_length = payload_len
            .checked_add(HEADER_LEN_WITHOUT_CRC)
            .ok_or(AdtsError::FrameTooLarge(payload_len))?;
        if frame_length > MAX_FRAME_LENGTH {
            return Err(AdtsError::FrameTooLarge(frame_length));
        }

        Ok(Self {
            mpeg_version,
            protection_absent: true,
            profile,
            sampling_frequency_index,
            private_bit: false,
            channel_configuration,
            original_copy: false,
            home: false,
            copyright_identification_bit: false,
            copyright_identification_start: false,
            frame_length,
            buffer_fullness: 0x07ff,
            number_of_raw_data_blocks_in_frame: 0,
            crc_check: None,
        })
    }

    pub fn header_len(self) -> usize {
        if self.protection_absent {
            HEADER_LEN_WITHOUT_CRC
        } else {
            HEADER_LEN_WITH_CRC
        }
    }

    pub fn payload_len(self) -> Result<usize, AdtsError> {
        self.frame_length
            .checked_sub(self.header_len())
            .ok_or(AdtsError::InvalidFrameLength {
                frame_length: self.frame_length,
                header_length: self.header_len(),
            })
    }

    pub fn sample_rate(self) -> Option<u32> {
        SAMPLE_RATES
            .get(self.sampling_frequency_index as usize)
            .copied()
            .filter(|rate| *rate != 0)
    }

    /// MPEG Audio Object Type inferred from the ADTS profile field.
    pub fn audio_object_type(self) -> u8 {
        self.profile + 1
    }

    /// Build a matching MPEG-4 AudioSpecificConfig from this ADTS header.
    ///
    /// ADTS stores the AAC profile as `audioObjectType - 1`, so this is a
    /// lossless conversion for the common profile/channel/sample-rate fields.
    /// Program Config Elements (`channel_configuration == 0`) are intentionally
    /// not parsed here; callers can still inspect the raw ADTS payload if they
    /// need full PCE support.
    pub fn audio_specific_config(self) -> Result<AudioSpecificConfig, AdtsError> {
        let sampling_frequency =
            self.sample_rate()
                .ok_or(AdtsError::InvalidSamplingFrequencyIndex(
                    self.sampling_frequency_index,
                ))?;

        Ok(AudioSpecificConfig {
            audio_object_type: self.audio_object_type(),
            sampling_frequency_index: self.sampling_frequency_index,
            sampling_frequency,
            channel_configuration: self.channel_configuration,
            extension: None,
            program_config: None,
            ga_specific: Some(GaSpecificConfig::default()),
            eld_specific: None,
            usac_config: None,
            error_protection_config: None,
            bits_read: 0,
        })
    }

    pub fn parse(input: &[u8]) -> Result<Self, AdtsError> {
        if input.len() < HEADER_LEN_WITHOUT_CRC {
            return Err(AdtsError::UnexpectedEof {
                needed: HEADER_LEN_WITHOUT_CRC,
                actual: input.len(),
            });
        }

        let syncword = ((input[0] as u16) << 4) | ((input[1] as u16) >> 4);
        if syncword != SYNCWORD {
            return Err(AdtsError::InvalidSyncword(syncword));
        }

        let layer = (input[1] >> 1) & 0x03;
        if layer != 0 {
            return Err(AdtsError::InvalidLayer(layer));
        }

        let protection_absent = (input[1] & 0x01) != 0;
        let header_len = if protection_absent {
            HEADER_LEN_WITHOUT_CRC
        } else {
            HEADER_LEN_WITH_CRC
        };
        if input.len() < header_len {
            return Err(AdtsError::UnexpectedEof {
                needed: header_len,
                actual: input.len(),
            });
        }

        let profile = (input[2] >> 6) & 0x03;
        let sampling_frequency_index = (input[2] >> 2) & 0x0f;
        if sampling_frequency_index >= 13 {
            return Err(AdtsError::InvalidSamplingFrequencyIndex(
                sampling_frequency_index,
            ));
        }

        let channel_configuration = ((input[2] & 0x01) << 2) | ((input[3] >> 6) & 0x03);
        let frame_length = (((input[3] & 0x03) as usize) << 11)
            | ((input[4] as usize) << 3)
            | (((input[5] & 0xe0) as usize) >> 5);
        if frame_length < header_len {
            return Err(AdtsError::InvalidFrameLength {
                frame_length,
                header_length: header_len,
            });
        }

        let crc_check = if protection_absent {
            None
        } else {
            Some(u16::from_be_bytes([input[7], input[8]]))
        };

        Ok(Self {
            mpeg_version: MpegVersion::from_bit((input[1] >> 3) & 0x01),
            protection_absent,
            profile,
            sampling_frequency_index,
            private_bit: (input[2] & 0x02) != 0,
            channel_configuration,
            original_copy: (input[3] & 0x20) != 0,
            home: (input[3] & 0x10) != 0,
            copyright_identification_bit: (input[3] & 0x08) != 0,
            copyright_identification_start: (input[3] & 0x04) != 0,
            frame_length,
            buffer_fullness: (((input[5] & 0x1f) as u16) << 6) | (((input[6] & 0xfc) as u16) >> 2),
            number_of_raw_data_blocks_in_frame: input[6] & 0x03,
            crc_check,
        })
    }

    /// Write this header to `out`, returning the number of bytes written.
    pub fn write(self, out: &mut [u8]) -> Result<usize, AdtsError> {
        let header_len = self.header_len();
        if out.len() < header_len {
            return Err(AdtsError::UnexpectedEof {
                needed: header_len,
                actual: out.len(),
            });
        }
        if self.profile > 3 {
            return Err(AdtsError::InvalidProfile(self.profile));
        }
        if self.sampling_frequency_index >= 13 {
            return Err(AdtsError::InvalidSamplingFrequencyIndex(
                self.sampling_frequency_index,
            ));
        }
        if self.channel_configuration > 7 {
            return Err(AdtsError::InvalidChannelConfiguration(
                self.channel_configuration,
            ));
        }
        if self.frame_length < header_len {
            return Err(AdtsError::InvalidFrameLength {
                frame_length: self.frame_length,
                header_length: header_len,
            });
        }
        if self.frame_length > MAX_FRAME_LENGTH {
            return Err(AdtsError::FrameTooLarge(self.frame_length));
        }
        if self.buffer_fullness > 0x07ff {
            return Err(AdtsError::InvalidBufferFullness(self.buffer_fullness));
        }
        if self.number_of_raw_data_blocks_in_frame > 3 {
            return Err(AdtsError::InvalidRawBlockCount(
                self.number_of_raw_data_blocks_in_frame,
            ));
        }

        out[0] = 0xff;
        out[1] = 0xf0 | (self.mpeg_version.bit() << 3) | u8::from(self.protection_absent);
        out[2] = (self.profile << 6)
            | (self.sampling_frequency_index << 2)
            | (u8::from(self.private_bit) << 1)
            | ((self.channel_configuration >> 2) & 0x01);
        out[3] = ((self.channel_configuration & 0x03) << 6)
            | (u8::from(self.original_copy) << 5)
            | (u8::from(self.home) << 4)
            | (u8::from(self.copyright_identification_bit) << 3)
            | (u8::from(self.copyright_identification_start) << 2)
            | (((self.frame_length >> 11) as u8) & 0x03);
        out[4] = (self.frame_length >> 3) as u8;
        out[5] =
            (((self.frame_length & 0x07) as u8) << 5) | ((self.buffer_fullness >> 6) as u8 & 0x1f);
        out[6] = (((self.buffer_fullness & 0x3f) as u8) << 2)
            | (self.number_of_raw_data_blocks_in_frame & 0x03);

        if !self.protection_absent {
            let crc = self.crc_check.unwrap_or(0).to_be_bytes();
            out[7] = crc[0];
            out[8] = crc[1];
        }

        Ok(header_len)
    }
}

/// Parsed ADTS frame view backed by the caller's input bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdtsFrame<'a> {
    pub header: AdtsHeader,
    pub bytes: &'a [u8],
    pub payload: &'a [u8],
}

/// One raw AAC data block extracted from an ADTS frame.
///
/// For a multi-block ADTS frame, the block payload excludes its trailing 16-bit
/// CRC. ADTS only signals block boundaries when CRC protection is enabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdtsRawDataBlock<'a> {
    pub payload: &'a [u8],
    pub crc_check: Option<u16>,
}

/// Owned ADTS frame returned by [`AdtsIncrementalStream`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedAdtsFrame {
    pub header: AdtsHeader,
    pub bytes: Vec<u8>,
    pub payload_range: std::ops::Range<usize>,
}

impl OwnedAdtsFrame {
    pub fn payload(&self) -> &[u8] {
        &self.bytes[self.payload_range.clone()]
    }

    pub fn as_frame(&self) -> AdtsFrame<'_> {
        AdtsFrame {
            header: self.header,
            bytes: &self.bytes,
            payload: self.payload(),
        }
    }
}

impl<'a> AdtsFrame<'a> {
    pub fn parse(input: &'a [u8]) -> Result<Self, AdtsError> {
        let mut header = AdtsHeader::parse(input)?;
        if input.len() < header.frame_length {
            return Err(AdtsError::UnexpectedEof {
                needed: header.frame_length,
                actual: input.len(),
            });
        }
        let bytes = &input[..header.frame_length];
        let payload_start =
            if !header.protection_absent && header.number_of_raw_data_blocks_in_frame != 0 {
                let positions_len = header.number_of_raw_data_blocks_in_frame as usize * 2;
                let header_crc_start = HEADER_LEN_WITHOUT_CRC + positions_len;
                let payload_start = header_crc_start + 2;
                if bytes.len() < payload_start {
                    return Err(AdtsError::UnexpectedEof {
                        needed: payload_start,
                        actual: bytes.len(),
                    });
                }
                header.crc_check = Some(u16::from_be_bytes([
                    bytes[header_crc_start],
                    bytes[header_crc_start + 1],
                ]));
                payload_start
            } else {
                header.header_len()
            };
        let payload = &bytes[payload_start..];
        Ok(Self {
            header,
            bytes,
            payload,
        })
    }

    pub fn audio_specific_config(self) -> Result<AudioSpecificConfig, AdtsError> {
        self.header.audio_specific_config()
    }

    /// Validate the ADTS header CRC for a protected multi-block frame.
    ///
    /// For a single-block frame the one transmitted CRC also includes
    /// codec-syntax regions selected during raw_data_block decoding, so it
    /// cannot be validated from transport bytes alone.
    pub fn validate_multi_block_header_crc(self) -> Result<(), AdtsError> {
        if self.header.protection_absent {
            return Err(AdtsError::CrcProtectionAbsent);
        }
        let raw_block_count = self.header.number_of_raw_data_blocks_in_frame;
        if raw_block_count == 0 {
            return Err(AdtsError::SyntaxRegionsRequiredForCrc);
        }
        let header_crc_start = HEADER_LEN_WITHOUT_CRC + raw_block_count as usize * 2;
        if self.bytes.len() < header_crc_start + 2 {
            return Err(AdtsError::UnexpectedEof {
                needed: header_crc_start + 2,
                actual: self.bytes.len(),
            });
        }
        let expected = u16::from_be_bytes([
            self.bytes[header_crc_start],
            self.bytes[header_crc_start + 1],
        ]);
        let calculated = adts_crc16(&self.bytes[..header_crc_start]);
        if calculated == expected {
            Ok(())
        } else {
            Err(AdtsError::CrcMismatch {
                expected,
                calculated,
            })
        }
    }

    /// Return all raw data blocks when ADTS carries enough boundary information.
    ///
    /// A frame with one block is always available. For multi-block ADTS, ISO
    /// ADTS places `raw_data_block_position` values before the header CRC only
    /// when `protection_absent == false`; CRC-less multi-block frames therefore
    /// need syntax-aware decoding to discover their boundaries and return an
    /// explicit error here instead.
    pub fn raw_data_blocks(self) -> Result<Vec<AdtsRawDataBlock<'a>>, AdtsError> {
        let blocks = self.header.number_of_raw_data_blocks_in_frame as usize + 1;
        if blocks == 1 {
            return Ok(vec![AdtsRawDataBlock {
                payload: self.payload,
                crc_check: None,
            }]);
        }
        if self.header.protection_absent {
            return Err(AdtsError::RawDataBlockBoundariesUnavailable);
        }

        let positions_len = (blocks - 1) * 2;
        let positions_start = HEADER_LEN_WITHOUT_CRC;
        let header_crc_start = positions_start + positions_len;
        let data_start = header_crc_start + 2;
        if self.bytes.len() < data_start {
            return Err(AdtsError::UnexpectedEof {
                needed: data_start,
                actual: self.bytes.len(),
            });
        }

        let mut end_offsets = Vec::with_capacity(blocks);
        for index in 0..blocks - 1 {
            let offset = positions_start + index * 2;
            end_offsets
                .push(u16::from_be_bytes([self.bytes[offset], self.bytes[offset + 1]]) as usize);
        }
        end_offsets.push(self.bytes.len() - data_start);

        let mut start = 0usize;
        let mut raw_blocks = Vec::with_capacity(blocks);
        for end in end_offsets {
            if end < start + 2 || end > self.bytes.len() - data_start {
                return Err(AdtsError::InvalidRawDataBlockPosition { start, end });
            }
            let block = &self.bytes[data_start + start..data_start + end];
            let crc_start = block.len() - 2;
            raw_blocks.push(AdtsRawDataBlock {
                payload: &block[..crc_start],
                crc_check: Some(u16::from_be_bytes([block[crc_start], block[crc_start + 1]])),
            });
            start = end;
        }
        Ok(raw_blocks)
    }
}

/// Iterator over an in-memory ADTS byte stream.
///
/// The iterator is strict: frames must be contiguous and complete. On the first
/// parse error it yields that error and then terminates.
#[derive(Debug, Clone)]
pub struct AdtsStream<'a> {
    input: &'a [u8],
    offset: usize,
    finished: bool,
}

impl<'a> AdtsStream<'a> {
    pub fn new(input: &'a [u8]) -> Self {
        Self {
            input,
            offset: 0,
            finished: false,
        }
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn remaining(&self) -> &'a [u8] {
        &self.input[self.offset..]
    }
}

impl<'a> Iterator for AdtsStream<'a> {
    type Item = Result<AdtsFrame<'a>, AdtsError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished || self.offset == self.input.len() {
            return None;
        }

        match AdtsFrame::parse(&self.input[self.offset..]) {
            Ok(frame) => {
                self.offset += frame.bytes.len();
                Some(Ok(frame))
            }
            Err(err) => {
                self.finished = true;
                Some(Err(err))
            }
        }
    }
}

pub fn adts_frames(input: &[u8]) -> AdtsStream<'_> {
    AdtsStream::new(input)
}

/// Incremental, resynchronizing ADTS frame buffer.
///
/// Call [`Self::push`] with arbitrary byte chunks and repeatedly call
/// [`Self::next_frame`]. Incomplete headers and frames remain buffered until
/// more input arrives. Bytes preceding the next byte-aligned ADTS syncword are
/// discarded so a later valid frame can be recovered after corruption.
#[derive(Debug, Clone, Default)]
pub struct AdtsIncrementalStream {
    buffered: Vec<u8>,
    discarded_bytes: usize,
}

impl AdtsIncrementalStream {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, input: &[u8]) {
        self.buffered.extend_from_slice(input);
    }

    pub fn buffered_len(&self) -> usize {
        self.buffered.len()
    }

    pub fn discarded_bytes(&self) -> usize {
        self.discarded_bytes
    }

    pub fn next_frame(&mut self) -> Option<OwnedAdtsFrame> {
        loop {
            let Some(sync_offset) = find_adts_syncword(&self.buffered) else {
                let keep = usize::from(self.buffered.last() == Some(&0xff));
                let discard = self.buffered.len().saturating_sub(keep);
                if discard != 0 {
                    self.discard(discard);
                }
                return None;
            };
            if sync_offset != 0 {
                self.discard(sync_offset);
            }
            if self.buffered.len() < HEADER_LEN_WITHOUT_CRC {
                return None;
            }
            let header = match AdtsHeader::parse(&self.buffered) {
                Ok(header) => header,
                Err(AdtsError::UnexpectedEof { .. }) => return None,
                Err(_) => {
                    self.discard(1);
                    continue;
                }
            };
            if self.buffered.len() < header.frame_length {
                // A damaged header can advertise up to 8191 bytes and would
                // otherwise pin the incremental decoder indefinitely. If a
                // later byte-aligned syncword already forms a complete valid
                // frame, prefer it and account the false candidate as loss.
                if let Some(recovery_offset) = find_complete_adts_frame(&self.buffered[1..]) {
                    self.discard(recovery_offset + 1);
                    continue;
                }
                return None;
            }
            let bytes = self
                .buffered
                .drain(..header.frame_length)
                .collect::<Vec<u8>>();
            let (parsed_header, payload_range) = match AdtsFrame::parse(&bytes) {
                Ok(frame) => {
                    let payload_start =
                        frame.payload.as_ptr() as usize - frame.bytes.as_ptr() as usize;
                    (frame.header, payload_start..frame.bytes.len())
                }
                Err(_) => {
                    self.discarded_bytes += bytes.len();
                    continue;
                }
            };
            return Some(OwnedAdtsFrame {
                header: parsed_header,
                bytes,
                payload_range,
            });
        }
    }

    fn discard(&mut self, count: usize) {
        self.buffered.drain(..count);
        self.discarded_bytes += count;
    }
}

fn find_complete_adts_frame(input: &[u8]) -> Option<usize> {
    let mut searched = 0usize;
    while searched < input.len() {
        let relative = find_adts_syncword(&input[searched..])?;
        let offset = searched + relative;
        let candidate = &input[offset..];
        if candidate.len() >= HEADER_LEN_WITHOUT_CRC {
            if let Ok(header) = AdtsHeader::parse(candidate) {
                if candidate.len() >= header.frame_length
                    && AdtsFrame::parse(&candidate[..header.frame_length]).is_ok()
                {
                    return Some(offset);
                }
            }
        }
        searched = offset + 1;
    }
    None
}

fn find_adts_syncword(bytes: &[u8]) -> Option<usize> {
    bytes
        .windows(2)
        .position(|pair| pair[0] == 0xff && (pair[1] & 0xf0) == 0xf0)
}

/// FDK-style constant-average-bitrate access-unit loss estimator.
///
/// The estimator retains the division remainder across synchronization events,
/// preventing repeated sub-frame byte losses from being rounded away.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccessUnitLossEstimator {
    average_bitrate: u32,
    sample_rate: u32,
    samples_per_frame: u32,
    remainder: i64,
}

impl AccessUnitLossEstimator {
    pub fn new(
        average_bitrate: u32,
        sample_rate: u32,
        samples_per_frame: u32,
    ) -> Result<Self, AdtsError> {
        if average_bitrate == 0 || sample_rate == 0 || samples_per_frame == 0 {
            return Err(AdtsError::InvalidLossEstimatorConfiguration);
        }
        Ok(Self {
            average_bitrate,
            sample_rate,
            samples_per_frame,
            remainder: 0,
        })
    }

    /// Estimate missing AUs once a valid current frame has been recovered.
    /// `bit_distance` excludes the current valid frame, while
    /// `buffer_fullness_delta_bits` mirrors FDK's optional fullness correction.
    pub fn recovered(
        &mut self,
        bit_distance: usize,
        current_frame_bits: usize,
        buffer_fullness_delta_bits: i64,
    ) -> u32 {
        let denominator = self.average_bitrate as i64 * self.samples_per_frame as i64;
        let distance = bit_distance.saturating_add(current_frame_bits) as i64;
        let numerator = self.sample_rate as i64
            * (buffer_fullness_delta_bits.saturating_add(distance))
            + self.remainder;
        if numerator <= 0 {
            self.remainder = numerator;
            return 0;
        }
        let mut access_units = numerator / denominator;
        self.remainder = numerator % denominator;

        // The synchronized current frame was included in the buffer math but
        // must not itself be reported as lost. Match FDK's nearest-frame
        // remainder adjustment before subtracting it.
        if denominator - self.remainder >= self.remainder {
            access_units -= 1;
        }
        self.remainder = 0;
        access_units.max(0) as u32
    }

    /// Account for bytes discarded while synchronization is still missing.
    /// The returned count can be concealed immediately; fractional AU distance
    /// remains in `remainder` for the next observation.
    pub fn synchronization_error(
        &mut self,
        bit_distance: usize,
        buffer_fullness_delta_bits: i64,
    ) -> u32 {
        let denominator = self.average_bitrate as i64 * self.samples_per_frame as i64;
        let numerator = self.sample_rate as i64
            * buffer_fullness_delta_bits.saturating_add(bit_distance as i64)
            + self.remainder;
        if numerator <= 0 {
            self.remainder = numerator;
            return 0;
        }
        let access_units = numerator / denominator;
        self.remainder = numerator % denominator;
        access_units as u32
    }

    pub fn remainder(&self) -> i64 {
        self.remainder
    }

    pub fn average_bitrate(&self) -> u32 {
        self.average_bitrate
    }

    pub fn reset(&mut self) {
        self.remainder = 0;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdtsError {
    UnexpectedEof {
        needed: usize,
        actual: usize,
    },
    InvalidSyncword(u16),
    InvalidLayer(u8),
    InvalidProfile(u8),
    InvalidSamplingFrequencyIndex(u8),
    InvalidChannelConfiguration(u8),
    InvalidFrameLength {
        frame_length: usize,
        header_length: usize,
    },
    InvalidLossEstimatorConfiguration,
    FrameTooLarge(usize),
    InvalidBufferFullness(u16),
    InvalidRawBlockCount(u8),
    CrcMismatch {
        expected: u16,
        calculated: u16,
    },
    CrcProtectionAbsent,
    InvalidRawDataBlockPosition {
        start: usize,
        end: usize,
    },
    InvalidCrcBitRange {
        start: usize,
        end: usize,
        available_bits: usize,
    },
    InvalidCrcPadding {
        region_bits: usize,
        padded_bits: usize,
    },
    RawDataBlockBoundariesUnavailable,
    SyntaxRegionsRequiredForCrc,
    UnsupportedSampleRate(u32),
}

impl fmt::Display for AdtsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::UnexpectedEof { needed, actual } => {
                write!(f, "ADTS input too short: need {needed} bytes, got {actual}")
            }
            Self::InvalidSyncword(value) => write!(f, "invalid ADTS syncword 0x{value:03x}"),
            Self::InvalidLayer(value) => write!(f, "invalid ADTS layer {value}"),
            Self::InvalidProfile(value) => write!(f, "invalid ADTS profile {value}"),
            Self::InvalidSamplingFrequencyIndex(value) => {
                write!(f, "invalid ADTS sampling frequency index {value}")
            }
            Self::InvalidChannelConfiguration(value) => {
                write!(f, "invalid ADTS channel configuration {value}")
            }
            Self::InvalidFrameLength {
                frame_length,
                header_length,
            } => write!(
                f,
                "invalid ADTS frame length {frame_length}; header length is {header_length}"
            ),
            Self::InvalidLossEstimatorConfiguration => write!(
                f,
                "ADTS loss estimator requires non-zero bitrate, sample rate, and frame length"
            ),
            Self::FrameTooLarge(value) => write!(f, "ADTS frame too large: {value} bytes"),
            Self::InvalidBufferFullness(value) => write!(f, "invalid ADTS buffer fullness {value}"),
            Self::InvalidRawBlockCount(value) => {
                write!(f, "invalid ADTS raw data block count {value}")
            }
            Self::CrcMismatch {
                expected,
                calculated,
            } => write!(
                f,
                "ADTS CRC mismatch: read 0x{expected:04x}, calculated 0x{calculated:04x}"
            ),
            Self::CrcProtectionAbsent => write!(f, "ADTS CRC protection is absent"),
            Self::InvalidRawDataBlockPosition { start, end } => write!(
                f,
                "invalid ADTS raw data block position range {start}..{end}"
            ),
            Self::InvalidCrcBitRange {
                start,
                end,
                available_bits,
            } => write!(
                f,
                "invalid ADTS CRC bit range {start}..{end} for {available_bits} bits"
            ),
            Self::InvalidCrcPadding {
                region_bits,
                padded_bits,
            } => write!(
                f,
                "ADTS CRC region has {region_bits} bits but padded length is {padded_bits}"
            ),
            Self::RawDataBlockBoundariesUnavailable => write!(
                f,
                "ADTS multi-raw-data-block boundaries require CRC protection"
            ),
            Self::SyntaxRegionsRequiredForCrc => write!(
                f,
                "ADTS CRC validation requires codec syntax-region boundaries"
            ),
            Self::UnsupportedSampleRate(value) => write!(f, "unsupported ADTS sample rate {value}"),
        }
    }
}

impl std::error::Error for AdtsError {}

pub fn sample_rate_index(sample_rate: u32) -> Option<u8> {
    SAMPLE_RATES
        .iter()
        .position(|rate| *rate == sample_rate)
        .map(|index| index as u8)
}

pub fn sample_rate_from_index(index: u8) -> Option<u32> {
    SAMPLE_RATES
        .get(index as usize)
        .copied()
        .filter(|rate| *rate != 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_fdk_adts_crc16_msb_first() {
        assert_eq!(adts_crc16(b""), 0xffff);
        assert_eq!(adts_crc16(b"123456789"), 0xaee7);
        assert_eq!(adts_crc16(&[0xaa, 0xbb]), 0x7f9d);
        assert_eq!(
            adts_crc16_regions([&[0xaa][..], &[0xbb][..]]),
            adts_crc16(&[0xaa, 0xbb])
        );
    }

    #[test]
    fn parses_common_aac_lc_header() {
        let bytes = [0xff, 0xf1, 0x50, 0x80, 0x01, 0x7f, 0xfc, 0xaa, 0xbb, 0xcc];
        let header = AdtsHeader::parse(&bytes).unwrap();
        assert_eq!(header.mpeg_version, MpegVersion::Mpeg4);
        assert!(header.protection_absent);
        assert_eq!(header.profile, 1);
        assert_eq!(header.audio_object_type(), 2);
        assert_eq!(header.sampling_frequency_index, 4);
        assert_eq!(header.sample_rate(), Some(44_100));
        assert_eq!(header.channel_configuration, 2);
        assert_eq!(header.frame_length, 11);
        assert_eq!(header.payload_len().unwrap(), 4);
        assert_eq!(header.buffer_fullness, 0x07ff);
    }

    #[test]
    fn parses_frame_payload_slice() {
        let bytes = [
            0xff, 0xf1, 0x50, 0x80, 0x01, 0x7f, 0xfc, 0xaa, 0xbb, 0xcc, 0xdd, 0xee,
        ];
        let frame = AdtsFrame::parse(&bytes).unwrap();
        assert_eq!(frame.bytes, &bytes[..11]);
        assert_eq!(frame.payload, &[0xaa, 0xbb, 0xcc, 0xdd]);
    }

    #[test]
    fn writes_and_roundtrips_header() {
        let header = AdtsHeader::aac_lc(48_000, 2, 123).unwrap();
        let mut out = [0u8; 7];
        assert_eq!(header.write(&mut out).unwrap(), 7);
        let parsed = AdtsHeader::parse(&out).unwrap();
        assert_eq!(parsed, header);
        assert_eq!(parsed.sample_rate(), Some(48_000));
        assert_eq!(parsed.payload_len().unwrap(), 123);
    }

    #[test]
    fn writes_crc_header() {
        let mut header = AdtsHeader::aac_lc(44_100, 1, 5).unwrap();
        header.protection_absent = false;
        header.frame_length = 14;
        header.crc_check = Some(0x1234);
        let mut out = [0u8; 9];
        assert_eq!(header.write(&mut out).unwrap(), 9);
        let parsed = AdtsHeader::parse(&out).unwrap();
        assert_eq!(parsed.crc_check, Some(0x1234));
        assert_eq!(parsed.header_len(), 9);
        assert_eq!(parsed.payload_len().unwrap(), 5);
    }

    #[test]
    fn extracts_crc_protected_multi_raw_data_blocks() {
        // ADTS header (7), one raw_data_block_position (2), header CRC (2),
        // then two raw blocks each followed by a 16-bit block CRC.
        let first_payload = [0xaa, 0xbb];
        let second_payload = [0xcc, 0xdd, 0xee];
        let mut header = AdtsHeader::aac_lc(44_100, 1, 13).unwrap();
        header.protection_absent = false;
        header.frame_length = 20;
        header.number_of_raw_data_blocks_in_frame = 1;
        header.crc_check = Some(0x1234);

        let mut standard_header = vec![0; header.header_len()];
        header.write(&mut standard_header).unwrap();
        let mut bytes = standard_header[..HEADER_LEN_WITHOUT_CRC].to_vec();
        bytes.extend_from_slice(&4u16.to_be_bytes()); // first payload + first block CRC
        bytes.extend_from_slice(&0x1234u16.to_be_bytes()); // header CRC
        bytes.extend_from_slice(&first_payload);
        bytes.extend_from_slice(&0xabcd_u16.to_be_bytes());
        bytes.extend_from_slice(&second_payload);
        bytes.extend_from_slice(&0x5678_u16.to_be_bytes());

        let frame = AdtsFrame::parse(&bytes).unwrap();
        let blocks = frame.raw_data_blocks().unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].payload, first_payload);
        assert_eq!(blocks[0].crc_check, Some(0xabcd));
        assert_eq!(blocks[1].payload, second_payload);
        assert_eq!(blocks[1].crc_check, Some(0x5678));
    }

    #[test]
    fn validates_crc_protected_multi_block_header() {
        let payload = [0xaa, 0xbb];
        let block_len = payload.len() + 2;
        let mut header = AdtsHeader::aac_lc(44_100, 1, 0).unwrap();
        header.protection_absent = false;
        header.number_of_raw_data_blocks_in_frame = 1;
        header.frame_length = 7 + 2 + 2 + block_len * 2;
        header.crc_check = Some(0);

        let mut standard_header = vec![0; header.header_len()];
        header.write(&mut standard_header).unwrap();
        let mut bytes = standard_header[..7].to_vec();
        bytes.extend_from_slice(&(block_len as u16).to_be_bytes());
        let header_crc = adts_crc16(&bytes);
        bytes.extend_from_slice(&header_crc.to_be_bytes());
        for block_crc in [0x1111u16, 0x2222] {
            bytes.extend_from_slice(&payload);
            bytes.extend_from_slice(&block_crc.to_be_bytes());
        }

        assert_eq!(
            AdtsFrame::parse(&bytes)
                .unwrap()
                .validate_multi_block_header_crc(),
            Ok(())
        );
        bytes[9] ^= 1;
        assert!(matches!(
            AdtsFrame::parse(&bytes)
                .unwrap()
                .validate_multi_block_header_crc(),
            Err(AdtsError::CrcMismatch { .. })
        ));
    }

    #[test]
    fn refuses_to_guess_crc_less_multi_block_boundaries() {
        let mut header = AdtsHeader::aac_lc(44_100, 1, 4).unwrap();
        header.number_of_raw_data_blocks_in_frame = 1;
        let mut bytes = vec![0; header.header_len()];
        header.write(&mut bytes).unwrap();
        bytes.extend_from_slice(&[0xaa, 0xbb, 0xcc, 0xdd]);

        assert_eq!(
            AdtsFrame::parse(&bytes)
                .unwrap()
                .raw_data_blocks()
                .unwrap_err(),
            AdtsError::RawDataBlockBoundariesUnavailable
        );
    }

    #[test]
    fn rejects_invalid_syncword() {
        let err = AdtsHeader::parse(&[0x00; 7]).unwrap_err();
        assert_eq!(err, AdtsError::InvalidSyncword(0));
    }

    #[test]
    fn converts_header_to_audio_specific_config() {
        let header = AdtsHeader::aac_lc(44_100, 2, 4).unwrap();
        let asc = header.audio_specific_config().unwrap();
        assert_eq!(asc.audio_object_type, 2);
        assert_eq!(asc.sampling_frequency_index, 4);
        assert_eq!(asc.sampling_frequency, 44_100);
        assert_eq!(asc.channel_configuration, 2);
        assert_eq!(asc.to_bytes().unwrap(), vec![0x12, 0x10]);
    }

    #[test]
    fn iterates_adts_stream() {
        let payloads: [&[u8]; 2] = [&[0xaa, 0xbb], &[0xcc, 0xdd, 0xee]];
        let mut stream = Vec::new();
        for payload in payloads {
            let header = AdtsHeader::aac_lc(48_000, 2, payload.len()).unwrap();
            let mut header_bytes = [0u8; 7];
            header.write(&mut header_bytes).unwrap();
            stream.extend_from_slice(&header_bytes);
            stream.extend_from_slice(payload);
        }

        let frames = AdtsStream::new(&stream)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].payload, &[0xaa, 0xbb]);
        assert_eq!(frames[1].payload, &[0xcc, 0xdd, 0xee]);
        assert_eq!(
            frames[0]
                .audio_specific_config()
                .unwrap()
                .to_bytes()
                .unwrap(),
            vec![0x11, 0x90]
        );
    }

    #[test]
    fn incrementally_buffers_partial_adts_frames() {
        let payload = [0xaa, 0xbb, 0xcc];
        let header = AdtsHeader::aac_lc(44_100, 1, payload.len()).unwrap();
        let mut frame = vec![0; header.header_len()];
        header.write(&mut frame).unwrap();
        frame.extend_from_slice(&payload);

        let mut stream = AdtsIncrementalStream::new();
        stream.push(&frame[..4]);
        assert!(stream.next_frame().is_none());
        assert_eq!(stream.buffered_len(), 4);
        stream.push(&frame[4..]);
        let decoded = stream.next_frame().unwrap();
        assert_eq!(decoded.header, header);
        assert_eq!(decoded.payload(), payload);
        assert!(stream.next_frame().is_none());
    }

    #[test]
    fn incrementally_resynchronizes_after_garbage_and_split_syncword() {
        let payload = [0x11, 0x22];
        let header = AdtsHeader::aac_lc(48_000, 2, payload.len()).unwrap();
        let mut frame = vec![0; header.header_len()];
        header.write(&mut frame).unwrap();
        frame.extend_from_slice(&payload);

        let mut stream = AdtsIncrementalStream::new();
        stream.push(&[0x00, 0x12, 0xff]);
        assert!(stream.next_frame().is_none());
        assert_eq!(stream.buffered_len(), 1); // preserve possible syncword prefix
        stream.push(&frame[1..]);
        let decoded = stream.next_frame().unwrap();
        assert_eq!(decoded.payload(), payload);
        assert_eq!(stream.discarded_bytes(), 2);
    }

    #[test]
    fn incremental_stream_recovers_from_incomplete_false_sync_header() {
        let false_header = AdtsHeader::aac_lc(44_100, 2, MAX_FRAME_LENGTH - 7).unwrap();
        let mut false_bytes = vec![0; false_header.header_len()];
        false_header.write(&mut false_bytes).unwrap();
        false_bytes.extend_from_slice(&[0x12, 0x34, 0x56]);

        let payload = [0xaa, 0xbb];
        let valid_header = AdtsHeader::aac_lc(48_000, 1, payload.len()).unwrap();
        let mut valid = vec![0; valid_header.header_len()];
        valid_header.write(&mut valid).unwrap();
        valid.extend_from_slice(&payload);

        let mut stream = AdtsIncrementalStream::new();
        stream.push(&false_bytes);
        assert!(stream.next_frame().is_none());
        assert_eq!(stream.buffered_len(), false_bytes.len());
        stream.push(&valid);
        let recovered = stream.next_frame().expect("later complete frame");
        assert_eq!(recovered.header, valid_header);
        assert_eq!(recovered.payload(), payload);
        assert_eq!(stream.discarded_bytes(), false_bytes.len());
        assert_eq!(stream.buffered_len(), 0);
    }

    #[test]
    fn stream_reports_truncated_frame() {
        let header = AdtsHeader::aac_lc(44_100, 2, 4).unwrap();
        let mut bytes = [0u8; 7];
        header.write(&mut bytes).unwrap();
        let mut iter = AdtsStream::new(&bytes);
        assert!(matches!(
            iter.next().unwrap(),
            Err(AdtsError::UnexpectedEof {
                needed: 11,
                actual: 7
            })
        ));
        assert!(iter.next().is_none());
    }

    #[test]
    fn estimates_lost_access_units_like_fdk_cbr_math() {
        let mut estimator = AccessUnitLossEstimator::new(128_000, 48_000, 1024).unwrap();
        // At this configuration an average AU is about 2730.67 bits.
        assert_eq!(estimator.recovered(2731, 2731, 0), 1);
        assert_eq!(estimator.remainder(), 0);

        // Fractional losses accumulate while sync remains unavailable.
        assert_eq!(estimator.synchronization_error(1365, 0), 0);
        assert_ne!(estimator.remainder(), 0);
        assert_eq!(estimator.synchronization_error(1366, 0), 1);
    }

    #[test]
    fn mpeg_version_and_sample_rate_tables_roundtrip() {
        assert_eq!(MpegVersion::from_bit(0), MpegVersion::Mpeg4);
        assert_eq!(MpegVersion::from_bit(1), MpegVersion::Mpeg2);
        assert_eq!(MpegVersion::Mpeg4.bit(), 0);
        assert_eq!(MpegVersion::Mpeg2.bit(), 1);
        for index in 0u8..13 {
            let rate = sample_rate_from_index(index).unwrap();
            assert_eq!(sample_rate_index(rate), Some(index));
        }
        assert_eq!(sample_rate_from_index(13), None);
        assert_eq!(sample_rate_index(12_345), None);
    }

    #[test]
    fn constructors_reject_all_invalid_header_fields() {
        assert_eq!(
            AdtsHeader::aac_lc(12_345, 1, 0),
            Err(AdtsError::UnsupportedSampleRate(12_345))
        );
        assert_eq!(
            AdtsHeader::new(MpegVersion::Mpeg4, 4, 4, 1, 0),
            Err(AdtsError::InvalidProfile(4))
        );
        assert_eq!(
            AdtsHeader::new(MpegVersion::Mpeg4, 1, 13, 1, 0),
            Err(AdtsError::InvalidSamplingFrequencyIndex(13))
        );
        assert_eq!(
            AdtsHeader::new(MpegVersion::Mpeg4, 1, 4, 8, 0),
            Err(AdtsError::InvalidChannelConfiguration(8))
        );
        assert!(matches!(
            AdtsHeader::new(MpegVersion::Mpeg4, 1, 4, 1, MAX_FRAME_LENGTH),
            Err(AdtsError::FrameTooLarge(_))
        ));
        assert!(matches!(
            AdtsHeader::new(MpegVersion::Mpeg4, 1, 4, 1, usize::MAX),
            Err(AdtsError::FrameTooLarge(_))
        ));
    }

    #[test]
    fn writer_validates_output_and_mutated_header_fields() {
        let base = AdtsHeader::aac_lc(44_100, 1, 0).unwrap();
        assert!(matches!(
            base.write(&mut [0; 6]),
            Err(AdtsError::UnexpectedEof { .. })
        ));
        let cases = [
            (
                AdtsHeader { profile: 4, ..base },
                AdtsError::InvalidProfile(4),
            ),
            (
                AdtsHeader {
                    sampling_frequency_index: 13,
                    ..base
                },
                AdtsError::InvalidSamplingFrequencyIndex(13),
            ),
            (
                AdtsHeader {
                    channel_configuration: 8,
                    ..base
                },
                AdtsError::InvalidChannelConfiguration(8),
            ),
            (
                AdtsHeader {
                    frame_length: 1,
                    ..base
                },
                AdtsError::InvalidFrameLength {
                    frame_length: 1,
                    header_length: 7,
                },
            ),
            (
                AdtsHeader {
                    frame_length: MAX_FRAME_LENGTH + 1,
                    ..base
                },
                AdtsError::FrameTooLarge(MAX_FRAME_LENGTH + 1),
            ),
            (
                AdtsHeader {
                    buffer_fullness: 0x800,
                    ..base
                },
                AdtsError::InvalidBufferFullness(0x800),
            ),
            (
                AdtsHeader {
                    number_of_raw_data_blocks_in_frame: 4,
                    ..base
                },
                AdtsError::InvalidRawBlockCount(4),
            ),
        ];
        for (header, expected) in cases {
            assert_eq!(header.write(&mut [0; 9]), Err(expected));
        }
    }

    #[test]
    fn header_accessors_cover_crc_and_invalid_sampling_index() {
        let mut header = AdtsHeader::aac_lc(44_100, 2, 4).unwrap();
        assert_eq!(header.payload_len().unwrap(), 4);
        assert_eq!(header.sample_rate(), Some(44_100));
        assert_eq!(header.audio_object_type(), 2);
        header.protection_absent = false;
        header.frame_length = 8;
        assert!(matches!(
            header.payload_len(),
            Err(AdtsError::InvalidFrameLength { .. })
        ));
        header.sampling_frequency_index = 13;
        assert_eq!(header.sample_rate(), None);
        assert_eq!(
            header.audio_specific_config(),
            Err(AdtsError::InvalidSamplingFrequencyIndex(13))
        );
    }

    #[test]
    fn crc_bit_regions_validate_both_range_directions() {
        let reversed_bounds = [7, 3];
        assert!(matches!(
            adts_crc16_bit_regions([(&[0xff][..], 7..9)]),
            Err(AdtsError::InvalidCrcBitRange { .. })
        ));
        assert!(matches!(
            adts_crc16_bit_regions([(&[0xff][..], reversed_bounds[0]..reversed_bounds[1])]),
            Err(AdtsError::InvalidCrcBitRange { .. })
        ));
        assert_eq!(adts_crc16_regions([&[][..]]), 0xffff);
        assert_eq!(adts_crc16(&[]), 0xffff);
    }

    #[test]
    fn stream_accessors_and_owned_frame_bridge_are_consistent() {
        let header = AdtsHeader::aac_lc(44_100, 1, 1).unwrap();
        let mut bytes = vec![0; 7];
        header.write(&mut bytes).unwrap();
        bytes.push(0xaa);
        let mut stream = adts_frames(&bytes);
        assert_eq!(stream.offset(), 0);
        assert_eq!(stream.remaining(), bytes);
        let frame = stream.next().unwrap().unwrap();
        assert_eq!(stream.offset(), bytes.len());
        assert!(stream.remaining().is_empty());
        let owned = OwnedAdtsFrame {
            header,
            bytes: bytes.clone(),
            payload_range: 7..8,
        };
        assert_eq!(owned.payload(), [0xaa]);
        assert_eq!(owned.as_frame(), frame);
    }

    #[test]
    fn loss_estimator_validates_configuration_and_negative_corrections() {
        for config in [(0, 48_000, 1024), (128_000, 0, 1024), (128_000, 48_000, 0)] {
            assert_eq!(
                AccessUnitLossEstimator::new(config.0, config.1, config.2),
                Err(AdtsError::InvalidLossEstimatorConfiguration)
            );
        }
        let mut estimator = AccessUnitLossEstimator::new(128_000, 48_000, 1024).unwrap();
        assert_eq!(estimator.average_bitrate(), 128_000);
        assert_eq!(estimator.recovered(0, 0, -1), 0);
        assert!(estimator.remainder() < 0);
        estimator.reset();
        assert_eq!(estimator.remainder(), 0);
        assert_eq!(estimator.synchronization_error(0, -1), 0);
        assert!(estimator.remainder() < 0);
    }

    #[test]
    fn formats_every_adts_error_variant() {
        let errors = [
            AdtsError::UnexpectedEof {
                needed: 7,
                actual: 1,
            },
            AdtsError::InvalidSyncword(0),
            AdtsError::InvalidLayer(1),
            AdtsError::InvalidProfile(4),
            AdtsError::InvalidSamplingFrequencyIndex(13),
            AdtsError::InvalidChannelConfiguration(8),
            AdtsError::InvalidFrameLength {
                frame_length: 1,
                header_length: 7,
            },
            AdtsError::InvalidLossEstimatorConfiguration,
            AdtsError::FrameTooLarge(9000),
            AdtsError::InvalidBufferFullness(2048),
            AdtsError::InvalidRawBlockCount(4),
            AdtsError::CrcMismatch {
                expected: 1,
                calculated: 2,
            },
            AdtsError::CrcProtectionAbsent,
            AdtsError::InvalidRawDataBlockPosition { start: 4, end: 2 },
            AdtsError::InvalidCrcBitRange {
                start: 8,
                end: 9,
                available_bits: 8,
            },
            AdtsError::RawDataBlockBoundariesUnavailable,
            AdtsError::SyntaxRegionsRequiredForCrc,
            AdtsError::UnsupportedSampleRate(12_345),
        ];
        assert!(errors.iter().all(|error| !error.to_string().is_empty()));
    }

    #[test]
    fn parser_rejects_short_crc_invalid_rate_and_short_frame_length() {
        assert_eq!(
            AdtsHeader::parse(&[0xff; 6]),
            Err(AdtsError::UnexpectedEof {
                needed: 7,
                actual: 6
            })
        );

        let mut protected = AdtsHeader::aac_lc(44_100, 1, 0).unwrap();
        protected.protection_absent = false;
        protected.frame_length = 9;
        protected.crc_check = Some(0);
        let mut bytes = [0u8; 9];
        protected.write(&mut bytes).unwrap();
        assert_eq!(
            AdtsHeader::parse(&bytes[..7]),
            Err(AdtsError::UnexpectedEof {
                needed: 9,
                actual: 7
            })
        );

        let mut invalid_rate = bytes;
        invalid_rate[2] = (invalid_rate[2] & 0xc3) | (13 << 2);
        assert_eq!(
            AdtsHeader::parse(&invalid_rate),
            Err(AdtsError::InvalidSamplingFrequencyIndex(13))
        );

        let mut invalid_length = bytes;
        invalid_length[3] &= 0xfc;
        invalid_length[4] = 0;
        invalid_length[5] &= 0x1f;
        assert_eq!(
            AdtsHeader::parse(&invalid_length),
            Err(AdtsError::InvalidFrameLength {
                frame_length: 0,
                header_length: 9
            })
        );
    }

    #[test]
    fn protected_multi_block_header_requires_position_and_crc_bytes() {
        let mut header = AdtsHeader::aac_lc(44_100, 1, 0).unwrap();
        header.protection_absent = false;
        header.number_of_raw_data_blocks_in_frame = 1;
        header.frame_length = 9;
        header.crc_check = Some(0);
        let mut bytes = [0u8; 9];
        header.write(&mut bytes).unwrap();
        assert_eq!(
            AdtsFrame::parse(&bytes),
            Err(AdtsError::UnexpectedEof {
                needed: 11,
                actual: 9
            })
        );

        let frame = AdtsFrame {
            header,
            bytes: &bytes[..8],
            payload: &[],
        };
        assert_eq!(
            frame.validate_multi_block_header_crc(),
            Err(AdtsError::UnexpectedEof {
                needed: 11,
                actual: 8
            })
        );
        let frame = AdtsFrame {
            header,
            bytes: &bytes[..8],
            payload: &[],
        };
        assert_eq!(
            frame.raw_data_blocks(),
            Err(AdtsError::UnexpectedEof {
                needed: 11,
                actual: 8
            })
        );
    }

    #[test]
    fn crc_validation_rejects_absent_and_single_block_modes() {
        let header = AdtsHeader::aac_lc(44_100, 1, 0).unwrap();
        let mut bytes = [0u8; 7];
        header.write(&mut bytes).unwrap();
        assert_eq!(
            AdtsFrame::parse(&bytes)
                .unwrap()
                .validate_multi_block_header_crc(),
            Err(AdtsError::CrcProtectionAbsent)
        );

        let mut protected = header;
        protected.protection_absent = false;
        protected.frame_length = 9;
        protected.crc_check = Some(0);
        let mut bytes = [0u8; 9];
        protected.write(&mut bytes).unwrap();
        assert_eq!(
            AdtsFrame::parse(&bytes)
                .unwrap()
                .validate_multi_block_header_crc(),
            Err(AdtsError::SyntaxRegionsRequiredForCrc)
        );
    }

    #[test]
    fn raw_block_positions_reject_too_short_and_out_of_range_blocks() {
        for position in [1u16, 100] {
            let mut header = AdtsHeader::aac_lc(44_100, 1, 0).unwrap();
            header.protection_absent = false;
            header.number_of_raw_data_blocks_in_frame = 1;
            header.frame_length = 15;
            header.crc_check = Some(0);
            let mut standard = vec![0; 9];
            header.write(&mut standard).unwrap();
            let mut bytes = standard[..7].to_vec();
            bytes.extend_from_slice(&position.to_be_bytes());
            bytes.extend_from_slice(&0u16.to_be_bytes());
            bytes.extend_from_slice(&[0, 0, 0, 0]);
            assert!(matches!(
                AdtsFrame::parse(&bytes).unwrap().raw_data_blocks(),
                Err(AdtsError::InvalidRawDataBlockPosition { .. })
            ));
        }
    }

    #[test]
    fn incremental_stream_discards_invalid_sync_candidates_and_malformed_frames() {
        let payload = [0xaa];
        let header = AdtsHeader::aac_lc(44_100, 1, payload.len()).unwrap();
        let mut valid = vec![0; 7];
        header.write(&mut valid).unwrap();
        valid.extend_from_slice(&payload);

        let mut stream = AdtsIncrementalStream::new();
        stream.push(&[0xff, 0xf7, 0, 0, 0, 0, 0]); // layer 3, invalid candidate
        stream.push(&valid);
        let frame = stream.next_frame().unwrap();
        assert_eq!(frame.payload(), payload);
        assert!(stream.discarded_bytes() >= 7);

        let mut malformed_header = AdtsHeader::aac_lc(44_100, 1, 0).unwrap();
        malformed_header.protection_absent = false;
        malformed_header.number_of_raw_data_blocks_in_frame = 1;
        malformed_header.frame_length = 9;
        malformed_header.crc_check = Some(0);
        let mut malformed = vec![0; 9];
        malformed_header.write(&mut malformed).unwrap();
        let mut stream = AdtsIncrementalStream::new();
        stream.push(&malformed);
        stream.push(&valid);
        let frame = stream.next_frame().unwrap();
        assert_eq!(frame.payload(), payload);
        assert!(stream.discarded_bytes() >= malformed.len());
    }

    #[test]
    fn crc_bit_regions_and_single_raw_block_take_success_paths() {
        assert_eq!(
            adts_crc16_bit_regions([(&[0xaa, 0xbb][..], 0..16)]).unwrap(),
            adts_crc16(&[0xaa, 0xbb])
        );
        let partial = adts_crc16_bit_regions([(&[0b1010_0000][..], 0..4)]).unwrap();
        assert_ne!(partial, 0xffff);

        let padded = adts_crc16_padded_bit_regions([(&[0b1010_0000][..], 0..4, 8)]).unwrap();
        assert_eq!(padded, adts_crc16(&[0b1010_0000]));
        assert_ne!(padded, partial);
        assert_eq!(
            adts_crc16_padded_bit_regions([(&[0xff][..], 0..8, 7)]),
            Err(AdtsError::InvalidCrcPadding {
                region_bits: 8,
                padded_bits: 7,
            })
        );

        let payload = [1, 2, 3];
        let header = AdtsHeader::aac_lc(44_100, 1, payload.len()).unwrap();
        let mut bytes = vec![0; 7];
        header.write(&mut bytes).unwrap();
        bytes.extend_from_slice(&payload);
        let blocks = AdtsFrame::parse(&bytes).unwrap().raw_data_blocks().unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].payload, payload);
        assert_eq!(blocks[0].crc_check, None);
    }

    #[test]
    fn incremental_stream_waits_for_crc_header_and_complete_search_returns_none() {
        let mut header = AdtsHeader::aac_lc(44_100, 1, 0).unwrap();
        header.protection_absent = false;
        header.frame_length = 9;
        header.crc_check = Some(0);
        let mut bytes = [0u8; 9];
        header.write(&mut bytes).unwrap();
        let mut stream = AdtsIncrementalStream::new();
        stream.push(&bytes[..7]);
        assert!(stream.next_frame().is_none());
        assert_eq!(stream.buffered_len(), 7);
        stream.push(&bytes[7..]);
        assert!(stream.next_frame().is_some());

        assert_eq!(find_complete_adts_frame(&bytes), Some(0));

        let incomplete_header = AdtsHeader::aac_lc(44_100, 1, 1).unwrap();
        let mut incomplete = [0u8; 7];
        incomplete_header.write(&mut incomplete).unwrap();
        assert_eq!(find_complete_adts_frame(&incomplete), None);
        assert_eq!(find_complete_adts_frame(&[0xff, 0xf7, 0, 0, 0, 0, 0]), None);
        assert_eq!(find_complete_adts_frame(&[0xff]), None);
        assert_eq!(find_complete_adts_frame(&[]), None);
    }
}
