//! Pure Rust LOAS framing for LATM AudioMuxElement payloads.

use std::fmt;

const LOAS_SYNCWORD: u16 = 0x02b7;
const LOAS_HEADER_LEN: usize = 3;

pub fn write_loas_frame(audio_mux_element: &[u8]) -> Result<Vec<u8>, LoasError> {
    if audio_mux_element.len() > 0x1fff {
        return Err(LoasError::PayloadTooLarge(audio_mux_element.len()));
    }
    let length = audio_mux_element.len();
    let mut output = Vec::with_capacity(LOAS_HEADER_LEN + length);
    output.push((LOAS_SYNCWORD >> 3) as u8);
    output.push(((LOAS_SYNCWORD as u8 & 0x07) << 5) | ((length >> 8) as u8 & 0x1f));
    output.push(length as u8);
    output.extend_from_slice(audio_mux_element);
    Ok(output)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoasFrame<'a> {
    pub bytes: &'a [u8],
    pub audio_mux_element: &'a [u8],
}

impl<'a> LoasFrame<'a> {
    pub fn parse(input: &'a [u8]) -> Result<Self, LoasError> {
        if input.len() < LOAS_HEADER_LEN {
            return Err(LoasError::UnexpectedEof {
                needed: LOAS_HEADER_LEN,
                actual: input.len(),
            });
        }
        let syncword = ((input[0] as u16) << 3) | ((input[1] as u16) >> 5);
        if syncword != LOAS_SYNCWORD {
            return Err(LoasError::InvalidSyncword(syncword));
        }
        let length = (((input[1] & 0x1f) as usize) << 8) | input[2] as usize;
        let frame_len = LOAS_HEADER_LEN + length;
        if input.len() < frame_len {
            return Err(LoasError::UnexpectedEof {
                needed: frame_len,
                actual: input.len(),
            });
        }
        let bytes = &input[..frame_len];
        Ok(Self {
            bytes,
            audio_mux_element: &bytes[LOAS_HEADER_LEN..],
        })
    }
}

#[derive(Debug, Clone)]
pub struct LoasStream<'a> {
    input: &'a [u8],
    offset: usize,
    finished: bool,
}

impl<'a> LoasStream<'a> {
    pub fn new(input: &'a [u8]) -> Self {
        Self {
            input,
            offset: 0,
            finished: false,
        }
    }
}

impl<'a> Iterator for LoasStream<'a> {
    type Item = Result<LoasFrame<'a>, LoasError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished || self.offset == self.input.len() {
            return None;
        }
        match LoasFrame::parse(&self.input[self.offset..]) {
            Ok(frame) => {
                self.offset += frame.bytes.len();
                Some(Ok(frame))
            }
            Err(error) => {
                self.finished = true;
                Some(Err(error))
            }
        }
    }
}

pub fn loas_frames(input: &[u8]) -> LoasStream<'_> {
    LoasStream::new(input)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedLoasFrame {
    pub bytes: Vec<u8>,
}

impl OwnedLoasFrame {
    pub fn audio_mux_element(&self) -> &[u8] {
        &self.bytes[LOAS_HEADER_LEN..]
    }
}

/// Incremental, resynchronizing LOAS frame buffer.
///
/// Arbitrarily split input can be appended with [`Self::push`]. Incomplete
/// headers and frames remain buffered, while bytes before the next byte-aligned
/// LOAS syncword are discarded.
#[derive(Debug, Clone, Default)]
pub struct LoasIncrementalStream {
    buffered: Vec<u8>,
    discarded_bytes: usize,
}

impl LoasIncrementalStream {
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

    pub fn next_frame(&mut self) -> Option<OwnedLoasFrame> {
        loop {
            let Some(sync_offset) = find_loas_syncword(&self.buffered) else {
                let keep = usize::from(self.buffered.last() == Some(&0x56));
                let discard = self.buffered.len().saturating_sub(keep);
                self.discard(discard);
                return None;
            };
            self.discard(sync_offset);
            if self.buffered.len() < LOAS_HEADER_LEN {
                return None;
            }
            let payload_len =
                (((self.buffered[1] & 0x1f) as usize) << 8) | self.buffered[2] as usize;
            let frame_len = LOAS_HEADER_LEN + payload_len;
            if self.buffered.len() < frame_len {
                // A false syncword can advertise an almost 8 KiB payload and
                // pin a stream even when a complete frame is already buffered
                // behind it. Prefer the later complete candidate, as the ADTS
                // incremental reader does.
                if let Some(recovery_offset) = find_complete_loas_frame(&self.buffered[1..]) {
                    self.discard(recovery_offset + 1);
                    continue;
                }
                return None;
            }
            let bytes = self.buffered.drain(..frame_len).collect::<Vec<_>>();
            return Some(OwnedLoasFrame { bytes });
        }
    }

    fn discard(&mut self, count: usize) {
        if count != 0 {
            self.buffered.drain(..count);
            self.discarded_bytes += count;
        }
    }
}

fn find_complete_loas_frame(input: &[u8]) -> Option<usize> {
    let mut searched = 0usize;
    while searched < input.len() {
        let Some(relative) = find_loas_syncword(&input[searched..]) else {
            break;
        };
        let offset = searched + relative;
        let candidate = &input[offset..];
        if candidate.len() >= LOAS_HEADER_LEN {
            let payload_len = (((candidate[1] & 0x1f) as usize) << 8) | candidate[2] as usize;
            let frame_len = LOAS_HEADER_LEN + payload_len;
            if candidate.len() >= frame_len && LoasFrame::parse(&candidate[..frame_len]).is_ok() {
                return Some(offset);
            }
        }
        searched = offset + 1;
    }
    None
}

fn find_loas_syncword(bytes: &[u8]) -> Option<usize> {
    bytes
        .windows(2)
        .position(|pair| pair[0] == 0x56 && (pair[1] & 0xe0) == 0xe0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoasError {
    UnexpectedEof { needed: usize, actual: usize },
    InvalidSyncword(u16),
    PayloadTooLarge(usize),
}

impl fmt::Display for LoasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof { needed, actual } => {
                write!(f, "LOAS input too short: need {needed} bytes, got {actual}")
            }
            Self::InvalidSyncword(value) => write!(f, "invalid LOAS syncword 0x{value:03x}"),
            Self::PayloadTooLarge(length) => write!(f, "LOAS payload is too large: {length} bytes"),
        }
    }
}

impl std::error::Error for LoasError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(payload: &[u8]) -> Vec<u8> {
        let length = payload.len();
        let mut bytes = vec![0x56, 0xe0 | ((length >> 8) as u8 & 0x1f), length as u8];
        bytes.extend_from_slice(payload);
        bytes
    }

    #[test]
    fn writer_roundtrips_loas_header_and_payload() {
        let bytes = write_loas_frame(&[1, 2, 3, 4]).unwrap();
        let parsed = LoasFrame::parse(&bytes).unwrap();
        assert_eq!(parsed.audio_mux_element, [1, 2, 3, 4]);
        assert_eq!(parsed.bytes, bytes);
        assert_eq!(
            write_loas_frame(&vec![0; 0x2000]),
            Err(LoasError::PayloadTooLarge(0x2000))
        );
    }

    #[test]
    fn parses_loas_audio_mux_element() {
        let bytes = frame(&[0x12, 0x34, 0x56]);
        let parsed = LoasFrame::parse(&bytes).unwrap();
        assert_eq!(parsed.bytes, bytes);
        assert_eq!(parsed.audio_mux_element, &[0x12, 0x34, 0x56]);
    }

    #[test]
    fn reports_truncated_and_invalid_loas_frames() {
        assert_eq!(
            LoasFrame::parse(&[0x56, 0xe0]),
            Err(LoasError::UnexpectedEof {
                needed: 3,
                actual: 2
            })
        );
        assert_eq!(
            LoasFrame::parse(&[0x00, 0x00, 0x00]),
            Err(LoasError::InvalidSyncword(0))
        );
        assert_eq!(
            LoasFrame::parse(&[0x56, 0xe0, 2, 0xaa]),
            Err(LoasError::UnexpectedEof {
                needed: 5,
                actual: 4
            })
        );
    }

    #[test]
    fn stream_stops_after_the_first_parse_error() {
        let mut stream = loas_frames(&[0x56, 0xe0, 1]);
        assert!(matches!(
            stream.next(),
            Some(Err(LoasError::UnexpectedEof { .. }))
        ));
        assert!(stream.next().is_none());
    }

    #[test]
    fn formats_all_loas_errors() {
        assert_eq!(
            LoasError::UnexpectedEof {
                needed: 5,
                actual: 3
            }
            .to_string(),
            "LOAS input too short: need 5 bytes, got 3"
        );
        assert_eq!(
            LoasError::InvalidSyncword(0x123).to_string(),
            "invalid LOAS syncword 0x123"
        );
        assert_eq!(
            LoasError::PayloadTooLarge(9000).to_string(),
            "LOAS payload is too large: 9000 bytes"
        );
    }

    #[test]
    fn iterates_contiguous_loas_frames() {
        let mut input = frame(&[1, 2]);
        input.extend_from_slice(&frame(&[3]));
        let frames = loas_frames(&input).collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].audio_mux_element, &[1, 2]);
        assert_eq!(frames[1].audio_mux_element, &[3]);
    }

    #[test]
    fn incrementally_buffers_partial_loas_frames() {
        let bytes = frame(&[1, 2, 3]);
        let mut stream = LoasIncrementalStream::new();
        stream.push(&bytes[..2]);
        assert!(stream.next_frame().is_none());
        assert_eq!(stream.buffered_len(), 2);
        stream.push(&bytes[2..]);
        let parsed = stream.next_frame().unwrap();
        assert_eq!(parsed.bytes, bytes);
        assert_eq!(parsed.audio_mux_element(), &[1, 2, 3]);
        assert_eq!(stream.buffered_len(), 0);
    }

    #[test]
    fn incrementally_resynchronizes_after_garbage_and_split_syncword() {
        let bytes = frame(&[0xaa]);
        let mut stream = LoasIncrementalStream::new();
        stream.push(&[0x00, 0x12, 0x56]);
        assert!(stream.next_frame().is_none());
        assert_eq!(stream.buffered_len(), 1);
        stream.push(&bytes[1..]);
        assert_eq!(stream.next_frame().unwrap().audio_mux_element(), &[0xaa]);
        assert_eq!(stream.discarded_bytes(), 2);
    }

    #[test]
    fn incremental_stream_recovers_from_incomplete_false_sync_header() {
        let valid = frame(&[0xaa, 0xbb]);
        let mut bytes = vec![0x56, 0xff, 0xff];
        bytes.extend_from_slice(&valid);
        assert_eq!(find_complete_loas_frame(&bytes), Some(3));

        let mut stream = LoasIncrementalStream::new();
        stream.push(&bytes);
        let parsed = stream.next_frame().unwrap();
        assert_eq!(parsed.bytes, valid);
        assert_eq!(stream.discarded_bytes(), 3);
        assert_eq!(stream.buffered_len(), 0);
        assert_eq!(find_complete_loas_frame(&[0x56, 0xff, 0xff]), None);
        assert_eq!(find_complete_loas_frame(&[0x56, 0xe0]), None);
    }

    #[test]
    fn incremental_stream_keeps_a_genuinely_incomplete_frame() {
        let mut stream = LoasIncrementalStream::new();
        stream.push(&[0x56, 0xe0, 2, 0xaa]);
        assert!(stream.next_frame().is_none());
        assert_eq!(stream.buffered_len(), 4);
        assert_eq!(stream.discarded_bytes(), 0);
    }
}
