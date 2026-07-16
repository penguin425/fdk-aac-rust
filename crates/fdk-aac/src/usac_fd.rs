//! USAC frequency-domain channel decoding for 768/1024 core frames.

use crate::bits::{BitError, BitReader};
use crate::filterbank::imdct_planned_f32;
use crate::inverse::inverse_quantize_value_f32;
use crate::scalefactor::{ScalefactorData, ScalefactorError};
use crate::tns::{TnsData, TnsError};
use crate::usac::{UsacError, UsacFdChannelSideInfo, UsacWindowSequence};
use crate::usac_arith::{UsacArithmeticDecoder, UsacArithmeticError};
use crate::usac_fac::{FacData, FacError};

const FDK_AAC_ROM: &str = include_str!(concat!(
    env!("FDK_AAC_UPSTREAM_DIR"),
    "/libAACdec/src/aac_rom.cpp"
));

#[derive(Debug, Clone, PartialEq)]
pub struct UsacFdFrame {
    pub side: UsacFdChannelSideInfo,
    pub scalefactors: ScalefactorData,
    pub quantized_windows: Vec<Vec<i32>>,
    pub spectrum_windows: Vec<Vec<f32>>,
    pub fac: Option<FacData>,
    pub tns: TnsData,
    pub bits_read: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UsacFdError {
    Bit(BitError),
    Side(UsacError),
    Scale(ScalefactorError),
    Arithmetic(UsacArithmeticError),
    InvalidFrameLength(usize),
    InvalidSamplingIndex(u8),
    InvalidLfeWindowSequence(UsacWindowSequence),
    LfeFacDataPresent,
    Fac(FacError),
    Tns(TnsError),
}

impl From<BitError> for UsacFdError {
    fn from(v: BitError) -> Self {
        Self::Bit(v)
    }
}
impl From<ScalefactorError> for UsacFdError {
    fn from(v: ScalefactorError) -> Self {
        Self::Scale(v)
    }
}
impl From<UsacArithmeticError> for UsacFdError {
    fn from(v: UsacArithmeticError) -> Self {
        Self::Arithmetic(v)
    }
}
impl From<FacError> for UsacFdError {
    fn from(v: FacError) -> Self {
        Self::Fac(v)
    }
}
impl From<TnsError> for UsacFdError {
    fn from(v: TnsError) -> Self {
        Self::Tns(v)
    }
}

#[derive(Debug, Clone)]
pub struct UsacFdChannelDecoder {
    frame_length: usize,
    sampling_index: u8,
    arithmetic: UsacArithmeticDecoder,
    overlap: Vec<f32>,
}

impl UsacFdChannelDecoder {
    pub fn new(frame_length: usize, sampling_index: u8) -> Result<Self, UsacFdError> {
        if !matches!(frame_length, 768 | 1024) {
            return Err(UsacFdError::InvalidFrameLength(frame_length));
        }
        if sampling_index > 12 {
            return Err(UsacFdError::InvalidSamplingIndex(sampling_index));
        }
        Ok(Self {
            frame_length,
            sampling_index,
            arithmetic: UsacArithmeticDecoder::new(),
            overlap: vec![0.0; frame_length],
        })
    }

    pub fn parse(
        &mut self,
        reader: &mut BitReader<'_>,
        noise_filling: bool,
        independent: bool,
    ) -> Result<UsacFdFrame, UsacFdError> {
        let start = reader.bits_read();
        let long_offsets = sfb_offsets(self.frame_length, self.sampling_index, false);
        let short_offsets = sfb_offsets(self.frame_length, self.sampling_index, true);
        let side = UsacFdChannelSideInfo::parse(
            reader,
            noise_filling,
            (long_offsets.len() - 1) as u8,
            (short_offsets.len() - 1) as u8,
        )
        .map_err(UsacFdError::Side)?;
        self.parse_after_side(reader, side, independent, start)
    }

    /// Parse the restricted USAC LFE frequency-domain syntax. Unlike an SCE,
    /// an LFE has no core-mode, TNS-present, or noise-filling fields and is
    /// constrained to an ONLY_LONG window without FAC data.
    pub fn parse_lfe(
        &mut self,
        reader: &mut BitReader<'_>,
        independent: bool,
    ) -> Result<UsacFdFrame, UsacFdError> {
        let start = reader.bits_read();
        let global_gain = reader.read_u8(8)?;
        let (long_count, short_count) = self.sfb_counts();
        let ics = crate::usac::UsacIcsInfo::parse(reader, long_count, short_count)
            .map_err(UsacFdError::Side)?;
        if ics.window_sequence != UsacWindowSequence::OnlyLong {
            return Err(UsacFdError::InvalidLfeWindowSequence(ics.window_sequence));
        }
        let side = UsacFdChannelSideInfo {
            tns_data_present: false,
            global_gain,
            noise_level_and_offset: None,
            ics,
            bits_read: reader.bits_read() - start,
        };
        self.parse_after_side_with_fac(reader, side, independent, start, false)
    }

    pub fn sfb_counts(&self) -> (u8, u8) {
        (
            (sfb_offsets(self.frame_length, self.sampling_index, false).len() - 1) as u8,
            (sfb_offsets(self.frame_length, self.sampling_index, true).len() - 1) as u8,
        )
    }

    pub fn band_offsets(&self, short: bool) -> Vec<usize> {
        sfb_offsets(self.frame_length, self.sampling_index, short)
    }

    pub fn read_side_with_ics(
        &self,
        reader: &mut BitReader<'_>,
        noise_filling: bool,
        ics: crate::usac::UsacIcsInfo,
        tns_data_present: bool,
    ) -> Result<UsacFdChannelSideInfo, UsacFdError> {
        let start = reader.bits_read();
        let global_gain = reader.read_u8(8)?;
        let noise_level_and_offset = noise_filling.then(|| reader.read_u8(8)).transpose()?;
        Ok(UsacFdChannelSideInfo {
            tns_data_present,
            global_gain,
            noise_level_and_offset,
            ics,
            bits_read: reader.bits_read() - start,
        })
    }

    pub fn parse_after_side(
        &mut self,
        reader: &mut BitReader<'_>,
        side: UsacFdChannelSideInfo,
        independent: bool,
        start: usize,
    ) -> Result<UsacFdFrame, UsacFdError> {
        self.parse_after_side_with_fac(reader, side, independent, start, true)
    }

    fn parse_after_side_with_fac(
        &mut self,
        reader: &mut BitReader<'_>,
        side: UsacFdChannelSideInfo,
        independent: bool,
        start: usize,
        fac_allowed: bool,
    ) -> Result<UsacFdFrame, UsacFdError> {
        let long_offsets = sfb_offsets(self.frame_length, self.sampling_index, false);
        let short_offsets = sfb_offsets(self.frame_length, self.sampling_index, true);
        let scalefactors = ScalefactorData::decode_usac(
            reader,
            side.global_gain,
            side.ics.window_group_lengths.len(),
            usize::from(side.ics.max_sfb),
        )?;
        let short = side.ics.window_sequence == UsacWindowSequence::EightShort;
        let offsets = if short { &short_offsets } else { &long_offsets };
        let tns = if side.tns_data_present {
            TnsData::parse_present_usac(reader, short, (offsets.len() - 1) as u8)?
        } else {
            TnsData::absent(if short { 8 } else { 1 })
        };
        let transmitted = offsets[usize::from(side.ics.max_sfb)];
        let quantized_windows = self.arithmetic.decode_windows(
            reader,
            transmitted,
            self.frame_length,
            short,
            independent,
        )?;
        let mut spectrum_windows = quantized_windows
            .iter()
            .map(|window| vec![0.0; window.len()])
            .collect::<Vec<_>>();
        let mut window = 0;
        for (group, &group_length) in side.ics.window_group_lengths.iter().enumerate() {
            for group_window in 0..usize::from(group_length) {
                for band in 0..usize::from(side.ics.max_sfb) {
                    for line in offsets[band]..offsets[band + 1] {
                        spectrum_windows[window + group_window][line] = inverse_quantize_value_f32(
                            quantized_windows[window + group_window][line],
                            scalefactors.values[group][band],
                        );
                    }
                }
            }
            window += usize::from(group_length);
        }
        tns.apply_to_windows_f32(&mut spectrum_windows, offsets)?;
        let fac_present = reader.read_bool()?;
        if fac_present && !fac_allowed {
            return Err(UsacFdError::LfeFacDataPresent);
        }
        let fac = fac_present
            .then(|| {
                FacData::parse(
                    reader,
                    if short {
                        self.frame_length / 16
                    } else {
                        self.frame_length / 8
                    },
                    true,
                )
            })
            .transpose()?;
        Ok(UsacFdFrame {
            side,
            scalefactors,
            quantized_windows,
            spectrum_windows,
            fac,
            tns,
            bits_read: reader.bits_read() - start,
        })
    }

    pub fn render(&mut self, frame: &UsacFdFrame) -> Vec<f32> {
        let mut time = vec![0.0; 2 * self.frame_length];
        if frame.spectrum_windows.len() == 1 {
            time = imdct_planned_f32(&frame.spectrum_windows[0]);
            sine_window(&mut time);
        } else {
            let short = self.frame_length / 8;
            let base = (self.frame_length - short) / 2;
            for (index, spectrum) in frame.spectrum_windows.iter().enumerate() {
                let mut block = imdct_planned_f32(spectrum);
                sine_window(&mut block);
                let offset = base + index * short;
                for (target, sample) in time[offset..].iter_mut().zip(block) {
                    *target += sample;
                }
            }
        }
        let output = (0..self.frame_length)
            .map(|i| time[i] + self.overlap[i])
            .collect();
        self.overlap.copy_from_slice(&time[self.frame_length..]);
        output
    }
}

fn sine_window(time: &mut [f32]) {
    let length = time.len();
    for (i, sample) in time.iter_mut().enumerate() {
        *sample *= (std::f32::consts::PI * (i as f32 + 0.5) / length as f32).sin();
    }
}

fn sfb_offsets(frame_length: usize, index: u8, short: bool) -> Vec<usize> {
    let mut rate = match index {
        0 | 1 => 96,
        2 => 64,
        3 | 4 => 48,
        5 => 32,
        6 | 7 => 24,
        8..=10 => 16,
        _ => 8,
    };
    // FDK deliberately reuses the 48 kHz short-window table at sampling
    // index 5 (32 kHz); there is no sfb_32_128/sfb_32_96 ROM table.
    if short && index == 5 {
        rate = 48;
    }
    let length = if short {
        frame_length / 8
    } else {
        frame_length
    };
    let name = format!("sfb_{rate}_{length}");
    let start = FDK_AAC_ROM
        .match_indices(&name)
        .find(|(offset, _)| FDK_AAC_ROM.as_bytes().get(offset + name.len()) == Some(&b'['))
        .map(|(offset, _)| offset)
        .expect("FDK SFB ROM");
    let source = &FDK_AAC_ROM[start..];
    let body = &source[source.find('{').unwrap() + 1..source.find("};").unwrap()];
    body.split(|c: char| !c.is_ascii_digit())
        .filter(|v| !v.is_empty())
        .map(|v| v.parse().unwrap())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitWriter;

    fn ics(sequence: UsacWindowSequence, max_sfb: u8) -> crate::usac::UsacIcsInfo {
        crate::usac::UsacIcsInfo {
            window_sequence: sequence,
            window_shape: false,
            max_sfb,
            scale_factor_grouping: (sequence == UsacWindowSequence::EightShort).then_some(0x7f),
            window_group_lengths: if sequence == UsacWindowSequence::EightShort {
                vec![8]
            } else {
                vec![1]
            },
        }
    }

    #[test]
    fn loads_768_long_and_short_sfb_tables() {
        assert_eq!(*sfb_offsets(768, 3, false).last().unwrap(), 768);
        assert_eq!(*sfb_offsets(768, 3, true).last().unwrap(), 96);
    }

    #[test]
    fn renders_empty_long_fd_frame() {
        let mut decoder = UsacFdChannelDecoder::new(1024, 3).unwrap();
        let frame = UsacFdFrame {
            side: UsacFdChannelSideInfo {
                tns_data_present: false,
                global_gain: 0,
                noise_level_and_offset: None,
                ics: crate::usac::UsacIcsInfo {
                    window_sequence: UsacWindowSequence::OnlyLong,
                    window_shape: false,
                    max_sfb: 0,
                    scale_factor_grouping: None,
                    window_group_lengths: vec![1],
                },
                bits_read: 0,
            },
            scalefactors: ScalefactorData {
                values: vec![vec![]],
            },
            quantized_windows: vec![vec![0; 1024]],
            spectrum_windows: vec![vec![0.0; 1024]],
            fac: None,
            tns: TnsData::absent(1),
            bits_read: 0,
        };
        assert_eq!(decoder.render(&frame), vec![0.0; 1024]);
    }

    #[test]
    fn validates_constructor_and_exposes_all_sfb_rate_classes() {
        assert!(matches!(
            UsacFdChannelDecoder::new(512, 3),
            Err(UsacFdError::InvalidFrameLength(512))
        ));
        assert!(matches!(
            UsacFdChannelDecoder::new(1024, 13),
            Err(UsacFdError::InvalidSamplingIndex(13))
        ));
        for frame_length in [768, 1024] {
            for index in 0..=12 {
                let decoder = UsacFdChannelDecoder::new(frame_length, index).unwrap();
                let (long, short) = decoder.sfb_counts();
                assert!(long > 0 && short > 0);
                assert_eq!(decoder.band_offsets(false).last(), Some(&frame_length));
                assert_eq!(decoder.band_offsets(true).last(), Some(&(frame_length / 8)));
            }
        }
    }

    #[test]
    fn reads_side_info_with_external_ics() {
        let decoder = UsacFdChannelDecoder::new(1024, 3).unwrap();
        let mut writer = BitWriter::new();
        writer.write(123, 8);
        writer.write(0xab, 8);
        let bytes = writer.finish();
        let side = decoder
            .read_side_with_ics(
                &mut BitReader::new(&bytes),
                true,
                ics(UsacWindowSequence::OnlyLong, 0),
                true,
            )
            .unwrap();
        assert_eq!(side.global_gain, 123);
        assert_eq!(side.noise_level_and_offset, Some(0xab));
        assert!(side.tns_data_present);
        assert_eq!(side.bits_read, 16);

        assert!(matches!(
            decoder.read_side_with_ics(
                &mut BitReader::new(&[]),
                false,
                ics(UsacWindowSequence::OnlyLong, 0),
                false,
            ),
            Err(UsacFdError::Bit(BitError::UnexpectedEof { .. }))
        ));
    }

    #[test]
    fn parses_zero_band_frame_after_side_info() {
        let mut decoder = UsacFdChannelDecoder::new(1024, 3).unwrap();
        let side = UsacFdChannelSideInfo {
            tns_data_present: false,
            global_gain: 100,
            noise_level_and_offset: None,
            ics: ics(UsacWindowSequence::OnlyLong, 0),
            bits_read: 0,
        };
        let frame = decoder
            .parse_after_side(&mut BitReader::new(&[0]), side, true, 0)
            .unwrap();
        assert_eq!(frame.quantized_windows, [vec![0; 1024]]);
        assert_eq!(frame.spectrum_windows, [vec![0.0; 1024]]);
        assert_eq!(frame.fac, None);
        assert_eq!(frame.bits_read, 1);
    }

    #[test]
    fn parses_restricted_lfe_fd_syntax_and_rejects_forbidden_tools() {
        let mut decoder = UsacFdChannelDecoder::new(1024, 3).unwrap();
        let mut writer = BitWriter::new();
        writer.write(100, 8); // global_gain; no preceding TNS flag
        writer.write(0, 2); // ONLY_LONG
        writer.write_bool(false); // window_shape
        writer.write(0, 6); // max_sfb
        writer.write_bool(false); // no FAC
        let frame = decoder
            .parse_lfe(&mut BitReader::new(&writer.finish()), true)
            .unwrap();
        assert_eq!(frame.side.global_gain, 100);
        assert!(!frame.side.tns_data_present);
        assert_eq!(frame.side.noise_level_and_offset, None);
        assert_eq!(frame.spectrum_windows, [vec![0.0; 1024]]);

        let mut short = BitWriter::new();
        short.write(0, 8);
        short.write(2, 2); // EIGHT_SHORT
        short.write_bool(false);
        short.write(0, 4);
        short.write(0, 7);
        assert_eq!(
            decoder
                .parse_lfe(&mut BitReader::new(&short.finish()), true)
                .unwrap_err(),
            UsacFdError::InvalidLfeWindowSequence(UsacWindowSequence::EightShort)
        );

        let mut fac = BitWriter::new();
        fac.write(0, 8);
        fac.write(0, 2);
        fac.write_bool(false);
        fac.write(0, 6);
        fac.write_bool(true);
        assert_eq!(
            decoder
                .parse_lfe(&mut BitReader::new(&fac.finish()), true)
                .unwrap_err(),
            UsacFdError::LfeFacDataPresent
        );
    }

    #[test]
    fn renders_empty_short_windows_and_preserves_overlap_shape() {
        let mut decoder = UsacFdChannelDecoder::new(1024, 3).unwrap();
        let frame = UsacFdFrame {
            side: UsacFdChannelSideInfo {
                tns_data_present: false,
                global_gain: 0,
                noise_level_and_offset: None,
                ics: ics(UsacWindowSequence::EightShort, 0),
                bits_read: 0,
            },
            scalefactors: ScalefactorData {
                values: vec![vec![]],
            },
            quantized_windows: vec![vec![0; 128]; 8],
            spectrum_windows: vec![vec![0.0; 128]; 8],
            fac: None,
            tns: TnsData::absent(8),
            bits_read: 0,
        };
        assert_eq!(decoder.render(&frame), vec![0.0; 1024]);
        assert_eq!(decoder.render(&frame), vec![0.0; 1024]);

        let mut samples = vec![1.0; 8];
        sine_window(&mut samples);
        assert!(samples.iter().all(|sample| *sample > 0.0 && *sample <= 1.0));
        assert!((samples[0] - samples[7]).abs() < 1e-6);
    }

    #[test]
    fn converts_all_nested_decoder_errors() {
        assert!(matches!(
            UsacFdError::from(BitError::UnexpectedEof {
                needed_bits: 1,
                remaining_bits: 0
            }),
            UsacFdError::Bit(_)
        ));
        assert!(matches!(
            UsacFdError::from(ScalefactorError::RaggedCodebookGrid),
            UsacFdError::Scale(_)
        ));
        assert!(matches!(
            UsacFdError::from(UsacArithmeticError::EscapeOverflow),
            UsacFdError::Arithmetic(_)
        ));
        assert!(matches!(
            UsacFdError::from(FacError::InvalidLength(7)),
            UsacFdError::Fac(_)
        ));
        assert!(matches!(
            UsacFdError::from(TnsError::LayoutMismatch),
            UsacFdError::Tns(_)
        ));
    }

    #[test]
    fn parse_after_side_propagates_scale_tns_arithmetic_and_fac_errors() {
        let make_side = |ics, tns_data_present| UsacFdChannelSideInfo {
            tns_data_present,
            global_gain: 100,
            noise_level_and_offset: None,
            ics,
            bits_read: 0,
        };

        let mut decoder = UsacFdChannelDecoder::new(1024, 3).unwrap();
        let mut grouped_short = ics(UsacWindowSequence::EightShort, 1);
        grouped_short.scale_factor_grouping = Some(0);
        grouped_short.window_group_lengths = vec![1; 8];
        assert!(matches!(
            decoder.parse_after_side(
                &mut BitReader::new(&[]),
                make_side(grouped_short, false),
                true,
                0,
            ),
            Err(UsacFdError::Scale(_))
        ));

        assert!(matches!(
            decoder.parse_after_side(
                &mut BitReader::new(&[]),
                make_side(ics(UsacWindowSequence::OnlyLong, 0), true),
                true,
                0,
            ),
            Err(UsacFdError::Tns(_))
        ));

        let mut decoder = UsacFdChannelDecoder::new(1024, 3).unwrap();
        assert!(matches!(
            decoder.parse_after_side(
                &mut BitReader::new(&[0]),
                make_side(ics(UsacWindowSequence::OnlyLong, 0), false),
                false,
                0,
            ),
            Err(UsacFdError::Arithmetic(
                UsacArithmeticError::MissingPreviousContext
            ))
        ));

        for sequence in [UsacWindowSequence::OnlyLong, UsacWindowSequence::EightShort] {
            let mut decoder = UsacFdChannelDecoder::new(1024, 3).unwrap();
            assert!(matches!(
                decoder.parse_after_side(
                    &mut BitReader::new(&[0x80]),
                    make_side(ics(sequence, 0), false),
                    true,
                    0,
                ),
                Err(UsacFdError::Fac(_))
            ));
        }
    }

    #[test]
    fn decodes_and_inverse_quantizes_one_transmitted_band() {
        let mut state = 1u32;
        let mut payload = vec![0u8; 512];
        for byte in &mut payload {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *byte = (state >> 24) as u8;
        }
        let side = UsacFdChannelSideInfo {
            tns_data_present: false,
            global_gain: 100,
            noise_level_and_offset: None,
            ics: ics(UsacWindowSequence::OnlyLong, 1),
            bits_read: 0,
        };
        let frame = UsacFdChannelDecoder::new(1024, 3)
            .unwrap()
            .parse_after_side(&mut BitReader::new(&payload), side, true, 0)
            .unwrap();
        assert_eq!(frame.quantized_windows.len(), 1);
        assert_eq!(frame.spectrum_windows.len(), 1);
        assert_eq!(frame.spectrum_windows[0].len(), 1024);
        assert!(frame.spectrum_windows[0]
            .iter()
            .all(|sample| sample.is_finite()));
    }
}
