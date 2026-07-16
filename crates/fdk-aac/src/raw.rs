//! Pure Rust AAC raw_data_block top-level syntax helpers.
//!
//! This incremental parser identifies MPEG-4 AAC top-level elements, parses
//! in-band PCEs, and skips DSE/FIL payloads. SCE/CPE/LFE/CCE spectral payloads
//! are variable length and are intentionally left to follow-up modules; when one
//! is encountered this parser returns its element tag and stops at the payload
//! boundary.

use std::fmt;

use crate::asc::{AscError, ProgramConfig};
use crate::bits::{BitError, BitReader};
use crate::ics::{IcsError, IcsInfo, IcsLimits};
use crate::scalefactor::{ScalefactorError, ScalefactorPlan};
use crate::section::{SectionData, SectionError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementId {
    SingleChannel,
    ChannelPair,
    CouplingChannel,
    Lfe,
    DataStream,
    ProgramConfig,
    Fill,
    End,
}

impl ElementId {
    pub fn from_bits(bits: u8) -> Self {
        match bits & 0x07 {
            0 => Self::SingleChannel,
            1 => Self::ChannelPair,
            2 => Self::CouplingChannel,
            3 => Self::Lfe,
            4 => Self::DataStream,
            5 => Self::ProgramConfig,
            6 => Self::Fill,
            _ => Self::End,
        }
    }

    pub fn bits(self) -> u8 {
        match self {
            Self::SingleChannel => 0,
            Self::ChannelPair => 1,
            Self::CouplingChannel => 2,
            Self::Lfe => 3,
            Self::DataStream => 4,
            Self::ProgramConfig => 5,
            Self::Fill => 6,
            Self::End => 7,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawDataBlock {
    pub elements: Vec<RawElement>,
    pub terminated: bool,
    pub bits_read: usize,
}

impl RawDataBlock {
    pub fn parse(input: &[u8]) -> Result<Self, RawError> {
        let mut reader = BitReader::new(input);
        let mut elements = Vec::new();
        let mut terminated = false;

        while reader.remaining_bits() >= 3 {
            let id = ElementId::from_bits(reader.read_u8(3)?);
            match id {
                ElementId::End => {
                    terminated = true;
                    break;
                }
                ElementId::SingleChannel
                | ElementId::ChannelPair
                | ElementId::CouplingChannel
                | ElementId::Lfe => {
                    let tag = reader.read_u8(4)?;
                    elements.push(RawElement::Audio {
                        id,
                        element_instance_tag: tag,
                    });
                    return Ok(Self {
                        elements,
                        terminated,
                        bits_read: reader.bits_read(),
                    });
                }
                ElementId::DataStream => {
                    let element_instance_tag = reader.read_u8(4)?;
                    let byte_align = reader.read_bool()?;
                    let mut count = reader.read_u8(8)? as usize;
                    if count == 255 {
                        count += reader.read_u8(8)? as usize;
                    }
                    if byte_align {
                        reader.byte_align();
                    }
                    skip_bytes(&mut reader, count)?;
                    elements.push(RawElement::DataStream {
                        element_instance_tag,
                        byte_align,
                        byte_count: count,
                    });
                }
                ElementId::ProgramConfig => {
                    let pce = ProgramConfig::parse_from_reader(&mut reader)?;
                    elements.push(RawElement::ProgramConfig(pce));
                }
                ElementId::Fill => {
                    let mut count = reader.read_u8(4)? as usize;
                    if count == 15 {
                        count += reader.read_u8(8)? as usize - 1;
                    }
                    skip_bytes(&mut reader, count)?;
                    elements.push(RawElement::Fill { byte_count: count });
                }
            }
        }

        Ok(Self {
            elements,
            terminated,
            bits_read: reader.bits_read(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SingleChannelElementSideInfo {
    pub id: ElementId,
    pub element_instance_tag: u8,
    pub global_gain: u8,
    pub ics: IcsInfo,
    pub bits_read: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SingleChannelElementSyntaxPrefix {
    pub side_info: SingleChannelElementSideInfo,
    pub section_data: SectionData,
    pub scalefactor_plan: ScalefactorPlan,
    pub bits_read: usize,
}

impl SingleChannelElementSyntaxPrefix {
    /// Parse SCE/LFE through `section_data`, stopping before scalefactors.
    pub fn parse_aac_lc(input: &[u8], limits: IcsLimits) -> Result<Self, RawError> {
        let mut reader = BitReader::new(input);
        let start = reader.bits_read();
        let side_info =
            SingleChannelElementSideInfo::parse_aac_lc_from_reader(&mut reader, limits)?;
        let section_data = SectionData::parse_aac_lc(&mut reader, &side_info.ics)?;
        let scalefactor_plan = ScalefactorPlan::from_section_data(&section_data)?;
        Ok(Self {
            side_info,
            section_data,
            scalefactor_plan,
            bits_read: reader.bits_read() - start,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelPairElementSideInfoPrefix {
    pub element_instance_tag: u8,
    pub common_window: bool,
    pub shared_ics: Option<IcsInfo>,
    pub bits_read: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CouplingChannelElementPrefix {
    pub element_instance_tag: u8,
    pub independently_switched: bool,
    pub targets: Vec<CouplingTarget>,
    pub coupling_domain: bool,
    pub gain_element_sign: bool,
    pub gain_element_scale: u8,
    pub gain_element_lists: usize,
    pub bits_read: usize,
}

impl CouplingChannelElementPrefix {
    /// AAC CCE coupling point derived from `ind_sw_cce_flag` and `cc_domain`.
    /// An independently switched CCE is always applied after the IMDCT; for
    /// ordinary CCEs `cc_domain` selects before or after TNS.
    pub fn coupling_point(&self) -> CouplingPoint {
        if self.independently_switched {
            CouplingPoint::AfterImdct
        } else if self.coupling_domain {
            CouplingPoint::BetweenTnsAndImdct
        } else {
            CouplingPoint::BeforeTns
        }
    }

    pub fn uses_frequency_coupling(&self) -> bool {
        !self.independently_switched
    }

    pub fn uses_time_coupling(&self) -> bool {
        self.independently_switched
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CouplingPoint {
    BeforeTns,
    BetweenTnsAndImdct,
    AfterImdct,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CouplingTarget {
    pub is_cpe: bool,
    pub tag_select: u8,
    pub left: bool,
    pub right: bool,
}

impl ChannelPairElementSideInfoPrefix {
    /// Parse the first CPE side-info fields: `id_syn_ele`, tag, common_window,
    /// and shared ICS when common_window is present.
    ///
    /// This stops before common_max_sfb/MS or per-channel payload so it can be
    /// used as an incremental building block.
    pub fn parse_aac_lc(input: &[u8], limits: IcsLimits) -> Result<Self, RawError> {
        let mut reader = BitReader::new(input);
        Self::parse_aac_lc_from_reader(&mut reader, limits)
    }

    pub fn parse_aac_lc_from_reader(
        reader: &mut BitReader<'_>,
        limits: IcsLimits,
    ) -> Result<Self, RawError> {
        let start = reader.bits_read();
        let id = ElementId::from_bits(reader.read_u8(3)?);
        if id != ElementId::ChannelPair {
            return Err(RawError::UnexpectedElementForChannelPair(id));
        }
        let element_instance_tag = reader.read_u8(4)?;
        let common_window = reader.read_bool()?;
        let shared_ics = if common_window {
            Some(IcsInfo::parse_aac_lc(reader, limits)?)
        } else {
            None
        };

        Ok(Self {
            element_instance_tag,
            common_window,
            shared_ics,
            bits_read: reader.bits_read() - start,
        })
    }
}

impl SingleChannelElementSideInfo {
    /// Parse the side information prefix for an SCE or LFE raw_data_block element.
    ///
    /// This consumes `id_syn_ele`, `element_instance_tag`, `global_gain`, and
    /// `ics_info`. It intentionally stops before section/scalefactor/spectral
    /// data so the remaining syntax can be ported incrementally.
    pub fn parse_aac_lc(input: &[u8], limits: IcsLimits) -> Result<Self, RawError> {
        let mut reader = BitReader::new(input);
        Self::parse_aac_lc_from_reader(&mut reader, limits)
    }

    pub fn parse_aac_lc_from_reader(
        reader: &mut BitReader<'_>,
        limits: IcsLimits,
    ) -> Result<Self, RawError> {
        let start = reader.bits_read();
        let id = ElementId::from_bits(reader.read_u8(3)?);
        if id != ElementId::SingleChannel && id != ElementId::Lfe {
            return Err(RawError::UnexpectedElementForSingleChannel(id));
        }
        let element_instance_tag = reader.read_u8(4)?;
        let global_gain = reader.read_u8(8)?;
        let ics = IcsInfo::parse_aac_lc(reader, limits)?;
        if id == ElementId::Lfe && !ics.window_sequence.is_long() {
            return Err(RawError::LfeMayNotUseShortWindow);
        }
        Ok(Self {
            id,
            element_instance_tag,
            global_gain,
            ics,
            bits_read: reader.bits_read() - start,
        })
    }
}

impl CouplingChannelElementPrefix {
    /// Parse the AAC-LC coupling_channel_element() prefix through the coupling
    /// target list and gain element header.
    ///
    /// This intentionally stops before the coupled channel stream spectral
    /// payload and the variable gain_element lists. It is a syntax bridge for
    /// later CCE tool application.
    pub fn parse_aac_lc(input: &[u8]) -> Result<Self, RawError> {
        let mut reader = BitReader::new(input);
        Self::parse_aac_lc_from_reader(&mut reader)
    }

    pub fn parse_aac_lc_from_reader(reader: &mut BitReader<'_>) -> Result<Self, RawError> {
        let start = reader.bits_read();
        let id = ElementId::from_bits(reader.read_u8(3)?);
        if id != ElementId::CouplingChannel {
            return Err(RawError::UnexpectedElementForCouplingChannel(id));
        }

        let element_instance_tag = reader.read_u8(4)?;
        let independently_switched = reader.read_bool()?;
        let num_coupled_elements = reader.read_u8(3)? as usize;
        let mut targets = Vec::with_capacity(num_coupled_elements + 1);
        let mut gain_element_lists = 0usize;
        for _ in 0..=num_coupled_elements {
            gain_element_lists += 1;
            let is_cpe = reader.read_bool()?;
            let tag_select = reader.read_u8(4)?;
            let (left, right) = if is_cpe {
                let left = reader.read_bool()?;
                let right = reader.read_bool()?;
                if left && right {
                    gain_element_lists += 1;
                }
                (left, right)
            } else {
                (true, false)
            };
            targets.push(CouplingTarget {
                is_cpe,
                tag_select,
                left,
                right,
            });
        }
        let coupling_domain = reader.read_bool()?;
        let gain_element_sign = reader.read_bool()?;
        let gain_element_scale = reader.read_u8(2)?;

        Ok(Self {
            element_instance_tag,
            independently_switched,
            targets,
            coupling_domain,
            gain_element_sign,
            gain_element_scale,
            gain_element_lists,
            bits_read: reader.bits_read() - start,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RawElement {
    Audio {
        id: ElementId,
        element_instance_tag: u8,
    },
    DataStream {
        element_instance_tag: u8,
        byte_align: bool,
        byte_count: usize,
    },
    ProgramConfig(ProgramConfig),
    Fill {
        byte_count: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RawError {
    Bit(BitError),
    Asc(AscError),
    Ics(IcsError),
    Section(SectionError),
    Scalefactor(ScalefactorError),
    UnexpectedElementForSingleChannel(ElementId),
    UnexpectedElementForChannelPair(ElementId),
    UnexpectedElementForCouplingChannel(ElementId),
    LfeMayNotUseShortWindow,
}

impl From<BitError> for RawError {
    fn from(value: BitError) -> Self {
        Self::Bit(value)
    }
}

impl From<AscError> for RawError {
    fn from(value: AscError) -> Self {
        Self::Asc(value)
    }
}

impl From<IcsError> for RawError {
    fn from(value: IcsError) -> Self {
        Self::Ics(value)
    }
}

impl From<SectionError> for RawError {
    fn from(value: SectionError) -> Self {
        Self::Section(value)
    }
}

impl From<ScalefactorError> for RawError {
    fn from(value: ScalefactorError) -> Self {
        Self::Scalefactor(value)
    }
}

impl fmt::Display for RawError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bit(err) => err.fmt(f),
            Self::Asc(err) => err.fmt(f),
            Self::Ics(err) => err.fmt(f),
            Self::Section(err) => err.fmt(f),
            Self::Scalefactor(err) => err.fmt(f),
            Self::UnexpectedElementForSingleChannel(id) => {
                write!(f, "expected SCE/LFE element, got {id:?}")
            }
            Self::UnexpectedElementForChannelPair(id) => {
                write!(f, "expected CPE element, got {id:?}")
            }
            Self::UnexpectedElementForCouplingChannel(id) => {
                write!(f, "expected CCE element, got {id:?}")
            }
            Self::LfeMayNotUseShortWindow => write!(f, "AAC LFE element may not use short windows"),
        }
    }
}

impl std::error::Error for RawError {}

fn skip_bytes(reader: &mut BitReader<'_>, count: usize) -> Result<(), BitError> {
    for _ in 0..count {
        reader.read_u8(8)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asc::{ProgramConfig, ProgramElement};
    use crate::bits::BitWriter;
    use crate::ics::{WindowSequence, WindowShape};

    #[test]
    fn parses_end_only_block() {
        let mut writer = BitWriter::new();
        writer.write(ElementId::End.bits() as u32, 3);
        let block = RawDataBlock::parse(&writer.finish()).unwrap();
        assert!(block.terminated);
        assert!(block.elements.is_empty());
        assert_eq!(block.bits_read, 3);
    }

    #[test]
    fn parses_fill_then_end() {
        let mut writer = BitWriter::new();
        writer.write(ElementId::Fill.bits() as u32, 3);
        writer.write(2, 4);
        writer.write(0xaa, 8);
        writer.write(0xbb, 8);
        writer.write(ElementId::End.bits() as u32, 3);

        let block = RawDataBlock::parse(&writer.finish()).unwrap();
        assert_eq!(block.elements, vec![RawElement::Fill { byte_count: 2 }]);
        assert!(block.terminated);
    }

    #[test]
    fn parses_pce_then_end() {
        let pce = ProgramConfig {
            element_instance_tag: 0,
            profile: 1,
            sampling_frequency_index: 4,
            front: vec![ProgramElement {
                is_cpe: false,
                tag_select: 0,
            }],
            num_channels: 1,
            num_effective_channels: 1,
            ..ProgramConfig::default()
        };

        let mut writer = BitWriter::new();
        writer.write(ElementId::ProgramConfig.bits() as u32, 3);
        pce.write_to_writer(&mut writer).unwrap();
        writer.write(ElementId::End.bits() as u32, 3);

        let block = RawDataBlock::parse(&writer.finish()).unwrap();
        assert!(block.terminated);
        assert_eq!(block.elements, vec![RawElement::ProgramConfig(pce)]);
    }

    #[test]
    fn stops_at_audio_element_tag_boundary() {
        let mut writer = BitWriter::new();
        writer.write(ElementId::SingleChannel.bits() as u32, 3);
        writer.write(5, 4);
        writer.write(0xffff, 16);

        let block = RawDataBlock::parse(&writer.finish()).unwrap();
        assert_eq!(
            block.elements,
            vec![RawElement::Audio {
                id: ElementId::SingleChannel,
                element_instance_tag: 5,
            }]
        );
        assert!(!block.terminated);
        assert_eq!(block.bits_read, 7);
    }

    #[test]
    fn parses_single_channel_side_info_prefix() {
        let mut writer = BitWriter::new();
        writer.write(ElementId::SingleChannel.bits() as u32, 3);
        writer.write(2, 4); // tag
        writer.write(120, 8); // global_gain
        writer.write_bool(false); // reserved
        writer.write(WindowSequence::OnlyLong.bits() as u32, 2);
        writer.write_bool(false); // sine
        writer.write(40, 6); // max_sfb
        writer.write_bool(false); // predictor absent

        let side =
            SingleChannelElementSideInfo::parse_aac_lc(&writer.finish(), IcsLimits::AAC_LC_MAX)
                .unwrap();
        assert_eq!(side.id, ElementId::SingleChannel);
        assert_eq!(side.element_instance_tag, 2);
        assert_eq!(side.global_gain, 120);
        assert_eq!(side.ics.window_sequence, WindowSequence::OnlyLong);
        assert_eq!(side.ics.window_shape, WindowShape::Sine);
        assert_eq!(side.ics.max_sfb, 40);
        assert_eq!(side.bits_read, 26);
    }

    #[test]
    fn rejects_lfe_short_window_side_info() {
        let mut writer = BitWriter::new();
        writer.write(ElementId::Lfe.bits() as u32, 3);
        writer.write(0, 4);
        writer.write(100, 8);
        writer.write_bool(false);
        writer.write(WindowSequence::EightShort.bits() as u32, 2);
        writer.write_bool(false);
        writer.write(10, 4);
        writer.write(0, 7);

        assert_eq!(
            SingleChannelElementSideInfo::parse_aac_lc(&writer.finish(), IcsLimits::AAC_LC_MAX)
                .unwrap_err(),
            RawError::LfeMayNotUseShortWindow
        );
    }

    #[test]
    fn parses_channel_pair_common_window_prefix() {
        let mut writer = BitWriter::new();
        writer.write(ElementId::ChannelPair.bits() as u32, 3);
        writer.write(7, 4);
        writer.write_bool(true); // common_window
        writer.write_bool(false); // reserved
        writer.write(WindowSequence::EightShort.bits() as u32, 2);
        writer.write_bool(true); // kbd
        writer.write(11, 4); // max_sfb
        writer.write(0b000_0000, 7); // eight groups

        let cpe =
            ChannelPairElementSideInfoPrefix::parse_aac_lc(&writer.finish(), IcsLimits::AAC_LC_MAX)
                .unwrap();
        assert_eq!(cpe.element_instance_tag, 7);
        assert!(cpe.common_window);
        let ics = cpe.shared_ics.unwrap();
        assert_eq!(ics.window_sequence, WindowSequence::EightShort);
        assert_eq!(ics.window_group_lengths, vec![1, 1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(cpe.bits_read, 23);
    }

    #[test]
    fn parses_channel_pair_without_common_window_prefix() {
        let mut writer = BitWriter::new();
        writer.write(ElementId::ChannelPair.bits() as u32, 3);
        writer.write(1, 4);
        writer.write_bool(false);

        let cpe =
            ChannelPairElementSideInfoPrefix::parse_aac_lc(&writer.finish(), IcsLimits::AAC_LC_MAX)
                .unwrap();
        assert_eq!(cpe.element_instance_tag, 1);
        assert!(!cpe.common_window);
        assert!(cpe.shared_ics.is_none());
        assert_eq!(cpe.bits_read, 8);
    }

    #[test]
    fn parses_coupling_channel_element_prefix_for_sce_target() {
        let mut writer = BitWriter::new();
        writer.write(ElementId::CouplingChannel.bits() as u32, 3);
        writer.write(2, 4); // element_instance_tag
        writer.write_bool(true); // independently_switched
        writer.write(0, 3); // one coupled element
        writer.write_bool(false); // target is SCE
        writer.write(3, 4); // target tag
        writer.write_bool(true); // coupling_domain
        writer.write_bool(false); // gain_element_sign
        writer.write(2, 2); // gain_element_scale
        let prefix = CouplingChannelElementPrefix::parse_aac_lc(&writer.finish()).unwrap();

        assert_eq!(prefix.element_instance_tag, 2);
        assert!(prefix.independently_switched);
        assert_eq!(prefix.targets.len(), 1);
        assert_eq!(prefix.targets[0].tag_select, 3);
        assert!(!prefix.targets[0].is_cpe);
        assert_eq!(prefix.targets[0].left, true);
        assert_eq!(prefix.targets[0].right, false);
        assert_eq!(prefix.gain_element_lists, 1);
        assert!(prefix.coupling_domain);
        assert!(!prefix.gain_element_sign);
        assert_eq!(prefix.gain_element_scale, 2);
        assert_eq!(prefix.coupling_point(), CouplingPoint::AfterImdct);
        assert!(prefix.uses_time_coupling());
    }

    #[test]
    fn parses_coupling_channel_element_prefix_for_cpe_target_channels() {
        let mut writer = BitWriter::new();
        writer.write(ElementId::CouplingChannel.bits() as u32, 3);
        writer.write(1, 4); // element_instance_tag
        writer.write_bool(false); // independently_switched
        writer.write(0, 3); // one coupled element
        writer.write_bool(true); // target is CPE
        writer.write(4, 4); // target tag
        writer.write_bool(true); // left
        writer.write_bool(true); // right
        writer.write_bool(false); // coupling_domain
        writer.write_bool(true); // gain_element_sign
        writer.write(1, 2); // gain_element_scale
        let prefix = CouplingChannelElementPrefix::parse_aac_lc(&writer.finish()).unwrap();

        assert_eq!(prefix.targets.len(), 1);
        assert!(prefix.targets[0].is_cpe);
        assert_eq!(prefix.targets[0].tag_select, 4);
        assert!(prefix.targets[0].left);
        assert!(prefix.targets[0].right);
        assert_eq!(prefix.gain_element_lists, 2);
        assert!(!prefix.coupling_domain);
        assert!(prefix.gain_element_sign);
        assert_eq!(prefix.gain_element_scale, 1);
        assert_eq!(prefix.coupling_point(), CouplingPoint::BeforeTns);
        assert!(prefix.uses_frequency_coupling());
    }

    #[test]
    fn parses_single_channel_prefix_through_section_data() {
        let mut writer = BitWriter::new();
        writer.write(ElementId::SingleChannel.bits() as u32, 3);
        writer.write(0, 4);
        writer.write(100, 8);
        writer.write_bool(false);
        writer.write(WindowSequence::OnlyLong.bits() as u32, 2);
        writer.write_bool(false);
        writer.write(4, 6);
        writer.write_bool(false);
        writer.write(1, 4); // section cb
        writer.write(4, 5); // section len covers max_sfb

        let prefix =
            SingleChannelElementSyntaxPrefix::parse_aac_lc(&writer.finish(), IcsLimits::AAC_LC_MAX)
                .unwrap();
        assert_eq!(prefix.side_info.global_gain, 100);
        assert_eq!(prefix.section_data.codebooks, vec![vec![1, 1, 1, 1]]);
        assert_eq!(prefix.bits_read, 35);
    }

    #[test]
    fn parses_aligned_extended_data_stream_and_extended_fill() {
        let mut writer = BitWriter::new();
        writer.write(ElementId::DataStream.bits() as u32, 3);
        writer.write(9, 4);
        writer.write_bool(true);
        writer.write(255, 8);
        writer.write(2, 8); // 257 bytes total
        writer.byte_align();
        for value in 0..257 {
            writer.write((value & 0xff) as u32, 8);
        }
        writer.write(ElementId::Fill.bits() as u32, 3);
        writer.write(15, 4);
        writer.write(1, 8); // 15 + 1 - 1 = 15 bytes
        for _ in 0..15 {
            writer.write(0xaa, 8);
        }
        writer.write(ElementId::End.bits() as u32, 3);
        let block = RawDataBlock::parse(&writer.finish()).unwrap();
        assert_eq!(
            block.elements,
            [
                RawElement::DataStream {
                    element_instance_tag: 9,
                    byte_align: true,
                    byte_count: 257,
                },
                RawElement::Fill { byte_count: 15 },
            ]
        );
        assert!(block.terminated);
    }

    #[test]
    fn raw_payload_skip_propagates_eof() {
        let mut writer = BitWriter::new();
        writer.write(ElementId::DataStream.bits() as u32, 3);
        writer.write(0, 4);
        writer.write_bool(false);
        writer.write(2, 8);
        writer.write(0xaa, 8);
        let bits = writer.bits_written();
        let bytes = writer.finish();
        assert!(matches!(
            RawDataBlock::parse(&bytes[..bits.div_ceil(8)]),
            Err(RawError::Bit(BitError::UnexpectedEof { .. }))
        ));
    }

    #[test]
    fn prefix_parsers_reject_wrong_element_types() {
        let bytes = [0xe0]; // ID_END followed by padding
        assert_eq!(
            ChannelPairElementSideInfoPrefix::parse_aac_lc(&bytes, IcsLimits::AAC_LC_MAX),
            Err(RawError::UnexpectedElementForChannelPair(ElementId::End))
        );
        assert_eq!(
            SingleChannelElementSideInfo::parse_aac_lc(&bytes, IcsLimits::AAC_LC_MAX),
            Err(RawError::UnexpectedElementForSingleChannel(ElementId::End))
        );
        assert_eq!(
            CouplingChannelElementPrefix::parse_aac_lc(&bytes),
            Err(RawError::UnexpectedElementForCouplingChannel(
                ElementId::End
            ))
        );
    }

    #[test]
    fn coupling_points_and_element_ids_cover_all_variants() {
        for bits in 0..8 {
            let id = ElementId::from_bits(bits);
            assert_eq!(id.bits(), bits);
        }
        let prefix = CouplingChannelElementPrefix {
            element_instance_tag: 0,
            independently_switched: false,
            targets: Vec::new(),
            coupling_domain: true,
            gain_element_sign: false,
            gain_element_scale: 0,
            gain_element_lists: 0,
            bits_read: 0,
        };
        assert_eq!(prefix.coupling_point(), CouplingPoint::BetweenTnsAndImdct);
        assert!(prefix.uses_frequency_coupling());
        assert!(!prefix.uses_time_coupling());
    }

    #[test]
    fn converts_and_formats_all_raw_error_classes() {
        let bit = BitError::UnexpectedEof {
            needed_bits: 1,
            remaining_bits: 0,
        };
        let errors = [
            RawError::from(bit.clone()),
            RawError::from(AscError::InvalidAudioObjectType(0)),
            RawError::from(IcsError::PredictionUnsupported),
            RawError::from(SectionError::InvalidCodebook(12)),
            RawError::from(ScalefactorError::RaggedCodebookGrid),
            RawError::UnexpectedElementForSingleChannel(ElementId::End),
            RawError::UnexpectedElementForChannelPair(ElementId::End),
            RawError::UnexpectedElementForCouplingChannel(ElementId::End),
            RawError::LfeMayNotUseShortWindow,
        ];
        for error in errors {
            assert!(!error.to_string().is_empty());
        }
        assert_eq!(RawError::from(bit.clone()), RawError::Bit(bit));
    }
}
