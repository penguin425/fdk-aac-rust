//! Top-level mono USAC core decoder dispatching FD and LPD frames.

use crate::asc::{
    usac_channel_count, usac_element_layout_matches, AscError, Mps212Config, UsacConfig,
    UsacElementConfig,
};
use crate::audio_preroll::AudioPreRollError;
use crate::bits::{BitError, BitReader};
use crate::drc::DrcError;
use crate::ld_sbr::{LdSbrError, LdSbrFrequencyTables};
use crate::ld_sbr_qmf::{LdSbrChannelProcessor, LdSbrProcessingError, LdSbrQmfAnalysis, QmfSlot};
use crate::sbr::{
    ParsedUsacSbrFrame, ParsedUsacSbrStereoFrame, SbrError, UsacSbrMonoParser, UsacSbrPayloadFrame,
    UsacSbrStereoParser,
};
use crate::tns::TnsData;
use crate::usac::{UsacIcsInfo, UsacWindowSequence};
use crate::usac_fd::{UsacFdChannelDecoder, UsacFdError};
use crate::usac_lpd::{LpdError, UsacLpdAccessUnitDecoder};
use crate::usac_mps::Mps212QmfProcessor;
use crate::usac_mps::{Mps212Frame, Mps212FrameDecoder, MpsError};
use crate::usac_stereo::UsacStereoData;

#[derive(Debug, Clone, PartialEq)]
pub struct UsacDecodedFrame {
    pub samples: Vec<f32>,
    pub independent: bool,
    pub lpd: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UsacDecodeError {
    AudioPreRoll(AudioPreRollError),
    Asc(AscError),
    Bit(BitError),
    Drc(DrcError),
    Fd(UsacFdError),
    Lpd(LpdError),
    Mps(MpsError),
    Sbr(SbrError),
    SbrProcessing(LdSbrProcessingError),
    UnsupportedConfiguration,
}

impl From<AudioPreRollError> for UsacDecodeError {
    fn from(v: AudioPreRollError) -> Self {
        Self::AudioPreRoll(v)
    }
}

impl From<AscError> for UsacDecodeError {
    fn from(v: AscError) -> Self {
        Self::Asc(v)
    }
}

impl From<BitError> for UsacDecodeError {
    fn from(v: BitError) -> Self {
        Self::Bit(v)
    }
}
impl From<DrcError> for UsacDecodeError {
    fn from(v: DrcError) -> Self {
        Self::Drc(v)
    }
}
impl From<UsacFdError> for UsacDecodeError {
    fn from(v: UsacFdError) -> Self {
        Self::Fd(v)
    }
}
impl From<LpdError> for UsacDecodeError {
    fn from(v: LpdError) -> Self {
        Self::Lpd(v)
    }
}
impl From<MpsError> for UsacDecodeError {
    fn from(v: MpsError) -> Self {
        Self::Mps(v)
    }
}
impl From<SbrError> for UsacDecodeError {
    fn from(v: SbrError) -> Self {
        Self::Sbr(v)
    }
}
impl From<LdSbrProcessingError> for UsacDecodeError {
    fn from(v: LdSbrProcessingError) -> Self {
        Self::SbrProcessing(v)
    }
}
impl From<LdSbrError> for UsacDecodeError {
    fn from(v: LdSbrError) -> Self {
        Self::SbrProcessing(v.into())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct UsacMps212AccessUnit {
    pub downmix: UsacDecodedFrame,
    pub residual: Option<Vec<f32>>,
    pub sbr: Option<ParsedUsacSbr>,
    pub spatial: Mps212Frame,
    pub bits_read: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParsedUsacSbr {
    Mono(ParsedUsacSbrFrame),
    Stereo(ParsedUsacSbrStereoFrame),
}

#[derive(Debug, Clone)]
pub struct UsacMultichannelDecoder {
    elements: Vec<UsacCoreElementDecoder>,
    channels: usize,
}

#[derive(Debug, Clone)]
enum UsacCoreElementDecoder {
    Mono(UsacMonoDecoder),
    Stereo(UsacStereoDecoder),
    Mps212(UsacMps212Decoder),
}

impl UsacMultichannelDecoder {
    pub fn new(config: UsacConfig) -> Result<Self, UsacDecodeError> {
        let mut elements = Vec::new();
        let mut channels = 0usize;
        for element in &config.elements {
            let mut element_config = config.clone();
            element_config.elements = vec![element.clone()];
            match element {
                UsacElementConfig::SingleChannel { .. } | UsacElementConfig::Lfe { .. } => {
                    element_config.channel_configuration_index = 1;
                    elements.push(UsacCoreElementDecoder::Mono(UsacMonoDecoder::new(
                        element_config,
                    )?));
                    channels += 1;
                }
                UsacElementConfig::ChannelPair {
                    stereo_config_index: 1 | 2 | 3,
                    mps212: Some(_),
                    ..
                } => {
                    element_config.channel_configuration_index = 2;
                    elements.push(UsacCoreElementDecoder::Mps212(UsacMps212Decoder::new(
                        element_config,
                    )?));
                    channels += 2;
                }
                UsacElementConfig::ChannelPair { .. } => {
                    element_config.channel_configuration_index = 2;
                    elements.push(UsacCoreElementDecoder::Stereo(UsacStereoDecoder::new(
                        element_config,
                    )?));
                    channels += 2;
                }
                UsacElementConfig::Extension(_) => {
                    return Err(UsacDecodeError::UnsupportedConfiguration);
                }
            }
        }
        if elements.len() < 2
            || Some(channels) != usac_channel_count(config.channel_configuration_index)
            || !usac_element_layout_matches(config.channel_configuration_index, &config.elements)
        {
            return Err(UsacDecodeError::UnsupportedConfiguration);
        }
        Ok(Self { elements, channels })
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn element_count(&self) -> usize {
        self.elements.len()
    }

    pub fn decode_element_after_independent(
        &mut self,
        index: usize,
        reader: &mut BitReader<'_>,
        independent: bool,
    ) -> Result<Vec<Vec<f32>>, UsacDecodeError> {
        let element = self
            .elements
            .get_mut(index)
            .ok_or(UsacDecodeError::UnsupportedConfiguration)?;
        match element {
            UsacCoreElementDecoder::Mono(decoder) => Ok(vec![
                decoder
                    .decode_after_independent(reader, independent)?
                    .samples,
            ]),
            UsacCoreElementDecoder::Stereo(decoder) => Ok(decoder
                .decode_after_independent(reader, independent)?
                .into_iter()
                .collect()),
            UsacCoreElementDecoder::Mps212(decoder) => {
                let access_unit = decoder.decode_after_independent(reader, independent)?;
                Ok(decoder
                    .render_access_unit(access_unit)?
                    .into_iter()
                    .collect())
            }
        }
    }

    pub fn decode_after_independent(
        &mut self,
        reader: &mut BitReader<'_>,
        independent: bool,
    ) -> Result<Vec<Vec<f32>>, UsacDecodeError> {
        let mut channels = Vec::with_capacity(self.channels);
        for index in 0..self.elements.len() {
            channels.extend(self.decode_element_after_independent(index, reader, independent)?);
        }
        Ok(channels)
    }
}

#[derive(Debug, Clone)]
pub struct UsacMps212Decoder {
    core: UsacMpsCore,
    spatial: Mps212FrameDecoder,
    analysis: LdSbrQmfAnalysis,
    renderer: Mps212QmfProcessor,
    residual_analysis: Option<LdSbrQmfAnalysis>,
    residual_bands: usize,
    sampling_frequency: u32,
    sbr_time_step: u8,
    sbr_parser: Option<UsacMpsSbrParser>,
    sbr_processors: Option<[LdSbrChannelProcessor; 2]>,
}

#[derive(Debug, Clone)]
enum UsacMpsCore {
    Mono(UsacMonoDecoder),
    Residual(UsacStereoDecoder),
}

#[derive(Debug, Clone)]
enum UsacMpsSbrParser {
    Mono(UsacSbrMonoParser),
    Stereo(UsacSbrStereoParser),
}

impl UsacMps212Decoder {
    pub fn new(config: UsacConfig) -> Result<Self, UsacDecodeError> {
        let (noise_filling, sbr_config, stereo_config_index, mps) = match config.elements.as_slice()
        {
            [UsacElementConfig::ChannelPair {
                noise_filling,
                sbr,
                stereo_config_index: stereo_config_index @ (1 | 2 | 3),
                mps212: Some(mps),
            }] => (
                *noise_filling,
                sbr.clone(),
                *stereo_config_index,
                mps.clone(),
            ),
            _ => return Err(UsacDecodeError::UnsupportedConfiguration),
        };
        let core = if stereo_config_index == 1 {
            let mut mono = config.clone();
            mono.channel_configuration_index = 1;
            mono.elements = vec![UsacElementConfig::SingleChannel {
                noise_filling,
                sbr: None,
            }];
            UsacMpsCore::Mono(UsacMonoDecoder::new(mono)?)
        } else {
            let mut stereo = config.clone();
            stereo.elements = vec![UsacElementConfig::ChannelPair {
                noise_filling,
                sbr: None,
                stereo_config_index: 0,
                mps212: None,
            }];
            UsacMpsCore::Residual(UsacStereoDecoder::new(stereo)?)
        };
        Ok(Self {
            core,
            spatial: mps_frame_decoder(&config, &mps),
            analysis: LdSbrQmfAnalysis::new_with_channels(64).map_err(MpsError::Qmf)?,
            renderer: Mps212QmfProcessor::new(
                usize::from(mps.frequency_resolution_bands),
                mps.decorrelation_config,
            )?,
            residual_analysis: (stereo_config_index > 1)
                .then(|| LdSbrQmfAnalysis::new_with_channels(64).map_err(MpsError::Qmf))
                .transpose()?,
            residual_bands: usize::from(mps.residual_bands.unwrap_or(0)),
            sampling_frequency: config.sampling_frequency,
            sbr_time_step: if config.sbr_ratio_index == 1 { 4 } else { 2 },
            sbr_parser: sbr_config
                .clone()
                .map(|sbr| {
                    if stereo_config_index == 3 {
                        UsacSbrStereoParser::new(sbr, config.sampling_frequency)
                            .map(UsacMpsSbrParser::Stereo)
                    } else {
                        UsacSbrMonoParser::new(sbr, config.sampling_frequency)
                            .map(UsacMpsSbrParser::Mono)
                    }
                })
                .transpose()?,
            sbr_processors: sbr_config
                .map(|_| {
                    Ok::<_, LdSbrProcessingError>([
                        LdSbrChannelProcessor::new_usac(
                            config.sampling_frequency,
                            config.sbr_ratio_index,
                            0x1234,
                        )?,
                        LdSbrChannelProcessor::new_usac(
                            config.sampling_frequency,
                            config.sbr_ratio_index,
                            0x5678,
                        )?,
                    ])
                })
                .transpose()?,
        })
    }

    pub fn decode_access_unit(
        &mut self,
        bytes: &[u8],
    ) -> Result<UsacMps212AccessUnit, UsacDecodeError> {
        let mut reader = BitReader::new(bytes);
        let independent = reader.read_bool()?;
        self.decode_after_independent(&mut reader, independent)
    }

    pub fn decode_after_independent(
        &mut self,
        reader: &mut BitReader<'_>,
        independent: bool,
    ) -> Result<UsacMps212AccessUnit, UsacDecodeError> {
        let (downmix, residual) = match &mut self.core {
            UsacMpsCore::Mono(core) => (core.decode_after_independent(reader, independent)?, None),
            UsacMpsCore::Residual(core) => {
                let channels = core.decode_after_independent(reader, independent)?;
                (
                    UsacDecodedFrame {
                        samples: channels[0].clone(),
                        independent,
                        lpd: false,
                    },
                    Some(channels[1].clone()),
                )
            }
        };
        let sbr = self
            .sbr_parser
            .as_mut()
            .map(|parser| match parser {
                UsacMpsSbrParser::Mono(parser) => parser
                    .parse(reader, downmix.independent)
                    .map(ParsedUsacSbr::Mono),
                UsacMpsSbrParser::Stereo(parser) => parser
                    .parse(reader, downmix.independent)
                    .map(ParsedUsacSbr::Stereo),
            })
            .transpose()?;
        let spatial = self.spatial.parse(reader, downmix.independent)?;
        Ok(UsacMps212AccessUnit {
            downmix,
            residual,
            sbr,
            spatial,
            bits_read: reader.bits_read(),
        })
    }

    pub fn decode_and_render_access_unit(
        &mut self,
        bytes: &[u8],
    ) -> Result<[Vec<f32>; 2], UsacDecodeError> {
        let access_unit = self.decode_access_unit(bytes)?;
        self.render_access_unit(access_unit)
    }

    pub fn render_access_unit(
        &mut self,
        access_unit: UsacMps212AccessUnit,
    ) -> Result<[Vec<f32>; 2], UsacDecodeError> {
        let downmix: Vec<_> = access_unit
            .downmix
            .samples
            .iter()
            .map(|&sample| f64::from(sample))
            .collect();
        let (qmf, sbr_residual_qmf): (Vec<QmfSlot>, Option<Vec<QmfSlot>>) = match &access_unit.sbr {
            Some(ParsedUsacSbr::Mono(sbr)) => {
                let processors = self
                    .sbr_processors
                    .as_mut()
                    .ok_or(UsacDecodeError::UnsupportedConfiguration)?;
                let qmf = match &sbr.payload {
                    UsacSbrPayloadFrame::Ordinary(frame) => processors[0]
                        .process_usac_mono_to_qmf(&downmix, frame, self.sbr_time_step)?,
                    UsacSbrPayloadFrame::Pvc(frame) => {
                        let tables = LdSbrFrequencyTables::from_header(
                            &sbr.active_header,
                            self.sampling_frequency,
                        )?;
                        processors[0].process_usac_pvc_to_qmf(
                            &downmix,
                            &sbr.active_header,
                            &tables,
                            frame,
                            sbr.info.pvc_mode,
                        )?
                    }
                };
                (qmf, None)
            }
            Some(ParsedUsacSbr::Stereo(sbr)) => {
                let residual = access_unit
                    .residual
                    .as_ref()
                    .ok_or(UsacDecodeError::UnsupportedConfiguration)?;
                let residual: Vec<_> = residual.iter().map(|&sample| f64::from(sample)).collect();
                let processors = self
                    .sbr_processors
                    .as_mut()
                    .ok_or(UsacDecodeError::UnsupportedConfiguration)?;
                let (left, right) = processors.split_at_mut(1);
                let qmf = left[0].process_usac_stereo_channel_to_qmf(
                    &downmix,
                    &sbr.payload,
                    false,
                    self.sbr_time_step,
                )?;
                let residual_qmf = right[0].process_usac_stereo_channel_to_qmf(
                    &residual,
                    &sbr.payload,
                    true,
                    self.sbr_time_step,
                )?;
                (qmf, Some(residual_qmf))
            }
            None => (
                self.analysis
                    .process_frame(&downmix)
                    .map_err(MpsError::Qmf)?,
                None,
            ),
        };
        let (left, right) = if let Some(residual) = &access_unit.residual {
            let residual_qmf = if let Some(residual_qmf) = sbr_residual_qmf {
                residual_qmf
            } else {
                let residual: Vec<_> = residual.iter().map(|&sample| f64::from(sample)).collect();
                self.residual_analysis
                    .as_mut()
                    .ok_or(UsacDecodeError::UnsupportedConfiguration)?
                    .process_frame(&residual)
                    .map_err(MpsError::Qmf)?
            };
            self.renderer.process_qmf_with_residual(
                &qmf,
                &residual_qmf,
                self.residual_bands,
                &access_unit.spatial,
            )?
        } else {
            self.renderer.process_qmf(&qmf, &access_unit.spatial)?
        };
        Ok([
            left.into_iter().map(|sample| sample as f32).collect(),
            right.into_iter().map(|sample| sample as f32).collect(),
        ])
    }
}

fn mps_frame_decoder(config: &UsacConfig, mps: &Mps212Config) -> Mps212FrameDecoder {
    let time_slots = if config.core_sbr_frame_length_index == 4 {
        64
    } else {
        32
    };
    Mps212FrameDecoder::new(
        time_slots,
        usize::from(mps.frequency_resolution_bands),
        usize::from(mps.ott_bands_phase.unwrap_or(0)),
        mps.high_rate_mode,
        mps.phase_coding,
    )
    .with_temporal_shape_config(mps.temporal_shape_config)
}

#[derive(Debug, Clone)]
pub struct UsacMonoDecoder {
    noise_filling: bool,
    lfe: bool,
    fd: UsacFdChannelDecoder,
    lpd: Option<UsacLpdAccessUnitDecoder>,
    sbr_parser: Option<UsacSbrMonoParser>,
    sbr_processor: Option<LdSbrChannelProcessor>,
    sampling_frequency: u32,
    sbr_time_step: u8,
}

#[derive(Debug, Clone)]
pub struct UsacStereoDecoder {
    noise_filling: bool,
    fd: [UsacFdChannelDecoder; 2],
    lpd: [UsacLpdAccessUnitDecoder; 2],
    previous_downmix: Option<Vec<f32>>,
    sbr_parser: Option<UsacSbrStereoParser>,
    sbr_processors: Option<[LdSbrChannelProcessor; 2]>,
    sbr_time_step: u8,
}

impl UsacStereoDecoder {
    pub fn new(config: UsacConfig) -> Result<Self, UsacDecodeError> {
        let (noise_filling, sbr) = match config.elements.as_slice() {
            [UsacElementConfig::ChannelPair {
                noise_filling,
                sbr,
                stereo_config_index: 0,
                mps212: None,
            }] => (*noise_filling, sbr.clone()),
            _ => return Err(UsacDecodeError::UnsupportedConfiguration),
        };
        let make_lpd = || {
            let mut mono = config.clone();
            mono.channel_configuration_index = 1;
            mono.elements = vec![UsacElementConfig::SingleChannel {
                noise_filling,
                sbr: None,
            }];
            UsacLpdAccessUnitDecoder::new(mono).map_err(UsacDecodeError::Lpd)
        };
        let make_fd = || {
            UsacFdChannelDecoder::new(
                usize::from(config.core_frame_length),
                config.sampling_frequency_index,
            )
            .map_err(UsacDecodeError::Fd)
        };
        let sbr_parser = sbr
            .clone()
            .map(|sbr| UsacSbrStereoParser::new(sbr, config.sampling_frequency))
            .transpose()?;
        let sbr_processors = sbr
            .map(|_| {
                Ok::<_, LdSbrProcessingError>([
                    LdSbrChannelProcessor::new_usac(
                        config.sampling_frequency,
                        config.sbr_ratio_index,
                        0x1234,
                    )?,
                    LdSbrChannelProcessor::new_usac(
                        config.sampling_frequency,
                        config.sbr_ratio_index,
                        0x5678,
                    )?,
                ])
            })
            .transpose()?;
        Ok(Self {
            noise_filling,
            fd: [make_fd()?, make_fd()?],
            lpd: [make_lpd()?, make_lpd()?],
            previous_downmix: None,
            sbr_parser,
            sbr_processors,
            sbr_time_step: if config.sbr_ratio_index == 1 { 4 } else { 2 },
        })
    }

    pub fn decode_access_unit(&mut self, bytes: &[u8]) -> Result<[Vec<f32>; 2], UsacDecodeError> {
        let mut reader = BitReader::new(bytes);
        Ok(self.decode_from_reader_with_info(&mut reader)?.0)
    }

    pub fn decode_from_reader_with_info(
        &mut self,
        reader: &mut BitReader<'_>,
    ) -> Result<([Vec<f32>; 2], bool), UsacDecodeError> {
        let independent = reader.read_bool()?;
        Ok((
            self.decode_after_independent(reader, independent)?,
            independent,
        ))
    }

    pub fn decode_after_independent(
        &mut self,
        reader: &mut BitReader<'_>,
        independent: bool,
    ) -> Result<[Vec<f32>; 2], UsacDecodeError> {
        let mut channels = self.decode_core_after_independent(reader, independent)?;
        let Some(parser) = self.sbr_parser.as_mut() else {
            return Ok(channels);
        };
        let parsed = parser.parse(reader, independent)?;
        let processors = self
            .sbr_processors
            .as_mut()
            .ok_or(UsacDecodeError::UnsupportedConfiguration)?;
        for channel in 0..2 {
            let core = channels[channel]
                .iter()
                .map(|&sample| f64::from(sample))
                .collect::<Vec<_>>();
            let qmf = processors[channel].process_usac_stereo_channel_to_qmf(
                &core,
                &parsed.payload,
                channel == 1,
                self.sbr_time_step,
            )?;
            channels[channel] = processors[channel]
                .synthesize_qmf(&qmf)?
                .into_iter()
                .map(|sample| sample as f32)
                .collect();
        }
        Ok(channels)
    }

    fn decode_core_after_independent(
        &mut self,
        reader: &mut BitReader<'_>,
        independent: bool,
    ) -> Result<[Vec<f32>; 2], UsacDecodeError> {
        let modes = [reader.read_bool()?, reader.read_bool()?];
        if modes == [false, false] {
            return self.decode_both_fd(reader, independent);
        }
        let mut channels = [Vec::new(), Vec::new()];
        for channel in 0..2 {
            channels[channel] = if modes[channel] {
                self.lpd[channel].decode_from_reader(reader, independent)?
            } else {
                let frame = self.fd[channel].parse(reader, self.noise_filling, independent)?;
                self.fd[channel].render(&frame)
            };
        }
        Ok(channels)
    }

    fn decode_both_fd(
        &mut self,
        reader: &mut BitReader<'_>,
        independent: bool,
    ) -> Result<[Vec<f32>; 2], UsacDecodeError> {
        let tns_active = reader.read_bool()?;
        let common_window = reader.read_bool()?;
        let (long_count, short_count) = self.fd[0].sfb_counts();
        let (mut left, mut right, stereo) = if common_window {
            let left_ics = UsacIcsInfo::parse(reader, long_count, short_count)
                .map_err(|_| UsacDecodeError::UnsupportedConfiguration)?;
            let common_max_sfb = reader.read_bool()?;
            let mut right_ics = left_ics.clone();
            if !common_max_sfb {
                right_ics.max_sfb = reader.read_u8(
                    if left_ics.window_sequence == UsacWindowSequence::EightShort {
                        4
                    } else {
                        6
                    },
                )?;
                let maximum = if left_ics.window_sequence == UsacWindowSequence::EightShort {
                    short_count
                } else {
                    long_count
                };
                if right_ics.max_sfb > maximum {
                    return Err(UsacDecodeError::UnsupportedConfiguration);
                }
            }
            let stereo = UsacStereoData::parse(
                reader,
                left_ics.window_group_lengths.len(),
                usize::from(left_ics.max_sfb.max(right_ics.max_sfb)),
                independent,
            )
            .map_err(|_| UsacDecodeError::UnsupportedConfiguration)?;
            let _common_tw = reader.read_bool()?;
            let (tns_present, shared_tns) = if tns_active && reader.read_bool()? {
                let _tns_on_lr = reader.read_bool()?;
                let short = left_ics.window_sequence == UsacWindowSequence::EightShort;
                let tns = TnsData::parse_present_usac(
                    reader,
                    short,
                    if short { short_count } else { long_count },
                )
                .map_err(|_| UsacDecodeError::UnsupportedConfiguration)?;
                ([false; 2], Some(tns))
            } else if tns_active {
                (read_tns_channel_flags(reader)?, None)
            } else {
                ([false; 2], None)
            };
            let left_start = reader.bits_read();
            let left_side = self.fd[0].read_side_with_ics(
                reader,
                self.noise_filling,
                left_ics,
                tns_present[0],
            )?;
            let mut left =
                self.fd[0].parse_after_side(reader, left_side, independent, left_start)?;
            let right_start = reader.bits_read();
            let right_side = self.fd[1].read_side_with_ics(
                reader,
                self.noise_filling,
                right_ics,
                tns_present[1],
            )?;
            let mut right =
                self.fd[1].parse_after_side(reader, right_side, independent, right_start)?;
            if let Some(tns) = shared_tns {
                let short = left.side.ics.window_sequence == UsacWindowSequence::EightShort;
                let offsets = self.fd[0].band_offsets(short);
                tns.apply_to_windows_f32(&mut left.spectrum_windows, &offsets)
                    .map_err(|_| UsacDecodeError::UnsupportedConfiguration)?;
                tns.apply_to_windows_f32(&mut right.spectrum_windows, &offsets)
                    .map_err(|_| UsacDecodeError::UnsupportedConfiguration)?;
                left.tns = tns.clone();
                right.tns = tns;
            }
            (left, right, Some(stereo))
        } else {
            let _common_tw = reader.read_bool()?;
            let tns_present = read_individual_cpe_tns_flags(reader, tns_active, false)?;
            let left_start = reader.bits_read();
            let left_side =
                self.read_individual_fd_side(reader, tns_present[0], long_count, short_count)?;
            let left = self.fd[0].parse_after_side(reader, left_side, independent, left_start)?;
            let right_start = reader.bits_read();
            let right_side =
                self.read_individual_fd_side(reader, tns_present[1], long_count, short_count)?;
            let right =
                self.fd[1].parse_after_side(reader, right_side, independent, right_start)?;
            (left, right, None)
        };
        if let Some(stereo) = stereo {
            let short = left.side.ics.window_sequence == UsacWindowSequence::EightShort;
            let offsets = self.fd[0].band_offsets(short);
            if stereo.complex_prediction.is_some() {
                self.previous_downmix = stereo.apply_complex_prediction(
                    &mut left.spectrum_windows,
                    &mut right.spectrum_windows,
                    &offsets,
                    &left.side.ics.window_group_lengths,
                    self.previous_downmix.as_deref(),
                );
            } else {
                stereo.apply_ms(
                    &mut left.spectrum_windows,
                    &mut right.spectrum_windows,
                    &offsets,
                    &left.side.ics.window_group_lengths,
                );
            }
        }
        Ok([self.fd[0].render(&left), self.fd[1].render(&right)])
    }

    fn read_individual_fd_side(
        &self,
        reader: &mut BitReader<'_>,
        tns_data_present: bool,
        long_count: u8,
        short_count: u8,
    ) -> Result<crate::usac::UsacFdChannelSideInfo, UsacDecodeError> {
        let global_gain = reader.read_u8(8)?;
        let noise_level_and_offset = self.noise_filling.then(|| reader.read_u8(8)).transpose()?;
        let ics = UsacIcsInfo::parse(reader, long_count, short_count)
            .map_err(|_| UsacDecodeError::UnsupportedConfiguration)?;
        Ok(crate::usac::UsacFdChannelSideInfo {
            tns_data_present,
            global_gain,
            noise_level_and_offset,
            ics,
            bits_read: 0,
        })
    }
}

fn read_individual_cpe_tns_flags(
    reader: &mut BitReader<'_>,
    active: bool,
    common_window: bool,
) -> Result<[bool; 2], UsacDecodeError> {
    if !active {
        return Ok([false; 2]);
    }
    if common_window && reader.read_bool()? {
        // common_tns carries its filter payload immediately and is handled by
        // the dedicated shared-filter path still being integrated.
        return Err(UsacDecodeError::UnsupportedConfiguration);
    }
    read_tns_channel_flags(reader)
}

fn read_tns_channel_flags(reader: &mut BitReader<'_>) -> Result<[bool; 2], UsacDecodeError> {
    let _tns_on_lr = reader.read_bool()?;
    if reader.read_bool()? {
        Ok([true; 2])
    } else {
        let right = reader.read_bool()?;
        Ok([!right, right])
    }
}

impl UsacMonoDecoder {
    pub fn new(config: UsacConfig) -> Result<Self, UsacDecodeError> {
        let (noise_filling, lfe, sbr) = match config.elements.as_slice() {
            [UsacElementConfig::SingleChannel { noise_filling, sbr }] => {
                (*noise_filling, false, sbr.clone())
            }
            [UsacElementConfig::Lfe { sbr }] => (false, true, sbr.clone()),
            _ => return Err(UsacDecodeError::UnsupportedConfiguration),
        };
        if config.channel_configuration_index != 1 {
            return Err(UsacDecodeError::UnsupportedConfiguration);
        }
        let fd = UsacFdChannelDecoder::new(
            usize::from(config.core_frame_length),
            config.sampling_frequency_index,
        )?;
        let mut core_config = config.clone();
        if !lfe {
            core_config.elements = vec![UsacElementConfig::SingleChannel {
                noise_filling,
                sbr: None,
            }];
        }
        let lpd = (!lfe)
            .then(|| UsacLpdAccessUnitDecoder::new(core_config))
            .transpose()?;
        let sbr_time_step = if config.sbr_ratio_index == 1 { 4 } else { 2 };
        let sbr_parser = sbr
            .clone()
            .map(|sbr| UsacSbrMonoParser::new(sbr, config.sampling_frequency))
            .transpose()?;
        let sbr_processor = sbr
            .map(|_| {
                LdSbrChannelProcessor::new_usac(
                    config.sampling_frequency,
                    config.sbr_ratio_index,
                    0x1234,
                )
            })
            .transpose()?;
        Ok(Self {
            noise_filling,
            lfe,
            fd,
            lpd,
            sbr_parser,
            sbr_processor,
            sampling_frequency: config.sampling_frequency,
            sbr_time_step,
        })
    }

    pub fn decode_access_unit(
        &mut self,
        bytes: &[u8],
    ) -> Result<UsacDecodedFrame, UsacDecodeError> {
        let mut reader = BitReader::new(bytes);
        self.decode_from_reader(&mut reader)
    }

    pub fn decode_from_reader(
        &mut self,
        reader: &mut BitReader<'_>,
    ) -> Result<UsacDecodedFrame, UsacDecodeError> {
        let independent = reader.read_bool()?;
        self.decode_after_independent(reader, independent)
    }

    pub fn decode_after_independent(
        &mut self,
        reader: &mut BitReader<'_>,
        independent: bool,
    ) -> Result<UsacDecodedFrame, UsacDecodeError> {
        if self.lfe {
            let frame = self.fd.parse_lfe(reader, independent)?;
            let mut decoded = UsacDecodedFrame {
                samples: self.fd.render(&frame),
                independent,
                lpd: false,
            };
            self.apply_sbr(reader, &mut decoded)?;
            return Ok(decoded);
        }
        let lpd = reader.read_bool()?;
        let samples = if lpd {
            self.lpd
                .as_mut()
                .expect("non-LFE decoder has an LPD core")
                .decode_from_reader(reader, independent)?
        } else {
            let frame = self.fd.parse(reader, self.noise_filling, independent)?;
            self.fd.render(&frame)
        };
        let mut decoded = UsacDecodedFrame {
            samples,
            independent,
            lpd,
        };
        self.apply_sbr(reader, &mut decoded)?;
        Ok(decoded)
    }

    fn apply_sbr(
        &mut self,
        reader: &mut BitReader<'_>,
        decoded: &mut UsacDecodedFrame,
    ) -> Result<(), UsacDecodeError> {
        let Some(parser) = self.sbr_parser.as_mut() else {
            return Ok(());
        };
        let parsed = parser.parse(reader, decoded.independent)?;
        let core = decoded
            .samples
            .iter()
            .map(|&sample| f64::from(sample))
            .collect::<Vec<_>>();
        let processor = self
            .sbr_processor
            .as_mut()
            .ok_or(UsacDecodeError::UnsupportedConfiguration)?;
        let output = match &parsed.payload {
            UsacSbrPayloadFrame::Ordinary(frame) => {
                processor.process_usac_mono(&core, frame, self.sbr_time_step)?
            }
            UsacSbrPayloadFrame::Pvc(frame) => {
                let tables = LdSbrFrequencyTables::from_header(
                    &parsed.active_header,
                    self.sampling_frequency,
                )?;
                processor.process_usac_pvc(
                    &core,
                    &parsed.active_header,
                    &tables,
                    frame,
                    parsed.info.pvc_mode,
                )?
            }
        };
        decoded.samples = output.into_iter().map(|sample| sample as f32).collect();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asc::{LdSbrHeader, Mps212Config};
    use crate::bits::BitWriter;
    use crate::ld_sbr::{decode_sbr_huffman, SbrHuffmanBook};

    fn mono_config() -> UsacConfig {
        UsacConfig {
            sampling_frequency_index: 3,
            sampling_frequency: 48_000,
            core_sbr_frame_length_index: 1,
            core_frame_length: 1024,
            output_frame_length: 1024,
            sbr_ratio_index: 0,
            channel_configuration_index: 1,
            elements: vec![UsacElementConfig::SingleChannel {
                noise_filling: false,
                sbr: None,
            }],
            extensions: Vec::new(),
        }
    }

    fn stereo_config(noise_filling: bool) -> UsacConfig {
        let mut config = mono_config();
        config.channel_configuration_index = 2;
        config.elements = vec![UsacElementConfig::ChannelPair {
            noise_filling,
            sbr: None,
            stereo_config_index: 0,
            mps212: None,
        }];
        config
    }

    fn write_empty_individual_fd(bits: &mut BitWriter, noise_filling: bool) {
        bits.write(0, 8); // global gain
        if noise_filling {
            bits.write(0, 8);
        }
        bits.write(0, 2); // ONLY_LONG
        bits.write_bool(false); // window shape
        bits.write(0, 6); // max_sfb
        bits.write_bool(false); // no FAC
    }

    fn write_q2_pair(bits: &mut BitWriter) {
        bits.write(0, 2);
        bits.write(0, 2);
        bits.write(0, 8);
        bits.write(0, 8);
    }

    fn write_complete_lpd_channel(bits: &mut BitWriter) {
        bits.write(0, 3); // ACELP core mode
        bits.write(0, 5); // four ACELP20 divisions
        bits.write_bool(false); // BPF disabled
        bits.write_bool(false); // previous frame was not LPD
        bits.write_bool(false); // no externally signalled FAC
        for _ in 0..4 {
            bits.write(0, 2);
            for pitch_bits in [9usize, 6, 9, 6] {
                bits.write(0, pitch_bits);
                bits.write_bool(true);
                bits.write(0, 20);
                bits.write(0, 7);
            }
        }
        bits.write(0, 8); // LPC4 absolute vector
        write_q2_pair(bits);
        for _ in 0..2 {
            bits.write_bool(false);
            bits.write(0, 8);
            write_q2_pair(bits);
        }
        bits.write_bool(false); // LPC1 mode 0
        write_q2_pair(bits);
        bits.write_bool(false); // LPC3 mode 0
        bits.write_bool(false);
        bits.write_bool(false);
    }

    fn sbr_huffman_code(book: SbrHuffmanBook, symbol: i8) -> Option<Vec<bool>> {
        for length in 1..=16 {
            for value in 0..(1u32 << length) {
                let mut writer = BitWriter::new();
                writer.write(value, length);
                let bytes = writer.finish();
                let mut reader = BitReader::new(&bytes);
                if decode_sbr_huffman(&mut reader, book) == Ok(symbol)
                    && reader.bits_read() == length
                {
                    return Some(
                        (0..length)
                            .rev()
                            .map(|bit| value & (1 << bit) != 0)
                            .collect(),
                    );
                }
            }
        }
        None
    }

    fn write_sbr_code(writer: &mut BitWriter, code: &[bool]) {
        for &bit in code {
            writer.write_bool(bit);
        }
    }

    #[test]
    fn constructs_mono_fd_lpd_dispatch_decoder() {
        UsacMonoDecoder::new(mono_config()).unwrap();
        assert_eq!(
            sbr_huffman_code(SbrHuffmanBook::EnvelopeLevel15Time, 127),
            None
        );
    }

    #[test]
    fn rejects_stereo_configuration_in_mono_decoder() {
        let mut config = mono_config();
        config.channel_configuration_index = 2;
        assert_eq!(
            UsacMonoDecoder::new(config).unwrap_err(),
            UsacDecodeError::UnsupportedConfiguration
        );
    }

    #[test]
    fn constructors_reject_wrong_elements_and_invalid_core_parameters() {
        let mut empty = mono_config();
        empty.elements.clear();
        assert_eq!(
            UsacMonoDecoder::new(empty).unwrap_err(),
            UsacDecodeError::UnsupportedConfiguration
        );
        assert_eq!(
            UsacStereoDecoder::new(mono_config()).unwrap_err(),
            UsacDecodeError::UnsupportedConfiguration
        );
        assert_eq!(
            UsacMps212Decoder::new(mono_config()).unwrap_err(),
            UsacDecodeError::UnsupportedConfiguration
        );

        let mut config = mono_config();
        config.core_frame_length = 512;
        assert!(matches!(
            UsacMonoDecoder::new(config),
            Err(UsacDecodeError::Fd(UsacFdError::InvalidFrameLength(512)))
        ));
        let mut config = mono_config();
        config.sampling_frequency_index = 13;
        assert!(matches!(
            UsacMonoDecoder::new(config),
            Err(UsacDecodeError::Fd(UsacFdError::InvalidSamplingIndex(13)))
        ));

        let mut config = mono_config();
        config.channel_configuration_index = 2;
        config.elements = vec![UsacElementConfig::ChannelPair {
            noise_filling: false,
            sbr: None,
            stereo_config_index: 1,
            mps212: Some(Mps212Config {
                frequency_resolution_index: 0,
                frequency_resolution_bands: 6,
                fixed_gain_downmix: 0,
                temporal_shape_config: 0,
                decorrelation_config: 0,
                high_rate_mode: false,
                phase_coding: false,
                ott_bands_phase: None,
                residual_bands: None,
                pseudo_lr: false,
                environment_quantization_mode: None,
            }),
        }];
        assert!(matches!(
            UsacMps212Decoder::new(config),
            Err(UsacDecodeError::Mps(MpsError::InvalidParameterSets))
        ));
    }

    #[test]
    fn mps_slot_selection_and_tns_flag_encodings_cover_all_branches() {
        let mps = Mps212Config {
            frequency_resolution_index: 1,
            frequency_resolution_bands: 28,
            fixed_gain_downmix: 0,
            temporal_shape_config: 0,
            decorrelation_config: 0,
            high_rate_mode: false,
            phase_coding: false,
            ott_bands_phase: None,
            residual_bands: None,
            pseudo_lr: false,
            environment_quantization_mode: None,
        };
        let config = mono_config();
        let _slots32 = mps_frame_decoder(&config, &mps);
        let mut config64 = config;
        config64.core_sbr_frame_length_index = 4;
        let _slots64 = mps_frame_decoder(&config64, &mps);

        assert_eq!(
            read_individual_cpe_tns_flags(&mut BitReader::new(&[]), false, false).unwrap(),
            [false; 2]
        );
        let mut common = BitWriter::new();
        common.write_bool(true);
        assert_eq!(
            read_individual_cpe_tns_flags(&mut BitReader::new(&common.finish()), true, true,),
            Err(UsacDecodeError::UnsupportedConfiguration)
        );
        for (shared, right, expected) in [
            (true, false, [true, true]),
            (false, false, [true, false]),
            (false, true, [false, true]),
        ] {
            let mut flags = BitWriter::new();
            flags.write_bool(false); // tns_on_lr
            flags.write_bool(shared);
            if !shared {
                flags.write_bool(right);
            }
            assert_eq!(
                read_individual_cpe_tns_flags(&mut BitReader::new(&flags.finish()), true, false,)
                    .unwrap(),
                expected
            );
        }
        assert!(matches!(
            read_individual_cpe_tns_flags(&mut BitReader::new(&[]), true, false),
            Err(UsacDecodeError::Bit(BitError::UnexpectedEof { .. }))
        ));
    }

    #[test]
    fn mono_decoder_dispatches_empty_fd_and_propagates_truncation() {
        let mut decoder = UsacMonoDecoder::new(mono_config()).unwrap();
        assert!(matches!(
            decoder.decode_access_unit(&[]),
            Err(UsacDecodeError::Bit(BitError::UnexpectedEof { .. }))
        ));

        let mut bits = BitWriter::new();
        bits.write_bool(true); // independent
        bits.write_bool(false); // FD
        write_empty_individual_fd(&mut bits, false);
        let frame = decoder.decode_access_unit(&bits.finish()).unwrap();
        assert!(frame.independent);
        assert!(!frame.lpd);
        assert_eq!(frame.samples, vec![0.0; 1024]);

        let mut lpd = BitWriter::new();
        lpd.write_bool(true); // independent
        lpd.write_bool(true); // LPD
        write_complete_lpd_channel(&mut lpd);
        let frame = decoder.decode_access_unit(&lpd.finish()).unwrap();
        assert!(frame.independent);
        assert!(frame.lpd);
        assert_eq!(frame.samples.len(), 1024);
        assert!(frame.samples.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn mono_decoder_parses_and_synthesizes_ordinary_usac_sbr() {
        let sbr = crate::asc::UsacSbrConfig {
            harmonic_sbr: false,
            inter_tes: true,
            pvc: false,
            start_frequency: 5,
            stop_frequency: 8,
            frequency_scale: Some(1),
            alter_scale: Some(false),
            noise_bands: Some(2),
            limiter_bands: Some(2),
            limiter_gains: Some(2),
            interpol_frequency: Some(true),
            smoothing_mode: Some(true),
        };
        let mut config = mono_config();
        config.sampling_frequency = 44_100;
        config.core_sbr_frame_length_index = 3;
        config.sbr_ratio_index = 3;
        config.output_frame_length = 2048;
        config.elements = vec![UsacElementConfig::SingleChannel {
            noise_filling: false,
            sbr: Some(sbr),
        }];
        let mut ratio_one_config = config.clone();
        ratio_one_config.core_sbr_frame_length_index = 4;
        ratio_one_config.sbr_ratio_index = 1;
        ratio_one_config.output_frame_length = 4096;
        let mut ratio_two_config = config.clone();
        ratio_two_config.core_sbr_frame_length_index = 2;
        ratio_two_config.sbr_ratio_index = 2;
        ratio_two_config.core_frame_length = 768;
        ratio_two_config.output_frame_length = 2048;
        let header = LdSbrHeader {
            amp_resolution: true,
            start_frequency: 5,
            stop_frequency: 8,
            crossover_band: 2,
            frequency_scale: Some(1),
            alter_scale: Some(false),
            noise_bands: Some(2),
            limiter_bands: Some(2),
            limiter_gains: Some(2),
            interpol_frequency: Some(true),
            smoothing_mode: Some(true),
            ..LdSbrHeader::default()
        };
        let tables = LdSbrFrequencyTables::from_header(&header, 44_100).unwrap();
        let zero = sbr_huffman_code(SbrHuffmanBook::EnvelopeLevel30Frequency, 0).unwrap();
        let mut bits = BitWriter::new();
        bits.write_bool(true); // independent
        bits.write_bool(false); // FD core
        bits.write_bool(false); // no TNS
        write_empty_individual_fd(&mut bits, false);
        bits.write_bool(true); // SBR amp resolution
        bits.write(2, 4); // crossover
        bits.write_bool(false); // preprocessing
        bits.write(0, 2); // ordinary SBR
        bits.write_bool(true); // use default header
        bits.write(0, 2); // FIXFIX
        bits.write(0, 2); // one envelope
        bits.write_bool(true); // high resolution
        for _ in 0..tables.noise_band_count() {
            bits.write(0, 2);
        }
        bits.write(8, 6); // absolute envelope
        for _ in 1..tables.high_band_count() {
            write_sbr_code(&mut bits, &zero);
        }
        bits.write_bool(false); // inter-TES inactive
        bits.write(4, 5); // absolute noise
        for _ in 1..tables.noise_band_count() {
            write_sbr_code(&mut bits, &zero);
        }
        bits.write_bool(false); // no harmonics

        let bytes = bits.finish();
        let frame = UsacMonoDecoder::new(config)
            .unwrap()
            .decode_access_unit(&bytes)
            .unwrap();
        assert_eq!(frame.samples.len(), 2048);
        assert!(frame.samples.iter().all(|sample| sample.is_finite()));

        let frame = UsacMonoDecoder::new(ratio_one_config)
            .unwrap()
            .decode_access_unit(&bytes)
            .unwrap();
        assert_eq!(frame.samples.len(), 4096);
        assert!(frame.samples.iter().all(|sample| sample.is_finite()));

        let frame = UsacMonoDecoder::new(ratio_two_config)
            .unwrap()
            .decode_access_unit(&bytes)
            .unwrap();
        assert_eq!(frame.samples.len(), 2048);
        assert!(frame.samples.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn mono_decoder_accepts_usac_lfe_without_core_mode_or_tns_flag() {
        let mut config = mono_config();
        config.elements = vec![UsacElementConfig::Lfe { sbr: None }];
        let mut decoder = UsacMonoDecoder::new(config).unwrap();

        let mut bits = BitWriter::new();
        bits.write_bool(true); // independent
        bits.write(0, 8); // global_gain follows immediately for LFE
        bits.write(0, 2); // ONLY_LONG
        bits.write_bool(false); // window_shape
        bits.write(0, 6); // max_sfb
        bits.write_bool(false); // no FAC
        let frame = decoder.decode_access_unit(&bits.finish()).unwrap();
        assert!(frame.independent);
        assert!(!frame.lpd);
        assert_eq!(frame.samples, vec![0.0; 1024]);
    }

    #[test]
    fn multichannel_decoder_dispatches_sce_and_lfe_in_asc_order() {
        let mut config = mono_config();
        config.channel_configuration_index = 6;
        config.elements = vec![
            UsacElementConfig::SingleChannel {
                noise_filling: false,
                sbr: None,
            },
            UsacElementConfig::ChannelPair {
                noise_filling: false,
                sbr: None,
                stereo_config_index: 0,
                mps212: None,
            },
            UsacElementConfig::ChannelPair {
                noise_filling: false,
                sbr: None,
                stereo_config_index: 0,
                mps212: None,
            },
            UsacElementConfig::Lfe { sbr: None },
        ];
        let mut decoder = UsacMultichannelDecoder::new(config).unwrap();
        assert_eq!(decoder.channels(), 6);

        let mut bits = BitWriter::new();
        bits.write_bool(false); // SCE FD core mode
        write_empty_individual_fd(&mut bits, false);
        for _ in 0..2 {
            bits.write_bool(false); // left FD core mode
            bits.write_bool(false); // right FD core mode
            bits.write_bool(false); // TNS inactive
            bits.write_bool(false); // individual windows
            bits.write_bool(false); // common_tw
            write_empty_individual_fd(&mut bits, false);
            write_empty_individual_fd(&mut bits, false);
        }
        bits.write(0, 8); // LFE global_gain, with no core mode or TNS flag
        bits.write(0, 2); // ONLY_LONG
        bits.write_bool(false);
        bits.write(0, 6); // max_sfb
        bits.write_bool(false); // no FAC
        let channels = decoder
            .decode_after_independent(&mut BitReader::new(&bits.finish()), true)
            .unwrap();
        assert_eq!(channels, vec![vec![0.0; 1024]; 6]);
    }

    #[test]
    fn constructs_mps212_downmix_and_residual_bit_decoder() {
        let mut config = mono_config();
        config.channel_configuration_index = 2;
        config.elements = vec![UsacElementConfig::ChannelPair {
            noise_filling: false,
            sbr: None,
            stereo_config_index: 1,
            mps212: Some(Mps212Config {
                frequency_resolution_index: 1,
                frequency_resolution_bands: 28,
                fixed_gain_downmix: 0,
                temporal_shape_config: 0,
                decorrelation_config: 0,
                high_rate_mode: false,
                phase_coding: false,
                ott_bands_phase: None,
                residual_bands: None,
                pseudo_lr: false,
                environment_quantization_mode: None,
            }),
        }];
        let mut decoder = UsacMps212Decoder::new(config.clone()).unwrap();
        assert!(matches!(
            decoder.decode_access_unit(&[]),
            Err(UsacDecodeError::Bit(BitError::UnexpectedEof { .. }))
        ));
        assert!(matches!(
            decoder.decode_and_render_access_unit(&[]),
            Err(UsacDecodeError::Bit(BitError::UnexpectedEof { .. }))
        ));

        let mut bits = BitWriter::new();
        bits.write_bool(true); // independent core frame
        bits.write_bool(false); // FD core
        bits.write_bool(false); // TNS absent in the mono downmix
        write_empty_individual_fd(&mut bits, false);
        bits.write(0, 2); // default CLD parameter set
        bits.write(0, 2); // default ICC parameter set
        let bytes = bits.finish();
        let mut render_decoder = decoder.clone();
        let access_unit = decoder.decode_access_unit(&bytes).unwrap();
        assert!(access_unit.downmix.independent);
        assert!(access_unit.residual.is_none());
        assert!(access_unit.sbr.is_none());
        assert_eq!(access_unit.spatial.parameter_sets.len(), 1);
        assert_eq!(access_unit.spatial.parameter_sets[0].cld, vec![0; 28]);
        assert_eq!(
            render_decoder.decode_and_render_access_unit(&bytes),
            Err(UsacDecodeError::Mps(MpsError::InvalidParameterSlot))
        );
    }

    #[test]
    fn constructs_mps212_residual_core_decoder() {
        let mut config = mono_config();
        config.channel_configuration_index = 2;
        config.elements = vec![UsacElementConfig::ChannelPair {
            noise_filling: false,
            sbr: None,
            stereo_config_index: 2,
            mps212: Some(Mps212Config {
                frequency_resolution_index: 1,
                frequency_resolution_bands: 28,
                fixed_gain_downmix: 0,
                temporal_shape_config: 0,
                decorrelation_config: 0,
                high_rate_mode: false,
                phase_coding: false,
                ott_bands_phase: None,
                residual_bands: Some(8),
                pseudo_lr: false,
                environment_quantization_mode: None,
            }),
        }];
        let mut decoder = UsacMps212Decoder::new(config).unwrap();
        assert!(matches!(
            decoder.decode_access_unit(&[]),
            Err(UsacDecodeError::Bit(BitError::UnexpectedEof { .. }))
        ));
        assert!(matches!(
            decoder.decode_and_render_access_unit(&[]),
            Err(UsacDecodeError::Bit(BitError::UnexpectedEof { .. }))
        ));

        let mut bits = BitWriter::new();
        bits.write_bool(true); // independent
        bits.write_bool(false); // left FD
        bits.write_bool(false); // right FD
        bits.write_bool(false); // TNS inactive
        bits.write_bool(false); // individual windows
        bits.write_bool(false); // common_tw inactive
        write_empty_individual_fd(&mut bits, false);
        write_empty_individual_fd(&mut bits, false);
        bits.write(0, 2); // default CLD
        bits.write(0, 2); // default ICC
        let bytes = bits.finish();
        let mut render_decoder = decoder.clone();
        let access_unit = decoder.decode_access_unit(&bytes).unwrap();
        assert!(access_unit.downmix.independent);
        assert_eq!(access_unit.downmix.samples.len(), 1024);
        assert_eq!(access_unit.residual.as_ref().unwrap().len(), 1024);
        assert!(access_unit.sbr.is_none());
        assert_eq!(
            render_decoder.decode_and_render_access_unit(&bytes),
            Err(UsacDecodeError::Mps(MpsError::InvalidParameterSlot))
        );
    }

    #[test]
    fn constructs_mps212_with_usac_sbr_qmf_handoff() {
        let mut config = mono_config();
        config.sampling_frequency = 44_100;
        config.core_sbr_frame_length_index = 3;
        config.sbr_ratio_index = 3;
        config.output_frame_length = 2048;
        config.channel_configuration_index = 2;
        config.elements = vec![UsacElementConfig::ChannelPair {
            noise_filling: false,
            sbr: Some(crate::asc::UsacSbrConfig {
                harmonic_sbr: false,
                inter_tes: true,
                pvc: true,
                start_frequency: 5,
                stop_frequency: 8,
                frequency_scale: Some(1),
                alter_scale: Some(false),
                noise_bands: Some(2),
                limiter_bands: Some(2),
                limiter_gains: Some(2),
                interpol_frequency: Some(true),
                smoothing_mode: Some(true),
            }),
            stereo_config_index: 1,
            mps212: Some(Mps212Config {
                frequency_resolution_index: 1,
                frequency_resolution_bands: 28,
                fixed_gain_downmix: 0,
                temporal_shape_config: 0,
                decorrelation_config: 0,
                high_rate_mode: false,
                phase_coding: false,
                ott_bands_phase: None,
                residual_bands: None,
                pseudo_lr: false,
                environment_quantization_mode: None,
            }),
        }];
        let header = LdSbrHeader {
            amp_resolution: true,
            start_frequency: 5,
            stop_frequency: 8,
            crossover_band: 2,
            frequency_scale: Some(1),
            alter_scale: Some(false),
            noise_bands: Some(2),
            limiter_bands: Some(2),
            limiter_gains: Some(2),
            interpol_frequency: Some(true),
            smoothing_mode: Some(true),
            ..LdSbrHeader::default()
        };
        let tables = LdSbrFrequencyTables::from_header(&header, 44_100).unwrap();
        let zero = sbr_huffman_code(SbrHuffmanBook::EnvelopeLevel30Frequency, 0).unwrap();
        let mut decoder = UsacMps212Decoder::new(config.clone()).unwrap();
        let mut bits = BitWriter::new();
        bits.write_bool(true); // independent core frame
        bits.write_bool(false); // FD core
        bits.write_bool(false); // TNS absent
        write_empty_individual_fd(&mut bits, false);
        bits.write_bool(true); // SBR amp resolution
        bits.write(2, 4); // crossover
        bits.write_bool(false); // preprocessing
        bits.write(0, 2); // ordinary SBR rather than PVC
        bits.write_bool(true); // use default SBR header
        bits.write(0, 2); // FIXFIX
        bits.write(0, 2); // one envelope
        bits.write_bool(true); // high frequency resolution
        for _ in 0..tables.noise_band_count() {
            bits.write(0, 2); // inverse filtering off
        }
        bits.write(8, 6); // absolute envelope
        for _ in 1..tables.high_band_count() {
            write_sbr_code(&mut bits, &zero);
        }
        bits.write_bool(false); // inter-TES inactive
        bits.write(4, 5); // absolute noise
        for _ in 1..tables.noise_band_count() {
            write_sbr_code(&mut bits, &zero);
        }
        bits.write_bool(false); // no harmonics
        bits.write(0, 2); // default CLD
        bits.write(0, 2); // default ICC
        let bytes = bits.finish();
        let mut render_decoder = decoder.clone();
        let access_unit = decoder.decode_access_unit(&bytes).unwrap();
        assert!(matches!(access_unit.sbr, Some(ParsedUsacSbr::Mono(_))));
        assert_eq!(access_unit.spatial.parameter_sets.len(), 1);
        let rendered = render_decoder
            .decode_and_render_access_unit(&bytes)
            .unwrap();
        assert_eq!(rendered[0].len(), 2048);
        assert_eq!(rendered[1].len(), 2048);
        assert!(rendered.iter().flatten().all(|sample| sample.is_finite()));

        let mut pvc = BitWriter::new();
        pvc.write_bool(true); // independent core frame
        pvc.write_bool(false); // FD core
        pvc.write_bool(false); // TNS absent
        write_empty_individual_fd(&mut pvc, false);
        pvc.write_bool(true); // SBR amp resolution
        pvc.write(2, 4); // crossover
        pvc.write_bool(false); // preprocessing
        pvc.write(1, 2); // PVC mode 1
        pvc.write_bool(true); // use default SBR header
        pvc.write(0, 4); // noise position: one envelope
        pvc.write_bool(false); // fixed HF end
        for _ in 0..tables.noise_band_count() {
            pvc.write(2, 2); // inverse filtering mode
        }
        pvc.write(0, 3); // PVC division mode
        pvc.write_bool(false); // noise shaping mode
        pvc.write(37, 7); // first PVC ID
        pvc.write(5, 5); // absolute noise
        for _ in 1..tables.noise_band_count() {
            write_sbr_code(&mut pvc, &zero);
        }
        pvc.write_bool(false); // no harmonics
        pvc.write(0, 2); // default CLD
        pvc.write(0, 2); // default ICC
        let pvc = pvc.finish();
        let mut decoder = UsacMps212Decoder::new(config.clone()).unwrap();
        let access_unit = decoder.decode_access_unit(&pvc).unwrap();
        assert!(matches!(
            access_unit.sbr,
            Some(ParsedUsacSbr::Mono(ParsedUsacSbrFrame {
                payload: UsacSbrPayloadFrame::Pvc(_),
                ..
            }))
        ));
        let mut decoder = decoder.clone();
        let rendered = decoder.decode_and_render_access_unit(&pvc).unwrap();
        assert_eq!(rendered[0].len(), 2048);
        assert_eq!(rendered[1].len(), 2048);
        assert!(rendered.iter().flatten().all(|sample| sample.is_finite()));

        let mut invalid_tables = UsacMps212Decoder::new(config.clone()).unwrap();
        invalid_tables.sampling_frequency = 1;
        assert!(matches!(
            invalid_tables.decode_and_render_access_unit(&pvc),
            Err(UsacDecodeError::SbrProcessing(_))
        ));
    }

    #[test]
    fn constructs_mps212_stereo_sbr_residual_qmf_handoff() {
        let mut config = mono_config();
        config.sampling_frequency = 44_100;
        config.core_sbr_frame_length_index = 3;
        config.sbr_ratio_index = 3;
        config.output_frame_length = 2048;
        config.channel_configuration_index = 2;
        config.elements = vec![UsacElementConfig::ChannelPair {
            noise_filling: false,
            sbr: Some(crate::asc::UsacSbrConfig {
                harmonic_sbr: false,
                inter_tes: true,
                pvc: false,
                start_frequency: 5,
                stop_frequency: 8,
                frequency_scale: Some(1),
                alter_scale: Some(false),
                noise_bands: Some(2),
                limiter_bands: Some(2),
                limiter_gains: Some(2),
                interpol_frequency: Some(true),
                smoothing_mode: Some(true),
            }),
            stereo_config_index: 3,
            mps212: Some(Mps212Config {
                frequency_resolution_index: 1,
                frequency_resolution_bands: 28,
                fixed_gain_downmix: 0,
                temporal_shape_config: 0,
                decorrelation_config: 0,
                high_rate_mode: false,
                phase_coding: false,
                ott_bands_phase: None,
                residual_bands: Some(8),
                pseudo_lr: false,
                environment_quantization_mode: None,
            }),
        }];
        let header = LdSbrHeader {
            amp_resolution: true,
            start_frequency: 5,
            stop_frequency: 8,
            crossover_band: 2,
            frequency_scale: Some(1),
            alter_scale: Some(false),
            noise_bands: Some(2),
            limiter_bands: Some(2),
            limiter_gains: Some(2),
            interpol_frequency: Some(true),
            smoothing_mode: Some(true),
            ..LdSbrHeader::default()
        };
        let tables = LdSbrFrequencyTables::from_header(&header, 44_100).unwrap();
        let zero = sbr_huffman_code(SbrHuffmanBook::EnvelopeLevel30Frequency, 0).unwrap();
        let mut bits = BitWriter::new();
        bits.write_bool(true); // independent core frame
        bits.write_bool(false); // left FD
        bits.write_bool(false); // right FD
        bits.write_bool(false); // TNS inactive
        bits.write_bool(false); // individual windows
        bits.write_bool(false); // common_tw inactive
        write_empty_individual_fd(&mut bits, false);
        write_empty_individual_fd(&mut bits, false);

        bits.write_bool(true); // SBR amp resolution
        bits.write(2, 4); // crossover
        bits.write_bool(false); // preprocessing
        bits.write_bool(true); // use default SBR header
        bits.write_bool(false); // uncoupled stereo SBR
        for _ in 0..2 {
            bits.write(0, 2); // FIXFIX
            bits.write(0, 2); // one envelope
            bits.write_bool(true); // high frequency resolution
        }
        for channel in 0..2 {
            for _ in 0..tables.noise_band_count() {
                bits.write(channel + 1, 2);
            }
        }
        for (absolute, mode) in [(9, 1), (11, 3)] {
            bits.write(absolute, 6);
            for _ in 1..tables.high_band_count() {
                write_sbr_code(&mut bits, &zero);
            }
            bits.write_bool(true); // inter-TES active
            bits.write(mode, 2);
        }
        for absolute in [5, 7] {
            bits.write(absolute, 5);
            for _ in 1..tables.noise_band_count() {
                write_sbr_code(&mut bits, &zero);
            }
        }
        bits.write_bool(false); // no left harmonics
        bits.write_bool(false); // no right harmonics
        bits.write(0, 2); // default CLD
        bits.write(0, 2); // default ICC
        let bytes = bits.finish();

        let mut ordinary_stereo_config = config.clone();
        if let UsacElementConfig::ChannelPair {
            stereo_config_index,
            mps212,
            ..
        } = &mut ordinary_stereo_config.elements[0]
        {
            *stereo_config_index = 0;
            *mps212 = None;
        }
        let ordinary = UsacStereoDecoder::new(ordinary_stereo_config)
            .unwrap()
            .decode_access_unit(&bytes)
            .unwrap();
        assert_eq!(ordinary[0].len(), 2048);
        assert_eq!(ordinary[1].len(), 2048);
        assert!(ordinary.iter().flatten().all(|sample| sample.is_finite()));

        let mut decoder = UsacMps212Decoder::new(config.clone()).unwrap();
        let access_unit = decoder.decode_access_unit(&bytes).unwrap();
        assert!(matches!(access_unit.sbr, Some(ParsedUsacSbr::Stereo(_))));
        assert_eq!(access_unit.residual.as_ref().unwrap().len(), 1024);

        let mut decoder = decoder.clone();
        let rendered = decoder.decode_and_render_access_unit(&bytes).unwrap();
        assert_eq!(rendered[0].len(), 2048);
        assert_eq!(rendered[1].len(), 2048);
        assert!(rendered.iter().flatten().all(|sample| sample.is_finite()));
    }

    #[test]
    fn constructs_stereo_lpd_mixed_core_decoder() {
        UsacStereoDecoder::new(stereo_config(true)).unwrap();
    }

    #[test]
    fn decodes_non_common_window_empty_fd_channels_with_noise_side_info() {
        let mut bits = BitWriter::new();
        bits.write_bool(true);
        bits.write_bool(false);
        bits.write_bool(false);
        bits.write_bool(false); // tns inactive
        bits.write_bool(false); // individual windows
        bits.write_bool(false); // common_tw
        write_empty_individual_fd(&mut bits, true);
        write_empty_individual_fd(&mut bits, true);
        let channels = UsacStereoDecoder::new(stereo_config(true))
            .unwrap()
            .decode_access_unit(&bits.finish())
            .unwrap();
        assert_eq!(channels, [vec![0.0; 1024], vec![0.0; 1024]]);
    }

    #[test]
    fn decodes_complete_stereo_lpd_channels() {
        let mut bits = BitWriter::new();
        bits.write_bool(true); // independent
        bits.write_bool(true); // left LPD
        bits.write_bool(true); // right LPD
        write_complete_lpd_channel(&mut bits);
        write_complete_lpd_channel(&mut bits);
        let mut decoder = UsacStereoDecoder::new(stereo_config(false)).unwrap();
        let (channels, independent) = decoder
            .decode_from_reader_with_info(&mut BitReader::new(&bits.finish()))
            .unwrap();
        assert!(independent);
        assert_eq!(channels[0].len(), 1024);
        assert_eq!(channels[1].len(), 1024);
        assert!(channels.iter().flatten().all(|sample| sample.is_finite()));
    }

    #[test]
    fn decodes_common_window_empty_short_fd_channels() {
        let mut bits = BitWriter::new();
        bits.write_bool(true); // independent
        bits.write_bool(false); // left FD
        bits.write_bool(false); // right FD
        bits.write_bool(false); // TNS inactive
        bits.write_bool(true); // common window
        bits.write(2, 2); // EIGHT_SHORT
        bits.write_bool(false); // window shape
        bits.write(0, 4); // left max_sfb
        bits.write(0, 7); // eight separate window groups
        bits.write_bool(false); // distinct right max_sfb
        bits.write(0, 4); // right max_sfb
        bits.write(0, 2); // no M/S
        bits.write_bool(false); // common_tw
        for _ in 0..2 {
            bits.write(0, 8); // global gain
            bits.write_bool(false); // no FAC
        }
        let channels = UsacStereoDecoder::new(stereo_config(false))
            .unwrap()
            .decode_access_unit(&bits.finish())
            .unwrap();
        assert_eq!(channels, [vec![0.0; 1024], vec![0.0; 1024]]);
    }

    #[test]
    fn common_window_complex_prediction_with_no_bands_is_stateful() {
        let mut bits = BitWriter::new();
        bits.write_bool(true);
        bits.write_bool(false);
        bits.write_bool(false);
        bits.write_bool(false); // tns inactive
        bits.write_bool(true); // common window
        bits.write(0, 2);
        bits.write_bool(false);
        bits.write(0, 6);
        bits.write_bool(true); // common max_sfb
        bits.write(3, 2); // complex prediction mask
        bits.write_bool(true); // all bands (there are zero)
        bits.write_bool(false); // mid predicts side
        bits.write_bool(false); // real coefficients
        bits.write_bool(false); // common_tw
        for _ in 0..2 {
            bits.write(0, 8);
            bits.write_bool(false);
        }
        let mut decoder = UsacStereoDecoder::new(stereo_config(false)).unwrap();
        let channels = decoder.decode_access_unit(&bits.finish()).unwrap();
        assert_eq!(channels, [vec![0.0; 1024], vec![0.0; 1024]]);
        assert_eq!(decoder.previous_downmix, Some(vec![0.0; 1024]));
    }

    #[test]
    fn rejects_common_window_right_max_sfb_above_table_limit() {
        let mut bits = BitWriter::new();
        bits.write_bool(true);
        bits.write_bool(false);
        bits.write_bool(false);
        bits.write_bool(false);
        bits.write_bool(true);
        bits.write(0, 2);
        bits.write_bool(false);
        bits.write(0, 6);
        bits.write_bool(false); // distinct right max_sfb
        bits.write(63, 6);
        assert_eq!(
            UsacStereoDecoder::new(stereo_config(false))
                .unwrap()
                .decode_access_unit(&bits.finish()),
            Err(UsacDecodeError::UnsupportedConfiguration)
        );
    }

    #[test]
    fn common_window_propagates_truncation_at_right_max_sfb_and_each_channel_side() {
        let common_prefix = || {
            let mut bits = BitWriter::new();
            bits.write_bool(true); // independent
            bits.write_bool(false); // left FD
            bits.write_bool(false); // right FD
            bits.write_bool(false); // TNS inactive
            bits.write_bool(true); // common window
            bits.write(0, 2); // ONLY_LONG
            bits.write_bool(false); // window shape
            bits.write(0, 6); // left max_sfb
            bits.write_bool(false); // distinct right max_sfb
            bits
        };

        let bits = common_prefix();
        let bit_len = bits.bits_written();
        let bytes = bits.finish();
        let mut decoder = UsacStereoDecoder::new(stereo_config(false)).unwrap();
        assert!(matches!(
            decoder.decode_from_reader_with_info(
                &mut BitReader::with_bit_len(&bytes, bit_len).unwrap()
            ),
            Err(UsacDecodeError::Bit(BitError::UnexpectedEof { .. }))
        ));

        let mut bits = common_prefix();
        bits.write(0, 6); // right max_sfb
        bits.write(0, 2); // no M/S stereo
        bits.write_bool(false); // common_tw
        let bit_len = bits.bits_written();
        let bytes = bits.finish();
        let mut decoder = UsacStereoDecoder::new(stereo_config(false)).unwrap();
        assert!(matches!(
            decoder.decode_from_reader_with_info(
                &mut BitReader::with_bit_len(&bytes, bit_len).unwrap()
            ),
            Err(UsacDecodeError::Fd(UsacFdError::Bit(
                BitError::UnexpectedEof { .. }
            )))
        ));

        let mut bits = common_prefix();
        bits.write(0, 6); // right max_sfb
        bits.write(0, 2); // no M/S stereo
        bits.write_bool(false); // common_tw
        bits.write(0, 8); // complete left global gain
        bits.write_bool(false); // no left FAC
        let bit_len = bits.bits_written();
        let bytes = bits.finish();
        let mut decoder = UsacStereoDecoder::new(stereo_config(false)).unwrap();
        assert!(matches!(
            decoder.decode_from_reader_with_info(
                &mut BitReader::with_bit_len(&bytes, bit_len).unwrap()
            ),
            Err(UsacDecodeError::Fd(UsacFdError::Bit(
                BitError::UnexpectedEof { .. }
            )))
        ));
    }

    #[test]
    fn reads_all_individual_tns_flag_encodings() {
        assert_eq!(
            read_individual_cpe_tns_flags(&mut BitReader::new(&[]), false, false).unwrap(),
            [false; 2]
        );
        assert_eq!(
            read_tns_channel_flags(&mut BitReader::new(&[0b0100_0000])).unwrap(),
            [true; 2]
        );
        assert_eq!(
            read_tns_channel_flags(&mut BitReader::new(&[0b0010_0000])).unwrap(),
            [false, true]
        );
        assert_eq!(
            read_tns_channel_flags(&mut BitReader::new(&[0])).unwrap(),
            [true, false]
        );
        assert_eq!(
            read_individual_cpe_tns_flags(&mut BitReader::new(&[0x80]), true, true),
            Err(UsacDecodeError::UnsupportedConfiguration)
        );
    }

    #[test]
    fn mixed_fd_lpd_dispatch_propagates_each_channel_error() {
        let mut decoder = UsacStereoDecoder::new(stereo_config(false)).unwrap();
        // Left enters LPD immediately and reports its missing side information.
        assert!(matches!(
            decoder.decode_access_unit(&[0b1100_0000]),
            Err(UsacDecodeError::Lpd(_))
        ));

        // Left FD is complete and rendered before the truncated right LPD.
        let mut bits = BitWriter::new();
        bits.write_bool(true);
        bits.write_bool(false);
        bits.write_bool(true);
        write_empty_individual_fd(&mut bits, false);
        assert!(matches!(
            decoder.decode_access_unit(&bits.finish()),
            Err(UsacDecodeError::Lpd(_))
        ));
    }

    #[test]
    fn top_level_error_conversions_preserve_the_source_variant() {
        let bit = BitError::UnexpectedEof {
            needed_bits: 2,
            remaining_bits: 1,
        };
        assert_eq!(
            UsacDecodeError::from(bit.clone()),
            UsacDecodeError::Bit(bit)
        );
        assert_eq!(
            UsacDecodeError::from(UsacFdError::InvalidFrameLength(512)),
            UsacDecodeError::Fd(UsacFdError::InvalidFrameLength(512))
        );
        assert_eq!(
            UsacDecodeError::from(LpdError::InvalidCoreLength(512)),
            UsacDecodeError::Lpd(LpdError::InvalidCoreLength(512))
        );
        assert_eq!(
            UsacDecodeError::from(MpsError::InvalidDataMode),
            UsacDecodeError::Mps(MpsError::InvalidDataMode)
        );
        assert_eq!(
            UsacDecodeError::from(SbrError::InvalidGrid),
            UsacDecodeError::Sbr(SbrError::InvalidGrid)
        );
        assert_eq!(
            UsacDecodeError::from(LdSbrProcessingError::MissingRightChannel),
            UsacDecodeError::SbrProcessing(LdSbrProcessingError::MissingRightChannel)
        );
        assert_eq!(
            UsacDecodeError::from(LdSbrError::InvalidFrequencyRange),
            UsacDecodeError::SbrProcessing(LdSbrProcessingError::Syntax(
                LdSbrError::InvalidFrequencyRange,
            ))
        );
    }

    #[test]
    fn decodes_both_fd_common_window_empty_spectrum() {
        let mut config = mono_config();
        config.channel_configuration_index = 2;
        config.elements = vec![UsacElementConfig::ChannelPair {
            noise_filling: false,
            sbr: None,
            stereo_config_index: 0,
            mps212: None,
        }];
        let mut bits = BitWriter::new();
        bits.write_bool(true); // independent
        bits.write_bool(false); // left FD
        bits.write_bool(false); // right FD
        bits.write_bool(false); // tns_active
        bits.write_bool(true); // common_window
        bits.write(0, 2); // ONLY_LONG
        bits.write_bool(false); // window shape
        bits.write(0, 6); // max_sfb=0
        bits.write_bool(true); // common max_sfb
        bits.write(0, 2); // no M/S
        bits.write_bool(false); // common_tw
        for _ in 0..2 {
            bits.write(0, 8); // global gain
            bits.write_bool(false); // no FAC; no SF/spectral bits for max_sfb=0
        }
        let channels = UsacStereoDecoder::new(config)
            .unwrap()
            .decode_access_unit(&bits.finish())
            .unwrap();
        assert_eq!(channels[0], vec![0.0; 1024]);
        assert_eq!(channels[1], vec![0.0; 1024]);
    }

    #[test]
    fn decodes_both_fd_with_shared_empty_tns_payload() {
        let mut config = mono_config();
        config.channel_configuration_index = 2;
        config.elements = vec![UsacElementConfig::ChannelPair {
            noise_filling: false,
            sbr: None,
            stereo_config_index: 0,
            mps212: None,
        }];
        let mut bits = BitWriter::new();
        bits.write_bool(true);
        bits.write_bool(false);
        bits.write_bool(false);
        bits.write_bool(true); // tns_active
        bits.write_bool(true); // common_window
        bits.write(0, 2);
        bits.write_bool(false);
        bits.write(0, 6);
        bits.write_bool(true);
        bits.write(0, 2);
        bits.write_bool(false); // common_tw
        bits.write_bool(true); // common_tns
        bits.write_bool(false); // tns_on_lr
        bits.write(0, 2); // zero long-window TNS filters
        for _ in 0..2 {
            bits.write(0, 8);
            bits.write_bool(false);
        }
        let channels = UsacStereoDecoder::new(config)
            .unwrap()
            .decode_access_unit(&bits.finish())
            .unwrap();
        assert_eq!(channels[0].len(), 1024);
        assert_eq!(channels[1].len(), 1024);

        let mut bits = BitWriter::new();
        bits.write_bool(true);
        bits.write_bool(false);
        bits.write_bool(false);
        bits.write_bool(true); // tns_active
        bits.write_bool(true); // common_window
        bits.write(0, 2);
        bits.write_bool(false);
        bits.write(0, 6);
        bits.write_bool(true);
        bits.write(0, 2);
        bits.write_bool(false); // common_tw
        bits.write_bool(false); // separate TNS payloads
        bits.write_bool(false); // tns_on_lr
        bits.write_bool(true); // TNS present on both channels
        for _ in 0..2 {
            bits.write(0, 8); // global gain
            bits.write(0, 2); // zero long-window TNS filters
            bits.write_bool(false); // no FAC
        }
        let channels = UsacStereoDecoder::new(stereo_config(false))
            .unwrap()
            .decode_access_unit(&bits.finish())
            .unwrap();
        assert_eq!(channels[0].len(), 1024);
        assert_eq!(channels[1].len(), 1024);
    }
}
