//! AAC fill-element framing for ordinary (non-ELD) SBR payloads.

use std::fmt;

use crate::asc::{AscError, LdSbrHeader, UsacSbrConfig};
use crate::bits::{BitError, BitReader};
use crate::ld_sbr::{
    read_add_harmonics, read_extended_data, read_invf, read_noise, LdSbrChannelControl,
    LdSbrChannelValues, LdSbrDequantizedChannel, LdSbrError, LdSbrFrequencyTables, LdSbrGrid,
    LdSbrPreviousValues,
};
use crate::usac_sbr::{
    HarmonicSbrControl, InterTesEnvelope, PvcEnvelope, UsacPvcGrid, UsacSbrError, UsacSbrFrameInfo,
};

pub const EXT_SBR_DATA: u8 = 13;
pub const EXT_SBR_DATA_CRC: u8 = 14;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SbrFillPayload {
    pub extension_type: u8,
    pub transmitted_crc: Option<u16>,
    pub header_present: bool,
    pub header: Option<LdSbrHeader>,
    /// Frame-data bits after the optional header, packed MSB first.
    pub frame_data: Vec<u8>,
    pub frame_data_bits: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SbrFrameClass {
    FixFix,
    FixVar,
    VarFix,
    VarVar,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SbrGrid {
    pub frame_class: SbrFrameClass,
    pub borders: Vec<u8>,
    pub frequency_resolution: Vec<bool>,
    pub noise_borders: Vec<u8>,
    pub transient_envelope: Option<usize>,
    pub pointer: usize,
}

impl SbrGrid {
    pub fn parse(reader: &mut BitReader<'_>, time_slots: u8) -> Result<Self, SbrError> {
        let frame_class = match reader.read_u8(2)? {
            0 => SbrFrameClass::FixFix,
            1 => SbrFrameClass::FixVar,
            2 => SbrFrameClass::VarFix,
            _ => SbrFrameClass::VarVar,
        };
        let (borders, frequency_resolution, pointer, transient_envelope) = match frame_class {
            SbrFrameClass::FixFix => {
                let envelope_count = 1usize << reader.read_u8(2)?;
                let resolution = reader.read_bool()?;
                let borders = (0..=envelope_count)
                    .map(|index| (index * time_slots as usize / envelope_count) as u8)
                    .collect();
                (borders, vec![resolution; envelope_count], 0, None)
            }
            SbrFrameClass::FixVar => {
                let right = time_slots + reader.read_u8(2)?;
                let relative_count = reader.read_u8(2)? as usize;
                let envelope_count = relative_count + 1;
                let mut borders = vec![0; envelope_count + 1];
                borders[envelope_count] = right;
                let mut border = right as i16;
                for index in (1..envelope_count).rev() {
                    border -= 2 * reader.read_u8(2)? as i16 + 2;
                    if border < 0 {
                        return Err(SbrError::InvalidGrid);
                    }
                    borders[index] = border as u8;
                }
                let pointer = reader.read(pointer_bits(envelope_count))? as usize;
                if pointer > envelope_count {
                    return Err(SbrError::InvalidGrid);
                }
                let transient = (pointer != 0).then_some(envelope_count + 1 - pointer);
                let mut resolution = vec![false; envelope_count];
                for index in (0..envelope_count).rev() {
                    resolution[index] = reader.read_bool()?;
                }
                (borders, resolution, pointer, transient)
            }
            SbrFrameClass::VarFix => {
                let left = reader.read_u8(2)?;
                let relative_count = reader.read_u8(2)? as usize;
                let envelope_count = relative_count + 1;
                let mut borders = vec![0; envelope_count + 1];
                borders[0] = left;
                let mut border = left;
                for value in borders.iter_mut().take(envelope_count).skip(1) {
                    border += 2 * reader.read_u8(2)? + 2;
                    *value = border;
                }
                borders[envelope_count] = time_slots;
                let pointer = reader.read(pointer_bits(envelope_count))? as usize;
                if pointer > envelope_count {
                    return Err(SbrError::InvalidGrid);
                }
                let transient = (pointer > 1).then(|| pointer - 1);
                let resolution = (0..envelope_count)
                    .map(|_| reader.read_bool())
                    .collect::<Result<Vec<_>, _>>()?;
                (borders, resolution, pointer, transient)
            }
            SbrFrameClass::VarVar => {
                let left = reader.read_u8(2)?;
                let right = time_slots + reader.read_u8(2)?;
                let left_count = reader.read_u8(2)? as usize;
                let right_count = reader.read_u8(2)? as usize;
                let envelope_count = left_count + right_count + 1;
                let mut borders = vec![0; envelope_count + 1];
                borders[0] = left;
                let mut border = left;
                for value in borders.iter_mut().take(left_count + 1).skip(1) {
                    border += 2 * reader.read_u8(2)? + 2;
                    *value = border;
                }
                borders[envelope_count] = right;
                let mut border = right as i16;
                for index in (left_count + 1..envelope_count).rev() {
                    border -= 2 * reader.read_u8(2)? as i16 + 2;
                    if border < 0 {
                        return Err(SbrError::InvalidGrid);
                    }
                    borders[index] = border as u8;
                }
                let pointer = reader.read(pointer_bits(envelope_count))? as usize;
                if pointer > envelope_count {
                    return Err(SbrError::InvalidGrid);
                }
                let transient = (pointer != 0).then_some(envelope_count + 1 - pointer);
                let resolution = (0..envelope_count)
                    .map(|_| reader.read_bool())
                    .collect::<Result<Vec<_>, _>>()?;
                (borders, resolution, pointer, transient)
            }
        };
        if borders.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(SbrError::InvalidGrid);
        }
        let noise_borders = if borders.len() == 2 {
            vec![borders[0], borders[1]]
        } else {
            let envelope_count = borders.len() - 1;
            let middle = match frame_class {
                SbrFrameClass::FixFix => borders[envelope_count / 2],
                SbrFrameClass::FixVar | SbrFrameClass::VarVar => {
                    if pointer <= 1 {
                        borders[envelope_count - 1]
                    } else {
                        borders[transient_envelope.ok_or(SbrError::InvalidGrid)?]
                    }
                }
                SbrFrameClass::VarFix => match pointer {
                    0 => borders[1],
                    1 => borders[envelope_count - 1],
                    _ => borders[transient_envelope.ok_or(SbrError::InvalidGrid)?],
                },
            };
            vec![borders[0], middle, *borders.last().unwrap()]
        };
        Ok(Self {
            frame_class,
            borders,
            frequency_resolution,
            noise_borders,
            transient_envelope,
            pointer,
        })
    }
}

impl From<SbrGrid> for LdSbrGrid {
    fn from(grid: SbrGrid) -> Self {
        Self {
            transient: grid.frame_class != SbrFrameClass::FixFix,
            amp_resolution: None,
            borders: grid.borders,
            frequency_resolution: grid.frequency_resolution,
            transient_envelope: grid.transient_envelope,
            noise_borders: grid.noise_borders,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SbrMonoFrame {
    pub active_header: LdSbrHeader,
    pub frequency_tables: LdSbrFrequencyTables,
    pub data_extra: Option<u8>,
    pub control: LdSbrChannelControl,
    pub values: LdSbrChannelValues,
    pub dequantized: LdSbrDequantizedChannel,
    pub harmonics: Vec<bool>,
    pub extended_data: Vec<u8>,
    pub bits_read: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UsacSbrMonoFrame {
    pub frame: SbrMonoFrame,
    pub harmonic_control: Option<HarmonicSbrControl>,
    pub inter_tes: Vec<InterTesEnvelope>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UsacPvcSbrFrame {
    pub grid: UsacPvcGrid,
    pub envelope: PvcEnvelope,
    pub inverse_filtering_modes: Vec<u8>,
    pub noise: Vec<Vec<i16>>,
    pub harmonics: Vec<bool>,
    pub harmonic_control: Option<HarmonicSbrControl>,
    pub bits_read: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UsacSbrPayloadFrame {
    Ordinary(UsacSbrMonoFrame),
    Pvc(UsacPvcSbrFrame),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParsedUsacSbrFrame {
    pub info: UsacSbrFrameInfo,
    pub active_header: LdSbrHeader,
    pub payload: UsacSbrPayloadFrame,
    pub bits_read: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParsedUsacSbrStereoFrame {
    pub info: UsacSbrFrameInfo,
    pub active_header: LdSbrHeader,
    pub payload: UsacSbrStereoFrame,
    pub bits_read: usize,
}

#[derive(Debug, Clone)]
pub struct UsacSbrMonoParser {
    config: UsacSbrConfig,
    default_header: LdSbrHeader,
    active_header: LdSbrHeader,
    frame_parser: SbrMonoFrameParser,
}

impl UsacSbrMonoParser {
    pub fn new(config: UsacSbrConfig, sampling_frequency: u32) -> Result<Self, SbrError> {
        let default_header = header_from_usac_config(&config, true, 0);
        let frame_parser =
            SbrMonoFrameParser::new_usac(default_header.clone(), sampling_frequency)?;
        Ok(Self {
            config,
            active_header: default_header.clone(),
            default_header,
            frame_parser,
        })
    }

    pub fn parse(
        &mut self,
        reader: &mut BitReader<'_>,
        independent: bool,
    ) -> Result<ParsedUsacSbrFrame, SbrError> {
        let start = reader.bits_read();
        let info = UsacSbrFrameInfo::parse(reader, independent, self.config.pvc, false)
            .map_err(SbrError::Usac)?;
        if info.info_present {
            self.active_header.amp_resolution = info.amplitude_resolution.unwrap();
            self.active_header.crossover_band = info.crossover_band.unwrap();
        }
        if info.header_present {
            if reader.read_bool()? {
                self.active_header = self.default_header.clone();
                self.active_header.amp_resolution = info.amplitude_resolution.unwrap();
                self.active_header.crossover_band = info.crossover_band.unwrap();
            } else {
                self.active_header = parse_usac_sbr_header(
                    reader,
                    info.amplitude_resolution.unwrap(),
                    info.crossover_band.unwrap(),
                )?;
            }
            self.frame_parser
                .set_usac_header(self.active_header.clone())?;
        }
        let payload = if info.pvc_mode == 0 {
            UsacSbrPayloadFrame::Ordinary(self.frame_parser.parse_usac(
                reader,
                independent,
                self.config.harmonic_sbr,
                self.config.inter_tes,
            )?)
        } else {
            UsacSbrPayloadFrame::Pvc(self.frame_parser.parse_usac_pvc(
                reader,
                independent,
                info.pvc_mode,
                self.config.harmonic_sbr,
            )?)
        };
        Ok(ParsedUsacSbrFrame {
            info,
            active_header: self.active_header.clone(),
            payload,
            bits_read: reader.bits_read() - start,
        })
    }
}

#[derive(Debug, Clone)]
pub struct UsacSbrStereoParser {
    config: UsacSbrConfig,
    default_header: LdSbrHeader,
    active_header: LdSbrHeader,
    frame_parser: SbrStereoFrameParser,
}

impl UsacSbrStereoParser {
    pub fn new(config: UsacSbrConfig, sampling_frequency: u32) -> Result<Self, SbrError> {
        let default_header = header_from_usac_config(&config, true, 0);
        let frame_parser =
            SbrStereoFrameParser::new_usac(default_header.clone(), sampling_frequency)?;
        Ok(Self {
            config,
            active_header: default_header.clone(),
            default_header,
            frame_parser,
        })
    }

    pub fn parse(
        &mut self,
        reader: &mut BitReader<'_>,
        independent: bool,
    ) -> Result<ParsedUsacSbrStereoFrame, SbrError> {
        let start = reader.bits_read();
        let info = UsacSbrFrameInfo::parse(reader, independent, self.config.pvc, true)
            .map_err(SbrError::Usac)?;
        if info.info_present {
            self.active_header.amp_resolution = info.amplitude_resolution.unwrap();
            self.active_header.crossover_band = info.crossover_band.unwrap();
        }
        if info.header_present {
            if reader.read_bool()? {
                self.active_header = self.default_header.clone();
                self.active_header.amp_resolution = info.amplitude_resolution.unwrap();
                self.active_header.crossover_band = info.crossover_band.unwrap();
            } else {
                self.active_header = parse_usac_sbr_header(
                    reader,
                    info.amplitude_resolution.unwrap(),
                    info.crossover_band.unwrap(),
                )?;
            }
            self.frame_parser
                .set_usac_header(self.active_header.clone())?;
        }
        let payload = self.frame_parser.parse_usac(
            reader,
            independent,
            self.config.harmonic_sbr,
            self.config.inter_tes,
        )?;
        Ok(ParsedUsacSbrStereoFrame {
            info,
            active_header: self.active_header.clone(),
            payload,
            bits_read: reader.bits_read() - start,
        })
    }
}

fn header_from_usac_config(
    config: &UsacSbrConfig,
    amp_resolution: bool,
    crossover_band: u8,
) -> LdSbrHeader {
    LdSbrHeader {
        amp_resolution,
        crossover_band,
        reserved: 0,
        start_frequency: config.start_frequency,
        stop_frequency: config.stop_frequency,
        frequency_scale: config.frequency_scale,
        alter_scale: config.alter_scale,
        noise_bands: config.noise_bands,
        limiter_bands: config.limiter_bands,
        limiter_gains: config.limiter_gains,
        interpol_frequency: config.interpol_frequency,
        smoothing_mode: config.smoothing_mode,
    }
}

fn parse_usac_sbr_header(
    reader: &mut BitReader<'_>,
    amp_resolution: bool,
    crossover_band: u8,
) -> Result<LdSbrHeader, SbrError> {
    let start_frequency = reader.read_u8(4)?;
    let stop_frequency = reader.read_u8(4)?;
    let extra_1 = reader.read_bool()?;
    let extra_2 = reader.read_bool()?;
    let (frequency_scale, alter_scale, noise_bands) = if extra_1 {
        (
            Some(reader.read_u8(2)?),
            Some(reader.read_bool()?),
            Some(reader.read_u8(2)?),
        )
    } else {
        (Some(2), Some(true), Some(2))
    };
    let (limiter_bands, limiter_gains, interpol_frequency, smoothing_mode) = if extra_2 {
        (
            Some(reader.read_u8(2)?),
            Some(reader.read_u8(2)?),
            Some(reader.read_bool()?),
            Some(reader.read_bool()?),
        )
    } else {
        (Some(2), Some(2), Some(true), Some(true))
    };
    Ok(LdSbrHeader {
        amp_resolution,
        crossover_band,
        reserved: 0,
        start_frequency,
        stop_frequency,
        frequency_scale,
        alter_scale,
        noise_bands,
        limiter_bands,
        limiter_gains,
        interpol_frequency,
        smoothing_mode,
    })
}

#[derive(Debug, Clone)]
pub struct SbrMonoFrameParser {
    header: LdSbrHeader,
    sampling_frequency: u32,
    time_slots: u8,
    previous: LdSbrPreviousValues,
    previous_pvc_id: u8,
    previous_pvc_right_border: Option<i8>,
    previous_was_pvc: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SbrStereoFrame {
    pub active_header: LdSbrHeader,
    pub frequency_tables: LdSbrFrequencyTables,
    pub data_extra: Option<(u8, u8)>,
    pub coupling: bool,
    pub left_control: LdSbrChannelControl,
    pub right_control: LdSbrChannelControl,
    pub left: LdSbrChannelValues,
    pub right: LdSbrChannelValues,
    pub left_dequantized: LdSbrDequantizedChannel,
    pub right_dequantized: LdSbrDequantizedChannel,
    pub left_harmonics: Vec<bool>,
    pub right_harmonics: Vec<bool>,
    pub extended_data: Vec<u8>,
    pub bits_read: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UsacSbrStereoFrame {
    pub frame: SbrStereoFrame,
    pub harmonic_controls: [Option<HarmonicSbrControl>; 2],
    pub inter_tes: [Vec<InterTesEnvelope>; 2],
}

#[derive(Debug, Clone)]
pub struct SbrStereoFrameParser {
    header: LdSbrHeader,
    sampling_frequency: u32,
    time_slots: u8,
    previous_left: LdSbrPreviousValues,
    previous_right: LdSbrPreviousValues,
}

fn ordinary_time_slots(core_frame_length: usize) -> Result<u8, SbrError> {
    match core_frame_length {
        960 => Ok(15),
        1024 => Ok(16),
        _ => Err(SbrError::UnsupportedFrameLength(core_frame_length)),
    }
}

fn read_control(
    reader: &mut BitReader<'_>,
    grid: LdSbrGrid,
) -> Result<LdSbrChannelControl, SbrError> {
    let envelope_time_domain = (0..grid.envelope_count())
        .map(|_| reader.read_bool())
        .collect::<Result<Vec<_>, _>>()?;
    let noise_time_domain = (0..grid.noise_envelope_count())
        .map(|_| reader.read_bool())
        .collect::<Result<Vec<_>, _>>()?;
    Ok(LdSbrChannelControl {
        grid,
        envelope_time_domain,
        noise_time_domain,
    })
}

fn read_usac_control(
    reader: &mut BitReader<'_>,
    grid: LdSbrGrid,
    independent: bool,
) -> Result<LdSbrChannelControl, SbrError> {
    let envelope_time_domain = (0..grid.envelope_count())
        .map(|index| {
            if independent && index == 0 {
                Ok(false)
            } else {
                reader.read_bool()
            }
        })
        .collect::<Result<Vec<_>, BitError>>()?;
    let noise_time_domain = (0..grid.noise_envelope_count())
        .map(|index| {
            if independent && index == 0 {
                Ok(false)
            } else {
                reader.read_bool()
            }
        })
        .collect::<Result<Vec<_>, BitError>>()?;
    Ok(LdSbrChannelControl {
        grid,
        envelope_time_domain,
        noise_time_domain,
    })
}

impl SbrMonoFrameParser {
    pub fn new(
        header: LdSbrHeader,
        sampling_frequency: u32,
        core_frame_length: usize,
    ) -> Result<Self, SbrError> {
        let time_slots = ordinary_time_slots(core_frame_length)?;
        LdSbrFrequencyTables::from_header(&header, sampling_frequency)?;
        Ok(Self {
            header,
            sampling_frequency,
            time_slots,
            previous: LdSbrPreviousValues::default(),
            previous_pvc_id: 0,
            previous_pvc_right_border: None,
            previous_was_pvc: false,
        })
    }

    pub fn clear_history(&mut self) {
        self.previous = LdSbrPreviousValues::default();
        self.previous_pvc_id = 0;
        self.previous_pvc_right_border = None;
        self.previous_was_pvc = false;
    }

    pub fn parse(&mut self, payload: &SbrFillPayload) -> Result<SbrMonoFrame, SbrError> {
        let active_header = payload
            .header
            .clone()
            .unwrap_or_else(|| self.header.clone());
        let frequency_tables =
            LdSbrFrequencyTables::from_header(&active_header, self.sampling_frequency)?;
        let mut reader = BitReader::new(&payload.frame_data);
        let start = reader.bits_read();
        let data_extra = reader.read_bool()?.then(|| reader.read_u8(4)).transpose()?;
        let grid = SbrGrid::parse(&mut reader, self.time_slots)?;
        let grid = LdSbrGrid::from(grid);
        let control = read_control(&mut reader, grid)?;
        let mut values = LdSbrChannelValues::parse_mono_after_prefix(
            &mut reader,
            &control,
            &frequency_tables,
            active_header.amp_resolution,
        )?;
        let mut previous = self.previous.clone();
        values.reconstruct_deltas(&control, &frequency_tables, &mut previous)?;
        let dequantized = values.dequantize_uncoupled(&control, active_header.amp_resolution);
        let harmonics = read_add_harmonics(&mut reader, frequency_tables.high_band_count())?;
        let extended_data = read_extended_data(&mut reader)?;
        if reader.bits_read() > payload.frame_data_bits {
            return Err(SbrError::TruncatedFrameData);
        }
        let bits_read = reader.bits_read() - start;
        self.header = active_header.clone();
        self.previous = previous;
        Ok(SbrMonoFrame {
            active_header,
            frequency_tables,
            data_extra,
            control,
            values,
            dequantized,
            harmonics,
            extended_data,
            bits_read,
        })
    }

    pub fn new_usac(header: LdSbrHeader, sampling_frequency: u32) -> Result<Self, SbrError> {
        LdSbrFrequencyTables::from_header(&header, sampling_frequency)?;
        Ok(Self {
            header,
            sampling_frequency,
            time_slots: 16,
            previous: LdSbrPreviousValues::default(),
            previous_pvc_id: 0,
            previous_pvc_right_border: None,
            previous_was_pvc: false,
        })
    }

    pub fn set_usac_header(&mut self, header: LdSbrHeader) -> Result<(), SbrError> {
        LdSbrFrequencyTables::from_header(&header, self.sampling_frequency)?;
        self.header = header;
        Ok(())
    }

    pub fn parse_usac(
        &mut self,
        reader: &mut BitReader<'_>,
        independent: bool,
        harmonic_sbr: bool,
        inter_tes: bool,
    ) -> Result<UsacSbrMonoFrame, SbrError> {
        let start = reader.bits_read();
        let tables = LdSbrFrequencyTables::from_header(&self.header, self.sampling_frequency)?;
        let harmonic_control = harmonic_sbr
            .then(|| HarmonicSbrControl::parse(reader))
            .transpose()?;
        let grid = LdSbrGrid::from(SbrGrid::parse(reader, self.time_slots)?);
        let control = read_usac_control(reader, grid, independent)?;
        let (mut values, inter_tes_envelopes) = LdSbrChannelValues::parse_mono_after_prefix_usac(
            reader,
            &control,
            &tables,
            self.header.amp_resolution,
            inter_tes,
        )?;
        let mut previous = self.previous.clone();
        values.reconstruct_deltas(&control, &tables, &mut previous)?;
        let dequantized = values.dequantize_uncoupled(&control, self.header.amp_resolution);
        let harmonics = read_add_harmonics(reader, tables.high_band_count())?;
        self.previous = previous;
        self.previous_pvc_right_border = control
            .grid
            .borders
            .last()
            .copied()
            .map(|value| value as i8);
        self.previous_was_pvc = false;
        Ok(UsacSbrMonoFrame {
            harmonic_control,
            inter_tes: inter_tes_envelopes,
            frame: SbrMonoFrame {
                active_header: self.header.clone(),
                frequency_tables: tables,
                data_extra: None,
                control,
                values,
                dequantized,
                harmonics,
                extended_data: Vec::new(),
                bits_read: reader.bits_read() - start,
            },
        })
    }

    pub fn parse_usac_pvc(
        &mut self,
        reader: &mut BitReader<'_>,
        independent: bool,
        pvc_mode: u8,
        harmonic_sbr: bool,
    ) -> Result<UsacPvcSbrFrame, SbrError> {
        let start = reader.bits_read();
        let tables = LdSbrFrequencyTables::from_header(&self.header, self.sampling_frequency)?;
        let harmonic_control = harmonic_sbr
            .then(|| HarmonicSbrControl::parse(reader))
            .transpose()?;
        let grid = UsacPvcGrid::parse(
            reader,
            self.previous_pvc_right_border,
            self.previous_was_pvc,
        )
        .map_err(SbrError::Usac)?;
        let noise_count = grid.noise_borders.len() - 1;
        let noise_time_domain = (0..noise_count)
            .map(|index| {
                if independent && index == 0 {
                    Ok(false)
                } else {
                    reader.read_bool()
                }
            })
            .collect::<Result<Vec<_>, BitError>>()?;
        let inverse_filtering_modes = read_invf(reader, tables.noise_band_count())?;
        let envelope = PvcEnvelope::parse(reader, pvc_mode, independent, self.previous_pvc_id)
            .map_err(SbrError::Usac)?;
        let noise_grid = LdSbrGrid {
            transient: false,
            amp_resolution: None,
            borders: grid
                .borders
                .iter()
                .map(|&value| value.max(0) as u8)
                .collect(),
            frequency_resolution: vec![false; grid.borders.len() - 1],
            transient_envelope: None,
            noise_borders: grid
                .noise_borders
                .iter()
                .map(|&value| value.max(0) as u8)
                .collect(),
        };
        let control = LdSbrChannelControl {
            grid: noise_grid,
            envelope_time_domain: Vec::new(),
            noise_time_domain,
        };
        let mut noise = read_noise(reader, &control, &tables, false)?;
        if self.previous.noise.len() != tables.noise_band_count() {
            self.previous.noise.resize(tables.noise_band_count(), 0);
        }
        for index in 0..noise.len() {
            if control.noise_time_domain[index] {
                let reference = if index == 0 {
                    &self.previous.noise
                } else {
                    &noise[index - 1]
                }
                .clone();
                for (value, prior) in noise[index].iter_mut().zip(reference) {
                    *value += prior;
                }
            } else {
                for band in 1..noise[index].len() {
                    noise[index][band] += noise[index][band - 1];
                }
            }
        }
        if let Some(last) = noise.last() {
            self.previous.noise.clone_from(last);
        }
        let harmonics = read_add_harmonics(reader, tables.high_band_count())?;
        self.previous_pvc_id = envelope.ids[15];
        self.previous_pvc_right_border = grid.borders.last().copied();
        self.previous_was_pvc = true;
        Ok(UsacPvcSbrFrame {
            grid,
            envelope,
            inverse_filtering_modes,
            noise,
            harmonics,
            harmonic_control,
            bits_read: reader.bits_read() - start,
        })
    }
}

impl SbrStereoFrameParser {
    pub fn new(
        header: LdSbrHeader,
        sampling_frequency: u32,
        core_frame_length: usize,
    ) -> Result<Self, SbrError> {
        let time_slots = ordinary_time_slots(core_frame_length)?;
        LdSbrFrequencyTables::from_header(&header, sampling_frequency)?;
        Ok(Self {
            header,
            sampling_frequency,
            time_slots,
            previous_left: LdSbrPreviousValues::default(),
            previous_right: LdSbrPreviousValues::default(),
        })
    }

    pub fn clear_history(&mut self) {
        self.previous_left = LdSbrPreviousValues::default();
        self.previous_right = LdSbrPreviousValues::default();
    }

    pub fn parse(&mut self, payload: &SbrFillPayload) -> Result<SbrStereoFrame, SbrError> {
        let active_header = payload
            .header
            .clone()
            .unwrap_or_else(|| self.header.clone());
        let tables = LdSbrFrequencyTables::from_header(&active_header, self.sampling_frequency)?;
        let mut reader = BitReader::new(&payload.frame_data);
        let start = reader.bits_read();
        let data_extra = if reader.read_bool()? {
            Some((reader.read_u8(4)?, reader.read_u8(4)?))
        } else {
            None
        };
        let coupling = reader.read_bool()?;
        let left_grid = LdSbrGrid::from(SbrGrid::parse(&mut reader, self.time_slots)?);
        let right_grid = if coupling {
            left_grid.clone()
        } else {
            LdSbrGrid::from(SbrGrid::parse(&mut reader, self.time_slots)?)
        };
        let left_control = read_control(&mut reader, left_grid)?;
        let right_control = read_control(&mut reader, right_grid)?;
        let prefix = crate::ld_sbr::LdSbrChannelElementPrefix {
            data_extra: data_extra.map(|(left, right)| (left, Some(right))),
            coupling,
            left: left_control.clone(),
            right: Some(right_control.clone()),
        };
        let (mut left, mut right) = LdSbrChannelValues::parse_stereo_after_prefix(
            &mut reader,
            &prefix,
            &tables,
            active_header.amp_resolution,
        )?;
        let mut previous_left = self.previous_left.clone();
        let mut previous_right = self.previous_right.clone();
        left.reconstruct_deltas(&left_control, &tables, &mut previous_left)?;
        right.reconstruct_deltas(&right_control, &tables, &mut previous_right)?;
        let (left_dequantized, right_dequantized) = if coupling {
            LdSbrChannelValues::dequantize_coupled_pair(
                &left,
                &right,
                &left_control,
                active_header.amp_resolution,
            )?
        } else {
            (
                left.dequantize_uncoupled(&left_control, active_header.amp_resolution),
                right.dequantize_uncoupled(&right_control, active_header.amp_resolution),
            )
        };
        let left_harmonics = read_add_harmonics(&mut reader, tables.high_band_count())?;
        let right_harmonics = read_add_harmonics(&mut reader, tables.high_band_count())?;
        let extended_data = read_extended_data(&mut reader)?;
        if reader.bits_read() > payload.frame_data_bits {
            return Err(SbrError::TruncatedFrameData);
        }
        let bits_read = reader.bits_read() - start;
        self.header = active_header.clone();
        self.previous_left = previous_left;
        self.previous_right = previous_right;
        Ok(SbrStereoFrame {
            active_header,
            frequency_tables: tables,
            data_extra,
            coupling,
            left_control,
            right_control,
            left,
            right,
            left_dequantized,
            right_dequantized,
            left_harmonics,
            right_harmonics,
            extended_data,
            bits_read,
        })
    }

    pub fn new_usac(header: LdSbrHeader, sampling_frequency: u32) -> Result<Self, SbrError> {
        LdSbrFrequencyTables::from_header(&header, sampling_frequency)?;
        Ok(Self {
            header,
            sampling_frequency,
            time_slots: 16,
            previous_left: LdSbrPreviousValues::default(),
            previous_right: LdSbrPreviousValues::default(),
        })
    }

    pub fn set_usac_header(&mut self, header: LdSbrHeader) -> Result<(), SbrError> {
        LdSbrFrequencyTables::from_header(&header, self.sampling_frequency)?;
        self.header = header;
        Ok(())
    }

    pub fn parse_usac(
        &mut self,
        reader: &mut BitReader<'_>,
        independent: bool,
        harmonic_sbr: bool,
        inter_tes: bool,
    ) -> Result<UsacSbrStereoFrame, SbrError> {
        let start = reader.bits_read();
        let tables = LdSbrFrequencyTables::from_header(&self.header, self.sampling_frequency)?;
        let coupling = reader.read_bool()?;
        let left_harmonic = harmonic_sbr
            .then(|| HarmonicSbrControl::parse(reader))
            .transpose()?;
        let right_harmonic = if harmonic_sbr && !coupling {
            Some(HarmonicSbrControl::parse(reader)?)
        } else {
            left_harmonic
        };
        let left_grid = LdSbrGrid::from(SbrGrid::parse(reader, self.time_slots)?);
        let right_grid = if coupling {
            left_grid.clone()
        } else {
            LdSbrGrid::from(SbrGrid::parse(reader, self.time_slots)?)
        };
        let left_control = read_usac_control(reader, left_grid, independent)?;
        let right_control = read_usac_control(reader, right_grid, independent)?;
        let prefix = crate::ld_sbr::LdSbrChannelElementPrefix {
            data_extra: None,
            coupling,
            left: left_control.clone(),
            right: Some(right_control.clone()),
        };
        let ((mut left, left_tes), (mut right, right_tes)) =
            LdSbrChannelValues::parse_stereo_after_prefix_usac(
                reader,
                &prefix,
                &tables,
                self.header.amp_resolution,
                inter_tes,
            )?;
        let mut previous_left = self.previous_left.clone();
        let mut previous_right = self.previous_right.clone();
        left.reconstruct_deltas(&left_control, &tables, &mut previous_left)?;
        right.reconstruct_deltas(&right_control, &tables, &mut previous_right)?;
        let (left_dequantized, right_dequantized) = if coupling {
            LdSbrChannelValues::dequantize_coupled_pair(
                &left,
                &right,
                &left_control,
                self.header.amp_resolution,
            )?
        } else {
            (
                left.dequantize_uncoupled(&left_control, self.header.amp_resolution),
                right.dequantize_uncoupled(&right_control, self.header.amp_resolution),
            )
        };
        let left_harmonics = read_add_harmonics(reader, tables.high_band_count())?;
        let right_harmonics = read_add_harmonics(reader, tables.high_band_count())?;
        self.previous_left = previous_left;
        self.previous_right = previous_right;
        Ok(UsacSbrStereoFrame {
            harmonic_controls: [left_harmonic, right_harmonic],
            inter_tes: [left_tes, right_tes],
            frame: SbrStereoFrame {
                active_header: self.header.clone(),
                frequency_tables: tables,
                data_extra: None,
                coupling,
                left_control,
                right_control,
                left,
                right,
                left_dequantized,
                right_dequantized,
                left_harmonics,
                right_harmonics,
                extended_data: Vec::new(),
                bits_read: reader.bits_read() - start,
            },
        })
    }
}

fn pointer_bits(envelope_count: usize) -> usize {
    usize::BITS as usize - envelope_count.leading_zeros() as usize
}

/// Parse the body of an AAC `fill_element()`, starting at `count`.
/// Non-SBR extension payloads are consumed and reported as `None`.
pub fn parse_sbr_fill_element(
    reader: &mut BitReader<'_>,
) -> Result<Option<SbrFillPayload>, SbrError> {
    let mut count = reader.read_u8(4)? as usize;
    if count == 15 {
        count += reader.read_u8(8)? as usize;
        count = count.saturating_sub(1);
    }
    let mut bytes = Vec::with_capacity(count);
    for _ in 0..count {
        bytes.push(reader.read_u8(8)?);
    }
    if bytes.is_empty() {
        return Ok(None);
    }
    let mut payload = BitReader::new(&bytes);
    let extension_type = payload.read_u8(4)?;
    if !matches!(extension_type, EXT_SBR_DATA | EXT_SBR_DATA_CRC) {
        return Ok(None);
    }
    let transmitted_crc = if extension_type == EXT_SBR_DATA_CRC {
        Some(payload.read_u16(10)?)
    } else {
        None
    };
    let header_present = payload.read_bool()?;
    let header = header_present
        .then(|| LdSbrHeader::parse(&mut payload))
        .transpose()?;
    let frame_data_bits = payload.remaining_bits();
    let mut frame_data = vec![0u8; frame_data_bits.div_ceil(8)];
    for bit in 0..frame_data_bits {
        if payload.read_bool()? {
            frame_data[bit / 8] |= 1 << (7 - bit % 8);
        }
    }
    Ok(Some(SbrFillPayload {
        extension_type,
        transmitted_crc,
        header_present,
        header,
        frame_data,
        frame_data_bits,
    }))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SbrError {
    Bit(BitError),
    Asc(AscError),
    LdSbr(LdSbrError),
    InvalidGrid,
    UnsupportedFrameLength(usize),
    TruncatedFrameData,
    MissingInitialHeader,
    Usac(UsacSbrError),
}

impl From<BitError> for SbrError {
    fn from(value: BitError) -> Self {
        Self::Bit(value)
    }
}

impl From<AscError> for SbrError {
    fn from(value: AscError) -> Self {
        Self::Asc(value)
    }
}

impl From<LdSbrError> for SbrError {
    fn from(value: LdSbrError) -> Self {
        Self::LdSbr(value)
    }
}

impl fmt::Display for SbrError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bit(error) => error.fmt(formatter),
            Self::Asc(error) => error.fmt(formatter),
            Self::LdSbr(error) => error.fmt(formatter),
            Self::InvalidGrid => write!(formatter, "invalid ordinary SBR frame grid"),
            Self::UnsupportedFrameLength(length) => {
                write!(
                    formatter,
                    "unsupported ordinary SBR core frame length {length}"
                )
            }
            Self::TruncatedFrameData => write!(formatter, "truncated ordinary SBR frame data"),
            Self::MissingInitialHeader => {
                write!(formatter, "ordinary SBR requires an initial header")
            }
            Self::Usac(error) => write!(formatter, "USAC SBR error: {error:?}"),
        }
    }
}

impl std::error::Error for SbrError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitWriter;
    use crate::ld_sbr::SbrHuffmanBook;

    #[test]
    fn converts_and_formats_every_sbr_error_variant() {
        let bit = BitError::UnexpectedEof {
            needed_bits: 1,
            remaining_bits: 0,
        };
        let asc = AscError::InvalidSamplingFrequencyIndex(15);
        let ld = LdSbrError::InvalidFrequencyRange;
        assert_eq!(SbrError::from(bit.clone()), SbrError::Bit(bit));
        assert_eq!(SbrError::from(asc.clone()), SbrError::Asc(asc));
        assert_eq!(SbrError::from(ld.clone()), SbrError::LdSbr(ld));
        let errors = [
            SbrError::Bit(BitError::UnexpectedEof {
                needed_bits: 1,
                remaining_bits: 0,
            }),
            SbrError::Asc(AscError::InvalidSamplingFrequencyIndex(15)),
            SbrError::LdSbr(LdSbrError::InvalidFrequencyRange),
            SbrError::InvalidGrid,
            SbrError::UnsupportedFrameLength(1),
            SbrError::TruncatedFrameData,
            SbrError::MissingInitialHeader,
            SbrError::Usac(UsacSbrError::ReservedPvcMode),
        ];
        for error in errors {
            assert!(!error.to_string().is_empty());
        }
        assert_eq!(ordinary_time_slots(960).unwrap(), 15);
        assert_eq!(ordinary_time_slots(1024).unwrap(), 16);
        assert_eq!(
            ordinary_time_slots(512),
            Err(SbrError::UnsupportedFrameLength(512))
        );
    }

    #[test]
    fn usac_header_parser_covers_default_explicit_and_truncated_forms() {
        let mut defaults = BitWriter::new();
        defaults.write(5, 4);
        defaults.write(8, 4);
        defaults.write_bool(false);
        defaults.write_bool(false);
        let bytes = defaults.finish();
        let header = parse_usac_sbr_header(&mut BitReader::new(&bytes), true, 3).unwrap();
        assert_eq!(header.start_frequency, 5);
        assert_eq!(header.stop_frequency, 8);
        assert_eq!(header.frequency_scale, Some(2));
        assert_eq!(header.alter_scale, Some(true));
        assert_eq!(header.noise_bands, Some(2));
        assert_eq!(header.limiter_bands, Some(2));
        assert_eq!(header.limiter_gains, Some(2));
        assert_eq!(header.interpol_frequency, Some(true));
        assert_eq!(header.smoothing_mode, Some(true));

        let mut explicit = BitWriter::new();
        explicit.write(6, 4);
        explicit.write(7, 4);
        explicit.write_bool(true);
        explicit.write_bool(true);
        explicit.write(1, 2);
        explicit.write_bool(false);
        explicit.write(3, 2);
        explicit.write(0, 2);
        explicit.write(1, 2);
        explicit.write_bool(false);
        explicit.write_bool(false);
        let bits = explicit.bits_written();
        let bytes = explicit.finish();
        let mut reader = BitReader::with_bit_len(&bytes, bits).unwrap();
        let header = parse_usac_sbr_header(&mut reader, false, 4).unwrap();
        assert_eq!(reader.bits_read(), bits);
        assert_eq!(header.frequency_scale, Some(1));
        assert_eq!(header.alter_scale, Some(false));
        assert_eq!(header.noise_bands, Some(3));
        assert_eq!(header.limiter_bands, Some(0));
        assert_eq!(header.limiter_gains, Some(1));
        assert_eq!(header.interpol_frequency, Some(false));
        assert_eq!(header.smoothing_mode, Some(false));
        let mut stereo = SbrStereoFrameParser::new_usac(header.clone(), 44_100).unwrap();
        stereo.set_usac_header(header).unwrap();
        assert!(matches!(
            parse_usac_sbr_header(&mut BitReader::new(&[]), false, 0),
            Err(SbrError::Bit(BitError::UnexpectedEof { .. }))
        ));

        let mut empty_fill = BitWriter::new();
        empty_fill.write(0, 4);
        assert_eq!(
            parse_sbr_fill_element(&mut BitReader::new(&empty_fill.finish())).unwrap(),
            None
        );
    }

    fn huffman_code(book: SbrHuffmanBook, symbol: i8) -> Vec<bool> {
        crate::ld_sbr::encode_sbr_huffman(book, symbol)
            .expect("requested symbol exists in the SBR Huffman book")
    }

    fn write_code(writer: &mut BitWriter, code: &[bool]) {
        for &bit in code {
            writer.write_bool(bit);
        }
    }

    #[test]
    fn parses_crc_header_and_preserves_ordinary_sbr_frame_bits() {
        let header = LdSbrHeader {
            amp_resolution: true,
            start_frequency: 5,
            stop_frequency: 8,
            crossover_band: 2,
            frequency_scale: Some(1),
            alter_scale: Some(false),
            noise_bands: Some(2),
            limiter_bands: Some(2),
            limiter_gains: Some(1),
            interpol_frequency: Some(true),
            smoothing_mode: Some(false),
            ..LdSbrHeader::default()
        };
        let mut body = BitWriter::new();
        body.write(EXT_SBR_DATA_CRC as u32, 4);
        body.write(0x155, 10);
        body.write_bool(true);
        header.write(&mut body).unwrap();
        body.write(0b10101, 5);
        let body = body.finish();
        let mut fill = BitWriter::new();
        fill.write(body.len() as u32, 4);
        for byte in body {
            fill.write(byte as u32, 8);
        }

        let parsed = parse_sbr_fill_element(&mut BitReader::new(&fill.finish()))
            .unwrap()
            .unwrap();
        assert_eq!(parsed.extension_type, EXT_SBR_DATA_CRC);
        assert_eq!(parsed.transmitted_crc, Some(0x155));
        assert_eq!(parsed.header, Some(header));
        assert!(parsed.frame_data_bits >= 5);
        assert_eq!(parsed.frame_data[0] >> 3, 0b10101);
    }

    #[test]
    fn consumes_non_sbr_fill_extension() {
        let mut fill = BitWriter::new();
        fill.write(1, 4);
        fill.write(0x00, 8);
        assert_eq!(
            parse_sbr_fill_element(&mut BitReader::new(&fill.finish())).unwrap(),
            None
        );
    }

    #[test]
    fn parses_all_four_ordinary_sbr_frame_classes() {
        let mut fixfix = BitWriter::new();
        fixfix.write(0, 2);
        fixfix.write(2, 2); // four envelopes
        fixfix.write_bool(false);
        let grid = SbrGrid::parse(&mut BitReader::new(&fixfix.finish()), 16).unwrap();
        assert_eq!(grid.frame_class, SbrFrameClass::FixFix);
        assert_eq!(grid.borders, vec![0, 4, 8, 12, 16]);
        assert_eq!(grid.noise_borders, vec![0, 8, 16]);

        let mut fixvar = BitWriter::new();
        fixvar.write(1, 2);
        fixvar.write(1, 2); // right border 17
        fixvar.write(2, 2); // three envelopes
        fixvar.write(0, 2);
        fixvar.write(0, 2);
        fixvar.write(2, 2); // transient envelope 2
        fixvar.write_bool(true); // freqRes[2]
        fixvar.write_bool(false); // freqRes[1]
        fixvar.write_bool(true); // freqRes[0]
        let grid = SbrGrid::parse(&mut BitReader::new(&fixvar.finish()), 16).unwrap();
        assert_eq!(grid.frame_class, SbrFrameClass::FixVar);
        assert_eq!(grid.borders, vec![0, 13, 15, 17]);
        assert_eq!(grid.transient_envelope, Some(2));
        assert_eq!(grid.frequency_resolution, vec![true, false, true]);
        assert_eq!(grid.noise_borders, vec![0, 15, 17]);

        let mut varfix = BitWriter::new();
        varfix.write(2, 2);
        varfix.write(1, 2); // left border 1
        varfix.write(1, 2); // two envelopes
        varfix.write(0, 2); // relative border 3
        varfix.write(2, 2); // transient envelope 1
        varfix.write_bool(false);
        varfix.write_bool(true);
        let grid = SbrGrid::parse(&mut BitReader::new(&varfix.finish()), 16).unwrap();
        assert_eq!(grid.frame_class, SbrFrameClass::VarFix);
        assert_eq!(grid.borders, vec![1, 3, 16]);
        assert_eq!(grid.transient_envelope, Some(1));
        assert_eq!(grid.noise_borders, vec![1, 3, 16]);

        let mut varvar = BitWriter::new();
        varvar.write(3, 2);
        varvar.write(1, 2); // left border 1
        varvar.write(1, 2); // right border 17
        varvar.write(1, 2); // one left relative border
        varvar.write(1, 2); // one right relative border
        varvar.write(0, 2); // left border 3
        varvar.write(0, 2); // right border 15
        varvar.write(2, 2); // transient envelope 2
        varvar.write_bool(true);
        varvar.write_bool(false);
        varvar.write_bool(true);
        let grid = SbrGrid::parse(&mut BitReader::new(&varvar.finish()), 16).unwrap();
        assert_eq!(grid.frame_class, SbrFrameClass::VarVar);
        assert_eq!(grid.borders, vec![1, 3, 15, 17]);
        assert_eq!(grid.transient_envelope, Some(2));
        assert_eq!(grid.noise_borders, vec![1, 15, 17]);
    }

    #[test]
    fn grid_pointer_and_invalid_layout_branches_are_total() {
        for pointer in [0, 1] {
            let mut writer = BitWriter::new();
            writer.write(1, 2); // FIXVAR
            writer.write(0, 2); // right border 16
            writer.write(1, 2); // two envelopes
            writer.write(0, 2); // relative border 14
            writer.write(pointer, 2);
            writer.write_bool(false);
            writer.write_bool(false);
            let grid = SbrGrid::parse(&mut BitReader::new(&writer.finish()), 16).unwrap();
            assert_eq!(grid.noise_borders, vec![0, 14, 16]);
        }

        for (pointer, expected_middle) in [(0, 2), (1, 4)] {
            let mut writer = BitWriter::new();
            writer.write(2, 2); // VARFIX
            writer.write(0, 2); // left border 0
            writer.write(2, 2); // three envelopes
            writer.write(0, 2); // relative borders 2 and 4
            writer.write(0, 2);
            writer.write(pointer, 2);
            for _ in 0..3 {
                writer.write_bool(false);
            }
            let grid = SbrGrid::parse(&mut BitReader::new(&writer.finish()), 16).unwrap();
            assert_eq!(grid.noise_borders, vec![0, expected_middle, 16]);
        }

        let mut writer = BitWriter::new();
        writer.write(3, 2); // VARVAR
        writer.write(1, 2);
        writer.write(1, 2);
        writer.write(1, 2);
        writer.write(1, 2);
        writer.write(0, 2);
        writer.write(0, 2);
        writer.write(0, 2); // pointer <= 1
        for _ in 0..3 {
            writer.write_bool(false);
        }
        let grid = SbrGrid::parse(&mut BitReader::new(&writer.finish()), 16).unwrap();
        assert_eq!(grid.noise_borders, vec![1, 15, 17]);

        let mut fixvar_negative = BitWriter::new();
        fixvar_negative.write(1, 2);
        fixvar_negative.write(0, 2);
        fixvar_negative.write(3, 2);
        for _ in 0..3 {
            fixvar_negative.write(3, 2);
        }
        assert_eq!(
            SbrGrid::parse(&mut BitReader::new(&fixvar_negative.finish()), 16),
            Err(SbrError::InvalidGrid)
        );

        let mut fixvar_pointer = BitWriter::new();
        fixvar_pointer.write(1, 2);
        fixvar_pointer.write(0, 2);
        fixvar_pointer.write(1, 2);
        fixvar_pointer.write(0, 2);
        fixvar_pointer.write(3, 2);
        assert_eq!(
            SbrGrid::parse(&mut BitReader::new(&fixvar_pointer.finish()), 16),
            Err(SbrError::InvalidGrid)
        );

        let mut varfix_pointer = BitWriter::new();
        varfix_pointer.write(2, 2);
        varfix_pointer.write(0, 2);
        varfix_pointer.write(1, 2);
        varfix_pointer.write(0, 2);
        varfix_pointer.write(3, 2);
        assert_eq!(
            SbrGrid::parse(&mut BitReader::new(&varfix_pointer.finish()), 16),
            Err(SbrError::InvalidGrid)
        );

        let mut varvar_negative = BitWriter::new();
        varvar_negative.write(3, 2);
        varvar_negative.write(0, 2);
        varvar_negative.write(0, 2);
        varvar_negative.write(0, 2);
        varvar_negative.write(3, 2);
        for _ in 0..3 {
            varvar_negative.write(3, 2);
        }
        assert_eq!(
            SbrGrid::parse(&mut BitReader::new(&varvar_negative.finish()), 16),
            Err(SbrError::InvalidGrid)
        );

        let mut varvar_pointer = BitWriter::new();
        varvar_pointer.write(3, 2);
        varvar_pointer.write(0, 2);
        varvar_pointer.write(0, 2);
        varvar_pointer.write(1, 2);
        varvar_pointer.write(0, 2);
        varvar_pointer.write(0, 2);
        varvar_pointer.write(3, 2);
        assert_eq!(
            SbrGrid::parse(&mut BitReader::new(&varvar_pointer.finish()), 16),
            Err(SbrError::InvalidGrid)
        );

        let mut non_monotonic = BitWriter::new();
        non_monotonic.write(2, 2);
        non_monotonic.write(3, 2);
        non_monotonic.write(3, 2);
        for _ in 0..3 {
            non_monotonic.write(3, 2);
        }
        non_monotonic.write(0, 3);
        for _ in 0..4 {
            non_monotonic.write_bool(false);
        }
        assert_eq!(
            SbrGrid::parse(&mut BitReader::new(&non_monotonic.finish()), 16),
            Err(SbrError::InvalidGrid)
        );
    }

    #[test]
    fn usac_control_reads_dependent_flags_and_implies_independent_first_flags() {
        let grid = LdSbrGrid {
            transient: false,
            amp_resolution: Some(true),
            borders: vec![0, 8, 16],
            frequency_resolution: vec![true, false],
            transient_envelope: None,
            noise_borders: vec![0, 8, 16],
        };
        let mut writer = BitWriter::new();
        for value in [true, false, false, true] {
            writer.write_bool(value);
        }
        let dependent =
            read_usac_control(&mut BitReader::new(&writer.finish()), grid.clone(), false).unwrap();
        assert_eq!(dependent.envelope_time_domain, [true, false]);
        assert_eq!(dependent.noise_time_domain, [false, true]);

        let mut writer = BitWriter::new();
        writer.write_bool(true);
        writer.write_bool(true);
        let independent =
            read_usac_control(&mut BitReader::new(&writer.finish()), grid, true).unwrap();
        assert_eq!(independent.envelope_time_domain, [false, true]);
        assert_eq!(independent.noise_time_domain, [false, true]);
    }

    #[test]
    fn parses_mono_envelope_noise_and_dequantizes_values() {
        let header = LdSbrHeader {
            amp_resolution: true,
            start_frequency: 5,
            stop_frequency: 8,
            crossover_band: 2,
            frequency_scale: Some(1),
            alter_scale: Some(false),
            noise_bands: Some(2),
            ..LdSbrHeader::default()
        };
        let tables = LdSbrFrequencyTables::from_header(&header, 44_100).unwrap();
        let zero = huffman_code(SbrHuffmanBook::EnvelopeLevel30Frequency, 0);
        let mut writer = BitWriter::new();
        writer.write_bool(false); // bs_data_extra
        writer.write(0, 2); // FIXFIX
        writer.write(0, 2); // one envelope
        writer.write_bool(true); // high frequency resolution
        writer.write_bool(false); // envelope frequency direction
        writer.write_bool(false); // noise frequency direction
        for _ in 0..tables.noise_band_count() {
            writer.write(2, 2); // inverse filtering mode
        }
        writer.write(10, 6); // absolute envelope value
        for _ in 1..tables.high_band_count() {
            write_code(&mut writer, &zero);
        }
        writer.write(7, 5); // absolute noise value
        for _ in 1..tables.noise_band_count() {
            write_code(&mut writer, &zero);
        }
        writer.write_bool(false); // no harmonics
        writer.write_bool(false); // no extended data
        let frame_data_bits = writer.bits_written();
        let payload = SbrFillPayload {
            extension_type: EXT_SBR_DATA,
            transmitted_crc: None,
            header_present: false,
            header: None,
            frame_data: writer.finish(),
            frame_data_bits,
        };
        let mut parser = SbrMonoFrameParser::new(header.clone(), 44_100, 1024).unwrap();
        let frame = parser.parse(&payload).unwrap();
        assert_eq!(frame.bits_read, frame_data_bits);
        assert_eq!(
            frame.values.inverse_filtering_modes,
            vec![2; tables.noise_band_count()]
        );
        assert_eq!(
            frame.values.envelopes[0],
            vec![10; tables.high_band_count()]
        );
        assert_eq!(frame.values.noise[0], vec![7; tables.noise_band_count()]);
        assert!(frame.harmonics.iter().all(|&enabled| !enabled));
        assert!(frame.extended_data.is_empty());
        assert!(frame.dequantized.envelope_energy[0]
            .iter()
            .all(|&energy| energy > 0.0));

        let mut truncated = payload.clone();
        truncated.frame_data_bits -= 1;
        let mut parser =
            SbrMonoFrameParser::new(frame.active_header.clone(), 44_100, 1024).unwrap();
        assert_eq!(
            parser.parse(&truncated).unwrap_err(),
            SbrError::TruncatedFrameData
        );
        for byte_len in 0..payload.frame_data.len() {
            let mut truncated = payload.clone();
            truncated.frame_data.truncate(byte_len);
            truncated.frame_data_bits = byte_len * 8;
            assert!(SbrMonoFrameParser::new(header.clone(), 44_100, 1024)
                .unwrap()
                .parse(&truncated)
                .is_err());
        }
    }

    #[test]
    fn parses_independent_usac_mono_without_legacy_framing_bits() {
        let header = LdSbrHeader {
            amp_resolution: true,
            start_frequency: 5,
            stop_frequency: 8,
            crossover_band: 2,
            frequency_scale: Some(1),
            alter_scale: Some(false),
            noise_bands: Some(2),
            ..LdSbrHeader::default()
        };
        let tables = LdSbrFrequencyTables::from_header(&header, 44_100).unwrap();
        let zero = huffman_code(SbrHuffmanBook::EnvelopeLevel30Frequency, 0);
        let mut writer = BitWriter::new();
        writer.write(0, 2); // FIXFIX; no bs_data_extra in USAC
        writer.write(0, 2); // one envelope
        writer.write_bool(true);
        // independent USAC implies frequency direction for first envelope/noise floor.
        for _ in 0..tables.noise_band_count() {
            writer.write(1, 2);
        }
        writer.write(9, 6);
        for _ in 1..tables.high_band_count() {
            write_code(&mut writer, &zero);
        }
        writer.write_bool(true); // inter-TES active
        writer.write(2, 2); // inter-TES mode
        writer.write(6, 5);
        for _ in 1..tables.noise_band_count() {
            write_code(&mut writer, &zero);
        }
        writer.write_bool(false); // no add harmonics
        let bits = writer.bits_written();
        let bytes = writer.finish();
        let mut reader = BitReader::with_bit_len(&bytes, bits).unwrap();
        let mut parser = SbrMonoFrameParser::new_usac(header.clone(), 44_100).unwrap();
        let frame = parser.parse_usac(&mut reader, true, false, true).unwrap();
        assert_eq!(reader.bits_read(), bits);
        assert_eq!(
            frame.frame.values.envelopes[0],
            vec![9; tables.high_band_count()]
        );
        assert_eq!(
            frame.frame.values.noise[0],
            vec![6; tables.noise_band_count()]
        );
        assert_eq!(
            frame.inter_tes,
            [InterTesEnvelope {
                active: true,
                mode: 2
            }]
        );
        assert!(frame.frame.extended_data.is_empty());
        for bit_len in 0..bits {
            let mut reader = BitReader::with_bit_len(&bytes, bit_len).unwrap();
            assert!(SbrMonoFrameParser::new_usac(header.clone(), 44_100)
                .unwrap()
                .parse_usac(&mut reader, true, false, true)
                .is_err());
        }
    }

    #[test]
    fn optional_mono_prefixes_propagate_truncated_payloads() {
        let header = LdSbrHeader {
            amp_resolution: true,
            start_frequency: 5,
            stop_frequency: 8,
            ..LdSbrHeader::default()
        };

        let payload = SbrFillPayload {
            extension_type: EXT_SBR_DATA,
            transmitted_crc: None,
            header_present: false,
            header: Some(header.clone()),
            frame_data: vec![0x80],
            frame_data_bits: 1,
        };
        let mut parser = SbrMonoFrameParser::new(header.clone(), 44_100, 1024).unwrap();
        assert!(parser.parse(&payload).is_err());

        let bytes = [0x80];
        let mut parser = SbrMonoFrameParser::new_usac(header.clone(), 44_100).unwrap();
        assert!(matches!(
            parser.parse_usac(
                &mut BitReader::with_bit_len(&bytes, 1).unwrap(),
                true,
                true,
                false,
            ),
            Err(SbrError::Bit(BitError::UnexpectedEof { .. }))
        ));

        let mut parser = SbrMonoFrameParser::new_usac(header, 44_100).unwrap();
        assert!(parser
            .parse_usac_pvc(
                &mut BitReader::with_bit_len(&bytes, 1).unwrap(),
                true,
                1,
                true,
            )
            .is_err());
    }

    #[test]
    fn usac_sbr_orchestrators_propagate_header_table_and_payload_errors() {
        let config = UsacSbrConfig {
            harmonic_sbr: false,
            inter_tes: false,
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

        let mut truncated_header = BitWriter::new();
        truncated_header.write_bool(true); // amplitude resolution
        truncated_header.write(2, 4); // crossover
        truncated_header.write_bool(false); // preprocessing
        truncated_header.write_bool(false); // explicit header
        truncated_header.write(5, 4); // start frequency, no stop frequency
        let bits = truncated_header.bits_written();
        let bytes = truncated_header.finish();
        let mut reader = BitReader::with_bit_len(&bytes, bits).unwrap();
        assert!(UsacSbrMonoParser::new(config.clone(), 44_100)
            .unwrap()
            .parse(&mut reader, true)
            .is_err());

        let mut invalid_header = BitWriter::new();
        invalid_header.write_bool(true); // amplitude resolution
        invalid_header.write(15, 4); // crossover beyond the generated table
        invalid_header.write_bool(false); // preprocessing
        invalid_header.write_bool(false); // explicit header
        invalid_header.write(15, 4); // start frequency
        invalid_header.write(0, 4); // stop frequency
        invalid_header.write_bool(false); // default frequency settings
        invalid_header.write_bool(false); // default limiter settings
        let bits = invalid_header.bits_written();
        let bytes = invalid_header.finish();
        let mut reader = BitReader::with_bit_len(&bytes, bits).unwrap();
        assert!(UsacSbrMonoParser::new(config.clone(), 44_100)
            .unwrap()
            .parse(&mut reader, true)
            .is_err());

        let mut truncated_payload = BitWriter::new();
        truncated_payload.write_bool(true); // amplitude resolution
        truncated_payload.write(2, 4); // crossover
        truncated_payload.write_bool(false); // preprocessing
        truncated_payload.write_bool(true); // use default header, no payload follows
        let bits = truncated_payload.bits_written();
        let bytes = truncated_payload.finish();
        let mut reader = BitReader::with_bit_len(&bytes, bits).unwrap();
        assert!(UsacSbrMonoParser::new(config.clone(), 44_100)
            .unwrap()
            .parse(&mut reader, true)
            .is_err());

        let mut pvc_config = config.clone();
        pvc_config.pvc = true;
        let mut truncated_pvc = BitWriter::new();
        truncated_pvc.write_bool(true); // amplitude resolution
        truncated_pvc.write(2, 4); // crossover
        truncated_pvc.write_bool(false); // preprocessing
        truncated_pvc.write(1, 2); // PVC mode
        truncated_pvc.write_bool(true); // use default header, no payload follows
        let bits = truncated_pvc.bits_written();
        let bytes = truncated_pvc.finish();
        let mut reader = BitReader::with_bit_len(&bytes, bits).unwrap();
        assert!(UsacSbrMonoParser::new(pvc_config, 44_100)
            .unwrap()
            .parse(&mut reader, true)
            .is_err());

        let mut reader = BitReader::with_bit_len(&[0b1001_0000], 7).unwrap();
        assert!(UsacSbrStereoParser::new(config.clone(), 44_100)
            .unwrap()
            .parse(&mut reader, true)
            .is_err());

        let mut invalid_stereo_header = BitWriter::new();
        invalid_stereo_header.write_bool(true); // amplitude resolution
        invalid_stereo_header.write(15, 4); // crossover beyond the generated table
        invalid_stereo_header.write_bool(false); // preprocessing
        invalid_stereo_header.write_bool(false); // explicit header
        invalid_stereo_header.write(15, 4); // start frequency
        invalid_stereo_header.write(0, 4); // stop frequency
        invalid_stereo_header.write_bool(false); // default frequency settings
        invalid_stereo_header.write_bool(false); // default limiter settings
        let bits = invalid_stereo_header.bits_written();
        let bytes = invalid_stereo_header.finish();
        let mut reader = BitReader::with_bit_len(&bytes, bits).unwrap();
        assert!(UsacSbrStereoParser::new(config.clone(), 44_100)
            .unwrap()
            .parse(&mut reader, true)
            .is_err());

        let mut reader = BitReader::with_bit_len(&[0b1001_0010], 7).unwrap();
        assert!(UsacSbrStereoParser::new(config, 44_100)
            .unwrap()
            .parse(&mut reader, true)
            .is_err());
    }

    #[test]
    fn parses_independent_usac_stereo_through_direct_and_orchestrated_paths() {
        let config = UsacSbrConfig {
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
        let header = header_from_usac_config(&config, true, 2);
        let tables = LdSbrFrequencyTables::from_header(&header, 44_100).unwrap();
        let zero = huffman_code(SbrHuffmanBook::EnvelopeLevel30Frequency, 0);
        let write_payload = |writer: &mut BitWriter, harmonic_controls: bool, independent: bool| {
            writer.write_bool(false); // uncoupled
            if harmonic_controls {
                writer.write_bool(true); // left: default patching
                writer.write_bool(true); // right: default patching
            }
            for _ in 0..2 {
                writer.write(0, 2); // FIXFIX
                writer.write(0, 2); // one envelope
                writer.write_bool(true); // high frequency resolution
            }
            if !independent {
                for _ in 0..2 {
                    writer.write_bool(false); // envelope frequency direction
                    writer.write_bool(false); // noise frequency direction
                }
            }
            // Independent USAC implies frequency direction for the first
            // envelope and noise floor, so no direction bits are present.
            for channel in 0..2 {
                for _ in 0..tables.noise_band_count() {
                    writer.write(channel + 1, 2); // independent invf values
                }
            }
            for (absolute, mode) in [(9, 1), (11, 3)] {
                writer.write(absolute, 6);
                for _ in 1..tables.high_band_count() {
                    write_code(writer, &zero);
                }
                writer.write_bool(true); // inter-TES active
                writer.write(mode, 2);
            }
            for absolute in [5, 7] {
                writer.write(absolute, 5);
                for _ in 1..tables.noise_band_count() {
                    write_code(writer, &zero);
                }
            }
            writer.write_bool(false); // no left harmonics
            writer.write_bool(false); // no right harmonics
        };

        let mut direct_bits = BitWriter::new();
        write_payload(&mut direct_bits, true, true);
        let direct_len = direct_bits.bits_written();
        let direct_bytes = direct_bits.finish();
        let mut direct_reader = BitReader::with_bit_len(&direct_bytes, direct_len).unwrap();
        let mut direct_parser = SbrStereoFrameParser::new_usac(header.clone(), 44_100).unwrap();
        let direct = direct_parser
            .parse_usac(&mut direct_reader, true, true, true)
            .unwrap();
        assert_eq!(direct_reader.bits_read(), direct_len);
        assert!(!direct.frame.coupling);
        assert_eq!(direct.frame.left.envelopes[0][0], 9);
        assert_eq!(direct.frame.right.envelopes[0][0], 11);
        assert_eq!(direct.frame.left.noise[0][0], 5);
        assert_eq!(direct.frame.right.noise[0][0], 7);
        assert_eq!(direct.inter_tes[0][0].mode, 1);
        assert_eq!(direct.inter_tes[1][0].mode, 3);
        assert!(direct.harmonic_controls[0].unwrap().patching_mode);
        assert!(direct.harmonic_controls[1].unwrap().patching_mode);

        let mut orchestrated_bits = BitWriter::new();
        orchestrated_bits.write_bool(true); // amp resolution in independent sbr_info
        orchestrated_bits.write(2, 4); // crossover
        orchestrated_bits.write_bool(false); // preprocessing
        orchestrated_bits.write_bool(true); // use default header
        write_payload(&mut orchestrated_bits, false, true);
        let orchestrated_len = orchestrated_bits.bits_written();
        let orchestrated_bytes = orchestrated_bits.finish();
        let mut reader = BitReader::with_bit_len(&orchestrated_bytes, orchestrated_len).unwrap();
        let mut parser = UsacSbrStereoParser::new(config.clone(), 44_100).unwrap();
        let parsed = parser.parse(&mut reader, true).unwrap();
        assert_eq!(reader.bits_read(), orchestrated_len);
        assert_eq!(parsed.bits_read, orchestrated_len);
        assert_eq!(parsed.active_header, header);
        assert_eq!(parsed.payload.frame.left.envelopes[0][0], 9);
        assert_eq!(parsed.payload.frame.right.envelopes[0][0], 11);

        let mut dependent_bits = BitWriter::new();
        dependent_bits.write_bool(false); // no dependent sbr_info or header
        write_payload(&mut dependent_bits, false, false);
        let dependent_len = dependent_bits.bits_written();
        let dependent_bytes = dependent_bits.finish();
        let mut reader = BitReader::with_bit_len(&dependent_bytes, dependent_len).unwrap();
        let dependent = parser.parse(&mut reader, false).unwrap();
        assert_eq!(dependent.bits_read, dependent_len);
        assert_eq!(dependent.active_header, header);

        let mut explicit_bits = BitWriter::new();
        explicit_bits.write_bool(true); // amp resolution
        explicit_bits.write(2, 4); // crossover
        explicit_bits.write_bool(false); // preprocessing
        explicit_bits.write_bool(false); // explicit header follows
        explicit_bits.write(5, 4);
        explicit_bits.write(8, 4);
        explicit_bits.write_bool(true);
        explicit_bits.write_bool(true);
        explicit_bits.write(1, 2);
        explicit_bits.write_bool(false);
        explicit_bits.write(2, 2);
        explicit_bits.write(2, 2);
        explicit_bits.write(2, 2);
        explicit_bits.write_bool(true);
        explicit_bits.write_bool(true);
        write_payload(&mut explicit_bits, false, true);
        let explicit_len = explicit_bits.bits_written();
        let explicit_bytes = explicit_bits.finish();
        let mut reader = BitReader::with_bit_len(&explicit_bytes, explicit_len).unwrap();
        let mut parser = UsacSbrStereoParser::new(config, 44_100).unwrap();
        let parsed = parser.parse(&mut reader, true).unwrap();
        assert_eq!(reader.bits_read(), explicit_len);
        assert_eq!(parsed.active_header, header);
    }

    #[test]
    fn parses_independent_usac_pvc_frame() {
        let header = LdSbrHeader {
            amp_resolution: true,
            start_frequency: 5,
            stop_frequency: 8,
            crossover_band: 2,
            frequency_scale: Some(1),
            alter_scale: Some(false),
            noise_bands: Some(2),
            ..LdSbrHeader::default()
        };
        let tables = LdSbrFrequencyTables::from_header(&header, 44_100).unwrap();
        let zero = huffman_code(SbrHuffmanBook::EnvelopeLevel30Frequency, 0);
        let mut writer = BitWriter::new();
        writer.write(0, 4); // noise position: one envelope
        writer.write_bool(false); // fixed HF end
        for _ in 0..tables.noise_band_count() {
            writer.write(2, 2); // invf
        }
        writer.write(0, 3); // PVC division mode
        writer.write_bool(false); // noise shaping mode
        writer.write(37, 7); // first PVC ID; no reuse flag when independent
        writer.write(5, 5); // noise absolute
        for _ in 1..tables.noise_band_count() {
            write_code(&mut writer, &zero);
        }
        writer.write_bool(false); // no harmonics
        let bits = writer.bits_written();
        let bytes = writer.finish();
        let mut reader = BitReader::with_bit_len(&bytes, bits).unwrap();
        let mut parser = SbrMonoFrameParser::new_usac(header, 44_100).unwrap();
        let frame = parser.parse_usac_pvc(&mut reader, true, 1, false).unwrap();
        assert_eq!(reader.bits_read(), bits);
        assert_eq!(frame.envelope.ids, [37; 16]);
        assert_eq!(frame.noise[0], vec![5; tables.noise_band_count()]);
        assert_eq!(
            frame.inverse_filtering_modes,
            vec![2; tables.noise_band_count()]
        );

        let noise_zero = huffman_code(SbrHuffmanBook::NoiseLevelTime, 0);
        let mut writer = BitWriter::new();
        writer.write(8, 4); // two noise envelopes
        writer.write_bool(false); // fixed right border
        writer.write_bool(true); // first noise envelope uses previous frame
        writer.write_bool(true); // second noise envelope uses first envelope
        for _ in 0..tables.noise_band_count() {
            writer.write(2, 2);
        }
        writer.write(0, 3); // PVC division mode
        writer.write_bool(false); // noise shaping mode
        writer.write_bool(true); // reuse previous PVC id
        for _ in 0..2 {
            for _ in 0..tables.noise_band_count() {
                write_code(&mut writer, &noise_zero);
            }
        }
        writer.write_bool(false); // no harmonics
        let bits = writer.bits_written();
        let bytes = writer.finish();
        let mut reader = BitReader::with_bit_len(&bytes, bits).unwrap();
        let dependent = parser.parse_usac_pvc(&mut reader, false, 1, false).unwrap();
        assert_eq!(reader.bits_read(), bits);
        assert_eq!(dependent.noise.len(), 2);
        assert_eq!(dependent.noise[0], frame.noise[0]);
        assert_eq!(dependent.noise[1], frame.noise[0]);
    }

    #[test]
    fn orchestrates_independent_usac_info_default_header_and_frame() {
        let config = UsacSbrConfig {
            harmonic_sbr: false,
            inter_tes: false,
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
        let header = header_from_usac_config(&config, true, 2);
        let tables = LdSbrFrequencyTables::from_header(&header, 44_100).unwrap();
        let zero = huffman_code(SbrHuffmanBook::EnvelopeLevel30Frequency, 0);
        let write_payload = |writer: &mut BitWriter, independent: bool| {
            writer.write(0, 2); // FIXFIX
            writer.write(0, 2); // one envelope
            writer.write_bool(true);
            if !independent {
                writer.write_bool(false); // envelope frequency direction
                writer.write_bool(false); // noise frequency direction
            }
            for _ in 0..tables.noise_band_count() {
                writer.write(0, 2);
            }
            writer.write(8, 6);
            for _ in 1..tables.high_band_count() {
                write_code(writer, &zero);
            }
            writer.write(4, 5);
            for _ in 1..tables.noise_band_count() {
                write_code(writer, &zero);
            }
            writer.write_bool(false);
        };
        let mut writer = BitWriter::new();
        writer.write_bool(true); // amp resolution in mandatory independent sbr_info
        writer.write(2, 4); // crossover
        writer.write_bool(false); // preprocessing
        writer.write_bool(true); // use default header
        write_payload(&mut writer, true);
        let bits = writer.bits_written();
        let bytes = writer.finish();
        let mut reader = BitReader::with_bit_len(&bytes, bits).unwrap();
        let mut parser = UsacSbrMonoParser::new(config.clone(), 44_100).unwrap();
        let parsed = parser.parse(&mut reader, true).unwrap();
        assert_eq!(parsed.bits_read, bits);
        assert_eq!(parsed.active_header.crossover_band, 2);
        assert!(matches!(parsed.payload, UsacSbrPayloadFrame::Ordinary(_)));

        let mut writer = BitWriter::new();
        writer.write_bool(false); // no dependent sbr_info or header
        write_payload(&mut writer, false);
        let bits = writer.bits_written();
        let bytes = writer.finish();
        let mut reader = BitReader::with_bit_len(&bytes, bits).unwrap();
        let parsed = parser.parse(&mut reader, false).unwrap();
        assert_eq!(parsed.bits_read, bits);
        assert_eq!(parsed.active_header.crossover_band, 2);

        let mut writer = BitWriter::new();
        writer.write_bool(true); // amp resolution
        writer.write(2, 4); // crossover
        writer.write_bool(false); // preprocessing
        writer.write_bool(false); // explicit header follows
        writer.write(5, 4);
        writer.write(8, 4);
        writer.write_bool(true);
        writer.write_bool(true);
        writer.write(1, 2);
        writer.write_bool(false);
        writer.write(2, 2);
        writer.write(2, 2);
        writer.write(2, 2);
        writer.write_bool(true);
        writer.write_bool(true);
        write_payload(&mut writer, true);
        let bits = writer.bits_written();
        let bytes = writer.finish();
        let mut reader = BitReader::with_bit_len(&bytes, bits).unwrap();
        let mut parser = UsacSbrMonoParser::new(config, 44_100).unwrap();
        let parsed = parser.parse(&mut reader, true).unwrap();
        assert_eq!(reader.bits_read(), bits);
        assert_eq!(parsed.active_header, header);
    }

    #[test]
    fn parses_coupled_stereo_and_splits_level_balance_energy() {
        let header = LdSbrHeader {
            amp_resolution: true,
            start_frequency: 5,
            stop_frequency: 8,
            crossover_band: 2,
            frequency_scale: Some(1),
            alter_scale: Some(false),
            noise_bands: Some(2),
            ..LdSbrHeader::default()
        };
        let tables = LdSbrFrequencyTables::from_header(&header, 44_100).unwrap();
        let level_zero = huffman_code(SbrHuffmanBook::EnvelopeLevel30Frequency, 0);
        let balance_zero = huffman_code(SbrHuffmanBook::EnvelopeBalance30Frequency, 0);
        let mut writer = BitWriter::new();
        writer.write_bool(true); // bs_data_extra
        writer.write(3, 4);
        writer.write(9, 4);
        writer.write_bool(true); // coupling
        writer.write(0, 2); // FIXFIX
        writer.write(0, 2); // one envelope
        writer.write_bool(true); // high frequency resolution
        writer.write_bool(false); // left envelope frequency direction
        writer.write_bool(false); // left noise frequency direction
        writer.write_bool(false); // right envelope frequency direction
        writer.write_bool(false); // right noise frequency direction
        for _ in 0..tables.noise_band_count() {
            writer.write(1, 2); // shared inverse filtering mode
        }
        writer.write(9, 6); // coupled level envelope
        for _ in 1..tables.high_band_count() {
            write_code(&mut writer, &level_zero);
        }
        writer.write(6, 5); // coupled level noise
        for _ in 1..tables.noise_band_count() {
            write_code(&mut writer, &level_zero);
        }
        writer.write(6, 5); // centered envelope balance, scaled to 12
        for _ in 1..tables.high_band_count() {
            write_code(&mut writer, &balance_zero);
        }
        writer.write(6, 5); // centered noise balance, scaled to 12
        for _ in 1..tables.noise_band_count() {
            write_code(&mut writer, &balance_zero);
        }
        writer.write_bool(true); // left harmonics
        for _ in 0..tables.high_band_count() {
            writer.write_bool(true);
        }
        writer.write_bool(true); // right harmonics
        for _ in 0..tables.high_band_count() {
            writer.write_bool(false);
        }
        writer.write_bool(true); // extended data
        writer.write(1, 4);
        writer.write(0xa5, 8);
        let frame_data_bits = writer.bits_written();
        let payload = SbrFillPayload {
            extension_type: EXT_SBR_DATA,
            transmitted_crc: None,
            header_present: false,
            header: None,
            frame_data: writer.finish(),
            frame_data_bits,
        };
        let mut parser = SbrStereoFrameParser::new(header, 44_100, 1024).unwrap();
        let frame = parser.parse(&payload).unwrap();
        assert!(frame.coupling);
        assert_eq!(frame.data_extra, Some((3, 9)));
        assert_eq!(frame.bits_read, frame_data_bits);
        assert!(frame.left_harmonics.iter().all(|enabled| *enabled));
        assert!(frame.right_harmonics.iter().all(|enabled| !*enabled));
        assert_eq!(frame.extended_data, [0xa5]);
        assert_eq!(frame.left.envelopes.len(), frame.right.envelopes.len());
        for (left, right) in frame.left_dequantized.envelope_energy[0]
            .iter()
            .zip(&frame.right_dequantized.envelope_energy[0])
        {
            assert!((left - right).abs() < 1.0e-12);
        }

        let mut usac = BitWriter::new();
        usac.write_bool(true); // coupling
        usac.write(0, 2); // FIXFIX shared grid
        usac.write(0, 2); // one envelope
        usac.write_bool(true); // high frequency resolution
        for _ in 0..tables.noise_band_count() {
            usac.write(1, 2);
        }
        usac.write(9, 6);
        for _ in 1..tables.high_band_count() {
            write_code(&mut usac, &level_zero);
        }
        usac.write(6, 5);
        for _ in 1..tables.noise_band_count() {
            write_code(&mut usac, &level_zero);
        }
        usac.write(6, 5);
        for _ in 1..tables.high_band_count() {
            write_code(&mut usac, &balance_zero);
        }
        usac.write(6, 5);
        for _ in 1..tables.noise_band_count() {
            write_code(&mut usac, &balance_zero);
        }
        usac.write_bool(false);
        usac.write_bool(false);
        let bits = usac.bits_written();
        let bytes = usac.finish();
        let mut reader = BitReader::with_bit_len(&bytes, bits).unwrap();
        let mut usac_parser =
            SbrStereoFrameParser::new_usac(frame.active_header.clone(), 44_100).unwrap();
        let usac_frame = usac_parser
            .parse_usac(&mut reader, true, false, false)
            .unwrap();
        assert_eq!(reader.bits_read(), bits);
        assert!(usac_frame.frame.coupling);
        assert_eq!(
            usac_frame.frame.left_control.grid,
            usac_frame.frame.right_control.grid
        );
        for (left, right) in usac_frame.frame.left_dequantized.envelope_energy[0]
            .iter()
            .zip(&usac_frame.frame.right_dequantized.envelope_energy[0])
        {
            assert!((left - right).abs() < 1.0e-12);
        }

        let mut truncated = payload.clone();
        truncated.frame_data_bits -= 1;
        let mut parser =
            SbrStereoFrameParser::new(frame.active_header.clone(), 44_100, 1024).unwrap();
        assert_eq!(
            parser.parse(&truncated).unwrap_err(),
            SbrError::TruncatedFrameData
        );
        for byte_len in 0..payload.frame_data.len() {
            let mut truncated = payload.clone();
            truncated.frame_data.truncate(byte_len);
            truncated.frame_data_bits = byte_len * 8;
            assert!(
                SbrStereoFrameParser::new(frame.active_header.clone(), 44_100, 1024,)
                    .unwrap()
                    .parse(&truncated)
                    .is_err()
            );
        }
        for bit_len in 0..bits {
            let mut reader = BitReader::with_bit_len(&bytes, bit_len).unwrap();
            assert!(
                SbrStereoFrameParser::new_usac(frame.active_header.clone(), 44_100)
                    .unwrap()
                    .parse_usac(&mut reader, true, false, false)
                    .is_err()
            );
        }
    }
}
