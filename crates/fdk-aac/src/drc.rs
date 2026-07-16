//! MPEG-D Unified DRC configuration, loudness metadata, and gain application.

use std::fmt;

use crate::bits::{BitError, BitReader};

const DELTA_GAIN_PROFILE_0_1_HUFFMAN: [[i8; 2]; 24] = [
    [1, 2],
    [3, 4],
    [-63, -65],
    [5, -66],
    [-64, 6],
    [-80, 7],
    [8, 9],
    [-68, 10],
    [11, 12],
    [-56, -67],
    [-61, 13],
    [-62, -69],
    [14, 15],
    [16, -72],
    [-71, 17],
    [-70, -60],
    [18, -59],
    [19, 20],
    [21, -79],
    [-57, -73],
    [22, -58],
    [-76, 23],
    [-75, -74],
    [-78, -77],
];

const DELTA_GAIN_PROFILE_2_HUFFMAN: [[i8; 2]; 48] = [
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8],
    [9, 10],
    [11, 12],
    [13, -65],
    [14, -64],
    [15, -66],
    [16, -67],
    [17, 18],
    [19, -68],
    [20, -63],
    [-69, 21],
    [-59, 22],
    [-61, -62],
    [-60, 23],
    [24, -58],
    [-70, -57],
    [-56, -71],
    [25, 26],
    [27, -55],
    [-72, 28],
    [-54, 29],
    [-53, 30],
    [-73, -52],
    [31, -74],
    [32, 33],
    [-75, 34],
    [-76, 35],
    [-51, 36],
    [-78, 37],
    [-77, 38],
    [-96, 39],
    [-48, 40],
    [-50, -79],
    [41, 42],
    [-80, -81],
    [-82, 43],
    [44, -49],
    [45, -84],
    [-83, -89],
    [-86, 46],
    [-90, -85],
    [-91, -93],
    [-92, 47],
    [-88, -87],
    [-95, -94],
];

const SLOPE_STEEPNESS_HUFFMAN: [[i8; 2]; 14] = [
    [1, -57],
    [-58, 2],
    [3, 4],
    [5, 6],
    [7, -56],
    [8, -60],
    [-61, -55],
    [9, -59],
    [10, -54],
    [-64, 11],
    [-51, 12],
    [-62, -50],
    [-63, 13],
    [-52, -53],
];

const SLOPE_STEEPNESS: [f32; 15] = [
    -3.0518, -1.2207, -0.4883, -0.1953, -0.0781, -0.0312, -0.005, 0.0, 0.005, 0.0312, 0.0781,
    0.1953, 0.4883, 1.2207, 3.0518,
];

const DOWNMIX_COEFFICIENT_V0: [f32; 16] = [
    1.0,
    0.944_060_86,
    0.891_250_9,
    0.841_395_14,
    0.794_328_2,
    0.749_894_2,
    0.707_945_76,
    0.668_343_9,
    0.630_957_37,
    0.595_662_1,
    0.562_341_33,
    0.530_884_44,
    0.501_187_2,
    0.421_696_5,
    0.354_813_4,
    0.0,
];

const DOWNMIX_COEFFICIENT_V1: [f32; 32] = [
    3.1622776602,
    1.9952623150,
    1.6788040181,
    1.4125375446,
    1.1885022274,
    1.0,
    0.9440608763,
    0.8912509381,
    0.8413951416,
    0.7943282347,
    0.7498942093,
    0.7079457844,
    0.6683439176,
    0.6309573445,
    0.5956621435,
    0.5623413252,
    0.5308844442,
    0.5011872336,
    0.4731512590,
    0.4466835922,
    0.4216965034,
    0.3981071706,
    0.3548133892,
    0.3162277660,
    0.2818382931,
    0.2511886432,
    0.1778279410,
    0.1,
    0.0562341325,
    0.0316227766,
    0.01,
    0.0,
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelLayout {
    pub base_channel_count: u8,
    pub defined_layout: Option<u8>,
    pub speaker_positions: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UniDrcConfig {
    pub sample_rate: Option<u32>,
    pub channel_layout: ChannelLayout,
    pub downmix_instructions: Vec<DownmixInstructions>,
    pub coefficients: Vec<DrcCoefficients>,
    pub instructions: Vec<DrcInstruction>,
    pub extension_present: bool,
    pub extensions: Vec<DrcExtension>,
    pub bits_read: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DrcExtension {
    pub extension_type: u8,
    pub bit_size: usize,
    /// Extension bits packed MSB first. Unused low bits in the last byte are zero.
    pub payload: Vec<u8>,
}

impl UniDrcConfig {
    /// Parse the configuration foundation, including the Basic DRC syntax
    /// that the reference decoder consumes but does not retain for rendering.
    pub fn parse_foundation(input: &[u8]) -> Result<Self, DrcError> {
        let mut reader = BitReader::new(input);
        let sample_rate = if reader.read_bool()? {
            Some(reader.read(18)? + 1000)
        } else {
            None
        };
        let downmix_count = reader.read_u8(7)?;
        let basic_present = reader.read_bool()?;
        let (basic_coefficients, basic_instructions) = if basic_present {
            (reader.read_u8(3)?, reader.read_u8(4)?)
        } else {
            (0, 0)
        };
        let coefficient_count = reader.read_u8(3)?;
        let instruction_count = reader.read_u8(6)?;
        let channel_layout = read_channel_layout(&mut reader)?;
        let mut downmix_instructions = Vec::with_capacity(downmix_count.min(6) as usize);
        for index in 0..downmix_count {
            let downmix = DownmixInstructions::parse_v0_from_reader(
                &mut reader,
                channel_layout.base_channel_count,
            )?;
            if index < 6 {
                downmix_instructions.push(downmix);
            }
        }
        for _ in 0..basic_coefficients {
            skip_drc_coefficients_basic(&mut reader)?;
        }
        for _ in 0..basic_instructions {
            skip_drc_instructions_basic(&mut reader)?;
        }
        let mut coefficients = Vec::with_capacity(coefficient_count.min(2) as usize);
        for index in 0..coefficient_count {
            let coefficient = DrcCoefficients::parse_v0_from_reader(&mut reader)?;
            if index < 2 {
                coefficients.push(coefficient);
            }
        }
        let mut instructions = Vec::with_capacity(instruction_count.min(12) as usize);
        for index in 0..instruction_count {
            let instruction = DrcInstruction::parse_v0_from_reader(
                &mut reader,
                &channel_layout,
                &downmix_instructions,
                &coefficients,
            )?;
            if index < 12 {
                instructions.push(instruction);
            }
        }
        let extension_present = reader.read_bool()?;
        let extensions = if extension_present {
            read_drc_extensions(&mut reader, 4)?
        } else {
            Vec::new()
        };
        for extension in extensions
            .iter()
            .filter(|extension| extension.extension_type == 2)
        {
            merge_v1_config_extension(
                extension,
                &channel_layout,
                &mut downmix_instructions,
                &mut coefficients,
                &mut instructions,
            )?;
        }
        Ok(Self {
            sample_rate,
            channel_layout,
            downmix_instructions,
            coefficients,
            instructions,
            extension_present,
            extensions,
            bits_read: reader.bits_read(),
        })
    }

    pub fn select_instruction(&self, request: DrcSelectionRequest) -> Option<&DrcInstruction> {
        if !request.enabled {
            return None;
        }
        self.instructions
            .iter()
            .filter(|instruction| {
                instruction.downmix_ids.contains(&request.downmix_id)
                    || instruction.downmix_ids.contains(&0x7f)
            })
            .filter(|instruction| {
                request.allow_dependent || instruction.depends_on_drc_set.is_none()
            })
            .filter(|instruction| !instruction.no_independent_use || request.allow_dependent)
            .filter(|instruction| {
                request.target_loudness.is_none_or(|target| {
                    let upper = instruction.target_loudness_upper.unwrap_or(0) as f32;
                    let lower = instruction.target_loudness_lower.unwrap_or(-63) as f32;
                    target <= upper && target >= lower
                })
            })
            .max_by_key(|instruction| {
                let effect_match = request.preferred_effect_mask != 0
                    && instruction.effect & request.preferred_effect_mask != 0;
                (
                    u8::from(effect_match),
                    u8::from(instruction.downmix_ids.contains(&request.downmix_id)),
                    u8::from(instruction.depends_on_drc_set.is_none()),
                )
            })
    }

    pub fn apply_instruction_f32(
        &self,
        instruction: &DrcInstruction,
        gain: &UniDrcGain,
        samples: &mut [f32],
        channels: usize,
    ) -> Result<(), DrcError> {
        apply_instruction_samples(
            self,
            instruction,
            gain,
            samples,
            channels,
            1.0,
            1.0,
            |sample, linear_gain| *sample *= linear_gain,
        )
    }

    pub fn apply_instruction_f32_scaled(
        &self,
        instruction: &DrcInstruction,
        gain: &UniDrcGain,
        samples: &mut [f32],
        channels: usize,
        attenuation_scale: f32,
        boost_scale: f32,
    ) -> Result<(), DrcError> {
        apply_instruction_samples(
            self,
            instruction,
            gain,
            samples,
            channels,
            attenuation_scale,
            boost_scale,
            |sample, linear_gain| *sample *= linear_gain,
        )
    }

    pub fn apply_instruction_i16(
        &self,
        instruction: &DrcInstruction,
        gain: &UniDrcGain,
        samples: &mut [i16],
        channels: usize,
    ) -> Result<(), DrcError> {
        apply_instruction_samples(
            self,
            instruction,
            gain,
            samples,
            channels,
            1.0,
            1.0,
            |sample, linear_gain| {
                *sample = (*sample as f32 * linear_gain)
                    .round()
                    .clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            },
        )
    }

    pub fn apply_instruction_i16_scaled(
        &self,
        instruction: &DrcInstruction,
        gain: &UniDrcGain,
        samples: &mut [i16],
        channels: usize,
        attenuation_scale: f32,
        boost_scale: f32,
    ) -> Result<(), DrcError> {
        apply_instruction_samples(
            self,
            instruction,
            gain,
            samples,
            channels,
            attenuation_scale,
            boost_scale,
            |sample, linear_gain| {
                *sample = (*sample as f32 * linear_gain)
                    .round()
                    .clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            },
        )
    }
}

fn skip_drc_coefficients_basic(reader: &mut BitReader<'_>) -> Result<(), DrcError> {
    reader.read_u8(4)?; // drcLocation
    reader.read_u8(7)?; // drcCharacteristic
    Ok(())
}

fn skip_drc_instructions_basic(reader: &mut BitReader<'_>) -> Result<(), DrcError> {
    reader.read_u8(6)?; // drcSetId
    reader.read_u8(4)?; // drcLocation
    reader.read_u8(7)?; // downmixId
    if reader.read_bool()? {
        let count = reader.read_u8(3)?;
        for _ in 0..count {
            reader.read_u8(7)?; // additionalDownmixId
        }
    }
    let effect = reader.read_u16(16)?;
    if effect & (DRC_EFFECT_DUCK_OTHER | DRC_EFFECT_DUCK_SELF) == 0 && reader.read_bool()? {
        reader.read_u8(8)?; // bsLimiterPeakTarget
    }
    if reader.read_bool()? {
        reader.read_u8(6)?; // bsDrcSetTargetLoudnessValueUpper
        if reader.read_bool()? {
            reader.read_u8(6)?; // bsDrcSetTargetLoudnessValueLower
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrcSelectionRequest {
    pub downmix_id: u8,
    pub target_loudness: Option<f32>,
    pub preferred_effect_mask: u16,
    pub allow_dependent: bool,
    pub enabled: bool,
    pub attenuation_scale: f32,
    pub boost_scale: f32,
    pub album_mode: bool,
}

/// MPEG-4 `dynamic_range_info()` carried by an `EXT_DYNAMIC_RANGE` fill
/// payload. This is the legacy AAC DRC syntax, distinct from MPEG-D UniDRC.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mpeg4DrcPayload {
    pub pce_instance_tag: Option<u8>,
    pub excluded_channels: Vec<bool>,
    pub interpolation_scheme: u8,
    pub band_top: Vec<u8>,
    pub program_reference_level: Option<u8>,
    /// Raw `dyn_rng_sgn | dyn_rng_ctl` values, one for each DRC band.
    pub dynamic_range: Vec<u8>,
}

impl Mpeg4DrcPayload {
    pub fn channel_is_excluded(&self, channel: usize) -> bool {
        self.excluded_channels
            .get(channel)
            .copied()
            .unwrap_or(false)
    }

    /// Gain for the common one-band form. Multi-band payloads have to be
    /// applied in the spectral/QMF domain and deliberately return `None`.
    pub fn one_band_gain(
        &self,
        channel: usize,
        attenuation_scale: f32,
        boost_scale: f32,
        target_reference_level: Option<u8>,
    ) -> Option<f32> {
        if self.dynamic_range.len() != 1 || self.channel_is_excluded(channel) {
            return None;
        }
        let control = self.dynamic_range[0];
        let magnitude = f32::from(control & 0x7f);
        let scale = if control & 0x80 != 0 {
            -attenuation_scale
        } else {
            boost_scale
        };
        let drc_gain = 2.0f32.powf(scale * magnitude / 24.0);
        let normalization = match (target_reference_level, self.program_reference_level) {
            (Some(target), Some(program)) => {
                2.0f32.powf(-(f32::from(target) - f32::from(program)) / 24.0)
            }
            _ => 1.0,
        };
        Some(drc_gain * normalization)
    }

    pub fn one_band_control_gain(
        &self,
        channel: usize,
        attenuation_scale: f32,
        boost_scale: f32,
    ) -> Option<f32> {
        if self.dynamic_range.len() != 1 || self.channel_is_excluded(channel) {
            return None;
        }
        let control = self.dynamic_range[0];
        let magnitude = f32::from(control & 0x7f);
        let scale = if control & 0x80 != 0 {
            -attenuation_scale
        } else {
            boost_scale
        };
        Some(2.0f32.powf(scale * magnitude / 24.0))
    }

    pub fn control_gains(
        &self,
        channel: usize,
        attenuation_scale: f32,
        boost_scale: f32,
    ) -> Option<Vec<f32>> {
        if self.channel_is_excluded(channel) {
            return None;
        }
        Some(
            self.dynamic_range
                .iter()
                .map(|&control| {
                    let magnitude = f32::from(control & 0x7f);
                    let scale = if control & 0x80 != 0 {
                        -attenuation_scale
                    } else {
                        boost_scale
                    };
                    2.0f32.powf(scale * magnitude / 24.0)
                })
                .collect(),
        )
    }

    pub fn normalization_gain(&self, target_reference_level: Option<u8>) -> f32 {
        match (target_reference_level, self.program_reference_level) {
            (Some(target), Some(program)) => {
                2.0f32.powf(-(f32::from(target) - f32::from(program)) / 24.0)
            }
            _ => 1.0,
        }
    }
}

/// DVB ancillary heavy-compression metadata carried in an AAC Data Stream
/// Element. This is the ETSI/ISO legacy `compression_value` syntax, not an
/// MPEG-4 `dynamic_range_info()` control word.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DvbAncillaryDrcPayload {
    pub presentation_mode: u8,
    pub compression_value: u8,
}

/// MPEG-4 advanced downmix metadata carried by the same DVB ancillary DSE as
/// [`DvbAncillaryDrcPayload`]. Optional fields retain their previous value in
/// the C decoder, so callers merge newly parsed values into their state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DvbAncillaryDownmixMetadata {
    pub pseudo_surround: bool,
    pub center_mix_level_index: Option<u8>,
    pub surround_mix_level_index: Option<u8>,
    pub downmix_a_index: Option<u8>,
    pub downmix_b_index: Option<u8>,
    pub lfe_mix_level_index: Option<u8>,
    pub stereo_downmix_gain_index: Option<u8>,
    pub five_channel_downmix_gain_index: Option<u8>,
}

impl DvbAncillaryDownmixMetadata {
    pub fn merge(&mut self, update: Self) {
        self.pseudo_surround = update.pseudo_surround;
        if update.center_mix_level_index.is_some() {
            self.center_mix_level_index = update.center_mix_level_index;
        }
        if update.surround_mix_level_index.is_some() {
            self.surround_mix_level_index = update.surround_mix_level_index;
        }
        if update.downmix_a_index.is_some() {
            self.downmix_a_index = update.downmix_a_index;
        }
        if update.downmix_b_index.is_some() {
            self.downmix_b_index = update.downmix_b_index;
        }
        if update.lfe_mix_level_index.is_some() {
            self.lfe_mix_level_index = update.lfe_mix_level_index;
        }
        if update.stereo_downmix_gain_index.is_some() {
            self.stereo_downmix_gain_index = update.stereo_downmix_gain_index;
        }
        if update.five_channel_downmix_gain_index.is_some() {
            self.five_channel_downmix_gain_index = update.five_channel_downmix_gain_index;
        }
    }
}

/// Parse MPEG-4 advanced downmix fields from a DVB ancillary DSE. A valid
/// header is returned even when no optional coefficient field is present,
/// because `stereo_downmix_mode` (pseudo-surround compatibility) is always
/// signalled in `bs_info`.
pub fn parse_dvb_ancillary_downmix(data: &[u8]) -> Option<DvbAncillaryDownmixMetadata> {
    if data.len() < 3 || data[0] != 0xbc {
        return None;
    }
    let bs_info = data[1];
    if bs_info >> 6 != 3 || bs_info & 1 != 0 {
        return None;
    }
    let status = data[2];
    if status >> 5 != 0 {
        return None;
    }

    let downmix_levels_present = status & 0x10 != 0;
    let extension_present = status & 0x08 != 0;
    let compression_present = status & 0x04 != 0;
    let coarse_timecode_present = status & 0x02 != 0;
    let fine_timecode_present = status & 0x01 != 0;
    let mut position = 3usize;
    let mut metadata = DvbAncillaryDownmixMetadata {
        pseudo_surround: bs_info & 0x02 != 0,
        ..DvbAncillaryDownmixMetadata::default()
    };

    if downmix_levels_present {
        let levels = *data.get(position)?;
        position += 1;
        if levels & 0x80 != 0 {
            metadata.center_mix_level_index = Some((levels >> 4) & 7);
        }
        if levels & 0x08 != 0 {
            metadata.surround_mix_level_index = Some(levels & 7);
        }
    }
    position = position.checked_add(usize::from(compression_present) * 2)?;
    position = position.checked_add(usize::from(coarse_timecode_present) * 2)?;
    position = position.checked_add(usize::from(fine_timecode_present) * 2)?;

    if extension_present {
        let extension = *data.get(position)?;
        position += 1;
        let extended_levels_present = extension & 0x40 != 0;
        let downmix_gain_present = extension & 0x20 != 0;
        let lfe_level_present = extension & 0x10 != 0;
        if extended_levels_present {
            let levels = *data.get(position)?;
            position += 1;
            metadata.downmix_a_index = Some((levels >> 5) & 7);
            metadata.downmix_b_index = Some((levels >> 2) & 7);
        }
        if downmix_gain_present {
            let five_channel = *data.get(position)?;
            let stereo = *data.get(position + 1)?;
            position += 2;
            metadata.five_channel_downmix_gain_index = Some(five_channel >> 1);
            metadata.stereo_downmix_gain_index = Some(stereo >> 1);
        }
        if lfe_level_present {
            metadata.lfe_mix_level_index = Some(*data.get(position)? >> 4);
        }
    }
    Some(metadata)
}

impl DvbAncillaryDrcPayload {
    /// Decode the DVB logarithmic compression value used by libAACdec.
    pub fn gain(self) -> f32 {
        if self.compression_value == 0x7f {
            return 1.0;
        }
        let coarse = f32::from(self.compression_value >> 4);
        let fine = f32::from(self.compression_value & 0x0f);
        let gain_db = 48.164 - 6.0206 * coarse - 0.4014 * fine;
        10.0f32.powf(gain_db / 20.0)
    }
}

/// Parse the DVB AAC ancillary-data header beginning with sync byte `0xbc`.
/// A payload is returned only when MPEG-4 audio and an enabled compression
/// value are signalled, matching `aacDecoder_drcReadCompression()`.
pub fn parse_dvb_ancillary_drc(data: &[u8]) -> Option<DvbAncillaryDrcPayload> {
    let (&sync, rest) = data.split_first()?;
    if sync != 0xbc || rest.len() < 2 {
        return None;
    }
    let bs_info = rest[0];
    if bs_info >> 6 != 3 || bs_info & 1 != 0 {
        return None;
    }
    let presentation_mode = (bs_info >> 2) & 3;
    let status = rest[1];
    if status >> 5 != 0 {
        return None;
    }
    let downmix_levels_present = status & 0x10 != 0;
    let compression_present = status & 0x04 != 0;
    if !compression_present {
        return None;
    }
    let mut position = 2 + usize::from(downmix_levels_present);
    if rest.len() < position + 2 {
        return None;
    }
    let audio_mode_and_switch = rest[position];
    position += 1;
    if audio_mode_and_switch >> 1 != 0 || audio_mode_and_switch & 1 == 0 {
        return None;
    }
    Some(DvbAncillaryDrcPayload {
        presentation_mode,
        compression_value: rest[position],
    })
}

/// Parse a complete AAC `fill_element()` and return its legacy MPEG-4 DRC
/// payload. Non-DRC fill elements are consumed and reported as `None`.
pub fn parse_mpeg4_drc_fill_element(
    reader: &mut BitReader<'_>,
) -> Result<Option<Mpeg4DrcPayload>, DrcError> {
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
    if payload.read_u8(4)? != 0x0b {
        return Ok(None);
    }

    Ok(Some(parse_mpeg4_drc_payload(&mut payload)?))
}

/// Parse `dynamic_range_info()` after the four-bit
/// `EXT_DYNAMIC_RANGE` extension type. ER AAC carries this syntax directly
/// instead of wrapping it in a GA `fill_element()`.
pub fn parse_mpeg4_drc_payload(payload: &mut BitReader<'_>) -> Result<Mpeg4DrcPayload, DrcError> {
    let pce_instance_tag = if payload.read_bool()? {
        let tag = payload.read_u8(4)?;
        payload.read_u8(4)?; // drc_tag_reserved_bits
        Some(tag)
    } else {
        None
    };

    let mut excluded_channels = Vec::new();
    if payload.read_bool()? {
        loop {
            for _ in 0..7 {
                excluded_channels.push(payload.read_bool()?);
            }
            if !payload.read_bool()? {
                break;
            }
        }
    }

    let mut interpolation_scheme = 0;
    let band_count = if payload.read_bool()? {
        let count = usize::from(payload.read_u8(4)?) + 1;
        interpolation_scheme = payload.read_u8(4)?;
        count
    } else {
        1
    };
    let mut band_top = Vec::with_capacity(band_count);
    if band_count == 1 {
        band_top.push(255);
    } else {
        for _ in 0..band_count {
            band_top.push(payload.read_u8(8)?);
        }
    }

    let program_reference_level = if payload.read_bool()? {
        let level = payload.read_u8(7)?;
        payload.read_bool()?; // reserved
        Some(level)
    } else {
        None
    };
    let mut dynamic_range = Vec::with_capacity(band_count);
    for _ in 0..band_count {
        dynamic_range.push(payload.read_u8(8)?);
    }
    Ok(Mpeg4DrcPayload {
        pce_instance_tag,
        excluded_channels,
        interpolation_scheme,
        band_top,
        program_reference_level,
        dynamic_range,
    })
}

impl Default for DrcSelectionRequest {
    fn default() -> Self {
        Self {
            downmix_id: 0,
            target_loudness: None,
            preferred_effect_mask: 0,
            allow_dependent: false,
            enabled: true,
            attenuation_scale: 1.0,
            boost_scale: 1.0,
            album_mode: false,
        }
    }
}

fn apply_instruction_samples<T>(
    config: &UniDrcConfig,
    instruction: &DrcInstruction,
    gain: &UniDrcGain,
    samples: &mut [T],
    channels: usize,
    attenuation_scale: f32,
    boost_scale: f32,
    mut apply: impl FnMut(&mut T, f32),
) -> Result<(), DrcError> {
    if channels == 0
        || !samples.len().is_multiple_of(channels)
        || instruction.gain_set_index_per_channel.len() < channels
    {
        return Err(DrcError::InstructionLayoutMismatch);
    }
    let coefficients = config
        .coefficients
        .iter()
        .find(|coefficients| coefficients.drc_location == instruction.drc_location)
        .ok_or(DrcError::MissingDrcCoefficients(instruction.drc_location))?;
    let mut unique_sets = Vec::new();
    for &set in &instruction.gain_set_index_per_channel {
        if !unique_sets.contains(&set) {
            unique_sets.push(set);
        }
    }
    for frame in 0..samples.len() / channels {
        for channel in 0..channels {
            let set_index = instruction.gain_set_index_per_channel[channel];
            if set_index < 0 {
                continue;
            }
            let gain_set = coefficients
                .gain_sets
                .get(set_index as usize)
                .ok_or(DrcError::MissingGainSet(set_index as usize))?;
            let sequence_index = gain_set
                .bands
                .first()
                .ok_or(DrcError::InvalidBandCount(0))?
                .sequence_index as usize;
            let sequence = gain
                .sequences
                .get(sequence_index)
                .ok_or(DrcError::MissingGainSequence(sequence_index))?;
            let mut gain_db = sequence.gain_db_at(frame as i32);
            if instruction.effect & (DRC_EFFECT_DUCK_OTHER | DRC_EFFECT_DUCK_SELF) != 0 {
                gain_db *= instruction
                    .ducking_modifications
                    .get(channel)
                    .map_or(1.0, |modification| modification.scaling);
            } else if let Some(modification) = unique_sets
                .iter()
                .position(|&set| set == set_index)
                .and_then(|group| instruction.gain_modifications.get(group))
            {
                gain_db = if gain_db < 0.0 {
                    gain_db * modification.attenuation_scaling
                } else {
                    gain_db * modification.amplification_scaling
                } + modification.gain_offset_db;
            }
            gain_db *= if gain_db < 0.0 {
                attenuation_scale
            } else {
                boost_scale
            };
            apply(
                &mut samples[frame * channels + channel],
                10.0f32.powf(gain_db / 20.0),
            );
        }
    }
    Ok(())
}

fn read_drc_extensions(
    reader: &mut BitReader<'_>,
    bit_size_len_bits: usize,
) -> Result<Vec<DrcExtension>, DrcError> {
    let mut extensions = Vec::new();
    loop {
        let extension_type = reader.read_u8(4)?;
        if extension_type == 0 {
            return Ok(extensions);
        }
        if extensions.len() >= 7 {
            return Err(DrcError::TooManyExtensions);
        }
        let bit_size_len = reader.read_u8(bit_size_len_bits)? as usize + 4;
        let bit_size = reader.read(bit_size_len)? as usize + 1;
        if bit_size > reader.remaining_bits() {
            return Err(BitError::UnexpectedEof {
                needed_bits: bit_size,
                remaining_bits: reader.remaining_bits(),
            }
            .into());
        }
        let mut payload = vec![0u8; bit_size.div_ceil(8)];
        for bit in 0..bit_size {
            if reader.read_bool()? {
                payload[bit / 8] |= 1 << (7 - bit % 8);
            }
        }
        extensions.push(DrcExtension {
            extension_type,
            bit_size,
            payload,
        });
    }
}

fn merge_v1_config_extension(
    extension: &DrcExtension,
    channel_layout: &ChannelLayout,
    downmixes: &mut Vec<DownmixInstructions>,
    coefficients: &mut Vec<DrcCoefficients>,
    instructions: &mut Vec<DrcInstruction>,
) -> Result<(), DrcError> {
    let mut reader = BitReader::new(&extension.payload);
    if reader.read_bool()? {
        let count = reader.read_u8(7)?;
        for index in 0..count {
            let downmix = DownmixInstructions::parse_v1_from_reader(
                &mut reader,
                channel_layout.base_channel_count,
            )?;
            if downmixes.len() < 6 && index < 6 {
                downmixes.push(downmix);
            }
        }
    }
    if reader.read_bool()? {
        let coefficient_count = reader.read_u8(3)?;
        for index in 0..coefficient_count {
            let coefficient = DrcCoefficients::parse_v1_from_reader(&mut reader)?;
            if coefficients.len() < 2 && index < 2 {
                coefficients.push(coefficient);
            }
        }
        let instruction_count = reader.read_u8(6)?;
        for index in 0..instruction_count {
            let instruction = DrcInstruction::parse_v1_from_reader(
                &mut reader,
                channel_layout,
                downmixes,
                coefficients,
            )?;
            if instructions.len() < 12 && index < 12 {
                instructions.push(instruction);
            }
        }
    }
    let loud_eq_present = reader.read_bool()?;
    if loud_eq_present {
        let count = reader.read_u8(4)?;
        for _ in 0..count {
            skip_loud_eq_instruction(&mut reader)?;
        }
    }
    let eq_present = reader.read_bool()?;
    if eq_present {
        skip_eq_coefficients(&mut reader)?;
        let instruction_count = reader.read_u8(4)?;
        for _ in 0..instruction_count {
            skip_eq_instruction(&mut reader, channel_layout, downmixes)?;
        }
    }
    if reader.bits_read() != extension.bit_size {
        return Err(DrcError::ExtensionSizeMismatch {
            declared: extension.bit_size,
            consumed: reader.bits_read(),
        });
    }
    Ok(())
}

fn skip_bits(reader: &mut BitReader<'_>, mut count: usize) -> Result<(), DrcError> {
    while count != 0 {
        let chunk = count.min(32);
        reader.read(chunk)?;
        count -= chunk;
    }
    Ok(())
}

fn skip_eq_subband_gain_spline(reader: &mut BitReader<'_>) -> Result<(), DrcError> {
    let node_count = reader.read_u8(5)? as usize + 2;
    for _ in 0..node_count {
        if !reader.read_bool()? {
            skip_bits(reader, 4)?;
        }
    }
    skip_bits(reader, 4 * (node_count - 1))?;
    let slope_bits = match reader.read_u8(2)? {
        0 => 5,
        1 | 2 => 4,
        _ => 3,
    };
    skip_bits(reader, slope_bits)?;
    skip_bits(reader, 5 * (node_count - 1))
}

fn skip_eq_coefficients(reader: &mut BitReader<'_>) -> Result<(), DrcError> {
    if reader.read_bool()? {
        skip_bits(reader, 8)?;
    }
    let filter_block_count = reader.read_u8(6)?;
    for _ in 0..filter_block_count {
        let element_count = reader.read_u8(6)?;
        for _ in 0..element_count {
            skip_bits(reader, 6)?;
            if reader.read_bool()? {
                skip_bits(reader, 10)?;
            }
        }
    }
    let td_element_count = reader.read_u8(6)?;
    for _ in 0..td_element_count {
        if !reader.read_bool()? {
            let radius_one = reader.read_u8(3)? as usize;
            let real_zero = reader.read_u8(6)? as usize;
            let generic_zero = reader.read_u8(6)? as usize;
            let real_pole = reader.read_u8(4)? as usize;
            let complex_pole = reader.read_u8(4)? as usize;
            skip_bits(
                reader,
                2 * radius_one
                    + 8 * real_zero
                    + 14 * generic_zero
                    + 8 * real_pole
                    + 14 * complex_pole,
            )?;
        } else {
            let order = reader.read_u8(7)? as usize;
            skip_bits(reader, 1 + (order / 2 + 1) * 11)?;
        }
    }
    let subband_gains_count = reader.read_u8(6)?;
    if subband_gains_count != 0 {
        let spline = reader.read_bool()?;
        let gain_count = match reader.read_u8(4)? {
            1 => 32,
            2 => 39,
            3 => 64,
            4 => 71,
            5 => 128,
            6 => 135,
            _ => reader.read_u8(8)? as usize + 1,
        };
        for _ in 0..subband_gains_count {
            if spline {
                skip_eq_subband_gain_spline(reader)?;
            } else {
                skip_bits(reader, gain_count * 9)?;
            }
        }
    }
    Ok(())
}

fn skip_td_filter_cascade(
    reader: &mut BitReader<'_>,
    channel_group_count: usize,
) -> Result<(), DrcError> {
    for _ in 0..channel_group_count {
        if reader.read_bool()? {
            skip_bits(reader, 10)?;
        }
        let filter_block_count = reader.read_u8(4)? as usize;
        skip_bits(reader, filter_block_count * 7)?;
    }
    if reader.read_bool()? {
        skip_bits(
            reader,
            channel_group_count.saturating_mul(channel_group_count.saturating_sub(1)) / 2,
        )?;
    }
    Ok(())
}

fn skip_eq_instruction(
    reader: &mut BitReader<'_>,
    channel_layout: &ChannelLayout,
    downmixes: &[DownmixInstructions],
) -> Result<(), DrcError> {
    skip_bits(reader, 6 + 4)?; // set ID and complexity
    let mut downmix_id = 0;
    let mut apply_to_downmix = false;
    let mut additional_downmix_count = 0;
    if reader.read_bool()? {
        downmix_id = reader.read_u8(7)?;
        apply_to_downmix = reader.read_bool()?;
        if reader.read_bool()? {
            additional_downmix_count = reader.read_u8(7)?;
            skip_bits(reader, additional_downmix_count as usize * 7)?;
        }
    }
    skip_bits(reader, 6)?; // drcSetId
    if reader.read_bool()? {
        let additional_count = reader.read_u8(6)? as usize;
        skip_bits(reader, additional_count * 6)?;
    }
    skip_bits(reader, 16)?; // purpose
    if reader.read_bool()? {
        skip_bits(reader, 6)?;
    } else {
        skip_bits(reader, 1)?;
    }
    let channel_count = if apply_to_downmix
        && downmix_id != 0
        && downmix_id != 0x7f
        && additional_downmix_count == 0
    {
        downmixes
            .iter()
            .find(|downmix| downmix.downmix_id == downmix_id)
            .ok_or(DrcError::MissingDownmixInstruction(downmix_id))?
            .target_channel_count as usize
    } else if downmix_id == 0x7f || additional_downmix_count > 1 {
        1
    } else {
        channel_layout.base_channel_count as usize
    };
    if channel_count > 8 {
        return Err(DrcError::TooManyChannels(channel_count as u8));
    }
    let mut groups = Vec::with_capacity(channel_count);
    for _ in 0..channel_count {
        let group = reader.read_u8(7)?;
        if !groups.contains(&group) {
            groups.push(group);
        }
    }
    if reader.read_bool()? {
        skip_td_filter_cascade(reader, groups.len())?;
    }
    if reader.read_bool()? {
        skip_bits(reader, groups.len() * 6)?;
    }
    if reader.read_bool()? {
        skip_bits(reader, 5)?;
    }
    Ok(())
}

fn skip_loud_eq_instruction(reader: &mut BitReader<'_>) -> Result<(), DrcError> {
    reader.read(4)?; // loudEqSetId
    reader.read(4)?; // drcLocation
    if reader.read_bool()? {
        reader.read(7)?; // downmixId
        if reader.read_bool()? {
            let count = reader.read_u8(7)?;
            for _ in 0..count {
                reader.read(7)?;
            }
        }
    }
    if reader.read_bool()? {
        reader.read(6)?; // drcSetId
        if reader.read_bool()? {
            let count = reader.read_u8(6)?;
            for _ in 0..count {
                reader.read(6)?;
            }
        }
    }
    if reader.read_bool()? {
        reader.read(6)?; // eqSetId
        if reader.read_bool()? {
            let count = reader.read_u8(6)?;
            for _ in 0..count {
                reader.read(6)?;
            }
        }
    }
    reader.read(1)?; // loudnessAfterDrc
    reader.read(1)?; // loudnessAfterEq
    let sequence_count = reader.read_u8(6)?;
    for _ in 0..sequence_count {
        reader.read(6)?; // gainSequenceIndex
        if reader.read_bool()? {
            reader.read(7)?; // CICP characteristic
        } else {
            reader.read(4)?; // left custom characteristic
            reader.read(4)?; // right custom characteristic
        }
        reader.read(6)?; // frequencyRangeIndex
        reader.read(3)?; // scaling
        reader.read(5)?; // offset
    }
    Ok(())
}

fn read_channel_layout(reader: &mut BitReader<'_>) -> Result<ChannelLayout, DrcError> {
    let base_channel_count = reader.read_u8(7)?;
    if base_channel_count > 8 {
        return Err(DrcError::TooManyChannels(base_channel_count));
    }
    let layout_signaling_present = reader.read_bool()?;
    let mut defined_layout = None;
    let mut speaker_positions = Vec::new();
    if layout_signaling_present {
        let layout = reader.read_u8(8)?;
        defined_layout = Some(layout);
        if layout == 0 {
            for _ in 0..base_channel_count {
                speaker_positions.push(reader.read_u8(7)?);
            }
        }
    }
    Ok(ChannelLayout {
        base_channel_count,
        defined_layout,
        speaker_positions,
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct DownmixInstructions {
    pub downmix_id: u8,
    pub target_channel_count: u8,
    pub target_layout: u8,
    pub coefficient_offset: u8,
    /// Target-major matrix: `coefficients[target][source]`.
    pub coefficients: Option<Vec<Vec<f32>>>,
}

impl DownmixInstructions {
    pub fn parse_v0(input: &[u8], base_channel_count: u8) -> Result<Self, DrcError> {
        Self::parse_v0_from_reader(&mut BitReader::new(input), base_channel_count)
    }

    fn parse_v0_from_reader(
        reader: &mut BitReader<'_>,
        base_channel_count: u8,
    ) -> Result<Self, DrcError> {
        let downmix_id = reader.read_u8(7)?;
        let target_channel_count = reader.read_u8(7)?;
        let target_layout = reader.read_u8(8)?;
        if base_channel_count > 8 || target_channel_count > 8 {
            return Err(DrcError::InvalidDownmixDimensions {
                source_channels: base_channel_count,
                target_channels: target_channel_count,
            });
        }
        let coefficients = if reader.read_bool()? {
            let mut matrix =
                vec![vec![0.0; base_channel_count as usize]; target_channel_count as usize];
            for target in &mut matrix {
                for coefficient in target {
                    *coefficient = DOWNMIX_COEFFICIENT_V0[reader.read_u8(4)? as usize];
                }
            }
            Some(matrix)
        } else {
            None
        };
        Ok(Self {
            downmix_id,
            target_channel_count,
            target_layout,
            coefficient_offset: 0,
            coefficients,
        })
    }

    pub fn parse_v1(input: &[u8], base_channel_count: u8) -> Result<Self, DrcError> {
        Self::parse_v1_from_reader(&mut BitReader::new(input), base_channel_count)
    }

    fn parse_v1_from_reader(
        reader: &mut BitReader<'_>,
        base_channel_count: u8,
    ) -> Result<Self, DrcError> {
        let downmix_id = reader.read_u8(7)?;
        let target_channel_count = reader.read_u8(7)?;
        let target_layout = reader.read_u8(8)?;
        if base_channel_count > 8 || target_channel_count > 8 {
            return Err(DrcError::InvalidDownmixDimensions {
                source_channels: base_channel_count,
                target_channels: target_channel_count,
            });
        }
        let (coefficient_offset, coefficients) = if reader.read_bool()? {
            let offset = reader.read_u8(4)?;
            let mut matrix =
                vec![vec![0.0; base_channel_count as usize]; target_channel_count as usize];
            for target in &mut matrix {
                for coefficient in target {
                    *coefficient = DOWNMIX_COEFFICIENT_V1[reader.read_u8(5)? as usize];
                }
            }
            (offset, Some(matrix))
        } else {
            (0, None)
        };
        Ok(Self {
            downmix_id,
            target_channel_count,
            target_layout,
            coefficient_offset,
            coefficients,
        })
    }

    pub fn apply_interleaved_f32(
        &self,
        input: &[f32],
        source_channels: usize,
    ) -> Result<Vec<f32>, DrcError> {
        let matrix = self
            .coefficients
            .as_ref()
            .ok_or(DrcError::DownmixCoefficientsAbsent)?;
        if source_channels == 0
            || input.len() % source_channels != 0
            || matrix.len() != self.target_channel_count as usize
            || matrix.iter().any(|row| row.len() != source_channels)
        {
            return Err(DrcError::DownmixLayoutMismatch);
        }
        let frames = input.len() / source_channels;
        let mut output = vec![0.0; frames * matrix.len()];
        for frame in 0..frames {
            for (target, row) in matrix.iter().enumerate() {
                output[frame * matrix.len() + target] = row
                    .iter()
                    .enumerate()
                    .map(|(source, coefficient)| {
                        input[frame * source_channels + source] * coefficient
                    })
                    .sum();
            }
        }
        Ok(output)
    }

    pub fn apply_interleaved_i16(
        &self,
        input: &[i16],
        source_channels: usize,
    ) -> Result<Vec<i16>, DrcError> {
        let normalized = input
            .iter()
            .map(|&sample| sample as f32)
            .collect::<Vec<_>>();
        Ok(self
            .apply_interleaved_f32(&normalized, source_channels)?
            .into_iter()
            .map(|sample| sample.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16)
            .collect())
    }
}

const DRC_EFFECT_DUCK_OTHER: u16 = 0x0400;
const DRC_EFFECT_DUCK_SELF: u16 = 0x0800;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GainModification {
    pub target_characteristic_left: Option<u8>,
    pub target_characteristic_right: Option<u8>,
    pub attenuation_scaling: f32,
    pub amplification_scaling: f32,
    pub gain_offset_db: f32,
    pub shape_filter_index: Option<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DuckingModification {
    pub scaling: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DrcInstruction {
    pub drc_set_id: u8,
    pub complexity_level: u8,
    pub drc_location: u8,
    pub downmix_ids: Vec<u8>,
    pub apply_to_downmix: bool,
    pub effect: u16,
    pub limiter_peak_target_db: Option<f32>,
    pub target_loudness_upper: Option<i8>,
    pub target_loudness_lower: Option<i8>,
    pub depends_on_drc_set: Option<u8>,
    pub no_independent_use: bool,
    pub requires_eq: bool,
    pub channel_count: u8,
    pub gain_set_index_per_channel: Vec<i8>,
    pub gain_modifications: Vec<GainModification>,
    pub gain_modifications_per_band: Vec<Vec<GainModification>>,
    pub ducking_modifications: Vec<DuckingModification>,
}

impl DrcInstruction {
    fn parse_v0_from_reader(
        reader: &mut BitReader<'_>,
        channel_layout: &ChannelLayout,
        downmixes: &[DownmixInstructions],
        coefficients: &[DrcCoefficients],
    ) -> Result<Self, DrcError> {
        let drc_set_id = reader.read_u8(6)?;
        let drc_location = reader.read_u8(4)?;
        let first_downmix_id = reader.read_u8(7)?;
        let mut downmix_ids = vec![first_downmix_id];
        if reader.read_bool()? {
            let additional = reader.read_u8(3)?;
            for _ in 0..additional {
                downmix_ids.push(reader.read_u8(7)?);
            }
        }
        let apply_to_downmix = first_downmix_id != 0;
        let effect = reader.read_u16(16)?;
        let ducking = effect & (DRC_EFFECT_DUCK_OTHER | DRC_EFFECT_DUCK_SELF) != 0;
        let limiter_peak_target_db = if !ducking && reader.read_bool()? {
            Some(-(reader.read_u8(8)? as f32) * 0.125)
        } else {
            None
        };
        let (target_loudness_upper, target_loudness_lower) = if reader.read_bool()? {
            let upper = reader.read_u8(6)? as i8 - 63;
            let lower = if reader.read_bool()? {
                Some(reader.read_u8(6)? as i8 - 63)
            } else {
                None
            };
            (Some(upper), lower)
        } else {
            (None, None)
        };
        let (depends_on_drc_set, no_independent_use) = if reader.read_bool()? {
            (Some(reader.read_u8(6)?), false)
        } else {
            (None, reader.read_bool()?)
        };
        let mut channel_count = channel_layout.base_channel_count;
        if first_downmix_id != 0 && first_downmix_id != 0x7f && downmix_ids.len() == 1 {
            channel_count = downmixes
                .iter()
                .find(|downmix| downmix.downmix_id == first_downmix_id)
                .map_or(1, |downmix| downmix.target_channel_count);
        } else if first_downmix_id == 0x7f || downmix_ids.len() > 1 {
            channel_count = 8;
        }
        let mut gain_set_index_per_channel = Vec::with_capacity(channel_count as usize);
        let mut ducking_modifications = Vec::new();
        while gain_set_index_per_channel.len() < channel_count as usize {
            let gain_set_index = reader.read_u8(6)? as i8 - 1;
            let duck = if ducking {
                let scaling = if reader.read_bool()? {
                    let raw = reader.read_u8(4)?;
                    let mu = (raw & 7) as f32;
                    if raw >> 3 != 0 {
                        1.0 - 0.125 * (1.0 + mu)
                    } else {
                        1.0 + 0.125 * (1.0 + mu)
                    }
                } else {
                    1.0
                };
                Some(DuckingModification { scaling })
            } else {
                None
            };
            gain_set_index_per_channel.push(gain_set_index);
            if let Some(duck) = duck {
                ducking_modifications.push(duck);
            }
            if reader.read_bool()? {
                let repeats = reader.read_u8(5)? as usize + 1;
                for _ in 0..repeats {
                    if gain_set_index_per_channel.len() >= channel_count as usize {
                        return Err(DrcError::InstructionChannelOverflow);
                    }
                    gain_set_index_per_channel.push(gain_set_index);
                    if let Some(duck) = duck {
                        ducking_modifications.push(duck);
                    }
                }
            }
        }
        let mut gain_modifications = Vec::new();
        if !ducking {
            let mut unique_sets = Vec::new();
            for &set in &gain_set_index_per_channel {
                if !unique_sets.contains(&set) {
                    unique_sets.push(set);
                }
            }
            for set in unique_sets {
                let _band_count = coefficients
                    .iter()
                    .find(|coefficient| coefficient.drc_location == drc_location)
                    .and_then(|coefficient| coefficient.gain_sets.get(set.max(0) as usize))
                    .map_or(1, |gain_set| gain_set.bands.len());
                gain_modifications.push(read_gain_modification_v0(reader)?);
            }
        }
        Ok(Self {
            drc_set_id,
            complexity_level: 2,
            drc_location,
            downmix_ids,
            apply_to_downmix,
            effect,
            limiter_peak_target_db,
            target_loudness_upper,
            target_loudness_lower,
            depends_on_drc_set,
            no_independent_use,
            requires_eq: false,
            channel_count,
            gain_set_index_per_channel,
            gain_modifications,
            gain_modifications_per_band: Vec::new(),
            ducking_modifications,
        })
    }

    pub fn parse_v1(
        input: &[u8],
        channel_layout: &ChannelLayout,
        downmixes: &[DownmixInstructions],
        coefficients: &[DrcCoefficients],
    ) -> Result<Self, DrcError> {
        Self::parse_v1_from_reader(
            &mut BitReader::new(input),
            channel_layout,
            downmixes,
            coefficients,
        )
    }

    fn parse_v1_from_reader(
        reader: &mut BitReader<'_>,
        channel_layout: &ChannelLayout,
        downmixes: &[DownmixInstructions],
        coefficients: &[DrcCoefficients],
    ) -> Result<Self, DrcError> {
        let drc_set_id = reader.read_u8(6)?;
        let complexity_level = reader.read_u8(4)?;
        let drc_location = reader.read_u8(4)?;
        let (downmix_ids, apply_to_downmix) = if reader.read_bool()? {
            let first = reader.read_u8(7)?;
            let apply = reader.read_bool()?;
            let mut ids = vec![first];
            if reader.read_bool()? {
                let additional = reader.read_u8(3)?;
                for _ in 0..additional {
                    ids.push(reader.read_u8(7)?);
                }
            }
            (ids, apply)
        } else {
            (vec![0], false)
        };
        let effect = reader.read_u16(16)?;
        let ducking = effect & (DRC_EFFECT_DUCK_OTHER | DRC_EFFECT_DUCK_SELF) != 0;
        let limiter_peak_target_db = if !ducking && reader.read_bool()? {
            Some(-(reader.read_u8(8)? as f32) * 0.125)
        } else {
            None
        };
        let (target_loudness_upper, target_loudness_lower) = if reader.read_bool()? {
            let upper = reader.read_u8(6)? as i8 - 63;
            let lower = if reader.read_bool()? {
                Some(reader.read_u8(6)? as i8 - 63)
            } else {
                None
            };
            (Some(upper), lower)
        } else {
            (None, None)
        };
        let (depends_on_drc_set, no_independent_use) = if reader.read_bool()? {
            (Some(reader.read_u8(6)?), false)
        } else {
            (None, reader.read_bool()?)
        };
        let requires_eq = reader.read_bool()?;
        let first_downmix_id = downmix_ids[0];
        let mut channel_count = channel_layout.base_channel_count;
        let mut coded_channel_count = channel_count;
        let derive_channel_count = apply_to_downmix
            && first_downmix_id != 0
            && first_downmix_id != 0x7f
            && downmix_ids.len() == 1
            && downmixes.is_empty();
        if apply_to_downmix
            && first_downmix_id != 0
            && first_downmix_id != 0x7f
            && downmix_ids.len() == 1
        {
            coded_channel_count = downmixes
                .iter()
                .find(|downmix| downmix.downmix_id == first_downmix_id)
                .map_or(1, |downmix| downmix.target_channel_count);
            channel_count = coded_channel_count;
        } else if apply_to_downmix && (first_downmix_id == 0x7f || downmix_ids.len() > 1) {
            channel_count = 8;
            coded_channel_count = 1;
        }
        let mut gain_set_index_per_channel = Vec::with_capacity(channel_count as usize);
        let mut ducking_modifications = Vec::new();
        while gain_set_index_per_channel.len() < coded_channel_count as usize {
            let gain_set_index = reader.read_u8(6)? as i8 - 1;
            let duck = if ducking {
                Some(read_ducking_modification(reader)?)
            } else {
                None
            };
            gain_set_index_per_channel.push(gain_set_index);
            if let Some(duck) = duck {
                ducking_modifications.push(duck);
            }
            if reader.read_bool()? {
                let repeats = reader.read_u8(5)? as usize + 1;
                if derive_channel_count {
                    channel_count = (repeats + 1) as u8;
                    coded_channel_count = channel_count;
                }
                for _ in 0..repeats {
                    if gain_set_index_per_channel.len() >= 8 {
                        return Err(DrcError::InstructionChannelOverflow);
                    }
                    gain_set_index_per_channel.push(gain_set_index);
                    if let Some(duck) = duck {
                        ducking_modifications.push(duck);
                    }
                }
            }
        }
        if gain_set_index_per_channel.len() > channel_count as usize {
            return Err(DrcError::InstructionChannelOverflow);
        }
        if coded_channel_count == 1 && channel_count > 1 {
            gain_set_index_per_channel
                .resize(channel_count as usize, gain_set_index_per_channel[0]);
            if ducking {
                ducking_modifications.resize(channel_count as usize, ducking_modifications[0]);
            }
        }
        let mut gain_modifications_per_band = Vec::new();
        let mut gain_modifications = Vec::new();
        if !ducking {
            let mut unique_sets = Vec::new();
            for &set in &gain_set_index_per_channel {
                if !unique_sets.contains(&set) {
                    unique_sets.push(set);
                }
            }
            for set in unique_sets {
                let band_count = coefficients
                    .iter()
                    .find(|coefficient| coefficient.drc_location == drc_location)
                    .and_then(|coefficient| coefficient.gain_sets.get(set.max(0) as usize))
                    .map_or(1, |gain_set| gain_set.bands.len());
                let modifications = read_gain_modifications_v1(reader, band_count)?;
                gain_modifications.push(modifications[0]);
                gain_modifications_per_band.push(modifications);
            }
        }
        Ok(Self {
            drc_set_id,
            complexity_level,
            drc_location,
            downmix_ids,
            apply_to_downmix,
            effect,
            limiter_peak_target_db,
            target_loudness_upper,
            target_loudness_lower,
            depends_on_drc_set,
            no_independent_use,
            requires_eq,
            channel_count,
            gain_set_index_per_channel,
            gain_modifications,
            gain_modifications_per_band,
            ducking_modifications,
        })
    }
}

fn read_ducking_modification(reader: &mut BitReader<'_>) -> Result<DuckingModification, DrcError> {
    let scaling = if reader.read_bool()? {
        let raw = reader.read_u8(4)?;
        let mu = (raw & 7) as f32;
        if raw >> 3 != 0 {
            1.0 - 0.125 * (1.0 + mu)
        } else {
            1.0 + 0.125 * (1.0 + mu)
        }
    } else {
        1.0
    };
    Ok(DuckingModification { scaling })
}

fn read_gain_modifications_v1(
    reader: &mut BitReader<'_>,
    band_count: usize,
) -> Result<Vec<GainModification>, DrcError> {
    let mut modifications = Vec::with_capacity(band_count);
    for _ in 0..band_count {
        let target_characteristic_left =
            reader.read_bool()?.then(|| reader.read_u8(4)).transpose()?;
        let target_characteristic_right =
            reader.read_bool()?.then(|| reader.read_u8(4)).transpose()?;
        let (attenuation_scaling, amplification_scaling) = if reader.read_bool()? {
            (
                reader.read_u8(4)? as f32 * 0.125,
                reader.read_u8(4)? as f32 * 0.125,
            )
        } else {
            (1.0, 1.0)
        };
        let gain_offset_db = if reader.read_bool()? {
            let negative = reader.read_bool()?;
            let magnitude = (reader.read_u8(5)? as f32 + 1.0) * 0.25;
            if negative {
                -magnitude
            } else {
                magnitude
            }
        } else {
            0.0
        };
        modifications.push(GainModification {
            target_characteristic_left,
            target_characteristic_right,
            attenuation_scaling,
            amplification_scaling,
            gain_offset_db,
            shape_filter_index: None,
        });
    }
    if band_count == 1 && reader.read_bool()? {
        modifications[0].shape_filter_index = Some(reader.read_u8(4)?);
    }
    Ok(modifications)
}

fn read_gain_modification_v0(reader: &mut BitReader<'_>) -> Result<GainModification, DrcError> {
    let (attenuation_scaling, amplification_scaling) = if reader.read_bool()? {
        (
            reader.read_u8(4)? as f32 * 0.125,
            reader.read_u8(4)? as f32 * 0.125,
        )
    } else {
        (1.0, 1.0)
    };
    let gain_offset_db = if reader.read_bool()? {
        let negative = reader.read_bool()?;
        let value = (reader.read_u8(5)? as f32 + 1.0) * 0.25;
        if negative {
            -value
        } else {
            value
        }
    } else {
        0.0
    };
    Ok(GainModification {
        target_characteristic_left: None,
        target_characteristic_right: None,
        attenuation_scaling,
        amplification_scaling,
        gain_offset_db,
        shape_filter_index: None,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GainCodingProfile {
    Regular,
    Fading,
    Clipping,
    Constant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GainInterpolationType {
    Linear,
    Spline,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BandBorder {
    CrossoverFrequencyIndex(u8),
    StartSubBandIndex(u16),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrcCharacteristic {
    Cicp(u8),
    Custom { left_index: u8, right_index: u8 },
}

#[derive(Debug, Clone, PartialEq)]
pub enum CustomDrcCharacteristic {
    Sigmoid {
        gain_db: f32,
        io_ratio: f32,
        exponent: Option<f32>,
        flip_sign: bool,
    },
    Nodes(Vec<CharacteristicNode>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CharacteristicNode {
    pub level_db: f32,
    pub gain_db: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShapeFilter {
    pub lf_cut: Option<u8>,
    pub lf_boost: Option<u8>,
    pub hf_cut: Option<u8>,
    pub hf_boost: Option<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GainBand {
    pub sequence_index: u8,
    pub cicp_characteristic_index: Option<u8>,
    pub characteristic: Option<DrcCharacteristic>,
    pub border: Option<BandBorder>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GainSet {
    pub coding_profile: GainCodingProfile,
    pub interpolation_type: GainInterpolationType,
    pub full_frame: bool,
    pub time_alignment: bool,
    pub time_delta_min: Option<u16>,
    pub drc_band_type: bool,
    pub bands: Vec<GainBand>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DrcCoefficients {
    pub drc_location: u8,
    pub drc_frame_size: Option<u16>,
    pub gain_sequence_count: usize,
    pub gain_sets: Vec<GainSet>,
    pub custom_characteristics_left: Vec<CustomDrcCharacteristic>,
    pub custom_characteristics_right: Vec<CustomDrcCharacteristic>,
    pub shape_filters: Vec<ShapeFilter>,
}

impl DrcCoefficients {
    pub fn parse_v0(input: &[u8]) -> Result<Self, DrcError> {
        Self::parse_v0_from_reader(&mut BitReader::new(input))
    }

    fn parse_v0_from_reader(reader: &mut BitReader<'_>) -> Result<Self, DrcError> {
        let drc_location = reader.read_u8(4)?;
        let drc_frame_size = if reader.read_bool()? {
            Some(reader.read_u16(15)? + 1)
        } else {
            None
        };
        let gain_set_count = reader.read_u8(6)?;
        let mut next_sequence_index = 0u8;
        let mut gain_sequence_count = 0usize;
        let mut gain_sets = Vec::with_capacity(gain_set_count.min(12) as usize);
        for index in 0..gain_set_count {
            let gain_set = read_gain_set_v0(reader, &mut next_sequence_index)?;
            gain_sequence_count += gain_set.bands.len();
            if index < 12 {
                gain_sets.push(gain_set);
            }
        }
        Ok(Self {
            drc_location,
            drc_frame_size,
            gain_sequence_count,
            gain_sets,
            custom_characteristics_left: Vec::new(),
            custom_characteristics_right: Vec::new(),
            shape_filters: Vec::new(),
        })
    }

    pub fn parse_v1(input: &[u8]) -> Result<Self, DrcError> {
        Self::parse_v1_from_reader(&mut BitReader::new(input))
    }

    fn parse_v1_from_reader(reader: &mut BitReader<'_>) -> Result<Self, DrcError> {
        let drc_location = reader.read_u8(4)?;
        let drc_frame_size = if reader.read_bool()? {
            Some(reader.read_u16(15)? + 1)
        } else {
            None
        };
        let custom_characteristics_left = read_custom_characteristics(reader, true)?;
        let custom_characteristics_right = read_custom_characteristics(reader, false)?;
        let shape_filters = if reader.read_bool()? {
            let count = reader.read_u8(4)?;
            (0..count)
                .map(|_| read_shape_filter(reader))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            Vec::new()
        };
        let gain_sequence_count = reader.read_u8(6)? as usize;
        let gain_set_count = reader.read_u8(6)?;
        let mut next_sequence_index = -1i16;
        let mut gain_sets = Vec::with_capacity(gain_set_count.min(12) as usize);
        for index in 0..gain_set_count {
            let gain_set = read_gain_set_v1(reader, &mut next_sequence_index)?;
            if index < 12 {
                gain_sets.push(gain_set);
            }
        }
        Ok(Self {
            drc_location,
            drc_frame_size,
            gain_sequence_count,
            gain_sets,
            custom_characteristics_left,
            custom_characteristics_right,
            shape_filters,
        })
    }

    pub fn parse_gain_payload(
        &self,
        input: &[u8],
        frame_size: usize,
        default_time_delta_min: u16,
    ) -> Result<UniDrcGain, DrcError> {
        let mut reader = BitReader::new(input);
        let mut sequences = Vec::with_capacity(self.gain_sequence_count.min(12));
        for sequence_index in 0..self.gain_sequence_count.min(12) {
            let gain_set = self
                .gain_sets
                .iter()
                .find(|set| {
                    set.bands
                        .iter()
                        .any(|band| band.sequence_index as usize == sequence_index)
                })
                .ok_or(DrcError::MissingGainSetForSequence(sequence_index))?;
            if gain_set.coding_profile == GainCodingProfile::Constant {
                sequences.push(GainSequence {
                    interpolation_type: gain_set.interpolation_type,
                    nodes: vec![GainNode {
                        time: frame_size as i32 - 1,
                        gain_db: 0.0,
                        slope: 0.0,
                    }],
                });
                continue;
            }
            let simple_mode = !reader.read_bool()?;
            if !simple_mode {
                sequences.push(decode_complex_gain_sequence(
                    &mut reader,
                    gain_set,
                    frame_size,
                    gain_set.time_delta_min.unwrap_or(default_time_delta_min),
                )?);
                continue;
            }
            let gain_db = decode_initial_gain(&mut reader, gain_set.coding_profile)?;
            let time_delta_min = gain_set.time_delta_min.unwrap_or(default_time_delta_min);
            let time_offset = if gain_set.time_alignment {
                -(time_delta_min as i32) + (time_delta_min as i32 - 1) / 2
            } else {
                -1
            };
            sequences.push(GainSequence {
                interpolation_type: gain_set.interpolation_type,
                nodes: vec![GainNode {
                    time: frame_size as i32 + time_offset,
                    gain_db,
                    slope: 0.0,
                }],
            });
        }
        let extension_present = if self.gain_sequence_count <= 12 {
            reader.read_bool()?
        } else {
            false
        };
        let extensions = if extension_present {
            read_drc_extensions(&mut reader, 3)?
        } else {
            Vec::new()
        };
        Ok(UniDrcGain {
            sequences,
            extension_present,
            extensions,
            bits_read: reader.bits_read(),
        })
    }
}

fn decode_initial_gain(
    reader: &mut BitReader<'_>,
    profile: GainCodingProfile,
) -> Result<f32, DrcError> {
    Ok(match profile {
        GainCodingProfile::Regular => {
            let negative = reader.read_bool()?;
            let magnitude = reader.read_u8(8)? as f32 * 0.125;
            if negative {
                -magnitude
            } else {
                magnitude
            }
        }
        GainCodingProfile::Fading => {
            if reader.read_bool()? {
                -(reader.read_u16(10)? as f32 + 1.0) * 0.125
            } else {
                0.0
            }
        }
        GainCodingProfile::Clipping => {
            if reader.read_bool()? {
                -(reader.read_u8(8)? as f32 + 1.0) * 0.125
            } else {
                0.0
            }
        }
        GainCodingProfile::Constant => 0.0,
    })
}

fn decode_complex_gain_sequence(
    reader: &mut BitReader<'_>,
    gain_set: &GainSet,
    frame_size: usize,
    time_delta_min: u16,
) -> Result<GainSequence, DrcError> {
    let mut node_count = 1usize;
    while node_count < 128 && !reader.read_bool()? {
        node_count += 1;
    }
    let mut slopes = vec![0.0f32; node_count];
    if gain_set.interpolation_type == GainInterpolationType::Spline {
        for slope in &mut slopes {
            let index = decode_drc_huffman(reader, &SLOPE_STEEPNESS_HUFFMAN)?;
            *slope = SLOPE_STEEPNESS
                .get(index as usize)
                .copied()
                .ok_or(DrcError::InvalidHuffmanValue(index))?;
        }
    }
    let z = drc_time_delta_z(frame_size / time_delta_min as usize);
    let time_offset = if gain_set.time_alignment {
        -(time_delta_min as i32) + (time_delta_min as i32 - 1) / 2
    } else {
        -1
    };
    let full_frame = gain_set.full_frame;
    let frame_end = full_frame || reader.read_bool()?;
    // The reference decoder only retains 16 nodes.  Its frame-end handling is
    // slightly subtler than first accumulating every time and partitioning the
    // result: the first node beyond the frame is replaced by the frame-end
    // node, while the overflowing time is carried in the following slot.
    // Preserve that observable node-reservoir layout exactly.
    let retained_count = node_count.min(16);
    let mut times = vec![0i32; retained_count];
    let mut current = time_offset;
    if frame_end {
        let frame_end_time = frame_size as i32 + time_offset;
        let mut reservoir_started = false;
        for index in 0..node_count.saturating_sub(1) {
            let delta = decode_time_delta(reader, z)? as i32 * time_delta_min as i32;
            if index >= 15 {
                continue;
            }
            let next = current + delta;
            if next > frame_end_time {
                if !reservoir_started {
                    times[index] = frame_end_time;
                    reservoir_started = true;
                }
                times[index + 1] = next;
            } else {
                times[index] = next;
            }
            current = next;
        }
        if !reservoir_started {
            times[node_count.saturating_sub(1).min(15)] = frame_end_time;
        }
    } else {
        for index in 0..node_count {
            let delta = decode_time_delta(reader, z)? as i32 * time_delta_min as i32;
            if index >= 16 {
                continue;
            }
            times[index] = current + delta;
            current = times[index];
        }
    }

    let mut gains = Vec::with_capacity(node_count);
    gains.push(decode_initial_gain(reader, gain_set.coding_profile)?);
    let table = if gain_set.coding_profile == GainCodingProfile::Clipping {
        &DELTA_GAIN_PROFILE_2_HUFFMAN[..]
    } else {
        &DELTA_GAIN_PROFILE_0_1_HUFFMAN[..]
    };
    for index in 1..node_count {
        let delta = decode_drc_huffman(reader, table)? as f32 * 0.125;
        gains.push(gains[index - 1] + delta);
    }
    let mut reordered_times = Vec::with_capacity(retained_count);
    reordered_times.extend(
        times
            .iter()
            .filter(|&&time| time >= frame_size as i32)
            .map(|&time| time - 2 * frame_size as i32),
    );
    reordered_times.extend(
        times
            .iter()
            .filter(|&&time| time < frame_size as i32)
            .copied(),
    );
    Ok(GainSequence {
        interpolation_type: gain_set.interpolation_type,
        nodes: (0..retained_count)
            .map(|index| GainNode {
                time: reordered_times[index],
                gain_db: gains[index],
                slope: slopes[index],
            })
            .collect(),
    })
}

fn drc_time_delta_z(max_nodes: usize) -> usize {
    let mut z = 1usize;
    while (1usize << z) < 2 * max_nodes.max(1) {
        z += 1;
    }
    z
}

fn decode_time_delta(reader: &mut BitReader<'_>, z: usize) -> Result<usize, DrcError> {
    Ok(match reader.read_u8(2)? {
        0 => 1,
        1 => reader.read_u8(2)? as usize + 2,
        2 => reader.read_u8(3)? as usize + 6,
        _ => reader.read(z)? as usize + 14,
    })
}

fn decode_drc_huffman(reader: &mut BitReader<'_>, table: &[[i8; 2]]) -> Result<i8, DrcError> {
    let mut index = 0i8;
    while index >= 0 {
        let row = table
            .get(index as usize)
            .ok_or(DrcError::InvalidHuffmanIndex(index as usize))?;
        index = row[usize::from(reader.read_bool()?)];
    }
    Ok(index + 64)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GainNode {
    pub time: i32,
    pub gain_db: f32,
    pub slope: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GainSequence {
    pub interpolation_type: GainInterpolationType,
    pub nodes: Vec<GainNode>,
}

impl GainSequence {
    pub fn gain_db_at(&self, time: i32) -> f32 {
        let Some(first) = self.nodes.first() else {
            return 0.0;
        };
        if time <= first.time {
            return first.gain_db;
        }
        for pair in self.nodes.windows(2) {
            let left = pair[0];
            let right = pair[1];
            if time <= right.time {
                let duration = (right.time - left.time).max(1) as f32;
                let x = (time - left.time) as f32 / duration;
                return match self.interpolation_type {
                    GainInterpolationType::Linear => {
                        left.gain_db + (right.gain_db - left.gain_db) * x
                    }
                    GainInterpolationType::Spline => {
                        let x2 = x * x;
                        let x3 = x2 * x;
                        (2.0 * x3 - 3.0 * x2 + 1.0) * left.gain_db
                            + (x3 - 2.0 * x2 + x) * duration * left.slope
                            + (-2.0 * x3 + 3.0 * x2) * right.gain_db
                            + (x3 - x2) * duration * right.slope
                    }
                };
            }
        }
        self.nodes.last().map_or(0.0, |node| node.gain_db)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct UniDrcGain {
    pub sequences: Vec<GainSequence>,
    pub extension_present: bool,
    pub extensions: Vec<DrcExtension>,
    pub bits_read: usize,
}

impl UniDrcGain {
    pub fn apply_sequence_f32(
        &self,
        sequence_index: usize,
        samples: &mut [f32],
    ) -> Result<(), DrcError> {
        let sequence = self
            .sequences
            .get(sequence_index)
            .ok_or(DrcError::MissingGainSequence(sequence_index))?;
        for (time, sample) in samples.iter_mut().enumerate() {
            let gain = 10.0f32.powf(sequence.gain_db_at(time as i32) / 20.0);
            *sample *= gain;
        }
        Ok(())
    }

    pub fn apply_sequence_i16(
        &self,
        sequence_index: usize,
        samples: &mut [i16],
    ) -> Result<(), DrcError> {
        let sequence = self
            .sequences
            .get(sequence_index)
            .ok_or(DrcError::MissingGainSequence(sequence_index))?;
        for (time, sample) in samples.iter_mut().enumerate() {
            let gain = 10.0f32.powf(sequence.gain_db_at(time as i32) / 20.0);
            *sample = (*sample as f32 * gain)
                .round()
                .clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        }
        Ok(())
    }
}

fn read_gain_set_v0(
    reader: &mut BitReader<'_>,
    next_sequence_index: &mut u8,
) -> Result<GainSet, DrcError> {
    let coding_profile = match reader.read_u8(2)? {
        0 => GainCodingProfile::Regular,
        1 => GainCodingProfile::Fading,
        2 => GainCodingProfile::Clipping,
        _ => GainCodingProfile::Constant,
    };
    let interpolation_type = if reader.read_bool()? {
        GainInterpolationType::Linear
    } else {
        GainInterpolationType::Spline
    };
    let full_frame = reader.read_bool()?;
    let time_alignment = reader.read_bool()?;
    let time_delta_min = if reader.read_bool()? {
        Some(reader.read_u16(11)? + 1)
    } else {
        None
    };
    let band_count = if coding_profile == GainCodingProfile::Constant {
        1
    } else {
        let count = reader.read_u8(4)?;
        if count == 0 || count > 4 {
            return Err(DrcError::InvalidBandCount(count));
        }
        count
    };
    let drc_band_type =
        coding_profile != GainCodingProfile::Constant && band_count > 1 && reader.read_bool()?;
    let mut bands = Vec::with_capacity(band_count as usize);
    for _ in 0..band_count {
        let sequence_index = *next_sequence_index;
        *next_sequence_index = next_sequence_index
            .checked_add(1)
            .ok_or(DrcError::TooManyGainSequences)?;
        let cicp = reader.read_u8(7)?;
        bands.push(GainBand {
            sequence_index,
            cicp_characteristic_index: (cicp != 0).then_some(cicp),
            characteristic: (cicp != 0).then_some(DrcCharacteristic::Cicp(cicp)),
            border: None,
        });
    }
    for band in bands.iter_mut().skip(1) {
        band.border = Some(if drc_band_type {
            BandBorder::CrossoverFrequencyIndex(reader.read_u8(4)?)
        } else {
            BandBorder::StartSubBandIndex(reader.read_u16(10)?)
        });
    }
    Ok(GainSet {
        coding_profile,
        interpolation_type,
        full_frame,
        time_alignment,
        time_delta_min,
        drc_band_type,
        bands,
    })
}

fn read_custom_characteristics(
    reader: &mut BitReader<'_>,
    left_side: bool,
) -> Result<Vec<CustomDrcCharacteristic>, DrcError> {
    if !reader.read_bool()? {
        return Ok(Vec::new());
    }
    let count = reader.read_u8(4)?;
    (0..count)
        .map(|_| read_custom_characteristic(reader, left_side))
        .collect()
}

fn read_custom_characteristic(
    reader: &mut BitReader<'_>,
    left_side: bool,
) -> Result<CustomDrcCharacteristic, DrcError> {
    if !reader.read_bool()? {
        let magnitude = reader.read_u8(6)? as f32;
        let gain_db = if left_side { magnitude } else { -magnitude };
        let io_ratio = 0.05 + 0.15 * reader.read_u8(4)? as f32;
        let raw_exponent = reader.read_u8(4)?;
        let exponent = (raw_exponent < 15).then_some((1 + 2 * raw_exponent) as f32);
        let flip_sign = reader.read_bool()?;
        Ok(CustomDrcCharacteristic::Sigmoid {
            gain_db,
            io_ratio,
            exponent,
            flip_sign,
        })
    } else {
        let count = reader.read_u8(2)? as usize + 1;
        let mut nodes = Vec::with_capacity(count + 1);
        let mut level_db = -31.0;
        nodes.push(CharacteristicNode {
            level_db,
            gain_db: 0.0,
        });
        for _ in 0..count {
            let delta = (reader.read_u8(5)? as f32 + 1.0) * 0.25;
            level_db += if left_side { -delta } else { delta };
            let gain_db = reader.read_u8(8)? as f32 * 0.5 - 64.0;
            nodes.push(CharacteristicNode { level_db, gain_db });
        }
        Ok(CustomDrcCharacteristic::Nodes(nodes))
    }
}

fn read_shape_filter(reader: &mut BitReader<'_>) -> Result<ShapeFilter, DrcError> {
    fn parameter(reader: &mut BitReader<'_>) -> Result<Option<u8>, DrcError> {
        Ok(reader.read_bool()?.then(|| reader.read_u8(5)).transpose()?)
    }
    Ok(ShapeFilter {
        lf_cut: parameter(reader)?,
        lf_boost: parameter(reader)?,
        hf_cut: parameter(reader)?,
        hf_boost: parameter(reader)?,
    })
}

fn read_gain_set_v1(
    reader: &mut BitReader<'_>,
    next_sequence_index: &mut i16,
) -> Result<GainSet, DrcError> {
    let coding_profile = match reader.read_u8(2)? {
        0 => GainCodingProfile::Regular,
        1 => GainCodingProfile::Fading,
        2 => GainCodingProfile::Clipping,
        _ => GainCodingProfile::Constant,
    };
    let interpolation_type = if reader.read_bool()? {
        GainInterpolationType::Linear
    } else {
        GainInterpolationType::Spline
    };
    let full_frame = reader.read_bool()?;
    let time_alignment = reader.read_bool()?;
    let time_delta_min = if reader.read_bool()? {
        Some(reader.read_u16(11)? + 1)
    } else {
        None
    };
    let band_count = if coding_profile == GainCodingProfile::Constant {
        1
    } else {
        let count = reader.read_u8(4)?;
        if count == 0 || count > 4 {
            return Err(DrcError::InvalidBandCount(count));
        }
        count
    };
    let drc_band_type =
        coding_profile != GainCodingProfile::Constant && band_count > 1 && reader.read_bool()?;
    let mut bands = Vec::with_capacity(band_count as usize);
    for _ in 0..band_count {
        if coding_profile == GainCodingProfile::Constant || !reader.read_bool()? {
            *next_sequence_index += 1;
        } else {
            *next_sequence_index = reader.read_u8(6)? as i16;
        }
        let sequence_index =
            u8::try_from(*next_sequence_index).map_err(|_| DrcError::TooManyGainSequences)?;
        let characteristic = if coding_profile == GainCodingProfile::Constant {
            None
        } else if reader.read_bool()? {
            if reader.read_bool()? {
                Some(DrcCharacteristic::Cicp(reader.read_u8(7)?))
            } else {
                Some(DrcCharacteristic::Custom {
                    left_index: reader.read_u8(4)?,
                    right_index: reader.read_u8(4)?,
                })
            }
        } else {
            None
        };
        bands.push(GainBand {
            sequence_index,
            cicp_characteristic_index: match characteristic {
                Some(DrcCharacteristic::Cicp(index)) => Some(index),
                _ => None,
            },
            characteristic,
            border: None,
        });
    }
    for band in bands.iter_mut().skip(1) {
        band.border = Some(if drc_band_type {
            BandBorder::CrossoverFrequencyIndex(reader.read_u8(4)?)
        } else {
            BandBorder::StartSubBandIndex(reader.read_u16(10)?)
        });
    }
    Ok(GainSet {
        coding_profile,
        interpolation_type,
        full_frame,
        time_alignment,
        time_delta_min,
        drc_band_type,
        bands,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoudnessMethod {
    UnknownOther,
    ProgramLoudness,
    AnchorLoudness,
    MaximumLoudnessRange,
    MomentaryLoudnessMax,
    ShortTermLoudnessMax,
    LoudnessRange,
    MixingLevel,
    RoomType,
    ShortTermLoudness,
}

impl LoudnessMethod {
    fn from_bits(value: u8) -> Result<Self, DrcError> {
        Ok(match value {
            0 => Self::UnknownOther,
            1 => Self::ProgramLoudness,
            2 => Self::AnchorLoudness,
            3 => Self::MaximumLoudnessRange,
            4 => Self::MomentaryLoudnessMax,
            5 => Self::ShortTermLoudnessMax,
            6 => Self::LoudnessRange,
            7 => Self::MixingLevel,
            8 => Self::RoomType,
            9 => Self::ShortTermLoudness,
            _ => return Err(DrcError::InvalidLoudnessMethod(value)),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoudnessMeasurement {
    pub method: LoudnessMethod,
    pub value: f32,
    pub measurement_system: u8,
    pub reliability: u8,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoudnessInfo {
    pub drc_set_id: u8,
    pub downmix_id: u8,
    pub sample_peak_level: Option<f32>,
    pub true_peak_level: Option<f32>,
    pub true_peak_measurement_system: Option<u8>,
    pub true_peak_reliability: Option<u8>,
    pub measurements: Vec<LoudnessMeasurement>,
}

impl LoudnessInfo {
    pub fn program_loudness(&self) -> Option<f32> {
        self.measurements
            .iter()
            .find(|measurement| measurement.method == LoudnessMethod::ProgramLoudness)
            .map(|measurement| measurement.value)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoudnessInfoSet {
    pub album: Vec<LoudnessInfo>,
    pub track: Vec<LoudnessInfo>,
    pub extension_present: bool,
    pub bits_read: usize,
}

impl LoudnessInfoSet {
    pub fn parse_v0(input: &[u8]) -> Result<Self, DrcError> {
        let mut reader = BitReader::new(input);
        let album_count = reader.read_u8(6)?;
        let track_count = reader.read_u8(6)?;
        let mut album = Vec::with_capacity(album_count.min(12) as usize);
        for index in 0..album_count {
            let info = read_loudness_info_v0(&mut reader)?;
            if index < 12 {
                album.push(info);
            }
        }
        let mut track = Vec::with_capacity(track_count.min(12) as usize);
        for index in 0..track_count {
            let info = read_loudness_info_v0(&mut reader)?;
            if index < 12 {
                track.push(info);
            }
        }
        let extension_present = reader.read_bool()?;
        Ok(Self {
            album,
            track,
            extension_present,
            bits_read: reader.bits_read(),
        })
    }

    pub fn select_program_loudness(
        &self,
        drc_set_id: u8,
        downmix_id: u8,
        album_mode: bool,
    ) -> Option<f32> {
        let primary = if album_mode { &self.album } else { &self.track };
        primary
            .iter()
            .find(|info| info.drc_set_id == drc_set_id && info.downmix_id == downmix_id)
            .and_then(LoudnessInfo::program_loudness)
            .or_else(|| {
                primary
                    .iter()
                    .find(|info| info.drc_set_id == drc_set_id && info.downmix_id == 0x7f)
                    .and_then(LoudnessInfo::program_loudness)
            })
    }
}

fn read_loudness_info_v0(reader: &mut BitReader<'_>) -> Result<LoudnessInfo, DrcError> {
    let drc_set_id = reader.read_u8(6)?;
    let downmix_id = reader.read_u8(7)?;
    let sample_peak_level = read_peak_level(reader)?;
    let true_peak_level = read_peak_level(reader)?;
    let (true_peak_measurement_system, true_peak_reliability) = if true_peak_level.is_some() {
        (Some(reader.read_u8(4)?), Some(reader.read_u8(2)?))
    } else {
        (None, None)
    };
    let measurement_count = reader.read_u8(4)?;
    let mut measurements = Vec::with_capacity(measurement_count.min(8) as usize);
    for index in 0..measurement_count {
        let method = LoudnessMethod::from_bits(reader.read_u8(4)?)?;
        let raw = reader.read_u8(if method == LoudnessMethod::MixingLevel {
            5
        } else if method == LoudnessMethod::RoomType {
            2
        } else {
            8
        })?;
        let value = decode_method_value(method, raw);
        let measurement_system = reader.read_u8(4)?;
        let reliability = reader.read_u8(2)?;
        if index < 8 {
            measurements.push(LoudnessMeasurement {
                method,
                value,
                measurement_system,
                reliability,
            });
        }
    }
    Ok(LoudnessInfo {
        drc_set_id,
        downmix_id,
        sample_peak_level,
        true_peak_level,
        true_peak_measurement_system,
        true_peak_reliability,
        measurements,
    })
}

fn read_peak_level(reader: &mut BitReader<'_>) -> Result<Option<f32>, DrcError> {
    if !reader.read_bool()? {
        return Ok(None);
    }
    let raw = reader.read_u16(12)?;
    Ok((raw != 0).then_some(20.0 - raw as f32 * 0.03125))
}

fn decode_method_value(method: LoudnessMethod, raw: u8) -> f32 {
    match method {
        LoudnessMethod::UnknownOther
        | LoudnessMethod::ProgramLoudness
        | LoudnessMethod::AnchorLoudness
        | LoudnessMethod::MaximumLoudnessRange
        | LoudnessMethod::MomentaryLoudnessMax
        | LoudnessMethod::ShortTermLoudnessMax => -57.75 + raw as f32 * 0.25,
        LoudnessMethod::LoudnessRange if raw == 0 => 0.0,
        LoudnessMethod::LoudnessRange if raw <= 128 => raw as f32 * 0.25,
        LoudnessMethod::LoudnessRange if raw <= 204 => raw as f32 * 0.5 - 32.0,
        LoudnessMethod::LoudnessRange => raw as f32 - 134.0,
        LoudnessMethod::MixingLevel => raw as f32 + 80.0,
        LoudnessMethod::RoomType => raw as f32,
        LoudnessMethod::ShortTermLoudness => -116.0 + raw as f32 * 0.5,
    }
}

pub fn loudness_normalization_gain_db(target_loudness: f32, measured_loudness: f32) -> f32 {
    target_loudness - measured_loudness
}

pub fn apply_gain_f32(samples: &mut [f32], gain_db: f32) {
    let gain = 10.0f32.powf(gain_db / 20.0);
    for sample in samples {
        *sample *= gain;
    }
}

pub fn apply_gain_i16(samples: &mut [i16], gain_db: f32) {
    let gain = 10.0f32.powf(gain_db / 20.0);
    for sample in samples {
        *sample = (*sample as f32 * gain)
            .round()
            .clamp(i16::MIN as f32, i16::MAX as f32) as i16;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DrcError {
    Bit(BitError),
    DownmixCoefficientsAbsent,
    DownmixLayoutMismatch,
    ExtensionSizeMismatch {
        declared: usize,
        consumed: usize,
    },
    InstructionLayoutMismatch,
    InvalidBandCount(u8),
    InvalidDownmixDimensions {
        source_channels: u8,
        target_channels: u8,
    },
    InvalidHuffmanIndex(usize),
    InvalidHuffmanValue(i8),
    InstructionChannelOverflow,
    InvalidLoudnessMethod(u8),
    MissingGainSequence(usize),
    MissingDrcCoefficients(u8),
    MissingDownmixInstruction(u8),
    MissingGainSet(usize),
    MissingGainSetForSequence(usize),
    TooManyChannels(u8),
    TooManyExtensions,
    TooManyGainSequences,
}

impl From<BitError> for DrcError {
    fn from(value: BitError) -> Self {
        Self::Bit(value)
    }
}

impl fmt::Display for DrcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bit(error) => error.fmt(f),
            Self::DownmixCoefficientsAbsent => write!(f, "DRC downmix coefficients are absent"),
            Self::DownmixLayoutMismatch => write!(f, "DRC downmix layout mismatch"),
            Self::ExtensionSizeMismatch { declared, consumed } => write!(
                f,
                "DRC extension declares {declared} bits but parser consumed {consumed}"
            ),
            Self::InstructionLayoutMismatch => write!(f, "DRC instruction layout mismatch"),
            Self::InvalidBandCount(value) => write!(f, "invalid DRC gain band count {value}"),
            Self::InvalidDownmixDimensions {
                source_channels,
                target_channels,
            } => write!(
                f,
                "invalid DRC downmix dimensions {source_channels} -> {target_channels}"
            ),
            Self::InvalidHuffmanIndex(value) => write!(f, "invalid DRC Huffman index {value}"),
            Self::InvalidHuffmanValue(value) => write!(f, "invalid DRC Huffman value {value}"),
            Self::InstructionChannelOverflow => {
                write!(f, "DRC instruction repeats exceed channel count")
            }
            Self::InvalidLoudnessMethod(value) => write!(f, "invalid DRC loudness method {value}"),
            Self::MissingGainSequence(value) => write!(f, "missing DRC gain sequence {value}"),
            Self::MissingDrcCoefficients(value) => {
                write!(f, "missing DRC coefficients at location {value}")
            }
            Self::MissingDownmixInstruction(value) => {
                write!(f, "missing DRC downmix instruction {value}")
            }
            Self::MissingGainSet(value) => write!(f, "missing DRC gain set {value}"),
            Self::MissingGainSetForSequence(value) => {
                write!(f, "missing DRC gain set for sequence {value}")
            }
            Self::TooManyChannels(value) => {
                write!(f, "Unified DRC channel count {value} exceeds 8")
            }
            Self::TooManyExtensions => write!(f, "too many Unified DRC extensions"),
            Self::TooManyGainSequences => write!(f, "too many DRC gain sequences"),
        }
    }
}

impl std::error::Error for DrcError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitWriter;

    #[test]
    fn parses_legacy_mpeg4_drc_fill_and_computes_one_band_gain() {
        let mut fill = BitWriter::new();
        fill.write(4, 4); // four payload bytes
        fill.write(0x0b, 4); // EXT_DYNAMIC_RANGE
        fill.write_bool(false); // no PCE tag
        fill.write_bool(true); // excluded channels present
        fill.write(0b010_0000, 7); // exclude channel 1
        fill.write_bool(false); // no further exclusion mask
        fill.write_bool(false); // one DRC band
        fill.write_bool(true); // program reference level present
        fill.write(80, 7);
        fill.write_bool(false); // reserved
        fill.write(0x98, 8); // attenuation by 24 * 0.25 dB

        let payload = parse_mpeg4_drc_fill_element(&mut BitReader::new(&fill.finish()))
            .unwrap()
            .unwrap();
        assert_eq!(payload.program_reference_level, Some(80));
        assert_eq!(payload.dynamic_range, vec![0x98]);
        assert!(!payload.channel_is_excluded(0));
        assert!(payload.channel_is_excluded(1));
        // -6 dB DRC and -4 dB normalization from -20 to -24 dBFS.
        let gain = payload.one_band_gain(0, 1.0, 1.0, Some(96)).unwrap();
        assert!((gain - 2.0f32.powf(-40.0 / 24.0)).abs() < 1e-6);
        assert_eq!(payload.one_band_gain(1, 1.0, 1.0, Some(96)), None);
    }

    #[test]
    fn parses_dvb_ancillary_heavy_compression_and_validates_header() {
        let payload = parse_dvb_ancillary_drc(&[0xbc, 0xc4, 0x04, 0x01, 0x90]).unwrap();
        assert_eq!(payload.presentation_mode, 1);
        assert_eq!(payload.compression_value, 0x90);
        assert!((payload.gain() - 0.5).abs() < 0.001);
        assert_eq!(
            DvbAncillaryDrcPayload {
                presentation_mode: 2,
                compression_value: 0x7f,
            }
            .gain(),
            1.0
        );

        for invalid in [
            &[0u8][..],
            &[0xbc, 0x84, 0x04, 0x01, 0x90], // not MPEG-4 audio
            &[0xbc, 0xc5, 0x04, 0x01, 0x90], // reserved bit
            &[0xbc, 0xc4, 0x24, 0x01, 0x90], // reserved status
            &[0xbc, 0xc4, 0x00, 0x01, 0x90], // no compression field
            &[0xbc, 0xc4, 0x04, 0x03, 0x90], // reserved audio mode
            &[0xbc, 0xc4, 0x04, 0x00, 0x90], // compression disabled
        ] {
            assert_eq!(parse_dvb_ancillary_drc(invalid), None);
        }
        let with_downmix = parse_dvb_ancillary_drc(&[0xbc, 0xc8, 0x14, 0x55, 0x01, 0x80]).unwrap();
        assert_eq!(with_downmix.presentation_mode, 2);
        assert_eq!(with_downmix.compression_value, 0x80);
    }

    #[test]
    fn parses_and_merges_dvb_advanced_downmix_metadata() {
        let base = parse_dvb_ancillary_downmix(&[0xbc, 0xc2, 0x10, 0xdd]).unwrap();
        assert!(base.pseudo_surround);
        assert_eq!(base.center_mix_level_index, Some(5));
        assert_eq!(base.surround_mix_level_index, Some(5));

        let extended =
            parse_dvb_ancillary_downmix(&[0xbc, 0xc0, 0x08, 0x70, 0x70, 0x55, 0x23, 0xa0]).unwrap();
        assert_eq!(extended.downmix_a_index, Some(3));
        assert_eq!(extended.downmix_b_index, Some(4));
        assert_eq!(extended.five_channel_downmix_gain_index, Some(0x2a));
        assert_eq!(extended.stereo_downmix_gain_index, Some(0x11));
        assert_eq!(extended.lfe_mix_level_index, Some(10));

        let mut merged = base;
        merged.merge(extended);
        assert!(!merged.pseudo_surround);
        assert_eq!(merged.center_mix_level_index, Some(5));
        assert_eq!(merged.downmix_a_index, Some(3));
        assert_eq!(parse_dvb_ancillary_downmix(&[0xbc, 0x80, 0]), None);
        assert_eq!(parse_dvb_ancillary_downmix(&[0xbc, 0xc0, 0x08]), None);
    }

    #[test]
    fn converts_formats_every_error_and_reads_all_ducking_scales() {
        let bit = BitError::UnexpectedEof {
            needed_bits: 1,
            remaining_bits: 0,
        };
        assert_eq!(DrcError::from(bit.clone()), DrcError::Bit(bit));
        let errors = [
            DrcError::Bit(BitError::UnexpectedEof {
                needed_bits: 1,
                remaining_bits: 0,
            }),
            DrcError::DownmixCoefficientsAbsent,
            DrcError::DownmixLayoutMismatch,
            DrcError::ExtensionSizeMismatch {
                declared: 1,
                consumed: 2,
            },
            DrcError::InstructionLayoutMismatch,
            DrcError::InvalidBandCount(0),
            DrcError::InvalidDownmixDimensions {
                source_channels: 1,
                target_channels: 2,
            },
            DrcError::InvalidHuffmanIndex(1),
            DrcError::InvalidHuffmanValue(-1),
            DrcError::InstructionChannelOverflow,
            DrcError::InvalidLoudnessMethod(15),
            DrcError::MissingGainSequence(1),
            DrcError::MissingDrcCoefficients(1),
            DrcError::MissingDownmixInstruction(1),
            DrcError::MissingGainSet(1),
            DrcError::MissingGainSetForSequence(1),
            DrcError::TooManyChannels(9),
            DrcError::TooManyExtensions,
            DrcError::TooManyGainSequences,
        ];
        for error in errors {
            assert!(!error.to_string().is_empty());
        }

        let parse_scale = |present: bool, raw: u32| {
            let mut writer = BitWriter::new();
            writer.write_bool(present);
            if present {
                writer.write(raw, 4);
            }
            read_ducking_modification(&mut BitReader::new(&writer.finish()))
                .unwrap()
                .scaling
        };
        assert_eq!(parse_scale(false, 0), 1.0);
        assert_eq!(parse_scale(true, 2), 1.375);
        assert_eq!(parse_scale(true, 10), 0.625);

        let mut layout = BitWriter::new();
        layout.write(2, 7);
        layout.write_bool(false);
        assert_eq!(
            read_channel_layout(&mut BitReader::new(&layout.finish())).unwrap(),
            ChannelLayout {
                base_channel_count: 2,
                defined_layout: None,
                speaker_positions: Vec::new(),
            }
        );
        let mut oversized = BitWriter::new();
        oversized.write(9, 7);
        assert_eq!(
            read_channel_layout(&mut BitReader::new(&oversized.finish())),
            Err(DrcError::TooManyChannels(9))
        );
    }

    #[test]
    fn parses_foundation_config_and_explicit_speakers() {
        let mut writer = BitWriter::new();
        writer.write_bool(true);
        writer.write(47_000, 18); // +1000 = 48000
        writer.write(0, 7);
        writer.write_bool(false);
        writer.write(0, 3);
        writer.write(0, 6);
        writer.write(2, 7);
        writer.write_bool(true);
        writer.write(0, 8);
        writer.write(1, 7);
        writer.write(2, 7);
        writer.write_bool(false);
        let config = UniDrcConfig::parse_foundation(&writer.finish()).unwrap();
        assert_eq!(config.sample_rate, Some(48_000));
        assert_eq!(config.channel_layout.base_channel_count, 2);
        assert_eq!(config.channel_layout.speaker_positions, [1, 2]);
        assert!(config.downmix_instructions.is_empty());
    }

    #[test]
    fn foundation_propagates_nested_downmix_instruction_and_v1_extension_errors() {
        let foundation_prefix = |downmix_count: u8, instruction_count: u8| {
            let mut writer = BitWriter::new();
            writer.write_bool(false); // sample rate absent
            writer.write(downmix_count as u32, 7);
            writer.write_bool(false); // basic DRC absent
            writer.write(0, 3); // coefficient count
            writer.write(instruction_count as u32, 6);
            writer.write(1, 7); // one base channel
            writer.write_bool(false); // layout signaling absent
            writer
        };

        let downmix = foundation_prefix(1, 0).finish();
        assert!(matches!(
            UniDrcConfig::parse_foundation(&downmix),
            Err(DrcError::Bit(BitError::UnexpectedEof { .. }))
        ));

        let instruction = foundation_prefix(0, 1).finish();
        assert!(matches!(
            UniDrcConfig::parse_foundation(&instruction),
            Err(DrcError::Bit(BitError::UnexpectedEof { .. }))
        ));

        let mut extension = foundation_prefix(0, 0);
        extension.write_bool(true); // extension present
        extension.write(2, 4); // v1 configuration extension
        extension.write(0, 4); // four-bit payload-size field
        extension.write(0, 4); // one payload bit
        extension.write_bool(false);
        extension.write(0, 4); // extension terminator
        assert!(matches!(
            UniDrcConfig::parse_foundation(&extension.finish()),
            Err(DrcError::ExtensionSizeMismatch { .. })
        ));
    }

    #[test]
    fn v1_merge_and_eq_skip_helpers_propagate_truncated_nested_payloads() {
        let layout = ChannelLayout {
            base_channel_count: 1,
            defined_layout: None,
            speaker_positions: Vec::new(),
        };

        let mut downmix_payload = BitWriter::new();
        downmix_payload.write_bool(true);
        downmix_payload.write(1, 7);
        let downmix_bits = downmix_payload.bits_written();
        let downmix_extension = DrcExtension {
            extension_type: 2,
            bit_size: downmix_bits,
            payload: downmix_payload.finish(),
        };
        assert!(matches!(
            merge_v1_config_extension(
                &downmix_extension,
                &layout,
                &mut Vec::new(),
                &mut Vec::new(),
                &mut Vec::new(),
            ),
            Err(DrcError::Bit(BitError::UnexpectedEof { .. }))
        ));

        let mut instruction_payload = BitWriter::new();
        instruction_payload.write_bool(false); // no downmix instructions
        instruction_payload.write_bool(true); // coefficient/instruction section present
        instruction_payload.write(0, 3); // no coefficients
        instruction_payload.write(1, 6); // one truncated instruction
        let instruction_bits = instruction_payload.bits_written();
        let instruction_extension = DrcExtension {
            extension_type: 2,
            bit_size: instruction_bits,
            payload: instruction_payload.finish(),
        };
        assert!(matches!(
            merge_v1_config_extension(
                &instruction_extension,
                &layout,
                &mut Vec::new(),
                &mut Vec::new(),
                &mut Vec::new(),
            ),
            Err(DrcError::Bit(BitError::UnexpectedEof { .. }))
        ));

        let mut eq = BitWriter::new();
        eq.write_bool(false); // no EQ delay maximum
        eq.write(0, 6); // no filter blocks
        eq.write(1, 6); // one time-domain element
        eq.write_bool(false); // pole/zero representation
        eq.write(0, 3); // radius-one zeros
        eq.write(0, 6); // real zeros
        eq.write(1, 6); // one generic zero, whose data is absent
        eq.write(0, 4); // real poles
        eq.write(0, 4); // complex poles
        assert!(matches!(
            skip_eq_coefficients(&mut BitReader::new(&eq.finish())),
            Err(DrcError::Bit(BitError::UnexpectedEof { .. }))
        ));

        let mut cascade = BitWriter::new();
        for _ in 0..4 {
            cascade.write_bool(false); // no cascade gain
            cascade.write(0, 4); // no filter blocks
        }
        cascade.write_bool(true); // truncated phase-alignment matrix
        assert!(matches!(
            skip_td_filter_cascade(&mut BitReader::new(&cascade.finish()), 4),
            Err(DrcError::Bit(BitError::UnexpectedEof { .. }))
        ));
    }

    #[test]
    fn basic_instruction_skip_consumes_all_optional_fields() {
        let mut writer = BitWriter::new();
        writer.write(1, 6);
        writer.write(2, 4);
        writer.write(3, 7);
        writer.write_bool(true);
        writer.write(2, 3);
        writer.write(4, 7);
        writer.write(5, 7);
        writer.write(0, 16);
        writer.write_bool(true);
        writer.write(8, 8);
        writer.write_bool(true);
        writer.write(40, 6);
        writer.write_bool(true);
        writer.write(34, 6);
        let bits = writer.bits_written();
        let bytes = writer.finish();
        let mut reader = BitReader::with_bit_len(&bytes, bits).unwrap();
        skip_drc_instructions_basic(&mut reader).unwrap();
        assert_eq!(reader.bits_read(), bits);

        let mut minimal = BitWriter::new();
        minimal.write(1, 6); // drcSetId
        minimal.write(2, 4); // drcLocation
        minimal.write(3, 7); // downmixId
        minimal.write_bool(false); // no additional downmix IDs
        minimal.write(DRC_EFFECT_DUCK_SELF as u32, 16); // limiter field omitted
        minimal.write_bool(false); // no target loudness
        let bits = minimal.bits_written();
        let bytes = minimal.finish();
        let mut reader = BitReader::with_bit_len(&bytes, bits).unwrap();
        skip_drc_instructions_basic(&mut reader).unwrap();
        assert_eq!(reader.bits_read(), bits);

        let mut upper_only = BitWriter::new();
        upper_only.write(1, 6);
        upper_only.write(2, 4);
        upper_only.write(3, 7);
        upper_only.write_bool(false); // no additional downmix IDs
        upper_only.write(0, 16); // no ducking effects
        upper_only.write_bool(false); // no limiter peak target
        upper_only.write_bool(true); // target loudness present
        upper_only.write(6, 6); // upper value
        upper_only.write_bool(false); // no lower value
        let bits = upper_only.bits_written();
        let bytes = upper_only.finish();
        let mut reader = BitReader::with_bit_len(&bytes, bits).unwrap();
        skip_drc_instructions_basic(&mut reader).unwrap();
        assert_eq!(reader.bits_read(), bits);

        let mut coefficient = BitWriter::new();
        coefficient.write(1, 4);
        coefficient.write(2, 7);
        let bits = coefficient.bits_written();
        let bytes = coefficient.finish();
        let mut reader = BitReader::with_bit_len(&bytes, bits).unwrap();
        skip_drc_coefficients_basic(&mut reader).unwrap();
        assert_eq!(reader.bits_read(), bits);
    }

    #[test]
    fn parses_complete_v0_foundation_downmix_coefficients_and_instruction() {
        let mut writer = BitWriter::new();
        writer.write_bool(false); // sample rate absent
        writer.write(1, 7); // one downmix
        writer.write_bool(false); // no basic DRC description
        writer.write(1, 3); // one coefficient set
        writer.write(1, 6); // one instruction
        writer.write(2, 7); // stereo base layout
        writer.write_bool(false); // no explicit speaker positions

        writer.write(4, 7); // downmixId
        writer.write(1, 7); // mono target
        writer.write(1, 8); // target layout
        writer.write_bool(false); // coefficient matrix absent

        writer.write(1, 4); // coefficient drcLocation
        writer.write_bool(false); // frame size absent
        writer.write(1, 6); // one gain set
        writer.write(3, 2); // constant coding profile
        writer.write_bool(true); // linear interpolation
        writer.write_bool(true); // full frame
        writer.write_bool(false); // no time alignment
        writer.write_bool(false); // no timeDeltaMin
        writer.write(1, 7); // CICP characteristic

        writer.write(3, 6); // instruction drcSetId
        writer.write(1, 4); // matching drcLocation
        writer.write(4, 7); // apply to parsed mono downmix
        writer.write_bool(false); // no additional downmix IDs
        writer.write(0, 16); // non-ducking effect
        writer.write_bool(false); // limiter absent
        writer.write_bool(false); // target loudness absent
        writer.write_bool(false); // dependency absent
        writer.write_bool(false); // independent use allowed
        writer.write(1, 6); // gain set index 0
        writer.write_bool(false); // no channel repeat
        writer.write_bool(false); // default gain scaling
        writer.write_bool(false); // no gain offset
        writer.write_bool(false); // no config extension

        let bits = writer.bits_written();
        let config = UniDrcConfig::parse_foundation(&writer.finish()).unwrap();
        assert_eq!(config.bits_read, bits);
        assert_eq!(config.downmix_instructions.len(), 1);
        assert_eq!(config.coefficients.len(), 1);
        assert_eq!(config.instructions.len(), 1);
        assert_eq!(config.instructions[0].channel_count, 1);
        assert_eq!(config.instructions[0].downmix_ids, [4]);
        assert_eq!(config.coefficients[0].gain_sequence_count, 1);
    }

    #[test]
    fn skips_basic_drc_description_without_losing_following_alignment() {
        let mut writer = BitWriter::new();
        writer.write_bool(false); // sample rate absent
        writer.write(0, 7); // no downmix instructions
        writer.write_bool(true); // basic DRC description present
        writer.write(1, 3); // one basic coefficient
        writer.write(1, 4); // one basic instruction
        writer.write(0, 3); // no UniDRC coefficients
        writer.write(0, 6); // no UniDRC instructions
        writer.write(1, 7); // mono base channel count
        writer.write_bool(false); // no defined/speaker layout

        writer.write(2, 4); // basic coefficient drcLocation
        writer.write(37, 7); // basic drcCharacteristic

        writer.write(3, 6); // basic instruction drcSetId
        writer.write(2, 4); // drcLocation
        writer.write(0, 7); // downmixId
        writer.write_bool(true); // additional downmix IDs present
        writer.write(2, 3); // two additional IDs
        writer.write(4, 7);
        writer.write(5, 7);
        writer.write(1, 16); // non-ducking effect
        writer.write_bool(true); // limiter peak target present
        writer.write(42, 8);
        writer.write_bool(true); // target loudness present
        writer.write(12, 6); // upper
        writer.write_bool(true); // lower present
        writer.write(22, 6); // lower
        writer.write_bool(false); // no config extension

        let bits = writer.bits_written();
        let config = UniDrcConfig::parse_foundation(&writer.finish()).unwrap();
        assert_eq!(config.channel_layout.base_channel_count, 1);
        assert!(config.coefficients.is_empty());
        assert!(config.instructions.is_empty());
        assert!(!config.extension_present);
        assert_eq!(config.bits_read, bits);
    }

    #[test]
    fn preserves_config_and_gain_extensions_at_bit_precision() {
        let mut config_writer = BitWriter::new();
        config_writer.write_bool(false); // sample rate absent
        config_writer.write(0, 7); // no downmix
        config_writer.write_bool(false); // no basic description
        config_writer.write(0, 3); // no coefficients
        config_writer.write(0, 6); // no instructions
        config_writer.write(1, 7); // mono
        config_writer.write_bool(false); // no layout signaling
        config_writer.write_bool(true); // config extension present
        config_writer.write(3, 4); // unknown extension type
        config_writer.write(0, 4); // bitSizeLen = 4
        config_writer.write(4, 4); // payload is 5 bits
        config_writer.write(0b10101, 5);
        config_writer.write(0, 4); // terminator
        let config_bits = config_writer.bits_written();
        let config = UniDrcConfig::parse_foundation(&config_writer.finish()).unwrap();
        assert_eq!(config.extensions.len(), 1);
        assert_eq!(config.extensions[0].extension_type, 3);
        assert_eq!(config.extensions[0].bit_size, 5);
        assert_eq!(config.extensions[0].payload, [0b1010_1000]);
        assert_eq!(config.bits_read, config_bits);

        let coefficients = DrcCoefficients {
            drc_location: 0,
            drc_frame_size: None,
            gain_sequence_count: 0,
            gain_sets: Vec::new(),
            custom_characteristics_left: Vec::new(),
            custom_characteristics_right: Vec::new(),
            shape_filters: Vec::new(),
        };
        let mut gain_writer = BitWriter::new();
        gain_writer.write_bool(true); // gain extension present
        gain_writer.write(5, 4); // unknown extension type
        gain_writer.write(0, 3); // bitSizeLen = 4
        gain_writer.write(8, 4); // payload is 9 bits
        gain_writer.write(0b1100_0011, 8);
        gain_writer.write(1, 1);
        gain_writer.write(0, 4); // terminator
        let gain_bits = gain_writer.bits_written();
        let gain = coefficients
            .parse_gain_payload(&gain_writer.finish(), 1024, 32)
            .unwrap();
        assert_eq!(gain.extensions.len(), 1);
        assert_eq!(gain.extensions[0].extension_type, 5);
        assert_eq!(gain.extensions[0].bit_size, 9);
        assert_eq!(gain.extensions[0].payload, [0b1100_0011, 0b1000_0000]);
        assert_eq!(gain.bits_read, gain_bits);
    }

    #[test]
    fn merges_v1_downmix_from_config_extension() {
        let mut payload = BitWriter::new();
        payload.write_bool(true); // v1 downmix instructions present
        payload.write(1, 7); // one downmix
        payload.write(5, 7); // downmixId
        payload.write(1, 7); // target channels
        payload.write(1, 8); // target layout
        payload.write_bool(false); // coefficient matrix absent
        payload.write_bool(false); // v1 coefficients/instructions absent
        payload.write_bool(true); // loud-EQ present
        payload.write(1, 4); // one loud-EQ instruction
        payload.write(1, 4); // loudEqSetId
        payload.write(1, 4); // drcLocation
        payload.write_bool(false); // downmixId absent
        payload.write_bool(false); // drcSetId absent
        payload.write_bool(false); // eqSetId absent
        payload.write_bool(true); // loudnessAfterDrc
        payload.write_bool(false); // loudnessAfterEq
        payload.write(0, 6); // no loud-EQ gain sequences
        payload.write_bool(false); // EQ absent
        let payload_bits = payload.bits_written();
        let payload_bytes = payload.finish();

        let mut writer = BitWriter::new();
        writer.write_bool(false); // sample rate absent
        writer.write(0, 7); // no v0 downmix
        writer.write_bool(false); // no basic description
        writer.write(0, 3); // no v0 coefficients
        writer.write(0, 6); // no v0 instructions
        writer.write(1, 7); // mono base layout
        writer.write_bool(false); // no explicit layout
        writer.write_bool(true); // extension present
        writer.write(2, 4); // UNIDRCCONFEXT_V1
        writer.write(2, 4); // bitSizeLen = 6
        writer.write((payload_bits - 1) as u32, 6);
        for bit in 0..payload_bits {
            writer.write(((payload_bytes[bit / 8] >> (7 - bit % 8)) & 1) as u32, 1);
        }
        writer.write(0, 4); // extension terminator

        let config = UniDrcConfig::parse_foundation(&writer.finish()).unwrap();
        assert_eq!(config.extensions[0].extension_type, 2);
        assert_eq!(config.downmix_instructions.len(), 1);
        assert_eq!(config.downmix_instructions[0].downmix_id, 5);
        assert_eq!(config.downmix_instructions[0].target_channel_count, 1);
    }

    #[test]
    fn merges_v1_coefficients_and_instruction_from_config_extension() {
        let mut payload = BitWriter::new();
        payload.write_bool(false); // no v1 downmix
        payload.write_bool(true); // coefficients/instructions present
        payload.write(1, 3); // one coefficient set
        payload.write(1, 4); // coefficient location
        payload.write_bool(false); // frame size absent
        payload.write_bool(false); // left characteristics absent
        payload.write_bool(false); // right characteristics absent
        payload.write_bool(false); // shape filters absent
        payload.write(1, 6); // gainSequenceCount
        payload.write(1, 6); // gainSetCount
        payload.write(3, 2); // constant profile
        payload.write_bool(true); // linear
        payload.write_bool(true); // full frame
        payload.write_bool(false); // time alignment
        payload.write_bool(false); // timeDeltaMin absent
        payload.write(1, 6); // one instruction
        payload.write(7, 6); // drcSetId
        payload.write(3, 4); // complexity
        payload.write(1, 4); // location
        payload.write_bool(false); // base layout
        payload.write(0, 16); // effect
        payload.write_bool(false); // limiter absent
        payload.write_bool(false); // target loudness absent
        payload.write_bool(false); // dependency absent
        payload.write_bool(false); // independent use allowed
        payload.write_bool(false); // EQ not required
        payload.write(1, 6); // gain set index 0
        payload.write_bool(false); // no channel repeat
        payload.write_bool(false); // target left absent
        payload.write_bool(false); // target right absent
        payload.write_bool(false); // scaling absent
        payload.write_bool(false); // offset absent
        payload.write_bool(false); // shape filter absent
        payload.write_bool(false); // loud-EQ absent
        payload.write_bool(true); // EQ present
        payload.write_bool(false); // eqDelayMax absent
        payload.write(0, 6); // no filter blocks
        payload.write(0, 6); // no TD filter elements
        payload.write(0, 6); // no subband gains
        payload.write(1, 4); // one EQ instruction
        payload.write(2, 6); // eqSetId
        payload.write(1, 4); // complexity
        payload.write_bool(false); // downmix absent
        payload.write(0, 6); // drcSetId
        payload.write_bool(false); // no additional drcSetId
        payload.write(1, 16); // purpose
        payload.write_bool(false); // no dependency
        payload.write_bool(false); // independent use allowed
        payload.write(3, 7); // mono channel group
        payload.write_bool(false); // no TD cascade
        payload.write_bool(false); // no subband gains
        payload.write_bool(false); // no transition duration
        let payload_bits = payload.bits_written();
        let payload_bytes = payload.finish();
        let size_bits = (usize::BITS - (payload_bits - 1).leading_zeros()) as usize;
        let size_bits = size_bits.max(4);

        let mut writer = BitWriter::new();
        writer.write_bool(false);
        writer.write(0, 7);
        writer.write_bool(false);
        writer.write(0, 3);
        writer.write(0, 6);
        writer.write(1, 7);
        writer.write_bool(false);
        writer.write_bool(true);
        writer.write(2, 4); // UNIDRCCONFEXT_V1
        writer.write((size_bits - 4) as u32, 4);
        writer.write((payload_bits - 1) as u32, size_bits);
        for bit in 0..payload_bits {
            writer.write(((payload_bytes[bit / 8] >> (7 - bit % 8)) & 1) as u32, 1);
        }
        writer.write(0, 4);

        let config = UniDrcConfig::parse_foundation(&writer.finish()).unwrap();
        assert_eq!(config.coefficients.len(), 1);
        assert_eq!(config.coefficients[0].gain_sequence_count, 1);
        assert_eq!(config.instructions.len(), 1);
        assert_eq!(config.instructions[0].drc_set_id, 7);
        assert_eq!(config.instructions[0].complexity_level, 3);
    }

    #[test]
    fn consumes_v1_eq_coefficient_filter_and_spline_syntax() {
        fn zeros(writer: &mut BitWriter, mut count: usize) {
            while count != 0 {
                let chunk = count.min(32);
                writer.write(0, chunk);
                count -= chunk;
            }
        }

        let mut writer = BitWriter::new();
        writer.write_bool(true); // eqDelayMax present
        writer.write(5, 8);
        writer.write(1, 6); // one filter block
        writer.write(1, 6); // one filter element
        writer.write(2, 6); // element index
        writer.write_bool(true); // element gain present
        writer.write(9, 10);
        writer.write(2, 6); // two TD elements
        writer.write_bool(false); // pole/zero format
        writer.write(1, 3); // radius-one zeros
        writer.write(1, 6); // real zeros
        writer.write(1, 6); // generic zeros
        writer.write(1, 4); // real poles
        writer.write(1, 4); // complex poles
        zeros(&mut writer, 2 + 8 + 14 + 8 + 14);
        writer.write_bool(true); // FIR format
        writer.write(3, 7); // order
        zeros(&mut writer, 1 + 2 * 11);
        writer.write(1, 6); // one subband-gain set
        writer.write_bool(true); // spline representation
        writer.write(1, 4); // QMF32
        writer.write(0, 5); // two spline nodes
        writer.write_bool(true); // default node gain
        writer.write_bool(true); // default node gain
        zeros(&mut writer, 4); // frequency delta
        writer.write(0, 2); // slope coding case 0
        zeros(&mut writer, 5); // slope
        zeros(&mut writer, 5); // node frequency delta
        let bits = writer.bits_written();
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        skip_eq_coefficients(&mut reader).unwrap();
        assert_eq!(reader.bits_read(), bits);
    }

    #[test]
    fn parses_selects_and_applies_program_loudness() {
        let mut writer = BitWriter::new();
        writer.write(0, 6); // album count
        writer.write(1, 6); // track count
        writer.write(3, 6); // drcSetId
        writer.write(2, 7); // downmixId
        writer.write_bool(false);
        writer.write_bool(false);
        writer.write(1, 4); // one measurement
        writer.write(1, 4); // program loudness
        writer.write(127, 8); // -26.0 LKFS
        writer.write(1, 4);
        writer.write(3, 2);
        writer.write_bool(false);
        let set = LoudnessInfoSet::parse_v0(&writer.finish()).unwrap();
        assert_eq!(set.select_program_loudness(3, 2, false), Some(-26.0));
        let gain = loudness_normalization_gain_db(-23.0, -26.0);
        assert_eq!(gain, 3.0);
        let mut samples = [1000i16, -1000];
        apply_gain_i16(&mut samples, gain);
        assert!(samples[0] > 1000 && samples[1] < -1000);
    }

    #[test]
    fn loudness_set_caps_album_entries_and_reads_true_peak_special_methods() {
        fn write_empty_info(writer: &mut BitWriter) {
            writer.write(0, 6);
            writer.write(0, 7);
            writer.write_bool(false);
            writer.write_bool(false);
            writer.write(0, 4);
        }

        let mut writer = BitWriter::new();
        writer.write(13, 6); // album count is capped at twelve stored entries
        writer.write(0, 6); // no track entries
        writer.write(1, 6);
        writer.write(2, 7);
        writer.write_bool(false); // no sample peak
        writer.write_bool(true); // true peak present
        writer.write(32, 12);
        writer.write(3, 4);
        writer.write(2, 2);
        writer.write(3, 4); // three measurements
        writer.write(7, 4); // mixing level
        writer.write(5, 5);
        writer.write(1, 4);
        writer.write(1, 2);
        writer.write(8, 4); // room type
        writer.write(2, 2);
        writer.write(2, 4);
        writer.write(3, 2);
        writer.write(9, 4); // short-term loudness uses eight raw bits
        writer.write(10, 8);
        writer.write(3, 4);
        writer.write(0, 2);
        for _ in 1..13 {
            write_empty_info(&mut writer);
        }
        writer.write_bool(false); // no extension
        let set = LoudnessInfoSet::parse_v0(&writer.finish()).unwrap();
        assert_eq!(set.album.len(), 12);
        assert_eq!(set.album[0].true_peak_level, Some(19.0));
        assert_eq!(set.album[0].true_peak_measurement_system, Some(3));
        assert_eq!(set.album[0].true_peak_reliability, Some(2));
        assert_eq!(set.album[0].measurements[0].value, 85.0);
        assert_eq!(set.album[0].measurements[1].value, 2.0);
        assert_eq!(set.album[0].measurements[2].value, -111.0);
    }

    #[test]
    fn parses_v0_gain_sets_bands_and_borders() {
        let mut writer = BitWriter::new();
        writer.write(2, 4); // drcLocation
        writer.write_bool(true);
        writer.write(1023, 15); // frame size 1024
        writer.write(1, 6); // one gain set
        writer.write(0, 2); // regular profile
        writer.write_bool(false); // spline (GIT_SPLINE = 0)
        writer.write_bool(true); // full frame
        writer.write_bool(false); // time alignment
        writer.write_bool(true);
        writer.write(31, 11); // timeDeltaMin 32
        writer.write(2, 4); // two bands
        writer.write_bool(true); // crossover-frequency borders
        writer.write(5, 7);
        writer.write(6, 7);
        writer.write(9, 4); // second band border

        let coefficients = DrcCoefficients::parse_v0(&writer.finish()).unwrap();
        assert_eq!(coefficients.drc_location, 2);
        assert_eq!(coefficients.drc_frame_size, Some(1024));
        assert_eq!(coefficients.gain_sequence_count, 2);
        let set = &coefficients.gain_sets[0];
        assert_eq!(set.coding_profile, GainCodingProfile::Regular);
        assert_eq!(set.interpolation_type, GainInterpolationType::Spline);
        assert_eq!(set.time_delta_min, Some(32));
        assert_eq!(set.bands[0].sequence_index, 0);
        assert_eq!(set.bands[1].sequence_index, 1);
        assert_eq!(
            set.bands[1].border,
            Some(BandBorder::CrossoverFrequencyIndex(9))
        );
    }

    #[test]
    fn parses_v1_custom_characteristics_shape_filters_and_gain_sets() {
        let mut writer = BitWriter::new();
        writer.write(2, 4); // drcLocation
        writer.write_bool(true); // frame size present
        writer.write(1023, 15);
        writer.write_bool(true); // left characteristics present
        writer.write(1, 4); // one left characteristic
        writer.write_bool(false); // sigmoid
        writer.write(12, 6); // +12 dB
        writer.write(2, 4); // I/O ratio 0.35
        writer.write(3, 4); // exponent 7
        writer.write_bool(true); // flip sign
        writer.write_bool(true); // right characteristics present
        writer.write(1, 4); // one right characteristic
        writer.write_bool(true); // nodes
        writer.write(0, 2); // one node after the anchor
        writer.write(3, 5); // +1 dB level delta
        writer.write(120, 8); // -4 dB gain
        writer.write_bool(true); // shape filters present
        writer.write(1, 4); // one filter
        writer.write_bool(true);
        writer.write(7, 5); // LF cut
        writer.write_bool(false); // no LF boost
        writer.write_bool(false); // no HF cut
        writer.write_bool(false); // no HF boost
        writer.write(2, 6); // gainSequenceCount
        writer.write(1, 6); // gainSetCount
        writer.write(0, 2); // regular
        writer.write_bool(true); // linear (GIT_LINEAR = 1)
        writer.write_bool(true); // full frame
        writer.write_bool(false); // time alignment
        writer.write_bool(false); // no timeDeltaMin
        writer.write(2, 4); // two bands
        writer.write_bool(true); // crossover-frequency borders
        writer.write_bool(true); // explicit sequence index
        writer.write(0, 6);
        writer.write_bool(true); // characteristic present
        writer.write_bool(false); // custom indices
        writer.write(1, 4);
        writer.write(1, 4);
        writer.write_bool(true); // explicit sequence index
        writer.write(1, 6);
        writer.write_bool(true); // characteristic present
        writer.write_bool(true); // CICP
        writer.write(5, 7);
        writer.write(9, 4); // second band crossover

        let coefficients = DrcCoefficients::parse_v1(&writer.finish()).unwrap();
        assert_eq!(coefficients.drc_frame_size, Some(1024));
        assert_eq!(coefficients.gain_sequence_count, 2);
        assert_eq!(coefficients.custom_characteristics_left.len(), 1);
        assert_eq!(
            coefficients.custom_characteristics_left[0],
            CustomDrcCharacteristic::Sigmoid {
                gain_db: 12.0,
                io_ratio: 0.05f32 + 0.15 * 2.0,
                exponent: Some(7.0),
                flip_sign: true,
            }
        );
        assert_eq!(
            coefficients.custom_characteristics_right[0],
            CustomDrcCharacteristic::Nodes(vec![
                CharacteristicNode {
                    level_db: -31.0,
                    gain_db: 0.0,
                },
                CharacteristicNode {
                    level_db: -30.0,
                    gain_db: -4.0,
                },
            ])
        );
        assert_eq!(coefficients.shape_filters[0].lf_cut, Some(7));
        let bands = &coefficients.gain_sets[0].bands;
        assert_eq!(
            bands[0].characteristic,
            Some(DrcCharacteristic::Custom {
                left_index: 1,
                right_index: 1,
            })
        );
        assert_eq!(bands[1].characteristic, Some(DrcCharacteristic::Cicp(5)));
        assert_eq!(
            bands[1].border,
            Some(BandBorder::CrossoverFrequencyIndex(9))
        );
    }

    #[test]
    fn gain_set_readers_cover_profiles_subband_borders_and_invalid_counts() {
        for (profile_bits, profile, band_count) in [
            (1, GainCodingProfile::Fading, 2),
            (2, GainCodingProfile::Clipping, 1),
        ] {
            let mut writer = BitWriter::new();
            writer.write(profile_bits, 2);
            writer.write_bool(true); // linear
            writer.write_bool(false); // not full-frame
            writer.write_bool(true); // time aligned
            writer.write_bool(false); // no timeDeltaMin
            writer.write(band_count, 4);
            if band_count > 1 {
                writer.write_bool(false); // start-subband borders
            }
            for _ in 0..band_count {
                writer.write(0, 7); // no CICP characteristic
            }
            if band_count > 1 {
                writer.write(321, 10);
            }
            let mut next = 0;
            let set = read_gain_set_v0(&mut BitReader::new(&writer.finish()), &mut next).unwrap();
            assert_eq!(set.coding_profile, profile);
            assert_eq!(set.bands.len(), band_count as usize);
            if band_count > 1 {
                assert_eq!(
                    set.bands[1].border,
                    Some(BandBorder::StartSubBandIndex(321))
                );
            }
        }

        for (profile_bits, profile, band_count, time_delta_min) in [
            (1, GainCodingProfile::Fading, 2, Some(17)),
            (2, GainCodingProfile::Clipping, 1, None),
        ] {
            let mut writer = BitWriter::new();
            writer.write(profile_bits, 2);
            writer.write_bool(false); // spline
            writer.write_bool(false);
            writer.write_bool(true);
            writer.write_bool(time_delta_min.is_some());
            if let Some(value) = time_delta_min {
                writer.write((value - 1) as u32, 11);
            }
            writer.write(band_count, 4);
            if band_count > 1 {
                writer.write_bool(false); // start-subband borders
            }
            for _ in 0..band_count {
                writer.write_bool(false); // implicit sequence index
                writer.write_bool(false); // no characteristic
            }
            if band_count > 1 {
                writer.write(654, 10);
            }
            let mut next = -1;
            let set = read_gain_set_v1(&mut BitReader::new(&writer.finish()), &mut next).unwrap();
            assert_eq!(set.coding_profile, profile);
            assert_eq!(set.interpolation_type, GainInterpolationType::Spline);
            assert_eq!(set.time_delta_min, time_delta_min);
            assert!(set.bands.iter().all(|band| band.characteristic.is_none()));
            if band_count > 1 {
                assert_eq!(
                    set.bands[1].border,
                    Some(BandBorder::StartSubBandIndex(654))
                );
            }
        }

        for invalid_count in [0, 5] {
            let mut writer = BitWriter::new();
            writer.write(0, 2);
            writer.write_bool(true);
            writer.write_bool(false);
            writer.write_bool(false);
            writer.write_bool(false);
            writer.write(invalid_count, 4);
            assert_eq!(
                read_gain_set_v0(&mut BitReader::new(&writer.finish()), &mut 0),
                Err(DrcError::InvalidBandCount(invalid_count as u8))
            );

            let mut writer = BitWriter::new();
            writer.write(0, 2);
            writer.write_bool(true);
            writer.write_bool(false);
            writer.write_bool(false);
            writer.write_bool(false);
            writer.write(invalid_count, 4);
            assert_eq!(
                read_gain_set_v1(&mut BitReader::new(&writer.finish()), &mut -1),
                Err(DrcError::InvalidBandCount(invalid_count as u8))
            );
        }
    }

    #[test]
    fn gain_and_ducking_modifications_cover_sign_and_default_branches() {
        let mut writer = BitWriter::new();
        writer.write_bool(false); // no left target
        writer.write_bool(false); // no right target
        writer.write_bool(false); // default scaling
        writer.write_bool(true); // gain offset present
        writer.write_bool(false); // positive
        writer.write(3, 5); // +1 dB
        writer.write_bool(false); // no shape filter
        let modification =
            read_gain_modifications_v1(&mut BitReader::new(&writer.finish()), 1).unwrap();
        assert_eq!(modification[0].gain_offset_db, 1.0);

        let mut writer = BitWriter::new();
        writer.write_bool(false); // no left target
        writer.write_bool(true); // right target present
        writer.write(9, 4);
        writer.write_bool(false); // default scaling
        writer.write_bool(false); // no gain offset
        writer.write_bool(false); // no shape filter
        let modification =
            read_gain_modifications_v1(&mut BitReader::new(&writer.finish()), 1).unwrap();
        assert_eq!(modification[0].target_characteristic_right, Some(9));

        let mut writer = BitWriter::new();
        writer.write_bool(false); // default scaling
        writer.write_bool(true);
        writer.write_bool(true); // negative
        writer.write(3, 5);
        let modification =
            read_gain_modification_v0(&mut BitReader::new(&writer.finish())).unwrap();
        assert_eq!(modification.gain_offset_db, -1.0);

        assert_eq!(
            read_ducking_modification(&mut BitReader::new(&[0])).unwrap(),
            DuckingModification { scaling: 1.0 }
        );
        let mut writer = BitWriter::new();
        writer.write_bool(true);
        writer.write(2, 4);
        assert_eq!(
            read_ducking_modification(&mut BitReader::new(&writer.finish())).unwrap(),
            DuckingModification { scaling: 1.375 }
        );
    }

    #[test]
    fn skips_optional_basic_loud_eq_and_explicit_layout_fields() {
        let mut basic = BitWriter::new();
        basic.write(1, 6);
        basic.write(2, 4);
        basic.write(3, 7);
        basic.write_bool(true);
        basic.write(1, 3);
        basic.write(4, 7);
        basic.write(0, 16);
        basic.write_bool(true);
        basic.write(5, 8);
        basic.write_bool(true);
        basic.write(6, 6);
        basic.write_bool(true);
        basic.write(7, 6);
        let bits = basic.bits_written();
        let bytes = basic.finish();
        let mut reader = BitReader::with_bit_len(&bytes, bits).unwrap();
        skip_drc_instructions_basic(&mut reader).unwrap();
        assert_eq!(reader.bits_read(), bits);

        let mut loud_eq = BitWriter::new();
        loud_eq.write(1, 4);
        loud_eq.write(2, 4);
        for bits in [7usize, 6, 6] {
            loud_eq.write_bool(true);
            loud_eq.write(1, bits);
            loud_eq.write_bool(true);
            loud_eq.write(1, bits);
            loud_eq.write(2, bits);
        }
        loud_eq.write_bool(false);
        loud_eq.write_bool(true);
        loud_eq.write(0, 6);
        let bits = loud_eq.bits_written();
        let bytes = loud_eq.finish();
        let mut reader = BitReader::with_bit_len(&bytes, bits).unwrap();
        skip_loud_eq_instruction(&mut reader).unwrap();
        assert_eq!(reader.bits_read(), bits);

        let mut layout = BitWriter::new();
        layout.write(2, 7);
        layout.write_bool(true);
        layout.write(0, 8);
        layout.write(3, 7);
        layout.write(4, 7);
        let parsed = read_channel_layout(&mut BitReader::new(&layout.finish())).unwrap();
        assert_eq!(parsed.base_channel_count, 2);
        assert_eq!(parsed.defined_layout, Some(0));
        assert_eq!(parsed.speaker_positions, vec![3, 4]);
    }

    #[test]
    fn extension_reader_rejects_truncation_count_and_size_mismatch() {
        let mut truncated = BitWriter::new();
        truncated.write(1, 4);
        truncated.write(0, 3); // four size bits
        truncated.write(15, 4); // sixteen payload bits
        assert!(matches!(
            read_drc_extensions(&mut BitReader::new(&truncated.finish()), 3),
            Err(DrcError::Bit(BitError::UnexpectedEof { .. }))
        ));

        let mut excessive = BitWriter::new();
        for _ in 0..7 {
            excessive.write(1, 4);
            excessive.write(0, 3);
            excessive.write(0, 4);
            excessive.write_bool(false);
        }
        excessive.write(1, 4);
        assert_eq!(
            read_drc_extensions(&mut BitReader::new(&excessive.finish()), 3),
            Err(DrcError::TooManyExtensions)
        );

        let extension = DrcExtension {
            extension_type: 1,
            bit_size: 8,
            payload: vec![0],
        };
        assert_eq!(
            merge_v1_config_extension(
                &extension,
                &ChannelLayout {
                    base_channel_count: 2,
                    defined_layout: None,
                    speaker_positions: Vec::new(),
                },
                &mut Vec::new(),
                &mut Vec::new(),
                &mut Vec::new(),
            ),
            Err(DrcError::ExtensionSizeMismatch {
                declared: 8,
                consumed: 4,
            })
        );
    }

    #[test]
    fn decodes_and_applies_simple_uni_drc_gain_sequence() {
        let coefficients = DrcCoefficients {
            drc_location: 1,
            drc_frame_size: Some(1024),
            gain_sequence_count: 1,
            gain_sets: vec![GainSet {
                coding_profile: GainCodingProfile::Regular,
                interpolation_type: GainInterpolationType::Linear,
                full_frame: true,
                time_alignment: false,
                time_delta_min: None,
                drc_band_type: false,
                bands: vec![GainBand {
                    sequence_index: 0,
                    cicp_characteristic_index: Some(1),
                    characteristic: Some(DrcCharacteristic::Cicp(1)),
                    border: None,
                }],
            }],
            custom_characteristics_left: Vec::new(),
            custom_characteristics_right: Vec::new(),
            shape_filters: Vec::new(),
        };
        let mut writer = BitWriter::new();
        writer.write_bool(false); // simple coding mode
        writer.write_bool(true); // negative
        writer.write(16, 8); // -2 dB
        writer.write_bool(false); // no extension
        let gain = coefficients
            .parse_gain_payload(&writer.finish(), 1024, 32)
            .unwrap();
        assert_eq!(gain.sequences[0].nodes[0].gain_db, -2.0);
        assert_eq!(gain.sequences[0].nodes[0].time, 1023);
        let mut samples = [1000i16, -1000];
        gain.apply_sequence_i16(0, &mut samples).unwrap();
        assert!(samples[0] < 1000 && samples[0] > 700);
        assert!(samples[1] > -1000 && samples[1] < -700);
    }

    #[test]
    fn decodes_complex_spline_gain_node() {
        let coefficients = DrcCoefficients {
            drc_location: 1,
            drc_frame_size: Some(1024),
            gain_sequence_count: 1,
            gain_sets: vec![GainSet {
                coding_profile: GainCodingProfile::Regular,
                interpolation_type: GainInterpolationType::Spline,
                full_frame: true,
                time_alignment: false,
                time_delta_min: Some(32),
                drc_band_type: false,
                bands: vec![GainBand {
                    sequence_index: 0,
                    cicp_characteristic_index: Some(1),
                    characteristic: Some(DrcCharacteristic::Cicp(1)),
                    border: None,
                }],
            }],
            custom_characteristics_left: Vec::new(),
            custom_characteristics_right: Vec::new(),
            shape_filters: Vec::new(),
        };
        let mut writer = BitWriter::new();
        writer.write_bool(true); // complex mode
        writer.write_bool(true); // unary node-count terminator: one node
        writer.write_bool(true); // slope Huffman value 7 => zero slope
        writer.write_bool(false); // positive initial gain
        writer.write(8, 8); // +1 dB
        writer.write_bool(false); // no extension
        let gain = coefficients
            .parse_gain_payload(&writer.finish(), 1024, 32)
            .unwrap();
        assert_eq!(gain.sequences[0].nodes.len(), 1);
        assert_eq!(gain.sequences[0].nodes[0].time, 1023);
        assert_eq!(gain.sequences[0].nodes[0].gain_db, 1.0);
        assert_eq!(gain.sequences[0].nodes[0].slope, 0.0);
    }

    #[test]
    fn applies_per_sample_linear_and_spline_gain_interpolation() {
        let linear = GainSequence {
            interpolation_type: GainInterpolationType::Linear,
            nodes: vec![
                GainNode {
                    time: 0,
                    gain_db: 0.0,
                    slope: 0.0,
                },
                GainNode {
                    time: 4,
                    gain_db: 8.0,
                    slope: 0.0,
                },
            ],
        };
        assert_eq!(linear.gain_db_at(2), 4.0);
        let spline = GainSequence {
            interpolation_type: GainInterpolationType::Spline,
            nodes: linear.nodes.clone(),
        };
        assert!((spline.gain_db_at(2) - 4.0).abs() < 1e-6);

        let gain = UniDrcGain {
            sequences: vec![linear],
            extension_present: false,
            extensions: Vec::new(),
            bits_read: 0,
        };
        let mut samples = [1000.0f32; 5];
        gain.apply_sequence_f32(0, &mut samples).unwrap();
        assert!(samples.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn parses_and_applies_v0_downmix_matrix() {
        let mut writer = BitWriter::new();
        writer.write(4, 7); // downmixId
        writer.write(1, 7); // mono target
        writer.write(1, 8); // target layout
        writer.write_bool(true);
        writer.write(6, 4); // left * 0.7079458
        writer.write(6, 4); // right * 0.7079458
        let downmix = DownmixInstructions::parse_v0(&writer.finish(), 2).unwrap();
        let output = downmix
            .apply_interleaved_i16(&[1000, 1000, -1000, -1000], 2)
            .unwrap();
        assert_eq!(output.len(), 2);
        assert!((output[0] as i32 - 1416).abs() <= 2);
        assert!((output[1] as i32 + 1416).abs() <= 2);
    }

    #[test]
    fn parses_and_applies_v1_downmix_matrix() {
        let mut writer = BitWriter::new();
        writer.write(5, 7); // downmixId
        writer.write(1, 7); // mono target
        writer.write(1, 8); // target layout
        writer.write_bool(true);
        writer.write(9, 4); // bsDownmixOffset
        writer.write(5, 5); // left * 1.0
        writer.write(11, 5); // right * 0.7079458
        let downmix = DownmixInstructions::parse_v1(&writer.finish(), 2).unwrap();
        assert_eq!(downmix.coefficient_offset, 9);
        let output = downmix.apply_interleaved_f32(&[1000.0, 1000.0], 2).unwrap();
        assert!((output[0] - 1707.9458).abs() < 0.01);
    }

    #[test]
    fn parses_v0_drc_instruction_channel_groups_and_modification() {
        let layout = ChannelLayout {
            base_channel_count: 2,
            defined_layout: None,
            speaker_positions: Vec::new(),
        };
        let mut writer = BitWriter::new();
        writer.write(3, 6); // drcSetId
        writer.write(1, 4); // drcLocation
        writer.write(0, 7); // base layout
        writer.write_bool(false); // no additional downmix
        writer.write(0, 16); // no ducking effect
        writer.write_bool(true); // limiter peak
        writer.write(8, 8); // -1 dB
        writer.write_bool(true); // target loudness
        writer.write(40, 6); // -23
        writer.write_bool(true);
        writer.write(34, 6); // -29
        writer.write_bool(false); // no dependency
        writer.write_bool(false); // independent use allowed
        writer.write(1, 6); // gain set index 0
        writer.write_bool(true); // repeat
        writer.write(0, 5); // repeat once for second channel
        writer.write_bool(true); // gain scaling
        writer.write(8, 4); // attenuation 1.0
        writer.write(8, 4); // amplification 1.0
        writer.write_bool(true); // gain offset
        writer.write_bool(false);
        writer.write(3, 5); // +1 dB
        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        let instruction =
            DrcInstruction::parse_v0_from_reader(&mut reader, &layout, &[], &[]).unwrap();
        assert_eq!(instruction.drc_set_id, 3);
        assert_eq!(instruction.limiter_peak_target_db, Some(-1.0));
        assert_eq!(instruction.target_loudness_upper, Some(-23));
        assert_eq!(instruction.target_loudness_lower, Some(-29));
        assert_eq!(instruction.gain_set_index_per_channel, [0, 0]);
        assert_eq!(instruction.gain_modifications[0].gain_offset_db, 1.0);
    }

    #[test]
    fn parses_v0_and_v1_ducking_instructions_and_expands_channels() {
        let stereo = ChannelLayout {
            base_channel_count: 2,
            defined_layout: None,
            speaker_positions: Vec::new(),
        };
        let mut v0 = BitWriter::new();
        v0.write(5, 6); // drcSetId
        v0.write(1, 4); // drcLocation
        v0.write(0, 7); // base layout
        v0.write_bool(false); // no additional downmix IDs
        v0.write(DRC_EFFECT_DUCK_OTHER as u32, 16);
        v0.write_bool(true); // target loudness present
        v0.write(40, 6); // -23 LUFS upper
        v0.write_bool(false); // lower absent
        v0.write_bool(true); // dependency present
        v0.write(2, 6);
        v0.write(1, 6); // gain set index 0
        v0.write_bool(true); // ducking scaling present
        v0.write(10, 4); // 0.625
        v0.write_bool(true); // repeat for the right channel
        v0.write(0, 5);
        let v0_bytes = v0.finish();
        let instruction =
            DrcInstruction::parse_v0_from_reader(&mut BitReader::new(&v0_bytes), &stereo, &[], &[])
                .unwrap();
        assert_eq!(instruction.target_loudness_upper, Some(-23));
        assert_eq!(instruction.depends_on_drc_set, Some(2));
        assert_eq!(instruction.gain_set_index_per_channel, [0, 0]);
        assert_eq!(instruction.ducking_modifications.len(), 2);
        assert!(instruction
            .ducking_modifications
            .iter()
            .all(|modification| modification.scaling == 0.625));

        let mut v0_mixed = BitWriter::new();
        v0_mixed.write(5, 6);
        v0_mixed.write(1, 4);
        v0_mixed.write(0, 7);
        v0_mixed.write_bool(false);
        v0_mixed.write(DRC_EFFECT_DUCK_OTHER as u32, 16);
        v0_mixed.write_bool(false); // no target loudness
        v0_mixed.write_bool(false); // no dependency
        v0_mixed.write_bool(false); // independent use allowed
        v0_mixed.write(1, 6);
        v0_mixed.write_bool(true);
        v0_mixed.write(2, 4); // positive 1.375 scaling
        v0_mixed.write_bool(false);
        v0_mixed.write(1, 6);
        v0_mixed.write_bool(false); // default 1.0 scaling
        v0_mixed.write_bool(false);
        let instruction = DrcInstruction::parse_v0_from_reader(
            &mut BitReader::new(&v0_mixed.finish()),
            &stereo,
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(
            instruction.ducking_modifications,
            [
                DuckingModification { scaling: 1.375 },
                DuckingModification { scaling: 1.0 },
            ]
        );

        let mut v0_any_downmix = BitWriter::new();
        v0_any_downmix.write(6, 6);
        v0_any_downmix.write(1, 4);
        v0_any_downmix.write(0x7f, 7); // any downmix expands to eight channels
        v0_any_downmix.write_bool(true); // one additional downmix ID
        v0_any_downmix.write(1, 3);
        v0_any_downmix.write(3, 7);
        v0_any_downmix.write(DRC_EFFECT_DUCK_SELF as u32, 16);
        v0_any_downmix.write_bool(false); // no target loudness
        v0_any_downmix.write_bool(false); // no dependency
        v0_any_downmix.write_bool(false); // independent use allowed
        v0_any_downmix.write(1, 6); // gain set index 0
        v0_any_downmix.write_bool(false); // default ducking scale
        v0_any_downmix.write_bool(true); // repeat through all remaining channels
        v0_any_downmix.write(6, 5); // seven repeats plus the first channel
        let instruction = DrcInstruction::parse_v0_from_reader(
            &mut BitReader::new(&v0_any_downmix.finish()),
            &stereo,
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(instruction.downmix_ids, [0x7f, 3]);
        assert_eq!(instruction.channel_count, 8);
        assert_eq!(instruction.gain_set_index_per_channel, [0; 8]);
        assert_eq!(instruction.ducking_modifications.len(), 8);

        let mut v0_overflow = BitWriter::new();
        v0_overflow.write(5, 6);
        v0_overflow.write(1, 4);
        v0_overflow.write(0, 7);
        v0_overflow.write_bool(false);
        v0_overflow.write(DRC_EFFECT_DUCK_OTHER as u32, 16);
        v0_overflow.write_bool(false);
        v0_overflow.write_bool(false);
        v0_overflow.write_bool(false);
        v0_overflow.write(1, 6);
        v0_overflow.write_bool(false);
        v0_overflow.write_bool(true);
        v0_overflow.write(1, 5); // two repeats overflow stereo layout
        assert_eq!(
            DrcInstruction::parse_v0_from_reader(
                &mut BitReader::new(&v0_overflow.finish()),
                &stereo,
                &[],
                &[],
            ),
            Err(DrcError::InstructionChannelOverflow)
        );

        let mono = ChannelLayout {
            base_channel_count: 1,
            defined_layout: None,
            speaker_positions: Vec::new(),
        };
        let mut v1 = BitWriter::new();
        v1.write(6, 6); // drcSetId
        v1.write(3, 4); // complexity
        v1.write(1, 4); // drcLocation
        v1.write_bool(true); // downmix signaling present
        v1.write(0x7f, 7); // any downmix expands to eight channels
        v1.write_bool(true); // apply to downmix
        v1.write_bool(true); // additional downmix ID
        v1.write(1, 3);
        v1.write(3, 7);
        v1.write(DRC_EFFECT_DUCK_SELF as u32, 16);
        v1.write_bool(true); // target loudness present
        v1.write(40, 6);
        v1.write_bool(false); // lower absent
        v1.write_bool(false); // dependency absent
        v1.write_bool(true); // no independent use
        v1.write_bool(false); // EQ not required
        v1.write(1, 6); // gain set index 0
        v1.write_bool(true); // ducking scaling present
        v1.write(2, 4); // 1.375
        v1.write_bool(false); // no repeats in coded channel layout
        let instruction = DrcInstruction::parse_v1(&v1.finish(), &mono, &[], &[]).unwrap();
        assert_eq!(instruction.channel_count, 8);
        assert_eq!(instruction.gain_set_index_per_channel, [0; 8]);
        assert_eq!(instruction.ducking_modifications.len(), 8);
        assert!(instruction.no_independent_use);
        assert!(instruction
            .ducking_modifications
            .iter()
            .all(|modification| modification.scaling == 1.375));

        let mut derived = BitWriter::new();
        derived.write(7, 6); // drcSetId
        derived.write(4, 4); // complexity
        derived.write(1, 4); // drcLocation
        derived.write_bool(true); // downmix signaling present
        derived.write(3, 7); // unknown concrete downmix
        derived.write_bool(true); // apply to downmix
        derived.write_bool(false); // no additional IDs
        derived.write(0, 16); // non-ducking effect
        derived.write_bool(true); // limiter present
        derived.write(8, 8); // -1 dB
        derived.write_bool(true); // target loudness present
        derived.write(40, 6); // -23 LUFS upper
        derived.write_bool(true);
        derived.write(34, 6); // -29 LUFS lower
        derived.write_bool(true); // dependency present
        derived.write(4, 6);
        derived.write_bool(true); // EQ required
        derived.write(1, 6); // gain set index 0
        derived.write_bool(true); // repeated coded channel
        derived.write(1, 5); // two repeats derive a three-channel layout
        for _ in 0..5 {
            derived.write_bool(false); // default gain modification fields
        }
        let derived = DrcInstruction::parse_v1(&derived.finish(), &mono, &[], &[]).unwrap();
        assert_eq!(derived.channel_count, 3);
        assert_eq!(derived.gain_set_index_per_channel, [0; 3]);
        assert_eq!(derived.limiter_peak_target_db, Some(-1.0));
        assert_eq!(derived.target_loudness_upper, Some(-23));
        assert_eq!(derived.target_loudness_lower, Some(-29));
        assert_eq!(derived.depends_on_drc_set, Some(4));
        assert!(derived.requires_eq);

        let mut eight_channel_overflow = BitWriter::new();
        eight_channel_overflow.write(1, 6);
        eight_channel_overflow.write(0, 4);
        eight_channel_overflow.write(1, 4);
        eight_channel_overflow.write_bool(true);
        eight_channel_overflow.write(0x7f, 7);
        eight_channel_overflow.write_bool(true);
        eight_channel_overflow.write_bool(false);
        eight_channel_overflow.write(DRC_EFFECT_DUCK_SELF as u32, 16);
        eight_channel_overflow.write_bool(false);
        eight_channel_overflow.write_bool(false);
        eight_channel_overflow.write_bool(false);
        eight_channel_overflow.write_bool(false);
        eight_channel_overflow.write(1, 6);
        eight_channel_overflow.write_bool(false);
        eight_channel_overflow.write_bool(true);
        eight_channel_overflow.write(7, 5);
        assert_eq!(
            DrcInstruction::parse_v1(&eight_channel_overflow.finish(), &mono, &[], &[],),
            Err(DrcError::InstructionChannelOverflow)
        );

        let downmix = DownmixInstructions {
            downmix_id: 5,
            target_channel_count: 2,
            target_layout: 0,
            coefficient_offset: 0,
            coefficients: None,
        };
        let mut target_overflow = BitWriter::new();
        target_overflow.write(1, 6);
        target_overflow.write(0, 4);
        target_overflow.write(1, 4);
        target_overflow.write_bool(true);
        target_overflow.write(5, 7);
        target_overflow.write_bool(true);
        target_overflow.write_bool(false);
        target_overflow.write(0, 16);
        target_overflow.write_bool(false); // limiter absent
        target_overflow.write_bool(false);
        target_overflow.write_bool(false);
        target_overflow.write_bool(false);
        target_overflow.write_bool(false);
        target_overflow.write(1, 6);
        target_overflow.write_bool(true);
        target_overflow.write(1, 5);
        assert_eq!(
            DrcInstruction::parse_v1(&target_overflow.finish(), &mono, &[downmix], &[],),
            Err(DrcError::InstructionChannelOverflow)
        );
    }

    #[test]
    fn parses_v1_instruction_complexity_eq_and_per_band_modification() {
        let layout = ChannelLayout {
            base_channel_count: 1,
            defined_layout: None,
            speaker_positions: Vec::new(),
        };
        let coefficients = DrcCoefficients {
            drc_location: 1,
            drc_frame_size: None,
            gain_sequence_count: 1,
            gain_sets: vec![GainSet {
                coding_profile: GainCodingProfile::Regular,
                interpolation_type: GainInterpolationType::Linear,
                full_frame: true,
                time_alignment: false,
                time_delta_min: None,
                drc_band_type: false,
                bands: vec![GainBand {
                    sequence_index: 0,
                    cicp_characteristic_index: Some(1),
                    characteristic: Some(DrcCharacteristic::Cicp(1)),
                    border: None,
                }],
            }],
            custom_characteristics_left: Vec::new(),
            custom_characteristics_right: Vec::new(),
            shape_filters: Vec::new(),
        };
        let mut writer = BitWriter::new();
        writer.write(3, 6); // drcSetId
        writer.write(4, 4); // complexity
        writer.write(1, 4); // location
        writer.write_bool(false); // base downmix defaults
        writer.write(0, 16); // effect
        writer.write_bool(false); // limiter absent
        writer.write_bool(false); // target loudness absent
        writer.write_bool(false); // no dependency
        writer.write_bool(false); // independent use allowed
        writer.write_bool(true); // requires EQ
        writer.write(1, 6); // gain set 0
        writer.write_bool(false); // no channel repeat
        writer.write_bool(true); // target left present
        writer.write(2, 4);
        writer.write_bool(false); // target right absent
        writer.write_bool(true); // scaling present
        writer.write(8, 4); // attenuation 1.0
        writer.write(4, 4); // amplification 0.5
        writer.write_bool(true); // offset present
        writer.write_bool(true); // negative
        writer.write(3, 5); // -1 dB
        writer.write_bool(true); // shape filter present
        writer.write(6, 4);
        let instruction =
            DrcInstruction::parse_v1(&writer.finish(), &layout, &[], &[coefficients]).unwrap();
        assert_eq!(instruction.complexity_level, 4);
        assert!(instruction.requires_eq);
        let modification = instruction.gain_modifications_per_band[0][0];
        assert_eq!(modification.target_characteristic_left, Some(2));
        assert_eq!(modification.amplification_scaling, 0.5);
        assert_eq!(modification.gain_offset_db, -1.0);
        assert_eq!(modification.shape_filter_index, Some(6));
    }

    #[test]
    fn selects_and_applies_drc_instruction_per_channel() {
        let coefficients = DrcCoefficients {
            drc_location: 1,
            drc_frame_size: Some(1024),
            gain_sequence_count: 1,
            gain_sets: vec![GainSet {
                coding_profile: GainCodingProfile::Regular,
                interpolation_type: GainInterpolationType::Linear,
                full_frame: true,
                time_alignment: false,
                time_delta_min: None,
                drc_band_type: false,
                bands: vec![GainBand {
                    sequence_index: 0,
                    cicp_characteristic_index: Some(1),
                    characteristic: Some(DrcCharacteristic::Cicp(1)),
                    border: None,
                }],
            }],
            custom_characteristics_left: Vec::new(),
            custom_characteristics_right: Vec::new(),
            shape_filters: Vec::new(),
        };
        let instruction = DrcInstruction {
            drc_set_id: 3,
            complexity_level: 2,
            drc_location: 1,
            downmix_ids: vec![0],
            apply_to_downmix: false,
            effect: 1,
            limiter_peak_target_db: None,
            target_loudness_upper: Some(-20),
            target_loudness_lower: Some(-30),
            depends_on_drc_set: None,
            no_independent_use: false,
            requires_eq: false,
            channel_count: 2,
            gain_set_index_per_channel: vec![0, -1],
            gain_modifications: vec![GainModification {
                target_characteristic_left: None,
                target_characteristic_right: None,
                attenuation_scaling: 0.5,
                amplification_scaling: 1.0,
                gain_offset_db: 1.0,
                shape_filter_index: None,
            }],
            gain_modifications_per_band: Vec::new(),
            ducking_modifications: Vec::new(),
        };
        let config = UniDrcConfig {
            sample_rate: Some(48_000),
            channel_layout: ChannelLayout {
                base_channel_count: 2,
                defined_layout: None,
                speaker_positions: Vec::new(),
            },
            downmix_instructions: Vec::new(),
            coefficients: vec![coefficients],
            instructions: vec![instruction],
            extension_present: false,
            extensions: Vec::new(),
            bits_read: 0,
        };
        let selected = config
            .select_instruction(DrcSelectionRequest {
                target_loudness: Some(-23.0),
                preferred_effect_mask: 1,
                ..DrcSelectionRequest::default()
            })
            .unwrap();
        assert_eq!(selected.drc_set_id, 3);
        let mut fallback = config.clone();
        fallback.instructions[0].downmix_ids = vec![0x7f];
        assert!(fallback
            .select_instruction(DrcSelectionRequest {
                downmix_id: 9,
                ..DrcSelectionRequest::default()
            })
            .is_some());

        let gain = UniDrcGain {
            sequences: vec![GainSequence {
                interpolation_type: GainInterpolationType::Linear,
                nodes: vec![GainNode {
                    time: 0,
                    gain_db: -6.0,
                    slope: 0.0,
                }],
            }],
            extension_present: false,
            extensions: Vec::new(),
            bits_read: 0,
        };
        assert_eq!(
            config.apply_instruction_f32(selected, &gain, &mut [], 0),
            Err(DrcError::InstructionLayoutMismatch)
        );
        assert_eq!(
            config.apply_instruction_i16(selected, &gain, &mut [0], 2),
            Err(DrcError::InstructionLayoutMismatch)
        );
        let mut samples = [1000.0f32, 1000.0, 1000.0, 1000.0];
        config
            .apply_instruction_f32(selected, &gain, &mut samples, 2)
            .unwrap();
        // -6 dB * 0.5 + 1 dB = -2 dB. The unassigned channel is unchanged.
        let expected = 1000.0 * 10.0f32.powf(-2.0 / 20.0);
        assert!((samples[0] - expected).abs() < 0.01);
        assert!((samples[2] - expected).abs() < 0.01);
        assert_eq!(samples[1], 1000.0);
        assert_eq!(samples[3], 1000.0);

        let mut integer_samples = [1000i16, 1000, 1000, 1000];
        config
            .apply_instruction_i16(selected, &gain, &mut integer_samples, 2)
            .unwrap();
        assert_eq!(integer_samples[0], expected.round() as i16);
        assert_eq!(integer_samples[1], 1000);

        let mut half_attenuation = [1000.0f32, 1000.0];
        config
            .apply_instruction_f32_scaled(selected, &gain, &mut half_attenuation, 2, 0.5, 1.0)
            .unwrap();
        let half_attenuation_expected = 1000.0 * 10.0f32.powf(-1.0 / 20.0);
        assert!((half_attenuation[0] - half_attenuation_expected).abs() < 0.01);

        let positive_gain = UniDrcGain {
            sequences: vec![GainSequence {
                interpolation_type: GainInterpolationType::Linear,
                nodes: vec![GainNode {
                    time: 0,
                    gain_db: 6.0,
                    slope: 0.0,
                }],
            }],
            extension_present: false,
            extensions: Vec::new(),
            bits_read: 0,
        };
        let mut amplified = [1000.0f32, 1000.0];
        config
            .apply_instruction_f32(selected, &positive_gain, &mut amplified, 2)
            .unwrap();
        let amplified_expected = 1000.0 * 10.0f32.powf(7.0 / 20.0);
        assert!((amplified[0] - amplified_expected).abs() < 0.01);
        let mut amplified_i16 = [1000i16, 1000];
        config
            .apply_instruction_i16(selected, &positive_gain, &mut amplified_i16, 2)
            .unwrap();
        assert_eq!(amplified_i16[0], amplified_expected.round() as i16);
        let mut half_boost = [1000i16, 1000];
        config
            .apply_instruction_i16_scaled(selected, &positive_gain, &mut half_boost, 2, 1.0, 0.5)
            .unwrap();
        assert_eq!(
            half_boost[0],
            (1000.0 * 10.0f32.powf(3.5 / 20.0)).round() as i16
        );

        let mut unmodified = selected.clone();
        unmodified.gain_modifications.clear();
        let mut unchanged = [1000.0f32, 1000.0];
        config
            .apply_instruction_f32(&unmodified, &gain, &mut unchanged, 2)
            .unwrap();
        assert!((unchanged[0] - 1000.0 * 10.0f32.powf(-6.0 / 20.0)).abs() < 0.01);

        let mut ducking = selected.clone();
        ducking.effect = DRC_EFFECT_DUCK_SELF;
        ducking.gain_set_index_per_channel = vec![0, 0];
        ducking.ducking_modifications = vec![
            DuckingModification { scaling: 0.5 },
            DuckingModification { scaling: 1.0 },
        ];
        let mut ducked = [1000.0f32, 1000.0];
        config
            .apply_instruction_f32(&ducking, &gain, &mut ducked, 2)
            .unwrap();
        assert!((ducked[0] - 1000.0 * 10.0f32.powf(-3.0 / 20.0)).abs() < 0.01);
        assert!((ducked[1] - 1000.0 * 10.0f32.powf(-6.0 / 20.0)).abs() < 0.01);
        let mut ducked_i16 = [1000i16, 1000];
        config
            .apply_instruction_i16(&ducking, &gain, &mut ducked_i16, 2)
            .unwrap();
        assert_eq!(
            ducked_i16,
            [
                (1000.0 * 10.0f32.powf(-3.0 / 20.0)).round() as i16,
                (1000.0 * 10.0f32.powf(-6.0 / 20.0)).round() as i16,
            ]
        );
    }

    #[test]
    fn gain_sequence_covers_empty_edges_linear_and_spline_slopes() {
        let empty = GainSequence {
            interpolation_type: GainInterpolationType::Linear,
            nodes: Vec::new(),
        };
        assert_eq!(empty.gain_db_at(10), 0.0);
        let linear = GainSequence {
            interpolation_type: GainInterpolationType::Linear,
            nodes: vec![
                GainNode {
                    time: 2,
                    gain_db: -4.0,
                    slope: 0.0,
                },
                GainNode {
                    time: 2,
                    gain_db: 4.0,
                    slope: 0.0,
                },
            ],
        };
        assert_eq!(linear.gain_db_at(0), -4.0);
        assert_eq!(linear.gain_db_at(3), 4.0);
        let spline = GainSequence {
            interpolation_type: GainInterpolationType::Spline,
            nodes: vec![
                GainNode {
                    time: 0,
                    gain_db: 0.0,
                    slope: 2.0,
                },
                GainNode {
                    time: 4,
                    gain_db: 0.0,
                    slope: -2.0,
                },
            ],
        };
        assert!(spline.gain_db_at(2) > 0.0);
        assert_eq!(spline.gain_db_at(5), 0.0);
    }

    #[test]
    fn gain_application_validates_sequence_and_saturates_i16() {
        let gain = UniDrcGain {
            sequences: vec![GainSequence {
                interpolation_type: GainInterpolationType::Linear,
                nodes: vec![GainNode {
                    time: 0,
                    gain_db: 20.0,
                    slope: 0.0,
                }],
            }],
            extension_present: false,
            extensions: Vec::new(),
            bits_read: 0,
        };
        assert_eq!(
            gain.apply_sequence_f32(1, &mut []),
            Err(DrcError::MissingGainSequence(1))
        );
        assert_eq!(
            gain.apply_sequence_i16(1, &mut []),
            Err(DrcError::MissingGainSequence(1))
        );
        let mut floats = [1.0, -1.0];
        gain.apply_sequence_f32(0, &mut floats).unwrap();
        assert_eq!(floats, [10.0, -10.0]);
        let mut integers = [10_000, -10_000];
        gain.apply_sequence_i16(0, &mut integers).unwrap();
        assert_eq!(integers, [i16::MAX, i16::MIN]);
    }

    #[test]
    fn time_delta_decoder_covers_all_prefix_classes() {
        assert_eq!(drc_time_delta_z(0), 1);
        assert_eq!(drc_time_delta_z(16), 5);
        for (bytes, z, expected) in [
            (&[0b0000_0000][..], 5, 1),
            (&[0b0100_0000][..], 5, 2),
            (&[0b1000_0000][..], 5, 6),
            (&[0b1100_0000][..], 5, 14),
        ] {
            assert_eq!(
                decode_time_delta(&mut BitReader::new(bytes), z).unwrap(),
                expected
            );
        }
        assert_eq!(
            decode_drc_huffman(&mut BitReader::new(&[0]), &[]),
            Err(DrcError::InvalidHuffmanIndex(0))
        );
        assert_eq!(
            decode_drc_huffman(&mut BitReader::new(&[0]), &[[-1, -2]]).unwrap(),
            63
        );
    }

    #[test]
    fn initial_gain_decoder_covers_every_profile_and_sign() {
        for (negative, expected) in [(false, 1.0), (true, -1.0)] {
            let mut writer = BitWriter::new();
            writer.write_bool(negative);
            writer.write(8, 8);
            assert_eq!(
                decode_initial_gain(
                    &mut BitReader::new(&writer.finish()),
                    GainCodingProfile::Regular,
                )
                .unwrap(),
                expected
            );
        }

        for (profile, width) in [
            (GainCodingProfile::Fading, 10),
            (GainCodingProfile::Clipping, 8),
        ] {
            let mut absent = BitWriter::new();
            absent.write_bool(false);
            assert_eq!(
                decode_initial_gain(&mut BitReader::new(&absent.finish()), profile).unwrap(),
                0.0
            );

            let mut present = BitWriter::new();
            present.write_bool(true);
            present.write(7, width);
            assert_eq!(
                decode_initial_gain(&mut BitReader::new(&present.finish()), profile).unwrap(),
                -1.0
            );
        }

        assert_eq!(
            decode_initial_gain(&mut BitReader::new(&[]), GainCodingProfile::Constant,).unwrap(),
            0.0
        );
    }

    #[test]
    fn complex_gain_sequence_covers_non_frame_end_and_clipping_delta_table() {
        fn huffman_code(table: &[[i8; 2]], symbol: i8) -> Vec<bool> {
            fn search(table: &[[i8; 2]], node: i8, target: i8, bits: &mut Vec<bool>) -> bool {
                if node < 0 {
                    return node + 64 == target;
                }
                for (bit, &next) in table[node as usize].iter().enumerate() {
                    bits.push(bit != 0);
                    if search(table, next, target, bits) {
                        return true;
                    }
                    bits.pop();
                }
                false
            }
            let mut bits = Vec::new();
            assert!(search(table, 0, symbol, &mut bits));
            bits
        }

        for profile in [GainCodingProfile::Regular, GainCodingProfile::Clipping] {
            for frame_end in [false, true] {
                let gain_set = GainSet {
                    coding_profile: profile,
                    interpolation_type: GainInterpolationType::Linear,
                    full_frame: false,
                    time_alignment: true,
                    time_delta_min: Some(32),
                    drc_band_type: false,
                    bands: Vec::new(),
                };
                let mut writer = BitWriter::new();
                writer.write_bool(false);
                writer.write_bool(true); // two nodes
                writer.write_bool(frame_end);
                writer.write(0, 2);
                if !frame_end {
                    writer.write(0, 2);
                }
                writer.write_bool(false); // positive/absent initial gain
                if profile == GainCodingProfile::Regular {
                    writer.write(0, 8);
                }
                let table = if profile == GainCodingProfile::Clipping {
                    &DELTA_GAIN_PROFILE_2_HUFFMAN[..]
                } else {
                    &DELTA_GAIN_PROFILE_0_1_HUFFMAN[..]
                };
                for bit in huffman_code(table, 0) {
                    writer.write_bool(bit);
                }
                let sequence = decode_complex_gain_sequence(
                    &mut BitReader::new(&writer.finish()),
                    &gain_set,
                    1024,
                    32,
                )
                .unwrap();
                assert_eq!(sequence.nodes.len(), 2);
                assert_eq!(sequence.nodes[0].time, 15);
                assert_eq!(sequence.nodes[1].time, if frame_end { 1007 } else { 47 });
                assert_eq!(sequence.nodes[0].gain_db, 0.0);
                assert_eq!(sequence.nodes[1].gain_db, 0.0);
            }
        }

        let gain_set = GainSet {
            coding_profile: GainCodingProfile::Regular,
            interpolation_type: GainInterpolationType::Linear,
            full_frame: false,
            time_alignment: false,
            time_delta_min: Some(32),
            drc_band_type: false,
            bands: Vec::new(),
        };
        let mut writer = BitWriter::new();
        writer.write_bool(true); // one node
        writer.write_bool(false); // frame does not end at the node
        writer.write(1, 2); // extended time-delta class
        writer.write(1, 2); // delta 3: time 95 for a 64-sample frame
        writer.write_bool(false); // positive initial gain
        writer.write(0, 8);
        let sequence =
            decode_complex_gain_sequence(&mut BitReader::new(&writer.finish()), &gain_set, 64, 32)
                .unwrap();
        assert_eq!(sequence.nodes[0].time, -33);

        // C's _decodeTimes inserts the frame-end node in front of overflowing
        // node-reservoir times, then _readDrcGainSequence moves those overflow
        // times to the start of the following frame's node list.
        let gain_set = GainSet {
            coding_profile: GainCodingProfile::Regular,
            interpolation_type: GainInterpolationType::Linear,
            full_frame: true,
            time_alignment: false,
            time_delta_min: Some(32),
            drc_band_type: false,
            bands: Vec::new(),
        };
        let mut writer = BitWriter::new();
        writer.write_bool(false);
        writer.write_bool(false);
        writer.write_bool(true); // three nodes
        writer.write(1, 2);
        writer.write(1, 2); // delta 3: first time 95, beyond frame end 63
        writer.write(0, 2); // delta 1: second reservoir time 127
        writer.write_bool(false);
        writer.write(0, 8); // zero initial gain
        for _ in 0..2 {
            for bit in huffman_code(&DELTA_GAIN_PROFILE_0_1_HUFFMAN, 0) {
                writer.write_bool(bit);
            }
        }
        let sequence =
            decode_complex_gain_sequence(&mut BitReader::new(&writer.finish()), &gain_set, 64, 32)
                .unwrap();
        assert_eq!(
            sequence
                .nodes
                .iter()
                .map(|node| node.time)
                .collect::<Vec<_>>(),
            [-33, -1, 63]
        );
    }

    #[test]
    fn gain_payload_covers_constant_sequences_alignment_and_implicit_extension_absence() {
        let gain_set = |sequence_index, profile, time_alignment| GainSet {
            coding_profile: profile,
            interpolation_type: GainInterpolationType::Linear,
            full_frame: true,
            time_alignment,
            time_delta_min: Some(32),
            drc_band_type: false,
            bands: vec![GainBand {
                sequence_index,
                cicp_characteristic_index: None,
                characteristic: None,
                border: None,
            }],
        };
        let coefficients = DrcCoefficients {
            drc_location: 1,
            drc_frame_size: Some(1024),
            gain_sequence_count: 2,
            gain_sets: vec![
                gain_set(0, GainCodingProfile::Constant, false),
                gain_set(1, GainCodingProfile::Regular, true),
            ],
            custom_characteristics_left: Vec::new(),
            custom_characteristics_right: Vec::new(),
            shape_filters: Vec::new(),
        };
        let mut writer = BitWriter::new();
        writer.write_bool(false); // regular sequence simple mode
        writer.write_bool(false); // positive gain
        writer.write(0, 8);
        writer.write_bool(false); // no extension
        let gain = coefficients
            .parse_gain_payload(&writer.finish(), 1024, 32)
            .unwrap();
        assert_eq!(gain.sequences[0].nodes[0].time, 1023);
        assert_eq!(gain.sequences[0].nodes[0].gain_db, 0.0);
        assert_eq!(gain.sequences[1].nodes[0].time, 1007);

        let gain_sets = (0..13)
            .map(|index| gain_set(index, GainCodingProfile::Constant, false))
            .collect();
        let coefficients = DrcCoefficients {
            drc_location: 1,
            drc_frame_size: Some(1024),
            gain_sequence_count: 13,
            gain_sets,
            custom_characteristics_left: Vec::new(),
            custom_characteristics_right: Vec::new(),
            shape_filters: Vec::new(),
        };
        let gain = coefficients.parse_gain_payload(&[], 1024, 32).unwrap();
        assert_eq!(gain.sequences.len(), 12);
        assert!(!gain.extension_present);
        assert_eq!(gain.bits_read, 0);

        let coefficients = DrcCoefficients {
            drc_location: 1,
            drc_frame_size: Some(1024),
            gain_sequence_count: 1,
            gain_sets: vec![gain_set(0, GainCodingProfile::Regular, false)],
            custom_characteristics_left: Vec::new(),
            custom_characteristics_right: Vec::new(),
            shape_filters: Vec::new(),
        };
        assert!(matches!(
            coefficients.parse_gain_payload(&[0x80], 1024, 32),
            Err(DrcError::Bit(BitError::UnexpectedEof { .. }))
        ));
    }

    #[test]
    fn truncated_loudness_measurement_value_is_rejected() {
        let mut writer = BitWriter::new();
        writer.write(1, 6); // drcSetId
        writer.write(0, 7); // downmixId
        writer.write_bool(false); // no sample peak
        writer.write_bool(false); // no true peak
        writer.write(1, 4); // one measurement
        writer.write(LoudnessMethod::ProgramLoudness as u32, 4);
        writer.write(0, 7); // one bit short of the eight-bit method value
        let bit_len = writer.bits_written();
        let bytes = writer.finish();
        let mut reader = BitReader::with_bit_len(&bytes, bit_len).unwrap();
        assert!(matches!(
            read_loudness_info_v0(&mut reader),
            Err(DrcError::Bit(BitError::UnexpectedEof {
                needed_bits: 8,
                remaining_bits: 7,
            }))
        ));
    }

    #[test]
    fn loudness_methods_decode_all_piecewise_value_ranges() {
        let methods = [
            LoudnessMethod::UnknownOther,
            LoudnessMethod::ProgramLoudness,
            LoudnessMethod::AnchorLoudness,
            LoudnessMethod::MaximumLoudnessRange,
            LoudnessMethod::MomentaryLoudnessMax,
            LoudnessMethod::ShortTermLoudnessMax,
            LoudnessMethod::LoudnessRange,
            LoudnessMethod::MixingLevel,
            LoudnessMethod::RoomType,
            LoudnessMethod::ShortTermLoudness,
        ];
        for (bits, method) in methods.into_iter().enumerate() {
            assert_eq!(LoudnessMethod::from_bits(bits as u8).unwrap(), method);
            assert!(decode_method_value(method, 10).is_finite());
        }
        assert_eq!(
            LoudnessMethod::from_bits(10),
            Err(DrcError::InvalidLoudnessMethod(10))
        );
        assert_eq!(decode_method_value(LoudnessMethod::LoudnessRange, 0), 0.0);
        assert_eq!(
            decode_method_value(LoudnessMethod::LoudnessRange, 128),
            32.0
        );
        assert_eq!(
            decode_method_value(LoudnessMethod::LoudnessRange, 129),
            32.5
        );
        assert_eq!(
            decode_method_value(LoudnessMethod::LoudnessRange, 205),
            71.0
        );
        assert_eq!(decode_method_value(LoudnessMethod::MixingLevel, 5), 85.0);
        assert_eq!(decode_method_value(LoudnessMethod::RoomType, 2), 2.0);
        assert_eq!(
            decode_method_value(LoudnessMethod::ShortTermLoudness, 10),
            -111.0
        );
    }

    fn loudness_info(drc_set_id: u8, downmix_id: u8, value: Option<f32>) -> LoudnessInfo {
        LoudnessInfo {
            drc_set_id,
            downmix_id,
            sample_peak_level: None,
            true_peak_level: None,
            true_peak_measurement_system: None,
            true_peak_reliability: None,
            measurements: value
                .map(|value| {
                    vec![LoudnessMeasurement {
                        method: LoudnessMethod::ProgramLoudness,
                        value,
                        measurement_system: 0,
                        reliability: 0,
                    }]
                })
                .unwrap_or_default(),
        }
    }

    #[test]
    fn loudness_selection_uses_track_album_and_any_downmix_fallback() {
        let set = LoudnessInfoSet {
            album: vec![loudness_info(1, 2, Some(-24.0))],
            track: vec![
                loudness_info(1, 2, None),
                loudness_info(1, 0x7f, Some(-26.0)),
            ],
            extension_present: false,
            bits_read: 0,
        };
        assert_eq!(set.album[0].program_loudness(), Some(-24.0));
        assert_eq!(set.track[0].program_loudness(), None);
        assert_eq!(set.select_program_loudness(1, 2, true), Some(-24.0));
        assert_eq!(set.select_program_loudness(1, 2, false), Some(-26.0));
        assert_eq!(set.select_program_loudness(2, 2, false), None);
    }

    #[test]
    fn peak_and_global_gain_helpers_cover_absent_zero_and_saturation() {
        assert_eq!(read_peak_level(&mut BitReader::new(&[0])).unwrap(), None);
        let mut writer = BitWriter::new();
        writer.write_bool(true);
        writer.write(0, 12);
        assert_eq!(
            read_peak_level(&mut BitReader::new(&writer.finish())).unwrap(),
            None
        );
        let mut writer = BitWriter::new();
        writer.write_bool(true);
        writer.write(32, 12);
        assert_eq!(
            read_peak_level(&mut BitReader::new(&writer.finish())).unwrap(),
            Some(19.0)
        );

        assert_eq!(loudness_normalization_gain_db(-23.0, -26.0), 3.0);
        let mut floats = [1.0, -1.0];
        apply_gain_f32(&mut floats, 20.0);
        assert_eq!(floats, [10.0, -10.0]);
        let mut integers = [10_000, -10_000];
        apply_gain_i16(&mut integers, 20.0);
        assert_eq!(integers, [i16::MAX, i16::MIN]);
    }

    #[test]
    fn downmix_parser_and_application_reject_all_invalid_layouts() {
        let mut writer = BitWriter::new();
        writer.write(1, 7);
        writer.write(9, 7);
        writer.write(0, 8);
        assert!(matches!(
            DownmixInstructions::parse_v0(&writer.finish(), 2),
            Err(DrcError::InvalidDownmixDimensions { .. })
        ));
        let mut writer = BitWriter::new();
        writer.write(1, 7);
        writer.write(1, 7);
        writer.write(0, 8);
        assert!(matches!(
            DownmixInstructions::parse_v1(&writer.finish(), 9),
            Err(DrcError::InvalidDownmixDimensions { .. })
        ));

        let absent = DownmixInstructions {
            downmix_id: 1,
            target_channel_count: 1,
            target_layout: 0,
            coefficient_offset: 0,
            coefficients: None,
        };
        assert_eq!(
            absent.apply_interleaved_f32(&[1.0], 1),
            Err(DrcError::DownmixCoefficientsAbsent)
        );
        let matrix = DownmixInstructions {
            coefficients: Some(vec![vec![10.0, 10.0]]),
            ..absent
        };
        for (input, channels) in [(&[1.0][..], 0), (&[1.0][..], 2), (&[1.0, 2.0][..], 1)] {
            assert_eq!(
                matrix.apply_interleaved_f32(input, channels),
                Err(DrcError::DownmixLayoutMismatch)
            );
        }
        assert_eq!(
            matrix.apply_interleaved_i16(&[10_000, 10_000], 2).unwrap(),
            [i16::MAX]
        );
    }

    #[test]
    fn eq_subband_spline_and_td_cascade_skip_every_optional_field() {
        for slope_selector in 0..=3 {
            let mut writer = BitWriter::new();
            writer.write(0, 5); // two spline nodes
            writer.write_bool(false);
            writer.write(0xa, 4);
            writer.write_bool(true);
            writer.write(0xb, 4); // frequency delta
            writer.write(slope_selector, 2);
            writer.write(
                0,
                if slope_selector == 0 {
                    5
                } else if slope_selector < 3 {
                    4
                } else {
                    3
                },
            );
            writer.write(0x1f, 5); // gain delta
            let bits = writer.bits_written();
            let bytes = writer.finish();
            let mut reader = BitReader::with_bit_len(&bytes, bits).unwrap();
            skip_eq_subband_gain_spline(&mut reader).unwrap();
            assert_eq!(reader.bits_read(), bits);
        }

        let mut writer = BitWriter::new();
        writer.write_bool(true); // group 0 has cascade gain
        writer.write(0, 10);
        writer.write(2, 4);
        writer.write(0, 14);
        writer.write_bool(false); // group 1 has no cascade gain
        writer.write(1, 4);
        writer.write(0, 7);
        writer.write_bool(true); // phase alignment present
        writer.write_bool(false); // one pair for two groups
        let bits = writer.bits_written();
        let bytes = writer.finish();
        let mut reader = BitReader::with_bit_len(&bytes, bits).unwrap();
        skip_td_filter_cascade(&mut reader, 2).unwrap();
        assert_eq!(reader.bits_read(), bits);

        let mut coefficients = BitWriter::new();
        coefficients.write_bool(false); // no eqDelayMax
        coefficients.write(1, 6); // one filter block
        coefficients.write(1, 6); // one filter element
        coefficients.write(0, 6); // element index
        coefficients.write_bool(false); // no element gain
        coefficients.write(0, 6); // no TD elements
        coefficients.write(0, 6); // no subband gains
        let bits = coefficients.bits_written();
        let bytes = coefficients.finish();
        let mut reader = BitReader::with_bit_len(&bytes, bits).unwrap();
        skip_eq_coefficients(&mut reader).unwrap();
        assert_eq!(reader.bits_read(), bits);

        let mut cascade = BitWriter::new();
        for _ in 0..2 {
            cascade.write_bool(false); // no cascade gain
            cascade.write(0, 4); // no filter blocks
        }
        cascade.write_bool(false); // no phase alignment
        let bits = cascade.bits_written();
        let bytes = cascade.finish();
        let mut reader = BitReader::with_bit_len(&bytes, bits).unwrap();
        skip_td_filter_cascade(&mut reader, 2).unwrap();
        assert_eq!(reader.bits_read(), bits);
    }

    #[test]
    fn eq_instruction_skip_covers_downmix_groups_and_validation() {
        let layout = ChannelLayout {
            base_channel_count: 6,
            defined_layout: None,
            speaker_positions: Vec::new(),
        };
        let downmixes = [DownmixInstructions {
            downmix_id: 5,
            target_channel_count: 2,
            target_layout: 0,
            coefficient_offset: 0,
            coefficients: None,
        }];

        let mut writer = BitWriter::new();
        writer.write(0, 10);
        writer.write_bool(true); // downmix ID present
        writer.write(5, 7);
        writer.write_bool(true); // apply to downmix
        writer.write_bool(false); // no additional downmix IDs
        writer.write(0, 6); // drcSetId
        writer.write_bool(true); // additional drcSetIds
        writer.write(1, 6);
        writer.write(0, 6);
        writer.write(0, 16); // purpose
        writer.write_bool(true);
        writer.write(0, 6);
        writer.write(1, 7); // channel groups
        writer.write(2, 7);
        writer.write_bool(true); // TD cascade
        writer.write_bool(true); // group 0 cascade gain
        writer.write(0, 10);
        writer.write(1, 4);
        writer.write(0, 7);
        writer.write_bool(false); // group 1 cascade gain
        writer.write(0, 4);
        writer.write_bool(true); // phase alignment
        writer.write_bool(false);
        writer.write_bool(true); // subband gains
        writer.write(0, 12);
        writer.write_bool(true); // transition duration
        writer.write(0, 5);
        let bits = writer.bits_written();
        let bytes = writer.finish();
        let mut reader = BitReader::with_bit_len(&bytes, bits).unwrap();
        skip_eq_instruction(&mut reader, &layout, &downmixes).unwrap();
        assert_eq!(reader.bits_read(), bits);

        let mut writer = BitWriter::new();
        writer.write(0, 10);
        writer.write_bool(true);
        writer.write(0x7f, 7);
        writer.write_bool(false);
        writer.write_bool(true);
        writer.write(2, 7);
        writer.write(1, 7);
        writer.write(2, 7);
        writer.write(0, 6);
        writer.write_bool(false);
        writer.write(0, 16);
        writer.write_bool(false);
        writer.write_bool(false);
        writer.write(3, 7);
        writer.write_bool(false);
        writer.write_bool(false);
        writer.write_bool(false);
        let bits = writer.bits_written();
        let bytes = writer.finish();
        let mut reader = BitReader::with_bit_len(&bytes, bits).unwrap();
        skip_eq_instruction(&mut reader, &layout, &downmixes).unwrap();
        assert_eq!(reader.bits_read(), bits);

        let write_lookup_prefix = |writer: &mut BitWriter, downmix_id| {
            writer.write(0, 10);
            writer.write_bool(true);
            writer.write(downmix_id, 7);
            writer.write_bool(true);
            writer.write_bool(false);
            writer.write(0, 6);
            writer.write_bool(false);
            writer.write(0, 16);
            writer.write_bool(false);
            writer.write_bool(false);
        };
        let mut missing = BitWriter::new();
        write_lookup_prefix(&mut missing, 6);
        assert_eq!(
            skip_eq_instruction(&mut BitReader::new(&missing.finish()), &layout, &downmixes,),
            Err(DrcError::MissingDownmixInstruction(6))
        );

        let mut too_many = BitWriter::new();
        too_many.write(0, 10);
        too_many.write_bool(false);
        too_many.write(0, 6);
        too_many.write_bool(false);
        too_many.write(0, 16);
        too_many.write_bool(false);
        too_many.write_bool(false);
        assert_eq!(
            skip_eq_instruction(
                &mut BitReader::new(&too_many.finish()),
                &ChannelLayout {
                    base_channel_count: 9,
                    defined_layout: None,
                    speaker_positions: Vec::new(),
                },
                &[],
            ),
            Err(DrcError::TooManyChannels(9))
        );
    }

    #[test]
    fn eq_fixed_subband_gain_covers_every_count_selector() {
        fn zeros(writer: &mut BitWriter, mut count: usize) {
            while count != 0 {
                let chunk = count.min(32);
                writer.write(0, chunk);
                count -= chunk;
            }
        }

        for (selector, gain_count) in [(2, 39), (3, 64), (4, 71), (5, 128), (6, 135), (7, 1)] {
            let mut writer = BitWriter::new();
            writer.write_bool(false); // no eqDelayMax
            writer.write(0, 6); // no filter blocks
            writer.write(0, 6); // no TD elements
            writer.write(1, 6); // one subband gain set
            writer.write_bool(false); // fixed representation
            writer.write(selector, 4);
            if selector == 7 {
                writer.write(0, 8);
            }
            zeros(&mut writer, gain_count * 9);
            let bits = writer.bits_written();
            let bytes = writer.finish();
            let mut reader = BitReader::with_bit_len(&bytes, bits).unwrap();
            skip_eq_coefficients(&mut reader).unwrap();
            assert_eq!(reader.bits_read(), bits);
        }
    }

    #[test]
    fn loud_eq_skip_consumes_all_lists_and_characteristic_forms() {
        let mut writer = BitWriter::new();
        writer.write(3, 4); // loudEqSetId
        writer.write(2, 4); // drcLocation
        writer.write_bool(true);
        writer.write(4, 7);
        writer.write_bool(true);
        writer.write(2, 7);
        writer.write(5, 7);
        writer.write(6, 7);
        writer.write_bool(true);
        writer.write(7, 6);
        writer.write_bool(true);
        writer.write(2, 6);
        writer.write(8, 6);
        writer.write(9, 6);
        writer.write_bool(true);
        writer.write(10, 6);
        writer.write_bool(true);
        writer.write(1, 6);
        writer.write(11, 6);
        writer.write_bool(true);
        writer.write_bool(false);
        writer.write(2, 6);
        writer.write(12, 6);
        writer.write_bool(true);
        writer.write(13, 7); // CICP characteristic
        writer.write(14, 6);
        writer.write(5, 3);
        writer.write(16, 5);
        writer.write(15, 6);
        writer.write_bool(false);
        writer.write(1, 4); // custom left
        writer.write(2, 4); // custom right
        writer.write(16, 6);
        writer.write(6, 3);
        writer.write(17, 5);
        let bits = writer.bits_written();
        let bytes = writer.finish();
        let mut reader = BitReader::with_bit_len(&bytes, bits).unwrap();
        skip_loud_eq_instruction(&mut reader).unwrap();
        assert_eq!(reader.bits_read(), bits);

        let mut minimal = BitWriter::new();
        minimal.write(0, 4); // loudEqSetId
        minimal.write(0, 4); // drcLocation
        minimal.write_bool(true);
        minimal.write(0, 7); // downmixId
        minimal.write_bool(false); // no additional downmix IDs
        minimal.write_bool(true);
        minimal.write(0, 6); // drcSetId
        minimal.write_bool(false); // no additional DRC set IDs
        minimal.write_bool(true);
        minimal.write(0, 6); // eqSetId
        minimal.write_bool(false); // no additional EQ set IDs
        minimal.write_bool(false); // loudnessAfterDrc
        minimal.write_bool(false); // loudnessAfterEq
        minimal.write(0, 6); // no gain sequences
        let bits = minimal.bits_written();
        let bytes = minimal.finish();
        let mut reader = BitReader::with_bit_len(&bytes, bits).unwrap();
        skip_loud_eq_instruction(&mut reader).unwrap();
        assert_eq!(reader.bits_read(), bits);

        let mut layout = BitWriter::new();
        layout.write(2, 7); // base channels
        layout.write_bool(true);
        layout.write(2, 8); // predefined layout, no explicit positions
        let parsed = read_channel_layout(&mut BitReader::new(&layout.finish())).unwrap();
        assert_eq!(parsed.defined_layout, Some(2));
        assert!(parsed.speaker_positions.is_empty());
    }
}
