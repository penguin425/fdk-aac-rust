//! FDK-compatible AAC encoder metadata setup and payload serialization.

use crate::bits::BitWriter;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u16)]
pub enum MetadataDrcProfile {
    #[default]
    None = 0,
    FilmStandard = 1,
    FilmLight = 2,
    MusicStandard = 3,
    MusicLight = 4,
    Speech = 5,
    NotPresent = 256,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtendedEncoderMetadata {
    pub enabled: bool,
    pub downmix_levels_enabled: bool,
    pub downmix_level_a: u8,
    pub downmix_level_b: u8,
    pub downmix_gains_enabled: bool,
    pub downmix_gain_5_q16: i32,
    pub downmix_gain_2_q16: i32,
    pub lfe_downmix_enabled: bool,
    pub lfe_downmix_level: u8,
}

impl Default for ExtendedEncoderMetadata {
    fn default() -> Self {
        Self {
            enabled: false,
            downmix_levels_enabled: false,
            downmix_level_a: 0,
            downmix_level_b: 0,
            downmix_gains_enabled: false,
            downmix_gain_5_q16: 0,
            downmix_gain_2_q16: 0,
            lfe_downmix_enabled: false,
            lfe_downmix_level: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncoderMetadata {
    pub drc_profile: MetadataDrcProfile,
    pub compression_profile: MetadataDrcProfile,
    pub drc_target_reference_level_q16: i32,
    pub compression_target_reference_level_q16: i32,
    pub program_reference_level: Option<i32>,
    pub pce_mixdown_index_present: bool,
    pub etsi_downmix_levels_present: bool,
    pub center_mix_level: u8,
    pub surround_mix_level: u8,
    pub dolby_surround_mode: u8,
    pub drc_presentation_mode: u8,
    pub extended: ExtendedEncoderMetadata,
}

impl Default for EncoderMetadata {
    fn default() -> Self {
        Self {
            drc_profile: MetadataDrcProfile::None,
            compression_profile: MetadataDrcProfile::NotPresent,
            drc_target_reference_level_q16: -(31 << 16),
            compression_target_reference_level_q16: -(23 << 16),
            program_reference_level: None,
            pce_mixdown_index_present: false,
            etsi_downmix_levels_present: false,
            center_mix_level: 0,
            surround_mix_level: 0,
            dolby_surround_mode: 0,
            drc_presentation_mode: 0,
            extended: ExtendedEncoderMetadata::default(),
        }
    }
}

impl EncoderMetadata {
    /// Serialize MPEG-4 `dynamic_range_info()` without its four-bit extension
    /// type. `gain_q16` uses FDK's signed Q16 dB representation.
    pub fn dynamic_range_payload(&self, gain_q16: i32) -> (Vec<u8>, usize) {
        let mut writer = BitWriter::new();
        writer.write_bool(false); // pce_tag_present
        writer.write_bool(false); // excluded_chns_present
        writer.write_bool(false); // drc_bands_present
        writer.write_bool(self.program_reference_level.is_some());
        if let Some(level) = self.program_reference_level {
            writer.write(dialnorm_to_program_reference_level(level) as u32, 7);
            writer.write_bool(false); // reserved
        }
        let (sign, control) = encode_dynamic_range_gain(gain_q16);
        writer.write_bool(sign);
        writer.write(control as u32, 7);
        let bits = writer.bits_written();
        (writer.finish(), bits)
    }

    /// Serialize the ETSI TS 101 154 `ancillary_data()` payload carried in an
    /// AAC data element. `compression_gain_q16` is omitted when heavy
    /// compression is not signalled.
    pub fn etsi_ancillary_payload(&self, compression_gain_q16: Option<i32>) -> Vec<u8> {
        let mut writer = BitWriter::new();
        writer.write(0xbc, 8);
        writer.write(3, 2); // MPEG audio type
        writer.write((self.dolby_surround_mode & 3) as u32, 2);
        writer.write((self.drc_presentation_mode & 3) as u32, 2);
        writer.write(0, 2); // stereo_downmix_mode + reserved

        let compression = compression_gain_q16
            .filter(|_| self.compression_profile != MetadataDrcProfile::NotPresent);
        writer.write(0, 3); // reserved
        writer.write_bool(self.etsi_downmix_levels_present);
        writer.write_bool(self.extended.enabled);
        writer.write_bool(compression.is_some());
        writer.write(0, 2); // coarse/fine timecode absent
        if self.etsi_downmix_levels_present {
            writer.write(
                0x80 | ((self.center_mix_level & 7) << 4) as u32
                    | 0x08
                    | u32::from(self.surround_mix_level & 7),
                8,
            );
        }
        if let Some(gain) = compression {
            writer.write(1, 8); // audio_coding_mode
            writer.write(encode_compression_gain(gain) as u32, 8);
        }
        if self.extended.enabled {
            writer.write_bool(false); // reserved
            writer.write_bool(self.extended.downmix_levels_enabled);
            writer.write_bool(self.extended.downmix_gains_enabled);
            writer.write_bool(self.extended.lfe_downmix_enabled);
            writer.write(0, 4);
            if self.extended.downmix_levels_enabled {
                writer.write((self.extended.downmix_level_a & 7) as u32, 3);
                writer.write((self.extended.downmix_level_b & 7) as u32, 3);
                writer.write(0, 2);
            }
            if self.extended.downmix_gains_enabled {
                for gain in [
                    self.extended.downmix_gain_5_q16,
                    self.extended.downmix_gain_2_q16,
                ] {
                    let (sign, index) = encode_dynamic_range_gain(gain);
                    writer.write_bool(sign);
                    writer.write(index as u32, 6);
                    writer.write_bool(false);
                }
            }
            if self.extended.lfe_downmix_enabled {
                writer.write((self.extended.lfe_downmix_level & 15) as u32, 4);
                writer.write(0, 4);
            }
        }
        writer.byte_align();
        writer.finish()
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct WeightingState {
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

/// Stateful profile compressor and overload limiter used by FDK's metadata
/// generator. Samples use the encoder's signed-16-bit PCM scale even though
/// they are supplied as `f32`.
#[derive(Debug, Clone)]
pub struct MetadataCompressor {
    sample_rate: u32,
    frame_length: usize,
    channels: usize,
    profiles: [MetadataDrcProfile; 2],
    weighting: Vec<WeightingState>,
    smooth_level: [f64; 2],
    smooth_gain: [f64; 2],
    hold_count: [usize; 2],
    limiter_gain: [f64; 2],
    previous_peak: [f64; 2],
}

impl MetadataCompressor {
    pub fn new(sample_rate: u32, frame_length: usize, channels: usize) -> Self {
        Self {
            sample_rate,
            frame_length,
            channels,
            profiles: [MetadataDrcProfile::None; 2],
            weighting: vec![WeightingState::default(); channels],
            smooth_level: [-135.0; 2],
            smooth_gain: [0.0; 2],
            hold_count: [0; 2],
            limiter_gain: [0.0; 2],
            previous_peak: [0.0; 2],
        }
    }

    /// Calculate MPEG line-mode and ETSI RF-mode gains in signed Q16 dB.
    pub fn process(&mut self, input: &[f32], metadata: &EncoderMetadata) -> (i32, i32) {
        debug_assert_eq!(input.len(), self.frame_length * self.channels);
        let profiles = [metadata.drc_profile, metadata.compression_profile];
        if profiles != self.profiles {
            self.profiles = profiles;
            self.smooth_gain = [0.0; 2];
        }

        let dialnorm_q16 = metadata.program_reference_level.unwrap_or(-(23 << 16));
        let dialnorm = f64::from(dialnorm_q16) / 65536.0;
        let mut drc_target = f64::from(metadata.drc_target_reference_level_q16) / 65536.0;
        let mut compression_target =
            f64::from(metadata.compression_target_reference_level_q16) / 65536.0;
        match metadata.drc_presentation_mode {
            1 => {
                drc_target = drc_target.max(-31.0);
                compression_target = compression_target.max(-20.0);
            }
            2 => {
                drc_target = drc_target.max(-23.0);
                compression_target = compression_target.max(-23.0);
            }
            _ => {}
        }
        if metadata.compression_profile == MetadataDrcProfile::NotPresent
            && metadata.drc_presentation_mode != 0
        {
            drc_target = drc_target.max(compression_target);
        }

        if profiles
            .iter()
            .any(|profile| *profile != MetadataDrcProfile::None)
        {
            let energy = self.weighted_energy(input);
            let mut level = 10.0 * energy.max(1.0e-10).log10() + 3.0;
            level -= dialnorm + 31.0;
            for output in 0..2 {
                self.update_profile_gain(output, level);
            }
        } else {
            self.smooth_gain = [0.0; 2];
        }

        let mut peak = self.peak_levels(input);
        for output in 0..2 {
            let previous = self.previous_peak[output];
            self.previous_peak[output] = peak[output];
            peak[output] = peak[output].max(previous);
            peak[output] = 20.0 * peak[output].max(1.0e-6).log10() + 0.5 + self.smooth_gain[output];
        }
        peak[0] -= dialnorm - drc_target;
        peak[1] -= dialnorm - compression_target;

        let limiter_decay = 0.006 * self.frame_length as f64 / 256.0;
        self.limiter_gain[0] = (self.limiter_gain[0] + limiter_decay).min(-peak[0]);
        self.limiter_gain[1] = (self.limiter_gain[1] + 2.0 * limiter_decay).min(-peak[1]);

        let gains = [0, 1].map(|output| {
            let limiter = self.limiter_gain[output].min(0.0);
            ((self.smooth_gain[output] + limiter) * 65536.0).round() as i32
        });
        (gains[0], gains[1])
    }

    fn weighted_energy(&mut self, input: &[f32]) -> f64 {
        const B0: f64 = 0.530_506_62;
        const A1: f64 = -0.952_379_83;
        const A2: f64 = -0.022_488_36;
        let mut energy = 0.0;
        for channel in 0..self.channels {
            let state = &mut self.weighting[channel];
            for frame in input.chunks_exact(self.channels) {
                let x = f64::from(frame[channel]) / 32768.0;
                let y = B0 * (x - state.x2) - A1 * state.y1 - A2 * state.y2;
                state.x2 = state.x1;
                state.x1 = x;
                state.y2 = state.y1;
                state.y1 = y;
                energy += y * y;
            }
        }
        energy / self.frame_length as f64
    }

    fn peak_levels(&self, input: &[f32]) -> [f64; 2] {
        let line = input
            .iter()
            .map(|&sample| (f64::from(sample) / 32768.0).abs())
            .fold(0.0f64, f64::max);
        let mut rf = line;
        // FDK's level filter consumes planar channel blocks.  findPeakLevels
        // subsequently walks that same buffer with `i * channels` indexing
        // when it forms the RF mono downmix.  Preserve this observable layout
        // behavior: for stereo it sums adjacent time samples in each planar
        // half rather than the original interleaved L/R pair.
        for sample in 0..self.frame_length {
            let mono = (0..self.channels)
                .map(|offset| {
                    let planar_index = sample * self.channels + offset;
                    let channel = planar_index / self.frame_length;
                    let channel_sample = planar_index % self.frame_length;
                    f64::from(input[channel_sample * self.channels + channel]) / 32768.0
                })
                .sum::<f64>();
            rf = rf.max(mono.abs());
        }
        [line, rf]
    }

    fn update_profile_gain(&mut self, output: usize, level: f64) {
        let profile = self.profiles[output];
        if profile == MetadataDrcProfile::None {
            self.smooth_gain[output] = 0.0;
            return;
        }
        let params = ProfileParameters::for_profile(profile);
        let max_early_cut =
            -(params.cut_threshold - params.early_cut_threshold) * params.early_cut_factor;
        let gain = if level <= params.max_boost_threshold {
            params.max_boost
        } else if level < params.boost_threshold {
            (level - params.boost_threshold) * params.boost_factor
        } else if level <= params.early_cut_threshold {
            0.0
        } else if level <= params.cut_threshold {
            (level - params.early_cut_threshold) * params.early_cut_factor
        } else if level < params.max_cut_threshold {
            (level - params.cut_threshold) * params.cut_factor - max_early_cut
        } else {
            -params.max_cut
        };

        let level_delta = level - self.smooth_level[output];
        let time_constant = if gain < self.smooth_gain[output] {
            if level_delta > params.attack_threshold {
                params.fast_attack
            } else {
                params.slow_attack
            }
        } else if level_delta < -params.decay_threshold {
            params.fast_decay
        } else {
            params.slow_decay
        };
        let alpha =
            time_constant_to_coefficient(time_constant, self.sample_rate, self.frame_length);
        if gain < self.smooth_gain[output] || self.hold_count[output] == 0 {
            self.smooth_level[output] += alpha * (level - self.smooth_level[output]);
            self.smooth_gain[output] += alpha * (gain - self.smooth_gain[output]);
        }
        if self.hold_count[output] != 0 {
            self.hold_count[output] -= 1;
        }
        if gain < self.smooth_gain[output] {
            self.hold_count[output] = params.hold_off * 256 / self.frame_length;
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ProfileParameters {
    max_boost_threshold: f64,
    boost_threshold: f64,
    early_cut_threshold: f64,
    cut_threshold: f64,
    max_cut_threshold: f64,
    boost_factor: f64,
    early_cut_factor: f64,
    cut_factor: f64,
    max_boost: f64,
    max_cut: f64,
    fast_attack: f64,
    fast_decay: f64,
    slow_attack: f64,
    slow_decay: f64,
    hold_off: usize,
    attack_threshold: f64,
    decay_threshold: f64,
}

impl ProfileParameters {
    fn for_profile(profile: MetadataDrcProfile) -> Self {
        let index = match profile {
            MetadataDrcProfile::None
            | MetadataDrcProfile::NotPresent
            | MetadataDrcProfile::FilmStandard => 0,
            MetadataDrcProfile::FilmLight => 1,
            MetadataDrcProfile::MusicStandard => 2,
            MetadataDrcProfile::MusicLight => 3,
            MetadataDrcProfile::Speech => 4,
        };
        const MAX_BOOST_THRESHOLD: [f64; 5] = [-43.0, -53.0, -55.0, -65.0, -50.0];
        const BOOST_THRESHOLD: [f64; 5] = [-31.0, -41.0, -31.0, -41.0, -31.0];
        const EARLY_CUT_THRESHOLD: [f64; 5] = [-26.0, -21.0, -26.0, -21.0, -26.0];
        const CUT_THRESHOLD: [f64; 5] = [-16.0, -11.0, -16.0, -21.0, -16.0];
        const MAX_CUT_THRESHOLD: [f64; 5] = [4.0, 9.0, 4.0, 9.0, 4.0];
        const BOOST_FACTOR: [f64; 5] = [-0.5, -0.5, -0.5, -0.5, -0.8];
        const EARLY_CUT_FACTOR: [f64; 5] = [-0.5, -0.5, -0.5, 0.0, -0.5];
        const CUT_FACTOR: [f64; 5] = [-0.95, -0.95, -0.95, -0.5, -0.95];
        const MAX_BOOST: [f64; 5] = [6.0, 6.0, 12.0, 12.0, 15.0];
        const MAX_CUT: [f64; 5] = [24.0, 24.0, 24.0, 15.0, 24.0];
        const FAST_DECAY: [f64; 5] = [1.0, 1.0, 1.0, 1.0, 0.2];
        const SLOW_DECAY: [f64; 5] = [3.0, 3.0, 10.0, 3.0, 1.0];
        const ATTACK_THRESHOLD: [f64; 5] = [15.0, 15.0, 15.0, 15.0, 10.0];
        const DECAY_THRESHOLD: [f64; 5] = [20.0, 20.0, 20.0, 20.0, 10.0];
        Self {
            max_boost_threshold: MAX_BOOST_THRESHOLD[index],
            boost_threshold: BOOST_THRESHOLD[index],
            early_cut_threshold: EARLY_CUT_THRESHOLD[index],
            cut_threshold: CUT_THRESHOLD[index],
            max_cut_threshold: MAX_CUT_THRESHOLD[index],
            boost_factor: BOOST_FACTOR[index],
            early_cut_factor: EARLY_CUT_FACTOR[index],
            cut_factor: CUT_FACTOR[index],
            max_boost: MAX_BOOST[index],
            max_cut: MAX_CUT[index],
            fast_attack: 0.010,
            fast_decay: FAST_DECAY[index],
            slow_attack: 0.100,
            slow_decay: SLOW_DECAY[index],
            hold_off: 10,
            attack_threshold: ATTACK_THRESHOLD[index],
            decay_threshold: DECAY_THRESHOLD[index],
        }
    }
}

fn time_constant_to_coefficient(seconds: f64, sample_rate: u32, frame_length: usize) -> f64 {
    if seconds <= 0.0 {
        1.0
    } else {
        1.0 - (-(frame_length as f64) / (seconds * f64::from(sample_rate))).exp()
    }
}

fn dialnorm_to_program_reference_level(value_q16: i32) -> u8 {
    ((-value_q16 + (1 << 13)) >> 14).clamp(0, 127) as u8
}

fn encode_dynamic_range_gain(gain_q16: i32) -> (bool, u8) {
    let sign = gain_q16 < 0;
    let magnitude = gain_q16.saturating_abs().min(127 << 14);
    (sign, ((magnitude + (1 << 13)) >> 14) as u8)
}

fn encode_compression_gain(gain_q16: i32) -> u8 {
    let value = ((3_156_476i64 - i64::from(gain_q16)) * 15 + 197_283) / 394_566;
    if value >= 240 {
        0xff
    } else if value < 0 {
        0
    } else {
        let x = value / 15;
        let y = value % 15;
        ((x << 4) | y) as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_fdk_metadata_setup_payloads() {
        let metadata = EncoderMetadata::default();
        assert_eq!(metadata.dynamic_range_payload(0), (vec![0, 0], 12));
        assert_eq!(metadata.etsi_ancillary_payload(None), [0xbc, 0xc0, 0x00]);
    }

    #[test]
    fn serializes_every_etsi_optional_group() {
        let metadata = EncoderMetadata {
            compression_profile: MetadataDrcProfile::FilmStandard,
            program_reference_level: Some(-(23 << 16)),
            etsi_downmix_levels_present: true,
            center_mix_level: 2,
            surround_mix_level: 3,
            dolby_surround_mode: 2,
            drc_presentation_mode: 1,
            extended: ExtendedEncoderMetadata {
                enabled: true,
                downmix_levels_enabled: true,
                downmix_level_a: 1,
                downmix_level_b: 2,
                downmix_gains_enabled: true,
                downmix_gain_5_q16: -(3 << 16),
                downmix_gain_2_q16: 2 << 16,
                lfe_downmix_enabled: true,
                lfe_downmix_level: 7,
            },
            ..EncoderMetadata::default()
        };
        let (dynamic, bits) = metadata.dynamic_range_payload(-(2 << 16));
        assert_eq!(bits, 20);
        assert_eq!(dynamic.len(), 3);
        let etsi = metadata.etsi_ancillary_payload(Some(-(4 << 16)));
        assert!(etsi.len() >= 10);
        assert_eq!(etsi[0], 0xbc);
    }

    #[test]
    fn compressor_is_stateful_and_applies_profile_and_limiter_gains() {
        let metadata = EncoderMetadata {
            drc_profile: MetadataDrcProfile::FilmStandard,
            compression_profile: MetadataDrcProfile::MusicLight,
            ..EncoderMetadata::default()
        };
        let mut compressor = MetadataCompressor::new(48_000, 1024, 1);
        let quiet = vec![32.0; 1024];
        let loud = vec![32_000.0; 1024];
        let quiet_gain = compressor.process(&quiet, &metadata);
        let loud_gain = compressor.process(&loud, &metadata);
        assert!(quiet_gain.0 > 0);
        assert!(loud_gain.0 < quiet_gain.0);
        assert!(loud_gain.1 < quiet_gain.1);
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn compressor_payloads_match_c_profiles_at_control_boundaries() {
        fn c_profile(profile: MetadataDrcProfile) -> i32 {
            match profile {
                MetadataDrcProfile::None => 0,
                MetadataDrcProfile::FilmStandard => 1,
                MetadataDrcProfile::FilmLight => 2,
                MetadataDrcProfile::MusicStandard => 3,
                MetadataDrcProfile::MusicLight => 4,
                MetadataDrcProfile::Speech => 5,
                MetadataDrcProfile::NotPresent => -2,
            }
        }

        for channels in [1usize, 2] {
            for profile in [
                MetadataDrcProfile::FilmStandard,
                MetadataDrcProfile::FilmLight,
                MetadataDrcProfile::MusicStandard,
                MetadataDrcProfile::MusicLight,
                MetadataDrcProfile::Speech,
            ] {
                let frames = 12usize;
                let frame_length = 1024usize;
                let mut pcm = Vec::with_capacity(frames * frame_length * channels);
                for frame in 0..frames {
                    let amplitude = [64.0, 512.0, 4_096.0, 16_000.0, 30_000.0][frame % 5];
                    for sample in 0..frame_length {
                        for channel in 0..channels {
                            let phase = 2.0 * std::f64::consts::PI * 997.0 * sample as f64
                                / 48_000.0
                                + channel as f64 * 0.37;
                            pcm.push((phase.sin() * amplitude) as i16);
                        }
                    }
                }
                let mut c_dynamic = vec![0; frames];
                let mut c_compression = vec![0; frames];
                let result = unsafe {
                    crate::sys::fdk_metadata_compressor_test(
                        pcm.as_ptr(),
                        frames as i32,
                        frame_length as i32,
                        48_000,
                        channels as i32,
                        channels as i32,
                        c_profile(profile),
                        c_profile(profile),
                        -(23 << 16),
                        -(31 << 16),
                        -(23 << 16),
                        c_dynamic.as_mut_ptr(),
                        c_compression.as_mut_ptr(),
                    )
                };
                assert_eq!(result, 0);

                let metadata = EncoderMetadata {
                    drc_profile: profile,
                    compression_profile: profile,
                    ..EncoderMetadata::default()
                };
                let mut rust = MetadataCompressor::new(48_000, frame_length, channels);
                for frame in 0..frames {
                    let start = frame * frame_length * channels;
                    let input = pcm[start..start + frame_length * channels]
                        .iter()
                        .map(|&sample| f32::from(sample))
                        .collect::<Vec<_>>();
                    let (dynamic, compression) = rust.process(&input, &metadata);
                    assert_eq!(
                        encode_dynamic_range_gain(dynamic),
                        encode_dynamic_range_gain(c_dynamic[frame]),
                        "line payload differs for {profile:?}, {channels}ch, frame {frame}"
                    );
                    assert_eq!(
                        encode_compression_gain(compression),
                        encode_compression_gain(c_compression[frame]),
                        "RF payload differs for {profile:?}, {channels}ch, frame {frame}"
                    );
                }
            }
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn default_etsi_payload_matches_c_encoder_output() {
        use crate::decoder::AacLcDecoder;
        use crate::{sys, Encoder};

        let mut encoder = Encoder::open(1).unwrap();
        encoder.set_param(sys::AACENC_AOT, 2).unwrap();
        encoder.set_param(sys::AACENC_CHANNELMODE, 1).unwrap();
        encoder.set_param(sys::AACENC_SAMPLERATE, 48_000).unwrap();
        encoder.set_param(sys::AACENC_BITRATE, 96_000).unwrap();
        encoder.set_param(sys::AACENC_TRANSMUX, 0).unwrap();
        encoder.set_param(sys::AACENC_METADATA_MODE, 3).unwrap();
        encoder.initialize().unwrap();
        let info = encoder.info().unwrap();
        let mut raw = vec![0; info.max_output_bytes as usize];
        let mut bytes = 0;
        for _ in 0..8 {
            bytes = encoder
                .encode_interleaved_i16(&vec![0; info.frame_length as usize], &mut raw)
                .unwrap();
            if bytes != 0 {
                break;
            }
        }
        raw.truncate(bytes);
        let mut decoder = AacLcDecoder::new(3, 1).unwrap();
        decoder.init_ancillary_data(32);
        decoder.decode_raw_data_block_f32(&raw).unwrap();
        assert_eq!(decoder.ancillary_data().len(), 1);
        assert_eq!(
            decoder.ancillary_data()[0].data,
            EncoderMetadata::default().etsi_ancillary_payload(None)
        );
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn extended_etsi_payload_and_ffi_layout_match_c_encoder() {
        use crate::decoder::AacLcDecoder;
        use crate::{sys, Encoder};

        assert_eq!(std::mem::size_of::<sys::AACENC_MetaData>() as i32, unsafe {
            sys::fdk_aacenc_metadata_size_test()
        });
        let rust_metadata = EncoderMetadata {
            etsi_downmix_levels_present: true,
            center_mix_level: 2,
            surround_mix_level: 3,
            dolby_surround_mode: 2,
            drc_presentation_mode: 1,
            extended: ExtendedEncoderMetadata {
                enabled: true,
                downmix_levels_enabled: true,
                downmix_level_a: 1,
                downmix_level_b: 2,
                downmix_gains_enabled: true,
                downmix_gain_5_q16: -(3 << 16),
                downmix_gain_2_q16: 2 << 16,
                lfe_downmix_enabled: true,
                lfe_downmix_level: 7,
            },
            ..EncoderMetadata::default()
        };
        let c_metadata = sys::AACENC_MetaData {
            ETSI_DmxLvl_present: 1,
            centerMixLevel: 2,
            surroundMixLevel: 3,
            dolbySurroundMode: 2,
            drcPresentationMode: 1,
            ExtMetaData: sys::AACENC_ExtMetaData {
                extAncDataEnable: 1,
                extDownmixLevelEnable: 1,
                extDownmixLevel_A: 1,
                extDownmixLevel_B: 2,
                dmxGainEnable: 1,
                dmxGain5: -(3 << 16),
                dmxGain2: 2 << 16,
                lfeDmxEnable: 1,
                lfeDmxLevel: 7,
            },
            ..sys::AACENC_MetaData::default()
        };

        let mut encoder = Encoder::open(2).unwrap();
        encoder.set_param(sys::AACENC_AOT, 2).unwrap();
        encoder.set_param(sys::AACENC_CHANNELMODE, 2).unwrap();
        encoder.set_param(sys::AACENC_SAMPLERATE, 48_000).unwrap();
        encoder.set_param(sys::AACENC_BITRATE, 128_000).unwrap();
        encoder.set_param(sys::AACENC_TRANSMUX, 0).unwrap();
        encoder.set_param(sys::AACENC_METADATA_MODE, 3).unwrap();
        encoder.initialize().unwrap();
        let info = encoder.info().unwrap();
        let mut raw = vec![0; info.max_output_bytes as usize];
        let mut bytes = 0;
        for frame in 0..8 {
            bytes = encoder
                .encode_interleaved_i16_with_ancillary_and_metadata(
                    &vec![0; info.frame_length as usize * 2],
                    &[],
                    (frame == 0).then_some(&c_metadata),
                    &mut raw,
                )
                .unwrap()
                .0;
            if bytes != 0 {
                break;
            }
        }
        raw.truncate(bytes);
        let mut decoder = AacLcDecoder::new(3, 2).unwrap();
        decoder.init_ancillary_data(64);
        decoder.decode_raw_data_block_f32(&raw).unwrap();
        assert_eq!(decoder.ancillary_data().len(), 1);
        assert_eq!(
            decoder.ancillary_data()[0].data,
            rust_metadata.etsi_ancillary_payload(None)
        );
    }
}
