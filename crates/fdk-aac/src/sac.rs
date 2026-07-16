//! MPEG Surround spatial-specific configuration used by `libSACdec`.

use std::fmt;

use crate::bits::{BitError, BitReader, BitWriter};
use crate::ld_sbr_qmf::{LdSbrQmfAnalysis, QmfError, QmfSlot};
use crate::usac_mps::{
    Mps212Frame, Mps212FrameDecoder, Mps212FrameEncoder, Mps212QmfProcessor, MpsError,
};

const SAMPLING_FREQUENCIES: [u32; 13] = [
    96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025, 8_000,
    7_350,
];
const LD_FREQUENCY_RESOLUTIONS: [u8; 8] = [0, 23, 15, 12, 9, 7, 5, 4];
const PARAMETER_BAND_15: [u8; 64] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
];
const CLD_QUANT_DB: [f64; 31] = [
    -50.0, -45.0, -40.0, -35.0, -30.0, -25.0, -22.0, -19.0, -16.0, -13.0, -10.0, -8.0, -6.0, -4.0,
    -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 13.0, 16.0, 19.0, 22.0, 25.0, 30.0, 35.0, 40.0, 45.0,
    50.0,
];
const ICC_QUANT: [f64; 8] = [1.0, 0.937, 0.84118, 0.60092, 0.36764, 0.0, -0.589, -0.99];

/// The standard 1-to-2 tree is the only tree accepted by this FDK build.
pub const TREE_212: u8 = 7;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpatialSpecificConfig {
    pub sampling_frequency: u32,
    pub time_slots: u8,
    pub frequency_resolution: u8,
    pub tree_config: u8,
    pub input_channels: u8,
    pub output_channels: u8,
    pub ott_boxes: u8,
    pub ttt_boxes: u8,
    pub quant_mode: u8,
    pub arbitrary_downmix: bool,
    pub fixed_gain_downmix: u8,
    pub temporal_shape_config: u8,
    pub decorrelation_config: u8,
    pub envelope_quant_mode: Option<bool>,
    /// Byte-aligned SpatialExtensionConfig payload retained for later tools.
    pub extension_data: Vec<u8>,
    pub bits_read: usize,
}

impl SpatialSpecificConfig {
    pub fn parse(input: &[u8]) -> Result<Self, SacError> {
        let mut reader = BitReader::new(input);
        let sampling_index = reader.read_u8(4)?;
        let sampling_frequency = match sampling_index {
            0..=12 => SAMPLING_FREQUENCIES[sampling_index as usize],
            15 => reader.read(24)?,
            value => return Err(SacError::ReservedSamplingFrequency(value)),
        };
        let time_slots = reader.read_u8(5)? + 1;
        let resolution_index = reader.read_u8(3)?;
        let frequency_resolution = LD_FREQUENCY_RESOLUTIONS[resolution_index as usize];
        let tree_config = reader.read_u8(4)?;
        if tree_config != TREE_212 {
            return Err(SacError::UnsupportedTreeConfig(tree_config));
        }
        let quant_mode = reader.read_u8(2)?;
        let arbitrary_downmix = reader.read_bool()?;
        let fixed_gain_downmix = reader.read_u8(3)?;
        let temporal_shape_config = reader.read_u8(2)?;
        if temporal_shape_config > 2 {
            return Err(SacError::ReservedTemporalShape(temporal_shape_config));
        }
        let decorrelation_config = reader.read_u8(2)?;
        if decorrelation_config > 2 {
            return Err(SacError::ReservedDecorrelation(decorrelation_config));
        }
        let envelope_quant_mode = (temporal_shape_config == 2)
            .then(|| reader.read_bool())
            .transpose()?;

        while reader.bits_read() % 8 != 0 {
            reader.read_bool()?;
        }
        let mut extension_data = Vec::with_capacity(reader.remaining_bits() / 8);
        while reader.remaining_bits() >= 8 {
            extension_data.push(reader.read_u8(8)?);
        }
        Ok(Self {
            sampling_frequency,
            time_slots,
            frequency_resolution,
            tree_config,
            input_channels: 1,
            output_channels: 2,
            ott_boxes: 1,
            ttt_boxes: 0,
            quant_mode,
            arbitrary_downmix,
            fixed_gain_downmix,
            temporal_shape_config,
            decorrelation_config,
            envelope_quant_mode,
            extension_data,
            bits_read: reader.bits_read(),
        })
    }

    /// FDK's default standard 1-to-2 configuration used when no SSC exists.
    pub fn default_212(sampling_frequency: u32, time_slots: u8) -> Result<Self, SacError> {
        if sampling_frequency == 0 {
            return Err(SacError::InvalidSamplingFrequency);
        }
        if time_slots == 0 || time_slots > 64 {
            return Err(SacError::InvalidTimeSlots(time_slots));
        }
        Ok(Self {
            sampling_frequency,
            time_slots,
            frequency_resolution: 28,
            tree_config: TREE_212,
            input_channels: 1,
            output_channels: 2,
            ott_boxes: 1,
            ttt_boxes: 0,
            quant_mode: 0,
            arbitrary_downmix: false,
            fixed_gain_downmix: 0,
            temporal_shape_config: 0,
            decorrelation_config: 0,
            envelope_quant_mode: None,
            extension_data: Vec::new(),
            bits_read: 0,
        })
    }

    /// Serialize the low-delay MPEG Surround SpatialSpecificConfig syntax.
    ///
    /// This is the byte-aligned syntax emitted by libSACenc and embedded in an
    /// AAC-ELD `ELDEXT_LDSAC` extension.  The returned bit count includes the
    /// zero alignment bits, just like `FDK_sacenc_writeSpatialSpecificConfig`.
    pub fn write(&self) -> Result<(Vec<u8>, usize), SacError> {
        if self.sampling_frequency == 0 || self.sampling_frequency > 0x00ff_ffff {
            return Err(SacError::InvalidSamplingFrequency);
        }
        if self.time_slots == 0 || self.time_slots > 32 {
            return Err(SacError::InvalidTimeSlots(self.time_slots));
        }
        let resolution_index = LD_FREQUENCY_RESOLUTIONS
            .iter()
            .position(|&bands| bands == self.frequency_resolution)
            .ok_or(SacError::UnsupportedFrequencyResolution(
                self.frequency_resolution,
            ))? as u32;
        if self.tree_config != TREE_212 {
            return Err(SacError::UnsupportedTreeConfig(self.tree_config));
        }
        if self.quant_mode > 3
            || self.fixed_gain_downmix > 7
            || self.temporal_shape_config > 2
            || self.decorrelation_config > 2
        {
            return Err(SacError::InvalidConfiguration);
        }

        let sampling_index = SAMPLING_FREQUENCIES
            .iter()
            .position(|&frequency| frequency == self.sampling_frequency)
            .map_or(15, |index| index as u32);
        let mut writer = BitWriter::new();
        writer.write(sampling_index, 4);
        if sampling_index == 15 {
            writer.write(self.sampling_frequency, 24);
        }
        writer.write(u32::from(self.time_slots - 1), 5);
        writer.write(resolution_index, 3);
        writer.write(u32::from(self.tree_config), 4);
        writer.write(u32::from(self.quant_mode), 2);
        writer.write_bool(self.arbitrary_downmix);
        writer.write(u32::from(self.fixed_gain_downmix), 3);
        writer.write(u32::from(self.temporal_shape_config), 2);
        writer.write(u32::from(self.decorrelation_config), 2);
        if self.temporal_shape_config == 2 {
            writer.write_bool(self.envelope_quant_mode.unwrap_or(false));
        }
        writer.byte_align();
        for &byte in &self.extension_data {
            writer.write(u32::from(byte), 8);
        }
        let bits = writer.bits_written();
        Ok((writer.finish(), bits))
    }

    /// Configuration used by the FDK AAC-ELD 2-to-1-to-2 encoder.
    pub fn eld_212(sampling_frequency: u32, time_slots: u8) -> Result<Self, SacError> {
        let mut config = Self::default_212(sampling_frequency, time_slots)?;
        config.frequency_resolution = 15;
        config.fixed_gain_downmix = 2; // FDK's default 3 dB downmix gain.
        Ok(config)
    }
}

/// Stateful AAC-ELD low-delay MPEG Surround 2-to-1-to-2 encoder core.
#[derive(Debug, Clone)]
pub struct Sac212Encoder {
    config: SpatialSpecificConfig,
    qmf_bands: usize,
    left_analysis: LdSbrQmfAnalysis,
    right_analysis: LdSbrQmfAnalysis,
    frame_encoder: Mps212FrameEncoder,
    frame_count: usize,
    independency_factor: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sac212EncodedFrame {
    pub downmix: Vec<f32>,
    pub payload: Vec<u8>,
    pub payload_bits: usize,
    pub cld: Vec<i8>,
    pub icc: Vec<i8>,
    pub independent: bool,
}

impl Sac212EncodedFrame {
    /// Wrap the SAC frame in the ER AAC extension syntax used by AAC-ELD.
    /// The `0x3` nibble is the FDK low-delay SAC framing marker; unlike other
    /// MPEG Surround profiles ELD carries its SSC explicitly in the ASC.
    pub fn eld_extension_payload(&self) -> (Vec<u8>, usize) {
        let mut writer = BitWriter::new();
        writer.write(0x09, 4); // EXT_LDSAC_DATA
        writer.write(0x03, 4); // sacHeaderFlag=0 framing marker
        for bit in 0..self.payload_bits {
            writer.write(u32::from((self.payload[bit / 8] >> (7 - bit % 8)) & 1), 1);
        }
        let bits = writer.bits_written();
        (writer.finish(), bits)
    }
}

impl Sac212Encoder {
    pub fn new(sampling_frequency: u32, time_slots: u8) -> Result<Self, SacEncodeError> {
        let qmf_bands = if sampling_frequency < 27_713 { 32 } else { 64 };
        if sampling_frequency > 55_426 {
            return Err(SacEncodeError::UnsupportedQmfBands(128));
        }
        Self::new_with_qmf_bands(sampling_frequency, time_slots, qmf_bands)
    }

    pub(crate) fn new_with_qmf_bands(
        sampling_frequency: u32,
        time_slots: u8,
        qmf_bands: usize,
    ) -> Result<Self, SacEncodeError> {
        if !matches!(qmf_bands, 32 | 64) {
            return Err(SacEncodeError::UnsupportedQmfBands(qmf_bands));
        }
        let config = SpatialSpecificConfig::eld_212(sampling_frequency, time_slots)?;
        Ok(Self {
            config,
            qmf_bands,
            left_analysis: LdSbrQmfAnalysis::new_with_channels(qmf_bands)?,
            right_analysis: LdSbrQmfAnalysis::new_with_channels(qmf_bands)?,
            frame_encoder: Mps212FrameEncoder::new(time_slots as usize, 15)?
                .with_low_delay_framing(),
            frame_count: 0,
            independency_factor: 20,
        })
    }

    pub fn config(&self) -> &SpatialSpecificConfig {
        &self.config
    }

    pub fn encode(
        &mut self,
        left: &[f32],
        right: &[f32],
    ) -> Result<Sac212EncodedFrame, SacEncodeError> {
        if left.len() != right.len() {
            return Err(SacEncodeError::ChannelLengthMismatch);
        }
        let expected = self.frame_encoder.time_slots() * self.qmf_bands;
        if left.len() != expected {
            return Err(SacEncodeError::InvalidFrameLength {
                expected,
                actual: left.len(),
            });
        }
        let left_f64 = left
            .iter()
            .map(|&sample| f64::from(sample))
            .collect::<Vec<_>>();
        let right_f64 = right
            .iter()
            .map(|&sample| f64::from(sample))
            .collect::<Vec<_>>();
        let left_qmf = self.left_analysis.process_frame(&left_f64)?;
        let right_qmf = self.right_analysis.process_frame(&right_f64)?;
        let (cld, icc) = extract_212_parameters(&left_qmf, &right_qmf);
        let independent = self.frame_count % self.independency_factor == 0;
        let (payload, payload_bits) = self.frame_encoder.encode(&cld, &icc, independent)?;
        self.frame_count += 1;
        let downmix = left
            .iter()
            .zip(right)
            .map(|(&l, &r)| (l + r) * std::f32::consts::FRAC_1_SQRT_2)
            .collect();
        Ok(Sac212EncodedFrame {
            downmix,
            payload,
            payload_bits,
            cld,
            icc,
            independent,
        })
    }
}

fn extract_212_parameters(left: &[QmfSlot], right: &[QmfSlot]) -> (Vec<i8>, Vec<i8>) {
    let qmf_bands = left.first().map_or(64, |slot| slot.real.len());
    let mut left_power = [0.0f64; 15];
    let mut right_power = [0.0f64; 15];
    let mut product = [0.0f64; 15];
    for (left_slot, right_slot) in left.iter().zip(right) {
        for band in 0..qmf_bands {
            // SACENC indexes the 64-entry low-delay mapping directly even
            // when the analysis bank has only 32 channels.  Scaling a 32-QMF
            // index to every other table entry incorrectly leaves alternating
            // low parameter bands empty.
            let parameter_band = usize::from(PARAMETER_BAND_15[band]);
            let lr = left_slot.real[band];
            let li = left_slot.imaginary[band];
            let rr = right_slot.real[band];
            let ri = right_slot.imaginary[band];
            left_power[parameter_band] += lr * lr + li * li;
            right_power[parameter_band] += rr * rr + ri * ri;
            product[parameter_band] += lr * rr + li * ri;
        }
    }
    let mut cld = Vec::with_capacity(15);
    let mut icc = Vec::with_capacity(15);
    for band in 0..15 {
        // With a 32-channel QMF bank the direct FDK mapping has no source
        // subband for parameter band 14.  SACENC leaves that band's
        // initialized CLD/ICC values at zero instead of quantizing 0/0 as an
        // uncorrelated ICC value.
        if left_power[band] == 0.0 && right_power[band] == 0.0 {
            cld.push(0);
            icc.push(0);
            continue;
        }
        let db = 10.0 * ((left_power[band] + 1e-30) / (right_power[band] + 1e-30)).log10();
        let cld_index = CLD_QUANT_DB
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| (db - **a).abs().total_cmp(&(db - **b).abs()))
            .map_or(15, |(index, _)| index) as i8
            - 15;
        let correlation = product[band] / (left_power[band] * right_power[band]).sqrt().max(1e-30);
        let icc_index = ICC_QUANT
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                (correlation - **a)
                    .abs()
                    .total_cmp(&(correlation - **b).abs())
            })
            .map_or(0, |(index, _)| index) as i8;
        cld.push(cld_index);
        icc.push(icc_index);
    }
    (cld, icc)
}

/// Stateful standard MPEG Surround 1-to-2 payload decoder and QMF renderer.
#[derive(Debug, Clone)]
pub struct Sac212Decoder {
    config: SpatialSpecificConfig,
    frame_decoder: Mps212FrameDecoder,
    qmf_processor: Mps212QmfProcessor,
}

impl Sac212Decoder {
    pub fn new(config: SpatialSpecificConfig) -> Result<Self, SacDecodeError> {
        if config.tree_config != TREE_212
            || config.input_channels != 1
            || config.output_channels != 2
        {
            return Err(SacDecodeError::UnsupportedLayout);
        }
        let frame_decoder = Mps212FrameDecoder::new(
            config.time_slots as usize,
            config.frequency_resolution as usize,
            0,
            true,
            false,
        )
        .with_temporal_shape_config(config.temporal_shape_config)
        .with_low_delay_framing();
        let qmf_bands = if config.sampling_frequency < 27_713 {
            32
        } else {
            64
        };
        let qmf_processor = Mps212QmfProcessor::new_with_qmf_bands(
            config.frequency_resolution as usize,
            config.decorrelation_config,
            qmf_bands,
        )?;
        Ok(Self {
            config,
            frame_decoder,
            qmf_processor,
        })
    }

    pub fn config(&self) -> &SpatialSpecificConfig {
        &self.config
    }

    pub fn parse_frame(
        &mut self,
        payload: &[u8],
        payload_bits: usize,
    ) -> Result<Mps212Frame, SacDecodeError> {
        let storage = low_delay_payload_storage(payload, payload_bits)?;
        let mut reader = BitReader::new(&storage);
        let mut trial = self.frame_decoder.clone();
        let frame = trial.parse(&mut reader, false)?;
        validate_low_delay_zero_overread(reader.bits_read(), payload_bits)?;
        self.frame_decoder = trial;
        Ok(frame)
    }

    pub fn decode_qmf(
        &mut self,
        mono: &[QmfSlot],
        payload: &[u8],
        payload_bits: usize,
    ) -> Result<(Vec<f64>, Vec<f64>), SacDecodeError> {
        if mono.len() != self.config.time_slots as usize {
            return Err(SacDecodeError::QmfSlotCount {
                expected: self.config.time_slots as usize,
                actual: mono.len(),
            });
        }
        let storage = low_delay_payload_storage(payload, payload_bits)?;
        let mut reader = BitReader::new(&storage);
        let mut trial_decoder = self.frame_decoder.clone();
        let mut trial_processor = self.qmf_processor.clone();
        let frame = trial_decoder.parse(&mut reader, false)?;
        validate_low_delay_zero_overread(reader.bits_read(), payload_bits)?;
        let pcm = trial_processor.process_qmf(mono, &frame)?;
        self.frame_decoder = trial_decoder;
        self.qmf_processor = trial_processor;
        Ok(pcm)
    }
}

fn low_delay_payload_storage(payload: &[u8], payload_bits: usize) -> Result<Vec<u8>, BitError> {
    BitReader::with_bit_len(payload, payload_bits)?;
    let bytes = payload_bits.div_ceil(8);
    let mut storage = payload[..bytes].to_vec();
    if payload_bits % 8 != 0 {
        let keep = payload_bits % 8;
        storage[bytes - 1] &= 0xff << (8 - keep);
    }
    // The FDK LD-SAC Huffman reader can inspect at most the next seven
    // zero-valued termination bits even when nOutputBits ends on a byte.
    storage.push(0);
    Ok(storage)
}

fn validate_low_delay_zero_overread(bits_read: usize, payload_bits: usize) -> Result<(), BitError> {
    if bits_read > payload_bits.saturating_add(7) {
        Err(BitError::UnexpectedEof {
            needed_bits: bits_read - payload_bits,
            remaining_bits: 0,
        })
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SacDecodeError {
    Bit(BitError),
    Mps(MpsError),
    UnsupportedLayout,
    QmfSlotCount { expected: usize, actual: usize },
}

impl From<BitError> for SacDecodeError {
    fn from(value: BitError) -> Self {
        Self::Bit(value)
    }
}

impl From<MpsError> for SacDecodeError {
    fn from(value: MpsError) -> Self {
        Self::Mps(value)
    }
}

impl fmt::Display for SacDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MPEG Surround frame decode error: {self:?}")
    }
}

impl std::error::Error for SacDecodeError {}

#[derive(Debug, Clone, PartialEq)]
pub enum SacEncodeError {
    Config(SacError),
    Qmf(QmfError),
    Mps(MpsError),
    UnsupportedQmfBands(usize),
    ChannelLengthMismatch,
    InvalidFrameLength { expected: usize, actual: usize },
}

impl From<SacError> for SacEncodeError {
    fn from(value: SacError) -> Self {
        Self::Config(value)
    }
}

impl From<QmfError> for SacEncodeError {
    fn from(value: QmfError) -> Self {
        Self::Qmf(value)
    }
}

impl From<MpsError> for SacEncodeError {
    fn from(value: MpsError) -> Self {
        Self::Mps(value)
    }
}

impl fmt::Display for SacEncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MPEG Surround frame encode error: {self:?}")
    }
}

impl std::error::Error for SacEncodeError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SacError {
    Bit(BitError),
    ReservedSamplingFrequency(u8),
    UnsupportedTreeConfig(u8),
    ReservedTemporalShape(u8),
    ReservedDecorrelation(u8),
    InvalidSamplingFrequency,
    InvalidTimeSlots(u8),
    UnsupportedFrequencyResolution(u8),
    InvalidConfiguration,
}

impl From<BitError> for SacError {
    fn from(value: BitError) -> Self {
        Self::Bit(value)
    }
}

impl fmt::Display for SacError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MPEG Surround configuration error: {self:?}")
    }
}

impl std::error::Error for SacError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitWriter;

    #[test]
    fn parses_supported_standard_212_spatial_config() {
        let mut writer = BitWriter::new();
        writer.write(3, 4); // 48 kHz
        writer.write(29, 5); // 30 slots
        writer.write(1, 3); // 23 parameter bands
        writer.write(TREE_212.into(), 4);
        writer.write(2, 2);
        writer.write_bool(false);
        writer.write(1, 3);
        writer.write(2, 2);
        writer.write(1, 2);
        writer.write_bool(true); // env quant mode
        while writer.bits_written() % 8 != 0 {
            writer.write_bool(false);
        }
        writer.write(0xa5, 8);
        let config = SpatialSpecificConfig::parse(&writer.finish()).unwrap();
        assert_eq!(config.sampling_frequency, 48_000);
        assert_eq!(config.time_slots, 30);
        assert_eq!(config.frequency_resolution, 23);
        assert_eq!((config.input_channels, config.output_channels), (1, 2));
        assert_eq!(config.envelope_quant_mode, Some(true));
        assert_eq!(config.extension_data, [0xa5]);
    }

    #[test]
    fn constructs_fdk_default_212_config() {
        let config = SpatialSpecificConfig::default_212(48_000, 30).unwrap();
        assert_eq!(config.frequency_resolution, 28);
        assert_eq!(config.tree_config, TREE_212);
        assert_eq!((config.ott_boxes, config.ttt_boxes), (1, 0));
    }

    #[test]
    fn writes_fdk_eld_212_spatial_config_bit_exactly() {
        let config = SpatialSpecificConfig::eld_212(48_000, 16).unwrap();
        let (encoded, bits) = config.write().unwrap();
        // samplingFrequencyIndex=3, frameLength=15, freqResIndex=2 (15
        // bands), tree=7, fine quantization, no arbitrary downmix, 3 dB
        // fixed downmix gain, temporal shaping off, decorrelator 0, fill=0.
        assert_eq!(encoded, [0x37, 0xa7, 0x08, 0x00]);
        assert_eq!(bits, 32);
        let parsed = SpatialSpecificConfig::parse(&encoded).unwrap();
        assert_eq!(parsed.sampling_frequency, 48_000);
        assert_eq!(parsed.time_slots, 16);
        assert_eq!(parsed.frequency_resolution, 15);
        assert_eq!(parsed.tree_config, TREE_212);
        assert_eq!(parsed.quant_mode, 0);
        assert_eq!(parsed.fixed_gain_downmix, 2);
    }

    #[test]
    fn spatial_config_writer_uses_explicit_sampling_frequency_escape() {
        let config = SpatialSpecificConfig::eld_212(50_000, 15).unwrap();
        let (encoded, bits) = config.write().unwrap();
        assert_eq!(bits, 56);
        let parsed = SpatialSpecificConfig::parse(&encoded).unwrap();
        assert_eq!(parsed.sampling_frequency, 50_000);
        assert_eq!(parsed.time_slots, 15);
    }

    #[test]
    fn eld_sac_encoder_extracts_spatial_parameters_and_roundtrips_payload() {
        let mut encoder = Sac212Encoder::new(48_000, 16).unwrap();
        let left = (0..1024)
            .map(|sample| (sample as f32 * 0.071).sin() * 0.8)
            .collect::<Vec<_>>();
        let right = (0..1024)
            .map(|sample| (sample as f32 * 0.071 + 0.6).sin() * 0.2)
            .collect::<Vec<_>>();
        let encoded = encoder.encode(&left, &right).unwrap();
        assert!(encoded.independent);
        assert_eq!(encoded.downmix.len(), 1024);
        assert!(encoded.cld.iter().any(|&value| value > 0));
        assert!(encoded.icc.iter().any(|&value| value > 0));

        let mut decoder = Sac212Decoder::new(encoder.config().clone()).unwrap();
        let frame = decoder
            .parse_frame(&encoded.payload, encoded.payload_bits)
            .unwrap();
        assert_eq!(frame.parameter_sets[0].cld, encoded.cld);
        assert_eq!(frame.parameter_sets[0].icc, encoded.icc);
        let (extension, extension_bits) = encoded.eld_extension_payload();
        let mut extension_reader = BitReader::with_bit_len(&extension, extension_bits).unwrap();
        assert_eq!(extension_reader.read_u8(4).unwrap(), 0x09);
        assert_eq!(extension_reader.read_u8(4).unwrap(), 0x03);
        let remaining = (0..encoded.payload_bits)
            .map(|_| extension_reader.read_bool().unwrap())
            .collect::<Vec<_>>();
        let expected = (0..encoded.payload_bits)
            .map(|bit| ((encoded.payload[bit / 8] >> (7 - bit % 8)) & 1) != 0)
            .collect::<Vec<_>>();
        assert_eq!(remaining, expected);

        for _ in 1..20 {
            assert!(!encoder.encode(&left, &right).unwrap().independent);
        }
        assert!(encoder.encode(&left, &right).unwrap().independent);
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn eld_sac_parameters_are_compared_directly_with_c_encoder() {
        for (sampling_frequency, time_slots) in [(24_000, 16), (48_000, 8)] {
            let mut random = 0x1234_5678u32 ^ sampling_frequency;
            let left = (0..512)
                .map(|_| {
                    random = random.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (random >> 16) as i16 / 2
                })
                .collect::<Vec<_>>();
            let right = left.iter().map(|&sample| sample / 4).collect::<Vec<_>>();
            let mut c_payload = vec![0u8; 1_024];
            let mut c_bits = 0u32;
            let mut c_downmix = vec![0i16; 512];
            let result = unsafe {
                crate::sys::fdk_mps_encode_frame_test(
                    sampling_frequency,
                    512,
                    left.as_ptr(),
                    right.as_ptr(),
                    c_payload.as_mut_ptr(),
                    c_payload.len() as u32,
                    &mut c_bits,
                    c_downmix.as_mut_ptr(),
                )
            };
            assert_eq!(result, 0, "sampling frequency {sampling_frequency}");
            c_payload.truncate(c_bits.div_ceil(8) as usize);
            assert_eq!(c_bits % 8, 0);
            assert!(!c_payload.is_empty());
            let mut c_cld = [0i8; 15];
            let mut c_icc = [0i8; 15];
            assert_eq!(
                unsafe {
                    crate::sys::fdk_mps_last_parameters_test(
                        c_cld.as_mut_ptr(),
                        c_icc.as_mut_ptr(),
                        15,
                    )
                },
                15
            );
            let mut c_payload_decoder = Sac212Decoder::new(
                SpatialSpecificConfig::eld_212(sampling_frequency, time_slots).unwrap(),
            )
            .unwrap();
            let decoded = c_payload_decoder
                .parse_frame(&c_payload, c_bits as usize)
                .unwrap();
            assert_eq!(decoded.parameter_sets[0].cld, c_cld);
            assert_eq!(decoded.parameter_sets[0].icc, c_icc);
            let mut rust = Sac212Encoder::new(sampling_frequency, time_slots).unwrap();
            let rust_frame = rust
                .encode(
                    &left.iter().map(|&v| f32::from(v)).collect::<Vec<_>>(),
                    &right.iter().map(|&v| f32::from(v)).collect::<Vec<_>>(),
                )
                .unwrap();
            assert_eq!(
                c_cld.as_slice(),
                rust_frame.cld,
                "CLD at {sampling_frequency} Hz"
            );
            assert_eq!(
                c_icc.as_slice(),
                rust_frame.icc,
                "ICC at {sampling_frequency} Hz"
            );
            assert_eq!(
                (c_bits as usize, c_payload.as_slice()),
                (rust_frame.payload_bits, rust_frame.payload.as_slice()),
                "SpatialFrame payload at {sampling_frequency} Hz"
            );
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn eld_sac_dependent_payload_matches_c_huffman_choice() {
        let mut random = 0x8bad_f00du32;
        let first_left = (0..512)
            .map(|_| {
                random = random.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (random >> 16) as i16 / 2
            })
            .collect::<Vec<_>>();
        let second_left = (0..512)
            .map(|index| first_left[(index + 37) % 512])
            .collect::<Vec<_>>();
        let mut left = first_left.clone();
        left.extend_from_slice(&second_left);
        let mut right = first_left
            .iter()
            .map(|&sample| sample / 4)
            .collect::<Vec<_>>();
        // Change the CLD by a small, robust amount while keeping ICC exactly
        // correlated.  This makes the second frame exercise FDK's temporal
        // differential choice without placing fixed/floating analysis on an
        // ICC quantizer boundary.
        right.extend(second_left.iter().map(|&sample| sample / 3));

        const STRIDE: usize = 1_024;
        let mut c_payload = vec![0u8; STRIDE * 2];
        let mut c_bits = [0u32; 2];
        assert_eq!(
            unsafe {
                crate::sys::fdk_mps_encode_two_frames_test(
                    48_000,
                    512,
                    left.as_ptr(),
                    right.as_ptr(),
                    c_payload.as_mut_ptr(),
                    STRIDE as u32,
                    c_bits.as_mut_ptr(),
                )
            },
            0
        );
        let mut final_c_cld = [0i8; 15];
        let mut final_c_icc = [0i8; 15];
        assert_eq!(
            unsafe {
                crate::sys::fdk_mps_last_parameters_test(
                    final_c_cld.as_mut_ptr(),
                    final_c_icc.as_mut_ptr(),
                    15,
                )
            },
            15
        );

        let mut rust = Sac212Encoder::new(48_000, 8).unwrap();
        for frame in 0..2 {
            let range = frame * 512..(frame + 1) * 512;
            let encoded = rust
                .encode(
                    &left[range.clone()]
                        .iter()
                        .map(|&sample| f32::from(sample))
                        .collect::<Vec<_>>(),
                    &right[range]
                        .iter()
                        .map(|&sample| f32::from(sample))
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            if frame == 1 {
                assert_eq!(encoded.cld, final_c_cld);
                assert_eq!(encoded.icc, final_c_icc);
            }
            let c_bytes = c_bits[frame].div_ceil(8) as usize;
            assert_eq!(
                encoded.payload_bits, c_bits[frame] as usize,
                "frame {frame}"
            );
            assert_eq!(
                encoded.payload,
                c_payload[frame * STRIDE..frame * STRIDE + c_bytes],
                "frame {frame}"
            );
        }
    }

    fn independent_default_frame() -> (Vec<u8>, usize) {
        let mut writer = BitWriter::new();
        writer.write_bool(false); // fixed framing
        writer.write(0, 1); // one parameter set (LD-SAC syntax)
        writer.write_bool(true); // independent
        writer.write(0, 2); // default CLD
        writer.write(0, 2); // default ICC
        writer.write(0, 2); // no smoothing
        let bits = writer.bits_written();
        (writer.finish(), bits)
    }

    fn zero_slot() -> QmfSlot {
        QmfSlot {
            real: vec![0.0; 64],
            imaginary: vec![0.0; 64],
        }
    }

    #[test]
    fn parses_and_renders_standard_212_frame_transactionally() {
        let config = SpatialSpecificConfig::default_212(48_000, 30).unwrap();
        let mut decoder = Sac212Decoder::new(config).unwrap();
        let (payload, payload_bits) = independent_default_frame();
        let frame = decoder.parse_frame(&payload, payload_bits).unwrap();
        assert!(frame.independent);
        assert_eq!(frame.parameter_sets[0].slot, 29);
        assert_eq!(frame.parameter_sets[0].cld, vec![0; 28]);

        let slots = vec![zero_slot(); 30];
        let (left, right) = decoder.decode_qmf(&slots, &payload, payload_bits).unwrap();
        assert_eq!((left.len(), right.len()), (1920, 1920));
        assert!(left.iter().chain(&right).all(|sample| sample.is_finite()));
    }

    #[test]
    fn rejects_wrong_sac_qmf_slot_count_without_advancing_state() {
        let config = SpatialSpecificConfig::default_212(48_000, 30).unwrap();
        let mut decoder = Sac212Decoder::new(config).unwrap();
        let (payload, payload_bits) = independent_default_frame();
        assert!(matches!(
            decoder.decode_qmf(&[zero_slot()], &payload, payload_bits),
            Err(SacDecodeError::QmfSlotCount { .. })
        ));
        assert!(decoder
            .decode_qmf(&vec![zero_slot(); 30], &payload, payload_bits)
            .is_ok());
    }

    fn config_bits(sampling_index: u8, tree: u8, temporal_shape: u8, decorrelation: u8) -> Vec<u8> {
        let mut writer = BitWriter::new();
        writer.write(sampling_index.into(), 4);
        if sampling_index == 15 {
            writer.write(48_000, 24);
        }
        writer.write(31, 5);
        writer.write(0, 3);
        writer.write(tree.into(), 4);
        writer.write(0, 2);
        writer.write_bool(false);
        writer.write(0, 3);
        writer.write(temporal_shape.into(), 2);
        writer.write(decorrelation.into(), 2);
        if temporal_shape == 2 {
            writer.write_bool(false);
        }
        writer.finish()
    }

    #[test]
    fn parses_explicit_frequency_and_all_resolution_indices() {
        let explicit = SpatialSpecificConfig::parse(&config_bits(15, TREE_212, 0, 0)).unwrap();
        assert_eq!(explicit.sampling_frequency, 48_000);
        assert_eq!(explicit.time_slots, 32);
        let envelope = SpatialSpecificConfig::parse(&config_bits(3, TREE_212, 2, 0)).unwrap();
        assert_eq!(envelope.envelope_quant_mode, Some(false));
        for resolution in 0..8u32 {
            let mut writer = BitWriter::new();
            writer.write(3, 4);
            writer.write(0, 5);
            writer.write(resolution, 3);
            writer.write(TREE_212.into(), 4);
            writer.write(0, 2);
            writer.write_bool(false);
            writer.write(0, 3);
            writer.write(0, 2);
            writer.write(0, 2);
            let parsed = SpatialSpecificConfig::parse(&writer.finish()).unwrap();
            assert_eq!(
                parsed.frequency_resolution,
                LD_FREQUENCY_RESOLUTIONS[resolution as usize]
            );
        }
    }

    #[test]
    fn rejects_reserved_configuration_values_and_truncation() {
        assert_eq!(
            SpatialSpecificConfig::parse(&config_bits(13, TREE_212, 0, 0)),
            Err(SacError::ReservedSamplingFrequency(13))
        );
        assert_eq!(
            SpatialSpecificConfig::parse(&config_bits(3, 0, 0, 0)),
            Err(SacError::UnsupportedTreeConfig(0))
        );
        assert_eq!(
            SpatialSpecificConfig::parse(&config_bits(3, TREE_212, 3, 0)),
            Err(SacError::ReservedTemporalShape(3))
        );
        assert_eq!(
            SpatialSpecificConfig::parse(&config_bits(3, TREE_212, 0, 3)),
            Err(SacError::ReservedDecorrelation(3))
        );
        assert!(matches!(
            SpatialSpecificConfig::parse(&[]),
            Err(SacError::Bit(BitError::UnexpectedEof { .. }))
        ));
    }

    #[test]
    fn default_config_validates_frequency_and_slot_range() {
        assert_eq!(
            SpatialSpecificConfig::default_212(0, 32),
            Err(SacError::InvalidSamplingFrequency)
        );
        for slots in [0, 65] {
            assert_eq!(
                SpatialSpecificConfig::default_212(48_000, slots),
                Err(SacError::InvalidTimeSlots(slots))
            );
        }
        assert!(SpatialSpecificConfig::default_212(48_000, 64).is_ok());
    }

    #[test]
    fn decoder_validates_layout_and_exposes_configuration() {
        let base = SpatialSpecificConfig::default_212(48_000, 32).unwrap();
        let decoder = Sac212Decoder::new(base.clone()).unwrap();
        assert_eq!(decoder.config(), &base);
        for invalid in [
            SpatialSpecificConfig {
                tree_config: 0,
                ..base.clone()
            },
            SpatialSpecificConfig {
                input_channels: 2,
                ..base.clone()
            },
            SpatialSpecificConfig {
                output_channels: 1,
                ..base.clone()
            },
        ] {
            assert!(matches!(
                Sac212Decoder::new(invalid),
                Err(SacDecodeError::UnsupportedLayout)
            ));
        }
        let mut invalid_decorrelation = base;
        invalid_decorrelation.decorrelation_config = 3;
        assert!(matches!(
            Sac212Decoder::new(invalid_decorrelation),
            Err(SacDecodeError::Mps(MpsError::InvalidDataMode))
        ));
    }

    #[test]
    fn frame_parser_rejects_invalid_bit_length_without_mutating_state() {
        let config = SpatialSpecificConfig::default_212(48_000, 30).unwrap();
        let mut decoder = Sac212Decoder::new(config).unwrap();
        assert!(matches!(
            decoder.parse_frame(&[], 1),
            Err(SacDecodeError::Bit(BitError::UnexpectedEof { .. }))
        ));
        let (payload, bits) = independent_default_frame();
        assert!(decoder.parse_frame(&payload, bits).is_ok());
        assert!(matches!(
            decoder.decode_qmf(&vec![zero_slot(); 30], &[], 1),
            Err(SacDecodeError::Bit(BitError::UnexpectedEof { .. }))
        ));
        assert!(decoder
            .decode_qmf(&vec![zero_slot(); 30], &payload, bits)
            .is_ok());
    }

    #[test]
    fn error_conversions_and_messages_preserve_variants() {
        let bit = BitError::UnexpectedEof {
            needed_bits: 1,
            remaining_bits: 0,
        };
        assert_eq!(SacError::from(bit.clone()), SacError::Bit(bit.clone()));
        assert_eq!(SacDecodeError::from(bit.clone()), SacDecodeError::Bit(bit));
        let mps = MpsError::InvalidParameterSlot;
        assert_eq!(SacDecodeError::from(mps.clone()), SacDecodeError::Mps(mps));
        let config_errors = [
            SacError::ReservedSamplingFrequency(13),
            SacError::UnsupportedTreeConfig(0),
            SacError::ReservedTemporalShape(3),
            SacError::ReservedDecorrelation(3),
            SacError::InvalidSamplingFrequency,
            SacError::InvalidTimeSlots(0),
        ];
        assert!(config_errors
            .iter()
            .all(|error| !error.to_string().is_empty()));
        let decode_errors = [
            SacDecodeError::UnsupportedLayout,
            SacDecodeError::QmfSlotCount {
                expected: 32,
                actual: 1,
            },
        ];
        assert!(decode_errors
            .iter()
            .all(|error| !error.to_string().is_empty()));
    }
}
