//! Pure Rust SBR encoder analysis primitives.

use std::fmt;

use crate::asc::LdSbrHeader;
use crate::bits::{BitReader, BitWriter};
use crate::ld_sbr::{encode_sbr_huffman, LdSbrError, LdSbrFrequencyTables, SbrHuffmanBook};
use crate::ld_sbr_qmf::{LdSbrQmfAnalysis, QmfError, QmfSlot};
use crate::sbr::EXT_SBR_DATA;

#[derive(Debug, Clone, PartialEq)]
pub struct SbrEncoderBand {
    pub energy: f64,
    /// Normalized adjacent-slot complex correlation, from 0 (noise) to 1 (tone).
    pub tonality: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SbrEncoderEnvelope {
    pub start_slot: usize,
    pub end_slot: usize,
    pub bands: Vec<SbrEncoderBand>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SbrEncoderAnalysisFrame {
    pub slots: Vec<QmfSlot>,
    pub envelopes: Vec<SbrEncoderEnvelope>,
    pub transient_ratio: f64,
    /// ELD FIXFIXonly transient position and per-envelope resolution. Ordinary
    /// SBR frames leave these unset.
    pub low_delay_transient_position: Option<u8>,
    pub low_delay_frequency_resolution: Option<Vec<bool>>,
    low_delay_amp_resolution: Option<bool>,
    low_delay_global_tonality: Option<f64>,
    low_delay_envelope_coding: Option<Vec<LowDelayEnvelopeCoding>>,
    low_delay_noise_coding: Option<Vec<LowDelayEnvelopeCoding>>,
    low_delay_invf_modes: Option<Vec<u8>>,
    low_delay_patch_map: Option<Vec<Option<usize>>>,
    pub(crate) low_delay_prequant_debug: Option<LowDelayPrequantDebug>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LowDelayPrequantDebug {
    pub(crate) energies: Vec<Vec<i32>>,
    pub(crate) counts: Vec<Vec<i32>>,
    pub(crate) ybuffer_scales: (i32, i32),
    pub(crate) qmf_scale: i32,
    pub(crate) common_scale: i32,
}

#[derive(Debug, Clone, PartialEq)]
struct LowDelayEnvelopeCoding {
    time: bool,
    deltas: Vec<i8>,
}

#[derive(Debug, Clone)]
pub(crate) struct LowDelaySbrCodingState {
    previous_envelope: Option<Vec<i8>>,
    previous_noise: Option<Vec<i8>>,
    first_envelope_time_streak: usize,
    noise_level_history: Vec<[f64; 4]>,
    invf_bands: Vec<InverseFilterBandState>,
    previous_invf_modes: Vec<u8>,
    noise_floor_cap: f64,
    transient_next_frame: bool,
    current_transient_frame: bool,
}

impl Default for LowDelaySbrCodingState {
    fn default() -> Self {
        Self {
            previous_envelope: None,
            previous_noise: None,
            first_envelope_time_streak: 0,
            noise_level_history: Vec::new(),
            invf_bands: Vec::new(),
            previous_invf_modes: Vec::new(),
            // Synthetic/general analysis keeps the uncapped +6 dB mode.
            // Concrete AAC-LD/ELD encoders override this from the C tuning
            // table when SBR is enabled.
            noise_floor_cap: 1.0,
            transient_next_frame: false,
            current_transient_frame: false,
        }
    }
}

impl LowDelaySbrCodingState {
    pub(crate) fn with_noise_floor_cap(mut self, cap: f64) -> Self {
        self.noise_floor_cap = cap;
        self
    }
}

#[derive(Debug, Clone, Default)]
struct InverseFilterBandState {
    orig_quota_history: [f64; 3],
    sbr_quota_history: [f64; 3],
    previous_orig_region: usize,
    previous_sbr_region: usize,
}

impl SbrEncoderAnalysisFrame {
    pub(crate) fn select_low_delay_amp_resolution(&mut self, bitrate: u32) {
        self.low_delay_amp_resolution =
            (self.envelopes.len() == 1 && self.low_delay_transient_position.is_none()).then(|| {
                if bitrate < 28_000 {
                    true
                } else if bitrate > 48_000 {
                    false
                } else {
                    self.low_delay_global_tonality.unwrap_or(0.0) <= 75.0
                }
            });
    }

    pub(crate) fn uses_low_delay_coupling(left: &Self, right: &Self) -> bool {
        low_delay_grids_match(left, right)
            && left.low_delay_amp_resolution == right.low_delay_amp_resolution
            && stereo_correlation(left, right) > 0.8
    }
    pub(crate) fn prepare_coupled_low_delay_coding(
        left: &mut Self,
        right: &mut Self,
        header: &LdSbrHeader,
        tables: &LdSbrFrequencyTables,
        header_present: bool,
        level_state: &mut LowDelaySbrCodingState,
        balance_state: &mut LowDelaySbrCodingState,
    ) -> Result<(), SbrEncoderError> {
        if left.envelopes.len() != right.envelopes.len() {
            return Err(SbrEncoderError::EnvelopeLayoutMismatch);
        }
        update_low_delay_transient_frame(left, level_state);
        update_low_delay_transient_frame(right, balance_state);
        left.low_delay_invf_modes = Some(estimate_low_delay_inverse_filtering(
            left,
            tables,
            level_state,
        ));
        right.low_delay_invf_modes = Some(estimate_low_delay_inverse_filtering(
            right,
            tables,
            balance_state,
        ));
        let amp_resolution = left
            .low_delay_amp_resolution
            .unwrap_or(header.amp_resolution);
        let level_frequency = if amp_resolution {
            SbrHuffmanBook::EnvelopeLevel30Frequency
        } else {
            SbrHuffmanBook::EnvelopeLevel15Frequency
        };
        let level_time = if amp_resolution {
            SbrHuffmanBook::EnvelopeLevel30Time
        } else {
            SbrHuffmanBook::EnvelopeLevel15Time
        };
        let balance_frequency = if amp_resolution {
            SbrHuffmanBook::EnvelopeBalance30Frequency
        } else {
            SbrHuffmanBook::EnvelopeBalance15Frequency
        };
        let balance_time = if amp_resolution {
            SbrHuffmanBook::EnvelopeBalance30Time
        } else {
            SbrHuffmanBook::EnvelopeBalance15Time
        };
        let divisor = if amp_resolution { 1.0 } else { 2.0 };
        let maximum = if amp_resolution { 63.0 } else { 127.0 };
        let mut levels = Vec::with_capacity(left.envelopes.len());
        let mut balances = Vec::with_capacity(left.envelopes.len());
        for (left_envelope, right_envelope) in left.envelopes.iter().zip(&right.envelopes) {
            if left_envelope.bands.len() != right_envelope.bands.len() {
                return Err(SbrEncoderError::EnvelopeLayoutMismatch);
            }
            let slots = (left_envelope.end_slot - left_envelope.start_slot).max(1) as f64;
            let mut level = Vec::with_capacity(left_envelope.bands.len());
            let mut balance = Vec::with_capacity(left_envelope.bands.len());
            for (left_band, right_band) in left_envelope.bands.iter().zip(&right_envelope.bands) {
                let average = (left_band.energy + right_band.energy) / (2.0 * slots);
                level.push(if average <= f64::EPSILON {
                    0
                } else {
                    ((average * 16384.0).log2() * divisor)
                        .round()
                        .clamp(0.0, maximum) as i8
                });
                let ratio = (left_band.energy + 1.0e-20) / (right_band.energy + 1.0e-20);
                balance.push(
                    ((divisor * (12.0 + ratio.log2())).round() as i16).clamp(0, 30) as i8 & !1,
                );
            }
            constrain_frequency_deltas(&mut level, level_frequency);
            constrain_scaled_frequency_deltas(&mut balance, balance_frequency, 2);
            levels.push(level);
            balances.push(balance);
        }
        left.low_delay_envelope_coding = Some(prepare_low_delay_delta_coding(
            &levels,
            &mut level_state.previous_envelope,
            level_frequency,
            level_time,
            if amp_resolution { 6 } else { 7 },
            1,
            header_present,
            low_delay_envelope_time_weight_q15(level_state.first_envelope_time_streak),
        ));
        level_state.first_envelope_time_streak = if left
            .low_delay_envelope_coding
            .as_ref()
            .and_then(|coding| coding.first())
            .is_some_and(|coding| coding.time)
        {
            level_state.first_envelope_time_streak.saturating_add(1)
        } else {
            0
        };
        right.low_delay_envelope_coding = Some(prepare_low_delay_delta_coding(
            &balances,
            &mut balance_state.previous_envelope,
            balance_frequency,
            balance_time,
            if amp_resolution { 5 } else { 6 },
            2,
            header_present,
            low_delay_envelope_time_weight_q15(balance_state.first_envelope_time_streak),
        ));
        balance_state.first_envelope_time_streak = if right
            .low_delay_envelope_coding
            .as_ref()
            .and_then(|coding| coding.first())
            .is_some_and(|coding| coding.time)
        {
            balance_state.first_envelope_time_streak.saturating_add(1)
        } else {
            0
        };
        Self::synchronize_stereo_time_streak(left, right, level_state, balance_state);

        let left_noise = estimate_low_delay_noise_levels(left, tables, level_state);
        let right_noise = estimate_low_delay_noise_levels(right, tables, balance_state);
        level_state.previous_invf_modes = left.low_delay_invf_modes.clone().unwrap_or_default();
        balance_state.previous_invf_modes = right.low_delay_invf_modes.clone().unwrap_or_default();
        let (level_noise, balance_noise) =
            couple_low_delay_noise_levels(&left_noise, &right_noise)?;
        left.low_delay_noise_coding = Some(prepare_low_delay_delta_coding(
            &level_noise,
            &mut level_state.previous_noise,
            SbrHuffmanBook::EnvelopeLevel30Frequency,
            SbrHuffmanBook::NoiseLevelTime,
            5,
            1,
            header_present,
            32_768,
        ));
        right.low_delay_noise_coding = Some(prepare_low_delay_delta_coding(
            &balance_noise,
            &mut balance_state.previous_noise,
            SbrHuffmanBook::EnvelopeBalance30Frequency,
            SbrHuffmanBook::NoiseBalanceTime,
            5,
            2,
            header_present,
            32_768,
        ));
        Ok(())
    }

    pub(crate) fn synchronize_stereo_time_streak(
        left: &Self,
        right: &Self,
        left_state: &mut LowDelaySbrCodingState,
        right_state: &mut LowDelaySbrCodingState,
    ) {
        let left_time = left
            .low_delay_envelope_coding
            .as_ref()
            .and_then(|coding| coding.first())
            .is_some_and(|coding| coding.time);
        let right_time = right
            .low_delay_envelope_coding
            .as_ref()
            .and_then(|coding| coding.first())
            .is_some_and(|coding| coding.time);
        let streak = if left_time || right_time {
            left_state
                .first_envelope_time_streak
                .max(right_state.first_envelope_time_streak)
        } else {
            0
        };
        left_state.first_envelope_time_streak = streak;
        right_state.first_envelope_time_streak = streak;
    }

    pub(crate) fn prepare_mono_low_delay_coding(
        &mut self,
        header: &LdSbrHeader,
        tables: &LdSbrFrequencyTables,
        header_present: bool,
        state: &mut LowDelaySbrCodingState,
    ) {
        update_low_delay_transient_frame(self, state);
        self.low_delay_invf_modes = Some(estimate_low_delay_inverse_filtering(self, tables, state));
        let amp_resolution = self
            .low_delay_amp_resolution
            .unwrap_or(header.amp_resolution);
        let frequency_book = if amp_resolution {
            SbrHuffmanBook::EnvelopeLevel30Frequency
        } else {
            SbrHuffmanBook::EnvelopeLevel15Frequency
        };
        let time_book = if amp_resolution {
            SbrHuffmanBook::EnvelopeLevel30Time
        } else {
            SbrHuffmanBook::EnvelopeLevel15Time
        };
        let scale = if amp_resolution { 1.0 } else { 2.0 };
        let maximum = if amp_resolution { 63.0 } else { 127.0 };
        let start_bits = if amp_resolution { 6 } else { 7 };
        let mut previous = state.previous_envelope.clone();
        let mut coding = Vec::with_capacity(self.envelopes.len());
        for (index, envelope) in self.envelopes.iter().enumerate() {
            let shortened = usize::from(
                index == 0
                    && self
                        .low_delay_transient_position
                        .is_some_and(|position| position >= 2)
                    && envelope.end_slot - envelope.start_slot > 2,
            ) * 2;
            let slots = (envelope.end_slot - envelope.start_slot - shortened).max(1) as f64;
            let frequency_table = if self
                .low_delay_frequency_resolution
                .as_ref()
                .and_then(|values| values.get(index))
                .copied()
                .unwrap_or(true)
            {
                &tables.high
            } else {
                &tables.low
            };
            let mut values = if let Some(fixed) = self
                .low_delay_prequant_debug
                .as_ref()
                .filter(|fixed| index < fixed.energies.len() && index < fixed.counts.len())
            {
                fixed.energies[index]
                    .iter()
                    .zip(&fixed.counts[index])
                    .map(|(&energy, &count)| {
                        fixed_quantize_sbr_energy(
                            energy,
                            count as usize,
                            fixed.common_scale,
                            !amp_resolution,
                        )
                        .clamp(0, maximum as i32) as i8
                    })
                    .collect::<Vec<_>>()
            } else {
                envelope
                    .bands
                    .iter()
                    .zip(frequency_table.windows(2))
                    .map(|(band, range)| {
                        if band.energy <= f64::EPSILON {
                            0
                        } else {
                            let width = f64::from(range[1] - range[0]);
                            let quantized = ((band.energy / (slots * width))
                                .log2()
                                .mul_add(scale, 26.0 * scale))
                            .floor();
                            (quantized - if width >= 4.0 { 1.0 } else { 0.0 }).clamp(0.0, maximum)
                                as i8
                        }
                    })
                    .collect::<Vec<_>>()
            };
            constrain_frequency_deltas(&mut values, frequency_book);
            let mut frequency = Vec::with_capacity(values.len());
            frequency.push(values[0]);
            frequency.extend(values.windows(2).map(|pair| pair[1] - pair[0]));
            let frequency_bits = start_bits
                + frequency[1..]
                    .iter()
                    .map(|&delta| encode_sbr_huffman(frequency_book, delta).unwrap().len())
                    .sum::<usize>();
            let time = previous
                .as_ref()
                .filter(|old| old.len() == values.len())
                .and_then(|old| {
                    let deltas = values
                        .iter()
                        .zip(old)
                        .map(|(&current, &old)| current - old)
                        .collect::<Vec<_>>();
                    let bits = deltas
                        .iter()
                        .map(|&delta| encode_sbr_huffman(time_book, delta).map(|code| code.len()))
                        .collect::<Option<Vec<_>>>()?
                        .into_iter()
                        .sum::<usize>();
                    let threshold = if index == 0 {
                        low_delay_first_time_threshold(
                            bits,
                            low_delay_envelope_time_weight_q15(state.first_envelope_time_streak),
                        )
                    } else {
                        bits
                    };
                    (!header_present && frequency_bits > threshold).then_some(deltas)
                });
            coding.push(if let Some(deltas) = time {
                LowDelayEnvelopeCoding { time: true, deltas }
            } else {
                LowDelayEnvelopeCoding {
                    time: false,
                    deltas: frequency,
                }
            });
            previous = Some(values);
        }
        state.previous_envelope = previous;
        self.low_delay_envelope_coding = Some(coding);
        state.first_envelope_time_streak = if self
            .low_delay_envelope_coding
            .as_ref()
            .and_then(|coding| coding.first())
            .is_some_and(|coding| coding.time)
        {
            state.first_envelope_time_streak.saturating_add(1)
        } else {
            0
        };

        let noise_values = estimate_low_delay_noise_levels(self, tables, state);
        state.previous_invf_modes = self.low_delay_invf_modes.clone().unwrap_or_default();
        self.low_delay_noise_coding = (!noise_values.is_empty()).then(|| {
            let mut previous_noise = state.previous_noise.clone();
            let mut result = Vec::with_capacity(noise_values.len());
            for (index, noise) in noise_values.iter().enumerate() {
                let mut frequency = Vec::with_capacity(noise.len());
                frequency.push(noise[0]);
                frequency.extend(noise.windows(2).map(|pair| pair[1] - pair[0]));
                let frequency_bits = 5 + frequency[1..]
                    .iter()
                    .map(|&delta| {
                        encode_sbr_huffman(SbrHuffmanBook::EnvelopeLevel30Frequency, delta)
                            .unwrap()
                            .len()
                    })
                    .sum::<usize>();
                let time_deltas = previous_noise.as_ref().and_then(|old| {
                    (old.len() == noise.len()).then(|| {
                        noise
                            .iter()
                            .zip(old)
                            .map(|(&new, &old)| new - old)
                            .collect::<Vec<_>>()
                    })
                });
                let time_bits = time_deltas.as_ref().and_then(|deltas| {
                    deltas
                        .iter()
                        .map(|&delta| {
                            encode_sbr_huffman(SbrHuffmanBook::NoiseLevelTime, delta)
                                .map(|code| code.len())
                        })
                        .collect::<Option<Vec<_>>>()
                        .map(|bits| bits.into_iter().sum::<usize>())
                });
                let time = previous_noise
                    .as_ref()
                    .is_some_and(|old| old.len() == noise.len())
                    && (!header_present || index > 0)
                    && time_bits.is_some_and(|bits| {
                        frequency_bits > if index == 0 { (bits + 1) / 2 } else { bits }
                    });
                result.push(LowDelayEnvelopeCoding {
                    time,
                    deltas: if time {
                        time_deltas.expect("time coding has matching history")
                    } else {
                        frequency
                    },
                });
                previous_noise = Some(noise.clone());
            }
            state.previous_noise = previous_noise;
            result
        });
    }

    pub fn write_stereo_low_delay_payload(
        left: &Self,
        right: &Self,
        writer: &mut BitWriter,
        header: &LdSbrHeader,
        tables: &LdSbrFrequencyTables,
        header_present: bool,
        crc_present: bool,
    ) -> Result<(), SbrEncoderError> {
        let mut payload = BitWriter::new();
        payload.write_bool(header_present);
        if header_present {
            header.write(&mut payload)?;
        }
        payload.write_bool(false); // bs_data_extra
        let coupling = Self::uses_low_delay_coupling(left, right);
        payload.write_bool(coupling);
        write_low_delay_grid(&mut payload, left, header, tables)?;
        if !coupling {
            write_low_delay_grid(&mut payload, right, header, tables)?;
        }
        for frame in [left, right] {
            let envelopes = frame.envelopes.len();
            for index in 0..envelopes {
                payload.write_bool(
                    frame
                        .low_delay_envelope_coding
                        .as_ref()
                        .is_some_and(|coding| coding[index].time),
                );
            }
            for index in 0..if envelopes == 1 { 1 } else { 2 } {
                payload.write_bool(
                    frame
                        .low_delay_noise_coding
                        .as_ref()
                        .is_some_and(|coding| coding[index].time),
                );
            }
        }
        write_low_delay_invf(&mut payload, left, tables);
        if coupling {
            write_low_delay_coupled_values(&mut payload, left, right, header, tables)?;
        } else {
            write_low_delay_invf(&mut payload, right, tables);
            write_low_delay_envelopes(&mut payload, left, header)?;
            write_low_delay_envelopes(&mut payload, right, header)?;
            if let Some(coding) = &left.low_delay_noise_coding {
                write_low_delay_noise_coding(&mut payload, coding)?;
            } else {
                write_constant_noise(
                    &mut payload,
                    if left.envelopes.len() == 1 { 1 } else { 2 },
                    tables,
                )?;
            }
            if let Some(coding) = &right.low_delay_noise_coding {
                write_low_delay_noise_coding(&mut payload, coding)?;
            } else {
                write_constant_noise(
                    &mut payload,
                    if right.envelopes.len() == 1 { 1 } else { 2 },
                    tables,
                )?;
            }
        }
        write_low_delay_harmonics(&mut payload, left, tables);
        write_low_delay_harmonics(&mut payload, right, tables);
        payload.write_bool(false);
        append_low_delay_payload(writer, payload, crc_present);
        Ok(())
    }

    pub fn write_mono_low_delay_payload_with_crc(
        &self,
        writer: &mut BitWriter,
        header: &LdSbrHeader,
        tables: &LdSbrFrequencyTables,
        header_present: bool,
        crc_present: bool,
    ) -> Result<(), SbrEncoderError> {
        if !crc_present {
            return self.write_mono_low_delay_payload(writer, header, tables, header_present);
        }
        let mut payload = BitWriter::new();
        self.write_mono_low_delay_payload(&mut payload, header, tables, header_present)?;
        append_low_delay_payload(writer, payload, true);
        Ok(())
    }

    /// Write an ELD low-delay SBR single-channel payload directly into an
    /// ER access unit.  Unlike ordinary SBR this payload is not wrapped in a
    /// fill element and does not carry an `EXT_SBR_DATA` nibble.
    pub fn write_mono_low_delay_payload(
        &self,
        writer: &mut BitWriter,
        header: &LdSbrHeader,
        tables: &LdSbrFrequencyTables,
        header_present: bool,
    ) -> Result<(), SbrEncoderError> {
        let envelopes = self.envelopes.len();
        let resolutions = self
            .low_delay_frequency_resolution
            .clone()
            .unwrap_or_else(|| vec![true; envelopes]);
        if !matches!(envelopes, 1 | 2 | 3 | 4)
            || resolutions.len() != envelopes
            || self
                .envelopes
                .iter()
                .zip(&resolutions)
                .any(|(envelope, &high)| {
                    envelope.bands.len()
                        != if high {
                            tables.high_band_count()
                        } else {
                            tables.low_band_count()
                        }
                })
        {
            return Err(SbrEncoderError::EnvelopeLayoutMismatch);
        }

        writer.write_bool(header_present);
        if header_present {
            header.write(writer)?;
        }
        writer.write_bool(false); // bs_data_extra

        // AAC-ELD FIXFIXonly grid: one class bit, two exponent bits, an
        // optional one-envelope amplitude resolution and one shared frequency
        // resolution bit.  The 15/16-slot borders are implicit.
        if let Some(position) = self.low_delay_transient_position {
            if position as usize >= self.slots.len() {
                return Err(SbrEncoderError::EnvelopeLayoutMismatch);
            }
            writer.write_bool(true);
            writer.write(position as u32, 4);
            for &high in &resolutions {
                writer.write_bool(high);
            }
        } else {
            if !envelopes.is_power_of_two() {
                return Err(SbrEncoderError::EnvelopeLayoutMismatch);
            }
            writer.write_bool(false);
            writer.write(envelopes.trailing_zeros(), 2);
            if envelopes == 1 {
                writer.write_bool(header.amp_resolution);
            }
            writer.write_bool(resolutions[0]);
        }

        for index in 0..envelopes {
            writer.write_bool(
                self.low_delay_envelope_coding
                    .as_ref()
                    .is_some_and(|coding| coding[index].time),
            );
        }
        let noise_envelopes = if envelopes == 1 { 1 } else { 2 };
        for index in 0..noise_envelopes {
            writer.write_bool(
                self.low_delay_noise_coding
                    .as_ref()
                    .is_some_and(|coding| coding[index].time),
            );
        }
        write_low_delay_invf(writer, self, tables);

        let envelope_book = if header.amp_resolution {
            SbrHuffmanBook::EnvelopeLevel30Frequency
        } else {
            SbrHuffmanBook::EnvelopeLevel15Frequency
        };
        let scale = if header.amp_resolution { 1.0 } else { 2.0 };
        let maximum = if header.amp_resolution { 63.0 } else { 127.0 };
        if let Some(coding) = &self.low_delay_envelope_coding {
            let time_book = if header.amp_resolution {
                SbrHuffmanBook::EnvelopeLevel30Time
            } else {
                SbrHuffmanBook::EnvelopeLevel15Time
            };
            for envelope in coding {
                if envelope.time {
                    for &delta in &envelope.deltas {
                        write_sbr_code(writer, time_book, delta)?;
                    }
                } else {
                    writer.write(
                        envelope.deltas[0] as u32,
                        if header.amp_resolution { 6 } else { 7 },
                    );
                    for &delta in &envelope.deltas[1..] {
                        write_sbr_code(writer, envelope_book, delta)?;
                    }
                }
            }
        } else {
            let quantized = self
                .envelopes
                .iter()
                .map(|envelope| {
                    let slots = (envelope.end_slot - envelope.start_slot).max(1) as f64;
                    let mut values = envelope
                        .bands
                        .iter()
                        .map(|band| {
                            if band.energy <= f64::EPSILON {
                                0
                            } else {
                                ((band.energy / slots * 16384.0).log2() * scale)
                                    .round()
                                    .clamp(0.0, maximum) as i8
                            }
                        })
                        .collect::<Vec<_>>();
                    constrain_frequency_deltas(&mut values, envelope_book);
                    values
                })
                .collect::<Vec<_>>();
            write_quantized_envelopes(writer, &quantized, header, envelope_book)?;
        }
        if let Some(coding) = &self.low_delay_noise_coding {
            write_low_delay_noise_coding(writer, coding)?;
        } else {
            write_constant_noise(writer, noise_envelopes, tables)?;
        }
        write_low_delay_harmonics(writer, self, tables);
        writer.write_bool(false); // no extended data
        Ok(())
    }

    /// Encode two uncoupled channels with shared SBR header signalling.
    pub fn write_stereo_fill_element(
        left: &Self,
        right: &Self,
        header: &LdSbrHeader,
        tables: &LdSbrFrequencyTables,
        header_present: bool,
    ) -> Result<Vec<u8>, SbrEncoderError> {
        let envelopes = left.envelopes.len();
        if !matches!(envelopes, 1 | 2)
            || right.envelopes.len() != envelopes
            || left
                .envelopes
                .iter()
                .chain(&right.envelopes)
                .any(|env| env.bands.len() != tables.high_band_count())
        {
            return Err(SbrEncoderError::EnvelopeLayoutMismatch);
        }
        if stereo_correlation(left, right) > 0.8 {
            return write_coupled_stereo_fill(left, right, header, tables, header_present);
        }
        let book = if header.amp_resolution {
            SbrHuffmanBook::EnvelopeLevel30Frequency
        } else {
            SbrHuffmanBook::EnvelopeLevel15Frequency
        };
        let quantize = |frame: &Self| {
            let scale = if header.amp_resolution { 1.0 } else { 2.0 };
            let maximum = if header.amp_resolution { 63.0 } else { 127.0 };
            frame
                .envelopes
                .iter()
                .map(|envelope| {
                    let slots = (envelope.end_slot - envelope.start_slot).max(1) as f64;
                    let mut values = envelope
                        .bands
                        .iter()
                        .map(|band| {
                            if band.energy <= f64::EPSILON {
                                0
                            } else {
                                ((band.energy / slots * 16384.0).log2() * scale)
                                    .round()
                                    .clamp(0.0, maximum) as i8
                            }
                        })
                        .collect::<Vec<_>>();
                    constrain_frequency_deltas(&mut values, book);
                    values
                })
                .collect::<Vec<_>>()
        };
        let left_values = quantize(left);
        let right_values = quantize(right);
        let mut body = BitWriter::new();
        body.write(EXT_SBR_DATA as u32, 4);
        body.write_bool(header_present);
        if header_present {
            header.write(&mut body)?;
        }
        body.write_bool(false); // bs_data_extra
        body.write_bool(false); // coupling
        for _ in 0..2 {
            body.write(0, 2); // FIXFIX
            body.write(if envelopes == 1 { 0 } else { 1 }, 2);
            body.write_bool(true);
        }
        for _ in 0..2 {
            for _ in 0..envelopes {
                body.write_bool(false);
            }
            for _ in 0..envelopes {
                body.write_bool(false);
            }
        }
        for frame in [left, right] {
            write_invf_from_tonality(&mut body, frame, tables);
        }
        for values in [&left_values, &right_values] {
            write_quantized_envelopes(&mut body, values, header, book)?;
        }
        for _ in 0..2 {
            write_constant_noise(&mut body, envelopes, tables)?;
        }
        write_harmonics(&mut body, left);
        write_harmonics(&mut body, right);
        body.write_bool(false);
        pack_fill_body(body)
    }

    /// Encode an ordinary-SBR mono fill element using a one- or two-envelope
    /// FIXFIX grid and frequency-domain deltas.
    pub fn write_mono_fill_element(
        &self,
        header: &LdSbrHeader,
        tables: &LdSbrFrequencyTables,
        header_present: bool,
    ) -> Result<Vec<u8>, SbrEncoderError> {
        self.write_mono_fill_element_with_extension(header, tables, header_present, None)
    }

    pub fn write_mono_fill_element_with_extension(
        &self,
        header: &LdSbrHeader,
        tables: &LdSbrFrequencyTables,
        header_present: bool,
        extended_data: Option<&[u8]>,
    ) -> Result<Vec<u8>, SbrEncoderError> {
        if !matches!(self.envelopes.len(), 1 | 2)
            || self
                .envelopes
                .iter()
                .any(|env| env.bands.len() != tables.high_band_count())
        {
            return Err(SbrEncoderError::EnvelopeLayoutMismatch);
        }
        let mut tonalities = vec![0.0; tables.high_band_count()];
        for envelope in &self.envelopes {
            for (band, value) in envelope.bands.iter().enumerate() {
                tonalities[band] += value.tonality;
            }
        }
        let normalizer = self.envelopes.len() as f64;
        let envelope_book = if header.amp_resolution {
            SbrHuffmanBook::EnvelopeLevel30Frequency
        } else {
            SbrHuffmanBook::EnvelopeLevel15Frequency
        };
        let scale = if header.amp_resolution { 1.0 } else { 2.0 };
        let maximum = if header.amp_resolution { 63.0 } else { 127.0 };
        let quantized = self
            .envelopes
            .iter()
            .map(|envelope| {
                let slots = (envelope.end_slot - envelope.start_slot).max(1) as f64;
                let mut values = envelope
                    .bands
                    .iter()
                    .map(|band| {
                        if band.energy <= f64::EPSILON {
                            0
                        } else {
                            ((band.energy / slots * 16384.0).log2() * scale)
                                .round()
                                .clamp(0.0, maximum) as i8
                        }
                    })
                    .collect::<Vec<_>>();
                constrain_frequency_deltas(&mut values, envelope_book);
                values
            })
            .collect::<Vec<_>>();

        let mut body = BitWriter::new();
        body.write(EXT_SBR_DATA as u32, 4);
        body.write_bool(header_present);
        if header_present {
            header.write(&mut body)?;
        }
        body.write_bool(false); // bs_data_extra
        write_mono_grid(&mut body, self)?;
        for _ in &self.envelopes {
            body.write_bool(false); // envelope frequency direction
        }
        for _ in &self.envelopes {
            body.write_bool(false); // noise frequency direction
        }
        for noise in tables.noise.windows(2) {
            let high = tables
                .high
                .windows(2)
                .enumerate()
                .filter(|(_, high)| high[0] >= noise[0] && high[1] <= noise[1])
                .map(|(index, _)| tonalities[index] / normalizer)
                .fold(0.0, f64::max);
            body.write(if high > 0.6 { 2 } else { 1 }, 2);
        }
        for values in &quantized {
            body.write(values[0] as u32, if header.amp_resolution { 6 } else { 7 });
            for pair in values.windows(2) {
                write_sbr_code(&mut body, envelope_book, pair[1] - pair[0])?;
            }
        }
        for _ in &self.envelopes {
            let mut noise = vec![7i8; tables.noise_band_count()];
            constrain_frequency_deltas(&mut noise, SbrHuffmanBook::EnvelopeLevel30Frequency);
            body.write(noise[0] as u32, 5);
            for pair in noise.windows(2) {
                write_representable_sbr_code(
                    &mut body,
                    SbrHuffmanBook::EnvelopeLevel30Frequency,
                    pair[1] - pair[0],
                );
            }
        }
        write_harmonics(&mut body, self);
        if let Some(extension) = extended_data {
            if extension.len() > 269 {
                return Err(SbrEncoderError::PayloadTooLarge(extension.len()));
            }
            body.write_bool(true);
            if extension.len() < 15 {
                body.write(extension.len() as u32, 4);
            } else {
                body.write(15, 4);
                body.write((extension.len() - 15) as u32, 8);
            }
            for &byte in extension {
                body.write(byte as u32, 8);
            }
        } else {
            body.write_bool(false);
        }
        pack_fill_body(body)
    }
}

fn make_sbr_patch_map(
    tables: &LdSbrFrequencyTables,
    sampling_frequency: u32,
    qmf_channels: usize,
) -> Vec<Option<usize>> {
    let mut result = (0..qmf_channels).map(Some).collect::<Vec<_>>();
    if tables.master.len() < 2 || sampling_frequency == 0 {
        return result;
    }
    let lsb = usize::from(tables.master[0]);
    let usb = usize::from(*tables.master.last().unwrap());
    let high_start = usize::from(tables.high[0]);
    let crossover_offset = high_start.saturating_sub(lsb);
    let closest = |goal: usize, up: bool| {
        tables
            .master
            .iter()
            .map(|&value| usize::from(value))
            .filter(|&value| if up { value >= goal } else { value <= goal })
            .min_by_key(|&value| value.abs_diff(goal))
            .unwrap_or(if up { usb } else { lsb })
    };
    let rounded_16khz = ((2_u64 * qmf_channels as u64 * 16_000 + u64::from(sampling_frequency / 2))
        / u64::from(sampling_frequency)) as usize;
    let mut goal = closest(rounded_16khz, true);
    let mut source_start = 1 + crossover_offset;
    let mut target_stop = lsb + crossover_offset;
    let mut patches = Vec::new();
    while target_stop < usb && patches.len() < 6 {
        let target_start = target_stop;
        let mut bands = goal.saturating_sub(target_stop);
        if bands >= lsb.saturating_sub(source_start) {
            let distance = target_stop.saturating_sub(source_start) & !1;
            bands = lsb.saturating_sub(target_stop.saturating_sub(distance));
            bands = closest(target_stop + bands, false).saturating_sub(target_stop);
        }
        let distance = (bands + target_stop).saturating_sub(lsb).div_ceil(2) * 2;
        if bands == 0 || distance > target_start {
            break;
        }
        let patch_source = target_start - distance;
        patches.push((patch_source, target_start, bands));
        target_stop += bands;
        source_start = 1;
        if target_stop.abs_diff(goal) < 3 {
            goal = usb;
        }
    }
    if patches.len() > 1 && patches.last().is_some_and(|patch| patch.2 < 3) {
        patches.pop();
    }
    for target in high_start..usb.min(result.len()) {
        result[target] = None;
    }
    for (source, target, bands) in patches {
        for offset in 0..bands {
            if target + offset < result.len() {
                result[target + offset] = Some(source + offset);
            }
        }
    }
    result
}

fn stereo_correlation(left: &SbrEncoderAnalysisFrame, right: &SbrEncoderAnalysisFrame) -> f64 {
    let mut left_energy = 0.0;
    let mut right_energy = 0.0;
    let mut real = 0.0;
    let mut imaginary = 0.0;
    for (left, right) in left.slots.iter().zip(&right.slots) {
        for band in 0..left.real.len().min(right.real.len()) {
            let (lr, li) = (left.real[band], left.imaginary[band]);
            let (rr, ri) = (right.real[band], right.imaginary[band]);
            left_energy += lr * lr + li * li;
            right_energy += rr * rr + ri * ri;
            real += lr * rr + li * ri;
            imaginary += li * rr - lr * ri;
        }
    }
    let denominator = (left_energy * right_energy).sqrt();
    if denominator <= f64::EPSILON {
        1.0
    } else {
        (real.hypot(imaginary) / denominator).clamp(0.0, 1.0)
    }
}

fn write_coupled_stereo_fill(
    left: &SbrEncoderAnalysisFrame,
    right: &SbrEncoderAnalysisFrame,
    header: &LdSbrHeader,
    tables: &LdSbrFrequencyTables,
    header_present: bool,
) -> Result<Vec<u8>, SbrEncoderError> {
    let envelope_count = left.envelopes.len();
    let level_book = if header.amp_resolution {
        SbrHuffmanBook::EnvelopeLevel30Frequency
    } else {
        SbrHuffmanBook::EnvelopeLevel15Frequency
    };
    let balance_book = if header.amp_resolution {
        SbrHuffmanBook::EnvelopeBalance30Frequency
    } else {
        SbrHuffmanBook::EnvelopeBalance15Frequency
    };
    let divisor = if header.amp_resolution { 1.0 } else { 2.0 };
    let level_maximum = if header.amp_resolution { 63.0 } else { 127.0 };
    let mut levels = Vec::with_capacity(envelope_count);
    let mut balances = Vec::with_capacity(envelope_count);
    for (left, right) in left.envelopes.iter().zip(&right.envelopes) {
        let slots = (left.end_slot - left.start_slot).max(1) as f64;
        let mut level = Vec::with_capacity(left.bands.len());
        let mut balance = Vec::with_capacity(left.bands.len());
        for (left, right) in left.bands.iter().zip(&right.bands) {
            let average = (left.energy + right.energy) / (2.0 * slots);
            level.push(if average <= f64::EPSILON {
                0
            } else {
                ((average * 16384.0).log2() * divisor)
                    .round()
                    .clamp(0.0, level_maximum) as i8
            });
            let ratio = (left.energy + 1.0e-20) / (right.energy + 1.0e-20);
            let value = (divisor * (12.0 + ratio.log2())).round();
            balance.push(((value as i16).clamp(0, 30) & !1) as i8);
        }
        constrain_frequency_deltas(&mut level, level_book);
        constrain_scaled_frequency_deltas(&mut balance, balance_book, 2);
        levels.push(level);
        balances.push(balance);
    }

    let mut body = BitWriter::new();
    body.write(EXT_SBR_DATA as u32, 4);
    body.write_bool(header_present);
    if header_present {
        header.write(&mut body)?;
    }
    body.write_bool(false);
    body.write_bool(true); // coupling
    body.write(0, 2);
    body.write(if envelope_count == 1 { 0 } else { 1 }, 2);
    body.write_bool(true);
    for _ in 0..2 {
        for _ in 0..envelope_count {
            body.write_bool(false);
        }
        for _ in 0..envelope_count {
            body.write_bool(false);
        }
    }
    write_invf_from_tonality(&mut body, left, tables);
    write_quantized_envelopes(&mut body, &levels, header, level_book)?;
    write_constant_noise(&mut body, envelope_count, tables)?;
    write_scaled_envelopes(&mut body, &balances, header, balance_book, 2)?;
    for _ in 0..envelope_count {
        let values = vec![12i8; tables.noise_band_count()];
        body.write(6, 5);
        for pair in values.windows(2) {
            write_representable_sbr_code(
                &mut body,
                SbrHuffmanBook::EnvelopeBalance30Frequency,
                (pair[1] - pair[0]) / 2,
            );
        }
    }
    write_harmonics(&mut body, left);
    write_harmonics(&mut body, right);
    body.write_bool(false);
    pack_fill_body(body)
}

fn constrain_scaled_frequency_deltas(values: &mut [i8], book: SbrHuffmanBook, _scale: i8) {
    constrain_frequency_deltas(values, book);
}

fn write_scaled_envelopes(
    writer: &mut BitWriter,
    values: &[Vec<i8>],
    header: &LdSbrHeader,
    book: SbrHuffmanBook,
    scale: i8,
) -> Result<(), SbrEncoderError> {
    for values in values {
        writer.write(
            (values[0] / scale) as u32,
            if header.amp_resolution { 5 } else { 6 },
        );
        for pair in values.windows(2) {
            write_sbr_code(writer, book, (pair[1] - pair[0]) / scale)?;
        }
    }
    Ok(())
}

fn write_invf_from_tonality(
    writer: &mut BitWriter,
    frame: &SbrEncoderAnalysisFrame,
    tables: &LdSbrFrequencyTables,
) {
    for noise in tables.noise.windows(2) {
        let tonality = tables
            .high
            .windows(2)
            .enumerate()
            .filter(|(_, high)| high[0] >= noise[0] && high[1] <= noise[1])
            .flat_map(|(index, _)| {
                frame
                    .envelopes
                    .iter()
                    .map(move |env| env.bands[index].tonality)
            })
            .fold(0.0, f64::max);
        writer.write(if tonality > 0.6 { 2 } else { 1 }, 2);
    }
}

fn write_low_delay_invf(
    writer: &mut BitWriter,
    frame: &SbrEncoderAnalysisFrame,
    tables: &LdSbrFrequencyTables,
) {
    if let Some(modes) = &frame.low_delay_invf_modes {
        for &mode in modes {
            writer.write(u32::from(mode), 2);
        }
        return;
    }
    let bands = analyze_bands(&frame.slots, &tables.high);
    for noise in tables.noise.windows(2) {
        let tonality = tables
            .high
            .windows(2)
            .enumerate()
            .filter(|(_, high)| high[0] >= noise[0] && high[1] <= noise[1])
            .map(|(index, _)| bands[index].tonality)
            .fold(0.0, f64::max);
        writer.write(if tonality > 0.6 { 2 } else { 1 }, 2);
    }
}

fn low_delay_resolutions(frame: &SbrEncoderAnalysisFrame) -> Vec<bool> {
    frame
        .low_delay_frequency_resolution
        .clone()
        .unwrap_or_else(|| vec![true; frame.envelopes.len()])
}

fn low_delay_grids_match(left: &SbrEncoderAnalysisFrame, right: &SbrEncoderAnalysisFrame) -> bool {
    left.low_delay_transient_position == right.low_delay_transient_position
        && low_delay_resolutions(left) == low_delay_resolutions(right)
        && left
            .envelopes
            .iter()
            .map(|envelope| (envelope.start_slot, envelope.end_slot))
            .eq(right
                .envelopes
                .iter()
                .map(|envelope| (envelope.start_slot, envelope.end_slot)))
}

fn write_low_delay_coupled_values(
    writer: &mut BitWriter,
    left: &SbrEncoderAnalysisFrame,
    right: &SbrEncoderAnalysisFrame,
    header: &LdSbrHeader,
    tables: &LdSbrFrequencyTables,
) -> Result<(), SbrEncoderError> {
    let amp_resolution = left
        .low_delay_amp_resolution
        .unwrap_or(header.amp_resolution);
    let level_book = if amp_resolution {
        SbrHuffmanBook::EnvelopeLevel30Frequency
    } else {
        SbrHuffmanBook::EnvelopeLevel15Frequency
    };
    let balance_book = if amp_resolution {
        SbrHuffmanBook::EnvelopeBalance30Frequency
    } else {
        SbrHuffmanBook::EnvelopeBalance15Frequency
    };
    let divisor = if amp_resolution { 1.0 } else { 2.0 };
    let maximum = if amp_resolution { 63.0 } else { 127.0 };
    if let (Some(level), Some(balance), Some(level_noise), Some(balance_noise)) = (
        left.low_delay_envelope_coding.as_ref(),
        right.low_delay_envelope_coding.as_ref(),
        left.low_delay_noise_coding.as_ref(),
        right.low_delay_noise_coding.as_ref(),
    ) {
        write_prepared_low_delay_coding(
            writer,
            level,
            level_book,
            if amp_resolution {
                SbrHuffmanBook::EnvelopeLevel30Time
            } else {
                SbrHuffmanBook::EnvelopeLevel15Time
            },
            if amp_resolution { 6 } else { 7 },
        )?;
        write_prepared_low_delay_coding(
            writer,
            level_noise,
            SbrHuffmanBook::EnvelopeLevel30Frequency,
            SbrHuffmanBook::NoiseLevelTime,
            5,
        )?;
        write_prepared_low_delay_coding(
            writer,
            balance,
            balance_book,
            if amp_resolution {
                SbrHuffmanBook::EnvelopeBalance30Time
            } else {
                SbrHuffmanBook::EnvelopeBalance15Time
            },
            if amp_resolution { 5 } else { 6 },
        )?;
        write_prepared_low_delay_coding(
            writer,
            balance_noise,
            SbrHuffmanBook::EnvelopeBalance30Frequency,
            SbrHuffmanBook::NoiseBalanceTime,
            5,
        )?;
        return Ok(());
    }
    let mut levels = Vec::with_capacity(left.envelopes.len());
    let mut balances = Vec::with_capacity(left.envelopes.len());
    for (left, right) in left.envelopes.iter().zip(&right.envelopes) {
        if left.bands.len() != right.bands.len() {
            return Err(SbrEncoderError::EnvelopeLayoutMismatch);
        }
        let slots = (left.end_slot - left.start_slot).max(1) as f64;
        let mut level = Vec::with_capacity(left.bands.len());
        let mut balance = Vec::with_capacity(left.bands.len());
        for (left, right) in left.bands.iter().zip(&right.bands) {
            let average = (left.energy + right.energy) / (2.0 * slots);
            level.push(if average <= f64::EPSILON {
                0
            } else {
                ((average * 16384.0).log2() * divisor)
                    .round()
                    .clamp(0.0, maximum) as i8
            });
            let ratio = (left.energy + 1.0e-20) / (right.energy + 1.0e-20);
            balance
                .push(((divisor * (12.0 + ratio.log2())).round() as i16).clamp(0, 30) as i8 & !1);
        }
        constrain_frequency_deltas(&mut level, level_book);
        constrain_scaled_frequency_deltas(&mut balance, balance_book, 2);
        levels.push(level);
        balances.push(balance);
    }
    write_quantized_envelopes(writer, &levels, header, level_book)?;
    let noise_envelopes = if left.envelopes.len() == 1 { 1 } else { 2 };
    write_constant_noise(writer, noise_envelopes, tables)?;
    write_scaled_envelopes(writer, &balances, header, balance_book, 2)?;
    for _ in 0..noise_envelopes {
        writer.write(6, 5);
        for _ in 1..tables.noise_band_count() {
            write_representable_sbr_code(writer, SbrHuffmanBook::EnvelopeBalance30Frequency, 0);
        }
    }
    Ok(())
}

fn write_prepared_low_delay_coding(
    writer: &mut BitWriter,
    coding: &[LowDelayEnvelopeCoding],
    frequency_book: SbrHuffmanBook,
    time_book: SbrHuffmanBook,
    start_bits: usize,
) -> Result<(), SbrEncoderError> {
    for envelope in coding {
        if envelope.time {
            for &delta in &envelope.deltas {
                write_sbr_code(writer, time_book, delta)?;
            }
        } else {
            writer.write(envelope.deltas[0] as u32, start_bits);
            for &delta in &envelope.deltas[1..] {
                write_sbr_code(writer, frequency_book, delta)?;
            }
        }
    }
    Ok(())
}

fn write_low_delay_grid(
    writer: &mut BitWriter,
    frame: &SbrEncoderAnalysisFrame,
    header: &LdSbrHeader,
    tables: &LdSbrFrequencyTables,
) -> Result<(), SbrEncoderError> {
    let envelopes = frame.envelopes.len();
    let resolutions = low_delay_resolutions(frame);
    if !matches!(envelopes, 1 | 2 | 3 | 4)
        || resolutions.len() != envelopes
        || frame
            .envelopes
            .iter()
            .zip(&resolutions)
            .any(|(envelope, &high)| {
                envelope.bands.len()
                    != if high {
                        tables.high_band_count()
                    } else {
                        tables.low_band_count()
                    }
            })
    {
        return Err(SbrEncoderError::EnvelopeLayoutMismatch);
    }
    if let Some(position) = frame.low_delay_transient_position {
        if position as usize >= frame.slots.len() {
            return Err(SbrEncoderError::EnvelopeLayoutMismatch);
        }
        writer.write_bool(true);
        writer.write(position as u32, 4);
        for high in resolutions {
            writer.write_bool(high);
        }
    } else {
        if !envelopes.is_power_of_two() {
            return Err(SbrEncoderError::EnvelopeLayoutMismatch);
        }
        writer.write_bool(false);
        writer.write(envelopes.trailing_zeros(), 2);
        if envelopes == 1 {
            writer.write_bool(
                frame
                    .low_delay_amp_resolution
                    .unwrap_or(header.amp_resolution),
            );
        }
        writer.write_bool(resolutions[0]);
    }
    Ok(())
}

fn write_low_delay_envelopes(
    writer: &mut BitWriter,
    frame: &SbrEncoderAnalysisFrame,
    header: &LdSbrHeader,
) -> Result<(), SbrEncoderError> {
    let amp_resolution = frame
        .low_delay_amp_resolution
        .unwrap_or(header.amp_resolution);
    let book = if amp_resolution {
        SbrHuffmanBook::EnvelopeLevel30Frequency
    } else {
        SbrHuffmanBook::EnvelopeLevel15Frequency
    };
    if let Some(coding) = &frame.low_delay_envelope_coding {
        let time_book = if amp_resolution {
            SbrHuffmanBook::EnvelopeLevel30Time
        } else {
            SbrHuffmanBook::EnvelopeLevel15Time
        };
        for envelope in coding {
            if envelope.time {
                for &delta in &envelope.deltas {
                    write_sbr_code(writer, time_book, delta)?;
                }
            } else {
                writer.write(
                    envelope.deltas[0] as u32,
                    if amp_resolution { 6 } else { 7 },
                );
                for &delta in &envelope.deltas[1..] {
                    write_sbr_code(writer, book, delta)?;
                }
            }
        }
        return Ok(());
    }
    let scale = if amp_resolution { 1.0 } else { 2.0 };
    let maximum = if amp_resolution { 63.0 } else { 127.0 };
    let values = frame
        .envelopes
        .iter()
        .map(|envelope| {
            let slots = (envelope.end_slot - envelope.start_slot).max(1) as f64;
            let mut values = envelope
                .bands
                .iter()
                .map(|band| {
                    if band.energy <= f64::EPSILON {
                        0
                    } else {
                        ((band.energy / slots * 16384.0).log2() * scale)
                            .round()
                            .clamp(0.0, maximum) as i8
                    }
                })
                .collect::<Vec<_>>();
            constrain_frequency_deltas(&mut values, book);
            values
        })
        .collect::<Vec<_>>();
    write_quantized_envelopes(writer, &values, header, book)
}

fn append_low_delay_payload(writer: &mut BitWriter, payload: BitWriter, crc_present: bool) {
    let bit_len = payload.bits_written();
    let bytes = payload.finish();
    let mut reader = BitReader::with_bit_len(&bytes, bit_len)
        .expect("the payload writer owns all declared bits");
    if crc_present {
        writer.write(reader.crc_msb(0, bit_len, 10, 0x0633), 10);
    }
    while reader.remaining_bits() != 0 {
        writer.write_bool(reader.read_bool().expect("declared payload bit exists"));
    }
}

fn write_harmonics(writer: &mut BitWriter, frame: &SbrEncoderAnalysisFrame) {
    let bands = frame
        .envelopes
        .first()
        .map_or(0, |envelope| envelope.bands.len());
    let harmonic = (0..bands)
        .map(|band| {
            frame
                .envelopes
                .iter()
                .map(|envelope| envelope.bands[band].tonality)
                .fold(0.0, f64::max)
                > 0.85
        })
        .collect::<Vec<_>>();
    let present = harmonic.iter().any(|&enabled| enabled);
    writer.write_bool(present);
    if present {
        for enabled in harmonic {
            writer.write_bool(enabled);
        }
    }
}

fn write_low_delay_harmonics(
    writer: &mut BitWriter,
    frame: &SbrEncoderAnalysisFrame,
    tables: &LdSbrFrequencyTables,
) {
    let bands = analyze_bands(&frame.slots, &tables.high);
    let harmonic = bands
        .iter()
        .map(|band| band.tonality > 0.85)
        .collect::<Vec<_>>();
    let present = harmonic.iter().any(|&enabled| enabled);
    writer.write_bool(present);
    if present {
        for enabled in harmonic {
            writer.write_bool(enabled);
        }
    }
}

fn write_quantized_envelopes(
    writer: &mut BitWriter,
    values: &[Vec<i8>],
    header: &LdSbrHeader,
    book: SbrHuffmanBook,
) -> Result<(), SbrEncoderError> {
    for values in values {
        writer.write(values[0] as u32, if header.amp_resolution { 6 } else { 7 });
        for pair in values.windows(2) {
            write_sbr_code(writer, book, pair[1] - pair[0])?;
        }
    }
    Ok(())
}

fn write_constant_noise(
    writer: &mut BitWriter,
    envelopes: usize,
    tables: &LdSbrFrequencyTables,
) -> Result<(), SbrEncoderError> {
    for _ in 0..envelopes {
        let noise = vec![7i8; tables.noise_band_count()];
        writer.write(noise[0] as u32, 5);
        for pair in noise.windows(2) {
            write_representable_sbr_code(
                writer,
                SbrHuffmanBook::EnvelopeLevel30Frequency,
                pair[1] - pair[0],
            );
        }
    }
    Ok(())
}

fn write_low_delay_noise_coding(
    writer: &mut BitWriter,
    coding: &[LowDelayEnvelopeCoding],
) -> Result<(), SbrEncoderError> {
    for envelope in coding {
        if envelope.time {
            for &delta in &envelope.deltas {
                write_sbr_code(writer, SbrHuffmanBook::NoiseLevelTime, delta)?;
            }
        } else {
            writer.write(envelope.deltas[0] as u32, 5);
            for &delta in &envelope.deltas[1..] {
                write_sbr_code(writer, SbrHuffmanBook::EnvelopeLevel30Frequency, delta)?;
            }
        }
    }
    Ok(())
}

fn pack_fill_body(mut body: BitWriter) -> Result<Vec<u8>, SbrEncoderError> {
    body.byte_align();
    let body = body.finish();
    if body.len() > 269 {
        return Err(SbrEncoderError::PayloadTooLarge(body.len()));
    }
    let mut fill = BitWriter::new();
    if body.len() < 15 {
        fill.write(body.len() as u32, 4);
    } else {
        fill.write(15, 4);
        fill.write((body.len() - 14) as u32, 8);
    }
    for byte in body {
        fill.write(byte as u32, 8);
    }
    Ok(fill.finish())
}

fn constrain_frequency_deltas(values: &mut [i8], book: SbrHuffmanBook) {
    let lav = (0_i8..=127)
        .take_while(|&delta| encode_sbr_huffman(book, delta).is_some())
        .last()
        .unwrap_or(0);
    for index in (1..values.len()).rev() {
        if i16::from(values[index]) - i16::from(values[index - 1]) > i16::from(lav) {
            values[index - 1] = values[index] - lav;
        }
    }
    for index in 1..values.len() {
        if i16::from(values[index - 1]) - i16::from(values[index]) > i16::from(lav) {
            values[index] = values[index - 1] - lav;
        }
    }
}

fn write_sbr_code(
    writer: &mut BitWriter,
    book: SbrHuffmanBook,
    symbol: i8,
) -> Result<(), SbrEncoderError> {
    let code = encode_sbr_huffman(book, symbol)
        .ok_or(SbrEncoderError::UnrepresentableHuffmanSymbol(symbol))?;
    for bit in code {
        writer.write_bool(bit);
    }
    Ok(())
}

fn write_representable_sbr_code(writer: &mut BitWriter, book: SbrHuffmanBook, symbol: i8) {
    let code = encode_sbr_huffman(book, symbol)
        .expect("constrained SBR deltas must have an embedded Huffman codeword");
    for bit in code {
        writer.write_bool(bit);
    }
}

#[derive(Debug, Clone)]
pub struct SbrEncoderAnalysis {
    qmf: LdSbrQmfAnalysis,
    tables: LdSbrFrequencyTables,
    low_delay_detector_bandwidth: Option<f64>,
    low_delay_energy_history: [f64; 2],
    low_delay_ratio_history: [f64; 2],
    low_delay_candidate_history: [bool; 2],
    low_delay_slot_history: Vec<QmfSlot>,
    low_delay_fixed_energy_history: Vec<Vec<i32>>,
    low_delay_fixed_energy_scale: i32,
    low_delay_previous_tonality: f64,
    patch_map: Vec<Option<usize>>,
}

impl SbrEncoderAnalysis {
    pub fn new(header: &LdSbrHeader, sampling_frequency: u32) -> Result<Self, SbrEncoderError> {
        let tables = LdSbrFrequencyTables::from_header(header, sampling_frequency)?;
        Ok(Self {
            qmf: LdSbrQmfAnalysis::new_with_channels(64)?,
            patch_map: make_sbr_patch_map(&tables, sampling_frequency, 64),
            tables,
            low_delay_detector_bandwidth: None,
            low_delay_energy_history: [0.0; 2],
            low_delay_ratio_history: [0.0; 2],
            low_delay_candidate_history: [false; 2],
            low_delay_slot_history: Vec::new(),
            low_delay_fixed_energy_history: Vec::new(),
            low_delay_fixed_energy_scale: 15,
            low_delay_previous_tonality: 0.0,
        })
    }

    /// Construct the ELD analysis bank. Single-rate ELD uses the 32-band
    /// complex low-delay filter bank; dual-rate ELD uses 64 bands. Frequency
    /// tables are nevertheless derived at twice the AAC core rate in both
    /// modes, as prescribed by the ELD ASC/FDK setup.
    pub fn new_low_delay(
        header: &LdSbrHeader,
        core_sampling_frequency: u32,
        dual_rate: bool,
    ) -> Result<Self, SbrEncoderError> {
        let qmf_bands = if dual_rate { 64 } else { 32 };
        let sampling_frequency = core_sampling_frequency.saturating_mul(2);
        let tables = LdSbrFrequencyTables::from_header(header, sampling_frequency)?;
        Ok(Self {
            qmf: if dual_rate {
                LdSbrQmfAnalysis::new_cldfb(64)?
            } else {
                LdSbrQmfAnalysis::new_cldfb_32()
            },
            patch_map: make_sbr_patch_map(&tables, sampling_frequency, qmf_bands),
            tables,
            low_delay_detector_bandwidth: Some(
                f64::from(core_sampling_frequency) / qmf_bands as f64,
            ),
            low_delay_energy_history: [0.0; 2],
            low_delay_ratio_history: [0.0; 2],
            low_delay_candidate_history: [false; 2],
            low_delay_slot_history: Vec::new(),
            low_delay_fixed_energy_history: Vec::new(),
            low_delay_fixed_energy_scale: 15,
            low_delay_previous_tonality: 0.0,
        })
    }

    pub fn frequency_tables(&self) -> &LdSbrFrequencyTables {
        &self.tables
    }

    pub fn analyze(&mut self, samples: &[f32]) -> Result<SbrEncoderAnalysisFrame, SbrEncoderError> {
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err(SbrEncoderError::NonFiniteInput);
        }
        let input = samples
            .iter()
            .map(|&sample| sample as f64)
            .collect::<Vec<_>>();
        let slots = self.qmf.process_frame(&input)?;
        if slots.is_empty() {
            return Err(SbrEncoderError::EmptyFrame);
        }
        let slot_energy = slots
            .iter()
            .map(|slot| {
                slot.real
                    .iter()
                    .zip(&slot.imaginary)
                    .map(|(&real, &imaginary)| real * real + imaginary * imaginary)
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        let mean = slot_energy.iter().sum::<f64>() / slot_energy.len() as f64;
        let transient_ratio = if mean <= f64::EPSILON {
            0.0
        } else {
            slot_energy.iter().copied().fold(0.0, f64::max) / mean
        };
        let borders = if transient_ratio > 4.0 && slots.len() >= 8 {
            let peak = slot_energy
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .map(|(index, _)| index)
                .unwrap_or(slots.len() / 2);
            let time_slots = slots.len() / 2;
            let mut boundary = (peak / 2).clamp(2, time_slots.saturating_sub(2));
            boundary = (boundary + 1) & !1;
            vec![0, boundary * 2, slots.len()]
        } else {
            vec![0, slots.len()]
        };
        let envelopes = borders
            .windows(2)
            .map(|border| SbrEncoderEnvelope {
                start_slot: border[0],
                end_slot: border[1],
                bands: analyze_bands(&slots[border[0]..border[1]], &self.tables.high),
            })
            .collect();
        Ok(SbrEncoderAnalysisFrame {
            slots,
            envelopes,
            transient_ratio,
            low_delay_transient_position: None,
            low_delay_frequency_resolution: None,
            low_delay_amp_resolution: None,
            low_delay_global_tonality: None,
            low_delay_envelope_coding: None,
            low_delay_noise_coding: None,
            low_delay_invf_modes: None,
            low_delay_patch_map: Some(self.patch_map.clone()),
            low_delay_prequant_debug: None,
        })
    }

    /// Analyze one 480/512-sample ELD core interval. The QMF configuration
    /// determines whether this consumes one (single-rate) or two (dual-rate)
    /// input samples per core sample. A stable one-envelope FIXFIXonly grid is
    /// emitted here; transient LD grids are selected by the later grid stage.
    pub fn analyze_low_delay(
        &mut self,
        samples: &[f32],
        frame_length: usize,
    ) -> Result<SbrEncoderAnalysisFrame, SbrEncoderError> {
        if !matches!(frame_length, 480 | 512) {
            return Err(SbrEncoderError::UnsupportedFrameLength(frame_length));
        }
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err(SbrEncoderError::NonFiniteInput);
        }
        let input = samples
            .iter()
            .map(|&sample| sample as f64)
            .collect::<Vec<_>>();
        let current_slots = self.qmf.process_frame(&input)?;
        let expected_slots = frame_length / 32;
        if current_slots.len() != expected_slots {
            return Err(SbrEncoderError::LowDelaySlotCountMismatch {
                expected: expected_slots,
                actual: current_slots.len(),
            });
        }
        let qmf_bands = current_slots.first().map_or(0, |slot| slot.real.len());
        let required_bands = usize::from(self.tables.high.last().copied().unwrap_or(0));
        if required_bands > qmf_bands {
            return Err(SbrEncoderError::QmfBandRangeMismatch {
                available: qmf_bands,
                required: required_bands,
            });
        }
        // FDK writes the current energies at a half-frame Y-buffer offset;
        // both transient detection and envelope extraction therefore operate
        // on the previous trailing half followed by the current leading half.
        let history_slots = expected_slots / 2;
        if self.low_delay_slot_history.len() != history_slots {
            self.low_delay_slot_history = (0..history_slots)
                .map(|_| QmfSlot {
                    real: vec![0.0; qmf_bands],
                    imaginary: vec![0.0; qmf_bands],
                })
                .collect();
        }
        let current_prefix = expected_slots - history_slots;
        let mut detector_slots = self.low_delay_slot_history.clone();
        detector_slots.extend_from_slice(&current_slots[..current_prefix + 2]);
        let slots = detector_slots[..expected_slots].to_vec();
        let fixed_debug_context = (qmf_bands == 64).then(|| {
            let (current_energy, current_scale, qmf_scale) =
                fixed_cldfb64_energy_block(&current_slots);
            if self.low_delay_fixed_energy_history.len() != history_slots {
                self.low_delay_fixed_energy_history = vec![vec![0; qmf_bands]; history_slots];
            }
            let previous_scale = self.low_delay_fixed_energy_scale;
            let mut rows = self.low_delay_fixed_energy_history.clone();
            rows.extend_from_slice(&current_energy[..current_prefix]);
            let common_scale = previous_scale.min(current_scale) - 7;
            self.low_delay_fixed_energy_history =
                current_energy[expected_slots - history_slots..].to_vec();
            self.low_delay_fixed_energy_scale = current_scale;
            (rows, previous_scale, current_scale, qmf_scale, common_scale)
        });
        self.low_delay_slot_history = current_slots[expected_slots - history_slots..].to_vec();
        let detector_bandwidth = self
            .low_delay_detector_bandwidth
            .expect("low-delay analysis has detector geometry");
        let detector_stop = ((13_500.0 / detector_bandwidth) as usize).min(qmf_bands);
        let detector_start = usize::from(self.tables.high.first().copied().unwrap_or(0))
            .min(detector_stop.saturating_sub(4));
        // The fast detector preserves indices 0 and 1 internally and reads
        // indices 2..N+2 from the energy Y-buffer. Supplying this exact slice
        // avoids applying its two-slot history twice.
        let slot_energy = detector_slots[2..expected_slots + 2]
            .iter()
            .map(|slot| {
                slot.real[detector_start..detector_stop]
                    .iter()
                    .zip(&slot.imaginary[detector_start..detector_stop])
                    .enumerate()
                    .map(|(band, (&real, &imaginary))| {
                        // FDK's fast detector applies a 20 dB / 16 kHz
                        // high-pass tilt before summing QMF-band energies.
                        let weight =
                            2.0_f64.powf(0.000_752_75 * detector_bandwidth * (band + 1) as f64);
                        (real * real + imaginary * imaginary) * weight
                    })
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        let mean = slot_energy.iter().sum::<f64>() / slots.len() as f64;
        let transient_ratio = if mean <= f64::EPSILON {
            0.0
        } else {
            slot_energy.iter().copied().fold(0.0, f64::max) / mean
        };
        let transient_position = detect_low_delay_transient(
            &slot_energy,
            &mut self.low_delay_energy_history,
            &mut self.low_delay_ratio_history,
            &mut self.low_delay_candidate_history,
        )
        .map(|slot| slot.min(slots.len() - 1) as u8);
        let borders = transient_position
            .map(|position| low_delay_transient_borders(slots.len() as u8, position))
            .unwrap_or_else(|| vec![0, slots.len() as u8]);
        let resolutions = borders
            .windows(2)
            .map(|border| border[1] - border[0] >= 6)
            .collect::<Vec<_>>();
        let envelopes = borders
            .windows(2)
            .zip(&resolutions)
            .enumerate()
            .map(|(index, (border, &high))| {
                let table = if high {
                    &self.tables.high
                } else {
                    &self.tables.low
                };
                let analysis_stop = if index == 0
                    && transient_position.is_some_and(|position| position >= 2)
                    && border[1] - border[0] > 2
                {
                    border[1] - 2
                } else {
                    border[1]
                };
                SbrEncoderEnvelope {
                    start_slot: border[0] as usize,
                    end_slot: border[1] as usize,
                    bands: analyze_bands(&slots[border[0] as usize..analysis_stop as usize], table),
                }
            })
            .collect();
        let low_delay_prequant_debug = fixed_debug_context.as_ref().map(
            |(rows, previous_scale, current_scale, qmf_scale, common_scale)| {
                let scale0 = previous_scale - common_scale;
                let scale1 = current_scale - common_scale;
                let mut energies = Vec::with_capacity(resolutions.len());
                let mut counts = Vec::with_capacity(resolutions.len());
                for (index, (border, &high)) in borders.windows(2).zip(&resolutions).enumerate() {
                    let table = if high {
                        &self.tables.high
                    } else {
                        &self.tables.low
                    };
                    let start = usize::from(border[0]);
                    let stop = if index == 0
                        && transient_position.is_some_and(|position| position >= 2)
                        && border[1] - border[0] > 2
                    {
                        usize::from(border[1] - 2)
                    } else {
                        usize::from(border[1])
                    };
                    let mut envelope_energies = Vec::with_capacity(table.len() - 1);
                    let mut envelope_counts = Vec::with_capacity(table.len() - 1);
                    for (band_index, range) in table.windows(2).enumerate() {
                        let mut lower = usize::from(range[0]);
                        let upper = usize::from(range[1]);
                        if band_index == 0
                            && (high && upper - lower > 1 || !high && upper - lower > 2)
                        {
                            lower += 1;
                        }
                        envelope_energies.push(fixed_sfb_energy_split(
                            &rows[start..stop],
                            lower,
                            upper,
                            history_slots.saturating_sub(start).min(stop - start),
                            scale0,
                            scale1,
                        ));
                        envelope_counts.push(((stop - start) * (upper - lower)) as i32);
                    }
                    energies.push(envelope_energies);
                    counts.push(envelope_counts);
                }
                LowDelayPrequantDebug {
                    energies,
                    counts,
                    ybuffer_scales: (*previous_scale, *current_scale),
                    qmf_scale: *qmf_scale,
                    common_scale: *common_scale,
                }
            },
        );
        let current_tonality = low_delay_frame_tonality(
            &slots,
            &current_slots,
            usize::from(self.tables.high.first().copied().unwrap_or(0)).saturating_add(1),
        );
        let global_tonality = 0.5 * (current_tonality + self.low_delay_previous_tonality);
        self.low_delay_previous_tonality = current_tonality;
        Ok(SbrEncoderAnalysisFrame {
            envelopes,
            slots,
            transient_ratio,
            low_delay_transient_position: transient_position,
            low_delay_frequency_resolution: Some(resolutions),
            low_delay_amp_resolution: None,
            low_delay_global_tonality: Some(global_tonality),
            low_delay_envelope_coding: None,
            low_delay_noise_coding: None,
            low_delay_invf_modes: None,
            low_delay_patch_map: Some(self.patch_map.clone()),
            low_delay_prequant_debug,
        })
    }
}

fn detect_low_delay_transient(
    slot_energy: &[f64],
    energy_history: &mut [f64; 2],
    ratio_history: &mut [f64; 2],
    candidate_history: &mut [bool; 2],
) -> Option<usize> {
    if slot_energy.len() < 2 {
        return None;
    }
    let mut energies = Vec::with_capacity(slot_energy.len() + 2);
    energies.extend(*energy_history);
    energies.extend_from_slice(slot_energy);
    let mut candidates = Vec::with_capacity(slot_energy.len() + 2);
    candidates.extend(*candidate_history);
    candidates.resize(energies.len(), false);
    let mut ratios = Vec::with_capacity(energies.len());
    ratios.extend(*ratio_history);
    ratios.resize(energies.len(), 0.0);
    for index in 2..energies.len() {
        let ratio = energies[index] / (energies[index - 1] + 1.0e-2);
        ratios[index] = ratio;
        let isolated = !candidates[index - 2] && !candidates[index - 1];
        let dominates_recent = energies[index] / 1.4 >= energies[index - 1]
            || energies[index] / 1.4 >= energies[index - 2];
        if ratio >= 5.0 && (isolated || dominates_recent) {
            candidates[index] = true;
        }
    }
    // FDK's low-delay detector keeps two QMF slots as lookahead. Positions
    // 0 and 1 here are therefore the previous call's lookahead and become
    // the start of this frame; the final two newly analysed slots are only
    // candidates for the next call.
    let strongest = (0..slot_energy.len())
        .filter(|&index| candidates[index])
        .max_by(|&left, &right| ratios[left].total_cmp(&ratios[right]));
    *energy_history = [energies[energies.len() - 2], energies[energies.len() - 1]];
    *ratio_history = [ratios[ratios.len() - 2], ratios[ratios.len() - 1]];
    *candidate_history = [
        candidates[candidates.len() - 2],
        candidates[candidates.len() - 1],
    ];
    strongest
}

fn update_low_delay_transient_frame(
    frame: &SbrEncoderAnalysisFrame,
    state: &mut LowDelaySbrCodingState,
) {
    let transient_position = frame.low_delay_transient_position.map(usize::from);
    let final_border = frame
        .envelopes
        .last()
        .map_or(frame.slots.len(), |envelope| envelope.end_slot);
    let starts_in_frame = transient_position.is_some_and(|position| position + 4 < final_border);
    state.current_transient_frame = if state.transient_next_frame {
        state.transient_next_frame =
            transient_position.is_some_and(|position| position + 4 >= final_border);
        true
    } else if transient_position.is_some() {
        state.transient_next_frame = !starts_in_frame;
        starts_in_frame
    } else {
        false
    };
}

fn estimate_low_delay_noise_levels(
    frame: &SbrEncoderAnalysisFrame,
    tables: &LdSbrFrequencyTables,
    state: &mut LowDelaySbrCodingState,
) -> Vec<Vec<i8>> {
    const SMOOTH: [f64; 4] = [0.058_578_643_762_69, 0.2, 0.341_421_356_237_31, 0.4];
    let noise_bands = tables.noise_band_count();
    if noise_bands == 0 || frame.slots.is_empty() {
        return Vec::new();
    }
    if state.noise_level_history.len() != noise_bands {
        state.noise_level_history = vec![[0.0; 4]; noise_bands];
    }
    let split = if frame.slots.len() == 15 {
        8
    } else {
        frame.slots.len() / 2
    };
    let estimates = [&frame.slots[..split], &frame.slots[split..]];
    let estimate_ranges = if frame.envelopes.len() <= 1 {
        vec![0..2]
    } else {
        vec![0..1, 1..2]
    };
    let transient = state.current_transient_frame;
    let noise_floor_cap = state.noise_floor_cap;
    estimate_ranges
        .into_iter()
        .map(|estimate_range| {
            tables
                .noise
                .windows(2)
                .zip(&mut state.noise_level_history)
                .enumerate()
                .map(|(noise_index, (range, history))| {
                    // FDK estimates adaptive noise as the inverse original
                    // tonality quota, caps it at +6 dB, smooths four values,
                    // then converts 6-log2(noise)-2 to the transmitted index.
                    // The normalized complex-correlation quota is converted
                    // to the equivalent tone/noise odds before that step.
                    let band_start = usize::from(range[0]);
                    let band_stop = usize::from(range[1]);
                    let selected = &estimates[estimate_range.clone()];
                    let quota_mean = |patched: bool| {
                        let mut energy = 0.0;
                        let mut quota = 0.0;
                        let mut count = 0usize;
                        for &block in selected {
                            for target in band_start..band_stop {
                                let band = if patched {
                                    frame
                                        .low_delay_patch_map
                                        .as_ref()
                                        .and_then(|map| map.get(target))
                                        .copied()
                                        .flatten()
                                        .unwrap_or(target)
                                } else {
                                    target
                                };
                                for slot in block {
                                    if band < slot.real.len() {
                                        energy += slot.real[band] * slot.real[band]
                                            + slot.imaginary[band] * slot.imaginary[band];
                                    }
                                }
                                quota += f64::from(fixed_lpc_quota(block, band))
                                    / 2_147_483_648.0
                                    / 1.0e-6;
                                count += 1;
                            }
                        }
                        (energy, quota / count.max(1) as f64)
                    };
                    let (energy, mut quota) = quota_mean(false);
                    let (_, source_quota) = quota_mean(true);
                    if energy <= f64::EPSILON && source_quota <= f64::EPSILON {
                        quota = 101.593_667_3;
                    }
                    let quota = if energy <= f64::EPSILON {
                        101.593_667_3
                    } else {
                        // Quotas above were divided by RELAXATION (1e-6).
                        // C's RELAXATION floor is therefore 1.0 in these
                        // normalized units, not another factor of 1e-6.
                        quota.max(1.0)
                    };
                    let mode = state
                        .previous_invf_modes
                        .get(noise_index)
                        .copied()
                        .unwrap_or(0);
                    let difference = if mode > 2 {
                        (0.25 * source_quota.max(1.0) / quota).max(1.0)
                    } else {
                        1.0
                    };
                    // qmfBasedNoiseFloorDetection stores the adaptive level
                    // with an explicit 0.25 scale before temporal smoothing.
                    let linear = (0.25 * difference / quota).min(noise_floor_cap);
                    if transient {
                        *history = [linear; 4];
                    } else {
                        history.rotate_left(1);
                        history[3] = linear;
                    }
                    let smoothed = history
                        .iter()
                        .zip(SMOOTH)
                        .map(|(&value, weight)| value * weight)
                        .sum::<f64>()
                        .max(f64::MIN_POSITIVE);
                    // sbrNoiseFloorLevelsQuantisation truncates the positive
                    // LD_DATA value and adds one whenever it is non-zero.
                    // This is a ceil operation, not round-to-nearest.
                    (4.0 - smoothed.log2()).ceil().clamp(0.0, 30.0) as i8
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn estimate_low_delay_inverse_filtering(
    frame: &SbrEncoderAnalysisFrame,
    tables: &LdSbrFrequencyTables,
    state: &mut LowDelaySbrCodingState,
) -> Vec<u8> {
    const FIR: [f64; 3] = [0.125, 0.375, 0.5];
    let count = tables.noise_band_count();
    if count == 0 || frame.slots.is_empty() {
        return Vec::new();
    }
    if state.invf_bands.len() != count {
        state.invf_bands = vec![InverseFilterBandState::default(); count];
    }
    let crossover = usize::from(tables.high.first().copied().unwrap_or(0));
    let transient = state.current_transient_frame;
    // The C tonality matrix is evaluated on the delayed YBuffer window, while
    // nrgVector contains the energy summed over every active QMF channel (not
    // merely the current inverse-filter band).
    let quota_slots = &frame.slots;
    let qmf_bands = quota_slots.first().map_or(0, |slot| slot.real.len());
    let total_energy = lpc_band_statistics(quota_slots, 0, qmf_bands).0;
    let energy = 3.0 * total_energy.max(1.0e-30).log2() + 120.0;
    tables
        .noise
        .windows(2)
        .zip(&mut state.invf_bands)
        .map(|(range, detector)| {
            let start = usize::from(range[0]);
            let stop = usize::from(range[1]);
            let width = stop.saturating_sub(start);
            let source_stop = crossover.saturating_sub(start.saturating_sub(crossover));
            let source_start = source_stop.saturating_sub(width);
            let (_, original_quota) = lpc_band_statistics(quota_slots, start, stop);
            let source_quota = if let Some(patch_map) = &frame.low_delay_patch_map {
                let quotas = (start..stop)
                    .filter_map(|target| patch_map.get(target).copied().flatten())
                    .map(|source| lpc_band_statistics(quota_slots, source, source + 1).1)
                    .collect::<Vec<_>>();
                quotas.iter().sum::<f64>() / quotas.len().max(1) as f64
            } else if source_start < source_stop {
                lpc_band_statistics(quota_slots, source_start, source_stop).1
            } else {
                original_quota
            };
            detector.orig_quota_history.rotate_left(1);
            detector.sbr_quota_history.rotate_left(1);
            detector.orig_quota_history[2] = original_quota;
            detector.sbr_quota_history[2] = source_quota;
            let orig_linear = detector
                .orig_quota_history
                .iter()
                .zip(FIR)
                .map(|(&value, weight)| value * weight)
                .sum::<f64>();
            let sbr_linear = detector
                .sbr_quota_history
                .iter()
                .zip(FIR)
                .map(|(&value, weight)| value * weight)
                .sum::<f64>();
            let orig = 3.0 * orig_linear.max(1.0e-30).log2();
            let sbr = 3.0 * sbr_linear.max(1.0e-30).log2();
            inverse_filter_decision(orig, sbr, energy, transient, detector)
        })
        .collect()
}

fn lpc_band_statistics(slots: &[QmfSlot], start_band: usize, stop_band: usize) -> (f64, f64) {
    if slots.len() < 3 || start_band >= stop_band {
        return (0.0, 0.0);
    }
    let split = if slots.len() == 15 {
        8
    } else {
        slots.len() / 2
    };
    let mut energy = 0.0;
    let mut quota_sum = 0.0;
    let mut estimates = 0;
    for band in start_band..stop_band {
        for block in [&slots[..split], &slots[split..]] {
            if block.len() < 3 || band >= block[0].real.len() {
                continue;
            }
            for index in 2..block.len() {
                energy += block[index].real[band] * block[index].real[band]
                    + block[index].imaginary[band] * block[index].imaginary[band];
            }
            quota_sum += f64::from(fixed_lpc_quota(block, band)) / 2_147_483_648.0 / 1.0e-6;
            estimates += 1;
        }
    }
    (energy, quota_sum / estimates.max(1) as f64)
}

fn low_delay_frame_tonality(
    aligned_energy_slots: &[QmfSlot],
    current_quota_slots: &[QmfSlot],
    start_band: usize,
) -> f64 {
    let stop_band = aligned_energy_slots
        .first()
        .map_or(0, |slot| slot.real.len());
    if start_band >= stop_band {
        return 0.0;
    }
    let mut energetic = (start_band..stop_band)
        .map(|band| {
            let energy = aligned_energy_slots
                .iter()
                .map(|slot| {
                    slot.real[band] * slot.real[band] + slot.imaginary[band] * slot.imaginary[band]
                })
                .sum::<f64>();
            (band, energy)
        })
        .collect::<Vec<_>>();
    energetic.sort_by(|left, right| right.1.total_cmp(&left.1));
    let count = energetic.len().min(5);
    let mean_quota = energetic[..count]
        .iter()
        .map(|&(band, _)| lpc_band_statistics(current_quota_slots, band, band + 1).1)
        .sum::<f64>()
        / count.max(1) as f64;
    // FDK accumulates two quota estimates with /2, five selected bands with
    // /4, then interprets the fixed value using RELAXATION and a 2/3
    // correction. In the unscaled quota units used here this is 10/3.
    mean_quota * (10.0 / 3.0)
}

#[derive(Default)]
struct FixedAutoCorrelation {
    r00: i32,
    r11: i32,
    r01r: i32,
    r01i: i32,
    r12r: i32,
    r12i: i32,
    r02r: i32,
    r02i: i32,
    determinant: i32,
    determinant_scale: i32,
}

fn fixed_mul_div2(left: i32, right: i32) -> i32 {
    ((i64::from(left) * i64::from(right)) >> 32) as i32
}

fn fixed_mul(left: i32, right: i32) -> i32 {
    fixed_mul_div2(left, right).wrapping_shl(1)
}

fn fixed_norm(value: i32) -> i32 {
    if value == 0 {
        0
    } else {
        let magnitude = if value < 0 { !value } else { value };
        magnitude.leading_zeros() as i32 - 1
    }
}

fn fixed_scale(value: i32, shift: i32) -> i32 {
    if shift >= 0 {
        value.wrapping_shl(shift as u32)
    } else {
        value >> (-shift as u32)
    }
}

fn fixed_schur_div(numerator: i32, denominator: i32) -> i32 {
    if numerator == denominator {
        i32::MAX
    } else {
        ((i64::from(numerator) << 31) / i64::from(denominator)) as i32
    }
}

fn fixed_autocorrelation(real: &[i32], imaginary: &[i32]) -> FixedAutoCorrelation {
    let length = real.len() - 2;
    // C uses DFRACT_BITS-fNormz(len), i.e. ceil(log2(len)) for positive
    // non-powers of two. Using floor(log2(len)) changes every accumulated
    // product by one bit for the 5/6-sample LD LPC segments.
    let length_scale = (32 - (length as u32).leading_zeros() as i32).max(1);
    let product = |left: i32, right: i32| fixed_mul_div2(left, right);
    let mut r11 = 0_i32;
    let mut r01r = 0_i32;
    let mut r01i = 0_i32;
    let mut r02r = 0_i32;
    let mut r02i = 0_i32;
    let energy = |position: usize| {
        product(real[position], real[position])
            .wrapping_add(product(imaginary[position], imaginary[position]))
            >> length_scale
    };
    let real_cross = |left: usize, right: usize| {
        product(real[left], real[right]).wrapping_add(product(imaginary[left], imaginary[right]))
            >> length_scale
    };
    let imag_cross = |current: usize, previous: usize| {
        product(imaginary[current], real[previous])
            .wrapping_sub(product(real[current], imaginary[previous]))
            >> length_scale
    };
    r02r = r02r.wrapping_add(real_cross(2, 0));
    r02i = r02i.wrapping_add(imag_cross(2, 0));
    for previous in 1..length {
        r11 = r11.wrapping_add(energy(previous));
        r01r = r01r.wrapping_add(real_cross(previous, previous + 1));
        r01i = r01i.wrapping_add(imag_cross(previous + 1, previous));
        r02r = r02r.wrapping_add(real_cross(previous + 2, previous));
        r02i = r02i.wrapping_add(imag_cross(previous + 2, previous));
    }
    let mut r22 = energy(0).wrapping_add(r11);
    r11 = r11.wrapping_add(energy(length));
    let mut r00 = energy(length + 1).wrapping_sub(energy(1)).wrapping_add(r11);
    let mut r12r = real_cross(1, 0).wrapping_add(r01r);
    r01r = r01r.wrapping_add(real_cross(length + 1, length));
    let mut r12i = imag_cross(1, 0).wrapping_add(r01i);
    r01i = r01i.wrapping_add(imag_cross(length + 1, length));
    let combined = r00
        | r11
        | r22
        | r01r.wrapping_abs()
        | r01i.wrapping_abs()
        | r12r.wrapping_abs()
        | r12i.wrapping_abs()
        | r02r.wrapping_abs()
        | r02i.wrapping_abs();
    let scale = combined.leading_zeros().saturating_sub(1);
    for value in [
        &mut r00, &mut r11, &mut r22, &mut r01r, &mut r01i, &mut r12r, &mut r12i, &mut r02r,
        &mut r02i,
    ] {
        *value = value.wrapping_shl(scale);
    }
    let mut determinant = (fixed_mul_div2(r11, r22) >> 1)
        .wrapping_sub(fixed_mul_div2(r12r, r12r).wrapping_add(fixed_mul_div2(r12i, r12i)) >> 1);
    let determinant_shift = determinant.unsigned_abs().leading_zeros().saturating_sub(1);
    determinant = determinant.wrapping_shl(determinant_shift);
    FixedAutoCorrelation {
        r00,
        r11,
        r01r,
        r01i,
        r12r,
        r12i,
        r02r,
        r02i,
        determinant,
        determinant_scale: determinant_shift as i32 - 2,
    }
}

fn fixed_lpc_quota(block: &[QmfSlot], band: usize) -> i32 {
    const RELAXATION_FRACT: i32 = 1_125_899_904; // Q1.31 0.524288f
    const RELAXATION_SHIFT: i32 = 19;
    if block.len() < 3 || band >= block[0].real.len() {
        return 0;
    }
    // CLDFB exposes normalized input with a 2^-31 QMF conversion and raw
    // INT_PCM input with a 2^-24 conversion. Recover the original FIXP_DBL
    // words before running the fixed LPC path. Clamping raw QMF values to
    // [-1, 1) destroys strong low-band tonal components.
    let raw_pcm_scale = qmf_block_exceeds_unit(block);
    let multiplier = if raw_pcm_scale {
        16_777_216.0
    } else {
        2_147_483_648.0
    };
    let to_fixed = |value: f64| (value * multiplier).clamp(i32::MIN as f64, i32::MAX as f64) as i32;
    let mut real = block
        .iter()
        .map(|slot| to_fixed(slot.real[band]))
        .collect::<Vec<_>>();
    let mut imaginary = block
        .iter()
        .map(|slot| to_fixed(slot.imaginary[band]))
        .collect::<Vec<_>>();
    let scale_factor = |values: &[i32]| {
        let combined = values
            .iter()
            .fold(0_i32, |combined, &value| combined | (value ^ (value >> 31)));
        (combined.leading_zeros() as i32 - 1).max(0)
    };
    let shift = scale_factor(&real)
        .min(scale_factor(&imaginary))
        .saturating_sub(1)
        .max(0);
    for value in real.iter_mut().chain(&mut imaginary) {
        *value = value.wrapping_shl(shift as u32);
    }
    let ac = fixed_autocorrelation(&real, &imaginary);
    let (alpha0r, alpha0i, alpha1r, alpha1i, fac) = if ac.determinant == 0 {
        (
            ac.r01r >> 2,
            ac.r01i >> 2,
            0,
            0,
            fixed_mul_div2(ac.r00, ac.r11) >> 1,
        )
    } else {
        let alpha1r = (fixed_mul_div2(ac.r01r, ac.r12r) >> 1)
            .wrapping_sub(fixed_mul_div2(ac.r01i, ac.r12i) >> 1)
            .wrapping_sub(fixed_mul_div2(ac.r02r, ac.r11) >> 1);
        let alpha1i = (fixed_mul_div2(ac.r01i, ac.r12r) >> 1)
            .wrapping_add(fixed_mul_div2(ac.r01r, ac.r12i) >> 1)
            .wrapping_sub(fixed_mul_div2(ac.r02i, ac.r11) >> 1);
        let divisor_shift = (ac.determinant_scale + 1).max(0) as u32;
        let alpha0r = (fixed_mul_div2(ac.r01r, ac.determinant) >> divisor_shift)
            .wrapping_add(fixed_mul(alpha1r, ac.r12r))
            .wrapping_add(fixed_mul(alpha1i, ac.r12i));
        let alpha0i = (fixed_mul_div2(ac.r01i, ac.determinant) >> divisor_shift)
            .wrapping_add(fixed_mul(alpha1i, ac.r12r))
            .wrapping_sub(fixed_mul(alpha1r, ac.r12i));
        let fac = fixed_mul_div2(ac.r00, fixed_mul(ac.determinant, ac.r11)) >> divisor_shift;
        (alpha0r, alpha0i, alpha1r, alpha1i, fac)
    };
    if fac == 0 {
        return 0;
    }
    let mut numerator = fixed_mul_div2(alpha0r, ac.r01r)
        .wrapping_add(fixed_mul_div2(alpha0i, ac.r01i))
        .wrapping_sub(fixed_mul_div2(alpha1r, fixed_mul(ac.r02r, ac.r11)))
        .wrapping_sub(fixed_mul_div2(alpha1i, fixed_mul(ac.r02i, ac.r11)))
        .wrapping_abs();
    let mut denominator = (fac >> 1)
        .wrapping_add(fixed_mul_div2(fac, RELAXATION_FRACT) >> RELAXATION_SHIFT)
        .wrapping_sub(numerator)
        .wrapping_abs();
    numerator = fixed_mul(numerator, RELAXATION_FRACT);
    if numerator <= 0 || denominator == 0 {
        return 0;
    }
    let numerator_shift = fixed_norm(numerator) - 2;
    numerator = fixed_scale(numerator, numerator_shift);
    let denominator_shift = fixed_norm(denominator);
    denominator = denominator.wrapping_shl(denominator_shift as u32);
    let common_shift = (numerator_shift - denominator_shift + RELAXATION_SHIFT).min(30);
    if numerator > denominator {
        return i32::MAX;
    }
    let quota = if common_shift < 0 {
        let value = fixed_schur_div(numerator, denominator);
        value.wrapping_shl((-common_shift).min(fixed_norm(value)) as u32)
    } else {
        fixed_schur_div(numerator, denominator) >> common_shift
    };
    quota
}

fn qmf_block_exceeds_unit(block: &[QmfSlot]) -> bool {
    block.iter().any(|slot| {
        slot.real.iter().any(|value| value.abs() > 1.0)
            || slot.imaginary.iter().any(|value| value.abs() > 1.0)
    })
}

fn fixed_complex_energies(real: &[i32], imaginary: &[i32], qmf_scale: i32) -> (Vec<i32>, i32, i32) {
    debug_assert_eq!(real.len(), imaginary.len());
    let scale_factor = |values: &[i32]| {
        values
            .iter()
            .map(|&value| fixed_norm(value))
            .min()
            .unwrap_or(31)
    };
    let input_shift = scale_factor(real)
        .min(scale_factor(imaginary))
        .saturating_sub(1)
        .max(0);
    let qmf_scale = qmf_scale + input_shift;
    let mut energies = real
        .iter()
        .zip(imaginary)
        .map(|(&real, &imaginary)| {
            let real = real.wrapping_shl(input_shift as u32);
            let imaginary = imaginary.wrapping_shl(input_shift as u32);
            fixed_mul_div2(real, real).wrapping_add(fixed_mul_div2(imaginary, imaginary))
        })
        .collect::<Vec<_>>();
    let maximum = energies.iter().copied().max().unwrap_or(0);
    let energy_shift = fixed_norm(maximum).max(0);
    for energy in &mut energies {
        *energy = energy.wrapping_shl(energy_shift as u32);
    }
    let energy_scale = 2 * qmf_scale - 1 + energy_shift;
    (energies, energy_scale, qmf_scale)
}

#[allow(dead_code)]
fn fixed_cldfb64_energy_block(slots: &[QmfSlot]) -> (Vec<Vec<i32>>, i32, i32) {
    const BANDS: usize = 64;
    const LB_SCALE: i32 = -8;
    let convert = |value: f64| {
        value
            .mul_add(2.0_f64.powi(16 - LB_SCALE), 0.0)
            .round()
            .clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32
    };
    let real = slots
        .iter()
        .flat_map(|slot| slot.real.iter().take(BANDS).map(|&value| convert(value)))
        .collect::<Vec<_>>();
    let imaginary = slots
        .iter()
        .flat_map(|slot| {
            slot.imaginary
                .iter()
                .take(BANDS)
                .map(|&value| convert(value))
        })
        .collect::<Vec<_>>();
    let (energies, scale, qmf_scale) = fixed_complex_energies(&real, &imaginary, LB_SCALE + 7);
    (
        energies.chunks_exact(BANDS).map(<[i32]>::to_vec).collect(),
        scale,
        qmf_scale,
    )
}

#[allow(dead_code)] // Integrated after the QMF exponent path is made stateful.
fn fixed_sfb_energy(rows: &[Vec<i32>], lower_band: usize, upper_band: usize, scale: i32) -> i32 {
    fixed_sfb_energy_split(rows, lower_band, upper_band, rows.len(), scale, scale)
}

#[allow(dead_code)]
fn fixed_sfb_energy_split(
    rows: &[Vec<i32>],
    lower_band: usize,
    upper_band: usize,
    border: usize,
    scale0: i32,
    scale1: i32,
) -> i32 {
    if rows.is_empty() || lower_band >= upper_band {
        return 0;
    }
    let width = upper_band - lower_band;
    // CalcLdInt(width) is log2(width) in ld64 representation; the C shift
    // converts it to the integer binary logarithm, rounded downward.
    let dynamic_scale = (width as f64).log2().floor() as i32;
    let clipped0 = scale0.min(5);
    let clipped1 = scale1.min(5);
    let residual0 = (scale0 - clipped0).min(dynamic_scale).max(0);
    let residual1 = (scale1 - clipped1).min(dynamic_scale).max(0);
    let mut accumulated0 = 0_i32;
    let mut accumulated1 = 0_i32;
    let saturating_add = |left: i32, right: i32| {
        let half = i64::from(left >> 1) + i64::from(right >> 1);
        (half.clamp(i64::from(i32::MIN >> 1), i64::from(i32::MAX >> 1)) as i32) << 1
    };
    for band in lower_band..upper_band {
        let mut energy0 = 0_i32;
        let mut energy1 = 0_i32;
        for row in rows.iter().take(border) {
            energy0 = energy0.wrapping_add(row.get(band).copied().unwrap_or(0) >> clipped0);
        }
        for row in rows.iter().skip(border) {
            energy1 = energy1.wrapping_add(row.get(band).copied().unwrap_or(0) >> clipped1);
        }
        accumulated0 = saturating_add(accumulated0, energy0 >> residual0);
        accumulated1 = saturating_add(accumulated1, energy1 >> residual1);
    }
    (accumulated0 >> (scale0 - clipped0 - residual0).clamp(0, 31))
        .wrapping_add(accumulated1 >> (scale1 - clipped1 - residual1).clamp(0, 31))
}

#[allow(dead_code)]
fn fixed_log2_ld64(value: i32) -> i32 {
    const COEFFICIENTS: [i16; 10] = [
        -32_768, -16_384, -10_923, -8_192, -6_554, -5_461, -4_681, -4_096, -3_641, -3_277,
    ];
    if value <= 0 {
        return i32::MIN;
    }
    let normalization = value.leading_zeros() as i32 - 1;
    let normalized = value.wrapping_shl(normalization as u32);
    let mapped = normalized.wrapping_add(i32::MIN).wrapping_neg();
    let mut power = mapped;
    let mut result = 0_i32;
    for coefficient in COEFFICIENTS {
        result = result.wrapping_add(fixed_mul_div2(i32::from(coefficient) << 16, power));
        power = fixed_mul(power, mapped);
    }
    result = result.wrapping_add(fixed_mul_div2(result, 1_901_360_723));
    let exponent = -normalization;
    let (result, result_exponent) = if exponent == 0 {
        (result, 1)
    } else {
        let result_exponent = 32 - fixed_norm(exponent);
        (
            (result >> (result_exponent - 1))
                .wrapping_add(exponent.wrapping_shl((31 - result_exponent) as u32)),
            result_exponent,
        )
    };
    fixed_scale(result, result_exponent - 6)
}

#[allow(dead_code)]
fn fixed_quantize_sbr_energy(
    energy: i32,
    sample_count: usize,
    common_scale: i32,
    amplitude_resolution_3db: bool,
) -> i32 {
    let one_bit_less = i32::from(amplitude_resolution_3db);
    let mut logarithmic = if energy > 0 {
        let normalization = fixed_norm(energy);
        let normalized = energy.wrapping_shl(normalization as u32);
        let energy_log = fixed_log2_ld64(normalized);
        let scale_log = (common_scale + normalization).wrapping_shl(24);
        let count_value = ((sample_count as i32).wrapping_mul(64)).wrapping_shl(16);
        let count_log = fixed_log2_ld64(count_value);
        ((energy_log.wrapping_sub(count_log)) >> 1)
            .wrapping_add(486_539_264_i32.wrapping_sub(scale_log))
    } else {
        i32::MIN
    };
    logarithmic = logarithmic.clamp(0, 0x4000_0000 >> one_bit_less);
    logarithmic >>= 23 - one_bit_less;
    logarithmic.wrapping_add(1) >> 1
}

fn inverse_filter_region(value: f64, borders: &[f64; 4], previous: usize) -> usize {
    let mut adjusted = *borders;
    if previous < adjusted.len() {
        adjusted[previous] += 1.0;
    }
    if previous > 0 {
        adjusted[previous - 1] -= 1.0;
    }
    adjusted
        .iter()
        .take_while(|&&border| value >= border)
        .count()
}

fn inverse_filter_decision(
    original_quota: f64,
    sbr_quota: f64,
    energy: f64,
    transient: bool,
    state: &mut InverseFilterBandState,
) -> u8 {
    const SBR_BORDERS: [f64; 4] = [1.0, 10.0, 14.0, 19.0];
    const ORIG_BORDERS: [f64; 4] = [0.0, 3.0, 7.0, 10.0];
    const ENERGY_BORDERS: [f64; 4] = [25.0, 30.0, 35.0, 40.0];
    const NORMAL: [[u8; 5]; 5] = [
        [2, 1, 0, 0, 0],
        [2, 1, 0, 0, 0],
        [3, 2, 1, 0, 0],
        [3, 3, 2, 0, 0],
        [3, 3, 2, 0, 0],
    ];
    const TRANSIENT: [[u8; 5]; 5] = [
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [3, 2, 2, 0, 0],
        [3, 3, 2, 0, 0],
        [3, 3, 2, 0, 0],
    ];
    const ENERGY_REDUCTION: [i8; 5] = [-4, -3, -2, -1, 0];
    let sbr_region = inverse_filter_region(sbr_quota, &SBR_BORDERS, state.previous_sbr_region);
    let orig_region =
        inverse_filter_region(original_quota, &ORIG_BORDERS, state.previous_orig_region);
    let energy_region = ENERGY_BORDERS
        .iter()
        .take_while(|&&border| energy >= border)
        .count();
    state.previous_sbr_region = sbr_region;
    state.previous_orig_region = orig_region;
    let level = if transient {
        TRANSIENT[sbr_region][orig_region]
    } else {
        NORMAL[sbr_region][orig_region]
    } as i8;
    let mode = (level + ENERGY_REDUCTION[energy_region]).max(0) as u8;
    mode
}

fn couple_low_delay_noise_levels(
    left: &[Vec<i8>],
    right: &[Vec<i8>],
) -> Result<(Vec<Vec<i8>>, Vec<Vec<i8>>), SbrEncoderError> {
    if left.len() != right.len()
        || left
            .iter()
            .zip(right)
            .any(|(left, right)| left.len() != right.len())
    {
        return Err(SbrEncoderError::EnvelopeLayoutMismatch);
    }
    let panorama_steps = [0_i16, 2, 4, 8, 12];
    let mut levels = Vec::with_capacity(left.len());
    let mut balances = Vec::with_capacity(left.len());
    for (left, right) in left.iter().zip(right) {
        let mut level = Vec::with_capacity(left.len());
        let mut balance = Vec::with_capacity(left.len());
        for (&left_index, &right_index) in left.iter().zip(right) {
            let left_linear = 2.0_f64.powi(4 - i32::from(left_index));
            let right_linear = 2.0_f64.powi(4 - i32::from(right_index));
            let average = (left_linear + right_linear) * 0.5;
            level.push((4.0 - average.log2()).round().clamp(0.0, 30.0) as i8);

            // C quantizes log2(left/right) to the 3 dB panorama table and
            // transmits it around the center value 12.
            let raw = i16::from(right_index) - i16::from(left_index);
            let magnitude = raw.unsigned_abs() as i16;
            let nearest = panorama_steps
                .iter()
                .copied()
                .min_by_key(|step| (magnitude - *step).abs())
                .unwrap_or(0);
            balance.push((12 + raw.signum() * nearest).clamp(0, 24) as i8);
        }
        constrain_frequency_deltas(&mut level, SbrHuffmanBook::EnvelopeLevel30Frequency);
        constrain_scaled_frequency_deltas(
            &mut balance,
            SbrHuffmanBook::EnvelopeBalance30Frequency,
            2,
        );
        levels.push(level);
        balances.push(balance);
    }
    Ok((levels, balances))
}

#[allow(clippy::too_many_arguments)]
fn prepare_low_delay_delta_coding(
    values: &[Vec<i8>],
    previous: &mut Option<Vec<i8>>,
    frequency_book: SbrHuffmanBook,
    time_book: SbrHuffmanBook,
    start_bits: usize,
    divisor: i8,
    header_present: bool,
    first_time_weight_q15: usize,
) -> Vec<LowDelayEnvelopeCoding> {
    let mut history = previous.clone();
    let mut result = Vec::with_capacity(values.len());
    for (index, values) in values.iter().enumerate() {
        let mut frequency = Vec::with_capacity(values.len());
        frequency.push(values[0] / divisor);
        frequency.extend(values.windows(2).map(|pair| (pair[1] - pair[0]) / divisor));
        let frequency_bits = start_bits
            + frequency[1..]
                .iter()
                .map(|&delta| encode_sbr_huffman(frequency_book, delta).unwrap().len())
                .sum::<usize>();
        let time = history
            .as_ref()
            .filter(|old| old.len() == values.len())
            .and_then(|old| {
                let deltas = values
                    .iter()
                    .zip(old)
                    .map(|(&current, &old)| (current - old) / divisor)
                    .collect::<Vec<_>>();
                let time_bits = deltas
                    .iter()
                    .map(|&delta| encode_sbr_huffman(time_book, delta).map(|code| code.len()))
                    .collect::<Option<Vec<_>>>()?
                    .into_iter()
                    .sum::<usize>();
                let threshold = if index == 0 {
                    low_delay_first_time_threshold(time_bits, first_time_weight_q15)
                } else {
                    time_bits
                };
                (!header_present && frequency_bits > threshold
                    || index > 0 && frequency_bits > threshold)
                    .then_some(deltas)
            });
        result.push(if let Some(deltas) = time {
            LowDelayEnvelopeCoding { time: true, deltas }
        } else {
            LowDelayEnvelopeCoding {
                time: false,
                deltas: frequency,
            }
        });
        history = Some(values.clone());
    }
    *previous = history;
    result
}

fn low_delay_envelope_time_weight_q15(streak: usize) -> usize {
    const POINT_THREE_Q31: i64 = 644_245_094;
    let increment = (POINT_THREE_Q31 * ((streak as i64) << 15)) >> 31;
    32_768 + 9_830 + increment as usize
}

fn low_delay_first_time_threshold(time_bits: usize, weight_q15: usize) -> usize {
    (((time_bits * weight_q15) >> 14) + 1) >> 1
}

fn low_delay_transient_borders(time_slots: u8, position: u8) -> Vec<u8> {
    let last_three_envelope_position = if time_slots == 15 { 9 } else { 10 };
    if position < 2 {
        vec![0, position + 4, time_slots]
    } else if position <= last_three_envelope_position {
        vec![0, position, position + 4, time_slots]
    } else {
        vec![0, position, time_slots]
    }
}

fn write_mono_grid(
    writer: &mut BitWriter,
    frame: &SbrEncoderAnalysisFrame,
) -> Result<(), SbrEncoderError> {
    if frame.envelopes.len() == 1 {
        writer.write(0, 2);
        writer.write(0, 2);
        writer.write_bool(true);
        return Ok(());
    }
    let total_time_slots = frame.slots.len() / 2;
    let boundary = frame.envelopes[0].end_slot / 2;
    if boundary == total_time_slots / 2 {
        writer.write(0, 2);
        writer.write(1, 2);
        writer.write_bool(true);
    } else if boundary < total_time_slots / 2 && (2..=8).contains(&boundary) {
        writer.write(2, 2); // VARFIX
        writer.write(0, 2); // left border
        writer.write(1, 2); // one relative border
        writer.write(((boundary - 2) / 2) as u32, 2);
        writer.write(2, 2); // transient envelope 1
        writer.write_bool(true);
        writer.write_bool(true);
    } else if boundary > total_time_slots / 2
        && boundary <= total_time_slots.saturating_sub(2)
        && total_time_slots - boundary <= 8
    {
        writer.write(1, 2); // FIXVAR
        writer.write(0, 2); // right border == time_slots
        writer.write(1, 2); // one relative border
        writer.write(((total_time_slots - boundary - 2) / 2) as u32, 2);
        writer.write(1, 2); // transient envelope 2
        writer.write_bool(true);
        writer.write_bool(true);
    } else {
        return Err(SbrEncoderError::EnvelopeLayoutMismatch);
    }
    Ok(())
}

fn analyze_bands(slots: &[QmfSlot], borders: &[u8]) -> Vec<SbrEncoderBand> {
    borders
        .windows(2)
        .map(|border| {
            let mut energy = 0.0;
            let mut correlation_real = 0.0;
            let mut correlation_imaginary = 0.0;
            let mut previous_energy = 0.0;
            let mut current_energy = 0.0;
            for band in usize::from(border[0])..usize::from(border[1]) {
                for slot in slots {
                    energy += slot.real[band] * slot.real[band]
                        + slot.imaginary[band] * slot.imaginary[band];
                }
                for pair in slots.windows(2) {
                    let (ar, ai) = (pair[0].real[band], pair[0].imaginary[band]);
                    let (br, bi) = (pair[1].real[band], pair[1].imaginary[band]);
                    correlation_real += ar * br + ai * bi;
                    correlation_imaginary += ai * br - ar * bi;
                    previous_energy += ar * ar + ai * ai;
                    current_energy += br * br + bi * bi;
                }
            }
            let denominator = (previous_energy * current_energy).sqrt();
            let tonality = if denominator <= f64::EPSILON {
                0.0
            } else {
                (correlation_real.hypot(correlation_imaginary) / denominator).clamp(0.0, 1.0)
            };
            SbrEncoderBand { energy, tonality }
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SbrEncoderError {
    Qmf(QmfError),
    Frequency(LdSbrError),
    NonFiniteInput,
    EmptyFrame,
    EnvelopeLayoutMismatch,
    UnrepresentableHuffmanSymbol(i8),
    PayloadTooLarge(usize),
    UnsupportedFrameLength(usize),
    LowDelaySlotCountMismatch { expected: usize, actual: usize },
    QmfBandRangeMismatch { available: usize, required: usize },
    Asc(crate::asc::AscError),
}

impl From<QmfError> for SbrEncoderError {
    fn from(value: QmfError) -> Self {
        Self::Qmf(value)
    }
}

impl From<LdSbrError> for SbrEncoderError {
    fn from(value: LdSbrError) -> Self {
        Self::Frequency(value)
    }
}

impl From<crate::asc::AscError> for SbrEncoderError {
    fn from(value: crate::asc::AscError) -> Self {
        Self::Asc(value)
    }
}

impl fmt::Display for SbrEncoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Qmf(error) => write!(f, "SBR QMF analysis error: {error:?}"),
            Self::Frequency(error) => error.fmt(f),
            Self::NonFiniteInput => write!(f, "SBR encoder input contains NaN or infinity"),
            Self::EmptyFrame => write!(f, "SBR encoder input contains no QMF slots"),
            Self::EnvelopeLayoutMismatch => write!(f, "SBR encoder envelope layout mismatch"),
            Self::UnrepresentableHuffmanSymbol(symbol) => {
                write!(f, "unrepresentable SBR Huffman symbol {symbol}")
            }
            Self::PayloadTooLarge(bytes) => write!(f, "SBR payload is too large: {bytes} bytes"),
            Self::UnsupportedFrameLength(length) => {
                write!(f, "unsupported low-delay SBR frame length {length}")
            }
            Self::LowDelaySlotCountMismatch { expected, actual } => write!(
                f,
                "expected {expected} low-delay SBR time slots, got {actual}"
            ),
            Self::QmfBandRangeMismatch {
                available,
                required,
            } => write!(
                f,
                "low-delay SBR frequency table requires QMF band {required}, but the analysis bank has {available} bands"
            ),
            Self::Asc(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for SbrEncoderError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitReader;
    use crate::ld_sbr::LdSbrFrameParser;
    use crate::ps::PsParser;
    use crate::ps_encoder::PsEncoderFrame;
    use crate::sbr::{parse_sbr_fill_element, SbrMonoFrameParser, SbrStereoFrameParser};

    #[cfg(feature = "ffi")]
    static TONALITY_QUOTA_C_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn header() -> LdSbrHeader {
        LdSbrHeader {
            start_frequency: 5,
            stop_frequency: 8,
            crossover_band: 0,
            ..LdSbrHeader::default()
        }
    }

    fn synthetic_frame(bands: usize, envelope_ends: &[usize]) -> SbrEncoderAnalysisFrame {
        let slots = vec![
            QmfSlot {
                real: vec![0.0; 64],
                imaginary: vec![0.0; 64],
            };
            32
        ];
        let mut start = 0;
        let envelopes = envelope_ends
            .iter()
            .map(|&end| {
                let envelope = SbrEncoderEnvelope {
                    start_slot: start,
                    end_slot: end,
                    bands: vec![
                        SbrEncoderBand {
                            energy: 0.0,
                            tonality: 0.0
                        };
                        bands
                    ],
                };
                start = end;
                envelope
            })
            .collect();
        SbrEncoderAnalysisFrame {
            slots,
            envelopes,
            transient_ratio: 0.0,
            low_delay_transient_position: None,
            low_delay_frequency_resolution: None,
            low_delay_amp_resolution: None,
            low_delay_global_tonality: None,
            low_delay_envelope_coding: None,
            low_delay_noise_coding: None,
            low_delay_invf_modes: None,
            low_delay_patch_map: None,
            low_delay_prequant_debug: None,
        }
    }

    #[test]
    fn writes_unframed_eld_low_delay_mono_payloads() {
        let header = header();
        let tables = LdSbrFrequencyTables::from_header(&header, 48_000).unwrap();
        for (ends, expected) in [
            (vec![32], 1usize),
            (vec![16, 32], 2usize),
            (vec![8, 16, 24, 32], 4usize),
        ] {
            let frame = synthetic_frame(tables.high_band_count(), &ends);
            let mut writer = BitWriter::new();
            frame
                .write_mono_low_delay_payload(&mut writer, &header, &tables, true)
                .unwrap();
            let bytes = writer.finish();
            let mut reader = BitReader::new(&bytes);
            let parsed = LdSbrFrameParser::new(header.clone(), 48_000, 512, false, false)
                .unwrap()
                .parse(&mut reader)
                .unwrap();
            assert!(parsed.header_present);
            assert_eq!(parsed.prefix.left.grid.envelope_count(), expected);
            assert_eq!(parsed.left.envelopes.len(), expected);
            assert_eq!(parsed.left.noise.len(), if expected == 1 { 1 } else { 2 });
            assert_eq!(parsed.left_harmonics.len(), tables.high_band_count());
        }
    }

    #[test]
    fn writes_every_eld_transient_position_with_c_table_borders() {
        let header = header();
        let tables = LdSbrFrequencyTables::from_header(&header, 48_000).unwrap();
        for time_slots in [15u8, 16] {
            for position in 0..time_slots {
                let borders = low_delay_transient_borders(time_slots, position);
                let resolutions = borders
                    .windows(2)
                    .map(|border| border[1] - border[0] >= 6)
                    .collect::<Vec<_>>();
                let envelopes = borders
                    .windows(2)
                    .zip(&resolutions)
                    .map(|(border, &high)| SbrEncoderEnvelope {
                        start_slot: border[0] as usize,
                        end_slot: border[1] as usize,
                        bands: vec![
                            SbrEncoderBand {
                                energy: 0.0,
                                tonality: 0.0,
                            };
                            if high {
                                tables.high_band_count()
                            } else {
                                tables.low_band_count()
                            }
                        ],
                    })
                    .collect();
                let frame = SbrEncoderAnalysisFrame {
                    slots: vec![
                        QmfSlot {
                            real: vec![0.0; 64],
                            imaginary: vec![0.0; 64],
                        };
                        time_slots as usize
                    ],
                    envelopes,
                    transient_ratio: 8.0,
                    low_delay_transient_position: Some(position),
                    low_delay_frequency_resolution: Some(resolutions.clone()),
                    low_delay_amp_resolution: None,
                    low_delay_global_tonality: None,
                    low_delay_envelope_coding: None,
                    low_delay_noise_coding: None,
                    low_delay_invf_modes: None,
                    low_delay_patch_map: None,
                    low_delay_prequant_debug: None,
                };
                let mut writer = BitWriter::new();
                frame
                    .write_mono_low_delay_payload(&mut writer, &header, &tables, false)
                    .unwrap();
                let bytes = writer.finish();
                let mut reader = BitReader::new(&bytes);
                let parsed = LdSbrFrameParser::new(
                    header.clone(),
                    48_000,
                    if time_slots == 15 { 480 } else { 512 },
                    false,
                    false,
                )
                .unwrap()
                .parse(&mut reader)
                .unwrap();
                assert!(parsed.prefix.left.grid.transient);
                assert_eq!(parsed.prefix.left.grid.borders, borders);
                assert_eq!(parsed.prefix.left.grid.frequency_resolution, resolutions);
            }
        }
    }

    #[test]
    fn writes_eld_crc10_over_the_exact_unaligned_payload_region() {
        let header = header();
        let tables = LdSbrFrequencyTables::from_header(&header, 48_000).unwrap();
        let frame = synthetic_frame(tables.high_band_count(), &[32]);
        let mut writer = BitWriter::new();
        frame
            .write_mono_low_delay_payload_with_crc(&mut writer, &header, &tables, true, true)
            .unwrap();
        let mut bytes = writer.finish();
        let mut parser = LdSbrFrameParser::new(header.clone(), 48_000, 512, false, true).unwrap();
        assert!(parser.parse(&mut BitReader::new(&bytes)).is_ok());
        bytes[1] ^= 0x20; // first payload bit after the ten-bit CRC
        let mut parser = LdSbrFrameParser::new(header, 48_000, 512, false, true).unwrap();
        assert!(parser.parse(&mut BitReader::new(&bytes)).is_err());
    }

    #[test]
    fn writes_uncoupled_eld_stereo_with_independent_grids_and_crc() {
        let header = header();
        let tables = LdSbrFrequencyTables::from_header(&header, 48_000).unwrap();
        let left = synthetic_frame(tables.high_band_count(), &[32]);
        let mut right = synthetic_frame(tables.high_band_count(), &[16, 32]);
        right.slots.truncate(16);
        right.envelopes[0].end_slot = 8;
        right.envelopes[1].start_slot = 8;
        right.envelopes[1].end_slot = 16;
        let mut writer = BitWriter::new();
        SbrEncoderAnalysisFrame::write_stereo_low_delay_payload(
            &left,
            &right,
            &mut writer,
            &header,
            &tables,
            true,
            true,
        )
        .unwrap();
        let bytes = writer.finish();
        let frame = LdSbrFrameParser::new(header, 48_000, 512, true, true)
            .unwrap()
            .parse(&mut BitReader::new(&bytes))
            .unwrap();
        assert!(!frame.prefix.coupling);
        assert_eq!(frame.left.envelopes.len(), 1);
        assert_eq!(frame.right.as_ref().unwrap().envelopes.len(), 2);

        let active_header = frame.active_header.clone();
        let coupled = synthetic_frame(tables.high_band_count(), &[32]);
        let mut writer = BitWriter::new();
        SbrEncoderAnalysisFrame::write_stereo_low_delay_payload(
            &coupled,
            &coupled,
            &mut writer,
            &active_header,
            &tables,
            false,
            false,
        )
        .unwrap();
        let bytes = writer.finish();
        let parsed = LdSbrFrameParser::new(active_header, 48_000, 512, true, false)
            .unwrap()
            .parse(&mut BitReader::new(&bytes))
            .unwrap();
        assert!(parsed.prefix.coupling);
        assert_eq!(parsed.left.envelopes.len(), 1);
        assert_eq!(parsed.right.unwrap().envelopes.len(), 1);
    }

    #[test]
    fn uncoupled_eld_stereo_keeps_independent_delta_histories() {
        let header = header();
        let tables = LdSbrFrequencyTables::from_header(&header, 48_000).unwrap();
        let mut left_state = LowDelaySbrCodingState::default();
        let mut right_state = LowDelaySbrCodingState::default();
        let mut left = synthetic_frame(tables.high_band_count(), &[32]);
        let mut right = left.clone();
        left.prepare_mono_low_delay_coding(&header, &tables, true, &mut left_state);
        right.prepare_mono_low_delay_coding(&header, &tables, true, &mut right_state);

        let mut left_next = left.clone();
        let mut right_next = right.clone();
        left_next.slots[0].real[0] = 1.0;
        right_next.slots[0].real[1] = 1.0;
        left_next.low_delay_envelope_coding = None;
        left_next.low_delay_noise_coding = None;
        right_next.low_delay_envelope_coding = None;
        right_next.low_delay_noise_coding = None;
        left_next.prepare_mono_low_delay_coding(&header, &tables, false, &mut left_state);
        right_next.prepare_mono_low_delay_coding(&header, &tables, false, &mut right_state);
        assert!(!left_next.low_delay_envelope_coding.as_ref().unwrap()[0].time);
        assert!(!right_next.low_delay_envelope_coding.as_ref().unwrap()[0].time);

        let mut writer = BitWriter::new();
        SbrEncoderAnalysisFrame::write_stereo_low_delay_payload(
            &left_next,
            &right_next,
            &mut writer,
            &header,
            &tables,
            false,
            false,
        )
        .unwrap();
        let parsed = LdSbrFrameParser::new(header, 48_000, 512, true, false)
            .unwrap()
            .parse(&mut BitReader::new(&writer.finish()))
            .unwrap();
        assert!(!parsed.prefix.coupling);
    }

    #[test]
    fn coupled_eld_stereo_tracks_level_balance_and_noise_histories() {
        let header = header();
        let tables = LdSbrFrequencyTables::from_header(&header, 48_000).unwrap();
        let mut level_state = LowDelaySbrCodingState::default();
        let mut balance_state = LowDelaySbrCodingState::default();
        let mut left = synthetic_frame(tables.high_band_count(), &[32]);
        let mut right = left.clone();
        for envelope in [&mut left.envelopes[0], &mut right.envelopes[0]] {
            for band in &mut envelope.bands {
                band.energy = 64.0;
            }
        }
        SbrEncoderAnalysisFrame::prepare_coupled_low_delay_coding(
            &mut left,
            &mut right,
            &header,
            &tables,
            true,
            &mut level_state,
            &mut balance_state,
        )
        .unwrap();

        let mut left_next = left.clone();
        let mut right_next = right.clone();
        SbrEncoderAnalysisFrame::prepare_coupled_low_delay_coding(
            &mut left_next,
            &mut right_next,
            &header,
            &tables,
            false,
            &mut level_state,
            &mut balance_state,
        )
        .unwrap();
        assert!(!left_next.low_delay_envelope_coding.as_ref().unwrap()[0].time);
        assert!(!right_next.low_delay_envelope_coding.as_ref().unwrap()[0].time);
        assert!(left_next.low_delay_noise_coding.as_ref().unwrap()[0].time);
        assert!(right_next.low_delay_noise_coding.as_ref().unwrap()[0].time);
        assert!(left_next.low_delay_noise_coding.as_ref().unwrap()[0]
            .deltas
            .iter()
            .all(|&delta| delta == 0));
        assert!(right_next.low_delay_noise_coding.as_ref().unwrap()[0]
            .deltas
            .iter()
            .all(|&delta| delta == 0));

        let mut writer = BitWriter::new();
        SbrEncoderAnalysisFrame::write_stereo_low_delay_payload(
            &left_next,
            &right_next,
            &mut writer,
            &header,
            &tables,
            false,
            false,
        )
        .unwrap();
        let parsed = LdSbrFrameParser::new(header, 48_000, 512, true, false)
            .unwrap()
            .parse(&mut BitReader::new(&writer.finish()))
            .unwrap();
        assert!(parsed.prefix.coupling);
    }

    #[test]
    fn low_delay_analysis_uses_eld_32_and_64_band_slot_geometry() {
        let header = header();
        let single_header = (0..16)
            .find_map(|stop_frequency| {
                let candidate = LdSbrHeader {
                    stop_frequency,
                    ..header.clone()
                };
                let tables = LdSbrFrequencyTables::from_header(&candidate, 48_000).ok()?;
                (tables.high.last().copied()? <= 32).then_some(candidate)
            })
            .expect("ELD single-rate table must fit the 32-band CLDFB");
        for frame_length in [480usize, 512] {
            let mut single =
                SbrEncoderAnalysis::new_low_delay(&single_header, 24_000, false).unwrap();
            let frame = single
                .analyze_low_delay(&vec![0.0; frame_length], frame_length)
                .unwrap();
            assert_eq!(frame.slots.len(), frame_length / 32);
            assert_eq!(frame.slots[0].real.len(), 32);

            let mut dual = SbrEncoderAnalysis::new_low_delay(&header, 24_000, true).unwrap();
            let frame = dual
                .analyze_low_delay(&vec![0.0; 2 * frame_length], frame_length)
                .unwrap();
            assert_eq!(frame.slots.len(), frame_length / 32);
            assert_eq!(frame.slots[0].real.len(), 64);

            let mut transient = SbrEncoderAnalysis::new_low_delay(&header, 24_000, true).unwrap();
            let mut impulse = vec![0.0; 2 * frame_length];
            impulse[frame_length / 2] = 32_000.0;
            let frame = transient.analyze_low_delay(&impulse, frame_length).unwrap();
            assert!(frame.transient_ratio > 4.0);
            assert!(frame.low_delay_transient_position.is_some());
            assert!(matches!(frame.envelopes.len(), 2 | 3));
        }
    }

    #[test]
    fn low_delay_prequant_energy_exposes_c_comparable_fixed_snapshot() {
        let mut analysis = SbrEncoderAnalysis::new_low_delay(&header(), 24_000, true).unwrap();
        let input = (0..1024)
            .map(|sample| f64::from(((sample as f64 * 0.071).sin() * 12_000.0) as i16) as f32)
            .collect::<Vec<_>>();
        let frame = analysis.analyze_low_delay(&input, 512).unwrap();
        let fixed = frame.low_delay_prequant_debug.unwrap();
        assert_eq!(fixed.ybuffer_scales.0, 15);
        assert_eq!(fixed.qmf_scale, 1);
        assert_eq!(fixed.common_scale, fixed.ybuffer_scales.1 - 7);
        assert_eq!(fixed.counts.len(), frame.envelopes.len());
        assert_eq!(fixed.energies.len(), frame.envelopes.len());
        assert!(fixed.energies[0].iter().all(|&energy| energy >= 0));
    }

    #[test]
    fn low_delay_transient_detector_keeps_two_slot_candidate_history() {
        let mut energy_history = [1.0, 1.0];
        let mut ratio_history = [1.0, 1.0];
        let mut candidate_history = [false, false];
        assert_eq!(
            detect_low_delay_transient(
                &[1.0, 1.0, 1.0, 1.0],
                &mut energy_history,
                &mut ratio_history,
                &mut candidate_history,
            ),
            None
        );

        // The previous frame's final energy is the denominator for a new
        // frame-edge onset; this is the state the old frame-local peak/mean
        // heuristic could not preserve.
        assert_eq!(
            detect_low_delay_transient(
                &[10.0, 8.0, 1.0, 4.0],
                &mut energy_history,
                &mut ratio_history,
                &mut candidate_history,
            ),
            Some(2)
        );

        // A candidate in the final two slots is delayed until the following
        // call, where it becomes frame position zero or one.
        assert_eq!(
            detect_low_delay_transient(
                &[1.0, 1.0, 1.0, 10.0],
                &mut energy_history,
                &mut ratio_history,
                &mut candidate_history,
            ),
            None
        );
        assert_eq!(
            detect_low_delay_transient(
                &[1.0, 1.0, 1.0, 1.0],
                &mut energy_history,
                &mut ratio_history,
                &mut candidate_history,
            ),
            Some(1)
        );
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn low_delay_transient_lookahead_positions_match_c_detector() {
        const FRAMES: usize = 4;
        const SLOTS: usize = 16;
        const BANDS: usize = 64;
        const LOOKAHEAD: usize = 2;
        const BASELINE: i32 = 1 << 22;
        let mut c_energy = vec![BASELINE; FRAMES * (SLOTS + LOOKAHEAD) * BANDS];
        let set_slot = |energy: &mut [i32], frame: usize, slot: usize, value: i32| {
            let start = (frame * (SLOTS + LOOKAHEAD) + LOOKAHEAD + slot) * BANDS;
            energy[start..start + BANDS].fill(value);
        };
        set_slot(&mut c_energy, 1, 5, BASELINE * 12);
        set_slot(&mut c_energy, 2, 14, BASELINE * 12);
        let mut c_info = [0_u8; FRAMES * 3];
        assert_eq!(
            unsafe {
                crate::sys::fdk_sbr_fast_transient_test(
                    c_energy.as_ptr(),
                    FRAMES as i32,
                    SLOTS as i32,
                    BANDS as i32,
                    375,
                    12,
                    c_info.as_mut_ptr(),
                )
            },
            0
        );

        let mut energy_history = [0.0; 2];
        let mut ratio_history = [0.0; 2];
        let mut candidate_history = [false; 2];
        let mut rust_positions = Vec::new();
        for frame in 0..FRAMES {
            let mut energies = vec![1.0; SLOTS];
            if frame == 1 {
                energies[5] = 12.0;
            }
            if frame == 2 {
                energies[14] = 12.0;
            }
            rust_positions.push(detect_low_delay_transient(
                &energies,
                &mut energy_history,
                &mut ratio_history,
                &mut candidate_history,
            ));
        }
        let c_positions = c_info
            .chunks_exact(3)
            .map(|info| (info[1] != 0).then_some(info[0] as usize))
            .collect::<Vec<_>>();
        assert_eq!(rust_positions[1..], c_positions[1..]);
        assert_eq!(c_info[2 * 3 + 2], 1);
        assert_eq!(rust_positions[3], Some(0));
    }

    #[test]
    fn low_delay_envelope_coding_uses_c_weighted_previous_frame_decision() {
        let header = header();
        let tables = LdSbrFrequencyTables::from_header(&header, 48_000).unwrap();
        let mut state = LowDelaySbrCodingState::default();
        let mut first = synthetic_frame(tables.high_band_count(), &[32]);
        for (band, range) in first.envelopes[0]
            .bands
            .iter_mut()
            .zip(tables.high.windows(2))
        {
            band.energy = 64.0 * f64::from(range[1] - range[0]);
        }
        first.prepare_mono_low_delay_coding(&header, &tables, true, &mut state);
        assert!(!first.low_delay_envelope_coding.as_ref().unwrap()[0].time);

        let mut second = first.clone();
        second.low_delay_envelope_coding = None;
        second.prepare_mono_low_delay_coding(&header, &tables, false, &mut state);
        let coding = second.low_delay_envelope_coding.as_ref().unwrap();
        assert!(!coding[0].time);
        assert!(coding[0].deltas[1..].iter().all(|&delta| delta == 0));
        let noise = second.low_delay_noise_coding.as_ref().unwrap();
        assert!(noise[0].time);
        assert!(noise[0].deltas.iter().all(|&delta| delta == 0));

        let mut writer = BitWriter::new();
        second
            .write_mono_low_delay_payload(&mut writer, &header, &tables, false)
            .unwrap();
        assert!(writer.bits_written() > 0);

        let mut header_frame = synthetic_frame(tables.high_band_count(), &[8, 16, 32]);
        let mut header_state = LowDelaySbrCodingState::default();
        header_frame.prepare_mono_low_delay_coding(&header, &tables, true, &mut header_state);
        let noise = header_frame.low_delay_noise_coding.as_ref().unwrap();
        assert!(!noise[0].time);
        assert_eq!(noise.len(), 2);
    }

    #[test]
    fn low_delay_noise_floor_tracks_qmf_tonality_and_smoothing() {
        let header = header();
        let tables = LdSbrFrequencyTables::from_header(&header, 48_000).unwrap();
        let mut tonal = synthetic_frame(tables.high_band_count(), &[32]);
        for (slot, qmf) in tonal.slots.iter_mut().enumerate() {
            for band in usize::from(tables.noise[0])..usize::from(*tables.noise.last().unwrap()) {
                let perturbation = (((slot * 29 + band * 3) % 17) as f64 - 8.0) * 0.0001;
                qmf.real[band] = (slot as f64 * 0.2).cos() * 0.2 + perturbation;
                qmf.imaginary[band] = (slot as f64 * 0.2).sin() * 0.2 - perturbation;
            }
        }
        let mut noisy = tonal.clone();
        for (slot, qmf) in noisy.slots.iter_mut().enumerate() {
            for band in usize::from(tables.noise[0])..usize::from(*tables.noise.last().unwrap()) {
                qmf.real[band] = if (slot + band) % 2 == 0 { 1.0 } else { -0.3 };
                qmf.imaginary[band] = if (slot * 3 + band) % 5 < 2 { 0.7 } else { -0.8 };
            }
        }
        let tonal_levels = estimate_low_delay_noise_levels(
            &tonal,
            &tables,
            &mut LowDelaySbrCodingState::default(),
        );
        let noisy_levels = estimate_low_delay_noise_levels(
            &noisy,
            &tables,
            &mut LowDelaySbrCodingState::default(),
        );
        assert!(tonal_levels[0]
            .iter()
            .zip(&noisy_levels[0])
            .all(|(tonal, noisy)| tonal > noisy));

        let mut high_invf = tonal.clone();
        high_invf.low_delay_invf_modes = Some(vec![3; tables.noise_band_count()]);
        let high_invf_levels = estimate_low_delay_noise_levels(
            &high_invf,
            &tables,
            &mut LowDelaySbrCodingState::default(),
        );
        assert!(high_invf_levels[0]
            .iter()
            .zip(&tonal_levels[0])
            .all(|(high, off)| high >= off));

        let mut state = LowDelaySbrCodingState::default();
        let first = estimate_low_delay_noise_levels(&tonal, &tables, &mut state);
        let second = estimate_low_delay_noise_levels(&tonal, &tables, &mut state);
        assert_ne!(first, second);
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn low_delay_first_envelope_time_streak_matches_c_code_envelope() {
        let frames = [
            vec![20_i8, 20, 20, 20],
            vec![21, 21, 21, 21],
            vec![22, 22, 22, 22],
            vec![23, 23, 23, 23],
            vec![23, 10, 30, 15],
            vec![24, 24, 24, 24],
        ];
        let flat = frames.iter().flatten().copied().collect::<Vec<_>>();
        let mut c_coded = vec![0_i8; flat.len()];
        let mut c_directions = vec![0_i32; frames.len()];
        assert_eq!(
            unsafe {
                crate::sys::fdk_sbr_code_envelope_test(
                    flat.as_ptr(),
                    frames.len() as i32,
                    frames[0].len() as i32,
                    c_coded.as_mut_ptr(),
                    c_directions.as_mut_ptr(),
                )
            },
            0
        );

        let mut previous = None;
        let mut streak = 0_usize;
        let mut rust_coded = Vec::new();
        let mut rust_directions = Vec::new();
        for (frame, values) in frames.iter().enumerate() {
            let mut values = values.clone();
            constrain_frequency_deltas(&mut values, SbrHuffmanBook::EnvelopeLevel30Frequency);
            let coding = prepare_low_delay_delta_coding(
                &[values],
                &mut previous,
                SbrHuffmanBook::EnvelopeLevel30Frequency,
                SbrHuffmanBook::EnvelopeLevel30Time,
                6,
                1,
                frame == 0,
                low_delay_envelope_time_weight_q15(streak),
            );
            rust_directions.push(i32::from(coding[0].time));
            rust_coded.extend_from_slice(&coding[0].deltas);
            streak = if coding[0].time { streak + 1 } else { 0 };
        }
        assert_eq!(rust_directions, c_directions);
        assert_eq!(rust_coded, c_coded);
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn coupled_envelope_delta_bit_costs_match_c_tables() {
        let previous = [22_i8, 22, 22, 22];
        for amp_res_1_5 in [false, true] {
            for (channel, current) in [(0, [23_i8, 10, 30, 15]), (1, [24_i8, 0, 20, 8])] {
                let (frequency_book, time_book, start_bits, divisor) = match (amp_res_1_5, channel)
                {
                    (false, 0) => (
                        SbrHuffmanBook::EnvelopeLevel30Frequency,
                        SbrHuffmanBook::EnvelopeLevel30Time,
                        6,
                        1,
                    ),
                    (false, 1) => (
                        SbrHuffmanBook::EnvelopeBalance30Frequency,
                        SbrHuffmanBook::EnvelopeBalance30Time,
                        5,
                        2,
                    ),
                    (true, 0) => (
                        SbrHuffmanBook::EnvelopeLevel15Frequency,
                        SbrHuffmanBook::EnvelopeLevel15Time,
                        7,
                        1,
                    ),
                    (true, 1) => (
                        SbrHuffmanBook::EnvelopeBalance15Frequency,
                        SbrHuffmanBook::EnvelopeBalance15Time,
                        6,
                        2,
                    ),
                    _ => unreachable!(),
                };
                let rust_frequency = start_bits
                    + current
                        .windows(2)
                        .map(|pair| {
                            encode_sbr_huffman(frequency_book, (pair[1] - pair[0]) / divisor)
                                .unwrap()
                                .len()
                        })
                        .sum::<usize>();
                let rust_time = current
                    .iter()
                    .zip(previous)
                    .map(|(&value, old)| {
                        encode_sbr_huffman(time_book, (value - old) / divisor)
                            .unwrap()
                            .len()
                    })
                    .sum::<usize>();
                let mut c_frequency = 0;
                let mut c_time = 0;
                assert_eq!(
                    unsafe {
                        crate::sys::fdk_sbr_coupled_delta_bits_test(
                            current.as_ptr(),
                            previous.as_ptr(),
                            current.len() as i32,
                            channel,
                            i32::from(amp_res_1_5),
                            &mut c_frequency,
                            &mut c_time,
                        )
                    },
                    0
                );
                assert_eq!(
                    rust_frequency, c_frequency as usize,
                    "frequency amp1.5={amp_res_1_5}, channel={channel}"
                );
                assert_eq!(
                    rust_time, c_time as usize,
                    "time amp1.5={amp_res_1_5}, channel={channel}"
                );
            }
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn coupled_envelope_stateful_directions_and_deltas_match_c() {
        let level = [
            [20_i8, 20, 20, 20],
            [21, 21, 21, 21],
            [22, 22, 22, 22],
            [23, 10, 30, 15],
            [24, 24, 24, 24],
        ];
        let balance = [
            [12_i8, 12, 12, 12],
            [14, 14, 14, 14],
            [16, 16, 16, 16],
            [24, 0, 20, 8],
            [18, 18, 18, 18],
        ];
        for amp_res_1_5 in [false, true] {
            for (channel, frames) in [(0, &level), (1, &balance)] {
                let (frequency_book, time_book, start_bits, divisor) = match (amp_res_1_5, channel)
                {
                    (false, 0) => (
                        SbrHuffmanBook::EnvelopeLevel30Frequency,
                        SbrHuffmanBook::EnvelopeLevel30Time,
                        6,
                        1,
                    ),
                    (false, 1) => (
                        SbrHuffmanBook::EnvelopeBalance30Frequency,
                        SbrHuffmanBook::EnvelopeBalance30Time,
                        5,
                        2,
                    ),
                    (true, 0) => (
                        SbrHuffmanBook::EnvelopeLevel15Frequency,
                        SbrHuffmanBook::EnvelopeLevel15Time,
                        7,
                        1,
                    ),
                    (true, 1) => (
                        SbrHuffmanBook::EnvelopeBalance15Frequency,
                        SbrHuffmanBook::EnvelopeBalance15Time,
                        6,
                        2,
                    ),
                    _ => unreachable!(),
                };
                let flat = frames.iter().flatten().copied().collect::<Vec<_>>();
                let mut c_coded = vec![0_i8; flat.len()];
                let mut c_directions = vec![0_i32; frames.len()];
                assert_eq!(
                    unsafe {
                        crate::sys::fdk_sbr_code_envelope_coupled_test(
                            flat.as_ptr(),
                            frames.len() as i32,
                            frames[0].len() as i32,
                            channel,
                            i32::from(amp_res_1_5),
                            c_coded.as_mut_ptr(),
                            c_directions.as_mut_ptr(),
                        )
                    },
                    0
                );
                let mut previous = None;
                let mut streak = 0;
                let mut rust_coded = Vec::new();
                let mut rust_directions = Vec::new();
                for (index, frame) in frames.iter().enumerate() {
                    let mut values = frame.to_vec();
                    if channel == 0 {
                        constrain_frequency_deltas(&mut values, frequency_book);
                    } else {
                        constrain_scaled_frequency_deltas(&mut values, frequency_book, divisor);
                    }
                    let coding = prepare_low_delay_delta_coding(
                        &[values],
                        &mut previous,
                        frequency_book,
                        time_book,
                        start_bits,
                        divisor,
                        index == 0,
                        low_delay_envelope_time_weight_q15(streak),
                    );
                    rust_directions.push(i32::from(coding[0].time));
                    rust_coded.extend_from_slice(&coding[0].deltas);
                    streak = if coding[0].time { streak + 1 } else { 0 };
                }
                assert_eq!(
                    rust_directions, c_directions,
                    "amp1.5={amp_res_1_5}, channel={channel}"
                );
                assert_eq!(
                    rust_coded, c_coded,
                    "amp1.5={amp_res_1_5}, channel={channel}"
                );
            }
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn first_envelope_time_weight_rounding_matches_c() {
        for streak in 0..=6 {
            for time_bits in 0..=100 {
                let rust = low_delay_first_time_threshold(
                    time_bits,
                    low_delay_envelope_time_weight_q15(streak),
                );
                let c = unsafe {
                    crate::sys::fdk_sbr_first_env_threshold_test(time_bits as i32, streak as i32)
                };
                assert_eq!(rust as i32, c, "time bits {time_bits}, streak {streak}");
            }
        }
    }

    #[test]
    fn qmf_envelope_analysis_preserves_silence_and_detects_tonal_energy() {
        let mut analysis = SbrEncoderAnalysis::new(&header(), 48_000).unwrap();
        let silence = analysis.analyze(&vec![0.0; 2048]).unwrap();
        assert_eq!(silence.slots.len(), 32);
        assert_eq!(silence.transient_ratio, 0.0);
        assert!(silence.envelopes[0]
            .bands
            .iter()
            .all(|band| band.energy == 0.0 && band.tonality == 0.0));

        let input = (0..2048)
            .map(|index| (2.0 * std::f32::consts::PI * 13.0 * index as f32 / 128.0).sin())
            .collect::<Vec<_>>();
        let tonal = analysis.analyze(&input).unwrap();
        assert!(tonal
            .envelopes
            .iter()
            .flat_map(|env| &env.bands)
            .any(|band| band.energy > 0.0));
        assert!(tonal
            .envelopes
            .iter()
            .flat_map(|env| &env.bands)
            .any(|band| band.tonality > 0.5));

        let fill = tonal
            .write_mono_fill_element(&header(), analysis.frequency_tables(), true)
            .unwrap();
        let payload = parse_sbr_fill_element(&mut BitReader::new(&fill))
            .unwrap()
            .unwrap();
        let mut parser = SbrMonoFrameParser::new(header(), 48_000, 1024).unwrap();
        let decoded = parser.parse(&payload).unwrap();
        assert_eq!(
            decoded.values.envelopes[0].len(),
            analysis.frequency_tables().high_band_count()
        );
        assert_eq!(
            decoded.values.noise[0].len(),
            analysis.frequency_tables().noise_band_count()
        );

        let mut transient_analysis = SbrEncoderAnalysis::new(&header(), 48_000).unwrap();
        let mut impulse = vec![0.0; 2048];
        impulse[128] = 100.0;
        let transient = transient_analysis.analyze(&impulse).unwrap();
        assert_eq!(transient.envelopes.len(), 2);
        let fill = transient
            .write_mono_fill_element(&header(), analysis.frequency_tables(), false)
            .unwrap();
        let payload = parse_sbr_fill_element(&mut BitReader::new(&fill))
            .unwrap()
            .unwrap();
        let decoded = parser.parse(&payload).unwrap();
        assert_eq!(decoded.values.envelopes.len(), 2);
        assert_eq!(decoded.values.noise.len(), 2);
        assert!(decoded.control.grid.transient);
        assert_ne!(decoded.control.grid.borders, vec![0, 8, 16]);

        let mut harmonic = tonal.clone();
        for band in harmonic.envelopes.iter_mut().flat_map(|env| &mut env.bands) {
            band.tonality = 1.0;
        }
        let fill = harmonic
            .write_mono_fill_element(&header(), analysis.frequency_tables(), false)
            .unwrap();
        let payload = parse_sbr_fill_element(&mut BitReader::new(&fill))
            .unwrap()
            .unwrap();
        let decoded = parser.parse(&payload).unwrap();
        assert!(decoded.harmonics.iter().any(|&enabled| enabled));

        let stereo_fill = SbrEncoderAnalysisFrame::write_stereo_fill_element(
            &tonal,
            &tonal,
            &header(),
            analysis.frequency_tables(),
            true,
        )
        .unwrap();
        let payload = parse_sbr_fill_element(&mut BitReader::new(&stereo_fill))
            .unwrap()
            .unwrap();
        let mut stereo_parser = SbrStereoFrameParser::new(header(), 48_000, 1024).unwrap();
        let stereo = stereo_parser.parse(&payload).unwrap();
        assert!(stereo.coupling);
        assert_eq!(stereo.left.envelopes.len(), tonal.envelopes.len());
        assert_eq!(stereo.right.envelopes.len(), tonal.envelopes.len());

        let mut decorrelated = tonal.clone();
        for (slot_index, slot) in decorrelated.slots.iter_mut().enumerate() {
            for band in 0..slot.real.len() {
                slot.real[band] = ((slot_index * 17 + band * 13) as f64).sin();
                slot.imaginary[band] = ((slot_index * 7 + band * 19) as f64).cos();
            }
        }
        let fill = SbrEncoderAnalysisFrame::write_stereo_fill_element(
            &tonal,
            &decorrelated,
            &header(),
            analysis.frequency_tables(),
            false,
        )
        .unwrap();
        let payload = parse_sbr_fill_element(&mut BitReader::new(&fill))
            .unwrap()
            .unwrap();
        assert!(!stereo_parser.parse(&payload).unwrap().coupling);

        let ps = PsEncoderFrame {
            iid: vec![0; 20],
            icc: vec![0; 20],
        }
        .write_sbr_extension(true)
        .unwrap();
        let fill = tonal
            .write_mono_fill_element_with_extension(
                &header(),
                analysis.frequency_tables(),
                false,
                Some(&ps),
            )
            .unwrap();
        let payload = parse_sbr_fill_element(&mut BitReader::new(&fill))
            .unwrap()
            .unwrap();
        let sbr = parser.parse(&payload).unwrap();
        let ps = PsParser::new()
            .parse_sbr_extension(&sbr.extended_data, 32)
            .unwrap()
            .unwrap();
        assert_eq!(ps.iid_mapped_20[0], vec![0; 20]);
    }

    #[test]
    fn low_delay_single_envelope_amp_resolution_uses_fdk_bitrate_regions() {
        let mut frame = synthetic_frame(4, &[32]);
        frame.select_low_delay_amp_resolution(27_999);
        assert_eq!(frame.low_delay_amp_resolution, Some(true));
        frame.select_low_delay_amp_resolution(48_001);
        assert_eq!(frame.low_delay_amp_resolution, Some(false));
        frame.select_low_delay_amp_resolution(32_000);
        assert_eq!(frame.low_delay_amp_resolution, Some(true));

        frame.low_delay_transient_position = Some(3);
        frame.select_low_delay_amp_resolution(64_000);
        assert_eq!(frame.low_delay_amp_resolution, None);

        let mut left = synthetic_frame(4, &[32]);
        let mut right = left.clone();
        for slot in [&mut left.slots, &mut right.slots] {
            for sample in slot {
                sample.real.fill(1.0);
            }
        }
        left.low_delay_amp_resolution = Some(true);
        right.low_delay_amp_resolution = Some(false);
        assert!(!SbrEncoderAnalysisFrame::uses_low_delay_coupling(
            &left, &right
        ));
        right.low_delay_amp_resolution = Some(true);
        assert!(SbrEncoderAnalysisFrame::uses_low_delay_coupling(
            &left, &right
        ));
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn low_delay_global_tonality_amp_resolution_matches_c_threshold() {
        const ESTIMATES: usize = 2;
        const SLOTS: usize = 16;
        const BANDS: usize = 8;
        for quota in [10.0, 20.0, 22.0, 23.0, 60.0] {
            let quota_fixed = (quota * 1.0e-6 * 2_147_483_648.0) as i32;
            let quotas = vec![quota_fixed; ESTIMATES * BANDS];
            let mut energies = vec![0_i32; SLOTS * BANDS];
            for slot in 0..SLOTS {
                for band in 1..BANDS {
                    energies[slot * BANDS + band] = ((band + 1) * 1024) as i32;
                }
            }
            let mut current = 0;
            let mut steady = 0;
            let mut ignored = 0;
            assert_eq!(
                unsafe {
                    crate::sys::fdk_sbr_global_tonality_test(
                        quotas.as_ptr(),
                        energies.as_ptr(),
                        ESTIMATES as i32,
                        SLOTS as i32,
                        BANDS as i32,
                        1,
                        0,
                        &mut current,
                        &mut steady,
                        &mut ignored,
                    )
                },
                0
            );
            let mut global = 0;
            let mut c_amp_3db = 0;
            assert_eq!(
                unsafe {
                    crate::sys::fdk_sbr_global_tonality_test(
                        quotas.as_ptr(),
                        energies.as_ptr(),
                        ESTIMATES as i32,
                        SLOTS as i32,
                        BANDS as i32,
                        1,
                        current,
                        &mut ignored,
                        &mut global,
                        &mut c_amp_3db,
                    )
                },
                0
            );
            assert_eq!(
                quota * (10.0 / 3.0) <= 75.0,
                c_amp_3db != 0,
                "steady quota {quota}, C current {current}, global {global}"
            );
        }

        let mut c_previous = 0;
        let mut rust_previous = 0.0;
        for quota in [10.0, 60.0, 10.0, 30.0] {
            let quota_fixed = (quota * 1.0e-6 * 2_147_483_648.0) as i32;
            let quotas = vec![quota_fixed; ESTIMATES * BANDS];
            let energies = vec![1024_i32; SLOTS * BANDS];
            let mut current = 0;
            let mut global = 0;
            let mut c_amp_3db = 0;
            assert_eq!(
                unsafe {
                    crate::sys::fdk_sbr_global_tonality_test(
                        quotas.as_ptr(),
                        energies.as_ptr(),
                        ESTIMATES as i32,
                        SLOTS as i32,
                        BANDS as i32,
                        1,
                        c_previous,
                        &mut current,
                        &mut global,
                        &mut c_amp_3db,
                    )
                },
                0
            );
            let rust_current = quota * (10.0 / 3.0);
            let rust_global = 0.5 * (rust_current + rust_previous);
            assert_eq!(
                rust_global <= 75.0,
                c_amp_3db != 0,
                "sequence quota {quota}, Rust global {rust_global}, C global {global}"
            );
            c_previous = current;
            rust_previous = rust_current;
        }
    }

    #[test]
    fn validates_envelope_layouts_and_extended_payload_sizes() {
        let analysis = SbrEncoderAnalysis::new(&header(), 48_000).unwrap();
        let bands = analysis.frequency_tables().high_band_count();
        let empty = synthetic_frame(bands, &[]);
        assert_eq!(
            empty.write_mono_fill_element(&header(), analysis.frequency_tables(), false),
            Err(SbrEncoderError::EnvelopeLayoutMismatch)
        );
        let wrong = synthetic_frame(bands + 1, &[32]);
        assert_eq!(
            wrong.write_mono_fill_element(&header(), analysis.frequency_tables(), false),
            Err(SbrEncoderError::EnvelopeLayoutMismatch)
        );
        let valid = synthetic_frame(bands, &[32]);
        assert_eq!(
            valid.write_mono_fill_element_with_extension(
                &header(),
                analysis.frequency_tables(),
                false,
                Some(&vec![0; 270]),
            ),
            Err(SbrEncoderError::PayloadTooLarge(270))
        );
        let fill = valid
            .write_mono_fill_element_with_extension(
                &header(),
                analysis.frequency_tables(),
                false,
                Some(&[0xaa; 15]),
            )
            .unwrap();
        assert!(!fill.is_empty());

        assert_eq!(
            SbrEncoderAnalysisFrame::write_stereo_fill_element(
                &valid,
                &wrong,
                &header(),
                analysis.frequency_tables(),
                false,
            ),
            Err(SbrEncoderError::EnvelopeLayoutMismatch)
        );

        let mut invalid_header = header();
        invalid_header.start_frequency = 16;
        assert_eq!(
            SbrEncoderAnalysisFrame::write_stereo_fill_element(
                &valid,
                &valid,
                &invalid_header,
                analysis.frequency_tables(),
                true,
            ),
            Err(SbrEncoderError::Asc(
                crate::asc::AscError::InvalidLdSbrHeader
            ))
        );
    }

    #[test]
    fn amp_resolution_and_silent_stereo_take_alternate_books() {
        let analysis = SbrEncoderAnalysis::new(&header(), 48_000).unwrap();
        let bands = analysis.frequency_tables().high_band_count();
        let frame = synthetic_frame(bands, &[32]);
        let mut high_resolution = header();
        high_resolution.amp_resolution = true;
        assert!(!frame
            .write_mono_fill_element(&high_resolution, analysis.frequency_tables(), true)
            .unwrap()
            .is_empty());
        // Both channels have zero energy: correlation deliberately defaults
        // to one and selects coupled stereo coding.
        assert_eq!(stereo_correlation(&frame, &frame), 1.0);
        assert!(!SbrEncoderAnalysisFrame::write_stereo_fill_element(
            &frame,
            &frame,
            &high_resolution,
            analysis.frequency_tables(),
            true,
        )
        .unwrap()
        .is_empty());
        assert!(!SbrEncoderAnalysisFrame::write_stereo_fill_element(
            &frame,
            &frame,
            &high_resolution,
            analysis.frequency_tables(),
            false,
        )
        .unwrap()
        .is_empty());

        let mut left = frame.clone();
        let mut right = frame.clone();
        for (slot_index, (left_slot, right_slot)) in
            left.slots.iter_mut().zip(&mut right.slots).enumerate()
        {
            left_slot.real.fill(1.0);
            right_slot
                .real
                .fill(if slot_index & 1 == 0 { 1.0 } else { -1.0 });
        }
        assert!(stereo_correlation(&left, &right) < 0.8);
        assert!(!SbrEncoderAnalysisFrame::write_stereo_fill_element(
            &left,
            &right,
            &high_resolution,
            analysis.frequency_tables(),
            true,
        )
        .unwrap()
        .is_empty());
    }

    #[test]
    fn grid_writer_covers_fixfix_varfix_fixvar_and_invalid_boundaries() {
        let mut writer = BitWriter::new();
        write_mono_grid(&mut writer, &synthetic_frame(1, &[32])).unwrap();
        write_mono_grid(&mut writer, &synthetic_frame(1, &[16, 32])).unwrap();
        write_mono_grid(&mut writer, &synthetic_frame(1, &[8, 32])).unwrap();
        write_mono_grid(&mut writer, &synthetic_frame(1, &[24, 32])).unwrap();
        assert_eq!(
            write_mono_grid(&mut writer, &synthetic_frame(1, &[2, 32])),
            Err(SbrEncoderError::EnvelopeLayoutMismatch)
        );
    }

    #[test]
    fn delta_constraints_and_payload_packer_cover_limits() {
        let mut values = [0, 120, 0, 100];
        constrain_frequency_deltas(&mut values, SbrHuffmanBook::EnvelopeLevel30Frequency);
        assert!(values.windows(2).all(|pair| encode_sbr_huffman(
            SbrHuffmanBook::EnvelopeLevel30Frequency,
            pair[1] - pair[0]
        )
        .is_some()));
        let mut scaled = [0, 30, 0, 30];
        constrain_scaled_frequency_deltas(
            &mut scaled,
            SbrHuffmanBook::EnvelopeBalance30Frequency,
            2,
        );
        assert!(scaled.windows(2).all(|pair| encode_sbr_huffman(
            SbrHuffmanBook::EnvelopeBalance30Frequency,
            (pair[1] - pair[0]) / 2,
        )
        .is_some()));

        let mut short = BitWriter::new();
        short.write(0xaa, 8);
        assert!(!pack_fill_body(short).unwrap().is_empty());
        let mut long = BitWriter::new();
        for _ in 0..15 {
            long.write(0xaa, 8);
        }
        assert!(!pack_fill_body(long).unwrap().is_empty());
        let mut too_long = BitWriter::new();
        for _ in 0..270 {
            too_long.write(0, 8);
        }
        assert_eq!(
            pack_fill_body(too_long),
            Err(SbrEncoderError::PayloadTooLarge(270))
        );

        assert_eq!(
            write_sbr_code(
                &mut BitWriter::new(),
                SbrHuffmanBook::EnvelopeLevel30Frequency,
                127,
            ),
            Err(SbrEncoderError::UnrepresentableHuffmanSymbol(127))
        );
    }

    #[test]
    fn analysis_rejects_non_finite_and_empty_input() {
        let mut analysis = SbrEncoderAnalysis::new(&header(), 48_000).unwrap();
        assert_eq!(
            analysis.analyze(&[f32::NAN]),
            Err(SbrEncoderError::NonFiniteInput)
        );
        assert_eq!(analysis.analyze(&[]), Err(SbrEncoderError::EmptyFrame));
    }

    #[test]
    fn inverse_filter_detector_uses_c_regions_energy_reduction_and_hysteresis() {
        let mut state = InverseFilterBandState::default();
        assert_eq!(
            inverse_filter_decision(1.0, 15.0, 45.0, false, &mut state),
            3
        );
        assert_eq!(state.previous_sbr_region, 3);
        assert_eq!(state.previous_orig_region, 1);

        // The previous SBR region widens by one detector unit on both sides.
        assert_eq!(
            inverse_filter_decision(1.0, 13.5, 45.0, false, &mut state),
            3
        );
        assert_eq!(state.previous_sbr_region, 3);

        // FDK reduces the selected mode by four through zero at low energy.
        assert_eq!(
            inverse_filter_decision(1.0, 15.0, 20.0, false, &mut state),
            0
        );
        assert_eq!(
            inverse_filter_decision(1.0, 15.0, 33.0, false, &mut state),
            1
        );
        assert_eq!(
            inverse_filter_decision(1.0, 15.0, 45.0, true, &mut state),
            3
        );
    }

    #[test]
    fn second_order_complex_lpc_quota_separates_tones_from_noise() {
        let make_slots = |tonal: bool| {
            (0..16)
                .map(|index| {
                    let phase = index as f64 * 0.43;
                    let (real, imaginary) = if tonal {
                        let perturbation = (((index * 29 + 3) % 17) as f64 - 8.0) * 0.0001;
                        (
                            phase.cos() * 0.2 + perturbation,
                            phase.sin() * 0.2 - perturbation,
                        )
                    } else {
                        let real = (((index * 37 + 11) % 101) as f64 / 50.0) - 1.0;
                        let imaginary = (((index * 61 + 7) % 97) as f64 / 48.0) - 1.0;
                        (real, imaginary)
                    };
                    QmfSlot {
                        real: vec![real],
                        imaginary: vec![imaginary],
                    }
                })
                .collect::<Vec<_>>()
        };
        let (_, tonal) = lpc_band_statistics(&make_slots(true), 0, 1);
        let (_, noise) = lpc_band_statistics(&make_slots(false), 0, 1);
        assert!(tonal > 100.0, "tonal quota {tonal}");
        assert!(noise < tonal / 10.0, "noise quota {noise}, tonal {tonal}");
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn patch_mapping_matches_c_reset_patch() {
        for (sample_rate, channels) in [(32_000, 32), (44_100, 64), (48_000, 64), (96_000, 64)] {
            for start in [2, 5, 8] {
                let mut header = header();
                header.start_frequency = start;
                let Ok(tables) = LdSbrFrequencyTables::from_header(&header, sample_rate) else {
                    continue;
                };
                let usb = usize::from(*tables.master.last().unwrap());
                if usb > channels {
                    continue;
                }
                let rust = make_sbr_patch_map(&tables, sample_rate, channels);
                let mut c = vec![0_i8; channels];
                assert_eq!(
                    unsafe {
                        fdk_aac_sys::fdk_sbr_patch_map_test(
                            tables.master.as_ptr(),
                            tables.master.len() as i32,
                            i32::from(tables.high[0]),
                            sample_rate as i32,
                            channels as i32,
                            c.as_mut_ptr(),
                        )
                    },
                    0
                );
                let rust = rust[..usb]
                    .iter()
                    .map(|value| value.map_or(-1, |value| value as i8))
                    .collect::<Vec<_>>();
                assert_eq!(rust, c[..usb], "rate {sample_rate}, start {start}");
            }
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn lpc_tonality_quota_tracks_c_detector_regions() {
        // FDK's C_ALLOC_SCRATCH storage is process-global and is not safe when
        // two quota bridges execute concurrently in the parallel test runner.
        let _c_guard = TONALITY_QUOTA_C_LOCK.lock().unwrap();
        for tonal in [false, true] {
            let slots = (0..16)
                .map(|index| {
                    let phase = index as f64 * 0.43;
                    let (real, imaginary) = if tonal {
                        let perturbation = (((index * 29 + 3) % 17) as f64 - 8.0) * 0.0001;
                        (
                            phase.cos() * 0.2 + perturbation,
                            phase.sin() * 0.2 - perturbation,
                        )
                    } else {
                        (
                            ((((index * 37 + 11) % 101) as f64 / 50.0) - 1.0) * 0.2,
                            ((((index * 61 + 7) % 97) as f64 / 48.0) - 1.0) * 0.2,
                        )
                    };
                    QmfSlot {
                        real: vec![real],
                        imaginary: vec![imaginary],
                    }
                })
                .collect::<Vec<_>>();
            let real = slots
                .iter()
                .map(|slot| (slot.real[0] * 2_147_483_648.0) as i32)
                .collect::<Vec<_>>();
            let imaginary = slots
                .iter()
                .map(|slot| (slot.imaginary[0] * 2_147_483_648.0) as i32)
                .collect::<Vec<_>>();
            let mut c_quotas = [0; 2];
            for block in 0..2 {
                let mut c_energy = 0;
                assert_eq!(
                    unsafe {
                        fdk_aac_sys::fdk_sbr_tonality_quota_test(
                            real[block * 8..].as_ptr(),
                            imaginary[block * 8..].as_ptr(),
                            8,
                            &mut c_quotas[block],
                            &mut c_energy,
                        )
                    },
                    0
                );
                let rust_raw = fixed_lpc_quota(&slots[block * 8..block * 8 + 8], 0);
                assert_eq!(
                    rust_raw, c_quotas[block],
                    "raw quota block {block}, tonal {tonal}"
                );
            }
            let rust_quota = lpc_band_statistics(&slots, 0, 1).1;
            let c_quota = c_quotas
                .iter()
                .map(|&quota| f64::from(quota) / 2_147_483_648.0 / 1.0e-6)
                .sum::<f64>()
                / 2.0;
            if tonal {
                assert!(
                    c_quota > 100.0 && rust_quota > 100.0,
                    "C {c_quota}, Rust {rust_quota}"
                );
            } else {
                assert!(
                    c_quota < 10.0 && rust_quota < 10.0,
                    "C {c_quota}, Rust {rust_quota}"
                );
            }
            let c_coordinate = |quota: f64| (3.0 * (quota + 1.0).log2()).min(20.0);
            let rust_coordinate = |quota: f64| (3.0 * (quota + 1.0).log2()).min(20.0);
            assert_eq!(
                inverse_filter_region(c_coordinate(c_quota), &[1.0, 10.0, 14.0, 19.0], 0),
                inverse_filter_region(rust_coordinate(rust_quota), &[1.0, 10.0, 14.0, 19.0], 0),
                "C {c_quota}, Rust {rust_quota}"
            );
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn lpc_tonality_quota_matches_c_for_seven_slot_ld_segment() {
        let _c_guard = TONALITY_QUOTA_C_LOCK.lock().unwrap();
        for tonal in [false, true] {
            let slots = (0..7)
                .map(|index| {
                    let phase = index as f64 * 0.37;
                    let (real, imaginary) = if tonal {
                        (phase.cos() * 0.18, phase.sin() * 0.18)
                    } else {
                        (
                            ((((index * 19 + 5) % 43) as f64 / 21.0) - 1.0) * 0.18,
                            ((((index * 31 + 9) % 47) as f64 / 23.0) - 1.0) * 0.18,
                        )
                    };
                    QmfSlot {
                        real: vec![real],
                        imaginary: vec![imaginary],
                    }
                })
                .collect::<Vec<_>>();
            let real = slots
                .iter()
                .map(|slot| (slot.real[0] * 2_147_483_648.0) as i32)
                .collect::<Vec<_>>();
            let imaginary = slots
                .iter()
                .map(|slot| (slot.imaginary[0] * 2_147_483_648.0) as i32)
                .collect::<Vec<_>>();
            let mut c_quota = 0;
            let mut c_energy = 0;
            assert_eq!(
                unsafe {
                    fdk_aac_sys::fdk_sbr_tonality_quota_test(
                        real.as_ptr(),
                        imaginary.as_ptr(),
                        slots.len() as i32,
                        &mut c_quota,
                        &mut c_energy,
                    )
                },
                0
            );
            let rust_quota = fixed_lpc_quota(&slots, 0);
            assert_eq!(rust_quota, c_quota, "tonal {tonal}");
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn fixed_multiply_high_word_matches_c_for_all_sign_combinations() {
        let mut values = vec![
            i32::MIN,
            i32::MIN + 1,
            -1_500_000_000,
            -1,
            0,
            1,
            1_500_000_000,
            i32::MAX - 1,
            i32::MAX,
        ];
        let mut seed = 0x1234_5678_u32;
        for _ in 0..128 {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            values.push(seed as i32);
        }
        let mut left = Vec::new();
        let mut right = Vec::new();
        for &a in &values {
            for &b in &values[..9] {
                left.push(a);
                right.push(b);
            }
        }
        let mut c = vec![0; left.len()];
        assert_eq!(
            unsafe {
                fdk_aac_sys::fdk_fixed_mul_div2_test(
                    left.as_ptr(),
                    right.as_ptr(),
                    left.len() as i32,
                    c.as_mut_ptr(),
                )
            },
            0
        );
        for ((&left, &right), &expected) in left.iter().zip(&right).zip(&c) {
            assert_eq!(fixed_mul_div2(left, right), expected, "{left} * {right}");
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn complex_qmf_energy_scaling_matches_c_fixed_point_path() {
        let _c_guard = TONALITY_QUOTA_C_LOCK.lock().unwrap();
        const SLOTS: usize = 16;
        const BANDS: usize = 8;
        let real = (0..SLOTS * BANDS)
            .map(|index| ((index * 1_103_515_245 + 12_345) as u32 as i32) >> 5)
            .collect::<Vec<_>>();
        let imaginary = (0..SLOTS * BANDS)
            .map(|index| ((index * 214_013 + 2_531_011) as u32 as i32) >> 4)
            .collect::<Vec<_>>();
        let mut c = vec![0_i32; real.len()];
        let mut c_scale = 0;
        let c_qmf_scale = unsafe {
            fdk_aac_sys::fdk_sbr_complex_energy_test(
                real.as_ptr(),
                imaginary.as_ptr(),
                SLOTS as i32,
                BANDS as i32,
                -1,
                c.as_mut_ptr(),
                &mut c_scale,
            )
        };
        let (rust, rust_scale, rust_qmf_scale) = fixed_complex_energies(&real, &imaginary, -1);
        assert_eq!(rust, c);
        assert_eq!(rust_scale, c_scale);
        assert_eq!(rust_qmf_scale, c_qmf_scale);
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn raw_pcm_cldfb_energy_block_matches_c_scaling() {
        let _c_guard = TONALITY_QUOTA_C_LOCK.lock().unwrap();
        const SLOTS: usize = 16;
        const BANDS: usize = 64;
        let pcm = (0..SLOTS * BANDS)
            .map(|sample| ((sample as f64 * 0.071).sin() * 12_000.0) as i16)
            .collect::<Vec<_>>();
        let mut c_real = vec![0_i32; SLOTS * BANDS];
        let mut c_imaginary = vec![0_i32; SLOTS * BANDS];
        let mut lb_scale = 0;
        assert_eq!(
            unsafe {
                fdk_aac_sys::fdk_qmf_analysis64_cldfb_pcm_test(
                    pcm.as_ptr(),
                    pcm.len() as i32,
                    c_real.as_mut_ptr(),
                    c_imaginary.as_mut_ptr(),
                    &mut lb_scale,
                )
            },
            0
        );
        let mut c_energy = vec![0_i32; SLOTS * BANDS];
        let mut c_energy_scale = 0;
        let c_qmf_scale = unsafe {
            fdk_aac_sys::fdk_sbr_complex_energy_test(
                c_real.as_ptr(),
                c_imaginary.as_ptr(),
                SLOTS as i32,
                BANDS as i32,
                lb_scale + 7,
                c_energy.as_mut_ptr(),
                &mut c_energy_scale,
            )
        };
        let slots = LdSbrQmfAnalysis::new_cldfb(BANDS)
            .unwrap()
            .process_frame(
                &pcm.iter()
                    .map(|&value| f64::from(value))
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let (rust_rows, rust_scale, rust_qmf_scale) = fixed_cldfb64_energy_block(&slots);
        let rust_energy = rust_rows.into_iter().flatten().collect::<Vec<_>>();
        assert_eq!(rust_scale, c_energy_scale);
        assert_eq!(rust_qmf_scale, c_qmf_scale);
        assert_eq!(c_qmf_scale, 1);
        let maximum = rust_energy
            .iter()
            .zip(&c_energy)
            .map(|(&rust, &c)| rust.abs_diff(c))
            .max()
            .unwrap_or(0);
        assert_eq!(maximum, 0, "maximum fixed energy error {maximum}");
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn scale_factor_band_energy_accumulation_matches_c() {
        let _c_guard = TONALITY_QUOTA_C_LOCK.lock().unwrap();
        const SLOTS: usize = 16;
        const BANDS: usize = 12;
        let flat = (0..SLOTS * BANDS)
            .map(|index| ((index * 214_013 + 2_531_011) as i32 & 0x01ff_ffff) + 1)
            .collect::<Vec<_>>();
        let rows = flat
            .chunks_exact(BANDS)
            .map(<[i32]>::to_vec)
            .collect::<Vec<_>>();
        for scale in 0..=12 {
            for &(lower, upper) in &[(0, 1), (1, 3), (2, 6), (3, 11)] {
                let c = unsafe {
                    fdk_aac_sys::fdk_sbr_sfb_energy_test(
                        flat.as_ptr(),
                        SLOTS as i32,
                        BANDS as i32,
                        lower,
                        upper,
                        0,
                        SLOTS as i32,
                        scale,
                        scale,
                    )
                };
                assert_ne!(c, i32::MIN);
                assert_eq!(
                    fixed_sfb_energy(&rows, lower as usize, upper as usize, scale),
                    c,
                    "scale={scale} range={lower}..{upper}"
                );
            }
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn split_scale_factor_band_energy_accumulation_matches_c() {
        let _c_guard = TONALITY_QUOTA_C_LOCK.lock().unwrap();
        const SLOTS: usize = 16;
        const BANDS: usize = 12;
        let flat = (0..SLOTS * BANDS)
            .map(|index| ((index * 214_013 + 2_531_011) as i32 & 0x01ff_ffff) + 1)
            .collect::<Vec<_>>();
        let rows = flat
            .chunks_exact(BANDS)
            .map(<[i32]>::to_vec)
            .collect::<Vec<_>>();
        for scale0 in [0, 5, 9, 17] {
            for scale1 in [0, 5, 8, 16] {
                for &(lower, upper) in &[(0, 1), (1, 3), (2, 6), (3, 11)] {
                    let c = unsafe {
                        fdk_aac_sys::fdk_sbr_sfb_energy_split_test(
                            flat.as_ptr(),
                            SLOTS as i32,
                            BANDS as i32,
                            lower,
                            upper,
                            0,
                            SLOTS as i32,
                            8,
                            scale0,
                            scale1,
                        )
                    };
                    assert_eq!(
                        fixed_sfb_energy_split(
                            &rows,
                            lower as usize,
                            upper as usize,
                            8,
                            scale0,
                            scale1,
                        ),
                        c,
                        "scales=({scale0},{scale1}) range={lower}..{upper}"
                    );
                }
            }
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn ld64_logarithm_matches_c_fixed_point_polynomial() {
        let _c_guard = TONALITY_QUOTA_C_LOCK.lock().unwrap();
        let mut coefficients = [0_i32; 10];
        let mut conversion = 0_i32;
        assert_eq!(
            unsafe {
                fdk_aac_sys::fdk_log2_coefficients_test(
                    coefficients.as_mut_ptr(),
                    coefficients.len() as i32,
                    &mut conversion,
                )
            },
            0
        );
        assert_eq!(
            coefficients,
            [-32_768, -16_384, -10_923, -8_192, -6_554, -5_461, -4_681, -4_096, -3_641, -3_277,]
        );
        assert_eq!(conversion, 1_901_360_723);
        let mut values = vec![1, 2, 3, 7, 31, 1 << 20, 1 << 29, i32::MAX];
        let mut seed = 0x9e37_79b9_u32;
        for _ in 0..256 {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            values.push((seed >> 1).max(1) as i32);
        }
        let mut c = vec![0_i32; values.len()];
        assert_eq!(
            unsafe {
                fdk_aac_sys::fdk_sbr_log2_ld64_test(
                    values.as_ptr(),
                    values.len() as i32,
                    c.as_mut_ptr(),
                )
            },
            0
        );
        for (&value, &expected) in values.iter().zip(&c) {
            assert_eq!(fixed_log2_ld64(value), expected, "value={value}");
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn fixed_sbr_envelope_quantization_matches_c() {
        let _c_guard = TONALITY_QUOTA_C_LOCK.lock().unwrap();
        let energies = [0, 1, 17, 0x000f_ffff, 0x0fff_ffff, 0x3fff_ffff, i32::MAX];
        for energy in energies {
            for count in [1, 2, 7, 16, 32, 64, 224] {
                for common_scale in -24..=24 {
                    for amp_res_3db in [false, true] {
                        let c = unsafe {
                            fdk_aac_sys::fdk_sbr_quantize_energy_test(
                                energy,
                                count,
                                common_scale,
                                i32::from(amp_res_3db),
                            )
                        };
                        assert_eq!(
                            fixed_quantize_sbr_energy(
                                energy,
                                count as usize,
                                common_scale,
                                amp_res_3db,
                            ),
                            c,
                            "energy={energy} count={count} scale={common_scale} amp={amp_res_3db}"
                        );
                    }
                }
            }
        }
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn inverse_filter_regions_match_c_across_stateful_boundaries() {
        let mut original = Vec::new();
        let mut sbr = Vec::new();
        let mut energy = Vec::new();
        let mut transient = Vec::new();
        for index in 0..160 {
            original.push([-0.5, 0.0, 0.5, 2.5, 3.5, 6.5, 7.5, 9.5, 10.5][index % 9]);
            sbr.push([0.5, 1.5, 9.5, 10.5, 13.5, 14.5, 18.5, 19.5][(index * 5) % 8]);
            energy.push([20.0, 24.5, 25.5, 29.5, 30.5, 34.5, 35.5, 39.5, 40.5][(index * 7) % 9]);
            transient.push(u8::from(index % 7 == 0));
        }
        let q16 = |values: &[f64]| {
            values
                .iter()
                .map(|value| (value * 65536.0) as i32)
                .collect::<Vec<_>>()
        };
        let mut c_modes = vec![0; original.len()];
        assert_eq!(
            unsafe {
                fdk_aac_sys::fdk_sbr_invf_regions_test(
                    q16(&original).as_ptr(),
                    q16(&sbr).as_ptr(),
                    q16(&energy).as_ptr(),
                    transient.as_ptr(),
                    original.len() as i32,
                    c_modes.as_mut_ptr(),
                )
            },
            0
        );
        let mut rust_state = InverseFilterBandState::default();
        let rust_modes = original
            .iter()
            .zip(&sbr)
            .zip(&energy)
            .zip(&transient)
            .map(|(((&original, &sbr), &energy), &transient)| {
                inverse_filter_decision(original, sbr, energy, transient != 0, &mut rust_state)
            })
            .collect::<Vec<_>>();
        assert_eq!(rust_modes, c_modes);
    }

    #[cfg(feature = "ffi")]
    #[test]
    fn full_inverse_filter_detector_matches_c_after_linear_quota_smoothing() {
        for odds in [0.0_f64, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0] {
            let raw = (odds * 1.0e-6 * 2_147_483_648.0).round() as i32;
            let quotas = [raw, raw];
            let energies = [i32::MAX, i32::MAX];
            let patch = [0_i8];
            let bands = [0_i32, 1];
            let transient = [0_u8];
            let mut mode = [0_u8];
            let mut orig_region = [0_i32];
            let mut sbr_region = [0_i32];
            assert_eq!(
                unsafe {
                    fdk_aac_sys::fdk_sbr_invf_detector_test(
                        quotas.as_ptr(),
                        energies.as_ptr(),
                        patch.as_ptr(),
                        bands.as_ptr(),
                        transient.as_ptr(),
                        1,
                        2,
                        1,
                        1,
                        mode.as_mut_ptr(),
                        orig_region.as_mut_ptr(),
                        sbr_region.as_mut_ptr(),
                    )
                },
                0
            );
            let quantized_odds = f64::from(raw) / 2_147_483_648.0 / 1.0e-6;
            let coordinate = 3.0 * (0.5 * quantized_odds).max(1.0e-30).log2();
            let expected_orig = inverse_filter_region(coordinate, &[0.0, 3.0, 7.0, 10.0], 0);
            let expected_sbr = inverse_filter_region(coordinate, &[1.0, 10.0, 14.0, 19.0], 0);
            let mut state = InverseFilterBandState::default();
            let expected_mode =
                inverse_filter_decision(coordinate, coordinate, 45.0, false, &mut state);
            assert_eq!(orig_region[0] as usize, expected_orig, "odds {odds}");
            assert_eq!(sbr_region[0] as usize, expected_sbr, "odds {odds}");
            assert_eq!(mode[0], expected_mode, "odds {odds}");
        }
    }

    #[test]
    fn converts_and_formats_every_encoder_error() {
        let errors = [
            SbrEncoderError::from(QmfError::InvalidSampleCount(1)),
            SbrEncoderError::from(LdSbrError::UnsupportedTimeSlots(7)),
            SbrEncoderError::from(crate::asc::AscError::InvalidAudioObjectType(0)),
            SbrEncoderError::NonFiniteInput,
            SbrEncoderError::EmptyFrame,
            SbrEncoderError::EnvelopeLayoutMismatch,
            SbrEncoderError::UnrepresentableHuffmanSymbol(127),
            SbrEncoderError::PayloadTooLarge(270),
        ];
        for error in errors {
            assert!(!error.to_string().is_empty());
        }
    }
}
