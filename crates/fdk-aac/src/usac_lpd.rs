//! Integrated USAC linear-prediction-domain channel payload parsing.

use crate::asc::{UsacConfig, UsacElementConfig};
use crate::bits::{BitError, BitReader};
use crate::filterbank::imdct_planned_f32;
use crate::usac::{LpdChannelSideInfo, LpdDivisionMode};
use crate::usac_acelp::AcelpDecoder;
use crate::usac_acelp::{AcelpError, AcelpFrame};
use crate::usac_arith::UsacArithmeticDecoder;
use crate::usac_fac::{FacData, FacError};
use crate::usac_lpc::{LpcFrame, UsacLpcError};
use crate::usac_tcx::{adaptive_low_frequency_deemphasis, apply_fdns, TcxError, TcxFrame};

#[derive(Debug, Clone, PartialEq)]
pub enum LpdUnit {
    Acelp {
        division: usize,
        frame: AcelpFrame,
    },
    Tcx {
        division: usize,
        mode: u8,
        frame: TcxFrame,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct LpdFacTransition {
    pub division: usize,
    pub data: FacData,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LpdFramePayload {
    pub units: Vec<LpdUnit>,
    pub fac: Vec<LpdFacTransition>,
    pub lpc: LpcFrame,
    pub bits_read: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LpdError {
    Acelp(AcelpError),
    Tcx(TcxError),
    Fac(FacError),
    Lpc(UsacLpcError),
    InvalidCoreLength(usize),
    MissingLpcSlot(usize),
    Bit(BitError),
    UnsupportedConfiguration,
    FrequencyDomainFrame,
}

#[derive(Debug, Clone, Default)]
pub struct LpdRenderer {
    acelp: AcelpDecoder,
    tcx_overlap: Vec<f32>,
    output_history: Vec<f32>,
}

impl LpdRenderer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn render(&mut self, payload: &LpdFramePayload) -> Result<Vec<f32>, LpdError> {
        let mut output = Vec::new();
        for unit in &payload.units {
            let mut shaped_fac = None;
            let (division, mut rendered) = match unit {
                LpdUnit::Acelp { division, frame } => {
                    let old =
                        payload.lpc.lsp[*division].ok_or(LpdError::MissingLpcSlot(*division))?;
                    let new_slot = (*division + 1).min(4);
                    let new =
                        payload.lpc.lsp[new_slot].ok_or(LpdError::MissingLpcSlot(new_slot))?;
                    (*division, self.acelp.decode_frame(frame, &old, &new)?)
                }
                LpdUnit::Tcx {
                    division,
                    mode,
                    frame,
                } => {
                    let mut spectrum = frame.spectrum.clone();
                    let end_slot = (*division + (1usize << (*mode - 1))).min(4);
                    if let (Some(old), Some(new)) = (
                        payload.lpc.coefficients[*division],
                        payload.lpc.coefficients[end_slot],
                    ) {
                        let alfd = adaptive_low_frequency_deemphasis(&mut spectrum);
                        apply_fdns(&mut spectrum, &old, &new);
                        if let Some(transition) =
                            payload.fac.iter().find(|fac| fac.division == *division)
                        {
                            let mut fac = transition.data.clone();
                            fac.apply_tcx_gains(1.0, &alfd, *mode)?;
                            shaped_fac = Some(fac);
                        }
                    }
                    let mut time = imdct_planned_f32(&spectrum);
                    let length = frame.spectrum.len();
                    for (i, sample) in time.iter_mut().enumerate() {
                        *sample *=
                            (std::f32::consts::PI * (i as f32 + 0.5) / (2 * length) as f32).sin();
                    }
                    if self.tcx_overlap.len() != length {
                        self.tcx_overlap.resize(length, 0.0);
                    }
                    let mut block = vec![0.0; length];
                    for i in 0..length {
                        block[i] = time[i] + self.tcx_overlap[i];
                    }
                    self.tcx_overlap.copy_from_slice(&time[length..]);
                    (*division, block)
                }
            };
            if let Some(transition) = payload.fac.iter().find(|fac| fac.division == division) {
                let fac_signal = shaped_fac
                    .as_ref()
                    .unwrap_or(&transition.data)
                    .synthesize_transition(
                        &payload.lpc.coefficients[division].unwrap_or([0.0; 16]),
                        &self.output_history,
                    );
                for (sample, fac) in rendered.iter_mut().zip(fac_signal) {
                    *sample += fac;
                }
            }
            self.output_history.extend_from_slice(&rendered);
            let drain = self.output_history.len().saturating_sub(1024);
            self.output_history.drain(..drain);
            output.extend(rendered);
        }
        Ok(output)
    }
}

impl From<AcelpError> for LpdError {
    fn from(value: AcelpError) -> Self {
        Self::Acelp(value)
    }
}
impl From<TcxError> for LpdError {
    fn from(value: TcxError) -> Self {
        Self::Tcx(value)
    }
}
impl From<FacError> for LpdError {
    fn from(value: FacError) -> Self {
        Self::Fac(value)
    }
}
impl From<UsacLpcError> for LpdError {
    fn from(value: UsacLpcError) -> Self {
        Self::Lpc(value)
    }
}
impl From<BitError> for LpdError {
    fn from(value: BitError) -> Self {
        Self::Bit(value)
    }
}

#[derive(Debug, Clone)]
pub struct UsacLpdAccessUnitDecoder {
    config: UsacConfig,
    arithmetic: UsacArithmeticDecoder,
    renderer: LpdRenderer,
    previous_mode: Option<LpdDivisionMode>,
    previous_lpc4: Option<[f32; 16]>,
    noise_seed: u32,
}

impl UsacLpdAccessUnitDecoder {
    pub fn new(config: UsacConfig) -> Result<Self, LpdError> {
        let supported = config.channel_configuration_index == 1
            && config.elements.len() == 1
            && matches!(
                config.elements[0],
                UsacElementConfig::SingleChannel { sbr: None, .. }
            )
            && matches!(config.core_frame_length, 768 | 1024);
        if !supported {
            return Err(LpdError::UnsupportedConfiguration);
        }
        Ok(Self {
            config,
            arithmetic: UsacArithmeticDecoder::new(),
            renderer: LpdRenderer::new(),
            previous_mode: None,
            previous_lpc4: None,
            noise_seed: 0x1234_5678,
        })
    }

    pub fn decode_access_unit(&mut self, bytes: &[u8]) -> Result<Vec<f32>, LpdError> {
        let mut reader = BitReader::new(bytes);
        let independent = reader.read_bool()?;
        if !reader.read_bool()? {
            return Err(LpdError::FrequencyDomainFrame);
        }
        self.decode_from_reader(&mut reader, independent)
    }

    pub fn decode_from_reader(
        &mut self,
        reader: &mut BitReader<'_>,
        independent: bool,
    ) -> Result<Vec<f32>, LpdError> {
        let side = LpdChannelSideInfo::parse(reader).map_err(|error| match error {
            crate::usac::UsacError::Bit(bit) => LpdError::Bit(bit),
            _ => LpdError::UnsupportedConfiguration,
        })?;
        let pitch_offset =
            ((i64::from(self.config.sampling_frequency) * 34 + 6400) / 12800 - 34) as i32;
        let payload = LpdFramePayload::parse(
            reader,
            &side,
            usize::from(self.config.core_frame_length),
            pitch_offset,
            independent,
            self.previous_mode,
            self.previous_lpc4,
            false,
            true,
            &mut self.arithmetic,
            &mut self.noise_seed,
        )?;
        let output = self.renderer.render(&payload)?;
        self.previous_mode = Some(side.divisions[3]);
        self.previous_lpc4 = payload.lpc.lsf[4];
        Ok(output)
    }
}

impl LpdFramePayload {
    #[allow(clippy::too_many_arguments)]
    pub fn parse(
        reader: &mut BitReader<'_>,
        side: &LpdChannelSideInfo,
        core_frame_length: usize,
        pitch_offset: i32,
        independent: bool,
        previous_mode: Option<LpdDivisionMode>,
        previous_lpc4: Option<[f32; 16]>,
        last_lpc_lost: bool,
        last_frame_ok: bool,
        arithmetic: &mut UsacArithmeticDecoder,
        noise_seed: &mut u32,
    ) -> Result<Self, LpdError> {
        if !matches!(core_frame_length, 768 | 1024) {
            return Err(LpdError::InvalidCoreLength(core_frame_length));
        }
        let start = reader.bits_read();
        let granule = core_frame_length / 8;
        let mut units = Vec::new();
        let mut fac = Vec::new();
        let mut division = 0;
        let mut last_mode = previous_mode;
        let mut first_tcx = true;
        while division < 4 {
            let mode = side.divisions[division];
            let mode_index = match mode {
                LpdDivisionMode::Acelp20 => 0,
                LpdDivisionMode::Tcx20 => 1,
                LpdDivisionMode::Tcx40 => 2,
                LpdDivisionMode::Tcx80 => 3,
            };
            let transition = (division == 0 && previous_mode.is_some() && side.fac_data_present)
                || (matches!(last_mode, Some(LpdDivisionMode::Acelp20))
                    && !matches!(mode, LpdDivisionMode::Acelp20))
                || (matches!(
                    last_mode,
                    Some(LpdDivisionMode::Tcx20 | LpdDivisionMode::Tcx40 | LpdDivisionMode::Tcx80)
                ) && matches!(mode, LpdDivisionMode::Acelp20));
            if transition {
                fac.push(LpdFacTransition {
                    division,
                    data: FacData::parse(reader, granule, false)?,
                });
            }
            if mode_index == 0 {
                units.push(LpdUnit::Acelp {
                    division,
                    frame: AcelpFrame::parse(
                        reader,
                        side.acelp_core_mode,
                        core_frame_length,
                        pitch_offset,
                    )?,
                });
                division += 1;
            } else {
                let length = granule * (1usize << mode_index);
                units.push(LpdUnit::Tcx {
                    division,
                    mode: mode_index,
                    frame: TcxFrame::parse(
                        reader,
                        arithmetic,
                        length,
                        first_tcx,
                        independent,
                        noise_seed,
                    )?,
                });
                first_tcx = false;
                division += 1usize << (mode_index - 1);
            }
            last_mode = Some(mode);
        }
        let modes = side.divisions.map(|mode| match mode {
            LpdDivisionMode::Acelp20 => 0,
            LpdDivisionMode::Tcx20 => 1,
            LpdDivisionMode::Tcx40 => 2,
            LpdDivisionMode::Tcx80 => 3,
        });
        let lpc = LpcFrame::parse(
            reader,
            modes,
            previous_mode.is_none(),
            previous_lpc4,
            last_lpc_lost,
            last_frame_ok,
        )?;
        Ok(Self {
            units,
            fac,
            lpc,
            bits_read: reader.bits_read() - start,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitWriter;
    use crate::usac_acelp::AcelpSubframe;

    fn lpc_with(value: Option<[f32; 16]>) -> LpcFrame {
        LpcFrame {
            lsf: [None; 5],
            lsp: [value; 5],
            stability: [None; 5],
            coefficients: [value; 5],
            adaptive_mean: [0.0; 16],
            bits_read: 0,
        }
    }

    fn mono_config(core_frame_length: u16) -> UsacConfig {
        UsacConfig {
            sampling_frequency_index: 3,
            sampling_frequency: 48_000,
            core_sbr_frame_length_index: 1,
            core_frame_length,
            output_frame_length: core_frame_length,
            sbr_ratio_index: 0,
            channel_configuration_index: 1,
            elements: vec![UsacElementConfig::SingleChannel {
                noise_filling: false,
                sbr: None,
            }],
            extensions: Vec::new(),
        }
    }

    fn write_q2_pair(bits: &mut BitWriter) {
        bits.write(0, 2);
        bits.write(0, 2);
        bits.write(0, 8);
        bits.write(0, 8);
    }

    fn write_complete_acelp_payload(bits: &mut BitWriter) {
        for _ in 0..4 {
            bits.write(0, 2);
            for pitch_bits in [9usize, 6, 9, 6] {
                bits.write(0, pitch_bits);
                bits.write_bool(true);
                bits.write(0, 20);
                bits.write(0, 7);
            }
        }
        // LPC4 absolute vector.
        bits.write(0, 8);
        write_q2_pair(bits);
        // LPC0 and LPC2: absolute first stage + nk_mode 0 refinement.
        for _ in 0..2 {
            bits.write_bool(false);
            bits.write(0, 8);
            write_q2_pair(bits);
        }
        // LPC1 mode 0, nk_mode 2 refinement.
        bits.write_bool(false);
        write_q2_pair(bits);
        // LPC3 mode 0, nk_mode 1 Q0/Q0 refinement.
        bits.write_bool(false);
        bits.write_bool(false);
        bits.write_bool(false);
    }

    #[test]
    fn renders_tcx80_unit_to_core_frame_length() {
        let payload = LpdFramePayload {
            units: vec![LpdUnit::Tcx {
                division: 0,
                mode: 3,
                frame: TcxFrame {
                    noise_factor: 0,
                    global_gain: 0,
                    arithmetic_reset: true,
                    quantized_spectrum: vec![0; 1024],
                    spectrum: vec![0.0; 1024],
                    bits_read: 0,
                },
            }],
            fac: Vec::new(),
            lpc: LpcFrame {
                lsf: [None; 5],
                lsp: [None; 5],
                stability: [None; 5],
                coefficients: [None; 5],
                adaptive_mean: [0.0; 16],
                bits_read: 0,
            },
            bits_read: 0,
        };
        let output = LpdRenderer::new().render(&payload).unwrap();
        assert_eq!(output, vec![0.0; 1024]);
    }

    #[test]
    fn renderer_reports_missing_acelp_lpc_slots_before_decoding() {
        let payload = LpdFramePayload {
            units: vec![LpdUnit::Acelp {
                division: 0,
                frame: AcelpFrame {
                    core_mode: 0,
                    mean_energy_index: 0,
                    subframes: Vec::new(),
                    bits_read: 0,
                },
            }],
            fac: Vec::new(),
            lpc: lpc_with(None),
            bits_read: 0,
        };
        assert_eq!(
            LpdRenderer::new().render(&payload),
            Err(LpdError::MissingLpcSlot(0))
        );
    }

    #[test]
    fn renderer_handles_tcx20_fac_and_overlap_state() {
        let frame = TcxFrame {
            noise_factor: 0,
            global_gain: 0,
            arithmetic_reset: true,
            quantized_spectrum: vec![0; 256],
            spectrum: vec![0.0; 256],
            bits_read: 0,
        };
        let payload = LpdFramePayload {
            units: vec![
                LpdUnit::Tcx {
                    division: 0,
                    mode: 1,
                    frame: frame.clone(),
                },
                LpdUnit::Tcx {
                    division: 1,
                    mode: 1,
                    frame,
                },
            ],
            fac: vec![LpdFacTransition {
                division: 0,
                data: FacData {
                    gain_code: None,
                    coefficients: vec![0.0; 8],
                    bits_read: 0,
                },
            }],
            lpc: lpc_with(Some([0.0; 16])),
            bits_read: 0,
        };
        let mut renderer = LpdRenderer::new();
        assert_eq!(renderer.render(&payload).unwrap(), vec![0.0; 512]);
        assert_eq!(renderer.render(&payload).unwrap(), vec![0.0; 512]);
    }

    #[test]
    fn access_unit_decoder_validates_configuration_and_frame_domain() {
        assert!(UsacLpdAccessUnitDecoder::new(mono_config(1024)).is_ok());

        let mut invalid_length = mono_config(512);
        assert!(matches!(
            UsacLpdAccessUnitDecoder::new(invalid_length.clone()),
            Err(LpdError::UnsupportedConfiguration)
        ));
        invalid_length.core_frame_length = 768;
        invalid_length.channel_configuration_index = 2;
        assert!(matches!(
            UsacLpdAccessUnitDecoder::new(invalid_length),
            Err(LpdError::UnsupportedConfiguration)
        ));

        let mut invalid_element = mono_config(1024);
        invalid_element.elements = vec![UsacElementConfig::ChannelPair {
            noise_filling: false,
            sbr: None,
            stereo_config_index: 0,
            mps212: None,
        }];
        assert_eq!(
            UsacLpdAccessUnitDecoder::new(invalid_element).unwrap_err(),
            LpdError::UnsupportedConfiguration
        );

        let mut decoder = UsacLpdAccessUnitDecoder::new(mono_config(1024)).unwrap();
        assert!(matches!(
            decoder.decode_access_unit(&[]),
            Err(LpdError::Bit(BitError::UnexpectedEof { .. }))
        ));
        assert_eq!(
            decoder.decode_access_unit(&[0x00]),
            Err(LpdError::FrequencyDomainFrame)
        );

        let mut valid_side = BitWriter::new();
        valid_side.write(0, 3); // ACELP core mode
        valid_side.write(0, 5); // four ACELP20 divisions
        valid_side.write_bool(false); // BPF disabled
        valid_side.write_bool(false); // previous frame was not LPD
        valid_side.write_bool(false); // no FAC
        assert!(matches!(
            decoder.decode_from_reader(&mut BitReader::new(&valid_side.finish()), true),
            Err(LpdError::Acelp(_))
        ));
    }

    #[test]
    fn payload_dispatches_complete_tcx_modes_before_lpc() {
        for mode in [
            LpdDivisionMode::Tcx20,
            LpdDivisionMode::Tcx40,
            LpdDivisionMode::Tcx80,
        ] {
            let side = LpdChannelSideInfo {
                acelp_core_mode: 0,
                lpd_mode: 0,
                divisions: [mode; 4],
                bpf_control_info: false,
                previous_frame_was_lpd: false,
                fac_data_present: false,
                bits_read: 0,
            };
            let mut state = 1u32;
            let mut bytes = vec![0; 4096];
            for byte in &mut bytes {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *byte = (state >> 24) as u8;
            }
            let result = LpdFramePayload::parse(
                &mut BitReader::new(&bytes),
                &side,
                1024,
                0,
                true,
                None,
                None,
                false,
                true,
                &mut UsacArithmeticDecoder::new(),
                &mut 1,
            );
            assert!(result.is_ok(), "deterministic {mode:?} payload must decode");
        }
    }

    #[test]
    fn payload_parser_rejects_invalid_core_length_without_consuming_bits() {
        let mut reader = BitReader::new(&[]);
        let side = LpdChannelSideInfo {
            acelp_core_mode: 0,
            lpd_mode: 0,
            divisions: [LpdDivisionMode::Acelp20; 4],
            bpf_control_info: false,
            previous_frame_was_lpd: false,
            fac_data_present: false,
            bits_read: 0,
        };
        assert_eq!(
            LpdFramePayload::parse(
                &mut reader,
                &side,
                512,
                0,
                true,
                None,
                None,
                false,
                true,
                &mut UsacArithmeticDecoder::new(),
                &mut 1,
            ),
            Err(LpdError::InvalidCoreLength(512))
        );
        assert_eq!(reader.bits_read(), 0);
    }

    #[test]
    fn renderer_decodes_acelp_and_checks_both_lpc_slots() {
        let subframe = AcelpSubframe {
            pitch_lag: 64,
            pitch_fraction_quarters: 0,
            ltp_filtering: true,
            innovative_indices: vec![0; 4],
            gain_index: 20,
        };
        let frame = AcelpFrame {
            core_mode: 0,
            mean_energy_index: 0,
            subframes: vec![subframe; 4],
            bits_read: 0,
        };
        let lsp = crate::usac_lpc::lsf_to_lsp(&std::array::from_fn(|i| 300.0 + i as f32 * 350.0));
        let payload = LpdFramePayload {
            units: vec![LpdUnit::Acelp {
                division: 0,
                frame: frame.clone(),
            }],
            fac: Vec::new(),
            lpc: lpc_with(Some(lsp)),
            bits_read: 0,
        };
        let output = LpdRenderer::new().render(&payload).unwrap();
        assert_eq!(output.len(), 256);
        assert!(output.iter().all(|value| value.is_finite()));

        let mut missing_new = lpc_with(Some(lsp));
        missing_new.lsp[1] = None;
        let payload = LpdFramePayload {
            units: vec![LpdUnit::Acelp { division: 0, frame }],
            fac: Vec::new(),
            lpc: missing_new,
            bits_read: 0,
        };
        assert_eq!(
            LpdRenderer::new().render(&payload),
            Err(LpdError::MissingLpcSlot(1))
        );
    }

    #[test]
    fn payload_dispatches_all_unit_modes_and_fac_transition_errors() {
        let parse =
            |side: LpdChannelSideInfo, previous_mode: Option<LpdDivisionMode>, bytes: &[u8]| {
                LpdFramePayload::parse(
                    &mut BitReader::new(bytes),
                    &side,
                    1024,
                    0,
                    true,
                    previous_mode,
                    None,
                    false,
                    true,
                    &mut UsacArithmeticDecoder::new(),
                    &mut 1,
                )
            };
        for mode in [
            LpdDivisionMode::Acelp20,
            LpdDivisionMode::Tcx20,
            LpdDivisionMode::Tcx40,
            LpdDivisionMode::Tcx80,
        ] {
            let side = LpdChannelSideInfo {
                acelp_core_mode: 0,
                lpd_mode: 0,
                divisions: [mode; 4],
                bpf_control_info: false,
                previous_frame_was_lpd: false,
                fac_data_present: false,
                bits_read: 0,
            };
            let error = parse(side, None, &[]).unwrap_err();
            if mode == LpdDivisionMode::Acelp20 {
                assert!(matches!(error, LpdError::Acelp(_)));
            } else {
                assert!(matches!(error, LpdError::Tcx(_)));
            }
        }

        let side = LpdChannelSideInfo {
            acelp_core_mode: 0,
            lpd_mode: 0,
            divisions: [LpdDivisionMode::Tcx20; 4],
            bpf_control_info: false,
            previous_frame_was_lpd: true,
            fac_data_present: true,
            bits_read: 0,
        };
        assert!(matches!(
            parse(side, Some(LpdDivisionMode::Acelp20), &[]),
            Err(LpdError::Fac(_))
        ));
    }

    #[test]
    fn all_acelp_units_reach_lpc_parser() {
        let mut bits = BitWriter::new();
        for _ in 0..4 {
            bits.write(0, 2);
            for pitch_bits in [9usize, 6, 9, 6] {
                bits.write(0, pitch_bits);
                bits.write_bool(true);
                bits.write(0, 20);
                bits.write(0, 7);
            }
        }
        let bit_len = bits.bits_written();
        let bytes = bits.finish();
        let side = LpdChannelSideInfo {
            acelp_core_mode: 0,
            lpd_mode: 0,
            divisions: [LpdDivisionMode::Acelp20; 4],
            bpf_control_info: false,
            previous_frame_was_lpd: false,
            fac_data_present: false,
            bits_read: 0,
        };
        let mut reader = BitReader::with_bit_len(&bytes, bit_len).unwrap();
        assert!(matches!(
            LpdFramePayload::parse(
                &mut reader,
                &side,
                1024,
                0,
                true,
                None,
                None,
                false,
                true,
                &mut UsacArithmeticDecoder::new(),
                &mut 1,
            ),
            Err(LpdError::Lpc(_))
        ));
    }

    #[test]
    fn parses_and_renders_complete_all_acelp_access_unit() {
        let mut payload_bits = BitWriter::new();
        write_complete_acelp_payload(&mut payload_bits);
        let bit_len = payload_bits.bits_written();
        let bytes = payload_bits.finish();
        let side = LpdChannelSideInfo {
            acelp_core_mode: 0,
            lpd_mode: 0,
            divisions: [LpdDivisionMode::Acelp20; 4],
            bpf_control_info: false,
            previous_frame_was_lpd: false,
            fac_data_present: false,
            bits_read: 0,
        };
        let payload = LpdFramePayload::parse(
            &mut BitReader::with_bit_len(&bytes, bit_len).unwrap(),
            &side,
            1024,
            0,
            true,
            None,
            None,
            false,
            true,
            &mut UsacArithmeticDecoder::new(),
            &mut 1,
        )
        .unwrap();
        assert_eq!(payload.units.len(), 4);
        assert!(payload.fac.is_empty());
        assert_eq!(LpdRenderer::new().render(&payload).unwrap().len(), 1024);

        let mut access = BitWriter::new();
        access.write_bool(true); // independent
        access.write_bool(true); // LPD domain
        access.write(0, 3); // ACELP core mode
        access.write(0, 5); // all ACELP divisions
        access.write_bool(false); // bpf
        access.write_bool(false); // previous frame was not LPD
        access.write_bool(false); // no externally signalled FAC
        write_complete_acelp_payload(&mut access);
        let mut decoder = UsacLpdAccessUnitDecoder::new(mono_config(1024)).unwrap();
        let output = decoder.decode_access_unit(&access.finish()).unwrap();
        assert_eq!(output.len(), 1024);
        assert!(output.iter().all(|sample| sample.is_finite()));
        assert_eq!(decoder.previous_mode, Some(LpdDivisionMode::Acelp20));
        assert!(decoder.previous_lpc4.is_some());
    }

    #[test]
    fn tcx_to_acelp_transition_requires_fac_payload() {
        let side = LpdChannelSideInfo {
            acelp_core_mode: 0,
            lpd_mode: 0,
            divisions: [LpdDivisionMode::Acelp20; 4],
            bpf_control_info: false,
            previous_frame_was_lpd: true,
            fac_data_present: false,
            bits_read: 0,
        };
        assert!(matches!(
            LpdFramePayload::parse(
                &mut BitReader::new(&[]),
                &side,
                1024,
                0,
                true,
                Some(LpdDivisionMode::Tcx20),
                Some([0.0; 16]),
                false,
                true,
                &mut UsacArithmeticDecoder::new(),
                &mut 1,
            ),
            Err(LpdError::Fac(_))
        ));
    }

    #[test]
    fn access_decoder_maps_invalid_side_info_and_validates_more_configs() {
        let mut config = mono_config(1024);
        config.elements.clear();
        assert!(matches!(
            UsacLpdAccessUnitDecoder::new(config),
            Err(LpdError::UnsupportedConfiguration)
        ));

        let mut decoder = UsacLpdAccessUnitDecoder::new(mono_config(1024)).unwrap();
        let mut writer = BitWriter::new();
        writer.write(0, 3);
        writer.write(31, 5); // invalid lpd_mode
        let bytes = writer.finish();
        assert_eq!(
            decoder.decode_from_reader(&mut BitReader::new(&bytes), true),
            Err(LpdError::UnsupportedConfiguration)
        );
        assert!(matches!(
            decoder.decode_from_reader(&mut BitReader::new(&[]), true),
            Err(LpdError::Bit(_))
        ));
    }

    #[test]
    fn converts_all_nested_lpd_errors() {
        let bit = BitError::UnexpectedEof {
            needed_bits: 1,
            remaining_bits: 0,
        };
        assert!(matches!(
            LpdError::from(AcelpError::InvalidCoreMode(8)),
            LpdError::Acelp(_)
        ));
        assert!(matches!(
            LpdError::from(TcxError::InvalidLength(0)),
            LpdError::Tcx(_)
        ));
        assert!(matches!(
            LpdError::from(FacError::InvalidLength(7)),
            LpdError::Fac(_)
        ));
        assert!(matches!(
            LpdError::from(UsacLpcError::MissingPreviousLsf),
            LpdError::Lpc(_)
        ));
        assert_eq!(LpdError::from(bit.clone()), LpdError::Bit(bit));
    }
}
