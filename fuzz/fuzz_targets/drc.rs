#![no_main]

use fdk_aac_rust::{
    bits::BitReader,
    drc::{
        parse_dvb_ancillary_downmix, parse_dvb_ancillary_drc, parse_mpeg4_drc_payload,
        DownmixInstructions, LoudnessInfoSet, UniDrcConfig,
    },
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let data = &data[..data.len().min(1 << 16)];
    let _ = UniDrcConfig::parse_foundation(data);
    let _ = DownmixInstructions::parse_v0(data, data.first().copied().unwrap_or(0) & 31);
    let _ = DownmixInstructions::parse_v1(data, data.first().copied().unwrap_or(0) & 31);
    let _ = LoudnessInfoSet::parse_v0(data);
    let _ = parse_dvb_ancillary_drc(data);
    let _ = parse_dvb_ancillary_downmix(data);
    let _ = parse_mpeg4_drc_payload(&mut BitReader::new(data));
});
