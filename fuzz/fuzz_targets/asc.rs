#![no_main]

use fdk_aac_rust::{
    asc::{AudioSpecificConfig, ProgramConfig},
    transport::PureRustTransportDecoder,
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let data = &data[..data.len().min(4096)];
    if let Ok(config) = AudioSpecificConfig::parse(data) {
        let _ = config.to_bytes();
    }
    let _ = ProgramConfig::parse_from_bytes(data);
    let _ = PureRustTransportDecoder::from_asc_bytes(data);
});
