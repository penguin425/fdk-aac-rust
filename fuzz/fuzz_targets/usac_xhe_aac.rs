#![no_main]

use fdk_aac_rust::{asc::AudioSpecificConfig, transport::PureRustTransportDecoder};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let data = &data[..data.len().min(1 << 16)];
    if let Ok(config) = AudioSpecificConfig::parse(data) {
        if config.audio_object_type == 42 {
            let _ = config.to_bytes();
        }
    }
    let _ = PureRustTransportDecoder::from_asc_bytes(data);
    let _ = PureRustTransportDecoder::from_drm_xhe_static_config(data);
});
