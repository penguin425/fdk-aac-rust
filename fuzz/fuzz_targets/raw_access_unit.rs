#![no_main]

use fdk_aac_rust::{raw::RawDataBlock, transport::PureRustTransportDecoder};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let data = &data[..data.len().min(1 << 16)];
    let _ = RawDataBlock::parse(data);
    if let Ok(mut decoder) = PureRustTransportDecoder::from_asc_bytes(&[0x12, 0x10]) {
        let _ = decoder.decode_raw_interleaved_f32(data);
    }
});
