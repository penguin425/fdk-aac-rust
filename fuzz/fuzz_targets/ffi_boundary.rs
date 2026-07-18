#![no_main]

use fdk_aac_rust::{Decoder, Encoder, TransportType};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let data = &data[..data.len().min(1 << 16)];
    if let Ok(mut decoder) = Decoder::open(TransportType::Adts) {
        let mut input = data.to_vec();
        let _ = decoder.fill(&mut input);
        let mut output = vec![0_i16; 8192];
        let _ = decoder.decode_frame(&mut output);
    }
    if let Ok(mut encoder) = Encoder::open(data.first().map_or(2, |v| u32::from(v % 8) + 1)) {
        if data.len() >= 9 {
            let parameter = i32::from_le_bytes(data[1..5].try_into().unwrap());
            let value = u32::from_le_bytes(data[5..9].try_into().unwrap());
            let _ = encoder.set_param(parameter, value);
        }
    }
});
