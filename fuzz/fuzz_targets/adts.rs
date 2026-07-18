#![no_main]

use fdk_aac_rust::adts::{adts_frames, AdtsFrame, AdtsIncrementalStream};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let data = &data[..data.len().min(1 << 16)];
    let _ = AdtsFrame::parse(data).and_then(|frame| frame.raw_data_blocks().map(|_| frame));
    for frame in adts_frames(data).take(64) {
        if let Ok(frame) = frame {
            let _ = frame.validate_multi_block_header_crc();
        }
    }
    let split = data.len() / 2;
    let mut stream = AdtsIncrementalStream::new();
    stream.push(&data[..split]);
    stream.push(&data[split..]);
    while stream.next_frame().is_some() {}
});
