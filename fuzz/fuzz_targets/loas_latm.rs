#![no_main]

use fdk_aac_rust::{
    latm::LatmAudioMuxElement,
    loas::{loas_frames, LoasFrame, LoasIncrementalStream},
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let data = &data[..data.len().min(1 << 16)];
    let _ = LatmAudioMuxElement::parse_aac_lc(data);
    if let Ok(frame) = LoasFrame::parse(data) {
        let _ = LatmAudioMuxElement::parse_aac_lc(frame.audio_mux_element);
    }
    for frame in loas_frames(data).take(64).flatten() {
        let _ = LatmAudioMuxElement::parse_aac_lc(frame.audio_mux_element);
    }
    let mut stream = LoasIncrementalStream::new();
    stream.push(data);
    while stream.next_frame().is_some() {}
});
