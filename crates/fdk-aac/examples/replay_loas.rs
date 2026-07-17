//! Replay a LOAS stream through the incremental Pure Rust transport decoder.

use fdk_aac_rust::{loas::LoasFrame, transport::PureRustTransportDecoder};

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: replay_loas <input-file> [chunk-size]");
    let chunk_size = std::env::args()
        .nth(2)
        .map(|value| value.parse::<usize>().expect("invalid chunk size"))
        .unwrap_or(37);
    assert!(chunk_size > 0, "chunk size must be nonzero");

    let bytes = std::fs::read(path).expect("failed to read input file");
    let first = LoasFrame::parse(&bytes).expect("input does not start with a LOAS frame");
    let mut decoder = PureRustTransportDecoder::from_loas_frame(first.bytes)
        .expect("failed to configure decoder from first frame");
    let mut frames = 0usize;
    for chunk in bytes.chunks(chunk_size) {
        decoder
            .push_loas_bytes(chunk)
            .expect("failed to buffer LOAS bytes");
        frames += decoder
            .drain_loas_interleaved_f32()
            .expect("failed to decode LOAS frame")
            .len();
    }
    println!(
        "frames={frames} buffered={} discarded={}",
        decoder.buffered_loas_bytes().unwrap(),
        decoder.discarded_loas_bytes().unwrap()
    );
}
