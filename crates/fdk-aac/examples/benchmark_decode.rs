//! Reproducible AAC-LC decoder throughput smoke benchmark.

use std::{hint::black_box, time::Instant};

use fdk_aac_rust::{aac_encoder::PureRustAacLcMonoEncoder, decoder::AacLcDecoder};

fn main() {
    let frames = std::env::args()
        .nth(1)
        .map(|value| value.parse::<usize>().expect("invalid frame count"))
        .unwrap_or(10_000);
    assert!(frames > 0, "frame count must be nonzero");

    let input = (0..1024)
        .map(|index| {
            (2.0 * std::f32::consts::PI * 997.0 * index as f32 / 44_100.0).sin() * 12_000.0
        })
        .collect::<Vec<_>>();
    let mut encoder = PureRustAacLcMonoEncoder::new(4, 4000, 2000).unwrap();
    let access_unit = encoder.encode_raw_data_block(&input).unwrap();
    let mut decoder = AacLcDecoder::new(4, 1).unwrap();

    for _ in 0..100 {
        black_box(
            decoder
                .decode_raw_data_block_interleaved_f32(black_box(&access_unit))
                .unwrap(),
        );
    }

    let started = Instant::now();
    for _ in 0..frames {
        black_box(
            decoder
                .decode_raw_data_block_interleaved_f32(black_box(&access_unit))
                .unwrap(),
        );
    }
    let elapsed = started.elapsed();
    let audio_seconds = frames as f64 * 1024.0 / 44_100.0;
    println!(
        "frames={frames} elapsed_seconds={:.6} realtime_factor={:.2}",
        elapsed.as_secs_f64(),
        audio_seconds / elapsed.as_secs_f64()
    );
}
