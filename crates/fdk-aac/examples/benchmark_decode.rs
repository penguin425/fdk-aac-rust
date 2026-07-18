//! Reproducible AAC-LC decoder throughput and resource smoke benchmark.

use std::{hint::black_box, time::Instant};

use fdk_aac_rust::{aac_encoder::PureRustAacLcMonoEncoder, decoder::AacLcDecoder};

#[derive(Debug)]
struct Measurement {
    elapsed_seconds: f64,
    realtime_factor: f64,
    first_frame_microseconds: f64,
}

fn main() {
    let mut json = false;
    let mut positional = Vec::new();
    for argument in std::env::args().skip(1) {
        if argument == "--json" {
            json = true;
        } else {
            positional.push(argument);
        }
    }
    let frames = positional
        .first()
        .map(|value| value.parse::<usize>().expect("invalid frame count"))
        .unwrap_or(10_000);
    let iterations = positional
        .get(1)
        .map(|value| value.parse::<usize>().expect("invalid iteration count"))
        .unwrap_or(5);
    assert!(frames > 0, "frame count must be nonzero");
    assert!(iterations > 0, "iteration count must be nonzero");

    let input = (0..1024)
        .map(|index| {
            (2.0 * std::f32::consts::PI * 997.0 * index as f32 / 44_100.0).sin() * 12_000.0
        })
        .collect::<Vec<_>>();
    let mut encoder = PureRustAacLcMonoEncoder::new(4, 4000, 2000).unwrap();
    let access_unit = encoder.encode_raw_data_block(&input).unwrap();
    let mut measurements = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let mut decoder = AacLcDecoder::new(4, 1).unwrap();
        let first_started = Instant::now();
        black_box(
            decoder
                .decode_raw_data_block_interleaved_f32(black_box(&access_unit))
                .unwrap(),
        );
        let first_frame_microseconds = first_started.elapsed().as_secs_f64() * 1_000_000.0;

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
        let elapsed_seconds = started.elapsed().as_secs_f64();
        let audio_seconds = frames as f64 * 1024.0 / 44_100.0;
        measurements.push(Measurement {
            elapsed_seconds,
            realtime_factor: audio_seconds / elapsed_seconds,
            first_frame_microseconds,
        });
    }

    measurements.sort_by(|left, right| {
        left.realtime_factor
            .partial_cmp(&right.realtime_factor)
            .unwrap()
    });
    let median_realtime_factor = measurements[measurements.len() / 2].realtime_factor;
    let mut first_frames = measurements
        .iter()
        .map(|measurement| measurement.first_frame_microseconds)
        .collect::<Vec<_>>();
    first_frames.sort_by(f64::total_cmp);
    let median_first_frame_microseconds = first_frames[first_frames.len() / 2];
    let peak_rss_kib = linux_peak_rss_kib();
    let binary_bytes = std::env::current_exe()
        .and_then(std::fs::metadata)
        .map(|metadata| metadata.len())
        .unwrap_or(0);

    if json {
        let peak_rss = peak_rss_kib.map_or_else(|| "null".to_owned(), |value| value.to_string());
        println!(
            concat!(
                "{{\"schema_version\":1,\"benchmark\":\"aac_lc_mono_decode\",",
                "\"frames_per_iteration\":{},\"iterations\":{},",
                "\"median_realtime_factor\":{:.4},",
                "\"median_first_frame_microseconds\":{:.2},",
                "\"peak_rss_kib\":{},\"binary_bytes\":{}}}"
            ),
            frames,
            iterations,
            median_realtime_factor,
            median_first_frame_microseconds,
            peak_rss,
            binary_bytes
        );
    } else {
        for (index, measurement) in measurements.iter().enumerate() {
            println!(
                "iteration={} frames={} elapsed_seconds={:.6} realtime_factor={:.2} first_frame_microseconds={:.2}",
                index + 1,
                frames,
                measurement.elapsed_seconds,
                measurement.realtime_factor,
                measurement.first_frame_microseconds
            );
        }
        println!(
            "median_realtime_factor={median_realtime_factor:.2} median_first_frame_microseconds={median_first_frame_microseconds:.2} peak_rss_kib={} binary_bytes={binary_bytes}",
            peak_rss_kib.map_or_else(|| "unavailable".to_owned(), |value| value.to_string())
        );
    }
}

fn linux_peak_rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
    line.split_whitespace().nth(1)?.parse().ok()
}
