use fdk_aac_rust::{aac_encoder::PureRustAacLcMonoEncoder, transport::PureRustTransportDecoder};

const FRAME_COUNT: usize = 12;
const SEEK_FRAME: usize = 5;

fn application_stream() -> Vec<Vec<u8>> {
    let mut encoder = PureRustAacLcMonoEncoder::new(4, 44_100, 64_000).unwrap();
    (0..FRAME_COUNT)
        .map(|frame| {
            let pcm = (0..1024)
                .map(|sample| {
                    let phase = (frame * 1024 + sample) as f32 * 0.03125;
                    phase.sin() * 0.2
                })
                .collect::<Vec<_>>();
            encoder.encode_adts_frame(&pcm).unwrap()
        })
        .collect()
}

fn decode_chunked(bytes: &[u8]) -> Vec<Vec<f32>> {
    let mut decoder = PureRustTransportDecoder::from_adts_frame(bytes).unwrap();
    let chunk_sizes = [1, 2, 7, 31, 3, 127, 11, 251];
    let mut output = Vec::new();
    let mut offset = 0;
    let mut chunk = 0;
    while offset < bytes.len() {
        let end = (offset + chunk_sizes[chunk % chunk_sizes.len()]).min(bytes.len());
        decoder.push_adts_bytes(&bytes[offset..end]).unwrap();
        output.extend(decoder.drain_adts_interleaved_f32().unwrap());
        offset = end;
        chunk += 1;
    }
    output.extend(decoder.drain_adts_interleaved_f32().unwrap());
    assert_eq!(decoder.buffered_adts_bytes().unwrap(), 0);
    output
}

#[test]
fn streaming_application_decodes_continuously_across_arbitrary_chunks() {
    let frames = application_stream();
    let stream = frames.concat();
    let output = decode_chunked(&stream);

    assert_eq!(output.len(), FRAME_COUNT);
    assert!(output.iter().all(|pcm| pcm.len() == 1024));
    assert!(output
        .iter()
        .flatten()
        .all(|sample| sample.is_finite() && sample.abs() <= 1.0));
}

#[test]
fn seeking_to_an_adts_frame_restarts_and_decodes_the_suffix() {
    let frames = application_stream();
    let seek_offset = frames[..SEEK_FRAME].iter().map(Vec::len).sum::<usize>();
    let stream = frames.concat();
    let suffix = &stream[seek_offset..];

    let after_seek = decode_chunked(suffix);
    let independently_reopened = decode_chunked(suffix);

    assert_eq!(after_seek.len(), FRAME_COUNT - SEEK_FRAME);
    assert_eq!(after_seek, independently_reopened);
}
