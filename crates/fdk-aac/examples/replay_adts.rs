//! Replay an ADTS byte stream through the optional reference FDK decoder.
//!
//! This small harness is primarily used by `tools/replay-issue-89-corpus.sh`.

#[cfg(feature = "ffi")]
use fdk_aac_rust::{Decoder, TransportType};

#[cfg(feature = "ffi")]
fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: replay_adts <input-file>");
    let mut input = std::fs::read(path).expect("failed to read input file");
    let mut decoder = Decoder::open(TransportType::Adts).expect("failed to open ADTS decoder");
    let _ = decoder.fill(&mut input);
    let mut output = vec![0i16; 8 * 2048];
    for _ in 0..64 {
        if decoder.decode_frame(&mut output).is_err() {
            break;
        }
    }
}

#[cfg(not(feature = "ffi"))]
fn main() {
    eprintln!("replay_adts requires the default `ffi` feature");
}
