#![no_main]

use fdk_aac_rust::encoder::{EncoderParameter, PureRustEncoderParameters};
use libfuzzer_sys::fuzz_target;

const PARAMETERS: [EncoderParameter; 21] = [
    EncoderParameter::AudioObjectType,
    EncoderParameter::Bitrate,
    EncoderParameter::BitrateMode,
    EncoderParameter::SampleRate,
    EncoderParameter::SbrMode,
    EncoderParameter::GranuleLength,
    EncoderParameter::ChannelMode,
    EncoderParameter::ChannelOrder,
    EncoderParameter::SbrRatio,
    EncoderParameter::Afterburner,
    EncoderParameter::Bandwidth,
    EncoderParameter::PeakBitrate,
    EncoderParameter::TransportMux,
    EncoderParameter::HeaderPeriod,
    EncoderParameter::SignalingMode,
    EncoderParameter::TransportSubframes,
    EncoderParameter::AudioMuxVersion,
    EncoderParameter::Protection,
    EncoderParameter::AncillaryBitrate,
    EncoderParameter::MetadataMode,
    EncoderParameter::ControlState,
];

fuzz_target!(|data: &[u8]| {
    let mut config =
        PureRustEncoderParameters::new(data.first().map_or(2, |v| usize::from(v % 8) + 1));
    for (index, bytes) in data
        .get(1..)
        .unwrap_or_default()
        .chunks_exact(4)
        .take(64)
        .enumerate()
    {
        let value = u32::from_le_bytes(bytes.try_into().unwrap());
        let parameter = PARAMETERS[index % PARAMETERS.len()];
        let _ = config.set_parameter(parameter, value);
    }
    let _ = config.resolve();
});
