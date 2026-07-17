# Dynamic range control and loudness normalization

Dynamic range control (DRC) metadata describes gains a decoder may apply. It
does not make an encoder's target level an unconditional PCM peak or RMS
target. The result depends on whether the stream contains usable DRC and
loudness metadata, which instruction is selected, the decoder target, and the
requested boost and attenuation scales.

## Decoder controls

`PureRustTransportDecoder::set_parameter` accepts the FDK-compatible controls:

| Parameter | Meaning |
| --- | --- |
| `DrcReferenceLevel` | Desired output loudness, in quarter-dB units below 0 dB. For example, `92` selects -23 dB. A negative value disables explicit normalization. |
| `DrcBoostFactor` | Scale positive DRC gains from `0` (none) to `127` (full metadata gain). |
| `DrcAttenuationFactor` | Scale negative DRC gains from `0` to `127`. |
| `DrcHeavyCompression` | Prefer the legacy heavy-compression path when that metadata exists. |
| `DrcDefaultPresentationMode` | Select the fallback presentation mode for legacy MPEG/DVB metadata. |
| `DrcEncoderTargetLevel` | Encoder reference used by the legacy normalization path; this is not itself a requested output level. |
| `UniDrcSetEffect` / `UniDrcAlbumMode` | Select MPEG-D DRC effect and track/album loudness behavior. |

The equivalent direct decoder methods are `set_drc_reference_level`,
`set_drc_boost_factor`, `set_drc_attenuation_factor`, and the other
`set_drc_*` methods on `AacLcDecoder`.

Setting both gain factors to `127` only asks for 100% of the gain carried by
the selected metadata. It cannot make the output match a target when the
stream has no matching instruction or loudness measurement. Likewise,
`DrcEncoderTargetLevel` supplies a reference for legacy metadata; use
`DrcReferenceLevel` to request decoder-side loudness normalization.

## Recommended sequence

1. Construct the decoder and select the DRC effect or legacy presentation
   mode required by the application.
2. Set `DrcReferenceLevel` to the desired loudness code and set boost and
   attenuation factors.
3. Decode normally. In-band DRC and loudness updates take effect when parsed.
4. Inspect `stream_info()` after decoding. `drc_program_reference_level`,
   `drc_presentation_mode`, `output_loudness`, and the DRC-present flag report
   what metadata and target the decoder actually selected.

For a controlled validation, compare DRC enabled and disabled output from the
same access units and metadata. Do not compare only PCM peaks: loudness
normalization and dynamic-range compression are different operations, and
both may be time-varying.

