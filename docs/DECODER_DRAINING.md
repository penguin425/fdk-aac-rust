# Decoder delay, random access, and draining

AAC transform frames overlap in time. Decoding an arbitrary access unit with a
fresh decoder therefore cannot reproduce the same first samples as decoding it
after its predecessor: the previous frame supplies the first half of the MDCT
overlap. This explains the approximately half-frame mismatch reported in
[`mstorsjo/fdk-aac#148`](https://github.com/mstorsjo/fdk-aac/issues/148); it is
not output that can be reconstructed by repeatedly flushing a decoder that was
started at that access unit.

For random access or seeking:

1. reset the decoder with `signal_interruption()` (or the transport
   `INTERRUPTION` flag);
2. begin at a container-defined random-access point, or decode at least one
   valid preceding access unit as preroll;
3. discard preroll PCM and apply the container timestamp/edit-list decision;
4. do not infer raw AAC packet boundaries from the payload bytes.

`DecoderStreamInfo::output_delay` reports the configured codec and
post-processing startup delay. It is not a count of missing predecessor samples
after an arbitrary midstream start.

At ordinary end of stream, `flush_interleaved_f32()` or
`flush_interleaved_i16()` drains the stored synthesis overlap once. A second
flush returns silence because no delayed synthesis samples remain. Container
padding and the exact intended presentation length are separate metadata and
must be trimmed by the container layer.
