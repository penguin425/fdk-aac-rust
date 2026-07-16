# Pure Rust C/C++ parity record

The Pure Rust implementation covers the supported public decoder/encoder
profiles and transports below. The optional `ffi` feature retains the original
C/C++ wrappers and differential validation.

## Closed parity gaps

The checked items below record parity gaps that were identified during the
migration and subsequently closed. Together with the validation evidence at the
end of this document, they define the completed project scope. They do not claim
drop-in equivalence for private FDK internals, configurations outside the
declared public scope, every possible input, or future upstream revisions.

- [x] ER AAC-LD complete rate-control/psychoacoustic behavior
  - [x] 480/512-sample mono/stereo spectral quantization, ER access-unit
    writing, ASC generation, Pure Rust decode roundtrip, and C FDK acceptance
  - [x] Stateful long-window PCM MDCT frontend with C FDK decoded-waveform
    correlation coverage
  - [x] C-matched transient state machine, signaled 25%-overlap window,
    adaptive IMDCT state, output normalization, and direct C/Rust transient
    waveform differential coverage
  - [x] CBR low-delay reservoir sizing (500--4000 bit interpolation over
    12--70 kbit/s/channel), byte/capacity limits, and full initial state
  - [x] CBR `EXT_FILL_DATA` finalization, reservoir-overflow accounting, and
    C-matched low-delay AU byte size with C/Pure Rust decoder acceptance
  - [x] Iterative LD/ELD threshold fitting to the complete granted access-unit
    size, replacing coarse power-of-two relaxation with a deterministic
    sub-0.1% bracket correction
  - [x] Complete AAC-(E)LD bits-to-PE factor ROM for 16--48 kHz, bitrate
    interpolation, afterburner/dead-zone variants, and long-window reservoir
    bit-save/bit-spend allocation
  - [x] VBR modes 1--5 quality factors, C-exact long-window fourth-root energy
    multiplication, SCALE_NRGS chaos normalization and stateful smoothing,
    min-SNR/absolute ld64 threshold limiting, and variable-size AU generation
    with direct fixed-point per-band/avoid-hole/chaos differentials
  - [x] Long-window CBR bandwise fourth-root threshold correction to granted
    PE with common reduction, min-SNR avoid-hole limiting, and bounded
    iterative convergence
  - [x] Exact long-window form-factor chaos from `sum(sqrt(abs(MDCT)))`,
    active-line reconstruction, energy weighting, and stateful smoothing
  - [x] Stateful PE-to-dynamic-bit correction with FDK's validity window,
    dead zone, direction reset, 0.15/0.30 adaptation, and 0.85--1.15 limits
  - [x] CBR failure recovery with highest-SFB-first 1 dB min-SNR relaxation
    followed by eight logarithmic energy-level hole passes above the
    bitrate-dependent protected low-band region
  - [x] AAC-LD VBR modes 1--5 configuration/default bitrate resolution with
    representative direct C initialization differentials
  - [x] FDK-equivalent threshold adjustment, PE-to-bit distribution, and
    VBR/SFR control
- [x] ER AAC-ELD 480/512-sample mono/stereo core PCM encoder
  - [x] Ported ELDAnalysis ROM and stateful low-overlap analysis filterbank
  - [x] ELD raw access-unit/ASC generation and C FDK mono/stereo acceptance
- [x] AAC-ELD LD-SBR and ELD MPS encoder integration
  - [x] Unframed ELD LD-SBR mono payload writer with LD FIXFIXonly grid,
    envelope/noise/Huffman/harmonic syntax and parser roundtrip coverage
  - [x] 32-band single-rate CLDFB and 64-band dual-rate ELD analysis geometry
    for 480/512-sample core frames, including frequency-range validation
  - [x] Complete C `CODEC_AACLD` mono tuning ROM selection, single-rate
    stop-band adaptation, dual-rate halfband core downsampling, ELD ASC/AU and
    parameter-driven backend integration, with Pure Rust and C decoder
    acceptance
  - [x] C-table 15/16-slot transient grids, per-envelope low/high frequency
    resolution, exact-bit-region CRC10, uncoupled stereo and correlation-driven
    level/balance coupling
  - [x] Complete C `CODEC_AACLD` stereo tuning ROM, single/dual-rate stereo
    analysis/downsampling, ASC/AU and parameter-driven backend integration
  - [x] Correct `CODEC_AACLD` ROM column mapping (`stopFreq` rather than
    `startFreqSpeech`), ELD 1.5 dB amplitude-resolution signaling, executable-C
    rejection gaps, and byte-exact mono/stereo ASC differentials at every
    tuning interval boundary for single- and dual-rate operation
  - [x] Direct C stereo payload differential (using encoder-side capture since
    the bundled C decoder rejects stereo ELD-SBR ASC during initialization),
    plus FDK-equivalent stateful
    time-domain envelope/noise delta and transient-decision tuning
    - [x] Stateful two-slot LD transient energy/candidate history, 5x onset
      threshold, recent-onset dominance filtering, and SBR-band-limited energy
    - [x] FDK fast-detector 4.5--13.5 kHz band selection, 20 dB/16 kHz
      high-pass energy weighting, two-QMF-slot lookahead carry/positioning,
      strongest-ratio selection and direct multi-frame C detector differential
    - [x] C-equivalent 64-band dual-rate CLDFB analysis (including its distinct
      output scale), with direct standard-QMF and CLDFB C subband differentials
    - [x] LD envelope extractor half-frame Y-buffer timing for both 15- and
      16-slot frames: previous trailing half plus current leading half, with
      detector indices 0/1 retained internally and current two-slot lookahead
    - [x] Full C/Rust stereo ELD-SBR AU parsing through the Rust decoder, with
      byte-identical ASC and exact active-header, coupling and independent
      left/right transient-grid comparison
    - [x] Stateful mono envelope frequency/time delta bit-count selection,
      including header reset, first-envelope 1.3 edge weighting and C-decoder
      acceptance of a dependent second access unit
    - [x] C-exact first-envelope frequency-edge adaptation across frames
      (fixed-point 1.3 initial weight, +0.3 per consecutive time-coded frame,
      frequency reset), stereo-channel streak synchronization, bit-exact Q15
      threshold rounding, and direct mono/coupled level/balance multi-frame
      `FDKsbrEnc_codeEnvelope` differentials for both amplitude resolutions
    - [x] C-exact backward/forward two-pass envelope delta limiting, including
      balance-channel LAV limiting before the coupling divisor and stateful
      history updates from the constrained values
    - [x] FIXFIX single-envelope amplitude-resolution selection using
      the C bitrate regions (<28 kbit/s 3.0 dB, >48 kbit/s 1.5 dB and the
      five-highest-energy-band tonality decision between them), with the
      per-frame resolution applied consistently to grid signaling,
      quantization and mono/coupled Huffman books
    - [x] C-exact middle-region global-tonality state: aligned energy selects
      the five bands, current-frame two-segment LPC quotas are converted by the
      FDK 10/3 relaxation scale and averaged with the previous frame; direct C
      threshold and multi-frame transition differentials cover the boundary
    - [x] C `SBR_SWITCH_LRC` fallback when left/right single-envelope
      amplitude resolutions differ, preventing an invalid shared coupled grid
    - [x] FDK multichannel element-bit allocation retained through encoder
      construction and applied independently to each SCE/CPE LD-SBR
      amplitude-resolution decision, with LFE excluded from SBR processing
    - [x] Full C and Rust stereo ELD-SBR access units parsed through the same
      Pure Rust decoder, covering byte-exact ASC, header, coupling and grid
      structure; the C core's extreme spectrum also exposed and fixed an i64
      overflow in the floating-point fixed-MS bridge
    - [x] Stateful mono/stereo noise frequency/time delta selection, including
      noise-specific time Huffman coding and within-header second-envelope use
    - [x] Signal-dependent mono/uncoupled noise-floor values from per-noise-band
      QMF complex-correlation tone/noise odds, FDK four-tap smoothing history,
      transient history reset, +6 dB cap and logarithmic 0--30 quantization,
      replacing the previous constant noise indices
    - [x] Independent left/right envelope and noise history for uncoupled ELD
      stereo payloads
    - [x] Coupled level/balance envelope and level/balance noise histories with
      their distinct frequency/time Huffman books and coding-unit divisors
  - [x] ELD MPS encoder integration
    - [x] Bit-exact byte-aligned 15-band tree-212 SpatialSpecificConfig writer
      for `ELDEXT_LDSAC`
    - [x] Stateful 64/32-QMF stereo CLD/ICC extraction, fine grouped-PCM
      MPS212 NEW/KEEP frame writing, 20-frame independence cadence, 3 dB
      time-domain downmix, and `EXT_LDSAC_DATA` payload framing
    - [x] AAC-ELD mono-core ASC/AU backend, public f32/fixed-i16 decoder QMF
      handoff, dependent-frame state, bit-exact C encoder ASC differential,
      C decoder acceptance of Rust-generated stereo output, and Rust decoder
      acceptance of C-generated low-delay Huffman MPS frames
    - [x] Non-SBR ELDv2 public total/core delay reporting from SACENC's
      LD-QMF delay equations, C-matched mono-core bandwidth selection, and
      byte-aligned LD spatial-frame termination
    - [x] 128-QMF high-rate rejection matching the bundled executable C
      implementation (`mp4SpaceEnc_GetNumQmfBands` returns zero at >=55426 Hz)
    - [x] Correct FDK 15-band mapping plus direct C SACENC broadband CLD/ICC
      extraction and exact parameter differential
    - [x] Direct C byte-aligned Huffman SpatialFrame parsing, including FDK's
      LAV-before-partition-zero syntax and bounded zero-bit termination
    - [x] Narrow-band 32-QMF direct-index parameter mapping, unused-band
      initialization, and direct C CLD/ICC differential
    - [x] FDK minimum-bit grouped-PCM/1D/2D-frequency Huffman selection,
      temporal differentials, escapes, symmetry coding, and byte-exact direct
      C payload differentials for independent and dependent frames
- [x] Full `aacEncoder_SetParam` behavior, including all bitrate modes,
  afterburner, bandwidth, metadata, channel-order and signaling controls
  - [x] All public parameter IDs, setter value validation, C-compatible
    initialization flags/input-buffer resets, sticky ELD downscale semantics,
    and direct C differential coverage
  - [x] Operational `AACENC_CHANNELORDER` mapping for AAC-LC 3.0--5.1 MPEG,
    WAV and the executable C implementation's WG4-as-WAV behavior, with
    byte-identical C encoder differentials
  - [x] Operational `AACENC_AFTERBURNER` propagation for AAC-LC mono, stereo
    and 3.0--5.1 backends, with inverse-quantized 1.25-NMR scalefactor
    refinement and C/Rust on/off access-unit differentials
  - [x] Operational AAC-LC VBR modes 1--5 for mono, stereo and 3.0--5.1,
    using FDK's quality-factor ROM, stateful form-factor chaos threshold
    shaping, full per-frame VBR reservoir availability, variable decodable
    access units, and direct C quality-order differentials
  - [x] Propagation of `AACENC_AFTERBURNER` and VBR 1--5 through HE-AAC mono
    and HE-AAC v2/PS into their AAC-LC core quantizers
  - [x] Operational `AACENC_SIGNALING_MODE` 0/1/2 for HE-AAC and HE-AAC v2:
    implicit, backward-compatible 0x2b7/0x548 sync-extension, and hierarchical
    ASC syntax in LATM/LOAS, including corrected core/output rate ordering,
    backward-extension parsing and byte-exact direct C ASC differentials
  - [x] Deferred initialization checks/defaults for AOT/frame length,
    CBR/VBR/peak bitrate selection, ELD automatic SBR, SBR ratio/signaling,
    and ADTS/LOAS transport selection
  - [x] Unified parameter-driven factory for the implemented LC, HE-AAC,
    HE-AAC v2/PS, AAC-LD, and non-SBR AAC-ELD mono/stereo backends
  - [x] FDK bandwidth tables and MDCT low-pass application for the implemented
    LC/LD/ELD/HE mono/stereo backends, including ordinary HE-AAC mono/core-mono
    SBR tuning selection and C-matched crossover bandwidths at 16-96 kHz
  - [x] Unified RAW/ADIF/ADTS/LATM/LOAS framing, AudioMuxVersion 0/1,
    one-to-four transport subframes, and periodic configuration headers
  - [x] CRC-protected single- and multi-raw-block ADTS output with FDK-compatible
    syntax-region zero padding, block positions, and C decoder acceptance
  - [x] Ancillary input consumption and GA DSE/ER EXT_DATA_ELEMENT output,
    including fixed-rate truncation, zero-rate all-or-nothing limits, and C
    encoder/decoder bidirectional interoperability
  - [x] GA metadata modes 1/2/3, MPEG-4 dynamic-range and ETSI ancillary
    serialization, sticky setup, stateful PCM-derived DRC compression, and
    independent user ancillary elements
  - [x] FDK metadata/audio alignment for AAC-LC, dual-rate HE-AAC and
    HE-AAC v2/PS, including mode-zero audio alignment, PS two-frame metadata
    delay, public total/core encoder-delay reporting, and direct C delay
    differentials
  - [x] Public metadata restriction for AAC-LD/AAC-ELD and PS mono-core
    channel-mode/default-bitrate reconfiguration
  - [x] Live metadata on/off reconfiguration, delayed enable, compressor
    restart, and explicit default-configuration finalization, with a direct C
    access-unit transition differential
  - [x] Bit-exact line/RF compressor payload controls across every profile,
    mono/stereo level transitions and limiter release boundaries, including
    FDK's planar-buffer RF peak traversal behavior
  - [x] AAC-LC and ordinary HE-AAC 3.0 through 7.1 multichannel encoder
    backends, including channel modes 7/11/12/14/33/34, standard and PCE ASC,
    MPEG/WAV input mappings, per-element-type SCE/CPE/LFE tags, synchronized
    main-channel block switching, independent long-window LFE state, Pure Rust
    decode roundtrips, and bundled C decoder acceptance for raw and ADIF output
  - [x] AAC-LD 3.0 through 7.1 multichannel encoder for channel modes
    3/4/5/6/7/11/12/14, including C-exact extended GA ASC signaling,
    type-specific SCE/CPE/LFE tags, synchronized low-overlap switching,
    MPEG/WAV input ordering, metadata/ancillary extensions, and bundled C
    decoder acceptance across every supported layout
  - [x] Public `MODE_212` mapping across AAC-LC/HE-AAC/AAC-LD: mono-core
    LC/HE operation and AAC-LD 32-band low-delay MPEG Surround encoding with
    explicit SSC payload framing, C-exact ASC and bundled C decoder acceptance
  - [x] ELD SBR/MPS multichannel encoder backends
    - [x] Non-SBR AAC-ELD 3.0--7.1 core encoder with channel modes
      7/11/12/14 and standard SCE/CPE/LFE
      element layouts, shared CBR reservoir/fill, MPEG/WAV input ordering,
      Pure Rust roundtrips, byte-exact C ASC differentials and bundled C
      decoder acceptance
    - [x] Single/dual-rate AAC-ELD 3.0--7.1 LD-SBR backends with per-element
      SCE/CPE header tables, payload ordering and independent coding history,
      LFE exclusion, dual-rate core downsampling, byte-exact C ASC
      differentials, Pure Rust roundtrips and bundled C decoder acceptance
    - [x] C-exact binary32-to-Q1.31 element bitrate weights through 7.1,
      remainder-to-first
      distribution and common maximum start/stop frequency selection across
      the complete 3.0--5.1 single/dual-rate ASC differential matrix
- [x] Full decoder operational API behavior corresponding to
  `aacDecoder_SetParam`, flush/conceal/interruption flags, and the complete
  `CStreamInfo` surface
  - [x] Frame-local DSE ancillary-data registration/retrieval with libAACdec's
    seven-element and configured-buffer-capacity limits
  - [x] Owned `CStreamInfo`-shaped configuration snapshot with AAC/core and
    rendered sample rates/frame sizes, PS channel expansion, channel labels and
    indices, extension signaling, ER/SBR/PS/MPS/DRC flags, and metadata defaults
  - [x] Raw/ADTS/ADIF/LATM/LOAS transport byte, successful/bad access-unit and
    bad-byte counters, lost-ADTS estimate, instantaneous bitrate, counter reset,
    and incremental ADTS/LOAS accounting in the stream-info snapshot
  - [x] Typed `AACDEC_CONCEAL`/`FLUSH`/`INTR`/`CLRHIST` decode flags with
    input-independent concealment, core/SBR filter draining, stream-discontinuity
    reset, configuration-preserving history clear, and transport-buffer clear
  - [x] Numeric `aacDecoder_SetParam` equivalents for dual-channel output,
    MPEG/WAV channel mapping, minimum/maximum output-channel constraints, and
    transport-buffer clear, applied consistently to direct, multi-block, and
    incremental PCM output; extension-only channels are reported as `ACT_NONE`
    equivalents in stream info
  - [x] Stateful FDK-style PCM look-ahead limiter with auto/forced enable,
    attack/release controls, history reset, and `outputDelay` reporting
  - [x] Default/explicit concealment frame delay plus GA/LD SBR and MPS module
    delay formulas; AAC-LC limiter/concealment and encoded ELD-SBR delay values
    are checked directly against the bundled C decoder
  - [x] Unified DRC boost/cut scaling, target reference level, requested effect,
    DRC-off, and album-mode parameter routing into instruction selection and
    f32/i16 gain application
  - [x] Remaining decoder parameter behavior
    - [x] `AAC_METADATA_PROFILE` and `AAC_METADATA_EXPIRY_TIME`: numeric
      validation, profile state, ARIB's 550 ms default, frame-count conversion,
      and legacy DRC payload expiry are implemented; DVB DSE center/surround,
      extended A/B/LFE/gain and pseudo-surround metadata plus PCE
      `matrix_mixdown` are parsed, merged, expired, and selected according to
      Standard/Legacy/Legacy-Priority/ARIB precedence for 5.1-to-stereo/mono;
      greater-than-six-channel input uses FDK's metadata-aware first-stage
      reduction (including the MPEG-4 configuration-7 five-front layout,
      A/B levels and gain-5) before the mono/stereo stage, with a direct
      `libPCMutils` differential
    - [x] Legacy MPEG-4 DRC: `EXT_DYNAMIC_RANGE` parsing, channel exclusion,
      program-reference reporting, one-band boost/cut and normalization, plus
      `AAC_DRC_HEAVY_COMPRESSION`, `AAC_DRC_DEFAULT_PRESENTATION_MODE`, and
      `AAC_DRC_ENC_TARGET_LEVEL` parameter validation/state are implemented;
      AAC-LC normalization includes FDK's one-frame delay and first-order
      smoothing and is covered by a genuine C decoder gain-ratio differential;
      core-domain f32/Q31 spectral multi-band application (including repeated
      and interpolated concealment spectra) is implemented and covered by a
      genuine C decoder gain-ratio differential; DVB Data Stream Element
      ancillary compression parsing, heavy-gain application, presentation
      modes 0/1/2, encoder-headroom and output-downmix parameter adjustment are
      also implemented, with a genuine C decoder heavy-compression
      differential. AAC-LC SBR/PS floating and fixed paths now apply the
      multi-band control in the QMF domain with FDK's current/next-frame time
      interpolation, short-window band mapping, concealment reuse, and
      discontinuity reset
    - [x] `AAC_QMF_LOWPOWER`: FDK-compatible real-only standard-QMF and CLDFB
      analysis/synthesis, real LPC transposition, alias-degree reduction and
      energy compensation, LP noise/harmonic rendering, and automatic/explicit
      decoder-mode synchronization (including PS/MPS/USAC constraints), covered
      by direct C correlation/RMS and encoded ELD-SBR waveform differentials
    - [x] Complete processing-method routing for all three
      `AAC_CONCEAL_METHOD` values
      - [x] Parameter validation, codec-dependent defaults, FDK-compatible
        delay selection, spectral mute/noise substitution, and explicit
        `AACDEC_CONCEAL` energy interpolation with one-frame look-ahead are
        implemented for f32/i16 across the supported direct transports; the
        isolated-loss timing and waveform are covered by a C decoder
        differential, and incremental ADTS loss recovery selects the same
        methods
      - [x] The C-style operational decode facade transactionally routes
        complete framed ADTS/LOAS access units with corrupt codec syntax
        through the selected method while retaining bad-AU statistics; strict
        parser facades continue to return the syntax error
      - [x] Automatic corruption classification for complete direct RAW/ADIF
        calls and length-validated direct LATM AudioMuxElements, with f32/i16
        concealment routing while strict parser facades retain syntax errors
    - [x] Unified DRC simple and complex Huffman-coded gain sequences,
      including spline slopes, profile-specific gain deltas, frame-end node
      insertion and C-matched cross-frame node-reservoir ordering
    - [x] Unified DRC v1 configuration extensions, including appended downmix,
      coefficient and instruction sets plus loud-EQ/EQ syntax consumption
  - [x] MPEG-4 DRC program-reference and output-loudness reporting, including
    metadata absence, default normalization and normalization-off behavior,
    with a direct C `CStreamInfo` differential
  - [x] DVB presentation-mode 0/1/2 reporting, including independence from
    the user default-processing mode and persistence beyond compression-data
    expiry, with a direct C `CStreamInfo` differential
  - [x] MPEG-D `LoudnessInfoSet` track/album and any-downmix selection wired
    into output-loudness reporting for the selected Unified DRC instruction
  - [x] In-band USAC `ID_EXT_ELE_UNI_DRC` configuration/gain and
    `ID_CONFIG_EXT_LOUDNESS_INFO` parsing, including default/explicit payload
    lengths, leading/trailing extension ordering for mono/stereo cores,
    automatic gain application and output-loudness state updates
  - [x] USAC MPS leading/trailing extension-element gain parsing, preserving
    core/SBR/spatial reader position and applying selected DRC after stereo
    rendering
  - [x] ELDv2 MPS delay, frame-geometry and sample-rate acceptance
    differentials across every supported 16--48 kHz rate, 480/512 core frame,
    and non-/single-/dual-rate LD-SBR mode, including C-matched rejection of
    out-of-range rates and fractional QMF time-slot configurations
  - [x] Exhaustive ELDv2 initialization matrix over supported sample rates,
    480/512 frames, non-/single-/dual-rate SBR and representative 12--64
    kbit/s rates, including C-matched 27,713 Hz single/dual-rate boundary
- [x] Public encoder profile/configuration differential corpus across AOT
  2/5/29/23/39, every exposed channel mode, standard 8--96 kHz rates,
  480/512/960/1024 granules and off/single-/dual-rate SBR combinations. The
  acceptance matrix is exact against C; signal differentials remain
  representative because every possible PCM stream is not enumerable.

## Completed foundation

- [x] AAC-LC raw access-unit parsing and synthesis for the supported SCE/CPE/
  PCE/CCE subset
- [x] ADTS header/frame parsing, writing, stream iteration, and one
  `raw_data_block` per frame
- [x] ASC parsing/writing and an ASC-configured raw/ADTS transport facade
- [x] AAC-LC spectral Huffman codebooks, scalefactors, PNS, pulse, TNS, MS,
  intensity, f32 reference path, and the current Q31/i16 path
- [x] Explicit rejection of unsupported syntax and transport selections

## Completed C/C++ feature-parity work

### MPEG transport (`libMpegTPDec`)

- [x] ADTS frames containing multiple `raw_data_block`s, including block
  positions and per-block CRC syntax (`tpdec_adts.cpp`)
  - [x] CRC-protected `raw_data_block_position` extraction and sequential f32/i16 block dispatch
  - [x] CRC-less sequential block decode for supported AAC-LC raw blocks with `ID_END` terminators
  - [x] Multi-block ADTS transport-header CRC validation and multi-region CRC accumulator
  - [x] Codec syntax-specific SCE/CPE/CCE/PCE/DSE CRC-region validation for single- and multi-block frames
- [x] Stateful incremental input buffering, synchronization and recovery
  semantics (`tpdec_lib.cpp`)
  - [x] ADTS incremental byte buffering, partial-frame retention, and byte-aligned syncword recovery
  - [x] Recovery from a false but syntactically valid sync header advertising an incomplete maximum-length frame when a later complete frame is buffered
  - [x] ADTS sample-rate/channel configuration change recovery by decoder reinitialization
  - [x] FDK-style CBR lost-access-unit estimation across ADTS sync recovery
  - [x] Optional PCM repeat/fade concealment with FDK default fade factors
  - [x] Fixed AAC-LC last-spectrum concealment with random phase, fade-out, window transition, and overlap preservation
  - [x] Fixed concealment Single/FadeOut/Mute/FadeIn/Ok state transitions and recovery fading
  - [x] Floating-point spectral concealment with matching Single/FadeOut/Mute/FadeIn/Ok transitions, overlap preservation, and ADTS multi-loss integration
  - [x] Q31 scale-factor-band energy interpolation core between surrounding good spectra
  - [x] Transactional one-frame look-ahead connecting surrounding-frame interpolation to ADTS i16
  - [x] Long/short expansion with LongStart/LongStop surrounding-frame interpolation
  - [x] Transactional delayed f32 surrounding-frame interpolation path
- [x] ADIF header and Program Config Element transport handling (`tpdec_adif.cpp`)
  - [x] ADIF header/PCE parsing and byte-aligned AAC-LC raw access-unit dispatch
  - [x] Transactional ADIF incremental buffering through raw_data_block ID_END boundaries
  - [x] Unified f32/Q31 ADIF facade dispatch, including header-prefixed first access units
  - [x] AAC Main/SSR/LTP rejection at configuration, matching the C++ FDK supported-AOT gate
  - [x] ER profiles are rejected as not representable by ADIF: its PCE maps the
    2-bit `profile` directly to AOT 1-4, matching `tpdec_lib.cpp`; supported ER
    AOT 17/23/39 configuration remains available through ASC/RAW/LATM
- [x] LATM AudioMuxElement parsing with in-band and out-of-band configuration
  (`tpdec_latm.cpp`)
  - [x] AAC-LC AudioMuxVersion 0 single-program/layer StreamMuxConfig, PayloadLengthInfo, and useSameStreamMux reuse
  - [x] AudioMuxVersion 0 otherData extraction and StreamMuxConfig CRC field handling
  - [x] AudioMuxVersion 1 single-stream taraBufferFullness, length-delimited ASC, and otherData length parsing
  - [x] In-band AAC-LC configuration change recovery by decoder reinitialization
  - [x] Multiple AAC-LC program/layer streams, multiple subframes, complete mux-state reuse, and frameLengthType 1
  - [x] Generic inline/length-delimited ASC parsing for ER AAC-ELD and USAC, with LOAS USAC multichannel PCM dispatch
  - [x] Direct `LatmMuxConfigPresent` and `LatmOutOfBandConfig` transport facade dispatch with state reuse
- [x] LOAS framing (`tpdec_latm.cpp`, `tpdec_lib.cpp`)
  - [x] LOAS frame/stream framing and LOAS-to-LATM-to-AAC-LC raw dispatch
- [x] Incremental LOAS buffering and synchronization recovery
- [x] DRM transport/configuration parsing (`tpdec_drm.cpp`)
  - [x] SDC type-9 base audio configuration for AAC/CELP/HVXC/xHE-AAC and DRM CRC-8
  - [x] xHE-AAC static decoder configuration and ER/DRM AAC frame decoding, including DRM PS/surround
    - [x] DRM xHE-AAC mono/stereo static configuration mapping into the common USAC decoder model
    - [x] DRM-specific MPS212 spatial static configuration, including temporal-shaping, phase, residual-band, and pseudo-LR fields
    - [x] Stateful DRM xHE-AAC access-unit dispatch through common USAC/MPS and interleaved f32/i16 output
    - [x] DRM AAC CRC-region-aware VCB11/HCR framed payload dispatch
      - [x] FDK-compatible arbitrary bit-offset/bit-length DRM CRC-8 regions
      - [x] 960/120-sample long/short SFB ROM and f32/Q31 filterbank construction
      - [x] Transactional CRC-protected DRM AAC mono VCB11/HCR parsing and 960-sample f32 rendering
      - [x] Transactional CRC-protected DRM AAC CPE shared-ICS/MS, dual HCR, and stereo f32/Q31 rendering
      - [x] DRM AAC direct integer inverse/PNS/TNS, fixed stereo tools, and Q31 rendering without the f32 spectrum bridge
    - [x] DRM AAC SBR/Parametric Stereo and bundled 1-to-2 DRM Surround payload integration
      - [x] Exact-AU-bit-length core/SBR boundary tracking and non-byte-aligned whole-payload DRM bit reversal
      - [x] DRM scalable-SBR default header, frame parsing, QMF rendering, and PS extension handoff
        - [x] DRM default-header mono SBR CRC/frame parsing and transactional dual-rate QMF rendering
        - [x] DRM Parametric Stereo extension parsing, 30-slot QMF stereo rendering, and previous-frame reuse
        - [x] DRM coupled/uncoupled stereo SBR parsing and dual-channel QMF rendering
      - [x] ETSI MPEG Surround mode decoding (5.1/7.1/stream-defined), reserved-value rejection, and implicit 30-slot/28-band 1-to-2 SSC derivation
      - [x] Externally demultiplexed standard 1-to-2 SpatialSpecificConfig/frame attachment and direct SBR-QMF spatial rendering
      - [x] 5.1/7.1 SDC modes retained as informational metadata; bundled `libSACdec` accepts only tree 7 and a maximum of two output channels

### AAC decoder profiles and error-resilience (`libAACdec`, `libArithCoding`)

- [x] Full AAC Main/SSR/LTP/scalable profile syntax where supported by FDK
  - [x] AAC Main/SSR/LTP are outside this FDK decoder's supported-AOT set and are explicitly rejected
  - [x] ER AAC-LC/LD/ELD and single-layer ER scalable profiles supported by FDK
- [x] Error-resilient AAC: RVLC, HCR, arithmetic coding and related concealment
  - [x] ER GASpecificConfig epConfig 0/1, scalable layer, BSAC fields, resilience flags, and extensionFlag3 parsing/writing
  - [x] ER AAC-LC config-driven SCE/CPE/LFE raw_data_block mapping and f32/Q31 decode without resilience tools
  - [x] ER AAC-LC LATM AudioMuxVersion 0/1 configuration signaling
  - [x] ER AAC Scalable AOT 20 layer 0 mapped to the ER AAC-LC 1024/960-point decode path; enhancement layers rejected like FDK
  - [x] RVLC ESC1 side-info, reversible scalefactor Huffman tree, forbidden-codeword detection, and escape tree primitives
  - [x] RVLC forward/backward scalefactor/noise/intensity reconstruction, escape application, and anchor consistency checks
  - [x] RVLC scalefactor payload integration in ER AAC-LC f32/Q31 channel streams
  - [x] HCR protected side-info bounds, length consistency checks, codebook priority ordering, and PCW segmentation grid
  - [x] HCR long/short-window section expansion, priority-codeword decoding, and bidirectional segment cursors
  - [x] HCR non-priority set traversal, segment-spanning codewords, spectral deinterleave, and ER f32/Q31 integration
  - [x] ER RVLC/HCR declared-region validation, stable scalefactor fallback, HCR spectral muting, and truncation distinction
  - [x] ER VCB11 5-bit section syntax, implicit unit sections, virtual codebooks 16-31, LAV checks, and f32/Q31 integration
  - [x] ER AAC-LD AOT 23 configuration, 512/480 SFB tables, RAW/LATM signaling, long-block spectral decode, normal-IMDCT f32/Q31 synthesis, and non-zero FDK differential test
  - [x] ER AAC-ELD AOT 39 ASC flags/extensions, implicit long ICS, epConfig 0 mono/CPE raw syntax, and zero-frame f32/Q31 decoding
  - [x] AAC-ELD 3N synthesis ROM, orthonormal DCT-IV, 2N low-overlap state machine, f32/Q31 integration, and non-zero FDK differential coverage
  - [x] AAC-ELD epConfig 1 CPE side-info/TNS/spectral staging for ordinary and VCB11 streams
  - [x] AAC-ELD epConfig 1 CPE staging with RVLC/HCR
  - [x] AAC-ELD LD-SBR
    - [x] ASC default-header parsing/writing, including per-channel-element header counts and optional header fields
    - [x] 480/512-sample FIXFIX/transient grid tables, stereo coupling/grid selection, and envelope/noise direction-vector parsing
    - [x] Master/high/low/noise frequency-table derivation, inverse-filter modes, and FDK envelope/noise Huffman decoding primitives
    - [x] Coupled/uncoupled stereo payload ordering, frequency/time delta reconstruction with prior-frame state, harmonic flags, and extension-byte framing
    - [x] Stateful explicit frame parser with header overrides, CRC10 validation/rollback, and AOT39 core-tail decoder integration
    - [x] Coupled level/balance energy unmapping and envelope/noise dequantization
    - [x] LD-SBR QMF processing and output integration
      - [x] FDK 640-tap ROM, 32-band phase tables, analysis polyphase state, DCT-IV/DST-IV modulation
      - [x] FDK-style patch layout/copy, envelope gain, fixed 512-entry ROM noise phase sequencing/attack suppression, stateful four-phase center-band harmonic mapping with prior-frame continuation, 64/32-band synthesis state, and AOT39 output sizing integration
      - [x] Mode-transition-smoothed inverse-filter whitening, patch-aware bands-per-octave limiter tables, average-energy-relative gain caps, +4 dB boost compensation, and FDK four-slot/attack-aware persistent gain smoothing
      - [x] Direct C full limiter-component differential covering multi-subband average limiting, clipped gain/noise tracking, and +4 dB boost
      - [x] C FDK acceptance/output-shape differential for a generated dual-rate harmonic LD-SBR frame
      - [x] Direct C FDK QMF differentials: 32-band analysis correlation >0.985, 64-band synthesis correlation >0.9999/RMS ±0.1%, and 32→64 roundtrip correlation >0.985/RMS ±0.5%
      - [x] Q15/32-band envelope-energy normalization with non-zero C FDK integrated RMS calibration (0.75–1.5× guard)
      - [x] Non-zero C FDK waveform shape and delay calibration
        - [x] FDK startup-frame gain-smoothing history initialization from current envelope gains
        - [x] FDK-encoder-generated single-rate ELD-SBR ASC and stateful independent/dependent raw-AU Pure Rust decode fixture
        - [x] FDK frequency-table count differential and stop-frequency index off-by-one correction
        - [x] Bit-exact FDK Q31/Q15 master-band border generation and C border-array differential
        - [x] Single-rate versus dual-rate four-bit envelope-energy scaling with genuine encoded RMS guard
        - [x] Genuine encoded f32 waveform/delay correlation (lag 0, correlation >0.99999)
        - [x] Fixed-i16 ELD public decode waveform/delay correlation via the pure-Rust exponent-preserving ELD path (lag 0, correlation >0.99, RMS ±5%)
        - [x] Native Q31 ELD inverse-spectrum block exponent parity
          - [x] Per-window mantissa/exponent representation, non-saturating ELD inverse quantization, PNS exponent alignment, and exponent-aware LDFB core synthesis
          - [x] Preserve Q31 LDFB core through LD-SBR without intermediate i16 precision loss; genuine encoded lag 0, correlation >0.99999, RMS ±0.01%
- [x] Complete CCE gain/coupling behavior and all channel-element combinations
  - [x] CCE target syntax, coupled spectral stream decoding, common/bandwise DPCM gains with signed exponent coding and all four gain steps, frequency/time-domain f32/Q31 application, and SCE/CPE/LFE target routing
  - [x] Implicit unity gain-element-list 0 alignment and sampling-rate-specific SFB routing
  - [x] Multi-target gain-list ordering across SCE and both CPE channels
  - [x] Multiple trailing frequency-domain CCE parsing and additive non-zero f32/Q31 integration
  - [x] PCE element-tag-routed non-zero CCE integration in strict f32/Q31 raw decoding
  - [x] Three-way CCE coupling-point derivation and correct frequency/time-domain routing from `ind_sw_cce_flag` plus `cc_domain`
  - [x] Apply coupling-point 0 before target-channel TNS, coupling-point 1 after TNS, and coupling-point 3 after IMDCT in both f32/Q31 paths
  - [x] External differential limitation documented: the bundled C FDK decoder parses but explicitly does not render CCE; Rust non-zero generated vectors cover the implemented superset
- [x] Decoder concealment and streaming error-recovery behavior
  - [x] ADTS sync-loss estimation, single-loss surrounding-frame interpolation, and multi-loss f32/Q31 spectral concealment
  - [x] Transactional ADTS decoding with automatic f32/Q31 concealment for syntactically present but corrupt access units
  - [x] AAC-ELD core concealment through the profile-specific 3N low-delay f32/Q31 filterbanks
  - [x] LD-SBR-aware concealment with retained grid/envelope/noise/harmonic frames and continuous QMF state/output rate
  - [x] Profile-specific surrounding-frame energy interpolation for supported AAC-LD/AAC-ELD 480/512-point spectra
- [x] Dynamic-range-control integration

### SBR, PS, and xHE-AAC (`libSBRdec`, `libAACdec/usacdec_*`)

- [x] HE-AAC SBR bitstream parsing and QMF synthesis
  - [x] AAC fill-element EXT_SBR_DATA/EXT_SBR_DATA_CRC extraction, CRC10/header flag parsing, shared SBR header decoding, and bit-exact frame-data preservation
  - [x] Ordinary FIXFIX/FIXVAR/VARFIX/VARVAR grids, relative borders, pointers, transient-envelope mapping, frequency resolution, and noise borders
  - [x] Stateful ordinary mono-SBR direction, inverse-filter, envelope/noise Huffman, delta reconstruction, dequantization, harmonic, and extended-data parsing
  - [x] Ordinary stereo/coupled grids, shared inverse filtering, level/balance Huffman reconstruction, and coupled dequantization
  - [x] AAC-LC AudioSpecificConfig/fill-element integration and dual-rate floating-point QMF output
  - [x] AAC-LC SBR fixed-point API output with independent parser and QMF state
  - [x] AAC-LC SBR concealment for explicit losses and missing fill payloads with retained mono/stereo frame and QMF state
- [x] Parametric Stereo decoding
  - [x] Stateful PS header inheritance, FIX/VAR envelope borders, IID/ICC coarse/fine Huffman decoding, time/frequency delta reconstruction, 10/20/34-bin resolution mapping, and extension-byte consumption
  - [x] PS SBR-extension-id extraction, FDK scale/alpha ROM loading, IID/ICC matrix reconstruction, envelope interpolation, stateful QMF decorrelation, and dual-channel QMF synthesis API
  - [x] Implicit HE-AAC v2 PS detection and f32/i16 decoder integration with stereo labels, retained PS/SBR concealment, and missing-fill recovery
  - [x] FDK THREE_TO_TEN 13-tap hybrid low-band analysis/synthesis, 6+2+2 split, 71-band group mapping, and six-slot high-band delay compensation
  - [x] FDK PS complex all-pass coefficient ROM, 2-slot input delay, 3/4/5-stage rings for hybrid bands 0-29, and 14/1-slot upper-band decorrelation delays
  - [x] FDK PS transient ducker peak decay, peak-difference and smoothed direct-energy tracking with 20-band gain limiting
- [x] USAC/xHE-AAC transform and linear-prediction decode paths
  - [x] AOT 42 ASC and non-SBR UsacConfig parsing/writing: sampling frequency, core frame-length index, channel configuration, SCE/CPE/LFE core config, extension elements, escaped lengths, and config extensions
  - [x] USAC SbrConfig default header, harmonic SBR, inter-TES, PVC, and 4:1/8:3/2:1 core/output frame-length mapping
  - [x] USAC SBR ordinary-grid mono/stereo payload parsing with no legacy data-extra/extension syntax, independent first-DF forcing, harmonic patch controls, and shared Huffman/delta/dequantization state
  - [x] USAC SBR mono inter-TES envelope-loop plus stateful PVC noise/variable-HF grid, ID, inverse-filter, noise-delta, and harmonic payload parsing
  - [x] USAC SBR stereo coupled/uncoupled inter-TES parsing in envelope payload order
  - [x] USAC SBR f64 inter-TES energy-preserving QMF shaping and stateful PVC ROM prediction/group gain/noise/harmonic rendering
  - [x] USAC MPS stereoConfigIndex 1/2 `sbr: Some` shared core/SBR/MPS reader, info/default/full-header orchestration, ordinary/PVC QMF rendering, and direct MPS handoff
  - [x] USAC Mps212Config for nonzero stereoConfigIndex
  - [x] USAC MPS212 fixed/high-rate parameter-set framing, data-mode state machine, PCM plus 1D/2D Huffman frequency/time-delta lossless coding, frequency/time pairing, LAV/escape coding, coarse/fine history conversion, and centered frequency-stride expansion
  - [x] USAC MPS212 CLD/ICC/IPD dequantization matrix with power-normalized complex spatial upmix
  - [x] USAC MPS212 four-reverb-band USAC all-pass decorrelator, config band offsets, FDK 4/5/7/10/14/20/28-to-71 mappings, parameter-set interpolation, and 64-QMF/71-hybrid dual-channel PCM processor
  - [x] USAC stereoConfigIndex=1 public-decoder construction and shared-reader core-to-MPS residual-bit frame parsing API
  - [x] USAC MPS212 standard 64-band time interface and direct public multichannel PCM API integration
  - [x] USAC MPS212 stereoConfigIndex 2 dual-core residual decode, prediction matrix, residual-band decorrelator replacement, independent QMF/hybrid state, and public decoder construction
  - [x] USAC MPS212 stereoConfigIndex 3 unified stereo and SBR-domain QMF handoff
  - [x] USAC MPS212 high-rate parameter smoothing mode/time/frequency-stride/band flags
  - [x] USAC MPS212 TSD enable, 32/64-slot combinatorial transient-position decoding, and per-transient phase payload
  - [x] USAC MPS212 STP channel enables and GES reshape run-length Huffman envelopes
  - [x] USAC frame independency flag, per-channel FD/LPD core modes, ACELP core mode, all 0-25 lpd_mode mappings, BPF control, previous-core flag, and FAC-present side information
  - [x] USAC FD TNS-present flag, global gain, noise level/offset, long/short window sequence and shape, max_sfb validation, scale-factor grouping, and window-group derivation
  - [x] USAC implied-first scale factors and stateful context-adaptive arithmetic spectral decoding for long/eight-short windows
  - [x] USAC AVQ qn/index parsing, complete RE8 leader/permutation/Voronoi lattice decoding, logarithmic FAC gain, and FAC payload decoding
  - [x] USAC 256-entry absolute LSF codebook, distance-weighted AVQ refinement, 50 Hz LSF reordering, interpolation, and LSF-to-LSP conversion
  - [x] USAC five-slot LPC frame ordering with abs/mid/relative modes, previous-LPC recovery, adaptive mean/stability, and LSP-to-LPC predictor conversion
  - [x] USAC TCX payload parsing, arithmetic residual decoding, zero-band noise filling, RMS/global-gain normalization, FAC mode gains, and DCT-IV synthesis
  - [x] USAC ACELP payload parsing for all eight core modes, fractional/relative pitch lag, LTP flags, 12-64 bit innovative indices, and pitch/innovative gain ROM
  - [x] USAC ACELP 4-track/64-sample algebraic codevectors, quarter-sample pitch interpolation, LTP postfilter, code pre-emphasis/sharpening, excitation mixing, and stateful LPC synthesis
  - [x] USAC integrated LPD channel payload ordering across ACELP/TCX20/40/80, FAC transitions, shared arithmetic state, and trailing five-slot LPC data
  - [x] USAC LPD renderer with ACELP LSP interpolation/deemphasis, TCX IMDCT overlap-add, adaptive low-frequency deemphasis, interpolated FDNS, and FAC boundary addition
  - [x] USAC FAC weighted-LPC zero-input response/window crossfade and stateful mono LPD access-unit decoder constructed from UsacConfig
  - [x] USAC 768/1024 FD long/eight-short SFB, scale-factor/arithmetic/inverse-quantization/IMDCT path and top-level mono FD/LPD access-unit dispatch
  - [x] USAC long/short TNS and FD FAC payloads, MS/complex-prediction side information, and AOT 42 construction/public access-unit API on AacLcDecoder
  - [x] USAC complex-prediction spectral upmix with alpha scaling/previous downmix, CPE LPD and mixed FD/LPD state, and public multichannel AOT 42 access-unit API
  - [x] USAC both-FD CPE common/non-common windows, common_max_sfb, common/individual TNS routing, stereo-tool application, and dual-channel rendering
- [x] USAC ACELP, FAC, LPC/LPD, arithmetic coding, and configuration handling

### DRC (`libDRCdec`)

- [x] MPEG-D DRC configuration/payload parsing
  - [x] Base channel layout and empty UniDrcConfig foundation parsing
  - [x] Basic DRC coefficient/instruction syntax consumption, matching the C++ decoder's skip behavior
  - [x] v0 downmix instructions, coefficient table, and f32/i16 matrix application
  - [x] v0 DRC instruction headers, effects, loudness/peak targets, dependencies, channel grouping, ducking, and gain modification
  - [x] v0 DRC coefficients, gain-set profiles, interpolation flags, bands, characteristics, and borders
  - [x] v0 LoudnessInfoSet, all measurement value mappings, peak metadata, and selection
  - [x] v1 configuration payload integration
    - [x] v1 custom sigmoid/node characteristics, shape filters, explicit gain-sequence indices, and band characteristics
    - [x] Bit-exact unknown configuration/gain extension envelope preservation
    - [x] v1 downmix coefficient matrices and config-extension integration
    - [x] v1 instructions with complexity/EQ flags and per-band gain/shape modifications
    - [x] loud-EQ instruction syntax consumption (matching the C++ decoder's skip behavior)
    - [x] v1 EQ coefficient/instruction syntax consumption (matching the C++ decoder's skip behavior)
- [x] Gain decoding, selection and application
  - [x] Program loudness normalization selection and static f32/i16 gain application
  - [x] Constant and simple-mode regular/fading/clipping UniDrcGain sequences and PCM application
  - [x] Complex node count, slope/delta-gain Huffman, time-delta, and node-reservoir decoding
  - [x] Per-sample linear and cubic-Hermite spline time-varying gain application
  - [x] Instruction selection by downmix/effect/target loudness and per-channel gain-set application
  - [x] Stateful UniDrcConfig/UniDrcGain attachment to normal and concealed AAC decoder output with automatic selected-instruction application in f32/i16 paths

### Encoder parity (`libAACenc`, `libSBRenc`)

- [x] Complete C/C++ encoder feature parity
- [x] AAC-LC Pure Rust encoder analysis, quantization, bit allocation and
  bitstream writing
  - [x] Stateful 1024/960-point 2N sine-window MDCT analysis with overlap history and transient-energy metric
  - [x] Long-window SFB psychoacoustic energy/flatness, asymmetric spreading, absolute threshold, masking threshold, and perceptual entropy
  - [x] Transient-driven legal OnlyLong/LongStart/EightShort/LongStop block-switch state machine
  - [x] Eight-short 8×128 MDCT at AAC overlap positions, energy-change window grouping, and 14-band short-SFB psychoacoustic analysis
  - [x] Decoder-inverse-matched 3/4-power SFB quantization, masking-noise scalefactor search, 8191 LAV guard, and iterative target-bit threshold relaxation
  - [x] Exact decoder-table-derived Huffman codebook costs and joint dynamic-programming SFB section selection
  - [x] Stateful nominal/saved-bit reservoir allocation with exact coded-size accounting and underflow rejection
  - [x] Long-window SCE raw_data_block ICS, sections, scalefactors, spectral Huffman/ESC, ID_END, and byte-padding writing
  - [x] Eight-short grouped SCE and long/short common-window CPE raw_data_block writing
  - [x] Stateful mono PCM-to-raw/ADTS encoder facade with legal transition windows and reservoir accounting
  - [x] Stateful stereo PCM-to-common-window-CPE/ADTS facade with synchronized long/short switching
  - [x] Stateful 3--8 channel PCM frontend, synchronized main-channel
    long/short switching, independent long-window LFE, standard
    channel-configuration SCE/CPE/LFE writing, MPEG/WAV/WG4 input mapping and
    C encoder/decoder acceptance
  - [x] C-table SCE/CPE/LFE relative dynamic-bit allocation for every 3.0--7.1
    channel mode, including CPE-local channel splitting
  - [x] AAC-LC afterburner analysis-by-synthesis distortion refinement for
    long and grouped-short windows across mono, stereo and multichannel
  - [x] AAC-LC VBR 1--5 long/grouped-short psychoacoustic threshold control,
    stateful chaos smoothing and variable access-unit sizing
- [x] SBR/PS encoder
  - [x] Stateful 64-band QMF analysis, transient envelope split, high-band energy/tonality estimation
  - [x] Mono FIXFIX envelope/noise quantization, SBR Huffman writing, fill-element framing, and AAC FIL integration
  - [x] One/two-envelope FIXFIX mono and uncoupled-stereo SBR payload writing
  - [x] 20-band IID/ICC QMF analysis, PS Huffman/extended-data writing, and stateful HE-AAC v2 stereo-to-mono facade
  - [x] Stateful halfband downsampling HE-AAC mono facade with implicit-SBR decoder roundtrip
  - [x] Ordinary HE-AAC stereo and 3.0--5.1 facades with per-element SBR
    fill placement, LFE upsampling-only decode, and C decoder acceptance
  - [x] MPEG-2 virtual AOT 129/132 mapping across the supported channel layouts
  - [x] Tonality-driven SBR harmonic flag writing
  - [x] Peak-position VARFIX/FIXVAR adaptive grids and QMF-correlation stereo coupling decision
- [x] Encoder transports and metadata paths
  - [x] MPEG-4 AAC-LC ADTS access-unit wrapping and encoder-to-decoder roundtrip
  - [x] AAC-LC/HE-AAC 6.1 and nonstandard 7.1 ADTS signaling with
    channel-configuration zero, per-access-unit PCE insertion, HE core-rate
    headers/PCEs, and bundled C decoder acceptance
  - [x] Stateful LATM StreamMuxConfig/useSameStreamMux output and LOAS framing with encoder-to-decoder roundtrip
  - [x] Constant-rate mono ADIF/PCE header and stateful header-once access-unit output
  - [x] AAC-LC 3.0--5.1 ADIF/PCE front/back/LFE layouts with C decoder acceptance
  - [x] Unified parameter-driven RAW/ADIF/ADTS/LATM/LOAS framing with
    multi-subframe aggregation and periodic StreamMuxConfig emission
  - [x] DRM mono/stereo 960-sample ER-AAC, fixed-priority HCR, CRC-8 packets, and AAC SDC metadata
  - [x] AAC-LC/HE-AAC/HE-AAC-v2 metadata modes, compressor, payloads, audio
    alignment and `nDelay`/`nDelayCore` parity
- [x] ER AAC-LD spectral coding foundation
  - [x] 480/512-sample mono/stereo SFB quantization and ER raw access units
  - [x] AOT 23 ASC generation and Pure Rust decoder roundtrip
  - [x] C FDK decoder acceptance for non-zero mono access units
- [x] ER AAC-LD stateful 480/512-sample long-window PCM analysis frontend
- [x] ER AAC-LD FDK-equivalent transient low-overlap switching
  - [x] Direct C block-switch decisions, adaptive IMDCT window-state
    differential, and decoded transient correlation/RMS coverage
- [x] ER AAC-LD FDK-equivalent psychoacoustic rate control
  - [x] Low-delay CBR reservoir capacity and initial fullness
  - [x] CBR fill-bit finalization and reservoir-overflow accounting
  - [x] Iterative complete-AU threshold fitting within the granted bit budget
  - [x] Bits-to-PE tables and PE/reservoir-dependent long-window bit grant
  - [x] Long-window VBR quality threshold curve and stateful chaos control,
    including a direct C fixed-point differential for channel chaos, history,
    per-band thresholds and avoid-hole transitions
  - [x] Bandwise CBR threshold correction to the granted PE
  - [x] Exact form-factor chaos and PE correction history
  - [x] High-band hole/min-SNR fallback
  - [x] Static/extension versus dynamic grant split and stateful FDK
    `adjustPeMinMax` reservoir-control range adaptation
  - [x] Direct C fixed-point differentials for multi-frame `peMin`/`peMax`
    state and long-window reservoir bit factor
  - [x] Direct C per-band adapted-threshold differential against
    `FDKaacEnc_calcThreshExp`/`FDKaacEnc_reduceThresholdsCBR`, including
    no-avoid-hole, inactive/active avoid-hole state, min-SNR clipping and the
    29 dB threshold floor (the internal SFR QC mode remains unavailable through
    FDK's public encoder initialization API)
- [x] ER AAC-ELD core PCM/spectral mono/stereo encoding
- [x] ER AAC-ELD LD-SBR and MPS encoding
  - [x] Configured single/dual-rate mono/stereo LD-SBR, transient grids,
    switchable coupling and CRC10; mono is accepted by the C decoder
  - [x] Exact stateful envelope/noise decisions, stereo C differential and MPS
    - [x] Stateful inverse-filter mode selection with FDK AAC region tables,
      transient table, three-frame FIR smoothing, low-energy compensation and
      previous-region hysteresis; modes are retained per noise band and emitted
      by mono, coupled and uncoupled LD-SBR paths
    - [x] Second-order complex LPC tonal/noise quota analysis in two LD frame
      segments and FDK `resetPatch`-equivalent source-band mapping, including a
      direct C mapping differential across sampling rates and crossover bands
    - [x] Shared LPC quotas for noise-floor estimation, including the FDK music
      `weightFac=0.25` original/patched-band correction when inverse filtering
      exceeds MID, silence quota, transient reset and four-tap smoothing
    - [x] Pure Rust Q1.31 LPC quota path with FDK-style input normalization,
      exact `ceil(log2(len))` complex-autocorrelation accumulation scaling,
      determinant normalization, relaxation arithmetic and 16-bit Schur
      division; direct C tests are raw-bit-exact for tonal/noise-like 7- and
      8-slot LD segments and prove identical detector regions
    - [x] Direct C `fMultDiv2` high-word differential over signed extrema and
      1,233 deterministic operand pairs, proving the residual tonal quota drift
      is confined to autocorrelation/determinant normalization rather than the
      primitive Q1.31 multiply
    - [x] Direct C LD-SBR envelope pipeline differentials for complex QMF
      energy scaling, dynamic/saturating SFB accumulation, 16-bit-coefficient
      ld64 logarithm and final energy quantization
    - [x] Test-only C/Rust pre-quantization snapshots proving identical
      Y-buffer/QMF/common scales, envelope geometry and SFB sample counts; the
      stereo envelope drift at that checkpoint was confined to CLDFB mantissa rounding
    - [x] Bit-exact fixed-point CLDFB modulation. Prototype and phase ROM
      coefficients now receive the C build's Q15 quantization, reducing raw-PCM
      mantissa error. Direct C FFT32/DCT-IV/DST-IV stage bridges now expose the
      exact scale factors and outputs independently. A Rust Q31 radix-2 FFT32
      probe matches the C scale and impulse exactly and bounds deterministic
      broadband input drift to 10 LSB. Test-only captures now expose all three
      C FFT32 butterfly-stage buffers. All three FFT32 stages have now been
      transcribed with C-identical add/shift, Q15 twiddle and separate-product
      rounding order; every intermediate and final word is raw-bit-exact for
      deterministic broadband input. The complete kernel now lives in the
      no-default-compatible production module and its composed API is directly
      C-differential-tested. The complete 64-point DCT-IV and DST-IV pre/post
      Q15 twiddle paths are also raw-bit-exact against C, including their scale
      factor of six. They and the Q15 complex phase rotation are integrated in
      the 64-band analysis path. The Q15 prototype FIR now also uses the exact
      i16 state grid, five-product accumulation order and left shift, making
      all raw-PCM output mantissas bit-exact against C (previous drift roughly
      1,651). The encoder-to-QMF five-sample PCM handoff and half-frame energy
      history are now C-exact as well; the direct stereo encoder differential
      proves identical QMF input, complex subbands, Y-buffer scales, prequant
      energies/counts, envelope/noise values and harmonic flags
    - [x] ELD MPS SSC, 32/64-QMF CLD/ICC extraction, grouped-PCM payload,
      independence cadence and time-domain downmix foundation
    - [x] ELD MPS AAC mono-core backend, ASC/AU integration, Rust f32/i16
      decode roundtrip, C decoder acceptance, and C-generated AU decoding
    - [x] Single/dual-rate AAC-ELDv2 LD-SBR+MPS combined backend, including
      separated core/spatial sampling geometry, shared mono-core SBR analysis,
      ordered SBR/LDSAC payloads, 32/64-QMF MPS rendering, Pure Rust
      roundtrip, byte-exact single/dual-rate C ASC differentials, and bundled
      C decoder acceptance
    - [x] Non-SBR ELDv2 total/core delay and mono-core bandwidth direct C
      differentials, with C decoder acceptance across the changed core/MPS
      bit boundary
    - [x] Bundled-C-compatible rejection of the non-instantiable 128-QMF mode
    - [x] Correct FDK 15-band mapping and direct C broadband CLD/ICC differential
    - [x] Direct C low-delay Huffman payload parsing and parameter recovery
    - [x] Narrow-band 32-QMF parameter extraction direct C differential
    - [x] FDK Huffman/PCM coding choice with byte-exact independent/dependent
      C SpatialFrame payload differentials

## Completion criteria

Pure Rust feature parity for the declared scope is complete: the supported
C/C++ public decoder/encoder transports and profiles above have Pure Rust
implementations, the `fdk-aac-rust-sys` C/C++ dependency is optional for those
paths, and differential tests cover representative C/C++ fixtures for each
feature.

## Current validation evidence

- [x] `cargo test -p fdk-aac-rust --lib`: Pure Rust plus C/C++ differential suite
- [x] `cargo test -p fdk-aac-rust --no-default-features --lib`: codec/transport suite without `fdk-aac-rust-sys`
- [x] Bidirectional FDK-generated AAC → Pure Rust decoder and Pure Rust encoder → FDK decoder interoperability
- [x] `fdk-aac-rust-sys` is an optional dependency selected only by the default `ffi` feature

These gates establish the implemented interoperability claims above. Continued
testing against newly pinned upstream revisions maintains that status; it does
not broaden the scope into a universal guarantee for every bitstream, platform,
private implementation detail, or future FDK configuration.
