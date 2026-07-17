//! Pure Rust fixed-point primitives used by the FDK AAC codec.
//!
//! This module mirrors a small, well-tested subset of `libFDK`'s fixed-point
//! helpers.  The goal is to move DSP building blocks into safe Rust before
//! porting larger AAC decode stages.

pub type FixpDbl = i32;
pub type FixpSgl = i16;
pub type Pcm16 = i16;

pub const FRACT_BITS: i32 = 16;
pub const DFRACT_BITS: i32 = 32;
pub const SAMPLE_BITS: i32 = 16;

pub const MAXVAL_DBL: FixpDbl = i32::MAX;
/// FDK avoids using `0x8000_0000` as a saturated negative fractional value in
/// several scale paths and clamps to `MINVAL_DBL + 1` instead.
pub const MINVAL_DBL_PLUS_ONE: FixpDbl = i32::MIN + 1;
pub const MAXVAL_SGL: FixpSgl = i16::MAX;
pub const MINVAL_SGL: FixpSgl = i16::MIN;

pub fn scale_value(value: FixpDbl, scalefactor: i32) -> FixpDbl {
    if scalefactor >= 0 {
        value.wrapping_shl(scalefactor.min(DFRACT_BITS - 1) as u32)
    } else {
        value >> (-scalefactor).min(DFRACT_BITS - 1)
    }
}

pub fn scale_value_saturate(value: FixpDbl, scalefactor: i32) -> FixpDbl {
    if scalefactor >= 0 {
        let shift = scalefactor.min(DFRACT_BITS - 1) as u32;
        let shifted = (value as i64) << shift;
        clamp_i64_to_fixp_dbl_fractional(shifted)
    } else {
        let shift = (-scalefactor).min(DFRACT_BITS - 1) as u32;
        if shift >= 31 {
            0
        } else {
            (value >> shift).max(MINVAL_DBL_PLUS_ONE)
        }
    }
}

pub fn scale_values(values: &mut [FixpDbl], scalefactor: i32) {
    if scalefactor == 0 {
        return;
    }

    let shift = scalefactor.abs().min(DFRACT_BITS - 1) as u32;
    if scalefactor > 0 {
        for value in values {
            *value = value.wrapping_shl(shift);
        }
    } else {
        for value in values {
            *value >>= shift;
        }
    }
}

pub fn scale_values_saturate(values: &mut [FixpDbl], scalefactor: i32) {
    if scalefactor == 0 {
        return;
    }
    for value in values {
        *value = scale_value_saturate(*value, scalefactor);
    }
}

pub fn scale_values_saturate_from(dst: &mut [FixpDbl], src: &[FixpDbl], scalefactor: i32) {
    assert_eq!(dst.len(), src.len());
    for (dst, src) in dst.iter_mut().zip(src) {
        *dst = scale_value_saturate(*src, scalefactor);
    }
}

pub fn sgl_to_dbl(value: FixpSgl) -> FixpDbl {
    // Rust defines signed left shift as a two's-complement bit operation. This
    // is intentionally not the undefined C operation reported upstream in
    // mstorsjo/fdk-aac#152.
    (value as FixpDbl) << (DFRACT_BITS - FRACT_BITS)
}

pub fn dbl_to_sgl(value: FixpDbl) -> FixpSgl {
    (value >> (DFRACT_BITS - FRACT_BITS)) as FixpSgl
}

pub fn pcm16_to_dbl(value: Pcm16) -> FixpDbl {
    sgl_to_dbl(value)
}

pub fn dbl_to_pcm16(value: FixpDbl) -> Pcm16 {
    dbl_to_sgl(value)
}

pub fn scale_value_sgl_saturate(value: FixpSgl, scalefactor: i32) -> FixpSgl {
    dbl_to_sgl(scale_value_saturate(sgl_to_dbl(value), scalefactor))
}

pub fn scale_values_sgl_saturate(values: &mut [FixpSgl], scalefactor: i32) {
    if scalefactor == 0 {
        return;
    }
    for value in values {
        *value = scale_value_sgl_saturate(*value, scalefactor);
    }
}

pub fn scale_values_sgl_saturate_from(dst: &mut [FixpSgl], src: &[FixpSgl], scalefactor: i32) {
    assert_eq!(dst.len(), src.len());
    for (dst, src) in dst.iter_mut().zip(src) {
        *dst = scale_value_sgl_saturate(*src, scalefactor);
    }
}

pub fn scale_values_pcm16_from_dbl(dst: &mut [Pcm16], src: &[FixpDbl], scalefactor: i32) {
    assert_eq!(dst.len(), src.len());
    for (dst, src) in dst.iter_mut().zip(src) {
        *dst = dbl_to_pcm16(scale_value_saturate(*src, scalefactor));
    }
}

pub fn get_scalefactor_dbl(values: &[FixpDbl]) -> i32 {
    let mut max_val = 0i32;
    for &value in values {
        max_val |= value ^ (value >> 31);
    }
    (fixnormz_d(max_val) - 1).max(0)
}

pub fn get_scalefactor_sgl(values: &[FixpSgl]) -> i32 {
    let mut max_val = 0i16;
    for &value in values {
        max_val |= value ^ (value >> 15);
    }
    (fixnormz_s(max_val) - 1).max(0)
}

pub fn get_scalefactor_pcm_stride(values: &[Pcm16], len: usize, stride: usize) -> i32 {
    if len == 0 {
        return 31;
    }
    assert!(stride > 0);
    assert!((len - 1) * stride < values.len());

    let mut max_val = 0i16;
    let mut index = 0;
    for _ in 0..len {
        let value = values[index];
        max_val |= value ^ (value >> 15);
        index += stride;
    }

    (fixnormz_d(max_val as i32) - 1 - (DFRACT_BITS - SAMPLE_BITS)).max(0)
}

pub fn saturating_add_dbl(lhs: FixpDbl, rhs: FixpDbl) -> FixpDbl {
    clamp_i64_to_fixp_dbl_fractional(lhs as i64 + rhs as i64)
}

pub fn saturating_sub_dbl(lhs: FixpDbl, rhs: FixpDbl) -> FixpDbl {
    clamp_i64_to_fixp_dbl_fractional(lhs as i64 - rhs as i64)
}

/// Q31 x Q31 -> Q31 saturating multiply.
pub fn mul_q31(lhs: FixpDbl, rhs: FixpDbl) -> FixpDbl {
    let product = ((lhs as i64) * (rhs as i64)) >> 31;
    clamp_i64_to_fixp_dbl_fractional(product)
}

pub fn fixnormz_d(value: FixpDbl) -> i32 {
    if value == 0 {
        DFRACT_BITS
    } else {
        value.leading_zeros() as i32
    }
}

pub fn fixnormz_s(value: FixpSgl) -> i32 {
    if value == 0 {
        FRACT_BITS
    } else {
        (value as u16).leading_zeros() as i32
    }
}

pub(crate) fn clamp_i64_to_fixp_dbl_fractional(value: i64) -> FixpDbl {
    if value > MAXVAL_DBL as i64 {
        MAXVAL_DBL
    } else if value < MINVAL_DBL_PLUS_ONE as i64 {
        MINVAL_DBL_PLUS_ONE
    } else {
        value as FixpDbl
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saturating_scale_clamps_positive_and_negative() {
        assert_eq!(scale_value_saturate(0x4000_0000, 1), MAXVAL_DBL);
        assert_eq!(scale_value_saturate(-0x4000_0000, 2), MINVAL_DBL_PLUS_ONE);
        assert_eq!(scale_value_saturate(0x4000_0000, -1), 0x2000_0000);
    }

    #[test]
    fn vector_scale_saturates() {
        let mut values = [0x1000_0000, 0x4000_0000, -0x4000_0000];
        scale_values_saturate(&mut values, 2);
        assert_eq!(values, [0x4000_0000, MAXVAL_DBL, MINVAL_DBL_PLUS_ONE]);
    }

    #[test]
    fn converts_between_single_and_double_precision() {
        assert_eq!(sgl_to_dbl(0x4000), 0x4000_0000);
        assert_eq!(dbl_to_sgl(0x4000_1234), 0x4000);
        assert_eq!(pcm16_to_dbl(-0x4000), -0x4000_0000);
        assert_eq!(dbl_to_pcm16(-0x4000_1234), -0x4001);
        assert_eq!(sgl_to_dbl(i16::MIN), i32::MIN);
        assert_eq!(sgl_to_dbl(-1), -0x1_0000);
    }

    #[test]
    fn single_precision_scale_uses_double_saturation_path() {
        assert_eq!(scale_value_sgl_saturate(0x4000, 1), MAXVAL_SGL);
        assert_eq!(scale_value_sgl_saturate(-0x4000, 2), MINVAL_SGL);

        let mut values = [0x1000, 0x4000, -0x4000];
        scale_values_sgl_saturate(&mut values, 2);
        assert_eq!(values, [0x4000, MAXVAL_SGL, MINVAL_SGL]);
    }

    #[test]
    fn pcm16_scaling_from_double_precision() {
        let src = [0x4000_0000, -0x4000_0000, 0x1000_0000];
        let mut dst = [0i16; 3];
        scale_values_pcm16_from_dbl(&mut dst, &src, 1);
        assert_eq!(dst, [MAXVAL_SGL, MINVAL_SGL, 0x2000]);
    }

    #[test]
    fn scalefactor_for_dbl_matches_headroom_expectations() {
        assert_eq!(get_scalefactor_dbl(&[0, 0]), 31);
        assert_eq!(get_scalefactor_dbl(&[0x4000_0000]), 0);
        assert_eq!(get_scalefactor_dbl(&[0x1000_0000]), 2);
        assert_eq!(get_scalefactor_dbl(&[-0x1000_0000]), 3);
    }

    #[test]
    fn scalefactor_for_pcm_stride() {
        let pcm = [1i16, 100, 2, 200, 3, 300];
        assert!(get_scalefactor_pcm_stride(&pcm, 3, 2) > 0);
        assert_eq!(get_scalefactor_pcm_stride(&[0, 0], 2, 1), 15);
    }

    #[test]
    fn q31_multiply_saturates_fractional_domain() {
        assert_eq!(mul_q31(0x4000_0000, 0x4000_0000), 0x2000_0000);
        assert_eq!(mul_q31(i32::MIN, i32::MIN), MAXVAL_DBL);
    }

    #[test]
    fn wrapping_scalar_and_vector_scaling_cover_all_directions() {
        assert_eq!(scale_value(3, 2), 12);
        assert_eq!(scale_value(-16, -2), -4);
        assert_eq!(scale_value(1, 99), i32::MIN);

        let mut unchanged = [1, -2];
        scale_values(&mut unchanged, 0);
        assert_eq!(unchanged, [1, -2]);
        scale_values(&mut unchanged, 2);
        assert_eq!(unchanged, [4, -8]);
        scale_values(&mut unchanged, -1);
        assert_eq!(unchanged, [2, -4]);

        scale_values_saturate(&mut unchanged, 0);
        assert_eq!(unchanged, [2, -4]);
    }

    #[test]
    fn copy_scalers_match_in_place_scalers() {
        let source = [0x4000_0000, -0x4000_0000];
        let mut double = [0; 2];
        scale_values_saturate_from(&mut double, &source, 1);
        assert_eq!(double, [MAXVAL_DBL, MINVAL_DBL_PLUS_ONE]);

        let source = [0x4000i16, -0x4000];
        let mut single = [0; 2];
        scale_values_sgl_saturate_from(&mut single, &source, 1);
        assert_eq!(single, [MAXVAL_SGL, MINVAL_SGL]);

        scale_values_sgl_saturate(&mut single, 0);
        assert_eq!(single, [MAXVAL_SGL, MINVAL_SGL]);
    }

    #[test]
    fn extreme_right_shift_and_saturating_arithmetic_are_bounded() {
        assert_eq!(scale_value_saturate(i32::MAX, -31), 0);
        assert_eq!(scale_value_saturate(i32::MIN, -99), 0);
        assert_eq!(saturating_add_dbl(i32::MAX, 1), MAXVAL_DBL);
        assert_eq!(
            saturating_add_dbl(MINVAL_DBL_PLUS_ONE, -1),
            MINVAL_DBL_PLUS_ONE
        );
        assert_eq!(saturating_sub_dbl(i32::MAX, -1), MAXVAL_DBL);
        assert_eq!(
            saturating_sub_dbl(MINVAL_DBL_PLUS_ONE, 1),
            MINVAL_DBL_PLUS_ONE
        );
        assert_eq!(saturating_add_dbl(7, -3), 4);
        assert_eq!(saturating_sub_dbl(7, 3), 4);
    }

    #[test]
    fn single_scalefactor_and_normalization_cover_zero_and_signs() {
        assert_eq!(get_scalefactor_sgl(&[0, 0]), 15);
        assert_eq!(get_scalefactor_sgl(&[0x1000]), 2);
        assert_eq!(get_scalefactor_sgl(&[-0x1000]), 3);
        assert_eq!(get_scalefactor_pcm_stride(&[], 0, 0), 31);
        assert_eq!(fixnormz_d(0), 32);
        assert_eq!(fixnormz_d(1), 31);
        assert_eq!(fixnormz_s(0), 16);
        assert_eq!(fixnormz_s(1), 15);
    }

    #[test]
    #[should_panic]
    fn double_copy_scaler_rejects_length_mismatch() {
        scale_values_saturate_from(&mut [0], &[1, 2], 0);
    }

    #[test]
    #[should_panic]
    fn single_copy_scaler_rejects_length_mismatch() {
        scale_values_sgl_saturate_from(&mut [0], &[1, 2], 0);
    }

    #[test]
    #[should_panic]
    fn pcm_copy_scaler_rejects_length_mismatch() {
        scale_values_pcm16_from_dbl(&mut [0], &[1, 2], 0);
    }

    #[test]
    #[should_panic]
    fn pcm_stride_rejects_zero_stride_for_nonempty_input() {
        get_scalefactor_pcm_stride(&[1], 1, 0);
    }

    #[test]
    #[should_panic]
    fn pcm_stride_rejects_an_out_of_bounds_span() {
        get_scalefactor_pcm_stride(&[1], 2, 1);
    }
}
