//! Bit-exact fixed-point FFT kernels used by the FDK-compatible CLDFB path.

#[allow(unused_assignments)]
pub(crate) fn fft32_radix4_stage1(values: &mut [i32; 64]) {
    let (mut vi, mut ui, mut vi2, mut ui2, mut vi3, mut ui3) = (0i32, 0i32, 0i32, 0i32, 0i32, 0i32);
    let (mut vr, mut ur, mut vr2, mut ur2, mut vr3, mut ur3, mut vr4, mut ur4) =
        (0i32, 0i32, 0i32, 0i32, 0i32, 0i32, 0i32, 0i32);
    // i = 0
    vr = (values[0] + values[32]) >> 1; /* Re A + Re B */
    ur = (values[1] + values[33]) >> 1; /* Im A + Im B */
    vi = (values[16] + values[48]) >> 1; /* Re C + Re D */
    ui = (values[17] + values[49]) >> 1; /* Im C + Im D */

    values[0] = vr + (vi); /* Re A' = ReA + ReB +ReC + ReD */
    values[1] = ur + (ui); /* Im A' = sum of imag values */

    vr2 = (values[4] + values[36]) >> 1; /* Re A + Re B */
    ur2 = (values[5] + values[37]) >> 1; /* Im A + Im B */

    values[4] = vr - (vi); /* Re C' = -(ReC+ReD) + (ReA+ReB) */
    values[5] = ur - (ui); /* Im C' = -Im C -Im D +Im A +Im B */

    vr -= values[32]; /* Re A - Re B */
    ur -= values[33]; /* Im A - Im B */
    vi = (vi) - values[48]; /* Re C - Re D */
    ui = (ui) - values[49]; /* Im C - Im D */

    vr3 = (values[2] + values[34]) >> 1; /* Re A + Re B */
    ur3 = (values[3] + values[35]) >> 1; /* Im A + Im B */

    values[2] = ui + vr; /* Re B' = Im C - Im D  + Re A - Re B */
    values[3] = ur - vi; /* Im B'= -Re C + Re D + Im A - Im B */

    vr4 = (values[6] + values[38]) >> 1; /* Re A + Re B */
    ur4 = (values[7] + values[39]) >> 1; /* Im A + Im B */

    values[6] = vr - ui; /* Re D' = -Im C + Im D + Re A - Re B */
    values[7] = vi + ur; /* Im D'= Re C - Re D + Im A - Im B */

    // i=16
    vi = (values[20] + values[52]) >> 1; /* Re C + Re D */
    ui = (values[21] + values[53]) >> 1; /* Im C + Im D */

    values[16] = vr2 + (vi); /* Re A' = ReA + ReB +ReC + ReD */
    values[17] = ur2 + (ui); /* Im A' = sum of imag values */
    values[20] = vr2 - (vi); /* Re C' = -(ReC+ReD) + (ReA+ReB) */
    values[21] = ur2 - (ui); /* Im C' = -Im C -Im D +Im A +Im B */

    vr2 -= values[36]; /* Re A - Re B */
    ur2 -= values[37]; /* Im A - Im B */
    vi = (vi) - values[52]; /* Re C - Re D */
    ui = (ui) - values[53]; /* Im C - Im D */

    vi2 = (values[18] + values[50]) >> 1; /* Re C + Re D */
    ui2 = (values[19] + values[51]) >> 1; /* Im C + Im D */

    values[18] = ui + vr2; /* Re B' = Im C - Im D  + Re A - Re B */
    values[19] = ur2 - vi; /* Im B'= -Re C + Re D + Im A - Im B */

    vi3 = (values[22] + values[54]) >> 1; /* Re C + Re D */
    ui3 = (values[23] + values[55]) >> 1; /* Im C + Im D */

    values[22] = vr2 - ui; /* Re D' = -Im C + Im D + Re A - Re B */
    values[23] = vi + ur2; /* Im D'= Re C - Re D + Im A - Im B */

    // i = 32

    values[32] = vr3 + (vi2); /* Re A' = ReA + ReB +ReC + ReD */
    values[33] = ur3 + (ui2); /* Im A' = sum of imag values */
    values[36] = vr3 - (vi2); /* Re C' = -(ReC+ReD) + (ReA+ReB) */
    values[37] = ur3 - (ui2); /* Im C' = -Im C -Im D +Im A +Im B */

    vr3 -= values[34]; /* Re A - Re B */
    ur3 -= values[35]; /* Im A - Im B */
    vi2 = (vi2) - values[50]; /* Re C - Re D */
    ui2 = (ui2) - values[51]; /* Im C - Im D */

    values[34] = ui2 + vr3; /* Re B' = Im C - Im D  + Re A - Re B */
    values[35] = ur3 - vi2; /* Im B'= -Re C + Re D + Im A - Im B */

    // i=48

    values[48] = vr4 + (vi3); /* Re A' = ReA + ReB +ReC + ReD */
    values[52] = vr4 - (vi3); /* Re C' = -(ReC+ReD) + (ReA+ReB) */
    values[49] = ur4 + (ui3); /* Im A' = sum of imag values */
    values[53] = ur4 - (ui3); /* Im C' = -Im C -Im D +Im A +Im B */

    vr4 -= values[38]; /* Re A - Re B */
    ur4 -= values[39]; /* Im A - Im B */

    values[38] = vr3 - ui2; /* Re D' = -Im C + Im D + Re A - Re B */
    values[39] = vi2 + ur3; /* Im D'= Re C - Re D + Im A - Im B */

    vi3 = (vi3) - values[54]; /* Re C - Re D */
    ui3 = (ui3) - values[55]; /* Im C - Im D */

    values[50] = ui3 + vr4; /* Re B' = Im C - Im D  + Re A - Re B */
    values[54] = vr4 - ui3; /* Re D' = -Im C + Im D + Re A - Re B */
    values[51] = ur4 - vi3; /* Im B'= -Re C + Re D + Im A - Im B */
    values[55] = vi3 + ur4; /* Im D'= Re C - Re D + Im A - Im B */

    // i=8
    vr = (values[8] + values[40]) >> 1; /* Re A + Re B */
    ur = (values[9] + values[41]) >> 1; /* Im A + Im B */
    vi = (values[24] + values[56]) >> 1; /* Re C + Re D */
    ui = (values[25] + values[57]) >> 1; /* Im C + Im D */

    values[8] = vr + (vi); /* Re A' = ReA + ReB +ReC + ReD */
    values[9] = ur + (ui); /* Im A' = sum of imag values */

    vr2 = (values[12] + values[44]) >> 1; /* Re A + Re B */
    ur2 = (values[13] + values[45]) >> 1; /* Im A + Im B */

    values[12] = vr - (vi); /* Re C' = -(ReC+ReD) + (ReA+ReB) */
    values[13] = ur - (ui); /* Im C' = -Im C -Im D +Im A +Im B */

    vr -= values[40]; /* Re A - Re B */
    ur -= values[41]; /* Im A - Im B */
    vi = (vi) - values[56]; /* Re C - Re D */
    ui = (ui) - values[57]; /* Im C - Im D */

    vr3 = (values[10] + values[42]) >> 1; /* Re A + Re B */
    ur3 = (values[11] + values[43]) >> 1; /* Im A + Im B */

    values[10] = ui + vr; /* Re B' = Im C - Im D  + Re A - Re B */
    values[11] = ur - vi; /* Im B'= -Re C + Re D + Im A - Im B */

    vr4 = (values[14] + values[46]) >> 1; /* Re A + Re B */
    ur4 = (values[15] + values[47]) >> 1; /* Im A + Im B */

    values[14] = vr - ui; /* Re D' = -Im C + Im D + Re A - Re B */
    values[15] = vi + ur; /* Im D'= Re C - Re D + Im A - Im B */

    // i=24
    vi = (values[28] + values[60]) >> 1; /* Re C + Re D */
    ui = (values[29] + values[61]) >> 1; /* Im C + Im D */

    values[24] = vr2 + (vi); /* Re A' = ReA + ReB +ReC + ReD */
    values[28] = vr2 - (vi); /* Re C' = -(ReC+ReD) + (ReA+ReB) */
    values[25] = ur2 + (ui); /* Im A' = sum of imag values */
    values[29] = ur2 - (ui); /* Im C' = -Im C -Im D +Im A +Im B */

    vr2 -= values[44]; /* Re A - Re B */
    ur2 -= values[45]; /* Im A - Im B */
    vi = (vi) - values[60]; /* Re C - Re D */
    ui = (ui) - values[61]; /* Im C - Im D */

    vi2 = (values[26] + values[58]) >> 1; /* Re C + Re D */
    ui2 = (values[27] + values[59]) >> 1; /* Im C + Im D */

    values[26] = ui + vr2; /* Re B' = Im C - Im D  + Re A - Re B */
    values[27] = ur2 - vi; /* Im B'= -Re C + Re D + Im A - Im B */

    vi3 = (values[30] + values[62]) >> 1; /* Re C + Re D */
    ui3 = (values[31] + values[63]) >> 1; /* Im C + Im D */

    values[30] = vr2 - ui; /* Re D' = -Im C + Im D + Re A - Re B */
    values[31] = vi + ur2; /* Im D'= Re C - Re D + Im A - Im B */

    // i=40

    values[40] = vr3 + (vi2); /* Re A' = ReA + ReB +ReC + ReD */
    values[44] = vr3 - (vi2); /* Re C' = -(ReC+ReD) + (ReA+ReB) */
    values[41] = ur3 + (ui2); /* Im A' = sum of imag values */
    values[45] = ur3 - (ui2); /* Im C' = -Im C -Im D +Im A +Im B */

    vr3 -= values[42]; /* Re A - Re B */
    ur3 -= values[43]; /* Im A - Im B */
    vi2 = (vi2) - values[58]; /* Re C - Re D */
    ui2 = (ui2) - values[59]; /* Im C - Im D */

    values[42] = ui2 + vr3; /* Re B' = Im C - Im D  + Re A - Re B */
    values[43] = ur3 - vi2; /* Im B'= -Re C + Re D + Im A - Im B */

    // i=56

    values[56] = vr4 + (vi3); /* Re A' = ReA + ReB +ReC + ReD */
    values[60] = vr4 - (vi3); /* Re C' = -(ReC+ReD) + (ReA+ReB) */
    values[57] = ur4 + (ui3); /* Im A' = sum of imag values */
    values[61] = ur4 - (ui3); /* Im C' = -Im C -Im D +Im A +Im B */

    vr4 -= values[46]; /* Re A - Re B */
    ur4 -= values[47]; /* Im A - Im B */

    values[46] = vr3 - ui2; /* Re D' = -Im C + Im D + Re A - Re B */
    values[47] = vi2 + ur3; /* Im D'= Re C - Re D + Im A - Im B */

    vi3 = (vi3) - values[62]; /* Re C - Re D */
    ui3 = (ui3) - values[63]; /* Im C - Im D */

    values[58] = ui3 + vr4; /* Re B' = Im C - Im D  + Re A - Re B */
    values[62] = vr4 - ui3; /* Re D' = -Im C + Im D + Re A - Re B */
    values[59] = ur4 - vi3; /* Im B'= -Re C + Re D + Im A - Im B */
    values[63] = vi3 + ur4; /* Im D'= Re C - Re D + Im A - Im B */
}

pub(crate) fn fft32_radix4_stage2(values: &mut [i32; 64]) {
    const PI_FOURTH_Q15: i32 = 0x5a82;
    let mul_div2 = |left: i32, right: i32| ((left as i64 * right as i64) >> 16) as i32;
    for base in (0..64).step_by(16) {
        let mut vr = values[base + 8];
        let mut vi = values[base + 9];
        let mut ur = values[base] >> 1;
        let mut ui = values[base + 1] >> 1;
        values[base] = ur + (vr >> 1);
        values[base + 1] = ui + (vi >> 1);
        values[base + 8] = ur - (vr >> 1);
        values[base + 9] = ui - (vi >> 1);

        vr = values[base + 13];
        vi = values[base + 12];
        ur = values[base + 4] >> 1;
        ui = values[base + 5] >> 1;
        values[base + 4] = ur + (vr >> 1);
        values[base + 5] = ui - (vi >> 1);
        values[base + 12] = ur - (vr >> 1);
        values[base + 13] = ui + (vi >> 1);

        let wa = mul_div2(values[base + 10], PI_FOURTH_Q15);
        let wb = mul_div2(values[base + 11], PI_FOURTH_Q15);
        vi = wb - wa;
        vr = wb + wa;
        ur = values[base + 2];
        ui = values[base + 3];
        values[base + 2] = (ur >> 1) + vr;
        values[base + 3] = (ui >> 1) + vi;
        values[base + 10] = (ur >> 1) - vr;
        values[base + 11] = (ui >> 1) - vi;

        let wa = mul_div2(values[base + 14], PI_FOURTH_Q15);
        let wb = mul_div2(values[base + 15], PI_FOURTH_Q15);
        vr = wb - wa;
        vi = wb + wa;
        ur = values[base + 6];
        ui = values[base + 7];
        values[base + 6] = (ur >> 1) + vr;
        values[base + 7] = (ui >> 1) - vi;
        values[base + 14] = (ur >> 1) - vr;
        values[base + 15] = (ui >> 1) + vi;
    }
}

#[allow(unused_assignments)]
pub(crate) fn fft32_radix4_stage3(values: &mut [i32; 64]) {
    const TWIDDLES: [(i32, i32); 6] = [
        (0x7642, 0x30fc),
        (0x30fc, 0x7642),
        (0x7d8a, 0x18f9),
        (0x6a6e, 0x471d),
        (0x471d, 0x6a6e),
        (0x18f9, 0x7d8a),
    ];
    const PI_FOURTH_Q15: i32 = 0x5a82;
    let mul_div2 = |left: i32, right: i32| ((left as i64 * right as i64) >> 16) as i32;
    let cplx_mult_div2 = |ar: i32, ai: i32, (br, bi): (i32, i32)| {
        (
            mul_div2(ar, br) - mul_div2(ai, bi),
            mul_div2(ar, bi) + mul_div2(ai, br),
        )
    };
    let sumdiff_pi_fourth = |a: i32, b: i32| {
        let wa = mul_div2(a, PI_FOURTH_Q15);
        let wb = mul_div2(b, PI_FOURTH_Q15);
        (wb - wa, wb + wa)
    };
    let (mut vi, mut ui, mut vr, mut ur) = (0i32, 0i32, 0i32, 0i32);
    vr = values[16];
    vi = values[17];
    ur = values[0] >> 1;
    ui = values[1] >> 1;
    values[0] = ur + (vr >> 1);
    values[1] = ui + (vi >> 1);
    values[16] = ur - (vr >> 1);
    values[17] = ui - (vi >> 1);

    vi = values[24];
    vr = values[25];
    ur = values[8] >> 1;
    ui = values[9] >> 1;
    values[8] = ur + (vr >> 1);
    values[9] = ui - (vi >> 1);
    values[24] = ur - (vr >> 1);
    values[25] = ui + (vi >> 1);

    vr = values[48];
    vi = values[49];
    ur = values[32] >> 1;
    ui = values[33] >> 1;
    values[32] = ur + (vr >> 1);
    values[33] = ui + (vi >> 1);
    values[48] = ur - (vr >> 1);
    values[49] = ui - (vi >> 1);

    vi = values[56];
    vr = values[57];
    ur = values[40] >> 1;
    ui = values[41] >> 1;
    values[40] = ur + (vr >> 1);
    values[41] = ui - (vi >> 1);
    values[56] = ur - (vr >> 1);
    values[57] = ui + (vi >> 1);

    (vi, vr) = cplx_mult_div2(values[19], values[18], TWIDDLES[0]);
    ur = values[2];
    ui = values[3];
    values[2] = (ur >> 1) + vr;
    values[3] = (ui >> 1) + vi;
    values[18] = (ur >> 1) - vr;
    values[19] = (ui >> 1) - vi;

    (vr, vi) = cplx_mult_div2(values[27], values[26], TWIDDLES[0]);
    ur = values[10];
    ui = values[11];
    values[10] = (ur >> 1) + vr;
    values[11] = (ui >> 1) - vi;
    values[26] = (ur >> 1) - vr;
    values[27] = (ui >> 1) + vi;

    (vi, vr) = cplx_mult_div2(values[51], values[50], TWIDDLES[0]);
    ur = values[34];
    ui = values[35];
    values[34] = (ur >> 1) + vr;
    values[35] = (ui >> 1) + vi;
    values[50] = (ur >> 1) - vr;
    values[51] = (ui >> 1) - vi;

    (vr, vi) = cplx_mult_div2(values[59], values[58], TWIDDLES[0]);
    ur = values[42];
    ui = values[43];
    values[42] = (ur >> 1) + vr;
    values[43] = (ui >> 1) - vi;
    values[58] = (ur >> 1) - vr;
    values[59] = (ui >> 1) + vi;

    (vi, vr) = sumdiff_pi_fourth(values[20], values[21]);
    ur = values[4];
    ui = values[5];
    values[4] = (ur >> 1) + vr;
    values[5] = (ui >> 1) + vi;
    values[20] = (ur >> 1) - vr;
    values[21] = (ui >> 1) - vi;

    (vr, vi) = sumdiff_pi_fourth(values[28], values[29]);
    ur = values[12];
    ui = values[13];
    values[12] = (ur >> 1) + vr;
    values[13] = (ui >> 1) - vi;
    values[28] = (ur >> 1) - vr;
    values[29] = (ui >> 1) + vi;

    (vi, vr) = sumdiff_pi_fourth(values[52], values[53]);
    ur = values[36];
    ui = values[37];
    values[36] = (ur >> 1) + vr;
    values[37] = (ui >> 1) + vi;
    values[52] = (ur >> 1) - vr;
    values[53] = (ui >> 1) - vi;

    (vr, vi) = sumdiff_pi_fourth(values[60], values[61]);
    ur = values[44];
    ui = values[45];
    values[44] = (ur >> 1) + vr;
    values[45] = (ui >> 1) - vi;
    values[60] = (ur >> 1) - vr;
    values[61] = (ui >> 1) + vi;

    (vi, vr) = cplx_mult_div2(values[23], values[22], TWIDDLES[1]);
    ur = values[6];
    ui = values[7];
    values[6] = (ur >> 1) + vr;
    values[7] = (ui >> 1) + vi;
    values[22] = (ur >> 1) - vr;
    values[23] = (ui >> 1) - vi;

    (vr, vi) = cplx_mult_div2(values[31], values[30], TWIDDLES[1]);
    ur = values[14];
    ui = values[15];
    values[14] = (ur >> 1) + vr;
    values[15] = (ui >> 1) - vi;
    values[30] = (ur >> 1) - vr;
    values[31] = (ui >> 1) + vi;

    (vi, vr) = cplx_mult_div2(values[55], values[54], TWIDDLES[1]);
    ur = values[38];
    ui = values[39];
    values[38] = (ur >> 1) + vr;
    values[39] = (ui >> 1) + vi;
    values[54] = (ur >> 1) - vr;
    values[55] = (ui >> 1) - vi;

    (vr, vi) = cplx_mult_div2(values[63], values[62], TWIDDLES[1]);
    ur = values[46];
    ui = values[47];

    values[46] = (ur >> 1) + vr;
    values[47] = (ui >> 1) - vi;
    values[62] = (ur >> 1) - vr;
    values[63] = (ui >> 1) + vi;

    vr = values[32];
    vi = values[33];
    ur = values[0] >> 1;
    ui = values[1] >> 1;
    values[0] = ur + (vr >> 1);
    values[1] = ui + (vi >> 1);
    values[32] = ur - (vr >> 1);
    values[33] = ui - (vi >> 1);

    vi = values[48];
    vr = values[49];
    ur = values[16] >> 1;
    ui = values[17] >> 1;
    values[16] = ur + (vr >> 1);
    values[17] = ui - (vi >> 1);
    values[48] = ur - (vr >> 1);
    values[49] = ui + (vi >> 1);

    (vi, vr) = cplx_mult_div2(values[35], values[34], TWIDDLES[2]);
    ur = values[2];
    ui = values[3];
    values[2] = (ur >> 1) + vr;
    values[3] = (ui >> 1) + vi;
    values[34] = (ur >> 1) - vr;
    values[35] = (ui >> 1) - vi;

    (vr, vi) = cplx_mult_div2(values[51], values[50], TWIDDLES[2]);
    ur = values[18];
    ui = values[19];
    values[18] = (ur >> 1) + vr;
    values[19] = (ui >> 1) - vi;
    values[50] = (ur >> 1) - vr;
    values[51] = (ui >> 1) + vi;

    (vi, vr) = cplx_mult_div2(values[37], values[36], TWIDDLES[0]);
    ur = values[4];
    ui = values[5];
    values[4] = (ur >> 1) + vr;
    values[5] = (ui >> 1) + vi;
    values[36] = (ur >> 1) - vr;
    values[37] = (ui >> 1) - vi;

    (vr, vi) = cplx_mult_div2(values[53], values[52], TWIDDLES[0]);
    ur = values[20];
    ui = values[21];
    values[20] = (ur >> 1) + vr;
    values[21] = (ui >> 1) - vi;
    values[52] = (ur >> 1) - vr;
    values[53] = (ui >> 1) + vi;

    (vi, vr) = cplx_mult_div2(values[39], values[38], TWIDDLES[3]);
    ur = values[6];
    ui = values[7];
    values[6] = (ur >> 1) + vr;
    values[7] = (ui >> 1) + vi;
    values[38] = (ur >> 1) - vr;
    values[39] = (ui >> 1) - vi;

    (vr, vi) = cplx_mult_div2(values[55], values[54], TWIDDLES[3]);
    ur = values[22];
    ui = values[23];
    values[22] = (ur >> 1) + vr;
    values[23] = (ui >> 1) - vi;
    values[54] = (ur >> 1) - vr;
    values[55] = (ui >> 1) + vi;

    (vi, vr) = sumdiff_pi_fourth(values[40], values[41]);
    ur = values[8];
    ui = values[9];
    values[8] = (ur >> 1) + vr;
    values[9] = (ui >> 1) + vi;
    values[40] = (ur >> 1) - vr;
    values[41] = (ui >> 1) - vi;

    (vr, vi) = sumdiff_pi_fourth(values[56], values[57]);
    ur = values[24];
    ui = values[25];
    values[24] = (ur >> 1) + vr;
    values[25] = (ui >> 1) - vi;
    values[56] = (ur >> 1) - vr;
    values[57] = (ui >> 1) + vi;

    (vi, vr) = cplx_mult_div2(values[43], values[42], TWIDDLES[4]);
    ur = values[10];
    ui = values[11];

    values[10] = (ur >> 1) + vr;
    values[11] = (ui >> 1) + vi;
    values[42] = (ur >> 1) - vr;
    values[43] = (ui >> 1) - vi;

    (vr, vi) = cplx_mult_div2(values[59], values[58], TWIDDLES[4]);
    ur = values[26];
    ui = values[27];
    values[26] = (ur >> 1) + vr;
    values[27] = (ui >> 1) - vi;
    values[58] = (ur >> 1) - vr;
    values[59] = (ui >> 1) + vi;

    (vi, vr) = cplx_mult_div2(values[45], values[44], TWIDDLES[1]);
    ur = values[12];
    ui = values[13];
    values[12] = (ur >> 1) + vr;
    values[13] = (ui >> 1) + vi;
    values[44] = (ur >> 1) - vr;
    values[45] = (ui >> 1) - vi;

    (vr, vi) = cplx_mult_div2(values[61], values[60], TWIDDLES[1]);
    ur = values[28];
    ui = values[29];
    values[28] = (ur >> 1) + vr;
    values[29] = (ui >> 1) - vi;
    values[60] = (ur >> 1) - vr;
    values[61] = (ui >> 1) + vi;

    (vi, vr) = cplx_mult_div2(values[47], values[46], TWIDDLES[5]);
    ur = values[14];
    ui = values[15];
    values[14] = (ur >> 1) + vr;
    values[15] = (ui >> 1) + vi;
    values[46] = (ur >> 1) - vr;
    values[47] = (ui >> 1) - vi;

    (vr, vi) = cplx_mult_div2(values[63], values[62], TWIDDLES[5]);
    ur = values[30];
    ui = values[31];
    values[30] = (ur >> 1) + vr;
    values[31] = (ui >> 1) - vi;
    values[62] = (ur >> 1) - vr;
    values[63] = (ui >> 1) + vi;
}

pub(crate) fn fixed_fft32(values: &mut [i32; 64]) {
    fft32_radix4_stage1(values);
    fft32_radix4_stage2(values);
    fft32_radix4_stage3(values);
}

const DCT64_PRE_TWIDDLES_Q31: [(u32, u32); 32] = [
    (0x7ffd885a, 0x01921d20),
    (0x7fe9cbc0, 0x04b6195d),
    (0x7fc25596, 0x07d95b9e),
    (0x7f872bf3, 0x0afb6805),
    (0x7f3857f6, 0x0e1bc2e4),
    (0x7ed5e5c6, 0x1139f0cf),
    (0x7e5fe493, 0x145576b1),
    (0x7dd6668f, 0x176dd9de),
    (0x7d3980ec, 0x1a82a026),
    (0x7c894bde, 0x1d934fe5),
    (0x7bc5e290, 0x209f701c),
    (0x7aef6323, 0x23a6887f),
    (0x7a05eead, 0x26a82186),
    (0x7909a92d, 0x29a3c485),
    (0x77fab989, 0x2c98fbba),
    (0x76d94989, 0x2f875262),
    (0x75a585cf, 0x326e54c7),
    (0x745f9dd1, 0x354d9057),
    (0x7307c3d0, 0x382493b0),
    (0x719e2cd2, 0x3af2eeb7),
    (0x7023109a, 0x3db832a6),
    (0x6e96a99d, 0x4073f21d),
    (0x6cf934fc, 0x4325c135),
    (0x6b4af279, 0x45cd358f),
    (0x698c246c, 0x4869e665),
    (0x67bd0fbd, 0x4afb6c98),
    (0x65ddfbd3, 0x4d8162c4),
    (0x63ef3290, 0x4ffb654d),
    (0x61f1003f, 0x5269126e),
    (0x5fe3b38d, 0x54ca0a4b),
    (0x5dc79d7c, 0x571deefa),
    (0x5b9d1154, 0x59646498),
];

const DCT64_POST_TWIDDLES_Q31: [(u32, u32); 15] = [
    (0x7fd8878e, 0x0647d97c),
    (0x7f62368f, 0x0c8bd35e),
    (0x7e9d55fc, 0x12c8106f),
    (0x7d8a5f40, 0x18f8b83c),
    (0x7c29fbee, 0x1f19f97b),
    (0x7a7d055b, 0x25280c5e),
    (0x78848414, 0x2b1f34eb),
    (0x7641af3d, 0x30fbc54d),
    (0x73b5ebd1, 0x36ba2014),
    (0x70e2cbc6, 0x3c56ba70),
    (0x6dca0d14, 0x41ce1e65),
    (0x6a6d98a4, 0x471cece7),
    (0x66cf8120, 0x4c3fdff4),
    (0x62f201ac, 0x5133cc94),
    (0x5ed77c8a, 0x55f5a4d2),
];

fn q31_rom_to_q15(value: u32) -> i32 {
    (((u64::from(value) + 0x8000) >> 16).min(0x7fff)) as i32
}

fn complex_mul_div2_q15(ar: i32, ai: i32, twiddle: (u32, u32)) -> (i32, i32) {
    let br = q31_rom_to_q15(twiddle.0);
    let bi = q31_rom_to_q15(twiddle.1);
    let mul = |left: i32, right: i32| ((i64::from(left) * i64::from(right)) >> 16) as i32;
    (mul(ar, br) - mul(ai, bi), mul(ar, bi) + mul(ai, br))
}

fn complex_mul_q15(ar: i32, ai: i32, twiddle: (u32, u32)) -> (i32, i32) {
    let (real, imaginary) = complex_mul_div2_q15(ar, ai, twiddle);
    (real.wrapping_shl(1), imaginary.wrapping_shl(1))
}

pub(crate) fn fixed_dct_iv_64(values: &mut [i32; 64]) -> i32 {
    let mut left = 0usize;
    let mut right = 62usize;
    for i in (0..31).step_by(2) {
        let (accu1, accu2) =
            complex_mul_div2_q15(values[right + 1], values[left], DCT64_PRE_TWIDDLES_Q31[i]);
        let (accu3, accu4) = complex_mul_div2_q15(
            values[right],
            values[left + 1],
            DCT64_PRE_TWIDDLES_Q31[i + 1],
        );
        values[left] = accu2 >> 1;
        values[left + 1] = accu1 >> 1;
        values[right] = accu4 >> 1;
        values[right + 1] = -(accu3 >> 1);
        left += 2;
        right -= 2;
    }

    fixed_fft32(values);

    left = 0;
    right = 62;
    let mut accu1 = values[right];
    let mut accu2 = values[right + 1];
    values[right + 1] = -values[left + 1];
    for &twiddle in &DCT64_POST_TWIDDLES_Q31 {
        let (accu3, accu4) = complex_mul_q15(accu1, accu2, twiddle);
        values[left + 1] = accu3;
        values[right] = accu4;
        left += 2;
        right -= 2;
        let (accu3, accu4) = complex_mul_q15(values[left + 1], values[left], twiddle);
        accu1 = values[right];
        accu2 = values[right + 1];
        values[right + 1] = -accu3;
        values[left] = accu4;
    }
    let mul_half =
        |value: i32| (((i64::from(value) * i64::from(0x5a82)) >> 16) as i32).wrapping_shl(1);
    accu1 = mul_half(accu1);
    accu2 = mul_half(accu2);
    values[right] = accu1 + accu2;
    values[left + 1] = accu1 - accu2;
    6
}

pub(crate) fn fixed_dst_iv_64(values: &mut [i32; 64]) -> i32 {
    let mut left = 0usize;
    let mut right = 62usize;
    for i in (0..31).step_by(2) {
        let (accu1, accu2) = complex_mul_div2_q15(
            values[right + 1] >> 1,
            -(values[left] >> 1),
            DCT64_PRE_TWIDDLES_Q31[i],
        );
        let (accu3, accu4) = complex_mul_div2_q15(
            -(values[right] >> 1),
            values[left + 1] >> 1,
            DCT64_PRE_TWIDDLES_Q31[i + 1],
        );
        values[left] = accu2;
        values[left + 1] = accu1;
        values[right] = accu4;
        values[right + 1] = -accu3;
        left += 2;
        right -= 2;
    }

    fixed_fft32(values);

    left = 0;
    right = 62;
    let mut accu1 = values[right];
    let mut accu2 = values[right + 1];
    values[right + 1] = -values[left];
    values[left] = values[left + 1];
    for &twiddle in &DCT64_POST_TWIDDLES_Q31 {
        let (accu3, accu4) = complex_mul_q15(accu1, accu2, twiddle);
        values[right] = -accu3;
        values[left + 1] = -accu4;
        left += 2;
        right -= 2;
        let (accu3, accu4) = complex_mul_q15(values[left + 1], values[left], twiddle);
        accu1 = values[right];
        accu2 = values[right + 1];
        values[left] = accu3;
        values[right + 1] = -accu4;
    }
    let mul_half =
        |value: i32| (((i64::from(value) * i64::from(0x5a82)) >> 16) as i32).wrapping_shl(1);
    accu1 = mul_half(accu1);
    accu2 = mul_half(accu2);
    values[left + 1] = -accu1 - accu2;
    values[right] = accu2 - accu1;
    6
}
