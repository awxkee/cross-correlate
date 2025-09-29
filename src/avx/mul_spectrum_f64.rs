/*
 * // Copyright (c) Radzivon Bartoshyk 5/2025. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
use crate::spectrum::SpectrumMultiplier;
use num_complex::Complex;
use std::arch::x86_64::*;

#[derive(Copy, Clone, Default)]
pub(crate) struct MulSpectrumDoubleAvxFma {}

impl SpectrumMultiplier<f64> for MulSpectrumDoubleAvxFma {
    fn mul_spectrum(&self, buffer: &mut [Complex<f64>], other: &[Complex<f64>], len: usize) {
        unsafe {
            mul_spectrum_in_place_f64_impl(buffer, other, len);
        }
    }
}

#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn avx_mul_complex(a: __m256d, b: __m256d) -> __m256d {
    // Swap real and imaginary parts of 'a' for FMA
    let a_yx = _mm256_permute_pd::<0b0101>(a); // [a_im, a_re, b_im, b_re]

    // Duplicate real and imaginary parts of 'b'
    let b_xx = _mm256_permute_pd::<0b0000>(b); // [c_re, c_re, d_re, d_re]
    let b_yy = _mm256_permute_pd::<0b1111>(b); // [c_im, c_im, d_im, d_im]

    // Compute (a_re*b_re - a_im*b_im) + i(a_re*b_im + a_im*b_re)
    _mm256_fmaddsub_pd(a, b_xx, _mm256_mul_pd(a_yx, b_yy))
}

#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn sse_fma_mul_complex(a: __m128d, b: __m128d) -> __m128d {
    let mut temp1 = _mm_unpacklo_pd(b, b);
    let mut temp2 = _mm_unpackhi_pd(b, b);
    temp1 = _mm_mul_pd(temp1, a);
    temp2 = _mm_mul_pd(temp2, a);
    temp2 = _mm_shuffle_pd(temp2, temp2, 0x01);
    _mm_addsub_pd(temp1, temp2)
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn mul_spectrum_in_place_f64_impl(
    value1: &mut [Complex<f64>],
    other: &[Complex<f64>],
    len: usize,
) {
    unsafe {
        let normalization_factor = 1f64 / len as f64;

        static CONJ_FACTORS: [f64; 4] = [0.0, -0.0, 0.0, -0.0];
        let conj_factors = _mm256_loadu_pd(CONJ_FACTORS.as_ptr());

        let v_norm_factor = _mm256_set1_pd(normalization_factor);
        let value1 = &mut value1[..];
        let other = &other;

        for (dst, kernel) in value1.chunks_exact_mut(8).zip(other.chunks_exact(8)) {
            let vd0 = _mm256_loadu_pd(dst.as_ptr().cast());
            let vd1 = _mm256_loadu_pd(dst.get_unchecked(2..).as_ptr().cast());
            let vd2 = _mm256_loadu_pd(dst.get_unchecked(4..).as_ptr().cast());
            let vd3 = _mm256_loadu_pd(dst.get_unchecked(6..).as_ptr().cast());

            let mut vk0 = _mm256_loadu_pd(kernel.as_ptr().cast());
            let mut vk1 = _mm256_loadu_pd(kernel.get_unchecked(2..).as_ptr().cast());
            let mut vk2 = _mm256_loadu_pd(kernel.get_unchecked(4..).as_ptr().cast());
            let mut vk3 = _mm256_loadu_pd(kernel.get_unchecked(6..).as_ptr().cast());

            vk0 = _mm256_xor_pd(vk0, conj_factors);
            vk1 = _mm256_xor_pd(vk1, conj_factors);
            vk2 = _mm256_xor_pd(vk2, conj_factors);
            vk3 = _mm256_xor_pd(vk3, conj_factors);

            let d0 = _mm256_mul_pd(avx_mul_complex(vd0, vk0), v_norm_factor);
            let d1 = _mm256_mul_pd(avx_mul_complex(vd1, vk1), v_norm_factor);
            let d2 = _mm256_mul_pd(avx_mul_complex(vd2, vk2), v_norm_factor);
            let d3 = _mm256_mul_pd(avx_mul_complex(vd3, vk3), v_norm_factor);

            _mm256_storeu_pd(dst.as_mut_ptr().cast(), d0);
            _mm256_storeu_pd(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), d1);
            _mm256_storeu_pd(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), d2);
            _mm256_storeu_pd(dst.get_unchecked_mut(6..).as_mut_ptr().cast(), d3);
        }

        let dst_rem = value1.chunks_exact_mut(8).into_remainder();
        let src_rem = other.chunks_exact(8).remainder();

        for (dst, kernel) in dst_rem.chunks_exact_mut(2).zip(src_rem.chunks_exact(2)) {
            let a0 = _mm256_loadu_pd(dst.as_ptr().cast());
            let mut b0 = _mm256_loadu_pd(kernel.as_ptr().cast());

            b0 = _mm256_xor_pd(b0, conj_factors);

            let d0 = _mm256_mul_pd(avx_mul_complex(a0, b0), v_norm_factor);

            _mm256_storeu_pd(dst.as_mut_ptr().cast(), d0);
        }

        let dst_rem = dst_rem.chunks_exact_mut(2).into_remainder();
        let src_rem = src_rem.chunks_exact(2).remainder();

        for (dst, kernel) in dst_rem.iter_mut().zip(src_rem.iter()) {
            let v0 = _mm_loadu_pd(dst as *const Complex<f64> as *const _);
            let mut v1 = _mm_loadu_pd(kernel as *const Complex<f64> as *const _);

            v1 = _mm_xor_pd(v1, _mm256_castpd256_pd128(conj_factors));

            let lo = _mm_mul_pd(
                sse_fma_mul_complex(v0, v1),
                _mm256_castpd256_pd128(v_norm_factor),
            );

            _mm_storeu_pd(dst as *mut Complex<f64> as *mut _, lo);
        }
    }
}
