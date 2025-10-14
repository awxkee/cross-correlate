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
pub(crate) struct MulSpectrumSingleSse4_2 {}

impl SpectrumMultiplier<f32> for MulSpectrumSingleSse4_2 {
    fn mul_spectrum(&self, buffer: &mut [Complex<f32>], other: &[Complex<f32>], len: usize) {
        unsafe {
            mul_spectrum_in_place_f32_impl(buffer, other, len);
        }
    }
}

#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn sse_mul_complex(a: __m128, b: __m128) -> __m128 {
    let mut temp1 = _mm_shuffle_ps::<0xA0>(b, b);
    let mut temp2 = _mm_shuffle_ps::<0xF5>(b, b);
    temp1 = _mm_mul_ps(temp1, a);
    temp2 = _mm_mul_ps(temp2, a);
    temp2 = _mm_shuffle_ps(temp2, temp2, 0xB1);
    _mm_addsub_ps(temp1, temp2)
}

#[target_feature(enable = "sse4.2")]
unsafe fn mul_spectrum_in_place_f32_impl(
    value1: &mut [Complex<f32>],
    other: &[Complex<f32>],
    len: usize,
) {
    unsafe {
        let normalization_factor = (1f64 / len as f64) as f32;

        static CONJ_FACTORS: [f32; 4] = [0.0, -0.0, 0.0, -0.0];
        let conj_factors = _mm_loadu_ps(CONJ_FACTORS.as_ptr());

        let v_norm_factor = _mm_set1_ps(normalization_factor);
        let value1 = &mut value1[..];
        let other = &other;

        for (dst, kernel) in value1.chunks_exact_mut(2).zip(other.chunks_exact(2)) {
            let a0 = _mm_loadu_ps(dst.as_ptr().cast());
            let mut b0 = _mm_loadu_ps(kernel.as_ptr().cast());

            b0 = _mm_xor_ps(b0, conj_factors);

            let d0 = _mm_mul_ps(sse_mul_complex(a0, b0), v_norm_factor);

            _mm_storeu_ps(dst.as_mut_ptr().cast(), d0);
        }

        let dst_rem = value1.chunks_exact_mut(2).into_remainder();
        let src_rem = other.chunks_exact(2).remainder();

        for (dst, kernel) in dst_rem.iter_mut().zip(src_rem.iter()) {
            let v0 = _mm_loadu_si64(dst as *const Complex<f32> as *const _);
            let mut v1 = _mm_loadu_si64(kernel as *const Complex<f32> as *const _);

            v1 = _mm_xor_si128(v1, _mm_castps_si128(conj_factors));

            let d0 = _mm_mul_ps(
                sse_mul_complex(_mm_castsi128_ps(v0), _mm_castsi128_ps(v1)),
                v_norm_factor,
            );

            _mm_storeu_si64(dst as *mut Complex<f32> as *mut _, _mm_castps_si128(d0));
        }
    }
}
