/*
 * // Copyright (c) Radzivon Bartoshyk 9/2025. All rights reserved.
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
use std::arch::aarch64::{
    vcmlaq_f64, vcmlaq_rot90_f64, vdupq_n_f64, veorq_u64, vld1q_f64, vmulq_f64,
    vreinterpretq_f64_u64, vreinterpretq_u64_f64, vst1q_f64,
};

#[derive(Copy, Clone, Default, Debug)]
pub(crate) struct SpectrumMulDoubleFcma {}

impl SpectrumMultiplier<f64> for SpectrumMulDoubleFcma {
    fn mul_spectrum(&self, buffer: &mut [Complex<f64>], other: &[Complex<f64>], len: usize) {
        unsafe { self.worker_impl(buffer, other, len) }
    }
}

impl SpectrumMulDoubleFcma {
    #[target_feature(enable = "fcma")]
    unsafe fn worker_impl(&self, buffer: &mut [Complex<f64>], other: &[Complex<f64>], len: usize) {
        unsafe {
            let normalization_factor = 1f64 / len as f64;
            let v_norm_factor = vdupq_n_f64(normalization_factor);

            static CONJ_FACTORS: [f64; 2] = [0.0, -0.0];
            let conj_factors = vreinterpretq_u64_f64(vld1q_f64(CONJ_FACTORS.as_ptr()));

            let source = &mut buffer[..];
            let other = &other;
            let zero = vdupq_n_f64(0.);

            for (dst, kernel) in source.chunks_exact_mut(4).zip(other.chunks_exact(4)) {
                let vd0 = vld1q_f64(dst.as_ptr().cast());
                let vd1 = vld1q_f64(dst.as_ptr().add(1).cast());
                let vd2 = vld1q_f64(dst.as_ptr().add(2).cast());
                let vd3 = vld1q_f64(dst.as_ptr().add(3).cast());

                let mut vk0 = vld1q_f64(kernel.as_ptr().cast());
                let mut vk1 = vld1q_f64(kernel.as_ptr().add(1).cast());
                let mut vk2 = vld1q_f64(kernel.as_ptr().add(2).cast());
                let mut vk3 = vld1q_f64(kernel.as_ptr().add(3).cast());

                vk0 = vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(vk0), conj_factors));
                vk1 = vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(vk1), conj_factors));
                vk2 = vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(vk2), conj_factors));
                vk3 = vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(vk3), conj_factors));

                let p0 = vmulq_f64(
                    vcmlaq_rot90_f64(vcmlaq_f64(zero, vd0, vk0), vd0, vk0),
                    v_norm_factor,
                );
                let p1 = vmulq_f64(
                    vcmlaq_rot90_f64(vcmlaq_f64(zero, vd1, vk1), vd1, vk1),
                    v_norm_factor,
                );
                let p2 = vmulq_f64(
                    vcmlaq_rot90_f64(vcmlaq_f64(zero, vd2, vk2), vd2, vk2),
                    v_norm_factor,
                );
                let p3 = vmulq_f64(
                    vcmlaq_rot90_f64(vcmlaq_f64(zero, vd3, vk3), vd3, vk3),
                    v_norm_factor,
                );

                vst1q_f64(dst.as_mut_ptr().cast(), p0);
                vst1q_f64(dst.get_unchecked_mut(1..).as_mut_ptr().cast(), p1);
                vst1q_f64(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), p2);
                vst1q_f64(dst.get_unchecked_mut(3..).as_mut_ptr().cast(), p3);
            }

            let dst_rem = source.chunks_exact_mut(4).into_remainder();
            let src_rem = other.chunks_exact(4).remainder();

            for (dst, kernel) in dst_rem.iter_mut().zip(src_rem.iter()) {
                let v0 = vld1q_f64(dst as *const Complex<f64> as *const f64);
                let mut v1 = vld1q_f64(kernel as *const Complex<f64> as *const f64);

                v1 = vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(v1), conj_factors));

                let p0 = vcmlaq_rot90_f64(vcmlaq_f64(zero, v0, v1), v0, v1);
                let p1 = vmulq_f64(p0, v_norm_factor);
                vst1q_f64(dst as *mut Complex<f64> as *mut f64, p1);
            }
        }
    }
}
