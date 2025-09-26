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
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};

pub(crate) trait SpectrumMultiplier<V> {
    fn mul_spectrum(&self, buffer: &mut [Complex<V>], other: &[Complex<V>], len: usize);
}

#[derive(Copy, Clone, Default, Debug)]
#[allow(dead_code)]
pub(crate) struct SpectrumMultiplierSingle {}

#[derive(Copy, Clone, Default, Debug)]
#[allow(dead_code)]
pub(crate) struct SpectrumMultiplierDouble {}

impl SpectrumMultiplier<f32> for SpectrumMultiplierSingle {
    fn mul_spectrum(&self, buffer: &mut [Complex<f32>], other: &[Complex<f32>], len: usize) {
        mul_spectrum_in_place_impl(buffer, other, len);
    }
}

impl SpectrumMultiplier<f64> for SpectrumMultiplierDouble {
    fn mul_spectrum(&self, buffer: &mut [Complex<f64>], other: &[Complex<f64>], len: usize) {
        mul_spectrum_in_place_impl(buffer, other, len);
    }
}

#[inline(always)]
fn mul_spectrum_in_place_impl<V: Copy + 'static + Float>(
    value1: &mut [Complex<V>],
    other: &[Complex<V>],
    len: usize,
) where
    f64: AsPrimitive<V>,
{
    let normalization_factor = (1f64 / len as f64).as_();
    for (dst, kernel) in value1.iter_mut().zip(other.iter()) {
        *dst = (*dst) * kernel.conj() * normalization_factor;
    }
}
