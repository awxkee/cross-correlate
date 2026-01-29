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
#![deny(unreachable_pub)]
#![deny(
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::print_literal,
    clippy::print_in_format_impl
)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(
    all(feature = "fcma", target_arch = "aarch64"),
    feature(stdarch_neon_fcma)
)]

use std::fmt::Debug;

pub(crate) trait CorrelateSample: Copy + 'static + Clone + Default + Debug {}

impl CorrelateSample for f32 {}
impl CorrelateSample for f64 {}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
mod avx;
mod correlate_complex;
mod cross_correlate;
mod error;
mod fast_divider;
mod mode;
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
mod neon;
mod pad;
mod real;
mod spectrum;
#[cfg(all(target_arch = "x86_64", feature = "sse"))]
mod sse;

pub use cross_correlate::{Correlate, CrossCorrelate};
pub use error::CrossCorrelateError;
pub use mode::CrossCorrelationMode;

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_f64() {
        static VALID: [f64; 5] = [
            9.13584942,
            6.299764045999998,
            4.989608066999999,
            4.388719885199999,
            5.8635182073,
        ];
        static SAME: [f64; 8] = [
            4.0952056,
            7.0888835,
            9.13584942,
            6.299764045999998,
            4.989608066999999,
            4.388719885199999,
            5.8635182073,
            6.432118037299999,
        ];
        static FULL: [f64; 11] = [
            0.6195199999999994,
            4.0952056,
            7.0888835,
            9.13584942,
            6.299764045999998,
            4.989608066999999,
            4.388719885199999,
            5.8635182073,
            6.432118037299999,
            3.626709587299999,
            1.6460550691,
        ];

        let src = vec![
            5.12, 6.2136, 7.2387, 1.52312, 2.52313, 3.52313, 4.52313, 5.23871,
        ];
        let dst = vec![0.31421, 0.421, 0.653, 0.121];

        let correlation_full =
            Correlate::create_real_f64(src.len(), dst.len(), CrossCorrelationMode::Full).unwrap();
        let full = correlation_full.correlate_managed(&src, &dst).unwrap();
        assert_eq!(full.len(), FULL.len());
        full.iter()
            .zip(FULL.iter())
            .for_each(|(a, b)| assert!((a - b).abs() < 1e-7));

        let correlation_same =
            Correlate::create_real_f64(src.len(), dst.len(), CrossCorrelationMode::Same).unwrap();
        let same = correlation_same.correlate_managed(&src, &dst).unwrap();
        assert_eq!(same.len(), SAME.len());
        same.iter()
            .zip(SAME.iter())
            .for_each(|(a, b)| assert!((a - b).abs() < 1e-7));

        let correlation_valid =
            Correlate::create_real_f64(src.len(), dst.len(), CrossCorrelationMode::Valid).unwrap();
        let valid = correlation_valid.correlate_managed(&src, &dst).unwrap();
        assert_eq!(valid.len(), VALID.len());
        valid
            .iter()
            .zip(VALID.iter())
            .for_each(|(a, b)| assert!((a - b).abs() < 1e-7));
    }
}
