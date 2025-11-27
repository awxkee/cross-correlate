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
use crate::double::CrossCorrelateDouble;
use crate::double_complex::CrossCorrelateComplexDouble;
use crate::single::CrossCorrelateSingle;
use crate::single_complex::CrossCorrelateComplexSingle;
use crate::{CrossCorrelateError, CrossCorrelationMode};
use num_complex::Complex;
use std::fmt::Debug;
use std::sync::Arc;

/// Trait representing an FFT executor for a given numeric type `V`.
///
/// Implementors of this trait provide a way to perform in-place
/// FFT computations on slices of complex numbers. This is typically
/// used internally for FFT-based cross-correlation or convolution.
pub trait FftExecutor<V> {
    /// Process the FFT in-place on the given slice of complex numbers.
    fn process(&self, in_out: &mut [Complex<V>]) -> Result<(), CrossCorrelateError>;
    /// Get the length of the FFT that this executor can process.
    fn length(&self) -> usize;
}

/// Trait for computing cross-correlation between two sequences.
///
/// The `CrossCorrelate` trait provides methods to perform cross-correlation
/// on real-valued sequences of type `V`. Implementors usually leverage FFT
/// for efficient computation, but the trait is generic enough to allow
/// direct or SIMD-based implementations.
pub trait CrossCorrelate<V: Clone + Debug + Default> {
    /// Compute cross-correlation and store in pre-allocated output slice.
    fn correlate(
        &self,
        output: &mut [V],
        buffer: &[V],
        other: &[V],
    ) -> Result<(), CrossCorrelateError>;
    /// Compute cross-correlation and return a new `Vec<V>` with the result.
    fn correlate_managed(&self, buffer: &[V], other: &[V]) -> Result<Vec<V>, CrossCorrelateError>;
}

/// A cross-correlation engine for signals.
///
/// The `Correlate` struct provides methods to compute cross-correlation
/// between two sequences, typically using FFT-based acceleration for
/// efficiency. It supports multiple correlation modes (`Full`, `Same`, `Valid`)
/// and can work with pre-planned FFT executors for reuse.
pub struct Correlate {}

impl Correlate {
    /// Create a real-valued cross-correlator using FFT.
    ///
    /// This function constructs a cross-correlator for `f32` signals, based on the
    /// provided correlation mode and FFT executors. It uses the forward FFT to
    /// transform both input signals, multiplies one by the conjugate spectrum of the other,
    /// and then applies the inverse FFT to obtain the cross-correlation result.
    ///
    /// # Arguments
    ///
    /// * `mode` - The [`CrossCorrelationMode`] that determines the shape of the output
    ///   (e.g. `Full`, `Same`, `Valid`).
    /// * `fft_forward` - A boxed [`FftExecutor`] used for the forward transform.
    ///   Typically, created using `rustfft::FftPlanner::plan_fft_forward`.
    /// * `fft_inverse` - A boxed [`FftExecutor`] used for the inverse transform.
    ///   Typically, created using `rustfft::FftPlanner::plan_fft_inverse`.
    ///
    /// # Returns
    ///
    /// A boxed [`CrossCorrelate`] instance implementing cross-correlation on `f32` signals,
    /// or an error if the FFT sizes do not match.
    ///
    /// # Errors
    ///
    /// Returns [`CrossCorrelateError`] if:
    /// - The forward and inverse FFT executors have mismatched lengths.
    /// - The FFT size computed from the input buffers does not match the provided executors.
    ///
    pub fn create_real_f32(
        mode: CrossCorrelationMode,
        fft_forward: Arc<dyn FftExecutor<f32> + Send + Sync>,
        fft_inverse: Arc<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Arc<dyn CrossCorrelate<f32> + Sync + Send>, CrossCorrelateError> {
        if fft_forward.length() != fft_inverse.length() {
            return Err(CrossCorrelateError::FftSizesDoNotMatch(
                fft_forward.length(),
                fft_inverse.length(),
            ));
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::MulSpectrumSingleAvxFma;
                return Ok(Arc::new(CrossCorrelateSingle {
                    fft_forward,
                    fft_inverse,
                    multiplier: Arc::new(MulSpectrumSingleAvxFma::default()),
                    mode,
                }));
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "sse"))]
        {
            if std::arch::is_x86_feature_detected!("sse4.2") {
                use crate::sse::MulSpectrumSingleSse4_2;
                return Ok(Arc::new(CrossCorrelateSingle {
                    fft_forward,
                    fft_inverse,
                    multiplier: Arc::new(MulSpectrumSingleSse4_2::default()),
                    mode,
                }));
            }
        }
        #[cfg(all(target_arch = "aarch64", feature = "fcma"))]
        {
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::neon::SpectrumMulSingleFcma;
                return Ok(Arc::new(CrossCorrelateSingle {
                    fft_forward,
                    fft_inverse,
                    multiplier: Arc::new(SpectrumMulSingleFcma::default()),
                    mode,
                }));
            }
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::SpectrumMulSingleNeon;
            Ok(Arc::new(CrossCorrelateSingle {
                fft_forward,
                fft_inverse,
                multiplier: Arc::new(SpectrumMulSingleNeon::default()),
                mode,
            }))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::spectrum::SpectrumMultiplierSingle;
            Ok(Arc::new(CrossCorrelateSingle {
                fft_forward,
                fft_inverse,
                multiplier: Arc::new(SpectrumMultiplierSingle::default()),
                mode,
            }))
        }
    }

    /// Creates a cross-correlation engine for complex `f32` sequences.
    ///
    /// This function constructs a `CrossCorrelate` implementation that operates
    /// on `Complex<f32>` data, using the provided FFT executors for forward and
    /// inverse transforms. It supports different correlation modes (`Full`, `Same`, `Valid`)
    /// as specified by `mode`.
    ///
    /// # Arguments
    ///
    /// * `mode` - The [`CrossCorrelationMode`] that determines the shape of the output
    ///   (e.g. `Full`, `Same`, `Valid`).
    /// * `fft_forward` - A boxed [`FftExecutor`] used for the forward transform.
    ///   Typically, created using `rustfft::FftPlanner::plan_fft_forward`.
    /// * `fft_inverse` - A boxed [`FftExecutor`] used for the inverse transform.
    ///   Typically, created using `rustfft::FftPlanner::plan_fft_inverse`.
    ///
    /// # Returns
    ///
    /// A boxed [`CrossCorrelate`] instance implementing cross-correlation on complex `f32` signals,
    /// or an error if the FFT sizes do not match.
    ///
    /// # Errors
    ///
    /// Returns [`CrossCorrelateError`] if:
    /// - The forward and inverse FFT executors have mismatched lengths.
    /// - The FFT size computed from the input buffers does not match the provided executors.
    ///
    pub fn create_complex_f32(
        mode: CrossCorrelationMode,
        fft_forward: Arc<dyn FftExecutor<f32> + Send + Sync>,
        fft_inverse: Arc<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Arc<dyn CrossCorrelate<Complex<f32>> + Sync + Send>, CrossCorrelateError> {
        if fft_forward.length() != fft_inverse.length() {
            return Err(CrossCorrelateError::FftSizesDoNotMatch(
                fft_forward.length(),
                fft_inverse.length(),
            ));
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::MulSpectrumSingleAvxFma;
                return Ok(Arc::new(CrossCorrelateComplexSingle {
                    fft_forward,
                    fft_inverse,
                    multiplier: Arc::new(MulSpectrumSingleAvxFma::default()),
                    mode,
                }));
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "sse"))]
        {
            if std::arch::is_x86_feature_detected!("sse4.2") {
                use crate::sse::MulSpectrumSingleSse4_2;
                return Ok(Arc::new(CrossCorrelateComplexSingle {
                    fft_forward,
                    fft_inverse,
                    multiplier: Arc::new(MulSpectrumSingleSse4_2::default()),
                    mode,
                }));
            }
        }
        #[cfg(all(target_arch = "aarch64", feature = "fcma"))]
        {
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::neon::SpectrumMulSingleFcma;
                return Ok(Arc::new(CrossCorrelateComplexSingle {
                    fft_forward,
                    fft_inverse,
                    multiplier: Arc::new(SpectrumMulSingleFcma::default()),
                    mode,
                }));
            }
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::SpectrumMulSingleNeon;
            Ok(Arc::new(CrossCorrelateComplexSingle {
                fft_forward,
                fft_inverse,
                multiplier: Arc::new(SpectrumMulSingleNeon::default()),
                mode,
            }))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::spectrum::SpectrumMultiplierSingle;
            Ok(Arc::new(CrossCorrelateComplexSingle {
                fft_forward,
                fft_inverse,
                multiplier: Arc::new(SpectrumMultiplierSingle::default()),
                mode,
            }))
        }
    }

    /// Create a real-valued cross-correlator using FFT.
    ///
    /// This function constructs a cross-correlator for `f64` signals, based on the
    /// provided correlation mode and FFT executors. It uses the forward FFT to
    /// transform both input signals, multiplies one by the conjugate spectrum of the other,
    /// and then applies the inverse FFT to obtain the cross-correlation result.
    ///
    /// # Arguments
    ///
    /// * `mode` - The [`CrossCorrelationMode`] that determines the shape of the output
    ///   (e.g. `Full`, `Same`, `Valid`).
    /// * `fft_forward` - A boxed [`FftExecutor`] used for the forward transform.
    ///   Typically, created using `rustfft::FftPlanner::plan_fft_forward`.
    /// * `fft_inverse` - A boxed [`FftExecutor`] used for the inverse transform.
    ///   Typically, created using `rustfft::FftPlanner::plan_fft_inverse`.
    ///
    /// # Returns
    ///
    /// A boxed [`CrossCorrelate`] instance implementing cross-correlation on `f64` signals,
    /// or an error if the FFT sizes do not match.
    ///
    /// # Errors
    ///
    /// Returns [`CrossCorrelateError`] if:
    /// - The forward and inverse FFT executors have mismatched lengths.
    /// - The FFT size computed from the input buffers does not match the provided executors.
    ///
    pub fn create_real_f64(
        mode: CrossCorrelationMode,
        fft_forward: Arc<dyn FftExecutor<f64> + Send + Sync>,
        fft_inverse: Arc<dyn FftExecutor<f64> + Send + Sync>,
    ) -> Result<Arc<dyn CrossCorrelate<f64> + Sync + Send>, CrossCorrelateError> {
        if fft_forward.length() != fft_inverse.length() {
            return Err(CrossCorrelateError::FftSizesDoNotMatch(
                fft_forward.length(),
                fft_inverse.length(),
            ));
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::MulSpectrumDoubleAvxFma;
                return Ok(Arc::new(CrossCorrelateDouble {
                    fft_forward,
                    fft_inverse,
                    multiplier: Arc::new(MulSpectrumDoubleAvxFma::default()),
                    mode,
                }));
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "sse"))]
        {
            if std::arch::is_x86_feature_detected!("sse4.2") {
                use crate::sse::MulSpectrumDoubleSse4_2;
                return Ok(Arc::new(CrossCorrelateDouble {
                    fft_forward,
                    fft_inverse,
                    multiplier: Arc::new(MulSpectrumDoubleSse4_2::default()),
                    mode,
                }));
            }
        }
        #[cfg(all(target_arch = "aarch64", feature = "fcma"))]
        {
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::neon::SpectrumMulDoubleFcma;
                return Ok(Arc::new(CrossCorrelateDouble {
                    fft_forward,
                    fft_inverse,
                    multiplier: Arc::new(SpectrumMulDoubleFcma::default()),
                    mode,
                }));
            }
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::SpectrumMulDoubleNeon;
            Ok(Arc::new(CrossCorrelateDouble {
                fft_forward,
                fft_inverse,
                multiplier: Arc::new(SpectrumMulDoubleNeon::default()),
                mode,
            }))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::spectrum::SpectrumMultiplierDouble;
            Ok(Arc::new(CrossCorrelateDouble {
                fft_forward,
                fft_inverse,
                multiplier: Arc::new(SpectrumMultiplierDouble::default()),
                mode,
            }))
        }
    }

    /// Create a real-valued cross-correlator using FFT.
    ///
    /// This function constructs a cross-correlator for complex `f64` signals, based on the
    /// provided correlation mode and FFT executors. It uses the forward FFT to
    /// transform both input signals, multiplies one by the conjugate spectrum of the other,
    /// and then applies the inverse FFT to obtain the cross-correlation result.
    ///
    /// # Arguments
    ///
    /// * `mode` - The [`CrossCorrelationMode`] that determines the shape of the output
    ///   (e.g. `Full`, `Same`, `Valid`).
    /// * `fft_forward` - A boxed [`FftExecutor`] used for the forward transform.
    ///   Typically, created using `rustfft::FftPlanner::plan_fft_forward`.
    /// * `fft_inverse` - A boxed [`FftExecutor`] used for the inverse transform.
    ///   Typically, created using `rustfft::FftPlanner::plan_fft_inverse`.
    ///
    /// # Returns
    ///
    /// A boxed [`CrossCorrelate`] instance implementing cross-correlation on complex `f64` signals,
    /// or an error if the FFT sizes do not match.
    ///
    /// # Errors
    ///
    /// Returns [`CrossCorrelateError`] if:
    /// - The forward and inverse FFT executors have mismatched lengths.
    /// - The FFT size computed from the input buffers does not match the provided executors.
    ///
    pub fn create_complex_f64(
        mode: CrossCorrelationMode,
        fft_forward: Arc<dyn FftExecutor<f64> + Send + Sync>,
        fft_inverse: Arc<dyn FftExecutor<f64> + Send + Sync>,
    ) -> Result<Arc<dyn CrossCorrelate<Complex<f64>> + Sync + Send>, CrossCorrelateError> {
        if fft_forward.length() != fft_inverse.length() {
            return Err(CrossCorrelateError::FftSizesDoNotMatch(
                fft_forward.length(),
                fft_inverse.length(),
            ));
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::MulSpectrumDoubleAvxFma;
                return Ok(Arc::new(CrossCorrelateComplexDouble {
                    fft_forward,
                    fft_inverse,
                    multiplier: Arc::new(MulSpectrumDoubleAvxFma::default()),
                    mode,
                }));
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "sse"))]
        {
            if std::arch::is_x86_feature_detected!("sse4.2") {
                use crate::sse::MulSpectrumDoubleSse4_2;
                return Ok(Arc::new(CrossCorrelateComplexDouble {
                    fft_forward,
                    fft_inverse,
                    multiplier: Arc::new(MulSpectrumDoubleSse4_2::default()),
                    mode,
                }));
            }
        }
        #[cfg(all(target_arch = "aarch64", feature = "fcma"))]
        {
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::neon::SpectrumMulDoubleFcma;
                return Ok(Arc::new(CrossCorrelateComplexDouble {
                    fft_forward,
                    fft_inverse,
                    multiplier: Arc::new(SpectrumMulDoubleFcma::default()),
                    mode,
                }));
            }
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::SpectrumMulDoubleNeon;
            Ok(Arc::new(CrossCorrelateComplexDouble {
                fft_forward,
                fft_inverse,
                multiplier: Arc::new(SpectrumMulDoubleNeon::default()),
                mode,
            }))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::spectrum::SpectrumMultiplierDouble;
            Ok(Arc::new(CrossCorrelateComplexDouble {
                fft_forward,
                fft_inverse,
                multiplier: Arc::new(SpectrumMultiplierDouble::default()),
                mode,
            }))
        }
    }
}
