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
use crate::cross_correlate::FftExecutor;
use crate::error::try_vec;
use crate::fast_divider::DividerUsize;
use crate::pad::pad_real_to_complex;
use crate::spectrum::SpectrumMultiplier;
use crate::{CrossCorrelate, CrossCorrelateError, CrossCorrelationMode};
use std::sync::Arc;

pub(crate) struct CrossCorrelateSingle {
    pub(crate) fft_forward: Arc<dyn FftExecutor<f32> + Send + Sync>,
    pub(crate) fft_inverse: Arc<dyn FftExecutor<f32> + Send + Sync>,
    pub(crate) multiplier: Arc<dyn SpectrumMultiplier<f32> + Send + Sync>,
    pub(crate) mode: CrossCorrelationMode,
}

impl CrossCorrelate<f32> for CrossCorrelateSingle {
    fn correlate(
        &self,
        output: &mut [f32],
        buffer: &[f32],
        other: &[f32],
    ) -> Result<(), CrossCorrelateError> {
        if buffer.is_empty() || other.is_empty() || output.is_empty() {
            return Err(CrossCorrelateError::BuffersMustNotHaveZeroSize);
        }
        if self.fft_forward.length() != self.fft_inverse.length() {
            return Err(CrossCorrelateError::FftSizesDoNotMatch(
                self.fft_forward.length(),
                self.fft_inverse.length(),
            ));
        }
        let data_length = self.mode.get_size(buffer, other);
        let fft_size = self.mode.fft_size(buffer, other);

        if fft_size != self.fft_forward.length() {
            return Err(CrossCorrelateError::FftAndBuffersSizeDoNotMatch(
                self.fft_forward.length(),
                fft_size,
            ));
        }

        if output.len() != data_length {
            return Err(CrossCorrelateError::OutputSizeDoNotMatch(
                data_length,
                output.len(),
            ));
        }

        let mut padded_src = pad_real_to_complex(buffer, fft_size)?;
        let mut padded_other = pad_real_to_complex(other, fft_size)?;
        self.fft_forward.process(&mut padded_src)?;
        self.fft_forward.process(&mut padded_other)?;
        self.multiplier
            .mul_spectrum(&mut padded_src, &padded_other, fft_size);
        self.fft_inverse.process(&mut padded_src)?;

        let lag = other.len() - 1;
        let offset = fft_size - lag;
        match self.mode {
            CrossCorrelationMode::Full => {
                if fft_size == 1 {
                    for dst in output.iter_mut() {
                        *dst = unsafe { padded_src.get_unchecked(0).re }
                    }
                } else {
                    let divisor = DividerUsize::new(fft_size);
                    for (i, dst) in output.iter_mut().enumerate() {
                        *dst = unsafe { padded_src.get_unchecked((i + offset) % divisor).re }
                    }
                }
            }
            CrossCorrelationMode::Valid | CrossCorrelationMode::Same => {
                let start = match self.mode {
                    CrossCorrelationMode::Valid => other.len() - 1,
                    CrossCorrelationMode::Same => (other.len() - 1) / 2,
                    CrossCorrelationMode::Full => unreachable!(),
                };
                if fft_size == 1 {
                    for dst in output.iter_mut() {
                        *dst = unsafe { padded_src.get_unchecked(0).re }
                    }
                } else {
                    let divisor = DividerUsize::new(fft_size);
                    for (i, dst) in output.iter_mut().enumerate() {
                        *dst =
                            unsafe { padded_src.get_unchecked((start + i + offset) % divisor).re };
                    }
                }
            }
        }

        Ok(())
    }

    fn correlate_managed(
        &self,
        buffer: &[f32],
        other: &[f32],
    ) -> Result<Vec<f32>, CrossCorrelateError> {
        let data_length = self.mode.get_size(buffer, other);
        let mut output = try_vec![0.; data_length];
        self.correlate(&mut output, buffer, other).map(|_| output)
    }
}
