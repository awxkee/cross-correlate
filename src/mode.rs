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

/// Specifies the output size for cross-correlation operations.
///
/// The mode determines how much of the cross-correlation result is returned
/// relative to the input sequences.
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash, Default)]
pub enum CrossCorrelationMode {
    /// Only fully overlapping elements are returned
    Valid,
    /// Output has the same length as the largest input
    Same,
    /// Full cross-correlation (default)
    #[default]
    Full,
}

impl CrossCorrelationMode {
    /// Compute the output length for a cross-correlation operation.
    ///
    /// The length of the correlation result depends on the chosen
    /// [`CrossCorrelationMode`] and the lengths of the input sequences.
    pub fn get_size(self, buffer_len: usize, other_len: usize) -> usize {
        match self {
            CrossCorrelationMode::Valid => {
                buffer_len.max(other_len) - buffer_len.min(other_len) + 1
            }
            CrossCorrelationMode::Same => buffer_len.max(other_len),
            CrossCorrelationMode::Full => buffer_len + other_len - 1,
        }
    }

    /// Compute the FFT size required for cross-correlation.
    ///
    /// This method determines the minimum "good" FFT size needed to perform
    /// cross-correlation between two real-valued input sequences. Internally it
    /// calls [`fft_next_good_size`] to round up to an efficient FFT length.
    #[inline]
    pub fn fft_size(self, buffer_len: usize, other_len: usize) -> usize {
        buffer_len + other_len - 1
    }
}
