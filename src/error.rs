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
use std::error::Error;
use std::fmt::Display;

#[derive(Clone, Debug)]
pub enum CrossCorrelateError {
    FftSizesDoNotMatch(usize, usize),
    FftError(String),
    OutOfMemory(usize),
    FftAndBuffersSizeDoNotMatch(usize, usize),
    OutputSizeDoNotMatch(usize, usize),
    BuffersMustNotHaveZeroSize,
}

impl Error for CrossCorrelateError {}

impl Display for CrossCorrelateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CrossCorrelateError::FftError(z) => f.write_str(z.as_str()),
            CrossCorrelateError::FftSizesDoNotMatch(s0, s1) => f.write_fmt(format_args!(
                "Forward fft size {s0} doesn't match inverse fft size {s1}"
            )),
            CrossCorrelateError::OutOfMemory(capacity) => f.write_fmt(format_args!(
                "There is no enough memory to allocate {capacity} bytes"
            )),
            CrossCorrelateError::FftAndBuffersSizeDoNotMatch(s0, s1) => f.write_fmt(format_args!(
                "Fft size {s0} and buffer size {s1} doesn't match"
            )),
            CrossCorrelateError::OutputSizeDoNotMatch(s0, s1) => {
                f.write_fmt(format_args!("Output size should be {s0} but it was {s1}"))
            }
            CrossCorrelateError::BuffersMustNotHaveZeroSize => {
                f.write_str("Buffers must have zero size")
            }
        }
    }
}

macro_rules! try_vec {
    () => {
        Vec::new()
    };
    ($elem:expr; $n:expr) => {{
        let mut v = Vec::new();
        v.try_reserve_exact($n)
            .map_err(|_| crate::error::CrossCorrelateError::OutOfMemory($n))?;
        v.resize($n, $elem);
        v
    }};
}

pub(crate) use try_vec;
