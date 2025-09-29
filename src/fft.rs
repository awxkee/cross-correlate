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

/// Find the next "good" FFT size greater than or equal to `n`.
///
/// A "good" FFT size is an integer whose prime factorization contains
/// only the factors 2, 3, and 5. Such lengths are considered efficient
/// for FFT implementations.
///
/// If `n` itself is already a good FFT size, it is returned directly.
/// Otherwise, the function increments `n` until it finds the next valid size.
pub fn fft_next_good_size(n: usize) -> usize {
    if n <= 2 {
        return 2;
    }

    // helper: smallest power of `base` >= n, computed in u128 to avoid overflow.
    fn next_pow_base(base: usize, n: usize) -> usize {
        let mut p: u128 = 1;
        let target: u128 = n as u128;
        let b: u128 = base as u128;

        while p < target {
            p *= b;
            if p > u128::from(usize::MAX as u128) {
                // overflow: return a sentinel large value so it won't be chosen as min
                return usize::MAX;
            }
        }
        p as usize
    }

    // compute candidates for each base
    let p2 = next_pow_base(2, n);
    let p3 = next_pow_base(3, n);
    let p4 = next_pow_base(4, n);
    let p5 = next_pow_base(5, n);

    // return the smallest candidate
    p2.min(p3).min(p4).min(p5)
}

#[cfg(test)]
mod tests {
    use crate::fft_next_good_size;

    #[test]
    fn test_fft_next_good_size() {
        assert_eq!(2, fft_next_good_size(1));
        assert_eq!(3, fft_next_good_size(3));
        assert_eq!(4, fft_next_good_size(4));
        assert_eq!(5, fft_next_good_size(5));
        assert_eq!(8, fft_next_good_size(6));
        assert_eq!(16, fft_next_good_size(12));
        assert_eq!(16, fft_next_good_size(16));
        assert_eq!(25, fft_next_good_size(20));
        assert_eq!(64, fft_next_good_size(37));
        assert_eq!(128, fft_next_good_size(128));
        assert_eq!(1024, fft_next_good_size(914));
    }
}
