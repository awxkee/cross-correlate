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
use cross_correlate::{Correlate, CrossCorrelationMode};

fn main() {
    let mut src = vec![
        5.12, 6.2136, 7.2387, 1.52312, 2.52313, 3.52313, 4.52313, 5.23871,
    ];
    let dst = vec![0.31421, 0.421, 0.653, 0.121];

    let mode = CrossCorrelationMode::Full;

    // [9.135849419999998, 6.299764045999998, 4.989608066999998, 4.3887198852, 5.8635182073]
    // [9.13585, 6.2997656, 4.989609, 4.388721, 5.8635187]

    let correlation = Correlate::create_real_f64(src.len(), dst.len(), mode).unwrap();
    let corr = correlation.correlate_managed(&src, &dst).unwrap();

    println!("Hello, world!");
    println!("Correlate {:?}", corr);
    //  [0.6195199999999983, 4.095205599999998, 7.0888835, 9.13584942, 6.299764045999999, 4.989608066999997, 4.388719885199999, 5.8635182073, 6.4321180372999995, 3.6267095872999997, 1.6460550691000004]
}
