#![no_main]

use arbitrary::Arbitrary;
use cross_correlate::{Correlate, CrossCorrelationMode};
use libfuzzer_sys::fuzz_target;
use num_complex::Complex;

#[derive(Clone, Debug, Arbitrary)]
pub struct Input {
    pub buffer_width: u8,
    pub other_width: u8,
    pub buffer_real: f64,
    pub buffer_imaginary: f64,
    pub other_real: f64,
    pub other_imaginary: f64,
    pub mode: u8,
}

fuzz_target!(|data: Input| {
    if data.buffer_width == 0 || data.other_width == 0 {
        return;
    }
    let mode = match data.mode % 3 {
        0 => CrossCorrelationMode::Valid,
        1 => CrossCorrelationMode::Same,
        _ => CrossCorrelationMode::Full,
    };

    let src = vec![
        Complex {
            re: data.buffer_real,
            im: data.buffer_imaginary,
        };
        data.buffer_width as usize
    ];
    let dst = vec![
        Complex {
            re: data.other_real,
            im: data.other_imaginary,
        };
        data.other_width as usize
    ];

    let correlation = Correlate::create_complex_f64(
        src.len(), dst.len()
        mode,
    )
    .unwrap();
    _ = correlation.correlate_managed(&src, &dst).unwrap();
});
