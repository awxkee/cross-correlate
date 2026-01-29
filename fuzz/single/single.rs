#![no_main]

use arbitrary::Arbitrary;
use cross_correlate::{Correlate, CrossCorrelationMode};
use libfuzzer_sys::fuzz_target;

#[derive(Clone, Debug, Arbitrary)]
pub struct Input {
    pub buffer_width: u8,
    pub other_width: u8,
    pub buffer_data: f32,
    pub other_data: f32,
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

    let src = vec![data.buffer_data; data.buffer_width as usize];
    let dst = vec![data.other_data; data.other_width as usize];

    let correlation = Correlate::create_real_f32(
        src.len(), dst.len(),
        mode,
    )
    .unwrap();
    _ = correlation.correlate_managed(&src, &dst).unwrap();
});
