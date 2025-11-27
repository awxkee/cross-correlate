#![no_main]

use arbitrary::Arbitrary;
use cross_correlate::{Correlate, CrossCorrelateError, CrossCorrelationMode, FftExecutor};
use libfuzzer_sys::fuzz_target;
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

#[derive(Clone, Debug, Arbitrary)]
pub struct Input {
    pub buffer_width: u8,
    pub other_width: u8,
    pub buffer_data: f64,
    pub other_data: f64,
    pub mode: u8,
}

struct FftCorrelate {
    executor: Arc<dyn Fft<f64>>,
}

impl FftExecutor<f64> for FftCorrelate {
    fn process(&self, in_out: &mut [Complex<f64>]) -> Result<(), CrossCorrelateError> {
        self.executor.process(in_out);
        Ok(())
    }

    fn length(&self) -> usize {
        self.executor.len()
    }
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

    let fft_size = mode.fft_size(&src, &dst);

    let mut planner = FftPlanner::<f64>::new();
    let fft_forward = planner.plan_fft_forward(fft_size);
    let fft_inverse = planner.plan_fft_inverse(fft_size);
    let correlation = Correlate::create_real_f64(
        mode,
        Arc::new(FftCorrelate {
            executor: fft_forward,
        }),
        Arc::new(FftCorrelate {
            executor: fft_inverse,
        }),
    )
    .unwrap();
    _ = correlation.correlate_managed(&src, &dst).unwrap();
});
