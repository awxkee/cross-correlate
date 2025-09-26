A Rust library for computing [cross-correlation](https://en.wikipedia.org/wiki/Cross-correlation) of signals using FFT.
This crate is useful in signal processing, pattern matching, and time-series analysis where cross-correlation is a common operation.

It supports multiple correlation modes (Full, Same, Valid) and allows different FFT backends.

Performance
Uses FFT-based convolution for O(n log n) performance.
Reuses FFT plans to avoid repeated allocations and planning overhead.
SIMD-friendly design (with optional optimizations for AVX2/NEON).

```rust
 let mut src = vec![
        5.12f32, 6.2136, 7.2387, 1.52312, 2.52313, 3.52313, 4.52313, 5.23871,
];
let dst = vec![0.31421, 0.421, 0.653, 0.121];

let mode = CrossCorrelationMode::Full;

// determine FFT size.
let fft_size = mode.fft_size(&src, &dst);

let mut planner = FftPlanner::<f32>::new();
let fft_forward = planner.plan_fft_forward(fft_size);
let fft_inverse = planner.plan_fft_inverse(fft_size);

struct FftCorrelate {
    executor: Arc<dyn Fft<f32>>,
}

impl FftExecutor<f32> for FftCorrelate {
    fn process(&self, in_out: &mut [Complex<f32>]) -> Result<(), CrossCorrelateError> {
        self.executor.process(in_out);
        Ok(())
    }

    fn length(&self) -> usize {
        self.executor.len()
    }
}

// create correlation executor
let correlation = Correlate::create_real_f32(
    mode,
    Box::new(FftCorrelate {
        executor: fft_forward,
    }),
    Box::new(FftCorrelate {
        executor: fft_inverse,
    }),
)
.unwrap();
let corr = correlation.correlate_managed(&src, &dst).unwrap();
```

This project is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE.md))

at your option.
