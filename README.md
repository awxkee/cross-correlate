A Rust library for computing [cross-correlation](https://en.wikipedia.org/wiki/Cross-correlation) of signals using FFT.
This crate is useful in signal processing, pattern matching, and time-series analysis where cross-correlation is a common operation.

It supports multiple correlation modes (Full, Same, Valid) and allows different FFT backends.

Performance
Uses FFT-based convolution for O(n log n) performance.
Reuses FFT plans to avoid repeated allocations and planning overhead.
SIMD-friendly design (with optional optimizations for AVX2/NEON).

```rust
 let mut src = vec![
    5.12, 6.2136, 7.2387, 1.52312, 2.52313, 3.52313, 4.52313, 5.23871,
];
let dst = vec![0.31421, 0.421, 0.653, 0.121];

let mode = CrossCorrelationMode::Full;

let correlation = Correlate::create_real_f64(src.len(), dst.len(), mode).unwrap();
let corr = correlation.correlate_managed(&src, &dst).unwrap();
```

This project is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE.md))

at your option.
