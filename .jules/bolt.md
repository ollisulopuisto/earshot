# Bolt's Journal

## 2026-10-03 - Complex Baseband Filtering in Dehum Engine
**Learning:** In audio restoration pipelines using demodulated complex baseband estimation (like `DehumEngine`), smoothing filters are evaluated thousands of times across harmonics and local offset bands. `signal.butter` has substantial overhead (designing analog filter, bilinear transform, pole-zero pairing to SOS), making repeated filter construction a major bottleneck. Furthermore, `sosfiltfilt` supports multidimensional arrays along an axis (`axis=-1`), allowing complex components to be filtered simultaneously in a single 2D pass instead of two separate passes for real and imaginary parts.
**Action:** When applying Butterworth/SOS filtering repeatedly with fixed (bandwidth, sample_rate) parameters, cache filter coefficients with `functools.lru_cache` and stack multi-channel/complex components along a new axis to run `sosfiltfilt` once.
