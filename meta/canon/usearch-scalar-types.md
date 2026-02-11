# USearch Scalar Type Analysis

Summary of findings on vector quantization options in USearch.

## Supported Types

| Type | Size | Description |
|------|------|-------------|
| f32 | 4 bytes | Full precision float (default) |
| f64 | 8 bytes | Double precision |
| f16 | 2 bytes | IEEE 754 half-precision float |
| bf16 | 2 bytes | Brain Float 16 |
| i8 | 1 byte | Signed 8-bit integer |
| b1 | 1 bit | Binary quantization |

## What About int16?

The C++ internals define `i16_k` but it's **not exposed in the C API**. Adding it would require:

1. Extending the C enum in `usearch.h`
2. Type conversion mappings in `lib.cpp`
3. Distance kernels using int32 intermediates (to prevent overflow)
4. SIMD optimizations for each architecture

**Verdict**: Not worth it. At 2 bytes per dimension, f16 provides floating-point semantics which are better suited for vector similarity than integer quantization.

## f16 vs bf16

| Aspect | f16 (IEEE 754) | bf16 (Brain Float) |
|--------|----------------|---------------------|
| Exponent bits | 5 | 8 (same as f32) |
| Mantissa bits | 10 | 7 |
| Dynamic range | ±65504 | ±3.4×10^38 |
| Precision | Higher | Lower |
| f32 conversion | Requires scaling | Trivial (truncate) |

- **f16**: Better precision, smaller dynamic range
- **bf16**: Lower precision, but same range as f32 (no overflow risk)

For vector search, what matters is preserving distance *ordering*. With normalized embeddings, f16's higher precision typically wins.

## Recommendation

**Use f16** for 2-byte quantization when:
- Embeddings are normalized (most modern models)
- You want best precision per byte
- Slightly longer indexing time is acceptable

**Use i8** for aggressive 1-byte quantization when memory is critical.

**Use f32** when precision matters more than memory/speed.
