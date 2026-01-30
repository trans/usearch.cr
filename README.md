# usearch.cr

Crystal bindings for [USearch](https://github.com/unum-cloud/usearch), a fast approximate nearest neighbor search library using HNSW.

## Features

- Fast ANN search via HNSW (Hierarchical Navigable Small World) graphs
- Multiple distance metrics (cosine, L2, inner product, etc.)
- Multiple quantization formats (f32, f16, i8, binary)
- Single-file persistence
- Memory-mapped indexes for large datasets
- Scales to millions of vectors

## Installation

### 1. Add the shard

```yaml
dependencies:
  usearch:
    github: trans/usearch.cr
```

```bash
shards install
```

### 2. Build libusearch

```bash
# Run the setup script (clones and builds usearch)
./scripts/setup.sh
```

This clones USearch into `vendor/usearch/` and builds the static library. The library is statically linked, so no runtime dependencies are needed.

#### Requirements

- CMake 3.14+
- C++17 compiler (GCC 8+ or Clang 10+)

#### Dynamic linking (alternative)

If you prefer dynamic linking:

```bash
# Build with dynamic linking flag
crystal build -Dusearch_dynamic src/myapp.cr

# Set library path at runtime
LD_LIBRARY_PATH=vendor/usearch/build ./myapp
```

## Usage

```crystal
require "usearch"

# Create an index
index = USearch::Index.new(
  dimensions: 128,
  metric: :cos,           # :cos, :l2sq, :ip, :hamming, etc.
  quantization: :f16      # :f32, :f16, :i8, :b1
)

# Add vectors (key = your database row ID)
index.add(1_u64, vector1)
index.add(2_u64, vector2)
index.add(3_u64, vector3)

# Search for nearest neighbors
results = index.search(query_vector, k: 10)
results.each do |r|
  puts "Key: #{r.key}, Distance: #{r.distance}"
end

# Check if key exists
index.contains?(1_u64)  # => true

# Remove a vector
index.remove(1_u64)

# Save to disk
index.save("vectors.usearch")

# Load later
index = USearch::Index.load("vectors.usearch", dimensions: 128)

# Or memory-map for large indexes
index = USearch::Index.view("vectors.usearch", dimensions: 128)

# Clean up
index.close
```

### Filtered Search

Search with a predicate to filter candidates:

```crystal
# Only return vectors with even keys
results = index.filtered_search(query, k: 10) { |key| key.even? }

# Only return vectors in a specific set
valid_ids = Set{1_u64, 5_u64, 10_u64}
results = index.filtered_search(query, k: 10) { |key| valid_ids.includes?(key) }
```

### Exact Search

Brute-force search (useful for ground truth or small datasets):

```crystal
dataset = [vec1, vec2, vec3, ...]  # Array(Array(Float32))
queries = [query1, query2]

results = USearch.exact_search(dataset, queries, k: 10, metric: :cos)
# results[0] = top-10 for query1, results[1] = top-10 for query2
```

### Serialization to Bytes

```crystal
# Serialize to bytes (for embedding in other formats)
bytes = index.to_bytes

# Load from bytes
index = USearch::Index.from_bytes(bytes, dimensions: 128)

# View from bytes (zero-copy, buffer must stay alive)
index = USearch::Index.view_bytes(bytes, dimensions: 128)

# Inspect metadata without loading
meta = USearch::Index.metadata("vectors.usearch")
puts meta.dimensions  # => 128
```

## Metrics

| Metric | Description |
|--------|-------------|
| `:cos` | Cosine similarity (default) |
| `:ip` | Inner product |
| `:l2sq` | Squared Euclidean distance |
| `:hamming` | Hamming distance (for binary) |
| `:jaccard` | Jaccard index |
| `:pearson` | Pearson correlation |

## Quantization

| Type | Bytes/dim | Use case |
|------|-----------|----------|
| `:f32` | 4 | Maximum precision |
| `:f16` | 2 | Good balance (default) |
| `:i8` | 1 | Memory constrained |
| `:b1` | 0.125 | Binary vectors |

## Performance Tips

- Use `f16` quantization for 2x memory savings with minimal recall loss
- Call `reserve(n)` before bulk inserts to avoid reallocations
- Use `view()` instead of `load()` for very large indexes
- Tune `expansion_search` for speed/accuracy tradeoff:
  ```crystal
  index.expansion_search = 128  # Higher = more accurate, slower
  index.expansion_add = 256     # Higher = better graph quality
  ```

## Utilities

```crystal
# Library version
USearch::Index.version  # => "2.x.x"

# SIMD acceleration in use
USearch.hardware_acceleration  # => "avx2"
```

## Development

```bash
crystal spec
```

## License

MIT
