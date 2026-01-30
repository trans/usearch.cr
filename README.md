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

### 1. Install libusearch

```bash
# Clone and build USearch
git clone https://github.com/unum-cloud/usearch.git
cd usearch
cmake -B build -DUSEARCH_BUILD_LIB_C=1
cmake --build build --config Release

# Install (adjust path as needed)
sudo cp build/libusearch_c.so /usr/local/lib/
sudo cp c/usearch.h /usr/local/include/
sudo ldconfig
```

### 2. Add the shard

```yaml
dependencies:
  usearch:
    github: trans/usearch.cr
```

```bash
shards install
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
- Adjust `expansion_search` for speed/accuracy tradeoff

## Development

```bash
crystal spec
```

## License

MIT
