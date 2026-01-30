require "spec"
require "../src/usearch"

describe USearch do
  describe "VERSION" do
    it "has a version" do
      USearch::VERSION.should_not be_nil
    end
  end

  describe ".hardware_acceleration" do
    it "returns a string describing SIMD support" do
      accel = USearch.hardware_acceleration
      accel.should_not be_empty
      # Common values: "serial", "neon", "sve", "avx2", "avx512"
    end
  end

  describe ".exact_search" do
    it "finds exact nearest neighbors" do
      dataset = [
        [1.0_f32, 0.0_f32, 0.0_f32, 0.0_f32],
        [0.0_f32, 1.0_f32, 0.0_f32, 0.0_f32],
        [0.0_f32, 0.0_f32, 1.0_f32, 0.0_f32],
        [0.0_f32, 0.0_f32, 0.0_f32, 1.0_f32],
      ]

      queries = [
        [0.9_f32, 0.1_f32, 0.0_f32, 0.0_f32],  # Closest to dataset[0]
        [0.0_f32, 0.9_f32, 0.1_f32, 0.0_f32],  # Closest to dataset[1]
      ]

      results = USearch.exact_search(dataset, queries, k: 2, metric: :cos)

      results.size.should eq 2
      results[0].size.should eq 2
      results[0][0].key.should eq 0_u64  # First query closest to first dataset vector
      results[1][0].key.should eq 1_u64  # Second query closest to second dataset vector
    end

    it "handles single query" do
      dataset = [
        [1.0_f32, 0.0_f32],
        [0.0_f32, 1.0_f32],
      ]

      queries = [
        [0.7_f32, 0.3_f32],
      ]

      results = USearch.exact_search(dataset, queries, k: 2, metric: :cos)

      results.size.should eq 1
      results[0].size.should eq 2
      results[0][0].key.should eq 0_u64  # Closer to [1, 0]
    end

    it "raises on empty dataset" do
      expect_raises(USearch::Error, /empty/) do
        USearch.exact_search([] of Array(Float32), [[1.0_f32]], k: 1)
      end
    end

    it "raises on empty queries" do
      expect_raises(USearch::Error, /empty/) do
        USearch.exact_search([[1.0_f32]], [] of Array(Float32), k: 1)
      end
    end

    it "raises on dimension mismatch" do
      expect_raises(USearch::Error, /dimension mismatch/) do
        USearch.exact_search(
          [[1.0_f32, 2.0_f32]],
          [[1.0_f32, 2.0_f32, 3.0_f32]],  # Wrong dimension
          k: 1
        )
      end
    end

    it "raises on invalid k" do
      expect_raises(USearch::Error, /k must be positive/) do
        USearch.exact_search([[1.0_f32]], [[1.0_f32]], k: 0)
      end
    end

    it "raises on invalid threads" do
      expect_raises(USearch::Error, /threads must be positive/) do
        USearch.exact_search([[1.0_f32]], [[1.0_f32]], k: 1, threads: 0)
      end
    end
  end

  describe USearch::Index do
    test_dims = 4

    describe ".new" do
      it "creates an empty index" do
        index = USearch::Index.new(dimensions: test_dims)
        index.size.should eq 0
        index.dimensions.should eq test_dims
        index.close
      end

      it "accepts metric and quantization options" do
        index = USearch::Index.new(
          dimensions: test_dims,
          metric: :l2sq,
          quantization: :f32
        )
        index.size.should eq 0
        index.close
      end
    end

    describe "#add and #search" do
      it "adds vectors and finds nearest neighbors" do
        index = USearch::Index.new(dimensions: test_dims, metric: :cos)

        # Add some vectors
        index.add(1_u64, [1.0_f32, 0.0_f32, 0.0_f32, 0.0_f32])
        index.add(2_u64, [0.0_f32, 1.0_f32, 0.0_f32, 0.0_f32])
        index.add(3_u64, [0.0_f32, 0.0_f32, 1.0_f32, 0.0_f32])

        index.size.should eq 3

        # Search for vector similar to first one
        results = index.search([0.9_f32, 0.1_f32, 0.0_f32, 0.0_f32], k: 2)

        results.size.should eq 2
        results[0].key.should eq 1_u64  # Should be closest to first vector

        index.close
      end

      it "raises on dimension mismatch" do
        index = USearch::Index.new(dimensions: test_dims)

        expect_raises(USearch::Error, /dimension mismatch/) do
          index.add(1_u64, [1.0_f32, 2.0_f32])  # Wrong size
        end

        index.close
      end
    end

    describe "#contains?" do
      it "checks if key exists" do
        index = USearch::Index.new(dimensions: test_dims)

        index.contains?(1_u64).should be_false

        index.add(1_u64, [1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32])

        index.contains?(1_u64).should be_true
        index.contains?(2_u64).should be_false

        index.close
      end
    end

    describe "#count" do
      it "returns count for a key" do
        index = USearch::Index.new(dimensions: test_dims)

        index.count(1_u64).should eq 0

        index.add(1_u64, [1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32])

        index.count(1_u64).should eq 1
        index.count(2_u64).should eq 0

        index.close
      end
    end

    describe "#get" do
      it "retrieves vector data by key" do
        index = USearch::Index.new(dimensions: test_dims, quantization: :f32)
        vec = [1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32]

        index.add(1_u64, vec)

        retrieved = index.get(1_u64)
        retrieved.should_not be_nil
        retrieved = retrieved.not_nil!

        # Check values are close (may have minor floating point differences)
        retrieved.size.should eq test_dims
        retrieved.each_with_index do |val, i|
          val.should be_close(vec[i], 0.01)
        end

        index.close
      end

      it "returns nil for non-existent key" do
        index = USearch::Index.new(dimensions: test_dims)

        index.get(999_u64).should be_nil

        index.close
      end
    end

    describe "#filtered_search" do
      it "filters results with a predicate" do
        index = USearch::Index.new(dimensions: test_dims, metric: :cos)

        # Add vectors with keys 0-9
        10.times do |i|
          vec = [i.to_f32, 0.0_f32, 0.0_f32, 0.0_f32]
          index.add(i.to_u64, vec)
        end

        # Search with filter: only even keys
        query = [5.0_f32, 0.0_f32, 0.0_f32, 0.0_f32]
        results = index.filtered_search(query, k: 5) { |key| key.even? }

        # All results should have even keys
        results.each do |r|
          r.key.even?.should be_true
        end

        index.close
      end

      it "filters with a set of valid keys" do
        index = USearch::Index.new(dimensions: test_dims, metric: :cos)

        5.times do |i|
          vec = [i.to_f32, 0.0_f32, 0.0_f32, 0.0_f32]
          index.add(i.to_u64, vec)
        end

        valid_keys = Set{1_u64, 3_u64}
        query = [2.0_f32, 0.0_f32, 0.0_f32, 0.0_f32]
        results = index.filtered_search(query, k: 5) { |key| valid_keys.includes?(key) }

        results.size.should be <= 2
        results.each do |r|
          valid_keys.includes?(r.key).should be_true
        end

        index.close
      end
    end

    describe "#remove" do
      it "removes a vector by key" do
        index = USearch::Index.new(dimensions: test_dims)

        index.add(1_u64, [1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32])
        index.contains?(1_u64).should be_true

        index.remove(1_u64)
        index.contains?(1_u64).should be_false

        index.close
      end
    end

    describe "#clear" do
      it "removes all vectors" do
        index = USearch::Index.new(dimensions: test_dims)

        index.add(1_u64, [1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32])
        index.add(2_u64, [5.0_f32, 6.0_f32, 7.0_f32, 8.0_f32])
        index.size.should eq 2

        index.clear
        index.size.should eq 0

        index.close
      end
    end

    describe "#save and .load" do
      it "persists and loads an index" do
        path = File.tempname("usearch_test", ".usearch")

        begin
          # Create and save
          index = USearch::Index.new(dimensions: test_dims, metric: :cos)
          index.add(1_u64, [1.0_f32, 0.0_f32, 0.0_f32, 0.0_f32])
          index.add(2_u64, [0.0_f32, 1.0_f32, 0.0_f32, 0.0_f32])
          index.save(path)
          index.close

          # Load and verify
          loaded = USearch::Index.load(path, dimensions: test_dims, metric: :cos)
          loaded.size.should eq 2
          loaded.contains?(1_u64).should be_true
          loaded.contains?(2_u64).should be_true

          # Search should still work
          results = loaded.search([0.9_f32, 0.1_f32, 0.0_f32, 0.0_f32], k: 1)
          results[0].key.should eq 1_u64

          loaded.close
        ensure
          File.delete(path) if File.exists?(path)
        end
      end
    end

    describe ".view" do
      it "memory-maps an index from a file" do
        path = File.tempname("usearch_view", ".usearch")

        begin
          # Create and save
          index = USearch::Index.new(dimensions: test_dims, metric: :cos)
          index.add(1_u64, [1.0_f32, 0.0_f32, 0.0_f32, 0.0_f32])
          index.add(2_u64, [0.0_f32, 1.0_f32, 0.0_f32, 0.0_f32])
          index.save(path)
          index.close

          # View (memory-mapped)
          viewed = USearch::Index.view(path, dimensions: test_dims, metric: :cos)
          viewed.size.should eq 2
          viewed.contains?(1_u64).should be_true

          # Search should work
          results = viewed.search([0.9_f32, 0.1_f32, 0.0_f32, 0.0_f32], k: 1)
          results[0].key.should eq 1_u64

          viewed.close
        ensure
          File.delete(path) if File.exists?(path)
        end
      end
    end

    describe "#reserve" do
      it "pre-allocates capacity" do
        index = USearch::Index.new(dimensions: test_dims)
        index.reserve(1000)
        index.capacity.should be >= 1000
        index.close
      end
    end

    describe "#to_bytes and .from_bytes" do
      it "serializes and deserializes an index" do
        # Create and populate
        index = USearch::Index.new(dimensions: test_dims, metric: :cos)
        index.add(1_u64, [1.0_f32, 0.0_f32, 0.0_f32, 0.0_f32])
        index.add(2_u64, [0.0_f32, 1.0_f32, 0.0_f32, 0.0_f32])

        # Serialize
        bytes = index.to_bytes
        bytes.size.should be > 0
        index.close

        # Deserialize
        loaded = USearch::Index.from_bytes(bytes, dimensions: test_dims, metric: :cos)
        loaded.size.should eq 2
        loaded.contains?(1_u64).should be_true
        loaded.contains?(2_u64).should be_true

        # Search should work
        results = loaded.search([0.9_f32, 0.1_f32, 0.0_f32, 0.0_f32], k: 1)
        results[0].key.should eq 1_u64

        loaded.close
      end
    end

    describe ".view_bytes" do
      it "views an index from a buffer without copying" do
        # Create and populate
        index = USearch::Index.new(dimensions: test_dims, metric: :cos)
        index.add(1_u64, [1.0_f32, 0.0_f32, 0.0_f32, 0.0_f32])
        index.add(2_u64, [0.0_f32, 1.0_f32, 0.0_f32, 0.0_f32])

        # Serialize
        bytes = index.to_bytes
        index.close

        # View (zero-copy)
        viewed = USearch::Index.view_bytes(bytes, dimensions: test_dims, metric: :cos)
        viewed.size.should eq 2
        viewed.contains?(1_u64).should be_true

        # Search should work
        results = viewed.search([0.9_f32, 0.1_f32, 0.0_f32, 0.0_f32], k: 1)
        results[0].key.should eq 1_u64

        viewed.close
      end
    end

    describe "#serialized_length" do
      it "returns the buffer size needed" do
        index = USearch::Index.new(dimensions: test_dims)
        index.add(1_u64, [1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32])

        length = index.serialized_length
        length.should be > 0

        # Should match actual serialized size
        bytes = index.to_bytes
        bytes.size.should eq length

        index.close
      end
    end

    describe "#rename" do
      it "changes a vector's key" do
        index = USearch::Index.new(dimensions: test_dims)

        index.add(1_u64, [1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32])
        index.contains?(1_u64).should be_true
        index.contains?(100_u64).should be_false

        index.rename(1_u64, 100_u64)

        index.contains?(1_u64).should be_false
        index.contains?(100_u64).should be_true

        index.close
      end
    end

    describe "#close" do
      it "prevents operations after close" do
        index = USearch::Index.new(dimensions: test_dims)
        index.close
        index.closed?.should be_true

        expect_raises(USearch::Error, /closed/) do
          index.size
        end
      end

      it "is idempotent" do
        index = USearch::Index.new(dimensions: test_dims)
        index.close
        index.close  # Should not raise
        index.closed?.should be_true
      end
    end

    describe ".distance" do
      it "computes distance between vectors" do
        a = [1.0_f32, 0.0_f32, 0.0_f32, 0.0_f32]
        b = [0.0_f32, 1.0_f32, 0.0_f32, 0.0_f32]

        # Cosine distance between orthogonal vectors should be ~1.0
        dist = USearch::Index.distance(a, b, :cos)
        dist.should be_close(1.0, 0.01)

        # Same vector should have distance ~0
        dist = USearch::Index.distance(a, a, :cos)
        dist.should be_close(0.0, 0.01)
      end
    end

    describe ".version" do
      it "returns a version string" do
        version = USearch::Index.version
        version.should_not be_empty
      end
    end

    describe ".metadata" do
      it "reads metadata from a saved index file" do
        path = File.tempname("usearch_meta", ".usearch")

        begin
          index = USearch::Index.new(dimensions: 8, metric: :cos, quantization: :f16)
          index.add(1_u64, Array(Float32).new(8) { |i| i.to_f32 })
          index.save(path)
          index.close

          meta = USearch::Index.metadata(path)
          meta.dimensions.should eq 8
          # Note: metric/quantization may not be preserved exactly in all usearch versions
        ensure
          File.delete(path) if File.exists?(path)
        end
      end
    end

    describe ".metadata_buffer" do
      it "reads metadata from a serialized buffer" do
        index = USearch::Index.new(dimensions: 16, metric: :l2sq, quantization: :f32)
        index.add(1_u64, Array(Float32).new(16) { 1.0_f32 })
        bytes = index.to_bytes
        index.close

        meta = USearch::Index.metadata_buffer(bytes)
        meta.dimensions.should eq 16
      end
    end

    describe "#expansion_add" do
      it "gets and sets expansion_add" do
        index = USearch::Index.new(dimensions: 4, expansion_add: 64)
        index.expansion_add.should eq 64

        index.expansion_add = 256
        index.expansion_add.should eq 256

        index.close
      end
    end

    describe "#expansion_search" do
      it "gets and sets expansion_search" do
        index = USearch::Index.new(dimensions: 4, expansion_search: 32)
        index.expansion_search.should eq 32

        index.expansion_search = 128
        index.expansion_search.should eq 128

        index.close
      end
    end

    describe "#metric=" do
      it "changes the metric at runtime" do
        index = USearch::Index.new(dimensions: 4, metric: :cos)

        # Change to L2
        index.metric = :l2sq

        # Add vectors and search with new metric
        index.add(1_u64, [1.0_f32, 0.0_f32, 0.0_f32, 0.0_f32])
        index.add(2_u64, [0.0_f32, 1.0_f32, 0.0_f32, 0.0_f32])

        results = index.search([1.0_f32, 0.0_f32, 0.0_f32, 0.0_f32], k: 1)
        results[0].key.should eq 1_u64

        index.close
      end
    end

    describe "#set_custom_metric" do
      it "accepts a custom metric callback" do
        index = USearch::Index.new(dimensions: 2, metric: :cos)

        # Define a simple L2 distance callback
        callback = LibUSearch::MetricCallback.new do |a, b|
          # Simple L2 for 2D vectors (hardcoded for test)
          va = a.as(Pointer(Float32))
          vb = b.as(Pointer(Float32))
          dx = va[0] - vb[0]
          dy = va[1] - vb[1]
          Math.sqrt(dx * dx + dy * dy).to_f32
        end

        index.set_custom_metric(callback, :l2sq)

        # Add and search with custom metric
        index.add(1_u64, [0.0_f32, 0.0_f32])
        index.add(2_u64, [1.0_f32, 1.0_f32])

        results = index.search([0.1_f32, 0.1_f32], k: 2)
        results[0].key.should eq 1_u64  # Closer to origin

        index.close
      end
    end

    describe "#threads_add=" do
      it "sets the number of threads for add operations" do
        index = USearch::Index.new(dimensions: test_dims)
        index.threads_add = 2
        # Just verify it doesn't raise
        index.close
      end
    end

    describe "#threads_search=" do
      it "sets the number of threads for search operations" do
        index = USearch::Index.new(dimensions: test_dims)
        index.threads_search = 2
        # Just verify it doesn't raise
        index.close
      end
    end

    describe "#memory_usage" do
      it "returns memory usage in bytes" do
        index = USearch::Index.new(dimensions: test_dims)
        index.add(1_u64, [1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32])

        usage = index.memory_usage
        usage.should be > 0

        index.close
      end
    end

    describe "#connectivity" do
      it "returns the HNSW connectivity parameter" do
        index = USearch::Index.new(dimensions: test_dims, connectivity: 32)
        index.connectivity.should eq 32
        index.close
      end
    end

    describe "#dimensions" do
      it "returns the vector dimensionality" do
        index = USearch::Index.new(dimensions: 128)
        index.dimensions.should eq 128
        index.close
      end
    end

    describe "larger scale test" do
      it "handles many vectors" do
        dims = 128
        count = 1000
        index = USearch::Index.new(dimensions: dims, metric: :cos, quantization: :f16)

        # Add vectors
        count.times do |i|
          vec = Array(Float32).new(dims) { |j| ((i + j) % 100).to_f32 / 100.0_f32 }
          index.add(i.to_u64, vec)
        end

        index.size.should eq count

        # Search
        query = Array(Float32).new(dims) { |j| (j % 100).to_f32 / 100.0_f32 }
        results = index.search(query, k: 10)

        results.size.should eq 10
        # Just verify we got results - exact ordering depends on HNSW approximation
        results[0].distance.should be >= 0

        index.close
      end
    end
  end
end
