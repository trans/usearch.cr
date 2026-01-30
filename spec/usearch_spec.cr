require "spec"
require "../src/usearch"

describe USearch do
  describe "VERSION" do
    it "has a version" do
      USearch::VERSION.should_not be_nil
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

    describe "#reserve" do
      it "pre-allocates capacity" do
        index = USearch::Index.new(dimensions: test_dims)
        index.reserve(1000)
        index.capacity.should be >= 1000
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
