# USearch Crystal bindings - Fast approximate nearest neighbor search.
#
# USearch is a high-performance vector search library using HNSW
# (Hierarchical Navigable Small World) graphs.
#
# ## Example
#
# ```crystal
# require "usearch"
#
# # Create an index
# index = USearch::Index.new(dimensions: 128, metric: :cos, quantization: :f16)
#
# # Add vectors
# index.add(1_u64, [0.1_f32, 0.2_f32, ...])
# index.add(2_u64, [0.3_f32, 0.4_f32, ...])
#
# # Search
# results = index.search(query_vector, k: 10)
# results.each do |r|
#   puts "Key: #{r.key}, Distance: #{r.distance}"
# end
#
# # Save to disk
# index.save("vectors.usearch")
#
# # Load later
# index = USearch::Index.load("vectors.usearch", dimensions: 128)
# ```

require "./usearch/lib_usearch"

module USearch
  VERSION = "0.1.0"

  # Alias for scalar types
  alias ScalarKind = LibUSearch::ScalarKind

  # Alias for metric types
  alias MetricKind = LibUSearch::MetricKind

  # Error raised when USearch operations fail.
  class Error < Exception
  end

  # Result of a nearest neighbor search.
  record SearchResult, key : UInt64, distance : Float32

  # High-level wrapper for a USearch HNSW index.
  #
  # The index stores vectors and allows fast approximate nearest neighbor search.
  # Each vector is associated with a 64-bit key (typically your database row ID).
  class Index
    # Default HNSW connectivity parameter (edges per node).
    DEFAULT_CONNECTIVITY = 16_u64

    # Default expansion factor during index construction.
    DEFAULT_EXPANSION_ADD = 128_u64

    # Default expansion factor during search.
    DEFAULT_EXPANSION_SEARCH = 64_u64

    @handle : LibUSearch::Index
    @dimensions : UInt64
    @closed : Bool = false
    @reserved : Bool = false

    # Creates a new empty index.
    #
    # - `dimensions`: Vector dimensionality (must match all vectors added)
    # - `metric`: Distance metric (default: cosine similarity)
    # - `quantization`: Storage precision (default: f16 for memory efficiency)
    # - `connectivity`: HNSW M parameter (higher = more accurate, more memory)
    # - `expansion_add`: ef_construction parameter
    # - `expansion_search`: ef_search parameter
    # - `multi`: Allow multiple vectors per key
    def initialize(
      dimensions : Int,
      metric : MetricKind = :cos,
      quantization : ScalarKind = :f16,
      connectivity : Int = DEFAULT_CONNECTIVITY,
      expansion_add : Int = DEFAULT_EXPANSION_ADD,
      expansion_search : Int = DEFAULT_EXPANSION_SEARCH,
      multi : Bool = false
    )
      @dimensions = dimensions.to_u64

      options = LibUSearch::InitOptions.new
      options.metric_kind = metric
      options.metric = Pointer(Void).null
      options.quantization = quantization
      options.dimensions = @dimensions
      options.connectivity = connectivity.to_u64
      options.expansion_add = expansion_add.to_u64
      options.expansion_search = expansion_search.to_u64
      options.multi = multi

      error = Pointer(LibC::Char).null
      @handle = LibUSearch.init(pointerof(options), pointerof(error))
      check_error(error)
    end

    # Creates an index wrapper from a raw handle (used internally).
    private def initialize(@handle : LibUSearch::Index, @dimensions : UInt64)
    end

    # Loads an index from a file.
    #
    # The file must have been saved with `#save`. You must provide the
    # dimensions since they're needed to create the index before loading.
    def self.load(path : String, dimensions : Int, metric : MetricKind = :cos, quantization : ScalarKind = :f16) : Index
      index = new(dimensions, metric, quantization)
      error = Pointer(LibC::Char).null
      LibUSearch.load(index.@handle, path, pointerof(error))
      index.check_error(error)
      index
    end

    # Memory-maps an index from a file (read-only, memory efficient).
    #
    # The index is not loaded into memory but accessed directly from disk.
    # This is useful for large indexes that don't fit in RAM.
    def self.view(path : String, dimensions : Int, metric : MetricKind = :cos, quantization : ScalarKind = :f16) : Index
      index = new(dimensions, metric, quantization)
      error = Pointer(LibC::Char).null
      LibUSearch.view(index.@handle, path, pointerof(error))
      index.check_error(error)
      index
    end

    # Adds a vector to the index.
    #
    # - `key`: Unique identifier for this vector (e.g., database row ID)
    # - `vector`: The vector data (must match index dimensions)
    #
    # Vectors are passed as Float32 and converted to the index's quantization format.
    def add(key : UInt64, vector : Array(Float32) | Slice(Float32))
      check_open
      raise Error.new("Vector dimension mismatch: expected #{@dimensions}, got #{vector.size}") if vector.size != @dimensions

      # Auto-reserve on first add if not already reserved
      ensure_reserved

      error = Pointer(LibC::Char).null
      LibUSearch.add(@handle, key, vector.to_unsafe.as(Void*), LibUSearch::ScalarKind::F32, pointerof(error))
      check_error(error)
    end

    # Ensures the index has reserved space (required before adding vectors).
    private def ensure_reserved
      return if @reserved
      reserve(1024)  # Default initial capacity
      @reserved = true
    end

    # Searches for the k nearest neighbors to a query vector.
    #
    # - `query`: The query vector (must match index dimensions)
    # - `k`: Number of neighbors to return (default: 10)
    #
    # Returns an array of `SearchResult` with keys and distances, sorted by distance.
    def search(query : Array(Float32) | Slice(Float32), k : Int = 10) : Array(SearchResult)
      check_open
      raise Error.new("Query dimension mismatch: expected #{@dimensions}, got #{query.size}") if query.size != @dimensions

      keys = Slice(UInt64).new(k, 0_u64)
      distances = Slice(Float32).new(k, 0_f32)
      error = Pointer(LibC::Char).null

      found = LibUSearch.search(
        @handle,
        query.to_unsafe.as(Void*),
        LibUSearch::ScalarKind::F32,
        k.to_u64,
        keys.to_unsafe,
        distances.to_unsafe,
        pointerof(error)
      )
      check_error(error)

      Array(SearchResult).new(found.to_i) do |i|
        SearchResult.new(keys[i], distances[i])
      end
    end

    # Searches for the k nearest neighbors with a filter predicate.
    #
    # - `query`: The query vector (must match index dimensions)
    # - `k`: Maximum number of neighbors to return
    # - `&filter`: Block that receives a key and returns true to include it
    #
    # Example:
    # ```
    # # Only return vectors with even keys
    # results = index.filtered_search(query, k: 10) { |key| key.even? }
    #
    # # Only return vectors in a specific set
    # valid_ids = Set{1_u64, 5_u64, 10_u64}
    # results = index.filtered_search(query, k: 10) { |key| valid_ids.includes?(key) }
    # ```
    def filtered_search(query : Array(Float32) | Slice(Float32), k : Int = 10, &filter : UInt64 -> Bool) : Array(SearchResult)
      check_open
      raise Error.new("Query dimension mismatch: expected #{@dimensions}, got #{query.size}") if query.size != @dimensions

      keys = Slice(UInt64).new(k, 0_u64)
      distances = Slice(Float32).new(k, 0_f32)
      error = Pointer(LibC::Char).null

      # Box the proc so we can pass it through void*
      boxed_filter = Box.box(filter)

      # C callback that unboxes and calls the Crystal proc
      callback = LibUSearch::FilterCallback.new do |key, state|
        proc = Box(typeof(filter)).unbox(state)
        proc.call(key) ? 1 : 0
      end

      found = LibUSearch.filtered_search(
        @handle,
        query.to_unsafe.as(Void*),
        LibUSearch::ScalarKind::F32,
        k.to_u64,
        callback,
        boxed_filter,
        keys.to_unsafe,
        distances.to_unsafe,
        pointerof(error)
      )
      check_error(error)

      Array(SearchResult).new(found.to_i) do |i|
        SearchResult.new(keys[i], distances[i])
      end
    end

    # Removes a vector by key.
    def remove(key : UInt64)
      check_open
      error = Pointer(LibC::Char).null
      LibUSearch.remove(@handle, key, pointerof(error))
      check_error(error)
    end

    # Checks if a key exists in the index.
    def contains?(key : UInt64) : Bool
      check_open
      error = Pointer(LibC::Char).null
      result = LibUSearch.contains(@handle, key, pointerof(error))
      check_error(error)
      result
    end

    # Renames a key (changes the ID associated with a vector).
    def rename(from : UInt64, to : UInt64)
      check_open
      error = Pointer(LibC::Char).null
      LibUSearch.rename(@handle, from, to, pointerof(error))
      check_error(error)
    end

    # Clears all vectors from the index.
    def clear
      check_open
      error = Pointer(LibC::Char).null
      LibUSearch.clear(@handle, pointerof(error))
      check_error(error)
    end

    # Saves the index to a file.
    def save(path : String)
      check_open
      error = Pointer(LibC::Char).null
      LibUSearch.save(@handle, path, pointerof(error))
      check_error(error)
    end

    # Returns the number of vectors in the index.
    def size : UInt64
      check_open
      error = Pointer(LibC::Char).null
      result = LibUSearch.size(@handle, pointerof(error))
      check_error(error)
      result.to_u64
    end

    # Returns the current capacity (number of vectors that fit without reallocation).
    def capacity : UInt64
      check_open
      error = Pointer(LibC::Char).null
      result = LibUSearch.capacity(@handle, pointerof(error))
      check_error(error)
      result.to_u64
    end

    # Returns the vector dimensionality.
    def dimensions : UInt64
      @dimensions
    end

    # Returns memory usage in bytes.
    def memory_usage : UInt64
      check_open
      error = Pointer(LibC::Char).null
      result = LibUSearch.memory_usage(@handle, pointerof(error))
      check_error(error)
      result.to_u64
    end

    # Pre-allocates space for the given number of vectors.
    def reserve(capacity : Int)
      check_open
      error = Pointer(LibC::Char).null
      LibUSearch.reserve(@handle, capacity.to_u64, pointerof(error))
      check_error(error)
      @reserved = true
    end

    # Sets the expansion factor for search operations.
    #
    # Higher values = more accurate but slower. Default is 64.
    def expansion_search=(value : Int)
      check_open
      error = Pointer(LibC::Char).null
      LibUSearch.change_expansion_search(@handle, value.to_u64, pointerof(error))
      check_error(error)
    end

    # Sets the number of threads for add operations.
    def threads_add=(value : Int)
      check_open
      error = Pointer(LibC::Char).null
      LibUSearch.change_threads_add(@handle, value.to_u64, pointerof(error))
      check_error(error)
    end

    # Sets the number of threads for search operations.
    def threads_search=(value : Int)
      check_open
      error = Pointer(LibC::Char).null
      LibUSearch.change_threads_search(@handle, value.to_u64, pointerof(error))
      check_error(error)
    end

    # Closes the index and frees resources.
    def close
      return if @closed
      error = Pointer(LibC::Char).null
      LibUSearch.free(@handle, pointerof(error))
      @closed = true
      check_error(error)
    end

    # Returns true if the index has been closed.
    def closed? : Bool
      @closed
    end

    # Ensures the index is closed when garbage collected.
    def finalize
      close
    end

    # Returns the library version.
    def self.version : String
      String.new(LibUSearch.version)
    end

    # Computes distance between two vectors without an index.
    def self.distance(
      a : Array(Float32) | Slice(Float32),
      b : Array(Float32) | Slice(Float32),
      metric : MetricKind = :cos
    ) : Float32
      raise Error.new("Vector dimension mismatch") if a.size != b.size

      error = Pointer(LibC::Char).null
      result = LibUSearch.distance(
        a.to_unsafe.as(Void*),
        b.to_unsafe.as(Void*),
        LibUSearch::ScalarKind::F32,
        a.size.to_u64,
        metric,
        pointerof(error)
      )
      check_error_static(error)
      result
    end

    private def check_open
      raise Error.new("Index is closed") if @closed
    end

    protected def check_error(error : LibC::Char*)
      if error && !error.null?
        message = String.new(error)
        raise Error.new(message) unless message.empty?
      end
    end

    private def self.check_error_static(error : LibC::Char*)
      if error && !error.null?
        message = String.new(error)
        raise Error.new(message) unless message.empty?
      end
    end
  end
end
