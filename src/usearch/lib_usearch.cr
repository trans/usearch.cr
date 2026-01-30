# Low-level FFI bindings for libusearch.
#
# This module provides direct bindings to the USearch C API.
# For most use cases, prefer the high-level `USearch::Index` class.

@[Link("usearch_c")]
lib LibUSearch
  # Opaque handle to a USearch index.
  type Index = Void*

  # Vector key type (ID for each vector).
  alias Key = UInt64

  # Distance/similarity value.
  alias Distance = Float32

  # Error string (null if no error).
  alias Error = LibC::Char*

  # Scalar/vector element types.
  enum ScalarKind : UInt32
    Unknown = 0
    F32     = 1   # 32-bit float
    F64     = 2   # 64-bit float
    F16     = 3   # 16-bit float (half precision)
    I8      = 4   # 8-bit signed integer
    B1      = 5   # 1-bit binary
    BF16    = 6   # Brain float 16
  end

  # Distance/similarity metrics.
  enum MetricKind : UInt32
    Unknown    = 0
    Cos        = 1   # Cosine similarity
    IP         = 2   # Inner product
    L2sq       = 3   # Squared Euclidean (L2)
    Haversine  = 4   # Geographic distance
    Divergence = 5   # Jensen-Shannon divergence
    Pearson    = 6   # Pearson correlation
    Jaccard    = 7   # Jaccard index
    Hamming    = 8   # Hamming distance (for binary)
    Tanimoto   = 9   # Tanimoto coefficient
    Sorensen   = 10  # Sorensen-Dice coefficient
  end

  # Index initialization options.
  struct InitOptions
    metric_kind : MetricKind
    metric : Void*             # Custom metric function pointer (can be null)
    quantization : ScalarKind
    dimensions : LibC::SizeT
    connectivity : LibC::SizeT
    expansion_add : LibC::SizeT
    expansion_search : LibC::SizeT
    multi : Bool
  end

  # Version info
  fun version = usearch_version : LibC::Char*

  # Lifecycle
  fun init = usearch_init(options : InitOptions*, error : Error*) : Index
  fun free = usearch_free(index : Index, error : Error*) : Void

  # Persistence
  fun save = usearch_save(index : Index, path : LibC::Char*, error : Error*) : Void
  fun load = usearch_load(index : Index, path : LibC::Char*, error : Error*) : Void
  fun view = usearch_view(index : Index, path : LibC::Char*, error : Error*) : Void

  # Stats
  fun size = usearch_size(index : Index, error : Error*) : LibC::SizeT
  fun capacity = usearch_capacity(index : Index, error : Error*) : LibC::SizeT
  fun dimensions = usearch_dimensions(index : Index, error : Error*) : LibC::SizeT
  fun connectivity = usearch_connectivity(index : Index, error : Error*) : LibC::SizeT
  fun memory_usage = usearch_memory_usage(index : Index, error : Error*) : LibC::SizeT
  fun serialized_length = usearch_serialized_length(index : Index, error : Error*) : LibC::SizeT

  # Configuration
  fun reserve = usearch_reserve(index : Index, capacity : LibC::SizeT, error : Error*) : Void
  fun change_expansion_add = usearch_change_expansion_add(index : Index, expansion : LibC::SizeT, error : Error*) : Void
  fun change_expansion_search = usearch_change_expansion_search(index : Index, expansion : LibC::SizeT, error : Error*) : Void
  fun change_threads_add = usearch_change_threads_add(index : Index, threads : LibC::SizeT, error : Error*) : Void
  fun change_threads_search = usearch_change_threads_search(index : Index, threads : LibC::SizeT, error : Error*) : Void

  # Data operations
  fun add = usearch_add(
    index : Index,
    key : Key,
    vector : Void*,
    kind : ScalarKind,
    error : Error*
  ) : Void

  fun get = usearch_get(
    index : Index,
    key : Key,
    count : LibC::SizeT,
    vector : Void*,
    kind : ScalarKind,
    error : Error*
  ) : LibC::SizeT

  fun remove = usearch_remove(index : Index, key : Key, error : Error*) : Void
  fun rename = usearch_rename(index : Index, from : Key, to : Key, error : Error*) : Void
  fun contains = usearch_contains(index : Index, key : Key, error : Error*) : Bool
  fun count = usearch_count(index : Index, key : Key, error : Error*) : LibC::SizeT
  fun clear = usearch_clear(index : Index, error : Error*) : Void

  # Search
  fun search = usearch_search(
    index : Index,
    query : Void*,
    query_kind : ScalarKind,
    count : LibC::SizeT,
    keys : Key*,
    distances : Distance*,
    error : Error*
  ) : LibC::SizeT

  # Distance computation (standalone)
  fun distance = usearch_distance(
    vector_a : Void*,
    vector_b : Void*,
    kind : ScalarKind,
    dimensions : LibC::SizeT,
    metric : MetricKind,
    error : Error*
  ) : Distance
end
