#!/bin/bash
# Setup script for usearch.cr
# Clones and builds the usearch C library

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENDOR_DIR="$PROJECT_DIR/vendor"
USEARCH_DIR="$VENDOR_DIR/usearch"

echo "Setting up usearch.cr..."

# Create vendor directory
mkdir -p "$VENDOR_DIR"

# Clone usearch if not present
if [ ! -d "$USEARCH_DIR" ]; then
    echo "Cloning usearch..."
    git clone --depth 1 https://github.com/unum-cloud/usearch.git "$USEARCH_DIR"
    cd "$USEARCH_DIR"
    git submodule update --init --recursive
else
    echo "usearch already cloned, updating..."
    cd "$USEARCH_DIR"
    git pull --ff-only || true
    git submodule update --init --recursive
fi

# Build
echo "Building usearch C library..."
cd "$USEARCH_DIR"
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSEARCH_BUILD_LIB_C=ON \
    -DUSEARCH_BUILD_TEST_CPP=OFF \
    -DUSEARCH_BUILD_BENCH_CPP=OFF

cmake --build build --config Release -j$(nproc)

echo ""
echo "Build complete!"
echo "  Static library: $USEARCH_DIR/build/libusearch_static_c.a"
echo "  Shared library: $USEARCH_DIR/build/libusearch_c.so"
echo ""
echo "You can now build your Crystal project with: crystal build"
