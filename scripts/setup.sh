#!/bin/bash
# Setup script for usearch.cr
# Clones and builds the usearch C library, at the release this shard is
# written against. USearch's main branch is 3.0 since 2026-11-10, which
# builds no C library, so the clone names a 2.x release.

set -e

USEARCH_VERSION="v2.26.6"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENDOR_DIR="$PROJECT_DIR/vendor"
USEARCH_DIR="$VENDOR_DIR/usearch"

echo "Setting up usearch.cr..."

# Create vendor directory
mkdir -p "$VENDOR_DIR"

# Clone usearch if not present
if [ ! -d "$USEARCH_DIR" ]; then
    echo "Cloning usearch $USEARCH_VERSION..."
    git clone --depth 1 --branch "$USEARCH_VERSION" https://github.com/unum-cloud/usearch.git "$USEARCH_DIR"
    cd "$USEARCH_DIR"
    git submodule update --init --recursive
else
    cd "$USEARCH_DIR"
    if [ "$(git rev-parse HEAD)" != "$(git rev-parse -q --verify "refs/tags/$USEARCH_VERSION^{commit}")" ]; then
        echo "usearch already cloned, moving it to $USEARCH_VERSION..."
        if [ -f "$(git rev-parse --git-dir)/shallow" ]; then
            git fetch --depth 1 origin tag "$USEARCH_VERSION"
        else
            git fetch origin tag "$USEARCH_VERSION"
        fi
        git checkout -q "$USEARCH_VERSION"
        # A build configured from other sources would mix the two
        rm -rf build
    fi
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

# cmake succeeds without the C library when its sources have no such
# target, so check for it here rather than at the first link
if [ ! -f build/libusearch_static_c.a ]; then
    echo "error: no build/libusearch_static_c.a after building usearch $USEARCH_VERSION" >&2
    exit 1
fi

echo ""
echo "Build complete!"
echo "  Static library: $USEARCH_DIR/build/libusearch_static_c.a"
echo "  Shared library: $USEARCH_DIR/build/libusearch_c.so"
echo ""
echo "You can now build your Crystal project with: crystal build"
