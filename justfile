# Layercast: NIXL GPUDirect Model Weight Distribution for Kubernetes

# Core
build:
    cargo build --workspace

build-release:
    cargo build --workspace --release

fmt:
    cargo fmt --all

clippy:
    cargo clippy --all --benches --tests --examples --all-features

test:
    cargo test --workspace

test-unit:
    cargo test --workspace --lib

test-integration:
    cargo test --workspace --test '*'

check:
    cargo fmt --all -- --check
    cargo clippy --all --benches --tests --examples --all-features -- -D warnings
    cargo test --workspace

# CRDs
crd-gen:
    cargo run -p discovery --bin crd-gen > deploy/crds/nodecache.yaml

kopium-inferencepool:
    kopium -f deploy/external-crds/inferencepool.yaml --docs > crates/discovery/src/inference_pool.rs
    cargo fmt -p discovery

# Benchmark: model load latency across HF, NFS, NIXL
benchmark *ARGS:
    ./scripts/benchmark.sh {{ARGS}}

benchmark-cleanup:
    ./scripts/benchmark.sh --cleanup

# Python plugin tests
test-plugin:
    cd vllm-plugin && uv run pytest tests/ -v

# Container images
build-model-mesh:
    docker build -t model-mesh:dev -f Containerfile.model-mesh .

build-vllm-plugin:
    docker build -t vllm-plugin:dev -f Containerfile.vllm-plugin .
