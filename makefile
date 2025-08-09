export RUSTFLAGS = -C target-feature=+avx

build:
	cargo build --release