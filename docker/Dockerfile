FROM rust:1-bookworm AS builder
RUN apt-get update && apt-get install -y --no-install-recommends protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
RUN cargo build --release --workspace

FROM debian:bookworm-slim AS runtime
RUN useradd --system --uid 10001 --create-home appuser
COPY --from=builder /app/target/release/router /usr/local/bin/router
COPY --from=builder /app/target/release/mock-triton /usr/local/bin/mock-triton
COPY --from=builder /app/target/release/loadgen /usr/local/bin/loadgen
COPY data/tokenizer.json /data/tokenizer.json
USER appuser
ENV TOKENIZER_PATH=/data/tokenizer.json
ENTRYPOINT ["/usr/local/bin/router"]
