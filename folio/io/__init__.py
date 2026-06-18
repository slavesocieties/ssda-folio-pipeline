"""Async, streaming S3 I/O. Bytes flow S3 -> RAM -> GPU -> S3 with bounded
queues for backpressure. No local-disk staging (the legacy bottleneck)."""
