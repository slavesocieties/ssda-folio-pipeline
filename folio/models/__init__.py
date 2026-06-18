"""Learned model wrappers. Each lazily imports heavy deps and exposes a small,
batched, framework-agnostic interface so the orchestrator never touches torch
directly. All run fp16 + torch.compile when available."""
