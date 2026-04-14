"""Minimal data package for smoke tests.

This provides lightweight implementations of the data generator helpers
used by `src.experiments.run_experiments` so smoke tests can run on the
cluster without external data artifacts.
"""

from .generators import sample_sphere, sample_torus, embed_via_random_orthonormal, add_orthogonal_noise

__all__ = ["sample_sphere", "sample_torus", "embed_via_random_orthonormal", "add_orthogonal_noise"]
