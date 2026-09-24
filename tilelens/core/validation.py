"""Shared validation for discrete GEMM dimensions and tile settings."""

from numbers import Integral


def require_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
