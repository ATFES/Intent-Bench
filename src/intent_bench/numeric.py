from __future__ import annotations
from collections.abc import Iterable
import math
import torch
from .corpus._common.model import TensorTree, Tolerance, IntegerTolerance, SimilarityTolerance, NumericalTolerance

COMPARISON_CHUNK_ELEMENTS = 1048576

class NumericalComparisonError(AssertionError):
    pass

def _leaves(value: TensorTree) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
        return
    if not isinstance(value, tuple):
        raise TypeError(
            "V2 result trees contain only tensors and tuples, got "
            f"{type(value).__name__}"
        )
    for element in value:
        yield from _leaves(element)

def _result_structure(value: TensorTree):
    if isinstance(value, torch.Tensor):
        return "tensor"
    if not isinstance(value, tuple):
        raise TypeError(
            "V2 result trees contain only tensors and tuples, got "
            f"{type(value).__name__}"
        )
    return tuple(_result_structure(element) for element in value)

def compare_outputs(
    generated: TensorTree,
    source: TensorTree,
    tolerance: NumericalTolerance | tuple[NumericalTolerance, ...],
) -> tuple[float, ...]:
    generated_structure = _result_structure(generated)
    source_structure = _result_structure(source)
    if generated_structure != source_structure:
        raise NumericalComparisonError(
            "generated/source result structure differs: "
            f"{generated_structure!r} != {source_structure!r}"
        )
    generated_values = tuple(_leaves(generated))
    source_values = tuple(_leaves(source))
    if len(generated_values) != len(source_values):
        raise NumericalComparisonError(
            "generated/source result count differs: "
            f"{len(generated_values)} != {len(source_values)}"
        )
    tolerances = (
        tolerance
        if isinstance(tolerance, tuple)
        else tuple(tolerance for _ in generated_values)
    )
    if len(tolerances) != len(generated_values):
        raise NumericalComparisonError(
            "tolerance count differs from result count: "
            f"{len(tolerances)} != {len(generated_values)}"
        )
    errors: list[float] = []
    for leaf_index, (generated_value, source_value, leaf_tolerance) in enumerate(
        zip(generated_values, source_values, tolerances)
    ):
        if generated_value.shape != source_value.shape:
            raise NumericalComparisonError(
                f"generated/source result {leaf_index} shape differs: "
                f"{tuple(generated_value.shape)} != {tuple(source_value.shape)}"
            )
        if generated_value.dtype != source_value.dtype:
            raise NumericalComparisonError(
                f"generated/source result {leaf_index} dtype differs: "
                f"{generated_value.dtype} != {source_value.dtype}"
            )
        if generated_value.device != source_value.device:
            raise NumericalComparisonError(
                f"generated/source result {leaf_index} device differs: "
                f"{generated_value.device} != {source_value.device}"
            )
        if isinstance(leaf_tolerance, IntegerTolerance):
            if generated_value.dtype not in (torch.int8, torch.uint8, torch.int16, torch.int32):
                raise TypeError("integer absolute comparison requires at most 32-bit integer outputs")
            actual, expected = generated_value.reshape(-1), source_value.reshape(-1)
            maximum = 0
            for begin in range(0, actual.numel(), COMPARISON_CHUNK_ELEMENTS):
                end = begin + COMPARISON_CHUNK_ELEMENTS
                difference = (actual[begin:end].long() - expected[begin:end].long()).abs()
                maximum = max(maximum, difference.max().item())
            if maximum > leaf_tolerance.max_abs:
                raise NumericalComparisonError(
                    f"generated/source integer result {leaf_index} differs: max_abs={maximum}, limit={leaf_tolerance.max_abs}"
                )
            errors.append(float(maximum))
            continue
        if isinstance(leaf_tolerance, SimilarityTolerance):
            if not generated_value.is_floating_point():
                raise TypeError("similarity comparison requires floating outputs")
            actual, expected = generated_value.double(), source_value.double()
            if not (torch.isfinite(actual).all() and torch.isfinite(expected).all()):
                raise NumericalComparisonError(f"similarity comparison requires finite result {leaf_index}")
            denominator = (actual * actual + expected * expected).sum()
            error = 0.0 if denominator.item() == 0.0 else abs(1.0 - (2.0 * (actual * expected).sum() / denominator).item())
            if not math.isfinite(error) or error > leaf_tolerance.max_error:
                raise NumericalComparisonError(
                    f"generated/source result {leaf_index} differs: similarity_error={error}, limit={leaf_tolerance.max_error}"
                )
            errors.append(error)
            continue
        if generated_value.dtype == torch.bool or not generated_value.is_floating_point():
            if not torch.equal(generated_value, source_value):
                raise NumericalComparisonError(
                    f"generated/source integer result {leaf_index} differs"
                )
            errors.append(0.0)
            continue
        generated_flat = generated_value.reshape(-1)
        source_flat = source_value.reshape(-1)
        maximum = 0.0
        for begin in range(0, generated_flat.numel(), COMPARISON_CHUNK_ELEMENTS):
            end = begin + COMPARISON_CHUNK_ELEMENTS
            maximum = max(
                maximum,
                _compare_float_chunk(
                    generated_flat[begin:end], source_flat[begin:end],
                    leaf_tolerance, leaf_index, begin,
                ),
            )
        errors.append(maximum)
    return tuple(errors)

def _compare_float_chunk(
    generated: torch.Tensor,
    source: torch.Tensor,
    tolerance: Tolerance,
    leaf_index: int,
    begin: int,
) -> float:
    comparison_dtype = torch.float64 if source.dtype == torch.float64 else torch.float32
    generated_compare = generated.to(comparison_dtype)
    source_compare = source.to(comparison_dtype)
    generated_finite = torch.isfinite(generated_compare)
    source_finite = torch.isfinite(source_compare)
    if not torch.equal(generated_finite, source_finite):
        raise NumericalComparisonError(
            f"generated/source result {leaf_index} finite-value masks differ"
        )
    finite = generated_finite & source_finite
    nonfinite = ~finite
    if nonfinite.any():
        generated_nonfinite = generated_compare[nonfinite]
        source_nonfinite = source_compare[nonfinite]
        if not (
            torch.equal(
                torch.isnan(generated_nonfinite),
                torch.isnan(source_nonfinite),
            )
            and torch.equal(
                torch.isposinf(generated_nonfinite),
                torch.isposinf(source_nonfinite),
            )
            and torch.equal(
                torch.isneginf(generated_nonfinite),
                torch.isneginf(source_nonfinite),
            )
        ):
            raise NumericalComparisonError(
                f"generated/source result {leaf_index} non-finite values differ"
            )
    if finite.any():
        difference = (generated_compare[finite] - source_compare[finite]).abs()
        limit = tolerance.atol + tolerance.rtol * source_compare[finite].abs()
        maximum = difference.max().item()
        if torch.any(difference > limit):
            worst = (difference - limit).argmax()
            first_failure = torch.nonzero(difference > limit)[0, 0]
            first_position = torch.nonzero(finite)[first_failure, 0].item()
            raise NumericalComparisonError(
                f"generated/source floating result {leaf_index} differs: "
                f"max_abs={maximum}, atol={tolerance.atol}, "
                f"rtol={tolerance.rtol}; first_failure={begin + first_position}; "
                f"largest tolerance excess has "
                f"generated={generated_compare[finite][worst].item()}, "
                f"source={source_compare[finite][worst].item()}, "
                f"abs_error={difference[worst].item()}, limit={limit[worst].item()}"
            )
        return maximum
    return 0.0
