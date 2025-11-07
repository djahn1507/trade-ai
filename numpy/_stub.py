"""Minimal numpy stub used for the project test-suite.

This is a deliberately tiny, pure Python implementation that mimics just
enough of NumPy to satisfy the requirements of the unit tests.  It should
not be used as a general purpose numerical library.
"""
from __future__ import annotations

import builtins as _builtins
import math
import random as _stdlib_random
from typing import Any, Iterable, List, Sequence, Tuple

__version__ = "0.0-teststub"

Number = float
bool_ = bool
float64 = float
int64 = int


class NDArray:
    """A light-weight multi dimensional array backed by nested Python lists."""

    __slots__ = ("_data", "shape", "ndim", "dtype")

    def __init__(self, data: Any, dtype: type | None = None):
        if isinstance(data, NDArray):
            self._data = _deep_copy(data._data)
            self.shape = data.shape
            self.ndim = data.ndim
            self.dtype = data.dtype if dtype is None else dtype
            return

        nested = _to_nested_list(data)
        self.shape = _infer_shape(nested)
        self.ndim = len(self.shape)
        self.dtype = dtype or _infer_dtype(nested)
        self._data = _convert_dtype(nested, self.dtype)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"NDArray(shape={self.shape}, dtype={self.dtype.__name__})"

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------
    def tolist(self) -> Any:
        return _deep_copy(self._data)

    def copy(self) -> "NDArray":
        return NDArray(self)

    def __len__(self) -> int:
        return self.shape[0] if self.ndim else 0

    def __setitem__(self, index: Any, value: Any) -> None:
        index = _normalize_index(index, self.ndim)
        _assign_in_nested(self._data, index, value)

    # ------------------------------------------------------------------
    # Indexing and slicing
    # ------------------------------------------------------------------
    def __getitem__(self, index: Any) -> Any:
        index = _normalize_index(index, self.ndim)
        data = _slice_data(self._data, index)
        if _is_scalar_shape(_infer_shape(data)):
            return data
        return NDArray(data, dtype=self.dtype)

    # ------------------------------------------------------------------
    # Array manipulations
    # ------------------------------------------------------------------
    def reshape(self, *shape: int) -> "NDArray":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])  # type: ignore[assignment]
        if _product(shape) != _product(self.shape):
            raise ValueError("total size of new array must be unchanged")
        flat = _flatten(self._data)
        return NDArray(_unflatten(flat, shape), dtype=self.dtype)

    # ------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------
    def sum(self, axis: int | None = None) -> Any:
        if axis is None:
            return _builtins.sum(_flatten(self._data))
        return _aggregate_over_axis(self, axis, lambda values: _builtins.sum(values))

    def mean(self, axis: int | None = None, keepdims: bool = False) -> "NDArray" | float:
        if axis is None:
            values = _flatten(self._data)
            return _builtins.sum(values) / len(values) if values else 0.0
        aggregated = _aggregate_over_axis(
            self,
            axis,
            lambda values: _builtins.sum(values) / len(values) if values else 0.0,
        )
        if keepdims:
            target_shape = _insert_dim(self.shape, axis % self.ndim)
            return broadcast_to(aggregated, target_shape)
        return aggregated

    def std(self, axis: int | None = None, keepdims: bool = False) -> "NDArray" | float:
        if axis is None:
            values = _flatten(self._data)
            if not values:
                return 0.0
            mean_val = _builtins.sum(values) / len(values)
            variance = _builtins.sum((v - mean_val) ** 2 for v in values) / len(values)
            return math.sqrt(variance)
        aggregated = _aggregate_over_axis(
            self,
            axis,
            lambda values: (
                math.sqrt(
                    _builtins.sum((v - (_builtins.sum(values) / len(values))) ** 2 for v in values)
                    / len(values)
                )
                if values
                else 0.0
            ),
        )
        if keepdims:
            target_shape = _insert_dim(self.shape, axis % self.ndim)
            return broadcast_to(aggregated, target_shape)
        return aggregated

    # ------------------------------------------------------------------
    # Element-wise operations
    # ------------------------------------------------------------------
    def _binary_op(self, other: Any, op) -> "NDArray":
        other_arr = asarray(other)
        result_shape = _broadcast_shape(self.shape, other_arr.shape)
        left = broadcast_to(self, result_shape)
        right = broadcast_to(other_arr, result_shape)
        data = _elementwise_op(left._data, right._data, op)
        return NDArray(data, dtype=self.dtype)

    def __add__(self, other: Any) -> "NDArray":
        return self._binary_op(other, lambda a, b: a + b)

    def __radd__(self, other: Any) -> "NDArray":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "NDArray":
        return self._binary_op(other, lambda a, b: a - b)

    def __rsub__(self, other: Any) -> "NDArray":
        return asarray(other)._binary_op(self, lambda a, b: a - b)

    def __mul__(self, other: Any) -> "NDArray":
        return self._binary_op(other, lambda a, b: a * b)

    def __rmul__(self, other: Any) -> "NDArray":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "NDArray":
        return self._binary_op(other, lambda a, b: a / b)

    def __rtruediv__(self, other: Any) -> "NDArray":
        return asarray(other)._binary_op(self, lambda a, b: a / b)

    def __pow__(self, exponent: Any) -> "NDArray":
        return self._binary_op(exponent, lambda a, b: a ** b)

    def __neg__(self) -> "NDArray":
        return NDArray(_apply_unary(self._data, lambda x: -x), dtype=self.dtype)

    # Support numpy-like attributes
    @property
    def T(self) -> "NDArray":  # pragma: no cover - unused in tests
        return transpose(self)


ndarray = NDArray


# =============================================================================
# Helper utilities
# =============================================================================

def _to_nested_list(data: Any) -> Any:
    if isinstance(data, (list, tuple)):
        return [_to_nested_list(item) for item in data]
    return data


def _infer_shape(data: Any) -> Tuple[int, ...]:
    if isinstance(data, list):
        if not data:
            return (0,)
        inner_shape = _infer_shape(data[0])
        for item in data[1:]:
            if _infer_shape(item) != inner_shape:
                raise ValueError("Inconsistent shapes in nested data")
        return (len(data),) + inner_shape
    return ()


def _is_scalar_shape(shape: Tuple[int, ...]) -> bool:
    return shape == ()


def _flatten(data: Any) -> List[Number]:
    if isinstance(data, (list, tuple)):
        result: List[Number] = []
        for item in data:
            result.extend(_flatten(item))
        return result
    return [data]


def _unflatten(flat: List[Number], shape: Tuple[int, ...]) -> Any:
    if not shape:
        return flat[0]
    size = shape[0]
    chunk = _product(shape[1:])
    result = []
    for i in range(size):
        start = i * chunk
        end = start + chunk
        result.append(_unflatten(flat[start:end], shape[1:]))
    return result


def _product(shape: Sequence[int]) -> int:
    prod = 1
    for value in shape:
        prod *= value
    return prod


def _deep_copy(data: Any) -> Any:
    if isinstance(data, list):
        return [_deep_copy(item) for item in data]
    return data


def _normalize_index(index: Any, ndim: int) -> Tuple[Any, ...]:
    if index is Ellipsis:
        return tuple(slice(None) for _ in range(ndim))
    if not isinstance(index, tuple):
        index = (index,)
    result = []
    extra_axes = ndim
    for item in index:
        if item is None:
            result.append(None)
        else:
            extra_axes -= 1
            result.append(item)
    result.extend(slice(None) for _ in range(extra_axes))
    return tuple(result)


def _slice_data(data: Any, indices: Tuple[Any, ...]) -> Any:
    if not indices:
        return data
    index = indices[0]
    if index is None:
        return [_slice_data(data, indices[1:])]
    if isinstance(index, slice):
        if not isinstance(data, list):
            data = [data]
        sliced = data[index]
        return [_slice_data(item, indices[1:]) for item in sliced]
    if isinstance(index, int):
        return _slice_data(data[index], indices[1:])
    raise TypeError("Unsupported index type")


def _infer_dtype(data: Any) -> type:
    if isinstance(data, list) and data:
        return _infer_dtype(data[0])
    return float if not isinstance(data, str) else str


def _convert_dtype(data: Any, dtype: type) -> Any:
    if isinstance(data, list):
        return [_convert_dtype(item, dtype) for item in data]
    try:
        return dtype(data)
    except (TypeError, ValueError):
        return data


def _broadcast_shape(shape_a: Tuple[int, ...], shape_b: Tuple[int, ...]) -> Tuple[int, ...]:
    result = []
    for a, b in zip(_pad_shape(shape_a, shape_b), _pad_shape(shape_b, shape_a)):
        if a == b:
            result.append(a)
        elif a == 1:
            result.append(b)
        elif b == 1:
            result.append(a)
        else:
            raise ValueError("operands could not be broadcast together")
    return tuple(result)


def _pad_shape(shape: Tuple[int, ...], reference: Tuple[int, ...]) -> Tuple[int, ...]:
    padding = (1,) * (len(reference) - len(shape))
    return padding + shape


def broadcast_to(array: Any, shape: Tuple[int, ...]) -> NDArray:
    arr = asarray(array)
    padded_shape = _pad_shape(arr.shape, shape)
    result = zeros(shape, dtype=arr.dtype)

    for idx in _iter_indices(shape):
        source_idx = []
        for dim_size, idx_value in zip(padded_shape, idx):
            if dim_size == 1:
                source_idx.append(0)
            else:
                source_idx.append(idx_value)
        arr_indices = tuple(source_idx[-arr.ndim :]) if arr.ndim else ()
        value = _get_from_nested(arr._data, arr_indices) if arr.ndim else arr._data
        _set_in_nested(result._data, idx, value)

    return result


def _elementwise_op(a: Any, b: Any, op) -> Any:
    if isinstance(a, list) and isinstance(b, list):
        return [_elementwise_op(x, y, op) for x, y in zip(a, b)]
    if isinstance(a, list):
        return [_elementwise_op(x, b, op) for x in a]
    if isinstance(b, list):
        return [_elementwise_op(a, y, op) for y in b]
    return op(a, b)


def _apply_unary(data: Any, op) -> Any:
    if isinstance(data, list):
        return [_apply_unary(item, op) for item in data]
    return op(data)


def asarray(obj: Any, dtype: type | None = None) -> NDArray:
    if isinstance(obj, NDArray):
        return obj if dtype is None else NDArray(obj, dtype=dtype)
    return NDArray(obj, dtype=dtype)


array = asarray


def zeros(shape: Tuple[int, ...], dtype: type = float) -> NDArray:
    if isinstance(shape, int):
        shape = (shape,)
    return NDArray(_fill_shape(shape, dtype(0)), dtype=dtype)


def ones(shape: Tuple[int, ...], dtype: type = float) -> NDArray:
    if isinstance(shape, int):
        shape = (shape,)
    return NDArray(_fill_shape(shape, dtype(1)), dtype=dtype)


def zeros_like(a: Any, dtype: type | None = None) -> NDArray:
    arr = asarray(a)
    return zeros(arr.shape, dtype or arr.dtype)


def ones_like(a: Any, dtype: type | None = None) -> NDArray:
    arr = asarray(a)
    return ones(arr.shape, dtype or arr.dtype)


def empty(shape: Tuple[int, ...], dtype: type = float) -> NDArray:
    return zeros(shape, dtype)


def arange(stop: int, dtype: type = float) -> NDArray:
    return NDArray([dtype(i) for i in range(int(stop))], dtype=dtype)


def linspace(start: float, stop: float, num: int, dtype: type = float) -> NDArray:
    if num == 1:
        return NDArray([dtype(start)], dtype=dtype)
    step = (stop - start) / (num - 1)
    values = [dtype(start + i * step) for i in range(num)]
    return NDArray(values, dtype=dtype)


def concatenate(arrays: Sequence[Any], axis: int = 0) -> NDArray:
    arrays = [asarray(arr) for arr in arrays]
    if not arrays:
        raise ValueError("need at least one array to concatenate")
    axis = axis % arrays[0].ndim
    data = [_deep_copy(arr._data) for arr in arrays]
    concatenated = _concatenate_data(data, axis)
    return NDArray(concatenated, dtype=arrays[0].dtype)


def _concatenate_data(arrays: List[Any], axis: int) -> Any:
    if axis == 0:
        result = []
        for arr in arrays:
            result.extend(arr)
        return result
    result = []
    for values in zip(*arrays):
        result.append(_concatenate_data(list(values), axis - 1))
    return result


nan = float("nan")


def mean(a: Any, axis: int | None = None, keepdims: bool = False):
    return asarray(a).mean(axis=axis, keepdims=keepdims)


def sum(a: Any, axis: int | None = None):
    return asarray(a).sum(axis=axis)


def std(a: Any, axis: int | None = None, keepdims: bool = False):
    return asarray(a).std(axis=axis, keepdims=keepdims)


def sqrt(value: Any) -> Any:
    if isinstance(value, NDArray):
        return NDArray(_apply_unary(value._data, math.sqrt), dtype=value.dtype)
    return math.sqrt(value)


def transpose(a: Any) -> NDArray:
    arr = asarray(a)
    if arr.ndim < 2:
        return arr.copy()
    data = arr._data

    def _transpose_recursive(values: Any) -> Any:
        if not isinstance(values[0], list):
            return [[item] for item in values]
        transposed = list(map(list, zip(*values)))
        return [_transpose_recursive(item) for item in transposed]

    return NDArray(_transpose_recursive(data), dtype=arr.dtype)


def polyfit(x: Sequence[float], y: Sequence[float], deg: int) -> List[float]:
    if deg != 1:
        raise NotImplementedError("Only linear fits are supported")
    x_vals = asarray(x).tolist() if isinstance(x, NDArray) else list(x)
    y_vals = asarray(y).tolist() if isinstance(y, NDArray) else list(y)
    n = len(x_vals)
    if n == 0:
        return [0.0, 0.0]
    mean_x = _builtins.sum(x_vals) / n
    mean_y = _builtins.sum(y_vals) / n
    num = _builtins.sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x_vals, y_vals))
    den = _builtins.sum((xi - mean_x) ** 2 for xi in x_vals)
    slope = num / den if den else 0.0
    intercept = mean_y - slope * mean_x
    return [slope, intercept]


class _TestingModule:
    @staticmethod
    def assert_allclose(actual, desired, rtol=1e-7, atol=0):
        a_arr = asarray(actual)
        d_arr = asarray(desired)
        if a_arr.shape != d_arr.shape:
            target_shape = _broadcast_shape(a_arr.shape, d_arr.shape)
            a_arr = broadcast_to(a_arr, target_shape)
            d_arr = broadcast_to(d_arr, target_shape)

        def _compare(a_data, d_data, path=""):
            if isinstance(a_data, list):
                for idx, (a_val, d_val) in enumerate(zip(a_data, d_data)):
                    _compare(a_val, d_val, f"{path}/{idx}")
            else:
                if not math.isclose(a_data, d_data, rel_tol=rtol, abs_tol=atol):
                    raise AssertionError(
                        f"Mismatch at {path}: {a_data} != {d_data}"
                    )

        _compare(a_arr._data, d_arr._data)

    @staticmethod
    def assert_array_equal(actual, desired):
        a_arr = asarray(actual)
        d_arr = asarray(desired)
        if a_arr.shape != d_arr.shape:
            target_shape = _broadcast_shape(a_arr.shape, d_arr.shape)
            a_arr = broadcast_to(a_arr, target_shape)
            d_arr = broadcast_to(d_arr, target_shape)

        def _compare(a_data, d_data, path=""):
            if isinstance(a_data, list):
                for idx, (a_val, d_val) in enumerate(zip(a_data, d_data)):
                    _compare(a_val, d_val, f"{path}/{idx}")
            else:
                if a_data != d_data:
                    raise AssertionError(f"Mismatch at {path}: {a_data} != {d_data}")

        _compare(a_arr._data, d_arr._data)


testing = _TestingModule()


def select(conditions: Sequence[Any], choices: Sequence[Any], default: Any = 0) -> NDArray:
    if len(conditions) != len(choices):
        raise ValueError("conditions and choices must have the same length")
    cond_arrays = [asarray(cond) for cond in conditions]
    choice_arrays = [asarray(choice) for choice in choices]
    result_shape = cond_arrays[0].shape
    output = zeros(result_shape, dtype=float)

    for idx in _iter_indices(result_shape):
        value = default
        for cond, choice in zip(cond_arrays, choice_arrays):
            if _get_from_nested(cond._data, idx):
                value = _get_from_nested(choice._data, idx)
                break
        updated = _set_in_nested(output._data, idx, value)
        if idx == ():
            output._data = updated
    return output


def _iter_indices(shape: Tuple[int, ...]):
    if not shape:
        yield ()
        return
    from itertools import product

    ranges = [range(dim) for dim in shape]
    for idx in product(*ranges):
        yield idx


def _get_from_nested(data: Any, indices: Tuple[int, ...]):
    for idx in indices:
        data = data[idx]
    return data


def _set_in_nested(data: Any, indices: Tuple[int, ...], value: Any):
    if not indices:
        return value
    for idx in indices[:-1]:
        data = data[idx]
    data[indices[-1]] = value
    return data


def _aggregate_over_axis(array: NDArray, axis: int, aggregator) -> NDArray:
    axis = axis % array.ndim
    result_shape = array.shape[:axis] + array.shape[axis + 1 :]

    def _recurse(prefix: List[int]) -> Any:
        if len(prefix) == len(result_shape):
            values = []
            for axis_index in range(array.shape[axis]):
                full_index = list(prefix)
                full_index.insert(axis, axis_index)
                values.append(_get_from_nested(array._data, tuple(full_index)))
            return aggregator(values)
        dim = result_shape[len(prefix)] if result_shape else 0
        return [_recurse(prefix + [i]) for i in range(dim)]

    aggregated_data = _recurse([])
    return NDArray(aggregated_data, dtype=array.dtype)


def _assign_in_nested(data: Any, indices: Tuple[Any, ...], value: Any):
    if not indices:
        return
    idx = indices[0]
    if isinstance(idx, int):
        if len(indices) == 1:
            data[idx] = value
        else:
            _assign_in_nested(data[idx], indices[1:], value)
    elif idx is None:
        if len(indices) == 1:
            data[:] = value
        else:
            for element in data:
                _assign_in_nested(element, indices[1:], value)
    else:
        raise TypeError("Unsupported assignment index")


def _fill_shape(shape: Tuple[int, ...], value: Number) -> Any:
    if not shape:
        return value
    return [_fill_shape(shape[1:], value) for _ in range(shape[0])]


def isscalar(obj: Any) -> bool:
    return not isinstance(obj, (NDArray, list, tuple))


def _insert_dim(shape: Tuple[int, ...], axis: int) -> Tuple[int, ...]:
    return shape[:axis] + (1,) + shape[axis + 1 :]


class _RandomGenerator:
    def __init__(self, seed: int | None = None):
        self._rng = _stdlib_random.Random(seed)

    def normal(self, loc: float = 0.0, scale: float = 1.0, size: Tuple[int, ...] | None = None) -> NDArray:
        if size is None:
            return NDArray(self._rng.gauss(loc, scale))
        total = _product(size)
        values = [self._rng.gauss(loc, scale) for _ in range(total)]
        return NDArray(_unflatten(values, size))


class _RandomModule:
    def default_rng(self, seed: int | None = None) -> _RandomGenerator:
        return _RandomGenerator(seed)


random = _RandomModule()


__all__ = [
    "NDArray",
    "ndarray",
    "array",
    "asarray",
    "zeros",
    "zeros_like",
    "ones_like",
    "ones",
    "empty",
    "arange",
    "linspace",
    "concatenate",
    "nan",
    "mean",
    "sum",
    "std",
    "sqrt",
    "polyfit",
    "select",
    "testing",
    "random",
    "isscalar",
    "bool_",
    "float64",
    "int64",
]
