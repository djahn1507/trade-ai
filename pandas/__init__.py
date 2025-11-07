"""Minimal pandas stub for the project tests."""
from __future__ import annotations

import datetime as _dt
import math
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

__version__ = "0.0-teststub"


class Series:
    def __init__(self, data: Iterable[Any], index: Iterable[Any] | None = None, name: str | None = None):
        values = list(data)
        if index is None:
            index = list(range(len(values)))
        index_list = list(index)
        if len(index_list) != len(values):
            raise ValueError("Length of index must match data")
        self._data = values
        self._index = index_list
        self.name = name

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return Series(self._data[item], self._index[item], name=self.name)
        if isinstance(item, list):
            return Series([self._data[i] for i in item], [self._index[i] for i in item], name=self.name)
        return self._data[item]

    def __setitem__(self, key, value):
        if isinstance(key, int):
            self._data[key] = value
        else:
            raise TypeError("Unsupported assignment type")

    @property
    def values(self) -> List[Any]:
        return list(self._data)

    def tolist(self) -> List[Any]:
        return list(self._data)

    def copy(self) -> "Series":
        return Series(self._data, self._index, name=self.name)

    def squeeze(self) -> "Series":
        return self

    @property
    def index(self) -> List[Any]:
        return list(self._index)

    @property
    def iloc(self) -> "_SeriesILoc":
        return _SeriesILoc(self)

    def _binary_op(self, other: Any, op) -> "Series":
        if isinstance(other, Series):
            other_values = other._align_to(self._index)
        else:
            other_values = [other] * len(self)
        result = [op(a, b) for a, b in zip(self._data, other_values)]
        return Series(result, self._index, name=self.name)

    def _align_to(self, index: Sequence[Any]) -> List[Any]:
        mapping = {idx: value for idx, value in zip(self._index, self._data)}
        return [mapping[idx] for idx in index]

    def __add__(self, other: Any) -> "Series":
        return self._binary_op(other, lambda a, b: a + b)

    def __sub__(self, other: Any) -> "Series":
        return self._binary_op(other, lambda a, b: a - b)

    def __radd__(self, other: Any) -> "Series":
        return self.__add__(other)

    def __rsub__(self, other: Any) -> "Series":
        return Series([other - value for value in self._data], self._index, name=self.name)

    def __mul__(self, other: Any) -> "Series":
        return self._binary_op(other, lambda a, b: a * b)

    def __truediv__(self, other: Any) -> "Series":
        return self._binary_op(other, lambda a, b: a / b if b not in (0, None) else math.nan)

    def __rtruediv__(self, other: Any) -> "Series":
        return Series([other / v if v not in (0, None) else math.nan for v in self._data], self._index, name=self.name)

    def __gt__(self, other: Any) -> "Series":
        return self._binary_op(other, lambda a, b: a > b)

    def __lt__(self, other: Any) -> "Series":
        return self._binary_op(other, lambda a, b: a < b)

    def __ge__(self, other: Any) -> "Series":
        return self._binary_op(other, lambda a, b: a >= b)

    def __le__(self, other: Any) -> "Series":
        return self._binary_op(other, lambda a, b: a <= b)

    def __and__(self, other: "Series") -> "Series":
        return self._binary_op(other, lambda a, b: bool(a) and bool(b))

    def __neg__(self) -> "Series":
        return Series([-value for value in self._data], self._index, name=self.name)

    def astype(self, dtype: type) -> "Series":
        return Series([dtype(value) for value in self._data], self._index, name=self.name)

    def shift(self, periods: int) -> "Series":
        if periods == 0:
            return self.copy()
        fill = [math.nan] * abs(periods)
        if periods > 0:
            data = fill + self._data[:-periods]
        else:
            data = self._data[-periods:] + fill
        return Series(data, self._index, name=self.name)

    def diff(self, periods: int = 1) -> "Series":
        data = [math.nan] * periods
        for i in range(periods, len(self._data)):
            data.append(self._data[i] - self._data[i - periods])
        return Series(data, self._index, name=self.name)

    def clip(self, lower: float | None = None, upper: float | None = None) -> "Series":
        def _clip(value):
            if lower is not None and value < lower:
                return lower
            if upper is not None and value > upper:
                return upper
            return value

        return Series([_clip(v) for v in self._data], self._index, name=self.name)

    def rolling(self, window: int) -> "Rolling":
        return Rolling(self, window)

    def pct_change(self) -> "Series":
        data = [math.nan]
        for prev, curr in zip(self._data[:-1], self._data[1:]):
            if prev in (0, None):
                data.append(math.nan)
            else:
                data.append(curr / prev - 1)
        return Series(data, self._index, name=self.name)

    def fillna(self, value: Any | None = None, method: str | None = None) -> "Series":
        data = list(self._data)
        if method == "ffill":
            last = None
            for idx, val in enumerate(data):
                if not _isnan(val):
                    last = val
                elif last is not None:
                    data[idx] = last
        elif method == "bfill":
            next_val = None
            for idx in range(len(data) - 1, -1, -1):
                val = data[idx]
                if not _isnan(val):
                    next_val = val
                elif next_val is not None:
                    data[idx] = next_val
        elif value is not None:
            data = [value if _isnan(v) else v for v in data]
        return Series(data, self._index, name=self.name)

    def replace(self, to_replace: Any, value: Any) -> "Series":
        data = [value if v == to_replace else v for v in self._data]
        return Series(data, self._index, name=self.name)

    def mean(self) -> float:
        valid = [v for v in self._data if not _isnan(v)]
        return sum(valid) / len(valid) if valid else 0.0

    def std(self) -> float:
        valid = [v for v in self._data if not _isnan(v)]
        if not valid:
            return 0.0
        mean = sum(valid) / len(valid)
        variance = sum((v - mean) ** 2 for v in valid) / len(valid)
        return math.sqrt(variance)

    def max(self) -> float:
        valid = [v for v in self._data if not _isnan(v)]
        return max(valid) if valid else math.nan

    def min(self) -> float:
        valid = [v for v in self._data if not _isnan(v)]
        return min(valid) if valid else math.nan

    def sum(self) -> float:
        valid = [v for v in self._data if not _isnan(v)]
        return sum(valid)

    def apply(self, func) -> "Series":
        return Series([func(v) for v in self._data], self._index, name=self.name)

    def dropna(self) -> "Series":
        data = [v for v in self._data if not _isnan(v)]
        index = [idx for idx, v in zip(self._index, self._data) if not _isnan(v)]
        return Series(data, index, name=self.name)


class Rolling:
    def __init__(self, series: Series, window: int):
        self._series = series
        self._window = max(1, window)

    def mean(self) -> Series:
        data = []
        values = self._series._data
        for i in range(len(values)):
            window_values = values[max(0, i - self._window + 1) : i + 1]
            if len(window_values) < self._window:
                data.append(math.nan)
            else:
                valid = [v for v in window_values if not _isnan(v)]
                data.append(sum(valid) / len(valid) if valid else math.nan)
        return Series(data, self._series._index, name=self._series.name)

    def std(self) -> Series:
        data = []
        values = self._series._data
        for i in range(len(values)):
            window_values = values[max(0, i - self._window + 1) : i + 1]
            if len(window_values) < self._window:
                data.append(math.nan)
            else:
                valid = [v for v in window_values if not _isnan(v)]
                if not valid:
                    data.append(math.nan)
                else:
                    mean = sum(valid) / len(valid)
                    variance = sum((v - mean) ** 2 for v in valid) / len(valid)
                    data.append(math.sqrt(variance))
        return Series(data, self._series._index, name=self._series.name)


class _SeriesILoc:
    def __init__(self, series: Series):
        self._series = series

    def __getitem__(self, index):
        data = self._series._data
        if isinstance(index, slice):
            return Series(data[index], self._series._index[index], name=self._series.name)
        return data[index]


class DataFrame:
    def __init__(self, data: Dict[str, Iterable[Any]], index: Iterable[Any] | None = None):
        if not data:
            raise ValueError("DataFrame requires at least one column")
        columns = list(data.keys())
        values = [list(data[col]) for col in columns]
        lengths = {len(col_values) for col_values in values}
        if len(lengths) != 1:
            raise ValueError("Columns must have the same length")
        length = lengths.pop()
        if index is None:
            index = list(range(length))
        index_list = list(index)
        if len(index_list) != length:
            raise ValueError("Index length mismatch")
        self._columns = columns
        self._data = {col: list(values[idx]) for idx, col in enumerate(columns)}
        self._index = index_list

    @property
    def columns(self) -> List[str]:
        return list(self._columns)

    @property
    def index(self) -> List[Any]:
        return list(self._index)

    def copy(self) -> "DataFrame":
        return DataFrame({col: list(self._data[col]) for col in self._columns}, self._index)

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, item: str) -> bool:
        return item in self._data

    def __getitem__(self, key):
        if isinstance(key, str):
            return Series(self._data[key], self._index, name=key)
        if isinstance(key, Series):
            mask = key._data
            filtered_index = [idx for idx, keep in zip(self._index, mask) if keep]
            filtered_data = {col: [val for val, keep in zip(values, mask) if keep] for col, values in self._data.items()}
            return DataFrame(filtered_data, filtered_index)
        if isinstance(key, list):
            data = {col: self._data[col] for col in key}
            return DataFrame(data, self._index)
        raise TypeError("Unsupported key type")

    def __setitem__(self, key: str, value: Iterable[Any] | Series):
        if isinstance(value, Series):
            aligned = value._align_to(self._index)
        elif hasattr(value, "tolist"):
            aligned = value.tolist()
            if not isinstance(aligned, list):
                aligned = [aligned] * len(self)
            if len(aligned) != len(self):
                aligned = list(aligned)
        else:
            aligned = list(value)
            if len(aligned) != len(self):
                raise ValueError("Column length mismatch")
        self._data[key] = aligned
        if key not in self._columns:
            self._columns.append(key)

    def get(self, key: str) -> Series | None:
        if key in self._data:
            return Series(self._data[key], self._index, name=key)
        return None

    @property
    def iloc(self) -> "_DataFrameILoc":
        return _DataFrameILoc(self)

    @property
    def loc(self) -> "_DataFrameLoc":
        return _DataFrameLoc(self)

    def dropna(self) -> "DataFrame":
        keep_indices = []
        for idx in range(len(self)):
            row_valid = True
            for col in self._columns:
                value = self._data[col][idx]
                if _isnan(value):
                    row_valid = False
                    break
            if row_valid:
                keep_indices.append(idx)
        data = {col: [self._data[col][i] for i in keep_indices] for col in self._columns}
        index = [self._index[i] for i in keep_indices]
        return DataFrame(data, index)

    def drop(self, columns: List[str]) -> "DataFrame":
        data = {col: values for col, values in self._data.items() if col not in columns}
        return DataFrame(data, self._index)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self), len(self._columns)

    @property
    def values(self) -> List[List[Any]]:
        return [[self._data[col][i] for col in self._columns] for i in range(len(self))]


class _DataFrameILoc:
    def __init__(self, df: DataFrame):
        self._df = df

    def __getitem__(self, key):
        if isinstance(key, slice):
            indices = list(range(len(self._df)))[key]
            data = {col: [self._df._data[col][i] for i in indices] for col in self._df._columns}
            index = [self._df._index[i] for i in indices]
            return DataFrame(data, index)
        if isinstance(key, int):
            return {col: self._df._data[col][key] for col in self._df._columns}
        raise TypeError("Unsupported iloc key")


class _DataFrameLoc:
    def __init__(self, df: DataFrame):
        self._df = df

    def __getitem__(self, key):
        if isinstance(key, tuple):
            rows, columns = key
            if rows == slice(None):
                indices = self._df._index
            elif isinstance(rows, list):
                indices = rows
            else:
                indices = [rows]
            if columns == slice(None):
                cols = self._df._columns
            elif isinstance(columns, list):
                cols = columns
            else:
                cols = [columns]
            data = {col: [self._df._data[col][self._df._index.index(idx)] for idx in indices] for col in cols}
            if len(cols) == 1:
                return Series(list(data.values())[0], indices, name=cols[0])
            return DataFrame(data, indices)
        if isinstance(key, list):
            data = {col: [self._df._data[col][self._df._index.index(idx)] for idx in key] for col in self._df._columns}
            return DataFrame(data, key)
        raise TypeError("Unsupported loc key")

    def __call__(self, rows, columns=None):
        if columns is None:
            raise NotImplementedError
        if isinstance(rows, Series):
            row_indices = [idx for idx, flag in zip(self._df._index, rows._data) if flag]
        else:
            row_indices = rows if isinstance(rows, list) else [rows]
        for col_name, col_values in columns.items():
            for idx, value in zip(row_indices, col_values):
                pos = self._df._index.index(idx)
                self._df._data[col_name][pos] = value


def concat(objects: Sequence[Series], axis: int = 0) -> DataFrame:
    if axis != 1:
        raise NotImplementedError("Only axis=1 is supported in this stub")
    if not objects:
        raise ValueError("Need at least one object to concatenate")
    index = objects[0].index
    data: Dict[str, List[Any]] = {}
    for idx, series in enumerate(objects):
        data[str(idx)] = series._align_to(index)
    return DataFrame(data, index)


def date_range(start: str, periods: int, freq: str = "D") -> List[str]:
    start_date = _dt.datetime.strptime(start, "%Y-%m-%d")
    delta = _dt.timedelta(days=1 if freq == "D" else 0)
    return [(start_date + i * delta).strftime("%Y-%m-%d") for i in range(periods)]


class _TestingModule:
    @staticmethod
    def assert_series_equal(left: Series, right: Series):
        if left.index != right.index:
            raise AssertionError(f"Index mismatch: {left.index} != {right.index}")
        np.testing.assert_allclose(left.values, right.values)


testing = _TestingModule()


def _isnan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


__all__ = [
    "Series",
    "DataFrame",
    "concat",
    "date_range",
    "testing",
]
