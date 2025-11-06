import importlib
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


class FakeMask:
    def __init__(self, values):
        self.values = list(values)

    def __and__(self, other):
        return FakeMask([a and b for a, b in zip(self.values, other.values)])


class FakeIndex(list):
    def __ge__(self, other):
        return FakeMask([value >= other for value in self])

    def __le__(self, other):
        return FakeMask([value <= other for value in self])


class FakeDataFrame:
    def __init__(self, index):
        self.index = FakeIndex(index)

    def __getitem__(self, mask):
        if isinstance(mask, FakeMask):
            filtered_index = [value for value, keep in zip(self.index, mask.values) if keep]
            return FakeDataFrame(filtered_index)
        raise KeyError("Mask expected")


class FakeArray:
    def __init__(self, shape):
        self.shape = shape


class DummyModel:
    def __init__(self):
        self.fit_kwargs = None
        self.saved_path = None

    def fit(self, *args, **kwargs):
        self.fit_kwargs = kwargs
        return types.SimpleNamespace(history={})

    def save(self, path):
        self.saved_path = path


def _install_tensorflow_stub(monkeypatch):
    class DummyCallback:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    callbacks = types.SimpleNamespace(
        EarlyStopping=DummyCallback,
        ReduceLROnPlateau=DummyCallback,
    )

    tf_stub = types.SimpleNamespace(
        keras=types.SimpleNamespace(callbacks=callbacks)
    )

    monkeypatch.setitem(sys.modules, "tensorflow", tf_stub)


def _install_project_stubs(fake_index, dummy_model, monkeypatch):
    data_pkg = sys.modules.get("data", types.ModuleType("data"))
    data_pkg.__path__ = []
    sys.modules["data"] = data_pkg

    fetcher_module = types.ModuleType("data.fetcher")
    fetcher_module.fetch_stock_data = lambda ticker, start, end: FakeDataFrame(fake_index)
    sys.modules["data.fetcher"] = fetcher_module
    setattr(data_pkg, "fetcher", fetcher_module)

    indicators_module = types.ModuleType("data.indicators")
    indicators_module.add_technical_indicators = lambda df: df
    sys.modules["data.indicators"] = indicators_module
    setattr(data_pkg, "indicators", indicators_module)

    features_pkg = sys.modules.get("features", types.ModuleType("features"))
    features_pkg.__path__ = []
    sys.modules["features"] = features_pkg

    windowing_module = types.ModuleType("features.windowing")
    windowing_module.create_lstm_dataset_classification = lambda df, seq_length: (
        FakeArray((64, seq_length, 1)),
        FakeArray((64, 1)),
    )
    sys.modules["features.windowing"] = windowing_module
    setattr(features_pkg, "windowing", windowing_module)

    models_pkg = sys.modules.get("models", types.ModuleType("models"))
    models_pkg.__path__ = []
    sys.modules["models"] = models_pkg

    lstm_module = types.ModuleType("models.lstm")
    lstm_module.build_lstm_model = lambda input_shape: dummy_model
    sys.modules["models.lstm"] = lstm_module
    setattr(models_pkg, "lstm", lstm_module)

    sklearn_pkg = types.ModuleType("sklearn")
    sklearn_pkg.__path__ = []
    model_selection_module = types.ModuleType("sklearn.model_selection")
    model_selection_module.train_test_split = (
        lambda X, y, test_size=None, shuffle=None: (X, X, y, y)
    )
    sys.modules["sklearn"] = sklearn_pkg
    sys.modules["sklearn.model_selection"] = model_selection_module
    setattr(sklearn_pkg, "model_selection", model_selection_module)

    monkeypatch.setitem(sys.modules, "data", data_pkg)
    monkeypatch.setitem(sys.modules, "data.fetcher", fetcher_module)
    monkeypatch.setitem(sys.modules, "data.indicators", indicators_module)
    monkeypatch.setitem(sys.modules, "features", features_pkg)
    monkeypatch.setitem(sys.modules, "features.windowing", windowing_module)
    monkeypatch.setitem(sys.modules, "models", models_pkg)
    monkeypatch.setitem(sys.modules, "models.lstm", lstm_module)
    monkeypatch.setitem(sys.modules, "sklearn", sklearn_pkg)
    monkeypatch.setitem(sys.modules, "sklearn.model_selection", model_selection_module)


def test_train_model_disables_shuffling(monkeypatch):
    _install_tensorflow_stub(monkeypatch)

    fake_index = [
        "2019-12-31",
        "2020-01-01",
        "2021-01-01",
        "2023-01-01",
        "2024-12-31",
        "2025-01-01",
    ]
    dummy_model = DummyModel()

    _install_project_stubs(fake_index, dummy_model, monkeypatch)

    sys.modules.pop("train", None)
    train = importlib.import_module("train")

    model, *_ = train.train_model()

    assert model is dummy_model
    assert dummy_model.fit_kwargs is not None
    assert dummy_model.fit_kwargs.get("shuffle") is False
