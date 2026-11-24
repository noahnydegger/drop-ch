"""Checks for writing predictions after slicing a chunked input dataset."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import xarray as xr

from predict_local import PortablePredictionWriter
from project.core.utils import get_zarr_v3_encoding


@pytest.mark.parametrize("dims", [("time", "hru"), ("hru", "time")])
def test_sliced_predictions_write_to_zarr(tmp_path, dims):
    values = np.arange(730 * 3, dtype="float32").reshape(730, 3)
    ds = xr.Dataset(
        {
            "Q_obs_m3s": (("time", "hru"), values),
            "T_down": (("hru", "k"), np.ones((3, 2), dtype="float32")),
            "subset": ("hru", [1, 2, 3]),
        },
        coords={"time": np.arange(730), "hru": [10, 20, 30], "k": [0, 1]},
    )
    ds["Q_obs_m3s"] = ds.Q_obs_m3s.transpose(*dims).chunk({"time": (10, 366, 354), "hru": 3})
    expected = ds.compute()
    encoding = get_zarr_v3_encoding(ds)
    output = tmp_path / "preds.zarr"
    ds.to_zarr(output, encoding=encoding, zarr_format=3, consolidated=False)
    with xr.open_zarr(output, consolidated=False) as restored:
        xr.testing.assert_equal(restored.compute(), expected)


def test_local_writer_preserves_discharge_and_runoff(tmp_path):
    ds = xr.Dataset(
        {
            "Q_obs_m3s": (("time", "hru"), np.zeros((3, 2))),
            "area": ("hru", [1.0, 2.0]),
            "has_obs": ("hru", [True, True]),
        },
        coords={"time": np.arange(3), "hru": [10, 20]},
    )

    def variable(value):
        return {
            "ds": torch.full((2, 3, 1), value),
            "obs_counterpart": "Q_obs_m3s",
            "units": "m³/s",
            "long_name": "Flow",
        }

    prediction = {
        "data": {"Q_pred_m3s": variable(100.0), "q_pred_m3s": variable(2.0)},
        "batch": SimpleNamespace(hru_idx=[0, 1], time_slice=[torch.tensor([0]), torch.tensor([3])]),
        "obs_var": "Q_obs_m3s",
        "pred_var": "Q_pred_m3s",
    }
    trainer = SimpleNamespace(
        predict_dataloaders=SimpleNamespace(dataset=SimpleNamespace(ds=ds)),
        datamodule=SimpleNamespace(train_hrus=None, valid_hrus=None, test_hrus=None, temporal_context_length=1),
        log_dir=str(tmp_path),
    )
    PortablePredictionWriter().write_on_epoch_end(trainer, None, [prediction], [])
    with xr.open_zarr(tmp_path / "preds.zarr") as restored:
        assert bool((restored.Q_pred_m3s.isel(time=slice(1, None)) == 100).all().compute())
        assert bool((restored.q_local_pred_m3s.isel(time=slice(1, None)) == 2).all().compute())
        assert restored.Q_pred_m3s.isel(time=0).isnull().all()
