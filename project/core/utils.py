"""Utils for core CLI and tuning engine."""

import logging
import os
from collections.abc import Sequence
from typing import Any, cast

import lightning.pytorch as pl
import numpy as np
import xarray as xr
from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch.profilers import PyTorchProfiler
from torch.profiler import ProfilerActivity, tensorboard_trace_handler
from torch.utils.data import DataLoader
from zarr.codecs import BloscCodec, BloscShuffle

logger = logging.getLogger("pytorch_lightning")


def get_zarr_v3_encoding(ds: xr.Dataset, clevel: int = 3) -> dict:
    """Get encoding for the dataset to be saved to zarr.

    This fixes a known issue with the compressor when reading zarr v2 files and writing them with zarr v3.

    Args:
        ds (xr.Dataset): The dataset to get the encoding for.
        clevel (int, optional): Compression level for data variables. Defaults to 3.

    Returns:
        dict: A dictionary with the encoding for each variable and coordinate in the dataset.
    """

    encoding = {}

    for var in ds.data_vars:
        chunks = {}
        for dim in ds[var].dims:
            if dim == "time":
                c = 1000
            else:
                c = len(ds[dim])
            chunks[dim] = c

        encoding[var] = {
            "compressors": BloscCodec(cname="zstd", clevel=clevel, shuffle=BloscShuffle.shuffle),
            "chunks": tuple(chunks.values()),
        }
        ds[var].encoding.pop("compressors", None)
        ds[var].encoding.pop("chunks", None)
        if "time" in ds[var].dims:
            ds[var] = ds[var].chunk(chunks)

    for coord in ds.coords:
        encoding[coord] = {
            "compressors": BloscCodec(cname="zstd", clevel=1, shuffle=BloscShuffle.shuffle),
        }
        ds[coord].encoding.pop("compressors", None)

    return encoding


class PredictionWriter(BasePredictionWriter):
    """Custom prediction writer for PyTorch Lightning that saves predictions into a Zarr dataset.

    Attributes:
    ----------
    write_all_variables : bool
        Whether to write all variables from the dataset or just the ones in `variables`.
    """

    def __init__(self, write_all_variables: bool = False) -> None:
        """Initialize the prediction writer.

        Parameters
        ----------
        write_all_variables : bool, optional
            If True, include all dataset variables in the output, by default False.
        """
        super().__init__(write_interval="epoch")
        self.write_all_variables = write_all_variables

    def write_on_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        predictions: Sequence[dict],
        batch_indices: Sequence[Any],
    ) -> None:
        """Write predictions into a Zarr dataset at the end of the prediction epoch.

        Parameters
        ----------
        trainer : pl.Trainer
            The PyTorch Lightning trainer instance.
        pl_module : pl.LightningModule
            The Lightning module that produced the predictions.
        predictions : Sequence[list[dict]]
            Predictions returned by the model for each batch.
        batch_indices : Sequence[Any]
            Indices of the batches (unused here, but required by API).
        """
        if trainer.predict_dataloaders is None:
            raise ValueError("No predict dataloaders found. Make sure to pass a dataloader to the predict method.")

        datamodule = cast(pl.LightningDataModule, trainer.datamodule)  # type: ignore
        pdl = cast(DataLoader, trainer.predict_dataloaders)  # type: ignore
        ds = cast(xr.Dataset, pdl.dataset.ds.copy())  # type: ignore
        k_hops = getattr(pdl.dataset, "k_hops", None)  # type: ignore

        # Prediction varaibles sample
        p = predictions[0]["data"]
        save_variables = list(p.keys())
        reference_variables = [v["obs_counterpart"] for v in p.values()]
        ds.attrs["description"] = "DROP predictions"
        ds.attrs["obs_var"] = predictions[0]["obs_var"]
        ds.attrs["pred_var"] = predictions[0]["pred_var"]

        # Initialize subset and prediction variables
        ds["subset"] = xr.full_like(ds.area, fill_value=0, dtype="int32")
        for var, ref in zip(save_variables, reference_variables, strict=True):
            if p[var].get("is_downstream_aligned", False):
                if k_hops is None:
                    raise ValueError("k_hops is required for downstream aligned variables.")
                ds[var] = xr.full_like(
                    ds["area"].expand_dims(k=range(k_hops), axis=1), fill_value=np.nan, dtype="float32"
                )
            else:
                ds[var] = xr.full_like(ds[ref], fill_value=np.nan, dtype="float32")

            for attr in ["units", "long_name"]:
                val = p[var][attr]
                ds[var].attrs[attr] = val

                if not p[var].get("exclude_ref_from_zarr", False):
                    ds[ref].attrs[attr] = val

        # Assign subset values based on HRUs from datamodule
        if datamodule.train_hrus is not None:  # type: ignore
            ds["subset"].loc[{"hru": datamodule.train_hrus}] = 1  # type: ignore
        if datamodule.valid_hrus is not None:  # type: ignore
            ds["subset"].loc[{"hru": datamodule.valid_hrus}] = 2  # type: ignore
        if datamodule.test_hrus is not None:  # type: ignore
            ds["subset"].loc[{"hru": datamodule.test_hrus}] = 3  # type: ignore

        ds["subset"].attrs["legend"] = "0: no set, 1: train, 2: valid, 3: test"
        temporal_context_length = datamodule.temporal_context_length  # type: ignore

        # Insert predictions into dataset
        for pred in predictions:
            hru_subset = pred["batch"].hru_idx
            time_slice = pred["batch"].time_slice

            # Depending on collate (PYG vs PyTorch), time_slice gets stacked differently.
            if isinstance(time_slice[0], list):
                time_subset = slice(time_slice[0][0] + temporal_context_length, time_slice[0][1])
            else:
                time_subset = slice(time_slice[0][0] + temporal_context_length, time_slice[1][0])

            for var in save_variables:
                if p[var].get("is_downstream_aligned", False):
                    ds[var][
                        {
                            "hru": hru_subset,
                        }
                    ] = pred["data"][var]["ds"]
                else:
                    ds[var][{"hru": hru_subset, "time": time_subset}] = pred["data"][var]["ds"][
                        ..., datamodule.temporal_context_length :, 0
                    ].T  # type: ignore

        # Keep only selected variables if specified
        if not self.write_all_variables:
            reference_variables = [v for v in reference_variables if v is not None]
            ds = ds[save_variables + reference_variables + ["subset", "has_obs"]]

        encoding = get_zarr_v3_encoding(ds)

        # Write dataset to disk
        write_path = os.path.join(trainer.log_dir, "preds.zarr")  # type: ignore
        ds.to_zarr(store=write_path, mode="w", encoding=encoding)

        logger.info(f"Predictions written to '{write_path}'.")


def setup_profiler(trainer) -> None:
    """Set up the PyTorch profiler for the trainer.

    Args:
        trainer (pl.Trainer): The PyTorch Lightning trainer instance.

    """

    dir_name = os.path.join(trainer.log_dir, "profiler")  # type: ignore

    trainer.profiler = PyTorchProfiler(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        on_trace_ready=tensorboard_trace_handler(dir_name),
    )
