"""Run CPU inference with the published routed/random DROP checkpoint.

Examples (from the repository root, with .venv activated):
    python predict_local.py --dry-run
    python predict_local.py
    python predict_local.py --all-hrus --output outputs_local/routed_random_all_2007

The dates include warm-up. Predictions for the first 365 days are left missing
by the existing PredictionWriter. No training or normalization fitting is done.
Local runoff is saved as q_local_pred_m3s to avoid a case-insensitive filesystem
collision with river discharge, Q_pred_m3s.
"""

import argparse
from collections import defaultdict
from datetime import date
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import xarray as xr
import yaml
from lightning.pytorch.loggers import TensorBoardLogger
from torch_geometric.loader import DataLoader

from project.core.utils import PredictionWriter
from project.data.datamodule import HRUData
from project.nn.drop_model import DROP
from project.utils.paths import get_path


def upstream_hrus(ds, outlet):
    """Collect the outlet and every upstream HRU, preserving dataset order."""
    ids = ds.hru.values
    if outlet not in ids:
        raise ValueError(f"Outlet HRU {outlet} is not in the input dataset.")
    upstream = defaultdict(list)
    for source, target in zip(ids, ds.downstream.values, strict=True):
        upstream[int(target)].append(int(source))
    selected = set()
    pending = [outlet]
    while pending:
        current = pending.pop()
        if current not in selected:
            selected.add(current)
            pending.extend(upstream[current])
    return ids[np.isin(ids, list(selected))]


class InferenceDataModule(pl.LightningDataModule):
    """Use the existing HRUData loader without initializing training splits.

    A spatial crop cannot go through GraphDataModule's normal constructor,
    which expects predefined national catchment outlets. This inference-only
    adapter retains the full selected upstream network and checkpoint scaling.
    """

    def __init__(self, ds, config, outlets, published_subsets):
        super().__init__()
        self.ds = ds
        self.config = config
        self.outlets = outlets
        self.temporal_context_length = config["temporal_context_length"]
        # Preserve the published spatial labels instead of recomputing random
        # assignments from the shorter inference time window.
        self.train_hrus = ds.hru.values[published_subsets == 1]
        self.valid_hrus = ds.hru.values[published_subsets == 2]
        self.test_hrus = ds.hru.values[published_subsets == 3]

    def predict_dataloader(self):
        """Build a deterministic loader over complete upstream networks."""
        dataset = HRUData(
            ds=self.ds,
            obs_mask=self.ds.has_obs,
            stat_features=self.config["stat_features"],
            stat_routing_features=self.config["stat_routing_features"],
            dyn_features=self.config["dyn_features"],
            k_hops=self.config["k_hops"],
            temporal_context_length=-1,
            prediction_length=-1,
            hru_samples=self.outlets,
            aggregate_catchments=False,
        )
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


class PortablePredictionWriter(PredictionWriter):
    """Keep discharge and local runoff distinct on case-insensitive filesystems."""

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        """Rename local runoff before the existing writer creates Zarr arrays."""
        portable = []
        for prediction in predictions:
            data = dict(prediction["data"])
            data["q_local_pred_m3s"] = data.pop("q_pred_m3s")
            portable.append({**prediction, "data": data})
        super().write_on_epoch_end(trainer, pl_module, portable, batch_indices)


def main():
    """Configure a small catchment or national inference run."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--outlet", type=int, default=153964, help="Catchment outlet HRU (default: Breggia, 153964).")
    scope.add_argument("--all-hrus", action="store_true", help="Predict every outlet network in the dataset.")
    parser.add_argument("--start", default="2006-01-01", help="Input start date, including warm-up.")
    parser.add_argument("--end", default="2007-12-31", help="Input end date, inclusive.")
    parser.add_argument("--output", type=Path, help="New output directory; existing directories are refused.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check inputs and load the checkpoint without predicting or writing outputs.",
    )
    args = parser.parse_args()
    if date.fromisoformat(args.start) > date.fromisoformat(args.end):
        parser.error("--start must be earlier than --end.")

    data_root = get_path("data_root")
    model_dir = data_root / "models/routed/random"
    config = yaml.safe_load((model_dir / "config.yaml").read_text())["data"]["init_args"]
    source = xr.open_zarr(data_root / "harmonized_hru_data.zarr")
    ds = source.sel(time=slice(args.start, args.end))
    context = config["temporal_context_length"]
    if ds.sizes["time"] <= context:
        parser.error(f"Input must contain more than {context} days, including warm-up.")

    if args.all_hrus:
        outlets = [int(h) for h in ds.hru.values[ds.downstream.values == -1]]
        tag = "all"
    else:
        ds = ds.sel(hru=upstream_hrus(source, args.outlet))
        outlets = [args.outlet]
        tag = str(args.outlet)
    if not outlets:
        raise ValueError("No prediction outlets found.")

    # This file supplies labels only; predictions are computed from scratch.
    with xr.open_zarr(get_path("predictions_file")) as published:
        labels = published["subset"].sel(model="routed_random", hru=ds.hru).compute().values
    datamodule = InferenceDataModule(ds, config, outlets, labels)
    output = args.output or get_path("log_root") / f"routed_random_{tag}_{args.start}_{args.end}"
    output = output.resolve()
    if output.exists() and not args.dry_run:
        raise FileExistsError(f"{output} exists. Choose a new --output directory.")

    print(f"Model: {model_dir / 'best.ckpt'}", flush=True)
    print(
        f"CPU inference: {ds.sizes['hru']:,} HRUs, {len(outlets)} outlet(s), {ds.sizes['time']} input days.", flush=True
    )
    print(f"First saved prediction date: {str(ds.time.values[context])[:10]} ({context} warm-up days).", flush=True)
    print(f"Output: {output / 'preds.zarr'}", flush=True)
    pl.seed_everything(99, workers=True)
    model = DROP.load_from_checkpoint(
        model_dir / "best.ckpt",
        map_location="cpu",
        runoff_net_exp_dir=str(data_root / "models/prevah/best.ckpt"),
    )
    model.eval()
    print("Checkpoint loaded; using its saved normalization and weights.", flush=True)
    if args.dry_run:
        # Build the selected graph too, without running a forward pass.
        batch = next(iter(datamodule.predict_dataloader()))
        print(f"First graph: {batch.num_nodes} nodes, {batch.y.shape[1]} days.", flush=True)
        print("Dry run complete: no predictions or output files written.", flush=True)
        return

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        logger=TensorBoardLogger(save_dir=str(output), name="", version=""),
        callbacks=[PortablePredictionWriter()],
        enable_checkpointing=False,
    )
    trainer.predict(model, datamodule=datamodule)
    with xr.open_zarr(output / "preds.zarr") as result:
        valid = result["Q_pred_m3s"].isel(time=slice(context, None))
        finite = np.isfinite(valid).all().compute().item()
        nonnegative = (valid >= 0).all().compute().item()
        print(
            f"Saved {result.sizes['hru']:,} HRUs. "
            f"All post-warm-up discharge finite: {finite}; non-negative: {nonnegative}."
        )
        if not finite or not nonnegative:
            raise RuntimeError("Prediction checks failed; inspect the output before scaling up.")
    print(f"Finished: {output / 'preds.zarr'}")


if __name__ == "__main__":
    main()
