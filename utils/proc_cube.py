import os
import re
import json
import glob
from datetime import datetime, timedelta
from itertools import product

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray  # noqa: F401
import lightgbm as lgb
from sklearn.metrics import brier_score_loss


def parse_timestamp(filename):
    """Extrai timestamp suportando padrões A e B."""
    base = os.path.basename(filename)

    m_gee = re.search(r"M(\d{2})_D(\d{2})_(\d{4})_H(\d{2})", base)
    if m_gee:
        mm, dd, yyyy, hh = m_gee.groups()
        return datetime(int(yyyy), int(mm), int(dd), int(hh))

    m_a = re.search(r"(\d{8})(?:_(\d{2}))?", base)
    if m_a:
        ymd = m_a.group(1)
        hh = m_a.group(2) if m_a.group(2) else "00"
        return datetime.strptime(f"{ymd}{hh}", "%Y%m%d%H")

    raise ValueError(f"Timestamp não encontrado em: {filename}")


def list_sorted_tiffs(input_dir, max_files=None):
    files = glob.glob(os.path.join(input_dir, "*.tif"))
    parsed = []
    for f in files:
        try:
            parsed.append((parse_timestamp(f), f))
        except ValueError:
            continue
    parsed.sort(key=lambda x: x[0])
    if max_files is not None:
        parsed = parsed[:max_files]
    return parsed


def load_p95_grid(p95_file, stride=2):
    ds = xr.open_dataset(p95_file, engine="rasterio")
    var = list(ds.data_vars)[0]
    da = ds[var].squeeze(drop=True)
    p95 = da.values[::stride, ::stride]

    ny, nx = p95.shape
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    pixel_id = (yy * nx + xx).reshape(-1)

    return pd.DataFrame({"pixel_id": pixel_id, "p95": p95.reshape(-1)})


def extract_pixel_timeseries(sorted_files, stride=2):
    """Leitura incremental por arquivo, usando parte dos dados para desempenho."""
    rows = []
    base_shape = None

    for dt0, path in sorted_files:
        ds = xr.open_dataset(path, engine="rasterio")
        var = list(ds.data_vars)[0]
        da = ds[var]

        y_dim = "y" if "y" in da.dims else "latitude"
        x_dim = "x" if "x" in da.dims else "longitude"
        shape = (da.sizes[y_dim], da.sizes[x_dim])

        if base_shape is None:
            base_shape = shape
            print(f"Shape base: {base_shape} ({os.path.basename(path)})")
        elif shape != base_shape:
            print(f"Ignorando {os.path.basename(path)}: shape {shape} != {base_shape}")
            continue

        if "band" in da.dims:
            arr = da.values[:, ::stride, ::stride]
        else:
            arr = da.values[::stride, ::stride][None, ...]

        nodata = da.attrs.get("_FillValue", None)
        nb, ny, nx = arr.shape
        yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        pixel_id = (yy * nx + xx).reshape(-1)

        for b in range(nb):
            t = dt0 + timedelta(hours=b)
            vals = arr[b].reshape(-1)
            frame = pd.DataFrame({
                "time": t,
                "pixel_id": pixel_id,
                "precipitation": vals,
            })
            if nodata is not None:
                frame = frame[frame["precipitation"] != nodata]
            rows.append(frame)

    if not rows:
        raise RuntimeError("Nenhum dado válido foi extraído dos TIFFs.")

    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["precipitation"])
    return out


def build_feature_target_table(raw_df, p95_df, p_values, q_values, fuzzy_config):
    slope = fuzzy_config.get("slope", 2.0)
    offset = fuzzy_config.get("offset", 0.0)

    data = raw_df.merge(p95_df, on="pixel_id", how="left")
    data = data.dropna(subset=["p95"]).sort_values(["pixel_id", "time"])

    g = data.groupby("pixel_id", group_keys=False)

    max_p = max(p_values)
    for p in range(1, max_p + 1):
        data[f"lag_{p}"] = g["precipitation"].shift(p)

    for q in sorted(set(q_values)):
        fut = g["precipitation"].shift(-q)
        data[f"target_q{q}"] = 1.0 / (1.0 + np.exp(-slope * (fut - (data["p95"] + offset))))

    return data.dropna()


def make_splits(df, split_cfg):
    train_end = pd.Timestamp(split_cfg["train_end"])
    val_end = pd.Timestamp(split_cfg["val_end"])

    train = df[df["time"] <= train_end]
    val = df[(df["time"] > train_end) & (df["time"] <= val_end)]
    test = df[df["time"] > val_end]
    return train, val, test


def train_evaluate(train_df, val_df, test_df, feature_cols, target_col):
    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
    )

    preds = np.clip(model.predict(X_test), 0.0, 1.0)
    brier = brier_score_loss((y_test > 0.5).astype(int), preds)
    return {"brier_score": float(brier), "best_iter": int(model.best_iteration_ or 0)}


def run_grid_search(base_df, p_values, q_values, split_cfg, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for p, q in product(p_values, q_values):
        exp_name = f"exp_P{p}_Q{q}"
        exp_dir = os.path.join(output_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)

        feature_cols = [f"lag_{i}" for i in range(1, p + 1)]
        target_col = f"target_q{q}"
        cols = ["time", "pixel_id", *feature_cols, target_col]

        df = base_df[cols].dropna()
        train_df, val_df, test_df = make_splits(df, split_cfg)
        if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
            print(f"{exp_name}: split vazio, pulando")
            continue

        metrics = train_evaluate(train_df, val_df, test_df, feature_cols, target_col)

        test_df.to_parquet(os.path.join(exp_dir, "test_set.parquet"), index=False)
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "P": p,
            "Q": q,
            "feature_cols": feature_cols,
            "target_col": target_col,
            "rows": {
                "train": len(train_df),
                "val": len(val_df),
                "test": len(test_df),
            },
            "metrics": metrics,
        }
        with open(os.path.join(exp_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        results.append({"experiment": exp_name, **metrics})
        print(f"{exp_name}: Brier={metrics['brier_score']:.4f}")

    if results:
        pd.DataFrame(results).to_csv(os.path.join(output_dir, "summary_results.csv"), index=False)


def main():
    input_dir = "data/raster/cube"
    p95_file = "data/raster/p95_precipitacao.tif"
    output_dir = "experiments_fast"

    # Parâmetros de performance
    max_files = None
    spatial_stride = 2
    sample_frac = 0.20

    p_values = [3, 6]
    q_values = [1, 3]
    fuzzy_config = {"slope": 2.0, "offset": 0.0}
    split_cfg = {"train_end": "2022-12-31", "val_end": "2023-02-28"}

    sorted_files = list_sorted_tiffs(input_dir, max_files=max_files)
    if not sorted_files:
        raise RuntimeError("Nenhum arquivo TIFF elegível encontrado.")

    print(f"Arquivos usados: {len(sorted_files)}")
    raw_df = extract_pixel_timeseries(sorted_files, stride=spatial_stride)
    p95_df = load_p95_grid(p95_file, stride=spatial_stride)
    base_df = build_feature_target_table(raw_df, p95_df, p_values, q_values, fuzzy_config)

    if sample_frac < 1.0:
        base_df = base_df.sample(frac=sample_frac, random_state=42)

    os.makedirs(output_dir, exist_ok=True)
    base_df.to_parquet(os.path.join(output_dir, "base_table.parquet"), index=False)
    run_grid_search(base_df, p_values, q_values, split_cfg, output_dir)


if __name__ == "__main__":
    main()