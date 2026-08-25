"""
Predict Ux, Uy and P for a user-supplied SDF field using the trained model.

The SDF can be any array the user provides — it does not need to come from
data/CNN_df.pkl, so there is no ground truth to compare against, only the
predicted fields.

The SDF must be provided as a .npy file containing a single 2D array of
shape (H, W) matching `configs/config.yaml`'s `model.input_shape` (256x512
by default), in the same convention used throughout this project: physical
distance to the wall, with cells inside the airfoil <= 0.

Usage:
    python -m src.inference.predict_sdf --sdf path/to/my_sdf.npy
    python -m src.inference.predict_sdf --sdf path/to/my_sdf.npy --name wing_01
"""

import argparse
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.unet_hybrid import unet_model_multi_output
from src.inference.predict import predict_fields
from src.visualization.plot_fields import plot_predictions, plot_sdf


def load_config(config_path="configs/config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def predict_sdf(sdf_path, config_path="configs/config.yaml", out_name=None):
    cfg = load_config(config_path)
    paths_cfg = cfg["paths"]
    H, W = cfg["model"]["input_shape"][:2]

    sdf = np.load(sdf_path)
    if sdf.shape != (H, W):
        raise ValueError(
            f"SDF shape {sdf.shape} does not match model input shape ({H}, {W}) "
            f"from configs/config.yaml (model.input_shape)."
        )
    sdf = np.maximum(sdf, 0.0)  # inside the airfoil (SDF < 0) -> 0, same convention as training

    # ---- Load normalization params and model weights persisted by train.py ----
    nparams = np.load(paths_cfg["norm_params"])
    norm_params = {
        "Ux": (float(nparams["ux_mn"]), float(nparams["ux_mx"])),
        "Uy": (float(nparams["uy_mn"]), float(nparams["uy_mx"])),
        "P":  (float(nparams["p_mn"]),  float(nparams["p_mx"])),
    }
    sdf_mn, sdf_mx = float(nparams["sdf_mn"]), float(nparams["sdf_mx"])
    sdf_norm = (sdf - sdf_mn) / (sdf_mx - sdf_mn) if sdf_mx != sdf_mn else np.zeros_like(sdf)
    sdf_norm = sdf_norm.astype(np.float32)

    model = unet_model_multi_output(input_shape=tuple(cfg["model"]["input_shape"]))
    model.load_weights(paths_cfg["saved_model"])

    fields, sdf_vis = predict_fields(model, sdf_norm, norm_params)

    # ---- Save plots + raw predicted arrays ----
    results_dir = Path(paths_cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    name = out_name or Path(sdf_path).stem

    fig_sdf = plot_sdf(sdf_vis, case_label=name)
    fig_sdf.savefig(results_dir / f"{name}_SDF.png", dpi=100)
    plt.close(fig_sdf)

    fig = plot_predictions(fields)
    fig.savefig(results_dir / f"{name}_prediction.png", dpi=100)
    plt.close(fig)

    npz_path = results_dir / f"{name}_fields.npz"
    np.savez(npz_path, Ux=fields["Ux"], Uy=fields["Uy"], P=fields["P"])

    print(f"Prediction for: {sdf_path}")
    for key, arr in fields.items():
        print(f"  {key}: min={arr.min():.4f}  max={arr.max():.4f}  mean={arr.mean():.4f}")
    print(f"Prediction plot : {results_dir / f'{name}_prediction.png'}")
    print(f"Fields (.npz)   : {npz_path}")

    return fields


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sdf", type=str, required=True,
                         help="Path to a .npy file with the SDF array, shape (H, W).")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                         help="Path to the config YAML file.")
    parser.add_argument("--name", type=str, default=None,
                         help="Basename for the output files (default: SDF filename stem).")
    args = parser.parse_args()
    predict_sdf(args.sdf, config_path=args.config, out_name=args.name)
