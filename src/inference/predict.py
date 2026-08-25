import numpy as np


def predict_fields(model, sdf_norm, norm_params):
    """
    Runs the prediction and denormalizes the output fields.

    Args:
        model      : loaded Keras model
        sdf_norm   : array (H, W), normalized SDF
        norm_params: dict with min/max per field
                     {"Ux": (min, max), "Uy": (min, max), "P": (min, max)}

    Returns:
        dict with denormalized fields: {"Ux", "Uy", "P"}
        and the flipped SDF for visualization
    """
    sdf_2d    = np.squeeze(sdf_norm)              # (H, W), independent of the input shape
    sdf_input = sdf_2d.reshape(1, *sdf_2d.shape, 1)
    ux_pred, uy_pred, p_pred = model.predict(sdf_input, verbose=0)

    def denorm(arr, key):
        mn, mx = norm_params[key]
        return arr * (mx - mn) + mn

    sdf_vis  = np.flipud(sdf_2d)
    ux_field = np.flipud(np.squeeze(denorm(ux_pred[0], "Ux")))
    uy_field = np.flipud(np.squeeze(denorm(uy_pred[0], "Uy")))
    p_field  = np.flipud(np.squeeze(denorm(p_pred[0],  "P")))

    # Enforce boundary condition: velocity and pressure = 0 inside the airfoil
    mask = sdf_vis == 0
    ux_field[mask] = 0
    uy_field[mask] = 0
    p_field[mask]  = 0

    return {"Ux": ux_field, "Uy": uy_field, "P": p_field}, sdf_vis
