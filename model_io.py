# ==========================================================
# Model I/O Utilities (Production-grade)
# รองรับ:
# - backward compatibility
# - scaler restore
# - torch model restore
# ==========================================================

import os
import pickle

MODEL_DIR = "saved_models"

os.makedirs(MODEL_DIR, exist_ok=True)


def save_model(artifact, model_name):

    save_path = os.path.join(
        MODEL_DIR,
        f"{model_name}.pkl"
    )

    model = artifact["model"]

    artifact_to_save = {
        "config": artifact["config"],
        "test_true": artifact.get("test_true"),
        "test_pred": artifact.get("test_pred"),
        "learning_curve": artifact.get("learning_curve"),
    }

    # ==================================================
    # CASE 1: LSTM (PyTorch)
    # ==================================================
    if hasattr(model, "model") and hasattr(model.model, "state_dict"):

        artifact_to_save["model_type"] = "pytorch"
        artifact_to_save["model_state_dict"] = model.model.state_dict()

        if hasattr(model, "scaler"):
            artifact_to_save["scaler"] = model.scaler

    # ==================================================
    # CASE 2: sklearn model (LinearRegression)
    # ==================================================
    else:

        artifact_to_save["model_type"] = "sklearn"
        artifact_to_save["model_object"] = model

    # save
    with open(save_path, "wb") as f:

        pickle.dump(
            artifact_to_save,
            f
        )

# ==========================================================
# LOAD MODEL
# ==========================================================

def load_model(model_name):

    load_path = os.path.join(
        MODEL_DIR,
        f"{model_name}.pkl"
    )

    with open(load_path, "rb") as f:

        artifact = pickle.load(f)

    # ==================================================
    # LSTM model
    # ==================================================
    if artifact.get("model_type") == "pytorch":

        from src.models.factory import create_model

        config = artifact["config"]

        model = create_model(
            config["model_type"],
            hidden_size=config.get("hidden_size"),
            num_layers=config.get("num_layers"),
            dropout=config.get("dropout"),
            epochs=config.get("epochs"),
        )

        model.model.load_state_dict(
            artifact["model_state_dict"]
        )

        if "scaler" in artifact:
            model.scaler = artifact["scaler"]

        model.model.eval()

        artifact["model"] = model

    # ==================================================
    # sklearn model
    # ==================================================
    elif artifact.get("model_type") == "sklearn":

        artifact["model"] = artifact["model_object"]

    return artifact

# ==========================================================
# LIST MODELS
# ==========================================================

def list_models():

    files = os.listdir(MODEL_DIR)

    return [
        f.replace(".pkl", "")
        for f in files
        if f.endswith(".pkl")
    ]