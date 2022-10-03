from dataclasses import dataclass, asdict
import json


@dataclass
class Params:
    name: str
    epochs: int
    batch_size: int
    learning_rate: float
    git_commit_hash: str
    best_val_acc: float
    best_val_acc_epoch: int


PARAMS_FILE = "model_params.json"


def save_params(run_path, params):
    with open(run_path / PARAMS_FILE, "w") as f:
        json.dump(asdict(params), f, indent=2)


def load_params(run_path):
    with open(run_path / PARAMS_FILE, "r") as f:
        return Params(**json.load(f))
