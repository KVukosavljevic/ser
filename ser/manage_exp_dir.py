from dataclasses import asdict
from datetime import date,datetime
from pathlib import Path
import json
import torch

PROJECT_ROOT = Path(__file__).parent.parent
EXP_DIR = PROJECT_ROOT / "results"

def create_exp_dir(name, exp_dir=EXP_DIR):

    time = datetime.now().strftime("%d-%m-%Y-%H-%M")
    exp_dir = exp_dir / name / time

    exp_dir.mkdir(parents=True, exist_ok=True)

    return exp_dir

def save_model_params(exp_dir, hyperparams, best_val_acc = 0.0, best_val_acc_epoch = 0.0 ):

    model_dict = asdict(hyperparams)

    model_dict['best_val_acc'] = best_val_acc
    model_dict['best_val_acc_epoch'] = best_val_acc_epoch
    
    with open(exp_dir / "model_params.json", "w") as f:
        json.dump(model_dict, f)


def save_model(exp_dir, model,name):
    torch.save(model, exp_dir / (name + "_model.pt") )