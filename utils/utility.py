import numpy as np
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the main script.")
    parser.add_argument(
        "-i",
        "--index",
        type=int,
        default=5,
        help="Index value for the loop (default is 5)."
    )
    return parser.parse_args()


def to_cuda(tensor: torch.Tensor) -> torch.Tensor: 
    return tensor.cuda() if torch.cuda.is_available() else tensor

def create_results_folder(idx: int) -> None:
    if not os.path.exists(f"results/test{idx+1}"):
        os.makedirs(f"results/test{idx+1}")

def save_weights(encoder_weights: list, softmax_weights: torch.Tensor, idx: int):  
    file_path = f"results/test{idx+1}/weights_test.pth"
    weights_dict = {
        "encoder_weights": encoder_weights,
        "softmax_weights": softmax_weights
    }
    torch.save(weights_dict, file_path)

def save_plot(cm: np.ndarray, idx: int) -> None: 
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig(f"results/test{idx+1}/confusion_matrix.png")

def save_report(report: str, idx: int):
    with open(f"results/test{idx+1}/classification_report.txt", "w") as file:
        file.write(report)

def load_dae_config(file_path: str) -> dict:
    cnf_dae = np.loadtxt(file_path, delimiter = ",")
    return {
        "nclasses"    : int(cnf_dae[0]),
        "nframe"      : int(cnf_dae[1]),
        "frame_size"  : int(cnf_dae[2]),
        "p_training"  : cnf_dae[3],
        "encoder_act" : int(cnf_dae[4]),
        "max_iter"    : int(cnf_dae[5]),
        "batch_size"  : int(cnf_dae[6]),
        "alpha"       : cnf_dae[7],
        "encoders"    : cnf_dae[8:],
    }

def load_softmax_config(file_path: str) -> dict:
    cnf_softmax = np.loadtxt(file_path, delimiter = ",")
    return {
        "max_iter"  : int(cnf_softmax[0]),
        "alpha"     : int(cnf_softmax[1]),
        "batch_size": int(cnf_softmax[2])
    }


def load_raw_data(file_name: str, n_classes: int)-> dict:
    return {
        f"class{i}": np.loadtxt(
            f"{file_name}/class{i}.csv", delimiter=",")
        for i in range(1, n_classes + 1)
    }


def save_data(features: list, p_training: float) -> None:
    file  = "data/processed_data"
    X, Y  = features[0], features[1]
    p     = 1 - p_training
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=p)
    np.savetxt(f"{file}/X_train.csv", X_train, delimiter=",", fmt="%.4f")
    np.savetxt(f"{file}/X_test.csv", X_test, delimiter=",", fmt="%.4f")
    np.savetxt(f"{file}/Y_train.csv", Y_train, delimiter=",", fmt="%.4f")
    np.savetxt(f"{file}/Y_test.csv", Y_test, delimiter=",", fmt="%.4f")

def load_data_trn():
  x_train = np.loadtxt(open("data/processed_data/X_train.csv", "rb"), delimiter=",")
  y_train = np.loadtxt(open("data/processed_data/Y_train.csv", "rb"), delimiter=",")
  return x_train, y_train

def load_data_tst():
  x_train = np.loadtxt(open("data/processed_data/X_test.csv", "rb"), delimiter=",")
  y_train = np.loadtxt(open("data/processed_data/Y_test.csv", "rb"), delimiter=",")
  return x_train, y_train