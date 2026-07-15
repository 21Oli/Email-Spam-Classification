import json
import numpy as np
import pandas as pd


def load_metrics():

    with open("results/metrics.json") as f:

        return json.load(f)


def load_report():

    return pd.read_csv(
        "results/classification_report.csv"
    )


def load_confusion_matrix():

    return np.load(
        "results/confusion_matrix.npy"
    )