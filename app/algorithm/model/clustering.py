import numpy as np, pandas as pd
import joblib
import sys
import os, warnings
from sklearn.cluster import SpectralClustering, KMeans

warnings.filterwarnings("ignore")

model_fname = "model.save"

MODEL_NAME = "clustering_base_spectral_clustering"


class ClusteringModel:
    def __init__(self, K, verbose=False, **kwargs) -> None:
        self.K = K
        self.verbose = verbose
        self.cluster_centers = None
        self.feature_names_in_ = None

        self.model = self.build_model()

    def build_model(self):
        model = SpectralClustering(
            n_clusters=self.K,
            verbose=self.verbose,
            eigen_solver="arpack",
            affinity="nearest_neighbors",
            random_state=1,
        )
        return model

    def fit_predict(self, *args, **kwargs):
        return self.model.fit_predict(*args, **kwargs)

    def save(self, model_path):
        joblib.dump(self, os.path.join(model_path, model_fname))

    @classmethod
    def load(cls, model_path):
        clusterer = joblib.load(os.path.join(model_path, model_fname))
        return clusterer


def save_model(model, model_path):
    model.save(model_path)


def load_model(model_path):
    model = ClusteringModel.load(model_path)
    return model
