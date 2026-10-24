import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.Data.Utils.samplers import temp_sampler
from src.Data.Utils.score_functions import binary_score
from src.SHAP.shapely_values import kNN_shapely_values

pool = temp_sampler(10000, np.random.default_rng(1))
dataset = pd.read_csv(ROOT / "caches/data_cache/test_dataset.csv")
X = dataset[list(pool)].to_numpy()

dataset_pool = {name: dataset[name].to_numpy() for name in pool}

ranking = pd.DataFrame(index=list(pool))

for name, reference in {"large_pool": pool, "dataset": dataset_pool}.items():
    phi = np.array([
        kNN_shapely_values(
            x, reference, binary_score, continuous_features=[1], k=200
        )
        for x in X
    ])
    ranking[name] = np.abs(phi).mean(axis=0)

print(ranking.sort_values("large_pool", ascending=False))