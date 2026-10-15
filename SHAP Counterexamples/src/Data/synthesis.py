### -------------------------------------------------------------------------------------------
### Generate .npz file containing dataset, specifically {data, scores}, where data is generated
### for some generator, and scores is generated for some score function.
### -------------------------------------------------------------------------------------------

import numpy as np
from Utils.score_functions import polynomial_score
from Utils.generators import uniform_bit_generator
from pathlib import Path

### Generate Dataset based on a generator and a score function, and save it to a .npz file.
def generate_dataset(n_features):
    """
    n_features: Number of features in the dataset.
    """
    X = np.asarray(uniform_bit_generator(n_features), dtype=np.float32)
    scores = polynomial_score(X).astype(np.float32)
    return {"X": X, "scores": scores}


def main():
    data = generate_dataset(3)
    print(data)

    cache_dir = Path(__file__).resolve().parents[2] / "caches" / "data_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_path = cache_dir / "bit_data.npz"
    np.savez(output_path, **data)

    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()