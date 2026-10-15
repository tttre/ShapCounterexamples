import numpy as np
from Utils.score_functions import polynomial_score
from Utils.generators import bit_generator
from pathlib import Path


def generate_dataset(n_features):
    ### Bit Generator Dataset
    X = np.asarray(bit_generator(n_features), dtype=np.float32)
    scores = polynomial_score(X).astype(np.float32)
    return {"X": X, "scores": scores}


def main():
    data = generate_dataset(8)

    cache_dir = Path(__file__).resolve().parents[2] / "caches" / "data_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_path = cache_dir / "bit_data.npz"
    np.savez(output_path, **data)

    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()