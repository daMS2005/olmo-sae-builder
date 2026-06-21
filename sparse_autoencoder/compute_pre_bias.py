"""Compute and cache the current layer's pre_bias for SAE training.

Run this once after collecting activations. Future runner.py launches load the
small cache file instead of rescanning all activation chunks.
"""

import torch

import sparse_autoencoder.sparse_autoencoder.model as model
from sparse_autoencoder.sparse_autoencoder.runner import (
    ACTS_DIR,
    LAYER,
    N_INPUTS,
    NORMALIZE_INPUTS,
    PRE_BIAS_CACHE,
    activation_files,
    load_activation_file,
)


def compute_pre_bias() -> tuple[torch.Tensor, dict[str, object]]:
    """Average activations in the same space runner.py will encode."""
    files = activation_files()
    activation_sum = torch.zeros(N_INPUTS, dtype=torch.float64)
    n_tokens = 0

    for file_index, path in enumerate(files, start=1):
        acts = load_activation_file(path)
        acts = acts.float()
        if NORMALIZE_INPUTS:
            acts, _, _ = model.LN(acts)
        activation_sum += acts.sum(dim=0).double()
        n_tokens += int(acts.shape[0])
        del acts

        if file_index % 25 == 0:
            print(f"pre_bias mean pass: {file_index}/{len(files)} files", flush=True)

    if n_tokens == 0:
        raise ValueError(f"No token activations found in {ACTS_DIR}")

    metadata = {
        "layer": LAYER,
        "n_inputs": N_INPUTS,
        "normalize": NORMALIZE_INPUTS,
        "acts_dir": str(ACTS_DIR),
        "n_files": len(files),
        "n_tokens": n_tokens,
        "file_names": [path.name for path in files],
        "file_sizes": [path.stat().st_size for path in files],
    }
    return (activation_sum / n_tokens).float(), metadata


def main() -> None:
    PRE_BIAS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    pre_bias, metadata = compute_pre_bias()
    torch.save({"pre_bias": pre_bias, "metadata": metadata}, PRE_BIAS_CACHE)
    print(
        f"saved pre_bias cache {PRE_BIAS_CACHE} "
        f"from {metadata['n_tokens']:,} token activations",
        flush=True,
    )


if __name__ == "__main__":
    main()
