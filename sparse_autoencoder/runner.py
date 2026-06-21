from pathlib import Path
import os
import sys

# Let this file work both as `python -m ...runner` from the repo root and as a
# direct script path from another cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

import sparse_autoencoder.sparse_autoencoder.loss as loss
import sparse_autoencoder.sparse_autoencoder.model as model


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable in the common shell formats."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "y", "on"}


def env_path(name: str, default: Path) -> Path:
    """Read a path env var, falling back to a repo-root-relative default."""
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


DATA_ROOT = env_path("SAE_DATA_ROOT", REPO_ROOT / "data")

# The activation collector saves one folder per OLMo layer. Train one SAE at a
# time so we do not keep several huge SAE dictionaries in VRAM.
LAYER = int(os.environ.get("SAE_LAYER", "7"))
ACTS_ROOT = env_path("SAE_ACTS_ROOT", DATA_ROOT / "olmo_acts")
ACTS_DIR = env_path("SAE_ACTS_DIR", ACTS_ROOT / f"layer_{LAYER:02d}")
CHECKPOINT_ROOT = env_path("SAE_CHECKPOINT_ROOT", DATA_ROOT / "sae_checkpoints")
CHECKPOINT_DIR = env_path("SAE_CHECKPOINT_DIR", CHECKPOINT_ROOT / f"layer_{LAYER:02d}")
NORMALIZE_INPUTS = env_flag("SAE_NORMALIZE_INPUTS")
PRE_BIAS_KIND = "ln" if NORMALIZE_INPUTS else "raw"
PRE_BIAS_ROOT = env_path("SAE_PRE_BIAS_ROOT", DATA_ROOT / "pre_bias")
PRE_BIAS_CACHE = env_path(
    "SAE_PRE_BIAS_CACHE",
    PRE_BIAS_ROOT / f"layer_{LAYER:02d}_{PRE_BIAS_KIND}_pre_bias.pt",
)

# OLMo-3-7B activations have width 4096. Keep this configurable so the same
# runner can train on another activation width without editing code.
N_INPUTS = int(os.environ.get("SAE_N_INPUTS", "4096"))
N_LATENTS = int(os.environ.get("SAE_N_LATENTS", "131072"))
TOP_K = int(os.environ.get("SAE_TOP_K", "64"))

# Batch size controls how many saved activation vectors go through one
# forward/backward/update step. Lower this first if the L4 runs out of memory.
BATCH_SIZE = int(os.environ.get("SAE_BATCH_SIZE", "256"))
EPOCHS = int(os.environ.get("SAE_EPOCHS", "1"))
LR = float(os.environ.get("SAE_LR", "3e-4"))

# Each activation file is about 50k token vectors. A full model+Adam checkpoint
# is very large, so periodic saves overwrite one "latest" file instead of
# keeping a new multi-GB checkpoint every time.
SAVE_EVERY_FILES = int(os.environ.get("SAE_SAVE_EVERY_FILES", "25"))
LOG_EVERY_STEPS = int(os.environ.get("SAE_LOG_EVERY_STEPS", "500"))


def activation_files() -> list[Path]:
    """Find activation chunks produced by collect_data.py for one layer."""
    files = sorted(ACTS_DIR.glob("acts_*.pt"))
    if not files:
        raise FileNotFoundError(f"No activation chunks found in {ACTS_DIR}")
    return files


def load_activation_file(path: Path) -> torch.Tensor:
    """Load one activation chunk and verify it matches this SAE's input width."""
    acts = torch.load(path, map_location="cpu")
    if acts.shape[-1] != N_INPUTS:
        raise ValueError(f"{path} has activation width {acts.shape[-1]}, expected {N_INPUTS}")
    return acts


def load_pre_bias_cache(files: list[Path]) -> torch.Tensor:
    """Load the pre_bias computed by compute_pre_bias.py."""
    if not PRE_BIAS_CACHE.exists():
        raise FileNotFoundError(
            f"Missing pre_bias cache {PRE_BIAS_CACHE}. "
            "Run: python -m sparse_autoencoder.sparse_autoencoder.compute_pre_bias"
        )

    cache = torch.load(PRE_BIAS_CACHE, map_location="cpu")
    metadata = cache.get("metadata", {})
    expected_file_names = [path.name for path in files]
    expected_file_sizes = [path.stat().st_size for path in files]

    if metadata.get("layer") != LAYER:
        raise ValueError(f"pre_bias cache is for layer {metadata.get('layer')}, expected {LAYER}")
    if metadata.get("n_inputs") != N_INPUTS:
        raise ValueError(
            f"pre_bias cache has n_inputs={metadata.get('n_inputs')}, expected {N_INPUTS}"
        )
    if metadata.get("normalize") != NORMALIZE_INPUTS:
        raise ValueError(
            f"pre_bias cache normalize={metadata.get('normalize')}, "
            f"expected {NORMALIZE_INPUTS}"
        )
    if metadata.get("file_names") != expected_file_names:
        raise ValueError("pre_bias cache was computed from a different activation file list")
    if metadata.get("file_sizes") != expected_file_sizes:
        raise ValueError("pre_bias cache was computed from different activation file contents")

    pre_bias = cache["pre_bias"].float()
    if pre_bias.shape != (N_INPUTS,):
        raise ValueError(f"pre_bias shape is {tuple(pre_bias.shape)}, expected {(N_INPUTS,)}")

    print(
        f"loaded pre_bias from {PRE_BIAS_CACHE} "
        f"({metadata.get('n_tokens', 'unknown')} token activations)",
        flush=True,
    )
    return pre_bias


def unit_norm_tied_weights_(sae: model.Autoencoder) -> None:
    """Keep tied encoder rows / decoder columns at length 1."""
    with torch.no_grad():
        sae.encoder.weight.div_(sae.encoder.weight.norm(dim=1, keepdim=True).clamp_min(1e-8))


def unit_norm_tied_weight_grad_adjustment_(sae: model.Autoencoder) -> None:
    """Remove gradient components that only change feature-vector length."""
    if sae.encoder.weight.grad is None:
        return
    with torch.no_grad():
        parallel = (sae.encoder.weight * sae.encoder.weight.grad).sum(dim=1, keepdim=True)
        sae.encoder.weight.grad.sub_(parallel * sae.encoder.weight)


def save_checkpoint(
    *,
    sae: model.Autoencoder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    file_index: int,
    global_step: int,
    mean_loss: float,
    path: Path,
) -> None:
    """Save enough state to keep training later instead of starting over.

    `file_index` records how far through the activation-file list we were when
    this checkpoint was written. The runner does not auto-resume yet, but this
    metadata makes the checkpoint useful when we add resume or load it manually.
    """
    checkpoint = {
        "epoch": epoch,
        "file_index": file_index,
        "global_step": global_step,
        "layer": LAYER,
        "n_inputs": N_INPUTS,
        "n_latents": N_LATENTS,
        "top_k": TOP_K,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "normalize": NORMALIZE_INPUTS,
        "mean_loss": mean_loss,
        "model": sae.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)
    print(f"saved checkpoint {path}", flush=True)


def train() -> None:
    """Train one TopK SAE from saved OLMo activation chunk files."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    files = activation_files()
    print(
        "training config: "
        f"layer={LAYER}, files={len(files)}, n_latents={N_LATENTS}, "
        f"top_k={TOP_K}, batch_size={BATCH_SIZE}, lr={LR}, normalize={NORMALIZE_INPUTS}",
        flush=True,
    )
    print(
        "training paths: "
        f"acts_dir={ACTS_DIR}, checkpoint_dir={CHECKPOINT_DIR}, "
        f"pre_bias_cache={PRE_BIAS_CACHE}",
        flush=True,
    )
    pre_bias = load_pre_bias_cache(files)

    # The SAE reconstructs one OLMo layer's activation vectors, so input and
    # reconstruction width are both 4096.
    sae = model.Autoencoder(
        n_latents=N_LATENTS,
        n_inputs=N_INPUTS,
        tied=True,
        activation=model.TopK(k=TOP_K),
        normalize=NORMALIZE_INPUTS,
    ).to("cuda")

    # pre_bias lives in the same space that the SAE encodes. With
    # NORMALIZE_INPUTS=False, that is raw activation space.
    with torch.no_grad():
        sae.pre_bias.copy_(pre_bias.to(sae.pre_bias.device))
    unit_norm_tied_weights_(sae)

    optimizer = torch.optim.Adam(sae.parameters(), lr=LR)
    sae.train()

    global_step = 0

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        n_batches = 0
        recent_loss = 0.0
        recent_batches = 0

        for file_index, path in enumerate(files, start=1):
            # Load one saved activation file onto CPU, then move only the current
            # mini-batch to CUDA. This avoids loading the whole activation
            # dataset into VRAM.
            acts = load_activation_file(path)

            for start in range(0, acts.shape[0], BATCH_SIZE):
                x = acts[start : start + BATCH_SIZE].to("cuda").float()

                optimizer.zero_grad()
                _, latents, recons = sae(x)

                # TopK already enforces sparsity by keeping only TOP_K features,
                # so start with no extra L1 pressure.
                loss_v = loss.autoencoder_loss(
                    recons,
                    x,
                    latent_activations=latents,
                    l1_weight=0.0,
                )

                loss_v.backward()
                unit_norm_tied_weights_(sae)
                unit_norm_tied_weight_grad_adjustment_(sae)
                optimizer.step()
                unit_norm_tied_weights_(sae)

                epoch_loss += loss_v.item()
                recent_loss += loss_v.item()
                n_batches += 1
                recent_batches += 1
                global_step += 1

                if global_step % LOG_EVERY_STEPS == 0:
                    print(
                        f"epoch={epoch} file={file_index}/{len(files)} "
                        f"step={global_step} "
                        f"recent_loss={recent_loss / max(recent_batches, 1):.6f} "
                        f"mean_loss={epoch_loss / max(n_batches, 1):.6f}",
                        flush=True,
                    )
                    recent_loss = 0.0
                    recent_batches = 0

            del acts

            if file_index % SAVE_EVERY_FILES == 0:
                save_checkpoint(
                    sae=sae,
                    optimizer=optimizer,
                    epoch=epoch,
                    file_index=file_index,
                    global_step=global_step,
                    mean_loss=epoch_loss / max(n_batches, 1),
                    path=CHECKPOINT_DIR / "olmo_sae_latest.pt",
                )

        mean_loss = epoch_loss / max(n_batches, 1)
        save_checkpoint(
            sae=sae,
            optimizer=optimizer,
            epoch=epoch,
            file_index=len(files),
            global_step=global_step,
            mean_loss=mean_loss,
            path=CHECKPOINT_DIR / f"olmo_sae_epoch_{epoch}.pt",
        )
        print(f"epoch {epoch}: mean_loss={mean_loss:.6f}", flush=True)


if __name__ == "__main__":
    train()
