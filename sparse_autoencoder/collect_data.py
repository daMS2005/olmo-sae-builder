"""Collect OLMo-3-7B-Think activations for SAE training.

This file does one expensive job:

    local text shards -> token chunks -> OLMo forward pass -> saved activations

The important memory rule is that the full text dataset, the full tokenized
dataset, and the full activation dataset are never loaded at once. The script
reads one local text row at a time, sends only the current token batch to the
GPU, saves activation chunks to disk, and then moves on.
"""

from __future__ import annotations

import gzip
import json
import os
import random
import shutil
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer


def env_path(name: str, default: Path) -> Path:
    """Read a path env var, falling back to a repo-root-relative default."""
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


# Anchor defaults to the repository root so collection works from any cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = env_path("SAE_DATA_ROOT", REPO_ROOT / "data")
MODELS_ROOT = env_path("SAE_MODELS_ROOT", REPO_ROOT / "models")
MODEL_PATH = env_path(
    "OLMO_MODEL_PATH",
    env_path("SAE_MODEL_DIR", MODELS_ROOT / "Olmo-3-7B-Think"),
)
DATA_DIR = env_path("SAE_RAW_TEXT_DIR", DATA_ROOT / "raw_text")
ACTS_DIR = env_path("SAE_ACTS_ROOT", DATA_ROOT / "olmo_acts")


def target_layers_from_env(default: list[int]) -> list[int]:
    """Let one VM command choose which layer to collect without editing code."""
    raw_layers = os.environ.get("SAE_TARGET_LAYERS") or os.environ.get("SAE_LAYER")
    if raw_layers is None:
        return default
    return [int(layer.strip()) for layer in raw_layers.split(",") if layer.strip()]


# Collect one layer at a time because each layer's activation cache is huge.
TARGET_LAYERS = target_layers_from_env([7])
MAX_SEQ_LEN = 8192
LENGTH_BUCKETS = [(512, 2048), (3500, 5000), (7000, 8192)]
TOKENS_PER_ACT_FILE = 50_000

BATCH_SIZE = 1
SEED = 0
MODEL_DTYPE = torch.bfloat16
SAVE_DTYPE = torch.float16

# Leave as None for the real run. Set to a small number, e.g. 20_000, when you
# want to smoke-test the whole pipeline without collecting a large activation set.
MAX_ACTIVATION_TOKENS: int | None = None

DOLMA_REPO_ID = "allenai/dolma"
DOLMA_URL_LIST = "urls/v1_6-sample.txt"
DOLMA_EXPECTED_SHARDS = 103
DOLCI_REPO_ID = "allenai/Dolci-Instruct-SFT"
DOLCI_PARQUET_FILES = [f"data/train-{i:05d}-of-00015.parquet" for i in range(15)]


def is_complete_file(path: Path) -> bool:
    """A local shard is usable only if it exists and has real bytes in it."""
    return path.is_file() and path.stat().st_size > 0


def download_url(url: str, destination: Path) -> Path:
    """Download one URL to destination unless the local file already exists."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if is_complete_file(destination):
        print(f"using existing {destination}")
        return destination

    tmp_destination = destination.with_suffix(destination.suffix + ".part")
    print(f"downloading {url}")
    # olmo-data.org accepts normal HTTP clients but rejects Python's default
    # urllib user-agent, so we identify this as a regular script download.
    request = Request(url, headers={"User-Agent": "sae-builder/0.1"})
    with urlopen(request) as response, tmp_destination.open("wb") as f:
        shutil.copyfileobj(response, f)
    tmp_destination.replace(destination)
    return destination


def download_dolma_shards() -> list[Path]:
    """Download every Dolma v1_6-sample JSONL gzip shard listed by AllenAI."""
    dolma_dir = DATA_DIR / "dolma_v1_6_sample"
    dolma_dir.mkdir(parents=True, exist_ok=True)

    # If the full Dolma sample is already on disk, avoid even touching the Hub.
    existing_files = sorted(path for path in dolma_dir.glob("*.json.gz") if is_complete_file(path))
    if len(existing_files) >= DOLMA_EXPECTED_SHARDS:
        print(f"using {len(existing_files)} existing Dolma shards from {dolma_dir}")
        return existing_files

    url_list_path = hf_hub_download(
        repo_id=DOLMA_REPO_ID,
        repo_type="dataset",
        filename=DOLMA_URL_LIST,
    )
    urls = [
        line.strip()
        for line in Path(url_list_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    local_files = []
    for i, url in enumerate(urls, start=1):
        filename = Path(urlparse(url).path).name
        destination = dolma_dir / filename
        print(f"dolma shard {i}/{len(urls)}: {filename}")
        if is_complete_file(destination):
            print(f"using existing {destination}")
            local_files.append(destination)
            continue

        local_files.append(download_url(url, destination))

    return local_files


def download_dolci_shards() -> list[Path]:
    """Download every Dolci train parquet shard into the local raw text folder."""
    dolci_dir = DATA_DIR / "dolci_instruct_sft"
    dolci_dir.mkdir(parents=True, exist_ok=True)

    local_files = []
    for i, filename in enumerate(DOLCI_PARQUET_FILES, start=1):
        destination = dolci_dir / filename
        print(f"dolci shard {i}/{len(DOLCI_PARQUET_FILES)}: {filename}")
        if is_complete_file(destination):
            print(f"using existing {destination}")
            local_files.append(destination)
        else:

            path = hf_hub_download(
                repo_id=DOLCI_REPO_ID,
                repo_type="dataset",
                filename=filename,
                local_dir=dolci_dir,
            )
            local_files.append(Path(path))

    return local_files


def ensure_local_text_data() -> tuple[list[Path], list[Path]]:
    """Make sure all raw text shards are present locally before collecting."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dolma_files = download_dolma_shards()
    dolci_files = download_dolci_shards()
    return dolma_files, dolci_files


def text_from_row(row: dict) -> str:
    """Turn Dolma/Dolci row shapes into plain text for tokenization."""
    value = row.get("text")
    if isinstance(value, str) and value.strip():
        return value

    messages = row.get("messages")
    if isinstance(messages, list):
        parts = []
        for message in messages:
            if isinstance(message, dict):
                role = message.get("role")
                content = message.get("content") or message.get("text") or ""
                parts.append(f"{role}: {content}" if role else str(content))
            else:
                parts.append(str(message))
        return "\n".join(part for part in parts if part)

    parts = []
    for key in ("instruction", "input", "prompt", "question", "output", "response"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value)
    if parts:
        return "\n".join(parts)

    return ""


def iter_dolma_texts(path: Path):
    """Yield text strings from one local Dolma `.json.gz` shard."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            text = text_from_row(row)
            if text:
                yield text


def iter_dolci_texts(path: Path):
    """Yield text strings from one local Dolci parquet shard without loading it."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "Reading Dolci parquet shards requires pyarrow. Install it with:\n"
            "    pip install pyarrow"
        ) from exc

    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=1024):
        for row in batch.to_pylist():
            text = text_from_row(row)
            if text:
                yield text


def iter_local_texts(dolma_files: list[Path], dolci_files: list[Path]):
    """Yield text rows from all local shards in a deterministic shuffled order."""
    rng = random.Random(SEED)
    files = [("dolma", path) for path in dolma_files] + [("dolci", path) for path in dolci_files]
    rng.shuffle(files)

    for kind, path in files:
        print(f"reading {kind}: {path}")
        if kind == "dolma":
            yield from iter_dolma_texts(path)
        else:
            yield from iter_dolci_texts(path)


def choose_chunk_length(rng: random.Random) -> int:
    """Choose the next chunk length so examples are not all the same size."""
    low, high = rng.choice(LENGTH_BUCKETS)
    high = min(high, MAX_SEQ_LEN)
    if low > high:
        raise ValueError(f"Invalid bucket {(low, high)} for MAX_SEQ_LEN={MAX_SEQ_LEN}")
    return rng.randint(low, high)


def iter_token_chunks(texts, tokenizer):
    """Pack streamed text into variable-length token chunks."""
    rng = random.Random(SEED)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id for packed chunks.")

    token_buffer: list[int] = []
    target_len = choose_chunk_length(rng)

    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            continue

        token_buffer.extend(ids)
        token_buffer.append(eos_token_id)

        while len(token_buffer) >= target_len:
            yield token_buffer[:target_len]
            token_buffer = token_buffer[target_len:]
            target_len = choose_chunk_length(rng)


def iter_batches(chunks, batch_size: int):
    """Group token chunks into small batches for OLMo forward passes."""
    batch = []
    for chunk in chunks:
        batch.append(chunk)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def make_batch(chunks: list[list[int]], pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Pad variable-length chunks into tensors for one model call."""
    seq_lens = [len(chunk) for chunk in chunks]
    max_len = max(seq_lens)

    input_ids = torch.full((len(chunks), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(chunks), max_len), dtype=torch.long)

    for row_idx, chunk in enumerate(chunks):
        seq_len = len(chunk)
        input_ids[row_idx, :seq_len] = torch.tensor(chunk, dtype=torch.long)
        attention_mask[row_idx, :seq_len] = 1

    return input_ids, attention_mask, seq_lens


def model_input_device(olmo) -> torch.device:
    """Find the device that should receive input ids for this loaded model."""
    return olmo.model.embed_tokens.weight.device


def next_file_index(layer: int) -> int:
    """Continue numbering after existing activation files for this layer."""
    layer_dir = ACTS_DIR / f"layer_{layer:02d}"
    if not layer_dir.exists():
        return 0
    return len(list(layer_dir.glob("acts_*.pt")))


def save_activation_chunk(
    *,
    layer: int,
    file_index: int,
    activations: list[torch.Tensor],
    seq_lens: list[int],
) -> int:
    """Save one buffered activation tensor and append a manifest row."""
    if not activations:
        return file_index

    layer_dir = ACTS_DIR / f"layer_{layer:02d}"
    layer_dir.mkdir(parents=True, exist_ok=True)

    acts = torch.cat(activations, dim=0).contiguous()
    path = layer_dir / f"acts_{file_index:06d}.pt"
    while path.exists():
        file_index += 1
        path = layer_dir / f"acts_{file_index:06d}.pt"

    torch.save(acts, path)

    manifest_row = {
        "layer": layer,
        "file": str(path),
        "n_token_activations": int(acts.shape[0]),
        "d_model": int(acts.shape[1]),
        "dtype": str(acts.dtype),
        "n_sequences": len(seq_lens),
        "min_seq_len": min(seq_lens),
        "max_seq_len": max(seq_lens),
    }
    with (ACTS_DIR / "manifest.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(manifest_row) + "\n")

    print(f"saved {path} with shape {tuple(acts.shape)}")
    return file_index + 1


def load_olmo_and_tokenizer():
    """Load the tokenizer and OLMo model for activation collection."""
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    olmo = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=MODEL_DTYPE,
        device_map="cuda",
    )
    olmo.eval()
    olmo.config.use_cache = False

    return olmo, tokenizer


def collect_activations() -> None:
    """Download text shards locally, stream them, and save OLMo activations."""
    ACTS_DIR.mkdir(parents=True, exist_ok=True)
    dolma_files, dolci_files = ensure_local_text_data()
    olmo, tokenizer = load_olmo_and_tokenizer()

    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer: int):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured[layer] = tensor.detach()

        return hook

    for layer in TARGET_LAYERS:
        handles.append(olmo.model.layers[layer].register_forward_hook(make_hook(layer)))

    buffers: dict[int, list[torch.Tensor]] = {layer: [] for layer in TARGET_LAYERS}
    buffer_tokens: dict[int, int] = {layer: 0 for layer in TARGET_LAYERS}
    buffer_seq_lens: dict[int, list[int]] = {layer: [] for layer in TARGET_LAYERS}
    file_indices = {layer: next_file_index(layer) for layer in TARGET_LAYERS}

    texts = iter_local_texts(dolma_files, dolci_files)
    chunks = iter_token_chunks(texts, tokenizer)
    batches = iter_batches(chunks, BATCH_SIZE)

    total_token_positions = 0
    input_device = model_input_device(olmo)
    assert tokenizer.pad_token_id is not None

    try:
        for token_chunks in batches:
            input_ids, attention_mask, seq_lens = make_batch(token_chunks, tokenizer.pad_token_id)
            input_ids = input_ids.to(input_device)
            attention_mask = attention_mask.to(input_device)

            captured.clear()
            with torch.no_grad():
                olmo(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

            for layer in TARGET_LAYERS:
                if layer not in captured:
                    raise RuntimeError(f"Layer {layer} hook did not capture activations.")

                layer_acts = captured[layer]
                real_token_mask = attention_mask.to(layer_acts.device).bool()
                real_acts = layer_acts[real_token_mask].to(SAVE_DTYPE).cpu()

                buffers[layer].append(real_acts)
                buffer_tokens[layer] += int(real_acts.shape[0])
                buffer_seq_lens[layer].extend(seq_lens)

                if buffer_tokens[layer] >= TOKENS_PER_ACT_FILE:
                    file_indices[layer] = save_activation_chunk(
                        layer=layer,
                        file_index=file_indices[layer],
                        activations=buffers[layer],
                        seq_lens=buffer_seq_lens[layer],
                    )
                    buffers[layer] = []
                    buffer_tokens[layer] = 0
                    buffer_seq_lens[layer] = []

            total_token_positions += sum(seq_lens)
            print(f"collected {total_token_positions:,} token positions; last seq_lens={seq_lens}")

            del input_ids, attention_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if (
                MAX_ACTIVATION_TOKENS is not None
                and total_token_positions >= MAX_ACTIVATION_TOKENS
            ):
                break

        for layer in TARGET_LAYERS:
            file_indices[layer] = save_activation_chunk(
                layer=layer,
                file_index=file_indices[layer],
                activations=buffers[layer],
                seq_lens=buffer_seq_lens[layer],
            )
    finally:
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    collect_activations()
