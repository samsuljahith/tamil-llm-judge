"""Step 05: frozen MuRIL embeddings for every prompt + response (Tamil and Hindi).

LiteJudge (step 06) does not read text. It reads one 768-number vector per response, made here
by Google's MuRIL encoder (a BERT model pre-trained on 17 Indian languages, including Tamil and
Hindi). "Frozen" means MuRIL's weights are never changed: it is used only as a fixed feature
extractor, so the only thing trained later is LiteJudge's small scoring layer.

For each response in data/processed/responses.csv this script:
  1. builds the input "[CLS] prompt [SEP] response [SEP]", so the vector describes the response
     *as an answer to its prompt* (a prompt-only signal would be identical for both answers in a
     pair and would cancel out in a pairwise model);
  2. if the response is too long for MuRIL's 512-token limit, splits it into overlapping windows,
     each starting with the full prompt;
  3. runs MuRIL once per window and averages the last-layer vectors of the response tokens only
     (mean pooling; prompt and special tokens are left out);
  4. averages the window vectors, giving one 768-dim vector per response.

Reproducibility: MuRIL is pinned to one HuggingFace commit, the weight file's SHA-256 is
recorded, the script runs on the CPU in float32 with deterministic algorithms and seed 42, and
it re-embeds a few responses at the end to check the numbers come out the same.

Tokenizer check: transformers 5.x prints a warning about an "incorrect regex pattern" (a
Mistral tokenizer issue). MuRIL uses a BERT WordPiece tokenizer, which does not use that regex.
To show this rather than assume it, the script glues each text's tokens back together and checks
that no character is lost except U+FFFD (broken characters already in the source data, which
BERT's text cleaner removes by design) and words that become [UNK].

Usage:  python scripts/05_embed_muril.py                # both languages
        python scripts/05_embed_muril.py --language tamil
Reads:  data/processed/prompts.csv, data/processed/responses.csv
Writes: data/embeddings/muril_<language>.npz and data/embeddings/MANIFEST.json (git-ignored)
Needs:  about 960 MB download on the first run (HuggingFace cache in ~/.cache/huggingface).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "data" / "embeddings"

SEED = 42
LANGUAGES = ("tamil", "hindi")
MODEL_ID = "google/muril-base-cased"
# The exact HuggingFace commit of MuRIL (last changed 2022-06-10). Pinning it means a later
# upload to the model page cannot silently change our embeddings.
MODEL_REVISION = "afd9f36c7923d54e97903922ff1b260d091d202f"
# Only the PyTorch weights and the tokenizer files; the TensorFlow and Flax copies (2.5 GB) are skipped.
MODEL_FILES = ["config.json", "pytorch_model.bin", "vocab.txt", "tokenizer_config.json",
               "special_tokens_map.json"]
MAX_TOKENS = 512    # MuRIL's (BERT's) maximum input length, special tokens included
OVERLAP = 128       # tokens shared by consecutive windows, so no sentence is cut without context
BATCH_SIZE = 8      # windows per forward pass; small enough for an 8 GB Mac
N_REPEAT_CHECK = 4  # responses re-embedded at the end to check the run is deterministic


# ---------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------
def sha256(path: Path) -> str:
    """SHA-256 of a file, read in 1 MB blocks so a large file does not fill the memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_muril() -> tuple[Path, AutoTokenizer, AutoModel]:
    """Download (or reuse from the cache) the pinned MuRIL files and load them in evaluation mode."""
    path = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION, allow_patterns=MODEL_FILES))
    tokenizer = AutoTokenizer.from_pretrained(path)
    # The checkpoint also holds MuRIL's pre-training heads (cls.*). AutoModel loads only the encoder,
    # so transformers lists those heads as "UNEXPECTED"; that is expected and harmless.
    model = AutoModel.from_pretrained(path, dtype=torch.float32)
    model.eval()  # switches dropout off, so the same input always gives the same output
    for param in model.parameters():
        param.requires_grad_(False)  # frozen: no gradients, weights never change
    return path, tokenizer, model


def check_tokenization(tokenizer, texts: list[str]) -> dict:
    """Glue each text's WordPiece tokens back together and compare with the original characters.

    WordPiece only splits text ("##" marks a piece that continues a word), so the glued tokens
    should equal the text with whitespace and control characters removed. Texts with [UNK]
    are counted separately, because [UNK] replaces a whole word.
    """
    n_unk = n_lossy = 0
    for text in texts:
        tokens = tokenizer.tokenize(text)
        if tokenizer.unk_token in tokens:
            n_unk += 1
            continue
        glued = "".join(t[2:] if t.startswith("##") else t for t in tokens)
        expected = "".join(ch for ch in unicodedata.normalize("NFC", text)
                           if not ch.isspace()                    # BERT splits on whitespace
                           and unicodedata.category(ch)[0] != "C"  # and drops control characters
                           and ch != "\ufffd")               # and the U+FFFD replacement char
        if unicodedata.normalize("NFC", glued) != expected:
            n_lossy += 1
    if n_lossy:
        raise SystemExit(f"Tokenizer check failed: {n_lossy} texts lose characters. Do not use these embeddings.")
    return {"texts_checked": len(texts), "texts_with_unk": n_unk, "texts_losing_characters": n_lossy}


# ---------------------------------------------------------------------------------------------
# Building the input windows
# ---------------------------------------------------------------------------------------------
def make_windows(tokenizer, prompt: str, response: str) -> list[dict]:
    """Token ids for "[CLS] prompt [SEP] response-window [SEP]", one dict per window.

    `response_mask` marks the response tokens (1) so pooling can ignore the prompt and the
    special tokens (0).
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    room = MAX_TOKENS - len(prompt_ids) - 3  # 3 special tokens: [CLS], [SEP], [SEP]
    if room <= OVERLAP:
        raise SystemExit(f"Prompt too long ({len(prompt_ids)} tokens) to leave room for the response.")
    step = room - OVERLAP  # how far each window moves along the response
    starts = list(range(0, max(len(response_ids) - OVERLAP, 1), step))
    windows = []
    for start in starts:
        piece = response_ids[start:start + room]
        ids = [tokenizer.cls_token_id] + prompt_ids + [tokenizer.sep_token_id] + piece + [tokenizer.sep_token_id]
        # Segment ids tell BERT which tokens are the first text (0: prompt) and the second (1: response).
        segments = [0] * (len(prompt_ids) + 2) + [1] * (len(piece) + 1)
        mask = [0] * (len(prompt_ids) + 2) + [1] * len(piece) + [0]
        windows.append({"input_ids": ids, "token_type_ids": segments, "response_mask": mask})
    return windows


def pad(windows: list[dict]) -> dict[str, torch.Tensor]:
    """Right-pad a batch of windows to the same length; attention_mask=0 tells MuRIL to ignore the padding."""
    width = max(len(w["input_ids"]) for w in windows)
    batch = {k: [] for k in ("input_ids", "token_type_ids", "attention_mask", "response_mask")}
    for w in windows:
        gap = width - len(w["input_ids"])
        batch["input_ids"].append(w["input_ids"] + [0] * gap)  # 0 is [PAD] in MuRIL's vocabulary
        batch["token_type_ids"].append(w["token_type_ids"] + [0] * gap)
        batch["attention_mask"].append([1] * len(w["input_ids"]) + [0] * gap)
        batch["response_mask"].append(w["response_mask"] + [0] * gap)
    return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


# ---------------------------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------------------------
@torch.inference_mode()  # no gradient bookkeeping: faster, less memory, and nothing can be trained
def embed_windows(model, windows: list[dict]) -> np.ndarray:
    """One mean-pooled vector per window: the average last-layer vector of its response tokens."""
    vectors = []
    for i in range(0, len(windows), BATCH_SIZE):
        batch = pad(windows[i:i + BATCH_SIZE])
        hidden = model(input_ids=batch["input_ids"], token_type_ids=batch["token_type_ids"],
                       attention_mask=batch["attention_mask"]).last_hidden_state  # (batch, tokens, 768)
        weights = batch["response_mask"].unsqueeze(-1).to(hidden.dtype)            # (batch, tokens, 1)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1)                # mean over response tokens
        vectors.append(pooled.numpy())
    return np.concatenate(vectors)


def embed_language(language: str, tokenizer, model, prompts: pd.DataFrame,
                   responses: pd.DataFrame) -> tuple[dict, dict]:
    """Embed every response of one language. Returns the arrays to save and summary statistics."""
    prompt_text = prompts.loc[prompts.language == language].set_index("prompt_id")["prompt"]
    rows = responses.loc[responses.language == language].reset_index(drop=True)
    missing = set(rows.prompt_id) - set(prompt_text.index)
    if missing:
        raise SystemExit(f"{language}: {len(missing)} responses have no prompt text.")

    # Build every window first, remembering which response it belongs to.
    windows, owner = [], []
    for i, row in rows.iterrows():
        for w in make_windows(tokenizer, prompt_text[row.prompt_id], row.response):
            windows.append(w)
            owner.append(i)
    owner = np.array(owner)

    start = time.time()
    window_vectors = embed_windows(model, windows)
    seconds = time.time() - start

    # A response's vector is the plain average of its window vectors (most responses have one window).
    embeddings = np.stack([window_vectors[owner == i].mean(axis=0) for i in range(len(rows))]).astype(np.float32)
    n_windows = np.bincount(owner, minlength=len(rows))
    n_tokens = np.array([len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in rows.response])

    if not np.isfinite(embeddings).all():
        raise SystemExit(f"{language}: embeddings contain NaN or infinite values.")

    arrays = {
        "embeddings": embeddings,                       # (n_responses, 768), float32
        "prompt_id": rows.prompt_id.to_numpy(dtype=str),  # join keys for step 06
        "model": rows.model.to_numpy(dtype=str),
        "n_tokens": n_tokens,
        "n_windows": n_windows,
    }
    stats = {
        "responses": int(len(rows)),
        "windows": int(len(windows)),
        "responses_with_more_than_one_window": int((n_windows > 1).sum()),
        "max_response_tokens": int(n_tokens.max()),
        "seconds": round(seconds, 1),
    }
    return arrays, stats


def repeat_check(language: str, tokenizer, model, prompts: pd.DataFrame, responses: pd.DataFrame,
                 saved: np.ndarray) -> float:
    """Re-embed the first few responses one at a time (no padding) and return the largest difference.

    This tests two things at once: the run is deterministic, and padding inside a batch did not
    change the result.
    """
    prompt_text = prompts.loc[prompts.language == language].set_index("prompt_id")["prompt"]
    rows = responses.loc[responses.language == language].reset_index(drop=True).head(N_REPEAT_CHECK)
    worst = 0.0
    for i, row in rows.iterrows():
        windows = make_windows(tokenizer, prompt_text[row.prompt_id], row.response)
        again = np.stack([embed_windows(model, [w])[0] for w in windows]).mean(axis=0)
        worst = max(worst, float(np.abs(again - saved[i]).max()))
    return worst


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--language", choices=[*LANGUAGES, "all"], default="all")
    args = parser.parse_args()
    languages = LANGUAGES if args.language == "all" else (args.language,)

    # Determinism: fixed seed, CPU only, and PyTorch's deterministic kernels.
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)

    prompts = pd.read_csv(PROCESSED / "prompts.csv")
    responses = pd.read_csv(PROCESSED / "responses.csv")

    model_path, tokenizer, model = load_muril()
    print(f"MuRIL {MODEL_REVISION[:7]} loaded on CPU ({sum(p.numel() for p in model.parameters()):,} parameters, frozen)")

    texts = list(responses.response) + list(prompts.prompt)
    token_check = check_tokenization(tokenizer, texts)
    print(f"Tokenizer check: {token_check['texts_checked']} texts, none lose characters "
          f"({token_check['texts_with_unk']} contain [UNK] words)")

    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update({
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "weights_sha256": sha256(model_path / "pytorch_model.bin"),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": "cpu",
        "dtype": "float32",
        "seed": SEED,
        "input_format": "[CLS] prompt [SEP] response window [SEP]",
        "pooling": "mean of last-layer vectors over response tokens; mean over windows",
        "max_tokens": MAX_TOKENS,
        "window_overlap_tokens": OVERLAP,
        "responses_csv_sha256": sha256(PROCESSED / "responses.csv"),
        "prompts_csv_sha256": sha256(PROCESSED / "prompts.csv"),
        "tokenizer_check": token_check,
    })
    manifest.setdefault("languages", {})

    for language in languages:
        arrays, stats = embed_language(language, tokenizer, model, prompts, responses)
        diff = repeat_check(language, tokenizer, model, prompts, responses, arrays["embeddings"])
        if diff > 1e-4:
            raise SystemExit(f"{language}: repeat check failed (max difference {diff:.2e}).")
        out_file = OUT / f"muril_{language}.npz"
        np.savez(out_file, **arrays)  # plain arrays only, so step 06 can load it without pickle
        stats.update({"file": out_file.name, "file_sha256": sha256(out_file),
                      "repeat_check_max_abs_diff": diff,
                      "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
        manifest["languages"][language] = stats
        print(f"{language:<6} {stats['responses']} responses | {stats['windows']} windows "
              f"({stats['responses_with_more_than_one_window']} responses need >1) | "
              f"{stats['seconds']} s | repeat check max diff {diff:.1e} -> {out_file.relative_to(ROOT)}")

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Manifest: {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
