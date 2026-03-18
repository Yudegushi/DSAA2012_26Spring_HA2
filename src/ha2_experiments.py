from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import clip
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import Caltech101
from tqdm import tqdm


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "-")


def resolve_caltech_root(data_root: Path) -> Path:
    candidates = [
        data_root,
        data_root / "caltech_fallback",
    ]
    for root in candidates:
        marker = root / "caltech101" / "101_ObjectCategories"
        if marker.exists():
            return root
    raise FileNotFoundError(
        f"Could not find Caltech101 under {data_root}. Expected one of: "
        f"{[str(c / 'caltech101' / '101_ObjectCategories') for c in candidates]}"
    )


def load_caltech_dataset(root: Path, transform=None) -> Caltech101:
    return Caltech101(root=str(root), download=False, transform=transform)


def format_prompt(template: str, class_name: str) -> str:
    if "{}" in template:
        return template.format(class_name)
    if "{CLASS}" in template:
        return template.replace("{CLASS}", class_name)
    return template


def encode_images(
    dataset: Caltech101,
    indices: np.ndarray,
    model,
    device: str,
    batch_size: int,
    num_workers: int,
    normalize: bool = True,
    use_amp: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    subset = Subset(dataset, indices.tolist())
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    model.eval()
    all_feats: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    start = time.perf_counter()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    autocast_enabled = use_amp and device == "cuda"
    autocast_dtype = torch.float16

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Encode images", leave=False):
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_enabled):
                feats = model.encode_image(images)
            feats = feats.float()
            if normalize:
                feats = F.normalize(feats, dim=-1)
            all_feats.append(feats.cpu())
            all_labels.append(labels.cpu())

    elapsed = time.perf_counter() - start
    feats_np = torch.cat(all_feats, dim=0).numpy().astype(np.float32)
    labels_np = torch.cat(all_labels, dim=0).numpy().astype(np.int64)
    indices_np = np.array(indices, dtype=np.int64)

    peak_mem_gb = 0.0
    if device == "cuda":
        peak_mem_gb = float(torch.cuda.max_memory_allocated() / (1024**3))

    runtime = {
        "elapsed_sec": float(elapsed),
        "peak_mem_gb": peak_mem_gb,
        "num_samples": int(len(indices_np)),
        "samples_per_sec": float(len(indices_np) / elapsed) if elapsed > 0 else math.nan,
    }
    return feats_np, labels_np, indices_np, runtime


def build_text_features_per_template(
    model,
    class_names: Sequence[str],
    templates: Sequence[str],
    device: str,
    normalize: bool = True,
) -> np.ndarray:
    model.eval()
    template_feats = []
    with torch.no_grad():
        for template in templates:
            prompts = [format_prompt(template, cls) for cls in class_names]
            tokens = clip.tokenize(prompts, truncate=True).to(device)
            text_feat = model.encode_text(tokens).float()
            if normalize:
                text_feat = F.normalize(text_feat, dim=-1)
            template_feats.append(text_feat.cpu())
    stacked = torch.stack(template_feats, dim=0)  # [T, C, D]
    return stacked.numpy().astype(np.float32)


def predict_feature_mean(image_feats: np.ndarray, text_per_template: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_t = torch.from_numpy(image_feats)
    text_t = torch.from_numpy(text_per_template)
    image_t = F.normalize(image_t, dim=-1)
    text_t = F.normalize(text_t, dim=-1)
    text_mean = text_t.mean(dim=0)
    text_mean = F.normalize(text_mean, dim=-1)
    logits = image_t @ text_mean.T
    conf, pred = torch.max(logits, dim=1)
    return logits.numpy(), pred.numpy().astype(np.int64), conf.numpy().astype(np.float32)


def predict_feature_mean_with_options(
    image_feats: np.ndarray,
    text_per_template: np.ndarray,
    normalize_image: bool,
    normalize_text: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_t = torch.from_numpy(image_feats)
    text_t = torch.from_numpy(text_per_template)
    if normalize_image:
        image_t = F.normalize(image_t, dim=-1)
    if normalize_text:
        text_t = F.normalize(text_t, dim=-1)
    text_mean = text_t.mean(dim=0)
    if normalize_text:
        text_mean = F.normalize(text_mean, dim=-1)
    logits = image_t @ text_mean.T
    conf, pred = torch.max(logits, dim=1)
    return logits.numpy(), pred.numpy().astype(np.int64), conf.numpy().astype(np.float32)


def predict_logit_mean(image_feats: np.ndarray, text_per_template: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_t = torch.from_numpy(image_feats)
    text_t = torch.from_numpy(text_per_template)  # [T, C, D]
    image_t = F.normalize(image_t, dim=-1)
    text_t = F.normalize(text_t, dim=-1)
    logits_t = torch.einsum("nd,tcd->ntc", image_t, text_t)
    logits = logits_t.mean(dim=1)
    conf, pred = torch.max(logits, dim=1)
    return logits.numpy(), pred.numpy().astype(np.int64), conf.numpy().astype(np.float32)


def predict_logit_mean_with_options(
    image_feats: np.ndarray,
    text_per_template: np.ndarray,
    normalize_image: bool,
    normalize_text: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_t = torch.from_numpy(image_feats)
    text_t = torch.from_numpy(text_per_template)  # [T, C, D]
    if normalize_image:
        image_t = F.normalize(image_t, dim=-1)
    if normalize_text:
        text_t = F.normalize(text_t, dim=-1)
    logits_t = torch.einsum("nd,tcd->ntc", image_t, text_t)
    logits = logits_t.mean(dim=1)
    conf, pred = torch.max(logits, dim=1)
    return logits.numpy(), pred.numpy().astype(np.int64), conf.numpy().astype(np.float32)


def evaluate_predictions(pred: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(np.int64)
    labels = labels.astype(np.int64)
    acc = float((pred == labels).mean())
    return {"accuracy": acc, "num_samples": int(labels.shape[0])}


def template_subsets(
    templates: Sequence[str],
    k: int,
    seeds: Sequence[int],
    force_include: Sequence[str] | None = None,
) -> List[Tuple[int, List[str]]]:
    force_include = list(force_include or [])
    force_include = [t for t in force_include if t in templates]
    remaining = [t for t in templates if t not in force_include]
    out: List[Tuple[int, List[str]]] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        need = max(0, k - len(force_include))
        if need > len(remaining):
            raise ValueError(f"Cannot sample {k} templates with force_include={len(force_include)}")
        chosen = list(force_include)
        if need > 0:
            idx = rng.choice(len(remaining), size=need, replace=False)
            chosen.extend([remaining[i] for i in sorted(idx.tolist())])
        out.append((int(seed), chosen))
    return out


def save_npz(path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **kwargs)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}
