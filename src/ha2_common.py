from __future__ import annotations

import json
import random
import re
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests
import torch
from tqdm import tqdm


PROMPTS_RAW_URL = "https://raw.githubusercontent.com/openai/CLIP/main/data/prompts.md"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs(paths: List[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def fetch_text(url: str) -> str:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def extract_dataset_section(markdown_text: str, dataset_name: str) -> str:
    parts = re.split(r"^##\s+(.+)$", markdown_text, flags=re.M)
    for i in range(1, len(parts), 2):
        name = parts[i].strip()
        body = parts[i + 1]
        if name == dataset_name:
            return body
    raise ValueError(f"Dataset section not found: {dataset_name}")


def parse_single_quoted_list(block_text: str) -> List[str]:
    return re.findall(r"'([^']+)'", block_text)


def parse_prompts_dataset(markdown_text: str, dataset_name: str) -> Tuple[List[str], List[str]]:
    section = extract_dataset_section(markdown_text, dataset_name)
    classes_match = re.search(r"classes\s*=\s*\[(.*?)\]\n\n", section, flags=re.S)
    templates_match = re.search(r"templates\s*=\s*\[(.*?)\]\n", section, flags=re.S)
    if not classes_match or not templates_match:
        raise ValueError(f"Could not parse classes/templates for {dataset_name}")

    classes_block = classes_match.group(1)
    templates_block = templates_match.group(1)
    classes = parse_single_quoted_list(classes_block)
    templates = parse_single_quoted_list(templates_block)
    return classes, templates


def normalize_label(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return text


def build_caltech101_mapping(dataset_categories: List[str], prompt_classes: List[str]) -> Dict[str, str]:
    prompt_norm = {normalize_label(c): c for c in prompt_classes}
    alias = {
        "background google": "background",
        "faces": "centered face",
        "faces easy": "off-center face",
        "leopards": "leopard",
        "motorbikes": "motorbike",
        "airplanes": "airplane",
        "car side": "side of a car",
        "cougar body": "body of a cougar cat",
        "cougar face": "face of a cougar cat",
        "crocodile head": "head of a  crocodile",
        "flamingo head": "head of a flamingo",
        "snoopy": "snoopy (cartoon beagle)",
        "yin yang": "yin and yang symbol",
    }

    mapping: Dict[str, str] = {}
    for cat in dataset_categories:
        norm_cat = normalize_label(cat)
        if norm_cat in alias:
            mapping[cat] = alias[norm_cat]
            continue

        if norm_cat in prompt_norm:
            mapping[cat] = prompt_norm[norm_cat]
            continue

        mapped = None
        if norm_cat.endswith("s") and norm_cat[:-1] in prompt_norm:
            mapped = prompt_norm[norm_cat[:-1]]
        elif f"{norm_cat}s" in prompt_norm:
            mapped = prompt_norm[f"{norm_cat}s"]

        if mapped is None:
            raise ValueError(f"No prompt class mapping for dataset category: {cat}")
        mapping[cat] = mapped

    return mapping


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def download_file(url: str, dst_path: Path, retries: int = 5) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while attempt < retries:
        attempt += 1
        try:
            existing = dst_path.stat().st_size if dst_path.exists() else 0
            headers = {"Range": f"bytes={existing}-"} if existing > 0 else {}
            with requests.get(url, stream=True, timeout=120, headers=headers) as resp:
                if resp.status_code not in (200, 206):
                    resp.raise_for_status()

                content_len = int(resp.headers.get("content-length", 0))
                total = existing + content_len if content_len > 0 else 0
                mode = "ab" if existing > 0 and resp.status_code == 206 else "wb"
                initial = existing if mode == "ab" else 0

                with dst_path.open(mode) as f, tqdm(
                    total=total,
                    initial=initial,
                    unit="B",
                    unit_scale=True,
                    desc=f"{dst_path.name} (try {attempt}/{retries})",
                ) as pbar:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        pbar.update(len(chunk))

            if dst_path.stat().st_size > 0:
                return

        except Exception:
            if attempt >= retries:
                raise
            time.sleep(2 * attempt)


def is_valid_zip(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with zipfile.ZipFile(path, "r") as zf:
            return zf.testzip() is None
    except Exception:
        return False


def extract_zip(zip_path: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst_dir)


def extract_tar(archive_path: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as tf:
        tf.extractall(dst_dir)
