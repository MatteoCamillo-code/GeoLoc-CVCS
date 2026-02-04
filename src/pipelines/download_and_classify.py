from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

from models.classifier import SceneClassifierWithConfidence


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    input_metadata: Path
    output_dir: Path
    output_csv: Path
    confidence_threshold: float | None = None  # If None, uses default from DownloadAndClassifyConfig


@dataclass(frozen=True)
class DownloadAndClassifyConfig:
    confidence_threshold: float = 0.7
    max_retries: int = 2
    timeout: int = 5
    batch_size: int = 128  # Increased for better GPU utilization
    num_download_threads: int = 16
    test_limit: int | None = None


@dataclass
class DatasetRunResult:
    df_results: pd.DataFrame
    failed_downloads: list[dict]
    low_confidence: list[dict]
    indoor_rejected: list[dict]
    downloaded_count: int
    already_downloaded_count: int


def get_default_dataset_configs(project_root: Path) -> dict[str, DatasetConfig]:
    return {
        "mp16": DatasetConfig(
            name="mp16",
            input_metadata=project_root
            / "data"
            / "metadata"
            / "original-datasets"
            / "metadata_mp16_for_osvmini.csv",
            output_dir=project_root / "data" / "raw" / "mp16_images",
            output_csv=project_root
            / "data"
            / "metadata"
            / "places-classification"
            / "mp16_with_predictions.csv",
        ),
        "osv_mini": DatasetConfig(
            name="osv_mini",
            input_metadata=project_root / "data" / "metadata" / "original-datasets" / "train_mini.csv",
            output_dir=project_root / "data" / "raw" / "osv5m" / "train_images",
            output_csv=project_root
            / "data"
            / "metadata"
            / "places-classification"
            / "osv_mini_with_predictions.csv",
        ),
    }


def select_dataset_configs(
    dataset_configs: dict[str, DatasetConfig], dataset_mode: str
) -> list[DatasetConfig]:
    if dataset_mode == "both":
        return list(dataset_configs.values())
    if dataset_mode not in dataset_configs:
        raise ValueError(
            f"Unknown dataset_mode: {dataset_mode}. Use one of {list(dataset_configs.keys())} or 'both'."
        )
    return [dataset_configs[dataset_mode]]


def download_osv_mini_dataset() -> Path:
    try:
        import kagglehub
    except Exception as exc:
        raise ImportError(
            "kagglehub is required to download osv_mini. Install it and retry."
        ) from exc

    raw_path = kagglehub.dataset_download("josht000/osv-mini-129k")
    return Path(raw_path) / "osv5m"


def update_osv_mini_config(
    dataset_configs: dict[str, DatasetConfig],
    osv_mini_path: Path,
    output_csv: Path,
) -> dict[str, DatasetConfig]:
    dataset_configs = dict(dataset_configs)
    dataset_configs["osv_mini"] = DatasetConfig(
        name="osv_mini",
        input_metadata=osv_mini_path / "test_mini.csv",
        output_dir=osv_mini_path,
        output_csv=output_csv,
    )
    return dataset_configs


def download_image_from_url(url: str, max_retries: int, timeout: int) -> Image.Image | None:
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                continue
        except requests.exceptions.RequestException:
            return None
        except Exception:
            return None
    return None


def _is_already_downloaded(image_id: str, existing_files: set[str]) -> bool:
    if image_id in existing_files:
        return True

    if image_id.endswith(".jpg"):
        base_id = image_id.replace(".jpg", "")
        if base_id in existing_files:
            return True
    else:
        if f"{image_id}.jpg" in existing_files:
            return True

    id_unix = image_id.replace("\\", "/")
    id_win = image_id.replace("/", "\\")
    if id_unix in existing_files or id_win in existing_files:
        return True

    return False


def _build_existing_files(output_dir: Path) -> tuple[set[str], dict[str, Path]]:
    """Build set of existing file references and a filename->path lookup dict."""
    existing_files: set[str] = set()
    filename_to_path: dict[str, Path] = {}
    
    for p in output_dir.glob("**/*"):
        if p.is_file():
            existing_files.add(p.name)
            rel_path = p.relative_to(output_dir)
            existing_files.add(str(rel_path))
            existing_files.add(str(rel_path).replace("\\", "/"))
            # Store filename -> full path mapping for fast lookup
            filename_to_path[p.name] = p
    
    return existing_files, filename_to_path


def _download_single_image(row_tuple, max_retries: int, timeout: int) -> dict:
    idx, row = row_tuple
    img = download_image_from_url(row["url"], max_retries=max_retries, timeout=timeout)
    return {
        "idx": idx,
        "row": row,
        "img": img,
        "status": "success" if img is not None else "failed",
    }


def download_images_for_dataset(
    df_metadata: pd.DataFrame,
    output_dir: Path,
    config: DownloadAndClassifyConfig,
) -> tuple[list[dict], list[dict], int]:
    output_dir.mkdir(parents=True, exist_ok=True)

    df_to_process = df_metadata.head(config.test_limit) if config.test_limit else df_metadata
    
    print(f"Scanning {output_dir} for existing images...")
    existing_files, filename_to_path = _build_existing_files(output_dir)
    print(f"Found {len(existing_files)} existing file references")

    df_already_downloaded = df_to_process[
        df_to_process["id"].astype(str).apply(lambda x: _is_already_downloaded(x, existing_files))
    ]
    df_needs_download = df_to_process[
        ~df_to_process["id"].astype(str).apply(lambda x: _is_already_downloaded(x, existing_files))
    ]
    
    print(f"Already downloaded: {len(df_already_downloaded)} images")
    print(f"Need to download: {len(df_needs_download)} images")

    downloaded_data: list[dict] = []
    failed_downloads: list[dict] = []

    for _, row in df_already_downloaded.iterrows():
        img_id = str(row["id"])
        
        # Try different path variations
        possible_paths = [
            output_dir / img_id,
            output_dir / f"{img_id}.jpg",
            output_dir / img_id.replace(".jpg", ""),
        ]
        if not img_id.endswith(".jpg"):
            possible_paths.append(output_dir / f"{img_id}.jpg")
        
        img_path = next((p for p in possible_paths if p.exists() and p.is_file()), None)
        
        # If not found in direct paths, use filename lookup (fast O(1) lookup)
        if not img_path:
            search_name = img_id if img_id.endswith(".jpg") else f"{img_id}.jpg"
            img_path = filename_to_path.get(search_name)
        
        if img_path:
            downloaded_data.append({"row": row, "img_path": img_path})

    if len(df_needs_download) > 0:
        with ThreadPoolExecutor(max_workers=config.num_download_threads) as executor:
            future_to_row = {
                executor.submit(
                    _download_single_image, row_tuple, config.max_retries, config.timeout
                ): row_tuple
                for row_tuple in df_needs_download.iterrows()
            }
            with tqdm(total=len(future_to_row), desc="Downloading", unit="img") as pbar:
                for future in as_completed(future_to_row):
                    try:
                        result = future.result()
                        if result["status"] == "failed":
                            failed_downloads.append(
                                {
                                    "id": result["row"]["id"],
                                    "url": result["row"]["url"],
                                    "reason": "download_failed",
                                }
                            )
                        else:
                            img_filename = result["row"]["id"]
                            if not img_filename.endswith(".jpg"):
                                img_filename = f"{img_filename}.jpg"
                            img_path = output_dir / img_filename
                            img_path.parent.mkdir(parents=True, exist_ok=True)
                            result["img"].save(img_path)
                            downloaded_data.append({"row": result["row"], "img_path": img_path})
                    except Exception:
                        pass
                    pbar.update(1)

    return downloaded_data, failed_downloads, len(df_already_downloaded)


def classify_downloaded_images(
    downloaded_data: list[dict],
    classifier: SceneClassifierWithConfidence,
    config: DownloadAndClassifyConfig,
    threshold: float | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    results: list[dict] = []
    low_confidence: list[dict] = []
    indoor_rejected: list[dict] = []
    # Use dataset-specific threshold if provided, otherwise use default
    confidence_threshold = threshold if threshold is not None else config.confidence_threshold

    for batch_start in tqdm(
        range(0, len(downloaded_data), config.batch_size),
        desc="Classifying",
        unit="batch",
    ):
        batch_end = min(batch_start + config.batch_size, len(downloaded_data))
        batch_items = downloaded_data[batch_start:batch_end]

        batch_images: list[Image.Image] = []
        batch_rows: list[dict] = []

        for item in batch_items:
            try:
                img = Image.open(item["img_path"]).convert("RGB")
                batch_images.append(img)
                batch_rows.append(item["row"])
            except Exception:
                continue

        if len(batch_images) == 0:
            continue

        labels, confidences = classifier.process_images(batch_images)

        for row, label_int, conf in zip(batch_rows, labels, confidences):
            label_str = classifier.label_int_to_str(label_int)
            if label_str == "Indoor":
                indoor_rejected.append(
                    {"id": row["id"], "predicted_label": label_str, "confidence": conf}
                )
                continue

            if conf < confidence_threshold:
                low_confidence.append(
                    {"id": row["id"], "predicted_label": label_str, "confidence": conf}
                )
                continue

            # Handle both 'url' and 'thumb_original_url' column names
            url = row.get("url") or row.get("thumb_original_url", "")
            
            results.append(
                {
                    "id": str(row["id"]).replace(".jpg", ""),
                    "latitude": row["latitude"],
                    "longitude": row["longitude"],
                    "predicted_label": label_str,
                    "confidence": conf,
                    "url": url,
                }
            )

    return results, low_confidence, indoor_rejected


def process_dataset(
    dataset_config: DatasetConfig,
    config: DownloadAndClassifyConfig,
    classifier: SceneClassifierWithConfidence,
) -> DatasetRunResult:
    df_metadata = pd.read_csv(dataset_config.input_metadata)
    downloaded_data, failed_downloads, already_downloaded_count = download_images_for_dataset(
        df_metadata=df_metadata,
        output_dir=dataset_config.output_dir,
        config=config,
    )

    results, low_confidence, indoor_rejected = classify_downloaded_images(
        downloaded_data=downloaded_data,
        classifier=classifier,
        config=config,
        threshold=dataset_config.confidence_threshold,
    )

    df_results = pd.DataFrame(results)
    return DatasetRunResult(
        df_results=df_results,
        failed_downloads=failed_downloads,
        low_confidence=low_confidence,
        indoor_rejected=indoor_rejected,
        downloaded_count=len(downloaded_data),
        already_downloaded_count=already_downloaded_count,
    )


def run_pipeline(
    dataset_configs: Iterable[DatasetConfig],
    config: DownloadAndClassifyConfig,
) -> dict[str, DatasetRunResult]:
    classifier = SceneClassifierWithConfidence()
    results: dict[str, DatasetRunResult] = {}
    for dataset_config in dataset_configs:
        results[dataset_config.name] = process_dataset(dataset_config, config, classifier)
    return results


def save_results(dataset_config: DatasetConfig, result: DatasetRunResult) -> None:
    if len(result.df_results) > 0:
        dataset_config.output_csv.parent.mkdir(parents=True, exist_ok=True)
        result.df_results.to_csv(dataset_config.output_csv, index=False)
    
    if len(result.failed_downloads) > 0:
        failed_df = pd.DataFrame(result.failed_downloads)
        failed_path = dataset_config.output_csv.parent / f"{dataset_config.name}_failed_downloads.csv"
        failed_df.to_csv(failed_path, index=False)

    if len(result.low_confidence) > 0:
        low_conf_df = pd.DataFrame(result.low_confidence)
        low_conf_path = dataset_config.output_csv.parent / f"{dataset_config.name}_low_confidence.csv"
        low_conf_df.to_csv(low_conf_path, index=False)

    if len(result.indoor_rejected) > 0:
        indoor_df = pd.DataFrame(result.indoor_rejected)
        indoor_path = dataset_config.output_csv.parent / f"{dataset_config.name}_indoor_rejected.csv"
        indoor_df.to_csv(indoor_path, index=False)
