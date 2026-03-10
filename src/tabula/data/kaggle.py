from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
from typing import Iterable, Literal

from tabula.data.catalog import KaggleDatasetEntry, get_dataset_entry
from tabula.data.env import load_repo_env_file
from tabula.data.manifest import DatasetManifest, sanitize_dataset_id, write_manifest


@dataclass(frozen=True)
class CsvSummary:
    path: Path
    columns: list[str]
    row_count: int


@dataclass(frozen=True)
class KaggleSearchResult:
    slug: str
    title: str
    size_bytes: int
    last_updated: str
    download_count: int
    vote_count: int
    usability_rating: float

    @property
    def dataset_url(self) -> str:
        return f"https://www.kaggle.com/datasets/{self.slug}"


@dataclass(frozen=True)
class KagglePreparedDataset:
    dataset_id: str
    raw_dir: str
    processed_dir: str
    config_path: str
    target_column: str
    train_rows: int
    val_rows: int
    test_rows: int


def _load_env_file() -> dict[str, str]:
    return load_repo_env_file()


def _standard_kaggle_config_candidates() -> list[Path]:
    candidates: list[Path] = []
    config_dir = os.environ.get("KAGGLE_CONFIG_DIR")
    if config_dir:
        candidates.append(Path(config_dir) / "kaggle.json")
    home = Path.home()
    candidates.extend(
        [
            home / ".kaggle" / "kaggle.json",
            home / "AppData" / "Roaming" / "kaggle" / "kaggle.json",
        ]
    )
    return candidates


def _find_standard_kaggle_config() -> Path | None:
    for candidate in _standard_kaggle_config_candidates():
        if candidate.exists():
            return candidate
    return None


def _load_standard_kaggle_config() -> dict[str, str]:
    config_path = _find_standard_kaggle_config()
    if config_path is None:
        return {}
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    username = payload.get("username")
    key = payload.get("key")
    return {
        "KAGGLE_USERNAME": str(username) if username else "",
        "KAGGLE_KEY": str(key) if key else "",
    }


def _extract_kaggle_credentials(env_values: dict[str, str]) -> dict[str, str]:
    username = env_values.get("KAGGLE_USERNAME") or os.environ.get("KAGGLE_USERNAME")
    key = env_values.get("KAGGLE_KEY") or os.environ.get("KAGGLE_KEY")

    if username and key:
        return {"KAGGLE_USERNAME": username, "KAGGLE_KEY": key}

    token = env_values.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        return {}

    token_path = Path(token)
    if token_path.exists():
        token = token_path.read_text(encoding="utf-8").strip()

    try:
        parsed = json.loads(token)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        parsed_username = parsed.get("username")
        parsed_key = parsed.get("key")
        if parsed_username and parsed_key:
            return {"KAGGLE_USERNAME": str(parsed_username), "KAGGLE_KEY": str(parsed_key)}

    if ":" in token:
        parsed_username, parsed_key = token.split(":", 1)
        if parsed_username and parsed_key:
            return {"KAGGLE_USERNAME": parsed_username, "KAGGLE_KEY": parsed_key}

    fallback_username = username or _load_standard_kaggle_config().get("KAGGLE_USERNAME", "")
    if fallback_username and token:
        return {"KAGGLE_USERNAME": fallback_username, "KAGGLE_KEY": token}

    raise RuntimeError(
        "Found `KAGGLE_API_TOKEN`, but it is not usable as Kaggle credentials. "
        "Set `KAGGLE_USERNAME` and `KAGGLE_KEY`, or set `KAGGLE_API_TOKEN` to either "
        "a JSON string like {\"username\": \"...\", \"key\": \"...\"}, a path to `kaggle.json`, "
        "or a `username:key` string. A raw API key is also accepted when `KAGGLE_USERNAME` "
        "is available separately."
    )


def build_kaggle_env() -> dict[str, str]:
    env_values = _load_env_file()
    merged = os.environ.copy()
    try:
        credentials = _extract_kaggle_credentials(env_values)
    except RuntimeError:
        if _find_standard_kaggle_config() is None:
            raise
        credentials = {}
    merged.update(credentials)
    return merged


def _load_kaggle_credentials() -> dict[str, str]:
    env_values = _load_env_file()
    try:
        return _extract_kaggle_credentials(env_values)
    except RuntimeError:
        if _find_standard_kaggle_config() is not None:
            return {}
        raise


def resolve_kaggle_cli() -> str | None:
    cli = shutil.which("kaggle")
    if cli:
        return cli

    script_candidates = [
        Path(sysconfig.get_path("scripts")) / "kaggle.exe",
        Path(sysconfig.get_path("scripts")) / "kaggle",
        Path(sys.prefix) / "Scripts" / "kaggle.exe",
        Path(sys.prefix) / "Scripts" / "kaggle",
        Path(os.environ.get("APPDATA", "")) / "Python" / f"Python{sys.version_info.major}{sys.version_info.minor}" / "Scripts" / "kaggle.exe",
        Path(os.environ.get("APPDATA", "")) / "Python" / f"Python{sys.version_info.major}{sys.version_info.minor}" / "Scripts" / "kaggle",
    ]
    for candidate in script_candidates:
        if candidate.exists():
            return str(candidate)
    return None


def kaggle_auth_status() -> dict[str, str | bool]:
    cli_path = resolve_kaggle_cli()
    env_values = _load_env_file()
    standard_config = _find_standard_kaggle_config()
    try:
        _resolve_kagglehub()
        kagglehub_available = True
    except RuntimeError:
        kagglehub_available = False
    try:
        credentials = _extract_kaggle_credentials(env_values)
        credential_source = "resolved" if credentials else "missing"
        username = credentials.get("KAGGLE_USERNAME", "")
        masked_username = f"{username[:2]}***" if username else ""
        return {
            "cli_found": cli_path is not None,
            "cli_path": cli_path or "",
            "kagglehub_available": kagglehub_available,
            "credentials_resolved": bool(credentials),
            "credential_source": credential_source,
            "username_hint": masked_username,
            "standard_config_found": standard_config is not None,
            "standard_config_path": str(standard_config) if standard_config else "",
        }
    except RuntimeError as exc:
        if standard_config is not None:
            return {
                "cli_found": cli_path is not None,
                "cli_path": cli_path or "",
                "kagglehub_available": kagglehub_available,
                "credentials_resolved": True,
                "credential_source": "standard_config",
                "username_hint": "",
                "standard_config_found": True,
                "standard_config_path": str(standard_config),
                "warning": str(exc),
            }
        return {
            "cli_found": cli_path is not None,
            "cli_path": cli_path or "",
            "kagglehub_available": kagglehub_available,
            "credentials_resolved": False,
            "credential_source": "invalid",
            "standard_config_found": False,
            "standard_config_path": "",
            "error": str(exc),
        }


def configure_kaggle_credentials() -> dict[str, str]:
    credentials = _load_kaggle_credentials()
    if credentials:
        os.environ.update(credentials)
    return credentials


def _resolve_kagglehub():
    try:
        import kagglehub  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The `kagglehub` package is not installed. Install project dependencies again to enable KaggleHub fetches."
        ) from exc
    return kagglehub


def _kaggle_command(cli_path: str, entry: KaggleDatasetEntry, output_dir: Path, unzip: bool, force: bool) -> list[str]:
    if entry.source_type == "competition":
        command = [cli_path, "competitions", "download", "-c", entry.kaggle_slug, "-p", str(output_dir)]
    else:
        command = [cli_path, "datasets", "download", "-d", entry.kaggle_slug, "-p", str(output_dir)]
    if unzip:
        command.append("--unzip")
    if force:
        command.append("--force")
    return command


def _kaggle_dataset_download_command(cli_path: str, slug: str, output_dir: Path, unzip: bool, force: bool) -> list[str]:
    command = [cli_path, "datasets", "download", "-d", slug, "-p", str(output_dir)]
    if unzip:
        command.append("--unzip")
    if force:
        command.append("--force")
    return command


def _run_kaggle_command(command: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True, env=env)


def _raise_kaggle_error(command: list[str], stderr: str) -> None:
    raise RuntimeError(
        f"Kaggle command failed. Command: {' '.join(command)}. Output: {stderr}"
    )


def download_dataset(
    dataset_id: str,
    output_root: str | Path,
    unzip: bool = True,
    force: bool = False,
) -> Path:
    entry = get_dataset_entry(dataset_id)
    kaggle_env = build_kaggle_env()
    cli_path = resolve_kaggle_cli()
    if cli_path is None:
        raise RuntimeError(
            "The `kaggle` CLI is not installed or could not be located."
        )

    output_dir = Path(output_root) / entry.id
    output_dir.mkdir(parents=True, exist_ok=True)
    command = _kaggle_command(cli_path, entry, output_dir, unzip=unzip, force=force)
    result = _run_kaggle_command(command, kaggle_env)
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        if "401" in stderr and entry.source_type == "competition":
            stderr = (
                f"{stderr} Accept the competition rules at "
                f"https://www.kaggle.com/competitions/{entry.kaggle_slug} before downloading files."
            )
        _raise_kaggle_error(command, f"Kaggle download failed for {dataset_id}. {stderr}")
    return output_dir


def _parse_csv_output(text: str) -> list[dict[str, str]]:
    if not text.strip():
        return []
    reader = csv.DictReader(text.splitlines())
    return [dict(row) for row in reader]


def search_kaggle_datasets(
    search: str | None = None,
    tags: Iterable[str] | None = None,
    sort_by: str = "votes",
    page: int = 1,
    min_usability_rating: float = 0.9,
) -> list[KaggleSearchResult]:
    cli_path = resolve_kaggle_cli()
    if cli_path is None:
        raise RuntimeError("The `kaggle` CLI is not installed or could not be located.")
    kaggle_env = build_kaggle_env()

    command = [cli_path, "datasets", "list", "--sort-by", sort_by, "-p", str(page), "-v"]
    if search:
        command.extend(["-s", search])
    tag_list = list(tags or [])
    if tag_list:
        command.extend(["--tags", ",".join(tag_list)])

    result = _run_kaggle_command(command, kaggle_env)
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        _raise_kaggle_error(command, stderr)

    rows = _parse_csv_output(result.stdout)
    output: list[KaggleSearchResult] = []
    for row in rows:
        usability = float(row.get("usabilityRating", 0) or 0)
        if usability < min_usability_rating:
            continue
        output.append(
            KaggleSearchResult(
                slug=row["ref"],
                title=row["title"],
                size_bytes=int(row.get("size", 0) or 0),
                last_updated=row.get("lastUpdated", ""),
                download_count=int(row.get("downloadCount", 0) or 0),
                vote_count=int(row.get("voteCount", 0) or 0),
                usability_rating=usability,
            )
        )
    return output


def _guess_train_file(output_dir: Path) -> str | None:
    candidates: list[Path] = []
    for pattern in ("*.csv", "*.parquet", "*.jsonl", "*.tsv"):
        candidates.extend(sorted(output_dir.rglob(pattern)))
    for preferred in ["train.csv", "training.csv", "train.parquet", "training.parquet"]:
        for path in candidates:
            if path.name.lower() == preferred:
                return path.name
    if len(candidates) == 1:
        return candidates[0].name
    return None


def _copy_downloaded_tree(source_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in source_dir.rglob("*"):
        relative = path.relative_to(source_dir)
        target = output_dir / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _download_dataset_via_kagglehub(slug: str, output_dir: Path, force: bool) -> Path:
    configure_kaggle_credentials()
    kagglehub = _resolve_kagglehub()
    downloaded_path = Path(kagglehub.dataset_download(slug, force_download=force))
    if downloaded_path.resolve() != output_dir.resolve():
        _copy_downloaded_tree(downloaded_path, output_dir)
    return output_dir


def download_kaggle_slug(
    slug: str,
    output_root: str | Path = "data/raw",
    dataset_id: str | None = None,
    unzip: bool = True,
    force: bool = False,
    backend: Literal["auto", "hub", "cli"] = "auto",
    title: str | None = None,
    task_type: str | None = None,
    target_column: str | None = None,
    notes: str = "",
) -> Path:
    local_id = dataset_id or sanitize_dataset_id(slug)
    output_dir = Path(output_root) / local_id
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_backend = backend
    if resolved_backend == "auto":
        resolved_backend = "hub" if unzip else "cli"

    if resolved_backend == "hub":
        if not unzip:
            raise ValueError("KaggleHub dataset downloads always materialize extracted files. Use backend='cli' with unzip=False.")
        _download_dataset_via_kagglehub(slug, output_dir, force=force)
    else:
        cli_path = resolve_kaggle_cli()
        if cli_path is None:
            raise RuntimeError("The `kaggle` CLI is not installed or could not be located.")
        kaggle_env = build_kaggle_env()
        command = _kaggle_dataset_download_command(cli_path, slug, output_dir, unzip=unzip, force=force)
        result = _run_kaggle_command(command, kaggle_env)
        if result.returncode != 0:
            stderr = result.stderr.strip() or result.stdout.strip()
            _raise_kaggle_error(command, stderr)

    manifest = DatasetManifest(
        id=local_id,
        title=title or slug,
        provider="kaggle",
        source_type="dataset",
        external_ref=slug,
        source_url=f"https://www.kaggle.com/datasets/{slug}",
        task_type=task_type,
        target_column=target_column,
        train_file=_guess_train_file(output_dir),
        notes=notes,
    )
    write_manifest(output_dir, manifest)
    return output_dir


def ingest_kaggle_dataset(
    slug: str,
    output_root: str | Path = "data/raw",
    processed_root: str | Path = "data/processed",
    dataset_id: str | None = None,
    unzip: bool = True,
    force: bool = False,
    backend: Literal["auto", "hub", "cli"] = "auto",
    seed: int = 42,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    max_rows: int | None = None,
    drop_identifier_columns: bool = True,
    title: str | None = None,
    task_type: str | None = None,
    target_column: str | None = None,
    train_file: str | None = None,
    notes: str = "",
    feature_engineering: bool = True,
) -> KagglePreparedDataset:
    from tabula.data.prep import prepare_dataset

    raw_dir = download_kaggle_slug(
        slug,
        output_root=output_root,
        dataset_id=dataset_id,
        unzip=unzip,
        force=force,
        backend=backend,
        title=title,
        task_type=task_type,
        target_column=target_column,
        notes=notes,
    )
    prepared = prepare_dataset(
        dataset_id or sanitize_dataset_id(slug),
        raw_root=output_root,
        processed_root=processed_root,
        seed=seed,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        max_rows=max_rows,
        drop_identifier_columns=drop_identifier_columns,
        task_type=task_type,
        target_column=target_column,
        train_file=train_file,
        title=title,
        notes=notes or None,
        feature_engineering=feature_engineering,
    )
    return KagglePreparedDataset(
        dataset_id=prepared.dataset_id,
        raw_dir=str(raw_dir),
        processed_dir=prepared.processed_dir,
        config_path=prepared.config_path,
        target_column=prepared.target_column,
        train_rows=prepared.train_rows,
        val_rows=prepared.val_rows,
        test_rows=prepared.test_rows,
    )


def summarize_csv(csv_path: str | Path) -> CsvSummary:
    path = Path(csv_path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        columns = next(reader)
        row_count = sum(1 for _ in reader)
    return CsvSummary(path=path, columns=columns, row_count=row_count)


def discover_csvs(root: str | Path) -> list[CsvSummary]:
    base = Path(root)
    summaries: list[CsvSummary] = []
    for csv_path in sorted(base.rglob("*.csv")):
        try:
            summaries.append(summarize_csv(csv_path))
        except UnicodeDecodeError:
            continue
    return summaries
