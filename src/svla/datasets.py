from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from svla.soarm import DEFAULT_DATASET_DIR, find_lerobot_dataset_path, lerobot_dataset_path


def _safe_dataset_path(repo_id: str) -> Path:
    dataset_path = lerobot_dataset_path(repo_id).resolve()
    dataset_root = DEFAULT_DATASET_DIR.resolve()
    if dataset_root != dataset_path and dataset_root not in dataset_path.parents:
        raise RuntimeError(f"Refusing to write outside project datasets: {dataset_path}")
    return dataset_path


def _copy_dataset(source: Path, target: Path, overwrite: bool) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {source}")
    if not (source / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Source does not look like a LeRobot dataset: {source}")

    if target.exists():
        if not overwrite:
            raise FileExistsError(f"Target already exists. Use --overwrite: {target}")
        shutil.rmtree(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)


def import_dataset(source: str, repo_id: str, overwrite: bool = False) -> int:
    source_path = Path(source).expanduser().resolve()
    target_path = _safe_dataset_path(repo_id)
    _copy_dataset(source_path, target_path, overwrite=overwrite)
    print(f"Imported dataset: {source_path}")
    print(f"repo_id: {repo_id}")
    print(f"path: {target_path}")
    return 0


def import_dataset_command(args: argparse.Namespace) -> int:
    return import_dataset(source=args.source, repo_id=args.repo_id, overwrite=args.overwrite)


def _load_variants(path: str | None) -> dict[str, list[str]]:
    if path is None:
        return {}

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    variants: dict[str, list[str]] = {}
    for source, values in data.items():
        if isinstance(values, str):
            variants[str(source)] = [values]
        elif isinstance(values, list):
            variants[str(source)] = [str(value) for value in values if str(value).strip()]
        else:
            raise ValueError(f"Variants for {source!r} must be a string or list of strings")
    return variants


def _default_variants(task: str) -> list[str]:
    task = task.strip()
    if not task:
        return []

    lower = task[:1].lower() + task[1:]
    return [
        task,
        f"Please {lower}.",
        f"Use the robot arm to {lower}.",
        f"Complete the task: {lower}.",
    ]


TASK_INDEX_SENTINEL = "__task_index__"


def _task_text_column(tasks: pd.DataFrame) -> str:
    preferred_columns = ["task", "tasks", "instruction", "language_instruction"]
    for column in preferred_columns:
        if column in tasks.columns:
            return column

    if tasks.index.name in preferred_columns or (
        tasks.index.dtype == "object" and tasks.index.notna().any()
    ):
        return TASK_INDEX_SENTINEL

    text_columns = [
        column
        for column in tasks.columns
        if pd.api.types.is_object_dtype(tasks[column])
        or pd.api.types.is_string_dtype(tasks[column])
    ]
    if not text_columns:
        raise ValueError("No text column found in meta/tasks.parquet")
    return text_columns[0]


def _task_texts(tasks: pd.DataFrame, text_column: str) -> list[str]:
    if text_column == TASK_INDEX_SENTINEL:
        return [str(task) for task in tasks.index.tolist()]
    return [str(task) for task in tasks[text_column].tolist()]


def _set_task_texts(tasks: pd.DataFrame, text_column: str, values: list[str]) -> pd.DataFrame:
    updated_tasks = tasks.copy()
    if text_column == TASK_INDEX_SENTINEL:
        updated_tasks.index = pd.Index(values, name=tasks.index.name)
        return updated_tasks

    updated_tasks[text_column] = values
    return updated_tasks


def _task_index_column(tasks: pd.DataFrame) -> str | None:
    for column in ("task_index", "index"):
        if column in tasks.columns:
            return column
    return None


def _expanded_tasks(tasks: pd.DataFrame, variants: dict[str, list[str]]) -> pd.DataFrame:
    text_column = _task_text_column(tasks)
    task_index_column = _task_index_column(tasks)
    rows = []

    for _, row in tasks.iterrows():
        original_task = str(row.name if text_column == TASK_INDEX_SENTINEL else row[text_column])
        choices = variants.get(original_task) or _default_variants(original_task)
        if not choices:
            choices = [original_task]

        for choice in choices:
            expanded_row = row.copy()
            if text_column != TASK_INDEX_SENTINEL:
                expanded_row[text_column] = choice
            rows.append((choice, expanded_row))

    expanded = pd.DataFrame([row for _, row in rows])
    if task_index_column is not None:
        expanded[task_index_column] = range(len(expanded))

    if text_column == TASK_INDEX_SENTINEL:
        expanded.index = pd.Index([task for task, _ in rows], name=tasks.index.name)
    else:
        expanded.index = pd.RangeIndex(len(expanded))

    return expanded


def _variant_rows(
    tasks: pd.DataFrame,
    variants: dict[str, list[str]],
) -> list[tuple[str, int, str]]:
    text_column = _task_text_column(tasks)
    task_index_column = _task_index_column(tasks)
    rows = []

    for row_index, row in tasks.iterrows():
        original_task = str(row_index if text_column == TASK_INDEX_SENTINEL else row[text_column])
        original_task_index = int(row[task_index_column]) if task_index_column else len(rows)
        choices = variants.get(original_task) or _default_variants(original_task) or [original_task]
        for choice in choices:
            rows.append((original_task, original_task_index, choice))

    return rows


def _episode_task_text(episode: pd.Series, tasks_by_index: dict[int, str]) -> str:
    episode_tasks = episode.get("tasks")
    if isinstance(episode_tasks, list) and episode_tasks:
        return str(episode_tasks[0])

    task_index = episode.get("task_index")
    if task_index is not None:
        return tasks_by_index[int(task_index)]

    return next(iter(tasks_by_index.values()))


def _write_single_parquet(df: pd.DataFrame, directory: Path) -> Path:
    if directory.exists():
        for parquet_path in directory.glob("*/*.parquet"):
            parquet_path.unlink()

    output_path = directory / "chunk-000" / "file-000.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return output_path


def _update_episode_stats(
    row: pd.Series,
    episode_index: int,
    from_index: int,
    to_index: int,
    task_index: int,
) -> None:
    length = int(row["length"])
    index_mean = (from_index + to_index - 1) / 2

    row["stats/episode_index/min"] = episode_index
    row["stats/episode_index/max"] = episode_index
    row["stats/episode_index/mean"] = float(episode_index)
    row["stats/episode_index/std"] = 0.0
    row["stats/episode_index/count"] = [length]
    row["stats/episode_index/q01"] = [float(episode_index)]
    row["stats/episode_index/q10"] = [float(episode_index)]
    row["stats/episode_index/q50"] = [float(episode_index)]
    row["stats/episode_index/q90"] = [float(episode_index)]
    row["stats/episode_index/q99"] = [float(episode_index)]

    row["stats/index/min"] = from_index
    row["stats/index/max"] = to_index - 1
    row["stats/index/mean"] = float(index_mean)
    row["stats/index/count"] = [length]
    row["stats/index/q01"] = [float(from_index)]
    row["stats/index/q10"] = [float(from_index)]
    row["stats/index/q50"] = [float(index_mean)]
    row["stats/index/q90"] = [float(to_index - 1)]
    row["stats/index/q99"] = [float(to_index - 1)]

    row["stats/task_index/min"] = task_index
    row["stats/task_index/max"] = task_index
    row["stats/task_index/mean"] = float(task_index)
    row["stats/task_index/std"] = 0.0
    row["stats/task_index/count"] = [length]
    row["stats/task_index/q01"] = [float(task_index)]
    row["stats/task_index/q10"] = [float(task_index)]
    row["stats/task_index/q50"] = [float(task_index)]
    row["stats/task_index/q90"] = [float(task_index)]
    row["stats/task_index/q99"] = [float(task_index)]


def _select_variant(original_task: str, variants: dict[str, list[str]], row_index: int) -> str:
    explicit_choices = variants.get(original_task)
    choices = explicit_choices or _default_variants(original_task)
    if not choices:
        return original_task

    variant_index = row_index % len(choices) if explicit_choices else (row_index + 1) % len(choices)
    return choices[variant_index]


def augment_language(
    source_repo_id: str,
    target_repo_id: str,
    variants_file: str | None = None,
    overwrite: bool = False,
) -> int:
    source_path = lerobot_dataset_path(source_repo_id).resolve()
    target_path = _safe_dataset_path(target_repo_id)
    _copy_dataset(source_path, target_path, overwrite=overwrite)

    tasks_path = target_path / "meta" / "tasks.parquet"
    tasks = pd.read_parquet(tasks_path)
    text_column = _task_text_column(tasks)
    variants = _load_variants(variants_file)

    updated_texts = [
        _select_variant(str(task), variants, row_index)
        for row_index, task in enumerate(_task_texts(tasks, text_column))
    ]
    updated_tasks = _set_task_texts(tasks, text_column, updated_texts)
    updated_tasks.to_parquet(tasks_path)

    manifest = {
        "source_repo_id": source_repo_id,
        "target_repo_id": target_repo_id,
        "tasks_path": str(tasks_path),
        "text_column": text_column,
        "variants_file": variants_file,
    }
    manifest_path = target_path / "meta" / "language_augmentation.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Created language-augmented dataset: {target_repo_id}")
    print(f"path: {target_path}")
    print(f"text_column: {text_column}")
    print(f"manifest: {manifest_path}")
    return 0


def expand_language(
    source_repo_id: str,
    target_repo_id: str,
    variants_file: str | None = None,
    overwrite: bool = False,
) -> int:
    source_path = find_lerobot_dataset_path(source_repo_id).resolve()
    target_path = _safe_dataset_path(target_repo_id)
    _copy_dataset(source_path, target_path, overwrite=overwrite)

    tasks_path = target_path / "meta" / "tasks.parquet"
    tasks = pd.read_parquet(tasks_path)
    variants = _load_variants(variants_file)
    expanded_tasks = _expanded_tasks(tasks, variants)
    expanded_tasks.to_parquet(tasks_path)

    manifest = {
        "source_repo_id": source_repo_id,
        "target_repo_id": target_repo_id,
        "tasks_path": str(tasks_path),
        "variants_file": variants_file,
        "source_total_tasks": len(tasks),
        "target_total_tasks": len(expanded_tasks),
        "note": "Only meta/tasks.parquet is expanded. Episode task_index values are not remapped.",
    }
    manifest_path = target_path / "meta" / "language_expansion.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Created language-expanded dataset: {target_repo_id}")
    print(f"path: {target_path}")
    print(f"tasks: {len(tasks)} -> {len(expanded_tasks)}")
    print("Note: episode task_index values still point to their original task index.")
    print(f"manifest: {manifest_path}")
    return 0


def expand_language_episodes(
    source_repo_id: str,
    target_repo_id: str,
    variants_file: str | None = None,
    overwrite: bool = False,
) -> int:
    source_path = find_lerobot_dataset_path(source_repo_id).resolve()
    target_path = _safe_dataset_path(target_repo_id)
    _copy_dataset(source_path, target_path, overwrite=overwrite)

    tasks_path = target_path / "meta" / "tasks.parquet"
    tasks = pd.read_parquet(tasks_path)
    variants = _load_variants(variants_file)
    variant_rows = _variant_rows(tasks, variants)
    expanded_tasks = _expanded_tasks(tasks, variants)
    expanded_tasks.to_parquet(tasks_path)

    tasks_by_index = {
        original_task_index: original_task for original_task, original_task_index, _ in variant_rows
    }
    variants_by_original_task = {}
    for variant_task_index, (original_task, _, variant_task) in enumerate(variant_rows):
        variants_by_original_task.setdefault(original_task, []).append(
            (variant_task_index, variant_task)
        )

    episodes_dir = target_path / "meta" / "episodes"
    data_dir = target_path / "data"
    episodes = pd.concat(
        [pd.read_parquet(path) for path in sorted(episodes_dir.glob("*/*.parquet"))],
        ignore_index=True,
    )
    data = pd.concat(
        [pd.read_parquet(path) for path in sorted(data_dir.glob("*/*.parquet"))],
        ignore_index=True,
    )

    expanded_episode_rows = []
    expanded_data_frames = []
    next_episode_index = 0
    next_frame_index = 0

    for _, episode in episodes.iterrows():
        source_episode_index = int(episode["episode_index"])
        source_rows = data[data["episode_index"] == source_episode_index].copy()
        source_task = _episode_task_text(episode, tasks_by_index)
        choices = variants_by_original_task[source_task]

        for task_index, task_text in choices:
            episode_rows = source_rows.copy()
            episode_rows["episode_index"] = next_episode_index
            episode_rows["index"] = range(next_frame_index, next_frame_index + len(episode_rows))
            episode_rows["task_index"] = task_index

            updated_episode = episode.copy()
            updated_episode["episode_index"] = next_episode_index
            updated_episode["tasks"] = [task_text]
            updated_episode["dataset_from_index"] = next_frame_index
            updated_episode["dataset_to_index"] = next_frame_index + len(episode_rows)
            updated_episode["data/chunk_index"] = 0
            updated_episode["data/file_index"] = 0
            updated_episode["meta/episodes/chunk_index"] = 0
            updated_episode["meta/episodes/file_index"] = 0
            _update_episode_stats(
                updated_episode,
                episode_index=next_episode_index,
                from_index=next_frame_index,
                to_index=next_frame_index + len(episode_rows),
                task_index=task_index,
            )

            expanded_data_frames.append(episode_rows)
            expanded_episode_rows.append(updated_episode)
            next_frame_index += len(episode_rows)
            next_episode_index += 1

    expanded_data = pd.concat(expanded_data_frames, ignore_index=True)
    expanded_episodes = pd.DataFrame(expanded_episode_rows)
    _write_single_parquet(expanded_data, data_dir)
    _write_single_parquet(expanded_episodes, episodes_dir)

    info_path = target_path / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["total_tasks"] = len(expanded_tasks)
    info["total_episodes"] = len(expanded_episodes)
    info["total_frames"] = len(expanded_data)
    info["splits"] = {"train": f"0:{len(expanded_episodes)}"}
    info_path.write_text(json.dumps(info, indent=4, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "source_repo_id": source_repo_id,
        "target_repo_id": target_repo_id,
        "variants_file": variants_file,
        "source_total_tasks": len(tasks),
        "target_total_tasks": len(expanded_tasks),
        "source_total_episodes": len(episodes),
        "target_total_episodes": len(expanded_episodes),
        "source_total_frames": len(data),
        "target_total_frames": len(expanded_data),
        "video_note": (
            "Video files are shared; duplicated episodes reuse original video timestamp ranges."
        ),
    }
    manifest_path = target_path / "meta" / "language_episode_expansion.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Created language-expanded episode dataset: {target_repo_id}")
    print(f"path: {target_path}")
    print(f"tasks: {len(tasks)} -> {len(expanded_tasks)}")
    print(f"episodes: {len(episodes)} -> {len(expanded_episodes)}")
    print(f"frames: {len(data)} -> {len(expanded_data)}")
    print(f"manifest: {manifest_path}")
    return 0


def expand_language_episodes_command(args: argparse.Namespace) -> int:
    return expand_language_episodes(
        source_repo_id=args.source_repo_id,
        target_repo_id=args.target_repo_id,
        variants_file=args.variants_file,
        overwrite=args.overwrite,
    )


def dataset_tasks(repo_id: str, limit: int | None = None) -> int:
    dataset_path = find_lerobot_dataset_path(repo_id)
    tasks_path = dataset_path / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        print(f"Dataset tasks not found: {tasks_path}")
        return 1

    tasks = pd.read_parquet(tasks_path)
    text_column = _task_text_column(tasks)
    task_index_column = _task_index_column(tasks)
    task_texts = _task_texts(tasks, text_column)

    print(f"repo_id: {repo_id}")
    print(f"path: {dataset_path}")
    print(f"tasks: {len(tasks)}")
    print(f"text_column: {text_column}")
    print()

    max_rows = len(task_texts) if limit is None else min(limit, len(task_texts))
    for row_index, task in enumerate(task_texts[:max_rows]):
        if task_index_column is not None:
            task_index = tasks.iloc[row_index][task_index_column]
        else:
            task_index = row_index
        print(f"{task_index}: {task}")

    if max_rows < len(task_texts):
        print(f"... {len(task_texts) - max_rows} more")
    return 0


def dataset_tasks_command(args: argparse.Namespace) -> int:
    return dataset_tasks(repo_id=args.repo_id, limit=args.limit)


def augment_language_command(args: argparse.Namespace) -> int:
    return augment_language(
        source_repo_id=args.source_repo_id,
        target_repo_id=args.target_repo_id,
        variants_file=args.variants_file,
        overwrite=args.overwrite,
    )


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    import_parser = subparsers.add_parser("import-dataset")
    import_parser.add_argument("--source", required=True)
    import_parser.add_argument("--repo-id", required=True)
    import_parser.add_argument("--overwrite", action="store_true")
    import_parser.set_defaults(func=import_dataset_command)

    augment_parser = subparsers.add_parser("augment-language")
    augment_parser.add_argument("--source-repo-id", required=True)
    augment_parser.add_argument("--target-repo-id", required=True)
    augment_parser.add_argument("--variants-file")
    augment_parser.add_argument("--overwrite", action="store_true")
    augment_parser.set_defaults(func=augment_language_command)

    expand_parser = subparsers.add_parser("expand-language-episodes")
    expand_parser.add_argument("--source-repo-id", required=True)
    expand_parser.add_argument("--target-repo-id", required=True)
    expand_parser.add_argument("--variants-file")
    expand_parser.add_argument("--overwrite", action="store_true")
    expand_parser.set_defaults(func=expand_language_episodes_command)

    tasks_parser = subparsers.add_parser("dataset-tasks")
    tasks_parser.add_argument("--repo-id", required=True)
    tasks_parser.add_argument("--limit", type=int)
    tasks_parser.set_defaults(func=dataset_tasks_command)
