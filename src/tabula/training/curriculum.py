"""Curriculum training: persistent queue and ledger for multi-dataset background training.

The **queue** (``curriculum_queue.json``) is an ordered list of dataset entries each
with a status (``pending``, ``in_progress``, ``done``, ``failed``).  The worker picks
the next pending entry, trains one cycle, then updates the queue.

The **ledger** (``curriculum_ledger.jsonl``) is an append-only log; every completed or
interrupted session writes a single JSON line.  This gives a full audit trail and lets
us resume correctly after a crash.

Typical directory layout::

    artifacts/
        curriculum_queue.json
        curriculum_ledger.jsonl
        curriculum_hf_adult_census_income/
            best.pt
            latest.pt
            run_manifest.json
            train_state.json
            progress.jsonl
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
import uuid


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUEUE_FILENAME = "curriculum_queue.json"
LEDGER_FILENAME = "curriculum_ledger.jsonl"

EntryStatus = str  # "pending" | "in_progress" | "done" | "failed" | "paused"
ExitReason = str   # "max_steps" | "early_stop" | "interrupted" | "error" | "max_total_steps"


# ---------------------------------------------------------------------------
# Queue entry
# ---------------------------------------------------------------------------

@dataclass
class CurriculumEntry:
    """One dataset slot in the curriculum training queue."""

    # --- identity ---
    dataset_id: str
    """Unique local identifier, e.g. ``hf_adult_census_income``."""
    prepared_dir: str
    """Path to the prepared dataset directory (must contain ``schema.json``,
    ``val.csv``, and ``train_config.json``)."""
    hf_repo_id: str
    """Hugging Face repo id to stream from, e.g. ``scikit-learn/adult-census-income``."""

    # --- optional HF params ---
    hf_config_name: str | None = None
    hf_split: str = "train"

    # --- scheduling ---
    status: EntryStatus = "pending"
    priority: int = 100
    """Lower priority value → scheduled sooner when multiple entries are pending."""

    # --- step budgets ---
    steps_per_cycle: int = 2000
    """Optimizer steps per worker cycle (incremental progress per wake-up)."""
    max_total_steps: int = 20000
    """Hard upper bound on total lifetime steps for this entry.  Once reached the
    entry is marked ``done``."""

    # --- accumulated progress (updated in-place by the worker) ---
    total_steps: int = 0
    total_rows_seen: int = 0
    best_val_loss: float | None = None

    # --- metadata ---
    experiment_name: str | None = None
    """Defaults to ``curriculum_{dataset_id}`` at training time."""
    last_session_at: str | None = None
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    def effective_experiment_name(self) -> str:
        return self.experiment_name or f"curriculum_{self.dataset_id}"

    def remaining_steps(self) -> int:
        return max(0, self.max_total_steps - self.total_steps)

    def next_cycle_target(self) -> int:
        """Absolute step count that should be the ``max_steps`` cap for the next session."""
        return min(self.total_steps + self.steps_per_cycle, self.max_total_steps)


# ---------------------------------------------------------------------------
# Ledger session record
# ---------------------------------------------------------------------------

@dataclass
class LedgerSession:
    """Immutable record written to the ledger at the end of each training session."""

    session_id: str
    dataset_id: str
    experiment_name: str
    started_at: str
    ended_at: str
    steps_trained: int
    cumulative_steps: int
    rows_seen: int
    cumulative_rows: int
    exit_reason: ExitReason
    best_val_loss: float | None = None
    checkpoint_path: str | None = None
    trunk_source: str | None = None
    """Checkpoint used to warm-start the transformer trunk (``None`` if training from scratch)."""
    error: str | None = None

    @staticmethod
    def new_session_id() -> str:
        return uuid.uuid4().hex[:12]

    @staticmethod
    def utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# CurriculumQueue
# ---------------------------------------------------------------------------

class CurriculumQueue:
    """Mutable ordered list of :class:`CurriculumEntry` objects, backed by a JSON file."""

    VERSION = 1

    def __init__(self, entries: list[CurriculumEntry] | None = None) -> None:
        self.entries: list[CurriculumEntry] = entries or []

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "CurriculumQueue":
        p = Path(path)
        if not p.exists():
            return cls()
        raw = json.loads(p.read_text(encoding="utf-8"))
        entries = [CurriculumEntry(**e) for e in raw.get("entries", [])]
        return cls(entries)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "version": self.VERSION,
            "entries": [asdict(e) for e in self.entries],
        }
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def next_pending(self) -> CurriculumEntry | None:
        """Return the highest-priority pending entry (lowest ``priority`` value,
        stable order for ties)."""
        candidates = [e for e in self.entries if e.status == "pending"]
        if not candidates:
            return None
        return min(candidates, key=lambda e: e.priority)

    def get(self, dataset_id: str) -> CurriculumEntry | None:
        for e in self.entries:
            if e.dataset_id == dataset_id:
                return e
        return None

    def summary(self) -> list[dict[str, Any]]:
        return [
            {
                "dataset_id": e.dataset_id,
                "status": e.status,
                "priority": e.priority,
                "total_steps": e.total_steps,
                "max_total_steps": e.max_total_steps,
                "best_val_loss": e.best_val_loss,
                "last_session_at": e.last_session_at,
            }
            for e in self.entries
        ]

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def add(self, entry: CurriculumEntry) -> None:
        """Add an entry; raise if ``dataset_id`` already exists."""
        if any(e.dataset_id == entry.dataset_id for e in self.entries):
            raise ValueError(f"Entry with dataset_id={entry.dataset_id!r} already exists.")
        self.entries.append(entry)

    def _get_required(self, dataset_id: str) -> CurriculumEntry:
        entry = self.get(dataset_id)
        if entry is None:
            raise KeyError(f"No queue entry found for dataset_id={dataset_id!r}.")
        return entry

    def mark_in_progress(self, dataset_id: str) -> None:
        self._get_required(dataset_id).status = "in_progress"

    def mark_done(self, dataset_id: str) -> None:
        self._get_required(dataset_id).status = "done"

    def mark_failed(self, dataset_id: str) -> None:
        self._get_required(dataset_id).status = "failed"

    def reset_to_pending(self, dataset_id: str) -> None:
        """Re-queue a failed or done entry so it will be picked up again."""
        self._get_required(dataset_id).status = "pending"

    def update_progress(
        self,
        dataset_id: str,
        *,
        steps_delta: int,
        rows_delta: int,
        best_val_loss: float | None,
        session_timestamp: str,
    ) -> None:
        entry = self._get_required(dataset_id)
        entry.total_steps += steps_delta
        entry.total_rows_seen += rows_delta
        if best_val_loss is not None:
            if entry.best_val_loss is None or best_val_loss < entry.best_val_loss:
                entry.best_val_loss = best_val_loss
        entry.last_session_at = session_timestamp
        # Transition: if step budget exhausted mark done, otherwise re-queue
        if entry.total_steps >= entry.max_total_steps:
            entry.status = "done"
        else:
            entry.status = "pending"


# ---------------------------------------------------------------------------
# CurriculumLedger  (append-only JSONL)
# ---------------------------------------------------------------------------

class CurriculumLedger:
    """Append-only JSONL log of training sessions across all datasets."""

    @staticmethod
    def append(session: LedgerSession, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(asdict(session)) + "\n")

    @staticmethod
    def load(path: str | Path) -> list[LedgerSession]:
        p = Path(path)
        if not p.exists():
            return []
        sessions: list[LedgerSession] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                sessions.append(LedgerSession(**json.loads(line)))
        return sessions

    @staticmethod
    def latest_best_checkpoint(path: str | Path) -> str | None:
        """Return the most recently recorded checkpoint path from the ledger."""
        sessions = CurriculumLedger.load(path)
        for session in reversed(sessions):
            if session.checkpoint_path and Path(session.checkpoint_path).exists():
                return session.checkpoint_path
        return None


# ---------------------------------------------------------------------------
# Helper: default file paths given an artifacts root
# ---------------------------------------------------------------------------

def queue_path(artifacts_root: str | Path = "artifacts") -> Path:
    return Path(artifacts_root) / QUEUE_FILENAME


def ledger_path(artifacts_root: str | Path = "artifacts") -> Path:
    return Path(artifacts_root) / LEDGER_FILENAME
