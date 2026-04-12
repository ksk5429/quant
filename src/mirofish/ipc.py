"""File-based IPC protocol for distributed Fish agents.

Inspired by MiroFish (666ghj) ipc_commands/ipc_responses pattern,
adapted for prediction market swarm intelligence.

Architecture:
  Master (this process)
    ├── writes task files → shared_state/tasks/{task_id}.json
    ├── monitors responses ← shared_state/responses/{task_id}_{persona}.json
    └── tracks state → shared_state/state.json

  Fish (separate Claude Code instances or CLI processes)
    ├── polls tasks ← shared_state/tasks/
    ├── writes responses → shared_state/responses/
    └── heartbeats → shared_state/heartbeats/{persona}.json

Protocol:
  1. Master creates task file with market data and assigned personas
  2. Fish agents pick up tasks, analyze, write response files
  3. Master collects responses, aggregates, produces final prediction
  4. Master writes result to shared_state/results/

Task lifecycle: PENDING → CLAIMED → COMPLETED / TIMEOUT / ERROR
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger


class TaskStatus(str, Enum):
    PENDING = "pending"
    CLAIMED = "claimed"
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class FishTask:
    """A task assigned to one or more Fish agents."""
    task_id: str
    market_id: str
    question: str
    description: str
    outcomes: list[str]
    personas: list[str]          # which Fish should analyze this
    research_briefing: str = ""  # from Researcher Fish (injected into prompt)
    round_number: int = 1        # 1 = independent, 2 = Delphi update
    peer_estimates: dict[str, float] = field(default_factory=dict)  # round 2 only
    status: str = "pending"
    created_at: str = ""
    timeout_s: int = 600         # 10 minutes default

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class FishResponse:
    """A response from a Fish agent to a task."""
    task_id: str
    market_id: str
    persona: str
    probability: float
    confidence: float
    reasoning: str = ""
    steps: list[str] = field(default_factory=list)
    round_number: int = 1
    model: str = ""
    elapsed_s: float = 0.0
    completed_at: str = ""

    def __post_init__(self):
        if not self.completed_at:
            self.completed_at = datetime.now().isoformat()


@dataclass
class SwarmState:
    """Global state of the distributed swarm."""
    active_tasks: int = 0
    completed_tasks: int = 0
    total_markets: int = 0
    running_brier: float = 0.0
    total_responses: int = 0
    last_updated: str = ""
    fish_online: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.last_updated:
            self.last_updated = datetime.now().isoformat()


class IPCMaster:
    """Master controller for distributed Fish swarm.

    Manages the task lifecycle: create tasks, monitor responses,
    handle timeouts, aggregate results.
    """

    def __init__(self, base_dir: str | Path = "shared_state") -> None:
        self.base = Path(base_dir)
        self.tasks_dir = self.base / "tasks"
        self.responses_dir = self.base / "responses"
        self.results_dir = self.base / "results"
        self.heartbeats_dir = self.base / "heartbeats"
        self.state_file = self.base / "state.json"

        # Create all directories
        for d in [self.tasks_dir, self.responses_dir, self.results_dir, self.heartbeats_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self._state = SwarmState()
        self._save_state()

    def create_task(
        self,
        market_id: str,
        question: str,
        description: str = "",
        outcomes: list[str] | None = None,
        personas: list[str] | None = None,
        research_briefing: str = "",
        round_number: int = 1,
        peer_estimates: dict[str, float] | None = None,
    ) -> FishTask:
        """Create a new task for Fish agents to process."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_id = f"{ts}_{market_id}"

        task = FishTask(
            task_id=task_id,
            market_id=market_id,
            question=question,
            description=description[:2000],
            outcomes=outcomes or ["Yes", "No"],
            personas=personas or [],
            research_briefing=research_briefing,
            round_number=round_number,
            peer_estimates=peer_estimates or {},
        )

        task_path = self.tasks_dir / f"{task_id}.json"
        task_path.write_text(
            json.dumps(asdict(task), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        self._state.active_tasks += 1
        self._save_state()

        logger.info(f"Task created: {task_id} for {len(task.personas)} Fish")
        return task

    def collect_responses(
        self, task_id: str, timeout_s: int = 600,
    ) -> list[FishResponse]:
        """Collect all responses for a given task.

        Polls the responses directory for matching files.
        Does NOT block — returns whatever responses exist now.
        """
        responses = []
        pattern = f"{task_id}_*.json"

        for resp_path in self.responses_dir.glob(pattern):
            try:
                data = json.loads(resp_path.read_text(encoding="utf-8"))
                resp = FishResponse(**{
                    k: v for k, v in data.items()
                    if k in FishResponse.__dataclass_fields__
                })
                responses.append(resp)
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse response {resp_path.name}: {e}")

        return responses

    def wait_for_responses(
        self, task_id: str, expected: int, timeout_s: int = 600, poll_interval: float = 5.0,
    ) -> list[FishResponse]:
        """Block until all expected responses arrive or timeout.

        Args:
            task_id: The task to wait for.
            expected: Number of expected responses.
            timeout_s: Max wait time in seconds.
            poll_interval: Seconds between checks.

        Returns:
            List of collected FishResponse objects.
        """
        deadline = time.monotonic() + timeout_s
        last_count = 0

        while time.monotonic() < deadline:
            responses = self.collect_responses(task_id)
            if len(responses) >= expected:
                logger.info(f"Task {task_id}: all {len(responses)} responses collected")
                return responses

            if len(responses) > last_count:
                logger.info(
                    f"Task {task_id}: {len(responses)}/{expected} responses..."
                )
                last_count = len(responses)

            time.sleep(poll_interval)

        # Timeout — return whatever we have
        responses = self.collect_responses(task_id)
        logger.warning(
            f"Task {task_id}: timeout after {timeout_s}s. "
            f"Got {len(responses)}/{expected} responses."
        )
        return responses

    def save_result(self, task_id: str, result: dict[str, Any]) -> Path:
        """Save aggregated result for a task."""
        result_path = self.results_dir / f"{task_id}_result.json"
        result_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._state.completed_tasks += 1
        self._state.active_tasks = max(0, self._state.active_tasks - 1)
        self._save_state()
        return result_path

    def get_online_fish(self, stale_threshold_s: float = 120.0) -> list[str]:
        """Get list of Fish personas that have recent heartbeats."""
        online = []
        now = time.time()

        for hb_path in self.heartbeats_dir.glob("*.json"):
            try:
                data = json.loads(hb_path.read_text(encoding="utf-8"))
                last_ts = data.get("timestamp", 0)
                if now - last_ts < stale_threshold_s:
                    online.append(data.get("persona", hb_path.stem))
            except (json.JSONDecodeError, ValueError):
                continue

        return online

    def cleanup_old_tasks(self, max_age_hours: int = 24) -> int:
        """Remove task/response files older than max_age_hours."""
        cutoff = time.time() - max_age_hours * 3600
        removed = 0

        for directory in [self.tasks_dir, self.responses_dir]:
            for fp in directory.glob("*.json"):
                if fp.stat().st_mtime < cutoff:
                    fp.unlink()
                    removed += 1

        if removed:
            logger.info(f"Cleaned up {removed} old IPC files")
        return removed

    def _save_state(self) -> None:
        self._state.last_updated = datetime.now().isoformat()
        self.state_file.write_text(
            json.dumps(asdict(self._state), indent=2),
            encoding="utf-8",
        )


class IPCFishWorker:
    """Worker that a Fish agent uses to pick up and respond to tasks.

    Used by each Fish (in a separate Claude Code instance or process)
    to poll for tasks, process them, and write responses.
    """

    def __init__(self, persona: str, base_dir: str | Path = "shared_state") -> None:
        self.persona = persona
        self.base = Path(base_dir)
        self.tasks_dir = self.base / "tasks"
        self.responses_dir = self.base / "responses"
        self.heartbeats_dir = self.base / "heartbeats"

        self.responses_dir.mkdir(parents=True, exist_ok=True)
        self.heartbeats_dir.mkdir(parents=True, exist_ok=True)

    def heartbeat(self) -> None:
        """Write a heartbeat file so the master knows this Fish is alive."""
        hb_path = self.heartbeats_dir / f"{self.persona}.json"
        hb_path.write_text(json.dumps({
            "persona": self.persona,
            "timestamp": time.time(),
            "status": "online",
        }), encoding="utf-8")

    def get_pending_tasks(self) -> list[FishTask]:
        """Get tasks that include this persona and haven't been responded to."""
        pending = []
        for task_path in sorted(self.tasks_dir.glob("*.json")):
            try:
                data = json.loads(task_path.read_text(encoding="utf-8"))
                task = FishTask(**{
                    k: v for k, v in data.items()
                    if k in FishTask.__dataclass_fields__
                })

                # Check if this persona is assigned
                if self.persona not in task.personas:
                    continue

                # Check if already responded
                resp_path = self.responses_dir / f"{task.task_id}_{self.persona}.json"
                if resp_path.exists():
                    continue

                pending.append(task)
            except (json.JSONDecodeError, TypeError):
                continue

        return pending

    def submit_response(self, response: FishResponse) -> Path:
        """Write a response file for the master to collect."""
        resp_path = self.responses_dir / f"{response.task_id}_{response.persona}.json"
        resp_path.write_text(
            json.dumps(asdict(response), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Fish [{self.persona}] submitted response for {response.task_id}")
        return resp_path
