import os
import json
import time
import fcntl
from pathlib import Path
from aeon.tools.base import BaseTool
from aeon.core.logger import get_logger

logger = get_logger()

# =============================================================================
# Task-scoped shared blackboard
# =============================================================================
# Lives at the WORKSPACE ROOT (cwd), NOT under any per-process instance_id.
#
# Why not instance_id: every sub-agent process builds its own Worker with its
# own random instance_id (Worker.__init__ -> uuid4). Siblings therefore do not
# share an instance_id, so it is unusable as a coordination key. The workspace
# directory IS the natural task scope: SpawnSubAgent symlinks each sub-agent's
# workspace to the main cwd, and the wrapper chdir's into it, so os.getcwd()
# resolves to the SAME physical directory for the primary agent and every
# sub-agent. A single file there is genuinely shared by all of them.
#
# Concurrency: append-only writes guarded by an advisory flock (the same
# fcntl pattern used throughout this codebase). Reads parse line-by-line and
# skip any malformed line, so a read racing a partial write degrades to
# "missed the latest entry" rather than an error. This is a passive board:
# agents read it lazily when they next run; nothing requires them to be live
# at the same instant (which matters because GPU contention serializes them).

MAX_FINDING_LEN = 4000
MAX_READ_ENTRIES = 50


def _blackboard_path() -> Path:
    base = Path(os.getcwd()) / "aeon_output"
    base.mkdir(parents=True, exist_ok=True)
    return base / "blackboard.jsonl"


class BlackboardPost(BaseTool):
    def __init__(self, worker=None, llm_client=None):
        super().__init__(
            name="blackboard_post",
            description=(
                "Posts a finding to the shared task blackboard, visible to the primary agent "
                "and ALL sub-agents working on this task. Use it to record intermediate results "
                "(a working approach, a confirmed fact, a produced artifact path, a dead end) so "
                "parallel agents can read them and avoid redoing the same work. Append-only; "
                "persists for the whole task.\n"
                "Schema:\n"
                "  topic (str, required): short tag for the finding (e.g. 'dataset_path', 'auth', 'dead_ends').\n"
                "  finding (str, required): the information to share. Be concise and factual.\n"
                "Example: {\"tool_name\": \"blackboard_post\", \"parameters\": {\"topic\": \"dataset_path\", \"finding\": \"Cleaned CSV written to data/clean/train.csv (12k rows).\"}}"
            )
        )
        self.worker = worker
        self.llm_client = llm_client

    def execute(self, topic: str = None, finding: str = None) -> str:
        if not topic or not finding:
            return "Error: both 'topic' and 'finding' are required."

        author = getattr(getattr(self, "worker", None), "instance_id", "unknown")
        finding = str(finding)
        if len(finding) > MAX_FINDING_LEN:
            finding = finding[:MAX_FINDING_LEN] + " ...[truncated]"

        entry = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "author": author,
            "topic": str(topic),
            "finding": finding,
        }

        try:
            path = _blackboard_path()
            with open(path, "a", encoding="utf-8") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(entry) + "\n")
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
            return f"Posted to the shared blackboard under topic '{topic}'."
        except Exception as e:
            logger.error(f"blackboard_post failed: {e}")
            return f"Error posting to blackboard: {e}"


class BlackboardRead(BaseTool):
    def __init__(self, worker=None, llm_client=None):
        super().__init__(
            name="blackboard_read",
            description=(
                "Reads findings posted to the shared task blackboard by you and other agents. "
                "Call this BEFORE starting a self-contained chunk of work to check whether a "
                "parallel agent has already produced the result or already hit the same dead end. "
                "Returns the most recent entries, optionally filtered by topic.\n"
                "Schema:\n"
                "  topic (str, optional): if given, only entries with this exact tag are returned. Omit to read everything.\n"
                "Example: {\"tool_name\": \"blackboard_read\", \"parameters\": {\"topic\": \"dataset_path\"}}"
            )
        )
        self.worker = worker
        self.llm_client = llm_client

    def execute(self, topic: str = None) -> str:
        path = _blackboard_path()
        if not path.exists():
            return "The shared blackboard is empty. No findings have been posted yet."

        try:
            with open(path, "r", encoding="utf-8") as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                try:
                    lines = f.readlines()
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
        except Exception as e:
            logger.error(f"blackboard_read failed: {e}")
            return f"Error reading blackboard: {e}"

        entries = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if topic and obj.get("topic") != topic:
                continue
            entries.append(obj)

        if not entries:
            if topic:
                # The board has content but nothing under this exact tag. Topic
                # match is exact, so surface the available topics to spare the
                # agent from guessing (e.g. 'auth' vs 'authentication').
                all_topics = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        t = json.loads(line).get("topic")
                    except json.JSONDecodeError:
                        continue
                    if t and t not in all_topics:
                        all_topics.append(t)
                if all_topics:
                    return (f"No blackboard findings under topic '{topic}'. "
                            f"Available topics: {', '.join(all_topics)}. "
                            f"Retry with one of these, or omit 'topic' to read everything.")
            return "No blackboard findings yet."

        total = len(entries)
        shown = entries[-MAX_READ_ENTRIES:]
        noun = "entry" if total == 1 else "entries"
        header = f"Shared blackboard ({total} matching {noun}"
        if total > len(shown):
            header += f", showing latest {len(shown)}"
        header += "):"

        out = [header]
        for e in shown:
            out.append(
                f"- [{e.get('ts', '?')}] ({e.get('author', '?')}) "
                f"[{e.get('topic', '?')}] {e.get('finding', '')}"
            )
        return "\n".join(out)
