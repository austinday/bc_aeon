"""Self-improvement substrate for Aeon.

Turns one-off self-editing into a measurable, recursive loop:

    diagnose -> propose -> evaluate (isolated, scored) -> select (ratchet) -> integrate -> record

The pieces here provide the missing primitive that makes the loop *recursive*
rather than a random walk: a **fitness signal**. ``benchmark`` defines scored
capability tasks, ``evaluate`` runs them against a candidate in an isolated git
worktree, ``scorer`` aggregates and compares to the champion baseline, and
``ledger`` durably records every experiment so the agent does not re-try what it
already knows fails.
"""

from . import ledger, benchmark, scorer  # noqa: F401
