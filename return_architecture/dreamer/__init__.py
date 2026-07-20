"""The autodreamer — the nightly consolidation pass over the memory graph.

Ported from the battle-tested silas_dream pipeline (collect -> metabolize ->
validate -> apply -> summarize), rebuilt against the node model:

- collect   reads items, derived nodes, conversation, and letters (read-only)
- metabolize asks the cheap-tier model to propose candidate items WITH citations
- validate  the deterministic boundary layer: distrusts the model entirely
- apply     writes surviving candidates into the graph as PROPOSED nodes/edges
            (auto-allowed types land open; nothing else enters recall unreviewed)
- summarize writes the human-readable morning summary + run artifacts

The dreamer proposes; the human decides. It physically cannot write outside
its run folder and the graph's proposal states.
"""

from .run import DEFAULT_CONFIG, run_night_pass

__all__ = ["run_night_pass", "DEFAULT_CONFIG"]
