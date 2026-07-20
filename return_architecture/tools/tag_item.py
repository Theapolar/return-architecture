"""Agent-side tagging tool.

Lets the agent record something it notices as a note, important moment,
open question, or commitment. Stored alongside hashtag-tagged items from
the human in the per-agent items.db.
"""

from __future__ import annotations

from typing import Any

from return_architecture import items
from return_architecture.tools.base import Tool, ToolContext, ToolResult


class TagItemTool(Tool):
    name = "tag_item"
    description = (
        "Record something as a tagged item in this agent's persistent items "
        "store. Use this when you notice something worth tracking — a note "
        "you want to keep, a moment that matters, an open question, or a "
        "commitment the human (or you) has made. To place it in one of the "
        "human's home containers, set kind='note' and tag='idea' (Ideas panel) "
        "or tag='canvas' (pinned to the Canvas)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": list(items.KINDS),
                "description": "What kind of item this is.",
            },
            "body": {
                "type": "string",
                "description": "The content of the item — a clear, self-contained sentence.",
            },
            "tag": {
                "type": "string",
                "description": "Optional container tag, e.g. 'idea' or 'canvas'.",
            },
        },
        "required": ["kind", "body"],
    }

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        kind = (args.get("kind") or "").strip().lower()
        body = (args.get("body") or "").strip()
        tag = (args.get("tag") or "").strip().lower()
        if kind not in items.KINDS:
            return ToolResult(
                content=f"Error: kind must be one of {list(items.KINDS)}, got '{kind}'."
            )
        if not body:
            return ToolResult(content="Error: body is empty.")
        try:
            item_id = items.add_item(
                context.slug,
                kind=kind,
                body=body,
                source="agent",
                metadata={"tag": tag} if tag else None,
            )
        except ValueError as e:
            return ToolResult(content=f"Error: {e}")
        where = f" in {tag}" if tag else ""
        return ToolResult(content=f"Tagged as {kind}{where} (id {item_id}).")
