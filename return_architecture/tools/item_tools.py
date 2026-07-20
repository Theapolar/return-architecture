"""Agent-side item management: list, edit, remove.

Companions to tag_item (which creates). Together these give the agent hands
on the human's items store — to see what's there, refine an item's wording,
or remove one. The corresponding human side is the app's item UI.

Removal is deliberately framed as a careful act: the items are largely the
human's, and the principle is to curate and propose, not to quietly rewrite
someone's world.
"""

from __future__ import annotations

from typing import Any

from return_architecture import items
from return_architecture.tools.base import Tool, ToolContext, ToolResult


class ListItemsTool(Tool):
    name = "list_items"
    description = (
        "List the human's open items (notes, ideas, important moments, questions, "
        "commitments) with their ids — most recent first. Optionally filter by "
        "kind, or by container tag (tag='idea' for the Ideas panel, tag='canvas' "
        "for the Canvas). Use this to see what's there before editing or removing."
    )
    parameters = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "description": "Optional: note | important | question | commitment."},
            "tag": {"type": "string", "description": "Optional container tag, e.g. 'idea' or 'canvas'."},
        },
        "required": [],
    }

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        kind = (args.get("kind") or "").strip().lower() or None
        tag = (args.get("tag") or "").strip().lower() or None
        rows = items.list_items(context.slug, kind=kind, status="open", limit=100)
        if tag:
            rows = [r for r in rows if (r.metadata or {}).get("tag") == tag]
        if not rows:
            return ToolResult(content="(no matching items)")
        lines = []
        for r in rows:
            t = (r.metadata or {}).get("tag")
            label = f"{r.kind}/{t}" if t else r.kind
            lines.append(f"{r.id}. [{label}] {r.body}")
        return ToolResult(content="\n".join(lines))


class EditItemTool(Tool):
    name = "edit_item"
    description = (
        "Edit the text of an existing item by id (find ids with list_items). "
        "Use to refine or correct an item's wording. The human can edit too."
    )
    parameters = {
        "type": "object",
        "properties": {
            "id": {"type": "integer", "description": "The item's id."},
            "body": {"type": "string", "description": "The new text."},
        },
        "required": ["id", "body"],
    }

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        try:
            iid = int(args.get("id"))
        except (TypeError, ValueError):
            return ToolResult(content="Error: id must be an integer.")
        body = (args.get("body") or "").strip()
        if not body:
            return ToolResult(content="Error: body is empty.")
        ok = items.update_item(context.slug, iid, body=body)
        return ToolResult(content=f"Edited item {iid}." if ok else f"No open item {iid}.")


class RemoveItemTool(Tool):
    name = "remove_item"
    description = (
        "Remove an item by id (find ids with list_items) — resolves it so it no "
        "longer shows. Use sparingly and only when clearly right: these are mostly "
        "the human's, so prefer to suggest removal in conversation unless they've "
        "asked you to tidy. The human can also remove items themselves."
    )
    parameters = {
        "type": "object",
        "properties": {"id": {"type": "integer", "description": "The item's id."}},
        "required": ["id"],
    }

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        try:
            iid = int(args.get("id"))
        except (TypeError, ValueError):
            return ToolResult(content="Error: id must be an integer.")
        ok = items.resolve_item(context.slug, iid)
        return ToolResult(content=f"Removed item {iid}." if ok else f"No open item {iid}.")
