"""Built-in tools that live inside the runtime.

External tools come in as MCP servers in a later step. Built-in tools are
tightly coupled to the runtime (e.g. no_response, which the loop needs to
recognise to do the right thing) or to the project's core rituals.
"""

from return_architecture.tools.base import Tool, ToolContext, ToolResult
from return_architecture.tools.no_response import NoResponseTool
from return_architecture.tools.send_telegram import SendToHumanTelegramTool
from return_architecture.tools.artifact_tools import (
    ArtifactDeleteReactionTool,
    ArtifactShareMoreTool,
)
from return_architecture.tools.tag_item import TagItemTool
from return_architecture.tools.item_tools import (
    ListItemsTool,
    EditItemTool,
    RemoveItemTool,
)
from return_architecture.tools.write_letter import WriteLetterTool
from return_architecture.tools.private_writings import (
    WritePrivatelyTool,
    ListPrivateWritingsTool,
    ReadPrivateWritingTool,
)
from return_architecture.tools.schedule_self import (
    ScheduleSelfTool,
    ListMySchedulesTool,
    CancelMyScheduleTool,
)
from return_architecture.tools.inbox import (
    ListInboxTool,
    ReadInboxLetterTool,
)
from return_architecture.tools.presence import (
    UpdateNowTool,
    SitWithThisTool,
)
from return_architecture.tools.generate_image import GenerateImageTool
from return_architecture.tools.set_temperature import SetTemperatureTool

BUILTIN_TOOLS: dict[str, Tool] = {
    "no_response": NoResponseTool(),
    "send_to_human_telegram": SendToHumanTelegramTool(),
    "artifact_delete_reaction": ArtifactDeleteReactionTool(),
    "artifact_share_more": ArtifactShareMoreTool(),
    "tag_item": TagItemTool(),
    "list_items": ListItemsTool(),
    "edit_item": EditItemTool(),
    "remove_item": RemoveItemTool(),
    "write_letter": WriteLetterTool(),
    "write_privately": WritePrivatelyTool(),
    "list_private_writings": ListPrivateWritingsTool(),
    "read_private_writing": ReadPrivateWritingTool(),
    "schedule_self": ScheduleSelfTool(),
    "list_my_schedules": ListMySchedulesTool(),
    "cancel_my_schedule": CancelMyScheduleTool(),
    "list_inbox": ListInboxTool(),
    "read_inbox_letter": ReadInboxLetterTool(),
    "update_now": UpdateNowTool(),
    "sit_with_this": SitWithThisTool(),
    "generate_image": GenerateImageTool(),
    "set_temperature": SetTemperatureTool(),
}

__all__ = ["Tool", "ToolContext", "ToolResult", "BUILTIN_TOOLS"]
