#!/usr/bin/env python3
"""
Improved OpenAI server with better function calling support for SWE-agent.
This is a patch that improves function calling compatibility.
"""

import json
import re
import uuid
from typing import List, Optional

from server import *  # Import everything from the original server

# Override the parse_tool_calls_from_response function
def parse_tool_calls_from_response_improved(response_text: str) -> List[ToolCall]:
    """Improved parser for tool calls from model response."""
    tool_calls = []
    
    # First, check if the response contains actual function call JSON
    if '"tool_calls"' in response_text or '"function"' in response_text:
        # The original parsing logic already handles this well
        tool_calls = parse_tool_calls_from_response(response_text)
        if tool_calls:
            return tool_calls
    
    # For SWE-agent, we need to detect when the model is trying to call functions
    # even if it doesn't format them perfectly
    
    # Look for common SWE-agent tools in the response
    swe_agent_tools = ['bash', 'edit_file', 'read_file', 'write_file', 'submit']
    
    # Pattern 1: Detect markdown code blocks with tool names
    code_block_pattern = r'```(\w+)?\n(.*?)\n```'
    matches = re.findall(code_block_pattern, response_text, re.DOTALL)
    
    for lang, content in matches:
        # Check if this looks like a bash command
        if lang == 'bash' or (not lang and content.strip() and not content.startswith('def ') and not content.startswith('class ')):
            tool_call = ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                type="function",
                function={
                    "name": "bash",
                    "arguments": json.dumps({"command": content.strip()})
                }
            )
            tool_calls.append(tool_call)
            
    # Pattern 2: Direct tool mentions (e.g., "run bash command: ls -la")
    for tool in swe_agent_tools:
        pattern = rf'{tool}[:\s]+([^\n]+)'
        matches = re.findall(pattern, response_text, re.IGNORECASE)
        for match in matches:
            if tool == 'bash':
                tool_call = ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    type="function",
                    function={
                        "name": "bash",
                        "arguments": json.dumps({"command": match.strip()})
                    }
                )
                tool_calls.append(tool_call)
                
    return tool_calls

# Override the original function
parse_tool_calls_from_response = parse_tool_calls_from_response_improved

# Create an improved format_tools_for_prompt function that's more explicit
def format_tools_for_prompt_improved(tools: List[ToolDefinition]) -> str:
    """Format tools information for the prompt with clearer instructions."""
    if not tools:
        return ""
    
    tools_text = "You have access to the following tools:\n\n"
    
    for tool in tools:
        func = tool.function
        tools_text += f"Tool: {func.name}\n"
        tools_text += f"Description: {func.description}\n"
        
        if func.parameters and func.parameters.get("properties"):
            tools_text += "Parameters:\n"
            for param_name, param_info in func.parameters["properties"].items():
                required = param_name in func.parameters.get("required", [])
                req_text = " (required)" if required else " (optional)"
                tools_text += f"  - {param_name}{req_text}: {param_info.get('description', 'No description')}\n"
        tools_text += "\n"
    
    tools_text += """To use a tool, format your response as follows:

For code execution:
```bash
your command here
```

Or as explicit JSON:
{
  "tool_calls": [
    {
      "id": "call_1",
      "type": "function",
      "function": {
        "name": "tool_name",
        "arguments": "{\"param1\": \"value1\"}"
      }
    }
  ]
}

Important: When using tools, make sure your response includes the tool call in one of these formats."""
    
    return tools_text

# Override the original function
format_tools_for_prompt = format_tools_for_prompt_improved

if __name__ == "__main__":
    # This improved version can be run the same way as the original
    print("This is a patch file. Import it after importing the original server.py")
    print("Example usage:")
    print("  from server import *")
    print("  from server_improved import *")
    print("  # Now the improved functions will be used")