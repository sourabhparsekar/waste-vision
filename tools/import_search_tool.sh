#!/bin/bash

# Common requirements file
REQUIREMENTS="requirements.txt"

TOOL_APP_NAME="groq_search"

orchestrate tools remove -n

orchestrate connections remove -a "$TOOL_APP_NAME"

orchestrate connections add -a "$TOOL_APP_NAME"

orchestrate connections configure -a "$TOOL_APP_NAME" --env draft -t team -k bearer

orchestrate connections configure -a "$TOOL_APP_NAME" --env live -t team -k bearer

# Import each tool
echo "Importing $tool_file..."
orchestrate tools import -k python \
    -f "search_tool.py" \
    -r "$REQUIREMENTS" \
    -a "$TOOL_APP_NAME"

echo "All tools imported successfully."
