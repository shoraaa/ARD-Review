"""Offline CLI double: exercises the real stdio MCP server before emitting results."""
import asyncio
import json
import os
from pathlib import Path
import sys
import tomllib

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    argv = sys.argv[1:]
    prompt = sys.stdin.read()
    codex = argv[0] == "exec"
    if codex:
        overrides = [argv[i + 1] for i, arg in enumerate(argv) if arg == "-c"]
        config = tomllib.loads("\n".join(overrides))
        server = config["mcp_servers"]["ard_review"]
        assert config["approval_policy"] == "never"
        assert argv[argv.index("--sandbox") + 1] == "read-only"
    else:
        config = json.loads(Path(argv[argv.index("--mcp-config") + 1]).read_text())
        server = config["mcpServers"]["ard_review"]
        assert argv[argv.index("--tools") + 1] == ""
        assert "--strict-mcp-config" in argv
    if os.environ.get("EXPECTED_MODEL"):
        assert argv[argv.index("--model") + 1] == os.environ["EXPECTED_MODEL"]
    args = server["args"]
    paper = args[args.index("--paper") + 1]
    stage = args[args.index("--stage") + 1]
    assert "{{" not in prompt, "Unexpanded prompt placeholder"
    assert "A synthetic paper" not in prompt, "Paper was included inline"
    env = {**os.environ, **server["env"]}
    async with stdio_client(StdioServerParameters(command=server["command"], args=args, env=env)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            names = {tool.name for tool in (await session.list_tools()).tools}
            result = await session.call_tool("read_file", {"abs_path": paper, "start_line": 1, "end_line": 2})
            assert not result.isError, result
            assert "synthetic" in result.content[0].text.lower(), result
            if stage == "critic":
                assert names == {"read_file", "grep_file"}
                assert "Read the paper progressively" in prompt
                review = "Critic checked the paper. Missing baseline comparison."
            else:
                assert "Critic checked the paper" in prompt
                assert "draft_review" in names
                result = await session.call_tool("draft_review", {"draft": "Checked draft"})
                assert not result.isError, result
                if "--no_cal" in args:
                    assert "calibration_search" not in names
                else:
                    result = await session.call_tool("calibration_search", {"queries": [{"query": "synthetic", "low_score": 4, "high_score": 8}]})
                    assert not result.isError, result
                    assert "avg_score: 6.00" in result.content[0].text, result
                review = "Final checked review.\n<score>7.5</score><decision>Accept</decision>"
                if os.environ.get("FAKE_BAD_TAGS"):
                    review = "Final checked review.\n<score>invalid</score>"
    if codex:
        Path(argv[argv.index("--output-last-message") + 1]).write_text(review)
        print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 100, "output_tokens": 50}}))
    else:
        print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": review,
                          "usage": {"input_tokens": 100, "output_tokens": 50}, "num_turns": 3}))


asyncio.run(main())
