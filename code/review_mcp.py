"""Stdio tools used by the Codex and Claude CLI review sessions."""
import argparse
import contextlib
import sys

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

import tools


class CalibrationQuery(BaseModel):
    query: str
    n: int = Field(default=4, ge=1)
    low_score: float = -1.0
    high_score: float = 11.0


def make_server(paper_path: str, stage: str, no_cal: bool, mode: str) -> FastMCP:
    # Grant access to the paper itself, not every file beside it.
    tools.ALLOWED_PATHS[:] = [paper_path]
    if stage == "merger" and not no_cal:
        tools.allow_path(tools.CALIBRATION_REVIEW_DIR)
    server = FastMCP("ard_review", log_level="WARNING")
    read_only = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)
    server.tool(annotations=read_only)(tools.read_file)
    server.tool(annotations=read_only)(tools.grep_file)
    draft_recorded = False
    search_calls = 0

    if stage == "merger":
        @server.tool(annotations=read_only)
        def draft_review(draft: str) -> str:
            """Record the post-filtering draft once, before calibration or final writing."""
            nonlocal draft_recorded
            if draft_recorded:
                raise ValueError("The draft has already been recorded.")
            if not draft.strip():
                raise ValueError("The draft must not be empty.")
            draft_recorded = True
            return "draft recorded"

        if not no_cal:
            @server.tool(annotations=read_only)
            def calibration_search(queries: list[CalibrationQuery]) -> str:
                """Retrieve human-reviewed anchors in score bands; at most three batches.

                Call draft_review first. Each query returns n paths, average human
                scores and previews, with exclusive low_score/high_score bounds.
                """
                nonlocal search_calls
                if not draft_recorded:
                    raise ValueError("Call draft_review before calibration_search.")
                if search_calls >= 3:
                    raise ValueError("At most three calibration_search calls are allowed.")
                if not queries:
                    raise ValueError("queries must not be empty.")
                sections = []
                # Downloads/index diagnostics must never enter the JSON-RPC stream.
                with contextlib.redirect_stdout(sys.stderr):
                    for i, q in enumerate(queries, 1):
                        body = tools.search_file_impl(q.query, q.n, mode, q.low_score, q.high_score)
                        sections.append(f"### Query {i}: {q.query!r}\n{body}")
                search_calls += 1
                return "\n\n".join(sections)

    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", required=True)
    parser.add_argument("--stage", choices=["critic", "merger"], required=True)
    parser.add_argument("--no_cal", action="store_true")
    parser.add_argument("--calibration_mode", choices=["bm25", "vector"], default="bm25")
    args = parser.parse_args()
    make_server(args.paper, args.stage, args.no_cal, args.calibration_mode).run()
