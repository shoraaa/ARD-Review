"""
ARD-Review: two-stage agentic paper reviewer.

  Phase 1  Harsh Critic — reads the paper from disk in chunks, reasons through it
           incrementally, and produces a critical review.
  Phase 2  Merger — cross-checks every weakness against the paper, retrieves
           human-reviewed anchor papers from the calibration corpus (iterative
           RAG, up to 3 batched queries), and writes the final review with
           <score> and <decision> tags.

Both stages support the Codex and Claude CLIs, the OpenAI Agents SDK, and the
Claude Agent SDK. CLI runs use the CLI's existing authentication.

Configuration uses CLI flags and environment variables — see README.md.
"""
import argparse
import asyncio
import csv
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import dotenv
dotenv.load_dotenv()

from paths import prompt_path, RESULTS_DIR
from tools import (
    API_BASE_URL,
    API_KEY,
    CALIBRATION_REVIEW_DIR,
    allow_path,
    embedding_config,
    grep_file,
    read_file,
    search_file_impl,
)

from cli_backend import cli_model, run_cli


@dataclass
class ReviewConfig:
    harsh_model: str
    merger_model: str
    effort: str | None = None
    calibration_mode: str = "vector"
    pdf_parser: str = "datalab"
    calibration_csv: Path | None = None


def api_client():
    from openai import AsyncOpenAI
    if not API_KEY:
        raise ValueError("The API backend requires REVIEW_API_KEY. Use --backend codex or --backend claude for CLI authentication.")
    return AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)


HUMAN_REVIEW_DIR = CALIBRATION_REVIEW_DIR

# ── Agent-level retry ────────────────────────────────────────────────
MAX_RETRIES = 5
RETRY_DELAY = 10


async def run_agent_with_retry(agent, prompt: str, max_turns: int = 30) -> tuple[str, object]:
    from agents import Runner
    agent_name = agent.name
    print(f"  [{agent_name}] starting ...")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = await Runner.run(agent, prompt, max_turns=max_turns)
            output = result.final_output
            if not output or not output.strip():
                if attempt < MAX_RETRIES:
                    print(f"  [{agent_name}] empty response (attempt {attempt}/{MAX_RETRIES}), retrying ...")
                    await asyncio.sleep(RETRY_DELAY + attempt * 5)
                    continue
                raise RuntimeError(f"[{agent_name}] empty response after {MAX_RETRIES} attempts")
            print(f"  [{agent_name}] done")
            return output, result.context_wrapper.usage
        except Exception as e:
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY * attempt
                print(f"  [{agent_name}] error (attempt {attempt}/{MAX_RETRIES}), waiting {wait}s ... {e}")
                await asyncio.sleep(wait)
            else:
                raise RuntimeError(f"[{agent_name}] {e}") from e
    raise RuntimeError(f"[{agent_name}] failed after {MAX_RETRIES} attempts")


# ── Prompt loading ───────────────────────────────────────────────────
with open(prompt_path("timeline.md"), "r") as f:
    timeline = f.read().replace("{{CURRENT_DATE}}", time.strftime("%Y-%m-%d"))

PAPER_ACCESS_FILE = "The paper path is provided in the user message. Use read_file to read the paper (it reads the whole file by default — do not pass start_line/end_line unless you specifically need a slice) and verify reviewer claims directly."
PAPER_ACCESS_CHUNKED = """The paper path is provided in the user message. The paper is NOT included inline — read it from disk in sequential chunks using read_file with start_line/end_line, and use grep_file to locate specific claims or sections.

Read the paper progressively, one chunk at a time. After each chunk, before reading the next one, pause and reason: think through what this part of the paper claims, whether the method/evidence/argument in it holds up, and note any concerns or strengths it surfaces. Build your assessment incrementally as you go — do not dump the whole paper into context and review it all at the end. Only move to the next chunk once you have thought through the current one. Choose reasonable chunk sizes (e.g. a section or a few hundred lines at a time)."""

with open(prompt_path("cal_with.md"), "r") as _f:
    CAL_INSTRUCTION_WITH = _f.read()

with open(prompt_path("cal_without.md"), "r") as _f:
    CAL_INSTRUCTION_WITHOUT = _f.read()


def load_prompts(path, paper_access: str, no_cal: bool):
    with open(prompt_path(path), "r") as f:
        raw_lines = f.readlines()
    kept_lines = []
    for lineno, line in enumerate(raw_lines, start=1):
        if line.lstrip().startswith("&&"):
            print(f"WARNING: ignoring commented line {lineno} in prompts/{path}: {line.rstrip()}")
            continue
        kept_lines.append(line)
    content = "".join(kept_lines)
    content = content.replace("{{PAPER_ACCESS_INSTRUCTION}}", paper_access)
    cal_instruction = CAL_INSTRUCTION_WITHOUT if no_cal else CAL_INSTRUCTION_WITH
    content = content.replace("{{CALIBRATION_INSTRUCTION}}", cal_instruction)
    return content + "\n\n" + timeline


# ── Agent definitions ────────────────────────────────────────────────
async def run_stage(config: ReviewConfig, model: str, stage: str, prompt: str,
                    paper_path: str, no_cal: bool):
    label = "Harsh Critic" if stage == "critic" else "Merger"
    prompt_file = "harsh_critic.md" if stage == "critic" else "merger.md"
    access = PAPER_ACCESS_CHUNKED if stage == "critic" else PAPER_ACCESS_FILE
    instructions = load_prompts(prompt_file, access, no_cal)
    if cli_model(model):
        text, usage = await run_cli(
            model=model, system_prompt=instructions, user_prompt=prompt,
            paper_path=paper_path, stage=stage, no_cal=no_cal,
            calibration_mode=config.calibration_mode, effort=config.effort,
        )
        return text, "cli", usage
    if model.startswith("claude_sdk:"):
        from claude_merger import run_harsh_claude_sdk, run_merger_claude_sdk
        model_id = model.removeprefix("claude_sdk:")
        paper_dir = str(Path(paper_path).parent)
        if stage == "critic":
            text, usage = await run_harsh_claude_sdk(model_id, prompt, paper_dir, instructions)
        else:
            text, usage = await run_merger_claude_sdk(
                model_id, prompt, paper_dir, no_cal=no_cal, calibration_mode=config.calibration_mode,
            )
        return text, "sdk", usage

    from agents import Agent, OpenAIResponsesModel, function_tool, set_tracing_disabled
    from agents.model_settings import ModelSettings
    set_tracing_disabled(True)
    if not config.effort:
        raise ValueError("The API backend requires REASONING_EFFORT or --reasoning_effort.")
    extra_body = {"effort": config.effort}
    if os.environ.get("OPENROUTER_PROVIDER"):
        extra_body["provider"] = {"only": [os.environ["OPENROUTER_PROVIDER"]]}
    agent_tools = [function_tool(read_file), function_tool(grep_file)]
    if stage == "merger":
        @function_tool
        def draft_review(draft: str) -> str:
            """Record the merger's post-filtering draft before calibration or final writing."""
            return "draft recorded"
        agent_tools.append(draft_review)
        if not no_cal:
            from review_mcp import CalibrationQuery

            @function_tool
            def calibration_search(queries: list[CalibrationQuery]) -> str:
                """Retrieve human-reviewed anchors in score bands. At most three batches.

                Each query supplies query, n, low_score and high_score (exclusive bounds).
                Call draft_review first, then bracket and narrow as instructed.
                """
                if not queries:
                    raise ValueError("queries must be a non-empty list.")
                return "\n\n".join(
                    f"### Query {i}: {q.query!r}\n" + search_file_impl(
                        q.query, q.n, config.calibration_mode, q.low_score, q.high_score,
                    ) for i, q in enumerate(queries, 1)
                )
            agent_tools.append(calibration_search)
    async with api_client() as client:
        agent = Agent(name=label, instructions=instructions,
                      model=OpenAIResponsesModel(model=model, openai_client=client),
                      tools=agent_tools, model_settings=ModelSettings(extra_body=extra_body))
        text, usage = await run_agent_with_retry(agent, prompt)
    return text, "api", usage


def parse_review_tags(review: str) -> tuple[float, str]:
    """Return sentinel values for missing, malformed or out-of-range tags."""
    score = -1.0
    matches = re.findall(r"<score>\s*(.*?)\s*</score>", review, re.IGNORECASE | re.DOTALL)
    if matches:
        try:
            value = float(matches[-1])
            if math.isfinite(value) and 0 <= value <= 10:
                score = value
        except ValueError:
            pass
    decisions = re.findall(r"<decision>\s*(Accept|Reject)\s*</decision>", review, re.IGNORECASE)
    return score, decisions[-1].title() if decisions else "N/A"


REVIEW_PROMPT = """Review the following paper thoroughly.

The paper may have been extracted from PDF by an automated parser. Treat formatting artifacts (broken equations, garbled tables, OCR errors) as parser issues, not paper flaws. If the supplied text omits the appendix or references, assume they may exist in the original submission and do not flag them as missing.

Paper path: {paper_path}. The paper is not included inline — read it from disk in chunks (read_file / grep_file), reasoning through each chunk before reading the next, following the paper-access protocol in your instructions."""


# ── Core pipeline ────────────────────────────────────────────────────
async def run_pipeline(paper_path: str, config: ReviewConfig, no_cal: bool = False) -> dict:
    paper_path_abs = str(Path(paper_path).resolve())
    allow_path(paper_path_abs)
    agent_usages, sdk_usages, cli_usages = {}, {}, {}

    def record(label, kind, usage):
        {"api": agent_usages, "sdk": sdk_usages, "cli": cli_usages}[kind][label] = usage

    print("  Phase 1: Running harsh critic ...")
    harsh_text, kind, usage = await run_stage(
        config, config.harsh_model, "critic", REVIEW_PROMPT.format(paper_path=paper_path_abs),
        paper_path_abs, no_cal=True,
    )
    record("Harsh Critic", kind, usage)
    labeled = [f"### Harsh Critic\n{harsh_text}"]

    print("  Phase 2: Merger ...")
    merger_prompt = (
        f"Paper path: {paper_path_abs} — use read_file / grep_file to verify the review.\n\n"
        f"Human reviews directory (for calibration): {HUMAN_REVIEW_DIR}\n\n"
        f"Here is the input review:\n\n{chr(10).join(labeled)}\n\n"
        "Now produce the final consolidated review following your instructions. "
        "Cross-check every weakness against the actual paper before including it. "
        "End with <score>NUMBER</score> and <decision>Accept|Reject</decision>."
    )
    merged_review, kind, usage = await run_stage(
        config, config.merger_model, "merger", merger_prompt, paper_path_abs, no_cal,
    )
    record("Merger", kind, usage)
    scorer_output, decision = parse_review_tags(merged_review)
    if scorer_output == -1 or decision == "N/A":
        extractor_model = os.environ.get("EXTRACTOR_MODEL")
        if not cli_model(config.merger_model) and extractor_model and API_KEY:
            print(f"  Re-extracting missing tags with {extractor_model}")
            async with api_client() as client:
                response = await client.chat.completions.create(
                    model=extractor_model,
                    messages=[
                        {"role": "system", "content": "Extract the final numeric score and accept/reject decision. Respond with <score>NUMBER</score><decision>Accept|Reject</decision>. Do not guess; use -1 or N/A if absent."},
                        {"role": "user", "content": merged_review},
                    ],
                    extra_body={"reasoning": {"enabled": False}},
                )
            extracted_score, extracted_decision = parse_review_tags(response.choices[0].message.content or "")
            if scorer_output == -1:
                scorer_output = extracted_score
            if decision == "N/A":
                decision = extracted_decision
        if scorer_output == -1 or decision == "N/A":
            print("  Review has missing/invalid score or decision tags; preserving the full review without guessing.")

    total_input = total_output = total_tokens = 0
    token_lines = []
    for agent_name, usage in agent_usages.items():
        if usage is None:
            token_lines.append(f"  {agent_name}: N/A (claude_sdk path)")
        else:
            cached = usage.input_tokens_details.cached_tokens
            reasoning = usage.output_tokens_details.reasoning_tokens
            token_lines.append(
                f"  {agent_name}: input={usage.input_tokens} (cached={cached}) "
                f"output={usage.output_tokens} (reasoning={reasoning}) "
                f"total={usage.total_tokens} requests={usage.requests}"
            )
            total_input += usage.input_tokens
            total_output += usage.output_tokens
            total_tokens += usage.total_tokens
    token_lines.append(f"  TOTAL: input={total_input} output={total_output} total={total_tokens}")

    sdk_lines = []
    sdk_total_cost = 0.0
    for sdk_name, su in sdk_usages.items():
        u = su["usage"]
        sdk_lines.append(f"  [{sdk_name}]")
        sdk_lines.append(f"    Model: {su['model']}")
        sdk_lines.append(f"    Session ID: {su['session_id']}")
        sdk_lines.append(f"    Cost (USD): {su['total_cost_usd']}")
        sdk_lines.append(f"    Turns: {su['num_turns']}")
        sdk_lines.append(f"    Duration: total={su['duration_ms']}ms api={su['duration_api_ms']}ms")
        sdk_lines.append(
            f"    Tokens: input={u['input_tokens']} output={u['output_tokens']} "
            f"cache_read={u['cache_read_input_tokens']} cache_creation={u['cache_creation_input_tokens']}"
        )
        rl = su["rate_limit"]
        if rl:
            util = rl["utilization"]
            util_str = f"{util*100:.1f}%" if util is not None else "n/a"
            sdk_lines.append(
                f"    Plan usage: type={rl['type']} util={util_str} "
                f"status={rl['status']} overage={rl['overage_status']}"
            )
        sdk_total_cost += su["total_cost_usd"]
    if sdk_lines:
        sdk_lines.append(f"  TOTAL Claude SDK cost (USD): {sdk_total_cost:.4f}")

    log_path = RESULTS_DIR / "pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as log_f:
        log_f.write(f"\n{'='*60}\n")
        log_f.write(f"Paper: {paper_path}\n")
        log_f.write(f"Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
        log_f.write(f"Models: critic={config.harsh_model}, merger={config.merger_model}\n")
        log_f.write(f"Calibration retrieval: {'disabled' if no_cal else config.calibration_mode}\n")
        if cli_usages:
            log_f.write("\n--- CLI Usage ---\n" + json.dumps(cli_usages, indent=2) + "\n")
        log_f.write("\n--- Token Usage ---\n" + "\n".join(token_lines) + "\n")
        if sdk_lines:
            log_f.write("\n--- Claude SDK Usage ---\n" + "\n".join(sdk_lines) + "\n")
        log_f.write(f"\n--- Merged Inputs ---\n\n{chr(10).join(labeled)}\n")
        log_f.write(f"\n--- Merged Review ---\n{merged_review}\n")
        log_f.write(f"\n--- Scorer Output ---\n{scorer_output}\n")
        log_f.write(f"\n--- Decision ---\n{decision}\n")

    return {"merged_review": merged_review, "scorer_output": scorer_output, "decision": decision, "sdk_usages": sdk_usages, "cli_usages": cli_usages}


# ── Score calibration against the shipped reference runs ─────────────
# results/claude.csv    — this pipeline on Claude Sonnet 4.6 (Claude Agent SDK)
# results/deepseek.csv  — this pipeline on DeepSeek V4 Flash (OpenAI Agents SDK)
# Both are 390-394 ICLR 2026 submissions with real reviewer scores + decisions.
# The calibration set is picked from the merger backend actually being run.
def calibrate(score: float, calibration_csv: Path, window: float = 0.5) -> dict:
    """Map a raw pipeline score onto the reference run: percentile, OLS-calibrated
    score against real reviewer averages, and empirical acceptance rate."""
    import numpy as np

    preds = []
    gts = []
    accepts = []
    with open(calibration_csv, "r") as f:
        for row in csv.DictReader(f):
            preds.append(float(row["pred_score"]))
            gts.append(float(row["gt_avg_score"]))
            accepts.append(row["gt_binary"] == "Accept")
    preds = np.array(preds)
    gts = np.array(gts)
    accepts = np.array(accepts)

    below = (preds < score).sum()
    equal = np.isclose(preds, score).sum()
    percentile = (below + 0.5 * equal) / len(preds) * 100

    slope, intercept = np.polyfit(preds, gts, 1)
    fitted = slope * preds + intercept
    r = np.corrcoef(preds, gts)[0, 1]

    exact_mask = np.isclose(preds, score)
    window_mask = np.abs(preds - score) <= window

    return {
        "csv": str(calibration_csv),
        "n": len(preds),
        "percentile": percentile,
        "calibrated_score": slope * score + intercept,
        "slope": slope,
        "intercept": intercept,
        "pearson_r": r,
        "r2": r ** 2,
        "residual_mae": float(np.abs(gts - fitted).mean()),
        "exact_rate": accepts[exact_mask].mean() if exact_mask.any() else float("nan"),
        "exact_n": int(exact_mask.sum()),
        "window": window,
        "window_rate": accepts[window_mask].mean() if window_mask.any() else float("nan"),
        "window_n": int(window_mask.sum()),
    }


# ── PDF -> markdown ──────────────────────────────────────────────────
async def pdf_to_markdown(pdf_path: Path, parser: str = "datalab") -> str:
    if parser == "local":
        import shutil
        from cli_backend import run_process
        executable = shutil.which("pdftotext")
        if not executable:
            raise RuntimeError("Local PDF conversion requires pdftotext (Poppler). Install poppler-utils, provide .md/.txt, or select --pdf_parser datalab.")
        text, _ = await run_process(
            [executable, "-layout", str(pdf_path.resolve()), "-"], "",
            cwd=str(pdf_path.resolve().parent), timeout=120,
        )
        if not text.strip():
            raise ValueError("PDF contains no extractable text; provide OCR text or select --pdf_parser datalab.")
        return text
    from datalab_sdk import AsyncDatalabClient, ConvertOptions
    options = ConvertOptions(
        output_format="markdown",
        mode="fast",
        paginate=True,
        page_range="0-9",
        token_efficient_markdown=True,
    )
    async with AsyncDatalabClient() as client:
        result = await client.convert(pdf_path, options=options)
    return result.markdown + "\n\n Rest of paper (reference and Appendix) is removed."


# ── Single paper ─────────────────────────────────────────────────────
async def run_single_paper(paper_path: str, config: ReviewConfig, no_cal: bool = False, skip_calibration: bool = False):
    print(f"Reviewing: {paper_path}")

    if Path(paper_path).suffix.lower() == ".pdf":
        md = await pdf_to_markdown(Path(paper_path), config.pdf_parser)
        md = re.sub(r"Published as a conference paper at ICLR \d{4}\s*\n?", "", md)
        suffix = ".extracted.md" if config.pdf_parser == "local" else ".md"
        md_path = Path(paper_path).with_suffix(suffix)
        md_path.write_text(md, encoding="utf-8")
        paper_path = str(md_path)
        print(f"Converted PDF to markdown: {paper_path}")

    result = await run_pipeline(paper_path, config, no_cal=no_cal)
    print(f"\n{'=' * 72}\nFINAL REVIEW\n{'=' * 72}\n{result['merged_review']}")
    score = result["scorer_output"]
    cal = None
    if score != -1:
        print(f"\nPredicted score: {score}")
        if not skip_calibration and config.calibration_csv is not None:
            cal = calibrate(score, config.calibration_csv)
            print(f"\n{'=' * 72}\nCalibration ({Path(cal['csv']).name}, n={cal['n']})\n{'=' * 72}")
            print(f"  Percentile of score={score}: {cal['percentile']:.1f}%")
            print(f"  Calibrated score (OLS vs real reviewer avg): {cal['calibrated_score']:.2f}")
            print(f"    fit: gt = {cal['slope']:.3f} * pred + {cal['intercept']:.3f}  "
                  f"(pearson r={cal['pearson_r']:.3f}, R2={cal['r2']:.3f}, residual MAE={cal['residual_mae']:.2f})")
            print(f"  Acceptance rate @ score={score}: {cal['exact_rate']:.2%} (n={cal['exact_n']})")
            print(f"  Acceptance rate @ score={score}+/-{cal['window']}: {cal['window_rate']:.2%} (n={cal['window_n']})")

    if result["cli_usages"]:
        print("\nCLI Usage:\n" + json.dumps(result["cli_usages"], indent=2))
    sdk_usages = result["sdk_usages"]
    if sdk_usages:
        print(f"\n{'=' * 72}\nClaude SDK Usage\n{'=' * 72}")
        total_cost = 0.0
        for name, su in sdk_usages.items():
            u = su["usage"]
            print(f"  [{name}]")
            print(f"    Model:         {su['model']}")
            print(f"    Session ID:    {su['session_id']}")
            print(f"    Cost (USD):    ${su['total_cost_usd']}")
            print(f"    Turns:         {su['num_turns']}")
            print(f"    Duration:      total={su['duration_ms']}ms api={su['duration_api_ms']}ms")
            print(f"    Input tokens:  {u['input_tokens']}")
            print(f"    Output tokens: {u['output_tokens']}")
            print(f"    Cache read:    {u['cache_read_input_tokens']}")
            print(f"    Cache create:  {u['cache_creation_input_tokens']}")
            rl = su["rate_limit"]
            if rl:
                util = rl["utilization"]
                util_str = f"{util*100:.1f}%" if util is not None else "n/a"
                print(f"    Plan usage:    type={rl['type']} util={util_str} status={rl['status']} overage={rl['overage_status']}")
            total_cost += su["total_cost_usd"]
        print(f"  TOTAL Claude SDK cost (USD): ${total_cost:.4f}")

    out_dir = RESULTS_DIR / "reviews"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (os.path.basename(paper_path).split(".")[0] + f"_review_{time.strftime('%Y_%m_%d_%H_%M_%S')}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Review of {paper_path}\n\n")
        f.write(result["merged_review"])
        if score != -1:
            f.write(f"\n\n**Predicted score: {score}**\n")
        if cal is not None:
            f.write(f"\n**Percentile of score={score}: {cal['percentile']:.1f}% (n={cal['n']}, {Path(cal['csv']).name})**\n")
            f.write(f"\n**Calibrated score (OLS vs real reviewer avg): {cal['calibrated_score']:.2f}**\n")
            f.write(f"\n**Acceptance rate @ score={score}: {cal['exact_rate']:.2%} (n={cal['exact_n']})**\n")
            f.write(f"\n**Acceptance rate @ score={score}+/-{cal['window']}: {cal['window_rate']:.2%} (n={cal['window_n']})**\n")
    print(f"\nReview written to {out_path}")


# ── CLI ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ARD-Review: agentic paper reviewer")
    parser.add_argument("--single_paper", type=str, required=True, help="Path to the paper (.pdf, .md or .txt)")
    parser.add_argument("--backend", choices=["codex", "claude"], help="Use an authenticated CLI for both stages (overrides model environment variables)")
    parser.add_argument("--model", help="Model for both CLI stages; omitted uses the CLI's configured default")
    parser.add_argument("--reasoning_effort", help="Reasoning effort passed to the selected backend")
    parser.add_argument("--no_cal", action="store_true", help="Skip calibration-anchor retrieval; score on paper merits alone")
    parser.add_argument("--calibration_mode", choices=["bm25", "vector"], help="CLI default: local BM25; SDK/API default: vector (requires embedding API credentials)")
    parser.add_argument("--pdf_parser", choices=["local", "datalab"], help="CLI default: local pdftotext; SDK/API default: Datalab")
    parser.add_argument("--skip_calibration", action="store_true", help="Do not map the raw score onto a reference run")
    parser.add_argument("--calibration_csv", type=Path, help="Reference run for score mapping (CLI backends have no matching shipped reference)")
    args = parser.parse_args()
    if args.model and not args.backend:
        parser.error("--model requires --backend; use HARSH_MODEL/MERGER_MODEL for mixed backends")
    if args.backend:
        harsh_model = merger_model = args.backend + (":" + args.model if args.model else "")
    else:
        harsh_model = os.environ.get("HARSH_MODEL", "")
        merger_model = os.environ.get("MERGER_MODEL", "")
        if not harsh_model or not merger_model:
            parser.error("select --backend codex/claude, or set HARSH_MODEL and MERGER_MODEL")
    uses_cli = bool(cli_model(harsh_model) or cli_model(merger_model))
    calibration_csv = args.calibration_csv
    if calibration_csv is None and not cli_model(merger_model):
        calibration_csv = RESULTS_DIR / ("claude.csv" if merger_model.startswith("claude_sdk:") else "deepseek.csv")
    config = ReviewConfig(
        harsh_model, merger_model, args.reasoning_effort or os.environ.get("REASONING_EFFORT"),
        args.calibration_mode or os.environ.get("REVIEW_CALIBRATION_MODE") or ("bm25" if uses_cli else "vector"),
        args.pdf_parser or ("local" if uses_cli else "datalab"), calibration_csv,
    )
    if not Path(args.single_paper).is_file():
        parser.error(f"paper not found: {args.single_paper}")
    if Path(args.single_paper).suffix.lower() not in {".pdf", ".md", ".txt"}:
        parser.error("paper must be a .pdf, .md or .txt file")
    if not args.skip_calibration and calibration_csv is not None and not calibration_csv.is_file():
        parser.error(f"calibration CSV not found: {calibration_csv}")
    print(f"Models: critic={harsh_model}, merger={merger_model}")
    print(f"Calibration retrieval: {'disabled' if args.no_cal else config.calibration_mode}")
    if cli_model(merger_model) and calibration_csv is None and not args.skip_calibration:
        print("Reference score mapping skipped: no matching CLI reference run. Use --calibration_csv to opt into a reference.")
    try:
        if not args.no_cal:
            if not Path(CALIBRATION_REVIEW_DIR).is_dir():
                raise FileNotFoundError(f"Calibration corpus not found at {CALIBRATION_REVIEW_DIR}. Run `python code/download_calibration_corpus.py` first, or pass --no_cal.")
            if config.calibration_mode == "vector":
                embedding_config()
        if config.calibration_mode not in {"bm25", "vector"}:
            raise ValueError("REVIEW_CALIBRATION_MODE must be bm25 or vector.")
        for model in (harsh_model, merger_model):
            cli = cli_model(model)
            if cli:
                import shutil
                executable = os.environ.get(f"{cli[0].upper()}_CLI_PATH", cli[0])
                if not shutil.which(executable):
                    raise ValueError(f"{executable!r} was not found; install and sign in to {cli[0]} first.")
            elif not model.startswith("claude_sdk:"):
                if not API_KEY or not config.effort:
                    raise ValueError("The API backend requires REVIEW_API_KEY and REASONING_EFFORT (or --reasoning_effort).")
        asyncio.run(run_single_paper(args.single_paper, config, no_cal=args.no_cal, skip_calibration=args.skip_calibration))
    except (RuntimeError, ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
