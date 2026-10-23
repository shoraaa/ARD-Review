# ARD-Review
Paper: https://doi.org/10.13140/RG.2.2.34444.48005
Try it On CSPaper: https://cspaper.org/review?venue=ARD-Review

An agentic paper reviewer. Give it a paper (PDF, Markdown or plain text) and it
produces a full peer review plus a numeric score that is calibrated against real
human reviews.

This repository is the **minimal software release**: the working pipeline and
nothing else. The experiment harness, baselines, ablations, judge setups and the
paper datasets are not here — the complete research repository will be released
separately, later.

---

## How it works

![ARD-Review pipeline](figures/pipeline.png)

The pipeline is two agent stages. Neither stage ever receives the paper inline;
both read it from disk through a sandboxed `read_file` / `grep_file` tool pair,
so a long paper is consumed and reasoned about incrementally instead of dumped
into one context window.

**Stage 1 — Harsh Critic** (`prompts/harsh_critic.md`)
Reads the paper in sequential chunks. After every chunk it stops and reasons
about what that part of the paper claims and whether the evidence holds, building
its assessment as it goes. Output is a deliberately critical review.

**Stage 2 — Merger** (`prompts/merger.md`)
Takes the critic's review and cross-checks every single weakness against the
actual paper text before keeping it, dropping the ones that do not survive
verification. It then calibrates the score by *comparison*, not by intuition:

- `draft_review` — the merger must commit its post-filtering draft before it is
  allowed to look at any anchor, so retrieval cannot rationalise a score it
  already decided on.
- `calibration_search` — iterative RAG over a corpus of ~13k real human-reviewed
  papers. Round 1 brackets the plausible score band by retrieving anchors from
  each band (strong reject through strong accept); rounds 2-3 narrow inside that
  bracket. At most 3 batched calls per review.
- Anchors are then read in full with `read_file` and compared against the paper
  under review.

The final review carries `<score>` and `<decision>` tags, which the pipeline
parses out.

Both stages can use any of these backends, including mixed backends:

| Backend | Selected by | Uses |
| --- | --- | --- |
| Codex CLI | `--backend codex` or `codex[:model]` | installed `codex exec`, using its existing authentication |
| Claude Code CLI | `--backend claude` or `claude[:model]` | installed `claude --print`, using its existing authentication |
| OpenAI Agents SDK | any plain model id | `OpenAIResponsesModel` against any OpenAI-compatible endpoint (default OpenRouter) |
| Claude Agent SDK | model id prefixed `claude_sdk:` | `ClaudeSDKClient` with the tools exposed as an in-process MCP server |

The CLI backends use the same stage prompts and a local stdio MCP server for
`read_file`, `grep_file`, `draft_review`, and `calibration_search`. Each stage
starts a fresh session. The CLI tool server enforces the draft-before-retrieval
protocol and the three-batch retrieval limit.

---

## Install

```bash
git clone https://github.com/weathon/ARD-Review.git
cd ARD-Review
pip install -r requirements.txt
cp .env.example .env    # optional for CLI-only runs
```

For direct CLI use, install and sign in to `codex` or `claude` first. Review
sessions reuse that CLI's authentication; no `REVIEW_API_KEY`, `EXTRACTOR_MODEL`,
or Datalab key is needed with the local defaults. Subscription limits still
apply, and the CLIs still contact their model services. See the
[Codex non-interactive documentation](https://developers.openai.com/codex/noninteractive)
and [Claude Code CLI reference](https://code.claude.com/docs/en/cli-reference).

CLI runs use the CLI's configured model unless `--model` is supplied. The API
backend still requires explicit model IDs and reasoning effort. See
`.env.example` for configuration.

For the Claude backend, either put an Anthropic key in `ANTHROPIC_API_KEY` or
leave it empty to use an authenticated Claude subscription.

### Calibration corpus

The calibration RAG needs a corpus of human-reviewed papers. No dataset is
shipped in this repository; build it from the public
[DeepReview-13K](https://huggingface.co/datasets/WestlakeNLP/DeepReview-13K)
dataset:

```bash
python code/download_calibration_corpus.py
```

This writes `datasets/deepreview_13k_calibration/<paper_id>.md` (about 13k files,
around 220 MB). The matching embedding matrix and per-paper average-score index
are downloaded automatically from
[weathon/paper_embeddings](https://huggingface.co/datasets/weathon/paper_embeddings)
the first time they are needed — no embedding calls and no cost on your side.

CLI runs default to **local BM25 keyword retrieval**, reading scores from the
corpus files. This requires no embedding API or downloaded embedding matrix.
It keeps the anchor-comparison workflow but differs from the vector retrieval
used in the published reference runs. Select `--calibration_mode vector` and
configure embedding credentials to use semantic retrieval; embedding each
search query makes an API call.

To use a **Gemini API key directly** while keeping the reviewer on Codex or
Claude, set these in your ignored `.env`:

```dotenv
REVIEW_EMBEDDING_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
REVIEW_EMBEDDING_API_KEY=your-gemini-api-key
REVIEW_EMBEDDING_MODEL=gemini-embedding-001
```

```bash
uv run python code/main.py --backend codex --single_paper paper.md --calibration_mode vector
```

Google supports `gemini-embedding-001` through its
[OpenAI-compatible embedding endpoint](https://ai.google.dev/gemini-api/docs/openai#embeddings).
The existing corpus vectors expect that model; switching to `text-embedding-004`
or Gemini Embedding 2 requires rebuilding the corpus embeddings. Query/corpus
dimension mismatches produce an explicit error.

The first vector retrieval downloads the existing corpus vectors and score
index; it does not re-embed the 13k papers. Only the search queries are embedded
with your key. The embedding key and endpoint are independent of reviewer
authentication. For backward compatibility, they fall back to `REVIEW_API_KEY`
and `REVIEW_API_BASE_URL` when no separate endpoint is selected.

The corpus file format is load-bearing: the published embeddings were computed
from exactly these files, so do not reformat or rename them.

To run without the corpus at all, pass `--no_cal`; the merger then loses
`calibration_search` and scores on the paper's merits alone. This is a different
method, not a cheaper version of the same one, and the shipped reference runs do
not apply to it.

---

## Usage

Direct CLI review, without API credentials or the calibration corpus:

```bash
python code/main.py --backend codex --single_paper paper.pdf --no_cal
python code/main.py --backend claude --single_paper paper.md --no_cal

# Equivalent convenience scripts; additional review flags are forwarded.
./run_codex.sh paper.pdf --no_cal
./run_claude_cli.sh paper.md --no_cal --model sonnet
```

Remove `--no_cal` after downloading the calibration corpus to retrieve local
anchors. For PDFs, CLI runs use Poppler's `pdftotext` (on Debian/Ubuntu:
`sudo apt install poppler-utils`) and save the full extracted text as
`paper.extracted.md`. Local extraction does not perform OCR on scanned PDFs.
Markdown and text inputs need no converter.

To choose different CLIs or models for the two stages:

```bash
HARSH_MODEL=codex MERGER_MODEL=claude:sonnet \
python code/main.py --single_paper paper.md --no_cal
```

`--backend` overrides both model environment variables. `--model` applies to
both stages and requires `--backend`. Omit `REASONING_EFFORT` to keep each CLI's
default, or pass `--reasoning_effort` with a value supported by the chosen model.
`CODEX_CLI_PATH` / `CLAUDE_CLI_PATH` can select an executable outside `PATH`.
`REVIEW_CLI_TIMEOUT` sets the per-stage timeout in seconds (default 1800).
Each stage prints the path to a live log under `results/cli/`. Codex tool calls
and CLI diagnostics appear immediately in the terminal; a heartbeat every 15
seconds shows elapsed time and time since the last output. Logs contain the raw
stdout/stderr and persist on errors, timeouts, and Ctrl-C. To follow the detailed
output in another terminal, run `tail -f` on the printed log path. A long paper
can take several minutes per stage with a high reasoning effort. Each completed
stage is also saved as a neighboring `.md` file, and `results/pipeline.log`
records stage starts and links to their logs immediately.
If a CLI omits valid score/decision tags, the full review is saved and the
missing values are reported; no API extraction call is made.

Existing SDK entry points remain available:

```bash
./run_claude.sh  /abs/path/to/paper.pdf     # both stages on Claude Agent SDK
./run_openai.sh  /abs/path/to/paper.pdf     # both stages on OpenAI Agents SDK
```

Or directly:

```bash
HARSH_MODEL="claude_sdk:claude-sonnet-4-6" \
MERGER_MODEL="claude_sdk:claude-sonnet-4-6" \
REASONING_EFFORT="max" \
EXTRACTOR_MODEL="deepseek/deepseek-v4-flash" \
python code/main.py --single_paper paper.pdf
```

Flags:

| Flag | Effect |
| --- | --- |
| `--single_paper PATH` | the paper to review (`.pdf`, `.md`, `.txt`) |
| `--backend codex\|claude` | use an installed CLI for both stages |
| `--model MODEL` | override that CLI's configured model for both stages |
| `--reasoning_effort LEVEL` | override `REASONING_EFFORT` or the CLI default |
| `--no_cal` | drop `calibration_search`; score on paper merits alone |
| `--calibration_mode bm25\|vector` | local keyword retrieval or API query embeddings |
| `--pdf_parser local\|datalab` | local `pdftotext` or the Datalab API |
| `--skip_calibration` | do not map the raw score onto the reference runs |
| `--calibration_csv PATH` | explicitly choose a reference CSV for score mapping |

SDK/API runs default to converting PDFs to Markdown via the Datalab API (needs
`DATALAB_API_KEY`); the converted `.md` is written next to the PDF. References
and appendix are dropped by the converter, and the prompts tell both agents to
treat parser artifacts as parser problems rather than paper flaws.

Outputs:

- `results/reviews/<paper>_review_<timestamp>.md` — the review, its score, and
  the calibration read-out
- `results/pipeline.log` — per-run token usage, cost, both stages' raw outputs

---

## Score calibration

A raw score from a language model means little on its own. This release ships two
reference runs of this exact pipeline over ICLR 2026 submissions whose real
reviewer scores and decisions are known, and every review is reported against
them:

| File | Backend | n |
| --- | --- | --- |
| `results/claude.csv` | Claude Sonnet 4.6, Claude Agent SDK | 393 |
| `results/deepseek.csv` | DeepSeek V4 Flash over OpenRouter, OpenAI Agents SDK | 388 |

![Pipeline score against the mean real reviewer score, per reference run](figures/calibration_scatter.png)

Every point is one submission: the pipeline's score against the mean of that
paper's real reviewer scores, coloured by the real accept/reject outcome.

![Real accept/reject composition of each pipeline-score bin](figures/accept_rate_bars.png)

The same runs binned by pipeline score, each column normalised: the share of
papers in that bin that were really accepted and really rejected.

For SDK/API runs, the set is chosen automatically from the merger backend you are running
(`claude_sdk:` maps to `claude.csv`, anything else to `deepseek.csv`), and the
pipeline reports three things:

- **Percentile** — where the score sits in that run's own score distribution.
- **Calibrated score** — ordinary least squares fit of pipeline score onto the
  mean real reviewer score, applied to your score, reported with the fit's
  Pearson r, R-squared and residual MAE.
- **Empirical acceptance rate** — the fraction of reference papers at that exact
  score, and within a 0.5 window of it, that were actually accepted.

CLI merger runs skip this reference mapping by default: neither CLI backend has
a matching shipped reference run. You can opt into a chosen reference with
`--calibration_csv results/claude.csv`, for example, but its fit is specific to
the original model/runtime/retrieval setup. This is separate from retrieving
calibration anchors, which stays enabled unless `--no_cal` is supplied.

Fits over the shipped reference runs:

| Reference run | n | OLS fit | Pearson r | R-squared | Spearman | MAE raw | MAE calibrated | Decision accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `claude.csv` | 393 | `gt = 0.879 x pred - 0.402` | 0.692 | 0.479 | 0.635 | 1.39 | 1.04 | 70.2% |
| `deepseek.csv` | 388 | `gt = 1.013 x pred - 0.890` | 0.596 | 0.355 | 0.528 | 1.39 | 1.21 | 67.3% |

Scores are on the reviewer scale of the submissions in the set;
`gt_avg_score` is the mean of that paper's real reviewer scores and `gt_binary`
its real accept/reject outcome. Base accept rate is 40.7% and 39.9%
respectively, so decision accuracy is measured against a majority-class baseline
of about 60%.

The CSVs hold predictions and public outcomes only. No paper text, no human
review text, no dataset.

---

## Comparison

Review quality, judged head to head. Each of the 392 papers in the common set
gives one match of this pipeline against one competing system; a blind judge
(Claude Sonnet 4.6, scoring the six deficiency categories) reads the two reviews
and picks the better one, with a single seeded coin flip per match deciding
which review is shown first.

| Compared against | Win rate | Wins / matches |
| --- | --- | --- |
| Direct single-pass review | 0.941 | 369 / 392 |
| CSPaper | 0.522 | 204 / 391 |
| DeepReviewer 14B, meta review | 0.987 | 387 / 392 |
| DeepReviewer v2 | 0.753 | 295 / 392 |
| **Overall** | **0.801** | **1255 / 1567** |

Only this pipeline is compared against the others: there are no
opponent-versus-opponent matches and no full grid. Human reviews are excluded
from the comparison, because they were written against an earlier revision of
each paper than the final text the judge reads. One CSPaper match was dropped
because that system's review for the paper is missing its outcome section.

---

## Layout

```
code/main.py                          two-stage pipeline, CLI, score calibration
code/cli_backend.py                   Codex/Claude subprocess runners and usage parsing
code/review_mcp.py                    stdio tools for CLI review sessions
code/tools.py                         sandboxed file tools + calibration RAG index
code/claude_merger.py                 Claude Agent SDK backend (tools over MCP)
code/paths.py                         path resolution, HuggingFace asset fetch
code/download_calibration_corpus.py   builds the calibration corpus
prompts/harsh_critic.md               stage 1
prompts/merger.md                     stage 2
prompts/cal_with.md                   calibration protocol injected into stage 2
prompts/cal_without.md                its --no_cal replacement
prompts/timeline.md                   current-date grounding for both stages
results/claude.csv                    reference run, Claude backend
results/deepseek.csv                  reference run, OpenAI Agents SDK backend
figures/pipeline.py                   source of the pipeline figure (+ .pdf / .png)
figures/calibration_scatter.py        source of the scatter above (+ .pdf / .png)
figures/accept_rate_bars.py           source of the stacked bars above (+ .pdf / .png)
```

Each figure script writes a PDF (for LaTeX) and a PNG (for this page) from the
same source; run them with `python figures/<name>.py`. The pipeline script also
warns if any label no longer fits its box.

Run the offline integration tests with `python -m unittest discover -s tests -v`.
They exercise both CLI adapters and the real MCP server using simulated CLI
responses, without model calls or credentials.

---

## Notes

- Reviewing one paper is a handful of long-context agent turns per stage. Cost is
  dominated by the models you point it at; the Claude SDK path prints its exact
  cost per run.
- The pipeline fails loudly. If a stage returns nothing, if the corpus is
  missing, or if a required variable is unset, it raises rather than substituting
  something else and reporting a number that looks fine.
- The one exception is score extraction: if the merger's own `<score>` /
  `<decision>` tags cannot be parsed, `EXTRACTOR_MODEL` is asked to pull them out
  of the review text, and is instructed to return a sentinel rather than guess.

## Citation

The paper and the full research repository are not yet public. Please check back.
