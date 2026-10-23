import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import cli_backend
import main as pipeline
import tools
from review_mcp import make_server


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="review test ")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.paper = self.root / "paper with spaces.md"
        self.paper.write_text("# A synthetic paper\nA synthetic result.\n")
        self.fake_cli = self.root / "fake cli"
        self.fake_cli.write_text(f"#!{sys.executable}\n" + (ROOT / "tests/fake_review_cli.py").read_text())
        self.fake_cli.chmod(0o755)
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith(("REVIEW_", "HARSH_", "MERGER_", "REASONING_", "EXTRACTOR_", "EXPECTED_", "FAKE_"))}
        self.env.update({"PYTHON_DOTENV_DISABLED": "1", "REVIEW_API_KEY": "",
                         "OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": "",
                         "REVIEW_RESULTS_DIR": str(self.root / "results"),
                         "REVIEW_DATASETS_DIR": str(self.root / "datasets"),
                         "CODEX_CLI_PATH": str(self.fake_cli), "CLAUDE_CLI_PATH": str(self.fake_cli)})

    def run_pipeline(self, *flags, env=None):
        result = subprocess.run([sys.executable, str(ROOT / "code/main.py"),
                                 "--single_paper", str(self.paper), *flags],
                                env=env or self.env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_both_backends_without_keys_or_corpus(self):
        for backend in ("codex", "claude"):
            with self.subTest(backend=backend):
                result = self.run_pipeline("--backend", backend, "--no_cal", "--model", "test-model",
                                           env={**self.env, "EXPECTED_MODEL": "test-model"})
                self.assertIn("Predicted score: 7.5", result.stdout)
                self.assertIn("Reference score mapping skipped", result.stdout)
                reviews = list((self.root / "results/reviews").glob("*.md"))
                self.assertTrue(reviews)
                self.assertIn("Final checked review", reviews[0].read_text())
                self.assertIn("CLI Usage", (self.root / "results/pipeline.log").read_text())
                logs = list((self.root / "results/cli").glob(f"*_{backend}_*.log"))
                self.assertEqual(len(logs), 2)
                self.assertTrue(all("exit=0" in log.read_text() for log in logs))

    def test_mixed_cli_models_and_local_calibration(self):
        corpus = self.root / "datasets/deepreview_13k_calibration"
        corpus.mkdir(parents=True)
        for i, score in enumerate((2, 6, 9)):
            (corpus / f"{i}.md").write_text(f"# Synthetic anchor\n- Scores: {score}, {score}\nBaseline comparison")
        self.run_pipeline(env={**self.env, "HARSH_MODEL": "codex", "MERGER_MODEL": "claude"})
        self.assertFalse(list((self.root / "datasets").glob("*.pkl")))

    def test_malformed_score_preserves_review_without_extractor(self):
        result = self.run_pipeline("--backend", "codex", "--no_cal", env={
            **self.env, "FAKE_BAD_TAGS": "1", "EXTRACTOR_MODEL": "must-not-call", "REVIEW_API_KEY": "not-a-real-key",
        })
        self.assertIn("missing/invalid", result.stdout)
        self.assertIn("<score>invalid</score>", next((self.root / "results/reviews").glob("*.md")).read_text())

    def test_help_without_configuration(self):
        result = subprocess.run([sys.executable, str(ROOT / "code/main.py"), "--help"],
                                env=self.env, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--backend", result.stdout)

    def test_missing_executable_is_actionable(self):
        result = subprocess.run([sys.executable, str(ROOT / "code/main.py"), "--backend", "codex",
                                 "--single_paper", str(self.paper), "--no_cal"],
                                env={**self.env, "CODEX_CLI_PATH": str(self.root / "missing")},
                                capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("install and sign in", result.stderr)

    def test_relative_executable_path(self):
        result = subprocess.run([sys.executable, str(ROOT / "code/main.py"), "--backend", "codex",
                                 "--single_paper", str(self.paper), "--no_cal"],
                                cwd=self.root, env={**self.env, "CODEX_CLI_PATH": "./fake cli"},
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class ToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_calibration_protocol_and_score_filtering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for score in (2, 6, 9):
                (root / f"{score}.md").write_text(f"# Anchor {score}\n- Scores: {score}, {score}\nSynthetic study")
            with patch.multiple(tools, CALIBRATION_REVIEW_DIR=directory, database={}, score_index={}, ALLOWED_PATHS=[]):
                server = make_server(str(root / "paper.md"), "merger", False, "bm25")
                query = {"queries": [{"query": "study", "low_score": 2, "high_score": 9}]}
                with self.assertRaisesRegex(Exception, "draft_review"):
                    await server.call_tool("calibration_search", query)
                await server.call_tool("draft_review", {"draft": "Checked draft"})
                with self.assertRaisesRegex(Exception, "already"):
                    await server.call_tool("draft_review", {"draft": "Replacement"})
                for _ in range(3):
                    result = str(await server.call_tool("calibration_search", query))
                    self.assertIn("avg_score: 6.00", result)
                    self.assertNotIn("avg_score: 2.00", result)
                    self.assertNotIn("avg_score: 9.00", result)
                with self.assertRaisesRegex(Exception, "three"):
                    await server.call_tool("calibration_search", query)

    async def test_file_access_rejects_siblings_prefixes_and_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            allowed = root / "allowed"
            allowed.mkdir()
            paper = allowed / "paper.md"
            paper.write_text("first\nsecond\n")
            secret = root / "allowed-secret.md"
            secret.write_text("private")
            link = allowed / "link.md"
            link.symlink_to(secret)
            with patch.object(tools, "ALLOWED_PATHS", [str(allowed)]):
                self.assertEqual(tools.read_file(str(paper), 2, 2), "2: second\n")
                self.assertIn("Access denied", tools.read_file(str(secret)))
                self.assertIn("Access denied", tools.grep_file("private", str(link)))
            with patch.object(tools, "ALLOWED_PATHS", [str(paper)]):
                self.assertIn("Access denied", tools.read_file(str(link)))

    async def test_api_stage_still_exposes_file_and_calibration_tools(self):
        config = pipeline.ReviewConfig("api-model", "api-model", "high")
        seen = []

        async def fake_run(agent, prompt):
            seen.append({tool.name for tool in agent.tools})
            return "review", None

        with patch.object(pipeline, "API_KEY", "test-key"), patch.object(pipeline, "run_agent_with_retry", fake_run):
            await pipeline.run_stage(config, "api-model", "critic", "prompt", "/tmp/paper.md", True)
            await pipeline.run_stage(config, "api-model", "merger", "prompt", "/tmp/paper.md", False)
        self.assertEqual(seen, [{"read_file", "grep_file"}, {"read_file", "grep_file", "draft_review", "calibration_search"}])

    async def test_local_pdf_and_empty_extraction(self):
        with patch("shutil.which", return_value="/usr/bin/pdftotext"), patch.object(cli_backend, "run_process", AsyncMock(return_value=("synthetic text", ""))):
            self.assertEqual(await pipeline.pdf_to_markdown(Path("test.pdf"), "local"), "synthetic text")
        with patch("shutil.which", return_value="/usr/bin/pdftotext"), patch.object(cli_backend, "run_process", AsyncMock(return_value=("", ""))):
            with self.assertRaisesRegex(ValueError, "no extractable text"):
                await pipeline.pdf_to_markdown(Path("test.pdf"), "local")

    async def test_timeout_and_process_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "timeout.log"
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                await cli_backend.run_process([sys.executable, "-c", "import time; print('started', flush=True); time.sleep(20)"],
                                              "", cwd=directory, timeout=0.2, log_path=log_path)
            self.assertIn("started", log_path.read_text())
            self.assertIn("timed out", log_path.read_text())
            with self.assertRaisesRegex(RuntimeError, "not logged in"):
                await cli_backend.run_process([sys.executable, "-c", "import sys; sys.stderr.write('not logged in'); sys.exit(1)"], "", cwd=directory, timeout=5)

    async def test_logs_are_flushed_before_exit_and_large_events_are_drained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release = root / "release"
            log_path = root / "live.log"
            script = (
                "import sys, time\nfrom pathlib import Path\n"
                "print('ready', flush=True)\n"
                "while not Path(sys.argv[1]).exists(): time.sleep(0.01)\n"
                "sys.stdout.write('界' * 100000 + '\\n')\n"
                "sys.stderr.write('diagnostic' * 10000)\n"
            )
            run = asyncio.create_task(cli_backend.run_process(
                [sys.executable, "-c", script, str(release)], "", cwd=directory,
                timeout=10, log_path=log_path,
            ))
            try:
                async with asyncio.timeout(3):
                    while not log_path.exists() or "ready" not in log_path.read_text():
                        await asyncio.sleep(0.01)
                self.assertFalse(run.done(), "Log must be readable while the child is still running")
                release.touch()
                stdout, stderr = await run
                self.assertEqual(stdout, "ready\n" + "界" * 100000 + "\n")
                self.assertEqual(stderr, "diagnostic" * 10000)
                self.assertIn("exit=0", log_path.read_text())
            finally:
                if not run.done():
                    run.cancel()
                    await asyncio.gather(run, return_exceptions=True)

    async def test_cancellation_keeps_log_and_reaps_process(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "cancel.log"
            run = asyncio.create_task(cli_backend.run_process(
                [sys.executable, "-c", "import time; print('ready', flush=True); time.sleep(20)"], "",
                cwd=directory, timeout=30, log_path=log_path,
            ))
            try:
                async with asyncio.timeout(3):
                    while not log_path.exists() or "ready" not in log_path.read_text():
                        await asyncio.sleep(0.01)
            finally:
                run.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await run
            self.assertIn("interrupted: CancelledError", log_path.read_text())
            self.assertIn("exit=-", log_path.read_text())

    async def test_cli_error_results_and_missing_tags(self):
        for result in ({"is_error": True, "result": "Login required"}, {"subtype": "error_max_turns"},
                       {"permission_denials": ["read_file"]}):
            with self.assertRaises(RuntimeError):
                cli_backend.parse_claude_output(json.dumps(result))
        with self.assertRaises(RuntimeError):
            cli_backend.parse_claude_output("not JSON")
        with self.assertRaises(RuntimeError):
            cli_backend.parse_codex_output('{"type":"turn.failed","error":"bad model"}', Path("missing"))
        for review in ("no tags", "<score>nan</score>", "<score>7", "<score>100</score>"):
            self.assertEqual(pipeline.parse_review_tags(review), (-1, "N/A"))
        self.assertEqual(pipeline.parse_review_tags("<score>6.5</score><decision>accept</decision>"), (6.5, "Accept"))


if __name__ == "__main__":
    unittest.main()
