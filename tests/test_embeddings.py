import os
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import tools


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {}, clear=True).start()
        patch.multiple(tools, API_BASE_URL="https://openrouter.ai/api/v1", API_KEY="reviewer-key",
                       vectors=None, filenames=[], score_index={}, embed_client=None).start()

    def test_legacy_credentials_and_separate_gemini_settings(self):
        self.assertEqual(tools.embedding_config(),
                         ("https://openrouter.ai/api/v1", "reviewer-key", "google/gemini-embedding-001"))
        os.environ.update({"REVIEW_EMBEDDING_BASE_URL": "https://generativelanguage.googleapis.com/v1beta/openai/",
                           "REVIEW_EMBEDDING_API_KEY": "gemini-test-key"})
        with patch.object(tools, "API_KEY", ""):
            self.assertEqual(tools.embedding_config(),
                             (os.environ["REVIEW_EMBEDDING_BASE_URL"], "gemini-test-key", "gemini-embedding-001"))

    def test_different_endpoint_never_receives_reviewer_key(self):
        os.environ["REVIEW_EMBEDDING_BASE_URL"] = "https://generativelanguage.googleapis.com/v1beta/openai/"
        with self.assertRaisesRegex(ValueError, "REVIEW_EMBEDDING_API_KEY"):
            tools.embedding_config()

    def test_explicit_model_is_preserved(self):
        os.environ["REVIEW_EMBEDDING_MODEL"] = "explicit-embedding-model"
        self.assertEqual(tools.embedding_config()[2], "explicit-embedding-model")

    def setup_corpus(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        vectors = {"low.md": [1., 0.], "middle.md": [.9, .1], "high.md": [0., 1.]}
        scores = {"low.md": 2., "middle.md": 6., "high.md": 9.}
        for name in vectors:
            (root / name).write_text(f"# Anchor\n- Avg Score: {scores[name]}\n")
        for name, contents in (("human_reviews_embeddings_deepreview.pkl", vectors),
                               ("human_review_score_index_deepreview.pkl", scores)):
            with (root / name).open("wb") as stream:
                pickle.dump(contents, stream)
        patch.object(tools, "CALIBRATION_REVIEW_DIR", str(root)).start()
        patch.object(tools, "ensure_hf_file", side_effect=lambda name: str(root / name)).start()
        os.environ.update({"REVIEW_EMBEDDING_BASE_URL": "https://generativelanguage.googleapis.com/v1beta/openai/",
                           "REVIEW_EMBEDDING_API_KEY": "gemini-test-key"})
        client = Mock()
        client.embeddings.create.return_value = SimpleNamespace(data=[SimpleNamespace(embedding=[1., 0.])])
        constructor = patch("openai.OpenAI", return_value=client).start()
        return client, constructor

    def test_vector_retrieval_uses_gemini_key_and_exclusive_score_bounds(self):
        client, constructor = self.setup_corpus()
        with patch.object(tools, "API_KEY", ""):
            result = tools.search_file_impl("routing constraints", 2, "vector", 2, 9)
        self.assertIn("middle.md", result)
        self.assertNotIn("low.md", result)
        self.assertNotIn("high.md", result)
        constructor.assert_called_once_with(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key="gemini-test-key")
        client.embeddings.create.assert_called_once_with(
            model="gemini-embedding-001", input="routing constraints", encoding_format="float")

    def test_dimension_mismatch_is_actionable(self):
        client, _ = self.setup_corpus()
        client.embeddings.create.return_value.data[0].embedding = [1., 0., 0.]
        with self.assertRaisesRegex(ValueError, "Embedding dimension mismatch"):
            tools.search_file_impl("routing constraints", 1, "vector")

    def test_nonfinite_query_is_rejected(self):
        client, _ = self.setup_corpus()
        client.embeddings.create.return_value.data[0].embedding = [float("nan"), 0.]
        with self.assertRaisesRegex(ValueError, "non-finite"):
            tools.search_file_impl("routing constraints", 1, "vector")


if __name__ == "__main__":
    unittest.main()
