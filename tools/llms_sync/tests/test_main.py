import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.llms_sync import main


class ParseNameStatusTests(unittest.TestCase):
    def test_parses_modified_deleted_and_renamed_entries(self) -> None:
        diff = "M\tREADME.md\nD\told.md\nR100\ta.md\tb.md\n"
        parsed = main.parse_name_status(diff)
        self.assertEqual(
            parsed,
            [
                {"status": "M", "path": "README.md"},
                {"status": "D", "path": "old.md"},
                {"status": "R", "old_path": "a.md", "path": "b.md"},
            ],
        )


class FailureClassificationTests(unittest.TestCase):
    def test_classifies_quota_failure(self) -> None:
        error = main.classify_api_failure(
            stage="translate:test",
            http_status=402,
            body_text='{"error":{"message":"quota exceeded"}}',
            raw_message="provider said no",
        )
        self.assertEqual(error.category, "quota_exceeded")

    def test_classifies_rate_limit_failure(self) -> None:
        error = main.classify_api_failure(
            stage="translate:test",
            http_status=429,
            body_text='{"error":{"message":"rate limit"}}',
            raw_message="provider said slow down",
        )
        self.assertEqual(error.category, "rate_limited")

    def test_sanitizes_sensitive_provider_message(self) -> None:
        summary = main.sanitize_provider_message("authorization: Bearer sk-test-secret")
        self.assertEqual(
            summary,
            "Provider returned an error payload containing potentially sensitive fields. The raw payload was suppressed.",
        )


class NotebookTranslationTests(unittest.TestCase):
    def test_only_translates_markdown_cells(self) -> None:
        notebook = {
            "cells": [
                {"cell_type": "markdown", "source": ["# Title\n", "Paragraph\n"]},
                {"cell_type": "code", "source": ["print('hi')\n"], "outputs": []},
            ],
            "metadata": {"kernelspec": {"name": "python3"}},
            "nbformat": 4,
            "nbformat_minor": 5,
        }

        client = mock.Mock()
        client.translate.return_value = "# 标题\n段落\n"

        translated = main.translate_notebook(client, json.dumps(notebook), "translate:test")
        parsed = json.loads(translated)

        self.assertEqual(parsed["cells"][0]["source"], ["# 标题\n", "段落\n"])
        self.assertEqual(parsed["cells"][1]["source"], ["print('hi')\n"])


class SkipReasonTests(unittest.TestCase):
    def test_existing_unmanaged_file_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "README.md"
            path.write_text("current", encoding="utf-8")
            reason = main.build_skip_reason(path, None)
            self.assertEqual(reason, "existing translated file is not managed by automation yet")

    def test_diverged_managed_file_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "README.md"
            path.write_text("current", encoding="utf-8")
            reason = main.build_skip_reason(
                path,
                {"last_automated_target_hash": "different"},
            )
            self.assertEqual(reason, "local file diverged from last automated version")


if __name__ == "__main__":
    unittest.main()
