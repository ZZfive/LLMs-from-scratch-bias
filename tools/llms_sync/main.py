from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUPPORTED_SUFFIXES = {".md": "markdown", ".ipynb": "notebook"}
FAILURE_ISSUE_TITLE = "bot: translation sync failure"


@dataclass
class SyncFailure(Exception):
    category: str
    message: str
    stage: str
    http_status: int | None = None
    provider_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "message": self.message,
            "stage": self.stage,
            "http_status": self.http_status,
            "provider_message": self.provider_message,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def run_git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip() or "Unknown git error"
        raise SyncFailure(
            category="git_error",
            message=f"git {' '.join(args)} failed: {stderr}",
            stage="git",
        )
    return result.stdout


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_upstream_remote(repo_root: Path, upstream_url: str) -> None:
    try:
        current = run_git(["remote", "get-url", "upstream"], repo_root).strip()
        if current != upstream_url:
            run_git(["remote", "set-url", "upstream", upstream_url], repo_root)
    except SyncFailure:
        run_git(["remote", "add", "upstream", upstream_url], repo_root)


def fetch_upstream(repo_root: Path, target_ref: str) -> str:
    run_git(["fetch", "upstream", target_ref], repo_root)
    return run_git(["rev-parse", f"upstream/{target_ref}"], repo_root).strip()


def list_supported_files(repo_root: Path, target_ref: str) -> list[str]:
    raw = run_git(["ls-tree", "-r", "--name-only", f"upstream/{target_ref}"], repo_root)
    return [line for line in raw.splitlines() if Path(line).suffix.lower() in SUPPORTED_SUFFIXES]


def parse_name_status(diff_text: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for line in diff_text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        status = parts[0]
        if status.startswith("R") and len(parts) >= 3:
            entries.append({"status": "R", "old_path": parts[1], "path": parts[2]})
        elif len(parts) >= 2:
            entries.append({"status": status, "path": parts[1]})
    return entries


def diff_all_files(repo_root: Path, previous_sha: str, upstream_sha: str) -> list[dict[str, str]]:
    raw = run_git(
        ["diff", "--name-status", "--find-renames", previous_sha, upstream_sha],
        repo_root,
    )
    return parse_name_status(raw)


def git_show_text(repo_root: Path, ref: str, path: str) -> str:
    return run_git(["show", f"{ref}:{path}"], repo_root)


def string_to_lines(value: str) -> list[str]:
    return value.splitlines(keepends=True)


def read_local_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def build_markdown_messages(markdown: str) -> list[dict[str, str]]:
    system = (
        "You are a technical translator. Translate English educational machine learning "
        "content into Simplified Chinese. Preserve Markdown structure, heading levels, "
        "lists, tables, links, code fences, inline code, commands, paths, identifiers, "
        "URLs, and punctuation used for markup. Do not translate code. Return only the "
        "translated Markdown."
    )
    user = f"Translate the following Markdown content:\n\n{markdown}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_notebook_messages(markdown: str) -> list[dict[str, str]]:
    system = (
        "You are a technical translator. Translate English notebook markdown into "
        "Simplified Chinese. Preserve Markdown formatting, math, code fences, inline "
        "code, commands, paths, links, and identifiers. Return only translated markdown."
    )
    user = f"Translate the following Jupyter markdown cell:\n\n{markdown}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def classify_api_failure(
    stage: str,
    http_status: int | None,
    body_text: str | None,
    raw_message: str,
) -> SyncFailure:
    lower_body = (body_text or "").lower()
    category = "unknown_api_error"
    if http_status == 401 or "invalid api key" in lower_body or "authentication" in lower_body:
        category = "authentication_failed"
    elif http_status == 402 or "quota" in lower_body or "credit" in lower_body or "balance" in lower_body:
        category = "quota_exceeded"
    elif http_status == 429 or "rate limit" in lower_body or "too many requests" in lower_body:
        category = "rate_limited"
    elif http_status in {400, 404} and "model" in lower_body and ("not found" in lower_body or "does not exist" in lower_body):
        category = "model_not_found"
    elif http_status is not None and http_status >= 500:
        category = "provider_service_error"
    elif "timed out" in lower_body or "timeout" in lower_body:
        category = "network_timeout"

    return SyncFailure(
        category=category,
        message=raw_message,
        stage=stage,
        http_status=http_status,
        provider_message=body_text,
    )


class TranslationClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("TRANSLATION_API_KEY", "").strip()
        self.base_url = os.getenv("TRANSLATION_BASE_URL", "").strip()
        self.model = os.getenv("TRANSLATION_MODEL", "").strip()
        self.timeout = int(os.getenv("TRANSLATION_TIMEOUT_SECONDS", "120").strip())

    def is_configured(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    def completions_url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    def translate(self, messages: list[dict[str, str]], stage: str) -> str:
        if not self.is_configured():
            raise SyncFailure(
                category="configuration_error",
                message="Translation provider configuration is incomplete.",
                stage=stage,
            )

        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": 0,
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            self.completions_url(),
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body_text = response.read().decode("utf-8")
                data = json.loads(body_text)
        except urllib.error.HTTPError as error:
            body_text = error.read().decode("utf-8", errors="replace")
            raise classify_api_failure(
                stage=stage,
                http_status=error.code,
                body_text=body_text,
                raw_message=f"Translation API returned HTTP {error.code}.",
            ) from error
        except urllib.error.URLError as error:
            raise classify_api_failure(
                stage=stage,
                http_status=None,
                body_text=str(error.reason),
                raw_message=f"Translation API request failed: {error.reason}",
            ) from error
        except TimeoutError as error:
            raise classify_api_failure(
                stage=stage,
                http_status=None,
                body_text="timeout",
                raw_message="Translation API request timed out.",
            ) from error

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise SyncFailure(
                category="unknown_response_format",
                message="Translation API response did not include choices[0].message.content.",
                stage=stage,
                provider_message=json.dumps(data, ensure_ascii=False)[:1200],
            ) from error

        if isinstance(content, list):
            fragments = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    fragments.append(item.get("text", ""))
            content = "".join(fragments)

        return str(content).strip()


def translate_markdown(client: TranslationClient, content: str, stage: str) -> str:
    if not content.strip():
        return content
    translated = client.translate(build_markdown_messages(content), stage)
    if content.endswith("\n") and not translated.endswith("\n"):
        translated += "\n"
    return translated


def translate_notebook(client: TranslationClient, content: str, stage: str) -> str:
    notebook = json.loads(content)
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "markdown":
            continue
        source = cell.get("source", [])
        text = "".join(source) if isinstance(source, list) else str(source)
        if not text.strip():
            continue
        translated = client.translate(build_notebook_messages(text), f"{stage}:cell:{index}")
        if text.endswith("\n") and not translated.endswith("\n"):
            translated += "\n"
        cell["source"] = string_to_lines(translated)
    return json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"


def make_relative(path: Path, repo_root: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def build_sync_report(result: dict[str, Any]) -> str:
    lines = [
        "# Upstream Sync Report",
        "",
        f"- Status: `{result['status']}`",
        f"- Generated at: `{result['generated_at']}`",
        f"- Upstream ref: `{result['target_ref']}`",
        f"- Upstream SHA: `{result['upstream_sha']}`",
        f"- Previous SHA: `{result.get('previous_sha') or 'none'}`",
        "",
    ]

    if result.get("bootstrap_only"):
        lines.extend(
            [
                "## Bootstrap",
                "",
                "The automation baseline was initialized without rewriting files.",
                "Use `workflow_dispatch` with `force_full_resync=true` if you want the bot to take over existing translated files.",
                "",
            ]
        )

    translated = result.get("translated_files", [])
    skipped = result.get("skipped_files", [])
    renamed = result.get("renamed_files", [])
    deleted = result.get("deleted_files", [])
    unsupported = result.get("unsupported_files", [])

    lines.extend(
        [
            "## Summary",
            "",
            f"- Updated files: {len(translated)}",
            f"- Skipped files: {len(skipped)}",
            f"- Deleted upstream files: {len(deleted)}",
            f"- Renamed upstream files: {len(renamed)}",
            f"- Unsupported changed files: {len(unsupported)}",
            "",
        ]
    )

    if translated:
        lines.extend(["## Updated Files", ""])
        for entry in translated:
            lines.append(f"- `{entry['path']}` ({entry['type']})")
        lines.append("")

    if skipped:
        lines.extend(["## Skipped Files", ""])
        for entry in skipped:
            lines.append(f"- `{entry['path']}`: {entry['reason']}")
        lines.append("")

    if deleted:
        lines.extend(["## Upstream Deletions", ""])
        for path in deleted:
            lines.append(f"- `{path}`")
        lines.append("")

    if renamed:
        lines.extend(["## Upstream Renames", ""])
        for entry in renamed:
            lines.append(f"- `{entry['old_path']}` -> `{entry['path']}`")
        lines.append("")

    if unsupported:
        lines.extend(["## Unsupported Changed Files", ""])
        for path in unsupported:
            lines.append(f"- `{path}`")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_pr_body(result: dict[str, Any], report_path: str) -> str:
    lines = [
        "## Automated Upstream Sync",
        "",
        f"- Upstream ref: `{result['target_ref']}`",
        f"- Upstream SHA: `{result['upstream_sha']}`",
        f"- Previous SHA: `{result.get('previous_sha') or 'none'}`",
        f"- Generated at: `{result['generated_at']}`",
        f"- Report: `{report_path}`",
        "",
        "### Summary",
        "",
        f"- Updated files: {len(result.get('translated_files', []))}",
        f"- Skipped files: {len(result.get('skipped_files', []))}",
        f"- Deleted upstream files reported only: {len(result.get('deleted_files', []))}",
        f"- Renamed upstream files reported only: {len(result.get('renamed_files', []))}",
        "",
    ]

    skipped = result.get("skipped_files", [])
    if skipped:
        lines.extend(["### Manual Review", ""])
        for entry in skipped[:20]:
            lines.append(f"- `{entry['path']}`: {entry['reason']}")
        if len(skipped) > 20:
            lines.append(f"- ... and {len(skipped) - 20} more")
        lines.append("")

    lines.extend(
        [
            "### Notes",
            "",
            "- This PR was created automatically by the upstream sync workflow.",
            "- Files are only rewritten when they are already managed by the bot or when they are newly added upstream.",
            "- If a translated file was edited manually after the last automated sync, it is skipped instead of being overwritten.",
            "",
        ]
    )

    return "\n".join(lines).rstrip() + "\n"


def build_failure_body(result: dict[str, Any]) -> str:
    failure = result["failure"]
    lines = [
        f"## {FAILURE_ISSUE_TITLE}",
        "",
        f"- Time: `{result['generated_at']}`",
        f"- Upstream ref: `{result['target_ref']}`",
        f"- Upstream SHA: `{result.get('upstream_sha') or 'unknown'}`",
        f"- Failure stage: `{failure['stage']}`",
        f"- Failure category: `{failure['category']}`",
        f"- Model: `{result.get('translation_model') or 'unset'}`",
        f"- Wrote files before failure: `{str(result.get('wrote_files', False)).lower()}`",
        "",
        "### Error",
        "",
        failure["message"],
        "",
    ]

    provider_message = sanitize_provider_message(failure.get("provider_message"))
    if provider_message:
        lines.extend(
            [
                "### Error Summary",
                "",
                "```text",
                provider_message,
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "### Suggested Checks",
            "",
            "- Confirm `TRANSLATION_API_KEY` is valid and not expired.",
            "- Confirm account quota, balance, or billing limits are sufficient.",
            "- Confirm `TRANSLATION_BASE_URL` points to a chat completions compatible endpoint.",
            "- Confirm `TRANSLATION_MODEL` still exists and is accessible to the API key.",
            "- Re-run the workflow manually after the provider issue is resolved.",
            "",
        ]
    )

    return "\n".join(lines).rstrip() + "\n"


def sanitize_provider_message(message: str | None) -> str | None:
    if not message:
        return None

    lowered = message.lower()
    sensitive_tokens = [
        "sk-",
        "api_key",
        "authorization",
        "bearer ",
        "x-api-key",
        "token",
        "secret",
        "base_url",
    ]
    if any(token in lowered for token in sensitive_tokens):
        return "Provider returned an error payload containing potentially sensitive fields. The raw payload was suppressed."

    compact = " ".join(message.split())
    if len(compact) > 500:
        compact = compact[:497] + "..."
    return compact


def determine_candidates(
    repo_root: Path,
    target_ref: str,
    previous_sha: str | None,
    force_full_resync: bool,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str], list[str]]:
    if force_full_resync:
        supported = list_supported_files(repo_root, target_ref)
        return [{"status": "A", "path": path} for path in supported], [], [], []

    if not previous_sha:
        return [], [], [], []

    parsed = diff_all_files(repo_root, previous_sha, f"upstream/{target_ref}")
    updatable: list[dict[str, str]] = []
    renamed: list[dict[str, str]] = []
    deleted: list[str] = []
    unsupported: list[str] = []
    for entry in parsed:
        path = entry["path"]
        suffix = Path(path).suffix.lower()
        if entry["status"] == "R":
            if suffix in SUPPORTED_SUFFIXES:
                renamed.append(entry)
            else:
                unsupported.append(path)
            continue

        if suffix not in SUPPORTED_SUFFIXES:
            unsupported.append(path)
            continue

        if entry["status"] == "D":
            deleted.append(path)
        else:
            updatable.append(entry)

    return updatable, renamed, deleted, unsupported


def build_skip_reason(local_path: Path, manifest_entry: dict[str, Any] | None) -> str | None:
    if manifest_entry is None:
        if local_path.exists():
            return "existing translated file is not managed by automation yet"
        return None

    if not local_path.exists():
        return None

    current_hash = sha256_text(read_local_text(local_path))
    if current_hash != manifest_entry.get("last_automated_target_hash"):
        return "local file diverged from last automated version"

    return None


def sync(args: argparse.Namespace) -> int:
    repo_root = resolve_repo_root()
    state_path = repo_root / args.state_path
    manifest_path = repo_root / args.manifest_path
    reports_dir = repo_root / args.reports_dir
    runtime_dir = repo_root / args.runtime_dir
    runtime_dir.mkdir(parents=True, exist_ok=True)

    state = load_json(
        state_path,
        {
            "last_synced_sha": None,
            "last_success_at": None,
            "automation_branch": args.automation_branch,
            "tracked_ref": args.target_ref,
        },
    )
    manifest = load_json(manifest_path, {"managed_files": {}})
    managed_files = manifest.setdefault("managed_files", {})
    client = TranslationClient()

    result: dict[str, Any] = {
        "status": "failure",
        "generated_at": utc_now(),
        "target_ref": args.target_ref,
        "translation_model": client.model,
        "wrote_files": False,
    }

    try:
        ensure_upstream_remote(repo_root, args.upstream_url)
        upstream_sha = fetch_upstream(repo_root, args.target_ref)
        previous_sha = state.get("last_synced_sha")
        result["upstream_sha"] = upstream_sha
        result["previous_sha"] = previous_sha

        if previous_sha == upstream_sha and not args.force_full_resync:
            result.update(
                {
                    "status": "success",
                    "bootstrap_only": False,
                    "has_changes": False,
                    "translated_files": [],
                    "skipped_files": [],
                    "deleted_files": [],
                    "renamed_files": [],
                    "unsupported_files": [],
                }
            )
            write_json(runtime_dir / "result.json", result)
            return 0

        if previous_sha is None and not args.force_full_resync:
            state["last_synced_sha"] = upstream_sha
            state["last_success_at"] = utc_now()
            state["tracked_ref"] = args.target_ref
            state["automation_branch"] = args.automation_branch
            write_json(state_path, state)
            result.update(
                {
                    "status": "success",
                    "bootstrap_only": True,
                    "has_changes": True,
                    "translated_files": [],
                    "skipped_files": [],
                    "deleted_files": [],
                    "renamed_files": [],
                    "unsupported_files": [],
                }
            )
        else:
            candidates, renamed_files, deleted_files, unsupported_files = determine_candidates(
                repo_root=repo_root,
                target_ref=args.target_ref,
                previous_sha=previous_sha,
                force_full_resync=args.force_full_resync,
            )

            max_changed_files = int(os.getenv("MAX_CHANGED_FILES_PER_RUN", "25"))
            if not args.force_full_resync and len(candidates) > max_changed_files:
                raise SyncFailure(
                    category="change_limit_exceeded",
                    message=(
                        f"Detected {len(candidates)} supported file changes, which exceeds the configured "
                        f"limit of {max_changed_files}. Increase MAX_CHANGED_FILES_PER_RUN or run a "
                        "manual force_full_resync."
                    ),
                    stage="planning",
                )

            translated_files: list[dict[str, str]] = []
            skipped_files: list[dict[str, str]] = []
            planned_writes: dict[Path, str] = {}
            planned_manifest_updates: dict[str, dict[str, Any]] = {}

            for entry in candidates:
                path = entry["path"]
                suffix = Path(path).suffix.lower()
                local_path = repo_root / path
                manifest_entry = managed_files.get(path)
                skip_reason = None if args.force_full_resync else build_skip_reason(local_path, manifest_entry)

                if skip_reason:
                    skipped_files.append({"path": path, "reason": skip_reason})
                    continue

                upstream_text = git_show_text(repo_root, f"upstream/{args.target_ref}", path)
                file_type = SUPPORTED_SUFFIXES[suffix]
                stage = f"translate:{path}"

                if file_type == "markdown":
                    translated_text = translate_markdown(client, upstream_text, stage)
                else:
                    translated_text = translate_notebook(client, upstream_text, stage)

                planned_writes[local_path] = translated_text
                planned_manifest_updates[path] = {
                    "type": file_type,
                    "last_automated_target_hash": sha256_text(translated_text),
                    "last_upstream_sha": upstream_sha,
                    "last_translated_at": utc_now(),
                }
                translated_files.append({"path": path, "type": file_type})

            for target, payload in planned_writes.items():
                write_text(target, payload)

            managed_files.update(planned_manifest_updates)
            state["last_synced_sha"] = upstream_sha
            state["last_success_at"] = utc_now()
            state["tracked_ref"] = args.target_ref
            state["automation_branch"] = args.automation_branch
            write_json(state_path, state)
            write_json(manifest_path, manifest)

            result.update(
                {
                    "status": "success",
                    "bootstrap_only": False,
                    "has_changes": bool(
                        planned_writes
                        or skipped_files
                        or deleted_files
                        or renamed_files
                        or unsupported_files
                    ),
                    "translated_files": translated_files,
                    "skipped_files": skipped_files,
                    "deleted_files": deleted_files,
                    "renamed_files": renamed_files,
                    "unsupported_files": unsupported_files,
                    "wrote_files": bool(planned_writes),
                }
            )

        report_name = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-upstream-sync.md"
        report_path = reports_dir / report_name
        write_text(report_path, build_sync_report(result))

        pr_title = f"chore(sync): upstream updates through {result['upstream_sha'][:12]}"
        pr_body_path = runtime_dir / "pr_body.md"
        write_text(pr_body_path, build_pr_body(result, make_relative(report_path, repo_root)))

        result.update(
            {
                "report_path": make_relative(report_path, repo_root),
                "pr_title": pr_title,
                "pr_body_path": make_relative(pr_body_path, repo_root),
            }
        )
        write_json(runtime_dir / "result.json", result)
        return 0
    except SyncFailure as error:
        result["failure"] = error.to_dict()
    except Exception as error:  # pragma: no cover
        result["failure"] = SyncFailure(
            category="unexpected_error",
            message=str(error),
            stage="unexpected",
        ).to_dict()

    failure_body_path = runtime_dir / "failure_body.md"
    write_text(failure_body_path, build_failure_body(result))
    result.update(
        {
            "status": "failure",
            "failure_body_path": make_relative(failure_body_path, repo_root),
        }
    )
    write_json(runtime_dir / "result.json", result)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synchronize translated content from upstream.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync_parser = subparsers.add_parser("sync", help="Run the upstream synchronization workflow.")
    sync_parser.add_argument("--upstream-url", required=True)
    sync_parser.add_argument("--target-ref", default="main")
    sync_parser.add_argument("--default-branch", default="main")
    sync_parser.add_argument("--automation-branch", default="bot/upstream-sync")
    sync_parser.add_argument("--force-full-resync", default="false")
    sync_parser.add_argument("--state-path", default=".sync/state.json")
    sync_parser.add_argument("--manifest-path", default=".sync/managed_manifest.json")
    sync_parser.add_argument("--reports-dir", default=".sync/reports")
    sync_parser.add_argument("--runtime-dir", default=".sync/runtime")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "sync":
        args.force_full_resync = str(args.force_full_resync).lower() == "true"
        return sync(args)
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
