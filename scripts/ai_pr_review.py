#!/usr/bin/env python3
"""AI pull-request reviewer, powered by GitHub Models.

Fetches a PR's title/body and unified diff from the GitHub REST API, asks a model
(via GitHub Models, OpenAI-compatible) for a structured review using a forced
``submit_review`` function call so the model must return schema-valid JSON,
renders one markdown comment, and upserts it on the PR so repeated pushes edit a
single comment instead of spamming new ones.

Why GitHub Models
-----------------
GitHub Models is free (rate-limited) and OpenAI-compatible, and the *same* GitHub
token authenticates both the REST calls and the inference call -- so there is no
separate API key to manage. In GitHub Actions, add ``models: read`` to the job's
permissions and the built-in ``GITHUB_TOKEN`` can do inference too; locally, a PAT
with the "Models" permission works for everything.

Design notes
------------
* It is "just a script in a repo" -- no server, no GitHub App.
* The bot NEVER merges, closes, or otherwise mutates the PR. It only comments.
* V1 is non-blocking: the process exits 0 even when it finds problems, UNLESS
  ``BLOCKING=1`` is set AND the review says ``do_not_merge`` -- only then exit 1.
* Every pure helper here (request building, response parsing, prompt building,
  truncation, comment rendering, exit policy) is importable and unit-testable
  with no network and no token (see test_ai_pr_review.py).

Configuration (all via environment variables):

    GITHUB_TOKEN         (required)  GitHub token; used for REST and (by default) inference
    GITHUB_MODELS_TOKEN  (optional)  separate token for inference; defaults to GITHUB_TOKEN
    REPO                 (required)  "owner/repo"
    PR_NUMBER            (required)  pull request number to review
    BASE_BRANCH          (optional)  target branch, default "main"
    MODEL                (optional)  GitHub Models id, default "openai/gpt-4.1"
    MODELS_ENDPOINT      (optional)  inference base URL, default GitHub Models
    MAX_DIFF_CHARS       (optional)  truncate the diff to this many chars, default 12000
    DRY_RUN              (optional)  "1" => print the comment + JSON instead of posting
    BLOCKING             (optional)  "1" => exit 1 when recommendation is do_not_merge
"""
from __future__ import annotations

import json
import os
import sys

import requests

GITHUB_API = "https://api.github.com"
MODELS_ENDPOINT = "https://models.github.ai/inference"
COMMENT_MARKER = "<!-- ai-pr-review -->"
DEFAULT_MODEL = "openai/gpt-4.1"
DEFAULT_BASE_BRANCH = "main"
DEFAULT_MAX_DIFF_CHARS = 12000
GITHUB_TIMEOUT = 30   # seconds, GitHub REST calls
MODEL_TIMEOUT = 120   # seconds, inference can be slower on large diffs
MAX_OUTPUT_TOKENS = 4000  # stays within the GitHub Models free-tier output cap

# Optional context files, read from the working directory if they happen to exist.
CI_RESULTS_FILE = "ci_results.txt"
REVIEW_RULES_FILE = ".github/ai-review-rules.md"

# ---------------------------------------------------------------------------
# Structured-output tool (OpenAI / GitHub Models "function" format). Forcing this
# single function call makes the model return schema-valid JSON; we read it off
# choices[0].message.tool_calls[0].function.arguments (a JSON string).
# ---------------------------------------------------------------------------
# Shared shape for a blocker/warning: every finding says how serious it is, what
# kind of issue it is, what's wrong (and the impact), and exactly how to fix it.
_FINDING_PROPS = {
    "file": {"type": "string", "description": "Path to the file as it appears in the diff."},
    "line": {"type": "string", "description": "Line or hunk reference, e.g. '42' or '40-55'."},
    "severity": {
        "type": "string",
        "enum": ["critical", "high", "medium", "low"],
        "description": "How serious this specific issue is.",
    },
    "category": {
        "type": "string",
        "enum": ["security", "auth", "data-loss", "crash", "api-contract",
                 "performance", "correctness", "quality", "tests", "other"],
        "description": "The kind of issue, for triage.",
    },
    "reason": {"type": "string", "description": "What is wrong AND the concrete impact if it ships."},
    "suggested_fix": {"type": "string", "description": "Specific, actionable fix; name the code and show a snippet when useful."},
}
_FINDING = {"type": "object", "properties": _FINDING_PROPS,
            "required": ["file", "line", "severity", "category", "reason", "suggested_fix"]}

_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string", "description": "One punchy sentence capturing the overall verdict."},
        "summary": {"type": "string", "description": "2-4 sentence overview of what the change does and its risk."},
        "risk_level": {
            "type": "string",
            "enum": ["none", "low", "medium", "high", "critical"],
            "description": "Overall risk of merging this diff as-is.",
        },
        "merge_recommendation": {
            "type": "string",
            "enum": ["merge", "merge_with_caution", "do_not_merge"],
            "description": "Is this ready to merge? do_not_merge if there is any blocker.",
        },
        "positives": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific things the PR does well. Be fair and concrete.",
        },
        "blockers": {
            "type": "array",
            "description": "Issues that should stop a merge. Empty if there are none.",
            "items": _FINDING,
        },
        "warnings": {
            "type": "array",
            "description": "Real but non-blocking concerns. Empty if there are none.",
            "items": _FINDING,
        },
        "nitpicks": {
            "type": "array",
            "description": "Minor, optional polish. Keep this rare and short.",
            "items": {
                "type": "object",
                "properties": {"file": {"type": "string"}, "note": {"type": "string"}},
                "required": ["file", "note"],
            },
        },
        "test_suggestions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Concrete tests worth adding for this change. Optional.",
        },
    },
    "required": ["headline", "summary", "risk_level", "merge_recommendation", "positives", "blockers", "warnings"],
}

SUBMIT_REVIEW_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_review",
        "description": (
            "Submit your structured review of this pull request. "
            "You MUST call this function exactly once and put the entire review in it."
        ),
        "parameters": _REVIEW_SCHEMA,
    },
}

SYSTEM_PROMPT = """You are a meticulous, fair senior software engineer reviewing a pull request.
Your job is to tell the author, clearly and kindly, whether this is safe to merge --
and if not, exactly WHAT to fix and HOW.

Review ONLY the unified diff you are given. Do not speculate about code you cannot \
see, and do not invent problems. A clean review is a good outcome, not a failure.

Be fair. In `positives`, call out what the PR genuinely does well (good naming, input \
validation, tests, clear structure). A review that only criticises is demoralising and \
trusted less.

Treat these as BLOCKERS (serious enough to stop a merge):
  - Security vulnerabilities (injection, SSRF, path traversal, unsafe deserialization, \
secrets committed to the repo, disabled signature/certificate checks, etc.).
  - Authentication / authorization / permission mistakes (missing access checks, \
privilege escalation, IDOR).
  - Data loss or destructive operations (unguarded deletes, dropping tables or columns, \
irreversible migrations).
  - Crashes and unhandled errors on realistic inputs.
  - Broken API contracts that existing callers rely on.

Use WARNINGS for genuine but non-blocking concerns; use NITPICKS sparingly.

For EVERY blocker and warning, fill in all fields:
  - `severity` and `category`, so the author can triage at a glance;
  - `reason` = what is wrong AND the concrete impact if it ships;
  - `suggested_fix` = specific, actionable steps -- name the function/line, describe the \
safer approach, and include a short code snippet when it helps.

Set `merge_recommendation` honestly: `do_not_merge` if there is ANY blocker; \
`merge_with_caution` if only warnings give you pause; `merge` if it is clean. Write a \
crisp one-line `headline` and a `summary`, and add `test_suggestions` when useful.

Prefer a few high-confidence findings over many speculative ones. You MUST respond by \
calling the `submit_review` function exactly once with schema-valid JSON, and write no \
prose outside the function call."""


# ---------------------------------------------------------------------------
# Pure helpers (no network, no token) -- these are what the tests exercise.
# ---------------------------------------------------------------------------
def read_optional_file(path):
    """Return the file's text, or ``None`` if it is missing/unreadable."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except (FileNotFoundError, OSError):
        return None


def truncate_diff(diff, max_chars):
    """Truncate ``diff`` to ``max_chars``.

    Returns ``(text, truncated)``. The ``truncated`` flag is ``False`` while the
    diff fits (including exactly at the limit) and flips to ``True`` only once the
    diff exceeds ``max_chars``.
    """
    if len(diff) <= max_chars:
        return diff, False
    return diff[:max_chars], True


def build_user_prompt(*, title, body, base_branch, diff, diff_truncated,
                      ci_results=None, review_rules=None):
    """Assemble the user-turn prompt with all available context for the review."""
    parts = []
    parts.append(f"You are reviewing a pull request that targets the `{base_branch}` branch.")
    parts.append("")
    parts.append("## PR title")
    parts.append((title or "").strip() or "(no title)")
    parts.append("")
    parts.append("## PR description")
    parts.append((body or "").strip() or "(no description provided)")
    parts.append("")

    if review_rules and review_rules.strip():
        parts.append("## Repository review rules (project-specific guidance -- apply these)")
        parts.append(review_rules.strip())
        parts.append("")

    if ci_results and ci_results.strip():
        parts.append("## CI / check results from this run")
        parts.append(ci_results.strip())
        parts.append("")

    if diff_truncated:
        parts.append(
            f"> NOTE: The diff below was truncated to the first {len(diff)} characters "
            "to stay within the review budget. Review only what is shown and do not "
            "assume anything about the omitted portion."
        )
        parts.append("")

    parts.append("## Unified diff to review")
    parts.append("```diff")
    parts.append(diff)
    parts.append("```")
    parts.append("")
    parts.append("Review the diff per your instructions, then call `submit_review`.")
    return "\n".join(parts)


def build_request_payload(model, system_prompt, user_prompt):
    """Build the OpenAI-compatible chat-completions body that forces submit_review."""
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "tools": [SUBMIT_REVIEW_TOOL],
        "tool_choice": {"type": "function", "function": {"name": "submit_review"}},
        "max_tokens": MAX_OUTPUT_TOKENS,
    }


def extract_review(response_json):
    """Pull the parsed submit_review arguments out of a chat-completions response.

    Raises RuntimeError with a helpful message if the model didn't make the call
    or returned non-JSON arguments.
    """
    choices = response_json.get("choices") or []
    if not choices:
        raise RuntimeError(f"Model response contained no choices: {json.dumps(response_json)[:300]}")
    message = choices[0].get("message") or {}
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        if fn.get("name") == "submit_review":
            try:
                return json.loads(fn.get("arguments") or "")
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"submit_review arguments were not valid JSON: {exc}") from exc
    raise RuntimeError(
        "Model did not return a submit_review function call "
        f"(finish_reason={choices[0].get('finish_reason')!r}, refusal={message.get('refusal')!r})."
    )


# Display maps for rendering.
_RISK_BADGE = {
    "none": "🟢 None", "low": "🟢 Low", "medium": "🟡 Medium",
    "high": "🟠 High", "critical": "🔴 Critical",
}
# merge_recommendation -> (emoji, banner verdict, table label, GitHub alert type)
_REC = {
    "merge": ("✅", "Ready to merge", "Merge", "TIP"),
    "merge_with_caution": ("⚠️", "Merge with caution", "Merge with caution", "WARNING"),
    "do_not_merge": ("⛔", "Not ready to merge", "Do not merge yet", "CAUTION"),
}
_SEV_BADGE = {
    "critical": "🔴 Critical", "high": "🟠 High", "medium": "🟡 Medium", "low": "🔵 Low",
}


def _format_location(file, line):
    """Render a ``file:line`` reference, tolerating missing pieces."""
    file = (file or "").strip()
    line = ("" if line is None else str(line)).strip()
    if file and line:
        return f"{file}:{line}"
    return file or "(general)"


def _blockquote_field(label, text):
    """Render ``> **label:** text`` as blockquote lines, keeping any multi-line
    text (e.g. a code snippet in a fix) inside the quoted card instead of letting
    it break out below the blockquote."""
    parts = (text or "").split("\n")
    lines = [f"> **{label}:** {parts[0]}".rstrip()]
    for p in parts[1:]:
        lines.append(f"> {p}".rstrip() if p.strip() else ">")
    return lines


def _render_finding(index, finding):
    """Render one blocker/warning as a tidy card: location, severity, what's wrong, how to fix."""
    loc = _format_location(finding.get("file"), finding.get("line"))
    sev = _SEV_BADGE.get((finding.get("severity") or "").strip(), (finding.get("severity") or "").strip())
    cat = (finding.get("category") or "").strip()
    meta = "  ·  ".join(x for x in [sev, f"`{cat}`" if cat else ""] if x)
    header = f"**{index}. `{loc}`**"
    out = [f"{header}  ·  {meta}" if meta else header, ""]
    reason = (finding.get("reason") or "").strip()
    fix = (finding.get("suggested_fix") or "").strip()
    if reason:
        out += _blockquote_field("🔍 What's wrong", reason)
    if fix:
        if reason:
            out.append(">")
        out += _blockquote_field("🛠️ How to fix", fix)
    out.append("")
    return out


def render_comment(review):
    """Render the review dict into one polished markdown comment body.

    Starts with the hidden ``COMMENT_MARKER`` (so the upsert can find and edit it),
    then a native GitHub alert banner with the merge verdict, an at-a-glance table,
    what the PR does well, then Blockers / Warnings as fix-cards. Nitpicks and test
    suggestions live in collapsed ``<details>``. Empty sections are omitted. Every
    field is read defensively so a minimal review dict still renders cleanly.
    """
    risk = (review.get("risk_level") or "unknown").strip()
    rec = (review.get("merge_recommendation") or "unknown").strip()
    headline = (review.get("headline") or "").strip()
    summary = (review.get("summary") or "").strip()
    positives = review.get("positives") or []
    blockers = review.get("blockers") or []
    warnings = review.get("warnings") or []
    nitpicks = review.get("nitpicks") or []
    tests = review.get("test_suggestions") or []

    emoji, verdict, rec_label, alert = _REC.get(rec, ("•", rec, rec, "NOTE"))

    lines = [COMMENT_MARKER, "# 🤖 AI Code Review", ""]

    # Verdict banner — a native GitHub alert (green tip / amber warning / red caution).
    banner = f"**{emoji} {verdict}**"
    if headline:
        banner += f" — {headline}"
    lines += [f"> [!{alert}]", f"> {banner}", ""]

    # At-a-glance table.
    lines += [
        "| | |",
        "|:--|:--|",
        f"| **Risk level** | {_RISK_BADGE.get(risk, risk)} |",
        f"| **Recommendation** | {emoji} {rec_label} |",
        f"| **Blockers** | {len(blockers)} |",
        f"| **Warnings** | {len(warnings)} |",
        "",
    ]

    if summary:
        lines += [summary, ""]

    if positives:
        lines += ["### ✅ What's done well", ""]
        lines += [f"- {p}" for p in positives]
        lines.append("")

    if blockers:
        lines += ["### ⛔ Blockers — must fix before merging", ""]
        for i, b in enumerate(blockers, 1):
            lines += _render_finding(i, b)

    if warnings:
        lines += ["### ⚠️ Warnings — worth addressing", ""]
        for i, w in enumerate(warnings, 1):
            lines += _render_finding(i, w)

    if nitpicks:
        lines += ["<details>", f"<summary>💡 Nitpicks ({len(nitpicks)})</summary>", ""]
        for n in nitpicks:
            f = (n.get("file") or "").strip()
            note = (n.get("note") or "").strip()
            lines.append(f"- **{f}** — {note}" if f else f"- {note}")
        lines += ["", "</details>", ""]

    if tests:
        lines += ["<details>", f"<summary>🧪 Suggested tests ({len(tests)})</summary>", ""]
        lines += [f"- {t}" for t in tests]
        lines += ["", "</details>", ""]

    lines.append("---")
    lines.append(
        "<sub>🤖 <b>Non-blocking AI review</b> — informational only. This bot never "
        "merges, closes, or approves anything; a human decides. Generated via GitHub Models.</sub>"
    )
    return "\n".join(lines)


def compute_exit_code(review, *, blocking):
    """V1 policy: always 0, unless BLOCKING is on and the review says do_not_merge."""
    if blocking and (review.get("merge_recommendation") == "do_not_merge"):
        return 1
    return 0


# ---------------------------------------------------------------------------
# GitHub REST I/O (plain requests, 30s timeouts, raise_for_status).
# ---------------------------------------------------------------------------
def _gh_headers(token, accept="application/vnd.github+json"):
    return {
        "Authorization": f"Bearer {token}",
        "Accept": accept,
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "ai-pr-review",
    }


def _check(resp, context):
    """Raise a clean RuntimeError (with GitHub's own error body) on HTTP >= 400.

    The body usually explains *why* (e.g. "Resource not accessible by personal
    access token" for a missing write permission), which beats a bare traceback.
    """
    if resp.status_code >= 400:
        raise RuntimeError(f"GitHub API error while {context}: HTTP {resp.status_code}: {resp.text[:400]}")
    return resp


def fetch_pr(repo, pr_number, token):
    """GET the PR object (title, body, etc.)."""
    url = f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}"
    resp = requests.get(url, headers=_gh_headers(token), timeout=GITHUB_TIMEOUT)
    _check(resp, f"fetching PR #{pr_number}")
    return resp.json()


def fetch_diff(repo, pr_number, token):
    """GET the unified diff for the PR (same endpoint, diff media type)."""
    url = f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}"
    resp = requests.get(
        url,
        headers=_gh_headers(token, accept="application/vnd.github.v3.diff"),
        timeout=GITHUB_TIMEOUT,
    )
    _check(resp, f"fetching diff for PR #{pr_number}")
    return resp.text


def list_issue_comments(repo, pr_number, token):
    """List all issue comments on the PR, following pagination."""
    comments = []
    url = f"{GITHUB_API}/repos/{repo}/issues/{pr_number}/comments"
    params = {"per_page": 100}
    while url:
        resp = requests.get(url, headers=_gh_headers(token), params=params, timeout=GITHUB_TIMEOUT)
        _check(resp, f"listing comments on PR #{pr_number}")
        comments.extend(resp.json())
        url = resp.links.get("next", {}).get("url")
        params = None  # the "next" URL already carries the cursor
    return comments


def upsert_comment(repo, pr_number, token, body):
    """Edit our existing marked comment if present, else post a new one.

    Returns ``(comment_json, action)`` where action is "updated" or "created".
    """
    existing = list_issue_comments(repo, pr_number, token)
    marked = next((c for c in existing if COMMENT_MARKER in (c.get("body") or "")), None)
    if marked:
        url = f"{GITHUB_API}/repos/{repo}/issues/comments/{marked['id']}"
        resp = requests.patch(url, headers=_gh_headers(token), json={"body": body}, timeout=GITHUB_TIMEOUT)
        _check(resp, f"updating comment {marked['id']}")
        return resp.json(), "updated"
    url = f"{GITHUB_API}/repos/{repo}/issues/{pr_number}/comments"
    resp = requests.post(url, headers=_gh_headers(token), json={"body": body}, timeout=GITHUB_TIMEOUT)
    _check(resp, f"posting a comment on PR #{pr_number}")
    return resp.json(), "created"


# ---------------------------------------------------------------------------
# GitHub Models inference (OpenAI-compatible chat completions via requests).
# ---------------------------------------------------------------------------
def call_model(*, token, endpoint, model, system_prompt, user_prompt):
    """Run the review and return the validated ``submit_review`` arguments dict."""
    url = endpoint.rstrip("/") + "/chat/completions"
    payload = build_request_payload(model, system_prompt, user_prompt)
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "ai-pr-review",
        },
        json=payload,
        timeout=MODEL_TIMEOUT,
    )
    if resp.status_code >= 400:
        # Surface the API's error body -- it's the most useful thing for debugging
        # auth/permission/rate-limit problems.
        raise RuntimeError(f"GitHub Models returned HTTP {resp.status_code}: {resp.text[:500]}")
    return extract_review(resp.json())


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------
def _die(message):
    print(f"ERROR: {message}", file=sys.stderr)
    return 1


def main():
    token = os.environ.get("GITHUB_TOKEN")
    models_token = os.environ.get("GITHUB_MODELS_TOKEN") or token
    repo = os.environ.get("REPO")
    pr_number = (os.environ.get("PR_NUMBER") or "").strip()
    base_branch = (os.environ.get("BASE_BRANCH") or "").strip() or DEFAULT_BASE_BRANCH
    model = (os.environ.get("MODEL") or "").strip() or DEFAULT_MODEL
    endpoint = (os.environ.get("MODELS_ENDPOINT") or "").strip() or MODELS_ENDPOINT
    dry_run = os.environ.get("DRY_RUN") == "1"
    blocking = os.environ.get("BLOCKING") == "1"

    try:
        max_diff_chars = int(os.environ.get("MAX_DIFF_CHARS") or DEFAULT_MAX_DIFF_CHARS)
    except ValueError:
        print(f"WARNING: MAX_DIFF_CHARS is not an integer; using {DEFAULT_MAX_DIFF_CHARS}.", file=sys.stderr)
        max_diff_chars = DEFAULT_MAX_DIFF_CHARS

    missing = [
        name for name, value in (
            ("GITHUB_TOKEN", token),
            ("REPO", repo),
            ("PR_NUMBER", pr_number),
        ) if not value
    ]
    if missing:
        return _die(f"Missing required environment variables: {', '.join(missing)}")

    print(f"Reviewing {repo} PR #{pr_number} (base: {base_branch}, model: {model})"
          + (" [DRY RUN]" if dry_run else ""))

    try:
        pr = fetch_pr(repo, pr_number, token)
        diff = fetch_diff(repo, pr_number, token)

        if not diff.strip():
            print("Diff is empty (nothing to review). Exiting 0.")
            return 0

        diff, truncated = truncate_diff(diff, max_diff_chars)
        if truncated:
            print(f"Diff truncated to {max_diff_chars} characters for the review.")

        user_prompt = build_user_prompt(
            title=pr.get("title"),
            body=pr.get("body"),
            base_branch=base_branch,
            diff=diff,
            diff_truncated=truncated,
            ci_results=read_optional_file(CI_RESULTS_FILE),
            review_rules=read_optional_file(REVIEW_RULES_FILE),
        )

        review = call_model(
            token=models_token,
            endpoint=endpoint,
            model=model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        comment = render_comment(review)

        if dry_run:
            print("\n===== DRY RUN: comment that WOULD be posted =====\n")
            print(comment)
            print("\n===== DRY RUN: raw review JSON =====\n")
            print(json.dumps(review, indent=2, ensure_ascii=False))
        else:
            _, action = upsert_comment(repo, pr_number, token, comment)
            print(f"Comment {action} on {repo} PR #{pr_number}.")
    except (requests.RequestException, RuntimeError) as exc:
        # Operational failure (bad/insufficient token, rate limit, model refusal,
        # network). Surface it clearly and exit non-zero so it's visible. This is
        # distinct from the review *recommendation*, which is governed by BLOCKING.
        return _die(str(exc))

    exit_code = compute_exit_code(review, blocking=blocking)
    if exit_code != 0:
        print("BLOCKING is on and the review recommends do_not_merge -> exiting 1.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
