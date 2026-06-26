"""Offline unit tests for the pure functions in ``ai_pr_review.py``.

These tests do NO network I/O and need NO GitHub token -- they only exercise the
pure helpers: request building, response parsing, prompt building, diff
truncation, comment rendering, and the exit-code policy. The actual HTTP call
(``call_model``) is never touched here.

Run with:  cd scripts && python -m pytest test_ai_pr_review.py -v
"""
import os
import sys

# Import the sibling module by path so the suite runs from any working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ai_pr_review as air  # noqa: E402


# ---------------------------------------------------------------------------
# truncate_diff: the flag must flip exactly at the limit, not before.
# ---------------------------------------------------------------------------
def test_truncate_diff_under_limit_keeps_everything():
    diff = "a" * 50
    text, truncated = air.truncate_diff(diff, 100)
    assert text == diff
    assert truncated is False


def test_truncate_diff_exactly_at_limit_is_not_truncated():
    diff = "a" * 100
    text, truncated = air.truncate_diff(diff, 100)
    assert text == diff
    assert truncated is False


def test_truncate_diff_one_over_limit_flips_flag_and_cuts():
    diff = "a" * 101
    text, truncated = air.truncate_diff(diff, 100)
    assert truncated is True
    assert len(text) == 100
    assert text == "a" * 100


# ---------------------------------------------------------------------------
# build_request_payload: must force the submit_review function call.
# ---------------------------------------------------------------------------
def test_build_request_payload_forces_submit_review():
    payload = air.build_request_payload("openai/gpt-4.1-mini", "SYS", "USER")
    assert payload["model"] == "openai/gpt-4.1-mini"
    # system + user turns, in order
    assert payload["messages"][0] == {"role": "system", "content": "SYS"}
    assert payload["messages"][1] == {"role": "user", "content": "USER"}
    # exactly one tool, named submit_review, and tool_choice forces it
    assert payload["tools"][0]["function"]["name"] == "submit_review"
    assert payload["tool_choice"] == {"type": "function", "function": {"name": "submit_review"}}


# ---------------------------------------------------------------------------
# extract_review: parse the tool-call arguments out of an OpenAI-style response.
# ---------------------------------------------------------------------------
def _response_with_arguments(arguments_str):
    return {
        "choices": [{
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "type": "function",
                    "function": {"name": "submit_review", "arguments": arguments_str},
                }],
            },
        }]
    }


def test_extract_review_parses_tool_arguments():
    review = air.extract_review(_response_with_arguments('{"risk_level": "low", "blockers": []}'))
    assert review["risk_level"] == "low"
    assert review["blockers"] == []


def test_extract_review_raises_when_no_tool_call():
    no_call = {"choices": [{"finish_reason": "stop", "message": {"content": "hi", "tool_calls": []}}]}
    try:
        air.extract_review(no_call)
        assert False, "expected RuntimeError when there is no submit_review call"
    except RuntimeError:
        pass


def test_extract_review_raises_on_bad_json_arguments():
    try:
        air.extract_review(_response_with_arguments("{not valid json"))
        assert False, "expected RuntimeError on non-JSON arguments"
    except RuntimeError:
        pass


# ---------------------------------------------------------------------------
# build_user_prompt: must fold in base branch, title, rules, CI results, diff.
# ---------------------------------------------------------------------------
def test_build_user_prompt_includes_all_context():
    prompt = air.build_user_prompt(
        title="Add login throttling",
        body="Caps failed login attempts.",
        base_branch="dev",
        diff="--- a/auth.py\n+++ b/auth.py\n+limit = 5",
        diff_truncated=False,
        ci_results="3 passed, 0 failed",
        review_rules="Never log secrets or tokens.",
    )
    assert "dev" in prompt                       # base branch
    assert "Add login throttling" in prompt      # title
    assert "Caps failed login attempts." in prompt   # body
    assert "Never log secrets or tokens." in prompt  # review rules
    assert "3 passed, 0 failed" in prompt        # CI results
    assert "```diff" in prompt                   # fenced diff block
    assert "--- a/auth.py" in prompt             # the diff body itself


def test_build_user_prompt_omits_optional_sections_when_absent():
    prompt = air.build_user_prompt(
        title="t", body="", base_branch="main",
        diff="x", diff_truncated=False,
    )
    # No CI results / rules provided => those headers should not appear.
    assert "CI / check results" not in prompt
    assert "Repository review rules" not in prompt
    assert "```diff" in prompt


def test_build_user_prompt_notes_truncation_only_when_truncated():
    truncated = air.build_user_prompt(
        title="t", body="", base_branch="main",
        diff="x" * 10, diff_truncated=True,
    )
    not_truncated = air.build_user_prompt(
        title="t", body="", base_branch="main",
        diff="x" * 10, diff_truncated=False,
    )
    assert "truncat" in truncated.lower()
    assert "truncat" not in not_truncated.lower()


# ---------------------------------------------------------------------------
# Test-awareness: file parsing, test-file detection, symbol detection, and the
# "Test coverage context" section the model reads. All offline, no network.
# ---------------------------------------------------------------------------
# A small but realistic two-file diff: one production file, one test file, plus a
# delete and a couple of declarations to exercise the symbol heuristics.
_SAMPLE_DIFF = """diff --git a/app/auth/login.py b/app/auth/login.py
index 1111111..2222222 100644
--- a/app/auth/login.py
+++ b/app/auth/login.py
@@ -1,3 +1,8 @@
+def authenticate(user, password):
+    return check(user, password)
+
+class SessionManager:
+    pass
diff --git a/tests/test_login.py b/tests/test_login.py
index 0000000..3333333 100644
--- /dev/null
+++ b/tests/test_login.py
@@ -0,0 +1,3 @@
+def test_authenticate():
+    assert authenticate("a", "b")
diff --git a/web/widget.ts b/web/widget.ts
index 4444444..5555555 100644
--- a/web/widget.ts
+++ b/web/widget.ts
@@ -1,2 +1,4 @@
+export function renderWidget() {}
+const helper = () => 1;
diff --git a/old/legacy.py b/old/legacy.py
deleted file mode 100644
index 6666666..0000000
--- a/old/legacy.py
+++ /dev/null
@@ -1,2 +0,0 @@
-def gone():
-    pass
"""


def test_parse_changed_files_extracts_paths_and_skips_dev_null():
    files = air.parse_changed_files(_SAMPLE_DIFF)
    assert "app/auth/login.py" in files
    assert "tests/test_login.py" in files
    assert "web/widget.ts" in files
    assert "old/legacy.py" in files            # deleted file: new path comes from the git header
    assert "/dev/null" not in files            # never the placeholder path
    assert len(files) == len(set(files))       # de-duplicated


def test_is_test_file_recognises_common_conventions():
    assert air.is_test_file("tests/test_login.py")      # tests/ dir
    assert air.is_test_file("pkg/__tests__/foo.js")     # __tests__/ dir
    assert air.is_test_file("test_thing.py")            # test_*.py
    assert air.is_test_file("thing_test.py")            # *_test.py
    assert air.is_test_file("a/b/Button.test.tsx")      # *.test.tsx
    assert air.is_test_file("a/b/Button.spec.ts")       # *.spec.ts
    assert air.is_test_file("a/widget.test.js")         # *.test.js


def test_is_test_file_rejects_plain_source_files():
    assert not air.is_test_file("app/auth/login.py")
    assert not air.is_test_file("web/widget.ts")
    assert not air.is_test_file("README.md")
    assert not air.is_test_file("")


def test_classify_changed_files_splits_test_and_source():
    test_files, source_files = air.classify_changed_files(_SAMPLE_DIFF)
    assert "tests/test_login.py" in test_files
    assert "app/auth/login.py" in source_files
    assert "web/widget.ts" in source_files
    assert "tests/test_login.py" not in source_files


def test_detect_changed_symbols_finds_py_and_ts_declarations():
    symbols = air.detect_changed_symbols(_SAMPLE_DIFF)
    assert "authenticate" in symbols      # python def
    assert "SessionManager" in symbols    # python class
    assert "renderWidget" in symbols      # ts export function
    assert "helper" in symbols            # ts const arrow


def test_build_test_coverage_context_lists_files_and_reminds():
    ctx = air.build_test_coverage_context(_SAMPLE_DIFF)
    assert "Test coverage context" in ctx
    assert "app/auth/login.py" in ctx        # production file listed
    assert "tests/test_login.py" in ctx      # test file listed
    assert "authenticate" in ctx             # changed symbol surfaced
    assert "test_suggestions" in ctx         # reminder points at the schema field


def test_build_test_coverage_context_flags_when_no_tests_changed():
    src_only = (
        "diff --git a/app/pay.py b/app/pay.py\n"
        "--- a/app/pay.py\n+++ b/app/pay.py\n"
        "@@ -1 +1,2 @@\n+def charge(card):\n+    pass\n"
    )
    ctx = air.build_test_coverage_context(src_only)
    assert "Test files changed: (none)" in ctx
    # The no-tests note must be explicit that this is NOT an automatic blocker.
    assert "not" in ctx.lower() and "blocker" in ctx.lower()


def test_build_user_prompt_includes_test_coverage_context():
    ctx = air.build_test_coverage_context(_SAMPLE_DIFF)
    prompt = air.build_user_prompt(
        title="Add auth", body="", base_branch="main",
        diff="--- a/app/auth/login.py\n+++ b/app/auth/login.py\n+def authenticate(): ...",
        diff_truncated=False,
        test_coverage_context=ctx,
    )
    assert "Test coverage context" in prompt   # the section is folded into the prompt
    assert "tests/test_login.py" in prompt      # with its detected test file


# ---------------------------------------------------------------------------
# render_comment: clean PR vs. a PR with a blocker.
# ---------------------------------------------------------------------------
def _clean_review():
    return {
        "headline": "Small, safe refactor — good to go.",
        "summary": "Small, self-contained refactor with no risky changes.",
        "risk_level": "low",
        "merge_recommendation": "merge",
        "positives": ["Clear naming", "Inputs are validated"],
        "blockers": [],
        "warnings": [],
        "nitpicks": [],
        "test_suggestions": [],
    }


def _blocking_review():
    return {
        "headline": "Missing auth on a destructive endpoint — do not merge.",
        "summary": "Adds a delete endpoint but skips the permission check.",
        "risk_level": "high",
        "merge_recommendation": "do_not_merge",
        "positives": ["Endpoint is small and readable"],
        "blockers": [
            {
                "file": "app/api/orders.py",
                "line": "42",
                "severity": "critical",
                "category": "auth",
                "reason": "No authorization check before deleting the order.",
                "suggested_fix": "Add require_role('admin') before the delete call.",
            }
        ],
        "warnings": [],
        "nitpicks": [],
        "test_suggestions": ["Add a test that a non-admin caller gets 403."],
    }


def test_render_comment_clean_pr_has_no_blockers_section():
    comment = air.render_comment(_clean_review())
    assert air.COMMENT_MARKER in comment              # hidden upsert marker present
    assert "Ready to merge" in comment                # green banner verdict
    assert "must fix before merging" not in comment   # blockers section omitted
    assert "What's done well" in comment              # positives are rendered
    assert "a human decides" in comment.lower()       # footer disclaimer


def test_render_comment_with_blocker_shows_location_fix_and_verdict():
    comment = air.render_comment(_blocking_review())
    assert "### ⛔ Blockers — must fix before merging" in comment           # section header
    assert "app/api/orders.py:42" in comment                               # file:line reference
    assert "Add require_role('admin') before the delete call." in comment  # the fix text
    assert "How to fix" in comment                                         # actionable label
    assert "Not ready to merge" in comment                                 # banner verdict
    assert "Do not merge yet" in comment                                   # table label


def test_render_comment_shows_severity_and_category():
    comment = air.render_comment(_blocking_review())
    assert "Critical" in comment   # severity badge on the finding
    assert "auth" in comment       # category on the finding


def test_render_finding_keeps_multiline_fix_inside_the_card():
    review = _blocking_review()
    review["blockers"][0]["suggested_fix"] = "Use a parameterized query:\ndb.execute(sql, (uid,))"
    comment = air.render_comment(review)
    # the code line must stay quoted (prefixed with '>'), not break out of the card
    assert "> db.execute(sql, (uid,))" in comment


def test_render_comment_nitpicks_go_in_collapsed_details():
    review = _clean_review()
    review["nitpicks"] = [{"file": "utils.py", "note": "Prefer a constant here."}]
    comment = air.render_comment(review)
    assert "<details>" in comment
    assert "Prefer a constant here." in comment


# ---------------------------------------------------------------------------
# compute_exit_code: non-blocking by default; only BLOCKING + do_not_merge fails.
# ---------------------------------------------------------------------------
def test_exit_code_is_zero_by_default_even_for_do_not_merge():
    assert air.compute_exit_code(_blocking_review(), blocking=False) == 0


def test_exit_code_is_zero_when_blocking_but_recommendation_is_merge():
    assert air.compute_exit_code(_clean_review(), blocking=True) == 0


def test_exit_code_is_one_when_blocking_and_do_not_merge():
    assert air.compute_exit_code(_blocking_review(), blocking=True) == 1


# ---------------------------------------------------------------------------
# upsert_comment: post exactly once, then edit that same comment in place.
# This is the "one comment, never spam" guarantee. We stub the HTTP layer so the
# test stays fully offline and deterministic (no network, no token).
# ---------------------------------------------------------------------------
class _FakeResp:
    def __init__(self, status=200, json_data=None):
        self.status_code = status
        self._json = json_data
        self.text = ""
        self.links = {}

    def json(self):
        return self._json


class _FakeRequests:
    """Minimal stand-in for the ``requests`` module the script imports.

    Models a real PR comment list: a POST appends a comment (so a subsequent GET
    sees it), and a PATCH edits in place. Records every call for assertions.
    """
    def __init__(self):
        self.calls = []
        self._comments = []
        self._next_id = 100

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append(("GET", url))
        return _FakeResp(200, list(self._comments))

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append(("POST", url))
        self._next_id += 1
        created = {"id": self._next_id, "body": json["body"]}
        self._comments.append(created)
        return _FakeResp(201, created)

    def patch(self, url, headers=None, json=None, timeout=None):
        self.calls.append(("PATCH", url))
        # edit the existing marked comment in place
        for c in self._comments:
            if air.COMMENT_MARKER in c["body"]:
                c["body"] = json["body"]
                return _FakeResp(200, c)
        raise AssertionError("PATCH with no existing comment to edit")


def test_upsert_creates_first_then_updates_in_place(monkeypatch):
    fake = _FakeRequests()
    monkeypatch.setattr(air, "requests", fake)

    # First push: no marked comment exists yet -> POST (created).
    created, action1 = air.upsert_comment("owner/repo", "1", "tok",
                                          air.COMMENT_MARKER + "\nfirst review")
    assert action1 == "created"

    # Second push: the marker is now present -> PATCH the same comment (updated).
    updated, action2 = air.upsert_comment("owner/repo", "1", "tok",
                                          air.COMMENT_MARKER + "\nsecond review")
    assert action2 == "updated"

    methods = [c[0] for c in fake.calls]
    assert methods.count("POST") == 1            # posted exactly once...
    assert methods.count("PATCH") == 1           # ...then edited, not re-posted
    assert created["id"] == updated["id"]        # same comment object
    assert len(fake._comments) == 1              # never a duplicate
    assert "second review" in fake._comments[0]["body"]
