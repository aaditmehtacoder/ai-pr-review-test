# AI PR Review (GitHub Models)

A tiny GitHub Action that asks an AI model to review every pull request and posts
**one** structured comment. Repeated pushes edit that same comment instead of
spamming new ones.

It is "just files in a repo" — no server, no GitHub App, no bot account, and
**no separate API key**. It runs on [GitHub Models](https://github.com/marketplace?type=models)
(free, rate-limited), and the built-in `GITHUB_TOKEN` authenticates *both* the
GitHub REST calls and the model inference.

**What it will and won't do**

- ✅ Reads the PR diff, posts a single review comment (risk level, summary,
  blockers, warnings, nitpicks, merge recommendation).
- ✅ Updates that one comment in place on every new push.
- ✅ Non-blocking by default — it never fails your PR check (see
  [Blocking mode](#blocking-mode)).
- 🚫 **Never** merges, closes, approves, or pushes. It only comments.

---

## How it works

1. The workflow triggers on a PR (or a manual dispatch).
2. The script fetches the PR's title/body and unified diff via the GitHub REST API.
3. It sends the diff to a GitHub Models chat-completions endpoint, **forcing** a
   single `submit_review` function call so the model must return schema-valid JSON.
4. It renders that JSON into one markdown comment and **upserts** it: the comment
   carries a hidden marker (`<!-- ai-pr-review -->`); if a marked comment already
   exists it's `PATCH`ed, otherwise a new one is `POST`ed. That's why new pushes
   edit one comment instead of piling up.

---

## 10-minute quickstart (try it on your own repo)

1. **Copy these files into your repo**, keeping the paths:

   ```
   .github/workflows/ai-pr-review.yml
   scripts/ai_pr_review.py
   scripts/test_ai_pr_review.py
   requirements.txt
   ```

   (The reviewer only ever touches `.github/` and `scripts/` plus
   `requirements.txt` / `README.md`. It never modifies application code.)

2. **No secret to add.** The workflow grants `permissions: models: read`, which
   lets the auto-injected `GITHUB_TOKEN` call GitHub Models. You don't create or
   paste any API key.

   > If your org/enterprise restricts `models: read` for the built-in token,
   > create a fine-grained PAT with the **Models** permission, save it as a repo
   > secret named `MODELS_TOKEN`, and uncomment the `GITHUB_MODELS_TOKEN` line in
   > the workflow.

3. **Commit and push** the files to your default branch.

4. **Open a test PR.** Make a small change on a branch and open a pull request.
   Within a minute the **AI PR Review** workflow runs and posts a comment. Push
   another commit — the comment updates in place rather than adding a second one.

Default model: **`openai/gpt-4.1`** (a strong general model, free on GitHub
Models, ~50 reviews/day). Reviewing a lot of PRs? Set `MODEL` to
`openai/gpt-4.1-mini` for ~150/day. See [Choosing a model](#choosing-a-model).

> **Tip:** the first time, do a [dry run](#dry-run-read-the-output-before-anything-posts)
> so you can read the model's output locally before anything is ever posted.

---

## Dry run (read the output before anything posts)

`DRY_RUN=1` does everything *except* post — it prints the exact comment it would
make plus the raw JSON, to your terminal. Run it locally against any real PR
(open **or** closed; the diff endpoint works for both).

You need a **GitHub PAT** with two permissions on the repo:
- **Models** (account permission) — to call inference.
- **Pull requests: Read** (or Contents: Read) — to fetch the PR + diff.

A dry run never posts, so **read-only is enough** — no write permission needed.

```bash
pip install -r requirements.txt

GITHUB_TOKEN="github_pat_…" \
REPO="your-org/your-repo" \
PR_NUMBER="1" \
DRY_RUN="1" \
MODEL="openai/gpt-4.1-mini" \
python scripts/ai_pr_review.py
```

You'll see the rendered markdown comment and the structured JSON. Nothing is
posted. When you're happy with it, let the workflow run for real on a PR (there,
the built-in token has write permission and actually posts).

> To do a **live local post** (not just a dry run), your PAT additionally needs
> **Pull requests: Write**. Most people don't bother — they let the Action handle
> real posting and use dry runs locally.

---

## Environment variables

The workflow sets these for you; you set them by hand only for local/dry runs.

| Variable              | Required | Default                      | Purpose |
|-----------------------|----------|------------------------------|---------|
| `GITHUB_TOKEN`        | yes      | —                            | GitHub auth for REST and (by default) inference. In Actions this is the auto-injected token; locally use a PAT. |
| `GITHUB_MODELS_TOKEN` | no       | falls back to `GITHUB_TOKEN` | Separate token for inference, if you don't want the built-in token doing it. |
| `REPO`                | yes      | —                            | `owner/repo`. The workflow passes `${{ github.repository }}`. |
| `PR_NUMBER`           | yes      | —                            | PR number to review. |
| `BASE_BRANCH`         | no       | `main`                       | Target branch, used as context in the prompt. |
| `MODEL`               | no       | `openai/gpt-4.1`             | Any GitHub Models id (see below). |
| `MODELS_ENDPOINT`     | no       | `https://models.github.ai/inference` | Inference base URL (override for GHES/proxy). |
| `MAX_DIFF_CHARS`      | no       | `12000`                      | Truncate the diff to this many chars; the prompt notes when it was cut. |
| `DRY_RUN`             | no       | off                          | `1` → print the comment + JSON instead of posting. |
| `BLOCKING`            | no       | off                          | `1` → exit 1 (fail the check) when the recommendation is `do_not_merge`. |

### Optional context files

If these exist in the repo, the script folds them into the prompt. Both optional.

- **`ci_results.txt`** — write your lint/test output here (the workflow has a
  commented-out placeholder step) and the model factors it into the review.
- **`.github/ai-review-rules.md`** — project-specific review guidance (house
  style, "always check X", things to ignore). Free-form markdown, passed verbatim.

---

## Choosing a model

GitHub Models is free with per-model rate limits grouped into tiers. **Low-tier**
models have the most generous free limits (good for a per-PR reviewer);
**high/custom** tiers are stronger but more rate-limited. All of these support
the forced function call this script relies on:

| Model id                  | Tier | Notes |
|---------------------------|------|-------|
| `openai/gpt-4.1`          | high | **default** — strongest general reviewer; ~50 reviews/day free |
| `openai/gpt-4.1-mini`     | low  | rate-limit-friendly (~150/day); great for high PR volume |
| `openai/gpt-4o-mini`      | low  | smaller/older, also fine |
| `openai/gpt-4o`           | high | higher quality, tighter limits |

Set `MODEL` to switch. (Reasoning models like `openai/o1`/`o3`/`gpt-5` use
different request parameters and aren't drop-in here.) Browse the full catalog at
<https://github.com/marketplace?type=models>.

---

## Blocking mode

This workflow ships with **`BLOCKING: "1"`**: when the review recommends
`do_not_merge`, the script posts its comment **and then exits `1`**, so the check
goes **red ✗** on risky PRs. Set `BLOCKING: "0"` to make it purely advisory — the
check stays green and the verdict lives only in the comment.

A red check is a strong signal, but it does not *prevent* merging on its own. To
actually block the merge button, add a branch-protection rule requiring the
**AI PR Review** check under Settings → Branches → Require status checks to pass.

Separately, genuine **operational failures** (bad/insufficient token, rate limit,
the model refusing to call the tool, network errors) print a clear `ERROR:` line
and exit non-zero so you notice the action is broken — this is distinct from the
review's recommendation. With no branch protection on the check, a red mark is
informational and doesn't actually block merging. The bot never merges or closes
anything in either mode.

---

## Running the tests

The tests are fully offline — no network, no token. They cover the request
builder, the response parser, prompt building, diff truncation, comment
rendering, the exit-code policy, and the upsert's "post once, then edit in place"
guarantee (via a stubbed HTTP layer).

```bash
cd scripts && python -m pytest test_ai_pr_review.py -v
```

---

## Rolling out to a team repo later

These same files drop into a shared repo unchanged. A few things to know:

- **Nothing to copy but the files** — with `models: read` and the built-in token,
  there's no secret to provision (unless your org restricts it; then add a
  `MODELS_TOKEN` secret as noted above).
- **Fork PRs run without secrets — by GitHub's design.** For PRs opened from a
  **fork**, GitHub does not expose repo secrets and downgrades `GITHUB_TOKEN` to
  read-only (so a malicious fork can't steal credentials or post as you). On fork
  PRs this action therefore can't post a comment and will fail quietly rather than
  review. For an internal team repo where everyone pushes branches to the *same*
  repo, every PR gets reviewed normally. Fork coverage requires a
  `pull_request_target` workflow, which has real security trade-offs and is
  intentionally **not** set up here.
- **Mind the rate limits.** GitHub Models free-tier limits are per-account/per-org
  and shared across everything using them. A busy repo on a high-tier model can
  hit daily caps; the default low-tier model has the most headroom.
- **Keep it non-blocking at first.** Let the team see the comments for a while
  before considering `BLOCKING` mode, so the bot earns trust instead of getting in
  the way.
- **Tune with `.github/ai-review-rules.md`** to encode team conventions so reviews
  match how your team actually works.
