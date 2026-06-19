# 🎬 Video Script — "I built an AI that reviews my pull requests"

**Format:** ~2.5–3 minute explainer / demo
**Tone:** confident, friendly, a little wow
**On-screen:** screen recording of GitHub + terminal, with the narration as voiceover

---

## SCENE 1 — The hook (0:00–0:15)

**[VISUAL]** A GitHub pull request opens. A second later, a bot comment appears with a big red banner: **⛔ Not ready to merge.**

**[NARRATION]**
> "Every time I open a pull request, an AI reads my code and tells me — in plain English — whether it's safe to merge. It catches security holes, explains exactly what's wrong, and shows me how to fix it. And it's just a few files in my repo. No server. No paid API. Let me show you."

---

## SCENE 2 — What it is (0:15–0:40)

**[VISUAL]** File tree of the repo: `.github/workflows/ai-pr-review.yml`, `scripts/ai_pr_review.py`, `requirements.txt`, `README.md`.

**[NARRATION]**
> "It's a GitHub Action. When a pull request opens or updates, GitHub runs a short Python script. That script grabs the diff, sends it to an AI model through GitHub Models — which is free — and posts one tidy review comment. The whole thing is about 300 lines of Python and one dependency: `requests`. No framework, no bot account, no API key to paste anywhere."

---

## SCENE 3 — How it works, step by step (0:40–1:20)

**[VISUAL]** Simple animated flow: `PR opened → Action runs → fetch diff → ask the AI → post one comment`. Highlight each box as it's described.

**[NARRATION]**
> "Here's the flow. One — you open a PR, and the `pull_request` event triggers the workflow. Two — the script calls GitHub's REST API to pull the PR's title, description, and unified diff. Three — it sends that to the model with a *forced tool call*: the AI is required to answer in strict JSON — a risk level, a merge recommendation, blockers, warnings, and how to fix each one. That's the trick that makes the output reliable instead of rambling. Four — the script renders that JSON into a beautiful Markdown comment. And five — it 'upserts': it hides a marker in the comment, so every new push *edits the same comment* instead of spamming new ones."

**[VISUAL]** Zoom in on `scripts/ai_pr_review.py` — highlight `SUBMIT_REVIEW_TOOL`, `tool_choice={"type":"function","function":{"name":"submit_review"}}`, and `render_comment`.

**[NARRATION]**
> "The forced function call is right here. We hand the model one tool, `submit_review`, with a JSON schema, and we *require* it to call that tool. So we always get back valid, structured data — never a wall of text we have to parse."

---

## SCENE 4 — The test scenario (1:20–2:10)

**[VISUAL]** Terminal: run the offline tests — `18 passed`. Then GitHub: two pull requests side by side.

**[NARRATION]**
> "How do I know it actually works? Two ways. First, 18 offline unit tests cover the logic — building the request, parsing the model's answer, rendering the comment, and the 'one comment, never spam' rule. They run in under a second with no network and no token."

**[VISUAL]** Open **PR #1 (clean code)** → bot comment: 🟢 **Ready to merge**.

**[NARRATION]**
> "Second, a real test on real PRs. This one adds a small, well-validated helper. The AI reads it and says: low risk, ready to merge — and even lists what the code does well."

**[VISUAL]** Open **PR #3 (JWT auth)** → bot comment: 🔴 **Not ready to merge**, two blockers.

**[NARRATION]**
> "This one adds JWT login code. The AI flags two blockers it had to *understand* to catch: the signature verification is switched off — so anyone could forge a token — and the secret key is hardcoded. For each one it tells me what's wrong, the impact, and the exact fix, with a code snippet. No keyword-matcher can do that — that's the model reasoning about the code."

**[VISUAL]** Push a fix to that branch; the SAME comment updates and flips to 🟢 **Ready to merge.**

**[NARRATION]**
> "And when I push a fix? The same comment updates in place — and the verdict flips to green. It re-read the new code and changed its mind."

---

## SCENE 5 — The beautiful output (2:10–2:35)

**[VISUAL]** Slow scroll through the rich comment: the colored alert banner, the at-a-glance table, ✅ What's done well, the ⛔ blocker cards with 🔍 What's wrong / 🛠️ How to fix, and the collapsed 🧪 Suggested tests.

**[NARRATION]**
> "And the review itself is genuinely nice to read: a color-coded verdict banner, a summary table, what you did well, every issue as a card with the problem and the fix, and suggested tests tucked away in a dropdown. It reads like a thoughtful teammate, not a linter."

---

## SCENE 6 — Outro (2:35–2:50)

**[VISUAL]** Back to the file tree; cursor highlights the four files. Text on screen: *"Copy 4 files. Open a PR. That's it."*

**[NARRATION]**
> "The best part: to add it to any repo, you copy four files and open a pull request. It never merges anything itself — it just gives you a clear, honest second opinion. Links and the full code are below. Thanks for watching."

---

## 🧪 Test scenario & commands (for the recording, or to try yourself)

### A. Offline tests (no network, no token)
```bash
cd scripts && python -m pytest test_ai_pr_review.py -v
# Expect: 18 passed
```

### B. Dry run on a real PR (reads + calls the AI, posts nothing)
Needs a GitHub token with the **Models** permission and **Pull requests: Read**.
```bash
pip install -r requirements.txt

GITHUB_TOKEN="github_pat_…" \
REPO="aaditmehtacoder/ai-pr-review-test" \
PR_NUMBER="3" \
MODEL="openai/gpt-4.1" \
DRY_RUN="1" \
python scripts/ai_pr_review.py
# Prints the exact comment + the raw JSON. Nothing is posted.
```

### C. The full live test (the Action posts for real)
1. Push these files to a repo's `main`.
2. Open a PR with clean code  → expect 🟢 **Ready to merge**.
3. Open a PR with this vulnerable snippet → expect 🔴 **Not ready to merge**:
   ```python
   import jwt
   def get_current_user(request):
       raw = request.headers.get("Authorization", "").replace("Bearer ", "")
       payload = jwt.decode(raw, options={"verify_signature": False})  # forgeable!
       return load_user(payload["sub"])
   ```
4. Push a fix to that branch → the **same** comment updates and flips to green.

### Two contrasting PRs that already exist in the test repo
- Clean / ready to merge: https://github.com/aaditmehtacoder/ai-pr-review-test/pull/1
- JWT auth / not ready:   https://github.com/aaditmehtacoder/ai-pr-review-test/pull/3
