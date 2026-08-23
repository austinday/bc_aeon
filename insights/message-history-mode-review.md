# Message-history mode (AEON_MESSAGE_HISTORY=1): three latent defects

Review of the newest feature (commit 97a600d), which replaces the single giant
per-turn prompt with system + turn history + volatile current-state message.
The design is sound (bounded 45% budget, compact turns, cache-friendly split),
but three structural issues will surface once someone actually flips it on.

## 1. Consecutive `user` messages every turn

`_append_history_turn` ends every turn's history with a **user** message (the
brief result). `call_messages` (worker.py:2012-2014) then appends the
current-state message, also role **user** — so every turn after the first sends
`..., assistant, user, user`. `get_primary_agent_response` does no role
merging (llm.py:805 just appends).

- Qwen-family chat templates tolerate this (each message rendered
  independently) — so it happens to work with the current flagship models.
- Strict-alternation templates (Gemma-lineage, Mistral) raise
  `Conversation roles must alternate` inside the server's template engine →
  every primary call fails with an opaque 400. Since the model menu includes
  Gemma-class builds, this is a real config-dependent hard failure.

**Fix:** merge the trailing history user message into the current-state
message, or emit the turn result as part of the *next* current-state message
instead of a separate history entry. Either preserves caching.

## 2. Trimming can cut mid-pair

`_trim_history` (worker.py:1620-1626) drops whole messages oldest-first with no
awareness that history is (assistant, user) pairs. It can keep a user result
whose assistant decision was dropped — an orphaned "result" of nothing —
and produce another consecutive-user seam at the trim boundary (marker is also
a user message). **Fix:** trim in steps of 2 from the oldest pair boundary.

## 3. Trim markers can accumulate

Each trim prepends `[earlier turns trimmed...]` (worker.py:1629). On the next
trim, if the old marker survives the budget but other messages are dropped, a
second marker is prepended. Long sessions collect markers. **Fix:** strip
existing markers before re-inserting one.

## Also worth knowing (not a bug)

- The mode changes *epistemics*, not just latency: the default single-prompt
  mode is stateless-per-turn (the model re-reads a curated state each turn),
  while history mode gives the model its own prior thoughts verbatim. Prior
  work in this repo (loop-guard, confabulated-completion) found the agent
  over-trusts its own narrative; a verbatim thought-history may *amplify*
  narrative lock-in vs. the current design where stale thoughts vanish and
  ground truth (files, attempt log) is re-presented fresh. Worth an A/B on a
  looping-prone task before making it default: measure stuck-banner rate and
  fabrication incidents, not just TTFT.
- History seeds from `action_log` on resume but `action_log` is also still
  rendered nowhere in this mode except that seed — the attempt-log section is
  absent from `_build_current_state_message`. In long sessions the trimmed
  history plus no attempt log means older intents are simply gone. Consider
  including the compressed attempt log in the current-state message (it exists
  already for single-prompt mode).

## Suggested verification once fixed

Run the same 30-iteration task twice (default vs AEON_MESSAGE_HISTORY=1) on the
local server and compare: per-turn TTFT (the motivation), tokens/turn growth,
stuck-banner count, and whether any turn fails with a template error.
