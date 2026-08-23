# External expert escalation

Aeon can optionally ask one stronger hosted model for advice after its local
Qwen3.8 agent fails twice consecutively. This does not add another primary model:
Qwen continues to plan, use tools, inspect results, and decide what to do. The
hosted model receives one bounded text summary and returns advice without tool
or execution access.

The feature is off by default. On an interactive start, the model picker includes
**External expert account: configure/login (optional)**. That action can use the
official Codex, Claude Code, or Gemini CLI login already installed on the machine.
Aeon launches the provider's supported login command; it does not automate a web
session, scrape a consumer site, or store an OAuth token itself. A separate
OpenAI-compatible API configuration remains available for operators who prefer it.

## Subscription setup from the model picker

Start Aeon interactively and choose the external-account entry:

~~~bash
python3 -m aeon.main
~~~

The setup screen offers:

- **Codex / ChatGPT subscription** — uses `codex login --device-auth`, reads the
  installed CLI's current model catalog, and asks you to choose both an explicit
  model and one of that model's supported reasoning-effort levels. Advice runs in
  a read-only, ephemeral `codex exec` session. Aeon pipes the prompt over stdin,
  reads the dedicated final-message file, and parses JSONL only for completion and
  usage metadata, so progress output cannot be mistaken for advice.
- **Claude subscription** — uses `claude auth login --claudeai` and invokes
  `claude --print` in safe/plan mode with an empty tool set and no persistent
  session.
- **Gemini Google account** — launches the official `gemini` sign-in flow and uses
  headless JSON output in plan mode. Google AI Pro and Ultra accounts are supported
  by the Gemini CLI.
- **Disable** — removes the tool on subsequent Aeon worker construction while
  leaving the provider CLI's own login untouched.

The provider choice, exact model ID, and (for Codex) reasoning effort are stored in
`~/.aeon/external_expert/config.json` with mode 0600. Credentials stay in the
official CLI's credential store. The setup action returns to the local model
picker after it finishes; it never replaces Qwen as Aeon's control model.

Install the desired provider's official CLI before using its entry. The menu shows
whether each CLI is installed and, where the CLI exposes a status command, whether
it is signed in. Provider subscriptions, quotas, model availability, and terms
still apply.

## Guardrails

- The tool is absent unless enabled through the startup picker or
  `AEON_EXTERNAL_EXPERT_ENABLED=1`.
- Aeon calls the expert automatically after two consecutive failed local turns.
  Successful local progress resets the pair counter; consultation starts a fresh
  pair but is not itself counted as task progress.
- Defaults allow at most 3 calls per Aeon process, 10 calls per rolling 24 hours,
  and 80,000 reserved/actual tokens per rolling 24 hours.
- Each request is limited to 16,000 input characters. API-backed requests set a
  5,000-token generation cap. Subscription CLIs do not share one portable token-cap
  option, so Aeon truncates the returned advice before adding it to local context;
  the provider may still count the full generated response against its quota.
- The usage ledger stores timestamp/model/status/token counts, never prompts or
  answers.
- Common credential forms and secret-looking environment values are redacted.
- Before every provider call, the local Qwen model reviews the exact redacted
  outbound prompt. It blocks disclosure if a mainstream censored model may warn,
  refuse, moderate, or require sanitization; if the task needs an uncensored model;
  or if the prompt contains information the operator may not want a large technology
  company to receive. An ambiguous result, unavailable local reviewer, or review
  error also blocks disclosure. This gate is mandatory and has no remote fallback.
  A blocked review contacts no provider and consumes no external-call budget; Aeon
  continues with its local model.
- Automatic escalation receives the same bounded replanning evidence as the next
  local turn: objective, current plan, failed intent/actions, compressed attempt
  log, exact latest error/tool result, and retry count. It does not receive files,
  screenshots, credentials, or unrelated transcript content automatically.
- Private transmission is refused unless explicitly enabled.
- Advice is labeled untrusted and must be verified locally.
- The remote model receives no tools, files, browser session, or terminal access.

The usage ledger defaults to `~/.aeon/external_expert/usage.json`, with directory
mode 0700 and files mode 0600.

## API configuration

Create a mode-0600 environment file outside the repository, for example
`~/.config/aeon/external-expert.env`:

~~~bash
AEON_EXTERNAL_EXPERT_ENABLED=1
AEON_EXTERNAL_EXPERT_BACKEND=api
AEON_EXTERNAL_EXPERT_MODEL=YOUR_EXACT_MODEL_ID
AEON_EXTERNAL_EXPERT_BASE_URL=https://api.openai.com/v1
AEON_EXTERNAL_EXPERT_API_KEY_ENV=OPENAI_API_KEY
OPENAI_API_KEY=YOUR_KEY
~~~

Gemini exposes an official OpenAI-compatible API. For it, use:

~~~bash
AEON_EXTERNAL_EXPERT_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
AEON_EXTERNAL_EXPERT_API_KEY_ENV=GEMINI_API_KEY
GEMINI_API_KEY=YOUR_KEY
~~~

Set `AEON_EXTERNAL_EXPERT_MODEL` to a model available to that account. Claude can
be reached through an operator-approved OpenAI-compatible gateway by setting its
HTTPS base URL, model ID, API-key environment name, and key. Do not place keys in
the repository, terminal objective, web UI, or agent memories.

Load the file before starting Aeon:

~~~bash
set -a
. ~/.config/aeon/external-expert.env
set +a
python3 -m aeon.main
~~~

For Aeon Remote, the reviewed service template loads the same protected file so
new tmux agents inherit the configuration. The key never enters the browser.

Optional limits:

~~~text
AEON_EXTERNAL_EXPERT_MAX_CALLS_PER_RUN=3
AEON_EXTERNAL_EXPERT_MAX_CALLS_PER_DAY=10
AEON_EXTERNAL_EXPERT_MAX_TOKENS_PER_DAY=80000
AEON_EXTERNAL_EXPERT_MAX_INPUT_CHARS=16000
AEON_EXTERNAL_EXPERT_MAX_OUTPUT_TOKENS=5000
AEON_EXTERNAL_EXPERT_TIMEOUT_SECONDS=120
~~~

Environment variables override the saved picker configuration. To select a CLI
backend without the interactive picker, set
`AEON_EXTERNAL_EXPERT_BACKEND=codex`, `claude`, or `gemini`; the matching official
CLI must already be installed and authenticated. For Codex, also set an explicit
`AEON_EXTERNAL_EXPERT_MODEL` and `AEON_EXTERNAL_EXPERT_REASONING_EFFORT`; Aeon
passes the latter as Codex's `model_reasoning_effort` configuration. Do not put
provider OAuth tokens in Aeon's environment file.

Two deliberately dangerous overrides remain off unless explicitly set:

~~~text
AEON_EXTERNAL_EXPERT_ALLOW_EARLY=1     # permit calls before a detected stall
AEON_EXTERNAL_EXPERT_ALLOW_PRIVATE=1   # permit sensitivity='private'
~~~

Only HTTPS endpoints are accepted. Insecure HTTP requires a separate explicit
development override and should never be used for a public cloud provider.
