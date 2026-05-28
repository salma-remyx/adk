---
name: poly-adk
description: Use the Poly CLI to manage PolyAI voice agents — provides context on commands, resource formats, and workflows. Use when building features or editing agent config.
allowed-tools: Read Grep Glob Bash Edit Write
---

# Poly ADK

Poly-ADK is a CLI tool for managing PolyAI Agent Studio projects locally. It provides a Git-like workflow for syncing agent configurations between your local filesystem and the Agent Studio platform.

Each project defines an AI voice or webchat agent. Resources in the project (flows, functions, topics, etc.) control the agent's runtime behavior.

## CLI Commands

| Command | Description |
|---------|-------------|
| `poly init` | Initialize a project locally (interactive or with `--region`, `--account_id`, `--project_id`) |
| `poly project create` | Create a new project (interactive or with `--region`, `--account_id`, `--name`) |
| `poly docs` | Resource documentation (`poly docs flows functions`, or `--all`; `-o` to write to file) |
| `poly pull` | Pull remote config into local project (`-f` to force overwrite) |
| `poly push` | Push local changes to Agent Studio (`-f`, `--dry-run`, `--skip-validation`) |
| `poly status` | List changed files |
| `poly diff` | Show diffs (files, deployment hashes, or `--before`/`--after`) |
| `poly revert` | Revert local changes (all by default, or specific files) |
| `poly branch` | Branch management: `list`, `create`, `switch`, `current`, `delete`, `merge` |
| `poly format` | Format resources (ruff, YAML/JSON; `--check`, `--files`, optional `--ty`) |
| `poly validate` | Validate project configuration locally |
| `poly review` | Diff review gist: `create` (needs `GITHUB_ACCESS_TOKEN`), `list`, `delete` |
| `poly chat` | Chat with agent (default `-e branch`; `--push`, `--metadata` / `--flows` / `--functions` / `--state`) |
| `poly deployments list` | List deployments (`-e`, `--limit`, `--offset`, `--details`) |
| `poly deployments show` | Deployment details; pre-release/live include bundled sandbox changes |
| `poly deployments promote` | Promote between environments. Agents: `--dry-run` only |
| `poly deployments rollback` | Roll back **sandbox** only (`--to`). Agents: `--dry-run` only |
| `poly conversations list` | List past conversations (`--limit`, `--offset`) |
| `poly conversations get` | Conversation details including turns |
| `poly conversations get-audio` | Download call audio as WAV (`--direction`, `--redacted`, `-o`) |

## Workflow Guide

### Prerequisites

- Python 3.14 installed locally
- Poly CLI installed (`pip install polyai-adk`)

Run commands from the project directory (or pass `--path`). Prefer `--json` for machine-readable output.
For flags and subcommands not covered here: `poly <command> -h` or `poly <command> <subcommand> -h`.
For resource schemas and examples: `poly docs <resource_type>` (not `-h`).

### Auth

- **New free-tier users** — `poly start` (creates account, saves credentials, can create first project)
- **Enterprise users** — `poly login` (browser sign-in; optional `--region us-1|uk-1|euw-1|studio`)

Credentials are stored in `~/.poly/credentials.json`. For CI, set `POLY_ADK_KEY` or a per-region key (e.g. `POLY_ADK_KEY_US`).

### Init and pull project

To load an existing project, run `poly init` and follow the prompts, or pass `--region`, `--account_id`, and `--project_id` directly. This creates a local folder with the project config.

To create a new project, run `poly project create` and follow the prompts or pass `--region`, `--account_id`, and `--name` directly.

To pull the latest config from Agent Studio: `poly pull`. Use `-f` to force overwrite local changes.

Switch branches with `poly branch switch {name}`. This also pulls the latest config for that branch.

### Branch management

Always create branches from `main`. Use descriptive names (e.g. `add-booking-flow`, `update-hours-topic`).

```
poly branch list --json
poly branch current --json
poly branch create {name} --json   # must be on main
poly branch switch {name} --json
poly branch delete {name} --json
```

### Making edits

Before making edits or plans, run `poly docs` to get basic information then `poly docs {resource_type}` (e.g. `poly docs functions flows topics`) or `poly docs --all` to get the exact YAML/Python format, required fields, validation rules, and examples.
**Always consult the docs before writing or editing a resource.**

Track and inspect changes as you go:
```
poly status --json       # list changed files
poly diff --json         # show full diff
poly validate --json     # validate locally before pushing
poly format --json       # format after edits
```

### Pushing changes

```
poly push --json                   # push local changes (pulls and merges first)
poly push --dry-run --json         # preview what would be pushed
poly push -f --json                # force overwrite remote
```

If `success` is false and `files_with_conflicts` is non-empty, resolve the conflict markers in the affected files and push again.

### Testing changes

To verify changes, use `poly chat`. Default environment is the current **branch** (`-e branch`). Push first, or use `--push` to push local changes before chatting.

Start a session and send the first message:
```
poly chat --json -m "first message"
poly chat --json --push -m "first message"
```

Continue turn by turn using the `conversation_id` from the response:
```
poly chat --json --conv-id {conversation_id} -m "next message"
```

To explicitly end the session, use `/exit`:
```
poly chat --json --conv-id {conversation_id} -m "/exit"
```

Use flags to inspect what the agent is doing each turn:
```
--flows       show active flow and step
--functions   show function/tool calls
--state       show per-turn state variable changes
--metadata    show flows, functions, and state (all of the above)
```

Test against other environments with `-e sandbox`, `-e pre-release`, or `-e live`.

### Inspecting conversations

Use `poly conversations` to review past calls or chats (e.g. after testing in Agent Studio or via `poly chat`).

List recent conversations:
```
poly conversations list --json
poly conversations list --json --limit 20 --offset 0
```

Inspect a specific conversation (turns, metadata):
```
poly conversations get {conversation_id} --json
```

Download call audio as WAV:
```
poly conversations get-audio {conversation_id} -o recording.wav
poly conversations get-audio {conversation_id} --direction user --json
```

`--direction` is `combined` (default), `user`, or `agent`.

### Merging

Before merging, always show the user the full diff and ask them to confirm. Merge merges the **current branch** into `main` (switch to the feature branch first).

```
poly diff --before main --after {branch_name} --json
poly branch merge "Merge message describing changes" --json
```

Use `poly review create --before main --after {branch_name}` to generate a shareable diff review page (requires `GITHUB_ACCESS_TOKEN`).

### Inspecting deployments

Use `poly deployments` to see what is deployed in each environment (`sandbox`, `pre-release`, `live`).

List recent versions:
```
poly deployments list --json
poly deployments list --json -e live
poly deployments list --json --details
```

Show one deployment (hash or prefix):
```
poly deployments show {hash} --json
poly deployments show {hash} --json -e pre-release
```

For `pre-release` and `live`, `show` includes the sandbox deployments bundled in that promotion (`included_deployments` in JSON). Defaults to `sandbox`.

Compare a deployment to local or another version:
```
poly diff {hash} --json
poly diff --before {hash} --after main --json
```

### Promoting and rolling back

`poly deployments promote` moves a version between environments (sandbox → pre-release → live).

`poly deployments rollback` rolls back the **sandbox** deployment to an earlier version. It does not affect pre-release or live.

**Agents may only preview** with `--dry-run`. A human must run promote/rollback without `--dry-run` in their own terminal.

Preview a promotion:
```
poly deployments promote --from {hash} --to pre-release --dry-run --json
poly deployments promote --from {hash} --to live --dry-run --json
```

Preview a sandbox rollback:
```
poly deployments rollback --to {hash} --dry-run --json
```

Tell the user the exact command to run when they are ready to promote or roll back for real.

### Guardrails

- Do not edit generated stubs in `_gen/` — immutable reference stubs.
- Never run `poly push -f` without checking the diff and confirming with the user.
- Never merge a branch without showing the diff and confirming with the user.
- Never delete a branch without confirming with the user.
- **Never promote to `live`** — do not run `poly deployments promote --to live` without `--dry-run`. Preview with `--dry-run`, then give the human the command to run.
- **Never run `poly deployments promote` or `poly deployments rollback` without `--dry-run`** — only humans execute real promotes/rollbacks.

## Quick Reference

### Project structure

```
<account>/<project>/
├── _gen/                               # Generated stubs - do not edit
├── agent_settings/
│   ├── personality.yaml
│   ├── role.yaml
│   ├── rules.txt
│   ├── safety_filters.yaml             # Optional
│   └── experimental_config.json        # Optional
├── config/
│   ├── api_integrations.yaml           # Optional
│   ├── entities.yaml                   # Optional
│   ├── handoffs.yaml                   # Optional
│   ├── sms_templates.yaml              # Optional
│   └── variant_attributes.yaml         # Optional
├── voice/
│   ├── configuration.yaml
│   ├── safety_filters.yaml             # Optional
│   ├── speech_recognition/
│   │   ├── asr_settings.yaml
│   │   ├── keyphrase_boosting.yaml     # Optional
│   │   └── transcript_corrections.yaml # Optional
│   └── response_control/
│       ├── pronunciations.yaml         # Optional
│       └── phrase_filtering.yaml       # Optional
├── chat/
│   ├── configuration.yaml
│   └── safety_filters.yaml             # Optional
├── flows/
│   └── {flow_name}/
│       ├── flow_config.yaml
│       ├── steps/
│       ├── function_steps/
│       └── functions/
├── functions/
│   ├── start_function.py               # Optional
│   ├── end_function.py                 # Optional
│   └── {function_name}.py
├── topics/
│   └── {topic_name}.yaml
├── variables/                            # Optional - state variable definitions
│   └── {variable_name}.yaml
└── project.yaml
```

### Placeholders and syntax

| Syntax | Resolves to | Usable in |
|--------|-------------|-----------|
| `{{fn:function_name}}` | Global function | Rules, topics (actions), advanced step prompts |
| `{{ft:function_name}}` | Flow transition function | Advanced step prompts (same flow only) |
| `{{entity:entity_name}}` | Collected entity value | Flow step prompts |
| `{{attr:attribute_name}}` | Variant attribute | Rules, prompts, topics, greeting, disclaimer, personality, role |
| `{{twilio_sms:template_name}}` | SMS template | Rules, topics (actions) |
| `{{ho:handoff_name}}` | Handoff destination | Rules |
| `{{vrbl:variable_name}}`| State variable | Prompts, topic actions, SMS templates |
