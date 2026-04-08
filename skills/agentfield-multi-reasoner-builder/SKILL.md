---
name: agentfield-multi-reasoner-builder
description: Architect and ship a complete multi-agent backend system on AgentField from a one-line user request. Use when the user asks to build, scaffold, design, or ship an agent system, multi-agent pipeline, reasoner network, AgentField project, financial reviewer, research agent, compliance agent, or any LLM composition that should outperform LangChain/CrewAI/AutoGen — especially when they want a runnable Docker-compose stack and a working curl smoke test.
---

# AgentField Multi-Reasoner Builder

You are not a prompt engineer. You are a **systems architect** building composite reasoning machines on AgentField. The intelligence is in the composition, not the components.

## HARD GATE — READ BEFORE ANYTHING ELSE

> **Do NOT write any code, generate any file, or scaffold any project until you have:**
> 1. Either (a) asked the ONE grooming question and received an answer, OR (b) confirmed that the user's first message ALREADY contains a clear use case — in which case **skip the question and proceed straight to design**. The "build now, key later" rule (below in the grooming protocol) ALWAYS overrides this gate when the brief is complete; you do NOT need a key in chat to start building because the user will paste it into `.env` themselves
> 2. Read `references/choosing-primitives.md` (mandatory — sets the philosophy and the real SDK signatures)
> 3. Designed the reasoner topology with **depth, not just breadth** (see "Reasoners are software APIs" below) — which `@app.reasoner` units, who calls whom, which are `.ai` vs deterministic skills, where the dynamic routing happens
>
> **Do NOT default to a single big reasoner with one `app.ai` call.** That's a CrewAI clone. Decompose.
>
> **Do NOT default to a single fat orchestrator that calls every specialist directly in one fan-out.** That's a star pattern, also a CrewAI clone wearing a different costume. Build deep call chains (see below).
>
> If you cannot draw your system as a non-trivial graph **with depth ≥ 3**, you have not architected anything.
>
> Violating the letter of this gate is violating the spirit of the gate. There are no exceptions for "simple" use cases.

## The unit of intelligence is the reasoner — treat them as software APIs

This is the most important framing in the entire skill. **Each reasoner is a microservice. Reasoners call other reasoners the way one REST API calls another.** The orchestrator at the top is not the only thing that calls reasoners — every reasoner can (and often should) call sub-reasoners that are themselves further decomposed.

**Bad shape — flat star (the default a coding agent will reach for):**
```
entry_orchestrator
├── specialist_1   ──┐
├── specialist_2   ──┤
├── specialist_3   ──┼── all called once, in parallel, by the orchestrator
├── specialist_4   ──┤
└── specialist_5   ──┘
        │
        v
   synthesizer
```

This is depth = 2 (entry → specialist → done). It's basically `asyncio.gather([llm_call_1, llm_call_2, ...])` with extra ceremony. Easy to write, but it doesn't earn the AgentField label.

**Good shape — composition cascade (depth ≥ 3, parallelism at multiple levels):**
```
triage_case (entry)
├── case_classifier ─────────────┐
│   └── chief_complaint_parser   │
│       └── medical_term_normalizer
│
├── ami_assessor                 │  ← all parallel
│   ├── cardiac_risk_calculator (deterministic skill)
│   ├── ami_pattern_matcher (.ai)
│   │   └── ecg_finding_classifier (.ai called by ami_pattern_matcher when needed)
│   └── biomarker_predictor (.ai)
│
├── pe_assessor                  │
│   ├── wells_score_calculator (deterministic skill)
│   ├── dyspnea_grader (.ai)
│   └── dvt_history_checker (.ai)
│
├── stroke_assessor              │
│   ├── fast_screen (.ai)
│   └── nihss_estimator (.ai called only if fast_screen positive)
│
└── adversarial_synthesizer ─────┘
    ├── steel_man_alternative_dx (.ai called once per primary assessment)
    └── confidence_reconciler (.ai)
        └── deterministic_safety_overrides (plain Python)
```

This system has depth 4, runs **at least three parallelism waves**, and each "specialist" is itself composed of 2–4 sub-reasoners that may call each other. **Each reasoner has a single cognitive responsibility you could write a one-line API contract for.** Reasoners that always co-execute become one reasoner; reasoners that have distinct judgment surfaces stay separate.

**Why this matters:**
1. **Each reasoner is replaceable.** Want to swap `wells_score_calculator` for a more accurate one? Change one file. The flat-star pattern would have that logic buried inside a 200-line `pe_assessor` reasoner.
2. **Each reasoner is testable in isolation.** You can `curl /api/v1/execute/medical-triage.wells_score_calculator` directly with a synthetic input. The flat-star pattern only exposes the entry reasoner.
3. **Each reasoner is reusable.** `medical_term_normalizer` can be called from `chief_complaint_parser` AND from `comorbidity_amplifier` AND from a future `discharge_summary_generator`. The flat-star pattern duplicates logic across specialists.
4. **Each reasoner is observable.** The control plane workflow DAG shows the full call tree, not just a single `gather`. The verifiable credential chain has structure.
5. **Parallelism happens at multiple levels.** The flat-star fan-outs N specialists once. The deep DAG fans out N specialists × M sub-calls each, with the orchestration `asyncio.gather`-ing at each layer. Total wall-clock time goes down even though total calls go up.

**Concrete rules:**
- If a reasoner has more than ~30 lines of body code, it's probably 2 reasoners
- If two reasoners always call each other in sequence, they should be one reasoner (or one reasoner with a deterministic helper)
- If your entry reasoner is the ONLY thing that calls `app.call`, the architecture is too flat — push the calls down into the specialists
- If your topology can be drawn as a literal star, throw it out and design for depth
- A reasoner should have a clear API contract you could write in one sentence: *"Given X, return Y. Calls Z, W."*

**The unit of intelligence is the reasoner. Treat them like software APIs and the system writes itself.**

## The non-negotiable promise

Every invocation of this skill must end with the user able to run **two commands** and get a working multi-reasoner system:

```bash
docker compose up --build
curl -X POST http://localhost:8080/api/v1/execute/<node>.<entry_reasoner> \
  -H 'Content-Type: application/json' \
  -d '{"input": {"...": "..."}}'
```

If you cannot deliver that, you have failed. No theoretical architectures. No "here's how you would do it." A running stack and a curl that returns a real reasoned answer.

**Note the curl body shape: `{"input": {...kwargs...}}`** — the control plane wraps reasoner kwargs in an `input` field. Verified against `control-plane/internal/handlers/execute.go:1000`. Many coding agents get this wrong.

## Workflow (universal — works for any coding agent)

1. **Announce** you're using the `agentfield-multi-reasoner-builder` skill.
2. **Probe the environment** with `af doctor --json` (one command, see "Environment introspection" below). This tells you which provider keys are set, which harness CLIs are present, and the recommended `AI_MODEL`. Use this output instead of guessing.
3. **Ask the one grooming question** (below) ONLY if the user hasn't already provided everything.
4. **Read `choosing-primitives.md` ALWAYS.** Read other references when their trigger fires (table below).
5. **Design the topology** before writing files.
6. **Lay down infrastructure** with `af init <slug> --language python --docker --defaults --non-interactive --default-model <model_from_doctor>` (one command, see "Infrastructure scaffold" below).
7. **Customize `main.py` and `reasoners.py`** with the real reasoner architecture per `scaffold-recipe.md`. Generate `CLAUDE.md` (from `project-claude-template.md`) and `README.md` AFTER you know the entry reasoner name and the curl payload.
8. **Validate**: `python3 -m py_compile main.py`, `docker compose config`, ideally `docker compose up --build` + verification ladder.
9. **Hand off** with the output contract below.

## Environment introspection: `af doctor`

Run this **once** at the start of every build. It returns ground truth about the local environment in a single JSON document instead of having you probe `which`, `env`, `docker image inspect`, etc. yourself:

```bash
af doctor --json
```

Key fields you'll consume:
- `recommendation.provider` — `openrouter` / `openai` / `anthropic` / `google` / `none`
- `recommendation.ai_model` — the LiteLLM-style model string to bake into the scaffold's `AI_MODEL` default
- `recommendation.harness_usable` — `true` only if at least one of `claude-code` / `codex` / `gemini` / `opencode` is on PATH. **If `false`, do not use `app.harness()` in the scaffold under any circumstance.**
- `recommendation.harness_providers` — list of available CLI names (use these as the `provider=` value if and only if `harness_usable` is true)
- `provider_keys.{name}.set` — boolean per provider (no values leaked)
- `control_plane.docker_image_local` — whether `agentfield/control-plane:latest` is already cached (informs whether the first `docker compose up` will need to pull)
- `control_plane.reachable` — whether a control plane is already running locally (so you can curl test reasoners against it before building your own)

**Use the doctor's output to set the `--default-model` flag on `af init` and to decide whether `app.harness()` is even an option in the architecture.** Do not hardcode your assumptions about the environment.

## Infrastructure scaffold: `af init --docker`

Run this **once** after `af doctor` and your architecture design. It produces the four infrastructure files that you should not customize plus the language scaffold (Python `main.py`, `reasoners.py`, `requirements.txt`):

```bash
af init <slug> --language python --docker --defaults --non-interactive \
  --default-model <model_string_from_doctor>
```

What it generates:
- `Dockerfile` — universal Python 3.11-slim, builds from project dir, no repo coupling
- `docker-compose.yml` — control-plane + agent service with healthcheck and service-healthy gating
- `.env.example` — all four provider keys (OpenRouter, OpenAI, Anthropic, Google) and `AI_MODEL` with the doctor-recommended default
- `.dockerignore`
- `main.py`, `reasoners.py`, `requirements.txt`, `README.md`, `.gitignore` — the standard language scaffold (you'll **rewrite `main.py` and `reasoners.py`** with your real architecture)

What it does NOT generate (intentionally):
- `CLAUDE.md` — you generate this from `references/project-claude-template.md` AFTER writing the real reasoners, so it can name them and justify the architecture
- A README with the real curl — the default `README.md` is generic; you replace it AFTER picking the entry reasoner so the curl uses real kwargs

The four infrastructure files are zero-change for the agent: Dockerfile installs `agentfield` from `requirements.txt` and copies the project dir; compose wires control-plane + agent with healthcheck; `.env.example` exposes all providers; `.dockerignore` covers the standard cases. **Do not modify them unless you have a real reason.**

## Reference table — load when

| File | Load when |
|---|---|
| `choosing-primitives.md` | **Every invocation** — before any code |
| `architecture-patterns.md` | Designing inter-reasoner flow / picking HUNT→PROVE, parallel hunters, fan-out, streaming, meta-prompting |
| `scaffold-recipe.md` | Actually writing files / docker-compose / Dockerfile |
| `verification.md` | Writing the smoke test ladder or declaring done |
| `project-claude-template.md` | Generating the per-project CLAUDE.md (always) |
| `anti-patterns.md` | When tempted to take a shortcut OR when the user pushes back on a rejection |

Reference files are one level deep from this file. Do not nest reads — if a reference points at another reference, come back here and load the second one directly.

## The grooming protocol (1 question, then build)

Ask **exactly one** question and **one** key request. Nothing else upfront:

> "Tell me in 1–2 sentences what you want this agent system to do, and paste your provider key. We support OpenRouter (default), OpenAI, or Anthropic — any LiteLLM-compatible model. Example: `OPENROUTER_API_KEY=sk-or-v1-...`"

**Skip-the-question rule:** if the user's first message ALREADY contains a clear use case, do NOT ask the grooming question — even if they didn't paste a provider key. This is the **"build now, key later"** policy:

- If the user gives a clear use case AND a provider key → proceed straight to design + build
- If the user gives a clear use case AND says they'll paste the key into `.env` later → ALSO proceed straight to design + build. The scaffold will work with `OPENROUTER_API_KEY=sk-or-v1-FAKE` for `docker compose config` validation. The user runs the real key from `.env` when they're ready
- If the user gives a clear use case AND says nothing about a key → proceed straight to design + build. The `.env.example` you generate makes it obvious where to put the key
- If the user's request is genuinely vague or ambiguous along an architecture-changing axis → THEN ask one question

The point is to **never block the build on a key the user is going to drop into `.env` themselves**. Asking a redundant question after the user has already given you the use case wastes their time and signals you're following a script instead of understanding.

Then proceed. Infer everything else from the use case. State your assumptions in the final handoff so the user can correct them in iteration 2.

**Only ask follow-up questions if the use case is genuinely ambiguous along an axis that changes the architecture** (not the wording). Examples that warrant a follow-up:

- Input is a 200-page document vs. a small JSON payload (changes whether you need a navigator harness)
- Output must include verifiable citations (changes whether you need a provenance reasoner)
- Synchronous request/response vs. event-driven (pattern 8 vs. patterns 1–7)

Examples that do **NOT** warrant a follow-up: model preference, file naming, port number, code style, what to call the entry reasoner. Decide and state.

## The five primitives (cheat sheet — full detail in `choosing-primitives.md`)

- **`@app.reasoner()`** — every cognitive unit. Schemas come from **type hints** (no `input_schema=` param exists).
- **`@app.skill()`** — deterministic functions. No LLM. Use whenever an LLM call is overkill.
- **`app.ai(system, user, schema, model, tools, ...)`** — single OR multi-turn LLM call. `tools=[...]` makes it stateful. `model="..."` per call overrides AIConfig default.
- **`app.harness(prompt, provider="claude-code"|"codex"|"gemini"|"opencode")`** — delegates to an external coding-agent CLI. **Not** a generic tool-using LLM (that's `app.ai(tools=[...])`). **REQUIRES** the chosen provider's CLI to be installed inside the agent container — see "Harness availability gate" below.
- **`app.call(target, **kwargs)`** — inter-reasoner traffic THROUGH the control plane. Returns `dict`. **No model override param** — thread `model` as a regular reasoner kwarg.

**The bias:** many small `@app.reasoner()` units. `@app.skill()` for anything code can do. `app.ai()` with explicit prompts. Reserve `app.harness()` for real coding-agent delegation.

## Harness availability gate (READ BEFORE USING `app.harness()`)

`app.harness()` runs an external coding-agent CLI inside the agent container — `claude-code`, `codex`, `gemini`, or `opencode`. **The default `python:3.11-slim` Docker image has none of these installed.** A scaffold that uses `app.harness()` without installing the CLI in the Dockerfile will crash at runtime.

**The check is automated.** `af doctor --json` reports `recommendation.harness_usable` (true/false) and `recommendation.harness_providers` (the list of CLIs on PATH). Use the doctor output as the source of truth — do not assume.

**Default rule:** scaffolds **MUST NOT** use `app.harness()` at all when `recommendation.harness_usable == false`. Use `app.ai(tools=[...])` for stateful reasoning, or a `@app.reasoner()` that loops `app.ai()` for chunked work. These work in the default container with zero extra setup.

**You may use `app.harness()` ONLY when ALL of the following are true:**

1. The use case **genuinely requires a real coding agent** in the loop — i.e. the reasoner needs to write/edit files on disk, run shell commands, or perform complex non-LLM coding work that `app.ai(tools=[...])` cannot do.
2. You modify the Dockerfile to install the chosen provider's CLI. Example for Claude Code:
   ```dockerfile
   RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm \
       && npm install -g @anthropic-ai/claude-code \
       && rm -rf /var/lib/apt/lists/*
   ```
3. You add a **startup availability check** in `main.py` that fails fast with a clear error if the CLI is not on PATH:
   ```python
   import shutil, sys
   if not shutil.which("claude"):  # or "codex" / "gemini" / "opencode"
       print("ERROR: app.harness(provider='claude-code') requires the `claude` CLI in PATH.", file=sys.stderr)
       sys.exit(1)
   ```
4. The README explicitly tells the user that the agent container ships with `claude-code` (or whatever) and explains the consequence on image size.

**If any of the four are not satisfied, do not use `app.harness()`.** Refactor the reasoner to use `app.ai(tools=[...])` or a chunked `@app.reasoner()` loop. There is no scenario where it's OK to write `app.harness(provider="claude-code")` in code that ships in a container without the `claude` binary.

When in doubt: **don't use harness.** The user can ask for it in iteration 2. The first build's job is to work on `docker compose up` with zero external CLI dependencies.

## Mandatory patterns (every build must have all three)

### 1. Per-request model propagation

The entry reasoner accepts `model: str | None = None` and threads it through every `app.ai(..., model=model)` and `app.call(..., model=model)`. Child reasoners accept `model` the same way and use it. The user can A/B test models per request:

```bash
curl -X POST http://localhost:8080/api/v1/execute/<slug>.<entry> \
  -d '{"input": {"...": "...", "model": "openrouter/openai/gpt-4o"}}'
```

If `model` is omitted, the AIConfig default from the env var `AI_MODEL` is used. **`app.call()` has no native model override — you MUST thread model through reasoner kwargs.**

### 2. Routers when reasoners > 4

Use `AgentRouter(prefix="domain", tags=["domain"])` and `app.include_router(router)` to split reasoners into separate files. Tags merge between router and per-decorator. **Note:** `prefix="clauses"` auto-namespaces reasoner IDs as `clauses_<func_name>` — call them as `app.call(f"{app.node_id}.clauses_<func_name>", ...)`.

### 3. Tags on the entry reasoner

The public entry reasoner is decorated with `tags=["entry"]` so it surfaces in the discovery API. Tags are free-form (not reserved); use domain tags for internal reasoners.

## Hard rejections — refuse these without negotiation

| ❌ Rejected pattern | ✅ AgentField alternative |
|---|---|
| Direct HTTP between reasoners (`httpx.post(...)`) | `await app.call(f"{app.node_id}.X", ...)` — control plane needs to see every call to track DAG, generate VCs, replay |
| One giant reasoner doing 5 things | Decompose into 5 reasoners coordinated by an orchestrator using `app.call` + `asyncio.gather` |
| Static linear chain `A → B → C → D` (always, no routing) | Dynamic routing: intake reasoner picks downstream reasoners based on what it found |
| `app.ai(prompt=full_50_page_doc)` | `@app.reasoner` that loops `app.ai` per chunk, OR `app.ai(tools=[...])` with explicit tool calls |
| Unbounded `while not confident: app.ai(...)` | Hard cap: `for _ in range(MAX_ROUNDS): ...` with explicit break |
| Passing structured JSON between two LLM reasoners | Convert to prose. LLMs reason over natural language, not JSON serialization |
| Replicating sort/dedup/score work with `app.ai` | `@app.skill()` with plain Python |
| Scaffold without a working `curl` that returns real output | The promise is `docker compose up` + curl. Always include it |
| Multi-container agent fleet when one node would do | One agent node, many reasoners — unless there's a real boundary |
| Hardcoded `node_id` in `app.call("financial-reviewer.X", ...)` | `app.call(f"{app.node_id}.X", ...)` — survives `AGENT_NODE_ID` rename |
| Hardcoded model | `model=os.getenv("AI_MODEL", default)` AND per-request override via reasoner kwarg |
| `app.ai()` schema with no `confident` field and no fallback | Schema must include `confident: bool`, call site checks it and escalates |
| `app.harness(provider="claude-code")` in a default scaffold | Default container has no `claude` CLI. Use `app.ai(tools=[...])` or a chunked-loop reasoner. See "Harness availability gate" |
| `input_schema=` or `output_schema=` parameter on `@app.reasoner` | These don't exist. Schemas come from type hints |
| `app.serve()` in `__main__` | `app.run()` — auto-detects CLI vs server mode |

When the user explicitly demands a rejected pattern, name the rejection, explain *why* in one sentence, propose the AgentField alternative, and only build it their way after they've confirmed they understand the tradeoff. Add a `# NOTE: User requested X over canonical Y` comment.

## Rationalization counters & red flags

These thoughts mean STOP. If you notice any of them, re-read the linked reference and reconsider.

| Thought / symptom | Reality / re-read |
|---|---|
| "Quick demo, I'll skip the architecture" | The skill exists to be stronger than a chain. Weak demo proves nothing |
| "I'll pass JSON between two reasoners" | LLMs reason over prose. Strings between LLMs, JSON only for code |
| "One big `analyze()` reasoner is fewer files" | Decompose. Granularity is the forcing function for parallelism. `choosing-primitives.md` |
| "I'll skip the CLAUDE.md / README" | They're how the next coding agent extends without breaking it. Always generate |
| "I'll ask 5 questions to be safe" | One question. State assumptions. Iterate |
| "Curl is enough, skip discovery API" | Discovery API tells you in 2s which step actually failed. `verification.md` |
| "I need stateful tool-using → `app.harness()`" | NO. `app.harness()` is external coding-agent CLI delegation AND requires the CLI in the container. Use `app.ai(tools=[...])` or a chunked-loop reasoner |
| "I'll add `app.harness(provider='claude-code')` for the deep reasoning step" | The default Python container has no `claude` CLI. The scaffold will crash on first run. Read "Harness availability gate" |
| "I'll add `input_schema=` to the decorator" | That param doesn't exist. Schemas come from type hints |
| ".ai() for a 50-page document" | `app.ai(tools=[...])` or a chunked-loop reasoner. `choosing-primitives.md` |
| "Static `for` loop of LLM calls, no routing" | Add dynamic routing or admit AgentField isn't justified. `architecture-patterns.md` |
| "Skipping `python3 -m py_compile` and `docker compose config`" | Always run. `scaffold-recipe.md` |
| "I'll write `import requests` to call the other reasoner" | Use `app.call(f"{app.node_id}.X", ...)`. `choosing-primitives.md` |
| "I'll use `app.serve()` in main" | Use `app.run()`. Auto-detects CLI vs server |

## Output contract (every build)

The final message to the user MUST contain these sections, in this order, in a clean copy-pasteable format. The whole point is the first-time user can read the message top to bottom and within 60 seconds have the system running and a working curl in another terminal.

### 1. What was scaffolded

Generated file tree with absolute paths.

### 2. Architecture sketch

4–6 bullets: what each reasoner does, who calls whom, where the dynamic routing happens, where the safety guardrails fire.

### 3. Assumptions made

5–10 bullets — the things you inferred without asking.

### 4. 🚀 Run it (3 commands)

```bash
cd <absolute_project_path>
cp .env.example .env       # then paste your OPENROUTER_API_KEY into .env
docker compose up --build
```

Wait until you see `agent registered` in the logs (~30–90 seconds first run).

### 5. 🌐 Open the UI

After the stack is up, open these URLs in your browser:

| URL | What it shows |
|---|---|
| **http://localhost:8080/ui/** | AgentField control plane web UI — live workflow DAG, reasoner discovery, execution history, verifiable credential chains |
| **http://localhost:8080/api/v1/discovery/capabilities** | JSON: every reasoner registered with the control plane (proves your build deployed) |
| **http://localhost:8080/api/v1/health** | Health check |

### 6. ✅ Verify the build (in another terminal)

```bash
# 1. Control plane up?
curl -fsS http://localhost:8080/api/v1/health | jq '.status'

# 2. Agent node registered? (use ?health_status=any — default filter can hide healthy nodes)
curl -fsS 'http://localhost:8080/api/v1/nodes?health_status=any' | jq '.nodes[] | {id: .node_id, status: .status}'

# 3. All reasoners discoverable?
#    Response shape: .capabilities[].reasoners[].id (NOT .reasoners[].name)
curl -fsS http://localhost:8080/api/v1/discovery/capabilities \
  | jq '.capabilities[] | select(.agent_id=="<slug>") | .reasoners | map({id, tags})'
```

### 7. 🎯 Try it — sample curl

```bash
curl -X POST http://localhost:8080/api/v1/execute/<slug>.<entry_reasoner> \
  -H 'Content-Type: application/json' \
  -d '{
    "input": {
      "<param1>": "<realistic value>",
      "<param2>": <realistic value>,
      "model": "openrouter/google/gemini-2.5-flash"
    }
  }' | jq
```

**The curl above must use realistic data the user can run as-is and see a real reasoned answer.** Do not use placeholder values like `"foo"` or `"test"`. Use concrete data that actually exercises every reasoner in the system. The optional `"model"` field overrides the AIConfig default per-request — show it in the example so users discover the per-request override.

If the user provided test data in the brief (e.g. a sample patient case, a sample contract, a sample loan application), use THAT data verbatim in this curl. The first execution they run should be the most demonstrative one.

### 8. 🏆 Showpiece — verifiable workflow chain

```bash
LAST_EXEC=$(curl -s http://localhost:8080/api/v1/executions | jq -r '.[0].workflow_id')
curl -s http://localhost:8080/api/v1/did/workflow/$LAST_EXEC/vc-chain | jq
```

This is the cryptographic verifiable credential chain — every reasoner that ran, with provenance. No other agent framework gives you this. Mention it.

### 9. Next iteration upgrade

One concrete suggestion (e.g., "swap the intake `.ai()` for a chunked-loop reasoner if inputs grow past 2 pages", "add a second adversarial wave with a different prompt for the highest-stakes branches").

## TypeScript

A TypeScript SDK exists at `sdk/typescript/` and mirrors the Python API. **Default to Python.** If the user explicitly says "TypeScript" or "Node", point them at `sdk/typescript/` and use the equivalent shape: `new Agent({nodeId, agentFieldUrl, aiConfig})` + `agent.reasoner('name', async (ctx) => {...})`. Otherwise stay Python — every reference and recipe in this skill is Python-first.

## Bottom line

Your output is judged by three things:
1. **Does the curl return a real reasoned answer?** (the user can run the command and see intelligence happen)
2. **Does the architecture look like composite intelligence?** (parallel reasoners, dynamic routing, decomposition — not a chain wearing a costume)
3. **Can a future coding agent extend it without breaking the contract?** (CLAUDE.md present, anti-patterns listed, validation commands documented)

If all three are true, you've done it right. The first-time AgentField user must see the value within minutes of running the curl.
