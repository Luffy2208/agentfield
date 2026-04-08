# Verification — Prove the Build Is Real

A scaffold that "looks right" but isn't actually wired up is worse than no scaffold. The control plane exposes a discovery API that lets you prove the system works in seconds. Use it.

## The verification ladder (run all four, in order)

```bash
# 1. Control plane health
curl -fsS http://localhost:8080/api/v1/health | jq

# 2. Agent node has registered itself with the control plane
curl -fsS http://localhost:8080/api/v1/nodes | jq '.[] | {id: .node_id, status, last_seen}'

# 3. Every reasoner you defined is discoverable
curl -fsS http://localhost:8080/api/v1/discovery/capabilities \
  | jq --arg slug "<slug>" '.reasoners[] | select(.node_id==$slug) | {name, tags, description}'

# 4. The entry reasoner produces a real reasoned answer
#    NOTE: control plane wraps kwargs in {"input": {...}} (verified at execute.go:1000)
curl -X POST http://localhost:8080/api/v1/execute/<slug>.<entry_reasoner> \
  -H 'Content-Type: application/json' \
  -d '{
    "input": {
      "<kwarg1>": "<value>",
      "<kwarg2>": <value>,
      "model": "openrouter/google/gemini-2.5-flash"
    }
  }' | jq
```

If any step fails, **do not hand off**. Diagnose and fix.

## Common failures and fast diagnosis

| Symptom | Likely cause | Fix |
|---|---|---|
| `/api/v1/health` hangs or refuses connection | Control plane container is still booting | Wait 5–10s, retry. If still failing, `docker compose logs control-plane` |
| `/api/v1/nodes` returns `[]` | Agent node hasn't registered. Network issue or agent crashed at boot | `docker compose logs <slug>` — look for `OPENROUTER_API_KEY` missing, import errors, or `agent registered` |
| Node listed but no reasoners in `/discovery/capabilities` | The Python file imported, but the `@app.reasoner()` decorators didn't run (e.g., reasoners are in a router that wasn't included) | Verify `app.include_router(...)` is called in `main.py` before `app.run()` |
| Reasoners present but execute hangs | Reasoner is making an LLM call that's failing silently | `docker compose logs <slug> --follow` while running curl. Look for litellm errors |
| Execute returns 500 with "model not found" | `AI_MODEL` env var doesn't match the provider key you set | Check `.env` — `OPENROUTER_API_KEY` requires `openrouter/...` model names, etc. |
| Execute returns 200 but the output is empty/garbage | The reasoner ran but the architecture is wrong (e.g., `.ai()` got truncated input) | Look at logs to see what input each reasoner actually got |

## Sync execute timeout (90s) — IMPORTANT

`POST /api/v1/execute/<target>` is a **synchronous** endpoint with a hard **90-second timeout** at the control plane. If the entry reasoner's full pipeline (including all child `app.call`s, all `app.ai` calls, and any retries) takes longer than 90s, the control plane returns `HTTP 400 {"error":"execution timeout after 1m30s"}`.

**Implications for the architecture you generate:**
- **Pick fast models for the default.** `openrouter/google/gemini-2.5-flash` and `openrouter/openai/gpt-4o-mini` finish a 6–10 step parallel pipeline in 10–25 seconds. Slower models like `openrouter/anthropic/claude-3-5-sonnet-*`, `openrouter/minimax/minimax-m2.7`, or `openrouter/openai/o1` often blow the budget.
- **Parallelize aggressively at multiple depths.** A pipeline of 10 sequential `app.ai` calls at 5s each = 50s (close to the limit). The same 10 calls organized as a deep DAG with 3 parallelism waves = 15s. Use `asyncio.gather` for every fan-out, and push fan-outs DOWN into sub-reasoners (see `architecture-patterns.md` "Reasoner Composition Cascade"), not just at the entry orchestrator.
- **For workflows that genuinely need >90s** (large fan-outs, slow models, navigation-heavy harnesses): use `POST /api/v1/execute/async/<target>` instead. It returns immediately with an `execution_id`; poll `GET /api/v1/executions/<id>` for the result. Document this in the README so users know which endpoint to hit.

When the user's brief implies a slow pipeline, default to `gemini-2.5-flash` and document the async endpoint as the upgrade path.

## Useful introspection endpoints

| Endpoint | What it tells you |
|---|---|
| `GET /api/v1/health` | Control plane up |
| `GET /api/v1/nodes?health_status=any` | Which agent nodes have registered (the default filter is `active`, which can return empty even when agents are healthy — use `?health_status=any` to be safe) |
| `GET /api/v1/nodes/:node_id` | Details of one node |
| `GET /api/v1/discovery/capabilities` | All reasoners and skills. **Response shape:** `{capabilities: [{agent_id, reasoners: [{id, tags, ...}]}]}` — note `agent_id` not `node_id`, and reasoners live under `.capabilities[].reasoners[]` not `.reasoners[]`. The reasoner identifier field is `id` not `name` |
| `GET /api/v1/agentic/discover?q=<keyword>` | Search the API catalog by keyword |
| `POST /api/v1/execute/:target` | **Sync** execute. Body is `{"input": {...kwargs...}}`. **90-second hard timeout at the control plane.** |
| `POST /api/v1/execute/async/:target` | Async execute, returns an `execution_id` immediately. Use this when the pipeline > 90s |
| `GET /api/v1/executions/:id` | Status of an async execution |
| `GET /api/v1/did/workflow/:workflow_id/vc-chain` | Verifiable credential chain for an executed workflow (the AgentField superpower no other framework has) |

## Inspect the live workflow DAG

After running an execution, hit:

```bash
# Get the most recent executions
curl -s http://localhost:8080/api/v1/executions | jq '.[0:3]'

# Get the VC chain for one — this shows you the full reasoning DAG with cryptographic provenance
curl -s http://localhost:8080/api/v1/did/workflow/<workflow_id>/vc-chain | jq
```

This is the **single best demo** of why AgentField beats CrewAI: you get a cryptographic, replayable, introspectable record of every reasoner that ran, what it called, and what came back. Show the user this output in the handoff — it makes the "this is composite intelligence as production infrastructure" case for itself.

## The smoke-test contract (every build)

In the README, give the user EXACTLY these commands in this order. Do not abbreviate. Do not say "and so on."

```bash
# After docker compose up, in another terminal:

# 1. Health
curl -fsS http://localhost:8080/api/v1/health | jq '.status'

# 2. Node registered? (use ?health_status=any — default filter can hide healthy nodes)
curl -fsS 'http://localhost:8080/api/v1/nodes?health_status=any' | jq '.nodes[] | {id: .node_id, status: .status}'

# 3. Reasoners discoverable? (note .capabilities[].reasoners[].id, NOT .reasoners[].name)
curl -fsS http://localhost:8080/api/v1/discovery/capabilities \
  | jq '.capabilities[] | select(.agent_id=="<slug>") | .reasoners | map({id, tags})'

# 4. THE BIG ONE — run the entry reasoner with real data
#    Body shape: {"input": {...kwargs...}} — kwargs are NEVER raw at the top level
curl -X POST http://localhost:8080/api/v1/execute/<slug>.<entry_reasoner> \
  -H 'Content-Type: application/json' \
  -d '{"input": {"<kwarg1>": "<value>", "model": "openrouter/google/gemini-2.5-flash"}}' | jq

# 5. (Optional showpiece) the full verifiable workflow chain
LAST_EXEC=$(curl -s http://localhost:8080/api/v1/executions | jq -r '.[0].workflow_id')
curl -s http://localhost:8080/api/v1/did/workflow/$LAST_EXEC/vc-chain | jq
```

## When you cannot run docker locally

If the environment running the skill doesn't have Docker, you can still:

1. `python3 -m py_compile main.py` — catches syntax errors
2. `docker compose config` — catches compose errors
3. Read the generated files back with `cat` to spot obvious issues
4. Provide the verification commands in the README as a checklist for the user to run themselves

You **must** still validate the Python and the compose file syntactically. "I generated it but didn't check" is a failure mode.
