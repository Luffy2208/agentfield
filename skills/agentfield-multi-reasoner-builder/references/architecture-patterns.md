# Architecture Patterns — The 8 AgentField Compositions

These are battle-tested patterns from real AgentField systems (`sec-af`, `af-swe`, `contract-af`, `af-deep-research`, `reactive-atlas`). Pick one, compose two, or invent your own — but never default to a static linear chain.

For each pattern: when to use it, the shape, and a real-system reference.

---

## 1. Parallel Hunters + Signal Cascade

**Shape:**
```
input ──┬──> hunter_A ──┐
        ├──> hunter_B ──┼──> findings_pool ──> downstream
        ├──> hunter_C ──┘
        └──> hunter_D
```

**When:** Any problem with multiple independent analysis dimensions that can be examined concurrently. Each hunter is a specialist that knows about ONE dimension deeply.

**Reference:** `examples/sec-af/` — parallel strategy hunters analyzing SEC filings; `examples/contract-af/` — parallel clause analysts (IP / liability / non-compete / data / termination).

**Code shape:**
```python
@app.reasoner()
async def review(document: str) -> dict:
    findings = await asyncio.gather(*[
        app.call(f"{app.node_id}.{h}", document=document)
        for h in ["profitability_hunter", "liquidity_hunter", "risk_hunter", "efficiency_hunter"]
    ])
    return await app.call(f"{app.node_id}.synthesizer", findings=findings)
```

**Common mistake:** Making the hunters do "everything" each. Each hunter is a NARROW specialist. If hunters overlap heavily, you decomposed wrong.

---

## 2. HUNT → PROVE Adversarial Tension

**Shape:**
```
input ──> hunters ──> candidate findings ──> provers ──> verified findings
                                           ↑
                                           adversary tries to disprove each one
```

**When:** Any problem where false positives are catastrophic — security, legal, compliance, medical, financial.

**Reference:** `examples/sec-af/` — vulnerability hunters → exploit provers; `examples/contract-af/` — clause analysts → adversary reviewer.

**Why it works:** Hunters are biased toward sensitivity (find everything). Provers are biased toward specificity (refuse anything unproven). The tension between them is the intelligence — neither alone produces a good answer.

```python
@app.reasoner()
async def adversarial_review(input: str) -> dict:
    candidates = await app.call(f"{app.node_id}.hunter_pool", input=input)
    verified = await asyncio.gather(*[
        app.call(f"{app.node_id}.prover", finding=f, original=input)
        for f in candidates
    ])
    return [v for v in verified if v["proven"]]
```

---

## 3. Streaming Pipeline (asyncio.Queue)

**Shape:**
```
upstream ──emits──> queue ──consumes──> downstream
                                          (starts working before upstream finishes)
```

**When:** Downstream reasoners can start working on partial results — and waiting for the full upstream batch wastes time and misses interaction effects.

**Reference:** `examples/sec-af/` — HUNT→PROVE streaming; `examples/contract-af/` — analysts → cross-reference + adversary streaming.

```python
findings_queue = asyncio.Queue()

async def producer(items):
    for item in items:
        finding = await app.call(f"{app.node_id}.analyze", item=item)
        await findings_queue.put(finding)
    await findings_queue.put(None)  # sentinel

async def consumer():
    seen = []
    while (f := await findings_queue.get()) is not None:
        # Check this finding against everything seen so far
        await app.call(f"{app.node_id}.cross_ref", new=f, prior=seen)
        seen.append(f)
```

---

## 4. Meta-Prompting (Harnesses Spawning Harnesses)

**Shape:**
```
parent_harness ──discovers X──> crafts a SPECIFIC prompt ──spawns──> child_harness ──> findings
       ↑                                                                                    │
       └────────────────── integrates findings ─────────────────────────────────────────────┘
```

**When:** The investigation path depends on what gets discovered. You cannot pre-define which sub-reasoners will run, because you don't know yet what's there.

**Reference:** `examples/contract-af/` — clause analysts spawning definition-impact analyzers when they discover a referenced defined term; cross-reference resolver spawning combination deep-dives.

**This is the pattern that no framework chain can replicate.** It's pure dynamic intelligence.

```python
@app.reasoner()
async def clause_analyst(clause: str, context: str) -> dict:
    initial = await app.harness(
        goal=f"Analyze this clause: {clause}",
        tools=["read_section", "lookup_definition"],
        max_iterations=10,
    )

    # The harness discovered a defined term that needs deeper analysis.
    # Craft a SPECIFIC prompt for a child harness at runtime.
    if initial.discovered_terms:
        for term in initial.discovered_terms:
            sub_prompt = (
                f"You are analyzing the cascading impact of the defined term '{term}' "
                f"in the context of clause: {clause}. "
                f"Read every section that references '{term}' and determine if any "
                f"interaction creates risk. Return: affected_sections, risk_level, rationale."
            )
            sub = await app.call(
                f"{app.node_id}.term_impact_analyzer",
                prompt=sub_prompt,
                term=term,
            )
            initial.term_impacts.append(sub)
    return initial.model_dump()
```

**Hard rule:** every meta-spawn point has a depth cap.

---

## 5. Three Nested Control Loops (Inner / Middle / Outer)

**Shape:**

| Loop | Scope | Trigger | Cap |
|---|---|---|---|
| **Inner** | Per-reasoner self-adaptation | Found a reference, escalation needed | `max_follows=3`, `max_escalations=1` |
| **Middle** | Cross-reasoner deep-dives | Critical combination, hidden interaction | `max_spawns=5` |
| **Outer** | Pipeline-wide coverage | Coverage gate detects a gap | `max_iterations=3` |

**When:** Long-running analysis where you can't predict upfront how deep you need to go. Coverage matters and edge cases are dangerous.

**Reference:** `examples/af-swe/` — inner coding loop / middle sprint loop / outer factory loop; `examples/contract-af/` — analyst loop / cross-ref loop / coverage loop.

**Hard rule:** every loop has an absolute cap. "Keep going until confident" is how you get a $400 bug report.

---

## 6. Fan-Out → Filter → Gap-Find → Recurse

**Shape:**
```
seed ──> [generate N candidates] ──> [filter to top K] ──> [gap analysis]
                                                                │
                                                                ├─ gaps found ──> recurse with new seeds
                                                                └─ no gaps    ──> done
```

**When:** Comprehensive coverage problems where you don't know the shape of the answer upfront — research, due diligence, audits, literature reviews.

**Reference:** `examples/af-deep-research/` — recursive research with quality-driven loops.

```python
@app.reasoner()
async def deep_research(question: str, max_rounds: int = 3) -> dict:
    seeds = [question]
    all_findings = []
    for round in range(max_rounds):
        findings = await asyncio.gather(*[
            app.call(f"{app.node_id}.investigator", seed=s) for s in seeds
        ])
        all_findings.extend(findings)
        gaps = await app.call(f"{app.node_id}.gap_finder", findings=all_findings, original=question)
        if not gaps.gaps:
            break
        seeds = gaps.gaps  # next round's seeds
    return await app.call(f"{app.node_id}.synthesizer", findings=all_findings)
```

---

## 7. Factory Control Loops

**Shape:** Three nested loops for long-running multi-step execution with adaptive replanning.

```
outer (factory)  ──> sprint planner   ──> goals
middle (sprint)  ──> task executor    ──> tasks
inner (coding)   ──> per-task agent   ──> code
                              │
                              └─ fails ──> outer replan
```

**When:** Multi-step execution that needs to replan based on intermediate results — code generation, document production, migration execution, multi-step research.

**Reference:** `examples/af-swe/`.

---

## 8. Reasoner Composition Cascade (READ THIS — it's the master pattern)

**This is the pattern that distinguishes a real AgentField system from a fancy `asyncio.gather` wrapper.** Every other pattern in this file should be interpreted through this lens.

**Shape — depth, not breadth:**

```
entry_reasoner
├── classifier_reasoner ─────────────────┐
│   ├── input_normalizer (skill)         │
│   └── intent_extractor (.ai)           │
│       └── slot_filler (.ai called by intent_extractor when ambiguous)
│
├── analysis_dimension_A_reasoner ────── ┤  ← all parallel via asyncio.gather
│   ├── deterministic_metric_calc (skill)
│   ├── pattern_judge (.ai)
│   │   └── citation_finder (.ai called by pattern_judge)
│   └── confidence_scorer (.ai)
│
├── analysis_dimension_B_reasoner ────── │
│   ├── different_metric_calc (skill)
│   ├── different_pattern_judge (.ai)
│   └── confidence_scorer (.ai REUSED — same reasoner)  ← reuse across branches!
│
├── analysis_dimension_C_reasoner ───────┤
│   └── (3 sub-calls, similar shape)
│
└── adversarial_synthesizer ─────────────┘
    ├── steel_man_alternative (.ai)  ← called once per dimension
    ├── disagreement_detector (.ai)
    └── final_decision_reasoner (.ai)
        └── safety_override (deterministic skill)
```

**Each layer fans out via `asyncio.gather`. Each reasoner has a single cognitive responsibility.** The orchestrator at the top is NOT the only thing that calls `app.call` — every dimension reasoner is itself a small orchestrator that calls 2–4 sub-reasoners.

**Used in:** This is the pattern the medical-triage and loan-underwriter examples should follow when they're deep enough. Most large AgentField systems compose this pattern as the backbone, with the other 8 patterns layered on top (HUNT→PROVE between layers, streaming for partial results, etc.).

**Why it's the master pattern:**

1. **Reasoners as software APIs.** Each reasoner has a one-line API contract: *"Given X, return Y. Calls Z, W."* Other reasoners call it the way one microservice calls another.
2. **Composability over monolithic prompts.** A specialist reasoner like `pe_assessor` is NOT a 200-line `.ai()` prompt — it's an orchestrator that calls `wells_score_calculator`, `dyspnea_grader`, and `dvt_history_checker` and synthesizes their outputs. Each piece is testable, replaceable, reusable.
3. **Reuse across branches.** `confidence_scorer` is called from THREE different dimension reasoners. The flat-star pattern would have to copy-paste the logic three times. The composition cascade calls it once per branch — same code, three different contexts.
4. **Multi-layer parallelism.** `asyncio.gather` runs at the entry-reasoner layer (across dimensions A/B/C) AND inside each dimension reasoner (across its sub-calls). Total wall-clock time is dominated by the slowest path through the DAG, not by the sum.
5. **Observability has structure.** The control plane workflow DAG shows the actual call tree. The verifiable credential chain has hierarchy. A future debugger can ask "which sub-call inside `pe_assessor` flagged the concern" — the flat-star pattern can only tell you "pe_assessor returned X."
6. **Each reasoner is independently curl-able.** You can `POST /api/v1/execute/<slug>.wells_score_calculator` directly with synthetic input to debug or A/B test it. The flat-star only exposes the entry reasoner.

**Decomposition rules:**

- **30-line ceiling.** If a reasoner body is > 30 lines, it's probably 2 reasoners. Look for the seam — usually a "compute X then judge Y" boundary becomes "X is a `@app.skill`, Y is a `@app.reasoner` that calls X".
- **Single-judgment rule.** A reasoner makes ONE judgment call. If your reasoner is making three judgments ("is this concerning, is this acute, what's the risk score"), split into three reasoners.
- **Deterministic-vs-judgment split.** Anything that doesn't require LLM judgment (math, formula, regex, lookup, sort) is `@app.skill()` or a plain helper, not part of an `.ai()` reasoner body.
- **Reuse signal.** If the same logic appears in 2+ reasoners, extract it as its own reasoner and call it from both.
- **One-sentence API contract test.** Can you write a one-sentence contract for each reasoner ("Given a chief complaint string, return a list of red flag categories with confidence scores")? If not, the reasoner is doing too many things.

**Anti-patterns that mean you fell back to a flat star:**

- Your entry reasoner is the ONLY thing that calls `app.call`
- Your specialists each have a single fat `.ai()` call with a 500-token prompt
- Your DAG is depth 2 (`entry → specialists → done`)
- You can draw the architecture as a literal asterisk
- Two specialists have the same 50-line prompt with one line different — you should have had one parameterized sub-reasoner

**Concrete medical-triage example:**

A flat-star `red_flag_detector` reasoner with one big `.ai()` prompt → bad.

A `red_flag_detector` reasoner that calls `cardiac_red_flag_checker`, `stroke_red_flag_checker`, `bleeding_red_flag_checker`, `psych_red_flag_checker` in parallel via `asyncio.gather`, each of which is itself a focused `.ai()` with its own narrow prompt and confidence flag → good. The deeper structure means a future agent can swap the cardiac checker for one with a more accurate prompt without touching anything else.

**When you finish your design, count the depth.** If max depth from entry to leaf is < 3, redesign. A real composite-intelligence system has at least 3 layers of reasoner-calling-reasoner.

---

## 9. Reactive Document Enrichment

**Shape:**
```
event source (DB change stream / webhook) ──> enrichment pipeline ──> output
```

**When:** Work is triggered by data arriving — incidents, PRs, contracts on upload, form submissions, telemetry events.

**Reference:** `examples/reactive-atlas/` — MongoDB change streams → enrichment agents.

**The point:** the engine is domain-agnostic; the config defines the domain. The same pattern handles "new contract uploaded → enrich → score → route" as it handles "new incident filed → triage → assign → notify".

---

## How to pick a pattern (or compose your own)

**Always start with pattern 8 (Reasoner Composition Cascade) as the backbone.** It's not optional. Every other pattern is layered on top.

Then ask:

1. **What triggers the work?** Event stream → pattern 9 (reactive enrichment). Direct API call → patterns 1–7 layered onto 8.
2. **Is the input large/navigable?** Yes → consider meta-prompting (pattern 4) inside one of your dimension reasoners.
3. **Multiple independent analysis dimensions?** Yes → parallel hunters (pattern 1) becomes the second-layer fan-out inside the cascade.
4. **False positives expensive?** Yes → add HUNT→PROVE (pattern 2) as a second-stage reasoner per dimension or one global adversarial reasoner.
5. **Downstream can start before upstream finishes?** Yes → streaming (pattern 3).
6. **Coverage matters and you can't predict shape upfront?** Pattern 6.
7. **Multi-round adaptive execution?** Pattern 5 or 7.
8. **The investigation path depends on discoveries?** Pattern 4 (meta-prompting), always.

Most strong systems compose **pattern 8 (cascade) as the backbone + 2–3 of the others as layers**. Example: contract-af = composition cascade (8) + parallel hunters (1) at the second layer + HUNT→PROVE (2) at the third layer + streaming (3) between layers + meta-prompting (4) inside the deepest reasoners + nested loops (5).

## When NONE of these fit

Then the use case probably doesn't justify AgentField at all — it's a one-shot LLM call wearing a costume. Tell the user honestly.
