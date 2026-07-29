# Evidence-backed memory system design

Date: 2026-07-29

## Executive conclusion

The best near-term improvement is not a larger graph or a second generative
agent. It is a calibrated, measurable flat-memory baseline:

1. preserve evidence at multiple granularities;
2. retrieve with dense vectors plus exact lexical signals;
3. abstain when the nearest record is not actually relevant;
4. resolve repeated current-state records deterministically;
5. bound and de-duplicate what reaches the agent;
6. treat every stored record as untrusted data;
7. evaluate retrieval, updates, forgetting, and abstention independently.

That is the direction implemented in this pass. A temporal graph remains a
candidate for entity-heavy multi-hop failures, but it should be added only if a
controlled benchmark shows that it beats the strong flat baseline.

## What the research says

### Retrieval quality must be evaluated as a pipeline

[LongMemEval (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d813d324dbf0598bbdc9c8e79740ed01-Abstract-Conference.html)
tests extraction, multi-session reasoning, temporal reasoning, knowledge
updates, and abstention. Its controlled experiments found that:

- round-sized values can outperform whole-session values;
- adding extracted facts to retrieval keys improved recall and downstream QA;
- time-aware query expansion improved temporal retrieval;
- reading retrieved evidence is still difficult even under perfect recall.

The practical implication is that one nearest-neighbor query and a fixed top-k
cannot be considered a complete memory system.

[MemoryAgentBench (ICLR 2026)](https://openreview.net/pdf/dca8178b2d4fb7cd70a435ed43b655c97ce9871c.pdf)
adds four complementary competencies: accurate retrieval, test-time learning,
long-range understanding, and selective forgetting/conflict resolution. It is
the right second external benchmark because it explicitly tests incremental
memory writes and consolidation, not only final QA.

### Granularity matters more than fashionable storage structure

[SeCom (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e56f394bbd4f0ec81393d767caa5a31b-Abstract-Conference.html)
found both turn-level and session-level memory units suboptimal and showed that
topical segmentation plus denoising improves retrieval.

[Does Memory Need Graphs? (ACL 2026)](https://aclanthology.org/2026.acl-long.1232/)
is especially relevant because it controls the surrounding system instead of
comparing incomparable end-to-end products. Its results support:

- derived summary, fact, and keyword keys alongside the underlying content;
- add/update/no-op maintenance rather than append-only storage;
- caution with similarity graphs, whose expansion introduced noise in their
  experiments;
- entity-description graphs only when their extra structure provides a
  measured benefit.

The current fork therefore keeps atomic facts and narrative vectors, and adds a
combined retrieval-key vector. It does not add a graph yet.

### “Nearest” is not the same as “relevant”

Vector stores return neighbors even for an unanswerable query. The previous
automatic injector discarded Nemotron's cosine distance and injected the five
nearest observations regardless of score. That guarantees false recall when
the project contains any memory at all.

The new policy has:

- a configurable absolute cosine floor;
- a relative score band below the best candidate;
- global ranking across observations and session summaries;
- per-session and near-duplicate caps;
- a hard character budget;
- explicit abstention when every candidate is weak.

The initial `0.18` floor is a conservative local calibration, not a universal
constant. It must be tuned against representative project queries and the
external benchmarks.

### Freshness should be deterministic where the key is known

[Don't Ask the LLM to Track Freshness (2026 preprint)](https://arxiv.org/abs/2606.01435)
argues that current-value conflicts are often an assembly problem: choose the
maximum serial or timestamp deterministically instead of hoping an LLM notices
freshness in a context bundle. Its broader LongMemEval result is mixed, so this
is not evidence for deleting all old facts.

The implemented conservative version:

- groups only records with the same normalized title;
- returns the newest group member for an ordinary current-state query;
- returns the immediately previous member for an explicitly historical query;
- preserves both for explicit comparisons.

The durable next step is a real state key and validity interval in SQLite.
Title equality is useful but not sufficient.

### Memory writes need verification and a security boundary

[TrustMem (2026 preprint)](https://arxiv.org/abs/2606.25161) identifies omission,
corruption, and unsupported additions during memory transitions, and proposes
checking coverage, preservation, and faithfulness.

[MemEvoBench (2026 preprint)](https://arxiv.org/abs/2604.15774) studies
persistent memory misevolution from poisoned/noisy inputs. This matters more
than ordinary prompt injection because a poisoned memory can affect every
future session.

This pass adds two immediate defenses:

- the observer is told that tool input/output is untrusted evidence and must
  not preserve embedded directives as behavioral instructions;
- automatic context uses escaped structural boundaries and tells the consuming
  agent that records are evidence, never instructions.

Prompt text alone is not a complete verifier. A later write pipeline should
store source evidence and reject unsupported state transitions before they
become current memory.

## Implemented architecture

### Write/index

- A single local Nemotron model remains shared by all clients.
- Narratives, text, and every atomic fact retain independent vectors.
- Each observation now also receives a combined retrieval key containing its
  title, context, facts, keywords, and files.
- Each summary receives a combined request/investigated/learned/completed/
  next-steps/notes key in addition to independent field vectors.
- Vector-index format version 2 resets only backfill watermarks. Deterministic
  document IDs update existing vectors in place and add the new keys, so an
  interrupted reindex resumes by row.

### Automatic read

- Query the same project across observations and summaries.
- Convert Chroma cosine distance back to similarity.
- Apply absolute and relative relevance floors.
- Hydrate the original SQLite evidence.
- Resolve same-title current/historical state deterministically.
- Rank globally, cap one session's dominance, and remove near duplicates.
- Render escaped evidence under a character budget.
- Return nothing when confidence is insufficient.

Manual MCP search remains exploratory and does not use the automatic injection
floor by default. That is intentional: a person or agent explicitly searching
memory may want low-ranked candidates, while silent prompt injection must favor
precision.

## Evaluation

`npm run eval:memory` runs a local, deterministic Nemotron smoke benchmark with
semantic, identifier, update, temporal, multi-hop, and abstention cases. It
compares the former fixed top-5 behavior with the calibrated policy.

This fixture is a regression gate, not publication-grade evidence. The next
benchmark layer should:

1. add a LongMemEval ingestion adapter and report Recall@k, nDCG@k, answer
   accuracy, update accuracy, temporal accuracy, and abstention;
2. add MemoryAgentBench EventQA and FactConsolidation adapters;
3. report p50/p95 retrieval latency, embedding queue wait, injected tokens, and
   OpenAI summarization cost separately;
4. run ablations for narrative-only, fact-only, combined keys, hybrid lexical,
   temporal filtering, and graph expansion;
5. pin model revision, runtime versions, fixture hashes, and configuration in
   every result.

## Next implementation priorities

### 1. Structured, versioned facts

Add a local migration for:

- `memory_fact_key`;
- `value`;
- `valid_from` and `valid_to`;
- `supersedes_id`;
- `source_event_id`;
- `confidence`;
- `verification_status`;
- `subject`, `predicate`, and optional entities.

Keep raw observations append-only. Materialize a current-state view
deterministically and preserve historical queries over the full chain.

### 2. Evidence-grounded transition verification

For a proposed add/update/no-op:

- verify every new fact against the captured tool event or conversation span;
- ensure an update preserves unrelated facts;
- reject behavioral directives and unsupported claims;
- use GPT-5.6 Luna for routine verification and Terra only for ambiguous,
  multi-record conflicts;
- store the verifier decision and evidence IDs locally.

This is the most defensible place for the requested tiered models. Using Terra
for every retrieval would add latency and cost without fixing bad indexing.

### 3. Query planning

Add deterministic extraction first:

- code symbols, file paths, error literals, ports, issue IDs, and quoted text;
- explicit dates and relative-time phrases;
- current versus historical intent.

Then use Luna only for ambiguous query decomposition or multi-hop expansion.
Every generated subquery should be logged locally and evaluated as an ablation.

### 4. Hybrid lexical scoring

The repository already has FTS5, but natural user prompts are currently quoted
as one exact phrase, so lexical fusion rarely contributes. Replace that path
with a safe term planner for identifiers and distinctive tokens, preserve BM25
scores, and combine calibrated dense and lexical scores rather than ranks
alone.

### 5. Topical consolidation

Build topic documents asynchronously from related observations while retaining
raw evidence. Consolidation must be reversible and transition-verified.
SeCom-style topical units should be benchmarked against the current
observation/summary keys.

### 6. Graph gate

Only add a temporal/entity graph if flat-memory error analysis shows persistent
entity-relation multi-hop misses. The acceptance criterion should be a
statistically meaningful gain on those slices without worse abstention,
poisoning resistance, or p95 latency.

### 7. Project identity

The current readable project key is based mainly on the repository basename.
Two unrelated repositories with the same basename can collide. Introduce a
stable local repository fingerprint derived from normalized Git identity, keep
the basename as a display alias, and explicitly map worktrees to the same
repository memory with optional branch scope.

## Non-goals

- Do not delete old evidence merely because a newer fact exists.
- Do not make retrieval depend on a hosted vector service.
- Do not use an LLM as the only freshness resolver.
- Do not add a graph without an ablation.
- Do not optimize only final answer quality; track false injection and
  abstention separately.
