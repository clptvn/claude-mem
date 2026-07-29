# Local memory benchmark suite

This suite measures the system in layers so one aggregate score cannot hide a
failure:

1. **Embedding retrieval**: raw Nemotron ranking with Recall-any/all, nDCG,
   MRR, and precision.
2. **Automatic injection policy**: abstention, current-versus-stale state,
   duplicate suppression, session diversity, and injected context volume.
3. **Local system behavior**: model/worker readiness, telemetry-off state,
   durable vector writes, semantic ordering, queue concurrency, and cleanup.
4. **Memory generation**: opt-in Codex observation/summary XML validity,
   factual coverage, hallucination checks, and prompt-poisoning resistance.
5. **End-to-end QA**: retrieved evidence, Luna answers, and an optional Terra
   correctness judge.
6. **External protocols**: LongMemEval retrieval plus MemoryAgentBench task
   export and deterministic scoring.

The fast smoke corpus is a regression fixture, not evidence that the memory
system is generally solved.

## Quick start

With the shared local worker and Nemotron service running:

```bash
npm run bench:memory
npm run bench:memory:system
```

The smoke command runs seven ablations and fails when the production policy
drops below checked-in gates:

- raw vector ranking;
- legacy fixed top-5 injection;
- threshold only;
- absolute plus relative confidence window;
- calibrated policy without freshness;
- calibrated policy without deduplication;
- the exact policy imported by production `SearchManager`.

Every run writes a timestamped JSON artifact under `runs/`. Embeddings are
cached in a benchmark-only SQLite database under `cache/`. Both paths are
ignored by git.

## Official LongMemEval retrieval

Downloads are always explicit:

```bash
bun benchmarks/memory-retrieval/cli.ts download longmemeval-oracle
bun benchmarks/memory-retrieval/cli.ts run \
  --dataset longmemeval-oracle \
  --granularity session \
  --limit 25
```

Remove `--limit` for all 500 instances. Turn retrieval supports the official
cutoffs through 50:

```bash
bun benchmarks/memory-retrieval/cli.ts run \
  --dataset longmemeval-oracle \
  --granularity turn
```

The adapter follows the official flat-index construction: session documents
contain user turns, turn documents contain individual user turns, and each
query is restricted to its own haystack. It reports the official
`recall_all@5/10` and `ndcg@5/10` family, with `@50` for turn retrieval, plus
additional diagnostic metrics. The suite does not call an LLM judge during
retrieval evaluation.

For an opt-in end-to-end pass, run:

```bash
bun benchmarks/memory-retrieval/cli.ts qa \
  --dataset longmemeval-s \
  --limit 3
```

This uses `gpt-5.6-luna` for answers and `gpt-5.6-terra` as the higher-tier
judge. It imports the same calibrated selections produced by the retrieval
run, enforces a 12,000-character context budget, and records answers, evidence
IDs, token usage, latency, and category accuracy. The local judge follows
LongMemEval's temporal and knowledge-update guidance but is deliberately
labeled non-official; use the upstream evaluator for leaderboard comparison.

## MemoryAgentBench

MemoryAgentBench evaluates Accurate Retrieval, Test-Time Learning, Long-Range
Understanding, and Conflict Resolution using an “inject once, query many”
protocol. Downloading all four official parquet splits is explicit:

```bash
bun benchmarks/memory-retrieval/cli.ts download memoryagentbench
bun benchmarks/memory-retrieval/cli.ts mab:prepare accurate --limit 2
```

The converter requires `pyarrow`:

```bash
python3 -m pip install pyarrow
```

It emits portable JSONL records with `context`, `questions`, `answers`, source,
and metadata. A system runner should ingest each context once, answer every
question without resetting memory, and write one JSONL row per answer:

```json
{"task_id":"eventqa_full:0","question_index":0,"prediction":"..."}
```

Score deterministic metrics with:

```bash
bun benchmarks/memory-retrieval/cli.ts mab:score \
  --tasks benchmarks/memory-retrieval/data/memoryagentbench/Accurate_Retrieval.jsonl \
  --predictions /absolute/path/to/predictions.jsonl
```

The local scorer provides normalized exact match, substring match, token F1,
and EventQA all-facts recall. LongMemEval and InfBench subsets still require
their official LLM judges for leaderboard-comparable end-to-end QA results;
the artifact labels local scores as deterministic only.

## Generation quality

Generation tests invoke Codex deliberately and are therefore not part of the
default local-only retrieval command:

```bash
npm run bench:memory:generation
bun benchmarks/memory-retrieval/cli.ts generation \
  --model gpt-5.6-terra \
  --reasoning-effort low
```

The default model is `gpt-5.6-luna`. The runner uses the production prompt
builders, XML parser, and Codex CLI sandbox. Web search and telemetry
environment variables are disabled by the same production runner used by the
worker.

## Local system conformance and privacy

`system` creates one uniquely named `cm__benchmark_<pid>-<timestamp>` Chroma
collection, writes two synthetic records, verifies ranking, and deletes that
exact collection in a `finally` block. It never opens the real SQLite database
or production `cm__claude-mem` collection.

External traffic is limited to:

- Hugging Face when the user explicitly runs `download`;
- Codex/OpenAI when the user explicitly runs `generation`.

Retrieval, caches, Chroma data, benchmark inputs, and run artifacts remain
local. The conformance test fails if Chroma anonymized telemetry is enabled.

## Research basis and limits

- [LongMemEval](https://github.com/xiaowu0162/longmemeval) defines the
  extraction, multi-session, temporal, knowledge-update, and abstention
  categories and its official retrieval metrics.
- [LongMemEval (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d813d324dbf0598bbdc9c8e79740ed01-Abstract-Conference.html)
  motivates evaluating both memory retrievers and final answers.
- [MemoryAgentBench (ICLR 2026)](https://github.com/HUST-AI-HYZ/MemoryAgentBench)
  motivates separating four memory competencies instead of collapsing them
  into one conversational QA score.
- [LongMemEval-V2](https://github.com/xiaowu0162/LongMemEval-V2) adds 451
  multimodal web-agent questions over trajectories up to 115 million tokens.
  Its May 2026 paper is a recent preprint. A full V2 run is intentionally not
  auto-downloaded or attempted on a 32 GB laptop; its `insert(trajectory)` and
  `query(...)` backend contract is the next external harness integration.
- [MemoryArena](https://memoryarena.github.io/) extends evaluation into
  interactive agent environments and is a future system-level target.

Benchmark artifacts record dataset hash/version, git state, exact policy
configuration, embedding service health, per-query selections, latency
percentiles, and per-category metrics. Compare artifacts produced from the same
dataset revision, granularity, and machine state.
