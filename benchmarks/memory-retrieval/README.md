# Memory retrieval evaluation

This is a small, deterministic smoke benchmark for the local Nemotron retrieval
path. It is not a replacement for LongMemEval or MemoryAgentBench. It catches
the regressions that matter most for this fork before running a large external
benchmark:

- semantic paraphrase and code-symbol retrieval;
- current versus historical state;
- abstention when no stored memory answers the query;
- context volume from automatic injection.

Run it against the shared local service:

```bash
npm run eval:memory
```

Override the service or calibrated cosine floor when measuring a candidate
configuration:

```bash
CLAUDE_MEM_NEMOTRON_URL=http://127.0.0.1:37901 \
CLAUDE_MEM_EVAL_MIN_SIMILARITY=0.18 \
npm run eval:memory
```

The fixed top-5 row represents the previous automatic-injection policy: it
always injected the nearest memories, even when every match was unrelated. The
calibrated row uses the same core policy as `SearchManager.retrieveContext`.
Keep the fixture small enough for local iteration, and validate major retrieval
changes on LongMemEval and MemoryAgentBench before treating them as proven.
