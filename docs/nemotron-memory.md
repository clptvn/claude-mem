# Shared Nemotron Memory

## Outcome

Claude Code and Codex use the same resident embedding model, queue, and vector
database. Starting five chats does not load five copies of the model. The
existing Claude-Mem hooks still capture tool activity, generate structured
observations and session summaries, and inject relevant memories. Memories are
project-scoped rather than client-scoped: Claude and Codex sessions in the same
Git checkout read the same corpus, while `platform_source` remains attached as
provenance and can still be used as an explicit search filter. Only the
embedding and vector-search layer is replaced.

The generation model and embedding model have distinct jobs:

- Claude-Mem's configured language-model provider turns raw session activity
  into compact observations and summaries.
- `nvidia/Nemotron-3-Embed-1B-BF16` maps those memories and future queries into
  retrieval vectors. It does not itself write summaries.

## Architecture

```text
Claude Code sessions ----\
                          +--> Claude-Mem worker --> structured SQLite memory
Codex sessions ----------/             |
                                       | vector tool contract
                                       v
                         localhost:37901 Nemotron service
                         +-----------------------------+
                         | durable vector-write queue  |
                         | shared embedding batcher    |
                         | one MPS model instance      |
                         | Chroma persistent client    |
                         +-----------------------------+
                                       |
                                       v
                         semantic + FTS5 rank fusion
                                       |
                                       v
                         automatic per-prompt context
```

The service binds only to `127.0.0.1`. It is managed by the
`ai.claude-mem.nemotron` user LaunchAgent and starts at login.

## Storage and recovery

| Path | Purpose |
| --- | --- |
| `~/.claude-mem/claude-mem.db` | Canonical sessions, prompts, observations, and summaries |
| `~/.claude-mem/nemotron/chroma/` | Nemotron vector index |
| `~/.claude-mem/nemotron/jobs.sqlite3` | Durable vector-write queue |
| `~/.claude-mem/nemotron-runtime/` | Isolated Python environment and service copy |
| `~/.claude-mem/logs/nemotron-service*.log` | LaunchAgent stdout and stderr |
| `~/Library/LaunchAgents/ai.claude-mem.nemotron.plist` | Login service definition |

Every add, update, or delete reaches the SQLite write queue before Chroma.
Interrupted `processing` jobs become `pending` on the next startup and retry up
to three times. Claude-Mem's deterministic document IDs and sync watermarks
provide a second recovery layer: incomplete observations are backfilled from
the canonical SQLite database.

## Retrieval behavior

- Passage and query requests use the checkpoint's separate
  `encode_document`/`encode_query` paths.
- Embeddings are L2 normalized, converted to float32 for Chroma, and compared
  with cosine distance.
- Concurrent requests are coalesced for a short configurable batching window.
- Exact FTS5 matches and semantic matches are combined with reciprocal-rank
  fusion.
- Recency is a soft tie-breaker. Relevant memories do not expire after an
  arbitrary 90-day cutoff.
- `CLAUDE_MEM_SEMANTIC_INJECT=true` retrieves relevant project memories on each
  user prompt. Normal MCP `search`, `timeline`, and `get_observations` tools
  remain available for deliberate progressive disclosure.
- If Nemotron is unavailable, Claude-Mem falls back to SQLite/FTS5 and never
  blocks the host chat.

## Install and operate

From this fork's checkout, build and register the final plugin bundle in both
hosts:

```bash
bun install
npm run build
npm run nemotron:install
node dist/npx-cli/index.js install --ide claude-code --provider codex --runtime worker --no-auto-start --disable-auto-memory
node dist/npx-cli/index.js install --ide codex-cli --provider codex --runtime worker --no-auto-start --disable-auto-memory
bun plugin/scripts/worker-service.cjs start
```

Restart already-open Claude Code and Codex sessions afterward so they load the
new native hooks. Future sessions capture and recall automatically.

For day-to-day service checks:

```bash
npm run nemotron:smoke
npm run nemotron:status
```

The install command copies the service into a stable per-user runtime, creates
or reuses a Python 3.11 virtual environment with `uv`, registers the
LaunchAgent, downloads the model through Hugging Face on first use, waits for
readiness, and atomically enables Nemotron plus semantic injection in
`~/.claude-mem/settings.json`. It also records an explicit Claude-Mem telemetry
opt-out in `~/.claude-mem/telemetry.json` and starts Chroma with anonymous
product telemetry disabled.

```bash
npm run nemotron:stop
npm run nemotron:start
npm run nemotron:uninstall
```

Uninstalling the LaunchAgent switches the embedding provider back to upstream
Chroma but deliberately preserves the model runtime, memory database, vector
index, and queue. This avoids accidental data loss.

## Configuration

Set service environment variables in the shell that runs
`npm run nemotron:install`; the installer persists them into the LaunchAgent.

| Variable | Default | Meaning |
| --- | --- | --- |
| `CLAUDE_MEM_NEMOTRON_MODEL` | `nvidia/Nemotron-3-Embed-1B-BF16` | Hugging Face model ID |
| `CLAUDE_MEM_NEMOTRON_PORT` | `37901` | Local service port |
| `CLAUDE_MEM_NEMOTRON_MAX_TOKENS` | `4096` | Per-input cap, configurable up to the checkpoint's 32,768 |
| `CLAUDE_MEM_NEMOTRON_BATCH_SIZE` | `8` | Inference batch size |
| `CLAUDE_MEM_NEMOTRON_QUEUE_BATCH_SIZE` | `16` | Cross-client request group |
| `CLAUDE_MEM_NEMOTRON_BATCH_WAIT_MS` | `12` | Coalescing window |
| `CLAUDE_MEM_NEMOTRON_WRITE_TIMEOUT_SECONDS` | `180` | Synchronous write wait |
| `CLAUDE_MEM_NEMOTRON_EAGER_LOAD` | `true` | Load the model when the service starts |

Claude-Mem settings live in `~/.claude-mem/settings.json`:

| Setting | Value written by installer |
| --- | --- |
| `CLAUDE_MEM_EMBEDDING_PROVIDER` | `nemotron` |
| `CLAUDE_MEM_NEMOTRON_URL` | `http://127.0.0.1:37901` |
| `CLAUDE_MEM_NEMOTRON_AUTO_START` | `true` |
| `CLAUDE_MEM_NEMOTRON_REQUEST_TIMEOUT_MS` | `180000` |
| `CLAUDE_MEM_SEMANTIC_INJECT` | `true` |
| `CLAUDE_MEM_PROVIDER` | `codex` |
| `CLAUDE_MEM_MODEL` | `gpt-5.6-luna` |
| `CLAUDE_MEM_TIER_SIMPLE_MODEL` | `gpt-5.6-luna` |
| `CLAUDE_MEM_TIER_SUMMARY_MODEL` | `gpt-5.6-luna` |
| `CLAUDE_MEM_TIER_FAST_MODEL` | `gpt-5.6-luna` |
| `CLAUDE_MEM_TIER_SMART_MODEL` | `gpt-5.6-terra` |

Set `CLAUDE_MEM_EMBEDDING_PROVIDER=chroma` to restore the upstream
`chroma-mcp` MiniLM path.

## Network boundary

Memory databases, generated observations, embeddings, vector searches, and the
web viewer remain local. Cloud sync is off unless its URL and credentials are
explicitly configured. Claude-Mem analytics and Chroma product telemetry are
disabled by the installer.

The configured Codex provider sends session activity to OpenAI through your
logged-in Codex CLI to generate compact observations and summaries. Its
subprocesses are ephemeral, read-only, receive prompts through stdin, ignore
repo/user Codex configuration and rules, and have web search disabled. Hugging
Face remains reachable so the Nemotron checkpoint can be downloaded and its
cache checked; package managers and GitHub are contacted only when installing
or updating dependencies/source.

## Mac resource profile

The checked configuration uses FP16 on PyTorch MPS. On an M2 Max with 32 GB of
unified memory, the downloaded checkpoint is about 2.1 GB and the isolated
Python environment is about 1 GB. Actual process and Metal allocation varies
with input length and batch size. The conservative 4,096-token default avoids
the quadratic attention cost of embedding a full 32K record while remaining
far above MiniLM-sized memory chunks.

## Model terms

Claude-Mem source remains Apache-2.0. NVIDIA distributes the Nemotron checkpoint
separately under OpenMDW 1.1, and the installer downloads it directly from
Hugging Face rather than redistributing weights in this repository.
