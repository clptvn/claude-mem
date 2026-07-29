import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import {
  downloadDataset,
  loadLongMemEval,
  loadSmokeDataset,
  resolveDownloadedLongMemEval,
} from './datasets.js';
import { evaluateSmokeGates } from './gates.js';
import { runGenerationCheck } from './generation-check.js';
import {
  loadJsonLines,
  scoreMemoryAgentBench,
  type MemoryAgentBenchPrediction,
  type MemoryAgentBenchTask,
} from './memoryagentbench.js';
import { runRetrievalBenchmark } from './runner.js';
import { runSystemCheck } from './system-check.js';
import { runQaCheck } from './qa-check.js';

const benchmarkDir = import.meta.dir;
const dataDir = join(benchmarkDir, 'data');
const cachePath = join(benchmarkDir, 'cache', 'embeddings.sqlite');
const runsDir = join(benchmarkDir, 'runs');
const serviceUrl = process.env.CLAUDE_MEM_NEMOTRON_URL ?? 'http://127.0.0.1:37901';
const workerUrl = process.env.CLAUDE_MEM_WORKER_URL ?? 'http://127.0.0.1:37701';

function option(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

function numberOption(name: string): number | undefined {
  const value = option(name);
  if (value === undefined) return undefined;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) throw new Error(`${name} must be a number`);
  return parsed;
}

function hasFlag(name: string): boolean {
  return process.argv.includes(name);
}

function requirePath(path: string, hint: string): string {
  if (!existsSync(path)) throw new Error(`Missing ${path}. ${hint}`);
  return path;
}

async function runRetrieval(datasetName: string): Promise<void> {
  const limit = numberOption('--limit');
  const granularity = (option('--granularity') ?? 'session') as 'session' | 'turn';
  if (!['session', 'turn'].includes(granularity)) {
    throw new Error('--granularity must be session or turn');
  }
  const dataset = datasetName === 'smoke'
    ? loadSmokeDataset()
    : loadLongMemEval(
        requirePath(
          resolveDownloadedLongMemEval(
            dataDir,
            datasetName as 'longmemeval-oracle' | 'longmemeval-s',
          ),
          `Run "bun benchmarks/memory-retrieval/cli.ts download ${datasetName}" first.`,
        ),
        { granularity, limit },
      );
  const result = await runRetrievalBenchmark(dataset, {
    serviceUrl,
    cachePath,
    runsDir,
    minimumSimilarity: numberOption('--minimum-similarity'),
    relativeBand: numberOption('--relative-band'),
    limit: numberOption('--injection-limit'),
    maxPerSession: numberOption('--max-per-session'),
  });

  if (datasetName === 'smoke' && !hasFlag('--no-gates')) {
    const failures = evaluateSmokeGates(result.artifact);
    if (failures.length > 0) {
      console.table(failures);
      throw new Error(`${failures.length} smoke benchmark gate(s) failed`);
    }
    console.log('Regression gates: passed');
  }
}

async function runQa(): Promise<void> {
  const datasetName = option('--dataset') ?? 'longmemeval-oracle';
  if (!['longmemeval-oracle', 'longmemeval-s'].includes(datasetName)) {
    throw new Error('qa --dataset must be longmemeval-oracle or longmemeval-s');
  }
  const limit = numberOption('--limit') ?? 3;
  const granularity = (option('--granularity') ?? 'session') as 'session' | 'turn';
  if (!['session', 'turn'].includes(granularity)) {
    throw new Error('--granularity must be session or turn');
  }
  const dataset = loadLongMemEval(
    requirePath(
      resolveDownloadedLongMemEval(
        dataDir,
        datasetName as 'longmemeval-oracle' | 'longmemeval-s',
      ),
      `Run "bun benchmarks/memory-retrieval/cli.ts download ${datasetName}" first.`,
    ),
    { granularity, limit },
  );
  const retrieval = await runRetrievalBenchmark(dataset, {
    serviceUrl,
    cachePath,
    runsDir,
    minimumSimilarity: numberOption('--minimum-similarity'),
    relativeBand: numberOption('--relative-band'),
    limit: numberOption('--injection-limit'),
    maxPerSession: numberOption('--max-per-session'),
  });
  await runQaCheck({
    dataset,
    retrievalArtifact: retrieval.artifact,
    answerModel: option('--answer-model') ?? 'gpt-5.6-luna',
    judgeModel: option('--judge-model') ?? 'gpt-5.6-terra',
    reasoningEffort: option('--reasoning-effort') ?? 'low',
    timeoutMs: numberOption('--timeout-ms') ?? 180_000,
    maxContextChars: numberOption('--max-context-chars') ?? 12_000,
    useJudge: !hasFlag('--no-judge'),
    runsDir,
  });
}

async function prepareMemoryAgentBench(): Promise<void> {
  const splitMap: Record<string, string> = {
    accurate: 'Accurate_Retrieval',
    learning: 'Test_Time_Learning',
    understanding: 'Long_Range_Understanding',
    conflict: 'Conflict_Resolution',
  };
  const split = process.argv[3] ?? 'accurate';
  const basename = splitMap[split];
  if (!basename) {
    throw new Error(`Unknown split "${split}". Choose: ${Object.keys(splitMap).join(', ')}`);
  }
  const input = requirePath(
    join(
      dataDir,
      'memoryagentbench',
      `${basename}-00000-of-00001.parquet`,
    ),
    'Run "bun benchmarks/memory-retrieval/cli.ts download memoryagentbench" first.',
  );
  const output = join(dataDir, 'memoryagentbench', `${basename}.jsonl`);
  const args = [
    'python3',
    join(benchmarkDir, 'convert_memoryagentbench.py'),
    '--input',
    input,
    '--output',
    output,
  ];
  const limit = option('--limit');
  if (limit) args.push('--limit', limit);
  const child = Bun.spawn(args, {
    stdout: 'inherit',
    stderr: 'inherit',
  });
  const exitCode = await child.exited;
  if (exitCode !== 0) {
    throw new Error(
      `MemoryAgentBench conversion failed (${exitCode}). Install pyarrow with ` +
      '"python3 -m pip install pyarrow" if it is missing.',
    );
  }
}

function scoreMab(): void {
  const tasksPath = resolve(option('--tasks') ?? '');
  const predictionsPath = resolve(option('--predictions') ?? '');
  if (!option('--tasks') || !option('--predictions')) {
    throw new Error('mab:score requires --tasks <jsonl> and --predictions <jsonl>');
  }
  const tasks = loadJsonLines<MemoryAgentBenchTask>(
    requirePath(tasksPath, 'Prepare a MemoryAgentBench split first.'),
  );
  const predictions = loadJsonLines<MemoryAgentBenchPrediction>(
    requirePath(predictionsPath, 'Generate predictions using the exported protocol tasks.'),
  );
  const score = scoreMemoryAgentBench(tasks, predictions);
  mkdirSync(runsDir, { recursive: true });
  const artifactPath = join(
    runsDir,
    `${new Date().toISOString().replace(/[:.]/g, '-')}-memoryagentbench-score.json`,
  );
  writeFileSync(artifactPath, `${JSON.stringify({
    schemaVersion: 1,
    createdAt: new Date().toISOString(),
    tasksPath,
    predictionsPath,
    deterministicMetricsOnly: true,
    score,
  }, null, 2)}\n`);
  console.log(JSON.stringify(score, null, 2));
  console.log(`Artifact: ${artifactPath}`);
}

function printHelp(): void {
  console.log(`
Local memory benchmark suite

  smoke
      Run checked-in retrieval regression fixtures and enforce quality gates.
  run --dataset smoke|longmemeval-oracle|longmemeval-s
      Run retrieval and injection-policy ablations. LongMemEval supports
      --granularity session|turn and --limit N.
  system
      Verify worker/Nemotron health, telemetry-off state, durable writes,
      semantic ranking, concurrent requests, and exact ephemeral cleanup.
  generation [--model gpt-5.6-luna]
      Opt-in live Codex observation/summary grounding and poison-resistance test.
  qa --dataset longmemeval-oracle|longmemeval-s [--limit 3]
      End-to-end retrieval plus Luna answers and a Terra local judge.
  download longmemeval-oracle|longmemeval-s|memoryagentbench
      Explicitly download official data from Hugging Face.
  mab:prepare accurate|learning|understanding|conflict [--limit N]
      Convert an official MemoryAgentBench parquet split to protocol JSONL.
  mab:score --tasks tasks.jsonl --predictions predictions.jsonl
      Score deterministic exact/substring/F1/EventQA metrics.

Environment:
  CLAUDE_MEM_NEMOTRON_URL (default ${serviceUrl})
  CLAUDE_MEM_WORKER_URL    (default ${workerUrl})
`);
}

const command = process.argv[2] ?? 'smoke';
switch (command) {
  case 'smoke':
    await runRetrieval('smoke');
    break;
  case 'run': {
    const datasetName = option('--dataset') ?? 'smoke';
    if (!['smoke', 'longmemeval-oracle', 'longmemeval-s'].includes(datasetName)) {
      throw new Error(`Unknown retrieval dataset "${datasetName}"`);
    }
    await runRetrieval(datasetName);
    break;
  }
  case 'system':
    await runSystemCheck({ serviceUrl, workerUrl, runsDir });
    break;
  case 'generation':
    await runGenerationCheck({
      model: option('--model') ?? 'gpt-5.6-luna',
      reasoningEffort: option('--reasoning-effort') ?? 'low',
      timeoutMs: numberOption('--timeout-ms') ?? 180_000,
      runsDir,
    });
    break;
  case 'qa':
    await runQa();
    break;
  case 'download': {
    const name = process.argv[3] as
      | 'longmemeval-oracle'
      | 'longmemeval-s'
      | 'memoryagentbench'
      | undefined;
    if (!name) throw new Error('download requires a dataset name');
    const downloaded = await downloadDataset(name, dataDir);
    console.table(downloaded);
    break;
  }
  case 'mab:prepare':
    await prepareMemoryAgentBench();
    break;
  case 'mab:score':
    scoreMab();
    break;
  case 'help':
  case '--help':
  case '-h':
    printHelp();
    break;
  default:
    printHelp();
    throw new Error(`Unknown command "${command}"`);
}
