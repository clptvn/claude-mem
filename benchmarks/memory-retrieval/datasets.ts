import { createHash } from 'node:crypto';
import {
  createReadStream,
  createWriteStream,
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { basename, dirname, join } from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import fixtures from './fixtures.json';
import type {
  BenchmarkDataset,
  BenchmarkDocument,
  BenchmarkQuery,
} from './types.js';

interface LongMemEvalTurn {
  role: string;
  content: string;
  has_answer?: boolean;
}

interface LongMemEvalEntry {
  question_id: string;
  question_type: string;
  question: string;
  answer: string;
  question_date: string;
  haystack_session_ids: string[];
  haystack_dates: string[];
  haystack_sessions: LongMemEvalTurn[][];
  answer_session_ids?: string[];
}

interface DownloadSpec {
  filename: string;
  url: string;
  revision: string;
  bytes: number;
}

const DOWNLOADS: Record<string, DownloadSpec[]> = {
  'longmemeval-oracle': [{
    filename: 'longmemeval_oracle.json',
    url: 'https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/98d7416c24c778c2fee6e6f3006e7a073259d48f/longmemeval_oracle.json',
    revision: '98d7416c24c778c2fee6e6f3006e7a073259d48f',
    bytes: 15_388_478,
  }],
  'longmemeval-s': [{
    filename: 'longmemeval_s_cleaned.json',
    url: 'https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/98d7416c24c778c2fee6e6f3006e7a073259d48f/longmemeval_s_cleaned.json',
    revision: '98d7416c24c778c2fee6e6f3006e7a073259d48f',
    bytes: 277_383_467,
  }],
  memoryagentbench: [
    {
      filename: 'Accurate_Retrieval-00000-of-00001.parquet',
      url: 'https://huggingface.co/datasets/ai-hyz/MemoryAgentBench/resolve/7ea066982b140a19337e17e60d45d4076e042faf/data/Accurate_Retrieval-00000-of-00001.parquet',
      revision: '7ea066982b140a19337e17e60d45d4076e042faf',
      bytes: 20_024_386,
    },
    {
      filename: 'Test_Time_Learning-00000-of-00001.parquet',
      url: 'https://huggingface.co/datasets/ai-hyz/MemoryAgentBench/resolve/7ea066982b140a19337e17e60d45d4076e042faf/data/Test_Time_Learning-00000-of-00001.parquet',
      revision: '7ea066982b140a19337e17e60d45d4076e042faf',
      bytes: 3_947_476,
    },
    {
      filename: 'Long_Range_Understanding-00000-of-00001.parquet',
      url: 'https://huggingface.co/datasets/ai-hyz/MemoryAgentBench/resolve/7ea066982b140a19337e17e60d45d4076e042faf/data/Long_Range_Understanding-00000-of-00001.parquet',
      revision: '7ea066982b140a19337e17e60d45d4076e042faf',
      bytes: 49_342_452,
    },
    {
      filename: 'Conflict_Resolution-00000-of-00001.parquet',
      url: 'https://huggingface.co/datasets/ai-hyz/MemoryAgentBench/resolve/7ea066982b140a19337e17e60d45d4076e042faf/data/Conflict_Resolution-00000-of-00001.parquet',
      revision: '7ea066982b140a19337e17e60d45d4076e042faf',
      bytes: 1_491_588,
    },
  ],
};

export function loadSmokeDataset(): BenchmarkDataset {
  return {
    name: 'memory-enhanced-smoke',
    version: '2',
    source: 'checked-in synthetic regression fixture',
    granularity: 'memory',
    documents: fixtures.documents as BenchmarkDocument[],
    queries: fixtures.queries as BenchmarkQuery[],
    metadata: {
      caveat: 'Small curated regression fixture; never report as a general benchmark.',
    },
  };
}

function sessionText(turns: LongMemEvalTurn[]): string {
  return turns
    .filter(turn => turn.role === 'user')
    .map(turn => turn.content)
    .join(' ');
}

function timestamp(value: string, fallback: number): number {
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export function parseLongMemEval(
  entries: LongMemEvalEntry[],
  options: {
    granularity: 'session' | 'turn';
    limit?: number;
    source?: string;
    version?: string;
  },
): BenchmarkDataset {
  const selectedEntries = entries.slice(0, options.limit ?? entries.length);
  const documents: BenchmarkDocument[] = [];
  const queries: BenchmarkQuery[] = [];

  for (const [entryIndex, entry] of selectedEntries.entries()) {
    if (
      entry.haystack_session_ids.length !== entry.haystack_sessions.length
      || entry.haystack_dates.length !== entry.haystack_sessions.length
    ) {
      throw new Error(`LongMemEval ${entry.question_id} has misaligned haystack arrays`);
    }
    const candidateIds: string[] = [];
    const expected: string[] = [];
    const answerSessions = new Set(entry.answer_session_ids ?? []);

    for (let sessionIndex = 0; sessionIndex < entry.haystack_sessions.length; sessionIndex += 1) {
      const sessionId = entry.haystack_session_ids[sessionIndex];
      const turns = entry.haystack_sessions[sessionIndex];
      const createdAtEpoch = timestamp(
        entry.haystack_dates[sessionIndex],
        entryIndex * 1_000 + sessionIndex,
      );

      if (options.granularity === 'session') {
        const id = `${entry.question_id}:${sessionId}`;
        documents.push({
          id,
          session: `${entry.question_id}:${sessionId}`,
          createdAtEpoch,
          text: sessionText(turns),
        });
        candidateIds.push(id);
        const userTurns = turns.filter(turn => turn.role === 'user');
        const hasTurnLabels = userTurns.some(turn => 'has_answer' in turn);
        const isAnswerSession = hasTurnLabels
          ? userTurns.some(turn => turn.has_answer === true)
          : answerSessions.has(sessionId);
        if (isAnswerSession) {
          expected.push(id);
        }
        continue;
      }

      for (let turnIndex = 0; turnIndex < turns.length; turnIndex += 1) {
        const turn = turns[turnIndex];
        if (turn.role !== 'user') continue;
        const id = `${entry.question_id}:${sessionId}_${turnIndex + 1}`;
        documents.push({
          id,
          session: `${entry.question_id}:${sessionId}`,
          createdAtEpoch: createdAtEpoch + turnIndex,
          text: turn.content,
        });
        candidateIds.push(id);
        if (turn.has_answer === true) expected.push(id);
      }
    }

    queries.push({
      id: entry.question_id,
      category: entry.question_type,
      text: entry.question,
      expected: [...new Set(expected)],
      candidateIds,
      referenceAnswer: entry.answer,
      questionDate: entry.question_date,
    });
  }

  return {
    name: 'longmemeval',
    version: options.version ?? 'unknown',
    source: options.source ?? 'LongMemEval cleaned dataset',
    granularity: options.granularity,
    documents,
    queries,
    metadata: {
      instances: selectedEntries.length,
      officialProtocol: options.granularity === 'session'
        ? ['recall_all@5', 'ndcg_any@5', 'recall_all@10', 'ndcg_any@10']
        : [
            'recall_all@5',
            'ndcg_any@5',
            'recall_all@10',
            'ndcg_any@10',
            'recall_all@50',
            'ndcg_any@50',
          ],
      note: 'Each query is evaluated only against its own haystack.',
    },
  };
}

export function loadLongMemEval(
  path: string,
  options: { granularity: 'session' | 'turn'; limit?: number },
): BenchmarkDataset {
  const parsed = JSON.parse(readFileSync(path, 'utf8')) as LongMemEvalEntry[];
  return parseLongMemEval(parsed, {
    ...options,
    source: path,
    version: fileSha256(path),
  });
}

export function datasetSha256(dataset: BenchmarkDataset): string {
  return createHash('sha256')
    .update(JSON.stringify({
      name: dataset.name,
      version: dataset.version,
      granularity: dataset.granularity,
      documents: dataset.documents,
      queries: dataset.queries,
    }))
    .digest('hex');
}

export function fileSha256(path: string): string {
  const hash = createHash('sha256');
  hash.update(readFileSync(path));
  return hash.digest('hex');
}

async function downloadFile(spec: DownloadSpec, destination: string): Promise<void> {
  if (existsSync(destination)) {
    const actualBytes = Bun.file(destination).size;
    if (actualBytes !== spec.bytes) {
      throw new Error(
        `Cached size mismatch for ${destination}: expected ${spec.bytes}, got ${actualBytes}`,
      );
    }
    return;
  }
  mkdirSync(dirname(destination), { recursive: true });
  const temporary = `${destination}.partial-${process.pid}`;
  rmSync(temporary, { force: true });
  const response = await fetch(spec.url, {
    redirect: 'follow',
    signal: AbortSignal.timeout(30 * 60_000),
  });
  if (!response.ok || !response.body) {
    throw new Error(`Download failed (${response.status}) for ${spec.url}`);
  }
  try {
    await pipeline(
      Readable.fromWeb(response.body as never),
      createWriteStream(temporary, { flags: 'wx' }),
    );
    renameSync(temporary, destination);
  } catch (error) {
    rmSync(temporary, { force: true });
    throw error;
  }
  const actualBytes = Bun.file(destination).size;
  if (actualBytes !== spec.bytes) {
    throw new Error(
      `Downloaded size mismatch for ${destination}: expected ${spec.bytes}, got ${actualBytes}`,
    );
  }
}

export async function downloadDataset(
  name: keyof typeof DOWNLOADS,
  dataDir: string,
): Promise<Array<{ path: string; sha256: string; bytes: number }>> {
  const specs = DOWNLOADS[name];
  if (!specs) {
    throw new Error(`Unknown download "${name}". Choose: ${Object.keys(DOWNLOADS).join(', ')}`);
  }
  const datasetDir = join(dataDir, name);
  const results: Array<{ path: string; sha256: string; bytes: number }> = [];
  for (const spec of specs) {
    const destination = join(datasetDir, spec.filename);
    await downloadFile(spec, destination);
    const file = Bun.file(destination);
    results.push({
      path: destination,
      sha256: fileSha256(destination),
      bytes: file.size,
    });
  }
  writeFileSync(
    join(datasetDir, 'manifest.json'),
    `${JSON.stringify({
      dataset: name,
      downloadedAt: new Date().toISOString(),
      huggingFaceOnly: true,
      files: specs.map(spec => ({
        filename: basename(spec.filename),
        url: spec.url,
        revision: spec.revision,
        expectedBytes: spec.bytes,
        actual: results.find(result => basename(result.path) === spec.filename),
      })),
    }, null, 2)}\n`,
  );
  return results;
}

export function resolveDownloadedLongMemEval(
  dataDir: string,
  name: 'longmemeval-oracle' | 'longmemeval-s',
): string {
  return join(
    dataDir,
    name,
    name === 'longmemeval-oracle'
      ? 'longmemeval_oracle.json'
      : 'longmemeval_s_cleaned.json',
  );
}
