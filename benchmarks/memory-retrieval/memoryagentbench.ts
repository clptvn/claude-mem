import { readFileSync } from 'node:fs';

export interface MemoryAgentBenchTask {
  task_id: string;
  competency: string;
  source: string;
  context: string;
  questions: string[];
  answers: unknown[][];
  metadata: Record<string, unknown>;
}

export interface MemoryAgentBenchPrediction {
  task_id: string;
  question_index: number;
  prediction: string;
}

interface SourceScore {
  scored: number;
  exactMatch: number;
  substringMatch: number;
  tokenF1: number;
}

function normalize(value: string): string {
  return value
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .replace(/\b(a|an|the)\b/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function flattenAnswers(value: unknown): string[] {
  if (typeof value === 'string' || typeof value === 'number') return [String(value)];
  if (!Array.isArray(value)) return [];
  return value.flatMap(flattenAnswers);
}

function tokenF1(prediction: string, answer: string): number {
  const predicted = normalize(prediction).split(' ').filter(Boolean);
  const expected = normalize(answer).split(' ').filter(Boolean);
  if (predicted.length === 0 || expected.length === 0) return 0;
  const counts = new Map<string, number>();
  for (const token of expected) counts.set(token, (counts.get(token) ?? 0) + 1);
  let overlap = 0;
  for (const token of predicted) {
    const remaining = counts.get(token) ?? 0;
    if (remaining > 0) {
      overlap += 1;
      counts.set(token, remaining - 1);
    }
  }
  if (overlap === 0) return 0;
  const precision = overlap / predicted.length;
  const recall = overlap / expected.length;
  return 2 * precision * recall / (precision + recall);
}

export function loadJsonLines<T>(path: string): T[] {
  return readFileSync(path, 'utf8')
    .split('\n')
    .filter(line => line.trim().length > 0)
    .map(line => JSON.parse(line) as T);
}

export function scoreMemoryAgentBench(
  tasks: MemoryAgentBenchTask[],
  predictions: MemoryAgentBenchPrediction[],
): {
  scored: number;
  missing: number;
  exactMatch: number;
  substringMatch: number;
  tokenF1: number;
  eventQaAllFacts: number;
  bySource: Record<string, SourceScore>;
} {
  const predictionMap = new Map(
    predictions.map(prediction => [
      `${prediction.task_id}:${prediction.question_index}`,
      prediction.prediction,
    ]),
  );
  const rows: Array<{
    source: string;
    exact: number;
    substring: number;
    f1: number;
    eventQa: number | null;
  }> = [];
  let missing = 0;

  for (const task of tasks) {
    for (let questionIndex = 0; questionIndex < task.questions.length; questionIndex += 1) {
      const prediction = predictionMap.get(`${task.task_id}:${questionIndex}`);
      if (prediction === undefined) {
        missing += 1;
        continue;
      }
      const answers = flattenAnswers(task.answers[questionIndex] ?? []);
      const normalizedPrediction = normalize(prediction);
      const exact = answers.some(answer => normalizedPrediction === normalize(answer)) ? 1 : 0;
      const substring = answers.some(answer =>
        normalizedPrediction.includes(normalize(answer))
      ) ? 1 : 0;
      const f1 = answers.length > 0
        ? Math.max(...answers.map(answer => tokenF1(prediction, answer)))
        : 0;
      const isEventQa = task.source.toLowerCase().includes('eventqa');
      const eventQa = isEventQa
        ? (answers.length > 0 && answers.every(answer =>
            prediction.toLowerCase().includes(answer.toLowerCase())
          ) ? 1 : 0)
        : null;
      rows.push({ source: task.source, exact, substring, f1, eventQa });
    }
  }

  const average = (values: number[]) =>
    values.length > 0
      ? values.reduce((sum, value) => sum + value, 0) / values.length
      : 0;
  const bySourceRows = new Map<string, typeof rows>();
  for (const row of rows) {
    const group = bySourceRows.get(row.source) ?? [];
    group.push(row);
    bySourceRows.set(row.source, group);
  }
  const bySource: Record<string, SourceScore> = {};
  for (const [source, sourceRows] of bySourceRows) {
    bySource[source] = {
      scored: sourceRows.length,
      exactMatch: average(sourceRows.map(row => row.exact)),
      substringMatch: average(sourceRows.map(row => row.substring)),
      tokenF1: average(sourceRows.map(row => row.f1)),
    };
  }
  const eventRows = rows.filter(
    (row): row is typeof row & { eventQa: number } => row.eventQa !== null,
  );
  return {
    scored: rows.length,
    missing,
    exactMatch: average(rows.map(row => row.exact)),
    substringMatch: average(rows.map(row => row.substring)),
    tokenF1: average(rows.map(row => row.f1)),
    eventQaAllFacts: average(eventRows.map(row => row.eventQa)),
    bySource,
  };
}
