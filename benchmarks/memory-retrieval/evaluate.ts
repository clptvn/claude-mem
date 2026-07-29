import fixtures from './fixtures.json';

interface DocumentFixture {
  id: string;
  session: string;
  stateKey?: string;
  createdAtEpoch: number;
  text: string;
}

interface QueryFixture {
  id: string;
  category: string;
  text: string;
  expected: string[];
  forbidden?: string[];
  historical?: boolean;
}

interface RankedDocument extends DocumentFixture {
  similarity: number;
}

interface EvaluationMetrics {
  recallAt5: number;
  mrr: number;
  abstentionAccuracy: number;
  forbiddenSelectionRate: number;
  meanInjected: number;
}

const documents = fixtures.documents as DocumentFixture[];
const queries = fixtures.queries as QueryFixture[];
const serviceUrl = process.env.CLAUDE_MEM_NEMOTRON_URL ?? 'http://127.0.0.1:37901';
const minimumSimilarity = Number(process.env.CLAUDE_MEM_EVAL_MIN_SIMILARITY ?? '0.18');
const limit = 5;

async function embed(input: string[], inputType: 'query' | 'passage'): Promise<number[][]> {
  const response = await fetch(`${serviceUrl}/v1/embeddings`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ input, input_type: inputType }),
  });
  if (!response.ok) {
    throw new Error(`Nemotron embedding request failed (${response.status}): ${await response.text()}`);
  }
  const payload = await response.json() as {
    data: Array<{ index: number; embedding: number[] }>;
  };
  return payload.data
    .sort((left, right) => left.index - right.index)
    .map(item => item.embedding);
}

function dot(left: number[], right: number[]): number {
  let result = 0;
  for (let index = 0; index < left.length; index += 1) {
    result += left[index] * right[index];
  }
  return result;
}

function baselinePolicy(ranked: RankedDocument[]): RankedDocument[] {
  return ranked.slice(0, limit);
}

function calibratedPolicy(ranked: RankedDocument[], query: QueryFixture): RankedDocument[] {
  if (ranked.length === 0) return [];
  const effectiveFloor = Math.max(minimumSimilarity, ranked[0].similarity - 0.24);
  let accepted = ranked.filter(document => document.similarity >= effectiveFloor);

  const byStateKey = new Map<string, RankedDocument[]>();
  for (const document of accepted) {
    if (!document.stateKey) continue;
    const group = byStateKey.get(document.stateKey) ?? [];
    group.push(document);
    byStateKey.set(document.stateKey, group);
  }
  const selectedState = new Map<string, RankedDocument>();
  for (const [stateKey, group] of byStateKey) {
    const byNewest = [...group].sort((left, right) => right.createdAtEpoch - left.createdAtEpoch);
    const selected = query.historical && byNewest.length > 1 ? byNewest[1] : byNewest[0];
    selectedState.set(stateKey, selected);
  }
  if (selectedState.size > 0) {
    accepted = accepted.filter(document => {
      if (!document.stateKey) return true;
      return selectedState.get(document.stateKey) === document;
    });
  }

  const perSession = new Map<string, number>();
  const selected: RankedDocument[] = [];
  for (const document of accepted) {
    if ((perSession.get(document.session) ?? 0) >= 2) continue;
    selected.push(document);
    perSession.set(document.session, (perSession.get(document.session) ?? 0) + 1);
    if (selected.length >= limit) break;
  }
  return selected;
}

function calculateMetrics(
  rankedByQuery: Map<string, RankedDocument[]>,
  policy: (ranked: RankedDocument[], query: QueryFixture) => RankedDocument[],
): EvaluationMetrics {
  const answerable = queries.filter(query => query.expected.length > 0);
  const unanswerable = queries.filter(query => query.expected.length === 0);
  let recall = 0;
  let reciprocalRank = 0;
  let forbiddenSelections = 0;
  let forbiddenQueries = 0;
  let totalInjected = 0;

  for (const query of queries) {
    const ranked = rankedByQuery.get(query.id) ?? [];
    const selected = policy(ranked, query);
    totalInjected += selected.length;
    if (query.expected.length > 0) {
      const selectedIds = new Set(selected.map(document => document.id));
      const found = query.expected.filter(id => selectedIds.has(id)).length;
      recall += found / query.expected.length;
      const firstExpectedRank = selected.findIndex(document => query.expected.includes(document.id));
      reciprocalRank += firstExpectedRank >= 0 ? 1 / (firstExpectedRank + 1) : 0;
    }
    if ((query.forbidden?.length ?? 0) > 0) {
      forbiddenQueries += 1;
      if (selected.some(document => query.forbidden!.includes(document.id))) {
        forbiddenSelections += 1;
      }
    }
  }

  const abstentions = unanswerable.filter(query =>
    policy(rankedByQuery.get(query.id) ?? [], query).length === 0
  ).length;
  return {
    recallAt5: answerable.length > 0 ? recall / answerable.length : 0,
    mrr: answerable.length > 0 ? reciprocalRank / answerable.length : 0,
    abstentionAccuracy: unanswerable.length > 0 ? abstentions / unanswerable.length : 0,
    forbiddenSelectionRate: forbiddenQueries > 0 ? forbiddenSelections / forbiddenQueries : 0,
    meanInjected: totalInjected / queries.length,
  };
}

function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

const [queryVectors, passageVectors] = await Promise.all([
  embed(queries.map(query => query.text), 'query'),
  embed(documents.map(document => document.text), 'passage'),
]);

const rankedByQuery = new Map<string, RankedDocument[]>();
for (let queryIndex = 0; queryIndex < queries.length; queryIndex += 1) {
  rankedByQuery.set(
    queries[queryIndex].id,
    documents
      .map((document, documentIndex) => ({
        ...document,
        similarity: dot(queryVectors[queryIndex], passageVectors[documentIndex]),
      }))
      .sort((left, right) => right.similarity - left.similarity),
  );
}

const baseline = calculateMetrics(rankedByQuery, ranked => baselinePolicy(ranked));
const calibrated = calculateMetrics(rankedByQuery, calibratedPolicy);

console.log(`Nemotron retrieval evaluation (${documents.length} memories, ${queries.length} queries)`);
console.log(`Service: ${serviceUrl}`);
console.log(`Calibrated policy: cosine >= ${minimumSimilarity.toFixed(2)}, relative band 0.24, latest-state resolution, max 2/session`);
console.table([
  {
    policy: 'fixed top-5 baseline',
    'Recall@5': percent(baseline.recallAt5),
    MRR: baseline.mrr.toFixed(3),
    abstention: percent(baseline.abstentionAccuracy),
    'stale selection': percent(baseline.forbiddenSelectionRate),
    'mean injected': baseline.meanInjected.toFixed(2),
  },
  {
    policy: 'calibrated automatic recall',
    'Recall@5': percent(calibrated.recallAt5),
    MRR: calibrated.mrr.toFixed(3),
    abstention: percent(calibrated.abstentionAccuracy),
    'stale selection': percent(calibrated.forbiddenSelectionRate),
    'mean injected': calibrated.meanInjected.toFixed(2),
  },
]);

console.log('\nPer-query top match:');
for (const query of queries) {
  const top = rankedByQuery.get(query.id)?.[0];
  const selected = calibratedPolicy(rankedByQuery.get(query.id) ?? [], query);
  console.log(JSON.stringify({
    query: query.id,
    category: query.category,
    top: top ? { id: top.id, similarity: Number(top.similarity.toFixed(4)) } : null,
    selected: selected.map(document => document.id),
    expected: query.expected,
  }));
}
