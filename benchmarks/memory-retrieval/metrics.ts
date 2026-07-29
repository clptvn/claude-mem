import type {
  BenchmarkDocument,
  BenchmarkQuery,
  RankedDocument,
  RetrievalMetrics,
} from './types.js';

function average(values: number[]): number {
  return values.length > 0
    ? values.reduce((sum, value) => sum + value, 0) / values.length
    : 0;
}

function percentile(values: number[], quantile: number): number {
  if (values.length === 0) return 0;
  const ordered = [...values].sort((left, right) => left - right);
  const index = Math.min(
    ordered.length - 1,
    Math.max(0, Math.ceil(quantile * ordered.length) - 1),
  );
  return ordered[index];
}

function reciprocalRank(selected: RankedDocument[], expected: Set<string>): number {
  const index = selected.findIndex(document => expected.has(document.id));
  return index >= 0 ? 1 / (index + 1) : 0;
}

function ndcg(selected: RankedDocument[], expected: Set<string>, k: number): number {
  if (expected.size === 0) return 0;
  let actual = 0;
  for (let index = 0; index < Math.min(k, selected.length); index += 1) {
    if (expected.has(selected[index].id)) {
      actual += 1 / Math.log2(index + 2);
    }
  }
  let ideal = 0;
  for (let index = 0; index < Math.min(k, expected.size); index += 1) {
    ideal += 1 / Math.log2(index + 2);
  }
  return ideal > 0 ? actual / ideal : 0;
}

export function calculateRetrievalMetrics(
  queries: BenchmarkQuery[],
  selections: Map<string, RankedDocument[]>,
  latenciesMs: Map<string, number>,
  ks: number[] = [1, 5, 10],
): RetrievalMetrics {
  const answerable = queries.filter(query => query.expected.length > 0);
  const unanswerable = queries.filter(query => query.expected.length === 0);
  const recallAnyAtK: Record<string, number> = {};
  const recallAllAtK: Record<string, number> = {};
  const precisionAtK: Record<string, number> = {};
  const ndcgAtK: Record<string, number> = {};

  for (const k of ks) {
    const answerableRows = answerable.map(query => {
      const selected = (selections.get(query.id) ?? []).slice(0, k);
      const expected = new Set(query.expected);
      const found = selected.filter(document => expected.has(document.id)).length;
      return {
        any: found > 0 ? 1 : 0,
        all: found === expected.size ? 1 : 0,
        precision: selected.length > 0 ? found / selected.length : 0,
        ndcg: ndcg(selected, expected, k),
      };
    });
    recallAnyAtK[String(k)] = average(answerableRows.map(row => row.any));
    recallAllAtK[String(k)] = average(answerableRows.map(row => row.all));
    precisionAtK[String(k)] = average(answerableRows.map(row => row.precision));
    ndcgAtK[String(k)] = average(answerableRows.map(row => row.ndcg));
  }

  const forbiddenQueries = queries.filter(query => (query.forbidden?.length ?? 0) > 0);
  const forbiddenSelections = forbiddenQueries.filter(query => {
    const forbidden = new Set(query.forbidden);
    return (selections.get(query.id) ?? []).some(document => forbidden.has(document.id));
  }).length;

  const selectedCounts = queries.map(query => (selections.get(query.id) ?? []).length);
  const selectedChars = queries.map(query =>
    (selections.get(query.id) ?? [])
      .reduce((sum, document) => sum + document.text.length, 0)
  );
  const falsePositives = unanswerable.filter(
    query => (selections.get(query.id) ?? []).length > 0,
  ).length;
  const categoryGroups = new Map<string, BenchmarkQuery[]>();
  for (const query of answerable) {
    const group = categoryGroups.get(query.category) ?? [];
    group.push(query);
    categoryGroups.set(query.category, group);
  }
  const byCategory: RetrievalMetrics['byCategory'] = {};
  for (const [category, categoryQueries] of categoryGroups) {
    const rows = categoryQueries.map(query => {
      const selected = selections.get(query.id) ?? [];
      const expected = new Set(query.expected);
      const found = selected.slice(0, 5).filter(document => expected.has(document.id)).length;
      return {
        any: found > 0 ? 1 : 0,
        all: found === expected.size ? 1 : 0,
        reciprocalRank: reciprocalRank(selected, expected),
      };
    });
    byCategory[category] = {
      queries: rows.length,
      recallAnyAt5: average(rows.map(row => row.any)),
      recallAllAt5: average(rows.map(row => row.all)),
      mrr: average(rows.map(row => row.reciprocalRank)),
    };
  }

  const latencyValues = queries.map(query => latenciesMs.get(query.id) ?? 0);
  return {
    queryCount: queries.length,
    answerableCount: answerable.length,
    unanswerableCount: unanswerable.length,
    recallAnyAtK,
    recallAllAtK,
    precisionAtK,
    ndcgAtK,
    mrr: average(answerable.map(query =>
      reciprocalRank(selections.get(query.id) ?? [], new Set(query.expected))
    )),
    abstentionAccuracy: unanswerable.length > 0
      ? 1 - falsePositives / unanswerable.length
      : 0,
    falsePositiveRate: unanswerable.length > 0
      ? falsePositives / unanswerable.length
      : 0,
    forbiddenSelectionRate: forbiddenQueries.length > 0
      ? forbiddenSelections / forbiddenQueries.length
      : 0,
    meanSelected: average(selectedCounts),
    meanSelectedChars: average(selectedChars),
    latencyMs: {
      mean: average(latencyValues),
      p50: percentile(latencyValues, 0.50),
      p95: percentile(latencyValues, 0.95),
      p99: percentile(latencyValues, 0.99),
    },
    byCategory,
  };
}

export function rankByDotProduct(
  documents: BenchmarkDocument[],
  queryVector: number[],
  documentVectors: number[][],
): RankedDocument[] {
  return documents.map((document, index) => {
    const vector = documentVectors[index];
    let similarity = 0;
    for (let dimension = 0; dimension < queryVector.length; dimension += 1) {
      similarity += queryVector[dimension] * vector[dimension];
    }
    return { ...document, similarity };
  }).sort((left, right) => right.similarity - left.similarity);
}
