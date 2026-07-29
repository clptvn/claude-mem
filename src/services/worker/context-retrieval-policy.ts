import { logger } from '../../utils/logger.js';

// This is intentionally a pure hot-path module. Keep the standard worker
// logger dependency available for future diagnostics without logging query or
// memory content from policy evaluation.
void logger;

export interface RankedMemoryCandidate {
  memorySessionId: string;
  title: string;
  body: string;
  createdAtEpoch: number;
  similarity: number;
  /**
   * Optional explicit identity for facts that change over time. Production
   * observations fall back to a normalized title; benchmark fixtures can use
   * a stable key without manufacturing identical titles.
   */
  stateKey?: string;
}

export interface ConfidenceWindowOptions {
  minimumSimilarity?: number;
  relativeBand?: number;
}

export interface RetrievalPolicyOptions extends ConfidenceWindowOptions {
  query: string;
  limit?: number;
  maxPerSession?: number;
  resolveFreshness?: boolean;
  deduplicate?: boolean;
  diversifySessions?: boolean;
  duplicateThreshold?: number;
}

export interface RetrievalPolicyResult<T extends RankedMemoryCandidate> {
  candidates: T[];
  accepted: number;
  rejectedLowConfidence: number;
  rejectedRedundant: number;
  effectiveMinimumSimilarity: number;
}

const DEFAULT_MINIMUM_SIMILARITY = 0.18;
const DEFAULT_RELATIVE_BAND = 0.24;
const DEFAULT_LIMIT = 5;
const DEFAULT_MAX_PER_SESSION = 2;
const DEFAULT_DUPLICATE_THRESHOLD = 0.82;

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(Math.max(value, minimum), maximum);
}

function normalizedMemoryKey(candidate: RankedMemoryCandidate): string {
  const value = candidate.stateKey?.trim() || candidate.title;
  return value
    .toLowerCase()
    .replace(/[^a-z0-9_./-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function tokenSet(value: string): Set<string> {
  return new Set(
    value
      .toLowerCase()
      .match(/[a-z0-9_./-]{3,}/g)
      ?.filter(token => !['this', 'that', 'with', 'from', 'have', 'were', 'into'].includes(token))
      ?? []
  );
}

function jaccardSimilarity(left: Set<string>, right: Set<string>): number {
  if (left.size === 0 || right.size === 0) return 0;
  let intersection = 0;
  for (const token of left) {
    if (right.has(token)) intersection += 1;
  }
  return intersection / (left.size + right.size - intersection);
}

function requestsHistoricalState(query: string): boolean {
  return /\b(previous|previously|before|former|formerly|old|older|original|originally|used to|at the time|history|historical)\b/i.test(query);
}

function requestsCurrentHistoricalComparison(query: string): boolean {
  return /\b(compare|comparison|versus|vs\.?|then and now|current and previous|previous and current)\b/i.test(query);
}

export function calculateEffectiveMinimumSimilarity(
  candidates: Array<{ similarity: number }>,
  options: ConfidenceWindowOptions = {},
): number {
  const configuredFloor = clamp(
    options.minimumSimilarity ?? DEFAULT_MINIMUM_SIMILARITY,
    -1,
    1,
  );
  const finiteSimilarities = candidates
    .map(candidate => candidate.similarity)
    .filter(Number.isFinite);
  if (finiteSimilarities.length === 0) return configuredFloor;

  const relativeBand = clamp(
    options.relativeBand ?? DEFAULT_RELATIVE_BAND,
    0,
    2,
  );
  return Math.max(configuredFloor, Math.max(...finiteSimilarities) - relativeBand);
}

export function filterCandidatesByConfidence<T extends { similarity: number }>(
  candidates: T[],
  options: ConfidenceWindowOptions = {},
): { candidates: T[]; effectiveMinimumSimilarity: number } {
  const effectiveMinimumSimilarity = calculateEffectiveMinimumSimilarity(candidates, options);
  return {
    candidates: candidates.filter(candidate =>
      Number.isFinite(candidate.similarity)
      && candidate.similarity >= effectiveMinimumSimilarity
    ),
    effectiveMinimumSimilarity,
  };
}

export function rankAndSelectContextCandidates<T extends RankedMemoryCandidate>(
  acceptedCandidates: T[],
  options: Omit<RetrievalPolicyOptions, keyof ConfidenceWindowOptions>,
): T[] {
  const limit = clamp(options.limit ?? DEFAULT_LIMIT, 1, 20);
  const maxPerSession = clamp(options.maxPerSession ?? DEFAULT_MAX_PER_SESSION, 1, 10);
  const resolveFreshness = options.resolveFreshness ?? true;
  const deduplicate = options.deduplicate ?? true;
  const diversifySessions = options.diversifySessions ?? true;
  const duplicateThreshold = clamp(
    options.duplicateThreshold ?? DEFAULT_DUPLICATE_THRESHOLD,
    0,
    1,
  );

  let eligible = [...acceptedCandidates];
  if (resolveFreshness) {
    const historicalQuery = requestsHistoricalState(options.query);
    const temporalComparison = requestsCurrentHistoricalComparison(options.query);
    const candidatesByKey = new Map<string, T[]>();
    for (const candidate of eligible) {
      const key = normalizedMemoryKey(candidate);
      if (!key) continue;
      const group = candidatesByKey.get(key) ?? [];
      group.push(candidate);
      candidatesByKey.set(key, group);
    }

    const newestByKey = new Map<string, T>();
    const previousByKey = new Map<string, T>();
    for (const [key, group] of candidatesByKey) {
      const byNewest = [...group].sort(
        (left, right) => right.createdAtEpoch - left.createdAtEpoch,
      );
      newestByKey.set(key, byNewest[0]);
      if (byNewest.length > 1) previousByKey.set(key, byNewest[1]);
    }

    eligible = eligible.filter(candidate => {
      const key = normalizedMemoryKey(candidate);
      if (!key || (candidatesByKey.get(key)?.length ?? 0) < 2) return true;
      if (historicalQuery && !temporalComparison) {
        return previousByKey.get(key) === candidate;
      }
      if (!historicalQuery) {
        return newestByKey.get(key) === candidate;
      }
      return true;
    });
  }

  const ranked = eligible.sort((left, right) => {
    const scoreDifference = right.similarity - left.similarity;
    if (Math.abs(scoreDifference) > 0.01) return scoreDifference;
    return right.createdAtEpoch - left.createdAtEpoch;
  });

  const selected: T[] = [];
  const perSession = new Map<string, number>();
  const selectedTokens: Set<string>[] = [];
  for (const candidate of ranked) {
    if (
      diversifySessions
      && (perSession.get(candidate.memorySessionId) ?? 0) >= maxPerSession
    ) {
      continue;
    }

    const tokens = tokenSet(`${candidate.title}\n${candidate.body}`);
    if (
      deduplicate
      && selectedTokens.some(existing =>
        jaccardSimilarity(existing, tokens) >= duplicateThreshold
      )
    ) {
      continue;
    }

    selected.push(candidate);
    selectedTokens.push(tokens);
    perSession.set(
      candidate.memorySessionId,
      (perSession.get(candidate.memorySessionId) ?? 0) + 1,
    );
    if (selected.length >= limit) break;
  }
  return selected;
}

export function applyRetrievalPolicy<T extends RankedMemoryCandidate>(
  candidates: T[],
  options: RetrievalPolicyOptions,
): RetrievalPolicyResult<T> {
  const confidence = filterCandidatesByConfidence(candidates, options);
  const selected = rankAndSelectContextCandidates(confidence.candidates, options);
  return {
    candidates: selected,
    accepted: confidence.candidates.length,
    rejectedLowConfidence: candidates.length - confidence.candidates.length,
    rejectedRedundant: Math.max(0, confidence.candidates.length - selected.length),
    effectiveMinimumSimilarity: confidence.effectiveMinimumSimilarity,
  };
}
