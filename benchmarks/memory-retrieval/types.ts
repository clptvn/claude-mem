export interface BenchmarkDocument {
  id: string;
  session: string;
  stateKey?: string;
  createdAtEpoch: number;
  text: string;
}

export interface BenchmarkQuery {
  id: string;
  category: string;
  text: string;
  expected: string[];
  forbidden?: string[];
  /** Restrict retrieval to this instance's haystack (used by LongMemEval). */
  candidateIds?: string[];
  /** Reference answer for optional end-to-end QA evaluation. */
  referenceAnswer?: string;
  questionDate?: string;
}

export interface BenchmarkDataset {
  name: string;
  version: string;
  source: string;
  granularity: 'memory' | 'session' | 'turn';
  documents: BenchmarkDocument[];
  queries: BenchmarkQuery[];
  metadata?: Record<string, unknown>;
}

export interface RankedDocument extends BenchmarkDocument {
  similarity: number;
}

export interface RetrievalMetrics {
  queryCount: number;
  answerableCount: number;
  unanswerableCount: number;
  recallAnyAtK: Record<string, number>;
  recallAllAtK: Record<string, number>;
  precisionAtK: Record<string, number>;
  ndcgAtK: Record<string, number>;
  mrr: number;
  abstentionAccuracy: number;
  falsePositiveRate: number;
  forbiddenSelectionRate: number;
  meanSelected: number;
  meanSelectedChars: number;
  latencyMs: {
    mean: number;
    p50: number;
    p95: number;
    p99: number;
  };
  byCategory: Record<string, {
    queries: number;
    recallAnyAt5: number;
    recallAllAt5: number;
    mrr: number;
  }>;
}

export interface PolicyRun {
  policy: string;
  description: string;
  metrics: RetrievalMetrics;
  selections: Array<{
    queryId: string;
    category: string;
    expected: string[];
    forbidden: string[];
    selected: Array<{ id: string; similarity: number }>;
    effectiveMinimumSimilarity: number | null;
    latencyMs: number;
  }>;
}

export interface BenchmarkArtifact {
  schemaVersion: 1;
  runId: string;
  createdAt: string;
  dataset: {
    name: string;
    version: string;
    source: string;
    granularity: BenchmarkDataset['granularity'];
    documents: number;
    queries: number;
    sha256: string;
    metadata?: Record<string, unknown>;
  };
  environment: {
    gitCommit: string | null;
    gitDirty: boolean | null;
    platform: string;
    architecture: string;
    bunVersion: string;
    serviceUrl: string;
    embeddingService: Record<string, unknown> | null;
    processMemoryBytes: NodeJS.MemoryUsage;
  };
  config: {
    minimumSimilarity: number;
    relativeBand: number;
    limit: number;
    maxPerSession: number;
    ks: number[];
  };
  policies: PolicyRun[];
}
