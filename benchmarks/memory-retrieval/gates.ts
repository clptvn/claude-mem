import type { BenchmarkArtifact } from './types.js';

export interface GateFailure {
  metric: string;
  expected: string;
  actual: number;
}

export function evaluateSmokeGates(artifact: BenchmarkArtifact): GateFailure[] {
  const production = artifact.policies.find(
    policy => policy.policy === 'calibrated-production',
  );
  if (!production) {
    return [{ metric: 'policy', expected: 'calibrated-production present', actual: 0 }];
  }
  const checks: Array<{
    metric: string;
    actual: number;
    minimum?: number;
    maximum?: number;
  }> = [
    {
      metric: 'recallAll@5',
      actual: production.metrics.recallAllAtK['5'] ?? 0,
      minimum: 0.90,
    },
    {
      metric: 'MRR',
      actual: production.metrics.mrr,
      minimum: 0.90,
    },
    {
      metric: 'abstentionAccuracy',
      actual: production.metrics.abstentionAccuracy,
      minimum: 0.90,
    },
    {
      metric: 'forbiddenSelectionRate',
      actual: production.metrics.forbiddenSelectionRate,
      maximum: 0,
    },
    {
      metric: 'meanSelected',
      actual: production.metrics.meanSelected,
      maximum: 2.5,
    },
  ];
  return checks.flatMap(check => {
    if (check.minimum !== undefined && check.actual < check.minimum) {
      return [{
        metric: check.metric,
        expected: `>= ${check.minimum}`,
        actual: check.actual,
      }];
    }
    if (check.maximum !== undefined && check.actual > check.maximum) {
      return [{
        metric: check.metric,
        expected: `<= ${check.maximum}`,
        actual: check.actual,
      }];
    }
    return [];
  });
}
