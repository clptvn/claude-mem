import { SettingsDefaultsManager } from '../../shared/SettingsDefaultsManager.js';
import type { SettingsDefaults } from '../../shared/SettingsDefaultsManager.js';
import { USER_SETTINGS_PATH } from '../../shared/paths.js';
import { findCodexExecutable } from '../../shared/find-codex-executable.js';
import { logger } from '../../utils/logger.js';
import type { ActiveSession, ConversationMessage } from '../worker-types.js';
import { DatabaseManager } from './DatabaseManager.js';
import { SessionManager } from './SessionManager.js';
import {
  classifyCodexError,
  renderCodexConversation,
  runCodexCli,
} from './CodexCliRunner.js';
import {
  OpenAICompatibleProvider,
  type ProviderQueryResult,
} from './OpenAICompatibleProvider.js';
import { resolveTierAlias } from './model-aliases.js';

interface CodexConfig {
  executable: string;
  model: string;
  reasoningEffort: string;
  timeoutMs: number;
}

const CHARS_PER_TOKEN_ESTIMATE = 4;
const VALID_REASONING_EFFORTS = new Set([
  'minimal',
  'low',
  'medium',
  'high',
  'xhigh',
  'max',
]);
const SIMPLE_TOOLS = new Set([
  'Read',
  'Glob',
  'Grep',
  'LS',
  'ListMcpResourcesTool',
]);

export function resolveCodexTierModel(
  defaultModel: string,
  requestKind: 'init' | 'observation',
  toolName: string | undefined,
  settings: SettingsDefaults,
): string {
  if (settings.CLAUDE_MEM_TIER_ROUTING_ENABLED === 'false') return defaultModel;
  if (requestKind === 'init' || (toolName && SIMPLE_TOOLS.has(toolName))) {
    return settings.CLAUDE_MEM_TIER_SIMPLE_MODEL || 'gpt-5.6-luna';
  }
  return settings.CLAUDE_MEM_TIER_SMART_MODEL || settings.CLAUDE_MEM_MODEL || defaultModel;
}

export class CodexProvider extends OpenAICompatibleProvider<CodexConfig> {
  protected readonly providerName = 'Codex';
  protected readonly syntheticIdPrefix = 'codex';
  protected readonly forwardEmptyMessageResponse = false;

  constructor(dbManager: DatabaseManager, sessionManager: SessionManager) {
    super(dbManager, sessionManager);
  }

  protected getConfig(): CodexConfig {
    const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);
    const model = resolveTierAlias(settings.CLAUDE_MEM_MODEL, settings);
    const requestedEffort = settings.CLAUDE_MEM_CODEX_REASONING_EFFORT.trim().toLowerCase();
    const reasoningEffort = VALID_REASONING_EFFORTS.has(requestedEffort)
      ? requestedEffort
      : 'low';
    const parsedTimeout = Number.parseInt(settings.CLAUDE_MEM_CODEX_TIMEOUT_MS, 10);
    const timeoutMs = Number.isFinite(parsedTimeout) && parsedTimeout >= 10_000
      ? parsedTimeout
      : 180_000;

    try {
      return {
        executable: findCodexExecutable('SDK'),
        model,
        reasoningEffort,
        timeoutMs,
      };
    } catch (error) {
      throw classifyCodexError(error);
    }
  }

  protected assertConfigured(config: CodexConfig): void {
    if (!config.executable) {
      throw new Error('Codex CLI is not configured.');
    }
  }

  protected configForInit(config: CodexConfig): CodexConfig {
    const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);
    const model = resolveCodexTierModel(config.model, 'init', undefined, settings);
    return model === config.model ? config : { ...config, model };
  }

  protected configForObservation(
    config: CodexConfig,
    message: { tool_name?: string },
  ): CodexConfig {
    const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);
    const model = resolveCodexTierModel(
      config.model,
      'observation',
      message.tool_name,
      settings,
    );
    return model === config.model ? config : { ...config, model };
  }

  protected estimateTokens(text: string): number {
    return Math.ceil(text.length / CHARS_PER_TOKEN_ESTIMATE);
  }

  protected buildLastUsage(result: ProviderQueryResult): ActiveSession['lastUsage'] {
    if (
      typeof result.inputTokens !== 'number' ||
      typeof result.outputTokens !== 'number'
    ) {
      return null;
    }
    return {
      input: result.inputTokens,
      output: result.outputTokens,
    };
  }

  protected async query(
    history: ConversationMessage[],
    config: CodexConfig,
  ): Promise<ProviderQueryResult> {
    const prompt = renderCodexConversation(history);
    logger.debug('SDK', `Querying Codex CLI (${config.model})`, {
      turns: history.length,
      estimatedTokens: this.estimateTokens(prompt),
      reasoningEffort: config.reasoningEffort,
    });

    const result = await runCodexCli({
      executable: config.executable,
      model: config.model,
      reasoningEffort: config.reasoningEffort,
      timeoutMs: config.timeoutMs,
      prompt,
    });

    const tokensUsed =
      typeof result.inputTokens === 'number' && typeof result.outputTokens === 'number'
        ? result.inputTokens + result.outputTokens
        : undefined;

    logger.info('SDK', 'Codex CLI usage', {
      model: config.model,
      inputTokens: result.inputTokens ?? 0,
      outputTokens: result.outputTokens ?? 0,
      totalTokens: tokensUsed ?? 0,
    });

    return {
      content: result.content,
      tokensUsed,
      inputTokens: result.inputTokens,
      outputTokens: result.outputTokens,
      servedModel: config.model,
    };
  }
}

export function isCodexSelected(): boolean {
  const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);
  return settings.CLAUDE_MEM_PROVIDER === 'codex';
}
