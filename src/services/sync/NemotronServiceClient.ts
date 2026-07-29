import { execFile } from 'child_process';
import { promisify } from 'util';
import { logger } from '../../utils/logger.js';
import { SettingsDefaultsManager } from '../../shared/SettingsDefaultsManager.js';
import { USER_SETTINGS_PATH } from '../../shared/paths.js';
import { ChromaUnavailableError } from '../worker/search/errors.js';

const execFileAsync = promisify(execFile);
const LAUNCH_AGENT_LABEL = 'ai.claude-mem.nemotron';
const DEFAULT_TIMEOUT_MS = 180_000;

interface VectorCallResponse {
  result: unknown;
}

/**
 * HTTP client for the single user-level Nemotron service.
 *
 * Claude, Codex, and every concurrent session share this URL, so only the
 * LaunchAgent owns model weights. This client deliberately preserves the
 * chroma-mcp tool contract to keep the mature sync/search code unchanged.
 */
export class NemotronServiceClient {
  private static instance: NemotronServiceClient | null = null;
  private kickstartAttempted = false;

  static getInstance(): NemotronServiceClient {
    if (!this.instance) {
      this.instance = new NemotronServiceClient();
    }
    return this.instance;
  }

  static isEnabled(): boolean {
    const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);
    return settings.CLAUDE_MEM_EMBEDDING_PROVIDER === 'nemotron';
  }

  private getSettings(): {
    baseUrl: string;
    timeoutMs: number;
    autoStart: boolean;
  } {
    const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);
    const parsedTimeout = Number.parseInt(settings.CLAUDE_MEM_NEMOTRON_REQUEST_TIMEOUT_MS, 10);
    return {
      baseUrl: settings.CLAUDE_MEM_NEMOTRON_URL.replace(/\/+$/, ''),
      timeoutMs: Number.isFinite(parsedTimeout) && parsedTimeout > 0
        ? parsedTimeout
        : DEFAULT_TIMEOUT_MS,
      autoStart: settings.CLAUDE_MEM_NEMOTRON_AUTO_START !== 'false',
    };
  }

  async callTool(toolName: string, toolArguments: Record<string, unknown>): Promise<unknown> {
    const settings = this.getSettings();
    try {
      return await this.request(toolName, toolArguments, settings);
    } catch (error) {
      if (!settings.autoStart || !this.isTransportError(error)) {
        throw this.asUnavailableError(toolName, error);
      }

      await this.kickstart();
      for (let attempt = 0; attempt < 20; attempt += 1) {
        await new Promise(resolve => setTimeout(resolve, 250));
        try {
          return await this.request(toolName, toolArguments, settings);
        } catch (retryError) {
          if (!this.isTransportError(retryError) || attempt === 19) {
            throw this.asUnavailableError(toolName, retryError);
          }
        }
      }
      throw this.asUnavailableError(toolName, error);
    }
  }

  private async request(
    toolName: string,
    toolArguments: Record<string, unknown>,
    settings: { baseUrl: string; timeoutMs: number },
  ): Promise<unknown> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), settings.timeoutMs);
    try {
      const response = await fetch(`${settings.baseUrl}/v1/vector/call`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          tool_name: toolName,
          arguments: toolArguments,
        }),
        signal: controller.signal,
      });
      const bodyText = await response.text();
      if (!response.ok) {
        let detail = bodyText;
        try {
          const parsed = JSON.parse(bodyText) as { detail?: unknown };
          detail = typeof parsed.detail === 'string'
            ? parsed.detail
            : JSON.stringify(parsed.detail ?? parsed);
        } catch {
          // Preserve the raw response text.
        }
        throw new Error(`Nemotron service ${response.status}: ${detail}`);
      }
      const parsed = JSON.parse(bodyText) as VectorCallResponse;
      return parsed.result;
    } finally {
      clearTimeout(timeout);
    }
  }

  async isHealthy(): Promise<boolean> {
    const settings = this.getSettings();
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 2_000);
    try {
      const response = await fetch(`${settings.baseUrl}/readyz`, {
        signal: controller.signal,
      });
      return response.ok;
    } catch {
      return false;
    } finally {
      clearTimeout(timeout);
    }
  }

  private async kickstart(): Promise<void> {
    if (this.kickstartAttempted || process.platform !== 'darwin') {
      return;
    }
    this.kickstartAttempted = true;
    const uid = process.getuid?.();
    if (uid === undefined) {
      return;
    }
    try {
      await execFileAsync('/bin/launchctl', [
        'kickstart',
        '-k',
        `gui/${uid}/${LAUNCH_AGENT_LABEL}`,
      ]);
      logger.info('NEMOTRON', 'Requested shared embedding service start via launchd');
    } catch (error) {
      logger.debug('NEMOTRON', 'LaunchAgent kickstart unavailable; search will use SQLite fallback', {
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  private isTransportError(error: unknown): boolean {
    const message = error instanceof Error ? error.message : String(error);
    return /fetch failed|ECONNREFUSED|ENOTFOUND|aborted|network/i.test(message);
  }

  private asUnavailableError(toolName: string, error: unknown): ChromaUnavailableError {
    const cause = error instanceof Error ? error : new Error(String(error));
    return new ChromaUnavailableError(
      `Shared Nemotron vector service failed during "${toolName}": ${cause.message}`,
      cause,
    );
  }
}
