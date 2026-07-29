import { execFileSync, execSync } from 'child_process';
import { existsSync, realpathSync } from 'fs';
import { homedir } from 'os';
import { join } from 'path';
import { SettingsDefaultsManager } from './SettingsDefaultsManager.js';
import { USER_SETTINGS_PATH, expandTilde } from './paths.js';
import { logger, type Component } from '../utils/logger.js';

const VERSION_CHECK_TIMEOUT_MS = 10_000;
const RESOLUTION_CACHE_TTL_MS = 15 * 60_000;

interface CachedResolution {
  path: string;
  version: string;
  expiresAtMs: number;
}

let cachedResolution: CachedResolution | null = null;

export function resetCodexExecutableCache(): void {
  cachedResolution = null;
}

export const _internals = {
  execSync,
  execFileSync,
  existsSync,
  realpathSync,
  homedir,
  platform: (): NodeJS.Platform => process.platform,
  loadSettings: () => SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH),
};

function probeCandidate(candidate: string): string | null {
  try {
    const version = _internals.execFileSync(candidate, ['--version'], {
      encoding: 'utf8',
      timeout: VERSION_CHECK_TIMEOUT_MS,
      windowsHide: true,
      stdio: ['ignore', 'pipe', 'pipe'],
    }).trim();
    return version || null;
  } catch {
    return null;
  }
}

function discoverCandidates(): string[] {
  const candidates: string[] = [];

  if (_internals.platform() === 'win32') {
    for (const command of ['where codex.cmd', 'where codex']) {
      try {
        const output = _internals.execSync(command, {
          encoding: 'utf8',
          windowsHide: true,
          stdio: ['ignore', 'pipe', 'ignore'],
        });
        candidates.push(...output.split('\n').map(line => line.trim()).filter(Boolean));
      } catch {
        // Try the remaining discovery locations.
      }
    }
  } else {
    try {
      const output = _internals.execSync('which -a codex', {
        encoding: 'utf8',
        windowsHide: true,
        stdio: ['ignore', 'pipe', 'ignore'],
      });
      candidates.push(...output.split('\n').map(line => line.trim()).filter(Boolean));
    } catch {
      // Known local install locations below may still exist.
    }
    for (const knownPath of [
      join(_internals.homedir(), '.local', 'bin', 'codex'),
      join(_internals.homedir(), '.npm-global', 'bin', 'codex'),
    ]) {
      if (_internals.existsSync(knownPath)) candidates.push(knownPath);
    }
  }

  const seenLiteral = new Set<string>();
  const seenReal = new Set<string>();
  return candidates.filter(candidate => {
    if (seenLiteral.has(candidate)) return false;
    seenLiteral.add(candidate);
    let realPath = candidate;
    try {
      realPath = _internals.realpathSync(candidate);
    } catch {
      // The probe below will reject dangling or unreadable candidates.
    }
    if (seenReal.has(realPath)) return false;
    seenReal.add(realPath);
    return true;
  });
}

export function findCodexExecutable(logComponent: Component = 'SDK'): string {
  if (
    cachedResolution &&
    cachedResolution.expiresAtMs > Date.now() &&
    _internals.existsSync(cachedResolution.path)
  ) {
    return cachedResolution.path;
  }
  cachedResolution = null;

  const settings = _internals.loadSettings();
  if (settings.CLAUDE_MEM_CODEX_PATH) {
    const configuredPath = expandTilde(
      settings.CLAUDE_MEM_CODEX_PATH,
      _internals.homedir(),
    );
    if (!_internals.existsSync(configuredPath)) {
      throw new Error(
        `CLAUDE_MEM_CODEX_PATH is set to "${settings.CLAUDE_MEM_CODEX_PATH}" but the file does not exist.`,
      );
    }
    const version = probeCandidate(configuredPath);
    if (!version) {
      throw new Error(
        `CLAUDE_MEM_CODEX_PATH is set to "${settings.CLAUDE_MEM_CODEX_PATH}" but it failed the --version check.`,
      );
    }
    cachedResolution = {
      path: configuredPath,
      version,
      expiresAtMs: Date.now() + RESOLUTION_CACHE_TTL_MS,
    };
    logger.info(logComponent, `Using configured Codex CLI: ${configuredPath} (${version})`);
    return configuredPath;
  }

  for (const candidate of discoverCandidates()) {
    const version = probeCandidate(candidate);
    if (!version) continue;
    cachedResolution = {
      path: candidate,
      version,
      expiresAtMs: Date.now() + RESOLUTION_CACHE_TTL_MS,
    };
    logger.info(logComponent, `Using Codex CLI: ${candidate} (${version})`);
    return candidate;
  }

  throw new Error(
    'Codex CLI executable not found. Install @openai/codex, run `codex login`, or set CLAUDE_MEM_CODEX_PATH in ~/.claude-mem/settings.json.',
  );
}
