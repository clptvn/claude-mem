import { afterEach, describe, expect, it } from 'bun:test';
import {
  _internals,
  findCodexExecutable,
  resetCodexExecutableCache,
} from '../../src/shared/find-codex-executable.js';
import { SettingsDefaultsManager } from '../../src/shared/SettingsDefaultsManager.js';

const originals = { ..._internals };

afterEach(() => {
  Object.assign(_internals, originals);
  resetCodexExecutableCache();
});

describe('findCodexExecutable', () => {
  it('honors an explicit configured path after expanding tilde', () => {
    _internals.homedir = () => '/home/tester';
    _internals.loadSettings = () => ({
      ...SettingsDefaultsManager.getAllDefaults(),
      CLAUDE_MEM_CODEX_PATH: '~/.local/bin/codex',
    });
    _internals.existsSync = path => path === '/home/tester/.local/bin/codex';
    _internals.execFileSync = ((path: string, args: string[]) => {
      expect(path).toBe('/home/tester/.local/bin/codex');
      expect(args).toEqual(['--version']);
      return 'codex-cli 0.146.0\n';
    }) as typeof _internals.execFileSync;

    expect(findCodexExecutable()).toBe('/home/tester/.local/bin/codex');
  });

  it('discovers the known local install when daemon PATH is minimal', () => {
    _internals.platform = () => 'darwin';
    _internals.homedir = () => '/home/tester';
    _internals.loadSettings = () => ({
      ...SettingsDefaultsManager.getAllDefaults(),
      CLAUDE_MEM_CODEX_PATH: '',
    });
    _internals.execSync = (() => {
      throw new Error('which found nothing');
    }) as typeof _internals.execSync;
    _internals.existsSync = path => path === '/home/tester/.local/bin/codex';
    _internals.realpathSync = path => path;
    _internals.execFileSync = (() => 'codex-cli 0.146.0\n') as typeof _internals.execFileSync;

    expect(findCodexExecutable()).toBe('/home/tester/.local/bin/codex');
  });
});
