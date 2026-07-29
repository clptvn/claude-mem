#!/usr/bin/env node

import { execFileSync, spawnSync } from 'node:child_process';
import {
  cpSync,
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { homedir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const LABEL = 'ai.claude-mem.nemotron';
const PORT = Number.parseInt(process.env.CLAUDE_MEM_NEMOTRON_PORT ?? '37901', 10);
if (!Number.isInteger(PORT) || PORT < 1024 || PORT > 65535) {
  fail(`Invalid CLAUDE_MEM_NEMOTRON_PORT: ${process.env.CLAUDE_MEM_NEMOTRON_PORT}`);
}
const ROOT = join(homedir(), '.claude-mem', 'nemotron-runtime');
const APP_DIR = join(ROOT, 'app');
const VENV_DIR = join(ROOT, 'venv');
const PYTHON = join(VENV_DIR, 'bin', 'python');
const LOG_DIR = join(homedir(), '.claude-mem', 'logs');
const SETTINGS_PATH = join(homedir(), '.claude-mem', 'settings.json');
const TELEMETRY_PATH = join(homedir(), '.claude-mem', 'telemetry.json');
const PLIST = join(homedir(), 'Library', 'LaunchAgents', `${LABEL}.plist`);
const SOURCE_DIR = resolve(dirname(fileURLToPath(import.meta.url)), '..', 'services', 'nemotron');
const HEALTH_URL = `http://127.0.0.1:${PORT}`;
const OPTIONAL_SERVICE_ENV = [
  'CLAUDE_MEM_NEMOTRON_MODEL',
  'CLAUDE_MEM_NEMOTRON_MAX_TOKENS',
  'CLAUDE_MEM_NEMOTRON_BATCH_SIZE',
  'CLAUDE_MEM_NEMOTRON_QUEUE_BATCH_SIZE',
  'CLAUDE_MEM_NEMOTRON_BATCH_WAIT_MS',
  'CLAUDE_MEM_NEMOTRON_WRITE_TIMEOUT_SECONDS',
  'CLAUDE_MEM_NEMOTRON_EAGER_LOAD',
];

function fail(message) {
  process.stderr.write(`${message}\n`);
  process.exit(1);
}

function xml(value) {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&apos;');
}

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    encoding: 'utf8',
    stdio: options.capture ? 'pipe' : 'inherit',
    ...options,
  });
  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0 && !options.allowFailure) {
    const detail = result.stderr?.trim();
    throw new Error(`${command} exited ${result.status}${detail ? `: ${detail}` : ''}`);
  }
  return result;
}

function commandPath(name) {
  try {
    return execFileSync('/usr/bin/which', [name], { encoding: 'utf8' }).trim();
  } catch {
    return '';
  }
}

function assertMac() {
  if (process.platform !== 'darwin' || process.arch !== 'arm64') {
    fail('The bundled Nemotron runtime currently supports Apple Silicon macOS only.');
  }
}

function plistContents() {
  const args = [PYTHON, '-m', 'nemotron_memory_service'];
  const argXml = args.map(arg => `      <string>${xml(arg)}</string>`).join('\n');
  const optionalEnvironment = OPTIONAL_SERVICE_ENV
    .filter(key => process.env[key] !== undefined)
    .map(key => `    <key>${key}</key>\n    <string>${xml(process.env[key])}</string>`)
    .join('\n');
  return `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>
  <key>ProgramArguments</key>
  <array>
${argXml}
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>ThrottleInterval</key>
  <integer>10</integer>
  <key>WorkingDirectory</key>
  <string>${xml(APP_DIR)}</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PYTORCH_ENABLE_MPS_FALLBACK</key>
    <string>1</string>
    <key>CLAUDE_MEM_NEMOTRON_PORT</key>
    <string>${PORT}</string>
    <key>CLAUDE_MEM_NEMOTRON_DATA_DIR</key>
    <string>${xml(join(homedir(), '.claude-mem', 'nemotron'))}</string>
    <key>ANONYMIZED_TELEMETRY</key>
    <string>false</string>
${optionalEnvironment ? `${optionalEnvironment}\n` : ''}    <key>PATH</key>
    <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
  <key>StandardOutPath</key>
  <string>${xml(join(LOG_DIR, 'nemotron-service.log'))}</string>
  <key>StandardErrorPath</key>
  <string>${xml(join(LOG_DIR, 'nemotron-service.error.log'))}</string>
</dict>
</plist>
`;
}

function launchDomain() {
  return `gui/${process.getuid()}`;
}

function bootout() {
  run('/bin/launchctl', ['bootout', launchDomain(), PLIST], {
    capture: true,
    allowFailure: true,
  });
}

function updateSettings(updates) {
  mkdirSync(dirname(SETTINGS_PATH), { recursive: true });
  let settings = {};
  if (existsSync(SETTINGS_PATH)) {
    try {
      const parsed = JSON.parse(readFileSync(SETTINGS_PATH, 'utf8'));
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        settings = parsed;
      }
    } catch (error) {
      throw new Error(`Cannot safely update ${SETTINGS_PATH}: ${error.message}`);
    }
  }
  const next = { ...settings, ...updates };
  const temporaryPath = `${SETTINGS_PATH}.nemotron-${process.pid}.tmp`;
  writeFileSync(temporaryPath, `${JSON.stringify(next, null, 2)}\n`, {
    encoding: 'utf8',
    mode: 0o600,
  });
  renameSync(temporaryPath, SETTINGS_PATH);
}

function disableClaudeMemTelemetry() {
  let telemetry = {};
  if (existsSync(TELEMETRY_PATH)) {
    try {
      const parsed = JSON.parse(readFileSync(TELEMETRY_PATH, 'utf8'));
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        telemetry = parsed;
      }
    } catch (error) {
      throw new Error(`Cannot safely update ${TELEMETRY_PATH}: ${error.message}`);
    }
  }
  const next = {
    ...telemetry,
    enabled: false,
    installId: typeof telemetry.installId === 'string' ? telemetry.installId : '',
    decidedAt: new Date().toISOString(),
  };
  const temporaryPath = `${TELEMETRY_PATH}.nemotron-${process.pid}.tmp`;
  writeFileSync(temporaryPath, `${JSON.stringify(next, null, 2)}\n`, {
    encoding: 'utf8',
    mode: 0o600,
  });
  renameSync(temporaryPath, TELEMETRY_PATH);
}

async function fetchJson(path, options) {
  const response = await fetch(`${HEALTH_URL}${path}`, options);
  const text = await response.text();
  let body;
  try {
    body = JSON.parse(text);
  } catch {
    body = text;
  }
  if (!response.ok) {
    const error = new Error(`HTTP ${response.status}: ${text}`);
    error.status = response.status;
    throw error;
  }
  return body;
}

async function waitUntilReady(timeoutMs = 300_000) {
  const deadline = Date.now() + timeoutMs;
  let lastState = '';
  while (Date.now() < deadline) {
    try {
      const health = await fetchJson('/healthz');
      const state = health?.model?.state ?? 'unknown';
      if (state !== lastState) {
        process.stdout.write(`Nemotron model state: ${state}\n`);
        lastState = state;
      }
      if (state === 'failed') {
        throw new Error(health?.model?.error ?? 'model load failed');
      }
      if (state === 'ready') {
        return health;
      }
    } catch (error) {
      if (error?.status && error.status !== 503) {
        throw error;
      }
    }
    await new Promise(resolve => setTimeout(resolve, 1_000));
  }
  throw new Error(`Nemotron service did not become ready within ${timeoutMs / 1000}s`);
}

async function install() {
  assertMac();
  const uv = commandPath('uv');
  if (!uv) {
    fail('uv is required. Install it from https://docs.astral.sh/uv/ and retry.');
  }
  if (!existsSync(SOURCE_DIR)) {
    fail(`Nemotron service source not found at ${SOURCE_DIR}`);
  }

  mkdirSync(ROOT, { recursive: true });
  mkdirSync(LOG_DIR, { recursive: true });
  mkdirSync(dirname(PLIST), { recursive: true });
  if (existsSync(APP_DIR)) {
    rmSync(APP_DIR, { recursive: true, force: true });
  }
  cpSync(SOURCE_DIR, APP_DIR, {
    recursive: true,
    filter: source => !source.includes('__pycache__') && !source.endsWith('.pyc'),
  });

  if (!existsSync(PYTHON)) {
    run(uv, ['venv', VENV_DIR, '--python', '3.11']);
  }
  run(uv, ['pip', 'install', '--python', PYTHON, '--upgrade', APP_DIR]);

  bootout();
  writeFileSync(PLIST, plistContents(), { encoding: 'utf8', mode: 0o600 });
  run('/bin/launchctl', ['bootstrap', launchDomain(), PLIST]);
  run('/bin/launchctl', ['kickstart', '-k', `${launchDomain()}/${LABEL}`]);

  const health = await waitUntilReady();
  updateSettings({
    CLAUDE_MEM_CHROMA_ENABLED: 'true',
    CLAUDE_MEM_EMBEDDING_PROVIDER: 'nemotron',
    CLAUDE_MEM_NEMOTRON_URL: HEALTH_URL,
    CLAUDE_MEM_NEMOTRON_AUTO_START: 'true',
    CLAUDE_MEM_NEMOTRON_REQUEST_TIMEOUT_MS: '180000',
    CLAUDE_MEM_SEMANTIC_INJECT: 'true',
  });
  disableClaudeMemTelemetry();
  process.stdout.write(
    `Nemotron service ready on ${HEALTH_URL} (${health.model.device}, ` +
      `${health.model.embedding_dimensions} dimensions, ` +
      `${health.model.load_seconds?.toFixed?.(2) ?? health.model.load_seconds}s load). ` +
      'Claude-Mem and Chroma telemetry are disabled.\n',
  );
}

async function status() {
  try {
    const health = await fetchJson('/healthz');
    process.stdout.write(`${JSON.stringify(health, null, 2)}\n`);
  } catch (error) {
    const listing = run('/bin/launchctl', ['print', `${launchDomain()}/${LABEL}`], {
      capture: true,
      allowFailure: true,
    });
    if (listing.status === 0) {
      process.stdout.write('LaunchAgent is registered but its HTTP service is unavailable.\n');
      process.stdout.write(`${listing.stdout}\n`);
    } else {
      fail(`Nemotron service is not installed or running: ${error.message}`);
    }
  }
}

async function start() {
  assertMac();
  if (!existsSync(PLIST)) {
    fail('Nemotron LaunchAgent is not installed. Run npm run nemotron:install first.');
  }
  run('/bin/launchctl', ['bootstrap', launchDomain(), PLIST], {
    capture: true,
    allowFailure: true,
  });
  run('/bin/launchctl', ['kickstart', '-k', `${launchDomain()}/${LABEL}`]);
  await waitUntilReady();
  process.stdout.write('Nemotron service started.\n');
}

function stop() {
  assertMac();
  bootout();
  process.stdout.write('Nemotron service stopped. Its model and vector data remain installed.\n');
}

function uninstall() {
  assertMac();
  bootout();
  if (existsSync(PLIST)) {
    rmSync(PLIST);
  }
  let settings = {};
  if (existsSync(SETTINGS_PATH)) {
    settings = JSON.parse(readFileSync(SETTINGS_PATH, 'utf8'));
  }
  if (settings.CLAUDE_MEM_EMBEDDING_PROVIDER === 'nemotron') {
    updateSettings({ CLAUDE_MEM_EMBEDDING_PROVIDER: 'chroma' });
  }
  process.stdout.write(
    `Nemotron LaunchAgent removed. Runtime and memory data remain under ${join(homedir(), '.claude-mem')}.\n`,
  );
}

async function smoke() {
  const suffix = `${process.pid}-${Date.now()}`;
  const collection = `cm__smoke_${suffix}`;
  const call = async (toolName, args) => {
    const body = await fetchJson('/v1/vector/call', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ tool_name: toolName, arguments: args }),
    });
    return body.result;
  };

  await call('chroma_create_collection', { collection_name: collection });
  try {
    await call('chroma_add_documents', {
      collection_name: collection,
      ids: ['obs_1_narrative', 'obs_2_narrative'],
      documents: [
        'Claude and Codex share one durable local memory service backed by Nemotron embeddings.',
        'A sourdough starter is fed with flour and water.',
      ],
      metadatas: [
        { sqlite_id: 1, doc_type: 'observation', created_at_epoch: Date.now() },
        { sqlite_id: 2, doc_type: 'observation', created_at_epoch: Date.now() },
      ],
    });
    const result = await call('chroma_query_documents', {
      collection_name: collection,
      query_texts: ['How do the AI coding agents remember across sessions?'],
      n_results: 2,
      include: ['documents', 'metadatas', 'distances'],
    });
    if (result?.ids?.[0]?.[0] !== 'obs_1_narrative') {
      throw new Error(`Unexpected top result: ${JSON.stringify(result)}`);
    }
    process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
    process.stdout.write('Nemotron embedding and vector smoke test passed.\n');
  } finally {
    await call('chroma_delete_collection', { collection_name: collection });
  }
}

const command = process.argv[2] ?? 'status';
const commands = { install, status, start, stop, uninstall, smoke };
if (!commands[command]) {
  const ownSource = readFileSync(fileURLToPath(import.meta.url), 'utf8');
  fail(
    `Unknown command "${command}". Use one of: ${Object.keys(commands).join(', ')} ` +
      `(launcher ${ownSource.length} bytes).`,
  );
}

await commands[command]();
