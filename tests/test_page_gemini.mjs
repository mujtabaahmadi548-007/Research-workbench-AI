// Gemini request rules in the page (index.html): gemBody() and gemErrFrom(), extracted and run in Node.
// Run:  node tests/test_page_gemini.mjs
import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';

const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
const grab = re => { const m = html.match(re); assert.ok(m, 'not found in index.html: ' + re); return m[0]; };
const src = [
  grab(/const GEMINI4_LEVELS = .*;/),
  grab(/const GEMINI4_MIN_OUTPUT = .*;/),
  grab(/const isGemini4 = .*;/),
  grab(/class GemErr extends Error \{[^\n]*\}/),
  grab(/function gemErrFrom\([\s\S]*?\n\}\n/),
  grab(/function gemBody\([\s\S]*?\n\}\n/),
].join('\n');
const { gemBody, gemErrFrom } = new Function('STATIC', src + '\nreturn { gemBody, gemErrFrom };')(true);

let n = 0;
const test = (name, fn) => { fn(); n++; console.log('ok -', name); };
const rules4 = (body, level) => {
  const gc = body.generationConfig, text = JSON.stringify(body);
  for (const k of ['temperature', 'topP', 'topK']) assert.ok(!(k in gc), k + ' must not be sent to Gemini 4');
  assert.deepEqual(gc.thinkingConfig, { thinkingLevel: level });
  assert.ok(gc.maxOutputTokens >= 32768, 'maxOutputTokens ' + gc.maxOutputTokens);
  for (const bad of ['thinkingBudget', 'minimal', 'cachedContent']) assert.ok(!text.includes(bad), bad + ' in body');
};

test('hydrogen + schema + "minimal" requested -> low, no sampling params, 32768+', () =>
  rules4(gemBody({ prompt: 'p', system: 's', schema: { type: 'object' }, thinking: 'minimal', maxTokens: 8192 }, 'gemini-4-hydrogen', true), 'low'));
test('hydrogen Triage (tools, medium)', () =>
  rules4(gemBody({ prompt: 'p', system: 's', tools: [{ url_context: {} }], thinking: 'medium', json: false }, 'gemini-4-hydrogen', true), 'medium'));
test('hydrogen plain JSON mode never gets temperature', () =>
  rules4(gemBody({ prompt: 'p', system: 's', maxTokens: 2048 }, 'gemini-4-hydrogen', true), 'low'));
test('gemini-3.8-flash keeps Gemini 3 rules', () => {
  const gc = gemBody({ prompt: 'p', system: 's', schema: { type: 'object' }, thinking: 'low' }, 'gemini-3.8-flash', true).generationConfig;
  assert.deepEqual(gc.thinkingConfig, { thinkingLevel: 'low' });
});
test('error kinds: 503 server, 429 rate_limit, 404 model, 403/402 denied, bad key auth', () => {
  assert.equal(gemErrFrom(503, { error: { message: 'overloaded' } }, 'm').kind, 'server');
  assert.equal(gemErrFrom(429, { error: { message: 'quota' } }, 'm').kind, 'rate_limit');
  assert.equal(gemErrFrom(404, { error: { message: 'not found' } }, 'm').kind, 'model');
  assert.equal(gemErrFrom(403, { error: { message: 'Permission denied' } }, 'm').kind, 'denied');
  assert.equal(gemErrFrom(402, { error: { message: 'Billing required' } }, 'm').kind, 'denied');
  assert.equal(gemErrFrom(403, { error: { message: 'API key not valid' } }, 'm').kind, 'auth');
});
test('page chain and defaults', () => {
  assert.ok(html.includes("const DEFAULT_DEEP_MODEL = 'gemini-4-hydrogen';"));
  assert.ok(html.includes("const DEEP_FALLBACKS = ['gemini-3.8-flash', 'gemini-3.5-flash-lite'];"));
  assert.ok(/async function runTriage[\s\S]*?thinking: 'medium'/.test(html), 'Triage asks for medium thinking');
});
console.log(`${n} page tests passed`);
