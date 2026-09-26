// Offline renderer: loads scene.html in headless Chromium (WebGL via SwiftShader),
// renders every frame of a flight log and pipes PNG frames into ffmpeg.
//
//   node render3d/render.mjs flight.json out.mp4 /path/to/ffmpeg
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { dirname, extname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const here = dirname(fileURLToPath(import.meta.url));
const [dataPath, outPath, ffmpeg = 'ffmpeg'] = process.argv.slice(2);
if (!dataPath || !outPath) {
  console.error('usage: node render.mjs flight.json out.mp4 [ffmpeg]');
  process.exit(2);
}
const data = readFileSync(dataPath);
const { width, height, fps, frames } = JSON.parse(data);

function findChromium() {
  if (process.env.CHROMIUM_PATH) return process.env.CHROMIUM_PATH;
  const base = process.env.PLAYWRIGHT_BROWSERS_PATH || '/opt/pw-browsers';
  if (existsSync(base)) {
    for (const d of readdirSync(base).filter((d) => /^chromium-\d+$/.test(d)).sort().reverse()) {
      const exe = join(base, d, 'chrome-linux', 'chrome');
      if (existsSync(exe)) return exe;
    }
  }
  return undefined; // let playwright-core try its own default
}

const types = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json' };
const server = createServer((req, res) => {
  const url = decodeURIComponent(req.url.split('?')[0]);
  if (url === '/data.json') {
    res.writeHead(200, { 'content-type': 'application/json' });
    return res.end(data);
  }
  const file = join(here, url === '/' ? 'scene.html' : url);
  if (!file.startsWith(here) || !existsSync(file)) {
    res.writeHead(404);
    return res.end();
  }
  res.writeHead(200, { 'content-type': types[extname(file)] || 'application/octet-stream' });
  res.end(readFileSync(file));
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const port = server.address().port;

const browser = await chromium.launch({
  executablePath: findChromium(),
  args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
});
const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
page.on('console', (m) => { if (m.type() === 'error') console.error('[page]', m.text()); });
page.on('pageerror', (e) => console.error('[page error]', e.message));
await page.goto(`http://127.0.0.1:${port}/`);
await page.waitForFunction(() => window.sceneReady === true, null, { timeout: 120000 });

// Preview mode: PREVIEW_FRAMES=0,200,400 writes <out>/frame_XXXX.png instead of a video.
if (process.env.PREVIEW_FRAMES) {
  const { mkdirSync, writeFileSync } = await import('node:fs');
  mkdirSync(outPath, { recursive: true });
  for (const k of process.env.PREVIEW_FRAMES.split(',').map(Number)) {
    const t = Date.now();
    await page.evaluate((j) => window.renderFrame(j), Math.min(k, frames.length - 1));
    writeFileSync(join(outPath, `frame_${String(k).padStart(4, '0')}.png`), await page.screenshot({ type: 'png' }));
    console.log(`  preview frame ${k} (${Date.now() - t} ms)`);
  }
  await browser.close();
  server.close();
  process.exit(0);
}

const enc = spawn(ffmpeg, [
  '-loglevel', 'error', '-y', '-f', 'image2pipe', '-framerate', String(fps), '-i', '-',
  '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '19', '-preset', 'medium', '-movflags', '+faststart', outPath,
], { stdio: ['pipe', 'inherit', 'inherit'] });

const t0 = Date.now();
for (let i = 0; i < frames.length; i++) {
  await page.evaluate((k) => window.renderFrame(k), i);
  const png = await page.screenshot({ type: 'png' });
  if (!enc.stdin.write(png)) await new Promise((r) => enc.stdin.once('drain', r));
  if (i % 60 === 0) {
    const s = (Date.now() - t0) / 1000;
    console.log(`  frame ${i}/${frames.length}  (${(i / Math.max(s, 1e-3)).toFixed(1)} fps)`);
  }
}
enc.stdin.end();
await new Promise((r) => enc.on('close', r));
await browser.close();
server.close();
console.log(`  done in ${((Date.now() - t0) / 1000).toFixed(0)} s`);
