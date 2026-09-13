import { cp, mkdir, writeFile } from 'node:fs/promises';
const raw = process.env.RAG_BACKEND_URL;
if (!raw) throw new Error('Set RAG_BACKEND_URL to your deployed Python backend HTTPS origin.');
const url = new URL(raw);
if (url.protocol !== 'https:' || url.username || url.password || url.search || url.hash || !['', '/'].includes(url.pathname)) {
  throw new Error('RAG_BACKEND_URL must be a plain HTTPS origin without a path or credentials.');
}
await mkdir('dist/assets', {recursive: true});
await cp('web/index.html', 'dist/index.html');
for (const name of ['app.js', 'style.css']) await cp(`web/${name}`, `dist/assets/${name}`);
await writeFile('dist/_redirects', `/api/* ${url.origin}/api/:splat 200\n`);
console.log('Netlify frontend built with Python API proxy.');
