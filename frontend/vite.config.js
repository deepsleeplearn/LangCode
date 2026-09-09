import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
import { readdir, readFile, writeFile } from 'node:fs/promises';
import { isAbsolute, join, resolve } from 'node:path';
import { brotliCompress, constants as zlibConstants, gzip } from 'node:zlib';
import { promisify } from 'node:util';

const gzipAsync = promisify(gzip);
const brotliAsync = promisify(brotliCompress);

// Fixed levels and an explicit size hint keep both encoders byte-for-byte reproducible for
// the same input, so a rebuild that changes nothing produces identical .gz/.br artifacts.
const GZIP_OPTIONS = { level: 9 };

function brotliOptions(size) {
  return {
    params: {
      [zlibConstants.BROTLI_PARAM_QUALITY]: zlibConstants.BROTLI_MAX_QUALITY,
      [zlibConstants.BROTLI_PARAM_SIZE_HINT]: size,
    },
  };
}

async function precompress(directory) {
  // Sorted so the traversal order does not depend on the filesystem.
  const entries = (await readdir(directory, { withFileTypes: true })).sort((a, b) => (a.name < b.name ? -1 : 1));
  for (const entry of entries) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) {
      await precompress(path);
    } else if (/\.(?:css|html|js)$/.test(entry.name)) {
      const source = await readFile(path);
      if (source.length < 1024) continue;
      await Promise.all([
        writeFile(`${path}.gz`, await gzipAsync(source, GZIP_OPTIONS)),
        writeFile(`${path}.br`, await brotliAsync(source, brotliOptions(source.length))),
      ]);
    }
  }
}

// closeBundle also fires for `vite build --watch` and library builds; the guard keeps the
// pass out of `vite dev`, where no bundle is written at all.
const precompressPlugin = {
  name: 'precompress-assets',
  apply: 'build',
  configResolved(config) {
    precompressPlugin.outDir = isAbsolute(config.build.outDir)
      ? config.build.outDir
      : resolve(config.root, config.build.outDir);
  },
  closeBundle() {
    if (!precompressPlugin.outDir) return undefined;
    return precompress(precompressPlugin.outDir);
  },
};

export default defineConfig({
  plugins: [react(), precompressPlugin],
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          react: ['react', 'react-dom'],
          markdown: ['react-markdown', 'remark-gfm'],
        },
      },
    },
  },
});
