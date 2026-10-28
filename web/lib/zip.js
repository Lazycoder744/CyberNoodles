/* zip.js — minimal, dependency-free ZIP reader.
 *
 * Only what a BeatSaver map archive needs: read the central directory, then
 * inflate each entry with the platform's own DecompressionStream
 * ('deflate-raw').  Works identically in a Web Worker and in Node >= 18.
 *
 *   const files = await CN.zip.readZip(arrayBuffer);  // Map<lowercased path, Uint8Array>
 */
(function (global) {
  'use strict';
  const CN = (global.CN = global.CN || {});

  const SIG_EOCD = 0x06054b50;
  const SIG_EOCD64 = 0x06064b50;
  const SIG_CENTRAL = 0x02014b50;
  const SIG_LOCAL = 0x04034b50;

  async function inflateRaw(bytes) {
    const ds = new DecompressionStream('deflate-raw');
    const stream = new Blob([bytes]).stream().pipeThrough(ds);
    return new Uint8Array(await new Response(stream).arrayBuffer());
  }

  function findEOCD(u8, dv) {
    const n = u8.length;
    const lo = Math.max(0, n - 65557);
    for (let i = n - 22; i >= lo; i--) {
      if (dv.getUint32(i, true) === SIG_EOCD) {
        // Sanity: comment length must match the remaining bytes.
        const clen = dv.getUint16(i + 20, true);
        if (i + 22 + clen <= n) return i;
      }
    }
    return -1;
  }

  async function readZip(arrayBuffer) {
    const u8 = arrayBuffer instanceof Uint8Array ? arrayBuffer : new Uint8Array(arrayBuffer);
    const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
    const eocd = findEOCD(u8, dv);
    if (eocd < 0) throw new Error('zip: end-of-central-directory not found');

    let count = dv.getUint16(eocd + 10, true);
    let cdOff = dv.getUint32(eocd + 16, true);
    if (count === 0xffff || cdOff === 0xffffffff) {
      // ZIP64 locator sits immediately before the EOCD.
      const loc = eocd - 20;
      if (loc >= 0 && dv.getUint32(loc, true) === 0x07064b50) {
        const z64 = Number(dv.getBigUint64(loc + 8, true));
        if (dv.getUint32(z64, true) !== SIG_EOCD64) throw new Error('zip: bad zip64 EOCD');
        count = Number(dv.getBigUint64(z64 + 32, true));
        cdOff = Number(dv.getBigUint64(z64 + 48, true));
      }
    }

    const files = new Map();
    let p = cdOff;
    for (let e = 0; e < count; e++) {
      if (p + 46 > u8.length || dv.getUint32(p, true) !== SIG_CENTRAL) break;
      const method = dv.getUint16(p + 10, true);
      let compSize = dv.getUint32(p + 20, true);
      let uncompSize = dv.getUint32(p + 24, true);
      const nameLen = dv.getUint16(p + 28, true);
      const extraLen = dv.getUint16(p + 30, true);
      const commentLen = dv.getUint16(p + 32, true);
      const lho = dv.getUint32(p + 42, true);
      const name = CN.util.decodeUtf8(u8.subarray(p + 46, p + 46 + nameLen));

      if (compSize === 0xffffffff || lho === 0xffffffff) {
        // ZIP64 extended info lives in the extra field; parse just enough.
        let q = p + 46 + nameLen;
        const end = q + extraLen;
        while (q + 4 <= end) {
          const tag = dv.getUint16(q, true);
          const sz = dv.getUint16(q + 2, true);
          if (tag === 0x0001) {
            let r = q + 4;
            if (uncompSize === 0xffffffff) { uncompSize = Number(dv.getBigUint64(r, true)); r += 8; }
            if (compSize === 0xffffffff) { compSize = Number(dv.getBigUint64(r, true)); r += 8; }
            break;
          }
          q += 4 + sz;
        }
      }

      // Local header gives the true data offset (its extra field can differ).
      let dataOff = -1;
      if (p + 46 <= u8.length && dv.getUint32(lho, true) === SIG_LOCAL) {
        const lNameLen = dv.getUint16(lho + 26, true);
        const lExtraLen = dv.getUint16(lho + 28, true);
        dataOff = lho + 30 + lNameLen + lExtraLen;
      }

      const key = name.replace(/\\/g, '/').toLowerCase();
      if (name.endsWith('/') || dataOff < 0) {
        files.set(key, new Uint8Array(0));
      } else {
        const raw = u8.subarray(dataOff, dataOff + compSize);
        files.set(key, method === 0 ? raw.slice() : await inflateRaw(raw));
      }
      p += 46 + nameLen + extraLen + commentLen;
    }
    return files;
  }

  CN.zip = { readZip };
})(typeof globalThis !== 'undefined' ? globalThis : self);
