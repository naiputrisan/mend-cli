// mend — a small patch tool that reads the edit formats AI models produce
// (unified diffs, SEARCH/REPLACE blocks, V4A patches and JSON edit calls) and
// applies them to a tree of text files, all-or-nothing. The behaviour contract
// is SPEC.md in this repository; this file is a from-scratch implementation of
// it using only the Node standard library.

import { mkdirSync, readFileSync, rmSync, statSync, writeFileSync, writeSync } from 'node:fs';
import { dirname } from 'node:path';

// ---------------------------------------------------------------------------
// Control-flow signals: a ParseProblem aborts the whole run with exit code 2,
// an EditProblem fails just the edit being applied (the run continues so every
// failure can be reported, then exits 1 and writes nothing).
// ---------------------------------------------------------------------------

class ParseProblem {
  readonly lineNo: number;
  constructor(lineNo: number) {
    this.lineNo = lineNo;
  }
}

class EditProblem {
  readonly code: string;
  readonly path: string;
  constructor(code: string, path: string) {
    this.code = code;
    this.path = path;
  }
}

// ---------------------------------------------------------------------------
// Text helpers: the line model of section 3.
// ---------------------------------------------------------------------------

const isBlankLine = (s: string): boolean => s.trim() === '';
const stripTrailWs = (s: string): string => s.replace(/[ \t]+$/, '');
const stripBothWs = (s: string): string => s.replace(/^[ \t]+/, '').replace(/[ \t]+$/, '');
const leadWs = (s: string): string => /^[ \t]*/.exec(s)![0];

/** Split patch text into patch lines: split at \n, one trailing \r removed
 *  from each line, and no empty piece after a final \n. */
function toPatchLines(text: string): string[] {
  if (text === '') return [];
  const parts = text.split('\n');
  if (text.endsWith('\n')) parts.pop();
  return parts.map((p) => (p.endsWith('\r') ? p.slice(0, -1) : p));
}

/** A path is safe when it is relative, backslash-free and has no empty, `.`
 *  or `..` segment. */
function pathIsSafe(p: string): boolean {
  if (p === '' || p.startsWith('/') || p.includes('\\')) return false;
  return p.split('/').every((seg) => seg !== '' && seg !== '.' && seg !== '..');
}

interface FileModel {
  lines: string[];
  br: string; // '\n' or '\r\n'
  tail: boolean; // the file ends with a line break
}

function modelOf(text: string): FileModel {
  const at = text.indexOf('\n');
  const br = at > 0 && text[at - 1] === '\r' ? '\r\n' : '\n';
  if (text === '') return { lines: [], br, tail: false };
  const lines = text.split(br);
  const tail = text.endsWith(br);
  if (tail) lines.pop();
  return { lines, br, tail };
}

function render(m: FileModel): string {
  if (m.lines.length === 0) return '';
  return m.lines.join(m.br) + (m.tail ? m.br : '');
}

/** Lines for a freshly created file (LF breaks) from a list of line texts. */
function createdText(lines: string[], finalBreak: boolean): string {
  if (lines.length === 0) return '';
  return lines.join('\n') + (finalBreak ? '\n' : '');
}

// ---------------------------------------------------------------------------
// The virtual tree. Edits mutate `pending`; the disk is only touched once, at
// the very end of a fully successful run (and never under --check).
// ---------------------------------------------------------------------------

class VirtualTree {
  private cache = new Map<string, string | null>();
  pending = new Map<string, string | null>();
  moves: Array<[string, string]> = [];

  /** Content before the patch (null: did not exist). */
  before(p: string): string | null {
    if (!this.cache.has(p)) {
      let v: string | null = null;
      try {
        if (statSync(p).isFile()) v = readFileSync(p, 'utf8');
      } catch {
        v = null;
      }
      this.cache.set(p, v);
    }
    return this.cache.get(p)!;
  }

  /** Content as the edits so far have left it. */
  current(p: string): string | null {
    return this.pending.has(p) ? this.pending.get(p)! : this.before(p);
  }

  put(p: string, text: string | null): void {
    this.pending.set(p, text);
  }
}

type EditFn = (tree: VirtualTree, strict: boolean) => void;

function needSafe(p: string): void {
  if (!pathIsSafe(p)) throw new EditProblem('unsafe', p);
}

function needFile(tree: VirtualTree, p: string): string {
  const t = tree.current(p);
  if (t === null) throw new EditProblem('missing', p);
  return t;
}

// ---------------------------------------------------------------------------
// Section 5 matching. A replacement is described by the new-side entries:
// context entries point back at the old-side index they pair with, added
// entries carry their own text.
// ---------------------------------------------------------------------------

interface NewEntry {
  ctx: boolean;
  text: string;
  oldIdx: number; // valid when ctx is true
}

interface Found {
  tier: number;
  at: number[]; // every occurrence position at that tier
}

function lineEqAtTier(fileLine: string, patchLine: string, tier: number): boolean {
  if (tier === 0) return fileLine === patchLine;
  if (tier === 1) return stripTrailWs(fileLine) === stripTrailWs(patchLine);
  return stripBothWs(fileLine) === stripBothWs(patchLine);
}

/** First tier (0, or 0..2 unless strict) at which `old` occurs at or after
 *  `from`; null when it occurs nowhere. */
function locate(lines: string[], old: string[], strict: boolean, from = 0): Found | null {
  const top = strict ? 0 : 2;
  for (let tier = 0; tier <= top; tier += 1) {
    const at: number[] = [];
    for (let i = Math.max(0, from); i + old.length <= lines.length; i += 1) {
      let ok = true;
      for (let j = 0; j < old.length; j += 1) {
        if (!lineEqAtTier(lines[i + j]!, old[j]!, tier)) {
          ok = false;
          break;
        }
      }
      if (ok) at.push(i);
    }
    if (at.length > 0) return { tier, at };
  }
  return null;
}

/** The lines that replace the matched segment, honouring context lines and
 *  the tier-2 re-indentation rule. */
function buildReplacement(lines: string[], pos: number, old: string[], entries: NewEntry[], tier: number): string[] {
  const out: string[] = [];
  let f = '';
  let p = '';
  if (tier === 2 && old.length > 0) {
    f = leadWs(lines[pos]!);
    p = leadWs(old[0]!);
  }
  for (const e of entries) {
    if (e.ctx) {
      out.push(lines[pos + e.oldIdx]!);
    } else if (tier === 2 && e.text !== '' && e.text.trim() !== '' && e.text.startsWith(p)) {
      out.push(f + e.text.slice(p.length));
    } else {
      out.push(e.text);
    }
  }
  return out;
}

const addedEntries = (texts: string[]): NewEntry[] => texts.map((t) => ({ ctx: false, text: t, oldIdx: -1 }));

// ---------------------------------------------------------------------------
// Section 9: JSON edit objects (array form and JSON Lines form).
// ---------------------------------------------------------------------------

function jsonToEdit(v: unknown): EditFn | null {
  if (v === null || typeof v !== 'object' || Array.isArray(v)) return null;
  const o = v as Record<string, unknown>;
  const path = o.path;
  if (typeof path !== 'string') return null;

  if (o.command === 'create') {
    if (typeof o.file_text !== 'string') return null;
    const body = o.file_text;
    return (tree) => {
      needSafe(path);
      if (tree.current(path) !== null) throw new EditProblem('exists', path);
      tree.put(path, body);
    };
  }

  if (o.command === 'str_replace') {
    if (typeof o.old_str !== 'string' || o.old_str === '') return null;
    if (o.new_str !== undefined && typeof o.new_str !== 'string') return null;
    const oldStr = o.old_str;
    const newStr = typeof o.new_str === 'string' ? o.new_str : '';
    return (tree) => {
      needSafe(path);
      const before = needFile(tree, path);
      const crlf = modelOf(before).br === '\r\n';
      const norm = (s: string): string => (crlf ? s.split('\r\n').join('\n') : s);
      const hay = norm(before);
      const needle = norm(oldStr);
      let hits = 0;
      for (let k = hay.indexOf(needle); k >= 0; k = hay.indexOf(needle, k + 1)) hits += 1;
      if (hits === 0) throw new EditProblem('no-match', path);
      if (hits > 1) throw new EditProblem('ambiguous', path);
      const at = hay.indexOf(needle);
      const joined = hay.slice(0, at) + norm(newStr) + hay.slice(at + needle.length);
      tree.put(path, crlf ? joined.split('\n').join('\r\n') : joined);
    };
  }

  if (o.command === 'insert') {
    if (typeof o.insert_line !== 'number' || !Number.isInteger(o.insert_line)) return null;
    if (typeof o.new_str !== 'string') return null;
    const lineNo = o.insert_line;
    const add = toPatchLines(o.new_str);
    return (tree) => {
      needSafe(path);
      const before = needFile(tree, path);
      const m = modelOf(before);
      if (lineNo < 0 || lineNo > m.lines.length) throw new EditProblem('no-match', path);
      const lines = [...m.lines];
      lines.splice(lineNo, 0, ...add);
      tree.put(path, render({ lines, br: m.br, tail: m.tail }));
    };
  }

  return null;
}

function readJsonEdits(text: string, lines: string[]): EditFn[] {
  const first = lines.findIndex((l) => !isBlankLine(l));
  if (first < 0) throw new ParseProblem(1);
  if (lines[first]!.trimStart().startsWith('[')) {
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch {
      throw new ParseProblem(first + 1);
    }
    if (!Array.isArray(parsed)) throw new ParseProblem(first + 1);
    const edits = parsed.map((item) => {
      const e = jsonToEdit(item);
      if (e === null) throw new ParseProblem(first + 1);
      return e;
    });
    return edits;
  }
  const edits: EditFn[] = [];
  lines.forEach((line, idx) => {
    if (isBlankLine(line)) return;
    let e: EditFn | null = null;
    try {
      e = jsonToEdit(JSON.parse(line));
    } catch {
      e = null;
    }
    if (e === null) throw new ParseProblem(idx + 1);
    edits.push(e);
  });
  return edits;
}

// ---------------------------------------------------------------------------
// Section 7: SEARCH/REPLACE blocks.
// ---------------------------------------------------------------------------

const MARK_SEARCH = '<<<<<<< SEARCH';
const MARK_MID = '=======';
const MARK_END = '>>>>>>> REPLACE';

function readSearchReplace(lines: string[]): EditFn[] {
  const edits: EditFn[] = [];
  let prevPath: string | null = null;
  let prevEnd = -1; // index of the previous block's REPLACE line
  let i = 0;
  while (i < lines.length) {
    if (stripTrailWs(lines[i]!) !== MARK_SEARCH) {
      i += 1;
      continue;
    }
    const searchAt = i;
    // Find the ======= separator and then the >>>>>>> REPLACE line.
    let mid = -1;
    for (let j = searchAt + 1; j < lines.length; j += 1) {
      if (stripTrailWs(lines[j]!) === MARK_MID) {
        mid = j;
        break;
      }
    }
    let end = -1;
    if (mid >= 0) {
      for (let j = mid + 1; j < lines.length; j += 1) {
        if (stripTrailWs(lines[j]!) === MARK_END) {
          end = j;
          break;
        }
      }
    }
    if (mid < 0 || end < 0) throw new ParseProblem(searchAt + 1);
    // The path is the nearest usable line above SEARCH, after the last block.
    let path: string | null = null;
    for (let j = searchAt - 1; j > prevEnd; j -= 1) {
      const raw = lines[j]!;
      if (isBlankLine(raw)) continue;
      if (raw.trim().startsWith('```')) continue;
      path = raw.trim();
      break;
    }
    if (path === null) path = prevPath;
    if (path === null) throw new ParseProblem(searchAt + 1);
    const oldLines = lines.slice(searchAt + 1, mid);
    const newLines = lines.slice(mid + 1, end);
    const blockPath = path;
    const blockOld = oldLines;
    const blockNew = newLines;
    edits.push((tree, strict) => {
      needSafe(blockPath);
      if (blockOld.length === 0) {
        if (tree.current(blockPath) !== null) throw new EditProblem('exists', blockPath);
        tree.put(blockPath, createdText(blockNew, true));
        return;
      }
      const m = modelOf(needFile(tree, blockPath));
      const found = locate(m.lines, blockOld, strict);
      if (found === null) throw new EditProblem('no-match', blockPath);
      if (found.at.length > 1) throw new EditProblem('ambiguous', blockPath);
      const pos = found.at[0]!;
      const lines2 = [...m.lines];
      lines2.splice(pos, blockOld.length, ...buildReplacement(m.lines, pos, blockOld, addedEntries(blockNew), found.tier));
      tree.put(blockPath, render({ lines: lines2, br: m.br, tail: m.tail }));
    });
    prevPath = path;
    prevEnd = end;
    i = end + 1;
  }
  return edits;
}

// ---------------------------------------------------------------------------
// Section 6: unified diffs.
// ---------------------------------------------------------------------------

interface BodyLine {
  kind: 'ctx' | 'del' | 'add' | 'slash';
  text: string;
  rawEmpty: boolean;
}

interface Hunk {
  a: number; // the -A number of the header
  body: BodyLine[];
}

interface Section {
  diffOld: string | null;
  diffNew: string | null;
  headOld: string | null | undefined; // undefined: no --- line seen
  headNew: string | null | undefined;
  renameFrom: string | null;
  renameTo: string | null;
  newFile: boolean;
  delFile: boolean;
  hunks: Hunk[];
}

const HUNK_RE = /^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@(?: .*)?$/;
const GIT_RE = /^diff --git a\/(.+?) b\/(.+)$/;

function headerPath(raw: string): string | null {
  const upToTab = raw.split('\t')[0]!;
  const t = upToTab.trim();
  if (t === '/dev/null') return null;
  return t.startsWith('a/') || t.startsWith('b/') ? t.slice(2) : t;
}

function readUnified(lines: string[]): EditFn[] {
  const sections: Section[] = [];
  let sec: Section | null = null;
  let hunk: Hunk | null = null;

  const closeHunk = (): void => {
    if (hunk !== null && sec !== null) {
      while (hunk.body.length > 0) {
        const last = hunk.body[hunk.body.length - 1]!;
        if (last.kind === 'ctx' && last.rawEmpty) hunk.body.pop();
        else break;
      }
      sec.hunks.push(hunk);
    }
    hunk = null;
  };
  const newSection = (): Section => {
    closeHunk();
    const s: Section = {
      diffOld: null,
      diffNew: null,
      headOld: undefined,
      headNew: undefined,
      renameFrom: null,
      renameTo: null,
      newFile: false,
      delFile: false,
      hunks: [],
    };
    sections.push(s);
    sec = s;
    return s;
  };
  const inHeader = (): boolean => sec !== null && sec.hunks.length === 0 && hunk === null;

  let i = 0;
  while (i < lines.length) {
    const line = lines[i]!;
    const lineNo = i + 1;

    // A diff --git line ends any body and starts a fresh section.
    if (line.startsWith('diff --git')) {
      const m = GIT_RE.exec(line);
      if (m === null) throw new ParseProblem(lineNo);
      const s = newSection();
      s.diffOld = m[1]!;
      s.diffNew = m[2]!;
      i += 1;
      continue;
    }

    // A --- / +++ pair: a section header, or the start of a new section.
    if (line.startsWith('--- ') && (lines[i + 1] ?? '').startsWith('+++ ')) {
      if (inHeader() && sec!.headOld === undefined) {
        sec!.headOld = headerPath(line.slice(4));
        sec!.headNew = headerPath(lines[i + 1]!.slice(4));
      } else {
        const s = newSection();
        s.headOld = headerPath(line.slice(4));
        s.headNew = headerPath(lines[i + 1]!.slice(4));
      }
      i += 2;
      continue;
    }
    if (line.startsWith('--- ') && inHeader() && sec!.headOld === undefined) {
      throw new ParseProblem(lineNo + 1);
    }

    // Hunk headers.
    if (line.startsWith('@@')) {
      if (sec === null) throw new ParseProblem(lineNo);
      const m = HUNK_RE.exec(line);
      if (m === null) throw new ParseProblem(lineNo);
      closeHunk();
      hunk = { a: Number(m[1]), body: [] };
      i += 1;
      continue;
    }

    // Hunk body lines.
    if (hunk !== null) {
      const c = line[0];
      if (line === '' || c === ' ' || c === '-' || c === '+' || c === '\\') {
        const kind = line === '' || c === ' ' ? 'ctx' : c === '-' ? 'del' : c === '+' ? 'add' : 'slash';
        hunk.body.push({ kind, text: line === '' ? '' : line.slice(1), rawEmpty: line === '' });
        i += 1;
        continue;
      }
      closeHunk();
      // fall through: the line is handled as an outside-body line below
    }

    // Section header lines (before the first hunk).
    if (inHeader()) {
      if (line.startsWith('rename from ')) sec!.renameFrom = line.slice('rename from '.length);
      else if (line.startsWith('rename to ')) sec!.renameTo = line.slice('rename to '.length);
      else if (line.startsWith('new file mode')) sec!.newFile = true;
      else if (line.startsWith('deleted file mode')) sec!.delFile = true;
      i += 1;
      continue;
    }

    // Anything else outside a body is ignored.
    i += 1;
  }
  closeHunk();

  return sections.map((s) => unifiedEdit(s));
}

/** Whether a `\` line follows the last context-or-add entry among `body`
 *  entries up to index `upto` (or at all when upto is omitted). */
function slashAfterLastWrite(body: BodyLine[]): boolean {
  let lastWrite = -1;
  body.forEach((b, idx) => {
    if (b.kind === 'ctx' || b.kind === 'add') lastWrite = idx;
  });
  return lastWrite >= 0 && body[lastWrite + 1]?.kind === 'slash';
}

function unifiedEdit(s: Section): EditFn {
  const oldPath = s.headOld !== undefined ? s.headOld : s.renameFrom !== null ? s.renameFrom : s.diffOld;
  const newPath = s.headNew !== undefined ? s.headNew : s.renameTo !== null ? s.renameTo : s.diffNew;
  const isCreate = oldPath === null || s.newFile;
  const isDelete = newPath === null || s.delFile;

  return (tree, strict) => {
    // Safety of every named path comes before anything else.
    if (!isCreate && oldPath !== null) needSafe(oldPath);
    if (!isDelete && newPath !== null) needSafe(newPath);
    if (isCreate) needSafe(newPath ?? '');
    if (isDelete) needSafe(oldPath ?? '');

    if (isCreate) {
      const target = newPath!;
      if (tree.current(target) !== null) throw new EditProblem('exists', target);
      const content: string[] = [];
      let noBreak = false;
      for (const h of s.hunks) {
        for (const b of h.body) if (b.kind === 'add') content.push(b.text);
      }
      // The final-break rule looks at the last + line of the whole section.
      outer: for (let hi = s.hunks.length - 1; hi >= 0; hi -= 1) {
        const body = s.hunks[hi]!.body;
        for (let bi = body.length - 1; bi >= 0; bi -= 1) {
          if (body[bi]!.kind === 'add') {
            noBreak = body[bi + 1]?.kind === 'slash';
            break outer;
          }
        }
      }
      tree.put(target, createdText(content, !noBreak));
      return;
    }

    if (isDelete) {
      const target = oldPath!;
      needFile(tree, target);
      tree.put(target, null);
      return;
    }

    const from = oldPath!;
    const to = newPath!;
    const before = needFile(tree, from);
    if (from !== to && tree.current(to) !== null) throw new EditProblem('exists', to);
    const m = modelOf(before);
    const lines = [...m.lines];
    let tail = m.tail;
    let d = 0;
    for (const h of s.hunks) {
      const oldSide: string[] = [];
      const entries: NewEntry[] = [];
      for (const b of h.body) {
        if (b.kind === 'ctx') {
          entries.push({ ctx: true, text: b.text, oldIdx: oldSide.length });
          oldSide.push(b.text);
        } else if (b.kind === 'del') {
          oldSide.push(b.text);
        } else if (b.kind === 'add') {
          entries.push({ ctx: false, text: b.text, oldIdx: -1 });
        }
      }
      if (oldSide.length === 0) {
        const at = Math.min(h.a + d, lines.length);
        const atEnd = at >= lines.length;
        lines.splice(at, 0, ...entries.map((e) => e.text));
        if (atEnd) tail = !slashAfterLastWrite(h.body);
        d += entries.length;
        continue;
      }
      const found = locate(lines, oldSide, strict);
      if (found === null) throw new EditProblem('no-match', from);
      const target = h.a - 1 + d;
      let pos = found.at[0]!;
      for (const cand of found.at) {
        if (Math.abs(cand - target) < Math.abs(pos - target)) pos = cand;
      }
      const touchesEnd = pos + oldSide.length === lines.length;
      const repl = buildReplacement(lines, pos, oldSide, entries, found.tier);
      lines.splice(pos, oldSide.length, ...repl);
      if (touchesEnd) tail = !slashAfterLastWrite(h.body);
      d += entries.length - oldSide.length;
    }
    tree.put(to, render({ lines, br: m.br, tail }));
    if (from !== to) {
      tree.put(from, null);
      tree.moves.push([from, to]);
    }
  };
}

// ---------------------------------------------------------------------------
// Section 8: V4A patches.
// ---------------------------------------------------------------------------

interface VChunk {
  anchor: string | null;
  eof: boolean;
  oldSide: string[];
  entries: NewEntry[];
}

interface VSection {
  kind: 'add' | 'del' | 'upd';
  path: string;
  headerAt: number;
  content: string[]; // add: file lines
  moveTo: string | null;
  chunks: VChunk[];
}

function readV4a(lines: string[]): EditFn[] {
  const first = lines.findIndex((l) => !isBlankLine(l));
  if (first < 0) throw new ParseProblem(1);
  if (lines[first]!.trim() !== '*** Begin Patch') throw new ParseProblem(first + 1);
  let last = lines.length - 1;
  while (last >= 0 && isBlankLine(lines[last]!)) last -= 1;
  if (last < 0 || lines[last]!.trim() !== '*** End Patch') throw new ParseProblem(lines.length + 1);

  const sections: VSection[] = [];
  let i = first + 1;
  const isHeader = (l: string): boolean =>
    l.startsWith('*** Add File: ') || l.startsWith('*** Delete File: ') || l.startsWith('*** Update File: ');

  while (i < last) {
    const line = lines[i]!;
    if (isBlankLine(line)) {
      i += 1;
      continue;
    }
    let kind: VSection['kind'] | null = null;
    let path = '';
    if (line.startsWith('*** Add File: ')) {
      kind = 'add';
      path = line.slice('*** Add File: '.length).trim();
    } else if (line.startsWith('*** Delete File: ')) {
      kind = 'del';
      path = line.slice('*** Delete File: '.length).trim();
    } else if (line.startsWith('*** Update File: ')) {
      kind = 'upd';
      path = line.slice('*** Update File: '.length).trim();
    }
    if (kind === null) throw new ParseProblem(i + 1);
    const headerAt = i + 1;
    // Collect the section's raw lines (up to the next header or End Patch),
    // then drop trailing empty lines, which are not part of the section.
    const raw: Array<{ text: string; at: number }> = [];
    let j = i + 1;
    while (j < last && !isHeader(lines[j]!)) {
      raw.push({ text: lines[j]!, at: j + 1 });
      j += 1;
    }
    // "Blank" here is the section 4 sense: empty or only whitespace, so a
    // trailing context line that holds just its leading space is dropped too.
    while (raw.length > 0 && raw[raw.length - 1]!.text.trim() === '') raw.pop();

    if (kind === 'add') {
      const content: string[] = [];
      for (const r of raw) {
        if (r.text.startsWith('+')) content.push(r.text.slice(1));
        else if (r.text === '') content.push('');
        else throw new ParseProblem(r.at);
      }
      sections.push({ kind, path, headerAt, content, moveTo: null, chunks: [] });
    } else if (kind === 'del') {
      for (const r of raw) {
        if (r.text.trim() !== '') throw new ParseProblem(r.at);
      }
      sections.push({ kind, path, headerAt, content: [], moveTo: null, chunks: [] });
    } else {
      let moveTo: string | null = null;
      let start = 0;
      if (raw.length > 0 && raw[0]!.text.startsWith('*** Move to: ')) {
        moveTo = raw[0]!.text.slice('*** Move to: '.length).trim();
        start = 1;
      }
      const chunks: VChunk[] = [];
      let cur: VChunk | null = null;
      for (let k = start; k < raw.length; k += 1) {
        const r = raw[k]!;
        const t = r.text;
        if (t.startsWith('@@')) {
          cur = { anchor: t.slice(2).trim() === '' ? null : t.slice(2).trim(), eof: false, oldSide: [], entries: [] };
          chunks.push(cur);
        } else if (t === '*** End of File') {
          if (cur === null) {
            cur = { anchor: null, eof: true, oldSide: [], entries: [] };
            chunks.push(cur);
          }
          cur.eof = true;
          cur = null;
        } else if (t === '' || t.startsWith(' ') || t.startsWith('-') || t.startsWith('+')) {
          if (cur === null) {
            cur = { anchor: null, eof: false, oldSide: [], entries: [] };
            chunks.push(cur);
          }
          if (t === '' || t.startsWith(' ')) {
            cur.entries.push({ ctx: true, text: t === '' ? '' : t.slice(1), oldIdx: cur.oldSide.length });
            cur.oldSide.push(t === '' ? '' : t.slice(1));
          } else if (t.startsWith('-')) {
            cur.oldSide.push(t.slice(1));
          } else {
            cur.entries.push({ ctx: false, text: t.slice(1), oldIdx: -1 });
          }
        } else {
          throw new ParseProblem(r.at);
        }
      }
      if (chunks.length === 0 && moveTo === null) throw new ParseProblem(headerAt);
      sections.push({ kind, path, headerAt, content: [], moveTo, chunks });
    }
    i = j;
  }

  return sections.map((s) => v4aEdit(s));
}

function v4aEdit(s: VSection): EditFn {
  return (tree, strict) => {
    needSafe(s.path);
    if (s.moveTo !== null) needSafe(s.moveTo);
    if (s.kind === 'add') {
      if (tree.current(s.path) !== null) throw new EditProblem('exists', s.path);
      tree.put(s.path, createdText(s.content, true));
      return;
    }
    if (s.kind === 'del') {
      needFile(tree, s.path);
      tree.put(s.path, null);
      return;
    }
    const before = needFile(tree, s.path);
    if (s.moveTo !== null && tree.current(s.moveTo) !== null) throw new EditProblem('exists', s.moveTo);
    const m = modelOf(before);
    const lines = [...m.lines];
    let cursor = 0;
    for (const chunk of s.chunks) {
      if (chunk.anchor !== null) {
        const found = locate(lines, [chunk.anchor], strict, cursor);
        if (found === null) throw new EditProblem('no-match', s.path);
        cursor = found.at[0]! + 1;
      }
      if (chunk.oldSide.length === 0) {
        const at = chunk.eof ? lines.length : Math.min(cursor, lines.length);
        const repl = chunk.entries.map((e) => e.text);
        lines.splice(at, 0, ...repl);
        cursor = at + repl.length;
        continue;
      }
      const found = locate(lines, chunk.oldSide, strict, cursor);
      if (found === null) throw new EditProblem('no-match', s.path);
      let pos = -1;
      for (const cand of found.at) {
        if (chunk.eof && cand + chunk.oldSide.length !== lines.length) continue;
        pos = cand;
        break;
      }
      if (pos < 0) throw new EditProblem('no-match', s.path);
      const repl = buildReplacement(lines, pos, chunk.oldSide, chunk.entries, found.tier);
      lines.splice(pos, chunk.oldSide.length, ...repl);
      cursor = pos + repl.length;
    }
    const target = s.moveTo ?? s.path;
    tree.put(target, render({ lines, br: m.br, tail: m.tail }));
    if (s.moveTo !== null) {
      tree.put(s.path, null);
      tree.moves.push([s.path, s.moveTo]);
    }
  };
}

// ---------------------------------------------------------------------------
// Section 4: format detection.
// ---------------------------------------------------------------------------

function detectFormat(lines: string[]): string | null {
  const first = lines.findIndex((l) => !isBlankLine(l));
  if (first < 0) return null;
  if (lines[first]!.trim() === '*** Begin Patch') return 'v4a';
  const lead = lines[first]!.trimStart()[0];
  if (lead === '[' || lead === '{') return 'json';
  if (lines.some((l) => stripTrailWs(l) === MARK_SEARCH)) return 'search-replace';
  const hasDiff = lines.some((l) => l.startsWith('diff --git '));
  const hasPair = lines.some((l, idx) => l.startsWith('--- ') && (lines[idx + 1] ?? '').startsWith('+++ '));
  if (hasDiff || hasPair) return 'unified';
  return null;
}

// ---------------------------------------------------------------------------
// Sections 1, 2 and 10: the command line, the run, the write-back.
// ---------------------------------------------------------------------------

const FORMAT_NAMES = ['auto', 'unified', 'search-replace', 'v4a', 'json'];

function runCli(args: string[]): { code: number; out: string[] } {
  let format: string | null = null;
  let checkOnly = false;
  let strict = false;
  const positional: string[] = [];
  let afterDashDash = false;

  for (const a of args) {
    if (!afterDashDash && a === '--') {
      afterDashDash = true;
    } else if (!afterDashDash && a.startsWith('-') && a !== '-') {
      if (a.startsWith('--format=') && format === null && FORMAT_NAMES.includes(a.slice('--format='.length))) {
        format = a.slice('--format='.length);
      } else if (a === '--check' && !checkOnly) {
        checkOnly = true;
      } else if (a === '--strict' && !strict) {
        strict = true;
      } else {
        return { code: 2, out: ['error: usage'] };
      }
    } else {
      positional.push(a);
    }
  }
  if (positional.length > 1) return { code: 2, out: ['error: usage'] };

  const patchArg = positional.length === 0 ? '-' : positional[0]!;
  let patchText: string;
  if (patchArg === '-') {
    try {
      patchText = readFileSync(0, 'utf8');
    } catch {
      patchText = '';
    }
  } else {
    try {
      patchText = readFileSync(patchArg, 'utf8');
    } catch {
      return { code: 2, out: ['error: no-patch'] };
    }
  }

  const lines = toPatchLines(patchText);
  const kind = format === null || format === 'auto' ? detectFormat(lines) : format;
  if (kind === null) return { code: 2, out: ['error: format'] };

  let edits: EditFn[];
  try {
    if (kind === 'json') edits = readJsonEdits(patchText, lines);
    else if (kind === 'search-replace') edits = readSearchReplace(lines);
    else if (kind === 'unified') edits = readUnified(lines);
    else edits = readV4a(lines);
    if (edits.length === 0) throw new ParseProblem(1);
  } catch (e) {
    if (e instanceof ParseProblem) return { code: 2, out: [`error: parse ${e.lineNo}`] };
    throw e;
  }

  const tree = new VirtualTree();
  const failures: EditProblem[] = [];
  for (const edit of edits) {
    const savedPending = new Map(tree.pending);
    const savedMoves = [...tree.moves];
    try {
      edit(tree, strict);
    } catch (e) {
      if (!(e instanceof EditProblem)) throw e;
      tree.pending = savedPending;
      tree.moves = savedMoves;
      failures.push(e);
    }
  }
  if (failures.length > 0) {
    return {
      code: 1,
      out: [...failures.map((f) => `error: ${f.code} ${f.path}`), `failed: ${failures.length} of ${edits.length} edits`],
    };
  }

  // Work out the report lines, pairing a recorded move's D and A into one R
  // line when the before/after states fit section 2's move description.
  interface Row {
    key: string;
    text: string;
  }
  const rows: Row[] = [];
  const consumed = new Set<string>();
  for (const [oldP, newP] of tree.moves) {
    if (consumed.has(oldP) || consumed.has(newP)) continue;
    const wasOld = tree.before(oldP);
    const nowOld = tree.current(oldP);
    const wasNew = tree.before(newP);
    const nowNew = tree.current(newP);
    if (wasOld !== null && nowOld === null && wasNew === null && nowNew !== null) {
      rows.push({ key: newP, text: `R ${oldP} -> ${newP}` });
      consumed.add(oldP);
      consumed.add(newP);
    }
  }
  for (const p of tree.pending.keys()) {
    if (consumed.has(p)) continue;
    const was = tree.before(p);
    const now = tree.pending.get(p)!;
    if (was === now) continue;
    rows.push({ key: p, text: `${was === null ? 'A' : now === null ? 'D' : 'M'} ${p}` });
  }
  rows.sort((x, y) => Buffer.compare(Buffer.from(x.key, 'utf8'), Buffer.from(y.key, 'utf8')));

  if (!checkOnly) {
    for (const [p, now] of tree.pending) {
      if (tree.before(p) === now) continue;
      if (now === null) {
        rmSync(p, { force: true });
      } else {
        mkdirSync(dirname(p), { recursive: true });
        writeFileSync(p, now);
      }
    }
  }
  return { code: 0, out: [...rows.map((r) => r.text), `ok: ${rows.length} ${rows.length === 1 ? 'file' : 'files'}`] };
}

// ---------------------------------------------------------------------------
// Entry point: a trial hands over to the kit runner; anything else is the CLI.
// ---------------------------------------------------------------------------

const envUrl = process.env.MAKEFUN_ENV_URL;
if (envUrl !== undefined && envUrl !== '') {
  const { runTrial } = await import('./runner.ts');
  await runTrial(envUrl);
} else {
  const result = runCli(process.argv.slice(2));
  writeSync(1, result.out.map((l) => `${l}\n`).join(''));
  process.exitCode = result.code;
}
