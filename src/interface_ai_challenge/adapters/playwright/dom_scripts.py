HELPERS = r"""
const norm = (s) => (s || '').replace(/\s+/g, ' ').trim();
const labelText = (s) => norm(s).replace(/:\s*$/, '');
const textOf = (el) => norm(el.innerText !== undefined ? el.innerText : el.textContent);
const visible = (el) => {
  if (!el || !el.isConnected) return false;
  const rect = el.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) return false;
  return el.checkVisibility ? el.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true }) : true;
};
const inputType = (el) => (el.getAttribute('type') || 'text').toLowerCase();
const roleOf = (el) => {
  const tag = el.tagName.toLowerCase();
  if (el.getAttribute('role') === 'button') return 'button';
  if (tag === 'a' && el.hasAttribute('href')) return 'link';
  if (tag === 'button') return 'button';
  if (tag === 'input') {
    if (['submit', 'button', 'reset'].includes(inputType(el))) return 'button';
    if (['text', 'search', 'number', 'tel', 'email', 'password'].includes(inputType(el))) return 'textbox';
    return 'other';
  }
  if (tag === 'textarea') return 'textbox';
  if (tag === 'canvas') return 'canvas';
  if (tag === 'td' || tag === 'th') return 'cell';
  return 'text';
};
const nameOf = (el) => {
  const aria = el.getAttribute('aria-label');
  if (aria) return norm(aria);
  const tag = el.tagName.toLowerCase();
  if (tag === 'input') return ['submit', 'button', 'reset'].includes(inputType(el)) ? norm(el.value) : '';
  if (tag === 'textarea' || tag === 'canvas') return '';
  return textOf(el);
};
const headerRow = (table) => {
  if (table.tHead && table.tHead.rows.length) return table.tHead.rows[0];
  for (const row of table.rows) {
    const cells = Array.from(row.cells);
    if (cells.length && cells.every((cell) => cell.tagName === 'TH')) return row;
  }
  return null;
};
const ownerTable = (row) => (row.parentElement ? row.parentElement.closest('table') : null);
"""

ELEMENT_FROM_POINT = r"""([x, y]) => {
  const el = document.elementFromPoint(x, y);
  if (!el) return null;
  return el.closest('iframe,a[href],button,input,textarea,select,[role=button],canvas,td,th') || el;
}"""

ACTIVE_ELEMENT = r"""() => {
  const el = document.activeElement;
  return el && el !== document.body && el !== document.documentElement ? el : null;
}"""

IFRAME_BORDER = r"""(el) => [el.clientLeft, el.clientTop]"""

IS_IFRAME = r"""(el) => el.tagName === 'IFRAME'"""

SAME_ELEMENT = r"""([a, b]) => a === b"""

DESCRIBE = (
    r"""(el) => {"""
    + HELPERS
    + r"""
  const cell = el.closest('td,th');
  let rowLabel = null;
  let columnHeader = null;
  const rowCells = {};
  if (cell) {
    const row = cell.parentElement;
    const cells = Array.from(row.cells);
    const index = cells.indexOf(cell);
    const table = ownerTable(row);
    const header = table ? headerRow(table) : null;
    if (header && header !== row) {
      const headers = Array.from(header.cells).map((h) => textOf(h));
      columnHeader = headers[index] ?? null;
      cells.forEach((c, i) => { if (headers[i]) rowCells[headers[i]] = textOf(c); });
    } else if (index > 0) {
      rowLabel = labelText(textOf(cells[0]));
    }
  }
  const label = el.labels && el.labels.length ? labelText(textOf(el.labels[0])) : null;
  return {
    role: roleOf(el),
    name: nameOf(el),
    label,
    row_label: rowLabel,
    column_header: columnHeader,
    row_cells: rowCells,
  };
}"""
)

RESOLVE = (
    r"""(spec) => {"""
    + HELPERS
    + r"""
  const found = [];
  const push = (el) => { if (visible(el) && !found.includes(el)) found.push(el); };
  if (spec.strategy === 'role') {
    for (const el of document.querySelectorAll('a[href],button,input,textarea,[role=button]')) {
      if (roleOf(el) === spec.role && nameOf(el) === spec.name) push(el);
    }
  } else if (spec.strategy === 'labeled_field' || spec.strategy === 'labeled_value') {
    for (const row of document.querySelectorAll('tr')) {
      const cells = Array.from(row.cells);
      if (cells.length < 2 || cells.every((cell) => cell.tagName === 'TH')) continue;
      if (labelText(textOf(cells[0])) !== spec.label) continue;
      if (spec.strategy === 'labeled_value') { push(cells[1]); continue; }
      for (const cell of cells.slice(1)) {
        for (const control of cell.querySelectorAll('input,textarea')) {
          if (roleOf(control) === 'textbox' && control.closest('tr') === row) push(control);
        }
      }
    }
    if (spec.strategy === 'labeled_field') {
      for (const label of document.querySelectorAll('label')) {
        if (labelText(textOf(label)) === spec.label && label.control) push(label.control);
      }
    }
  } else if (spec.strategy === 'table_column') {
    for (const table of document.querySelectorAll('table')) {
      const header = headerRow(table);
      if (!header) continue;
      const column = Array.from(header.cells).findIndex((h) => textOf(h) === spec.column);
      if (column < 0) continue;
      for (const row of table.rows) {
        if (row !== header && row.cells[column]) push(row.cells[column]);
      }
    }
  } else if (spec.strategy === 'table_cell') {
    for (const table of document.querySelectorAll('table')) {
      const header = headerRow(table);
      if (!header) continue;
      const headers = Array.from(header.cells).map((h) => textOf(h));
      const column = headers.indexOf(spec.column);
      const keyColumn = headers.indexOf(spec.row_key_column);
      if (column < 0 || keyColumn < 0) continue;
      for (const row of table.rows) {
        if (row === header) continue;
        const cells = Array.from(row.cells);
        if (!cells[column] || !cells[keyColumn] || textOf(cells[keyColumn]) !== spec.row_key) continue;
        const cell = cells[column];
        if (!spec.control) { push(cell); continue; }
        for (const el of cell.querySelectorAll('a[href],button,input,[role=button]')) {
          if (el.closest('td,th') === cell && roleOf(el) === spec.control.role && nameOf(el) === spec.control.name) push(el);
        }
      }
    }
  }
  return found;
}"""
)

TEXT_VISIBLE = (
    r"""(text) => {"""
    + HELPERS
    + r"""
  return document.body ? norm(document.body.innerText).includes(norm(text)) : false;
}"""
)

DIALOG_TITLES = (
    r"""() => {"""
    + HELPERS
    + r"""
  return Array.from(document.querySelectorAll('[role=dialog],[role=alertdialog],dialog[open]'))
    .filter((el) => visible(el))
    .map((el) => norm(el.getAttribute('aria-label') || '') || 'untitled dialog');
}"""
)
