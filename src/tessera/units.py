"""Structure-completed units from page HTML (paper §3.3).

`page_units(title, html)` walks the DOM in document order and emits two kinds
of units, each a dict with `completed` (context + body, what the scorer reads),
`group`/`key` (the context a packer prints once), `body`, `labels`, `values`,
`kind`, and `pos` (page order):

- block units: paragraphs, list items, definition terms/values, and figure
  captions under the path "page › section › subsection";
- row units: one per table or infobox row, read from a faithful grid that
  expands rowspan/colspan, turns full-width leading rows into the table title,
  joins multi-row headers per column, turns full-width body rows into group
  labels, reads header-less rows that start with a header cell as key: value,
  and lets nested tables inherit their parent's title and column labels.

`ZimPages` resolves titles, redirects, and Kiwix soft redirects in a ZIM file.
"""

from __future__ import annotations

import html as html_lib
import re
from urllib.parse import unquote

BLOCK_TAGS = {
    "p": "paragraph",
    "li": "list_item",
    "dt": "definition_term",
    "dd": "definition_value",
    "figcaption": "figure_caption",
}
DROP_TAGS = {"script", "style", "noscript", "template", "nav"}
SPACE = re.compile(r"\s+")
SUPERSCRIPT = str.maketrans("0123456789+-=()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾")
REFRESH_URL = re.compile(r"\burl\s*=\s*(.+)$", re.IGNORECASE)


def _tag(element) -> str:
    return str(element.tag).rsplit("}", 1)[-1].casefold()


def _norm(text: str) -> str:
    return SPACE.sub(" ", text).strip()


def clean_superscripts(root) -> None:
    """Drop citation superscripts; keep exponents and stacked music meters."""
    for element in list(root.iter()):
        if "music-symbol" not in (element.get("class") or "").split():
            continue
        upper, lower = element.findall(".//sup"), element.findall(".//sub")
        if len(upper) == len(lower) == 1:
            top, bottom = upper[0].text_content().strip(), lower[0].text_content().strip()
            if top.isdigit() and bottom.isdigit():
                element.clear(keep_tail=True)
                element.text = f"{top}/{bottom}"
    citations = [
        element
        for element in root.iter("sup")
        if "reference" in (element.get("class") or "").split()
        or re.fullmatch(r"\[\d+\]", element.text_content().strip())
    ]
    for element in reversed(citations):
        element.drop_tree()
    for element in list(root.iter("sup")):
        text = element.text_content().strip()
        element.clear(keep_tail=True)
        element.text = text.translate(SUPERSCRIPT)


def cell_text(cell) -> str:
    """Cell text with line breaks and nested list items kept as '; '."""
    for element in cell.iter():
        tag = _tag(element)
        if tag == "br" or (tag in ("li", "p", "div", "dd", "dt") and element is not cell):
            element.tail = "\n" + (element.tail or "")
    parts = [_norm(line) for line in cell.text_content().split("\n")]
    return "; ".join(p for p in parts if p)


def table_grid(table) -> list[list[dict | None]]:
    """Rows of the table as full-width cell lists after span expansion."""
    grid: dict[tuple[int, int], dict] = {}
    count = 0
    rows = table.xpath("./tr|./thead/tr|./tbody/tr|./tfoot/tr")
    for r, tr in enumerate(rows):
        c = 0
        for element in tr.xpath("./th|./td"):
            while (r, c) in grid:
                c += 1
            cell = {"id": count, "text": cell_text(element), "th": _tag(element) == "th"}
            span = [
                max(1, min(int(re.sub(r"\D", "", element.get(k) or "") or 1), cap))
                for k, cap in (("rowspan", 200), ("colspan", 50))
            ]
            for dr in range(span[0]):
                for dc in range(span[1]):
                    grid.setdefault((r + dr, c + dc), cell)
            c += span[1]
            count += 1
    if not grid:
        return []
    height = max(r for r, _ in grid) + 1
    # Width comes from rows with several cells, so an oversized colspan on a
    # full-width title row (colspan="10" over three columns) adds no columns.
    row_ids: dict[int, set[int]] = {}
    for (r, _), cell in grid.items():
        row_ids.setdefault(r, set()).add(cell["id"])
    multi = [c + 1 for (r, c) in grid if len(row_ids[r]) > 1]
    width = max(multi) if multi else max(c for _, c in grid) + 1
    return [[grid.get((r, c)) for c in range(width)] for r in range(height)]


def _full_width(row: list[dict | None]) -> bool:
    ids = {cell["id"] for cell in row if cell}
    return len(row) > 1 and len(ids) == 1 and all(row)


def table_units(
    table, prefix: str, key: str, start: int, inherited: list[str] | None = None
) -> tuple[list[dict], str, list[str]]:
    """Completed row units of one table, its context path, and column labels.

    A table without header rows takes `inherited` column labels (from the table
    that wraps it) when the column counts match.
    """
    grid = table_grid(table)
    caption = table.find("caption")
    title = [cell_text(caption)] if caption is not None else []
    header_rows = []
    body_start = 0
    for row in grid:
        if _full_width(row) and not header_rows:
            title.append(row[0]["text"])
        elif all(cell and cell["th"] for cell in row if cell) and any(row):
            if _full_width(row):
                title.append(row[0]["text"])
            else:
                header_rows.append(row)
        else:
            break
        body_start += 1
    labels = []
    for c in range(len(grid[0]) if grid else 0):
        seen = []
        for row in header_rows:
            text = row[c]["text"] if row[c] else ""
            if text and text not in seen:
                seen.append(text)
        labels.append(" › ".join(seen))
    if not any(labels) and inherited and grid and len(inherited) == len(grid[0]):
        labels = list(inherited)
    base = " › ".join([prefix, *[t for t in title if t]])
    units, group = [], ""
    for offset, row in enumerate(grid[body_start:]):
        if _full_width(row):
            group = row[0]["text"]
            continue
        cells, seen_ids = [], set()
        for c, cell in enumerate(row):
            if cell and cell["id"] not in seen_ids:
                seen_ids.add(cell["id"])
                cells.append((labels[c] if c < len(labels) else "", cell["text"]))
        values = [text for _, text in cells]
        if not any(values):
            continue
        if any(labels):
            body = " | ".join(f"{lab}: {v}" if lab else v for lab, v in cells if v)
        elif _row_header(row) and len(cells) > 1:
            body = f"{values[0]}: " + " | ".join(v for v in values[1:] if v)
        else:
            body = " | ".join(v for v in values if v)
        context = " › ".join(p for p in (base, group) if p)
        units.append(
            {
                "completed": f"{context} — {body}",
                "group": context,
                "key": f"{prefix}|{key}|{group}",
                "body": body,
                "labels": [lab for lab, _ in cells] if any(labels) else [],
                "values": values,
                "kind": "table_row",
                "pos": start + offset,
            }
        )
    return units, base, labels


def _row_header(row: list[dict | None]) -> bool:
    """Whether the row's first distinct cell is a header cell (infobox key)."""
    first = next((cell for cell in row if cell), None)
    return bool(first and first["th"])


def page_units(title: str, html: bytes | str) -> list[dict]:
    """All units of one page: faithful table rows plus block units."""
    from lxml import html as lxml_html

    root = lxml_html.fromstring(html)
    clean_superscripts(root)
    for element in list(root.iter()):
        if _tag(element) in DROP_TAGS:
            element.drop_tree()
    page = title.replace("_", " ")
    sections: list[str] = []
    units: list[dict] = []
    tables = 0

    def emit_table(table, prefix: str, inherited: list[str] | None = None) -> None:
        """A table's rows, then each directly nested table under its title."""
        nonlocal tables
        nested = [
            inner
            for inner in table.iter("table")
            if inner is not table and next(inner.iterancestors("table"), None) is table
        ]
        for inner in nested:
            inner.getparent().remove(inner)
        rows, base, labels = table_units(table, prefix, f"t{tables}", len(units), inherited)
        units.extend(rows)
        tables += 1
        for inner in nested:
            emit_table(inner, base, labels)

    def walk(element) -> None:
        nonlocal sections
        for child in element:
            if not isinstance(child.tag, str):
                continue
            tag = _tag(child)
            prefix = " › ".join([page, *sections[1:]])
            if len(tag) == 2 and tag[0] == "h" and tag[1].isdigit():
                text = _norm(child.text_content())
                if text:
                    sections = sections[: int(tag[1]) - 1] + [text]
            elif tag == "table":
                emit_table(child, prefix)
            elif tag in BLOCK_TAGS:
                text = _norm(child.text_content())
                if len(text) >= 2:
                    units.append(
                        {
                            "completed": f"{prefix}: {text}",
                            "group": prefix,
                            "key": prefix,
                            "body": text,
                            "labels": [],
                            "values": [],
                            "kind": BLOCK_TAGS[tag],
                            "pos": len(units),
                        }
                    )
            else:
                walk(child)

    walk(root)
    return units


def soft_redirect_target(content: bytes) -> str | None:
    """The meta-refresh target of a Kiwix soft-redirect page, if any."""
    from lxml import html as lxml_html

    root = lxml_html.fromstring(content[:8192].decode("utf-8", errors="replace"))
    for meta in root.iter("meta"):
        if (meta.get("http-equiv") or "").casefold() != "refresh":
            continue
        if found := REFRESH_URL.search(meta.get("content") or ""):
            target = html_lib.unescape(unquote(found.group(1)))
            return target.strip("'\"").removeprefix("./")
    return None


class ZimPages:
    """Page HTML by title from a Kiwix ZIM, following both redirect kinds."""

    def __init__(self, zim: str) -> None:
        from libzim.reader import Archive

        self.archive = Archive(zim)

    def html(self, title: str) -> bytes | None:
        visited = []
        while title not in visited:
            visited.append(title)
            if not self.archive.has_entry_by_path(title):
                return None
            entry = self.archive.get_entry_by_path(title)
            if entry.is_redirect:
                title = entry.get_redirect_entry().path
                continue
            raw = bytes(entry.get_item().content)
            if target := soft_redirect_target(raw):
                title = target.split("#", 1)[0]
                continue
            return raw
        return None
