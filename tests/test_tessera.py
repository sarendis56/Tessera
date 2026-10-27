from tessera.common import mcnemar_p
from tessera.layer import pack, prefilter
from tessera.retrieve import rrf
from tessera.units import page_units

PAGE = b"""<html><body>
<h1>Ashes</h1><p>The Ashes is a Test cricket series.</p>
<h2>Results</h2>
<table>
  <caption>Series results</caption>
  <tr><th rowspan="2">Season</th><th colspan="2">Score</th></tr>
  <tr><th>England</th><th>Australia</th></tr>
  <tr><td colspan="3">20th century</td></tr>
  <tr><td>1932<br>1933</td><td>4</td><td>1</td></tr>
</table>
<table><tr><th>Captain</th><td>Douglas Jardine<sup class="reference">[1]</sup></td></tr></table>
</body></html>"""


class WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


def test_faithful_grid_labels_every_cell():
    units = page_units("Ashes", PAGE)
    row = next(u for u in units if u["kind"] == "table_row" and u["values"][0].startswith("1932"))
    assert row["labels"] == ["Season", "Score › England", "Score › Australia"]
    assert row["values"] == ["1932; 1933", "4", "1"]
    assert row["group"] == "Ashes › Results › Series results › 20th century"
    infobox = next(u for u in units if u["body"].startswith("Captain"))
    assert infobox["body"] == "Captain: Douglas Jardine"  # citation superscript dropped


def test_pack_prints_each_group_once_within_budget():
    units = [dict(u, score=float(i)) for i, u in enumerate(page_units("Ashes", PAGE))]
    text = pack(units, budget=40, tokenizer=WordTokenizer())
    assert len(text.split()) <= 40
    assert text.count("Series results") <= 1
    assert prefilter("Who was captain in the Ashes?", units, keep=1)[0]["body"].startswith("Captain")


def test_rrf_fuses_by_vector_id():
    dense = [{"load_index": 0, "hits": [{"vector_id": 1}, {"vector_id": 2}]}]
    sparse = [
        {"load_index": 0, "hits": [{"vector_id": 2, "bm25_score": 9.0}, {"vector_id": 3, "bm25_score": 1.0}]}
    ]
    assert [h["vector_id"] for h in rrf(dense, sparse)[0]["hits"]] == [2, 1, 3]


def test_mcnemar():
    assert mcnemar_p(0, 0) == 1.0
    assert abs(mcnemar_p(2, 8) - 0.109375) < 1e-12
