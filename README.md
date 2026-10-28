# Tessera

Code for *Tessera*, a training-free evidence delivery layer for web retrieval-augmented generation.

Web RAG pipelines use one object, a text chunk or a screenshot tile, as the unit of indexing, ranking, and reading. Tessera leaves the pipeline's index, retriever, and reader as they are and changes what the reader gets. It takes the pages behind the ranked list, rebuilds their HTML into small *structure-completed units* (a table row with its column headers, a list item with its section path), selects the units relevant to the query, and packs them into the token budget of the last evidence item. On NQ, NQ-Tables, and SimpleQA, Tessera adds 5 to 6 points of answer accuracy to text and screenshot pipelines.

## Layout

```
src/tessera/
  units.py       page HTML -> structure-completed units (faithful table grids, block units)
  layer.py       Tessera: anchor pages, prefilter, BGE scoring, budget-neutral packing
  data.py        question populations (NQ, NQ-Tables, SimpleQA, FRAMES, MuSiQue)
  corpus.py      hybrid pipeline: Trafilatura extraction and 1,024-token chunks
  dense.py       hybrid pipeline: Qwen3-Embedding-0.6B and a FAISS IVF index
  bm25.py        hybrid pipeline: Anserini BM25
  retrieve.py    hybrid pipeline end to end: dense + BM25, RRF, BGE reranking
  pixelrag.py    PixelRAG's released text and screenshot pipelines
  read.py        reader (any OpenAI-compatible vLLM server)
  grade.py       SimpleQA-template grading and paired McNemar tests
  baselines.py   Docling / Unstructured unit builders, HtmlRAG
  agent.py       multi-step search agent with Tessera on every search call
scripts/         download.sh, index build, and one run script per paper table
```

Every file that a step writes is JSONL with one row per question. A row holds `problem`, `original_data` (gold answers), and `hits`, the ranked evidence.

## Using Tessera on your own pipeline

Tessera needs, per question, a ranked list whose hits carry a page `title`, `text`, and `rerank_score`, plus a way to fetch page HTML. With pages in a Kiwix ZIM:

```sh
python -m tessera.layer pools --candidates my_ranked.jsonl --zim wiki.zim --out run/
python -m tessera.layer score --out run/
python -m tessera.layer build --candidates my_ranked.jsonl --out run/   # -> run/tessera_candidates.jsonl
```

For other page stores, call `tessera.units.page_units(title, html)` and the `prefilter` and `pack` functions in `tessera/layer.py` directly.

## Setup

Linux, Python 3.12, [uv](https://docs.astral.sh/uv/), and NVIDIA GPUs (we used A100 80 GB).

```sh
uv sync --all-extras                     # or plain `uv sync` for the text pipelines only
source .venv/bin/activate
export OPENAI_API_KEY=...                # grading
```

The screenshot pipeline re-renders tiles with PixelRAG's renderer at the commit we used, with one fix for a startup race:

```sh
git clone https://github.com/StarTrail-org/PixelRAG third_party/PixelRAG
git -C third_party/PixelRAG checkout 2d0ff8aaed1bf4181d655debb4b7b090c7dcb237
git -C third_party/PixelRAG apply ../../patches/pixelrag-fast-cdp-fork-race.patch
export PYTHONPATH=$PWD/third_party/PixelRAG/render/src:$PWD/third_party/PixelRAG/embed/src:$PWD/third_party/PixelRAG/serve/src
```

## Data and models

`scripts/download.sh GROUP...` downloads everything into `models/<hf-id>/`, `data/`, and `tools/`, pinned to the revisions used in the paper. `data/` and `models/` may be symlinks to a larger disk.

| Group             | What                                                                                 | Source                                                                                                                      | Size   |
| ----------------- | ------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- | ------ |
| `core`          | Wikipedia snapshot, 2025-08 (7,060,072 articles used)                                | [Kiwix](https://download.kiwix.org/zim/wikipedia/) `wikipedia_en_all_maxi_2025-08.zim`                                     | 112 GB |
|                   | Reader Qwen3.5-4B, reranker BGE-reranker-v2-m3                                       | HF `Qwen/Qwen3.5-4B`, `BAAI/bge-reranker-v2-m3`                                                                         | 12 GB  |
|                   | Natural Questions (validation), NQ-Tables (dev), SimpleQA                            | HF `google-research-datasets/natural_questions`; TAPAS GCS; OpenAI simple-evals                                           | 1.4 GB |
| `hybrid`        | Embedding model, chunk tokenizer                                                     | HF `Qwen/Qwen3-Embedding-0.6B`, `Qwen/Qwen3-VL-Embedding-2B`                                                            | 6 GB   |
|                   | Article list and IDs of PixelRAG's corpus                                            | HF `StarTrail-org/pixelrag-tiles`, `StarTrail-org/pixelrag-faiss-indexes`                                               | 0.6 GB |
|                   | Java 21, Anserini 2.2.0                                                              | Adoptium, Maven Central                                                                                                     | 0.2 GB |
| `pixelrag-text` | PixelRAG's released text index (15.7M chunks)                                        | HF `StarTrail-org/pixelrag-faiss-indexes`                                                                                 | 130 GB |
| `screenshot`    | PixelRAG's fine-tuned screenshot index (26.3M tiles), its LoRA, Qwen3-VL-Reranker-2B | HF `StarTrail-org/pixelrag-faiss-indexes`, `Chrisyichuan/wiki-screenshot-embedding-lora`, `Qwen/Qwen3-VL-Reranker-2B` | 225 GB |
| `baselines`     | HtmlRAG pruners                                                                      | HF `BAAI/bge-large-en`, `zstanjj/HTML-Pruner-Llama-1B`                                                                  | 4 GB   |
| `agent`         | FRAMES (test), MuSiQue (answerable dev)                                              | HF `google/frames-benchmark`, `dgslibisey/MuSiQue`                                                                      | <1 GB  |
| `readers`       | Gemma 4 E4B and 31B readers                                                          | HF `google/gemma-4-E4B-it`, `google/gemma-4-31B-it`                                                                     | 80 GB  |

Then freeze the question populations (7,851 single-hop questions; 824 FRAMES and 1,000 MuSiQue questions):

```sh
bash scripts/download.sh core hybrid agent
python -m tessera.data          # -> data/populations/{nq,nq_tables,simpleqa,frames,musique}.jsonl
```

### Building the hybrid text index

Our hybrid pipeline is built from the same snapshot and article set as PixelRAG's text index: Trafilatura 2.0 text, 15,513,418 chunks of 1,024 tokens, Qwen3-Embedding-0.6B with an IVF4096 index, and Anserini BM25.

```sh
GPUS="0 1 2 3 4 5 6 7" bash scripts/build_hybrid_index.sh
```

Extraction takes about 2 hours with 32 CPU workers, and embedding 53 A100 GPU-hours. The outputs are 14 GB of chunks, 32 GB of intermediate embeddings, a 60 GB IVF index, and a 5 GB BM25 index under `data/hybrid/`.

## Reproducing the main results

Each run script retrieves, builds Tessera's reader inputs, serves the reader with vLLM, answers, grades with GPT-5.6 Luna, and prints paired comparisons (accuracy, gain, exact McNemar p). A step whose output exists is skipped, so a stopped script resumes.

### Table 3: one layer on many pipelines

Each line compares the pipeline's top 3 items with its top 2 items plus Tessera, as pooled accuracy over the 7,851 questions:

```sh
GPU=0 bash scripts/run_hybrid.sh                      # Hybrid text + BGE:          70.7 -> 75.7
GPU=0 bash scripts/run_pixelrag.sh text               # PixelRAG text + BGE:        67.7 -> 73.9
GPU=0 bash scripts/run_pixelrag.sh screenshot         # PixelRAG screenshot + VL:   69.6 -> 75.2
```

For the other readers, rerun a script with the reader swapped. All readers read the same inputs, packed under the Qwen3.5-4B tokenizer:

```sh
READER=models/google/gemma-4-E4B-it READER_NAME=google/gemma-4-E4B-it MAX_TOKENS=2048 \
    bash scripts/run_hybrid.sh                        # 70.7 -> 75.6
READER=models/google/gemma-4-31B-it READER_NAME=google/gemma-4-31B-it READER_GPU=0,1 \
    READER_ARGS="--tensor-parallel-size 2" bash scripts/run_hybrid.sh   # 72.0 -> 76.7
```

### Table 4: matched-budget baselines

These run on the hybrid pipeline and reuse the outputs of `run_hybrid.sh`:

```sh
GPU=0 HTMLRAG_GPUS="0 1" bash scripts/run_baselines.sh
# pooled: Unstructured 72.86, Docling 74.97, HtmlRAG 70.30, HtmlRAG-CE 70.02; Tessera 75.74
```

### Table 5: agentic search

The agent is Qwen3.5-4B with thinking on and up to eight searches:

```sh
AGENT_GPU=0 SEARCH_GPU=1 bash scripts/run_agent.sh chunks tessera docling chunks4
# FRAMES / MuSiQue: top-3 chunks 59.22 / 32.40, Tessera 64.44 / 33.50
```

The agent samples at temperature 1.0 (the Qwen3.5 model card settings for thinking mode), so a rerun gives slightly different numbers. The single-shot runs decode greedily.

### Notes

- `tessera.grade` sends each answer to GPT-5.6 Luna with the SimpleQA grading template (`src/tessera/simpleqa_grader_template.txt`, from PixelRAG's evaluation assets), reasoning effort `none`. A full single-hop run is 15,702 grading calls.
- Building the units of a page takes 15 ms on one CPU core, and scoring takes 0.2 GPU-seconds per question on an A100. The screenshot pipeline also renders every page among the top-50 tiles from the ZIM.
- Search over PixelRAG's released indexes uses nprobe 128 on NQ and NQ-Tables and 2,000 on SimpleQA, and the hybrid index uses nprobe 128 everywhere.

## Citation

```latex
@article{hua2026tessera,
  title={Tessera: Structure-Completed Evidence Units for Web Retrieval-Augmented Generation},
  author={Peichun Hua and He Sun and Zirui Zhuang and Qi Qi and Yiming Qiu and Shinan Liu and Yunming Xiao},
  journal={CoRR},
  volume={abs/2610.37080},
  year={2026},
  url={https://arxiv.org/abs/2610.37080},
  doi={10.48550/arXiv.2610.37080},
  eprinttype={arXiv},
  eprint={2610.37080},
  primaryClass={cs.IR}
}
```
