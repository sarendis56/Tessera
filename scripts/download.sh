#!/usr/bin/env bash
# Download the models, datasets, Wikipedia snapshot, PixelRAG indexes, and tools.
# Usage: bash scripts/download.sh [core|hybrid|pixelrag-text|screenshot|baselines|agent|all]...
#   core           reader, BGE reranker, 2025-08 Wikipedia ZIM, questions        (~125 GB)
#   hybrid         Qwen3-Embedding-0.6B, PixelRAG article list, Java + Anserini (~7 GB)
#   pixelrag-text  PixelRAG released text index + Qwen3-VL-Embedding-2B         (~134 GB)
#   screenshot     PixelRAG fine-tuned screenshot index, LoRA, Qwen3-VL-Reranker (~225 GB)
#   baselines      HtmlRAG pruners                                               (~4 GB)
#   agent          FRAMES and MuSiQue                                            (<1 GB)
#   readers        Gemma 4 E4B and 31B readers                                   (~80 GB)
# `data/` and `models/` may be symlinks to a larger disk. Every source is pinned to
# the revision used in the paper.
set -euo pipefail
cd "$(dirname "$0")/.."

model() {  # model HF_ID REVISION
    hf download "$1" --revision "$2" --local-dir "models/$1" \
        --include "*.json" "*.safetensors" "*.model" "*.txt" "*.py" "*.jinja" "*.bin"
}
dataset() {  # dataset HF_ID REVISION LOCAL_DIR PATTERN...
    local id="$1" rev="$2" dir="$3"
    shift 3
    hf download "$id" --repo-type dataset --revision "$rev" --local-dir "$dir" --include "$@"
}
fetch() {  # fetch URL DEST (resumes a partial download)
    [[ -s "$2" ]] && return
    mkdir -p "$(dirname "$2")"
    curl -fL --retry 5 -C - -o "$2.part" "$1"
    mv "$2.part" "$2"
}
PIXELRAG_REV=37c02243cec463f5369270f79a08a8432d5ab55e

core() {
    model Qwen/Qwen3.5-4B 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
    model BAAI/bge-reranker-v2-m3 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e
    fetch https://download.kiwix.org/zim/wikipedia/wikipedia_en_all_maxi_2025-08.zim \
        data/wikipedia/wikipedia_en_all_maxi_2025-08.zim
    dataset google-research-datasets/natural_questions e8103d566bef4154c2c12b17c6095ec5275840cc \
        data/raw/natural_questions "default/validation-*.parquet"
    fetch https://storage.googleapis.com/tapas_models/2021_07_22/nq_tables/interactions/dev.jsonl \
        data/raw/nq_tables/dev.jsonl
    fetch https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv \
        data/raw/simpleqa/simple_qa_test_set.csv
}

hybrid() {
    model Qwen/Qwen3-Embedding-0.6B 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
    # Tokenizer PixelRAG chunked its text index with; the hybrid corpus reuses it.
    model Qwen/Qwen3-VL-Embedding-2B 9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda
    dataset StarTrail-org/pixelrag-tiles 84bc3285edf5b22fa73e39ea3b03f9d012181cbe data/pixelrag articles.json
    # Article IDs of the released text index define the corpus (7,060,072 articles).
    dataset StarTrail-org/pixelrag-faiss-indexes "$PIXELRAG_REV" data/pixelrag \
        "text_search_index_1024_normed/metadata.npz" "text_search_index_1024_normed/summary.json"
    mkdir -p tools
    fetch https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.11%2B10/OpenJDK21U-jre_x64_linux_hotspot_21.0.11_10.tar.gz \
        tools/jre.tar.gz
    [[ -d tools/jre ]] || { mkdir -p tools/jre && tar -xzf tools/jre.tar.gz -C tools/jre --strip-components=1; }
    fetch https://repo1.maven.org/maven2/io/anserini/anserini/2.2.0/anserini-2.2.0-fatjar.jar \
        tools/anserini-2.2.0-fatjar.jar
}

pixelrag-text() {
    model Qwen/Qwen3-VL-Embedding-2B 9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda
    dataset StarTrail-org/pixelrag-tiles 84bc3285edf5b22fa73e39ea3b03f9d012181cbe data/pixelrag articles.json
    dataset StarTrail-org/pixelrag-faiss-indexes "$PIXELRAG_REV" data/pixelrag "text_search_index_1024_normed/*"
}

screenshot() {
    model Qwen/Qwen3-VL-Embedding-2B 9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda
    model Qwen/Qwen3-VL-Reranker-2B 4bd860ac4f15ad1897a214615cccc700f8f71818
    hf download Chrisyichuan/wiki-screenshot-embedding-lora --revision d2d93b91928fbf37e2ffbcb807969c94637ea11c \
        --local-dir models/Chrisyichuan/wiki-screenshot-embedding-lora \
        --include "lora_vit/ckpt200/adapter_config.json" "lora_vit/ckpt200/adapter_model.safetensors"
    dataset StarTrail-org/pixelrag-tiles 84bc3285edf5b22fa73e39ea3b03f9d012181cbe data/pixelrag articles.json
    dataset StarTrail-org/pixelrag-faiss-indexes "$PIXELRAG_REV" data/pixelrag "search_index_lora_vit_ckpt200_v2/*"
}

baselines() {
    model BAAI/bge-large-en abe7d9d814b775ca171121fb03f394dc42974275
    model zstanjj/HTML-Pruner-Llama-1B b150e7aad1e100c2a9975cb43bbbf9aea65751a0
}

agent() {
    dataset google/frames-benchmark 58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef data/raw/frames test.tsv
    dataset dgslibisey/MuSiQue c8f4f8c9465fb69d31a8eae894c3fd509c4ca321 data/raw/musique musique_ans_v1.0_dev.jsonl
}

readers() {
    model google/gemma-4-E4B-it ee0ef6023621cff504d758262d4e04895a5af4a2
    model google/gemma-4-31B-it 842da3794eaa0b77d5f08bae87a17459d91ff475
}

groups=("$@")
[[ ${#groups[@]} -gt 0 ]] || groups=(core)
[[ "${groups[0]}" != all ]] || groups=(core hybrid pixelrag-text screenshot baselines agent readers)
for group in "${groups[@]}"; do
    echo "== $group"
    "$group"
done
