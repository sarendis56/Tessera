"""HtmlRAG generative block scorer on native transformers Llama.

Port of `LlamaForHTMLTreeGeneration` from the released HTML-Pruner-Llama-1B
remote code (written against transformers 4.45). The released class is a stock
Llama causal LM plus `generate_html_tree`, which scores the block tree by the
model's next-token probabilities over path-token prefixes. The method is copied
unchanged except that the node tree is exported with a preorder walk, which is
what the released `TokenDotExporter` produced. `GenHTMLPruner` loads it via
`load_gen_pruner` instead of `trust_remote_code`.
"""

from __future__ import annotations

import bs4
import numpy as np
import torch
from anytree import PreOrderIter
from htmlrag.html_utils import TokenIdNode
from transformers import AutoTokenizer, LlamaForCausalLM

TEMPLATE = (
    "**HTML**: ```{input_html}```\n**Question**: **{question}**\n Your task is to "
    "identify the most relevant text piece to the given question in the HTML "
    "document. This text piece could either be a direct paraphrase to the fact, or "
    "a supporting evidence that can be used to infer the fact. The overall length "
    "of the text piece should be more than 300 words and less than 500 words. You "
    "should provide the path to the text piece in the HTML document. An example for "
    "the output is: <html 1><body><div 2><p>Some key information..."
)


def _node_name(node) -> str:
    return f"{node.name}|{node.prob}|{node.input_ids}"


class LlamaForHTMLTreeGeneration(LlamaForCausalLM):
    """Stock Llama with the released block-tree scoring method."""

    @torch.inference_mode()
    def generate_html_tree(self, tokenizer, query, htmls, block_tree, **kwargs):
        max_seq_length = kwargs.pop("max_seq_length", 131072)
        res_html_refs = []
        for idx, _htmls in enumerate(htmls):
            if isinstance(_htmls, str):
                _htmls = [_htmls]
            else:
                lens = [len(tokenizer.encode(html)) for html in _htmls]
                while sum(lens) > max_seq_length - 2048 and len(_htmls) > 1:
                    longest = lens.index(max(lens))
                    lens.pop(longest)
                    _htmls.pop(longest)
            model_input = TEMPLATE.format(input_html="\n".join(_htmls), question=query)
            inputs = tokenizer.apply_chat_template(
                [{"role": "user", "content": model_input}],
                add_special_tokens=True,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            )
            soup = bs4.BeautifulSoup("", "html.parser")
            for html in _htmls:
                soup.append(bs4.BeautifulSoup(html, "html.parser"))

            tree = block_tree[idx]
            is_leaf = [p[2] for p in tree]
            paths = [p[1] for p in tree]
            token_id_paths = [
                tokenizer.encode("<" + "><".join(path) + ">", add_special_tokens=False) for path in paths
            ]
            root = TokenIdNode(-1)
            for path in token_id_paths:
                parent = root
                for i, token_id in enumerate(path):
                    found = [c for c in parent.children if c.name == token_id]
                    parent = (
                        found[0] if found else TokenIdNode(token_id, parent=parent, input_ids=path[: i + 1])
                    )

            queue = [root]
            while queue:
                node = queue.pop(0)
                children = node.children
                if len(children) == 1:
                    children[0].prob = str(np.float32(1.0))
                    queue.append(children[0])
                    continue
                if not children:
                    continue
                prefix = torch.tensor(node.input_ids, dtype=torch.long).unsqueeze(0)
                ids = torch.cat([inputs["input_ids"][idx : idx + 1], prefix], dim=1)
                logits = self(input_ids=ids.to(self.device), return_dict=True).logits
                force = torch.tensor([c.name for c in children], device=self.device)
                # Released behaviour: position 0 of the untrimmed logits.
                probs = torch.gather(logits[:, 0, :], -1, force.unsqueeze(0))
                probs = torch.nn.functional.softmax(probs, dim=-1)
                probs = probs.squeeze(0).detach().to(torch.float32).cpu().numpy()
                for i, child in enumerate(children):
                    child.prob = str(probs[i])
                    queue.append(child)

            res_html_refs.append(
                {
                    "html": str(soup),
                    "paths": paths,
                    "is_leaf": is_leaf,
                    "path_token_ids": token_id_paths,
                    "node_tree": [_node_name(n) for n in PreOrderIter(root)],
                }
            )
        return res_html_refs


def load_gen_pruner(model_path: str, device: str = "cuda"):
    """A `GenHTMLPruner` whose model is the native-Llama port."""
    from htmlrag import GenHTMLPruner

    pruner = GenHTMLPruner.__new__(GenHTMLPruner)
    # The config names flash_attention_2, which transformers 4.45 ignored (it read
    # `_attn_implementation`), so the released run used SDPA.
    pruner.gen_model = LlamaForHTMLTreeGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).eval()
    pruner.gen_model.to(device)
    pruner.node_tokenizer = AutoTokenizer.from_pretrained(model_path)
    pruner.log_threshold = 1e-9
    return pruner
