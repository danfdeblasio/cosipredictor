"""SPECTER2 document embeddings for submissions (title + keywords + abstract)."""
import numpy as np
import torch
from adapters import AutoAdapterModel
from transformers import AutoTokenizer

from common import prepare_text

BASE = "allenai/specter2_base"
ADAPTERS = {"proximity": "allenai/specter2", "classification": "allenai/specter2_classification"}


class Specter2:
    def __init__(self, adapter="proximity", device=None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE)
        self.model = AutoAdapterModel.from_pretrained(BASE)
        self.model.load_adapter(ADAPTERS[adapter], source="hf", load_as=adapter)
        self.model.set_active_adapters(adapter)
        self.model.to(self.device).eval()

    def texts(self, df):
        df = prepare_text(df)
        sep = self.tokenizer.sep_token
        return (df["title"] + sep + df["keywords"] + ". " + df["abstract"]).tolist()

    @torch.no_grad()
    def encode(self, df, batch_size=8, progress=False):
        texts = self.texts(df)
        order = np.argsort([len(t) for t in texts])  # length-sorted batches minimize padding
        out = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float32)
        for n, i in enumerate(range(0, len(texts), batch_size)):
            idx = order[i:i + batch_size]
            batch = self.tokenizer([texts[j] for j in idx], padding=True, truncation=True,
                                   max_length=512, return_tensors="pt")
            try:
                out[idx] = self._cls(batch)
            except RuntimeError as e:
                if self.device != "mps" or "out of memory" not in str(e):
                    raise
                print("  MPS out of memory; continuing on CPU", flush=True)
                self.device = "cpu"
                self.model.to("cpu")
                torch.mps.empty_cache()
                out[idx] = self._cls(batch)
            if self.device == "mps" and n % 20 == 0:
                torch.mps.empty_cache()
            if progress and n % 100 == 0:
                print(f"  {i + len(idx)}/{len(texts)}", flush=True)
        return out

    def _cls(self, batch):
        batch = batch.to(self.device)
        return self.model(**batch).last_hidden_state[:, 0].float().cpu().numpy()  # [CLS] token
