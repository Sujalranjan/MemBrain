"""Evaluation datasets: public BEIR benchmarks (downloaded on demand) and the bundled in-domain set."""

import csv
import io
import json
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
CACHE_DIR = EVAL_DIR / ".cache"
BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
BEIR_DATASETS = {"scifact", "nfcorpus"}


@dataclass
class Doc:
    id: str
    title: str
    text: str


@dataclass
class Dataset:
    name: str
    docs: list[Doc]
    queries: dict[str, str]
    qrels: dict[str, set[str]]  # query id -> relevant doc ids
    categories: dict[str, str] = field(default_factory=dict)  # query id -> query type, if labelled
    source: str = ""


def load_beir(name: str, split: str = "test") -> Dataset:
    root = CACHE_DIR / name
    if not (root / "corpus.jsonl").exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading BEIR {name}...")
        with urllib.request.urlopen(BEIR_URL.format(name=name), timeout=120) as resp:
            zipfile.ZipFile(io.BytesIO(resp.read())).extractall(CACHE_DIR)

    qrels: dict[str, set[str]] = {}
    with open(root / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if int(row["score"]) > 0:
                qrels.setdefault(row["query-id"], set()).add(row["corpus-id"])

    queries = {}
    with open(root / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            if q["_id"] in qrels:
                queries[q["_id"]] = q["text"]

    docs = []
    with open(root / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["text"].strip():
                docs.append(Doc(d["_id"], d.get("title", "").strip() or "Untitled", d["text"]))
    return Dataset(name, docs, queries, qrels, source=f"BEIR {name} ({split} split)")


def load_bundled(name: str = "membrain") -> Dataset:
    data = json.loads((EVAL_DIR / "data" / f"{name}.json").read_text(encoding="utf-8"))
    docs = [Doc(d["id"], d["title"], d["body"]) for d in data["docs"]]
    queries = {q["id"]: q["query"] for q in data["queries"]}
    qrels = {q["id"]: set(q["relevant"]) for q in data["queries"]}
    categories = {q["id"]: q["type"] for q in data["queries"]}
    return Dataset(name, docs, queries, qrels, categories, source=data.get("source", "bundled"))


def load(name: str) -> Dataset:
    return load_beir(name) if name in BEIR_DATASETS else load_bundled(name)
