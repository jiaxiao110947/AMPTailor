#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare length-conditioned training corpus from raw peptide lines."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

AA20 = set("ACDEFGHIKLMNPQRSTVWY")
EOS_TEXT = "<|endoftext|>"
LEN_PREFIX_RX = re.compile(r"^LEN([1-9][0-9]*):")


def _clean_seq(raw: str) -> str:
    s = raw.strip().upper().replace(" ", "")
    # Length prefixes are conditioning metadata, not peptide residues.
    # Strip repeated prefixes too, so reprocessing cannot add LEN residues.
    while LEN_PREFIX_RX.match(s):
        s = LEN_PREFIX_RX.sub("", s, count=1)
    if s.endswith(EOS_TEXT.upper()):
        s = s[: -len(EOS_TEXT)]
    return "".join(c for c in s if c in AA20)


def make_length_line(seq: str, length: int, ensure_eos: bool) -> str:
    body = f"LEN{int(length)}:{seq}"
    if ensure_eos and (not body.endswith(EOS_TEXT)):
        body += EOS_TEXT
    return body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--stats_json", default="")
    ap.add_argument("--len_min", type=int, default=2)
    ap.add_argument("--len_max", type=int, default=60)
    ap.add_argument("--ensure_eos", action="store_true")
    ap.add_argument("--skip_existing_prefix", action="store_true")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    length_counter: Counter = Counter()
    n_in = n_keep = n_drop = 0

    with inp.open("r", encoding="utf-8") as fr, out.open("w", encoding="utf-8") as fw:
        for ln in fr:
            n_in += 1
            raw = ln.strip()
            if not raw:
                continue
            if args.skip_existing_prefix and LEN_PREFIX_RX.match(raw.upper()):
                fw.write(raw + ("\n" if not raw.endswith("\n") else ""))
                m = LEN_PREFIX_RX.match(raw.upper())
                if m:
                    length_counter[int(m.group(1))] += 1
                n_keep += 1
                continue

            seq = _clean_seq(raw)
            if not seq:
                n_drop += 1
                continue
            L = len(seq)
            if L < int(args.len_min) or L > int(args.len_max):
                n_drop += 1
                continue
            length_counter[L] += 1
            fw.write(make_length_line(seq, L, bool(args.ensure_eos)) + "\n")
            n_keep += 1

    stats = {
        "input": str(inp),
        "output": str(out),
        "n_in": int(n_in),
        "n_keep": int(n_keep),
        "n_drop": int(n_drop),
        "len_min": int(min(length_counter) if length_counter else 0),
        "len_max": int(max(length_counter) if length_counter else 0),
        "length_histogram": {str(k): int(v) for k, v in sorted(length_counter.items())},
        "top_lengths": [{"length": int(k), "count": int(v)} for k, v in length_counter.most_common(20)],
    }
    stats_path = Path(args.stats_json) if args.stats_json else out.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[OK] in={n_in} keep={n_keep} drop={n_drop}")
    print(f"[OK] length range: {stats['len_min']}-{stats['len_max']}")
    print(f"[OK] wrote: {out}")
    print(f"[OK] stats: {stats_path}")


if __name__ == "__main__":
    main()
