#!/usr/bin/env python3  # shebang
# -*- coding: utf-8 -*-  # encoding
import os  # os
import math  # math
import json  # json
import time  # time
import random  # random
import argparse  # argparse
from pathlib import Path  # Path
from typing import List, Dict, Tuple, Optional  # typing
from collections import Counter  # Counter
import statistics  # statistics
import torch  # torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast  # hf

EOS_TEXT = "<|endoftext|>"  # eos text marker
AA20 = "ACDEFGHIKLMNPQRSTVWY"  # aa20 letters
AA20_SET = set(list(AA20))  # aa20 set

def set_seed(seed: int) -> None:  # set seed
    random.seed(int(seed))  # py seed
    torch.manual_seed(int(seed))  # torch seed
    torch.cuda.manual_seed_all(int(seed))  # cuda seed

def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0, filter_value: float = -float("inf")) -> torch.Tensor:  # filter
    if logits.dim() == 1:  # 1d
        logits = logits.unsqueeze(0)  # add batch
        squeeze_back = True  # mark
    else:  # 2d
        squeeze_back = False  # mark
    bsz, vsz = logits.shape  # shape
    if int(top_k) > 0 and int(top_k) < int(vsz):  # topk
        topk_vals, _ = torch.topk(logits, int(top_k), dim=-1)  # topk vals
        kth = topk_vals[:, -1].unsqueeze(-1)  # kth
        logits = torch.where(logits < kth, torch.full_like(logits, float(filter_value)), logits)  # mask
    if float(top_p) < 1.0:  # topp
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)  # sort
        probs = torch.softmax(sorted_logits, dim=-1)  # softmax
        cumprobs = torch.cumsum(probs, dim=-1)  # cumsum
        remove = cumprobs > float(top_p)  # remove
        remove[:, 0] = False  # keep at least one
        remove_mask = torch.zeros_like(remove).scatter(1, sorted_idx, remove)  # scatter back
        logits = logits.masked_fill(remove_mask, float(filter_value))  # apply
    return logits.squeeze(0) if squeeze_back else logits  # return

def _read_first_letter_counts(train_eos_txt: str) -> Counter:  # read counts
    p = Path(train_eos_txt)  # path
    cnt: Counter = Counter()  # counter
    with p.open("r", encoding="utf-8") as f:  # open
        for ln in f:  # loop
            s = ln.strip()  # strip
            if not s:  # empty
                continue  # skip
            if s.endswith(EOS_TEXT):  # strip eos text
                s = s[:-len(EOS_TEXT)]  # drop
            s = "".join([c for c in s if c.isalpha()]).upper()  # keep letters
            if s and (s[0] in AA20_SET):  # valid first aa
                cnt[s[0]] += 1  # add
    return cnt  # return

def read_first_aa_distribution(train_eos_files: List[str], train_weights: Optional[List[float]] = None) -> Tuple[List[str], List[float]]:  # dist
    if not train_eos_files:  # guard
        raise ValueError("train_eos_files is empty")  # error
    if train_weights is None:  # default
        train_weights = [1.0] * len(train_eos_files)  # ones
    if len(train_weights) != len(train_eos_files):  # guard
        raise ValueError("len(train_weights) must equal len(train_eos_files)")  # error
    final: Counter = Counter()  # final
    for fp, w in zip(train_eos_files, train_weights):  # loop
        c = _read_first_letter_counts(fp)  # read
        if sum(c.values()) == 0:  # empty
            raise RuntimeError(f"首字母分布为空或无 AA20 首字母：{fp}")  # error
        for k, v in c.items():  # accumulate
            final[k] += float(w) * float(v)  # add weighted
    letters = list(AA20)  # fixed order
    weights = [float(final.get(a, 0.0)) for a in letters]  # weights
    if sum(weights) <= 0:  # guard
        raise RuntimeError(f"混合后首字母权重为 0：files={train_eos_files}")  # error
    return letters, weights  # return

def write_fasta(path: Path, seqs: List[str], header_prefix: str) -> None:  # write fasta
    path.parent.mkdir(parents=True, exist_ok=True)  # mkdir
    with path.open("w", encoding="utf-8") as f:  # open
        for i, s in enumerate(seqs, 1):  # loop
            f.write(f">{header_prefix}_{i}\n{s}\n")  # write

def summarize_seqs(seqs: List[str]) -> Dict[str, float]:  # summarize
    n = int(len(seqs))  # n
    bad = [s for s in seqs if any(c not in AA20_SET for c in s)]  # bad
    lens = [len(s) for s in seqs]  # lens
    lens_s = sorted(lens)  # sorted
    def q(sorted_vals: List[int], frac: float) -> float:  # quantile
        if not sorted_vals:  # empty
            return 0.0  # zero
        idx = int(float(frac) * float(len(sorted_vals) - 1))  # idx
        return float(sorted_vals[idx])  # val
    short8 = sum(1 for l in lens if int(l) < 8)  # short
    uniq_ratio = float(len(set(seqs)) / n) if n else 0.0  # uniq
    aa_counts = Counter("".join(seqs))  # counts
    total_aa = int(sum(aa_counts.values())) or 1  # total
    frac_pos = float((aa_counts.get("K", 0) + aa_counts.get("R", 0) + aa_counts.get("H", 0)) / total_aa)  # pos
    frac_neg = float((aa_counts.get("D", 0) + aa_counts.get("E", 0)) / total_aa)  # neg
    return {  # dict
        "n": float(n),  # n
        "bad_AA": float(len(bad)),  # bad
        "bad_AA_%": float(100.0 * len(bad) / n) if n else 0.0,  # bad%
        "short<8_%": float(100.0 * short8 / n) if n else 0.0,  # short%
        "len_mean": float(sum(lens) / n) if n else 0.0,  # mean
        "len_median": float(statistics.median(lens)) if n else 0.0,  # median
        "len_p5": float(q(lens_s, 0.05)),  # p5
        "len_p95": float(q(lens_s, 0.95)),  # p95
        "uniq_ratio": float(uniq_ratio),  # uniq
        "net_charge_proxy": float(frac_pos - frac_neg),  # proxy
    }  # end

class AMPTailorGeneratorAAOnly:  # tri generator
    def __init__(self, base_dir: str, expert_dir: str, anti_dir: str, tok_dir: str, seed: int = 42, fp16: bool = False, bf16: bool = False) -> None:  # init
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # device
        dtype = None  # dtype
        if self.device.type == "cuda":  # cuda
            if bool(bf16) and getattr(torch.cuda, "is_bf16_supported", lambda: False)():  # bf16
                dtype = torch.bfloat16  # bf16
            elif bool(fp16):  # fp16
                dtype = torch.float16  # fp16
        def _load(dirpath: str) -> GPT2LMHeadModel:  # load model
            try:  # new api
                return GPT2LMHeadModel.from_pretrained(dirpath, dtype=dtype if dtype else None)  # load
            except TypeError:  # old api
                return GPT2LMHeadModel.from_pretrained(dirpath, torch_dtype=dtype if dtype else None)  # load
        self.base = _load(base_dir).to(self.device).eval()  # base
        self.expert = _load(expert_dir).to(self.device).eval()  # expert
        self.anti = _load(anti_dir).to(self.device).eval()  # anti
        self.tok = GPT2TokenizerFast.from_pretrained(tok_dir)  # tok
        self.tok.padding_side = "left"  # left pad
        if self.tok.pad_token is None:  # pad none
            self.tok.pad_token = self.tok.eos_token  # pad=eos
        if self.tok.eos_token_id != self.tok.pad_token_id:  # guard
            raise RuntimeError(f"pad_id({self.tok.pad_token_id}) != eos_id({self.tok.eos_token_id})")  # error
        self.eos_id = int(self.tok.eos_token_id)  # eos id
        vs = int(self.tok.vocab_size)  # vocab size
        for name, m in [("base", self.base), ("expert", self.expert), ("anti", self.anti)]:  # check
            if int(m.config.vocab_size) != vs:  # mismatch
                raise RuntimeError(f"{name}.vocab_size={m.config.vocab_size} != tok.vocab_size={vs}")  # error
        for m in [self.base, self.expert, self.anti]:  # set ids
            m.config.pad_token_id = int(self.tok.pad_token_id)  # pad id
            m.config.eos_token_id = int(self.tok.eos_token_id)  # eos id
        vocab = self.tok.get_vocab()  # vocab
        def aa_to_id(a: str) -> int:  # map aa
            if a in vocab:  # direct
                ids = [int(vocab[a])]  # id
            else:  # encode
                ids = [int(x) for x in self.tok.encode(a, add_special_tokens=False)]  # ids
            if len(ids) != 1:  # guard
                raise RuntimeError(f"Tokenizer 不满足 1AA=1token：'{a}' -> ids={ids}")  # error
            return int(ids[0])  # return
        aa_ids = [aa_to_id(a) for a in AA20]  # aa ids
        allowed = torch.zeros(vs, dtype=torch.bool, device=self.device)  # allowed mask
        for tid in aa_ids:  # set aa
            allowed[int(tid)] = True  # allow
        allowed[int(self.eos_id)] = True  # allow eos
        self.allowed_token_mask = allowed  # store
        aa_no_eos = torch.zeros(vs, dtype=torch.bool, device=self.device)  # aa only
        for tid in aa_ids:  # set aa
            aa_no_eos[int(tid)] = True  # allow
        self.aa_no_eos = aa_no_eos  # store
        self.rng = torch.Generator(device=self.device)  # rng
        self.rng.manual_seed(int(seed))  # seed rng
        print(f"[mode] AMPTailor strict AA-only (AA20+EOS). allowed={int(allowed.sum().item())}/{vs}", flush=True)  # log

    @torch.no_grad()  # no grad
    def generate_batch(self, prompts: List[str], max_new_tokens: int, min_new_tokens: int, alpha: float, alpha_eos: Optional[float], temperature: float, top_k: int, top_p: float, keep_eos_after_filter: bool) -> List[str]:  # gen batch
        if alpha_eos is None:  # default
            alpha_eos = float(alpha)  # set
        enc = self.tok(prompts, add_special_tokens=False, padding=True, return_tensors="pt")  # encode
        input_ids = enc["input_ids"].to(self.device)  # ids
        attention_mask = enc["attention_mask"].to(self.device)  # mask
        position_ids = attention_mask.long().cumsum(dim=1) - 1  # pos
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)  # pad pos
        bsz = int(input_ids.shape[0])  # bsz
        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()  # prompt lens
        unfinished = torch.ones(bsz, dtype=torch.long, device=self.device)  # unfinished
        past_b = None  # past base
        past_e = None  # past expert
        past_a = None  # past anti
        for step in range(int(max_new_tokens)):  # steps
            if past_b is None:  # first
                inp = input_ids  # full
                am = attention_mask  # full mask
                pos = position_ids  # full pos
            else:  # next
                inp = input_ids[:, -1:].contiguous()  # last token
                am = attention_mask  # full mask
                pos = (am.long().sum(dim=1) - 1).unsqueeze(1)  # last pos
            ob = self.base(input_ids=inp, attention_mask=am, position_ids=pos, past_key_values=past_b, use_cache=True)  # forward base
            oe = self.expert(input_ids=inp, attention_mask=am, position_ids=pos, past_key_values=past_e, use_cache=True)  # forward expert
            oa = self.anti(input_ids=inp, attention_mask=am, position_ids=pos, past_key_values=past_a, use_cache=True)  # forward anti
            past_b, past_e, past_a = ob.past_key_values, oe.past_key_values, oa.past_key_values  # cache
            if step == 0:  # first step logits index
                last_idx = (attention_mask.sum(dim=1) - 1)  # last index
                base_next = ob.logits[torch.arange(bsz, device=self.device), last_idx, :]  # base
                expert_next = oe.logits[torch.arange(bsz, device=self.device), last_idx, :]  # expert
                anti_next = oa.logits[torch.arange(bsz, device=self.device), last_idx, :]  # anti
            else:  # later
                base_next = ob.logits[:, -1, :]  # base
                expert_next = oe.logits[:, -1, :]  # expert
                anti_next = oa.logits[:, -1, :]  # anti

            base_for_trunc = base_next.clone()  # base copy
            base_for_trunc = base_for_trunc.masked_fill(~self.allowed_token_mask.unsqueeze(0), -float("inf"))  # AA-only
            if int(min_new_tokens) > 0 and int(step) < int(min_new_tokens):  # min len
                base_for_trunc[:, int(self.eos_id)] = -float("inf")  # forbid eos
            base_for_trunc = torch.nan_to_num(base_for_trunc, nan=-1e4, posinf=1e4, neginf=-1e4)  # sanitize
            if float(temperature) != 1.0:  # temp for gating
                base_for_trunc = base_for_trunc / float(temperature)  # scale

            if int(top_k) > 0 or float(top_p) < 1.0:  # truncation
                base_trunc_logits = top_k_top_p_filtering(base_for_trunc, top_k=int(top_k), top_p=float(top_p))  # filter
                keep_mask = torch.isfinite(base_trunc_logits)  # V' mask
            else:  # no truncation
                keep_mask = self.allowed_token_mask.unsqueeze(0).expand(bsz, -1).clone()  # keep all

            if bool(keep_eos_after_filter):  # keep eos
                keep_mask[:, int(self.eos_id)] = True  # keep eos
            empty_rows = (keep_mask.sum(dim=1) == 0)  # empty
            if bool(empty_rows.any().item()):  # fix empty
                keep_mask[empty_rows] = self.allowed_token_mask.unsqueeze(0).expand(int(empty_rows.sum().item()), -1)  # fallback

            base_trunc = base_next.clone()  # base (unscaled) for combine
            base_trunc = base_trunc.masked_fill(~keep_mask, -float("inf"))  # apply V'

            logits = base_trunc + float(alpha) * (expert_next - anti_next)  # amptailor combine
            logits[:, int(self.eos_id)] = base_trunc[:, int(self.eos_id)] + float(alpha_eos) * (expert_next[:, int(self.eos_id)] - anti_next[:, int(self.eos_id)])  # eos combine

            if int(min_new_tokens) > 0 and int(step) < int(min_new_tokens):  # min len
                logits[:, int(self.eos_id)] = -float("inf")  # forbid eos
            if (unfinished == 0).any():  # finished
                logits[unfinished == 0, :] = -float("inf")  # block
                logits[unfinished == 0, int(self.eos_id)] = 0.0  # force eos
            logits = logits.masked_fill(~self.allowed_token_mask.unsqueeze(0), -float("inf"))  # AA-only
            logits = torch.nan_to_num(logits, nan=-1e4, posinf=1e4, neginf=-1e4)  # sanitize
            if float(temperature) != 1.0:  # temp for sampling
                logits = logits / float(temperature)  # scale

            probs = torch.softmax(logits, dim=-1)  # probs
            bad = (torch.isnan(probs).any(dim=-1) | torch.isinf(probs).any(dim=-1) | (probs.sum(dim=-1) <= 0))  # bad rows
            if bool(bad.any().item()):  # fix bad
                probs[bad] = 0.0  # zero
                if int(min_new_tokens) > 0 and int(step) < int(min_new_tokens):  # early
                    probs[bad][:, self.aa_no_eos] = 1.0  # uniform aa
                else:  # allow eos
                    probs[bad][:, int(self.eos_id)] = 1.0  # eos
                probs[bad] = probs[bad] / probs[bad].sum(dim=-1, keepdim=True)  # renorm

            next_tokens = torch.multinomial(probs, num_samples=1, generator=self.rng).squeeze(1)  # sample
            tokens_to_add = next_tokens * unfinished + int(self.tok.pad_token_id) * (1 - unfinished)  # add token
            eos_in = (tokens_to_add == int(self.eos_id))  # eos hit
            unfinished.mul_((~eos_in).long())  # update unfinished
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(1)], dim=1)  # append ids
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((bsz, 1))], dim=1)  # append mask
            new_pos = (attention_mask.long().cumsum(dim=1) - 1)[:, -1:]  # new pos
            new_pos = new_pos.masked_fill(attention_mask[:, -1:] == 0, 0)  # pad pos
            position_ids = torch.cat([position_ids, new_pos], dim=1)  # append pos
            if int(unfinished.max().item()) == 0:  # all done
                break  # stop

        outs: List[str] = []  # outs
        for i in range(int(bsz)):  # decode
            gen_ids = input_ids[i, int(prompt_lens[i]):].tolist()  # gen ids
            if int(self.eos_id) in gen_ids:  # cut eos
                gen_ids = gen_ids[:gen_ids.index(int(self.eos_id))]  # cut
            gen_text = self.tok.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)  # decode
            outs.append((str(prompts[i]) + str(gen_text)).replace(" ", "").upper())  # normalize
        return outs  # return

def main() -> None:  # main
    ap = argparse.ArgumentParser("AMPTailor (base+alpha*(expert-anti)) AA-only generator")  # parser
    ap.add_argument("--base_model_dir", required=True, type=str)  # base
    ap.add_argument("--expert_model_dir", required=True, type=str)  # expert
    ap.add_argument("--anti_model_dir", required=True, type=str)  # anti
    ap.add_argument("--tokenizer_dir", required=True, type=str)  # tok
    ap.add_argument("--train_eos_file", required=True, nargs="+", type=str)  # eos files
    ap.add_argument("--train_weights", type=float, nargs="*", default=None)  # weights
    ap.add_argument("--out_dir", required=True, type=str)  # out
    ap.add_argument("--header_prefix", default="dexp", type=str)  # header
    ap.add_argument("--nsamples", type=int, default=10000)  # n
    ap.add_argument("--batch_size", type=int, default=256)  # bs
    ap.add_argument("--max_new_tokens", type=int, default=256)  # cap
    ap.add_argument("--min_new_tokens", type=int, default=0)  # min
    ap.add_argument("--temperature", type=float, default=1.05)  # temp
    ap.add_argument("--top_k", type=int, default=10)  # topk
    ap.add_argument("--top_p", type=float, default=1.0)  # topp
    ap.add_argument("--keep_eos_after_filter", action="store_true")  # keep eos
    ap.add_argument("--alpha", type=float, default=1.5)  # alpha
    ap.add_argument("--alpha_eos", type=float, default=None)  # alpha eos
    ap.add_argument("--seed", type=int, default=42)  # seed
    ap.add_argument("--fp16", action="store_true")  # fp16
    ap.add_argument("--bf16", action="store_true")  # bf16
    args = ap.parse_args()  # parse

    set_seed(int(args.seed))  # seed
    out_dir = Path(args.out_dir)  # out path
    out_dir.mkdir(parents=True, exist_ok=True)  # mkdir

    if args.train_weights is not None and len(args.train_weights) == 0:  # empty weights
        args.train_weights = None  # reset

    letters, weights = read_first_aa_distribution(list(args.train_eos_file), args.train_weights)  # dist
    preview = "".join(random.choices(letters, weights=weights, k=40))  # preview
    print("[prompt] first-letter dist preview(40) =", preview, flush=True)  # log

    gen = AMPTailorGeneratorAAOnly(base_dir=args.base_model_dir, expert_dir=args.expert_model_dir, anti_dir=args.anti_model_dir, tok_dir=args.tokenizer_dir, seed=int(args.seed), fp16=bool(args.fp16), bf16=bool(args.bf16))  # init

    all_out: List[str] = []  # outputs
    while len(all_out) < int(args.nsamples):  # loop
        cur = int(min(int(args.batch_size), int(args.nsamples) - len(all_out)))  # cur
        prompts = random.choices(letters, weights=weights, k=int(cur))  # prompts
        outs = gen.generate_batch(prompts=prompts, max_new_tokens=int(args.max_new_tokens), min_new_tokens=int(args.min_new_tokens), alpha=float(args.alpha), alpha_eos=float(args.alpha_eos) if args.alpha_eos is not None else None, temperature=float(args.temperature), top_k=int(args.top_k), top_p=float(args.top_p), keep_eos_after_filter=bool(args.keep_eos_after_filter))  # gen
        all_out.extend(list(outs))  # extend
        print(f"[gen] collected={len(all_out)}/{int(args.nsamples)}", flush=True)  # progress

    tag = f"a{float(args.alpha)}_ae{float(args.alpha_eos) if args.alpha_eos is not None else float(args.alpha)}_t{float(args.temperature)}_p{float(args.top_p)}_k{int(args.top_k)}_{int(args.nsamples)}"  # tag
    out_fa = out_dir / f"amptailor_{tag}.fasta"  # fasta path
    write_fasta(out_fa, all_out[:int(args.nsamples)], str(args.header_prefix))  # write

    st = summarize_seqs(all_out[:int(args.nsamples)])  # stats
    out_json = out_dir / f"amptailor_{tag}.summary.json"  # json path
    out_json.write_text(json.dumps(st, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # write json

    print(f"\n[summary] {out_fa.name}", flush=True)  # header
    for k in ["n","bad_AA","bad_AA_%","short<8_%","len_mean","len_median","len_p5","len_p95","uniq_ratio","net_charge_proxy"]:  # keys
        print(f"  - {k}: {st[k]}", flush=True)  # print
    print("[OK] out_fasta =", str(out_fa), flush=True)  # done
    print("[OK] out_summary_json =", str(out_json), flush=True)  # done


def _apply_repetition_penalty_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_lens: List[int],
    penalty: float,
    window: int,
    allowed_mask: torch.Tensor,
) -> None:
    if float(penalty) == 1.0:
        return
    bsz, seq_len = input_ids.shape
    for b in range(bsz):
        pl = int(prompt_lens[b])
        start = max(pl, seq_len - int(window))
        ids = [int(x) for x in input_ids[b, start:seq_len].tolist()]
        if not ids:
            continue
        row = logits[b]
        for tid in set(ids):
            if row[tid] > 0:
                row[tid] = row[tid] / float(penalty)
            else:
                row[tid] = row[tid] * float(penalty)


def _apply_homopolymer_cap_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_lens: List[int],
    max_run: int,
    eos_id: int,
    allowed_mask: torch.Tensor,
) -> None:
    _ = (eos_id, allowed_mask)
    m = int(max_run)
    if m <= 0:
        return
    bsz, seq_len = input_ids.shape
    for b in range(bsz):
        pl = int(prompt_lens[b])
        gen_ids = [int(x) for x in input_ids[b, pl:seq_len].tolist()]
        if not gen_ids:
            continue
        last = int(gen_ids[-1])
        run = 1
        for tid in reversed(gen_ids[:-1]):
            if int(tid) == last:
                run += 1
            else:
                break
        if run >= m:
            logits[b, last] = -float("inf")


def _apply_no_repeat_ngram_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_lens: List[int],
    ngram_size: int,
    eos_id: int,
    allowed_mask: torch.Tensor,
) -> None:
    _ = (eos_id, allowed_mask)
    n = int(ngram_size)
    if n <= 0:
        return
    bsz, seq_len = input_ids.shape
    for b in range(bsz):
        pl = int(prompt_lens[b])
        gen_ids = [int(x) for x in input_ids[b, pl:seq_len].tolist()]
        if len(gen_ids) < max(1, n - 1):
            continue
        banned = set()
        if n == 1:
            banned.update(gen_ids)
        else:
            prefix = tuple(gen_ids[-(n - 1) :])
            for i in range(0, len(gen_ids) - n + 1):
                if tuple(gen_ids[i : i + n - 1]) == prefix:
                    banned.add(int(gen_ids[i + n - 1]))
        for tid in banned:
            logits[b, int(tid)] = -float("inf")


if __name__ == "__main__":  # entry
    main()  # run
