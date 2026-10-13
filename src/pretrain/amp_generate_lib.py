#!/usr/bin/env python3  # shebang
# -*- coding: utf-8 -*-  # encoding
# AMP generation helpers: start-letter stats + FASTA generation.  # desc

import os  # os
import re  # re
import json  # json
import random  # random
from pathlib import Path  # Path
from typing import Dict  # Dict
from typing import List  # List
from typing import Tuple  # Tuple
from typing import Optional  # Optional

import torch  # torch
from transformers import GPT2LMHeadModel  # model
from transformers import GPT2TokenizerFast  # tokenizer

AA20 = "ACDEFGHIKLMNPQRSTVWY"  # aa20
EOS_TEXT = "<|endoftext|>"  # eos text

DEFAULT_N_GEN = int(os.environ.get("AMP_N_GEN", "10000"))  # default n_generate from env (fallback 10000)

def _clean_seq(seq: str) -> str:  # clean seq
    s = seq.strip().upper()  # upper
    s = re.sub(r"[^A-Z]", "", s)  # strip non letters
    s = "".join([c for c in s if c in AA20])  # keep aa20
    return s  # return


def drop_echoed_len_control(text: str) -> str:
    """Remove explicit echoed LEN<number>: prefixes, preserving real LEN residues."""
    return re.sub(r"^(?:LEN[1-9][0-9]*:\s*)+", "", str(text).strip(), flags=re.IGNORECASE)


def extract_peptide_after_length_prompt(text: str, prompt: str) -> str:
    """Extract residues after the prompt and before the first EOS marker."""
    body = str(text).strip()
    p = str(prompt).strip()
    if p and body.upper().startswith(p.upper()):
        body = body[len(p) :]
    body = re.split(re.escape(EOS_TEXT), body, maxsplit=1, flags=re.IGNORECASE)[0]
    return _clean_seq(drop_echoed_len_control(body))

def ensure_tokenizer(tokenizer_dir: str) -> GPT2TokenizerFast:  # ensure tokenizer
    tok = GPT2TokenizerFast.from_pretrained(tokenizer_dir)  # load
    if tok.pad_token is None:  # pad missing
        tok.pad_token = tok.eos_token  # set pad=eos
    tok.padding_side = "left"  # left pad
    return tok  # return

def ensure_model(model_dir: str, device: torch.device, tok: GPT2TokenizerFast) -> GPT2LMHeadModel:  # ensure model
    model = GPT2LMHeadModel.from_pretrained(model_dir)  # load
    model.config.pad_token_id = tok.pad_token_id  # align pad
    model.config.eos_token_id = tok.eos_token_id  # align eos
    model.to(device)  # to device
    model.eval()  # eval mode
    return model  # return

def read_amp_start_letters(amp_start_file: str) -> List[str]:  # read start letters
    p = Path(amp_start_file)  # path
    letters: List[str] = []  # init
    if not p.exists():  # missing
        return letters  # return empty
    with p.open("r", encoding="utf-8") as f:  # open
        for ln in f:  # iter
            s = _clean_seq(ln)  # clean
            if not s:  # empty
                continue  # skip
            letters.append(s[0])  # first aa
    return letters  # return

def get_start_letter_stats(amp_start_file: str) -> Tuple[List[str], Dict[str, float]]:  # stats
    letters = read_amp_start_letters(amp_start_file)  # read
    cnt: Dict[str, int] = {a: 0 for a in AA20}  # init counts
    for c in letters:  # count
        if c in cnt:  # valid
            cnt[c] += 1  # inc
    total = sum(cnt.values())  # total
    if total <= 0:  # none
        top5 = list("SACDE")  # fallback top5
        nat = {a: 1.0 / len(AA20) for a in AA20}  # uniform
        return top5, nat  # return
    top5 = sorted(cnt.keys(), key=lambda k: cnt[k], reverse=True)[:5]  # top5
    nat = {a: (cnt[a] / float(total)) for a in AA20}  # probs
    return top5, nat  # return

def sample_start_letters(mode: str, n: int, top5: List[str], nat: Dict[str, float], seed: int) -> List[str]:  # sample
    rng = random.Random(seed)  # rng
    if mode == "top5":  # top5 mode
        if not top5:  # guard
            top5 = list("SACDE")  # fallback
        return [rng.choice(top5) for _ in range(n)]  # sample uniform among top5
    if mode == "natural":  # natural mode
        keys = list(nat.keys())  # keys
        weights = [float(nat[k]) for k in keys]  # weights
        s = sum(weights)  # sum
        if s <= 0:  # guard
            weights = [1.0 for _ in keys]  # uniform
        return rng.choices(keys, weights=weights, k=n)  # weighted sample
    return [""] * n  # fallback

def _normalize_start_letters(start_letters: List[str], n_generate: int) -> List[str]:  # normalize start letters
    if n_generate <= 0:  # guard
        return []  # return
    if not start_letters:  # empty input
        return [""] * int(n_generate)  # fill with empty (random AA will be used)
    if len(start_letters) >= int(n_generate):  # too many
        return list(start_letters[: int(n_generate)])  # truncate
    pad = [""] * (int(n_generate) - len(start_letters))  # pad list
    return list(start_letters) + pad  # padded

def generate_fasta_from_model(  # generate fasta
    model: GPT2LMHeadModel,  # model
    tok: GPT2TokenizerFast,  # tok
    out_fasta: str,  # out
    start_letters: List[str],  # starts
    *,  # kw
    n_generate: int = DEFAULT_N_GEN,  # number to generate (default 10000)
    max_new_tokens: int = 128,  # max new
    temperature: float = 1.0,  # temp
    top_p: float = 0.95,  # top_p
    top_k: int = 0,  # top_k
    seed: int = 42,  # seed
    max_attempts_multiplier: int = 10,  # attempts multiplier to guarantee enough valid seqs
) -> str:  # end
    device = next(model.parameters()).device  # device
    rng = random.Random(seed)  # rng
    n_generate_int = int(n_generate)  # n_generate int
    if n_generate_int <= 0:  # guard
        raise ValueError(f"n_generate must be > 0, got {n_generate}")  # raise
    starts = _normalize_start_letters(start_letters, n_generate_int)  # normalize starts
    max_attempts = int(n_generate_int) * int(max_attempts_multiplier)  # max attempts
    Path(out_fasta).parent.mkdir(parents=True, exist_ok=True)  # mkdir
    n_written = 0  # written count
    attempts = 0  # attempts
    with open(out_fasta, "w", encoding="utf-8") as w:  # open out
        while (n_written < n_generate_int) and (attempts < max_attempts):  # loop until enough
            st = starts[n_written] if n_written < len(starts) else ""  # select start letter
            prompt = (st if st else rng.choice(AA20))  # prompt
            enc = tok(prompt, return_tensors="pt")  # encode  #
            in_ids = enc.input_ids.to(device)  # input ids  #
            attn = enc.attention_mask.to(device)  # attention mask  #
            with torch.no_grad():  # no grad
                gen = model.generate(  # generate
                    input_ids=in_ids,  # input
                    attention_mask=attn,  # attention mask  #
                    do_sample=True,  # sample
                    temperature=float(temperature),  # temp
                    top_p=float(top_p),  # top_p
                    top_k=int(top_k),  # top_k
                    max_new_tokens=int(max_new_tokens),  # max
                    pad_token_id=tok.pad_token_id,  # pad id
                    eos_token_id=tok.eos_token_id,  # eos id
                )  # end generate
            text = tok.decode(gen[0], skip_special_tokens=True)  # decode
            seq = _clean_seq(text)  # clean aa20
            attempts += 1  # attempts +1
            if not seq:  # empty
                continue  # retry
            w.write(f">seq_{n_written:06d}|start={prompt}\n")  # header
            w.write(seq + "\n")  # seq
            n_written += 1  # written +1
    if n_written < n_generate_int:  # not enough
        raise RuntimeError(f"Failed to generate enough sequences: wrote={n_written}, target={n_generate_int}, attempts={attempts}")  # raise
    return out_fasta  # return


def make_length_prompt(target_len: int) -> str:
    return f"LEN{int(target_len)}:"


def generate_fasta_with_length_targets(  # length-conditioned generation
    model: GPT2LMHeadModel,
    tok: GPT2TokenizerFast,
    out_fasta: str,
    target_lengths: List[int],
    *,
    n_per_length: int = 10,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 0,
    seed: int = 42,
    max_attempts_multiplier: int = 10,
) -> str:
    device = next(model.parameters()).device
    rng = random.Random(seed)
    targets = [int(x) for x in target_lengths if int(x) > 0]
    n_per = int(n_per_length)
    if not targets or n_per <= 0:
        raise ValueError("target_lengths and n_per_length must be non-empty / > 0")
    Path(out_fasta).parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out_fasta, "w", encoding="utf-8") as w:
        for tgt in targets:
            got = 0
            attempts = 0
            max_attempts = n_per * int(max_attempts_multiplier)
            prompt = make_length_prompt(tgt)
            while got < n_per and attempts < max_attempts:
                enc = tok(prompt, return_tensors="pt")
                in_ids = enc.input_ids.to(device)
                attn = enc.attention_mask.to(device)
                with torch.no_grad():
                    gen = model.generate(
                        input_ids=in_ids,
                        attention_mask=attn,
                        do_sample=True,
                        temperature=float(temperature),
                        top_p=float(top_p),
                        top_k=int(top_k),
                        max_new_tokens=int(max_new_tokens),
                        pad_token_id=tok.pad_token_id,
                        eos_token_id=tok.eos_token_id,
                    )
                text = tok.decode(gen[0], skip_special_tokens=True)
                seq = extract_peptide_after_length_prompt(text, prompt)
                attempts += 1
                if not seq:
                    continue
                w.write(f">seq_{written:07d}|LEN={tgt}|prompt={prompt}\n")
                w.write(seq + "\n")
                got += 1
                written += 1
            if got < n_per:
                print(
                    f"[WARN] length={tgt}: only generated {got}/{n_per} sequences",
                    flush=True,
                )
    return out_fasta
