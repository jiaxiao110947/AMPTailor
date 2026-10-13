#!/usr/bin/env python3  # shebang  # 
# -*- coding: utf-8 -*-  # encoding  # 
from __future__ import annotations  # defer annotation eval (SmallEvalSet forward refs)  # 

import sys as _eos_sys
from pathlib import Path as _EosPath
_eos_src = str(_EosPath(__file__).resolve().parents[1])
if _eos_src not in _eos_sys.path:
    _eos_sys.path.insert(0, _eos_src)
from eos_compat import load_eos_tokenizer, EOS_TEXT, normalize_eos_text
# Stop-v3 trainer: streaming dataset + unified wandb metrics + safe AMPEP/AMPsorter callback.  # desc  # 
import os  # os  # 
import json  # json  # 
import math  # math  # 
import argparse  # argparse  # 
import inspect  # inspect  # 
import hashlib  # hashlib  # 
from pathlib import Path  # Path  # 
from typing import Dict, List, Optional  # typing  # 
import numpy as np  # offset index  # 
import torch  # torch  # 
from torch.utils.data import IterableDataset  # IterableDataset  # 
from torch.utils.data import Dataset  # Dataset  # 
from transformers import GPT2LMHeadModel  # model  # 
from transformers import GPT2TokenizerFast  # tokenizer  # 
from transformers import Trainer  # Trainer  # 
from transformers import TrainingArguments  # TrainingArguments  # 
from transformers import set_seed  # set_seed  # 
from transformers import EarlyStoppingCallback  # early stop  # 
from transformers.trainer_callback import TrainerCallback  # callback  # 
from packaging import version as pkg_version  # version compare for resume guard  # 
try:
    from ampsorter_eval_callback import AmpSorterEvalCallback  # optional, not shipped
except Exception:
    AmpSorterEvalCallback = None


def _allow_resume_torch_load() -> None:  # bypass bad torch dist-metadata vs runtime version  # 
    try:  # try  # 
        tv = pkg_version.parse(str(torch.__version__).split("+")[0])  # runtime  # 
        if tv < pkg_version.parse("2.6"):  # actually unsafe  # 
            return  # keep guard  # 
        noop = lambda: None  # noqa: E731  # 
        import transformers.utils.import_utils as _iu  # lazy  # 
        import transformers.trainer as _tr  # trainer binds check at import  # 
        _iu.check_torch_load_is_safe = noop  # noqa  # 
        _tr.check_torch_load_is_safe = noop  # noqa  # 
    except Exception:  # ignore  # 
        pass  # noop  # 
try:  # try import external callback  # 
    from ampep_eval_callback import AMPEPAndSorterCallback  # ampep callback  # 
except Exception:  # if missing  # 
    AMPEPAndSorterCallback = None  # fallback  # 
try:  # length eval callback  # 
    from length_eval_callback import LengthEvalCallback  # length eval  # 
except Exception:  # if missing  # 
    LengthEvalCallback = None  # fallback  # 
try:  # ampsorter-only eval callback  # 
    from ampsorter_only_eval_callback import AmpSorterOnlyEvalCallback  # sorter only  # 
except Exception:  # if missing  # 
    AmpSorterOnlyEvalCallback = None  # fallback  # 

UNLIMITED_MAX_STEPS = 999_999_999  # practical cap when max_steps<=0  # 

# -----------------------------  # section  # 
# 0) Utils  # section  # 
# -----------------------------  # section  # 
def _hash_to_unit_interval(s: str, seed: int) -> float:  # stable hash -> [0,1)  # 
    h = hashlib.md5((str(seed) + "|" + s).encode("utf-8")).hexdigest()  # md5  # 
    v = int(h[:12], 16)  # take 48 bits  # 
    return (v / float(1 << 48))  # scale  # 

def _maybe_add_eos(s: str, ensure_eos: bool) -> str:  # append eos  # 
    x = normalize_eos_text(s).strip()  # strip  #
    if not x:  # empty  # 
        return ""  # empty  # 
    if ensure_eos and (not x.endswith(EOS_TEXT)):  # need eos  # 
        x = x + EOS_TEXT  # add eos  # 
    return x  # return  # 

def _safe_exp(x: float) -> float:  # safe exp  # 
    try:  # try  # 
        return float(math.exp(float(x)))  # exp  # 
    except Exception:  # overflow  # 
        return float("inf")  # inf  # 

def resolve_max_steps(max_steps: int) -> int:  # max_steps<=0 => train until early stop  # 
    ms = int(max_steps)  # int  # 
    if ms <= 0:  # unlimited  # 
        return int(UNLIMITED_MAX_STEPS)  # cap  # 
    return ms  # return  # 

def export_global_best_model(  # save best checkpoint + final val/test eval  # 
    trainer: Trainer,  # trainer  # 
    tok: GPT2TokenizerFast,  # tokenizer  # 
    outdir: Path,  # output root  # 
    val_ds: SmallEvalSet,  # val  # 
    test_ds: SmallEvalSet,  # test  # 
) -> Path:  # return best dir  # 
    best_dir = outdir / "best_model"  # canonical best  # 
    best_dir.mkdir(parents=True, exist_ok=True)  # mkdir  # 
    state = trainer.state  # state  # 
    best_ckpt = getattr(state, "best_model_checkpoint", None)  # ckpt path  # 
    best_metric = getattr(state, "best_metric", None)  # best eval_loss  # 
    print(  # log  # 
        f"[BestModel] checkpoint={best_ckpt} best_eval_loss={best_metric}",  # head  # 
        flush=True,  # flush  # 
    )  # end  # 
    val_metrics = trainer.evaluate(eval_dataset=val_ds)  # global val  # 
    test_metrics = trainer.evaluate(eval_dataset=test_ds)  # global test  # 
    print("VAL (best_model):", val_metrics, flush=True)  # val  # 
    print("TEST(best_model):", test_metrics, flush=True)  # test  # 
    trainer.save_model(str(best_dir))  # weights  # 
    tok.save_pretrained(str(best_dir))  # tokenizer  # 
    summary = {  # summary  # 
        "metric_for_best_model": "eval_loss",  # metric  # 
        "greater_is_better": False,  # lower is better  # 
        "best_eval_loss": float(best_metric) if best_metric is not None else None,  # best  # 
        "best_model_checkpoint": str(best_ckpt) if best_ckpt else None,  # ckpt  # 
        "global_step_at_end": int(state.global_step),  # step  # 
        "val": {k: float(v) for k, v in val_metrics.items()},  # val  # 
        "test": {k: float(v) for k, v in test_metrics.items()},  # test  # 
    }  # end summary  # 
    summary_path = best_dir / "best_eval_summary.json"  # path  # 
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")  # write  # 
    print(f"[OK] global best_model -> {best_dir} (summary: {summary_path})", flush=True)  # ok  # 
    return best_dir  # return  # 

def run_final_heavy_evaluations(  # one full AMPsorter + length eval on best weights  # 
    trainer: Trainer,  # trainer  # 
    tok: GPT2TokenizerFast,  # tok  # 
    targs: TrainingArguments,  # args  # 
    *,  # kw  # 
    ampsorter_enable: bool,  # sorter  # 
    length_enable: bool,  # length  # 
) -> None:  # none  # 
    os.environ["HEAVY_EVAL_FORCE"] = "1"  # force  # 
    final_n = os.environ.get("AMPSORTER_NSAMPLES_FINAL", "").strip()  # final sample count  # 
    old_n = os.environ.get("AMPSORTER_NSAMPLES", "")  # backup  # 
    if final_n:  # override for final eval  # 
        os.environ["AMPSORTER_NSAMPLES"] = final_n  # set  # 
    state = trainer.state  # state  # 
    step = int(state.global_step)  # step  # 
    kw = {"model": trainer.model, "tokenizer": tok}  # kwargs  # 
    print(f"[FinalHeavyEval] step={step} full generation eval (post-train)", flush=True)  # log  # 
    if ampsorter_enable and AmpSorterOnlyEvalCallback is not None:  # sorter  # 
        try:  # try  # 
            cb = AmpSorterOnlyEvalCallback(tokenizer=tok, enabled=True)  # cb  # 
            cb.last_step = -1  # reset guard  # 
            cb.on_evaluate(targs, state, None, metrics=None, **kw)  # run  # 
        except Exception as e:  # fail-open  # 
            print(f"[WARN] final AMPsorter eval failed: {e}", flush=True)  # warn  # 
    if length_enable and LengthEvalCallback is not None:  # length  # 
        try:  # try  # 
            cb = LengthEvalCallback(tokenizer=tok, enabled=True)  # cb  # 
            cb.last_step = -1  # reset  # 
            cb.on_evaluate(targs, state, None, metrics=None, **kw)  # run  # 
        except Exception as e:  # fail-open  # 
            print(f"[WARN] final length eval failed: {e}", flush=True)  # warn  # 
    os.environ.pop("HEAVY_EVAL_FORCE", None)  # cleanup  # 
    if final_n:  # restore  # 
        if old_n:  # had  # 
            os.environ["AMPSORTER_NSAMPLES"] = old_n  # restore  # 
        else:  # none  # 
            os.environ.pop("AMPSORTER_NSAMPLES", None)  # pop  # 

# -----------------------------  # section  # 
# 1) Dataset  # section  # 
# -----------------------------  # section  # 
class LineStreamDataset(IterableDataset):  # streaming dataset  # 
    def __init__(self, data_file: str, split: str, *, seed: int, val_ratio: float, test_ratio: float, ensure_eos: bool):  # init  # 
        super().__init__()  # super  # 
        self.data_file = str(data_file)  # file  # 
        self.split = str(split)  # split  # 
        self.seed = int(seed)  # seed  # 
        self.val_ratio = float(val_ratio)  # val  # 
        self.test_ratio = float(test_ratio)  # test  # 
        self.ensure_eos = bool(ensure_eos)  # eos  # 
    def _assign(self, line: str) -> str:  # assign split  # 
        u = _hash_to_unit_interval(line, self.seed)  # u  # 
        if u < self.test_ratio:  # test  # 
            return "test"  # test  # 
        if u < (self.test_ratio + self.val_ratio):  # val  # 
            return "val"  # val  # 
        return "train"  # train  # 
    def __iter__(self):  # iter  # 
        with open(self.data_file, "r", encoding="utf-8") as f:  # open  # 
            for ln in f:  # loop  # 
                s = _maybe_add_eos(ln, self.ensure_eos)  # eos  # 
                if not s:  # empty  # 
                    continue  # skip  # 
                sp = self._assign(s)  # split  # 
                if sp != self.split:  # not target  # 
                    continue  # skip  # 
                yield {"text": s}  # yield  # 

class SmallEvalSet(Dataset):  # small eval set  # 
    def __init__(self, data_file: str, split: str, *, seed: int, val_ratio: float, test_ratio: float, ensure_eos: bool, max_samples: int):  # init  # 
        self.items: List[Dict[str, str]] = []  # items  # 
        ds = LineStreamDataset(data_file, split, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio, ensure_eos=ensure_eos)  # stream  # 
        for _i, ex in enumerate(ds):  # loop  # 
            self.items.append(ex)  # add  # 
            if len(self.items) >= int(max_samples):  # cap  # 
                break  # stop  # 
    def __len__(self):  # len  # 
        return len(self.items)  # len  # 
    def __getitem__(self, i):  # get  # 
        return self.items[i]  # item  # 


class IndexedLineDataset(Dataset):  # random-access by byte offset  # 
    def __init__(self, data_file: str, offsets_path: str, *, ensure_eos: bool):  # init  # 
        self.data_file = str(data_file)  # file  # 
        self.offsets = np.load(str(offsets_path), mmap_mode="r")  # mmap  # 
        self.ensure_eos = bool(ensure_eos)  # eos  # 
        self._fh = None  # per-worker file handle  # 
    def __len__(self) -> int:  # len  # 
        return int(self.offsets.shape[0])  # n  # 
    def _readline_at(self, off: int) -> str:  # read one line  # 
        if self._fh is None:  # lazy open (one handle per DataLoader worker)  # 
            self._fh = open(self.data_file, "r", encoding="utf-8")  # noqa: SIM115  # 
        self._fh.seek(off)  # seek  # 
        return self._fh.readline()  # line  # 
    def __getitem__(self, idx: int) -> Dict[str, str]:  # get  # 
        off = int(self.offsets[int(idx)])  # offset  # 
        s = _maybe_add_eos(self._readline_at(off), self.ensure_eos)  # eos  # 
        if not s:  # guard  # 
            s = "A" + EOS_TEXT  # fallback  # 
        return {"text": s}  # item  # 


def _load_indexed_meta(index_dir: Path) -> Optional[dict]:  # meta  # 
    meta_path = index_dir / "meta.json"  # path  # 
    if not meta_path.is_file():  # missing  # 
        return None  # none  # 
    return json.loads(meta_path.read_text(encoding="utf-8"))  # load  # 


def build_train_val_test_datasets(a) -> tuple:  # datasets  # 
    indexed_dir = getattr(a, "indexed_splits_dir", None)  # optional  # 
    if indexed_dir:  # indexed mode  # 
        index_dir = Path(str(indexed_dir))  # path  # 
        meta = _load_indexed_meta(index_dir)  # meta  # 
        if meta is None:  # bad  # 
            raise FileNotFoundError(f"indexed meta missing: {index_dir / 'meta.json'}")  # err  # 
        data_file = str(meta.get("data_file") or a.data_file)  # source  # 
        eos = bool(meta.get("ensure_eos", a.ensure_eos))  # eos  # 
        train_ds = IndexedLineDataset(data_file, str(index_dir / "train_offsets.npy"), ensure_eos=eos)  # train  # 
        val_ds = IndexedLineDataset(data_file, str(index_dir / "val_offsets.npy"), ensure_eos=eos)  # val  # 
        test_ds = IndexedLineDataset(data_file, str(index_dir / "test_offsets.npy"), ensure_eos=eos)  # test  # 
        print(  # log  # 
            f"[DATA] indexed splits: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
            f"dir={index_dir}",
            flush=True,
        )
        return train_ds, val_ds, test_ds  # return  # 
    train_ds = LineStreamDataset(  # stream fallback  # 
        str(a.data_file), "train", seed=int(a.seed), val_ratio=float(a.val_ratio),
        test_ratio=float(a.test_ratio), ensure_eos=bool(a.ensure_eos),
    )
    val_ds = SmallEvalSet(  # val  # 
        str(a.data_file), "val", seed=int(a.seed), val_ratio=float(a.val_ratio),
        test_ratio=float(a.test_ratio), ensure_eos=bool(a.ensure_eos), max_samples=int(a.eval_max_samples),
    )
    test_ds = SmallEvalSet(  # test  # 
        str(a.data_file), "test", seed=int(a.seed), val_ratio=float(a.val_ratio),
        test_ratio=float(a.test_ratio), ensure_eos=bool(a.ensure_eos), max_samples=int(a.eval_max_samples),
    )
    print("[DATA] streaming IterableDataset (train); consider --indexed_splits_dir", flush=True)  # warn  # 
    return train_ds, val_ds, test_ds  # return  # 

# -----------------------------  # section  # 
# 2) Collator  # section  # 
# -----------------------------  # section  # 
class CollatorWithPos:  # collator  # 
    def __init__(self, tok: GPT2TokenizerFast, pad_to_multiple_of: int = 8, max_length: int = 512):  # init  # 
        self.tok = tok  # tok  # 
        self.pad_to_multiple_of = int(pad_to_multiple_of)  # pad multiple  # 
        self.max_length = int(max_length)  # max length  # 
    def __call__(self, features: List[Dict[str, str]]):  # call  # 
        texts = [f.get("text", "") for f in features]  # safe get text  # 
        texts = [t for t in texts if isinstance(t, str) and t.strip() != ""]  # filter  # 
        if not texts:  # guard  # 
            texts = ["A" + EOS_TEXT]  # fallback  # 
        enc = self.tok(  # tokenize  # 
            texts,  # batch  # 
            truncation=True,  # trunc  # 
            max_length=self.max_length,  # max len  # 
            padding=True,  # pad  # 
            pad_to_multiple_of=self.pad_to_multiple_of,  # align  # 
            return_tensors="pt",  # tensors  # 
            return_attention_mask=True,  # mask  # 
        )  # end  # 
        ids = enc["input_ids"]  # ids  # 
        attn = enc["attention_mask"]  # attn  # 
        labels = ids.clone()  # labels  # 
        labels[attn == 0] = -100  # ignore pad  # 
        enc["labels"] = labels  # set labels  # 
        pos = (attn.long().cumsum(-1) - 1).clamp(min=0)  # pos  # 
        pos[attn == 0] = 0  # pad pos  # 
        enc["position_ids"] = pos  # set pos  # 
        return enc  # return  # 

# -----------------------------  # section  # 
# 3) Unified W&B logging (single source of truth)  # section  # 
# -----------------------------  # section  # 
class UnifiedWandbCallback(TrainerCallback):  # unified wandb  # 
    def __init__(self, enabled: bool):  # init  # 
        self.enabled = bool(enabled)  # enabled  # 
        self._wandb = None  # wandb  # 
        self._last_step = -1  # last step  # 
        self._last_lr: Optional[float] = None  # last lr  # 
        if self.enabled:  # if enabled  # 
            try:  # try  # 
                import wandb  # wandb  # 
                self._wandb = wandb  # set  # 
            except Exception:  # fail  # 
                self.enabled = False  # disable  # 
                self._wandb = None  # none  # 
    def on_train_begin(self, args, state, control, **kwargs):  # begin  # 
        if (not self.enabled) or (self._wandb is None):  # guard  # 
            return  # skip  # 
        if getattr(self._wandb, "run", None) is None:  # init if needed  # 
            try:  # try  # 
                self._wandb.init(  # init  # 
                    project=os.environ.get("WANDB_PROJECT", None),  # project  # 
                    group=os.environ.get("WANDB_GROUP", None),  # group  # 
                    name=os.environ.get("WANDB_RUN_NAME", None),  # name  # 
                    config={k: getattr(args, k) for k in ["learning_rate", "warmup_ratio", "lr_scheduler_type", "weight_decay"] if hasattr(args, k)},  # config  # 
                )  # end init  # 
            except Exception:  # ignore  # 
                pass  # noop  # 
        try:  # define metrics  # 
            self._wandb.define_metric("trainer/global_step")  # step  # 
            self._wandb.define_metric("train/*", step_metric="trainer/global_step")  # train  # 
            self._wandb.define_metric("eval/*", step_metric="trainer/global_step")  # eval  # 
            self._wandb.define_metric("ampep/*", step_metric="trainer/global_step")  # ampep  # 
            self._wandb.define_metric("ampsorter/*", step_metric="trainer/global_step")  # sorter  # 
            self._wandb.define_metric("length/*", step_metric="trainer/global_step")  # length  # 
        except Exception:  # ignore  # 
            pass  # noop  # 
    def _safe_step(self, step: int) -> int:  # safe step  # 
        step_i = int(step)  # int  # 
        if step_i < self._last_step:  # fix non-monotonic  # 
            step_i = self._last_step  # clamp  # 
        self._last_step = step_i  # update  # 
        return step_i  # return  # 
    def on_log(self, args, state, control, logs=None, **kwargs):  # train log  # 
        if not logs:  # empty  # 
            return  # skip  # 
        step = self._safe_step(int(state.global_step))  # step  # 
        out = {"trainer/global_step": step}  # base  # 
        if logs.get("loss") is not None:  # loss  # 
            out["train/loss"] = float(logs["loss"])  # train loss  # 
            out["train/ppl"] = _safe_exp(float(logs["loss"]))  # train ppl  # 
        if logs.get("learning_rate") is not None:  # lr  # 
            self._last_lr = float(logs["learning_rate"])  # cache lr  # 
            out["train/lr"] = float(logs["learning_rate"])  # train lr  # 
        elif self._last_lr is not None:  # fallback lr  # 
            out["train/lr"] = float(self._last_lr)  # lr  # 
        if logs.get("grad_norm") is not None:  # grad  # 
            out["train/grad_norm"] = float(logs["grad_norm"])  # grad  # 
        if logs.get("epoch") is not None:  # epoch  # 
            out["trainer/epoch"] = float(logs["epoch"])  # epoch  # 
        print(f"[Train] step={step} "  # print head  # 
              f"loss={out.get('train/loss', float('nan')):.6f} "  # loss  # 
              f"ppl={out.get('train/ppl', float('nan')):.3f} "  # ppl  # 
              f"lr={out.get('train/lr', float('nan')):.6g} "  # lr  # 
              f"gn={out.get('train/grad_norm', float('nan')):.3f}", flush=True)  # print  # 
        if self.enabled and (self._wandb is not None):  # wandb  # 
            try:  # try  # 
                self._wandb.log(out, step=step)  # log  # 
            except Exception:  # ignore  # 
                pass  # noop  # 
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # eval log  # 
        if not metrics:  # empty  # 
            return  # skip  # 
        step = self._safe_step(int(state.global_step))  # step  # 
        out = {"trainer/global_step": step}  # base  # 
        if metrics.get("eval_loss") is not None:  # loss  # 
            out["eval/loss"] = float(metrics["eval_loss"])  # eval loss  # 
            out["eval/ppl"] = _safe_exp(float(metrics["eval_loss"]))  # eval ppl  # 
        if self._last_lr is not None:  # lr  # 
            out["eval/lr"] = float(self._last_lr)  # lr  # 
        if metrics.get("eval_runtime") is not None:  # runtime  # 
            out["eval/runtime"] = float(metrics["eval_runtime"])  # runtime  # 
        if metrics.get("eval_samples_per_second") is not None:  # sps  # 
            out["eval/sps"] = float(metrics["eval_samples_per_second"])  # sps  # 
        if metrics.get("eval_steps_per_second") is not None:  # stepsps  # 
            out["eval/stepsps"] = float(metrics["eval_steps_per_second"])  # stepsps  # 
        print(f"[Eval ] step={step} "  # print head  # 
              f"loss={out.get('eval/loss', float('nan')):.6f} "  # loss  # 
              f"ppl={out.get('eval/ppl', float('nan')):.3f}", flush=True)  # print  # 
        if self.enabled and (self._wandb is not None):  # wandb  # 
            try:  # try  # 
                self._wandb.log(out, step=step)  # log  # 
            except Exception:  # ignore  # 
                pass  # noop  # 

class EvalLossTrendCallback(TrainerCallback):  # eval loss trend for early-stop decisions  # 
    def __init__(self, output_dir: Path, window: int = 6, enabled: bool = True):  # init  # 
        self.output_dir = Path(output_dir)  # out  # 
        self.window = max(2, int(window))  # window  # 
        self.enabled = bool(enabled)  # enabled  # 
        self.history: List[tuple] = []  # (step, loss)  # 
        self._wandb = None  # wandb  # 
        try:  # try  # 
            import wandb  # wandb  # 
            self._wandb = wandb  # set  # 
        except Exception:  # ignore  # 
            self._wandb = None  # none  # 
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # eval  # 
        if not self.enabled or not metrics:  # guard  # 
            return  # skip  # 
        loss = metrics.get("eval_loss")  # loss  # 
        if loss is None:  # missing  # 
            return  # skip  # 
        step = int(state.global_step)  # step  # 
        loss_f = float(loss)  # float  # 
        self.history.append((step, loss_f))  # append  # 
        hist_path = self.output_dir / "eval_loss_history.tsv"  # path  # 
        with hist_path.open("a", encoding="utf-8") as f:  # append  # 
            if hist_path.stat().st_size == 0:  # header  # 
                f.write("step\teval_loss\n")  # header  # 
            f.write(f"{step}\t{loss_f:.8f}\n")  # row  # 
        recent = self.history[-self.window :]  # recent  # 
        losses = [x[1] for x in recent]  # losses  # 
        monotonic_down = all(losses[i] < losses[i - 1] for i in range(1, len(losses)))  # strict down  # 
        best_step, best_loss = min(self.history, key=lambda x: x[1])  # best  # 
        prev = losses[-2] if len(losses) >= 2 else loss_f  # prev  # 
        delta = prev - loss_f  # delta  # 
        trend = "down" if loss_f < prev else ("flat" if abs(delta) < 1e-9 else "up")  # trend  # 
        print(  # print  # 
            f"[EvalTrend] step={step} loss={loss_f:.6f} ({trend}, d={delta:+.6f}) "  # head  # 
            f"window={len(losses)}/{self.window} monotonic_down={monotonic_down} "  # window  # 
            f"best=({best_step},{best_loss:.6f}) history={hist_path}",  # best  # 
            flush=True,  # flush  # 
        )  # end  # 
        if self._wandb is not None and getattr(self._wandb, "run", None) is not None:  # wandb  # 
            try:  # try  # 
                self._wandb.log(  # log  # 
                    {  # dict  # 
                        "eval/trend_monotonic_down": float(monotonic_down),  # mono  # 
                        "eval/window_min_loss": float(min(losses)),  # min  # 
                        "eval/best_loss": float(best_loss),  # best  # 
                    },  # end  # 
                    step=step,  # step  # 
                )  # end log  # 
            except Exception:  # ignore  # 
                pass  # noop  # 

# -----------------------------  # section  # 
# 4) Safe wrapper for AMPEPAndSorterCallback (fail-open)  # section  # 
# -----------------------------  # section  # 
class SafeAmpSorterOnlyEvalCallback(TrainerCallback):  # safe ampsorter-only wrapper  # 
    def __init__(self, tokenizer: GPT2TokenizerFast, enabled: bool = True):  # init  # 
        self.enabled = bool(enabled)  # enabled  # 
        self.tokenizer = tokenizer  # tokenizer  # 
        self._inner = None  # inner  # 
        if self.enabled and (AmpSorterOnlyEvalCallback is not None):  # can build  # 
            try:  # try  # 
                self._inner = AmpSorterOnlyEvalCallback(tokenizer=self.tokenizer, enabled=True)  # inner  # 
            except Exception:  # fail  # 
                self._inner = None  # none  # 
                self.enabled = False  # disable  # 
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # evaluate  # 
        if (not self.enabled) or (self._inner is None):  # guard  # 
            return control  # return  # 
        try:  # try  # 
            return self._inner.on_evaluate(args, state, control, metrics=metrics, **kwargs)  # delegate  # 
        except Exception as e:  # fail-open  # 
            print(f"[WARN] AMPsorter eval callback failed at step={int(state.global_step)}: {e}", flush=True)  # warn  # 
            return control  # continue  # 

class SafeLengthEvalCallback(TrainerCallback):  # safe length eval wrapper  # 
    def __init__(self, tokenizer: GPT2TokenizerFast, enabled: bool = True):  # init  # 
        self.enabled = bool(enabled)  # enabled  # 
        self.tokenizer = tokenizer  # tokenizer  # 
        self._inner = None  # inner  # 
        if self.enabled and (LengthEvalCallback is not None):  # can build  # 
            try:  # try  # 
                self._inner = LengthEvalCallback(tokenizer=self.tokenizer, enabled=True)  # inner  # 
            except Exception:  # fail  # 
                self._inner = None  # none  # 
                self.enabled = False  # disable  # 
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # evaluate  # 
        if (not self.enabled) or (self._inner is None):  # guard  # 
            return control  # return  # 
        try:  # try  # 
            return self._inner.on_evaluate(args, state, control, metrics=metrics, **kwargs)  # delegate  # 
        except Exception as e:  # fail-open  # 
            print(f"[WARN] Length eval callback failed at step={int(state.global_step)}: {e}", flush=True)  # warn  # 
            return control  # continue  # 

class SafeAMPEPAndSorterCallback(TrainerCallback):  # safe wrapper  # 
    def __init__(self, tokenizer: GPT2TokenizerFast, enabled: bool = True):  # init  # 
        self.enabled = bool(enabled)  # enabled  # 
        self.tokenizer = tokenizer  # tokenizer  # 
        self._inner = None  # inner  # 
        ampep_py = os.environ.get("AMPEP_PY", "").strip()  # ampep python  # 
        if ampep_py:  # if set  # 
            ampep_exe = str(Path(ampep_py).resolve().with_name("ampep"))  # ampep exe  # 
            if Path(ampep_exe).is_file():  # exists  # 
                os.environ["AMPEP_EXE"] = ampep_exe  # export exe  # 
                os.environ["PATH"] = str(Path(ampep_exe).parent) + os.pathsep + os.environ.get("PATH", "")  # prepend PATH  # 
        if self.enabled and (AMPEPAndSorterCallback is not None):  # can build  # 
            try:  # try  # 
                self._inner = AMPEPAndSorterCallback(tokenizer=self.tokenizer, enabled=True)  # inner  # 
            except Exception:  # fail  # 
                self._inner = None  # none  # 
                self.enabled = False  # disable  # 
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # evaluate  # 
        if (not self.enabled) or (self._inner is None):  # guard  # 
            return control  # return  # 
        try:  # try  # 
            return self._inner.on_evaluate(args, state, control, metrics=metrics, **kwargs)  # delegate  # 
        except Exception as e:  # fail-open  # 
            print(f"[WARN] AMPEP/AMPsorter callback failed at step={int(state.global_step)}: {e}", flush=True)  # warn  # 
            return control  # continue  # 

# -----------------------------  # section  # 
# 5) TrainingArguments compatibility  # section  # 
# -----------------------------  # section  # 
def _trainer_tokenizer_kwargs(tok):  # compat  # 
    sig = inspect.signature(Trainer.__init__)  # signature  # 
    if "processing_class" in sig.parameters:  # new  # 
        return {"processing_class": tok}  # new kw  # 
    return {"tokenizer": tok}  # old kw  # 

def _is_train_only(a) -> bool:  # all data for train, no val/test holdout  # 
    return float(getattr(a, "val_ratio", 0.0)) <= 0.0 and float(getattr(a, "test_ratio", 0.0)) <= 0.0  # both zero  # 


def build_training_args(a, outdir: Path) -> TrainingArguments:  # build args  # 
    sig = inspect.signature(TrainingArguments.__init__)  # sig  # 
    eval_key = "eval_strategy" if "eval_strategy" in sig.parameters else "evaluation_strategy"  # compat  # 
    use_bf16 = torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()  # bf16  # 
    use_fp16 = torch.cuda.is_available() and (not use_bf16)  # fp16  # 
    train_only = _is_train_only(a)  # train-only mode  # 
    load_best = bool(a.load_best_model_at_end) and (not train_only)  # cannot load_best without eval  # 
    if train_only:  # no val → no eval_loss early-stop  # 
        extra = {  # extra  # 
            eval_key: "no",  # no trainer eval  # 
            "max_steps": resolve_max_steps(int(a.max_steps)),  # max steps  # 
        }  # end extra  # 
        print("[Train] train_only=1 (val_ratio=test_ratio=0): eval_strategy=no, load_best disabled", flush=True)  # log  # 
    else:  # normal  # 
        extra = {  # extra  # 
            eval_key: "steps",  # eval strategy  # 
            "eval_steps": int(a.eval_steps),  # eval steps  # 
            "max_steps": resolve_max_steps(int(a.max_steps)),  # max steps  # 
        }  # end extra  # 
    return TrainingArguments(  # args  # 
        output_dir=str(outdir),  # out  # 
        overwrite_output_dir=True,  # overwrite  # 
        per_device_train_batch_size=int(a.train_bs),  # train bs  # 
        per_device_eval_batch_size=int(a.eval_bs),  # eval bs  # 
        gradient_accumulation_steps=int(a.grad_accum),  # accum  # 
        learning_rate=float(a.lr),  # lr  # 
        weight_decay=float(a.weight_decay),  # wd  # 
        warmup_ratio=float(a.warmup_ratio),  # warmup  # 
        lr_scheduler_type=str(a.lr_scheduler),  # sched  # 
        logging_steps=int(a.logging_steps),  # log steps  # 
        logging_first_step=True,  # first step  # 
        save_strategy="steps",  # save  # 
        save_steps=int(a.save_steps),  # save steps  # 
        save_total_limit=int(a.save_total_limit),  # keep  # 
        load_best_model_at_end=load_best,  # best  # 
        metric_for_best_model="eval_loss",  # metric  # 
        greater_is_better=False,  # lower better  # 
        report_to=["none"],  # IMPORTANT: avoid duplicate wandb logging  # 
        run_name=str(a.wandb_run_name),  # run name  # 
        bf16=bool(use_bf16),  # bf16  # 
        fp16=bool(use_fp16),  # fp16  # 
        max_grad_norm=float(a.max_grad_norm),  # clip  # 
        optim=str(a.optim),  # optim  # 
        dataloader_num_workers=int(a.num_workers),  # workers  # 
        dataloader_pin_memory=True,  # pin  # 
        ignore_data_skip=True,  # LineStreamDataset: resume skip = full-file scan per batch  # 
        remove_unused_columns=False,  # IMPORTANT: keep "text" for collator  # 
        **extra,  # extra  # 
    )  # end  # 

# -----------------------------  # section  # 
# 6) Main  # section  # 
# -----------------------------  # section  # 
def main():  # main  # 
    p = argparse.ArgumentParser("train_model.py (unified-metrics)")  # parser  # 
    p.add_argument("--data_file", required=True)  # data  # 
    p.add_argument("--indexed_splits_dir", default="", help="dir with train/val/test_offsets.npy from prepare_indexed_splits.py")  # indexed  # 
    p.add_argument("--base_model_dir", required=True)  # base  # 
    p.add_argument("--tokenizer_dir", required=True)  # tok  # 
    p.add_argument("--output_dir", required=True)  # out  # 
    p.add_argument("--max_length", type=int, default=512)  # max len  # 
    p.add_argument("--max_steps", type=int, default=0, help="<=0: no fixed cap (early stop decides); >0: hard cap")  # steps  # 
    p.add_argument("--train_bs", type=int, default=64)  # train bs  # 
    p.add_argument("--eval_bs", type=int, default=64)  # eval bs  # 
    p.add_argument("--grad_accum", type=int, default=2)  # accum  # 
    p.add_argument("--lr", type=float, default=1e-4)  # lr  # 
    p.add_argument("--weight_decay", type=float, default=0.01)  # wd  # 
    p.add_argument("--warmup_ratio", type=float, default=0.02)  # warmup  # 
    p.add_argument("--lr_scheduler", default="cosine")  # sched  # 
    p.add_argument("--logging_steps", type=int, default=20)  # log  # 
    p.add_argument("--eval_steps", type=int, default=4000)  # eval steps  # 
    p.add_argument("--save_steps", type=int, default=4000)  # save steps  # 
    p.add_argument("--save_total_limit", type=int, default=2)  # keep  # 
    p.add_argument("--load_best_model_at_end", action="store_true")  # best  # 
    p.add_argument("--seed", type=int, default=42)  # seed  # 
    p.add_argument("--tf32", action="store_true")  # tf32  # 
    p.add_argument("--optim", default="adamw_torch_fused")  # optim (faster on GPU)  # 
    p.add_argument("--num_workers", type=int, default=2)  # workers  # 
    p.add_argument("--max_grad_norm", type=float, default=1.0)  # clip  # 
    p.add_argument("--early_stop_patience", type=int, default=6)  # early stop evals without improvement  # 
    p.add_argument("--early_stop_min_delta", type=float, default=1e-4)  # min eval_loss improvement  # 
    p.add_argument("--eval_loss_trend_window", type=int, default=6)  # trend window (eval count)  # 
    p.add_argument("--ensure_eos", action="store_true")  # eos  # 
    p.add_argument("--val_ratio", type=float, default=0.001)  # val  # 
    p.add_argument("--test_ratio", type=float, default=0.001)  # test  # 
    p.add_argument("--eval_max_samples", type=int, default=50000)  # eval cap  # 
    p.add_argument("--wandb_enable", action="store_true")  # wandb  # 
    p.add_argument("--wandb_project", default="AMP_Model_Training")  # project  # 
    p.add_argument("--wandb_group", default="uniprot_stop_v3")  # group  # 
    p.add_argument("--wandb_run_name", default="uniprot_stop_v3")  # run name  # 
    p.add_argument("--wandb_mode", default="online")  # mode  # 
    p.add_argument("--ampep_eval_enable", action="store_true")  # ampep eval  # 
    p.add_argument("--length_eval_enable", action="store_true")  # length eval  # 
    p.add_argument("--ampsorter_eval_enable", action="store_true")  # ampsorter eval  # 
    p.add_argument("--final_heavy_eval", action="store_true")  # full gen-eval once at train end  # 
    a = p.parse_args()  # args  # 

    set_seed(int(a.seed))  # seed  # 
    if bool(a.tf32) and torch.cuda.is_available():  # tf32  # 
        torch.backends.cuda.matmul.allow_tf32 = True  # matmul  # 
        torch.backends.cudnn.allow_tf32 = True  # cudnn  # 
        try:  # torch2  # 
            torch.set_float32_matmul_precision("high")  # high  # 
        except Exception:  # ignore  # 
            pass  # noop  # 
        print("[性能] TF32 已启用。", flush=True)  # log  # 

    outdir = Path(str(a.output_dir))  # out  # 
    outdir.mkdir(parents=True, exist_ok=True)  # mkdir  # 
    resolved_max_steps = resolve_max_steps(int(a.max_steps))  # resolved  # 
    unlimited = int(a.max_steps) <= 0  # unlimited flag  # 
    if unlimited:  # log  # 
        print(  # print  # 
            f"[Train] max_steps=unlimited (cap={resolved_max_steps}), stop by eval_loss early stopping",  # msg  # 
            flush=True,  # flush  # 
        )  # end  # 
    if unlimited and str(a.lr_scheduler) == "cosine":  # cosine needs fixed horizon  # 
        print("[Train] unlimited mode: recommend --lr_scheduler constant (cosine will decay too slowly)", flush=True)  # warn  # 

    if bool(a.wandb_enable):  # wandb env  # 
        os.environ["WANDB_PROJECT"] = str(a.wandb_project)  # project  # 
        os.environ["WANDB_GROUP"] = str(a.wandb_group)  # group  # 
        os.environ["WANDB_RUN_NAME"] = str(a.wandb_run_name)  # name  # 
        os.environ["WANDB_MODE"] = str(a.wandb_mode)  # mode  # 

    tok = load_eos_tokenizer(str(a.tokenizer_dir))  # tok  #
    if tok.pad_token is None:  # pad  # 
        tok.pad_token = tok.eos_token  # pad=eos  # 
    tok.padding_side = "left"  # left pad  # 
    print(f"[tokenizer] vocab={tok.vocab_size} eos_id={tok.eos_token_id} pad_id={tok.pad_token_id}", flush=True)  # log  # 

    model = GPT2LMHeadModel.from_pretrained(str(a.base_model_dir))  # model  # 
    model.config.pad_token_id = tok.pad_token_id  # pad id  # 
    model.config.eos_token_id = tok.eos_token_id  # eos id  # 
    model.config.use_cache = False  # training stability  # 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # device  # 
    model.to(device)  # to device  # 

    if str(a.indexed_splits_dir).strip():  # normalize  # 
        a.indexed_splits_dir = str(Path(a.indexed_splits_dir).resolve())  # abs  # 
    else:  # empty  # 
        a.indexed_splits_dir = None  # none  # 
    train_ds, val_ds, test_ds = build_train_val_test_datasets(a)  # datasets  # 
    train_only = _is_train_only(a) or (hasattr(val_ds, "__len__") and len(val_ds) == 0)  # train-only  # 
    if train_only and int(a.max_steps) <= 0:  # need fixed horizon without eval early-stop  # 
        raise SystemExit("[ERR] train_only requires --max_steps > 0 (no eval_loss early stopping)")  # fail  # 

    targs = build_training_args(a, outdir)  # args  # 
    collator = CollatorWithPos(tok, pad_to_multiple_of=8, max_length=int(a.max_length))  # collator  # 

    callbacks: List[TrainerCallback] = []  # callbacks  # 
    callbacks.append(UnifiedWandbCallback(bool(a.wandb_enable)))  # unified logger  # 
    if not train_only:  # trend needs eval_loss  # 
        callbacks.append(  # trend logger  # 
            EvalLossTrendCallback(  # trend  # 
                output_dir=outdir,  # dir  # 
                window=int(a.eval_loss_trend_window),  # window  # 
                enabled=True,  # on  # 
            )  # end  # 
        )  # end append  # 
    if bool(a.ampep_eval_enable):  # ampep enable  # 
        callbacks.append(SafeAMPEPAndSorterCallback(tokenizer=tok, enabled=True))  # safe eval  # 
        print("INFO:__main__:AMPEP evaluation callback attached (safe).", flush=True)  # log  # 
    if bool(a.length_eval_enable):  # length enable  # 
        callbacks.append(SafeLengthEvalCallback(tokenizer=tok, enabled=True))  # length eval  # 
        print("INFO:__main__:Length evaluation callback attached (safe).", flush=True)  # log  # 
    if bool(a.ampsorter_eval_enable):  # ampsorter enable  # 
        callbacks.append(SafeAmpSorterOnlyEvalCallback(tokenizer=tok, enabled=True))  # sorter eval  # 
        print("INFO:__main__:AMPsorter-only evaluation callback attached (safe).", flush=True)  # log  # 
    if (not train_only) and int(a.early_stop_patience) > 0:  # early stop  # 
        callbacks.append(  # add  # 
            EarlyStoppingCallback(  # early stop  # 
                early_stopping_patience=int(a.early_stop_patience),  # patience evals  # 
                early_stopping_threshold=float(a.early_stop_min_delta),  # min delta  # 
            )  # end  # 
        )  # end append  # 
        print(  # log  # 
            f"INFO:__main__:Early stopping on eval_loss: patience={int(a.early_stop_patience)} "  # head  # 
            f"min_delta={float(a.early_stop_min_delta):.2e} (load_best_model_at_end should be on)",  # tail  # 
            flush=True,  # flush  # 
        )  # end  # 
    elif train_only:  # log  # 
        print(f"INFO:__main__:train_only: skip early stopping; max_steps={int(a.max_steps)}", flush=True)  # log  # 

    if (not bool(a.ampsorter_eval_enable)) and AmpSorterEvalCallback is not None:  # optional, not shipped
        try:  # try  # 
            callbacks.append(AmpSorterEvalCallback())  # legacy ampsorter eval  # 
        except Exception as _err:  # except  # 
            print("[WARN] AmpSorterEvalCallback disabled: " + str(_err), flush=True)  # warn  # 
    trainer = Trainer(  # trainer  # 
        model=model,  # model  # 
        args=targs,  # args  # 
        train_dataset=train_ds,  # train  # 
        eval_dataset=None if train_only else val_ds,  # val  # 
        data_collator=collator,  # collator  # 
        callbacks=callbacks,  # callbacks  # 
        **_trainer_tokenizer_kwargs(tok),  # compat  # 
    )  # end  # 

    last_ckpt = None  # ckpt  # 
    force = os.environ.get("FORCE_RESUME_CKPT", "").strip()  # optional override  # 
    if force and Path(force).is_dir():  # forced resume path  # 
        last_ckpt = str(Path(force).resolve())  # use  # 
        print(f"INFO:__main__:FORCE_RESUME_CKPT={last_ckpt}", flush=True)  # log  # 
    else:  # auto  # 
        try:  # try  # 
            from transformers.trainer_utils import get_last_checkpoint  # util  # 
            last_ckpt = get_last_checkpoint(str(outdir))  # last  # 
        except Exception:  # ignore  # 
            last_ckpt = None  # none  # 

    if last_ckpt:  # resume  # 
        ckpt_dir = Path(last_ckpt)  # path  # 
        opt_bak: list[Path] = []  # restored after train if we strip optimizer  # 
        if os.environ.get("RESET_OPTIMIZER_ON_RESUME", "").strip().lower() in ("1", "true", "yes"):  # fresh opt + --lr  # 
            for name in ("optimizer.pt", "scheduler.pt", "scaler.pt"):  # files  # 
                p = ckpt_dir / name  # ckpt file  # 
                if p.is_file():  # exists  # 
                    bak = ckpt_dir / f"{name}.bak_resume"  # sidecar  # 
                    if bak.exists():  # stale  # 
                        bak.unlink()  # rm  # 
                    p.rename(bak)  # move aside  # 
                    opt_bak.append(bak)  # track  # 
            print(  # log  # 
                f"INFO:__main__:RESET_OPTIMIZER_ON_RESUME: stripped optimizer/scheduler in {ckpt_dir}",  # msg  # 
                flush=True,  # flush  # 
            )  # end  # 
        print(f"INFO:__main__:Found checkpoint: {last_ckpt}. Resuming training...", flush=True)  # log  # 
        _allow_resume_torch_load()  # metadata may report torch<2.6 while runtime is 2.6+  # 
        try:  # train  # 
            trainer.train(resume_from_checkpoint=last_ckpt)  # train  # 
        finally:  # restore ckpt dir  # 
            for bak in opt_bak:  # each  # 
                orig = bak.parent / bak.name.replace(".bak_resume", "")  # orig name  # 
                if bak.is_file() and not orig.exists():  # restore  # 
                    bak.rename(orig)  # put back  # 
    else:  # scratch  # 
        print("INFO:__main__:Starting training from scratch...", flush=True)  # log  # 
        trainer.train()  # train  # 

    trainer.state.save_to_json(str(outdir / "trainer_state.json"))  # save state  # 
    if bool(a.load_best_model_at_end) and (not train_only):  # export canonical best  # 
        export_global_best_model(trainer, tok, outdir, val_ds, test_ds)  # best_model/  # 
    if bool(a.final_heavy_eval):  # post-train full gen eval on converged weights  # 
        run_final_heavy_evaluations(  # run  # 
            trainer,  # trainer  # 
            tok,  # tok  # 
            targs,  # args  # 
            ampsorter_enable=bool(a.ampsorter_eval_enable),  # sorter  # 
            length_enable=bool(a.length_eval_enable),  # length  # 
        )  # end  # 
    else:  # fallback: last weights at output root  # 
        if (not train_only) and hasattr(val_ds, "__len__") and len(val_ds) > 0:  # val  # 
            print("VAL :", trainer.evaluate(eval_dataset=val_ds), flush=True)  # val  # 
        if (not train_only) and hasattr(test_ds, "__len__") and len(test_ds) > 0:  # test  # 
            print("TEST:", trainer.evaluate(eval_dataset=test_ds), flush=True)  # test  # 
        trainer.save_model(str(outdir))  # save model  # 
        tok.save_pretrained(str(outdir))  # save tok  # 
        # also write best_model/ as last checkpoint for downstream convenience  # 
        best_dir = outdir / "best_model"  # best  # 
        best_dir.mkdir(parents=True, exist_ok=True)  # mkdir  # 
        trainer.save_model(str(best_dir))  # save  # 
        tok.save_pretrained(str(best_dir))  # tok  # 
        print("[OK] saved (last step, no load_best):", str(outdir), flush=True)  # ok  # 
        print("[OK] best_model (last weights) ->", str(best_dir), flush=True)  # ok  # 

if __name__ == "__main__":  # entry  # 
    main()  # main  # 
