"""Use literal EOS while preserving the end-token ID of existing GPT-2 weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

EOS_TEXT = "EOS"
LEGACY_EOS_TEXT = "<|endoftext|>"


def normalize_eos_text(text: str) -> str:
    """Translate legacy corpus markers without changing peptide residues."""
    return re.sub(re.escape(LEGACY_EOS_TEXT), EOS_TEXT, str(text), flags=re.IGNORECASE)


def migrate_eos_tokenizer(tok):
    """Rename the actual special token, retaining its vocabulary and embedding ID."""
    from tokenizers import Tokenizer

    old_text = tok.eos_token
    old_id = tok.eos_token_id
    if old_text is None or old_id is None:
        raise ValueError("Tokenizer must define an existing EOS token; adding an ID would require model migration")
    if old_text == EOS_TEXT:
        if tok.encode(EOS_TEXT, add_special_tokens=False) != [int(old_id)]:
            raise ValueError("EOS must encode as one existing special token")
        return tok
    vocab = tok.get_vocab()
    if EOS_TEXT in vocab and int(vocab[EOS_TEXT]) != int(old_id):
        raise ValueError("EOS already has a different token ID; refusing to change the meaning of existing model weights")
    payload = json.loads(tok.backend_tokenizer.to_str())
    model_vocab = payload.get("model", {}).get("vocab")
    if not isinstance(model_vocab, dict):
        raise ValueError("EOS migration requires a GPT-2 BPE/WordLevel dictionary vocabulary")
    if old_text in model_vocab:
        if int(model_vocab[old_text]) != int(old_id):
            raise ValueError("Inconsistent original EOS vocabulary ID")
        model_vocab[EOS_TEXT] = model_vocab.pop(old_text)
    if payload["model"].get("unk_token") == old_text:
        payload["model"]["unk_token"] = EOS_TEXT
    changed = old_text in vocab
    for added in payload.get("added_tokens", []):
        if added["content"] == old_text:
            if int(added["id"]) != int(old_id):
                raise ValueError("Inconsistent added EOS token ID")
            added["content"] = EOS_TEXT
            added["special"] = True
            changed = True
    if not changed:
        raise ValueError("Original EOS token was not found in the tokenizer vocabulary")
    # Some tokenizers name EOS/BOS in their processing templates as well.
    def replace_template(value):
        if isinstance(value, dict):
            return {EOS_TEXT if k == old_text else k: replace_template(v) for k, v in value.items()}
        if isinstance(value, list):
            return [replace_template(v) for v in value]
        return EOS_TEXT if value == old_text else value
    if payload.get("post_processor"):
        payload["post_processor"] = replace_template(payload["post_processor"])
    aliases = {name for name in tok.SPECIAL_TOKENS_ATTRIBUTES
               if name != "additional_special_tokens" and getattr(tok, name, None) == old_text}
    extra_specials = list(tok.additional_special_tokens)
    old_size = len(tok)
    tok._tokenizer = Tokenizer.from_str(json.dumps(payload))
    for name in aliases:
        setattr(tok, name, EOS_TEXT)
        tok.init_kwargs[name] = EOS_TEXT
    if old_text in extra_specials:
        tok.additional_special_tokens = [EOS_TEXT if t == old_text else t for t in extra_specials]
        tok.init_kwargs["additional_special_tokens"] = list(tok.additional_special_tokens)
    if len(tok) != old_size or tok.eos_token_id != old_id:
        raise RuntimeError("EOS migration changed vocabulary size or end-token ID")
    if tok.encode(EOS_TEXT, add_special_tokens=False) != [int(old_id)]:
        raise RuntimeError("Migrated EOS is not a single special token")
    return tok


def load_eos_tokenizer(path, **kwargs):
    from transformers import GPT2TokenizerFast

    return migrate_eos_tokenizer(GPT2TokenizerFast.from_pretrained(path, **kwargs))


def main():
    parser = argparse.ArgumentParser(description="Export an EOS tokenizer compatible with existing model weights")
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--output_dir", required=True, help="New, empty tokenizer directory; weights are not modified")
    args = parser.parse_args()
    source = Path(args.tokenizer_dir).resolve()
    out = Path(args.output_dir).resolve()
    if source == out or (out.exists() and any(out.iterdir())):
        raise SystemExit("Use a distinct empty output directory to preserve the original tokenizer")
    tok = load_eos_tokenizer(str(source), local_files_only=True)
    out.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(str(out))
    print(f"Saved EOS tokenizer to {out}; eos_token_id={tok.eos_token_id}; vocabulary_size={len(tok)}")


if __name__ == "__main__":
    main()
