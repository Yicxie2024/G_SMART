#!/usr/bin/env python3

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:  # Optional dependency for 4/8-bit quantization
    from transformers import BitsAndBytesConfig  # type: ignore
except ImportError:  # pragma: no cover - optional path
    BitsAndBytesConfig = None  # type: ignore

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional path
    yaml = None

from huggingface_hub import hf_hub_download

try:
    from luh.auto_uncertainty_head import AutoUncertaintyHead
except ImportError:  # pragma: no cover - optional dependency
    AutoUncertaintyHead = None  # type: ignore


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_UHEAD_REPO = "rediska0123/uhead_Qwen2.5-1.5B-Instruct_6epochs"
DEFAULT_PROMPT = "请简要介绍一下你的训练方式和能力范围。"


@dataclass
class TokenScore:
    token: str
    token_id: int
    logprob: float
    prob: float


def resolve_device(preferred: Optional[str] = None) -> str:
    """Pick the best available device, optionally honoring a user preference."""
    if preferred:
        preferred = preferred.lower()
        if preferred == "cpu":
            return "cpu"
        if preferred == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            print("[Setup] Requested CUDA but no GPU detected; falling back to auto selection.")
        if preferred == "mps":
            if torch.backends.mps.is_available():
                return "mps"
            print("[Setup] Requested MPS but it is unavailable; falling back to auto selection.")
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_quantization_config(mode: Optional[str], compute_dtype: torch.dtype):
    if not mode or mode == "none":
        return None
    if BitsAndBytesConfig is None:
        raise RuntimeError(
            "BitsAndBytes is not available. Install bitsandbytes to use quantized loading."
        )
    mode = mode.lower()
    if mode not in {"4bit", "8bit"}:
        raise ValueError(f"Unsupported quantization mode: {mode}")
    load_in_4bit = mode == "4bit"
    load_in_8bit = mode == "8bit"
    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        llm_int8_has_fp16_weight=True,
    )


def compute_token_scores(
    generation_scores: List[torch.Tensor],
    generated_token_ids: torch.Tensor,
    tokenizer,
) -> List[TokenScore]:
    """Compute per-token probabilities for the generated continuation."""
    token_scores: List[TokenScore] = []
    if torch.is_tensor(generated_token_ids):
        generated_ids = generated_token_ids.detach().cpu().tolist()
    else:
        generated_ids = list(generated_token_ids)

    if len(generation_scores) != len(generated_ids):
        overlap = min(len(generation_scores), len(generated_ids))
        print(
            f"[Scores] Warning: score tensors ({len(generation_scores)}) and generated tokens ({len(generated_ids)}) differ; truncating to {overlap}."
        )
    else:
        overlap = len(generated_ids)

    for step in range(overlap):
        score_tensor = generation_scores[step]
        token_id = int(generated_ids[step])
        log_probs = torch.nn.functional.log_softmax(score_tensor, dim=-1)
        logprob = log_probs[0, token_id].item()
        prob = math.exp(logprob) if math.isfinite(logprob) else float("nan")
        token_text = tokenizer.decode([token_id])
        token_scores.append(TokenScore(token=token_text, token_id=token_id, logprob=logprob, prob=prob))
        if step < 5:  # print a short preview while iterating
            pretty_text = token_text.replace("\n", "\\n")
            print(
                f"[Token {step:02d}] id={token_id:>6} logprob={logprob:7.4f} prob={prob:7.4f} text='{pretty_text}'"
            )
    return token_scores


def summarize_scores(token_scores: List[TokenScore]) -> Dict[str, float]:
    logprob_sum = sum(t.logprob for t in token_scores)
    avg_logprob = logprob_sum / len(token_scores) if token_scores else float("nan")
    return {"total_logprob": logprob_sum, "avg_logprob": avg_logprob}


def maybe_load_uhead(repo_id: str):
    """Download uHead weights/config and return paths. Parsing happens elsewhere."""
    print(f"[uHead] Attempting to locate config + weights in {repo_id} ...")
    try:
        weight_path = hf_hub_download(repo_id, "weights.pth")
    except Exception as exc:  # pragma: no cover - network required
        print(f"[uHead] Could not download weights.pth: {exc}")
        return None
    try:
        config_path = hf_hub_download(repo_id, "config.yaml")
    except Exception as exc:  # pragma: no cover - network required
        print(f"[uHead] Could not download config.yaml: {exc}")
        config_path = None
    return {"weights": weight_path, "config": config_path}


def warn_missing_luh(msg: Optional[str] = None):
    base = (
        "[uHead] Detected custom head weights but additional annotations (claims) are required "
        "to obtain claim-level scores."
    )
    if msg:
        base += f" {msg}"
    print(base)


def build_full_attention_mask(
    sequences: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    generated_steps: int,
) -> torch.Tensor:
    """Construct a full attention mask matching CalculatorInferLuh semantics."""
    batch_size, total_len = sequences.shape
    full_mask = torch.zeros_like(sequences, dtype=torch.bool, device=sequences.device)
    for i in range(batch_size):
        context_len = int(prompt_attention_mask[i].sum().item())
        used_len = min(total_len, context_len + generated_steps)
        full_mask[i, :used_len] = True
    return full_mask


def build_dummy_claims(
    full_attention_mask: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
) -> List[torch.Tensor]:
    """Create a single-claim mask covering the generated tokens (placeholder for demo)."""
    batch_size, full_len = full_attention_mask.shape
    claims: List[torch.Tensor] = []
    for i in range(batch_size):
        context_len = int(prompt_attention_mask[i].sum().item())
        mask_full = torch.zeros(full_len, dtype=torch.int64, device=full_attention_mask.device)
        used = int(full_attention_mask[i].sum().item())
        if used > context_len:
            mask_full[context_len:used] = 1
        claims.append(mask_full[1:].unsqueeze(0))  # drop <s>, keep dims (num_claims, seq_len-1)
    return claims


def main() -> None:
    parser = argparse.ArgumentParser(description="Run base Qwen2.5 model and inspect outputs / scores.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="User prompt for generation.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Hugging Face id for the base LLM.")
    parser.add_argument("--uhead-repo", default=DEFAULT_UHEAD_REPO, help="Hugging Face repo containing the uHead weights.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum new tokens to generate.")
    parser.add_argument(
        "--quantization",
        choices=["none", "4bit", "8bit"],
        default="none",
        help="Quantization mode. Requires bitsandbytes for 4/8 bit loading.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Device preference. Use auto to pick the best available.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3, help="Sampling temperature when do_sample is True."
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9, help="Top-p nucleus sampling parameter if do_sample is True."
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Disable sampling and use greedy decoding (deterministic).",
    )
    parser.add_argument(
        "--save-json",
        help="Optional path to dump the structured output as JSON.",
    )
    parser.add_argument(
        "--enable-uhead",
        action="store_true",
        help="Load the LUH uncertainty head and compute claim-level logits (requires the luh package).",
    )
    args = parser.parse_args()

    device = resolve_device(None if args.device == "auto" else args.device)
    print(f"[Setup] Using device: {device}")

    compute_dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
    quant_config = build_quantization_config(args.quantization, compute_dtype)

    print(f"[Load] Loading tokenizer from {args.base_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)

    print(f"[Load] Loading model from {args.base_model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=compute_dtype,
        device_map="auto" if quant_config is None else None,
        low_cpu_mem_usage=True,
        quantization_config=quant_config,
        trust_remote_code=True,
    )

    if quant_config is None:
        model.to(device)
        model_device = torch.device(device)
    else:
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device(device)

    messages = [
        {"role": "system", "content": "你是一个乐于助人的助手。"},
        {"role": "user", "content": args.prompt},
    ]

    chat_prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    model_inputs = tokenizer(
        chat_prompt,
        return_tensors="pt",
    )

    model_inputs = model_inputs.to(model_device)

    prompt_length = model_inputs["input_ids"].shape[-1]

    generation_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=not args.no_sample,
        return_dict_in_generate=True,
        output_scores=True,
        output_attentions=True,
    )
    if not args.no_sample:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p

    print("[Run] Generating response ...")
    with torch.no_grad():
        generation = model.generate(**model_inputs, **generation_kwargs)

    decoded = tokenizer.decode(generation.sequences[0], skip_special_tokens=False)

    print("\n=== RAW DECODED SEQUENCE ===")
    print(decoded)

    continuation_length = len(generation.scores)
    assistant_token_slice = generation.sequences[0, prompt_length : prompt_length + continuation_length]
    assistant_only = tokenizer.decode(
        assistant_token_slice,
        skip_special_tokens=True,
    )

    print("\n=== ASSISTANT ANSWER (trimmed) ===")
    print(assistant_only.strip())

    print("\n=== GENERATION KEYS ===")
    print(list(generation.keys()))

    if generation.scores:
        print(f"[Scores] Received {len(generation.scores)} score tensors (per decoding step).")
        print(f"[Scores] Tensor shape example: {tuple(generation.scores[0].shape)}")
        token_scores = compute_token_scores(generation.scores, assistant_token_slice, tokenizer)
        summary = summarize_scores(token_scores)
        print(f"[Scores] Total logprob: {summary['total_logprob']:.4f}")
        print(f"[Scores] Average logprob/token: {summary['avg_logprob']:.4f}")
    else:
        token_scores = []
        print("[Scores] No token-level scores returned (likely due to configuration).")

    uhead_summary = None
    if args.enable_uhead:
        if AutoUncertaintyHead is None:
            warn_missing_luh("(the 'luh' package is not installed in this environment)")
        elif assistant_token_slice.numel() == 0:
            print("[uHead] No continuation tokens were generated; skipping uncertainty head inference.")
        else:
            try:
                print(f"[uHead] Loading AutoUncertaintyHead from {args.uhead_repo} ...")
                uq_head = AutoUncertaintyHead.from_pretrained(args.uhead_repo, model)
                uq_head = uq_head.to(model_device)
                uq_head.eval()

                prompt_attention_mask = model_inputs["attention_mask"].detach().clone()
                full_attention_mask = build_full_attention_mask(
                    generation.sequences,
                    prompt_attention_mask,
                    generated_steps=continuation_length,
                ).to(model_device)
                generation["full_attention_mask"] = full_attention_mask
                context_lengths = prompt_attention_mask.sum(dim=1).to(torch.int64)
                generation["context_lengths"] = context_lengths
                model_inputs["context_lenghts"] = context_lengths

                dummy_claims = build_dummy_claims(full_attention_mask, prompt_attention_mask)
                model_inputs["claims"] = dummy_claims

                with torch.no_grad():
                    claim_logits = uq_head(model_inputs, generation)

                claim_logits = claim_logits.squeeze(-1).detach().cpu()
                if claim_logits.numel() == 0:
                    warn_missing_luh("(no claims were constructed – logits array is empty)")
                else:
                    claim_probs = torch.sigmoid(claim_logits)
                    print("\n=== CLAIM-LEVEL LOGITS (naive span) ===")
                    for idx, (logit, prob) in enumerate(zip(claim_logits, claim_probs)):
                        print(f"Claim {idx:02d}: logit={float(logit):+.4f} prob={float(prob):.4f}")
                    print("[uHead] NOTE: claims are approximated by a single span covering all generated tokens. Provide real claim annotations for meaningful scores.")
                    uhead_summary = {
                        "claim_logits": claim_logits.tolist(),
                        "claim_probabilities": claim_probs.tolist(),
                        "claim_mask": dummy_claims[0].int().cpu().tolist() if dummy_claims else [],
                        "span_mode": "generated_tokens_full",
                    }
            except Exception as exc:
                warn_missing_luh(f"Encountered error while applying uHead: {exc}")

        artifact_paths = maybe_load_uhead(args.uhead_repo)
        if artifact_paths is not None:
            print(f"[uHead] weights: {artifact_paths['weights']}")
            if artifact_paths.get("config"):
                if yaml is None:
                    print("[uHead] Install pyyaml to pretty-print config.yaml")
                else:
                    with open(artifact_paths["config"], "r", encoding="utf-8") as f:
                        config_data = yaml.safe_load(f)
                    print("[uHead] config.yaml snippet:")
                    print(json.dumps(config_data, ensure_ascii=False, indent=2)[:2000])

    # Optionally dump structured info
    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        payload = {
            "decoded": decoded,
            "assistant_only": assistant_only,
            "generation_keys": list(generation.keys()),
            "token_scores": [t.__dict__ for t in token_scores],
            "uhead_summary": uhead_summary,
        }
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[Save] Wrote structured output to {args.save_json}")


if __name__ == "__main__":
    main()
