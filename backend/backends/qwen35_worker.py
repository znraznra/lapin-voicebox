"""
Worker process for Qwen3.5 text generation.

Runs with the private transformers 5 install (see utils/transformers5_env.py)
at the front of sys.path, so the main process can stay on transformers 4.57.

Protocol: newline-delimited JSON. The worker prints {"ready": true} once the
model is loaded, then answers each request line
  {"messages": [...], "max_tokens": N, "temperature": T}
with {"ok": true, "text": "..."} or {"ok": false, "error": "..."}.
It exits when stdin closes.

Standalone on purpose: it must not import the ``backend`` package.
"""

import argparse
import json
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tf5-dir", required=True)
    args = parser.parse_args()

    # The protocol owns the real stdout; anything libraries print goes to stderr.
    proto = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1, encoding="utf-8")
    sys.stdout = sys.stderr

    sys.path.insert(0, args.tf5_dir)

    def reply(obj: dict) -> None:
        proto.write(json.dumps(obj, ensure_ascii=False) + "\n")
        proto.flush()

    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"qwen35 worker: transformers {transformers.__version__}", file=sys.stderr)

        device = args.device if args.device in ("cuda", "xpu", "mps", "cpu") else "cpu"
        dtype = torch.float32 if device in ("cpu", "mps") else torch.bfloat16

        tokenizer = AutoTokenizer.from_pretrained(args.repo)
        model = AutoModelForCausalLM.from_pretrained(args.repo, dtype=dtype)
        model.to(device)
        model.eval()
    except BaseException as exc:  # noqa: BLE001 - reported to the parent
        reply({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1

    reply({"ready": True})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            text = tokenizer.apply_chat_template(
                req["messages"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(text, return_tensors="pt").to(device)

            temperature = float(req.get("temperature", 0.0))
            kwargs = {
                "max_new_tokens": int(req["max_tokens"]),
                "do_sample": temperature > 0,
                "pad_token_id": tokenizer.eos_token_id,
            }
            if temperature > 0:
                kwargs["temperature"] = temperature
                kwargs["top_p"] = 0.9

            with torch.no_grad():
                out = model.generate(**inputs, **kwargs)
            new_tokens = out[0, inputs["input_ids"].shape[1]:]
            reply({"ok": True, "text": tokenizer.decode(new_tokens, skip_special_tokens=True).strip()})
        except BaseException as exc:  # noqa: BLE001
            reply({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    return 0


if __name__ == "__main__":
    sys.exit(main())
