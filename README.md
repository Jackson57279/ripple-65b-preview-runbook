# Ripple-65B Preview — Kaggle derisk

Source: [ricursiveAI/Ripple-65B-Preview](https://huggingface.co/ricursiveAI/Ripple-65B-Preview) (Apache-2.0).

This repo records why that checkpoint was not re-quantized or re-uploaded. The reference package this was compared against edits residual-write matrices to soften guardrails. That edit is not in this repo. The weights here are the upstream checkpoint, untouched.

## What the checkpoint is

`RicursiveForCausalLM`, `model_type: ricursive`. Dense, 80 layers, `hidden_size` 5120, `intermediate_size` 40960, 80 heads / 8 KV heads, sliding window 1024 on 64 of 80 layers, gated attention, value residuals, FFN clamp, vocab 100352, untied embeddings. Sixteen BF16 safetensor shards. Stock llama.cpp and stock vLLM do not load `model_type: ricursive`.

## Why Kaggle cannot run it

Kaggle on this account lands on T4x2: 2×15 GB VRAM, ~31 GB RAM, 4 vCPU. The BF16 tree is on the order of 130 GB. It does not fit in VRAM, and it does not fit in RAM for a CPU offload either. A quant small enough for 30 GB would still need the BF16 shards on disk first, and this host has under 7 GB free.

`kaggle/probe.py` is the kernel. It prints the GPU the scheduler actually gave, sums shard sizes from the Hub API, and writes `derisk.json` with `release: false`. It does not download weight shards.

## Release gate

No Hugging Face weight repo. A quant or mirror goes up only after a smoke generation that writes a real Python file. That run has not happened.

```bash
kaggle kernels push -p kaggle
kaggle kernels status otdoges/ripple-65b-derisk
kaggle kernels output otdoges/ripple-65b-derisk -p /tmp/ripple-derisk
```
