# Ripple-65B Preview runbook

Source weights: [ricursiveAI/Ripple-65B-Preview](https://huggingface.co/ricursiveAI/Ripple-65B-Preview) (Apache-2.0). Published quant: [Ripple-65B-Preview-DERISKED](https://huggingface.co/freakyskittle/Ripple-65B-Preview-DERISKED).

`RicursiveForCausalLM` is dense, 80 layers, hidden size 5120, intermediate size 40960, gated attention, value residuals, sliding window 1024. Sixteen BF16 shards are 120.67 GiB. Stock llama.cpp and stock vLLM do not load `model_type: ricursive`.

## What was checked

Kaggle on this account is 2× Tesla T4 (15360 MiB) and 31 GB RAM. The probe is `kaggle/probe.py`. It does not download shards. BF16 does not fit there.

One secure-cloud NVIDIA A40 (46068 MiB, $0.59/hr) can hold the 36.13 GiB Q4_K_M GGUF from [mrfakename/Ripple-65B-Preview-GGUF](https://huggingface.co/mrfakename/Ripple-65B-Preview-GGUF). That file was converted with llama.cpp `6f1e654` plus `ricursive.patch` from the same repo. The commands used on the pod are `runpod/a40_q4.sh`. llama.cpp itself is not in this repo.

The Hugging Face repo is that Q4 file under the name `Ripple-65B-Preview-DERISKED`. No residual-write edit was applied.
