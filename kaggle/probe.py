import json
import os
import shutil
import subprocess
import urllib.request

REPO = "ricursiveAI/Ripple-65B-Preview"
API = f"https://huggingface.co/api/models/{REPO}"
CFG = f"https://huggingface.co/api/models/{REPO}/raw/main/config.json"


def sh(argv):
    p = subprocess.run(argv, capture_output=True, text=True)
    return (p.stdout or "") + (p.stderr or "")


def get(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


print("GPU")
print(sh(["nvidia-smi", "-L"]))
print(sh(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"]))
print(sh(["free", "-g"]))
print("nproc", sh(["nproc"]).strip())

info = json.loads(get(API))
cfg = json.loads(get(CFG))
siblings = info.get("siblings") or []
names = [s.get("rfilename") for s in siblings]
shards = [n for n in names if n and n.startswith("model-") and n.endswith(".safetensors")]

total = 0
for name in shards:
    url = f"https://huggingface.co/{REPO}/resolve/main/{name}"
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=120) as r:
        n = int(r.headers.get("Content-Length") or 0)
    total += n
    print(f"shard {name} {n}")

vram = sh(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"])
vram_mib = []
for line in vram.splitlines():
    line = line.strip()
    if line.isdigit():
        vram_mib.append(int(line))

report = {
    "repo": REPO,
    "model_type": cfg.get("model_type"),
    "architectures": cfg.get("architectures"),
    "num_hidden_layers": cfg.get("num_hidden_layers"),
    "hidden_size": cfg.get("hidden_size"),
    "intermediate_size": cfg.get("intermediate_size"),
    "use_gated_attention": cfg.get("use_gated_attention"),
    "use_valres": cfg.get("use_valres"),
    "shard_count": len(shards),
    "weight_bytes": total,
    "weight_gib": round(total / (1024 ** 3), 2),
    "vram_mib": vram_mib,
    "disk_free_bytes": shutil.disk_usage("/").free,
    "release": False,
    "reason": "BF16 Ripple-65B does not fit T4x2 RAM or VRAM, and stock llama.cpp has no ricursive arch. No generation smoke test, so no weight upload.",
}
path = "/kaggle/working/derisk.json" if os.path.isdir("/kaggle/working") else "derisk.json"
with open(path, "w") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
