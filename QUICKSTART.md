# Quickstart — Running an LLM on the Qualcomm IQ8 NPU, end to end

From a fresh device to generated text on the Hexagon V75 NPU. Every step, with a real
Hugging Face model. The hard part (building the ExecuTorch + QNN toolchain) is one command.

```
HF model (base fp16)  ->  QNN QDQ quantize + AOT compile (V75)  ->  run on the HTP  ->  tokens
```

Small models (≤0.5B) run this whole flow in RAM with no fuss. Models from ~1B up hit a
**compile-time memory wall** and need an SSD + swap (or a host compile) — that's its own
section below, with the why and the exact recipe.

---

## 0. Prerequisites (what the device must already have)

`npufast setup` builds the toolchain for you, but it does **not** install these — they ship
with the IQ8 or its OS image. Confirm them first:

- **Hardware:** Qualcomm Dragonwing IQ8 (QCS8300, Hexagon **V75** HTP), Ubuntu 24.04 aarch64.
- **Qualcomm QNN / QAIRT SDK** under `/usr` (from the IQ8 BSP / Ubuntu image). Required files:
  - `/usr/lib/libQnnHtpPrepare.so` — and it must be **aarch64** (so the device can compile its own models)
  - `/usr/lib/dsp/cdsp/libQnnHtpV75Skel.so` — the vendor-**signed** V75 skel (no testsig wall)
- **System toolchain:**
  ```bash
  sudo apt-get update
  sudo apt-get install -y git cmake ninja-build build-essential python3.12 python3.12-venv
  ```
- A Hugging Face account is only needed for **gated** models (e.g. Meta Llama); the example
  below is open and needs no login.

---

## 1. Clone the repo

```bash
git clone https://github.com/<you>/Running-LLM-Inference-on-Qualcomm-IQ8-NPUs
cd Running-LLM-Inference-on-Qualcomm-IQ8-NPUs
./install.sh          # symlinks `npufast` onto your PATH (or just call ./npufast)
```

---

## 2. Build the toolchain (one time, ~20–60 min)

This is the entire ExecuTorch + QNN bring-up — pinned dependencies, source patches, three
builds — that otherwise takes hours to discover by hand:

```bash
npufast setup
```

What it does, in order: preflight checks → clones ExecuTorch at the pinned commit → creates
the `~/executorch-env` virtualenv and installs the **pinned CPU/aarch64 dependency set**
(torch 2.11.0+cpu and the rest) → applies the source patches (the aarch64 pybind un-gate, the
runner token-echo, the torchao shim) → builds the main tree, the QNN AOT Python bindings, and
the `qnn_llama_runner`. It ends by printing a status summary.

> The Python environment lives in `~/executorch-env` and is **not** in the repo (it's huge,
> path-locked, and arch-specific — see `.gitignore`). `npufast setup` recreates it; the pins
> are also in `requirements.txt` for reference.

> **One-time fixup for non-SmolLM2 models.** ExecuTorch's QNN AOT flow assumes it runs on an
> x86 host, so for some recipes (e.g. Qwen) it tries to load the backend lib from
> `/usr/lib/x86_64-linux-clang/libQnnHtp.so`, which doesn't exist on aarch64. The fix is a
> one-time symlink (the libs are aarch64 and run fine regardless of the directory name):
> ```bash
> sudo mkdir -p /usr/lib/x86_64-linux-clang
> sudo ln -sf /usr/lib/libQnn*.so /usr/lib/x86_64-linux-clang/
> ```
> SmolLM2 doesn't trip this; the first 1B+ model will. Harmless to run preemptively.

---

## 3. Verify before running (fail-fast)

```bash
npufast doctor
```

Expected — everything `ok`:

```
== machine ==
libQnnHtpPrepare : aarch64 (on-device compile OK)
signed V75 skel  : present (no testsig wall)
SOC target       : SM8650 (Hexagon V75)
== toolchain ==
runner       : ok
AOT pybind   : ok
python env   : ok
```

If anything is missing, re-run `npufast setup` (it's idempotent — safe to re-run).

---

## 4. Pick a model (example: SmolLM2-135M)

Example HF link: **https://huggingface.co/HuggingFaceTB/SmolLM2-135M**
→ repo id: `HuggingFaceTB/SmolLM2-135M` (open, Apache-2.0, ~135M params — small + fast, ideal
for a first run, and it compiles entirely in RAM with no swap).

You pass the **base** model; QNN quantizes it (QDQ) during compile — you never hand it a
pre-quantized file. See the supported architectures any time with:

```bash
npufast list
```

---

## 5. Build the model for the NPU + benchmark

```bash
npufast auto HuggingFaceTB/SmolLM2-135M
```

This runs the whole pipeline:
1. **Download** the base fp16 weights from Hugging Face.
2. **Quantize + AOT-compile** for the V75 HTP (`libQnnHtpPrepare`) — you'll see the graph
   stages scroll (Graph Optimizations → VTCM Allocation → Parallelization → Finalize). For a
   135M model this is **~10 minutes on-device and fits in RAM** — it is working, not hung.
3. **Benchmark** — 7 runs, warmup dropped, mean ± std.

Artifact produced: `models/smollm2_135m_qnn/kv_llama_qnn.pte` (+ `tokenizer.json`).
Expected benchmark tail:

```
>> decode: ~106 +/- <1 tok/s (n=6, warmup dropped)
```

(To split the steps: `npufast prep <model>` then `npufast bench models/<name>_qnn`.)

> The compile's memory cost scales hard with model size. 135M is trivial; ~1B and up will
> exhaust the device's RAM and get `Killed` (OOM) unless you give it swap — see **The
> compile-time memory wall** below before you try a bigger model.

---

## 6. Generate output

```bash
npufast run models/smollm2_135m_qnn "Once upon a time"
```

The tokens stream to your terminal. Sampling is on by default (`TEMP=0.8`) so a small model
won't loop. Example:

```
Once upon a time, there was a little girl who loved to explore the forest near her home ...
```

> **Reading the output honestly:** SmolLM2-135M is a *tiny base model* — fluent but not smart.
> Coherent, varied text confirms the **pipeline** is correct (quantization preserved the model);
> quality is the model's ceiling, not the toolchain's. A larger instruct model (e.g.
> Qwen2.5-1.5B, or the gated `meta-llama/Llama-3.2-3B-Instruct`) produces genuinely useful text
> through the identical flow.

---

## Running bigger models

Same command, different repo id — for any architecture in `npufast list`:

```bash
npufast auto Qwen/Qwen2.5-0.5B-Instruct              # open, fits in RAM, no swap
npufast auto Qwen/Qwen2.5-1.5B-Instruct              # open, needs SSD + swap (see below)
huggingface-cli login                                # once, for gated repos
npufast auto meta-llama/Llama-3.2-3B-Instruct        # gated, host-compile recommended
```

The model runs fine on the IQ8 at any of these sizes — a 3B fits in one V75 session and
inference is memory-mapped. **The only thing that scales painfully with size is the compile.**
Pick the compile path by model size:

| Model size | fp16 weights | How to compile |
|---|---|---|
| ≤ 0.5B | ≤ ~1 GB | On-device, **no swap**. Just `npufast auto`. |
| ~1–2B | ~3–4 GB | On-device with **SSD storage + a big swapfile** (works, slow), **or** host-compile. |
| ≥ 3B | ~6 GB+ | **Host-compile** (`npufast-host`). On-device swap would thrash for hours. |

Architectures **not** in `npufast list` need a converter + quant recipe + an op-coverage
check — not just weights.

### Measured on this IQ8 (V75 HTP)

| Model | Decode | DDR read / token-pass | Compile |
|---|---|---|---|
| SmolLM2-135M | ~106 tok/s | 163 MB | ~10 min, in-RAM |
| Qwen2.5-1.5B-Instruct | 26.7 ± 0.2 tok/s | 1.01 GB | ~58 min, SSD + internal swap |
| Granite-3.3-2B-Instruct | 23.0 ± 0.5 tok/s | — | ~1h45m on-device (SSD swap, MAXSEQ=512, A78-pinned) |
| Qwen2.5-1.5B on **CPU** (llama.cpp) | ~6 tok/s | — | — |

The NPU is **~4.3× faster than the CPU on the same 1.5B model** — and the read-per-token
column is the story behind the decode numbers: decode is memory-bandwidth-bound, so a model
that reads ~6× more weight data per token runs proportionally slower. That's the wall, not the
HTP's compute.

---

## The compile-time memory wall (why bigger models need an SSD + swap)

**The one-command path:** everything below is automated by `npufast-bigmem`, which sets up the
SSD (fast `ntfs3` mount), the swapfile, `swappiness=100`, A78 core-pinning, and a reduced
`MAXSEQ`, then compiles:

```bash
npufast-bigmem ibm-granite/granite-3.3-2b-instruct      # 2B, compiled entirely on the IQ8
# if it OOMs at the compile stage, add fast SSD swap and shrink the peak:
SSD_SWAP_G=48 MAXSEQ=256 npufast-bigmem ibm-granite/granite-3.3-2b-instruct
```

The rest of this section is what that script does, and how to do it by hand.

**Why it's required.** The AOT compile is far more memory-hungry than the finished model. To
QDQ-quantize, `llama.py` holds the full fp checkpoint **and** quantized copies **and**
calibration observers in memory at once — and it does this over **two quantize passes**. The
peak RAM is several times the model's on-disk size. The IQ8 has only ~10 GB usable RAM, so:

- **0.5B** quantizes inside 10 GB → no swap needed.
- **1.5B** blew past **10 GB RAM + 15 GB swap (25 GB total)** in testing — its peak needs
  roughly **~50 GB** of RAM+swap to clear both passes without the OOM killer.
- **3B**'s peak is roughly **2× the 1.5B** → don't fight it on-device; host-compile.

The flip side, and the thing to internalize: **only the compile is memory-hungry. Runtime is
not.** Once the `.pte` exists it's memory-mapped into one V75 session, and the IQ8 runs a 1.5B
(or 3B) with no memory pressure at all. So everything below is a *compile-time* scratch setup
you can tear down afterward.

### The recipe (NTFS external SSD — the common case)

A Linux **swapfile can't live on NTFS** (or exFAT). So split the roles: **SSD holds the model
files** (downloads + `.pte` + the multi-GB `.pt2` intermediate), **internal ext4 disk holds
the swapfile**. Find your SSD with `lsblk` / `df -h` (here it's `/dev/sdX1` → `/mnt/ssd`).

```bash
# 1. Mount the SSD with the FAST in-kernel ntfs3 driver (not ntfs-3g/FUSE, which is
#    single-threaded and pegs one core on multi-GB writes)
sudo blkid /dev/sdX1                      # confirm TYPE="ntfs"
sudo mkdir -p /mnt/ssd; sudo umount /mnt/ssd 2>/dev/null
sudo mount -t ntfs3 -o uid=$(id -u),gid=$(id -g),umask=022 /dev/sdX1 /mnt/ssd

# 2. Big swapfile on the INTERNAL ext4 disk (sized to cover the model's peak)
sudo swapoff -a
sudo fallocate -l 45G /swapfile
sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
sudo sysctl -w vm.swappiness=100          # actually use the swap instead of OOM-killing
free -h                                   # Swap should now read ~45Gi

# 3. Send downloads + artifacts to the SSD (NTFS-safe: no symlinks)
export MODELS=/mnt/ssd/models
export HF_HUB_DISABLE_SYMLINKS=1
mkdir -p "$MODELS"

# 4. Compile on the performance cores, with a reduced max_seq_len to shrink the peak
FAST=$(for d in /sys/devices/system/cpu/cpu[0-9]*; do echo "$(cat $d/cpufreq/cpuinfo_max_freq) ${d##*cpu}"; done | sort -rn | head -4 | awk '{print $2}' | paste -sd,)
taskset -c "$FAST" env MAXSEQ=512 npufast auto Qwen/Qwen2.5-1.5B-Instruct
#   - taskset -> A78 cores (the QNN compile is single-threaded; an A55 crawls)
#   - MAXSEQ=512 -> halves the sequence-dependent compile memory AND speeds it up
#   - need more headroom? add fast SSD loop-swap (ntfs3 only): see below
```

Two settings that matter:
- **`vm.swappiness=100`** — the kernel's default (60) will sometimes invoke the OOM killer
  while swap still has room. Bumping it tells the kernel to actually use the swap you gave it.
  This alone was part of why "25 GB wasn't enough" the first time.
- **`HF_HUB_DISABLE_SYMLINKS=1`** — the HF downloader uses symlinks by default, which break on
  NTFS. This makes it copy real files so the SSD writes succeed.

**If your SSD is ext4 instead of NTFS**, skip the split: put the swapfile *on the SSD* (it's
faster than the internal eMMC), and the compile won't crawl the way it does on internal swap.

### After the compile — reclaim everything

The swap and SSD were only for the compile. Tear them down and (optionally) move the `.pte`
to internal storage so the model is self-contained:

```bash
sudo swapoff /swapfile && sudo rm -f /swapfile      # inference needs no swap
mkdir -p ~/models/qwen2_5-1_5b_qnn
rsync -av --exclude='decode_qdq.pt2' /mnt/ssd/models/qwen2_5-1_5b_qnn/ ~/models/qwen2_5-1_5b_qnn/
npufast bench ~/models/qwen2_5-1_5b_qnn             # same tok/s; SSD only ever affected load time
```

Reading the `.pte` off the SSD does **not** slow inference — decode reads weights from DRAM,
not disk; the SSD only adds a one-time ~1–2 s model-load. Moving to internal just drops that
dependency.

### Or skip all of it: host-compile

For 1.5B and especially 3B, the clean answer is to compile on an x86 box with real RAM and
push the `.pte` — see **Faster compiles** below. The IQ8 then only ever *runs* the model.

---

## Tunables (environment variables)

| Var | Default | Meaning |
|---|---|---|
| `MAXSEQ` | 1024 | compile-time max sequence length |
| `SEQ` | 256 | runtime tokens to generate (bench/run) |
| `TEMP` | 0.8 | sampling temperature for `run` (0 = greedy → repeats) |
| `SOC` | SM8650 | HTP arch target (V75; don't change for QCS8300) |
| `MODELS` | `./models` | where artifacts go — point at the SSD for big compiles |
| `HF_HUB_DISABLE_SYMLINKS` | — | set to `1` when `MODELS` is on NTFS/exFAT |
| `LFAST_DECODER` | — | force a decoder architecture for an unmapped repo |
| `QNN_EXTRA` | — | extra args passed to `llama.py` (e.g. `--checkpoint <dir>`) |

Example: `SEQ=512 TEMP=0.6 npufast run models/smollm2_135m_qnn "Write a haiku about the sea"`

---

## Faster compiles: offload to an x86 host (optional)

On-device compile is slow and RAM-bound. If you have an x86 Linux box (with the toolchain
built), compile there and push the artifact — the `.pte` targets the V75 arch, not the host
CPU, so it runs unchanged on the IQ8, and the host's RAM makes the quantize passes painless:

```bash
# on the x86 host:
DEVICE=<iq8-ssh-host> npufast-host deploy meta-llama/Llama-3.2-3B-Instruct
npufast-host bench models/llama3_2-3b_instruct_qnn       # runs remotely on the IQ8
```

On x86 the QNN AOT bindings build with **no aarch64 patch and no x86-symlink workaround**, so
host setup is actually simpler than on-device. This is the recommended path for anything ≥3B.

---

## Troubleshooting

- **`Cannot Open QNN library /usr/lib/x86_64-linux-clang/libQnnHtp.so`** — ExecuTorch's QNN
  flow looks for the backend lib under the x86 host subdir. Create the symlink (one-time):
  `sudo mkdir -p /usr/lib/x86_64-linux-clang && sudo ln -sf /usr/lib/libQnn*.so /usr/lib/x86_64-linux-clang/`.
  The libs are aarch64 and run fine; the directory name is just a label. SmolLM2 doesn't hit
  this; the first 1B+ model does.
- **`Killed` mid-compile (no error, no core dump)** — the Linux **OOM killer**: the quantize
  passes ran out of RAM. Confirm with `dmesg | tail -20 | grep -iE 'kill|oom'`. Fix: add swap
  + `vm.swappiness=100` per **The compile-time memory wall**, or host-compile. Runtime never
  OOMs — this is compile-only.
- **`huggingface-cli: command not found`** — the tool calls the venv's CLI by path, so this
  shouldn't happen; if it does, your venv is missing → `npufast setup`.
- **`download failed (gated?)`** — the model is gated; run `~/executorch-env/bin/huggingface-cli login`.
- **`no architecture mapping for '<repo>'`** — not a supported architecture; check `npufast list`
  or set `LFAST_DECODER=<name>`.
- **Compile seems stuck** — it's not; `libQnnHtpPrepare` runs for several minutes on-device
  (and far longer if it's swapping — a 1.5B on internal swap can take ~an hour).
- **`runner failed`** — run `npufast doctor`; a missing skel or AOT pybind means re-run `npufast setup`.
- **NTFS write errors / symlink errors when `MODELS` is on the SSD** — set
  `export HF_HUB_DISABLE_SYMLINKS=1` and ensure the SSD is mounted with your `uid`/`gid`.
- **Re-run without recompiling** — point bench/run at an existing artifact dir:
  `npufast bench models/smollm2_135m_qnn`.

---

## How the dependencies are tracked

The repo versions the *recipe*, never the environment:
- **`npufast setup`** — the full reproducer (CPU index, source-built ExecuTorch, the patches).
- **`requirements.txt`** — the readable pinned package manifest.
- **`requirements-freeze.txt`** — exact `pip freeze` snapshot for byte-level reproduction.
- **`.gitignore`** — keeps the venv (`~/executorch-env`), `models/`, and `*.pte`/`*.pt2` out of git.

Full technical reference (hardware/SDK facts, every build and patch, results): see
`iq8_qnn_npu_pipeline.md`.
