# On-Device LLM Inference on the Qualcomm Dragonwing IQ8 (QCS8300 / Hexagon V75)
### ExecuTorch + QNN pipeline — full build, conversion, and runtime reference

This documents the complete pipeline we stood up: taking a model from Hugging Face,
quantizing and compiling it ahead-of-time **natively on the IQ8 (aarch64)**, and
running it on the device's Hexagon NPU. It records every build, every patch, the
hardware specifics, and what does / does not carry over to llama.cpp or future work.

---

## 0. TL;DR

- **Device:** Qualcomm Dragonwing IQ8 EVK — **QCS8300** SoC, Ubuntu 24.04, aarch64.
- **NPU used:** the **single** on-chip **Hexagon Tensor Processor (HTP), arch version V75**. One NPU, one session. (See §6 — this is important and commonly misunderstood.)
- **Stack:** ExecuTorch 1.4.0 (commit `55c54c7`) + Qualcomm QNN / QAIRT 2.45 SDK (installed under `/usr`).
- **Result:** SmolLM2-135M, INT-quantized, compiled to a QNN `.pte` and run on the HTP at **~106.6 ± 0.6 tok/s decode**, with coherent (sampled) output — pipeline fully validated end to end.
- **Key unlock:** ExecuTorch hard-gates the QNN ahead-of-time Python bindings to x86 only. We un-gated them for aarch64, which let us do the entire export/compile **on the device itself** instead of needing an x86 host.

---

## 1. The pipeline, end to end

```
[Hugging Face model]                         (online: HuggingFaceTB/SmolLM2-135M)
        │  safetensors + tokenizer.json
        ▼
[Weight conversion]                          torchtune FullModelHFCheckpointer
        │  HF param names -> static-llama param names
        ▼
[Graph construction]                         StaticLlamaModel (KV-cache mode)
        │
        ▼
[torch.export]                               capture ExportedProgram
        │
        ▼
[PTQ quantization]                           calibrate -> QDQ integer format
        │  (intermediate: decode_qdq.pt2)
        ▼
[QNN lowering]                               qnn_preprocess: each aten op -> QNN op
        │  ("Visiting: aten_matmul_default ...")
        ▼
[HTP AOT compile]                            libQnnHtpPrepare.so (AARCH64, on-device)
        │  Graph Optimizations / VTCM Allocation / Parallelization / Finalize
        │  ~10 min on-device; emits a V75 context binary
        ▼
[Serialize .pte]                             kv_llama_qnn.pte (196 MB) + tokenizer.json
        │
        ▼
[Runtime]                                    qnn_llama_runner (ExecuTorch Module)
        │  QNN backend type 2 = HTP; RESTORE mode (loads embedded context)
        │  FastRPC -> cDSP (libxdsprpc / rpcmem) -> executes on V75 HTP
        ▼
[Tokens out]                                 ~106 tok/s decode
```

Two valid topologies exist for this; we chose the second:

- **Standard (two-machine):** x86 host does the AOT export/compile, aarch64 device runs the `.pte`. This is what ExecuTorch assumes by default.
- **What we did (one-machine, native):** the IQ8 does **both** AOT and runtime. This required un-gating the x86-only QNN Python bindings (§4). Advantage: the graph is compiled for the device's exact Hexagon arch with zero SoC-mismatch risk, and the whole flow lives on one box.

---

## 2. Hardware / SDK facts (the durable knowledge)

| Item | Value |
|---|---|
| Board | Qualcomm Dragonwing IQ8 EVK |
| SoC | QCS8300 |
| CPU | 4× Cortex-A78 + 4× Cortex-A55, 12 GB RAM |
| OS | Ubuntu 24.04, aarch64 |
| **NPU** | **Hexagon Tensor Processor (HTP), arch V75** — one unit |
| QNN/QAIRT SDK | 2.45.0, installed under `/usr` (not the zip layout) |
| HTP AOT compiler | `/usr/lib/libQnnHtpPrepare.so` — **ARM aarch64** (this is why native on-device AOT is possible) |
| HTP host lib | `/usr/lib/libQnnHtp.so`, `/usr/lib/libQnnSystem.so` |
| Signed device skel | `/usr/lib/dsp/cdsp/libQnnHtpV75Skel.so` (vendor-signed) |
| SoC→arch table | `backends/qualcomm/serialization/qc_schema.py`: `SM8650 -> HtpArch.V75` |

**SoC targeting:** QCS8300 is not in ExecuTorch's SoC map, so we compile with its
**V75 sibling, `--soc_model SM8650`** (Snapdragon 8 Gen 3, also Hexagon V75). The
context binary is keyed to the HTP *arch version*, so SM8650's V75 output is
binary-compatible with the QCS8300's V75 HTP. (SM8550 = V73 and SM8750 = V79 would
**not** load.)

**No testsig wall:** earlier QNN attempts on the Rock Pi / Rubik Pi hit Qualcomm's
unsigned-skel rejection. The IQ8 ships vendor-**signed** V75 skels in `/usr/lib/dsp/cdsp/`,
so the HTP loads cleanly with no self-signing required. This is a meaningful
advantage of the IQ8 over the hobbyist boards.

---

## 3. Build A — the runtime (`qnn_llama_runner`)

`examples/qualcomm` is a **post-install consumer**, not an in-tree subproject.
Trying `add_subdirectory(examples/qualcomm)` fails (gflags/abseil/tokenizers target
collisions, `executorch_load_build_variables()` needs Bazel context). The correct
flow is: build + install the main tree first, then build the examples standalone
against the install.

**Step 1 — build & install the main ExecuTorch tree** (note the full flag set; each
was discovered via the preset dependency checker):

```bash
cmake -S . -B cmake-out \
  -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_QNN=ON \
  -DEXECUTORCH_BUILD_EXTENSION_LLM=ON \
  -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
  -DEXECUTORCH_BUILD_KERNELS_LLM=ON \
  -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DQNN_SDK_ROOT=/usr \
  -DCMAKE_INSTALL_PREFIX="$HOME/executorch/install"
cmake --build cmake-out -j$(nproc)
cmake --install cmake-out
```

Dependency chain learned (preset enforces these): `QNN` needs `EXTENSION_TENSOR`;
`KERNELS_LLM` needs `KERNELS_OPTIMIZED`; `EXTENSION_MODULE` needs
`EXTENSION_DATA_LOADER` + `EXTENSION_FLAT_TENSOR` + `EXTENSION_NAMED_DATA_MAP`.
`EXTENSION_MODULE`, `quantized_ops_lib`, `quantized_kernels` must be built and
installed because `extension_llm_runner`'s installed CMake config pulls
`extension_module` transitively.

**Step 2 — patch the llama example CMakeLists** (`examples/qualcomm/oss_scripts/llama/CMakeLists.txt`)
to tolerate IMPORTED targets from the installed package:
- wrap `add_library(custom_ops ...)` in `if(NOT TARGET custom_ops)` (it already exists as an imported target);
- remove the redundant `executorch_target_link_options_shared_lib(quantized_ops_lib)` call (can't set link options on an imported target);
- the missing direct link libs (`extension_module`, `quantized_ops_lib`, `quantized_kernels`) resolve once Step 1 installs them.

**Step 3 — build the runner standalone:**

```bash
cmake -S examples/qualcomm -B cmake-out-qnn \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$HOME/executorch/install" \
  -DQNN_SDK_ROOT=/usr \
  -Dgflags_DIR="$HOME/executorch/cmake-out/third-party/gflags"
cmake --build cmake-out-qnn --target qnn_llama_runner -j$(nproc)
# -> cmake-out-qnn/oss_scripts/llama/qnn_llama_runner
```

`gflags_DIR` points at the in-tree build's config because `examples/qualcomm` does
`find_package(gflags REQUIRED)`; we also removed the system `libgflags-dev` to avoid
a target-name collision.

---

## 4. Build B — the AOT Python bindings (the key unlock)

The export pipeline (`llama.py`) imports `executorch.backends.qualcomm.python.PyQnnManagerAdaptor`,
a compiled pybind module. It did not exist on the IQ8 because:

```cmake
# backends/qualcomm/CMakeLists.txt  -- the QNN pybind block:
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64|AMD64")   #  <-- x86 ONLY
    add_library(PyQnnManagerAdaptor MODULE)
    ...
endif()
```

ExecuTorch assumes AOT happens on an x86 host, so it never builds the bindings on
aarch64. **The fix that made native on-device export possible:**

```cmake
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64|AMD64|aarch64")
```

Then reconfigure with `-DPYTHON_EXECUTABLE=$(which python3)` and build the target:

```bash
cmake --build cmake-out --target PyQnnManagerAdaptor -j$(nproc)
cp cmake-out/backends/qualcomm/PyQnnManagerAdaptor.cpython-312-aarch64-linux-gnu.so \
   backends/qualcomm/python/
```

It links only against libs that build fine on aarch64 (`qnn_manager`, `wrappers`,
`qnn_schema`, `executorch`, `extension_tensor`, …) and uses the aarch64
`libQnnHtpPrepare.so` at runtime. In this ExecuTorch version there is **only one**
pybind module (`PyQnnManagerAdaptor`); the "wrapper" adaptor of older layouts does
not exist and nothing imports it.

---

## 5. The Python environment (the part that ate the most time)

The export's transitive imports are huge and fragile on aarch64. Root principle:
**keep `torch` pinned to the CPU build and never let a dependency swap it.** All
installs used `--no-deps` or a constraint file pinning the ecosystem.

Working, mutually-compatible set (mirrors the x86 export box, but aarch64/CPU):

```
torch            2.11.0+cpu          # never touched; everything pinned around it
torchvision      0.26.0+cpu          # from download.pytorch.org/whl/cpu (fixes torchvision::nms)
torchao          0.17.0              # API note below
torchtune        0.4.0
transformers     4.57.6
tokenizers       0.22.2
safetensors      0.8.0
huggingface_hub  0.36.2
lm_eval, accelerate, pydot           # pulled in by eager imports
```

Three classes of problem and their fixes:

1. **Compiled-package ABI/arch mismatch.** `torchvision::nms does not exist` was a
   torchvision built against the wrong torch/arch. Fix: reinstall the matching
   CPU/aarch64 wheel from `--index-url https://download.pytorch.org/whl/cpu`. (My
   initial assumption that torchao had the same problem was wrong — see #2.)

2. **torchao API drift.** `int4_weight_only` was **removed** from modern torchao
   (renamed `Int4WeightOnlyConfig`). torchtune 0.4.0 still imports the old name. No
   released aarch64 wheel has both the old symbol and torch-2.11 compatibility — only
   a specific pre-removal source commit does. Since those quant functions are **never
   called** during weight conversion or QNN export, we guarded the import in
   `torchtune/training/quantization.py`:
   ```python
   try:
       from torchao.quantization import (int4_weight_only,
           int8_dynamic_activation_int4_weight, quantize_)
   except Exception:
       int4_weight_only = int8_dynamic_activation_int4_weight = quantize_ = None
   ```

3. **Eager imports of unused models.** `llama/__init__.py` eagerly imports *every*
   model's weight converter (gemma, granite_speech, phi_4_mini, …) and the
   vision/audio encoders, dragging in unused heavy deps. We wrapped the converter
   imports in `try/except` (fall back to `None`, which the framework already accepts,
   e.g. `stories110m` has `convert_weights = None`).

---

## 6. Did we use other NPUs? (read this carefully)

**No — there is exactly one NPU on this chip, and we used it.**

- The QCS8300 has a **single Hexagon NPU**, the **HTP (Hexagon Tensor Processor),
  arch version V75**. That is the only neural accelerator on the SoC. There is also
  a CPU (8 cores) and an Adreno GPU, but those are not NPUs and we did not use them
  for inference (the CPU only provided the 7.21 tok/s *baseline reference*).
- The HTP internally has multiple HVX vector threads / an HMX matrix engine, but it
  is exposed to software as **one device**.
- **"HTP0 / HTP1 / HTP2 / HTP3" are NOT separate NPUs.** Those are multiple
  **sessions** (Hexagon Process Domains) on the *same* physical NPU. Each session is
  limited to roughly a 3.5 GB memory mapping (~2 GB usable in practice). You allocate
  more than one session only to **split a model too large for one** (e.g. an 8B needs
  2, a 20B needs 4). This session-splitting concept comes from llama.cpp's Hexagon
  backend docs; ExecuTorch/QNN handles large models with its own multi-context scheme.
- **We used a single session.** SmolLM2-135M is ~135 M params and trivially fits.
  Even Llama-3.2-3B quantized (~2 GB) fits in **one** V75 session, so the future 3B
  run also needs only one.
- At runtime the QNN backend reports **`backend type 2`** = HTP, and the
  `libxdsprpc` / `rpcmem` lines confirm FastRPC traffic to the **cDSP** (the Hexagon).
  That is the proof the work ran on the NPU, not the CPU/GPU.

So: **one NPU, the Hexagon V75 HTP, one session.** No multi-NPU, no GPU.

---

## 7. Does this transfer to llama.cpp or future builds?

**The artifacts do not transfer; the hardware/environment knowledge does.**

What is ExecuTorch-specific (does **not** carry to llama.cpp):
- the `.pte` file and its embedded QNN context binary,
- `qnn_llama_runner`,
- the `PyQnnManagerAdaptor` bindings and the whole AOT/PTQ flow.

llama.cpp has a **completely separate** Hexagon path:
- Backend flag `GGML_HEXAGON=ON`; it builds `libggml-hexagon.so` (CPU-side) +
  `libggml-htp-vNN.so` (per-arch NPU kernels: v73/v75/v79/v81).
- It consumes **GGUF** models (ideally `Q4_0`), not `.pte`.
- The official upstream backend (in `ggml-org/llama.cpp`,
  `docs/backend/snapdragon/`) is **Android-oriented**: cross-compiled with the NDK in
  a toolchain Docker image, pushed via `adb`, run on `/data/local/tmp`. Running it on
  the IQ8's *Linux* userspace is an open question (it needs the Linux FastRPC/cDSP
  path, not Android's) and would still need the Hexagon SDK.
- In llama.cpp the HTP appears as a "GPU" device for `-ngl`/offload; `D=HTP0`,
  `GGML_HEXAGON_NDEV` controls session count.

What **does** transfer to any future on-device-NPU work on this board:
- **The hardware is V75**, and `--soc_model SM8650` / `HtpArch.V75` is the target.
- **`libQnnHtpPrepare.so` is aarch64**, so on-device AOT is feasible (not just an x86
  thing).
- **The signed V75 skels in `/usr/lib/dsp/cdsp/` work** — no testsig wall, FastRPC to
  the cDSP succeeds from Linux userspace.
- **The QNN SDK lives at `/usr`** (non-standard layout — `QNN_SDK_ROOT=/usr`, host
  libs in `/usr/lib`, device skels in `/usr/lib/dsp/cdsp`).
- The **two-stage build pattern** (install ExecuTorch, then build examples against
  it) and the **x86-guard un-gate** are reusable for any ExecuTorch+QNN target.
- The **dependency-coherence discipline** (pin torch, `--no-deps`, match the working
  version set) applies to any torch-ecosystem build on aarch64.

For llama.cpp specifically, the most useful carried knowledge is: this device's
Linux cDSP stack is reachable and the skels are signed, so the main remaining unknown
for a llama.cpp Hexagon build here is whether the upstream backend supports a Linux
(non-Android) target — and you already have a working GGUF + 7.21 tok/s CPU baseline
to test against.

---

## 8. Patches applied this session (so they can be re-found / upstreamed / reverted)

| File | Change | Why |
|---|---|---|
| `backends/qualcomm/CMakeLists.txt` | pybind guard `x86_64\|AMD64` → `+\|aarch64` | build QNN AOT bindings natively |
| `examples/qualcomm/oss_scripts/llama/CMakeLists.txt` | guard `custom_ops` add_library; drop redundant `quantized_ops_lib` shared-lib call | IMPORTED-target collisions in standalone build |
| `torchtune/training/quantization.py` (site-packages) | guard `from torchao.quantization import (...)` → `None` | torchao removed `int4_weight_only`; unused for QNN export |
| `examples/qualcomm/oss_scripts/llama/__init__.py` | wrap model-converter imports in try/except | tolerate unused models' broken optional deps |
| `examples/qualcomm/oss_scripts/llama/qnn_llama_runner.cpp` | add `fputs(piece.c_str(), stdout)` in the token callback | the runner only buffered text to `--output_path`, never printed it |

---

## 9. Reproduce: export + run

**Export (native AOT on the IQ8):**

```bash
python3 examples/qualcomm/oss_scripts/llama/llama.py \
  --decoder_model smollm2_135m --model_mode kv --max_seq_len 1024 \
  --compile_only --soc_model "SM8650" \
  --build_folder build-qnn-smollm2 \
  --prompt "Once upon a time" \
  --artifact ~/models/smollm2_135m_qnn
# -> ~/models/smollm2_135m_qnn/kv_llama_qnn.pte (196 MB) + tokenizer.json
```

**Run (on the HTP):**

```bash
cmake-out-qnn/oss_scripts/llama/qnn_llama_runner \
  --model_path ~/models/smollm2_135m_qnn/kv_llama_qnn.pte \
  --tokenizer_path ~/models/smollm2_135m_qnn/tokenizer.json \
  --prompt "Once upon a time" --seq_len 128 --eval_mode 0 \
  --temperature 0.8
```

- `--eval_mode 0` = KV TokenGenerator (matches `--model_mode kv`).
- HF `tokenizer.json` loads via the runner's first-choice HF loader (the
  sentencepiece/llama2c format from stories110M did *not* work — use HF json).
- `--num_iters` overwrites rather than appends its perf file; for N-run stats, loop
  the command externally and collect each `inference_speed.txt`.

---

## 10. Results & honest caveats

- **Throughput:** SmolLM2-135M on V75 HTP = **106.6 ± 0.6 tok/s decode**, ~105 tok/s
  prefill, ~0.5 s model load, <40 ms to first token. Sub-1% run-to-run spread.
- **Correctness:** validated. Greedy output is repetitive (expected: greedy decoding
  on a 135M *base* model); with `--temperature 0.8` it produces varied, fluent text.
  Coherent grammar confirms the QDQ→HTP quantization preserved model function.
- **Compile cost:** the on-device aarch64 AOT compile took ~10 min (slower than x86,
  but it works) — a noteworthy data point for "native vs cross-compiled AOT."
- **Memory-bandwidth signal:** the HTP compiler reported per-pass DDR traffic
  (`read_total_bytes ≈ 163 MB` vs `write_total_bytes ≈ 0.65 MB`) — a clean
  read-dominated, memory-bound signature for the bandwidth-bottleneck narrative.

**The big caveat:** 106 tok/s (135M, HTP) is **not** comparable to the 7.21 tok/s
baseline (Llama-3.2-3B Q4_K_M, **CPU**) — different models, ~22× apart. This run
proves the *pipeline*, not an NPU-vs-CPU speedup.

---

## 11. What's next (infrastructure done; only models + measurement remain)

1. **Same-model speedup (no approvals needed):** export SmolLM2-135M to a CPU (fp /
   XNNPACK) `.pte` or use a SmolLM2 GGUF in llama.cpp, run on CPU, compare to the
   106 tok/s HTP number. Isolates the NPU speedup *and* the quantization quality cost
   for the same model — a clean figure for SCC/ERSP.
2. **Headline benchmark (waiting on gated Meta weights):** export
   `llama3_2-3b_instruct` through the identical flow, run on the HTP, compare to
   7.21 tok/s CPU. Fits in one V75 session.
3. **Output quality:** SmolLM2 emits ChatML tokens (`<|im_start|>`/`<|im_end|>`), so
   wrap prompts with the chat template (`get_formatted_prompt` + the
   `chat_template.jinja` in the artifact) for cleaner behavior.
