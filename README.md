# npufast

Run quantized Hugging Face LLMs on a Qualcomm **Hexagon NPU**, the easy way.

```
HF model (base fp16)  ->  QNN QDQ quantize + AOT compile (V75)  ->  run on the HTP
```

Built and verified on a **Qualcomm Dragonwing IQ8 (QCS8300, Hexagon V75 HTP)** running
Ubuntu 24.04 aarch64. It drives **ExecuTorch + QNN** — the only working on-device NPU path
on this platform.

## Why NPU-only

llama.cpp's own Hexagon backend is Android/Windows-only as of 2026, so on Linux the working
NPU path is ExecuTorch + QNN, full stop. CPU inference is a different tool and a different
purpose; `npufast` exists to make the **NPU** trivial, because standing it up by hand is a
multi-hour gauntlet of build flags, dependency pins, and source patches. `npufast setup`
collapses that into one command; then `npufast auto <hf-model>` does download → quantize →
compile → benchmark.

You supply the **base fp16 weights** from Hugging Face; QNN quantizes them (QDQ) during the
compile. There is no "use a pre-quantized GGUF/GPTQ" path — QNN quantizes from full precision.

## Hardware facts (baked in)

- NPU: a **single** Hexagon Tensor Processor (HTP), arch **V75**, on the QCS8300. One NPU,
  one session — a 3B fits in one session.
- SoC target: `--soc_model SM8650` (QCS8300's V75 sibling; SM8550=V73 / SM8750=V79 won't load).
- QNN/QAIRT SDK at `/usr`; `libQnnHtpPrepare.so` is aarch64 (on-device compile works); signed
  V75 skel in `/usr/lib/dsp/cdsp/` (no testsig wall).

## Commands

```
npufast setup                         # ONE-TIME: build the ExecuTorch+QNN toolchain
npufast doctor                        # toolchain + NPU status
npufast list                          # supported architectures
npufast prep  <hf_repo|dir> [decoder] # download + QNN compile -> .pte
npufast bench <artifact>              # 7-run HTP benchmark, mean +/- std (warmup dropped)
npufast run   <artifact> ["prompt"]   # generate on the HTP
npufast auto  <hf_repo> [decoder]     # prep + bench
npufast-bigmem <hf_repo> [decoder]    # big models on-device: SSD + swap + A78-pin + reduced MAXSEQ
```

## Quickstart

```bash
git clone https://github.com/<you>/npufast && cd npufast && ./install.sh

npufast setup                                  # first time only (minutes to ~an hour)
npufast auto HuggingFaceTB/SmolLM2-135M        # download -> quantize -> compile -> bench
npufast run  models/smollm2_135m_qnn "Once upon a time"
```

Gated models (Meta Llama) differ only at download — `huggingface-cli login` first, then
`npufast auto meta-llama/Llama-3.2-3B-Instruct`.

### Offload the compile to an x86 host (optional)

`npufast-host` compiles the `.pte` on an x86 Linux box (faster than on-device aarch64) and
pushes it to the device. The artifact targets the V75 arch, not the host CPU, so it runs
unchanged on the IQ8.

```bash
# on an x86 Linux box with the toolchain built:
DEVICE=<device-ssh-host> npufast-host deploy meta-llama/Llama-3.2-3B-Instruct
npufast-host bench models/llama3_2-3b_instruct_qnn
```

## Supported models

The architectures in ExecuTorch's decoder set: Llama-3.2-1B/3B, Qwen2.5-0.5B/1.5B,
Qwen3-0.6B/1.7B, Gemma-2B/2-2B/3-1B, Phi-4-mini, GLM-Edge-1.5B, Granite-3.3-2B, CodeGen2-1B,
SmolLM2/SmolLM3. Run `npufast list`.
Other architectures need a converter + quant recipe + an op-coverage check — not just weights.

## Honesty / caveats

- `npufast setup` is **pinned to a known-good ExecuTorch commit**; on a newer commit a patch
  anchor may move — it **warns and skips** rather than corrupting a file. One bespoke CMake
  patch (`custom_ops` guard) is left as warn-and-do-manually; see the docs.
- NPU support is **per-architecture** (op coverage), not universal.
- Throughput is model-dependent. A 135M model at ~106 tok/s is **not** comparable to a 3B at
  CPU baseline speed — compare like-for-like (same model, same quant).
- Models from ~1B up are gated by the **compile**'s RAM peak, not runtime. Use `npufast-bigmem`
  (on-device, SSD + swap) or `npufast-host` (x86/host) for those; runtime on the IQ8 is fine.
- QNN CLI flags drift between releases; tunables are env vars at the top of the script.

## Measured (Qualcomm IQ8, Hexagon V75, on-device)

| Model | Decode | Compile |
|---|---|---|
| SmolLM2-135M | ~106 tok/s | ~10 min, in-RAM |
| Qwen2.5-1.5B-Instruct | 26.7 tok/s | ~58 min, SSD + swap |
| Granite-3.3-2B-Instruct | 23.0 tok/s | ~1h45m, `npufast-bigmem` (SSD swap, MAXSEQ=512, A78-pinned) |
| Qwen2.5-1.5B on CPU (llama.cpp) | ~6 tok/s | — |

Same-model NPU-vs-CPU: **~4.3×** on the 1.5B. Decode is memory-bandwidth-bound, so tok/s
declines with model size (more weight bytes read per token).

## Docs

[`docs/iq8_qnn_npu_pipeline.md`](docs/iq8_qnn_npu_pipeline.md) — full reference: hardware/SDK
facts, every build, every patch, the single-NPU explanation, ExecuTorch-vs-llama.cpp, results.

## License

MIT — see [LICENSE](LICENSE).
