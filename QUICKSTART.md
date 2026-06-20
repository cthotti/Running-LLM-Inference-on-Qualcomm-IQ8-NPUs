# Quickstart: SmolLM2-135M end to end on the IQ8 NPU

Zero-error, copy-paste path from a Hugging Face model to tokens on the Hexagon V75 HTP.

Model: https://huggingface.co/HuggingFaceTB/SmolLM2-135M  (open, Apache-2.0)

## 0. One-time, per machine

```bash
git clone <this-repo> && cd <this-repo>
./install.sh                       # symlinks npufast onto PATH (or just use ./npufast)
./npufast setup                    # builds the ExecuTorch+QNN toolchain (skip if already built)
```

## 1. Verify before running (fail-fast)

```bash
./npufast doctor
```
Expect: libQnnHtpPrepare aarch64, signed V75 skel present, runner ok, AOT pybind ok,
python env ok. If anything is missing, run `./npufast setup`.

## 2. Full pipeline from the HF link

```bash
./npufast auto HuggingFaceTB/SmolLM2-135M
```
This downloads the base fp16 weights, QNN-quantizes (QDQ) + AOT-compiles for V75
(~10 min on-device), then runs a 7-run benchmark and prints `decode: <mean> +/- <std> tok/s`.

## 3. Generate

```bash
./npufast run models/smollm2_135m_qnn "Once upon a time"
```
Sampled by default (TEMP=0.8) so it won't loop. A 135M base model stays simple — that's the
model's ceiling, not the pipeline.

## Notes

- No venv activation needed: npufast calls the venv's tools (python, huggingface-cli) by path.
- Gated models (Meta Llama): run `~/executorch-env/bin/huggingface-cli login` once first.
- Re-run a model instantly without recompiling by pointing at an existing artifact dir:
  `./npufast bench models/smollm2_135m_qnn`
- Tunables (env): `MAXSEQ` (compile max_seq_len), `SEQ` (runtime length), `TEMP`, `SOC`,
  `MODELS`, `LFAST_DECODER` (force an architecture), `QNN_EXTRA` (extra llama.py args).
