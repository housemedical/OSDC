# AirLLM repository review

Repository reviewed: <https://github.com/lyogavin/airllm>

## What this project is

AirLLM is an inference-focused wrapper around Hugging Face causal LMs designed to run very large models under tight VRAM constraints. The core claim is that it can run models like 70B on low-VRAM hardware by loading model weights layer-by-layer from disk instead of keeping the full model resident in GPU memory.

## How it works (mechanically)

1. **Model-type autodetection**
   - `AutoModel.from_pretrained(...)` inspects `config.architectures[0]` and maps each architecture to a custom AirLLM class (Llama, Qwen/Qwen2, Baichuan, ChatGLM, InternLM, Mistral, Mixtral).
   - On macOS, it routes to an MLX-specific implementation.

2. **One-time checkpoint transformation into per-layer artifacts**
   - On first use, AirLLM resolves/downloads the model and creates a `splitted_model` directory containing per-layer persisted tensors.
   - It supports both PyTorch shard index and safetensors index sources.
   - Before splitting, it checks disk capacity to avoid partial writes.

3. **Meta initialization to avoid full weight allocation**
   - It builds the transformer model with `init_empty_weights()` so parameters start on meta tensors.
   - It then loads only buffers globally; actual layer parameters are brought in just-in-time.

4. **Layer streaming in forward pass**
   - During `forward`, each layer is loaded from persisted storage to CPU, moved to runtime device, executed, then immediately sent back to `meta` and memory-cleaned.
   - This bounds peak VRAM roughly to active layer + activations + KV cache, rather than full model weights.

5. **Generation integration**
   - The base class subclasses `GenerationMixin` and implements `prepare_inputs_for_generation`, cache handling, attention/position-id plumbing, and returns logits / past-key-values in standard transformer style.

6. **Optional speed/IO features**
   - **Prefetching**: pins CPU memory and overlaps load/compute (CUDA path).
   - **Weight-only compression** (`4bit` / `8bit`): compresses persisted layer files with bitsandbytes, then dequantizes when loading layers.
   - This is framed as primarily reducing disk IO bottleneck, not full activation quantization.

## Why this design exists

AirLLM prioritizes **memory feasibility first**, accepting IO overhead as the tradeoff:

- Keeping all weights on GPU is impossible for 70B+ models on small cards.
- Keeping all weights on CPU still often exceeds RAM and adds transfer overhead.
- Streaming per layer from disk permits inference to proceed with very low VRAM.
- Additional compression reduces read volume to recover some speed while preserving model behavior better than aggressive end-to-end quantized execution.

In short: the project targets users who have limited VRAM but can tolerate slower token latency in exchange for running larger base models.

## Practical strengths

- Broad model-family coverage through architecture dispatch.
- Familiar API surface (`from_pretrained`, tokenizer, `generate`).
- Cross-platform intent (CUDA/Linux plus MLX/macOS path).
- Explicit profiling hooks and configurable knobs (`compression`, `prefetching`, `delete_original`, custom shard path).

## Notable caveats and tradeoffs

- **Disk and filesystem dependence**: first-run split requires large free space and can be long for big models.
- **Latency sensitivity**: per-layer disk reads can hurt throughput/latency if storage is slow.
- **Complex cache/model compatibility surface**: custom forward logic and architecture-specific subclasses need ongoing maintenance as `transformers` evolves.
- **Compression constraints**: bitsandbytes is required; prefetching is disabled for compression mode in current implementation.

## High-level architecture summary

- `air_llm/airllm/auto_model.py`: architecture dispatch entrypoint.
- `air_llm/airllm/airllm_base.py`: main streaming-forward implementation and generation integration.
- `air_llm/airllm/utils.py`: splitting, loading, compression/decompression, disk checks.
- `air_llm/airllm/persist/*`: persistence backend abstraction (safetensors vs MLX).

## Commands used for this review

```bash
cd /tmp && git clone --depth 1 https://github.com/lyogavin/airllm.git airllm_repo
cd /tmp/airllm_repo && rg --files | head -n 40
cd /tmp/airllm_repo && sed -n '1,220p' README.md
cd /tmp/airllm_repo && sed -n '1,260p' air_llm/airllm/airllm_base.py
cd /tmp/airllm_repo && sed -n '260,620p' air_llm/airllm/airllm_base.py
cd /tmp/airllm_repo && sed -n '1,260p' air_llm/airllm/utils.py
cd /tmp/airllm_repo && sed -n '1,240p' air_llm/airllm/auto_model.py
cd /tmp/airllm_repo && sed -n '1,220p' air_llm/airllm/persist/model_persister.py
```
