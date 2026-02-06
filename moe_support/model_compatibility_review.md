# Model compatibility review (architecture detection)

Reviewed repositories:
- `moonshotai/Kimi-K2.5`
- `zai-org/GLM-4.7`
- `internlm/Intern-S1-Pro`
- `cerebras/MiniMax-M2.1-REAP-139B-A10B`

## Findings from Hugging Face metadata

- **Kimi-K2.5**
  - `architectures`: `KimiK25ForConditionalGeneration`
  - `model_type`: `kimi_k25`
  - shard keys use root like `language_model.model.layers.<n>...`

- **GLM-4.7**
  - `architectures`: `Glm4MoeForCausalLM`
  - `model_type`: `glm4_moe`
  - shard keys use root like `model.layers.<n>...`

- **Intern-S1-Pro**
  - `architectures`: `InternS1ProForConditionalGeneration`
  - `model_type`: `interns1_pro`
  - shard keys use root like `model.language_model.layers.<n>...`

- **MiniMax-M2.1-REAP-139B-A10B**
  - `architectures`: `MiniMaxM2ForCausalLM`
  - `model_type`: `minimax_m2`
  - shard keys use root like `model.layers.<n>...`

## Compatibility assessment with updated code

The updated extension can now:
1. Detect family/backend class from architecture/model_type/repo with generic keyword rules.
2. Infer layer schema from shard index keys, including non-standard roots such as
   `language_model.model.layers` or `transformer.h`.

This means these models are now far more likely to initialize correctly without per-model hardcoding.

## Remaining caveat

Automatic routing + schema inference does not guarantee runtime parity for every custom architecture. Some families may still require AirLLM core forward/cache logic changes if their attention/cache contract diverges from the selected backend class.
