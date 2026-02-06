from moe_support import detect_airllm_class_name, infer_layer_names_dict


def test_detect_glm_family_routes_to_chatglm_loader():
    class_name = detect_airllm_class_name(
        architecture="Glm4MoeForCausalLM",
        model_type="glm4_moe",
        repo_id="zai-org/GLM-4.7",
    )
    assert class_name == "AirLLMChatGLM"


def test_detect_intern_family_routes_to_internlm_loader():
    class_name = detect_airllm_class_name(
        architecture="InternS1ProForConditionalGeneration",
        model_type="interns1_pro",
        repo_id="internlm/Intern-S1-Pro",
    )
    assert class_name == "AirLLMInternLM"


def test_infer_layer_schema_with_language_model_root():
    keys = [
        "language_model.model.embed_tokens.weight",
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "language_model.model.layers.1.self_attn.q_proj.weight",
        "language_model.model.norm.weight",
        "language_model.lm_head.weight",
    ]
    schema = infer_layer_names_dict(keys)
    assert schema["embed"] == "language_model.model.embed_tokens"
    assert schema["layer_prefix"] == "language_model.model.layers"
    assert schema["norm"] == "language_model.model.norm"
    assert schema["lm_head"] == "language_model.lm_head"


def test_infer_layer_schema_transformer_h_style():
    keys = [
        "transformer.wte.weight",
        "transformer.h.0.attn.c_attn.weight",
        "transformer.h.1.attn.c_attn.weight",
        "transformer.ln_f.weight",
        "lm_head.weight",
    ]
    schema = infer_layer_names_dict(keys)
    assert schema["embed"] == "transformer.wte"
    assert schema["layer_prefix"] == "transformer.h"
    assert schema["norm"] == "transformer.ln_f"
    assert schema["lm_head"] == "lm_head"
