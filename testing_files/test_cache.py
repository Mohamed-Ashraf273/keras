import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


from keras_hub.src.models.gemma.gemma_backbone import GemmaBackbone
from keras_hub.src.models.gemma.gemma_causal_lm import GemmaCausalLM
from keras_hub.src.models.gemma.gemma_causal_lm_preprocessor import (
    GemmaCausalLMPreprocessor,
)
from keras_hub.src.models.gemma.gemma_tokenizer import GemmaTokenizer

from keras import ops

# === 1. Setup ===
tokenizer = GemmaTokenizer(
    proto="/home/mohamed-ashraf/Desktop/GSoC2025/keras-hub/keras_hub/src/tests/test_data/gemma_test_vocab.spm"
)

preprocessor = GemmaCausalLMPreprocessor(
    tokenizer=tokenizer,
    sequence_length=8,
)

backbone = GemmaBackbone(
    vocabulary_size=tokenizer.vocabulary_size(),
    num_layers=2,
    num_query_heads=4,
    num_key_value_heads=2,
    hidden_dim=8,
    intermediate_dim=16,
    head_dim=2,
    sliding_window_size=3,
    use_sliding_window_attention=True,
    attention_logit_soft_cap=50,
    final_logit_soft_cap=30,
    query_head_dim_normalize=False,
    use_post_ffw_norm=True,
    use_post_attention_norm=True,
)

causal_lm = GemmaCausalLM(
    preprocessor=preprocessor, backbone=backbone, dtype="float16"
)

# === 2. Prepare input ===
train_data = ["the quick brown fox", "the quick brown fox"]
input_data = preprocessor(train_data)[0]
token_ids = input_data["token_ids"]
padding_mask = ops.ones_like(input_data["padding_mask"])


full_logits = causal_lm(
    {
        "token_ids": token_ids,
        "padding_mask": padding_mask,
    }
)

# === 4. Cached logits ===
_, cache = causal_lm._build_cache(token_ids)
cache = ops.zeros_like(cache)

cached_logits = []
for i in range(preprocessor.sequence_length):
    sliced = token_ids[:, i][:, None]
    logits, _, cache = causal_lm.call_with_cache(sliced, cache, i)
    cached_logits.append(logits)

cached_logits = ops.concatenate(cached_logits, axis=1)

# === 5. Compare ===
full_logits = ops.convert_to_numpy(full_logits)
cached_logits = ops.convert_to_numpy(cached_logits)

# === 6. Assertion ===
# np.testing.assert_allclose(full_logits, cached_logits, atol=0.002, rtol=1e-6)
print(full_logits == cached_logits)
