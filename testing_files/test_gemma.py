# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# from unittest.mock import patch

# from keras import ops

# import keras_hub

# # Load the model
# causal_lm = keras_hub.models.GemmaCausalLM.from_preset(
#     "gemma_2b_en", dtype="float16"
# )
# tokenizer = causal_lm.preprocessor.tokenizer
# end_token_id = tokenizer.end_token_id

# # Patch call_with_cache to always return EOS
# original_call = causal_lm.call_with_cache


# def force_eos_logits(*args, **kwargs):
#     logits, hidden_states, cache = original_call(*args, **kwargs)
#     update = ops.ones_like(logits)[:, :, end_token_id] * ops.convert_to_tensor(1.0e4, dtype=logits.dtype)
#     update = ops.expand_dims(update, axis=-1)
#     logits = ops.slice_update(logits, (0, 0, end_token_id), update)
#     return logits, hidden_states, cache


# # Apply the patch and generate output
# with patch.object(causal_lm, "call_with_cache", wraps=force_eos_logits):
#     prompt = ["the quick brown fox", "the quick"]
#     output = causal_lm.generate(prompt)

# # Print generated token IDs and decoded text
# print("Generated token IDs:", output)


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from keras_hub.src.models.gemma.gemma_backbone import GemmaBackbone
from keras_hub.src.models.gemma.gemma_causal_lm import GemmaCausalLM
from keras_hub.src.models.gemma.gemma_causal_lm_preprocessor import (
    GemmaCausalLMPreprocessor,
)
from keras_hub.src.models.gemma.gemma_tokenizer import GemmaTokenizer

tokenizer = GemmaTokenizer(
    proto="/home/mohamed-ashraf/Desktop/GSoC2025/keras-hub/keras_hub/src/tests/test_data/gemma_test_vocab.spm"
)

# 2. Preprocessor with fixed sequence length
preprocessor = GemmaCausalLMPreprocessor(
    tokenizer,
    sequence_length=8,
)

# 3. Miniature Gemma-style backbone
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

# 4. Construct the full model
causal_lm = GemmaCausalLM(
    preprocessor=preprocessor, backbone=backbone, dtype="float32"
)

prompt = ["keras is", "the quick brown fox"]
output = causal_lm.generate(prompt)
output2 = causal_lm.generate(prompt)


print(output)
