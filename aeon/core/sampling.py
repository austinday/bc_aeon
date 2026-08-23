"""Canonical Qwen3.8 sampling settings for Aeon's control-plane calls."""

# Aeon is an autonomous tool driver, not a creative chat surface. Greedy
# decoding made tool and argument selection repeatable across the release suite,
# while the formerly recommended stochastic sampler changed grounded actions on
# identical evidence. It also raises native-MTP acceptance and decode speed.
QWEN_CONTROL_TEMPERATURE = 0.0
QWEN_CONTROL_TOP_P = 1.0
QWEN_CONTROL_TOP_K = -1
