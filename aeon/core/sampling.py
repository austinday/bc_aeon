"""Canonical Qwen3.8 sampling settings for Aeon's control-plane calls."""

# Aeon is an autonomous tool driver, not a creative chat surface. Greedy
# decoding made tool and argument selection repeatable across the release suite,
# while the formerly recommended stochastic sampler changed grounded actions on
# identical evidence. It also raises native-MTP acceptance and decode speed.
QWEN_CONTROL_TEMPERATURE = 0.0
QWEN_CONTROL_TOP_P = 1.0
QWEN_CONTROL_TOP_K = -1


# The speed lab may compare the production control sampler with Qwen's published
# recommendation without changing production behavior.  Names, values, and
# reasoning modes are deliberately closed over here so a Fleet job cannot inject
# arbitrary generation parameters.
QWEN_SPEED_LAB_SAMPLING_PROFILES = {
    "aeon-greedy-medium": {
        "temperature": QWEN_CONTROL_TEMPERATURE,
        "top_p": QWEN_CONTROL_TOP_P,
        "top_k": QWEN_CONTROL_TOP_K,
        "reasoning_effort": "medium",
        "thinking": True,
    },
    "qwen-recommended-medium": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "reasoning_effort": "medium",
        "thinking": True,
    },
    "qwen-recommended-xhigh": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "reasoning_effort": "xhigh",
        "thinking": True,
    },
}
