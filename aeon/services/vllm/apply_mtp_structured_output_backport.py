#!/usr/bin/env python3
"""Apply the audited vLLM structured-output/spec-decode backport.

Qwen reasoning, native MTP, and post-reasoning JSON schemas interact in the
vLLM scheduler.  vLLM upstream PR #44993 (merge commit
0416dab275d51327b331a1c6baaec754a68d7764) fixes three coupled problems:

* speculative placeholder accounting can hide the ``</think>`` token;
* speculative positions after that token can be sampled without a grammar;
* a mixed reasoning/final token block can be advanced through the JSON FSM.

The production image intentionally stays on vLLM 0.23.0 because that exact
release is already validated for Qwen3.8 NVFP4 and its native MTP head.  This
script backports only the pure-Python scheduler fix and refuses to touch any
unexpected package build.  It runs while the Docker image is built.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace


VLLM_ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")
UPSTREAM_PR = "https://github.com/vllm-project/vllm/pull/44993"
UPSTREAM_MERGE = "0416dab275d51327b331a1c6baaec754a68d7764"

BASE_SHA256 = {
    "v1/structured_output/__init__.py":
        "23fef6a034ea960e6470fd4b40401eae45c3828883dcdb5cc291ae304a255b3c",
    "v1/structured_output/request.py":
        "704c1a4b10669f469d6a963dfe3adf7f4c1abb2307cdd8772fc54e00cb4182d9",
    "v1/core/sched/scheduler.py":
        "9ae1491f57a6294e32416b93ef2ea68107fa0d65c3a4d7f745f1261012f735e9",
}

PATCHED_SHA256 = {
    "v1/structured_output/__init__.py":
        "6aa84efbc509127442c842bc9d94c3687796e96082497144c44be418e6c2d935",
    "v1/structured_output/request.py":
        "527a12a73b9f6b2d8635712de49d6adb229597986f84e6a5cbf128e1dea26acb",
    "v1/core/sched/scheduler.py":
        "70768722d52aa0bedff3e5f91cdb051be227c570a0a80e0e88e3a8d7c3ec71de",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one source match, found {count}")
    return text.replace(old, new, 1)


def _write_atomic(path: Path, text: str) -> None:
    mode = path.stat().st_mode
    temporary = path.with_name(path.name + ".aeon-backport-tmp")
    temporary.write_text(text, encoding="utf-8")
    os.chmod(temporary, mode)
    temporary.replace(path)


def _self_test(root: Path) -> None:
    """Exercise the boundary transition and per-position mask without a GPU."""
    sys.path.insert(0, str(root.parent))
    try:
        import torch
        from vllm.v1.structured_output import StructuredOutputManager
    finally:
        sys.path.pop(0)

    marker, before, after = 9001, 101, 202

    class Reasoner:
        @staticmethod
        def is_reasoning_end(_tokens):
            return False

        @staticmethod
        def is_reasoning_end_streaming(_tokens, delta):
            return marker in list(delta)

    class Grammar:
        def __init__(self):
            self.accepted = []

        @staticmethod
        def is_terminated():
            return False

        @staticmethod
        def fill_bitmask(mask, index):
            mask[index].zero_()

        def accept_tokens(self, _request_id, tokens):
            self.accepted.extend(tokens)
            return True

        def rollback(self, count):
            del self.accepted[-count:]

    class Backend:
        @staticmethod
        def allocate_token_bitmask(rows):
            return torch.full((rows, 2), -1, dtype=torch.int32)

    grammar = Grammar()
    structured = SimpleNamespace(
        grammar=grammar,
        reasoner=Reasoner(),
        reasoning_ended=False,
        reasoning_end_token_index=None,
    )
    request = SimpleNamespace(
        request_id="aeon-backport-self-test",
        structured_output_request=structured,
        use_structured_output=True,
        prompt_token_ids=[11],
        all_token_ids=[11, before],
        num_computed_tokens=2,
        num_output_placeholders=0,
    )
    manager = object.__new__(StructuredOutputManager)
    manager.backend = Backend()
    manager.reasoner_cls = Reasoner
    manager.enable_in_reasoning = False
    manager._grammar_bitmask = None
    manager._full_mask = torch.tensor(-1, dtype=torch.int32)
    manager.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
    )
    manager.fill_bitmask_parallel_threshold = 128

    mask = manager.grammar_bitmask(
        {request.request_id: request},
        [request.request_id],
        {request.request_id: [before, marker, after]},
    )
    assert mask.shape == (4, 2)
    assert (mask[0] == -1).all() and (mask[1] == -1).all()
    assert (mask[2] != -1).any() and (mask[3] != -1).any()
    assert grammar.accepted == [], "grammar simulation was not rolled back"

    accepted_block = [before, marker, after]
    request.all_token_ids = [11] + accepted_block
    assert manager.should_advance(request, new_token_ids=accepted_block)
    assert structured.reasoning_ended is True
    assert structured.reasoning_end_token_index == 2
    assert manager.trim_reasoning_for_advance(request, accepted_block) == [after]


def apply(root: Path = VLLM_ROOT) -> dict[str, str]:
    paths = {relative: root / relative for relative in BASE_SHA256}
    for relative, path in paths.items():
        actual = _sha256(path.read_bytes())
        if actual != BASE_SHA256[relative]:
            raise RuntimeError(
                f"refusing structured-output backport: {relative} has SHA-256 "
                f"{actual}, expected exact vLLM 0.23.0 source {BASE_SHA256[relative]}"
            )

    request_path = paths["v1/structured_output/request.py"]
    request_source = request_path.read_text(encoding="utf-8")
    request_source = _replace_once(
        request_source,
        """    reasoning_ended: bool | None = None
    reasoning_parser_kwargs: dict[str, Any] | None = None
""",
        """    reasoning_ended: bool | None = None
    # Absolute index of the reasoning-end marker. Tokens at or before this
    # index must never advance the final-output grammar (vLLM PR #44993).
    reasoning_end_token_index: int | None = None
    reasoning_parser_kwargs: dict[str, Any] | None = None
""",
        "StructuredOutputRequest boundary field",
    )

    manager_path = paths["v1/structured_output/__init__.py"]
    manager_source = manager_path.read_text(encoding="utf-8")
    old_bitmask = """                state_advancements = 0
                req_tokens = scheduled_spec_decode_tokens.get(req_id, ())
                for token in itertools.chain(req_tokens, (-1,)):
                    self._fill_bitmasks(((grammar, cumulative_index, apply_bitmask),))
                    if token == -1:
                        # Stop advancing the grammar once we hit a padding token.
                        apply_bitmask = False
                    if apply_bitmask and not grammar.is_terminated():
                        accepted = grammar.accept_tokens(req_id, [token])
                        assert accepted, (token, req_id, scheduled_spec_decode_tokens)
                        state_advancements += 1
                    cumulative_index += 1
                if state_advancements > 0:
                    grammar.rollback(state_advancements)
"""
    new_bitmask = """                # A speculative window can straddle </think>. Detect the
                # transition while simulating draft positions so every later
                # position (including the target bonus token) receives the JSON
                # grammar before sampling.
                reasoner = self._get_reasoner(request)
                detect_reasoning_end = (
                    not apply_bitmask
                    and reasoner is not None
                    and not self.enable_in_reasoning
                )
                simulated_buf: list[int] | None = None
                history_len = 0

                state_advancements = 0
                post_reasoning_end_in_window = False
                req_tokens = scheduled_spec_decode_tokens.get(req_id, ())
                for i, token in enumerate(req_tokens):
                    self._fill_bitmasks(((grammar, cumulative_index, apply_bitmask),))
                    advance_grammar = apply_bitmask
                    if token == -1:
                        apply_bitmask = False
                        advance_grammar = False
                    elif detect_reasoning_end and not apply_bitmask:
                        if simulated_buf is None:
                            history = list(request.all_token_ids)
                            history_len = len(history)
                            simulated_buf = history + list(req_tokens)
                        simulated = simulated_buf[: history_len + i + 1]
                        if reasoner.is_reasoning_end_streaming(simulated, [token]):
                            # The marker is reasoning, not grammar content. Apply
                            # the grammar only to positions after the marker.
                            apply_bitmask = True
                            advance_grammar = False
                            post_reasoning_end_in_window = True
                    if advance_grammar and not grammar.is_terminated():
                        accepted = grammar.accept_tokens(req_id, [token])
                        if accepted:
                            state_advancements += 1
                        elif not post_reasoning_end_in_window:
                            raise AssertionError(
                                (token, req_id, scheduled_spec_decode_tokens)
                            )
                    cumulative_index += 1

                # This vLLM release has no diffusion-LLM path: every decode
                # window has one target bonus position after the draft rows.
                bonus_apply = self.should_fill_bitmask(request) or apply_bitmask
                self._fill_bitmasks(((grammar, cumulative_index, bonus_apply),))
                cumulative_index += 1
                if state_advancements > 0:
                    grammar.rollback(state_advancements)
"""
    manager_source = _replace_once(
        manager_source, old_bitmask, new_bitmask, "speculative grammar bitmask"
    )

    old_advance = """    def should_advance(self, request: \"Request\") -> bool:
        if not request.use_structured_output:
            return False

        # To determine whether we can advance the FSM.
        # Supports thinking usage where we skip the reasoning components.
        if TYPE_CHECKING:
            assert request.structured_output_request is not None
            assert request.structured_output_request.grammar is not None
        # by default, we should always advance
        # for cases that don't use thinking mode.
        reasoner = self._get_reasoner(request)
        if reasoner is None:
            return True

        # if the model needs structured in reasoning, we should advance
        if self.enable_in_reasoning:
            return True

        structured_req = request.structured_output_request
        if structured_req.reasoning_ended:
            return True

        # Check if reasoning ends in *this* step
        delta_from = request.num_computed_tokens - request.num_output_placeholders
        all_token_ids = request.all_token_ids
        start = (
            delta_from if delta_from >= 0 else max(len(all_token_ids) + delta_from, 0)
        )
        if reasoner.is_reasoning_end_streaming(
            all_token_ids, itertools.islice(all_token_ids, start, None)
        ):
            structured_req.reasoning_ended = True

            # Reasoning just ended this step. Defer FSM advance until the next
            # pass (see reasoning_ended check above) for JSON/regex/choice/grammar:
            # advancing on the closing boundary token can accept tokens that still
            # belong to the reasoning stream. Structural tags are the only safe
            # same-step exception: they model phased output (e.g. thinking tag ->
            # answer tag), and speculative decoding must run grammar.validate_tokens
            # on draft tokens produced immediately after that transition.
            if (
                self.vllm_config.speculative_config is not None
                and structured_req.structured_output_key[0]
                == StructuredOutputOptions.STRUCTURAL_TAG
            ):
                return True

        return False
"""
    new_advance = """    def should_advance(
        self,
        request: \"Request\",
        new_token_ids: list[int] | None = None,
    ) -> bool:
        if not request.use_structured_output:
            return False

        # To determine whether we can advance the FSM.
        # Supports thinking usage where we skip the reasoning components.
        if TYPE_CHECKING:
            assert request.structured_output_request is not None
            assert request.structured_output_request.grammar is not None
        reasoner = self._get_reasoner(request)
        if reasoner is None:
            return True
        if self.enable_in_reasoning:
            return True

        structured_req = request.structured_output_request
        if structured_req.reasoning_ended:
            return True

        # Use the actual accepted tokens when the scheduler has them. Deriving
        # this window from speculative placeholders can begin *after* </think>
        # when some drafts were rejected, permanently bypassing the grammar.
        all_token_ids = request.all_token_ids
        if new_token_ids:
            start = len(all_token_ids) - len(new_token_ids)
            delta_ids = new_token_ids
        else:
            delta_from = request.num_computed_tokens - request.num_output_placeholders
            start = (
                delta_from
                if delta_from >= 0
                else max(len(all_token_ids) + delta_from, 0)
            )
            delta_ids = itertools.islice(all_token_ids, start, None)

        if reasoner.is_reasoning_end_streaming(all_token_ids, delta_ids):
            structured_req.reasoning_ended = True
            structured_req.reasoning_end_token_index = (
                self._find_reasoning_end_index(reasoner, all_token_ids, start)
            )
            return True
        return False

    @staticmethod
    def _find_reasoning_end_index(
        reasoner: \"ReasoningParser\", all_token_ids, start: int
    ) -> int:
        \"\"\"Locate the last reasoning token within this accepted block.\"\"\"
        prefix = list(itertools.islice(all_token_ids, start))
        for idx in range(start, len(all_token_ids)):
            token = all_token_ids[idx]
            prefix.append(token)
            if reasoner.is_reasoning_end_streaming(prefix, [token]):
                return idx
        # Conservative fallback for a multi-token marker recognized only when
        # the parser sees the complete delta.
        return len(all_token_ids) - 1

    def trim_reasoning_for_advance(
        self, request: \"Request\", new_token_ids: list[int]
    ) -> list[int]:
        \"\"\"Return only final-output tokens after a mid-block </think>.\"\"\"
        structured_req = request.structured_output_request
        if structured_req is None:
            return new_token_ids
        end_idx = structured_req.reasoning_end_token_index
        if end_idx is None:
            return new_token_ids
        first_idx = len(request.all_token_ids) - len(new_token_ids)
        num_reasoning = end_idx + 1 - first_idx
        if num_reasoning <= 0:
            return new_token_ids
        return new_token_ids[num_reasoning:]
"""
    manager_source = _replace_once(
        manager_source, old_advance, new_advance, "reasoning/FSM transition"
    )

    scheduler_path = paths["v1/core/sched/scheduler.py"]
    scheduler_source = scheduler_path.read_text(encoding="utf-8")
    old_scheduler = """            if new_token_ids and self.structured_output_manager.should_advance(request):
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                assert struct_output_request.grammar is not None
                if not struct_output_request.grammar.accept_tokens(  # type: ignore[union-attr]
                    req_id, new_token_ids
                ):
                    logger.error(
                        \"Unexpected: grammar rejected tokens %s for request %s. \"
                        \"Terminating request.\",
                        new_token_ids,
                        req_id,
                    )
                    request.status = RequestStatus.FINISHED_ERROR
                    request.resumable = False
                    stopped = True
"""
    new_scheduler = """            if new_token_ids and self.structured_output_manager.should_advance(
                request, new_token_ids=new_token_ids
            ):
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                grammar = struct_output_request.grammar
                assert grammar is not None
                # A speculative result can contain reasoning, </think>, and
                # final JSON together. Only the suffix belongs to the grammar.
                advance_token_ids = (
                    self.structured_output_manager.trim_reasoning_for_advance(
                        request, new_token_ids
                    )
                )
                if advance_token_ids and not grammar.accept_tokens(
                    req_id, advance_token_ids
                ):
                    logger.error(
                        \"Unexpected: grammar rejected tokens %s for request %s. \"
                        \"Terminating request.\",
                        advance_token_ids,
                        req_id,
                    )
                    request.status = RequestStatus.FINISHED_ERROR
                    request.resumable = False
                    stopped = True
"""
    scheduler_source = _replace_once(
        scheduler_source, old_scheduler, new_scheduler, "scheduler FSM advance"
    )

    outputs = {
        request_path: request_source,
        manager_path: manager_source,
        scheduler_path: scheduler_source,
    }
    for path, source in outputs.items():
        compile(source, str(path), "exec")
    for path, source in outputs.items():
        _write_atomic(path, source)

    hashes = {
        str(path.relative_to(root)): _sha256(path.read_bytes())
        for path in outputs
    }
    if hashes != PATCHED_SHA256:
        raise RuntimeError(f"patched vLLM source hashes are unexpected: {hashes}")
    _self_test(root)
    return hashes


if __name__ == "__main__":
    hashes = apply()
    print(
        "Applied vLLM MTP/structured-output backport "
        f"from {UPSTREAM_PR} at {UPSTREAM_MERGE}"
    )
    for relative, digest in sorted(hashes.items()):
        print(f"{digest}  {relative}")
