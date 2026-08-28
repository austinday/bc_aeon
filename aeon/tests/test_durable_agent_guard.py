from __future__ import annotations

import types
import unittest

from aeon.core.durable_agent_guard import (
    INTENT_CAPABILITY,
    INTENT_CREATE,
    INTENT_NONE,
    DurableAgentTurnGuard,
    VerifiedNexusAgentStart,
    claims_agent_creation_success,
    classify_project_manager_agent_intent,
    verified_start_receipt,
)


class DurableAgentIntentTests(unittest.TestCase):
    def test_live_capability_question_is_informational(self):
        self.assertEqual(
            classify_project_manager_agent_intent(
                "can you make me an agent to take care of the bananacoconut site"
            ),
            INTENT_CAPABILITY,
        )

    def test_live_imperative_is_durable_creation(self):
        self.assertEqual(
            classify_project_manager_agent_intent(
                "make an agent for the bananacoconut website"
            ),
            INTENT_CREATE,
        )
        self.assertEqual(
            classify_project_manager_agent_intent(
                "Please start a new Aeon agent for that site"
            ),
            INTENT_CREATE,
        )

    def test_explicit_software_artifacts_and_agent_ui_work_remain_generic(self):
        for request in (
            "build an LLM agent application",
            "create an LLM agent script for the website",
            "write a program for an AI agent",
            "fix the agent page",
            "make the agent page responsive",
            "make the agent better",
            "how do we fix this Aeon agent behavior?",
            "why did you say you created an agent?",
        ):
            with self.subTest(request=request):
                self.assertEqual(
                    classify_project_manager_agent_intent(request), INTENT_NONE
                )

    def test_role_vocabulary_recognizes_new_tabs_and_agent_sessions(self):
        for request in (
            "create a new tab for the bananacoconut site",
            "create an agent session for the bananacoconut site",
        ):
            with self.subTest(request=request):
                self.assertEqual(
                    classify_project_manager_agent_intent(request), INTENT_CREATE
                )

    def test_short_confirmation_requires_a_pending_capability_question(self):
        self.assertEqual(
            classify_project_manager_agent_intent(
                "go ahead", pending_confirmation=True
            ),
            INTENT_CREATE,
        )
        self.assertEqual(
            classify_project_manager_agent_intent("go ahead"), INTENT_NONE
        )
        self.assertEqual(
            classify_project_manager_agent_intent(
                "go ahead and create it", pending_confirmation=True
            ),
            INTENT_CREATE,
        )

    def test_negated_request_does_not_arm_later_confirmation(self):
        guard = DurableAgentTurnGuard(project_manager=True)
        guard.begin_user_turn("don't create an agent yet; just explain it")
        self.assertEqual(guard.intent, INTENT_CAPABILITY)
        self.assertFalse(guard.pending_confirmation)

        guard.begin_user_turn("go ahead")
        self.assertEqual(guard.intent, INTENT_NONE)


class PromptAuthorizationContractTests(unittest.TestCase):
    def test_main_orchestrator_prompt_keeps_capability_questions_informational(self):
        from aeon.core.prompts import MAIN_ORCHESTRATOR_INSTRUCTIONS

        self.assertIn("informational", MAIN_ORCHESTRATOR_INSTRUCTIONS)
        self.assertIn("DO NOT", MAIN_ORCHESTRATOR_INSTRUCTIONS)
        self.assertIn("explicitly directs present action", MAIN_ORCHESTRATOR_INSTRUCTIONS)
        self.assertIn("registers an idle tab", MAIN_ORCHESTRATOR_INSTRUCTIONS)
        self.assertIn("same Fleet-managed", MAIN_ORCHESTRATOR_INSTRUCTIONS)
        self.assertIn("Never claim an Aeon uses Llama", MAIN_ORCHESTRATOR_INSTRUCTIONS)


class DurableAgentTurnGuardTests(unittest.TestCase):
    def setUp(self):
        self.guard = DurableAgentTurnGuard(project_manager=True)

    @staticmethod
    def _start_action():
        return {
            "tool_name": "start_agent_instance",
            "parameters": {
                "name": "Bananacoconut Site Agent",
                "directory": "/home/aday/website_hosting/bananacoconut",
                "kind": "aeon",
            },
        }

    @staticmethod
    def _receipt():
        return verified_start_receipt(
            {
                "id": "agent-123",
                "name": "Bananacoconut Site Agent",
                "workspace": "/home/aday/website_hosting/bananacoconut",
                "kind": "aeon",
                "mode": "agent",
                "status": "idle",
                "awaiting_objective": True,
            },
            expected_name="Bananacoconut Site Agent",
            expected_workspace="/home/aday/website_hosting/bananacoconut",
            expected_kind="aeon",
        )

    def test_inflight_creation_and_verified_receipt_survive_restart(self):
        self.guard.begin_user_turn(
            "create an agent session for the bananacoconut site"
        )
        self.guard.observe_tool_result("start_agent_instance", self._receipt())

        restored = DurableAgentTurnGuard(project_manager=True)
        restored.restore_state_dict(self.guard.to_state_dict())

        self.assertEqual(restored.intent, INTENT_CREATE)
        self.assertTrue(restored.attempted)
        self.assertEqual(restored.verified_instance["id"], "agent-123")
        self.assertEqual(restored.completion_error("The idle tab is registered."), "")

    def test_malformed_persisted_guard_state_fails_closed(self):
        restored = DurableAgentTurnGuard(project_manager=True)
        restored.restore_state_dict(
            {"intent": INTENT_CREATE, "verified_instance": {"id": "../../bad"}}
        )

        self.assertEqual(restored.intent, INTENT_NONE)
        self.assertIsNone(restored.verified_instance)

    def test_capability_turn_bypasses_skill_routing_and_blocks_actions(self):
        note = self.guard.begin_user_turn(
            "can you make me an agent to take care of the bananacoconut site"
        )
        self.assertTrue(self.guard.bypass_skill_routing)
        self.assertIn("not authorization", note)

        actions, error = self.guard.prepare_actions([self._start_action()])
        self.assertEqual(actions, [])
        self.assertIn("informational", error)

        actions, error = self.guard.prepare_actions([
            {
                "tool_name": "say_to_user",
                "parameters": {"message": "I can do that; here is the plan."},
            },
            {"tool_name": "task_complete", "parameters": {"reason": "Plan delivered"}},
        ])
        self.assertEqual(len(actions), 2)
        self.assertEqual(error, "")

    def test_exact_bad_turn_is_blocked_before_shell_or_old_memory_can_count(self):
        self.guard.begin_user_turn("make an agent for the bananacoconut website")
        proposed = [
            {
                "tool_name": "run_command",
                "parameters": {
                    "command": "python old_site_agent.py --health && ollama list"
                },
            },
            {
                "tool_name": "say_to_user",
                "parameters": {
                    "message": (
                        "The Bananacoconut Site Agent has been created and is ready "
                        "to use with Ollama and llama 3.1."
                    )
                },
            },
            {"tool_name": "task_complete", "parameters": {"reason": "Done"}},
        ]

        actions, error = self.guard.prepare_actions(proposed)

        self.assertEqual(actions, [])
        self.assertIn("shell, file, skill, memory, Ollama", error)
        self.assertFalse(self.guard.attempted)
        self.assertIsNone(self.guard.verified_instance)

    def test_bridge_is_selected_without_preliminary_shell_actions(self):
        self.guard.begin_user_turn("make an agent for the bananacoconut website")
        start = self._start_action()
        actions, error = self.guard.prepare_actions([
            {"tool_name": "run_command", "parameters": {"command": "ollama list"}},
            start,
            {
                "tool_name": "say_to_user",
                "parameters": {"message": "It is ready."},
            },
        ])
        self.assertEqual(actions, [start])
        self.assertEqual(error, "")

    def test_before_attempt_only_an_explicit_clarification_wait_is_allowed(self):
        self.guard.begin_user_turn("make an agent for the bananacoconut website")
        clarification = [
            {
                "tool_name": "say_to_user",
                "parameters": {"message": "Which existing directory should it use?"},
            },
            {
                "tool_name": "get_user_input",
                "parameters": {"prompt": "Please provide the directory."},
            },
        ]
        self.assertEqual(self.guard.prepare_actions(clarification), (clarification, ""))

        report_only = [
            {
                "tool_name": "say_to_user",
                "parameters": {"message": "I am done."},
            }
        ]
        actions, error = self.guard.prepare_actions(report_only)
        self.assertEqual(actions, [])
        self.assertIn("do not report completion", error)

        false_get = [{
            "tool_name": "get_user_input",
            "parameters": {
                "prompt": "The agent has been created and is ready; what next?"
            },
        }]
        actions, error = self.guard.prepare_actions(false_get)
        self.assertEqual(actions, [])
        self.assertIn("verified start_agent_instance receipt", error)

    def test_clarification_response_stays_in_the_creation_transaction(self):
        self.guard.begin_user_turn("make an agent for the bananacoconut website")
        clarification = [
            {
                "tool_name": "say_to_user",
                "parameters": {"message": "Which directory should it use?"},
            },
            {
                "tool_name": "get_user_input",
                "parameters": {"prompt": "Please provide the directory."},
            },
        ]
        self.assertEqual(self.guard.prepare_actions(clarification), (clarification, ""))

        self.guard.begin_user_turn("/home/aday/website_hosting/bananacoconut")

        self.assertEqual(self.guard.intent, INTENT_CREATE)
        self.assertTrue(self.guard.bypass_skill_routing)

    def test_native_ask_user_clarification_stays_in_creation_transaction(self):
        self.guard.begin_user_turn("make me another agent for the bananacoconut website")

        self.assertEqual(
            self.guard.prepare_ask_user(
                "What exact project directory should the new tab use?"
            ),
            "",
        )
        self.guard.begin_user_turn("/home/aday/website_hosting/bananacoconut")

        self.assertEqual(self.guard.intent, INTENT_CREATE)
        self.assertTrue(self.guard.bypass_skill_routing)

    def test_pending_creation_can_be_cancelled(self):
        self.guard.begin_user_turn("make me another agent for the website")
        self.assertEqual(
            self.guard.prepare_ask_user("Which directory should it use?"), ""
        )

        self.guard.begin_user_turn("never mind")

        self.assertEqual(self.guard.intent, INTENT_NONE)
        self.assertFalse(self.guard.awaiting_clarification)

    def test_waiting_creation_state_is_rearmed_after_worker_restart(self):
        self.guard.resume_waiting_request(
            "make me another agent for the bananacoconut website",
            "What exact project directory should it use?",
        )
        self.guard.begin_user_turn("/home/aday/website_hosting/bananacoconut")

        self.assertEqual(self.guard.intent, INTENT_CREATE)

    def test_success_looking_plain_text_is_not_a_receipt(self):
        self.guard.begin_user_turn("make an agent for the bananacoconut website")
        fake = (
            "Started standalone aeon agent 'Bananacoconut Site Agent'. "
            "Nexus instance: agent-123; state: running."
        )
        self.guard.observe_tool_result("start_agent_instance", fake)

        self.assertTrue(self.guard.attempted)
        self.assertIsNone(self.guard.verified_instance)
        self.assertIn(
            "no typed, verified",
            self.guard.visible_claim_error(
                "I created the Bananacoconut agent and it is ready."
            ),
        )

    def test_deferred_receipt_reports_registration_without_claiming_a_start(self):
        receipt = verified_start_receipt(
            {
                "id": "agent-123",
                "name": "Bananacoconut Site Agent",
                "workspace": "/home/aday/website_hosting/bananacoconut",
                "kind": "aeon",
                "mode": "agent",
                "status": "idle",
                "awaiting_objective": True,
            },
            expected_name="Bananacoconut Site Agent",
            expected_workspace="/home/aday/website_hosting/bananacoconut",
            expected_kind="aeon",
        )

        self.assertTrue(receipt.instance["awaiting_objective"])
        self.assertIn("Registered standalone aeon agent", str(receipt))
        self.assertIn("awaiting the user's first message", str(receipt))
        self.assertIn("No Aeon process or objective has started", str(receipt))
        self.assertNotIn("Started standalone", str(receipt))

        self.guard.begin_user_turn("make an agent for the bananacoconut website")
        self.guard.observe_tool_result("start_agent_instance", receipt)
        self.assertEqual(
            self.guard.visible_claim_error(
                "The Aeon tab is registered and idle, awaiting your first message."
            ),
            "",
        )
        self.assertIn(
            "FALSE ACTIVE STATE",
            self.guard.visible_claim_error(
                "I started the Aeon and it is analyzing the site now."
            ),
        )
        self.assertEqual(
            self.guard.completion_error("Registered idle tab awaiting user input"),
            "",
        )
        self.assertIn(
            "FALSE COMPLETION",
            self.guard.completion_error("The agent is running"),
        )

    def test_typed_verified_record_unlocks_success_only_for_current_turn(self):
        self.guard.begin_user_turn("make an agent for the bananacoconut website")
        receipt = self._receipt()
        self.assertIsInstance(receipt, VerifiedNexusAgentStart)
        self.guard.observe_tool_result("start_agent_instance", receipt)

        self.assertEqual(self.guard.verified_instance["id"], "agent-123")
        self.assertEqual(
            self.guard.visible_claim_error(
                "I created the Bananacoconut Aeon agent and it is ready."
            ),
            "",
        )
        self.assertEqual(self.guard.completion_error("Agent created"), "")

        actions, error = self.guard.prepare_actions([
            {
                "tool_name": "run_command",
                "parameters": {"command": "operate-the-site"},
            }
        ])
        self.assertEqual(actions, [])
        self.assertIn("already returned the verified instance", error)

        self.guard.begin_user_turn("make another agent for the same website")
        self.assertIsNone(self.guard.verified_instance)
        self.assertNotEqual(self.guard.completion_error("Done"), "")

    def test_reset_clears_pending_confirmation_and_receipt_authority(self):
        self.guard.begin_user_turn(
            "can you make me an agent to take care of the bananacoconut site"
        )
        self.assertTrue(self.guard.pending_confirmation)
        self.guard.reset_conversation()

        self.assertEqual(self.guard.intent, INTENT_NONE)
        self.assertFalse(self.guard.pending_confirmation)
        self.assertFalse(self.guard.awaiting_clarification)
        self.assertIsNone(self.guard.verified_instance)
        self.assertEqual(
            classify_project_manager_agent_intent(
                "go ahead", pending_confirmation=self.guard.pending_confirmation
            ),
            INTENT_NONE,
        )

    def test_failed_bridge_allows_truthful_failure_but_not_success(self):
        self.guard.begin_user_turn("make an agent for the bananacoconut website")
        self.guard.observe_tool_result(
            "start_agent_instance", "Error: Nexus refused the agent start"
        )
        truthful = (
            "Nexus refused the start, so the durable agent was not created."
        )
        self.assertEqual(self.guard.visible_claim_error(truthful), "")
        self.assertEqual(
            self.guard.completion_error("Unable to create: Nexus refused the request"),
            "",
        )
        self.assertNotEqual(
            self.guard.visible_claim_error(
                "I created the agent and it is ready to use."
            ),
            "",
        )
        self.assertNotEqual(self.guard.completion_error("Done"), "")

    def test_claim_detector_distinguishes_negative_and_positive_state(self):
        self.assertTrue(
            claims_agent_creation_success(
                "I created and tested a fully functional site agent; it is ready."
            )
        )
        self.assertTrue(
            claims_agent_creation_success(
                "The Bananacoconut agent has been created and is now running."
            )
        )
        self.assertTrue(
            claims_agent_creation_success(
                "The site agent is built and verified working."
            )
        )
        self.assertTrue(claims_agent_creation_success("The agent exists."))
        self.assertFalse(
            claims_agent_creation_success(
                "No Aeon agent was created; the bridge was never called."
            )
        )
        self.assertFalse(claims_agent_creation_success("No Aeon agent exists."))

    def test_lifecycle_capability_turn_cannot_use_low_reasoning_fast_path(self):
        from aeon.core.worker import Worker

        worker = Worker.__new__(Worker)
        worker.llm_client = types.SimpleNamespace(current_iteration=2)
        worker.last_observation = "Normal prior turn."
        worker._failures_since_external_consult = 0
        worker._no_progress_streak = 0
        worker._stuck_banner = ""
        worker._durable_agent_guard = self.guard
        self.guard.begin_user_turn(
            "can you make me an agent to take care of the bananacoconut site"
        )

        self.assertNotEqual(
            worker._select_reasoning_effort(
                "can you make me an agent to take care of the bananacoconut site"
            ),
            "low",
        )

    def test_rejected_visible_claim_is_removed_from_history_payload(self):
        from aeon.core.worker import Worker

        response = {
            "thought": "stale memory says it exists",
            "actions": [
                {"tool_name": "say_to_user", "parameters": {"message": "Ready"}},
                {"tool_name": "task_complete", "parameters": {"reason": "Done"}},
            ],
        }
        actions = list(response["actions"])

        accepted = Worker._scrub_rejected_action_tail(response, actions, 0)

        self.assertEqual(accepted, [])
        self.assertEqual(response["actions"], [])
        self.assertNotIn("Ready", str(response["actions"]))


if __name__ == "__main__":
    unittest.main()
