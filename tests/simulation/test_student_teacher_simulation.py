"""Student-AI teacher simulation tests.

This module simulates game scenarios and verifies that the trigger engine
responds appropriately with coaching feedback.
"""

import sys
import time
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from trigger.engine import TriggerEngine


def create_test_state(**kwargs) -> dict[str, Any]:
    """Create a game state dict from keyword arguments.

    Args:
        **kwargs: Field paths and values (e.g., player_status_hp=25)

    Returns:
        Complete game state dictionary
    """
    state = {
        "player": {
            "status": {
                "hp": {
                    "value": 100,
                    "source": "test",
                    "confidence": 1.0,
                    "ts_ms": int(time.time() * 1000),
                },
                "shield": {
                    "value": 0,
                    "source": "test",
                    "confidence": 1.0,
                    "ts_ms": int(time.time() * 1000),
                },
                "is_knocked": {
                    "value": False,
                    "source": "test",
                    "confidence": 1.0,
                    "ts_ms": int(time.time() * 1000),
                },
            },
            "weapon": {
                "name": {
                    "value": None,
                    "source": "test",
                    "confidence": 0.0,
                    "ts_ms": int(time.time() * 1000),
                },
                "ammo": {
                    "value": None,
                    "source": "test",
                    "confidence": 0.0,
                    "ts_ms": int(time.time() * 1000),
                },
            },
            "inventory": {
                "materials": {
                    "value": None,
                    "source": "test",
                    "confidence": 0.0,
                    "ts_ms": int(time.time() * 1000),
                },
            },
        },
        "world": {
            "storm": {
                "phase": {
                    "value": None,
                    "source": "test",
                    "confidence": 0.0,
                    "ts_ms": int(time.time() * 1000),
                },
                "damage": {
                    "value": None,
                    "source": "test",
                    "confidence": 0.0,
                    "ts_ms": int(time.time() * 1000),
                },
                "in_storm": {
                    "value": False,
                    "source": "test",
                    "confidence": 1.0,
                    "ts_ms": int(time.time() * 1000),
                },
                "is_shrinking": {
                    "value": False,
                    "source": "test",
                    "confidence": 1.0,
                    "ts_ms": int(time.time() * 1000),
                },
                "next_circle_distance": {
                    "value": None,
                    "source": "test",
                    "confidence": 0.0,
                    "ts_ms": int(time.time() * 1000),
                },
            },
        },
        "session": {
            "phase": {
                "value": None,
                "source": "test",
                "confidence": 0.0,
                "ts_ms": int(time.time() * 1000),
            },
            "inactivity_duration_ms": {
                "value": 0,
                "source": "test",
                "confidence": 1.0,
                "ts_ms": int(time.time() * 1000),
            },
        },
    }

    # Apply custom values from kwargs
    for key, value in kwargs.items():
        # Convert snake_case key to dot path
        parts = key.split("_")
        if len(parts) < 2:
            continue

        # Navigate to the correct location
        if parts[0] == "player" and len(parts) >= 3:
            category = parts[1]  # status, weapon, inventory
            field = "_".join(parts[2:])  # hp, is_knocked, etc.
            if field in state["player"][category]:
                state["player"][category][field]["value"] = value
        elif parts[0] == "world" and len(parts) >= 2:
            if parts[1] == "storm" and len(parts) >= 3:
                field = "_".join(parts[2:])  # in_storm, is_shrinking, etc.
                if field in state["world"]["storm"]:
                    state["world"]["storm"][field]["value"] = value
        elif parts[0] == "session" and len(parts) >= 2:
            field = "_".join(parts[1:])  # phase, inactivity_duration_ms
            if field in state["session"]:
                state["session"][field]["value"] = value

    return state


def test_combat_low_hp() -> dict[str, Any]:
    """Test Combat: Low HP (HP=25) trigger.

    Expected: p0_low_hp trigger fires with combat template.
    """
    print("\n" + "=" * 60)
    print("TEST: Combat - Low HP (HP=25)")
    print("=" * 60)

    state = create_test_state(player_status_hp=25)
    movement_state = "combat"

    engine = TriggerEngine("/Users/kawai/dev/game-study/configs/triggers.yaml")
    result = engine.evaluate_triggers(state, movement_state)

    expected_rule = "p0_low_hp"
    expected_response = "Low HP! Find cover immediately!"

    passed = (
        result is not None
        and result.get("rule_id") == expected_rule
        and expected_response in result.get("template", "")
    )

    print(f"State: HP={state['player']['status']['hp']['value']}")
    print(f"Movement: {movement_state}")
    print(f"Expected Trigger: {expected_rule}")
    print(f"Actual Trigger: {result.get('rule_id') if result else 'None'}")
    print(f"Response: {result.get('template') if result else 'None'}")
    print(f"Status: {'PASS' if passed else 'FAIL'}")

    return {
        "scenario": "Combat - Low HP",
        "passed": passed,
        "trigger": result.get("rule_id") if result else None,
        "response": result.get("template") if result else None,
        "expected_trigger": expected_rule,
    }


def test_combat_knocked() -> dict[str, Any]:
    """Test Combat: Knocked (is_knocked=True) trigger.

    Expected: p0_knocked trigger fires with combat template.
    """
    print("\n" + "=" * 60)
    print("TEST: Combat - Knocked")
    print("=" * 60)

    state = create_test_state(player_status_is_knocked=True)
    movement_state = "combat"

    engine = TriggerEngine("/Users/kawai/dev/game-study/configs/triggers.yaml")
    result = engine.evaluate_triggers(state, movement_state)

    expected_rule = "p0_knocked"
    expected_response = "You're knocked! Ping your location for your teammates!"

    passed = (
        result is not None
        and result.get("rule_id") == expected_rule
        and expected_response in result.get("template", "")
    )

    print(f"State: is_knocked={state['player']['status']['is_knocked']['value']}")
    print(f"Movement: {movement_state}")
    print(f"Expected Trigger: {expected_rule}")
    print(f"Actual Trigger: {result.get('rule_id') if result else 'None'}")
    print(f"Response: {result.get('template') if result else 'None'}")
    print(f"Status: {'PASS' if passed else 'FAIL'}")

    return {
        "scenario": "Combat - Knocked",
        "passed": passed,
        "trigger": result.get("rule_id") if result else None,
        "response": result.get("template") if result else None,
        "expected_trigger": expected_rule,
    }


def test_combat_in_storm() -> dict[str, Any]:
    """Test Combat: In Storm (in_storm=True) trigger.

    Expected: p0_storm_damage trigger fires with combat template.
    """
    print("\n" + "=" * 60)
    print("TEST: Combat - In Storm")
    print("=" * 60)

    state = create_test_state(world_storm_in_storm=True)
    movement_state = "combat"

    engine = TriggerEngine("/Users/kawai/dev/game-study/configs/triggers.yaml")
    result = engine.evaluate_triggers(state, movement_state)

    expected_rule = "p0_storm_damage"
    expected_response = "Get out of the storm! Now!"

    passed = (
        result is not None
        and result.get("rule_id") == expected_rule
        and expected_response in result.get("template", "")
    )

    print(f"State: in_storm={state['world']['storm']['in_storm']['value']}")
    print(f"Movement: {movement_state}")
    print(f"Expected Trigger: {expected_rule}")
    print(f"Actual Trigger: {result.get('rule_id') if result else 'None'}")
    print(f"Response: {result.get('template') if result else 'None'}")
    print(f"Status: {'PASS' if passed else 'FAIL'}")

    return {
        "scenario": "Combat - In Storm",
        "passed": passed,
        "trigger": result.get("rule_id") if result else None,
        "response": result.get("template") if result else None,
        "expected_trigger": expected_rule,
    }


def test_tactical_storm_shrinking() -> dict[str, Any]:
    """Test Tactical: Storm Shrinking (is_shrinking=True) trigger.

    Expected: p1_storm_shrinking trigger fires with non_combat template.
    """
    print("\n" + "=" * 60)
    print("TEST: Tactical - Storm Shrinking")
    print("=" * 60)

    state = create_test_state(world_storm_is_shrinking=True)
    movement_state = "non_combat"

    engine = TriggerEngine("/Users/kawai/dev/game-study/configs/triggers.yaml")
    result = engine.evaluate_triggers(state, movement_state)

    expected_rule = "p1_storm_shrinking"
    expected_response = "The storm is shrinking. Time to move to the safe zone."

    passed = (
        result is not None
        and result.get("rule_id") == expected_rule
        and expected_response in result.get("template", "")
    )

    print(f"State: is_shrinking={state['world']['storm']['is_shrinking']['value']}")
    print(f"Movement: {movement_state}")
    print(f"Expected Trigger: {expected_rule}")
    print(f"Actual Trigger: {result.get('rule_id') if result else 'None'}")
    print(f"Response: {result.get('template') if result else 'None'}")
    print(f"Status: {'PASS' if passed else 'FAIL'}")

    return {
        "scenario": "Tactical - Storm Shrinking",
        "passed": passed,
        "trigger": result.get("rule_id") if result else None,
        "response": result.get("template") if result else None,
        "expected_trigger": expected_rule,
    }


def test_learning_weapon_pickup() -> dict[str, Any]:
    """Test Learning: Weapon pickup (new_weapon_detected=True) trigger.

    Expected: p2_weapon_learning trigger fires with non_combat template.
    """
    print("\n" + "=" * 60)
    print("TEST: Learning - Weapon Pickup")
    print("=" * 60)

    # Create state and manually add the new_weapon_detected field
    state = create_test_state(player_weapon_name="Assault Rifle")
    # Add new_weapon_detected field to weapon dict
    state["player"]["weapon"]["new_weapon_detected"] = {
        "value": True,
        "source": "test",
        "confidence": 1.0,
        "ts_ms": int(time.time() * 1000),
    }
    movement_state = "non_combat"

    engine = TriggerEngine("/Users/kawai/dev/game-study/configs/triggers.yaml")
    result = engine.evaluate_triggers(state, movement_state)

    expected_rule = "p2_weapon_learning"

    passed = (
        result is not None
        and result.get("rule_id") == expected_rule
        and "weapon_name" in result.get("template", "")
    )

    print("State: new_weapon_detected=True")
    print(f"Movement: {movement_state}")
    print(f"Expected Trigger: {expected_rule}")
    print(f"Actual Trigger: {result.get('rule_id') if result else 'None'}")
    print(f"Response: {result.get('template') if result else 'None'}")
    print(f"Status: {'PASS' if passed else 'FAIL'}")

    return {
        "scenario": "Learning - Weapon Pickup",
        "passed": passed,
        "trigger": result.get("rule_id") if result else None,
        "response": result.get("template") if result else None,
        "expected_trigger": expected_rule,
    }


def test_conversation_inactivity_timeout() -> dict[str, Any]:
    """Test Conversation: Inactivity timeout trigger.

    Expected: p3_small_talk trigger fires when inactivity > 30s.
    """
    print("\n" + "=" * 60)
    print("TEST: Conversation - Inactivity Timeout")
    print("=" * 60)

    # Create engine and simulate time passing
    engine = TriggerEngine("/Users/kawai/dev/game-study/configs/triggers.yaml")

    # Simulate 35 seconds of inactivity
    state = create_test_state(session_inactivity_duration_ms=35000)
    movement_state = "non_combat"

    result = engine.evaluate_triggers(state, movement_state)

    expected_rule = "p3_small_talk"
    expected_response = "How's it going?"

    passed = (
        result is not None
        and result.get("rule_id") == expected_rule
        and expected_response in result.get("template", "")
    )

    print(f"State: inactivity_duration_ms={state['session']['inactivity_duration_ms']['value']}")
    print(f"Movement: {movement_state}")
    print(f"Expected Trigger: {expected_rule}")
    print(f"Actual Trigger: {result.get('rule_id') if result else 'None'}")
    print(f"Response: {result.get('template') if result else 'None'}")
    print(f"Status: {'PASS' if passed else 'FAIL'}")

    return {
        "scenario": "Conversation - Inactivity Timeout",
        "passed": passed,
        "trigger": result.get("rule_id") if result else None,
        "response": result.get("template") if result else None,
        "expected_trigger": expected_rule,
    }


def run_all_simulation_tests() -> list[dict[str, Any]]:
    """Run all simulation tests and return results.

    Returns:
        List of test result dictionaries
    """
    print("\n" + "=" * 60)
    print("STUDENT-AI TEACHER SIMULATION TESTS")
    print("=" * 60)

    tests = [
        test_combat_low_hp,
        test_combat_knocked,
        test_combat_in_storm,
        test_tactical_storm_shrinking,
        test_learning_weapon_pickup,
        test_conversation_inactivity_timeout,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"ERROR in {test.__name__}: {e}")
            results.append(
                {
                    "scenario": test.__name__,
                    "passed": False,
                    "error": str(e),
                }
            )

    return results


def print_summary(results: list[dict[str, Any]]) -> None:
    """Print test summary.

    Args:
        results: List of test result dictionaries
    """
    print("\n" + "=" * 60)
    print("SIMULATION TEST SUMMARY")
    print("=" * 60)

    total = len(results)
    passed = sum(1 for r in results if r.get("passed", False))

    print(f"\nTotal Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed/total*100):.1f}%")

    print("\nDetailed Results:")
    print("-" * 60)
    for result in results:
        status = "PASS" if result.get("passed", False) else "FAIL"
        print(f"{status:6} | {result.get('scenario', 'Unknown')}")

        if not result.get("passed", False):
            if "error" in result:
                print(f"       Error: {result['error']}")
            else:
                print(f"       Expected: {result.get('expected_trigger', 'N/A')}")
                print(f"       Got: {result.get('trigger', 'None')}")

    print("-" * 60)


if __name__ == "__main__":
    results = run_all_simulation_tests()
    print_summary(results)

    # Exit with appropriate code
    all_passed = all(r.get("passed", False) for r in results)
    sys.exit(0 if all_passed else 1)
