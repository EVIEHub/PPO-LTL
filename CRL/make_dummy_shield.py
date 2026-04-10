"""
Generate shield_table.pkl for CarlaShieldWrapper.

Run from the CRL/ directory:
    python make_dummy_shield.py

Abstract state z = (lane_bin, speed_bin, red_bin, front_bin)
  lane_bin : 0=good, 1=warn, 2=bad
  speed_bin: 0=slow, 1=mid,  2=fast
  red_bin  : 0=green, 1=red
  front_bin: 0=close, 1=near, 2=far

discrete_actions format: {action_id: [throttle, steer]}
  throttle in [-1.0, 1.0]  (negative = brake)
  steer    in [-1.0, 1.0]  (negative = left)
"""

import pickle
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from carla_env.envs.carla_route_env import discrete_actions


def build_shield_table(actions: dict) -> dict:
    table = {}

    for lane_bin in [0, 1, 2]:
        for speed_bin in [0, 1, 2]:
            for red_bin in [0, 1]:
                for front_bin in [0, 1, 2]:
                    z = (lane_bin, speed_bin, red_bin, front_bin)
                    safe_ids = []

                    for aid, (throttle, steer) in actions.items():
                        safe = True

                        # Rule 1: Red light -> no acceleration
                        if red_bin == 1 and throttle > 0:
                            safe = False

                        # Rule 2: Close front vehicle -> no hard acceleration
                        if front_bin == 0 and throttle > 0.3:
                            safe = False

                        # Rule 3: Severe lane deviation -> no large steer
                        # (counterintuitive large steer may worsen deviation)
                        if lane_bin == 2 and abs(steer) > 0.5:
                            safe = False

                        # Rule 4: High speed -> no sharp steering
                        if speed_bin == 2 and abs(steer) > 0.7:
                            safe = False

                        if safe:
                            safe_ids.append(aid)

                    # Fallback: always allow action 0 (full brake) to avoid deadlock
                    if not safe_ids:
                        safe_ids = [0]

                    table[z] = safe_ids

    return table


if __name__ == "__main__":
    table = build_shield_table(discrete_actions)

    out_path = "shield_table.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(table, f)

    n_states = len(table)
    avg_safe = sum(len(v) for v in table.values()) / n_states
    print(f"Generated {out_path}")
    print(f"  Total states    : {n_states}")
    print(f"  Total actions   : {len(discrete_actions)}")
    print(f"  Avg safe actions: {avg_safe:.1f} per state")

    # Quick sanity check
    red_states = [(z, ids) for z, ids in table.items() if z[2] == 1]
    all_throttles = [discrete_actions[aid][0] for z, ids in red_states for aid in ids]
    assert all(t <= 0 for t in all_throttles), "Rule 1 violated: throttle > 0 allowed on red!"
    print("  Sanity check passed: no acceleration allowed on red light.")
