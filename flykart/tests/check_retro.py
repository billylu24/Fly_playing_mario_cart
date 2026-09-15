"""Headless random-action smoke check; no training."""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.pop("DISPLAY", None)
os.environ.pop("WAYLAND_DISPLAY", None)
os.environ["SDL_AUDIODRIVER"] = "dummy"
import numpy as np
import stable_retro as retro

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--mario", action="store_true")
parser.add_argument("--steps", type=int, default=5000)
args = parser.parse_args()
game = "SuperMarioKart-Snes" if args.mario else "Airstriker-Genesis-v0"
if args.mario:
    retro.data.Integrations.add_custom_path(str(root / "mario_kart"))
options = {"inttype": retro.data.Integrations.CUSTOM_ONLY} if args.mario else {}
env = retro.make(game, render_mode="rgb_array", **options)
try:
    obs, info = env.reset(seed=42)
    initial = obs.copy()
    env.action_space.seed(42)
    result = {"game": game, "headless": True, "observation_shape": list(obs.shape),
              "observation_dtype": str(obs.dtype), "action_space": str(env.action_space),
              "buttons": env.buttons, "initial_info": info}
    resets = 1
    changed = 0
    started = time.monotonic()
    for step in range(args.steps):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert obs.shape == initial.shape and obs.dtype == np.uint8
        assert np.isfinite(reward)
        changed += int(not np.array_equal(initial, obs))
        if terminated or truncated or (step + 1) % 1000 == 0:
            obs, info = env.reset()
            assert np.array_equal(obs, initial), "Reset did not restore initial observation"
            resets += 1
    result.update(steps=args.steps, resets=resets, changed_frames=changed,
                  elapsed_seconds=round(time.monotonic() - started, 2),
                  final_info=info, reset_restores_initial_frame=True)
    frame = env.render()
    assert frame.shape == initial.shape
    report = json.dumps(result, indent=2, default=int)
    print(report)
    (root / "tests" / ("mario_smoke.json" if args.mario else "retro_smoke.json")).write_text(report + "\n")
finally:
    env.close()
