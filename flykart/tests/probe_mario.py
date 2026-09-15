"""Compare fixed button inputs and inspect frames/RAM; no learning or reward design."""
import json
import os
from pathlib import Path

os.environ.pop("DISPLAY", None)
os.environ.pop("WAYLAND_DISPLAY", None)
os.environ["SDL_AUDIODRIVER"] = "dummy"
import cv2
import numpy as np
import stable_retro as retro

root = Path(__file__).resolve().parents[1]
out = root / "tests" / "mario_frames"
out.mkdir(exist_ok=True)
retro.data.Integrations.add_custom_path(str(root / "mario_kart"))
env = retro.make("SuperMarioKart-Snes", inttype=retro.data.Integrations.CUSTOM_ONLY,
                 render_mode="rgb_array")
fields = ["GameMode", "DrivingMode", "getGameMode", "lap", "current_checkpoint",
          "lapsize", "kart1_speed", "kart1_X", "kart1_Y", "surface", "isTurnedAround", "course"]
result = {}
try:
    initial, _ = env.reset()
    cv2.imwrite(str(out / "reset.png"), cv2.cvtColor(initial, cv2.COLOR_RGB2BGR))
    for name, buttons in [("no_buttons", []), ("accelerate", ["B"]),
                          ("accelerate_right", ["B", "RIGHT"])]:
        obs, _ = env.reset()
        assert np.array_equal(obs, initial)
        action = np.zeros(len(env.buttons), dtype=np.int8)
        for button in buttons:
            action[env.buttons.index(button)] = 1
        samples = []
        observed = {key: set() for key in fields}
        for step in range(600):
            obs, _, terminated, truncated, info = env.step(action)
            for key in fields:
                if key in info:
                    observed[key].add(int(info[key]))
            if step in [0, 119, 239, 359, 599] or terminated or truncated:
                samples.append({"step": step + 1, "terminated": terminated,
                                "truncated": truncated, **{k: info.get(k) for k in fields}})
                cv2.imwrite(str(out / f"{name}_{step+1}.png"), cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
            if terminated or truncated:
                break
        result[name] = {"samples": samples,
                        "ranges": {k: [min(v), max(v)] if v else None for k, v in observed.items()}}
    obs, _ = env.reset()
    assert np.array_equal(obs, initial)
    result["final_reset_restores_frame"] = True
    report = json.dumps(result, indent=2, default=int)
    (root / "tests" / "mario_probe.json").write_text(report + "\n")
    print(report)
finally:
    env.close()
