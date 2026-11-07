"""go1_play.py <run-dir> — drive a trained Go1 policy live in a window: arrows drive and turn, shift+arrows strafe,
backspace resets (2026-11-01). Runs the upstream mjbatch example's play loop (vendor, unmodified) on the run's policy.pt;
needs envs/mjbatch (mjbatch + glfw) and a GUI session; the window must own the main thread, so run it as its own process.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "vendor", "mjbatch", "examples"))
import go1_joystick as upstream  # noqa: E402  (imports window -> glfw)
run = sys.argv[1] if len(sys.argv) > 1 else "runs/go1-base"
upstream.POLICY = os.path.abspath(os.path.join(run, "policy.pt"))
print(f"driving {upstream.POLICY}: arrows drive/turn, shift+arrows strafe, backspace resets, close the window to stop")
upstream.play()
