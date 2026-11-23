# /eta on open models (2026-11-23)

Harness: pi 0.73.1 with an isolated config (`PI_CODING_AGENT_DIR=verify/pi-agent`), skill loaded from the real install at `~/.agents/skills/eta`. Models served locally on an M5 Pro (48 GB) with llama.cpp (`llama-server --jinja`). No GPU rental, no paid API.

Two test styles:
- `run.sh`: three separate `pi -p` turns (partial task, then `/skill:eta`, then a follow-up).
- `rpc_eta.py`: the real case. The full task runs over pi RPC, and `/skill:eta` is **steered into the running turn** after the 3rd tool call.

| Run | Model | What happened | Verdict |
|---|---|---|---|
| run-gptoss | gpt-oss-20b | Ran `measure`, then guessed an ETA itself. Script bug: pi's `<skill name="eta">` prompt was taken as the task. | fixed (parser + clearer next step) |
| run-gptoss-2/3 | gpt-oss-20b | `measure` + `estimate`, correct reply shape, logged to the right pi session; then stopped | pass (procedure) |
| run-qwen | Qwen3-Coder-30B-A3B | Treated the skill as "continue" and finished the task without any ETA | fixed (command moved to the top of SKILL.md) |
| run-qwen-2 | Qwen3-Coder-30B-A3B | Task already done; ran `measure`, then copied the **example numbers** from SKILL.md | fixed (placeholders, explicit "Already done.") |
| run-qwen-rpc | Qwen3-Coder-30B-A3B | Ignored the mid-turn `/eta`, finished, then ran `done` and said "Already done." | model limitation |
| run-gptoss-rpc | gpt-oss-20b | Mid-turn: `measure`, `estimate`, correct ETA reply with real numbers, then ended its turn with a test failing | pass (ETA), fail (continue) |
| run-gptoss-rpc2 | gpt-oss-20b | Mid-turn: `measure`, `estimate`, **kept working** (tests, fix, tests), finished with all tests passing; ETA text posted at the end of the turn | pass (ETA + continue), ETA text late |

Script checks, all on real pi transcripts: session found through `PI_CODING_AGENT_DIR`; the task is the right prompt; `/skill:eta` detected as an ETA request; predictions logged with `kind: pi`; an agent stopping right after answering is resolved as `inconclusive` and left out of calibration.

Earlier noted, not fixed: the MLX server (mlx-lm 0.32.2) returns Qwen3-Coder tool calls as raw text, so use llama.cpp for these models.
