#!/usr/bin/env python3
"""Real mid-task /eta test: drive pi over RPC, steer /skill:eta into a running turn.

usage: rpc_eta.py <provider> <model> <label> [steer_after_tool_calls=3]
"""
import json, os, shutil, subprocess, sys, threading, time

prov, model, label = sys.argv[1:4]
after = int(sys.argv[4]) if len(sys.argv) > 4 else 3
V = os.path.expanduser("~/projects/eta/verify")
W = f"{V}/work-{label}"
shutil.rmtree(W, ignore_errors=True)
os.makedirs(W)
shutil.copy(f"{V}/TASK.md", W)
env = dict(os.environ, PI_CODING_AGENT_DIR=f"{V}/pi-agent", PI_SKIP_VERSION_CHECK="1",
           PI_TELEMETRY="0", ETA_HOME=f"{V}/eta-home-{label}")
shutil.rmtree(env["ETA_HOME"], ignore_errors=True)

p = subprocess.Popen(["pi", "--mode", "rpc", "--provider", prov, "--model", model], cwd=W, env=env,
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def send(obj):
    p.stdin.write((json.dumps(obj) + "\n").encode())
    p.stdin.flush()


t0 = time.time()
stamp = lambda: f"[{time.time() - t0:6.1f}s]"
tools = 0
steered = False
ends = 0
calls_after_eta = []
eta_cmds = []
send({"type": "prompt", "message": "Do the whole task in TASK.md, start to finish."})
deadline = time.time() + 900
for raw in p.stdout:
    if time.time() > deadline:
        print("TIMEOUT")
        break
    try:
        ev = json.loads(raw)
    except json.JSONDecodeError:
        continue
    t = ev.get("type")
    if t == "tool_execution_start":
        args = json.dumps(ev.get("args") or ev.get("input") or {})[:110]
        print(stamp(), "tool", ev.get("toolName"), args)
        if steered:
            calls_after_eta.append(ev.get("toolName"))
            if "eta.py" in args:
                eta_cmds.append(args)
    elif t == "tool_execution_end":
        tools += 1
        if tools == after and not steered:
            print(stamp(), ">>> steering /skill:eta into the running turn")
            send({"type": "steer", "message": "/skill:eta"})
            steered = True
    elif t == "message_end":
        m = ev.get("message") or {}
        if m.get("role") == "assistant":
            for b in m.get("content") or []:
                if b.get("type") == "text" and b.get("text", "").strip():
                    print(stamp(), "SAYS:", b["text"].strip().replace("\n", "\n            "))
    elif t == "agent_end":
        ends += 1
        print(stamp(), "agent_end")
        if ends == 1:
            send({"type": "prompt", "message": "Thanks. Reply with just: ok"})
        else:
            break
p.stdin.close()
p.wait(timeout=30)

print("\n=== RESULT", label)
print("eta.py calls after steer:", len(eta_cmds))
print("tool calls after the ETA commands:", max(0, len(calls_after_eta) - len(eta_cmds)))
print("files:", sorted(os.listdir(W)))
r = subprocess.run(["python3", "-m", "unittest"], cwd=W, capture_output=True, text=True)
print("tests:", (r.stderr.strip().splitlines() or ["?"])[-1])
pred = f"{env['ETA_HOME']}/predictions.jsonl"
if os.path.exists(pred):
    for line in open(pred):
        d = json.loads(line)
        print("logged:", {k: d[k] for k in ("kind", "task", "calls", "spc", "pace_source", "raw_s", "est_s", "note")})
else:
    print("logged: NOTHING")
