#!/usr/bin/env python3
"""
Replay Server - Play back recorded trajectories on the Panda robot.
After recording via web teleop, run this to watch the robot replay autonomously.
"""
import os, sys, json, time
from pathlib import Path

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
from flask import Flask, render_template_string, request, jsonify
import threading, io, base64
from PIL import Image
from scipy.spatial.transform import Rotation

app = Flask(__name__)

import sys
sys.path.insert(0, "/data/coding/lingbot-vla/vla_project")
from vla_project.pick_place_box import PickPlaceBox

DATA_DIR = Path("/data/coding/lingbot-vla/vla_project/teleop_data")
env = None
obs = None
frame_b64 = ""
frame_lock = threading.Lock()
running = True

# Replay state
trajectory = []
replay_idx = 0
is_replaying = False
replay_done = False
episode_num = "?"
total_steps = 0

# Camera - use agentview which is most reliable
current_camera = "agentview"


def init_env():
    global env, obs
    if env is not None:
        env.close()
    env = PickPlaceBox(
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        control_freq=20,
        horizon=10000,
        reward_shaping=True,
    )
    obs = env.reset()


def render():
    global env
    if env is None:
        return ""
    try:
        img = env.sim.render(width=480, height=360, camera_name=current_camera)
        img = np.flipud(img)
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=40)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print(f"[Render Error] {e}", flush=True)
        return ""


def sim_loop():
    global obs, env, frame_b64, replay_idx, is_replaying, replay_done
    # Create environment in THIS thread so EGL context is bound to the render thread
    init_env()
    dt = 0.08
    while running:
        t0 = time.time()

        if env and obs is not None and is_replaying:
            if replay_idx == 0:
                # Reset at start of replay (in render thread for GL context)
                obs = env.reset()
            if replay_idx < len(trajectory):
                action = np.array(trajectory[replay_idx]["action"])
                try:
                    obs, reward, done, info = env.step(action)
                    replay_idx += 1
                    if done:
                        obs = env.reset()
                except Exception as e:
                    print(f"[Step Error] {e}", flush=True)
                    replay_idx += 1
            else:
                is_replaying = False
                replay_done = True

        with frame_lock:
            frame_b64 = render()
        time.sleep(max(0, dt - (time.time() - t0)))


# ---- Flask Routes ----
@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/frame")
def get_frame():
    global replay_idx, total_steps, is_replaying, replay_done, episode_num
    with frame_lock:
        f = frame_b64
    eef = obs["robot0_eef_pos"].tolist() if obs is not None else [0, 0, 0]
    status = "Replaying" if is_replaying else ("Done!" if replay_done else "Idle")
    return jsonify({
        "image": f,
        "step": f"{replay_idx}/{total_steps}",
        "eef": [round(x, 3) for x in eef],
        "status": status,
        "episode": episode_num,
    })


@app.route("/replay/start", methods=["POST"])
def start_replay():
    global is_replaying, replay_idx, replay_done
    data = request.get_json() or {}
    filename = data.get("filename", "")

    loaded = load_trajectory(filename)
    if not loaded:
        return jsonify({"ok": False, "error": "No trajectory loaded"})

    # Set flags only - sim_loop thread will handle env.reset()
    # (calling env.reset() here from Flask main thread would steal GL context)
    replay_idx = 0
    replay_done = False
    is_replaying = True
    return jsonify({"ok": True, "steps": total_steps})


@app.route("/replay/stop", methods=["POST"])
def stop_replay():
    global is_replaying
    is_replaying = False
    return jsonify({"ok": True})


@app.route("/replay/list", methods=["GET"])
def list_recordings():
    files = sorted(DATA_DIR.glob("ep_*.json"), key=os.path.getmtime, reverse=True)
    result = []
    for f in files[:20]:
        try:
            with open(f) as fh:
                d = json.load(fh)
            result.append({
                "name": f.name,
                "episode": d.get("episode", "?"),
                "steps": d.get("steps", 0),
            })
        except:
            result.append({"name": f.name, "episode": "?", "steps": 0})
    return jsonify(result)


def load_trajectory(filename=None):
    global trajectory, total_steps, episode_num
    if filename:
        path = DATA_DIR / filename
    else:
        files = sorted(DATA_DIR.glob("ep_*.json"), key=os.path.getmtime, reverse=True)
        if not files:
            print("No recorded trajectories found!")
            return False
        path = files[0]

    with open(path) as f:
        data = json.load(f)
    trajectory = data["trajectory"]
    total_steps = len(trajectory)
    episode_num = data.get("episode", "?")
    print(f"Loaded {path.name}: {total_steps} steps, episode {episode_num}")
    return True


# ---- HTML ----
HTML = r"""<!DOCTYPE html><html lang="zh-CN">
<head><meta charset="UTF-8"><title>Replay - Panda Robot</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d1117;color:#c9d1d9;font-family:system-ui;text-align:center;overflow:hidden}
.top{background:#161b22;padding:12px;border-bottom:2px solid #30363d}
.top h1{font-size:1.2em;color:#58a6ff}
.row{display:flex;gap:15px;padding:15px;max-width:1100px;margin:0 auto;flex-wrap:wrap;justify-content:center}
.vid{background:#161b22;border-radius:8px;padding:6px;border:1px solid #30363d}
.vid img{width:480px;height:360px;border-radius:4px;background:#000}
.panel{background:#161b22;border-radius:8px;padding:18px;min-width:260px;border:1px solid #30363d;text-align:left}
.panel h3{color:#58a6ff;margin-bottom:10px;font-size:.95em}
.s{font-size:.85em;margin:5px 0}.s b{color:#7ee787}
.btn{padding:10px 18px;margin:6px 6px 6px 0;border:none;border-radius:6px;cursor:pointer;font-weight:bold;font-size:.85em}
.play{background:#238636;color:#fff}.stop{background:#da3633;color:#fff}
.list-btn{background:#1f6feb;color:#fff}
#msg{font-size:.8em;color:#f78166;margin-top:6px}
.recording-list{background:#0d1117;padding:8px;border-radius:6px;margin-top:10px;font-size:.78em;max-height:300px;overflow-y:auto}
.recording-list div{padding:4px 8px;cursor:pointer;border-radius:3px;margin:2px 0}
.recording-list div:hover{background:#21262d}
.recording-list .selected{background:#1f6feb;color:#fff}
</style></head>
<body>
<div class="top"><h1>Panda Robot - Trajectory Replay</h1><div id="msg">Loading...</div></div>
<div class="row">
<div class="vid"><img id="feed" src=""></div>
<div class="panel">
<h3>Status</h3>
<div class="s">Episode: <b id="sEp">-</b></div>
<div class="s">Progress: <b id="sStep">-</b></div>
<div class="s">EEF: <b id="sPos">-</b></div>
<div class="s">Status: <b id="sStatus">-</b></div>
<div style="margin-top:12px">
<button class="btn play" onclick="doReplay()">Play (Latest)</button>
<button class="btn stop" onclick="doStop()">Stop</button>
<button class="btn list-btn" onclick="loadList()">Refresh List</button>
</div>
<h3 style="margin-top:15px">Recordings</h3>
<div class="recording-list" id="recList">Click "Refresh List"</div>
</div></div>
<script>
var selectedFile = "";
function poll(){fetch('/frame').then(function(r){return r.json()}).then(function(d){
 feed.src='data:image/jpeg;base64,'+d.image;
 document.getElementById('sStep').textContent=d.step;
 document.getElementById('sPos').textContent=d.eef.join(', ');
 document.getElementById('sStatus').textContent=d.status;
 document.getElementById('sEp').textContent=d.episode;
 document.getElementById('msg').textContent='Connected';
});setTimeout(poll,150)}
function doReplay(){
  var body = selectedFile ? JSON.stringify({filename:selectedFile}) : '{}';
  fetch('/replay/start',{method:'POST',headers:{'Content-Type':'application/json'},body:body})
    .then(function(r){return r.json()}).then(function(d){if(d.ok)document.getElementById('msg').textContent='Replaying '+d.steps+' steps';else alert(d.error);});
}
function doStop(){fetch('/replay/stop',{method:'POST'})}
function loadList(){
  fetch('/replay/list').then(function(r){return r.json()}).then(function(data){
    var html=''; for(var i=0;i<data.length;i++){
      html+='<div onclick=\"selectFile('+"'"+data[i].name+"'"+')\" id=\"f_'+i+'\">'+data[i].name+' ('+data[i].steps+' steps)</div>';
    }
    document.getElementById('recList').innerHTML=html||'(no recordings yet)';
    if(data.length>0){selectFile(data[0].name);document.getElementById('msg').textContent=data.length+' recordings found';}
  });
}
function selectFile(name){
  selectedFile=name;
  var divs=document.querySelectorAll('.recording-list div');
  for(var i=0;i<divs.length;i++){divs[i].classList.remove('selected');}
  document.getElementById('msg').textContent='Selected: '+name;
}
poll();
loadList();
</script></body></html>"""


if __name__ == "__main__":
    print("=" * 55)
    print("  Panda Robot Trajectory Replay")
    print("  Open: http://localhost:9091 (via SSH tunnel)")
    print("=" * 55)

    # init_env() is now called inside sim_loop thread for EGL context binding
    load_trajectory()

    threading.Thread(target=sim_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=9091, threaded=True, debug=False)
