#!/usr/bin/env python3
"""
Panda Robot Web Teleoperation - Multi-camera + Mouse orbit
Open: http://eclfwzruav21i8stalk4090.funhpc.com:8080
"""
import os, sys, json, io, time, base64, threading, math
from pathlib import Path
from datetime import datetime

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
from PIL import Image
from flask import Flask, render_template_string, request, jsonify
from scipy.spatial.transform import Rotation
# Always use manual fallback (MuJoCo 3.8+ API changed)
try:
    from mujoco import mju_quat2Mat as _mujoco_mju_quat2Mat
    raise ImportError("Use manual fallback for MuJoCo 3.8+")
except ImportError:
    # Fallback: quat to matrix manually
    def mju_quat2Mat(quat, mat):
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        mat[0] = 1 - 2*(y*y + z*z); mat[1] = 2*(x*y - z*w);   mat[2] = 2*(x*z + y*w)
        mat[3] = 2*(x*y + z*w);   mat[4] = 1 - 2*(x*x + z*z); mat[5] = 2*(y*z - x*w)
        mat[6] = 2*(x*z - y*w);   mat[7] = 2*(y*z + x*w);     mat[8] = 1 - 2*(x*x + y*y)

app = Flask(__name__)

import sys
sys.path.insert(0, "/data/coding/lingbot-vla/vla_project")
from vla_project.pick_place_box import PickPlaceBox

env = None
obs = None
frame_b64 = ""
frame_lock = threading.Lock()
running = True

joint_vel = np.zeros(8)

# Multi-camera support
CAMERAS = ["frontview", "birdview", "agentview", "sideview", "robot0_robotview", "robot0_eye_in_hand"]
CAMERA_LABELS = {
    "frontview": "Front", "birdview": "Bird's Eye", "agentview": "Agent",
    "sideview": "Side", "robot0_robotview": "Robot", "robot0_eye_in_hand": "Eye-in-Hand"
}
current_camera = "frontview"

reset_flag = False
save_flag = False
step_count = 0
episode_count = 0
episode_states = []
is_recording = False

# Mouse orbit state
orbit_dx = 0.0
orbit_dy = 0.0
orbit_zoom = 0.0
orbit_changed = True  # track if orbit has changed since last render

# Camera offsets (per-camera accumulated orbit)
cam_orbit = {name: {"azimuth": 0.0, "elevation": 0.0, "distance": 0.0} for name in CAMERAS}
# Track last applied orbit values to skip redundant computation
_last_orbit = {name: {"azimuth": 0.0, "elevation": 0.0, "distance": 0.0} for name in CAMERAS}

SAVE_DIR = Path("/data/coding/lingbot-vla/vla_project/teleop_data")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# Store original camera states for reset
_cam_original = {}  # {name: {"pos": ..., "quat": ...}}

def init_env():
    global env, obs, _cam_original
    if env is not None:
        env.close()
    env = PickPlaceBox(
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        control_freq=20,
        horizon=2000,
        reward_shaping=True,
    )
    obs = env.reset()
    # Store original camera states
    _cam_original = {}
    for name in CAMERAS:
        cam_id = env.sim.model.camera_name2id(name)
        _cam_original[name] = {
            "pos": env.sim.model.cam_pos[cam_id].copy(),
            "quat": env.sim.model.cam_quat[cam_id].copy(),
        }


def _reset_cam_to_original(name):
    """Reset a specific camera to its original position/rotation."""
    if env is None or name not in _cam_original:
        return
    cam_id = env.sim.model.camera_name2id(name)
    env.sim.model.cam_pos[cam_id] = _cam_original[name]["pos"].copy()
    env.sim.model.cam_quat[cam_id] = _cam_original[name]["quat"].copy()
    env.sim.forward()


def _apply_orbit():
    """Apply accumulated orbit offsets to the current camera in MuJoCo."""
    global orbit_changed
    if env is None:
        return
    # Skip if orbit hasn't changed since last render
    if not orbit_changed:
        return
    orbit_changed = False
    cam_id = env.sim.model.camera_name2id(current_camera)
    offset = cam_orbit[current_camera]

    # Get original camera position and rotation
    orig_pos = _cam_original[current_camera]["pos"].copy()
    orig_quat = _cam_original[current_camera]["quat"].copy()  # w,x,y,z

    # Convert quaternion to rotation matrix
    orig_mat = np.zeros(9)
    mju_quat2Mat(orig_quat, orig_mat)
    orig_mat = orig_mat.reshape(3, 3)

    # Camera local axes: forward=-z, right=x, up=y
    forward = -orig_mat[:, 2]
    right = orig_mat[:, 0]
    up = orig_mat[:, 1]

    # Orbit: rotate camera position around target
    # Target is the original lookat point (ahead of camera)
    lookat_dist = np.linalg.norm(orig_pos) if np.linalg.norm(orig_pos) > 0.1 else 2.0
    effective_dist = lookat_dist + offset["distance"]
    effective_dist = max(0.5, min(8.0, effective_dist))

    az = math.radians(offset["azimuth"])
    el = math.radians(offset["elevation"])

    # Start from original camera direction
    cam_dir = -orig_pos / lookat_dist  # unit vector toward scene

    # Apply elevation (rotate around right axis)
    cos_el = math.cos(el)
    sin_el = math.sin(el)
    cam_dir_el = cos_el * cam_dir + sin_el * up
    cam_dir_el = cam_dir_el / np.linalg.norm(cam_dir_el)

    # Apply azimuth (rotate around world up)
    cos_az = math.cos(az)
    sin_az = math.sin(az)
    world_up = np.array([0, 0, 1.0])
    cam_dir_final = cos_az * cam_dir_el + sin_az * np.cross(world_up, cam_dir_el)
    cam_dir_final = cam_dir_final / np.linalg.norm(cam_dir_final)

    new_pos = -cam_dir_final * effective_dist

    # Update camera position
    env.sim.model.cam_pos[cam_id] = new_pos[:3]

    # Update camera lookat to face the origin
    target = np.zeros(3)
    look_dir = target - new_pos
    look_dir = look_dir / (np.linalg.norm(look_dir) + 1e-8)

    # Build rotation matrix: z=look_dir, x=right, y=up
    new_z = -look_dir
    new_x = np.cross(world_up, new_z)
    if np.linalg.norm(new_x) < 1e-6:
        new_x = np.array([1, 0, 0])
    new_x = new_x / np.linalg.norm(new_x)
    new_y = np.cross(new_z, new_x)
    new_y = new_y / np.linalg.norm(new_y)

    rot_mat = np.column_stack([new_x, new_y, new_z])
    new_quat = Rotation.from_matrix(rot_mat).as_quat()  # x,y,z,w
    env.sim.model.cam_quat[cam_id] = np.array([new_quat[3], new_quat[0], new_quat[1], new_quat[2]])  # w,x,y,z

    env.sim.forward()


def render():
    if env is None:
        return ""
    _apply_orbit()
    img = env.sim.render(camera_name=current_camera, width=480, height=360, depth=False)
    img = np.flipud(img)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=40)
    return base64.b64encode(buf.getvalue()).decode()


def sim_loop():
    global obs, env, frame_b64, step_count, reset_flag, save_flag
    global episode_states, is_recording, episode_count, joint_vel
    global current_camera, orbit_dx, orbit_dy, orbit_zoom
    dt = 0.08  # ~12.5 FPS (reduced from 20Hz to save CPU)
    while running:
        t0 = time.time()
        if env and obs is not None:
            if reset_flag:
                obs = env.reset()
                step_count = 0
                reset_flag = False
            if save_flag and episode_states:
                episode_count += 1
                fn = SAVE_DIR / f"ep_{episode_count}_{datetime.now().strftime('%m%d_%H%M')}.json"
                with open(fn, "w") as f:
                    json.dump({"episode": episode_count, "steps": len(episode_states),
                               "trajectory": episode_states}, f)
                print(f"[Save] {fn}")
                episode_states = []
                is_recording = False
                save_flag = False

            action = joint_vel.copy()
            try:
                obs, reward, done, info = env.step(action)
                step_count += 1
                if is_recording:
                    episode_states.append({
                        "step": step_count,
                        "eef_pos": obs["robot0_eef_pos"].tolist(),
                        "action": [round(float(a), 4) for a in action],
                    })
                if done:
                    obs = env.reset()
                    step_count = 0
            except:
                pass

        with frame_lock:
            frame_b64 = render()
        time.sleep(max(0, dt - (time.time() - t0)))


# API -----------------------------------------------------
@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/frame")
def get_frame():
    with frame_lock:
        f = frame_b64
    eef = obs["robot0_eef_pos"].tolist() if obs is not None else [0, 0, 0]
    return jsonify({
        "image": f, "step": step_count,
        "gripper": "Close" if joint_vel[7] > 0 else "Open",
        "eef": [round(x, 3) for x in eef], "saved": episode_count,
        "recording": is_recording,
        "camera": current_camera,
        "camera_label": CAMERA_LABELS.get(current_camera, current_camera),
    })


@app.route("/camera", methods=["POST"])
def switch_camera():
    global current_camera, orbit_changed
    data = request.get_json()
    name = data.get("name", "")
    if name in CAMERAS:
        # Reset orbit offsets for the new camera
        current_camera = name
        cam_orbit[name] = {"azimuth": 0.0, "elevation": 0.0, "distance": 0.0}
        orbit_changed = True
        # Reset camera to original position
        _reset_cam_to_original(name)
        return jsonify({"ok": True, "camera": name})
    return jsonify({"ok": False}), 400


@app.route("/camera/cycle", methods=["POST"])
def cycle_camera():
    global current_camera
    data = request.get_json()
    direction = data.get("direction", 1)  # 1=next, -1=prev
    idx = CAMERAS.index(current_camera)
    idx = (idx + direction) % len(CAMERAS)
    name = CAMERAS[idx]
    current_camera = name
    cam_orbit[name] = {"azimuth": 0.0, "elevation": 0.0, "distance": 0.0}
    _reset_cam_to_original(name)
    return jsonify({"ok": True, "camera": name, "label": CAMERA_LABELS.get(name, name)})


@app.route("/camera/orbit", methods=["POST"])
def camera_orbit():
    """Mouse drag orbit: delta_x, delta_y rotate camera; zoom changes distance."""
    global orbit_changed
    data = request.get_json()
    dx = data.get("dx", 0.0)
    dy = data.get("dy", 0.0)
    dz = data.get("dz", 0.0)  # scroll zoom
    offset = cam_orbit[current_camera]
    offset["azimuth"] += dx * 0.3
    offset["elevation"] += dy * 0.3
    offset["elevation"] = max(-80.0, min(80.0, offset["elevation"]))
    offset["distance"] += dz * 0.3
    offset["distance"] = max(-3.0, min(5.0, offset["distance"]))
    orbit_changed = True
    return jsonify({"ok": True, "azimuth": round(offset["azimuth"], 1),
                    "elevation": round(offset["elevation"], 1),
                    "distance": round(offset["distance"], 2)})


@app.route("/camera/reset", methods=["POST"])
def reset_camera():
    """Reset orbit for current camera."""
    global orbit_changed
    cam_orbit[current_camera] = {"azimuth": 0.0, "elevation": 0.0, "distance": 0.0}
    orbit_changed = True
    _reset_cam_to_original(current_camera)
    return jsonify({"ok": True})


@app.route("/key", methods=["POST"])
def handle_key():
    global joint_vel, reset_flag, save_flag, is_recording, episode_states
    data = request.get_json()
    key = data.get("key", "")
    act = data.get("action", "down")
    v = 0.5 if act == "down" else 0.0

    mapping = {
        "w": (0, v), "s": (0, -v),
        "a": (1, v), "d": (1, -v),
        "q": (2, -v), "e": (2, v),
        "r": (3, v), "f": (3, -v),
        "t": (4, v), "g": (4, -v),
        "y": (5, v), "h": (5, -v),
        "u": (6, v), "j": (6, -v),
        "z": (7, -v), "x": (7, v),
    }

    if key in mapping:
        idx, val = mapping[key]
        joint_vel[idx] = val

    if act == "down":
        if key == "reset":
            reset_flag = True
        elif key == "Enter" and is_recording:
            save_flag = True

    return jsonify({"ok": True})


@app.route("/record/start", methods=["POST"])
def start_rec():
    global is_recording, episode_states
    is_recording = True
    episode_states = []
    return jsonify({"ok": True})


@app.route("/record/stop", methods=["POST"])
def stop_rec():
    global save_flag
    save_flag = True
    return jsonify({"ok": True})


# HTML -----------------------------------------------------
HTML = r"""<!DOCTYPE html><html lang="zh-CN">
<head><meta charset="UTF-8"><title>Panda Robot Teleop</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d1117;color:#c9d1d9;font-family:system-ui;text-align:center;overflow:hidden}
.top{background:#161b22;padding:12px;border-bottom:2px solid #30363d}
.top h1{font-size:1.2em;color:#f78166}
.row{display:flex;gap:15px;padding:15px;max-width:1100px;margin:0 auto;flex-wrap:wrap;justify-content:center}
.vid{background:#161b22;border-radius:8px;padding:6px;border:1px solid #30363d;position:relative}
.vid img{width:480px;height:360px;border-radius:4px;background:#000;cursor:grab;user-select:none}
.vid img:active{cursor:grabbing}
.vid .cam-label{position:absolute;top:12px;left:12px;background:rgba(0,0,0,0.7);color:#f78166;padding:3px 10px;border-radius:4px;font-size:0.75em;font-weight:bold}
.panel{background:#161b22;border-radius:8px;padding:18px;min-width:240px;border:1px solid #30363d;text-align:left}
.panel h3{color:#f78166;margin-bottom:10px;font-size:.95em}
.s{font-size:.85em;margin:5px 0}.s b{color:#7ee787}
.btn{padding:8px 14px;margin:4px 4px 4px 0;border:none;border-radius:4px;cursor:pointer;font-weight:bold;font-size:.8em}
.rec{background:#da3633;color:#fff}.sav{background:#238636;color:#fff}.res{background:#484f58;color:#fff}
.cam-btn{padding:5px 10px;margin:2px;border:1px solid #30363d;border-radius:4px;cursor:pointer;font-size:.7em;background:#21262d;color:#c9d1d9}
.cam-btn.active{background:#f78166;color:#000;border-color:#f78166;font-weight:bold}
.cam-btn:hover{background:#30363d}
.keys{background:#0d1117;padding:10px;border-radius:6px;margin-top:12px;font-size:.78em;line-height:1.7}
.keys kbd{background:#f78166;color:#000;padding:1px 6px;border-radius:3px;font-weight:bold}
#msg{font-size:.8em;color:#f78166;margin-top:6px}
</style></head>
<body>
<div class="top"><h1>Panda Robot Web Teleoperation</h1><div id="msg">Connecting...</div></div>
<div class="row">
<div class="vid">
<span class="cam-label" id="camLabel">Front</span>
<img id="feed" src="">
</div>
<div class="panel">
<h3>Status</h3>
<div class="s">Step: <b id="sStep">0</b></div>
<div class="s">Gripper: <b id="sGrip">-</b></div>
<div class="s">EEF Pos: <b id="sPos">-</b></div>
<div class="s">Saved: <b id="sSaved">0</b></div>
<div class="s">Record: <b id="sRec">Off</b></div>
<div style="margin-top:12px">
<button class="btn rec" onclick="startRec()">Record</button>
<button class="btn sav" onclick="stopRec()">Save</button>
<button class="btn res" onclick="resetR()">Reset</button>
</div>
<h3 style="margin-top:15px">Cameras</h3>
<div id="camBtns"></div>
<div style="margin-top:4px;font-size:.72em;color:#8b949e">
<kbd>&larr;/&rarr;</kbd> cycle &nbsp; <b>Drag</b> to orbit &nbsp; <b>Scroll</b> to zoom &nbsp; <kbd>C</kbd> reset view
</div>
<div class="keys"><b>Joint Controls (hold key to move):</b><br>
<kbd>W/S</kbd> Joint0 Base &nbsp; <kbd>A/D</kbd> Joint1 Shoulder<br>
<kbd>Q/E</kbd> Joint2 Elbow &nbsp; <kbd>R/F</kbd> Joint3 Wrist1<br>
<kbd>T/G</kbd> Joint4 Wrist2 &nbsp; <kbd>Y/H</kbd> Joint5 Wrist3<br>
<kbd>U/J</kbd> Joint6 Fingers &nbsp; <kbd>Z/X</kbd> Joint7 Gripper<br>
<kbd>Enter</kbd> Save (when recording)</div></div></div>
<script>
var p={};
var camNames=["frontview","birdview","agentview","sideview","robot0_robotview","robot0_eye_in_hand"];
var camLabels={frontview:"Front",birdview:"Bird's Eye",agentview:"Agent",sideview:"Side",robot0_robotview:"Robot",robot0_eye_in_hand:"Eye-in-Hand"};

// Build camera buttons
var cb='';
for(var i=0;i<camNames.length;i++){
  cb+='<button class="cam-btn" onclick="setCam('+"'"+camNames[i]+"'"+')">'+camLabels[camNames[i]]+'</button>';
}
document.getElementById('camBtns').innerHTML=cb;

function updateCamBtns(active){
  var btns=document.querySelectorAll('.cam-btn');
  for(var i=0;i<btns.length;i++){
    btns[i].classList.toggle('active', btns[i].textContent===camLabels[active]);
  }
  document.getElementById('camLabel').textContent=camLabels[active]||active;
}

function setCam(name){
  fetch('/camera',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:name})})
    .then(function(r){return r.json()}).then(function(d){if(d.ok)updateCamBtns(d.camera);});
}

function cycleCam(dir){
  dir=dir||1;
  fetch('/camera/cycle',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({direction:dir})})
    .then(function(r){return r.json()}).then(function(d){if(d.ok)updateCamBtns(d.camera);});
}

// Mouse orbit
var dragging=false, lastX=0, lastY=0;
var feed=document.getElementById('feed');
feed.addEventListener('mousedown',function(e){dragging=true;lastX=e.clientX;lastY=e.clientY;e.preventDefault();});
document.addEventListener('mousemove',function(e){
  if(!dragging)return;
  var dx=e.clientX-lastX, dy=e.clientY-lastY;
  lastX=e.clientX;lastY=e.clientY;
  fetch('/camera/orbit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dx:dx,dy:dy,dz:0})});
});
document.addEventListener('mouseup',function(){dragging=false;});
feed.addEventListener('wheel',function(e){
  fetch('/camera/orbit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dx:0,dy:0,dz:e.deltaY>0?1:-1})});
  e.preventDefault();
});

function poll(){fetch('/frame').then(function(r){return r.json()}).then(function(d){
 feed.src='data:image/jpeg;base64,'+d.image;
 document.getElementById('sStep').textContent=d.step;
 document.getElementById('sGrip').textContent=d.gripper;
 document.getElementById('sPos').textContent=d.eef.join(', ');
 document.getElementById('sSaved').textContent=d.saved;
 document.getElementById('sRec').textContent=d.recording?'Recording...':'Off';
 document.getElementById('msg').textContent='Connected! Hold keys to move joints | Drag to orbit | Scroll to zoom';
 if(d.camera)updateCamBtns(d.camera);
});setTimeout(poll,100)}
function sendKey(k,a){fetch('/key',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key:k,action:a})})}
var validKeys=['w','s','a','d','q','e','r','f','t','g','y','h','u','j','z','x','Enter'];
document.addEventListener('keydown',function(e){
 if(p[e.key])return;p[e.key]=true;
 if(e.key==='ArrowLeft'){cycleCam(-1);e.preventDefault();return;}
 if(e.key==='ArrowRight'){cycleCam(1);e.preventDefault();return;}
 if(e.key==='c'||e.key==='C'){fetch('/camera/reset',{method:'POST'});e.preventDefault();return;}
 if(validKeys.includes(e.key)){sendKey(e.key,'down');e.preventDefault()}
})
document.addEventListener('keyup',function(e){
 p[e.key]=false;
 if(validKeys.includes(e.key)){sendKey(e.key,'up');e.preventDefault()}
})
function startRec(){fetch('/record/start',{method:'POST'})}
function stopRec(){fetch('/record/stop',{method:'POST'})}
function resetR(){fetch('/key',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key:'reset',action:'down'})})}
poll()
</script></body></html>"""


if __name__ == "__main__":
    print("=" * 55)
    print("  Panda Robot Web Teleoperation")
    print("  Open: http://localhost:9090 (via SSH tunnel)")
    print("=" * 55)
    init_env()
    threading.Thread(target=sim_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=9090, threaded=True, debug=False)
