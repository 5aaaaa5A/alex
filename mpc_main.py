# -*- coding: utf-8 -*-
"""MPC rectangular-flight runner (replaces the SDK / setpoint controller).

Simulation (validates MPC, no hardware):
    python3 mpc_main.py --sim --rect 0.6

Real flight, safe first step (takeoff -> hover -> land, no rectangle):
    python3 mpc_main.py --config config.json --flight --hover-only --with-mocap

Real flight, rectangle:
    python3 mpc_main.py --config config.json --flight --rect 0.6 --with-mocap

Keys:  Enter=start flight (then land),  q=estop,  Ctrl+C=quit
"""

import argparse
import json
import math
import os
import sys
import threading
import time

import numpy as np

import mpc_controller as mpc


# --------------------------------------------------------------------------- #
class SimPlant:
    def __init__(self, dt=0.10, tau=0.30, noise=0.008):
        self.dt = float(dt)
        self.beta = math.exp(-self.dt / tau)
        self.noise = float(noise)
        self.s = np.zeros(6)
        self.rng = np.random.default_rng(7)

    def step(self, u, dt):
        u = np.asarray(u, dtype=float).reshape(3)
        dt = float(dt)
        ratio = dt / self.dt
        p = self.s[0:3] + dt * self.s[3:6]
        v = self.beta ** ratio * self.s[3:6] + (1.0 - self.beta ** ratio) * u
        s = np.concatenate([p, v])
        if s[2] < 0.0:
            s[2] = 0.0
            if s[5] < 0.0:
                s[5] = 0.0
        s[0:3] += self.rng.standard_normal(3) * self.noise
        s[3:6] += self.rng.standard_normal(3) * self.noise * 2.0
        self.s = s
        return s.copy()

    def measure(self):
        return self.s.copy()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_state(link, obs):
    st = link.get_state() if hasattr(link, "get_state") else {}
    x, y, z = st.get("x"), st.get("y"), st.get("z")
    now = time.time()
    if None in (x, y, z):
        return None, obs.v.copy()
    pos = np.array([float(x), float(y), float(z)])
    vx, vy, vz = st.get("vx"), st.get("vy"), st.get("vz")
    if None not in (vx, vy, vz):
        vel = np.array([float(vx), float(vy), float(vz)])
    else:
        vel = obs.update(pos, now)
    return pos, vel


def send_command(link, u, landing=False, estop=False):
    if landing:
        link.send_velocity_world_setpoint(0.0, 0.0, -0.25, 0.0)
        return
    if estop:
        if hasattr(link, "send_stop"):
            link.send_stop()
        return
    link.send_velocity_world_setpoint(float(u[0]), float(u[1]), float(u[2]), 0.0)


def build_mocap(cfg, args):
    import nokov_client
    import transform as tf
    source = nokov_client.make_source(cfg, sim=args.sim)
    source.connect()
    frame0 = None
    deadline = time.time() + 10.0
    while time.time() < deadline:
        f = source.latest_frame()
        if f is not None and f.valid():
            frame0 = f
            break
        time.sleep(0.01)
    if frame0 is None:
        raise RuntimeError("no valid mocap frame within 10s")
    if args.no_auto_origin:
        origin = tuple(cfg["origin"]); yaw_off = cfg["yaw_offset_deg"]
    else:
        origin = (frame0.x_mm, frame0.y_mm, frame0.z_mm); yaw_off = frame0.yaw_deg
        print("[mocap] auto origin=(%.2f,%.2f,%.2f)mm yaw=%.2f deg" % (origin + (yaw_off,)))
    tr = tf.MocapToCF(origin, yaw_off)
    return source, tr


def main():
    ap = argparse.ArgumentParser(description="MPC rectangular flight")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--flight", action="store_true")
    ap.add_argument("--rect", type=float, default=0.6)
    ap.add_argument("--hover-only", action="store_true",
                    help="takeoff -> hover -> land, no rectangle")
    ap.add_argument("--auto", action="store_true",
                    help="start flight automatically after preflight")
    ap.add_argument("--hover-z", type=float, default=None)
    ap.add_argument("--unlock-s", type=float, default=3.0)
    ap.add_argument("--takeoff-s", type=float, default=4.0)
    ap.add_argument("--hover-s", type=float, default=3.0)
    ap.add_argument("--cruise", type=float, default=0.5)
    ap.add_argument("--dt", type=float, default=0.10)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--tau", type=float, default=0.30)
    ap.add_argument("--vmax", type=float, default=0.9)
    ap.add_argument("--cmd", choices=["pos", "vel"], default="pos",
                    help="pos=send position setpoints (robust), vel=velocity setpoints")
    ap.add_argument("--look", type=int, default=1,
                    help="lookahead steps used as the position setpoint")
    ap.add_argument("--max-flight", type=float, default=150.0)
    ap.add_argument("--with-mocap", action="store_true")
    ap.add_argument("--no-auto-origin", action="store_true")
    ap.add_argument("--no-sleep", action="store_true")
    ap.add_argument("--tag", default="mpc_rect")
    args = ap.parse_args()

    cfg = load_config(args.config)
    hover_z = float(args.hover_z if args.hover_z is not None else cfg.get("hover_z", 1.0))
    size = float(args.rect)

    mission = mpc.MissionRef(size=size, hover_z=hover_z,
                             takeoff_s=args.takeoff_s, hover_s=args.hover_s,
                             cruise=args.cruise, unlock_s=args.unlock_s,
                             rect_enabled=(not args.hover_only))
    planner = mpc.MpcVelocityPlanner(dt=args.dt, horizon=args.horizon,
                                     tau=args.tau, vmax=args.vmax)
    obs = mpc.VelocityObserver()

    if args.sim:
        return run_sim(planner, mission, obs, args, size, hover_z)
    return run_real(planner, mission, obs, cfg, args)


# --------------------------------------------------------------------------- #
def run_sim(planner, mission, obs, args, size, hover_z):
    dt = args.dt
    plant = SimPlant(dt=dt, tau=args.tau)
    t_mission = 0.0
    u_prev = np.zeros(3)
    n_h = planner.n
    max_err = np.zeros(3)
    total_err = np.zeros(3)
    rows = 0
    t0 = time.time()
    est_end = mission.t_land_end
    print("[sim] MPC rect=%s size=%.2f hover_z=%.2f cruise=%.2f horizon=%d dt=%.2f"
          % ("ON" if mission.rect_enabled else "OFF", size, hover_z, args.cruise,
             n_h, dt))
    print("[sim] phases: unlock %.1fs -> takeoff %.1fs -> hover %.1fs %s-> land %.1fs"
          % (mission.unlock_s, mission.takeoff_s, mission.hover_s,
             ("-> rect %.1fs " % mission.rect_s if mission.rect_enabled else ""),
             mission.land_s))
    while t_mission <= est_end + 1.5:
        pos = plant.measure()[0:3]
        vel = plant.measure()[3:6]
        s0 = np.concatenate([pos, vel])
        refs = []
        for k in range(1, n_h + 1):
            rs, _ = mission.ref_state(t_mission + k * dt)
            refs.append(rs)
        u, _ = planner.solve(s0, refs, u_prev, warm=planner.last_solve.get("z"))
        plant.step(u, dt)
        u_prev = u.copy()
        pr, _, ph = mission.sample(min(t_mission, mission.t_land_end))
        err = np.abs(pos - np.array(pr))
        max_err = np.maximum(max_err, err)
        total_err += err
        rows += 1
        if rows % 10 == 0:
            print("[sim] t=%5.1f ph=%-8s pos=(%.3f,%.3f,%.3f) ref=(%.3f,%.3f,%.3f)"
                  % (t_mission, ph, pos[0], pos[1], pos[2], pr[0], pr[1], pr[2]))
        t_mission += dt
        if not args.no_sleep:
            time.sleep(0.01)
    print("[sim] done. elapsed=%.1fs intervals=%d" % (time.time() - t0, rows))
    print("[sim] max |err| x=%.3f y=%.3f z=%.3f" % (max_err[0], max_err[1], max_err[2]))
    print("[sim] mean|err| x=%.3f y=%.3f z=%.3f"
          % (total_err[0] / max(1, rows), total_err[1] / max(1, rows),
             total_err[2] / max(1, rows)))
    return 0


# --------------------------------------------------------------------------- #
def run_real(planner, mission, obs, cfg, args):
    import crazyflie_link
    uri = cfg["radio_uri"]
    print("[cf] connecting %s ..." % uri)
    link = crazyflie_link.CrazyflieLink(uri)
    if not link.connect(timeout=10.0):
        print("[cf] connect failed")
        return 2
    if not link.arm():
        print("[cf] arming request unavailable")
    dt = args.dt
    link.start_state_log(period_ms=max(50, int(1000 * dt)))

    feeder = None
    source = None
    if args.with_mocap:
        try:
            source, tr = build_mocap(cfg, args)
            import crazyflie_link as cl
            import safety as safety_mod
            safety = safety_mod.SafetyMonitor(cfg.get("safety", {}))
            feeder = cl.ExtposFeeder(source, tr, safety, link=link,
                                     rate_hz=cfg.get("extpos_rate_hz", 20),
                                     use_extpose=cfg.get("use_extpose", True))
            feeder.start()
        except Exception as exc:  # pragma: no cover
            print("[mocap] unavailable, flying on EKF only: %s" % exc)
            source = None
            feeder = None

    link.reset_ekf(0.0, 0.0, 0.0, 0.0)

    # ---- preflight: wait for a settled EKF state -------------------------- #
    if hasattr(link, "wait_ekf_convergence"):
        try:
            ok = link.wait_ekf_convergence(timeout_s=15.0)
            print("[cf] EKF convergence ok=%s" % ok)
        except Exception as exc:
            print("[cf] EKF convergence check failed: %s" % exc)
    time.sleep(args.unlock_s)

    start_evt = threading.Event()
    land_req = threading.Event()
    estop_req = threading.Event()
    started = [False]

    def kb():
        while True:
            try:
                ch = input()
            except (EOFError, OSError):
                return
            s = ch.strip().lower()
            if s == "":
                if not started[0]:
                    start_evt.set()
                    print("[kb] flight start requested")
                else:
                    land_req.set()
                    print("[kb] land requested")
            elif s == "q":
                estop_req.set()
                print("[kb] estop requested")
    threading.Thread(target=kb, daemon=True).start()

    print("[ready] Press Enter to start flight, q=estop, Ctrl+C=quit")
    if args.auto:
        print("[ready] --auto: starting in 3s")
        time.sleep(3.0)
        start_evt.set()

    t_mission = 0.0
    u_prev = np.zeros(3)
    t0 = None
    next_t = time.time()
    try:
        while True:
            now = time.time()
            if link.link_lost() or estop_req.is_set():
                send_command(link, u_prev, estop=True)
                print("[main] estop/link lost")
                break
            pos, vel = read_state(link, obs)
            if pos is None:
                print("[cf] no EKF position yet")
                time.sleep(0.05)
                continue

            if not start_evt.is_set():
                # hold on the ground until the pilot confirms
                link.send_velocity_world_setpoint(0.0, 0.0, 0.0, 0.0)
                time.sleep(0.05)
                continue

            if t0 is None:
                t0 = now
                t_mission = 0.0
            if land_req.is_set() or now - t0 > args.max_flight or \
               pos[2] > cfg.get("max_altitude_m", 1.45):
                send_command(link, u_prev, landing=True)
                if now - t0 > args.max_flight:
                    print("[main] max flight time -> land")
                continue

            s0 = np.concatenate([pos, vel])
            refs = []
            for k in range(1, planner.n + 1):
                rs, _ = mission.ref_state(t_mission + k * dt)
                refs.append(rs)
            u, z = planner.solve(s0, refs, u_prev, warm=planner.last_solve.get("z"))
            u_prev = u.copy()
            if args.cmd == "pos":
                pred = planner.predict_states(s0, z)
                look = min(args.look, len(pred) - 1)
                ps = pred[look][0:3]
                link.send_position_setpoint(float(ps[0]), float(ps[1]),
                                            float(ps[2]), 0.0)
                cmd_out = ps
            else:
                send_command(link, u)
                cmd_out = u
            if int(round(t_mission * 5)) % 5 == 0:
                pr, _, ph = mission.sample(min(t_mission, mission.t_land_end))
                print("[f] t=%5.1f ph=%-8s pos=(%.3f,%.3f,%.3f) ref=(%.3f,%.3f,%.3f) cmd=(%.3f,%.3f,%.3f)"
                      % (t_mission, ph, pos[0], pos[1], pos[2],
                         pr[0], pr[1], pr[2], cmd_out[0], cmd_out[1], cmd_out[2]))
            t_mission += dt
            next_t += dt
            delay = next_t - time.time()
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        print("\n[main] Ctrl+C -> estop")
        try:
            link.send_stop()
        except Exception:
            pass
    finally:
        if feeder is not None:
            try:
                feeder.stop()
            except Exception:
                pass
        if source is not None:
            try:
                source.disconnect()
            except Exception:
                pass
        try:
            link.send_stop()
            link.close()
        except Exception:
            pass
    print("[main] done")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
