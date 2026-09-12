"""
Self-contained real-time verification of the cognitive swarm controller.

Scenario
--------
- A compact flock of small car-like agents moves forward in +x.
- A continuing stream of larger "unruly" vehicles approaches the flock.
- Arrival time, speed, entry y-position, and travel angle vary on every pass.
- Agents keep a safe distance from one another using both instantaneous and
  short-horizon predictive separation.
- Obstacle avoidance uses closest-approach prediction plus a tangential dodge,
  giving earlier/smoother navigation than pure radial repulsion.
- Positive acceleration is used mainly to rejoin/close gaps after avoidance;
  braking is used mainly when a strong evasive manoeuvre is demanded.

The same control variables are intended for ROS2/Gazebo. In Gazebo the
verification plant is removed; odometry is read and velocity/yaw-rate
commands are published instead.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, writers
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import Affine2D


@dataclass
class Config:
    dt: float = 0.04

    # Geometry / sensing
    sensing_radius: float = 12.0
    safe_agent_distance: float = 2.4
    prediction_horizon_agents: float = 0.75
    obstacle_margin: float = 8.5
    prediction_horizon_obs: float = 1.0

    # Internal perceptual dynamics
    T1: float = 0.6
    T2: float = 0.3
    Gamma: float = 1.0
    kappa: float = 1.2
    g_align: float = 1.0
    gamma_s: float = 0.03
    lambda_fb: float = 2.0
    s_base: float = 0.0
    caution_gain: float = 1.0

    # Steering
    k_align: float = 1.0
    k_coh: float = 0.45
    k_sep: float = 3.4
    k_obs_radial: float = 5.2
    k_obs_tangent: float = 4.6
    k_goal: float = 1.65
    omega_max: float = 1.7

    # Speed extension
    v0: float = 3.0
    v_min: float = 0.9
    v_max: float = 4.6
    eta_rejoin: float = 0.55
    eta_brake: float = 1.0
    rejoin_length: float = 10.0
    tau_v: float = 0.42
    a_acc: float = 2.2
    a_brake: float = 4.5

    # Demo vehicle geometry (visual + collision envelope)
    agent_length: float = 0.95
    agent_width: float = 0.50
    obstacle_length: float = 4.2
    obstacle_width: float = 2.0
    obstacle_radius: float = 2.25  # conservative circular envelope for controller

    # Animation-only traffic and camera settings.  These do not enter the
    # cognitive controller or the physical state equations.
    obstacle_slots: int = 5
    obstacle_speed_min: float = 2.4
    obstacle_speed_max: float = 6.2
    obstacle_spawn_delay_min: float = 1.3
    obstacle_spawn_delay_max: float = 4.8
    camera_half_width: float = 43.0
    camera_half_height: float = 22.0
    camera_look_ahead: float = 20.0
    camera_dead_zone_x: float = 5.5
    camera_dead_zone_y: float = 3.0
    camera_response_time: float = 1.25


DEFAULT_MOVIE = "cognitive_swarm_demo.mp4"


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def positive_float(value):
    value = float(value)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run the cognitive-swarm animation live or export it as an MP4."
        )
    )
    parser.add_argument(
        "--save", nargs="?", const=DEFAULT_MOVIE, default=None,
        metavar="FILE.mp4",
        help=(
            "save an MP4 and exit; when FILE is omitted, use "
            f"{DEFAULT_MOVIE!r}"
        ),
    )
    parser.add_argument(
        "--duration", type=positive_float, default=30.0, metavar="SECONDS",
        help="simulated duration of an exported movie (default: 30)",
    )
    parser.add_argument(
        "--fps", type=positive_int, default=25,
        help="export playback frame rate (default: 25, matching model dt)",
    )
    parser.add_argument(
        "--dpi", type=positive_int, default=140,
        help="export resolution in dots per inch (default: 140)",
    )
    parser.add_argument(
        "--agents", type=positive_int, default=36,
        help="number of flock agents (default: 36)",
    )
    parser.add_argument(
        "--obstacles", type=positive_int, default=5,
        help="number of reusable obstacle slots (default: 5)",
    )
    parser.add_argument(
        "--seed", type=int, default=11,
        help="random seed for reproducible traffic (default: 11)",
    )
    args = parser.parse_args(argv)
    if args.save is not None and Path(args.save).suffix.lower() != ".mp4":
        parser.error("--save output must have an .mp4 extension")
    return args


class CognitiveSwarm:
    def __init__(self, n_agents, cfg):
        self.cfg = cfg
        self.N = n_agents
        self.bloch = np.zeros((n_agents, n_agents, 3), dtype=float)
        self.slow = np.zeros(n_agents, dtype=float)

    @staticmethod
    def unit(v):
        v = np.asarray(v, dtype=float)
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        return np.divide(v, n, out=np.zeros_like(v, dtype=float), where=n > 1e-12)

    @staticmethod
    def wrap(a):
        return (a + np.pi) % (2*np.pi) - np.pi

    def update_internal(self, pos, heading):
        c = self.cfg
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(delta, axis=2)
        A = (dist > 0.0) & (dist < c.sensing_radius)

        dot_h = heading @ heading.T
        hz = c.kappa * self.slow[:, None] + c.g_align * dot_h

        mx = self.bloch[:, :, 0]
        my = self.bloch[:, :, 1]
        mz = self.bloch[:, :, 2]
        mz_eq = np.tanh(hz)

        dmx = -2.0 * hz * my - mx / c.T2
        dmy = 2.0 * (hz * mx - c.Gamma * mz) - my / c.T2
        dmz = 2.0 * c.Gamma * my - (mz - mz_eq) / c.T1

        mx[A] += c.dt * dmx[A]
        my[A] += c.dt * dmy[A]
        mz[A] += c.dt * dmz[A]
        mz[:] = np.clip(mz, -1.0, 1.0)

        count = A.sum(axis=1)
        M = np.divide((mz * A).sum(axis=1), count,
                      out=np.zeros(self.N), where=count > 0)
        s_star = np.tanh(c.lambda_fb * M)
        self.slow += c.dt * (
            -c.gamma_s * (self.slow - c.s_base) + c.gamma_s * s_star
        )
        q = 1.0 + c.caution_gain / (1.0 + np.exp(-self.slow))
        w = 0.5 * (1.0 + mz) * A
        return A, dist, w, M, q

    def desired_direction(self, pos, heading, speed, obs_pos, obs_vel,
                          obs_radius, A, dist, w, q, flock_centre):
        """Return one desired direction per agent.

        Important: speed is an explicit argument.  This fixes the previous
        NameError and is also physically necessary because closest-approach
        prediction depends on relative velocity.
        """
        c = self.cfg
        D = np.zeros_like(pos)
        goal = np.array([1.0, 0.0])

        velocity = speed[:, None] * heading

        for i in range(self.N):
            nei = np.where(A[i])[0]

            # Cognitive weighted alignment
            wsum = w[i].sum()
            if wsum > 1e-12:
                align = c.k_align * (w[i, :, None] * heading).sum(axis=0) / wsum
            else:
                align = c.k_align * heading[i]

            # Local cohesion: intentionally moderate so safety dominates.
            if len(nei):
                coh = c.k_coh * (pos[nei].mean(axis=0) - pos[i])
            else:
                coh = np.zeros(2)

            # ----------------------------------------------------------
            # Agent-agent safety: current + predicted closest approach.
            # ----------------------------------------------------------
            sep = np.zeros(2)
            for j in nei:
                r = pos[j] - pos[i]
                v_rel = velocity[j] - velocity[i]
                vv = np.dot(v_rel, v_rel)
                tau = np.clip(
                    -np.dot(r, v_rel) / (vv + 1e-12),
                    0.0, c.prediction_horizon_agents
                )
                r_ca = r + v_rel * tau
                d_now = np.linalg.norm(r)
                d_ca = np.linalg.norm(r_ca)
                d_eff = min(d_now, d_ca)

                if d_eff < c.safe_agent_distance:
                    threat = r_ca if d_ca <= d_now else r
                    nt = np.linalg.norm(threat)
                    if nt > 1e-12:
                        z = np.clip(
                            (c.safe_agent_distance - d_eff)
                            / c.safe_agent_distance, 0.0, 1.0
                        )
                        smooth = z*z*(3.0 - 2.0*z)
                        sep += c.k_sep * smooth * (-threat / nt)
            sep *= q[i]

            # ----------------------------------------------------------
            # Moving obstacle vehicles: closest approach + tangent.
            # ----------------------------------------------------------
            obs = np.zeros(2)
            vi = velocity[i]
            for centre, u, rad in zip(obs_pos, obs_vel, obs_radius):
                r = centre - pos[i]         # agent -> obstacle
                v_rel = u - vi
                vv = np.dot(v_rel, v_rel)
                tau = np.clip(
                    -np.dot(r, v_rel) / (vv + 1e-12),
                    0.0, c.prediction_horizon_obs
                )
                r_ca = r + v_rel * tau

                d_now = np.linalg.norm(r) - rad
                d_ca = np.linalg.norm(r_ca) - rad
                threat = r_ca if d_ca <= d_now else r
                clearance = min(d_now, d_ca)
                nt = np.linalg.norm(threat)

                if clearance < c.obstacle_margin and nt > 1e-12:
                    z = np.clip(
                        (c.obstacle_margin - clearance) / c.obstacle_margin,
                        0.0, 1.0
                    )
                    smooth = z*z*(3.0 - 2.0*z)
                    n = threat / nt

                    # Radial component creates clearance.
                    radial = -n

                    # Group-consistent tangential passing side:
                    # if obstacle is above flock centre, go below; otherwise above.
                    # This reduces unnecessary flock splitting around the obstacle.
                    pass_sign = -1.0 if centre[1] >= flock_centre[1] else 1.0
                    tangent = pass_sign * np.array([-n[1], n[0]])

                    # Stronger only when relative motion is closing.
                    closing = max(
                        0.0,
                        -np.dot(r, v_rel) / (np.linalg.norm(r) + 1e-12)
                    )
                    gain = (1.0 + 0.18 * closing) * smooth
                    obs += gain * (
                        c.k_obs_radial * radial
                        + c.k_obs_tangent * tangent
                    )
            obs *= q[i]

            # Persistent forward migration avoids a stationary cluster.
            D[i] = align + coh + sep + obs + c.k_goal * goal
            if np.linalg.norm(D[i]) < 1e-12:
                D[i] = heading[i]

        return self.unit(D)

    def commands(self, pos, heading, speed, obs_pos, obs_vel, obs_radius,
                 rejoin_reference=None):
        c = self.cfg
        A, dist, w, M, q = self.update_internal(pos, heading)

        if rejoin_reference is None:
            rejoin_reference = pos.mean(axis=0)

        d_hat = self.desired_direction(
            pos, heading, speed, obs_pos, obs_vel, obs_radius,
            A, dist, w, q, rejoin_reference
        )

        # Rejoining / gap-closing demand.  The projection prevents acceleration
        # until the agent is actually pointing back toward the flock reference.
        to_ref = rejoin_reference[None, :] - pos
        ref_dist = np.linalg.norm(to_ref, axis=1)
        c_hat = self.unit(to_ref)
        toward = np.maximum(np.sum(heading * c_hat, axis=1), 0.0)
        rho = np.tanh(ref_dist / c.rejoin_length) * toward

        # Braking demand from required change of direction.
        mu = np.clip(1.0 - np.sum(heading * d_hat, axis=1), 0.0, 2.0)

        v_star = c.v0 * (
            1.0 + c.eta_rejoin * rho - 0.5 * c.eta_brake * mu
        )
        v_star = np.clip(v_star, c.v_min, c.v_max)
        accel = np.clip(
            (v_star - speed) / c.tau_v,
            -c.a_brake, c.a_acc
        )

        yaw = np.arctan2(heading[:, 1], heading[:, 0])
        yaw_des = np.arctan2(d_hat[:, 1], d_hat[:, 0])
        yaw_rate = np.clip(
            self.wrap(yaw_des - yaw) / c.dt,
            -c.omega_max, c.omega_max
        )

        speed_cmd = np.clip(speed + accel * c.dt, c.v_min, c.v_max)
        vel_cmd = speed_cmd[:, None] * d_hat

        return {
            "desired_direction_world": d_hat,
            "speed_setpoint": v_star,
            "acceleration_setpoint": accel,
            "yaw_rate_setpoint": yaw_rate,
            "linear_velocity_world": vel_cmd,
            "mu": mu,
            "rho": rho,
            "s": self.slow.copy(),
            "q": q,
            "M": M,
            "neighbor_count": A.sum(axis=1),
        }


def make_agent_formation(rng, N):
    """Compact forward-moving flock with safe initial spacing."""
    cols = 9
    rows = int(np.ceil(N / cols))
    spacing_x = 2.7
    spacing_y = 2.7
    pts = []
    for r in range(rows):
        for c in range(cols):
            if len(pts) >= N:
                break
            x = -31.0 + c * spacing_x + rng.normal(0, 0.10)
            y = -4.0 + r * spacing_y + rng.normal(0, 0.10)
            pts.append([x, y])
    return np.array(pts, dtype=float)


def main(argv=None):
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    cfg = Config()
    cfg.obstacle_slots = args.obstacles
    N = args.agents

    # Flock of small cars moving forward (+x). Coordinates are never wrapped
    # or reset; the plotting camera follows the flock instead.
    pos = make_agent_formation(rng, N)
    theta = rng.normal(0.0, 0.035, N)
    heading = np.column_stack([np.cos(theta), np.sin(theta)])
    speed = np.full(N, cfg.v0)
    ctl = CognitiveSwarm(N, cfg)

    # Reusable slots form an asynchronous stream of oncoming vehicles.  A slot
    # is invisible/inactive while waiting for its next independently randomized
    # arrival, so the traffic is not a synchronized pair on a fixed loop.
    n_obs = cfg.obstacle_slots
    obs_radius = np.full(n_obs, cfg.obstacle_radius)
    obs_pos = np.zeros((n_obs, 2), dtype=float)
    obs_vel = np.zeros((n_obs, 2), dtype=float)
    obs_speed = np.zeros(n_obs, dtype=float)
    obs_heading = np.full(n_obs, np.pi, dtype=float)
    obs_turn_rate = np.zeros(n_obs, dtype=float)
    obs_target_turn_rate = np.zeros(n_obs, dtype=float)
    obs_next_turn_change = np.zeros(n_obs, dtype=float)
    obs_next_spawn = np.zeros(n_obs, dtype=float)
    obs_active = np.zeros(n_obs, dtype=bool)
    sim_time = 0.0

    spawn_ahead_min = 50.0
    spawn_ahead_max = 70.0
    despawn_behind = 30.0

    def spawn_obstacle(k):
        """Start one new random encounter ahead in global coordinates."""
        flock_c = pos.mean(axis=0)
        ahead = rng.uniform(spawn_ahead_min, spawn_ahead_max)

        # Most encounters are deliberately close enough to perturb/split the
        # flock; some are wider misses so the controller also demonstrates
        # selective, non-overreactive navigation.
        if rng.random() < 0.78:
            y_offset = rng.uniform(-5.0, 5.0)
        else:
            y_offset = rng.uniform(-11.0, 11.0)

        obs_pos[k] = np.array([
            flock_c[0] + ahead,
            flock_c[1] + y_offset,
        ])

        # Each encounter gets its own speed and smooth bounded weave.
        obs_speed[k] = rng.uniform(
            cfg.obstacle_speed_min, cfg.obstacle_speed_max
        )
        obs_heading[k] = np.pi + rng.uniform(-0.38, 0.38)
        obs_vel[k] = obs_speed[k] * np.array([
            np.cos(obs_heading[k]), np.sin(obs_heading[k])
        ])
        obs_turn_rate[k] = 0.0
        obs_target_turn_rate[k] = rng.uniform(-0.18, 0.18)
        obs_next_turn_change[k] = sim_time + rng.uniform(0.9, 2.4)
        obs_active[k] = True

    # Random staggered first arrivals; the first is present immediately.
    obs_next_spawn[0] = 0.0
    for k in range(1, n_obs):
        obs_next_spawn[k] = obs_next_spawn[k - 1] + rng.uniform(1.8, 4.5)
    spawn_obstacle(0)

    # --------------------------- plot ---------------------------
    fig, ax = plt.subplots(figsize=(11, 6.3))
    ax.set_aspect("equal")
    ax.set_xlabel("global x")
    ax.set_ylabel("global y")
    ax.set_title(
        "Cognitive swarm in global motion: avoidance and reformation"
    )
    ax.set_facecolor("#f7f8f4")
    ax.grid(True, which="major", color="#ccd3cd", linewidth=0.65, alpha=0.75)
    ax.minorticks_on()
    ax.grid(True, which="minor", color="#e4e8e3", linewidth=0.4, alpha=0.55)

    def vehicle_polygons(centres, directions, length, width):
        """World-coordinate, heading-aware pentagons with a visible nose."""
        local = np.array([
            [-0.50 * length, -0.50 * width],
            [0.28 * length, -0.50 * width],
            [0.50 * length, 0.0],
            [0.28 * length, 0.50 * width],
            [-0.50 * length, 0.50 * width],
        ])
        polygons = []
        for p, h in zip(centres, directions):
            normal = np.array([-h[1], h[0]])
            rotation = np.column_stack([h, normal])
            polygons.append(local @ rotation.T + p)
        return polygons

    # Unlike a fixed ">" scatter marker, every small body now turns with its
    # physical heading.  Colour is updated from speed to make acceleration and
    # braking visible without changing the model.
    speed_norm = plt.Normalize(cfg.v_min, cfg.v_max)
    agents = PolyCollection(
        vehicle_polygons(pos, heading, cfg.agent_length, cfg.agent_width),
        facecolors=plt.cm.Blues(speed_norm(speed)), edgecolors="#17365d",
        linewidths=0.45, zorder=4,
    )
    ax.add_collection(agents)
    centre = ax.scatter(
        [pos[:, 0].mean()], [pos[:, 1].mean()], marker="x", s=65,
        color="#1b1b1b", label="flock centre", zorder=5
    )
    centre_history = [pos.mean(axis=0).copy()]
    centre_trail, = ax.plot([], [], color="#4c78a8", linewidth=1.0,
                            alpha=0.38, zorder=2)

    # Obstacles are large filled car-like rectangles and visually distinct.
    obs_colors = ["#d1495b", "#f28e2b", "#8f5da2", "#b6632e", "#e15759"]
    obs_rects = []
    for k in range(n_obs):
        rect = Rectangle(
            (-cfg.obstacle_length/2, -cfg.obstacle_width/2),
            cfg.obstacle_length, cfg.obstacle_width,
            facecolor=obs_colors[k % len(obs_colors)], edgecolor="black",
            linewidth=1.0, alpha=0.84, visible=bool(obs_active[k]), zorder=3,
        )
        ax.add_patch(rect)
        obs_rects.append(rect)

    status = ax.text(0.01, 0.99, "", transform=ax.transAxes,
                     va="top", ha="left")
    agent_key = Patch(facecolor=plt.cm.Blues(0.65), edgecolor="#17365d",
                      label="flock agent (colour = speed)")
    obstacle_key = Patch(facecolor=obs_colors[0], edgecolor="black",
                         label="random oncoming obstacle")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [agent_key, obstacle_key],
              labels + [agent_key.get_label(), obstacle_key.get_label()],
              loc="lower right")

    # The camera has its own state.  It looks ahead, moves only after the flock
    # crosses a dead zone, and eases toward its target.  Consequently the flock
    # visibly travels through the world instead of being nailed to canvas centre.
    camera_centre = pos.mean(axis=0) + np.array([cfg.camera_look_ahead, 0.0])

    def update_camera():
        nonlocal camera_centre
        flock_c = pos.mean(axis=0)
        mean_heading = CognitiveSwarm.unit(heading.mean(axis=0))
        target = flock_c + cfg.camera_look_ahead * mean_heading
        error = target - camera_centre
        dead_zone = np.array([
            cfg.camera_dead_zone_x, cfg.camera_dead_zone_y
        ])
        correction = np.sign(error) * np.maximum(np.abs(error) - dead_zone, 0.0)
        ease = 1.0 - np.exp(-cfg.dt / cfg.camera_response_time)
        camera_centre += ease * correction
        ax.set_xlim(camera_centre[0] - cfg.camera_half_width,
                    camera_centre[0] + cfg.camera_half_width)
        ax.set_ylim(camera_centre[1] - cfg.camera_half_height,
                    camera_centre[1] + cfg.camera_half_height)

    def update_rectangles():
        for k, (rect, p, v) in enumerate(zip(obs_rects, obs_pos, obs_vel)):
            rect.set_visible(bool(obs_active[k]))
            if not obs_active[k]:
                continue
            angle_deg = np.degrees(np.arctan2(v[1], v[0]))
            tr = (Affine2D()
                  .rotate_deg(angle_deg)
                  .translate(p[0], p[1]) + ax.transData)
            rect.set_transform(tr)

    update_camera()
    update_rectangles()

    def animate(_):
        nonlocal pos, heading, speed, sim_time
        sim_time += cfg.dt

        flock_c = pos.mean(axis=0)

        # Activate independently scheduled traffic, then give active obstacles a
        # smoothly changing (rather than frame-jittering) bounded random weave.
        for k in range(n_obs):
            if not obs_active[k]:
                if sim_time >= obs_next_spawn[k]:
                    spawn_obstacle(k)
                else:
                    continue

            if sim_time >= obs_next_turn_change[k]:
                obs_target_turn_rate[k] = rng.uniform(-0.24, 0.24)
                obs_next_turn_change[k] = sim_time + rng.uniform(0.8, 2.3)

            turn_ease = 1.0 - np.exp(-cfg.dt / 0.55)
            obs_turn_rate[k] += turn_ease * (
                obs_target_turn_rate[k] - obs_turn_rate[k]
            )
            obs_heading[k] += obs_turn_rate[k] * cfg.dt

            # Keep each obstacle generally oncoming while allowing substantial
            # lateral randomness.  Wrap relative to pi into [-pi,pi].
            rel = CognitiveSwarm.wrap(obs_heading[k] - np.pi)
            rel = np.clip(rel, -0.55, 0.55)
            obs_heading[k] = np.pi + rel
            if abs(rel) >= 0.549:
                obs_target_turn_rate[k] *= -0.65
            obs_vel[k] = obs_speed[k] * np.array([
                np.cos(obs_heading[k]), np.sin(obs_heading[k])
            ])
            obs_pos[k] += obs_vel[k] * cfg.dt

            # Retire a passed vehicle, then wait a fresh random interval before
            # reusing its drawing slot.  Agent coordinates are never recycled.
            if obs_pos[k, 0] < flock_c[0] - despawn_behind:
                obs_active[k] = False
                obs_next_spawn[k] = sim_time + rng.uniform(
                    cfg.obstacle_spawn_delay_min,
                    cfg.obstacle_spawn_delay_max,
                )

        flock_ref = pos.mean(axis=0)
        active = np.where(obs_active)[0]
        out = ctl.commands(
            pos, heading, speed,
            obs_pos[active], obs_vel[active], obs_radius[active],
            rejoin_reference=flock_ref
        )

        # Verification plant only. In Gazebo, replace by odometry + cmd output.
        speed = np.clip(
            speed + out["acceleration_setpoint"] * cfg.dt,
            cfg.v_min, cfg.v_max
        )
        yaw = np.arctan2(heading[:, 1], heading[:, 0])
        yaw = CognitiveSwarm.wrap(
            yaw + out["yaw_rate_setpoint"] * cfg.dt
        )
        heading = np.column_stack([np.cos(yaw), np.sin(yaw)])
        pos = pos + speed[:, None] * heading * cfg.dt

        # No periodic boundary condition and no coordinate recycling.
        # Only the visual camera follows loosely, with lag and a dead zone.
        update_camera()
        agents.set_verts(vehicle_polygons(
            pos, heading, cfg.agent_length, cfg.agent_width
        ))
        agents.set_facecolors(plt.cm.Blues(speed_norm(speed)))
        centre.set_offsets(pos.mean(axis=0).reshape(1, 2))
        centre_history.append(pos.mean(axis=0).copy())
        if len(centre_history) > 900:
            del centre_history[:-900]
        trail = np.asarray(centre_history)
        centre_trail.set_data(trail[:, 0], trail[:, 1])
        update_rectangles()

        # Diagnostics: obstacle clearance, minimum inter-agent distance,
        # connected components (split/reformation), and flock span.
        if len(active):
            obs_clearance = min(
                np.min(np.linalg.norm(pos - obs_pos[k], axis=1)
                       - obs_radius[k])
                for k in active
            )
            obstacle_text = (
                f"active obstacles = {len(active)}   speeds = "
                + ", ".join(f"{obs_speed[k]:.1f}" for k in active)
            )
            clearance_text = f"obstacle clearance = {obs_clearance:.2f}"
        else:
            obstacle_text = "active obstacles = 0 (next traffic approaching)"
            clearance_text = "obstacle clearance = n/a"
        dd = pos[:, None, :] - pos[None, :, :]
        D = np.linalg.norm(dd, axis=2)
        np.fill_diagonal(D, np.inf)
        min_agent_dist = np.min(D)

        # Number of proximity-connected flock components.
        A = D < cfg.sensing_radius
        seen = np.zeros(N, dtype=bool)
        components = 0
        for seed in range(N):
            if seen[seed]:
                continue
            components += 1
            stack = [seed]
            seen[seed] = True
            while stack:
                i = stack.pop()
                js = np.where(A[i] & ~seen)[0]
                if len(js):
                    seen[js] = True
                    stack.extend(js.tolist())

        flock_span_y = np.ptp(pos[:, 1])
        flock_span_x = np.ptp(pos[:, 0])

        status.set_text(
            f"global flock x = {pos[:,0].mean():.1f}   "
            f"speed mean/min/max = {speed.mean():.2f} / "
            f"{speed.min():.2f} / {speed.max():.2f}\n"
            f"{obstacle_text}\n"
            f"components = {components}   "
            f"span x/y = {flock_span_x:.1f} / {flock_span_y:.1f}   "
            f"min agent gap = {min_agent_dist:.2f}\n"
            f"{clearance_text}   "
            f"mean rejoin rho = {out['rho'].mean():.2f}   "
            f"mean manoeuvre mu = {out['mu'].mean():.2f}"
        )

        return [agents, centre, centre_trail, status, *obs_rects]

    # A finite frame iterable is required for deterministic movie export.  Live
    # mode intentionally has no frame limit and runs until its window is closed.
    frames = (range(int(np.ceil(args.duration / cfg.dt)))
              if args.save is not None else None)
    anim = FuncAnimation(
        fig, animate,
        frames=frames,
        interval=int(cfg.dt * 1000),
        cache_frame_data=False,
        blit=False,
        repeat=False,
    )

    plt.tight_layout()
    if args.save is None:
        # Keep a live reference while the GUI event loop owns the figure.
        plt.show(block=True)
    else:
        if not writers.is_available("ffmpeg"):
            raise RuntimeError(
                "MP4 export requires FFmpeg. Install it (for example, "
                "`brew install ffmpeg` or `sudo apt install ffmpeg`) and retry."
            )
        output = Path(args.save).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = FFMpegWriter(
            fps=args.fps,
            metadata={"title": "Cognitive Swarm Robots 2-D"},
            bitrate=2400,
        )
        frame_count = len(frames)
        print(
            f"Saving {frame_count} frames ({args.duration:g} simulated s) "
            f"to {output} ...",
            flush=True,
        )
        anim.save(str(output), writer=writer, dpi=args.dpi)
        plt.close(fig)
        print(f"Saved MP4: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
