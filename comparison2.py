"""
benchmark_final.py

Quantitative matched comparison of:

  1. 2025-core variant
     Conceptual slow-fast perceptual core associated with arXiv:2510.23688

  2. 2026 fixed-speed variant
     Fixed-speed cognitive self-healing architecture associated with
     arXiv:2607.11960

  3. Present predictive + adaptive-speed controller

IMPORTANT
---------
The first two are matched variants re-evaluated under the PRESENT benchmark.
They are NOT claimed to reproduce the original numerical experiments.

Two tests:
(A) Dynamic-obstacle safety:
        minimum obstacle clearance d_min^obs
    Higher is better; d_min < 0 indicates penetration of the conservative
    obstacle collision envelope.

(B) Post-perturbation recovery:
        normalized recovery burden B_rec
    A subset of the flock is displaced backward after a settling period.
    Lower B_rec means faster restoration of nominal longitudinal compactness.

Evaluated for:
    N = 10, 20, 30 agents
    matched random seeds

Outputs:
    benchmark_results.csv
    comparison_two_panel.pdf
    comparison_two_panel.png
    comparison_safety.pdf
    comparison_recovery.pdf
"""

# ============================================================
# Prevent thread oversubscription inside multiprocessing workers
# ============================================================

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import csv
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from swarm_demo import Config, CognitiveSwarm, make_agent_formation


# ============================================================
# Benchmark configuration
# ============================================================

N_VALUES = (10, 20, 30)

MODELS = (
    "Non-Markovian [22]",
    "Self-Healing [23]",
    "Present Controller",
)

# Default publication run
DEFAULT_NSEEDS = 20

# ------------------------------------------------------------
# Experiment A: dynamic-obstacle safety
# ------------------------------------------------------------

T_SAFETY = 16.0

# obstacle speed / nominal swarm speed
OBSTACLE_SPEED_RATIO = 1.50

# ------------------------------------------------------------
# Experiment B: controlled catch-up / recovery
# ------------------------------------------------------------

T_RECOVERY = 15.0
PERTURB_TIME = 4.0

# Rear half of flock displaced backwards at t = PERTURB_TIME
LAG_DISTANCE = 5.5

# Recovery = longitudinal span returns within 10% of pre-perturbation span
RECOVERY_TOL = 1.10

# Must remain recovered for this long
RECOVERY_DWELL = 0.40


# ============================================================
# Controller variants
# ============================================================

def config_for(model):
    c = Config()

    if model == "Non-Markovian [22]":

        # Fixed physical speed
        c.v_min = c.v0
        c.v_max = c.v0

        # No adaptive braking / catch-up
        c.eta_rejoin = 0.0
        c.eta_brake = 0.0

        # No closest-approach prediction
        c.prediction_horizon_agents = 0.0
        c.prediction_horizon_obs = 0.0

        # Keep fast/slow perceptual dynamics but remove the
        # later state-dependent safety amplification.
        c.caution_gain = 0.0

    elif model == "Self-Healing [23]":

        # Cognitive slow-fast loop remains active,
        # including state-dependent caution.
        c.v_min = c.v0
        c.v_max = c.v0

        # Fixed speed
        c.eta_rejoin = 0.0
        c.eta_brake = 0.0

        # Instantaneous, rather than predictive, avoidance
        c.prediction_horizon_agents = 0.0
        c.prediction_horizon_obs = 0.0

    elif model == "Present Controller":

        # Full current Config:
        # - predictive inter-agent safety
        # - predictive moving-obstacle avoidance
        # - evasive braking
        # - adaptive catch-up acceleration
        pass

    else:
        raise ValueError(model)

    return c


# ============================================================
# Common plant update
# ============================================================

def plant_step(x, heading, speed, out, c, variable_speed):

    if variable_speed:
        speed += out["acceleration_setpoint"] * c.dt
        np.clip(speed, c.v_min, c.v_max, out=speed)
    else:
        speed.fill(c.v0)

    yaw = np.arctan2(
        heading[:, 1],
        heading[:, 0]
    )

    yaw += out["yaw_rate_setpoint"] * c.dt
    yaw = CognitiveSwarm.wrap(yaw)

    heading[:, 0] = np.cos(yaw)
    heading[:, 1] = np.sin(yaw)

    x += speed[:, None] * heading * c.dt

    return x, heading, speed


# ============================================================
# Experiment A: dynamic obstacle safety
# ============================================================

def simulate_safety(
    model,
    x0,
    heading0,
    obs_x0,
    obs_vel,
    obs_radius,
):
    c = config_for(model)

    x = x0.copy()
    heading = heading0.copy()
    speed = np.full(len(x), c.v0)

    obs_x = obs_x0.copy()

    ctl = CognitiveSwarm(len(x), c)

    dmin_obs = np.inf
    dmin_agent = np.inf

    variable_speed = (
        model == "Present Controller"
    )

    nsteps = int(np.ceil(T_SAFETY / c.dt))

    for _ in range(nsteps):

        obs_x += obs_vel * c.dt

        out = ctl.commands(
            x,
            heading,
            speed,
            obs_x,
            obs_vel,
            obs_radius,
            rejoin_reference=x.mean(axis=0),
        )

        x, heading, speed = plant_step(
            x,
            heading,
            speed,
            out,
            c,
            variable_speed,
        )

        # ----------------------------------------------------
        # Obstacle clearance: fully vectorized
        # ----------------------------------------------------

        d = np.linalg.norm(
            x[:, None, :]
            - obs_x[None, :, :],
            axis=2,
        )

        clearance = (
            d - obs_radius[None, :]
        )

        dmin_obs = min(
            dmin_obs,
            float(clearance.min()),
        )

        # ----------------------------------------------------
        # Minimum inter-agent gap
        # ----------------------------------------------------

        delta = (
            x[:, None, :]
            - x[None, :, :]
        )

        d2 = np.sum(
            delta * delta,
            axis=2,
        )

        np.fill_diagonal(
            d2,
            np.inf,
        )

        dmin_agent = min(
            dmin_agent,
            float(np.sqrt(d2.min())),
        )

    return dmin_obs, dmin_agent


# ============================================================
# Experiment B: standardized catch-up perturbation
# ============================================================

def simulate_recovery(
    model,
    x0,
    heading0,
):
    c = config_for(model)

    x = x0.copy()
    heading = heading0.copy()
    speed = np.full(len(x), c.v0)

    ctl = CognitiveSwarm(len(x), c)

    variable_speed = (
        model == "Present Controller"
    )

    # No obstacles in this experiment.
    obs_x = np.empty((0, 2))
    obs_v = np.empty((0, 2))
    obs_r = np.empty(0)

    nsteps = int(np.ceil(T_RECOVERY / c.dt))
    perturb_step = int(PERTURB_TIME / c.dt)

    pre_spans = []

    baseline_span = None
    recovery_burden = 0.0
    recovery_time = np.nan

    dwell_needed = max(
        1,
        int(RECOVERY_DWELL / c.dt)
    )
    dwell_count = 0

    perturbed = False

    for k in range(nsteps):

        # ----------------------------------------------------
        # Standardized perturbation
        # ----------------------------------------------------

        if k == perturb_step:

            baseline_span = float(
                np.median(pre_spans[-max(
                    1,
                    int(0.75 / c.dt)
                ):])
            )

            # Rear half is shifted backward.
            #
            # This preserves headings but creates a reproducible
            # longitudinal lag that the adaptive-speed mechanism
            # is specifically designed to heal.
            order = np.argsort(x[:, 0])
            rear = order[:len(x) // 2]

            x[rear, 0] -= LAG_DISTANCE

            perturbed = True

        out = ctl.commands(
            x,
            heading,
            speed,
            obs_x,
            obs_v,
            obs_r,
            rejoin_reference=x.mean(axis=0),
        )

        x, heading, speed = plant_step(
            x,
            heading,
            speed,
            out,
            c,
            variable_speed,
        )

        span_x = float(
            np.ptp(x[:, 0])
        )

        if not perturbed:

            pre_spans.append(span_x)
            continue

        # ----------------------------------------------------
        # Recovery burden
        #
        # Integral of excess longitudinal spread relative
        # to the pre-perturbation nominal span.
        # ----------------------------------------------------

        normalized_excess = max(
            span_x / baseline_span - 1.0,
            0.0,
        )

        recovery_burden += (
            normalized_excess * c.dt
        )

        # ----------------------------------------------------
        # Recovery time
        # ----------------------------------------------------

        if span_x <= RECOVERY_TOL * baseline_span:
            dwell_count += 1
        else:
            dwell_count = 0

        if (
            np.isnan(recovery_time)
            and dwell_count >= dwell_needed
        ):
            t_now = k * c.dt

            recovery_time = (
                t_now
                - PERTURB_TIME
                - RECOVERY_DWELL
            )

    return (
        recovery_burden,
        recovery_time,
        baseline_span,
    )


# ============================================================
# Present-controller obstacle/recovery videos
# ============================================================

def _vehicle_polygons(centres, directions, length, width):
    """Return car-shaped polygons for the animation collections."""
    local = np.array([
        [-0.50 * length, -0.50 * width],
        [0.28 * length, -0.50 * width],
        [0.50 * length, 0.0],
        [0.28 * length, 0.50 * width],
        [-0.50 * length, 0.50 * width],
    ])
    polygons = []
    for centre, direction in zip(centres, directions):
        normal = np.array([-direction[1], direction[0]])
        basis = np.column_stack([direction, normal])
        polygons.append(local @ basis.T + centre)
    return polygons


def _compact_video_formation(rng, count):
    """Tight hexagonal packing just above the controller safety distance."""
    columns = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / columns))
    spacing = 2.50
    row_spacing = 0.5 * np.sqrt(3.0) * spacing
    points = []
    for row in range(rows):
        for column in range(columns):
            if len(points) == count:
                break
            points.append([
                column * spacing + 0.5 * spacing * (row % 2),
                row * row_spacing,
            ])
    points = np.asarray(points, dtype=float)
    points -= points.mean(axis=0)
    return points + rng.normal(0.0, 0.01, (count, 2))


def export_present_recovery_video(
    agent_count,
    steps,
    output,
    fps=25,
    dpi=100,
    seed=2026,
    obstacle_speed_ratio=0.0,
):
    """Animate avoidance followed by autonomous post-encounter recovery."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.transforms import Affine2D
    from swarm_demo_random_obstacles import apply_safety_shield

    c = config_for("Present Controller")
    # Match the fixed-obstacle visualization configuration used by
    # swarm_demo_random_obstacles.py.  Static objects are fully predictable,
    # so the controller uses its shorter response range.
    stationary_obstacles = obstacle_speed_ratio == 0.0
    if stationary_obstacles:
        c.obstacle_margin = 2.0
    else:
        # For these small close-encounter visualizations, retain predictive
        # avoidance but use a compact response envelope.  The geometric shield
        # below still guarantees a clearly visible 0.4-unit body clearance.
        c.obstacle_margin = 1.0
    rng = np.random.default_rng(seed + 1009 * agent_count)

    x = _compact_video_formation(rng, agent_count)
    theta = rng.normal(0.0, 0.035, agent_count)
    heading = np.column_stack([np.cos(theta), np.sin(theta)])
    speed = np.full(agent_count, c.v0)
    ctl = CognitiveSwarm(agent_count, c)

    centre0 = x.mean(axis=0)
    formation_height = np.ptp(x[:, 1])
    # Scale the stagger to the smaller comparison flocks.  This is the same
    # near-path arrangement as the larger random-obstacle demonstration, but
    # avoids a fixed offset that would leave N=10 almost unperturbed.
    lateral_offset = max(1.0, min(3.5, 0.25 * formation_height))
    forward_edge = x[:, 0].max()
    forward_offsets = (10.0, 24.0) if stationary_obstacles else (28.0, 50.0)
    obs_x = np.array([
        [forward_edge + forward_offsets[0], centre0[1] + lateral_offset],
        [forward_edge + forward_offsets[1], centre0[1] - lateral_offset],
    ])
    if stationary_obstacles:
        obs_vel = np.zeros((2, 2))
    else:
        obstacle_speed = obstacle_speed_ratio * c.v0
        obs_vel = np.array([
            [-obstacle_speed, 0.0],
            [-obstacle_speed, 0.0],
        ])
    obs_heading = np.arctan2(obs_vel[:, 1], obs_vel[:, 0])

    centred0 = x - centre0
    baseline_radius = float(
        np.sqrt(np.mean(np.sum(centred0 * centred0, axis=1)))
    )
    peak_radius = baseline_radius
    body_diameter = np.hypot(c.agent_length, c.agent_width)
    visible_obstacle_clearance = 0.40
    obs_radius = np.full(2, body_diameter + visible_obstacle_clearance)
    hard_agent_distance = body_diameter + 0.18
    hard_obstacle_distance = body_diameter + visible_obstacle_clearance
    minimum_agent_clearance = np.inf
    minimum_obstacle_clearance = np.inf
    minimum_polarization = 1.0

    fig, ax = plt.subplots(figsize=(10.4, 6.4))
    ax.set_aspect("equal")
    ax.set_facecolor("#f7f8f4")
    ax.grid(True, color="#d7ddd7", linewidth=0.55, alpha=0.75)
    ax.set_xlabel("global x")
    ax.set_ylabel("global y")
    motion_label = (
        "static obstacles"
        if stationary_obstacles
        else f"moving obstacles {obstacle_speed_ratio:g}x"
    )
    obstacle_kind = "static obstacle" if stationary_obstacles else "moving obstacle"
    ax.set_title(
        f"Present controller: {motion_label} and recovery | N={agent_count}"
    )

    speed_norm = plt.Normalize(c.v_min, c.v_max)
    agents = PolyCollection(
        _vehicle_polygons(x, heading, c.agent_length, c.agent_width),
        facecolors=plt.cm.Blues(speed_norm(speed)),
        edgecolors="#17365d",
        linewidths=0.42,
        zorder=4,
    )
    ax.add_collection(agents)

    obstacle_colours = ("#e4572e", "#7b2cbf")
    obstacle_rectangles = []
    for index in range(2):
        rectangle = Rectangle(
            (-0.5 * c.agent_length, -0.5 * c.agent_width),
            c.agent_length,
            c.agent_width,
            facecolor=obstacle_colours[index],
            edgecolor="black",
            linewidth=0.85,
            alpha=0.9,
            zorder=3,
        )
        ax.add_patch(rectangle)
        obstacle_rectangles.append(rectangle)

    centre_marker = ax.scatter(
        [], [], marker="x", s=55, color="#111111", zorder=5
    )
    status = ax.text(
        0.01, 0.99, "", transform=ax.transAxes, va="top", ha="left"
    )
    ax.legend(
        handles=[
            Patch(
                facecolor=plt.cm.Blues(0.65), edgecolor="#17365d",
                label="swarm agent",
            ),
            Patch(
                facecolor=obstacle_colours[0], edgecolor="black",
                label=f"{obstacle_kind} 1",
            ),
            Patch(
                facecolor=obstacle_colours[1], edgecolor="black",
                label=f"{obstacle_kind} 2",
            ),
        ],
        loc="lower right",
    )

    step_number = 0

    def update_view():
        centre = x.mean(axis=0)
        ax.set_xlim(centre[0] - 43.0, centre[0] + 43.0)
        ax.set_ylim(centre[1] - 22.0, centre[1] + 22.0)
        for index, rectangle in enumerate(obstacle_rectangles):
            rectangle.set_transform(
                Affine2D()
                .rotate_deg(np.degrees(obs_heading[index]))
                .translate(*obs_x[index])
                + ax.transData
            )

    def current_phase(centre):
        relative_x = obs_x[:, 0] - centre[0]
        if np.any(np.abs(relative_x) <= 12.0):
            return "avoidance / flock deformation"
        if np.max(relative_x) > 12.0:
            return "approach"
        return "post-obstacle self-healing"

    def animate(_):
        nonlocal x, heading, speed, obs_x, step_number, peak_radius
        nonlocal minimum_agent_clearance, minimum_obstacle_clearance
        nonlocal minimum_polarization

        step_number += 1
        old_x = x.copy()
        old_obs_x = obs_x.copy()
        centre = x.mean(axis=0)
        out = ctl.commands(
            x,
            heading,
            speed,
            obs_x,
            obs_vel,
            obs_radius,
            rejoin_reference=centre,
        )
        x, heading, speed = plant_step(
            x,
            heading,
            speed,
            out,
            c,
            variable_speed=True,
        )
        proposed_x = x.copy()
        proposed_obs_x = old_obs_x + obs_vel * c.dt
        substeps = 4
        agent_delta = (proposed_x - old_x) / substeps
        obstacle_delta = (proposed_obs_x - old_obs_x) / substeps
        safe_x = old_x.copy()
        safe_obs_x = old_obs_x.copy()
        for _ in range(substeps):
            safe_x += agent_delta
            safe_obs_x += obstacle_delta
            safe_x = apply_safety_shield(
                safe_x,
                safe_obs_x,
                hard_agent_distance,
                hard_obstacle_distance,
            )
        x = safe_x
        obs_x = safe_obs_x

        centre = x.mean(axis=0)
        centred = x - centre
        radius = float(np.sqrt(np.mean(np.sum(centred * centred, axis=1))))
        peak_radius = max(peak_radius, radius)
        compactness = 100.0 * baseline_radius / max(radius, 1e-12)
        polarization = float(np.linalg.norm(heading.mean(axis=0)))
        minimum_polarization = min(minimum_polarization, polarization)

        pair_distance = np.linalg.norm(
            x[:, None, :] - x[None, :, :], axis=2
        )
        np.fill_diagonal(pair_distance, np.inf)
        agent_clearance = float(pair_distance.min() - body_diameter)
        obstacle_clearance = float(np.min(
            np.linalg.norm(x[:, None, :] - obs_x[None, :, :], axis=2)
            - body_diameter
        ))
        minimum_agent_clearance = min(
            minimum_agent_clearance, agent_clearance
        )
        minimum_obstacle_clearance = min(
            minimum_obstacle_clearance, obstacle_clearance
        )

        agents.set_verts(
            _vehicle_polygons(x, heading, c.agent_length, c.agent_width)
        )
        agents.set_facecolors(plt.cm.Blues(speed_norm(speed)))
        centre_marker.set_offsets(centre.reshape(1, 2))
        update_view()

        status.set_text(
            f"step = {step_number}/{steps}   phase = {current_phase(centre)}\n"
            f"obstacle speed = {obstacle_speed_ratio:g}x nominal\n"
            f"agent speed mean/min/max = {speed.mean():.2f} / "
            f"{speed.min():.2f} / {speed.max():.2f}\n"
            f"heading coherence = {polarization:.3f} "
            f"(run minimum {minimum_polarization:.3f})\n"
            f"flock RMS radius = {radius:.2f}   compactness vs start = "
            f"{compactness:.1f}%\n"
            f"clearance now = {agent_clearance:.2f} agent / "
            f"{obstacle_clearance:.2f} obstacle\n"
            f"run minima = {minimum_agent_clearance:.2f} / "
            f"{minimum_obstacle_clearance:.2f}"
        )
        return [
            agents, centre_marker, status, *obstacle_rectangles
        ]

    update_view()

    def init_animation():
        centre_marker.set_offsets(centre0.reshape(1, 2))
        status.set_text(
            f"step = 0/{steps}   phase = approach\n"
            f"obstacle speed = {obstacle_speed_ratio:g}x nominal\n"
            f"flock RMS radius = {baseline_radius:.2f}   "
            "compactness vs start = 100.0%"
        )
        return [
            agents, centre_marker, status, *obstacle_rectangles
        ]

    animation = FuncAnimation(
        fig,
        animate,
        frames=range(steps),
        interval=int(c.dt * 1000),
        init_func=init_animation,
        cache_frame_data=False,
        blit=False,
        repeat=False,
    )
    plt.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(
        fps=fps,
        metadata={
            "title": "Present controller obstacle encounter and self-healing",
            "comment": (
                f"agents={agent_count}; steps={steps}; "
                f"obstacle_speed_ratio={obstacle_speed_ratio:g}; "
                "obstacles=2; visible_clearance=0.4"
            ),
        },
        bitrate=2200,
    )
    print(
        f"Exporting {output} (N={agent_count}, {steps} steps) ...",
        flush=True,
    )
    animation.save(str(output), writer=writer, dpi=dpi)
    plt.close(fig)
    print(
        f"Saved {output.resolve()} | minimum clearances: "
        f"agent-agent={minimum_agent_clearance:.6f}, "
        f"agent-obstacle={minimum_obstacle_clearance:.6f}; "
        f"peak RMS radius={peak_radius:.6f}; "
        f"minimum polarization={minimum_polarization:.6f}",
        flush=True,
    )


# ============================================================
# One matched N,seed bundle
# ============================================================

def run_bundle(task):
    N, seed = task

    # ========================================================
    # SAFETY EXPERIMENT INITIAL CONDITIONS
    # ========================================================

    rng = np.random.default_rng(
        100000 + 1009 * N + seed
    )

    x_safe = make_agent_formation(
        rng,
        N,
    )

    theta = rng.normal(
        0.0,
        0.035,
        N,
    )

    h_safe = np.column_stack([
        np.cos(theta),
        np.sin(theta),
    ])

    flock_c = x_safe.mean(axis=0)

    # Two independent oncoming moving obstacles.
    #
    # Both variants and the present model see exactly
    # the same trajectories for a given N and seed.

    v0_reference = Config().v0
    u = (
        OBSTACLE_SPEED_RATIO
        * v0_reference
    )

    obs_x0 = np.array([
        flock_c + [
            28.0,
            rng.uniform(-3.0, 3.0)
        ],
        flock_c + [
            50.0,
            rng.uniform(-3.5, 3.5)
        ],
    ], dtype=float)

    obs_vel = np.array([
        [
            -u * rng.uniform(0.92, 1.08),
            rng.uniform(-0.30, 0.30)
        ],
        [
            -u * rng.uniform(0.92, 1.08),
            rng.uniform(-0.30, 0.30)
        ],
    ], dtype=float)

    obs_radius = np.full(
        2,
        Config().obstacle_radius,
    )

    # ========================================================
    # RECOVERY EXPERIMENT INITIAL CONDITIONS
    # ========================================================

    rng_rec = np.random.default_rng(
        300000 + 1013 * N + seed
    )

    x_rec = make_agent_formation(
        rng_rec,
        N,
    )

    theta_rec = rng_rec.normal(
        0.0,
        0.025,
        N,
    )

    h_rec = np.column_stack([
        np.cos(theta_rec),
        np.sin(theta_rec),
    ])

    # ========================================================
    # Run all models on matched initial conditions
    # ========================================================

    result = {}

    for model in MODELS:

        safety = simulate_safety(
            model,
            x_safe,
            h_safe,
            obs_x0,
            obs_vel,
            obs_radius,
        )

        recovery = simulate_recovery(
            model,
            x_rec,
            h_rec,
        )

        result[model] = {
            "dmin_obs": safety[0],
            "dmin_agent": safety[1],
            "recovery_burden": recovery[0],
            "recovery_time": recovery[1],
        }

    return N, seed, result


# ============================================================
# Statistics
# ============================================================

def bootstrap_median_ci(
    values,
    n_boot=4000,
    seed=12345,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    median = np.median(values)

    rng = np.random.default_rng(seed)

    idx = rng.integers(
        0,
        len(values),
        size=(n_boot, len(values)),
    )

    boot = np.median(
        values[idx],
        axis=1,
    )

    lo, hi = np.percentile(
        boot,
        [2.5, 97.5],
    )

    return median, lo, hi


# ============================================================
# Publication plotting
# ============================================================

def configure_matplotlib():

    import matplotlib.pyplot as plt

    latex_available = (
        shutil.which("latex")
        is not None
    )

    plt.rcParams.update({

        # -----------------------------------------
        # LaTeX / typography
        # -----------------------------------------
        "text.usetex": latex_available,
        "font.family": "serif",
        "font.serif": [
            "Computer Modern Roman",
            "CMU Serif",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "cm",

        # -----------------------------------------
        # Sizes suitable for journal figures
        # -----------------------------------------
        "font.size": 9.0,
        "axes.labelsize": 9.5,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.0,

        # -----------------------------------------
        # Lines
        # -----------------------------------------
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.7,
        "lines.markersize": 5.5,

        # -----------------------------------------
        # Ticks
        # -----------------------------------------
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,

        # -----------------------------------------
        # PDF / PS
        # -----------------------------------------
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # -----------------------------------------
        # Saving
        # -----------------------------------------
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
    })

    if latex_available:
        plt.rcParams[
            "text.latex.preamble"
        ] = (
            r"\usepackage{amsmath}"
            r"\usepackage{bm}"
        )
    else:
        print(
            "\n[plotting] LaTeX executable not found; "
            "using Matplotlib's Computer-Modern math renderer.\n"
        )

    return plt


def make_plots(results, nseeds):

    plt = configure_matplotlib()

    # Colorblind-friendly palette
    colors = {
        "Non-Markovian [22]": "#0072B2",
        "Self-Healing [23]": "#E69F00",
        "Present Controller": "#009E73",
    }

    markers = {
        "Non-Markovian [22]": "o",
        "Self-Healing [23]": "s",
        "Present Controller": "^",
    }

    legend_names = {
        "Non-Markovian [22]":
            r"Non-Markovian [22]",

        "Self-Healing [23]":
            r"Self-Healing [23]",

        "Present Controller":
            r"\textbf{Present Controller}",
    }

    x = np.arange(
        len(N_VALUES),
        dtype=float,
    )

    offsets = {
        MODELS[0]: -0.12,
        MODELS[1]:  0.00,
        MODELS[2]:  0.12,
    }

    # ========================================================
    # Two-panel main figure
    # ========================================================

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.15, 3.05),
    )

    ax1, ax2 = axes

    rng_jitter = np.random.default_rng(881)

    # ========================================================
    # Panel A: minimum obstacle clearance
    # ========================================================

    for model in MODELS:

        med = []
        lo = []
        hi = []

        for N in N_VALUES:

            vals = results[
                (model, N)
            ]["dmin_obs"]

            m, l, h = bootstrap_median_ci(
                vals,
                seed=7000 + N,
            )

            med.append(m)
            lo.append(l)
            hi.append(h)

            # Individual runs, lightly shown
            jitter = rng_jitter.normal(
                0.0,
                0.018,
                len(vals),
            )

            ax1.scatter(
                np.full(len(vals), x[
                    N_VALUES.index(N)
                ] + offsets[model])
                + jitter,
                vals,
                s=7,
                alpha=0.17,
                color=colors[model],
                linewidths=0,
                zorder=1,
            )

        med = np.asarray(med)
        lo = np.asarray(lo)
        hi = np.asarray(hi)

        ax1.errorbar(
            x + offsets[model],
            med,
            yerr=[
                med - lo,
                hi - med
            ],
            marker=markers[model],
            color=colors[model],
            capsize=3,
            capthick=1.0,
            linewidth=1.6,
            label=legend_names[model],
            zorder=4,
        )

    ax1.axhline(
        0.0,
        color="0.35",
        linestyle="--",
        linewidth=0.9,
        zorder=0,
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [str(n) for n in N_VALUES]
    )

    ax1.set_xlabel(
        r"Swarm size, $N$"
    )

    ax1.set_ylabel(
        r"Minimum obstacle clearance, "
        r"$d_{\min}^{\rm obs}$"
    )

    ax1.text(
        0.02,
        0.97,
        r"\textbf{(a)}",
        transform=ax1.transAxes,
        va="top",
        ha="left",
    )

    ax1.grid(
        axis="y",
        linewidth=0.45,
        alpha=0.22,
    )

    # ========================================================
    # Panel B: post-perturbation recovery burden
    # ========================================================

    for model in MODELS:

        med = []
        lo = []
        hi = []

        for N in N_VALUES:

            vals = results[
                (model, N)
            ]["recovery_burden"]

            m, l, h = bootstrap_median_ci(
                vals,
                seed=9000 + N,
            )

            med.append(m)
            lo.append(l)
            hi.append(h)

            jitter = rng_jitter.normal(
                0.0,
                0.018,
                len(vals),
            )

            ax2.scatter(
                np.full(len(vals), x[
                    N_VALUES.index(N)
                ] + offsets[model])
                + jitter,
                vals,
                s=7,
                alpha=0.17,
                color=colors[model],
                linewidths=0,
                zorder=1,
            )

        med = np.asarray(med)
        lo = np.asarray(lo)
        hi = np.asarray(hi)

        ax2.errorbar(
            x + offsets[model],
            med,
            yerr=[
                med - lo,
                hi - med
            ],
            marker=markers[model],
            color=colors[model],
            capsize=3,
            capthick=1.0,
            linewidth=1.6,
            label=legend_names[model],
            zorder=4,
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [str(n) for n in N_VALUES]
    )

    ax2.set_xlabel(
        r"Swarm size, $N$"
    )

    ax2.set_ylabel(
        r"Recovery burden, "
        r"$B_{\rm rec}$ [s]"
    )

    ax2.text(
        0.02,
        0.97,
        r"\textbf{(b)}",
        transform=ax2.transAxes,
        va="top",
        ha="left",
    )

    ax2.grid(
        axis="y",
        linewidth=0.45,
        alpha=0.22,
    )

    # Remove top/right spines for cleaner publication style
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Common legend
    handles, labels = ax1.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.035),
        ncol=3,
        frameon=False,
        columnspacing=1.3,
        handletextpad=0.5,
    )

    fig.subplots_adjust(
        top=0.81,
        bottom=0.18,
        left=0.10,
        right=0.985,
        wspace=0.34,
    )

    fig.savefig(
        "comparison_two_panel.pdf"
    )

    fig.savefig(
        "comparison_two_panel.png",
        dpi=500,
    )

    # ========================================================
    # Also save panels separately
    # ========================================================

    for which, metric, ylabel, filename in [

        (
            "safety",
            "dmin_obs",
            r"Minimum obstacle clearance, "
            r"$d_{\min}^{\rm obs}$",
            "comparison_safety.pdf",
        ),

        (
            "recovery",
            "recovery_burden",
            r"Recovery burden, "
            r"$B_{\rm rec}$ [s]",
            "comparison_recovery.pdf",
        ),
    ]:

        fig_single, ax = plt.subplots(
            figsize=(3.45, 2.8)
        )

        for model in MODELS:

            med = []
            lo = []
            hi = []

            for N in N_VALUES:

                vals = results[
                    (model, N)
                ][metric]

                m, l, h = bootstrap_median_ci(
                    vals,
                    seed=13000 + N,
                )

                med.append(m)
                lo.append(l)
                hi.append(h)

            med = np.asarray(med)
            lo = np.asarray(lo)
            hi = np.asarray(hi)

            ax.errorbar(
                x + offsets[model],
                med,
                yerr=[
                    med - lo,
                    hi - med
                ],
                marker=markers[model],
                color=colors[model],
                capsize=3,
                linewidth=1.6,
                label=legend_names[model],
            )

        if which == "safety":
            ax.axhline(
                0,
                color="0.35",
                linestyle="--",
                linewidth=0.9,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [str(n) for n in N_VALUES]
        )

        ax.set_xlabel(
            r"Swarm size, $N$"
        )
        ax.set_ylabel(ylabel)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.grid(
            axis="y",
            alpha=0.22,
            linewidth=0.45,
        )

        ax.legend(
            frameon=False,
            fontsize=7.1,
        )

        fig_single.tight_layout()

        fig_single.savefig(filename)

        plt.close(fig_single)

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--save-recovery-videos",
        action="store_true",
        help=(
            "export Present Controller obstacle-encounter/recovery videos "
            "for N=10, 20, and 30, then exit"
        ),
    )

    parser.add_argument(
        "--video-output-dir",
        type=Path,
        default=Path("comparison2_present_controller_videos"),
        help=(
            "folder for recovery MP4 files "
            "(default: comparison2_present_controller_videos)"
        ),
    )

    parser.add_argument(
        "--video-steps",
        type=int,
        default=1000,
        help="simulation steps per recovery video (default: 1000)",
    )

    parser.add_argument(
        "--video-fps",
        type=int,
        default=25,
        help="recovery-video playback frame rate (default: 25)",
    )

    parser.add_argument(
        "--video-dpi",
        type=int,
        default=100,
        help="recovery-video resolution in dots per inch (default: 100)",
    )

    parser.add_argument(
        "--video-seed",
        type=int,
        default=2026,
        help="base random seed for recovery videos (default: 2026)",
    )

    parser.add_argument(
        "--video-obstacle-speed-ratios",
        nargs="+",
        type=float,
        default=[0.0],
        help=(
            "obstacle-speed multiples of nominal agent speed; 0 is static "
            "(default: 0)"
        ),
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=DEFAULT_NSEEDS,
        help=(
            "number of matched seeds per swarm size "
            f"(default: {DEFAULT_NSEEDS})"
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "number of parallel CPU workers; "
            "default uses all but one core"
        ),
    )

    args = parser.parse_args()

    if args.save_recovery_videos:
        if args.video_steps < 1 or args.video_fps < 1 or args.video_dpi < 1:
            parser.error("video steps, fps, and dpi must all be positive")
        if any(ratio < 0.0 for ratio in args.video_obstacle_speed_ratios):
            parser.error("video obstacle speed ratios cannot be negative")
        if shutil.which("ffmpeg") is None:
            parser.error("video export requires FFmpeg")
        args.video_output_dir.mkdir(parents=True, exist_ok=True)
        for obstacle_speed_ratio in args.video_obstacle_speed_ratios:
            speed_label = f"{obstacle_speed_ratio:g}x".replace(".", "p")
            ratio_output_dir = (
                args.video_output_dir
                if obstacle_speed_ratio == 0.0
                else args.video_output_dir / f"speed_{speed_label}"
            )
            ratio_output_dir.mkdir(parents=True, exist_ok=True)
            for agent_count in N_VALUES:
                speed_suffix = (
                    ""
                    if obstacle_speed_ratio == 0.0
                    else f"_speed_{speed_label}"
                )
                filename = (
                    f"present_controller_recovery_N{agent_count}"
                    f"{speed_suffix}_{args.video_steps}_steps.mp4"
                )
                export_present_recovery_video(
                    agent_count=agent_count,
                    steps=args.video_steps,
                    output=ratio_output_dir / filename,
                    fps=args.video_fps,
                    dpi=args.video_dpi,
                    seed=args.video_seed,
                    obstacle_speed_ratio=obstacle_speed_ratio,
                )
        return

    from tqdm import tqdm

    nseeds = args.seeds

    total_tasks = (
        len(N_VALUES)
        * nseeds
    )

    cpu_count = (
        os.cpu_count()
        or 2
    )

    if args.workers is None:
        workers = max(
            1,
            min(
                cpu_count - 1,
                total_tasks,
            )
        )
    else:
        workers = max(
            1,
            args.workers,
        )

    print()
    print("=" * 68)
    print(" COGNITIVE SWARM MATCHED BENCHMARK")
    print("=" * 68)

    print(
        "Swarm sizes        : "
        + ", ".join(
            str(n)
            for n in N_VALUES
        )
    )

    print(
        f"Seeds / swarm size : {nseeds}"
    )

    print(
        f"Model variants     : {len(MODELS)}"
    )

    print(
        f"Experiments/model  : 2"
    )

    print(
        f"Parallel jobs      : {total_tasks}"
    )

    print(
        f"CPU workers        : {workers}"
    )

    print(
        f"Safety horizon     : {T_SAFETY:.1f} s"
    )

    print(
        f"Recovery horizon   : {T_RECOVERY:.1f} s"
    )

    print(
        "Obstacle ratio     : "
        rf"u_obs/v0 = {OBSTACLE_SPEED_RATIO:.2f}"
    )

    print("=" * 68)
    print()

    start = time.perf_counter()

    # ========================================================
    # Storage
    # ========================================================

    results = {}

    for model in MODELS:
        for N in N_VALUES:
            results[(model, N)] = {
                "dmin_obs":
                    np.full(nseeds, np.nan),

                "dmin_agent":
                    np.full(nseeds, np.nan),

                "recovery_burden":
                    np.full(nseeds, np.nan),

                "recovery_time":
                    np.full(nseeds, np.nan),
            }

    tasks = [
        (N, seed)
        for N in N_VALUES
        for seed in range(nseeds)
    ]

    # ========================================================
    # Parallel execution
    # ========================================================

    futures = {}

    with ProcessPoolExecutor(
        max_workers=workers
    ) as executor:

        for task in tasks:

            future = executor.submit(
                run_bundle,
                task,
            )

            futures[future] = task

        with tqdm(
            total=len(tasks),
            desc="Matched benchmark",
            unit="seed",
            dynamic_ncols=True,
            smoothing=0.08,
        ) as bar:

            for future in as_completed(
                futures
            ):

                N, seed, bundle = (
                    future.result()
                )

                for model in MODELS:

                    results[
                        (model, N)
                    ]["dmin_obs"][seed] = (
                        bundle[model][
                            "dmin_obs"
                        ]
                    )

                    results[
                        (model, N)
                    ]["dmin_agent"][seed] = (
                        bundle[model][
                            "dmin_agent"
                        ]
                    )

                    results[
                        (model, N)
                    ]["recovery_burden"][seed] = (
                        bundle[model][
                            "recovery_burden"
                        ]
                    )

                    results[
                        (model, N)
                    ]["recovery_time"][seed] = (
                        bundle[model][
                            "recovery_time"
                        ]
                    )

                bar.update(1)

    elapsed = (
        time.perf_counter()
        - start
    )

    print()
    print(
        f"Simulation finished in "
        f"{elapsed:.1f} s "
        f"({elapsed / 60:.2f} min)"
    )

    # ========================================================
    # CSV
    # ========================================================

    with open(
        "benchmark_results.csv",
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "N",
            "seed",
            "model",
            "minimum_obstacle_clearance",
            "minimum_agent_distance",
            "recovery_burden",
            "recovery_time",
        ])

        for N in N_VALUES:

            for seed in range(nseeds):

                for model in MODELS:

                    r = results[
                        (model, N)
                    ]

                    writer.writerow([
                        N,
                        seed,
                        model,
                        r["dmin_obs"][seed],
                        r["dmin_agent"][seed],
                        r["recovery_burden"][seed],
                        r["recovery_time"][seed],
                    ])

    # ========================================================
    # Terminal summary
    # ========================================================

    print()
    print("=" * 112)

    print(
        f"{'N':>3s}  "
        f"{'Model':29s}"
        f"{'d_obs median':>15s}"
        f"{'Safe runs':>12s}"
        f"{'B_rec median':>16s}"
        f"{'Recovered':>12s}"
        f"{'T_rec median':>16s}"
    )

    print("-" * 112)

    for N in N_VALUES:

        for model in MODELS:

            r = results[
                (model, N)
            ]

            clearance = r["dmin_obs"]
            burden = r[
                "recovery_burden"
            ]
            trec = r[
                "recovery_time"
            ]

            safe_fraction = np.mean(
                clearance > 0.0
            )

            recovered = np.isfinite(
                trec
            )

            if recovered.any():
                med_trec = np.nanmedian(
                    trec
                )
            else:
                med_trec = np.nan

            print(
                f"{N:3d}  "
                f"{model:29s}"
                f"{np.median(clearance):15.3f}"
                f"{100*safe_fraction:11.1f}%"
                f"{np.median(burden):16.3f}"
                f"{100*recovered.mean():11.1f}%"
                f"{med_trec:16.3f}"
            )

    print("=" * 112)

    # ========================================================
    # Figures
    # ========================================================

    make_plots(
        results,
        nseeds,
    )

    print()
    print("Saved:")
    print("  benchmark_results.csv")
    print("  comparison_two_panel.pdf")
    print("  comparison_two_panel.png")
    print("  comparison_safety.pdf")
    print("  comparison_recovery.pdf")
    print()


# Required for multiprocessing on macOS / Windows
if __name__ == "__main__":
    main()
