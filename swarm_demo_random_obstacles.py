"""Two-obstacle experiments for the cognitive swarm.

The two obstacles are the same size as a flock agent but use contrasting
colours. Their moving targets are anchored to the instantaneous flock
centroid, so they remain with the flock and make repeated smooth intrusions.
The default batch exports N=50, 100, and 200 at 0.5x, 1x, and 1.5x the minimum
centroid-relative obstacle speed. A final geometric safety shield guarantees
that circumscribed agent bodies do not overlap. Every movie contains 1,000
simulation steps. The stationary variant places two staggered obstacles close
to the initial forward path with a realistic visible clearance buffer.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, writers
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.transforms import Affine2D
import numpy as np

from swarm_demo import CognitiveSwarm, Config


DEFAULT_AGENT_COUNTS = (50, 100, 200)
DEFAULT_SPEED_RATIOS = (0.5, 1.0, 1.5)


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
    defaults = Config()
    parser = argparse.ArgumentParser(
        description="Export random-obstacle cognitive-swarm experiments."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("collision_free_obstacle_videos"),
        help="directory for MP4 files (default: collision_free_obstacle_videos)",
    )
    parser.add_argument(
        "--agents", nargs="+", type=positive_int,
        default=list(DEFAULT_AGENT_COUNTS),
        help="agent counts to export (default: 50 100 200)",
    )
    parser.add_argument(
        "--speed-ratios", nargs="+", type=positive_float,
        default=list(DEFAULT_SPEED_RATIOS),
        help="multipliers applied to the minimum centroid-relative obstacle speed",
    )
    parser.add_argument(
        "--obstacle-speed-min", type=positive_float,
        default=defaults.obstacle_speed_min,
        help=f"minimum obstacle speed (default: {defaults.obstacle_speed_min})",
    )
    parser.add_argument(
        "--steps", type=positive_int, default=1000,
        help="simulation steps per movie (default: 1000)",
    )
    parser.add_argument(
        "--stationary-obstacle", action="store_true",
        help=(
            "use two zero-speed obstacles fixed near the initial swarm path; "
            "speed ratios are ignored"
        ),
    )
    parser.add_argument(
        "--static-obstacle-count", type=int, choices=(0, 1, 2), default=None,
        help=(
            "run a fixed-obstacle case with 0, 1, or 2 obstacles; speed "
            "ratios are ignored"
        ),
    )
    parser.add_argument("--fps", type=positive_int, default=15)
    parser.add_argument("--dpi", type=positive_int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    if args.stationary_obstacle and args.static_obstacle_count is not None:
        parser.error(
            "use either --stationary-obstacle or --static-obstacle-count, not both"
        )
    return args


def make_formation(rng, count):
    """Create a compact near-square flock with safe initial spacing."""
    columns = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / columns))
    # Scaled for the 32 m square: N=200 spans about 28 m rather than 42 m.
    spacing = 2.0
    grid = []
    for row in range(rows):
        for column in range(columns):
            if len(grid) == count:
                break
            grid.append([
                (column - 0.5 * (columns - 1)) * spacing,
                (row - 0.5 * (rows - 1)) * spacing,
            ])
    return np.asarray(grid) + rng.normal(0.0, 0.08, (count, 2))


def vehicle_polygons(centres, directions, length, width):
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
        polygons.append(local @ np.column_stack([direction, normal]).T + centre)
    return polygons


def ratio_label(ratio):
    return f"{ratio:g}x".replace(".", "p")


def apply_safety_shield(agent_pos, obstacle_pos, agent_distance,
                        obstacle_distance, iterations=40):
    """Project positions onto conservative non-overlap constraints.

    Agent-agent corrections are shared equally. Obstacles retain their
    prescribed trajectories, so agent-obstacle corrections move only agents.
    Circumscribed-body distances make this independent of vehicle heading.
    """
    corrected = agent_pos.copy()
    count = len(corrected)
    upper_i, upper_j = np.triu_indices(count, k=1)

    for _ in range(iterations):
        shifts = np.zeros_like(corrected)
        pair_delta = corrected[upper_i] - corrected[upper_j]
        pair_dist = np.linalg.norm(pair_delta, axis=1)
        close = pair_dist < agent_distance
        max_overlap = 0.0
        if np.any(close):
            distance = pair_dist[close]
            delta = pair_delta[close]
            # Deterministic fallback is only relevant for exactly coincident
            # centres, which cannot occur in the initialized formation.
            direction = np.divide(
                delta, distance[:, None], out=np.tile([1.0, 0.0], (len(delta), 1)),
                where=distance[:, None] > 1e-12,
            )
            overlap = agent_distance - distance
            correction = 0.52 * overlap[:, None] * direction
            np.add.at(shifts, upper_i[close], correction)
            np.add.at(shifts, upper_j[close], -correction)
            max_overlap = float(overlap.max())

        obs_delta = corrected[:, None, :] - obstacle_pos[None, :, :]
        obs_dist = np.linalg.norm(obs_delta, axis=2)
        agent_index, obstacle_index = np.where(obs_dist < obstacle_distance)
        if len(agent_index):
            distance = obs_dist[agent_index, obstacle_index]
            delta = obs_delta[agent_index, obstacle_index]
            direction = np.divide(
                delta, distance[:, None], out=np.tile([0.0, 1.0], (len(delta), 1)),
                where=distance[:, None] > 1e-12,
            )
            overlap = obstacle_distance - distance
            np.add.at(shifts, agent_index, 1.04 * overlap[:, None] * direction)
            max_overlap = max(max_overlap, float(overlap.max()))

        corrected += shifts
        if max_overlap < 1e-8:
            break

    return corrected


def export_case(args, agent_count, speed_ratio, output):
    cfg = Config()
    # Preserve the original geometry/spacing ratios inside the smaller scene.
    spatial_scale = 2.0 / 3.0
    for attribute in (
        "sensing_radius",
        "safe_agent_distance",
        "obstacle_margin",
        "rejoin_length",
        "agent_length",
        "agent_width",
        "obstacle_length",
        "obstacle_width",
        "obstacle_radius",
    ):
        setattr(cfg, attribute, getattr(cfg, attribute) * spatial_scale)
    # Keep the controller tuning identical to swarm_demo.py.  This experiment
    # changes the obstacle geometry and motion.  Static objects use a shorter
    # response range because their zero velocity makes them fully predictable.
    stationary_obstacle = (
        args.stationary_obstacle or args.static_obstacle_count is not None
    )
    static_obstacle_count = (
        args.static_obstacle_count
        if args.static_obstacle_count is not None
        else 2
    )
    if stationary_obstacle:
        cfg.obstacle_margin = 2.0 * spatial_scale
    obstacle_speed = 0.0 if stationary_obstacle else args.obstacle_speed_min * speed_ratio

    case_seed = args.seed + 1009 * agent_count + int(round(100 * speed_ratio))
    rng = np.random.default_rng(case_seed)
    pos = make_formation(rng, agent_count)
    theta = rng.normal(0.0, 0.035, agent_count)
    heading = np.column_stack([np.cos(theta), np.sin(theta)])
    speed = np.full(agent_count, cfg.v0)
    controller = CognitiveSwarm(agent_count, cfg)

    obstacle_count = static_obstacle_count if stationary_obstacle else 2
    obs_pos = np.zeros((obstacle_count, 2))
    obs_vel = np.zeros((obstacle_count, 2))
    obs_heading = np.zeros(obstacle_count)
    obs_speed = np.zeros(obstacle_count)
    obs_turn_rate = np.zeros(obstacle_count)
    # The controller treats agents as points, so its obstacle radius includes
    # both equal-sized vehicle circumradii (one body diagonal) and a small
    # visible clearance buffer.  The safety shield uses the same envelope;
    # consequently it cannot permit a visually ambiguous near-touch.
    body_diameter = np.hypot(cfg.agent_length, cfg.agent_width)
    # Roughly 0.75 body widths of free space between circumscribed envelopes:
    # enough to look natural while remaining visibly and numerically safe.
    crowd_scale = np.clip((agent_count - 50) / 150.0, 0.0, 1.0)
    visible_obstacle_clearance = (
        0.40 + 0.30 * crowd_scale
    ) * spatial_scale
    obs_radius = np.full(
        obstacle_count, body_diameter + visible_obstacle_clearance
    )
    hard_agent_distance = body_diameter + (
        0.18 + 0.22 * crowd_scale
    ) * spatial_scale
    hard_obstacle_distance = body_diameter + visible_obstacle_clearance

    formation_width = np.ptp(pos[:, 0])
    formation_height = np.ptp(pos[:, 1])
    half_width = 16.0
    half_height = 16.0
    centre0 = pos.mean(axis=0)
    formation_x_extent = max(
        centre0[0] - pos[:, 0].min(),
        pos[:, 0].max() - centre0[0],
    )
    orbit_x = max(
        min(
            0.62 * half_width,
            0.5 * formation_width + 10.0 * spatial_scale,
        ),
        formation_x_extent + hard_obstacle_distance + 1.0 * spatial_scale,
    )
    orbit_y = min(
        0.60 * half_height,
        0.5 * formation_height + 8.0 * spatial_scale,
    )
    if stationary_obstacle:
        phase = np.zeros(obstacle_count)
        # Stagger the fixed obstacles in world coordinates.  Their lateral
        # offsets keep them close to (and partly inside) the flock corridor,
        # while leaving a natural route around each vehicle.
        forward_edge = pos[:, 0].max()
        lateral_offset = max(
            2.8 * spatial_scale,
            min(5.5 * spatial_scale, 0.18 * formation_height),
        )
        if obstacle_count == 1:
            obs_pos[0] = [
                forward_edge + 14.0 * spatial_scale,
                centre0[1],
            ]
        elif obstacle_count == 2:
            obs_pos[:] = np.array([
                [
                    forward_edge + 10.0 * spatial_scale,
                    centre0[1] + lateral_offset,
                ],
                [
                    forward_edge + 24.0 * spatial_scale,
                    centre0[1] - lateral_offset,
                ],
            ])
    else:
        # Start on opposite sides beyond the full formation envelope.
        # Subsequent orbital motion may enter the flock, but no run begins
        # with an obstacle interspersed among the agents.
        phase = np.array([-0.5 * np.pi, 0.5 * np.pi])
        obs_pos[0] = centre0 + np.array([
            orbit_x * np.sin(phase[0]),
            0.55 * orbit_y * np.sin(2.0 * phase[0]),
        ])
        obs_pos[1] = centre0 + np.array([
            orbit_x * np.sin(phase[1]),
            0.55 * orbit_y * np.sin(2.0 * phase[1] + 0.5 * np.pi),
        ])
        if (
            obs_pos[0, 0] > pos[:, 0].min() - hard_obstacle_distance
            or obs_pos[1, 0] < pos[:, 0].max() + hard_obstacle_distance
        ):
            raise RuntimeError("obstacles must initialize outside the formation")
        for index, phase_direction in enumerate((1.0, -1.0)):
            tangent = phase_direction * np.array([
                orbit_x * np.cos(phase[index]),
                1.1 * orbit_y * np.cos(
                    2.0 * phase[index] + 0.5 * np.pi * index
                ),
            ])
            tangent /= np.linalg.norm(tangent)
            initial_velocity = np.array([cfg.v0, 0.0]) + obstacle_speed * tangent
            obs_speed[index] = np.linalg.norm(initial_velocity)
            obs_heading[index] = np.arctan2(
                initial_velocity[1], initial_velocity[0]
            )
            obs_vel[index] = initial_velocity
    previous_centre = centre0.copy()

    fig, ax = plt.subplots(
        figsize=(640.0 / args.dpi, 540.0 / args.dpi),
        facecolor="white",
    )
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    # The camera contains 32 x 32 one-unit grid cells.  Label every alternate
    # line on a doubled display scale, so both axes visibly span 64 units.
    ax.xaxis.set_major_locator(MultipleLocator(2.0))
    ax.yaxis.set_major_locator(MultipleLocator(2.0))
    ax.xaxis.set_minor_locator(MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(MultipleLocator(1.0))
    doubled_units = FuncFormatter(lambda value, _position: f"{2.0 * value:g}")
    ax.xaxis.set_major_formatter(doubled_units)
    ax.yaxis.set_major_formatter(doubled_units)
    ax.grid(True, which="both", color="#686868", linewidth=0.45, alpha=0.72)
    ax.set_axisbelow(True)
    ax.set_xlabel("x", fontsize=24)
    ax.set_ylabel("y", fontsize=24)
    ax.tick_params(axis="both", which="major", labelsize=8)
    if stationary_obstacle:
        case_title = f"{obstacle_count} static obstacle"
        if obstacle_count != 1:
            case_title += "s"
    else:
        case_title = f"moving obstacles, {speed_ratio:g}x speed"
    ax.set_title(f"Cognitive swarm: {case_title} | N={agent_count}", fontsize=13)
    speed_norm = plt.Normalize(cfg.v_min, cfg.v_max)
    agents = PolyCollection(
        vehicle_polygons(pos, heading, cfg.agent_length, cfg.agent_width),
        facecolors=plt.cm.Blues(speed_norm(speed)), edgecolors="#17365d",
        linewidths=0.28, zorder=4,
    )
    ax.add_collection(agents)
    centre_marker = ax.scatter([], [], marker="x", s=55, color="#111111", zorder=5)
    obstacle_colours = ("#e4572e", "#7b2cbf")
    obstacle_rectangles = []
    for index in range(obstacle_count):
        rectangle = Rectangle(
            (-cfg.agent_length / 2, -cfg.agent_width / 2),
            cfg.agent_length, cfg.agent_width,
            facecolor=obstacle_colours[index], edgecolor="black",
            linewidth=0.8, alpha=0.88, zorder=3,
        )
        ax.add_patch(rectangle)
        obstacle_rectangles.append(rectangle)

    metric_handles = [
        Line2D([], [], linestyle="none", label="min/max speed: 0.00 / 0.00"),
        Line2D([], [], linestyle="none", label="heading consensus: 1.000"),
        Line2D([], [], linestyle="none", label=r"$B_{\mathrm{rec}}$: 0.000 s"),
    ]
    metric_legend = ax.legend(
        handles=metric_handles,
        loc="upper right",
        fontsize=11,
        frameon=True,
        framealpha=0.88,
        facecolor="white",
        edgecolor="none",
        handlelength=0,
        handletextpad=0,
        borderpad=0.35,
        labelspacing=0.3,
    )
    metric_texts = metric_legend.get_texts()

    step_number = 0
    filtered_centroid_velocity = np.array([cfg.v0, 0.0])
    minimum_agent_clearance = np.inf
    minimum_obstacle_clearance = np.inf
    pre_encounter_spans = []
    recovery_baseline_span = None
    recovery_samples = []
    recovery_burden = 0.0
    encounter_started = False
    recovery_dwell_count = 0
    recovery_complete = False

    def update_view_and_obstacles():
        centre = pos.mean(axis=0)
        ax.set_xlim(centre[0] - 16.0, centre[0] + 16.0)
        ax.set_ylim(centre[1] - 16.0, centre[1] + 16.0)
        for index, rectangle in enumerate(obstacle_rectangles):
            angle = np.degrees(obs_heading[index])
            rectangle.set_transform(
                Affine2D().rotate_deg(angle).translate(*obs_pos[index]) + ax.transData
            )

    def current_phase(centre):
        if obstacle_count == 0:
            return "no obstacles"
        relative_x = obs_pos[:, 0] - centre[0]
        if np.any(np.abs(relative_x) <= 12.0):
            return "avoidance"
        if np.max(relative_x) > 12.0:
            return "approach"
        return "post-obstacle self-healing"

    def animate(_):
        nonlocal pos, heading, speed, step_number, previous_centre
        nonlocal filtered_centroid_velocity
        nonlocal minimum_agent_clearance, minimum_obstacle_clearance
        nonlocal recovery_baseline_span, recovery_burden, encounter_started
        nonlocal recovery_dwell_count, recovery_complete
        step_number += 1
        flock_centre = pos.mean(axis=0)
        centroid_velocity = (flock_centre - previous_centre) / cfg.dt
        previous_centre = flock_centre.copy()
        centroid_ease = 1.0 - np.exp(-cfg.dt / 0.8)
        filtered_centroid_velocity += centroid_ease * (
            centroid_velocity - filtered_centroid_velocity
        )

        old_obs_pos = obs_pos.copy()
        if stationary_obstacle:
            obs_vel[:] = 0.0
            proposed_obs_pos = old_obs_pos.copy()
        else:
            angular_rate = obstacle_speed / max(orbit_x, orbit_y)
            phase[:] += angular_rate * cfg.dt * np.array([1.0, -1.0])
            for index in range(obstacle_count):
                target_offset = np.array([
                    orbit_x * np.sin(phase[index]),
                    0.55 * orbit_y * np.sin(
                        2.0 * phase[index] + 0.5 * np.pi * index
                    ),
                ])
                target = flock_centre + target_offset
                delta = target - obs_pos[index]
                direction = delta / (np.linalg.norm(delta) + 1e-12)
                desired_velocity = (
                    filtered_centroid_velocity + obstacle_speed * direction
                )
                desired_heading = np.arctan2(
                    desired_velocity[1], desired_velocity[0]
                )
                heading_error = CognitiveSwarm.wrap(
                    desired_heading - obs_heading[index]
                )
                target_turn_rate = np.clip(heading_error / 0.8, -0.65, 0.65)
                turn_ease = 1.0 - np.exp(-cfg.dt / 0.45)
                obs_turn_rate[index] += turn_ease * (
                    target_turn_rate - obs_turn_rate[index]
                )
                obs_heading[index] = CognitiveSwarm.wrap(
                    obs_heading[index] + obs_turn_rate[index] * cfg.dt
                )
                target_speed = np.linalg.norm(desired_velocity)
                speed_ease = 1.0 - np.exp(-cfg.dt / 0.6)
                obs_speed[index] += speed_ease * (
                    target_speed - obs_speed[index]
                )
                obs_vel[index] = obs_speed[index] * np.array([
                    np.cos(obs_heading[index]), np.sin(obs_heading[index])
                ])
            proposed_obs_pos = old_obs_pos + obs_vel * cfg.dt

        output_commands = controller.commands(
            pos, heading, speed, old_obs_pos, obs_vel, obs_radius,
            rejoin_reference=flock_centre,
        )
        speed = np.clip(
            speed + output_commands["acceleration_setpoint"] * cfg.dt,
            cfg.v_min, cfg.v_max,
        )
        yaw = np.arctan2(heading[:, 1], heading[:, 0])
        yaw = CognitiveSwarm.wrap(
            yaw + output_commands["yaw_rate_setpoint"] * cfg.dt
        )
        heading = np.column_stack([np.cos(yaw), np.sin(yaw)])
        proposed_pos = pos + speed[:, None] * heading * cfg.dt

        # Four simultaneous motion substeps plus a conservative geometric
        # projection prevent tunnelling and body overlap, even during a direct
        # obstacle incursion. The cognitive command remains the nominal motion.
        substeps = 4
        agent_delta = (proposed_pos - pos) / substeps
        obstacle_delta = (proposed_obs_pos - old_obs_pos) / substeps
        safe_pos = pos.copy()
        safe_obs_pos = old_obs_pos.copy()
        for _ in range(substeps):
            safe_pos += agent_delta
            safe_obs_pos += obstacle_delta
            safe_pos = apply_safety_shield(
                safe_pos, safe_obs_pos,
                hard_agent_distance, hard_obstacle_distance,
            )
        pos = safe_pos
        obs_pos[:] = safe_obs_pos

        agents.set_verts(vehicle_polygons(
            pos, heading, cfg.agent_length, cfg.agent_width
        ))
        agents.set_facecolors(plt.cm.Blues(speed_norm(speed)))
        centre_marker.set_offsets(pos.mean(axis=0).reshape(1, 2))
        update_view_and_obstacles()

        obstacle_clearance = (
            np.min(
                np.linalg.norm(
                    pos[:, None, :] - obs_pos[None, :, :], axis=2
                ) - body_diameter
            )
            if obstacle_count else np.inf
        )
        pair_distance = np.linalg.norm(
            pos[:, None, :] - pos[None, :, :], axis=2
        )
        np.fill_diagonal(pair_distance, np.inf)
        agent_clearance = pair_distance.min() - body_diameter
        minimum_agent_clearance = min(
            minimum_agent_clearance, float(agent_clearance)
        )
        minimum_obstacle_clearance = min(
            minimum_obstacle_clearance, float(obstacle_clearance)
        )
        if agent_clearance < -1e-7 or obstacle_clearance < -1e-7:
            raise RuntimeError("geometric safety shield failed")
        heading_consensus = float(np.linalg.norm(heading.mean(axis=0)))
        span_x = float(np.ptp(pos[:, 0]))
        phase_name = current_phase(pos.mean(axis=0))

        # Match comparison2.py: establish the reference before avoidance,
        # accumulate only during the encounter, then feed zero samples after
        # the obstacles pass so B_rec decays instead of increasing.
        if recovery_complete and phase_name == "approach":
            encounter_started = False
            recovery_complete = False
            pre_encounter_spans.clear()
            recovery_samples.clear()
            recovery_burden = 0.0
            recovery_dwell_count = 0

        if not encounter_started and phase_name == "approach":
            pre_encounter_spans.append(span_x)
        elif not encounter_started and phase_name == "avoidance":
            history_steps = max(1, int(0.75 / cfg.dt))
            reference = pre_encounter_spans[-history_steps:]
            recovery_baseline_span = max(
                float(np.median(reference) if reference else span_x),
                1e-12,
            )
            encounter_started = True

        if encounter_started and not recovery_complete:
            normalized_deformation = abs(
                span_x / recovery_baseline_span - 1.0
            )
            sample = (
                normalized_deformation * cfg.dt
                if phase_name == "avoidance"
                else 0.0
            )
            recovery_samples.append(sample)
            window_steps = max(1, int(1.0 / cfg.dt))
            if len(recovery_samples) > window_steps:
                del recovery_samples[:-window_steps]
            recovery_burden = float(np.sum(recovery_samples))

            if (
                phase_name == "post-obstacle self-healing"
                and heading_consensus >= 0.99
            ):
                recovery_dwell_count += 1
            else:
                recovery_dwell_count = 0
            if recovery_dwell_count >= max(1, int(0.40 / cfg.dt)):
                recovery_complete = True
                recovery_samples.clear()
                recovery_burden = 0.0
        metric_texts[0].set_text(
            f"min/max speed: {speed.min():.2f} / {speed.max():.2f}"
        )
        metric_texts[1].set_text(
            f"heading consensus: {heading_consensus:.3f}"
        )
        metric_texts[2].set_text(
            rf"$B_{{\mathrm{{rec}}}}$: {recovery_burden:.3f} s"
        )
        return [agents, centre_marker, *metric_texts, *obstacle_rectangles]

    update_view_and_obstacles()

    def init_animation():
        """Draw the initial state without consuming a simulation step."""
        centre_marker.set_offsets(pos.mean(axis=0).reshape(1, 2))
        metric_texts[0].set_text(
            f"min/max speed: {speed.min():.2f} / {speed.max():.2f}"
        )
        metric_texts[1].set_text("heading consensus: 1.000")
        metric_texts[2].set_text(r"$B_{\mathrm{rec}}$: 0.000 s")
        return [agents, centre_marker, *metric_texts, *obstacle_rectangles]

    animation = FuncAnimation(
        fig, animate, frames=range(args.steps), interval=int(cfg.dt * 1000),
        init_func=init_animation, cache_frame_data=False, blit=False,
        repeat=False,
    )
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(
        fps=args.fps,
        metadata={
            "title": (
                f"Cognitive swarm with {obstacle_count} stationary obstacles"
                if stationary_obstacle
                else "Cognitive swarm with random moving obstacles"
            ),
            "comment": (
                f"agents={agent_count}; steps={args.steps}; "
                f"relative_obstacle_speed={obstacle_speed:g}"
            ),
        },
        bitrate=900,
    )
    print(
        f"Exporting {output} ({agent_count} agents, {args.steps} steps, "
        f"relative obstacle speed {obstacle_speed:g}) ...",
        flush=True,
    )
    animation.save(str(output), writer=writer, dpi=args.dpi)
    plt.close(fig)
    print(
        f"Saved {output.resolve()} | minimum body clearances: "
        f"agent-agent={minimum_agent_clearance:.6f}, "
        f"agent-obstacle={minimum_obstacle_clearance:.6f}",
        flush=True,
    )


def main(argv=None):
    args = parse_args(argv)
    if not writers.is_available("ffmpeg"):
        raise RuntimeError("MP4 export requires FFmpeg")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for agent_count in args.agents:
        stationary_obstacle = (
            args.stationary_obstacle or args.static_obstacle_count is not None
        )
        static_obstacle_count = (
            args.static_obstacle_count
            if args.static_obstacle_count is not None
            else 2
        )
        speed_ratios = (0.0,) if stationary_obstacle else args.speed_ratios
        for speed_ratio in speed_ratios:
            obstacle_speed = args.obstacle_speed_min * speed_ratio
            if stationary_obstacle:
                if static_obstacle_count == 2:
                    prefix = "two_static_obstacles"
                else:
                    obstacle_word = (
                        "obstacle" if static_obstacle_count == 1 else "obstacles"
                    )
                    prefix = f"{static_obstacle_count}_static_{obstacle_word}"
                filename = (
                    f"{prefix}_agents_{agent_count}_speed_0_"
                    f"{args.steps}_steps.mp4"
                )
            else:
                filename = (
                    f"random_obstacles_agents_{agent_count}_speed_"
                    f"{obstacle_speed:g}_{ratio_label(speed_ratio)}_minimum_"
                    f"{args.steps}_steps.mp4"
                )
            export_case(
                args, agent_count, speed_ratio, args.output_dir / filename
            )


if __name__ == "__main__":
    main()
