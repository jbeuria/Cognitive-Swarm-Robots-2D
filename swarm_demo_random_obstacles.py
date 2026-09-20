"""Centroid-tracking two-obstacle experiments for the cognitive swarm.

The two obstacles are the same size as a flock agent but use contrasting
colours. Their moving targets are anchored to the instantaneous flock
centroid, so they remain with the flock and make repeated smooth intrusions.
The default batch exports N=50, 100, and 200 at 0.5x, 1x, and 1.5x the minimum
centroid-relative obstacle speed. A final geometric safety shield guarantees
that circumscribed agent bodies do not overlap. Every movie contains 1,000
simulation steps.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, writers
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch, Rectangle
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
    parser.add_argument("--fps", type=positive_int, default=25)
    parser.add_argument("--dpi", type=positive_int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    return args


def make_formation(rng, count):
    """Create a compact near-square flock with safe initial spacing."""
    columns = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / columns))
    spacing = 3.0
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
    # Keep the controller tuning identical to swarm_demo.py.  This experiment
    # changes the obstacle geometry and motion, not the controller itself.
    obstacle_speed = args.obstacle_speed_min * speed_ratio

    case_seed = args.seed + 1009 * agent_count + int(round(100 * speed_ratio))
    rng = np.random.default_rng(case_seed)
    pos = make_formation(rng, agent_count)
    theta = rng.normal(0.0, 0.035, agent_count)
    heading = np.column_stack([np.cos(theta), np.sin(theta)])
    speed = np.full(agent_count, cfg.v0)
    controller = CognitiveSwarm(agent_count, cfg)

    obstacle_count = 2
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
    visible_obstacle_clearance = 2.0
    obs_radius = np.full(
        obstacle_count, body_diameter + visible_obstacle_clearance
    )
    hard_agent_distance = body_diameter + 0.18
    hard_obstacle_distance = body_diameter + visible_obstacle_clearance

    formation_width = np.ptp(pos[:, 0])
    formation_height = np.ptp(pos[:, 1])
    half_width = max(43.0, 0.5 * formation_width + 22.0)
    half_height = max(28.0, 0.5 * formation_height + 19.0)
    orbit_x = min(0.62 * half_width, 0.5 * formation_width + 10.0)
    orbit_y = min(0.60 * half_height, 0.5 * formation_height + 8.0)
    centre0 = pos.mean(axis=0)
    phase = np.array([-0.8, 0.7])
    obs_pos[0] = centre0 + np.array([
        orbit_x * np.sin(phase[0]),
        0.55 * orbit_y * np.sin(2.0 * phase[0]),
    ])
    obs_pos[1] = centre0 + np.array([
        orbit_x * np.sin(phase[1]),
        0.55 * orbit_y * np.sin(2.0 * phase[1] + 0.5 * np.pi),
    ])
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
        obs_heading[index] = np.arctan2(initial_velocity[1], initial_velocity[0])
        obs_vel[index] = initial_velocity
    previous_centre = centre0.copy()

    fig, ax = plt.subplots(figsize=(10.4, 6.4))
    ax.set_aspect("equal")
    ax.set_facecolor("#f7f8f4")
    ax.grid(True, color="#d7ddd7", linewidth=0.55, alpha=0.75)
    ax.set_xlabel("global x")
    ax.set_ylabel("global y")
    ax.set_title(
        f"Centroid-driven obstacles | agents={agent_count} | relative speed "
        f"{obstacle_speed:g} ({speed_ratio:g}x minimum)"
    )

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

    status = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top", ha="left")
    ax.legend(handles=[
        Patch(facecolor=plt.cm.Blues(0.65), edgecolor="#17365d", label="swarm agent"),
        Patch(facecolor=obstacle_colours[0], edgecolor="black", label="obstacle 1"),
        Patch(facecolor=obstacle_colours[1], edgecolor="black", label="obstacle 2"),
    ], loc="lower right")

    step_number = 0
    filtered_centroid_velocity = np.array([cfg.v0, 0.0])
    minimum_agent_clearance = np.inf
    minimum_obstacle_clearance = np.inf

    def update_view_and_obstacles():
        centre = pos.mean(axis=0)
        ax.set_xlim(centre[0] - half_width, centre[0] + half_width)
        ax.set_ylim(centre[1] - half_height, centre[1] + half_height)
        for index, rectangle in enumerate(obstacle_rectangles):
            angle = np.degrees(obs_heading[index])
            rectangle.set_transform(
                Affine2D().rotate_deg(angle).translate(*obs_pos[index]) + ax.transData
            )

    def animate(_):
        nonlocal pos, heading, speed, step_number, previous_centre
        nonlocal filtered_centroid_velocity
        nonlocal minimum_agent_clearance, minimum_obstacle_clearance
        step_number += 1
        flock_centre = pos.mean(axis=0)
        centroid_velocity = (flock_centre - previous_centre) / cfg.dt
        previous_centre = flock_centre.copy()
        centroid_ease = 1.0 - np.exp(-cfg.dt / 0.8)
        filtered_centroid_velocity += centroid_ease * (
            centroid_velocity - filtered_centroid_velocity
        )

        old_obs_pos = obs_pos.copy()
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

        obstacle_clearance = np.min(
            np.linalg.norm(pos[:, None, :] - obs_pos[None, :, :], axis=2)
            - body_diameter
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
        status.set_text(
            f"step = {step_number}/{args.steps}   agents = {agent_count}\n"
            f"obstacle relative speed = {obstacle_speed:g} "
            f"({speed_ratio:g}x minimum)\n"
            f"agent speed mean/min/max = {speed.mean():.2f} / "
            f"{speed.min():.2f} / {speed.max():.2f}\n"
            f"body clearance now = {agent_clearance:.2f} agent / "
            f"{obstacle_clearance:.2f} obstacle\n"
            f"run minima = {minimum_agent_clearance:.2f} / "
            f"{minimum_obstacle_clearance:.2f}"
        )
        return [agents, centre_marker, status, *obstacle_rectangles]

    update_view_and_obstacles()

    def init_animation():
        """Draw the initial state without consuming a simulation step."""
        centre_marker.set_offsets(pos.mean(axis=0).reshape(1, 2))
        status.set_text(
            f"step = 0/{args.steps}   agents = {agent_count}\n"
            f"obstacle relative speed = {obstacle_speed:g} "
            f"({speed_ratio:g}x minimum)"
        )
        return [agents, centre_marker, status, *obstacle_rectangles]

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
            "title": "Cognitive swarm with random moving obstacles",
            "comment": (
                f"agents={agent_count}; steps={args.steps}; "
                f"relative_obstacle_speed={obstacle_speed:g}"
            ),
        },
        bitrate=2200,
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
        for speed_ratio in args.speed_ratios:
            obstacle_speed = args.obstacle_speed_min * speed_ratio
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
