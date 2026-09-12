"""
Single-file ROS 2 cognitive swarm controller for Gazebo / practical demos.

Inputs
------
/agent_<i>/odom      nav_msgs/Odometry
/swarm/obstacles     std_msgs/Float64MultiArray

Obstacle array rows:
    [x, y, vx, vy, radius]
flattened as one Float64MultiArray.

Outputs
-------
/agent_<i>/cmd_vel   geometry_msgs/TwistStamped
/swarm/diagnostics   std_msgs/Float64MultiArray

The controller implements the same logic as swarm_demo.py:
- directed internal Bloch-like pair states
- slow cognitive state and adaptive caution
- cognitively weighted alignment
- predictive inter-agent safe distance
- closest-approach + tangential moving-obstacle avoidance
- forward migration
- rejoining acceleration and manoeuvre braking

Gazebo/vehicle dynamics are NOT integrated here. Odometry supplies the
physical state; this node publishes velocity and yaw-rate references.
"""

from dataclasses import dataclass
import numpy as np

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float64MultiArray


@dataclass
class Config:
    dt: float = 0.04
    sensing_radius: float = 12.0
    safe_agent_distance: float = 2.4
    prediction_horizon_agents: float = 0.75
    obstacle_margin: float = 8.5
    prediction_horizon_obs: float = 1.0

    T1: float = 0.6
    T2: float = 0.3
    Gamma: float = 1.0
    kappa: float = 1.2
    g_align: float = 1.0
    gamma_s: float = 0.03
    lambda_fb: float = 2.0
    s_base: float = 0.0
    caution_gain: float = 1.0

    k_align: float = 1.0
    k_coh: float = 0.45
    k_sep: float = 3.4
    k_obs_radial: float = 5.2
    k_obs_tangent: float = 4.6
    k_goal: float = 1.65
    omega_max: float = 1.7

    v0: float = 3.0
    v_min: float = 0.9
    v_max: float = 4.6
    eta_rejoin: float = 0.55
    eta_brake: float = 1.0
    rejoin_length: float = 10.0
    tau_v: float = 0.42
    a_acc: float = 2.2
    a_brake: float = 4.5


class CognitiveSwarmROS2(Node):
    def __init__(self):
        super().__init__("cognitive_swarm_controller")

        self.declare_parameter("n_agents", 6)
        self.declare_parameter("control_dt", 0.04)
        self.declare_parameter("goal_x", 1.0)
        self.declare_parameter("goal_y", 0.0)

        self.N = int(self.get_parameter("n_agents").value)
        self.cfg = Config(dt=float(self.get_parameter("control_dt").value))
        self.goal = self.unit(np.array([
            float(self.get_parameter("goal_x").value),
            float(self.get_parameter("goal_y").value),
        ]))
        if np.linalg.norm(self.goal) < 1e-12:
            self.goal = np.array([1.0, 0.0])

        self.pos = np.zeros((self.N, 2), dtype=float)
        self.heading = np.tile(np.array([1.0, 0.0]), (self.N, 1))
        self.speed = np.zeros(self.N, dtype=float)
        self.odom_ok = np.zeros(self.N, dtype=bool)

        self.bloch = np.zeros((self.N, self.N, 3), dtype=float)
        self.slow = np.zeros(self.N, dtype=float)

        self.obs_pos = np.empty((0, 2), dtype=float)
        self.obs_vel = np.empty((0, 2), dtype=float)
        self.obs_radius = np.empty(0, dtype=float)

        self.cmd_pubs = []
        for i in range(self.N):
            self.create_subscription(
                Odometry, f"/agent_{i}/odom",
                lambda msg, idx=i: self.odom_cb(msg, idx), 10
            )
            self.cmd_pubs.append(self.create_publisher(
                TwistStamped, f"/agent_{i}/cmd_vel", 10
            ))

        self.create_subscription(
            Float64MultiArray, "/swarm/obstacles", self.obstacle_cb, 10
        )
        self.diag_pub = self.create_publisher(
            Float64MultiArray, "/swarm/diagnostics", 10
        )
        self.timer = self.create_timer(self.cfg.dt, self.control_cycle)

        self.get_logger().info(
            f"Cognitive swarm controller started for {self.N} agents"
        )

    @staticmethod
    def unit(v):
        v = np.asarray(v, dtype=float)
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        return np.divide(v, n, out=np.zeros_like(v, dtype=float), where=n > 1e-12)

    @staticmethod
    def wrap(a):
        return (a + np.pi) % (2*np.pi) - np.pi

    @staticmethod
    def yaw_from_quaternion(q):
        siny = 2.0 * (q.w*q.z + q.x*q.y)
        cosy = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        return np.arctan2(siny, cosy)

    def odom_cb(self, msg, i):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = self.yaw_from_quaternion(q)
        self.pos[i] = [p.x, p.y]
        self.heading[i] = [np.cos(yaw), np.sin(yaw)]
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.speed[i] = np.hypot(vx, vy)
        self.odom_ok[i] = True

    def obstacle_cb(self, msg):
        a = np.asarray(msg.data, dtype=float)
        if a.size == 0:
            self.obs_pos = np.empty((0, 2))
            self.obs_vel = np.empty((0, 2))
            self.obs_radius = np.empty(0)
            return
        if a.size % 5 != 0:
            self.get_logger().warning(
                "/swarm/obstacles must contain flattened [x,y,vx,vy,radius] rows"
            )
            return
        rows = a.reshape(-1, 5)
        self.obs_pos = rows[:, :2].copy()
        self.obs_vel = rows[:, 2:4].copy()
        self.obs_radius = np.maximum(rows[:, 4], 0.0)

    def update_internal(self):
        c = self.cfg
        delta = self.pos[:, None, :] - self.pos[None, :, :]
        dist = np.linalg.norm(delta, axis=2)
        valid = self.odom_ok[:, None] & self.odom_ok[None, :]
        A = (dist > 0.0) & (dist < c.sensing_radius) & valid

        dot_h = self.heading @ self.heading.T
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
        M = np.divide((mz*A).sum(axis=1), count,
                      out=np.zeros(self.N), where=count > 0)
        s_star = np.tanh(c.lambda_fb * M)
        self.slow += c.dt * (
            -c.gamma_s * (self.slow - c.s_base) + c.gamma_s * s_star
        )
        q = 1.0 + c.caution_gain / (1.0 + np.exp(-self.slow))
        w = 0.5 * (1.0 + mz) * A
        return A, dist, w, M, q

    def steering(self, A, dist, w, q):
        c = self.cfg
        D = np.zeros((self.N, 2), dtype=float)
        velocity = self.speed[:, None] * self.heading
        active = np.where(self.odom_ok)[0]
        flock_centre = self.pos[active].mean(axis=0) if len(active) else np.zeros(2)

        for i in range(self.N):
            if not self.odom_ok[i]:
                D[i] = self.heading[i]
                continue

            nei = np.where(A[i])[0]

            wsum = w[i].sum()
            align = (c.k_align * (w[i, :, None] * self.heading).sum(axis=0) / wsum
                     if wsum > 1e-12 else c.k_align * self.heading[i])

            coh = (c.k_coh * (self.pos[nei].mean(axis=0) - self.pos[i])
                   if len(nei) else np.zeros(2))

            # Predictive inter-agent safety.
            sep = np.zeros(2)
            for j in nei:
                r = self.pos[j] - self.pos[i]
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

            # Moving-obstacle closest-approach + tangential pass.
            obs = np.zeros(2)
            vi = velocity[i]
            for centre, u, rad in zip(self.obs_pos, self.obs_vel, self.obs_radius):
                r = centre - self.pos[i]
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
                    radial = -n
                    pass_sign = -1.0 if centre[1] >= flock_centre[1] else 1.0
                    tangent = pass_sign * np.array([-n[1], n[0]])
                    closing = max(
                        0.0,
                        -np.dot(r, v_rel) / (np.linalg.norm(r) + 1e-12)
                    )
                    gain = (1.0 + 0.18 * closing) * smooth
                    obs += gain * (
                        c.k_obs_radial * radial + c.k_obs_tangent * tangent
                    )
            obs *= q[i]

            D[i] = align + coh + sep + obs + c.k_goal * self.goal
            if np.linalg.norm(D[i]) < 1e-12:
                D[i] = self.heading[i]

        return self.unit(D), flock_centre

    def make_commands(self, d_hat, flock_centre):
        c = self.cfg
        to_centre = flock_centre[None, :] - self.pos
        dcentre = np.linalg.norm(to_centre, axis=1)
        centre_hat = self.unit(to_centre)
        toward = np.maximum(np.sum(self.heading * centre_hat, axis=1), 0.0)
        rho = np.tanh(dcentre / c.rejoin_length) * toward

        mu = np.clip(1.0 - np.sum(self.heading*d_hat, axis=1), 0.0, 2.0)
        v_star = c.v0 * (
            1.0 + c.eta_rejoin*rho - 0.5*c.eta_brake*mu
        )
        v_star = np.clip(v_star, c.v_min, c.v_max)
        accel = np.clip(
            (v_star - self.speed) / c.tau_v,
            -c.a_brake, c.a_acc
        )

        yaw = np.arctan2(self.heading[:, 1], self.heading[:, 0])
        yaw_des = np.arctan2(d_hat[:, 1], d_hat[:, 0])
        yaw_rate = np.clip(
            self.wrap(yaw_des - yaw) / c.dt,
            -c.omega_max, c.omega_max
        )

        speed_cmd = np.clip(
            self.speed + accel*c.dt, c.v_min, c.v_max
        )
        vel_world = speed_cmd[:, None] * d_hat
        return vel_world, v_star, accel, yaw_rate, mu, rho

    def publish_commands(self, vel_world, yaw_rate):
        """Adapt only this method if your bridge requires another message/frame."""
        stamp = self.get_clock().now().to_msg()
        for i in range(self.N):
            if not self.odom_ok[i]:
                continue
            msg = TwistStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = "world"
            msg.twist.linear.x = float(vel_world[i, 0])
            msg.twist.linear.y = float(vel_world[i, 1])
            msg.twist.linear.z = 0.0
            msg.twist.angular.z = float(yaw_rate[i])
            self.cmd_pubs[i].publish(msg)

    def publish_diagnostics(self, v_star, accel, yaw_rate, mu, rho, M, q, A):
        rows = np.column_stack([
            v_star, accel, yaw_rate, mu, rho,
            self.slow, q, M, A.sum(axis=1).astype(float)
        ])
        msg = Float64MultiArray()
        msg.data = rows.ravel().tolist()
        self.diag_pub.publish(msg)

    def control_cycle(self):
        if not np.any(self.odom_ok):
            return
        A, dist, w, M, q = self.update_internal()
        d_hat, flock_centre = self.steering(A, dist, w, q)
        vel_world, v_star, accel, yaw_rate, mu, rho = \
            self.make_commands(d_hat, flock_centre)
        self.publish_commands(vel_world, yaw_rate)
        self.publish_diagnostics(v_star, accel, yaw_rate, mu, rho, M, q, A)


def main(args=None):
    rclpy.init(args=args)
    node = CognitiveSwarmROS2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
