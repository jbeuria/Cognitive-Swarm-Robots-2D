# Purpose and relation to the fixed-speed model

The base cognitive-swarm model uses a physical state $(\bm{x}_i,\bm{e}_i)$ with $\|\bm{e}_i\|=1$ and fixed translational speed, $$\dot{\bm{x}}_i=v_0\bm{e}_i.$$ The internal controller consists of directed Bloch-type perceptual channels, a slow regulatory state, cognitively weighted alignment, adaptive caution, and bounded turning. The present extension retains that architecture and adds a scalar physical speed $v_i(t)$. Acceleration is used selectively for two functions: (i) positive acceleration to rejoin a spatially split flock, and (ii) braking during sudden or strong avoidance manoeuvres. Ordinary coherent motion remains close to $v_0$.

The complete agent state is therefore $$\boxed{\mathcal X_i=\{\bm{x}_i,\bm{e}_i,v_i,s_i,\{\bm m_{ij}\}_{j\in\mathcal N_i}\}.}$$

# Neighbour graph

For sensing radius $R$, $$A_{ij}(t)=\begin{cases}1,&0<\|\bm{x}_i-\bm{x}_j\|<R,\\0,&\text{otherwise},\end{cases}
\qquad
\mathcal N_i(t)=\{j:A_{ij}=1\}.$$ The graph is recomputed at every controller step.

# Fast Bloch-type perceptual register

Each directed visible channel $(i,j)$ carries $$\bm m_{ij}=(m_x^{ij},m_y^{ij},m_z^{ij}).$$ The longitudinal effective field is $$\boxed{h_z^{ij}=\kappa s_i+g_a\,\bm{e}_i\!\cdot\!\bm{e}_j+h_{\rm ext}^{ij}.}$$ The fast state evolves as $$\begin{aligned}
\dot m_x^{ij}&=-2h_z^{ij}m_y^{ij}-\frac{m_x^{ij}}{T_2},\\
\dot m_y^{ij}&=2\left(h_z^{ij}m_x^{ij}-\Gamma m_z^{ij}\right)-\frac{m_y^{ij}}{T_2},\\
\dot m_z^{ij}&=2\Gamma m_y^{ij}-\frac{m_z^{ij}-m_{z,\mathrm{eq}}^{ij}}{T_1},
\end{aligned}$$ with $$m_{z,\mathrm{eq}}^{ij}=\tanh h_z^{ij}.$$ The resolved alignment weight is $$\boxed{w_{ij}=A_{ij}\frac{1+m_z^{ij}}{2},\qquad 0\le w_{ij}\le1.}$$ Inactive channels make no alignment contribution. In the reference implementation their stored state is retained until the channel becomes visible again; resetting or passive relaxation can be substituted if desired for a particular sensor-memory interpretation.

# Slow regulatory state and cognitive caution

The mean resolved alignment is $$M_i=\begin{cases}\displaystyle \frac{1}{|\mathcal N_i|}\sum_{j\in\mathcal N_i}m_z^{ij},&|\mathcal N_i|>0,\\0,&|\mathcal N_i|=0.\end{cases}$$ The slow target and slow-state dynamics are $$\begin{aligned}
s_i^*&=\tanh(\lambda_{\rm fb}M_i),\\
\dot s_i&=-\gamma_s(s_i-s_{\rm base})+\gamma_s s_i^*.
\end{aligned}$$ The state-dependent caution factor is $$\boxed{q_i(s_i)=1+c_q\sigma(s_i),\qquad \sigma(s)=\frac{1}{1+e^{-s}}.}$$ Thus the closed loop is $$\bm m_{ij}\rightarrow M_i\rightarrow s_i\rightarrow h_z^{ij},q_i\rightarrow \bm m_{ij},$$ with $s_i$ affecting both future perceptual resolution and avoidance gain.

# Physical steering field

The cognitively weighted alignment vector is $$\bm{D}_i^{\rm align}
=
 k_{\rm align}
\frac{\sum_{j\in\mathcal N_i}w_{ij}\bm{e}_j}
{\sum_{j\in\mathcal N_i}w_{ij}+\varepsilon}.$$

Local cohesion is $$\bar{\bm{x}}_i=\frac{1}{|\mathcal N_i|}\sum_{j\in\mathcal N_i}\bm{x}_j,
\qquad
\bm A_i^{\rm coh}=k_{\rm coh}(\bar{\bm{x}}_i-\bm{x}_i),$$ with $\bar{\bm{x}}_i=\bm{x}_i$ for an empty neighbourhood.

## Predictive inter-agent safety

A finite geometric safety distance $d_{\rm safe}$ is enforced between flock agents. For neighbour $j$, define $$\bm r_{ij}=\bm{x}_j-\bm{x}_i,
\qquad
\bm v_{ij}^{\rm rel}=v_j\bm{e}_j-v_i\bm{e}_i.$$ The time of closest approach over a short horizon $T_a$ is $$\tau_{ij}^{*}
=
\operatorname{clip}_{[0,T_a]}
\left(
-\frac{\bm r_{ij}\cdot\bm v_{ij}^{\rm rel}}
{\|\bm v_{ij}^{\rm rel}\|^2+\varepsilon}
\right),$$ and $$\bm r_{ij}^{*}=\bm r_{ij}+\tau_{ij}^{*}\bm v_{ij}^{\rm rel}.$$ The controller uses the smaller of $\|\bm r_{ij}\|$ and $\|\bm r_{ij}^{*}\|$ as the effective separation $d_{ij}$. If $d_{ij}<d_{\rm safe}$, a smooth repulsive term is applied, $$\boxed{
\bm A_i^{\rm sep}
=
q_i\sum_{j\in\mathcal N_i}
 k_{\rm sep}
 S\!\left(\frac{d_{\rm safe}-d_{ij}}{d_{\rm safe}}\right)
 \hat{\bm n}_{ij},
}$$ where $\hat{\bm n}_{ij}$ points away from the more threatening current or predicted relative position and $S(z)=3z^2-2z^3$ on $[0,1]$. Thus agents preserve a safety gap not only when already close, but also when their relative motion predicts an imminent close pass.

## Predictive moving-obstacle navigation

For moving obstacle $a$ with centre $\bm{o}_a$, velocity $\bm{u}_a$, and conservative collision radius $\rho_a$, let $$\bm r_{ia}=\bm{o}_a-\bm{x}_i,
\qquad
\bm v_{ia}^{\rm rel}=\bm{u}_a-v_i\bm{e}_i.$$ The predicted time of closest approach over horizon $T_p$ is $$\boxed{
\tau_{ia}^{*}
=
\operatorname{clip}_{[0,T_p]}
\left(
-\frac{\bm r_{ia}\cdot\bm v_{ia}^{\rm rel}}
{\|\bm v_{ia}^{\rm rel}\|^2+\varepsilon}
\right).
}$$ Then $$\bm r_{ia}^{*}=\bm r_{ia}+\tau_{ia}^{*}\bm v_{ia}^{\rm rel},
\qquad
d_{ia}^{*}=\|\bm r_{ia}^{*}\|-\rho_a.$$ The more threatening of the current and predicted clearances is denoted $d_{ia}$, with outward unit direction $\hat{\bm n}_{ia}$. To avoid the inefficient behaviour of pure radial repulsion, the obstacle response contains both a clearance-producing radial component and a tangential passing component, $$\boxed{
\bm A_i^{\rm obs}
=
q_i\sum_a
S\!\left(
\frac{\ell_{\rm obs}-d_{ia}}{\ell_{\rm obs}}
\right)
(1+c_v\nu_{ia})
\left[
 k_r\hat{\bm n}_{ia}+k_t\hat{\bm t}_{ia}
\right].
}$$ Here $\nu_{ia}$ is the positive closing-speed contribution. The tangent $\hat{\bm t}_{ia}$ is chosen consistently for the flock: agents pass below an obstacle lying above the instantaneous flock centre and above one lying below it. This group-consistent side choice reduces unnecessary splitting and oscillation around an oncoming vehicle.

The demonstration represents the flock as small car-like agents moving in the fixed task direction $\hat{\bm m}=(1,0)$; there is no corridor, lane potential, periodic boundary condition, or coordinate recycling. Agent coordinates evolve continuously in the global plane. Only the plotting camera translates with the flock for visualization. The forward task contribution is simply $$\bm A_i^{\rm goal}=k_{\rm goal}\hat{\bm m}.$$ The complete steering vector is therefore $$\boxed{
\bm{D}_i
=
\bm{D}_i^{\rm align}
+\bm A_i^{\rm coh}
+\bm A_i^{\rm sep}
+\bm A_i^{\rm obs}
+\bm A_i^{\rm goal}.
}$$ The target direction is $$\hat{\bm d}_i=\frac{\bm{D}_i}{\|\bm{D}_i\|},$$ with $\hat{\bm d}_i=\bm{e}_i$ if the norm is numerically zero.

# Bounded heading actuation

In 2-D let $\theta_i=\operatorname{atan2}(e_{iy},e_{ix})$ and $\theta_i^*=\operatorname{atan2}(\hat d_{iy},\hat d_{ix})$. Then $$\Delta\theta_i=\operatorname{wrap}(\theta_i^*-\theta_i),$$ and the commanded yaw rate is $$\boxed{\dot\psi_i^{\rm cmd}=\operatorname{clip}\left(\frac{\Delta\theta_i}{\Delta t},-\omega_{\max},\omega_{\max}\right).}$$ For a point-agent simulation, $$\theta_i^{n+1}=\theta_i^n+\dot\psi_i^{\rm cmd}\Delta t,
\qquad
\bm{e}_i^{n+1}=(\cos\theta_i^{n+1},\sin\theta_i^{n+1}).$$ In Gazebo this angle integration is performed by the vehicle/autopilot, not by the cognitive controller.

# Selective variable-speed extension

## Braking demand

Use the directional disagreement already produced by the steering field, $$\boxed{\mu_i=1-\bm{e}_i\cdot\hat{\bm d}_i,\qquad 0\le\mu_i\le2.}$$ A sudden obstacle rotates $\hat{\bm d}_i$ away from the current heading; therefore $\mu_i$ rises automatically and can trigger braking without a separate danger state.

## Rejoining demand

Let $\bm{x}_i^r$ denote a rejoining reference available to agent $i$. In a centralized simulation it may be the global swarm centroid $$\bm{x}_c=\frac1N\sum_j\bm{x}_j.$$ For real robots, $\bm{x}_i^r$ should be supplied by a swarm-state aggregator, a distributed/multi-hop centroid estimate, or a last-known group reference. This dependency is explicit because two physically disconnected components cannot infer each other’s centroids from purely local sensing alone.

Define $$\hat{\bm c}_i=\frac{\bm{x}_i^r-\bm{x}_i}{\|\bm{x}_i^r-\bm{x}_i\|+\varepsilon}$$ and $$\boxed{\rho_i=\tanh\!\left(\frac{\|\bm{x}_i^r-\bm{x}_i\|}{L_r}\right)\,[\bm{e}_i\cdot\hat{\bm c}_i]_+.}$$ Hence a separated agent accelerates only after it has turned approximately toward the rejoining reference. This prevents a speed boost in the wrong direction.

## Desired speed and acceleration

The desired speed is $$\boxed{v_i^*=\operatorname{clip}_{[v_{\min},v_{\max}]}\left\{v_0\left[1+\eta_r\rho_i-\frac{\eta_b}{2}\mu_i\right]\right\}.}$$ The physical acceleration command is $$\boxed{a_i^{\rm cmd}=\operatorname{sat}_{[-a_{\rm br},a_{\rm acc}]}\left(\frac{v_i^*-v_i}{\tau_v}\right).}$$ The interpretation is intentionally asymmetric: $$\begin{aligned}
\rho_i>0,\ \mu_i\approx0 &\Rightarrow \text{catch-up acceleration},\\
\mu_i>0 &\Rightarrow \text{evasive braking},\\
\rho_i\approx0,\ \mu_i\approx0 &\Rightarrow v_i^*\approx v_0.
\end{aligned}$$

For the point-agent verifier, $$\begin{aligned}
v_i^{n+1}&=\operatorname{clip}_{[v_{\min},v_{\max}]}\left(v_i^n+a_i^{\rm cmd}\Delta t\right),\\
\bm{x}_i^{n+1}&=\bm{x}_i^n+v_i^{n+1}\bm{e}_i^{n+1}\Delta t.
\end{aligned}$$

# Exact fixed-speed limit

Set $$\boxed{\eta_r=\eta_b=0}$$ and initialize $v_i(0)=v_0$. Then $v_i^*=v_0$, $a_i^{\rm cmd}=0$, and $$v_i(t)=v_0,\qquad \dot{\bm{x}}_i=v_0\bm{e}_i.$$ Thus the previous constant-speed model is recovered exactly while all internal cognitive dynamics and the heading update remain unchanged.

# Complete update order

At each controller cycle $n\to n+1$:

1.  Read physical states $(\bm{x}_i,\bm{e}_i,v_i)$, neighbour states, and obstacle states $(\bm{o}_a,\bm{u}_a,\rho_a)$.

2.  Recompute $A_{ij}$ and $\mathcal N_i$.

3.  Compute $h_z^{ij}$ for all active directed channels.

4.  Integrate $(m_x^{ij},m_y^{ij},m_z^{ij})$ and obtain $w_{ij}$.

5.  Compute $M_i$, $s_i^*$, update $s_i$, and obtain $q_i$.

6.  Build weighted alignment, cohesion, separation, obstacle and optional task steering; form $\bm{D}_i$ and $\hat{\bm d}_i$.

7.  Compute yaw-rate command from the bounded heading error.

8.  Compute rejoin demand $\rho_i$ and manoeuvre demand $\mu_i$.

9.  Compute $v_i^*$ and rate-limited $a_i^{\rm cmd}$.

10. Export the physical commands to the simulator/robot interface.

The information flow is $$\boxed{\text{sensors/peers}\to \bm m_{ij}\to w_{ij}\to M_i\to s_i\to q_i\to \hat{\bm d}_i\to(\dot\psi_i^{\rm cmd},v_i^*,a_i^{\rm cmd}).}$$

# ROS 2/Gazebo-ready interface

The implementation is deliberately reduced to two self-contained Python files. The ROS 2 file does not integrate an artificial point-mass plant: it reads the physical state from odometry, updates all internal cognitive variables, computes the swarm control law, and publishes only physical command references.

For agent $i$, the ROS 2 node subscribes to $$\texttt{/agent\_i/odom}\qquad (\texttt{nav\_msgs/Odometry}),$$ from which it obtains $\bm{x}_i$, heading $\bm{e}_i$, and measured speed $v_i$. Moving obstacles are supplied on $$\texttt{/swarm/obstacles}\qquad (\texttt{std\_msgs/Float64MultiArray})$$ as flattened rows $$,$$ so that measured obstacle velocity can be used in the short-horizon prediction $\bm o_a^p=\bm o_a+T_p\bm u_a$.

At each control cycle the node executes the complete chain $$\boxed{
\text{odometry/obstacles}
\to \bm m_{ij}
\to w_{ij}
\to M_i
\to s_i
\to q_i
\to \hat{\bm d}_i
\to (v_i^*,a_i^{\rm cmd},\dot\psi_i^{\rm cmd})
\to \bm v_i^{\rm cmd}.
}$$ The world-frame planar velocity reference is $$\bm v_i^{\rm cmd}=v_i^{\rm cmd}\hat{\bm d}_i,$$ with $v_i^{\rm cmd}$ obtained by one acceleration-limited step toward $v_i^*$. The bounded yaw-rate reference is computed from the desired/current heading error.

The node publishes

<div class="center">

| ROS 2 output         | Content                                                                                            |
|:---------------------|:---------------------------------------------------------------------------------------------------|
| `/agent_i/cmd_vel`   | `geometry_msgs/TwistStamped`; world-frame $v_x^{\rm cmd},v_y^{\rm cmd}$ and $\dot\psi_i^{\rm cmd}$ |
| `/swarm/diagnostics` | flattened per-agent rows $[v_i^*,a_i^{\rm cmd},\dot\psi_i^{\rm cmd},\mu_i,\rho_i,s_i,q_i,M_i,n_i]$ |

</div>

Thus the variables directly available for Gazebo/robot experiments are $$\boxed{
\bm v_i^{\rm cmd},\;\dot\psi_i^{\rm cmd},\;v_i^*,\;a_i^{\rm cmd},\;
\mu_i,\;\rho_i,\;s_i,\;q_i,\;M_i,\;n_i .
}$$ The present `publish_commands()` function uses `TwistStamped`. If PX4, ArduPilot, or a particular Gazebo bridge expects body-frame velocity or a different message type, only this output function needs adaptation; the cognitive and swarm equations remain unchanged. Vehicle mass, inertia, thrust, attitude response, and motor dynamics remain the responsibility of Gazebo/autopilot.

# Two-file software implementation

Only two Python files are supplied:

<div class="center">

| File                 | Responsibility                                                                                                                                                                                                                                                                                                                                                                       |
|:---------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `swarm_demo.py`      | self-contained real-time 2-D validation: full internal cognitive dynamics, steering, rejoining acceleration, evasive braking, bounded physical update, a compact flock, and a continuing asynchronous stream of moving obstacles with randomized arrival times, speeds, lateral offsets, and crossing angles; the obstacles are visually distinct from the agents and no corridor is imposed |
| `ros2_swarm_node.py` | self-contained ROS 2/Gazebo controller: odometry and obstacle input, persistent $\bm m_{ij}$ and $s_i$, full cognitive-to-physical update, `cmd_vel` publication, and diagnostic publication for every controlled agent                                                                                                                                                              |

</div>

No auxiliary local Python modules are required. Model parameters are held in an internal `Config` dataclass inside each self-contained file. This duplication is intentional: the real-time verifier can run directly with NumPy/Matplotlib, while the ROS 2 file can be copied into a ROS package without dependency on a custom Python module tree.

## Running the real-time verifier

Run the standalone verifier from a terminal with

    python swarm_demo.py

The program keeps an explicit live reference to the Matplotlib `FuncAnimation` object until the plotting window is closed. This is necessary because an anonymous animation object may otherwise be garbage collected before its first frame is rendered, producing Matplotlib’s “Animation was deleted without rendering anything” warning. The call to `plt.show(block=True)` keeps the GUI event loop active for the duration of the demonstration. A normal interactive Matplotlib backend (for example TkAgg, QtAgg, or the macOS backend) is therefore required for the live plot.

## Unbounded longitudinal evolution and split–reformation diagnostic

No periodic boundary condition is used for the flock. The physical update is always $$\bm{x}_i^{n+1}=\bm{x}_i^n+v_i^{n+1}\bm{e}_i^{n+1}\Delta t,$$ and the resulting global coordinates are retained for the full simulation. For visualization only, the plotting camera has a separate state. It looks ahead of the mean flock heading, moves only when the flock crosses a dead zone, and approaches its target with a finite response time. The flock therefore travels visibly within the canvas instead of being fixed to its centre, while its continuously advancing global coordinates remain unchanged.

Each demonstration obstacle is assigned a speed $u_a$ sampled independently for its current encounter, $$\bm{u}_a=u_a(\cos\phi_a,\sin\phi_a),
\qquad
\dot\phi_a=\omega_a(t),$$ where the turn-rate target is piecewise-random and the applied turn rate approaches that target smoothly. The relative heading remains bounded around the oncoming direction, $$|\operatorname{wrap}(\phi_a-\pi)|\leq \phi_{\max}.$$ After a vehicle passes a fixed distance behind the flock it becomes inactive, waits for a random interval, and is spawned far ahead of the flock's current global position with a new speed, lateral offset, and heading. This is asynchronous obstacle generation, not a periodic boundary condition on the swarm.

To show the desired split–reformation behaviour explicitly, the demo reports the number of connected flock components under the sensing graph $$A_{ij}=\mathbf 1\!\left(\|\bm{x}_i-\bm{x}_j\|<R\right).$$ A close obstacle can temporarily increase the number of connected components; successful rejoining is indicated when the count returns to one, accompanied by a reduction of the flock span and rejoining demand $\rho_i$.

# Reference defaults and randomized road-like scenario

The internal cognitive parameters remain close to the representative scales used in the present model ($T_1=0.6$, $T_2=0.3$, $\Gamma=1$, $\kappa=1.2$, $\gamma_s=0.03$, $\lambda_{\rm fb}=2$). The supplied demonstration is intentionally an unbounded road-like encounter rather than a corridor-like arena: a compact flock of small agents advances indefinitely in global $+x$. A reusable pool generates larger unruly vehicles asynchronously ahead of the current flock. Each travels generally in the opposite direction and executes a bounded random heading weave. Their lateral spawn positions are biased toward the flock so that close encounters can split the group; inactive slots wait for a fresh random delay after the corresponding vehicle has passed well behind the moving flock.

The controller-level defaults are

<div class="center">

| Parameter        | Value              | Parameter                | Value                       |
|:-----------------|:-------------------|:-------------------------|:----------------------------|
| $\Delta t$       | $0.04$             | $R$                      | $12$                        |
| $d_{\rm safe}$   | $2.4$              | $T_a$                    | $0.75\,\mathrm{s}$          |
| $\ell_{\rm obs}$ | $8.5$              | $T_p$                    | $1.0\,\mathrm{s}$           |
| $k_{\rm sep}$    | $3.4$              | $k_r,k_t$                | $5.2,\,4.6$                 |
| $k_{\rm goal}$   | $1.65$             | $\omega_{\max}$          | $1.7\,\mathrm{rad\,s^{-1}}$ |
| $v_0$            | $3.0$              | $v_{\min},v_{\max}$      | $0.9,\,4.6$                 |
| $\eta_r,\eta_b$  | $0.55,\,1.0$       | $L_r$                    | $10$                        |
| $\tau_v$         | $0.42\,\mathrm{s}$ | $a_{\rm acc},a_{\rm br}$ | $2.2,\,4.5$                 |

</div>

The demonstration obstacles use a larger car-like footprint (approximately $4.2\times2.0$ simulation units), an independently sampled speed in $[2.4,6.2]$ simulation units per second for each encounter, and a conservative circular control envelope $\rho_a=2.25$. Their heading is allowed to wander within approximately $\pm0.55$ rad of the oncoming direction. Flock agents are intentionally drawn much smaller (approximately $0.95\times0.50$ units visually), rotate with their physical headings, and are coloured by speed so that braking and catch-up acceleration are visible.

These are controller-level working defaults, not universal optima. The tuning emphasizes four objectives simultaneously: (i) no obstacle contact, (ii) preservation of a finite inter-agent gap, (iii) early smooth lateral navigation rather than last-second radial escape, and (iv) rapid recovery of flock compactness after the encounter. The tangential obstacle component is deliberately comparable to the radial component because a purely radial field tends to brake and scatter the flock around a large oncoming vehicle. Final Gazebo values must be retuned against actual vehicle dimensions, steering limits, sensing delay, and the precise frame conventions used by the simulator or autopilot.

# Practical recommendation for Gazebo transition

First validate three separate cases: (i) $\eta_r=\eta_b=0$ to reproduce the constant-speed baseline, (ii) $\eta_r>0,\eta_b=0$ to isolate rejoining acceleration, and (iii) $\eta_r>0,\eta_b>0$ with moving obstacles. Only after the controller-level behavior is stable should the point-agent plant be replaced with a multirotor model. This preserves a clean distinction between cognitive/swarm dynamics and vehicle dynamics.
