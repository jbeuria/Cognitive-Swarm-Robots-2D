# Cognitive Swarm Robots 2-D

A self-contained cognitive-swarm controller with a real-time 2-D verifier and a ROS 2/Gazebo-ready control node. The verifier models predictive agent separation, moving-obstacle avoidance, bounded turning, evasive braking, and acceleration to rejoin the flock.

## Quick start

Python 3.9 or newer is recommended. A clean virtual environment avoids conflicts with system Python packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the open-ended live animation (close the window to stop it):

```bash
python swarm_demo.py
```

Export a 30-second MP4 using the default filename `cognitive_swarm_demo.mp4`:

```bash
python swarm_demo.py --save
```

Export a customized movie:

```bash
python swarm_demo.py --save results/swarm.mp4 --duration 60 --fps 25 --dpi 160
```

MP4 export requires [FFmpeg](https://ffmpeg.org/). Install it with `brew install ffmpeg` on macOS or `sudo apt install ffmpeg` on Ubuntu/Debian. The exporter creates the output directory when necessary and prints the completed absolute path.

All command-line controls have defaults:

| Option | Default | Meaning |
|:--|:--|:--|
| `--save [FILE.mp4]` | disabled; filename is `cognitive_swarm_demo.mp4` when flagged without a value | Export MP4 instead of opening the live window |
| `--duration SECONDS` | `30` | Simulated duration of the exported movie |
| `--fps FPS` | `25` | Movie playback frame rate; 25 matches the model time step |
| `--dpi DPI` | `140` | Export resolution |
| `--agents N` | `36` | Number of flock agents |
| `--obstacles N` | `5` | Number of reusable random-obstacle slots |
| `--seed N` | `11` | Random seed for reproducible traffic |
| `-h`, `--help` | — | Show the complete command-line help |

## Purpose and relation to the fixed-speed model

The base cognitive-swarm model uses a physical state $(\mathbf{x}_i,\mathbf{e}_i)$ with $\|\mathbf{e}_i\|=1$ and fixed translational speed,
```math
\dot{\mathbf{x}}_i=v_0\mathbf{e}_i.
```
The internal controller consists of directed Bloch-type perceptual channels, a slow regulatory state, cognitively weighted alignment, adaptive caution, and bounded turning. The present extension retains that architecture and adds a scalar physical speed $v_i(t)$. Acceleration is used selectively for two functions: (i) positive acceleration to rejoin a spatially split flock, and (ii) braking during sudden or strong avoidance manoeuvres. Ordinary coherent motion remains close to $v_0$.

The complete agent state is therefore
```math
\boxed{\mathcal X_i=\{\mathbf{x}_i,\mathbf{e}_i,v_i,s_i,\{\mathbf m_{ij}\}_{j\in\mathcal N_i}\}.}
```

## Neighbour graph

For sensing radius $R$,
```math
A_{ij}(t)=\begin{cases}1,&0<\|\mathbf{x}_i-\mathbf{x}_j\|<R,\\0,&\text{otherwise},\end{cases}
\qquad
\mathcal N_i(t)=\{j:A_{ij}=1\}.
```
The graph is recomputed at every controller step.

## Fast Bloch-type perceptual register

Each directed visible channel $(i,j)$ carries
```math
\mathbf m_{ij}=(m_x^{ij},m_y^{ij},m_z^{ij}).
```
The longitudinal effective field is
```math
\boxed{h_z^{ij}=\kappa s_i+g_a\,\mathbf{e}_i\!\cdot\!\mathbf{e}_j+h_{\mathrm{ext}}^{ij}.}
```
The fast state evolves as
```math
\begin{aligned}
\dot m_x^{ij}&=-2h_z^{ij}m_y^{ij}-\frac{m_x^{ij}}{T_2},\\
\dot m_y^{ij}&=2\left(h_z^{ij}m_x^{ij}-\Gamma m_z^{ij}\right)-\frac{m_y^{ij}}{T_2},\\
\dot m_z^{ij}&=2\Gamma m_y^{ij}-\frac{m_z^{ij}-m_{z,\mathrm{eq}}^{ij}}{T_1},
\end{aligned}
```
with
```math
m_{z,\mathrm{eq}}^{ij}=\tanh h_z^{ij}.
```
The resolved alignment weight is
```math
\boxed{w_{ij}=A_{ij}\frac{1+m_z^{ij}}{2},\qquad 0\le w_{ij}\le1.}
```
Inactive channels make no alignment contribution. In the reference implementation their stored state is retained until the channel becomes visible again; resetting or passive relaxation can be substituted if desired for a particular sensor-memory interpretation.

## Slow regulatory state and cognitive caution

The mean resolved alignment is
```math
M_i=\begin{cases}\displaystyle \frac{1}{|\mathcal N_i|}\sum_{j\in\mathcal N_i}m_z^{ij},&|\mathcal N_i|>0,\\0,&|\mathcal N_i|=0.\end{cases}
```
The slow target and slow-state dynamics are
```math
\begin{aligned}
s_i^*&=\tanh(\lambda_{\mathrm{fb}}M_i),\\
\dot s_i&=-\gamma_s(s_i-s_{\mathrm{base}})+\gamma_s s_i^*.
\end{aligned}
```
The state-dependent caution factor is
```math
\boxed{q_i(s_i)=1+c_q\sigma(s_i),\qquad \sigma(s)=\frac{1}{1+e^{-s}}.}
```
Thus the closed loop is
```math
\mathbf m_{ij}\rightarrow M_i\rightarrow s_i\rightarrow h_z^{ij},q_i\rightarrow \mathbf m_{ij},
```
with $s_i$ affecting both future perceptual resolution and avoidance gain.

## Physical steering field

The cognitively weighted alignment vector is
```math
\mathbf{D}_i^{\mathrm{align}}
=
 k_{\mathrm{align}}
\frac{\sum_{j\in\mathcal N_i}w_{ij}\mathbf{e}_j}
{\sum_{j\in\mathcal N_i}w_{ij}+\varepsilon}.
```

Local cohesion is
```math
\bar{\mathbf{x}}_i=\frac{1}{|\mathcal N_i|}\sum_{j\in\mathcal N_i}\mathbf{x}_j,
\qquad
\mathbf A_i^{\mathrm{coh}}=k_{\mathrm{coh}}(\bar{\mathbf{x}}_i-\mathbf{x}_i),
```
with $\bar{\mathbf{x}}_i=\mathbf{x}_i$ for an empty neighbourhood.

### Predictive inter-agent safety

A finite geometric safety distance $d_{\mathrm{safe}}$ is enforced between flock agents. For neighbour $j$, define
```math
\mathbf r_{ij}=\mathbf{x}_j-\mathbf{x}_i,
\qquad
\mathbf v_{ij}^{\mathrm{rel}}=v_j\mathbf{e}_j-v_i\mathbf{e}_i.
```
The time of closest approach over a short horizon $T_a$ is
```math
\tau_{ij}^{*}
=
\operatorname{clip}_{[0,T_a]}
\left(
-\frac{\mathbf r_{ij}\cdot\mathbf v_{ij}^{\mathrm{rel}}}
{\|\mathbf v_{ij}^{\mathrm{rel}}\|^2+\varepsilon}
\right),
```
and
```math
\mathbf r_{ij}^{*}=\mathbf r_{ij}+\tau_{ij}^{*}\mathbf v_{ij}^{\mathrm{rel}}.
```
The controller uses the smaller of $\|\mathbf r_{ij}\|$ and $\|\mathbf r_{ij}^{*}\|$ as the effective separation $d_{ij}$. If $d_{ij}<d_{\mathrm{safe}}$, a smooth repulsive term is applied,
```math
\boxed{
\mathbf A_i^{\mathrm{sep}}
=
q_i\sum_{j\in\mathcal N_i}
 k_{\mathrm{sep}}
 S\!\left(\frac{d_{\mathrm{safe}}-d_{ij}}{d_{\mathrm{safe}}}\right)
 \hat{\mathbf n}_{ij},
}
```
where $\hat{\mathbf n}_{ij}$ points away from the more threatening current or predicted relative position and $S(z)=3z^2-2z^3$ on $[0,1]$. Thus agents preserve a safety gap not only when already close, but also when their relative motion predicts an imminent close pass.

### Predictive moving-obstacle navigation

For moving obstacle $a$ with centre $\mathbf{o}_a$, velocity $\mathbf{u}_a$, and conservative collision radius $\rho_a$, let
```math
\mathbf r_{ia}=\mathbf{o}_a-\mathbf{x}_i,
\qquad
\mathbf v_{ia}^{\mathrm{rel}}=\mathbf{u}_a-v_i\mathbf{e}_i.
```
The predicted time of closest approach over horizon $T_p$ is
```math
\boxed{
\tau_{ia}^{*}
=
\operatorname{clip}_{[0,T_p]}
\left(
-\frac{\mathbf r_{ia}\cdot\mathbf v_{ia}^{\mathrm{rel}}}
{\|\mathbf v_{ia}^{\mathrm{rel}}\|^2+\varepsilon}
\right).
}
```
Then
```math
\mathbf r_{ia}^{*}=\mathbf r_{ia}+\tau_{ia}^{*}\mathbf v_{ia}^{\mathrm{rel}},
\qquad
d_{ia}^{*}=\|\mathbf r_{ia}^{*}\|-\rho_a.
```
The more threatening of the current and predicted clearances is denoted $d_{ia}$, with outward unit direction $\hat{\mathbf n}_{ia}$. To avoid the inefficient behaviour of pure radial repulsion, the obstacle response contains both a clearance-producing radial component and a tangential passing component,
```math
\boxed{
\mathbf A_i^{\mathrm{obs}}
=
q_i\sum_a
S\!\left(
\frac{\ell_{\mathrm{obs}}-d_{ia}}{\ell_{\mathrm{obs}}}
\right)
(1+c_v\nu_{ia})
\left[
 k_r\hat{\mathbf n}_{ia}+k_t\hat{\mathbf t}_{ia}
\right].
}
```
Here $\nu_{ia}$ is the positive closing-speed contribution. The tangent $\hat{\mathbf t}_{ia}$ is chosen consistently for the flock: agents pass below an obstacle lying above the instantaneous flock centre and above one lying below it. This group-consistent side choice reduces unnecessary splitting and oscillation around an oncoming vehicle.

The demonstration represents the flock as small car-like agents moving in the fixed task direction $\hat{\mathbf m}=(1,0)$; there is no corridor, lane potential, periodic boundary condition, or coordinate recycling. Agent coordinates evolve continuously in the global plane. Only the plotting camera translates with the flock for visualization. The forward task contribution is simply
```math
\mathbf A_i^{\mathrm{goal}}=k_{\mathrm{goal}}\hat{\mathbf m}.
```
The complete steering vector is therefore
```math
\boxed{
\mathbf{D}_i
=
\mathbf{D}_i^{\mathrm{align}}
+\mathbf A_i^{\mathrm{coh}}
+\mathbf A_i^{\mathrm{sep}}
+\mathbf A_i^{\mathrm{obs}}
+\mathbf A_i^{\mathrm{goal}}.
}
```
The target direction is
```math
\hat{\mathbf d}_i=\frac{\mathbf{D}_i}{\|\mathbf{D}_i\|},
```
with $\hat{\mathbf d}_i=\mathbf{e}_i$ if the norm is numerically zero.

## Bounded heading actuation

In 2-D let $\theta_i=\operatorname{atan2}(e_{iy},e_{ix})$ and $\theta_i^*=\operatorname{atan2}(\hat d_{iy},\hat d_{ix})$. Then
```math
\Delta\theta_i=\operatorname{wrap}(\theta_i^*-\theta_i),
```
and the commanded yaw rate is
```math
\boxed{\dot\psi_i^{\mathrm{cmd}}=\operatorname{clip}\left(\frac{\Delta\theta_i}{\Delta t},-\omega_{\max},\omega_{\max}\right).}
```
For a point-agent simulation,
```math
\theta_i^{n+1}=\theta_i^n+\dot\psi_i^{\mathrm{cmd}}\Delta t,
\qquad
\mathbf{e}_i^{n+1}=(\cos\theta_i^{n+1},\sin\theta_i^{n+1}).
```
In Gazebo this angle integration is performed by the vehicle/autopilot, not by the cognitive controller.

## Selective variable-speed extension

### Braking demand

Use the directional disagreement already produced by the steering field,
```math
\boxed{\mu_i=1-\mathbf{e}_i\cdot\hat{\mathbf d}_i,\qquad 0\le\mu_i\le2.}
```
A sudden obstacle rotates $\hat{\mathbf d}_i$ away from the current heading; therefore $\mu_i$ rises automatically and can trigger braking without a separate danger state.

### Rejoining demand

Let $\mathbf{x}_i^r$ denote a rejoining reference available to agent $i$. In a centralized simulation it may be the global swarm centroid
```math
\mathbf{x}_c=\frac1N\sum_j\mathbf{x}_j.
```
For real robots, $\mathbf{x}_i^r$ should be supplied by a swarm-state aggregator, a distributed/multi-hop centroid estimate, or a last-known group reference. This dependency is explicit because two physically disconnected components cannot infer each other’s centroids from purely local sensing alone.

Define
```math
\hat{\mathbf c}_i=\frac{\mathbf{x}_i^r-\mathbf{x}_i}{\|\mathbf{x}_i^r-\mathbf{x}_i\|+\varepsilon}
```
and
```math
\boxed{\rho_i=\tanh\!\left(\frac{\|\mathbf{x}_i^r-\mathbf{x}_i\|}{L_r}\right)\,[\mathbf{e}_i\cdot\hat{\mathbf c}_i]_+.}
```
Hence a separated agent accelerates only after it has turned approximately toward the rejoining reference. This prevents a speed boost in the wrong direction.

### Desired speed and acceleration

The desired speed is
```math
\boxed{v_i^*=\operatorname{clip}_{[v_{\min},v_{\max}]}\left\{v_0\left[1+\eta_r\rho_i-\frac{\eta_b}{2}\mu_i\right]\right\}.}
```
The physical acceleration command is
```math
\boxed{a_i^{\mathrm{cmd}}=\operatorname{sat}_{[-a_{\mathrm{br}},a_{\mathrm{acc}}]}\left(\frac{v_i^*-v_i}{\tau_v}\right).}
```
The interpretation is intentionally asymmetric:
```math
\begin{aligned}
\rho_i>0,\ \mu_i\approx0 &\Rightarrow \text{catch-up acceleration},\\
\mu_i>0 &\Rightarrow \text{evasive braking},\\
\rho_i\approx0,\ \mu_i\approx0 &\Rightarrow v_i^*\approx v_0.
\end{aligned}
```

For the point-agent verifier,
```math
\begin{aligned}
v_i^{n+1}&=\operatorname{clip}_{[v_{\min},v_{\max}]}\left(v_i^n+a_i^{\mathrm{cmd}}\Delta t\right),\\
\mathbf{x}_i^{n+1}&=\mathbf{x}_i^n+v_i^{n+1}\mathbf{e}_i^{n+1}\Delta t.
\end{aligned}
```

## Exact fixed-speed limit

Set
```math
\boxed{\eta_r=\eta_b=0}
```
and initialize $v_i(0)=v_0$. Then $v_i^*=v_0$, $a_i^{\mathrm{cmd}}=0$, and
```math
v_i(t)=v_0,\qquad \dot{\mathbf{x}}_i=v_0\mathbf{e}_i.
```
Thus the previous constant-speed model is recovered exactly while all internal cognitive dynamics and the heading update remain unchanged.

## Complete update order

At each controller cycle $n\to n+1$:

1.  Read physical states $(\mathbf{x}_i,\mathbf{e}_i,v_i)$, neighbour states, and obstacle states $(\mathbf{o}_a,\mathbf{u}_a,\rho_a)$.

2.  Recompute $A_{ij}$ and $\mathcal N_i$.

3.  Compute $h_z^{ij}$ for all active directed channels.

4.  Integrate $(m_x^{ij},m_y^{ij},m_z^{ij})$ and obtain $w_{ij}$.

5.  Compute $M_i$, $s_i^*$, update $s_i$, and obtain $q_i$.

6.  Build weighted alignment, cohesion, separation, obstacle and optional task steering; form $\mathbf{D}_i$ and $\hat{\mathbf d}_i$.

7.  Compute yaw-rate command from the bounded heading error.

8.  Compute rejoin demand $\rho_i$ and manoeuvre demand $\mu_i$.

9.  Compute $v_i^*$ and rate-limited $a_i^{\mathrm{cmd}}$.

10. Export the physical commands to the simulator/robot interface.

The information flow is
```math
\boxed{\text{sensors/peers}\to \mathbf m_{ij}\to w_{ij}\to M_i\to s_i\to q_i\to \hat{\mathbf d}_i\to(\dot\psi_i^{\mathrm{cmd}},v_i^*,a_i^{\mathrm{cmd}}).}
```

## ROS 2/Gazebo-ready interface

The implementation is deliberately reduced to two self-contained Python files. The ROS 2 file does not integrate an artificial point-mass plant: it reads the physical state from odometry, updates all internal cognitive variables, computes the swarm control law, and publishes only physical command references.

For agent $i$, the ROS 2 node subscribes to `/agent_i/odom` (`nav_msgs/Odometry`), from which it obtains $\mathbf{x}_i$, heading $\mathbf{e}_i$, and measured speed $v_i$. Moving obstacles are supplied on `/swarm/obstacles` (`std_msgs/Float64MultiArray`) as flattened rows `[x, y, vx, vy, radius]`, so measured obstacle velocity can be used in the short-horizon prediction $\mathbf{o}_a^p=\mathbf{o}_a+T_p\mathbf{u}_a$.

At each control cycle the node executes the complete chain
```math
\boxed{
\text{odometry/obstacles}
\to \mathbf m_{ij}
\to w_{ij}
\to M_i
\to s_i
\to q_i
\to \hat{\mathbf d}_i
\to (v_i^*,a_i^{\mathrm{cmd}},\dot\psi_i^{\mathrm{cmd}})
\to \mathbf v_i^{\mathrm{cmd}}.
}
```
The world-frame planar velocity reference is
```math
\mathbf v_i^{\mathrm{cmd}}=v_i^{\mathrm{cmd}}\hat{\mathbf d}_i,
```
with $v_i^{\mathrm{cmd}}$ obtained by one acceleration-limited step toward $v_i^*$. The bounded yaw-rate reference is computed from the desired/current heading error.

The node publishes


| ROS 2 output         | Content                                                                                            |
|:---------------------|:---------------------------------------------------------------------------------------------------|
| `/agent_i/cmd_vel`   | `geometry_msgs/TwistStamped`; world-frame $v_x^{\mathrm{cmd}},v_y^{\mathrm{cmd}}$ and $\dot\psi_i^{\mathrm{cmd}}$ |
| `/swarm/diagnostics` | flattened per-agent rows $[v_i^*,a_i^{\mathrm{cmd}},\dot\psi_i^{\mathrm{cmd}},\mu_i,\rho_i,s_i,q_i,M_i,n_i]$ |


Thus the variables directly available for Gazebo/robot experiments are
```math
\boxed{
\mathbf v_i^{\mathrm{cmd}},\;\dot\psi_i^{\mathrm{cmd}},\;v_i^*,\;a_i^{\mathrm{cmd}},\;
\mu_i,\;\rho_i,\;s_i,\;q_i,\;M_i,\;n_i .
}
```
The present `publish_commands()` function uses `TwistStamped`. If PX4, ArduPilot, or a particular Gazebo bridge expects body-frame velocity or a different message type, only this output function needs adaptation; the cognitive and swarm equations remain unchanged. Vehicle mass, inertia, thrust, attitude response, and motor dynamics remain the responsibility of Gazebo/autopilot.

## Two-file software implementation

Only two Python files are supplied:


| File                 | Responsibility                                                                                                                                                                                                                                                                                                                                                                       |
|:---------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `swarm_demo.py`      | self-contained real-time 2-D validation: full internal cognitive dynamics, steering, rejoining acceleration, evasive braking, bounded physical update, a compact flock, and a continuing asynchronous stream of moving obstacles with randomized arrival times, speeds, lateral offsets, and crossing angles; the obstacles are visually distinct from the agents and no corridor is imposed |
| `ros2_swarm_node.py` | self-contained ROS 2/Gazebo controller: odometry and obstacle input, persistent $\mathbf m_{ij}$ and $s_i$, full cognitive-to-physical update, `cmd_vel` publication, and diagnostic publication for every controlled agent                                                                                                                                                              |


No auxiliary local Python modules are required. Model parameters are held in an internal `Config` dataclass inside each self-contained file. This duplication is intentional: the real-time verifier can run directly with NumPy/Matplotlib, while the ROS 2 file can be copied into a ROS package without dependency on a custom Python module tree.

### Animation execution

Live mode keeps an explicit reference to the Matplotlib `FuncAnimation` object while the GUI event loop owns the figure. It requires an interactive Matplotlib backend such as TkAgg, QtAgg, or the macOS backend. Export mode uses a finite, deterministic frame sequence, writes it through Matplotlib's FFmpeg writer, closes the figure, and exits without requiring a display.

### Unbounded longitudinal evolution and split–reformation diagnostic

No periodic boundary condition is used for the flock. The physical update is always
```math
\mathbf{x}_i^{n+1}=\mathbf{x}_i^n+v_i^{n+1}\mathbf{e}_i^{n+1}\Delta t,
```
and the resulting global coordinates are retained for the full simulation. For visualization only, the plotting camera has a separate state. It looks ahead of the mean flock heading, moves only when the flock crosses a dead zone, and approaches its target with a finite response time. The flock therefore travels visibly within the canvas instead of being fixed to its centre, while its continuously advancing global coordinates remain unchanged.

Each demonstration obstacle is assigned a speed $u_a$ sampled independently for its current encounter,
```math
\mathbf{u}_a=u_a(\cos\phi_a,\sin\phi_a),
\qquad
\dot\phi_a=\omega_a(t),
```
where the turn-rate target is piecewise-random and the applied turn rate approaches that target smoothly. The relative heading remains bounded around the oncoming direction,
```math
|\operatorname{wrap}(\phi_a-\pi)|\leq \phi_{\max}.
```
After a vehicle passes a fixed distance behind the flock it becomes inactive, waits for a random interval, and is spawned far ahead of the flock's current global position with a new speed, lateral offset, and heading. This is asynchronous obstacle generation, not a periodic boundary condition on the swarm.

To show the desired split–reformation behaviour explicitly, the demo reports the number of connected flock components under the sensing graph
```math
A_{ij}=\mathbf 1\!\left(\|\mathbf{x}_i-\mathbf{x}_j\|<R\right).
```
A close obstacle can temporarily increase the number of connected components; successful rejoining is indicated when the count returns to one, accompanied by a reduction of the flock span and rejoining demand $\rho_i$.

## Reference defaults and randomized road-like scenario

The internal cognitive parameters remain close to the representative scales used in the present model ($T_1=0.6$, $T_2=0.3$, $\Gamma=1$, $\kappa=1.2$, $\gamma_s=0.03$, $\lambda_{\mathrm{fb}}=2$). The supplied demonstration is intentionally an unbounded road-like encounter rather than a corridor-like arena: a compact flock of small agents advances indefinitely in global $+x$. A reusable pool generates larger unruly vehicles asynchronously ahead of the current flock. Each travels generally in the opposite direction and executes a bounded random heading weave. Their lateral spawn positions are biased toward the flock so that close encounters can split the group; inactive slots wait for a fresh random delay after the corresponding vehicle has passed well behind the moving flock.

The controller-level defaults are


| Parameter        | Value              | Parameter                | Value                       |
|:-----------------|:-------------------|:-------------------------|:----------------------------|
| $\Delta t$       | $0.04$             | $R$                      | $12$                        |
| $d_{\mathrm{safe}}$   | $2.4$              | $T_a$                    | $0.75\,\mathrm{s}$          |
| $\ell_{\mathrm{obs}}$ | $8.5$              | $T_p$                    | $1.0\,\mathrm{s}$           |
| $k_{\mathrm{sep}}$    | $3.4$              | $k_r,k_t$                | $5.2,\,4.6$                 |
| $k_{\mathrm{goal}}$   | $1.65$             | $\omega_{\max}$          | $1.7\,\mathrm{rad\,s^{-1}}$ |
| $v_0$            | $3.0$              | $v_{\min},v_{\max}$      | $0.9,\,4.6$                 |
| $\eta_r,\eta_b$  | $0.55,\,1.0$       | $L_r$                    | $10$                        |
| $\tau_v$         | $0.42\,\mathrm{s}$ | $a_{\mathrm{acc}},a_{\mathrm{br}}$ | $2.2,\,4.5$                 |


The demonstration obstacles use a larger car-like footprint (approximately $4.2\times2.0$ simulation units), an independently sampled speed in $[2.4,6.2]$ simulation units per second for each encounter, and a conservative circular control envelope $\rho_a=2.25$. Their heading is allowed to wander within approximately $\pm0.55$ rad of the oncoming direction. Flock agents are intentionally drawn much smaller (approximately $0.95\times0.50$ units visually), rotate with their physical headings, and are coloured by speed so that braking and catch-up acceleration are visible.

These are controller-level working defaults, not universal optima. The tuning emphasizes four objectives simultaneously: (i) no obstacle contact, (ii) preservation of a finite inter-agent gap, (iii) early smooth lateral navigation rather than last-second radial escape, and (iv) rapid recovery of flock compactness after the encounter. The tangential obstacle component is deliberately comparable to the radial component because a purely radial field tends to brake and scatter the flock around a large oncoming vehicle. Final Gazebo values must be retuned against actual vehicle dimensions, steering limits, sensing delay, and the precise frame conventions used by the simulator or autopilot.

## Practical recommendation for Gazebo transition

First validate three separate cases: (i) $\eta_r=\eta_b=0$ to reproduce the constant-speed baseline, (ii) $\eta_r>0,\eta_b=0$ to isolate rejoining acceleration, and (iii) $\eta_r>0,\eta_b>0$ with moving obstacles. Only after the controller-level behavior is stable should the point-agent plant be replaced with a multirotor model. This preserves a clean distinction between cognitive/swarm dynamics and vehicle dynamics.
