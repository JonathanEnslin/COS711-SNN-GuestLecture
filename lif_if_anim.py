#!/usr/bin/env python3
# If the plot is the wrong size at start, just resize the window.

# STACKED COMPARISON (one column):
# 1) Input current (shared; red event lines)
# 2) LIF membrane potential (with V_th, E_L, red threshold dots; overshoot shown)
# 3) IF  membrane potential (with V_th, E_L, red threshold dots; overshoot shown)
# 4) LIF spike raster  (shorter)
# 5) IF  spike raster   (shorter)
#
# Traces fill left→right, then scroll once full. Both neurons receive the EXACT same input,
# including noise and discrete event pulses.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from dataclasses import dataclass, field

# ---------------- Parameters (tweak these) ----------------
@dataclass
class LIFParams:
    C: float = 200e-12         # F
    gL: float = 10e-9          # S (leak)
    EL: float = -70e-3         # V
    Vth: float = -54e-3        # V
    Vreset: float = -65e-3     # V
    tau_ref: float = 1.0e-3    # s

@dataclass
class IFParams:
    C: float = 200e-12         # F
    EL: float = -70e-3         # V (ref line only)
    Vth: float = -54e-3        # V
    Vreset: float = -65e-3     # V
    tau_ref: float = 1.0e-3    # s
    # No leak (gL = 0): dV/dt = I/C

@dataclass
class DriveParams:
    # Phase 1: static
    I_static: float = 240e-12  # A
    t_static: float = 0.2      # s

    # Phase 2: sine + noise
    I_bias: float = 220e-12    # A
    I_amp: float = 400e-12     # A
    f_hz: float = 30.0         # Hz
    sigma: float = 15e-12      # A
    t_sine: float = 0.15       # s

    # Phase 3: off
    sigma_after: float = 0.0   # A

@dataclass
class EventParams:
    """Discrete 'spiking inputs' injected AFTER the sine phase."""
    n_slow: int = 4
    slow_gap: float = 0.06
    n_burst: int = 8
    burst_gap: float = 0.01
    jitter: float = 0.006
    start_after: float = 0.01
    delta_v_mv: float = 10.5
    times: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))

# ---------- Simulation/display ----------
dt = 1e-5              # s (0.01 ms)
fps = 60               # UI FPS
sim_speed = 0.15       # simulated seconds per real second
steps_per_frame = max(1, int(sim_speed / (dt * fps)))
window_s = 0.75        # s, rolling history

lif = LIFParams()
ifp = IFParams()
drv = DriveParams()
evp = EventParams()

# ---------- State ----------
win_steps = int(window_s / dt)
t = 0.0

# LIF
V_lif = lif.EL
ref_lif = 0.0
V_buf_lif = np.full(win_steps, lif.EL, float)
spike_times_lif = []

# IF
V_if = ifp.EL
ref_if = 0.0
V_buf_if = np.full(win_steps, ifp.EL, float)
spike_times_if = []

# Shared input
I_buf = np.zeros(win_steps, float)
write_idx = -1

# Event pulse amplitude: area = C*ΔV, so one-step I = area/dt
pulse_area = lif.C * (evp.delta_v_mv * 1e-3)
pulse_amp = pulse_area / dt

# ---------- Figure (stacked) ----------
plt.rcParams["toolbar"] = "toolmanager"
fig = plt.figure(figsize=(10, 8.0), constrained_layout=True)
# heights: current, Vm(LIF), Vm(IF), spikes(LIF small), spikes(IF small)
gs = fig.add_gridspec(nrows=5, ncols=1, height_ratios=[1.1, 2.0, 2.0, 0.7, 0.7])

ax_i  = fig.add_subplot(gs[0, 0])  # current (shared, TOP)
ax_vL = fig.add_subplot(gs[1, 0], sharex=ax_i)  # LIF Vm
ax_vR = fig.add_subplot(gs[2, 0], sharex=ax_i)  # IF  Vm
ax_sL = fig.add_subplot(gs[3, 0], sharex=ax_i)  # LIF spikes (shorter)
ax_sR = fig.add_subplot(gs[4, 0], sharex=ax_i)  # IF  spikes (shorter)

# Input current line
(line_I,) = ax_i.plot([], [], lw=1.2)
ax_i.set_ylabel("I (pA)")
ax_i.axhline(0, lw=0.8, ls=":", alpha=0.6)

# Vm traces + ref lines + threshold dots
(line_vm_L,) = ax_vL.plot([], [], lw=1.5, color='C0')
(line_vm_R,) = ax_vR.plot([], [], lw=1.5, color='C0')
for axv, Vth, EL in ((ax_vL, lif.Vth, lif.EL), (ax_vR, ifp.Vth, ifp.EL)):
    axv.set_ylabel("V_m (mV)")
    axv.set_ylim((EL - 0.015) * 1e3, (Vth + 0.010) * 1e3)
    axv.axhline(Vth * 1e3, ls="--", lw=1, alpha=0.7)  # threshold
    axv.axhline(EL  * 1e3, ls=":",  lw=1, alpha=0.7)  # resting

# Red dots at threshold crossings
(thresh_pts_L,) = ax_vL.plot([], [], 'o', ms=4, color='red', mec='none', zorder=3)
(thresh_pts_R,) = ax_vR.plot([], [], 'o', ms=4, color='red', mec='none', zorder=3)

# Titles
ax_i.set_title("Input (shared; red = event pulses)")
ax_vL.set_title("LIF: V_m (V_th • E_L) — overshoot shown")
ax_vR.set_title("IF:  V_m (V_th • E_L) — overshoot shown")

# Spike rasters
spike_lc_L = LineCollection([], linewidths=1.5)
spike_lc_R = LineCollection([], linewidths=1.5)
ax_sL.add_collection(spike_lc_L)
ax_sR.add_collection(spike_lc_R)
for axs in (ax_sL, ax_sR):
    axs.set_ylim(0, 1)
    axs.set_yticks([])
    axs.set_ylabel("Spikes")
    axs.set_xlabel("Time (s)")

# Red vertical event lines on current axis
event_lc = LineCollection([], linewidths=1.6, colors='red')
ax_i.add_collection(event_lc)

# ---------- Input & dynamics ----------
def phase_boundaries():
    t1 = drv.t_static
    t2 = t1 + drv.t_sine
    return t1, t2

def build_event_times(rng: np.random.Generator):
    t1, t2 = phase_boundaries()
    t_start = t2 + evp.start_after
    slow_ts, cur = [], t_start
    for _ in range(evp.n_slow):
        cur += evp.slow_gap + rng.uniform(-evp.jitter, evp.jitter)
        slow_ts.append(max(t_start, cur))
    burst_ts, cur = [], (slow_ts[-1] if slow_ts else t_start)
    for _ in range(evp.n_burst):
        cur += evp.burst_gap + rng.uniform(-evp.jitter, evp.jitter)
        burst_ts.append(max(t_start, cur))
    ts = np.array(slow_ts + burst_ts, dtype=float); ts.sort()
    return ts

_rng = np.random.default_rng(12345)
evp.times = build_event_times(_rng)

def input_current(now: float) -> float:
    """Static -> sine+noise -> off (+ one-step delta pulses)."""
    t1, t2 = phase_boundaries()
    if now < t1:
        base = drv.I_static
    elif now < t2:
        base = drv.I_bias + drv.I_amp * np.sin(2 * np.pi * drv.f_hz * (now - t1)) + drv.sigma * np.random.randn()
    else:
        base = drv.sigma_after * np.random.randn() if drv.sigma_after > 0.0 else 0.0
    # Add delta pulse exactly at event times (one step)
    if evp.times.size:
        idx = np.searchsorted(evp.times, now)
        if idx < evp.times.size and abs(now - evp.times[idx]) <= (dt * 0.5):
            base += pulse_amp
        elif idx > 0 and abs(now - evp.times[idx - 1]) <= (dt * 0.5):
            base += pulse_amp
    return base

def lif_step(V, ref_timer, I_t):
    """LIF step with overshoot preserved (reset applied after plotting)."""
    if ref_timer > 0.0:
        return V, max(0.0, ref_timer - dt), False
    dV = (-lif.gL * (V - lif.EL) + I_t) / lif.C
    Vn = V + dt * dV
    return Vn, ref_timer, (Vn >= lif.Vth)

def if_step(V, ref_timer, I_t):
    """IF (no leak): dV/dt = I/C. Overshoot preserved; reset after plotting."""
    if ref_timer > 0.0:
        return V, max(0.0, ref_timer - dt), False
    dV = I_t / ifp.C
    Vn = V + dt * dV
    return Vn, ref_timer, (Vn >= ifp.Vth)

def advance(n_steps):
    """Advance both neurons with the SAME input."""
    global t, V_lif, V_if, ref_lif, ref_if, write_idx
    for _ in range(n_steps):
        I_t = input_current(t)

        # Step both neurons
        Vn_L, ref_lif, spk_L = lif_step(V_lif, ref_lif, I_t)
        Vn_R, ref_if,  spk_R = if_step(V_if,  ref_if,  I_t)

        t += dt
        write_idx = (write_idx + 1) % win_steps

        # Record overshoot values for plotting
        V_buf_lif[write_idx] = Vn_L
        V_buf_if[write_idx]  = Vn_R
        I_buf[write_idx]     = I_t

        # Apply reset + refractory AFTER plotting
        if spk_L:
            spike_times_lif.append(t)
            V_lif = lif.Vreset
            ref_lif = lif.tau_ref
        else:
            V_lif = Vn_L

        if spk_R:
            spike_times_if.append(t)
            V_if = ifp.Vreset
            ref_if = ifp.tau_ref
        else:
            V_if = Vn_R

# ---------- Plot helpers ----------
def get_buf_ordered(buf):
    if write_idx < 0 or write_idx == win_steps - 1:
        return buf.copy()
    return np.concatenate((buf[write_idx + 1:], buf[:write_idx + 1]))

def init_axes():
    for ax in (ax_i, ax_vL, ax_vR, ax_sL, ax_sR):
        ax.set_xlim(0.0, window_s)
    # Current y-lims from static+sine ±3σ
    I_candidates = [drv.I_static, drv.I_bias + drv.I_amp, drv.I_bias - drv.I_amp]
    I_min = (min(I_candidates) - 3 * drv.sigma) * 1e12
    I_max = (max(I_candidates) + 3 * drv.sigma) * 1e12
    pad = max(5.0, 0.1 * (I_max - I_min))
    ax_i.set_ylim(I_min - pad, I_max + pad)
    return (line_I, line_vm_L, line_vm_R, spike_lc_L, spike_lc_R, thresh_pts_L, thresh_pts_R, event_lc)

def update_spike_raster(ax_lc, spikes):
    if t < window_s:
        segs = [((ts, 0.0), (ts, 1.0)) for ts in spikes if 0.0 <= ts <= t]
    else:
        t0 = t - window_s
        segs = [((ts - t0, 0.0), (ts - t0, 1.0)) for ts in spikes if ts >= t0]
    ax_lc.set_segments(segs)

def thresh_dot_positions(spikes, Vth):
    if t < window_s:
        xs = [ts for ts in spikes if 0.0 <= ts <= t]
    else:
        t0 = t - window_s
        xs = [ts - t0 for ts in spikes if ts >= t0]
    ys = [Vth * 1e3] * len(xs)
    return np.array(xs, float), np.array(ys, float)

def update_event_lines():
    if evp.times.size == 0:
        event_lc.set_segments([])
        return
    if t < window_s:
        xs = [te for te in evp.times if 0.0 <= te <= t]
    else:
        t0 = t - window_s
        xs = [te - t0 for te in evp.times if te >= t0]
    if not xs:
        event_lc.set_segments([])
        return
    y0, y1 = ax_i.get_ylim()
    segs = [((x, y0), (x, y1)) for x in xs]
    event_lc.set_segments(segs)

# ---------- Animation ----------
def animate(_):
    advance(steps_per_frame)

    VmL = get_buf_ordered(V_buf_lif) * 1e3
    VmR = get_buf_ordered(V_buf_if)  * 1e3
    Itr = get_buf_ordered(I_buf)     * 1e12

    if t < window_s:
        n = max(1, min(int(t / dt), win_steps))
        x = np.linspace(0.0, n * dt, n, endpoint=False)
        line_vm_L.set_data(x, VmL[-n:])
        line_vm_R.set_data(x, VmR[-n:])
        line_I.set_data(x, Itr[-n:])
    else:
        x = np.linspace(0.0, window_s, win_steps, endpoint=False)
        line_vm_L.set_data(x, VmL)
        line_vm_R.set_data(x, VmR)
        line_I.set_data(x, Itr)

    # Rasters
    update_spike_raster(spike_lc_L, spike_times_lif)
    update_spike_raster(spike_lc_R, spike_times_if)

    # Threshold dots
    txL, tyL = thresh_dot_positions(spike_times_lif, lif.Vth)
    txR, tyR = thresh_dot_positions(spike_times_if,  ifp.Vth)
    thresh_pts_L.set_data(txL, tyL)
    thresh_pts_R.set_data(txR, tyR)

    # Event lines
    update_event_lines()

    return (line_I, line_vm_L, line_vm_R, spike_lc_L, spike_lc_R, thresh_pts_L, thresh_pts_R, event_lc)

# ---------- Run ----------
if __name__ == "__main__":
    init_axes()
    t1, t2 = phase_boundaries()
    ax_vL.text(0.01, 0.02,
               f"LIF: τm={(lif.C/lif.gL)*1e3:.0f} ms | Vth={lif.Vth*1e3:.0f} mV | "
               f"Phase1 {drv.t_static:.2f}s @ {drv.I_static*1e12:.0f} pA | "
               f"Phase2 {drv.t_sine:.2f}s {drv.f_hz:.0f} Hz, amp {drv.I_amp*1e12:.0f} pA",
               transform=ax_vL.transAxes, fontsize=9, alpha=0.85)
    ax_vR.text(0.01, 0.02,
               f"IF: no leak | Vth={ifp.Vth*1e3:.0f} mV | events start ≈{t2+evp.start_after:.2f}s, "
               f"ΔV≈{evp.delta_v_mv:.1f} mV",
               transform=ax_vR.transAxes, fontsize=9, alpha=0.85)

    ani = FuncAnimation(fig, animate, init_func=init_axes,
                        interval=1000.0 / fps, blit=True)
    plt.show()
