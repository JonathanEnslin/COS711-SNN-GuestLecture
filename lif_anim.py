#!/usr/bin/env python3
# LIF neuron — phases: static current -> sine+noise -> off -> discrete input pulses.
# Input current shown on TOP (with red event lines), V_m in the MIDDLE (with V_th, E_L, and threshold dots),
# spike raster at the BOTTOM. Left→right fill, then scroll.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from dataclasses import dataclass, field

# ---------------- Parameters (tweak these) ----------------
@dataclass
class LIFParams:
    C: float = 200e-12         # F
    gL: float = 10e-9          # S
    EL: float = -70e-3         # V
    Vth: float = -50e-3        # V
    Vreset: float = -65e-3     # V
    tau_ref: float = 2e-3      # s

@dataclass
class DriveParams:
    # Phase 1: static
    I_static: float = 210e-12  # A
    t_static: float = 0.5      # s

    # Phase 2: sine + noise (starts right after t_static)
    I_bias: float = 120e-12    # A
    I_amp: float = 250e-12     # A (sine amplitude)
    f_hz: float = 20.0         # Hz
    sigma: float = 10e-12      # A (Gaussian noise)
    t_sine: float = 0.57       # s duration of sine phase

    # Phase 3: off (baseline after t_static + t_sine)
    sigma_after: float = 0.0   # A (residual noise after off; 0 => exact zero)

@dataclass
class EventParams:
    """Discrete 'spiking inputs' injected AFTER the sine phase."""
    n_slow: int = 4                 # widely spaced first pulses
    slow_gap: float = 0.20          # s between slow pulses (approx)
    n_burst: int = 6                # rapid pulses following
    burst_gap: float = 0.035        # s between burst pulses (approx)
    jitter: float = 0.010           # s of random +/- jitter for each pulse
    start_after: float = 0.10       # s after sine ends before first pulse
    delta_v_mv: float = 5.0         # ~desired jump in Vm per pulse (mV)
    # Computed at runtime:
    times: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))

# Sim / display
dt = 1e-4              # 0.1 ms integration
fps = 60               # UI target FPS
sim_speed = 0.2        # simulated seconds per real second (increase to run faster)
steps_per_frame = max(1, int(sim_speed / (dt * fps)))
window_s = 1.75        # seconds, rolling history

lif = LIFParams()
drv = DriveParams()
evp = EventParams()

# ---------------- State ----------------
win_steps = int(window_s / dt)
t = 0.0
V = lif.EL
ref_timer = 0.0

V_buf = np.full(win_steps, lif.EL, float)
I_buf = np.zeros(win_steps, float)
write_idx = -1

# Absolute spike times (s) from threshold crossings
spike_times = []

# Precompute event pulse amplitude from desired ΔV ≈ (area / C)
# We inject a 1-timestep rectangular pulse: area = I * dt => I = area / dt = (C * ΔV) / dt
pulse_area = lif.C * (evp.delta_v_mv * 1e-3)  # Coulombs
pulse_amp = pulse_area / dt                   # Amps (applied for one dt)

# ---------------- Figure (CURRENT ON TOP) ----------------
plt.rcParams["toolbar"] = "toolmanager"
fig = plt.figure(figsize=(9, 6.5), constrained_layout=True)
# Order: current (top), Vm (middle), spikes (bottom)
gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1, 2, 1])

ax_i = fig.add_subplot(gs[0, 0])  # current (TOP)
ax_v = fig.add_subplot(gs[1, 0], sharex=ax_i)  # Vm (MIDDLE)
ax_s = fig.add_subplot(gs[2, 0], sharex=ax_i)  # spikes (BOTTOM)

# Input current plot
(line_I,) = ax_i.plot([], [], lw=1.2)
ax_i.set_ylabel("I (pA)")
ax_i.axhline(0, lw=0.8, ls=":", alpha=0.6)

# Vm trace + reference lines + threshold dots
(line_vm,) = ax_v.plot([], [], lw=1.5)
ax_v.set_ylabel("V_m (mV)")
ax_v.set_ylim((lif.EL - 0.015) * 1e3, (lif.Vth + 0.010) * 1e3)
ax_v.axhline(lif.Vth * 1e3, ls="--", lw=1, alpha=0.7)  # threshold
ax_v.axhline(lif.EL  * 1e3, ls=":",  lw=1, alpha=0.7)  # resting potential

# Red dots at threshold crossing times
(thresh_pts,) = ax_v.plot([], [], 'o', ms=4, color='red', mec='none', zorder=3)

ax_v.set_title("LIF (live): current (top, red event lines), V_m (middle, V_th • E_L), spikes (bottom) — left→right fill")

# Spike raster (vertical ticks via LineCollection)
spike_lc = LineCollection([], linewidths=1.5)
ax_s.add_collection(spike_lc)
ax_s.set_ylim(0, 1)
ax_s.set_yticks([])
ax_s.set_ylabel("Spikes")
ax_s.set_xlabel("Time (s)")

# Red vertical lines for input events on the CURRENT axis
event_lc = LineCollection([], linewidths=1.6, colors='red')
ax_i.add_collection(event_lc)

# ---------------- Dynamics ----------------
def lif_step(V, ref_timer, I_t):
    spiked = False
    if ref_timer > 0.0:
        ref_timer = max(0.0, ref_timer - dt)
        return V, ref_timer, spiked
    dV = (-lif.gL * (V - lif.EL) + I_t) / lif.C
    V = V + dt * dV
    if V >= lif.Vth:
        spiked = True
        V = lif.Vreset
        ref_timer = lif.tau_ref
    return V, ref_timer, spiked

def phase_boundaries():
    t1 = drv.t_static
    t2 = t1 + drv.t_sine
    return t1, t2

def build_event_times(rng: np.random.Generator):
    """Create post-sine event times: slow (widely spaced) then a rapid burst, with jitter."""
    t1, t2 = phase_boundaries()
    t_start = t2 + evp.start_after

    # Slow pulses
    slow_ts = []
    cur = t_start
    for _ in range(evp.n_slow):
        cur += evp.slow_gap + rng.uniform(-evp.jitter, evp.jitter)
        slow_ts.append(max(t_start, cur))

    # Burst pulses
    burst_ts = []
    if slow_ts:
        cur = slow_ts[-1]
    else:
        cur = t_start
    for _ in range(evp.n_burst):
        cur += evp.burst_gap + rng.uniform(-evp.jitter, evp.jitter)
        burst_ts.append(max(t_start, cur))

    ts = np.array(slow_ts + burst_ts, dtype=float)
    ts.sort()
    return ts

# Build event schedule once
_rng = np.random.default_rng(12345)
evp.times = build_event_times(_rng)

def input_current(now: float) -> float:
    """Piecewise input: static -> sine+noise -> off (+ discrete event pulses)."""
    t1, t2 = phase_boundaries()

    # Base drive
    if now < t1:
        base = drv.I_static
    elif now < t2:
        base = drv.I_bias + drv.I_amp * np.sin(2 * np.pi * drv.f_hz * (now - t1)) + drv.sigma * np.random.randn()
    else:
        base = drv.sigma_after * np.random.randn() if drv.sigma_after > 0.0 else 0.0

    # Add delta-like pulses (one-timestep high-current injections)
    # If |now - t_event| < dt/2, apply one-step pulse.
    if evp.times.size:
        # Fast check using a small window around 'now'
        # Find nearest event index via searchsorted
        idx = np.searchsorted(evp.times, now)
        candidates = []
        if idx < evp.times.size:
            candidates.append(evp.times[idx])
        if idx > 0:
            candidates.append(evp.times[idx - 1])
        for te in candidates:
            if abs(now - te) <= (dt * 0.5):
                base += pulse_amp
                break

    return base

def advance(n_steps):
    global t, V, ref_timer, write_idx
    for _ in range(n_steps):
        I_t = input_current(t)
        V, ref_timer, spiked = lif_step(V, ref_timer, I_t)
        t += dt
        if spiked:
            spike_times.append(t)
        write_idx = (write_idx + 1) % win_steps
        V_buf[write_idx] = V
        I_buf[write_idx] = I_t

def get_buffer_in_order(buf):
    # chronological last window
    if write_idx < 0 or write_idx == win_steps - 1:
        return buf.copy()
    return np.concatenate((buf[write_idx + 1 :], buf[: write_idx + 1]))

def init_axes():
    # Fix the axis to [0, window_s] so lines fill from left to right initially.
    ax_i.set_xlim(0.0, window_s)
    ax_v.set_xlim(0.0, window_s)
    ax_s.set_xlim(0.0, window_s)

    # I-axis: cover static and sine ranges (±3σ)
    I_candidates = [drv.I_static, drv.I_bias + drv.I_amp, drv.I_bias - drv.I_amp]
    I_min = (min(I_candidates) - 3 * drv.sigma) * 1e12
    I_max = (max(I_candidates) + 3 * drv.sigma) * 1e12
    pad = max(5.0, 0.1 * (I_max - I_min))
    ax_i.set_ylim(I_min - pad, I_max + pad)

    return (line_I, line_vm, spike_lc, thresh_pts, event_lc)

def update_spike_raster_fill_mode():
    """Render spikes so they appear at correct x when filling (t < window_s)
    and when scrolling (t >= window_s)."""
    if t < window_s:
        segs = [((ts, 0.0), (ts, 1.0)) for ts in spike_times if 0.0 <= ts <= t]
    else:
        t_start = t - window_s
        segs = [((ts - t_start, 0.0), (ts - t_start, 1.0)) for ts in spike_times if ts >= t_start]
    spike_lc.set_segments(segs)

def threshold_marker_positions():
    """Return x, y arrays for threshold markers (one per spike) in current view."""
    if t < window_s:
        xs = [ts for ts in spike_times if 0.0 <= ts <= t]
    else:
        t_start = t - window_s
        xs = [ts - t_start for ts in spike_times if ts >= t_start]
    ys = [lif.Vth * 1e3] * len(xs)  # threshold level (mV)
    return np.array(xs, float), np.array(ys, float)

def update_event_lines_fill_mode():
    """Update red vertical lines for input events in the current view on ax_i."""
    if evp.times.size == 0:
        event_lc.set_segments([])
        return

    if t < window_s:
        xs = [te for te in evp.times if 0.0 <= te <= t]
    else:
        t_start = t - window_s
        xs = [te - t_start for te in evp.times if te >= t_start]

    if not xs:
        event_lc.set_segments([])
        return

    # Vertical lines across the full y-range of ax_i
    y0, y1 = ax_i.get_ylim()
    segs = [((x, y0), (x, y1)) for x in xs]
    event_lc.set_segments(segs)

def animate(_):
    advance(steps_per_frame)

    Vm_win = get_buffer_in_order(V_buf) * 1e3       # mV
    I_win  = get_buffer_in_order(I_buf) * 1e12      # pA

    if t < window_s:
        # Only show the portion we've simulated, growing left→right.
        n = max(1, min(int(t / dt), win_steps))
        x = np.linspace(0.0, n * dt, n, endpoint=False)
        line_vm.set_data(x, Vm_win[-n:])
        line_I.set_data(x, I_win[-n:])
    else:
        # Full window, mapped to [0, window_s] with scrolling.
        x = np.linspace(0.0, window_s, win_steps, endpoint=False)
        line_vm.set_data(x, Vm_win)
        line_I.set_data(x, I_win)

    # Spikes + threshold dots
    update_spike_raster_fill_mode()
    tx, ty = threshold_marker_positions()
    thresh_pts.set_data(tx, ty)

    # Event red lines on current
    update_event_lines_fill_mode()

    # x-lims remain [0, window_s] to preserve the left→right behavior
    return (line_I, line_vm, spike_lc, thresh_pts, event_lc)

# ---------------- Run ----------------
if __name__ == "__main__":
    init_axes()
    t1, t2 = phase_boundaries()
    txt = (f"τm={(lif.C/lif.gL)*1e3:.0f} ms | Phase1: {drv.t_static:.2f}s @ {drv.I_static*1e12:.0f} pA | "
           f"Phase2: {drv.t_sine:.2f}s sine {drv.f_hz:.0f} Hz, amp {drv.I_amp*1e12:.0f} pA | "
           f"events after t≈{t2+evp.start_after:.2f}s (slow→burst, ΔV≈{evp.delta_v_mv:.1f} mV each) | "
           f"sim×{sim_speed}")
    ax_v.text(0.01, 0.02, txt, transform=ax_v.transAxes, fontsize=9, alpha=0.85)

    ani = FuncAnimation(fig, animate, init_func=init_axes,
                        interval=1000.0 / fps, blit=True)
    plt.show()
