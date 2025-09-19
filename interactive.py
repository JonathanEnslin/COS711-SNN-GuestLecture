#!/usr/bin/env python3
# Tkinter + Matplotlib LIF demo
# - Drive buttons (Static / Sine / Off)
# - Sliders + text inputs (press Enter to apply)
# - Instantaneous postsynaptic spike (adds ΔVm that step)
# - Red threshold dots (crossing) + red overshoot segments (pre-reset Vm)
# - Adjustable R (MΩ) and τ (ms); C is computed implicitly: C = τ / R
# - Input current auto-scales Y; noise defaults to 0

import sys
import numpy as np
import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import LineCollection

# ---------- Model ----------
class LIFParams:
    # Defaults: tau_m = 20 ms, R = 100 MΩ => C = 0.2 nF = 200 pF, gL = 10 nS
    C   = 200e-12       # F (will be recomputed from R & tau on Apply)
    gL  = 10e-9         # S
    EL  = -70e-3        # V
    Vth = -50e-3        # V
    Vreset = -65e-3     # V
    tau_ref = 2e-3      # s

class DriveParams:
    # Static
    I_static = 180e-12  # A
    # Sine
    I_bias = 120e-12    # A
    I_amp  = 250e-12    # A
    f_hz   = 20.0       # Hz
    sigma  = 0.0        # A (noise = 0 by default)
    # Off
    sigma_off = 0.0     # A (noise = 0 by default)

# ---------- Simulation core ----------
class LIFSim:
    def __init__(self, dt=1e-4, window_s=2.0):
        self.dt = dt
        self.window_s = window_s
        self.win_steps = int(window_s / dt)
        self.params = LIFParams()
        self.drive  = DriveParams()
        self.mode = "STATIC"   # "STATIC" | "SINE" | "OFF"
        self.reset()

    def reset(self):
        self.t = 0.0
        self.V = self.params.EL
        self.ref_timer = 0.0
        self.V_buf = np.full(self.win_steps, self.params.EL, dtype=float)
        self.I_buf = np.zeros(self.win_steps, dtype=float)
        self.write_idx = -1
        self.spike_times = []        # raster (bottom plot)
        self.cross_times = []        # threshold crossing times (red dots)
        self.overshoot = []          # (t_end_step, V_over) before reset (red verticals)

    def lif_step(self, V, ref_timer, I_t):
        spiked = False
        if ref_timer > 0.0:
            ref_timer = max(0.0, ref_timer - self.dt)
            return V, ref_timer, spiked

        dV = (-self.params.gL * (V - self.params.EL) + I_t) / self.params.C
        V_new = V + self.dt * dV

        if V_new >= self.params.Vth:
            spiked = True
            # Linear crossing estimate within the step
            denom = (V_new - V) if V_new != V else 1e-12
            alpha = np.clip((self.params.Vth - V) / denom, 0.0, 1.0)
            t_cross = self.t + alpha * self.dt
            self.cross_times.append(t_cross)
            self.overshoot.append((self.t + self.dt, V_new))  # overshoot at end of step
            V_new = self.params.Vreset
            ref_timer = self.params.tau_ref

        return V_new, ref_timer, spiked

    def input_current(self, now):
        d = self.drive
        if self.mode == "STATIC":
            base = d.I_static; noise = d.sigma * np.random.randn() if d.sigma > 0 else 0.0
        elif self.mode == "SINE":
            base = d.I_bias + d.I_amp * np.sin(2 * np.pi * d.f_hz * now)
            noise = d.sigma * np.random.randn() if d.sigma > 0 else 0.0
        else:  # OFF
            base = 0.0; noise = d.sigma_off * np.random.randn() if d.sigma_off > 0 else 0.0
        return base + noise

    def inject_psp(self, delta_V_volts: float):
        """Instantaneous postsynaptic voltage jump (EPSP/IPSP)."""
        self.V += delta_V_volts

    def step(self, n_steps=1):
        for _ in range(n_steps):
            I_t = self.input_current(self.t)
            self.V, self.ref_timer, spiked = self.lif_step(self.V, self.ref_timer, I_t)
            self.t += self.dt
            if spiked:
                self.spike_times.append(self.t)
            self.write_idx = (self.write_idx + 1) % self.win_steps
            self.V_buf[self.write_idx] = self.V
            self.I_buf[self.write_idx] = I_t

            # prune events out of window
            t_min = self.t - self.window_s
            while self.spike_times and self.spike_times[0] < t_min - 0.01:
                self.spike_times.pop(0)
            while self.cross_times and self.cross_times[0] < t_min - 0.01:
                self.cross_times.pop(0)
            while self.overshoot and self.overshoot[0][0] < t_min - 0.01:
                self.overshoot.pop(0)

    def get_window_xy(self):
        x = np.linspace(self.t - self.window_s + self.dt, self.t, self.win_steps)

        def order(buf):
            if self.write_idx < 0 or self.write_idx == self.win_steps - 1:
                return buf.copy()
            return np.concatenate((buf[self.write_idx+1:], buf[:self.write_idx+1]))
        Vm = order(self.V_buf) * 1e3     # mV
        I  = order(self.I_buf) * 1e12    # pA

        # raster + markers
        t_min = self.t - self.window_s
        spike_segs = [((ts, 0.0), (ts, 1.0)) for ts in self.spike_times if ts >= t_min]
        dot_times = [ts for ts in self.cross_times if ts >= t_min]
        over_segs = [((ts, self.params.Vth*1e3), (ts, v_over*1e3))
                     for (ts, v_over) in self.overshoot if ts >= t_min]

        return x, Vm, I, spike_segs, dot_times, over_segs

# ---------- Tk App ----------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Interactive LIF — R & τ (C computed), modes, PSP, overshoot markers")
        self.geometry("1180x780")

        # Sim & runtime
        self.sim = LIFSim(dt=1e-4, window_s=2.0)
        self.running = False
        self.fps = 60
        self.sim_speed = tk.DoubleVar(value=2.0)

        # UI vars (amps in pA, freq Hz, noise pA, PSP mV, R MΩ, τ ms)
        d = self.sim.drive
        p = self.sim.params
        self.I_static = tk.DoubleVar(value=d.I_static * 1e12)
        self.I_bias   = tk.DoubleVar(value=d.I_bias   * 1e12)
        self.I_amp    = tk.DoubleVar(value=d.I_amp    * 1e12)
        self.f_hz     = tk.DoubleVar(value=d.f_hz)
        self.noise    = tk.DoubleVar(value=d.sigma * 1e12)      # 0
        self.noise_off= tk.DoubleVar(value=d.sigma_off * 1e12)  # 0
        self.psp_mV   = tk.DoubleVar(value=5.0)
        # Initialize R (MΩ) and τ (ms) from current C & gL
        R_init_MOhm = (1.0 / max(p.gL, 1e-15)) * 1e-6
        tau_init_ms = (p.C / max(p.gL, 1e-15)) * 1e3
        self.R_MOhm  = tk.DoubleVar(value=R_init_MOhm)
        self.tau_ms  = tk.DoubleVar(value=tau_init_ms)

        self._build_ui()
        self._after_id = None

    # ---------- UI builders ----------
    def _build_ui(self):
        root = ttk.Frame(self); root.pack(fill=tk.BOTH, expand=True)

        # Plots
        plot_frame = ttk.Frame(root); plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)
        fig = Figure(figsize=(7.6, 6.0), dpi=100, layout="constrained")
        self.ax_i = fig.add_subplot(3, 1, 1)
        self.ax_v = fig.add_subplot(3, 1, 2, sharex=self.ax_i)
        self.ax_s = fig.add_subplot(3, 1, 3, sharex=self.ax_i)

        # Current (top)
        (self.line_I,) = self.ax_i.plot([], [], lw=1.2)
        self.ax_i.set_ylabel("I (pA)")
        self.ax_i.axhline(0, lw=0.8, ls=":", alpha=0.6)

        # Vm (middle)
        (self.line_vm,) = self.ax_v.plot([], [], lw=1.5)
        self.ax_v.set_ylabel("V_m (mV)")
        self.ax_v.axhline(self.sim.params.Vth * 1e3, ls="--", lw=1, alpha=0.8, color="k")
        self.overshoot_lc = LineCollection([], linewidths=1.6, colors="red", alpha=0.9)
        self.ax_v.add_collection(self.overshoot_lc)
        self.thresh_dots = self.ax_v.scatter([], [], s=18, c="red", zorder=3)

        # Spike raster (bottom)
        self.spike_lc = LineCollection([], linewidths=1.5)
        self.ax_s.add_collection(self.spike_lc)
        self.ax_s.set_ylim(0, 1); self.ax_s.set_yticks([])
        self.ax_s.set_ylabel("Spikes"); self.ax_s.set_xlabel("Time (s)")

        # Limits
        w = self.sim.window_s
        self.ax_i.set_xlim(0.0, w); self.ax_v.set_xlim(0.0, w); self.ax_s.set_xlim(0.0, w)
        self._set_vm_ylim()

        self.canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Controls
        ctrl = ttk.Frame(root); ctrl.pack(side=tk.RIGHT, fill=tk.Y, padx=6, pady=6)

        # Drive mode buttons
        ttk.Label(ctrl, text="Drive Mode", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        bar = ttk.Frame(ctrl); bar.pack(fill=tk.X, pady=(2,6))
        self.mode_lbl = ttk.Label(ctrl, text="Mode: STATIC", foreground="#555")
        def set_mode(m): self.sim.mode = m; self.mode_lbl.config(text=f"Mode: {m}")
        ttk.Button(bar, text="Static", command=lambda: set_mode("STATIC")).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,4))
        ttk.Button(bar, text="Sine",   command=lambda: set_mode("SINE")).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)
        ttk.Button(bar, text="Off",    command=lambda: set_mode("OFF")).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4,0))
        self.mode_lbl.pack(anchor="w", pady=(0,8))

        # Instantaneous PSP
        ttk.Label(ctrl, text="Instantaneous PSP", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        psp_bar = ttk.Frame(ctrl); psp_bar.pack(fill=tk.X, pady=(2,6))
        ttk.Button(psp_bar, text="Inject Spike", command=self._do_inject_spike).pack(side=tk.LEFT, expand=True, fill=tk.X)
        self._slider_with_entry(ctrl, "PSP amplitude (mV)", self.psp_mV, -20.0, 20.0, 0.1, "{:.1f}", " mV")

        # Simulation speed
        ttk.Label(ctrl, text="Simulation", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(6,0))
        self._slider_with_entry(ctrl, "sim_speed (×)", self.sim_speed, 0.25, 8.0, 0.05, "{:.2f}", "")

        # Static mode
        ttk.Label(ctrl, text="Static Mode", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(6,0))
        self._slider_with_entry(ctrl, "I_static (pA)", self.I_static, -200, 800, 1.0, "{:.0f}", " pA")

        # Sine mode
        ttk.Label(ctrl, text="Sine Mode", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(6,0))
        self._slider_with_entry(ctrl, "I_bias (pA)",  self.I_bias,  -200, 800, 1.0, "{:.0f}", " pA")
        self._slider_with_entry(ctrl, "I_amp (pA)",   self.I_amp,      0, 1200, 1.0, "{:.0f}", " pA")
        self._slider_with_entry(ctrl, "f_hz (Hz)",    self.f_hz,        1,  150, 1.0, "{:.0f}", " Hz")
        self._slider_with_entry(ctrl, "noise σ (pA)", self.noise,       0,  100, 1.0, "{:.0f}", " pA")

        # Off mode
        ttk.Label(ctrl, text="Off Mode", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(6,0))
        self._slider_with_entry(ctrl, "noise_off σ (pA)", self.noise_off, 0, 50, 1.0, "{:.0f}", " pA")

        # R and τ inputs (C is derived)
        ttk.Label(ctrl, text="Membrane R and τ (C is computed)", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(8,0))
        rt = ttk.Frame(ctrl); rt.pack(fill=tk.X, pady=(2,6))
        ttk.Label(rt, text="R (MΩ)").grid(row=0, column=0, sticky="w")
        self.ent_R = ttk.Entry(rt, width=8, textvariable=self.R_MOhm); self.ent_R.grid(row=0, column=1, padx=4)
        ttk.Label(rt, text="τ (ms)").grid(row=0, column=2, sticky="w", padx=(10,0))
        self.ent_tau = ttk.Entry(rt, width=8, textvariable=self.tau_ms); self.ent_tau.grid(row=0, column=3, padx=4)
        ttk.Button(rt, text="Apply R, τ", command=self._apply_R_tau).grid(row=0, column=4, padx=(10,0))

        # Run buttons
        btns = ttk.Frame(ctrl); btns.pack(fill=tk.X, pady=(10,0))
        ttk.Button(btns, text="Start", command=self.start).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,4))
        ttk.Button(btns, text="Pause", command=self.pause).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)
        ttk.Button(btns, text="Reset", command=self.reset).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4,0))

        # Info
        self.info_lbl = ttk.Label(ctrl, text=self._info_text(), foreground="#555")
        self.info_lbl.pack(anchor="w", pady=8)

    def _slider_with_entry(self, parent, label, var, frm, to, step, fmt, unit):
        f = ttk.Frame(parent); f.pack(fill=tk.X, pady=3)
        ttk.Label(f, text=label).pack(anchor="w")
        s = ttk.Scale(f, from_=frm, to=to, orient=tk.HORIZONTAL, variable=var)
        s.pack(fill=tk.X, side=tk.LEFT, expand=True)
        e = ttk.Entry(f, width=8)
        e.insert(0, fmt.format(var.get()))
        e.pack(side=tk.LEFT, padx=6)
        unit_lbl = ttk.Label(f, text=unit, width=6); unit_lbl.pack(side=tk.LEFT)
        def on_slide(_):
            e.delete(0, tk.END); e.insert(0, fmt.format(var.get()))
        def on_enter(event):
            try:
                val = float(e.get()); var.set(val)
            except ValueError:
                e.delete(0, tk.END); e.insert(0, fmt.format(var.get()))
        s.configure(command=on_slide)
        e.bind("<Return>", on_enter)
        return s, e

    # ---------- control actions ----------
    def _apply_controls(self):
        d = self.sim.drive
        d.I_static = float(self.I_static.get()) * 1e-12
        d.I_bias   = float(self.I_bias.get())   * 1e-12
        d.I_amp    = float(self.I_amp.get())    * 1e-12
        d.f_hz     = float(self.f_hz.get())
        d.sigma    = float(self.noise.get())    * 1e-12
        d.sigma_off= float(self.noise_off.get())* 1e-12

    def _apply_R_tau(self):
        """Apply R (MΩ) and τ (ms); compute gL=1/R and C=τ/R."""
        try:
            R_M = float(self.R_MOhm.get())
            tau_ms = float(self.tau_ms.get())
        except ValueError:
            return
        R_ohm = max(R_M, 0.0) * 1e6
        tau_s = max(tau_ms, 0.0) * 1e-3
        gL = 0.0 if R_ohm == 0 else 1.0 / R_ohm
        C = 0.0 if R_ohm == 0 else tau_s / R_ohm
        self.sim.params.gL = gL
        self.sim.params.C  = C
        self._set_vm_ylim()
        self.info_lbl.config(text=self._info_text())

    def _set_vm_ylim(self):
        p = self.sim.params
        lo = (p.EL - 0.015) * 1e3
        hi = (p.Vth + 0.010) * 1e3
        if hi - lo < 5.0:
            hi = lo + 5.0
        self.ax_v.set_ylim(lo, hi)

    def _info_text(self):
        p = self.sim.params
        tau_m_ms = (p.C / max(p.gL, 1e-15)) * 1e3
        R_M = (1.0 / max(p.gL, 1e-15)) * 1e-6  # Ω -> MΩ
        return f"τm≈{tau_m_ms:.1f} ms | R≈{R_M:.1f} MΩ | C≈{p.C*1e12:.0f} pF | Vth={p.Vth*1e3:.0f} mV"

    def _do_inject_spike(self):
        delta_V = float(self.psp_mV.get()) * 1e-3
        self.sim.inject_psp(delta_V)

    def start(self):
        self._apply_controls()
        if not self.running:
            self.running = True
            self._tick()

    def pause(self):
        self.running = False
        if hasattr(self, "_after_id") and self._after_id:
            self.after_cancel(self._after_id); self._after_id = None

    def reset(self):
        self.pause()
        self.sim.reset()
        self._redraw(force_rescale=True)

    # ---------- loop ----------
    def _tick(self):
        if not self.running: return
        self._apply_controls()
        steps_per_frame = max(1, int(float(self.sim_speed.get()) / (self.sim.dt * self.fps)))
        self.sim.step(steps_per_frame)
        self._redraw(force_rescale=True)
        self._after_id = self.after(int(1000 / self.fps), self._tick)

    def _redraw(self, force_rescale=False):
        x, Vm, I, spike_segs, dot_times, over_segs = self.sim.get_window_xy()

        # Current (auto-scale)
        self.line_I.set_data(x, I)
        if len(I) > 0 and force_rescale:
            i_min, i_max = float(np.min(I)), float(np.max(I))
            if np.isfinite(i_min) and np.isfinite(i_max):
                if abs(i_max - i_min) < 1e-6:
                    center = 0.5 * (i_min + i_max); half = max(10.0, 0.1 * max(1.0, abs(center)))
                    self.ax_i.set_ylim(center - half, center + half)
                else:
                    pad = max(10.0, 0.1 * (i_max - i_min))
                    self.ax_i.set_ylim(i_min - pad, i_max + pad)

        # Vm + markers
        self.line_vm.set_data(x, Vm)
        self.overshoot_lc.set_segments(over_segs)
        if dot_times:
            y = [self.sim.params.Vth * 1e3] * len(dot_times)
            self.thresh_dots.set_offsets(np.c_[dot_times, y])
        else:
            self.thresh_dots.set_offsets(np.empty((0, 2)))

        # Raster
        self.spike_lc.set_segments(spike_segs)

        # Scroll
        t = self.sim.t; w = self.sim.window_s
        self.ax_i.set_xlim(t - w, t)
        self.ax_v.set_xlim(t - w, t)
        self.ax_s.set_xlim(t - w, t)

        self.canvas.draw_idle()

# ---------- main ----------
if __name__ == "__main__":
    try:
        App().mainloop()
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        raise
