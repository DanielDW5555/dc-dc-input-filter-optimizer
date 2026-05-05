import one_stage_input_filter as input_filter
import random
import math
import os
import subprocess

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ─────────────────────────────────────────────────────────────────────────────
# Component parasitics
# ─────────────────────────────────────────────────────────────────────────────

parasitics = {
    "L1_r":    50e-3,   # Inductor DCR — typical for small SMD power inductor
    "C1_r":    5e-3,    # C1 ESR — ceramic X5R/X7R at SMD size
    "C1_l":    1e-9,    # C1 parasitic inductance
    "Cdamp_r": 50e-3,   # Cdamp ESR — electrolytic or film cap
    "Cdamp_l": 5e-9,    # Cdamp parasitic inductance
}

# ─────────────────────────────────────────────────────────────────────────────
# Operating point — 5 V / 1 A point-of-load converter
# ─────────────────────────────────────────────────────────────────────────────

circuit_parameters = {
    "Vin":  12,      # V  — 12 V system bus (common in consumer electronics)
    "eff":  0.88,    # —    typical synchronous buck efficiency at 1 A
    "Pout": 5,       # W  = 5 V × 1 A
    "Fsw":  500e3,   # Hz — 500 kHz, typical for compact consumer POL
}

# ─────────────────────────────────────────────────────────────────────────────
# Derived quantities
# ─────────────────────────────────────────────────────────────────────────────

# Converter negative incremental input impedance (Ω)
Zin_conv = circuit_parameters['Vin'] ** 2 * circuit_parameters['eff'] / circuit_parameters['Pout']

# ─────────────────────────────────────────────────────────────────────────────
# Specifications — consumer electronics POL
# ─────────────────────────────────────────────────────────────────────────────

specs = {
    # Maximum resonance peaking (linear magnitude).
    # 3 dB (×1.41) is the well-damped target for consumer POL designs.
    "att_peak_limit":  input_filter.dB2mag(3),

    # Minimum filter input impedance (Ω).
    # Prevents the filter from presenting a near-short to the 12 V bus.
    "Zin_min":         0.5,

    # Middlebrook stability criterion: filter Zout must stay ≥ 10 dB below
    # the converter's negative incremental input impedance.
    "Zout_max":        Zin_conv / input_filter.dB2mag(10),

    # Maximum |H(Fsw)| — attenuation at the switching frequency.
    # −40 dB is a common design target for conducted switching noise.
    "att_at_Fsw_max":  input_filter.dB2mag(-40),
}

# ─────────────────────────────────────────────────────────────────────────────
# Search bounds  [L1, C1, Rdamp, Cdamp]
# ─────────────────────────────────────────────────────────────────────────────

bounds = [
    (1e-6,   100e-6),   # L1     1 µH  – 100 µH   (SMD power inductor)
    (10e-9,   10e-6),   # C1     10 nF –  10 µF   (ceramic MLCC)
    (0.1,    20.0),     # Rdamp  0.1 Ω –  20 Ω
    (100e-9, 100e-6),   # Cdamp  100 nF – 100 µF   (ceramic or film)
]


# ─────────────────────────────────────────────────────────────────────────────
# Cost function
# ─────────────────────────────────────────────────────────────────────────────

def violation(value, limit):
    """
    Quadratic penalty — 0 when satisfied, grows as (excess ratio − 1)².

        value ≤ limit  →  0        (no penalty)
        value > limit  →  (value/limit − 1)²
    """
    ratio = value / limit
    return max(0.0, ratio - 1.0) ** 2


def cost_function(params):
    try:
        filt = input_filter.filter(*params, parasitics, circuit_parameters)
        filt.update()
        max_att, min_Zin, max_Zout, att_Fsw = filt.get_outputs()

        # 1. Resonance damping — peak magnitude must stay below 3 dB
        J_att     = violation(max_att,  specs["att_peak_limit"])

        # 2. Source loading — Zin must not fall below Zin_min
        J_Zin     = violation(specs["Zin_min"], min_Zin)   # inverted: fires when Zin < min

        # 3. Middlebrook stability criterion
        J_Zout    = violation(max_Zout, specs["Zout_max"])

        # 4. EMI attenuation at Fsw — must reach −40 dB
        J_att_Fsw = violation(att_Fsw,  specs["att_at_Fsw_max"])

        # Small-component tiebreaker (1 % weight — keeps design compact once
        # all constraints are satisfied)
        size_terms = [params[k] / bounds[k][1] for k in range(len(bounds))]
        J_size = 0.01 * sum(size_terms) / len(bounds)

        return J_att + J_Zin + J_Zout + J_att_Fsw + J_size

    except Exception:
        return 1e9


# ─────────────────────────────────────────────────────────────────────────────
# Differential Evolution
# ─────────────────────────────────────────────────────────────────────────────

def init_population(bounds, NP):
    n = len(bounds)
    pop = []
    for _ in range(NP):
        x = [bounds[k][0] + random.random() * (bounds[k][1] - bounds[k][0])
             for k in range(n)]
        pop.append({'x': x, 'cost': cost_function(x)})
    return pop


def run_generation(pop, bounds, F, CR):
    n  = len(bounds)
    NP = len(pop)
    for i in range(NP):
        pool = [j for j in range(NP) if j != i]
        r1, r2, r3 = random.sample(pool, 3)

        mutant = [
            pop[r1]['x'][k] + F * (pop[r2]['x'][k] - pop[r3]['x'][k])
            for k in range(n)
        ]
        mutant = [
            max(bounds[k][0], min(bounds[k][1], mutant[k]))
            for k in range(n)
        ]

        jrand = random.randint(0, n - 1)
        trial = [
            mutant[k] if (random.random() < CR or k == jrand) else pop[i]['x'][k]
            for k in range(n)
        ]

        trial_cost = cost_function(trial)
        if trial_cost < pop[i]['cost']:
            pop[i] = {'x': trial, 'cost': trial_cost}

    return pop


def differential_evolution(bounds, NP=60, F=0.7, CR=0.7, max_gen=500):
    pop  = init_population(bounds, NP)
    best = min(pop, key=lambda p: p['cost'])
    print(f"Start: best cost = {best['cost']:.4f}")

    for gen in range(max_gen):
        pop  = run_generation(pop, bounds, F, CR)
        best = min(pop, key=lambda p: p['cost'])

        if (gen + 1) % 10 == 0:
            print(f"Gen {gen + 1:4d}: best cost = {best['cost']:.6f}")

        costs = [p['cost'] for p in pop]
        if max(costs) - min(costs) < 1e-8:
            print(f"Converged at generation {gen + 1}")
            break

    return best['x'], best['cost']


# ─────────────────────────────────────────────────────────────────────────────
# Options
# ─────────────────────────────────────────────────────────────────────────────

SAVE_PLOTS  = True
RUN_NGSPICE = True


# ─────────────────────────────────────────────────────────────────────────────
# NGSpice helpers
# ─────────────────────────────────────────────────────────────────────────────

def _spice_filter_body(params, parasitics):
    """SPICE component lines for the one-stage filter (n_in → n_out, gnd = 0)."""
    L1, C1, Rdamp, Cdamp = params
    return f"""\
L1      n_in   n_L1a    {L1}
R_L1    n_L1a  n_out    {parasitics["L1_r"]}
L_C1    n_out  n_C1a    {parasitics["C1_l"]}
R_C1    n_C1a  n_C1b    {parasitics["C1_r"]}
C1      n_C1b  0        {C1}
Rdamp   n_out  n_damp   {Rdamp}
L_Cd    n_damp n_Cda    {parasitics["Cdamp_l"]}
R_Cd    n_Cda  n_Cdb    {parasitics["Cdamp_r"]}
Cdamp   n_Cdb  0        {Cdamp}"""


def _run_ngspice(netlist_str, filename):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    with open(path, 'w') as f:
        f.write(netlist_str)
    try:
        result = subprocess.run(
            ["ngspice", "-b", path],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout + result.stderr
    except FileNotFoundError:
        print("  ngspice not found — install with: sudo apt install ngspice")
        return None
    except subprocess.TimeoutExpired:
        print("  ngspice timed out")
        return None


def _parse_ngspice(text):
    if text is None:
        return []
    rows, in_data = [], False
    for line in text.splitlines():
        s = line.strip()
        if "Index" in s and "frequency" in s.lower():
            in_data = True
            continue
        if not in_data:
            continue
        parts = s.split()
        if len(parts) >= 3:
            try:
                rows.append((float(parts[1]), abs(float(parts[2]))))
            except ValueError:
                continue
    return rows


def run_ngspice_sims(params, parasitics, circuit_parameters):
    body  = _spice_filter_body(params, parasitics)
    RL    = -(circuit_parameters['Vin'] ** 2 * circuit_parameters['eff']
              / circuit_parameters['Pout'])
    Fsw   = circuit_parameters['Fsw']
    f_lo  = max(500, Fsw / 200)
    f_hi  = Fsw * 10

    netlists = {
        "attenuation": f"""\
* Voltage attenuation |Vout/Vin| with converter load RL
V1 n_in 0 AC 1
{body}
Rload n_out 0 {RL}
.ac dec 200 {f_lo} {f_hi}
.print ac vm(n_out)
.end
""",
        "input_impedance": f"""\
* Input impedance |Zin| — 1 A injected, RL load at output
I1 0 n_in AC 1
{body}
Rload n_out 0 {RL}
.ac dec 200 {f_lo} {f_hi}
.print ac vm(n_in)
.end
""",
        "output_impedance": f"""\
* Output impedance |Zout| — input shorted, 1 A injected at output
Vshort n_in 0 AC 0
{body}
I1 0 n_out AC 1
.ac dec 200 {f_lo} {f_hi}
.print ac vm(n_out)
.end
""",
    }

    results = {}
    for name, netlist in netlists.items():
        print(f"  Running ngspice: {name}...")
        raw = _run_ngspice(netlist, f"_opt_{name}.sp")
        results[name] = _parse_ngspice(raw)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def save_filter_plots(filt, params, specs, circuit_parameters,
                      ngspice_data=None,
                      filename="optimal_filter_response.png"):
    if not HAS_MATPLOTLIB:
        print("matplotlib not available — install with: pip install matplotlib")
        return

    def to_dB(values):
        out = []
        for v in values:
            try:
                out.append(20 * math.log10(abs(v)))
            except (ValueError, ZeroDivisionError):
                out.append(-200.0)
        return out

    Fsw         = circuit_parameters["Fsw"]
    py_freqs    = filt.frequency_values
    py_att_dB   = to_dB(filt.attenuation_response)
    py_zin      = filt.input_impedance_response
    py_zout     = filt.output_impedance_response

    att_peak_dB     = 20 * math.log10(specs["att_peak_limit"])
    att_fsw_dB      = 20 * math.log10(specs["att_at_Fsw_max"])
    zin_min         = specs["Zin_min"]
    zout_max        = specs["Zout_max"]

    L1, C1, Rdamp, Cdamp = params
    title_str = (
        f"Optimal Filter (5 V / 1 A POL, Fsw = {Fsw/1e3:.0f} kHz) — "
        f"L1 = {L1*1e6:.1f} µH   C1 = {C1*1e6:.2f} µF   "
        f"Rdamp = {Rdamp:.2f} Ω   Cdamp = {Cdamp*1e6:.1f} µF"
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(title_str, fontsize=10, fontweight='bold')

    PY   = dict(color='#1f77b4', lw=2.0, ls='-',  label='Python model', zorder=3)
    SP   = dict(color='#ff7f0e', lw=1.5, ls='--', label='ngspice',       zorder=2)
    PASS = '#d4edda'
    FAIL = '#f8d7da'

    def add_fsw(ax):
        ax.axvline(Fsw, color='#555', ls=':', lw=1.0, label=f'Fsw = {Fsw/1e3:.0f} kHz')

    def fmt(ax):
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: (f'{x/1e6:.1f}M' if x >= 1e6
                          else f'{x/1e3:.0f}k' if x >= 1000
                          else f'{x:.0f}')))
        ax.set_xlim([py_freqs[0], py_freqs[-1]])
        ax.set_xlabel("Frequency (Hz)")
        ax.grid(True, which='both', alpha=0.35)

    # ── Attenuation ──────────────────────────────────────────────────────
    ax = axes[0]
    ax.axhspan(-200, att_peak_dB, alpha=0.12, color=PASS, zorder=0)
    ax.axhspan(att_peak_dB, 20,   alpha=0.12, color=FAIL, zorder=0)
    ax.axhline(att_peak_dB, color='#cc0000', ls='--', lw=1.2,
               label=f'Peak limit {att_peak_dB:.0f} dB')
    ax.axhline(att_fsw_dB, color='#2ca02c', ls='--', lw=1.2,
               label=f'EMI target {att_fsw_dB:.0f} dB @ Fsw')
    ax.semilogx(py_freqs, py_att_dB, **PY)
    if ngspice_data and ngspice_data.get("attenuation"):
        sp_f, sp_v = zip(*ngspice_data["attenuation"])
        ax.semilogx(sp_f, to_dB(list(sp_v)), **SP)
    add_fsw(ax)
    ax.set_title("Voltage Attenuation  |H_V|")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_ylim([-80, 10])
    ax.legend(fontsize=8)
    fmt(ax)

    # ── Input Impedance ───────────────────────────────────────────────────
    ax = axes[1]
    ax.axhspan(0,       zin_min, alpha=0.12, color=FAIL, zorder=0)
    ax.axhspan(zin_min, 1e6,     alpha=0.12, color=PASS, zorder=0)
    ax.axhline(zin_min, color='#cc0000', ls='--', lw=1.2,
               label=f'Min {zin_min:.1f} Ω')
    ax.loglog(py_freqs, py_zin, **PY)
    if ngspice_data and ngspice_data.get("input_impedance"):
        sp_f, sp_v = zip(*ngspice_data["input_impedance"])
        ax.loglog(sp_f, list(sp_v), **SP)
    add_fsw(ax)
    ax.set_title("Input Impedance  |Z_in|")
    ax.set_ylabel("Impedance (Ω)")
    ax.legend(fontsize=8)
    fmt(ax)

    # ── Output Impedance (Middlebrook) ────────────────────────────────────
    ax = axes[2]
    ax.axhspan(0,        zout_max, alpha=0.12, color=PASS, zorder=0)
    ax.axhspan(zout_max, 1e6,      alpha=0.12, color=FAIL, zorder=0)
    ax.axhline(zout_max, color='#cc0000', ls='--', lw=1.2,
               label=f'Middlebrook limit {zout_max:.2f} Ω')
    ax.axhline(Zin_conv, color='#888', ls=':', lw=1.0,
               label=f'|Z_in_conv| {Zin_conv:.1f} Ω')
    ax.loglog(py_freqs, py_zout, **PY)
    if ngspice_data and ngspice_data.get("output_impedance"):
        sp_f, sp_v = zip(*ngspice_data["output_impedance"])
        ax.loglog(sp_f, list(sp_v), **SP)
    add_fsw(ax)
    ax.set_title("Output Impedance  |Z_out|  (Middlebrook)")
    ax.set_ylabel("Impedance (Ω)")
    ax.legend(fontsize=8)
    fmt(ax)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    x_opt, cost_opt = differential_evolution(bounds, NP=60, F=0.7, CR=0.7, max_gen=500)

    L1, C1, Rdamp, Cdamp = x_opt
    filt_opt = input_filter.filter(*x_opt, parasitics, circuit_parameters)
    filt_opt.update()
    _, _, _, att_Fsw = filt_opt.get_outputs()

    print("\n── Optimal design ─────────────────────────────────")
    print(f"  L1    = {L1*1e6:.2f} µH")
    print(f"  C1    = {C1*1e6:.3f} µF")
    print(f"  Rdamp = {Rdamp:.2f} Ω")
    print(f"  Cdamp = {Cdamp*1e6:.2f} µF")
    print(f"  Final cost = {cost_opt:.6f}  (0.0 = all constraints satisfied)")
    print(f"\n── Constraint check ────────────────────────────────")
    print(f"  Attenuation at Fsw : {input_filter.mag2dB(att_Fsw):.1f} dB"
          f"  (target ≤ {input_filter.mag2dB(specs['att_at_Fsw_max']):.0f} dB)")
    print(f"  Zin_conv           : {Zin_conv:.2f} Ω")
    print(f"  Middlebrook limit  : {specs['Zout_max']:.3f} Ω")

    input_filter.save_plot(filt_opt.frequency_values,
                           filt_opt.attenuation_response,
                           "optimal-attenuation")
    print("\nSaved optimal-attenuation.csv")

    ngspice_data = None
    if RUN_NGSPICE:
        print("\nRunning ngspice validation...")
        ngspice_data = run_ngspice_sims(x_opt, parasitics, circuit_parameters)

    if SAVE_PLOTS:
        print("\nSaving plots...")
        save_filter_plots(filt_opt, x_opt, specs, circuit_parameters,
                          ngspice_data=ngspice_data)
