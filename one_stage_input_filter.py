import math


def s(freq):
    return 1j * 2 * math.pi * freq

def capacitor(freq, capacitance):
    return 1 / (s(freq) * capacitance)

def inductor(freq, inductance):
    return s(freq) * inductance

def ind(freq, inductance, series_resistance=0):
    return inductor(freq, inductance) + series_resistance

def cap(freq, capacitance, series_resistance, series_inductance):
    return capacitor(freq, capacitance) + series_resistance + inductor(freq, series_inductance)

def parallel(values):
    Zeq = 0
    for value in values:
        Zeq += 1 / value
    return 1 / Zeq

def voltage_divider(a, b):
    return b / (a + b)

def log_space(start, stop, num=50, base=10):
    start_log = math.log(start, base)
    stop_log  = math.log(stop,  base)
    step = (stop_log - start_log) / (num - 1)
    return [base ** (start_log + i * step) for i in range(num)]

def mag2dB(x):
    try:
        return 20 * math.log10(x)
    except TypeError:
        return [20 * math.log10(v) for v in x]

def dB2mag(x):
    return pow(10, x / 20)

def save_plot(xaxis, yaxis, file_name="plot",
              xaxis_name="xaxis", yaxis_name="yaxis", delimitation=','):
    try:
        f = open(file_name + ".csv", 'x')
        f.close()
    except FileExistsError:
        print("File already exists — overwriting")
    with open(file_name + ".csv", "w") as f:
        f.write(xaxis_name + delimitation + yaxis_name + "\n")
        for x, y in zip(xaxis, yaxis):
            f.write(f"{x}{delimitation}{y}\n")


class filter():
    """
    Single-stage LC input filter with parallel RC damping network.

    Topology  (source → converter):

        n_in ──L1(+RL1)──┬── n_out
                         ├── C1 ──────────── gnd
                         └── Rdamp──Cdamp─── gnd

    Parameters
    ----------
    L1, C1, Rdamp, Cdamp : float   Component values in SI units
    parasitics_params     : dict   Keys: L1_r, C1_r, C1_l, Cdamp_r, Cdamp_l
    circuit_params        : dict   Keys: Vin, eff, Pout, Fsw
    """

    def __init__(self, L1, C1, Rdamp, Cdamp, parasitics_params, circuit_params):
        self.L1     = L1
        self.L1_r   = parasitics_params["L1_r"]
        self.C1     = C1
        self.C1_r   = parasitics_params["C1_r"]
        self.C1_l   = parasitics_params["C1_l"]
        self.Rdamp  = Rdamp
        self.Cdamp  = Cdamp
        self.Cdamp_r = parasitics_params["Cdamp_r"]
        self.Cdamp_l = parasitics_params["Cdamp_l"]

        self.Vin  = circuit_params["Vin"]
        self.eff  = circuit_params["eff"]
        self.Pout = circuit_params["Pout"]
        self.Fsw  = circuit_params["Fsw"]

        # Sweep spans two decades below resonance to one decade above Fsw
        self.start_frequency = max(500, self.Fsw / 200)
        self.end_frequency   = self.Fsw * 10
        self.total_points    = 128

        self.frequency_values          = None
        self.attenuation_response      = None
        self.input_impedance_response  = None
        self.output_impedance_response = None
        self.max_attenuation           = None
        self.min_input_impedance       = None
        self.max_output_impedance      = None
        self.att_at_Fsw                = None   # |H(Fsw)| — key EMI metric

    # ------------------------------------------------------------------
    def _impedances(self, freq):
        Z_L1 = ind(freq, self.L1, self.L1_r)
        Z_C1 = cap(freq, self.C1,   self.C1_r,   self.C1_l)
        Z_d  = self.Rdamp + cap(freq, self.Cdamp, self.Cdamp_r, self.Cdamp_l)
        return Z_L1, Z_C1, Z_d

    def attenuation(self, freq):
        """Voltage transfer |Vout/Vin| with converter (negative incremental) load."""
        Z_L1, Z_C1, Z_d = self._impedances(freq)
        RL = -(self.Vin ** 2) * self.eff / self.Pout
        Z_load = parallel([Z_C1, Z_d, RL])
        return Z_load / (Z_L1 + Z_load)

    def input_impedance(self, freq):
        """Filter input impedance seen from the source (converter load connected)."""
        Z_L1, Z_C1, Z_d = self._impedances(freq)
        RL = -(self.Vin ** 2) * self.eff / self.Pout
        return Z_L1 + parallel([Z_C1, Z_d, RL])

    def output_impedance(self, freq):
        """Output (Thevenin) impedance with source shorted."""
        Z_L1, Z_C1, Z_d = self._impedances(freq)
        return parallel([Z_L1, Z_C1, Z_d])

    # ------------------------------------------------------------------
    def _sweep(self, method, frequencies):
        return [abs(method(f)) for f in frequencies]

    def update(self):
        self.frequency_values = log_space(
            self.start_frequency, self.end_frequency, self.total_points)
        self.attenuation_response      = self._sweep(self.attenuation,      self.frequency_values)
        self.input_impedance_response  = self._sweep(self.input_impedance,  self.frequency_values)
        self.output_impedance_response = self._sweep(self.output_impedance, self.frequency_values)
        self.max_attenuation      = max(self.attenuation_response)
        self.min_input_impedance  = min(self.input_impedance_response)
        self.max_output_impedance = max(self.output_impedance_response)
        self.att_at_Fsw           = abs(self.attenuation(self.Fsw))

    def get_outputs(self):
        return (self.max_attenuation, self.min_input_impedance,
                self.max_output_impedance, self.att_at_Fsw)
