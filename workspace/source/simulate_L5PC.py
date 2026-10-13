import os
import secrets
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
SHAI_ROOT = REPO_ROOT / "ShaiEtAl2015"

# Fixed protocol used for the dataset generation.
SAMPLE_INDEX = 0
BASE_SEED = None
NUM_SYNAPSES = 500
NUM_EXC = 250
NUM_INH = 250
DURATION_MS = 6000.0
DT_MS = 0.1
SAMPLE_INTERVAL_MS = 1.0
INPUT_RATE_HZ = 50.0
EXC_WEIGHT = 1.88e-4
INH_WEIGHT = 8.0e-5
EXC_TAU1_MS = 0.5
EXC_TAU2_MS = 5.0
INH_TAU1_MS = 1.0
INH_TAU2_MS = 10.0
INH_REVERSAL_MV = -80.0
V_INIT_MV = -75.0
SPIKE_THRESHOLD_MV = -20.0
BIOPHYS_FILE = "L5PCbiophys5.hoc"
MORPHOLOGY_FILE = "cell1.asc"

OUTPUT_DIR = REPO_ROOT / "figures"
OUTPUT_PNG = OUTPUT_DIR / "L5PC_simulation_voltage.png"


def ensure_mechanisms(root: Path):
    mech_lib = root / "x86_64" / "libnrnmech.so"
    if mech_lib.exists():
        return

    nrnivmodl = Path(sys.executable).with_name("nrnivmodl")
    if not nrnivmodl.exists():
        raise FileNotFoundError(f"Missing nrnivmodl at {nrnivmodl}")
    subprocess.run([str(nrnivmodl), "mod"], cwd=root, check=True)


def vec_to_np(vec):
    if hasattr(vec, "as_numpy"):
        return np.array(vec.as_numpy(), dtype=float, copy=True)
    return np.array(vec.to_python(), dtype=float)


def choose_sites_weighted(sections, count, rng):
    lengths = np.array([float(sec.L) for sec in sections], dtype=np.float64)
    probabilities = lengths / lengths.sum()
    indices = rng.choice(len(sections), size=count, replace=True, p=probabilities)
    return [(sections[i], float(rng.uniform(0.1, 0.9))) for i in indices]


def expected_event_count(total_ms, rate_hz):
    expected = total_ms * rate_hz / 1000.0
    return int(max(64, np.ceil(expected * 3.0)))


def build_network():
    from neuron import h, load_mechanisms

    load_mechanisms(str(SHAI_ROOT))

    h.load_file("stdrun.hoc")
    h.load_file("import3d.hoc")
    h.load_file(str(SHAI_ROOT / "models" / BIOPHYS_FILE))
    h.load_file(str(SHAI_ROOT / "models" / "L5PCtemplate.hoc"))

    cell = h.L5PCtemplate(str(SHAI_ROOT / "morphologies" / MORPHOLOGY_FILE))
    dend_sections = [sec for sec in cell.dend] + [sec for sec in cell.apic]

    # Keep synapse locations fixed, but allow input Poisson sequences to vary across runs.
    site_rng = np.random.default_rng(20260425)
    sites = choose_sites_weighted(dend_sections, NUM_SYNAPSES, site_rng)

    interval_ms = 1000.0 / INPUT_RATE_HZ
    event_count = expected_event_count(DURATION_MS, INPUT_RATE_HZ)

    channel_type = []
    channel_section = []
    channel_location = []
    synapses = []
    netstims = []
    netcons = []

    for channel_idx, (section, location) in enumerate(sites):
        syn = h.Exp2Syn(section(location))
        is_exc = channel_idx < NUM_EXC
        if is_exc:
            syn.e = 0.0
            syn.tau1 = EXC_TAU1_MS
            syn.tau2 = EXC_TAU2_MS
            weight = EXC_WEIGHT
            channel_type.append("exc")
        else:
            syn.e = INH_REVERSAL_MV
            syn.tau1 = INH_TAU1_MS
            syn.tau2 = INH_TAU2_MS
            weight = INH_WEIGHT
            channel_type.append("inh")

        netstim = h.NetStim()
        netstim.interval = interval_ms
        netstim.number = event_count
        netstim.noise = 1.0
        netstim.start = 0.0
        netstim.noiseFromRandom123(BASE_SEED, SAMPLE_INDEX + 1, channel_idx + 1)

        netcon = h.NetCon(netstim, syn)
        netcon.delay = 0.0
        netcon.weight[0] = weight

        synapses.append(syn)
        channel_section.append(section.name())
        channel_location.append(float(location))
        netstims.append(netstim)
        netcons.append(netcon)

    return {
        "h": h,
        "cell": cell,
        "channel_type": np.array(channel_type),
        "channel_section": np.array(channel_section),
        "channel_location": np.array(channel_location, dtype=np.float32),
        "synapses": synapses,
        "netstims": netstims,
        "netcons": netcons,
    }


def run_simulation(state):
    h = state["h"]
    cell = state["cell"]

    event_recorders = []
    for netcon in state["netcons"]:
        event_vec = h.Vector()
        netcon.record(event_vec)
        event_recorders.append(event_vec)

    voltage_vec = h.Vector().record(cell.soma[0](0.5)._ref_v)
    spike_vec = h.Vector()
    spike_nc = h.NetCon(cell.soma[0](0.5)._ref_v, None, sec=cell.soma[0])
    spike_nc.threshold = SPIKE_THRESHOLD_MV
    spike_nc.record(spike_vec)

    h.dt = DT_MS
    h.steps_per_ms = int(round(1.0 / DT_MS))
    h.finitialize(V_INIT_MV)
    h.fcurrent()
    h.frecord_init()
    h.continuerun(DURATION_MS)

    fine_voltage = vec_to_np(voltage_vec)
    fine_start = 0
    fine_stop = int(round(DURATION_MS / DT_MS)) + 1
    stride = int(round(SAMPLE_INTERVAL_MS / DT_MS))
    voltage_1ms = fine_voltage[fine_start:fine_stop:stride].astype(np.float32)

    spike_times = vec_to_np(spike_vec)
    spike_times = spike_times[(spike_times >= 0.0) & (spike_times <= DURATION_MS)]
    spike_times = np.round(spike_times, decimals=3)

    num_bins = int(round(DURATION_MS / SAMPLE_INTERVAL_MS)) + 1
    input_raster = np.zeros((NUM_SYNAPSES, num_bins), dtype=np.uint8)
    for channel_idx, event_vec in enumerate(event_recorders):
        event_times = vec_to_np(event_vec)
        event_times = event_times[(event_times >= 0.0) & (event_times <= DURATION_MS)]
        if event_times.size == 0:
            continue
        bins = np.round(event_times / SAMPLE_INTERVAL_MS).astype(int)
        bins = bins[(bins >= 0) & (bins < num_bins)]
        if bins.size:
            input_raster[channel_idx, np.unique(bins)] = 1

    return voltage_1ms, spike_times, input_raster


def save_outputs(voltage_1ms, spike_times):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    time_axis_sec = np.arange(voltage_1ms.shape[0], dtype=np.float32) / 1000.0
    plt.figure(figsize=(12, 4))
    plt.plot(time_axis_sec, voltage_1ms, color="black", linewidth=1.1)
    plt.xlabel("Time (s)")
    plt.ylabel("Somatic Voltage (mV)")
    plt.title(f"L5PC fixed 500-syn standalone run | sample={SAMPLE_INDEX} | spikes={len(spike_times)}")
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=200)
    plt.close()


def main():
    global BASE_SEED
    if BASE_SEED is None:
        BASE_SEED = secrets.randbelow(2**31 - 1) + 1
    ensure_mechanisms(SHAI_ROOT)
    state = build_network()
    voltage_1ms, spike_times, input_raster = run_simulation(state)
    save_outputs(voltage_1ms, spike_times)

    print(f"Poisson base seed: {BASE_SEED}")
    print(f"Saved voltage plot to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
