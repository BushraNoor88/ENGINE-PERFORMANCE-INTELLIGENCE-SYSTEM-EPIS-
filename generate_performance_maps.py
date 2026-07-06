"""
Generates classic gas-turbine performance maps:
  - Thrust vs Mach number, for several altitudes
  - TSFC vs Mach number, for several altitudes
  - Thrust lapse with altitude at fixed Mach
  - Efficiency sensitivity: how much does 1% compressor efficiency loss
    cost you in thrust and fuel burn?
"""

import numpy as np
import matplotlib.pyplot as plt

from data.reference_engine import build_reference_engine, TRUE_EFFICIENCIES
from engine.engine import FlightCondition

plt.rcParams.update({"font.size": 10, "figure.dpi": 130})

engine = build_reference_engine()

# ---------------------------------------------------------------
# Plot 1 & 2: Thrust and TSFC vs Mach, at several altitudes
# ---------------------------------------------------------------
altitudes = [0, 5000, 11000]
mach_range = np.linspace(0.05, 0.9, 25)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

for alt in altitudes:
    thrusts, tsfcs = [], []
    for m in mach_range:
        result = engine.run_cycle(FlightCondition(altitude=alt, mach=m))
        thrusts.append(result.thrust / 1000.0)          # kN
        tsfcs.append(result.tsfc * 3600 * 9.81)          # kg/(kgf*hr), common industry unit

    axes[0].plot(mach_range, thrusts, marker="o", markersize=3, label=f"{alt/1000:.0f} km")
    axes[1].plot(mach_range, tsfcs, marker="o", markersize=3, label=f"{alt/1000:.0f} km")

axes[0].set_xlabel("Mach number")
axes[0].set_ylabel("Net thrust (kN)")
axes[0].set_title("Thrust vs Mach Number")
axes[0].legend(title="Altitude")
axes[0].grid(alpha=0.3)

axes[1].set_xlabel("Mach number")
axes[1].set_ylabel("TSFC (kg / kgf·hr)")
axes[1].set_title("Specific Fuel Consumption vs Mach Number")
axes[1].legend(title="Altitude")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("thrust_tsfc_performance_map.png", bbox_inches="tight")
plt.close()
print("Saved thrust_tsfc_performance_map.png")

# ---------------------------------------------------------------
# Plot 3: Thrust lapse with altitude at fixed Mach 0.7
# ---------------------------------------------------------------
altitude_range = np.linspace(0, 13000, 25)
thrusts_alt = []
for alt in altitude_range:
    result = engine.run_cycle(FlightCondition(altitude=alt, mach=0.7))
    thrusts_alt.append(result.thrust / 1000.0)

plt.figure(figsize=(6, 4.2))
plt.plot(altitude_range / 1000, thrusts_alt, color="darkred", marker="o", markersize=3)
plt.xlabel("Altitude (km)")
plt.ylabel("Net thrust (kN)")
plt.title("Thrust Lapse Rate with Altitude (Mach 0.7)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("thrust_altitude_lapse.png", bbox_inches="tight")
plt.close()
print("Saved thrust_altitude_lapse.png")

# ---------------------------------------------------------------
# Plot 4: Sensitivity analysis -- compressor efficiency vs thrust/TSFC
# ---------------------------------------------------------------
compressor_effs = np.linspace(0.70, 0.92, 20)
cond = FlightCondition(altitude=0, mach=0.0)

sens_thrust, sens_tsfc = [], []
base_params = dict(TRUE_EFFICIENCIES)
for eff in compressor_effs:
    params = dict(base_params)
    params["compressor_eff"] = eff
    engine.set_calibratable_params(params)
    result = engine.run_cycle(cond)
    sens_thrust.append(result.thrust / 1000.0)
    sens_tsfc.append(result.tsfc * 3600 * 9.81)
engine.set_calibratable_params(base_params)  # restore

fig, ax1 = plt.subplots(figsize=(7, 4.5))
ax2 = ax1.twinx()

l1, = ax1.plot(compressor_effs * 100, sens_thrust, color="steelblue", marker="o", markersize=3, label="Thrust")
l2, = ax2.plot(compressor_effs * 100, sens_tsfc, color="darkorange", marker="s", markersize=3, label="TSFC")

ax1.set_xlabel("Compressor isentropic efficiency (%)")
ax1.set_ylabel("Sea-level static thrust (kN)", color="steelblue")
ax2.set_ylabel("TSFC (kg / kgf·hr)", color="darkorange")
ax1.set_title("Sensitivity: Compressor Efficiency Loss vs Performance\n(e.g. from engine aging/wear)")
ax1.tick_params(axis="y", labelcolor="steelblue")
ax2.tick_params(axis="y", labelcolor="darkorange")
ax1.legend(handles=[l1, l2], loc="center right")
ax1.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("compressor_efficiency_sensitivity.png", bbox_inches="tight")
plt.close()
print("Saved compressor_efficiency_sensitivity.png")
