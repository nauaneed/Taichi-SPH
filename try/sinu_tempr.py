import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def T_target(y, T_max, T_min, ymin, ymax):
    theta = (y - ymin) / (ymax - ymin) * np.pi - 0.5 * np.pi
    return T_min + (T_max - T_min) * (0.5 + 0.5 * np.sin(theta))

def VFT_viscosity(T, vftA, vftB, vftT0, vftViscosityMin, vftViscosityMax):
    viscosity = np.power(10, vftA  + vftB / (T - vftT0))
    viscosity = np.clip(viscosity, vftViscosityMin, vftViscosityMax)
    return viscosity


this_file_dir = Path(__file__).parent


# Parameters
T_max = 1100
T_min = 300
ymax = 2.2
ymin = 1.4
y = np.linspace(1.4, 2.2, 1000)

# Plotting
plt.figure(figsize=(10, 6))
T_Profile = T_target(y, T_max, T_min, ymin, ymax)


# Steep transition
plt.axhline(y=T_max, color='r', linestyle='--', label=f'T_max = {T_max}')
plt.axhline(y=T_min, color='b', linestyle='--', label=f'T_min = {T_min}')
plt.plot(y, T_Profile, linewidth=2, label='Target Temperature Profile')

plt.title(r'Target Temperature Profile $T_{\text{target}}(y)$')
plt.xlabel('y')
plt.ylabel(r'$T_{\text{target}}(y)$')
plt.grid(True)

plt.savefig(this_file_dir / 'target_temperature_profile.png')

vftA = -0.5
vftB = 3581
vftT0 = 102.05
vftViscosityMin = 1000.0
vftViscosityMax = 500000.0
viscosity_profile = VFT_viscosity(T_Profile, vftA, vftB, vftT0, vftViscosityMin, vftViscosityMax)
print(VFT_viscosity(800, vftA, vftB, vftT0, vftViscosityMin, vftViscosityMax))
plt.figure(figsize=(10, 6))

plt.axhline(y=vftViscosityMax, color='r', linestyle='--', label=f'VFT Viscosity Max = {vftViscosityMax}')
plt.axhline(y=vftViscosityMin, color='b', linestyle='--', label=f'VFT Viscosity Min = {vftViscosityMin}')
plt.plot(y, viscosity_profile, linewidth=2, label='VFT Viscosity Profile')
plt.title('VFT Viscosity Profile')
plt.xlabel('y')
plt.ylabel('Viscosity')
plt.grid(True)

plt.savefig(this_file_dir / 'vft_viscosity_profile.png')








