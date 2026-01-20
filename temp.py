import numpy as np
import matplotlib.pyplot as plt

# Physical constants (SI)
h = 6.62607015e-34
c = 299792458.0
kB = 1.380649e-23

# Plasma parameters
ne = 2e19  # m^-3
Z = 2  # helium ions
ni = ne / Z  # ion density for fully ionized helium (charge neutrality)
Te_eV = 10.0
Te_K = Te_eV * 11604.518  # Kelvin

# Wavelength range (nm) – EUV/UV where emission actually exists
lambda_nm = np.linspace(5, 300, 2000)
lambda_m = lambda_nm * 1e-9
nu = c / lambda_m

# Convert densities to cgs (required by RL formula)
ne_cgs = ne * 1e-6
ni_cgs = ni * 1e-6

# Thermal bremsstrahlung emissivity (Rybicki & Lightman, eq. 5.14a)
# erg s^-1 cm^-3 Hz^-1 sr^-1
# The formula is: eps_nu = (2^5 * pi * e^6)/(3 * m_e * c^3) * (2*pi/(3*k_B*m_e))^(1/2) * Z^2 * n_e * n_i * T_e^(-1/2) * exp(-h*nu/(k_B*T_e)) * g_ff
# The numerical coefficient is ~6.8e-38 (assuming Gaunt factor g_ff ~ 1)
eps_nu_cgs = (
    6.8e-38 * Z**2 * ne_cgs * ni_cgs * Te_K ** (-0.5) * np.exp(-h * nu / (kB * Te_K))
)

# Convert to SI: J s^-1 m^-3 Hz^-1
eps_nu = eps_nu_cgs * 1e-1

# Convert to per-wavelength emissivity
eps_lambda = eps_nu * c / lambda_m**2

# Normalize (PSD shape only)
eps_lambda /= np.max(eps_lambda)

# Plot
plt.figure()
plt.plot(lambda_nm, eps_lambda)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Normalized spectral power density")
plt.title("Optically Thin Bremsstrahlung PSD\n$T_e=10$ eV, $n_e=2\\times10^{19}$ m$^{-3}$, He II")
plt.yscale("log")
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.show()
