import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ============================================================================
# TWO-STEP FIT: First R_ac, then L separately
# ============================================================================

def resistance_skin_effect(f, alpha, beta):
    """R_ac(ω) = α * exp(β*√ω)"""
    omega = 2 * np.pi * f
    return alpha * np.exp(beta * np.sqrt(omega))

def reactance_inductive(f, L):
    """X_L = ωL"""
    omega = 2 * np.pi * f
    return omega * L

def fit_two_step(frequencies, Z_measured):
    """
    Step 1: Fit Re(Z) to get skin effect parameters α, β
    Step 2: Fit Im(Z) to get L
    """
    # STEP 1: Fit resistance (real part only)
    def model_resistance(f, alpha, beta):
        return resistance_skin_effect(f, alpha, beta)

    alpha_guess = np.min(Z_measured.real)
    beta_guess = 1e-4

    popt_R, pcov_R = curve_fit(model_resistance, frequencies, Z_measured.real,
                               p0=[alpha_guess, beta_guess],
                               bounds=([0, 0], [100, 1e-2]),
                               maxfev=10000)

    alpha_fit, beta_fit = popt_R
    perr_R = np.sqrt(np.diag(pcov_R))

    # STEP 2: Fit reactance (imaginary part only)
    def model_reactance(f, L):
        return reactance_inductive(f, L)

    omega_min = 2 * np.pi * frequencies.min()
    L_guess = Z_measured.imag[np.argmin(frequencies)] / omega_min

    popt_L, pcov_L = curve_fit(model_reactance, frequencies, Z_measured.imag,
                               p0=[L_guess],
                               bounds=([0], [1e-3]))

    L_fit = popt_L[0]
    L_err = np.sqrt(pcov_L[0, 0])

    return alpha_fit, beta_fit, L_fit, perr_R[0], perr_R[1], L_err

# ============================================================================
# YOUR DATA
# ============================================================================

# Known resonance frequency
f_res = 4.89e6  # Hz (82 kHz)

# Your measurements (ABOVE resonance at 1-2 MHz)
# frequencies = np.array([2e6, 1.5e6, 1.24e6, 1.1e6, 1e6])  # Hz - NO SHORT
# Z_real = np.array([21, 18.4, 17.2, 16.7, 16.2])  # Ohms
# Z_imag = np.array([171, 121, 98, 86.5, 78.2])  # Ohms
# frequencies = np.array([2.2e6, 2.1e6, 2e6, 1.5e6, 1.24e6, 1.1e6, 1e6])  # Hz - WITH SHORT
# Z_real = np.array([21.2, 20.05, 19.3, 17.5, 16.4, 15.8, 15.4])  # Ohms
# Z_imag = np.array([178, 169, 155, 111, 90, 80.1, 72.2])  # Ohms
# frequencies = np.array([2.2e6, 2.1e6, 2e6, 1.5e6, 1.24e6, 1.1e6, 1e6])  # Hz - WITH SHORT
# Z_real = np.array([17.9, 17, 16.8, 15.5, 14.6, 14.2, 13.9])  # Ohms
# Z_imag = np.array([273, 253, 239, 162, 130, 112, 102])  # Ohms
frequencies = np.array([477e3, 400e3, 352e3, 300e3, 250e3, 200e3, 150e3, 103e3, 50e3])  # Hz - correct range (actually the better fit would be for f << SRF/10, so 200 kHz and down to 50 kHz)
Z_real = np.array([2.09, 1.84, 1.7, 1.52, 1.31, 1.15, 0.95, 0.7, 0.428])  # Ohms
Z_imag = np.array([50.2, 42.5, 37.2, 32, 26.6, 21.5, 16.2, 11.2, 5.33])  # Ohms
Z_measured = Z_real + 1j * Z_imag

# ============================================================================
# FIT
# ============================================================================

alpha_fit, beta_fit, L_fit, alpha_err, beta_err, L_err = fit_two_step(frequencies, Z_measured)

# Calculate C from resonance
C_fit = 1 / ((2 * np.pi * f_res)**2 * L_fit)
C_err = C_fit / L_fit * L_err

# Calculate R_dc
R_dc = alpha_fit

f_res_check = 1 / (2 * np.pi * np.sqrt(L_fit * C_fit))

print("="*70)
print("TWO-STEP FIT: RESISTANCE + INDUCTANCE")
print("="*70)
print(f"\nInput:")
print(f"  f_resonance (known) = {f_res/1e3:.1f} kHz")
print(f"  Measurement range: {frequencies.min()/1e6:.2f} - {frequencies.max()/1e6:.2f} MHz")
print(f"\nStep 1 - Skin effect (Re(Z) only):")
print(f"  α = {alpha_fit:.3f} ± {alpha_err:.3f} Ω")
print(f"  β = {beta_fit*1e3:.3f} ± {beta_err*1e3:.3f} ×10⁻³")
print(f"  R_dc = {R_dc:.2f} Ω")
print(f"\nStep 2 - Inductance (Im(Z) only):")
print(f"  L = {L_fit*1e6:.2f} ± {L_err*1e6:.2f} µH")
print(f"\nDerived:")
print(f"  C = {C_fit*1e9:.2f} ± {C_err*1e9:.2f} nF (from f_res)")
print(f"  f_res (from L,C) = {f_res_check/1e3:.1f} kHz ✓")
print(f"  R_ac(2 MHz) = {alpha_fit * np.exp(beta_fit * np.sqrt(2*np.pi*2e6)):.2f} Ω")
print("="*70)

# ============================================================================
# PLOTTING
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Extended frequency range
f_plot = np.linspace(frequencies.min()*0.5, frequencies.max()*1.2, 500)

# Fitted models
R_fit_plot = resistance_skin_effect(f_plot, alpha_fit, beta_fit)
X_fit_plot = reactance_inductive(f_plot, L_fit)
R_fit_meas = resistance_skin_effect(frequencies, alpha_fit, beta_fit)
X_fit_meas = reactance_inductive(frequencies, L_fit)

# Plot 1: Real part (Resistance with skin effect)
ax1 = axes[0, 0]
ax1.plot(frequencies/1e6, Z_measured.real, 'o', label='Measured', markersize=10, color='blue')
ax1.plot(f_plot/1e6, R_fit_plot, '-', label='Fit: R=α·exp(β√ω)', linewidth=2, color='red')
ax1.axhline(R_dc, linestyle='--', color='gray', alpha=0.7, label=f'R_dc={R_dc:.2f}Ω')
ax1.set_xlabel('Frequency (MHz)', fontsize=12)
ax1.set_ylabel('Re(Z) (Ω)', fontsize=12)
ax1.set_title('Resistance with Skin Effect', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Imaginary part
ax2 = axes[0, 1]
ax2.plot(frequencies/1e6, Z_measured.imag, 'o', label='Measured', markersize=10, color='blue')
ax2.plot(f_plot/1e6, X_fit_plot, '-', label=f'Fit: L={L_fit*1e6:.2f}µH', linewidth=2, color='red')
ax2.set_xlabel('Frequency (MHz)', fontsize=12)
ax2.set_ylabel('Im(Z) (Ω)', fontsize=12)
ax2.set_title('Reactance (Inductive)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Plot 3: Im(Z) vs ω (linearity check)
ax3 = axes[1, 0]
omega_meas = 2 * np.pi * frequencies
omega_plot = 2 * np.pi * f_plot
ax3.plot(omega_meas/1e6, Z_measured.imag, 'o', label='Measured', markersize=10, color='blue')
ax3.plot(omega_plot/1e6, omega_plot * L_fit, '-', label=f'L={L_fit*1e6:.2f}µH', linewidth=2, color='red')
ax3.set_xlabel('ω (Mrad/s)', fontsize=12)
ax3.set_ylabel('Im(Z) (Ω)', fontsize=12)
ax3.set_title('Linearity Check: Im(Z) = ωL', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Plot 4: Residuals
ax4 = axes[1, 1]
res_real = Z_measured.real - R_fit_meas
res_imag = Z_measured.imag - X_fit_meas
ax4.plot(frequencies/1e6, res_real, 'o-', label='Re(Z)', markersize=8, color='blue')
ax4.plot(frequencies/1e6, res_imag, 's-', label='Im(Z)', markersize=8, color='orange')
ax4.axhline(y=0, color='k', linestyle=':', alpha=0.5)
ax4.set_xlabel('Frequency (MHz)', fontsize=12)
ax4.set_ylabel('Residual (Ω)', fontsize=12)
ax4.set_title('Fit Residuals (Two-Step)', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend()

plt.tight_layout()
plt.show()
