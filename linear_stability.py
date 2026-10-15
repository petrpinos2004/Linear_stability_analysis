import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eig
from math import *

class Stability:
    def __init__(self, setup, mode='convection'):

        self.setup = setup
        self.mode = mode

        #nondimensionalization
        self.setup['omega'] = 2 * pi *self.setup['f']
        self.setup['tau_D'] = self.setup['delta_diff']**2 / self.setup['kappa']
        self.setup['mu'] = (self.setup['gamma'] - 1) / self.setup['L'] * sqrt(2 * self.setup['kappa'] / self.setup['omega'])
        self.setup['D_val'] = self.setup['alpha_p'] * 9.81 * self.setup['L'] / self.setup['cp']
        self.setup['Pr'] = self.setup['nu'] / self.setup['kappa']
        self.setup['Ra_th'] = self.setup['rho'] * 9.81 * self.setup['cp'] * self.setup['L']**3 / (self.setup['nu'] * self.setup['lambda'])
        print(self.setup['Ra_th'] * self.setup['alpha_p'] * self.setup['A_T'], self.setup['Pr'], self.setup['D_val'])

    def create_chebyshev_grid(self):
        j = np.arange(self.setup['N'] + 1)
        z = np.cos(np.pi * j / self.setup['N'])
        return z

    def get_derivatives(self):
        j = np.arange(self.setup['N'] + 1)
        xi = np.cos(np.pi * j / self.setup['N'])

        c = np.ones(self.setup['N'] + 1)
        c[0] = 2.0
        c[self.setup['N']] = 2.0
        c = c * (-1) ** j

        X = np.outer(xi, np.ones(self.setup['N'] + 1))
        dX = X - X.T
        D1_comp = np.outer(c, 1.0 / c) / (dX + np.eye(self.setup['N'] + 1))
        D1_comp = D1_comp - np.diag(np.sum(D1_comp.T, axis=0))
        D2_comp = np.dot(D1_comp, D1_comp)

        z = 0.5 * xi
        D1 = 2.0 * D1_comp  
        D2 = 4.0 * D2_comp

        return z, D1, D2

    def analytic_model_temp_profile(self, z):
        theta_B = - np.arctan(1.0 / (1.0 + self.setup['mu']))
        theta_A = np.arctan(self.setup['mu'] / (2.0 + self.setup['mu']))

        #transform between nondimensional and normal coordinates
        x = self.setup['L']*z + 1/2 * self.setup['L']
        alpha_term = np.sqrt(self.setup['omega'] / (2.0 * self.setup['kappa']))

        coeff_1 = (self.setup['A_T'] * np.sqrt((2.0 + self.setup['mu']) ** 2 + self.setup['mu']**2) / ((1.0 + self.setup['mu']) ** 2 + 1.0))
        phase_1 = theta_A - alpha_term * x + self.setup['omega'] * self.setup['t']
        term_1 = (coeff_1 * np.exp(-alpha_term * x) * (np.cos(phase_1) + 1j * np.sin(phase_1)))

        coeff_2 = self.setup['A_T'] * self.setup['mu'] / np.sqrt((1.0 + self.setup['mu']) ** 2 + 1.0)
        phase_2 = theta_B + self.setup['omega'] * self.setup['t']
        term_2 = coeff_2 * (np.cos(phase_2) + 1j * np.sin(phase_2))

        non_dimensional = np.real(term_1 + term_2) / self.setup['A_T']

        return non_dimensional

    def analytic_model_pressure_profile(self, z):
        theta_B = - np.arctan(1.0 / (1.0 + self.setup['mu']))
        amplitude = self.setup['A_T'] * self.setup['gamma'] * (self.setup['alpha_p'] / self.setup['alpha_T']) * np.sqrt(2.0 / (self.setup['omega'] * self.setup['tau_D'])) / np.sqrt((1.0 + self.setup['mu']) ** 2 + 1.0)
        phase = theta_B + self.setup['omega'] * self.setup['t']
        p_complex = amplitude * (np.cos(phase) + 1j * np.sin(phase))
        return np.real(p_complex)

    def build_system_matrix(self):
        z, D1, D2 = self.get_derivatives()
        n_pts = self.setup['N'] + 1
        dim = 4 * n_pts  
        
        A = np.zeros((dim, dim), dtype=complex)
        B = np.zeros((dim, dim), dtype=complex)
        I = np.eye(n_pts)

        if self.mode == 'modulation':
            Tb = self.analytic_model_temp_profile(z)
            p_b_scalar = self.analytic_model_pressure_profile(z)
            rho_b = self.setup['alpha_p'] * Tb * self.setup['A_T']  + self.setup['alpha_T'] * p_b_scalar * np.ones_like(z)
            print(np.mean(rho_b))
            drho_dz = D1 @ rho_b
            dTb_dz = D1 @ Tb
        else:
            Tb = 1.0 - self.setup['a'] * z  #Convection regime
            dTb_dz = -self.setup['a'] * np.ones_like(z)
            rho_b = np.ones_like(z)
            drho_dz = np.zeros_like(z)
        
        dp_dz = -rho_b
        
        ux_slice = slice(0, n_pts)
        uz_slice = slice(n_pts, 2 * n_pts)
        p_slice = slice(2 * n_pts, 3 * n_pts)
        T_slice = slice(3 * n_pts, 4 * n_pts)
        
        B[ux_slice, p_slice] = self.setup['alpha_p'] * I
        B[ux_slice, T_slice] = self.setup['alpha_T'] * I
        A[ux_slice, ux_slice] = -1j * self.setup['kx'] * np.diag(rho_b)
        A[ux_slice, uz_slice] = -np.diag(rho_b) @ D1 - np.diag(drho_dz)
        
        B[uz_slice, ux_slice] = self.setup['Pr'] ** (-1) * np.diag(rho_b)
        A[uz_slice, ux_slice] = D2 - (4.0 / 3.0) * (self.setup['kx']**2) * I
        A[uz_slice, uz_slice] = (1j * self.setup['kx'] / 3.0) * D1
        A[uz_slice, p_slice] = -self.setup['Ra_th'] * 1j * self.setup['kx'] * I
        
        B[p_slice, uz_slice] = self.setup['Pr'] ** (-1) * np.diag(rho_b)
        A[p_slice, ux_slice] = (1j * self.setup['kx'] / 3.0) * D1
        A[p_slice, uz_slice] = (4.0 / 3.0) * D2 - (self.setup['kx']**2) * I
        A[p_slice, p_slice] = -self.setup['Ra_th'] * D1 - self.setup['Ra_th'] * self.setup['alpha_pb'] * I
        A[p_slice, T_slice] = -self.setup['Ra_th'] * self.setup['alpha_T'] * I
        
        B[T_slice, p_slice] = -self.setup['D_val'] * self.setup['alpha_pb'] * np.diag(Tb)
        B[T_slice, T_slice] = np.diag(rho_b * self.setup['cp_b'])
        A[T_slice, uz_slice] = np.diag(-rho_b * self.setup['cp_b'] * dTb_dz + self.setup['D_val'] * self.setup['alpha_pb'] * Tb * dp_dz)
        A[T_slice, T_slice] = D2 - (self.setup['kx']**2) * I
        
        for b_idx in [0, self.setup['N']]:
            for field_base in [0, n_pts, 3 * n_pts]:
                row = b_idx + field_base
                A[row, :] = 0
                B[row, :] = 0
                A[row, row] = 1.0
                
        return A, B

    def solve_stability(self, A, B):
        eigenvalues, eigenvectors = eig(A, B)
        valid = np.isfinite(eigenvalues) & (np.abs(eigenvalues) < 1e5)
        return eigenvalues[valid], eigenvectors[:, valid]

    def plot_spectrum(self, eigenvalues):
        plt.figure(figsize=(8, 6))
        real_parts = np.real(eigenvalues)
        imag_parts = np.imag(eigenvalues)

        unstable = real_parts > 0
        stable = ~unstable

        plt.scatter(real_parts[stable], imag_parts[stable], color="blue", alpha=0.6, label="Stable")
        plt.scatter(real_parts[unstable], imag_parts[unstable], color="red", alpha=0.8, s=40, label="Unstable")

        plt.axvline(x=0, color="black", linestyle="--", linewidth=1)
        plt.xlabel("Growth Rate (Re(sigma))", fontsize=12)
        plt.ylabel("Oscillation Frequency (Im(sigma))", fontsize=12)
        plt.title("Linear Stability Eigenvalue Spectrum", fontsize=14)
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.legend(loc="lower left")
        plt.tight_layout()

    def plot_top_eigenmodes(self, z, eigenvectors, eigenvalues, top_k=5):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
        n_pts = self.setup['N'] + 1
        sorted_idx = np.argsort(np.real(eigenvalues))[::-1]

        for i in range(min(top_k, len(sorted_idx))):
            idx = sorted_idx[i]
            val = eigenvalues[idx]

            uz_profile = np.real(eigenvectors[n_pts : 2 * n_pts, idx])
            T_profile = np.real(eigenvectors[3 * n_pts : 4 * n_pts, idx])

            if np.max(np.abs(uz_profile)) > 0:
                uz_profile /= np.max(np.abs(uz_profile))
            if np.max(np.abs(T_profile)) > 0:
                T_profile /= np.max(np.abs(T_profile))

            label_str = f"Mode {i+1} ($\sigma$={val.real:.2f})"
            ax1.plot(uz_profile, z, linewidth=2, label=label_str)
            ax2.plot(T_profile, z, linewidth=2, label=label_str)

        ax1.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
        ax1.set_ylabel("Vertical Coordinate ($z$)", fontsize=12)
        ax1.set_xlabel("Normalized Vertical Velocity ($u_z$)", fontsize=12)
        ax1.set_title("Vertical Velocity Eigenmodes", fontsize=13)
        ax1.grid(True, linestyle=":", alpha=0.7)

        ax2.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
        ax2.set_xlabel("Normalized Temperature Perturbation ($T$)", fontsize=12)
        ax2.set_title("Temperature Eigenmodes", fontsize=13)
        ax2.grid(True, linestyle=":", alpha=0.7)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

    def plot_analytical_temperature_profile(self, z):
        plt.figure(figsize=(9, 6))
        x = z - np.min(z)
        #times = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

        original_t = self.setup['t']
        for ti in self.setup['t']:
            self.setup['t'] = ti
            Tb = self.analytic_model_temp_profile(z)
            plt.plot(x, Tb, linewidth=1.5, label=f"t = {ti} s")
        self.setup['t'] = original_t

        plt.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
        plt.xlabel("Boundary Distance ($x$) [units]", fontsize=12)
        plt.ylabel("$T(x,t)$ [K]", fontsize=12)
        plt.title("Harmonic Solution", fontsize=14)
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

    def run(self):
        z, _, _ = self.get_derivatives()
        A, B = self.build_system_matrix()
        eigenvalues, eigenvectors = self.solve_stability(A, B)

        sorted_idx = np.argsort(np.real(eigenvalues))[::-1]
        sorted_sigma = eigenvalues[sorted_idx]

        print(f"\n--- Top 10 Eigenvalues (Growth Rates) ---")
        for i in range(min(10, len(sorted_sigma))):
            print(f"  Mode {i+1}: {sorted_sigma[i]}")

        max_sigma = np.real(sorted_sigma[0]) if len(sorted_sigma) > 0 else 0.0
        print(f"\nMax Growth Rate: {max_sigma:.5f}")
        print("Status: UNSTABLE" if max_sigma > 0 else "Status: STABLE")

        self.plot_spectrum(eigenvalues)
        plt.show()
        self.plot_top_eigenmodes(z, eigenvectors, eigenvalues, top_k=5)
        plt.show()

        if self.mode == 'modulation':
            self.plot_analytical_temperature_profile(z)
            plt.show()