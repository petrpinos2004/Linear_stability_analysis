import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eig
from math import *
from HEPAK_API import HEPAK

class Stability:

    def __init__(self, setup):

        self.setup = setup
        self.He = HEPAK()

        #Thermodynamic properties from HEPAK
        self.setup['alpha_p'] = self.He.alpha_p(self.setup['T_T'], self.setup['p0'])
        self.setup['alpha_T'] = self.He.alpha_T(self.setup['T_T'], self.setup['p0'])
        self.setup['cp']    =   self.He.cp(self.setup['T_T'], self.setup['p0'])
        self.setup['kappa']  =  self.He.kappa(self.setup['T_T'], self.setup['p0'])
        self.setup['gamma'] =   self.He.gamma(self.setup['T_T'], self.setup['p0'])
        self.setup['lambda'] =   self.He.lam(self.setup['T_T'], self.setup['p0'])
        self.setup['eta'] =   self.He.eta(self.setup['T_T'], self.setup['p0'])

        #Derived values
        self.setup['omega'] = 2 * pi *self.setup['f']
        self.setup['delta_diff'] = np.sqrt(self.setup['kappa'] / (self.setup['f'] * np.pi ))
        self.setup['tau_D'] = self.setup['delta_diff']**2 / self.setup['kappa']
        self.setup['mu'] = (self.setup['gamma'] - 1) / self.setup['L'] * sqrt(2 * self.setup['kappa'] / self.setup['omega'])
        self.setup['g'] = 9.81
       
        #Base solution from analytical model
        z, D1, _ = self.get_derivative_matricies()
        self.T = self.analytic_model_temp_profile(z)
        self.p = self.analytic_model_pressure_profile() + self.setup['p0']
        self.rho = self.He.rho(self.T, self.p)

        #Space derivatives of base solution
        self.drho_dz = D1 @ self.rho
        self.dT_dz = D1 @ self.T
        self.dp_dz = self.rho * self.setup['g']

        #Time derivatives of base solution
        self.dT_dt = self.analytic_model_temp_profile(z, derivative=True)
        self.dp_dt = self.analytic_model_pressure_profile(derivative=True)
        time_shift = self.setup['t'] + self.setup['dt']
        rho_shift = self.He.rho(self.analytic_model_temp_profile(z,time_shift), self.analytic_model_pressure_profile(time_shift) + self.setup['p0'])
        self.drho_dt = (rho_shift - self.rho) / self.setup['dt']

    def nondimensionalization(self):
        pass
        #nondimensionalization is not performed at this stage
        
    def create_chebyshev_grid(self):
        j = np.arange(self.setup['N'] + 1)
        xi = np.cos(np.pi * j / self.setup['N'])
        z = 0.5 * self.setup['L'] * (xi + 1.0)
        return z

    def get_derivative_matricies(self):
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

        z = 0.5 * self.setup['L'] * (xi + 1.0)
        D1 = (2.0 / self.setup['L']) * D1_comp  
        D2 = (4.0 / self.setup['L']**2 ) * D2_comp

        return z, D1, D2

    def analytic_model_temp_profile(self, z, t=None, derivative=False):

        if t is None:
            t = self.setup['t']

        theta_B = - np.arctan(1.0 / (1.0 + self.setup['mu']))
        theta_A = np.arctan(self.setup['mu'] / (2.0 + self.setup['mu']))

        #transform between nondimensional and normal coordinates
        x = z
        alpha_term = np.sqrt(self.setup['omega'] / (2.0 * self.setup['kappa']))

        amplitude_BL = (self.setup['A_T'] * np.sqrt((2.0 + self.setup['mu']) ** 2 + self.setup['mu']**2) / ((1.0 + self.setup['mu']) ** 2 + 1.0))
        phase_BL = theta_A - alpha_term * x + self.setup['omega'] * t
        BL = (amplitude_BL * np.exp(-alpha_term * x) * (np.cos(phase_BL) + 1j * np.sin(phase_BL)))

        amplitude_PE = self.setup['A_T'] * self.setup['mu'] / np.sqrt((1.0 + self.setup['mu']) ** 2 + 1.0)
        phase_PE = theta_B + self.setup['omega'] * t
        PE = amplitude_PE * (np.cos(phase_PE) + 1j * np.sin(phase_PE))

        T_background = self.setup['T_T'] + (self.setup['T_B'] - self.setup['T_T']) * z / self.setup['L']

        if derivative:
            return np.real((BL + PE) * 1j * self.setup['omega'])
        else:
            return np.real(BL + PE) + T_background

    def analytic_model_pressure_profile(self, t=None, derivative=False):

        if t is None:
            t = self.setup['t']

        theta_B = - np.arctan(1.0 / (1.0 + self.setup['mu']))

        amplitude = self.setup['A_T'] * (self.setup['delta_diff'] / self.setup['L']) * (self.setup['alpha_p'] / (self.setup['alpha_T'])) * self.setup['gamma'] / np.sqrt((1.0 + self.setup['mu']) ** 2 + 1.0)
        phase = theta_B + self.setup['omega'] * t
        pressure = amplitude * (np.cos(phase) + 1j * np.sin(phase))
        
        if derivative:
            return np.real(pressure * 1j * self.setup['omega'])
        else:
            return np.real(pressure)

    def build_system_matrix(self):
        n_pts = self.setup['N'] + 1
        dim = 5 * n_pts  
        
        A = np.zeros((dim, dim), dtype=complex)
        B = np.zeros((dim, dim), dtype=complex)
        I = np.eye(n_pts)
        
        ux_slice = slice(0, n_pts)
        uz_slice = slice(n_pts, 2 * n_pts)
        p_slice = slice(2 * n_pts, 3 * n_pts)
        T_slice = slice(3 * n_pts, 4 * n_pts)
        rho_slice = slice(4 * n_pts, 5 * n_pts)

        _, D1, D2 = self.get_derivative_matricies()
        
        B[p_slice, rho_slice] = I
        A[p_slice, ux_slice] = -1j * self.setup['kx'] * np.diag(self.rho)
        A[p_slice, uz_slice] = -np.diag(self.rho) @ D1 - np.diag(self.drho_dz)
        
        B[ux_slice, ux_slice] = np.diag(self.rho)
        A[ux_slice, ux_slice] = self.setup['eta'] * (D2 - self.setup['kx']**2 * I)
        A[ux_slice, uz_slice] = self.setup['eta'] * (1j * self.setup['kx'] / 3.0) * D1
        A[ux_slice, p_slice] = -1j * self.setup['kx'] * I
        
        B[uz_slice, uz_slice] = np.diag(self.rho)
        A[uz_slice, ux_slice] = self.setup['eta'] * (1j * self.setup['kx'] / 3.0) * D1
        A[uz_slice, uz_slice] = self.setup['eta'] * ((4.0 / 3.0) * D2 - (self.setup['kx']**2) * I)
        A[uz_slice, p_slice] = - D1
        A[uz_slice, rho_slice] = self.setup['g'] * I
        
        B[T_slice, p_slice] = -self.setup['alpha_p'] * np.diag(self.T)
        B[T_slice, T_slice] = self.setup['cp'] * np.diag(self.rho)
        A[T_slice, uz_slice] = -self.setup['cp'] * np.diag(self.rho * self.dT_dz) + self.setup['alpha_p'] * np.diag(self.T * self.dp_dz)
        A[T_slice, T_slice] = self.setup['lambda'] * (D2 - (self.setup['kx']**2) * I) + self.setup['alpha_p'] * self.dp_dt * I
        A[T_slice, rho_slice] = -self.setup['cp'] * np.diag(self.dT_dt)

        A[rho_slice, rho_slice] = I
        A[rho_slice, p_slice] = - np.diag(self.rho * self.He.alpha_p(self.T, self.p))
        A[rho_slice, T_slice] = np.diag(self.rho * self.He.alpha_T(self.T, self.p))

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
        x = z
        times = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

        original_t = self.setup['t']

        for i, ti in enumerate(times):
            self.setup['t'] = ti
            Tb = self.analytic_model_temp_profile(z)

            plt.plot(x, Tb, linewidth=1.5, label=f"t = {ti} s")

            if i == 0:
                self.setup['t'] = original_t
                T_main = self.analytic_model_temp_profile(z)
                plt.plot(
                    x, T_main,
                    linewidth=2.5,
                    color='black',
                    label=f"This stability analysis: t = {original_t} s"
                )

                # Computational grid points
                plt.scatter(
                    x, T_main,
                    color='black',
                    s=18,
                    zorder=5,
                    label=f"Grid points (N = {self.setup['N'] + 1})"
                )

        self.setup['t'] = original_t

        plt.xlabel("Vertical coordinate $z$ [m]", fontsize=12)
        plt.ylabel("$T(x,t)$ [K]", fontsize=12)
        plt.title("Harmonic Solution and Computational Grid", fontsize=14)
        plt.xlim(0, 0.05)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

    def run(self):
        z, _, _ = self.get_derivative_matricies()
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

        self.plot_analytical_temperature_profile(z)
        plt.show()
