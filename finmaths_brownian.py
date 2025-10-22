import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

st.set_page_config(page_title="Stochastic Processes Simulator", layout="wide")

# --- Title ---
st.title("Continuous-Time Stochastic Processes Simulator")
st.write("Interactive exploration of Brownian Motion, Geometric Brownian Motion, Ornstein-Uhlenbeck process, and Itô calculus verification.")

# --- Sidebar: User Inputs ---
st.sidebar.header("Simulation Parameters")

# General parameters
T = st.sidebar.slider("Time Horizon (T)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
N = st.sidebar.slider("Number of Time Steps (N)", min_value=50, max_value=5000, value=1000, step=50)
num_paths = st.sidebar.slider("Number of Sample Paths", min_value=1, max_value=50, value=5, step=1)
M = st.sidebar.slider("Number of Monte Carlo Paths (for Itô verification)", min_value=100, max_value=50000, value=10000, step=100)

# GBM parameters
S0 = st.sidebar.number_input("Initial Stock Price (S0)", value=100.0)
mu = st.sidebar.number_input("Drift (mu)", value=0.05, step=0.01)
sigma = st.sidebar.number_input("Volatility (sigma)", value=0.2, step=0.01)

# OU parameters
X0 = st.sidebar.number_input("Initial OU Process Value (X0)", value=1.0)
alpha = st.sidebar.number_input("Mean Reversion (alpha)", value=1.0)
sigma_ou = st.sidebar.number_input("OU Volatility (sigma)", value=0.3)

# Seed for reproducibility
seed = st.sidebar.number_input("Random Seed", value=42, step=1)
np.random.seed(seed)

# --- Class Definitions (as provided) ---
class StochasticProcesses:
    @staticmethod
    def simulate_brownian_motion(T: float, N: int, M: int = 1):
        dt = T / N
        t = np.linspace(0, T, N + 1)
        dW = np.sqrt(dt) * np.random.randn(M, N)
        W = np.zeros((M, N + 1))
        W[:, 1:] = np.cumsum(dW, axis=1)
        return t, W

    @staticmethod
    def simulate_geometric_brownian_motion(S0: float, mu: float, sigma: float,
                                           T: float, N: int, M: int = 1):
        dt = T / N
        t = np.linspace(0, T, N + 1)
        t_grid = np.tile(t, (M, 1))
        _, W = StochasticProcesses.simulate_brownian_motion(T, N, M)
        S = S0 * np.exp((mu - 0.5 * sigma**2) * t_grid + sigma * W)
        return t, S

    @staticmethod
    def simulate_gbm_euler(S0: float, mu: float, sigma: float,
                           T: float, N: int, M: int = 1):
        dt = T / N
        t = np.linspace(0, T, N + 1)
        S = np.zeros((M, N + 1))
        S[:, 0] = S0
        for i in range(N):
            dW = np.sqrt(dt) * np.random.randn(M)
            S[:, i+1] = S[:, i] + mu * S[:, i] * dt + sigma * S[:, i] * dW
        return t, S

    @staticmethod
    def simulate_ornstein_uhlenbeck(X0: float, alpha: float, sigma: float,
                                    T: float, N: int, M: int = 1):
        dt = T / N
        t = np.linspace(0, T, N + 1)
        X = np.zeros((M, N + 1))
        X[:, 0] = X0
        for i in range(N):
            dW = np.sqrt(dt) * np.random.randn(M)
            X[:, i+1] = X[:, i] - alpha * X[:, i] * dt + sigma * dW
        return t, X

    @staticmethod
    def ito_integral_verification(T: float = 1.0, N: int = 10000, M: int = 10000):
        t, W = StochasticProcesses.simulate_brownian_motion(T, N, M)
        dW = np.diff(W, axis=1)
        ito_integral = np.sum(W[:, :-1] * dW, axis=1)
        analytical = (W[:, -1]**2 - T)/2
        return ito_integral, analytical

# --- Tabs for different simulations ---
tab1, tab2, tab3, tab4 = st.tabs(["Brownian Motion", "GBM", "OU Process", "Itô Integral Verification"])

with tab1:
    st.subheader("Standard Brownian Motion")
    t, W = StochasticProcesses.simulate_brownian_motion(T, N, num_paths)
    fig, ax = plt.subplots(figsize=(10, 4))
    for i in range(num_paths):
        ax.plot(t, W[i, :], alpha=0.7)
    ax.set_title("Standard Brownian Motion Paths")
    ax.set_xlabel("Time")
    ax.set_ylabel("W(t)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with tab2:
    st.subheader("Geometric Brownian Motion")
    t, S = StochasticProcesses.simulate_geometric_brownian_motion(S0, mu, sigma, T, N, num_paths)
    fig, ax = plt.subplots(figsize=(10, 4))
    for i in range(num_paths):
        ax.plot(t, S[i, :], alpha=0.7)
    ax.set_title("Geometric Brownian Motion Paths")
    ax.set_xlabel("Time")
    ax.set_ylabel("S(t)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.subheader("Euler vs Exact GBM Comparison")
    t_exact, S_exact = StochasticProcesses.simulate_geometric_brownian_motion(S0, mu, sigma, T, N, M)
    t_euler, S_euler = StochasticProcesses.simulate_gbm_euler(S0, mu, sigma, T, N, M)
    st.write(f"Exact GBM Final Mean: {np.mean(S_exact[:,-1]):.4f}, Std: {np.std(S_exact[:,-1]):.4f}")
    st.write(f"Euler GBM Final Mean: {np.mean(S_euler[:,-1]):.4f}, Std: {np.std(S_euler[:,-1]):.4f}")

with tab3:
    st.subheader("Ornstein-Uhlenbeck Process")
    t, X = StochasticProcesses.simulate_ornstein_uhlenbeck(X0, alpha, sigma_ou, T, N, num_paths)
    fig, ax = plt.subplots(figsize=(10, 4))
    for i in range(num_paths):
        ax.plot(t, X[i, :], alpha=0.7)
    ax.set_title("Ornstein-Uhlenbeck Paths")
    ax.set_xlabel("Time")
    ax.set_ylabel("X(t)")
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with tab4:
    st.subheader("Itô Integral Verification")
    ito_int, analytical = StochasticProcesses.ito_integral_verification(T, N, M)
    diff = ito_int - analytical
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(diff, bins=50, density=True, alpha=0.6, color='purple', label='Numerical - Analytical')
    mu_diff, sigma_diff = np.mean(diff), np.std(diff)
    x_diff = np.linspace(min(diff), max(diff), 100)
    ax.plot(x_diff, norm.pdf(x_diff, mu_diff, sigma_diff), 'r--', label='Normal Fit')
    ax.set_title("Itô Integral Verification Histogram")
    ax.set_xlabel("Difference (Numerical - Analytical)")
    ax.set_ylabel("Density")
    ax.axvline(x=0, color='k', linestyle='-')
    ax.text(0.02, 0.9, f"Mean Diff: {mu_diff:.4e}\nStd Dev: {sigma_diff:.4e}", transform=ax.transAxes)
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)