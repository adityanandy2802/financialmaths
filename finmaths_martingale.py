import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Martingale & Option Pricing Simulator", layout="wide")
st.title("Martingale Theory and Risk-Neutral Option Pricing")
st.write("Interactive demonstration of martingale property, European and exotic option pricing under risk-neutral measure.")

# --- Sidebar: Inputs ---
st.sidebar.header("Simulation Parameters")
S0 = st.sidebar.number_input("Initial Stock Price (S0)", value=100.0, step=1.0)
r = st.sidebar.number_input("Risk-free Rate (r)", value=0.05, step=0.01)
sigma = st.sidebar.number_input("Volatility (sigma)", value=0.2, step=0.01)
T = st.sidebar.number_input("Time to Maturity (T in years)", value=1.0, step=0.1)
K = st.sidebar.number_input("Strike Price (K)", value=100.0, step=1.0)
N = st.sidebar.slider("Number of Time Steps (for Binomial paths)", min_value=10, max_value=1000, value=100, step=10)
num_paths = st.sidebar.slider("Number of Sample Paths", min_value=10, max_value=5000, value=1000, step=100)
M_MC = st.sidebar.slider("Number of Monte Carlo simulations", min_value=1000, max_value=500000, value=50000, step=5000)
seed = st.sidebar.number_input("Random Seed", value=42, step=1)
np.random.seed(seed)

# --- Class Definition ---
class MartingalePricing:
    def __init__(self, S0, r, sigma, T):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T

    def simulate_stock_paths_discrete(self, N, M, use_risk_neutral=True):
        dt = self.T / N
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(self.r * dt) - d) / (u - d) if use_risk_neutral else (np.exp(self.r * dt) - d) / (u - d)
        paths = np.zeros((M, N + 1))
        paths[:, 0] = self.S0
        for t in range(1, N + 1):
            moves = np.random.binomial(1, p, M)
            paths[:, t] = paths[:, t-1] * np.where(moves, u, d)
        return paths

    def verify_martingale_property(self, N=100, M=10000):
        paths = self.simulate_stock_paths_discrete(N, M)
        time_grid = np.linspace(0, self.T, N+1)
        discounted_paths = paths * np.exp(-self.r * time_grid)
        expected_values = np.mean(discounted_paths, axis=0)
        return time_grid, discounted_paths, expected_values

    def price_european_option_mc(self, K, option_type='call', M=100000):
        Z = np.random.standard_normal(M)
        S_T = self.S0 * np.exp((self.r - 0.5*self.sigma**2)*self.T + self.sigma*np.sqrt(self.T)*Z)
        if option_type=='call':
            payoffs = np.maximum(S_T - K, 0)
        elif option_type=='put':
            payoffs = np.maximum(K - S_T, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        price = np.exp(-self.r*self.T) * np.mean(payoffs)
        std_error = np.exp(-self.r*self.T) * np.std(payoffs)/np.sqrt(M)
        return price, std_error

    def price_exotic_options(self, K, M=100000):
        Z = np.random.standard_normal(M)
        S_T = self.S0 * np.exp((self.r - 0.5*self.sigma**2)*self.T + self.sigma*np.sqrt(self.T)*Z)
        prices = {}
        prices['Digital Call'] = np.exp(-self.r*self.T)*np.mean((S_T>K).astype(float))
        prices['Digital Put'] = np.exp(-self.r*self.T)*np.mean((S_T<K).astype(float))
        prices['Power Call'] = np.exp(-self.r*self.T)*np.mean(np.maximum(S_T**2-K**2,0))
        K1,K2 = K,K*1.1
        prices['Gap Call'] = np.exp(-self.r*self.T)*np.mean(np.where(S_T>K1,S_T-K2,0))
        return prices

    def verify_put_call_parity(self, K, M=100000):
        call, _ = self.price_european_option_mc(K, 'call', M)
        put, _ = self.price_european_option_mc(K, 'put', M)
        lhs = call - put
        rhs = self.S0 - K*np.exp(-self.r*self.T)
        return {'call':call, 'put':put, 'lhs':lhs, 'rhs':rhs, 'diff':abs(lhs-rhs), 'parity_holds':abs(lhs-rhs)<0.01}

# --- Instantiate Pricer ---
pricer = MartingalePricing(S0, r, sigma, T)

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["Martingale Verification", "European Option Pricing", "Put-Call Parity", "Exotic Options"])

with tab1:
    st.subheader("Martingale Property Verification")
    time_grid, discounted_paths, expected_values = pricer.verify_martingale_property(N, num_paths)
    fig, ax = plt.subplots(figsize=(12,5))
    for i in range(min(50, discounted_paths.shape[0])):
        ax.plot(time_grid, discounted_paths[i,:], alpha=0.3, linewidth=0.5)
    ax.plot(time_grid, expected_values, 'b-', linewidth=2, label=r'$E_Q[S^*_t]$')
    ax.axhline(y=S0, color='r', linestyle='--', linewidth=2, label=r'$S_0$')
    ax.set_xlabel("Time")
    ax.set_ylabel("Discounted Stock Price")
    ax.set_title("Sample Paths and Expected Discounted Price")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with tab2:
    st.subheader("European Option Pricing (Monte Carlo)")
    call_price, call_se = pricer.price_european_option_mc(K, 'call', M_MC)
    put_price, put_se = pricer.price_european_option_mc(K, 'put', M_MC)
    st.write(f"Call Price: ${call_price:.4f} ± ${1.96*call_se:.4f} (95% CI)")
    st.write(f"Put Price: ${put_price:.4f} ± ${1.96*put_se:.4f} (95% CI)")

with tab3:
    st.subheader("Put-Call Parity Verification")
    parity = pricer.verify_put_call_parity(K, M_MC)
    st.write(f"LHS (C-P) = {parity['lhs']:.4f}")
    st.write(f"RHS (S-K*exp(-rT)) = {parity['rhs']:.4f}")
    st.write(f"Difference = {parity['diff']:.4f}")
    st.write(f"Parity holds? {parity['parity_holds']}")

with tab4:
    st.subheader("Exotic Option Pricing")
    exotic_prices = pricer.price_exotic_options(K, M_MC)
    for name, price in exotic_prices.items():
        st.write(f"{name}: ${price:.4f}")
