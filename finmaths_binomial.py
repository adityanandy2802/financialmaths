import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Binomial Option Pricing Class
# ------------------------------
class BinomialOptionPricing:
    def __init__(self, S0: float, K: float, T: float, r: float, 
                 sigma: float = None, u: float = None, d: float = None, N: int = 1):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.N = N
        self.dt = T / N

        if u is None or d is None:
            if sigma is None:
                raise ValueError("Provide either (u, d) or sigma")
            self.u = np.exp(sigma * np.sqrt(self.dt))
            self.d = np.exp(-sigma * np.sqrt(self.dt))
        else:
            self.u = u
            self.d = d

        p_num = np.exp(r * self.dt) - self.d
        p_den = self.u - self.d
        if p_den == 0:
            raise ValueError("u and d cannot be equal")
        self.p_star = p_num / p_den

    def build_stock_tree(self):
        stock_tree = np.zeros((self.N + 1, self.N + 1))
        for i in range(self.N + 1):
            for j in range(i + 1):
                stock_tree[i, j] = self.S0 * (self.u ** j) * (self.d ** (i - j))
        return stock_tree

    def price_european_call(self):
        stock_tree = self.build_stock_tree()
        option_tree = np.zeros_like(stock_tree)
        for j in range(self.N + 1):
            option_tree[self.N, j] = max(stock_tree[self.N, j] - self.K, 0)
        disc = np.exp(-self.r * self.dt)
        for i in range(self.N - 1, -1, -1):
            for j in range(i + 1):
                option_tree[i, j] = disc * (self.p_star * option_tree[i + 1, j + 1] +
                                            (1 - self.p_star) * option_tree[i + 1, j])
        return option_tree[0, 0], option_tree

    def price_european_put(self):
        stock_tree = self.build_stock_tree()
        option_tree = np.zeros_like(stock_tree)
        for j in range(self.N + 1):
            option_tree[self.N, j] = max(self.K - stock_tree[self.N, j], 0)
        disc = np.exp(-self.r * self.dt)
        for i in range(self.N - 1, -1, -1):
            for j in range(i + 1):
                option_tree[i, j] = disc * (self.p_star * option_tree[i + 1, j + 1] +
                                            (1 - self.p_star) * option_tree[i + 1, j])
        return option_tree[0, 0], option_tree

    def price_american_put(self):
        stock_tree = self.build_stock_tree()
        option_tree = np.zeros_like(stock_tree)
        exercise_tree = np.zeros_like(stock_tree, dtype=bool)
        for j in range(self.N + 1):
            option_tree[self.N, j] = max(self.K - stock_tree[self.N, j], 0)
        disc = np.exp(-self.r * self.dt)
        for i in range(self.N - 1, -1, -1):
            for j in range(i + 1):
                hold = disc * (self.p_star * option_tree[i + 1, j + 1] +
                               (1 - self.p_star) * option_tree[i + 1, j])
                exer = max(self.K - stock_tree[i, j], 0)
                option_tree[i, j] = max(hold, exer)
                if exer > hold:
                    exercise_tree[i, j] = True
        return option_tree[0, 0], option_tree, exercise_tree

    def plot_tree(self, stock_tree, option_tree=None, exercise_tree=None):
        plt.figure(figsize=(10, 6))
        N = self.N
        for i in range(N + 1):
            for j in range(i + 1):
                x, y = i, 2 * j - i
                plt.plot(x, y, 'bo')
                plt.text(x, y + 0.3, f'{stock_tree[i, j]:.2f}', ha='center', fontsize=8)
                if option_tree is not None:
                    plt.text(x, y - 0.3, f'{option_tree[i, j]:.2f}', ha='center', fontsize=8, color='red')
                if exercise_tree is not None and exercise_tree[i, j]:
                    plt.plot(x, y, 'rs', markersize=8)
                if i < N:
                    plt.plot([x, x + 1], [y, y + 1], 'b-', alpha=0.3)
                    plt.plot([x, x + 1], [y, y - 1], 'b-', alpha=0.3)
        plt.xlabel("Time Step")
        plt.ylabel("State (2j-i)")
        plt.title(f"Binomial Tree (N={N})")
        plt.grid(True, alpha=0.3)
        st.pyplot(plt.gcf())
        plt.close()


# ------------------------------
# Streamlit App
# ------------------------------
st.title("Binomial Option Pricing Model")

option_type = st.selectbox("Select Option Type", ["European Call", "European Put", "American Put"])

S0 = st.number_input("Initial Stock Price (S0)", value=100.0)
K = st.number_input("Strike Price (K)", value=100.0)
T = st.number_input("Time to Maturity (T, in years)", value=1.0)
r = st.number_input("Risk-free Rate (r)", value=0.05)
sigma = st.number_input("Volatility (σ)", value=0.2)
N = st.slider("Number of Binomial Steps (N)", min_value=1, max_value=50, value=5)

# Instantiate model
model = BinomialOptionPricing(S0=S0, K=K, T=T, r=r, sigma=sigma, N=N)

# Compute prices
if option_type == "European Call":
    price, option_tree = model.price_european_call()
    st.success(f"European Call Price: ${price:.4f}")
    stock_tree = model.build_stock_tree()
    model.plot_tree(stock_tree, option_tree=option_tree)
elif option_type == "European Put":
    price, option_tree = model.price_european_put()
    st.success(f"European Put Price: ${price:.4f}")
    stock_tree = model.build_stock_tree()
    model.plot_tree(stock_tree, option_tree=option_tree)
else:  # American Put
    price, option_tree, exercise_tree = model.price_american_put()
    st.success(f"American Put Price: ${price:.4f}")
    stock_tree = model.build_stock_tree()
    model.plot_tree(stock_tree, option_tree=option_tree, exercise_tree=exercise_tree)
