import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

st.set_page_config(page_title="Black-Scholes & Binomial Option Pricing", layout="wide")
st.title("Black-Scholes Option Pricing, Greeks, and Binomial Convergence")
st.write("Interactive demonstration of Black-Scholes option pricing, implied volatility, Greeks, and binomial model convergence.")

# --- Sidebar: Inputs ---
st.sidebar.header("Model Parameters")
S0 = st.sidebar.number_input("Stock Price (S0)", value=100.0, step=1.0)
K = st.sidebar.number_input("Strike Price (K)", value=100.0, step=1.0)
T = st.sidebar.number_input("Time to Maturity (T in years)", value=1.0, step=0.1)
r = st.sidebar.number_input("Risk-free Rate (r)", value=0.05, step=0.01)
sigma = st.sidebar.number_input("Volatility (sigma)", value=0.2, step=0.01)
M_MC = st.sidebar.slider("Monte Carlo Simulations (for IV)", min_value=1000, max_value=500000, value=50000, step=5000)

# --- Black-Scholes Model Class ---
class BlackScholesModel:
    def __init__(self, S0, K, T, r, sigma):
        self.S0, self.K, self.T, self.r, self.sigma = S0, K, T, r, sigma
    def d1(self): 
        return (np.log(self.S0/self.K) + (self.r + 0.5*self.sigma**2)*self.T) / (self.sigma*np.sqrt(self.T)) if self.T>0 and self.sigma>0 else np.inf
    def d2(self): return self.d1() - self.sigma*np.sqrt(self.T)
    def call_price(self): return max(self.S0 - self.K,0) if self.T==0 else self.S0*norm.cdf(self.d1()) - self.K*np.exp(-self.r*self.T)*norm.cdf(self.d2())
    def put_price(self): return max(self.K - self.S0,0) if self.T==0 else self.K*np.exp(-self.r*self.T)*norm.cdf(-self.d2()) - self.S0*norm.cdf(-self.d1())
    def call_delta(self): return norm.cdf(self.d1())
    def put_delta(self): return norm.cdf(self.d1())-1
    def gamma(self): return 0 if self.T==0 or self.sigma==0 else norm.pdf(self.d1())/(self.S0*self.sigma*np.sqrt(self.T))
    def vega(self): return 0 if self.T==0 else self.S0*norm.pdf(self.d1())*np.sqrt(self.T)
    def call_theta(self): 
        if self.T==0: return 0
        d1,d2 = self.d1(), self.d2()
        return (-(self.S0*norm.pdf(d1)*self.sigma)/(2*np.sqrt(self.T)) - self.r*self.K*np.exp(-self.r*self.T)*norm.cdf(d2))/365
    def put_theta(self):
        if self.T==0: return 0
        d1,d2 = self.d1(), self.d2()
        return (-(self.S0*norm.pdf(d1)*self.sigma)/(2*np.sqrt(self.T)) + self.r*self.K*np.exp(-self.r*self.T)*norm.cdf(-d2))/365
    def call_rho(self): return self.K*self.T*np.exp(-self.r*self.T)*norm.cdf(self.d2())
    def put_rho(self): return -self.K*self.T*np.exp(-self.r*self.T)*norm.cdf(-self.d2())
    def all_greeks(self, option_type='call'):
        if option_type=='call': return {'price':self.call_price(),'delta':self.call_delta(),'gamma':self.gamma(),'vega':self.vega(),'theta':self.call_theta(),'rho':self.call_rho()}
        else: return {'price':self.put_price(),'delta':self.put_delta(),'gamma':self.gamma(),'vega':self.vega(),'theta':self.put_theta(),'rho':self.put_rho()}
    @staticmethod
    def implied_volatility(option_price,S0,K,T,r,option_type='call'):
        def obj(sigma):
            if sigma<=0: return 1e18
            m=BlackScholesModel(S0,K,T,r,sigma)
            return (m.call_price() if option_type=='call' else m.put_price())-option_price
        try: return brentq(obj,1e-6,5.0)
        except: return np.nan

# --- Functions for Binomial Convergence & Greeks Plot ---
def compare_binomial_to_bs(S0,K,T,r,sigma,periods=None):
    if periods is None: periods=[10,20,50,100,200,500,1000]
    bs=BlackScholesModel(S0,K,T,r,sigma)
    bs_call, bs_put=bs.call_price(), bs.put_price()
    call_prices, put_prices=[],[]
    for N in periods:
        dt=T/N; u=np.exp(sigma*np.sqrt(dt)); d=1/u; p_star=(np.exp(r*dt)-d)/(u-d); discount=np.exp(-r*dt)
        call_payoffs=np.array([max(S0*(u**j)*(d**(N-j))-K,0) for j in range(N+1)])
        for i in range(N-1,-1,-1): call_payoffs=discount*(p_star*call_payoffs[1:i+2]+(1-p_star)*call_payoffs[0:i+1])
        call_prices.append(call_payoffs[0])
        put_payoffs=np.array([max(K-S0*(u**j)*(d**(N-j)),0) for j in range(N+1)])
        for i in range(N-1,-1,-1): put_payoffs=discount*(p_star*put_payoffs[1:i+2]+(1-p_star)*put_payoffs[0:i+1])
        put_prices.append(put_payoffs[0])
    return periods, call_prices, put_prices, bs_call, bs_put

def plot_binomial_convergence(periods, call_prices, put_prices, bs_call, bs_put):
    call_errors=np.abs(np.array(call_prices)-bs_call)
    put_errors=np.abs(np.array(put_prices)-bs_put)
    fig, axes=plt.subplots(1,2,figsize=(16,6))
    axes[0].plot(periods,call_prices,'bo-',label='Binomial Call'); axes[0].axhline(y=bs_call,color='b',linestyle='--',label=f'BS Call {bs_call:.2f}')
    axes[0].plot(periods,put_prices,'ro-',label='Binomial Put'); axes[0].axhline(y=bs_put,color='r',linestyle='--',label=f'BS Put {bs_put:.2f}')
    axes[0].set_xscale('log'); axes[0].set_title('Price Convergence'); axes[0].set_xlabel('Time Steps N'); axes[0].set_ylabel('Price'); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(periods,call_errors,'b^-',label='Call Error'); axes[1].plot(periods,put_errors,'r^-',label='Put Error')
    axes[1].set_xscale('log'); axes[1].set_yscale('log'); axes[1].set_title('Absolute Error'); axes[1].set_xlabel('Time Steps N'); axes[1].set_ylabel('|BOPM - BS|'); axes[1].legend(); axes[1].grid(True)
    plt.tight_layout(); return fig

def plot_greeks_sensitivity(S0,K,T,r,sigma):
    S_range=np.linspace(0.5*K,1.5*K,100)
    call_prices, put_prices, call_deltas, put_deltas, gammas, vegas, call_thetas, put_thetas=[],[],[],[],[],[],[],[]
    for S in S_range:
        bs=BlackScholesModel(S,K,T,r,sigma)
        call_prices.append(bs.call_price()); put_prices.append(bs.put_price())
        call_deltas.append(bs.call_delta()); put_deltas.append(bs.put_delta())
        gammas.append(bs.gamma()); vegas.append(bs.vega())
        call_thetas.append(bs.call_theta()); put_thetas.append(bs.put_theta())
    fig, axes=plt.subplots(2,3,figsize=(18,10))
    axes[0,0].plot(S_range,call_prices,'b-',label='Call'); axes[0,0].plot(S_range,put_prices,'r-',label='Put'); axes[0,0].axvline(K,color='k',linestyle='--'); axes[0,0].set_title('Option Prices'); axes[0,0].legend(); axes[0,0].grid(True)
    axes[0,1].plot(S_range,call_deltas,'b-',label='Call Δ'); axes[0,1].plot(S_range,put_deltas,'r-',label='Put Δ'); axes[0,1].axvline(K,color='k',linestyle='--'); axes[0,1].set_title('Delta'); axes[0,1].legend(); axes[0,1].grid(True)
    axes[0,2].plot(S_range,gammas,'g-'); axes[0,2].axvline(K,color='k',linestyle='--'); axes[0,2].set_title('Gamma'); axes[0,2].grid(True)
    axes[1,0].plot(S_range,vegas,'m-'); axes[1,0].axvline(K,color='k',linestyle='--'); axes[1,0].set_title('Vega'); axes[1,0].grid(True)
    axes[1,1].plot(S_range,call_thetas,'b-',label='Call Θ'); axes[1,1].plot(S_range,put_thetas,'r-',label='Put Θ'); axes[1,1].axvline(K,color='k',linestyle='--'); axes[1,1].set_title('Theta'); axes[1,1].legend(); axes[1,1].grid(True)
    axes[1,2].set_visible(False)
    plt.tight_layout(); return fig

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["Black-Scholes & Greeks", "Binomial Convergence", "Implied Volatility"])

with tab1:
    st.subheader("Black-Scholes Prices and Greeks")
    bs=BlackScholesModel(S0,K,T,r,sigma)
    call_greeks=bs.all_greeks('call'); put_greeks=bs.all_greeks('put')
    st.write("Call Greeks & Price:", call_greeks)
    st.write("Put Greeks & Price:", put_greeks)
    parity_lhs=call_greeks['price']-put_greeks['price']; parity_rhs=S0-K*np.exp(-r*T)
    st.write(f"Put-Call Parity Check: C-P={parity_lhs:.4f}, S-Ke^(-rT)={parity_rhs:.4f}")
    fig=plot_greeks_sensitivity(S0,K,T,r,sigma)
    st.pyplot(fig)

with tab2:
    st.subheader("Binomial Convergence to Black-Scholes")
    periods, call_prices, put_prices, bs_call, bs_put=compare_binomial_to_bs(S0,K,T,r,sigma)
    st.write("Black-Scholes Call:", bs_call, "Put:", bs_put)
    fig=plot_binomial_convergence(periods,call_prices,put_prices,bs_call,bs_put)
    st.pyplot(fig)

with tab3:
    st.subheader("Implied Volatility Calculator")
    market_price=st.number_input("Market Option Price", value=bs.call_price())
    option_type=st.selectbox("Option Type", ['call','put'])
    implied_vol=BlackScholesModel.implied_volatility(market_price,S0,K,T,r,option_type)
    st.write(f"Implied Volatility: {implied_vol:.4f}")
