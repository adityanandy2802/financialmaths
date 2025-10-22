import streamlit as st
import subprocess

st.set_page_config(page_title="Option Pricing Dashboard", layout="wide")
st.title("Option Pricing Multi-App Dashboard")

st.write("Select an app from the dropdown to launch it:")

# Dropdown menu of apps
app_options = {
    "Black-Scholes & Greeks": "finmaths_bsm.py",
    "Binomial Convergence": "finmaths_binomial.py",
    "Martingale Simulator": "finmaths_martingale.py",
    "Brownian Process Simulator": "finmaths_brownian.py"
}

selected_app = st.selectbox("Choose App", list(app_options.keys()))

st.write(f"You selected **{selected_app}**.")

# Launch selected app when user clicks a button
if st.button("Launch App"):
    app_file = app_options[selected_app]
    st.write(f"Launching `{app_file}` ...")
    
    # This will open the app in a new Streamlit process (on the default port)
    # Use `subprocess.Popen` to run it asynchronously
    try:
        subprocess.Popen(["streamlit", "run", app_file])
        st.success(f"{selected_app} launched successfully! Check your browser windows.")
    except Exception as e:
        st.error(f"Failed to launch app: {e}")
