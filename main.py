import streamlit as st

# Page config
st.set_page_config(page_title="Unified Finance Dashboard", layout="wide")

# Main title
st.markdown(
    """
    <h1 style='text-align: center; color: #ffffff;'>
        Financial Mathematics Project
    </h1>
    <p style='text-align: center; color: #ffffff; font-size:18px;'>
        Select a financial simulator.
    </p>
    """,
    unsafe_allow_html=True
)

# Dictionary of models and URLs
models = {
    "Martingale Simulator": "https://financialmaths-martingale.streamlit.app/",
    "Binomial Process": "https://financialmaths-binomial.streamlit.app/",
    "Brownian Process": "https://financialmaths-brownian.streamlit.app/",
    "Black Scholes Model": "https://financialmaths-bsm.streamlit.app/"
}

# Create columns for buttons
cols = st.columns(2)
i = 0
for model_name, url in models.items():
    with cols[i % 2]:
        st.markdown(
            f"""
            <div style="
                background-color:#1f1f1f;
                padding:25px;
                margin:15px 0px;
                border-radius:12px;
                text-align:center;
                box-shadow: 3px 3px 8px rgba(0,0,0,0.3);
            ">
                <h2 style="color:#ffffff;">{model_name}</h2>
                <a href="{url}" target="_blank">
                    <button style="
                        background-color:#ff6f00;
                        color:white;
                        border:none;
                        padding:12px 25px;
                        font-size:17px;
                        border-radius:6px;
                        cursor:pointer;
                        font-weight:bold;
                        transition: all 0.3s ease;
                    " onmouseover="this.style.backgroundColor='#ff8f33'" onmouseout="this.style.backgroundColor='#ff6f00'">
                        Open App
                    </button>
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )
    i += 1
