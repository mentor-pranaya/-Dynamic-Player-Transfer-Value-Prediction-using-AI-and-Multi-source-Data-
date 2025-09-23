import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
import pickle
import os

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="TransferIQ", layout="wide")

# -------------------------------
# Load Prediction Model
# -------------------------------
model = None
model_path = os.path.join("..", "Backend", "transferiq_model.pkl")
try:
    if os.path.exists(model_path):
        model = pickle.load(open(model_path, "rb"))
    else:
        st.sidebar.warning("⚠️ Model not found, using dummy prediction logic.")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    model = None


# -------------------------------
# Custom Dynamic CSS Injector
# -------------------------------
def inject_dynamic_css(primary_color, accent_color, bg_color, card_color, text_color,
                       font_family, base_font_size, border_radius, shadow_value,
                       gradient_overlay, container_width, center_content):
    dynamic_css = f"""
    <style>
    :root {{
      --primary: {primary_color};
      --accent: {accent_color};
      --bg: {bg_color};
      --card: {card_color};
      --text: {text_color};
      --font: {font_family};
      --base-font-size: {base_font_size}px;
      --radius: {border_radius}px;
      --card-shadow: {shadow_value};
      --max-width: {container_width}px;
    }}

    html, body {{
      background: var(--bg);
      font-family: var(--font);
      color: var(--text);
      font-size: var(--base-font-size);
      transition: all 0.3s ease-in-out;
    }}

    section.main > div {{
      max-width: var(--max-width);
      {"margin: 0 auto;" if center_content else ""}
    }}

    section[data-testid="stSidebar"] > div {{
      background: linear-gradient(135deg, var(--primary), var(--accent));
      color: white !important;
      padding: 15px;
    }}

    .header-card {{
      background: {gradient_overlay};
      padding: 24px;
      border-radius: var(--radius);
      box-shadow: var(--card-shadow);
      color: white;
      margin-bottom: 20px;
      transition: all 0.3s ease-in-out;
    }}

    .control-card {{
      background: var(--card);
      padding: 20px;
      border-radius: var(--radius);
      box-shadow: var(--card-shadow);
      margin-bottom: 16px;
      transition: all 0.3s ease-in-out;
    }}

    .stButton>button {{
      background: linear-gradient(90deg, var(--primary), var(--accent));
      color: white !important;
      border: none;
      padding: 10px 18px;
      border-radius: calc(var(--radius) - 2px);
      font-size: var(--base-font-size);
      font-weight: 600;
      cursor: pointer;
      transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    }}
    .stButton>button:hover {{
      transform: translateY(-2px);
      box-shadow: 0 6px 12px rgba(0,0,0,0.25);
    }}

    .result-box {{
      background: linear-gradient(135deg, var(--primary), var(--accent));
      padding: 18px;
      border-radius: var(--radius);
      box-shadow: 0 6px 14px rgba(0,0,0,0.35);
      color: white;
      font-size: calc(var(--base-font-size) + 4px);
      font-weight: 600;
      text-align: center;
      margin-top: 16px;
      transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}

    .result-box strong {{
      display: block;
      font-size: calc(var(--base-font-size) + 6px);
      margin-bottom: 6px;
    }}

    .result-box:hover {{
      transform: scale(1.03);
      box-shadow: 0 10px 20px rgba(0,0,0,0.45);
    }}

    [data-testid="stMetricValue"] {{
      font-size: 1.6rem;
      font-weight: bold;
      color: var(--primary);
    }}
    </style>
    """
    st.markdown(dynamic_css, unsafe_allow_html=True)


# -------------------------------
# Default Theme
# -------------------------------
inject_dynamic_css(
    primary_color="#00c6ff",
    accent_color="#0072ff",
    bg_color="#0f2027",
    card_color="#ffffff15",
    text_color="white",
    font_family="Segoe UI, sans-serif",
    base_font_size=15,
    border_radius=12,
    shadow_value="0 4px 12px rgba(0,0,0,0.3)",
    gradient_overlay="linear-gradient(135deg, #667eea, #764ba2)",
    container_width=1300,
    center_content=True
)


# -------------------------------
# Sidebar Navigation
# -------------------------------
with st.sidebar:
    selected = option_menu(
        "",
        ["🏠 Home", "⚽ Prediction", "📊 Analysis", "⚙️ Settings"],
        icons=["house", "lightbulb", "bar-chart", "gear"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical"
    )


# -------------------------------
# Home Section
# -------------------------------
if selected == "🏠 Home":
    st.markdown('<div class="header-card"><h1>✨ TransferIQ Dashboard</h1><p>Welcome to your interactive real-time data platform.</p></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Active Users", "1,248", "+4.5%")
    col2.metric("Predictions Made", "3,410", "+300")
    col3.metric("Accuracy Rate", "82%", "↑ 2%")

    data = pd.DataFrame(np.random.randn(50, 3), columns=['Feature A', 'Feature B', 'Feature C'])
    st.line_chart(data)


# -------------------------------
# Prediction Section
# -------------------------------
elif selected == "⚽ Prediction":
    st.markdown('<div class="header-card"><h2>⚽ Transfer Value Prediction</h2></div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="control-card">', unsafe_allow_html=True)
        st.subheader("Player Inputs")

        df = st.session_state.get("uploaded_df", None)
        perf_list, sentiment, injury, contract = None, None, None, None
        column_names = {}

        if df is not None:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if len(numeric_cols) >= 4:
                perf_col = st.selectbox("Select Performance Column", numeric_cols, index=0)
                sentiment_col = st.selectbox("Select Sentiment Column", numeric_cols, index=1)
                injury_col = st.selectbox("Select Injury Column", numeric_cols, index=2)
                contract_col = st.selectbox("Select Contract Column", numeric_cols, index=3)

                column_names = {
                    "perf": perf_col,
                    "sentiment": sentiment_col,
                    "injury": injury_col,
                    "contract": contract_col
                }

                row_idx = st.number_input("Select Player Row Index", 0, len(df)-1, 0)
                perf_list = [df.loc[row_idx, perf_col]]
                sentiment = df.loc[row_idx, sentiment_col]
                injury = df.loc[row_idx, injury_col]
                contract = df.loc[row_idx, contract_col]
            else:
                st.warning("⚠️ Dataset needs at least 4 numeric columns. Switching to manual entry.")
                df = None

        if df is None:
            performance = st.text_input("Performance (comma-separated numbers)", "30, 28, 25")
            sentiment = st.number_input("Sentiment Score (−1 to 1)", -1.0, 1.0, 0.5, step=0.1)
            injury = st.number_input("Injury Count", 0, 20, 1)
            contract = st.number_input("Remaining Contract (months)", 0, 60, 24)
            perf_list = [float(x.strip()) for x in performance.split(",") if x.strip()]

            column_names = {
                "perf": "Performance",
                "sentiment": "Sentiment",
                "injury": "Injury",
                "contract": "Contract"
            }

        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🔮 Predict Transfer Value"):
            try:
                if not perf_list:
                    st.error("Enter at least one performance value.")
                else:
                    perf_mean, perf_max, perf_min, perf_std = (
                        np.mean(perf_list),
                        np.max(perf_list),
                        np.min(perf_list),
                        np.std(perf_list, ddof=0)
                    )

                    features = [perf_mean, perf_max, perf_min, perf_std,
                                float(sentiment), int(injury), int(contract)]

                    contributions = None
                    if model is not None:
                        try:
                            pred = model.predict([features])[0]
                        except Exception as e:
                            st.error(f"Model error: {e}")
                            pred = (perf_mean * 0.8 + perf_max * 0.1 - perf_std * 0.05 +
                                    sentiment * 10 - injury * 1.5 + (contract/12)*2)
                    else:
                        contrib_perf_mean = perf_mean * 0.8
                        contrib_perf_max = perf_max * 0.1
                        contrib_perf_std = -perf_std * 0.05
                        contrib_sentiment = sentiment * 10
                        contrib_injury = -injury * 1.5
                        contrib_contract = (contract/12) * 2

                        pred = (contrib_perf_mean + contrib_perf_max + contrib_perf_std +
                                contrib_sentiment + contrib_injury + contrib_contract)

                        contributions = {
                            f"Avg {column_names['perf']}": contrib_perf_mean,
                            f"Peak {column_names['perf']}": contrib_perf_max,
                            f"Consistency (Std Dev {column_names['perf']})": contrib_perf_std,
                            f"{column_names['sentiment']}": contrib_sentiment,
                            f"{column_names['injury']}": contrib_injury,
                            f"{column_names['contract']}": contrib_contract
                        }

                    st.markdown(
                        f'<div class="result-box"><strong>Predicted Transfer Value</strong> Rs.{pred:.2f}M</div>',
                        unsafe_allow_html=True
                    )

                    st.subheader("📌 Summary of Prediction")
                    st.write(f"- **Average {column_names['perf']}:** {perf_mean:.2f}")
                    st.write(f"- **Peak {column_names['perf']}:** {perf_max:.2f}")
                    st.write(f"- **Consistency (Std Dev {column_names['perf']}):** {perf_std:.2f}")
                    st.write(f"- **{column_names['sentiment']}:** {sentiment}")
                    st.write(f"- **{column_names['injury']}:** {injury}")
                    st.write(f"- **{column_names['contract']}:** {contract} months")

                    if pred > 50:
                        st.success("💰 High-value transfer target.")
                    elif pred > 20:
                        st.info("📈 Moderate transfer value with room to grow.")
                    else:
                        st.warning("⚠️ Low transfer value (injuries/poor performance).")

                    if contributions:
                        st.subheader("📊 Factor Contributions")
                        contrib_df = pd.DataFrame(list(contributions.items()), columns=["Factor", "Contribution (€M)"])
                        fig, ax = plt.subplots(figsize=(8, 4))
                        sns.barplot(data=contrib_df, x="Contribution (€M)", y="Factor", palette="coolwarm", ax=ax)
                        ax.axvline(0, color="black", linewidth=1)
                        ax.set_title("Impact of Each Factor on Transfer Value")
                        plt.tight_layout()
                        st.pyplot(fig)
            except Exception as e:
                st.error(f"Error: {e}")


# -------------------------------
# Analysis Section
# -------------------------------
elif selected == "📊 Analysis":
    st.markdown('<div class="header-card"><h2>📊 Data Analysis & Visualization</h2></div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV for Analysis", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state["uploaded_df"] = df
        st.success("✅ File uploaded successfully!")

        st.write("### Data Preview")
        st.dataframe(df.head())

        st.write("### Summary Statistics")
        st.write(df.describe())

        st.write("### Visualization")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if numeric_cols:
            col1, col2 = st.columns(2)
            with col1:
                col = st.selectbox("Select column", numeric_cols, key="hist_col")
                fig, ax = plt.subplots()
                sns.histplot(df[col], kde=True, ax=ax, color="#00c6ff")
                st.pyplot(fig)

            with col2:
                if len(numeric_cols) >= 2:
                    x_col = st.selectbox("X-axis", numeric_cols, key="x_col")
                    y_col = st.selectbox("Y-axis", numeric_cols, key="y_col")
                    fig, ax = plt.subplots()
                    sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax, color="#0072ff")
                    st.pyplot(fig)
        else:
            st.warning("⚠️ No numeric columns available.")
    else:
        st.info("⬆️ Upload a CSV file to start analysis.")


# -------------------------------
# Settings Section
# -------------------------------
elif selected == "⚙️ Settings":
    st.markdown('<div class="header-card"><h2>⚙️ Settings</h2></div>', unsafe_allow_html=True)
    theme = st.radio("Choose Theme", ["Light", "Dark", "Gradient"], index=2)

    if theme == "Light":
        inject_dynamic_css("#4facfe", "#00f2fe", "white", "#f9f9f9", "black",
                           "Segoe UI, sans-serif", 15, 12, "0 4px 12px rgba(0,0,0,0.1)",
                           "linear-gradient(135deg, #43cea2, #185a9d)", 1300, True)
    elif theme == "Dark":
        inject_dynamic_css("#ff512f", "#dd2476", "#1e1e1e", "#2c2c2c", "white",
                           "Segoe UI, sans-serif", 15, 12, "0 4px 12px rgba(0,0,0,0.4)",
                           "linear-gradient(135deg, #141e30, #243b55)", 1300, True)
    else:
        inject_dynamic_css("#00c6ff", "#0072ff", "#0f2027", "#ffffff15", "white",
                           "Segoe UI, sans-serif", 15, 12, "0 4px 12px rgba(0,0,0,0.3)",
                           "linear-gradient(135deg, #667eea, #764ba2)", 1300, True)

    st.success("✅ Settings applied in real-time!")
