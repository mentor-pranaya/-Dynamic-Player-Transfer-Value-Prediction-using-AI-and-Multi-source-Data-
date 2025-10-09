import streamlit as st
import pandas as pd
import joblib

# --- 1. Load the Saved Model and Artifacts ---
@st.cache_resource
def load_model():
    model = joblib.load('final_player_value_model.joblib')
    model_columns = joblib.load('model_columns.joblib')
    return model, model_columns

@st.cache_data
def load_player_data():
    df = pd.read_csv('data/final_top_10_player_data.csv')
    return df

model, model_columns = load_model()
player_df = load_player_data()


# --- 2. Set Up the Page Configuration ---
st.set_page_config(
    page_title="Player Value Predictor",
    page_icon="⚽",
    layout="wide"
)

# REMOVED: The background image function has been commented out to fix the error.
# def set_bg_image():
#     st.markdown(
#         f"""
#         <style>
#         .stApp {{
#             background-image: url("...");
#             background-size: cover;
#             background-repeat: no-repeat;
#             background-attachment: fixed;
#         }}
#         </style>
#         """,
#         unsafe_allow_html=True
#     )
# set_bg_image()


# --- 3. Set Up the User Interface (UI) ---
st.title('Dynamic Player Transfer Value Prediction ⚽')
st.write("This app predicts the market value of a football player based on their performance metrics.")

st.sidebar.header('Player Input Features')

# Categorical inputs for filtering
nationality = st.sidebar.selectbox('Nationality', player_df['Nationality'].unique())

st.sidebar.subheader("Players from selected country:")
players_in_country = player_df[player_df['Nationality'] == nationality]['player_name'].tolist()
for player in players_in_country:
    st.sidebar.write(f"- {player}")

st.sidebar.markdown("---")
goals = st.sidebar.number_input('Goals 🥅', min_value=0, max_value=50, value=15)
assists = st.sidebar.number_input('Assists 🤝', min_value=0, max_value=50, value=10)
successful_passes = st.sidebar.number_input('Successful Passes 🎯', min_value=0, max_value=3000, value=1500)
tackles_won = st.sidebar.number_input('Tackles Won 💪', min_value=0, max_value=100, value=30)
avg_sentiment_score = st.sidebar.slider('Average Sentiment Score 👍', min_value=-1.0, max_value=1.0, value=0.2, step=0.01)
injury_risk_score = st.sidebar.slider('Injury Risk Score 🤕', min_value=0.0, max_value=2.0, value=0.5, step=0.1)
position = st.sidebar.selectbox('Position', ['Forward', 'Midfielder', 'Winger'])


# --- 4. The Prediction Logic ---
if st.button('Predict Market Value'):
    input_data = {
        'goals': goals,
        'assists': assists,
        'successful_passes': successful_passes,
        'tackles_won': tackles_won,
        'avg_sentiment_score': avg_sentiment_score,
        'injury_risk_score': injury_risk_score,
        'position_' + position: 1,
        'Nationality_' + nationality: 1,
    }
    
    input_df = pd.DataFrame([input_data])
    final_input_df = input_df.reindex(columns=model_columns, fill_value=0)
    
    prediction = model.predict(final_input_df)
    
    st.subheader('Predicted Market Value')
    st.metric(label="Value in Millions (€) 💰", value=f"{prediction[0]:,.2f}")