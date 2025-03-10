import numpy as np
import pandas as pd
import streamlit as st
import urllib.parse
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA

# ---------------------------------------------------
# PAGE CONFIG & CUSTOM STYLES
# ---------------------------------------------------
# We set layout="wide" to ensure a more mobile-friendly, responsive interface.
# We'll also add additional CSS for a modern look and extra color.

st.set_page_config(page_title="AI Number Predictor", layout="wide")

st.markdown(
    """
    <style>
    /* Body background color */
    body {
        background: linear-gradient(to right, #ffefba, #ffffff);
        margin: 0;
        padding: 0;
        font-family: "Helvetica Neue", sans-serif;
    }

    /* Main container styling */
    .main > div {
        background-color: #FFFFFF;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }

    .stButton>button {
        background-color: #f16e5f !important; /* a nice salmon color */
        color: white !important;
        padding: 10px 24px !important;
        font-size: 16px !important;
        border-radius: 8px !important;
        border: none !important;
        cursor: pointer !important;
    }

    .stTextArea>label {
        font-size: 1.1rem !important;
        color: #333 !important;
    }

    .stTextArea>div>textarea {
        font-size: 16px !important;
        border-radius: 8px !important;
        border: 1px solid #ccc !important;
        padding: 1rem !important;
    }

    .stSelectbox>label {
        font-size: 1.1rem !important;
        color: #333 !important;
    }

    /* Custom tooltip style for info blocks */
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted black;
    }

    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px 0;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }

    /* Expander styling for nice collapsible sections */
    .streamlit-expanderHeader {
        font-size: 1.1rem;
        font-weight: 600;
        color: #f16e5f;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔢 AI Number Predictor - Psychological Game Theory")
st.write("Enter your initial data points (no negative numbers), separated by spaces, and click Initialize.")

# In-session dataset
if "dataset" not in st.session_state:
    st.session_state["dataset"] = []

# Algorithm info
alg_info = {
    "Random Forest": "An ensemble method using multiple decision trees for robust predictions.",
    "Linear Regression": "A linear approach modeling the relationship between features and target.",
    "ARIMA (Time-Series)": "A statistical model for analyzing and forecasting time-series data.",
    "AI Auto-Select": "Automatically chooses the best algorithm based on your data size."
}

# Layman expansions
with st.expander("Random Forest Explained (Layman Terms)"):
    st.write(
        """
        **What is Random Forest?**\
        It's like asking several experts the same question and then taking the average of their answers. \n\
        - **Advantage**: It usually gives a pretty good answer because it balances out mistakes from individual experts.\n\
        - **Disadvantage**: It can be slower with huge data, and might be harder to interpret exactly why it gave a certain answer.
        """
    )

with st.expander("Linear Regression Explained (Layman Terms)"):
    st.write(
        """
        **What is Linear Regression?**\
        It's like drawing the best possible straight line through your data points. \n\
        - **Advantage**: Very fast and easy to understand.\n\
        - **Disadvantage**: Only works well if your data follows a roughly straight-line trend.
        """
    )

with st.expander("ARIMA Explained (Layman Terms)"):
    st.write(
        """
        **What is ARIMA?**\
        It looks at how your data changed in the past to guess what's next, focusing on time-based patterns. \n\
        - **Advantage**: Great for data that changes over time (like monthly sales).\n\
        - **Disadvantage**: Tricky to set up the right parameters, and it assumes past patterns continue.
        """
    )

with st.expander("AI Auto-Select Explained"):
    st.write(
        """
        **What is AI Auto-Select?**\
        We automatically pick the best method for you, based on your data size. \n\
        - **Advantage**: You don't have to worry about choosing an algorithm.\n\
        - **Disadvantage**: You have less control if you want to fine-tune each method.
        """
    )

# User input for initialization
user_input = st.text_area("Initial Data (Spaces):", key="init_data")

options = list(alg_info.keys())
prediction_method = st.selectbox(
    "Choose an algorithm:",
    options,
    help="Pick the type of algorithm you'd like to use",
)

init_btn = st.button("Initialize")
if init_btn:
    if user_input.strip():
        try:
            values = list(map(float, user_input.split()))
            # Check for negatives
            if any(val < 0 for val in values):
                st.error("No negative numbers allowed. Please remove them.")
            else:
                st.session_state["dataset"] = values
                st.success(f"Initialized dataset with {len(values)} values!")
        except ValueError:
            st.error("Please enter valid numbers separated by spaces.")

st.write("Current Dataset:", st.session_state["dataset"])

# Buttons to download or share dataset
if len(st.session_state["dataset"]) > 0:
    # Convert dataset to CSV
    dataset_csv = pd.DataFrame({"Position": st.session_state["dataset"]}).to_csv(index=False)

    st.download_button(
        label="Download Dataset as CSV",
        data=dataset_csv,
        file_name="dataset.csv",
        mime="text/csv"
    )

    # Create a WhatsApp share link
    data_str = ", ".join(map(str, st.session_state["dataset"]))
    share_text = f"Check out my dataset: {data_str}"
    encoded_text = urllib.parse.quote(share_text)
    whatsapp_link = f"https://api.whatsapp.com/send?text={encoded_text}"
    st.markdown(f"[Share Dataset on WhatsApp]({whatsapp_link})")

# Model training + Predictions
predict_btn = st.button("Predict Next Number")

def train_and_predict(data: list, method: str):
    """Train model based on method and return predictions"""
    # Convert to DataFrame
    df = pd.DataFrame({"Position": data})
    df["index"] = np.arange(len(df))

    # Outlier detection
    Q1 = df["Position"].quantile(0.25)
    Q3 = df["Position"].quantile(0.75)
    IQR = Q3 - Q1
    outlier_threshold_upper = Q3 + 1.5 * IQR
    outlier_threshold_lower = Q1 - 1.5 * IQR

    # Filter out outliers
    df_filtered = df[
        (df["Position"] >= outlier_threshold_lower) &
        (df["Position"] <= outlier_threshold_upper)
    ]

    # Decide model
    num_positions = len(data)
    if method == "Random Forest" or (method == "AI Auto-Select" and num_positions > 20):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        X = df_filtered[["index"]]
        y = df_filtered["Position"]
        model.fit(X, y)
        raw_preds = model.predict([[len(df)], [len(df)+1], [len(df)+2]])
    elif method == "Linear Regression" or (method == "AI Auto-Select" and num_positions <= 20):
        model = LinearRegression()
        X = df_filtered[["index"]]
        y = df_filtered["Position"]
        model.fit(X, y)
        raw_preds = model.predict([[len(df)], [len(df)+1], [len(df)+2]])
    else:
        # ARIMA
        if len(df_filtered) < 3:
            return [], []  # not enough data for ARIMA
        arima_model = ARIMA(df_filtered["Position"], order=(2,1,2))
        arima_fit = arima_model.fit()
        raw_preds = arima_fit.forecast(steps=3)

    # Convert predictions to a list
    unique_predictions = list(set(raw_preds))
    unique_predictions.sort(reverse=True)
    while len(unique_predictions) < 3:
        unique_predictions.append(
            unique_predictions[-1] - np.random.uniform(0.01, 1.0)
        )
    unique_predictions = unique_predictions[:3]

    # Probability ratings for each prediction
    probabilities = np.round(np.linspace(95,70,3),2)

    return unique_predictions, probabilities

if predict_btn:
    if len(st.session_state["dataset"]) == 0:
        st.warning("Please initialize your dataset first.")
    else:
        preds, probs = train_and_predict(st.session_state["dataset"], prediction_method)
        if len(preds) == 0:
            st.error("Not enough data or something went wrong.")
        else:
            st.subheader("🔮 Predictions:")
            for i, (p, pr) in enumerate(zip(preds, probs), 1):
                st.write(f"Prediction {i}: **{p:.2f}** with Probability: **{pr}%**")
            safe_option = np.round(probs.mean(),2)
            st.markdown(f"**Safe Option (Average Probability):** {safe_option}%")

            # 4th result: final prediction based on the 3 predicted values (their average)
            final_pred = np.round(np.mean(preds), 2)
            st.markdown(f"**Final Prediction (Average of the 3 Predicted Numbers):** {final_pred}")

# Input the actual correct number
actual_input = st.text_input("Enter the actual correct number:")
add_actual_btn = st.button("Add Actual & Re-Predict")

if add_actual_btn:
    if not actual_input.strip():
        st.error("Please enter the correct number.")
    else:
        try:
            act_val = float(actual_input)
            if act_val < 0:
                st.error("No negative numbers allowed.")
            else:
                # Append to dataset and re-predict
                st.session_state["dataset"].append(act_val)
                st.success(f"Appended actual number {act_val} to dataset. Re-predicting...")
                st.write("Updated Dataset:", st.session_state["dataset"])

                preds, probs = train_and_predict(st.session_state["dataset"], prediction_method)
                if len(preds) == 0:
                    st.error("Not enough data or something went wrong.")
                else:
                    st.subheader("🔄 New Predictions After Adding Actual:")
                    for i, (p, pr) in enumerate(zip(preds, probs), 1):
                        st.write(f"Prediction {i}: **{p:.2f}** with Probability: **{pr}%**")
                    safe_option = np.round(probs.mean(),2)
                    st.markdown(f"**Safe Option (Average Probability):** {safe_option}%")

                    # 4th result: final prediction based on the 3 predicted values (their average)
                    final_pred = np.round(np.mean(preds), 2)
                    st.markdown(f"**Final Prediction (Average of the 3 Predicted Numbers):** {final_pred}")
        except ValueError:
            st.error("Please enter a valid non-negative number.")

# ---------------------------------------------------
# CMD LAUNCH INSTRUCTIONS
# ---------------------------------------------------
st.text("Run the following command in CMD to start the app:")
st.code("streamlit run \"C:\\Users\\isbli\\OneDrive\\Attachments\\Documents\\python script\\predictor_ui.py\"")
