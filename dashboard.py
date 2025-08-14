import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# ========================================
# PAGE CONFIG - MUST BE FIRST STREAMLIT COMMAND
# ========================================
st.set_page_config(page_title="Predictive Dashboard", layout="wide")

# ========================================
# LOAD TRAINED MODEL
# ========================================
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ========================================
# SIDEBAR STYLING & NAVIGATION
# ========================================
st.sidebar.markdown(
    """
    <style>
    [data-testid=stSidebar] {
        background-color: #1e293b;
    }
    [data-testid=stSidebar] * {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["📤 Upload Dataset", "📈 Data Insights", "🤖 Predictions"]
)

# ========================================
# UPLOAD PAGE
# ========================================
if page == "📤 Upload Dataset":
    st.title("📤 Upload Dataset")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.success("✅ Data uploaded successfully!")
        st.write("### Preview of Data:")
        st.dataframe(st.session_state.df.head())

# ========================================
# INSIGHTS PAGE
# ========================================
elif page == "📈 Data Insights":
    st.title("📈 Data Insights")
    if "df" in st.session_state:
        df = st.session_state.df

        tab1, tab2, tab3 = st.tabs(
            ["📊 Correlation Heatmap", "📉 Feature Distributions", "⭐ Top Features"]
        )

        with tab1:
            st.subheader("Correlation Heatmap")
            numeric_df = df.select_dtypes(include=['number'])
            if not numeric_df.empty:
                corr = numeric_df.corr()
                fig, ax = plt.subplots()
                sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)
            else:
                st.warning("No numeric columns found for correlation.")

        with tab2:
            st.subheader("Feature Distributions")
            numeric_df = df.select_dtypes(include=['number'])
            if not numeric_df.empty:
                for col in numeric_df.columns:
                    fig, ax = plt.subplots()
                    sns.histplot(df[col], kde=True, ax=ax)
                    st.pyplot(fig)
            else:
                st.warning("No numeric columns found for plotting.")

        with tab3:
            st.subheader("Top Features from Model")
            try:
                if hasattr(model, "feature_importances_"):
                    features = df.select_dtypes(include=['number']).columns
                    importances = model.feature_importances_
                    feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
                    fig, ax = plt.subplots()
                    feat_imp.head(10).plot(kind="barh", ax=ax, color='skyblue')
                    ax.set_title("Top 10 Feature Importances")
                    st.pyplot(fig)
                else:
                    st.info("This model does not support feature importance.")
            except Exception as e:
                st.error(f"Error showing feature importance: {e}")

    else:
        st.warning("Please upload a dataset first.")

# ========================================
# PREDICTION PAGE
# ========================================
elif page == "🤖 Predictions":
    st.title("🤖 Make Predictions")
    if "df" in st.session_state:
        df = st.session_state.df.copy()

        try:
            # Get required columns from model
            if hasattr(model, "feature_names_in_"):
                required_cols = list(model.feature_names_in_)
            else:
                st.error("Model does not have feature name information.")
                st.stop()

            # Fill missing columns with default values (0)
            for col in required_cols:
                if col not in df.columns:
                    df[col] = 0  # default value

            # Ensure correct column order
            df_model_input = df[required_cols]

            # Make predictions
            predictions = model.predict(df_model_input)
            df['Prediction'] = predictions

            st.write("### Predictions on Uploaded Data:")
            st.dataframe(df.head())

            # Download button for predictions
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions",
                data=csv,
                file_name='predictions.csv',
                mime='text/csv',
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    else:
        st.warning("Please upload a dataset first.")
