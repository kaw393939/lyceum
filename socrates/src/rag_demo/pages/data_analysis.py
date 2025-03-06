"""Data analysis page for Streamlit multi-page app."""
import streamlit as st
import pandas as pd
import numpy as np
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Data Analysis", page_icon="📊")

st.title("Data Analysis")
st.markdown("Perform basic analysis on your data.")

# Generate some sample data
@st.cache_data
def get_data() -> pd.DataFrame:
    """Generate sample data with caching.
    
    Returns:
        pd.DataFrame: Sample dataframe
    """
    logger.info("Generating cached data for analysis page")
    np.random.seed(42)
    data = {
        "category": np.random.choice(["A", "B", "C", "D"], size=100),
        "value1": np.random.randn(100),
        "value2": np.random.randn(100) * 2 + 1,
        "date": pd.date_range(start="2023-01-01", periods=100)
    }
    return pd.DataFrame(data)

data = get_data()

# Allow file upload for real data
uploaded_file = st.file_uploader("Upload your own CSV data", type=["csv"])
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        logger.info(f"User uploaded file: {uploaded_file.name}")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        logger.error(f"File upload error: {e}")

# Display data statistics
st.subheader("Data Overview")
st.dataframe(data.head())

col1, col2 = st.columns(2)
with col1:
    st.subheader("Summary Statistics")
    st.dataframe(data.describe())

with col2:
    if "category" in data.columns:
        st.subheader("Category Distribution")
        cat_counts = data["category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        st.bar_chart(cat_counts.set_index("Category"))

# Advanced analysis
st.subheader("Advanced Analysis")
numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns.tolist()

if len(numeric_cols) >= 2:
    selected_cols = st.multiselect(
        "Select columns for correlation analysis",
        options=numeric_cols,
        default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols
    )
    
    if len(selected_cols) >= 2:
        st.subheader("Correlation Matrix")
        corr = data[selected_cols].corr()
        st.dataframe(corr.style.background_gradient(cmap="coolwarm"))
        
        st.subheader("Scatter Plot")
        x_col = st.selectbox("X axis", options=selected_cols, index=0)
        y_col = st.selectbox("Y axis", options=selected_cols, index=min(1, len(selected_cols)-1))
        
        scatter_data = pd.DataFrame({
            "x": data[x_col],
            "y": data[y_col]
        })
        st.scatter_chart(scatter_data.set_index("x"))
