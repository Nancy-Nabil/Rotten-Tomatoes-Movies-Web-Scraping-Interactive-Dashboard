import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------- Page Setup ----------------
st.set_page_config(layout="wide", page_title="🎬 Rotten Tomatoes Movies Dashboard")
st.title("🎬 Rotten Tomatoes Movies Dashboard")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("📂 Upload Your Data")
    uploaded_file = st.file_uploader("Upload a Rotten Tomatoes CSV", type="csv")

    st.header("🎨 Theme Settings")
    theme = st.selectbox("Plot Theme", ['darkgrid', 'whitegrid', 'dark', 'white', 'ticks'])
    sns.set_style(theme)

# ---------------- Runtime Conversion ----------------
def convert_runtime_to_minutes(runtime_str):
    try:
        runtime_str = str(runtime_str).lower().strip()
        hours = minutes = 0
        if 'h' in runtime_str:
            parts = runtime_str.replace(' ', '').split('h')
            hours = int(parts[0])
            if len(parts) > 1 and 'm' in parts[1]:
                minutes = int(parts[1].replace('m', ''))
        elif 'm' in runtime_str:
            minutes = int(runtime_str.replace('m', ''))
        return hours * 60 + minutes
    except:
        return np.nan

def clean_critic_score(score_str):
    try:
        if isinstance(score_str, str):
            score_str = score_str.replace('%', '').strip()
        return float(score_str)
    except:
        return np.nan

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs(["Data Cleaning", "EDA", "Visualizations"])

# ---------------- Data Cleaning (Tab 1) ----------------
with tab1:
    st.header("🧹 Data Cleaning")

    # File upload and reading
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
        
        # Make a copy for cleaning
        df_cleaned = df_raw.copy()  

        # Show data types before cleaning
        st.subheader("Data Types Before Cleaning")
        st.write(df_raw.dtypes)

        # Normalize column names
        df_cleaned.columns = df_cleaned.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('[^0-9a-zA-Z_]', '', regex=True)

        # Remove duplicates
        initial_shape = df_cleaned.shape
        df_cleaned.drop_duplicates(inplace=True)
        duplicates_removed = initial_shape[0] - df_cleaned.shape[0]

        # Strip whitespace from string columns
        df_cleaned = df_cleaned.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        # Fill missing numeric values with median
        numeric_cols = df_cleaned.select_dtypes(include=np.number).columns
        df_cleaned[numeric_cols] = df_cleaned[numeric_cols].apply(lambda x: x.fillna(x.median()))

        # Drop rows with missing values in key categorical columns
        df_cleaned.dropna(subset=['title', 'director', 'genre'], inplace=True)

        # Convert critic_score from percentage string to numeric
        if 'critic_score' in df_cleaned.columns:
            df_cleaned['critic_score'] = df_cleaned['critic_score'].apply(clean_critic_score)

        # Convert year_of_release to numeric
        if 'year_of_release' in df_cleaned.columns:
            df_cleaned['year_of_release'] = pd.to_numeric(df_cleaned['year_of_release'], errors='coerce')

        # Optional: Drop columns selected by user
        st.subheader("🧾 Optional: Drop Unwanted Columns")
        cols_to_drop = st.multiselect("Select columns to drop", df_cleaned.columns)
        if cols_to_drop:
            df_cleaned.drop(columns=cols_to_drop, inplace=True)
            st.success(f"Dropped columns: {', '.join(cols_to_drop)}")

        # Show data types after cleaning
        st.subheader("Data Types After Cleaning")
        st.write(df_cleaned.dtypes)

        # Outlier detection using IQR
        st.subheader("📊 Outlier Detection (IQR Method)")

        # Convert 'runtime' and 'critic_score' to numeric, coercing errors to NaN
        df_cleaned['runtime'] = pd.to_numeric(df_cleaned['runtime'], errors='coerce')
        df_cleaned['critic_score'] = pd.to_numeric(df_cleaned['critic_score'], errors='coerce')

        # Optionally fill missing values (e.g., with median or drop rows) before performing IQR analysis
        df_cleaned['runtime'] = df_cleaned['runtime'].fillna(df_cleaned['runtime'].median())
        df_cleaned['critic_score'] = df_cleaned['critic_score'].fillna(df_cleaned['critic_score'].median())

        # Perform IQR outlier detection
        for col in ['runtime', 'critic_score']:
            if col in df_cleaned.columns:
                Q1 = df_cleaned[col].quantile(0.25)
                Q3 = df_cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df_cleaned[(df_cleaned[col] < Q1 - 1.5 * IQR) | (df_cleaned[col] > Q3 + 1.5 * IQR)]
                st.write(f"{col}: {len(outliers)} potential outliers")
                st.dataframe(outliers)

        # Summary of cleaning
        st.subheader("✅ Summary of Cleaning Actions")
        st.markdown(f"""
        - **Duplicates Removed**: {duplicates_removed}  
        - **Numeric Columns Filled with Median**: {', '.join(numeric_cols)}  
        - **Dropped Rows with Missing Values in**: `title`, `director`, `genre`  
        - **Column Names Normalized**: spaces → underscores, special chars removed  
        - **Converted**: `critic_score` to numeric, `year_of_release` to numeric  
        """)

        # Store cleaned dataframe in session state for use in other tabs
        st.session_state.df_cleaned = df_cleaned

        # Final cleaned preview
        st.subheader("🧾 Preview of Cleaned Data")
        st.dataframe(df_cleaned.head())

        # Download button for filtered data
        st.download_button("📥 Download Cleaned Data", df_cleaned.to_csv(index=False), "cleanedData_movies.csv")

# ---------------- EDA (Tab 2) ----------------
with tab2:
    st.header("📊 Exploratory Data Analysis (EDA)")

    # Access cleaned data from session state
    if 'df_cleaned' in st.session_state:
        df_cleaned = st.session_state.df_cleaned

        # Additional Statistics
        st.subheader("📊 Extended Statistics")

        if 'critic_score' in df_cleaned.columns:
            top_movies = df_cleaned.sort_values(by='critic_score', ascending=False).head(5)[['title', 'critic_score']]
            st.markdown("**🎖️ Top 5 Movies by Critic Score:**")
            st.dataframe(top_movies)

        if 'genre' in df_cleaned.columns:
            genre_counts = df_cleaned['genre'].str.split(', ').explode().value_counts()
            st.markdown(f"**🎭 Most Common Genre:** `{genre_counts.idxmax()}` ({genre_counts.max()} movies)")

        if 'rating' in df_cleaned.columns:
            rating_counts = df_cleaned['rating'].value_counts()
            st.markdown(f"**🔠 Most Common MPAA Rating:** `{rating_counts.idxmax()}` ({rating_counts.max()} movies)")

        if 'director' in df_cleaned.columns:
            st.markdown(f"**🎬 Number of Unique Directors:** `{df_cleaned['director'].nunique()}`")

        if 'year_of_release' in df_cleaned.columns:
            st.markdown(f"**📅 Release Year Range:** `{int(df_cleaned['year_of_release'].min())}` - `{int(df_cleaned['year_of_release'].max())}`")

        # High Score Percentage
        if 'critic_score' in df_cleaned.columns:
            high_score_pct = (df_cleaned['critic_score'] >= 90).mean() * 100
            st.markdown(f"**🔥 Movies with Critic Score ≥ 90%:** `{high_score_pct:.1f}%`")

            disp = df_cleaned.select_dtypes(include=np.number).agg(['min', 'max', 'std']).transpose()
            st.dataframe(disp)

        st.subheader("🧰 Filter Dataset")

        filter_columns = st.multiselect("Select columns to filter by", df_cleaned.columns.tolist())

        filtered_df = df_cleaned.copy()

        for col in filter_columns:
            if df_cleaned[col].dtype == 'object':
                unique_vals = df_cleaned[col].dropna().unique().tolist()
                selected_vals = st.multiselect(f"Filter `{col}`", unique_vals)
                if selected_vals:
                    filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]
            elif np.issubdtype(df_cleaned[col].dtype, np.number):
                min_val, max_val = float(df_cleaned[col].min()), float(df_cleaned[col].max())
                selected_range = st.slider(f"Filter `{col}`", min_val, max_val, (min_val, max_val))
                filtered_df = filtered_df[(df_cleaned[col] >= selected_range[0]) & (df_cleaned[col] <= selected_range[1])]

        st.success(f"✅ Filtered Data: {filtered_df.shape[0]} rows")
        st.dataframe(filtered_df)

        # Download button for filtered data
        st.download_button("📥 Download Filtered Data", filtered_df.to_csv(index=False), "filtered_movies.csv")

# ---------------- Visualization (Tab 3) ----------------
with tab3:
    st.header("🎨 Visualizations")

    # Access cleaned data from session state
    if 'df_cleaned' in st.session_state:
        df_cleaned = st.session_state.df_cleaned

        st.markdown("### 🎬 Most Frequent Directors in Top 300 Movies")
        top_directors = df_cleaned['director'].dropna().value_counts().head(10)

        plot_type = st.radio("Choose Plot Type for Directors", ["Bar Plot", "Pie Chart"], key="director_plot")

        fig, ax = plt.subplots(figsize=(10, 6))
        if plot_type == "Bar Plot":
            sns.barplot(x=top_directors.values, y=top_directors.index, palette='magma', ax=ax)
            ax.set_xlabel('Number of Movies')
            ax.set_ylabel('director')
            ax.set_title('Top 10 Directors by Number of Movies')
        else:
            ax.pie(top_directors, labels=top_directors.index, autopct='%1.1f%%', startangle=140)
            ax.set_title('Top 10 Directors by Number of Movies')
            ax.axis('equal')

        st.pyplot(fig)


        # Runtime vs Critic Score
        if 'year_of_release' in df_cleaned.columns and 'critic_score' in df_cleaned.columns:
            st.markdown("**⏱️ year_of_release vs. Critic Score**")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x='year_of_release', y='critic_score', data=df_cleaned, ax=ax)
            ax.set_title('year_of_release vs Critic Score')
            st.pyplot(fig)

        # Genre Distribution
        if 'genre' in df_cleaned.columns:
            st.markdown("**🎭 Genre Distribution**")
            genre_counts = df_cleaned['genre'].str.split(', ').explode().value_counts()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=genre_counts.index, y=genre_counts.values, ax=ax)
            ax.set_title('Genre Distribution')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        