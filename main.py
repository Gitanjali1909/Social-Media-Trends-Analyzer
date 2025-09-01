import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scripts.preprocessing import load_and_clean_data
from wordcloud import WordCloud

st.set_page_config(page_title="Social Media Trends Analyzer", layout="wide")

df = load_and_clean_data("data/Cleaned_Viral_Social_Media_Trends.csv")

st.markdown("<h1 style='text-align: center; color: #4A90E2;'>📊 Social Media Trends Analyzer</h1>", unsafe_allow_html=True)
st.write("Explore trending hashtags, platforms, and engagement patterns over time — all in one place.")

st.sidebar.header("⚙️ Filters")
platforms = st.sidebar.multiselect("Select Platforms", df["Platform"].unique(), default=df["Platform"].unique())
regions = st.sidebar.multiselect("Select Regions", df["Region"].unique(), default=df["Region"].unique())
metric = st.sidebar.radio("Engagement Metric", ["Likes", "Shares", "Comments", "Total_Engagement"])

df_filtered = df[df["Platform"].isin(platforms) & df["Region"].isin(regions)]

col1, col2 = st.columns([2,1])

with col1:
    st.subheader("📈 Top Trending Hashtags Over Time")
    top_n = st.slider("Select number of top hashtags", 3, 10, 5)
    top_hashtags = (
        df_filtered.groupby("Hashtag")[metric]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )
    trend_top = (
        df_filtered[df_filtered["Hashtag"].isin(top_hashtags)]
        .groupby(["Month","Hashtag"])[metric]
        .sum()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10,5))
    sns.lineplot(data=trend_top, x="Month", y=metric, hue="Hashtag", marker="o", ax=ax, linewidth=2.5, palette="Set2")
    plt.title(f"Top {top_n} Hashtags Over Time ({metric})", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, use_container_width=True)

with col2:
    st.subheader("☁️ Wordcloud of Trending Hashtags")
    month_map = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}
    unique_months = sorted(df["Month"].unique())
    chosen_month = st.selectbox("Pick a Month", unique_months, format_func=lambda x: month_map.get(x, x))
    monthly_data = df_filtered[df_filtered["Month"] == chosen_month]
    if not monthly_data.empty:
        wc = WordCloud(width=600, height=400, background_color="white", colormap="plasma").generate(" ".join(monthly_data["Hashtag"]))
        st.image(wc.to_array(), use_container_width=True)
    else:
        st.info("No data available for this month.")

st.subheader("📊 Posts per Platform")
platform_counts = df_filtered["Platform"].value_counts().reset_index()
platform_counts.columns = ["Platform", "Count"]
fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(data=platform_counts, y="Platform", x="Count", palette="coolwarm", ax=ax)
ax.bar_label(ax.containers[0], label_type="edge", fontsize=10, color="black", padding=3)
plt.title("Posts per Platform", fontsize=14, fontweight="bold")
plt.xlabel("Number of Posts")
plt.ylabel("")
st.pyplot(fig, use_container_width=True)