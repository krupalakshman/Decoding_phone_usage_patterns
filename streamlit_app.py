import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

st.set_page_config(
    page_title="Phone Usage Analysis Dashboard",
    page_icon="📱",
    layout="wide"
)

USAGE_CATEGORIES = {
    0: "Social Media",
    1: "Gaming",
    2: "Streaming",
    3: "Professional",
    4: "Educational",
    5: "Communication"
}

tab_home, tab1, tab2, tab3 = st.tabs(["Dashboard Overview", "Usage Patterns", "Classification", "User Segmentation"])

df = pd.read_csv("clustering_results.csv")

# Importing the Decision Tree model
try:
    with open("decision tree_model.pkl", "rb") as f:
        model = pickle.load(f)

except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    model = None

with tab_home:
    st.header("📊 Dashboard Overview")
    st.write("This is a Phone Usage Analysis application that allows users to analyze and predict phone usage patterns.")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Users", len(df))
        st.metric("Average Screen Time", f"{df['Screen Time (hrs/day)'].mean():.1f} hrs/day")

    with col2:
        st.metric("Average Data Usage", f"{df['Data Usage (GB/month)'].mean():.1f} GB/month")

        df['Primary Use'] = df['Primary Use'].fillna(-1).astype(int)  # Handling missing values safely

        try:
            use_counts = df['Primary Use'].value_counts()
            most_common = USAGE_CATEGORIES.get(int(use_counts.idxmax()), "Unknown Category")
        except Exception as e:
            st.error(f"Error calculating most common use: {str(e)}")
            most_common = "Error in calculation"

        st.metric("Most Common Use", most_common)

with tab1:
    st.header("Visualizations")
    st.subheader("📊 Select Feature for Visualization")

    # List of features to visualize
    feature_options = [
        "Screen Time (hrs/day)", "Social Media Time (hrs/day)", 
        "Streaming Time (hrs/day)", "Gaming Time (hrs/day)", 
        "Data Usage (GB/month)", "E-commerce Spend (INR/month)", 
        "Monthly Recharge Cost (INR)"
    ]

    # Let user choose a feature
    selected_feature = st.selectbox("Choose a feature to visualize", feature_options)

    st.subheader(f"📊 Distribution of {selected_feature}")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[selected_feature], bins=30, kde=True, ax=ax)

    plt.xlabel(selected_feature)
    plt.ylabel("Count")
    plt.title(f"Distribution of {selected_feature}")
    st.pyplot(fig)

with tab2:
    st.header("🎯 User Classification")

    if model is not None:
        st.subheader("Predict Primary Use")
        
        with st.form("classification_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                age = st.number_input("Age", min_value=18, max_value=80, value=25)
                screen_time = st.number_input("Screen Time (hrs/day)", min_value=0.5, max_value=24.0, value=4.0, step=0.5)
                data_usage = st.number_input("Data Usage (GB/month)", min_value=1.0, value=30.0, step=1.0)
                calls_duration = st.number_input("Calls Duration (mins/day)", min_value=1.0, value=45.0, step=5.0)

            with col2:
                num_apps = st.number_input("Number of Apps Installed", min_value=5, value=25, step=1)
                social_media = st.number_input("Social Media Time (hrs/day)", min_value=0.0, max_value=24.0, value=2.5, step=0.5)
                ecommerce_spend = st.number_input("E-commerce Spend (INR/month)", min_value=0.0, value=2000.0, step=100.0)
                streaming_time = st.number_input("Streaming Time (hrs/day)", min_value=0.0, max_value=24.0, value=1.5, step=0.5)

            with col3:
                gaming_time = st.number_input("Gaming Time (hrs/day)", min_value=0.0, max_value=24.0, value=1.0, step=0.5)
                monthly_cost = st.number_input("Monthly Recharge Cost (INR)", min_value=200.0, value=699.0, step=50.0)
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                os = st.selectbox("Operating System", ["Android", "iOS"])

            submit_button = st.form_submit_button("Predict Primary Use")
        
        if submit_button:
            total_time = screen_time + social_media + gaming_time + streaming_time
            if total_time > 24:
                st.error("❌ Total time across activities exceeds 24 hours. Please adjust your inputs.")
            else:
                try:
                    #  Correctly encode categorical variables
                    gender_mapping = {"Male": 0, "Female": 1, "Other": 2}
                    gender_encoded = gender_mapping[gender]

                    os_mapping = {"Android": 0, "iOS": 1}
                    os_encoded = os_mapping[os]

                    # ✅ Create input DataFrame
                    input_data = pd.DataFrame([[
                        age, gender_encoded, screen_time, data_usage, calls_duration, num_apps, 
                        social_media, ecommerce_spend, streaming_time, gaming_time, monthly_cost, os_encoded
                    ]], columns=[
                        'Age', 'Gender', 'Screen Time (hrs/day)', 'Data Usage (GB/month)', 
                        'Calls Duration (mins/day)', 'Number of Apps Installed', 
                        'Social Media Time (hrs/day)', 'E-commerce Spend (INR/month)', 
                        'Streaming Time (hrs/day)', 'Gaming Time (hrs/day)', 
                        'Monthly Recharge Cost (INR)', 'OS'
                    ])



                    # ✅ Predict the primary use
                    prediction_num = model.predict(input_data)[0]
                    prediction_category = USAGE_CATEGORIES.get(prediction_num, "Unknown Category")
                    st.success(f"🎯 Predicted Primary Use: {prediction_category}")

                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")
                    st.info("ℹ️ Please ensure all input values are valid.")

if df is not None:
    cluster_columns = [col for col in df.columns if col.startswith('cluster_')]

def create_cluster_plot(df, cluster_col):
    # Use plt.style.use('seaborn-v0_8') instead of 'seaborn'
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create color palette for better visibility
    n_clusters = len(df[cluster_col].unique())
    palette = sns.color_palette("husl", n_clusters)
    
    # Create scatter plot with enhanced styling
    sns.scatterplot(data=df, 
                   x='Screen Time (hrs/day)', 
                   y='Data Usage (GB/month)', 
                   hue=cluster_col,
                   palette=palette,
                   alpha=0.6)
    
    # Enhance plot styling
    plt.title('User Segments by Screen Time and Data Usage', pad=20)
    plt.xlabel('Screen Time (hours/day)', labelpad=10)
    plt.ylabel('Data Usage (GB/month)', labelpad=10)
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    return fig

with tab3:
    st.header("👥 User Segmentation")
    
    if cluster_columns:
        # Display cluster analysis
        st.subheader("User Clusters")
        
        # Cluster selection
        cluster_col = st.selectbox("Select Cluster Type", cluster_columns)
        
        # Create and display scatter plot
        fig = create_cluster_plot(df, cluster_col)
        st.pyplot(fig)
        plt.close()
        USAGE_CATEGORIES = {
            0: "Social Media",
            1: "Gaming",
            2: "Streaming",
            3: "Professional",
            4: "Educational"
            }
    # Cluster characteristics
        st.subheader("Cluster Characteristics")
        for cluster in sorted(df[cluster_col].unique()):
            cluster_data = df[df[cluster_col] == cluster]
        
            with st.expander(f"Cluster {cluster}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Number of Users", len(cluster_data))
                    st.metric("Avg Screen Time", f"{cluster_data['Screen Time (hrs/day)'].mean():.1f} hrs/day")
                
                with col2:
                    st.metric("Avg Data Usage", f"{cluster_data['Data Usage (GB/month)'].mean():.1f} GB/month")

            # Convert Primary Use to int & get most common use
                    most_common_use_num = cluster_data['Primary Use'].astype(int).mode()[0]
                    most_common_use = USAGE_CATEGORIES.get(most_common_use_num, f"Category {most_common_use_num}")

                    st.metric("Most Common Use", most_common_use)


    else:
        st.warning("⚠️ No cluster columns found in the dataset.")

# Add footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
    <p>Built with Streamlit • Decoding Phone Usage Patterns (Analysis and Visualization)</p>
    </div>
    """, unsafe_allow_html=True)
