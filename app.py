import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import io
import geopandas as gpd
from shapely.wkt import loads
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm as lgb
import shap
from streamlit_folium import st_folium
import folium
import pickle
import warnings
warnings.filterwarnings('ignore')

# 1. PAGE CONFIGURATION & STYLING

st.set_page_config(
    page_title="KL Property Market Analysis",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

GRAY1, GRAY2, GRAY3 = '#231F20', '#414040', '#555655'
GRAY4, GRAY5, GRAY6 = '#646369', '#76787B', '#828282'
GRAY7, GRAY8, GRAY9 = '#929497', '#A6A6A5', '#BFBEBE'
BLUE1, BLUE2, BLUE3, BLUE4 = '#174A7E', '#4A81BF', '#94B2D7', '#94AFC5'
RED1, RED2 = '#C3514E', '#E6BAB7'
GREEN1, GREEN2 = '#0C8040', '#9ABB59'
ORANGE1 = '#F79747'

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Helvetica', 'Arial']

# Injecting custom CSS
st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global font settings */
        .reportview-container .main .block-container,
        h1, h2, h3, h4, h5, h6, p, div, span {
            font-family: 'Inter', 'Segoe UI', 'Helvetica', 'Arial', sans-serif;
        }
        
        /* Custom metric styling with proper contrast */
        .metric-container {
            background: linear-gradient(145deg, #f8fafc, #e2e8f0);
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 4px solid #174A7E;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin: 0.5rem 0;
            border: 1px solid #e5e7eb;
        }
        
        /* Enhanced tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding: 0px 24px;
            background-color: #f1f5f9;
            border-radius: 8px 8px 0px 0px;
            color: #475569;
            font-weight: 500;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #174A7E;
            color: white;
        }
        
        /* Hero section styling */
        .hero-container {
            background: linear-gradient(135deg, #174A7E 0%, #4A81BF 100%);
            padding: 3rem 2rem;
            border-radius: 16px;
            color: white;
            margin: 2rem 0;
            text-align: center;
        }
        
        .hero-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 1rem;
            color: white;
        }
        
        .hero-subtitle {
            font-size: 1.2rem;
            opacity: 0.9;
            margin-bottom: 2rem;
            color: white;
        }
        
        /* Card styling with proper contrast */
        .info-card {
            background: linear-gradient(145deg, #f8fafc, #e2e8f0);
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            border: 1px solid #e5e7eb;
            margin: 1rem 0;
        }
        
        .info-card h4 {
            color: #174A7E !important;
            margin-bottom: 1rem !important;
            font-weight: 600;
        }
        
        .info-card p, .info-card li {
            color: #374151 !important;
        }
        
        /* Sidebar enhancement */
        .css-1d391kg {
            background-color: #f8fafc;
        }
        
        .css-1lcbmhc .css-1outpf7 {
            background-color: #174A7E;
            color: white;
        }
        
        /* Custom button styling */
        .stButton > button {
            background: linear-gradient(45deg, #174A7E, #4A81BF);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 2rem;
            font-weight: 500;
            transition: transform 0.2s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(23, 74, 126, 0.3);
        }
        
        /* Fix for metric text visibility */
        .metric-container h2, .metric-container p {
            color: #1f2937 !important;
        }
        
        .metric-container p {
            color: #6b7280 !important;
        }
    </style>
    """, unsafe_allow_html=True)

#2. OPTIMIZED DATA LOADING & CACHING (IMPORTANT!!!)

@st.cache_data
def load_and_clean_data():
    """
    Loads the property data from the Excel file and performs initial cleaning.
    Returns a cleaned pandas DataFrame.
    """
    filepath = 'travel_dist_df (1).xlsx'
    try:
        df = pd.read_excel(filepath, engine='openpyxl')
        
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        cols_to_drop = ['unnamed:_0', 'transaction_price'] + [col for col in df.columns if 'nearest' in col]
        df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        df['date'] = pd.to_datetime(df['date'])
        
        df_kl = df[df['district'] == 'KUALA LUMPUR'].copy()
        
        if 'geometry' in df_kl.columns:
            df_kl = df_kl.drop(columns=['geometry'])
        
        return df_kl

    except FileNotFoundError:
        st.error(f"Error: The data file '{filepath}' was not found. Please make sure it's in the same folder as app.py.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading or cleaning the data: {e}")
        return None

# OPTIMISING GEOSPATIAL DATA LOADING

@st.cache_data
def load_cluster_summary():
    """
    Loads a lightweight summary of cluster results instead of full geospatial data.
    This should be much faster for displaying cluster statistics.
    """
    try:
        cluster_results_df = pd.read_csv('cluster_results.csv')
        
        if 'cluster_id' in cluster_results_df.columns:
            cluster_summary = cluster_results_df.groupby('cluster_id').agg({
                'price_m2': ['count', 'mean', 'median', 'std'],
            }).round(2)
            cluster_summary.columns = ['Property_Count', 'Mean_Price', 'Median_Price', 'Price_StdDev']
            cluster_summary = cluster_summary.reset_index()
            return cluster_summary, cluster_results_df
        else:
            st.warning("cluster_id column not found in cluster results")
            return None, None
            
    except FileNotFoundError:
        st.error("Error: `cluster_results.csv` not found. Please generate it from the Jupyter Notebook.")
        return None, None
    except Exception as e:
        st.error(f"Error loading cluster data: {e}")
        return None, None

@st.cache_data
def create_simplified_map_data(cluster_results_df, selected_cluster):
    """
    Creates simplified map data for a specific cluster only.
    This avoids loading all geometries at once.
    """
    try:
        if cluster_results_df is None:
            return None
            
        selected_data = cluster_results_df[cluster_results_df['cluster_id'] == selected_cluster].copy()
        
        other_data = cluster_results_df[cluster_results_df['cluster_id'] != selected_cluster].sample(
            n=min(100, len(cluster_results_df[cluster_results_df['cluster_id'] != selected_cluster])),
            random_state=42
        ).copy()
        
        map_data = pd.concat([selected_data, other_data], ignore_index=True)
        
        if 'geometry_wkt' in map_data.columns:
            map_data['geometry'] = gpd.GeoSeries.from_wkt(map_data['geometry_wkt'])
            map_gdf = gpd.GeoDataFrame(map_data, geometry='geometry', crs='EPSG:4326')
            return map_gdf
        else:
            st.warning("geometry_wkt column not found")
            return None
            
    except Exception as e:
        st.error(f"Error creating map data: {e}")
        return None

df_cleaned = load_and_clean_data()

# OPTIMISING HELPER FUNCTIONS FOR EXPLORATION

def perform_exploratory_analysis_plotly(df_kl, area_column, top_n=8):
    """
    Generates and displays professional EDA plots for Kuala Lumpur using Plotly.
    Focuses on the top N areas (based on the area_column).
    """
    st.subheader(f"Analysis by Top {top_n} Areas: '{area_column.replace('_', ' ').title()}'")

    # Data Preparation

    top_areas_by_median_price = df_kl.groupby(area_column)['price_m2'].median().nlargest(top_n).index
    df_top_areas = df_kl[df_kl[area_column].isin(top_areas_by_median_price)].copy()

    # Layout
    col1, col2 = st.columns(2)

    with col1:
        # Plot 1
        avg_prices_sorted = df_kl.groupby('property_type')['price_m2'].mean().sort_values(ascending=False)
        fig_bar = px.bar(
            avg_prices_sorted,
            x=avg_prices_sorted.index,
            y=avg_prices_sorted.values,
            title='Average Price by Property Type in KL',
            labels={'x': 'Property Type', 'y': 'Average Price (RM/m²)'},
            color_discrete_sequence=[BLUE1]
        )
        fig_bar.update_layout(title_x=0.5, font_family='Segoe UI')
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        # Plot 2
        median_prices_sorted = df_top_areas.groupby(area_column)['price_m2'].median().sort_values(ascending=False)
        fig_box = px.box(
            df_top_areas,
            x='price_m2',
            y=area_column,
            orientation='h',
            title=f'Price Distribution by Top {top_n} Areas',
            labels={'price_m2': 'Price (RM/m²)', area_column: area_column.title()},
            category_orders={area_column: median_prices_sorted.index}, # Sort by median price
            color_discrete_sequence=[BLUE2]
        )
        fig_box.update_layout(title_x=0.5, font_family='Segoe UI')
        st.plotly_chart(fig_box, use_container_width=True)

    # Plot 3
    trend_data = df_top_areas.groupby([area_column, pd.Grouper(key='date', freq='QE')])['price_m2'].mean().reset_index()
    fig_line = px.line(
        trend_data,
        x='date',
        y='price_m2',
        color=area_column,
        title=f'Average Price Trend by Top {top_n} Areas (Quarterly)',
        labels={'date': 'Date', 'price_m2': 'Average Price (RM/m²)'},
        markers=True
    )
    fig_line.update_layout(title_x=0.5, font_family='Segoe UI', legend_title_text=area_column.title())
    st.plotly_chart(fig_line, use_container_width=True)
    
    # Plot 4
    st.markdown(f"**Relationship between Price and Distance to Park (by {area_column.replace('_', ' ').title()})**")
    
    if 'dist_to_park' in df_top_areas.columns:
        fig_facet = px.scatter(
            df_top_areas,
            x="dist_to_park",
            y="price_m2",
            facet_col=area_column,
            facet_col_wrap=4,
            trendline="ols",
            trendline_color_override=ORANGE1,
            opacity=0.6,
            labels={'dist_to_park': 'Distance to Nearest Park (km)', 'price_m2': 'Price (RM/m²)'},
            color_discrete_sequence=[BLUE1]
        )
        fig_facet.update_layout(title_x=0.5, font_family='Segoe UI')
        st.plotly_chart(fig_facet, use_container_width=True)
    else:
        st.info("Distance to park data not available for regression analysis.")

# 3. DATA CLEANING & WRANGLING SECTION
def show_data_cleaning():
    st.title("Data Cleaning & Wrangling")
    st.markdown("""
    This section documents the essential steps taken to clean and prepare the raw data for analysis. 
    A clean dataset is the foundation of any reliable analysis and modeling.
    """)

    st.subheader("Data Preparation Pipeline")

    with st.expander("Step 1: Loading the Raw Data"):
        st.markdown("""
        - **Action:** The data was loaded from the `travel_dist_df (1).xlsx` Excel file.
        - **Why:** This is the first step to bring the data into our Python environment for processing.
        """)
        st.code("df = pd.read_excel('travel_dist_df (1).xlsx', engine='openpyxl')", language="python")

    with st.expander("Step 2: Standardizing Column Names"):
        st.markdown("""
        - **Action:** All column names were converted to lowercase, leading/trailing spaces were removed, and spaces between words were replaced with underscores (e.g., `Property Type` became `property_type`).
        - **Why:** This creates consistent, predictable column names, which prevents errors and makes the code easier to write and read.
        """)
        st.code("df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')", language="python")

    with st.expander("Step 3: Dropping Unnecessary Columns"):
        st.markdown("""
        - **Action:** The following columns were removed: `unnamed:_0`, `transaction_price`, and any columns containing the word `nearest`.
        - **Why:** 
            - `unnamed:_0` was a redundant index column from the original file.
            - `transaction_price` was used to calculate our target variable `price_m2` and was no longer needed.
            - The `nearest_*` columns were not required for this specific analysis.
        """)
        st.code("""
cols_to_drop = ['unnamed:_0', 'transaction_price'] + [col for col in df.columns if 'nearest' in col]
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        """, language="python")

    with st.expander("Step 4: Converting the 'date' Column"):
        st.markdown("""
        - **Action:** The `date` column was converted from a plain text/object format into a proper `datetime` format.
        - **Why:** This allows us to perform time-based analysis, such as grouping data by quarter or year and plotting trends over time.
        """)
        st.code("df['date'] = pd.to_datetime(df['date'])", language="python")

    with st.expander("Step 5: Filtering for Kuala Lumpur Focus"):
        st.markdown("""
        - **Action:** The entire dataset was filtered to include only records where the `district` is 'KUALA LUMPUR'.
        - **Why:** The project's scope is specifically focused on the KL property market, so this step removes all irrelevant data from other districts.
        """)
        st.code("df_kl = df[df['district'] == 'KUALA LUMPUR'].copy()", language="python")

    st.subheader("The Cleaned Dataset")
    st.markdown("After completing these steps, the data is clean, properly formatted, and ready for exploration and analysis.")
    st.info(f"The final dataset for analysis has **{df_cleaned.shape[0]} rows** and **{df_cleaned.shape[1]} columns**.")

# OPTIMISING HELPER FUNCTIONS FOR ANALYSIS

@st.cache_data
def load_model_and_shap_results():
    try:
        results_df = pd.read_csv('model_performance_results.csv')
        with open('best_model.pkl', 'rb') as f:
            best_model = pickle.load(f)
            
        X_test_df = pd.read_csv('shap_test_data.csv', header=0)
        X_test_scaled = X_test_df.values
        feature_names = list(X_test_df.columns)
        
        shap_values_array = np.load('shap_values.npy')
        
        return results_df, best_model, X_test_scaled, feature_names, shap_values_array

    except FileNotFoundError as e:
        missing_file = str(e).split("'")[1] if "'" in str(e) else "unknown file"
        st.error(f"Error loading pre-computed file: {missing_file}. Please ensure all necessary files (`.csv`, `.pkl`, `.npy`) are in the project folder.")
        return None, None, None, None, None

def create_optimized_cluster_map(map_gdf, selected_cluster):
    """Creates an optimized Folium map with reduced complexity."""
    if map_gdf is None or map_gdf.empty:
        st.error("No map data available")
        return None
        
    try:
        if map_gdf.crs != 'EPSG:4326':
            map_gdf = map_gdf.to_crs(epsg=4326)
        
        map_gdf = map_gdf.copy()
        map_gdf['color'] = GRAY7
        map_gdf.loc[map_gdf['cluster_id'] == selected_cluster, 'color'] = BLUE1
        map_gdf['is_selected'] = map_gdf['cluster_id'] == selected_cluster
        
        selected_data = map_gdf[map_gdf['cluster_id'] == selected_cluster]
        if not selected_data.empty:
            bounds = selected_data.total_bounds
            center_lat = (bounds[1] + bounds[3]) / 2
            center_lon = (bounds[0] + bounds[2]) / 2
            map_center = [center_lat, center_lon]
        else:
            map_center = [map_gdf.geometry.centroid.y.mean(), map_gdf.geometry.centroid.x.mean()]

        m = folium.Map(
            location=map_center, 
            zoom_start=12, 
            tiles="CartoDB positron",
            prefer_canvas=True
        )

        selected_features = map_gdf[map_gdf['is_selected']].copy()
        other_features = map_gdf[~map_gdf['is_selected']].copy()
        
        if not other_features.empty:
            for _, row in other_features.iterrows():
                try:
                    folium.GeoJson(
                        row['geometry'], 
                        style_function=lambda x: {"color": GRAY7, "weight": 1, "opacity": 0.3},
                        tooltip=f"Cluster: {row.get('cluster_id', 'N/A')}"
                    ).add_to(m)
                except:
                    continue
        
        if not selected_features.empty:
            for _, row in selected_features.iterrows():
                try:
                    popup_html = f"""
                    <div style="font-family: Arial, sans-serif; min-width: 200px;">
                        <h4 style="margin: 0 0 10px 0; color: {BLUE1};">Property Details</h4>
                        <table style="width: 100%; border-collapse: collapse;">
                            <tr><td style="font-weight: bold; padding: 2px 8px 2px 0;">Cluster ID:</td><td>{row.get('cluster_id', 'N/A')}</td></tr>
                            <tr><td style="font-weight: bold; padding: 2px 8px 2px 0;">Price (RM/m²):</td><td>{row.get('price_m2', 0):,.2f}</td></tr>
                            <tr><td style="font-weight: bold; padding: 2px 8px 2px 0;">Road:</td><td>{row.get('road_name', 'N/A')}</td></tr>
                        </table>
                    </div>
                    """
                    
                    popup = folium.Popup(popup_html, max_width=300)
                    
                    folium.GeoJson(
                        row['geometry'], 
                        style_function=lambda x: {"color": BLUE1, "weight": 2.5, "opacity": 0.8, "fillOpacity": 0.3},
                        popup=popup,
                        tooltip=f"Selected Cluster: {row.get('cluster_id', 'N/A')}"
                    ).add_to(m)
                except:
                    continue
        
        return m
    
    except Exception as e:
        st.error(f"Error creating map: {e}")
        return None

# OPTIMISING MAIN PAGE FUNCTION

def show_data_analysis(df):
    st.title("Data Processing & Analysis")
    st.markdown("This section showcases the advanced analysis, from geospatial clustering to predictive modeling and interpretation.")

    if 'map_loaded' not in st.session_state:
        st.session_state.map_loaded = False
    if 'current_cluster_map' not in st.session_state:
        st.session_state.current_cluster_map = None

    tab1, tab2, tab3 = st.tabs(["Geospatial Clustering", "Model Performance", "Model Interpretation (SHAP)"])

    with tab1:
        st.subheader("Geospatial Clustering with DBSCAN")
        st.markdown("""
        To move beyond simple administrative boundaries like `mukim`, we used **DBSCAN**. This creates organic, 
        data-driven sub-markets based on geographic proximity. The results below are pre-computed for fast loading.
        """)
        
        cluster_summary, cluster_results_df = load_cluster_summary()
        
        if cluster_summary is not None:
            st.markdown("#### Cluster Summary Statistics")
            st.dataframe(cluster_summary.style.format({
                'Mean_Price': '{:,.2f}', 'Median_Price': '{:,.2f}', 'Price_StdDev': '{:,.2f}'
            }))
            
            st.markdown("#### Interactive Cluster Map")
            st.markdown("Select a cluster from the dropdown below to highlight it on the map.")
            
            cluster_list = sorted(cluster_summary['cluster_id'].unique(), key=lambda x: (x.startswith('-1'), x))
            selected_cluster = st.selectbox("Choose a cluster to focus on:", cluster_list)
            
            # ORIGINAL METRIC CODE
            selected_stats = cluster_summary[cluster_summary['cluster_id'] == selected_cluster].iloc[0]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Properties", f"{int(selected_stats['Property_Count'])}")
            with col2:
                st.metric("Mean Price", f"RM {selected_stats['Mean_Price']:,.0f}/m²")
            with col3:
                st.metric("Median Price", f"RM {selected_stats['Median_Price']:,.0f}/m²")
            with col4:
                st.metric("Price Std Dev", f"RM {selected_stats['Price_StdDev']:,.0f}")
            
            # NEW SESSION STATE LOGIC
            if selected_cluster != st.session_state.current_cluster_map:
                st.session_state.map_loaded = False

            if st.button("Load Map for Selected Cluster", help="Click to load the interactive map"):
                st.session_state.map_loaded = True
                st.session_state.current_cluster_map = selected_cluster

            if st.session_state.map_loaded:
                with st.spinner(f"Loading map for cluster {st.session_state.current_cluster_map}..."):
                    map_data = create_simplified_map_data(cluster_results_df, st.session_state.current_cluster_map)
                    if map_data is not None:
                        cluster_map = create_optimized_cluster_map(map_data, st.session_state.current_cluster_map)
                        if cluster_map is not None:
                            # Added a key for stability, especially inside tabs
                            st_folium(cluster_map, width=1200, height=500, key="folium_map")
                        else:
                            st.error("Could not create the map.")
                    else:
                        st.error("Could not load map data.")
        else:
            st.warning("Could not load cluster analysis results.")

    # Load all model/SHAP results once for the next two tabs
    model_results, best_model, X_test_scaled, feature_names, shap_values_array = load_model_and_shap_results()

    with tab2:
        st.subheader("Predictive Model Performance")
        st.markdown("""
        We trained several regression models to predict `price_m2`. The performance results shown below were 
        pre-computed to ensure the app is fast and responsive.
        """)
        
        if model_results is not None:
            st.markdown("#### Model Comparison")
            st.dataframe(
                model_results.style.format({
                    'Test R2 Score': '{:.3f}',
                    'Test RMSE': '{:,.2f}'
                })
            )
            
            best_model_name = model_results.iloc[0]['Model']
            best_r2 = model_results.iloc[0]['Test R2 Score']
            
            st.markdown(f"""
            **Conclusion:** The **{best_model_name}** model performed the best, achieving an R² score of **{best_r2:.3f}**. 
            This indicates that the model can explain approximately **{best_r2:.1%}** of the variance in property prices.
            """)
            
            # Visualise model comparison
            fig_comparison = px.bar(
                model_results,
                x='Model',
                y='Test R2 Score',
                title='Model Performance Comparison (R² Score)',
                labels={'Test R2 Score': 'R² Score'},
                color_discrete_sequence=[BLUE1]
            )
            fig_comparison.update_layout(title_x=0.5, font_family='Segoe UI')
            st.plotly_chart(fig_comparison, use_container_width=True)
            
        else:
            st.warning("Could not load model performance results.")

    with tab3:
        st.subheader("Model Interpretation with SHAP")
        st.markdown("""
        To understand *why* our model makes its predictions, we use **SHAP (SHapley Additive exPlanations)**. 
        The SHAP values below were pre-computed for fast loading.
        """)
        
        if shap_values_array is not None:
            st.markdown("#### SHAP Feature Importance")
            st.markdown("This chart shows the average impact of each feature on the model's output magnitude.")
            
            # Create SHAP summary plot
            fig_summary, ax_summary = plt.subplots(figsize=(8, 4))
            shap.summary_plot(
                shap_values_array,
                features=X_test_scaled,
                feature_names=feature_names,
                plot_type="bar",
                show=False,
                color=BLUE1
            )
            ax_summary.set_xlabel("Mean |SHAP Value|")
            ax_summary.set_title("Feature Importance (SHAP)")
            plt.tight_layout()
            st.pyplot(fig_summary)
            
            st.markdown("#### SHAP Dependence Plots")
            st.markdown("Select a feature to see how its value affects the predicted property price.")
            
            top_features = list(feature_names)
            selected_feature_shap = st.selectbox("Choose a feature to analyze:", top_features)
            
            if selected_feature_shap:
                try:
                    fig_dependence, ax_dependence = plt.subplots(figsize=(10, 6))
                    shap.dependence_plot(
                    selected_feature_shap, 
                    shap_values_array,
                    X_test_scaled, 
                    feature_names=feature_names, 
                    interaction_index="auto", 
                    ax=ax_dependence, 
                    show=False
                )
                    ax_dependence.set_title(f"SHAP Dependence Plot: {selected_feature_shap}")
                    plt.tight_layout()
                    st.pyplot(fig_dependence)
                except Exception as e:
                    st.error(f"Error creating dependence plot: {e}")
        else:
            st.warning("Could not load SHAP interpretation results.")

# DATA EXPLORATION FUNCTION

def show_data_exploration(df):
    st.title("📊 Data Exploration")
    st.markdown("---")
    st.markdown("This section provides a comprehensive overview of the Kuala Lumpur property dataset. Use the tabs below to navigate through the exploratory data analysis.")

    tab1, tab2 = st.tabs(["Dataset Snapshot", "Price Analysis by Mukim"])

    with tab1:
        st.subheader("Snapshot of the Cleaned Dataset")
        st.markdown("A quick look at the structure, data types, and summary statistics.")

        st.markdown("##### 📋 Data Columns and Types")
        
        info_df = pd.DataFrame({
            "Column": df.columns,
            "Non-Null Count": df.count().values,
            "Dtype": df.dtypes.values
        })
        st.dataframe(info_df, use_container_width=True)
        st.markdown(f"The dataset contains **{len(df)}** entries.")
        st.markdown("---")

        # 2. Data Preview
        st.markdown("##### 👀 First 5 Rows")
        st.dataframe(df.head())
        st.markdown("---")

        # 3. Descriptive Statistics
        st.markdown("##### 📈 Descriptive Statistics for Numerical Columns")
        st.dataframe(df.describe().style.format("{:,.2f}"))

    with tab2:
        st.subheader("Price Analysis by Mukim")
        st.markdown("""
        Here, we analyze property prices based on administrative districts (`mukim`). This helps us understand the geographical price variations across Kuala Lumpur before any advanced clustering.
        The analysis focuses on the top 8 mukims by median property price.
        """)
        
        perform_exploratory_analysis_plotly(df, 'mukim', top_n=8)

# 4. SIDEBAR NAVIGATION

st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="color: #174A7E; margin: 0;">🏠 KL Property</h2>
        <p style="color: #b0bdd1; margin: 0.5rem 0 2rem 0;">Market Analysis Dashboard</p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 🧭 Navigation")

navigation_options = {
    "🏡 Overview": {
        "key": "Overview",
        "desc": "Project summary and insights"
    },
    "📊 Data Exploration": {
        "key": "Data Exploration", 
        "desc": "Dataset analysis and patterns"
    },
    "🧹 Data Cleaning": {
        "key": "Data Cleaning & Wrangling",
        "desc": "Preprocessing pipeline"
    },
    "⚙️ Advanced Analysis": {
        "key": "Data Processing & Analysis",
        "desc": "Clustering, modeling & SHAP"
    }
}

selected_option = st.sidebar.radio(
    "Choose a section:",
    list(navigation_options.keys()),
    format_func=lambda x: x
)

selected_desc = navigation_options[selected_option]["desc"]
st.sidebar.markdown(f"*{selected_desc}*")

selection = navigation_options[selected_option]["key"]

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Quick Stats")
if df_cleaned is not None:
    st.sidebar.metric("Total Properties", f"{len(df_cleaned):,}")
    st.sidebar.metric("Date Range", f"{df_cleaned['date'].dt.year.min()} - {df_cleaned['date'].dt.year.max()}")
    st.sidebar.metric("Avg Price/m²", f"RM {df_cleaned['price_m2'].mean():,.0f}")

# 5. MAIN APP LOGIC

if df_cleaned is None:
    st.warning("Data could not be loaded. Please fix the error above to proceed.")
    st.stop()

# Page: Overview
if selection == "Overview":
    st.markdown("""
        <div class="hero-container">
            <div class="hero-title">🏠 KL Property Market Analysis</div>
            <div class="hero-subtitle">Advanced Analytics • Machine Learning • Geospatial Intelligence</div>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("📊 Project Insights at a Glance")
    
    cluster_summary, _ = load_cluster_summary()
    model_results, _, _, _, _ = load_model_and_shap_results()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div class="metric-container">
                <h3 style="color: #174A7E; margin: 0; font-size: 2rem;">🏢</h3>
                <h2 style="margin: 0.5rem 0 0 0; color: #1f2937; font-weight: 700;">{:,}</h2>
                <p style="margin: 0; color: #6b7280; font-weight: 500;">Properties Analyzed</p>
            </div>
        """.format(df_cleaned.shape[0]), unsafe_allow_html=True)
    
    with col2:
        cluster_count = cluster_summary['cluster_id'].nunique() - 1 if cluster_summary is not None else 0
        st.markdown("""
            <div class="metric-container">
                <h3 style="color: #174A7E; margin: 0; font-size: 2rem;">🗺️</h3>
                <h2 style="margin: 0.5rem 0 0 0; color: #1f2937; font-weight: 700;">{}</h2>
                <p style="margin: 0; color: #6b7280; font-weight: 500;">Sub-Markets Identified</p>
            </div>
        """.format(cluster_count), unsafe_allow_html=True)
    
    with col3:
        best_r2 = model_results.iloc[0]['Test R2 Score'] if model_results is not None else 0
        st.markdown("""
            <div class="metric-container">
                <h3 style="color: #174A7E; margin: 0; font-size: 2rem;">🎯</h3>
                <h2 style="margin: 0.5rem 0 0 0; color: #1f2937; font-weight: 700;">{:.2f}</h2>
                <p style="margin: 0; color: #6b7280; font-weight: 500;">Best Model R² Score</p>
            </div>
        """.format(best_r2), unsafe_allow_html=True)
    
    with col4:
        avg_price = df_cleaned['price_m2'].mean()
        st.markdown("""
            <div class="metric-container">
                <h3 style="color: #174A7E; margin: 0; font-size: 2rem;">💰</h3>
                <h2 style="margin: 0.5rem 0 0 0; color: #1f2937; font-weight: 700;">RM {:,.0f}</h2>
                <p style="margin: 0; color: #6b7280; font-weight: 500;">Avg Price per m²</p>
            </div>
        """.format(avg_price), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Project Description
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
            <div class="info-card">
                <h4 style="color: #174A7E; margin-bottom: 1rem;">🎯 About This Project</h4>
                <p style="color: #374151; line-height: 1.6;">
                    This comprehensive analysis explores the Kuala Lumpur property market using advanced data science techniques. 
                    We combine geospatial clustering, machine learning, and explainable AI to understand the key drivers of property values.
                <p style="color: #374151; line-height: 1.6;">
                    Our data is sourced from the <strong>Valuation and Property Services Department (JPPH)</strong>, 
                    providing reliable and comprehensive property transaction records.
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="info-card">
                <h4 style="color: #174A7E; margin-bottom: 1rem;">🔍 Key Features</h4>
                <ul style="color: #374151; line-height: 1.8;">
                    <li><strong>Advanced Clustering:</strong> DBSCAN algorithm identifies organic sub-markets</li>
                    <li><strong>Predictive Modeling:</strong> Multiple ML algorithms with performance comparison</li>
                    <li><strong>Explainable AI:</strong> SHAP values reveal model decision-making</li>
                    <li><strong>Interactive Visualization:</strong> Dynamic maps and charts</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="info-card">
                <h4 style="color: #174A7E; margin-bottom: 1rem;">🛠️ How to Navigate</h4>
                <ul style="color: #374151; line-height: 1.8;">
                    <li><strong>📊 Data Exploration:</strong> Understand the dataset structure and patterns</li>
                    <li><strong>🧹 Data Cleaning:</strong> Learn about our preprocessing pipeline</li>
                    <li><strong>⚙️ Analysis:</strong> Explore clustering, modeling, and insights</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="info-card">
                <h4 style="color: #174A7E; margin-bottom: 1rem;">👥 Our Team</h4>
                <ul style="font-size: 0.9rem; color: #374151; line-height: 1.6;">
                    <li>Wai Yan Moe (24065716)</li>
                    <li>Yew Yen Bin (24144198)</li>
                    <li>Phan Duc Duy Anh (24101602)</li>
                    <li>Ameiyrul Hassan Bin Ashruff Hassan (18091223)</li>
                    <li>Matthew Lo Hon Mun (25083247)</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    # Data Source and Preview
    st.subheader("📋 Data Source & Preview")
    st.info("💡 **Data Source:** [JPPH Valuation and Property Services Department](https://napic2.jpph.gov.my/ms/data-transaksi?category=36&id=241)")

    st.subheader("🎯 Business Goal")
    st.markdown("""
        - **Understand** the factors influencing property prices in KL.
        - **Create** data-driven sub-market clusters using geospatial analysis.
        - **Build** a reliable regression (machine learning) model to predict property price per square meter.
        """)

elif selection == "Data Exploration":
    show_data_exploration(df_cleaned)

elif selection == "Data Cleaning & Wrangling":
    show_data_cleaning()

elif selection == "Data Processing & Analysis":
    show_data_analysis(df_cleaned)


# FOOTER
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; padding: 2rem 0; border-top: 1px solid #e2e8f0; margin-top: 3rem;">
        <p style="color: #64748b; margin-bottom: 1rem;">
            Built by <strong>Wai Yan Moe</strong>
        <p style="color: #64748b; font-size: 0.9rem;">
            🚀 <a href="https://www.linkedin.com/in/waiyan-william-moe/" target="_blank" style="color: #174A7E; text-decoration: none;">Connect on LinkedIn</a>
    </div>
""", unsafe_allow_html=True)