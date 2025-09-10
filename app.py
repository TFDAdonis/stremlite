import streamlit as st
import ee
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import dates as mdates
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium
from io import StringIO
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import base64
from PIL import Image
import io
import requests
import time

# Initialize Earth Engine with error handling
try:
    # Try to initialize with the project
    ee.Initialize(project='citric-hawk-457513-i6')
    st.success("✅ Earth Engine initialized successfully!")
except Exception as e:
    try:
        # Fallback to default initialization
        ee.Initialize()
        st.success("✅ Earth Engine initialized successfully!")
    except:
        st.error(f"❌ Error initializing Earth Engine: {str(e)}")
        st.info("Please make sure you've authenticated with Earth Engine using: ee.Authenticate()")

# FAO GAUL Dataset
FAO_GAUL = ee.FeatureCollection("FAO/GAUL/2015/level0")  # Countries
FAO_GAUL_ADMIN1 = ee.FeatureCollection("FAO/GAUL/2015/level1")  # Admin1 (states/provinces)
FAO_GAUL_ADMIN2 = ee.FeatureCollection("FAO/GAUL/2015/level2")  # Admin2 (municipalities)

# 1. STUDY AREA SELECTION
def get_admin_boundaries(level, country_code=None, admin1_code=None):
    """Get administrative boundaries at different levels"""
    if level == 0:  # Countries
        return FAO_GAUL
    elif level == 1:  # Admin1 (states/provinces)
        if country_code:
            return FAO_GAUL_ADMIN1.filter(ee.Filter.eq('ADM0_CODE', country_code))
        return FAO_GAUL_ADMIN1
    elif level == 2:  # Admin2 (municipalities)
        if admin1_code:
            return FAO_GAUL_ADMIN2.filter(ee.Filter.eq('ADM1_CODE', admin1_code))
        return FAO_GAUL_ADMIN2
    return None

def get_boundary_names(fc, level):
    """Get names of boundaries in a feature collection for a specific level"""
    try:
        if level == 0:  # Countries
            names = fc.aggregate_array('ADM0_NAME').getInfo()
        elif level == 1:  # Admin1 (states/provinces)
            names = fc.aggregate_array('ADM1_NAME').getInfo()
        elif level == 2:  # Admin2 (municipalities)
            names = fc.aggregate_array('ADM2_NAME').getInfo()
        else:
            names = []
        return sorted(list(set(names)))  # Remove duplicates and sort
    except Exception as e:
        st.error(f"Error getting boundary names: {str(e)}")
        return []

# 2. CLOUD MASKING AND NDVI FUNCTIONS
def mask_clouds(image):
    qa = image.select('QA60')
    cloud_mask = qa.bitwiseAnd(1 << 10).eq(0)
    cirrus_mask = qa.bitwiseAnd(1 << 11).eq(0)
    return image.updateMask(cloud_mask.And(cirrus_mask))

def add_ndvi(image):
    return image.addBands(
        image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    )

# 3. AUTOMATIC GIF GENERATION
def generate_ndvi_gif(study_roi, start_date='2023-01-01', end_date='2023-03-01'):
    try:
        # Create monthly composites
        def create_monthly_composite(date):
            date = ee.Date(date)
            monthly_ndvi = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                          .filterBounds(study_roi)
                          .filterDate(date, date.advance(1, 'month'))
                          .map(mask_clouds)
                          .map(add_ndvi)
                          .median()
                          .clip(study_roi))
            return monthly_ndvi.set('system:time_start', date.millis())

        # Generate sequence of months
        n_months = ee.Date(end_date).difference(ee.Date(start_date), 'month').round()
        dates = ee.List.sequence(0, n_months).map(
            lambda n: ee.Date(start_date).advance(n, 'month'))

        collection = ee.ImageCollection(dates.map(create_monthly_composite))

        # GIF parameters
        vis_params = {
            'min': 0.0,
            'max': 1.0,
            'palette': ['red', 'yellow', 'green', 'darkgreen'],
            'bands': ['NDVI'],
            'region': study_roi,
            'dimensions': 600,
            'framesPerSecond': 2,
            'format': 'gif'
        }

        st.info("🔄 Generating NDVI timelapse GIF...")
        gif_url = collection.getVideoThumbURL(vis_params)
        
        # Download and display the GIF
        response = requests.get(gif_url)
        if response.status_code == 200:
            st.image(response.content, caption="NDVI Timelapse GIF")
            st.success("✅ GIF generated successfully!")
        else:
            st.error("❌ Failed to generate GIF. Please try a different date range or area.")
            
    except Exception as e:
        st.error(f"❌ Error generating GIF: {str(e)}")
        st.info("This might be due to insufficient satellite data for the selected area/date range.")

# 4. CUSTOM DATA PROCESSING
def analyze_custom_data(data_input):
    try:
        df = pd.read_csv(StringIO(data_input), header=None, names=['date', 'NDVI'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').dropna()

        # Create animation
        fig, ax = plt.subplots(figsize=(12, 6))

        def animate(i):
            ax.clear()
            current_date = df['date'].iloc[i]
            mask = df['date'] <= current_date

            ax.scatter(df[mask]['date'], df[mask]['NDVI'],
                     color='green', alpha=0.7, s=50, label='Your Data')
            ax.set_title(f'Your NDVI Measurements\n{current_date.strftime("%Y-%m-%d")}')
            ax.set_ylabel('NDVI Value')
            ax.set_xlabel('Date')
            ax.set_ylim(-0.5, 1.0)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()

        ani = animation.FuncAnimation(fig, animate, frames=len(df), interval=200)
        
        # Save animation as GIF
        buf = io.BytesIO()
        ani.save(buf, format='gif', writer='pillow')
        buf.seek(0)
        
        # Display the GIF
        st.image(buf, caption="Your NDVI Data Animation")
        plt.close(fig)
        
    except Exception as e:
        st.error(f"❌ Error processing custom data: {str(e)}")

# 5. REAL-TIME DATA ANALYSIS WITH FOREX-STYLE CHARTS
def create_forex_style_chart(df, title):
    """Create a forex-style chart for NDVI data"""
    try:
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Set background color
        fig.patch.set_facecolor('#1e1e1e')
        ax.set_facecolor('#1e1e1e')
        
        # Plot the data
        line = ax.plot(df.index, df['NDVI'], color='#00ff00', linewidth=1.5, alpha=0.8)[0]
        
        # Add fill between line and minimum value
        ax.fill_between(df.index, df['NDVI'], df['NDVI'].min(), color='#00ff00', alpha=0.1)
        
        # Customize grid
        ax.grid(True, color='#2a2a2a', linestyle='-', linewidth=0.5)
        
        # Set title and labels
        ax.set_title(title, color='white', fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('NDVI', color='white', fontsize=12)
        ax.set_xlabel('Date', color='white', fontsize=12)
        
        # Customize ticks
        ax.tick_params(colors='white', which='both')
        
        # Format y-axis to show 2 decimal places
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add a simple moving average
        if len(df) > 20:
            sma = df['NDVI'].rolling(window=20).mean()
            ax.plot(df.index, sma, color='#ff9900', linewidth=1.5, alpha=0.8, label='20-period SMA')
            ax.legend(facecolor='#1e1e1e', edgecolor='#1e1e1e', labelcolor='white')
        
        # Add value markers on the right side
        ax.axhline(y=df['NDVI'].iloc[-1], color='white', linestyle='--', alpha=0.3)
        ax.text(df.index[-1] + timedelta(days=5), df['NDVI'].iloc[-1], 
                f'{df["NDVI"].iloc[-1]:.3f}', color='white', 
                verticalalignment='center', fontweight='bold')
        
        # Add min/max markers
        max_val = df['NDVI'].max()
        min_val = df['NDVI'].min()
        ax.axhline(y=max_val, color='red', linestyle=':', alpha=0.5)
        ax.axhline(y=min_val, color='blue', linestyle=':', alpha=0.5)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.error(f"❌ Error creating chart: {str(e)}")
        return None

def create_interactive_forex_chart(df, title):
    """Create an interactive forex-style chart using Plotly"""
    try:
        # Create figure
        fig = go.Figure()
        
        # Add NDVI line
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df['NDVI'],
            mode='lines',
            name='NDVI',
            line=dict(color='#00ff00', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.1)'
        ))
        
        # Add moving average if enough data
        if len(df) > 20:
            sma = df['NDVI'].rolling(window=20).mean()
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=sma,
                mode='lines',
                name='20-period SMA',
                line=dict(color='#ff9900', width=2)
            ))
        
        # Add current value marker
        last_value = df['NDVI'].iloc[-1]
        fig.add_trace(go.Scatter(
            x=[df.index[-1]], 
            y=[last_value],
            mode='markers+text',
            name='Current',
            marker=dict(color='white', size=10),
            text=[f'{last_value:.3f}'],
            textposition='middle right',
            textfont=dict(color='white', size=12)
        ))
        
        # Update layout for forex style
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=24, color='white'),
                x=0.5
            ),
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#1e1e1e',
            xaxis=dict(
                title='Date',
                titlefont=dict(color='white'),
                tickfont=dict(color='white'),
                gridcolor='#2a2a2a',
                zerolinecolor='#2a2a2a'
            ),
            yaxis=dict(
                title='NDVI',
                titlefont=dict(color='white'),
                tickfont=dict(color='white'),
                gridcolor='#2a2a2a',
                zerolinecolor='#2a2a2a',
                tickformat='.2f'
            ),
            legend=dict(
                font=dict(color='white'),
                bgcolor='rgba(0,0,0,0)'
            ),
            hovermode='x unified',
            height=600
        )
        
        # Add range slider
        fig.update_xaxes(
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ]),
                bgcolor='#2a2a2a',
                font=dict(color='white')
            )
        )
        
        return fig
    except Exception as e:
        st.error(f"❌ Error creating interactive chart: {str(e)}")
        return None

def analyze_real_time_data(study_roi):
    """Analyze real-time NDVI data from Google Earth Engine"""
    try:
        # Get a point within the ROI for time series extraction
        centroid = study_roi.centroid()
        
        # Define date range (last 2 years)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)
        
        # Format dates for EE
        start_date_ee = start_date.strftime('%Y-%m-%d')
        end_date_ee = end_date.strftime('%Y-%m-%d')
        
        # Create NDVI collection
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                      .filterBounds(study_roi)
                      .filterDate(start_date_ee, end_date_ee)
                      .map(mask_clouds)
                      .map(add_ndvi))
        
        # Create a time series of NDVI values
        st.info("🔄 Extracting NDVI time series...")
        ndvi_ts = collection.getRegion(centroid, 10).getInfo()
        
        # Process the data
        df = pd.DataFrame(ndvi_ts[1:], columns=ndvi_ts[0])
        df['datetime'] = pd.to_datetime(df['time'], unit='ms')
        df['NDVI'] = pd.to_numeric(df['NDVI'])
        df = df[['datetime', 'NDVI']].set_index('datetime')
        df = df.sort_index()
        
        # Remove outliers and invalid values
        df = df[(df['NDVI'] >= -1) & (df['NDVI'] <= 1)]
        
        # Resample to weekly data to reduce noise
        df_weekly = df.resample('W').mean()
        
        if len(df_weekly) == 0:
            st.warning("No NDVI data found for the selected area and time period.")
            return None
        
        # Create the static forex-style chart
        st.subheader("📈 Static Chart")
        static_fig = create_forex_style_chart(df_weekly, 'NDVI Time Series (Static Chart)')
        if static_fig:
            st.pyplot(static_fig)
            plt.close(static_fig)
        
        # Also show statistics
        st.subheader("📊 NDVI Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean", f"{df_weekly['NDVI'].mean():.3f}")
        col2.metric("Min", f"{df_weekly['NDVI'].min():.3f}")
        col3.metric("Max", f"{df_weekly['NDVI'].max():.3f}")
        col4.metric("Latest", f"{df_weekly['NDVI'].iloc[-1]:.3f}")
        
        # Add interactive chart option
        if st.button("Transform to Interactive Chart"):
            st.subheader("📈 Interactive Forex-Style Chart")
            interactive_fig = create_interactive_forex_chart(df_weekly, 'NDVI Time Series (Interactive Forex Style)')
            if interactive_fig:
                st.plotly_chart(interactive_fig, use_container_width=True)
        
        # Return the dataframe for further analysis
        return df_weekly
        
    except Exception as e:
        st.error(f"❌ Error in real-time data analysis: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Advanced NDVI Analysis", layout="wide")
    
    st.title("🌱 Advanced NDVI Analysis Dashboard")
    st.write("Select administrative boundaries or draw custom area, then generate NDVI visualizations")
    
    # Analysis mode selection
    analysis_mode = st.selectbox(
        "Select Analysis Mode:",
        ["GIF Visualization", "Real-time Analysis with Charts"]
    )
    
    # Boundary selection
    boundary_type = st.radio(
        "Select Boundary Type:",
        ["Draw on Map", "Country", "State/Province", "Municipality"],
        horizontal=True
    )
    
    # Initialize options
    country_options = get_boundary_names(FAO_GAUL, level=0)
    
    # Display selection based on boundary type
    selected_country = None
    selected_admin1 = None
    selected_admin2 = None
    
    if boundary_type != "Draw on Map":
        selected_country = st.selectbox("Select Country:", country_options)
        
        if boundary_type in ["State/Province", "Municipality"]:
            if selected_country:
                try:
                    country_feature = FAO_GAUL.filter(ee.Filter.eq('ADM0_NAME', selected_country)).first()
                    country_code = country_feature.get('ADM0_CODE').getInfo()
                    admin1_fc = get_admin_boundaries(level=1, country_code=country_code)
                    admin1_options = get_boundary_names(admin1_fc, level=1)
                    selected_admin1 = st.selectbox("Select State/Province:", admin1_options)
                    
                    if boundary_type == "Municipality" and selected_admin1:
                        admin1_feature = FAO_GAUL_ADMIN1.filter(ee.Filter.eq('ADM1_NAME', selected_admin1)).first()
                        admin1_code = admin1_feature.get('ADM1_CODE').getInfo()
                        admin2_fc = get_admin_boundaries(level=2, admin1_code=admin1_code)
                        admin2_options = get_boundary_names(admin2_fc, level=2)
                        selected_admin2 = st.selectbox("Select Municipality:", admin2_options)
                except Exception as e:
                    st.error(f"Error loading administrative boundaries: {str(e)}")
    
    # Display map for drawing
    st.subheader("📍 Draw Your Area of Interest")
    m = folium.Map(location=[20, 0], zoom_start=2)
    drawn_data = st_folium(m, width=700, height=400)
    
    # Custom data input
    st.subheader("📊 Optional: Add Your Own NDVI Data")
    default_data = """2023-01-05,0.106
2023-01-05,0.225
2023-01-05,-0.161
2023-01-07,0.103
2023-01-07,0.174
2023-01-10,0.207
2023-01-10,-0.299
2023-01-15,0.109
2023-01-15,0.235"""
    
    custom_data = st.text_area("Paste your NDVI data (date,value):", 
                             value=default_data, height=150)
    
    # Date range for GIF
    if analysis_mode == "GIF Visualization":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime(2023, 1, 1))
        with col2:
            end_date = st.date_input("End Date", datetime(2023, 3, 1))
    
    # Run analysis button
    if st.button("🚀 Generate Analysis", type="primary"):
        study_roi = None
        selected_name = ""
        
        # Get selected boundary
        if boundary_type == "Draw on Map":
            if drawn_data and drawn_data.get("last_active_drawing"):
                geometry = drawn_data["last_active_drawing"]["geometry"]
                study_roi = ee.Geometry(geometry)
                selected_name = "Custom Drawn Area"
            else:
                st.warning("Please draw a boundary on the map first!")
                return
        else:
            if boundary_type == "Country":
                if not selected_country:
                    st.warning("Please select a country first!")
                    return
                selected_name = selected_country
                study_roi = FAO_GAUL.filter(ee.Filter.eq('ADM0_NAME', selected_country)).geometry()
            elif boundary_type == "State/Province":
                if not selected_admin1:
                    st.warning("Please select a state/province first!")
                    return
                selected_name = selected_admin1
                study_roi = FAO_GAUL_ADMIN1.filter(ee.Filter.eq('ADM1_NAME', selected_admin1)).geometry()
            else:  # Municipality
                if not selected_admin2:
                    st.warning("Please select a municipality first!")
                    return
                selected_name = selected_admin2
                study_roi = FAO_GAUL_ADMIN2.filter(ee.Filter.eq('ADM2_NAME', selected_admin2)).geometry()
        
        # Show study area
        st.success(f"📍 Selected Area: {selected_name}")
        
        # Run the selected analysis mode
        if analysis_mode == "GIF Visualization":
            # Generate satellite GIF
            st.subheader("🛰️ Satellite NDVI Timelapse")
            generate_ndvi_gif(study_roi, start_date=str(start_date), end_date=str(end_date))

            # Process custom data
            if custom_data.strip():
                st.subheader("📊 Your NDVI Data Animation")
                analyze_custom_data(custom_data)
                
        elif analysis_mode == "Real-time Analysis with Charts":
            # Run real-time analysis
            st.subheader("📈 Real-time NDVI Analysis")
            analyze_real_time_data(study_roi)
            
            # Process custom data if available
            if custom_data.strip():
                st.subheader("📊 Your NDVI Data Analysis")
                analyze_custom_data(custom_data)

if __name__ == "__main__":
    main()
