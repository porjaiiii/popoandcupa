import streamlit as st

import pandas as pd

import numpy as np

import pydeck as pdk

import plotly.express as px

import matplotlib.pyplot as plt

from datetime import datetime

from sklearn.cluster import DBSCAN

from sklearn.neighbors import KernelDensity


st.set_page_config(page_title="Traffy Fondue Hotspot Analysis", layout="wide")

st.title(' Traffy Fondue: PM2.5 Problem Hotspot Analysis')


# Load and prepare data

@st.cache_data

def load_data():


    try:


        data = pd.read_csv('mocdata.csv')

    except FileNotFoundError:

        st.error("Error: traffy_pm25_not_null.csv not found. Please check file path.")

        # คืนค่า DataFrame ที่มีคอลัมน์สำคัญตาม Schema ใหม่



       

        return pd.DataFrame({

            'latitude': [], 'longitude': [], 'pm25_value': [], 'report_hour': [],

            'district': [], 'timestamp': [], 'flag_ถนน': [] # เพิ่ม flag_ อื่นๆ ตามจำเป็น

        })


    # Clean and prepare data

    data['latitude_correct'] = pd.to_numeric(data['longitude'], errors='coerce')  # ใช้ค่าเดิมใน Lon เป็น Lat

    data['longitude_correct'] = pd.to_numeric(data['latitude'], errors='coerce') # ใช้ค่าเดิมใน Lat เป็น Lon

   

    data['latitude'] = data['latitude_correct']

    data['longitude'] = data['longitude_correct']

    # ... (ลบคอลัมน์ _correct ทิ้งทีหลังก็ได้)

   

   

    data['pm25_value'] = pd.to_numeric(data['pm25_value'], errors='coerce')

    data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')

    data = data.dropna(subset=['latitude', 'longitude', 'pm25_value', 'district'])

   

    # ใช้งานคอลัมน์ Flag

    flag_cols = [col for col in data.columns if col.startswith('flag_')]

    for col in flag_cols:

        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype(int)

   

    return data


# Load data

data = load_data()


# ตรวจสอบว่ามีข้อมูลหรือไม่

if data.empty:

    st.stop()



## Sidebar code


# Sidebar filters

st.sidebar.header('Filter Controls')


# 2. แก้ไข: เปลี่ยน Price Range เป็น PM2.5 Value Range

max_pm = data['pm25_value'].max()

min_pm = data['pm25_value'].min()

pm_range = st.sidebar.slider(

    'PM2.5 Value Range (μg/m³)',

    min_value=float(min_pm),

    max_value=float(max_pm),

    value=(float(min_pm), float(max_pm)),

    step=1.0,

    format='%.0f'

)


# 3. เพิ่ม: ตัวกรองชั่วโมง (Report Hour)

hour_min = int(data['report_hour'].min())

hour_max = int(data['report_hour'].max())

hour_range = st.sidebar.slider(

    "Report Hour of Day",

    hour_min, hour_max, (hour_min, hour_max)

)


# 4. เพิ่ม: ตัวกรองประเภทปัญหา (Flags)

flag_cols = [col for col in data.columns if col.startswith('flag_')]

selected_flags = st.sidebar.multiselect(

    "Select Problem Types (Flags)",

    options=flag_cols,

    default=flag_cols[0:3] if len(flag_cols) >= 3 else flag_cols # เลือก 3 อันดับแรกเป็นค่าเริ่มต้น

)


# 5. เพิ่ม: ตัวกรองเขต (District)

district_list = data['district'].unique()

selected_district = st.sidebar.multiselect(

    "Select District",

    options=district_list,

    default=district_list[:5] if len(district_list) >= 5 else district_list

)

st.sidebar.header('Map View Options')

show_hexagon = st.sidebar.checkbox('Show Hexagon/Aggregation Layer', value=True)



# DBSCAN clustering parameters

st.sidebar.header('DBSCAN Parameters')

# DBSCAN parameters (ไม่เปลี่ยนแปลง)

eps_degrees = st.sidebar.slider(

    "eps (degree)",

    min_value=0.001, max_value=0.005, value=0.002, step=0.001, format='%.3f'

)

min_samples = st.sidebar.slider(

    "min_samples",

    min_value=2, max_value=10 , value=3

)

num_top_clusters = st.sidebar.slider(

    "Number of Top Clusters to Show",

    min_value=1, max_value=10 , value=5

)


# Map style selection (ไม่เปลี่ยนแปลง)

map_style = st.sidebar.selectbox(

    'Select Base Map Style',

    options=['Dark', 'Light', 'Road', 'Satellite'],

    index=0

)


# KDE parameters (ไม่เปลี่ยนแปลง)

st.sidebar.header('KDE Parameters')

bandwidth = st.sidebar.slider(

    "Bandwidth",

    min_value=0.001, max_value=0.020, value=0.005, step=0.001, format='%.3f'

)



## Main panel code


# Define map style dictionary (ไม่เปลี่ยนแปลง)

MAP_STYLES = {

    'Dark': 'mapbox://styles/mapbox/dark-v10',

    'Light': 'mapbox://styles/mapbox/light-v10',

    'Road': 'mapbox://styles/mapbox/streets-v11',

    'Satellite': 'mapbox://styles/mapbox/satellite-v9'

}


# 6. แก้ไข: การกรองข้อมูล

filtered_data = data.copy()

filtered_data = filtered_data[

    (filtered_data['pm25_value'] >= pm_range[0]) &

    (filtered_data['pm25_value'] <= pm_range[1]) &

    (filtered_data['report_hour'] >= hour_range[0]) &

    (filtered_data['report_hour'] <= hour_range[1]) &

    (filtered_data['district'].isin(selected_district))

].copy()


# Apply Flag filter (ใช้ OR logic: แสดงปัญหาใด ๆ ที่ถูกเลือก)

if selected_flags:

    flag_condition = (filtered_data[selected_flags] == 1).any(axis=1)

    filtered_data = filtered_data[flag_condition]



def get_dominant_flags(row, flag_cols):

    # ค้นหา Flags ที่มีค่าเป็น 1

    active_flags = [col.replace('flag_', '') for col in flag_cols if row[col] == 1]

    if not active_flags:

        return 'No specific flag'

    # รวม Flags ที่ใช้งานอยู่ (แสดงได้สูงสุด 3 อันแรก) สำหรับ Tooltip

    return ', '.join(active_flags[:3])


flag_cols_all = [col for col in data.columns if col.startswith('flag_')]

filtered_data['dominant_flags'] = filtered_data.apply(

    lambda row: get_dominant_flags(row, flag_cols_all), axis=1

)

# *** END NEW LOGIC ***


# Main content - Key metrics

st.subheader(f"Total Reports Matching Filters: {len(filtered_data):,}")

if filtered_data.empty:

    st.warning("No reports match the current filter selection.")

    st.stop()

   

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric("Total Reports", len(filtered_data))

with col2:

    # 7. แก้ไข: เปลี่ยน Average Price เป็น Average PM2.5 Value

    st.metric("Avg PM2.5 Value", f"{filtered_data['pm25_value'].mean():.1f} μg/m³")

with col3:

    # 8. แก้ไข: เปลี่ยน Average Reviews เป็น Most Frequent Report Hour

    most_freq_hour = filtered_data['report_hour'].mode().iloc[0] if not filtered_data['report_hour'].mode().empty else 'N/A'

    st.metric("Most Frequent Hour", f"Hour {most_freq_hour}")

with col4:

    # 9. แก้ไข: เปลี่ยน Neighborhoods เป็น Districts

    st.metric("Districts Covered", filtered_data['district'].nunique())



# 10. แก้ไข: เปลี่ยน Price Distribution เป็น PM2.5 Distribution

st.header('PM2.5 Value Distribution')

fig_hist = px.histogram(

    filtered_data,

    x='pm25_value',

    nbins=50,  # You can adjust the number of bins

    title='Distribution of PM2.5 Values for Filtered Reports',

    labels={'pm25_value': 'PM2.5 Value (μg/m³)', 'count': 'Number of Reports'},

)

st.plotly_chart(fig_hist)



# 11. แก้ไข: เปลี่ยน Price by neighborhood เป็น PM2.5 by District

st.header('PM2.5 Value by District and Report Count')

pm_by_district = filtered_data.groupby('district')['pm25_value'].agg(['mean', 'count']).reset_index()

pm_by_district.columns = ['district', 'avg_pm25', 'report_count']


fig_scatter = px.scatter(pm_by_district,

                         x='report_count',

                         y='avg_pm25',

                         text='district',

                         title='Average PM2.5 vs Number of Reports by District',

                         labels={'report_count': 'Number of Reports',

                                'avg_pm25': 'Average PM2.5 (μg/m³)'})

fig_scatter.update_traces(textposition='top center')

st.plotly_chart(fig_scatter)



# Hotspot Analysis

st.header('Hotspot Analysis (DBSCAN)')


try:

    # Perform DBSCAN clustering

    coords = filtered_data[['latitude', 'longitude']]

    db = DBSCAN(eps=eps_degrees, min_samples=min_samples).fit(coords)

   

    # Add cluster labels to dataframe

    filtered_data['cluster'] = db.labels_

   

    # Analyze clusters

    clusters_count = filtered_data['cluster'].value_counts()

    clusters_count = clusters_count[clusters_count.index != -1]  # Exclude noise points

   

    if clusters_count.empty:

        st.warning("DBSCAN found no significant clusters (noise only). Try adjusting eps or min_samples.")

        viz_data = pd.DataFrame()

    else:

        top_clusters = clusters_count.head(num_top_clusters)

       

        # Generate colors for clusters

        unique_clusters = filtered_data[filtered_data['cluster'].isin(top_clusters.index)]['cluster'].unique()

        colormap = plt.get_cmap('hsv')

        cluster_colors = {cluster: [int(x*255) for x in colormap(i/len(unique_clusters))[:3]] + [160]

                          for i, cluster in enumerate(unique_clusters)}

       

        # Create visualization dataframe

        viz_data = filtered_data[filtered_data['cluster'].isin(top_clusters.index)].copy()

        viz_data['color'] = viz_data['cluster'].map(cluster_colors)

       

        # 12. แก้ไข: เปลี่ยน name เป็น organization และ price เป็น pm25_value ใน tooltip

        cluster_layer = pdk.Layer(

            "ScatterplotLayer",

            viz_data,

            get_position=['longitude', 'latitude'],

            get_color='color',

            get_radius=50,

            pickable=True

        )

       

        st.pydeck_chart(

            pdk.Deck(

                layers=[cluster_layer],

                initial_view_state=pdk.ViewState(

                    latitude=filtered_data['latitude'].mean(),

                    longitude=filtered_data['longitude'].mean(),

                    zoom=11,

                    pitch=0

                ),

                map_style=MAP_STYLES[map_style],

                tooltip={

                    "html": "<b>Cluster:</b> {cluster}<br/>"

                            "<b>PM2.5 Value:</b> {pm25_value:.1f} μg/m³<br/>" # แก้ไข

                            "<b>Organization:</b> {organization}<br/>"         # แก้ไข

                            "<b>District:</b> {district}"                      # แก้ไข

                }

            ),

            height=600

        )


        # Heatmap Layer (ใช้ viz_data ที่มี Cluster Labels)

        st.subheader('DBSCAN Cluster Heatmap')

        heatmapLayer = pdk.Layer(

            "HeatmapLayer",

            data=viz_data,

            get_position=['longitude', 'latitude'],

            opacity=0.5,

            radiusPixels=50,

        )

        st.pydeck_chart(

            pdk.Deck(

                layers=[heatmapLayer],

                initial_view_state=pdk.ViewState(

                    latitude=filtered_data['latitude'].mean(),

                    longitude=filtered_data['longitude'].mean(),

                    zoom=11,

                    pitch=0

                ),

                map_style=MAP_STYLES[map_style],

            ),

            height=600

        )


        # Hexagon Layer (ใช้ viz_data)

        st.subheader('DBSCAN Cluster Hexagon Layer')

        hexagonLayer = pdk.Layer(

            'HexagonLayer',

            data=viz_data,

            get_position='[longitude, latitude]',

            radius=1000,

            opacity=0.5,

            extruded=False, # 2D

        )

        st.pydeck_chart(

            pdk.Deck(

                layers=[hexagonLayer],

                initial_view_state=pdk.ViewState(

                    latitude=filtered_data['latitude'].mean(),

                    longitude=filtered_data['longitude'].mean(),

                    zoom=11,

                    pitch=0

                ),

                map_style=MAP_STYLES[map_style],

            ),

            height=600

        )

       

        # View Cluster Statistics

        st.subheader('Cluster Statistics')

        with st.expander("View Cluster Statistics", expanded=False):

            for cluster_id in top_clusters.index:

                cluster_data = viz_data[viz_data['cluster'] == cluster_id]

                # 13. แก้ไข: เปลี่ยน avg_price เป็น avg_pm25 และ avg_reviews เป็น avg_report_hour

                avg_pm25 = cluster_data['pm25_value'].mean()

                avg_report_hour = cluster_data['report_hour'].mean()

               

                st.write(f"**Cluster {cluster_id}** ({len(cluster_data)} reports)")

               

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric("Avg PM2.5 Value", f"{avg_pm25:.1f} μg/m³")

                with col2:

                    st.metric("Avg Report Hour", f"{avg_report_hour:.1f}")

                with col3:

                    st.metric("Districts", cluster_data['district'].nunique())

               

                st.write("**Top Districts:**", ', '.join(cluster_data['district'].unique()[:10]))

                if cluster_data['district'].nunique() > 10:

                    st.write(f"... and {cluster_data['district'].nunique() - 10} more")

               

                st.write("**Top Flags:**", ', '.join(cluster_data[flag_cols].sum().nlargest(3).index.str.replace('flag_', '')))

               

                st.divider()


except Exception as e:

    st.error(f"Error in DBSCAN analysis: {e}")

    st.exception(e)


st.header('Kernel Density Estimation (KDE) Analysis VS Hexagon Analysis')


try:

    # Prepare coordinates and Fit KDE model

    coords = filtered_data[['latitude', 'longitude']].values

    kde = KernelDensity(bandwidth = bandwidth, kernel='gaussian')

    kde.fit(coords)

   

    # Calculate density score and normalize

    log_density = kde.score_samples(coords)

    density = np.exp(log_density)

    filtered_data['density'] = density

    filtered_data['density_normalized'] = (density - density.min()) / (density.max() - density.min())

   

    # Format density for tooltip

    filtered_data['density_formatted'] = filtered_data['density'].apply(lambda x: f"{x:.4f}")

   

    # --- 1. Density Color Mapping (ใช้สำหรับ Scatterplot Standalone และ Combined Map Scatterplot) ---

    def density_to_color(density_normalized):

        """Convert normalized density to RGB color (blue to red gradient)"""

        r = int(density_normalized * 255)

        g = int((1 - abs(2 * density_normalized - 1)) * 255)

        b = int((1 - density_normalized) * 255)

        return [r, g, b, 180]

    filtered_data['density_color'] = filtered_data['density_normalized'].apply(density_to_color)

   

   

    # --- 2. PM2.5 Value Color Mapping (ใช้สำหรับ Hexagon Layer - เขียว-แดง) ---

    pm_min = filtered_data['pm25_value'].min()

    pm_max = filtered_data['pm25_value'].max()

   

    # Normalize PM2.5 Value

    filtered_data['pm25_normalized'] = (filtered_data['pm25_value'] - pm_min) / (pm_max - pm_min)


    def pm25_to_color(pm25_normalized):

        """Convert normalized PM2.5 (0 to 1) to RGB color (Green to Red gradient)"""

        r = int(pm25_normalized * 255)

        g = int((1 - pm25_normalized) * 255)

        b = 0

        return [r, g, b, 200]

   

    # Apply PM2.5 color mapping

    filtered_data['pm25_color'] = filtered_data['pm25_normalized'].apply(pm25_to_color)

   

    # --- 3. Flag Color Mapping (ไม่ถูกใช้ในการแสดงผลนี้ แต่ยังคงโค้ดไว้) ---

    if selected_flags:

        filtered_data['flag_risk'] = filtered_data[selected_flags].mean(axis=1).fillna(0)

    else:

        filtered_data['flag_risk'] = 0

       

    def flag_to_color(flag_risk):

        """Map total flag risk to color (Blue=Low, Yellow=High)"""

        r = int(flag_risk * 255)

        g = int(flag_risk * 255)

        b = int((1 - flag_risk) * 100)

        return [r, g, b, 220]


    filtered_data['flag_color'] = filtered_data['flag_risk'].apply(flag_to_color)

   

   

    # KDE Scatterplot Layer (Standalone map)
    c1,c2 = st.columns(2)
    with c1:
        st.subheader('KDE Ticket_Report') # <<< เปลี่ยนชื่อกลับไป

        kde_layer_standalone = pdk.Layer(

            "ScatterplotLayer",

            filtered_data,

            get_position="[longitude, latitude]",

            get_color='density_color', # <<< ใช้สี Density (ฟ้า-แดง)

            get_radius=50,

            opacity=0.5,

            pickable=True,

        )

    

        st.pydeck_chart(

            pdk.Deck(

                layers=[kde_layer_standalone],

                initial_view_state=pdk.ViewState(

                    latitude=filtered_data['latitude'].mean(),

                    longitude=filtered_data['longitude'].mean(),

                    zoom=11,

                    pitch=45

                ),

                map_style=MAP_STYLES[map_style],

                tooltip={

                "html": "<b>Flags:</b> {dominant_flags}<br/>"

                        "<b>PM2.5 Value:</b> {pm25_value:.1f} μg/m³<br/>"

                        "<b>Density:</b> {density_formatted}"

                }

            ),

            height=600

        )
    with c2:
        st.subheader('Hexagon PM Value')


            # 1. KDE Scatterplot Layer (Layer ที่อยู่ด้านบนสุด - สีตาม Density)

        density_scatter_layer = pdk.Layer(

                "ScatterplotLayer",

                filtered_data,

                get_position="[longitude, latitude]",

                get_color='density_color', # <<< ใช้สี Density (ฟ้า-แดง)

                get_radius=20,  # <<< ลดรัศมีเหลือ 20 เพื่อลดการบดบัง

                opacity=0.8,

                pickable=True,

                tooltip={

                    "html": "<b>Flags:</b> {dominant_flags}<br/>"

                            "<b>PM2.5 Value:</b> {pm25_value:.1f} μg/m³<br/>"

                            "<b>Density:</b> {density_formatted}"

                }

        )

        

        layers_to_show = []

        

            # 2. Hexagon Layer (Layer ฐาน - สีตาม PM2.5 Value)

        if show_hexagon:

            general_hexagon_layer = pdk.Layer(

                'HexagonLayer',

                data=filtered_data,

                get_position='[longitude, latitude]',

                radius=500,

                # ความสูงยังคงเป็นตามจำนวนรายงาน (Count)

                elevation_scale=200, # ให้สูงไว้ก่อน

                elevation_range=[0, 1000],

            

                # <<< การแก้ไขที่สำคัญ: ใช้ String 'mean' แทน Enum ที่พัง >>>

                get_color='pm25_color',

                color_aggregation='mean', # เปลี่ยนจาก pdk.types.AGGR_FUNCTIONS.MEAN เป็น 'mean'

            

                pickable=True,

                extruded=True,

                opacity=0.3,

                coverage=1.0

            )

            layers_to_show.append(general_hexagon_layer)


            # 3. Scatterplot ต้องอยู่ด้านบนเสมอ

        layers_to_show.append(density_scatter_layer)



        st.pydeck_chart(

            pdk.Deck(

                layers=layers_to_show,

                initial_view_state=pdk.ViewState(

                    latitude=filtered_data['latitude'].mean(),

                    longitude=filtered_data['longitude'].mean(),

                    zoom=11,

                    pitch=45

                ),

                map_style=MAP_STYLES[map_style],

            ),

            height=600

        )

    # --- End Combined Map ---



    # Display statistics

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric("Min Density", f"{filtered_data['density'].min():.4f}")

    with col2:

        st.metric("Mean Density", f"{filtered_data['density'].mean():.4f}")

    with col3:

        st.metric("Max Density", f"{filtered_data['density'].max():.4f}")

   

    # View top high-density locations

    with st.expander("View Highest Density Locations", expanded=False):

        st.subheader('Top 10 Highest Density Locations')

        top_density = filtered_data.nlargest(10, 'density')[

            ['organization', 'district', 'pm25_value', 'density', 'report_hour', 'dominant_flags']

        ].reset_index(drop=True)

        top_density['density'] = top_density['density'].apply(lambda x: f"{x:.4f}")

        st.dataframe(top_density, use_container_width=True)


except Exception as e:

    st.error(f"Error in KDE analysis: {e}")

    st.exception(e) 