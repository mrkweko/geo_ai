
import streamlit as st
import pandas as pd
import numpy as np
import rasterio
import folium
import plotly.express as px

from streamlit_folium import st_folium
from rasterio.warp import transform_bounds
from rasterio.enums import Resampling

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Uganda Geo-AI",
    page_icon="🌍",
    layout="wide"
)

# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

CLASS_NAMES = {
    0: "Built-up",
    1: "Dense vegetation",
    2: "Low vegetation/agriculture",
    3: "Bare/sparse vegetation",
    4: "Water/wetlands"
}

CLASS_COLORS = {
    0: "#FF0000",
    1: "#006400",
    2: "#7CFC00",
    3: "#D2B48C",
    4: "#0000FF"
}

RASTER_PATH = "data/kampala_landcover_classification.tif"

STATS_PATH = "data/kampala_landcover_statistics.csv"

EVALUATION_PATH = "data/kampala_classification_evaluation.csv"

SUMMARY_PATH = "data/kampala_model_summary.csv"


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_statistics():
    return pd.read_csv(STATS_PATH)


@st.cache_data
def load_evaluation():
    return pd.read_csv(EVALUATION_PATH)


@st.cache_data
def load_summary():
    return pd.read_csv(SUMMARY_PATH)


@st.cache_data
def load_raster():
    with rasterio.open(RASTER_PATH) as src:

        image = src.read(
            1,
            out_shape=(
                max(1, int(src.height * 0.5)),
                max(1, int(src.width * 0.5))
            ),
            resampling=Resampling.nearest
        )

        bounds = transform_bounds(
            src.crs,
            "EPSG:4326",
            *src.bounds
        )

        return image, bounds


df_area = load_statistics()
df_evaluation = load_evaluation()
df_summary = load_summary()

raster, bounds = load_raster()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🌍 Uganda Geo-AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Map Explorer",
        "Land-cover Analytics",
        "AI Methodology"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "AI-powered geospatial intelligence "
    "for environmental and urban analysis."
)


# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

if page == "Overview":

    st.title("Uganda Geo-AI")

    st.subheader(
        "Understanding Kampala through satellite imagery and Artificial Intelligence"
    )

    st.write(
        """
        Uganda Geo-AI uses Sentinel-2 satellite imagery,
        spectral indices and a Random Forest machine learning
        model to classify land cover across Kampala.
        """
    )

    st.info(
        "Current MVP: Kampala land-cover classification."
    )

    total_area = df_area["area_km2"].sum()

    accuracy = df_summary.iloc[0]["overall_accuracy"]

    kappa = df_summary.iloc[0]["kappa"]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Mapped area",
        f"{total_area:.2f} km²"
    )

    col2.metric(
        "Model agreement",
        f"{accuracy * 100:.2f}%"
    )

    col3.metric(
        "Land-cover classes",
        "5"
    )

    st.markdown("---")

    st.subheader("Land-cover distribution")

    chart = px.pie(
        df_area,
        names="land_cover",
        values="area_km2",
        hole=0.4,
        title="Distribution of land cover in Kampala"
    )

    st.plotly_chart(chart, use_container_width=True)


# --------------------------------------------------
# MAP EXPLORER
# --------------------------------------------------

elif page == "Map Explorer":

    st.title("Kampala Land-cover Map")

    st.write(
        "Explore the land-cover classification generated "
        "by our Random Forest model."
    )

    selected_classes = st.multiselect(
        "Select land-cover classes",
        options=list(CLASS_NAMES.keys()),
        default=list(CLASS_NAMES.keys()),
        format_func=lambda x: CLASS_NAMES[x]
    )

    # Hide unselected classes
    display_raster = np.where(
        np.isin(raster, selected_classes),
        raster,
        255
    )

    # Convert raster to RGBA
    rgba = np.zeros(
        (*display_raster.shape, 4),
        dtype=np.uint8
    )

    for class_id, color in CLASS_COLORS.items():

        rgb = tuple(
            int(color[i:i+2], 16)
            for i in (1, 3, 5)
        )

        rgba[display_raster == class_id, :3] = rgb
        rgba[display_raster == class_id, 3] = 180

    # Map center
    west, south, east, north = bounds

    center = [
        (south + north) / 2,
        (west + east) / 2
    ]

    m = folium.Map(
        location=center,
        zoom_start=12,
        tiles="CartoDB positron"
    )

    folium.raster_layers.ImageOverlay(
        image=rgba,
        bounds=[
            [south, west],
            [north, east]
        ],
        opacity=0.8,
        interactive=True,
        cross_origin=False,
        name="AI Land-cover Classification"
    ).add_to(m)

    folium.LayerControl().add_to(m)

    st_folium(
        m,
        width=None,
        height=600,
        returned_objects=[]
    )

    st.markdown("### Map legend")

    legend_cols = st.columns(5)

    for col, (class_id, name) in zip(
        legend_cols,
        CLASS_NAMES.items()
    ):

        col.markdown(
            f"""
            <div style="
                background-color:{CLASS_COLORS[class_id]};
                padding:10px;
                border-radius:5px;
                color:white;
                text-align:center;
                font-size:12px;
            ">
                {name}
            </div>
            """,
            unsafe_allow_html=True
        )


# --------------------------------------------------
# LAND-COVER ANALYTICS
# --------------------------------------------------

elif page == "Land-cover Analytics":

    st.title("Land-cover Analytics")

    st.write(
        "Explore the estimated spatial distribution "
        "of Kampala's land-cover classes."
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.bar(
            df_area,
            x="land_cover",
            y="area_km2",
            color="land_cover",
            title="Land-cover area (km²)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.pie(
            df_area,
            names="land_cover",
            values="percentage",
            hole=0.4,
            title="Percentage distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("Detailed statistics")

    st.dataframe(
        df_area,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download statistics CSV",
        df_area.to_csv(index=False),
        file_name="kampala_landcover_statistics.csv",
        mime="text/csv"
    )


# --------------------------------------------------
# AI METHODOLOGY
# --------------------------------------------------

elif page == "AI Methodology":

    st.title("AI Model & Methodology")

    st.subheader("Satellite data")

    st.write(
        """
        The project uses Sentinel-2 Surface Reflectance
        Harmonized imagery from Google Earth Engine.
        """
    )

    st.subheader("Machine learning model")

    st.write(
        """
        A Random Forest classifier with 100 decision trees
        was trained using spectral bands and derived indices.
        """
    )

    st.subheader("Predictor variables")

    st.code(
        "B2, B3, B4, B8, B11, B12, NDVI, NDBI, NDWI"
    )

    st.subheader("Validation performance")

    accuracy = df_summary.iloc[0]["overall_accuracy"]

    kappa = df_summary.iloc[0]["kappa"]

    col1, col2 = st.columns(2)

    col1.metric(
        "Overall accuracy",
        f"{accuracy * 100:.2f}%"
    )

    col2.metric(
        "Kappa coefficient",
        f"{kappa:.4f}"
    )

    st.subheader("Class-level evaluation")

    st.dataframe(
        df_evaluation,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Important limitations")

    st.markdown(
        """
        - ESA WorldCover was used as reference labelling data.
        - Validation measures agreement with these reference labels,
          not independently verified ground truth.
        - Spectral similarities can cause confusion between
          built-up areas, vegetation and bare surfaces.
        - The current MVP is limited to the Kampala study area.
        - Results represent a particular satellite imagery period,
          not a live real-time land-cover assessment.
        """
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Uganda Geo-AI | An experimental geospatial "
    "intelligence project."
)
