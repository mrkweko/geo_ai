from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import rasterio
import streamlit as st
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from streamlit_folium import st_folium


st.set_page_config(
    page_title="Geo-AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

CLASS_NAMES = {
    0: "Built-up",
    1: "Dense vegetation",
    2: "Low vegetation/agriculture",
    3: "Bare/sparse vegetation",
    4: "Water/wetlands",
}
CLASS_COLORS = {
    0: "#E74C3C",
    1: "#176B3A",
    2: "#8BC34A",
    3: "#C9A66B",
    4: "#2878B5",
}
PLOTLY_TEMPLATE = "plotly_white"
PLOTLY_COLORS = [CLASS_COLORS[class_id] for class_id in CLASS_NAMES]


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #F5F8FA; }
        .block-container { max-width: 1450px; padding-top: 2rem; padding-bottom: 3rem; }
        [data-testid="stSidebar"] { background: #101D2B; border-right: 1px solid #263747; }
        [data-testid="stSidebar"] * { color: #E5EDF3; }
        h1, h2, h3 { color: #172E3B; letter-spacing: -0.5px; }
        [data-testid="stMetric"] {
            background: white; border: 1px solid #E2EAF0; padding: 18px;
            border-radius: 12px; box-shadow: 0 2px 8px rgba(20,45,60,.04);
        }
        [data-testid="stMetricLabel"] { color: #647B83; }
        [data-testid="stMetricValue"] { color: #172E3B; font-weight: 700; }
        .stButton > button {
            background: #16A085; color: white; border: 0; border-radius: 8px;
            font-weight: 600;
        }
        .stButton > button:hover { background: #117A65; color: white; }
        hr { border-color: #E2EAF0; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_csv(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Required data file not found: {path}")
    return pd.read_csv(path)


@st.cache_data
def load_raster() -> tuple[np.ndarray, tuple[float, float, float, float], str, float | None]:
    path = DATA_DIR / "kampala_landcover_classification.tif"
    if not path.exists():
        raise FileNotFoundError(f"Required raster file not found: {path}")
    with rasterio.open(path) as src:
        image = src.read(
            1,
            out_shape=(max(1, int(src.height * 0.5)), max(1, int(src.width * 0.5))),
            resampling=Resampling.nearest,
            masked=True,
        )
        bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        return image.filled(255), bounds, str(src.crs), src.res[0]


def load_app_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, tuple, str, float | None]:
    try:
        area = load_csv("kampala_landcover_statistics.csv")
        evaluation = load_csv("kampala_classification_evaluation.csv")
        summary = load_csv("kampala_model_summary.csv")
        raster, bounds, crs, resolution = load_raster()
    except (FileNotFoundError, pd.errors.ParserError, ValueError) as error:
        st.error(str(error), icon=":material/error:")
        st.stop()

    required_area = {"class_id", "land_cover", "area_km2", "percentage"}
    required_eval = {"class_id", "land_cover", "producer_accuracy", "consumer_accuracy"}
    if not required_area.issubset(area.columns) or not required_eval.issubset(evaluation.columns):
        st.error("The supplied CSV files do not contain the expected columns.", icon=":material/error:")
        st.stop()
    if summary.empty or not {"overall_accuracy", "kappa"}.issubset(summary.columns):
        st.error("The model summary file is missing validation metrics.", icon=":material/error:")
        st.stop()
    try:
        for frame, columns in (
            (area, ["class_id", "area_km2", "percentage"]),
            (evaluation, ["class_id", "producer_accuracy", "consumer_accuracy"]),
            (summary, ["overall_accuracy", "kappa"]),
        ):
            for column in columns:
                frame[column] = pd.to_numeric(frame[column], errors="raise")
    except (TypeError, ValueError) as error:
        st.error(f"The supplied CSV files contain invalid numeric values: {error}", icon=":material/error:")
        st.stop()
    return area, evaluation, summary, raster, bounds, crs, resolution


def plot_layout(fig: go.Figure, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=title,
        margin={"l": 10, "r": 10, "t": 55 if title else 20, "b": 10},
        legend_title_text="",
    )
    return fig


def metric_row(area: pd.DataFrame, summary: pd.DataFrame) -> None:
    total_area = area["area_km2"].sum()
    accuracy = float(summary.iloc[0]["overall_accuracy"])
    kappa = float(summary.iloc[0]["kappa"])
    cols = st.columns(4)
    cols[0].metric("Mapped area", f"{total_area:.2f} km²", border=True)
    cols[1].metric("Land-cover classes", str(area["class_id"].nunique()), border=True)
    cols[2].metric("Overall accuracy", f"{accuracy:.2%}", border=True)
    cols[3].metric("Kappa coefficient", f"{kappa:.4f}", border=True)


def navigate_to(page: str) -> None:
    st.session_state.page = page


def show_sidebar() -> str:
    if "page" not in st.session_state:
        st.session_state.page = "Overview"
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 12px 0 24px">
              <div style="display:flex;align-items:center;gap:12px">
                <div style="background:#16A085;padding:10px;border-radius:10px;font-size:22px">🌍</div>
                <div>
                  <div style="font-size:17px;font-weight:800;color:white">UGANDA GEO-AI</div>
                  <div style="font-size:10px;color:#9BAFBE;letter-spacing:1px">GEOSPATIAL INTELLIGENCE</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("WORKSPACE")
        st.radio(
            "Navigation",
            ["Overview", "Map Explorer", "Land-cover Analytics", "AI Insights", "AI Methodology", "About"],
            key="page",
            label_visibility="collapsed",
        )
        st.markdown(
            """
            <div style="margin-top:30px;padding:14px;background:#1B3342;border-radius:10px">
              <div style="color:#64D9B5;font-size:11px">CURRENT STUDY AREA</div>
              <div style="color:white;font-weight:700;margin-top:5px">Kampala, Uganda</div>
              <div style="color:#A8BCCB;font-size:11px;margin-top:5px">Land-cover classification MVP</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='margin-top:35px;color:#8297A6;font-size:10px'>GEO-AI · 2026</div>", unsafe_allow_html=True)
    return st.session_state.page


def overview(area: pd.DataFrame, summary: pd.DataFrame) -> None:
    st.title("See the land. Understand the change.")
    st.subheader("Transforming satellite imagery into actionable geospatial intelligence for Africa.")
    st.write(
        "Geo-AI explores how Earth observation data and machine learning can support "
        "evidence-based urban and environmental decision-making. The current pilot maps land cover in Kampala."
    )
    action_col, method_col = st.columns([1, 1])
    with action_col:
        st.button(
            "Explore the Kampala map",
            icon=":material/map:",
            type="primary",
            on_click=navigate_to,
            args=("Map Explorer",),
        )
    with method_col:
        st.button(
            "See how the model works",
            icon=":material/science:",
            on_click=navigate_to,
            args=("AI Methodology",),
        )
    st.info("Current MVP: a precomputed Kampala land-cover classification. Results are model-derived estimates, not independently verified ground truth.")
    metric_row(area, summary)

    st.subheader("Kampala land-cover distribution")
    fig = px.pie(area, names="land_cover", values="area_km2", hole=0.52, color="land_cover", color_discrete_sequence=PLOTLY_COLORS)
    st.plotly_chart(plot_layout(fig, "Estimated area by class"), width="stretch")
    largest = area.loc[area["area_km2"].idxmax()]
    st.caption(f"{largest['land_cover']} is the largest classified category at {largest['area_km2']:.2f} km² ({largest['percentage']:.2f}% of the mapped raster area).")

    with st.container(border=True):
        st.subheader("A Kampala pilot with a wider ambition")
        st.write(
            "The pilot demonstrates a repeatable workflow for turning satellite imagery into interpretable land-cover information. "
            "Future work could extend the platform across Uganda and add vegetation, agriculture, wetlands, and multi-temporal monitoring—capabilities that are not implemented in this MVP."
        )


def raster_rgba(raster: np.ndarray, selected_classes: list[int]) -> np.ndarray:
    rgba = np.zeros((*raster.shape, 4), dtype=np.uint8)
    visible = np.isin(raster, selected_classes)
    for class_id, color in CLASS_COLORS.items():
        rgb = tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))
        pixels = (raster == class_id) & visible
        rgba[pixels, :3] = rgb
        rgba[pixels, 3] = 255
    return rgba


def inspect_raster_cell(
    raster: np.ndarray,
    bounds: tuple[float, float, float, float],
    latitude: float,
    longitude: float,
) -> tuple[int, int, int] | None:
    west, south, east, north = bounds
    if not south <= latitude <= north or not west <= longitude <= east:
        return None
    row = min(raster.shape[0] - 1, int((north - latitude) / (north - south) * raster.shape[0]))
    column = min(raster.shape[1] - 1, int((longitude - west) / (east - west) * raster.shape[1]))
    class_id = int(raster[row, column])
    if class_id not in CLASS_NAMES:
        return None
    return row, column, class_id


def map_explorer(raster: np.ndarray, bounds: tuple, crs: str, resolution: float | None) -> None:
    st.title("Map Explorer")
    st.write("Explore the precomputed Kampala land-cover classification and its spatial context.")
    controls = st.columns([1.3, 1, 1])
    with controls[0]:
        selected = st.multiselect("Visible classes", list(CLASS_NAMES), default=list(CLASS_NAMES), format_func=CLASS_NAMES.get)
    with controls[1]:
        opacity = st.slider("Classification opacity", 0.1, 1.0, 0.8, 0.05)
    with controls[2]:
        show_overlay = st.toggle("Show classification overlay", value=True)
        basemap = st.selectbox("Basemap", ["CartoDB positron", "OpenStreetMap", "Esri satellite"])

    west, south, east, north = bounds
    center = [(south + north) / 2, (west + east) / 2]
    reset_map = st.button("Reset map view", icon=":material/center_focus_strong:")
    tiles = {
        "CartoDB positron": (
            "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
            "© OpenStreetMap contributors © CARTO",
        ),
        "OpenStreetMap": ("OpenStreetMap", None),
        "Esri satellite": (
            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            "Esri",
        ),
    }
    with st.spinner("Loading Kampala map and classification overlay...", show_time=True):
        tile_url, attribution = tiles[basemap]
        m = folium.Map(location=center, zoom_start=12, tiles=tile_url, attr=attribution)
        if show_overlay and selected:
            folium.raster_layers.ImageOverlay(
                image=raster_rgba(raster, selected),
                bounds=[[south, west], [north, east]],
                opacity=opacity,
                interactive=True,
                cross_origin=False,
                name="AI land-cover classification",
            ).add_to(m)
        folium.LayerControl().add_to(m)
        map_state = st_folium(
            m,
            width=None,
            height=610,
            returned_objects=["last_clicked"],
            key=f"map-{reset_map}",
        )
    clicked = map_state.get("last_clicked")
    if clicked:
        inspected = inspect_raster_cell(raster, bounds, clicked["lat"], clicked["lng"])
        if inspected is None:
            st.info("The clicked location is outside the classified raster or falls on a NoData cell.", icon=":material/location_off:")
        else:
            row, column, class_id = inspected
            st.success(
                f"Nearest classified cell: {CLASS_NAMES[class_id]} (row {row + 1}, column {column + 1}).",
                icon=":material/location_on:",
            )
            st.caption(
                f"Clicked coordinate: {clicked['lat']:.5f}, {clicked['lng']:.5f} · "
                "Cell lookup uses the displayed nearest-neighbour raster grid."
            )

    st.subheader("Map legend")
    legend = st.columns(len(CLASS_NAMES))
    for column, class_id in zip(legend, CLASS_NAMES):
        column.markdown(
            f"<div style='background:{CLASS_COLORS[class_id]};padding:10px 6px;border-radius:5px;color:white;text-align:center;font-size:12px'>{CLASS_NAMES[class_id]}</div>",
            unsafe_allow_html=True,
        )
    with st.container(border=True):
        st.subheader("Map information")
        st.write("Study area: Kampala, Uganda")
        st.write("Data source: precomputed classification raster from Sentinel-2-derived features")
        st.write(f"Raster CRS: {crs}")
        if resolution:
            st.write(f"Native pixel size: approximately {resolution:g} map units (from raster metadata)")
        st.caption("The displayed raster is downsampled for browser performance. No classification date or imagery period is available in the supplied metadata.")


def analytics(area: pd.DataFrame) -> None:
    st.title("Land-cover Analytics")
    st.write("Explore the estimated spatial distribution of Kampala's classified land-cover categories.")
    left, right = st.columns(2)
    with left:
        fig = px.bar(area, x="land_cover", y="area_km2", color="land_cover", color_discrete_sequence=PLOTLY_COLORS, labels={"area_km2": "Area (km²)", "land_cover": ""})
        st.plotly_chart(plot_layout(fig, "Area by class"), width="stretch")
    with right:
        fig = px.pie(area, names="land_cover", values="percentage", hole=0.5, color="land_cover", color_discrete_sequence=PLOTLY_COLORS)
        st.plotly_chart(plot_layout(fig, "Percentage distribution"), width="stretch")

    st.subheader("Key observations")
    largest = area.loc[area["area_km2"].idxmax()]
    smallest = area.loc[area["area_km2"].idxmin()]
    built_up = area.loc[area["land_cover"].eq("Built-up"), "percentage"].sum()
    vegetation = area.loc[area["land_cover"].isin(["Dense vegetation", "Low vegetation/agriculture"]), "percentage"].sum()
    water = area.loc[area["land_cover"].eq("Water/wetlands"), "percentage"].sum()
    st.info(
        f"{largest['land_cover']} is the largest category ({largest['percentage']:.2f}%). "
        f"{smallest['land_cover']} is the smallest ({smallest['percentage']:.2f}%). "
        f"Built-up accounts for {built_up:.2f}%, while the two vegetation categories together account for {vegetation:.2f}%. "
        f"Water/wetlands account for {water:.2f}% of the mapped area.",
        title="Direct observations from the classification",
    )
    st.caption("These are descriptive statistics only; they do not establish change, degradation, risk, or causation.")

    st.subheader("Detailed statistics")
    display = area[["class_id", "land_cover", "area_km2", "percentage"]].copy()
    display["area_km2"] = display["area_km2"].round(2)
    display["percentage"] = display["percentage"].round(2)
    st.dataframe(display, hide_index=True, width="stretch", alt="Kampala land-cover statistics")
    st.download_button("Download statistics CSV", area.to_csv(index=False), "kampala_landcover_statistics.csv", "text/csv", icon=":material/download:", width="content")
    percentage_total = float(area["percentage"].sum())
    area_total = float(area["area_km2"].sum())
    st.caption(f"Data check: percentages sum to {percentage_total:.2f}% and the mapped raster totals {area_total:.2f} km². Pixel-grid and masking effects can make this differ from an administrative boundary area.")


def insights(area: pd.DataFrame) -> None:
    st.title("AI Insights")
    st.write("Programmatic observations derived from the supplied classification statistics—without external AI services or unsupported claims.")
    largest = area.loc[area["area_km2"].idxmax()]
    water = area.loc[area["land_cover"].eq("Water/wetlands")].iloc[0]
    vegetation = area.loc[area["land_cover"].eq("Dense vegetation")].iloc[0]
    for title, text in [
        ("Largest classified category", f"{largest['land_cover']} covers approximately {largest['area_km2']:.2f} km², or {largest['percentage']:.2f}% of the mapped raster area."),
        ("Vegetation signal", f"Dense vegetation accounts for approximately {vegetation['area_km2']:.2f} km² ({vegetation['percentage']:.2f}%)."),
        ("Water and wetlands", f"Water/wetlands account for approximately {water['percentage']:.2f}% of the mapped area."),
    ]:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(text)
    fig = px.bar(area, x="percentage", y="land_cover", orientation="h", color="land_cover", color_discrete_sequence=PLOTLY_COLORS, labels={"percentage": "Share of mapped area (%)", "land_cover": ""})
    st.plotly_chart(plot_layout(fig, "Class shares"), width="stretch")
    st.subheader("What this does—and does not—tell us")
    st.markdown(
        "- **Direct observation:** the model assigns pixels to five land-cover categories and the figures above summarize those assignments.\n"
        "- **Requires additional evidence:** urban expansion, deforestation, wetland degradation, and trends require comparable multi-temporal classifications and validation.\n"
        "- **Future capability:** a future version could add time-series change detection, uncertainty surfaces, and targeted monitoring workflows."
    )


def methodology(evaluation: pd.DataFrame, summary: pd.DataFrame) -> None:
    st.title("AI Methodology")
    st.write("A transparent view of the current data, features, model configuration, and validation evidence.")
    with st.container(border=True):
        st.subheader("Data sources")
        st.write("Sentinel-2 Surface Reflectance Harmonized imagery was acquired and processed through Google Earth Engine. ESA WorldCover was used as a reference labelling source. The supplied metadata does not establish exact imagery dates or a WorldCover product version.")
    with st.container(border=True):
        st.subheader("Feature engineering")
        st.write("The classifier uses spectral bands B2, B3, B4, B8, B11, and B12, plus NDVI, NDBI, and NDWI. These indices provide vegetation-, built-up-, and water-related spectral signals; they are not standalone guarantees of land-cover identity.")
    left, right = st.columns(2)
    with left:
        st.subheader("Random Forest configuration")
        st.markdown("- 100 decision trees\n- 3,879 training samples\n- 1,121 validation samples\n- Spatial block-based train/validation split")
    with right:
        st.subheader("Validation performance")
        st.metric("Overall accuracy", f"{float(summary.iloc[0]['overall_accuracy']):.2%}", border=True)
        st.metric("Kappa coefficient", f"{float(summary.iloc[0]['kappa']):.4f}", border=True)

    st.subheader("Class-level evaluation")
    eval_display = evaluation.copy()
    eval_display["producer_accuracy"] = eval_display["producer_accuracy"].map(lambda value: f"{value:.2%}")
    eval_display["consumer_accuracy"] = eval_display["consumer_accuracy"].map(lambda value: f"{value:.2%}")
    st.dataframe(eval_display, hide_index=True, width="stretch", alt="Class-level validation accuracy")
    st.caption("Producer's accuracy is recall: how much of a reference class was recovered. Consumer's accuracy is precision: how often predictions for a class were correct against the reference labels.")

    st.subheader("Confusion matrix")
    matrix = np.array([[118, 17, 30, 51, 0], [23, 156, 64, 7, 1], [21, 57, 109, 14, 3], [63, 8, 35, 141, 2], [0, 0, 0, 1, 200]])
    labels = list(CLASS_NAMES.values())
    heatmap = go.Figure(go.Heatmap(z=matrix, x=labels, y=labels, colorscale="Teal", text=matrix, texttemplate="%{text}", hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"))
    heatmap.update_xaxes(title="Predicted class")
    heatmap.update_yaxes(title="Actual class")
    st.plotly_chart(plot_layout(heatmap), width="stretch")

    st.subheader("Feature importance")
    st.info("No feature-importance values were included in the supplied model exports, so none are fabricated here.", icon=":material/info:")
    with st.container(border=True):
        st.subheader("Limitations")
        st.markdown("- ESA WorldCover labels are reference proxies, not perfect ground truth.\n- Validation measures agreement with those reference labels.\n- Spectral overlap can confuse built-up, vegetation, and bare surfaces.\n- The current model has moderate validation performance and covers Kampala only.\n- No temporal change detection is currently implemented.")


def about() -> None:
    st.title("Geo-AI")
    st.write(
        "Geo-AI is an emerging geospatial intelligence initiative exploring how artificial intelligence and Earth observation data can help address environmental and urban-development challenges in Africa."
    )
    with st.container(border=True):
        st.subheader("Mission")
        st.write("Make satellite-derived evidence easier to understand and apply in planning, environmental monitoring, and research.")
    st.subheader("Current MVP")
    st.markdown("- Kampala land-cover classification\n- Interactive raster map and class filtering\n- Area statistics and downloadable CSV\n- Transparent model validation and limitations")
    st.subheader("Technology stack")
    st.markdown("Streamlit · Python · Rasterio · NumPy · Pandas · Plotly · Folium · Sentinel-2 · Google Earth Engine · Random Forest")
    st.subheader("Long-term vision")
    st.write("The platform could expand to wider Africa coverage, multi-temporal monitoring, agriculture and vegetation indicators, wetland and water-resource monitoring, and decision-support workflows. These are future applications, not current MVP features.")


inject_styles()
page = show_sidebar()
area_df, evaluation_df, summary_df, raster_data, raster_bounds, raster_crs, raster_resolution = load_app_data()

if page == "Overview":
    overview(area_df, summary_df)
elif page == "Map Explorer":
    map_explorer(raster_data, raster_bounds, raster_crs, raster_resolution)
elif page == "Land-cover Analytics":
    analytics(area_df)
elif page == "AI Insights":
    insights(area_df)
elif page == "AI Methodology":
    methodology(evaluation_df, summary_df)
else:
    about()

st.markdown("---")
st.caption("Geo-AI · Experimental geospatial intelligence for Kampala · Model-derived results should be interpreted with the stated limitations.")
