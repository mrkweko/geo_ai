# Geo-AI

### Geospatial Intelligence for Smarter Urban and Environmental Planning

Geo-AI is an AI-powered geospatial analysis platform that uses satellite imagery, machine learning, and spatial data to generate insights about land use and land cover in Uganda.

The project explores how artificial intelligence and Earth observation data can help address real-world challenges such as urban expansion, environmental degradation, land-use change, and sustainable urban planning.

The current MVP focuses on **land-use and land-cover classification in Kampala, Uganda**, using Sentinel-2 satellite imagery and a Random Forest machine learning model.

---

## Project Overview

Urban areas are constantly changing. As cities expand, understanding how land is being used becomes increasingly important for urban planners, researchers, policymakers, and environmental organisations.

Geo-AI aims to make geospatial intelligence more accessible by transforming satellite imagery into understandable maps, statistics, and insights.

The current implementation demonstrates this through a web-based dashboard where users can explore a machine-learning-generated land-cover map of Kampala and examine the estimated distribution of different land-cover classes.

## Live Demo

**Application:** [Add your deployed Streamlit URL here]

**Source Code:** [GitHub Repository](https://github.com/mrkweko/geo_ai.git)

> The live application is an MVP demonstrating the current classification workflow. It uses previously exported classification data rather than performing live Google Earth Engine processing whenever the dashboard loads.

---

## Current Features

- **Interactive Map Explorer:** Explore the classified land-cover map of Kampala.
- **Land-cover Classification:** Visualise five broad land-cover categories.
- **Spatial Analytics:** Examine estimated land-cover area and percentage distribution.
- **Model Evaluation:** Review classification performance and evaluation metrics.
- **AI Methodology:** Understand the satellite data, feature engineering, sampling strategy, and Random Forest model used.
- **Visual Insights:** Explore charts and summaries derived from the classification results.

## Study Area

<details>
<summary><strong>Kampala, Uganda</strong></summary>

The current study focuses on Kampala, Uganda, covering an approximate region of 196.16 km².

The analysis uses Sentinel-2 satellite imagery to identify and classify different land-cover types within the study area.

</details>

## Land-cover Classes

The current model groups land cover into five broad categories:

| Class | Category                     | Description                                    |
| ----- | ---------------------------- | ---------------------------------------------- |
| 0     | Built-up                     | Buildings, roads, and other developed surfaces |
| 1     | Dense Vegetation             | Areas dominated by tree cover                  |
| 2     | Low Vegetation / Agriculture | Grassland, shrubland, and cropland             |
| 3     | Bare / Sparse Vegetation     | Bare ground and sparsely vegetated surfaces    |
| 4     | Water / Wetlands             | Permanent water bodies and herbaceous wetlands |

These categories are derived by grouping selected ESA WorldCover classes into broader categories for the current classification experiment.

## Technology Stack

| Technology           | Purpose                                            |
| -------------------- | -------------------------------------------------- |
| Python               | Data processing and machine learning               |
| Google Earth Engine  | Satellite imagery access and geospatial processing |
| Sentinel-2           | Earth observation imagery                          |
| ESA WorldCover       | Reference labels for training                      |
| Random Forest        | Land-cover classification                          |
| GeoPandas / Rasterio | Geospatial data handling                           |
| Streamlit            | Interactive web application                        |
| Folium               | Interactive mapping                                |
| Plotly               | Data visualisation                                 |
| NumPy & Pandas       | Numerical and tabular data processing              |

## Machine Learning Methodology

The current classification workflow consists of the following stages:

1. **Satellite Data Acquisition**  
   Sentinel-2 Surface Reflectance Harmonized imagery is accessed through Google Earth Engine.

2. **Preprocessing**  
   Cloud masking is applied to reduce the influence of clouds and cloud shadows.

3. **Feature Engineering**  
   Spectral bands and vegetation, built-up, and water indices are extracted:
   - NDVI — Normalized Difference Vegetation Index
   - NDBI — Normalized Difference Built-up Index
   - NDWI — Normalized Difference Water Index

4. **Training Data Preparation**  
   ESA WorldCover classes are remapped into five broader categories. Samples are collected and divided into training and validation sets using spatial blocks.

5. **Model Training**  
   A Random Forest classifier with 100 trees is trained using the selected spectral bands and indices.

6. **Classification**  
   The trained model is applied to the study area to generate a land-cover classification raster.

7. **Evaluation and Analysis**  
   The model is evaluated using a confusion matrix, overall accuracy, and Cohen's Kappa. Land-cover area estimates are also calculated.

8. **Visualisation**  
   The exported classification raster and associated statistics are displayed through the Streamlit dashboard.

## Model Performance

The current model produced the following validation results:

| Metric             | Result |
| ------------------ | -----: |
| Overall Accuracy   | 64.59% |
| Cohen's Kappa      | 0.5572 |
| Number of Trees    |    100 |
| Validation Samples |  1,121 |

**Important:** The evaluation uses reference labels derived from ESA WorldCover, not independently collected field-verified ground truth. The results should therefore be interpreted as agreement with the derived reference dataset rather than definitive real-world classification accuracy.

Some classes, particularly built-up and low vegetation/agriculture, exhibit confusion. Improving training data quality, class definitions, and model performance remains part of the project's development.

## Project Structure

```text
Geo-AI/
│
├── .streamlit/
│   └── config.toml
│
├── data/
│   ├── kampala_landcover_classification.tif
│   ├── kampala_landcover_statistics.csv
│   ├── kampala_classification_evaluation.csv
│   └── kampala_model_summary.csv
│
├── app.py
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.10 or later
- Git
- pip

### 1. Clone the repository

```bash
git clone https://github.com/mrkweko/geo_ai.git
```

### 2. Navigate into the project directory

```bash
cd geo_ai
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Linux / macOS**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

The application should open in your browser at:

```text
http://localhost:8501
```

## Data and Reproducibility

The current dashboard uses exported geospatial and statistical outputs stored in the `data/` directory.

The classification GeoTIFF contains the model's predicted land-cover classes. The CSV files contain the corresponding area statistics, evaluation metrics, and model summary.

Google Earth Engine is used in the model development and data-processing workflow. The deployed dashboard does not currently require users to authenticate with Google Earth Engine because it reads the exported outputs.

## Roadmap

Geo-AI is an evolving project. Future development will explore:

- [ ] Improve classification accuracy through better training data and model experimentation.
- [ ] Expand analysis beyond Kampala to other regions of Uganda.
- [ ] Introduce multi-temporal satellite imagery analysis.
- [ ] Detect and quantify urban expansion over time.
- [ ] Develop vegetation and environmental monitoring capabilities.
- [ ] Explore AI-assisted geospatial insights and natural-language queries.
- [ ] Introduce additional satellite data sources and geospatial datasets.
- [ ] Develop tools that support evidence-based urban and environmental planning.

## Limitations

The current implementation is an experimental MVP and has several limitations:

- Classification is currently limited to the Kampala study area.
- The model uses broad land-cover categories rather than highly detailed land-use labels.
- Accuracy is influenced by the quality and resolution of the reference dataset.
- The current application displays a previously generated classification rather than continuously updated satellite observations.
- Area estimates are model-derived and should not be treated as official land-cover measurements.

## Contributing

Contributions, suggestions, and discussions are welcome.

If you would like to contribute:

1. Fork the repository.
2. Create a new feature branch.
3. Make your changes.
4. Test your implementation.
5. Submit a pull request describing your contribution.

## Project Vision

Geo-AI seeks to explore the intersection of artificial intelligence, satellite technology, and geospatial science to make Earth observation data more useful for solving local challenges.

The long-term vision is to develop accessible geospatial intelligence tools that help communities, researchers, planners, and decision-makers better understand the changing environment.

**Built with curiosity, code, and a vision for smarter geospatial solutions in Africa.**

---

_Geo-AI — Turning satellite data into actionable intelligence._
