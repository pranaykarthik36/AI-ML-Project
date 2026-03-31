# 🌫️ AirLens — Air Quality Intelligence Dashboard for Indian Cities

> A machine learning-powered dashboard that forecasts AQI, identifies pollution drivers, and clusters Indian cities by their pollution profiles — built with Prophet, Random Forest, K-Means, and Streamlit.

---

## What It Does

AirLens turns raw air quality data into actionable intelligence:

- **30-day AQI forecast** using Facebook Prophet (time-series model with seasonality)
- **Pollutant contribution analysis** using Random Forest + feature importance
- **City clustering** using K-Means to group cities by pollution profile
- **Year-over-year heatmaps** to reveal seasonal patterns
- **Interactive Streamlit dashboard** — city selector drives all charts in real time

---

## Project Structure

```
airlens/
├── app.py                  # Main Streamlit dashboard
├── requirements.txt        # Python dependencies
├── README.md
├── data/
│   ├── raw/                # Original Kaggle CSV files (not committed)
│   └── processed/          # Cleaned, merged dataset
├── notebooks/
│   ├── 01_EDA.ipynb        # Exploratory data analysis
│   └── 02_Modeling.ipynb   # Model training and evaluation
└── src/
    ├── preprocessing.py    # Data cleaning and feature engineering
    ├── models.py           # Prophet, RF, K-Means wrappers
    └── utils.py            # Helper functions
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/airlens.git
cd airlens
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get the dataset

Download **"Air Quality Data in India (2015–2020)"** from Kaggle:

```
https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
```

Place the CSV files inside `data/raw/`. The app will use synthetic data automatically if no files are found (useful for demo purposes).

### 5. Run the dashboard

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser.

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Dashboard UI |
| `prophet` | Time-series AQI forecasting |
| `scikit-learn` | Random Forest, K-Means, preprocessing |
| `shap` | Pollutant contribution explainability |
| `plotly` | Interactive charts |
| `pandas` / `numpy` | Data manipulation |
| `joblib` | Model serialization |

Install all at once:

```bash
pip install streamlit prophet scikit-learn shap plotly pandas numpy joblib
```

> **Note for Windows users:** Prophet requires `pystan`. If installation fails, try:
> `conda install -c conda-forge prophet`

---

## How to Use

1. Launch the app with `streamlit run app.py`
2. Use the **city selector** in the left sidebar to choose a city
3. The dashboard updates all four panels automatically:
   - **Top row** — current AQI, 30-day average, severe days count, 7-day forecast
   - **Forecast chart** — Prophet model output with trend and confidence interval
   - **Pollutant bar chart** — which pollutants are driving AQI for this city
   - **Heatmap** — month-by-year AQI patterns (spot seasonal spikes)
   - **Cluster chart** — which cities share a similar pollution profile

---

## Models at a Glance

### Prophet (Time-Series Forecasting)
Trained per city on daily AQI values (2015–2020). Captures yearly and weekly seasonality with a 30-day horizon. Uncertainty intervals shown on the forecast chart.

### Random Forest Classifier
Trained on pollutant concentrations (PM2.5, PM10, NO2, SO2, O3, CO) to predict AQI category (Good / Satisfactory / Moderate / Poor / Severe). Feature importances visualized as the pollutant contribution chart.

### K-Means Clustering
Aggregates monthly AQI per city into a city x time matrix, scales it, then clusters into 4 groups. Cities in the same cluster share similar seasonal pollution patterns — useful for policy comparisons.

---

## Data Sources

- [CPCB / Kaggle — Air Quality Data in India](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) — primary dataset
- [Open-Meteo API](https://open-meteo.com/) — weather features (wind, humidity, temperature) — free, no API key required

---

## Known Limitations

- The current version uses synthetic data if the Kaggle CSV is absent. Replace with real data for meaningful forecasts.
- Prophet training can be slow (~10-20s per city) on first load; results are cached by Streamlit after that.
- Weather API integration is simulated in the current build. To connect live weather, update `get_weather_data()` in `app.py` with an Open-Meteo API call.

---

## License

MIT License. Free to use, modify, and distribute with attribution.

---

## Author

Built as a Bring Your Own Project (BYOP) capstone for an ML/AI course.
Contact: `<your-email>` | GitHub: `<your-username>`
