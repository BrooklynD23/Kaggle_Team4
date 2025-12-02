# 🎓 Student Success Prediction System

**Kaggle Team 4 Semester Project**

This project implements a machine learning pipeline to predict student outcomes (Dropout, Enrolled, Graduate) based on academic, demographic, and socio-economic data. The goal is to identify at-risk students early to enable timely intervention.

## 🚀 Getting Started

### Prerequisites
*   Python 3.8+
*   pip

### Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Pipeline

To train the models and generate a report, run:

```bash
python run_pipeline.py
```

This will:
1.  Load and clean `dataset.csv`.
2.  Train multiple models (Baselines, Random Forest, Ensembles).
3.  Select the best performing model.
4.  Save the best model to `models/saved_models/`.
5.  Print a comprehensive evaluation report.

### Exploring the Notebook

For interactive exploration and visualization, open `notebooks/main.ipynb`:

```bash
jupyter notebook notebooks/main.ipynb
```

## 📂 Project Structure

```
├── dataset.csv                 # Raw dataset
├── run_pipeline.py             # Main execution script
├── requirements.txt            # Python dependencies
├── INSIGHTS_GUIDE.md           # Guide for interpreting model results
├── QUICK_REFERENCE.md          # Quick model selection guide
├── REFINED_ML_ARCHITECTURE.md  # Detailed architecture documentation
├── notebooks/
│   └── main.ipynb              # Exploratory Data Analysis & Prototyping
└── src/
    ├── data/
    │   └── cleaning.py         # Data cleaning logic
    ├── models/
    │   ├── baselines.py        # Simple baseline models
    │   ├── tree_models.py      # Random Forest, XGBoost, etc.
    │   └── ensembles.py        # Voting, Stacking, Cascading ensembles
    ├── evaluation/
    │   ├── interpretation.py   # SHAP, Feature Importance
    │   └── fairness.py         # Fairness auditing tools
    └── train_pipeline.py       # Main training pipeline logic
```

## 🧠 Model Insights

We prioritize **explainability** and **fairness**.
*   See [INSIGHTS_GUIDE.md](INSIGHTS_GUIDE.md) for how to interpret predictions and audit fairness.
*   See [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for a decision framework on model selection.

## 🏗️ Architecture

The system follows a modular architecture:
1.  **Data Layer**: `src/data` handles loading and cleaning.
2.  **Model Layer**: `src/models` contains model definitions (Tier 1-4).
3.  **Evaluation Layer**: `src/evaluation` provides tools for deep analysis.
4.  **Pipeline**: `src/train_pipeline.py` orchestrates the entire flow.

See [REFINED_ML_ARCHITECTURE.md](REFINED_ML_ARCHITECTURE.md) for the detailed design philosophy.

## ⚡ Live Demo Stack (API + UI)

Keep stakeholders on live metrics with the FastAPI + React demo:

1. **Start the API** (serves metrics, predictions, artifact plots)
   ```bash
   uvicorn app.api.main:app --reload
   ```
2. **Start the UI** (Vite + React + React Query)
   ```bash
   cd app/ui
   npm install
   cp env.example .env.local  # adjust VITE_API_BASE_URL if needed
   npm run dev
   ```
3. Visit `http://localhost:5173` for KPI cards, refreshed plots, experiment timeline, and a "try a prediction" panel wired to `/predict`.