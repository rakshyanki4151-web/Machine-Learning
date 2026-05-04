# X-HydraAI: Multi-Objective Sustainability Auditor

> **An explainable machine learning framework for balancing carbon emissions and water usage in sustainable data center operations.**

## 1. Project Title
**An Explainable Multi-Objective ML Framework for Carbon-Water Efficiency in Data Centers: A US-Nepal Comparative Study**

## 2. Brief Description
This project is an AI-driven framework that addresses the "Carbon-Water Paradox." While many sustainability audits focus solely on carbon, this project integrates cooling-driven water stress using Wet-Bulb Temperature (WBT) logic. The system identifies critical "Pivot Points" where clean energy grids become unsustainable due to high local thermal demands.

## 3. Core Features
- **Multi-Objective Labelling:** Combines IPCC Carbon Intensity (60%) and ASHRAE Water Stress (40%).
- **Explainable AI (XAI):** Uses SHAP to provide a transparent audit trail for every sustainability decision.
- **Global Transferability:** A zero-shot study validating the framework's portability from US grids to Nepal's unique climatic zones.

## 4. Dataset Structure
- **US Data:** 35,040 hourly records (2023) from 4 major grid regions (ERCOT, PJM, CAISO, WECC).
- **Nepal Data:** 8,760 hourly records for Kathmandu, synthesized into 4 regional study zones.
- **Key Features:** Carbon Intensity, Dry-Bulb Temp, Humidity, Renewable %, and Time.

## 5. File Structure
- **Project_Notebook.ipynb**: The central demonstration and research pipeline.
- **X-HydraAI_Dashboard.py**: Interactive Streamlit application for live diagnostic audits.
- **Project_Report.tex**: Full IEEE-formatted research manuscript.
- **data/**: Organized sub-folders for `us/` and `nepal/` master datasets.
- **models/**: Serialized frozen weights (`best_model.pkl`) and encoders.
- **figures/**: Complete set of 31 diagnostic artifacts and visual evidence.
- **Step scripts**: Modular preprocessing and training scripts used in development.
- **download_2023_EIA.py** / **download_2023_Weather.py**: Data acquisition scripts for reproducing raw US telemetry from official APIs.

## 6. Performance Summary
- **Best Model:** SVM (RBF Kernel)
- **Accuracy:** **91.4%**
- **F1-Score:** **0.916**
- **Macro-AUC (ROC):** **0.918**
- **Key Discovery:** Identified a 22°C (WBT) threshold where water stress overrides grid cleanliness in clean-grid regions like Nepal.

## 7. Technologies Used
- **Language:** Python 3.10+
- **Machine Learning:** Scikit-learn, Imbalanced-learn (SMOTE)
- **Explainability:** SHAP (SHapley Additive exPlanations)
- **Data Engineering:** Pandas, NumPy, Scipy
- **Deployment:** Streamlit (Diagnostic Auditor Dashboard)

## 8. How to Run

### Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run the Interactive Dashboard
```bash
streamlit run X-HydraAI_Dashboard.py
```
> The dashboard will automatically open in your local web browser at `http://localhost:8501`.
> Use the sidebar sliders to adjust temperature, humidity, and renewable energy parameters in real time.

### Step 3 — Run the Full Research Pipeline
Open `Project_Notebook.ipynb` in Jupyter Notebook or VS Code and run all cells in order.

### Step 4 — (Optional) Re-download Raw US Data
To reproduce the pipeline from raw government telemetry:
```bash
python download_2023_EIA.py
python download_2023_Weather.py
python Step3_Merge_and_Engineer.py
```

## 9. Submitted By
**Rakshya Nakarmi**
MSc Data Science and Computational Intelligence
Softwarica College of IT and E-commerce
