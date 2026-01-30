# Heart Disease Prediction Application

**University of Aix-Marseille | M2 Software Development Project (2025-2026)**

## 👥 Group Members

* **Member 1:** Al khatib Lara
* **Member 2:** Brousse Antoine
* **Member 3:** Gouaref Hamza

## 📖 Project Overview

This application is a data-driven tool designed to process medical indicators and predict heart disease risk using Machine Learning. This project follows rigorous software engineering standards:

* **Python** for modular backend logic.
* **Scikit-Learn** implementation of a Random Forest Classifier.
* **Logging & Type Hinting** for robust code maintenance.
* **Docker Containerization** for reproducible deployment environments.

---

## 📂 Project Architecture

The project is organized into a modular structure to separate business logic from the user interface:

```text
heart-disease-app/
├── data/              # Raw data storage
├── src/               # Backend logic (Preprocessing, ML Training)
│   ├── __init__.py
│   └── processing.py
├── app/               # User Interface (Streamlit)
│   └── streamlit_app.py
├── tests/             # Unit tests (Pytest)
├── Dockerfile         # Container image configuration
├── compose.yaml       # Service orchestration
├── requirements.txt   # Dependency list
└── README.md          # Project documentation

```

---

## 🚀 Execution Instructions

### 1. Using Docker (Recommended)

The easiest way to run the project is using Docker Compose. This ensures all dependencies are correctly configured.

```bash
# Build and launch the container
docker-compose up --build

```

Once launched, visit: **`http://localhost:8501`**

### 2. Manual Installation (Conda)

If you prefer running it in a local environment using Conda:

```bash
# Create and activate the environment
conda create --name heart-env python=3.9
conda activate heart-env

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app/streamlit_app.py

```



## 🔗 Repository

**Public Git Link:** https://github.com/H-GRF/Heart-Disease/tree/main

