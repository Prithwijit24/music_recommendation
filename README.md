# ✨ PulseMix: Your Next-Gen Music Recommendation System! ✨

🎵 Dive into the future of music discovery with **PulseMix** – transforming raw data into an intelligent, deployable recommendation platform that understands your unique taste! 🚀

## What is PulseMix?
PulseMix is designed with a layered approach to deliver unparalleled music recommendations:
- **💡 Content-based Retrieval:** Utilizing the rich features of the Million Song Dataset to find similar tracks.
- **🤝 Collaborative Filtering:** Ready for user interaction data to predict what you and others will love.
- **🧠 Hybrid Ranking:** Blending the best of both content and collaborative signals for spot-on suggestions.
- **📊 Classical ML Baselines:** Robust models for year prediction and foundational analysis.
- **🌌 Deep Learning:** Advanced representation learning to uncover hidden patterns in audio features.
- **🖥️ Intuitive Streamlit UI:** A sleek, interactive interface to experience your recommendations.
- **🐳 Production-Ready:** Packaged with Docker, Kubernetes manifests, and clear documentation for seamless deployment.

## 🏗️ Project Structure & Architecture Overview

```
.
├───README.md                     # 📖 This file! Overall project information.
├───.git/                         # 📜 Git version control metadata.
└───src/
    └───project_folder/
        ├───.github/              # 🌐 GitHub Actions workflows for CI/CD.
        │   └───workflows/
        │       ├───cd.yml        # Continuous Deployment pipeline.
        │       └───ci.yml        # Continuous Integration pipeline.
        ├───artifacts/            # 📦 Output directory for models and reports.
        │   ├───models/           # 🧠 Trained machine learning models.
        │   └───reports/          # 📊 Evaluation reports and visualizations.
        ├───conf/                 # ⚙️ Configuration files.
        │   └───config.yaml       # Main project configuration.
        ├───data/                 # 📥 Raw and processed data storage.
        ├───docker/               # 🐳 Docker-related files.
        ├───docs/                 # 📝 Documentation files.
        │   └───CI_CD.md          # CI/CD specific documentation.
        ├───k8s/                  # ☸️ Kubernetes deployment manifests.
        │   ├───deployment.yaml   # Kubernetes Deployment definition.
        │   └───service.yaml      # Kubernetes Service definition.
        ├───notebooks/            # 📊 Jupyter notebooks for experimentation.
        │   └───music_reco.ipynb  # Main music recommendation notebook.
        ├───scripts/              # 🐚 Utility scripts.
        ├───src/music_recommendation/ # 🚀 Core application source code.
        │   ├───data/             # Data loading and preprocessing utilities.
        │   ├───features/         # Feature engineering and transformations.
        │   ├───models/           # Implementations of various recommendation models.
        │   ├───pipelines/        # End-to-end ML pipelines (training, prediction).
        │   ├───services/         # Business logic and API services.
        │   ├───ui/               # Streamlit user interface components.
        │   └───utils/            # General helper utilities.
        ├───tests/                # 🧪 Unit and integration tests.
        │   └───test_pipeline.py  # Example test for the ML pipeline.
        ├───main.py               # 🚀 Main application entry point (e.g., FastAPI, CLI).
        ├───pyproject.toml        # 🐍 Project metadata and dependencies (Poetry/Hatch).
        ├───QUICKSTART.md         # ⚡ Quick start guide.
        ├───streamlit_app.py      # 🖥️ Streamlit application entry point.
        └───uv.lock               # 🔒 `uv` dependency lock file.
```

## ✅ What's Ready to Rock?

- **📦 Deployment-ready Python package structure:** Clean and organized for easy development and deployment.
- **⚡ `uv`-managed environment:** Fast and efficient dependency management.
- **🎶 Content-based Recommender:** Discover new music with nearest neighbors over compressed song embeddings.
- **🤖 ML Baseline:** Predict song years with a robust machine learning model on available datasets.
- ** profundo 🧠 Deep Autoencoder:** Learn powerful representations from audio features for advanced insights.
- **🌐 Streamlit UI:** Instantly preview and interact with recommendations.
- **🐳 Docker & Kubernetes:** Built-in support for containerization and orchestration for scalable deployments.

## 📈 Supercharge Your Recommendations (Data Needed!)

The current `YearPredictionMSD.csv` is great for item features, but to unlock the full potential of collaborative filtering and a truly personalized hybrid stack, we need **user interaction data**!

Ideally, a dataset structured like:
```
user_id,track_id,rating
```
Even better: implicit feedback like `play_count`, `like`, `skip`, `session_id`, and timestamps will revolutionize the recommendations! 💖

## 🚀 Get Started Today!

It's super easy to get PulseMix up and running. Ensure you have `uv` installed (`pip install uv`).

### 🛠️ Local Development

1.  **Sync your environment:**
    ```bash
    cd src/project_folder
    uv sync
    ```
2.  **Train your models:**
    ```bash
    uv run music-rec train
    ```
3.  **Launch the Streamlit UI!**
    ```bash
    uv run streamlit run streamlit_app.py
    ```
4.  **Run Tests:**
    ```bash
    uv run pytest tests/
    ```
5.  **Lint and Format (using Ruff):**
    ```bash
    uv run ruff check .
    uv run ruff format .
    ```

### 🐳 Docker Commands

1.  **Build the Docker image:**
    ```bash
    docker build -t pulsemix:latest src/project_folder
    ```
2.  **Run the Docker container (example for Streamlit):**
    ```bash
    docker run -p 8501:8501 pulsemix:latest uv run streamlit run streamlit_app.py
    ```

## 📚 Essential Resources

- **⚡ Quickstart Guide:** [QUICKSTART.md](src/project_folder/QUICKSTART.md)
- **🧪 Jupyter Notebook:** [notebooks/music_reco.ipynb](src/project_folder/notebooks/music_reco.ipynb)
- **⚙️ CI/CD Workflow:** [docs/CI_CD.md](src/project_folder/docs/CI_CD.md)
- **☸️ Kubernetes Manifests:**
    - [k8s/deployment.yaml](src/project_folder/k8s/deployment.yaml)
    - [k8s/service.yaml](src/project_folder/k8s/service.yaml)

---
Made with ❤️ for music lovers and data enthusiasts!
