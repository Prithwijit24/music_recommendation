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

## 🏗️ Architecture Overview

```text
data/              #  raw datasets (e.g., YearPredictionMSD.csv)
src/music_recommendation/
  data/          # 📥 Data loaders and schema definitions
  features/      # ⚙️ Preprocessing and embeddings generation
  models/        # 🧠 Diverse recommendation models (content, collaborative, hybrid, ML, DL)
  pipelines/     #  orchestrate 🚀 training and evaluation workflows
  services/      # 📈 Metrics, reporting, and analysis
  ui/            # 🎨 The interactive Streamlit application
artifacts/
  models/        # 💾 Persisted trained models
  reports/       # 📄 Evaluation summaries and visualizations
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

It's super easy to get PulseMix up and running:

```bash
uv sync                      # Sync your environment
uv run music-rec train       # Train your models
uv run streamlit run streamlit_app.py # Launch the Streamlit UI!
```

## 📚 Essential Resources

- **⚡ Quickstart Guide:** [QUICKSTART.md](QUICKSTART.md)
- **🧪 Jupyter Notebook:** [notebooks/music_reco.ipynb](notebooks/music_reco.ipynb)
- **⚙️ CI/CD Workflow:** [docs/CI_CD.md](docs/CI_CD.md)
- **☸️ Kubernetes Manifests:**
    - [k8s/deployment.yaml](/home/prithwijit/programming/python/imp_projects/music_recommendation/src/project_folder/k8s/deployment.yaml)
    - [k8s/service.yaml](/home/prithwijit/programming/python/imp_projects/music_recommendation/src/project_folder/k8s/service.yaml)

---
Made with ❤️ for music lovers and data enthusiasts!