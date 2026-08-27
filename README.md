# NetSight 🛰️🛣️
**Road Network Resilience & Disaster Management Framework**

NetSight is an industry-grade urban infrastructure analysis tool built for disaster management and structural intelligence. It leverages deep learning to extract road networks directly from satellite imagery and uses advanced graph theory to simulate cascading failures (e.g., floods, blasts, earthquakes) to determine the resilience of a city's infrastructure.

![NetSight Dashboard](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![NetworkX](https://img.shields.io/badge/NetworkX-Graph_Theory-blue?style=for-the-badge)

## 🌟 Key Features

1. **AI Road Segmentation**: Uses state-of-the-art UNet++ with an EfficientNet-B4 backbone to accurately segment roads from high-resolution satellite/aerial imagery.
2. **Graph Extraction**: Converts binary road masks into mathematical graphs `G(V, E)` using skeletonization, preserving spatial road curvature and junctions.
3. **Structural Intelligence (Centrality)**: Automatically calculates weighted Edge and Node Betweenness Centrality to identify **Gatekeeper Nodes**—the most critical intersections that act as single points of failure.
4. **Disaster Simulation (Ablation & Blast Radius)**: 
   - *Spatial Disasters*: Simulates localized events by removing nodes within a blast radius.
   - *Cascade Failures*: Identifies how traffic bottlenecks shift when primary arterial roads are destroyed.
5. **Resilience Indexing**: Calculates the formal network Resilience Index (`R = E_post / E_pre`) based on Global Efficiency degradation.

## 🚀 Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install torch torchvision networkx streamlit matplotlib numpy rasterio folium streamlit-folium segmentation-models-pytorch
```

### Running the Dashboard
NetSight features a fully interactive Phase IV Dashboard built with Streamlit. To launch the application locally:

```bash
streamlit run app.py
```
1. Open the provided `localhost` link in your browser.
2. Upload a satellite image tile (e.g., `.tif`, `.png`, `.jpg`).
3. View the segmented road network and run emergency simulations directly from the sidebar.

## 🧠 Model Training

If you want to train your own custom segmentation model on a specific region (e.g., SpaceNet data), you can use the built-in training pipeline:

```bash
python train.py --model_type industry --encoder_name efficientnet-b4 --epochs 60 --batch_size 4
```
*The pipeline supports mixed-precision training (AMP), Exponential Moving Average (EMA) for stability, and a hybrid loss function (BCE + Dice + clDice) designed specifically for thin structures like roads.*

## 📁 Repository Structure

* `app.py`: The main Streamlit interactive dashboard.
* `train.py`: Production-grade training pipeline.
* `evaluate.py`: Standalone evaluation script for model performance.
* `my_agents/src/`
  * `model.py`: Model architecture definitions (UNet++, ResNetUNet) and Test-Time Augmentation (TTA).
  * `mask_to_graph.py`: Skeletonization and spatial graph extraction logic.
  * `network_analysis.py`: Graph theory engine (Betweenness Centrality, Global Efficiency, Disaster Simulation).

## 🤝 Purpose
Designed in alignment with disaster management frameworks (like ISRO NNRMS), NetSight aims to provide urban planners and emergency responders with immediate, data-driven insights into infrastructure vulnerabilities *before* a disaster strikes.
