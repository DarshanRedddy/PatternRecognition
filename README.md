# Kohonen Self-Organizing Map (SOM) - Circular Input Distribution

This project implements a 1D Kohonen Self-Organizing Map (SOM) trained on a dataset of points sampled from **two concentric circles**.

It visualizes how the weight vectors evolve during training using a neighborhood function.

---

## 🔧 Features

- Generate synthetic 2D data points from two circles.
- Initialize weight vectors in a rectangular region.
- Train a SOM with decreasing learning rate and neighborhood.
- Visualize:
  - Input distribution & initial weights
  - Weight evolution over time (t = 0, 100, 1000)
  - Neighbor connections

---

## 📊 Output Images

| Description | Output |
|------------|--------|
| Initial weights and input samples | ![t=0](outputs/turn_in_2_0.png) |
| After 100 iterations | ![t=100](outputs/turn_in_3_100.png) |
| After 1000 iterations | ![t=1000](outputs/turn_in_3_1000.png) |

> ℹ️ After downloading the PNG files from Google Colab, upload them to the `outputs/` folder in this repo and make sure to replace the paths above if needed.

---

## 🚀 How to Run on Google Colab

1. Open the notebook or Python script in [Google Colab](https://colab.research.google.com/)
2. Run the code cells.
3. To download the output plots:

```python
from google.colab import files
files.download("outputs/turn_in_3_1000.png")  # Or use shutil to zip and download all
