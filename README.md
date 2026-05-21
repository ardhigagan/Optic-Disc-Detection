# Optic Disc Detection
### Retinal Fundus Image Analysis — Traditional CV & Deep Learning Comparison

A comparative study of four detection methodologies for automated optic disc localization and segmentation — a critical step in diagnosing Glaucoma, Diabetic Retinopathy, and Macular Degeneration.



## Tech Stack

Python · OpenCV · Scikit-Learn · TensorFlow · PyTorch



## Methods

| Module | Approach |
|---|---|
| `CNN/` | U-Net / VGG — supervised segmentation via encoder-decoder architecture |
| `K-Mean Clustering/` | Unsupervised color/intensity clustering with geometric boundary refinement |
| `DBSCAN/` | Density-based clustering; filters vascular noise and retinal artifacts |
| `Morphological + Thresholding/` | Adaptive thresholding with erosion, dilation, and morphological closing |



## Results

| Method | Dice | IoU | Pixel Acc. |
|---|:---:|:---:|:---:|
| CNN (VGG / U-Net) | 92.17% | 86.63% | 99.76% |
| K-Means Clustering | 85.41% | 77.98% | 99.34% |
| DBSCAN Clustering | 84.34% | 76.40% | 99.25% |
| Morphological + Thresholding | 83.71% | 76.83% | 98.76% |

The CNN model leads across all metrics. Unsupervised methods (K-Means, DBSCAN) and the training-free morphological pipeline remain competitive — demonstrating viability in annotation-scarce settings.



## Repository

| File | Purpose |
|---|---|
| `CNN/` | Deep learning training and inference scripts |
| `K-Mean Clustering/` | Unsupervised clustering pipeline |
| `DBSCAN/` | Density-based segmentation pipeline |
| `Morphological + Thresholding/` | Classical CV pipeline |
| `original/` | Raw fundus image dataset |
| `summary.docx` | Full methodology, math background, and analysis |
