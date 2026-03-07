# Optic Disc Detection

This repository contains various methodologies and algorithms for the automatic detection and segmentation of the Optic Disc in retinal fundus images. Accurately locating the optic disc is a fundamental step in automated diagnostic systems for retinal diseases such as Glaucoma, Diabetic Retinopathy, and Macular Degeneration.

This project explores both traditional image processing techniques and modern machine learning/deep learning approaches to compare their effectiveness.

## 📁 Repository Structure

The repository is organized into different folders based on the approach used for detection:

*   **`CNN/`**: Contains the deep learning approach using Convolutional Neural Networks (CNN) for robust optic disc detection and segmentation.
*   **`DBSCAN/`**: Implements the Density-Based Spatial Clustering of Applications with Noise (DBSCAN) algorithm to cluster pixel intensities and identify the optic disc region.
*   **`K-Mean Clustering/`**: Utilizes the K-Means unsupervised machine learning algorithm to segment the image and isolate the high-intensity area corresponding to the optic disc.
*   **`Morphological + Thresholding/`**: Employs traditional computer vision techniques, using a combination of image thresholding and morphological operations (like erosion and dilation) to extract the optic disc.
*   **`original/`**: Contains the original, unprocessed retinal fundus image dataset used for training and testing the algorithms.
*   **`summary.docx`**: A comprehensive summary document detailing the project methodology, experimental setup, and comparison of results across the different techniques.

## 📊 Results & Summary
A detailed breakdown of the accuracy, computational efficiency, and qualitative results for each method (CNN, DBSCAN, K-Means, and Morphological operations) can be found in the summary.docx file included in the root directory.
