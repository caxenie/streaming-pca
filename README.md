# Streaming PCA Algorithms Experiments

Reference implementation of various Streaming PCA algorithms.

### Algorithms

- Covariance Free algorithm for PCA, Weng et al. (2003). Candid Covariance-free Incremental Principal Component Analysis. IEEE Trans. Pattern Analysis and Machine Intelligence.
- Generalized Hebbian Algorithm for PCA, Sanger (1989). Optimal unsupervised learning in a single-layer linear feedforward neural network. Neural Networks Journal.
- Stochastic Gradient Ascent PCA - Exact, QR decomposition based version, Oja (1992). Principal components, Minor components, and linear neural networks. Neural Networks.
- Stochastic Gradient Ascent PCA - Fast Neural Network version, Oja (1992). Principal components, Minor components, and linear neural networks. Neural Networks.

### Experiments

In the testing dataset a datastream with the property that the eigenvalues of the input X are close to 1, 2, ..., d and the corresponding eigenvectors are close to the canonical basis of R^d, where d is the number of principal components to extract.
Analysis on the total stream ingestion algorithm execution time, accuracy of eigenvectors and eigenvalues and single event latency.
