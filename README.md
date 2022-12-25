Spectral clustering model built in Sklearn for Clustering - Base problem category as per Ready Tensor specifications.

- spectral clustering
- clustering
- sklearn
- python
- pandas
- numpy
- docker

This is a Clustering Model that uses Spectral Clustering implemented through Sklearn.

Spectral clustering techniques make use of the spectrum (eigenvalues) of the similarity matrix of the data to perform dimensionality reduction before clustering in fewer dimensions

The data preprocessing step includes:

- for numerical variables
  - TruncatedSVD
  - Standard scale data

During the model development process, the algorithm was trained and evaluated on a variety of datasets such as iris, penguins, landsat_satellite, geture_phase_classification, vehicle_silhouettes, spambase, steel_plate_fault. Additionally, we also used synthetically generated datasets such as two concentric (noisy) circles, and unequal variance gaussian blobs.

This Clustering Model is written using Python as its programming language. ScikitLearn is used to implement the main algorithm, evaluate the model, and preprocess the data. Numpy, pandas, Sklearn, and feature-engine are used for the data preprocessing steps.
