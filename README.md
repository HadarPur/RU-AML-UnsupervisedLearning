# Unsupervised Learning

Timot Baruch, Hadar Pur

Submitted as a project report for Advanced Machine Learning course, IDC, 2024

# Background, data exploration, and preprocessing

- **Dataset Overview:** The MNIST dataset comprises 70,000 images. Each image represents a single, centered, handwritten digit (0-9) and is size-normalized and grey-scaled. The images are presented as a 28x28 pixel grid, resulting in 784 features.

![Digits Data Examples](plots/figure1.png)

- **Size-Normalization and Centering:** During initial preprocessing, the dataset underwent size-normalization and centering to ensure uniformity in digit representations and simplify the learning process for machine learning models.

- **Train-Test Split:** The data was split into a training set and a test set, with the test set size set to 20% of the training set. Measures were taken to prevent overlapping writers between the two sets.

- **Data Standardization:** Despite applying StandardScaler to standardize dataset features with a mean of 0 and a standard deviation of 1, no significant enhancement in classifier performance or accuracy was observed. Consequently, the decision was made to proceed without scaling the data. Additionally, during Principal Component Analysis (PCA) for dimensionality reduction, the scaled data exhibited lower cumulative variance compared to unscaled data for a given number of components.

# Dimensionality reduction

We utilized various dimensionality reduction techniques to enhance the understanding of the MNIST dataset and visualize it in a 2-dimensional feature space. The following brief explanations outline each algorithm's role:

- **PCA (Principal Component Analysis):** PCA identifies the principal components that explain the maximum variance in the data, providing a linear projection to reduce dimensionality while retaining key information about the dataset's variability.

- **TSVD (Truncated Singular Value Decomposition):** TSVD decomposes the original data matrix into singular vectors and values, allowing for dimensionality reduction to capture dominant features and patterns in the data.

- **t-SNE (t-Distributed Stochastic Neighbor Embedding):** t-SNE is a nonlinear dimensionality reduction technique effective in revealing clusters and structures in the data while preserving pairwise similarities between data points.

- **UMAP (Uniform Manifold Approximation and Projection):** UMAP is a nonlinear dimensionality reduction method that preserves both local and global structures in high-dimensional data, capturing complex relationships within the dataset.

For all these techniques, default parameters were applied to obtain a baseline reduction, capturing essential patterns and structures within the data.

![PCA TSVD Visualization](plots/figure2.png)
![UMAP and t-SNE Visualization](plots/figure3.png)

## Hyper-parameters Tuning On DR Methods

For the PCA and t-SNE methods, we conducted extensive hyperparameter testing to identify the most effective combination for optimizing performance. The key hyperparameters explored for each DR method are outlined below according to the DR used. The best run is highlighted in bold in each table.
\newline Due to exceptionally long running times, UMAP  testing was excluded from this phase of analysis. Also, TSVD provided unsatisfactory results in the visualization step, so it was excluded from this phase.

- **PCA:** he `n_components` parameter determines the number of principal components to retain. It influences the amount of variance retained in the reduced space. The `svd_solver` parameter, like 'randomized' or 'full', defines the algorithm used for singular value decomposition and impacts the computational efficiency.

<div align="center">

| n_components | svd_solver | explained_variance | reconstruction_error |
|--------------|------------|--------------------|----------------------|
| 5            | auto       | 0.333790           | 2910.795606          |
| 5            | full       | 0.333790           | 2910.796008          |
| 5            | randomized | 0.333790           | 2910.796227          |
| 10           | auto       | 0.489385           | 2230.018559          |
| 10           | full       | 0.489385           | 2230.019752          |
| 10           | randomized | 0.489385           | 2230.018290          |
| 50           | auto       | 0.825549           | 765.217696           |
| 50           | full       | 0.825614           | 765.071916           |
| 50           | randomized | 0.825601           | 765.134985           |
| 100          | auto       | 0.914748           | 374.082438           |
| 100          | full       | 0.915059           | 373.083830           |
| 100          | randomized | 0.914669           | 374.453304           |
| 200          | auto       | 0.966205           | 148.766095           |
| **200**      | **full**   | **0.966587**       | **147.483888**       |
| 200          | randomized | 0.966184           | 148.949969           |

*Table 1: PCA Hyper-parameters tuning Results*

</div>

![PCA Reconstruction of digits according to different n_components](plots/figure4.png)

![PCA Explained Variance for different number of n_components](plots/figure5.png)

- **t-SNE:** The `n_components` parameter specifies the dimensionality of the embedded space. A lower value may capture global structures, while higher values might preserve more local relationships. The `learning_rate` influences the step size during optimization, affecting the convergence and final layout of the data in the low-dimensional space. The `perplexity` is a hyperparameter that balances the attention given to local and global aspects of the data. It is used to define the effective number of neighbors for each data point during the dimensionality reduction process. 

<div align="center">

| n_components | perplexity | learning_rate | min_divergence |
|--------------|------------|---------------|----------------|
| 2            | 20         | 10            | 4.388708       |
| 2            | 20         | 200           | 3.163473       |
| 2            | 20         | auto          | 2.801678       |
| ...          | ...        | ...           | ...            |
| 2            | 150        | 10            | 2.298883       |
| 2            | 150        | 200           | 2.297647       |
| 2            | 150        | auto          | 2.156140       |
| 3            | 20         | 10            | 2.156140       |
| 3            | 20         | 200           | 2.156140       |
| 3            | 20         | auto          | 2.156140       |
| ...          | ...        | ...           | ...            |
| 3            | 150        | 10            | 2.066741       |
| 3            | 150        | 200           | 2.026015       |
| **3**        | **150**    | **auto**      | **1.925341**   |

*Table 2: t-SNE Hyper-parameters tuning Results*

</div>

![Effect of Various perplexity values on KL Divergence](plots/figure6.png)

### Selecting Optimal Evaluation Methods for Each DR Technique

In selecting the optimal run for each dimensionality reduction (DR) algorithm, we considered specific criteria tailored to the nature of each method. For PCA, we assessed reconstruction error and explained variance ratio, while for t-SNE, we examined the KL divergence.

#### PCA:

- **Reconstruction Error:** Examined to quantify the difference between the original data and the reconstructed data in the reduced dimensions.  
  As shown in Figure~\ref{fig:reconstruction}, better reconstruction of the digits was experienced with a higher number of components.

- **Explained Variance Ratio:** Assessed to measure the proportion of variance retained in the reduced dimensions, indicating the effectiveness of dimensionality reduction.  
  As shown in Figure~\ref{fig:variance}, the relationship between the number of components and the cumulative variance ratio is illustrated. With 40 principal components, approximately 80% cumulative variance is achieved, whereas with 200 principal components, nearly 97% cumulative variance is attained.

#### t-SNE:

- **KL Divergence:** Evaluated to measure the difference between the original distribution and the distribution in the embedded space. t-SNE minimizes KL divergence to preserve local structures and similarities between data points.  
  As shown in Figure~\ref{fig:kl_divergence}, the relationship between KL Divergence and perplexity is demonstrated, showing that a higher perplexity tends to correspond to a lower KL Divergence. To manage computational resources, we used perplexity at a maximum value of 150.

### Best Dimensionality Reduction Methods Selected

The optimal configuration was identified as t-SNE, utilizing 3 components, `learning_rate=auto`, and `perplexity=150`, to minimize KL divergence (1.95). Due to runtime constraints, we will utilize the best-performing t-SNE configuration with `n_components=2` to mitigate excessive computation time.

Additionally, PCA with `200 components` and `solver=full` was found to be effective, maximizing the explained variance ratio (97%) and minimizing reconstruction error (147.5).

![t-SNE](plots/part_1/tsne_digits.png) ![PCA](plots/figure7.png)

*Figure: Inverse Point Images*

t-SNE demonstrated superior separation and evaluation, making it suitable for clustering (part 3), while PCA achieved better results in classification (part 4). These chosen configurations will be utilized in subsequent sections for their specific purposes.

### Clustering the Raw Data

#### Clustering Algorithms Overview

We applied two clustering algorithms, namely K-Means and Gaussian Mixture Model (GMM), with a fixed number of clusters set to 10 for both methods, since there are 10 unique digits (0-9) in the MNIST datasets.

- **K-Means:** K-Means is a partitioning algorithm that divides the dataset into K clusters based on similarity. It minimizes the sum of squared distances from each point to the centroid of its assigned cluster. The algorithm aims to create clusters that have low within-cluster variance (same cluster) and high between-cluster variance (different clusters). The default implementation of K-Means uses the k-means++ initialization method, which initializes centroids in a way that speeds up convergence. It selects initial centroids with a probability proportional to the squared distance from the point to the nearest existing centroid, promoting more effective convergence and better final clustering quality.
  
- **Gaussian Mixture Model (GMM):** GMM is a probabilistic model that assumes that the data is generated by a mixture of several Gaussian distributions. It assigns probabilities to each point belonging to different clusters and allows for soft assignments.

##### Evaluating Clustering Results

As mentioned in class, there are several known metrics to evaluate clustering algorithm's performance. The homogeneity score measures how well each cluster contains only members of a single class, the completeness score evaluates if all members of a given class are assigned to the same cluster, the V-Measure is a balanced metric combining homogeneity and completeness, and the silhouette score quantifies the degree of separation between clusters.

The homogeneity score, completeness score, and V-Measure range from 0 to 1, where higher values indicate better performance. For the silhouette score, a higher value suggests better-defined clusters, typically ranging from -1 to 1. In this case, we aim for higher values in all scores.

<div align="center">

| Metric              | K-Means | GMM    |
|---------------------|---------|--------|
| Homogeneity Score   | 0.4943  | 0.2776 |
| Completeness Score  | 0.5015  | 0.3375 |
| V-Measure           | 0.4979  | 0.3047 |
| Silhouette Score    | 0.0593  | -0.0054 |

*Table 3: Clustering Results*

</div>

##### Clustering Algorithms Visualization After Dimensionality Reduction

Both K-Means and GMM are unsupervised learning methods, meaning they do not require labeled data for training. In our analysis, we used the default implementations and applied these clustering algorithms to the entire training dataset.

The visualizations above are conducted on the low-dimensional training data, after the model trained ('fit') on the high-dimensional data. It's evident that the best results are achieved with t-SNE dimensionality reduction.

![Visualization of Clustering on Low-Dimensional Data](plots/figure8.png) 

##### Clustering the Data After Dimensionality Reduction

We performed an experiment involving training the model ('fit' and 'predict') on the reduced data following the application of the optimal t-SNE (perplexity=150, n_components=2) and PCA (n_components=200, solver=full). The evaluation results, detailed in the table below, demonstrated substantial improvement for the data post t-SNE as for the high-dimensional data. This enhancement may be attributed to the heightened capability of these dimensionality reduction techniques in capturing pertinent patterns and reducing noise, thereby enhancing overall model performance while preventing from the model to overfit like on high-dimensional training data.

<div align="center">

| Metric              | K-Means | GMM    |
|---------------------|---------|--------|
| Homogeneity Score   | 0.4936  | 0.4497 |
| Completeness Score  | 0.5009  | 0.4672 |
| V-Measure           | 0.4972  | 0.4583 |
| Silhouette Score    | 0.0640  | 0.0069 |

*Table 4: PCA Clustering Results*

| Metric              | K-Means | GMM    |
|---------------------|---------|--------|
| Homogeneity Score   | 0.8336  | 0.8328 |
| Completeness Score  | 0.8356  | 0.8556 |
| V-Measure           | 0.8346  | 0.8446 |
| Silhouette Score    | 0.5027  | 0.5050 |

*Table 5: t-SNE Clustering Results*

</div>

![Visualization of Clustering After Training on Low-Dimensional Data](plots/figure9.png) 

#### Hyper-parameters Tuning on Clustering Methods

We conducted multiple tests with different hyper-parameters based on the chosen clustering method. The evaluation was performed on the entire dataset without dimensionality reduction to avoid losing important information present in the complete feature set. Despite the longer running time (approximately 1 hour for GMM hyper-parameters tuning) and sometimes suboptimal evaluation results, this approach ensured a comprehensive exploration of the hyper-parameter space considering the richness of information in the original feature space.

- **K-Means:** The hyper-parameters of K-means clustering, specifically the initialization algorithm and the training algorithm, significantly influence the quality of obtained clusters. In our experiments, we explored variations in `init` param such as "Random" and "K-means++" and `algorithm` like "Lloyd" and "Elkan." These choices impact the performance metrics as shown in table below.

<div align="center">

| Initialization  | Algorithm | Homogeneity | Completeness | V-Measure | Silhouette |
|-----------------|-----------|-------------|--------------|-----------|------------|
| Random          | Lloyd     | 0.49433     | 0.50157      | 0.49792   | 0.05934    |
| **K-means++**   | **Lloyd** | **0.49529** | **0.50155**  | **0.49791** | **0.05935**|
| Random          | Elkan     | 0.49432     | 0.50157      | 0.49792   | 0.05934    |
| K-means++       | Elkan     | 0.49429     | 0.50154      | 0.49895   | 0.05934    |

*Table 6: KMeans Hyperparameters Results*

</div>

**GMM:**
Gaussian Mixture Models (GMM) exhibit sensitivity to hyperparameters, particularly the `init_params` and `tol`. In our analysis, we considered different initialization strategies, such as "K-means++" and "Random from Data," combined with varied tolerance values (0.001, 0.1, 1e-06) which are the threshold for considering the algorithm as having converged during the optimization process. These choices impact the performance metrics as shown in the table below.

<div align="center">

| Initialization       | Tolerance | Homogeneity | Completeness | V-Measure | Silhouette |
|----------------------|-----------|-------------|--------------|-----------|------------|
| K-means++            | 0.001     | 0.26902     | 0.35726      | 0.30693   | -0.00193   |
| Random from Data     | 0.001     | 0.26217     | 0.31956      | 0.28803   | -0.00192   |
| K-means++            | 0.1       | 0.26912     | 0.35740      | 0.30704   | -0.00193   |
| **Random from Data** | **0.1**   | **0.26330** | **0.32088**  | **0.28925**| **-0.00162**|
| K-means++            | 1e-06     | 0.26902     | 0.35726      | 0.30693   | -0.00193   |
| Random from Data     | 1e-06     | 0.26202     | 0.31943      | 0.28790   | -0.00193   |

*Table 7: GMM Hyperparameters Results*

</div>

We choose the best hyper-parameters combinations by their silhouette score. The best combination in each table is highlighted in bold.
Please note that there is a difference of approximately 6 decimal places between the results. The variations are minimal, and we have presented only 5 decimal places in the tables due to space constraints.

### Best model Selected

As depicted in Tables 1 and 2, the optimal clustering method is K-means with K-means++ initialization using the Lloyd algorithm. This approach not only yielded superior scores in terms of clustering quality and better visualization but also demonstrated significantly faster runtime compared to GMM models. 

GMM might be less suitable for capturing the complex structures and variations in MNIST digits compared to the simplicity and cluster-centric approach of K-Means. The increased parameter complexity in GMM could lead to overfitting and misalignment with the dataset's true underlying structure, contributing to its suboptimal performance.

### Classification

We used two classifiers for this phase, each excelling in different aspects (with transformed data or with raw data).

- **k-NN (k-Nearest Neighbors):** A supervised classification algorithm that determines the class of a data point based on the majority class among its k-nearest neighbors, utilizing distance measurements.

- **Random Forest Classifier:** A supervised ensemble learning algorithm constructing multiple decision trees during training, with each tree trained on a random subset of the data. The final prediction is made by considering the most frequently predicted classes across the individual trees, reducing overfitting and enhancing accuracy.

### Raw data

We conducted classification on the entire dataset using two classifiers: Random Forest Classifier with 200 estimators and k-nearest neighbors (k-NN) with 5 neighbors. Both algorithms achieved approximately 97% accuracy on the data. It is logical since k-NN is adept at capturing local patterns and relationships in high-dimensional spaces, while Random Forest demonstrates robustness to noise and overfitting in such settings due to its ensemble learning approach. Subsequently, we proceeded with hyperparameter testing for KNN due to its more efficient running time on the high-dimensional training data. Additionally, we experimented with standardizing the data, as mentioned in the first section, but observed no improvement in accuracy, so we decided not to include this preprocessing step. Results are visible within the notebook.

<div align="center">

| Digit | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| 0     | 0.98      | 0.99   | 0.99     |
| 1     | 0.98      | 0.98   | 0.98     |
| 2     | 0.95      | 0.97   | 0.96     |
| 3     | 0.96      | 0.95   | 0.96     |
| 4     | 0.96      | 0.97   | 0.97     |
| 5     | 0.97      | 0.96   | 0.97     |
| 6     | 0.98      | 0.98   | 0.98     |
| 7     | 0.97      | 0.97   | 0.97     |
| 8     | 0.96      | 0.95   | 0.96     |
| 9     | 0.96      | 0.95   | 0.95     |
|-------|-----------|--------|----------|
| Accuracy |         |        | 0.97     |
| Macro avg | 0.97    | 0.97   | 0.97     |
| Weighted avg | 0.97 | 0.97   | 0.97     |

*Table 8: Classification Performance over RF Classifier*

| Digit | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| 0     | 0.98      | 0.99   | 0.99     |
| 1     | 0.96      | 0.99   | 0.98     |
| 2     | 0.98      | 0.96   | 0.97     |
| 3     | 0.97      | 0.97   | 0.97     |
| 4     | 0.97      | 0.97   | 0.97     |
| 5     | 0.97      | 0.97   | 0.97     |
| 6     | 0.98      | 0.99   | 0.98     |
| 7     | 0.96      | 0.97   | 0.97     |
| 8     | 0.99      | 0.93   | 0.96     |
| 9     | 0.95      | 0.96   | 0.96     |
|-------|-----------|--------|----------|
| Accuracy |         |        | 0.97     |
| Macro avg | 0.97    | 0.97   | 0.97     |
| Weighted avg | 0.97 | 0.97   | 0.97     |

*Table 9: Classification Performance over k-NN Classifier*

</div>


