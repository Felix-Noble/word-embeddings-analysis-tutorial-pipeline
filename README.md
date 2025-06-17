Contact: Felix Noble - felix.noble@live.co.uk | Repositories: https://github.com/Felix-Noble | Project Updates: https://www.linkedin.com/in/felix-noble-6901b117b/
# Textual Data Analysis Pipeline
This project provides an accessible, high-level tool for analyzing textual data, ideal for researchers and students new to Python. It centers around a flexible pipeline (M_Pipeline.py) for analyzing written natural language (NL) through three main steps:

Embedding: Converts text into numerical vectors using models like Sentence-BERT (SBERT).

Dimensionality Reduction: Uses techniques like UMAP to simplify the data.Clustering: Groups data using algorithms such as K-Means and Fuzzy C-Means.

The Jupyter notebooks (Analysis.ipynb and Classification.ipynb) demonstrate how to use the pipeline and build predictive models (e.g., Random Forest).

# Getting Started

PrerequisitesBefore you begin, ensure you have Python installed. It's highly recommended to work within a virtual environment to manage dependencies. You can find a helpful tutorial on creating virtual environments here.InstallationClone the repository:git clone [https://github.com/your_username/your_project.git](https://github.com/your_username/your_project.git)

# Install the required packages:
pip install pandas numpy umap-learn sentence-transformers scikit-learn scikit-fuzzy matplotlib

Ensure your data files are placed in the appropriate data/ subdirectory.

# Usage
The core of this project is the M_Pipeline object, which facilitates a step-by-step text analysis workflow. Below is a basic example of how to use it.

~~~
import sys
import pandas as pd
from M_Pipeline import Mpipe

# Set up project root and data paths
root = "path/to/root/"

data_file = "[root]/data/SBERT/playlists/spotify_descriptions AND memories all-MiniLM-L6.csv"

output_dir = "[root]/data/SBERT/playlists AND memories/"

# Initialize the pipeline with existing embeddings
pipe = Mpipe(root=root, data_file=data_file, output=output_dir)

# Define UMAP parameters and run dimensionality reduction
umap_args = {"n_components": 2, "n_neighbors": 15, "random_state": 2025}
u
map_features = pipe.UMAP_reduce(pipe.data, umap_args)

# Define K-Means parameters and run clustering
km_args = {"n_clusters": 7, "random_state": 2025, "n_init": 100}

kmeans_labels = pipe.KMEANS(umap_features, km_args)
~~~

The results at each step are automatically saved to the output directory.

# Exploring with Jupyter Notebooks
For more detailed examples and to learn how to apply these statistical methods, please refer to the provided Jupyter Notebooks:

Embed.ipynb: 
This notebook demonstrates how to generate text embeddings from your raw data.

Analysis.ipynb: Learn how to perform dimensionality reduction and clustering on your embeddings to uncover patterns and insights.

Classification.ipynb: This notebook guides you through building and evaluating a predictive model based on the processed features.

Default Arguments.ipynb: Here, you can explore the default parameters for the various methods and understand their impact on the results.

# Roadmap
[x] Embedding Module: Implemented SBERT for text embeddings.

[x] Dimensionality Reduction Module: Implemented UMAP.

[x] Clustering Module: Implemented K-Means and Fuzzy C-Means.

[x] Analysis & Classification: Created notebooks for pipeline execution, visualization, and building predictive models.

[ ] Implement additional embedding models (e.g., OpenAI, Llama).

[ ] Implement other dimensionality reduction techniques (e.g., PCA, t-SNE).

Users are encouraged to contribute by implementing additional analysis steps, for embedding or other techniques. As long as the output data maintains its shape of (observations, features), a wide array of other analytical techniques can be readily integrated. Your contributions are welcome and help to expand the capabilities of this tool.

License Distributed under the MIT LICENSE. See LICENSE.txt for more information.
