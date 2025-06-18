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

First, download M_pipeline.py and any desired scripts. Ensure they are all in the same directoy (folder).
~~~
import sys
import pandas as pd
from M_Pipeline import Mpipe

# Set up project root and data paths
root = "path/to/root/"

# example file containing text data
descriptions_cleaned = "[root]/data/text_data.csv"

# example file containing SBERT embeddings
embed_data_file = "[root]/data/SBERT/embeddings_L6_v2.csv"

# dir to save into
output_dir = "[root]/data/SBERT/.../"

# set random seed for reproducibility
RANDOM = 2025
~~~

# Initialize the pipeline
In the same folder as M_pipeline, create a new script and initialise the pipeline object
~~~
# Ensure M_Pipeline.py in same directory as this script
pipe = Mpipe(root=root, data_file=data_file, output=output_dir)
~~~

# Define SBERT parameters and extract word embeddings
Chose an SBERT model (https://huggingface.co/sentence-transformers) and add its name to sbert_ags as below
~~~
sbert_args = {"model_name":"all-MiniLM-L6-v2"}
embeddings = pipe.SBERT(pipe.data, sbert_args)

#    in: DataFrame/List with only text data
#    out: DataFrame with shape (n_observations, n_features)
~~~
# Define UMAP parameters and run dimensionality reduction
Familiarise yourself with UMAP input arguments, information linked here
3rd Party (reccomended): https://pair-code.github.io/understanding-umap/
Official: https://umap-learn.readthedocs.io/en/latest/parameters.html#n-neighbors
~~~
umap_args = {"n_components": 2, "n_neighbors": 15, "random_state": RANDOM}
umap_features = pipe.UMAP_reduce(embeddings, umap_args)

#    in: DataFrame of shape (n_observations, n_features)
#    out: DataFrame of shape (n_observations, n_features)
~~~

# Define K-Means parameters and run clustering
Select number of clusters (see Analysis for sillhetee score analysis of KMEANS solutions)
Note: the order of these argument dictionaries (e.g. km_args) does not affect result. 
~~~
km_args = {"n_clusters": 7, "random_state": RANDOM, "n_init": 100}
kmeans_labels = pipe.KMEANS(umap_features, km_args)

# pipe.KMEANS method:
#    in: DataFrame of shape (n_observations, n_features)
#    out: DataFrame of shape (n_observations, 1) - one label for each observation clustered
~~~

# Saving data
All methods in this pipeline return pandas DataFrames (n_observations, n_features) allowing for easy saving at any stage. 
Option 1:
Automatically saves file in chosen output directory
~~~
file_name = "text_data cleaned1 all-MiniLM-L6"
pipe.save(embed, file_name)
~~~
Option 2:
Utilise pandas saving methods direclty
~~~
file_name = "text_data cleaned1 all-MiniLM-L6.csv"
embed.to_csv(os.path.join(pipe.output, file_name))
~~~

# Exploring with Jupyter Notebooks
For more detailed examples and to learn how to apply these statistical methods, please refer to the provided Jupyter Notebooks:

Embed.ipynb: This notebook demonstrates how to generate text embeddings from your raw data.

Analysis.ipynb: Learn how to perform dimensionality reduction and clustering on your embeddings to uncover patterns and insights.

Classification.ipynb: This notebook guides you through building and evaluating a predictive model based on the processed features.

Default Arguments.ipynb: Here, you can explore the default parameters for the various methods.

# Roadmap
[x] Embedding Module: Implemented SBERT for text embeddings.

[x] Dimensionality Reduction Module: Implemented UMAP.

[x] Clustering Module: Implemented K-Means and Fuzzy C-Means.

[x] Analysis & Classification: Created notebooks for pipeline execution, visualization, and building predictive models.

[ ] Implement additional embedding models (e.g., OpenAI, Llama).

[ ] Implement other dimensionality reduction techniques (e.g., PCA, t-SNE).

Users are encouraged to contribute by implementing additional analysis steps, for embedding or other techniques. As long as the output data maintains its shape of (observations, features), a wide array of other analytical techniques can be readily integrated. Your contributions are welcome and help to expand the capabilities of this tool.

License Distributed under the MIT LICENSE. See LICENSE.txt for more information.
