# MEAMs pipeline.py

#Outline 
"""

0. Data structure 
task : define root folder and folder locs for data, output, figures, and logs

1. embeddings
in : df or string sep of memory data 
choice: which embedding model
    method for each reduction (to deal with imports)
out: embeddings data (df)

2. dim reduction 
in : embeddings data    shape = (n_observations, n_features)
choice: which method of dim reduction?
    method for each tequnique 
out : reduced values (df)   shape = (n_observations, n_features)

3. Clustering
in: matrix/df   shape = (n_observations, n_features)
choice : clustering method
    method for each 
out: cluster labels     shape = (observation, label)

"""
import warnings, os
import numpy as np
import pandas as pd
import umap
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import skfuzzy as fuzz

class base:
    def __init__(self, safe_run):
        self.compatible_ftypes = {".npy": np.load,
                                  ".csv": pd.read_csv}
        if type(safe_run) != bool:
            raise TypeError(f"safe_run argument must be bool type, {type(safe_run)} given")

        self.safe_run=safe_run
        return 
    
    def set_attributes(self, attrs):
        if type(attrs) != dict:
            raise TypeError(f"expected dictionary, got {type(attrs)}")

        for key, value in attrs.items():
            if key == "step":
                continue
            #if hasattr(self, key):  # Check if the attribute exists on the object
            setattr(self, key, value)  # Set the attribute value
            #print(key,value)
    
    def scale(self, data):
        return StandardScaler().fit_transform(data)
    def descr_data(self, object):
        print(f"-Type: {type(object)}\n-Shape:{object.shape}\n- -Obervations: {object.shape[0]}\n- -Features/measures: {object.shape[1]}\n")
    def missing_attr_list(self, length):
        return [x for x in range(1, length)]
    def get_file_ext(self, path):
        return path[path.rfind("."):]        
    def check_attr(self, obj, attrs):
        #obj = obj
        for key in attrs.keys():
            if not hasattr(obj, key):
                raise ValueError(f"{obj} has no attribute {key}")
            
    def load(self, path):
        if os.path.exists(path):
            ext = self.get_file_ext(path)
            if ext in self.compatible_ftypes.keys():
                data = self.compatible_ftypes[ext](path)
                return data
                
            warnings.WarningMessage(f"Incompatible file extension. Expected {self.compatible_ftypes.keys()}, got {ext}")
        warnings.WarningMessage(f"file path not found : {path}")
        cont = input("Quit session? (Y)")
        if cont in ["Y", "y"]:
            quit()

class M_embed(base):
    def __init__(self):
        return

    def SBERT(self, text_data, params=None):
        self.stepadd = "-SBERT"
        """ Sentence Bert Sentence Transformer encoding 
        IN: dataframe containing text data (will vertically concatenate columns if cols > 1)
        OUT: data frame of shape (n_observations, n_features)
        info: https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html

        """

        text_data = text_data.dropna() # ensure no missing values (sbert rejects)

        #model = SentenceTransformer()
        #model.__dict__.update(self.step_params)

        text_in = [] # initialise list for input text
        for col in text_data.columns:
            text_in += text_data[col].tolist() # concatenate each column in input df onto the text input list
        for (i,x) in enumerate(text_in): # check that all values in input list are string
            if type(x) != str:
                raise TypeError(f"dataframe passed to SBERT function contains non-string values ({text_in[i]} : {type(x)})")
        del text_data # clear memory 

        if params is not None:
            self.step_params = params

        model = SentenceTransformer(model_name_or_path=self.step_params["model_name"])# initialist SBERT model
        # if self.safe_run: # make sure model has attributes assigned to it (from the step attributes list)
        #     self.check_attr(model, self.step_params)
        # model.__dict__.update(self.step_params) # update model perameters 

        embeddings = model.encode(text_in, convert_to_numpy=True) # transform text to embeddings 
        # return df containing embeddings (n_observations, n_features)
        
        return pd.DataFrame(embeddings,
                            columns=[f"Feature_{x}" for x in range(1,embeddings.shape[1]+1)]) 
    
    def OPENAI(self):
        self.stepadd = "-OPENAI"
        return

class M_dim_reduce(base):
    def __init__(self):
        return
    
    def UMAP_reduce(self, data, params=None):
        self.stepadd = "-UMAP"
        """ UMAP dimentionality reduction 
            IN: dataframe containing only values to be reduced (index=False)
            OUT: umap dimensional values (n_observations, n_features)
            info: https://umap-learn.readthedocs.io/en/latest/parameters.html
        """ 
        # model = umap.UMAP(n_components=self.n_components, n_neighbors=self.n_neighbors, n_jobs=self.n_jobs,
        #               metric=self.metric, n_epochs=self.n_epochs, 
        #               min_dist=self.min_dist, negative_sample_rate=self.negative_sample_rate, init=self.init)

        if params is not None:
            self.step_params = params
        try:
            model = umap.UMAP() # initialise umap model
        except AttributeError:
            model = umap.umap_.UMAP()
        model.__dict__.update(self.step_params) # update perameters (attributes) from default
        if self.safe_run: # make sure model has attributes assigned to it (from the step attributes list)
            self.check_attr(model, self.step_params)

        out = model.fit_transform(self.scale(data)) # fit model to data

        #return pandas df containing reduced data
        return pd.DataFrame(out, dtype=np.float64,
                            columns=[f"Feature_{x}" for x in range(1, out.shape[1]+1)])
    
    def PCA_reduce(self):
        self.stepadd = "-PCA"
        return
        
    def TSNE_reduce(self):
        self.stepadd = "-TSNE"
        return

class M_cluster(base):
    def __init__(self):
        return
    
    def KMEANS(self, data, params=None):
        self.stepadd = "-KMEANS"
        """ KMEANS clustering algorithm
            IN: pandas dataframe containing data to be clustered (index=False)
            OUT:cluster labels (n_observations, labels)
            info: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        """
        if params is not None:
            self.step_params = params

        model = KMeans() # initialise model
        if self.safe_run: # make sure model has attributes assigned to it (from the step attributes list)
            self.check_attr(model, self.step_params)
        
        model.__dict__.update(self.step_params) # update model perameters 
        model.fit(self.scale(data)) # fit model to data

        # return model labels as df (n_observations, labels) note: only has 1 column
        return pd.DataFrame(model.labels_, columns=["labels"])
    
    def FUZZYCMEANS(self, data, params=None, score=False, centroids=False):
        self.stepadd = "-FUZZYCMEANS"
        """ FUZZY C Means (skfuzzy)
        IN: 2D array (n_observations, n_features)
        OUT: cluster membership (proportions) shape = (memberships, n_observations), score (float)
        INFO: https://scikit-fuzzy.github.io/scikit-fuzzy/auto_examples/plot_cmeans.html#example-plot-cmeans-py
        """
        data = data.to_numpy().T # transpose matrix (skfuzzy expexts n_feat, n_obvs)

        if params is not None:
            self.step_params = params

        if self.step_params["random_state"] == None:
            cntr, memb, _, _, _, _, fpc = fuzz.cluster.cmeans(self.scale(data), 
                                                          c=self.step_params["n_clusters"],
                                                          m=2, error=self.step_params["error"], 
                                                          maxiter=self.step_params["maxiter"], 
                                                          init=None)
        else:
            np.random.seed(self.step_params["random_state"])
            centroids_init = np.random.rand(self.step_params["n_clusters"], data.shape[1])  # Random initialization of centroids

            cntr, memb, _, _, _, _, fpc = fuzz.cluster.cmeans(self.scale(data), 
                                                            c=self.step_params["n_clusters"],
                                                            m=2, error=self.step_params["error"], 
                                                            maxiter=self.step_params["maxiter"], 
                                                            init=centroids_init)
        out = [memb.T]
        if score:
            out.append(fpc)
        if centroids:
            out.append(cntr)
        return out

class Mpipe(M_embed, M_dim_reduce, M_cluster, base):
    def __init__(self, root=None, data_file=None, output=None, figures="figures", 
                 custom_fname=None, step_attributes=[],safe_run=True, verbose=True):
        base.__init__(self, safe_run=safe_run)
        M_embed.__init__(self)
        M_dim_reduce.__init__(self)
        M_cluster.__init__(self)
        exit = False
        # check and set list of attribute dictionaries for each function in the pipeline
        if type(step_attributes) != list:
            raise TypeError(f"expected list with 1+ dicts, got {type(step_attributes)}")
            if type(step_attributes[0]) != dict:
                raise TypeError(f"expected list with 1+ dicts, got {type(step_attributes)} and {type(step_attributes[0])}")

        # for (i, attrs) in enumerate(step_attributes):
        #     attrs["step"].set_attributes(self, step_attributes[i])

        self.step_attributes = step_attributes
        self.verbose = verbose
        self.steps_taken = ""

        self.last_out = None
        if custom_fname is not None:
            self.custom_fname = custom_fname
        else:
            self.custom_fname = ""

        false_vals = []
        for val in (root, data_file, output, figures):
            if type(val) != str:
                false_vals.append(val)
        if len(false_vals) != 0:
            print("False values:")
            for x in false_vals:
                print(x, " :", type(x))
            raise TypeError(f"init vals given incorrect type (Must be string)")
        del false_vals
        if type(verbose) != bool:
            raise TypeError(f"verbose argument must be bool type, {type(verbose)} given")
        
        if root is None:
            raise ValueError("root folder must be specified")
        self.root = root
        if data_file is None:
            raise ValueError("data folder must be specified")
        self.data_file = os.path.join(root, data_file)
        if output is None:
            raise ValueError("data folder must be specified")            
        
        self.output = os.path.join(self.root, output)
        self.output = os.path.normpath(self.output)
        if figures is None:
            warnings.warn("no figures folder specified, figs will be saved to root folder", UserWarning)
            temp_join = ""
        else:
            temp_join = figures
        self.figures = os.path.join(root, temp_join)

        # Check validity of file structure
        no_paths = []
        for path in (self.root, self.data_file, self.figures):
            if os.path.exists(path):
                pass
            else:
                no_paths.append(path)
        if len(no_paths) != 0:
            print(no_paths)
            exit = True
        if exit:
            raise ValueError(f"file not found error for above paths")
        del no_paths
        
        if not os.path.exists(self.output):
            print(f"\033[31m Output folder does not exist, creating directory \033[0m{self.output}")
            os.mkdir(self.output)

        self.data=self.load(self.data_file)

        if verbose:
            print(f"Mpipe initialised.\nloaded data from: {self.data_file}\n")
            self.descr_data(self.data)
        
        # ext_ind = self.data_file.rfind(".")
        # data_ext = self.data_file[ext_ind:]
        # if data_ext in self.compatible_ftypes.keys():
        #     self.data = self.compatible_ftypes[data_ext](self.data_file)
        #     if self.safe_run:
        #         if type(self.data) != type(pd.DataFrame):
        #             self.data = pd.DataFrame(self.data)
        #         self.data.dropna()
        
        # if subject_ID is None:
        #     self.subj_id = [x for x in range(1, self.data.shape[0])]
        # else:
        #     self.subj_id = pd.read_csv(os.path.join(self.root, subject_ID)).iloc[:,0]

        
    def save(self, obj, fname=None):
        if type(obj) != pd.DataFrame:
            raise TypeError(f"expected pandas df object, got {type(obj)}\nLast output: {self.last_out}")
        if fname is not None:
            self.custom_fname = fname
        fname = f"{self.custom_fname}{self.steps_taken}.csv"
        path = os.path.join(self.output, fname)

        obj.to_csv(path, index=False)
        if self.verbose:
            print(f"DataFrame of shape {obj.shape} saved to {self.root[0]}: ... \\{os.path.basename(self.output)}\\{fname}")
        self.last_out = path
    
    def execute(self, pipe):
        """ Runs pipeline objects and assigns step_attributes at the same index to their functions
            - saves the output of every step taken, with a filename that shows which steps were taken (and which order)
            - 
        """ 
        data_obj = self.data
        for (i, step) in enumerate(pipe):
            self.step_params = self.step_attributes[i] #self.step_attributes[i]["step"].set_attributes(self, {"step_params":self.step_attributes[i]})
            
                #if self.step_params["step"] != pipe[i].__name__:
                    #raise ImportError(f"Expected step name {pipe[i].__name__} for step {i+1}, got {self.step_params['step']}")

            if self.verbose:
                print(f"\n===>\nStarting step: {pipe[i].__name__}\n- -Params: {self.step_params}\n")

            data_obj = step(data_obj) # each function must return a pandas df
            
            if self.verbose:
                print(f"Pipeline step: {self.stepadd[1:]} finished\n<===")
                self.descr_data(data_obj)
            self.steps_taken += self.stepadd
            self.save(data_obj)
        return data_obj


if __name__ == "__main__":
    root = "D:\\Dropbox\\2. Cognitive science\\Music evoked autobiographical memories\\"
    pipe = Mpipe(root, r"data\SBERT\SBERT-L6-v2_embed.csv", "\\data\\SBERT\\")
