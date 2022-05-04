import dask.array as da
import dask.dataframe as dd
import numpy as np
import umap
import plotly.express as px
import numpy as np
import os
import h5py
import pandas as pd

os.environ["HDF5_DISABLE_VERSION_CHECK"] = "1"

optimizers = [
    "Adagrad",
    "SGD",
    "Adam"
]

topologyfactors = [
    0.5, 
    1.0, 
    1.5
]


batchsizes = [
    3, 
    10
]

lossfunctions = [
    "mean_squared_error",
    "binary_crossentropy",
    "msssim"
]

kernelinitializers = [
    "he_normal",
    "he_uniform"
]

def get_identifiers():
    identifiers=[]
    #get all run_identifiers
    for bs in batchsizes:
        for lf in lossfunctions:
            for opt in optimizers:
                for tf in topologyfactors:
                    for ki in kernelinitializers:
                        identifiers.append('bs{0}-lf{1}-opt{2}-tf{3}-ki{4}'.format(bs,lf,opt,tf,ki))
    return identifiers


def reduce_dimensionality(features, dimensions,method = "umap"):

    plotdatalist = []


    if dimensions == 3 and method == "umap":
        umap_3d = umap.UMAP(n_components=3, random_state=42)
        umap_proj_3d = umap_3d.fit_transform(features)
        plotdatalist = umap_proj_3d.tolist()
    elif method == "umap":
        umap_2d = umap.UMAP(random_state=42)
        umap_proj_2d = umap_2d.fit_transform(features)
        plotdatalist = umap_proj_2d.tolist()
    
    else: 
        raise ValueError("parameter 'method' is not valid'")

    identifiers = get_identifiers()

    if dimensions == 3:
        insert_index = 1
    else:
        insert_index = 0

    #insert run information into the plotdatalist
    for i in range(len(identifiers)):   
        split_ids = identifiers[i].split("-")
        plotdatalist[i].insert(insert_index + 2,identifiers[i])
        plotdatalist[i].insert(insert_index + 3,int(split_ids[0].replace("bs","")))
        plotdatalist[i].insert(insert_index + 4,split_ids[1].replace("lf",""))
        plotdatalist[i].insert(insert_index + 5,split_ids[2].replace("opt",""))
        plotdatalist[i].insert(insert_index + 6,float(split_ids[3].replace("tf","")))
        plotdatalist[i].insert(insert_index + 7,split_ids[4].replace("ki",""))

    #insert truth info into the plotdatalist
    truth_index = len(identifiers)
    plotdatalist[truth_index].insert(insert_index + 2, "truth")
    plotdatalist[truth_index].insert(insert_index + 3, int(20))
    plotdatalist[truth_index].insert(insert_index + 4, "truth")
    plotdatalist[truth_index].insert(insert_index + 5, "truth")
    plotdatalist[truth_index].insert(insert_index + 6, float(0))
    plotdatalist[truth_index].insert(insert_index + 7, "truth")



    return plotdatalist

#hf = h5py.File('C:/Users/momok/Desktop/Bachelorarbeit/dev/vis/assets/data/predictiondata/23.hdf5', 'r')
#n1 = hf.get('feature_vectors_dask')
#print(np.array(n1))
#hf = dd.read_hdf("C:/Users/momok/Desktop/Bachelorarbeit/dev/vis/assets/data/predictiondata/23.hdf5", "/feature_vectors_dask", chunksize=5)
#print(hf)
#dask_array = hf.to_dask_array()
#print(dask_array)

data3d = pd.read_json("C:/Users/momok/Desktop/Bachelorarbeit/dev/vis/assets/data/predictiondata/23_old_method.json", orient="index")
reduced_feature_vectors = data3d[["x","y","z"]]
print("reduced_feature_vectors", reduced_feature_vectors)

dimensions = 2

data_to_plot = reduce_dimensionality(reduced_feature_vectors, dimensions, method = "umap")
 



if dimensions == 3:
    plotdatadf = pd.DataFrame(data_to_plot, columns=["x","y","z","run_id","batchsize","lossfunction","optimizer","topologyfactor","kernelinitializer"])
    fig = px.scatter_3d(plotdatadf,x="x",y="y",z="z",color="lossfunction", size="batchsize", symbol="optimizer")
else:
    plotdatadf = pd.DataFrame(data_to_plot, columns=["x","y","run_id","batchsize","lossfunction","optimizer","topologyfactor","kernelinitializer"])
    print("plotdatadf:", plotdatadf)
    fig = px.scatter(plotdatadf,x="x",y="y",color="lossfunction", size="batchsize", symbol="optimizer")

fig.show()