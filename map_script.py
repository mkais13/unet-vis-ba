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



def get_identifiers():

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

    identifiers=[]
    #get all run_identifiers
    for bs in batchsizes:
        for lf in lossfunctions:
            for opt in optimizers:
                for tf in topologyfactors:
                    for ki in kernelinitializers:
                        identifiers.append('bs{0}-lf{1}-opt{2}-tf{3}-ki{4}'.format(bs,lf,opt,tf,ki))
                        
    return identifiers


def reduce_dimensionality(features, dimensions, run_ids,method = "umap"):

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

    identifiers = run_ids

    if dimensions == 3:
        insert_index = 1
    else:
        insert_index = 0

    #insert run information into the plotdatalist
    for i in range(len(identifiers)):   

        if(identifiers[i] == "truth"):
            plotdatalist[i].insert(insert_index + 2, "truth")
            plotdatalist[i].insert(insert_index + 3, int(20))
            plotdatalist[i].insert(insert_index + 4, "truth")
            plotdatalist[i].insert(insert_index + 5, "truth")
            plotdatalist[i].insert(insert_index + 6, float(0))
            plotdatalist[i].insert(insert_index + 7, "truth")
        else:
            split_ids = identifiers[i].split("-")
            plotdatalist[i].insert(insert_index + 2,identifiers[i])
            plotdatalist[i].insert(insert_index + 3,int(split_ids[0].replace("bs","")))
            plotdatalist[i].insert(insert_index + 4,split_ids[1].replace("lf",""))
            plotdatalist[i].insert(insert_index + 5,split_ids[2].replace("opt",""))
            plotdatalist[i].insert(insert_index + 6,float(split_ids[3].replace("tf","")))
            plotdatalist[i].insert(insert_index + 7,split_ids[4].replace("ki",""))

  




    return plotdatalist

def map_data(path_to_3d_data, selected_run_ids):

    #read json file
    data3d = pd.read_json(path_to_3d_data, orient="index")
    #remove every row except the selected runs
    selected_data3d = data3d.loc[data3d["run_id"].isin(selected_run_ids)]
    #remove every column except the numeric data
    reduced_feature_vectors = selected_data3d[["x","y","z"]]

    data_to_plot = reduce_dimensionality(reduced_feature_vectors, 2, selected_run_ids)
 

    plotdatadf = pd.DataFrame(data_to_plot, columns=["x","y","run_id","batchsize","lossfunction","optimizer","topologyfactor","kernelinitializer"])
    fig = px.scatter(plotdatadf,x="x",y="y",color="lossfunction", size="batchsize", symbol="optimizer", custom_data= ["run_id"])
    for i in range(len(fig["data"])):
        legendname = fig["data"][i]["name"]
        print("legendname ", legendname)
        fig["data"][i]["marker"]["color"] = getcolor(legendname.split(",")[0])
        fig["data"][i]["marker"]["symbol"] = getsymbol(legendname.split(",")[1][1:])
    return fig

def getcolor(lf):
    switcher = {
                "msssim": "#00CC96",
                "mean_squared_error": "#636EFA",
                "binary_crossentropy" : "#EF553B",
                "truth" : "#AB63FA",
            }
    return switcher.get(lf)


def getsymbol(lf):
    switcher = {
                "SGD": "diamond",
                "Adam": "square",
                "Adagrad" : "circle",
                "truth" : "x",
            }
    return switcher.get(lf)