

import pandas as pd
import umap
import os
import plotly.express as px


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
  

complete_df = pd.DataFrame()
for i in range(30):
    current_3d_data = pd.read_json("C:/Users/momok/Desktop/Bachelorarbeit/dev/vis/assets/data/embeddata/3d pspnet_50_ADE_20K(-1)/{}.json".format(i), orient="index")
    reduced_feature_vectors = current_3d_data.loc[:,["x","y","z"]]

    reduced_feature_vectors.rename(columns={"x": i*3+1, "y":i*3+2, "z": i*3+3}, inplace=True)
    complete_df = pd.concat([complete_df, reduced_feature_vectors], axis=1)

print(complete_df)

dimension = 3

data_to_plot = reduce_dimensionality(reduced_feature_vectors, dimension)
 

print("data_to_plot: ", data_to_plot)

if dimension == 3:
    plotdatadf = pd.DataFrame(data_to_plot, columns=["x","y","z","run_id","batchsize","lossfunction","optimizer","topologyfactor","kernelinitializer"])
    fig = px.scatter_3d(plotdatadf,x="x",y="y",z="z",color="lossfunction", size="batchsize", symbol="optimizer")
else:
    plotdatadf = pd.DataFrame(data_to_plot, columns=["x","y","run_id","batchsize","lossfunction","optimizer","topologyfactor","kernelinitializer"])
    fig = px.scatter(plotdatadf,x="x",y="y",color="lossfunction", size="batchsize", symbol="optimizer")

fig.show()

jsonpath = "assets/data/embeddata/{}d pspnet_50_ADE_20K(-1)".format(dimension)

plotdatadf.to_json(orient ="index", path_or_buf= os.path.join(jsonpath, "aggregate.json"))