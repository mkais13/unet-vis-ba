 
from keras.preprocessing.image import load_img 
from keras.applications.vgg16 import preprocess_input 
from keras.applications.vgg16 import VGG16 
from keras.models import Model
import os
import argparse
from matplotlib.pyplot import plot
import numpy as np
import pandas as pd
import umap
import umap.plot
import plotly.express as px
from sklearn.decomposition import PCA

filepath= "C:/Users/momok/Desktop/Bachelorarbeit/dev/results/results"
groundtruthpath = "C:/Users/momok/Desktop/Bachelorarbeit/dev/UNet/unet/data/membrane/train/label"
outputpath = "C:/Users/momok/Desktop/Bachelorarbeit/dev/results/similarityplot"

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
    "focal_tversky"
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


def extract_features(file, model):
    img = load_img(file, target_size=(224,224))
    img = np.array(img) 
    reshaped_img = img.reshape(1,224,224,3) 
    imgx = preprocess_input(reshaped_img)
    features = model.predict(imgx, use_multiprocessing=False)
    return features[0]

def reduce_dimensionality(features,method ="umap"):

    plotdatalist = []

    if method == "umap":
        umap_2d = umap.UMAP(random_state=42)
        umap_proj_2d = umap_2d.fit_transform(features)
        plotdatalist = umap_proj_2d.tolist()
    elif method =="pca":
        pca = PCA(n_components=2)
        pca_proj_2d = pca.fit_transform(features)
        plotdatalist = pca_proj_2d.tolist()
    else: 
        raise ValueError("parameter 'method' has to be either 'umap' or 'pca'")



    for i in range(len(identifiers)):   
        split_ids = identifiers[i].split("-")
        plotdatalist[i].insert(2,identifiers[i])
        plotdatalist[i].insert(3,int(split_ids[0].replace("bs","")))
        plotdatalist[i].insert(4,split_ids[1].replace("lf",""))
        plotdatalist[i].insert(5,split_ids[2].replace("opt",""))
        plotdatalist[i].insert(6,float(split_ids[3].replace("tf","")))
        plotdatalist[i].insert(7,split_ids[4].replace("ki",""))

    truth_index = len(identifiers)
    plotdatalist[truth_index].insert(2, "truth")
    plotdatalist[truth_index].insert(3, int(1))
    plotdatalist[truth_index].insert(4, "truth")
    plotdatalist[truth_index].insert(5, "truth")
    plotdatalist[truth_index].insert(6, float(0))
    plotdatalist[truth_index].insert(7, "truth")

    return plotdatalist

parser = argparse.ArgumentParser(description='define which picture should be analyzed')
parser.add_argument('-id' , metavar='id', nargs='?', default=18, const=18, help='id')
args = parser.parse_args()

picture_id = str(args.id)
print("picture_id: " + picture_id)
model = VGG16(weights="imagenet")
#remove last layers to access featurevectors
model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

feature_vectors = []

#extract features for every run
for id in identifiers:
    vector = extract_features(os.path.join(filepath, id, picture_id +"_predict.png"),model)
    feature_vectors.append(vector)
    #remove all zeros from vector
    
#get features for ground truth
truth_vector = extract_features(os.path.join(groundtruthpath, picture_id + ".png"),model)
feature_vectors.append(truth_vector)
feature_vectors_np = np.array(feature_vectors)

data_to_plot = reduce_dimensionality(feature_vectors_np, "umap")


plotdatadf = pd.DataFrame(data_to_plot, columns=["x","y","run_id","batchsize","lossfunction","optimizer","topologyfactor","kernelinitializer"])
fig = px.scatter(plotdatadf,x="x",y="y",color="lossfunction", size="batchsize", symbol="optimizer", text="kernelinitializer")
fig.show()

plotdatadf.to_json(orient ="index", path_or_buf= os.path.join(outputpath, picture_id + ".json"))
