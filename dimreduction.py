 
from keras.preprocessing.image import load_img 
from keras.applications.vgg16 import preprocess_input 
from keras.applications.vgg16 import VGG16 
from keras.applications.resnet_v2 import ResNet101V2
from keras.models import Model
import tensorflow as tf

import os
import argparse
from matplotlib.pyplot import plot
import numpy as np
import pandas as pd
import umap
import umap.plot
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import matplotlib.image as mpimg




filepath= "C:/Users/momok/Desktop/Bachelorarbeit/dev/results/results2"
groundtruthpath = "C:/Users/momok/Desktop/Bachelorarbeit/dev/UNet/unet/data/membrane/train/label"
jsonpath = "C:/Users/momok/Desktop/Bachelorarbeit/dev/results/jsondata"
csvpath = "C:/Users/momok/Desktop/Bachelorarbeit/dev/results/csvdata"


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


#preprocesses the images and returns the prediction the given model made
def extract_features(file, model):
    img = load_img(file, target_size=(224,224), interpolation="bicubic",color_mode="grayscale")
    img = np.array(img)
    #check if image is only one color (faulty run)
    #if np.all(img == img[0]):
        #print(file)
    reshaped_img = img.reshape(1,224,224) 
    #add 2 fake color channels to fit the model requirements
    rgb_img = np.repeat(reshaped_img[..., np.newaxis], 3, -1)
    imgx = preprocess_input(rgb_img)
    features = model.predict(imgx, use_multiprocessing=False)
    result = tf.reshape(features[0],( -1,))
    return result

def reduce_dimensionality(features, dimensions,method = "umap"):

    plotdatalist = []


    if dimensions == 3 and method == "umap":
        umap_3d = umap.UMAP(n_components=3, random_state=42)
        umap_proj_3d = umap_3d.fit_transform(features)
        plotdatalist = umap_proj_3d.tolist()
    elif method == "umap":
        umap_2d = umap.UMAP()
        umap_proj_2d = umap_2d.fit_transform(features)
        plotdatalist = umap_proj_2d.tolist()
    elif method =="pca":
        pca = PCA(n_components=2)
        pca_proj_2d = pca.fit_transform(features)
        plotdatalist = pca_proj_2d.tolist()
    elif method=="tsne":
        tsne = TSNE()
        tsne_proj_2d = tsne.fit_transform(features)
        plotdatalist = tsne_proj_2d.tolist()

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

parser = argparse.ArgumentParser(description='set paramaters for feature extraction and dimensionality reduction')
parser.add_argument('-id' , metavar='id', nargs='?', default=25, const=25, help='picture id')
parser.add_argument('-d', '--dimensions', metavar='dimensions', type=int, nargs='?', default=3, const=3, help='data gets reduced to d dimensions. 3d is only possible in combination with method umap')
parser.add_argument('-m', '--method' , metavar='method', type=str, nargs='?', default="umap", const="umap", help='method for dimensionality reduction. Options: umap, tsne, pca')
args = parser.parse_args()

picture_id = str(args.id)
dimensions = args.dimensions
method = args.method


model = VGG16(weights="imagenet")
#remove last layers to access featurevectors

model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

feature_vectors = []

identifiers = get_identifiers()
#extract features for every run
for id in identifiers:
    vector = extract_features(os.path.join(filepath, id, picture_id +"_predict.png"),model)
    feature_vectors.append(vector)


#get features for ground truth
truth_vector = extract_features(os.path.join(groundtruthpath, picture_id + ".png"),model)
feature_vectors.append(truth_vector)
feature_vectors_np = np.array(feature_vectors)

data_to_plot = reduce_dimensionality(feature_vectors_np, dimensions, method = method)
 
print(data_to_plot[len(identifiers)])

#convert list into Panda-DataFrame

if dimensions == 3:
    plotdatadf = pd.DataFrame(data_to_plot, columns=["x","y","z","run_id","batchsize","lossfunction","optimizer","topologyfactor","kernelinitializer"])
    #fig = px.scatter_3d(plotdatadf,x="x",y="y",z="z",color="lossfunction", size="batchsize", symbol="optimizer", text="run_id")
else:
    plotdatadf = pd.DataFrame(data_to_plot, columns=["x","y","run_id","batchsize","lossfunction","optimizer","topologyfactor","kernelinitializer"])
    #fig = px.scatter(plotdatadf,x="x",y="y",color="lossfunction", size="batchsize", symbol="optimizer", text="run_id")

#fig.show()

plotdatadf.to_json(orient ="index", path_or_buf= os.path.join(jsonpath,"{}d".format(dimensions), picture_id + ".json"))
#plotdatadf.to_csv(path_or_buf= os.path.join(csvpath,"{}d".format(dimensions), picture_id + ".csv"))
