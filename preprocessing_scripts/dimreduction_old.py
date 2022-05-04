 
import gc
from sre_constants import CATEGORY_UNI_NOT_LINEBREAK
from keras.preprocessing.image import load_img 
from keras.applications.vgg16 import preprocess_input 
from keras.applications.vgg16 import VGG16 
from keras.applications.resnet_v2 import ResNet101V2
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
import keras.backend as K
import tensorflow as tf
from keras.applications.imagenet_utils import decode_predictions

from keras_segmentation.pretrained import pspnet_50_ADE_20K, pspnet_101_cityscapes, pspnet_101_voc12

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




filepath= "C:/Users/momok/Desktop/Bachelorarbeit/dev/vis/assets/images/resultsrun3"
groundtruthpath = "C:/Users/momok/Desktop/Bachelorarbeit/test-labels/test-labels-0-256"
embedpath = "C:/Users/momok/Desktop/Bachelorarbeit/test-labels/similarity_plots/2d pspnet_50_ADE_20K(-1)"
csvpath = "C:/Users/momok/Desktop/Bachelorarbeit/dev/results/csvdata"
predictionpath = "C:/Users/momok/Desktop/Bachelorarbeit/dev/vis/assets/data/psp_prediction_data"


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
    #img = load_img(file, target_size=(224,224), interpolation="bicubic",color_mode="grayscale")
    img = load_img(file, target_size=(473,473), interpolation="nearest",color_mode="grayscale")
    #img = load_img(file, target_size=(713,713), interpolation="bicubic",color_mode="grayscale")
    print("image loaded")
    img = np.array(img)
    #check if image is only one color (faulty run)
    #if np.all(img == img[0]):
        #print(file)
    #reshaped_img = img.reshape(1,224,224) 
    reshaped_img = img.reshape(1,473,473) 
    #reshaped_img = img.reshape(1,713,713) 
    print("image reshaped")
    #add 2 fake color channels to fit the model requirements
    rgb_img = np.repeat(reshaped_img[..., np.newaxis], 3, -1)
    #imgx = preprocess_input(rgb_img)
    input_tensor = tf.convert_to_tensor(rgb_img)
    features = model(input_tensor)
    result = tf.reshape(features[0],( -1,))
    K.clear_session()
    gc.collect()
    return result

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
parser.add_argument('-id' , metavar='id', nargs='?', default=23, const=23, help='picture id')
parser.add_argument('-d', '--dimensions', metavar='dimensions', type=int, nargs='?', default=3, const=3, help='data gets reduced to d dimensions. 3d is only possible in combination with method umap')
parser.add_argument('-m', '--method' , metavar='method', type=str, nargs='?', default="umap", const="umap", help='method for dimensionality reduction. Options: umap, tsne, pca')
args = parser.parse_args()

picture_id = str(args.id)
print("calculating picture {}".format(args.id))
dimensions = args.dimensions
method = args.method  

#model = pspnet_101_voc12()
#model = pspnet_101_cityscapes()
model = pspnet_50_ADE_20K()
#model = ResNet101V2(include_top=False)
#model = VGG16(weights="imagenet")
#model = InceptionV3(include_top=False)
#remove last layers to access featurevectors

model = Model(inputs = model.inputs, outputs = model.layers[-1].output)
#model = Model(inputs = model.inputs, outputs = model.get_layer("conv5_4").output)
#model = Model(inputs = model.inputs, outputs = model.get_layer("conv6").output)

feature_vectors = []

identifiers = get_identifiers()
counter = 0
#extract features for every run
for id in identifiers:
    vector = extract_features(os.path.join(filepath, id, picture_id +"_predict.png"),model)
    feature_vectors.append(vector)
    print("prediction for {} done".format(counter))
    counter += 1

#get features for ground truth
truth_vector = extract_features(os.path.join(groundtruthpath,"test-labels-"+ picture_id + ".png"),model)
feature_vectors.append(truth_vector)




print("making np array")
feature_vectors_np = np.array(feature_vectors)
print("feature_vectors_np",feature_vectors_np.shape)


#serialize the predicted data
#feature_vectors_np_to_ser = pd.DataFrame(feature_vectors_np)
#feature_vectors_np_to_ser.columns = feature_vectors_np_to_ser.columns.astype(str)
#print("feature_vectors_np_to_ser",feature_vectors_np_to_ser)
#print("saving feature vectors")
#feature_vectors_np_to_ser.to_csv(path_or_buf = os.path.join(predictionpath, picture_id + ".csv.gzip"), compression="gzip")
#feature_vectors_np_to_ser.to_feather(path = os.path.join(predictionpath, picture_id + ".feather"))


print("starting to reduce dimensions")
data_to_plot = reduce_dimensionality(feature_vectors_np, dimensions, method = method)
 

#convert list into Panda-DataFrame

if dimensions == 3:
    plotdatadf = pd.DataFrame(data_to_plot, columns=["x","y","z","run_id","batchsize","lossfunction","optimizer","topologyfactor","kernelinitializer"])
    #fig = px.scatter_3d(plotdatadf,x="x",y="y",z="z",color="lossfunction", size="batchsize", symbol="optimizer")
else:
    plotdatadf = pd.DataFrame(data_to_plot, columns=["x","y","run_id","batchsize","lossfunction","optimizer","topologyfactor","kernelinitializer"])
    #print("plotdatadf:", plotdatadf)
    #fig = px.scatter(plotdatadf,x="x",y="y",color="lossfunction", size="batchsize", symbol="optimizer")

#fig.show()

#TODO hier speichern und dann nochmal 3d auf 2d mappen
plotdatadf.to_json(orient ="index", path_or_buf= os.path.join("C:/Users/momok/Desktop/Bachelorarbeit/dev/vis/assets/data/predictiondata", picture_id + "_old_method.json"))
#plotdatadf.to_json(orient ="index", path_or_buf= os.path.join(embedpath, picture_id + ".json"))
#plotdatadf.to_csv(path_or_buf= os.path.join(csvpath,"{}d".format(dimensions), picture_id + ".csv"))