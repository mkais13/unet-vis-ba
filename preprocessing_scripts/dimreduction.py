
from keras.preprocessing.image import load_img 
from keras.applications.vgg16 import preprocess_input 
from keras.applications.vgg16 import VGG16 
from keras.applications.resnet_v2 import ResNet101V2
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
import tensorflow as tf
from keras.applications.imagenet_utils import decode_predictions
import keras.backend as K 
import gc

from keras_segmentation.pretrained import pspnet_50_ADE_20K, pspnet_101_cityscapes, pspnet_101_voc12

import os
import argparse
from matplotlib.pyplot import plot
import numpy as np
import pandas as pd
import skdim
import umap
import umap.plot
import plotly.express as px
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import dask.dataframe as dd
import dask.array as da



filepath= "C:/Users/momok/Desktop/Bachelorarbeit/dev/vis/assets/images/resultsrun3"
groundtruthpath = "C:/Users/momok/Desktop/Bachelorarbeit/test-labels/test-labels-0-256"
jsonpath = "C:/Users/momok/Desktop/Bachelorarbeit/test-labels/similarity_plots/2d pspnet_50_ADE_20K(-1)"
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
def extract_features(file, model, img_list):
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
    img_list.append(rgb_img)
    #features = model.predict(rgb_img, use_multiprocessing=True)
    #result = tf.reshape(features[0],( -1,))
    #return result
    return img_list

def reduce_dimensionality(features, dimensions,method = "umap"):

    plotdatalist = []


    if dimensions == 3 and method == "umap":
        umap_3d = umap.UMAP(n_components=3, random_state=42, low_memory=True)
        umap_proj_3d = umap_3d.fit_transform(features)
        plotdatalist = umap_proj_3d.tolist()
    elif method == "umap":
        umap_2d = umap.UMAP(random_state=42, low_memory=True)
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
    elif method =="svd":
        svd = TruncatedSVD()
        svd_proj_2d = svd.fit_transform(features)
        plotdatalist = svd_proj_2d.tolist()

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


#************************************************************************************************************************************************************************

parser = argparse.ArgumentParser(description='set paramaters for feature extraction and dimensionality reduction')
parser.add_argument('-id' , metavar='id', nargs='?', default=23, const=23, help='picture id')
parser.add_argument('-d', '--dimensions', metavar='dimensions', type=int, nargs='?', default=2, const=2, help='data gets reduced to d dimensions. 3d is only possible in combination with method umap')
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

model = Model(inputs = model.inputs, outputs = model.layers[-2].output)
#model = Model(inputs = model.inputs, outputs = model.get_layer("conv5_4").output)
#model = Model(inputs = model.inputs, outputs = model.get_layer("conv6").output)

feature_vectors = []

identifiers = get_identifiers()
counter = 0
#extract features for every run

img_list = []


for id in identifiers:
    #vector = extract_features(os.path.join(filepath, id, picture_id +"_predict.png"),model)
    img = load_img(os.path.join(filepath, id, picture_id +"_predict.png"), target_size=(473,473), interpolation="nearest",color_mode="grayscale")
    print("image loaded")
    img = np.array(img)
    reshaped_img = img.reshape(1,473,473)  
    print("image reshaped")
    rgb_img = np.repeat(reshaped_img[..., np.newaxis], 3, -1)
    print("rgb_img.shape:", rgb_img[0].shape)
    img_list.append(rgb_img[0])
    print("preprocessing for {} done".format(counter))
    counter += 1


img = load_img(os.path.join(groundtruthpath,"test-labels-"+ picture_id + ".png"), target_size=(473,473), interpolation="nearest",color_mode="grayscale")
print("image loaded")
img = np.array(img)
reshaped_img = img.reshape(1,473,473)  
print("image reshaped")
rgb_img = np.repeat(reshaped_img[..., np.newaxis], 3, -1)
img_list.append(rgb_img[0])
print("preprocessing for truth done")

batch_array = []

batch_index = -1
for i in range(len(img_list)):
    if i % 1 == 0:
        batch_index += 1
        batch_array.append([]) 
    batch_array[batch_index].append(img_list[i])

    
#for i in range(len(batch_array)):
#    print("batch_array_shape:", np.shape(batch_array[i]))
#print("batch_array_shape", np.array(batch_array).shape)
#get features for ground truth
#truth_vector = extract_features(os.path.join(groundtruthpath,"test-labels-"+ picture_id + ".png"),model)
#img_list = extract_features(os.path.join(groundtruthpath,"test-labels-"+ picture_id + ".png"),model, img_list)
#feature_vectors.append(truth_vector)

#print("img_list:",img_list)
#img_list_np = np.array(img_list)
#print("img_list_shape:", img_list_np.shape)
print("starting prediction")
#prediction = np.empty()

#for i in range(len(batch_array)):
#    print("starting prediction for batch ", i)
#    img_list_np = np.array(batch_array[i])
#    print("img_list_shape:", img_list_np.shape)
#    print("converting to tensor...")
#    input_tensor = tf.convert_to_tensor(img_list_np)
#    print("predicting...")
#    output_tensor = model.predict(input_tensor)
#    print("output_shape", output_tensor.shape)
#    print("concatenating np array...")
#    if i == 0:
#        prediction = np.reshape(output_tensor, (output_tensor.shape[0], -1))
#        print("current_prediction_shape:", prediction.shape)
#    else:
#        prediction = np.concatenate((prediction, np.reshape(output_tensor, (output_tensor.shape[0], -1))))
#        print("current_prediction_shape:", prediction.shape)
#    K.clear_session()
#    gc.collect()

print("starting prediction for batch ", i)
img_list_np = np.array(batch_array[0])
print("img_list_shape:", img_list_np.shape)
print("converting to tensor...")
input_tensor = tf.convert_to_tensor(img_list_np)
print("predicting...")
output_tensor = model.predict(input_tensor)
print("output_shape", output_tensor.shape)
prediction = np.reshape(output_tensor, (output_tensor.shape[0], -1))
print("current_prediction_shape:", prediction.shape)



#print("converting to tensor...")
#input_tensor = tf.convert_to_tensor(img_list_np)
#print("predicting...")
#output_tensor = model(input_tensor)
#print("making np array...")
#prediction = output_tensor.numpy()

#prediction = model.predict_on_batch(img_list_np)


#saving vectors with dask------------------------------------------------------
#print("predictionshape:", prediction.shape)
#feature_vectors_dask = da.from_array(prediction)
#predictionpath = "C:/Users/momok/Desktop/Bachelorarbeit/dev/vis/assets/data/predictiondata/{}.hdf5".format(picture_id)
#feature_vectors_dask.to_hdf5(predictionpath, "/feature_vectors_dask")
#-----------------------------------------------------------------------------

#calculating fischer------------------------------------------------------
#print("calculating fisher")
#fisher = skdim.id.FisherS().fit_transform(X=prediction)
#print("fisher.dim: ", fisher.dimension_)
#print("fisher.n_single: ", fisher.n_single)
#-----------------------------------------------------------------------

#calculating danco------------------------------------------------------
#print("calculating danco")
#danco = skdim.id.DANCo().fit_transform(prediction)
#print("danco", danco)
#print("dancodim", danco.dimension_)
#-----------------------------------------------------------------------


#calculating kNN------------------------------------------------------
#print("calculating KNN")
#knn = skdim.id.KNN().fit_transform(prediction)
#print("knn", knn)
#-----------------------------------------------------------------------

#calculating MiND_ML------------------------------------------------------
#print("calculating MiND_ML")
#mindml = skdim.id.MiND_ML(k=40, ver="MLi").fit_transform(prediction)
#print("MiND_ML", mindml)
#-----------------------------------------------------------------------



#feature_vectors_np = np.array(prediction)
#print("feature_shape:", feature_vectors_np.shape)

#sklearnpca = PCA(n_components=0.05).fit(feature_vectors_np)
#lpcadim = skdim.id.lPCA(ver="ratio").fit(feature_vectors_np).dimension_

#print("lpcadim", lpcadim)
#print("sklearnpca: ",sklearnpca)
#print("sklearnpca dim:", len(sklearnpca.singular_values_))
#print("saving data")
#feature_vectors_np_to_ser = pd.DataFrame(feature_vectors_np)
#predictionpath = "C:/Users/momok/Desktop/Bachelorarbeit/dev/vis/assets/data/psp_prediction_data"

#feature_vectors_np_to_ser.to_csv(path_or_buf = os.path.join(predictionpath, picture_id + ".csv.gzip"), compression="gzip")
#feature_vectors_np_to_ser = pd.DataFrame(feature_vectors_np)






print("starting to reduce dimensions")
data_to_plot = reduce_dimensionality(prediction, dimensions, method = method)

#convert list into Panda-DataFrame

if dimensions == 3:
    plotdatadf = pd.DataFrame(data_to_plot, columns=["x","y","z","run_id","batchsize","lossfunction","optimizer","topologyfactor","kernelinitializer"])
    #fig = px.scatter_3d(plotdatadf,x="x",y="y",z="z",color="lossfunction", size="batchsize", symbol="optimizer")
else:
    plotdatadf = pd.DataFrame(data_to_plot, columns=["x","y","run_id","batchsize","lossfunction","optimizer","topologyfactor","kernelinitializer"])
    print("plotdatadf:", plotdatadf)
    fig = px.scatter(plotdatadf,x="x",y="y",color="lossfunction", size="batchsize", symbol="optimizer")

fig.show()


#plotdatadf.to_json(orient ="index", path_or_buf= os.path.join("C:/Users/momok/Desktop/Bachelorarbeit/dev/vis/assets/data/predictiondata", picture_id + ".json"))
#plotdatadf.to_json(orient ="index", path_or_buf= os.path.join(jsonpath, picture_id + ".json"))
#plotdatadf.to_csv(path_or_buf= os.path.join(csvpath,"{}d".format(dimensions), picture_id + ".csv"))
