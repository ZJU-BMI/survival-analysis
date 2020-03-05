import numpy as np
from keras.layers.embeddings import Embedding
import os
from keras.models import Sequential
import tensorflow as tf


# 将原本的1/0 变成embedding编码
def embedding_category():
    model = Sequential()
    model.add(Embedding(2,256,input_length=93))
    half_year_features = np.load("pick_4_features_two_year.npy")
    time = half_year_features[:,:,0].reshape(-1,half_year_features.shape[1],1)
    features = half_year_features[:,:,1:].reshape(-1, half_year_features.shape[2]-1)
    model.compile("rmsprop",'mse')
    output_array = model.predict(features).reshape(-1,half_year_features.shape[1],93,256)
    np.save("4_two_year_embedding_features.npy",output_array)
    np.save("4_two_year_time.npy",time)
    print(output_array)
    print(time.shape)


if __name__ == "__main__":
    embedding_category()
