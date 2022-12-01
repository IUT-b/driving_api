import io

import numpy as np
import requests
from flask import jsonify
from keras.applications.vgg16 import VGG16, preprocess_input
from PIL import Image
from sklearn.cluster import KMeans


def classification(request):
    # フレームが保存されているURL
    url = request.json["url"]
    # フレームの名称のリスト
    images = request.json["images"]
    assert len(images) > 0

    # ニューラルネットワークのモデル取得
    model = VGG16(weights="imagenet", include_top=False)
    
    # モデルを使用してフレームの特徴抽出
    X = []
    for image in images:
        x = Image.open(io.BytesIO(requests.get(url + image).content))
        x = x.resize((224, 224))
        x = np.expand_dims(x, axis=0)  # add a dimention of samples
        x = preprocess_input(
            x
        )  # RGB 2 BGR and zero-centering by mean pixel based on the position of channels
        feat = model.predict(x)  # Get image features
        feat = feat.flatten()  # Convert 3-dimentional matrix to (1, n) array
        X.append(feat)
    X = np.array(X)

    # フレームの分類数
    Nc = request.json["Nc"]
    
    # k means法によりフレームを分類
    assert Nc > 0 and Nc < 9 and Nc <= len(images)
    kmeans = KMeans(n_clusters=Nc, random_state=0).fit(X)
    labels = kmeans.labels_

    # フレームで分類が変わるポイントを抽出
    scene = [0]
    label = labels[0]
    for i in range(len(labels)):
        if labels[i] != label:
            scene.append(i)
            label = labels[i]

    # jsonifyの引数はdic
    return jsonify({"scene": scene}), 201
