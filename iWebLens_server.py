from flask import Flask, request, jsonify, Response
import numpy as np
import cv2
import os
from io import BytesIO
from PIL import Image
from collections import defaultdict

# Thresholds for both confidence in detection and NMS
confthres = 0.5
nmsthres = 0.1

app = Flask(__name__)
# Without this Json output is in the wrong order
app.config["JSON_SORT_KEYS"] = False

class Result:
    def __init__(self,label,confidence):
        self.label = label
        self.confidence = confidence

# Functions to get paths of each config item, using os.getcwd makes the server easy to run without worrying about filepaths
def get_labels(labels_path):
    lpath = os.getcwd() + labels_path
    labels = open(lpath).read().strip().split("\n")
    return labels

def get_weights(weights_path):
    weightsPath = os.getcwd() + weights_path
    return weightsPath

def get_config(config_path):
    configPath = os.getcwd() + config_path
    return configPath

def load_model(configpath,weightspath):
    net = cv2.dnn.readNetFromDarknet(configpath, weightspath)
    return net

@app.route('/api/weblens', methods=['POST'])
def main():
    # Read image being sent from client
    i = request.files["image"].read()
    image = Image.open(BytesIO(i))
    npimg=np.array(image)
    image=npimg.copy()
    labelsPath="/config/coco.names"
    cfgpath="/config/yolov3.cfg"
    wpath="/config/yolov3-tiny.weights"
    labels=get_labels(labelsPath)
    CFG=get_config(cfgpath)
    Weights=get_weights(wpath)
    net=load_model(CFG,Weights)

    (H, W) = image.shape[:2]

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > confthres:
                # Scale detection box back to be relative to image size
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # Find top left corner of box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                classIDs.append(classID)
                confidences.append(float(confidence))
                print(classIDs)
                
    # Use NMS to remove any overlapping boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres,
                            nmsthres)

    objects = {}
    arr = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            objects[int(i)] = {}
            objects[int(i)]["Label"] = labels[classIDs[i]]
            objects[int(i)]["Accuracy"] = float("{:.2f}".format(confidences[i]*100))
            arr.append(objects[int(i)])

    return jsonify({"Objects":arr})

if __name__== "__main__":
  app.run(host='0.0.0.0')