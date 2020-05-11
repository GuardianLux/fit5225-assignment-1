from flask import Flask, request, jsonify, Response
import numpy as np
import cv2
import os
from io import BytesIO
from PIL import Image
from collections import defaultdict

confthres = 0.5
nmsthres = 0.1

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

class Result:
    def __init__(self,label,confidence):
        self.label = label
        self.confidence = confidence

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
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configpath, weightspath)
    return net

@app.route('/', methods=['POST'])
def main():
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

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confthres:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                classIDs.append(classID)
                confidences.append(float(confidence))
                

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres,
                            nmsthres)

    # ensure at least one detection exists
    objects = {}
    arr = []
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            objects[int(i)] = {}
            objects[int(i)]["Label"] = labels[classIDs[i]]
            objects[int(i)]["Accuracy"] = float("{:.2f}".format(confidences[i]*100))
            arr.append(objects[int(i)])

    return jsonify({"Objects":arr})

if __name__== "__main__":
  app.run()