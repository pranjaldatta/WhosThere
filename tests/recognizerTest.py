from PIL import Image
import os
from WhosThere.lib.faces_detect.detect_frames import detect_frames
import cv2
import glob
from facenet_pytorch import InceptionResnetV1
import torch
import pandas as pd
import matplotlib.pyplot as plt
from WhosThere.lib.recognition import recognizer


knowns = []
for class_id in os.listdir("/home/pranjal/Projects/WhosThere/tests/UntrainedPositives"):
    knowns.append({
        "name": class_id,
        "items" : os.listdir(os.path.join("/home/pranjal/Projects/WhosThere/tests/UntrainedPositives", class_id))
    })
unknowns = os.listdir("/home/pranjal/Projects/WhosThere/tests/UntrainedUnknown")

recog = recognizer.Recognizer(embedLoc="/home/pranjal/Projects/WhosThere/embeddings",threshold=.7, verbose=True)
embeddings = recog.onboard("/home/pranjal/Projects/WhosThere/tests/demo")
print("confidence : ", recog.confidence)

known_score = 0
unknown_score = 0
for i in range(len(knowns)):
    for j in range(len(knowns[i]['items'])):
        img = cv2.imread(os.path.join("/home/pranjal/Projects/WhosThere/tests/UntrainedPositives/"+knowns[i]['name'], knowns[i]['items'][j]))
        pred = recog.verify(lookIn=None, img = img)
        print("for {}, img name : {}, pred : {}".format(knowns[i]['name'], knowns[i]['items'][j], pred[0]['Prediction']))
        if pred[0]['Prediction'] == knowns[i]['name']:
            known_score += 1
for i in range(len(unknowns)):
    img = cv2.imread(os.path.join("/home/pranjal/Projects/WhosThere/tests/UntrainedUnknown", unknowns[i]))
    pred = recog.verify(lookIn=None, img = img)
    print("For {}, prediction: {}".format(unknowns[i], pred[0]['Prediction']))
    if pred[0]['Prediction'] == 'Unknown' :
        unknown_score += 1
    

