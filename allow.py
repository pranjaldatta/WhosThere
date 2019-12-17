from WhosThere.lib.recognition import recognizer
import cv2
import argparse
import sys
import os
import pandas as pd 
import numpy as np 

recog = recognizer.Recognizer("/home/pranjal", threshold=.5, verbose=True)

def onboard():
    img = None
    vid = cv2.VideoCapture(0)
    while True:
        _, img = vid.read()        
        cv2.imshow("Frame", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(5) & 0xFF == ord('y'):
            break
    vid.release()    
    recog.onboard_more(img, "me")  
    cv2.destroyAllWindows()  


def verify():
    embeddings = pd.read_pickle("/home/pranjal/embeddings.pkl")
    img = None
    vid = cv2.VideoCapture(0)
    while True:
        _, frame = vid.read()
        cv2.imshow("Frame", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(5) & 0xFF == ord('y'):
            break
    vid.release()
    cv2.destroyAllWindows()
    preds = recog.verify(lookIn=None, img=frame)   
    if isinstance(preds[0], Exception):      
        return 0
    elif preds[0]["Prediction"] == "me" :
        return 1
    else :
        return 0

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--onboard", help="onboard sudo users",action="store_true")
parser.add_argument("-v", "--verify", help="Verify users",action="store_true")
args = parser.parse_args()

if args.onboard:
        onboard()
if args.verify:
        print(verify())
    
