import cv2
from PIL import Image, ImageDraw
import os
from facenet_pytorch import MTCNN, extract_face
import numpy as np
import matplotlib.pyplot as plt


class detect_video:
    """
    A module to detect faces in a given video

    This class :
    -> reads a video
    -> runs MTCNN on each frame 
        --extracts faces and exposes them for furthur usage
        --draws bounding boxes around each detected face in each frame
    -> stiches the operated-upon-frames into a video

    Parameters:
    saveIn : {None} Dir to save the video in 
    lookIn : Dir to read the video from


    """

    def __init__(self, lookIn, saveIn = None):
        self.lookIn = lookIn
        self.saveIn = saveIn
        if self.saveIn is not None:
            self.writeMode = True
        else:
            self.writeMode = False
    def detect(self):
    
        vid = cv2.VideoCapture(self.lookIn)
        frameCount = int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

        mtcnn = MTCNN()

        bboxes_and_probs = []
        count = frameCount
        while vid.isOpened():
            
            #if count <  frameCount:
                #break

            _,  frame = vid.read()
            print("%d to go.." %(count))
            count -= 1
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            boxes, prob = mtcnn.detect(frame)
            
            frame_draw = frame.copy()
            draw = ImageDraw.Draw(frame_draw)
            if boxes is None :
                #print("Skipping Frame")
                if self.writeMode == True:
                    detected_frames.append(frame_draw)                 
                cv2.imshow("Frame", cv2.cvtColor(np.asarray(frame_draw), cv2.COLOR_BGR2RGB)) 
                if cv2.waitKey(2) & 0xFF == ord('y'):
                    break
                continue
            for box, p in zip(boxes,prob):
                
                if p > 0.80:   
                                   
                    #print("Not skipping!")      
                    draw.rectangle(box.tolist(), outline= (255, 0, 0), width= 1)
                    bboxes_and_probs.append({"bbox":box, "prob":p})

                if self.writeMode == True:
                    detected_frames.append(frame_draw)
            
            cv2.imshow("Frame", cv2.cvtColor(np.asarray(frame_draw), cv2.COLOR_BGR2RGB)) 
            if cv2.waitKey(1) & 0xFF == ord('y'):
                break
                       
            
        
        print("releasing capture")
        vid.release()
        
        if self.writeMode == True :
            dim = detected_frames[0].size
            print(dim , int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")    
            video_tracked = cv2.VideoWriter(self.saveIn, fourcc, 25.0, dim)
            for frame in detected_frames:
                video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
            video_tracked.release()
        return bboxes_and_probs




        