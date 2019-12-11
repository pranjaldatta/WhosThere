import cv2
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN, extract_face
import numpy as np  
from torchvision import transforms
import os
import time

class detect_webcam:
    """
    A module to detect faces from a live video feed.

    This class : 
    -> reads a video
    -> extracts faces and exposes them
    -> detects faces and paints bounding boxes in the live feed

    Parameters:
    extract : {False} Extracts faces from the feed and returns them frame by frame as tensors
    save_faces : {False} Saves the extracted faces
    saveIn : {None} Location to save faces in
    record_for : {None} Record a video for a given amount of seconds
    """

    def __init__(self, extract = False, save_faces = False, save_video = False, saveIn = None, record_for = None):
        self.extract = extract
        self.save = save_faces
        self.saveIn = saveIn
        self.record_for = record_for
        self.save_video = save_video
        if self.save == True : 
            self.tsfms = transforms.ToPILImage()
        else:
            self.tsfms = None    
        if self.save_video == True :
            self.frames_tracked = []
            

    def detect_live(self):
        
        mtcnn = MTCNN()
        faces = {}
        frameCount = 0

        vid = cv2.VideoCapture(0)

        if self.record_for is not None : 
            start_time = time.time()

        while vid.isOpened():

            if self.record_for is not None :
                curr_time = time.time() - start_time
                if curr_time > self.record_for :
                    break                 
        
            _, frame = vid.read()
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            frameCount = frameCount + 1

            boxes, probs = mtcnn.detect(frame)

            frame_draw = frame.copy()
            draw = ImageDraw.Draw(frame_draw)
            if boxes is not None:

                faces["frame_{}".format(frameCount)] = []

                for box, p in zip(boxes, probs) : 
                    if p > 0.70 :
                        draw.rectangle(box.tolist(), outline = (255, 0, 0), width = 1)
                    if self.extract == True :
                        face = extract_face(frame, box.tolist())
                        faces["frame_{}".format(frameCount)].append(face)
                        if self.save == True :
                            img = self.tsfms(face)

                            if self.saveIn is None :
                                raise ValueError

                            else :
                                img.save(os.path.join(self.saveIn, "frame_{}.jpg".format(len(faces))))

                cv2.imshow("Tracking window", cv2.cvtColor(np.array(frame_draw), cv2.COLOR_RGB2BGR))
                if self.save_video == True : 
                    self.frames_tracked.append(frame_draw)                
                if cv2.waitKey(1) == ord("a") :
                    break
                

        
        vid.release()
        
        if self.save_video == True:
            print(len(self.frames_tracked))
            self.saveVideo(self.saveIn, self.frames_tracked, "trackedVid")

        if self.save == True :
            return len(faces.keys()), faces
        else :
            return None, None         

    def saveVideo(self, saveIn, frames, title) :
        """
        An utility function that writes anda saves the givens sequence of frames as video
        """
        dim = frames[0].size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(os.path.join(self.saveIn, title + ".mp4"), fourcc, 25.0, dim)
        for frame in frames :
            video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
        video.release()    







        
        
        
