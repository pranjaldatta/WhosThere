import cv2
import numpy as np  
from PIL import Image
from facenet_pytorch import MTCNN, extract_face
from torchvision import transforms
import os

class detect_frames:
    """
    A module to detect faces in a given frame

    This class:
    -> Accepts a frame
    -> detects faces
    -> returns cropped images of all faces in tensor format

    Parameters : 
    frame : Frame on which detection is to be performed

    Return :

        
    """
    def __init__(self, threshold, saveIn = None):
        self.threshold = threshold
        if saveIn is not None :
            self.saveIn = saveIn
        else:
            self.saveIn = None
        self.toPIL = transforms.ToPILImage()    
 
    def detectAndCrop(self, frame) :

        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        mtcnn_module = MTCNN(keep_all=True)
        boxes, prob = mtcnn_module.detect(frame)
        faces = []
        count = 0
        for box,prob in zip(boxes,prob) :
            if prob > self.threshold:
                face = extract_face(frame, box)
                print("Face #{} detected with probability : {}".format(count + 1 , prob )) 
                faces.append({"bbox":box, "prob":prob})
                count = count + 1
                if self.saveIn is not None:
                    img = self.toPIL(face).convert('RGB')
                    img.save(os.path.join(self.saveIn, "face_{}.jpg".format(count)))
        
        return faces            

    def detect(self, frame) :

        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        mtcnn_module = MTCNN(keep_all= True)
        face_tensors, probs = mtcnn_module(frame, return_prob = True)

        tensors_and_probs = []
        
        try:
            if probs[0] is None:
                print("raising")
                raise Exception("FaceNotFoundError: No Face Found.")
        except Exception as ext:
            return [ext]

        count = 0     
        for tensor, prob in zip(face_tensors, probs):
            count += 1
            print("Face #{} detected with probability : {}".format(count, prob))
            tensors_and_probs.append((count, tensor, prob))
            if self.saveIn is not None : 
                self.toPIL(tensor).convert('RGB').save(os.path.join(self.saveIn, "face_{}.jpg".format(count)))

        return tensors_and_probs        
                
                

