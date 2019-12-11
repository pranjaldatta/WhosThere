from PIL import Image
import os
from WhosThere.lib.faces_detect.detect_frames import detect_frames
import cv2
import glob
from facenet_pytorch import InceptionResnetV1
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import pickle
import sys
from WhosThere.lib.recognition.face_classification import Face_Classifier

class Recognizer:

    '''
    A class module that performs the the primary recognition opearations.
    It has two primary functions :
    -> user onboarding : generates embeddings and stores them for new users
    -> user verification :  generates embeddings of the faces in the frame and compares against the present ones to check who all are present in the frame

    Functions :
    -> Onboard : generate and store embeddings of a face
    -> Verify : Veriy wheather a given face exists in the database or not


   '''

    def __init__(self, embedLoc, threshold, verbose = False, device='cpu'):

        self.embedLoc = embedLoc
        self.threshold = threshold
        self.device = device
        self.resnet = InceptionResnetV1(
            pretrained='casia-webface', classify=False).eval().to(device)
        self.face_extracter = detect_frames(threshold=.7)
        self.saveClusterIn = "/home/pranjal/Projects/WhosThere/embeddings/KNN_model.pkl"
        self.std_outpipe = sys.stdout
        self.verbose = verbose

        self.embeddings = None
        if(os.path.exists(os.path.join(self.embedLoc, "embeddings.pkl"))) :
            self.embeddings = pd.read_pickle(os.path.join(self.embedLoc, "embeddings.pkl"))

        self.confidence = 0
        self.clf = Face_Classifier(known_loc = "/home/pranjal/Projects/WhosThere/tests/UntrainedPositives", unknown_loc = "/home/pranjal/Projects/WhosThere/tests/UntrainedUnknown")                            

    def onboard(self, savedIn):
        """
        A function that generates embeddings for each of the true users.
        Input should be a single face-only picture of the true user with the name of the user as the picture title.

        Params :
        -> savedIn : Location where the image is stored in

         Prints the probability with which each face is detected and its name if verbose is enabled
         Saves the embeddings in Pickle format


        Returns :
        -> embeddings in pandas dataframe format
         
        """

        if self.verbose == True :
            sys.stdout = open(os.devnull, "w")

        frames = []        
        for ext in ("*.jpg", "*.jpeg"):
            frames.extend(glob.glob(os.path.join(savedIn, ext)))
        df = pd.DataFrame(columns=['name', 'embeddings'])
        X = []
        Y = []
        for frame in frames:
            name = os.path.basename(frame)
            img = Image.open(frame)
            face_tensors = self.face_extracter.detect(img)
            embeddings = self.resnet(torch.unsqueeze(face_tensors[0][1], 0))
            print("Embeddings generated for {} is of shape : {}".format(
                name, embeddings.shape))
            embeddings = embeddings.detach().numpy().astype('float32')
            X.append(embeddings)
            Y.append(name)
            df = df.append({"name": name, "embeddings": embeddings},
                           sort=False, ignore_index=True)
        self.embeddings = df
        df.to_pickle(os.path.join(self.embedLoc, "embeddings.pkl"))

        if self.verbose == True:
            sys.stdout = self.std_outpipe    

        if self.confidence == 0 :
            print("Execing cosing similarity")
            self.confidence = self.clf.cosine_similarity_func()

        return df   
    
    def onboard_more(self, img, name):

        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        df = pd.read_pickle(os.path.join(self.embedLoc, "embeddings.pkl"))
        face_tensors = self.face_extracter(img)
        face_embeddings = self.resnet(torch.unsqueeze(face_tensors[0][1], 0))
        print("Embeddings generated for {} is of shape : {}".format(
                name, embeddings.shape))
        face_embeddings = face_embeddings.detach().numpy().astype('float32')
        df = df.append({
            "name" : name,
            "embeddings" : face_embeddings
            }, sort = False, ignore_index = True)  

        df.to_pickle(os.path.join(self.embedLoc, "embeddings.pkl")) 
        self.confidence = self.clf.cosine_similarity_func()     

        return df

    def verify(self, lookIn, img=None, embeddings = None):
        """
        A function that either accepts input as an image or from folders and compares with stored embeddings

        Params : 
        -> lookIn : directory to look for inputs
        -> img : Image in PIL or array format
        
        Returns : 
        -> list of dicts :[{"Prediction" : <Pred>}]

        """

        if self.verbose == True :
            sys.stdout = open(os.devnull, "w")    


        try:
            if lookIn == None and img is None:
                raise Exception(
                    "lookIn and img cannot be None and False simultaneosly")
            if lookIn is not None and img is not None:
                raise Exception("lookIn and img cannot be supplied simultaneously")          
            else:
                pass
        except Exception as inst:
            if self.verbose == True:
                sys.stdout = self.std_outpipe
            print(inst)
            raise
        
        if self.embeddings is None:
            if embeddings is None:
                if(os.path.exists(os.path.join(self.embedLoc, "embeddings.pkl"))) :
                    self.embeddings = pd.read_pickle(os.path.join(self.embedLoc, "embeddings.pkl"))            
            else:
                self.embeddings = embeddings
        
        if img is not None:
            if isinstance(img, np.ndarray) == True:
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            frame = cv2.imread(lookIn)
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        frame_tensors = self.face_extracter.detect(frame)
        
        try:
            if isinstance(frame_tensors[0], Exception):
                raise frame_tensors[0]
        except Exception as ext:
            if self.verbose == True:
                sys.stdout = self.std_outpipe            
            return [ext]    
        
        preds = []
        for idx in range(len(frame_tensors)):             
            
            frame_embeddings = self.resnet(torch.unsqueeze(frame_tensors[idx][1], 0))

            max_simi_name = ""
            max_simi = 0
            simi_list = []
            for i in range(len(self.embeddings)):

                stored_embedding = torch.from_numpy(self.embeddings["embeddings"].iloc[i])
                similarity = F.cosine_similarity(stored_embedding, frame_embeddings).item()
                simi_list.append(similarity)
                if similarity > max_simi:
                    max_simi_name = self.embeddings["name"].iloc[i]
                    max_simi = similarity


            simi_list.sort()
            print("diff : ", simi_list[len(self.embeddings) - 1] - simi_list[0])
            print("confidence : ", self.confidence)
            if(simi_list[len(self.embeddings) - 1] - simi_list[0] <= self.confidence):
                max_simi_name = "Unknown"
            else :     
                max_simi_name, _ = max_simi_name.split(".")

            preds.append({"Prediction ": max_simi_name})
            max_simi_name = ""
            max_simi = 0   
            simi_list = []   

        if self.verbose == True:
            sys.stdout = self.std_outpipe

        return preds   



