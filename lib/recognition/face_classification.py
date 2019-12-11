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
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import neighbors
import time
import sys



class Face_Classifier:
    """
    A module that has techniques that classfies a given test face into either a known category or
    an unknown category. Classifying an un-regitered image as unknown is the primary
    issue. We explore various methods to get the desired result.
    """

    def __init__(self, known_loc, unknown_loc, verbose = True):
        self.embeddings = pd.read_pickle(
            "/home/pranjal/Projects/WhosThere/embeddings/embeddings.pkl")
        device = 'cpu'
        self.resnet = InceptionResnetV1(
            pretrained='casia-webface', classify=False).eval().to(device)
        self.face_extracter = detect_frames(threshold=.7)
        self.known_loc = known_loc
        self.unknown_loc = unknown_loc

        self.knowns = []
        self.unknowns = []
        for folder in os.listdir(self.known_loc):
            class_id = folder
            items = []
            for item in os.listdir(os.path.join(self.known_loc, folder)):
                items.append(item)
            self.knowns.append({"class_id": class_id, "items": items})
            items = []
        for ext in ("*jpg", "*jpeg", "*.png"):
            self.unknowns.extend(glob.glob(os.path.join(self.unknown_loc, ext)))

        self.verbose = verbose
        
    def _init_data(self):
        X = []
        Y = []
        for i in range(len(self.embeddings["embeddings"])):
            X.append(self.embeddings["embeddings"].iloc[i])
            Y.append(self.embeddings["name"].iloc[i])
        X = np.asarray(X).reshape(len(self.embeddings), 512)
        Y = np.asarray(Y).reshape(len(self.embeddings), 1)       

        return X, Y, self.embeddings["embeddings"].values

    def _average_distance_of_vector_elements(self, simi_list):
        distances = 0

        start_time = time.time()
        simi_mat = np.asarray(simi_list)
        for i in range(len(simi_mat)):
            distances += abs(simi_mat[i+1:len(simi_mat)] - simi_mat[i]).sum()
        end_time = time.time()
        print("avg distance : ", distances)
        print(end_time - start_time)


    def cosine_similarity_func(self) :

        if self.verbose == True:
            sys.stdout = open(os.devnull, 'w')
        
        X, Y, embeddings = self._init_data()
        
        known_min_max_similarities = []
        unknown_min_max_similarities = []
        for i in range(len(self.knowns)):
            
            class_id = self.knowns[i]["class_id"]
            imgs = self.knowns[i]["items"]
            for img in imgs:
                
                print("-"*50)
                print("Testing for {} ...".format(img))
                test_img = cv2.imread(os.path.join(os.path.join(self.known_loc, class_id), img))
                test_img_tensors = self.face_extracter.detect(test_img)
                test_img_embeddings = self.resnet(torch.unsqueeze(test_img_tensors[0][1], 0))                

                
                max_sim = 0
                max_name = None
                similarites = []
                for j in range(len(self.embeddings)):
                    known_trained_name = self.embeddings["name"].iloc[j]
                    known_trained_embed = self.embeddings["embeddings"].iloc[j]
                    cos_similarity = F.cosine_similarity(test_img_embeddings, torch.from_numpy(known_trained_embed)).item()
                    print("Comparing {} against {} we get : {} ".format(img, known_trained_name, cos_similarity))
                    similarites.append(cos_similarity)
                    if cos_similarity > max_sim :
                        max_sim = cos_similarity
                        max_name = known_trained_name
                similarites.sort()
                print("Difference : ", similarites[len(self.embeddings) - 1] - similarites[0])
                max_name,_ = max_name.split(".")  
                known_min_max_similarities.append({
                    "class_id" : class_id,
                    "test_img" : img,
                    "predict" : max_name,
                    "diff" : similarites[len(self.embeddings) - 1] - similarites[0]
                })                          
                print("Prediction : ", max_name)
                print("-"*50)

        
        for unknown in (self.unknowns):
            
            name = os.path.basename(unknown)            
            print("Testing {} ...".format(name))

            img_test = cv2.imread(unknown)
            unknown_untrained_tensors = self.face_extracter.detect(img_test)
            unknown_untrained_embed = self.resnet(torch.unsqueeze(unknown_untrained_tensors[0][1], 0))

            similarites = []
            max_sim = 0
            max_name = ""
            for i in range(len(self.embeddings)):
                known_trained_name = self.embeddings["name"].iloc[i]
                known_trained_embed = self.embeddings["embeddings"].iloc[i]
                cos_similarity = F.cosine_similarity(
                    unknown_untrained_embed, torch.from_numpy(known_trained_embed)).item()
                print("Comaparing {} against {} we get similarity of : {}".format(
                    name, known_trained_name, cos_similarity))
                similarites.append(cos_similarity)
                if cos_similarity > max_sim:
                    max_sim = cos_similarity
                    max_name = known_trained_name

            similarites.sort()
            unknown_min_max_similarities.append({
                    "class_id" : "unknown",
                    "test_img" : os.path.basename(unknown),
                    "predict" : max_name,
                    "diff" : similarites[len(self.embeddings) - 1] - similarites[0]
            }) 
            print("Predict : ", max_name)
                
        set_accuracy_score = .90
        
        unknown_diffs = []
        for i in range(len(unknown_min_max_similarities)):
            unknown_diffs.append(unknown_min_max_similarities[i]['diff'])
        unknown_diffs.sort()


        max_unknown_score = 0
        best_threshold = 0     
        total_score = 0   
        for idx in range(len(self.knowns)):
            total_score += len(self.knowns[idx]['items']) 
        print("Total score: ",total_score)        
        for i in range(len(unknown_diffs)):
            curr_threshold = unknown_diffs[i]
            known_score = 0
            unknown_score = 0
            for j in range(len(known_min_max_similarities)):
                if known_min_max_similarities[j]['diff'] > curr_threshold and known_min_max_similarities[j]['class_id'] == known_min_max_similarities[j]['predict']:
                    known_score += 1
            for j in range(len(unknown_min_max_similarities)):
                if unknown_min_max_similarities[j]['diff'] <= curr_threshold:
                    unknown_score += 1        
            print("For Threshold = {}, known score : {}, unknown score : {}".format(curr_threshold, known_score, unknown_score))         
            if known_score/(total_score) >= set_accuracy_score : #find better way
                max_unknown_score = unknown_score
                best_threshold = curr_threshold
                continue
            else:
                break
        
        if self.verbose == True:
            sys.stdout = sys.__stdout__
        return best_threshold            

        



            


