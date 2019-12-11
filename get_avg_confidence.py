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


            

    def get_avg_confidence(self):
        """
        The objective was to average out the probs of each classiier but failed spectacularly
        """

        frames = glob.glob(os.path.join(
            "/home/pranjal/Projects/WhosThere/tests/demo", "*jpg"))
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

        x = np.asarray(X).reshape(4, 512)
        y = np.asarray(Y).reshape(4, 1)

        positives = "/home/pranjal/Projects/WhosThere/tests/UntrainedPositives"
        unknowns = "/home/pranjal/Projects/WhosThere/tests/UntrainedUnknown"

        trained_positives = []
        untrainedUnknown = []
        for filetype in ("*.jpg", "*.png", "*.jpeg"):
            trained_positives.extend(
                glob.glob(os.path.join(positives, filetype)))
            untrainedUnknown.extend(
                glob.glob(os.path.join(unknowns, filetype)))

        knn_clf = neighbors.KNeighborsClassifier(
            n_neighbors=2, weights='distance')
        gaussian = GaussianNB()
        linearSVM = SVC(C=1, kernel='linear', probability=True)

        print("Fitting KNN ...")
        knn_clf.fit(x, y)
        print("Fitting Gaussian ...")
        gaussian.fit(x, y)
        print("Fitting Linear SVM ...")
        linearSVM.fit(x, y)

        embeddings = X
        sum_probs = 0
        for i, filename in enumerate(trained_positives):
            print("-"*50)
            print("Checking for {} ...".format(os.path.basename(filename)))

            img = cv2.imread(filename)
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            face_tensors = self.face_extracter.detect(img)
            frame_embeddings = self.resnet(
                torch.unsqueeze(face_tensors[0][1], 0))
            print("Predictions : SVM -> {} Gaussian -> {} KNN -> {}".format(
                linearSVM.predict(
                    frame_embeddings.detach().numpy().reshape(1, 512)),
                gaussian.predict(
                    frame_embeddings.detach().numpy().reshape(1, 512)),
                knn_clf.predict(
                    frame_embeddings.detach().numpy().reshape(1, 512))
            ))

            svm_probs = linearSVM.predict_proba(
                frame_embeddings.detach().numpy().reshape(1, 512)).ravel()
            gaussian_probs = gaussian.predict_proba(
                frame_embeddings.detach().numpy().reshape(1, 512)).ravel()
            knn_probs = knn_clf.predict_proba(
                frame_embeddings.detach().numpy().reshape(1, 512)).ravel()

            print("Prediction Probabilities : SVM -> {} Gaussian -> {} KNN -> {}".format(
                svm_probs[np.argmax(svm_probs)],
                gaussian_probs[np.argmax(gaussian_probs)],
                knn_probs[np.argmax(knn_probs)]
            ))

            sum_probs += svm_probs[np.argmax(svm_probs)] + gaussian_probs[np.argmax(
                gaussian_probs)] + knn_probs[np.argmax(knn_probs)]

        print("Threshold : ", sum_probs/21)
        print("-"*50 + "Stepping into the unknown" + "-"*50)

        for filename in untrainedUnknown:
            print("checking for {} ...".format(os.path.basename(filename)))
            img = cv2.imread(filename)
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            face_tensors = self.face_extracter.detect(img)
            frame_embeddings = self.resnet(
                torch.unsqueeze(face_tensors[0][1], 0))

            svm_probs = linearSVM.predict_proba(
                frame_embeddings.detach().numpy().reshape(1, 512)).ravel()
            gaussian_probs = gaussian.predict_proba(
                frame_embeddings.detach().numpy().reshape(1, 512)).ravel()
            knn_probs = knn_clf.predict_proba(
                frame_embeddings.detach().numpy().reshape(1, 512)).ravel()

            print("Prediction Probabilities : SVM -> {} Gaussiam -> {} KNN -> {}".format(
                svm_probs[np.argmax(svm_probs)],
                gaussian_probs[np.argmax(gaussian_probs)],
                knn_probs[np.argmax(knn_probs)]
            ))
