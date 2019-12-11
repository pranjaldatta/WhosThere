from PIL import Image
import os
from WhosThere.lib.faces_detect.detect_frames import detect_frames
import matplotlib.pyplot as plt 
import torch

img = Image.open("/home/pranjal/Pictures/people.jpg")
det_frames = detect_frames(threshold = .8, saveIn = os.getcwd())
result = det_frames.detectAndCrop(img)
result2 = det_frames.detect(img)
result3 = torch.unsqueeze(result2[0][1], 0)
print(result3.shape)