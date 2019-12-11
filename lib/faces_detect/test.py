from PIL import Image
import os
import detect_frames
import matplotlib.pyplot as plt 

img = Image.open("/home/pranjal/Pictures/people.jpg")
#plt.imshow(img)
#plt.show()

det_frames = detect_frames.detect_frames(0.5 , os.getcwd())
data = det_frames.detect(img)
