from facenet_pytorch import MTCNN
import cv2
import os
from PIL import Image, ImageDraw
import numpy as np

mtcnn_module = MTCNN(keep_all=True)

test_imgs = os.listdir("tests/demo")
for test_img in test_imgs :
    img = cv2.imread(os.path.join("tests/demo", test_img))
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    bbox, probs = mtcnn_module.detect(img)
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)
    draw.rectangle(bbox[0].tolist(), outline=(255,0,0), width=1)
    cv2.imshow("Tracked Image", cv2.cvtColor(np.asarray(img_draw), cv2.COLOR_BGR2RGB))
    if cv2.waitKey() and 0xFF == ord('y'):
        continue

