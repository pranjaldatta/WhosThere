# WhosThere
A Facial Recognition library that simplifies Detection, Recognition, and Clustering.

### About The Module

The module is built on top of a Pytorch Implementation of Facenet. For the purposes of this module, we use InceptionResnetV1 architecture pretrained on Casia-Webface. (VGFace2 is also available). The moudule was built with the motive of understanding FR architectures along with building something that simplifies an otherwise taxing process and also is fun along the way.

User registration and Verification is reduced to function calls. Images of users that need to be 'onboarded' the system, need to be put in a folder or image needs to supplied directly. Verification can be done by making the system look into folders or directly supplying images



A few more modules/modifications will be added onto the repository : 
* Module that clusters potraits of different people into separate categories
* Module that indicates who all are present in a video on frame by frame basis
* verify returns a truth or falsity value for each detected face along with predictions


The model performs well when it comes to correct classification of Known Positives (i.e. unseen pictures of the people in the database).The issue is arises when the model enocunters unknown-unseen pictures (i.e. pictures of people not in the database). This is a fundamental problem in such architectures. These issues have been referenced in OpenFace and David Sandberg's Github repo. To solve this out of the many approaches tried,  the one used is:

* Make available unseen-positive images of known persons and unseen-unknown images of unknown individuals
* Set a minimum accuracy for unseen-known images of known persons classification. (default is : .90)
* The model depending on the the stored-embeddings and the unknown-images provided, comes up with threshold that allows known-positive classification at or a better accuracy than the minimum and allows for best possible accuracy for unknown classification. (More changes are in line to make this process more optimized)


### Quick Start 

* Clone the repository with the following command and place it into site-packages of your conda environment
```
git clone https://github.com/pranjaldatta/WhosThere.git
```
* Open up an editor of your choice. And place in the following code
```
from WhosThere.lib.recognition import recognizer
recog = recognizer.Recognizer(embedLoc = <location to store the embeddings", threshold = <minimum accuracy required>, verbose=False)

```
* To Onboard users 
```
embeddings = recog.onboard(lookIn=<folder where images are stored>)
```
* To verify users 
```
img = cv2.imread("<read image from>")
pred = cv2.verify(lookIn=None, img=img)
print(pred['Prediction'])
```

More documentation/features will be provdided shortly.
This repository will undergo furthur changes! 
