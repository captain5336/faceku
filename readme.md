## 1. Brief Introduction
- The class FaceModel provides models for face detect, feature encoding, recognition. 
- The class FaceKu provides methods to establish face database, including add, update, delete face, and load face features of known people stored in databases. 
- The application are built on algorithms of dlib face recognition. 
- It was tested on Windows10, Ubuntu 22.04


##  2. Set up running environment


1) Create project structure, like the below 

```
-- MyProject 
    |-- FaceModel.py
    |-- FaceKu.py
    |-- face_demo.py
    |-- model
        |-- dlib_face_recognition_resnet_model_v1.dat 
        |-- shape_predictor_68_face_landmarks.dat
    |-- image
       |-- train
       |-- face
```

​		If you add batches faces from excel, recommand to put raw face images in MyProject/image/train/

2) Prepare database

Create face_db databases in MySql or use current database., then create table face_feature in MySql with the structure 

```
CREATE TABLE IF NOT EXISTS `face_feature` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `feature` blob NOT NULL,
  `people_id` int(11) DEFAULT NULL,
  `name` varchar(20) NOT NULL,
  `photo_path` varchar(300) NOT NULL,
  PRIMARY KEY (`id`)
) 
```



3) Copy .py files the  project directory,  copy model files into  MyProject/model/

4) install dependencies 

```
pip install opencv-python
pip install numpy
pip install pandas
pip install dlib
```


## 3. Examples to use 



In your .py file,  import FaceKu,  then use the module referring to examples. 


#### 1）Add face into database and recognize face.  
Refer to following example:

​    

```
from FaceKu import *　
    import cv2
    import numpy as np

​    fk = FaceKu()

    # Prepare test data, image including face

​    imgx = cv2.imread('image/train/t-1.jpg')
​    width = imgx.shape[1]
​    input_width = 500
​    ratio = input_width/width
​    input_height = int(imgx.shape[0]*ratio)
​    imgx = cv2.resize(imgx, (input_width, input_height), cv2.INTER_LINEAR)

    # add face into database

​    res = fk.add_face(imgx, people_info)
​    print("result is : ", res)

    # recognize face, return name of recognized person. 

​    res = fk.face_compare(imgx)
​    print("recognization result : ", res)
```




#### 2) Use FaceKu to recognize face in video streaming 

refer to face_demo.py 

## 4. Description for main classes and functions

#### 1) Class FaceModel:

```
NAME
    FaceModel

CLASSES
    builtins.object
        FaceModel
    
    class FaceModel(builtins.object)
     |  Provide functions for face detect, feature abstraction, face_recognition
     |  
     |  Methods defined here:
     |  
     |  __init__(self)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  compare_faces_ordered(self, face_encodings, face_names, encoding_to_check)
     |      Returns the ordered distances and names when comparing a list of face 
     |      encodings against a candidate feature to check
     |  
     |  face_encodings(self, face_image, num_upsample=1, num_jitters=1)
     |      Detect faces and abstract face features
     |      Returns the 128D features and face rectangle for each face in the image
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)

FILE
    FaceModel.py
```



#### 2) Class FaceKu:

```
NAME
    FaceKu

CLASSES
    builtins.object
        FaceKu
    
    class FaceKu(builtins.object)
     |  Attributes:
     |      feature_list: type(list)  face features for all valid faces
     |      name_list: type(list)     name labels to feature list  
     |  Methods:
     |      add_face()
     |      del_face()
     |      update_face()
     |      face_compare()
     |  Dependency: 
     |      class FaceModel
     |  
     |  Methods defined here:
     |  
     |  __del__(self)
     |  
     |  __init__(self)
     |      load all features and names into memory
     |  
     |  add_face(self, image, person_info)
     |      add one face into face feature table
     |      Args:
     |          image (str): absolute path of face photo file 
     |          person_info (dict): person info with input photo
     |      return
     |          res(bool): result
     |  
     |  add_face_from_excel(self, excel_file)
     |  
     |  del_face(self, name)
     |      delete face from face feature table
     |  
     |  face_compare(self, image)
     |  
     |  get_compare_level(self)
     |  
     |  load_features(self)
     |  
     |  save_to_db(self, data, action=1)
     |      Save face feature and person label info into database
     |      Arguments:
     |          data: type(tuple),  ( feature, name, photo_path)
     |          actions:  type(int), 1 - Insert, 2 - Update
     |      Returns:
     |          res: type(boolean)
     |  
     |  set_compare_level(self, value)
     |  
     |  update_face(self, image, person_info)
     |      update face feature with specified face image.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |  
     |  compare_level = 0.6
     |  
     |  face_store_path = 'image/face/'
     |  
     |  feature_list = []
     |  
     |  name_list = []

FILE
    FaceKu.py
```




