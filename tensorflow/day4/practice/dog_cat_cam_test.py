# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
#from google.colab.patches import cv2_imshow

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

names=['cat', 'dog'] ## names[0]= cat, names[1]=dog


# %%
model = tf.keras.models.load_model('cats_and_dogs_aug_binary_classification.hdf5')  # 모델을 새로 불러옴


# %%
capture = cv2.VideoCapture(0)#'dog_video.mp4')
#capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print(capture)
print("Run.....")
# %%
while True:
    ret, frame = capture.read()  ## ret =True or False // frame 웹캠으로 순간 캡쳐된 이미지
    #print(ret)
    #print(frame)
    if ret == False or ret ==False:
        continue
    
    img_resize = cv2.resize(frame, (128, 128))
    test_image_reshape = img_resize.reshape(1, 128, 128, 3).astype('float32')
    
    y = model.predict(test_image_reshape)
    ## y=[[0.6]]
    if y[0] <= 0.5:
        class_num = 0
    else:
        class_num = 1
    class_name=names[class_num]  ## y= [[0.9 , 0.1]]

    location = (10, 20 + 20)
    cv2.putText(frame, str(class_name) +str(y[0]), location, cv2.FONT_ITALIC, 1, (0, 255, 0), 2)
    
    print("VideoFrame.....")
    cv2.imshow("VideoFrame", frame)
    #cv2_imshow(frame)

    if cv2.waitKey(1) > 0: break


