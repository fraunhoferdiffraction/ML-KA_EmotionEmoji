![](https://github.com/fraunhoferdiffraction/ML-KA_EmotionEmoji/blob/master/emoji_recognition_cover.png)

# ML-KA_EmotionEmoji

This is a Python implementation of the emotion recognition neural network architecture, proposed by Gudi. [Paper](https://arxiv.org/abs/1512.00743). This network shows 67% accuracy and is capable of running in real time on CPUs.
<br/>  
The weights have not been uploaded due to GitHub's 100MB file size limit.

This network has been trained with `FER2013` and `FER2013+` datasets. The first one provides labeled 48x48 image dataset, labeled by one person per image. The second dataset provides better labeling with lower label noise, because each label has been made by 10 different people. To train the network datasets have to be placed in the folder `dataset_csv`. FER2013 Dataset can be downloaded from the official website of [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) while the FER2013+ dataset can be found [here](https://github.com/Microsoft/FERPlus)
## Training
`python3 train.py -cascade`  
</br>
This class uses standard HAAR CASCADE FILTER provided by OpenCV to recognize face on images from the FER2013 dataset in order to crop them out and feed to the neural network. This algorithm is very fast, but the quality of face recognition is extremely low. 
Instead you can use the flag `-yolo` to use the Darknet implementation of the state-of-art YOLOv2 Face Recognition algorithm 
</br>  

`python3 train.py -yolo`  
</br>  
Be sure you have installed [Darknet](https://pjreddie.com/darknet/install/) and [Darkflow](https://github.com/thtrieu/darkflow) frameworks in before. Yolo works significantly better with the dataset and can recognize faces on more than 98% of provided images. But it takes significantly longer to fully preprocess the dataset. </br>

 `python3 train.py -continue`   
</br>
You can cuntinue training your existing network by running this class with `-continue` flag

## Running
 `python3 train.py -cascade`   
</br>

This class uses HAAR CASCADE FILTER of OpenCV to recognize face on frames provided by the webcam. It is capable of running in real time on CPU with >20FPS. Although you can also run this class with `-yolo` flag:  </br>


 `python3 train.py -yolo`   
</br>
In this case you will need to have a Darknet build with support of nVidia CUDA to have good real time performance. YOLO can detect faces with very good accuracy even in bad conditions (bad light, glasses, covered face parts etc.) so it is worth using if you have a CUDA compatible GPU
