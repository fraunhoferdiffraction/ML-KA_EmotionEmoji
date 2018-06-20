![](https://github.com/fraunhoferdiffraction/ML-KA_EmotionEmoji/blob/master/emoji_recognition_cover.png)

# ML-KA_EmotionEmoji

This is a Python implementation of the neural network architecture for emotion classification, proposed by Gudi. [Paper](https://arxiv.org/abs/1512.00743). This network shows 67% accuracy on the FER2013 dataset and is capable of running in real time on CPUs.
<br/>  
The weights have not been uploaded due to GitHub's 100MB file size limit.

This network has been trained with `FER2013` and `FER2013+` datasets. The first one provides labeled 48x48 images, labeled by one person per image. The second dataset uses FER2013 as image source but provides better labeling with lower label noise. FER2013+ has been labeled by 10 different people and contains records about each decision.  
<\br>
To train the network datasets have to be placed in the folder `dataset_csv`. FER2013 Dataset can be downloaded from the official website of [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) and the FER2013+ dataset can be found [here](https://github.com/Microsoft/FERPlus)
## Training
`python3 train.py -cascade`  
</br>
This class uses standard HAAR CASCADE FILTER provided by OpenCV to recognize face on images from the FER2013 dataset. It is being used to crop the faces and feed them to the neural network. This algorithm is very fast, but the quality of face recognition is low and almost 30% of images in the dataset can not be cropped properly. 
Instead you can use the flag `-yolo` to use the Darknet implementation of the state-of-art YOLOv2 Face Recognition algorithm 
</br>  

`python3 train.py -yolo`  
</br>  
Make sure you have installed [Darknet](https://pjreddie.com/darknet/install/) and [Darkflow](https://github.com/thtrieu/darkflow) frameworks in before. Yolo works significantly better with the dataset and can recognize faces on more than 98% of provided images. But it takes longer to fully preprocess the dataset. This algorithm will also work with CPU builds of darknet. </br>

 `python3 train.py -continue`   
</br>
You can cuntinue training your existing network by running this class with `-continue` flag

## Running
 `python3 start.py -cascade`   
</br>

This class uses HAAR CASCADE FILTER of OpenCV to recognize face on frames provided by the webcam. It is capable of running in real time on CPU with >20FPS. Although you can also run this class with `-yolo` flag:  </br>


 `python3 start.py -yolo`   
</br>
In this case you will need to have a Darknet build with support of nVidia CUDA to have real time performance. YOLO can detect faces with very good accuracy even in bad conditions (bad light, glasses, covered face parts, side view etc.) so it is worth using if you have a CUDA compatible GPU
