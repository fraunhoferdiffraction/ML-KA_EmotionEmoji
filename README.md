![](https://github.com/fraunhoferdiffraction/ML-KA_EmotionEmoji/blob/master/emoji_recognition_cover.png)

# ML-KA_EmotionEmoji

This is a Python implementation of the emotion recognition neural network architecture, proposed by Gudi. [Paper](https://arxiv.org/abs/1512.00743). This network shows 67% accuracy and is capable of running in real time on CPUs.
<br/>  
The weights have not been uploaded due to GitHub's 100MB file size limit.

This network has been trained with `FER2013` and `FER2013+` datasets. The first one provides labeled 48x48 image dataset, labeled by one person per image. The second dataset provides better labeling with lower label noise, because each label has been made by 10 different people. To train the network datasets have to be placed in the folder `dataset_csv`. FER2013 Dataset can be downloaded from the official website of [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) while the FER2013+ dataset can be found [here](https://github.com/Microsoft/FERPlus)
