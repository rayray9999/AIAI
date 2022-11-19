import dataset
import adaboost
import utils
import detection
import matplotlib.pyplot as plt
import os

print('Loading images')
trainData = dataset.loadImages('data/train')
print(f'The number of training samples loaded: {len(trainData)}')
testData = dataset.loadImages('data/test')
print(f'The number of test samples loaded: {len(testData)}')
#%%
#dataset.loadImages()
print("done")
print('\nEvaluate your classifier with training dataset')
clf=adaboost.rgb();
utils.evaluate(clf, trainData)

print('\nEvaluate your classifier with test dataset')
utils.evaluate(clf, testData)
#%%
# Part 2: Implement selectBest function in adaboost.py and test the following code.
# Part 3: Modify difference values at parameter T of the Adaboost algorithm.


#%%

#%%
# Part 4: Implement detect function in detection.py and test the following code.
