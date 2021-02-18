import math
import operator
import random
from sklearn.model_selection import train_test_split
import numpy as np
import sys



def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: #the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * math.log(prob,2) #log base 2
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):        #iterate over all the features
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        uniqueVals = set(featList)       #get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)     
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature                      #returns an integer

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet,labels,maxDepth):#label is the name of features!
    classList = [example[-1] for example in dataSet]
    
    if maxDepth == 0:
        return majorityCnt(classList)
        
    maxDepth -= 1
    
    if classList.count(classList[0]) == len(classList): 
        return classList[0]#stop splitting when all of the classes are equal
    if len(dataSet[0]) == 2: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    subLabels = labels[:]
    del(subLabels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
               #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),
                                                  subLabels,maxDepth)
    return myTree  

def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

def accuracy(dataset,tree,fea_name):
    number=0.0
    for i in dataset:
        if classify(tree, fea_name, i[:-1])==i[-1]:
            number+=1
    precision=number/len(dataset)   
    return precision
 
'''
def split_train_test(dataset):
    dataset=np.array(dataset)
    a1=np.array(dataset)
    results=a1[:,-1].tolist()
    data=a1[0:len(a1),:-1].tolist()
    
    x_train,x_test,y_train,y_test=train_test_split(data, results,test_size=0.25,random_state=0)
    
    return  x_train,x_test,y_train,y_test
'''       







#jack=["rain","hot","high","weak"]
if __name__ == "__main__":

    
    
    if len(sys.argv) < 2:
        
        print("not enough arguments specified")
        sys.exit(1)
    else: 
        fname = sys.argv[1]
        fr = open(f'./{fname}')
        fea_name=fr.readline().strip().split('\t')
        dataset= [inst.strip().split('\t') for inst in fr.readlines()[0:]]

        if len(sys.argv) >= 3:
            maxDepth = int(sys.argv[2])
        else:
            maxDepth = 5
    
        tree =createTree(dataset,fea_name,maxDepth)
        print("Accuracy of training set is:",accuracy(dataset, tree, fea_name))
    
        if len(sys.argv) == 4:
            test_fname = sys.argv[3]
            fr = open(f'./{test_fname}')
            fea_name=fr.readline().strip().split('\t')
            dataset_test= [inst.strip().split('\t') for inst in fr.readlines()[0:]]
            print("Accuracy of testing set is:", accuracy(dataset_test, tree, fea_name))
        #print(tree)


