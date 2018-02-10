from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import matplotlib
import math


# fetch data from CSV files
def precesssing(path):
    #data_sets2/training_set.csv
    raw_data= pd.read_csv(path)
    return raw_data

#compute the entropy
def entropy(dataset):
    total=dataset.shape[0]
    groups=dataset.groupby('Class')
    E=0
    for key in groups.groups.keys():
        group = groups.get_group(key)
        count=group.shape[0]
        E= E-count/total*math.log2(count/total)
    return E

#InformationGain
def infogain(dataset,attribute):
    total=dataset.shape[0]
    groups=dataset.groupby(attribute)
    IG=entropy(dataset)
    for key in groups.groups.keys():
        group=groups.get_group(key)
        count=group.shape[0]
        IG=IG-count/total*entropy(group)
    return IG

# def split(tree,attribute):
#     return
#

#build the Desicion Tree
def buildtree(dataset):

    return
#
# def prune():
#     return
#
# def print():
#     return

#define Desicion Tree's Node
class TreeNode:
    def __init__(self,dataset):
        self.dataset = dataset
        self.left = None
        self.right = None
        self.attribute = None
        self.label = None

#define Desicion Tree
class DesicionTree:
    def __int__(self, dataset):
        self.root = None
        #self.dataset=dataset

    def train(dataset):

        return;

    def build(self,dataset):

        # if dataset has been used out
        if dataset.empty:
            return None

        self.root = TreeNode(dataset)

        list, best_attr = self.split(dataset)

        # if the class is pure
        if len(list) < 2:
            self.root.label =
            return self.root

        # if the class is not pure
        self.root.attribute = best_attr
        self.root.dataset = dataset
        self.root.left = self.build(list[0])
        self.root.right = self.build(list[1])

        return self.root


    # Split the dataset based on the optimal attirbute
    def split(self,dataset):
        max = 0
        best_attr = None
        # print(dataset.columns)
        for item in dataset.columns[0:-1]:
            # print(item)
            IG = infogain(dataset, item)
            # print(info)
            if (max < IG):
                max = IG
                best_attr = item
        # print(best_attr)
        # print(max)
        groups = dataset.groupby(best_attr)
        list = []
        for key in groups.groups.keys():
            group = groups.get_group(key)
            list.append(group)
        return list, best_attr



if __name__ == '__main__':
    training_set=input('Please input the path of trainning data set: ')
    #test_set=input('Please input the path of test data set: ')
    #validation_set = input('Please input the path of validation data set: ')
    training_set=precesssing(training_set)
    split(training_set)





