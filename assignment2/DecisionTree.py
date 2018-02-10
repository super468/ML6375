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
    raw_data = pd.read_csv(path)
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

    def __init__(self):
        self.root = None

    def createTree(self,dataset):

        self.root = self.build(self.root, dataset)

    def build(self, root, dataset):

        # if dataset has been used out
        if dataset.empty:
            return None

        if root == None:
            root = TreeNode(dataset)

        list, best_attr = self.split(dataset)

        # if attributes had been used out
        if best_attr == None:
            # get the majority element in the Class column
            root.label = list[0].mode().iloc[0]['Class']
            return root

        # if the class is pure
        try:
            if len(list) < 2:
                root.label = list[0].iloc[0]['Class']
                return root
        except:
            print('list error')

        # if the class is not pure and attributes still remain
        root.attribute = best_attr
        root.dataset = dataset
        root.left = self.build(root.left, list[0])
        root.right = self.build(root.right, list[1])

        return root

    def print(self):
        self.preorder(self.root,0)

    def preorder(self, root, depth, attr=None, lf=None):

        if root == None:
            return

        if attr != None and lf != None:
            for i in range(depth-1):
                print('| ', end='')
            print(attr + ' = ' + str(lf) + ' : ', end='')
            if root.label != None:
                print(' ' + str(root.label))
            else:
                print('')

        self.preorder(root=root.left, depth=depth+1, attr=root.attribute, lf=0)
        self.preorder(root=root.right, depth=depth+1, attr=root.attribute, lf=1)


    # Split the dataset based on the optimal attirbute
    def split(self,dataset):
        max = 0
        best_attr = None
        list = []  # return the dataset which exclude the optimal attribute
        # print(dataset.columns)
        for item in dataset.columns[0:-1]:
            # print(item)
            IG = infogain(dataset, item)
            # print(info)
            if (max <= IG):
                max = IG
                best_attr = item
        # print(best_attr)
        # print(max)
        # if best_attr == None:
        #     print(max)
        #     print(dataset)
        if best_attr == None:
            list.append(dataset)
            return list, best_attr
        try:
            groups = dataset.groupby(best_attr)
        except:
            print('Best_attr None')
            print(best_attr)
            print(dataset)

        for key in groups.groups.keys():
            group = groups.get_group(key)
            group = group.drop(best_attr, 1)
            list.append(group)
        return list, best_attr

if __name__ == '__main__':
    #training_set=input('Please input the path of trainning data set: ')
    training_set='data_sets2/training_set.csv'
    #test_set=input('Please input the path of test data set: ')
    #validation_set = input('Please input the path of validation data set: ')
    training_set=precesssing(training_set)
    dt = DesicionTree()
    print(dt.root)
    dt.createTree(training_set)
    dt.print()

    print("Done")





