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
        self.nodes = 0  # the number of all nodes
        self.leaf = 0  # the leafnodes of the tree

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

    def printtree(self):
        self.preorder(self.root,0)

    def predict(self,test):
        list = []
        for index, row in test.iterrows():
            ans = self.traverse(self.root, row)
            list.append(ans)
        return list

    def traverse(self, root, row):
        if root.attribute != None:
            if row[root.attribute] == 0:
                return self.traverse(root.left, row)
            else:
                return self.traverse(root.right, row)

        elif root.label != None:
            return root.label

    def prune(self, factor):
        return

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

    def countnodes(self):
        self.nodes = 0
        self.preorder_nodes(self.root)

    def preorder_nodes(self, root):
        if root == None:
            return 
        else:
            self.nodes = self.nodes+1
        self.preorder_nodes(root.left)
        self.preorder_nodes(root.right)

    def countleaf(self):
        self.leaf = 0
        self.preorder_leaf(self.root)

    def preorder_leaf(self,root):
        if root == None:
            return
        elif root.label != None:
            self.leaf = self.leaf + 1
        self.preorder_leaf(root.left)
        self.preorder_leaf(root.right)

    def accuracy(self,dataset):
        X, y = datasplit(dataset)
        y_pred = self.predict(X)
        y_true = y.values.tolist()
        return accuracy(y_pred, y_true)


def datasplit(dataset):
    columns = [i for i in list(dataset.columns) if i != 'Class']
    X = dataset[columns]
    y = dataset['Class']
    return X, y


def accuracy(y_pred,y_true):
    if len(y_pred) != len(y_true):
        return None
    count = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            count = count+1
    return count/len(y_pred)


def print_prepruned(dt,training_set,validation_set,test_set):
    print('Pre-Pruned Accuracy')
    print('-----------------------------------')
    print('Number of training instances=%d' % (training_set.shape[0]))
    print('Number of training attributes=%d' % (training_set.shape[1]))
    dt.countnodes()
    print('Total number	of nodes in	the	tree=%d' % (dt.nodes))
    dt.countleaf()
    print('Number of leaf nodes	in the tree=%d' % (dt.leaf))
    print('Accuracy of the model on	the	training dataset=%f' % (dt.accuracy(training_set)))
    print('')
    print('Number of validation	instances=%d' % (validation_set.shape[0]))
    print('Number of validation	attributes=%d' % (training_set.shape[1]))
    print('Accuracy	of the model on	the	validation dataset before pruning=%f' % (dt.accuracy(validation_set)))
    print('')
    print('Number of testing instances=%d' % (test_set.shape[0]))
    print('Number of testing attributes=%d' % (test_set.shape[1]))
    print('Accuracy	of the model on	the	testing	dataset=%f' % (dt.accuracy(test_set)))


def print_postpruned(dt,training_set,validation_set,test_set):
    print('Post-Pruned Accuracy')
    print('-----------------------------------')
    print('Number of training instances=%d' % (training_set.shape[0]))
    print('Number of training attributes=%d' % (training_set.shape[1]))
    dt.countnodes()
    print('Total number	of nodes in	the	tree=%d' % (dt.nodes))
    dt.countleaf()
    print('Number of leaf nodes	in the tree=%d' % (dt.leaf))
    print('Accuracy of the model on	the	training dataset=%f' % (dt.accuracy(training_set)))
    print('')
    print('Number of validation	instances=%d' % (validation_set.shape[0]))
    print('Number of validation	attributes=%d' % (training_set.shape[1]))
    print('Accuracy	of the model on	the	validation dataset after pruning=%f' % (dt.accuracy(validation_set)))
    print('')
    print('Number of testing instances=%d' % (test_set.shape[0]))
    print('Number of testing attributes=%d' % (test_set.shape[1]))
    print('Accuracy	of the model on	the	testing	dataset=%f' % (dt.accuracy(test_set)))


if __name__ == '__main__':
    #training_set=input('Please input the path of trainning data set: ')
    #test_set=input('Please input the path of test data set: ')
    #validation_set = input('Please input the path of validation data set: ')
    training_set = 'data_sets2/training_set.csv'
    validation_set = 'data_sets2/validation_set.csv'
    test_set = 'data_sets2/test_set.csv'
    training_set = precesssing(training_set)
    validation_set = precesssing(validation_set)
    test_set = precesssing(test_set)

    dt = DesicionTree()
    dt.createTree(training_set)
    print_prepruned(dt, training_set, validation_set, test_set)
    #X, y = datasplit(validation_set)
    #y_pred = dt.predict(X)
    #y_true = y.values.tolist()
    #print(accuracy(y_pred, y_true))





