from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import matplotlib
import math
import random
import copy


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
        self.inter = 0 # the internal nodes of the tree
        self.count = 0 # the helper count
        self.once = False

    def createTree(self,dataset):

        self.root = self.build(self.root, dataset)
        self.countnodes()
        self.countleaf()
        self.inter = self.nodes-self.leaf

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
            root.dataset = dataset
            return root

        # if the class is pure
        try:
            if len(list) < 2:
                root.label = list[0].iloc[0]['Class']
                root.dataset = dataset
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
        self.preorder(self.root, 0)

    def predict(self,test):
        list = []
        for index, row in test.iterrows():
            ans = self.traverse(self.root, row)
            list.append(ans)
        return list

    def traverse(self, root, row):
        if root == None:
            return
        if root.attribute != None:
            if row[root.attribute] == 0:
                return self.traverse(root.left, row)
            else:
                return self.traverse(root.right, row)

        elif root.label != None:
            return root.label

    def prune(self, factor, validation_set):
        internodes = self.nodes-self.leaf
        toprunenodes = int(internodes*factor)
        best_accuracy = self.accuracy(validation_set)
        current_accuracy = 0.
        max_times = 50
        iterate_times = 0
        while best_accuracy > current_accuracy:
            iterate_times += 1
            newtree = copy.deepcopy(self)
            for i in range(toprunenodes):
                # newtree.countinter()
                try:
                    p = random.randint(0, self.inter - 1)
                except:
                    print(newtree.inter)

                #newtree.countnodes()
                newtree.count = 0
                newtree.once = False
                newtree.preorder_prune(newtree.root, p)
                # newtree.countnodes()
                # print(newtree.nodes)
            current_accuracy = newtree.accuracy(validation_set)
            if current_accuracy > best_accuracy:
                print('-------------------Prune Summary-------------------')
                print('after %dth prune, the accuracy goes up!' % (iterate_times))
                print('---------------------------------------------------')
                #newtree.countnodes()
                # print('the newtree was pruned %d nodes and now has %d nodes' %)
                return newtree
            if iterate_times > max_times:
                print('pruned the tree for 30 times but the accuracy did not go up')
                return newtree
        return newtree


    def preorder_prune(self, root, p):
        if root == None:
            return
        if root.attribute!= None:
            if self.count == p and self.once == False:
                root.label = int(root.dataset.mode().iloc[0]['Class'])
                root.right = None
                root.left = None
                root.attribute = None
                self.once = True
                return
            else:
                self.count += 1
                self.preorder_prune(root.left, p)
                self.preorder_prune(root.right, p)
        else:
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
            if (max < IG):
                max = IG
                best_attr = item

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

    def countinter(self):
        self.inter = 0
        self.preorder_inter(self.root)

    def preorder_inter(self, root):
        if root == None:
            return
        elif root.label == None:
            self.inter = self.inter + 1
        self.preorder_inter(root.left)
        self.preorder_inter(root.right)

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
    print('Total number of nodes in the tree=%d' % (dt.nodes))
    dt.countleaf()
    print('Number of leaf nodes in the tree=%d' % (dt.leaf))
    print('Accuracy of the model on the training dataset=%f' % (dt.accuracy(training_set)))
    print('')
    print('Number of validation instances=%d' % (validation_set.shape[0]))
    print('Number of validation attributes=%d' % (training_set.shape[1]))
    print('Accuracy of the model on the validation dataset before pruning=%f' % (dt.accuracy(validation_set)))
    print('')
    print('Number of testing instances=%d' % (test_set.shape[0]))
    print('Number of testing attributes=%d' % (test_set.shape[1]))
    print('Accuracy of the model on the test dataset=%f' % (dt.accuracy(test_set)))


def print_postpruned(dt,training_set,validation_set,test_set):
    print('Post-Pruned Accuracy')
    print('-----------------------------------')
    print('Number of training instances=%d' % (training_set.shape[0]))
    print('Number of training attributes=%d' % (training_set.shape[1]))
    dt.countnodes()
    print('Total number of nodes in the tree=%d' % (dt.nodes))
    dt.countleaf()
    print('Number of leaf nodes in the tree=%d' % (dt.leaf))
    print('Accuracy of the model on the training dataset=%f' % (dt.accuracy(training_set)))
    print('')
    print('Number of validation instances=%d' % (validation_set.shape[0]))
    print('Number of validation attributes=%d' % (training_set.shape[1]))
    print('Accuracy of the model on the validation dataset before pruning=%f' % (dt.accuracy(validation_set)))
    print('')
    print('Number of testing instances=%d' % (test_set.shape[0]))
    print('Number of testing attributes=%d' % (test_set.shape[1]))
    print('Accuracy of the model on the test dataset=%f' % (dt.accuracy(test_set)))

if __name__ == '__main__':
    #training_set=input('Please input the path of trainning data set: ')
    #test_set=input('Please input the path of test data set: ')
    #validation_set = input('Please input the path of validation data set: ')
    #factor = float(input('Please input the prune factor: '))
    training_set = 'data_sets2/training_set.csv'
    validation_set = 'data_sets2/validation_set.csv'
    test_set = 'data_sets2/test_set.csv'
    factor = 0.01
    training_set = precesssing(training_set)
    validation_set = precesssing(validation_set)
    test_set = precesssing(test_set)

    dt = DesicionTree()
    dt.createTree(training_set)

    dt.printtree()
    print_prepruned(dt, training_set, validation_set, test_set)

    dt = dt.prune(factor, validation_set)

    print_postpruned(dt, training_set, validation_set, test_set)
    dt.printtree()





