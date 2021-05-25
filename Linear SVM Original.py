# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 05:37:16 2021

@author: ayomy
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
class LinearSVM:
 
    def __init__(self, max_iteration):
       self.max_iteration = max_iteration
       
       
    def fit1(self, X_train,y_train_1,lambdat):
       self.lambdat1 = lambdat 
       features_no = X_train.shape[1]
       obser_no = X_train.shape[0]
       y_train_1 = y_train_1.tolist()
       y_train_1 = [1 if i == 1 else -1 for i in y_train_1]
       y_train_1 = np.array(y_train_1)
       self.weights_1 = np.zeros(features_no)
       for i in range(self.max_iteration):
           learn_rate = 1 / (self.lambdat1*(i+1))
           j = np.random.choice(obser_no, 1)[0]
           x, y = X_train[j], y_train_1[j]
           score_1 = self.weights_1.dot(x)
           if y*score_1 < 1:
              self.weights_1 = (1 - learn_rate*self.lambdat1)*self.weights_1 + learn_rate*y*x
           else:
              self.weights_1 = (1 - learn_rate*self.lambdat1)*self.weights_1
       return self.weights_1      
    
    def fit2(self, X_train,y_train_2,lambdat):
      self.lambdat2 = lambdat  
      features_no = X_train.shape[1]
      obser_no = X_train.shape[0]
      y_train_2 = y_train_2.tolist()
      y_train_2 = [1 if i == 1 else -1 for i in y_train_2]
      y_train_2 = np.array(y_train_2)
      self.weights_2 = np.zeros(features_no)
      for i in range(self.max_iteration):
          learn_rate = 1 / (self.lambdat2*(i+1))
          j = np.random.choice(obser_no, 1)[0]
          x, y = X_train[j], y_train_2[j]
          score_2 = self.weights_2.dot(x)
          if y*score_2 < 1:
             self.weights_2 = (1 - learn_rate*self.lambdat2)*self.weights_2 + learn_rate*y*x
          else:
             self.weights_2 = (1 - learn_rate*self.lambdat2)*self.weights_2
      return self.weights_2 
  
    def fit3(self, X_train,y_train_3,lambdat):
      self.lambdat3 = lambdat  
      features_no = X_train.shape[1]
      obser_no = X_train.shape[0]
      y_train_3 = y_train_3.tolist()
      y_train_3 = [1 if i == 1 else -1 for i in y_train_3]
      y_train_3 = np.array(y_train_3)
      self.weights_3 = np.zeros(features_no)
      for i in range(self.max_iteration):
          learn_rate = 1 / (self.lambdat3*(i+1))
          j = np.random.choice(obser_no, 1)[0]
          x, y = X_train[j], y_train_3[j]
          score_3 = self.weights_3.dot(x)
          if y*score_3 < 1:
             self.weights_3 = (1 - learn_rate*self.lambdat3)*self.weights_3 + learn_rate*y*x
          else:
             self.weights_3 = (1 - learn_rate*self.lambdat3)*self.weights_3
      return self.weights_3 
  
    def fit4(self, X_train,y_train_4,lambdat):
      self.lambdat4 = lambdat  
      features_no = X_train.shape[1]
      obser_no = X_train.shape[0]
      y_train_4 = y_train_4.tolist()
      y_train_4 = [1 if i == 1 else -1 for i in y_train_4]
      y_train_4 = np.array(y_train_4)
      self.weights_4 = np.zeros(features_no)
      for i in range(self.max_iteration):
          learn_rate = 1 / (self.lambdat4*(i+1))
          j = np.random.choice(obser_no, 1)[0]
          x, y = X_train[j], y_train_4[j]
          score_4 = self.weights_4.dot(x)
          if y*score_4 < 1:
             self.weights_4 = (1 - learn_rate*self.lambdat4)*self.weights_4 + learn_rate*y*x
          else:
             self.weights_4 = (1 - learn_rate*self.lambdat4)*self.weights_4
      return self.weights_4 
   
    def predict1(self,X_test): 
       y_predict_final_1 = np.zeros(len(X_test)) 
       for i in range(len(X_test)):
           score_1 = self.weights_1.dot(X_test[i])
           if score_1 > 0:
               y_predict_final_1[i] = 1
           else:
               y_predict_final_1[i] = -1
       return y_predict_final_1
   
    def predict2(self,X_test): 
       y_predict_final_2 = np.zeros(len(X_test)) 
       for i in range(len(X_test)):
           score_2 = self.weights_2.dot(X_test[i])
           if score_2 > 0:
               y_predict_final_2[i] = 1
           else:
               y_predict_final_2[i] = -1
       return y_predict_final_2

    def predict3(self,X_test): 
       y_predict_final_3 = np.zeros(len(X_test)) 
       for i in range(len(X_test)):
           score_3 = self.weights_3.dot(X_test[i])
           if score_3 > 0:
               y_predict_final_3[i] = 1
           else:
               y_predict_final_3[i] = -1
       return y_predict_final_3 

    def predict4(self,X_test): 
       y_predict_final_4 = np.zeros(len(X_test)) 
       for i in range(len(X_test)):
           score_4 = self.weights_4.dot(X_test[i])
           if score_4 > 0:
               y_predict_final_4[i] = 1
           else:
               y_predict_final_4[i] = -1
       return y_predict_final_4
   
    
    
    def predict_bin_label_1(self):
        y_predict_bin_1 = [1 if i == 1 else 0 for i in self.predict1(X_test)]
        return y_predict_bin_1
    
    def predict_bin_label_2(self):
        y_predict_bin_2 = [1 if i == 1 else 0 for i in self.predict2(X_test)]
        return y_predict_bin_2
    
    def predict_bin_label_3(self):
        y_predict_bin_3 = [1 if i == 1 else 0 for i in self.predict3(X_test)]
        return y_predict_bin_3
    
    def predict_bin_label_4(self):
        y_predict_bin_4 = [1 if i == 1 else 0 for i in self.predict4(X_test)]
        return y_predict_bin_4
    
    def prediction_accuracy1(self,y_test_1):
        correctcount = 0
        wrongcount = 0
        y_predict_final1 = self.predict_bin_label_1()
        testlabel_and_predictedlabel = list(zip(y_test_1,y_predict_final1))
        for i in range(len(testlabel_and_predictedlabel)):
            if (testlabel_and_predictedlabel[i][0]) == (testlabel_and_predictedlabel[i][1]):
                correctcount += 1
            else:
                wrongcount += 1
        accuracyratio = (correctcount/(correctcount+wrongcount))
        return accuracyratio 
    
    def prediction_accuracy2(self,y_test_2):
        correctcount = 0
        wrongcount = 0
        y_predict_final2 = self.predict_bin_label_2()
        testlabel_and_predictedlabel = list(zip(y_test_1,y_predict_final2))
        for i in range(len(testlabel_and_predictedlabel)):
            if (testlabel_and_predictedlabel[i][0]) == (testlabel_and_predictedlabel[i][1]):
                correctcount += 1
            else:
                wrongcount += 1
        accuracyratio = (correctcount/(correctcount+wrongcount))
        return accuracyratio 
    
    def prediction_accuracy3(self,y_test_3):
        correctcount = 0
        wrongcount = 0
        y_predict_final3 = self.predict_bin_label_3()
        testlabel_and_predictedlabel = list(zip(y_test_1,y_predict_final3))
        for i in range(len(testlabel_and_predictedlabel)):
            if (testlabel_and_predictedlabel[i][0]) == (testlabel_and_predictedlabel[i][1]):
                correctcount += 1
            else:
                wrongcount += 1
        accuracyratio = (correctcount/(correctcount+wrongcount))
        return accuracyratio 
    
    def prediction_accuracy4(self,y_test_4):
        correctcount = 0
        wrongcount = 0
        y_predict_final4 = self.predict_bin_label_4()
        testlabel_and_predictedlabel = list(zip(y_test_1,y_predict_final4))
        for i in range(len(testlabel_and_predictedlabel)):
            if (testlabel_and_predictedlabel[i][0]) == (testlabel_and_predictedlabel[i][1]):
                correctcount += 1
            else:
                wrongcount += 1
        accuracyratio = (correctcount/(correctcount+wrongcount))
        return accuracyratio 
    
    #encoding part
    def encode_part1(self):
        final_y = np.vstack([self.predict_bin_label_1(),self.predict_bin_label_2(),self.predict_bin_label_3(),self.predict_bin_label_4()]) 
        final_y = np.transpose(final_y)
        final_y = final_y.tolist()
        final_y_predict = []
        for i in final_y: 
             if i == [0,0,0,0]:
                final_y_predict.append(0)
             elif i == [0,0,0,1]:
                final_y_predict.append(1)
             elif i == [0,0,1,0]:
                final_y_predict.append(2)
             elif i == [0,0,1,1]:
                final_y_predict.append(3)
             elif i == [0,1,0,0]:
                final_y_predict.append(4)
             elif i == [0,1,0,1]:
                final_y_predict.append(5) 
             elif i == [0,1,1,0]:
                final_y_predict.append(6)
             elif i == [0,1,1,1]:
                final_y_predict.append(7)
             elif i == [1,0,0,0]:
                final_y_predict.append(8)
             else:
                final_y_predict.append(9)
        return final_y_predict
        
    def encode_part2(self,y_test):
        final_y_test = y_test.tolist()
        final_y_test_a = []
        for i in final_y_test: 
             if i == [0,0,0,0]:
                final_y_test_a.append(0)
             elif i == [0,0,0,1]:
                final_y_test_a.append(1)
             elif i == [0,0,1,0]:
                final_y_test_a.append(2)
             elif i == [0,0,1,1]:
                final_y_test_a.append(3)
             elif i == [0,1,0,0]:
                final_y_test_a.append(4)
             elif i == [0,1,0,1]:
                final_y_test_a.append(5) 
             elif i == [0,1,1,0]:
                final_y_test_a.append(6)
             elif i == [0,1,1,1]:
                final_y_test_a.append(7)
             elif i == [1,0,0,0]:
                final_y_test_a.append(8)
             else:
                final_y_test_a.append(9)
        return final_y_test_a
    
    def final_prediction_accuracy(self):
        correctcount = 0
        wrongcount = 0
        y_predict_final = self.encode_part1()
        testlabel_and_predictedlabel = list(zip(self.encode_part2(y_test),y_predict_final))
        for i in range(len(testlabel_and_predictedlabel)):
            if (testlabel_and_predictedlabel[i][0]) == (testlabel_and_predictedlabel[i][1]):
                correctcount += 1
            else:
                wrongcount += 1
        accuracyratio = (correctcount/(correctcount+wrongcount))
        return accuracyratio 
    
    def Confusionmatrix(self,y_test):
         y_test = y_test.astype(int)
         y_test = self.encode_part2(y_test)
         y_pred = self.encode_part1()
         plt.figure(figsize=(10,10))
         ax = plt.subplot()
         cm = confusion_matrix(y_test,y_pred,labels = [0,1,2,3,4,5,6,7,8,9])
         sns.heatmap(cm,annot=True ,fmt ='g',ax =ax)
         ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
         ax.set_title('Confusion Matrix for SVM Method')
         return cm,ax
    

np.random.seed (0)
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
from joblib import Memory
mem = Memory("./mycache")

@mem.cache
def get_data():
    data = load_svmlight_file("mnist.scale.bz2")
    return data[0],data[1]

X,y = get_data()
y= y.astype(int)
y = y.tolist()
new_y = []
for i in y:
    if i == 0:
        new_y.append([0,0,0,0])
    elif i == 1:
        new_y.append([0,0,0,1])
    elif i == 2:
        new_y.append([0,0,1,0])
    elif i == 3:
        new_y.append([0,0,1,1])
    elif i == 4:
        new_y.append([0,1,0,0])
    elif i == 5:
        new_y.append([0,1,0,1]) 
    elif i == 6:
        new_y.append([0,1,1,0])
    elif i == 7:
        new_y.append([0,1,1,1])
    elif i == 8:
        new_y.append([1,0,0,0])
    else:
        new_y.append([1,0,0,1])
new_y = np.array(new_y)
X_train,X_test,y_train,y_test = train_test_split(X.toarray(),new_y,test_size = 0.3) 
y_train_1 = y_train[:,0] 
y_train_2 = y_train[:,1]  
y_train_3 = y_train[:,2]
y_train_4 = y_train[:,3]

y_test_1 = y_test[:,0] 
y_test_2 = y_test[:,1]  
y_test_3 = y_test[:,2]
y_test_4 = y_test[:,3]
    
bin = LinearSVM(10300)
bin.fit1(X_train,y_train_1,0.15)
bin.fit2(X_train,y_train_2,0.01)
bin.fit3(X_train,y_train_3,0.01)
bin.fit4(X_train,y_train_4,0.09)

print("The Linear SVM ECOC algorithm performance accuracy is "+str(bin.final_prediction_accuracy()))
print(bin.Confusionmatrix(y_test))