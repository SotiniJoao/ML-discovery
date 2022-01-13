import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm, tree
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

#abrindo o dataset e validando se não existem valores vazios
seeds=pd.read_excel("C:/Users/jl_sa/Downloads/seeds.xlsx")
n=0
for i in seeds.items():
    if i==' ':
        print('O item {0} deve ser removido'.format(i))
        n+=1
if(n==0):
    print('Dataset ok!')

#Criando objetos que recebem as funções de cada modelo de classificação
gnbmodel = GaussianNB()
svmmodel = svm.SVC()
treemodel = tree.DecisionTreeClassifier()

#Formatando os dados para iniciar o treinamento
X = seeds.loc[:,['V1','V2','V3','V4','V5','V6','V7']]
X = np.asarray(X)
y = seeds.loc[:,'Class']
y = np.asarray(y)

#Usando o train_test_split para separar os dados e treinando os modelos para cada tipo de classificação e fazer predições
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.047)
svmmodel.fit(x_train,y_train)
svm_predict= svmmodel.predict(x_test)
print("SVM model predictions:\n{}".format(svm_predict))
gnbmodel.fit(x_train,y_train)
gnbpredict = gnbmodel.predict(x_test)
print("Gaussian Naive Bayes model predictions:\n{}".format(gnbpredict))
treemodel.fit(x_train,y_train)
treepredict = treemodel.predict(x_test)
print("Tree Classification Model predictions:\n{}".format(treepredict))
labels = ['Class 1','Class 2', 'Class 3']

svmacc= 0
gnbacc= 0
treeacc= 0

#Medindo a acurácia dos modelos:
for i in range(0,9):
    if(svm_predict[i]==y_test[i]):
        svmacc +=1
    if(gnbpredict[i]==y_test[i]):
        gnbacc +=1
    if(treepredict[i]==y_test[i]):
        treeacc+=1

svmacc = svmacc/len(y_test)
gnbacc = gnbacc/len(y_test)
treeacc = treeacc/len(y_test)

print('The SVM model accuracy was:{0}'.format(svmacc))
print('The Gaussian Naive Bayes model accuracy was:{0}'.format(gnbacc))
print('The Tree Classification model accuracy was:{0}'.format(treeacc))


#Plotando as confusion matrices
plot_confusion_matrix(svmmodel,x_test,y_test,display_labels=labels, normalize='all', values_format='.2%', cmap='Blues')
plot_confusion_matrix(gnbmodel,x_test,y_test,display_labels=labels, normalize='all',values_format='.2%', cmap='Purples')
plot_confusion_matrix(treemodel,x_test,y_test,display_labels=labels,normalize='all',  values_format='.2%', cmap='Greens')
plt.show()