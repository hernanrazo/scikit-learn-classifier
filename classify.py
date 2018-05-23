from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier

#data of animals (body measurements)
#[height(m), length(m), weight(kg)]
X = [[1,2.1,433], [0.7,1.7,50], [0.8,1.6,34], 
     [1.5,2.4,521], [1.4,2.4,359], [0.8,1.5,43], 
     [0.8,1.5,44], [0.5,1.4,44], [1.4,2.4,543], 
     [1.5,2.2,487], [0.7,1.7,45],[1.4,2.5,458]]

#label above data as either 'Polar Bear' or 'Gray Wolf'
Y = ['Polar Bear', 'Gray Wolf', 'Gray Wolf', 'Polar Bear','Polar Bear',
	 'Gray Wolf', 'Gray Wolf','Gray Wolf', 'Polar Bear','Polar Bear',
	 'Gray Wolf','Polar Bear']

#Classifiers
#decision tree
clf_tree = tree.DecisionTreeClassifier()
#support vector classification
clf_svm = SVC()
#perceptron
clf_perceptron = Perceptron()
#K nearest neighbor
clf_KNN = KNeighborsClassifier()


#Train model for each classifier
clf_tree.fit(X, Y)
clf_svm.fit(X, Y)
clf_perceptron.fit(X, Y)
clf_KNN.fit(X, Y)


#use the data above to make a new prediction given new data
pred_tree = clf_tree.predict([[0.7, 1.6, 34]])
pred_svm = clf_svm.predict([[0.7, 1.6, 34]])
pred_perceptron = clf_perceptron.predict([[0.7, 1.6, 34]])
pred_KNN = clf_KNN.predict([[0.7, 1.6, 34]])

# print results
print pred_tree
print pred_svm
print pred_perceptron
print pred_KNN
