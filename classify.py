from sklearn import tree

clf = tree.DecisionTreeClassifier()

#[height(m), length(m), weight(kg)]
X = [[1,2.1,433], [0.7,1.7,50], [0.8,1.6,34], 
     [1.5,2.4,521], [1.4,2.4,359], [0.8,1.5,43], 
     [0.8,1.5,44], [0.5,1.4,44], [1.4,2.4,543], 
     [1.5,2.2,487], [0.7,1.7,45],[1.4,2.5,458]]

#label above data as either 'polar bear' or 'gray wolf'
Y = ['Polar Bear', 'Gray Wolf', 'Gray Wolf', 'Polar Bear','Polar Bear',
	 'Gray Wolf', 'Gray Wolf','Gray Wolf', 'Polar Bear','Polar Bear',
	 'Gray Wolf','Polar Bear']

clf = clf.fit(X, Y)

#use the data above to make a new prediction given new data
prediction = clf.predict([[0.7,1.6,34]])

print prediction
