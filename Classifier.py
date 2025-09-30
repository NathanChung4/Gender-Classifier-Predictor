from sklearn import tree
import random

# [height, weight, shoesize]
x = [[181,142,51], [154,127,39], [187,144,50], [161,122,34], [188,146,49], [118,129,35], [191,141,48], [122,126,39], [200,146,55], [171,124,32], [199,146,44]]

y = ['male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male']

random_height = random.randint(150,200)
random_weight = random.randint(110,160)
random_shoesize = random.randint(20,60)

clf = tree.DecisionTreeClassifier()

clf = clf.fit(x,y)

prediction = clf.predict([[random_height,random_weight,random_shoesize]])

print(prediction[0])