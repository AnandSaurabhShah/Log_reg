import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
digits=load_digits()
print(dir(digits))

plt.gray()
plt.matshow(digits.images[1])
plt.show()

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(digits.data,digits.target,test_size=0.3,random_state=3)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,Y_train)
print(model.score(X_test,Y_test))
Y=model.predict(X_test)

import seaborn as sn
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y)
print(cm)
plt.figure(figsize= (10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()


#Excersize 2

