form sklearn.datasets import load_digits 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 

digits = load_digits()
X= digits.data

y=digits.target

model= RandomForestClassifier()
model.fit(X,y)
