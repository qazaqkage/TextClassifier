from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import pandas as pd
import pickle

df = pd.read_csv('C:/Users/LEGION/PycharmProjects/TextClassifier/files/article-dataset.csv')

X = df[['text_len', 'punctuation%', 'unique_values']]
Y = df['label_encoded']

x_train, x_test, y_train, y_test = train_test_split(X, Y)

dtc = DecisionTreeClassifier(criterion='entropy', splitter='random', min_samples_leaf=6, min_samples_split=6, random_state=0)
knn = KNeighborsClassifier(n_neighbors=3)
svm = SVC(C=10, gamma=10)
modelNN = MLPClassifier(
    activation='relu',
    solver='adam',
    alpha=0.001,
    beta_1=0.9,
    beta_2=0.999,
    batch_size=256,
    epsilon=1e-08,
    hidden_layer_sizes=(300,),
    learning_rate='adaptive',
    max_iter=1000, # I've found for this task, loss converges at ~1000 iterations
    random_state=2,
)

dtc = dtc.fit(x_train, y_train)
knn = knn.fit(x_train, y_train)
svm = svm.fit(x_train, y_train)
modelNN = modelNN.fit(x_train, y_train)

pickle.dump(dtc, open('dtc_model.pkl', 'wb'))
pickle.dump(knn, open('knn_model.pkl', 'wb'))
pickle.dump(svm, open('svm_model.pkl', 'wb'))
pickle.dump(modelNN, open('MLPnn_model.pkl', 'wb'))
