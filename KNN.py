import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, confusion_matrix, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import sklearn.metrics as metrics
import time


# loading dataset to the environment
def load_dataset(file_path):
    return pd.read_csv(file_path, sep=',')

# here we preprocess the dataset, cleaning up/modifying it to better fit the environment
def preprocess_data(data):
    X = data.drop(['Characteristics', 'SizeOfCode', 'benign'], axis=1).values
    y = data['benign'].values

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X = imputer.fit_transform(X)

    return X, y


# splitting the dataset into training/testing
def split_data(X, y, test_size=0.20, random_state=0):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# transforming data
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


# instantiating KNN classifier
def train_knn_classifier(X_train, y_train, n_neighbors=8):
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors, metric='minkowski', p=2)
    classifier.fit(X_train, y_train)
    return classifier


# executing KNN classifier with training/test sets
def evaluate_classifier(classifier, X_test, y_test):
    y_prediction = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_prediction)
    return cm


# cross-validating or creating a cross-matrix for KNN to verify benign and malicious
def cross_validate_knn(X_train, y_train, neighbors_range=(1, 50), cv=20):
    neighbors = list(range(neighbors_range[0], neighbors_range[1] + 1))
    cv_scores = []

    for k in neighbors:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring='accuracy')
        score_mean = scores.mean()
        cv_scores.append(score_mean)

    MSE = [1 - x for x in cv_scores]
    return np.array(MSE), np.array(neighbors)


# plotting classification errors
def plot_classification_error(neighbors_list, MSE_list):
    plt.plot(neighbors_list, MSE_list, color='green')
    plt.xlabel('Number of Neighbors K')
    plt.ylabel('Accuracy')
    plt.show()


# plot AUC curve
def plot_roc_curve(y_test, prediction):
    fpr, tpr, threshold = roc_curve(y_test, prediction)
    roc_auc = auc(fpr, tpr)

    plt.title('Receiver Operating Characteristics')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.show()
    plt.savefig('AUC')


# main call
if __name__ == "__main__":
    start = time.time()

    # Load dataset
    dataset = load_dataset('ClaMP_Raw-5184.csv')

    # Preprocess data
    X, y = preprocess_data(dataset)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Scale data
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # Train KNN classifier
    knn_classifier = train_knn_classifier(X_train_scaled, y_train)

    # Evaluate classifier
    confusion_matrix_result = evaluate_classifier(knn_classifier, X_test_scaled, y_test)

    # Cross-validate KNN
    MSE_list, neighbors_list = cross_validate_knn(X_train_scaled, y_train)

    # Plot classification error vs k
    plot_classification_error(neighbors_list, MSE_list)

    # ROC Curve
    probs = knn_classifier.predict_proba(X_test_scaled)
    prediction = probs[:, 1]
    plot_roc_curve(y_test, prediction)

    end = time.time()
    print(end - start)