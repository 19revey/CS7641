import numpy as np
import networkx as nx
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

class alien:
    def __init__(self):
        self.data_size = 0
    def _read_data(self):
        df = pd.read_csv("./src/alien_data.csv") 
        df = df[df["Diameter"]<100]
        # df = df[df["Sex"]!="I"] 	

        self.data_size = len(df)
        return  df
        
    def dataset(self,ratio=0.2,modify_feature=False):
        df=self._read_data()



        X=df.drop(columns=['Sex'],axis=1)
        y = df['Sex']



        num_features = X.select_dtypes(exclude="object").columns
        cat_features = X.select_dtypes(include="object").columns

        num_pipeline= Pipeline(
            steps=[
            ("imputer",SimpleImputer(strategy="median")),
            ("scaler",StandardScaler())
            ]
        )
        cat_pipeline=Pipeline(
            steps=[
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoder",OneHotEncoder()),
            ("scaler",StandardScaler(with_mean=False))
            ]
        )
        preprocessor=ColumnTransformer(
            [
            ("num_pipeline",num_pipeline,num_features),
            ("cat_pipelines",cat_pipeline,cat_features)
            ]
        )

        X = preprocessor.fit_transform(X)

        if modify_feature:
            y = y.map({"M": 0, "F": 0, "I": 1 })
        else:
            y = y.map({"M": 0, "F": 1, "I": 2 })

        # X_sample, _, y_sample, _ = train_test_split(X, y, test_size=ratio, random_state=32)
        # X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=10) 
        
        
        return  X,y


    def train(self,clf=SVC(kernel='linear'),ratio=0.2,modify_feature=False):
        df=self._read_data()



        X=df.drop(columns=['Sex'],axis=1)
        y = df['Sex']



        num_features = X.select_dtypes(exclude="object").columns
        cat_features = X.select_dtypes(include="object").columns

        num_pipeline= Pipeline(
            steps=[
            ("imputer",SimpleImputer(strategy="median")),
            ("scaler",StandardScaler())
            ]
        )
        cat_pipeline=Pipeline(
            steps=[
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoder",OneHotEncoder()),
            ("scaler",StandardScaler(with_mean=False))
            ]
        )
        preprocessor=ColumnTransformer(
            [
            ("num_pipeline",num_pipeline,num_features),
            ("cat_pipelines",cat_pipeline,cat_features)
            ]
        )

        X = preprocessor.fit_transform(X)

        if modify_feature:
            y = y.map({"M": 1, "F": 1, "I": 2 })
        else:
            y = y.map({"M": 0, "F": 1, "I": 2 })

        X_sample, _, y_sample, _ = train_test_split(X, y, test_size=ratio, random_state=32)
        X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=10) 
        
        clf.fit(X_train, y_train)
        return clf, X_test, y_test,X_train, y_train

    def evaluate(self,clf=SVC(kernel='linear'),ratio_range=np.linspace(0.1, 0.9, 4), modify_feature=False):


        accuracy_test=[]
        accuracy_train=[]
        for ratio in ratio_range[::-1]:
            clf, X_test, y_test, X_train,y_train = self.train(clf, ratio,modify_feature)
            y_pred = clf.predict(X_test)
            accuracy_test.append(accuracy_score(y_test, y_pred))
            # accuracy_test.append(cross_val_score(clf,X, y,cv=5,scoring='accuracy').mean())
            y_train_pred = clf.predict(X_train)
            accuracy_train.append(accuracy_score(y_train, y_train_pred))

        df = {
        'data_size': ratio_range*self.data_size,
        'Training Accuracy': accuracy_train,
        'Testing Accuracy': accuracy_test
        }

        return pd.DataFrame(df), clf



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    obj=alien()
    result,clf=obj.evaluate(SVC(kernel='linear'),modify_feature=False)
    # result,clf=obj.evaluate(MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42))
    print(result)
    # plt.plot(clf.loss_curve_)
    # plt.savefig("train.png")

    # obj.evaluate(KNeighborsClassifier(n_neighbors=3))
    # obj.evaluate(MLPClassifier(hidden_layer_sizes=(20,20,10), activation='relu', solver='adam', max_iter=1000, random_state=19))
