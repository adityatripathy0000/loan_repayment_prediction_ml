#Import_Library
import numpy as np
import pandas as pd
import seaborn as sns

#Import_Dataset
dataset = pd.read_csv('loan_train.csv')
X = dataset.iloc[:,1:12].values
y = dataset.iloc[:,-1].values

#Handle_Missing_Data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan , strategy = "mean" , verbose = 0)
imputer1 = SimpleImputer(missing_values = np.nan , strategy = "most_frequent" , verbose = 0)

imputer = imputer.fit(X[:,7:9])
X[:,7:9] = imputer.transform(X[:,7:9])

imputer1 = imputer1.fit(X[:,9].reshape(-1,1))
X[:,9] = np.squeeze(imputer1.transform(X[:,9].reshape(-1,1)))

imputer1 = imputer1.fit(X[:,0].reshape(-1,1))
X[:,0] = np.squeeze(imputer1.transform(X[:,0].reshape(-1,1)))

imputer1 = imputer1.fit(X[:,1].reshape(-1,1))
X[:,1] = np.squeeze(imputer1.transform(X[:,1].reshape(-1,1)))

imputer1 = imputer1.fit(X[:,2].reshape(-1,1))
X[:,2] = np.squeeze(imputer1.transform(X[:,2].reshape(-1,1)))

imputer1 = imputer1.fit(X[:,3].reshape(-1,1))
X[:,3] = np.squeeze(imputer1.transform(X[:,3].reshape(-1,1)))

imputer1 = imputer1.fit(X[:,4].reshape(-1,1))
X[:,4] = np.squeeze(imputer1.transform(X[:,4].reshape(-1,1)))

imputer1 = imputer1.fit(X[:,10].reshape(-1,1))
X[:,10] = np.squeeze(imputer1.transform(X[:,10].reshape(-1,1)))

#Handle_Categorical_Data
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
X[:,1] = labelencoder_X.fit_transform(X[:,1])
X[:,2] = labelencoder_X.fit_transform(X[:,2])
X[:,3] = labelencoder_X.fit_transform(X[:,3])
X[:,4] = labelencoder_X.fit_transform(X[:,4])
X[:,10] = labelencoder_X.fit_transform(X[:,10])
ct = ColumnTransformer([('encoder',OneHotEncoder(),[0,1,2,3,4,10])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X), dtype = np.float)

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#Splitting_Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state = 0)

#Scaling_Dataset
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Fitting logistic Regression to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

#predicting_sample_test
y_pred = classifier.predict(X_test)

#confusion_matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#plotting_the_heatmap_of_confusion_matrix
sns.heatmap(cm, annot = True, cmap="YlGnBu", fmt="d")

#accuracy
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print(acc*100)

#Import_unseen_Test_Dataset
dataset_test = pd.read_csv('loan_test.csv')
X_test_1 = dataset_test.iloc[:,1:].values

#Handle_Missing_unseen_Test_Data
imputer = imputer.fit(X_test_1[:,7:9])
X_test_1[:,7:9] = imputer.transform(X_test_1[:,7:9])

imputer1 = imputer1.fit(X_test_1[:,9].reshape(-1,1))
X_test_1[:,9] = np.squeeze(imputer1.transform(X_test_1[:,9].reshape(-1,1)))

imputer1 = imputer1.fit(X_test_1[:,0].reshape(-1,1))
X_test_1[:,0] = np.squeeze(imputer1.transform(X_test_1[:,0].reshape(-1,1)))

imputer1 = imputer1.fit(X_test_1[:,1].reshape(-1,1))
X_test_1[:,1] = np.squeeze(imputer1.transform(X_test_1[:,1].reshape(-1,1)))

imputer1 = imputer1.fit(X_test_1[:,2].reshape(-1,1))
X_test_1[:,2] = np.squeeze(imputer1.transform(X_test_1[:,2].reshape(-1,1)))

imputer1 = imputer1.fit(X_test_1[:,3].reshape(-1,1))
X_test_1[:,3] = np.squeeze(imputer1.transform(X_test_1[:,3].reshape(-1,1)))

imputer1 = imputer1.fit(X_test_1[:,4].reshape(-1,1))
X_test_1[:,4] = np.squeeze(imputer1.transform(X_test_1[:,4].reshape(-1,1)))

imputer1 = imputer1.fit(X_test_1[:,10].reshape(-1,1))
X_test_1[:,10] = np.squeeze(imputer1.transform(X_test_1[:,10].reshape(-1,1)))

#Handle_Categorical_unseen_Test_Data
labelencoder_X_test = LabelEncoder()
X_test_1[:,0] = labelencoder_X_test.fit_transform(X_test_1[:,0])
X_test_1[:,1] = labelencoder_X_test.fit_transform(X_test_1[:,1])
X_test_1[:,2] = labelencoder_X_test.fit_transform(X_test_1[:,2])
X_test_1[:,3] = labelencoder_X_test.fit_transform(X_test_1[:,3])
X_test_1[:,4] = labelencoder_X_test.fit_transform(X_test_1[:,4])
X_test_1[:,10] = labelencoder_X_test.fit_transform(X_test_1[:,10])
ct = ColumnTransformer([('encoder',OneHotEncoder(),[0,1,2,3,4,10])], remainder = 'passthrough')
X_test_1 = np.array(ct.fit_transform(X_test_1), dtype = np.float)

#Scaling_the_unseen_test_data
X_test_1 = sc_X.transform(X_test_1)

#predicting_unseen_test_data
y_pred_1 = classifier.predict(X_test_1)

#convert_the_prediction
dataset_test['Loan_Status'] = y_pred_1
dataset_test_final = dataset_test.drop(['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area'], axis=1)
dataset_test_final['Loan_Status'] = dataset_test_final['Loan_Status'].map({0:'N', 1:'Y'})
dataset_test_final.to_csv('Final_prediction.csv', index=False)