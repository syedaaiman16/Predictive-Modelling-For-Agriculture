# All required libraries are imported here for you.
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import f1_score

# Load the dataset
crops = pd.read_csv("soil_measures.csv")

# Write your code here
crops.head()

crops.info()

crops.describe()

crops.isna().sum()
crops["crop"].value_counts()

# x=crops[["N", "P", "K", "ph"]]
x = crops.drop("crop", axis=1)
y= crops["crop"]
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# Define logistic regression model
model = LogisticRegression(max_iter=2000, multi_class='multinomial')

# Iterate over features
for f in x:
    # Fit model using only the current feature
    model.fit(X_train[[f]], y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test[[f]])
    
    # Calculate F1-score for the current feature
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Print F1-score
    print(f"F1-score for {f}: {f1}")


# In order to avoid selecting two features that are highly correlated, perform a correlation analysis for each pair of features, enabling you to build a final model without the presence of multicollinearity.
sns.heatmap(crops.corr(),annot=True)
plt.show()


crops_dummy=pd.get_dummies(crops['crop'],drop_first=True)
crops_dummy.info()

crops_dummy = pd.concat([crops, crops_dummy], axis=1) 
crops_dummy = crops_dummy.drop("crop", axis=1)
crops_dummy


final_features =['N','K','ph']
X_train, X_test, y_train, y_test = train_test_split(crops[final_features],crops["crop"],test_size=0.2,random_state=42)
log_reg = LogisticRegression(max_iter=2000,multi_class='multinomial')
log_reg.fit(X_train,y_train)
y_pred = log_reg.predict(X_test)
f_error = f1_score(y_test,y_pred,average='weighted')
model_performance = f1_score(y_test, y_pred, average="weighted")
print(model_performance)