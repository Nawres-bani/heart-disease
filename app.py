#pip install streamlit
#pip install pandas
#pip install sklearn


# IMPORT STATEMENTS
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




df = pd.read_csv("C:/Users/ASUS/Desktop/heart disease/heart.csv")

# loading the saved models
loaded_model = pickle.load(open('C:/Users/ASUS/Desktop/heart disease/heart_disease_model.sav', 'rb'))



# HEADINGS
st.title('heart  Checkup')
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(df.describe())


# X AND Y DATA
x = df.drop(['target'], axis = 1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

 
# FUNCTION
def user_report():
  age= st.sidebar.slider('age', 0,29, 77 )
  sex = st.sidebar.slider ('sex',0, 0,1)
  cp = st.sidebar.slider('cp', 0,0, 3 )
  trestbps = st.sidebar.slider('trestbps', 0,94, 200)
  chol= st.sidebar.slider('chol', 0,126, 564)
  fbs = st.sidebar.slider('fbs', 0,0, 1 )
  restecg = st.sidebar.slider('restecg', 0,0,2 )
  thalach = st.sidebar.slider('thalach', 0,71, 202 )
  exang = st.sidebar.slider('exang', 0,0, 1 )
  oldpeak= st.sidebar.slider('oldpeak', 0,0, 6 )
  slope= st.sidebar.slider('slope', 0,0, 2 )
  ca = st.sidebar.slider('ca', 21,88, 33 )
  thal = st.sidebar.slider('thal', 21,88, 33 )


  user_report_data = {
      ' age': age,
      'sex':sex,
      'cp':cp,
      'trestbps':trestbps,
      'chol': chol,
      'fbs':fbs,
      'restecg':restecg,
      'thalach':thalach,
      'exang':exang,
      'oldpeak':oldpeak,
      'slope':slope,
      'ca':ca,
      'thal':thal
     

  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data


# IMAGE
image_path = "C:/Users/ASUS/Desktop/heart disease/b.jpg"
image = Image.open(image_path)
st.sidebar.image(image, caption='Légende de l\'image', use_column_width=True)



# PATIENT DATA
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)


# VISUALISATIONS
st.title('Visualised Patient Report')


# Calculate the correlation matrix
corr_matrix = heart_data.corr()

# Create a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


