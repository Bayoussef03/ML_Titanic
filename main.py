import streamlit as st
import pandas as pd
import numpy as np 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import joblib



st.write("# Prédiction de survie sur le Titanic")
st.sidebar.header("PARAMETRES D'ENTREE")

model = joblib.load('./models/best_model_joblib.pkl')

def entree_user():
    Pclass = st.sidebar.slider('Classe (Pclass)', 1, 3)
    sex_mapping = {'Homme': 0, 'Femme': 1}
    Sex = st.sidebar.radio('Sexe', ('Homme', 'Femme'))
    Sexe = sex_mapping[Sex]  
    Age = st.sidebar.slider('Âge', 0, 100)
    Fare = st.sidebar.slider('Tarif (Fare)', 0, 600)
    data = {"Pclass": Pclass,"Sex": Sexe,"Age": Age,"Fare": Fare}
    titanic_paramètres = pd.DataFrame(data, index=[0])
    
    return titanic_paramètres

Mydata = entree_user()
st.subheader("On veut trouver l'état de survie d'un passager en fonction des critères suivants")
st.write(Mydata)

csv_path = r'C:\Users\MSI\Desktop\Projet_ML\titanic_data.csv'

Mydata = pd.read_csv(csv_path)
X = Mydata[['Pclass', 'Sex', 'Age', 'Fare']].values  
y = Mydata['Survived'].values

Mydata['Sex'] = Mydata['Sex'].map({'male': 1, 'female': 0})
Mydata['Embarked'] = Mydata['Embarked'].map({'C': 1, 'Q': 2, 'S': 3, np.nan: 1})
Mydata['Embarked'] = Mydata['Embarked'].astype(int)


scaler = preprocessing.StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
Mydata_user_scaled = scaler.transform(Mydata)

selected_features = [0, 1, 2, 3]  
X_train_selected = X_train[:, selected_features]
X_test_selected = X_test[:, selected_features]
Mydata_user_selected = Mydata_user_scaled[:, selected_features]

model = GradientBoostingClassifier(random_state=42)
model.fit(X_train_selected, y_train)

prediction = model.predict(Mydata_user_selected)

st.subheader("L'état de survie du passager est:")

result = 'Survécu' if prediction[0] == 1 else 'Non Survécu'
st.write('Résultat de la prédiction :', result)