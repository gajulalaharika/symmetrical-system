import streamlit as st
import pandas as pd
import time
import numpy as np
import altair as alt
import warnings
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")

# creating containers
header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()
precaution = st.container()
interactive = st.container()
footer = st.container()


st.markdown(
    """
    <style>
    .main{
    background-color:#F3e8e8
    }
    </style>
    """,
    unsafe_allow_html=True
)

img1 = st.sidebar.image('LiverTypes.jpg', caption="Stages of Liver Diseases", use_column_width='auto')
img2 = st.sidebar.image('liver.jpg', caption="Liver Diseases", use_column_width='auto')


with header:
    st.title('Liver Disease Prediction using Machine Learning Algorithms')
    st.text('In this Project we can estimate the percentage of occuring ')
    st.text('Liver disease to a patient according to the blood test reports.')


@st.cache
def get_data():
    dat = pd.read_csv('data/liver_dataset1.csv')
    dat.columns = dat.columns.str.replace(' ', '_')
    dat.Gender_of_the_patient = dat.Gender_of_the_patient.map({'Female': 1, 'Male': 0})
    dat.fillna(dat.Gender_of_the_patient.median(), inplace=True)
    dat.Total_Bilirubin.fillna(dat.Total_Bilirubin.median(), inplace=True)
    dat.Direct_Bilirubin.fillna(dat.Direct_Bilirubin.median(), inplace=True)
    dat.Alkphos_Alkaline_Phosphotase.fillna(dat.Alkphos_Alkaline_Phosphotase.median(), inplace=True)
    dat.Sgpt_Alamine_Aminotransferase.fillna(dat.Sgpt_Alamine_Aminotransferase.median(), inplace=True)
    dat.Sgot_Aspartate_Aminotransferase.fillna(dat.Sgot_Aspartate_Aminotransferase.median(), inplace=True)
    dat.Total_Protiens.fillna(dat.Total_Protiens.median(), inplace=True)
    dat.ALB_Albumin.fillna(dat.ALB_Albumin.median(), inplace=True)
    dat.rename(columns={'A/G_Ratio_Albumin_and_Globulin_Ratio': 'AbyG_Ratio'}, inplace=True)
    dat.AbyG_Ratio.fillna(dat.AbyG_Ratio.median(), inplace=True)
    return dat


with dataset:
    st.header('Liver Patient Datasets')
    st.text('This Dataset is taken from Kaggle with 30 thousand Records.')
    df = get_data()

    st.subheader('Albumin & Globulin Ratio of Patients')
    albumin = pd.DataFrame(df['AbyG_Ratio'].value_counts())
    st.bar_chart(albumin)
    # st.write(df.columns)

    st.write('A plot between Total Proteins and Total Bilirubin')
    c = alt.Chart(df).mark_tick().encode(
        x='Total_Bilirubin:Q',
        y='Total_Protiens:O'
    )
    st.altair_chart(c, use_container_width=True)


with interactive:
    st.header('A Closer Look into the data')
    fig = go.Figure(data=go.Table(
        header=dict(values=list(df[['Total_Bilirubin', 'Alkphos_Alkaline_Phosphotase',
                    'Sgot_Aspartate_Aminotransferase', 'ALB_Albumin']].columns),
                    fill_color='#FDBE72',
                    align="center"),
        cells=dict(values=[df.Total_Bilirubin, df.Alkphos_Alkaline_Phosphotase,
                           df.Sgot_Aspartate_Aminotransferase, df.ALB_Albumin],
                   fill_color='#E5ECF6',
                   align="left")
    ))
    st.write(fig)


with features:
    st.header('Time to train the Model!')
    st.text('Choose the parameters here')
    age = st.slider('Enter your Age ', min_value=0, max_value=100) # Age_of_the_patient
    gender = st.radio('Gender of the Patient ', ["Male", "Female"], index=0) # Gender_of_the_patient
    if gender == "Male":
        gap = 0
    else:
        gap = 1

    bili = st.number_input('Enter the total amount of Bilirubin')# Total_Bilirubin
    dbili = st.number_input('Enter the amount of direct bilirubin')# Direct_Bilirubin
    alkphose = st.slider('Enter the Alkaline Phosphotase from blood report', min_value=60, max_value=3000) # Alkphos_Alkaline_Phosphotase
    sgpt = st.slider('Enter the amount of Alamine Aminotransferase from blood reports', min_value=10, max_value=2000)# Sgpt_Alamine_Aminotransferase
    sgot = st.slider('Enter the Aspartate Aminotransferase quantity', min_value=10, max_value=5000)# Sgot_Aspartate_Aminotransferase
    proteins = st.number_input('Enter the total proteins content in the blood')# Total_Protiens
    albalbum = st.number_input('Enter the amount of Albumin in Blood')# ALB_Albumin
    abyg = st.number_input('Enter the ratio of Albumin and Globulin ration from blood test report')# AbyG_Ratio
    inputs = np.array([[age, bili, dbili, alkphose, sgpt, sgot, proteins, albalbum, abyg]])

    X = df.drop(['Gender_of_the_patient', 'Result'], axis=1)
    Y = df[['Result']]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=99)


with model_training:
    st.header('Here we are Training the model')
    st.text('This section trains the model with the training set of data with Random Forest Classifier....')

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    st.write('Accuracy: ', accuracy_score(y_test, y_pred))

    if st.button('Predict'):
        my_bar = st.progress(0)

        for percent_complete in range(100):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1)
        pred = rf.predict(inputs)
        if pred == 1:
            st.error("You have Liver Disease.")
            st.error("Please consult a doctor and follow precautions carefully.")
        else:
            st.success("You Liver is safe.")

with precaution:
    st.header('Some of the precautions to be followed if you are suffering from Liver disease are')
    st.markdown('* **Avoid Or Limit Alcohol.**')
    sel_col, disp_col = st.columns(2)
    sel_col.markdown('* **Avoiding foods and drinks that contain trans fats or high-fructose corn syrup.**')
    sel_col.markdown('* **Carefully managing your intake of prescription and over-the-counter medications to avoid liver damage, as medications like acetaminophen (Tylenol®) are a common cause of liver injury.**')
    sel_col.markdown('* **Getting regular exercise.**')
    sel_col.markdown('* **Limiting consumption of red meat.**')

    img3 = disp_col.image('Liverimg.jpg', caption="Liver", use_column_width='auto')


with footer:
    contact, about = st.columns(2)
