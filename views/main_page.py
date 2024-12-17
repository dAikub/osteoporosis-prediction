import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score , train_test_split
import warnings


def reload_df():
    file_path = 'osteoporosis.csv'
    file_path2 = 'predicted.csv'
    df2 = pd.read_csv(file_path)
    predict_df = pd.read_csv(file_path2)
    return file_path,file_path2,df2,predict_df

file_path,file_path2,df2,predict_df = reload_df()
df = df2.drop(['Alcohol Consumption','Medications','Body Weight','Id','Age'], axis=1)
raw_df2 = df2
raw_df3 = predict_df
st.markdown("""
    <style>
        .stButton>button {
            background-color: #FA6969;  /* เปลี่ยนสีปุ่ม */
            color: white;               /* เปลี่ยนสีข้อความในปุ่ม */
            font-size: 16px;            /* ขนาดฟอนต์ */
            padding: 10px 20px;         /* ขนาดปุ่ม */
            border-radius: 20px;         /* มุมโค้ง */
            width: 100%;                /* กำหนดความกว้างของปุ่ม */
            cursor: pointer;           /* เปลี่ยนลักษณะเมาส์เป็น pointer เมื่อเอาเมาส์ไปวาง */
            transition: all 0.3s ease;  /* เพิ่ม transition ให้การเปลี่ยนแปลง */
        }
        .stButton>button:hover {
            background-color: #C3E8A6;  /* สีเมื่อเอาเมาส์ไปวาง */
            color: white;              /* เปลี่ยนสีฟอนต์เมื่อเอาเมาส์ไปวาง */
            border: 2px solid #C3E8A6;  /* ขอบสีทองเมื่อเอาเมาส์ไปวาง */
        }
        .stButton>button:active {
            background-color: #FF1493;  /* สีพื้นหลังเมื่อกดปุ่ม */
            color: white;               /* สีข้อความเมื่อกดปุ่ม */
            border: 2px solid #8B0000;  /* ขอบสีแดงเข้มเมื่อกดปุ่ม */
        }
        /* จัดตำแหน่งปุ่มตรงกลาง */
        .stButton {
            display: flex;
            justify-content: center;
    
        }
    </style>
    """, unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .stRadio > div {
        display: flex;
        gap: 20px; /* ปรับระยะห่างที่นี่ */
    }
    </style>
    """,
    unsafe_allow_html=True
)


def display_choice(df):
    #ส่วนหน้ากรอก
    st.title("Osteoporosis Risk Prediction")
    st.subheader("กรุณากรอกข้อมูลเพื่อพยากรณ์โรคกระดูกพรุน")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")

    
    new_row = []
    new_row.append(st.number_input("ID", step=1,max_value=9999999,min_value=1000000))
    st.markdown('---')
    new_row.append(st.number_input("Age",min_value=1, max_value=100, step=1))
    st.markdown('---')

    df = df.drop(['Id','Age','Alcohol Consumption','Medications','Osteoporosis','Body Weight'], axis=1)



    #ลูปเรียก Radio Button
    for col in df.columns:
        if col == "Medical Conditions":
            unique_values = df[col].dropna().unique()
            radio_btn = st.radio(col,options=("Rheumatoid Arthritis","Hyperthyroidism","No"),horizontal=True,index=None)
            new_row.append(radio_btn)
            st.info('Any existing medical conditions that the individual may have. This can include conditions like Rheumatoid')
            st.markdown('---')
        else:
            unique_values = df[col].dropna().unique()
            radio_btn = st.radio(col,options=(unique_values),horizontal=True,index=None)
            new_row.append(radio_btn)
            if col == "Age":
                st.info("The age of the individual in years")
            if col == "Gender":
                st.info('The gender of the individual. This can be either "Male" or "Female".')
            if col == "Hormonal Changes":
                st.info("Indicates whether the individual has undergone hormonal changes, particularly related to menopause. This can be")
            if col == "Family History":
                st.info('Indicates whether there is a family history of osteoporosis or fractures. This can be "Yes" or "No".')
            if col == "Race/Ethnicity":
                st.info('The race or ethnicity of the individual. This can include categories such as "Caucasian", "African American", "Asian", etc.')
            if col == "Calcium Intake":
                st.info('The level of calcium intake in the individuals diet. This can be "Low" or "Adequate".')
            if col == "Vitamin D Intake":
                st.info('The level of vitamin D intake in the individuals diet. This can be "Insufficient" or "Sufficient".')
            if col == "Physical Activity":
                st.info('Indicates the level of physical activity of the individual. This can be "Sedentary" for low activity levels or "Active"')
            if col == "Smoking":
                st.info('Indicates whether the individual is a smoker. This can be "Yes" or "No".')
            if col == "Prior Fractures":
                st.info('Indicates whether the individual has previously experienced fractures. This can be "Yes" or "No".')

            st.markdown('---')

    save = st.radio("Do you allow the website to collect your predictions?",options=("Yes","No"),horizontal=True,index=None)
    st.markdown('---')





    data_input = {'Id' : [new_row[0]],
                    'Age' : [new_row[1]], 
                    'Gender' : [new_row[2]],
                    'Hormonal Changes' : [new_row[3]],
                    'Family History' : [new_row[4]],
                    'Race/Ethnicity' : [new_row[5]],
                    'Calcium Intake' : [new_row[6]],
                    'Vitamin D Intake' : [new_row[7]],
                    'Physical Activity' : [new_row[8]],
                    'Smoking' : [new_row[9]],
                    'Medical Conditions' : [new_row[10]],
                    'Prior Fractures' : [new_row[11]],
                    }

    if all(value is not None for value in new_row):
        user_enter = st.button("Enter")
        if user_enter:
            display_result(data_input,str(save))



def display_report(result,df_add_row,save):
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    if result == 0:
        st.image("https://i.postimg.cc/Px3Y74PY/0.png")
    else:
        st.image("https://i.postimg.cc/d1PVVZpp/1.png")

    if save == "Yes":
        st.success("บันทึกข้อมูลของคุณเรียบร้อยแล้ว")
        df_add_row.to_csv(file_path2, index=False)
    else:
        st.info("ข้อมูลของคุณไม่ถูกบันทึก")

def display_result(data_input,save):
    dataframe_input = pd.DataFrame(data_input)
    df_combined = pd.concat([raw_df2, dataframe_input], ignore_index=True)

    def random_impute(series):
        not_nan_values = series.dropna()
        if not_nan_values.empty:
            return series
        return series.apply(lambda x: np.random.choice(not_nan_values) if pd.isna(x) else x)

    df_combined = df_combined.apply(random_impute)
    df_combined_rename = rename_columns(df_combined)
    df_combined_encoder = Label_Encoder(df_combined_rename)


    df_combined_normalized = pd.DataFrame(scaler.fit_transform(df_combined_encoder), columns=df_combined_encoder.columns)
    df_combined_normalized = df_combined_normalized.drop(['Alcohol_Consumption','Medications','Body_Weight','Id'], axis=1)
    input_data = df_combined_normalized.tail(1).drop('Osteoporosis',axis=1)


    result = gb.predict(input_data)

    result_int = int(result[0])
    dataframe_input_predicted = dataframe_input.copy()
    dataframe_input_predicted['Osteoporosis'] = result_int
    dataframe_input_predicted.head()
    df_add_row = pd.concat([raw_df3, dataframe_input_predicted], ignore_index=True)




    
    display_report(int(result),df_add_row,save)








def random_impute(series):
    not_nan_values = series.dropna()
    if not_nan_values.empty:
        return series
    return series.apply(lambda x: np.random.choice(not_nan_values) if pd.isna(x) else x)

# เติมค่า NaN ด้วยการสุ่มจากค่าที่มีอยู่
df2 = df2.apply(random_impute)

# เปลี่ยนชื่อคอลัมน์
def rename_columns(df):
    df = df.rename(columns={
    "Hormonal Changes": "Hormonal_Changes",
    "Family History" : "Family_History",
    "Body Weight" : "Body_Weight",
    "Calcium Intake" : "Calcium_Intake",
    "Vitamin D Intake" : "Vitamin_D_Intake",
    "Physical Activity" : "Physical_Activity",
    "Alcohol Consumption" : "Alcohol_Consumption",
    "Medical Conditions" : "Medical_Conditions",
    "Prior Fractures" : "Prior_Fractures",
    })
    return df


def Label_Encoder(df):
    # แปลงข้อมูลหมวดหมู่ให้เป็นตัวเลข
    for col in df.columns:
      if df[col].dtype == 'object':
         le = LabelEncoder()
         df[col] = le.fit_transform(df[col])
    return df

df_rename = rename_columns(df2)
df_encoder = Label_Encoder(df_rename)



scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df_encoder), columns=df_encoder.columns)
df_normalized = df_normalized.drop(['Alcohol_Consumption','Medications','Body_Weight','Id'], axis=1)


# แยก Target เป็น โรคกระดูกพรุน(Osteoporosis)
x = df_normalized.drop("Osteoporosis",axis=1)
y = df_normalized["Osteoporosis"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)



gb = GradientBoostingClassifier(
    learning_rate=0.1,
    max_depth=3,
    min_samples_split=20,
    n_estimators=50,
    subsample=0.8
)

gb_model = gb
gb.fit(x_train,y_train)
gb_pred = gb.predict(x_test)


display_choice(df2)

















plt.figure(figsize=(10, 6))
sns.countplot(data=df2, x="Osteoporosis", hue="Race/Ethnicity")
plt.title("Count of Individuals with and without Osteoporosis by Race/Ethnicity")
#st.pyplot(plt.gcf())


