import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf

st.title('Lung Cancer Risk Prediction')
st.markdown('- Python libraries: Tensorflow, numpy, pandas, streamlit, matplotlib, seaborn')
st.markdown('- Data source: https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link')

df = pd.read_csv('cancer patient data sets.csv')

tab1, tab2 = st.tabs(["Dataset Table", "Prediction"])

with tab1:
    st.write('Lung cancer is the leading cause of cancer death worldwide, accounting for 1.59 million deaths in 2018. The majority of lung cancer cases are attributed to smoking, but exposure to air pollution is also a risk factor. A new study has found that air pollution may be linked to an increased risk of lung cancer, even in nonsmokers.\n\nThe study, which was published in the journal Nature Medicine, looked at data from over 462,000 people in China who were followed for an average of six years. The participants were divided into two groups: those who lived in areas with high levels of air pollution and those who lived in areas with low levels of air pollution.\n\nThe researchers found that the people in the high-pollution group were more likely to develop lung cancer than those in the low-pollution group. They also found that the risk was higher in nonsmokers than smokers, and that the risk increased with age.\n\nWhile this study does not prove that air pollution causes lung cancer, it does suggest that there may be a link between the two. More research is needed to confirm these findings and to determine what effect different types and levels of air pollution may have on lung cancer risk')
    
    st.table(data=df[:10])
    
    st.subheader('Total of Lung Cancer Patients Based on Age')
    fig, ax = plt.subplots(figsize=(25,7))
    sns.countplot(x="Age", data=df)
    st.pyplot(fig)
    
    st.subheader('Total of Lung Cancer Patients Based on Gender')
    Male = []
    Female = []
    label = df['Level'].unique()

    for j in label:
        Male.append(df[(df['Level'] == j) & (df['Gender'] == 1)]['Level'].count())
        Female.append(df[(df['Level'] == j) & (df['Gender'] == 2)]['Level'].count())

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,6))
    ax[0].pie(x=Male, autopct='%1.1f%%')
    ax[0].set_title('Male cancer level')
    ax[1].pie(x=Female, autopct='%1.1f%%')
    ax[1].set_title('Female cancer level')
    ax[0].legend(label,loc="upper left",fontsize=15)
    st.pyplot(fig)
    
    st.subheader('Heatmap for all columns')
    fig, ax = plt.subplots(figsize=(25, 10))
    corr = df.apply(lambda x: pd.factorize(x)[0]).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, linewidths=.2, cmap='coolwarm', vmin=-1, vmax=1)
    st.pyplot(fig)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input(label = "Age",
                                 min_value=14,
                                 max_value=90,
                                 step=1,
                                 placeholder="Place your age...")
        
        height = st.number_input(label = "Height",
                                 min_value=100.0,
                                 max_value=250.0,
                                 step=0.1,
                                 placeholder="Place your height...")
        
        weight = st.number_input(label = "Weight",
                                 min_value=30.0,
                                 max_value=300.0,
                                 step=0.1,
                                 placeholder="Place your Weight...")
        
        smoking = st.selectbox("Amount of cigarettes",
                               ("Not smoking","less than 1 per week", "1-2 per week", "1-5 per day", "6-10 per day", "11-20 per day", "21-30 per day", "more than 30 per day"),
                                key=(1))
        
        passive_smoking = st.selectbox("Amount of passive smoke exposure you experience",
                              ("No Exposure", "Rare Exposure", "Occasional Exposure", "Low Exposure", "Moderate Exposure", "High Exposure", "Very High Exposure", "Extreme Exposure"),
                              key=(2))
        
        air_pollution = st.selectbox("Where do you live",
                               ("Rural, little industrial or vehicular activity", "Suburban, little industrial activity", "Small town, some pollution", "Urban, vehicular and some industries", "Large city, significant vehicles and some industries", "Large city, high vehicles and many industries", "Industrial, many factories and heavy vehicles", "Heavy pollution from various sources"),
                                key=(3))
        
        occupation = st.selectbox("Occupation",
                               ("Office Worker", "Retail Worker", "Teacher/Professor", "Public Transport Worker", "Doctor/Healthcare Worker", "Construction Worker", "Manufacturing Industry Worker", "Mining Worker"),
                                key=(4))
        
    with col2: 
        genetic_risk = st.selectbox("Genetic risk in your family",
                               ("No family history", "Lung cancer in distant relatives", "Lung cancer in one close relative", "Lung cancer in one middle-aged close relative", "Lung cancer in multiple close relatives or one young close relative", "Lung cancer in several middle-aged close relatives or multiple young close relatives", "Lung cancer in many close relatives, including young ones"),
                                key=(5))
               
        coughing_of_blood = st.selectbox("Coughing of Blood (Batuk Darah):",
                                          ["None",
                                           "Very mild or rare",
                                           "Mild, few times/month",
                                           "Moderate, few times/week",
                                           "Frequent, may need attention",
                                           "Frequent, may indicate serious issue",
                                           "Severe, needs immediate attention",
                                           "Very severe, critical condition",
                                           "Life-threatening"],
                                          key=6)
        
        shortness_of_breath= st.selectbox("Shortness of Breath (Sesak Napas):",
                                            ["None",
                                             "During strenuous activity",
                                             "During light activity",
                                             "Periodic daily activities",
                                             "Frequent, may need attention",
                                             "May need care",
                                             "Severe, needs immediate attention",
                                             "Life-threatening, needs immediate attention",
                                             "May need assistance"],
                                            key=7)

        chest_pain = st.selectbox("Chest Pain (Nyeri Dada):",
                                   ["None",
                                    "Mild occasional pain",
                                    "Mild pain several times/month",
                                    "Moderate, several times/week",
                                    "Frequent, may need attention",
                                    "May indicate issue",
                                    "Severe, needs immediate attention",
                                    "Life-threatening, needs immediate attention",
                                    "Needs immediate attention"],
                                   key=8)


        weight_loss = st.selectbox("Weight Loss Category",
                                    ["< 2 kg, several weeks/months",
                                     "2-4 kg, several weeks/months",
                                     "3-5 kg, several weeks/months",
                                     "5-7 kg, several weeks/months",
                                     "7-10 kg, several months",
                                     "> 10-15 kg, several months",
                                     "> 15-20 kg, several months",
                                     "> 20 kg, several months"])

        clubbing_of_finger_nails = st.selectbox("Clubbing of Finger Nails (Penebalan Ujung Jari):",
                                                 ["None",
                                                  "Mild, possibly due to other factors",
                                                  "Mild, not significant",
                                                  "Moderate, within normal limits",
                                                  "Requires attention",
                                                  "Suspicious, may indicate issue",
                                                  "Requires medical attention",
                                                  "Needs immediate medical attention",
                                                  "Life-threatening"],
                                                 key=10)

        

    def bmi(height, weight):
        height = height/100
        bmi = float(weight / (height * height))
        if bmi < 16.0:
            bmi2 = 1
        elif bmi >=16.0 and bmi < 17.0:
            bmi2 = 2
        elif bmi >= 17.0 and bmi < 18.5:
            bmi2 = 3
        elif bmi >= 18.5 and bmi < 25.0:
            bmi2 = 4
        elif bmi >= 25.0 and bmi < 30.0:
            bmi2 = 5
        elif bmi >= 30.0 and bmi < 35.0:
            bmi2 = 6
        else:
            bmi2 = 7
        
        return bmi2

    def smoking_status(selected):
        if selected == 'Not smoking':
            return 1
        elif selected == "less than 1 per week":
            return 2
        elif selected == "1-2 per week":
            return 3
        elif selected == "1-5 per day":
            return 4
        elif selected == "6-10 per day":
            return 5
        elif selected == "11-20 per day":
            return 6
        elif selected == "21-30 per day":
            return 7
        else:
            return 8

    def passive_smoking_status(selected):
        if selected == "No Exposure":
            return 1
        elif selected == "Rare Exposure":
            return 2
        elif selected == "Occasional Exposure":
            return 3
        elif selected == "Low Exposure":
            return 4
        elif selected == "Moderate Exposure":
            return 5
        elif selected == "High Exposure":
            return 6
        elif selected == "Very High Exposure":
            return 7
        else:
            return 8
        
    def pollution(selected):
        if selected == "Rural, little industrial or vehicular activity":
            return 1
        elif selected == "Suburban, little industrial activity":
            return 2
        elif selected == "Small town, some pollution":
            return 3
        elif selected == "Urban, vehicular and some industries":
            return 4
        elif selected == "Large city, significant vehicles and some industries":
            return 5
        elif selected == "Large city, high vehicles and many industries":
            return 6
        elif selected == "Industrial, many factories and heavy vehicles":
            return 7
        else:
            return 8

    def occupation_status(selected):
        if selected == "Office Worker":
            return 1
        elif selected == "Retail Worker":
            return 2
        elif selected == "Teacher/Professor":
            return 3
        elif selected == "Public Transport Worker":
            return 4
        elif selected == "Doctor/Healthcare Worker":
            return 5
        elif selected == "Construction Worker":
            return 6
        elif selected == "Manufacturing Industry Worker":
            return 7
        else:
            return 8
        
    def genetic_risk_status(selected):
        if selected == "No family history":
            return 1
        elif selected == "Lung cancer in distant relatives":
            return 2
        elif selected == "Lung cancer in one close relative":
            return 3
        elif selected == "Lung cancer in one middle-aged close relative":
            return 4
        elif selected == "Lung cancer in multiple close relatives or one young close relative":
            return 5
        elif selected == "Lung cancer in several middle-aged close relatives or multiple young close relatives":
            return 6
        else:
            return 7
        
    def get_category_value(category):
        if category == "None":
            return 1
        elif category == "Only during strenuous activity":
            return 2
        elif category == "During light activity":
            return 3
        elif category == "Periodic daily activities":
            return 4
        elif category == "Frequent, may need attention":
            return 5
        elif category == "During light activity, may need care":
            return 6
        elif category == "Frequent and severe, needs immediate attention":
            return 7
        elif category == "Severe, life-threatening, needs immediate attention":
            return 8
        else:
            return 9
        
    def categorize_coughing_of_blood(category):
        if category == "None":
            return 0
        elif category == "Very mild or rare":
            return 1
        elif category == "Mild, few times/month":
            return 2
        elif category == "Moderate, few times/week":
            return 3
        elif category == "Frequent, may need attention":
            return 4
        elif category == "Frequent, may indicate serious issue":
            return 5
        elif category == "Severe, needs immediate attention":
            return 6
        elif category == "Very severe, critical condition":
            return 7
        elif category == "Life-threatening":
            return 8

    def categorize_shortness_of_breath(category):
        if category == "None":
            return 0
        elif category == "During strenuous activity":
            return 1
        elif category == "During light activity":
            return 2
        elif category == "Periodic daily activities":
            return 3
        elif category == "Frequent, may need attention":
            return 4
        elif category == "May need care":
            return 5
        elif category == "Severe, needs immediate attention":
            return 6
        elif category == "Life-threatening, needs immediate attention":
            return 7
        elif category == "May need assistance":
            return 8

    def categorize_chest_pain(category):
        if category == "None":
            return 0
        elif category == "Mild occasional pain":
            return 1
        elif category == "Mild pain several times/month":
            return 2
        elif category == "Moderate, several times/week":
            return 3
        elif category == "Frequent, may need attention":
            return 4
        elif category == "May indicate issue":
            return 5
        elif category == "Severe, needs immediate attention":
            return 6
        elif category == "Life-threatening, needs immediate attention":
            return 7
        elif category == "Needs immediate attention":
            return 8

    def categorize_weight_loss(category):
        if "< 2 kg" in category:
            return 1
        elif "2-4 kg" in category:
            return 2
        elif "3-5 kg" in category:
            return 3
        elif "5-7 kg" in category:
            return 4
        elif "7-10 kg" in category:
            return 5
        elif "> 10-15 kg" in category:
            return 6
        elif "> 15-20 kg" in category:
            return 7
        elif "> 20 kg" in category:
            return 8

    def categorize_clubbing_of_finger_nails(category):
        if category == "None":
            return 0
        elif category == "Mild, possibly due to other factors":
            return 1
        elif category == "Mild, not significant":
            return 2
        elif category == "Moderate, within normal limits":
            return 3
        elif category == "Requires attention":
            return 4
        elif category == "Suspicious, may indicate issue":
            return 5
        elif category == "Requires medical attention":
            return 6
        elif category == "Needs immediate medical attention":
            return 7
        elif category == "Life-threatening":
            return 8

    if st.button("Submit"):
        model = tf.keras.models.load_model('model_campuran.h5')
        
        Obesity = bmi(height, weight)
        Smoking = smoking_status(smoking)
        Passive_smoking = passive_smoking_status(passive_smoking)
        Air_pollution = pollution(air_pollution)
        Occupation = occupation_status(occupation)
        Genetic_risk = genetic_risk_status(genetic_risk)
        Coughing_of_blood = categorize_coughing_of_blood(coughing_of_blood)
        Shortness_of_breath = categorize_shortness_of_breath(shortness_of_breath)
        Chest_pain = categorize_chest_pain(chest_pain)
        Weight_loss = categorize_weight_loss(weight_loss)
        Clubbing_of_finger_nails = categorize_clubbing_of_finger_nails(clubbing_of_finger_nails)
        
        input_data = [[age, Obesity, Smoking, Passive_smoking, Air_pollution, Occupation, Genetic_risk, Coughing_of_blood, Shortness_of_breath, Chest_pain, Weight_loss, Clubbing_of_finger_nails]] 
        
        input_array = np.array(input_data)
        
        prediksi = np.argmax(model.predict(input_array))
        
        if prediksi == 0:
            st.subheader("Low Risk")
            st.write('Indicates a lower likelihood of developing lung cancer, often associated with factors such as younger age, non-smoking status, and minimal exposure to environmental hazards.')
        elif prediksi == 1:
            st.subheader("Medium Risk")
            st.write('Suggests a moderate probability of developing the disease, potentially influenced by a combination of factors like age, smoking history, and genetic predisposition. ')
        else:
            st.subheader("High Risk")
            st.write('Signifies a significantly elevated chance of developing lung cancer, commonly observed in older individuals with a history of heavy smoking, genetic mutations, or prolonged exposure to carcinogens.')
    
