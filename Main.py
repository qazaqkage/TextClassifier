import pickle
import pandas as pd
import streamlit as st

dtc=pickle.load(open('C:/Users/LEGION/PycharmProjects/TextClassifier/dtc_model.pkl','rb'))
knn=pickle.load(open('C:/Users/LEGION/PycharmProjects/TextClassifier/knn_model.pkl','rb'))
svm=pickle.load(open('C:/Users/LEGION/PycharmProjects/TextClassifier/svm_model.pkl','rb'))
modelNN=pickle.load(open('C:/Users/LEGION/PycharmProjects/TextClassifier/MLPnn_model.pkl', 'rb'))

df=pd.read_csv('C:/Users/LEGION/PycharmProjects/TextClassifier/files/article-dataset.csv')
unique_value_min=min(df['unique_values'])
unique_value_max=max(df['unique_values'])

for i in range(df.index.min(), len(df)):
    df.label[i]=df.label[i].lower()

text_len_min=min(df['text_len'])
text_len_max=max(df['text_len'])
labels = []
labels.append(df.label.unique())
labels=labels[0]
for i in range(0, len(labels)):
    labels[i] = labels[i].lower()

def Main():
    html_temp = """
    <div style="background-color:teal ;padding:0px">
    <h2 style="color:white;text-align:center;">Text classification predictor</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    activities=['Decision Tree','KNN','SVM', 'MLP']
    option=st.sidebar.selectbox('Which model would you like to use?',activities)
    st.subheader(option)
    text_len=st.slider('Select value of standartized length of text in article', text_len_min, text_len_max)
    unique_val=st.slider('Select unique value', unique_value_min, unique_value_max)
    punc=st.text_input('Enter percentage of punctuation in text')
    theme=st.text_input('Enter theme of article which you searching?')
    theme=theme.lower()
    inputs=[[text_len, unique_val, punc]]
    if st.button('Classify'):
        if theme=='big data':
            for i in range(df.index.min(), len(df)):
                if df.label[i]==theme:
                    st.text("Filename")
                    st.write(df.filename[i])
                    st.text("Abstract")
                    st.write(df.abstract[i])
                    st.text("Keywords")
                    st.write(df.keywords[i])
        elif theme=='business intelligence':
            for i in range(df.index.min(), len(df)):
                if df.label[i]==theme:
                    st.text("Filename")
                    st.write(df.filename[i])
                    st.text("Abstract")
                    st.write(df.abstract[i])
                    st.text("Keywords")
                    st.write(df.keywords[i])
        elif theme=='ecology':
            for i in range(df.index.min(), len(df)):
                if df.label[i]==theme:
                    st.text("Filename")
                    st.write(df.filename[i])
                    st.text("Abstract")
                    st.write(df.abstract[i])
                    st.text("Keywords")
                    st.write(df.keywords[i])
        elif theme=='virtual computing':
            for i in range(df.index.min(), len(df)):
                if df.label[i]==theme:
                    st.text("Filename")
                    st.write(df.filename[i])
                    st.text("Abstract")
                    st.write(df.abstract[i])
                    st.text("Keywords")
                    st.write(df.keywords[i])
        else:
            st.write('Error!')
        if option=='Decision Tree':
            st.success(dtc.predict(inputs))
        elif option=='KNN':
            st.success(knn.predict(inputs))
        elif option=='MLP':
            st.success(modelNN.predict(inputs))
        else:
            st.success(svm.predict(inputs))

if __name__=='__main__':
    Main()