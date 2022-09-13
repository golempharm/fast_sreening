import streamlit as st
import pandas as pd
import numpy as np

st.title('GoLem Pharm')
st.write('')

#input box

int_put2 =  st.text_input('Sequence of protein:')

if int_put2:

 word_index1 = np.load('./word_index_rnn.npy',allow_pickle='TRUE').item()
 df_bind = pd.read_csv('./data_good.csv', encoding = 'utf_8')

 df_bind['seq'] = str(int_put2)
 df_bind = df_bind.dropna()
 df_bind = df_bind.astype({'Ligand SMILES': str})
 df_bind = df_bind.astype({'seq': str})
 df_bind['to_a2'] = df_bind['Ligand SMILES'].str.cat(df_bind['seq'],sep=" ")
 lines = df_bind['to_a2'].values.tolist()

 review_lines = list()
 for line in lines:
     review_lines.append(line)

 from keras.preprocessing.text import Tokenizer
 from keras.preprocessing.sequence import pad_sequences
 from tensorflow.keras.utils import to_categorical

 MAX_SEQUENCE_LENGTH = 600

 tokenizer = Tokenizer(lower = False, char_level=True)
 tokenizer.word_index = word_index1
 sequences = tokenizer.texts_to_sequences(review_lines)

 review_pad = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, truncating = 'post')

 x_test = review_pad


 from keras.models import load_model
 model2 = load_model('./final_model_rnn.h5')
 predict_x=model2.predict(x_test)

 ynew1=np.argmax(predict_x,axis=1)
 df_bind['results'] = ynew1
 df_bind[0]
