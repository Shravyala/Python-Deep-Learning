import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
from keras.engine.saving import load_model
from sklearn.preprocessing import LabelEncoder
import numpy as np

data = pd.read_csv('/content/spam.csv',encoding="latin-1")

data = data[['v2','v1']]
print(data)
data['v2'] = data['v2'].apply(lambda x: x.lower())
data['v2'] = data['v2'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

for idx, row in data.iterrows():
    row[0] = row[0].replace('rt', ' ')

max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['v2'].values)
X = tokenizer.texts_to_sequences(data['v2'].values)

X = pad_sequences(X)

embed_dim = 128
lstm_out = 196
def createmodel():
    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    return model
labelencoder = LabelEncoder()
integer_encoded = labelencoder.fit_transform(data['v1'])
print(integer_encoded)
print(data['v1'])
y = to_categorical(integer_encoded)
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.33, random_state = 42)
batch_size = 32
model = createmodel()
model.fit(X_train, Y_train, epochs = 5, batch_size=batch_size, verbose = 2)
score,acc = model.evaluate(X_test,Y_test,verbose=2,batch_size=batch_size)
print('score',score)
print('accuracy',acc)
print('Model Metrics Names',model.metrics_names)
model.save('my_model1.h5')
model1 = load_model('my_model1.h5')

sentences = np.array(["This is the 2nd time we have tried 2 contact u."])

prediction=tokenizer.texts_to_sequences(sentences)
prediction=pad_sequences(prediction,maxlen=152)
model1.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model1.predict(prediction))
