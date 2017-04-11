import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from zipfile import ZipFile
from time import time
from numpy import empty
from subprocess import check_output


df_train = pd.read_csv('train_small.csv')
df_train.head()

texts = df_train[['question1','question2']]
labels = df_train['is_duplicate']

# Model params
MAX_NB_WORDS = 100000
MAX_SEQUENCE_LENGTH = 25
VALIDATION_SPLIT = 0.1
EMBEDDING_DIM = 32

# Train params
NB_EPOCHS = 1
BATCH_SIZE = 1024
VAL_SPLIT = 0.1
WEIGHTS_PATH = 'lstm_weights.h5'
SUBMIT_PATH = 'lstm_submission_1.csv'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tk = Tokenizer(nb_words=MAX_NB_WORDS)

tk.fit_on_texts(list(texts.question1.values.astype(str)) + list(texts.question2.values.astype(str)))
x1 = tk.texts_to_sequences(texts.question1.values.astype(str))
print(texts.question1.values.astype(str))
x1 = pad_sequences(x1, maxlen=MAX_SEQUENCE_LENGTH)

x2 = tk.texts_to_sequences(texts.question2.values.astype(str))
x2 = pad_sequences(x2, maxlen=MAX_SEQUENCE_LENGTH)



# Preprocessing Test
print("Acquiring Test Data")
t0 = time()
df_test = pd.read_csv('test.csv')
print("Done! Acquisition time:", time()-t0)

# Preprocessing
print("Preprocessing test data")
t0 = time()

i = 0
while True:
    if (i*BATCH_SIZE > df_test.shape[0]):
        break
    t1 = time()
    tk.fit_on_texts(list(df_test.iloc[i*BATCH_SIZE:(i+1)*BATCH_SIZE].question1.values.astype(str))
                    + list(df_test.iloc[i*BATCH_SIZE:(i+1)*BATCH_SIZE].question2.values.astype(str)))
    i += 1
    if (i % 100 == 0):
        print("Preprocessed Batch {0}/{1}, Word index size: {2}, ETC: {3} seconds".format(i,
                                                                int(df_test.shape[0]/BATCH_SIZE+1),
                                                                len(tk.word_index),
                                                                int(int(df_test.shape[0]/BATCH_SIZE+1)-i)*(time()-t1)))

word_index = tk.word_index

print("Done! Preprocessing time:", time()-t0)
print("Word index length:",len(word_index))

print('Shape of data tensor:', x1.shape, x2.shape)
print('Shape of label tensor:', labels.shape)

from keras.layers import Dense, Dropout, Lambda, TimeDistributed, PReLU, Merge, Activation, Embedding
from keras.models import Sequential, load_model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras import backend as K

def get_model(p_drop=0.0):
    encoder_1 = Sequential()
    encoder_1.add(Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                input_length=MAX_SEQUENCE_LENGTH))

    encoder_1.add(TimeDistributed(Dense(EMBEDDING_DIM, activation='relu')))
    encoder_1.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(EMBEDDING_DIM,)))

    encoder_2 = Sequential()
    encoder_2.add(Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                input_length=MAX_SEQUENCE_LENGTH))

    encoder_2.add(TimeDistributed(Dense(EMBEDDING_DIM, activation='relu')))
    encoder_2.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(EMBEDDING_DIM,)))

    model = Sequential()
    model.add(Merge([encoder_1, encoder_2], mode='concat'))
    model.add(BatchNormalization())

    model.add(Dense(EMBEDDING_DIM))
    model.add(PReLU())
    model.add(Dropout(p_drop))
    model.add(BatchNormalization())

    model.add(Dense(EMBEDDING_DIM))
    model.add(PReLU())
    model.add(Dropout(p_drop))
    model.add(BatchNormalization())

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

model = get_model(p_drop=0.2)
checkpoint = ModelCheckpoint('weights.h5', monitor='val_acc', save_best_only=True, verbose=2)

model.fit([x1, x2], y=labels, batch_size=384, nb_epoch=1,
                 verbose=1, validation_split=0.1, shuffle=True, callbacks=[checkpoint])
				 
# Load best model
#print("Loading best trained model")
#model = load_model(WEIGHTS_PATH)

# Predicting
i = 0
predictions = empty([df_test.shape[0],1])
while True:
    t1 = time()
    if (i * BATCH_SIZE > df_test.shape[0]):
        break
    x1 = pad_sequences(tk.texts_to_sequences(
        df_test.question1.iloc[i * BATCH_SIZE:(i + 1) * BATCH_SIZE].values.astype(str)), maxlen=MAX_SEQUENCE_LENGTH)
    x2 = pad_sequences(tk.texts_to_sequences(
        df_test.question2.iloc[i * BATCH_SIZE:(i + 1) * BATCH_SIZE].values.astype(str)), maxlen=MAX_SEQUENCE_LENGTH)
    try:
        predictions[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = model.predict([x1, x2], batch_size=BATCH_SIZE, verbose=0)
    except ValueError:
        predictions[i*BATCH_SIZE:] = model.predict([x1, x2], batch_size=BATCH_SIZE, verbose=0)[:(df_test.shape[0]-i*BATCH_SIZE)]

    i += 1
    if (i % 1000 == 0):
        print("Predicted Batch {0}/{1}, ETC: {2} seconds".format(i,
                                                                int(df_test.shape[0]/BATCH_SIZE),
                                                                int(int(df_test.shape[0]/BATCH_SIZE+1)-i)*(time()-t1)))

#predictions = [ 1 if x > 0.5 else 0 for x in predictions ]
df_test["is_duplicate"] = predictions


df_test[['test_id','is_duplicate']].to_csv(SUBMIT_PATH, header=True, index=False)
print("Done!")
print("Submission file saved to:",check_output(["ls", SUBMIT_PATH]).decode("utf8"))

