import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# config
MAX_VOCAB=60000
MAX_LEN=600
EMBED_DIM=128
LSTM_UNITS=128
BATCH_SIZE=64
EPOCHS=12

# clean text
def clean_text(text):
    text=str(text).lower()
    text=re.sub(r"http\S+"," ",text)
    text=re.sub(r"[^a-zA-Z0-9\s]"," ",text)
    text=re.sub(r"\s+"," ",text).strip()
    return text

# load datasets
def load_isot():
    true_df=pd.read_csv("True.csv")
    fake_df=pd.read_csv("Fake.csv")
    true_df["label"]=1
    fake_df["label"]=0
    df=pd.concat([true_df,fake_df])
    df["content"]=df["title"]+" "+df["text"]
    return df[["content","label"]]

def load_liar():
    cols=["id","label","statement","subject","speaker","job","state","party","barely_true","false","half_true","mostly_true","pants_fire","context"]
    df=pd.read_csv("train.tsv",sep="\t",names=cols)
    mapping={"true":1,"mostly-true":1,"half-true":1,"barely-true":0,"false":0,"pants-fire":0}
    df["label"]=df["label"].map(mapping)
    df=df.dropna(subset=["label"])
    df["content"]=df["statement"]
    return df[["content","label"]]

def load_fakenewsnet():
    bf_fake=pd.read_csv("BuzzFeed_fake_news_content.csv")
    bf_real=pd.read_csv("BuzzFeed_real_news_content.csv")
    pf_fake=pd.read_csv("PolitiFact_fake_news_content.csv")
    pf_real=pd.read_csv("PolitiFact_real_news_content.csv")
    bf_fake["label"]=0
    bf_real["label"]=1
    pf_fake["label"]=0
    pf_real["label"]=1
    df=pd.concat([bf_fake,bf_real,pf_fake,pf_real])
    df["content"]=df["title"].fillna("")+" "+df["text"].fillna("")
    return df[["content","label"]]

# load all
print("Loading datasets...")
isot=load_isot().sample(8000,random_state=42)
liar=load_liar()
fakenewsnet=load_fakenewsnet()
df=pd.concat([isot,liar,fakenewsnet]).reset_index(drop=True)

# clean
df["content"]=df["content"].apply(clean_text)
df=df[df["content"].str.split().str.len()>10]

# split
X=df["content"]
y=df["label"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15,stratify=y)

# tokenizer
tokenizer=Tokenizer(num_words=MAX_VOCAB,oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

def encode(texts):
    return pad_sequences(tokenizer.texts_to_sequences(texts),maxlen=MAX_LEN,padding="post")

X_train=encode(X_train)
X_test=encode(X_test)

with open("tokenizer.pkl","wb") as f:
    pickle.dump(tokenizer,f)

# class weights
class_weights=compute_class_weight(class_weight="balanced",classes=np.unique(y_train),y=y_train)
class_weights=dict(enumerate(class_weights))

# model
def build_model():
    inp=layers.Input(shape=(MAX_LEN,))
    x=layers.Embedding(MAX_VOCAB,EMBED_DIM)(inp)
    x=layers.SpatialDropout1D(0.3)(x)
    convs=[]
    for k in [3,5,7]:
        c=layers.Conv1D(128,k,activation="relu",padding="same")(x)
        c=layers.BatchNormalization()(c)
        c=layers.MaxPooling1D(2)(c)
        convs.append(c)
    x=layers.Concatenate()(convs)
    x=layers.Bidirectional(layers.LSTM(LSTM_UNITS,return_sequences=True))(x)
    x=layers.Bidirectional(layers.LSTM(LSTM_UNITS//2))(x)
    x=layers.Dense(128,activation="relu")(x)
    x=layers.BatchNormalization()(x)
    x=layers.Dropout(0.5)(x)
    x=layers.Dense(64,activation="relu")(x)
    x=layers.Dropout(0.3)(x)
    out=layers.Dense(1,activation="sigmoid")(x)
    model=models.Model(inp,out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4,clipnorm=1.0),loss="binary_crossentropy",metrics=["accuracy",tf.keras.metrics.AUC(name="auc")])
    return model

model=build_model()
model.summary()

# callbacks
callbacks_list=[
    callbacks.EarlyStopping(monitor="val_auc",patience=3,restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=1)
]

# train
model.fit(X_train,y_train,validation_split=0.1,epochs=EPOCHS,batch_size=BATCH_SIZE,class_weight=class_weights,callbacks=callbacks_list)

# evaluate
y_prob=model.predict(X_test).ravel()
best_thresh=0.5
best_f1=0
for t in np.arange(0.3,0.7,0.05):
    preds=(y_prob>t).astype(int)
    f1=f1_score(y_test,preds)
    if f1>best_f1:
        best_f1=f1
        best_thresh=t

print("Best threshold:",best_thresh)
y_pred=(y_prob>best_thresh).astype(int)

print("\nClassification Report:")
print(classification_report(y_test,y_pred))
print("ROC-AUC:",roc_auc_score(y_test,y_prob))

model.save("cnn_lstm_final.keras")
print("Model saved")
