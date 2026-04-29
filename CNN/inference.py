import sys,re,pickle,argparse,numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_PATH="cnn_lstm_fakenews.keras"
TOKENIZER_PATH="tokenizer.pkl"
MAX_LEN=500

# clean text
def clean_text(text:str)->str:
    text=text.lower()
    text=re.sub(r"https?://\S+|www\.\S+"," ",text)
    text=re.sub(r"<.*?>"," ",text)
    text=re.sub(r"[^a-z\s]"," ",text)
    text=re.sub(r"\s+"," ",text).strip()
    return text

# load model and tokenizer
def load_artifacts():
    model=tf.keras.models.load_model(MODEL_PATH)
    with open(TOKENIZER_PATH,"rb") as f:
        tokenizer=pickle.load(f)
    return model,tokenizer

# predict
def predict(texts:list[str],model,tokenizer)->list[dict]:
    cleaned=[clean_text(t) for t in texts]
    seqs=tokenizer.texts_to_sequences(cleaned)
    padded=pad_sequences(seqs,maxlen=MAX_LEN,padding="post",truncating="post")
    probs=model.predict(padded,verbose=0).ravel()
    results=[]
    for text,prob in zip(texts,probs):
        label="REAL" if prob>=0.5 else "FAKE"
        confidence=prob if prob>=0.5 else 1-prob
        results.append({"label":label,"confidence":float(confidence),"real_prob":float(prob),"preview":text[:120]})
    return results

# main
def main():
    parser=argparse.ArgumentParser(description="Fake News Detector")
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("text",nargs="?",help="Article text")
    group.add_argument("--file","-f",help="Path to txt file")
    args=parser.parse_args()

    if args.file:
        with open(args.file,"r",encoding="utf-8") as fh:text=fh.read()
    else:text=args.text

    print("Loading model...")
    model,tokenizer=load_artifacts()
    r=predict([text],model,tokenizer)[0]

    bar="█"*int(r["confidence"]*30)
    print(f"\n{'-'*50}")
    print(f"Verdict    : {r['label']}")
    print(f"Confidence : {r['confidence']:.1%} {bar}")
    print(f"Real prob  : {r['real_prob']:.4f}")
    print(f"Preview    : {r['preview']} ...")
    print(f"{'-'*50}\n")

if __name__=="__main__":
    main()
