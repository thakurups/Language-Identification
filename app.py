import numpy as np
from flask import Flask, request, render_template,url_for
import pickle
import string
import re

app = Flask(__name__,template_folder='templates')
# model = pickle.load(open('LIModel1.pkl', 'rb'))

def lang_detect(text):
    translate_table = dict((ord(char), None) for char in string.punctuation)

    global langDetectModel
    l_file = open("LIModel.pkl", "rb")
    langDetectModel = pickle.load(l_file)
    l_file.close()

    text = " ".join(text.split())
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(translate_table)
    pred = langDetectModel.predict([text])
    prob = langDetectModel.predict_proba([text])
    # print(pred[0])
    return pred[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index')
def predict():
    return render_template('index.html')

@app.route('/result',methods=['POST'])
def result():
    text=request.form.get("text")
    print(text)
    result=lang_detect(text)
    return render_template("index.html", prediction='"{}" is written in  language : {}'.format(text,result))


if __name__ == "__main__":
    app.run(debug=True)