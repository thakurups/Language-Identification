def lang_detect(text):
    import pickle
    import string
    import re
    import numpy as np
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
    return pred[0]
print(lang_detect("how are you"))
print (lang_detect("fahr zur Hölle"))