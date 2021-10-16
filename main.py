import pickle
import numpy as np
from flask import Flask, render_template, request
from tensorflow import lite
from utils import *

app = Flask(__name__)

# Imports tokenizers
eng_tok = pickle.load(open("tokenizers/English Tokenizer.pkl", 'rb'))
it_tok = pickle.load(open("tokenizers/Italian Tokenizer.pkl", 'rb'))

eng_seq_len = 20  # First dimension of encoder Input shape
eng_vocab_size = len(eng_tok.word_index) + 1  # Second dimension of encoder Input shape
it_seq_len = 20  # First dimension of decoder Input shape
it_vocab_size = len(it_tok.word_index) + 1  # Second dimension of decoder Input shape

# Imports models
enc_interpreter = lite.Interpreter(model_path="models/encoder.tflite")
dec_interpreter = lite.Interpreter(model_path="models/decoder.tflite")

# Allocates tensors
enc_interpreter.allocate_tensors()
dec_interpreter.allocate_tensors()

# Input/ Output layer details
en_input_details = enc_interpreter.get_input_details()
de_input_details = dec_interpreter.get_input_details()
en_output_details = enc_interpreter.get_output_details()
de_output_details = dec_interpreter.get_output_details()


def translate(eng_sentence):
    """ Returns Italian translation of given english sentence.

    Args:
        eng_sentence (str): English text to be translated.

    Returns:
        it_sent (str): Italian translated text.

    """

    en_seq = sent_to_seq([eng_sentence],
                         tokenizer=eng_tok,
                         reverse=True,
                         onehot=True,
                         vocab_size=eng_vocab_size)

    enc_interpreter.set_tensor(en_input_details[0]['index'], en_seq)
    enc_interpreter.invoke()

    en_st = enc_interpreter.get_tensor(en_output_details[0]['index'])
    de_seq = word_to_onehot(it_tok, "sos", it_vocab_size)

    it_sent = ""
    for i in range(it_seq_len):
        dec_interpreter.set_tensor(de_input_details[0]['index'], en_st)
        dec_interpreter.set_tensor(de_input_details[1]['index'], de_seq)
        dec_interpreter.invoke()
        en_st = dec_interpreter.get_tensor(de_output_details[0]['index'])
        de_prob = dec_interpreter.get_tensor(de_output_details[1]['index'])
        index = np.argmax(de_prob[0, :], axis=-1)
        de_w = it_tok.index_word[index]
        de_seq = word_to_onehot(it_tok, de_w, it_vocab_size)
        if de_w == 'eos':
            break
        it_sent += de_w + ' '

    return it_sent


@app.route("/", methods=["GET", "POST"])
def translator():
    if request.method == "GET":
        return render_template("index.html")
    elif request.method == "POST":
        source_text = request.form["source"]
        translated_text = translate(source_text)
        return render_template("index.html",
                               source_text=source_text,
                               translated_text=translated_text)


if __name__ == "__main__":
    app.run(debug=True)
