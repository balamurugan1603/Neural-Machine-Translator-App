from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences


def sent_to_seq(sequences, tokenizer, vocab_size=None, reverse=False, onehot=False):
    
    """ Converts text data into sequences supported by model input layers.
    
    Args:
        sequences (list): List of text data.
        tokenizer (tf.keras.preprocessing.text.Tokenizer): Tensorflow tokenizer object.
        vocab_size (int): Number of words in the whole vocabulary.
        reverse (bool): Reverses the padded sequence if set True. Defaults False.
                        (Eg: if set True, [1 2 3 0 0] becomes [0 0 3 2 1])
        onehot (bool): Creates onehot representation of the padded sequence if set True.
                       Defaults False.
                       
    Returns:
        preprocessed_seq (list): List of preprocessed sequences.
        
    """
    
    # Tokenizing
    seq = tokenizer.texts_to_sequences(sequences)
    
    # Padding
    preprocessed_seq = pad_sequences(seq, padding='post', truncating='post', maxlen=20)
    
    # Reversing
    if reverse:
        preprocessed_seq = preprocessed_seq[:, ::-1]
    
    # Onehot encoding
    if onehot:
        preprocessed_seq = to_categorical(preprocessed_seq, num_classes=vocab_size) 
    
    return preprocessed_seq          
            

def word_to_onehot(tokenizer, word, vocab_size):
    
    """ Converts a single word into onehot representation.
    
    Args:
        tokenizer (tf.keras.preprocessing.text.Tokenizer): Tensorflow tokenizer object.
        word (str): Word to be tokenized and onehot encoded.
        vocab_size (int): Number of words in the whole vocabulary.
    
    Returns:
        de_onhot (list): Onehot representation of given word.
        
    """
    
    de_seq = tokenizer.texts_to_sequences([[word]])
    de_onehot = to_categorical(de_seq, num_classes=vocab_size).reshape(1, 1, vocab_size)  
    
    return de_onehot
