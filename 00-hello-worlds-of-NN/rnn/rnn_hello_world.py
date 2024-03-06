import tensorflow as tf

from typing import Tuple
from prettytable import PrettyTable

def ground_truth() -> Tuple[list, list, list, list]:

    sentences = [
        ["I can code a simple RNN model that will try to fit this dataset"],
        ["I can code a simple app that runs on my local system without failing"],
        ["What is the time right now? as I need to go for and interview on Google Meet"]
    ]

    x = []
    y = []
    vocab = set()
    for sentence_number, sentence in enumerate(sentences):
        sentences[sentence_number] = [sentence[0].lower()]
        words = [word for word in sentences[sentence_number][0].split(" ")]
        for word in words:
            vocab.add(word)
        x.append(words[:10])
        y.append(words[10:])

    return x, y, sentences, list(vocab)

def train_model(dataset: list, vocab: list) -> None:
    vectorization_layer = tf.keras.layers.TextVectorization(max_tokens=len(vocab)+1, output_mode="int", output_sequence_length=20)
    vectorization_layer.adapt(tf.data.Dataset.from_tensor_slices(vocab))

    model = tf.keras.Sequential()
    # (1,) (because we need to guarantee that there is exactly one string
    model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
    model.add(vectorization_layer)
    model.add(tf.keras.layers.Embedding(input_dim=len(vocab)+1, output_dim=8, input_length=20))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=tf.keras.losses.MeanSquaredError())

    print(model.predict(dataset))

if __name__ == "__main__":
    x, y, dataset, vocab = ground_truth()
    train_model(dataset, vocab)
