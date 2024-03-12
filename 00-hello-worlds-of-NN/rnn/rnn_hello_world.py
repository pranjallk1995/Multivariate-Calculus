import tensorflow as tf

from typing import Tuple
from prettytable import PrettyTable

# IN PROGRESS

class RNNModel():

    def __init__(self) -> None:

        # encoder
        self.encoder_input = None
        self.encoder_state = None
        
        # decoder
        self.decoder_input = None
        self.final_output = None

    def encoder(self, dataset: tf.Tensor, vocab_size: int) -> None:
        self.encoder_input = tf.keras.layers.Input(shape=(None,), name="encoder_input")
        encoder_embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=8, input_length=20, name="encoder_embedding")(self.encoder_input)
        _, self.encoder_state = tf.keras.layers.SimpleRNN(units=8, return_state=True, name="encoder")(encoder_embedding)

    def decoder(self, vocab_size) -> None:
        self.decoder_input = tf.keras.Input(shape=(None,), name="decoder_input")
        decoder_embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=8, name="decoder_embedding")(self.decoder_input)
        decoder_output = tf.keras.layers.SimpleRNN(units=8, name="decoder")(decoder_embedding, initial_state=self.encoder_state)
        self.final_output = tf.keras.layers.Dense(units=8, activation="sigmoid")(decoder_output)
    
    def define_model(self, dataset: tf.Tensor, vocab_size: int) -> tf.keras.Model:
        self.encoder(dataset, vocab_size)
        self.decoder(vocab_size)
        return tf.keras.Model([self.encoder_input, self.decoder_input], self.final_output)


def ground_truth() -> Tuple[list, list, list, list]:

    sentences = [
        ["I can code a simple RNN model that will try to fit this dataset"],
        ["I can code a simple app that runs on my local system without failing"],
        ["What is the time right now? as I need to go for and interview on Google Meet"]
    ]

    vocab = set()
    for sentence_number, sentence in enumerate(sentences):
        sentences[sentence_number] = [sentence[0].lower()]
        words = [word for word in sentences[sentence_number][0].split(" ")]
        for word_number, word in enumerate(words):
            vocab.add(word)

    return sentences, list(vocab)

def vectorize(sentences: list, vocab: list) -> tf.Tensor:
    vectorization_layer = tf.keras.layers.TextVectorization(max_tokens=len(vocab)+1, output_mode="int", output_sequence_length=20, name="vectorize")
    vectorization_layer.adapt(tf.data.Dataset.from_tensor_slices(vocab))

    model = tf.keras.Sequential()
    # (1,) (because we need to guarantee that there is exactly one string
    model.add(tf.keras.Input(shape=(1,), dtype=tf.string, name="vectorize_input"))
    model.add(vectorization_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=tf.keras.losses.MeanSquaredError())
    return tf.convert_to_tensor(model.predict(sentences))

if __name__ == "__main__":

    sentences, vocab = ground_truth()
    dataset = vectorize(sentences, vocab)

    x = []
    y = []
    for sentence in dataset:
        x.append([sentence[:10]])
        y.append([sentence[10:]])

    model_obj = RNNModel()
    model = model_obj.define_model(dataset, len(vocab)+1)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss=tf.keras.losses.MeanSquaredError())
    model.summary()
