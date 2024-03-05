import tensorflow as tf
from prettytable import PrettyTable


### IN PROGRESS

def train_model() -> None:

    x = [
        ["I", "love"],
        ["I", "can"],
        ["I", "take", "good"]
    ]

    y = [
        ["swimming"],
        ["code"],
        ["photographs"]
    ]

    vocab = set()
    dataset = []
    for index in range(0, 3, 1):
        sentence = ""
        for x_items in x[index]:
            sentence += x_items + " "
            vocab.add(x_items)
        for y_items in y[index]:
            sentence += y_items
            vocab.add(y_items)
        dataset.append([sentence])

    vectorization_layer = tf.keras.layers.TextVectorization(max_tokens=10, output_mode="int", output_sequence_length=5)
    vectorization_layer.adapt(tf.data.Dataset.from_tensor_slices(list(vocab)))

    model = tf.keras.Sequential()
    # (1,) (because we need to guarantee that there is exactly one string
    model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
    model.add(vectorization_layer)

    print(model.predict(dataset))

if __name__ == "__main__":
    train_model()
