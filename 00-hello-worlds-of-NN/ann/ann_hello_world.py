import tensorflow as tf
from prettytable import PrettyTable
    
def train_model(train_id: int) -> None:

    x = tf.convert_to_tensor(
        [
            [1, 0, 1],
            [1, 1, 0],
            [0, 1, 0]
        ]
    )

    y = tf.convert_to_tensor(
        [
            [1, 0, 0, 0, 1],
            [1, 0, 1, 0, 0],
            [0, 0, 1, 0, 0]
        ]
    )

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(3,)))
    model.add(tf.keras.layers.Dense(units=8, activation="sigmoid"))
    model.add(tf.keras.layers.Dense(units=5, activation="sigmoid"))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.25), loss=tf.keras.losses.MeanSquaredError())

    model.fit(x, y, epochs=10)
    model.save(f"model-{train_id}.keras")

if __name__ == "__main__":

    train = False
    for train_id in range(0, 5, 1):
        if train:
            train_model(train_id)

    table = PrettyTable()
    table.field_names = ["Input", "Actual", "Predicted", "Train Attempt"]
    for train_id in range(0, 5, 1):
        model = tf.keras.saving.load_model(f"model-{train_id}.keras")

        table.add_row(
            [
                "[1, 0, 1]", "[1, 0, 0, 0, 1]", tf.round(model.predict([[1, 0, 1]])), ""
            ]
        )
        table.add_row(
            [
                "[1, 1, 0]", "[1, 0, 1, 0, 0]", tf.round(model.predict([[1, 1, 0]])), train_id
            ]
        )
        table.add_row(
            [
                "[0, 1, 0]", "[0, 0, 1, 0, 0]", tf.round(model.predict([[0, 1, 0]])), ""
            ], divider=True
        )
    print(table)
