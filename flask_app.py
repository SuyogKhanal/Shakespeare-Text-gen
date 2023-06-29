from flask import Flask, render_template, request
import tensorflow as tf
import pathlib
import numpy as np
import warnings

warnings.filterwarnings('ignore')
cache_dir = './tmp'
dataset_file_name = 'shakespeare.txt'
dataset_file_origin = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
dataset_file_path = tf.keras.utils.get_file(
    fname=dataset_file_name,
    origin=dataset_file_origin,
    cache_dir=pathlib.Path(cache_dir).absolute()
)
text = open(dataset_file_path, mode='r').read()
vocab = sorted(set(text))

char2index = {char: index for index, char in enumerate(vocab)}
index2char = np.array(vocab)

model = tf.keras.models.load_model('text_generation_shakespeare_rnn.h5')


app = Flask(__name__,template_folder='template')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate_text', methods=['POST'])
def generate_text():
    start_string = request.form['start_string']
    num_generate = int(request.form['num_generate'])
    temperature = float(request.form['temperature'])

    input_indices = [char2index[s] for s in start_string]
    input_indices = tf.expand_dims(input_indices, 0)

    text_generated = []

    model.reset_states()
    for char_index in range(num_generate):
        predictions = model(input_indices)
        predictions = tf.squeeze(predictions, 0)

        predictions = predictions / temperature
        predicted_id = tf.random.categorical(
            predictions,
            num_samples=1
        )[-1, 0].numpy()

        input_indices = tf.expand_dims([predicted_id], 0)

        text_generated.append(index2char[predicted_id])

    generated_text = start_string + ''.join(text_generated)

    return render_template('index.html', text_generated=generated_text)


if __name__ == '__main__':
    app.run(port=6868)
