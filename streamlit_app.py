import streamlit as st
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

st.set_page_config(page_title='Text Generation')

def generate_text(start_string, num_generate, temperature):
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

    return generated_text


def main():
    st.title('Text Generation')
    start_string = st.text_input('Enter Starting Text')
    num_generate = st.selectbox('Select Number of Characters to Generate', range(400, 1001, 100), index=0)
    temperature = st.slider('Select Fuzziness', 0.1, 1.0, 1.0, 0.1)
    generate_button = st.button('Generate')

    if generate_button:
        generated_text = generate_text(start_string, num_generate, temperature)
        st.subheader('Generated Text:')
        st.write(generated_text)


if __name__ == '__main__':
    main()
