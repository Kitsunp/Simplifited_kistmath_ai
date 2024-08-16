import tensorflow as tf
from tensorflow import keras
from keras.layers import Embedding, Bidirectional, LSTM, BatchNormalization, Dropout, MultiHeadAttention, Dense
from keras.regularizers import l2
from keras.utils import register_keras_serializable
from src.models.memory_system import MemorySystem
from src.models.math_problem import MathProblem

VOCAB_SIZE = 1000
MAX_LENGTH = 50

@register_keras_serializable()
class Kistmat_AI(keras.Model):
    def __init__(self, input_shape, output_shape, vocab_size=VOCAB_SIZE, name=None, **kwargs):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.vocab_size = vocab_size
        super(Kistmat_AI, self).__init__(name=name, **kwargs)

        # Define layers
        self.embedding = Embedding(input_dim=vocab_size, output_dim=64)
        self.lstm1 = Bidirectional(LSTM(1024, return_sequences=True, kernel_regularizer=l2(0.01)))
        self.batch_norm1 = BatchNormalization()
        self.lstm2 = Bidirectional(LSTM(1024, kernel_regularizer=l2(0.01)))
        self.batch_norm2 = BatchNormalization()
        self.dropout = Dropout(0.5)
        self.attention = MultiHeadAttention(num_heads=8, key_dim=32)

        self.memory = MemorySystem()
        self.memory_query = Dense(64, dtype='float32')

        self.reasoning_layer = Dense(512, activation='relu', kernel_regularizer=l2(0.01))
        self.batch_norm3 = BatchNormalization()
        # Output layers for different learning stages
        self.output_layers = {
            'elementary1': Dense(128, activation='linear'),
            'elementary2': Dense(128, activation='linear'),
            'elementary3': Dense(128, activation='linear'),
            'junior_high1': Dense(128, activation='linear'),
            'junior_high2': Dense(128, activation='linear'),
            'high_school1': Dense(128, activation='linear'),
            'high_school2': Dense(128, activation='linear'),
            'high_school3': Dense(128, activation='linear'),
            'university': Dense(128, activation='linear')
        }
        self.final_output = Dense(output_shape, activation='linear')

        self._learning_stage = tf.Variable('elementary1', trainable=False, dtype=tf.string)

    def get_learning_stage(self):
        return self._learning_stage.numpy().decode()

    def set_learning_stage(self, stage):
        self._learning_stage.assign(stage.encode())

    @tf.function
    def call(self, inputs, training=False):
        current_stage = self.get_learning_stage()
        if current_stage == 'university':
            x = inputs
        else:
            x = self.embedding(inputs)
            x = self.lstm1(x)
            x = self.batch_norm1(x, training=training)
            x = self.lstm2(x)
            x = self.batch_norm2(x, training=training)

            x_reshaped = tf.expand_dims(x, axis=1)
            context = self.attention(x_reshaped, x_reshaped)
            context = tf.squeeze(context, axis=1)

            query = self.memory_query(x)
            memory_output = self.memory.query(query)

            x = tf.concat([context, memory_output], axis=-1)

        x = self.reasoning_layer(x)
        x = self.batch_norm3(x, training=training)

        if training:
            x = self.dropout(x)

        x = self.output_layers[current_stage](x)

        if training and current_stage != 'university':
            self.memory.update(query, x)

        return self.final_output(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "vocab_size": self.vocab_size,
            "learning_stage": self.get_learning_stage()
        })
        return config

    def inspect_memory(self):
        return self.memory.inspect_memory()

    @classmethod
    def from_config(cls, config):
        input_shape = config.pop("input_shape", None)
        output_shape = config.pop("output_shape", None)
        vocab_size = config.pop("vocab_size", VOCAB_SIZE)
        learning_stage = config.pop("learning_stage", "elementary1")

        if input_shape is None:
            input_shape = (MAX_LENGTH,)
        if output_shape is None:
            output_shape = 1

        instance = cls(input_shape=input_shape, output_shape=output_shape, vocab_size=vocab_size, **config)
        instance.set_learning_stage(learning_stage)
        return instance