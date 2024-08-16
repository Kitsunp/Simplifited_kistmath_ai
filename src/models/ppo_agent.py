import tensorflow as tf
from tensorflow.keras.optimizers import Adam

class PPOAgent:
    def __init__(self, model, learning_rate=0.001, clip_ratio=0.2, entropy_coef=0.01):
        self.model = model
        self.optimizer = Adam(learning_rate)
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef

    def train(self, states, actions, advantages, old_log_probs):
        with tf.GradientTape() as tape:
            predictions = self.model(states)
            log_probs = self.compute_log_probs(predictions, actions)
            entropy = -tf.reduce_mean(log_probs * tf.exp(log_probs))

            ratio = tf.exp(log_probs - old_log_probs)
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            surrogate_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))

            loss = surrogate_loss - self.entropy_coef * entropy

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def compute_log_probs(self, predictions, actions):
        action_probs = tf.nn.softmax(predictions)
        action_indices = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=-1)
        log_probs = tf.math.log(tf.gather_nd(action_probs, action_indices))
        return log_probs

    def compute_loss(self, predictions, actions, advantages, old_log_probs):
        log_probs = self.compute_log_probs(predictions, actions)
        entropy = -tf.reduce_mean(log_probs * tf.exp(log_probs))

        ratio = tf.exp(log_probs - old_log_probs)
        clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        surrogate_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))

        loss = surrogate_loss - self.entropy_coef * entropy
        return loss