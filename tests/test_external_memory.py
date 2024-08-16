import unittest
from src.models.external_memory import ExternalMemory
import tensorflow as tf

class TestExternalMemory(unittest.TestCase):
    def setUp(self):
        self.memory = ExternalMemory()

    def test_query(self):
        query_key = tf.constant([1.0] * self.memory.key_size)
        result = self.memory.query(query_key)
        self.assertEqual(result.shape, (self.memory.value_size,))

    def test_update(self):
        key = tf.constant([1.0] * self.memory.key_size)
        value = tf.constant([1.0] * self.memory.value_size)
        self.memory.update(key, value)
        self.assertTrue((self.memory.keys.numpy() == key.numpy()).any())

    def test_memory_decay(self):
        key = tf.constant([1.0] * self.memory.key_size)
        value = tf.constant([1.0] * self.memory.value_size)
        self.memory.update(key, value)
        initial_usage = self.memory.usage.numpy().copy()
        self.memory.update(key, value)
        updated_usage = self.memory.usage.numpy()
        self.assertTrue((updated_usage < initial_usage).all())

    def test_inspect_memory(self):
        memory_snapshot = self.memory.inspect_memory()
        self.assertIn('keys', memory_snapshot)
        self.assertIn('values', memory_snapshot)
        self.assertIn('usage', memory_snapshot)

if __name__ == '__main__':
    unittest.main()