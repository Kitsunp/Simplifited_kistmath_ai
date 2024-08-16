import tensorflow as tf
from src.models.external_memory import ExternalMemory

class MemorySystem:
    def __init__(self):
        self.formulative_memory = ExternalMemory()
        self.conceptual_memory = ExternalMemory()
        self.short_term_memory = ExternalMemory()
        self.long_term_memory = ExternalMemory()
        self.inference_memory = ExternalMemory()

    def query(self, query_key):
        # Implement querying across different memory components
        query_key = tf.cast(query_key, tf.float32)
        
        # Query each memory component
        form_output = self.formulative_memory.query(query_key)
        conc_output = self.conceptual_memory.query(query_key)
        short_output = self.short_term_memory.query(query_key)
        long_output = self.long_term_memory.query(query_key)
        infer_output = self.inference_memory.query(query_key)
        
        # Combine the outputs
        combined_output = tf.concat([form_output, conc_output, short_output, long_output, infer_output], axis=-1)
        return combined_output

    def update(self, key, value):
        # Implement updating across different memory components
        key = tf.cast(key, tf.float32)
        value = tf.cast(value, tf.float32)
        
        # Update each memory component
        self.formulative_memory.update(key, value)
        self.conceptual_memory.update(key, value)
        self.short_term_memory.update(key, value)
        self.long_term_memory.update(key, value)
        self.inference_memory.update(key, value)