import tensorflow as tf
from tensorflow.keras import layers

class MultiHeadAttention(layers.Layer):
    
    def __init__(self, d_model, num_heads, dropout_rate=0.1, return_weights=False):
        super().__init__()

        self.h = num_heads
        self.d_m = d_model
        
        assert d_model % num_heads == 0 # d_m deve ser divisível pelo número de cabeças
        
        self.d_k = d_model//num_heads

        self.scaler = tf.math.sqrt(tf.cast(self.d_k, tf.float32))

        self.return_weights = return_weights
        
        # camadas lineares para projeção (q·W^Q, k·W^K, v·W^V):
        self.proj_q = layers.Dense(d_model)
        self.proj_k = layers.Dense(d_model)
        self.proj_v = layers.Dense(d_model)
        
        # camada linear da saída (W^O):
        self.proj_o = layers.Dense(d_model)
        
        # camada dropout para os pesos de atenção:
        self.dropout = layers.Dropout(dropout_rate)

        # avisando ao Keras que esta classe suporta máscaras:
        self.supports_masking = True
        
        
    def split_heads(self, x, batch_size):
        """
        Divide a última dimensão em (h, d_k).
        Transpõe o resultado para shape (batch_size, h, L, d_k)
        """
        
        x = tf.reshape(x, (batch_size, -1, self.h, self.d_k))
        
        return tf.transpose(x, perm=[0, 2, 1, 3])

    
    def attention(self, Q, K, V, mask=None, training=False):
        
        scores = tf.matmul(Q, K, transpose_b=True)/self.scaler
    
        # Mask: se houver máscara, somamos um número muito negativo (-1e9)
        # para que a softmax zere essas posições.
        if mask is not None:
            # a máscara entra como (batch, L), mas "scores" é (batch, h, L, L), 
            # logo, faremos mask -> (batch, 1, 1, L)    
            if len(mask.shape) == 2:
                mask = mask[:, tf.newaxis, tf.newaxis, :]
                
            mask = tf.cast(mask, dtype=tf.float32)
            scores += (1 - mask)*(-1e9)
    
        weights = tf.nn.softmax(scores, axis=-1)
        
        weights = self.dropout(weights, training=training)
    
        output = tf.matmul(weights, V)

        return output, weights

    
    def call(self, q, k, v, mask=None, training=False):
        
        batch_size = tf.shape(q)[0]
        
        # projeções lineares:
        Q = self.proj_q(q)   # (batch_size, L, d_m)
        K = self.proj_k(k)   # (batch_size, L, d_m)
        V = self.proj_v(v)   # (batch_size, L, d_m)
        
        # dividindo em múltiplas cabeças:
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        
        # calculando em paralelo todas as cabeças de atenção: 
        scaled_attention, attention_weights = self.attention(Q, K, V, mask, training=training)
        
        # reshape da "concatenação:"
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_m))
        
        # projeção final:
        output = self.proj_o(concat_attention)

        if self.return_weights: return output, attention_weights
        else: return output
        