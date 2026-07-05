import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model
from myattention import MultiHeadAttention

class BertEmbeddings(layers.Layer):
    
    def __init__(self, vocab_size, max_len, d_model, seg_dim=2):
        super().__init__()
    
        self.d_model = d_model
        self.seg_dim = seg_dim
        
        self.token_emb = layers.Embedding(vocab_size, d_model) 
        self.pos_emb = layers.Embedding(max_len, d_model)
        self.segment_emb = layers.Embedding(seg_dim, d_model)
        
        self.norm = layers.LayerNormalization(epsilon=1e-12)
        self.dropout = layers.Dropout(0.1)

    
    def call(self, inputs):

        input_ids, segment_ids = inputs
            
        positions = tf.range(start=0, limit=tf.shape(input_ids)[1], delta=1)
         
        x = self.token_emb(input_ids)       # (batch, seq, d_model)
        x += self.pos_emb(positions)        # (seq, d_model)
        
        if self.seg_dim > 1:
            x += self.segment_emb(segment_ids)  # (batch, seq, d_model)
                
        x = self.norm(x)

        return self.dropout(x)


class EncoderLayer(layers.Layer):
    
    def __init__(self, d_model, num_heads, dense_dim, rate=0.1):
        super().__init__()

        assert d_model % num_heads == 0
        
        self.mha = MultiHeadAttention(d_model, num_heads, dropout_rate=rate)
        
        self.ffn = Sequential([layers.Dense(dense_dim, activation='gelu'), 
                               layers.Dense(d_model)])

        self.norm1 = layers.LayerNormalization(epsilon=1e-12)
        self.norm2 = layers.LayerNormalization(epsilon=1e-12)
        
        self.dropout = layers.Dropout(rate) 
    
    def call(self, x, training, mask):

        x_attn = self.mha(x, x, x, mask=mask, training=training) # já temos dropout aqui          
        x = self.norm1(x + x_attn)
        
        x_ffn = self.ffn(x)
        x_ffn = self.dropout(x_ffn, training=training)
        
        return self.norm2(x + x_ffn)
        

class MiniBERT(Model):

    def __init__(self, num_layers, d_model, num_heads, dense_dim, dropout_rate, vocab_size, max_len, seg_dim):
        super().__init__()
    
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding_layer = BertEmbeddings(vocab_size, max_len, d_model, seg_dim)
        
        self.encoder_layers = [EncoderLayer(d_model, num_heads, dense_dim, dropout_rate) 
                               for _ in range(num_layers)]
        
        self.pooler_dense = layers.Dense(d_model, activation='tanh', name="pooler")

    
    def call(self, inputs, training=False):

        input_ids, segment_ids, mask = inputs
                
        x = self.embedding_layer((input_ids, segment_ids))
        
        for layer in self.encoder_layers:
            x = layer(x, training=training, mask=mask)
            
        # sequence output (x): (batch, seq_len, d_model) - representação de todos os tokens
        
        pooled_output = self.pooler_dense(x[:, 0, :]) # cls-token = x[:, 0, :] (batch, d_model)
        
        return x, pooled_output
        