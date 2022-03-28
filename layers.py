import tensorflow as tf
import tensorflow_text as tf_text

class AttentionHead(tf.keras.layers.Layer):
    def __init__(self, d_model, d_split):
        super(AttentionHead, self).__init__()
        self.Wq = self.add_weight(
            shape=(d_model, d_split),
            initializer="glorot_uniform",
            trainable=True, name = 'Wq')
        self.Wv = self.add_weight(
            shape=(d_model, d_split),
            initializer="glorot_uniform",
            trainable=True, name = 'Wv')
        self.d_split = d_split
        
    def call(self, input, mask= None):
        query = tf.matmul(input,self.Wq)
        keys = tf.matmul(input,self.Wv)
        values = tf.matmul(input,self.Wv)
        norm = tf.cast(self.d_split,tf.dtypes.float32)
        attention = tf.matmul(query,keys, transpose_b= True)/tf.math.sqrt(norm)
        # print('attention is: ', attention)
        # print('mask is: ', mask)
        if mask is not None:
           attention += (mask * -1e4)
        # print('attention after mask is :', attention)
        distribution = tf.keras.activations.softmax(attention,axis=-1)
        output = tf.matmul(distribution, values)

        return output

class MultiHeadAttention(tf.keras.layers.Layer):
     def __init__(self, d_model, d_split):
         super(MultiHeadAttention, self).__init__()
         self.num_heads = int(d_model/d_split)
         self.attention_array = []
         for i in range(1,self.num_heads):
             self.attention_array.append(AttentionHead(d_model,d_split))
         self.dense = tf.keras.layers.Dense(d_model)
     def call(self,input, mask=None):
         full_dim = self.attention_array[0](input, mask)
         for head in self.attention_array:
             full_dim = tf.concat([full_dim,head(input, mask)],axis=-1)
         output = self.dense(full_dim)
         return output

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, d_split, dff, rate):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, d_split)
        self.ffn1 = tf.keras.layers.Dense(dff, activation='relu')
        self.ffn2 = tf.keras.layers.Dense(d_model)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training= False, mask= None):

        attn_output = self.mha(x, mask)  
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output) 

        ffn_output = self.ffn1(out1)
        ffn_output = self.ffn2(ffn_output)  
        ffn_output = self.dropout2(ffn_output, training=training)
    
        out2 = self.layernorm2(out1 + ffn_output)  
        
        return out2


class Transformer(tf.keras.Model):
    def __init__(self, d_model, d_split, dff, rate, num_layers, vocab = None, num_class= 9):
        super().__init__()
        self.num_layers = num_layers
        self.vectorizer = vocab
        self.layer_list = []
        self.d_model = d_model
        self.embed_layer = tf.keras.layers.Embedding(self.vectorizer.vocab_size(), self.d_model)
        for i in range(num_layers):
            self.layer_list.append(EncoderLayer(d_model,d_split, dff, rate))
            
        self.dense = tf.keras.layers.Dense(num_class)
    def call(self, input, training= False):
        if not training:
            input = self.tokenize(input)
        tokens = self.vectorizer(input)
        if isinstance(tokens, tf.RaggedTensor):
            tokens = tokens.to_tensor()
        embeddings = self.embed_layer(tokens)
        
        mask = tf.cast(tf.math.equal(tokens,0),tf.dtypes.float32)
        mask = mask[:,tf.newaxis,:]
        for layer in self.layer_list:
            embeddings = layer(embeddings,training = training, mask = mask)
        logits = self.dense(embeddings)
    
        return logits
    
    def tokenize(self,input):
        x = tf.strings.regex_replace(input,'([.])', r' \1 ')
        tokens = tf_text.UnicodeScriptTokenizer().tokenize(x)
        return tokens