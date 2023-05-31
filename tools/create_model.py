import tensorflow as tf
from tensorflow import keras

# Adapted from: https://keras.io/examples/vision/supervised-contrastive-learning/
class SupCon_loss(keras.losses.Loss):
    def __init__(self, temperature=1, num_classes = 1000):
        super(SupCon_loss, self).__init__(name=None)
        self.temperature = temperature
        self.num_classes = num_classes
    # Go from numerical labels (e.g. [1, 0]) to one-hot, categorical (e.g. [[0,1,...,0], [1,0,...,0]])
    def to_categorical(self, y):
        y = tf.cast(y, tf.int32)
        input_shape = tf.shape(y)
        y = tf.reshape(y, [-1]) # flatten
        categorical = tf.one_hot(y, self.num_classes)
        output_shape = tf.concat([input_shape, [self.num_classes]], 0)
        categorical = tf.reshape(categorical, output_shape)
        return categorical
    
    def contrast_loss(self, y_true, y_pred):
        # y_pred = logits, y_true = labels
        batch_size = tf.shape(y_pred)[0]

        # Make the labels categorical, e.g. label 0 becomes [1, 0, 0, ..., 0]
        y_true_categorical = self.to_categorical(y_true)
        # We compute the positives including the diagonal (untiled_mask), the negatives (negatives_mask)
        # And then the positives without the diagonal (positives_mask)
        diagonal_mask = tf.one_hot(tf.range(batch_size), batch_size)
        untiled_mask = y_true_categorical @ tf.transpose(y_true_categorical)
        negatives_mask = 1-untiled_mask
        positives_mask = untiled_mask - diagonal_mask
        
        num_positives_per_row = tf.reduce_sum(untiled_mask, axis=1)
        num_valid_views_per_sample = (tf.reshape(num_positives_per_row, [batch_size]))
        
        # We already computed the logits before calling this function, so we only divide by the temperature
        logits = y_pred/self.temperature
        logits -= tf.reduce_max(tf.stop_gradient(logits), axis=1, keepdims=True)
        exp_logits = tf.exp(logits)
        
        denominator = tf.reduce_sum(exp_logits * negatives_mask, axis=1, keepdims=True) + tf.reduce_sum(
            exp_logits * positives_mask, axis=1, keepdims=True)
        
        # Add a small constant to the denominator to avoid taking the log of zero
        log_probs = (logits - tf.math.log(denominator+0.00000001)) * positives_mask
        log_probs = tf.reduce_sum(log_probs, axis=1)
        log_probs = tf.math.divide_no_nan(log_probs, num_positives_per_row)
        
        loss = -log_probs
        loss = tf.reshape(loss, [1, batch_size])
        num_valid_views_per_sample = (tf.reshape(num_positives_per_row, [1, batch_size]))
        loss = tf.squeeze(tf.math.divide_no_nan(loss, num_valid_views_per_sample))
        return loss 

    @tf.function
    def __call__(self, labels, features, sample_weight = None):
        features = tf.math.l2_normalize(features, axis=1) # Normalize
        features = features + 0.00001 # Add small constant to avoid similarity = 0
        # Compute cosine similarity matrix for features 
        features = tf.transpose(features)
        d = tf.transpose(features) @ features
        norm = tf.math.reduce_sum(features*features, axis = 0, keepdims= True) ** .5
        # cosine similarity = d/norm/norm.T
        logits = tf.math.divide_no_nan(d,norm) 
        logits = tf.math.divide_no_nan(logits,tf.transpose(norm))
        logits = logits + 0.00001 # Add small constant to avoid cosine sim = 0
        loss = self.contrast_loss(tf.squeeze(labels), logits)
        return loss 
    
# Adapted from: https://keras.io/examples/vision/supervised-contrastive-learning/
def create_encoder(input_shape, bidirectional = False, masking = False, dropout = 0, lstm_dropout = True, 
                   num_dense = 3, num_lstm = 2, dense_size = 256, dense_size_lst = [], lstm_size = 256, bn = True):
    reg = keras.regularizers.L1L2(0, 0)
    model = keras.models.Sequential()
    
    if masking: 
        model.add(keras.layers.Masking(mask_value=0.0, input_shape=input_shape))
    model.add(keras.layers.GaussianNoise(0.001))
    # Adding the (bi)LSTM layers
    for i in range(num_lstm):
        # Parameters are different depending on if it's the first and/or last layer
        if i == 0 and not masking:
            lstm = keras.layers.LSTM(lstm_size, activation = 'tanh', return_sequences = i < num_lstm - 1, 
                   input_shape = input_shape, dropout = dropout if lstm_dropout else 0,
                   kernel_initializer=keras.initializers.HeNormal())
        else:
            lstm =keras.layers.LSTM(lstm_size, activation = 'tanh', return_sequences = i<num_lstm-1, 
                            dropout = dropout if lstm_dropout else 0, kernel_initializer=keras.initializers.HeNormal())
        if bidirectional:
            model.add(keras.layers.Bidirectional(lstm))
        else:
            model.add(lstm)

    # Adding the dense layers (all of the same size)
    if len(dense_size_lst) == 0:
        for i in range(num_dense):
            model.add(keras.layers.Dense(dense_size, activation='relu', kernel_initializer=keras.initializers.HeNormal()))
            if i < num_dense-1: # If it's not the last layer, normalize & dropout after it
                model.add(keras.layers.BatchNormalization())
                if dropout > 0: model.add(keras.layers.Dropout(dropout))
    # If size of each dense layer is specified, we use that instead of constant size layers
    else:
        for i in range(len(dense_size_lst)):
            size = dense_size_lst[i]
            model.add(keras.layers.Dense(size, activation='relu', kernel_initializer=keras.initializers.HeNormal()))
            if i < num_dense-1: # If it's not the last layer, normalize & dropout after it
                if bn: model.add(keras.layers.BatchNormalization())
                if dropout > 0: model.add(keras.layers.Dropout(dropout))
    # if bn:
    #     model.add(keras.layers.BatchNormalization())
    return model

# Get-method for getting the loss in other files
def get_loss(temp = 0.07, num_classes = 1000):
    return SupCon_loss(temp, num_classes)