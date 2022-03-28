import tensorflow as tf
import tensorflow_addons as tfa
from tagging_utils import MaskedLoss

def train(train_set, validation_set,model= None, batch_size=16, epochs=1, num_class= None):

    epoch_train_loss = tf.keras.metrics.Mean()
    epoch_val_loss = tf.keras.metrics.Mean()
    metric = tfa.metrics.F1Score(num_class+1, average = 'macro')

    loss_fn = MaskedLoss()
    optimizer = tf.keras.optimizers.Adam()

    train_step_signature = [
    tf.RaggedTensorSpec(shape=(None, None), dtype=tf.string,ragged_rank = 1,row_splits_dtype = tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


    @tf.function(input_signature=train_step_signature)
    def run_train_step(data, labels):
        
        with tf.GradientTape() as tape:
            logits = model(data, training= True)
            loss = loss_fn(labels, logits, pad_idx=num_class)
            # Add any regularization losses.
            if model.losses:
                loss += tf.math.add_n(model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        epoch_train_loss.update_state(loss)

        metric_labels = tf.reshape(tf.one_hot(labels,num_class+1),[-1,num_class+1])
        logits = tf.reshape(logits, [-1,num_class+1])
        
        mask = tf.cast(tf.math.not_equal(labels,num_class),dtype = tf.float32)
        mask = tf.reshape(mask,[-1])

        metric.update_state(metric_labels,logits, sample_weight = mask)

    # Function to run the validation step.
    @tf.function(input_signature=train_step_signature)
    def run_val_step(data, labels):
        logits = model(data, training=True)
        loss = loss_fn(labels, logits, pad_idx = num_class)
        
        epoch_val_loss.update_state(loss)


    for epoch in range(epochs):
            start = tf.timestamp()
            # Iterate the training data to run the training step.
            for data, labels in train_set.batch(batch_size).take(1):
                
                labels = labels.to_tensor(default_value = num_class)

                run_train_step(data, labels)

            # Iterate the validation data to run the validation step.
            for data, labels in validation_set.batch(batch_size).take(1):
                labels = labels.to_tensor(default_value = num_class)
                run_val_step(data, labels)

           
            train_loss = float(epoch_train_loss.result().numpy())
            val_loss = float(epoch_val_loss.result().numpy())
            metric_r = float(metric.result().numpy())
    
            epoch_val_loss.reset_states()
            epoch_train_loss.reset_states()
            metric.reset_states()
            end = tf.timestamp()
            time = float((end-start).numpy())
            
            print(f"Epoch: {epoch} in {round(time,1)} seconds --- Training loss: {round(train_loss,5)}"  
            f" --- Validation loss: {round(val_loss,5)} --- F1-Score: {round(metric_r,2)}")
