import tensorflow as tf
import numpy as np
from mesh_nn import MeshNeuralNetwork

BATCH_SIZE = 50
HIDDEN_UNITS = 28
OUTPUT = 10
TICKS = 3
EPOCHS = 70
LEARNING_RATE = 0.001
LOGDIR = "logs/dense"
SEED = 42


(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape([X_train.shape[0],np.prod(X_train.shape[1:])]).astype(np.float32)/255
Y_train = Y_train.astype(np.float32)
X_test = X_test.reshape([X_test.shape[0],np.prod(X_test.shape[1:])]).astype(np.float32)/255
Y_test = Y_test.astype(np.float32)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(buffer_size=X_train.shape[0], seed=SEED).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))

file_writer = tf.summary.create_file_writer(LOGDIR, flush_millis=10000)

init = tf.initializers.RandomUniform(minval=-1,maxval=1,seed=SEED)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(HIDDEN_UNITS+OUTPUT,activation=tf.nn.relu,input_shape=X_test.shape[1:],kernel_initializer=init),
    tf.keras.layers.Dense(HIDDEN_UNITS+OUTPUT,activation=tf.nn.relu,kernel_initializer=init),
    tf.keras.layers.Dense(OUTPUT,kernel_initializer=init)
])


optimizer_mesh = tf.optimizers.Adam(LEARNING_RATE)
loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
loss = tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True)
for epoch in range(0, EPOCHS):
    accuracy.reset_states()
    loss.reset_states()
    for batch_X, batch_Y in train_dataset:
        with tf.GradientTape() as g:
            out = model(batch_X)
            loss_ = loss_function(batch_Y, out)
            loss_ = tf.reduce_mean(loss_)
        grads = g.gradient(loss_, model.trainable_variables)
        optimizer_mesh.apply_gradients(zip(grads,model.trainable_variables))
        loss.update_state(batch_Y, out)
        accuracy.update_state(batch_Y, tf.nn.softmax(out,axis=-1))

    with file_writer.as_default():
        tf.summary.scalar('train loss', data=loss.result().numpy(), step=epoch)
        tf.summary.scalar('train accuracy', data=accuracy.result().numpy(), step=epoch)

    out = model(X_test)
    test_loss = loss_function(Y_test, out)
    test_loss = tf.reduce_mean(test_loss)
    accuracy.reset_states()
    accuracy.update_state(Y_test, tf.nn.softmax(out,axis=-1))

    with file_writer.as_default():
        tf.summary.scalar('test loss', data=test_loss, step=epoch)
        tf.summary.scalar('test accuracy', data=accuracy.result().numpy(), step=epoch)

