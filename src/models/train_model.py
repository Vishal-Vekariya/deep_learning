from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def deep_learning( Y,X,X_scaled):
    
    x_train,x_test,y_train,y_test=train_test_split(X_scaled,Y,test_size=0.2,random_state=1,stratify=Y)
    
    tf.keras.utils.set_random_seed(42)

# set model_1 to None
    model_1 = None

    # 1. Create the model (same as model_1 but with an extra layer)
    model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(1), # add an extra layer
    tf.keras.layers.Dense(1) # output layer
    ])

    # 2. Compile the model
    comp = model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                    optimizer=tf.keras.optimizers.SGD(learning_rate=0.0009),
                    metrics=['accuracy'])

    # 3. Fit the model
    model_1.fit(x_train, y_train, epochs=50,verbose=0)
    eva = model_1.evaluate(x_train, y_train)
    
    return comp,eva

def best_rate( Y,X,X_scaled):
        x_train,x_test,y_train,y_test=train_test_split(X_scaled,Y,test_size=0.2,random_state=1,stratify=Y)
        tf.keras.utils.set_random_seed(42)

        # set model_1 to None
        model_1 = None

        # 1. Create the model (same as model_1 but with an extra layer)
        model_1 = tf.keras.Sequential([
        tf.keras.layers.Dense(1), # add an extra layer
        tf.keras.layers.Dense(1) # output layer
        ])

        # Compile the model
        model_1.compile(loss="binary_crossentropy", # we can use strings here too
                    optimizer=tf.keras.optimizers.SGD(),
                    metrics=["accuracy"])

        # Create a learning rate scheduler callback
        # traverse a set of learning rate values starting from 1e-3, increasing by 10**(epoch/20) every epoch
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: 0.001 * 0.9**(epoch/3)
        )


        # Fit the model (passing the lr_scheduler callback)
        history = model_1.fit(x_train,
                            y_train,
                            epochs=100,
                            verbose=0,
                            callbacks=[lr_scheduler])
        
        return history
    
def activation_function( Y,X,X_scaled):
    x_train,x_test,y_train,y_test=train_test_split(X_scaled,Y,test_size=0.2,random_state=1,stratify=Y)
    tf.keras.utils.set_random_seed(42)

# set model_1 to None
    model_1 = None

    # 1. Create the model (same as model_1 but with an extra layer)
    model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(1), # try activations LeakyReLU, sigmoid, Relu, tanh. Default is Linear
    tf.keras.layers.Dense(1, activation = 'sigmoid') # output layer
    ])

    # 2. Compile the model
    model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                    optimizer=tf.keras.optimizers.SGD(learning_rate=0.0009),
                    metrics=['accuracy'])

    # 3. Fit the model
    history1 = model_1.fit(x_train, y_train, epochs=50,verbose=0)
    model_1.evaluate(x_train, y_train)
    y_preds = tf.round(model_1.predict(x_test))
    y_preds[:3]
    return y_test, y_preds, history1
