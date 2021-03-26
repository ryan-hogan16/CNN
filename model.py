from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Conv3D, MaxPool3D, BatchNormalization, GlobalAveragePooling3D, Dense, Dropout
from tensorflow.python.keras.models import load_model
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

######################################################################################

# CNN MODEL

######################################################################################


# Loads the model and makes a single prediction on the import image
def import_and_predict(image, class_type):
    def nc_ad():
        model = load_model('models/classification.h5')
        img = np.array(image)
        prediction = model.predict(np.expand_dims(img, axis=0))
        scores = [1 - prediction[0], prediction[0]]

        class_names = ["Normal Control", "Alzheimer's Disease"]
        temp_score = 0
        temp_name = ''

        for score, name in zip(scores, class_names):
            if 100 * score > 50:
                temp_name = name
                temp_score = score

        col1, col2 = st.beta_columns(2)
        with col1:
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot(pie_c(temp_score, class_names, temp_name))

        st.write(
            "### The model is %.2f percent confident the MRI scan is %s"
            % ((100 * temp_score), temp_name)
        )

    def mci_ad():
        model = load_model('models/classification.h5')
        img = np.array(image)
        prediction = model.predict(np.expand_dims(img, axis=0))
        scores = [1 - prediction[0], prediction[0]]

        class_names = ["Mild Cognitive Impairment", "Alzheimer's Disease"]
        temp_score = 0
        temp_name = ''

        for score, name in zip(scores, class_names):
            if 100 * score > 50:
                temp_name = name
                temp_score = score

        col1, col2 = st.beta_columns(2)
        with col1:
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot(pie_c(temp_score, class_names, temp_name))

        st.write(
            "The model is %.2f percent confident the MRI scan is %s"
            % ((100 * temp_score), temp_name)
        )

    if class_type == "nc_ad":
        nc_ad()
    else:
        mci_ad()


# Implements the CNN model
def get_model():
    inputs = keras.Input((121, 145, 121, 1))

    conv1 = Conv3D(filters=32, kernel_size=5, activation="relu")(inputs)
    max_pool1 = MaxPool3D(pool_size=(2, 2, 2))(conv1)
    batch_norm1 = BatchNormalization()(max_pool1)
    dropout1 = Dropout(0.3)(batch_norm1)

    # Layer 2
    conv2 = Conv3D(filters=64, kernel_size=5, activation="relu")(dropout1)
    max_pool2 = MaxPool3D(pool_size=(2, 2, 2))(conv2)
    batch_norm2 = BatchNormalization()(max_pool2)

    # Layer 3
    conv3 = Conv3D(filters=128, kernel_size=3, activation="relu")(batch_norm2)
    max_pool3 = MaxPool3D(pool_size=(2, 2, 2))(conv3)
    batch_norm3 = BatchNormalization()(max_pool3)
    dropout2 = Dropout(0.3)(batch_norm3)

    conv4 = Conv3D(filters=256, kernel_size=3, activation="relu")(dropout2)
    max_pool4 = MaxPool3D(pool_size=(2, 2, 2))(conv4)
    batch_norm4 = BatchNormalization()(max_pool4)

    # Output Layer
    global_avg_pool = GlobalAveragePooling3D()(batch_norm4)
    dense = Dense(units=512, activation="relu")(global_avg_pool)
    dropout5 = Dropout(0.4)(dense)

    output = layers.Dense(units=1, activation="sigmoid")(dropout5)
    model_output = Model(inputs, output, name="alz_cnn")

    return model_output


def pie_c(score, class_names, high_label):
    high_score = 0
    low_score = 0
    labels = ""

    for k in score:
        high_score = k * 100
        low_score = 100 - high_score

    if class_names[0] == high_label:
        labels = [high_label, class_names[1]]
    elif class_names[1] == high_label:
        labels = [high_label, class_names[0]]

    sizes = [high_score, low_score]
    colors = ['yellowgreen', 'lightcoral']

    if high_score > low_score:
        explode = (0.1, 0)  # explode 1st slice
    else:
        explode = (0, 0.1)

    # Plot
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.2f%%', shadow=True, startangle=20)
    plt.title("Confidence between two classes")
    plt.legend(title="Two Classes:")
    plt.show()
