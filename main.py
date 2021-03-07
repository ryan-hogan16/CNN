import base64
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import streamlit as st
import tensorflow
from scipy import ndimage
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Conv3D, MaxPool3D, BatchNormalization, GlobalAveragePooling3D, Dense, Dropout
from tensorflow.python.keras.models import load_model

from gif import core as gif2nif

path = 'E:/AD/'


def main():
    # SETTING PAGE CONFIG TO WIDE MODE
    st.set_page_config(layout="wide")

    st.sidebar.title("About")

    st.sidebar.info(
        "This project was built using a Convolution Neural Network (CNN) to classify MRI "
        "images as normal or abnormal. The trained model used yields an accuracy of 80%"
        "in the binary classification between Alzheimer's Disease patients and cognitively"
        "healthy normal control.")

    st.title("Alzheimer's Disease Classification")
    st.write("Pick an image from the left. You'll be able to view the image and see the prediction.")
    st.write("Only accepts .nii NIFTI scans.")
    st.write("")
    st.sidebar.title("Classification")
    st.sidebar.write("Choose to classify between Normal Control vs Alzhiemer's Disease"
                     " or Mild Cognitive Impairment vs Alzeihmer's Disease")
    st.write("")
    upload_box("nc_ad")


# Upload box allows the user to upload a .nii image
# The image will be displayed between all axis as
#  well as a gif of the image slices moves through
#  the entire brain
def upload_box(class_type):
    uploaded_file = st.file_uploader("Choose a image file", type="nii")

    if uploaded_file is not None:
        img = read_nifti_file(path + uploaded_file.name)

        st.write("Shape of image: ", img.shape)
        slice1 = img[55, :, :]
        slice2 = img[:, 55, :]
        slice3 = img[:, :, 55]

        st.set_option('deprecation.showPyplotGlobalUse', False)

        # Plot slices
        st.title("Axial, Coronal, and Sagittal Slices")
        st.pyplot(show_slices([slice1, slice2, slice3]))

        # Write gif to file then show_gif()
        gif2nif.write_gif_pseudocolor(path + uploaded_file.name, size=1.35, colormap='gist_rainbow')
        st.title("GIF Brain Traversal")
        show_gif(path + uploaded_file.name.replace('.nii', '_gist_rainbow.gif'))

        # Write prediction
        st.title("Prediction")
        st.write(import_and_predict(img, class_type))


# Reads in a nifti file using nibabel
# Returns the processed scan by resizing and normalizing the image
def read_nifti_file(filepath):
    scan = nib.load(filepath)
    scan = scan.get_fdata()
    scan = process_scan(scan)
    return scan


# Displays the three axis slices (coronal, sagittal, and axial)
def show_slices(slices):
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gist_rainbow", origin="lower")


def plot_slices(num_rows, num_columns, width, height, image_data):
    image_data = np.rot90(np.array(image_data))
    image_data = np.transpose(image_data)
    image_data = np.reshape(image_data, (num_rows, num_columns, width, height))
    rows_data, columns_data = image_data.shape[0], image_data.shape[1]
    heights = [slc[0].shape[0] for slc in image_data]
    widths = [slc.shape[1] for slc in image_data[0]]
    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)
    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    )
    for i in range(rows_data):
        for j in range(columns_data):
            axarr[i, j].imshow(image_data[i][j], cmap="Blues")
            axarr[i, j].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()


# Displays a gif of the image that moves through the entire brain
def show_gif(filename):
    file_ = open(filename, "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="brain gif">',
        unsafe_allow_html=True,
    )


######################################################################################

# DATA PROCESSING

######################################################################################


# Normalized the data between min and max
# Returns volume as float32
def normalize(volume):
    mi = np.min(volume)
    ma = np.max(volume)
    volume = (volume - mi) / (ma - mi)
    volume = volume.astype("float32")
    return volume


# Resize the image to shape (121, 145, 121)
def resize_volume(img):
    desired_depth = 121
    desired_width = 121
    desired_height = 145
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height

    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img


# Expand the dimension axis to 3 using tensorflow.expand_dims()
# Flow from tensor slices to get the slices of the array
def image_preprocessing(image):
    volume = tensorflow.expand_dims(image, axis=3)
    volume = tensorflow.data.Dataset.from_tensor_slices(volume)
    return volume


# Process the scans by resizing and normalizing the image
def process_scan(image):
    volume = resize_volume(image)
    volume = normalize(volume)
    return volume


######################################################################################

# CNN MODEL

######################################################################################

# Loads the model and makes a single prediction on the import image
def import_and_predict(image, class_type):
    def nc_ad():
        model = load_model('E:/Classifications/1-9-classification(80%)/classification.h5')
        img = np.array(image)
        prediction = model.predict(np.expand_dims(img, axis=0))
        scores = [1 - prediction[0], prediction[0]]

        class_names = ["Normal", "AD"]
        for score, name in zip(scores, class_names):
            st.write(
                "This model is %.2f percent confident the MRI scan is %s"
                % ((100 * score), name)
            )

    def mci_ad():
        model = load_model('E:/Classifications/1-9-classification(80%)/classification.h5')
        img = np.array(image)
        prediction = model.predict(np.expand_dims(img, axis=0))
        scores = [1 - prediction[0], prediction[0]]

        class_names = ["MCI", "AD"]
        for score, name in zip(scores, class_names):
            st.write(
                "This model is %.2f percent confident the MRI scan is %s"
                % ((100 * score), name)
            )

    if class_type == "nc_ad":
        nc_ad()
    else:
        mci_ad()


# Implements the CNN model
def get_model():
    inputs = keras.Input((121, 145, 121, 1))
    conv1 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation="relu")(inputs)
    max_pool1 = MaxPool3D(pool_size=(2, 2, 2))(conv1)
    batch_norm1 = BatchNormalization()(max_pool1)
    dropout1 = Dropout(0.3)(batch_norm1)

    # Layer 2
    conv2 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation="relu")(dropout1)
    max_pool2 = MaxPool3D(pool_size=(2, 2, 2))(conv2)
    batch_norm2 = BatchNormalization()(max_pool2)

    # Layer 3
    conv3 = Conv3D(filters=128, kernel_size=(3, 3, 3), activation="relu")(batch_norm2)
    max_pool3 = MaxPool3D(pool_size=(2, 2, 2))(conv3)
    batch_norm3 = BatchNormalization()(max_pool3)
    dropout2 = Dropout(0.3)(batch_norm3)

    conv4 = Conv3D(filters=256, kernel_size=(3, 3, 3), activation="relu")(dropout2)
    max_pool4 = MaxPool3D(pool_size=(2, 2, 2))(conv4)
    batch_norm4 = BatchNormalization()(max_pool4)

    # Output Layer
    global_avg_pool = GlobalAveragePooling3D()(batch_norm4)
    dense = Dense(units=512, activation="relu")(global_avg_pool)
    dropout5 = Dropout(0.4)(dense)

    output = layers.Dense(units=1, activation="sigmoid")(dropout5)
    model_output = Model(inputs, output, name="alz_cnn")

    return model_output


if __name__ == '__main__':
    main()