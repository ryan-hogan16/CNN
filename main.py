import base64
import os
import random
import nibabel as nib
import numpy as np
import streamlit as st
import tensorflow
from matplotlib.backends.backend_agg import RendererAgg
from nilearn import plotting
from scipy import ndimage
import model
import gif.core as gif2nif

path = 'test_images/'
_lock = RendererAgg.lock


def main():
    # SETTING PAGE CONFIG TO WIDE MODE
    st.set_page_config(
        page_title="Alzheimer's Detection", page_icon='http://clipart-library.com/img/2101816.png',
        layout="wide"
    )

    st.title("Alzheimer's Disease Classification by 3D CNN")
    st.write("Pick an image from the left. You'll be able to view the image and see the prediction.")
    st.write("Only accepts .nii (NIFTI) scans.")

    with st.sidebar:
        st.info(
            "This project was built using a Convolution Neural Network (CNN) to classify MRI "
            "images as normal or abnormal. The trained model used yields an accuracy of 80%"
            " in the binary classification of Alzheimer's Disease patients and cognitively"
            " healthy normal control.")
        st.write("")
        st.title("Classification")
        st.write("Choose to classify between Normal Control vs Alzhiemer's Disease"
                 " or Mild Cognitive Impairment vs Alzeihmer's Disease")
        st.write("")

        status = st.radio("Select Classification Task: ", ('NC vs AD', 'MCI vs AD'))

    if status == 'NC vs AD':
        st.title("Normal Control vs. Alzheimer's Disease")
        start('nc_ad')
    else:
        st.title("Mild Cognitively Impaired vs. Alzheimer's Disease")
        start('mci_ad')


# Upload box allows the user to upload a .nii image
# The image will be displayed between all axis as
#  well as a gif of the image slices moves through
#  the entire brain
def start(class_type):
    st.write("Choose from the sample dataset to generate a prediction")

    if class_type == 'nc_ad':
        label = st.multiselect('Select Label', ("Alzheimer's Disease", 'Normal Control'))
    else:
        label = st.multiselect('Select Label', ("Alzheimer's Disease", 'Mild Cognitively Impaired'))

    if label:
        file = get_test_images(label)
        random_image = random.choice(os.listdir(file))
        st.write("Label: " + label[0])
        st.write(random_image)
        display_results(file + random_image, class_type)


def display_results(img_file, class_type):
    if img_file is not None:
        img = read_nifti_file(img_file)
        st.write("Shape of image: ", img.shape)

        st.set_option('deprecation.showPyplotGlobalUse', False)

        st.title("Model Prediction")
        st.write(model.import_and_predict(img, class_type))

        # Plot slices
        st.title("Axial, Coronal, and Sagittal Slices")
        st.write("Below is the three-axial slices within the 3D image. Each slice is taken from the middle section of "
                 "brain within each plane. In order from left to right below:")
        st.write("Coronal Plane (Face forward looking to the back)")
        st.write("Sagittal Plane (Side of the head)")
        st.write("Axial Plane (Above head looking down)")

        st.pyplot(stat_map(img_file))

        #st.title("GIF Brain Traversal")
        #st.write("Traverses through the brain within each plane")

        #gif_path = img_file.replace('/AD/', '/gifs/')
        #gif2nif.write_gif_pseudocolor(img_file, size=1.3, colormap='hot_black_bone_r')
        #gif_path = gif_path.replace('.nii', '_hot_black_bone_r.gif')
        #show_gif(img_file)


def get_test_images(label):
    path = 'test_images/'

    for x in label:
        if x == "Alzheimers Disease":
            path = path + 'AD/'
        elif x == 'Healthy Control':
            path = path + 'HC/'
        else:
            path = path + 'MCI/'

    return path


# Reads in a nifti file using nibabel
# Returns the processed scan by resizing and normalizing the image
def read_nifti_file(filepath):
    scan = nib.load(filepath)
    scan = scan.get_fdata()
    scan = process_scan(scan)
    return scan


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


def stat_map(img):
    plotting.plot_stat_map(img)


########################################

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


if __name__ == '__main__':
    main()
