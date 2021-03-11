import base64
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import streamlit as st
import tensorflow
from scipy import ndimage
from gif import core as gif2nif
import model
from nilearn import plotting, datasets
from matplotlib.backends.backend_agg import RendererAgg

from nilearn.regions import RegionExtractor
from nilearn import plotting
from nilearn.image import index_img
from nilearn.plotting import find_xyz_cut_coords

path = 'E:/AD/'
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
        upload_box('nc_ad')
    else:
        st.title("Mild Cognitively Impaired vs. Alzheimer's Disease")
        upload_box('mci_ad')


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
        st.write("Below is the three-axial slices within the 3D image. Each slice is taken from the middle section of "
                 "brain within each plane. In order from left to right below:")
        st.write("Coronal Plane (Face forward looking to the back)")
        st.write("Sagittal Plane (Side of the head)")
        st.write("Axial Plane (Above head looking down)")

        #st.pyplot(show_slices([slice3, slice2, slice1]))
        st.pyplot(stat_map(path + uploaded_file.name))

        # Write gif to file then show_gif()
        gif2nif.write_gif_pseudocolor(path + uploaded_file.name, size=1.1, colormap='gist_rainbow')

        row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.beta_columns(
            (.1, 1, .1, 1, .1))

        with row0_1, _lock:
            st.title("GIF Brain Traversal")
            st.write("Traverses through the brain within each plane")
            show_gif(path + uploaded_file.name.replace('.nii', '_gist_rainbow.gif'))

            # Write prediction
            st.title("Prediction")
            st.write(model.import_and_predict(img, class_type))

        with row0_2, _lock:
            st.pyplot(plot_roi(path + uploaded_file.name))


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


def plot_roi(img):
    plotting.plot_roi(img)

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


if __name__ == '__main__':
    main()
