import os
import random
import nibabel as nib
import streamlit as st
from matplotlib.backends.backend_agg import RendererAgg
from nilearn import plotting
import model
import matplotlib.pyplot as plt

path = 'test_images/'
_lock = RendererAgg.lock


def main():
    # SETTING PAGE CONFIG TO WIDE MODE
    st.set_page_config(
        page_title="Alzheimer's Detection",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon='https://www.flaticon.com/svg/vstatic/svg/882/882998.svg?token=exp=1616624912~hmac=12661e4152e7ff72c27d5bc8fa5aa39f'
    )

    st.write("# Alzheimer's Disease Classification by 3D CNN 🧠")
    st.write("Pick an image below. The prediction will be displayed along with the image.")

    with st.sidebar:
        st.info(
            "This project was built using a Convolution Neural Network (CNN) to classify MRI "
            "images as normal or abnormal. The trained model used yields an accuracy of 79%"
            " in the binary classification of Alzheimer's Disease patients and cognitively"
            " healthy normal control.")
        st.write("")
        st.write("#### Classification")
        #st.write("Choose to classify between Normal Control vs Alzheimer's Disease"
        #         " or Mild Cognitive Impairment vs Alzheimer's Disease")
        #st.write("")

        #status = st.radio("Select Classification Task: ", ('NC vs AD', 'MCI vs AD'))

        st.write('\n')
        st.write('\n')
        st.write('#### Github Repository')
        st.markdown('https://github.com/ryan-hogan16/CNN')

    # if status == 'NC vs AD':
        #st.write("# Normal Control vs. Alzheimer's Disease")
        #start('nc_ad')
   # else:
       # st.write("# Mild Cognitively Impaired vs. Alzheimer's Disease")
       # start('mci_ad')

    st.write("# Normal Control vs. Alzheimer's Disease")
    start('nc_ad')

# Upload box allows the user to upload a .nii image
# Each axis of the brain will be displayed
def start(class_type):
    st.write("Choose from the sample dataset to generate a prediction")

    if class_type == 'nc_ad':
        label = st.multiselect('Select Label', ("Alzheimer's Disease", 'Normal Control'))

    else:
        label = st.multiselect('Select Label', ("Alzheimer's Disease", 'Mild Cognitively Impaired'))

    if label:
        file = get_test_images(label)
        random_image = random.choice(os.listdir(file))
        st.write("File: " + random_image)
        st.write("Label: " + label[0])

        with st.spinner("Loading...⏳"):
            display_results(file + random_image, class_type)


def display_results(img_file, class_type):
    if img_file is not None:
        img = read_nifti_file(img_file)
        st.set_option('deprecation.showPyplotGlobalUse', False)

        st.write("# Model Prediction")
        st.write(model.import_and_predict(img, class_type))

        # Plot slices
        st.write("# Axial, Coronal, and Sagittal Slices")
        st.write("Below is the three-axial slices within the 3D image. Each slice is taken from the middle section of "
                 "brain within each plane. In order from left to right:")
        st.write("Coronal Plane: Face forward looking through head.")
        st.write("Sagittal Plane: Side of the head looking through.")
        st.write("Axial Plane: Above head looking down.")

        st.pyplot(stat_map(img_file))


# Returns the path for the specified image label
def get_test_images(label):
    test_path = 'test_images/'

    if label[0] == "Alzheimer's Disease":
        return test_path + 'AD/'

    elif label[0] == 'Normal Control':
        return test_path + 'HC/'

    elif label[0] == 'Mild Cognitively Impaired':
        return test_path + 'MCI/'


# Reads in a nifti file using nibabel
# Returns the processed scan by resizing and normalizing the image
def read_nifti_file(filepath):
    scan = nib.load(filepath)
    scan = scan.get_fdata()
    return scan

def pie_chart(score):
    # Data to plot
    x = score
    y = 100 - score

    labels = 'NC', 'AD'
    sizes = [10,10]
    colors = ['yellowgreen', 'lightcoral']

    if x > y:
        explode = (0.1, 0)
    else:
        explode = (0, 0.1)

    # Plot
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)

    plt.axis('equal')
    plt.show()


def stat_map(img):
    plotting.plot_stat_map(img, colorbar=False, cmap="hot_black_bone_r")


if __name__ == '__main__':
    main()
