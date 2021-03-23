"""Example usage of gif_your_nifti."""

from gif import core as gif2nif

filename = ''

# Create a normal grayscale gif.
gif2nif.write_gif_normal(filename)

# Create a pseudocolored gif.
gif2nif.write_gif_pseudocolor(filename, colormap='plasma')

# Create a depth gif.
gif2nif.write_gif_depth(filename)

# Change the size of gifs.
gif2nif.write_gif_pseudocolor(filename, size=0.5, colormap='cubehelix')
gif2nif.write_gif_pseudocolor(filename, size=0.5, colormap='inferno')
gif2nif.write_gif_pseudocolor(filename, size=0.5, colormap='viridis')

# Create an RGB gif, based on gray matter, white matter and cerebrospinal fluid
# images from MNI template.
filename1 = 'E:/AD/wmADNI_002_S_0619_MR_MPR__GradWarp__N3__Scaled_Br_20070717184209073_S24022_I60451.nii'
filename2 = 'E:/AD/wmADNI_002_S_0619_MR_MPR__GradWarp__N3__Scaled_Br_20070717184209073_S24022_I60451.nii'
filename3 = 'E:/AD/wmADNI_002_S_0619_MR_MPR__GradWarp__N3__Scaled_Br_20070717184209073_S24022_I60451.nii'
gif2nif.write_gif_rgb(filename1, filename2, filename3)
