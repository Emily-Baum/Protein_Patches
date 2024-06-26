# Protein Patches
\
The code in this repository creates protein surface patches and represents them as 3D Zernike descriptors, making them convenient inputs for machine learning models. Ply files of protein surfaces, such as those from EDTSurf, are the recommended input for the functions available here.

# Using this Code
\
Assuming you are on linux, to create a ply file for use with the Protein_Patches package: \
If you are not on linux, the setup may be different from the steps below. \
Install EDTSurf (available here - https://zhanggroup.org/EDTSurf/) \
Run the following command: \
chmod +x EDTSurf \
 \
Ensure that you have EDTSurf and your pdb file of interest in the same location. \
Run the following command: \
./EDTSurf -i your_file_name.pdb \
This should generate your_file_name.ply. \
\
Now you can install the python files in the Scripts folder and follow along with the jupyter notebook in the Examples folder. \
If you work directly with the jupyter notebook, ensure that your_file_name.ply and the script files are installed in the same location.

# Dependencies:
numpy (1.21.5) \
math 3.8 (python 3.8) \
BioPython (1.78) (https://biopython.org/) \
EDTSurf (https://zhanggroup.org/EDTSurf/)

# Questions?
Contact Emily Baum at byx3au@virginia.edu
