The code can be found at https://github.com/mehtanihar/CS7641/tree/master/Project3
The github repository includes a folder named datasets which includes the pulsar dataset and the Samsung Human Activity Recognition (HAR) dataset
The HAR dataset contains 7352 data points consisting of 561 features which are the time and frequency domain transformations of 
the sensor readings of accelerometer and gyrometer from a wearable device. The goal is to identify what activity the human is undergoing:
walking, going up stairs, going down stairs, sitting, standing or laying.
 
The pulsar dataset includes pulsar_stars.csv which has information about the star and a label
depicting whether it is a pulsar star or not.

There is an environment file that installs jupyter notebook and scikit learn.

The models have their own jupyter notebook file.
In order to run the model, please install the environment from the environment.yml file as:

conda env create -f environment.yml

To activate the environment, please useL

source activate ML_project3

Then run the files as:

jupyter notebook pulsar.ipynb

jupyter notebook HAR.ipynb
