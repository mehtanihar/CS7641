The directory includes a folder named datasets which includes mobile dataset and the pulsar dataset.
The mobile dataset includes train.csv consisting of 2000 entries of mobile phone features 
along with target class for prediction of price range. 
The pulsar dataset includes pulsar_stars.csv which has information about the star and a label
depicting whether it is a pulsar star or not.
There is an environment file that installs jupyter notebook and scikit learn

The models have their own jupyter notebook file.
In order to run the model, please install the environment from the environment.yml file as:

conda env create -f environment.yml

To activate the environment, please useL

source activate ML_project1

Then run the files as:

jupyter notebook svm.ipynb
