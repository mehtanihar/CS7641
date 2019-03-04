# Assignment 2 - Randomized Optimization

The code for this assignment chooses three toy problems, but there are other options available in ABAGAIL.

If you are running this code in OS X you should consider downloading Jython directly. The version provided by homebrew does mot seem to work as expected for this code.


#Data 
The pulsar dataset which was used in the last assignment has been used here again. It is present in the data folder.
The simulated annealing and randomized hill climbing have been written in python and rest are in jython.


## Usage

Because ABAGAIL does not implement cross validation some work must be done on the dataset before the other code can be run. Setup your loaders in run_experiment.py and generate the data with

python run_experiment.py --dump_data

## Output

Output CSVs and images are written to ./output and ./output/images respectively. Sub-folders will be created for
each toy problem (CONTPEAKS, FLIPFLOP, TSP) and the neural network from the Supervised Learning Project (NN_OUTPUT, NN).

If these folders do not exist the experiments module will attempt to create them.

## Running Experiments

Each experiment can be run as a separate script. Running the actual optimization algorithms to generate data requires
the use of Jython.

For the three toy problems, run:

- continuoutpeaks.py
- flipflop.py
- tsp.py


## Graphing

The plotting.py script takes care of all the plotting. Since the files output from the scripts above follow a common
naming scheme it will determine the problem, algorithm, and parameters as needed and write the output to sub-folders in
./output/images. This must be run via python, specifically an install of python that has the requirements from
requirements.txt installed.


