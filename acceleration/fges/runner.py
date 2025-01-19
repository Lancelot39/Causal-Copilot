import numpy as np
from SEMScore import *
from fges import *
import time
import sys

import pickle
import argparse

parser = argparse.ArgumentParser(description="Run FGES.")
parser.add_argument("dataset", type=str,
                    help="File to load data from (text file with a " \
                         "row for each datapoint and a column for each variable. \
                         If name of file ends in -checkpoint, this is assumed to be  \
                         a saved checkpoint instead")
parser.add_argument("save_name", type=str, help="File to save output to.")

parser.add_argument("sparsity", type=float, help="Sparsity penalty to use.")

parser.add_argument("-c", "--checkpoint", action="store_true",
                    help="dataset is a FGES pickled checkpoint.")
parser.add_argument("--checkpoint_frequency", type=int, default=0,
                    help="Frequency to checkpoint work (in seconds). \
                          Defaults to 0 to turn off checkpointing.")

def load_file(data_file):
    return np.loadtxt(data_file, skiprows = 1)

def main():
    args = parser.parse_args()
    if args.dataset[-15:] == "-checkpoint.pkl":
        start_time = time.time()
        fges = FGES.load_checkpoint(args.dataset)
        fges.last_checkpoint = time.time()
        result = fges.search(checkpoint=True)
    else:
        dataset = load_file(args.dataset)
        score = SEMBicScore(args.sparsity, dataset=dataset) # Initialize SEMBic Object
        variables = list(range(len(dataset[0])))
        print("Running FGES on graph with " + str(len(variables)) + " nodes.")
        fges = FGES(variables, score,
                    filename=args.dataset,
                    checkpoint_frequency=args.checkpoint_frequency,
                    save_name=args.save_name)
        start_time = time.time()
        result = fges.search()

    print("--- %s seconds ---" % (time.time() - start_time))
    with open(args.save_name + '.pkl', 'wb') as f:
        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
