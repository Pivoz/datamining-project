import os
import sys

DATASET_PATH = "../dataset/covid19_tweets.csv"
THREADS = 4
INSTALL_NLTK_LIB = False    # Set to True to install the libraries before the core execution
DEBUG = False               # Set to True to save also all the discarded words from the input dataset

if not (len(sys.argv) > 1 and sys.argv[1] == "--development"):
    if not os.path.exists(DATASET_PATH):
        sys.stderr.write("ERROR: the dataset file does not exist (path: {})\n".format(DATASET_PATH))
        sys.stderr.write("Download it and re-run this script\n")
        exit(1)

command = "cd ../src && python3 dataset-preprocessor.py {} --threads {}".format(DATASET_PATH, THREADS)
if INSTALL_NLTK_LIB:
    command += " --install"
if DEBUG:
    command += " --debug"

os.system(command)