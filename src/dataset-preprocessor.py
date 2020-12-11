import pandas as pd
import sys
import os
from datetime import datetime
import time
import nltk
import ntpath
import threading
from atpbar import flush
from atpbar import atpbar

class ProcessDatasetThread(threading.Thread):
    def __init__(self, dataset, start, end, debug):
        super().__init__()

        self.dataset = dataset
        self.startIndex = start
        self.endIndex = end
        self.debug = debug

    def run(self):
        self.result = []
        self.isDone = False
        if self.debug:
            self.debugResult = []

        self.bar = atpbar(range(self.endIndex - self.startIndex + 1), name="Thread {}".format(int(self.startIndex / (self.endIndex+1 - self.startIndex))))
        iterator = iter(self.bar)

        # Main processing
        for index, row in self.dataset.iterrows():

            # print("INDEX = -{}-\nROW[date] = -{}-\nROW[text] = -{}-".format(index, row[0], row[1]))

            if index < self.startIndex:
                continue
            elif index > self.endIndex:
                break

            # Date manipulation
            datetime_obj = datetime.strptime(row['date'], '%Y-%m-%d %H:%M:%S')
            row['date'] = int(datetime.timestamp(datetime_obj))

            # Text manipulation
            words = nltk.sent_tokenize(row['text'])

            data = []
            for word in words:
                data.append(nltk.pos_tag(nltk.word_tokenize(word)))

            text_bucket = []
            if self.debug:
                deleted_text = []

            for items in data:
                for item in items:
                    if is_word_considerable(item):
                        text_bucket.append(item[0].lower())
                    elif self.debug:
                        deleted_text.append(item[0])

            row['text'] = text_bucket

            self.result.append([row['date'], row['text']])
            if self.debug:
                self.debugResult.append([row['date'], deleted_text])

            # Let progress bar to increase
            try:
                next(iterator)
            except Exception:
                continue

        self.isDone = True

    def getResults(self):
        if self.isDone == False:
            return None

        return self.result

    def releaseListMemory(self):
        del self.result[:]
        del self.result

        if self.debug:
            del self.debugResult[:]
            del self.debugResult

    def getDebugResult(self):
        if self.debug == False:
            return None

        return self.debugResult


"""
This function returns true if the provided string is an integer. False otherwise
"""
def is_integer(n):
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()

"""
This function acts as a word filter. It returns true only if the passed string contained in word_tuple[0]
is usefull in the next stage of the project. Otherwise, it will return false
"""
def is_word_considerable(word_tuple):
    # If it's not a noun nor an adjective -> discard
    if not word_tuple[1].startswith("N") and not word_tuple[1].startswith("J"):
        return False

    # If it contains some kind of chars -> discard
    if word_tuple[0].__contains__("/"):
        return False

    # If it has only one character (e.g., "@", "#")
    if len(word_tuple[0]) == 1:
        return False

    # Manual check ('https' is considered a noun)
    if word_tuple[0] == "https":
        return False

    return True


"""
This function returns the filename without extension
"""
def getFilename(path):
    head, tail = ntpath.split(path)
    filename = tail or ntpath.basename(head)

    return filename.split(".")[0]


"""
This function delete from the input dataset all those columns that are useless.
It keeps only the columns ['date', 'text']
"""
def deleteUselessColumns(dataset):
    dataset_columns = dataset.columns.tolist()

    dataset_columns.remove('text')
    dataset_columns.remove("date")

    dataset = dataset.drop(dataset_columns, axis=1)
    return dataset


def processDataset(dataset_path, DEBUG=False, nThreads=4):
    start = time.process_time()

    # Reading the CSV dataset
    dataset = pd.read_csv(dataset_path)
    # dataset = pd.read_csv(dataset_path, quoting=csv.QUOTE_NONE,  error_bad_lines=False)
    # dataset = pd.read_csv(dataset_path, quotechar='"', quoting=csv.QUOTE_NONE, error_bad_lines=False)

    # Delete useless columns
    dataset = deleteUselessColumns(dataset)

    # Start the multithreading processing
    nRows = len(dataset.index)
    print("--- Processing {} entries ---".format(nRows))

    threads = []
    for i in range(nThreads):

        start = int((nRows / nThreads) * i)
        end = int((nRows / nThreads) * (i + 1) - 1)
        if i == nThreads:
            end = nRows

        thread = ProcessDatasetThread(dataset, start, end, DEBUG)

        thread.start()
        threads.append(thread)

    # Merging partial results
    result = []
    if DEBUG:
        debugResult = []

    for i in range(nThreads):
        threads[i].join()

        result = result + threads[i].getResults()

        if DEBUG:
            debugResult = debugResult + threads[i].getDebugResult()

        threads[i].releaseListMemory()

    flush()

    # Save the preprocessed dataset
    filename = getFilename(dataset_path)
    newDataset = pd.DataFrame(result)
    newDataset.to_csv("{}_preprocessed.csv".format(filename), index=False, header=False)

    if DEBUG:
        debug_df = pd.DataFrame(debugResult)
        debug_df.to_csv("{}_debug_cut.csv".format(filename), index=False, header=False)

    print("--- Data preprocessing completed in {} seconds ---".format(time.process_time() - start))


if __name__ == "__main__":

    # Parsing inline parameters
    if len(sys.argv) <= 1:
        sys.stderr.write("USAGE: python3 dataset-preprocessor.py <path-to-CSV dataset>")
        exit(1)

    if not os.path.exists(sys.argv[1]):
        sys.stderr.write("ERROR: the provided dataset file does not exist")
        exit(2)

    dataset_path = sys.argv[1]
    DEBUG = False
    nThreads = 4

    for i in range(2, len(sys.argv)):
        print("Checking position {}, value = {}".format(i, sys.argv[i]))
        if sys.argv[i] == "--debug":
            DEBUG = True
            print("--- Preprocessor executed in DEBUG mode ---")
        elif sys.argv[i] == "--install":
            nltk.download("punkt")
            nltk.download("averaged_perceptron_tagger")
            print("--- Downloaded nltk dependencies ---")
        elif sys.argv[i] == "--threads":
            if len(sys.argv) > i and is_integer(sys.argv[i+1]):
                nThreads = int(sys.argv[i+1])

    print("Dataset preprocessor started with the following parameters:\n\t> dataset_path = {}\n\t> DEBUG = {}\n\t> nThreads = {}\n\n".format(dataset_path, DEBUG, nThreads))
    processDataset(dataset_path, DEBUG=DEBUG, nThreads=nThreads)
