import pandas as pd
import sys
import os
from datetime import datetime
import time
import nltk
import ntpath
import threading
from atpbar import atpbar, flush
import ctypes
import re


class ProcessDatasetThread(threading.Thread):
    def __init__(self, dataset, start, end, aliases, debug):
        super().__init__()

        self.dataset = dataset
        self.startIndex = start
        self.endIndex = end
        self.aliasMap = aliases
        self.debug = debug

        self.result = []
        self.isDone = False
        if self.debug:
            self.debugResult = []

    def run(self):
        bar = atpbar(range(self.endIndex - self.startIndex + 1), name="Thread {}".format(int(self.startIndex / (self.endIndex + 1 - self.startIndex))))
        iterator = iter(bar)

        # Main processing
        for index, row in self.dataset.iterrows():

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

            # Check if words are considerable
            for items in data:
                for item in items:
                    if self.is_word_considerable(item):
                        text_bucket.append(self.wordFilter(item[0]))
                    elif self.debug:
                        deleted_text.append(item[0])

            # Check if there are doubled words inside text_bucket or if it is an empty string
            for text in text_bucket:
                if text == "" or text_bucket.count(text) > 1:
                    text_bucket.remove(text)

            row['text'] = text_bucket

            self.result.append([row['date'], row['text']])
            if self.debug:
                self.debugResult.append([row['date'], deleted_text])

            # Let progress bar to increase
            try:
                next(iterator)
            except Exception:
                pass

        self.isDone = True

    def getResults(self):
        if not self.isDone:
            return None

        return self.result

    def releaseListMemory(self):
        del self.result[:]
        del self.result

        if self.debug:
            del self.debugResult[:]
            del self.debugResult

    def getDebugResult(self):
        if not self.debug or not self.isDone:
            return None

        return self.debugResult

    """
    This function filters the input word in order to remove all possible junk chars
    """
    def wordFilter(self, word):

        # Lowercase the word
        word = word.lower()

        # Remove all the emojis
        regrex_pattern = re.compile(pattern="["
                                        u"\U0001F600-\U0001F64F"  # emoticons
                                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                        u"\U00002500-\U00002BEF"  # chinese char
                                        u"\U00002702-\U000027B0"
                                        u"\U00002702-\U000027B0"
                                        u"\U000024C2-\U0001F251"
                                        u"\U0001f926-\U0001f937"
                                        u"\U00010000-\U0010ffff"
                                        u"\u2640-\u2642" 
                                        u"\u2600-\u2B55"
                                        u"\u200d"
                                        u"\u23cf"
                                        u"\u23e9"
                                        u"\u231a"
                                        u"\ufe0f"  # dingbats
                                        u"\u3030"
                                        "]+", flags=re.UNICODE)
        word = regrex_pattern.sub(r'', word)

        # TO KEEP AS A FINAL CHECK: Find if word has a more general alias in aliasMap
        try:
            word = self.aliasMap[word]
        except KeyError:
            pass

        return word

    """
    This function acts as a word decision filter. It returns true only if the passed string contained in word_tuple[0]
    is useful in the next stage of the project. Otherwise, it will return false
    """
    def is_word_considerable(self, word_tuple):
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

        # If ends with unicode HORIZONTAL ELLIPSIS char -> truncated word inside the dataset -> useless
        if word_tuple[0].__contains__(u"\u2026"):
            return False

        return True


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

    # Delete frp, the list of column names those columns I NEED
    dataset_columns.remove('text')
    dataset_columns.remove("date")

    dataset = dataset.drop(dataset_columns, axis=1)
    return dataset


def processDataset(dataset_path, DEBUG=False, nThreads=4, aliases=None):
    start = time.process_time()

    # Reading the CSV dataset
    dataset = pd.read_csv(dataset_path)

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

        thread = ProcessDatasetThread(dataset, start, end, aliases, DEBUG)

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
    print()
    print(" --- Merging partial results ---")

    filename = getFilename(dataset_path)
    newDataset = pd.DataFrame(result)
    newDataset.to_csv("{}_preprocessed.csv".format(filename), index=False, header=False)

    if DEBUG:
        debug_df = pd.DataFrame(debugResult)
        debug_df.to_csv("{}_debug_cut.csv".format(filename), index=False, header=False)

    print("--- Data preprocessing completed in {} seconds ---".format(time.process_time()-start))


if __name__ == "__main__":

    ALIAS_PATH = "./utils/preprocessor-alias.csv"

    # Parsing inline parameters
    if len(sys.argv) <= 1:
        sys.stderr.write("USAGE: python3 dataset-preprocessor.py <path-to-CSV dataset>")
        exit(1)

    if not os.path.exists(sys.argv[1]):
        sys.stderr.write("ERROR: the provided dataset file does not exist")
        exit(2)

    # Read the alias csv file
    alias = pd.read_csv(ALIAS_PATH)
    aliasMap = {}
    for index, row in alias.iterrows():
        aliasMap[row["from"]] = row["to"]
    print("--- Read {} aliases from {} file ---".format(len(aliasMap.keys()), ALIAS_PATH))

    dataset_path = sys.argv[1]
    DEBUG = False
    nThreads = 4

    for i in range(2, len(sys.argv)):
        if sys.argv[i] == "--debug":
            DEBUG = True
            print("--- Preprocessor executed in DEBUG mode ---")
        elif sys.argv[i] == "--install":
            nltk.download("punkt")
            nltk.download("averaged_perceptron_tagger")
            print("--- Downloaded nltk dependencies ---")
        elif sys.argv[i] == "--threads":
            if len(sys.argv) > i and is_integer(sys.argv[i + 1]):
                nThreads = int(sys.argv[i + 1])

    # Fix terminal prints in windows for escape sequences used in atpbar
    if sys.platform.startswith("win"):
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

    print(
        "Dataset preprocessor started with the following parameters:\n\t> dataset_path = {}\n\t> DEBUG = {}\n\t> nThreads = {}\n\n".format(
            dataset_path, DEBUG, nThreads))
    processDataset(dataset_path, DEBUG=DEBUG, nThreads=nThreads, aliases=aliasMap)
