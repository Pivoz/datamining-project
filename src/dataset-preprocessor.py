import pandas as pd
import sys
import os
from datetime import datetime
import nltk
import ntpath
from atpbar import atpbar, flush, find_reporter, register_reporter
import ctypes
import re
import multiprocessing

class DatasetProcessor(multiprocessing.Process):
    def __init__(self, id, dataset, aliases, debug, pipe, bar_reporter):
        super().__init__()

        self.id = id
        self.dataset = dataset
        self.aliasMap = aliases
        self.debug = debug
        self.pipe = pipe
        self.bar_reporter = bar_reporter

    def run(self):
        register_reporter(self.bar_reporter)
        nRows = len(self.dataset.index)

        bar = atpbar(range(nRows), name="Process {}".format(self.id))
        iterator = iter(bar)

        result = []
        debugResult = []

        # Main processing
        for index, row in self.dataset.iterrows():

            # Date manipulation
            try:
                datetime_obj = datetime.strptime(row['date'], '%Y-%m-%d %H:%M:%S')
                computedTimestamp = int(datetime.timestamp(datetime_obj))
            except ValueError:
                # Used in tweets_sentiment_analysis dataset
                split = row["date"].split(" ")
                newDate = "{}-{}-{} {}".format(split[5], self.getMonth(split[1].lower()), split[2], split[3])

                datetime_obj = datetime.strptime(newDate, '%Y-%m-%d %H:%M:%S')
                computedTimestamp = int(datetime.timestamp(datetime_obj))

            # Text manipulation
            words = nltk.sent_tokenize(row['text'])

            data = []
            for word in words:
                data.append(nltk.pos_tag(nltk.word_tokenize(word)))

            text_bucket = []
            deleted_text = []

            # Check if words are considerable
            for items in data:
                for item in items:
                    if self.is_word_considerable(item):
                        splitWords = self.wordFilter(item[0])
                        for split in splitWords:
                            text_bucket.append(split)
                    elif self.debug:
                        deleted_text.append(item[0])

            # Check if there are doubled words inside text_bucket or if it is an empty string
            temp = []
            for i in range(len(text_bucket)):
                text = text_bucket[i]
                if len(text) > 1 and text_bucket.count(text) == 1 and not text.isnumeric():
                    temp.append(text)
                text_bucket[i] = ""

            text_bucket = temp

            result.append([computedTimestamp, text_bucket])
            if self.debug:
                debugResult.append([computedTimestamp, deleted_text])

            # Let progress bar to increase
            try:
                next(iterator)
            except Exception:
                pass

        # Final bar increase
        try:
            next(iterator)
        except Exception:
            pass

        # Send results in the process pipe
        self.pipe.send(result)
        if self.debug:
            self.pipe.send(debugResult)

        self.pipe.close()

    def getMonth(self, str):
        months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
        for i in range(len(months)):
            if months[i] == str:
                return i + 1

        return 12   # December

    """
    This function filters the input word in order to remove all possible junk chars
    """
    def wordFilter(self, word):

        # Lowercase the word
        word = word.lower()

        # Remove all the emojis
        word = self.deEmojify(word)

        # Substitute all non-alphanumeric chars with a blank
        filteredWord = ""
        for char in word:
            if str.isalnum(char):
                filteredWord += char
            else:
                filteredWord += " "
        word = filteredWord

        # With the above operations, there is the possibility that a word can be split in two or more words wrt the blank char
        wordsList = word.split(" ")

        # Cleaning the new list from empty or single-char strings or if it is a number
        for text in wordsList:
            if len(text) <= 1 or text.isnumeric():
                wordsList.remove(text)

        # TO KEEP AS A FINAL CHECK: Find if word has a more general alias in aliasMap
        for i in range(len(wordsList)):
            try:
                wordsList[i] = self.aliasMap[wordsList[i]]
            except KeyError:
                pass

        return wordsList

    def deEmojify(self, text):
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
        return regrex_pattern.sub(r'', text)

    """
    This function acts as a word decision filter. It returns true only if the passed string contained in word_tuple[0]
    is useful in the next stage of the project. Otherwise, it will return false
    """
    def is_word_considerable(self, word_tuple):
        # If it's not a noun nor an adjective nor a past verb -> discard
        if not word_tuple[1].startswith("N") and not word_tuple[1].startswith("J") and not word_tuple[1] == "VBD":
            return False

        # If it contains some kind of chars -> discard
        if word_tuple[0].__contains__("/"):
            return False

        # If it has only one char (e.g., "@", "#", "a")
        if len(word_tuple[0]) == 1:
            return False

        # Manual check ('https' is considered a noun)
        if word_tuple[0] == "https":
            return False

        # If ends with unicode HORIZONTAL ELLIPSIS char -> truncated word inside the dataset -> useless
        if word_tuple[0].__contains__(u"\u2026"):
            return False

        # If it is a number
        if word_tuple[0].isnumeric():
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

"""
Utility function used to sort the result list to indicate the timestamp as ordering parameter
"""
def customCompare(item):
    return int(item[0])

"""
Fuction invoked to process the given dataset
"""
def processDataset(dataset_path, DEBUG=False, nProcesses=4, aliases=None):

    # Reading the CSV dataset
    try:
        dataset = pd.read_csv(dataset_path)
    except Exception:
        dataset = pd.read_csv(dataset_path, encoding='latin-1')

    # Delete useless columns
    dataset = deleteUselessColumns(dataset)

    # Start the multithreading processing
    nRows = len(dataset.index)
    print("--- Processing {} entries ---".format(nRows))
    bar_reporter = find_reporter()

    processes = []
    pipes = []
    for i in range(nProcesses):

        start = int((nRows / nProcesses) * i)
        end = int((nRows / nProcesses) * (i + 1) - 1)
        if i == nProcesses:
            end = nRows

        parent_pipe, child_pipe = multiprocessing.Pipe()

        process = DatasetProcessor(int(i)+1, dataset.iloc[start:end+1], aliases, DEBUG, child_pipe, bar_reporter)

        process.start()
        processes.append(process)
        pipes.append(parent_pipe)

    # Release memory
    del dataset

    # Collecting partial results
    result = []
    debugResult = []

    for i in range(nProcesses):
        result = result + pipes[i].recv()
        if DEBUG:
            debugResult = debugResult + pipes[i].recv()

        processes[i].join()

    flush()

    print()
    print("--- Sorting the partial results ---")
    result.sort(key=customCompare)

    # Save the preprocessed dataset
    print("--- Saving the results ---")
    filename = getFilename(dataset_path)
    newDataset = pd.DataFrame(result)
    newDataset.to_csv("{}_preprocessed.csv".format(filename), index=False, header=False)

    if DEBUG:
        debug_df = pd.DataFrame(debugResult)
        debug_df.to_csv("{}_debug.csv".format(filename), index=False, header=False)

    print("--- Data preprocessing completed ---")


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

    # Parsing parameters
    dataset_path = sys.argv[1]
    DEBUG = False
    nProcesses = 4

    for i in range(2, len(sys.argv)):
        if sys.argv[i] == "--debug":
            DEBUG = True
        elif sys.argv[i] == "--install":
            nltk.download("punkt")
            nltk.download("averaged_perceptron_tagger")
            print("--- Downloaded nltk dependencies ---")
        elif sys.argv[i] == "--processes":
            if len(sys.argv) > i and is_integer(sys.argv[i + 1]):
                nProcesses = int(sys.argv[i + 1])

    # Fix terminal prints in windows for escape sequences used in atpbar
    if sys.platform.startswith("win"):
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

    # Start processing the dataset
    print(
        "Dataset preprocessor started with the following parameters:\n\t> dataset_path = {}\n\t> DEBUG = {}\n\t> N. processes = {}\n\t> Alias path = {}\n\t> N. aliases read = {}\n".format(
            dataset_path, DEBUG, nProcesses, ALIAS_PATH, len(aliasMap.keys())))
    processDataset(dataset_path, DEBUG=DEBUG, nProcesses=nProcesses, aliases=aliasMap)