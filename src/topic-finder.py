import sys
import os
import pandas as pd
from datetime import datetime
from atpbar import atpbar, register_reporter, find_reporter, flush
import ctypes
import multiprocessing
import ast

class TimeframeTopicsFinderProcess(multiprocessing.Process):
    def __init__(self, id, pipe_conn, timeframes, timespan_threshold, offset, bar_reporter):
        super().__init__()

        self.id = id
        self.pipe_conn = pipe_conn
        self.timeframes = timeframes
        self.threshold = timespan_threshold
        self.offset = offset
        self.bar_reporter = bar_reporter

    def run(self):
        final_result = []
        register_reporter(self.bar_reporter)

        for i in range(len(self.timeframes)):
            actual_timeframe = self.timeframes[i]

            # Create the progress bar - frequent words
            bar_iterator = iter(atpbar(range(len(actual_timeframe)), name="Process {} - Timeframe {} of {} - Computing frequent words".format(self.id, i+1, len(self.timeframes))))

            # PASS 1 - Count single words frequencies
            words_frequencies = {}
            for j in range(len(actual_timeframe)):
                word_list = ast.literal_eval(actual_timeframe[j])

                for k in range(len(word_list)):
                    entry = words_frequencies.get(word_list[k], 0)     # 0 as default value
                    words_frequencies[word_list[k]] = entry + 1

                # Let bar to progress
                try:
                    next(bar_iterator)
                except StopIteration:
                    pass

            # Final bar progress
            try:
                next(bar_iterator)
            except StopIteration:
                pass

            # Remove not frequent words and free unused memory
            frequent_words = {j:words_frequencies[j] for j in words_frequencies if words_frequencies[j] >= self.threshold}
            frequent_words_list = list(frequent_words.keys())
            frequent_words_list.sort()

            del words_frequencies
            del frequent_words

            # Allocate triangular matrix
            matrix = []
            for j in range(len(frequent_words_list)):
                column = []
                for k in range(j, 0, -1):
                    column.append(0)
                matrix.append(column)

            # PASS 2 - Compute frequent pairs
            for j in range(len(actual_timeframe)):
                tweet = ast.literal_eval(actual_timeframe[j])

                # Generate all the possible pairs
                for k in range(len(tweet)-1):
                    for l in range(k+1, len(tweet)):
                        try:
                            first_index = frequent_words_list.index(tweet[k])
                            second_index = frequent_words_list.index(tweet[l])
                        except ValueError:
                            # One of the two words was not considered frequent
                            continue

                        if second_index > first_index:
                            temp = first_index
                            first_index = second_index
                            second_index = temp

                        matrix[first_index][second_index] += 1

            #if self.id == 1 and i == 0:
            #    print(numbered_words)
            #    for j in range(len(matrix)):
            #        for k in range(len(matrix[j])):
            #            sys.stdout.write("{}\t".format(matrix[j][k]))
            #        print()

            frequent_pairs = []
            for j in range(len(matrix)):
                for k in range(len(matrix[j])):
                    if matrix[j][k] >= self.threshold:
                        pair = (frequent_words_list[j], frequent_words_list[k], matrix[j][k])
                        frequent_pairs.append(pair)

            timeframe_entry = (self.offset + i, frequent_pairs)
            final_result.append(timeframe_entry)

        self.pipe_conn.send(final_result)
        self.pipe_conn.close()


def timeframe_to_timestamp(timespan, timeunit):
    if timeunit == "day":
        return timespan * 86400
    if timeunit == "hour":
        return timespan * 3600

def is_in_same_timeframe(base_timestamp, timestamp_test, timespan, timeunit):
    offset = timeframe_to_timestamp(timespan, timeunit)
    return timestamp_test <= base_timestamp + offset

def split_dataset_by_timeframe(dataset_path, timespan, timeunit, debug):
    split_dataset_to_return = []
    dataset = pd.read_csv(dataset_path, header=None)
    nRows = len(dataset.index)

    barIterator = iter(atpbar(range(nRows), name="Split dataset process"))

    # For debug
    if debug:
        debug_file = open("split_dataset_timeframe_debug.txt", "w")

    base_timeframe_item = None
    bucket = []
    previous_timestamp = None
    for index, row in dataset.iterrows():

        actualTimestamp = row[0]
        actualTweet = row[1]

        if base_timeframe_item is None:
            base_timeframe_item = actualTimestamp
            bucket.append(actualTweet)
            previous_timestamp = actualTimestamp

            if debug:
                debug_file.write("Initial timestamp: \t{} ({})\n".format(actualTimestamp, datetime.fromtimestamp(actualTimestamp)))
        elif is_in_same_timeframe(base_timeframe_item, actualTimestamp, timespan, timeunit):
            bucket.append(actualTweet)
            previous_timestamp = actualTimestamp
        else:
            if debug:
                debug_file.write("Final timestamp: \t{} ({})\nN. items: {}\n\n".format(previous_timestamp, datetime.fromtimestamp(previous_timestamp), len(bucket)))

            split_dataset_to_return.append(bucket)
            bucket = []
            bucket.append(actualTweet)
            base_timeframe_item = actualTimestamp
            previous_timestamp = actualTimestamp

            if debug:
                debug_file.write("Initial timestamp: \t{} ({})\n".format(actualTimestamp, datetime.fromtimestamp(actualTimestamp)))

        # Let proceed the bar
        try :
            next(barIterator)
        except StopIteration:
            pass

    # Final step of the bar
    try:
        next(barIterator)
    except StopIteration:
        pass

    if len(bucket) > 0:
        split_dataset_to_return.append(bucket)
        if debug:
            debug_file.write("Final timestamp: \t{} ({})\nN. items: {}\n\n".format(previous_timestamp, datetime.fromtimestamp(previous_timestamp), len(bucket)))

    if debug:
        debug_file.close()

    return split_dataset_to_return


def check_arguments_values(timespan, timeunit, timespan_threshold, global_threshold):
    # Simple value check
    if timespan_threshold is None or global_threshold is None:
        return False
    elif timespan <= 0 or timespan_threshold <= 0 or global_threshold <= 0:
        return False
    elif timeunit != "day" and timeunit != "hour":
        return False

    return True

def dump_frequent_topics(timeframes_topics, filename):
    file = open(filename, "w")

    for timeframes_set in timeframes_topics:
        for single_timeframe in timeframes_set:
            file.write("TIMEFRAME {}\n".format(single_timeframe[0]))
            for tuple in single_timeframe[1]:
                to_write = "{} - {} - {}\n".format(tuple[0], tuple[1], tuple[2])
                try:
                    file.write(to_write)
                except Exception:
                    sys.stderr.write("ERROR WRITING -{}-".format(to_write))

            file.write("\n\n")
    file.close()


if __name__ == "__main__":
    # Inline arguments retrieval
    if len(sys.argv) <= 1 or sys.argv[1].lower() == "--help":
        sys.stderr.write("USAGE: python3 topic-finder.py <path-to-CSV dataset> --timespan <> --timeunit <> --timespan-threshold <> --global-threshold <> [(--debug), (--show <max-items>)]")
        exit(1)

    if not os.path.exists(sys.argv[1]):
        sys.stderr.write("ERROR: the provided dataset file does not exist")
        exit(2)

    dataset_path = sys.argv[1]
    timespan = None
    timeunit = None
    timespan_threshold = None
    global_threshold = None
    debug = False
    max_topics_to_show = sys.maxsize
    n_processes = 4

    # Parsing inline arguments
    arg_iter = iter(sys.argv)
    next(arg_iter)              # Script name to jump
    next(arg_iter)              # Dataset path already read
    while True:
        try:
            item = next(arg_iter).lower()

            if item == "--timespan":
                timespan = int(next(arg_iter))
            elif item == "--timeunit":
                timeunit = next(arg_iter).lower()
            elif item == "--timespan-threshold":
                timespan_threshold = int(next(arg_iter))
            elif item == "--global-threshold":
                global_threshold = int(next(arg_iter))
            elif item == "--debug":
                debug = True
            elif item == "--show":
                max_topics_to_show = int(next(arg_iter))
            elif item == "--processes":
                n_processes = int(next(arg_iter))
        except ValueError:
            sys.stderr.write("ERROR: wrong integer conversion of inline arguments\n")
            exit(3)
        except Exception:
            break

    # Check the inserted values
    if not check_arguments_values(timespan, timeunit, timespan_threshold, global_threshold):
        sys.stderr.write("ERROR: the provided arguments are not valid. See --help for further information\n")
        exit(4)

    # Fix terminal prints in windows for escape sequences used in atpbar
    if sys.platform.startswith("win"):
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

    print("Topic finder started with the following parameters:\n\t> Dataset path: \t{}\n\t> Time frame: \t\t{} {}\n\t> Timespan threshold: \t{}\n\t> Global threshold: \t{}\n\t> Max topics to show: \t{}\n\t> N. processes: \t{}\n\t> Debug: \t\t{}\n"
          .format(dataset_path, timespan, timeunit, timespan_threshold, global_threshold, max_topics_to_show, n_processes, debug))

    # Split dataset items into different lists with respect to the provided time frame
    print("--- Splitting the dataset by timeframe ---")
    split_dataset = split_dataset_by_timeframe(dataset_path, timespan, timeunit, debug)
    nBuckets = len(split_dataset)
    print("--- Found {} timeframes ---\n".format(nBuckets))

    # Find frequent topics inside each time frame
    pipe_connections = []
    processes = []
    bar_reporter = find_reporter()
    for i in range(n_processes):
        start = int((nBuckets / n_processes) * i)
        end = int((nBuckets / n_processes) * (i + 1) - 1)
        if i == n_processes:
            end = nBuckets-1

        parent_conn, child_conn = multiprocessing.Pipe()
        process = TimeframeTopicsFinderProcess(i+1, child_conn, split_dataset[start:end+1], timespan_threshold, start, bar_reporter)
        process.start()

        processes.append(process)
        pipe_connections.append(parent_conn)

    frequent_timeframe_topics = []
    for i in range(n_processes):
        frequent_timeframe_topics.append(pipe_connections[i].recv())
        processes[i].join()

    flush()

    if debug:
        dump_frequent_topics(frequent_timeframe_topics, "./frequent_pairs_found.txt")

    #for timeframe_topics_list in frequent_timeframe_topics:
    #    for single_timeframe in timeframe_topics_list:
    #        print(single_timeframe)
    #        print()