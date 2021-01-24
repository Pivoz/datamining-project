from pyspark.sql import SparkSession
import sys
import ctypes
import os
from datetime import datetime
from atpbar import atpbar
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import size
import ast

WEIGHT_LOW = 1
WEIGHT_MEDIUM = 3
WEIGHT_HIGH = 5

def find_consistent_topics(frequent_topics_per_timeframe, frequencies_dict, min_timeframes_threshold):
    raise NotImplemented()


def find_frequent_topics(timeframes_limits, tweets_dataset, support_threshold):
    frequent_topics_per_timeframe = []
    num_timeframes = len(timeframes_limits)
    bar_iterator = iter(atpbar(range(num_timeframes), name="Computing frequent itemsets per timeframe"))
    tweets_dataset = tweets_dataset.select("_c1").collect()
    tf_id = 0

    frequent_itemsets_dict = {}

    for start, end in timeframes_limits:
        tweets_in_timeframe = tweets_dataset[start:end+1]

        df_tweets_in_timeframe = [(tf_id, ast.literal_eval(item[0])) for item in tweets_in_timeframe]
        df = spark.createDataFrame(df_tweets_in_timeframe, schema=["tf_id", "tweets"])

        fpGrowth = FPGrowth(itemsCol="tweets", minSupport=0.01, minConfidence=0.001)
        model = fpGrowth.fit(df)
        freq = model.freqItemsets
        freq = freq.filter((size(freq.items) >= 2) & (freq.freq >= support_threshold))
        frequent_topics_per_timeframe.append(freq.select("items"))

        # Add to the global_dict
        max_freq = freq.select("freq").groupby().max("freq").first()["max(freq)"]

        for item in freq.collect():
            key = tuple(item["items"])
            if not frequent_itemsets_dict.keys().__contains__(key):
                frequent_itemsets_dict[key] = [None for i in range(num_timeframes)]

            # Update the frequencies
            values = frequent_itemsets_dict[key]
            freq_ratio = item["freq"] / max_freq

            if freq_ratio >= 0.8:
                freq_weight = WEIGHT_HIGH
            elif freq_ratio < 0.3:
                freq_weight = WEIGHT_LOW
            else:
                freq_weight = WEIGHT_MEDIUM

            values[tf_id] = freq_weight
            frequent_itemsets_dict[key] = values

        # Let proceed the bar
        bar_step(bar_iterator)

        tf_id += 1

    # Final step of the bar
    bar_step(bar_iterator)

    for key in frequent_itemsets_dict.keys():
        print(key, " ---- ", frequent_itemsets_dict[key])

    return frequent_topics_per_timeframe, frequent_itemsets_dict

def bar_step(bar_iterator):
    try:
        next(bar_iterator)
    except StopIteration:
        pass

def timeframe_to_timestamp(timespan, timeunit):
    if timeunit == "day":
        return timespan * 86400
    if timeunit == "hour":
        return timespan * 3600

def is_in_same_timeframe(base_timestamp, timestamp_test, timespan, timeunit):
    offset = timeframe_to_timestamp(timespan, timeunit)
    return timestamp_test <= base_timestamp + offset

def split_dataset_by_timeframe(dataset, timespan, timeunit, debug):
    bar_iterator = iter(atpbar(range(dataset.count()), name="Split dataset process"))

    # For debug
    if debug:
        debug_file = open("split_dataset_timeframe_debug.txt", "w")

    base_timeframe_item = None
    bucket = []
    previous_timestamp = None
    timeframe_number = 0

    start = 0
    end = 0
    timeframes = []
    for row in dataset.select("_c0").collect():

        actualTimestamp = int(row[0])

        if base_timeframe_item is None:
            base_timeframe_item = actualTimestamp
            previous_timestamp = actualTimestamp

            if debug:
                debug_file.write("Timeframe {}\nInitial timestamp: \t{} ({})\n".format(timeframe_number,actualTimestamp, datetime.fromtimestamp(actualTimestamp)))
                timeframe_number += 1
        elif is_in_same_timeframe(base_timeframe_item, actualTimestamp, timespan, timeunit):
            previous_timestamp = actualTimestamp
            end += 1
        else:
            if debug:
                debug_file.write("Final timestamp: \t{} ({})\nN. items: {}\n\n".format(previous_timestamp, datetime.fromtimestamp(previous_timestamp), len(bucket)))

            timeframes.append((start, end))
            start = end + 1
            end += 1
            base_timeframe_item = actualTimestamp
            previous_timestamp = actualTimestamp

            if debug:
                debug_file.write("Timeframe {}\nInitial timestamp: \t{} ({})\n".format(timeframe_number, actualTimestamp, datetime.fromtimestamp(actualTimestamp)))
                timeframe_number += 1

        # Let proceed the bar
        bar_step(bar_iterator)

    # Final step of the bar
    bar_step(bar_iterator)

    if start != end:
        timeframes.append((start, end))
        if debug:
            debug_file.write("Final timestamp: \t{} ({})\nN. items: {}\n\n".format(previous_timestamp, datetime.fromtimestamp(previous_timestamp), len(bucket)))

    if debug:
        debug_file.close()

    return timeframes

def check_arguments_values(timespan, timeunit, timespan_threshold, global_threshold):
    # Simple value check
    if timespan_threshold is None or global_threshold is None:
        return False
    elif timespan <= 0 or timespan_threshold <= 0 or global_threshold <= 0:
        return False
    elif timeunit != "day" and timeunit != "hour":
        return False

    return True

def compute(item):
    print(item)
    return True

if __name__ == "__main__":

    # Inline arguments retrieval
    if len(sys.argv) <= 1 or sys.argv[1].lower() == "--help":
        sys.stderr.write(
            "USAGE: python3 topic-finder.py <path-to-CSV dataset> --timespan <> --timeunit <> --timespan-threshold <> --global-threshold <> [(--debug), (--show <max-items>)]")
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
    next(arg_iter)  # Script name to jump
    next(arg_iter)  # Dataset path already read
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

    # Start Spark session
    spark = SparkSession \
        .builder \
        .appName("Data mining project - Davide Piva") \
        .config("spark.ui.showConsoleProgress", "false") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    dataset = spark.read.load(dataset_path, format="csv", header="false")
    pandas_dataset = dataset.toPandas()

    # Split dataset items into different lists with respect to the provided timeframe
    print("--- Splitting the dataset by timeframe ---")
    timeframes_limits = split_dataset_by_timeframe(dataset.select("_c0").cache(), timespan, timeunit, debug)
    nBuckets = len(timeframes_limits)
    print("--- Found {} timeframes ---\n".format(nBuckets))

    # Find frequent itemsets/topics for each timeframe
    frequent_topics_per_timeframe, frequencies_dict = find_frequent_topics(timeframes_limits, dataset.select("_c1").cache(), timespan_threshold)

    # Find consistent topics in time
    # consistent_topics_in_time = find_consistent_topics(frequent_topics_per_timeframe, frequencies_dict, global_threshold)

    spark.stop()