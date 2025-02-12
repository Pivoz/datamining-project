from pyspark.sql import SparkSession
import sys
import ctypes
import os
from datetime import datetime
from atpbar import atpbar
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import size
import ast
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.backends.backend_pdf
import time

WEIGHT_LOW = 1
WEIGHT_MEDIUM = 3
WEIGHT_HIGH = 5

def make_report(topics_dict, report_filename, support_threshold, global_threshold, n_timeframes, max_topics_to_show):
    pdf = matplotlib.backends.backend_pdf.PdfPages(report_filename)
    figs = []

    # Parse topics_dict
    actual_key = 0
    n_keys_done = 0
    n_topics_already_considered = 0

    while n_keys_done < len(topics_dict.keys()) and n_topics_already_considered < max_topics_to_show:
        if topics_dict.__contains__(actual_key):
            values = topics_dict[actual_key]
            x_axis = []
            y_axis = []

            for tuple in values:
                x_axis.append(round(tuple[0], 2))           # Frequency
                y_axis.append(tuple_to_string(tuple[1]))    # Topic label

                # Increase number of considered topics and eventually stop
                n_topics_already_considered += 1
                if n_topics_already_considered == max_topics_to_show:
                    break

            plt.rcdefaults()
            fig, ax = plt.subplots()
            ax.barh(np.arange(len(y_axis)), x_axis, align='center', color='blue', ecolor='black')
            ax.set_yticks(np.arange(len(y_axis)))
            ax.set_yticklabels(y_axis)
            ax.set_title("Consistent topics in {} ({}%) timeframes\n\nSupport threshold: {} entries per timeframe\nN. timeframes threshold: {} timeframes".format(actual_key, int(actual_key/n_timeframes*100),support_threshold, global_threshold))
            plt.xlabel("Average frequency\n{} - Low frequent\n{} - High frequent".format(WEIGHT_LOW, WEIGHT_HIGH))
            plt.xlim([0, WEIGHT_HIGH])
            ax.invert_yaxis()
            figs.append(fig)

            n_keys_done += 1

        actual_key += 1

    for fig in figs:
        pdf.savefig(fig, bbox_inches='tight')
    pdf.close()

def tuple_to_string(in_tuple):
    string = str(in_tuple[0])
    for i in range(1, len(in_tuple)):
        string += ", {}".format(in_tuple[i])
    return string

def find_consistent_topics(frequencies_dict, min_timeframes_threshold, max_topics_to_show):
    ranking_dict = {}
    max_key = 0

    # Identify valuable consistent topics in time
    for key in frequencies_dict.keys():
        n_of_non_none = sum(x is not None for x in frequencies_dict[key])
        if n_of_non_none >= min_timeframes_threshold:
            # Add the empty entry if it does not exists
            if not ranking_dict.keys().__contains__(n_of_non_none):
                ranking_dict[n_of_non_none] = []

            # Compute the mean weight
            sum_weights = 0
            for entry in frequencies_dict[key]:
                if entry is not None:
                    sum_weights += entry
            weight_mean = sum_weights / n_of_non_none

            ranking_dict_value = ranking_dict[n_of_non_none]
            ranking_dict_value.append((weight_mean, key))
            ranking_dict[n_of_non_none] = ranking_dict_value

            # Update max_key
            if n_of_non_none > max_key:
                max_key = n_of_non_none

    # Generate the ranking
    truncated = []
    count = 0
    n_topics_already_considered = 0
    actual_key = min_timeframes_threshold
    while actual_key <= max_key and n_topics_already_considered < max_topics_to_show:
        if ranking_dict.keys().__contains__(actual_key):
            # Sort the entries
            values = ranking_dict[actual_key]
            values.sort(key=compare_function_for_frequent_topics, reverse=True)

            # Add to the final list applying variation of closed itemsets
            new_values = []
            for topic_tuple in values:
                superset_found = False

                for topic_to_compare in values:
                    if len(topic_to_compare[1]) > len(topic_tuple[1]) and is_superset(topic_tuple[1], topic_to_compare[1]):
                        superset_found = True

                if not superset_found:
                    new_values.append(topic_tuple)
                    count += 1

                    # Increase number of considered topics and eventually stop
                    n_topics_already_considered += 1
                    if n_topics_already_considered == max_topics_to_show:
                        break
                else:
                    truncated.append((topic_tuple[1], actual_key, topic_tuple[0]))

            ranking_dict[actual_key] = new_values

        actual_key += 1

    return ranking_dict, count, truncated

def is_superset(actual_tuple, potential_superset):
    for item in actual_tuple:
        if not potential_superset.__contains__(item):
            return False

    return True

def compare_function_for_frequent_topics(item):
    return item[0]


def find_frequent_topics(timeframes_limits, tweets_dataset, support_threshold):
    num_timeframes = len(timeframes_limits)
    bar_iterator = iter(atpbar(range(num_timeframes), name="Computing frequent itemsets per timeframe"))
    tweets_dataset = tweets_dataset.select("_c1").collect()
    tf_id = 0

    frequent_itemsets_dict = {}

    for start, end in timeframes_limits:
        tweets_in_timeframe = tweets_dataset[start:end+1]

        df_tweets_in_timeframe = [(tf_id, ast.literal_eval(item[0])) for item in tweets_in_timeframe]
        df = spark.createDataFrame(df_tweets_in_timeframe, schema=["tf_id", "tweets"])

        fpGrowth = FPGrowth(itemsCol="tweets", minSupport=0.01, minConfidence=0.6)
        model = fpGrowth.fit(df)
        freq = model.freqItemsets
        freq = freq.filter((size(freq.items) >= 2) & (freq.freq >= support_threshold))

        # Add to the global_dict
        max_freq = freq.select("freq").groupby().max("freq").first()["max(freq)"]

        for item in freq.collect():
            key = tuple(item["items"])
            if not frequent_itemsets_dict.keys().__contains__(key):
                frequent_itemsets_dict[key] = [None for i in range(num_timeframes)]

            # Update the frequencies weight
            values = frequent_itemsets_dict[key]
            values[tf_id] = (4 * item["freq"] / max_freq) + 1
            frequent_itemsets_dict[key] = values

        # Let proceed the bar
        bar_step(bar_iterator)

        tf_id += 1

    # Final step of the bar
    bar_step(bar_iterator)

    return frequent_itemsets_dict

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
                debug_file.write("Final timestamp: \t{} ({})\nN. items: {}\n\n".format(previous_timestamp, datetime.fromtimestamp(previous_timestamp), end-start))

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
            debug_file.write("Final timestamp: \t{} ({})\nN. items: {}\n\n".format(previous_timestamp, datetime.fromtimestamp(previous_timestamp), end-start))

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
    output_file_report = "report.pdf"   # Default

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
            elif item == "--output":
                output_file_report = next(arg_iter).lower()
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

    print("Topic finder started with the following parameters:\n\t> Dataset path: \t{}\n\t> Time frame: \t\t{} {}\n\t> Timespan threshold: \t{}\n\t> Global threshold: \t{}\n\t> Max topics to show: \t{}\n\t> Output report file: \t{}\n\t> Debug: \t\t{}\n"
        .format(dataset_path, timespan, timeunit, timespan_threshold, global_threshold, max_topics_to_show, output_file_report, debug))

    # Start Spark session
    spark = SparkSession \
        .builder \
        .appName("Data mining project - Davide Piva") \
        .master("local[*]") \
        .config("spark.ui.showConsoleProgress", "false") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    dataset = spark.read.load(dataset_path, format="csv", header="false")

    # Split dataset items into different lists with respect to the provided timeframe
    print("--- Splitting the dataset by timeframe ---")
    timeframes_limits = split_dataset_by_timeframe(dataset.select("_c0").cache(), timespan, timeunit, debug)
    n_timeframes = len(timeframes_limits)
    print("--- Found {} timeframes ---".format(n_timeframes))

    # Find frequent itemsets/topics for each timeframe
    print("--- Searching for frequent topics ---")
    frequencies_dict = find_frequent_topics(timeframes_limits, dataset.select("_c1").cache(), timespan_threshold)
    print("--- Found {} frequent topics ---".format(len(frequencies_dict.keys())))

    # Find consistent topics in time
    print("--- Searching for consistent topics in time ---")
    consistent_topics_in_time_dict, count, truncated = find_consistent_topics(frequencies_dict, global_threshold, max_topics_to_show)
    print("--- Found {} consistent topics in time (truncated {} topics using closed itemsets technique) ---".format(count, len(truncated)))

    # Produce final report of findings
    print("--- Producing the final report ---")
    make_report(consistent_topics_in_time_dict, output_file_report, timespan_threshold, global_threshold, n_timeframes, max_topics_to_show)
    time.sleep(2)
    print("--- Report saved ---")

    # Finish the execution of Spark
    print("--- Execution terminated correctly ---")
    spark.stop()
