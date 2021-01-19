import json
import os
import sys

if __name__ == "__main__":
    try :
        with open("./topicFinderSettings.json") as settings:
            data = json.load(settings)

            # Build the command to execute
            command = "cd ../src && python3 topic-finder.py {} --timespan {} --timeunit {} --timespan-threshold {} --global-threshold {}".format(
                data["dataset_relative_path"], data["timespan"], data["timeunit"], data["timespan_threshold"], data["global_threshold"])

            # Adding optional fields
            if "max_topics_to_show" in data.keys():
                command += " --show {}".format(data["max_topics_to_show"])
            if "processes" in data.keys():
                command += " --processes {}".format(data["processes"])
            if "debug" in data.keys() and data["debug"]:
                command += " --debug"
                print("RUNNING COMMAND: ", command, "\n")

            # Execute
            os.system(command)
    except KeyError:
        sys.stderr.write("ERROR: Some required fields not found in the settings file\n")
    except FileNotFoundError:
        sys.stderr.write("ERROR: the settings file has not been found\n")