import os
import sys
import json

if __name__ == "__main__":
    try:
        with open("./preprocessorSettings.json") as settings:
            data = json.load(settings)
            command = "cd ../src && python3 dataset-preprocessor.py {}".format(data["dataset_relative_path"])

            if "threads" in data.keys():
                command += " --threads {}".format(data["threads"])
            if "install_nltk_lib" in data.keys() and data["install_nltk_lib"]:
                command += " --install"
            if "debug" in data.keys() and data["debug"]:
                command += " --debug"
                print("RUNNING COMMAND: ", command, "\n")

            os.system(command)
    except KeyError:
        sys.stderr.write("ERROR: Some required fields not found in the settings file\n")
    except FileNotFoundError:
        sys.stderr.write("ERROR: the settings file has not been found\n")