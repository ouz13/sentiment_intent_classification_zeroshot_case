import json
import yaml


def load_yaml(file_name):
    try:
        with open(file_name, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print("The file was not found.")
    except yaml.YAMLError as e:
        print("Error decoding YAML.")


def load_json(file_name):
    try:
        with open('../data/' + file_name, 'r') as file:
            return json.load(file)['conversation']
    except FileNotFoundError:
        print("The file was not found.")
    except json.JSONDecodeError:
        print("Error decoding JSON.")

