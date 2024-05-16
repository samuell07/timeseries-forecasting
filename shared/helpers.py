import pickle
import os
import re
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dateutil import parser

def store_model(model, model_name, write_size=50000000):
    partnum = 0
    binary = pickle.dumps(model)
    folder = f"../stored_models/{model_name}"
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i in range(0, len(binary), write_size): 
        chunk = binary[i:i+write_size]
        partnum += 1
        filename = f"{folder}/model_part{partnum}"
        with open(filename, 'wb') as dest_file: 
            dest_file.write(chunk) 

def sort_key(file_path):
    match = re.search(r'(\d+)', file_path)
    if match:
        return int(match.group(1))
    return 0 

def load_model(model_folder, data_set_folder = None): 
    combined_binary_data = b''
    folder = "."
    if data_set_folder:
        folder += f"/{data_set_folder}"
    else:
        folder += "."
    folder += f"/stored_models/{model_folder}"
    files = os.listdir(folder)
    sorted_files = sorted(files, key=sort_key)
    for file_path in sorted_files:
        if not file_path.startswith("model_part"):
            continue
        print(f"Loading {file_path} in folder {folder}")
        with open(f"{folder}/{file_path}", 'rb') as file:
            binary_data = file.read()
            combined_binary_data += binary_data

    try:
        model = pickle.loads(combined_binary_data)
        print("Object loaded successfully.")
    except pickle.UnpicklingError as e:
        print("Error while loading the combined binary data:", e)
    return model

def store_predictions_and_create_graph(model_name, dates, predictions, real, dataset_name):
    json_file_path = "results.json"
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as file:
            data = json.load(file)
    else:
        data = {}
    if "real" not in data:
        data["real"] = []
        for date, value in zip(dates, real):
            data["real"].append({"date": date, "value": value})
    data[model_name] = []
    
    for date, prediction in zip(dates, predictions):
        data[model_name].append({"date": date, "value": prediction})

    with open(json_file_path, 'w') as file:
        json.dump(data, file, indent=4, default=str)
    _, ax = plt.subplots(1, 1, figsize=(1280 / 96, 720 / 96))
    with open(json_file_path, 'r') as file:
            data = json.load(file)
    for model, model_predictions in data.items():
        model_dates = [parser.parse(entry['date']) for entry in model_predictions]
        model_values = [entry['value'] for entry in model_predictions]
        ax.plot(model_dates, model_values, label=model)
    
    ax.set_title(f"{dataset_name} predictions")
    ax.set_ylabel("Value")
    ax.set_xlabel("Time")
    ax.legend()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.savefig(f"{dataset_name}_all_models.png")
