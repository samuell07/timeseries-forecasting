import pickle
import os

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

def load_model(model_folder, data_set_folder = None): 
    combined_binary_data = b''
    folder = "."
    if data_set_folder:
        folder += f"/{data_set_folder}"
    else:
        folder += "."
    folder += f"/stored_models/{model_folder}"
    files = os.listdir(folder)
    for file_path in sorted(files):
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