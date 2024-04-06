import pickle
def load_model(model_folder): 
    combined_binary_data = b''
    for file_path in ["../stored_models/final_model1", "../stored_models/final_model2", "../stored_models/final_model3"]:
        with open(file_path, 'rb') as file:
            binary_data = file.read()
            combined_binary_data += binary_data

    try:
        model = pickle.loads(combined_binary_data)
        print("Object loaded successfully.")
    except pickle.UnpicklingError as e:
        print("Error while loading the combined binary data:", e)
    return model