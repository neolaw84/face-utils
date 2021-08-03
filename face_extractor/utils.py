def read_image_bytes(file_path):
    with open(file_path, "rb") as f:
        return f.read()

def get_model(file_path):
    with open(file_path, "rb") as f:
        return f.read()