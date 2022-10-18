TARGET = None

def log(s):
    print("test2: " + s)

def update_fn(new_value):
    global TARGET
    TARGET = new_value
    log(f"update TARGET to {TARGET}")

def print_fn():
    log(f"TARGET {TARGET}")