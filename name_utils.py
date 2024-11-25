import os


def add_name(name):
    names_file = "usersdb.txt"

    # Determine the next ID
    if not os.path.exists(names_file):
        with open(names_file, "w") as f:
            pass

    with open(names_file, "r") as f:
        lines = f.readlines()
        next_id = len(lines) + 1

    # Append to names file
    with open(names_file, "a") as f:
        f.write(f"{next_id},{name}\n")

    return next_id


def get_name_by_id(ID):
    names_file = "usersdb.txt"

    with open(names_file, "r") as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 2 and int(parts[0]) == ID:
                return parts[1]

    return "Unknown"
