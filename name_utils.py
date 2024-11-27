import os


def add_name(name):
    users_file = "usersdb.txt"

    # Determine the next person_id
    if not os.path.exists(users_file):
        with open(users_file, "w") as user_file:
            pass

    with open(users_file, "r") as user_file:
        lines = user_file.readlines()
        next_id = len(lines) + 1

    # Append to names file
    with open(users_file, "a") as user_file:
        user_file.write(f"{next_id},{name}\n")

    return next_id


def get_name_by_id(person_id):
    users_file = "usersdb.txt"

    with open(users_file, "r") as user_file:
        for line in user_file:
            key_value = line.strip().split(',')
            if len(key_value) == 2 and int(key_value[0]) == person_id:
                return key_value[1]

    return "Not found"
