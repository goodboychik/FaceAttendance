import threading
import tkinter as tk
from tkinter import messagebox
import os
import csv
from datetime import datetime
import face_capture
import face_trainer
import face_recognition


def start_face_recognition(file_name):
    recognition_thread = threading.Thread(target=face_recognition.recognize_and_attendance, args=(file_name,))
    recognition_thread.daemon = True
    recognition_thread.start()


def end_attendance(window, file_name):
    # Move attendance log to CSV
    if os.path.exists(file_name + "_log.txt"):
        with open(file_name + "_log.txt", "r") as log_file, \
                open(file_name + ".csv", "w", newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Name", "Timestamp"])

            for line in log_file:
                name, timestamp = line.strip().split(",")
                csv_writer.writerow([name, timestamp])

        os.remove(file_name + "_log.txt")
        messagebox.showinfo("Success", f"Attendance saved to {file_name}")
    else:
        messagebox.showwarning("Warning", "No attendance records found.")

    window.destroy()


class AttendanceManagementSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Attendance Management System")
        self.root.geometry("400x300")

        # Create directories if they don't exist
        os.makedirs("photos", exist_ok=True)
        os.makedirs("traindata", exist_ok=True)

        # Create Main Menu
        self.create_menu()

    def create_menu(self):
        # Add New Person Button
        add_person_btn = tk.Button(self.root, width=20, height=2,
                                   text="Add New Person", font="15", command=self.add_new_person)
        add_person_btn.pack(pady=20)

        # Take Attendance Button
        take_attendance_btn = tk.Button(self.root, width=20, height=2, font="15",
                                        text="Take Attendance", command=self.take_attendance)
        take_attendance_btn.pack(pady=20)

        # Close Button
        take_attendance_btn = tk.Button(self.root, width=20, height=2, font="15", text="Close", command=self.close_app)
        take_attendance_btn.pack(pady=20)

    def close_app(self):
        self.root.destroy()

    def add_new_person(self):
        # Open a new window for adding a person
        add_person_window = tk.Toplevel(self.root)
        add_person_window.title("Add New Person")
        add_person_window.geometry("400x200")

        # Enter name
        tk.Label(add_person_window, text="Enter Name:", font=("Arial", 12)).pack(pady=10)
        name_entry = tk.Entry(add_person_window, font=("Arial", 12))
        name_entry.pack(pady=10)

        def capture_and_train():
            name = name_entry.get()
            if name:
                face_capture.capture_faces(name)  # Pass name to face_capture
                face_trainer.train_recognizer()
                messagebox.showinfo("Success", f"New person '{name}' added and trained successfully!")
                add_person_window.destroy()
            else:
                messagebox.showerror("Error", "Name cannot be empty!")

        tk.Button(add_person_window, text="Capture & Train", command=capture_and_train, font=("Arial", 12)).pack(
            pady=20)

    def take_attendance(self):
        # Open attendance taking window
        attendance_window = tk.Toplevel(self.root)
        attendance_window.title("Take Attendance")
        attendance_window.geometry("400x200")

        filename = "attendance(" + datetime.now().strftime("%d-%m-%Y_%H-%M") + ")"

        # Start attendance button
        start_btn = tk.Button(attendance_window, text="Start Attendance", width=20, height=2, font="15",
                              command=lambda: start_face_recognition(filename))
        start_btn.pack(pady=20)

        # End attendance button
        end_btn = tk.Button(attendance_window, text="End Attendance", width=20, height=2, font="15",
                            command=lambda: end_attendance(attendance_window, filename))
        end_btn.pack(pady=20)


def main():
    root = tk.Tk()
    AttendanceManagementSystem(root)
    root.mainloop()


if __name__ == "__main__":
    main()
