"""
data_pipeline_gui.py
====================

This script provides a graphical user interface (GUI) using Tkinter to interact
with the `data_pipeline.py` script. The GUI allows users to select an input CSV
file, specify output file paths for the processed data and QA report, and run
the data pipeline by clicking a button.

Functions
---------
1. open_file_dialog() - Open dialog to select input CSV file
2. save_file_dialog() - Open dialog to select where to save output CSV file
3. save_qa_report_dialog() - Open dialog to select where to save QA report file
4. run_pipeline() - Run the data pipeline and handle errors

"""

import tkinter as tk
from tkinter import filedialog, messagebox
from pipeline import process_data_pipeline  # Importing the main pipeline function from data_pipeline.py


def open_file_dialog():
    """
    Open a file dialog for the user to select an input CSV file.

    The selected file path is displayed in the input entry field.
    """
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])  # Filter only .csv files
    if file_path:
        input_entry.delete(0, tk.END)  # Clear the current text in the input entry field
        input_entry.insert(0, file_path)  # Insert the selected file path into the entry field


def save_file_dialog():
    """
    Open a save dialog for the user to specify the output CSV file.

    The chosen file path is displayed in the output entry field.
    """
    file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                             filetypes=[("CSV files", "*.csv")])  # Save as .csv
    if file_path:
        output_entry.delete(0, tk.END)  # Clear the current text in the output entry field
        output_entry.insert(0, file_path)  # Insert the selected file path into the entry field


def save_qa_report_dialog():
    """
    Open a save dialog for the user to specify the QA report file.

    The chosen file path is displayed in the QA report entry field.
    """
    file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                             filetypes=[("Text files", "*.txt")])  # Save as .txt
    if file_path:
        qa_report_entry.delete(0, tk.END)  # Clear the current text in the QA report entry field
        qa_report_entry.insert(0, file_path)  # Insert the selected file path into the entry field


def run_pipeline():
    """
    Run the data pipeline based on user inputs.

    This function retrieves file paths from the entry fields, validates them, and
    runs the `process_data_pipeline` function. It also handles success or failure
    messages using message boxes.
    """
    # Retrieve file paths from the entry fields
    input_file = input_entry.get()
    output_file = output_entry.get()
    qa_report_file = qa_report_entry.get()

    # Validate that all file paths have been provided
    if not input_file or not output_file or not qa_report_file:
        messagebox.showerror("Error", "Please provide all file paths!")  # Show error message if any paths are missing
        return

    try:
        # Run the data pipeline with the provided file paths
        process_data_pipeline(input_file, output_file, qa_report_file)
        messagebox.showinfo("Success", "Data pipeline completed successfully!")  # Show success message if completed
    except Exception as e:
        # Show error message if the pipeline fails
        messagebox.showerror("Error", f"An error occurred: {e}")


# GUI setup
root = tk.Tk()  # Initialize the main GUI window
root.title("Data Pipeline")  # Set the title of the window

# Input CSV File
tk.Label(root, text="Input CSV File:").grid(row=0, column=0, padx=10, pady=10)  # Label for the input file
input_entry = tk.Entry(root, width=50)  # Entry field to display the selected input file path
input_entry.grid(row=0, column=1, padx=10, pady=10)  # Place the entry field in the grid
tk.Button(root, text="Browse", command=open_file_dialog).grid(row=0, column=2, padx=10,
                                                              pady=10)  # Browse button to select input file

# Output CSV File
tk.Label(root, text="Output CSV File:").grid(row=1, column=0, padx=10, pady=10)  # Label for the output file
output_entry = tk.Entry(root, width=50)  # Entry field to display the selected output file path
output_entry.grid(row=1, column=1, padx=10, pady=10)  # Place the entry field in the grid
tk.Button(root, text="Browse", command=save_file_dialog).grid(row=1, column=2, padx=10,
                                                              pady=10)  # Browse button to select output file

# QA Report File
tk.Label(root, text="QA Report File:").grid(row=2, column=0, padx=10, pady=10)  # Label for the QA report file
qa_report_entry = tk.Entry(root, width=50)  # Entry field to display the selected QA report file path
qa_report_entry.grid(row=2, column=1, padx=10, pady=10)  # Place the entry field in the grid
tk.Button(root, text="Browse", command=save_qa_report_dialog).grid(row=2, column=2, padx=10,
                                                                   pady=10)  # Browse button to select QA report file

# Run Pipeline Button
tk.Button(root, text="Run Pipeline", command=run_pipeline, width=20).grid(row=3, column=1, padx=10,
                                                                          pady=20)  # Button to run the pipeline

# Start the GUI loop
root.mainloop()  # Start the main event loop for the GUI
