import csv
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from tkinter import ttk
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score


class PandasModel:
    def __init__(self, df):
        self.df = df

    def getColumnNames(self):
        return list(self.df.columns)

    def getData(self):
        return self.df.values.tolist()


class DataFrameTable(tk.Frame):
    def __init__(self, parent, dataframe):
        tk.Frame.__init__(self, parent)
        self.dataframe = dataframe

        self.treeview = ttk.Treeview(self, columns=dataframe.getColumnNames(), show="headings")
        self.treeview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Set up vertical scrollbar
        yscrollbar = ttk.Scrollbar(self, orient="vertical", command=self.treeview.yview)
        yscrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.treeview.configure(yscrollcommand=yscrollbar.set)

        self.populate()

    def populate(self):
        # Add column names to the treeview
        for column in self.dataframe.getColumnNames():
            self.treeview.heading(column, text=column)
            self.treeview.column(column, width=80)

        # Add data rows to the treeview
        for row in self.dataframe.getData():
            self.treeview.insert("", tk.END, values=row)


window = tk.Tk()
window.config(bg='white')

# Set the window size
window.geometry("1265x750")

# Load the image using PIL
image = Image.open("logo.png")

# Create a PhotoImage object using the image
photo = ImageTk.PhotoImage(image)

# Create a label to display the image
label = tk.Label(image=photo, bg='white')
label.pack()

custom_font = ("Arial", 16, "bold")
custom_font1 = ("Arial", 12, "bold")
custom_font2 = ("Arial", 12)
custom_font3 = ("Arial", 10, "bold")
custom_font4 = ("Arial", 14, "bold")

csv_file = None
csv_file_path = "Predict Price of Airline Tickets.csv"
data = None
data_df = pd.read_csv(csv_file_path,  parse_dates=True, infer_datetime_format=True)
print(data_df.head())


def upload_csv():
    global csv_file_path
    csv_file_path = filedialog.askopenfilename()
    csv_label.config(text=f"Number of Rows: {len(read_csv(csv_file_path)) - 1}")


def read_csv(file):
    # NUMBER
    global data
    with open(file, 'r') as f:
        reader = csv.reader(f)
        data = [row for row in reader]
    return data


def create_dataset():
    for widget in window.winfo_children():
        widget.destroy()

    # Create a PandasModel instance
    model = PandasModel(data_df)

    # Create a DataFrameTable instance
    table = DataFrameTable(window, model)

    table.grid(column=1, row=0, sticky="nsew")
    create_menu()


def create_menu():
    global data_df
    # Left Frame
    left_frame = ttk.Frame(window, padding=10)
    left_frame.grid(row=0, column=0, sticky="nsew")
    left_frame.columnconfigure(0, weight=1)

    # Data Wrangling
    data_wrangling_label = ttk.Label(left_frame, text="Data Wrangling", font=custom_font1)
    data_wrangling_label.grid(row=0, column=0, pady=5, sticky="ew")
    drop_na_button = ttk.Button(left_frame, text="Drop NA", padding=4, style="Custom.TButton", command=drop_na)
    drop_na_button.grid(row=1, column=0, pady=5, sticky="ew")
    remove_dup_button = ttk.Button(left_frame, text="Remove Duplicates", padding=4, style="Custom.TButton",command=remove_duplicates)
    remove_dup_button.grid(row=2, column=0, pady=5, sticky="ew")
    check_button = ttk.Button(left_frame, text="Check Consistency", padding=4, style="Custom.TButton",command=check_consistency)
    check_button.grid(row=3, column=0, pady=5, sticky="ew")
    feature_button = ttk.Button(left_frame, text="Feature Engineering", padding=4, style="Custom.TButton",command=feature_engineering)
    feature_button.grid(row=4, column=0, pady=5, sticky="ew")

    # Sep
    sep00_label = ttk.Separator(left_frame, orient="horizontal")
    sep00_label.grid(row=5, column=0, pady=5, sticky="ew")

    # Sep
    sep00_label = ttk.Separator(left_frame, orient="horizontal")
    sep00_label.grid(row=11, column=0, pady=5, sticky="ew")

    reg_label = ttk.Label(left_frame, text="Regression Algorithm", font=custom_font1)
    reg_label.grid(row=12, column=0, pady=5, sticky="ew")

    linear_reg_button = ttk.Button(left_frame, text="Random Forest", padding=4, style="Custom.TButton", command=random_forest_regression)
    linear_reg_button.grid(row=13, column=0, pady=5, sticky="ew")

    linear_reg_button = ttk.Button(left_frame, text="Tune Model", padding=4, style="Custom.TButton", command=tune_model)
    linear_reg_button.grid(row=14, column=0, pady=5, sticky="ew")

    # Right Frame
    right_frame = ttk.Frame(window, padding=10)
    right_frame.grid(row=0, column=2, sticky="nsew")
    right_frame.columnconfigure(0, weight=1)

    data_vis_label = ttk.Label(right_frame, text="Data Visualization", font=custom_font1)
    data_vis_label.grid(row=6, column=0, pady=5, sticky="ew")

    air_vis_button = ttk.Button(right_frame, text="Airlines Visualization", padding=4, style="Custom.TButton", command=vis_airlines)
    air_vis_button.grid(row=7, column=0, pady=5, sticky="ew")
    dest_vis_button = ttk.Button(right_frame, text="Destinations Visualization", padding=4, style="Custom.TButton", command=vis_destination)
    dest_vis_button.grid(row=8, column=0, pady=5, sticky="ew")
    dest_vis_button = ttk.Button(right_frame, text="Journeys/Month Visualization", padding=4, style="Custom.TButton", command=source_vis)
    dest_vis_button.grid(row=9, column=0, pady=5, sticky="ew")
    dest_vis_button = ttk.Button(right_frame, text="Durations VS Price Visualization", padding=4, style="Custom.TButton", command=duration_price_vis)
    dest_vis_button.grid(row=10, column=0, pady=5, sticky="ew")


sep1_label = tk.Label(window, text="", font=custom_font, bg='white')
sep1_label.pack()

# Create a button to browse for a file
read_button = tk.Button(window, text="Upload Data CSV File", command=upload_csv, padx=10, pady=5, bd=0, font=custom_font, fg="white", bg="#0cb6d0")
read_button.pack()

# Create a label to display the selected file path
csv_label = tk.Label(window, text="", font=custom_font1, bg='white')
csv_label.pack()

sep2_label = tk.Label(window, text="", font=custom_font, bg='white')
sep2_label.pack()

read_button = tk.Button(window, text="Create", command=create_dataset, padx=10, pady=5, bd=0, font=custom_font, fg="white", bg="#0cb6d0")
read_button.pack()

########################################################################################################################


def drop_na():
    global data_df
    for widget in window.winfo_children():
        widget.destroy()

    create_dataset()

    separator = ttk.Separator(window, orient='horizontal')
    separator.grid(column=1, row=1, padx=10, pady=10)

    if data_df.isnull().sum().sum() == 0:
        info = tk.Label(window, text=f"No Null Values", font=custom_font1, pady=5, padx=5, justify="right", fg="white", bg="#78a5de")
        info.grid(column=1, row=2)
    else:
        info = tk.Label(window, text=f"{data_df.isnull().sum()}", font=custom_font, pady=5, padx=5, justify="right")
        info.grid(column=1, row=2)

        data_df.dropna(inplace=True)
        separator = ttk.Separator(window, orient='horizontal')
        separator.grid(column=1, row=3, padx=10, pady=10)

        info1 = tk.Label(window, text=f"Null Values Removed Successfully!", font=custom_font1, pady=5, padx=5, justify="right", fg="white", bg="#78a5de")
        info1.grid(column=1, row=4)

    create_menu()


def remove_duplicates():
    global data_df
    for widget in window.winfo_children():
        widget.destroy()

    create_dataset()

    separator = ttk.Separator(window, orient='horizontal')
    separator.grid(column=1, row=1, padx=10, pady=10)

    if data_df.duplicated().any():
        info = tk.Label(window, text=f"{data_df[data_df.duplicated()]}", font=custom_font1, pady=5, padx=5, justify="left")
        info.grid(column=1, row=2)

        data_df.drop_duplicates(keep='first', inplace=True)

        separator1 = ttk.Separator(window, orient='horizontal')
        separator1.grid(column=1, row=3, padx=10, pady=10)

        info1 = tk.Label(window, text=f"Duplicated Values Removed Successfully!", font=custom_font1, pady=5, padx=5, justify="right", fg="white", bg="#78a5de")
        info1.grid(column=1, row=4)
    else:
        info = tk.Label(window, text=f"No Duplicated Values", font=custom_font, pady=5, padx=5, justify="right", fg="white", bg="#78a5de")
        info.grid(column=1, row=2)

    create_menu()


options = ["Airline", "Source", "Destination", "Additional_Info"]


def check_consistency():
    global data_df
    for widget in window.winfo_children():
        widget.destroy()

    create_dataset()

    separator = ttk.Separator(window, orient='horizontal')
    separator.grid(column=1, row=1, padx=10, pady=10)

    selected_option = tk.StringVar()
    # Create a Combobox widget and set its values
    dropdown = ttk.Combobox(window, values=options, textvariable=selected_option, state="readonly", font=custom_font2, width=35)

    # Set the default value of the dropdown to the first option
    dropdown.current(0)

    type_label = tk.Label(window, text="Choose Column", font=custom_font, bg='white')
    type_label.grid(column=1, row=2, padx=10, pady=10)
    # Place the dropdown and button on the window using grid layout
    dropdown.grid(column=1, row=3, padx=10, pady=10)

    # Create a button to save the selected option
    choose_con_button = tk.Button(window, text="Show", command=lambda : show_consistency(selected_option.get()), padx=7, pady=3, bd=0,font=custom_font1, fg="white", bg="#78a5de")
    choose_con_button.grid(column=1, row=4, padx=10, pady=10)

    create_menu()


def show_consistency(choice):
    global data_df
    for widget in window.winfo_children():
        widget.destroy()

    create_dataset()

    separator = ttk.Separator(window, orient='horizontal')
    separator.grid(column=1, row=1, padx=10, pady=10)

    info = tk.Label(window, text=f"{data_df[choice].value_counts()}", font=custom_font, pady=20, padx=20, justify="right")
    info.grid(column=1, row=2)

    print(choice)
    print(data_df[choice].value_counts())

    choose_con_button = tk.Button(window, text="Apply", command=lambda : apply_consistency(choice), padx=7, pady=3, bd=0,font=custom_font1, fg="white", bg="#78a5de")
    choose_con_button.grid(column=1, row=4, padx=10, pady=10)

    create_menu()


def apply_consistency(choice):
    global data_df
    for widget in window.winfo_children():
        widget.destroy()

    create_dataset()

    separator = ttk.Separator(window, orient='horizontal')
    separator.grid(column=1, row=1, padx=10, pady=10)

    data_df[choice] = data_df[choice].apply(lambda x: x.lower())
    print(data_df[choice].value_counts())

    info = tk.Label(window, text=f"{data_df[choice].value_counts()}", font=custom_font, pady=20, padx=20, justify="right")
    info.grid(column=1, row=2)

    info1 = tk.Label(window, text=f"Changed Successfully!", font=custom_font, pady=5, padx=5,
                     justify="right", fg="white", bg="#78a5de")
    info1.grid(column=1, row=4)

    create_menu()


def feature_engineering():
    global data_df
    for widget in window.winfo_children():
        widget.destroy()

    # Duration convert hours in min.
    data_df['Duration'] = data_df['Duration'].str.replace("h", '*60').str.replace(' ', '+').str.replace('m', '*1').apply(eval)

    # Date_of_Journey
    data_df["Journey_day"] = data_df['Date_of_Journey'].str.split('/').str[0].astype(int)
    data_df["Journey_month"] = data_df['Date_of_Journey'].str.split('/').str[1].astype(int)
    data_df.drop(["Date_of_Journey"], axis=1, inplace=True)

    # Dep_Time
    data_df["Dep_hour"] = pd.to_datetime(data_df["Dep_Time"]).dt.hour
    data_df["Dep_min"] = pd.to_datetime(data_df["Dep_Time"]).dt.minute
    data_df.drop(["Dep_Time"], axis=1, inplace=True)

    # Arrival_Time
    data_df["Arrival_hour"] = pd.to_datetime(data_df.Arrival_Time).dt.hour
    data_df["Arrival_min"] = pd.to_datetime(data_df.Arrival_Time).dt.minute
    data_df.drop(["Arrival_Time"], axis=1, inplace=True)

    # Total_Stops
    data_df['Total_Stops'].replace(['1 stop', 'non-stop', '2 stops', '3 stops', '4 stops'], [1, 0, 2, 3, 4], inplace=True)

    create_dataset()

    separator = ttk.Separator(window, orient='horizontal')
    separator.grid(column=1, row=1, padx=10, pady=10)

    info1 = tk.Label(window, text=f"Changed Successfully!", font=custom_font, pady=5, padx=5,
                     justify="right", fg="white", bg="#78a5de")
    info1.grid(column=1, row=2)

    create_menu()


def vis_airlines():
    global data_df
    for widget in window.winfo_children():
        widget.destroy()

    fig = plt.Figure(figsize=(10,7), dpi=100)
    ax = fig.add_subplot(111)
    ax.spines['bottom'].set_visible(True)

    sns.countplot(x='Airline', data=data_df, ax=ax)
    ax.tick_params(axis='x', rotation=45)
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().grid(column=1, row=0)

    create_menu()


def source_vis():
    global data_df
    for widget in window.winfo_children():
        widget.destroy()

    fig = plt.Figure(figsize=(10, 7), dpi=100)
    ax = fig.add_subplot(111)

    sns.countplot(x='Journey_month', data=data_df, palette='Dark2', ax=ax)
    ax.tick_params(axis='x', rotation=45)
    plt.legend(loc='upper right')

    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().grid(column=1, row=0)

    create_menu()


def vis_destination():
    global data_df
    for widget in window.winfo_children():
        widget.destroy()

    plt.figure(figsize=(10, 9))
    plt.subplot(111)
    data_df['Destination'].value_counts().plot.pie(autopct='%1.1f%%')
    centre = plt.Circle((0, 0), 0.7, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre)

    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().grid(column=1, row=0)

    create_menu()


def vis_stops():
    global data_df
    for widget in window.winfo_children():
        widget.destroy()

    plt.figure(figsize=(15, 8))
    plt.subplot(1, 2, 1)
    data_df['Total_Stops'].value_counts().plot.pie(autopct='%1.1f%%')
    centre = plt.Circle((0, 0), 0.7, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre)

    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().grid(column=1, row=0)

    create_menu()


def duration_price_vis():
    global data_df
    for widget in window.winfo_children():
        widget.destroy()

    fig = plt.Figure(figsize=(10, 7), dpi=100)
    ax = fig.add_subplot(111)

    sns.scatterplot(x='Duration', y='Price', hue='Airline', data=data_df, marker='D', palette='Set1', ax=ax)

    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().grid(column=1, row=0)

    create_menu()


def random_forest_regression():
    for widget in window.winfo_children():
        widget.destroy()

    data = data_df.drop(["Price"], axis=1)

    train_categorical_data = data.select_dtypes(exclude=['int64', 'float', 'int32'])
    train_numerical_data = data.select_dtypes(include=['int64', 'float', 'int32'])

    print(train_categorical_data.head())

    train_categorical_data = train_categorical_data.apply(LabelEncoder().fit_transform)

    print("######################################")

    print(train_categorical_data.head())

    X = pd.concat([train_categorical_data, train_numerical_data], axis=1)
    y = data_df['Price']

    # Create a PandasModel instance
    model = PandasModel(pd.concat([X, y]))

    # Create a DataFrameTable instance
    table = DataFrameTable(window, model)

    # Pack the Treeview widget and start the Tkinter event loop
    table.grid(column=1, row=0, sticky="nsew")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    reg = RandomForestRegressor()
    reg.fit(X_train, y_train)

    y_pred_train = reg.predict(X_train)
    y_pred_test = reg.predict(X_test)

    mse_train = mean_squared_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)

    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)

    print("MSE - Train: {:} Test: {:}".format(mse_train, mse_test))
    print("R2 - Train: {:} Test: {:}".format(r2_train, r2_test))

    separator = ttk.Separator(window, orient='horizontal')
    separator.grid(column=1, row=1, padx=10, pady=10)

    info = tk.Label(window, text=f"R2 Score: {round(r2_test * 100,3)}%", font=custom_font, pady=5, padx=5, justify="center", fg="white", bg="#78a5de")
    info.grid(column=1, row=2)

    separator = ttk.Separator(window, orient='horizontal')
    separator.grid(column=1, row=3, padx=10, pady=10)

    info1 = tk.Label(window, text=f"MSE: {round(mse_test, 3)}", font=custom_font, pady=5, padx=5,
                     justify="center", fg="white", bg="#78a5de")
    info1.grid(column=1, row=4)

    separator1 = ttk.Separator(window, orient='horizontal')
    separator1.grid(column=1, row=5, padx=10, pady=10)

    info2 = tk.Label(window, text=f"RMSE: {round(np.sqrt(mean_squared_error(y_test, y_pred_test)),4)}", font=custom_font, pady=5, padx=5,
                     justify="center", fg="white", bg="#78a5de")
    info2.grid(column=1, row=6)

    create_menu()


def tune_model():
    global data_df
    for widget in window.winfo_children():
        widget.destroy()

    data = data_df.drop(["Price"], axis=1)

    train_categorical_data = data.select_dtypes(exclude=['int64', 'float', 'int32'])
    train_numerical_data = data.select_dtypes(include=['int64', 'float', 'int32'])

    print(train_categorical_data.head())

    train_categorical_data = train_categorical_data.apply(LabelEncoder().fit_transform)

    print("######################################")

    print(train_categorical_data.head())

    X = pd.concat([train_categorical_data, train_numerical_data], axis=1)
    y = data_df['Price']

    # Create a PandasModel instance
    model = PandasModel(pd.concat([X, y]))

    # Create a DataFrameTable instance
    table = DataFrameTable(window, model)

    # Pack the Treeview widget and start the Tkinter event loop
    table.grid(column=1, row=0, sticky="nsew")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    n_estimators = [int(x) for x in np.linspace(start=100, stop=800, num=6)]  # Number of trees in random forest
    max_depth = [int(x) for x in np.linspace(30, 70, num=6)]  # Max number of levels in tree
    min_samples_split = [3, 5, 10, 12, 15, 17, 20]  # Min number of samples required to split a node
    min_samples_leaf = [1, 2, 5, 10]  # Min number of samples requred at each leaft node

    rf_param_grid = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
    }

    rfr = RandomForestRegressor()
    rf_tunned = RandomizedSearchCV(scoring='neg_mean_squared_error', estimator=rfr, param_distributions=rf_param_grid, n_iter=30, cv=5, verbose=2, n_jobs=1)
    rf_tunned.fit(X_train, y_train)

    print(rf_param_grid)

    print(rf_tunned.best_params_)

    y_pred = rf_tunned.predict(X_test)
    rmse = float(format(np.sqrt(mean_squared_error(y_test, y_pred)), '.3f'))
    print("RMSE: ", rmse)
    mse = float(format(mean_squared_error(y_test, y_pred), '.3f'))
    print("MSE: ", mse)
    r2 = float(format(r2_score(y_test, y_pred), '.3f'))
    print("R2 Score: ", r2)

    info = tk.Label(window, text=f"R2 Score: {round(r2_score(y_test, y_pred) * 100,3)}%", font=custom_font, pady=5, padx=5, justify="center", fg="white", bg="#78a5de")
    info.grid(column=1, row=2)

    separator = ttk.Separator(window, orient='horizontal')
    separator.grid(column=1, row=3, padx=10, pady=10)

    info1 = tk.Label(window, text=f"MSE: {round(mean_squared_error(y_test, y_pred), 3)}", font=custom_font, pady=5, padx=5,
                     justify="center", fg="white", bg="#78a5de")
    info1.grid(column=1, row=4)

    separator1 = ttk.Separator(window, orient='horizontal')
    separator1.grid(column=1, row=5, padx=10, pady=10)

    info2 = tk.Label(window, text=f"RMSE: {round(rmse, 3)}", font=custom_font, pady=5, padx=5,
                     justify="center", fg="white", bg="#78a5de")
    info2.grid(column=1, row=6)

    create_menu()

window.mainloop()
