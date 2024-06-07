import mysql.connector
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import tkinter as tk
from tkinter import ttk

# Zmienna globalna, dane z bazy danych
data = []

sensor_names = {
    1: 'Accelerometer',
    2: 'Magnetic_field',
    4: 'Gyroscope',
    5: 'Light'
}

sensor_xyz = {
    1: 'x',
    2: 'y',
    3: 'z'
}

def create_connection():
    try:
        connection = mysql.connector.connect(
            host='93.176.248.32',
            port=3306,
            user='admin',
            password='A8ElocM87B7Ijf5e',
            database='SensorDataCollector'
        )
        if connection.is_connected():
            print("Połączono z bazą danych")
            return connection
    except mysql.connector.Error as e:
        print(f"Błąd podczas łączenia z bazą danych: {e}")
        return None


def close_connection(connection):
    if connection.is_connected():
        connection.close()
        print("Połączenie z bazą danych zostało zamknięte")


def fetch_users(connection):
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT id, first_name, last_name FROM User")
        return cursor.fetchall()
    except mysql.connector.Error as e:
        print(f"Błąd podczas pobierania użytkowników: {e}")
        return []


def fetch_data(connection, user_id):
    global data
    try:
        cursor = connection.cursor()
        # Dodajemy filtr daty do zapytania SQL
        query = """
            SELECT sensor_id, `index`, timestamp, value 
            FROM Data 
            WHERE user_id = %s AND DATE(timestamp) = '2024-06-05'
            """
        cursor.execute(query, (user_id,))
        data = cursor.fetchall()
    except mysql.connector.Error as e:
        print(f"Błąd podczas pobierania danych: {e}")


def plot_data(frame):
    sensors = set(sensor_id for sensor_id, _, _, _ in data)
    fig, axs = plt.subplots(len(sensors), 1, figsize=(15, 10))

    if len(sensors) == 1:
        axs = [axs]  # Ensure axs is always iterable

    for ax, sensor_id in zip(axs, sensors):
        sensor_data = {}
        for sid, index, timestamp, value in data:
            if sid == sensor_id:
                if index not in sensor_data:
                    sensor_data[index] = []
                sensor_data[index].append((timestamp, value))

        for idx, (index, values) in enumerate(sensor_data.items()):
            timestamps = [timestamp for timestamp, _ in values]
            values = [value for _, value in values]

            # Używamy nazwy sensora zamiast ID i dodajemy oznaczenie osi XYZ
            axis_label = sensor_xyz.get(index, f"Index: {index}")
            ax.plot(timestamps, values, label=axis_label)
            ax.set_xlabel('Timestamp')
            ax.set_ylabel('Value')
            ax.set_title(f'Sensor: {sensor_names.get(sensor_id, "Unknown")}')
            ax.legend()
            ax.tick_params(axis='x', rotation=45)

    fig.tight_layout()

    # Clear the frame and pack the canvas
    for widget in frame.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def update_data(connection, user_id, frame):
    fetch_data(connection, user_id)
    plot_data(frame)


def main():
    connection = create_connection()
    if connection:
        root = tk.Tk()
        root.title("Sensor Data Viewer")

        frame = tk.Frame(root)
        frame.pack(fill=tk.BOTH, expand=True)

        def on_user_select(event):
            selected_user = user_combobox.get()
            user_id = user_id_map[selected_user]
            update_data(connection, user_id, frame)

        user_id_map = {}
        users = fetch_users(connection)
        for user_id, first_name, last_name in users:
            user_id_map[f"{first_name} {last_name}"] = user_id

        user_label = tk.Label(root, text="Wybierz użytkownika:")
        user_label.pack(pady=10)

        user_combobox = ttk.Combobox(root, values=list(user_id_map.keys()))
        user_combobox.bind("<<ComboboxSelected>>", on_user_select)
        user_combobox.pack(pady=10)

        root.mainloop()
        close_connection(connection)


if __name__ == "__main__":
    main()
