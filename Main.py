import mysql.connector
import matplotlib.pyplot as plt
import threading

# Zmienna globalna, która będzie przechowywać dane z bazy danych
data = []

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

def fetch_data(connection):
    global data
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT sensor_id, `index`, value FROM Data")
        data = cursor.fetchall()
    except mysql.connector.Error as e:
        print(f"Błąd podczas pobierania danych: {e}")

def plot_data():
    sensors = set(sensor_id for sensor_id, _, _ in data)
    for sensor_id in sensors:
        sensor_data = [(index, value) for sid, index, value in data if sid == sensor_id]
        indexes = [index for index, _ in sensor_data]
        values = [value for _, value in sensor_data]
        plt.plot(indexes, values, label=f"Sensor ID: {sensor_id}")
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Sensor Data')
    plt.legend()
    plt.show()

def data_updater(connection):
    while True:
        fetch_data(connection)
        plot_data()
        if input("Wciśnij Enter aby zakończyć, lub naciśnij klawisz innego by pobrać dane ponownie: ") == "":
            break

def main():
    connection = create_connection()
    if connection:
        updater_thread = threading.Thread(target=data_updater, args=(connection,))
        updater_thread.start()
        updater_thread.join()  # Czekaj, aż wątek aktualizacji danych zakończy działanie
        close_connection(connection)

if __name__ == "__main__":
    main()
