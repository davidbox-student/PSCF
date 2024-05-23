import mysql.connector
from mysql.connector import Error
import pandas as pd
import matplotlib.pyplot as plt


def connect_to_database():
    try:
        # Połączenie z bazą danych
        connection = mysql.connector.connect(
            host='93.176.248.32',
            port=3306,
            user='admin',
            password='A8ElocM87B7Ijf5e'
        )

        if connection.is_connected():
            db_info = connection.get_server_info()
            print(f"Połączono z MySQL wersja {db_info}")
            cursor = connection.cursor()
            cursor.execute("SELECT DATABASE();")
            record = cursor.fetchone()
            print(f"Jesteś połączony z bazą danych: {record}")

    except Error as e:
        print(f"Błąd podczas łączenia z MySQL: {e}")

    def fetch_data(connection):
        query = """
        SELECT 
            d.id,
            d.sensor_id,
            d.group_id,
            d.index,
            d.value,
            g.name as group_name,
            s.name as sensor_name
        FROM Data d
        JOIN `Group` g ON d.group_id = g.id
        JOIN Sensor s ON d.sensor_id = s.id
        """
        return pd.read_sql(query, connection)

    def plot_data(data):
        # Grupowanie danych według indeksu
        grouped_data = data.groupby('index')

        plt.figure(figsize=(10, 6))

        # Tworzenie serii dla każdego indeksu
        for name, group in grouped_data:
            plt.plot(group['id'], group['value'], label=f'Seria {name}')

        plt.xlabel('ID')
        plt.ylabel('Wartość')
        plt.title('Wykres danych z tabeli Data')
        plt.legend()
        plt.show()

    if __name__ == "__main__":
        connection = connect_to_database()
        if connection:
            data = fetch_data(connection)
            connection.close()
            print("Połączenie z MySQL zostało zamknięte.")
            plot_data(data)


if __name__ == "__main__":
    connect_to_database()