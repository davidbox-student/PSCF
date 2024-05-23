import mysql.connector
from mysql.connector import Error

def create_connection():
    try:
        connection = mysql.connector.connect(
            host='93.176.248.32',
            port=3306,
            user='admin',
            password='A8ElocM87B7Ijf5e',
            database = 'SensorDataCollector'
        )
        if connection.is_connected():
            print("Połączono z bazą danych")
            return connection
    except Error as e:
        print(f"Błąd podczas łączenia z bazą danych: {e}")
        return None

def close_connection(connection):
    if connection.is_connected():
        connection.close()
        print("Połączenie z bazą danych zostało zamknięte")


def read_data(connection):
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM Data")
        rows = cursor.fetchall()
        print("Dane w tabeli Data:")
        for row in rows:
            print(row)
    except Error as e:
        print(f"Błąd podczas wykonywania zapytania: {e}")

def main():
    connection = create_connection()
    if connection:
        # Tutaj możesz wykonać operacje na bazie danych
        # Na przykład odczytanie danych z bazy
        try:
            read_data(connection)

            cursor = connection.cursor()
            cursor.execute("SHOW DATABASES")
            databases = cursor.fetchall()
            print("Dostępne bazy danych:")
            for db in databases:
                print(db)
        except Error as e:
            print(f"Błąd podczas wykonywania zapytania: {e}")



        # Zamknięcie połączenia
        close_connection(connection)

if __name__ == "__main__":
    main()
