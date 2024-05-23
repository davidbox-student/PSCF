import mysql.connector
from mysql.connector import Error

def create_connection():
    try:
        connection = mysql.connector.connect(
            host='93.176.248.32',
            port=3306,
            user='admin',
            password='A8ElocM87B7Ijf5e'
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

def main():
    connection = create_connection()
    if connection:
        # Tutaj możesz wykonać operacje na bazie danych
        # Na przykład odczytanie danych z bazy
        try:
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
