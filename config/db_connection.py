import mysql.connector
from mysql.connector import Error

def create_connection():
    try:
        connection = mysql.connector.connect(
            host="localhost",        # Host database
            user="root",             # Username MySQL Anda
            password="",     # Password MySQL Anda
            database="tugas_akhir_CNN" # Nama database yang ingin digunakan
        )
        if connection.is_connected():
            print("Koneksi ke MySQL berhasil!")
        return connection
    except Error as e:
        print("Error saat menghubungkan ke MySQL:", e)
        return None
