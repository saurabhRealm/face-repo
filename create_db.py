import sqlite3

def create_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            image TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
    print("Database and table created successfully.")

if __name__ == '__main__':
    create_db()
