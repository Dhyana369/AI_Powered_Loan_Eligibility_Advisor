import sqlite3

conn = sqlite3.connect("users.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
)
""")

# Add sample users
cursor.execute("INSERT OR IGNORE INTO users (username, password) VALUES (?, ?)", ("admin", "1234"))
cursor.execute("INSERT OR IGNORE INTO users (username, password) VALUES (?, ?)", ("dhyana", "abcd"))

conn.commit()
conn.close()

print("Database and users created successfully!")
