import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), 'witflix_log.db')

class WitflixDBLogger:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT NOT NULL,
                    input TEXT NOT NULL,
                    output TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
            ''')
            conn.commit()

    def log(self, action, input_data, output_data):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute(
                'INSERT INTO logs (action, input, output, timestamp) VALUES (?, ?, ?, ?)',
                (action, input_data, output_data, datetime.now().isoformat())
            )
            conn.commit()

    def get_all_logs(self):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('SELECT id, action, input, output, timestamp FROM logs ORDER BY id DESC')
            rows = c.fetchall()
            return [
                {
                    'id': row[0],
                    'action': row[1],
                    'input': row[2],
                    'output': row[3],
                    'timestamp': row[4],
                }
                for row in rows
            ]

    def delete_log(self, log_id):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('DELETE FROM logs WHERE id = ?', (log_id,))
            conn.commit() 