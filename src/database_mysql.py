import mysql.connector
from datetime import datetime

class DatabaseManager:

    def __init__(self):
        
        self.conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="scam_detection"
        )

        self.cursor = self.conn.cursor()

    def insert_website(self, url, data, features):

        query = """
        INSERT INTO websites
        (
        url,
        title, 
        meta_description, 
        text_content, 
        scraped_at, 
        status, 
        error_message, 
        url_length, 
        num_dots, 
        num_hyphen, 
        num_slashes, 
        https, 
        subdomains, 
        has_at_symbol, 
        has_ip)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """

        values = (
            url,
            data["title"],
            data["meta_description"],
            data["text"],
            datetime.now(),
            data["status"],
            data["error"],
            features["url_length"],
            features["num_dots"],
            features["num_hyphen"],
            features["num_slashes"],
            features["https"],
            features["subdomains"],
            features["has_at_symbol"],
            features["has_ip"]
        )

        self.cursor.execute(query, values)

        self.conn.commit()

    def get_all_urls(self):
        self.cursor.execute("SELECT url FROM websites")
        return [row[0] for row in self.cursor.fetchall()]
    
    def update_url_features(self, url, features):

        query = """
        UPDATE websites
        SET
        url_length=%s,
        num_dots=%s,
        num_hyphen=%s,
        num_slashes=%s,
        https=%s,
        subdomains=%s,
        has_at_symbol=%s,
        has_ip=%s
        WHERE url=%s
        """

        values = (
            features["url_length"],
            features["num_dots"],
            features["num_hyphen"],
            features["num_slashes"],
            features["https"],
            features["subdomains"],
            features["has_at_symbol"],
            features["has_ip"],
            url
        )

        self.cursor.execute(query, values)

        self.conn.commit()