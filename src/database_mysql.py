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

    def update_domain_features(self, url, features):

        query = """
        UPDATE websites
        SET
        domain_age_days=%s,
        domain_expiry_days=%s,
        registrar=%s,
        has_ssl=%s,
        ssl_valid=%s
        WHERE url=%s
        """

        values = (
            features["domain_age_days"],
            features["domain_expiry_days"],
            features["registrar"],
            features["has_ssl"],
            features["ssl_valid"],
            url
        )

        self.cursor.execute(query, values)
        self.conn.commit()

    def update_content_features(self, url, features):

        query = """
        UPDATE websites
        SET
        text_length = %s,
        token_count = %s,
        scam_keyword_count = %s,
        scam_keyword_density = %s
        WHERE url = %s
        """

        values = (
            features["text_length"],
            features["token_count"],
            features["scam_keyword_count"],
            features["scam_keyword_density"],
            url
        )

        self.cursor.execute(query, values)
        self.conn.commit()

    def get_all_rows(self):

        query = "SELECT url, title, text_content FROM websites"

        self.cursor.execute(query)

        rows = self.cursor.fetchall()

        result = []

        for row in rows:

            result.append({
                "url": row[0],
                "title": row[1],
                "text_content": row[2]
            })

        return result