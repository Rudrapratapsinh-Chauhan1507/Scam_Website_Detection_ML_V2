import mysql.connector
from mysql.connector import pooling
from datetime import datetime
from typing import Any


_POOL = None  # module-level connection pool (created once)


def _get_pool() -> pooling.MySQLConnectionPool:
    global _POOL
    if _POOL is None:
        _POOL = pooling.MySQLConnectionPool(
            pool_name="scam_pool",
            pool_size=5,
            host="localhost",
            user="root",
            password="",
            database="scam_website_detection",
        )
    return _POOL


class DatabaseManager:
    """
    Thin wrapper around a MySQL connection pool.

    Each public method acquires a connection, executes its query,
    and releases the connection back to the pool automatically.
    """

    # internal helpers

    def _execute(
        self,
        query: str,
        values: tuple = (),
        fetch: str = "none",   # "none" | "one" | "all"
    ) -> Any:
        conn = _get_pool().get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, values)
            if fetch == "all":
                result = cursor.fetchall()
            elif fetch == "one":
                result = cursor.fetchone()
            else:
                conn.commit()
                result = cursor.lastrowid
            cursor.close()
            return result
        finally:
            conn.close()   # returns to pool

    # write methods
    def insert_website(self, url: str, data: dict) -> int | None:
        query = """
            INSERT IGNORE INTO websites (
                url, title, meta_description, text_content,
                final_url, status, error_message,
                redirect_count, scraped_at
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """
        values = (
            url,
            data.get("title"),
            data.get("meta_description"),
            data.get("text"),
            data.get("final_url"),
            data.get("status"),
            data.get("error"),
            data.get("redirect_count"),
            datetime.now(),
        )
        return self._execute(query, values)
    
    def update_label(self, url: str, label: int) -> None:
        query = "UPDATE websites SET label=%s WHERE url=%s"
        self._execute(query, (label, url))

    def update_url_features(self, url: str, features: dict) -> None:
        query = """
            UPDATE websites
            SET url_length=%s, num_dots=%s, num_hyphen=%s, num_slashes=%s,
                https=%s, subdomains=%s, has_at_symbol=%s, has_double_slash=%s, has_ip=%s,
                num_underscores=%s, num_percent=%s, num_digits=%s,
                num_query_params=%s, has_query=%s, path_depth=%s,
                suspicious_tld=%s, brand_in_url=%s, is_shortened=%s,
                suspicious_word_count=%s
            WHERE url=%s
        """
        values = (
            features.get("url_length"),
            features.get("num_dots"),
            features.get("num_hyphen"),
            features.get("num_slashes"),
            features.get("https"),
            features.get("subdomains"),
            features.get("has_at_symbol"),
            features.get("has_double_slash"),
            features.get("has_ip"),
            features.get("num_underscores"),
            features.get("num_percent"),
            features.get("num_digits"),
            features.get("num_query_params"),
            features.get("has_query"),
            features.get("path_depth"),
            features.get("suspicious_tld"),
            features.get("brand_in_url"),
            features.get("is_shortened"),
            features.get("suspicious_word_count"),
            url,
        )
        self._execute(query, values)

    def update_domain_features(self, url: str, features: dict) -> None:
        query = """
            UPDATE websites
            SET domain_age_days=%s, domain_expiry_days=%s, registrar=%s,
                has_ssl=%s, ssl_valid=%s, ssl_days_remaining=%s,
                is_new_domain=%s, short_expiry_domain=%s
            WHERE url=%s
        """
        values = (
            features.get("domain_age_days"),
            features.get("domain_expiry_days"),
            features.get("registrar"),
            features.get("has_ssl"),
            features.get("ssl_valid"),
            features.get("ssl_days_remaining"),
            features.get("is_new_domain"),
            features.get("short_expiry_domain"),
            url,
        )
        self._execute(query, values)

    def update_content_features(self, url: str, features: dict) -> None:
        query = """
            UPDATE websites
            SET text_length=%s, token_count=%s,
                scam_keyword_count=%s, scam_keyword_density=%s,
                has_form=%s, has_iframe=%s,
                exclamation_count=%s, caps_ratio=%s,
                avg_word_length=%s
            WHERE url=%s
        """
        values = (
            features.get("text_length"),
            features.get("token_count"),
            features.get("scam_keyword_count"),
            features.get("scam_keyword_density"),
            features.get("has_form"),
            features.get("has_iframe"),
            features.get("exclamation_count"),
            features.get("caps_ratio"),
            features.get("avg_word_length"),
            url,
        )
        self._execute(query, values)

    # read methods

    def url_exists(self, url: str) -> bool:
        row = self._execute(
            "SELECT 1 FROM websites WHERE url=%s LIMIT 1",
            (url,),
            fetch="one",
        )
        return row is not None

    def get_all_urls(self) -> list[str]:
        rows = self._execute("SELECT url FROM websites", fetch="all")
        return [r[0] for r in rows]

    def get_training_data(self) -> list[dict]:
        rows = self._execute(
            """
            SELECT url, title, text_content, label
            FROM websites
            WHERE label IS NOT NULL
            """,
            fetch="all",
        )

        return [
            {
                "url": r[0],
                "title": r[1],
                "text": r[2],
                "label": r[3],
            }
            for r in rows
        ]