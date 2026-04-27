import os
from datetime import datetime
from typing import Any

import mysql.connector
from mysql.connector import pooling


_POOL = None


def _db_config() -> dict:
    return {
        "host": os.getenv("DB_HOST", "127.0.0.1"),
        "user": os.getenv("DB_USER", "root"),
        "password": os.getenv("DB_PASSWORD", ""),
        "database": os.getenv("DB_NAME", "scam_shield_database"),
        "port": int(os.getenv("DB_PORT", 3307))
    }


def _get_pool() -> pooling.MySQLConnectionPool:
    global _POOL
    if _POOL is None:
        config = _db_config()
        _POOL = pooling.MySQLConnectionPool(
            pool_name="scam_pool",
            pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
            pool_reset_session=True,
            **config,
        )
    return _POOL


class DatabaseManager:
    """
    Thin wrapper around a MySQL connection pool.

    Each public method acquires a connection, executes its query,
    and releases the connection back to the pool automatically.
    """

    def _execute(
        self,
        query: str,
        values: tuple = (),
        fetch: str = "none",  # "none" | "one" | "all"
    ) -> Any:
        conn = _get_pool().get_connection()
        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute(query, values)

            if fetch == "all":
                return cursor.fetchall()
            if fetch == "one":
                return cursor.fetchone()

            conn.commit()
            return cursor.lastrowid
        except mysql.connector.Error:
            conn.rollback()
            raise
        finally:
            if cursor is not None:
                cursor.close()
            conn.close()

    @staticmethod
    def _normalize_text(value: Any) -> str | None:
        if value is None:
            return None
        value = str(value).strip()
        return value if value else None

    @staticmethod
    def _normalize_int(value: Any) -> int | None:
        if value is None or value == "":
            return None
        return int(value)

    @staticmethod
    def _normalize_float(value: Any) -> float | None:
        if value is None or value == "":
            return None
        return float(value)

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
            self._normalize_text(url),
            self._normalize_text(data.get("title")),
            self._normalize_text(data.get("meta_description")),
            self._normalize_text(data.get("text")),
            self._normalize_text(data.get("final_url")),
            self._normalize_text(data.get("status")),
            self._normalize_text(data.get("error")),
            self._normalize_int(data.get("redirect_count")),
            datetime.now(),
        )
        return self._execute(query, values)

    def update_label(self, url: str, label: int) -> None:
        query = "UPDATE websites SET label=%s WHERE url=%s"
        self._execute(query, (self._normalize_int(label), self._normalize_text(url)))

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
            self._normalize_int(features.get("url_length")),
            self._normalize_int(features.get("num_dots")),
            self._normalize_int(features.get("num_hyphen")),
            self._normalize_int(features.get("num_slashes")),
            self._normalize_int(features.get("https")),
            self._normalize_int(features.get("subdomains")),
            self._normalize_int(features.get("has_at_symbol")),
            self._normalize_int(features.get("has_double_slash")),
            self._normalize_int(features.get("has_ip")),
            self._normalize_int(features.get("num_underscores")),
            self._normalize_int(features.get("num_percent")),
            self._normalize_int(features.get("num_digits")),
            self._normalize_int(features.get("num_query_params")),
            self._normalize_int(features.get("has_query")),
            self._normalize_int(features.get("path_depth")),
            self._normalize_int(features.get("suspicious_tld")),
            self._normalize_int(features.get("brand_in_url")),
            self._normalize_int(features.get("is_shortened")),
            self._normalize_int(features.get("suspicious_word_count")),
            self._normalize_text(url),
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
            self._normalize_int(features.get("domain_age_days")),
            self._normalize_int(features.get("domain_expiry_days")),
            self._normalize_text(features.get("registrar")),
            self._normalize_int(features.get("has_ssl")),
            self._normalize_int(features.get("ssl_valid")),
            self._normalize_int(features.get("ssl_days_remaining")),
            self._normalize_int(features.get("is_new_domain")),
            self._normalize_int(features.get("short_expiry_domain")),
            self._normalize_text(url),
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
            self._normalize_int(features.get("text_length")),
            self._normalize_int(features.get("token_count")),
            self._normalize_int(features.get("scam_keyword_count")),
            self._normalize_float(features.get("scam_keyword_density")),
            self._normalize_int(features.get("has_form")),
            self._normalize_int(features.get("has_iframe")),
            self._normalize_int(features.get("exclamation_count")),
            self._normalize_float(features.get("caps_ratio")),
            self._normalize_float(features.get("avg_word_length")),
            self._normalize_text(url),
        )
        self._execute(query, values)

    def url_exists(self, url: str) -> bool:
        row = self._execute(
            "SELECT 1 FROM websites WHERE url=%s LIMIT 1",
            (self._normalize_text(url),),
            fetch="one",
        )
        return row is not None

    def get_all_urls(self) -> list[str]:
        rows = self._execute("SELECT url FROM websites", fetch="all")
        return [row[0] for row in rows]

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
                "url": row[0],
                "title": row[1],
                "text": row[2],
                "label": row[3],
            }
            for row in rows
        ]

    def get_text_corpus(self) -> list[str]:
        rows = self._execute(
            """
            SELECT title, text_content
            FROM websites
            WHERE text_content IS NOT NULL
              AND TRIM(text_content) <> ''
            """,
            fetch="all",
        )
        return [
            " ".join(part for part in [row[0], row[1]] if part)
            for row in rows
        ]

    def get_dataset_rows(self, labeled_only: bool = True) -> list[dict]:
        query = """
            SELECT
                url,
                final_url,
                title,
                meta_description,
                text_content,
                status,
                error_message,
                redirect_count,
                label,
                url_length,
                num_dots,
                num_hyphen,
                num_slashes,
                https,
                subdomains,
                has_at_symbol,
                has_double_slash,
                has_ip,
                num_underscores,
                num_percent,
                num_digits,
                num_query_params,
                has_query,
                path_depth,
                suspicious_tld,
                brand_in_url,
                is_shortened,
                suspicious_word_count,
                domain_age_days,
                domain_expiry_days,
                registrar,
                has_ssl,
                ssl_valid,
                ssl_days_remaining,
                is_new_domain,
                short_expiry_domain,
                text_length,
                token_count,
                scam_keyword_count,
                scam_keyword_density,
                has_form,
                has_iframe,
                exclamation_count,
                caps_ratio,
                avg_word_length,
                scraped_at
            FROM websites
        """
        if labeled_only:
            query += " WHERE label IS NOT NULL"

        rows = self._execute(query, fetch="all")
        columns = [
            "url",
            "final_url",
            "title",
            "meta_description",
            "text_content",
            "status",
            "error_message",
            "redirect_count",
            "label",
            "url_length",
            "num_dots",
            "num_hyphen",
            "num_slashes",
            "https",
            "subdomains",
            "has_at_symbol",
            "has_double_slash",
            "has_ip",
            "num_underscores",
            "num_percent",
            "num_digits",
            "num_query_params",
            "has_query",
            "path_depth",
            "suspicious_tld",
            "brand_in_url",
            "is_shortened",
            "suspicious_word_count",
            "domain_age_days",
            "domain_expiry_days",
            "registrar",
            "has_ssl",
            "ssl_valid",
            "ssl_days_remaining",
            "is_new_domain",
            "short_expiry_domain",
            "text_length",
            "token_count",
            "scam_keyword_count",
            "scam_keyword_density",
            "has_form",
            "has_iframe",
            "exclamation_count",
            "caps_ratio",
            "avg_word_length",
            "scraped_at",
        ]
        return [dict(zip(columns, row)) for row in rows]
