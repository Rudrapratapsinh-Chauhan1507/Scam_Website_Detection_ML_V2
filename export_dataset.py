import argparse
from pathlib import Path

import pandas as pd

from src.database_mysql import DatabaseManager


DEFAULT_OUTPUT = Path("dataset") / "training_dataset.csv"


def build_dataset(labeled_only: bool = True, drop_text: bool = False) -> pd.DataFrame:
    db = DatabaseManager()
    rows = db.get_dataset_rows(labeled_only=labeled_only)
    df = pd.DataFrame(rows)

    if df.empty:
        return df

    if drop_text:
        columns_to_drop = ["title", "meta_description", "text_content", "registrar", "error_message"]
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export website records from MySQL into a CSV dataset."
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path to the CSV file to create.",
    )
    parser.add_argument(
        "--include-unlabeled",
        action="store_true",
        help="Include rows without labels. By default only labeled rows are exported.",
    )
    parser.add_argument(
        "--drop-text",
        action="store_true",
        help="Drop raw text-heavy columns and keep mostly structured ML features.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_dataset(
        labeled_only=not args.include_unlabeled,
        drop_text=args.drop_text,
    )

    if df.empty:
        print("[WARN] No rows found for export. Check your database and labels first.")
        return

    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[OK] Exported {len(df)} rows to {output_path}")
    print(f"[INFO] Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()
