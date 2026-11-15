import os
import glob
import duckdb
import pandas as pd

class F1SQLStore:
    def __init__(self, data_dir: str = "data/gold"):
        self.data_dir = self._find_gold_dir(data_dir)
        self.conn = duckdb.connect(database=':memory:', read_only=False)
        self._initialize_views()

    def _find_gold_dir(self, target: str) -> str:
        candidates = [
            target,
            os.path.abspath(os.path.join(os.getcwd(), target)),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/gold")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/gold"))
        ]
        for d in candidates:
            if os.path.exists(d):
                return d
        return target

    def _initialize_views(self):
        gold_path = self.data_dir
        print(f"[SQLStore] Connecting DuckDB to Gold path: {gold_path}")
        tables = [
            "fact_session_results", 
            "dim_drivers", 
            "dim_constructors", 
            "dim_races", 
            "ref_nationality_region"
        ]

        for table in tables:
            nested_glob = os.path.join(gold_path, table, "**", "*.parquet").replace("\\", "/")
            flat_glob = os.path.join(gold_path, table, "*.parquet").replace("\\", "/")
            single_file = os.path.join(gold_path, f"{table}.parquet").replace("\\", "/")

            try:
                # Try binding with Hive partitioning enabled to correctly ingest season=YYYY partitions
                self.conn.execute(
                    f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet(['{nested_glob}', '{flat_glob}', '{single_file}'], union_by_name=True, hive_partitioning=1)"
                )
                print(f"[SQLStore] Successfully bound view '{table}' with Hive partitioning (1950-2025).")
            except Exception as e_hive:
                try:
                    # Fallback without hive partitioning if folder structure is flat
                    self.conn.execute(
                        f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet(['{nested_glob}', '{flat_glob}', '{single_file}'], union_by_name=True)"
                    )
                    print(f"[SQLStore] Successfully bound view '{table}' using standard union fallback.")
                except Exception as e:
                    print(f"[SQLStore Error binding {table}]: {e}")

    def execute_query(self, sql_query: str) -> pd.DataFrame:
        try:
            print(f"[SQLStore Executing]: {sql_query}")
            df = self.conn.execute(sql_query.strip()).fetchdf()
            print(f"[SQLStore Result Rows]: {len(df)}")
            return df
        except Exception as e:
            print(f"[SQLStore Error]: {e}")
            return pd.DataFrame()