import logging
import pandas as pd
import vecs
from .base import BaseTarget

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SupabaseTarget(BaseTarget):
    def __init__(self, config):
        self.config = config
        self.vx = None
        self.collection = None

    def connect(self):
        logger.info("Connecting to Supabase...")
        self.vx = vecs.create_client(self.config["supabase_uri"])
        logger.info("Connected to Supabase successfully.")

    def create_index_if_not_exists(self, dimension):
        if self.vx is None:
            self.connect()

        index_name = self.config["index_name"]
        try:
            self.collection = self.vx.get_or_create_collection(
                name=index_name,
                dimension=dimension
            )
        except Exception:
            logger.error(f"Failed to create or access collection '{index_name}' in Supabase.")
            raise

    def write_data(self, df, columns, domain=None):
        logger.info("Writing embeddings to Supabase...")
        if self.vx is None:
            self.connect()

        index_name = self.config["index_name"]
        docs = self.vx.get_or_create_collection(name=index_name, dimension=len(df['embeddings'].iat[0]))

        data = []
        for _, row in df.iterrows():
            metadata = {
                col: str(row[col]) if isinstance(row[col], list) else str(row[col])
                for col in df.columns
                if col not in ["df_uuid", "embeddings"] and (col not in columns or col == "__concat_final") and pd.notna(row[col])
            }

            if domain:
                metadata["domain"] = domain

            data.append((str(row["df_uuid"]), row["embeddings"], metadata))

        docs.upsert(records=data)
        logger.info("Completed writing embeddings to Supabase.")
