from google import genai
from google.genai import types
import logging
from .base import BaseEmbedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoogleGeminiEmbedding(BaseEmbedding):
    def __init__(self, config):
        self.config = config
        self.client = genai.Client(api_key=config['api_key'])

    def embed(self, df, embed_column='__concat_final'):
        logger.info("Starting the Google Gemini embedding process...")

        text_data = df[embed_column].str.strip().tolist()

        response = self.client.models.embed_content(
            model=self.config['model_name'],
            contents=text_data,
            config=types.EmbedContentConfig(
                output_dimensionality=1536,
                task_type="RETRIEVAL_DOCUMENT",
            )
        )

        embeddings = [item.values for item in response.embeddings]
        df['embeddings'] = embeddings

        logger.info("Completed Google Gemini embedding process.")
        return df
