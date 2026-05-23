import os
import logging
from groq import Groq

class RAGPipeline:
    def __init__(self):
        logging.info('Initializing Groq API...')
        self.client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        logging.info('Groq API ready!')

    def generate_answer(self, prompt):
        logging.info('Generating answer with Groq...')
        try:
            response = self.client.chat.completions.create(
                model='llama-3.1-8b-instant',
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f'Groq error: {str(e)}')
            return f'Error: {str(e)}'