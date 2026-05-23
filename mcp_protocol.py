class ModelContextProtocol:
    def get_context_prompt(self, style, question, context):
        style_instructions = {
            'legal': 'You are a legal expert. Provide a formal, professional analysis.',
            'kid': 'Explain in simple terms a 5-year-old would understand.',
            'short': 'Provide a concise answer in 2-3 sentences.',
            'default': 'You are a helpful assistant. Provide a thorough, insightful answer.'
        }
        
        instruction = style_instructions.get(style, style_instructions['default'])
        
        return f'''{instruction}

Based on the following document content, answer the question. Only use information from the document:

DOCUMENT:
{context}

QUESTION: {question}

ANSWER:'''