from typing import List
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        # Consider try/except with a friendly message if model missing
        self.model = CrossEncoder(model_name)

    def score_pairs(self, jd_text: str, resume_texts: List[str]) -> List[float]:
        pairs = [(jd_text, t) for t in resume_texts]
        scores = self.model.predict(pairs)
        return [float(s) for s in scores]


