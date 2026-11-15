import torch
from sentence_transformers import CrossEncoder

class F1CrossEncoderReranker:
    """
    Post-retrieval context compressor using BAAI/bge-reranker-v2-m3 
    to eliminate near-miss semantic noise.
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        print(f"Loading Cross-Encoder model ({model_name})...")
        # Automatically use MPS (Apple Silicon), CUDA, or CPU
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CrossEncoder(model_name, device=device, max_length=512)

    def rerank(self, query: str, candidates: list, top_k: int = 5, threshold: float = 0.3) -> list:
        """
        Jointly encodes query and candidate text chunks, scoring relevance 
        from 0.0 to 1.0 and keeping only top-K high-confidence items.
        """
        if not candidates:
            return []

        # Prepare pairs for cross-attention scoring
        pairs = [[query, str(cand)] for cand in candidates]
        scores = self.model.predict(pairs)

        # Sort by score descending
        scored_results = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

        # Filter by threshold and take top_k
        filtered_context = [cand for score, cand in scored_results if score >= threshold]
        return filtered_context[:top_k]