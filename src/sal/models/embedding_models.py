#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Embedding models for uncertainty quantification (UQ) in CoCoA methods.
Supports lazy loading and singleton pattern to avoid repeated loading.
"""

from typing import Optional
import numpy as np


class EmbeddingModel:
    """Base class for embedding models used in UQ calculations."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
    
    def encode(self, texts: list[str], **kwargs) -> np.ndarray:
        """Encode texts into embeddings."""
        raise NotImplementedError
    
    @property
    def model(self):
        """Lazy loading: only load model when first accessed."""
        if self._model is None:
            self._model = self._load_model()
        return self._model
    
    def _load_model(self):
        """Load the actual model. Implemented by subclasses."""
        raise NotImplementedError


class SentenceTransformerModel(EmbeddingModel):
    """SentenceTransformer embedding model for semantic consistency calculation."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        super().__init__(model_name)
    
    def _load_model(self):
        """Load SentenceTransformer model."""
        from sentence_transformers import SentenceTransformer
        import logging
        
        logger = logging.getLogger(__name__)
        logger.info(f"Loading SentenceTransformer: {self.model_name}")
        
        # Some models require trust_remote_code=True (e.g., Alibaba-NLP/gte-large-en-v1.5)
        try:
            model = SentenceTransformer(self.model_name, trust_remote_code=True)
        except Exception as e:
            # If trust_remote_code causes issues, try without it
            logger.warning(f"Failed to load with trust_remote_code=True: {e}. Trying without...")
            model = SentenceTransformer(self.model_name)
        return model
    
    def encode(
        self, 
        texts: list[str], 
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
        **kwargs
    ) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: List of text strings to encode
            convert_to_numpy: Convert output to numpy array
            normalize_embeddings: Normalize embeddings to unit length
            **kwargs: Additional arguments passed to the model
            
        Returns:
            Numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        return self.model.encode(
            texts,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=normalize_embeddings,
            **kwargs
        )


# Global singleton instance
_EMBEDDING_MODEL_CACHE: Optional[EmbeddingModel] = None


def get_embedding_model(
    #model_name: str = "Alibaba-NLP/gte-large-en-v1.5"
    model_name: str = "sentence-transformers/all-mpnet-base-v2"

) -> EmbeddingModel:
    """
    Get or create embedding model instance (singleton pattern).
    
    This ensures the model is only loaded once across the entire program,
    avoiding repeated loading and memory issues.
    
    Args:
        model_name: Name of the embedding model to load
        
    Returns:
        EmbeddingModel instance (cached after first call)
    """
    global _EMBEDDING_MODEL_CACHE
    
    if _EMBEDDING_MODEL_CACHE is None:
        _EMBEDDING_MODEL_CACHE = SentenceTransformerModel(model_name)
    
    return _EMBEDDING_MODEL_CACHE


def reset_embedding_model_cache():
    """
    Reset the global embedding model cache.
    Useful for testing or when switching models.
    """
    global _EMBEDDING_MODEL_CACHE
    _EMBEDDING_MODEL_CACHE = None

