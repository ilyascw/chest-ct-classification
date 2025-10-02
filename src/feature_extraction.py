"""
Feature extraction using CT-CLIP based on ct_lipro_inference.py
"""

from typing import Union, List, Optional, Tuple, Any
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ImageLatentsClassifier(nn.Module):
    """
    Exact copy from ct_lipro_inference.py с исправленным forward()
    """
    
    def __init__(self, trained_model, latent_dim: int = 512, num_classes: int = 18, dropout_prob: float = 0.3):
        super(ImageLatentsClassifier, self).__init__()
        self.trained_model = trained_model  # Это должна быть НАСТОЯЩАЯ CTCLIP модель!
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(latent_dim, num_classes)
    
    def forward(self, *args, latents: bool = False, **kwargs):
        """
        ИСПРАВЛЕННЫЙ forward на основе обсуждения архитектуры
        """
        kwargs['return_latents'] = True
        
        # Правильный вызов CTCLIP модели (возвращает 3 значения)
        text_latents, image_latents, enc_image_send = self.trained_model(*args, **kwargs)
        
        image_latents = self.relu(image_latents)
        
        if latents:
            return image_latents  # ← ВОТ НАШИ 512D EMBEDDINGS!
            
        image_latents = self.dropout(image_latents)
        return self.classifier(image_latents)
    
    def save(self, filepath):
        torch.save(self.state_dict(), filepath)
    
    def load(self, filepath):
        loaded_state_dict = torch.load(filepath, map_location='cpu')
        missing_keys, unexpected_keys = self.load_state_dict(loaded_state_dict, strict=False)
        if unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {len(unexpected_keys)}")
        if missing_keys:
            logger.warning(f"Missing keys in checkpoint: {len(missing_keys)}")


class CTCLIPFeatureExtractor:
    """
    ОБНОВЛЁННЫЙ CT-CLIP feature extraction wrapper
    """
    
    def __init__(
        self,
        model: ImageLatentsClassifier,
        tokenizer: BertTokenizer,
        device: Optional[str] = None,
        verbose: bool = True
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.verbose = verbose
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.model.to(self.device)
        self.model.eval()
        
        if self.verbose:
            logger.info(f"CTCLIPFeatureExtractor initialized on {self.device}")
    
    def _prepare_text_tokens(self, text: str = "chest computed tomography scan") -> dict:
        """Prepare text tokens с правильным текстом по умолчанию"""
        return self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=200
        ).to(self.device)
    
    def extract_single(
        self,
        volume_tensor: torch.Tensor,
        text: str = "chest computed tomography scan for pathology detection",
        return_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        ИСПРАВЛЁННЫЙ extract_single с правильной подготовкой tensor
        """
        try:
            # Правильная подготовка tensor для CT-CLIP
            if volume_tensor.dim() == 4:  # (1, D, H, W) → (1, 1, D, H, W)
                volume_tensor = volume_tensor.unsqueeze(1)
            elif volume_tensor.dim() == 3:  # (D, H, W) → (1, 1, D, H, W)
                volume_tensor = volume_tensor.unsqueeze(0).unsqueeze(0)
                
            volume_tensor = volume_tensor.to(self.device)
            text_tokens = self._prepare_text_tokens(text)
            
            with torch.no_grad():
                # ПРАВИЛЬНЫЙ вызов с корректными параметрами
                embeddings = self.model(
                    text_tokens,
                    volume_tensor,
                    latents=True,
                    device=self.device
                )
                
                if embeddings.dim() > 1:
                    embeddings = embeddings.flatten()
                
                if return_numpy:
                    embeddings = embeddings.cpu().numpy()
                
                return embeddings
                
        except Exception as e:
            logger.error(f"Error in extract_single: {e}")
            return None
    
    def extract_batch(
        self,
        volume_tensors: List[torch.Tensor],
        show_progress: bool = True
    ) -> np.ndarray:
        """Extract embeddings from multiple volumes"""
        results = []
        iterator = volume_tensors
        if show_progress:
            iterator = tqdm(volume_tensors, desc="Extracting embeddings")
            
        for volume in iterator:
            embedding = self.extract_single(volume, return_numpy=True)
            if embedding is not None:
                results.append(embedding)
            else:
                # Добавляем zero embedding при ошибке для сохранения размерности
                results.append(np.zeros(512, dtype=np.float32))
                
        return np.array(results)
    
    def extract_from_files(
        self,
        file_paths: List[Union[str, Path]],
        metadata_df: pd.DataFrame,
        preprocessor = None
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """Extract embeddings directly from files"""
        if preprocessor is None:
            from src.preprocessing import CTPreprocessor
            preprocessor = CTPreprocessor(verbose=False)
            
        embeddings_list = []
        success_files = []
        failed_files = []
        
        for file_path in tqdm(file_paths, desc="Processing files"):
            file_path = Path(file_path)
            filename = file_path.name
            
            try:
                # Find metadata
                meta_row = metadata_df[metadata_df['VolumeName'] == filename]
                if meta_row.empty:
                    failed_files.append(f"{filename}: No metadata")
                    continue
                
                # Preprocess
                tensor = preprocessor.preprocess_file(file_path, meta_row.iloc[0])
                
                # Extract embeddings
                embedding = self.extract_single(tensor, return_numpy=True)
                
                if embedding is not None:
                    embeddings_list.append(embedding)
                    success_files.append(filename)
                else:
                    failed_files.append(f"{filename}: Embedding extraction failed")
                    
            except Exception as e:
                failed_files.append(f"{filename}: {str(e)}")
                continue
        
        embeddings_array = np.array(embeddings_list) if embeddings_list else np.empty((0, 512))
        return embeddings_array, success_files, failed_files
    
    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            'device': str(self.device),
            'embedding_dim': 512,
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'model_type': 'CTCLIPFeatureExtractor',
            'ready': True
        }


def create_ctclip_model_and_extractor(
    checkpoint_path: Union[str, Path],
    device: Optional[str] = None
) -> Tuple[ImageLatentsClassifier, CTCLIPFeatureExtractor]:
    """
    НОВАЯ функция для правильного создания CT-CLIP на основе обсуждения
    """
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Правильные импорты из src
    import sys
    from pathlib import Path
    
    # Добавляем пути к src модулям
    src_path = Path(__file__).parent
    sys.path.insert(0, str(src_path / "ct_clip"))
    sys.path.insert(0, str(src_path / "transformer_maskgit" / "transformer_maskgit"))
    
    try:
        # Импорт из локальных модулей
        from src.ct_clip.ct_clip import CTCLIP
        from src.transformer_maskgit.transformer_maskgit.ctvit import CTViT
        
        # Tokenizer и text encoder - ТОЧНО как в ct_lipro_inference
        tokenizer = BertTokenizer.from_pretrained(
            'microsoft/BiomedVLP-CXR-BERT-specialized',
            do_lower_case=True
        )
        text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
        text_encoder.resize_token_embeddings(len(tokenizer))
        
        # Image encoder - ТОЧНО как в ct_lipro_inference
        image_encoder = CTViT(
            dim=512,
            codebook_size=8192,
            image_size=480,
            patch_size=20,
            temporal_patch_size=10,
            spatial_depth=4,
            temporal_depth=4,
            dim_head=32,
            heads=8
        )
        
        # CTCLIP модель - ТОЧНО как в ct_lipro_inference
        clip = CTCLIP(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            dim_image=294912,     # ТОЧНОЕ значение
            dim_text=768,
            dim_latent=512,
            extra_latent_projection=False,
            use_mlm=False,
            downsample_image_embeds=False,
            use_all_token_embeds=False
        )
        
        # ImageLatentsClassifier wrapper
        image_classifier = ImageLatentsClassifier(clip, 512, 18)
        
        # Загрузка весов
        image_classifier.load(checkpoint_path)
        image_classifier.eval()
        image_classifier.to(device)
        
        # Feature extractor
        feature_extractor = CTCLIPFeatureExtractor(
            model=image_classifier,
            tokenizer=tokenizer,
            device=device
        )
        
        logger.info("CT-CLIP модель и feature extractor созданы успешно")
        return image_classifier, feature_extractor
        
    except Exception as e:
        logger.error(f"Ошибка создания CT-CLIP: {e}")
        raise
