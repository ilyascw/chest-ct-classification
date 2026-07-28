"""
Feature extraction using CT-CLIP based on ct_lipro_inference.py
"""

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.typing import NDArray
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

logger = logging.getLogger(__name__)


class VolumePreprocessor(Protocol):
    """Minimal preprocessing interface used by file-based extraction."""

    def preprocess_file(self, file_path: Path, metadata: pd.Series) -> torch.Tensor: ...


class ImageLatentsClassifier(nn.Module):  # type: ignore[misc]
    """Classifier head used to load the published CT-CLIP checkpoint."""

    def __init__(
        self,
        trained_model: nn.Module,
        latent_dim: int = 512,
        num_classes: int = 18,
        dropout_prob: float = 0.3,
    ) -> None:
        super().__init__()
        self.trained_model = trained_model
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(
        self,
        *args: Any,
        latents: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Returns CT-CLIP image latents or classification logits."""
        kwargs["return_latents"] = True

        model_output = self.trained_model(*args, **kwargs)
        _, image_latents, _ = cast(tuple[Any, torch.Tensor, Any], model_output)

        image_latents = self.relu(image_latents)

        if latents:
            return image_latents

        image_latents = self.dropout(image_latents)
        return self.classifier(image_latents)

    def save(self, filepath: str | Path) -> None:
        torch.save(self.state_dict(), filepath)

    def load(self, filepath: str | Path) -> None:
        loaded_state_dict = torch.load(
            filepath,
            map_location="cpu",
            weights_only=True,
        )
        missing_keys, unexpected_keys = self.load_state_dict(loaded_state_dict, strict=False)
        if unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {len(unexpected_keys)}")
        if missing_keys:
            logger.warning(f"Missing keys in checkpoint: {len(missing_keys)}")


class CTCLIPFeatureExtractor:
    """Inference wrapper that converts a CT volume into a 512D embedding."""

    def __init__(
        self,
        model: ImageLatentsClassifier,
        tokenizer: BertTokenizer,
        device: str | None = None,
        verbose: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.verbose = verbose

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

        if self.verbose:
            logger.info(f"CTCLIPFeatureExtractor initialized on {self.device}")

    def _prepare_text_tokens(
        self,
        text: str = "chest computed tomography scan",
    ) -> dict[str, torch.Tensor]:
        """Tokenizes the fixed multimodal prompt."""
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=200,
        ).to(self.device)
        return cast(dict[str, torch.Tensor], tokens)

    def extract_single(
        self,
        volume_tensor: torch.Tensor,
        text: str = "chest computed tomography scan for pathology detection",
        return_numpy: bool = True,
    ) -> NDArray[np.float32] | torch.Tensor:
        """Extracts and validates one 512-dimensional image embedding."""
        try:
            if volume_tensor.dim() == 4:  # (1, D, H, W) → (1, 1, D, H, W)
                volume_tensor = volume_tensor.unsqueeze(1)
            elif volume_tensor.dim() == 3:  # (D, H, W) → (1, 1, D, H, W)
                volume_tensor = volume_tensor.unsqueeze(0).unsqueeze(0)
            elif volume_tensor.dim() != 5:
                raise ValueError(f"Expected a 3D, 4D or 5D tensor; got {volume_tensor.dim()}D")

            volume_tensor = volume_tensor.to(self.device)
            text_tokens = self._prepare_text_tokens(text)

            with torch.no_grad():
                embeddings = self.model(
                    text_tokens, volume_tensor, latents=True, device=self.device
                )

                if embeddings.dim() > 1:
                    embeddings = embeddings.flatten()
                if embeddings.numel() != 512:
                    raise ValueError(f"Expected 512 embedding values; got {embeddings.numel()}")

                if return_numpy:
                    return np.asarray(embeddings.cpu().numpy(), dtype=np.float32)

                return embeddings

        except Exception as exc:
            raise RuntimeError("CT-CLIP feature extraction failed") from exc

    def extract_batch(
        self, volume_tensors: list[torch.Tensor], show_progress: bool = True
    ) -> NDArray[np.float32]:
        """Extract embeddings from multiple volumes"""
        results: list[NDArray[np.float32]] = []
        iterator: Iterable[torch.Tensor] = volume_tensors
        if show_progress:
            iterator = tqdm(volume_tensors, desc="Extracting embeddings")

        for volume in iterator:
            try:
                embedding = self.extract_single(volume, return_numpy=True)
                if not isinstance(embedding, np.ndarray):
                    raise TypeError("Expected numpy embeddings from extract_single")
                results.append(embedding)
            except RuntimeError:
                logger.exception("Batch item failed; returning a zero embedding")
                results.append(np.zeros(512, dtype=np.float32))

        return np.asarray(results, dtype=np.float32)

    def extract_from_files(
        self,
        file_paths: list[str | Path],
        metadata_df: pd.DataFrame,
        preprocessor: VolumePreprocessor | None = None,
    ) -> tuple[NDArray[np.float32], list[str], list[str]]:
        """Extract embeddings directly from files"""
        if preprocessor is None:
            raise ValueError("preprocessor must be provided")

        embeddings_list: list[NDArray[np.float32]] = []
        success_files: list[str] = []
        failed_files: list[str] = []

        for file_path in tqdm(file_paths, desc="Processing files"):
            file_path = Path(file_path)
            filename = file_path.name

            try:
                # Find metadata
                meta_row = metadata_df[metadata_df["VolumeName"] == filename]
                if meta_row.empty:
                    failed_files.append(f"{filename}: No metadata")
                    continue

                # Preprocess
                tensor = preprocessor.preprocess_file(file_path, meta_row.iloc[0])

                # Extract embeddings
                embedding = self.extract_single(tensor, return_numpy=True)

                if not isinstance(embedding, np.ndarray):
                    raise TypeError("Expected numpy embeddings from extract_single")
                embeddings_list.append(embedding)
                success_files.append(filename)

            except Exception as exc:
                failed_files.append(f"{filename}: {exc}")
                continue

        embeddings_array = (
            np.asarray(embeddings_list, dtype=np.float32)
            if embeddings_list
            else np.empty((0, 512), dtype=np.float32)
        )
        return embeddings_array, success_files, failed_files

    def get_model_info(self) -> dict[str, object]:
        """Get model information"""
        return {
            "device": str(self.device),
            "embedding_dim": 512,
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
            "model_type": "CTCLIPFeatureExtractor",
            "ready": True,
        }


def create_ctclip_model_and_extractor(
    checkpoint_path: str | Path, device: str | None = None
) -> tuple[ImageLatentsClassifier, CTCLIPFeatureExtractor]:
    """
    функция для создания CT-CLIP
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        from ct_clip.ct_clip import CTCLIP
        from transformer_maskgit.transformer_maskgit.ctvit import CTViT

        # Tokenizer и text encoder - ТОЧНО как в ct_lipro_inference
        tokenizer = BertTokenizer.from_pretrained(
            "microsoft/BiomedVLP-CXR-BERT-specialized", do_lower_case=True
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
            heads=8,
        )

        # CTCLIP модель - ТОЧНО как в ct_lipro_inference
        clip = CTCLIP(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            dim_image=294912,  # ТОЧНОЕ значение
            dim_text=768,
            dim_latent=512,
            extra_latent_projection=False,
            use_mlm=False,
            downsample_image_embeds=False,
            use_all_token_embeds=False,
        )

        # ImageLatentsClassifier wrapper
        image_classifier = ImageLatentsClassifier(clip, 512, 18)

        # Загрузка весов
        image_classifier.load(checkpoint_path)
        image_classifier.eval()
        image_classifier.to(device)

        # Feature extractor
        feature_extractor = CTCLIPFeatureExtractor(
            model=image_classifier, tokenizer=tokenizer, device=device
        )

        logger.info("CT-CLIP модель и feature extractor созданы успешно")
        return image_classifier, feature_extractor

    except Exception as e:
        logger.error(f"Ошибка создания CT-CLIP: {e}")
        raise


# Alias для обратной совместимости
create_ct_clip_model_and_extractor = create_ctclip_model_and_extractor
