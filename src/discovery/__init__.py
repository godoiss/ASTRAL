"""ASTRAL Discovery Engine for reproducible transit-like signal searches."""

from src.discovery.config import CampaignConfig, DetectionConfig, DiscoveryConfig
from src.discovery.detection import detect_periodic_signal

__all__ = [
    "CampaignConfig",
    "DetectionConfig",
    "DiscoveryConfig",
    "detect_periodic_signal",
]
