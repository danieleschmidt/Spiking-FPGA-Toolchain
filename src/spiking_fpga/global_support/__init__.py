"""Global-first implementation with multi-region support, internationalization, and compliance."""

from .i18n import InternationalizationManager, translate, get_supported_locales
from .regions import RegionManager, get_region_config, deploy_to_region  
from .compliance import GlobalComplianceManager, check_data_residency, validate_cross_border_transfer

__all__ = [
    'InternationalizationManager',
    'RegionManager', 
    'GlobalComplianceManager',
    'translate',
    'get_supported_locales',
    'get_region_config',
    'deploy_to_region',
    'check_data_residency',
    'validate_cross_border_transfer'
]