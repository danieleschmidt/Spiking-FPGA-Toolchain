"""Internationalization and localization support for global deployment."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
import re
from datetime import datetime
import locale


class SupportedLocale(Enum):
    """Supported locales for the application."""
    
    # Major markets
    EN_US = "en-US"  # English (United States)
    EN_GB = "en-GB"  # English (United Kingdom) 
    EN_CA = "en-CA"  # English (Canada)
    EN_AU = "en-AU"  # English (Australia)
    
    # European markets
    DE_DE = "de-DE"  # German (Germany)
    FR_FR = "fr-FR"  # French (France)
    ES_ES = "es-ES"  # Spanish (Spain)
    IT_IT = "it-IT"  # Italian (Italy)
    NL_NL = "nl-NL"  # Dutch (Netherlands)
    SV_SE = "sv-SE"  # Swedish (Sweden)
    NO_NO = "no-NO"  # Norwegian (Norway)
    
    # Asian markets
    JA_JP = "ja-JP"  # Japanese (Japan)
    ZH_CN = "zh-CN"  # Chinese (Simplified, China)
    ZH_TW = "zh-TW"  # Chinese (Traditional, Taiwan)
    KO_KR = "ko-KR"  # Korean (South Korea)
    
    # Other important markets
    PT_BR = "pt-BR"  # Portuguese (Brazil)
    RU_RU = "ru-RU"  # Russian (Russia)
    AR_SA = "ar-SA"  # Arabic (Saudi Arabia)
    HI_IN = "hi-IN"  # Hindi (India)
    
    @property
    def language_code(self) -> str:
        """Get the language code (e.g., 'en' from 'en-US')."""
        return self.value.split('-')[0]
    
    @property  
    def country_code(self) -> str:
        """Get the country code (e.g., 'US' from 'en-US')."""
        return self.value.split('-')[1]


@dataclass
class LocaleInfo:
    """Information about a supported locale."""
    
    locale: SupportedLocale
    display_name: str
    native_name: str
    rtl: bool = False  # Right-to-left text direction
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"
    number_format: str = "1,234.56"
    currency_symbol: str = "$"
    currency_code: str = "USD"
    region: str = "global"
    
    def format_date(self, date: datetime) -> str:
        """Format date according to locale preferences."""
        return date.strftime(self.date_format)
    
    def format_time(self, time: datetime) -> str:
        """Format time according to locale preferences."""
        return time.strftime(self.time_format)


class InternationalizationManager:
    """Manages internationalization and localization for global deployment."""
    
    def __init__(self, translations_dir: Optional[Path] = None, default_locale: SupportedLocale = SupportedLocale.EN_US):
        self.default_locale = default_locale
        self.current_locale = default_locale
        self.translations_dir = translations_dir or Path(__file__).parent / "translations"
        self.translations_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self._translations: Dict[str, Dict[str, str]] = {}
        self._lock = threading.RLock()
        
        # Initialize locale information
        self._locale_info = self._initialize_locale_info()
        
        # Load translations
        self._load_all_translations()
        
        # Create default translations if they don't exist
        self._ensure_default_translations()
    
    def _initialize_locale_info(self) -> Dict[SupportedLocale, LocaleInfo]:
        """Initialize locale information for all supported locales."""
        return {
            SupportedLocale.EN_US: LocaleInfo(
                locale=SupportedLocale.EN_US,
                display_name="English (United States)",
                native_name="English (United States)",
                date_format="%m/%d/%Y",
                currency_symbol="$",
                currency_code="USD",
                region="americas"
            ),
            SupportedLocale.EN_GB: LocaleInfo(
                locale=SupportedLocale.EN_GB,
                display_name="English (United Kingdom)", 
                native_name="English (United Kingdom)",
                date_format="%d/%m/%Y",
                currency_symbol="£",
                currency_code="GBP",
                region="europe"
            ),
            SupportedLocale.DE_DE: LocaleInfo(
                locale=SupportedLocale.DE_DE,
                display_name="German (Germany)",
                native_name="Deutsch (Deutschland)",
                date_format="%d.%m.%Y",
                number_format="1.234,56",
                currency_symbol="€",
                currency_code="EUR",
                region="europe"
            ),
            SupportedLocale.FR_FR: LocaleInfo(
                locale=SupportedLocale.FR_FR,
                display_name="French (France)",
                native_name="Français (France)",
                date_format="%d/%m/%Y",
                currency_symbol="€",
                currency_code="EUR",
                region="europe"
            ),
            SupportedLocale.JA_JP: LocaleInfo(
                locale=SupportedLocale.JA_JP,
                display_name="Japanese (Japan)",
                native_name="日本語 (日本)",
                date_format="%Y/%m/%d",
                currency_symbol="¥",
                currency_code="JPY",
                region="asia"
            ),
            SupportedLocale.ZH_CN: LocaleInfo(
                locale=SupportedLocale.ZH_CN,
                display_name="Chinese Simplified (China)",
                native_name="中文 (简体，中国)",
                date_format="%Y年%m月%d日",
                currency_symbol="¥",
                currency_code="CNY",
                region="asia"
            ),
            SupportedLocale.ES_ES: LocaleInfo(
                locale=SupportedLocale.ES_ES,
                display_name="Spanish (Spain)",
                native_name="Español (España)",
                date_format="%d/%m/%Y",
                currency_symbol="€",
                currency_code="EUR",
                region="europe"
            ),
            SupportedLocale.PT_BR: LocaleInfo(
                locale=SupportedLocale.PT_BR,
                display_name="Portuguese (Brazil)",
                native_name="Português (Brasil)",
                date_format="%d/%m/%Y",
                currency_symbol="R$",
                currency_code="BRL",
                region="americas"
            ),
            SupportedLocale.AR_SA: LocaleInfo(
                locale=SupportedLocale.AR_SA,
                display_name="Arabic (Saudi Arabia)",
                native_name="العربية (المملكة العربية السعودية)",
                rtl=True,
                date_format="%d/%m/%Y",
                currency_symbol="﷼",
                currency_code="SAR",
                region="middle_east"
            ),
        }
    
    def _load_all_translations(self) -> None:
        """Load all available translations."""
        for locale_file in self.translations_dir.glob("*.json"):
            locale_code = locale_file.stem
            try:
                with open(locale_file, 'r', encoding='utf-8') as f:
                    self._translations[locale_code] = json.load(f)
                self.logger.info(f"Loaded translations for {locale_code}")
            except Exception as e:
                self.logger.error(f"Failed to load translations for {locale_code}: {e}")
    
    def _ensure_default_translations(self) -> None:
        """Create default translations if they don't exist."""
        default_translations = {
            # Application messages
            "app.name": "Spiking FPGA Toolchain",
            "app.description": "Advanced neuromorphic computing platform",
            "app.version": "Enterprise Edition",
            
            # Common UI elements
            "common.ok": "OK",
            "common.cancel": "Cancel",
            "common.submit": "Submit",
            "common.save": "Save",
            "common.delete": "Delete",
            "common.edit": "Edit",
            "common.create": "Create",
            "common.update": "Update",
            "common.close": "Close",
            "common.back": "Back",
            "common.next": "Next",
            "common.previous": "Previous",
            "common.loading": "Loading...",
            "common.error": "Error",
            "common.warning": "Warning",
            "common.success": "Success",
            "common.info": "Information",
            
            # Compilation messages
            "compilation.started": "Compilation started",
            "compilation.completed": "Compilation completed successfully",
            "compilation.failed": "Compilation failed",
            "compilation.progress": "Compilation in progress: {progress}%",
            "compilation.estimated_time": "Estimated time remaining: {time}",
            
            # Network messages
            "network.loaded": "Network loaded successfully",
            "network.invalid": "Invalid network format",
            "network.neurons": "Neurons: {count}",
            "network.synapses": "Synapses: {count}",
            "network.layers": "Layers: {count}",
            
            # FPGA messages  
            "fpga.target": "Target FPGA: {target}",
            "fpga.resources": "Resources: {luts} LUTs, {bram} BRAM",
            "fpga.utilization": "Utilization: {percent}%",
            "fpga.frequency": "Clock frequency: {freq} MHz",
            
            # Error messages
            "error.file_not_found": "File not found: {filename}",
            "error.permission_denied": "Permission denied: {filename}",
            "error.invalid_format": "Invalid file format: {format}",
            "error.compilation_failed": "Compilation failed: {reason}",
            "error.out_of_memory": "Insufficient memory for operation",
            "error.network_error": "Network communication error",
            
            # Security messages
            "security.threat_detected": "Security threat detected: {threat}",
            "security.access_denied": "Access denied",
            "security.authentication_required": "Authentication required",
            "security.authorization_failed": "Authorization failed",
            
            # Performance messages
            "performance.optimization_started": "Performance optimization started",
            "performance.benchmark_running": "Running performance benchmark",
            "performance.results": "Performance results: {metrics}",
            
            # Data compliance messages
            "compliance.gdpr_notice": "This application processes personal data in accordance with GDPR",
            "compliance.consent_required": "User consent required for data processing",
            "compliance.data_retention": "Data retention period: {period} days",
            "compliance.right_to_delete": "You have the right to request deletion of your data",
            
            # Regional messages
            "region.data_processing": "Data will be processed in {region}",
            "region.transfer_consent": "Data may be transferred to {countries} for processing",
            "region.local_storage": "Data is stored locally in {region}",
        }
        
        # Save default translations for English
        default_file = self.translations_dir / f"{SupportedLocale.EN_US.value}.json"
        if not default_file.exists():
            with open(default_file, 'w', encoding='utf-8') as f:
                json.dump(default_translations, f, indent=2, ensure_ascii=False)
            self.logger.info("Created default English translations")
        
        # Load the default translations
        self._translations[SupportedLocale.EN_US.value] = default_translations
    
    def set_locale(self, locale: Union[SupportedLocale, str]) -> None:
        """Set the current locale."""
        if isinstance(locale, str):
            try:
                locale = SupportedLocale(locale)
            except ValueError:
                self.logger.warning(f"Unsupported locale: {locale}, using default")
                locale = self.default_locale
        
        with self._lock:
            self.current_locale = locale
            self.logger.info(f"Locale set to {locale.value}")
    
    def get_current_locale(self) -> SupportedLocale:
        """Get the current locale."""
        return self.current_locale
    
    def get_supported_locales(self) -> List[SupportedLocale]:
        """Get list of supported locales."""
        return list(SupportedLocale)
    
    def get_locale_info(self, locale: Optional[SupportedLocale] = None) -> LocaleInfo:
        """Get locale information."""
        if locale is None:
            locale = self.current_locale
        
        return self._locale_info.get(locale, self._locale_info[self.default_locale])
    
    def translate(self, key: str, locale: Optional[SupportedLocale] = None, **kwargs) -> str:
        """Translate a message key to the current or specified locale."""
        if locale is None:
            locale = self.current_locale
        
        with self._lock:
            # Get translations for the locale
            locale_translations = self._translations.get(locale.value, {})
            
            # Try to get the translation
            translation = locale_translations.get(key)
            
            # Fallback to default locale if not found
            if translation is None:
                default_translations = self._translations.get(self.default_locale.value, {})
                translation = default_translations.get(key, key)  # Fallback to key itself
            
            # Format with provided arguments
            if kwargs:
                try:
                    translation = translation.format(**kwargs)
                except (KeyError, ValueError) as e:
                    self.logger.warning(f"Translation formatting error for key '{key}': {e}")
                    return key
            
            return translation
    
    def add_translation(self, locale: SupportedLocale, key: str, value: str) -> None:
        """Add or update a translation."""
        with self._lock:
            if locale.value not in self._translations:
                self._translations[locale.value] = {}
            
            self._translations[locale.value][key] = value
            
            # Save to file
            try:
                translation_file = self.translations_dir / f"{locale.value}.json"
                with open(translation_file, 'w', encoding='utf-8') as f:
                    json.dump(self._translations[locale.value], f, indent=2, ensure_ascii=False)
            except Exception as e:
                self.logger.error(f"Failed to save translation for {locale.value}: {e}")
    
    def bulk_add_translations(self, locale: SupportedLocale, translations: Dict[str, str]) -> None:
        """Add multiple translations at once."""
        with self._lock:
            if locale.value not in self._translations:
                self._translations[locale.value] = {}
            
            self._translations[locale.value].update(translations)
            
            # Save to file
            try:
                translation_file = self.translations_dir / f"{locale.value}.json"
                with open(translation_file, 'w', encoding='utf-8') as f:
                    json.dump(self._translations[locale.value], f, indent=2, ensure_ascii=False)
                    
                self.logger.info(f"Added {len(translations)} translations for {locale.value}")
            except Exception as e:
                self.logger.error(f"Failed to save translations for {locale.value}: {e}")
    
    def get_completion_percentage(self, locale: SupportedLocale) -> float:
        """Get translation completion percentage for a locale."""
        with self._lock:
            default_keys = set(self._translations.get(self.default_locale.value, {}).keys())
            locale_keys = set(self._translations.get(locale.value, {}).keys())
            
            if not default_keys:
                return 100.0
            
            return (len(locale_keys & default_keys) / len(default_keys)) * 100.0
    
    def get_missing_translations(self, locale: SupportedLocale) -> List[str]:
        """Get list of missing translation keys for a locale."""
        with self._lock:
            default_keys = set(self._translations.get(self.default_locale.value, {}).keys())
            locale_keys = set(self._translations.get(locale.value, {}).keys())
            
            return list(default_keys - locale_keys)
    
    def detect_locale_from_header(self, accept_language: str) -> SupportedLocale:
        """Detect preferred locale from Accept-Language header."""
        # Parse Accept-Language header (e.g., "en-US,en;q=0.9,de;q=0.8")
        language_ranges = []
        
        for lang_range in accept_language.split(','):
            parts = lang_range.strip().split(';')
            lang = parts[0].strip()
            quality = 1.0
            
            if len(parts) > 1:
                q_part = parts[1].strip()
                if q_part.startswith('q='):
                    try:
                        quality = float(q_part[2:])
                    except ValueError:
                        quality = 1.0
            
            language_ranges.append((lang, quality))
        
        # Sort by quality (preference)
        language_ranges.sort(key=lambda x: x[1], reverse=True)
        
        # Find best match
        for lang, _ in language_ranges:
            # Try exact match first
            for supported_locale in SupportedLocale:
                if supported_locale.value.lower() == lang.lower():
                    return supported_locale
            
            # Try language code match
            lang_code = lang.split('-')[0].lower()
            for supported_locale in SupportedLocale:
                if supported_locale.language_code.lower() == lang_code:
                    return supported_locale
        
        # Default fallback
        return self.default_locale
    
    def format_number(self, number: float, locale: Optional[SupportedLocale] = None) -> str:
        """Format number according to locale preferences."""
        if locale is None:
            locale = self.current_locale
        
        locale_info = self.get_locale_info(locale)
        
        # Simple formatting based on locale
        if ',' in locale_info.number_format and '.' in locale_info.number_format:
            # European style (1.234,56)
            if locale in [SupportedLocale.DE_DE, SupportedLocale.FR_FR]:
                return f"{number:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
        
        # Default US style (1,234.56)
        return f"{number:,.2f}"
    
    def format_currency(self, amount: float, locale: Optional[SupportedLocale] = None) -> str:
        """Format currency according to locale preferences."""
        if locale is None:
            locale = self.current_locale
        
        locale_info = self.get_locale_info(locale)
        formatted_number = self.format_number(amount, locale)
        
        # Currency symbol placement varies by locale
        if locale in [SupportedLocale.EN_US, SupportedLocale.EN_CA]:
            return f"{locale_info.currency_symbol}{formatted_number}"
        else:
            return f"{formatted_number} {locale_info.currency_symbol}"


# Global instance for easy access
_i18n_manager = None
_i18n_lock = threading.Lock()


def get_i18n_manager() -> InternationalizationManager:
    """Get the global internationalization manager instance."""
    global _i18n_manager
    
    with _i18n_lock:
        if _i18n_manager is None:
            _i18n_manager = InternationalizationManager()
        
        return _i18n_manager


def translate(key: str, locale: Optional[SupportedLocale] = None, **kwargs) -> str:
    """Convenience function for translation."""
    manager = get_i18n_manager()
    return manager.translate(key, locale, **kwargs)


def get_supported_locales() -> List[SupportedLocale]:
    """Get list of supported locales."""
    manager = get_i18n_manager()
    return manager.get_supported_locales()


def set_locale(locale: Union[SupportedLocale, str]) -> None:
    """Set the global locale."""
    manager = get_i18n_manager()
    manager.set_locale(locale)


def detect_locale(accept_language: str) -> SupportedLocale:
    """Detect locale from Accept-Language header."""
    manager = get_i18n_manager()
    return manager.detect_locale_from_header(accept_language)