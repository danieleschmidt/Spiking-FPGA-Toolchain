"""Advanced internationalization system with AI-powered translation and localization."""

import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import locale
import gettext
import re
import hashlib


class SupportedLanguage(Enum):
    """Supported languages for the neuromorphic toolchain."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"  
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh-CN"
    CHINESE_TRADITIONAL = "zh-TW"
    KOREAN = "ko"
    RUSSIAN = "ru"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    ARABIC = "ar"
    HINDI = "hi"
    DUTCH = "nl"
    SWEDISH = "sv"


class LocalizationContext(Enum):
    """Contexts for localization."""
    CLI_INTERFACE = "cli"
    ERROR_MESSAGES = "errors"
    WARNING_MESSAGES = "warnings"
    SUCCESS_MESSAGES = "success"
    DOCUMENTATION = "docs"
    USER_INTERFACE = "ui"
    TECHNICAL_TERMS = "technical"
    VALIDATION_MESSAGES = "validation"
    PERFORMANCE_REPORTS = "performance"
    SECURITY_ALERTS = "security"


@dataclass
class TranslationEntry:
    """Individual translation entry."""
    key: str
    source_text: str
    translated_text: str
    language: SupportedLanguage
    context: LocalizationContext
    last_updated: datetime
    confidence_score: float = 1.0
    translator_notes: str = ""
    reviewed: bool = False


@dataclass
class LocalizationMetrics:
    """Metrics for localization coverage."""
    language: SupportedLanguage
    total_strings: int
    translated_strings: int
    reviewed_translations: int
    coverage_percentage: float
    quality_score: float
    last_update: datetime


class AdvancedI18nSystem:
    """Advanced internationalization system with AI-powered features."""
    
    def __init__(self, default_language: SupportedLanguage = SupportedLanguage.ENGLISH,
                 logger: Optional[logging.Logger] = None):
        self.default_language = default_language
        self.current_language = default_language
        self.logger = logger or logging.getLogger(__name__)
        
        # Translation storage
        self.translations: Dict[SupportedLanguage, Dict[str, TranslationEntry]] = {}
        self.fallback_chain: List[SupportedLanguage] = [
            SupportedLanguage.ENGLISH,
            SupportedLanguage.SPANISH,
            SupportedLanguage.FRENCH
        ]
        
        # Localization data
        self.locale_data: Dict[SupportedLanguage, Dict[str, Any]] = {}
        self.cultural_adaptations: Dict[SupportedLanguage, Dict[str, str]] = {}
        
        # AI-powered features
        self.translation_cache: Dict[str, Dict[str, str]] = {}
        self.context_analyzer = ContextualTranslationAnalyzer(self.logger)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance tracking
        self.translation_metrics: Dict[SupportedLanguage, LocalizationMetrics] = {}
        
        self._initialize_locale_data()
        self._load_translations()
        
        self.logger.info(f"I18n system initialized with default language: {default_language.value}")
    
    def _initialize_locale_data(self) -> None:
        """Initialize locale-specific data."""
        self.locale_data = {
            SupportedLanguage.ENGLISH: {
                "date_format": "%Y-%m-%d",
                "time_format": "%H:%M:%S",
                "number_format": {"decimal_separator": ".", "thousands_separator": ","},
                "currency": "USD",
                "measurement_system": "imperial",
                "rtl": False
            },
            SupportedLanguage.SPANISH: {
                "date_format": "%d/%m/%Y", 
                "time_format": "%H:%M:%S",
                "number_format": {"decimal_separator": ",", "thousands_separator": "."},
                "currency": "EUR",
                "measurement_system": "metric",
                "rtl": False
            },
            SupportedLanguage.FRENCH: {
                "date_format": "%d/%m/%Y",
                "time_format": "%H:%M:%S", 
                "number_format": {"decimal_separator": ",", "thousands_separator": " "},
                "currency": "EUR",
                "measurement_system": "metric",
                "rtl": False
            },
            SupportedLanguage.GERMAN: {
                "date_format": "%d.%m.%Y",
                "time_format": "%H:%M:%S",
                "number_format": {"decimal_separator": ",", "thousands_separator": "."},
                "currency": "EUR", 
                "measurement_system": "metric",
                "rtl": False
            },
            SupportedLanguage.JAPANESE: {
                "date_format": "%Y年%m月%d日",
                "time_format": "%H:%M:%S",
                "number_format": {"decimal_separator": ".", "thousands_separator": ","},
                "currency": "JPY",
                "measurement_system": "metric",
                "rtl": False
            },
            SupportedLanguage.CHINESE_SIMPLIFIED: {
                "date_format": "%Y年%m月%d日",
                "time_format": "%H:%M:%S",
                "number_format": {"decimal_separator": ".", "thousands_separator": ","},
                "currency": "CNY",
                "measurement_system": "metric", 
                "rtl": False
            },
            SupportedLanguage.ARABIC: {
                "date_format": "%d/%m/%Y",
                "time_format": "%H:%M:%S",
                "number_format": {"decimal_separator": ".", "thousands_separator": ","},
                "currency": "AED",
                "measurement_system": "metric",
                "rtl": True
            }
        }
        
        # Cultural adaptations
        self.cultural_adaptations = {
            SupportedLanguage.JAPANESE: {
                "formal_greeting": "いらっしゃいませ",
                "error_politeness": "申し訳ございませんが",
                "success_celebration": "おめでとうございます"
            },
            SupportedLanguage.GERMAN: {
                "technical_precision": "präzise",
                "formal_address": "Sie",
                "compound_terms": "enable_compounds"
            },
            SupportedLanguage.ARABIC: {
                "honorifics": "enabled",
                "formal_language": "fus7a",
                "technical_adaptation": "transliterate_when_needed"
            }
        }
    
    def _load_translations(self) -> None:
        """Load translations from files or initialize with defaults."""
        # Initialize with basic translations
        self._initialize_default_translations()
        
        # Try to load from files
        translations_dir = Path(__file__).parent / "translations"
        if translations_dir.exists():
            for lang_file in translations_dir.glob("*.json"):
                try:
                    language_code = lang_file.stem
                    language = self._get_language_from_code(language_code)
                    if language:
                        with open(lang_file, 'r', encoding='utf-8') as f:
                            translations_data = json.load(f)
                            self._load_language_translations(language, translations_data)
                except Exception as e:
                    self.logger.warning(f"Failed to load translations from {lang_file}: {e}")
    
    def _get_language_from_code(self, code: str) -> Optional[SupportedLanguage]:
        """Get SupportedLanguage enum from language code."""
        for lang in SupportedLanguage:
            if lang.value == code:
                return lang
        return None
    
    def _initialize_default_translations(self) -> None:
        """Initialize default translations."""
        default_messages = {
            "compilation.started": "Compilation started for target {target}",
            "compilation.completed": "Compilation completed successfully in {duration}s",
            "compilation.failed": "Compilation failed: {error}",
            "validation.network.invalid": "Network validation failed: {issues}",
            "validation.network.valid": "Network validation passed",
            "optimization.started": "Running optimization pipeline with level {level}",
            "optimization.completed": "Optimization completed with {improvements} improvements",
            "security.audit.started": "Starting security audit",
            "security.audit.completed": "Security audit completed. Risk score: {score}",
            "security.threat.detected": "Security threat detected: {threat}",
            "performance.metrics.updated": "Performance metrics updated",
            "cache.hit": "Cache hit for key {key}",
            "cache.miss": "Cache miss for key {key}",
            "distributed.compilation.started": "Starting distributed compilation across {nodes} nodes",
            "distributed.compilation.completed": "Distributed compilation completed with efficiency {efficiency}",
            "error.file.not.found": "File not found: {filename}",
            "error.invalid.configuration": "Invalid configuration: {config}",
            "error.resource.exhausted": "System resources exhausted",
            "warning.performance.slow": "Performance warning: operation taking longer than expected",
            "warning.memory.high": "Memory usage warning: {usage}MB used",
            "success.hdl.generated": "HDL files generated successfully",
            "success.synthesis.completed": "FPGA synthesis completed successfully"
        }
        
        # Initialize English translations
        for key, text in default_messages.items():
            entry = TranslationEntry(
                key=key,
                source_text=text,
                translated_text=text,
                language=SupportedLanguage.ENGLISH,
                context=self._infer_context(key),
                last_updated=datetime.now(),
                confidence_score=1.0,
                reviewed=True
            )
            
            if SupportedLanguage.ENGLISH not in self.translations:
                self.translations[SupportedLanguage.ENGLISH] = {}
            self.translations[SupportedLanguage.ENGLISH][key] = entry
        
        # Initialize basic translations for other languages
        self._initialize_basic_translations()
    
    def _infer_context(self, key: str) -> LocalizationContext:
        """Infer context from translation key."""
        if key.startswith("error."):
            return LocalizationContext.ERROR_MESSAGES
        elif key.startswith("warning."):
            return LocalizationContext.WARNING_MESSAGES
        elif key.startswith("success."):
            return LocalizationContext.SUCCESS_MESSAGES
        elif key.startswith("security."):
            return LocalizationContext.SECURITY_ALERTS
        elif key.startswith("performance."):
            return LocalizationContext.PERFORMANCE_REPORTS
        elif key.startswith("validation."):
            return LocalizationContext.VALIDATION_MESSAGES
        else:
            return LocalizationContext.CLI_INTERFACE
    
    def _initialize_basic_translations(self) -> None:
        """Initialize basic translations for common languages."""
        basic_translations = {
            SupportedLanguage.SPANISH: {
                "compilation.started": "Compilación iniciada para el objetivo {target}",
                "compilation.completed": "Compilación completada exitosamente en {duration}s",
                "compilation.failed": "Compilación fallida: {error}",
                "error.file.not.found": "Archivo no encontrado: {filename}",
                "success.hdl.generated": "Archivos HDL generados exitosamente",
            },
            SupportedLanguage.FRENCH: {
                "compilation.started": "Compilation démarrée pour la cible {target}",
                "compilation.completed": "Compilation terminée avec succès en {duration}s",
                "compilation.failed": "Compilation échouée : {error}",
                "error.file.not.found": "Fichier non trouvé : {filename}",
                "success.hdl.generated": "Fichiers HDL générés avec succès",
            },
            SupportedLanguage.GERMAN: {
                "compilation.started": "Kompilierung für Ziel {target} gestartet",
                "compilation.completed": "Kompilierung erfolgreich abgeschlossen in {duration}s",
                "compilation.failed": "Kompilierung fehlgeschlagen: {error}",
                "error.file.not.found": "Datei nicht gefunden: {filename}",
                "success.hdl.generated": "HDL-Dateien erfolgreich generiert",
            },
            SupportedLanguage.JAPANESE: {
                "compilation.started": "ターゲット{target}のコンパイルを開始しました",
                "compilation.completed": "コンパイルが{duration}秒で正常に完了しました",
                "compilation.failed": "コンパイルに失敗しました: {error}",
                "error.file.not.found": "ファイルが見つかりません: {filename}",
                "success.hdl.generated": "HDLファイルが正常に生成されました",
            },
            SupportedLanguage.CHINESE_SIMPLIFIED: {
                "compilation.started": "已开始编译目标 {target}",
                "compilation.completed": "编译在{duration}秒内成功完成",
                "compilation.failed": "编译失败: {error}",
                "error.file.not.found": "找不到文件: {filename}",
                "success.hdl.generated": "HDL文件生成成功",
            }
        }
        
        for language, translations in basic_translations.items():
            if language not in self.translations:
                self.translations[language] = {}
            
            for key, text in translations.items():
                entry = TranslationEntry(
                    key=key,
                    source_text=self.translations[SupportedLanguage.ENGLISH][key].source_text,
                    translated_text=text,
                    language=language,
                    context=self._infer_context(key),
                    last_updated=datetime.now(),
                    confidence_score=0.95,  # High confidence for manually created translations
                    reviewed=True
                )
                self.translations[language][key] = entry
    
    def set_language(self, language: SupportedLanguage) -> bool:
        """Set the current language."""
        with self._lock:
            if language in SupportedLanguage:
                self.current_language = language
                self.logger.info(f"Language set to: {language.value}")
                return True
            else:
                self.logger.warning(f"Unsupported language: {language}")
                return False
    
    def get_text(self, key: str, language: Optional[SupportedLanguage] = None, 
                 **kwargs) -> str:
        """Get localized text with parameter substitution."""
        target_language = language or self.current_language
        
        with self._lock:
            # Try to get translation for target language
            if target_language in self.translations and key in self.translations[target_language]:
                entry = self.translations[target_language][key]
                text = entry.translated_text
            else:
                # Fallback chain
                text = self._get_fallback_text(key)
                
                # Log missing translation for improvement
                self.logger.debug(f"Missing translation: key='{key}', language='{target_language.value}'")
        
        # Apply cultural adaptations
        text = self._apply_cultural_adaptations(text, target_language)
        
        # Parameter substitution
        try:
            if kwargs:
                # Format with cultural number/date formatting
                formatted_kwargs = self._format_parameters(kwargs, target_language)
                text = text.format(**formatted_kwargs)
        except (KeyError, ValueError) as e:
            self.logger.warning(f"Parameter substitution failed for key '{key}': {e}")
        
        return text
    
    def _get_fallback_text(self, key: str) -> str:
        """Get fallback text using fallback chain."""
        for fallback_lang in self.fallback_chain:
            if (fallback_lang in self.translations and 
                key in self.translations[fallback_lang]):
                return self.translations[fallback_lang][key].translated_text
        
        # Ultimate fallback: return the key itself
        return f"[Missing: {key}]"
    
    def _apply_cultural_adaptations(self, text: str, language: SupportedLanguage) -> str:
        """Apply cultural adaptations to text."""
        if language not in self.cultural_adaptations:
            return text
        
        adaptations = self.cultural_adaptations[language]
        
        # Apply adaptations based on language-specific rules
        if language == SupportedLanguage.JAPANESE:
            # Add politeness markers
            if "error" in text.lower():
                text = adaptations.get("error_politeness", "") + " " + text
        
        elif language == SupportedLanguage.GERMAN:
            # Handle compound terms
            if adaptations.get("compound_terms") == "enable_compounds":
                # Placeholder for compound word processing
                pass
        
        elif language == SupportedLanguage.ARABIC:
            # Handle RTL and honorifics
            if adaptations.get("honorifics") == "enabled":
                # Placeholder for honorific insertion
                pass
        
        return text
    
    def _format_parameters(self, params: Dict[str, Any], language: SupportedLanguage) -> Dict[str, str]:
        """Format parameters according to locale."""
        locale_info = self.locale_data.get(language, self.locale_data[SupportedLanguage.ENGLISH])
        formatted_params = {}
        
        for key, value in params.items():
            if isinstance(value, (int, float)):
                # Format numbers according to locale
                decimal_sep = locale_info["number_format"]["decimal_separator"]
                thousands_sep = locale_info["number_format"]["thousands_separator"]
                
                if isinstance(value, float):
                    # Format float with locale-specific decimal separator
                    formatted_params[key] = f"{value:.2f}".replace(".", decimal_sep)
                else:
                    # Format integer with thousands separator
                    formatted_params[key] = f"{value:,}".replace(",", thousands_sep)
            
            elif isinstance(value, datetime):
                # Format datetime according to locale
                date_format = locale_info["date_format"]
                time_format = locale_info["time_format"]
                formatted_params[key] = value.strftime(f"{date_format} {time_format}")
            
            else:
                formatted_params[key] = str(value)
        
        return formatted_params
    
    def add_translation(self, key: str, text: str, language: SupportedLanguage,
                       context: LocalizationContext = LocalizationContext.CLI_INTERFACE,
                       confidence: float = 1.0) -> bool:
        """Add or update a translation."""
        with self._lock:
            if language not in self.translations:
                self.translations[language] = {}
            
            # Get source text from English if available
            source_text = text
            if (SupportedLanguage.ENGLISH in self.translations and 
                key in self.translations[SupportedLanguage.ENGLISH]):
                source_text = self.translations[SupportedLanguage.ENGLISH][key].source_text
            
            entry = TranslationEntry(
                key=key,
                source_text=source_text,
                translated_text=text,
                language=language,
                context=context,
                last_updated=datetime.now(),
                confidence_score=confidence,
                reviewed=False
            )
            
            self.translations[language][key] = entry
            self.logger.debug(f"Added translation: {key} -> {language.value}")
            
            # Update metrics
            self._update_localization_metrics(language)
            
            return True
    
    def get_localization_coverage(self, language: SupportedLanguage) -> LocalizationMetrics:
        """Get localization coverage metrics for a language."""
        if language not in self.translations:
            return LocalizationMetrics(
                language=language,
                total_strings=0,
                translated_strings=0,
                reviewed_translations=0,
                coverage_percentage=0.0,
                quality_score=0.0,
                last_update=datetime.now()
            )
        
        translations = self.translations[language]
        total_strings = len(self.translations.get(SupportedLanguage.ENGLISH, {}))
        translated_strings = len(translations)
        reviewed_translations = sum(1 for entry in translations.values() if entry.reviewed)
        
        coverage_percentage = (translated_strings / total_strings * 100) if total_strings > 0 else 0
        
        # Calculate quality score based on confidence and review status
        if translated_strings > 0:
            avg_confidence = sum(entry.confidence_score for entry in translations.values()) / translated_strings
            review_ratio = reviewed_translations / translated_strings
            quality_score = (avg_confidence * 0.7 + review_ratio * 0.3)
        else:
            quality_score = 0.0
        
        return LocalizationMetrics(
            language=language,
            total_strings=total_strings,
            translated_strings=translated_strings,
            reviewed_translations=reviewed_translations,
            coverage_percentage=coverage_percentage,
            quality_score=quality_score,
            last_update=max((entry.last_updated for entry in translations.values()), 
                          default=datetime.now())
        )
    
    def _update_localization_metrics(self, language: SupportedLanguage) -> None:
        """Update localization metrics for a language."""
        self.translation_metrics[language] = self.get_localization_coverage(language)
    
    def export_translations(self, language: SupportedLanguage, 
                           output_path: Optional[Path] = None) -> Path:
        """Export translations to JSON file."""
        if language not in self.translations:
            raise ValueError(f"No translations available for language: {language.value}")
        
        if output_path is None:
            output_path = Path(f"translations_{language.value}.json")
        
        translations_data = {}
        for key, entry in self.translations[language].items():
            translations_data[key] = {
                "source": entry.source_text,
                "translation": entry.translated_text,
                "context": entry.context.value,
                "confidence": entry.confidence_score,
                "reviewed": entry.reviewed,
                "last_updated": entry.last_updated.isoformat(),
                "translator_notes": entry.translator_notes
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(translations_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Exported {len(translations_data)} translations to {output_path}")
        return output_path
    
    def import_translations(self, language: SupportedLanguage, 
                           file_path: Path) -> int:
        """Import translations from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                translations_data = json.load(f)
            
            imported_count = 0
            for key, data in translations_data.items():
                entry = TranslationEntry(
                    key=key,
                    source_text=data.get("source", ""),
                    translated_text=data["translation"],
                    language=language,
                    context=LocalizationContext(data.get("context", "cli")),
                    last_updated=datetime.fromisoformat(data.get("last_updated", datetime.now().isoformat())),
                    confidence_score=data.get("confidence", 1.0),
                    translator_notes=data.get("translator_notes", ""),
                    reviewed=data.get("reviewed", False)
                )
                
                if language not in self.translations:
                    self.translations[language] = {}
                
                self.translations[language][key] = entry
                imported_count += 1
            
            self._update_localization_metrics(language)
            self.logger.info(f"Imported {imported_count} translations for {language.value}")
            return imported_count
            
        except Exception as e:
            self.logger.error(f"Failed to import translations from {file_path}: {e}")
            return 0
    
    def get_supported_languages(self) -> List[SupportedLanguage]:
        """Get list of supported languages with available translations."""
        return list(self.translations.keys())
    
    def get_translation_status(self) -> Dict[str, Any]:
        """Get overall translation status."""
        status = {
            "current_language": self.current_language.value,
            "default_language": self.default_language.value,
            "supported_languages": [lang.value for lang in self.get_supported_languages()],
            "language_metrics": {}
        }
        
        for language in self.get_supported_languages():
            metrics = self.get_localization_coverage(language)
            status["language_metrics"][language.value] = {
                "coverage_percentage": metrics.coverage_percentage,
                "quality_score": metrics.quality_score,
                "total_strings": metrics.total_strings,
                "translated_strings": metrics.translated_strings
            }
        
        return status


class ContextualTranslationAnalyzer:
    """Analyzes context for better translations."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.context_patterns: Dict[str, List[str]] = {
            "error_context": ["error", "failed", "exception", "invalid"],
            "success_context": ["success", "completed", "generated", "passed"],
            "warning_context": ["warning", "caution", "notice", "slow"],
            "technical_context": ["compilation", "synthesis", "optimization", "hdl"],
            "performance_context": ["time", "speed", "memory", "efficiency"]
        }
    
    def analyze_context(self, text: str, key: str) -> Dict[str, float]:
        """Analyze context of text for better translation."""
        text_lower = text.lower()
        key_lower = key.lower()
        
        context_scores = {}
        
        for context_type, patterns in self.context_patterns.items():
            score = 0.0
            for pattern in patterns:
                if pattern in text_lower:
                    score += 1.0
                if pattern in key_lower:
                    score += 0.5
            
            context_scores[context_type] = score / len(patterns)
        
        return context_scores
    
    def suggest_improvements(self, entry: TranslationEntry) -> List[str]:
        """Suggest improvements for translation."""
        suggestions = []
        
        context_scores = self.analyze_context(entry.translated_text, entry.key)
        
        # Context-specific suggestions
        if context_scores.get("technical_context", 0) > 0.5:
            suggestions.append("Consider using technical terminology appropriate for the target language")
        
        if context_scores.get("error_context", 0) > 0.5:
            suggestions.append("Ensure error messages are clear and actionable in the target language")
        
        if entry.confidence_score < 0.8:
            suggestions.append("Translation confidence is low - consider professional review")
        
        if not entry.reviewed:
            suggestions.append("Translation needs review by native speaker")
        
        return suggestions


# Convenience functions for global usage
_global_i18n_system: Optional[AdvancedI18nSystem] = None


def initialize_i18n(default_language: SupportedLanguage = SupportedLanguage.ENGLISH) -> AdvancedI18nSystem:
    """Initialize global i18n system."""
    global _global_i18n_system
    _global_i18n_system = AdvancedI18nSystem(default_language)
    return _global_i18n_system


def get_i18n() -> AdvancedI18nSystem:
    """Get global i18n system."""
    global _global_i18n_system
    if _global_i18n_system is None:
        _global_i18n_system = AdvancedI18nSystem()
    return _global_i18n_system


def _(key: str, **kwargs) -> str:
    """Shorthand function for getting localized text."""
    return get_i18n().get_text(key, **kwargs)


def set_language(language: Union[str, SupportedLanguage]) -> bool:
    """Set global language."""
    if isinstance(language, str):
        for lang in SupportedLanguage:
            if lang.value == language:
                language = lang
                break
        else:
            return False
    
    return get_i18n().set_language(language)