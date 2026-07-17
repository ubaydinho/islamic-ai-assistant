"""
Islamic AI Assistant — Islamic Content Filter

Filter untuk memvalidasi content berdasarkan ajaran Islam.
Menggunakan kombinasi hardcoded blacklist patterns dan ML-based scoring.

Pipeline:
1. Hard blacklist check (CRITICAL violations)
2. ML-based keyword scoring (shirk, bidah, misleading)
3. Confidence calculation
4. Final verdict: is_halal boolean
"""

import re
from typing import Optional
from enum import Enum

from src.models.schemas import (
    FilterResult,
    Violation,
    ViolationCategory,
    IntentCategory
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

class SeverityLevel(str, Enum):
    """Severity level untuk violations"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class FilterMode(str, Enum):
    """Mode filter untuk query vs response"""
    QUERY = "query"
    RESPONSE = "response"

# ============================================================
# HARDCODED BLACKLIST PATTERNS (CRITICAL VIOLATIONS)
# ============================================================

HARD_PROHIBITED_PATTERNS = {
    # Shirk patterns (menyekutukan Allah)
    "shirk": [
        r"\b(menyembah|ibadah kepada|berdoa kepada)\s+(patung|berhala|kuburan|jin|setan|makhluk)",
        r"\btuhan\s+(banyak|lebih dari satu|tiga|trinity)",
        r"\b(yesus|isa)\s+(adalah|itu)\s+(tuhan|allah|god)",
        r"\bminta\s+pertolongan\s+kepada\s+(orang\s+mati|arwah|roh)",
        r"\bAllah\s+(punya|memiliki)\s+anak",
    ],

    # Bid'ah patterns (inovasi dalam agama)
    "bidah": [
        r"\bmaulid\s+nabi\s+(wajib|sunnah\s+muakkad)",
        r"\btahlilan\s+40\s+hari\s+(wajib|sunnah)",
        r"\bqunut\s+subuh\s+setiap\s+hari",  # kontroversial antar mazhab
        r"\bwirid\s+yang\s+tidak\s+ada\s+dalam\s+sunnah",
    ],

    # Haram content patterns
    "haram": [
        r"\b(minum|jual|beli)\s+(alkohol|arak|minuman\s+keras|khamr)",
        r"\b(makan|jual)\s+(babi|daging\s+babi)",
        r"\b(riba|bunga|interest)\s+(halal|boleh|diperbolehkan)",
        r"\b(judi|gambling|taruhan)\s+(halal|boleh)",
        r"\b(zina|prostitusi|pelacuran)\s+(halal|boleh)",
        r"\bmembunuh\s+(muslim|orang\s+tidak\s+bersalah)",
    ],

    # Misleading patterns (ajaran sesat)
    "misleading": [
        r"\bnabi\s+setelah\s+(muhammad|nabi\s+terakhir)",
        r"\bal-quran\s+(salah|keliru|perlu\s+direvisi)",
        r"\bhadits?\s+(semua|seluruh)nya\s+tidak\s+valid",
        r"\bshalat\s+lima\s+waktu\s+(tidak\s+wajib|opsional)",
        r"\bhijab\s+tidak\s+wajib",  # kontroversial tapi mainstream wajib
    ]
}

class IslamicFilter:
    """
    Filter content berdasarkan ajaran Islam.
    Kombinasi rule-based (blacklist) dan heuristic scoring.
    """

    def __init__(self):
        """Initialize Islamic filter dengan compiled regex patterns"""

        # Compile regex patterns untuk performa
        self.compiled_patterns = {}
        for category, patterns in HARD_PROHIBITED_PATTERNS.items():
            self.compiled_patterns[category] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

        logger.info(
            f"Islamic filter initialized with "
            f"{sum(len(p) for p in HARD_PROHIBITED_PATTERNS.values())} patterns"
        )

    def check_hard_blacklist(self, text: str) -> list[Violation]:
        """
        Check text terhadap hardcoded blacklist patterns.
        Jika ada match → CRITICAL violation.

        Args:
            text: Text yang akan di-check

        Returns:
            List Violation yang ditemukan (bisa kosong)
        """
        violations = []

        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    violation = Violation(
                        category=ViolationCategory(category.upper()),
                        severity=SeverityLevel.CRITICAL,
                        confidence=1.0,  # Hard match = confidence 100%
                        reason=f"Matched blacklist pattern: '{match.group()}'",
                        matched_text=match.group()
                    )
                    violations.append(violation)
                    logger.warning(
                        f"Hard blacklist violation detected: "
                        f"{category} - '{match.group()}'"
                    )

        return violations

    def _calculate_keyword_scores(self, text: str) -> dict[str, float]:
        """
        Hitung scores berdasarkan keyword heuristics.

        Args:
            text: Text yang akan di-score

        Returns:
            Dictionary dengan scores per kategori (0.0 - 1.0)
        """
        text_lower = text.lower()

        # Shirk indicators (menyekutukan Allah)
        shirk_keywords = [
            "tuhan banyak", "menyembah selain", "berdoa kepada jin",
            "patung suci", "trinity", "tiga tuhan"
        ]
        shirk_score = sum(
            1 for kw in shirk_keywords if kw in text_lower
        ) / max(len(shirk_keywords), 1)

        # Bid'ah indicators (inovasi dalam agama yang tidak ada dasarnya)
        bidah_keywords = [
            "ibadah baru", "tidak ada dalam sunnah tapi wajib",
            "tambahan ibadah yang wajib", "wirid yang diada-adakan"
        ]
        bidah_score = sum(
            1 for kw in bidah_keywords if kw in text_lower
        ) / max(len(bidah_keywords), 1)

        # Misleading indicators (ajaran yang menyesatkan)
        misleading_keywords = [
            "nabi setelah muhammad", "al-quran salah", "hadits palsu semua",
            "shalat tidak wajib", "puasa tidak wajib"
        ]
        misleading_score = sum(
            1 for kw in misleading_keywords if kw in text_lower
        ) / max(len(misleading_keywords), 1)

        return {
            "shirk_score": min(shirk_score, 1.0),
            "bidah_score": min(bidah_score, 1.0),
            "misleading_score": min(misleading_score, 1.0)
        }

    # TO BE CONTINUED...
    # (Bagian 2 akan berisi: classify_intent, calculate_confidence,
    # filter_query, filter_response methods)

    def classify_intent(self, text: str) -> IntentCategory:
        """
        Klasifikasi intent query berdasarkan keyword matching.

        Args:
            text: Query text

        Returns:
            IntentCategory yang paling sesuai
        """
        text_lower = text.lower()

        # Intent keyword mapping
        intent_keywords = {
            IntentCategory.QURAN_TAFSIR: [
                "tafsir", "arti ayat", "makna ayat", "surah", "al-quran",
                "quran", "ayat tentang", "penjelasan ayat", "ayat kursi",
                "al-baqarah", "al-fatihah", "al-ikhlas", "jelaskan ayat",
                "maksud ayat", "qs", "surat", "juz"
            ],
            IntentCategory.HADITH_LOOKUP: [
                "hadits", "hadith", "riwayat", "sabda nabi", "rasulullah bersabda",
                "diriwayatkan", "bukhari", "muslim", "hr", "shahih"
            ],
            IntentCategory.FIQH_QUESTION: [
                "hukum", "boleh", "halal", "haram", "makruh", "mubah",
                "diperbolehkan", "dilarang", "wajib", "sunnah", "fiqh", "fiqih"
            ],
            IntentCategory.IBADAH_GUIDE: [
                "cara shalat", "cara wudhu", "cara puasa", "cara haji",
                "tata cara", "rukun", "syarat", "niat", "sholat", "solat",
                "dzikir", "doa", "zakat", "umrah"
            ],
            IntentCategory.AQEEDAH: [
                "akidah", "aqidah", "iman", "tauhid", "rukun iman",
                "sifat allah", "nama allah", "keyakinan", "asmaul husna"
            ],
            IntentCategory.DAWA_CONTENT: [
                "dakwah", "nasihat", "motivasi", "ceramah", "tausiyah",
                "hikmah", "pelajaran", "ibrah", "kisah"
            ]
        }

        # Score setiap intent
        intent_scores = {}
        for intent, keywords in intent_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                intent_scores[intent] = score

        # Return intent dengan score tertinggi
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            logger.debug(f"Classified intent: {best_intent.value}")
            return best_intent

        # Default: GENERAL_ISLAMIC jika tidak match pattern tertentu
        # Cek apakah ada kata-kata Islam umum
        general_islamic_keywords = [
            "islam", "muslim", "allah", "nabi", "rasul", "agama"
        ]
        if any(kw in text_lower for kw in general_islamic_keywords):
            return IntentCategory.GENERAL_ISLAMIC

        # Jika tidak ada keyword Islam sama sekali
        return IntentCategory.OUT_OF_SCOPE

    def calculate_confidence(
        self,
        scores: dict[str, float],
        violations: list[Violation]
    ) -> float:
        """
        Hitung confidence score dari violation scores.
        Pastikan output selalu dalam rentang [0.0, 1.0].

        Args:
            scores: Dictionary dengan shirk_score, bidah_score, misleading_score
            violations: List violations yang ditemukan

        Returns:
            Confidence score (0.0 - 1.0)
        """
        # Jika ada CRITICAL violation, confidence = 1.0
        if any(v.severity == SeverityLevel.CRITICAL for v in violations):
            return 1.0

        # Jika tidak ada violation, return 0.0 (content aman)
        if not violations and all(score == 0.0 for score in scores.values()):
            return 0.0

        # Weighted average dari scores
        weights = {
            "shirk_score": 0.5,      # Shirk paling berat
            "bidah_score": 0.3,      # Bid'ah sedang
            "misleading_score": 0.2  # Misleading relatif ringan
        }

        weighted_sum = sum(
            scores.get(key, 0.0) * weight
            for key, weight in weights.items()
        )

        # Clamp ke [0.0, 1.0]
        confidence = max(0.0, min(1.0, weighted_sum))

        return confidence

    def filter_query(
        self,
        text: str,
        threshold: float = 0.7
    ) -> FilterResult:
        """
        Filter user query untuk validasi sebelum processing.

        Args:
            text: Query text dari user
            threshold: Confidence threshold untuk blocking (default: 0.7)

        Returns:
            FilterResult dengan verdict is_halal dan list violations
        """
        logger.debug(f"Filtering query (mode=QUERY, threshold={threshold})")

        # Step 1: Hard blacklist check
        violations = self.check_hard_blacklist(text)

        # Jika ada CRITICAL violation → langsung block
        if any(v.severity == SeverityLevel.CRITICAL for v in violations):
            return FilterResult(
                is_halal=False,
                confidence=1.0,
                violations=violations,
                intent=IntentCategory.OUT_OF_SCOPE,
                suggested_redirect="Query contains CRITICAL Islamic violations"
            )

        # Step 2: ML-based keyword scoring
        scores = self._calculate_keyword_scores(text)

        # Step 3: Classify intent
        intent = self.classify_intent(text)

        # Step 4: Calculate confidence
        confidence = self.calculate_confidence(scores, violations)

        # Step 5: Final verdict
        is_halal = confidence < threshold

        suggested_redirect = ""
        if not is_halal:
            suggested_redirect = (
                f"Query flagged with confidence {confidence:.2f} "
                f"(threshold: {threshold})"
            )

        result = FilterResult(
            is_halal=is_halal,
            confidence=confidence,
            violations=violations,
            intent=intent,
            suggested_redirect=suggested_redirect
        )

        logger.info(
            f"Query filter result: is_halal={is_halal}, "
            f"confidence={confidence:.2f}, intent={intent.value}"
        )

        return result

    def filter_response(
        self,
        text: str,
        threshold: float = 0.5
    ) -> FilterResult:
        """
        Filter LLM response untuk validasi sebelum dikirim ke user.
        Lebih strict: violation severity > LOW akan di-block.

        Args:
            text: Response text dari LLM
            threshold: Confidence threshold untuk blocking (default: 0.5)

        Returns:
            FilterResult dengan verdict is_halal dan list violations
        """
        logger.debug(f"Filtering response (mode=RESPONSE, threshold={threshold})")

        # Step 1: Hard blacklist check
        violations = self.check_hard_blacklist(text)

        # Step 2: ML-based keyword scoring
        scores = self._calculate_keyword_scores(text)

        # Step 3: Calculate confidence
        confidence = self.calculate_confidence(scores, violations)

        # Step 4: Final verdict
        # Response mode lebih strict: block jika ada violation > LOW severity
        has_serious_violation = any(
            v.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH, SeverityLevel.MEDIUM]
            for v in violations
        )

        is_halal = not has_serious_violation and confidence < threshold

        suggested_redirect = ""
        reason = None
        if not is_halal:
            if has_serious_violation:
                suggested_redirect = "Response contains serious Islamic violations"
            else:
                suggested_redirect = (
                    f"Response flagged with confidence {confidence:.2f} "
                    f"(threshold: {threshold})"
                )

        result = FilterResult(
            is_halal=is_halal,
            confidence=confidence,
            violations=violations,
            intent=IntentCategory.GENERAL_ISLAMIC,  # Intent only for queries, use general for responses
            suggested_redirect=suggested_redirect
        )

        logger.info(
            f"Response filter result: is_halal={is_halal}, "
            f"confidence={confidence:.2f}"
        )

        return result

# Convenience function untuk akses singleton
_filter_instance: Optional[IslamicFilter] = None

def get_islamic_filter() -> IslamicFilter:
    """
    Get singleton instance dari IslamicFilter.

    Returns:
        IslamicFilter instance

    Example:
        >>> from src.core.islamic_filter import get_islamic_filter
        >>> filter = get_islamic_filter()
        >>> result = filter.filter_query("Bolehkah shalat tanpa wudhu?")
        >>> result.is_halal
        True
    """
    global _filter_instance
    if _filter_instance is None:
        _filter_instance = IslamicFilter()
    return _filter_instance
