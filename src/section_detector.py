# IBM 10-K RAG System - Section Detector
# Detects which section of a 10-K filing a chunk belongs to

import re
from typing import Optional, Dict, List

class SectionDetector:
    """
    Detects 10-K sections from chunk text for taxonomy-aware retrieval.

    Standard 10-K sections:
    - Item 1: Business
    - Item 1A: Risk Factors
    - Item 7: MD&A (Management's Discussion and Analysis)
    - Item 7A: Market Risk
    - Item 8: Financial Statements
    """

    # Section patterns for IBM 10-K
    SECTION_PATTERNS: Dict[str, List[str]] = {
        'risk_factors': [
            'risk factors',
            'risks relating to',
            'risk related to',
            'principal risks',
            'key risks',
            'item 1a',
        ],
        'business': [
            'business overview',
            'our business',
            'company overview',
            'business description',
            'item 1 ',
            'item 1.',
        ],
        'revenue': [
            'revenue',
            'segment results',
            'financial results',
            'results of operations',
            'net income',
            'earnings',
        ],
        'strategy': [
            'strategy',
            'strategic',
            'business strategy',
            'growth strategy',
            'competitive strategy',
        ],
        'technology': [
            'research and development',
            'technology',
            'innovation',
            'artificial intelligence',
            ' ai ',
            'machine learning',
            'cloud computing',
        ],
        'legal': [
            'legal proceedings',
            'litigation',
            'regulatory',
            'compliance',
            'item 3',
        ],
        'mda': [
            "management's discussion",
            'md&a',
            'item 7',
            'liquidity',
            'capital resources',
        ],
        'financial_statements': [
            'consolidated balance',
            'consolidated statement',
            'notes to consolidated',
            'item 8',
            'financial statements',
        ],
        'executive': [
            'dear ibm investor',
            'dear shareholder',
            'letter to shareholders',
            'ceo letter',
            'chairman letter',
        ],
    }

    # Keywords that suggest user wants specific sections
    QUERY_SECTION_HINTS: Dict[str, str] = {
        'risk': 'risk_factors',
        'danger': 'risk_factors',
        'threat': 'risk_factors',
        'challenge': 'risk_factors',
        'concern': 'risk_factors',
        'revenue': 'revenue',
        'income': 'revenue',
        'sales': 'revenue',
        'profit': 'revenue',
        'earnings': 'revenue',
        'strategy': 'strategy',
        'plan': 'strategy',
        'goal': 'strategy',
        'objective': 'strategy',
        'technology': 'technology',
        'ai': 'technology',
        'research': 'technology',
        'innovation': 'technology',
        'cloud': 'technology',
        'legal': 'legal',
        'lawsuit': 'legal',
        'litigation': 'legal',
        'regulatory': 'legal',
    }

    def __init__(self):
        """Initialize section detector."""
        pass

    def detect_section(self, text: str) -> str:
        """
        Detect which 10-K section a chunk belongs to.

        Args:
            text: Chunk text to analyze

        Returns:
            Section name (e.g., 'risk_factors', 'revenue') or 'other'
        """
        text_lower = text.lower()

        # Check each section's patterns
        for section, patterns in self.SECTION_PATTERNS.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return section

        return 'other'

    def detect_query_section(self, query: str) -> Optional[str]:
        """
        Detect which section a user query is likely targeting.

        Args:
            query: User's question

        Returns:
            Target section name or None if no clear match
        """
        query_lower = query.lower()

        for keyword, section in self.QUERY_SECTION_HINTS.items():
            if keyword in query_lower:
                return section

        return None

    def apply_section_boost(
        self,
        query: str,
        results: List[tuple],
        boost_factor: float = 1.3
    ) -> List[tuple]:
        """
        Boost chunks from sections relevant to the query.

        Args:
            query: User's question
            results: List of (chunk, score) tuples
            boost_factor: Multiplier for matching sections (default 1.3 = 30% boost)

        Returns:
            Re-sorted results with section boosts applied
        """
        target_section = self.detect_query_section(query)

        if not target_section:
            return results

        # Apply boost to matching sections
        boosted = []
        for chunk, score in results:
            chunk_section = chunk.get('section', 'other')
            if chunk_section == target_section:
                boosted.append((chunk, score * boost_factor))
            else:
                boosted.append((chunk, score))

        # Re-sort by boosted scores
        boosted.sort(key=lambda x: x[1], reverse=True)
        return boosted

    def get_section_display_name(self, section: str) -> str:
        """Get human-readable section name."""
        display_names = {
            'risk_factors': 'Risk Factors',
            'business': 'Business Overview',
            'revenue': 'Revenue & Financials',
            'strategy': 'Business Strategy',
            'technology': 'Technology & R&D',
            'legal': 'Legal & Regulatory',
            'mda': "Management's Discussion",
            'financial_statements': 'Financial Statements',
            'executive': 'Executive Letter',
            'other': 'Other',
        }
        return display_names.get(section, section.replace('_', ' ').title())
