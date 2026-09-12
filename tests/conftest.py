from datetime import date
from pathlib import Path

import pytest

from marseille_agenda.fetch import SourceDocument, document_from_body

FIXTURES = Path(__file__).parent / "fixtures"
TODAY = date(2026, 9, 12)


@pytest.fixture
def today() -> date:
    return TODAY


@pytest.fixture
def amis_diplo_doc() -> SourceDocument:
    body = (FIXTURES / "amis_diplo" / "marseille.html").read_text(encoding="utf-8")
    return document_from_body("https://www.amis.monde-diplomatique.fr/-Marseille-.html", "html", body)


@pytest.fixture
def mucem_doc() -> SourceDocument:
    body = (FIXTURES / "mucem" / "evenement_upcoming.json").read_text(encoding="utf-8")
    return document_from_body("https://mucem.org/api/mainApi/posts/evenement?upcoming=1&perPage=100", "json", body)
