from datetime import date, time

from marseille_agenda.filters import is_special_screening, publishable
from marseille_agenda.schema import Event, Venue


def _ev(title, summary=None, evidence=()):
    return Event(uid="x", venue_id="c", venue_name="C", category="cinema", title=title, start_date=date(2026, 9, 20),
                 start_time=time(20, 0), summary=summary, source_url="https://x.test/", source_kind="html",
                 evidence=list(evidence), first_seen=date(2026, 9, 13), last_seen=date(2026, 9, 13))


def test_regular_screenings_are_not_special():
    for e in [
        _ev("Séance Jeune public 11h Lydia et le vaisseau des tempêtes", evidence=["Séance Jeune public 11h Lydia et le vaisseau des tempêtes De Nancy Florence Savard"]),
        _ev("La Pat’Patrouille : Le Film Mission Dino", "VF À partir de 3 ans Billetterie"),
        _ev("LA LIGNE BLEUE de Marie Dumora", "Projection du film documentaire 'La Ligne bleue' de Marie Dumora"),
    ]:
        assert not is_special_screening(e), e.title


def test_special_sessions_are_kept():
    for e in [
        _ev("L’aventure rêvée", evidence=["Retour de Cannes 2026 Séance Séance spéciale 19h L’aventure rêvée"]),
        _ev("[rdv doc] HAIR, PAPER, WATER...", "▉ En présence de Arnaud Alain et Alyzée Soudet du Labo Largent"),
        _ev("HANTEES ciné-club : REBECCA d'Hitchcock"),
        _ev("MERCI D'ÊTRE VENU d'Alain Cavalier", "Avant-première spéciale Anniversaire de La Baleine !"),
        _ev("FATHERLAND", "Opening night of the Kinovisions festival with the premiere of Pawel P."),
        _ev("Séance unique : Teenage Sex and Death at Camp Miasma"),
    ]:
        assert is_special_screening(e), e.title


def test_rule_only_applies_to_cinemas():
    regular = _ev("Un film")
    kept, dropped = publishable(Venue(name="Ciné", category="cinema"), [regular])
    assert (kept, dropped) == ([], 1)
    kept, dropped = publishable(Venue(name="Bar", category="bars"), [regular])
    assert (kept, dropped) == ([regular], 0)
