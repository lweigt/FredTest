"""
AVM-System Backend – FastAPI + Claude API
Hybrides KI-Bewertungssystem für Wohnimmobilien (Deutschland)
"""
 
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import anthropic
import os
from dotenv import load_dotenv
 
load_dotenv()
 
app = FastAPI(
    title="AVM-System – KI-Immobilienbewertung",
    description="Hybrides, erklärbares Bewertungssystem für Wohnimmobilien in Deutschland",
    version="1.0.0"
)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
 
 
# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────
 
class ObjektInput(BaseModel):
    flaeche: float = Field(..., ge=20, le=1000, description="Wohnfläche in m²")
    baujahr: int = Field(..., ge=1850, le=2024, description="Baujahr des Gebäudes")
    lage: str = Field(..., description="Lageklasse: 1a | 1b | 2 | 3")
    zustand: str = Field(..., description="neuwertig | gut | mittel | schlecht")
    ausstattung: str = Field(..., description="hoch | mittel | einfach")
    objekttyp: str = Field(..., description="etw | einfh | doppelh | reihenh | mfh | denkmal | sonder")
    markt: str = Field(..., description="gross | mittel | land")
    zimmer: Optional[int] = Field(None, ge=1, le=20)
    etage: Optional[int] = Field(None, ge=0, le=30)
    aufzug: Optional[bool] = False
    keller: Optional[bool] = False
    balkon: Optional[bool] = False
    garage: Optional[bool] = False
    energieklasse: Optional[str] = Field(None, description="A+ | A | B | C | D | E | F | G | H")
    besonderheiten: Optional[List[str]] = Field(default_factory=list,
        description="Liste: erbbau | mieter | altlast | mikrolage | datenschwach | denkmal_auflagen | rechtliche_belastung")
    ort: Optional[str] = Field(None, description="Optionaler Ortsname für Kontextualisierung")
 
 
class ModulErgebnis(BaseModel):
    marktwert: float
    marktwert_min: float
    marktwert_max: float
    qm_preis: float
    konfidenz: float
    faktoren: dict
 
 
class HumanAnalogueErgebnis(BaseModel):
    plausibel: bool
    kohaerenz: bool
    marktgaengig: bool
    risiko_score: int
    flags: List[dict]
    pruefschritte: List[dict]
 
 
class XAIErgebnis(BaseModel):
    werttreiber: List[dict]
    begruendung: str
    methodik: str
 
 
class GovernanceErgebnis(BaseModel):
    stufe: str
    farbe: str
    titel: str
    empfehlung: str
    gruende: List[str]
    protokoll: dict
 
 
class BewertungsAntwort(BaseModel):
    basis: ModulErgebnis
    human_analogue: HumanAnalogueErgebnis
    xai: XAIErgebnis
    governance: GovernanceErgebnis
    ki_analyse: Optional[str] = None
 
 
# ─────────────────────────────────────────────
# MODUL 1 – BASISMODUL (Hedonic Pricing)
# ─────────────────────────────────────────────
 
BASISPREISE = {
    "gross":  {"1a": 7800, "1b": 5600, "2": 3900, "3": 2600},
    "mittel": {"1a": 5000, "1b": 3800, "2": 2700, "3": 1900},
    "land":   {"1a": 3200, "1b": 2400, "2": 1750, "3": 1250},
}
 
ZUSTAND_FAKTOREN     = {"neuwertig": 1.14, "gut": 1.0, "mittel": 0.86, "schlecht": 0.70}
AUSSTATTUNG_FAKTOREN = {"hoch": 1.12, "mittel": 1.0, "einfach": 0.87}
TYP_FAKTOREN         = {"etw": 1.0, "einfh": 1.08, "doppelh": 1.03, "reihenh": 0.97,
                         "mfh": 0.91, "denkmal": 0.84, "sonder": 0.76}
ENERGIE_FAKTOREN     = {"A+": 1.06, "A": 1.04, "B": 1.02, "C": 1.0,
                         "D": 0.98, "E": 0.95, "F": 0.91, "G": 0.87, "H": 0.82}
 
def modul_basis(inp: ObjektInput) -> ModulErgebnis:
    basispreis = BASISPREISE.get(inp.markt, BASISPREISE["mittel"]).get(inp.lage, 3000)
    alter = 2024 - inp.baujahr
    if alter < 3:
        alter_faktor = 1.08
    elif alter < 15:
        alter_faktor = 1.0 - alter * 0.003
    else:
        alter_faktor = max(0.62, 1.0 - 0.045 - alter * 0.005)
 
    zustand_f     = ZUSTAND_FAKTOREN.get(inp.zustand, 1.0)
    ausstattung_f = AUSSTATTUNG_FAKTOREN.get(inp.ausstattung, 1.0)
    typ_f         = TYP_FAKTOREN.get(inp.objekttyp, 1.0)
    energie_f     = ENERGIE_FAKTOREN.get(inp.energieklasse, 1.0) if inp.energieklasse else 1.0
 
    bonus = 0.0
    if inp.balkon: bonus += 0.02
    if inp.garage: bonus += 0.025
    if inp.keller: bonus += 0.01
    if inp.aufzug and (inp.etage or 0) > 2: bonus += 0.015
 
    qm_preis  = basispreis * alter_faktor * zustand_f * ausstattung_f * typ_f * energie_f * (1 + bonus)
    marktwert = qm_preis * inp.flaeche
 
    konfidenz = 0.82
    if inp.objekttyp in ["denkmal", "sonder"]:              konfidenz -= 0.22
    if inp.markt == "land":                                  konfidenz -= 0.10
    if "datenschwach" in (inp.besonderheiten or []):         konfidenz -= 0.18
    if "altlast"      in (inp.besonderheiten or []):         konfidenz -= 0.15
    if "mikrolage"    in (inp.besonderheiten or []):         konfidenz -= 0.08
    konfidenz = max(0.25, min(0.95, konfidenz))
 
    spanne = 1 - konfidenz
    return ModulErgebnis(
        marktwert     = round(marktwert / 1000) * 1000,
        marktwert_min = round(marktwert * (1 - spanne * 0.7) / 1000) * 1000,
        marktwert_max = round(marktwert * (1 + spanne * 0.7) / 1000) * 1000,
        qm_preis      = round(qm_preis),
        konfidenz     = round(konfidenz, 3),
        faktoren={
            "basispreis_qm":      basispreis,
            "alter_faktor":       round(alter_faktor, 3),
            "zustand_faktor":     zustand_f,
            "ausstattung_faktor": ausstattung_f,
            "typ_faktor":         typ_f,
            "energie_faktor":     energie_f,
            "ausstattungs_bonus": round(bonus, 3),
        }
    )
 
 
# ─────────────────────────────────────────────
# MODUL 2 – HUMAN-ANALOGUE
# ─────────────────────────────────────────────
 
def modul_human_analogue(inp: ObjektInput, basis: ModulErgebnis) -> HumanAnalogueErgebnis:
    flags  = []
    risiko = 0
    bes    = inp.besonderheiten or []
 
    def add(text, stufe, punkte):
        nonlocal risiko
        flags.append({"text": text, "stufe": stufe})
        risiko += punkte
 
    if "erbbau"               in bes: add("Erbbaurecht",                   "rot",  30)
    if "altlast"              in bes: add("Altlasten bekannt",              "rot",  35)
    if "datenschwach"         in bes: add("Schwache Datenlage",             "rot",  25)
    if "mikrolage"            in bes: add("Ungewöhnliche Mikrolage",        "gelb", 20)
    if "mieter"               in bes: add("Vermietet (Mietrecht beachten)", "gelb", 15)
    if "denkmal_auflagen"     in bes: add("Denkmalschutz-Auflagen",         "rot",  28)
    if "rechtliche_belastung" in bes: add("Rechtliche Belastung",           "rot",  32)
    if inp.objekttyp == "denkmal":    add("Denkmalschutzobjekt",            "rot",  25)
    if inp.objekttyp == "sonder":     add("Sonderobjekt",                   "rot",  40)
    if inp.baujahr < 1950:            add(f"Altbau ({inp.baujahr})",        "gelb", 10)
    if inp.baujahr < 1920:            add("Sehr alter Bestand – erhöhtes Risiko", "rot", 15)
 
    plausibel    = basis.konfidenz >= 0.60 and risiko < 35
    kohaerenz    = True
    marktgaengig = inp.objekttyp not in ["sonder"] and risiko < 60
 
    pruefschritte = [
        {"name": "Plausibilitätsprüfung",   "kuerzel": "P", "ok": plausibel,
         "detail": "Wertschätzung im realistischen Marktpreisband." if plausibel else "Wert außerhalb des erwarteten Marktpreisbandes."},
        {"name": "Kohärenzprüfung",         "kuerzel": "K", "ok": kohaerenz,
         "detail": "Merkmalskombination widerspruchsfrei."},
        {"name": "Marktgängigkeit",         "kuerzel": "M", "ok": marktgaengig,
         "detail": "Objekt ist standardisiert und handelbar." if marktgaengig else "Eingeschränkte Marktgängigkeit."},
        {"name": "Sondermerkmal-Erkennung", "kuerzel": "S", "ok": risiko < 20,
         "detail": f"Risiko-Score: {min(100, risiko)}/100"},
    ]
 
    return HumanAnalogueErgebnis(
        plausibel=plausibel, kohaerenz=kohaerenz, marktgaengig=marktgaengig,
        risiko_score=min(100, risiko), flags=flags, pruefschritte=pruefschritte,
    )
 
 
# ─────────────────────────────────────────────
# MODUL 3 – ERKLÄRUNGSMODUL (regelbasiert + Claude)
# ─────────────────────────────────────────────
 
def modul_xai_regelbasiert(inp: ObjektInput, basis: ModulErgebnis) -> list:
    f = basis.faktoren
    treiber = [
        {"name": f"Lage ({inp.lage})",          "effekt": round((f["basispreis_qm"] / 4500 - 1) * 100, 1), "gewicht": 35},
        {"name": f"Baujahr ({inp.baujahr})",     "effekt": round((f["alter_faktor"]       - 1) * 100, 1),   "gewicht": 20},
        {"name": f"Zustand ({inp.zustand})",     "effekt": round((f["zustand_faktor"]     - 1) * 100, 1),   "gewicht": 18},
        {"name": "Ausstattung",                  "effekt": round((f["ausstattung_faktor"] - 1) * 100, 1),   "gewicht": 12},
        {"name": f"Objekttyp ({inp.objekttyp})", "effekt": round((f["typ_faktor"]         - 1) * 100, 1),   "gewicht": 10},
        {"name": "Energieeffizienz",             "effekt": round((f["energie_faktor"]     - 1) * 100, 1),   "gewicht":  5},
    ]
    return sorted(treiber, key=lambda x: abs(x["effekt"]), reverse=True)
 
 
async def modul_xai(inp: ObjektInput, basis: ModulErgebnis, human: HumanAnalogueErgebnis) -> XAIErgebnis:
    treiber = modul_xai_regelbasiert(inp, basis)
    lage_text = {"1a": "sehr guter", "1b": "guter", "2": "mittlerer", "3": "einfacher"}.get(inp.lage, "")
 
    fallback = XAIErgebnis(
        treiber=treiber,
        begruendung=(
            f"Das Objekt ({inp.objekttyp.upper()}, {inp.flaeche} m², Baujahr {inp.baujahr}) befindet sich "
            f"in {lage_text} Lage im {inp.markt}städtischen Markt. "
            f"Der geschätzte Marktwert von {int(basis.marktwert):,} € entspricht einem Quadratmeterpreis "
            f"von {basis.qm_preis:,} €/m². Dominanter Werttreiber ist die Lageklasse ({inp.lage}). "
            + ("Hinweis: Besondere Risikomerkmale wurden erkannt: "
               + ", ".join([f["text"] for f in human.flags]) + "."
               if human.flags else "Keine wertrelevanten Sondermerkmale festgestellt.")
        ),
        methodik="Regelbasierte Faktorzerlegung (Hedonic Pricing)"
    )
 
    if not ANTHROPIC_API_KEY:
        return fallback
 
    prompt = f"""Du bist ein erfahrener Immobiliensachverständiger in Deutschland.
Erstelle eine präzise, sachverständige Begründung für folgende Marktwertschätzung.
 
Objektdaten:
- Typ: {inp.objekttyp}, {inp.flaeche} m², Baujahr {inp.baujahr}
- Lage: Klasse {inp.lage}, Markt: {inp.markt}städtisch
- Zustand: {inp.zustand}, Ausstattung: {inp.ausstattung}
- Energieklasse: {inp.energieklasse or 'nicht angegeben'}
- Besonderheiten: {', '.join(inp.besonderheiten) if inp.besonderheiten else 'keine'}
 
Bewertungsergebnis:
- Marktwert: {int(basis.marktwert):,} € (Spanne: {int(basis.marktwert_min):,} – {int(basis.marktwert_max):,} €)
- Preis/m²: {basis.qm_preis:,} €
- Konfidenz: {round(basis.konfidenz * 100)}%
- Risiko-Score: {human.risiko_score}/100
- Hauptwerttreiber: {', '.join([t['name'] + ' (' + str(t['effekt']) + '%)' for t in treiber[:3]])}
 
Schreibe eine sachliche, professionelle Begründung in 3–4 Sätzen (max. 120 Wörter).
Antworte nur mit der Begründung, ohne Überschrift oder Einleitung."""
 
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        return XAIErgebnis(
            treiber=treiber,
            begruendung=message.content[0].text.strip(),
            methodik="Hybride Analyse: Hedonic Pricing + Claude Sonnet (Erklärungsmodul)"
        )
    except Exception:
        return fallback
 
 
# ─────────────────────────────────────────────
# MODUL 4+5 – SONDERFALL + GOVERNANCE
# ─────────────────────────────────────────────
 
def modul_governance(inp: ObjektInput, basis: ModulErgebnis, human: HumanAnalogueErgebnis) -> GovernanceErgebnis:
    from datetime import datetime
    stufe   = "gruen"
    gruende = []
    bes     = inp.besonderheiten or []
 
    if human.risiko_score >= 50 or not human.plausibel:
        stufe = "rot"; gruende.append("Hoher Risiko-Score oder fehlende Plausibilität der Schätzung")
    elif human.risiko_score >= 20 or not human.marktgaengig or basis.konfidenz < 0.65:
        stufe = "gelb"; gruende.append("Erhöhte Unsicherheit oder eingeschränkte Marktgängigkeit")
 
    if "altlast"               in bes: stufe = "rot"; gruende.append("Altlasten bekannt – zwingende Sachverständigenprüfung (§§ 4–6 BBodSchG)")
    if "erbbau"                in bes: stufe = "rot"; gruende.append("Erbbaurecht – rechtliche Sondersituation (ErbbauRG)")
    if "rechtliche_belastung"  in bes: stufe = "rot"; gruende.append("Rechtliche Belastung – manuelle Prüfung erforderlich")
    if inp.objekttyp == "denkmal":     stufe = "rot"; gruende.append("Denkmalschutz – erhöhter Prüfbedarf (DSchG)")
 
    if not gruende:
        gruende.append("Alle Prüfschritte bestanden – Schätzung als belastbar eingestuft")
 
    texte = {
        "gruen": ("Automatische Freigabe",
                  "Der Schätzwert ist belastbar und kann als Orientierungswert verwendet werden. Keine manuelle Sachverständigenprüfung erforderlich."),
        "gelb":  ("Orientierungswert mit Vorbehalt",
                  "Die Schätzung ist eingeschränkt belastbar. Eine ergänzende Prüfung durch einen Sachverständigen wird empfohlen."),
        "rot":   ("Zwingende Sachverständigenprüfung",
                  "Das Objekt weist Besonderheiten auf, die eine vollständige manuelle Bewertung durch einen qualifizierten Sachverständigen (§ 194 BauGB) erfordern. Keine automatische Freigabe."),
    }
    titel, empfehlung = texte[stufe]
    return GovernanceErgebnis(
        stufe=stufe,
        farbe={"gruen": "#22c55e", "gelb": "#f59e0b", "rot": "#ef4444"}[stufe],
        titel=titel, empfehlung=empfehlung,
        gruende=list(dict.fromkeys(gruende)),
        protokoll={
            "zeitstempel":   datetime.now().isoformat(),
            "konfidenz_pct": round(basis.konfidenz * 100),
            "risiko_score":  human.risiko_score,
            "marktwert":     int(basis.marktwert),
            "stufe":         stufe.upper(),
            "modell":        "AVM-System v1.0 / Claude Sonnet",
        }
    )
 
 
# ─────────────────────────────────────────────
# CLAUDE – GESAMTANALYSE
# ─────────────────────────────────────────────
 
async def claude_gesamtanalyse(inp: ObjektInput, basis: ModulErgebnis,
                                human: HumanAnalogueErgebnis, gov: GovernanceErgebnis) -> str:
    if not ANTHROPIC_API_KEY:
        return ""
 
    prompt = f"""Du bist ein erfahrener Immobiliensachverständiger. Gib eine kurze Gesamteinschätzung (max. 80 Wörter):
 
Typ: {inp.objekttyp}, {inp.flaeche}m², Baujahr {inp.baujahr}, Lage {inp.lage}, Zustand {inp.zustand}
Marktwert: {int(basis.marktwert):,}€ | Konfidenz: {round(basis.konfidenz*100)}% | Risiko: {human.risiko_score}/100
Governance-Stufe: {gov.stufe.upper()} – {gov.titel}
Besonderheiten: {', '.join(inp.besonderheiten) if inp.besonderheiten else 'keine'}
 
Schreibe sachlich, präzise, ohne Überschrift."""
 
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text.strip()
    except Exception:
        return ""
 
 
# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────
 
@app.get("/")
def root():
    return {"status": "AVM-System läuft", "version": "1.0.0"}
 
@app.get("/health")
def health():
    return {"status": "ok", "claude_api": bool(ANTHROPIC_API_KEY)}
 
@app.post("/api/bewertung", response_model=BewertungsAntwort)
async def bewertung(inp: ObjektInput):
    try:
        basis      = modul_basis(inp)
        human      = modul_human_analogue(inp, basis)
        xai        = await modul_xai(inp, basis, human)
        gov        = modul_governance(inp, basis, human)
        ki_analyse = await claude_gesamtanalyse(inp, basis, human, gov)
        return BewertungsAntwort(basis=basis, human_analogue=human, xai=xai,
                                 governance=gov, ki_analyse=ki_analyse)
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "fehler": "Interner Serverfehler",
            "meldung": str(e),
            "code": "INTERNAL_ERROR"
        })
 
 
@app.get("/api/marktdaten")
def marktdaten():
    return {
        "lagepreise": BASISPREISE,
        "faktoren": {
            "zustand":       ZUSTAND_FAKTOREN,
            "ausstattung":   AUSSTATTUNG_FAKTOREN,
            "objekttyp":     TYP_FAKTOREN,
            "energieklasse": ENERGIE_FAKTOREN,
        }
    }
 