# AVM-System – KI-Immobilienbewertung
## Hybrides, erklärbares Bewertungssystem für Wohnimmobilien (Deutschland)

Bachelorarbeit Friedrich Kraus – Prototyp v1.0

---

## Projektstruktur

```
avm-system/
├── backend/
│   ├── main.py              # FastAPI-Backend mit allen 5 Modulen
│   └── requirements.txt     # Python-Abhängigkeiten
├── frontend/
│   └── index.html           # Einzelne HTML-Datei (keine Build-Tools nötig)
└── README.md
```

---

## Schnellstart

### 1. Backend starten

```bash
cd backend

# Abhängigkeiten installieren
pip install -r requirements.txt

# API-Key setzen (für Claude-Erklärungsmodul)
export ANTHROPIC_API_KEY="sk-ant-..."   # Linux/Mac
set ANTHROPIC_API_KEY=sk-ant-...        # Windows CMD

# Server starten
uvicorn main:app --reload --port 8000
```

Backend läuft dann unter: http://localhost:8000
API-Dokumentation: http://localhost:8000/docs

### 2. Frontend öffnen

Einfach `frontend/index.html` im Browser öffnen.
Keine Build-Tools, kein npm, keine Installation nötig.

Sicherstellen, dass die Backend-URL im Formular auf `http://localhost:8000` steht.

---

## Ohne API-Key

Das System funktioniert vollständig **ohne Anthropic API-Key**:
- Module 1, 2, 4, 5 laufen rein regelbasiert
- Modul 3 (Erklärungsmodul) generiert eine regelbasierte Textbegründung
- Mit API-Key werden Begründung und Gesamtanalyse durch Claude Sonnet erstellt

---

## Architektur: 5-Modul-System

| Modul | Funktion | Methodik |
|-------|----------|----------|
| M1 – Basismodul | KI-Marktwertschätzung | Hedonic Pricing (regelbasiert) |
| M2 – Human-Analogue | Sachverständige Heuristiken | Regelbasiert (Plausibilität, Kohärenz, Marktgängigkeit) |
| M3 – XAI | Werttreiber & Begründung | SHAP-analog + Claude Sonnet |
| M4 – Sonderfall | Risiko-Erkennung | Regelbasierte Flags |
| M5 – Governance | Eskalationslogik | Grün / Gelb / Rot + Protokoll |

---

## API-Endpunkte

```
POST /api/bewertung     – Vollständige Bewertung (alle 5 Module)
GET  /api/marktdaten    – Einsehbare Preistabellen und Faktoren
GET  /health            – Systemstatus und Claude-API-Verfügbarkeit
GET  /docs              – Interaktive API-Dokumentation (Swagger)
```

### Beispiel-Request

```json
POST /api/bewertung
{
  "flaeche": 85,
  "baujahr": 1990,
  "lage": "1b",
  "zustand": "gut",
  "ausstattung": "mittel",
  "objekttyp": "etw",
  "markt": "gross",
  "energieklasse": "C",
  "balkon": true,
  "besonderheiten": []
}
```

---

## Erweiterungsideen (für die Bachelorarbeit)

- Echte ML-Modelle einbinden (Scikit-learn Gradient Boosting auf Gutachterausschuss-Daten)
- Datenbank-Anbindung für Vergleichsobjekte
- Exportfunktion (PDF-Gutachten)
- Experteninterface für manuelle Sachverständigen-Prüfung
- ImmoWertV-konforme Ausgabestruktur
