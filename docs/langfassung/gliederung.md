# Langfassung — Gliederung, Seitenbudget und Arbeitsplan

Stand: 2026-09-27. Arbeitsdokument, nicht die Arbeit selbst. Zweck: Die schriftliche
Arbeit entsteht **schrittweise ab Oktober** statt als Sprint im Januar — genau der
Schritt, an dem es beim Jugendwettbewerb Informatik zweimal gescheitert ist.

---

## 0. Formale Vorgaben (Leitfaden, Stand Juli 2025)

Quelle: *Leitfaden zum Verfassen der schriftlichen Arbeit im Wettbewerb Jugend forscht*,
Stiftung Jugend forscht e. V., Stand Juli 2025.

- **Höchstens 15 DIN-A4-Seiten.** Nicht mitgezählt: Deckblatt, Projektüberblick,
  Inhaltsverzeichnis, Quellenangaben, Unterstützungsleistungen.
  **Mitgezählt: Fußnoten, Tabellen, Grafiken, Bilder.** Jede Abbildung kostet Platz.
- Schrift ≥ 10 pt, Arial oder Times New Roman; Zeilenabstand ≥ 1,5;
  Ränder ≥ 2,5 cm links/rechts/oben, ≥ 2 cm unten.
- **Kein Anhang** in der Arbeit. Ausführliche Tabellen, Code, Zusatzgrafiken → am Stand.
- Jede Tabelle/Abbildung mit Titel, fortlaufend nummeriert.
- Literaturverzeichnis alphabetisch; Internetquellen mit URL **und Abrufdatum**.
- Zeitbedarf laut Leitfaden: Auswertung + Schreiben ≥ **Hälfte** des Gesamtaufwands;
  für das Schreiben **mindestens vier Wochen** einplanen.

**Vorgeschriebene Reihenfolge:**
Deckblatt · Projektüberblick · Inhaltsverzeichnis · Fachliche Kurzfassung ·
Motivation und Fragestellung · Hintergrund und theoretische Grundlagen ·
Vorgehensweise, Materialien und Methoden · Ergebnisse · Ergebnisdiskussion ·
Fazit und Ausblick · Quellen- und Literaturverzeichnis · Unterstützungsleistungen

---

## 1. Kernaussage (vor dem Schreiben festlegen, danach nicht mehr ändern)

> Aus offenen Daten lassen sich Verspätungen im Berliner Nahverkehr **über Zufall**
> vorhersagen — aber was sie bestimmt, ist nicht Wetter oder Kalender, sondern **welche
> Linie** fährt. Zum Fahrplanzeitpunkt erreicht kein gelerntes Modell mehr als eine
> einfache Nachschlagetabelle. Erst die **eigene Echtzeitschätzung einer Fahrt** hilft
> deutlich — kurz vor Abfahrt sehr, nach einer Stunde gar nicht mehr. Der Zustand der
> übrigen Linien am Halt bringt dagegen nichts Messbares.
>
> (Korrigiert 05.10.: Die frühere Fassung behauptete, der Netzzustand treibe die
> Verspätung mit. Mit besseren Labels und 6 Folds hielt das nicht.)

Jedes Kapitel muss diese Aussage stützen, einschränken oder begründen. Was das nicht
tut, gehört an den Stand, nicht in die 15 Seiten.

---

## 2. Kapitel, Seitenbudget, Inhalt, Belege

Seitenbudget gesamt: **15,0 Seiten** inklusive aller Abbildungen und Tabellen.

### Projektüberblick (zählt nicht mit, max. 1 Seite, 6–8 Sätze)
Allgemeinverständlich, ohne Literaturhinweise, je Teil 1–2 Sätze. **Zuletzt schreiben.**

### Fachliche Kurzfassung — 0,5 S.
Wissenschaftliche Kernbefunde mit Zahlen. **Zuletzt schreiben**, muss dem Endstand
entsprechen.

### Motivation und Fragestellung — 1,0 S.
- Ziel, wie das Thema gefunden wurde, **überprüfte Hypothesen**:
  H1 Verspätungen sind aus offenen Daten vorhersagbar.
  H2 Wetter und Kalender sind wesentliche Treiber.
  H3 Echtzeitdaten verbessern die Vorhersage, abhängig vom Horizont.
- Wichtig: Die Hypothesen **so stehen lassen, wie sie am Anfang waren** — dass H2
  widerlegt wird, ist ein Ergebnis, kein Makel.
- Belege: Entscheidungslog 2026-08-10 „Zielgröße (A)/(B)".

### Hintergrund und theoretische Grundlagen — 1,5 S.
- Was ist Verspätung im Echtzeitfeed (Schätzung, keine Messung; Minutenauflösung).
- GTFS / GTFS-Realtime in drei Sätzen.
- **Stand der Forschung:** Bahn-Vorhersage (Döllmann, Jugend forscht 2023; BWKI 2020) —
  Fernverkehr, Anschlusssicherheit, Produktcharakter. Abgrenzung: hier Stadtverkehr,
  U-Bahn/Tram/Bus, Frage nach **Treibern**, nicht Produkt.
- Deutsche-Bahn-Datensatz (piebro, CC BY 4.0) als Vergleich; Berliner S-Bahn ist
  dort schon enthalten → Neuheit liegt bei U-Bahn/Tram/Bus.
- Bewertungsmaße: ROC-AUC vs. **PR-AUC bei seltenen Ereignissen**; warum MAE hier täuscht.
- Belege: 2026-08-10 „Novitätsanspruch korrigiert", 2026-08-10 „Baseline Median/Mittel",
  2026-08-17 „Zielgröße quantisiert und nulllastig".

### Vorgehensweise, Materialien und Methoden — 4,0 S.
Laut Leitfaden hier auch **aufgetretene Schwierigkeiten** und **klare Darstellung des
Eigenanteils bzw. der Unterstützung durch Dritte** (siehe Abschnitt 5!).
1. **Datenerhebung** (1,2 S.) — vier Umsteigeknoten, 15-min-Takt, 65-min-Vorschau,
   GitHub Actions, Speicherformat, zwei Quellen (transport.rest + VBB GTFS-RT),
   Geschwisterhaltestellen. *Abb. 1: Datenfluss-Schema.*
2. **Datenaufbereitung** (0,8 S.) — Zielgröße = letzte Schätzung vor Abfahrt,
   `lead_time_s`, quellenübergreifende Deduplikation (47,5 % Duplikate!),
   Wetter-Join (UTC), Kalender (Ortszeit).
3. **Modelle** (0,6 S.) — Nachschlagetabelle (Linie × Stunde) als ehrliche Baseline,
   Gradient Boosting, Ausschluss aller leckenden Merkmale.
4. **Validierung** (0,8 S.) — zeitlicher Split, Rolling-Origin-Folds,
   Permutationstest, tagesgeclustertes Bootstrap; warum zufälliger Split täuscht (+0,098).
5. **Schwierigkeiten** (0,6 S.) — Quellenausfall 20.–25.08., Scheduler-Drosselung,
   Push-Verlust, die drei Überwachungsfehler. Kurz, sachlich, mit Konsequenz.
- Belege: fast alle Einträge 2026-08-05 bis 2026-09-21.

### Ergebnisse — 5,0 S.
Untergliedert, jede Aussage an Abbildung/Tabelle gebunden.
1. **Datensatz** (0,5 S.) — Umfang, Zeitraum, nutzbare Abdeckung beider Quellen. *Tab. 1.*
2. **Verteilung der Verspätung** (0,7 S.) — 74 % exakt 0, Minutenquantisierung,
   2,3 % der Fahrten tragen 67,6 % der Verspätung. *Abb. 2.*
3. **Signal ist echt** (0,4 S.) — Permutationstest p = 0,005.
4. **Fahrplan-Vorhersage ≈ Nachschlagetabelle** (0,8 S.) — der zentrale Negativbefund;
   Schwere vs. Vorhersagbarkeit. *Abb. 3.*
5. **Wert von Echtzeitinformation über den Horizont** (1,2 S.) — **Hauptabbildung**:
   PR-AUC über Vorlauf. *Abb. 4.* ✓ Neu validiert 05.10. (jüngste Labels, 6 Folds,
   245.134 Testabfahrten): eigene Schätzung 0,256 → 0,459 (10 min, 6/6 Folds), fällt bis
   60 min auf 0,262. **Netzzustand: kein messbarer Gewinn** (+0,001 [−0,011; +0,016]) —
   als Negativbefund berichten, mit der Geschichte, wie er schrumpfte (siehe Log 05.10.).
   Offen: Verfall der eigenen Schätzung ist teils Verfügbarkeit (60 min: nur 7,6 %).
6. **Ausfälle** (0,4 S.) — **nicht vorhersagbar**: über 4 Wochen-Folds ROC 0,572 für Modell
   und Tabelle gleich, nahe Zufall. Der frühere Einzelsplit-Befund (0,752) hielt nicht. *Tab. 2.*
7. **Widerlegt: Linienstruktur, Wetter** (1,0 S.) — Struktur erklärt innerhalb der
   Betriebsform nichts (p = 0,60); Wetter real aber vernachlässigbar, kein
   Saison-Artefakt. ⧗ *Wetter-Endwert erst nach Winterdaten (Mitte Januar).*

### Ergebnisdiskussion — 2,0 S.
- Deutung: Verspätung ist **strukturell und zustandsabhängig**, nicht umweltbedingt.
- Einordnung zur Literatur (Bahn-Vorhersage; Fernverkehr ≠ Stadtverkehr).
- **Fehlerquellen** (1,2 S.): Label ist Schätzung (`lead_time_s`), Minutenauflösung,
  eine Wetterstation für die ganze Stadt, nur vier Knoten, Nicht-Stationarität und
  Instrumentenwechsel, Überwachungsfehler als eigene Fehlerklasse.
- Belege: 2026-08-10 „lead_time_s", „DWD-Station 00433", 2026-09-03 „Belastbarkeitsprüfung",
  2026-09-12 „healthcheck".

### Fazit und Ausblick — 1,0 S.
- Antwort auf die Forschungsfrage in drei Sätzen; H1 bestätigt, H2 widerlegt, H3 bestätigt
  mit Horizontabhängigkeit.
- Ausblick: Streckenabschnitts-Fahrzeiten (erklären die 15-fache Streuung unter Bussen),
  mehr Knoten, Winterdaten.

### Quellen- und Literaturverzeichnis (zählt nicht mit)
Alle Datenquellen mit URL + Abrufdatum: v6.bvg.transport.rest, production.gtfsrt.vbb.de,
vbb.de/vbbgtfs, DWD Open Data (Station 00433), OpenHolidays API, Hugging Face
piebro/deutsche-bahn-data, Jugend-forscht-Projektdatenbank (Bahn-Vorhersage);
Bibliotheken (pandas, scikit-learn, httpx).

### Unterstützungsleistungen (zählt nicht mit) — siehe Abschnitt 5

---

## 3. Abbildungen und Tabellen (zählen gegen die 15 Seiten)

| Nr. | Inhalt | erzeugt durch | Status |
|---|---|---|---|
| Abb. 1 | Datenfluss: Quellen → Logger → GitHub → Datensatz | von Hand zeichnen | offen |
| Abb. 2 | Verteilung der Verspätung + Anteil an Gesamtverspätung | neu, aus `dataset.parquet` | offen |
| Abb. 3 | ROC/PR über Schwellenwert, Modell vs. Tabelle | `signal_check.py --sweep` | Daten da |
| Abb. 4 | **PR-AUC über Vorhersagehorizont** (Hauptabbildung) | `experiments/horizon_sweep.py` + `plot_horizon.py` | **fertig** (27.09.) |
| Tab. 1 | Datensatz und Abdeckung je Quelle | `collection_dashboard.py` | Daten da |
| Tab. 2 | Ausfall-Vorhersage vs. Tabelle (Negativbefund) | `analysis/model_quality.py` | Daten da (05.10.) |

Mehr als ~5 Abbildungen passen nicht ins Budget. Alles Weitere → Stand.

---

## 4. Zeitplan

| Zeitraum | Arbeitspaket |
|---|---|
| bis Mitte Okt. | Horizont-Validierung fertig, Kernaussage eingefroren, Abb. 3 + 4 als Entwurf |
| Oktober | **Methoden** schreiben (ändert sich nicht mehr) + Hintergrund |
| November | stabile Ergebnis-Abschnitte (1–6) schreiben; **Anmeldung vor dem 30.11.** |
| Dezember | Diskussion, Fehlerquellen, Fazit als Entwurf; Wetter-Abschnitt als Platzhalter |
| ~10. Januar | **Datenstand einfrieren**, Pipeline neu rechnen, alle Zahlen aktualisieren |
| Mitte Januar | Wetter/Winter einsetzen, Kurzfassung + Projektüberblick zuletzt |
| Abgabe | genaues Datum in der Jufo-Wettbewerbsverwaltung prüfen — **konservativ mit Mitte Januar planen** |

---

## 5. Unterstützungsleistungen und Eigenanteil — muss ehrlich gelöst werden

Der **KI-Leitbild** der Stiftung (Stand Nov. 2024) ist eindeutig:
- KI-Nutzung ist **erlaubt und ausdrücklich erwünscht**, wenn sie **klar gekennzeichnet**
  ist (als Unterstützungsleistung, KI-Programm benannt, Verwendung beschrieben).
- Beim Hochladen in Jufo WV 2.0 wird KI-Nutzung als Unterstützungsleistung abgefragt.
- **Das Jurygespräch gewinnt gegenüber der schriftlichen Arbeit an Bedeutung**; die
  Feststellung des Eigenanteils ist eine seiner Hauptaufgaben.
- KI-generierte Texte müssen geprüft und alle Aussagen belegt werden.

Der Leitfaden verlangt zusätzlich im Methodenkapitel: Von Dritten unterstützte oder
übernommene Arbeitsschritte **detailliert** nennen, den **selbstständig erbrachten
Anteil klar herausstellen**.

### Sachstand, den ich (Claude) belegen kann
In der Arbeitsphase ab 10.08.2026 wurde mit **Claude (Anthropic), genutzt über Claude
Code**, u. a. Folgendes erstellt oder wesentlich überarbeitet:
- Code: Speicherumstellung, GTFS-RT-Logger, Workflow-Logik (Schleife, Push-Wiederholung,
  Exit-Status), Dashboard, Healthcheck, Datensatz-Pipeline inkl. Deduplikation,
  Analyse- und Experimentskripte.
- Statistische Auswertung: Permutationstests, Bootstrap, Rolling-Origin-Validierung,
  Ablationen; mehrere Diagnosen (Quellenausfall, Geschwisterhaltestellen).
- **Ein Großteil des Textes in `docs/entscheidungen.md`.**

### Was nur du selbst ausfüllen kannst
- Welche Ideen und Entscheidungen von dir stammen (Forschungsfrage, Themenfindung,
  Wahl der Knoten, ursprüngliches Logger-Design, Hybridstrategie, …)?
- Was vor dem 10.08. entstand und mit welcher Hilfe?
- Weitere Unterstützung (Betreuungslehrkraft, das Treffen mit Frau Eckhardt, …).

**Achtung:** Die README-Passage „Mine: the logging design, schema, feature engineering,
models, and analysis" stimmt in dieser Form nicht mehr und sollte angepasst werden.

### Empfehlung
1. KI-Nutzung **vollständig und konkret** offenlegen — das ist regelkonform und wird
   nicht negativ bewertet, eine unvollständige Angabe dagegen schon.
2. Die Langfassung **selbst formulieren** und das Entscheidungslog nur als Faktenquelle
   nutzen. Das ist erlaubt auch mit KI — aber selbst schreiben ist die beste
   Vorbereitung auf das Jurygespräch, das den Eigenanteil prüft.
3. Jede Methode so verstehen, dass du sie ohne Unterlagen erklären kannst (Abschnitt 6).

---

## 6. Jurygespräch — das musst du frei erklären können

- Warum ein **zeitlicher** und kein zufälliger Split (und was +0,098 ROC bedeutet)?
- Was misst ein **Permutationstest**, und warum heißt p = 0,005 „Signal ist echt"?
- Warum **tagesgeclustertes** Bootstrap statt zeilenweise?
- ROC-AUC vs. **PR-AUC**, und warum PR-AUC bei 2 % Positivrate aussagekräftiger ist.
- Was ist **Leakage**? Warum ist `delay_drift_s` verboten, `first_delay_s` nur im Nowcast erlaubt?
- Warum schlägt eine **Nachschlagetabelle** fast jedes Modell — und warum ist das ein Ergebnis?
- Wie funktioniert die **Deduplikation** zwischen zwei Quellen, und warum konservativ?
- Welche **Fehlerquellen** hat das Label (`lead_time_s`, Minutenauflösung)?
- Was ging bei der Datenerhebung schief, und was hast du daraus geändert?

---

## 7. Anmeldung (bis 30.11.2026, nur Titel + Kurzbeschreibung)

**Titelvorschläge** (Frage statt Thema — der Leitfaden verlangt eine präzise Fragestellung):
- *Wie viel sind Echtzeitdaten wert? Vorhersagbarkeit von Verspätungen im Berliner Nahverkehr*
- *Warum kommt die Tram zu spät? Was offene Daten über Verspätungen in Berlin verraten*

**Kurzbeschreibung (Entwurf, selbst überarbeiten):**
Seit August 2026 erfasse ich alle 15 Minuten die Echtzeit-Abfahrten an vier Berliner
Umsteigeknoten und habe damit einen Datensatz zu U-Bahn, Tram und Bus aufgebaut, den es
öffentlich so nicht gibt. Mit Methoden des maschinellen Lernens untersuche ich, wie gut
sich Verspätungen aus offenen Fahrplan-, Wetter- und Echtzeitdaten vorhersagen lassen und
welche Faktoren sie tatsächlich bestimmen. Überraschend: Wetter und Kalender spielen kaum
eine Rolle, entscheidend ist die Linie selbst — und erst Echtzeitdaten kurz vor der
Abfahrt verbessern die Vorhersage deutlich.
