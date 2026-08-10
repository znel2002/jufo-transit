# Entscheidungslog

Fortlaufendes Protokoll aller nicht-offensichtlichen Entscheidungen: **Was / Warum /
Verworfene Alternative / Beleg.**

Zweck: Im Januar 2027 entstehen daraus die Kapitel *Methodik* und *Fehlerquellen* der
Langfassung. Diese Kapitel lassen sich nachträglich nicht rekonstruieren — warum
Station 00433 und nicht 00403 gewählt wurde, weiß man im Januar nicht mehr. Deshalb
wird hier ab dem ersten Tag mitgeschrieben, nicht erst ab Dezember.

Neue Einträge **unten** anhängen. Jeder Eintrag bekommt ein Datum.

---

## 2026-08-05 — Vier Messstationen statt vieler

**Was:** Alexanderplatz (900100003), Zoologischer Garten (900023201), Warschauer Str.
(900120004), Hauptbahnhof (900003201).

**Warum:** Alle vier sind Umsteigeknoten mit gemischtem Produktangebot, d. h. schon
vier Haltestellen decken U-Bahn, S-Bahn, Tram, Bus, Express und Regional ab (~231–242
Abfahrten pro Poll). Tiefe einer sauberen Zeitreihe schlägt Breite: eine lückenlose
Reihe über sechs Monate ist auswertbar, 50 lückenhafte Haltestellen nicht.

**Verworfen:** Flächendeckendes Loggen aller Berliner Haltestellen — hätte das
API-Ratelimit (100 req/min) und die GitHub-Actions-Laufzeit gesprengt.

---

## 2026-08-05 — Append-only-Speicherung statt Deduplikation beim Schreiben

**Was:** Jeder Poll schreibt eine Zeile pro gesehener Abfahrt; dieselbe Fahrt erscheint
mehrfach. Dedupliziert wird erst zur Analysezeit auf `(trip_id, stop_id, planned_when)`.

**Warum:** So bleibt erhalten, *wie sich eine Verspätungsprognose entwickelt*, während
sich das Fahrzeug nähert. Das ist eine eigene Teilfrage und zusätzlich das Rohmaterial
für das Nowcast-Modell (siehe 2026-08-10, Zielgröße).

**Verworfen:** Upsert beim Schreiben — hätte Speicher gespart, aber die Prognose-
Entwicklung unwiederbringlich vernichtet.

---

## 2026-08-10 — Wetter wird nachgeladen, Verkehr wird live geloggt

**Was:** Verkehrsdaten laufen kontinuierlich mit (GitHub Actions, alle 15 min).
Wetterdaten werden **nicht** geloggt, sondern bei Bedarf in einem Durchgang geholt.

**Warum:** Asymmetrie der Nachladbarkeit. Der DWD veröffentlicht Stationsmessungen
rückwirkend (verifiziert 2026-08-10: ~16 h Verzug, `/recent/`-Archiv hält ~13.200
Stundenwerte ≈ 550 Tage). Das BVG-Realtime-Feed dagegen ist unwiederbringlich weg,
wenn es nicht im Moment des Auftretens erfasst wird. Also: das Unwiederbringliche
zuerst sichern, das Nachladbare später holen.

**Verworfen:** Ein zweiter Dauer-Logger für Wetter — doppelte Ausfallfläche ohne
jeden Gewinn.

**Offen:** `fetch_weather.py` liest nur `/recent/` (~550 Tage, also zurück bis ca.
Februar 2025). Für Bahndaten ab 2024-07 müsste zusätzlich das `/historical/`-Archiv
angebunden werden.

---

## 2026-08-10 — DWD-Station 00433 Berlin-Tempelhof

**Was:** Stundenwerte von Station 00433 (Temperatur, Luftfeuchte, Niederschlagsmenge,
**Niederschlagsform**, Windgeschwindigkeit und -richtung).

**Warum:** Zentrale Lage und längste Reihe aller Berliner Stationen (seit 1951).
Niederschlagsform ist bewusst dabei: 6 = Regen, 7 = Schnee, 8 = beides — Schnee ist
für Verspätungen der interessante Fall, und die reine Niederschlags*menge* würde ihn
verstecken.

**Verworfen:** 00403 Dahlem, 00400 Buch, 00420 Marzahn, 00427 BER — alle brauchbar,
aber kürzer bzw. randlagig.

**Fehlerquelle, die ins Kapitel gehört:** Eine einzige Station repräsentiert das
gesamte Stadtgebiet. Berliner Sommergewitter sind kleinräumig — ein Gewitter über
Spandau taucht in Tempelhof womöglich gar nicht auf. Das dämpft jeden gemessenen
Wettereffekt systematisch nach unten (Messfehler in der erklärenden Variable →
Regressionsverzerrung Richtung null).

⚠️ **Falle:** Berlin-Alexanderplatz (00399) und Berlin-Tegel (00430) sind **tote
Stationen** (Reihen enden 2011 bzw. 2021). Plausible Namen, keine aktuellen Daten.

---

## 2026-08-10 — Hosting auf GitHub Actions statt VPS

**Was:** Ein Runner pollt alle 15 min und committet die Daten zurück ins Repo.

**Warum:** Kostenlos und ohne Kreditkarte (öffentliches Repo = unbegrenzte
Actions-Minuten). Die Daten-Commits gelten zugleich als Repo-Aktivität.

**Verworfen:** Oracle Always-Free-VM (verlangt Karte zur Identitätsprüfung), Raspberry
Pi (kein Dauerbetrieb sichergestellt).

**Bewusst akzeptierter Nachteil:** GitHub garantiert Cron-Läufe nicht. Kürzestes
Intervall 5 min, Ausführung „best effort" — Läufe können 5–20 min zu spät kommen oder
unter Last ausfallen. Lücken werden in `observed_at` sichtbar und **ausgewertet, nicht
versteckt**. Zusätzlich: Geplante Workflows werden nach 60 Tagen Repo-Inaktivität
automatisch deaktiviert; ob die Bot-Commits das zuverlässig verhindern, ist nicht
dokumentiert → monatliche manuelle Kontrolle statt Vertrauen auf die Annahme.

---

## 2026-08-10 — Eine gzip-Datei pro Poll statt Anhängen an eine Tagesdatei

**Was:** `data/observations/<Tag>/<Zeitstempel>Z.ndjson.gz`, einmal geschrieben, nie
verändert. Vorher: Anhängen an eine Datei pro Tag.

**Warum:** Zwei Gründe, beide gerechnet, nicht geschätzt.
1. *Größe.* Git speichert bei jedem Commit einen vollständigen neuen Blob der
   geänderten Datei. Bei 96 Commits/Tag an einer auf 5,5 MB wachsenden Tagesdatei
   wären das ~1 GB Arbeitsverzeichnis bis Februar. Gemessene Kompression: 82 kB → 4,9 kB
   (Faktor 16,7), also ~480 kB/Tag bzw. ~86 MB bis Februar.
2. *Konflikte.* Eindeutige Dateinamen machen Schreibkonflikte zwischen gleichzeitigen
   Läufen strukturell unmöglich. Damit entfiel der `git pull --rebase`-Schritt im
   Workflow — genau der Schritt, der gegen den flachen Klon von `actions/checkout`
   fehlschlagen und einen Poll stillschweigend verlieren konnte.

**Zeitpunkt:** Umstellung am Tag 1 bei einem einzigen geloggten Poll. Später hätte sie
ein Umschreiben der Git-Historie erfordert.

**Verworfen:** Unkomprimiertes NDJSON pro Poll (im GitHub-UI lesbar, löst aber das
Größenproblem nicht); Beibehalten der Tagesdatei (verlässt sich darauf, dass Gits
Delta-Kompression es schon richten wird).

---

## 2026-08-10 — Zielgröße: Fahrplanzeitpunkt-Vorhersage (A) als Hauptergebnis, Nowcast (B) als Nebenexperiment

**Was:** Zwei getrennt ausgewertete Aufgaben.
- **(A) Vorhersage zum Fahrplanzeitpunkt** — aus Linie, Haltestelle, Stunde, Wochentag,
  Kalender und Wetter. **Keinerlei Echtzeitinformation über genau die Fahrt, die
  vorhergesagt wird.**
- **(B) Nowcast** — (A) zuzüglich der ~15 min vor Abfahrt gemeldeten Verspätung.

**Warum:** Ohne diese Trennung ist die zweite Hälfte der Forschungsfrage („welche
Faktoren treiben sie tatsächlich?") nicht beantwortbar. Darf das Modell die
Echtzeit-Verspätung derselben Fahrt sehen, gibt es sie im Wesentlichen nur wieder;
dieses eine Merkmal dominiert dann jede Wichtigkeitsanalyse und alle übrigen Faktoren
verschwinden dahinter. Das ist Leakage im Hinblick auf die Fragestellung, auch wenn es
formal keines im Hinblick auf die Zeitachse ist.

**Nebengewinn:** Der *Abstand* zwischen (A) und (B) ist selbst ein Ergebnis — er
beziffert, wieviel Echtzeitdaten überhaupt wert sind.

---

## 2026-08-10 — `delay_in_min` im Bahn-Datensatz ist die **Abfahrts**verspätung (empirisch geklärt)

**Was:** `piebro/deutsche-bahn-data` dokumentiert die Spalte nur als „Delay in
minutes". Jede Zeile ist ein Halt und enthält Ankunfts- *und* Abfahrtszeiten — die
Angabe ist also mehrdeutig.

**Belegt statt angenommen:** Beide Kandidaten rekonstruiert und verglichen
(2025-01, 1,5 Mio. Zeilen):

| Kandidat | n | exakte Übereinstimmung | Korrelation |
|---|---|---|---|
| `departure_change_time − departure_planned_time` | 1.574.895 | **100,0 %** | **1,000** |
| `arrival_change_time − arrival_planned_time` | 1.573.442 | 61,6 % | 0,965 |

→ Es ist eindeutig die Abfahrtsverspätung.

**Zweiter Fund dabei:** `*_change_time` ist **null, wenn nichts abweicht** — also
genau bei den pünktlichen Zügen. Eine Differenzbildung ohne vorheriges Auffüllen
hätte jeden pünktlichen Zug zu `NaN` gemacht und aus der Auswertung geworfen. Das
hätte die gemessene mittlere Verspätung systematisch nach oben verzerrt. Behoben in
`_elapsed_min()`.

**Außerdem:** Das README des Datensatzes ist veraltet — es beschreibt eine Spalte
`train_name`, die im Parquet nicht existiert (dort: `train_number`, `line_number`).
Deshalb wird das Schema zur Laufzeit ausgegeben und geprüft, statt der Dokumentation
zu vertrauen.

---

## 2026-08-10 — Baseline: Gruppen-**Median** für MAE, Gruppen-**Mittelwert** für RMSE

**Was:** Die ursprüngliche Baseline („mittlere Verspätung je Zugtyp × Stunde ×
Wochentag") war **schlechter als die triviale Vorhersage 0**. Kein Bug — eine falsch
spezifizierte Baseline.

**Warum:** Der MAE wird vom **Median** minimiert, der RMSE vom **Mittelwert**. Die
Verspätungsverteilung ist stark rechtsschief (Mittelwert 3,20 min, Median 1,00 min,
14,0 % über 5 min), also liegen Gruppenmittelwerte systematisch zu hoch, wenn in MAE
gemessen wird.

**Messwerte (2025-01, deutschlandweit, Testfenster 22.–31.01.2025, 451.441 Zeilen):**

| Prädiktor | MAE (min) | RMSE (min) |
|---|---|---|
| Vorhersage 0 (naiv) | 3,126 | 8,201 |
| globaler Median | 2,897 | 7,879 |
| globaler Mittelwert | 3,720 | 7,597 |
| **Gruppen-Median** (Zugtyp × Stunde × Wochentag) | **2,845** | 7,627 |
| **Gruppen-Mittelwert** | 3,483 | **7,376** |

**Zu schlagende Werte (deutschlandweit, 2025-01): MAE 2,845 min, RMSE 7,376 min.**

**Nur Berlin (11 Stationen, 107.083 Testzeilen):** MAE 1,540 / RMSE 4,610. Berliner
Züge sind deutlich pünktlicher (Mittelwert 1,68 min statt 3,20 min; 5,6 % statt 14,0 %
über 5 min).

**Konsequenz für alles Weitere:** Immer MAE **und** RMSE berichten. Ein einzelner Wert
verbirgt, dass die beiden Maße verschiedene Prädiktoren bevorzugen — und ein Modell,
das nur gegen die jeweils schwächere Baseline antritt, sieht besser aus als es ist.

**Verworfen:** Nur den MAE berichten (verstünde das schiefe Verteilungsproblem nicht);
Verspätungen vorher log-transformieren (negative Verspätungen — verfrühte Abfahrten —
existieren real, z. B. Zugtyp RSM mit −3,46 min Mittelwert).

---

## 2026-08-10 — Novitätsanspruch korrigiert: Berliner S-Bahn ist **nicht** neu, U-Bahn/Tram/Bus schon

**Was:** Das README behauptete, es gebe „no public historical BVG delay dataset". Das
ist so **falsch** und wurde korrigiert.

**Befund:** `piebro/deutsche-bahn-data` enthält die Berliner S-Bahn — allein im Januar
2025 **296.163 Zeilen** vom Zugtyp `S` an 11 Berliner Bahnhöfen (Ostkreuz,
Friedrichstraße, Hauptbahnhof, Südkreuz, Gesundbrunnen, Ostbahnhof, Lichtenberg,
Zoologischer Garten, Potsdamer Platz, Wannsee, Spandau), rückwirkend bis 2024-07. Die
S-Bahn Berlin GmbH ist eine DB-Tochter und erscheint deshalb in der DB-Fahrplan-API.

**Was weiterhin neu ist:** U-Bahn, Tram und Bus — also der von der BVG selbst
betriebene Verkehr — kommen in keinem öffentlichen historischen Datensatz vor. Im
eigenen Log sind das rund zwei Drittel jedes Polls (Bus 55, U-Bahn 46, Tram 46 von
231 Abfahrten). **Dort liegt der eigene Beitrag.**

**Warum das ein Gewinn und kein Verlust ist:** Die eigenen Logs erfassen die S-Bahn an
Zoologischer Garten und Hauptbahnhof — beide Stationen sind auch im Bahndatensatz
enthalten. Damit lässt sich die selbst erhobene Messung gegen eine **unabhängige
Quelle validieren**. Eine externe Validierung des eigenen Datensatzes ist methodisch
deutlich mehr wert als der ursprüngliche Alleinstellungsanspruch.

**Konsequenz:** Der Novitätsanspruch wird auf U-Bahn/Tram/Bus eingegrenzt, und die
S-Bahn-Überlappung wird als Validierungsexperiment ausgewiesen — nicht verschwiegen.

---

## 2026-08-10 — Schulferien fest im Code statt API-Abfrage zur Laufzeit

**Was:** Die Berliner Schulferien stehen als Tabelle in `analysis/calendar_de.py`.
Quelle: OpenHolidays API (openholidaysapi.org, EU-gefördert), abgefragt am 2026-08-10
für DE-BE. Aktualisierbar über `python analysis/calendar_de.py --refresh`, das einen
neuen Block zum Einsetzen ausgibt.

**Warum:** `ferien-api.de` — die naheliegende Wahl — liefert für Berlin 2026 und 2027
**eine leere Liste** zurück (getestet 2026-08-10). Ein solcher Fehler stürzt nicht ab:
Er markiert stillschweigend jeden Tag als Schultag und schwächt damit einen realen
Effekt ab, ohne dass irgendetwas auffällt. Zweitens muss die Auswertung im Januar 2027
und für eine Jury danach reproduzierbar bleiben — eine eingecheckte Tabelle kann nicht
offline gehen.

**Verworfen:** Laufzeit-Abfrage einer Ferien-API (siehe oben); die Termine selbst
abtippen (fehleranfällig und ohne Beleg).

**Gesetzliche Feiertage** kommen dagegen aus dem Paket `holidays`
(`holidays.Germany(subdiv="BE")`) — offline, versioniert, und es kennt Berliner
Besonderheiten wie den Frauentag am 8. März.

**Glücklicher Umstand:** Die Sommerferien 2026 enden am 22.08.2026, also elf Tage nach
Messbeginn. Der Kontrast Ferien/Schulzeit liegt damit direkt am Anfang der Messreihe
und nicht erst Monate später.

---

## 2026-08-10 — Kalendermerkmale in Ortszeit, Wetter-Join in UTC

**Was:** Stunde, Wochentag, Berufsverkehrs-Flags usw. werden aus **Europe/Berlin**
berechnet; der Wetter-Join läuft dagegen über die **UTC**-Stunde.

**Warum:** Berufsverkehr richtet sich nach der Uhr an der Wand, nicht nach UTC. Würde
man die Stunde aus UTC ableiten, verschöbe sich jedes Kalendermerkmal um eine Stunde,
sobald die Sommerzeit endet — und zwar am **25.10.2026**, also mitten im Messzeitraum.
Der DWD stempelt seine Stundenwerte umgekehrt in UTC (`MESS_DATUM`), deshalb ist dort
UTC richtig. Beide Zeitbasen sind bewusst gewählt, nicht zufällig gemischt.

---

## 2026-08-10 — `lead_time_s`: das Label ist eine Prognose, keine Messung

**Was:** Für jede Abfahrt wird mitgeschrieben, wie lange vor der geplanten Abfahrt die
letzte Beobachtung lag.

**Warum das die wichtigste eigene Fehlerquelle ist:** Die BVG-API liefert die
*aktuelle Schätzung* der Verspätung. Sobald ein Fahrzeug abgefahren ist, verschwindet
es von der Anzeige — die tatsächlich realisierte Verspätung wird also nie direkt
beobachtet. Das Label ist die letzte Schätzung vor Abfahrt.

**Gemessen (erste 444 Abfahrten):** Median 7,7 min, 95 %-Quantil 18,5 min, und in
14,0 % der Fälle **negativ**, d. h. die Abfahrt stand zum Beobachtungszeitpunkt noch
auf der Anzeige, obwohl ihre planmäßige Zeit schon vorbei war.

**Der subtile Teil — ein Selektionseffekt:** Ein verspätetes Fahrzeug bleibt *länger*
auf der Anzeige und wird deshalb näher an seiner echten Abfahrt beobachtet als ein
pünktliches. Die Labelqualität ist also **nicht gleichmäßig** über den Wertebereich
verteilt, sondern gerade dort am besten, wo die Verspätungen am größten sind. Das
gehört so in die Diskussion und wird geprüft, indem der Modellfehler gegen
`lead_time_s` aufgetragen wird.

---

## 2026-08-10 — Analyse-Abhängigkeiten getrennt von `requirements.txt`

**Was:** Neue Datei `requirements-analysis.txt` (pandas, pyarrow, huggingface_hub,
holidays, scikit-learn, LightGBM, SHAP, matplotlib). `requirements.txt` enthält nur
noch `httpx`.

**Warum:** Der Actions-Workflow führt bei **jedem** 15-Minuten-Poll
`pip install -r requirements.txt` aus. Lägen pandas und LightGBM darin, würde aus
einem 25-Sekunden-Job ein mehrminütiger — 96-mal am Tag, ohne jeden Nutzen, denn der
Logger braucht ausschließlich `httpx`.
