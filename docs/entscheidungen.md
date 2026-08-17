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

---

## 2026-08-10 — Messung: GitHub führt `*/15` faktisch nur **stündlich** aus

**Was:** Der Cron läuft — aber deutlich seltener als konfiguriert. Erste Messung
über die ersten drei Betriebsstunden (15:17–18:27 UTC):

| | |
|---|---|
| geplante Läufe laut `*/15` | ~9 |
| tatsächliche Läufe | 4 |
| **Abdeckung** | **44,4 %** |
| längste Lücke | 63 min |

Die beiden planmäßigen Läufe kamen um 16:24 und 17:27 UTC — also im Abstand von
etwa einer Stunde, nicht von 15 Minuten. Der stündliche Dashboard-Workflow
(`7 * * * *`) lief dagegen pünktlich. Das Muster passt zu GitHubs dokumentiertem
Verhalten, hochfrequente Zeitpläne unter Last zu verwerfen.

**Warum das ernst ist:** Verworfene Slots sind keine Verzögerung, sondern
**endgültiger Datenverlust** — dieselbe Unwiederbringlichkeit, die überhaupt der
Grund war, früh mit dem Loggen anzufangen. Bei rund einem Poll pro Stunde statt
vier ist die Datendichte viergeteilt, und bei 30 Minuten Vorschau entstehen zwischen
zwei Polls Abfahrten, die **nie** beobachtet werden.

**Wie es gefunden wurde:** durch das Dashboard (`scripts/collection_dashboard.py`),
das Soll- gegen Ist-Slots stellt. Genau dafür misst es Abdeckung und nicht nur
Zeilenzahl — „wir haben N Zeilen" hätte das Problem vollständig verdeckt.

**Entschieden (2026-08-10):** zwei sich ergänzende Änderungen statt Verzicht auf
Dichte.

1. **Vorschaufenster von 30 auf 65 Minuten.** Ist das Fenster länger als der
   schlimmstenfalls auftretende Poll-Abstand, wird **jede** Abfahrt mindestens
   einmal gesehen — auch wenn ein geplanter Lauf ausfällt. Das beseitigt den
   eigentlichen Schaden (nie beobachtete Abfahrten), nicht nur die Symptomatik.
2. **Jeder Lauf pollt in einer Schleife statt einmal**
   (`timeout 3300 python -m transit_logger.logger --loop 900 --out ndjson`).
   Damit hängt die Messdichte nicht mehr davon ab, wie oft GitHub auslöst: ein
   einziger gestarteter Lauf deckt ~55 Minuten mit vier Zyklen ab. Über die
   `concurrency`-Gruppe startet ein wartender Lauf unmittelbar nach dem
   vorherigen, die Abdeckung wird also durchgehend.

**Dabei gefundener Folgefehler:** `RESULTS = 60` begrenzte die Antwort pro
Haltestelle. Mit dem längeren Fenster liefert allein der Alexanderplatz **195**
Abfahrten — die alte Grenze hätte über zwei Drittel davon **stillschweigend**
abgeschnitten. Auf 250 angehoben, und `poll_once` warnt jetzt aktiv, falls die
Grenze je erreicht wird. Genau diese Art Fehler wäre in den Daten unsichtbar
geblieben: keine Fehlermeldung, nur weniger Zeilen. Messwerte danach: 614 statt
231 Abfahrten pro Zyklus, Maximum 195 pro Halt.

**Ebenfalls gehärtet:** Die Zyklusdatei wird jetzt erst als `.tmp` geschrieben und
dann per `os.replace` umbenannt. Da der Prozess unter `timeout` läuft, kann er
mitten im Schreiben abgebrochen werden; eine halb geschriebene `.gz`-Datei wäre
ein unlesbares Loch in der Messreihe. So ist ein Zyklus entweder vollständig oder
gar nicht vorhanden.

**Bewusst akzeptierter Preis (1):** Der Runner läuft damit nahezu durchgehend
(~20–24 h/Tag). Auf einem öffentlichen Repo sind Actions-Minuten unbegrenzt, und
die Läufe erzeugen genau die Forschungsdaten, für die dieses Repo existiert.
Sollte GitHub das drosseln, bleibt der Umzug auf einen Dauerbetrieb-Rechner
(`scripts/transit-logger.service`) der Rückfallweg.

---

## 2026-08-10 — Poll-Protokoll für den NDJSON-Backend (stiller Fehlschlag behoben)

**Wie es auffiel:** Im ersten Schleifen-Lauf lieferte der mittlere Zyklus
`logged 0 departures`:

```
18:31:53  logged 611 departures
18:47:31  logged 0 departures     <-- ?
19:01:52  logged 533 departures
```

Die Ursache ließ sich **nur aus der Laufzeit erschließen**: Ein normaler Zyklus
dauert ~5 s, dieser 43 s (Start 18:46:48, Ausgabe 18:47:31). Bei
`HTTP_TIMEOUT_S = 20` entspricht das etwa zwei in den Timeout gelaufenen Stops —
also ein Netz-/API-Aussetzer, keine leere Antwort.

**Der eigentliche Fehler:** Beide Fehlerzweige in `poll_once` schrieben
ausschließlich nach SQLite (`if conn is not None: log_poll(...)`). Im
NDJSON-Backend — dem, das produktiv in der CI läuft — ist `conn` immer `None`.
Ein fehlgeschlagener Poll hinterließ damit **keinerlei Spur**: keine
Log-Zeile, keine Datei, nichts. Zusätzlich wurde die Zyklusdatei nur bei
`cycle_rows` ≠ leer geschrieben, ein Totalausfall also gar nicht.

**Warum das für die Auswertung gefährlich ist:** Ein Poll, der lief und an dem
die API scheiterte, war in den Daten **byteweise identisch** mit einem Slot, den
GitHub nie ausgelöst hat. Die Abdeckungsanalyse hätte einen fremden
API-Ausfall dem GitHub-Scheduler zugeschrieben — eine falsche Ursachenzuordnung
mitten im Kapitel Fehlerquellen.

**Behoben:**
1. Stop-Fehler werden **immer** nach stdout gemeldet (in den Actions-Logs sichtbar),
   unabhängig vom Backend.
2. Pro Zyklus entsteht eine kleine Statusdatei `<stamp>Z.poll.json` mit dem Ergebnis
   je Stop (`ok` / `http_error` / `exception` samt Ursache). Bewusst als eigene
   Datei, damit die Abfahrtszeilen genau ein Schema behalten.
3. Die Zyklusdatei wird **auch leer** geschrieben. Leere Datei + Statusdatei heißt
   „gepollt, API ausgefallen"; gar keine Datei heißt „Slot nie ausgeführt".
4. Das Dashboard liest die Statusdateien, zählt solche Zyklen als stattgefunden und
   weist Stop-Fehler getrennt von Planungslücken aus.

**Geprüft** gegen einen erzwungenen Totalausfall (`API_BASE` auf einen toten Port):
vier Fehler protokolliert, Ursache je Stop erfasst, leere Zyklusdatei geschrieben,
Dashboard weist ihn als API-Ausfall statt als Lücke aus. Die dabei erzeugten
synthetischen Datensätze wurden anschließend aus `data/observations/` entfernt —
in den Forschungsdaten stehen ausschließlich echte Messungen.


---

## 2026-08-10 — Datenverlust durch abgelehnten Push (Vorfall, Ursache, Behebung)

**Was passiert ist:** Lauf `31419516255` pollte 55 Minuten lang korrekt vier Zyklen
(611 / 0 / 533 / 509 Abfahrten), scheiterte dann aber am abschließenden Push:

```
! [rejected]  main -> main (fetch first)
```

Drei Zyklusdateien mit rund **1.650 Abfahrten sind endgültig verloren** — der
Runner wird nach dem Lauf samt Dateisystem verworfen. Das ist genau die
Unwiederbringlichkeit, wegen der dieses Projekt überhaupt früh mit dem Loggen
begonnen hat.

**Ursache 1 — falsche Begründung beim Entfernen des Rebase.** Der Schritt
`git pull --rebase` war mit dem Argument gestrichen worden, eindeutige Dateinamen
machten Konflikte unmöglich. Das verwechselt zwei verschiedene Dinge:
eindeutige Dateinamen verhindern **Merge-Konflikte**, aber ein Push wird
zurückgewiesen, sobald das Remote *irgendeinen* neuen Commit hat — ob er die
gleichen Dateien berührt oder nicht. Bei einem 55-Minuten-Lauf ist das der
Normalfall, nicht der Ausnahmefall.

**Ursache 2 — alles hing an einem einzigen Schritt.** Ein Commit am Ende des Laufs
stellte die Daten aller vier Zyklen hinter einen einzigen fehlbaren Vorgang. Ein
abgelehnter Push kostete damit nicht einen Zyklus, sondern alle.

**Behoben:**
1. Der Workflow schleift jetzt selbst und **committet nach jedem Zyklus**
   (`--once` statt `--loop`). Ein fehlgeschlagener Push kostet höchstens einen
   Zyklus statt vier.
2. `git pull --rebase` ist zurück, mit fünf Wiederholungen und **ohne** `|| true`.
   Da Dateinamen weiterhin nie kollidieren, kann der Rebase selbst nicht in einen
   Konflikt laufen.
3. Der Schritt endet mit Exit-Code ≠ 0, wenn die Versuche erschöpft sind — der
   Lauf schlägt also sichtbar fehl, statt Daten stillschweigend zu verlieren.

**Geprüft am reproduzierten Rennen, nicht am Glücksfall:** Während Lauf
`31424332673` lief, wurde absichtlich ein fremder Commit auf `main` gepusht —
exakt die Konstellation, an der der vorherige Lauf zerbrach. Ergebnis in der
Historie:

```
9aa681b data: poll 2026-08-10T19:44Z   <- 2. Zyklus, sauber obendrauf gepusht
bbb3f68 test: move remote during run   <- der störende Commit
b59ae58 data: poll 2026-08-10T19:29Z   <- 1. Zyklus
```

Der Lauf hat über den fremden Commit rebased und erfolgreich gepusht.

**Was daraus für die Langfassung folgt:** Bei einer selbst erhobenen Messreihe ist
nicht nur die Messung selbst eine Fehlerquelle, sondern auch der **Transportweg der
Daten**. Ein Fehler zwischen Messung und Speicherung ist von einem Messausfall in
den Daten nicht zu unterscheiden — und war hier obendrein durch eine plausible,
aber falsche Annahme verursacht. Deshalb protokolliert der Logger inzwischen jeden
Poll (siehe Eintrag zum Poll-Protokoll), und das Dashboard weist Abdeckung getrennt
von API-Fehlern aus.

---

## 2026-08-17 — Die Zielgröße ist quantisiert und extrem nulllastig (Konsequenz für die Modellwahl)

**Anlass:** Rückfrage, ob „0 min" auf dem Dashboard stimmen kann.

**Es stimmt — und zwar als *Median*, nicht als Mittelwert.** Über 35.235 nicht
ausgefallene Abfahrten (7 Tage):

| | |
|---|---|
| exakt 0 | **74,4 %** |
| verspätet (> 0) | 17,9 % |
| **verfrüht (< 0)** | **7,7 %** |
| ohne Echtzeitwert (null) | 7,9 % |
| Median / Mittelwert | **0,00 min / 0,50 min** |

**Befund 1 — die Daten sind echt, kein zurückgespiegelter Fahrplan.** Drei
unabhängige Belege: (a) `null` existiert getrennt von `0`, die API unterscheidet
also „keine Echtzeit" von „pünktlich"; (b) es gibt **negative** Verspätungen (Bus
13,7 %, Tram 12,8 % zu früh) — ein bloß zurückgegebener Fahrplan könnte das nie
erzeugen; (c) die Produkte unterscheiden sich stark (Express 45,1 % pünktlich,
Mittel 8,9 min gegenüber S-Bahn 88,1 %, Mittel 0,4 min), passend zum Befund aus
dem Bahndatensatz.

**Befund 2 — die Verspätung ist auf ganze Minuten quantisiert.** 100,00 % aller
Werte sind Vielfache von 60 s, insgesamt nur 75 verschiedene Werte. „Verspätung =
0" heißt also **„unter einer Minute"**, nicht „sekundengenau pünktlich". Das ist
eine Auflösungsgrenze der Messung und gehört als solche in die Fehlerquellen:
Unterschiede unterhalb einer Minute sind grundsätzlich nicht beobachtbar.

**Befund 3 — Verspätung ist ein seltenes Ereignis, kein Normalzustand.** Der
Mittelwert entsteht fast vollständig im Rand der Verteilung:

| Bereich | Anteil der Abfahrten | Anteil an der Gesamtverspätung |
|---|---|---|
| verfrüht | 7,7 % | −24,3 % |
| exakt 0 | 74,4 % | 0,0 % |
| 1–2 min | 12,1 % | 31,0 % |
| 3–5 min | 3,5 % | 25,7 % |
| **6+ min** | **2,3 %** | **67,6 %** |

Das schlechteste **1 %** der Abfahrten trägt **48,6 %** aller Verspätungsminuten.

**Konsequenz für die Modellierung — und der Grund, das hier festzuhalten:** Eine
MAE-Regression auf diese Zielgröße ist nahezu wertlos. Die konstante Vorhersage 0
erreicht bereits einen MAE von ~0,50 min und schlägt damit die meisten Modelle,
ohne irgendetwas erklärt zu haben. Die inhaltlich sinnvolle Aufgabe ist die
**unbalancierte Klassifikation** („Verspätung ≥ 3 min", ~5,8 % positive Fälle)
bzw. die Modellierung des Randes. Das ist keine Notlösung, sondern folgt direkt
aus der gemessenen Verteilung — und es beantwortet die Forschungsfrage besser:
Gefragt ist, welche Faktoren Verspätung *treiben*, und getrieben wird sie
nachweislich von den seltenen großen Fällen, nicht vom Normalbetrieb.

---

## 2026-08-17 — Kernbefund: Je schwerer die Störung, desto besser vorhersagbar

**Anlass:** Kritische Prüfung, ob das Projekt mit dieser Datenlage tragfähig ist.

**Messung** (7 Tage, 32.450 auswertbare Abfahrten, zeitlich getrennter Test):

| Ziel | Anteil | Ereignisse | GBM ROC-AUC | Baseline (Linie × Stunde) | Gewinn |
|---|---|---|---|---|---|
| ≥ 1 min | 17,91 % | 5.813 | 0,695 | 0,644 | +0,051 |
| ≥ 3 min | 5,78 % | 1.875 | 0,751 | 0,684 | +0,067 |
| ≥ 5 min | 3,00 % | 973 | 0,772 | 0,637 | +0,135 |
| **≥ 10 min** | **1,12 %** | **364** | **0,849** | 0,683 | **+0,166** |

**Befund:** Schwere und Vorhersagbarkeit steigen **gemeinsam**. Verspätungen unter
einer Minute sind Rauschen und praktisch nicht vorhersagbar; ernsthafte Störungen
ab 10 Minuten sind deutlich systematisch (AUC 0,849, PR-AUC 0,222 bei 0,63 %
Positivrate im Test — **35,5-facher** Lift gegenüber Zufall). Auch der *Mehrwert
des Modells* gegenüber der trivialen Nachschlagetabelle wächst mit der Schwere
(+0,051 → +0,166).

**Warum das die Fragestellung schärft statt sie zu beschädigen:** Dieselben
seltenen Ereignisse tragen den Großteil der Gesamtverspätung (2,3 % der Abfahrten
= 67,6 % aller Verspätungsminuten, siehe vorheriger Eintrag). Die praktisch
relevanten *und* die statistisch vorhersagbaren Fälle sind also dieselben. Die
starke Nulllastigkeit der Zielgröße ist damit kein Defekt der Daten, sondern
verweist auf die richtige Zielgröße: **nicht „wie viele Minuten Verspätung", sondern
„kommt es zu einer ernsthaften Störung".**

**Was ausdrücklich NICHT hilft — mit Zahlen belegt:**
- **Wetter:** Beitrag −0,016 (also nichts). Bislang jedoch **nur Sommerdaten**,
  ohne Schnee und Eis. Der eigentliche Test steht mit den Winterdaten Nov–Jan an.
- **Kalender:** Schulferien, Feiertage, Wochenende — jeweils ±0,000.
- **Netzzustand** (Verspätungsquote derselben Linie/Haltestelle in der
  vorangehenden Stunde, um 30 min verzögert und damit leckagefrei): nur **+0,015
  AUC**. Die ursprüngliche Vermutung, das Merkmalsset sei zu dünn und Netzzustand
  würde viel bringen, ist damit **widerlegt** — festgehalten, weil eine widerlegte
  Hypothese genauso zum Ergebnisteil gehört wie eine bestätigte.

**Vorläufige Antwort auf „welche Faktoren treiben sie tatsächlich?":** Verspätung
in Berlin ist **strukturell, nicht umweltbedingt** — sie hängt an Linie
(+0,136 AUC), Verkehrsmittel (+0,060) und Tageszeit, nicht am Wetter und nicht am
Kalender. Das ist ein belastbares und der Alltagsintuition zuwiderlaufendes
Ergebnis.

**Einschränkung, die mitgeschrieben gehört:** Die Zeile „≥ 10 min" beruht auf 364
Ereignissen, im Testfenster auf ~110. Das Konfidenzintervall ist entsprechend
breit. Bis Ende Januar sind rund 9.000 solcher Ereignisse zu erwarten; erst dann
ist der Wert belastbar. `analysis/signal_check.py --sweep` reproduziert die
Tabelle jederzeit.
