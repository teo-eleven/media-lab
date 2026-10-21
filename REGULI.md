# Reguli de lucru pentru acest proiect

Citește tot fișierul înainte să scrii orice linie de cod. Aceste reguli au prioritate față de orice obicei implicit al tău. Dacă o instrucțiune de-a mea din chat contrazice acest fișier, întrebi care are prioritate — nu presupui.

---

## 0. Reguli permanente (valabile în TOATE fazele)

1. **Faci exact ce cer, nimic în plus.** Fără funcționalități „bonus", fără refactorizări nesolicitate, fără schimbat structura de foldere, fără înlocuit librării.
2. **Nu inventezi.** Dacă nu ești sigur de un API, o versiune de pachet, o semnătură de funcție sau o coloană din baza de date — verifici în cod sau întrebi. Nu ghicești. Spui explicit „nu știu / nu am verificat" când e cazul.
3. **Fără cod placeholder.** Fără `TODO`, fără `// implement later`, fără funcții care returnează date fake. Dacă ceva nu poate fi implementat acum, te oprești și îmi spui de ce.
4. **Totul rulează local.** Fără servicii cloud, fără deploy, fără apeluri către API-uri externe care nu sunt deja în proiect, fără instalat nimic global fără să mă întrebi.
5. **Nu atingi fișiere din afara scopului pasului curent.** Dacă un fix cere modificarea altui fișier, îmi spui înainte.
6. **Fără secrete în cod și fără secrete în git.** Toate cheile, parolele, connection string-urile trec prin `.env`, care e în `.gitignore`. În repo intră doar `.env.example` cu valori goale.
7. **Un pas o dată.** Termini pasul, îl testezi, raportezi, aștepți confirmarea mea. Nu treci mai departe singur decât dacă ți-am zis explicit „mergi până la capăt".
8. **Nu ștergi și nu suprascrii cod existent** fără să-mi arăți întâi ce dispare și de ce.
9. **Când raportezi, raportezi adevărat.** Dacă un test pică, spui că pică. Dacă nu ai rulat ceva, spui că nu ai rulat. Niciodată „gata, funcționează" fără să fi rulat efectiv.
10. **Limba:** răspunzi în română, dar codul, comentariile, numele de variabile și mesajele de commit sunt în engleză.
11. **Ecosistem complet de agenți, skill-uri și tool-uri:** Pentru orice sarcină, etapă, review sau prompt, se folosesc proactiv toți agenții disponibili (subagenți `research`, `self`, agenți specializați), skill-urile relevante și tool-urile sistemului, orchestrate eficient pentru calitate și robustețe fără compromisuri.

---

## 1. FAZA 1 — Înțelegerea contextului (fără cod)

Înainte de orice:

1.1. Inventariezi ce există deja: structura de foldere, `package.json` / `requirements.txt` / `go.mod`, config-urile, migrațiile, testele existente.
1.2. Identifici: limbaj, framework, versiuni exacte, package manager, runner de teste, linter, formatter, ORM/driver de bază de date.
1.3. Rulezi comenzile de bază ca să vezi starea reală: instalare dependențe, build, teste. Îmi raportezi ce merge și ce e deja stricat.
1.4. **Îmi pui toate întrebările neclare deodată, într-o singură listă numerotată.** Nu începi implementarea cu presupuneri nerezolvate. Dacă ceva e ambiguu în cerințe, întrebi. E mai ieftin să întrebi decât să refac.

Output-ul fazei: un rezumat scurt al stack-ului + lista de întrebări. Te oprești aici.

---

## 2. FAZA 2 — Planul (fără cod)

2.1. Creezi `PLAN.md` în rădăcina proiectului cu:
- lista completă de funcționalități, împărțită în pași numerotați `1, 2, 3...`
- pentru fiecare pas: ce fișiere se ating, ce se adaugă, ce teste îl acoperă, de ce depinde
- ordinea de implementare (dependențele primele)
- ce e explicit în afara scopului

2.2. Pașii sunt **verticali și livrabili**: fiecare pas trebuie să lase proiectul într-o stare care rulează. Fără „scriem toate modelele acum și le legăm la sfârșit".
2.3. Îmi arăți planul și **aștepți aprobarea mea**. Nu scrii cod până nu zic „OK, mergi".
2.4. `PLAN.md` se actualizează pe parcurs: bifezi pașii terminați, notezi devierile.

---

## 3. FAZA 3 — Setup, configurare, mediu

Înainte de prima funcționalitate:

3.1. Configurezi complet mediul local: dependențe fixate pe versiuni (lockfile commis), scripturi de rulare în `package.json` / `Makefile`.
3.2. Centralizezi configurarea într-un singur loc (`config/`), citită din variabile de mediu, cu validare la pornire și valori implicite sensibile. Aplicația trebuie să crape clar și explicit la pornire dacă lipsește o variabilă obligatorie, nu peste 3 ore într-un request.
3.3. Creezi `.env.example` complet, cu toate cheile documentate.
3.4. Configurezi linter + formatter + type checking și te asiguri că trec pe curat.
3.5. Configurezi runner-ul de teste și scrii un test trivial care trece, ca să confirmi că infrastructura de test funcționează.
3.6. `.gitignore` corect: `.env`, artefacte de build, `node_modules`, fișiere de bază de date locale, cache-uri.

Raportezi: „setup gata, iată comenzile de rulare". Aștepți.

---

## 4. FAZA 4 — Implementarea, pas cu pas

Pentru **fiecare** pas din `PLAN.md`, în ordinea din plan:

4.1. Anunți ce pas începi și ce fișiere vei atinge.
4.2. Scrii implementarea completă — fără stub-uri, fără cod mort.
4.3. Tratezi erorile explicit: validare de input, cazuri limită, ce se întâmplă când lipsesc datele. Fără `try/catch` gol care înghite erori.
4.4. Scrii testele pentru pasul respectiv (happy path + cel puțin două cazuri limită + un caz de eroare).
4.5. **Rulezi efectiv**: teste, linter, type check, build. Nu „ar trebui să meargă". Rulezi și lipești rezultatul real.
4.6. Dacă pică ceva, repari înainte să treci mai departe. Un pas nu e terminat cât timp suita de teste e roșie.
4.7. Commit atomic per pas, mesaj în format `feat: ...` / `fix: ...` / `chore: ...`.
4.8. Raportezi: ce ai făcut, ce fișiere ai atins, ce teste au trecut, ce a rămas deschis. Apoi te oprești și aștepți.

**Regula de aur:** niciodată nu se cumulează două funcționalități netestate. Dacă vezi că un pas e prea mare, îl spargi și îmi spui.

---

## 5. FAZA 5 — Sincronizări (după fiecare pas care le atinge)

De fiecare dată când modifici ceva din lista de mai jos, actualizezi **toate** locurile dependente, în același commit:

- **Schema bazei de date** → migrație nouă (cu rollback) + modele + tipuri + seed-uri + query-uri afectate
- **Contract de API** → handler + validare + tipuri client + documentație + teste
- **Tipuri / interfețe** → toate implementările și toate consumatoarele
- **Variabile de config** → `.env.example` + validarea de config + README
- **Dependențe** → lockfile commis
- **Comportament vizibil** → README + `PLAN.md`

Înainte să declari un pas terminat, cauți explicit în tot proiectul referințe rămase la vechea formă (nume vechi de câmp, semnătură veche, import mort) și îmi raportezi ce ai găsit.

---

## 6. FAZA 6 — Integrare completă

După ce toți pașii din plan sunt implementați:

6.1. Pornești aplicația de la zero, într-un mediu curat (dependențe reinstalate, bază de date recreată din migrații + seed).
6.2. Rulezi suita completă de teste. Toate trebuie să treacă.
6.3. Parcurgi manual fiecare funcționalitate din `PLAN.md`, în ordine, și confirmi că fluxurile chiar funcționează cap-coadă, nu doar unitar.
6.4. Verifici că funcționalitățile se leagă între ele coerent — că nu am două module care fac același lucru diferit, două surse de adevăr pentru aceeași dată, sau două stiluri de tratare a erorilor.
6.5. Raportezi orice discrepanță găsită față de plan.

---

## 7. FAZA 7 — Code review (îl faci tu, pe propriul cod)

Reciteșți tot ce ai scris, ca și cum ar fi codul altcuiva pe care trebuie să-l aprobi. Îmi dai un raport structurat pe:

7.1. **Corectitudine** — logică greșită, cazuri limită netratate, off-by-one, condiții de cursă, resurse neînchise.
7.2. **Securitate** — input nevalidat, SQL injection, secrete expuse, autorizare lipsă, dependențe cu vulnerabilități cunoscute.
7.3. **Consistență** — convenții de denumire, structură, tratarea erorilor, formatul răspunsurilor.
7.4. **Duplicare și cod mort** — funcții nefolosite, importuri moarte, copy-paste.
7.5. **Performanță** — query-uri N+1, bucle inutile, lipsa indecșilor, operații sincrone care blochează.
7.6. **Testabilitate** — ce nu e acoperit de teste și ar trebui să fie.
7.7. **Documentație** — README-ul chiar permite unui om nou să pornească proiectul de la zero?

Formatul raportului: fiecare problemă cu **fișier:linie**, severitate (critic / major / minor), și fixul propus. **Nu repari nimic încă** — îmi arăți lista, decid eu ce se repară.

Fii sincer și critic aici. Un review care zice „totul arată bine" e un review inutil. Dacă chiar nu găsești nimic major, spui asta explicit și enumeri ce ai verificat.

---

## 8. FAZA 8 — Bug fixing

Pentru fiecare bug (găsit de tine sau raportat de mine):

8.1. **Reproduci întâi.** Fără reproducere, nu repari — întrebi cum se reproduce.
8.2. **Scrii un test care pică** din cauza bug-ului. Confirmi că pică.
8.3. Identifici **cauza reală**, nu simptomul. Îmi explici cauza în 2-3 propoziții înainte să repari. Fără fixuri cosmetice care ascund problema.
8.4. Aplici fixul minim necesar. Fără refactorizare oportunistă în același commit.
8.5. Rulezi testul — trebuie să treacă. Rulezi **toată** suita — nimic nu trebuie să se strice.
8.6. Verifici dacă aceeași greșeală apare și în alte locuri din proiect.
8.7. Commit separat, `fix: <ce anume>`.

---

## 9. FAZA 9 — Predarea

9.1. `README.md` complet: ce face proiectul, cerințe, instalare pas cu pas, cum se rulează, cum se rulează testele, variabilele de mediu, structura de foldere.
9.2. Verifici README-ul urmând instrucțiunile **literal**, de la zero. Dacă un pas nu merge exact cum e scris, îl corectezi.
9.3. Îmi dai un rezumat final: ce s-a implementat, ce a rămas în afara scopului, limitări cunoscute, ce ar urma logic mai departe.

---

## 10. Cum îmi comunici

- Răspunsuri scurte și concrete. Fără reformulat cerința mea înapoi la mine, fără „excelentă întrebare".
- Când ai o nelămurire, o pui **înainte** de implementare, nu după.
- Când nu ești de acord cu ce cer — spui, cu motivul tehnic. Vreau contra-argumente, nu aprobare automată.
- Când ceva a mers prost, spui direct ce a mers prost.
- La final de fiecare pas: **ce am făcut / ce am rulat / rezultatul real / ce urmează**.
