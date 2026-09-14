# PLAN.md — media-lab

Local video/photo editing pipeline for short social clips, built as a typed
wrapper over Kinocut (`kino` CLI). Every render is verified before it is
reported as done.

Working contract: see [REGULI.md](./REGULI.md).

---

## Decizii luate (confirmate de Teo, 2026-09-03)

| Subiect | Decizie |
|---|---|
| Formă proiect | Wrapper propriu peste kinocut, cu teste |
| ffmpeg | Binar static local în `bin/`, NU global |
| Python | 3.12.13 via uv, `.venv` în proiect (`upscale` cere ≤3.12) |
| Module kinocut | bază + `hyperframes` + `image` + `transcribe` + `stems` + `upscale` |
| Interfață unelte | CLI `kino` apelat prin subprocess, NU server MCP |
| Kinocut pinuit la | `1.15.1` |
| hyperframes pinuit la | `0.8.27` |

---

## Ce NU intră în scop (explicit)

- Orice serviciu cloud, API extern, upload sau deploy.
- `video_body_swap` și orice funcție de tip deepfake — nu o expunem în wrapper.
- Generare video AI (text-to-video). Nu rulează util pe 8 GB.
- Decupat OBIECTE (`birefnet-general`). Cere extra `object-matte`, care —
  verificat în docs/PRODUCT_MATTE.md — NU e în pip 1.15.0 publicat, doar pe dev
  tip. Decupăm doar PERSOANE (`u2net_human_seg`).
- Editor grafic / UI. Totul e CLI, condus din chat.
- Server MCP. Decis: CLI.

---

## Nume de comenzi CLI — VERIFICATE (2026-09-03, kinocut 1.15.1)

Rulat `kino --help` pe instalarea reala. 167 comenzi disponibile.
Numele deduse in planul initial, confirmate sau corectate:

| Ce credeam | Real | Stare |
|---|---|---|
| `hyperframes-remove-background` | `hyperframes-remove-background` | confirmat |
| `composite-layers` | `composite-layers` | confirmat |
| `resize` | `resize` | confirmat |
| `filter` | `filter` | confirmat |
| `repurpose` | `repurpose` | confirmat |
| `duck-audio` | **`audio-bed`** | CORECTAT - o singura comanda face ducking + normalizare loudness |
| `release-checkpoint` | **`video-quality-check`** / `video-publish-gate` | CORECTAT - nu exista sub numele presupus |
| `quality-check` | **`video-quality-check`** | CORECTAT |

Alte comenzi confirmate, folosite de pasii de mai jos: `info`, `trim`, `merge`,
`chroma-key`, `add-audio`, `normalize-audio`, `extract-frame`, `color-grade`,
`thumbnail`, `effect-vignette`, `effect-glow`, `effect-noise`,
`video-ai-color-grade`, `image-edit`, `still-grade`, `still-gate`,
`still-package`, `doctor`.

Modele de decupare confirmate prin `kino --format json
hyperframes-remove-background --info`:
- `u2net_human_seg` (implicit, backend hyperframes) - PERSOANE, disponibil acum
- `birefnet-general` (backend kinocut-onnx) - obiecte, cere extra `object-matte`, in afara scopului

---

## Ordinea de implementare

Pasul 0 e prerechizit pentru tot. Pasul 1 e fundația pe care se sprijină
pașii 2-7. Pașii 2-6 sunt independenți între ei și pot fi reordonați.
Pasul 7 depinde de 2-6.

---

## Pas 0 — Setup mediu (FAZA 3) — GATA (41bdcf9)

**Depinde de:** nimic.

**Ce se face:**
- `bin/` — ffmpeg + ffprobe static (Apple Silicon), versiune notată în README
- `.venv` via `uv`, Python 3.12.13
- `pyproject.toml` — kinocut==1.15.1 cu extras, ruff, mypy, pytest, pytest-cov
- `uv.lock` commis
- `node_modules/` local — `hyperframes@0.8.27`, plus `package.json`/`package-lock.json`
- `.gitignore` — `.venv`, `node_modules`, `bin/`, `in/`, `out/`, `.env`, `__pycache__`, `.pytest_cache`
- `.env.example` — documentat, cu toate cheile
- `Makefile` — `setup`, `test`, `lint`, `typecheck`, `check`
- `git init` + primul commit

**Fișiere atinse:** `pyproject.toml`, `uv.lock`, `package.json`, `package-lock.json`,
`.gitignore`, `.env.example`, `Makefile`, `bin/`, `README.md`

**Teste:** `tests/test_setup.py` — un test trivial care trece, ca să confirm că
infrastructura de test funcționează (cerut de 3.5).

**Notă onestă despre secrete:** proiectul e 100% local și nu are nicio cheie
API. `.env` va conține doar căi și praguri (`MEDIA_LAB_FFMPEG_DIR`,
`MEDIA_LAB_IN_DIR`, `MEDIA_LAB_OUT_DIR`, `MEDIA_LAB_KINO_TIMEOUT_S`,
`MCP_VIDEO_HYPERFRAMES_COMMAND`). Nu inventez secrete ca să bifez regula 0.6.

**Verificare (rulat efectiv):** `ffmpeg -version`, `kino --version`,
`npx hyperframes --version`, `make check`.

---

## Pas 1 — Schelet: config, runner kino, verificare — GATA (f23efa4)

**Depinde de:** Pas 0.

**Ce se face:**
- `src/media_lab/config.py` — citește env, validează LA PORNIRE, crapă explicit
  dacă lipsește ffmpeg sau kino. Valori implicite sensibile.
- `src/media_lab/kino.py` — un singur loc care apelează `kino` prin subprocess:
  timeout, cod de ieșire, stderr capturat, erori tipizate. Nicio comandă `kino`
  nu se apelează din altă parte.
- `src/media_lab/verify.py` — după fiecare randare: ffprobe pe output
  (durată, rezoluție, fps, prezență audio, canal alpha) + extras un cadru
  pentru inspecție vizuală. Ridică eroare dacă output-ul nu respectă
  așteptările. **Ăsta e stratul anti-rateuri.**
- `src/media_lab/paths.py` — non-distructiv: sursele din `in/` nu se ating
  NICIODATĂ, tot se scrie în `out/`. Refuză să scrie peste un fișier existent
  fără `--force`.
- `src/media_lab/cli.py` — entry point, deocamdată doar `media-lab doctor`

**Fișiere atinse:** cele de mai sus + `tests/`

**Teste:** config lipsă/invalid; kino inexistent; timeout; ffprobe pe fișier
corupt; refuz de suprascriere; refuz de scriere în `in/`.

**Livrabil:** `media-lab doctor` raportează starea reală a mediului.

---

## Pas 2 — `cutout`: decupat persoana de pe fundal — GATA (c233d93)

**Depinde de:** Pas 1.

**Ce se face:** `src/media_lab/recipes/cutout.py` — wrapper peste
`kino hyperframes-remove-background` (model `u2net_human_seg`). Merge pe video
și pe imagini. Output cu alpha păstrat.

**Teste:** clip sintetic scurt generat cu ffmpeg lavfi; verificare că output-ul
are canal alpha; input inexistent; input fără persoană (comportament declarat,
nu ascuns); format neacceptat.

**Livrabil:** `media-lab cutout in/clip.mp4 -o out/cutout.webm`

---

## Pas 3 — `backdrop`: fundal nou + compositing — GATA (32a4baf)

**Depinde de:** Pas 2.

**Ce se face:** `src/media_lab/recipes/backdrop.py` — construiește spec-ul de
layere și apelează `kino composite-layers`. Fundal imagine sau video.
Validează că fundalul și subiectul au rezoluții compatibile; nu redimensionează
tăcut, ci raportează.

**Teste:** fundal imagine; fundal video mai scurt decât subiectul; rezoluții
diferite; fundal lipsă.

**Livrabil:** `media-lab backdrop out/cutout.webm --bg in/bg.jpg -o out/composed.mp4`

---

## Pas 4 — `filters`: filtre și color grading — GATA (fc428b2)

**Depinde de:** Pas 1.

**Ce se face:** `src/media_lab/recipes/filters.py` — set restrâns de filtre
expuse controlat, cu parametri validați și limitați (nu pasăm valori brute în
ffmpeg). Include vignette, grain, glow, saturație, contrast, color grade.

**Teste:** parametru în afara intervalului; filtru inexistent; lanț de 2 filtre;
input fără video stream.

**Livrabil:** `media-lab filter out/composed.mp4 --preset warm -o out/graded.mp4`

---

## Pas 5 — `music`: muzică de fundal cu ducking — GATA (7ae0ca6)

**Depinde de:** Pas 1.

**Ce se face:** `src/media_lab/recipes/audio_bed.py` — muzica coboară automat
sub voce (sidechain ducking), normalizare EBU R128. Verifică LUFS-ul final cu
ffprobe/loudnorm și raportează valoarea reală.

**Teste:** video fără pistă audio; muzică mai scurtă decât clipul (loop);
muzică mai lungă (trim + fade); LUFS țintă atins.

**Livrabil:** `media-lab music out/graded.mp4 --track in/song.mp3 -o out/mixed.mp4`

---

## Pas 6 — `short`: export vertical 9:16 + quality gate — GATA (fc428b2)

**Depinde de:** Pas 1.

**Ce se face:** `src/media_lab/recipes/to_short.py` — reîncadrare 9:16,
export, apoi quality gate înainte de a declara gata (luminozitate, contrast,
nivel audio, thumbnail).

**Teste:** sursă 16:9; sursă deja verticală; sursă pătrată; clip care pică
quality gate-ul (trebuie să RAPORTEZE, nu să treacă tăcut).

**Livrabil:** `media-lab short out/mixed.mp4 -o out/final_9x16.mp4`

---

## Pas 7 — `pipeline`: fluxul complet într-o comandă — GATA (c4a6664)

**Depinde de:** Pașii 2-6.

**Ce se face:** `src/media_lab/pipeline.py` — înlănțuie cutout → backdrop →
filter → music → short. Fișiere intermediare într-un folder de lucru,
păstrate pentru inspecție. Dacă un pas pică, se oprește și raportează exact
unde, fără să șteargă ce s-a produs până acolo.

**Teste:** flux complet pe clip sintetic; eșec la pasul 3 din 5 (oprire curată);
reluare; verificare că `in/` a rămas neatins.

**Livrabil:**
`media-lab pipeline in/clip.mp4 --bg in/bg.jpg --track in/song.mp3 -o out/`

---

## Stare la 2026-09-03

Toți pașii 0-7 sunt implementați, testați și commiși. Ultimul commit: `dbb9a56`.
Suita: 115/115, exit 0. ruff + mypy curate. Build: wheel și sdist se produc.

---

## Extensii — NU intră în scop până nu le confirmi

Ai instalat modulele, dar nu mi-ai cerut comenzi pentru ele. Le las în afara
planului până zici tu:

- **E1 — `subtitles`**: transcriere Whisper + subtitrări arse (modul `transcribe`)
- **E2 — `stems`**: separare voce/muzică (modul `stems`)
- **E3 — `upscale`**: mărire AI (modul `upscale`)
- **E4 — `photo`**: procesare pe loturi de imagini

---

## Definiția de „terminat" pentru fiecare pas

1. Implementare completă, fără stub-uri
2. Teste: happy path + minim 2 cazuri limită + 1 caz de eroare
3. `make check` verde (pytest + ruff + mypy), coverage ≥ 80%
4. Rezultatul real al rulării, lipit în raport
5. Commit atomic `feat:` / `fix:` / `chore:`
6. Raport: ce am făcut / ce am rulat / rezultatul real / ce urmează


---

## Devieri față de planul inițial (FAZA 2.4)

1. **Detecția alpha era greșită** (Pas 2). VP9-in-WebM ține alpha în
   BlockAdditional, deci `pix_fmt` rămâne `yuv420p`. Verificarea din Pasul 1
   dădea fals negativ pe fiecare cutout. Corectat: se citește și tagul
   `alpha_mode` din container, insensibil la majuscule (ffmpeg îl scrie cu
   litere mici, kinocut cu majuscule).

2. **`composite-layers` confină sursele** (Pas 3). Orice `src` trebuie să
   fie sub directorul spec-ului — guardrail de securitate deliberat. Sursele
   se stagează prin hardlink lângă spec, în `work/`, nu se referențiază din
   `in/`.

3. **Compositing-ul produce video fără audio** (Pas 3). Vocea originală
   trebuie reatașată. Pasul 7 va folosi `kino extract-audio` pe sursă,
   `audio-bed` pentru mixul voce+muzică, apoi `kino add-audio` peste
   compozit. Nu era prevăzut în planul inițial.

4. **`kino audio-bed` nu funcționează pe macOS** (Pas 5). kinocut 1.15.1 îl
   blochează în spatele unor „immutable source snapshots" construite pe
   `os.memfd_create` — API exclusiv Linux. Eșuează cu `source_identity_changed`
   înainte să atingă vreun fișier. Verificat direct: `hasattr(os,'memfd_create')`
   e `False` pe Darwin. Afectate sunt doar `audio-bed` și `body-swap`; restul
   comenzilor merg. Ducking-ul e implementat în schimb direct în ffmpeg
   (`sidechaincompress` + `loudnorm`), în `recipes/audio_bed.py`, cu un modul
   nou `ffmpeg.py` ca singur punct de apel către ffmpeg.

5. **Clasă de eroare nouă: `ValidationError`** (integrare). Rețetele foloseau
   inconsecvent `ConfigError` și `MediaLabError` pentru validarea argumentelor.
   Unificat.

6. **`KinoError` păstra doar stderr** (integrare). `video-quality-check
   --fail-on-warning` iese cu cod 1, stderr gol, dar scrie raportul JSON pe
   stdout — care se pierdea. `KinoError` păstrează acum și stdout.

7. **Pas nou, neprevăzut: `recipes/audio_attach.py`** (Pas 7). Compositing-ul
   scoate audio, deci vocea originală trebuie reatașată înainte de mixul cu
   muzică. Nu era în planul inițial.

8. **Compositorul kinocut plafonează la 25 fps** (Pas 3, descoperit pe material
   real). Randează maximum 25 fps dar etichetează stream-ul cu fps-ul cerut,
   deci un clip de 30fps ieșea 5.37s în loc de 6.43s. Măsurat sistematic:
   durata x 30 cerut -> durata x 25 cadre. `backdrop.py` plafonează canvas-ul
   și raportează. Stratul de verificare a prins problema.

9. **Cutout-ul video s-a mutat de pe .webm pe .mov/ProRes 4444** (Pas 2,
   descoperit pe material real). Alpha-ul VP9 exista, dar doar decodorul
   `libvpx-vp9` îl expune, iar compositorul kinocut nu îl cere - compunea
   subiectul ca dreptunghi opac peste fundal. ProRes are alpha citit nativ.

10. **Verificare nouă: `measure_alpha_spread`** (Pas 2). Prezenta canalului
    alpha nu garanta ca exista un matte. Se masoara acum ca alpha chiar
    variaza; un matte uniform ridica VerificationError.

---

# FAZA 2 — Optimizare & Video-Agent Toolkit (A + B)

Extinderea pachetului `media-lab` cu uneltele native de compoziție din roadmap-ul Phase 2
și înlocuirea scripturilor legacy din `docs/video-agent/pipeline/` cu rețete native,
robuste, testate și integrate în CLI.

## Ce NU intră în scop (explicit)

- Servicii externe cloud sau API-uri terțe (fără scraping Pexels/Pixabay ce necesită conexiuni externe; fără backend CUDA IC-Light la distanță).
- Modificarea contractului sau comportamentului uneltelor existente care trec testele (`matte`, `compose`, `proxy`, `filter`, `audio-bed`, `short`).
- Orice operație distructivă asupra surselor din `in/`.

---

## Pași de implementare

### Pasul 1 — `upscale` (Real-ESRGAN native recipe + CLI) — GATA (b1304a0)
- **Ce se adaugă**:
  - `src/media_lab/ml/realesrgan_infer.py`: driver ML dedicat rulat ca subprocess prin `ml_runner` (pentru menținerea graniței de proces și izolarea memoriei); suportă `--scale` (2, 4), `tile`, `tile_pad`, device auto MPS/CPU; scalează canalele RGB cu RealESRGAN și canalul alpha cu Lanczos; procesează secvențe PNG.
  - `src/media_lab/recipes/upscale.py`: rețetă tipizată `upscale_frames` / `upscale` cu suport pentru secvențe PNG sau ProRes 4444 `.mov`, verificare căi, protecție `--force`.
  - `src/media_lab/cli.py`: adăugare subcomandă `media-lab upscale`.
- **Fișiere atinse**:
  - `src/media_lab/ml/realesrgan_infer.py` (nou)
  - `src/media_lab/recipes/upscale.py` (nou)
  - `src/media_lab/cli.py`
  - `tests/test_upscale.py` (nou)
- **Teste**: `tests/test_upscale.py` (happy path mockat prin `MlRunner`, validare scalare dimensiuni 2x, păstrare integritate canal alpha, cazuri de eroare: fișier inexistent, mod invalid).
- **Depinde de**: `src/media_lab/ml_runner.py` (deja existent).

---

### Pasul 2 — `subject-ground` (Foot pin, zoom-normalisation, contact shadow, canvas placement) — GATA (1210686)
- **Ce se adaugă**:
  - `src/media_lab/grounding.py`: modul pur de calcul geometric și grafic:
    - Detecție puncte de contact picioare (pe baza pragului de alpha, centroid X al benzii inferioare).
    - Median smoothing pe Y pentru a preveni flicker-ul și plutirea siluetei în mers (foot-lock).
    - Zoom-normalisation: scalare subtilă per frame raportată la înălțimea mediană a siluetei (elimină variația de distanță a camerei).
    - Contact shadow multi-layer: ambient occlusion dens la baza tălpii + elipsă cu falloff difuz orientată după vectorul solar (`sun_dir`).
  - `src/media_lab/recipes/subject_ground.py`: rețetă `ground_subject` care preia cadrele decupate, aplică grounding-ul și le plasează pe canvas-ul cerut (ex: 2160x3840) gata pentru `compose-spec`.
  - `src/media_lab/cli.py`: adăugare subcomandă `media-lab ground`.
- **Fișiere atinse**:
  - `src/media_lab/grounding.py` (nou)
  - `src/media_lab/recipes/subject_ground.py` (nou)
  - `src/media_lab/cli.py`
  - `tests/test_subject_ground.py` (nou)
- **Teste**: `tests/test_subject_ground.py` (calcul punct de sprijin pe siluete sintetice, generare corectă a celor 3 layere de umbră, plasare pe canvas cu offset dx/ground_y, tratare frame complet transparent).
- **Depinde de**: numpy, PIL.

---

### Pasul 3 — `scale-from-plate` (Calcul scară și poziție sol din referință) — GATA (6716243)
- **Ce se adaugă**:
  - `src/media_lab/scale_plate.py`: funcții pure pentru calculul scării subiectului și al liniei de sol pe baza unei persoane de referință din background plate sau a unei înălțimi țintă în pixeli la coordonata Y dată.
  - `src/media_lab/recipes/scale_plate.py` + CLI `media-lab scale-plate`: calculează parametrii optimi de scalare și poziționare pentru `ground_subject`.
- **Fișiere atinse**:
  - `src/media_lab/scale_plate.py` (nou)
  - `src/media_lab/recipes/scale_plate.py` (nou)
  - `src/media_lab/cli.py`
  - `tests/test_scale_plate.py` (nou)
- **Teste**: `tests/test_scale_plate.py` (formule matematice de scalare raportate la plate, validare praguri, erori pe dimensiuni nule sau negative).
- **Depinde de**: Pasul 2.

---

### Pasul 4 — `colour-match` (Transfer statistic Lab + relight scenic directional/bounce) — GATA (26e7e4c)
- **Ce se adaugă**:
  - `src/media_lab/colour_transfer.py`: modul pur de procesare coloristică:
    - Transfer statistic Reinhard în spațiul Lab (aliniere medie și deviație standard între regiunea de fundal și subiect).
    - Gradient direcțional key/fill (`key_dir`, `key_amt`).
    - Ambient color tint + ajustare gamma.
    - Ground bounce: nuanțare și lift pe partea inferioară a corpului (~ultimii 35%) pentru integrarea reflexiei solului.
    - Ajustări saturație și contrast menținând neafectat canalul alpha.
  - `src/media_lab/recipes/colour_match.py`: rețetă `colour_match` cu suport pentru procesare batch de cadre RGBA.
  - `src/media_lab/cli.py`: adăugare subcomandă `media-lab colour-match`.
- **Fișiere atinse**:
  - `src/media_lab/colour_transfer.py` (nou)
  - `src/media_lab/recipes/colour_match.py` (nou)
  - `src/media_lab/cli.py`
  - `tests/test_colour_match.py` (nou)
- **Teste**: `tests/test_colour_match.py` (integritate canal alpha, corectitudine transfer Lab pe culori sintetice, comportament la saturație 0, cazuri de eroare pe imagini corupte sau fără canal alpha).
- **Depinde de**: numpy, PIL.

---

### Pasul 5 — Refactorizarea runner-ului `punto` (eliminare scripturi legacy & staging improvizat) — GATA (3132c0b)
- **Ce se adaugă**:
  - Modificare `src/media_lab/recipes/punto_v23.py`:
    - Înlocuirea apelurilor către `docs/video-agent/pipeline/upscale_realesrgan.py` cu `recipes.upscale`.
    - Înlocuirea apelurilor către `docs/video-agent/pipeline/place_composite.py` cu `recipes.colour_match` urmat de `recipes.subject_ground`.
    - Eliminarea mutării manuale a folderului `work/punto-edit/isnet/up/ -> work/punto-edit/rvm_up/`.
    - Păstrarea opțiunii `--proxy` (fast-path).
- **Fișiere atinse**:
  - `src/media_lab/recipes/punto_v23.py`
  - `tests/test_punto_v23.py`
- **Teste**: adaptarea `tests/test_punto_v23.py` pentru a valida noul flux curat și complet integrat.
- **Depinde de**: Pașii 1, 2, 4.

---

### Pasul 6 — Verificare completă, documentație și sincronizare — GATA
- **Ce se adaugă**:
  - Actualizare `README.md` (noile comenzi CLI, exemple de utilizare).
  - Actualizare `DECISIONS.md` cu deciziile luate.
  - Arhivarea notată a scripturilor din `docs/video-agent/pipeline/` ca superseded de noile comenzi native.
  - Rularea tuturor verificărilor de calitate: `make lint` (`ruff`), `make typecheck` (`mypy` strict), `make test` (toată suita `pytest` verde cu coverage ≥ 80%).
- **Fișiere atinse**:
  - `README.md`
  - `DECISIONS.md`
  - `PLAN.md` (bifare pași finalizați)
  - `docs/video-agent/README.md`
- **Depinde de**: Pașii 1–5.

---

# FAZA 3 — Extensii Audio/Video/Foto & Agent Prompt-to-Edit

Completarea extensiilor native rămase din planul inițial (E1 subtitles, E2 stems, E4 photo)
și construirea nucleului pentru agentul autonom media-lab (inspecție media profundă și
orchestrare declarativă prompt-to-edit).

## Pași de implementare

### Pasul 1 — `stems` (Demucs audio separation & speech cleaning) — GATA
- **Ce se adaugă**:
  - `src/media_lab/ml/demucs_infer.py`: driver ML dedicat rulat prin `ml_runner` ca subprocess; suportă `--model` (`htdemucs`), `--two-stems` (`vocals`), selecție device auto/mps/cpu; extrage pista audio dacă inputul e video.
  - `src/media_lab/recipes/stems.py`: rețetă tipizată `separate_stems`:
    - separare piese în directorul țintă (`vocals.wav`, `no_vocals.wav`, sau 4-stems: `drums`, `bass`, `other`, `vocals`).
    - mod `--clean-speech`: înlocuiește pista audio a unui fișier video cu vocea izolată curățată de zgomot.
  - `src/media_lab/cli.py`: adăugare subcomandă `media-lab stems`.
  - `tests/test_stems.py`: teste unitare și de integrare cu mock-uri de subprocess, verificări de formate, force guard și erori.
- **Depinde de**: `demucs`, `ml_runner`.

---

### Pasul 2 — `subtitles` (Whisper speech-to-text + social subtitle styling/burning) — GATA
- **Ce se adaugă**:
  - `src/media_lab/ml/whisper_infer.py`: driver ML pentru Whisper local (`openai-whisper`), exportă segmente JSON cu timestamp-uri și cuvinte.
  - `src/media_lab/subtitles.py`: generator avansat de fișiere ASS / SRT cu preseturi social media:
    - `tiktok`: text centrat, bold, galben/alb, contur negru puternic, cuvânt activ evidențiat.
    - `clean`: text minimalist jos, alb cu umbră discretă.
    - `box`: fundal translucid rotunjit sub text pentru lizibilitate maximă.
  - `src/media_lab/recipes/subtitles.py`: rețetă `generate_subtitles` care produce fișierul `.ass`/`.srt` sau îl arde direct pe video prin `ffmpeg -vf ass=...`.
  - `src/media_lab/cli.py`: adăugare subcomandă `media-lab subtitles`.
  - `tests/test_subtitles.py`: teste generare ASS, formatare culori/timpi, ardere pe clip sintetic.
- **Depinde de**: `whisper`, `ffmpeg` (libass).

---

### Pasul 3 — `photo` (Editare foto, carusele & schimbare fundal) — GATA
- **Ce se adaugă**:
  - `src/media_lab/recipes/photo.py`: rețetă tipizată pentru editare de imagini statice:
    - decupare subiect (via `cutout` u2net).
    - înlocuire fundal cu scalare inteligentă (`cover` / `contain`) și ancorare la sol.
    - transfer de culoare Reinhard Lab pentru integrare lumină fundal.
    - preseturi de aspect ratio social media: `1:1` (feed Instagram), `4:5` (portret IG), `9:16` (Story), `16:9` (banner/YouTube).
    - filtru / grading + claritate (unsharp mask).
    - mod `--batch` pentru procesare foldere întregi de fotografii.
  - `src/media_lab/cli.py`: adăugare subcomandă `media-lab photo`.
  - `tests/test_photo.py`: teste pentru decupare, reîncadrare, transfer culoare, loturi de poze.
- **Depinde de**: PIL, numpy, `cutout`, `colour_transfer`.

---

### Pasul 4 — `inspect` & `edit` (Ochii agentului & Orchestratorul declarativ) — GATA
- **Ce se adaugă**:
  - `src/media_lab/inspect.py`: analizor profund care returnează JSON structurat despre fișiere audio/video/foto:
    - Video: durată, rezoluție, fps, bitrate, luminozitate medie, paletă culori dominante (RGB/Lab).
    - Audio: LUFS integrat, RMS peak, detectare vorbire / silențiu, raport semnal/zgomot estimat.
    - Persoană: detectare prezență siluetă umană, bounding box, raport înălțime/canvas.
  - `src/media_lab/edit_spec.py` + `src/media_lab/recipes/edit.py`: executor declarativ `edit-spec`:
    - Permite unui agent AI să definească într-un fișier YAML sau prin flag-uri operațiunile dorite (ex: curăță audio, reîncadrează la 9:16, aplică look cald, adaugă muzică de fundal ducked, generează subtitrări TikTok) și execută totul optimizat într-o singură trecere.
  - `src/media_lab/cli.py`: subcomenzi `media-lab inspect` și `media-lab edit`.
  - `tests/test_inspect.py` și `tests/test_edit.py`.
- **Depinde de**: toate rețetele anterioare.

---

### Pasul 5 — Code Review amplu, întărire securitate & documentație finală — GATA
- **Ce se adaugă**:
  - Revizuire a întregului codebase cu agenți specializați.
  - Verificare riguroasă: `make check` (zero linter warnings, zero mypy errors, 100% teste verzi cu coverage ≥ 80%).
  - Actualizare completă `README.md`, `DECISIONS.md`, `PLAN.md`.
  - Push pe noul branch git.

---

# FAZA 4 — Agentul Autonom Omnichannel (Sunet Studio, Video Pacing, Foto & MCP)

Transformarea `media-lab` într-un agent complet capabil să execute orice editare foto, video și audio din prompturi în limbaj natural sau prin protocolul deschis MCP (Model Context Protocol).

## Pași de implementare

### Pasul 1 — Zona 1: Sunet & Audio Studio (`audio_enhance`, `silence_trim`, `sfx`) — GATA
- **Ce se face**:
  - `src/media_lab/recipes/audio_enhance.py`: Mastering vocal de studio complet via ffmpeg filtergraph (high-pass 80Hz, EQ parametric pentru prezență și căldură vocală, de-esser sibilanțe 6.5kHz, compresor dinamic broadcast, noise gate/denoise).
  - `src/media_lab/recipes/silence_trim.py`: Tăiere automată a pauzelor moarte pe baza detecției de silențiu cu crossfade fin și tăieturi video sincronizate (jump-cut).
  - `src/media_lab/recipes/sfx.py`: Sound effects engine (efecte sonore whoosh, pop, ding, riser, impact sincronizate la tăieturi, subtitrări sau momente cheie).
  - CLI: `media-lab audio-enhance`, `media-lab cut-silence`, `media-lab sfx`.
  - Teste unitare și de integrare.

---

### Pasul 2 — Zona 2: Video & Pacing (`smart_reframe`, `punch_zoom`, `broll`) — GATA
- **Ce se face**:
  - `src/media_lab/recipes/smart_reframe.py`: Reîncadrare inteligentă 9:16 cu face tracking și panning fluid cinematic.
  - `src/media_lab/recipes/punch_zoom.py`: Punch-in zooms dinamice (1.1x–1.15x) pe punctele cheie pentru retenție video.
  - `src/media_lab/recipes/broll.py`: Inserare automată de cadre B-Roll (imagine/video cutaway) cu pista audio principală păstrată neîntreruptă (L-cut/J-cut).
  - CLI: `media-lab smart-reframe`, `media-lab punch-zoom`, `media-lab broll`.
  - Teste unitare și de integrare.

---

### Pasul 3 — Zona 3: Foto & Grafică (`typography`, `inpainting`, `face_retouch`)
- **Ce se face**:
  - `src/media_lab/recipes/typography.py`: Motor de text & titluri grafice peste imagini/video (drop shadow, stroke, capsule colorate, împachetare automată).
  - `src/media_lab/recipes/inpainting.py`: Ștergere obiecte nedorite și reconstrucție fundal (inpainting local).
  - `src/media_lab/recipes/face_retouch.py`: Retușare portret: netezire discretă a tenului și simulare adâncime de câmp (defocus blur pe fundal cu separare de mască).
  - CLI: `media-lab text-overlay`, `media-lab inpaint`, `media-lab retouch`.
  - Teste unitare și de integrare.

---

### Pasul 4 — Zona 4: Prompt-to-Edit & MCP Server (`prompt_agent`, `mcp_server`, `edit_spec`)
- **Ce se face**:
  - `src/media_lab/prompt_agent.py`: Interpret inteligent de prompturi în limbaj natural (română și engleză) care traduce cererea utilizatorului în `EditSpec`.
  - Extindere `src/media_lab/edit_spec.py` și `src/media_lab/recipes/edit.py` pentru noile capabilități.
  - `src/media_lab/mcp_server.py`: Server nativ Model Context Protocol (MCP) prin stdio pentru integrare directă cu Antigravity, Claude, Cursor.
  - CLI: `media-lab prompt` și `media-lab mcp`.
  - Teste complete.

---

### Pasul 5 — Verificare completă, Code Review, Documentație și Push
- Verificare completă: `make check` (0 linter errors, strict mypy 0 issues, pytest acoperire $\ge 80\%$).
- Documentare în `README.md` și `DECISIONS.md`.
- Sincronizare și push pe git.


