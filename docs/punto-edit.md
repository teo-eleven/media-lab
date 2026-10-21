# Plan aprobat — editare videoclip punto.mp4

Status: APROBAT, GATA DE EXECUȚIE. Nicio randare încă rulată. Citește tot
înainte să continui — conversația anterioară care a construit acest plan e
închisă, tot ce contează e scris aici.

Sursa originală: `/Users/teodorfotciuc/Downloads/videoclip punto.mp4`
(NU se atinge niciodată). Copie de lucru deja făcută:
`~/dev/media-lab/in/punto-source.mp4`.

Protocolul de sesiune video (regulile 1-8, furnizate de Teo) se aplică
integral — vezi rezumatul deciziilor mai jos, care înlocuiește nevoia de
a mai pune aceleași întrebări.

---

## Inventarul sursei (deja făcut, nu-l repeta)

| | |
|---|---|
| Container | MOV/MP4 |
| Video | H.264 Baseline, level 3.1, 8-bit, 4:2:0 |
| Rezoluție | 720×1280, verticală, SAR/DAR nespecificate |
| Framerate | 30 fps CONSTANT (confirmat) |
| Culoare | BT.709, progresiv |
| Bitrate video | ~1.87 Mbps |
| GOP | 7 I, 186 P, zero B-frames |
| Audio | AAC-LC, 44.1kHz, stereo, ~63kbps, -25.6 LUFS integrat |
| Durată | 6.433333s exact, 193 cadre video, 279 cadre audio |
| Rotație | fără |

**Ce limitează calitatea: sursa**, categoric. H.264 Baseline, fără B-frames,
fără CABAC. Niciun export nu recuperează detaliu absent din sursă.

---

## Deciziile luate (3 runde de întrebări, toate rezolvate)

1. **Fundaluri**: 6 locații, schimbate la fiecare ~1 secundă, distribuite
   UNIFORM (6.433333s ÷ 6 = 1.072222s fiecare, nu exact 1s + rest).
2. **Corectare corp**: SĂRITĂ complet azi (dorința #3 din cererea inițială).
   Nu există unealtă instalată pentru asta.
3. **Audio**: eliminat COMPLET — fără pistă audio deloc în livrarea finală.
4. **Calitate**: țintă 2160×3840 (portret, NU 3840×2160 landscape — sursa e
   verticală, am întrebat explicit, s-a confirmat portret), 60fps prin
   interpolare de cadre. FĂRĂ upscale AI (kino video-ai-upscale e stricat,
   vezi bug-uri de mai jos) — doar scalare Lanczos în ffmpeg.
5. **Livrare**: un singur fișier (nu master+delivery separat, e un test).
6. **Body correction / AI-look**: risc semnalat explicit lui Teo — subiectul
   se mișcă continuu într-un singur parc static; schimbarea fundalului la
   fiecare secundă va arăta probabil ca un compositing, indiferent de
   execuție. Teo a acceptat riscul (a zis explicit "e doar un test").

### Reperele exacte pentru cele 6 locații (confirmate de Teo)
1. Times Square (dat direct)
2. Tower Bridge, Londra (NU podul propriu-zis "London Bridge")
3. Shanghai — The Bund / skyline (NU Marele Zid, NU Beijing)
4. Shibuya Crossing, Tokyo (NU Tokyo Tower, NU Shinjuku)
5. Arcul de Triumf, București (dat direct) — POZĂ STATICĂ, nu video (vezi
   mai jos de ce)
6. Deșert — dune de nisip stil Sahara (presupunere neconfirmată explicit,
   dar necontrazisă)

---

## Sursele descărcate (deja pe disc, verificate cu curl, licență Pixabay
## Content License — gratuit, fără atribuire, uz comercial permis)

Toate în `~/dev/media-lab/in/backgrounds/`:

| Fișier | Locație | Rezoluție | FPS | Durată |
|---|---|---|---|---|
| `01-times-square.mp4` | Times Square | 2160×3840 (potrivire EXACTĂ cu canvas-ul țintă) | 30 | 17.1s |
| `02-tower-bridge.mp4` | Tower Bridge, Londra | 3840×2160 | 25 | 25.6s |
| `03-shanghai-bund.mp4` | Shanghai, The Bund | 2160×1440 | 30 | 12.2s |
| `04-shibuya.mp4` | Shibuya Crossing | 1920×1080 | 30 | 20.3s |
| `05-bucharest-arch.jpg` | Arcul de Triumf | 1280×891 (JOASĂ — vezi nota) | — | (poză) |
| `06-desert.mp4` | Deșert | 3840×2160 | 29.97 | 20.3s |

**De ce poză, nu video, la Arcul de Triumf**: căutat exhaustiv pe Pixabay,
Pexels, Mixkit, Coverr — niciun clip video liber-licențiat cu monumentul
specific. Videvo are unul, dar blochează accesul programatic (403). Doar
poza găsită, la 1280px — sub ținta de 4K, va fi vizibil mai slabă decât
celelalte 5 segmente după scalare. Teo a acceptat asta explicit.

**Sursele URL originale** (pentru re-descărcare dacă e nevoie, sau atribuire):
- Times Square: https://cdn.pixabay.com/video/2022/06/24/121994-724732238_large.mp4
- Tower Bridge: https://cdn.pixabay.com/video/2022/12/20/143615-784129606_large.mp4
- Shanghai: https://cdn.pixabay.com/video/2024/05/18/212404_large.mp4
- Shibuya: https://cdn.pixabay.com/video/2024/04/11/207611_large.mp4
- Arcul de Triumf (poză): https://cdn.pixabay.com/photo/2017/08/26/17/17/bucharest-2683691_1280.jpg
- Deșert: https://cdn.pixabay.com/video/2025/06/09/284566_large.mp4

---

## Bug-uri upstream descoperite azi (NU le mai investiga, sunt confirmate)

1. **`kino video-ai-upscale` complet stricat pe ffmpeg 9.0.** Folosește intern
   flagul `-vsync 0`, eliminat din ffmpeg (înlocuit cu `-fps_mode`). Eroare
   reprodusă manual: `Unrecognized option 'vsync'. Error splitting the
   argument list: Option not found`. Testat cu clipuri sintetice mici și
   medii, cu și fără alpha, cu model implicit și `--model realesrgan`
   explicit — eșuează identic de fiecare dată.
   **În plus**: chiar dacă ar merge, codul sursă
   (`kinocut/ai_engine/upscale.py`, funcția `_ai_upscale_opencv`) arată că
   NU folosește Real-ESRGAN, ci cade mereu pe FSRCNN (OpenCV DNN Super
   Resolution) — un model mult mai slab, indiferent de `--model` cerut.
   **Decizie**: renunțat la upscale AI. Doar scalare Lanczos în ffmpeg.

2. **`kino composite-layers` produce mereu H.264 la ieșire**, fără control
   de calitate (fără CRF, fără opțiune de codec), indiferent de extensia
   fișierului de output. Testat cu `.mov` — tot H.264 iese.
   **Decizie**: NU se folosește `composite-layers` pentru acest job. Se
   construiește compositing-ul direct în ffmpeg (vezi planul tehnic).

3. **(Cunoscut din sesiuni anterioare, relevant aici)**: `composite-layers`
   plafonează la 25fps intern dar etichetează cu fps-ul cerut. Irelevant
   pentru acest plan fiindcă nu mai folosim `composite-layers` deloc.

---

## PLANUL TEHNIC APROBAT — de executat exact așa

### Etapa 1 — Cutout persoană
```
kino hyperframes-remove-background in/punto-source.mp4 \
  -o work/punto-edit/01-cutout.mov \
  --model u2net_human_seg --quality best
```
Așteptat: ProRes 4444 cu alpha, 720×1280, 30fps, 6.433s.
Estimare: 1-2 minute (quality `best` mai lent decât `fast`, care era
~125ms/cadru pe teste anterioare).

### Etapa 2 — Pregătire 6 fundaluri (ÎN PARALEL, independente)

Segmentare exactă (6.433333 ÷ 6 = 1.072222s fiecare):

| # | Interval | Sursă | Fișier ieșire |
|---|---|---|---|
| 1 | 0.000000 – 1.072222 | `01-times-square.mp4` | `work/punto-edit/02-bg-01.mov` |
| 2 | 1.072222 – 2.144444 | `02-tower-bridge.mp4` | `work/punto-edit/02-bg-02.mov` |
| 3 | 2.144444 – 3.216667 | `03-shanghai-bund.mp4` | `work/punto-edit/02-bg-03.mov` |
| 4 | 3.216667 – 4.288889 | `04-shibuya.mp4` | `work/punto-edit/02-bg-04.mov` |
| 5 | 4.288889 – 5.361111 | `05-bucharest-arch.jpg` | `work/punto-edit/02-bg-05.mov` |
| 6 | 5.361111 – 6.433333 | `06-desert.mp4` | `work/punto-edit/02-bg-06.mov` |

Pentru fiecare sursă VIDEO: alege un punct de start în sursă (evită primele
1-2s, adesea fade-in/logo), taie exact `1.072222`s de acolo, scalează+
cadrează „cover" la 2160×3840 cu Lanczos, encodează ProRes 422 HQ la 30fps.

Filtru cover-fit (sursă landscape → canvas portret):
`scale=-2:3840:flags=lanczos,crop=2160:3840`
(scalează după înălțime la 3840, apoi decupează centrat la lățime 2160)

Times Square e deja 2160×3840 exact — doar trim, fără scale/crop.

Pentru poza Arcul de Triumf (1280×891, landscape): `-loop 1 -t 1.072222`
plus același filtru cover-fit. NOTĂ: scalare ~4.3× de la 891 la 3840 —
segmentul va fi vizibil moale, cum s-a discutat.

Comandă model (ajustează sursa/offset per fundal):
```
ffmpeg -y -ss <offset> -i in/backgrounds/NN-nume.mp4 -t 1.072222 \
  -vf "scale=-2:3840:flags=lanczos,crop=2160:3840,fps=30" \
  -c:v prores_ks -profile:v 3 -pix_fmt yuv422p10le \
  work/punto-edit/02-bg-0N.mov
```

### Etapa 3a — Concatenare fundaluri (stream-copy, zero pierdere)
```
# creezi work/punto-edit/concat-list.txt cu cele 6 fisiere in ordine
ffmpeg -y -f concat -safe 0 -i work/punto-edit/concat-list.txt \
  -c copy work/punto-edit/03-background-track.mov
```
Verifică după: durata trebuie să fie 6.433333s ±0.05s.

### Etapa 3b — Compositing (subiect peste banda de fundal, O SINGURĂ trecere)
```
ffmpeg -y -i work/punto-edit/03-background-track.mov \
  -i work/punto-edit/01-cutout.mov \
  -filter_complex "[1:v]scale=2160:3840[subj];[0:v][subj]overlay=0:0:format=auto" \
  -c:v prores_ks -profile:v 3 -pix_fmt yuv422p10le \
  work/punto-edit/04-composed.mov
```
Scalare subiect: 720×1280 × 3 = 2160×3840 — potrivire matematică exactă,
fără crop necesar pe subiect.

**STOP AICI pentru verificare vizuală** (regula 7 din protocol). Extrage
câte un cadru din mijlocul fiecărui segment (la ~0.5s, ~1.6s, ~2.7s, ~3.75s,
~4.8s, ~5.9s), arată-le lui Teo, așteaptă confirmare explicită înainte de
Etapa 4.

### Etapa 4 — Interpolare 30→60fps + encodare finală (O SINGURĂ trecere,
### cea mai costisitoare, NU pornești fără aprobare pe cadrele de test)
```
ffmpeg -y -i work/punto-edit/04-composed.mov \
  -vf "minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:vsbmd=1:mb_size=8" \
  -an \
  -c:v libx264 -profile:v high -preset slow -crf 17 \
  -pix_fmt yuv420p -movflags +faststart \
  out/punto-final_2160x3840_60fps_h264-crf17.mp4
```
Alternativ H.265 10-bit dacă Teo preferă fișier mai mic (întreabă dacă nu
s-a decis explicit — NU s-a decis explicit, doar propus H.264 ca implicit).

Estimare: 3-8 minute, incertă (minterpolate pe 3840px lățime + preset slow
sunt ambele grele). Fără estimare mai precisă până nu rulează efectiv.

### Control de calitate obligatoriu după Etapa 4 (regula 6 din protocol)
- ffprobe pe ieșire: rezoluție 2160×3840, 60fps, H.264 High, yuv420p
- durată = 6.433333s ±0.1s (interpolarea nu trebuie să schimbe durata)
- FĂRĂ pistă audio (verifică explicit că nu există stream audio)
- extrage cadre la început/mijloc/sfârșit, verifică artefacte de
  interpolare (ghosting, warping) mai ales la cele 5 tranziții între
  segmente de fundal
- verifică cele 6 tranziții de fundal — pot avea 1 cadru interpolat ciudat
  la fiecare graniță (minterpolate poate face artefacte pe schimbare de
  scenă); e un risc cunoscut, nu o eroare de execuție dacă apare minor

---

## Nerezolvat / de întrebat dacă apare

- Codec final: H.264 propus ca implicit, NU confirmat explicit de Teo.
  Întreabă dacă vrea H.265 în schimb, înainte de Etapa 4.
- Offset-urile exacte de start în fiecare sursă video (unde anume tai cele
  1.072s) nu au fost alese încă — alege puncte fără fade-in/watermark
  vizibil, verifică vizual înainte de a le fixa în Etapa 2.

## Ce NU face acest plan (explicit, per regula 1 a protocolului)
- Nicio corecție de culoare, stabilizare, muzică, text, tranziții
  suplimentare — doar ce s-a cerut.
- Niciun fișier master separat.
- Nicio corectare a defectelor corporale.

---

## Progres execuție (bifat pe măsură ce rulează)

- [x] Etapa 1 — Cutout persoană (`work/punto-edit/01-cutout.mov`)
- [x] Etapa 2 — 6 fundaluri pregătite (offset-uri: TS 3.0s, Tower 5.0s, Shanghai 3.0s, Shibuya 4.0s, Arc still, Desert 4.0s) (`work/punto-edit/02-bg-0N.mov`)
- [x] Etapa 3a — Concatenare fundaluri (192 cadre / 6.4s) (`work/punto-edit/03-background-track.mov`)
- [x] Etapa 3b — Compositing (193 cadre / 6.433333s) (`work/punto-edit/04-composed.mov`)
- [x] STOP verificare vizuală — trecut; Teo a cerut și placarda (scope nou)
- [x] Etapa 4 — Interpolare 30→60fps + H.264 CRF 17 (me=ds rapid; 8:30)
- [x] QC final — ffprobe OK (2160x3840, 60fps, H.264 High, yuv420p, fara audio, 6.383s), verify-render ok/0 probleme

### Note de execuție
- Se lucrează în `~/dev/media-lab` (proiectul media-lab existent). Intermediari în `work/punto-edit/`, livrare în `out/`.
- `ffmpeg`/`ffprobe`: binarele statice din `./bin/` (nu sunt pe PATH — se rulează cu `PATH="$PWD/bin:$PATH"`).
- `kino`: din `.venv` a proiectului.

### Devieri față de planul original (cerute de Teo pe parcurs)
- **Placardă reintrodusă.** `u2net_human_seg` taie panoul ținut în mâini. Ales: PNG recreat fidel (whiteboard alb, ramă gri, scris handwritten albastru — font Bradley Hand), text `M-A TRIMIS / ȘEFUL / SĂ ADUC / CLIENȚI / — / PUNTO`, poziționat pe traiectorie măsurată automat (centroid de alb, 26 puncte). Fișier: `work/punto-edit/placard/placard.png`. Compositing în `04b-composed-placard.mov` (Etapa 3c nouă). Tracking OK pe 5/6 segmente; la ~2–2.5s panoul plutește ~150px peste mâini.
- **Alfa cutout curățat** (`01-cutout-clean.mov`): erosion×2 + boxblur pe canalul alfa, ca să scadă haloul întunecat de pe brațe pe fundaluri deschise.
- **Segment 4 (Shibuya) re-tăiat** la offset 8.0s (față de 4.0s) — sursa e integral macro cu profunzime mică pe picioare/trecere, nu are cadru larg; 8.0s se citește cel mai bine.
- **Fix plan:** opțiunea `minterpolate` e `vsbmc`, nu `vsbmd` (typo în planul aprobat).

### Rezultat final
`out/punto-final_2160x3840_60fps_h264-crf17.mp4` — 30.5 MB, 383 cadre.
QC hard: trecut tot. Durata 6.383s (3 cadre sub sursa, minterpolate taie coada; in ±0.1s din plan).

Probleme cunoscute (test):
1. Placarda se detaseaza de maini in ~2.0-2.7s (varful saltului) si la coada ~6.2s+.
2. Warp usor pe membre din interpolarea rapida (me=ds), mai vizibil pe Shibuya si ultima ~0.3s.
3. Fundal Shibuya moale — sursa e integral macro cu profunzime mica, fara cadru larg.
4. Tranzitiile intre fundaluri: curate, fara cadru-gunoi (scd a functionat).

### Rework v3 (feedback Teo: "zboară", "se vede că nu e acolo", "placarda deraiază")
- **Matte nou cu `rembg` / `isnet-general-use`** (venv izolat `work/punto-edit/.matte-venv`, model 179MB local, aprobat de Teo): scoate **persoana + placarda ca un singur matte**. Placarda nu mai e strat separat, e blocată de mâini din sursă. `01-cutout-isnet.mov`.
- **Redus la 3 fundaluri, toate cu sol**: Times Square, Arc de Triumf, Deșert. Tower Bridge + Shanghai **eliminate** — ambele surse sunt filmate de peste râu, apă pe tot cadrul, niciun mal; nu se putea „merge pe trotuar". Shibuya eliminat — sursa e macro blurat pe picioare, fără linie de sol utilizabilă. Segmente 2.144s.
- **Grounding per segment**: subiect scalat 1380×2453, poziționat pe linia solului măsurată (TS y3200, Arc y3400, Deșert y3300), umbră de contact (PNG eliptic, alfa 0.42), potrivire de culoare per fundal (eq + colorbalance).
- Compositing: `04-composed-v3.mov`. Interpolare + export peste el.

### Rezultat final v3
`out/punto-final_2160x3840_60fps_h264-crf17.mp4` — 26 MB, 384 cadre, 6.45s.
QC: H.264 High, 2160x3840, yuv420p, 60fps, fara audio. verify-render: ok / 0 probleme.
Imbunatatiri vs varianta respinsa: placarda parte din siluromata (nu mai deraiaza), subiect grounded (talpi pe strada/iarba/nisip), umbra de contact, culoare potrivita per fundal, 3 fundaluri toate cu sol.
Limitari ramase: tot vizibil compositing la inspectie apropiata (sursa 720p, placi statice, fara relighting AI); ghosting usor de interpolare pe Times Square ~0.3s; grounding aproximativ pe duna inclinata; 6.45s (~3 cadre pierdute la coada de minterpolate).

### Rework v4/v5 (feedback: "pare ca merge prin aer", cere grounding real)
- **Foot-lock**: `work/punto-edit/isnet/place.py` detecteaza per cadru pixelul opac cel mai de jos (talpa de sprijin), pre-pozitioneaza subiectul pe canvas 2160x3840 astfel incat talpa sa cada exact pe linia solului in fiecare cadru. Piciorul de sprijin sta pe loc, corpul se leagana deasupra -> mers real, nu plutire.
- **Umbra care urmareste talpile**: bakuita pe canvas sub persoana, in doua straturi (contact AO stramt + falloff moale).
- **Desert**: re-esantionat la offset 14s + decupaj jos (scale 4600 -> crop y760) pentru teren plat cu urme de pasi, in loc de panta abrupta.
- **Arc**: subiect mutat 150px stanga, pe iarba libera langa tufis.
- Scara redusa 1.92 -> 1.80; potrivire de culoare + luminozitate per segment.
- Compositing: `04-composed-v5.mov`. Final: `out/punto-final_2160x3840_60fps_h264-crf17.mp4` 26.7MB, 383 cadre, 6.383s. QC + verify-render: trecut.

### Rework v6 (feedback: vrea generativ + fara zoom)
- **IC-Light (relighting generativ) - INCERCAT, ESUAT pe acest hardware.** Instalat (diffusers + SD1.5 + iclight_sd15_fbc, ~7GB, venv `work/punto-edit/.gen-venv`). Nu ruleaza in timp utilizabil: fara GPU CUDA, limita MPS ~9GB, >10 min/cadru fara output. Abandonat.
- **Real-ESRGAN x2** (`work/punto-edit/gen/upscale.py`, `RealESRGAN_x2plus.pth`) pe cele 193 de cadre isnet -> 1440x2560, subiect vizibil mai clar. ~18 min. (shim `torchvision/transforms/functional_tensor.py` adaugat in .venv pentru basicsr).
- **Fara zoom**: `place2.py` normalizeaza inaltimea subiectului per cadru (clamp 0.90-1.11) -> nu mai "creste" spre camera.
- **Relight per scena (manual, scene-aware)**: gradient directional key/fill dupa directia soarelui, culoare ambientala, "bounce" de la sol (asfalt gri / iarba verde / nisip), umbra de contact offset dupa soare.
- Compositing: `04-composed-v6.mov`. Final: `out/punto-final_2160x3840_60fps_h264-crf17.mp4` 27MB, 383 cadre, 6.383s. QC + verify-render: trecut.

### v7 - pas de finisaj "produs" (feedback: "din topor", vrea pregatit de social media)
- **Tranzitii**: xfade dissolve 0.4s intre cele 3 locatii; subiectul ramane continuu (lumea se schimba in spatele ei).
- **Grade unitar** pe tot: eq (contrast/saturatie/gamma), colorbalance cald, s-curve filmica prin curves.
- **Atmosfera/diffusion**: copie blurata (sigma 42) screen-uita la 10% peste tot -> leaga fg de bg, ascunde marginea de decupaj.
- **Grain de film** (noise alls=8), **unsharp** subtil, **vigneta** (PI/4.4).
- **Drift de camera**: scale 1.05 + crop animat (sway sin + urcare lenta) -> nu mai pare lipit static.
- O singura trecere ffmpeg peste `04-composed-v7.mov`.

### v9 = LIVRARE FINALA
`out/punto-final_2160x3840_60fps_h264-crf17.mp4` - 88 MB, 383 cadre, 6.383s, H.264 High 2160x3840 60fps, fara audio. verify-render: ok.
Peste v6 (Real-ESRGAN + fara zoom + relight per scena + foot-lock + umbra tracking):
grade unitar cald/filmic, strat de atmosfera (blur screen 9%) care leaga fg de bg,
vigneta blanda (PI/5), grain temporal, drift subtil de camera (sway + urcare).
v7 (xfade dissolve) abandonat - minterpolate ghostuia pe tranzitie; taieturi seci in loc.
v8 abandonat - fade in/out chained a albit tot clipul.

### v10 = LIVRARE FINALA (feedback: fara speed/interpolare, fara miscare camera, doar fundaluri de la nivelul solului f2f)
`out/punto-final_2160x3840_30fps_h264-crf17.mp4` (+ copie pe ~/Desktop/punto-final.mp4) - 195 MB, 192 cadre, 6.433s, H.264 High 2160x3840, **30fps NATIV (fara minterpolate)**, fara audio. verify-render: ok.
- **Doar 2 fundaluri**, ambele filmate de la nivelul solului, camera fixa, fata in fata: Times Square (stradal) + Arcul de Triumf (peluza). Desert eliminat (drona/aerian), Shibuya (macro pe picioare), Tower Bridge + Shanghai (de peste rau) - toate eliminate.
- **Fara interpolare** -> miscare naturala, fara warp/"speed".
- **Fara miscare de camera** -> cadru complet fix (fara zoom, fara sway).
- Subiect: isnet + Real-ESRGAN x2, foot-lock pe minim glisant (piciorul de sprijin pinuit), no-zoom (inaltime constanta), centrat pe X (fara drift stanga-dreapta), scara mai mare (prim-plan), relight per scena, umbra de contact.
- Grade unitar usor + atmosfera (screen 8%) + grain temporal slab + vigneta blanda (PI/5.5). Fara efecte de miscare.
- Limitare ramasa: usoara "saltare" a corpului intre pasi pe unele cadre; tot vizibil compositing la inspectie apropiata (sursa 720p, placi statice, fara relighting AI).

### v11 = LIVRARE (feedback: goluri pe ea/pancarda cand merge; adauga desert f2f; sol in fiecare cadru)
- **Matte curatat** (`isnet/mattefix.py`): median temporal +-2 cadre (omoara flickerul + reface bucatile pe care matte-ul per-cadru le pierde izolat) + umplere gauri inchise + dilatare 3px (inchide golul de la jonctiunea mana-pancarda). Fara filtrare de componente (aia taia pancarda/rupea corpul).
- **Desert readaugat**, filmat de la sol / fata in fata: clip nou Pixabay (`cand-a.mp4` -> `desert-ground.jpg` cadru static), tonat mai putin portocaliu. 3 fundaluri, toate sol: TS (trecere) + Arc (peluza) + Desert (dune).
- Pastreaza tot din v10: 30fps nativ (fara interpolare), cadru fix (fara zoom/sway), no-zoom, centrat pe X, relight per scena, foot-lock, umbra.
- Limitare ramasa: usoara saltare a corpului intre pasi pe cateva cadre; placa de desert e neteda (posibil 3D render).

### v13 = LIVRARE (feedback: fundaluri animate cu oameni, viteza normala, zi normala; desert f2f de la sol)
3 fundaluri, toate video real-time la zi cu viata: Times Square (multime/trafic) + strada Broadway NYC `street-real.mp4` (trafic, autobuz scolar) + Sahara reala `sahara-real.mp4` (nisip cu urme, om care merge in fundal, cer). Shibuya-macro si desert-3D eliminate. Sahara stabilizata (vidstab) ca sa fie cadru fix. Pastreaza: 30fps nativ, fara interpolare, fara camera move, hard foot-pin, matte curatat (mattefix WIN3 + foot extension).
- v14: Times Square segment slowed ~6.2x (setpts+minterpolate) to natural speed - "totul zboara" fixed. Other 2 segments unchanged.
- v15: matte fills only small holes (< 16000 px); the enclosed gap between the raised arms stays transparent so the real background shows through, not a patch of the source park.
- v16: single coherent location (Times Square crossing), hyperlapse de-sped 6x with mi_mode=blend (no pedestrian warping), single-segment placement, softened subtle contact shadow, arm-gap-transparent matte.

### v17 = LIVRARE (agent-assisted: RVM matting + real-time portrait crowd)
- **Matte: RVM (Robust Video Matting, resnet50)** in loc de isnet+mattefix. Recurent -> stabil temporal, zero flicker/goluri, pastreaza placarda. 32s pentru 193 cadre pe MPS. `work/punto-edit/rvm/` -> Real-ESRGAN x2 -> `work/punto-edit/rvm_up/`.
- **Fundal: clip Pexels portret 1440x2560 REAL-TIME** cu multime pe strada comerciala (Istanbul), scalat 1.5x -> 2160x3840. Timp real => fara retiming => fara "oameni care apar/dispar". `in/backgrounds/crowd-portrait.mp4`.
- **Ocluzie de adancime**: randul din front al multimii (banda de jos a fundalului, top feather 170px) re-suprapus PESTE subiect -> oamenii din fata trec prin fata ei. Ea la scara multimii de la adancimea ei (base_scale 0.72, ground 3070).
- O singura locatie coerenta. 30fps nativ, fara interpolare, fara camera move. Grade + atmosfera + grain slab.
- Unelte folosite: agent general-purpose (research matting + stock), RVM, Real-ESRGAN, Pexels.
- v17e: stronger crowd integration - subject buried waist-down behind front-row crowd (feathered occlusion), muted+desaturated+softened to match the overcast street, scaled to crowd depth. Reads as one person in the throng holding the sign up.
- v18b: full-body front-row crowd figure, legs+feet visible on the cobbles, no occlusion strip, less desaturated (sat 0.90), lighter edge blur, foot shadow shadow_a=78.
- v20: full-body FRONT-ROW crowd member - same scale + same ground plane (feet y3720) as the nearest people, 3-layer contact shadow on the cobbles, tighter foot-pin (+-2 median), minimal occlusion (bottom 170px). Reads as walking WITH the crowd at their level, feet on the ground.
- v23: NYC bg (Wall St, NYPD cars, open asphalt foreground). RIGID foot pin (single fixed ground y for whole clip). Firm asphalt shadow. Person walks in the open foreground on the crosswalk, crowd+police behind.
