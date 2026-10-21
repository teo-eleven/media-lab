---
name: viral-short-director
description: >-
  Strategic director guide for high-retention short-form video creation (TikTok, Reels, Shorts).
  Covers hook structure, silence elimination, pacing rules, Hormozi-style kinetic typography,
  social UI safe zones, and audio loudness standards.
---

# Viral Short Director — Social Media Mastery Guide

This guide codifies the exact editing heuristics used by top creator workflows (Opus Clip, Descript, Captions.ai, CapCut) for creating viral vertical videos (9:16).

---

## 1. The 3-Second Hook Rule
- **First 0.0s–3.0s:** Must contain either an energetic spoken statement (punchline/controversy) or an immediate visual shift. Never start with "Hey guys", "So today", or dead silence.
- **Visual change frequency:** Every 2.5–3.5 seconds must feature a visual transition (punch-in zoom, b-roll overlay, text card, or camera pan) to prevent eye fatigue.

---

## 2. Audio Pacing & Silence Thresholds
- **Silence cutoff:** Pauses longer than **0.35s–0.45s** must be removed via `media-lab cut-silence --min-silence 0.4`.
- **Vocal tone:** Always apply broadcast mastering via `media-lab audio-enhance` (80Hz high-pass, presence EQ, de-esser 6.5kHz, compression).
- **Background Ducking:** Background music should sit at **-18 to -22 LUFS**, ducking down **-6dB to -10dB** whenever dialogue is detected (`media-lab music`).

---

## 3. Kinetic Subtitles & Typography Rules
- **Font & Weight:** Heavy sans-serif (Impact, Arial Black, Montserrat ExtraBold).
- **Length on screen:** Max 3–5 words at a time. Viewer attention drops if complete paragraphs are displayed.
- **Palette:** Bright Yellow (`#FFE600`), Vibrant Green (`#00FF66`), or White with a thick black outline (stroke 3-4px) and subtle drop shadow.
- **Placement (Safe Zones):**
  - **Avoid Bottom 20%:** Reserved for platform username, caption, and music ticker.
  - **Avoid Top 15%:** Reserved for search bar and sound controls.
  - **Avoid Right 15%:** Reserved for Like, Comment, Bookmark, Share icons.
  - **Ideal Y-Zone:** 45%–65% from the top (the middle-third of the screen).

---

## 4. Visual Retention Multipliers
- **Dynamic Punch-Zoom:** 1.12x–1.18x subtle punch-ins on emphatic words or sentence transitions (`media-lab punch-zoom`).
- **Animated Progress Bar:** A thin (4–8px) bar along the bottom or top that provides visual feedback of clip duration, boosting completion rates (`media-lab progress-bar`).
- **B-Roll Cutaways:** Insert contextually relevant 1.5s–3.0s visual inserts (`media-lab broll`) while keeping the speech track continuous.
