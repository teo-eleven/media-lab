#!/bin/bash
# Captured v23 compose run, kept as a record. NOT a clean-checkout script:
# it assumes the v23 `work/punto-edit/` layout (matte + upscaled + placed
# frames already present) and a specific background. compose-spec (Phase 1)
# replaces this. Paths are relative to the repo root.
set -euo pipefail
cd ~/dev/media-lab
export PATH="$PWD/bin:$PATH"
W=work/punto-edit/rework
D=work/punto-edit
PR=(-c:v prores_ks -profile:v 3 -pix_fmt yuv422p10le -an)

echo "### bg: crowd-portrait, REAL-TIME, 30fps, 2160x3840, 193 frames"
ffmpeg -y -hide_banner -loglevel error -ss 1.0 -i in/backgrounds/nyc-wallst.mp4 \
  -vf "fps=30,scale=2160:3840:flags=lanczos" -frames:v 193 "${PR[@]}" $W/cp-bg.mov
ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames,width,height -of csv=p=0 $W/cp-bg.mov

echo "### placement: RVM matte (upscaled), hard foot-pin, subtle shadow"
rm -rf work/punto-edit/isnet/placed && mkdir -p work/punto-edit/isnet/placed
# place2.py used in the v23 run is byte-identical to the tracked script below.
.venv/bin/python docs/video-agent/pipeline/place_composite.py
ffmpeg -y -hide_banner -loglevel error -framerate 30 -i work/punto-edit/isnet/placed/f-%04d.png -c:v prores_ks -profile:v 4 -pix_fmt yuva444p10le $D/subj-placed.mov

echo "### composite (native 30fps, no interp, no camera move)"
ffmpeg -y -hide_banner -loglevel error -i $W/cp-bg.mov -i $D/subj-placed.mov -filter_complex "\
[0:v]split[bgA][bgB];\
[bgB]crop=2160:130:0:3710,format=yuva444p,geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='if(lt(Y,70),255*Y/70,255)'[fg];\
[bgA][1:v]overlay=0:0:format=auto[mid];\
[mid][fg]overlay=0:3710:format=auto[out]" -map "[out]" "${PR[@]}" $D/04-composed-v17.mov

echo "### grade + atmosphere + mild grain"
ffmpeg -y -hide_banner -loglevel error -i $D/04-composed-v17.mov -filter_complex "[0:v]format=gbrp,split[base][atm];[atm]scale=iw/6:ih/6,gblur=sigma=7,scale=2160:3840:flags=bilinear,eq=brightness=0.03[atmb];[base][atmb]blend=all_mode=screen:all_opacity=0.07[lit];[lit]eq=contrast=1.04:saturation=1.07:brightness=0.008:gamma=0.99,colorbalance=rs=0.008:bs=-0.012:rm=0.008:bm=-0.008,curves=master='0/0.01 0.5/0.5 1/0.993',unsharp=5:5:0.24,vignette=PI/5.8,noise=alls=4:allf=t,format=yuv420p[out]" -map "[out]" -c:v libx264 -profile:v high -preset medium -crf 18 -movflags +faststart out/punto-final_2160x3840_30fps_h264-crf17.mp4

ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=width,height,r_frame_rate,nb_read_frames -show_entries format=duration -of default=noprint_wrappers=1 out/punto-final_2160x3840_30fps_h264-crf17.mp4
cp out/punto-final_2160x3840_30fps_h264-crf17.mp4 ~/Desktop/punto-final.mp4
for t in 0.4 1.3 2.2 3.1 4.0 4.9 5.6 6.1; do ffmpeg -y -hide_banner -loglevel error -ss $t -i out/punto-final_2160x3840_30fps_h264-crf17.mp4 -vframes 1 -q:v 2 $W/v17-$t.jpg; done
ffmpeg -y -hide_banner -loglevel error -i $W/v17-0.4.jpg -i $W/v17-1.3.jpg -i $W/v17-2.2.jpg -i $W/v17-3.1.jpg -i $W/v17-4.0.jpg -i $W/v17-4.9.jpg -i $W/v17-5.6.jpg -i $W/v17-6.1.jpg -filter_complex "[0]scale=300:533[a];[1]scale=300:533[b];[2]scale=300:533[c];[3]scale=300:533[d];[4]scale=300:533[e];[5]scale=300:533[f];[6]scale=300:533[g];[7]scale=300:533[h];[a][b][c][d]hstack=4[r1];[e][f][g][h]hstack=4[r2];[r1][r2]vstack=2" -frames:v 1 -q:v 3 $W/sheet-v17.jpg
for t in 1.0 3.0 5.0; do ffmpeg -y -hide_banner -loglevel error -ss $t -i out/punto-final_2160x3840_30fps_h264-crf17.mp4 -vframes 1 -vf "crop=1400:1100:400:2500,scale=380:299" $W/v17feet-$t.jpg; done
ffmpeg -y -hide_banner -loglevel error -i $W/v17feet-1.0.jpg -i $W/v17feet-3.0.jpg -i $W/v17feet-5.0.jpg -filter_complex hstack=3 -frames:v 1 -q:v 2 $W/v17feet.jpg
ls -la $W/sheet-v17.jpg $W/v17feet.jpg ~/Desktop/punto-final.mp4
