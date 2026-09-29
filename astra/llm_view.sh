#!/bin/bash
# Build a "what the LLM saw" 2x2 video from per-turn frames: head | body map / left wrist | right wrist, 1 turn per 1.5 s.
cd "$1/frames" || exit 1
R=0.667
ffmpeg -y -loglevel error -framerate $R -i turn%02d_head.jpg -framerate $R -i turn%02d_body_map.jpg \
  -framerate $R -i turn%02d_left_wrist.jpg -framerate $R -i turn%02d_right_wrist.jpg -filter_complex \
"[0:v]pad=320:320:0:40:color=black,drawtext=text='head':x=8:y=8:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.5[a];\
[1:v]drawtext=text='body map':x=8:y=300:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.5[b];\
[2:v]pad=320:320:0:40:color=black,drawtext=text='left wrist':x=8:y=8:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.5[c];\
[3:v]pad=320:320:0:40:color=black,drawtext=text='right wrist':x=8:y=8:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.5[d];\
[a][b][c][d]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0,drawtext=text='turn %{eif\\:n\\:d}':x=w-110:y=8:fontsize=20:fontcolor=yellow:box=1:boxcolor=black@0.6[out]" \
  -map "[out]" -r 10 -c:v libx264 -pix_fmt yuv420p ../llm_view.mp4 && echo "wrote $1/llm_view.mp4"
