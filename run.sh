#!/usr/bin/env zsh
INPUT=""
OUTPUT=""
FPS_MODE="increase"
MODE="sequential"
PRECISION="fp16"
BATCH=4
FOURCC="mp4v"
GEN_FRAMES=1
# https://fourcc.org/codecs.php
# inference.py arg list
# IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# --config str
# --pretrained_path str
# --input_video str
# --frame0 str
# --frame1 str
# --output str
# --batch_size int
# --keep_fps
# --fps float
# --fourcc str
# --device str
# --precision str
# --seed int
# --strict_load

# ARG INPUT
while [[ $# -gt 0 ]]; do
  case "$1" in
    -i)
      INPUT="$2"
      shift 2 # Past argument only (flag)
      ;;
    -o)
      OUTPUT="$2"
      shift 2
      ;;
    --fps)
      FPS_MODE="$2"
      shift 2
      ;;
    -m|--mode)
      MODE="$2"
      shift 2
      ;;
    -p|--precision)
      PRECISION="$2"
      shift 2
      ;;
    -b|--batch-size)
      BATCH="$2"
      shift 2
      ;;
    --fourcc)
      FOURCC="$2"
      shift 2
      ;;
    -g|--gen-frames)
      GEN_FRAMES="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "  -i,   Set input"
      echo "  -o,  Set output        Set Output path: defaults to current working directory for parent. File type is overwritten by fourcc."
      echo "  -m, --mode,            Set mode: sequential, parallel (default: sequential)"
      echo "  --fps,                 Set fps method: increase (maintains video length), keep (maintains current fps) (default: increase)"
      echo "  -p, --precision,       Set precision: fp16 (VRAM 8GB), fp32 (VRAM 16GB), bf16 (VRAM 12GB) (default: fp16)"
      echo "  -b, --batch-size,      Set batch size: pick a reasonable number of batches for your gpu setup (default: 4)"
      echo "  --fourcc,              Set output video filetype with fourcc identifier: mp4v, avc1, FFV1, xvid, mjpg. see https://fourcc.org/codecs.php for more (default: 'mp4v')"
      echo "  -g, --gen-frames,      Set number of frames to generate between real frames: 1, 2 or 3 supported. (default: 1)"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done
SCRIPT_DIR="${0:A:h}"
echo $SCRIPT_DIR
source "$SCRIPT_DIR/.venv/bin/activate"
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/src/utils:${PYTHONPATH}"

FPS=$(ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate "$INPUT")
FPS=$((FPS))
case $FPS_MODE in
  increase)
    FPS=$(((GEN_FRAMES+1)*FPS))
    ;;
  keep)
    :
    ;;
  *)
    echo "Unknown FPS option: $FPS_MODE"
    exit 3
    ;;
esac

case $FOURCC in
  mp4*|avc1)
    OUTPUT_FILETYPE=".mp4"
    ;;
  ffv1|FFV1)
    FOURCC="FFV1"
    OUTPUT_FILETYPE=".mkv"
    ;;
  xvid|mjpg)
    OUTPUT_FILETYPE=".avi"
    ;;
  *)
    OUTPUT_FILETYPE=".avi"
    ;;
esac

if [[ -z "$OUTPUT" ]]; then
  F_BASE=$(basename $INPUT)
  F_NAME="${F_BASE%.*}"
  F_CONTAINER="${F_BASE:e}"
  F_DIR="${INPUT:h}"
  OUTPUT="$F_DIR/${F_NAME}_interpolated.$F_CONTAINER"
fi

OUTPUT="${OUTPUT%.*}$OUTPUT_FILETYPE"

case $MODE in
  parallel)
    python "$SCRIPT_DIR/inference.py" \
      --config "$SCRIPT_DIR/configs/eval_config.yaml" \
      --pretrained_path "$SCRIPT_DIR/ckpts/speed.pt" \
      --input_video "$INPUT" \
      --output "$OUTPUT" \
      --video_mode parallel \
      --batch_size $BATCH \
      --precision $PRECISION \
      --fps $FPS \
      --fourcc "$FOURCC" \
      --gen_frames $GEN_FRAMES
      ;;
  sequential)
    python "$SCRIPT_DIR/inference.py" \
      --config "$SCRIPT_DIR/configs/eval_config.yaml" \
      --pretrained_path "$SCRIPT_DIR/ckpts/speed.pt" \
      --input_video "$INPUT" \
      --output "$OUTPUT" \
      --video_mode sequential \
      --precision $PRECISION \
      --fps $FPS \
      --fourcc "$FOURCC" \
      --gen_frames $GEN_FRAMES
      ;;
    *)
      echo "Unknown option: $MODE"
      exit 2
      ;;
esac
deactivate
echo $OUTPUT
