# Running SPEED video interpolation model
I came across the SPEED video interpolation model:
https://github.com/bbldCVer/SPEED
https://huggingface.co/zhZ524/SPEED/tree/main

I was curious as I've been trying to make short video loops with speed changes.
# Requirements
- A `CUDA 12.4` capable system
	- RTX 2000 series and above (if my memory serves me)
	- Atleast 8GB of VRAM
- Tested on:
	- 9700k
	- 32GB DDR4
	- 2070 Super
	- Bazzite 44
# Installing
Installing something with specific dependencies is best done inside some sort of container.
I chose distrobox with podman as it's on bazzite and very user friendly.
## Make a cuda capable container
I'm not confident the distrobox command will work on all flavours of distro.
- If it doesn't work see if the flags are different 
- You might need to install nvidia-container-toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
	- For bazzite --nvidia enables cuda compatibility
		- I didn't have to change the OCI runtime to nvidia - ymmv
```bash
distrobox create --image docker.io/nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 --name speed --nvidia --additional-flags "-e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all"
```
You should automatically enter the container, if not: `distrobox enter speed`
```bash
sudo apt install git
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install uv
git clone git@github.com:PeterRobots/SPEED.git
cd SPEED
mkdir ckpts
cd ckpts
wget https://huggingface.co/zhZ524/SPEED/resolve/main/speed.pt
cd ..
```
## Making a python env with uv instead of conda
I am opinionated and do not like conda, so I used uv instead.
I could match cuda versions better, but honestly it worked and I didn't need to.
```bash
uv init
uv python pin 3.10
uv python install
uv add -r requirements-12-4.txt
```
# Running
You can run the model following the example in the SPEED repo, running inference.py arg.
#### inference.py has the follow arguments:
```sh
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
```
## A script for convienience
I made a bash script wrapping calls to inference.py.
- Activates the python venv
- Sets up env path required by SPEED
- (hopefully) Simplifies input arguments a bit
Download `run.sh` with your preferred method into the SPEED folder.
```bash
cd SPEED
wget https://github.com/PeterRobots/Running-SPEED-video-interpolater/blob/main/run.sh
sudo chmod +x run.sh
```

```bash
#!/usr/bin/env bash
INPUT=""
OUTPUT=""
FPS_MODE="double"
MODE="sequential"
PRECISION="fp16"
BATCH=4

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
    --help)
      echo "Usage: $0 [options]"
      echo "  -i,   Set input"
      echo "  -o,  Set output"
      echo "  -m, --mode,            Set mode: sequential, parallel (default: sequential)"
      echo "  --fps,                 Set fps method: double, keep (default: double)"
      echo "  -p, --precision,       Set precision: fp16 (VRAM 8GB), fp32 (VRAM 16GB), bf16 (VRAM 12GB) (default: fp16)"
      echo "  -b, --batch-size,      Set batch size: pick a reasonable number of batches for your gpu setup (default: 4)"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

source .venv/bin/activate
export PYTHONPATH="${PWD}:${PWD}/src/utils:${PYTHONPATH}"

FPS=$(ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate "$INPUT")
FPS=$((FPS))
case $FPS_MODE in
  double)
    FPS=$((2*FPS))
    ;;
  keep)
    :
    ;;
  *)
    echo "Unknown FPS option: $FPS_MODE"
    exit 3
    ;;
esac

case $MODE in
  parallel)
    python inference.py \
      --config configs/eval_config.yaml \
      --pretrained_path ckpts/speed.pt \
      --input_video "$INPUT" \
      --output "interpolation_outputs/$OUTPUT" \
      --video_mode parallel \
      --batch_size $BATCH \
      --precision $PRECISION \
      --fps $FPS
      ;;
  sequential)
    python inference.py \
      --config configs/eval_config.yaml \
      --pretrained_path ckpts/speed.pt \
      --input_video "$INPUT" \
      --output "interpolation_outputs/$OUTPUT" \
      --video_mode sequential \
      --precision $PRECISION \
      --fps $FPS
      ;;
    *)
      echo "Unknown option: $MODE"
      exit 2
      ;;
esac

deactivate
```
## Run the script
Below is an example running the script in sequential mode with double fps (retains video length).
```bash
./run.sh -i "path/input_file.mp4" -i "path/output_file.mp4"
```

# To do
1) Test extra frame generation options further
	- Test >3
	- Test parallel
 	- Validate the extra frames are interpolations and not copies
