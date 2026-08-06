# Running SPEED video interpolation model

I came across the SPEED video interpolation model:
https://github.com/bbldCVer/SPEED
https://huggingface.co/zhZ524/SPEED/tree/main

I was curious as I've been trying to make short video loops with speed changes.

# Installing
The best way to install something like this with specific dependencies, is inside some sort of container.
I chose `distrobox` with `podman` as it's on Bazzite and very user friendly.
## Make a cuda capable container
I'm not confident the `distrobox` command will work on all flavours of distro.
- If it doesn't work see if the flags are different 
- You might need to install nvidia-container-toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
	- For Bazzite `--nvidia` enables cuda compatibility
		- I didn't have to change the OCI runtime to nvidia - ymmv
```bash
distrobox create --image docker.io/nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 --name speed --nvidia --additional-flags "-e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all"
sudo apt install git
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install uv
git clone https://github.com/bbldCVer/SPEED.git
cd SPEED
mkdir ckpts
cd ckpts
wget https://huggingface.co/zhZ524/SPEED/resolve/main/speed.pt
cd ..
```
## Making a python env with uv instead of conda
I do not like conda, so I used uv instead.
Could match cuda versions better, but honestly it worked and I didn't need to.
```bash
uv init
uv python pin 3.10
uv python install
uv add -r requirements-12-4.txt
```
# Running
## A script for convenience
I made a bash script wrapping calls to `inference.py`.
- Activates the python venv
- Sets up env path required by SPEED
- (hopefully) Simplifies input arguments a bit
Download `run.sh` with your preferred method into the SPEED folder.
```bash
cd SPEED
wget https://github.com/PeterRobots/Running-SPEED-video-interpolater/blob/main/run.sh
sudo chmod +x run.sh
```

## Run the script
Below is an example running the script in sequential mode with double fps (retains video length).
```bash
./run.sh -i "path/input_file.mp4" -i "path/output_file.mp4"
```
## Without script 
You can run the model following the example in the SPEED repo, running `inference.py` arg.
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

# To do
Investigate additional modes of operation: 
- Greater than bisection possible? (trisection?)
	- `interpolate_batch()` passes `model()` a timestep value, can it be changed?
		-  Investigate timestep value
			-  `timestep = torch.full((frame0.shape[0],), 1000.0, dtype=frame0.dtype, device=device)`
				- `torch.full()` makes a tensor of size `arg1` and sets all values to `arg2`.
				- Why `1000.0`?
					- do other values return sensible outputs?
					- can I get two interpolated values at `500.0` and `1500.0`?
	- Give user a timestep argument.
		- Modify functions:
			- in `inference.py`
				- `parse_args`() in `main()`
					- `validate_args()`
				- `interpolate_video_parallel()`
				- `iterpolate_video_sequential()`
					-  `interpolate_batch()` 
