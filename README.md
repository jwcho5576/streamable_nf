# Streamable Neural Fields

### [Paper link](https://arxiv.org/abs/2207.09663)

Junwoo Cho\*, Seungtae Nam\*, Daniel Rho, [Jong Hwan Ko](http://iris.skku.edu/#hero), [Eunbyung Park](https://silverbottlep.github.io/)&dagger;<br>
\* Equal contribution, alphabetically ordered.<br>
&dagger; Corresponding author.

European Conference on Computer Vision (ECCV), 2022

## 0. Requirements
Setup a conda environment using commands below:
```
conda env create -f environment.yml
conda activate snf
```

## 1. Dataset
Download Kodak dataset from [here](http://www.cs.albany.edu/~xypan/research/snr/Kodak.html).

Download UVG dataset from [here](http://ultravideo.fi/#testsequences).<br>
When downloading UVG video, please use this version:<br>
* Resolution: 1080p<br>
* Bit depth: 8<br>
* Format: AVC<br>
* Container: MP4<br>

Download 3D point cloud dataset from [here](https://drive.google.com/drive/u/1/folders/1-K2460VPEvwk9CtIkjcMzia2OKY_l_7c).

'data/' directory must be in your working directory. The structure is as follows:

##### Data layout
```
data/
    kodak/
        kodim01.png
        ...
        kodim24.png
    shape/
        armadillo.xyz
        dragon.xyz
        happy_buddha.xyz
    uvg/
        Beauty.mp4
        ...
        YachtRide.mp4
```

## 2. Reproducing experiments

Run the commands below.

### Image spectral growing
```
bash scripts/train_image_spectral.sh
```

### Image spatial growing
```
bash scripts/train_image_spatial.sh
```

### Video temporal growing
```
bash scripts/train_video_temporal.sh
```

### SDF spectral growing
```
bash scripts/train_sdf_spectral.sh
```

## 3. Results
You can find both qualitative and quantitative results in \results directory.
