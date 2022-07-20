# Streamable Neural Fields

### [Paper link]

Junwoo Cho\*, Seungtae Nam\*, Daniel Rho, [Jong Hwan Ko](http://iris.skku.edu/#hero), [Eunbyung Park](https://silverbottlep.github.io/)&dagger;<br>
\* Equal contribution, alphabetically ordered.<br>
&dagger; Corresponding author.

European Conference on Computer Vision (ECCV), 2022

## Requirements
Setup a conda environment using commands below:
```
conda env create -f environment.yml
conda activate snf
```

## Dataset
Download Kodak dataset from [here](http://www.cs.albany.edu/~xypan/research/snr/Kodak.html).

Download UVG dataset from [here](http://ultravideo.fi/#testsequences).

Download 3D point cloud dataset from [here].

Set the data directories like below:

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
        YachtRide
```

## Reproducing experiments
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

## Results
You can find both qualitative and quantitative results in \results directory.

## Citation
'''
''' 
