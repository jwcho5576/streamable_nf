# Streamable Neural Fields

### [Paper link]

Junwoo Cho\*, Seungtae Nam\*, Daniel Rho, [Jong Hwan Ko](http://iris.skku.edu/#hero), [Eunbyung Park](https://silverbottlep.github.io/)&dagger;<br>
\* Equal contribution.<br>
&dagger; Corresponding author.


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
```
directory structure
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
