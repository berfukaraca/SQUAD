# SQUAD
SQUAD: A Unified Framework for 3D Dyadic Scene Understanding with Applications to Visual Focus of Attention and Object Use


## Setup
1. Download AlphaPose from https://github.com/MVIG-SJTU/AlphaPose/tree/master and DMMR from https://github.com/boycehbz/DMMR. Follow their installation instructions and download the required models.
2. Download the SMPL body models from the official SMPLify website https://smplify.is.tuebingen.mpg.de/ and place them in DMMR/models/smpl/, following DMMR’s model instructions.
3. Create the SQUAD Conda environment from the SQUAD directory:
conda env create -f SQUAD_env.yml

## Usage
Set the main project path in run_all.py, then run it to execute the full SQUAD pipeline.
The multiview_alljoints.py code performs object triangulation and extracts 3D pose information from the SQUAD output.
Data and pretrained models are not included. External dependencies and models retain their respective licenses.

## SQUAD folder structure
The project scripts remain in the SQUAD root directory.

```text
SQUAD/
├── checkpoints/
├── data/
│   ├── alphapose_results/
│   ├── annotations/
│   │   ├── frames/
│   │   └── participant1/
│   │       ├── attention/
│   │       └── touch/
│   ├── assigned_ids/
│   ├── camera_movement/
│   ├── detection/
│   ├── dmmr_output/
│   ├── extracted_frames/
│   ├── frame_offsets/
│   ├── fullvideo/
│   ├── input_videos/
│   ├── inputs/
│   ├── joints/
│   └── pose_tracks/
├── model_outputs/
└── train33/
    ├── args.yaml
    └── weights/
        └── best.pt
```
## Citation
If you use SQUAD in your research, please cite:

Berfu Karaca, Niilo V. Valtakari, Albert Ali Salah, Jaap J. A. Denissen, Sonja M. C. de Zwarte, and Ronald Poppe. 2026. SQUAD: A Unified Framework for 3D Dyadic Scene Understanding with Applications to Visual Focus of Attention and Object Use. In INTERNATIONAL CONFERENCE ON MULTIMODAL INTERACTION (ICMI ’26), October 05–09, 2026, Napoli, Italy. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3776574.3831168

## Acknowledgements

We thank the authors of AlphaPose, DMMR, and SMPL for making their code and models available to the research community.
