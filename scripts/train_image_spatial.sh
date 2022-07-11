python main.py --experiment image_spatial --model progressive --kodak_num 21 --widths 59 96 126 149 --epochs 50000 --gpu_id 2
python main.py --experiment image_spatial --model slimmable --kodak_num 21 --widths 59 84 104 120 --epochs 50000 --gpu_id 2
python main.py --experiment image_spatial --model individual --kodak_num 21 --widths 59 59 59 59 --epochs 50000 --gpu_id 2