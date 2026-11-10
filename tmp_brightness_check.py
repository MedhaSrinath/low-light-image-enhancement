import random, sys
from pathlib import Path
from PIL import Image, ImageStat
DATA_DIR = Path('data/LOL-v2-real')
image_ext = {'.png', '.jpg', '.jpeg', '.bmp'}
for split in ['Train', 'Test']:
    in_dir = DATA_DIR / split / 'Input'
    gt_dir = DATA_DIR / split / 'GT'
    in_files = sorted([p for p in in_dir.rglob('*') if p.is_file() and p.suffix.lower() in image_ext])
    sample = random.sample(in_files, 5)
    print(f'--- {split} sample ---')
    for inp_path in sample:
        gt_path = gt_dir / inp_path.name
        try:
            im_in = Image.open(inp_path).convert('L')
            im_gt = Image.open(gt_path).convert('L')
            mean_in = ImageStat.Stat(im_in).mean[0]
            mean_gt = ImageStat.Stat(im_gt).mean[0]
            print(f'{inp_path.name}: low-light mean={mean_in:.1f}, GT mean={mean_gt:.1f}')
        except Exception as e:
            print('Error', inp_path.name, e)

