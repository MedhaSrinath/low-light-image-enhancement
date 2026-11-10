import sys
from pathlib import Path
try:
    from PIL import Image
except ImportError:
    print('Pillow not installed')
    sys.exit(1)
DATA_DIR = Path('data/LOL-v2-real')
image_ext = {'.png', '.jpg', '.jpeg', '.bmp'}
report = []
for split in ['Train', 'Test']:
    in_dir = DATA_DIR / split / 'Input'
    gt_dir = DATA_DIR / split / 'GT'
    in_files = sorted([p for p in in_dir.rglob('*') if p.is_file() and p.suffix.lower() in image_ext])
    gt_files = sorted([p for p in gt_dir.rglob('*') if p.is_file() and p.suffix.lower() in image_ext])
    report.append(f"{split}: Input {len(in_files)} GT {len(gt_files)}")
    # mismatched names
    in_names = {p.name for p in in_files}
    gt_names = {p.name for p in gt_files}
    missing_in = gt_names - in_names
    missing_gt = in_names - gt_names
    report.append(f"  Missing input for GT: {len(missing_in)}")
    report.append(f"  Missing GT for input: {len(missing_gt)}")
    # check dimensions and ability to open
    dim_mismatch = 0
    corrupted = 0
    for inp_path, gt_path in zip(in_files, gt_files):
        try:
            im_in = Image.open(inp_path)
            im_gt = Image.open(gt_path)
            im_in.verify()
            im_gt.verify()
            # reopen to get size
            im_in = Image.open(inp_path)
            im_gt = Image.open(gt_path)
            if im_in.size != im_gt.size:
                dim_mismatch += 1
        except Exception as e:
            corrupted += 1
    report.append(f"  Corrupted or unreadable pairs: {corrupted}")
    report.append(f"  Dimension mismatches: {dim_mismatch}")
print('\n'.join(report))

