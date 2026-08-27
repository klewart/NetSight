import os
import zipfile

if os.path.exists('netsight_colab_fixed.zip'):
    os.remove('netsight_colab_fixed.zip')

print('Zipping code for Kaggle (no data needed!)...')
with zipfile.ZipFile('netsight_kaggle_code.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for folder in ['my_agents']:
        for root, _, files in os.walk(folder):
            for f in files:
                # Don't zip the huge weights file if it's there
                if 'weights' in root and f.endswith('.pth'):
                    continue
                zf.write(os.path.join(root, f))
    for file in ['train.py', 'app.py', 'evaluate.py', 'download_dataset.py', 'colab_requirements.txt']:
        if os.path.exists(file):
            zf.write(file)
print('Done zipping!')
