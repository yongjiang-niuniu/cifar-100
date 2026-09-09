"""Offline structural and execution checks, not CIFAR-100 accuracy benchmarks."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]


class TinyCifarFixture(Dataset):
    """Small generated image splits exercise code paths without a dataset download."""
    def __init__(self, root, train=True, download=False, transform=None):
        self.train = train
        self.transform = transform
        self.classes = [f'class-{i}' for i in range(100)]

    def __len__(self):
        return 16 if self.train else 11

    def __getitem__(self, index):
        rng = np.random.default_rng(index + (0 if self.train else 100))
        pixels = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
        return self.transform(image) if self.transform else image, index


class NotebookChecks(unittest.TestCase):
    def test_originals_and_working_notebook_schema(self):
        manifest = json.loads((ROOT / 'archive/original_notebooks.json').read_text())
        self.assertEqual(len(manifest['files']), 3)
        for item in manifest['files']:
            content = (ROOT / item['archive_path']).read_bytes()
            self.assertEqual(hashlib.sha256(content).hexdigest(), item['original_sha256'])
            notebook = nbformat.read(ROOT / item['working_path'], as_version=4)
            nbformat.validate(notebook)
            for cell in notebook.cells:
                if cell.cell_type == 'code':
                    self.assertEqual(cell.outputs, [])
                    self.assertIsNone(cell.execution_count)
                    source = '\n'.join(line for line in cell.source.splitlines()
                                       if not line.lstrip().startswith(('%', '!')))
                    compile(source, item['working_path'], 'exec')

    def test_resnet_training_evaluation_and_distinct_perturbations(self):
        notebook = json.loads((ROOT / 'notebooks/ResNet_cifar100.ipynb').read_text())
        state = {'__name__': '__notebook_smoke__'}
        old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        torch.manual_seed(42)
        np.random.seed(42)
        try:
            with tempfile.TemporaryDirectory(prefix='cifar-notebook-check-') as directory:
                output = io.StringIO()
                with contextlib.chdir(directory), contextlib.redirect_stdout(output):
                    with patch('torchvision.datasets.CIFAR100', TinyCifarFixture), \
                         patch('socket.socket.connect', side_effect=AssertionError('Network is forbidden in offline checks')):
                        for cell in notebook['cells']:
                            if cell['cell_type'] != 'code' or 'visualization-only' in cell['metadata'].get('tags', []):
                                continue
                            source = '\n'.join(line for line in cell['source']
                                               if not line.lstrip().startswith(('%', '!')))
                            exec(compile(source, 'ResNet_cifar100.ipynb', 'exec'), state)
                            if cell['metadata'].get('original_cell_index') == 2:
                                state.update(batch_size=8, num_epochs=1, use_pretrained=False,
                                             use_cuda=False, device=torch.device('cpu'))
                        self.assertTrue(state['dataloaders']['train'].dataset.train)
                        self.assertFalse(state['dataloaders']['test'].dataset.train)
                        self.assertEqual(state['total'], 11)
                        self.assertTrue((Path(directory) / 'saved_models/weights.h5').is_file(), output.getvalue())
                        model = state['resnet50_model'].eval()
                        weighted_loss, count = 0.0, 0
                        with torch.no_grad():
                            for images, labels in state['dataloaders']['test']:
                                logits = model(images)
                                self.assertEqual(logits.shape[1], 100)
                                weighted_loss += state['loss_function'](logits, labels).item() * len(labels)
                                count += len(labels)
                        self.assertAlmostEqual(state['test_losses'][-1], weighted_loss / count, places=5)
                        self.assertEqual(len(state['results']), 5)
                        for levels in state['results'].values():
                            self.assertEqual(set(levels), {1, 2, 3, 4, 5})
                            self.assertTrue(all(0 <= value <= 100 for value in levels.values()))
                        loaders = state['perturbed_loaders']
                        gaussian = lambda level: loaders['Gaussian'][level].dataset.transform.transforms[1]
                        sample = torch.full((3, 32, 32), 0.5)
                        torch.manual_seed(19)
                        noise1 = gaussian(1)(sample) - sample
                        torch.manual_seed(19)
                        noise5 = gaussian(5)(sample) - sample
                        self.assertTrue(torch.allclose(noise5, noise1 * 5, atol=3e-7))
                        salt_pepper = lambda level: loaders['SaltPepper'][level].dataset.transform.transforms[1]
                        torch.manual_seed(19)
                        changed1 = int((salt_pepper(1)(sample) != sample).sum())
                        torch.manual_seed(19)
                        changed5 = int((salt_pepper(5)(sample) != sample).sum())
                        self.assertGreater(changed1, 0)
                        self.assertGreater(changed5, changed1)
                        mask = lambda level: loaders['Mask'][level].dataset.transform.transforms[1](sample)
                        self.assertEqual(int((mask(1) == 0).sum()), 2 * 3 * 8 * 8)
                        self.assertEqual(int((mask(5) == 0).sum()), 12 * 3 * 8 * 8)
        finally:
            plt.close('all')
            torch.set_num_threads(old_threads)


if __name__ == '__main__':
    unittest.main()
