"""Testes automatizados para validação do pipeline de dados MNIST (dataset.py).

Compatível com pytest e unittest nativo.
"""

import unittest
import torch
from dataset import (
    CANONICAL_MEAN,
    CANONICAL_STD,
    get_class_distribution,
    get_mnist_dataloaders,
    get_mnist_datasets,
)


class TestMNISTDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Carrega os datasets uma única vez para a suíte de testes."""
        cls.train_ds, cls.val_ds, cls.test_ds = get_mnist_datasets(
            data_dir="./data", val_size=10000, seed=42
        )

    def test_split_dimensions(self):
        """Verifica se a partição 50.000 / 10.000 / 10.000 totaliza rigorosamente 70.000 imagens."""
        self.assertEqual(len(self.train_ds), 50000)
        self.assertEqual(len(self.val_ds), 10000)
        self.assertEqual(len(self.test_ds), 10000)
        self.assertEqual(len(self.train_ds) + len(self.val_ds) + len(self.test_ds), 70000)

    def test_tensor_shape_and_dtype(self):
        """Verifica se as amostras são tensores 2D (1x28x28) em ponto flutuante."""
        image, label = self.train_ds[0]

        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(image.shape, (1, 28, 28))
        self.assertEqual(image.dtype, torch.float32)
        self.assertTrue(0 <= int(label) <= 9)

    def test_dataloader_batching(self):
        """Valida o agrupamento em mini-batches e os shapes de saída dos DataLoaders."""
        batch_size = 32
        train_loader, val_loader, test_loader = get_mnist_dataloaders(
            data_dir="./data", batch_size=batch_size, val_size=10000, seed=42
        )

        images, labels = next(iter(train_loader))
        self.assertEqual(images.shape, (batch_size, 1, 28, 28))
        self.assertEqual(labels.shape, (batch_size,))

    def test_split_reproducibility(self):
        """Garante que a semente pseudoaleatória (seed) produz partições estritamente idênticas."""
        train_ds_1, val_ds_1, _ = get_mnist_datasets(data_dir="./data", val_size=10000, seed=42)
        train_ds_2, val_ds_2, _ = get_mnist_datasets(data_dir="./data", val_size=10000, seed=42)

        self.assertEqual(train_ds_1.indices, train_ds_2.indices)
        self.assertEqual(val_ds_1.indices, val_ds_2.indices)

    def test_class_distribution_integrity(self):
        """Verifica se a contagem de amostras por classe cobre exatamente o total de imagens."""
        for ds, expected_total in [(self.train_ds, 50000), (self.val_ds, 10000), (self.test_ds, 10000)]:
            counts = get_class_distribution(ds)
            self.assertEqual(len(counts), 10)
            self.assertEqual(sum(counts.values()), expected_total)
            for digit, count in counts.items():
                self.assertGreater(count, 0)


if __name__ == "__main__":
    unittest.main()
