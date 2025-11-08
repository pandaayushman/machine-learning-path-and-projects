import unittest
from src.data_loader import load_data
from src.preprocess import preprocess_data

class TestParkinsonPipeline(unittest.TestCase):

    def test_data_loading(self):
        df = load_data("data/parkinsons_dataset.csv")
        self.assertIsNotNone(df)
        self.assertFalse(df.empty)

    def test_preprocessing(self):
        df = load_data("data/parkinsons_dataset.csv")
        X, y, scaler = preprocess_data(df)
        self.assertEqual(X.shape[0], len(y))
        self.assertIsNotNone(scaler)

if __name__ == "__main__":
    unittest.main()
