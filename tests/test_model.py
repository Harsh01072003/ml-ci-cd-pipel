# tests/test_model.py
import unittest
import joblib
from sklearn.linear_model import LinearRegression

class TestHousePriceModel(unittest.TestCase):
    def test_model_instance(self):
        model = joblib.load('model/house_price_model.pkl')
        self.assertIsInstance(model, LinearRegression)

if __name__ == '__main__':
    unittest.main()
