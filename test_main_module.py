
import unittest
import pandas as pd
from main_module import preprocess_all, preprocess_for_prediction

class TestPreprocessing(unittest.TestCase):

    def test_preprocess_all_structure(self):
        # Création de jeux de données simulés
        df_train_input = pd.DataFrame({
            "ID": [1, 2],
            "ZONE": ["A", "B"],
            "ANNEE_ASSURANCE": [1, 2]
        })
        df_test_input = pd.DataFrame({
            "ID": [3],
            "ZONE": ["C"],
            "ANNEE_ASSURANCE": [3]
        })
        df_output = pd.DataFrame({
            "ID": [1, 2],
            "FREQ": [0.1, 0.2],
            "CM": [100, 200],
            "CHARGE": [10, 40]
        })

        df_train_input.to_csv("train_input.csv", index=False)
        df_test_input.to_csv("test_input.csv", index=False)
        df_output.to_csv("train_output.csv", index=False)

        X_train, X_test, y_freq, y_cm, df, X, y = preprocess_all(
            "train_input.csv", "test_input.csv", "train_output.csv", n_train=2
        )

        self.assertEqual(X_train.shape[0], 2)
        self.assertEqual(X_test.shape[0], 1)
        self.assertIn("ZONE", X.columns)

    def test_preprocess_for_prediction(self):
        df = pd.DataFrame({
            "ID": [1],
            "ZONE": ["A"],
            "ANNEE_ASSURANCE": [1]
        })
        df.to_csv("predict.csv", index=False)
        result = preprocess_for_prediction("predict.csv")
        self.assertEqual(result.shape[0], 1)
        self.assertIn("ZONE", result.columns)

if __name__ == "__main__":
    unittest.main()
