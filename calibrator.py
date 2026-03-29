import numpy as np

class BayesianCalibrator:
    def __init__(self, prior_mean=0.5, prior_std=0.2):
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.observations = []

    def update(self, new_data: float, data_weight=1.0):
        """Bayesian update with new information"""
        self.observations.append((new_data, data_weight))

        weighted_sum = sum(obs * w for obs, w in self.observations)
        total_weight = sum(w for _, w in self.observations)

        posterior = (self.prior_mean + weighted_sum) / (1 + total_weight)
        return np.clip(posterior, 0.01, 0.99)

    def tail_calibration(self, probability: float, threshold=0.1) -> float:
        """Adjust extreme probabilities for tail risk"""
        if probability < threshold:
            return probability * 0.9
        elif probability > (1 - threshold):
            return threshold + (probability - threshold) * 0.9
        return probability

    def brier_score(self, predictions: list, outcomes: list) -> float:
        """Calculate calibration error"""
        return np.mean([(p - o)**2 for p, o in zip(predictions, outcomes)])

if __name__ == "__main__":
    calibrator = BayesianCalibrator(prior_mean=0.5)

    # Simulate information arrival
    calibrator.update(0.6, weight=1.0)
    calibrator.update(0.65, weight=1.5)

    posterior = calibrator.update(0.7, weight=2.0)
    print(f"Updated probability: {posterior:.3f}")

    tail_adjusted = calibrator.tail_calibration(0.95)
    print(f"Tail calibration: 0.95 -> {tail_adjusted:.3f}")
