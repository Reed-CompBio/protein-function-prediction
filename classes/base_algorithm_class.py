class BaseAlgorithm:
    def __init__(self):
        self.y_true = None
        self.y_score = None

    def predict(
        self, positive_data_set, negative_data_set, sample_size, graph, output_path
    ):
        raise NotImplementedError("Predict method must be implemented in subclass")
