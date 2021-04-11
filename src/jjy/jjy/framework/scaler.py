# %%

import numpy as np


# %%

class StandardScaler:
    def __init__(self):
        self.std_array = None
        self.mean_array = None

    def fit(self, data):
        self.data = data

        self.std_array = np.std(data, axis=0)
        self.mean_array = np.mean(data, axis=0)

    def transform(self, data):
        return (data - self.mean_array) / self.std_array

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

# %%


