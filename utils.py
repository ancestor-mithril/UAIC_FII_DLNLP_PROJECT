from sklearn.model_selection import train_test_split


def train_validation_split(df):
    train, test = train_test_split(df, test_size=0.1)