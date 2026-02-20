import pandas as pd


def load_data(train_path, test_path, useful_columns=True):
    train_df = pd.read_json(train_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)
    if useful_columns:
        train_df = train_df[['message', 'label', 'group']]
        test_df = test_df[['message', 'label', 'group']]
    return train_df, test_df


def split_train_dataset(df, sample_threshold, train_ratio, random_state):
    train_set, validation_set = [], []
    for label in df['label'].unique():
        df_label = df[df['label'] == label]
        n_samples = min(df_label.shape[0], sample_threshold)
        df_sample = df_label.sample(n_samples, random_state=random_state)
        n_train = int(n_samples * train_ratio)
        train_set.append(df_sample[:n_train])
        validation_set.append(df_sample[n_train:])
    train_df = pd.concat(train_set).reset_index(drop=True)
    validation_df = pd.concat(validation_set).reset_index(drop=True)
    return train_df, validation_df






