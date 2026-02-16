import matplotlib.pyplot as plt


def histogram_df(train_df, test_df):
    train_label = (train_df.groupby('label').count().sort_values(by='message', ascending=False).reset_index())
    train_group = (train_df.groupby('group').count().sort_values(by='message', ascending=False).reset_index())
    test_label = (test_df.groupby('label').count().sort_values(by='message', ascending=False).reset_index())
    test_group = (test_df.groupby('group').count().sort_values(by='message', ascending=False).reset_index())
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    # Training dataset - Single Label
    axes[0, 0].bar(train_label['label'], train_label['message'])
    axes[0, 0].set_title("Training dataset - Single Label Count")
    axes[0, 0].set_xlabel("GitHub Single Label")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].tick_params(axis='x', rotation=-45)
    # Training dataset - Group Label
    axes[0, 1].bar(train_group['group'], train_group['message'])
    axes[0, 1].set_title("Training dataset - Group Label Count")
    axes[0, 1].set_xlabel("GitHub Group Label")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].tick_params(axis='x', rotation=-45)
    # Test dataset - Single Label
    axes[1, 0].bar(test_label['label'], test_label['message'])
    axes[1, 0].set_title("Test dataset - Single Label Count")
    axes[1, 0].set_xlabel("GitHub Single Label")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].tick_params(axis='x', rotation=-45)
    # Test dataset - Group Label
    axes[1, 1].bar(test_group['group'], test_group['message'])
    axes[1, 1].set_title("Test dataset - Group Label Count")
    axes[1, 1].set_xlabel("GitHub Group Label")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].tick_params(axis='x', rotation=-45)
    plt.tight_layout()
    plt.show()
