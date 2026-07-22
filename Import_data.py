from data_processor import CryptoDataProcessor


processor = CryptoDataProcessor(
    "data/BTCUSDT.csv"
)


train_data, _, test_data, features = (
    processor.process()
)


print(train_data.shape)
print(test_data.shape)
print(features)