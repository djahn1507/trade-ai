# config.py
ticker = "AAPL"
sequence_length = 30
sequence_length_candidates = [30, 60]
batch_size_candidates = [16, 32]
label_threshold = 0.01
label_lookahead = 5

train_start = "2015-01-01"
train_end   = "2022-12-31"
test_start  = "2023-01-01"
test_end    = "2024-12-31"
