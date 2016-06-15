function data = LogisticTransfer(data)

data = 1./(1 + exp(-data));