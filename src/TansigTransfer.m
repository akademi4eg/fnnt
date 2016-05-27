function data = TansigTransfer(data)

data = 2./(1 + exp(-2*data)) - 1;