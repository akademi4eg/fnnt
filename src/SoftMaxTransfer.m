function data = SoftMaxTransfer(data)

[~, max_ind] = max(data, [], 1);
data = exp(data);
data = bsxfun(@times, data, 1./sum(data, 1));
nan_mask = isnan(data) | isinf(data);
any_nan_mask = any(nan_mask, 1);
data(:, any_nan_mask) = 0;
for i = 1:size(data, 2)
    if ~any_nan_mask(i), continue; end
    data(max_ind(i), i) = 1;
end