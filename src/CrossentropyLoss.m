function [loss, err] = CrossentropyLoss(batch)

if ~ischar(batch)
    lim = 1e-3;
    y = min(max(batch.GetDataAsMatrix(), 0), 1);
    t = batch.GetLabelsAsMatrix();
    denom = y .* (1-y);
    denom(denom < lim) = 1;
    err = (y - t) ./ denom;
    loss = -mean(mean(log(y + eps).*t + log(1-y + eps).*(1-t), 2));
    if nargout > 1
        err = Batch(err);
    end
else
    if strcmp(batch, 'max')
        loss = 5;
    elseif strcmp(batch, 'min')
        loss = 0;
    end
end