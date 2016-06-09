function [loss, err] = MSELoss(batch)

if ~ischar(batch)
    err = batch.GetDataAsMatrix()-batch.GetLabelsAsMatrix();
    loss = mean(sqrt(mean(err.^2, 2)));
    if nargout > 1
        err = Batch(err);
    end
else
    if strcmp(batch, 'max')
        loss = 1;
    elseif strcmp(batch, 'min')
        loss = 0;
    end
end