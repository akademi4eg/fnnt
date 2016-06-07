function [loss, err] = MSELoss(batch)

err = batch.GetDataAsMatrix()-batch.GetLabelsAsMatrix();
loss = mean(sqrt(mean(err.^2, 2)));
if nargout > 1
    err = Batch(err);
end