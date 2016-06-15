function miss = GetMissclassRate(results)

[~, inds_pred] = max(results.GetDataAsMatrix());
[~, inds_lbl] = max(results.GetLabelsAsMatrix());
miss = 100*mean(abs(inds_pred-inds_lbl)>0.01);