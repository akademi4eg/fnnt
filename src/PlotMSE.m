function h = PlotMSE(h, results, ~)
% PlotMSE Plots mean-square-error as training progresses.

new_mse = mean(sqrt(mean((results(end).GetDataAsMatrix()-results(end).GetLabelsAsMatrix()).^2)));
if ~ishandle(h)
    h = figure();
    h.CurrentAxes = axes;
    semilogy(h.CurrentAxes, 1, new_mse);
    xlabel(h.CurrentAxes, 'Epochs');
    ylabel(h.CurrentAxes, 'MSE');
    ep = 1;
else
    x = get(h.CurrentAxes.Children, 'XData');
    y = get(h.CurrentAxes.Children, 'YData');
    ep = max(x) + 1;
    x = [x, ep];
    y = [y, new_mse];
    set(h.CurrentAxes.Children, 'XData', x, 'YData', y);
    min_y = 10^(-ceil(-log10(eps + min(y))));
    max_y = 10^(-floor(-log10(eps + max(y))));
    axis(h.CurrentAxes, [1, ep, min_y, max_y]);
end
title(h.CurrentAxes, ['MSE = ' num2str(new_mse) ' after training for ', num2str(ep), ' epochs']);