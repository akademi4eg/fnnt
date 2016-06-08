function h = PlotMSE(h, results, val_results, reg_loss)
% PlotMSE Plots mean-square-error as training progresses.

use_val = exist('val_results', 'var') && ~isempty(val_results.GetDataAsMatrix());
new_mse = MSELoss(results) + reg_loss; % TODO optimize double call of MSELoss
if use_val
    new_val_mse = MSELoss(val_results);
end
if ~ishandle(h)
    h = figure();
    h.CurrentAxes = axes;
    semilogy(h.CurrentAxes, 1, new_mse, 'r');
    if use_val
        hold(h.CurrentAxes, 'on');
        semilogy(h.CurrentAxes, 1, new_val_mse, 'b');
        legend(h.CurrentAxes, 'Train', 'Validation');
    end
    xlabel(h.CurrentAxes, 'Epochs');
    ylabel(h.CurrentAxes, 'MSE');
    ep = 1;
else
    x = get(h.CurrentAxes.Children(end), 'XData');
    y = get(h.CurrentAxes.Children(end), 'YData');
    ep = max(x) + 1;
    x = [x, ep];
    y = [y, new_mse];
    set(h.CurrentAxes.Children(end), 'XData', x, 'YData', y);
    min_y = 10^(-ceil(-log10(eps + min(y))));
    max_y = 10^(-floor(-log10(eps + max(y))));
    if use_val
        x = get(h.CurrentAxes.Children(end-1), 'XData');
        y = get(h.CurrentAxes.Children(end-1), 'YData');
        ep = max(x) + 1;
        x = [x, ep];
        y = [y, new_val_mse];
        set(h.CurrentAxes.Children(end-1), 'XData', x, 'YData', y);
        min_y = min(10^(-ceil(-log10(eps + min(y)))), min_y);
        max_y = max(10^(-floor(-log10(eps + max(y)))), max_y);
    end
    axis(h.CurrentAxes, [1, ep, min_y, max_y]);
end

if ~use_val
    title(h.CurrentAxes, ['MSE = ' num2str(new_mse) ' after training for ', num2str(ep), ' epochs']);
else
    title(h.CurrentAxes, ['Validation MSE = ' num2str(new_val_mse) ' after training for ', num2str(ep), ' epochs']);
end