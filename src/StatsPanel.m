function [f, p1, p2] = StatsPanel(f, p1, p2, stats, net)

margins = 5;
bar_height = 30;
win_height = (1+length(stats))*(bar_height + margins) + margins;
for i = 1:length(stats)
    if (stats(i).current>stats(i).min && stats(i).current>stats(i).max) ...
        || (stats(i).current<stats(i).min && stats(i).current<stats(i).max)
        stats(i).min = stats(i).current;
    end
end
if isempty(f)
    f = uifigure('Resize', 'off', 'ToolBar', 'none', 'MenuBar', 'none', ...
        'NumberTitle', 'off', 'Name', 'Training monitoring', ...
        'Position', [100 100 400 win_height]);
    cur_pos = margins;
    uibutton(f, 'Text', 'Save ANN', 'Position', [margins*2, cur_pos, f.Position(3)-4*margins, bar_height], ...
        'ButtonPushedFcn', @(btn, ev)SaveANN(net));
    cur_pos = cur_pos + margins + bar_height;
    p1 = cell(length(stats), 1);
    p2 = cell(length(stats), 1);
    for i = 1:length(stats)
        p1{i} = uipanel(f, 'BackgroundColor','white', 'BorderType', 'none', ...
            'Position', [margins cur_pos ...
                         f.Position(3)-2*margins bar_height], ...
            'Title', [stats(i).label ': ' num2str(stats(i).current)]);
        ratio = abs((stats(i).current-stats(i).min)/(stats(i).max-stats(i).min));
        p2{i} = uipanel(p1{i}, 'BackgroundColor',[0.8 0 0], 'BorderType', 'none', ...
            'Position', [1 1 ratio*p1{i}.Position(3) p1{i}.Position(4)]);
        cur_pos = cur_pos + margins + bar_height;
    end
else
    for i = 1:length(stats)
        p1{i}.Title = [stats(i).label ': ' num2str(stats(i).current)];
        ratio = abs((stats(i).current-stats(i).min)/(stats(i).max-stats(i).min));
        p2{i}.Position(3) = ratio*p1{i}.Position(3);
    end
end

function SaveANN(net)
if isa(net, 'Network')
    save('ann.mat', 'net');
    fprintf('Network saved to ann.mat.\n');
    drawnow;
end