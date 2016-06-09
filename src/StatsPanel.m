function [f, p1, p2] = StatsPanel(f, p1, p2, stats)

margins = 5;
bar_height = 20;
win_height = length(stats)*(bar_height + margins) + margins;
for i = 1:length(stats)
    if (stats(i).current>stats(i).min && stats(i).current>stats(i).max) ...
        || (stats(i).current<stats(i).min && stats(i).current<stats(i).max)
        stats(i).min = stats(i).current;
    end
end
if isempty(f)
    f = figure('Resize', 'off', 'ToolBar', 'none', 'MenuBar', 'none', ...
        'NumberTitle', 'off', 'Name', 'Training monitoring', ...
        'Position', [100 100 400 win_height]);
    cur_pos = margins;
    p1 = cell(length(stats), 1);
    p2 = cell(length(stats), 1);
    for i = 1:length(stats)
        p1{i} = uipanel(f, 'BackgroundColor','white', 'BorderType', 'none', ...
            'Position', [margins/f.Position(3) cur_pos/win_height ...
                         1-2*margins/f.Position(3) bar_height/win_height], ...
            'Title', [stats(i).label ': ' num2str(stats(i).current)]);
        ratio = abs((stats(i).current-stats(i).min)/(stats(i).max-stats(i).min));
        p2{i} = uipanel(p1{i}, 'BackgroundColor',[0.8 0 0], 'BorderType', 'none', ...
            'Position', [0 0 ratio 1]);
        cur_pos = cur_pos + margins + bar_height;
    end
else
    for i = 1:length(stats)
        p1{i}.Title = [stats(i).label ': ' num2str(stats(i).current)];
        ratio = abs((stats(i).current-stats(i).min)/(stats(i).max-stats(i).min));
        p2{i}.Position(3) = ratio;
    end
end