classdef Network < handle
    % Network Top-level object for network operations.
    properties (SetAccess = protected)
        Layers = {}; % sequence of network layers objects
        Training; % structure with training and monitoring parameters
        Mode;
    end
    methods
        function obj = Network()
            obj.Training = struct('epochs', 100, 'plots', {{}}, ...
                'show_gui', true, ...
                'plots_handles', {{}}, 'loss', @MSELoss, ...
                'early_stop', Inf, 'max_delta', 0.1, ...
                'learn_rate', struct('value', 1, 'min_value', 1e-8, 'mult', 0.99), ...
                'regularization', struct('type', 'none', 'param', 0), ...
                'momentum', struct('type', 'none', 'param', 0.5, 'step', 0.01, 'end_param', 0.99), ...
                'state', struct('grad_overflow', 0));
            obj.Mode = 'blank';
        end
        
        function SetMode(obj, mode)
            obj.Mode = mode;
        end
        
        function SetMonitoringMode(obj, mode)
            obj.Training.show_gui = strcmp(mode, 'gui');
        end
        
        function RemoveLayers(obj, layers)
            if ischar(layers)
                mask = cellfun(@(x)isa(x, layers), obj.Layers);
            else
                mask = false(size(obj.Layers));
                mask(layers) = true;
            end
            obj.Layers = obj.Layers(~mask);
        end
        
        function SetRegularization(obj, type, param)
            obj.Training.regularization.type = type;
            if exist('param', 'var')
                obj.Training.regularization.param = param;
            end
        end
        
        function SetMomentum(obj, type, param, step, end_param)
            obj.Training.momentum.type = type;
            obj.Training.momentum.param = param;
            obj.Training.momentum.step = step;
            obj.Training.momentum.end_param = end_param;
        end
        
        function SetTrainParams(obj, epochs, learn_rate, learn_mult)
            obj.Training.epochs = epochs;
            if exist('learn_rate', 'var')
                obj.Training.learn_rate.value = learn_rate;
            end
            if exist('learn_mult', 'var')
                obj.Training.learn_rate.mult = learn_mult;
            end
        end
        
        function SetEarlyStoping(obj, epochs)
            obj.Training.early_stop = epochs;
        end
        
        function SetLoss(obj, loss_fcn)
            obj.Training.loss = loss_fcn;
        end
        
        function AddTrainingPlot(obj, plot_fcn)
            obj.Training.plots{end+1} = plot_fcn;
            obj.Training.plots_handles{end+1} = NaN;
        end
        
        function AddLayer(obj, layer)
            obj.Layers{end+1} = layer;
        end
        
        function Configure(obj, batch)
            for l = obj.Layers
                l{1}.Configure(batch);
                l{1}.Forward(batch);
            end
            obj.Mode = 'train';
        end
        
        function PreTrain(obj, batches)
            for l = obj.Layers
                l{1}.PreTrain(batches, obj.Training);
                for bi = 1:length(batches)
                    l{1}.Forward(batches{bi});
                end
            end
        end
        
        function results = Apply(obj, batch)
            for l = obj.Layers
                if ~strcmp(obj.Mode, 'train') && isa(l{1}, 'DropoutLayer')
                    % don't use dropout outside of training
                    continue;
                end
                if nargout > 0
                    if ~exist('results', 'var')
                        results = copy(batch);
                    else
                        results(end+1) = copy(batch);%#ok
                    end
                end
                l{1}.Forward(batch);    
            end
            if nargout > 0
                if ~exist('results', 'var')
                    results = copy(batch);
                else
                    results(end+1) = copy(batch);
                end
            end
        end
        
        function grads = Backprop(obj, results)
            [~, delta] = obj.Training.loss(results(end));
            grads = copy(delta);
            for i = length(results):-1:2
                obj.Layers{i-1}.Backward(results(i), delta);
                grads = [copy(delta), grads];%#ok
            end
        end
        
        function Update(obj, results, grads)
            for i = 1:length(results)-1
                obj.Training = obj.Layers{i}.Update(results(i), results(i+1), grads(i+1), ...
                    obj.Training);
            end
        end
        
        function layers = GetLayersCopy(obj)
            layers = cell(size(obj.Layers));
            for i = 1:length(obj.Layers)
                layers{i} = copy(obj.Layers{i});
            end
        end
        
        function Train(obj, batches)
            tic;
            cur_best_net = obj.GetLayersCopy();
            fails = 0;
            len = 0;
            val_len = 0;
            val_loss = Inf;
            prev_train_loss = Inf;
            for bi = 1:length(batches)
                if strcmp(batches{bi}.set_id, 'trn')
                    len = len + batches{bi}.GetBatchSize();
                elseif strcmp(batches{bi}.set_id, 'val')
                    val_len = val_len + batches{bi}.GetBatchSize();
                end
            end
            for i = 1:obj.Training.epochs
                % TRAIN
                obj.Mode = 'train';
                % permute batches on each iteration
                obj.Training.flags.grad_overflow = 0;
                trn_batches_num = 0;
                for bi = randperm(length(batches))
                    if ~strcmp(batches{bi}.set_id, 'trn'), continue; end
                    trn_batches_num = trn_batches_num + 1;
                    results = obj.Apply(copy(batches{bi}));
                    grads = obj.Backprop(results);
                    obj.Update(results, grads);
                end
                if obj.Training.flags.grad_overflow > 0
                    perc = 100*obj.Training.flags.grad_overflow/trn_batches_num;
                    warning(['Gradient overflow detected in ' num2str(perc) '% of batches!']);
                end
                % EVAL
                obj.Mode = 'eval';
                results = zeros(batches{1}.GetLabelsNum()*2, len);
                val_results = zeros(batches{1}.GetLabelsNum()*2, val_len);
                cur_ind = 1;
                cur_val_ind = 1;
                for bi = 1:length(batches)
                    batch_res = obj.Apply(copy(batches{bi}));
                    if strcmp(batches{bi}.set_id, 'trn')
                        results(:, cur_ind:cur_ind+batches{bi}.GetBatchSize()-1) = [batch_res(end).GetDataAsMatrix();batch_res(end).GetLabelsAsMatrix()];
                        cur_ind = cur_ind + batches{bi}.GetBatchSize();
                    elseif strcmp(batches{bi}.set_id, 'val')
                        val_results(:, cur_val_ind:cur_val_ind+batches{bi}.GetBatchSize()-1) = [batch_res(end).GetDataAsMatrix();batch_res(end).GetLabelsAsMatrix()];
                        cur_val_ind = cur_val_ind + batches{bi}.GetBatchSize();
                    end
                end
                results = Batch(results(1:end/2, :), results(end/2+1:end, :));
                val_results = Batch(val_results(1:end/2, :), val_results(end/2+1:end, :));
                new_val_loss = obj.Training.loss(val_results);
                train_loss = obj.Training.loss(results);
                reg_loss = 0;
                for l = obj.Layers
                    reg_loss = reg_loss + l{1}.GetRegLoss(obj.Training);
                end
                new_val_loss = new_val_loss + reg_loss;
                if new_val_loss >= val_loss
                    fails = fails + 1;
                else
                    val_loss = new_val_loss;
                    fails = 0;
                    cur_best_net = obj.GetLayersCopy();
                end
                % SHOW
                for j = 1:length(obj.Training.plots)
                    obj.Training.plots_handles{j} = ...
                        obj.Training.plots{j}(obj.Training.plots_handles{j}, results, val_results, reg_loss);
                end
                if obj.Training.show_gui
                    stats = struct('label', {'Epoch', 'Learning rate', 'Momentum'}, ...
                        'min', {1, 10, 0}, ...
                        'current', {i, obj.Training.learn_rate.value, obj.Training.momentum.param}, ...
                        'max', {obj.Training.epochs, obj.Training.learn_rate.min_value, 1});
                    if ~strcmp(obj.Training.regularization.type, 'none')
                        stats = cat(2, stats, struct('label', 'Reg.loss', ...
                            'min', 1, 'max', 0, 'current', reg_loss));
                    end
                    if val_len > 0
                        stats = cat(2, stats, struct('label', 'Val.loss', ...
                            'min', obj.Training.loss('max'),...
                            'max', obj.Training.loss('min'), 'current', val_loss));
                    end
                    if ~isinf(obj.Training.early_stop)
                        stats = cat(2, stats, struct('label', 'Stale epochs', ...
                            'min', 0, 'max', obj.Training.early_stop, 'current', fails));
                    end
                    if i > 1
                        StatsPanel(f, p1, p2, stats);
                    else
                        [f, p1, p2] = StatsPanel([], [], [], stats, obj);
                    end
                else
                    fprintf('Epoch %d, loss is %f', ...
                        i, val_loss);
                    if ~strcmp(obj.Training.regularization.type, 'none')
                        fprintf(', reg.loss %f', reg_loss);
                    end
                    fprintf(', learn.rate %f', obj.Training.learn_rate.value);
                    fprintf('\n');
                end
                drawnow;
                if fails >= obj.Training.early_stop
                    break;
                end
                % UPDATE hyperparams
                if obj.Training.momentum.param < obj.Training.momentum.end_param
                    obj.Training.momentum.param = obj.Training.momentum.step ...
                        + obj.Training.momentum.param;
                    obj.Training.momentum.param = min(obj.Training.momentum.param, obj.Training.momentum.end_param);
                end
                if train_loss >= prev_train_loss || obj.Training.flags.grad_overflow
                    obj.Training.learn_rate.value = obj.Training.learn_rate.value*obj.Training.learn_rate.mult;
                    obj.Training.learn_rate.value = max(obj.Training.learn_rate.value, obj.Training.learn_rate.min_value);
                end
                prev_train_loss = train_loss;
            end
            el_time = toc;
            if el_time < 1000
                str_time = [num2str(round(el_time)) ' seconds'];
            elseif el_time < 300*60
                str_time = [num2str(round(el_time/60)) ' minutes'];
            else
                str_time = [num2str(round(el_time/60/60)) ' hours'];
            end
            fprintf('Done training. After %d epochs (%s) loss is %f.\n', ...
                i, str_time, val_loss);
            obj.Mode = 'eval';
            if ~isinf(obj.Training.early_stop)
                obj.Layers = cur_best_net;
            end
        end
    end
end