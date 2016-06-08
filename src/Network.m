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
                'plots_handles', {{}}, 'loss', @MSELoss, ...
                'early_stop', Inf, 'learn_rate', 0.1, ...
                'regularization', struct('type', 'none', 'param', 0));
            obj.Mode = 'blank';
        end
        
        function SetRegularization(obj, type, param)
            obj.Training.regularization.type = type;
            obj.Training.regularization.param = param;
        end
        
        function SetEpochsNum(obj, epochs)
            obj.Training.epochs = epochs;
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
                obj.Layers{i}.Update(results(i), results(i+1), grads(i+1), ...
                    obj.Training);
            end
        end
        
        function Train(obj, batches)
            tic;
            fails = 0;
            len = 0;
            val_len = 0;
            val_loss = Inf;
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
                for bi = randperm(length(batches))
                    if ~strcmp(batches{bi}.set_id, 'trn'), continue; end
                    results = obj.Apply(copy(batches{bi}));
                    grads = obj.Backprop(results);
                    obj.Update(results, grads);
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
                for l = obj.Layers
                    new_val_loss = new_val_loss + l{1}.GetRegLoss(obj.Training);
                end
                if new_val_loss >= val_loss
                    fails = fails + 1;
                    fprintf('.');
                else
                    if fails > 0, fprintf('\n'); end
                    val_loss = new_val_loss;
                    fails = 0;
                end
                % SHOW
                for j = 1:length(obj.Training.plots)
                    obj.Training.plots_handles{j} = ...
                        obj.Training.plots{j}(obj.Training.plots_handles{j}, results, val_results);
                end
                drawnow;
                if fails >= obj.Training.early_stop
                    fprintf('\nEarly stoping triggered after %d stale epochs.\n', fails);
                    break;
                end
            end
            el_time = toc;
            if el_time < 1000
                str_time = [num2str(round(el_time)) ' seconds'];
            elseif el_time < 300*60
                str_time = [num2str(round(el_time/60)) ' minutes'];
            else
                str_time = [num2str(round(el_time/60/60)) ' hours'];
            end
            fprintf('Done training. After %d epochs (%s) validation loss is %f.\n', ...
                i, str_time, new_val_loss);
            obj.Mode = 'eval';
        end
    end
end