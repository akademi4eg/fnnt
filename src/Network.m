classdef Network < handle
    % Network Top-level object for network operations.
    properties (SetAccess = protected)
        layers = {}; % sequence of network layers objects
        training; % structure with training and monitoring parameters
    end
    methods
        function obj = Network()
            obj.training = struct('epochs', 100, 'plots', {{}}, ...
                'plots_handles', {{}}, 'loss', @MSELoss, 'early_stop', Inf);
        end
        
        function SetEpochsNum(obj, epochs)
            obj.training.epochs = epochs;
        end
        
        function SetEarlyStoping(obj, epochs)
            obj.training.early_stop = epochs;
        end
        
        function SetLoss(obj, loss_fcn)
            obj.training.loss = loss_fcn;
        end
        
        function AddTrainingPlot(obj, plot_fcn)
            obj.training.plots{end+1} = plot_fcn;
            obj.training.plots_handles{end+1} = NaN;
        end
        
        function AddLayer(obj, layer)
            obj.layers{end+1} = layer;
        end
        
        function Configure(obj, batch)
            for l = obj.layers
                l{1}.Configure(batch);
                l{1}.Forward(batch);
            end
        end
        
        function results = Apply(obj, batch)
            for l = obj.layers
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
            [~, delta] = obj.training.loss(results(end));
            grads = copy(delta);
            for i = length(results):-1:2
                obj.layers{i-1}.Backward(results(i), delta);
                grads = [copy(delta), grads];%#ok
            end
        end
        
        function Update(obj, results, grads)
            for i = 1:length(results)-1
                obj.layers{i}.Update(results(i), results(i+1), grads(i+1));
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
            for i = 1:obj.training.epochs
                % train
                for bi = 1:length(batches)
                    if ~strcmp(batches{bi}.set_id, 'trn'), continue; end
                    results = obj.Apply(copy(batches{bi}));
                    grads = obj.Backprop(results);
                    obj.Update(results, grads);
                end
                % eval
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
                new_val_loss = obj.training.loss(val_results);
                if new_val_loss >= val_loss
                    fails = fails + 1;
                else
                    val_loss = new_val_loss;
                    fails = 0;
                end
                % show
                for j = 1:length(obj.training.plots)
                    obj.training.plots_handles{j} = ...
                        obj.training.plots{j}(obj.training.plots_handles{j}, results, val_results);
                end
                drawnow;
                if fails >= obj.training.early_stop
                    fprintf('Early stoping triggered after %d stale epochs.\n', fails);
                    break;
                end
            end
            el_time = toc;
            fprintf('Done training. After %d epochs (%d seconds) validation loss is %f.\n', ...
                i, el_time, new_val_loss);
        end
    end
end