classdef Network < handle
    % Network Top-level object for network operations.
    properties (SetAccess = protected)
        layers; % sequence of network layers objects
        training; % structure with training and monitoring parameters
    end
    methods
        function obj = Network()
            obj.training = struct('epochs', 100, 'plots', {{}}, ...
                'plots_handles', {{}});
        end
        
        function SetEpochsNum(obj, epochs)
            obj.training.epochs = epochs;
        end
        
        function AddTrainingPlot(obj, plot_fcn)
            obj.training.plots{end+1} = plot_fcn;
            obj.training.plots_handles{end+1} = NaN;
        end
        
        function AddLayer(obj, layer)
            if isempty(obj.layers)
                obj.layers = layer;
            else
                obj.layers(end+1) = layer;
            end
        end
        
        function Configure(obj, batch)
            for l = obj.layers
                l.Configure(batch);
                l.Forward(batch);
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
                l.Forward(batch);    
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
            delta = Batch(results(end).GetDataAsMatrix()-results(end).GetLabelsAsMatrix());
            grads = copy(delta);
            for i = length(results):-1:2
                obj.layers(i-1).Backward(results(i), delta);
                grads = [copy(delta), grads];%#ok
            end
        end
        
        function Update(obj, results, grads)
            for i = 1:length(results)-1
                obj.layers(i).Update(results(i), grads(i+1));
            end
        end
        
        function Train(obj, batch)
            results = obj.Apply(copy(batch));
            for i = 1:obj.training.epochs
                grads = obj.Backprop(results);
                obj.Update(results, grads);
                results = obj.Apply(copy(batch));
                for j = 1:length(obj.training.plots)
                    obj.training.plots_handles{j} = ...
                        obj.training.plots{j}(obj.training.plots_handles{j}, results, grads);
                end
                drawnow;
            end
        end
    end
end