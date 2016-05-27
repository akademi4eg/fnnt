classdef Batch < matlab.mixin.Copyable
    % Batch Holds a batch for training.
    
    properties (SetAccess = protected)
        data = []; % Features for training.
        labels = []; % Optional labels for training.
    end
    
    methods
        function obj = Batch(varargin)
            % Construct Batch object. Takes either data or data and labels
            % as inputs.
            if nargin < 1
                error('Please pass at least inputs to create a batch.');
            elseif nargin == 1 % only data
                obj.data = varargin{1};
            else % data and labels
                obj.data = varargin{1};
                obj.labels = varargin{2};
                assert(size(obj.data, 2) == size(obj.labels, 2));
            end
        end
        
        function bsize = GetBatchSize(obj)
            % Returns size of the batch.
            bsize = size(obj.data, 2);
        end
        
        function swidth = GetSampleWidth(obj)
            % Returns length of one sample.
            swidth = size(obj.data, 1);
        end
        
        function TransformData(obj, fun)
            % Applies inplace fun() transform to data.
            obj.data = fun(obj.data);
        end
        
        function TransformLabels(obj, fun)
            % Applies inplace fun() transform to labels.
            if ~ishandle(fun), error('Please pass function handler.'); end
            obj.labels = fun(obj.labels);
        end
        
        function data = GetDataAsMatrix(obj)
            % Returns copy of batch data as matrix.
            data = obj.data;
        end
        
        function labels = GetLabelsAsMatrix(obj)
            % Returns copy of batch labels as matrix.
            labels = obj.labels;
        end
    end
end