% Author   : Philipp Flotho
% Copyright 2021 by Philipp Flotho, All rights reserved.

classdef DS_file_reader < handle
    % utility class to wrap handling of the dataset name specification
    % for the file writers with multiple channels. 
    % formats are 'ch*_reg' for ch1_reg, ch2_reg, ...
    % cell {ch1_reg, ch2_reg} for the same output or 
    % 'ch1_reg' for single channel fixed dataset names
    
    properties (Access = protected)
        dataset_names;
        dimension_ordering;
    end
    
    methods (Access = protected)
        function obj = DS_file_reader(default_order, varargin)
            p = inputParser;
            addParameter(p, 'dataset_names', [], @(x) isstring(x) || ...
                ischar(x) || iscell(x));
            addParameter(p, 'dimension_ordering', default_order, @(x) ...
                isvector(x) & isnumeric(x));
            parse(p, varargin{:});
            
            obj.dataset_names = p.Results.dataset_names;
            obj.dimension_ordering = p.Results.dimension_ordering;
        end
        
        function process_datasets(obj, found_ds_names, idx_3D, dim_sum)
            
            % potential number of channels:
            n_channels = sum(idx_3D);
            idx_3D = find(idx_3D);
            
            % identifying the datasets:
            if isempty(obj.dataset_names)
                
                % generate list of dataset name candidates
                % will currently fail, if more 3D datasets 
                % with different size are detected:
                [dataset_candidates, dataset_candidates_idx] = ...
                    obj.unpack_ds(obj.known_datasetnames, n_channels);

                % search for matches known datasets, first match wins:
                obj.dataset_names = {};
                for i = 1:n_channels
                    dataset_name = found_ds_names{idx_3D(i)};
                    if any(strcmp(dataset_candidates, dataset_name))
                        obj.dataset_names{end + 1} = dataset_name;
                        tmp = find(strcmp(dataset_candidates, dataset_name));
                        if i > length(obj.known_dims)
                            obj.dimension_ordering = obj.known_dims{1}
                        else
                            obj.dimension_ordering = obj.known_dims{dataset_candidates_idx(tmp(1))};
                        end
                    end
                end
                if isempty(obj.dataset_names)
                    % find the ds with largest dimensions:
                    [dim_sum_sorted, dim_sum_sorted_idx] = sort(dim_sum);
                    tmp_idx = find(cumsum(diff(dim_sum_sorted) ~= 0));
                    if isempty(tmp_idx)
                        obj.dataset_names = found_ds_names;
                    else
                        obj.dataset_names = found_ds_names{1:dim_sum_sorted_idx(1:tmp_idx(1))};
                    end
                end
            end
            
            % testing datasets supplied to the constructor:
            if (isstring(obj.dataset_names) || ...
                    ischar(obj.dataset_names))
                obj.dataset_names = char(obj.dataset_names);
                if obj.dataset_names(1) == '/'
                    obj.dataset_names = obj.dataset_names(2:end);
                end
                % find matching ds names with * ch number:
                if contains(obj.dataset_names, '*')
                    tmp_ds_name = obj.dataset_names;
                    dataset_candidates = ...
                        obj.unpack_ds({tmp_ds_name}, n_channels);
                    obj.dataset_names = {};
                    for i = 1:n_channels
                        dataset_name = found_ds_names{idx_3D(i)};
                        if any(strcmp(dataset_candidates, dataset_name))
                            obj.dataset_names{end + 1} = dataset_name;
                        end
                    end
                    if isempty(obj.dataset_names)
                        error('No datasets of type %s could be found', ...
                            tmp_ds_name);
                    end
                else                    
                    obj.dataset_names = {obj.dataset_names};
                end
            end


        end
            
        
        function ds = get_ds_name(obj, channel_id, n_channels)
            ds = obj.get_ds_name_static(obj.dataset_names, channel_id, n_channels);
        end
    end
    
    methods (Static, Access = protected)
        
        function ds = get_ds_name_static(dataset_names, channel_id, n_channels)
            
            if ~isempty(dataset_names)
                if iscell(dataset_names)
                    assert(length(dataset_names) == n_channels);
                    ds = dataset_names{channel_id};
                else
                    if n_channels == 1
                        ds = dataset_names;
                    else
                        ds = DS_file_reader.split_ds(dataset_names, channel_id);
                    end
                end
            else
                ds = ['ch' num2str(channel_id)];
            end
        end
        
        function ds = split_ds(dataset_name, channel_id)
            tmp = split(dataset_name, '*');
            if length(tmp) == 1
                ds = [tmp{1} num2str(channel_id)];
            else
                ds = [tmp{1} num2str(channel_id) tmp{2}];
            end
        end
        
        function [ds_candidates, idx] = unpack_ds(ds_names, n_channels)
            ds_candidates = {};
            idx = [];
            for j = 1:length(ds_names)
                for i = 1:n_channels
                    if ~iscell(ds_names{j}) && ...
                            ~contains(ds_names{j}, '*')
                        ds_candidates{end+1} = ds_names{j};
                        idx(end+1) = i;
                    end
                    try
                        ds_candidates{end+1} = ...
                            DS_file_reader.get_ds_name_static(ds_names{j}, ...
                            i, n_channels);
                        idx(end+1) = i;
                    catch
                    end
                end
            end
        end
    end
end

