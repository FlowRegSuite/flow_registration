% OF_OPTIONS class to store options for a motion compensation 
% job. 

% Author   : Philipp Flotho
% Copyright 2021 by Philipp Flotho, All rights reserved.
classdef OF_options < handle & matlab.mixin.Copyable
    
    properties (Access = private)
        p;
        supported_extensions = {'.MDF', '.tif', '.tiff', ...
            '.mat'};
        ext_map = {'MDF', 'TIFSTACK', 'TIFSTACK', 'MAT'};
        quality_setting_old = 'quality';
    end
    
    properties (SetAccess = private, GetAccess = public)
        datatype = 'NONE';
        preproc_funct = []; % @(x) x;
    end
    
    properties
        input_file;
        output_path = 'results';
        output_format = 'MAT';
        channel_idx = [];
        output_file_name = [];
        output_file_writer = [];
        alpha = 1.5;
        weight = [0.5, 0.5];
        levels = 100;
        min_level = -1; % Min level overrides the quality setting!
        quality_setting = 'quality';
        eta = 0.8;
        update_lag = 5;
        iterations = 50;
        a_smooth = 1;
        a_data = 0.45;
        sigma = [1, 1, 0.1; ...
                 1, 1, 0.1];
        bin_size = 1;
        buffer_size = 400;
        verbose = false;
        reference_frames = 50:500;
        save_meta_info = true;
        save_w = false;
        save_valid_mask = false;
        output_typename = 'double';
        channel_normalization = 'joint';
        interpolation_method = 'cubic';
        cc_initialization = false;
        update_reference = false;
        update_initialization_w = true;
        n_references = 1;
        min_frames_per_reference = 20;
        naming_convention = 'default';
        save_valid_idx = false;
        % The following is for normalizing externally loaded references
        % differently:
        normalize_ref = false;
        use_gpu = false;
    end
    
    methods
        function obj = OF_options(varargin)
            p = inputParser;

            addParameter(p, 'output_path', 'results', @(x) obj.isStringOrChar(x));
            addParameter(p, 'output_format', obj.output_format);
            addParameter(p, 'input_file', []);            
            addParameter(p, 'weight', obj.weight, @isnumeric);
            addParameter(p, 'alpha', obj.alpha);
            addParameter(p, 'levels', obj.levels, @(x) isnumeric(x) && isscalar(x));
            addParameter(p, 'min_level', -1, @(x) isnumeric(x) && isscalar(x) && x >= 0); 
            addParameter(p, 'eta', obj.eta, @isnumeric);
            addParameter(p, 'update_lag', obj.update_lag, @isnumeric);
            addParameter(p, 'iterations', obj.iterations, @isnumeric);
            addParameter(p, 'reference_frames', obj.reference_frames);
            addParameter(p, 'a_smooth', obj.a_smooth, @isnumeric);
            addParameter(p, 'a_data', obj.a_data, @isnumeric);
            addParameter(p, 'sigma', obj.sigma, @isnumeric);
            addParameter(p, 'bin_size', obj.bin_size, @isnumeric);
            addParameter(p, 'buffer_size', obj.buffer_size, @isnumeric);
            addParameter(p, 'verbose', obj.verbose, @islogical);
            addParameter(p, 'save_meta_info', obj.save_meta_info, @islogical);
            addParameter(p, 'preproc_funct', obj.preproc_funct, ...
                @(x) isa(x, 'function_handle'))
            addParameter(p, 'output_typename', obj.output_typename, ...
                @(x) isempty(x) || exist(x, 'class'));
            addParameter(p, 'save_w', obj.save_w, ...
                @(x) isscalar(x) && islogical(x));
            addParameter(p, 'channel_normalization', obj.channel_normalization, ...
                @(x) obj.isStringOrChar(x) && any(strcmp(x, {'joint', 'separate'})))
            addParameter(p, 'quality_setting', obj.quality_setting, ...
                @(x) obj.isStringOrChar(x) &&  any(strcmp(x, ...
                {'quality', 'balanced', 'fast'})));
            addParameter(p, 'interpolation_method', obj.interpolation_method, ...
                @(x) obj.isStringOrChar(x) && any(strcmp(x, {'nearest', 'linear', 'cubic'})))
            addParameter(p, 'update_reference', obj.update_reference, ...
                @(x) islogical(x));
            addParameter(p, 'update_initialization_w', obj.update_initialization_w, ...
                @(x) islogical(x));
            addParameter(p, 'n_references', obj.n_references, ...
                @(x) isscalar(x) && x >= 1)
            addParameter(p, 'min_frames_per_reference', obj.min_frames_per_reference, ...
                @(x) isscalar(x) && x >= 1)
            addParameter(p, 'cc_initialization', obj.cc_initialization);
            addParameter(p, 'save_valid_mask', obj.save_valid_mask, ...
                @(x) isscalar(x) && islogical(x));
            addParameter(p, 'naming_convention', obj.naming_convention, ...
                @(x) obj.isStringOrChar(x) &&  any(strcmp(x, ...
                {'default', 'batch'})));
            addParameter(p, 'save_valid_idx', obj.save_valid_idx, ...
                @(x) isscalar(x) && islogical(x));

            addParameter(p, 'use_gpu', obj.use_gpu, @(x) islogical(x) && isscalar(x))

            parse(p, varargin{:});

            for i = 1:length(p.Parameters)
               obj.(p.Parameters{i}) = p.Results.(p.Parameters{i});
            end
            obj.quality_setting_old = p.Results.quality_setting;
            obj.p = p;
            
            if p.Results.min_level ~= -1
                obj.quality_setting = 'custom';
            end
        end
        
        %% custom setters and getters:
        function set.output_path(obj, output_path) 
            assert(isstring(output_path) | ischar(output_path))
            obj.output_path = output_path;
        end
        
        function set.input_file(obj, input_file)
            obj.check_input(input_file);
            obj.input_file = input_file;
        end
        
        function set.reference_frames(obj, reference_frames)
            obj.check_reference(reference_frames);
            obj.reference_frames = reference_frames;
        end
        
        function set.alpha(obj, alpha)
            obj.check_alpha(alpha);
            obj.alpha = alpha;
        end
        
        function set.sigma(obj, sigma)
            obj.check_sigma(sigma);
            obj.sigma = sigma;
        end
        
        function set.weight(obj, weight)
            obj.check_weight(weight);
            
            if isvector(weight)
                obj.weight = weight ./ sum(weight);
            else
                obj.weight = weight;
            end
        end
        
        function set.output_format(obj, output_format)
            if ~ischar(output_format) && ~isstring(output_format)
                error('output_format needs to be a string or character array!');
            end
            if ~any(strcmp(output_format, ...
                    {'TIFF', 'HDF5', 'MAT', 'MULTIFILE_TIFF', ...
                    'MULTIFILE_MAT', 'MULTIFILE_HDF5', 'CAIMAN_HDF5', 'BEGONIA'}))
                error('%s is no valid output format!', output_format);
            end
            obj.output_format = output_format;
        end
        
        function set.quality_setting(obj, quality_setting)
            assert(obj.isStringOrChar(quality_setting) && any(strcmp(quality_setting, ...
                {'quality', 'balanced', 'fast', 'custom'})));
            
            if ~strcmp(quality_setting, 'custom')
                obj.quality_setting_old = quality_setting;
            end
            obj.quality_setting = quality_setting;
        end
        
        function set.min_level(obj, min_level)
            assert(isscalar(min_level) && isnumeric(min_level));
            
            if min_level >= 0
                obj.quality_setting = 'custom';
            else
                obj.quality_setting = obj.quality_setting_old;
            end
            obj.min_level = min_level;
        end
        
        function min_level = get.min_level(obj)
            if obj.min_level == -1
                switch obj.quality_setting
                    case 'quality'
                        min_level = 0;
                    case 'balanced'
                        min_level = 4;
                    case 'fast'
                        min_level = 6;
                    case 'custom'
                        min_level = max(obj.min_level, 0);
                end
            else
                min_level = obj.min_level;
            end
        end
        
        
        function sigma = get_sigma_at(obj, i)
            if isvector(obj.sigma)
                sigma = obj.sigma;
                return;
            end
            if size(obj.sigma, 1) < i
                warning("sigma for channel %i not specified, using ch1", i);
                sigma = obj.sigma(1, :);
                return;
            end
            sigma = obj.sigma(i, :);
        end
        
        function weight = get_weight_at(obj, i, n_channels)
            if isvector(obj.weight)
                weight_channels = length(obj.weight);
            else
                weight_channels = size(obj.weight, 1);
            end
            if weight_channels < i
                if (~obj.verbose)
                    fprintf("Weight for channel %i not set, using default value 1/n_channels.\n", i);
                end
                weight = 1/n_channels;
                return;
            end
            if weight_channels > n_channels && isvector(obj.weight)
                obj.weight = obj.weight(1:n_channels);
            end
                
            if (size(obj.weight, 3) == 1)
                weight = obj.weight(i);
            else
                weight = obj.weight(i, :, :);
            end
        end
        
        function input_file = get.input_file(obj)
            input_file = obj.input_file;
        end
        
        function output_path = get_output_path(obj)
            output_path = obj.output_path;
        end
        
        %% utility functions
        function video_file_reader = get_video_file_reader(obj)
            if isa(obj.input_file, 'Video_file_reader')
                video_file_reader = obj.input_file;
                return;
            end
            
            video_file_reader = get_video_file_reader(obj.input_file, ...
                obj.buffer_size, obj.bin_size);

            obj.input_file = video_file_reader;
            
%             datatypes = Datatypes;
%             switch obj.datatype
%                 case datatypes.NONE
%                     error('No valid input file in the options set!');
%                 case datatypes.TIFSTACK
%                     video_file_reader = ...
%                         TIFSTACK_file_reader(obj.input_file, ...
%                         obj.buffer_size, obj.bin_size);
%                 case datatypes.MDF
%                     video_file_reader = MDF_file_reader(obj.input_file, ...
%                         obj.buffer_size, obj.bin_size);
%                 case datatypes.VIDEO
%                     video_file_reader = AVI_file_reader(obj.input_file, ...
%                         obj.buffer_size, obj.bin_size);
%                 case datatypes.MATRIX
%                     video_file_reader = MATRIX_file_reader(obj.input_file, ...
%                         obj.buffer_size, obj.bin_size);
%                 case datatypes.MULTICHANNEL
%                     video_file_reader = MULTICHANNEL_file_reader(...
%                         obj.input_file, obj.buffer_size, obj.bin_size);
%                 case IMG
%                     video_file_reader = IMG_file_reader(obj.input_file, ...
%                         obj.buffer_size, obj.bin_size);
%             end
        end
        
        function video_writer = get_video_writer(obj)
            if ~isempty(obj.output_file_writer) && ...
                    isa(obj.output_file_writer, 'Video_file_writer')
                video_writer = obj.output_file_writer;
                return;
            end
            
            if ~isempty(obj.output_file_name)
%                 [result, idx] = obj.check_extension(ext);
                filename = obj.output_file_name;
            else
                if strcmp(obj.naming_convention, "default")
                    filename = fullfile(obj.output_path, ['compensated.' ...
                        obj.output_format]);
                else
                    reader = obj.get_video_file_reader;
                    filename = fullfile(obj.output_path, ...
                        strcat(reader.input_file_name, '_', 'compensated.', ...
                        obj.output_format));
                end
            end

            input_args = {};
            if (isa(obj.get_video_file_reader, 'MDF_file_reader') || ...
                isa(obj.get_video_file_reader, 'begonia_file_reader'))  && ...
                    strcmp(obj.output_format, 'BEGONIA')
                disp('Found metadata for TSERIES format...');
                input_args{end+1} = 'mdf_reference';
                input_args{end+1} = obj.get_video_file_reader;
            end
            
            video_writer = get_video_file_writer(filename, ...
                obj.output_format, input_args{:});
        end
        
        function references = get_multi_reference_frames(obj, video_file_reader)

            references = {};

            if iscell(obj.reference_frames)
                for image = self.reference_frames
                    references{end + 1} = imread(image);
                end
            end

            options = copy(obj);
            options.n_references = 1;
            reference = obj.reference_frames;
            if nargin < 2
                return;
            end
            if isvector(obj.reference_frames)
                vid = double(video_file_reader.read_frames(reference));
                ref = mean(vid, 4);
                c_reg = compensate_inplace(vid, ref, options);
                ref = mean(c_reg, 4);

                energy = zeros(1, size(vid, 4));
                parfor i = 1:size(vid, 4)
                    energy(i) = get_energy(c_reg(:, :, :, i), ref);
                end
                idx = kmeans(medfilt1(energy', 20, 'truncate'), obj.n_references);

                for i = 1:obj.n_references
                    references{end + 1} = squeeze(mean(c_reg(:, :, :, find(idx == i)), 4));
                end
            end
        end

        % returning the reference or preregistering the reference frames
        function reference = get_reference_frame(obj, video_file_reader)
            if obj.n_references > 1
                reference = obj.get_multi_reference_frames(video_file_reader);
                return;
            end

            if isstring(obj.reference_frames) || ...
                    ischar(obj.reference_frames)
                reference = imread(obj.reference_frames);
                obj.normalize_ref = true;
                return;
            end
            reference = obj.reference_frames;
            if nargin < 2
                obj.normalize_ref = true;
                return;
            end
            assert(isa(video_file_reader, 'Video_file_reader'));
            if isvector(obj.reference_frames)
                tmp = double(video_file_reader.read_frames(reference));
                if size(tmp, 4) == 1
                    reference = tmp;
                    return;
                end
                
                n_channels = video_file_reader.n_channels;
                
%                 % for compatibility
%                 if verLessThan('matlab', '9.10')
%                     size_tmp = size(tmp);
%                     weight_2d = zeros(size_tmp(1:3));
%                 else
%                     weight_2d = zeros(size(tmp, 1:3));
%                 end
%                 for i = 1:n_channels
%                     weight_2d(:, :, 1) = obj.get_weight_at(i, n_channels);
%                 end
                weight_2d = [];
                for i = 1:n_channels
                    weight_2d(:, :, i) = obj.get_weight_at(i, n_channels);
                end

                if ~obj.verbose
                    disp('Preregistering reference frames...');
                end
                
                if strcmp(obj.channel_normalization, 'separate')
                    c1 = mat2gray_multichannel(imgaussfilt3_multichannel(tmp, obj, [1 1 0.5]));
                else
                    c1 = mat2gray(imgaussfilt3_multichannel(tmp, obj, [1 1 0.5]));
                end
                c_reg_tmp = compensate_sequence( ...
                    c1, mean(c1, 4),  ...
                    tmp, mean(tmp, 4), ...
                    'weight', weight_2d, ...
                    'alpha', obj.alpha + 2, ...
                    'levels', obj.levels, ...
                    'min_level', obj.min_level, ...
                    'eta', obj.eta, ...
                    'update_lag', obj.update_lag, ...
                    'iterations', obj.iterations, ...
                    'use_gpu', obj.use_gpu, ...
                    'a_smooth', obj.a_smooth, 'a_data', obj.a_data);
                                
                reference = mean(c_reg_tmp, 4);
                obj.reference_frames = reference;
                
                if ~obj.verbose
                    disp('Finished pre-registration of the reference frames...');
                end
            end
        end
        
        %% and saving 
        function load_options(obj, settings_file)
        %LOAD_OPTIONS Initializes the options file from a previously
        % stored one.
        % LOAD_OPTIONS(OBJ, SETTINGS_FILE) Loads the file SETTINGS_FILE. 
        % 
        % See also SAVE_OPTIONS
        
            assert(isfile(settings_file));
            
            fprintf("Loading ");
            fileID = fopen(settings_file, 'r');
            
            disp(fgetl(fileID));
%             fgetl(fileID);
            
            json_text = "";
            while ~feof(fileID)
                tmp = fgets(fileID);
                json_text = append(json_text, tmp);
            end

            
            output_obj = jsondecode(json_text);
            vars = fieldnames(output_obj);
            fprintf("Setting parameters ");
            for i = 1:length(vars)
                try
                    obj.(vars{i}) = output_obj.(vars{i});
                    if i < length(vars)
                        fprintf("%s, ", vars{i});
                    else
                        fprintf("%s.\n", vars{i});
                    end
                catch
                    warning("Could not parse parameter %s", vars{i});
                end
            end
            fclose(fileID);
        end
        
        
        function save_options(obj, settings_path)
        % SAVE_OPTIONS Stores the current OF_option object. 
        %
        % Matlab matrix style input data is not stored, matrix style 
        % reference frames are first saved into a multipage tiff 
        % with the filepath stored in the options file. 
        %
        % See also LOAD_OPTIONS
            
            if (nargin < 2)
                settings_path = fullfile(obj.output_path, 'options.json');
            end
            
            [settings_folder, ~, ~] = fileparts(settings_path);
            if (~exist(settings_folder, 'dir'))
                mkdir(settings_folder);
            end
            
            if ~(isstring(obj.input_file) || ischar(obj.input_file))
                output_obj.input_file = "";
            else
                output_obj.input_file = obj.input_file;
            end            
            
            vars = fieldnames(obj);
            for i = 1:length(vars)
                if (strcmp(vars{i}, 'reference_frames') || ...
                        strcmp(vars{i}, 'preproc_funct'))
                    continue
                end
                output_obj.(vars{i}) = obj.(vars{i});
            end
            
            % handling the reference frames to avoid storing large 
            % data in the settings files
            if  isvector(obj.reference_frames) || ...
                    isstring(obj.reference_frames) || ...
                    ischar(obj.reference_frames)
                output_obj.reference_frames = obj.reference_frames;
            else
                img = obj.reference_frames;
                if (size(img(1, 1, 1, :), 4) > 1)
                    obj.reference_warning();
                end
                reference_frame_path = fullfile(settings_folder, ...
                    'reference_frames.tif');

                vid_writer = get_video_file_writer(reference_frame_path, 'TIFF');
                vid_writer.write_frames(img);
                vid_writer.close;
                output_obj.reference_frames = reference_frame_path;
            end

            fileID = fopen(settings_path, 'w');
            
            formatOut = 'yyyy-mm-dd';
            fprintf(fileID, 'Compensation options %s\n\n', datestr(now, formatOut));
            
            
            if verLessThan('matlab', '9.10')
                output_obj_json = jsonencode(output_obj);
            else
                output_obj_json = jsonencode(output_obj, ...
                    'PrettyPrint', true);
            end
            
            fprintf(fileID, output_obj_json, true);

            fclose(fileID);
        end
    end
    
    methods (Access = private)
        
%         function [result, idx] = check_extension(obj, ext)
%             result = false;
%             for i = 1:length(obj.supported_extensions)
%                 result = ...
%                     result | strcmpi(ext, obj.supported_extensions{i});
%                 if result
%                     idx = i;
%                     return
%                 end
%             end
%         end
        
        function obj = check_reference(obj, x)
            if ~(isstring(x) || ischar(x))
                if size(x(1, 1, 1, :), 4) > 1
                    obj.reference_warning();
                end
                return;
            end
        end
        
        function obj = check_input(obj, x)
            
            datatypes = Datatypes;
            
            if (isa(x, 'Video_file_reader'))
                obj.datatype = x.datatype;
                x.buffer_size = obj.buffer_size;
                x.bin_size = obj.bin_size;
                return;
            end
            
            if (isempty(x))
                obj.datatype = datatypes.NONE;
                return;
            end
%             if ~(isstring(x) || ischar(x))
%                 obj.datatype = datatypes.MATRIX;
%                 if size(x(1, 1, 1, :), 4) > 1
%                     obj.reference_warning();
%                 end
%                 return;
%             end
            if iscell(x) && (isstring(x{1}) || ischar(x{1}))
                
%                 is_multichannel = true;
%                 for i = 1:length(x)
%                     [~, ~, ext] = fileparts(x{i});
% 
%                     is_multichannel = is_multichannel & check_extension(ext);
%                 end        
                obj.datatype = datatypes.MULTICHANNEL;
                return;
            end

            if (exist(x, 'file') == 2)
                [~, ~, ext] = fileparts(x);

                [result, ext] = check_extension(ext);
                if ~result
                    result = H5F.is_hdf5(x);
                    if result
                        obj.datatype = datatypes.HDF5;
                    else
                        try
                            VideoReader(x);
                            obj.datatype = datatypes.VIDEO;
                            result = true;
                        catch
                        end
                    end
                else
                    obj.datatype = ext;
                end
                
                if (~result)
                    error('File format currently not supported!');
                end
            end
            if isfolder(x)
                % checking for IMG files:
                dir_content = dir(x);
                % checking if at least one image present:
                IMG_ready = false;
                counter = 1;
                while (counter < length(dir_content))
                    try
                        imfinfo(fullfile(...
                            dir_content(counter).folder, ...
                            dir_content(counter).name));
                        IMG_ready = true;
                        break;
                    catch
                    end
                    counter = counter + 1;
                end
                if (~IMG_ready)
                    warning('Folder does not contain images!');
                end
                obj.datatype = datatypes.IMG;
            end
        end
        
        function check_alpha(~, x)
            if (~isnumeric(x))
                error("Alpha must be numeric!");
            end
            if (~isvector(x) || length(x) > 2)
                error("Alpha must be a scalar or two element vector!");
            end
            if x <= 0
                error("Alpha must be positive!");
            end
        end
        
        function check_sigma(~, x)
            if (~isnumeric(x))
                error("Sigma must be numeric!");
            end
%             if (~isvector(x) || length(x) ~= 3)
%                 error("Sigma must be a three element vector!");
%             end
            if (size(x, 2) ~= 3) || (size(x(1, 1, :), 4) > 1)
                error("Sigma must be a n by 3 matrix for n channels!");
            end
            if x <= 0
                error("Alpha must be positive!");
            end
        end
        
        function check_weight(~, x)
            if (~isnumeric(x))
                error("Weight must be numeric!");
            end
            if size(x, 4) > 1
                error("Weight must be a 1D or 3D array with the first dimension being the channel number!");
            end
        end
        
        function check_bins(~, x)
            if (~isnumeric(x) || ~isscalar(x) || ...
                    x <= 0 || ~(mod(x, 1) == 0))
                error("The number of bins must be a positive integer!");
            end
        end
        
        function check_buffer(~, x)
            if (~isnumeric(x) || ~isscalar(x) || ...
                    x <= 0 || ~(mod(x, 1) == 0))
                error("The buffer size must be a positive integer!");
            end
        end
        
        function check_iterations(~, x)
            if (~isnumeric(x) || ~isscalar(x) || ...
                    x <= 0 || ~(mod(x, 1) == 0))
                error("Pyramid levels, number of iterations and update lag must be a positive integers!");
            end
        end
    end
    
    methods(Access = private, Static)
        function reference_warning()
            warning("Reference frames in matrix format should be 2D or 3D matrices where the third dimension is the channel number!");
        end
        
        function result = isStringOrChar(x)
            result = isstring(x) || (isvector(x) && ischar(x));
        end
    end
end



