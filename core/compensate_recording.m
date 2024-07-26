% Author   : Philipp Flotho
% Copyright 2021 by Philipp Flotho, All rights reserved.

function reference_frame = compensate_recording(options, ...
    reference_frame)

    if (~exist(options.output_path, 'dir'))
        mkdir(options.output_path);
    end
        
    if (~options.verbose)
        fprintf('\nStarting compensation with quality setting %s and min_level = %i.\n', ...
            options.quality_setting, options.min_level);
        fprintf('Set the quality setting to fast, balanced or quality or increase min_level for fast, approximated solutions.\n');
        
        fprintf('Output format is %s', options.output_format);
        if ~strcmp(options.output_format, 'HDF5')
            fprintf(', change to HDF5 for faster writing performance.');
        else
            fprintf('.');
        end
        fprintf('\n\n');
    end

    if options.n_references > 1
        if nargin < 2
            compensate_multi_ref_recording(options);
        else
            compensate_multi_ref_recording(options, reference_frame);
        end
        return
    end

    video_file_reader = options.get_video_file_reader;
    n_channels = video_file_reader.n_channels;
    video_file_writer = get_video_writer(options);
    
    if (options.save_w)
        w_file_writer = get_video_file_writer(fullfile(options.output_path, 'w.hdf'), ...
            'HDF5', 'dataset_names', {'u', 'v'});
    end
    if options.save_valid_idx
        idx_file_writer = get_video_file_writer(fullfile(options.output_path, 'idx.hdf'), ...
            'HDF5', 'dataset_names', {'idx'});
    end
    
    if nargin < 2
        c_ref_raw = double(options.get_reference_frame(video_file_reader));
        reference_frame = c_ref_raw;
    else
        options.normalize_ref = true;
        c_ref_raw = reference_frame;
    end
    
    % setting the channel weight
    weight = [];
    for i = 1:n_channels
        weight(:, :, i) = options.get_weight_at(i, n_channels);
    end
    
    mean_div = [];
    mean_translation = [];
    mean_disp = [];
    max_disp = [];
    
    % motion compensation:
    i = 0;
    while(video_file_reader.has_batch())
        i = i + 1;
        buffer = video_file_reader.read_batch();
    
        % generating first w_init:
        clear c1;
        if i == 1
            
            if strcmp(options.channel_normalization, 'separate')
                c1 = mat2gray_multichannel(imgaussfilt3_multichannel(...
                    mat2gray(buffer(:, :, :, 1:min(22, size(buffer, 4)))), ...
                    options));
            else
                c1 = mat2gray(imgaussfilt3_multichannel(...
                    mat2gray(buffer(:, :, :, 1:min(22, size(buffer, 4)))), ...
                    options));
            end
            
            tmp = imgaussfilt3_multichannel(buffer, options);
            min_ref = double(min(tmp(:)));
            max_ref = double(max(tmp(:)));

            if options.normalize_ref
                fprintf("\nNormalizing the reference with respect to the video due to externally provided reference.\n\n")
                
                filtered_c_ref = imgaussfilt3_multichannel(c_ref_raw, options);
                
                for k = size(filtered_c_ref, 3)

                    tmp_ch = tmp(:, :, k, :);
                    mean_buffer = mean(double(tmp_ch(:)));
                    std_buffer = std(double(tmp_ch(:)));

                    ref_ch = filtered_c_ref(:, :, k, :);

                    mean_c_ref = mean(ref_ch(:));
                    std_c_ref = std(ref_ch(:));
                    
                    standardized_c_ref = (c_ref_raw(:, :, k, :) - mean_c_ref) / std_c_ref;
                    
                    c_ref_raw(:, :, k, :) = standardized_c_ref * std_buffer + mean_buffer;
                end
            end
            
            if strcmp(options.channel_normalization, 'separate')
                c_ref = mat2gray_multichannel(...
                    imgaussfilt3_multichannel(c_ref_raw, options), tmp);
            else
                c_ref = imgaussfilt3_multichannel(c_ref_raw, options);
                c_ref = (c_ref - min_ref) / (max_ref - min_ref);
            end
            clear tmp;

            if options.cc_initialization
                w_init = get_displacements_cc(c1, c_ref);
                w_init = mean(w_init, 4);
            else
                w_init = zeros(size(c_ref, 1), size(c_ref, 2), 2, "double");
            end

            w_init = get_displacements( ...
                c1, c_ref, ...
                    'sigma', 0.001, ...
                    'weight', weight, ...
                    'alpha', options.alpha, ...
                    'uv', w_init(:, :, 1), w_init(:, :, 2), ...
                    'levels', options.levels, ...
                    'min_level', options.min_level, ...
                    'eta', options.eta, ...
                    'update_lag', options.update_lag, ...
                    'iterations', options.iterations, ...
                    'a_smooth', options.a_smooth, 'a_data', options.a_data);
                
            w_init = mean(w_init, 4);

            if ~options.verbose
                fprintf('Done pre-registration to get w_init.\n');
            end
        end

        if ~options.update_initialization_w
            w_init = zeros(size(w_init), 'double');
        end

        [mean_disp_tmp, max_disp_tmp, mean_div_tmp, mean_translation_tmp, ...
            c_reg, c_ref_upd, w_init, w, idx_valid] ...
            = get_eval(options, buffer, c_ref, c_ref_raw, w_init, weight);

        if options.update_reference
            c_ref = c_ref_upd;
        end

        mean_div(end+1:end+length(mean_div_tmp)) = mean_div_tmp;
        mean_translation(end+1:end+length(mean_div_tmp)) = mean_translation_tmp;
        mean_disp(end+1:end+size(c_reg, 4)) = mean_disp_tmp;
        max_disp(end+1:end+size(c_reg, 4)) = max_disp_tmp;

        video_file_writer.write_frames(c_reg);
        
        if (options.save_w)
            w_file_writer.write_frames(w);
        end
        if options.save_valid_idx
            idx_file_writer.write_frames(idx_valid);
        end
        clear w
            
        if (~options.verbose)
            fprintf('Finished batch %i, %i batches left.\n', i, video_file_reader.batches_left());
        end
    end

    video_file_writer.close();

    if strcmp(options.naming_convention, "default")
        filename_pref = "";
    else
        filename_pref = strcat(video_file_reader.input_file_name, "_");
    end

    if options.save_meta_info
        save(fullfile(options.output_path, strcat(filename_pref, 'statistics.mat')), ...
            'mean_disp', 'max_disp', 'mean_div', 'mean_translation');
        save(fullfile(options.output_path, strcat(filename_pref, 'reference_frame.mat')), ...
            'c_ref_raw');
        
        mapping = multispectral_mapping(c_ref_raw);
        imwrite(mapping, fullfile(options.output_path, strcat(filename_pref, 'combined_ref.png')));
    end
end

function compensate_multi_ref_recording(options, ...
    reference_frames)

    if options.n_references > 2
        error("more than two references are currently not supported");
    end

    if (~options.verbose)
        fprintf("Compensating in multireference mode with %i min frames per reference. \n", options.min_frames_per_reference);
    end
    
    if nargin < 2
        if (~options.verbose)
            fprintf("Starting estimation of the best references...\n");
        end
        reference_frames = options.get_reference_frame(options.get_video_file_reader());
    end

    if ~iscell(reference_frames)
        error("Reference frames need to be in a cell array to work with the multireference mode!")
    end
    
    %% metric computation (fast run with parameters depending on min_frames_per_reference):
    ref = reference_frames{1};
    
    options_fast = copy(options);
    options_fast.n_references = 1;
    options_fast.quality_setting = 'fast';

    video_file_reader = options_fast.get_video_file_reader();

    energy = [];

    i = 0;
    if (~options.verbose)
        fprintf('Starting estimation of the energy based metric over %i frames...\n', ...
            video_file_reader.frame_count / video_file_reader.bin_size);
        tic;
    end
    while(video_file_reader.has_batch())
        i = i + 1;
        buffer = video_file_reader.read_batch();

        c_reg = compensate_inplace(buffer, ref, options_fast);
        
        for j = 1:size(buffer, 4)
            energy(end+1) = get_energy(c_reg(:, :, :, j), ref);
        end
    end
    if (~options.verbose)
        fprintf('Finished metric estimation (%i seconds).\n', toc);
    end
    energy_med = medfilt1(energy', options.min_frames_per_reference);
    idx = kmeans(medfilt1(energy_med, options.min_frames_per_reference), options.n_references);

    energies = [];
    for i = 1:options.n_references
        energies(end + 1) = mean(energy(idx == i));
    end
    [~, min_idx] = min(energies);
    if min_idx ~= 1
        idx = -idx + max(idx) + 1;
    end

    video_file_reader = options.get_video_file_reader();
    video_file_reader.reset();
    video_file_writer = get_video_writer(options);

    i = 1;
    batch_counter = 1;
    mean_disp = [];
    max_disp = [];

    mean_div = [];
    mean_translation = [];

    while(video_file_reader.has_batch())

        buffer = video_file_reader.read_batch();

        if isempty(options.output_typename)
            c_reg = zeros(size(buffer));
        else
            c_reg = zeros(size(buffer), options.output_typename);
        end
        mean_disp_buffer = zeros(1, size(buffer, 4));
        max_disp_buffer = zeros(1, size(buffer, 4));
        mean_div_buffer = zeros(1, size(buffer, 4));
        mean_translation_buffer = zeros(1, size(buffer, 4));        

        idx_tmp = idx(i:i+size(buffer, 4)-1);
        i = i + size(buffer, 4);

        for j = 1:options.n_references
            idx_j = find(idx_tmp == j);
            if isempty(idx_j)
                continue;
            end
            [c_reg(:, :, :, idx_j), w] = compensate_inplace(buffer(:, :, :, idx_j), reference_frames{j}, options);
            mean_disp_buffer(idx_j) = squeeze(mean(mean(sqrt(w(:, :, 1, :).^2 + w(:, :, 2, :).^2), 1), 2));
            max_disp_buffer(idx_j) = squeeze(max(max(sqrt(w(:, :, 1, :).^2 + w(:, :, 2, :).^2), [], 1), [], 2));
            
            mean_div_buffer(idx_j) = get_mean_divergence(w);
            mean_translation_buffer(idx_j) = get_mean_translation(w);
        end
        video_file_writer.write_frames(c_reg);
        mean_disp(end+1:end+length(mean_disp_buffer)) = mean_disp_buffer;
        max_disp(end+1:end+length(mean_disp_buffer)) = max_disp_buffer;
        mean_div(end+1:end+length(mean_disp_buffer)) = mean_div_buffer;
        mean_translation(end+1:end+length(mean_disp_buffer)) = mean_translation_buffer;
        if (~options.verbose)
            fprintf('Finished batch %i, %i batches left.\n', batch_counter, video_file_reader.batches_left());
        end
        batch_counter = batch_counter + 1;
    end
    video_file_writer.close();

    if strcmp(options.naming_convention, "default")
        filename_pref = "";
    else
        filename_pref = strcat(video_file_reader.input_file_name, "_");
    end

    if options.save_meta_info
        save(fullfile(options.output_path, strcat(filename_pref, 'statistics.mat')), ...
            'mean_disp', 'max_disp', 'mean_div', 'mean_translation', ...
            "idx", "energy");
        save(fullfile(options.output_path, strcat(filename_pref, 'reference_frame.mat')), ...
            "reference_frames");
    end
end

function [mean_disp, max_disp, mean_div, mean_translation, ...
    c_reg, c_ref, w_end, w, idx_valid] ...
    = get_eval(options, buffer, c_ref, c_ref_raw, w_init, weight)
    
    if strcmp(options.channel_normalization, 'separate')
        c1 = mat2gray_multichannel(imgaussfilt3_multichannel(mat2gray(buffer), options));
    else
        c1 = mat2gray(imgaussfilt3_multichannel(mat2gray(buffer), options));
    end
    
    if nargin < 3
        c_ref = mean(c1, 4);
    end
    
    if nargin < 4
        w_init = zeros(size(c_ref, 1), size(c_ref, 2), 2);
    end
    
    tic
    if (~options.verbose)
        fprintf('Starting registration\n');
    end

    if options.cc_initialization
        w_cc = get_displacements_cc(c1, c_ref);
        c1 = compensate_sequence_uv(c1, ...
            c_ref, w_cc);
    else
        w_cc = zeros(size(w_init));
    end

    w = get_displacements( ...
        c1, c_ref, ...
        'sigma', 0.001, ...
        'weight', weight, ...
        'uv', w_init(:, :, 1), w_init(:, :, 2), ...
        'alpha', options.alpha, ...
        'levels', options.levels, ...
        'min_level', options.min_level, ...
        'eta', options.eta, ...
        'update_lag', options.update_lag, ...
        'iterations', options.iterations, ...
        'a_smooth', options.a_smooth, 'a_data', options.a_data) + w_cc;
    clear w_cc;
    of_toc = toc;
    if (~options.verbose)
        fprintf('Displacement estimation took %f seconds.\n', of_toc);
    end
    
    mean_disp = squeeze(mean(mean(sqrt(w(:, :, 1, :).^2 + w(:, :, 2, :).^2), 1), 2));
    max_disp = squeeze(max(max(sqrt(w(:, :, 1, :).^2 + w(:, :, 2, :).^2), [], 1), [], 2));
    
    mean_div = get_mean_divergence(w);
    mean_translation = get_mean_translation(w);
    
    n_ref_frames = min(100, size(c1, 4) - 1);
    for c = 1:size(c_ref, 3)
        c_ref(:, :, c) = mean(...
            compensate_sequence_uv( double(c1(:, :, c, end-n_ref_frames:end)), ...
            mean(double(c1(:, :, c, :)), 4), w(:, :, :, end-n_ref_frames:end), ...
            options.interpolation_method), 4);
    end
    clear c1;
    
    if (~isempty(options.preproc_funct))
        for i = 1:size(buffer, 3)
            tic;
            buffer(:, :, i, :) = options.preproc_funct(squeeze(...
                buffer(:, :, i, :)));            
            preproc_toc = toc;
            if (~options.verbose)
                fprintf('Pre Processing took %f seconds.\n', preproc_toc);
            end
        end
    end
       
    if isempty(options.output_typename)
        [c_reg, idx_valid] = compensate_sequence_uv( buffer, ...
            c_ref_raw, w);
    else
        buffer = cast(buffer, options.output_typename);
        [c_reg, idx_valid] = compensate_sequence_uv( buffer, ...
            c_ref_raw, w);
    end
    
    if size(c_reg, 4) > 100
        w_end = mean(w(:, :, :, end-20:end), 4);
    else
        w_end = mean(w, 4); %w_init;
    end
end

function divergence = get_mean_divergence(w)

    divergence = zeros(1, size(w, 4));
    
    u = w(:, :, 1, :);
    v = w(:, :, 2, :);

    parfor i = 1:size(w, 4)
        [w_x, ~] = imgradientxy(u(:, :, 1, i));
        [~, w_y] = imgradientxy(v(:, :, 1, i));
        
        divergence(i) = mean(mean(w_x + w_y, 1), 2);
    end
end

function mean_translation = get_mean_translation(w)
    
    u = mean(mean(w(:, :, 1, :), 1), 2);
    v = mean(mean(w(:, :, 2, :), 1), 2);
    
    mean_translation = squeeze(sqrt(u.^2 + v.^2));
end
