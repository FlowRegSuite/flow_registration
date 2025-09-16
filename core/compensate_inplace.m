function [c_reg, w] = compensate_inplace(c1, c_ref, options)
%COMPENSATE_SIMPLE Method that compensates a workspace matrix inplace 
% and returns the compensated recording

    if nargin < 3
        options = OF_options();
    end

    squ = false;
    if length(size(c1)) == 3 && length(c_ref) == 2
        s = size(c1);
        c1 = reshape(c1, s(1), s(2), 1, s(3));
        squ = true;
    end

    if strcmp(options.channel_normalization, 'separate')
        c_low = mat2gray_multichannel(imgaussfilt3_multichannel(...
            mat2gray(c1), ...
            options));
    else
        c_low = mat2gray(imgaussfilt3_multichannel(...
            mat2gray(c1), ...
            options));
    end

    min_ref = double(min(c1(:)));
    max_ref = double(max(c1(:)));

    if strcmp(options.channel_normalization, 'separate')
        c_ref_low = mat2gray_multichannel(...
            imgaussfilt3_multichannel(double(c_ref), options), c_low);
    else
        c_ref_low = imgaussfilt3_multichannel(double(c_ref), options);
        c_ref_low = (c_ref_low - min_ref) / (max_ref - min_ref);
    end

    weight = [];
    n_channels = size(c_low, 3);
    for i = 1:n_channels
        weight(:, :, i) = options.get_weight_at(i, n_channels);
    end

    w = get_displacements( ...
        c_low, c_ref_low, ...
        'sigma', 0.001, ...
        'weight', weight, ...
        'alpha', options.alpha, ...
        'levels', options.levels, ...
        'min_level', options.min_level, ...
        'eta', options.eta, ...
        'update_lag', options.update_lag, ...
        'iterations', options.iterations, ...
        'use_gpu', options.use_gpu, ...
        'a_smooth', options.a_smooth, 'a_data', options.a_data);

    if (~isempty(options.preproc_funct))
        for i = 1:size(c1, 3)
            tic;
            c1(:, :, i, :) = options.preproc_funct(squeeze(...
                c1(:, :, i, :)));            
            preproc_toc = toc;
            if (~options.verbose)
                fprintf('Pre Processing took %f seconds.\n', preproc_toc);
            end
        end
    end

    if isempty(options.output_typename)
        c_reg = compensate_sequence_uv( c1, ...
            c_ref, w);
    else
        c1 = cast(c1, options.output_typename);
        c_reg = compensate_sequence_uv( c1, ...
            c_ref, w);
    end
    if squ
        c_reg = squeeze(c_reg);
    end
end

