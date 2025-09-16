% Author   : Philipp Flotho
% Copyright 2021 by Philipp Flotho, All rights reserved.

function w = get_displacement( fixed, moving, varargin )
% computes the displacements

    alpha = [2, 2];
    update_lag = 10;
    iterations = 20;
    
    min_level = 0;
        
    levels = 50;
    eta = 0.75;
    
    a_smooth = 0.5;
    
    [m, n, n_channels] = size(fixed);
    
    u_init = zeros(m, n);
    v_init = zeros(m, n);
    
    weight = ones(1, n_channels, 'double') / n_channels;
    
    a_data = 0.45 * ones(1, n_channels);
    use_gpu = false;
    
    for k = 1:length(varargin)
        if ~isa(varargin{k}, 'char')
            continue;
        end
        switch varargin{k}
            case 'weight'
                weight = varargin{k + 1};
            case 'alpha'
                alpha = varargin{k + 1};
                if length(alpha) == 1
                    alpha = alpha .* ones(1, 2);
                end
            case 'eta'
                eta = varargin{k + 1};
            case 'levels'
                levels = varargin{k + 1};       
            case 'update_lag'
                update_lag = varargin{k + 1};
            case 'iterations'
                iterations = varargin{k + 1};
            case 'uv'
                u_init = varargin{k + 1};
                v_init = varargin{k + 2};
            case 'a_data'
                a_data = varargin{k + 1};
                if (length(a_data) == 1)
                    a_data = a_data * ones(1, n_channels);
                end
            case 'a_smooth'
                a_smooth = varargin{k + 1};
            case 'min_level'
                min_level = varargin{k + 1};
            case 'use_gpu'
                use_gpu = logical(varargin{k+1});
            otherwise
                % fprintf(['could not parse input argument ' varargin{k} '\n']);
        end
    end
        

    f1_low = double(fixed);
    f2_low = double(moving);
    
    method = 'bicubic';
    
    max_level_y = warpingDepth(eta, levels, m, m);
    max_level_x = warpingDepth(eta, levels, n, n);
    
    max_level = min(max_level_x, max_level_y) * 4;
    
    max_level_y = min(max_level_y, max_level);
    max_level_x = min(max_level_x, max_level);
    
    local_weight = ndims(weight) == ndims(fixed) && sum(size(weight) == size(fixed)) == ndims(fixed);
    weight_level = weight;
    
    if max(max_level_x, max_level_y) <= min_level
        min_level = max(max_level_x, max_level_y) - 1;
    end
    if min_level < 0
        min_level = 0;
    end
        
    for i = max(max_level_x, max_level_y):-1:min_level       
        level_size = round([m * eta^(min(i, max_level_y)), ...
            n * eta^(min(i, max_level_x))]);
        
        f1_level = imresize(f1_low, ...
            level_size, ...
            method, 'Colormap', 'original', 'Antialiasing', true);
        f2_level = imresize(f2_low, ...
            level_size, ...
            method, 'Colormap', 'original', 'Antialiasing', true);
        
        if local_weight
            weight_level = padarray(imresize(weight, ...
                level_size, ...
                method, 'Colormap', 'original', 'Antialiasing', true), ...
                [1 1], 0.0);
        end
        
        hx = m / size(f1_level, 1);
        hy = n / size(f1_level, 2);
        
        if i == max(max_level_x, max_level_y)
            u = add_boundary(imresize(u_init, level_size, method, 'Colormap', 'original')); 
            v = add_boundary(imresize(v_init, level_size, method, 'Colormap', 'original')); 
            tmp = double(f2_level);
        else
            u = add_boundary(imresize(u(2:end - 1, 2:end - 1), level_size, method, 'Colormap', 'original'));
            v = add_boundary(imresize(v(2:end - 1, 2:end - 1), level_size, method, 'Colormap', 'original'));
            try
                tmp = imregister_wrapper(double(f2_level), ...
                    double(u(2:end-1, 2:end-1)) / hx, ...
                    double(v(2:end-1, 2:end-1)) / hy, double(f1_level));
            catch err
                disp(err.message);
                error("Error using imregister for compensating flow increments, try increasing alpha!");
            end
        end
        
        J11 = zeros([level_size + 2, n_channels]);
        J22 = zeros([level_size + 2, n_channels]);
        J33 = zeros([level_size + 2, n_channels]);
        J12 = zeros([level_size + 2, n_channels]);
        J13 = zeros([level_size + 2, n_channels]);
        J23 = zeros([level_size + 2, n_channels]);
        for j = 1:n_channels
            [ J11(:, :, j), J22(:, :, j), J33(:, :, j), ...
                J12(:, :, j), J13(:, :, j), J23(:, :, j)] = ...
                get_motion_tensor_gc(...
                f1_level(:, :, j), tmp(:, :, j), hx, hy);
        end
        
        if i == min_level
            alpha_scaling = 1;
        else
            alpha_scaling = eta.^(-0.5 * i);
        end
        
        if use_gpu
            [du, dv] = OF_solver_GPU(J11, J22, J33, J12, J13, J23, weight_level, ...
                u, v, alpha * alpha_scaling, iterations, update_lag, 0, a_data, a_smooth, hx, hy);
        else
            [du, dv] = level_solver(J11, J22, J33, J12, J13, J23, weight_level, ... 
                u, v, alpha * alpha_scaling, iterations, update_lag, 0, a_data, a_smooth, hx, hy);
        end
        
        if min(level_size > 5)
            du(2:end-1, 2:end-1) = medfilt2(du(2:end-1, 2:end-1), [5 5], 'symmetric');
            dv(2:end-1, 2:end-1) = medfilt2(dv(2:end-1, 2:end-1), [5 5], 'symmetric');
        end
        
        u = u + du;
        v = v + dv;
    end
    
    w = zeros([size(u) - 2, 2], 'double');
    w(:, :, 1) = u(2:end-1, 2:end-1);
    w(:, :, 2) = v(2:end-1, 2:end-1);
    
    if min_level > 0
        w = imresize(w, [m, n]);
    end
end

function f = add_boundary(f)
    f = padarray(f, [1 1]);
    
    f = set_boundary(f);
end

function f = set_boundary(f)
    f(:, 1) = f(:, 2);
    f(:, end) = f(:, end - 1);
    f(1, :) = f(2, :);
    f(end, :) = f(end - 1, :);
end
