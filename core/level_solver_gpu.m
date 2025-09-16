function [du, dv] = level_solver_gpu(J11, J22, J33, J12, J13, J23, weight_level, ...
            u, v, alpha, iterations, update_lag, ~, a_data, a_smooth, hx, hy)

    u_dev = prepare_input(u);
    v_dev = prepare_input(v);

    j11 = prepare_input(J11);
    j22 = prepare_input(J22);
    j33 = prepare_input(J33);
    j12 = prepare_input(J12);
    j13 = prepare_input(J13);
    j23 = prepare_input(J23);

    localW = (ndims(weight_level) == ndims(J11)) && all(size(weight_level) == size(J11));
    if localW
        W = prepare_input(weight_level);
    else
        W = to_like(reshape(weight_level, 1, 1, []), j11);
    end

    du = zeros(size(u_dev), 'like', u_dev);
    dv = zeros(size(u_dev), 'like', u_dev);

    [m,n] = size(du);
    c1 = 2:m - 1;
    c2 = 2:n - 1;
    l = 1:n - 2;
    r = 3:n;
    t = 1:m - 2;
    b = 3:m;

    [ii,jj] = ndgrid(to_like(c1, u_dev), to_like(c2, u_dev));
    R = bool_like(m, n, u_dev);
    B = R;
    R(c1,c2) = mod(ii + jj, 2) == 0;
    B(c1,c2) = ~R(c1, c2);

    ax = cast(alpha(1)/(hx.^2), 'like', u_dev);
    ay = cast(alpha(2)/(hy.^2), 'like', u_dev);
    epsv = cast(1e-6, 'like', u_dev);
    omega = cast(1.95, 'like', u_dev);

    K = size(j11, 3);
    if isscalar(a_data)
        A = to_like(a_data, j11);
        A_vec = repmat(A,[1 1 K]);
    else
        A_vec = to_like(reshape(a_data, 1, 1, []), j11);
    end

    psi_data   = ones(size(j11), 'like', j11);
    psi_smooth = ones(size(u_dev), 'like', u_dev);

    denom_u_data = zeros(size(u_dev), 'like', u_dev);
    denom_v_data = zeros(size(u_dev), 'like', u_dev);

    for it = 1:iterations
        if mod(it-1, update_lag)==0
            du3 = reshape(du, size(du, 1), size(du, 2), 1);
            dv3 = reshape(dv, size(dv, 1), size(dv, 2), 1);

            E = j11.*(du3.^2) + j22.*(dv3.^2) + 2 * j12.*(du3.*dv3) + ...
                2 * j13.*du3 + 2 * j23.*dv3 + j33;
            E = max(E, 0);
            psi_data = A_vec .* (E + epsv).^(A_vec - 1);

            if a_smooth ~= 1
                uc = u_dev + du;
                vc = v_dev + dv;
                ux = (uc(c1, r) - uc(c1, l)) / (2 * hx);
                uy = (uc(b, c2) - uc(t, c2)) / (2 * hy);
                vx = (vc(c1, r) - vc(c1, l)) / (2 * hx);
                vy = (vc(b, c2) - vc(t, c2)) / (2 * hy);
                mag = zeros(m, n, 'like', du);
                mag(c1,c2) = ux.^2 + uy.^2 + vx.^2 + vy.^2;
                psi_smooth = zeros(m, n, 'like', du);
                psi_smooth(c1,c2) = a_smooth * (mag(c1, c2) + epsv).^(a_smooth - 1);
                psi_smooth = set_boundary2D(psi_smooth);
            else
                psi_smooth = ones(size(u_dev), 'like', u_dev);
            end

            if localW
                denom_u_data = sum(W.*psi_data.*j11,3);
                denom_v_data = sum(W.*psi_data.*j22,3);
            else
                denom_u_data = sum(bsxfun(@times, W, psi_data).*j11, 3);
                denom_v_data = sum(bsxfun(@times, W, psi_data).*j22, 3);
            end
            denom_u_data = max(denom_u_data, epsv);
            denom_v_data = max(denom_v_data, epsv);
        end

        du = set_boundary2D(du);
        dv = set_boundary2D(dv);

        psiC = psi_smooth(c1, c2);
        if a_smooth ~= 1
            wL_u = 0.5*(psiC + psi_smooth(c1, l)) * ax;
            wR_u = 0.5*(psiC + psi_smooth(c1, r)) * ax;
            wT_u = 0.5*(psiC + psi_smooth(t, c2)) * ay;
            wB_u = 0.5*(psiC + psi_smooth(b, c2)) * ay;
            wL_v = wL_u;
            wR_v = wR_u;
            wT_v = wT_u;
            wB_v = wB_u;
        else
            wL_u = ax;
            wR_u = ax;
            wT_u = ay;
            wB_u = ay;
            wL_v = ax;
            wR_v = ax;
            wT_v = ay;
            wB_v = ay;
        end

        if localW
            num_u_data = -sum(W.*psi_data.*(j13 + j12.*reshape( ...
                dv, size(dv, 1), size(dv, 2), 1)), 3);
            num_v_data = -sum(W.*psi_data.*(j23 + j12.*reshape( ...
                du, size(du, 1), size(du, 2), 1)), 3);
        else
            num_u_data = -sum(bsxfun(@times, W, psi_data).*( ...
                j13 + j12.*reshape(dv, size(dv, 1), size(dv, 2), 1)), 3);
            num_v_data = -sum(bsxfun(@times, W, psi_data).*( ...
                j23 + j12.*reshape(du, size(du, 1), size(du, 2), 1)), 3);
        end

        num_u = zeros(m, n, 'like', du);
        num_v = zeros(m, n, 'like', dv);
        den_u = zeros(m, n, 'like', du);
        den_v = zeros(m, n, 'like', dv);

        num_u(c1,c2) = num_u_data(c1, c2) ...
            + wL_u.*(u_dev(c1, l) + du(c1, l) - u_dev(c1, c2)) ...
            + wR_u.*(u_dev(c1, r) + du(c1 ,r) - u_dev(c1, c2)) ...
            + wT_u.*(u_dev(t, c2) + du(t, c2) - u_dev(c1, c2)) ...
            + wB_u.*(u_dev(b, c2) + du(b, c2) - u_dev(c1, c2));
        den_u(c1,c2) = denom_u_data(c1, c2) + (wL_u + wR_u + wT_u + wB_u);

        num_v(c1,c2) = num_v_data(c1,c2) ...
            + wL_v.*(v_dev(c1, l) + dv(c1, l) - v_dev(c1, c2)) ...
            + wR_v.*(v_dev(c1, r) + dv(c1, r) - v_dev(c1, c2)) ...
            + wT_v.*(v_dev(t, c2) + dv(t, c2) - v_dev(c1, c2)) ...
            + wB_v.*(v_dev(b, c2) + dv(b, c2) - v_dev(c1, c2));
        den_v(c1, c2) = denom_v_data(c1, c2) + (wL_v + wR_v + wT_v + wB_v);

        upd_u = du;
        upd_v = dv;
        upd_u(c1, c2) = (1 - omega).*du(c1, c2) + omega.*( ...
            num_u(c1, c2)./max(den_u(c1, c2), epsv));
        upd_v(c1, c2) = (1 - omega).*dv(c1, c2) + omega.*( ...
            num_v(c1, c2)./max(den_v(c1, c2),epsv));

        du(R) = upd_u(R);
        dv(R) = upd_v(R);
        du = set_boundary2D(du);
        dv = set_boundary2D(dv);

        if localW
            num_u_data = -sum(W.*psi_data.*( ...
                j13 + j12.*reshape(dv, size(dv,1), size(dv,2), 1)),3);
            num_v_data = -sum(W.*psi_data.*( ...
                j23 + j12.*reshape(du, size(du,1), size(du,2), 1)),3);
        else
            num_u_data = -sum(bsxfun(@times, W, psi_data).*( ...
                j13 + j12.*reshape(dv, size(dv, 1), size(dv, 2), 1)), 3);
            num_v_data = -sum(bsxfun(@times,W,psi_data).*( ...
                j23 + j12.*reshape(du, size(du, 1), size(du, 2), 1)), 3);
        end

        num_u(c1,c2) = num_u_data(c1, c2) ...
            + wL_u.*(u_dev(c1, l)+du(c1, l) - u_dev(c1, c2)) ...
            + wR_u.*(u_dev(c1, r)+du(c1,r) - u_dev(c1, c2)) ...
            + wT_u.*(u_dev(t, c2)+du(t, c2) - u_dev(c1, c2)) ...
            + wB_u.*(u_dev(b, c2)+du(b, c2) - u_dev(c1, c2));
        den_u(c1,c2) = denom_u_data(c1, c2) + (wL_u + wR_u + wT_u + wB_u);

        num_v(c1, c2) = num_v_data(c1, c2) ...
            + wL_v.*(v_dev(c1, l) + dv(c1, l) - v_dev(c1, c2)) ...
            + wR_v.*(v_dev(c1, r) + dv(c1, r) - v_dev(c1, c2)) ...
            + wT_v.*(v_dev(t, c2) + dv(t, c2) - v_dev(c1, c2)) ...
            + wB_v.*(v_dev(b, c2) + dv(b, c2) - v_dev(c1, c2));
        den_v(c1, c2) = denom_v_data(c1, c2) + (wL_v + wR_v + wT_v + wB_v);

        upd_u(c1, c2) = (1 - omega).*du(c1, c2) + ...
            omega.*(num_u(c1, c2)./max(den_u(c1, c2), epsv));
        upd_v(c1, c2) = (1 - omega).*dv(c1, c2) + ...
            omega.*(num_v(c1, c2)./max(den_v(c1, c2), epsv));

        du(B) = upd_u(B); dv(B) = upd_v(B);
        du = set_boundary2D(du);
        dv = set_boundary2D(dv);
    end

    if isa(du,'gpuArray')
        du = gather(du);
    end
    if isa(dv,'gpuArray')
        dv = gather(dv);
    end
end

function y = prepare_input(x)
    persistent warned
    if gpu_available()
        y = gpuArray(x);
    else
        y = x;
        if isempty(warned) || ~warned
            warning('GPU not available. Running vectorized RBGS on CPU - set use_gpu to false to increase speed.');
            warned = true;
        end
    end
end

function tf = gpu_available()
    try
        tf = (gpuDeviceCount > 0);
    catch
        tf = false;
    end
end

function y = to_like(x, likeVar)
    if isa(likeVar, 'gpuArray')
        y = gpuArray(x);
    else
        y = x;
    end
end

function y = bool_like(m, n, likeVar)
    if isa(likeVar, 'gpuArray')
        y = gpuArray(false(m, n));
    else
        y = false(m, n);
    end
end

function A = set_boundary2D(A)
    A(:,1)   = A(:,2);
    A(:,end) = A(:,end-1);
    A(1,:)   = A(2,:);
    A(end,:) = A(end-1,:);
end
