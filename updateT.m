function [T,V1,V2,V3] = updateT(Ux,Uy,X, Y, L, F, R, B, alpha1, alpha2, eta, miu,V1, V2, V3,G,T,W,Y1,params)

    % --- 回溯线搜索参数 ---
    init_step = params.step;       % 初始步长
    shrink = params.shrink;   % 步长缩减率
    armijo_c = params.armijo_c; % Armijo条件参数
    % parameter
    epsilon = 1e-2;
    UX = alpha1 * Ux' * X + alpha2 * Uy' * Y;
    [r,n] = size(UX); % r: U的列数, n: X的列数
%     e = ones(r, 1);
%     e = e / norm(e);
    if nargin <= 11
        V = randn(r,n);
        V1 = V;
        V2 = V;
    end
    % ---- Step A: 优化第一项 ----
    gradA = -alpha1*Ux'*(X - Ux*V1)-alpha2*Uy'*(Y - Uy*V1) + epsilon * (V1-T);

    % --- 回溯循环 ---
    step = init_step;
    objective_V = @(V1) alpha1*norm(X - Ux*V1,'fro')^2 ...
                    + alpha2*norm(Y - Uy*V1,'fro')^2 ...
                   + epsilon*norm(V1 - T,'fro')^2;
    J_current = objective_V(V1);
    while objective_V(V1 - step * gradA) > J_current - armijo_c * step * norm(gradA, 'fro')^2
        step = step * shrink;
        if step < 1e-12
            break; % 防止死循环
        end
    end
    V_try = V1 - step*gradA;
%     V_try = V_try - e * (e' * V_try);
    V1 = V_try;

    % ---- Step B: 优化第二项 ----
    step = init_step;       % 初始步长

    gradB = eta * (V2*L'*F - B)*F'*L + epsilon * (V2-T);
    objective_V = @(V2) eta * norm(V2*L'*F - B,'fro')^2 ...
                   + epsilon*norm(V2 - T,'fro')^2;
    J_current = objective_V(V2);
    while objective_V(V2 - step * gradB) > J_current - armijo_c * step * norm(gradB, 'fro')^2
        step = step * shrink;
        if step < 1e-12
            break; % 防止死循环
        end
    end
    V_try = V2 - step*gradB;
    V2 = V_try;
    % ---- Step C: 优化第三项 ----
    step = init_step;       % 初始步长

    gradC = miu * W'*Y1'*(Y1*W*V3 - G * B) + epsilon * (V3-T);

    objective_V = @(V3) miu * norm(Y1*W*V3 - G * B,'fro')^2 ...
                   + epsilon*norm(V3 - T,'fro')^2;
    J_current = objective_V(V3);
    while objective_V(V3 - step * gradC) > J_current - armijo_c * step * norm(gradC, 'fro')^2
        step = step * shrink;
        if step < 1e-12
            break; % 防止死循环
        end
    end
    V_try = V3 - step*gradC;
    V3 = V_try;
    T = (V1+V2+V3)/3;
    %T = (V1+V2)/2;
end