function [U] = updateU_via_Armijo(X, V,Z, D,alpha, omega, params,r)
    % --- 回溯线搜索参数 ---
    step = params.step;       % 初始步长
    shrink = params.shrink;   % 步长缩减率
    armijo_c = params.armijo_c; % Armijo条件参数

    [m, ~] = size(X);
%    rank = size(V,1);
    U = randn(m, r);

    % 权重归一化
%     sumOfWeight = alpha + omega + mu;
%     alpha = alpha / sumOfWeight;
%     omega = omega / sumOfWeight;
%     mu = mu / sumOfWeight;

    % 定义目标函数
    objective_U = @(U) alpha * norm(X - U * V, 'fro')^2 + ...
                       omega * norm((Z * U) .* D, 'fro')^2;

    % 当前损失
    J_current = objective_U(U);

    % 梯度计算
    gradU = -2 * (alpha * X * V' - alpha * U * (V * V')) ...
            + 2 * omega * Z' * ((Z * U) .* (D.^2));

    
    % --- 回溯循环 ---
    while objective_U(U - step * gradU) > J_current - armijo_c * step * norm(gradU, 'fro')^2
        step = step * shrink;
        if step < 1e-12
            break; % 防止死循环
        end
    end

    % 更新 U
    U = U - step * gradU;


end