function [B,loss_history] = train_CDESE(L1,XKTrain,YKTrain,LTrain,param,YK_c)
    % parameters
    params.step = 10;       % 初始步长 
    params.shrink = 0.5;   % 步长缩减率
    params.armijo_c = 1e-3; % Armijo条件参数
    epsilon = 1e-2;
    nbits = param.nbits;
    n = size(LTrain,1);
    c = size(LTrain,2);
    d = size(XKTrain,2);
    % initization
    B = sign(randn(nbits, n)); B(B==0) = -1;
    V = randn(nbits, n);
    %T = randn(nbits, n); 
    T1 = randn(nbits,n);T2 = randn(nbits,n);
    V1 = T1; V2 = T1; V3 = T1;
    Va = T2; Vb = T2; Vc = T2;
    
    %W = randn(c,n);  %YW-GB
    W = randn(c,nbits);
    Z = eye(c);
    G = randn(512,nbits);                                 
    F = randn(c,n);
    R = eye(nbits);
    D1 = diag(randn(d,1)); D1 = D1 / norm(D1); %
    D2 = diag(randn(d,1)); D2 = D1 / norm(D2);% 
    % Uy = randn(size(YKTrain,2),nbits); 
    % Ux = randn(size(XKTrain,2),nbits); 
    S = L1*L1';
    X = XKTrain';
    Y = YKTrain';
    L = LTrain';
    %Y1 = YK_c;
    Y1 = randn(512,c);
    L1 = L1';
    r = nbits;
    
    % ========= @@@@ add - 2025-10-30 =============
    right1 = T1' / (T1 * T1' + 1e-4 * eye(nbits)); % 
    right2 = T2' / (T2 * T2' + 1e-4 * eye(nbits));
    Ux = X * right1; % 
    Uy = Y * right2; % 

    % ========= end ===============================
    
    loss_history = zeros(param.max_iter, 1);
    
    
%%==========================================iteration start==========================================
    % 计算平衡因子
%     beta = mean(sqrt(sum(Ux.^2, 2)) .* sqrt(sum(Uy.^2, 2))) * ones(d, 1);
%     
    for j = 1:param.max_iter
        fprintf('iter:%d\n', j);
        %==========================================update U==========================================

        Ux = updateU_via_Armijo(X, T1,D1, D2 * Uy,param.alpha1,param.omega,params,r);%V
        Uy = updateU_via_Armijo(Y, T2,D2, D1 * Ux,param.alpha2,param.omega,params,r);%V
        %==========================================update T==========================================
        [T1, V1,V2,V3] = updateT(Ux,Uy,X, Y, L, F, R, B, param.alpha1, param.alpha2, param.eta,param.miu1,V1, V2, V3,G,T1,W,Y1,params); %@@@@ revised - 2025-10-30
        [T2, Va,Vb,Vc] = updateT(Ux,Uy,X, Y, L, F, R, B, param.alpha1, param.alpha2, param.eta,param.miu1,Va, Vb, Vc,G,T2,W,Y1,params);

        % ==========================================update D==========================================
        D1_old = D1; %@@@@ revised - 2025-10-30
        D2_old = D2; %@@@@ revised - 2025-10-30
        temp = sum(Ux.^2 .* (D2_old*Uy).^2, 2);
        temp(temp == 0) = eps;
        w = 1 ./ (sqrt(temp));
        D1 = w / norm(w); D1 = diag(D1); %@@@@ revised - 2025-10-30

        temp = sum((D1_old*Ux).^2 .* Uy.^2, 2);
        temp(temp == 0) = eps;
        w = 1 ./ sqrt(temp);
        D2 = w / norm(w); D2 = diag(D2); %@@@@ revised - 2025-10-30
        % ==========================================update V==========================================
Z = param.nbits * (B * S) + ...         
            param.beta * B;
        
       Temp = Z * Z' - (1/n) * (Z * ones(n,1) * (ones(1,n) * Z')) + 1e3 * eye(size(Z,1));
if any(isnan(Temp(:))) || any(isinf(Temp(:)))
    error('Temp矩阵包含非法值，请检查Z的计算过程'); 
end

[~, Lmd, QQ] = svd(Temp, 'econ');
idx = (diag(Lmd) > 1e-6);
Q = QQ(:, idx);
Q_ = orth(QQ(:, ~idx));

sqrt_Lmd = sqrt(Lmd(idx, idx));
sqrt_Lmd(sqrt_Lmd < eps) = eps;  % 防止除以零
P = (Z' - (1/n) * ones(n,1) * (ones(1,n) * Z')) * (Q / sqrt_Lmd+eps);

k_remain = param.nbits - sum(idx);
if k_remain > 0
    P_ = orth(randn(n, k_remain));
else
    P_ = zeros(n, 0);
end

% --- 更新 V ---
V = sqrt(n) * [Q, Q_] * [P, P_]';

        %==========================================update W==========================================
        temp1=param.miu1*(Y1'*Y1)+param.lambda*eye(c);
     temp2=param.miu1*(Y1'*G*B*T1'+Y1'*G*B*T2');
       W = temp1\temp2/(param.miu1*(T1*T1'+T2*T2')+param.lambda*eye(r));
       clear temp1 temp2

%==========================================update G==========================================
     G = (param.miu1*(Y1*W*T1*B'+Y1*W*T2*B'))/(param.miu1*B*B'+param.lambda*eye(r));

    %==========================================update (F)==========================================
  
    temp1 = param.eta * ((L * T1') * (T1 * L')+(L * T2') * (T2 * L')) + param.lambda * eye(c);
    temp2 = param.eta * ((L * T1')*B+(L * T2') *B);
    F = temp1 \ temp2; 
    clear temp1 temp2;

    %==========================================update B==========================================
    B_old=B;
    B = sign(nbits*(V*S)+param.beta*V+param.miu1*(G'*Y1*W*T1+G'*Y1*W*T2)+param.eta*(T1*L'*F+T2*L'*F));
if any(isnan(B(:)))
    warning(['Iteration ', num2str(j), ': B matrix contained NaN. Resetting B to stabilize.']);
    % 重置 B 为一个随机的二值矩阵
    B = B_old;
    % 确保没有 0 值
    B(B==0) = 1; 
end
    
  

    final_B = sign(B);
    final_B(final_B==0) = -1;
    B = final_B;

end