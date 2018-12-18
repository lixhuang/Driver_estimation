function log_likelihood = ekf_filtering(x0, data, theta, R)
    % dx = Ax+bu
    % u = [theta1, theta2]*(x-gamma_f)
    %     + [theta3, theta4, theta5, theta6]*([x;0;0]-gamma_e)
    %     + N(theta7, theta8)
    % x_k+1 = Ad*x_k + Bdf*gammaf + Bde*gammae + C*N(theta7, theta8)
    % y = H*x + N[0,R], R\in R^2x2, H=eye(2)
    dt = 0.1;
    A = [0,1;0,0];
    B = [0;1];
    B2 = [eye(2);zeros(2)];
    K1 = theta(1:2)';
    K2 = theta(3:6)';
    Ad = eye(2)+dt*A+dt*B*K1+dt*B*K2*B2;
    Bdf = -dt*B*K1;
    Bde = -dt*B*K2;
    Cd = dt*B;
    H = eye(2);

    mean = x0;
    covm = R;
    log_likelihood = 0;
    N = size(data,2);
    for ii = 1:N-1
        gamma = data(:,ii); 
        gamma2 = data(:,ii+1); 
        xf = gamma(1:2);
        xe = gamma(3:6);
        y = gamma2(7:8);
        %% [mean, cov] = prediction_func(mean, cov, theta);
        mean = Ad*mean + Bdf*xf + Bde*xe + Cd*theta(7);
        covm = Ad*covm*Ad' + Cd*theta(8)*Cd';
        
        %%[mean, cov] = update_func(data_temp, mean, cov, R);
        delta = y - H*mean;
        S = H*covm*H' + R;
        S_inv = S^-1;
        ch = covm*H';
        mean = mean + ch*S_inv*delta;
        covm = covm - ch*S_inv*ch';
        
        %log_like_step = -1/2*delta'*S_inv*delta-log(sqrt(2*pi)*det(S));
        log_like_step = log(mvnpdf(delta,zeros([2,1]), S));
        log_likelihood = log_likelihood  + log_like_step;
        %means(:,ii+1) = mean;
        %covs(:,ii+1) = [cov(1,1);cov(2,2)]; 
    end
end