function [samples, accepted_rat, warm_param2] = generate_samples_DRAM(x0, N, data, R, log_prior, warm_flag, warm_param)
    SIGMA = 0.001;
    BURN_IN_NUM = 500;
    samples = zeros([size(x0,1),N+1]);
    accepted = zeros([N,1]);
    k0 = BURN_IN_NUM + 50;
    if(~warm_flag)
        samples(:,1) = x0;
        Sk_0 = SIGMA*eye(size(x0,1));
        Sk_0(end,end) = 0.01;
        log_post = 0;
        for ii = 1:length(data)
            % filtering => P(yn|theta)
            xs = data{ii}(7:8,1);
            log_post = log_post + ekf_filtering(xs, data{ii}, x0, R);
        end
        log_post = log_post + log_prior(x0);
    else
        Sk_0 = warm_param.S;
    end
    sd = 2.4/size(x0,1);
    csi = 1e-9;
    xk_m = x0;
    xk_m1 = x0;
    k = 0;
    for ii = 2: N+1
        if ii < k0
            S = Sk_0;
        else
            k = ii-BURN_IN_NUM;
            S = (k-2)/(k-1)*S+sd/(k-1)*(csi*eye(size(x0,1))+(k-1)*(xk_m1*xk_m1')-k*(xk_m*xk_m')+xk*xk');
        end
        [samples(:,ii), accepted(ii-1), log_post] = generate_sample_DM(samples(:,ii-1), S, data, R, log_prior, log_post);
        if ii > BURN_IN_NUM
            xk = samples(:,ii);
            xk_m1 = xk_m;
            xk_m = (xk_m*(ii-BURN_IN_NUM) + samples(:, ii))/(ii+1-BURN_IN_NUM);
        else
            xk_m = samples(:,ii);
        end
        if(mod(ii,100)==0)
            disp(ii);
        end
    end
    warm_param2.xk_m = xk_m;
    warm_param2.xk_m1 = xk_m1;
    warm_param2.S = S;
    warm_param2.k = k;
    warm_param2.log_post = log_post;
    %sum(accepted) / N %print 
    accepted_rat = sum(accepted(BURN_IN_NUM:end)) / (N-BURN_IN_NUM);
    return 
end
