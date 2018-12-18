function [candidate, ac_flag, log_post_current] = generate_sample_DM(current_sample, S, data, R, log_prior, log_post_current)
    Sk1 = S;
    %candidate1 = proposal_sampler(current_sample, Sk1);
    candidate1 = mvnrnd(current_sample, Sk1)';
    
    log_likelihood = 0;
    for ii = 1:length(data)
        % filtering => P(yn|theta)
        x0 = data{ii}(7:8,1);
        log_likelihood = log_likelihood + ekf_filtering(x0, data{ii}, candidate1, R);
    end
    log_posterior = log_likelihood + log_prior(candidate1);
    
    numerator = log_posterior + proposal_logpdf(current_sample, candidate1, Sk1);
    denominator = log_post_current + proposal_logpdf(candidate1, current_sample, Sk1);
    value = numerator - denominator;

    if value >= 0
        candidate = candidate1;
        ac_flag = 1;
        log_post_current = log_posterior;
        return
    else
        accept_rand = rand;
        if accept_rand < value
            candidate = candidate1;
            ac_flag = 1;
            log_post_current = log_posterior;
            return
        else
%             candidate = current_sample;
%             ac_flag = 0;
%             return
            Sk2 = S*0.5;
            candidate2 = mvnrnd(current_sample, Sk2)';
            
            log_likelihood = 0;
            for ii = 1:length(data)
                % filtering => P(yn|theta)
                x0 = data{ii}(7:8,1);
                log_likelihood = log_likelihood + ekf_filtering(x0, data{ii}, candidate2, R);
            end
            log_posterior2 = log_likelihood + log_prior(candidate2);
            
            numerator2 = log_posterior2 + proposal_logpdf(candidate1, candidate2, Sk1) + proposal_logpdf(current_sample, candidate2, Sk2);
            denominator2 = log_post_current + proposal_logpdf(candidate1, current_sample, Sk1) + proposal_logpdf(candidate2, current_sample, Sk2);
            loga1 = log_posterior + proposal_logpdf(candidate2, candidate1, Sk1) - log_posterior2 - proposal_logpdf(candidate1, candidate2, Sk1);
            if loga1 >= 0
                candidate = current_sample;
                ac_flag = 0;
                return
            end
            numerator2 = numerator2 + log(1-exp(loga1));
            denominator2 = denominator2 + log(1-exp(value));
            value2 = numerator2 - denominator2;
            
            if value2 >= 0
                candidate = candidate2;
                ac_flag = 1;
                log_post_current = log_posterior2;
                return
            else
                accept_rand = rand;
                if accept_rand < value2
                    candidate = candidate2;
                    ac_flag = 1;
                    log_post_current = log_posterior2;
                    return
                else
                    candidate = current_sample;
                    ac_flag = 0;
                    return
                end
            end
        end
    end
end

