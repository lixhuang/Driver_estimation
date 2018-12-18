function p = proposal_logpdf(mean, x, S)
    delta = mean - x;
    p = log(mvnpdf(delta, zeros([8,1]), S));
    %p = -1/2*delta'*S^-1*delta-log(sqrt(2*pi)*det(S));
end

