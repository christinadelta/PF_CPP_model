function [Eh,Eq] = pf_for_cpp_est(y,lambda)

% PF_FOR_CPP_EST  SIR particle filter for joint estimation of reward rate (q)
% and hazard rate / change-point probability (h).
%
% The model uses a discrete change-point mechanism on q and a leaky Beta
% update on h. 
%
% Inputs:
%   y       - binary outcome vector (1 = reward, 0 = no reward)
%   lambda  - volatility leakage rate (0 < lambda < 1); 
%             smaller values = gentler leakage
%
% Outputs:
%   Eh      - estimated hazard rate / change-point probability per trial
%   Eq      - estimated reward rate (q) per trial
%
% @christinadelta, March 2026


%% init variables 

eta_leak    = 1 - lambda;               % forgetting factor
scale       = 10;
alpha_leak  = eta_leak * scale;         % stability/forgetting weight in the beta distribution 
beta_leak   = lambda * scale;           % volatility/leakage weight in the beta distribution 

% convert outcomes into 0 and 1 (if they aren't)
y(y == 2)   = 0;  

% initiliase particles and set up state 
N           = 1000;  % number of particles
particles_q = unifrnd(0.01, 0.99, N, 1); % initial q samples (uniform prior)
h_grid      = logspace(log10(0.001), log10(0.25), N);
particles_h = h_grid(randperm(N))'; % shuffle to avoid ordering artefacts
weights     = ones(N, 1) / N;       % uniform weights

Eq          = zeros(1, length(y));
Eh          = zeros(1, length(y));

% fit model
for t = 1:length(y)

    % compute likelihood function for each particle (update/correct)
    if y(t) == 1
        lik = particles_q;
    else
        lik = 1 - particles_q;
    end

    % bayesian update
    weights = weights .* lik;  
    weights = weights / sum(weights);  % normalise

    % compute marginal expectations (weighted means)
    Eq(t)   = sum(weights .* particles_q);
    Eh(t)   = sum(weights .* particles_h);

    % propagate transition (predict)
    if t < length(y)

        % for each particle sample next q with probability based on v
        for p = 1:N

            h                   = particles_h(p);

             % 1. decide whether a change-point occurs (our core CPP mechanism)
            if rand < (1 - h) % stability part: keep q because no change occured 
                % no change 
                % particles_q(p) = particles_q(p);

            else % volatility part: chnage has occured 
                particles_q(p) = unifrnd(0.01, 0.99);
            end

            % 2. leaky update on h (this replaces the old Gaussian nudge)
            beta_draw       = betarnd(alpha_leak, beta_leak);   % single random draw (noise on h)
            e               = beta_draw / eta_leak;             % multiplicative factor with E[e] = 1
            particles_h(p)  = h * e;                            % leaky update
            
            % 3. optional gentle bound (prevents drift to ridiculous values)
            % particles_h(p) = max(0.001, min(0.25, particles_h(p)));  

        end % end of particle sampling loop

        % resample particles based on weights (systematic resampling)
        cumw        = cumsum(weights);
        u           = (rand + (0:N - 1)') / N;
        idx         = arrayfun(@(uu) find(uu <= cumw, 1), u);
        particles_q = particles_q(idx);
        particles_h = particles_h(idx);
        weights     = ones(N, 1) / N;

    end % end of if statement for propagation
end % end of trials loop



end