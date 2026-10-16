%% Experiment 12 - Monte Carlo BER over AWGN
% BPSK, QPSK, BFSK (coherent), 8-PSK, 16-QAM, 64-QAM
% Works on MATLAB R2015a: no toolbox, no functions, no implicit expansion.
%
% Conventions (explicit):
%   Es   = 1                 (every constellation normalized to unit average symbol energy)
%   k    = bits per symbol   (BPSK 1, QPSK 2, BFSK 1, 8-PSK 3, 16-QAM 4, 64-QAM 6)
%   Eb   = Es/k
%   Eb/N0 (linear) = 10^(EbN0_dB/10)
%   N0   = Eb/(Eb/N0)
%   Complex noise n = sqrt(N0/2)*(randn + 1j*randn) -> E|n|^2 = N0,
%                     variance per real dimension = N0/2
%   BFSK = coherent orthogonal signalling: points {1, 1j}.
%   Gray mapping for QPSK, 8-PSK, 16-QAM, 64-QAM. Detector = minimum distance.

clear; clc; close all;
rng(12,'twister');

%% Parameters
EbN0_dB   = 0:1:20;          % Eb/N0 points (dB)
maxErrors = 200;             % stop a point after 200 bit errors ...
maxBits   = 1e6;             % ... or after 10^6 transmitted bits
blockBits = 1e5;             % bits generated per block
names     = {'BPSK','QPSK','BFSK','8-PSK','16-QAM','64-QAM'};
kk        = [1 2 1 3 4 6]; % bits per symbol
nMod      = numel(names);
nPts      = numel(EbN0_dB);
Es        = 1;

%% Constellations (column vectors indexed by bit label + 1)
C = cell(1,nMod);
C{1} = [1; -1];                          % BPSK: bit 0 -> +1, bit 1 -> -1
C{3} = [1; 1j];                          % BFSK: coherent orthogonal signals
for m = [2 5 6]                          % QPSK, 16-QAM, 64-QAM (Gray)
    M   = 2^kk(m);
    L1  = sqrt(M);
    lev = zeros(L1,1);
    for p = 0:L1-1
        lev(bitxor(p,bitshift(p,-1))+1) = 2*p-(L1-1);
    end
    Cm = zeros(M,1);
    for L = 0:M-1
        Cm(L+1) = lev(floor(L/L1)+1) + 1j*lev(mod(L,L1)+1);
    end
    C{m} = Cm;
end
Cm = zeros(8,1);                         % 8-PSK (Gray)
for q = 0:7
    Cm(bitxor(q,bitshift(q,-1))+1) = exp(1j*2*pi*q/8);
end
C{4} = Cm;
for m = 1:nMod
    C{m} = C{m}/sqrt(mean(abs(C{m}).^2));    % Es = 1
end

%% Mandatory validation
% (a) noiseless input and very high SNR must be reproduced exactly
% (b) very low SNR must approach the random-decision limit BER = 0.5
fprintf('Validation\n');
for m = 1:nMod
    k  = kk(m);  Cm = C{m};  Eb = Es/k;
    N0list = [0, Eb/10^(100/10), Eb/10^(-60/10)];
    Blist  = [1e5, 1e5, 1e6];
    nerr   = zeros(1,3);  nbit = zeros(1,3);
    for t = 1:3
        done = 0;  ne = 0;
        while done < Blist(t)
            nSym = min(floor(1e5/k), floor((Blist(t)-done)/k));
            if nSym < 1, break; end
            bits    = randi([0 1],nSym,k);
            lab     = bits*(2.^(k-1:-1:0)).';
            r       = Cm(lab+1) + sqrt(N0list(t)/2)*(randn(nSym,1)+1j*randn(nSym,1));
            [~,idx] = min(abs(bsxfun(@minus,r,Cm.')),[],2);
            bitsHat = mod(floor(bsxfun(@rdivide,idx-1,2.^(k-1:-1:0))),2);
            ne      = ne + nnz(bitsHat ~= bits);
            done    = done + nSym*k;
        end
        nerr(t) = ne;  nbit(t) = done;
    end
    berLo = nerr(3)/nbit(3);
    pass  = (nerr(1)==0) && (nerr(2)==0) && (abs(berLo-0.5) < 0.01);
    if pass, msg = 'PASS'; else msg = 'FAIL'; end
    fprintf('%-8s noiseless errors=%d | 100 dB errors=%d | -60 dB BER=%.4f | %s\n', ...
            names{m}, nerr(1), nerr(2), berLo, msg);
    if ~pass
        error('Validation failed for %s.', names{m});
    end
end
fprintf('\n');

%% Adaptive Monte Carlo driver
ber    = nan(nMod,nPts);     % simulated BER (NaN where zero errors)
ub     = nan(nMod,nPts);     % 95% upper bound 3/N where zero errors
nErrM  = zeros(nMod,nPts);   % final error count of each point
nBitsM = zeros(nMod,nPts);   % bits transmitted when each point stopped

for m = 1:nMod
    k  = kk(m);  Cm = C{m};  Eb = Es/k;
    for p = 1:nPts
        g  = 10^(EbN0_dB(p)/10);     % Eb/N0 linear
        N0 = Eb/g;                   % noise spectral density
        nErr = 0;  nBits = 0;
        while nErr < maxErrors
            nSym = min(floor(blockBits/k), floor((maxBits-nBits)/k));
            if nSym < 1, break; end                       % 10^6 bits reached
            bits    = randi([0 1],nSym,k);
            lab     = bits*(2.^(k-1:-1:0)).';
            r       = Cm(lab+1) + sqrt(N0/2)*(randn(nSym,1)+1j*randn(nSym,1));
            [~,idx] = min(abs(bsxfun(@minus,r,Cm.')),[],2);
            bitsHat = mod(floor(bsxfun(@rdivide,idx-1,2.^(k-1:-1:0))),2);
            eMat    = (bitsHat ~= bits).';                % k x nSym
            e       = eMat(:);                            % bit-ordered error flags
            c       = cumsum(e);
            hit     = find(nErr + c >= maxErrors, 1);
            if isempty(hit)
                nErr  = nErr  + c(end);
                nBits = nBits + numel(e);
            else                                          % stop exactly at the 200th error
                nErr  = maxErrors;
                nBits = nBits + hit;
            end
        end
        nErrM(m,p)  = nErr;
        nBitsM(m,p) = nBits;
        if nErr > 0
            ber(m,p) = nErr/nBits;
        else
            ub(m,p)  = 3/nBits;      % zero errors: upper bound (95%), not BER = 0
        end
    end
end

%% Figure 1: semilog BER curves (theory lines + simulated markers)
Qf     = @(x) 0.5*erfc(x/sqrt(2));
gd     = linspace(EbN0_dB(1),EbN0_dB(end),401);
gdLin  = 10.^(gd/10);
colors = lines(nMod);
marks  = {'o','s','d','p','h','<'};

figure(1); clf; hold on; box on; grid on;
set(gcf,'Color','w','Position',[100 100 820 620]);
set(gca,'YScale','log','FontSize',12,'LineWidth',1);
hSim = gobjects(nMod,1);
for m = 1:nMod
    k = kk(m);
    switch names{m}
        case {'BPSK','QPSK'}                 % exact
            pb = Qf(sqrt(2*gdLin));                          valid = true(size(gdLin));
        case 'BFSK'                          % exact, coherent
            pb = Qf(sqrt(gdLin));                            valid = true(size(gdLin));
        case '8-PSK'                         % Gray, nearest-neighbour approximation
            pb = (2/k)*Qf(sqrt(2*k*gdLin)*sin(pi/8));        valid = pb <= 1e-2;
        case {'16-QAM','64-QAM'}             % Gray, nearest-neighbour approximation
            M  = 2^k;
            pb = (4/k)*(1-1/sqrt(M))*Qf(sqrt(3*k*gdLin/(M-1)));  valid = pb <= 1e-2;
    end
    plot(gd(valid),pb(valid),'-','Color',colors(m,:),'LineWidth',1.6,'HandleVisibility','off');
    ok = nErrM(m,:) > 0;
    hSim(m) = plot(EbN0_dB(ok),ber(m,ok),marks{m},'Color',colors(m,:), ...
                   'MarkerFaceColor',colors(m,:),'MarkerSize',7,'LineStyle','none', ...
                   'DisplayName',names{m});
    plot(EbN0_dB(~ok),ub(m,~ok),'v','Color',colors(m,:),'MarkerSize',8, ...
         'LineStyle','none','HandleVisibility','off');
end
hTh = plot(nan,nan,'k-','LineWidth',1.6,'DisplayName','Theory');
hUb = plot(nan,nan,'kv','MarkerSize',8,'LineStyle','none','DisplayName','Zero errors (upper bound 3/N)');
xlabel('E_b/N_0 (dB)','FontSize',13);
ylabel('Bit Error Rate (BER)','FontSize',13);
title('Monte Carlo BER over AWGN: simulation (markers) vs theory (lines)','FontSize',13);
legend([hSim; hTh; hUb],'Location','southwest');
xlim([EbN0_dB(1) EbN0_dB(end)]);
ylim([1e-7 1]);

%% Figure 2: error count versus transmitted bits
figure(2); clf; hold on; box on; grid on;
set(gcf,'Color','w','Position',[150 80 820 620]);
set(gca,'XScale','log','FontSize',12,'LineWidth',1);
hE = gobjects(nMod,1);
for m = 1:nMod
    hE(m) = plot(nBitsM(m,:),nErrM(m,:),marks{m},'Color',colors(m,:), ...
                 'MarkerFaceColor',colors(m,:),'MarkerSize',6,'LineStyle','none', ...
                 'DisplayName',names{m});
end
hL1 = plot([1e2 3e6],[maxErrors maxErrors],'k--','LineWidth',1.2,'DisplayName','200-error stop');
hL2 = plot([maxBits maxBits],[-5 215],'k:','LineWidth',1.5,'DisplayName','10^6-bit stop');
xlabel('Transmitted bits','FontSize',13);
ylabel('Error count','FontSize',13);
title('Error count versus transmitted bits (one point per E_b/N_0)','FontSize',13);
legend([hE; hL1; hL2],'Location','southwest');
xlim([1e2 3e6]);
ylim([-5 215]);