function [ipcByOrder, totalIPC, detailTable] = computeIPC_legendreNoZeroDup(X, u, dsmaxPairs)
% COMPUTEIPC_LEGENDRENOZERODUP
%   Enumerate "factor sets" { (n1, s1), (n2, s2), ... } with:
%       1) 1 <= n_i,
%       2) 1 <= s_i <= s_max,
%       3) sum(n_i) = d,
%       4) no duplicates (sorted in nondecreasing s; if s are equal, n is nondecreasing).
%   Then we train a linear readout from X(k,:) -> product of Legendre polynomials, 
%   measure capacity = squared correlation, and sum it up.
%
%   USAGE:
%     [ipcByOrder, totalIPC, detailTable] = ...
%         computeIPC_legendreNoZeroDup(X, u, dsmaxPairs)
%
%   Each row of dsmaxPairs is [d, s_max].  We print out each polynomial's
%   factor set, e.g. { (1,1), (1,2) }, and capacity, plus a sum for that d.
%
%   The final totalIPC is the sum across all enumerated polynomials
%   from all rows in dsmaxPairs.
% ------------------------------------------------------------------------

    u = u(:);
    [TT, Nres2] = size(X); %#ok<ASGLU>

    allDetails = {};  
    ipcByOrder = [];
    totalIPC   = 0;

    % Sort by degree if you want first-degree, second-degree, etc.
    [~, sortIdx] = sort(dsmaxPairs(:,1));
    dsmaxPairs   = dsmaxPairs(sortIdx,:);

    for row = 1:size(dsmaxPairs,1)
        d     = dsmaxPairs(row,1);
        s_max = dsmaxPairs(row,2);

        % Enumerate sets of factors summing to degree d
        factorSets = enumerateFactorSetsNoZeros(d, s_max);

        sumC = 0;  % sum of capacities for polynomials of this (d,s_max)

        fprintf('\n===== Degree d = %d,   s_max = %d =====\n', d, s_max);

        for fsIdx = 1:numel(factorSets)
            fset = factorSets{fsIdx};

            % Time alignment: largest delay
            sUsed = cellfun(@(x) x(2), fset);
            iMax  = max(sUsed);

            validK = (iMax+1):TT;
            if isempty(validK)
                continue; 
            end

            % Build target polynomial y(k)
            y = buildPolynomialProduct(u, fset, validK);

            % Linear readout
            Xd = X(validK,:);
            w  = pinv(Xd)*y;
            z  = Xd*w;
            R  = corrcoef(y,z);
            if size(R,1)>1
                C = R(1,2)^2;
            else
                C = 0;
            end

            sumC = sumC + C;

            % Build string
            polyStr = buildLegendreProductString(fset);
            tupleStr = factorSetToString(fset);
            fprintf('  %2d) %s   =>   %s,   capacity = %.4f\n', ...
                fsIdx, tupleStr, polyStr, C);

            % Save details
            allDetails(end+1,:) = { d, s_max, fset, C }; %#ok<AGROW>
        end

        fprintf('  --> Sum of capacities for degree %d, s_max=%d: %.4f\n', ...
            d, s_max, sumC);

        sEntry.degree      = d;
        sEntry.smax        = s_max;
        sEntry.sumCapacity = sumC;
        ipcByOrder = [ipcByOrder; sEntry];
        totalIPC = totalIPC + sumC;
    end

    detailTable = cell2table(allDetails,...
        'VariableNames',{'Degree','Smax','FactorSet','Capacity'});

    fprintf('\n=== TOTAL IPC across all enumerated polynomials: %.4f ===\n', totalIPC);
end

%% =======================================================================
function factorSets = enumerateFactorSetsNoZeros(d, s_max)
    factorSets = {};
    partialSeq = {};
    backtrack(1, 0, partialSeq);

    function backtrack(currS, sumDegreesSoFar, seq)
        if sumDegreesSoFar == d
            factorSets{end+1} = seq; %#ok<AGROW>
            return;
        end
        if sumDegreesSoFar > d
            return;
        end

        for s = currS : s_max
            nMin = 1;
            if ~isempty(seq)
                lastPair = seq{end};
                lastN    = lastPair(1);
                lastS    = lastPair(2);
                if s == lastS
                    nMin = lastN;
                elseif s < lastS
                    continue;
                end
            end

            maxN = d - sumDegreesSoFar;
            for n = nMin : maxN
                newSeq = [seq; {[n, s]}]; %#ok<AGROW>
                backtrack(s, sumDegreesSoFar + n, newSeq);
            end
        end
    end
end

%% =======================================================================
function y = buildPolynomialProduct(u, fset, validK)
    y = ones(length(validK),1);
    for j = 1:numel(fset)
        n = fset{j}(1);
        s = fset{j}(2);
        polyVals = legendrePoly(n, u(validK - s));
        y = y .* polyVals;
    end
end

%% =======================================================================
function strOut = buildLegendreProductString(fset)
    parts = cell(1, numel(fset));
    for i = 1:numel(fset)
        n = fset{i}(1);
        s = fset{i}(2);
        if n==1
            parts{i} = sprintf('u(k-%d)', s);
        else
            parts{i} = sprintf('P%d[u(k-%d)]', n, s);
        end
    end
    strOut = strjoin(parts, ' * ');
end

%% =======================================================================
function strOut = factorSetToString(fset)
    cellStrs = cell(1, numel(fset));
    for i = 1:numel(fset)
        cellStrs{i} = sprintf('(%d,%d)', fset{i}(1), fset{i}(2));
    end
    strOut = sprintf('{ %s }', strjoin(cellStrs, ', '));
end

%% =======================================================================
function P = legendrePoly(d, x)
    x = x(:);
    switch d
        case 1
            P = x;
        otherwise
            if d==0
                P = ones(size(x));
                return;
            end
            P0 = ones(size(x));
            P1 = x;
            for n = 1:(d-1)
                P2 = ((2*n+1)*x.*P1 - n*P0)/(n+1);
                P0 = P1;
                P1 = P2;
            end
            P = P1;
    end
    if isrow(x)
        P = P.';
    end
end
