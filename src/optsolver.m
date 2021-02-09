tic

X = readNPY('regions_road.npy');

N = length(X);

beta = 1;

z = zeros(N,1);
e = ones(N,1);

cvx_begin quiet
    variable L(N,N) symmetric
    W = L-diag(diag(L));
    F = beta*sum_square(vec(L));
    minimize( trace(X'*L*X) + F);
    subject to
        trace(L) == N;
        W(:) <= 0;
        L*e == z;
cvx_end
    
writeNPY(L,'Laplacian_road_beta=1.npy')

clear
toc
