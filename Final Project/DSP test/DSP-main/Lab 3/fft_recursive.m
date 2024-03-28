%this function recursively solves the fft of a 2^v length sequence
%it takes in the variable x_n, a vector, and spits out X_k, also a vector
% a column vector
function [X_k] = fft_recursive(x_n)
    %gets length of x_n
    inputLength = size(x_n,1);
    
    %creates the X_k vector, of the same length of x_n
    X_k = zeros([inputLength 1]);
    
    %first checks for the condition that the length is 2, which is the base
    %case
    if size(x_n,1) == 2
        X_k(1) = x_n(1) + x_n(2);
        X_k(2) = x_n(1) - x_n(2);       
    %otherwise, we go through the recursive algorithm, which takes the
    %input vector, breaks it into even and odd indexies, and takes the fft
    %of each of those, while also applying the twiddle factor
    else
        %sets the new length of the two subvectors, half of length
        subvectorLength = inputLength/2;

        %creates n_prime, a vector half the length of inputLength
        %that corresponds to the indecies of g_n, G_k, etc.
        n_prime = transpose(linspace(1,subvectorLength,subvectorLength));

        %fills g_n and h_n
        g_n = x_n(2.*n_prime-1);

        h_n = x_n(2.*n_prime);


        %calls the fft_recursive on g_n and h_n to get G_k and H_k
        G_k = fft_recursive(g_n);
        H_k = fft_recursive(h_n);

        %%creates the W_N twiddle factor
        W_N = exp(-1j*2*pi/inputLength);


        %iterates through and sets the first half of the bits of X_k
        for k = 1:subvectorLength
            X_k(k) = G_k(k) + (W_N^(k-1))*H_k(k);
        end

        %iterates through and sets the second half of the bits of X_k
        for k = (subvectorLength+1):inputLength
            X_k(k) = G_k(k-subvectorLength) + (W_N^(k-1))*H_k(k-subvectorLength);
        end


    end
end

