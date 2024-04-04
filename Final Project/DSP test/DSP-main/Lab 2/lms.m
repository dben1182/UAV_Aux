%this function takes in as an input the signal, x, the desired signal d,
%the scaling multiple mu, and the initial h vector, h_init


%this function iterates through using the specified algorithm and
%calculates the y, output signal, the e, the error signal, and the h, the
%new filter


function [y, e, h] = lms(x, d, mu, h_init)
    
    %gets the length of the filter
    M = size(h_init,1);
    
    %gets the signal length
    signal_length = size(x,1);
   


    %creates h_temp vector to temporarily store all of the h iterations
    h_temp = h_init;
    
    %creates an output signal vector to store the y output function
    y_temp = zeros([signal_length 1]);

    %creates the temporary error vector
    e_temp = zeros([signal_length 1]);

    %this function iterates through a specified number of times of
    %propogation, starting at M, to 
    for n = (M+1):(signal_length)
        
        %sets each y_temp output
        y_temp(n) = (h_temp.')*x(n:-1:(n-M+1));
        
        if n == M+1
            a = x(n:-1:(n-M+1))
        end

        %subtracts the output from the desired output to get the error
        e_temp(n) = d(n) - y_temp(n);
        
        h_temp = h_temp + mu*x(n:-1:(n-M+1))*e_temp(n);       
    end

    %sets the output y, e, and h to the y_temp, e_temp, h_temp
    y = y_temp;
    h = h_temp;
    e = e_temp;
    
end

