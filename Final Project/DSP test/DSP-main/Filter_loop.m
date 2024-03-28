function [FilteredSignal] = Filter_loop(signal, filter)
    signalLength = size(signal,1);
    filterLength = size(filter,1);


    %gets the number of iterations for the signal
    Num_iterations = signalLength + filterLength - 1;
    %creates an array to store the filtered signal
    FilteredSignal = zeros([Num_iterations 1]);

    %dummy variable to help with the sums
    
    %iterates through whole desired output
    for i = 1:Num_iterations
        summingVariable = 0;

        %iterates through each element in the filter
        for j = 1:filterLength
            %checks if we aren't accessing out of bounds
            if (i-j >= 0) && (i-j < signalLength) 
                summingVariable = summingVariable + filter(j)*signal(i-j+1);
            end    
        end
        FilteredSignal(i) = summingVariable;
    end
end

