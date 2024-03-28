%this function returns an N length vector of chebychev coefficients for a
%Bandpass FIR filter

%takes in as arguments:
%N: filter length
%SamplingFrequency: continuous time sampling frequency (Radians per second)
%CenterFrequency: the center frequency about which to create the bandpass
%filter (Radians Per Second)
%FilterRadius: the distance from the center frequency to the cutoff
%frequencies (Radians Per Second


function [chebychevCoefficients] = Bandpass_Chebychev_Filter_updated(filter_length, SamplingFrequency, UpperCutoff, LowerCutoff)
    %continuous time upper and lower cutoff frequencies

    %calculates discrete time lower and upper cutoff frequencies
    DiscreteUpperCutoff = 2*pi*UpperCutoff/SamplingFrequency;
    DiscreteLowerCutoff = 2*pi*LowerCutoff/SamplingFrequency;

    %sets helper variable M, which is the length minus 1
    M = filter_length-1;

    %sets the desired H coefficients, which are the sinc function, but cut
    %off, so we do not have an ideal filter. These coefficients will be
    %multiplied later by chebychev coefficients
    H_desired = zeros(filter_length,1);
    for i = 1:filter_length
        H_desired(i,1) = (DiscreteUpperCutoff/pi)*sinc((DiscreteUpperCutoff/pi)*(i-M/2)) - (DiscreteLowerCutoff/pi)*sinc((DiscreteLowerCutoff/pi)*(i-M/2));
    end
    %gets the coefficieints for the Chebychev Window
    ChebychevWindow = chebwin(filter_length);
    %the realized coefficients
    H_realized = zeros(filter_length,1);
    for i = 1:filter_length
        H_realized(i) = H_desired(i)*ChebychevWindow(i);
    end
    chebychevCoefficients = H_realized;

end
