% The following code allows your computer to process audio signals in
% MATLAB in real time. This was developed for use with the ECEn 487 class
% in the Winter 2013 semester.
%
% This code serves as a "wrapper" to set up the propper i/o handshaking
% between MATLAB and the PCs sound card.  You place your code inside the
% buffer loop where it says % Put your dsp code or function call here! %
% Note that real-time operation without data drop outs or queue overflow
% depends on the processor not being too busy with other tasks, so starting
% other programs, popping between windows, etc. will couse some data loss.
% 
% Brian Mazzeo
% bmazzeo@ee.byu.edu
% January 17, 20120000
%
% Modified January 18, 2013, Brian Jeffs
%
% Modified Dec. 30, 2022, Brian Jeffs
%  The dsp.AudioPlayer and dsp.AudioRecorder objects are no longer
%  supported in release R2022b (deprecated in release R2020b).
%  So, changed to use new functions: audioDeviceReader & audioDeviceWriter
%  However, for Linux, the system must have the ALSA driver, and I'm not
%  sure our Linux systems do!!!!
%
%

% These are important parameters for your sound card. They specify the
% sample rate and essentially the size of the block of data that you will
% periodically receive from the sound card.
SampleRate = 48000;
FrameSize = 4096;
NumChannels = 1;


% The real meat of the code is here. The reason that this is all preceded
% by a "try" statement is that when an error occurs, MATLAB will release
% the system resources that control the sound card. Otherwise, you often
% have to restart MATLAB.
try % VERY IMPORTANT

    % This sets up the characteristics of recording
    ar = audioDeviceReader(...
    'NumChannels', NumChannels,...
    'BitDepth', '24-bit integer', ...
    'SamplesPerFrame', FrameSize, ...
    'SampleRate', SampleRate);

    % This sets up the characteristics of playback
    ap = audioDeviceWriter(...
    'SampleRate', SampleRate, ...
    'BufferSize', FrameSize, ...
    'BitDepth', '24-bit integer');

    % This records the first set of data
    disp('Starting processing');
    input_data = step(ar); % This gets the first block of data from the sound card.
    loop_count = 0;
    while loop_count <= 20
        loop_count = loop_count + 1;
        
        %%%%%% Put your dsp code or function call here! %%%%%%%%%%%%%%%%%%%%
        y_data = input_data;
   
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        step(ap, y_data);
        input_data = step(ar);        
    end

    % You want to make sure that you release the system resources after using
    % them and they don't get tied up.
    release(ar)
    release(ap)

    % The following statements are really important so that you don't cause
    % problems and hang system resources when you actually terminate inside of
    % the loop. Otherwise, you need to restart MATLAB if you hit certain kinds
    % of errors.
catch err 
        release(ar)
        release(ap)
        rethrow(err)
end