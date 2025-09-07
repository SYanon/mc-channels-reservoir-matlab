function generateLorenzSimConfig(N_i, input_time_points)
% generateLorenzSimConfig  Writes a Smoldyn configuration file for the Lorenz task.
%
%   This function takes the serialized molecule counts (N_i) and their
%   corresponding release times and writes the full Smoldyn input file.

% Define the output filename
filename = 'MC_lorenz_smoldyn_config.txt';

% Open the file for writing
fileID = fopen(filename, 'w');
if fileID == -1
    error('Cannot open file for writing: %s', filename);
end

% --- Write the static part of the Smoldyn configuration ---
fprintf(fileID, '# Smoldyn configuration for Lorenz ''63 Task\n\n');
fprintf(fileID, 'dim 3\n');
fprintf(fileID, 'boundaries x -25 25\n');
fprintf(fileID, 'boundaries y -25 25\n');
fprintf(fileID, 'boundaries z -25 25\n\n');

fprintf(fileID, 'species Receptor Messenger ReceptorActive\n\n');

fprintf(fileID, 'difc Receptor 0\n');
fprintf(fileID, 'difc ReceptorActive 0\n');
fprintf(fileID, 'difc Messenger DIFFMESSENGER\n\n');

fprintf(fileID, 'color Receptor(all) maroon\n');
fprintf(fileID, 'color Messenger(all) orange\n');
fprintf(fileID, 'color ReceptorActive(up) fuchsia\n\n');

fprintf(fileID, 'display_size all(all) 3\n\n');

fprintf(fileID, 'time_start START_TIME\n');
fprintf(fileID, 'time_stop STOP_TIME\n');
fprintf(fileID, 'time_step TIME_STEP\n\n');

fprintf(fileID, 'start_surface receiver\n');
fprintf(fileID, '  action both all reflect\n');
fprintf(fileID, '  panel sphere TXRXDISTANCE 0 0 3 30 30\n'); % Receiver at (TXRXDISTANCE, 0, 0) with radius 3
fprintf(fileID, 'end_surface\n\n');

fprintf(fileID, 'reaction binding Messenger(fsoln) + Receptor(up) -> ReceptorActive(up) KON\n');
fprintf(fileID, 'reaction unbinding ReceptorActive(up) -> Messenger(fsoln) + Receptor(up) KOFF\n\n');

fprintf(fileID, 'surface_mol 500 Receptor(up) receiver all all\n\n');

% --- Define output files and commands ---
fprintf(fileID, 'output_files allmolecules.txt\n');
fprintf(fileID, 'cmd b overwrite allmolecules.txt\n');
fprintf(fileID, 'cmd b molcountheader allmolecules.txt\n');
fprintf(fileID, 'cmd i START_TIME STOP_TIME SAMPLING_PERIOD molcount allmolecules.txt\n\n');

% --- Write dynamic molecule release commands ---
fprintf(fileID, '# Molecule release commands based on serialized Lorenz series\n');
for i = 1:length(N_i)
    moleculeCount = round(N_i(i));
    if moleculeCount > 0
        % Smoldyn's 'cmd @' command releases molecules at a specific time point
        fprintf(fileID, 'cmd @ %.6f set mol %d Messenger 0 0 0\n', input_time_points(i), moleculeCount);
    end
end

fprintf(fileID, '\nend_file\n');

fclose(fileID);
end