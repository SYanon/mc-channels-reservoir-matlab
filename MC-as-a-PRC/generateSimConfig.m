function generateSimConfig(N_i, input_time_points)

% t = 0:dt:t_end;  % Time vector
% 
% % Initialize the series
% x = zeros(1, length(t));
% x(1) = 1.2; % Initial condition
% 
% % Generate the series using a simple Euler method
% for i = 2:length(t)
%     if i > tau/dt
%         x(i) = x(i-1) + dt * (beta * x(i - tau/dt) / (1 + x(i - tau/dt)^n) - gamma * x(i-1));
%     else
%         x(i) = x(i-1); % No delay term for the initial points
%     end
% end
% 
% % Normalize the series to scale the release of molecules
% x = (x - min(x)) / (max(x) - min(x)); % Normalize between 0 and 1
% x = N_min + (N_max - N_min) * x;

% Define the output filename
filename = 'MCpointTXsphRX_generated.txt';

% Open the file for writing
fileID = fopen(filename, 'w');

% === Write the text line by line ===
fprintf(fileID, '# Simulate molecular communication with single phase and constant bit interval\n\n');

fprintf(fileID, 'graphics opengl               # Enable OpenGL graphics for visualization\n');
fprintf(fileID, 'graphic_iter 1                # Update graphics every iteration\n\n');

fprintf(fileID, 'dim 3                                     # Set simulation to 3 dimensions\n');
fprintf(fileID, 'boundaries x -BOUNDARYLENGTH BOUNDARYLENGTH  # Define x-axis boundaries\n');
fprintf(fileID, 'boundaries y -BOUNDARYLENGTH BOUNDARYLENGTH  # Define y-axis boundaries\n');
fprintf(fileID, 'boundaries z -BOUNDARYLENGTH BOUNDARYLENGTH  # Define z-axis boundaries\n\n');

fprintf(fileID, 'species Receptor Messenger ReceptorActive  # Define molecular species\n\n');

fprintf(fileID, 'difc Receptor DIFFRECEPTOR                 # Diffusion coefficient for Receptor\n');
fprintf(fileID, 'difc ReceptorActive DIFFRECEPTORACT        # Diffusion coefficient for ReceptorActive\n');
fprintf(fileID, 'difc Messenger DIFFMESSENGER               # Diffusion coefficient for Messenger\n\n');

fprintf(fileID, 'color Receptor(all) maroon             # Color Receptor molecules maroon\n');
fprintf(fileID, 'color Messenger(all) orange            # Color Messenger molecules orange\n');
fprintf(fileID, 'color ReceptorActive(up) fuchsia       # Color active Receptors fuchsia\n\n');

fprintf(fileID, 'display_size all(all) 3                # Set default display size for all molecules\n');
fprintf(fileID, 'display_size *Active(all) 7            # Set larger display size for active molecules\n\n');

fprintf(fileID, 'time_start START_TIME                  # Simulation start time\n');
fprintf(fileID, 'time_stop STOP_TIME                    # Simulation end time\n');
fprintf(fileID, 'time_step TIME_STEP                    # Simulation time step\n\n');

fprintf(fileID, 'frame_thickness 0                      # Remove frame thickness from graphics\n\n');

fprintf(fileID, '# Define the outer boundaries of the simulation space\n');
fprintf(fileID, 'start_surface outsides\n');
fprintf(fileID, '  unbounded_emitter front Messenger 1 TXPOSITION 0 0    # Define an emitter for Messenger molecules at the transmitter position\n');
fprintf(fileID, '  panel rect +x -BOUNDARYLENGTH -BOUNDARYLENGTH -BOUNDARYLENGTH BOUNDARYLENGTH*2 BOUNDARYLENGTH*2   # Positive x boundary\n');
fprintf(fileID, '  panel rect -x BOUNDARYLENGTH -BOUNDARYLENGTH -BOUNDARYLENGTH BOUNDARYLENGTH*2 BOUNDARYLENGTH*2    # Negative x boundary\n');
fprintf(fileID, '  panel rect +y -BOUNDARYLENGTH -BOUNDARYLENGTH -BOUNDARYLENGTH BOUNDARYLENGTH*2 BOUNDARYLENGTH*2   # Positive y boundary\n');
fprintf(fileID, '  panel rect -y -BOUNDARYLENGTH BOUNDARYLENGTH -BOUNDARYLENGTH BOUNDARYLENGTH*2 BOUNDARYLENGTH*2    # Negative y boundary\n');
fprintf(fileID, '  panel rect +z -BOUNDARYLENGTH -BOUNDARYLENGTH -BOUNDARYLENGTH BOUNDARYLENGTH*2 BOUNDARYLENGTH*2   # Positive z boundary\n');
fprintf(fileID, '  panel rect -z -BOUNDARYLENGTH -BOUNDARYLENGTH BOUNDARYLENGTH BOUNDARYLENGTH*2 BOUNDARYLENGTH*2    # Negative z boundary\n');
fprintf(fileID, '  polygon both edge                         # Render only edges of the panels\n');
fprintf(fileID, 'color both green                            # Color the surface green\n');
fprintf(fileID, 'end_surface\n\n');

fprintf(fileID, 'start_surface receiver                       # Define the receiver surface\n');
fprintf(fileID, 'action both all reflect                      # All molecules reflect off this surface\n');
fprintf(fileID, 'color both grey                              # Color the receiver surface grey\n');
fprintf(fileID, 'polygon both edge                            # Render only edges\n');
fprintf(fileID, 'panel sphere TXPOSITION+TXRXDISTANCE 0 0 RXRADIUS 30 30  # Define receiver as a sphere at specific position\n');
fprintf(fileID, 'end_surface\n\n');

fprintf(fileID, 'start_surface receiverspace                   # Define the receiver space surface (outer sphere)\n');
fprintf(fileID, 'action both all transmit                      # All molecules pass through this surface\n');
fprintf(fileID, 'polygon both edge                             # Render only edges\n');
fprintf(fileID, 'panel sphere TXPOSITION+TXRXDISTANCE 0 0 RXRADIUS+RXRECEPTIONSPACETHICKNESS 30 30  # Sphere larger than receiver, defining the reception space\n');
fprintf(fileID, 'end_surface\n\n');

fprintf(fileID, 'reaction binding Messenger(fsoln) + Receptor(up) -> ReceptorActive(up) KON  # Binding reaction with rate KON\n');
fprintf(fileID, 'reaction unbinding ReceptorActive(up) -> Messenger(fsoln) + Receptor(up)  KOFF  # Unbinding reaction with rate KOFF\n\n');

fprintf(fileID, 'start_compartment receiver                   # Define receiver compartment\n');
fprintf(fileID, 'surface receiver                             # Associated with receiver surface\n');
fprintf(fileID, 'point TXPOSITION+TXRXDISTANCE 0 0            # Position of the receiver\n');
fprintf(fileID, 'end_compartment\n\n');

fprintf(fileID, 'start_compartment outreceiver                # Define outer receiver compartment\n');
fprintf(fileID, 'surface receiverspace                        # Associated with receiverspace surface\n');
fprintf(fileID, 'point TXPOSITION+TXRXDISTANCE 0 0            # Position of the outer receiver space\n');
fprintf(fileID, 'end_compartment\n\n');

fprintf(fileID, 'start_compartment receiverspace              # Define the reception space compartment\n');
fprintf(fileID, 'compartment equal outreceiver                # Start with outreceiver compartment\n');
fprintf(fileID, 'compartment andnot receiver                  # Subtract receiver compartment to get the space in between\n');
fprintf(fileID, 'end_compartment\n\n');

fprintf(fileID, 'surface_mol numRECEPTOR Receptor(up) receiver all all    # Place Receptor molecules on receiver surface\n\n');

fprintf(fileID, 'reaction_serialnum binding r2                    # Assign serial number r2 to binding reaction\n');
fprintf(fileID, 'reaction_serialnum unbinding new + r1            # Assign serial number r1 to unbinding reaction\n');
fprintf(fileID, 'product_placement unbinding pgemmax 0.2          # Set geminate recombination max probability\n\n');

fprintf(fileID, 'output_files bitsequence_varNo_variableNo_varValue_variableValue_iter_index.txt    # Output file for transmitted bit sequence\n');
fprintf(fileID, 'output_files receivedsignal_varNo_variableNo_varValue_variableValue_iter_index.txt # Output file for received signal data\n');
fprintf(fileID, 'output_files allmolecules_varNo_variableNo_varValue_variableValue_iter_index.txt   # Output file for all molecule counts\n');
fprintf(fileID, 'cmd b overwrite allmolecules_varNo_variableNo_varValue_variableValue_iter_index.txt     # Overwrite molecule counts file at start\n');
fprintf(fileID, 'cmd b overwrite receivedsignal_varNo_variableNo_varValue_variableValue_iter_index.txt   # Overwrite received signal file at start\n');
fprintf(fileID, 'cmd b overwrite bitsequence_varNo_variableNo_varValue_variableValue_iter_index.txt      # Overwrite bit sequence file at start\n\n');

% Write commands for releasing molecules based on the time series
%fprintf(fileID, 'cmd i START_TIME STOP_TIME TIME_STEP setflag 0\n');



% for i = 1:length(t)
%     moleculeCount = round(N_i(i)); % Scale the number of molecules
%     if moleculeCount > 0
%         % Release molecules if the count is greater than zero
%         fprintf(fileID, 'cmd i %.2f %.2f %.2f set mol %d Messenger TXPOSITION 0 0\n', ...
%                 t(i), t(i) + TIME_STEP, TIME_STEP, moleculeCount);
%     end
% end


for i = 1:length(N_i)
    moleculeCount = round(N_i(i)); % Scale the number of molecules
    if moleculeCount > 0
        % Release molecules if the count is greater than zero
        fprintf(fileID, 'cmd @ %.2f set mol %d Messenger TXPOSITION 0 0\n', input_time_points(i), moleculeCount);
    end
end

%fprintf(fileID, '# cmd i START_TIME STOP_TIME BIT_INTERVAL setflag 0    # At each BIT_INTERVAL, initialize setflag to 0\n');
%fprintf(fileID, '# cmd i START_TIME STOP_TIME BIT_INTERVAL ifprob BITONE_PROB setflag 1   # With probability BITONE_PROB, setflag to 1 (transmitting bit ''1'')\n');
%fprintf(fileID, '# cmd i START_TIME STOP_TIME BIT_INTERVAL ifflag > 0.5 set mol BITONE_MOLCOUNT Messenger TXPOSITION 0 0   # If setflag is 1, release BITONE_MOLCOUNT Messenger molecules\n');
%fprintf(fileID, '# cmd i START_TIME STOP_TIME BIT_INTERVAL ifflag > 0.5 evaluate bitsequence_varNo_variableNo_varValue_variableValue_iter_index.txt 1  # Record bit ''1'' in bitsequence file\n');
%fprintf(fileID, '# cmd i START_TIME STOP_TIME BIT_INTERVAL ifflag < 0.5 set mol BITZERO_MOLCOUNT Messenger TXPOSITION 0 0  # If setflag is 0, release BITZERO_MOLCOUNT Messenger molecules\n');
%fprintf(fileID, '# cmd i START_TIME STOP_TIME BIT_INTERVAL ifflag < 0.5 evaluate bitsequence_varNo_variableNo_varValue_variableValue_iter_index.txt 0  # Record bit ''0'' in bitsequence file\n\n');

fprintf(fileID, 'cmd b molcountheader allmolecules_varNo_variableNo_varValue_variableValue_iter_index.txt   # Initialize molecule counts file with header\n');
fprintf(fileID, 'cmd N SAMPLING_PERIOD molcountincmpt2 receiverspace soln receivedsignal_varNo_variableNo_varValue_variableValue_iter_index.txt  # Record molecule counts in receiverspace and solution at intervals\n');
fprintf(fileID, 'cmd N SAMPLING_PERIOD molcount allmolecules_varNo_variableNo_varValue_variableValue_iter_index.txt    # Record total molecule counts at intervals\n\n');

fprintf(fileID, 'text_display time Receptor(up) Messenger ReceptorActive(up)    # Display time and molecule counts during simulation\n\n');

fprintf(fileID, 'end_file    # End of Smoldyn script\n');

% Close the file
fclose(fileID);

% Inform the user that the exact ending part has been written
fprintf('The exact ending part of the Smoldyn configuration file has been successfully written.\n');
% Inform the user
fclose("all");
fprintf('Smoldyn configuration file "%s" generated successfully.\n', filename);
end