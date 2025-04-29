inputFile = 'SagamoreLog_03_29_2025_1702EST'; % Name of the input text file
outputFile = 'sensor_data.csv'; % Name of the output CSV file';

% Read file as text and process each line
fid = fopen(inputFile, 'r');
dataArray = [];
while ~feof(fid)
    line = fgetl(fid);
    tokens = regexp(line, '[-\d\.]+', 'match'); % Extract numeric values
    if numel(tokens) == 12 % Ensure correct number of columns
        dataArray = [dataArray; str2double(tokens)];
    else
        warning('Skipping line due to incorrect number of values: %s', line);
    end
end
fclose(fid);

% Convert to table and assign column names
dataTable = array2table(dataArray);
columnNames = {'state', 'accelX', 'accelY', 'accelZ', 'gyroX', 'gyroY', 'gyroZ', 'rawMagX', 'rawMagY', 'rawMagZ', 'loopCount', 'timestamp'};
dataTable.Properties.VariableNames = columnNames;

% Write to CSV
writetable(dataTable, outputFile);

disp(['CSV file saved as ', outputFile]);
