clear variables; close all; clc;

sensorData = readtable('sensor_data.csv');

% Accelerometer Data
figure('Name', 'Accel');
plot(sensorData.timestamp, sensorData.accelZ);
title('Acceleration Z');
grid("on");
xlabel("Time (s)");
ylabel("Acceleration Z (G)");

sagamoreDeskEndIdx = find(sensorData.timestamp == 80529);
sagamoreDoorEndIdx = find(sensorData.timestamp == 115709);
sagamoreFarStartIdx = find(sensorData.timestamp == 197779);
sagamoreFarEndIdx = find(sensorData.timestamp == 255289);

sagamoreDeskMag = [sensorData.rawMagX(1:sagamoreDeskEndIdx), sensorData.rawMagY(1:sagamoreDeskEndIdx), sensorData.rawMagZ(1:sagamoreDeskEndIdx)];
sagamoreDoorMag = [sensorData.rawMagX(sagamoreDeskEndIdx:sagamoreDoorEndIdx), sensorData.rawMagY(sagamoreDeskEndIdx:sagamoreDoorEndIdx), sensorData.rawMagZ(sagamoreDeskEndIdx:sagamoreDoorEndIdx)];
sagamoreFarMag = [sensorData.rawMagX(sagamoreFarStartIdx:sagamoreFarEndIdx), sensorData.rawMagY(sagamoreFarStartIdx:sagamoreFarEndIdx), sensorData.rawMagZ(sagamoreFarStartIdx:sagamoreFarEndIdx)];

% figure('Name', 'Desk Mag');
% scatter3(sagamoreDeskMag(:,1), sagamoreDeskMag(:, 2), sagamoreDeskMag(:,3));
% title('Desk Mag');
% axis equal;
% xlabel('Mag X (nT)');
% ylabel('Mag Y (nT)');
% zlabel('Mag Z (nT)');
% 
% figure('Name', 'Door Mag');
% scatter3(sagamoreDoorMag(:,1), sagamoreDoorMag(:, 2), sagamoreDoorMag(:,3));
% title('Door Mag');
% axis equal;
% xlabel('Mag X (nT)');
% ylabel('Mag Y (nT)');
% zlabel('Mag Z (nT)');
% 
% figure('Name', 'Far Door Mag');
% scatter3(sagamoreFarMag(:,1), sagamoreFarMag(:, 2), sagamoreFarMag(:,3));
% title('Far Door Mag');
% axis equal;
% xlabel('Mag X (nT)');
% ylabel('Mag Y (nT)');
% zlabel('Mag Z (nT)');

% Compute histograms for magnetometer components at each location
locations = {'Desk', 'Door', 'Far'};
components = {'X', 'Y', 'Z'};
colors = {'r', 'g', 'b'};

dataStruct = struct('Desk', sagamoreDeskMag, ...
                    'Door', sagamoreDoorMag, ...
                    'Far', sagamoreFarMag);

figure;
for locIdx = 1:length(locations)
    location = locations{locIdx};
    magData = dataStruct.(location);

    for compIdx = 1:3
        subplot(3,3, (locIdx-1)*3 + compIdx);
        histogram(magData(:,compIdx), 30, 'FaceColor', colors{compIdx});
        title([location ' Mag ' components{compIdx}]);
        xlabel(['Mag ' components{compIdx} ' (nT)']);
        ylabel('Frequency');
        grid on;
    end
end

% Compute mean and standard deviation for each location
stats = struct();
fprintf('Magnetometer Statistics:\n');
for locIdx = 1:length(locations)
    location = locations{locIdx};
    magData = dataStruct.(location);

    meanVals = mean(magData);
    stdVals = std(magData);

    stats.(location).Mean = meanVals;
    stats.(location).StdDev = stdVals;

    fprintf('\n%s:\n', location);
    fprintf('  Mean (X, Y, Z): [%.2f, %.2f, %.2f] nT\n', meanVals);
    fprintf('  Std Dev (X, Y, Z): [%.2f, %.2f, %.2f] nT\n', stdVals);
end

% Apply PCA to analyze dominant variation directions
figure;
for locIdx = 1:length(locations)
    location = locations{locIdx};
    magData = dataStruct.(location);

    % Center the data
    magDataCentered = magData - mean(magData);

    % Compute PCA
    [coeff, score, latent] = pca(magDataCentered);

    % Plot first two principal components
    subplot(1,3,locIdx);
    scatter(score(:,1), score(:,2), 20, 'filled');
    title([location ' PCA Projection']);
    xlabel('PC1');
    ylabel('PC2');
    grid on;
    axis equal;
end