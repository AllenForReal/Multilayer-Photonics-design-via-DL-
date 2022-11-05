clear;close;clc;
f = dir('mat_*');
for i = 1:length(f)
    try
    data = load(f(i).name);
%     data = data.data;
    data(:, 1) = data(:, 1) * 1000;
    writematrix(data, f(i).name(5:end), 'Delimiter','tab')
    catch
        continue;
    end
end