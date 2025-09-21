% 读取 CSV 文件  
data = readtable('data_hhi.csv'); % 假设文件名为 data.csv  

% 提取年份和 HHIndex 数据  
years = data.Year;  
hhIndex = data.HHIndex;  

% 绘制图形  
figure;  
plot(years, hhIndex, '-o', 'LineWidth', 1.5, 'MarkerSize', 6);  
grid on;  

% 设置图形标题和轴标签  
title('HHIndex Over the Years');  
xlabel('Year');  
ylabel('HHIndex');  

% 设置 x 轴刻度  
xticks(years);  
xtickangle(45); % 旋转 x 轴刻度以便更清晰显示  

% 显示图形  
legend('HHIndex', 'Location', 'best');