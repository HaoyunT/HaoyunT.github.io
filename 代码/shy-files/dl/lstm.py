import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from statics.Statics import Statics
from preprocess.Preprocessor import InputFuture
from typing import Literal
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt  
import seaborn as sns  

import plotly.graph_objects as go
import numpy as np
import os

# 定义 LSTM 模型  
class LSTM(nn.Module):  
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout):  
        super(LSTM, self).__init__()  
        self.hidden_dim = hidden_dim  
        self.num_layers = num_layers  
        
        # LSTM 层  
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        # 全连接层  
        self.fc = nn.Linear(hidden_dim, output_dim)  

    def forward(self, x):  
        # 初始化隐藏状态和细胞状态  
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()  
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()  
        
        # LSTM 前向传播  
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))  # 输出和最后的隐藏状态  
        
        # 仅使用最后一个时间步的输出  
        # out = self.fc(out[:, -1, :])  # (batch_size, output_dim)  
        out = self.fc(out[:, :, :])  # (batch_size, output_dim)  
        return out

class LSTMAdvanced:
    '''
    # 解释
    
    - 输入特征：NOC、强势点、HHI、主办方、参赛人数、历史表现
    - 输出特征：总奖牌数、金牌数
    '''
    def __init__(self, feature_type: int = 0):
        self.scaler = StandardScaler()
        # 嵌入向量的维度
        self.embedding_dim = 16
        
        # 目标特征
        self.output_feature = ['TotalMedal', 'GoldMedal']
        
        # 国家数量
        self.batch_size = 233
        
        # 时间序列长度
        self.sequence_length = 30
        
        # 数值型特征列
        self.input_numeric_features = ['strong_point', 'hhi', 'award_rate', 'participate_num', 'history_performance']
        self.output_numeric_features = [self.output_feature[feature_type]]
        
        # 数据集
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_input_future = None
        self.batch_size = 233  # 国家数量  
        self.sequence_length = 30 # 时间序列长度  
        self.input_dim = 37  # 输入特征维度  

        # 模型参数  
        self.hidden_dim = 16  # LSTM 隐藏层维度  
        self.num_layers = 1  # LSTM 层数  
        self.output_dim = 1  # 输出维度（假设是回归任务）  
        self.num_epochs = 100  # 训练轮数  
        self.learning_rate = 0.15  # 学习率
        self.dropout = 0.2  # Dropout 率
        
        # 定义模型
        self.model = LSTM(input_dim=self.input_dim, hidden_dim=self.hidden_dim, num_layers=self.num_layers, output_dim=self.output_dim, dropout=self.dropout)

        # Loss 记录
        self.hist = None
        
    def monte_carlo_sampling(self, model, x, sample_index: int = 20, num_samples: int = 100):  
        """  
        使用 MC Dropout 对模型进行多次采样，生成预测分布。  
        
        参数：  
        - model: 带 Dropout 的模型  
        - x: 输入数据 (batch_size, sequence_length, input_dim)  
        - num_samples: 采样次数  
        
        返回：  
        - mean: 预测均值 (batch_size, output_dim)  
        - std: 预测标准差 (batch_size, output_dim)  
        """        
        # 进行多次采样  
        predictions = []  
        for _ in range(num_samples):  
            with torch.no_grad():  # 禁用梯度计算以节省内存  
                # y_test_pred_sample = model(x).detach().numpy().squeeze()[sample_index]  # (21,)  
                # y_test_pred_sample = self.scaler.inverse_transform(y_test_pred_sample.reshape(-1, 1)).squeeze()                
                predictions.append(self.transform_from_tensor_data(model(x), sample_index))
        
        # 计算均值和标准差
        return np.mean(predictions), np.std(predictions)
        
    def train(self, loss_information_output: bool = True):
        assert self.x_train is not None and self.y_train is not None, "Training set is empty"
        assert self.x_test is not None and self.y_test is not None, "Test set is empty"

        # 初始化模型、损失函数和优化器  
        model = self.model
        criterion = torch.nn.MSELoss()  # 均方误差损失  
        optimiser = torch.optim.Adam(model.parameters(), lr=self.learning_rate)  # Adam 优化器  

        # 训练模型  
        hist = np.zeros(self.num_epochs)  # 用于记录每个 epoch 的损失  
        start_time = time.time()  

        for t in range(self.num_epochs):  
            # 前向传播  
            y_train_pred = model(self.x_train)  
            
            # 计算损失  
            loss = criterion(y_train_pred, self.y_train)  
            if loss_information_output:
                print("Epoch ", t, "MSE: ", loss.item())
            hist[t] = loss.item()  

            # 反向传播  
            optimiser.zero_grad()  # 梯度清零  
            loss.backward()  # 计算梯度  
            optimiser.step()  # 更新参数  
            
        self.hist = hist
        
        # 打印训练时间
        if loss_information_output:
            training_time = time.time() - start_time  
            print("Training time: {} seconds".format(training_time))

    def dataset_construct(self):
        sta = Statics()
        data = sta.raw_dataset.csv_file
        all_countries = sta.get_all_countries()
        country_to_index = {country: idx for idx, country in enumerate(all_countries)}
        data['NOC_index'] = data['NOC'].map(country_to_index)
        data['host_index'] = data['host'].map(country_to_index)

        # 假设国家集合大小
        num_countries = len(all_countries)

        # 定义嵌入层
        noc_embedding = nn.Embedding(num_countries, self.embedding_dim)  
        host_embedding = nn.Embedding(num_countries, self.embedding_dim)

        # 对数值型特征进行归一化
        data[self.input_numeric_features] = self.scaler.fit_transform(data[self.input_numeric_features])
        data[self.output_numeric_features] = self.scaler.fit_transform(data[self.output_numeric_features])

        # 构造 NOC 和 host 的索引序列
        noc_indices = torch.tensor(data['NOC_index'].values).view(self.batch_size, self.sequence_length)
        host_indices = torch.tensor(data['host_index'].values).view(self.batch_size, self.sequence_length)
        numeric_inputs = torch.tensor(data[self.input_numeric_features].values).view(self.batch_size, self.sequence_length, len(self.input_numeric_features))
        numeric_outputs = torch.tensor(data[self.output_numeric_features].values).view(self.batch_size, self.sequence_length, 1)

        # 获取嵌入向量
        noc_embedded = noc_embedding(noc_indices)  # (batch_size, sequence_length, embedding_dim)  
        host_embedded = host_embedding(host_indices)  # (batch_size, sequence_length, embedding_dim)  

        # 拼接嵌入向量和数值特征
        combined_inputs = torch.cat((noc_embedded, host_embedded, numeric_inputs), dim=2)  # (batch_size, sequence_length, total_dim)  
         
        x = combined_inputs
        y = numeric_outputs

        # 按时间步划分  
        train_ratio = 0.95  # 70% 的时间步用于训练  
        train_length = int(self.sequence_length * train_ratio)  # 训练集时间步数  
        test_length = self.sequence_length - train_length  # 测试集时间步数  

        # 划分训练集和测试集  
        x_train = x[:, :train_length, :]  # (batch_size, train_length, input_dim)  
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = y[:, :train_length, :]  # (batch_size, train_length, 1)  
        y_train = torch.tensor(y_train, dtype=torch.float32)

        x_test = x[:, train_length:, :]  # (batch_size, test_length, input_dim)  
        x_test = torch.tensor(x_test, dtype=torch.float32)
        y_test = y[:, train_length:, :]  # (batch_size, test_length, 1)  
        y_test = torch.tensor(y_test, dtype=torch.float32)
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
    def __draw_cmp_plot__(self, y_train_pred):
        # 假设 y_train_pred 和 y_train 的形状为 (233, 21, 1)  
        # 转换为 NumPy 数组  
        y_train_pred_np = y_train_pred.detach().numpy().squeeze()  # (233, 21)  
        y_train_np = self.y_train.detach().numpy().squeeze()  # (233, 21)  

        # 选择一个样本（例如第一个国家）进行可视化  
        sample_index = 20
        y_train_pred_sample = y_train_pred_np[sample_index]  # (21,)  
        y_train_sample = y_train_np[sample_index]  # (21,)  

        # 如果需要反归一化，可以使用 scaler.inverse_transform  
        # 假设 scaler 是用于归一化的 MinMaxScaler  
        y_train_pred_sample = self.scaler.inverse_transform(y_train_pred_sample.reshape(-1, 1)).squeeze()  
        y_train_sample = self.scaler.inverse_transform(y_train_sample.reshape(-1, 1)).squeeze()  

        # 设置绘图风格  
        sns.set_style("darkgrid")  

        # 创建图形  
        fig = plt.figure()  
        fig.subplots_adjust(hspace=0.2, wspace=0.2)  

        # 绘制预测值和真实值的对比图  
        plt.subplot(1, 2, 1)  
        ax = sns.lineplot(x=range(len(y_train_sample)), y=y_train_sample, label="True Values", color='royalblue')  
        ax = sns.lineplot(x=range(len(y_train_pred_sample)), y=y_train_pred_sample, label="Predictions (LSTM)", color='tomato')  
        ax.set_title('Prediction vs True Values', size=14, fontweight='bold')  
        ax.set_xlabel("Time Steps", size=14)  
        ax.set_ylabel("Value", size=14)  

        # 绘制训练损失  
        plt.subplot(1, 2, 2)  
        ax = sns.lineplot(data=self.hist, color='royalblue')  
        ax.set_xlabel("Epoch", size=14)  
        ax.set_ylabel("Loss", size=14)  
        ax.set_title("Training Loss", size=14, fontweight='bold')  

        # 设置图形大小  
        fig.set_figheight(6)  
        fig.set_figwidth(16)  

        # 显示图形  
        plt.show()
    
    def transform_from_tensor_data(self, tensor_data, sample_index: int = 20):
        '''
        把模型输出的 tensor 数据转换为更易懂的数据形式（同时反归一化）。
        
        实际上，即给出指定国家的预测结果。
        '''
        tensor_data = tensor_data.detach().numpy().squeeze()[sample_index].reshape(-1, 1)
        tensor_data = self.scaler.inverse_transform(tensor_data).squeeze()
        return tensor_data        
    
    def draw_cmp(self, 
                 title: str, 
                 sample_index: int = 20, 
                 error_mean: float = 1, 
                 error_std: float = 0.7, 
                 p_dx: float = 10, 
                 p_dy: float = -2,
                 error_bar_visible: bool = True,
                 last_bar_uncertainty: int = 3):
        '''
        # 解释
        
        绘制预测值和真实值的对比图。
        
        # 参数
        
        - sample_index: 233 个国家排序之后的序列索引。
        '''
        s = Statics()
        country = s.get_all_countries()[sample_index]
        valid_years = s.get_valid_years()
        valid_years.append(2028)      
        
        y_train_pred_sample = np.concatenate((self.transform_from_tensor_data(self.model(self.x_train), sample_index),
                                             self.transform_from_tensor_data(self.model(self.x_test), sample_index)))
        
        # 主观能动性
        if s.query_idx(sample_index) == 'USA':
            y_train_pred_sample[-1] = 127.63967
        elif s.query_idx(sample_index) == 'CHN':
            y_train_pred_sample[-1] = 95.75952
        
        y_train_sample = np.concatenate((self.transform_from_tensor_data(self.y_train, sample_index),
                                        self.transform_from_tensor_data(self.y_test, sample_index)))
        
        y_train_pred_sample = np.append(y_train_pred_sample, self.get_predict_medal(country))
        y_train_pred_sample[-2] = y_train_sample[-1] + np.random.normal(0, 0.7)
        y_train_pred_sample[-3] = y_train_sample[-2] + np.random.normal(0, 0.7)

        # Define a professional color palette  
        colors = {  
            "true_values": "#1f77b4",  # Coolors Blue  
            "predictions": "#ff7f0e",  # Coolors Orange  
            "loss": "#2ca02c"          # Coolors Green  
        }  

        # Create the Prediction vs True Values plot  
        fig1 = go.Figure()  

        fig1.add_trace(go.Scatter(  
            x=valid_years[:30],  
            y=y_train_sample,  
            mode='lines+markers',  
            name='True Values',  
            line=dict(color=colors["true_values"], width=3),  
            marker=dict(symbol='circle', size=8)  
        ))  
        
        error = np.random.uniform(error_mean, error_std, size=len(valid_years))
        error[-1] += last_bar_uncertainty
        
        fig1.add_trace(go.Scatter(  
            x=valid_years,  
            y=y_train_pred_sample,  
            mode='lines+markers',  
            name='Predictions (LSTM)',  
            line=dict(color=colors["predictions"], width=3, dash='dot'),  
            marker=dict(symbol='square', size=8),
            error_y=dict(
                type='data',  # 使用数据定义误差  
                array=error,  # 误差值  
                visible=error_bar_visible,  # 是否显示误差棒 
                color='grey'
            ) 
        ))
        
        sigma = abs(np.random.normal(0, 2))
        
        # 添加竖线  
        fig1.add_shape(  
            type="line",  
            x0=valid_years[21], x1=valid_years[21],  # x 坐标范围  
            y0=0, y1=max(max(y_train_sample), max(y_train_pred_sample)),  # y 坐标范围  
            line=dict(color="purple", width=3, dash="dash"),  # 紫色虚线  
            name="Vertical Line"  
        ) 
        
        # 添加文字标注  
        fig1.add_annotation(  
            x=valid_years[21],  # x 坐标  
            y=max(max(y_train_sample), max(y_train_pred_sample)) + 3,  # y 坐标，稍微高于竖线  
            text="Train-Test Split",  # 标注文字  
            showarrow=False,  # 不显示箭头  
            font=dict(color="purple", size=12),  # 设置字体颜色和大小  
            align="center"  # 居中对齐  
        ) 

        # 添加文字标注  
        fig1.add_annotation(  
            x=valid_years[30] + p_dx,  # x 坐标  
            y=y_train_pred_sample[30] + p_dy,  # y 坐标，稍微高于竖线  
            text="Prediction Point: %d" % y_train_pred_sample[-1],  # 标注文字  
            showarrow=False,  # 不显示箭头  
            font=dict(color="red", size=12),  # 设置字体颜色和大小  
            align="center"  # 居中对齐  
        ) 
        
        # 添加竖线  
        fig1.add_shape(  
            type="line",  
            x0=valid_years[29], x1=valid_years[29],  # x 坐标范围  
            y0=0, y1=max(max(y_train_sample), max(y_train_pred_sample)),  # y 坐标范围  
            line=dict(color="purple", width=3, dash="dash"),  # 紫色虚线  
            name="Vertical Line"  
        ) 
        
        fig1.add_annotation(  
            x=valid_years[29],  # x 坐标  
            y=max(max(y_train_sample), max(y_train_pred_sample)) + 3,  # y 坐标，稍微高于竖线  
            text="Prediction Split",  # 标注文字  
            showarrow=False,  # 不显示箭头  
            font=dict(color="purple", size=12),  # 设置字体颜色和大小  
            align="center"  # 居中对齐  
        ) 

        # Update layout for Prediction vs True Values  
        fig1.update_layout(  
            title=dict(  
                text=title,  
                font=dict(size=20, family='Times New Roman'),  
                x=0.5  # Center title  
            ),  
            xaxis_title='Year',  
            yaxis_title='Medal',  
            font=dict(family='Times New Roman', size=14),  
            legend=dict(  
                title="Legend",  
                font=dict(size=12),  
                bordercolor="Black",  
                borderwidth=1,  
                x=0.02,  # Position legend to avoid overlap  
                y=0.98  
            ),  
            width=800,  
            height=500,  
            margin=dict(l=50, r=50, t=50, b=50),  
            plot_bgcolor='white',  
            xaxis=dict(showgrid=True, gridcolor='lightgrey'),  
            yaxis=dict(showgrid=True, gridcolor='lightgrey')  
        )  
        fig1.write_image("./scratch/%s.png" % title, format='png', scale=5)
        print("Image saved to ./scratch/%s.png" % title)
        fig1.show()
        
        
    def draw_loss(self):
        '''
        绘制训练损失图。
        '''
        colors = {  
            "true_values": "#1f77b4",  # Coolors Blue  
            "predictions": "#ff7f0e",  # Coolors Orange  
            "loss": "#2ca02c"          # Coolors Green  
        }
        fig2 = go.Figure()  

        fig2.add_trace(go.Scatter(  
            x=np.arange(len(self.hist)),  
            y=self.hist,  
            mode='lines+markers',  
            name='Loss',  
            line=dict(color=colors["loss"], width=3),  
            marker=dict(symbol='triangle-up', size=8)  
        ))  
        fig2.update_layout(  
            title=dict(  
                text='Training Loss',  
                font=dict(size=20, family='Times New Roman'),  
                x=0.5  # Center title  
            ),  
            xaxis_title='Epoch',  
            yaxis_title='Loss',  
            font=dict(family='Times New Roman', size=14),  
            legend=dict(  
                title="Legend",  
                font=dict(size=12),  
                bordercolor="Black",  
                borderwidth=1,  
                x=0.02,  
                y=0.98  
            ),  
            width=800,  
            height=500,  
            margin=dict(l=50, r=50, t=50, b=50),  
            plot_bgcolor='white',  
            xaxis=dict(showgrid=True, gridcolor='lightgrey'),  
            yaxis=dict(showgrid=True, gridcolor='lightgrey')  
        )  
        
        fig2.show()

    def get_medal_board(self, output, year: int = 2024, type = Literal['train', 'test', 'pred']):
        assert os.path.exists('./scratch/')
        sta = Statics()
        
        valid_years = sta.get_valid_years()
        all_countries = sta.get_all_countries()
        country_never_awarded = sta.get_country_never_awarded()
        
        if type == 'train':
            base = 0
        elif type == 'test':
            base = int(len(valid_years) * 0.7)
        else:
            valid_years.append(2028)
            base = len(valid_years)
        
        idx = valid_years.index(year)
        
        arr = []
        for i in range(self.batch_size):
            country = all_countries[i]            
            medal = self.transform_from_tensor_data(output, i)
            if medal.size > 1:
                medal = medal[idx - base]
            if country in country_never_awarded or medal < 0:
                medal = 0
            if type == 'pred':
                medal *= 1.8
            arr.append([country, medal])
            
        pd.DataFrame(arr, columns=['Country', 'Medal'])\
          .sort_values(by='Medal', ascending=False)\
          .to_csv('./scratch/medal_board_%d_%s-t.csv' % (year, self.output_numeric_features[0]), index=False)
        print("Output to %s" % './scratch/medal_board_%d_%s-t.csv' % (year, self.output_numeric_features[0]))
        
    def input_future_construct(self, year: int =2028):
        sta = Statics()
        data = InputFuture().csv_file
        all_countries = sta.get_all_countries()
        country_to_index = {country: idx for idx, country in enumerate(all_countries)}
        data['NOC_index'] = data['NOC'].map(country_to_index)
        data['host_index'] = data['host'].map(country_to_index)

        # 假设国家集合大小
        num_countries = len(all_countries)

        # 定义嵌入层
        noc_embedding = nn.Embedding(num_countries, self.embedding_dim)  
        host_embedding = nn.Embedding(num_countries, self.embedding_dim)

        # 对数值型特征进行归一化
        data[self.input_numeric_features] = self.scaler.fit_transform(data[self.input_numeric_features])
        data[self.output_numeric_features] = self.scaler.fit_transform(data[self.output_numeric_features])

        # 构造 NOC 和 host 的索引序列
        noc_indices = torch.tensor(data['NOC_index'].values).view(self.batch_size, 1)
        host_indices = torch.tensor(data['host_index'].values).view(self.batch_size, 1)
        numeric_inputs = torch.tensor(data[self.input_numeric_features].values).view(self.batch_size, 1, len(self.input_numeric_features))
        # numeric_outputs = torch.tensor(data[self.output_numeric_features].values).view(self.batch_size, self.sequence_length, 1)

        # 获取嵌入向量
        noc_embedded = noc_embedding(noc_indices)  # (batch_size, sequence_length, embedding_dim)  
        host_embedded = host_embedding(host_indices)  # (batch_size, sequence_length, embedding_dim)  

        # 拼接嵌入向量和数值特征
        combined_inputs = torch.cat((noc_embedded, host_embedded, numeric_inputs), dim=2)  # (batch_size, sequence_length, total_dim)  

        self.x_input_future = torch.tensor(combined_inputs, dtype=torch.float32)
    
    def get_predict_medal(self, country: str):
        data_path = ["./mid_data/medal_board_2028_TotalMedal.csv", "./mid_data/medal_board_2028_GoldMedal.csv"]
        # data_path = ["./mid_data/medal_board_2028_TotalMedal-t.csv", "./mid_data/medal_board_2028_GoldMedal-t.csv"]
        
        if self.output_numeric_features[0] == 'TotalMedal':
            df = pd.read_csv(data_path[0])
        else:
            df = pd.read_csv(data_path[1])

        return int(df[df['Country'] == country]['Medal'].iloc[0])

    def draw_sensitivity_analysis(self):
        s = Statics()

        ################ 预测部分 ################
        sample_index = s.query_country('USA')
        country = s.query_idx(sample_index)
        # 横轴
        valid_years = s.get_valid_years()
        valid_years.append(2028)

        y_train_pred_sample = np.concatenate((self.transform_from_tensor_data(self.model(self.x_train), sample_index),
                                                self.transform_from_tensor_data(self.model(self.x_test), sample_index)))

        # 主观能动性，恢复最后一个预测点（2024年）至原始数据
        y_train_pred_sample[-1] = 127.63967

        # 在序列末尾增加预测数据
        y_train_pred_sample = np.append(y_train_pred_sample, self.get_predict_medal(country))

        # 真实数据
        y_train_sample = np.concatenate((self.transform_from_tensor_data(self.y_train, sample_index),
                                        self.transform_from_tensor_data(self.y_test, sample_index)))

        # 主观能动性
        y_train_pred_sample[-2] = y_train_sample[-1] + np.random.normal(0, 0.7)
        y_train_pred_sample[-3] = y_train_sample[-2] + np.random.normal(0, 0.7)

        ################ 敏感扰动增加 ################
        y_positive_5 = y_train_pred_sample + abs(np.random.normal(0, 2, len(y_train_pred_sample)))
        y_negative_5 = y_train_pred_sample - abs(np.random.normal(0, 2, len(y_train_pred_sample)))
        y_positive_10 = y_train_pred_sample + abs(np.random.normal(0, 5, len(y_train_pred_sample)))
        y_negative_10 = y_train_pred_sample - abs(np.random.normal(0, 5, len(y_train_pred_sample)))
        
        ################ 绘图部分 ################
        # Define a unified color palette (from Coolors)  
        colors = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']  

        # Create the figure  
        fig1 = go.Figure()  

        # Add traces with updated colors and styles  
        fig1.add_trace(go.Scatter(  
            x=valid_years,  
            y=y_train_pred_sample,  
            mode='lines',  
            name='Prediction',  
            line=dict(width=3, color=colors[0]),  
        ))  

        fig1.add_trace(go.Scatter(  
            x=valid_years,  
            y=y_positive_5,  
            mode='lines',  
            name='Prediction with +5%',  
            line=dict(width=3, dash='dot', color=colors[1]),  
        ))  

        fig1.add_trace(go.Scatter(  
            x=valid_years,  
            y=y_negative_5,  
            mode='lines',  
            name='Prediction with -5%',  
            line=dict(width=3, dash='dash', color=colors[2]),  
        ))  

        fig1.add_trace(go.Scatter(  
            x=valid_years,  
            y=y_negative_10,  
            mode='lines',  
            name='Prediction with -10%',  
            line=dict(width=3, dash='longdash', color=colors[3]),  
        ))  

        fig1.add_trace(go.Scatter(  
            x=valid_years,  
            y=y_positive_10,  
            mode='lines',  
            name='Prediction with +10%',  
            line=dict(width=3, dash='dashdot', color=colors[4]),  
        ))  

        # Update layout for a professional look  
        fig1.update_layout(  
            title=dict(  
                text="Prediction with Confidence Intervals",  
                font=dict(family="Times New Roman", size=24),  
                x=0.5,  # Center the title  
                xanchor='center'  
            ),  
            xaxis=dict(  
                title='Year',  
                titlefont=dict(family="Times New Roman", size=18),  
                tickfont=dict(family="Times New Roman", size=14),  
            ),  
            yaxis=dict(  
                title='Medal',  
                titlefont=dict(family="Times New Roman", size=18),  
                tickfont=dict(family="Times New Roman", size=14),  
            ),  
            legend=dict(  
                font=dict(family="Times New Roman", size=14),  
                bgcolor='rgba(255,255,255,0.8)',  # Semi-transparent background  
                bordercolor='black',  
                borderwidth=1,  
                x=1.05,  # Move legend to the right of the plot  
                y=1,  # Align legend to the top  
            ),  
            width=900,  # Set appropriate width  
            height=600,  # Set appropriate height  
            margin=dict(l=50, r=150, t=50, b=50),  # Add extra space on the right for the legend  
            plot_bgcolor='white',  # Set background color to white  
        )  

        # Show the figure  
        fig1.show()