# Guide Flow Decoder V1 项目总结

## 1. 项目定位

`guide_flow_decoder` 是规划训练流水线中的一个任务目录，核心目标是：

1. 复用共享编码器 `PlanningSharedEncoderV5` 提取场景上下文。
2. 基于一组离散 anchor 轨迹，先用 `Flow Match` 分支生成基础轨迹修正。
3. 再结合 `guide_flow_energy_feature` 中提供的场景类型和引导流路径信息，用 `Energy Field` 分支对轨迹进行引导式修正。
4. 训练时输出完整中间状态用于 loss、评估和可视化；部署时只导出 decoder 子模块，输出 `topk_trajectories`。

从代码实现上看，这个项目本质上是一个“共享编码器 + 双分支轨迹解码器”的组合任务：

- 编码器：负责把 ego、障碍物、道路图、红绿灯等多模态特征编码成统一 token。
- Flow Match 分支：从 anchor 轨迹出发，学习一个 velocity field，把 anchor 推向更接近 GT 的轨迹。
- Energy Field 分支：根据 guide flow polygon / path 对所有 anchor 预测额外的引导速度，使轨迹更符合指定的通行区域或通行意图。

当前目录名是 `guide_flow_decoder`，但脚本里的 `job_task` / `job_model` 仍保留了历史命名，如 `LaneChange`、`lane_change_decoder_v4`。这说明该任务很大概率是从 lane change 任务演进出来的，在阅读和排查时要注意“名字是旧的，逻辑是新的”。

---

## 2. 一句话看懂调用链

完整链路如下：

```text
scripts/train_feature_pool.sh
  -> trainer.py
    -> PlanningBaseTrainer
      -> Model
        -> WrappedSharedEncoder
        -> GuideFlowDecoderV1Wrapper
          -> GuideFlowDecoderV1
            -> flow_match_forward()
            -> energy_field_forward()
            -> get_loss()
      -> Evaluator
      -> Visualizer
```

如果走 feature pool 数据：

```text
config_feature_pool.yaml
  -> planning_dataset.py / FeaturePoolDataset
    -> tensor_maker_for_feature_pool_data.py
      -> planning_feature_data_index.proto
      -> planning_model_feature.proto
      -> *.pb.txt feature config
```

### 2.1 网络架构总图

下面这张图更偏“训练态视角”，把数据输入、共享编码器、两条 decoder 分支、融合逻辑和 loss 放到一张图里：

```text
                                   +----------------------------------+
                                   | dense_cluster_centers.npy        |
                                   | anchor vocabulary [2398, 80, 2] |
                                   +----------------+-----------------+
                                                    |
                                                    v
+---------------------+    +----------------------------------+    +----------------------+
| EGO_FEATURE         |    | WrappedSharedEncoder             |    | GUIDE_FLOW_ENERGY_   |
| MOVING_OBSTACLE     |--->|  PlanningSharedEncoderV5         |    | FEATURE [1, 34]      |
| STATIC_OBSTACLE     |    |                                  |    |  scene_id + 16 xy    |
| ROAD_GRAPH_SEGMENT  |    | output:                          |    +----------+-----------+
| TRAFFIC_LIGHT       |    |  - encoded_feature               |               |
+---------------------+    |  - encoded_feature_mask          |               v
                           +----------------+-----------------+    +----------------------+
                                            |                      | split_guide_flow_    |
                                            |                      | energy_feature()     |
                                            |                      |  - scene_feature     |
                                            |                      |  - polygon_coords    |
                                            |                      +----------+-----------+
                                            |                                 |
                                            |                                 |
                                            v                                 v
                           +----------------+-----------------+    +----------+-----------+
                           | flow_match_forward()             |    | energy_field_forward()|
                           |                                  |    |                       |
                           | train only:                      |    | scene projector       |
                           |  - GT find nearest/topk anchor   |    | polygon projector     |
                           |  - anchor + noise + GT interp    |    | anchor traj projector |
                           |                                  |    | cross attention       |
                           | anchor projector                 |    | energy head           |
                           | cross attention with encoder     |    |                       |
                           | velocity head                    |    | output: EF_velocity   |
                           |                                  |    +----------+-----------+
                           | output: FM_output_trajectory     |               |
                           |         anchor_velocity          |               |
                           +----------------+-----------------+               |
                                            |                                 |
                                            +---------------+-----------------+
                                                            |
                                                            v
                                          +-----------------+-----------------+
                                          | _process_forward()                |
                                          |                                   |
                                          | iter < 3 : output = FM + EF       |
                                          | iter >= 3: output = FM only       |
                                          +-----------------+-----------------+
                                                            |
                                  +-------------------------+------------------------+
                                  |                                                  |
                                  v                                                  v
                  +---------------+------------------+              +----------------+----------------+
                  | training / validation output     |              | inference / deploy output        |
                  | - output_trajectory              |              | - iterative refine x 6          |
                  | - nearest_anchor_indices         |              | - select topk_trajectories      |
                  | - FM / EF intermediate tensors   |              |                                |
                  +---------------+------------------+              +----------------+----------------+
                                  |                                                  |
                                  v                                                  v
                  +---------------+------------------+              +----------------+----------------+
                  | get_loss()                       |              | exported decoder output          |
                  | - FM_regression_loss             |              | - topk_trajectories             |
                  | - EF_regression_loss             |              +---------------------------------+
                  | - kinematic constraint losses    |
                  +----------------------------------+
```

如果换成一句话概括，就是：

```text
多模态场景特征 --共享编码--> encoded tokens
anchor词表 + encoded tokens --Flow Match--> 基础轨迹修正
guide flow条件 + encoded tokens --Energy Field--> 引导式轨迹修正
两支结果融合 --> output_trajectory / topk_trajectories
```

下面这张图是更偏“部署态视角”的简化版：

```text
planning inputs
  -> WrappedSharedEncoder
    -> encoded_feature + encoded_feature_mask
      -> GuideFlowDecoderV1
        -> Flow Match branch
        -> Energy Field branch
        -> iterative refine (6 rounds)
        -> topk_trajectories
```

---

## 3. 目录内文件总览

### 3.1 任务目录文件

| 文件 | 作用 |
| --- | --- |
| `model.py` | 任务级模型封装。把共享编码器和 `GuideFlowDecoderV1` 组装成统一的训练/推理接口。 |
| `trainer.py` | 训练入口。继承 `PlanningBaseTrainer`，创建模型、加载权重、创建 evaluator 和 visualizer。 |
| `evaluator.py` | 评估逻辑。实现多轮迭代推理，并计算轨迹误差指标。 |
| `visualizer.py` | 可视化逻辑。绘制场景、GT、anchor、预测轨迹、top-k 关系箭头、guide flow path。 |
| `config/config.yaml` | 普通 `TENSOR_DATAPACK` 训练配置。依赖传统 tensor 数据文件列表。 |
| `config/config_feature_pool.yaml` | `TENSOR_FEATURE_POOL` 训练配置。当前 guide_flow_v1 的主配置更像是这一份。 |
| `config/config_local.yaml` | 本地或快速调试配置。batch、epoch、数据路径都较小。 |
| `config/vector_image_config.json` | 可视化层开关。定义图像中显示哪些层。 |
| `config/cluster_centers.npy` | 较小的 anchor 词表，shape 是 `(250, 80, 2)`，目前默认未被主 decoder 使用。 |
| `config/dense_cluster_centers.npy` | 当前主 decoder 默认加载的 anchor 词表，shape 是 `(2398, 80, 2)`。 |
| `scripts/train.sh` | 基于 `config.yaml` 的 HOPE 提交脚本。 |
| `scripts/train_feature_pool.sh` | 基于 `config_feature_pool.yaml` 的 HOPE 提交脚本。 |
| `BUILD` | Bazel 构建定义，暴露 `model`、`evaluator`、`visualizer`、`trainer`、`trainer_package`。 |
| `config/BUILD` | 将配置目录打成 Bazel `filegroup`，供训练和部署打包时携带。 |

### 3.2 任务强依赖的公共文件

| 文件 | 作用 |
| --- | --- |
| `../../decoder/planning_guide_flow_decoder_v1.py` | 项目最核心的 decoder 实现，几乎所有关键算法都在这里。 |
| `../../planning_base_trainer.py` | 通用训练基类，负责数据集、优化器、验证、部署封装等公共逻辑。 |
| `../../dataset/planning_dataset.py` | 根据 `file_type` 构建 `Dataset` 或 `FeaturePoolDataset`。 |
| `../../dataset/planning_dataset_config.py` | 数据集配置合成和 feature config 路径校验。 |
| `../../dataset/tensor_maker_for_feature_pool_data.py` | 读取 feature pool 样本，把二进制样本恢复成模型输入 dict / label dict。 |
| `../../dataset/tensor_maker_for_feature_pool_data_test.py` | feature pool tensor maker 单测，覆盖重复特征、缺失特征、shape 校验和 end-to-end 读取。 |
| `../../dataset/planning_dataset_config_test.py` | dataset config 单测，覆盖 `TENSOR_FEATURE_POOL` 配置校验。 |
| `../../common/const_exprs.py` | 特征名、标签名、meta info 名称的统一定义。 |
| `../../proto/planning_model_feature.proto` | 规划训练中所有 feature / label 的枚举和 config proto 定义。 |
| `../../proto/planning_feature_data_index.proto` | feature pool 样本索引格式定义。 |
| `../../encoder/planning_shared_encoder_v5.py` | 共享编码器实现，guide flow 任务通过 wrapper 直接复用。 |

### 3.3 feature pool 模式下还会读取的外部特征定义文件

这些文件不在任务目录里，但 `config_feature_pool.yaml` 强依赖它们：

目录：

```text
walle2/config/module/planning/learning/feature/feature_config_version/
```

涉及文件：

| 文件 | 张量 shape | 作用 |
| --- | --- | --- |
| `guide_flow_energy_config_v1.pb.txt` | `[1, 34]` | Guide flow 特征。第 0 维是场景类型，后 32 维是 16 个 `(x, y)` 点。 |
| `ego_feature_config_v1.pb.txt` | `[1, 40, 8]` | Ego 历史特征。 |
| `moving_obstacle_feature_config_v1.pb.txt` | `[64, 40, 12]` | 动态障碍物历史特征。 |
| `static_obstacle_feature_config_v1.pb.txt` | `[32, 1, 60]` | 静态障碍物特征。 |
| `road_graph_segment_feature_config_v1.pb.txt` | `[450, 1, 13]` | 道路图 segment 特征。 |
| `traffic_light_feature_config_v2.pb.txt` | `[10, 40, 10]` | 红绿灯时序特征。 |
| `ego_lane_query_feature_config_v1.pb.txt` | `[10, 100, 2]` | Ego lane query 特征。 |
| `reference_line_segment_feature_config_v1.pb.txt` | `[50, 1, 12]` | 参考线 segment 特征。 |
| `route_aggregated_map_feature_config_v1.pb.txt` | `[64, 64]` | 聚合路由特征。 |
| `ego_lane_query_meta_label_config_v1.pb.txt` | `[10, 6]` | lane query 元标签。 |
| `ego_trajectory_label_config_v1.pb.txt` | `[1, 80, 2]` | Ego 未来轨迹标签。 |
| `obstacle_trajectory_label_config_v1.pb.txt` | `[64, 80, 4]` | 障碍物未来轨迹标签。 |

---

## 4. 核心模型结构

## 4.1 任务级模型封装：`model.py`

`model.py` 里定义了三个关键类：

### `WrappedSharedEncoder`

作用：

1. 包装 `PlanningSharedEncoderV5`，解决预训练权重 `state_dict` 的前缀兼容问题。
2. 统一 encoder 的部署接口和训练接口。
3. 在训练时对 ego 历史做一个简化版 dropout / padding，增加鲁棒性。

当前启用的是 “SE 5.2 风格” 的 wrapper：

- 训练态下将部分 ego 历史帧置为 `-300.0`。
- 输入特征包括：
  - `EGO_FEATURE`
  - `MOVING_OBSTACLE_FEATURE`
  - `STATIC_OBSTACLE_FEATURE`
  - `ROAD_GRAPH_SEGMENT_FEATURE`
  - `TRAFFIC_LIGHT_FEATURE`
- 输出：
  - `encoded_feature`
  - `encoded_feature_mask`

### `GuideFlowDecoderV1Wrapper`

作用：

1. 适配 `GuideFlowDecoderV1` 的任务级 `forward` 接口。
2. 区分训练、验证、多轮迭代验证和部署推理。
3. 负责把输入字典中的 key 映射给底层 decoder。

这里有一个很重要的细节：

- 推理 / deploy 路径读取的 key 是 `guide_flow_energy_feature`。
- 训练 / 验证路径读取的 key 是 `GUIDE_FLOW_ENERGY_FEATURE`。

也就是说，这个任务当前同时存在“小写 deploy 输入名”和“训练态 proto 名大写 key”两套命名体系。README 特别记录这一点，是因为这类命名差异非常容易在接模型或改导出逻辑时踩坑。

### `Model`

作用：

1. 将 `_encoder` 和 `_decoder` 串联起来。
2. `forward()` 中先跑 encoder，再把编码结果和原始输入一起交给 decoder。
3. loss 完全委托给 decoder。

所以从训练框架角度看，整个任务模型只是：

```text
Model
  |_ _encoder = WrappedSharedEncoder
  |_ _decoder = GuideFlowDecoderV1Wrapper
```

---

## 4.2 核心 decoder：`planning_guide_flow_decoder_v1.py`

这个文件是整个 guide flow v1 的“大脑”。它很长，但其实可以拆成 5 层来看。

### 第一层：几何与掩码辅助函数

这部分函数围绕“轨迹是否进入 / 停留 / 从哪条边离开 guide flow polygon”展开：

- `_points_in_polygon_or_on_boundary_torch_single`
- `_points_in_polygon_or_on_boundary`
- `_find_first_crossed_edges_vectorized`
- `_analyze_batch_touch_and_exit`
- `_build_nearest_good_trajectory_index`
- `_build_nearest_reference_good_trajectory_index_torch`
- `_build_batch_nearest_reference_good_trajectory_index_torch`
- `_build_anchor_polygon_mask_gpu`
- `build_anchor_polygon_mask`

这些函数的目标是把每条 anchor 轨迹分成：

- 好轨迹：触达 polygon，并满足“不离开”或“只从目标边离开”等条件。
- 坏轨迹：不满足 guide flow 约束的轨迹。

`TrajectoryMaskResult` 会返回：

- `good_trajectory_mask`
- `bad_trajectory_mask`
- `nearest_good_trajectory_index`
- `nearest_good_trajectory_distance`
- `first_touch_index`
- `exit_edge_index`
- `target_edge_index`
- `required_time_inside_polygon`

这正是 Energy Field loss 的基础。

### 第二层：通用轨迹约束辅助函数

包括：

- `min_limitation_loss`
- `max_limitation_loss`
- `smoothness_loss`
- `ego_history_trajectory`
- `ego_history_validity`

这些函数服务于速度、加速度、jerk、曲率等约束 loss。

### 第三层：两个回归头

#### `VelocityFieldRegressionHead`

作用：

- 输入 decoder 输出的 hidden feature。
- 输出每个 anchor 对应的 velocity 向量，shape 为 `num_trajectory_points * 2`，这里默认就是 `80 * 2 = 160`。

训练逻辑：

- 对 GT 最近 anchor 做主回归。
- 同时对 top-k 近邻 anchor 做扩展回归。
- 本质是在学习“如何把 noisy anchor 朝 GT 方向推过去”。

#### `EnergyFieldRegressionHead`

作用：

- 同样输出每个 anchor 的 velocity 向量，但它表达的是“guide flow 引导速度”。

训练逻辑：

1. 如果一条 noisy anchor 已经是好轨迹，则 `target_velocity = 0`，学习“保持不变”。
2. 如果 noisy anchor 是坏轨迹，则查找最近的 clean good trajectory，令 `target_velocity = nearest_good - noisy_anchor`，学习“被引导回好区域”。

这部分 loss 的核心思想是：

- 好轨迹学保持。
- 坏轨迹学纠正。

这是整个 guide flow v1 与普通 anchor decoder 最大的差异之一。

### 第四层：注意力模块

- `FFN`
- `CrossAttentionLayer`

这两个模块相对通用，本质是 decoder 内部的小型 cross-attention block：

- query：anchor 或 query feature
- key/value：共享编码器输出的 `encoded_feature`

### 第五层：主类 `GuideFlowDecoderV1`

这是实际的任务 decoder。

#### 默认配置

核心默认项包括：

- `project_dim = 256`
- `num_heads = 2`
- `vocabulary_path = tasks/guide_flow_decoder/config/dense_cluster_centers.npy`
- `num_trajectory_points = 80`
- 多种运动学约束阈值，如 `max_speed`、`max_acceleration`、`max_curvature`

#### anchor 词表

初始化时会读取：

```text
config/dense_cluster_centers.npy
```

当前 shape 是：

```text
(2398, 80, 2)
```

也就是共有 2398 条先验 anchor，每条 80 个点，每个点 2 维 `(x, y)`。

#### `dense_cluster_centers.npy` 最可能的离线生成流程（反推）

这一段不是代码里明写的官方流程，而是基于仓内现有文件、notebook 痕迹和 `npy` 本身统计特征做的高概率反推。

先给结论：

- `dense_cluster_centers.npy` 几乎可以确定不是在线训练时动态生成的，而是离线预计算好的 anchor vocabulary。
- 它最可能来自一批 `EGO_TRAJECTORY_LABEL` 候选轨迹的聚类 / 去重 / 稠密覆盖过程。
- `config/cluster_centers.npy` 更像早期小词表实验结果。
- `config/dense_cluster_centers.npy` 更像在更大数据量、或更小聚类阈值下重新跑出来的“稠密版词表”，而不是简单把 250 条直接 append 到 2398 条里。

##### 已确认的证据

1. decoder 默认直接从磁盘加载这个文件

- `planning_guide_flow_decoder_v1.py` 中 `DEFAULT_CONFIG["vocabulary_path"]` 默认指向：

```text
common/python/training/planning_training_pipeline/tasks/guide_flow_decoder/config/dense_cluster_centers.npy
```

- 初始化时直接：

```python
vocabulary_trajectory = np.load(self._config.vocabulary_path)
self.register_buffer('_anchor_trajectory', torch.from_numpy(vocabulary_trajectory).float())
```

说明它是训练前就准备好的静态先验，不依赖当前 batch 在线计算。

2. 仓里唯一明确“生成 cluster centers”的线索来自 `walle2/new_mode_function-Copy1.ipynb`

- notebook 里先加载：

```python
smoothed_trajectory_ = np.load('tarjectory_candidates.npy')
```

- 然后 reshape 成：

```python
smoothed_trajectory_reshape = smoothed_trajectory_.reshape(-1, 80, 2)
```

- 再调用一个 GPU 聚类原型：

```python
clusters, cluster_centers = trajectory_clustering_gpu_memory_efficient(
    smoothed_trajectory_reshape[:1000],
    init_trajectory_reshape,
    2.0,
    1000000,
)
np.save("cluster_centers.npy", cluster_centers)
```

- notebook 输出里能看到保存后的 shape 是：

```text
(250, 80, 2)
```

而仓库根目录的 `walle2/cluster_centers.npy` 与任务目录中的 `config/cluster_centers.npy` 内容完全一致。这基本可以说明：

- 250 条 `cluster_centers.npy` 确实来自这条 notebook 风格的离线轨迹聚类链路。
- `guide_flow_decoder` 后来直接复用了这份产物。

3. `dense_cluster_centers.npy` 与 250 条小词表风格一致，但不是它的逐条超集

- `cluster_centers.npy` 的 shape 是 `(250, 80, 2)`。
- `dense_cluster_centers.npy` 的 shape 是 `(2398, 80, 2)`。
- 两者起点都接近原点，终点几乎都在前向区域，明显像 ego-centric future trajectory，而不像地图 polyline。
- 但把两者做近似匹配后，250 条小词表并不是 dense 词表的逐条子集。

这说明更高概率的情况是：

- dense 版本是用同类方法、同类数据、但不同阈值 / 不同候选集重新生成的；
- 而不是“先有 250 条，再原样追加 2148 条”。

4. dense 版本明显更“稠密”

从词表内部最近邻距离看：

- 250 版最近邻距离均值大约是 `25.6`
- 2398 版最近邻距离均值大约是 `13.2`

这和文件名里的 `dense` 是一致的，说明它的目标就是让 anchor 空间覆盖得更细。

##### 最可能的完整流程

下面这条链路是我认为目前最像真实生成过程的版本。

1. 从规划训练数据中导出 ego future trajectory 候选

- 数据源大概率不是 guide flow 特有逻辑，而是更通用的规划样本导出链路。
- 最可能使用的是 feature pool / datapack 中的 `EGO_TRAJECTORY_LABEL`。
- 在当前任务配置里，这个 label 对应的 shape 就是 `[1, 80, 2]`。

也就是说，离线聚类前，原始候选轨迹大概率长这样：

```text
[sample_num, 80, 2]
```

或先以扁平形式保存成：

```text
[sample_num, 160]
```

再在脚本里 reshape 成 `[sample_num, 80, 2]`。

2. 做一轮清洗和标准化

最可能包含以下处理：

- 去掉 padding / 无效点过多的轨迹。
- 统一到 ego-centric 坐标系。
- 只保留未来 80 个点的 `(x, y)`。
- 可能做轻量平滑，所以 notebook 变量名叫 `smoothed_trajectory_`。

这一步的输出，大概率就是 notebook 中看到的：

```text
tarjectory_candidates.npy
```

名字里 `tarjectory` 虽然拼写错了，但这反而很像一次真实实验中留下的中间文件名。

3. 对候选轨迹做离线聚类 / 覆盖式去重

从 notebook 原型看，这一类算法的核心目标不是做语义聚类，而是做“轨迹空间覆盖”：

- 先选一条轨迹作为当前中心。
- 找出所有与它足够接近的轨迹。
- 这些轨迹视作已经被当前中心覆盖。
- 把它们移出待处理集合。
- 再从剩余轨迹里挑下一条代表轨迹，重复直到收敛。

从 notebook 代码看，prototype 使用的距离更接近：

- 对 80 个点逐点算 L2
- 再在时间维上求平均

而在线训练阶段，`flow_match_forward()` 里用于“GT 找最近 anchor”的距离则是：

- 把整条 80 点轨迹 flatten 成 160 维
- 用 `torch.cdist`
- 再除以 `sqrt(num_points)`，本质更接近 RMS point distance

这两种度量不是完全相同，但都在表达“整条轨迹的平均点位差”。所以最合理的判断是：

- 早期离线聚类原型先用了平均逐点 L2；
- 最终 dense 词表可能沿用了类似距离，也可能切到了更接近训练时使用的 RMS 距离。

4. 先得到一个小词表版本用于快速验证

这一步对应仓里的：

```text
config/cluster_centers.npy
```

它大概率承担的是：

- 快速验证 anchor 机制是否能跑通；
- 先粗看轨迹空间是否被覆盖；
- 给可视化和调试提供一个规模可控的 vocabulary。

从 notebook 留下的参数痕迹看，250 版很像是：

- 在较小候选集上试跑；
- 或者使用了较大的覆盖阈值；
- 因而每个 center 覆盖范围更大，最终 center 数更少。

5. 再生成 dense 版本作为正式 guide flow decoder 的默认词表

这一阶段最可能有两种实现路径：

路径 A，更可能：

- 直接在更大规模的候选轨迹集上重新聚类；
- 使用更小的覆盖阈值；
- 最终一次性产出 2398 条更细粒度的 centers。

路径 B，也有可能：

- 先用小词表过滤掉“已覆盖”的轨迹；
- 对剩余难样本 / 长尾轨迹再做一轮更细聚类；
- 但最终不是简单拼接，而是又做过一次重排、重采样或整体重生成。

我更偏向路径 A，原因是：

- `dense_cluster_centers.npy` 不是 250 版的逐条超集；
- 如果只是“小词表 + 新增 residual centers 直接拼接”，通常会保留一部分完全相同的旧 center；
- 但当前仓里的两个文件并不满足这一点。

6. 将最终结果保存为 `float32` 的 `npy`，并拷贝到任务配置目录

最后落地结果就是：

```text
common/python/training/planning_training_pipeline/tasks/guide_flow_decoder/config/dense_cluster_centers.npy
```

随后 decoder 在初始化时把它注册成 `_anchor_trajectory`，供 FM 和 EF 两个分支共同使用。

##### 如果今天要复刻，我会按这条链路做

如果目标不是“猜当年的实验脚本”，而是“现在重新产一版最像它的 dense vocabulary”，那我会建议按下面流程复刻：

1. 从 feature pool 中批量解出 `EGO_TRAJECTORY_LABEL`。
2. 过滤无效样本，只保留完整 `[80, 2]` 未来轨迹。
3. 统一到当前 decoder 使用的 ego 相对坐标系。
4. 导出 `trajectory_candidates.npy`。
5. 先用较大阈值跑一版 small vocabulary 做可视化 sanity check。
6. 再在全量候选上用更小阈值重跑 dense vocabulary。
7. 用当前 decoder 的 nearest-anchor 距离统计验证覆盖率，再决定是否替换 `dense_cluster_centers.npy`。

##### 目前仍不确定的点

这几个点仓里没有足够证据，只能保留为推断：

- 真正生成 `dense_cluster_centers.npy` 的最终脚本不在仓里。
- dense 版本到底是“一次全量重聚类”还是“小词表过滤后再重生成”，目前无法百分百确认。
- notebook 残留输出里还出现过 `(25, 2)` 轨迹实验痕迹，说明当时同一套思路可能也被复用于其他 decoder / vocabulary 试验，因此不能把 notebook 里的每个参数都直接当作 guide flow 正式产线参数。
- 真实使用的距离阈值、候选样本规模、平滑策略和是否做过额外 re-sample，目前都没有权威配置文件可追。

#### guide flow energy feature 的切分方式

`_split_guide_flow_energy_feature()` 会把输入拆成：

- `scene_feature`
- `polygon_coords`

默认读取规则是：

- 第 0 维：场景类型
- 第 1 到第 32 维：16 个 `(x, y)` guide flow 点

也就是说，当前 `[1, 34]` 的 feature 含义非常明确：

```text
[scene_id, x1, y1, x2, y2, ..., x16, y16]
```

#### `flow_match_forward()`

这一支负责基础轨迹生成。

训练时流程：

1. 用 GT 与所有 anchor 做距离计算，找最近 anchor。
2. 取 top-50 个近邻 anchor。
3. 对这些近邻 anchor 注入随机噪声，并按随机比例向 GT 插值。
4. 将这些 noisy anchor 投影后作为 query，与 `encoded_feature` 做 cross-attention。
5. 预测 velocity。
6. 输出 `FM_output_trajectory = input_anchor_feature + anchor_trajectory_velocity`。

这里的关键点是：

- 训练不是直接从固定 anchor 开始，而是从“加噪后的 anchor”开始。
- 这让模型学的是“速度场”而不是静态模板分类。

#### `energy_field_forward()`

这一支负责 guide flow 引导修正。

流程：

1. 读取 `scene_feature` 和 `query_region_feature`。
2. 给 polygon/path 坐标加位置编码。
3. 使用场景类型投影器和 query region 投影器构造 query feature。
4. 将 anchor 轨迹（或上一步迭代的输入轨迹）投影后拼接进去。
5. 再与 `encoded_feature` 做 cross-attention。
6. 输出 `EF_velocity`。

它的输入信息比 FM 更多，因为它还带了：

- scene type
- 16 个 guide flow path 点

所以它更像“条件引导修正器”。

#### `_process_forward()`

这是 FM 和 EF 的融合点。

融合策略：

- `iter < 3` 时：`output_trajectory = FM_output_trajectory + EF_velocity`
- `iter >= 3` 时：`output_trajectory = FM_output_trajectory`

也就是说：

- 前 3 轮允许 energy field 参与修正。
- 后几轮只保留 FM 输出。

这表明当前实现把 EF 当成“前期引导器”，而不是每一轮都持续生效的分支。

#### `forward()`

分两种模式：

1. 训练 / 验证模式：
   - 直接跑 `_process_forward()`。
2. 推理模式：
   - 内部固定迭代 6 轮。
   - 最后根据 `anchor_trajectory_velocity` 选择 `topk_trajectories` 返回。

#### `get_loss()`

loss 由 3 类组成：

1. FM 回归 loss
   - `FM_regression_loss`
2. EF 回归 loss
   - `EF_regression_loss`
3. 运动学约束 loss
   - `speed_limitation_loss`
   - `acceleration_limitation_loss`
   - `jerk_limitation_loss`
   - `curvature_limitation_loss`
   - `lateral_acceleration_limitation_loss`
   - `lateral_jerk_limitation_loss`

其中 guide flow 相关的关键超参是写死在这个文件顶部的：

- `ENABLE_TRAJECTORY_CLASSIFICATION = True`
- `TRAJECTORY_CLASSIFICATION_TARGET_EDGE = 8`
- `ENABLE_TRAJECTORY_CLASSIFICATION_REQUIRED_TIME = False`
- `TRAJECTORY_CLASSIFICATION_REQUIRED_TIME_SECONDS = 3.0`

这意味着当前版本：

- 会启用 polygon-based 好/坏轨迹划分。
- 目标边固定为第 8 条边。
- 但“必须在某个指定时间点仍位于 polygon 内”的约束当前默认不启用。

---

## 5. 数据流说明

## 5.1 普通 datapack 模式

由 `config/config.yaml` 驱动：

- `dataset.file_type = TENSOR_DATAPACK`
- 数据文件是传统 `train_file_list.bin` / `val_file_list.bin`

启用的主要输入：

- `EGO_FEATURE`
- `MOVING_OBSTACLE_FEATURE`
- `STATIC_OBSTACLE_FEATURE`
- `ROAD_GRAPH_SEGMENT_FEATURE`
- `TRAFFIC_LIGHT_FEATURE`

标签：

- `EGO_TRAJECTORY_LABEL`

这一模式更像项目早期或兼容模式。

## 5.2 feature pool 模式

由 `config/config_feature_pool.yaml` 驱动：

- `dataset.file_type = TENSOR_FEATURE_POOL`
- 这是当前 guide_flow_v1 更关键的训练入口

启用的 feature：

```text
guide_flow_energy_config_v1.pb.txt
ego_feature_config_v1.pb.txt
moving_obstacle_feature_config_v1.pb.txt
static_obstacle_feature_config_v1.pb.txt
road_graph_segment_feature_config_v1.pb.txt
traffic_light_feature_config_v2.pb.txt
ego_lane_query_feature_config_v1.pb.txt
reference_line_segment_feature_config_v1.pb.txt
route_aggregated_map_feature_config_v1.pb.txt
```

启用的 label：

```text
ego_lane_query_meta_label_config_v1.pb.txt
ego_trajectory_label_config_v1.pb.txt
obstacle_trajectory_label_config_v1.pb.txt
```

当前配置中的训练数据源是 4 个 `merged_feature_pool_train_v1_1of8_*` 文件，验证也暂时指向同一批文件。这更像是一个进行中的实验配置，而不是最终整理好的正式训练配置。

---

## 6. feature pool 数据加载机制

## 6.1 `planning_feature_data_index.proto`

这个 proto 描述了 feature pool 样本索引格式。

核心结构：

- `TensorDataIndex`
  - 指向某个 tensor 在 datapack 中的位置
  - 记录 `feature_name` 和 `feature_version`
  - 支持 `is_merge_features`
- `SampleLevelTensorDataIndex`
  - 由多个 `TensorDataIndex` 组成
  - 携带 `feature_meta`

所以 feature pool 不是“直接把 numpy 堆在一起”，而是“样本级索引 + datapack 真正数据内容”的两段式结构。

## 6.2 `planning_model_feature.proto`

这个 proto 定义了：

- `PlanningFeatureName` 枚举
- 各 feature 的 config message
- 模型名 `GUIDE_FLOW_DECODER`

guide flow v1 当前最关键的枚举项是：

- `GUIDE_FLOW_ENERGY_FEATURE = 68`

它对应的 `GuideFlowEnergyFeatureConfig` 中说明：

- `num_of_points = 16`
- 场景类型有：
  - `LANE_KEEP`
  - `LANE_CHANGE`
  - `JUNCTION_STRAIGHT`
  - `JUNCTION_LEFT`
  - `JUNCTION_RIGHT`

## 6.3 `tensor_maker_for_feature_pool_data.py`

这个文件是 feature pool 训练能跑起来的关键。

它做了几件事：

1. 根据 `enabled_features` / `enabled_labels` 加载 `*.pb.txt` 配置。
2. 构建 `(feature_name, feature_version) -> (tensor_key_str, shape, is_label)` 的查表。
3. 逐个读取 `SampleLevelTensorDataIndex` 中的 tensor 数据。
4. 校验 shape。
5. 组装成：
   - `input_data`
   - `label_data`
   - `metainfo_data`

如果缺少任何期望的 feature，会直接报错或返回空样本，这能避免训练静默吃坏数据。

它依赖 `planning_dataset_config.py` 中定义的：

```text
FEATURE_CONFIG_DIR = config/module/planning/learning/feature/feature_config_version
```

也就是说，`config_feature_pool.yaml` 里写的 `enabled_features` / `enabled_labels` 实际上都是相对这个目录解析的。

## 6.4 `planning_dataset.py`

这个文件负责把 config 转成真正的 dataset：

- `TENSOR_DATAPACK` -> `Dataset`
- `TENSOR_FEATURE_POOL` -> `FeaturePoolDataset`

两者最终都会输出：

```text
(inputs, targets, meta_infos)
```

给 trainer / evaluator 使用。

---

## 7. 训练、验证、部署链路

## 7.1 `trainer.py`

`Trainer` 继承自 `PlanningBaseTrainer`，这里主要做三件事：

1. `_get_model_impl()` 返回任务模型。
2. `_get_evaluator_impl()` 返回 `Evaluator`。
3. `_get_visualizer_impl()` 返回 `Visualizer`。

额外逻辑：

- `_load_pretrain_model()` 支持通过 `--ckpt` 额外加载 checkpoint。
- 加载前会比对 keys，确保与当前模型结构匹配。

## 7.2 `planning_base_trainer.py`

这是整个任务真正的训练底座，guide flow 目录本身没有重复造轮子。

它负责：

- 合并旧版 `unified_dataset_config` 和新版 `dataset`
- 构建训练 / 验证 dataset
- 自动缩放学习率
- 冻结指定模块
- 包装 deploy 子模型
- 初始化验证 dataloader

guide flow 当前配置里很重要的一点是：

```yaml
freeze_modules: ["_encoder"]
```

也就是默认冻结共享编码器，只训练 decoder 相关部分。

## 7.3 `config/config_feature_pool.yaml`

这是最值得优先阅读的一份配置。

重要信息包括：

- `max_epoch = 12`
- `batch_size_per_gpu = 128`
- `file_type = TENSOR_FEATURE_POOL`
- `freeze_modules = ["_encoder"]`
- 使用预训练模型
- `pretrain_exclude` 只排除 EF 分支若干模块

这说明当前实验设计大致是：

1. 编码器复用已有共享表示。
2. Flow Match 分支较大程度继承原任务参数。
3. Energy Field 的 scene/query/regression 子模块作为新加部分重点训练。

## 7.4 `scripts/train_feature_pool.sh`

这个脚本负责提交 HOPE 训练任务。

核心特点：

- 使用 `config_feature_pool.yaml`
- 默认 4 worker、8 GPU
- 输出目录在用户个人路径下
- Bazel 入口是 `//common/python/training/planning_training_pipeline/tasks/guide_flow_decoder:trainer_package`

## 7.5 `config/config.yaml`

这份配置更偏旧式训练：

- `file_type = TENSOR_DATAPACK`
- 只使用编码器必需的基础特征
- deploy 配置中定义了 `_decoder` 输入输出节点

如果需要看部署输入输出定义，优先读这份配置的 `deploy` 段。

---

## 8. 评估逻辑

`evaluator.py` 里的逻辑不是简单“前向一次然后算 ADE/FDE”，而是做了任务定制。

## 8.1 多轮迭代验证

验证时固定做 6 轮：

1. 第 0 轮：`context.model(inputs, targets)`
2. 后续轮次：把上一轮 `outputs["output_trajectory"]` 作为 `input_anchor_trajectory` 再喂回去

这说明：

- 训练时主要优化单步。
- 验证时观察多步 refinement 之后的效果。

## 8.2 指标计算方式

`GuideFlowTrajectoryDisplacementErrorCalculator` 基于 `TrajectoryDisplacementErrorCalculator` 改写。

当前计算：

- `MIN_ADE`
- `MIN_FDE`
- `GT_ADE`
- `GT_FDE`

一个非常关键的实现细节是：

- `MIN_ADE/MIN_FDE` 在这里会排除 GT 对应的 anchor，再从其余候选里取最小值。

## 8.3 evaluator 如何构造候选轨迹

`_evaluate_impl()` 中构造的候选并不只是模型一条输出，而是：

1. `nearest_anchor_indices` 对应的那条预测轨迹
2. 按 velocity norm 取最小的若干 anchor 预测
3. 再拼一些随机 anchor 预测

然后把这些候选拼成一个 trajectory set 去算 ADE/FDE。

这说明 evaluator 更像是在评估“候选集质量”，而不只是评估单条 top-1 轨迹。

---

## 9. 可视化逻辑

`visualizer.py` 是这个项目中信息量很高的文件，因为它把中间状态都画出来了。

它主要做几件事：

1. 画基础场景：
   - road graph
   - traffic light
   - static obstacle
   - movable obstacle
   - ego history
   - ego gt
2. 画预测：
   - 当前预测轨迹
   - anchor 轨迹
   - 所有 anchor
   - top-k 轨迹
3. 画关系箭头：
   - `all_anchor -> all_anchor_pred`
   - `top_k_anchor -> top_k_pred`
4. 画 guide flow path：
   - 从 `GUIDE_FLOW_ENERGY_FEATURE` 中把 16 个点解析出来，使用青色虚线绘制

### `vector_image_config.json`

这个文件控制可视化层开关。当前包含：

- `road_graph`
- `traffic_light`
- `reference_line`
- `static_obstacle`
- `movable_obstacle`
- `ego_history`
- `ego_gt`
- `pred_trajectory`
- `pred_trajectory_anchor`
- `all_anchors`
- `top_k_trajectory_prediction`
- `top_k_anchor_trajectory`
- `top_k_anchor_to_pred_relation`
- `guide_flow_path`

所以如果你想快速确认某个中间量是否被产出，最直接的方式通常不是加 print，而是看 visualizer 有没有把它画出来。

---

## 10. 配置文件差异与用途

## 10.1 `config.yaml`

适合看：

- 传统 datapack 训练方式
- deploy 节点定义
- 基础 evaluator / visualizer 配置

## 10.2 `config_feature_pool.yaml`

适合看：

- 当前 guide flow v1 的主实验配置
- feature pool 数据输入
- guide flow energy feature 的接入方式
- 预训练和冻结策略

## 10.3 `config_local.yaml`

适合看：

- 小 batch 本地调试
- 最小化配置示例

---

## 11. 当前项目中的关键输入输出字典

## 11.1 训练输入 `inputs`

训练态常见 key：

- `EGO_FEATURE`
- `MOVING_OBSTACLE_FEATURE`
- `STATIC_OBSTACLE_FEATURE`
- `ROAD_GRAPH_SEGMENT_FEATURE`
- `TRAFFIC_LIGHT_FEATURE`
- `GUIDE_FLOW_ENERGY_FEATURE`
- 可能还有：
  - `EGO_LANE_QUERY_FEATURE`
  - `REFERENCE_LINE_SEGMENT_FEATURE`
  - `ROUTE_AGGREGATED_MAP_FEATURE`

其中真正被 `WrappedSharedEncoder` 直接消费的是前 5 个；`GUIDE_FLOW_ENERGY_FEATURE` 直接给 decoder 的 EF 分支使用。

## 11.2 训练标签 `targets`

核心标签：

- `EGO_TRAJECTORY_LABEL`

feature pool 配置里还会带：

- `EGO_LANE_QUERY_META_LABEL`
- `OBSTACLE_TRAJECTORY_LABEL`

但当前 guide flow 主 loss 直接使用的仍然是 ego 未来轨迹。

## 11.3 decoder 训练输出

`GuideFlowDecoderV1.TrainingOutput` 包含：

- `anchor_trajectory`
- `anchor_trajectory_velocity`
- `mocked_anchor_trajectory`
- `nearest_anchor_indices`
- `topk_near_anchor_indices`
- `FM_output_trajectory`
- `EF_anchor_with_noise`
- `EF_velocity`
- `output_trajectory`

这说明训练态输出不是“最终轨迹”这么简单，而是完整保留了 FM/EF 两条支路的中间结果。

## 11.4 deploy 输出

根据配置，部署 `_decoder` 时导出的输出是：

- `topk_trajectories`

这与训练态的丰富中间结果不同，部署接口更精简。

---

## 12. 当前实现最值得注意的设计点

### 1. FM 和 EF 是并行解码后再融合，不是串行子模块

它们都读取同一个 `encoded_feature`，只是 query 构造方式不同。

### 2. EF 的 supervision 不是直接对 GT 回归，而是对“最近好轨迹”的引导速度回归

这使 EF 更像一个轨迹场修正器，而不是普通 trajectory head。

### 3. guide flow 特征目前非常轻量

主输入只有：

- 1 个 scene id
- 16 个二维点

说明当前版本更依赖 shared encoder 场景理解，guide flow feature 只是一个额外条件。

### 4. 验证是多轮迭代，训练主要是单轮

这意味着最终效果和单步训练目标之间存在一定“rollout gap”，分析问题时要把训练和验证区分开看。

### 5. anchor 词表是整个模型的先验基础

当前默认用 `dense_cluster_centers.npy`，2398 条 anchor 决定了轨迹空间的覆盖能力上限。

---

## 13. 文件级详细说明

下面按“你实际排查问题时最常打开的顺序”再梳理一遍。

### `tasks/guide_flow_decoder/model.py`

- 项目模型入口。
- 封装共享编码器和 decoder。
- 负责适配训练/验证/部署三种调用方式。

### `tasks/guide_flow_decoder/trainer.py`

- 命令行训练入口。
- 把任务挂接到通用训练框架中。
- 支持 `--ckpt` 额外加载权重。

### `tasks/guide_flow_decoder/evaluator.py`

- 验证阶段执行 6 轮 refinement。
- 构造轨迹候选集并计算 ADE/FDE 指标。

### `tasks/guide_flow_decoder/visualizer.py`

- 负责“把模型到底做了什么”画出来。
- 能直接看见 guide flow path、anchor 与预测的关系。

### `tasks/guide_flow_decoder/config/config_feature_pool.yaml`

- 当前最接近 guide_flow_v1 主实验的配置文件。
- 也是理解数据入口、预训练和冻结策略的第一文件。

### `tasks/guide_flow_decoder/scripts/train_feature_pool.sh`

- 生产环境 / 集群训练提交脚本。
- 可以最快看出真实跑实验时用的 config 和资源规格。

### `decoder/planning_guide_flow_decoder_v1.py`

- 真正的算法主体。
- 如果你只打算精读一个文件，优先读它。

### `dataset/tensor_maker_for_feature_pool_data.py`

- 如果你发现输入 key 缺失、shape 不匹配、feature 没读到，优先看它。

### `proto/planning_model_feature.proto`

- 如果你发现 feature 名、shape、枚举值或场景类型不清楚，优先看它。

---

## 14. 建议的阅读顺序

如果你是第一次接手这个项目，建议按下面顺序读：

1. `tasks/guide_flow_decoder/config/config_feature_pool.yaml`
2. `tasks/guide_flow_decoder/model.py`
3. `decoder/planning_guide_flow_decoder_v1.py`
4. `tasks/guide_flow_decoder/evaluator.py`
5. `tasks/guide_flow_decoder/visualizer.py`
6. `dataset/tensor_maker_for_feature_pool_data.py`
7. `proto/planning_model_feature.proto`

这样读的好处是：

- 先知道任务怎么配。
- 再知道模型怎么接。
- 然后精读核心算法。
- 最后再补数据格式和可视化细节。

---

## 15. 总结

`guide_flow_v1` 可以概括为：

- 一个复用共享编码器的规划 decoder 任务。
- 用 anchor vocabulary 建模轨迹空间。
- 用 Flow Match 分支学习基础轨迹速度场。
- 用 Energy Field 分支学习 guide flow 条件下的引导速度场。
- 用 polygon mask 和最近好轨迹匹配机制来定义 EF 的监督信号。
- 用多轮迭代验证来观察 refinement 后的最终轨迹效果。

如果后续要继续维护这个项目，最关键的抓手通常有 4 个：

1. `dense_cluster_centers.npy` 的覆盖能力，以及它背后的离线生成策略。
2. `GUIDE_FLOW_ENERGY_FEATURE` 的表达质量。
3. `build_anchor_polygon_mask()` 定义的好/坏轨迹划分逻辑。
4. `config_feature_pool.yaml` 中的真实训练数据和预训练加载策略。

这 4 个点基本决定了 guide flow v1 的上限和调参方向。
