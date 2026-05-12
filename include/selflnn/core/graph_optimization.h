/**
 * @file graph_optimization.h
 * @brief 计算图优化和算子融合系统
 * 
 * 计算图优化系统，支持算子融合、常量折叠、死代码消除、内存优化等。
 * 提供图分析、优化策略和变换功能，提升神经网络执行效率。
 */

#ifndef SELFLNN_CORE_GRAPH_OPTIMIZATION_H
#define SELFLNN_CORE_GRAPH_OPTIMIZATION_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 算子类型
 */
typedef enum {
    OP_TYPE_INPUT = 0,           /**< 输入节点 */
    OP_TYPE_CONSTANT,            /**< 常量节点 */
    OP_TYPE_WEIGHT,              /**< 权重节点 */
    OP_TYPE_ADD,                 /**< 加法 */
    OP_TYPE_SUB,                 /**< 减法 */
    OP_TYPE_MUL,                 /**< 乘法 */
    OP_TYPE_DIV,                 /**< 除法 */
    OP_TYPE_MATMUL,              /**< 矩阵乘法 */
    OP_TYPE_CONV,                /**< 卷积 */
    OP_TYPE_POOL,                /**< 池化 */
    OP_TYPE_RELU,                /**< ReLU激活 */
    OP_TYPE_SIGMOID,             /**< Sigmoid激活 */
    OP_TYPE_TANH,                /**< Tanh激活 */
    OP_TYPE_SOFTMAX,             /**< Softmax */
    OP_TYPE_BATCH_NORM,          /**< 批量归一化 */
    OP_TYPE_DROPOUT,             /**< Dropout */
    OP_TYPE_CONCAT,              /**< 拼接 */
    OP_TYPE_SPLIT,               /**< 分割 */
    OP_TYPE_RESHAPE,             /**< 重塑 */
    OP_TYPE_TRANSPOSE,           /**< 转置 */
    OP_TYPE_REDUCE_SUM,          /**< 求和归约 */
    OP_TYPE_REDUCE_MEAN,         /**< 平均归约 */
    OP_TYPE_REDUCE_MAX,          /**< 最大归约 */
    OP_TYPE_OUTPUT,              /**< 输出节点 */
    OP_TYPE_CUSTOM               /**< 自定义算子 */
} OperatorType;

/**
 * @brief 张量数据类型
 */
typedef enum {
    DATA_TYPE_FLOAT16 = 0,       /**< 半精度浮点（16位） */
    DATA_TYPE_FLOAT32,           /**< 单精度浮点（32位） */
    DATA_TYPE_FLOAT64,           /**< 双精度浮点（64位） */
    DATA_TYPE_INT8,              /**< 8位整数 */
    DATA_TYPE_INT16,             /**< 16位整数 */
    DATA_TYPE_INT32,             /**< 32位整数 */
    DATA_TYPE_INT64,             /**< 64位整数 */
    DATA_TYPE_UINT8,             /**< 8位无符号整数 */
    DATA_TYPE_UINT16,            /**< 16位无符号整数 */
    DATA_TYPE_UINT32,            /**< 32位无符号整数 */
    DATA_TYPE_UINT64,            /**< 64位无符号整数 */
    DATA_TYPE_BOOL               /**< 布尔类型 */
} DataType;

/**
 * @brief 张量形状
 */
typedef struct {
    int* dims;                   /**< 维度数组 */
    int rank;                    /**< 维度数量 */
    int64_t element_count;       /**< 元素总数 */
} TensorShape;

/**
 * @brief 张量描述
 */
typedef struct {
    DataType dtype;              /**< 数据类型 */
    TensorShape shape;           /**< 张量形状 */
    void* data;                  /**< 数据指针 */
    size_t data_size;            /**< 数据大小（字节） */
    int is_const;                /**< 是否为常量 */
    int requires_grad;           /**< 是否需要梯度 */
} TensorDesc;

/**
 * @brief 算子属性
 */
typedef struct {
    char* name;                  /**< 属性名称 */
    void* value;                 /**< 属性值 */
    size_t value_size;           /**< 属性值大小 */
    DataType value_type;         /**< 属性值类型 */
} OperatorAttribute;

/**
 * @brief 计算图节点
 */
typedef struct GraphNode GraphNode;

/**
 * @brief 计算图节点定义
 */
struct GraphNode {
    char* name;                  /**< 节点名称 */
    OperatorType op_type;        /**< 算子类型 */
    
    // 输入输出
    GraphNode** inputs;          /**< 输入节点数组 */
    int input_count;             /**< 输入节点数量 */
    TensorDesc* output_tensors;  /**< 输出张量数组 */
    int output_count;            /**< 输出张量数量 */
    
    // 属性
    OperatorAttribute* attrs;    /**< 属性数组 */
    int attr_count;              /**< 属性数量 */
    
    // 内部状态
    void* op_impl;               /**< 算子实现指针 */
    void* user_data;             /**< 用户数据 */
    
    // 图结构
    GraphNode** consumers;       /**< 消费此节点的节点 */
    int consumer_count;          /**< 消费者数量 */
    
    // 优化标记
    int optimized;               /**< 是否已优化 */
    int fused;                   /**< 是否已融合 */
    int eliminated;              /**< 是否已消除 */
    int folded;                  /**< 是否已折叠（常量折叠标记） */
    void* fusion_info;           /**< 融合信息指针，存储算子融合的详细信息 */
};

/**
 * @brief 计算图
 */
typedef struct {
    char* name;                  /**< 图名称 */
    GraphNode** nodes;           /**< 节点数组 */
    int node_count;              /**< 节点数量 */
    GraphNode** input_nodes;     /**< 输入节点数组 */
    int input_count;             /**< 输入节点数量 */
    GraphNode** output_nodes;    /**< 输出节点数组 */
    int output_count;            /**< 输出节点数量 */
    
    // 优化状态
    int optimized;               /**< 是否已优化 */
    int fused_count;             /**< 融合节点数量 */
    int eliminated_count;        /**< 消除节点数量 */
    
    // 性能统计
    float estimated_flops;       /**< 估计的FLOPs */
    float estimated_memory;      /**< 估计的内存使用 */
    float estimated_latency;     /**< 估计的延迟 */
} ComputationGraph;

/**
 * @brief 优化策略
 */
typedef struct {
    int enable_fusion;           /**< 启用算子融合 */
    int enable_const_folding;    /**< 启用常量折叠 */
    int enable_dead_code_elim;   /**< 启用死代码消除 */
    int enable_memory_opt;       /**< 启用内存优化 */
    int enable_kernel_fusion;    /**< 启用内核融合 */
    int enable_auto_mixed_precision; /**< 启用自动混合精度 */
    
    // 融合规则
    int fuse_conv_bn;            /**< 融合卷积和批量归一化 */
    int fuse_conv_relu;          /**< 融合卷积和ReLU */
    int fuse_matmul_add;         /**< 融合矩阵乘法和加法 */
    int fuse_batch_norm;         /**< 融合批量归一化 */
    int fuse_elementwise_ops;    /**< 融合逐元素操作 */
    
    // 优化级别
    int optimization_level;      /**< 优化级别（0-3） */
    
    // 特定优化
    int use_inplace_ops;         /**< 使用原地操作 */
    int reuse_memory;            /**< 重用内存 */
    int minimize_memory_footprint; /**< 最小化内存占用 */
} GraphOptimizationStrategy;

/**
 * @brief 融合模式
 */
typedef enum {
    FUSION_NONE = 0,             /**< 无融合 */
    FUSION_VERTICAL,             /**< 垂直融合（算子链） */
    FUSION_HORIZONTAL,           /**< 水平融合（并行算子） */
    FUSION_ELEMENTWISE,          /**< 逐元素算子融合 */
    FUSION_KERNEL,               /**< 内核融合 */
    FUSION_MEMORY                /**< 内存融合 */
} FusionMode;

/**
 * @brief 融合规则
 */
typedef struct {
    OperatorType* pattern;       /**< 算子模式数组 */
    int pattern_length;          /**< 模式长度 */
    OperatorType fused_op;       /**< 融合后的算子类型 */
    FusionMode fusion_mode;      /**< 融合模式 */
    float benefit_score;         /**< 效益评分 */
} FusionRule;

/**
 * @brief 优化结果
 */
typedef struct {
    ComputationGraph* graph;     /**< 优化后的图 */
    float original_flops;        /**< 原始FLOPs */
    float optimized_flops;       /**< 优化后FLOPs */
    float original_memory;       /**< 原始内存使用 */
    float optimized_memory;      /**< 优化后内存使用 */
    int fused_nodes;             /**< 融合节点数 */
    int eliminated_nodes;        /**< 消除节点数 */
    float optimization_time;     /**< 优化时间（毫秒） */
    float speedup_ratio;         /**< 加速比 */
} OptimizationResult;

/**
 * @brief 创建计算图
 * 
 * @param name 图名称
 * @return ComputationGraph* 计算图指针
 */
ComputationGraph* graph_create(const char* name);

/**
 * @brief 销毁计算图
 * 
 * @param graph 计算图指针
 */
void graph_destroy(ComputationGraph* graph);

/**
 * @brief 添加节点到计算图
 * 
 * @param graph 计算图
 * @param node 节点
 * @return int 成功返回0，失败返回-1
 */
int graph_add_node(ComputationGraph* graph, GraphNode* node);

/**
 * @brief 创建图节点
 * 
 * @param name 节点名称
 * @param op_type 算子类型
 * @return GraphNode* 图节点指针
 */
GraphNode* graph_node_create(const char* name, OperatorType op_type);

/**
 * @brief 销毁图节点
 * 
 * @param node 图节点
 */
void graph_node_destroy(GraphNode* node);

/**
 * @brief 连接节点
 * 
 * @param src 源节点
 * @param dst 目标节点
 * @param src_output_idx 源节点输出索引
 * @param dst_input_idx 目标节点输入索引
 * @return int 成功返回0，失败返回-1
 */
int graph_connect_nodes(GraphNode* src, GraphNode* dst, 
                       int src_output_idx, int dst_input_idx);

/**
 * @brief 应用优化策略
 * 
 * @param graph 计算图
 * @param strategy 优化策略
 * @return OptimizationResult* 优化结果，失败返回NULL
 */
OptimizationResult* graph_optimize(ComputationGraph* graph, 
                                  const GraphOptimizationStrategy* strategy);

/**
 * @brief 执行算子融合
 * 
 * @param graph 计算图
 * @param rules 融合规则数组
 * @param rule_count 规则数量
 * @return int 融合的节点数
 */
int graph_fuse_operators(ComputationGraph* graph, 
                        const FusionRule* rules, int rule_count);

/**
 * @brief 执行常量折叠
 * 
 * @param graph 计算图
 * @return int 折叠的节点数
 */
int graph_constant_folding(ComputationGraph* graph);

/**
 * @brief 执行死代码消除
 * 
 * @param graph 计算图
 * @return int 消除的节点数
 */
int graph_dead_code_elimination(ComputationGraph* graph);

/**
 * @brief 执行内存优化
 * 
 * @param graph 计算图
 * @return int 优化的内存块数
 */
int graph_memory_optimization(ComputationGraph* graph);

/**
 * @brief 获取默认优化策略
 * 
 * @param strategy 策略输出
 */
void graph_default_optimization_strategy(GraphOptimizationStrategy* strategy);

/**
 * @brief 获取默认融合规则
 * 
 * @param rules 规则数组输出
 * @param max_rules 最大规则数
 * @return int 实际规则数
 */
int graph_default_fusion_rules(FusionRule* rules, int max_rules);

/**
 * @brief 验证计算图
 * 
 * @param graph 计算图
 * @return int 有效返回0，发现错误返回错误码
 */
int graph_validate(const ComputationGraph* graph);

/**
 * @brief 可视化计算图（生成DOT格式）
 * 
 * @param graph 计算图
 * @param filename 输出文件名
 * @return int 成功返回0，失败返回-1
 */
int graph_visualize(const ComputationGraph* graph, const char* filename);

/**
 * @brief 评估计算图性能
 * 
 * @param graph 计算图
 * @param flops FLOPs输出
 * @param memory 内存输出
 * @param latency 延迟输出
 * @return int 成功返回0，失败返回-1
 */
int graph_evaluate_performance(const ComputationGraph* graph,
                              float* flops, float* memory, float* latency);

/**
 * @brief 执行计算图
 * 
 * 按照拓扑顺序逐个执行算子节点，每个节点使用实际运算内核处理数据。
 * 输入节点（OP_TYPE_INPUT）的数据需要在调用前设置到 output_tensors[0].data。
 * 执行后各节点输出数据存储在 output_tensors[j].data 中。
 * 
 * @param graph 计算图（已添加节点并连接好）
 * @return int 成功返回0，失败返回-1
 */
int graph_execute(ComputationGraph* graph);

#ifdef __cplusplus
}
#endif

#endif /* SELFLNN_CORE_GRAPH_OPTIMIZATION_H */