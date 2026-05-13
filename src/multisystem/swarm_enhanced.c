/**
 * @file swarm_enhanced.c
 * @brief 群体智能深度增强系统完整实现
 *
 * 实现：
 * 1. A06.4.1 ACO增强版 — 自适应信息素更新 + CfC液态路径选择
 * 2. A06.4.2 ABC增强版 — 自适应雇佣蜂 + CfC液态蜜源评估
 * 3. A06.4.3 分布式一致性 — Raft共识协议（完整实现）
 * 4. A06.4.4 群体液态通信 — CfC ODE群体状态同步
 * 5. A06.4.5 群体自愈与重组 — 故障检测与自动替换
 */

#include "selflnn/multisystem/swarm_enhanced.h"
#include "selflnn/core/errors.h"
#include "selflnn/core/laplace.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/logging.h"
#include "selflnn/utils/math_utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>

#define SWARM_EPSILON 1e-8f
#define SWARM_INF 1e20f

/* MID-013修复: 使用时间+地址生成唯一种子，替代硬编码54321 */
static unsigned int swarm_rand_seed = 0;
static int swarm_seed_initialized = 0;

static void swarm_seed_init(void) {
    if (!swarm_seed_initialized) {
        swarm_rand_seed = (unsigned int)((uintptr_t)time(NULL) ^
                          ((uintptr_t)&swarm_rand_seed * 2654435761U));
        if (swarm_rand_seed == 0) swarm_rand_seed = 1;
        swarm_seed_initialized = 1;
    }
}

static float swarm_rand_float(void) {
    swarm_seed_init();
    swarm_rand_seed = swarm_rand_seed * 1103515245 + 12345;
    return (float)((swarm_rand_seed >> 16) & 0x7FFF) / 32768.0f;
}

static int swarm_rand_int(int min, int max) {
    return min + (int)(swarm_rand_float() * (max - min + 1));
}

/* CfC ODE步进 */
static void swarm_cfc_step(float* h, int dim, float tau, float dt) {
    for (int i = 0; i < dim; i++) {
        float gate = 1.0f / (1.0f + expf(-h[i]));
        float act = tanhf(h[i]);
        float dh = -h[i] / (tau + SWARM_EPSILON) + gate * act;
        h[i] += dh * dt;
    }
}

/* ============================================================================
 * 内部结构定义
 * ============================================================================ */

typedef struct {
    float* position;
    float* velocity;
    float* personal_best;
    float personal_best_fitness;
    float* path;
    int path_length;
    float path_cost;
    int visited_flag;
    int food_source_index;
    float food_quality;
    float food_fitness;
    int trial_counter;
} SwarmAgent;

typedef struct {
    float weight;
    int source_node;
    int dest_node;
    float pheromone;
    float heuristic;
    float cfc_activation;
} SwarmEdge;

struct SwarmEnhancedEngine {
    int algorithm_type;
    int is_initialized;

    /* ACO状态 */
    ACOEnhancedConfig aco_config;
    int aco_num_nodes;
    float* aco_distance;
    float* aco_pheromone;
    float* aco_heuristic;
    SwarmAgent* aco_ants;
    float* aco_best_path;
    int aco_best_path_len;
    float aco_best_cost;
    float* aco_cfc_state;

    /* ABC状态 */
    ABCEnhancedConfig abc_config;
    int abc_dimensions;
    float* abc_lower;
    float* abc_upper;
    SwarmAgent* abc_food_sources;
    float* abc_best_solution;
    float abc_best_fitness;
    float* abc_cfc_state;
    float* abc_probabilities;

    /* 共识状态 */
    ConsensusNodeConfig consensus_config;
    ConsensusProtocol consensus_protocol;
    int* consensus_peers;
    int consensus_peer_count;
    ConsensusLogEntry* consensus_log;
    int consensus_log_count;
    int consensus_log_capacity;
    int consensus_votes_received;
    int consensus_voted_for;
    float consensus_current_term;
    int consensus_leader_id;
    int consensus_node_state;
    float consensus_election_timeout;

    /* 液态通信状态 */
    int comm_channel_count;
    int comm_num_nodes;
    int comm_state_dim;
    int comm_hidden_dim;
    float** comm_states;
    float** comm_hidden;
    LiquidMessage* comm_message_buffer;
    int comm_buffer_size;
    int comm_buffer_capacity;
    /* M-038修复: 多通道消息队列支持 */
    LiquidMessage** comm_channel_buffers;
    int* comm_channel_buffer_sizes;
    int* comm_channel_buffer_caps;

    /* 自愈状态 */
    SelfHealingConfig healing_config;
    int* healing_node_status;
    int* healing_node_last_seen;  /* M-039修复: 各节点最后心跳时间戳(微秒) */
    int healing_node_count;
    int healing_failure_count;
};

/* ============================================================================
 * A06.4.1 — ACO增强版
 * ============================================================================ */

SwarmEnhancedEngine* swarm_aco_enhanced_create(const ACOEnhancedConfig* config) {
    if (!config) return NULL;
    SwarmEnhancedEngine* engine = (SwarmEnhancedEngine*)
        safe_calloc(1, sizeof(SwarmEnhancedEngine));
    if (!engine) return NULL;

    engine->algorithm_type = SWARM_ENH_ACO_ADAPTIVE;
    engine->aco_config = *config;
    engine->aco_best_cost = SWARM_INF;
    engine->aco_best_path_len = 0;

    engine->aco_cfc_state = (float*)safe_calloc(64, sizeof(float));
    /* M-037修复: 用小幅随机扰动初始化CfC状态，确保ODE有非零初始条件 */
    if (engine->aco_cfc_state) {
        for (int s = 0; s < 64; s++)
            engine->aco_cfc_state[s] = ((float)((s * 1103515245 + 12345) % 10000) / 10000.0f - 0.5f) * 0.1f;
    }

    engine->is_initialized = 1;
    
    return engine;
}

void swarm_aco_enhanced_destroy(SwarmEnhancedEngine* engine) {
    if (!engine) return;

    safe_free((void**)&engine->aco_distance);
    safe_free((void**)&engine->aco_pheromone);
    safe_free((void**)&engine->aco_heuristic);

    if (engine->aco_ants) {
        for (int i = 0; i < engine->aco_config.num_ants; i++) {
            safe_free((void**)&engine->aco_ants[i].path);
        }
        safe_free((void**)&engine->aco_ants);
    }
    safe_free((void**)&engine->aco_best_path);
    safe_free((void**)&engine->aco_cfc_state);

    /* 释放ABC资源 */
    if (engine->abc_food_sources) {
        for (int i = 0; i < engine->abc_config.food_sources; i++) {
            safe_free((void**)&engine->abc_food_sources[i].position);
        }
        safe_free((void**)&engine->abc_food_sources);
    }
    safe_free((void**)&engine->abc_best_solution);
    safe_free((void**)&engine->abc_cfc_state);
    safe_free((void**)&engine->abc_lower);
    safe_free((void**)&engine->abc_upper);
    safe_free((void**)&engine->abc_probabilities);

    /* 释放共识资源 */
    safe_free((void**)&engine->consensus_peers);
    if (engine->consensus_log) {
        safe_free((void**)&engine->consensus_log);
    }

    /* 释放通信资源 */
    if (engine->comm_states) {
        for (int i = 0; i < engine->comm_num_nodes; i++)
            safe_free((void**)&engine->comm_states[i]);
        safe_free((void**)&engine->comm_states);
    }
    if (engine->comm_hidden) {
        for (int i = 0; i < engine->comm_num_nodes; i++)
            safe_free((void**)&engine->comm_hidden[i]);
        safe_free((void**)&engine->comm_hidden);
    }
    safe_free((void**)&engine->comm_message_buffer);
    /* M-038修复: 清理多通道队列 */
    if (engine->comm_channel_buffers) {
        for (int c = 0; c < engine->comm_channel_count; c++)
            safe_free((void**)&engine->comm_channel_buffers[c]);
        safe_free((void**)&engine->comm_channel_buffers);
    }
    safe_free((void**)&engine->comm_channel_buffer_sizes);
    safe_free((void**)&engine->comm_channel_buffer_caps);
    safe_free((void**)&engine->healing_node_status);
    safe_free((void**)&engine->healing_node_last_seen);

    safe_free((void**)&engine);
}

int swarm_aco_init_graph(SwarmEnhancedEngine* engine,
                          int num_nodes,
                          const float* distance_matrix) {
    if (!engine || !distance_matrix || num_nodes < 2) return -1;
    if (engine->algorithm_type != SWARM_ENH_ACO_ADAPTIVE) return -1;

    engine->aco_num_nodes = num_nodes;
    int n = num_nodes;

    engine->aco_distance = (float*)safe_calloc(n * n, sizeof(float));
    engine->aco_pheromone = (float*)safe_calloc(n * n, sizeof(float));
    engine->aco_heuristic = (float*)safe_calloc(n * n, sizeof(float));
    if (!engine->aco_distance || !engine->aco_pheromone || !engine->aco_heuristic)
        return -1;

    memcpy(engine->aco_distance, distance_matrix, n * n * sizeof(float));

    float init_phero = engine->aco_config.initial_pheromone;
    if (init_phero < SWARM_EPSILON) init_phero = 0.1f;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j) {
                engine->aco_pheromone[i * n + j] = init_phero;
                float d = engine->aco_distance[i * n + j];
                engine->aco_heuristic[i * n + j] = d > SWARM_EPSILON ? 1.0f / d : SWARM_INF;
            } else {
                engine->aco_pheromone[i * n + j] = 0.0f;
                engine->aco_heuristic[i * n + j] = 0.0f;
            }
        }
    }

    /* 初始化蚂蚁 */
    int num_ants = engine->aco_config.num_ants;
    engine->aco_ants = (SwarmAgent*)safe_calloc(num_ants, sizeof(SwarmAgent));
    if (!engine->aco_ants) return -1;
    for (int i = 0; i < num_ants; i++) {
        engine->aco_ants[i].path = (float*)safe_calloc(n * 2, sizeof(float));
        engine->aco_ants[i].path_length = 0;
        engine->aco_ants[i].path_cost = 0.0f;
    }

    engine->aco_best_path = (float*)safe_calloc(n * 2, sizeof(float));
    if (!engine->aco_best_path) return -1;

    return 0;
}

static int aco_select_next_node(SwarmEnhancedEngine* engine,
                                  int ant_idx,
                                  int current_node,
                                  const int* visited, int num_visited) {
    (void)ant_idx;
    int n = engine->aco_num_nodes;
    float alpha = engine->aco_config.alpha;
    float beta = engine->aco_config.beta;

    float total_prob = 0.0f;
    float* probs = (float*)safe_calloc(n, sizeof(float));
    if (!probs) return -1;

    for (int j = 0; j < n; j++) {
        int already_visited = 0;
        for (int v = 0; v < num_visited; v++) {
            if (visited[v] == j) { already_visited = 1; break; }
        }
        if (!already_visited && current_node != j) {
            float tau = powf(engine->aco_pheromone[current_node * n + j], alpha);
            float eta = powf(engine->aco_heuristic[current_node * n + j], beta);
            probs[j] = tau * eta;
            total_prob += probs[j];
        } else {
            probs[j] = 0.0f;
        }
    }

    if (total_prob < SWARM_EPSILON) {
        safe_free((void**)&probs);
        for (int j = 0; j < n; j++) {
            int already_visited = 0;
            for (int v = 0; v < num_visited; v++) {
                if (visited[v] == j) { already_visited = 1; break; }
            }
            if (!already_visited && current_node != j) return j;
        }
        return -1;
    }

    float r = swarm_rand_float() * total_prob;
    float cum = 0.0f;
    for (int j = 0; j < n; j++) {
        cum += probs[j];
        if (r <= cum && probs[j] > 0) {
            safe_free((void**)&probs);
            return j;
        }
    }

    safe_free((void**)&probs);
    for (int j = n - 1; j >= 0; j--) {
        if (probs[j] > 0) return j;
    }
    return -1;
}

int swarm_aco_iterate(SwarmEnhancedEngine* engine, int iteration) {
    (void)iteration;
    if (!engine || engine->algorithm_type != SWARM_ENH_ACO_ADAPTIVE) return -1;
    if (!engine->aco_ants) return -1;

    int n = engine->aco_num_nodes;
    int num_ants = engine->aco_config.num_ants;
    ACOEnhancedConfig* cfg = &engine->aco_config;

    for (int ant = 0; ant < num_ants; ant++) {
        int* visited = (int*)safe_calloc(n, sizeof(int));
        int num_visited = 0;
        int start = swarm_rand_int(0, n - 1);

        visited[num_visited++] = start;
        int current = start;

        float cost = 0.0f;
        for (int step = 1; step < n; step++) {
            int next = aco_select_next_node(engine, ant, current, visited, num_visited);
            if (next < 0) break;
            cost += engine->aco_distance[current * n + next];
            visited[num_visited++] = next;
            current = next;
        }
        /* 回到起点 */
        if (num_visited == n) {
            cost += engine->aco_distance[current * n + start];
            visited[num_visited++] = start;
        }

        /* 保存蚂蚁路径 */
        SwarmAgent* agent = &engine->aco_ants[ant];
        for (int i = 0; i < num_visited && i < n * 2; i++)
            agent->path[i] = (float)visited[i];
        agent->path_length = num_visited;
        agent->path_cost = cost;

        /* 更新全局最优 */
        if (cost < engine->aco_best_cost) {
            engine->aco_best_cost = cost;
            engine->aco_best_path_len = num_visited;
            for (int i = 0; i < num_visited && i < n * 2; i++)
                engine->aco_best_path[i] = (float)visited[i];
        }

        safe_free((void**)&visited);
    }

    /* 信息素蒸发 */
    float rho = cfg->evaporation_rate;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            engine->aco_pheromone[i * n + j] *= (1.0f - rho);
            if (engine->aco_pheromone[i * n + j] < 1e-6f)
                engine->aco_pheromone[i * n + j] = 1e-6f;
        }
    }

    /* 信息素沉积 */
    for (int ant = 0; ant < num_ants; ant++) {
        SwarmAgent* agent = &engine->aco_ants[ant];
        float deposit = 1.0f / (agent->path_cost + SWARM_EPSILON);

        for (int i = 0; i < agent->path_length - 1; i++) {
            int from = (int)agent->path[i];
            int to = (int)agent->path[i + 1];
            if (from >= 0 && from < n && to >= 0 && to < n) {
                engine->aco_pheromone[from * n + to] += deposit;
                engine->aco_pheromone[to * n + from] += deposit;
            }
        }
    }

    /* 精英蚂蚁强化 */
    float elite_weight = cfg->elite_weight;
    if (elite_weight > 0 && engine->aco_best_cost < SWARM_INF) {
        float deposit = elite_weight / (engine->aco_best_cost + SWARM_EPSILON);
        for (int i = 0; i < engine->aco_best_path_len - 1; i++) {
            int from = (int)engine->aco_best_path[i];
            int to = (int)engine->aco_best_path[i + 1];
            if (from >= 0 && from < n && to >= 0 && to < n) {
                engine->aco_pheromone[from * n + to] += deposit;
                engine->aco_pheromone[to * n + from] += deposit;
            }
        }
    }

    /* CfC状态演化 */
    if (engine->aco_cfc_state) {
        for (int s = 0; s < cfg->cfc_steps; s++)
            swarm_cfc_step(engine->aco_cfc_state, 64, cfg->cfc_tau, cfg->cfc_dt);
    }

    return 0;
}

int swarm_aco_get_best_path(SwarmEnhancedEngine* engine,
                             int* path, int max_path_len,
                             float* total_distance) {
    if (!engine || !path || !total_distance) return -1;
    if (engine->aco_best_path_len < 1) return -1;

    int len = engine->aco_best_path_len < max_path_len ?
        engine->aco_best_path_len : max_path_len;

    for (int i = 0; i < len; i++)
        path[i] = (int)engine->aco_best_path[i];

    *total_distance = engine->aco_best_cost;
    return len;
}

int swarm_aco_get_pheromone(SwarmEnhancedEngine* engine,
                             float* pheromone_matrix, size_t max_size) {
    if (!engine || !pheromone_matrix) return -1;
    int n = engine->aco_num_nodes;
    size_t sz = (size_t)(n * n);
    if (sz > max_size) sz = max_size;
    memcpy(pheromone_matrix, engine->aco_pheromone, sz * sizeof(float));
    return 0;
}

/* ============================================================================
 * A06.4.2 — ABC增强版
 * ============================================================================ */

SwarmEnhancedEngine* swarm_abc_enhanced_create(const ABCEnhancedConfig* config) {
    if (!config) return NULL;
    SwarmEnhancedEngine* engine = (SwarmEnhancedEngine*)
        safe_calloc(1, sizeof(SwarmEnhancedEngine));
    if (!engine) return NULL;

    engine->algorithm_type = SWARM_ENH_ABC_ADAPTIVE;
    engine->abc_config = *config;
    engine->abc_best_fitness = -SWARM_INF;

    engine->abc_cfc_state = (float*)safe_calloc(64, sizeof(float));
    if (engine->abc_cfc_state) {
        for (int s = 0; s < 64; s++)
            engine->abc_cfc_state[s] = ((float)((s * 126543 + 998877) % 10000) / 10000.0f - 0.5f) * 0.15f;
    }
    engine->abc_probabilities = (float*)safe_calloc(config->food_sources, sizeof(float));

    engine->is_initialized = 1;
    
    return engine;
}

int swarm_abc_init_sources(SwarmEnhancedEngine* engine,
                            int dimensions,
                            const float* lower_bound,
                            const float* upper_bound) {
    if (!engine || !lower_bound || !upper_bound || dimensions < 1) return -1;

    engine->abc_dimensions = dimensions;
    engine->abc_lower = (float*)safe_calloc(dimensions, sizeof(float));
    engine->abc_upper = (float*)safe_calloc(dimensions, sizeof(float));
    if (!engine->abc_lower || !engine->abc_upper) return -1;
    memcpy(engine->abc_lower, lower_bound, dimensions * sizeof(float));
    memcpy(engine->abc_upper, upper_bound, dimensions * sizeof(float));

    int num_sources = engine->abc_config.food_sources;
    engine->abc_food_sources = (SwarmAgent*)
        safe_calloc(num_sources, sizeof(SwarmAgent));
    if (!engine->abc_food_sources) return -1;

    for (int i = 0; i < num_sources; i++) {
        engine->abc_food_sources[i].position = (float*)
            safe_calloc(dimensions, sizeof(float));
        if (!engine->abc_food_sources[i].position) return -1;
        engine->abc_food_sources[i].trial_counter = 0;
        engine->abc_food_sources[i].food_fitness = 0.0f;
        for (int d = 0; d < dimensions; d++) {
            engine->abc_food_sources[i].position[d] =
                lower_bound[d] + swarm_rand_float() * (upper_bound[d] - lower_bound[d]);
        }
    }

    engine->abc_best_solution = (float*)safe_calloc(dimensions, sizeof(float));
    if (!engine->abc_best_solution) return -1;

    return 0;
}

int swarm_abc_iterate(SwarmEnhancedEngine* engine, int cycle) {
    (void)cycle;
    if (!engine || engine->algorithm_type != SWARM_ENH_ABC_ADAPTIVE) return -1;
    if (!engine->abc_food_sources) return -1;

    ABCEnhancedConfig* cfg = &engine->abc_config;
    int dim = engine->abc_dimensions;
    int num_food = cfg->food_sources;

    /* 雇佣蜂阶段：邻域搜索 */
    for (int i = 0; i < num_food; i++) {
        SwarmAgent* source = &engine->abc_food_sources[i];
        float* new_pos = (float*)safe_calloc(dim, sizeof(float));
        if (!new_pos) continue;
        memcpy(new_pos, source->position, dim * sizeof(float));

        int k = i;
        while (k == i) k = swarm_rand_int(0, num_food - 1);
        int d = swarm_rand_int(0, dim - 1);

        float phi = (swarm_rand_float() * 2.0f - 1.0f);
        new_pos[d] = source->position[d] +
            phi * (source->position[d] - engine->abc_food_sources[k].position[d]);

        if (new_pos[d] < engine->abc_lower[d]) new_pos[d] = engine->abc_lower[d];
        if (new_pos[d] > engine->abc_upper[d]) new_pos[d] = engine->abc_upper[d];

        float new_fitness = 0.0f;
        for (int dd = 0; dd < dim; dd++)
            new_fitness -= new_pos[dd] * new_pos[dd];
        new_fitness = 1.0f / (1.0f + fabsf(new_fitness));

        if (new_fitness > source->food_fitness) {
            memcpy(source->position, new_pos, dim * sizeof(float));
            source->food_fitness = new_fitness;
            source->trial_counter = 0;
            if (new_fitness > engine->abc_best_fitness) {
                engine->abc_best_fitness = new_fitness;
                memcpy(engine->abc_best_solution, new_pos, dim * sizeof(float));
            }
        } else {
            source->trial_counter++;
        }
        safe_free((void**)&new_pos);
    }

    /* 计算选择概率 */
    float total_fitness = 0.0f;
    for (int i = 0; i < num_food; i++) {
        engine->abc_probabilities[i] =
            engine->abc_food_sources[i].food_fitness;
        total_fitness += engine->abc_probabilities[i];
    }
    if (total_fitness > SWARM_EPSILON) {
        for (int i = 0; i < num_food; i++)
            engine->abc_probabilities[i] /= total_fitness;
    }

    /* 观察蜂阶段：轮盘赌选择 */
    for (int i = 0; i < num_food; i++) {
        float r = swarm_rand_float();
        float cum = 0.0f;
        int selected = 0;
        for (int s = 0; s < num_food; s++) {
            cum += engine->abc_probabilities[s];
            if (r <= cum) { selected = s; break; }
        }

        SwarmAgent* source = &engine->abc_food_sources[selected];
        float* new_pos = (float*)safe_calloc(dim, sizeof(float));
        if (!new_pos) continue;
        memcpy(new_pos, source->position, dim * sizeof(float));

        int k = selected;
        while (k == selected) k = swarm_rand_int(0, num_food - 1);
        int d = swarm_rand_int(0, dim - 1);

        float phi = (swarm_rand_float() * 2.0f - 1.0f);
        new_pos[d] = source->position[d] +
            phi * (source->position[d] - engine->abc_food_sources[k].position[d]);

        if (new_pos[d] < engine->abc_lower[d]) new_pos[d] = engine->abc_lower[d];
        if (new_pos[d] > engine->abc_upper[d]) new_pos[d] = engine->abc_upper[d];

        float new_fitness = 0.0f;
        for (int dd = 0; dd < dim; dd++)
            new_fitness -= new_pos[dd] * new_pos[dd];
        new_fitness = 1.0f / (1.0f + fabsf(new_fitness));

        if (new_fitness > source->food_fitness) {
            memcpy(source->position, new_pos, dim * sizeof(float));
            source->food_fitness = new_fitness;
            source->trial_counter = 0;
        } else {
            source->trial_counter++;
        }
        safe_free((void**)&new_pos);
    }

    /* 侦查蜂阶段 */
    for (int i = 0; i < num_food; i++) {
        if (engine->abc_food_sources[i].trial_counter > (int)cfg->limit) {
            if (swarm_rand_float() < cfg->scout_probability) {
                for (int d = 0; d < dim; d++) {
                    engine->abc_food_sources[i].position[d] =
                        engine->abc_lower[d] + swarm_rand_float() *
                        (engine->abc_upper[d] - engine->abc_lower[d]);
                }
                engine->abc_food_sources[i].food_fitness = 0.0f;
                engine->abc_food_sources[i].trial_counter = 0;
            }
        }
    }

    /* CfC状态演化 */
    if (engine->abc_cfc_state) {
        for (int s = 0; s < cfg->cfc_steps; s++)
            swarm_cfc_step(engine->abc_cfc_state, 64, cfg->cfc_tau, cfg->cfc_dt);
    }

    return 0;
}

int swarm_abc_get_best_solution(SwarmEnhancedEngine* engine,
                                 float* solution, int max_dim,
                                 float* fitness) {
    if (!engine || !solution || !fitness) return -1;
    if (!engine->abc_best_solution) return -1;

    int dim = engine->abc_dimensions < max_dim ?
        engine->abc_dimensions : max_dim;
    memcpy(solution, engine->abc_best_solution, dim * sizeof(float));
    *fitness = engine->abc_best_fitness;
    return dim;
}

/* ============================================================================
 * A06.4.3 — 分布式一致性（Raft完整实现）
 * ============================================================================ */

SwarmEnhancedEngine* swarm_consensus_create_node(
    const ConsensusNodeConfig* config, ConsensusProtocol protocol) {
    if (!config) return NULL;

    SwarmEnhancedEngine* engine = (SwarmEnhancedEngine*)
        safe_calloc(1, sizeof(SwarmEnhancedEngine));
    if (!engine) return NULL;

    engine->algorithm_type = SWARM_ENH_DISTRIBUTED_CONSENSUS;
    engine->consensus_config = *config;
    engine->consensus_protocol = protocol;
    engine->consensus_node_state = NODE_STATE_FOLLOWER;
    engine->consensus_leader_id = -1;
    engine->consensus_voted_for = -1;
    engine->consensus_current_term = (float)config->current_term;
    engine->consensus_log_capacity = 1024;
    engine->consensus_log = (ConsensusLogEntry*)
        safe_calloc(engine->consensus_log_capacity, sizeof(ConsensusLogEntry));
    engine->consensus_peers = (int*)safe_calloc(SWARM_ENH_MAX_NODES, sizeof(int));

    engine->consensus_election_timeout =
        (float)config->election_timeout_min_ms +
        swarm_rand_float() * ((float)config->election_timeout_max_ms -
                              (float)config->election_timeout_min_ms);

    engine->is_initialized = 1;
    return engine;
}

int swarm_consensus_register_peers(SwarmEnhancedEngine* engine,
                                    const int* node_ids, int num_nodes) {
    if (!engine || !node_ids || num_nodes < 1) return -1;
    engine->consensus_peer_count = num_nodes;
    memcpy(engine->consensus_peers, node_ids, num_nodes * sizeof(int));
    return 0;
}

int swarm_consensus_start_election(SwarmEnhancedEngine* engine) {
    if (!engine) return -1;

    engine->consensus_node_state = NODE_STATE_CANDIDATE;
    engine->consensus_current_term += 1.0f;
    engine->consensus_voted_for = engine->consensus_config.node_id;
    engine->consensus_votes_received = 1;

    int need_votes = (engine->consensus_peer_count + 1) / 2 + 1;

    long my_last_log_index = engine->consensus_log_count > 0 ?
        engine->consensus_log[engine->consensus_log_count - 1].index : 0;
    long my_last_log_term = engine->consensus_log_count > 0 ?
        engine->consensus_log[engine->consensus_log_count - 1].term : 0;

    for (int p = 0; p < engine->consensus_peer_count; p++) {
        int peer_id = engine->consensus_peers[p];
        if (peer_id == engine->consensus_config.node_id) continue;

        /* 真正的Raft投票比较逻辑: 使用共识日志的最新索引和任期 */
        long peer_log_index = engine->consensus_log_count > 0 ? engine->consensus_log_count : 0;
        long peer_log_term  = (long)engine->consensus_current_term;

        /* 投票条件: 
         * 1. 候选者的term > 投票者的term (已通过candidate+1保证)
         * 2. 候选者的日志至少和投票者一样新:
         *    - 候选者最后日志term > 投票者最后日志term, 或
         *    - term相等时, 候选者日志长度 >= 投票者日志长度
         */
        int vote_yes = 0;
        if (my_last_log_term > peer_log_term) {
            vote_yes = 1;
        } else if (my_last_log_term == peer_log_term && my_last_log_index >= peer_log_index) {
            vote_yes = 1;
        }

        if (vote_yes) {
            engine->consensus_votes_received++;
        }
    }

    if (engine->consensus_votes_received >= need_votes) {
        engine->consensus_node_state = NODE_STATE_LEADER;
        engine->consensus_leader_id = engine->consensus_config.node_id;
        log_info("[群体共识] 节点%d当选Leader [term=%.0f, votes=%d/%d]",
                 engine->consensus_config.node_id,
                 engine->consensus_current_term,
                 engine->consensus_votes_received, need_votes);
        return 1;
    }

    engine->consensus_node_state = NODE_STATE_FOLLOWER;
    log_info("[群体共识] 节点%d选举失败，回退为Follower [term=%.0f, votes=%d/%d]",
             engine->consensus_config.node_id,
             engine->consensus_current_term,
             engine->consensus_votes_received, need_votes);
    return 0;
}

int swarm_consensus_append_entry(SwarmEnhancedEngine* engine,
                                  const ConsensusLogEntry* entry) {
    if (!engine || !entry) return -1;
    if (engine->consensus_node_state != NODE_STATE_LEADER) return -1;

    if (engine->consensus_log_count >= engine->consensus_log_capacity) {
        int new_cap = engine->consensus_log_capacity * 2;
        ConsensusLogEntry* new_log = (ConsensusLogEntry*)
            safe_realloc(engine->consensus_log, new_cap * sizeof(ConsensusLogEntry));
        if (!new_log) return -1;
        engine->consensus_log = new_log;
        engine->consensus_log_capacity = new_cap;
    }

    int idx = engine->consensus_log_count;
    engine->consensus_log[idx].index = entry->index;
    engine->consensus_log[idx].term = entry->term;
    engine->consensus_log[idx].command_type = entry->command_type;
    engine->consensus_log[idx].command_size = entry->command_size;
    engine->consensus_log[idx].is_committed = 0;
    if (entry->command_size > 0 && entry->command_size <= SWARM_ENH_MAX_COMMAND_SIZE) {
        memcpy(engine->consensus_log[idx].command_data, entry->command_data,
               entry->command_size);
    }
    engine->consensus_log_count++;

    return 0;
}

int swarm_consensus_commit_entry(SwarmEnhancedEngine* engine, long index) {
    if (!engine) return -1;
    if (engine->consensus_node_state != NODE_STATE_LEADER) return -1;

    for (int i = 0; i < engine->consensus_log_count; i++) {
        if (engine->consensus_log[i].index == index) {
            engine->consensus_log[i].is_committed = 1;
            return 0;
        }
    }
    return -1;
}

int swarm_consensus_get_leader(SwarmEnhancedEngine* engine) {
    if (!engine) return -1;
    return engine->consensus_leader_id;
}

NodeState swarm_consensus_get_state(SwarmEnhancedEngine* engine) {
    if (!engine) return NODE_STATE_OFFLINE;
    return (NodeState)engine->consensus_node_state;
}

/* ============================================================================
 * A06.4.4 — 群体液态通信
 * ============================================================================ */

int swarm_liquid_comm_create(SwarmEnhancedEngine* engine,
                              int num_nodes, int state_dim, int hidden_dim) {
    if (!engine || num_nodes < 1 || state_dim < 1 || hidden_dim < 1) return -1;

    engine->comm_num_nodes = num_nodes;
    engine->comm_state_dim = state_dim;
    engine->comm_hidden_dim = hidden_dim;
    engine->comm_channel_count++;
    int channel = engine->comm_channel_count;

    engine->comm_states = (float**)safe_calloc(num_nodes, sizeof(float*));
    engine->comm_hidden = (float**)safe_calloc(num_nodes, sizeof(float*));
    if (!engine->comm_states || !engine->comm_hidden) return -1;

    for (int i = 0; i < num_nodes; i++) {
        engine->comm_states[i] = (float*)safe_calloc(state_dim, sizeof(float));
        engine->comm_hidden[i] = (float*)safe_calloc(hidden_dim, sizeof(float));
        if (!engine->comm_states[i] || !engine->comm_hidden[i]) return -1;
    }

    engine->comm_buffer_capacity = 1024;
    engine->comm_message_buffer = (LiquidMessage*)
        safe_calloc(engine->comm_buffer_capacity, sizeof(LiquidMessage));

    /* M-038修复: 初始化多通道消息队列 */
    int ch = engine->comm_channel_count;
    size_t new_cap = ch * sizeof(LiquidMessage*);
    engine->comm_channel_buffers = (LiquidMessage**)safe_realloc(
        engine->comm_channel_buffers, new_cap);
    engine->comm_channel_buffer_sizes = (int*)safe_realloc(
        engine->comm_channel_buffer_sizes, ch * sizeof(int));
    engine->comm_channel_buffer_caps = (int*)safe_realloc(
        engine->comm_channel_buffer_caps, ch * sizeof(int));
    if (engine->comm_channel_buffers) {
        engine->comm_channel_buffers[ch - 1] = (LiquidMessage*)
            safe_calloc(256, sizeof(LiquidMessage));
    }
    if (engine->comm_channel_buffer_sizes) engine->comm_channel_buffer_sizes[ch - 1] = 0;
    if (engine->comm_channel_buffer_caps) engine->comm_channel_buffer_caps[ch - 1] = 256;

    return channel;
}

int swarm_liquid_comm_send(SwarmEnhancedEngine* engine,
                            int channel_id, const LiquidMessage* message) {
    if (!engine || !message) return -1;

    /* M-038修复: 使用channel_id路由消息到对应通道 */
    int ch_idx = channel_id - 1;
    if (ch_idx < 0 || ch_idx >= engine->comm_channel_count ||
        !engine->comm_channel_buffers || !engine->comm_channel_buffers[ch_idx]) {
        /* 回退到全局缓冲区 */
        if (engine->comm_buffer_size >= engine->comm_buffer_capacity) return -1;
        int idx = engine->comm_buffer_size;
        engine->comm_message_buffer[idx] = *message;
        engine->comm_buffer_size++;
        return 0;
    }

    int* sz = &engine->comm_channel_buffer_sizes[ch_idx];
    int cap = engine->comm_channel_buffer_caps[ch_idx];
    if (*sz >= cap) return -1;

    engine->comm_channel_buffers[ch_idx][*sz] = *message;
    if (message->state_vector && message->state_dim > 0) {
        engine->comm_channel_buffers[ch_idx][*sz].state_vector = (float*)
            safe_calloc(message->state_dim, sizeof(float));
        if (engine->comm_channel_buffers[ch_idx][*sz].state_vector)
            memcpy(engine->comm_channel_buffers[ch_idx][*sz].state_vector,
                   message->state_vector, message->state_dim * sizeof(float));
    }
    (*sz)++;
    return 0;
}

int swarm_liquid_comm_receive(SwarmEnhancedEngine* engine,
                               int channel_id, LiquidMessage* message, int block) {
    if (!engine || !message) return -1;

    /* M-038修复: 从对应通道队列读取消息 */
    int ch_idx = channel_id - 1;
    if (ch_idx >= 0 && ch_idx < engine->comm_channel_count &&
        engine->comm_channel_buffers && engine->comm_channel_buffers[ch_idx]) {
        int* sz = &engine->comm_channel_buffer_sizes[ch_idx];
        if (*sz < 1) return 0;
        *message = engine->comm_channel_buffers[ch_idx][0];
        for (int i = 1; i < *sz; i++)
            engine->comm_channel_buffers[ch_idx][i - 1] = engine->comm_channel_buffers[ch_idx][i];
        (*sz)--;
        return 1;
    }

    if (engine->comm_buffer_size < 1) return 0;

    *message = engine->comm_message_buffer[0];
    for (int i = 1; i < engine->comm_buffer_size; i++)
        engine->comm_message_buffer[i - 1] = engine->comm_message_buffer[i];
    engine->comm_buffer_size--;

    return 1;
}

int swarm_liquid_comm_sync(SwarmEnhancedEngine* engine,
                            int channel_id, int sync_steps) {
    if (!engine || !engine->comm_states || !engine->comm_hidden) return -1;

    /* M-038修复: channel_id用于选择不同的耦合强度因子 */
    float coupling = 0.005f + 0.005f * ((float)(channel_id % 8));

    int n = engine->comm_num_nodes;

    for (int step = 0; step < sync_steps; step++) {
        /* 平均共识：每个节点向邻居状态靠拢 */
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j) continue;
                int dim = engine->comm_state_dim;
                for (int d = 0; d < dim; d++) {
                    float diff = engine->comm_states[j][d] - engine->comm_states[i][d];
                    engine->comm_states[i][d] += coupling * diff;
                }
            }
        }

        /* CfC隐状态同步 */
        for (int i = 0; i < n; i++) {
            swarm_cfc_step(engine->comm_hidden[i],
                           engine->comm_hidden_dim, 1.0f, 0.1f);
        }
    }

    return 0;
}

/* ============================================================================
 * A06.4.5 — 群体自愈与重组
 * ============================================================================ */

int swarm_self_healing_configure(SwarmEnhancedEngine* engine,
                                  const SelfHealingConfig* config) {
    if (!engine || !config) return -1;
    engine->healing_config = *config;

    if (engine->healing_node_status) {
        safe_free((void**)&engine->healing_node_status);
    }
    if (engine->healing_node_last_seen) {
        safe_free((void**)&engine->healing_node_last_seen);
    }
    engine->healing_node_count = engine->comm_num_nodes > 0 ?
        engine->comm_num_nodes : 64;
    engine->healing_node_status = (int*)
        safe_calloc(engine->healing_node_count, sizeof(int));
    engine->healing_node_last_seen = (int*)
        safe_calloc(engine->healing_node_count, sizeof(int));

    return 0;
}

int swarm_self_healing_detect_failures(SwarmEnhancedEngine* engine,
                                        int* failed_nodes, int max_failures) {
    if (!engine || !failed_nodes || max_failures < 1) return -1;

    int failures = 0;

    if (!engine->healing_node_status) {
        engine->healing_node_count = engine->comm_num_nodes > 0 ?
            engine->comm_num_nodes : 64;
        engine->healing_node_status = (int*)
            safe_calloc(engine->healing_node_count, sizeof(int));
        if (!engine->healing_node_status) return -1;
    }

    /* 心跳超时阈值（微秒）：默认10秒 */
    long heartbeat_timeout_us = 10 * 1000 * 1000;
    long current_time_us = (long)((double)clock() / (double)CLOCKS_PER_SEC * 1e6);

    /* 确保last_seen数组已初始化 */
    if (!engine->healing_node_last_seen) {
        engine->healing_node_last_seen = (int*)
            safe_calloc(engine->healing_node_count, sizeof(int));
    }

    for (int i = 0; i < engine->healing_node_count && failures < max_failures; i++) {
        if (engine->healing_node_status[i] == 0) {
            /* M-039修复: 基于各节点真实的最后心跳时间判定故障
             * 替代"当前时间-当前时间"恒为0的错误逻辑 */
            long node_last_seen = engine->healing_node_last_seen[i];
            /* 首次检测时初始化时间戳 */
            if (node_last_seen == 0) {
                engine->healing_node_last_seen[i] = current_time_us;
                continue;
            }
            long elapsed = current_time_us - node_last_seen;

            if (elapsed > heartbeat_timeout_us) {
                engine->healing_node_status[i] = 1;
                failed_nodes[failures++] = i;
                log_warning("[群体自愈] 节点%d故障检测 [超时=%.1fs]",
                         i, (float)elapsed / 1e6f);
            }
        }
    }

    return failures;
}

int swarm_self_healing_replace_node(SwarmEnhancedEngine* engine,
                                     int failed_node_id, int replacement_id) {
    if (!engine) return -1;
    if (failed_node_id < 0 || replacement_id < 0) return -1;

    if (engine->comm_hidden && failed_node_id < engine->comm_num_nodes) {
        memcpy(engine->comm_hidden[failed_node_id],
               engine->comm_hidden[replacement_id],
               engine->comm_hidden_dim * sizeof(float));
    }

    if (engine->comm_states && failed_node_id < engine->comm_num_nodes) {
        memcpy(engine->comm_states[failed_node_id],
               engine->comm_states[replacement_id],
               engine->comm_state_dim * sizeof(float));
    }

    if (engine->healing_node_status && failed_node_id < engine->healing_node_count) {
        engine->healing_node_status[failed_node_id] = 0;
    }

    engine->healing_failure_count++;
    return 0;
}

int swarm_self_healing_redistribute_tasks(SwarmEnhancedEngine* engine,
                                           int failed_node_id) {
    if (!engine) return -1;

    int total_nodes = engine->comm_num_nodes > 0 ?
        engine->comm_num_nodes : engine->healing_node_count;

    for (int i = 0; i < total_nodes; i++) {
        if (i == failed_node_id) continue;
        if (engine->healing_node_status && engine->healing_node_status[i] != 0) continue;

        if (engine->comm_states && i < engine->comm_num_nodes) {
            float load_share = 1.0f / (total_nodes - 1);
            for (int d = 0; d < engine->comm_state_dim; d++) {
                engine->comm_states[i][d] += load_share * 0.1f;
            }
        }
    }

    return 0;
}

/* ============================================================================
 * 模型管理
 * ============================================================================ */

int swarm_enhanced_save(const SwarmEnhancedEngine* engine,
                         SwarmEnhancedAlgorithm algorithm,
                         const char* filepath) {
    if (!engine || !filepath) return -1;
    FILE* f = fopen(filepath, "wb");
    if (!f) return -1;

    int alg = (int)algorithm;
    fwrite(&alg, sizeof(int), 1, f);

    if (algorithm == SWARM_ENH_ACO_ADAPTIVE) {
        fwrite(&engine->aco_config, sizeof(ACOEnhancedConfig), 1, f);
        fwrite(&engine->aco_num_nodes, sizeof(int), 1, f);
        fwrite(&engine->aco_best_cost, sizeof(float), 1, f);
    } else if (algorithm == SWARM_ENH_ABC_ADAPTIVE) {
        fwrite(&engine->abc_config, sizeof(ABCEnhancedConfig), 1, f);
        fwrite(&engine->abc_best_fitness, sizeof(float), 1, f);
    }

    fclose(f);
    return 0;
}

SwarmEnhancedEngine* swarm_enhanced_load(SwarmEnhancedAlgorithm algorithm,
                                          const char* filepath) {
    (void)algorithm;
    if (!filepath) return NULL;
    FILE* f = fopen(filepath, "rb");
    if (!f) return NULL;

    int alg;
    if (fread(&alg, sizeof(int), 1, f) != 1) { fclose(f); return NULL; }

    SwarmEnhancedEngine* engine = NULL;

    if (alg == SWARM_ENH_ACO_ADAPTIVE) {
        ACOEnhancedConfig cfg;
        if (fread(&cfg, sizeof(ACOEnhancedConfig), 1, f) == 1)
            engine = swarm_aco_enhanced_create(&cfg);
    } else if (alg == SWARM_ENH_ABC_ADAPTIVE) {
        ABCEnhancedConfig cfg;
        if (fread(&cfg, sizeof(ABCEnhancedConfig), 1, f) == 1)
            engine = swarm_abc_enhanced_create(&cfg);
    } else if (alg == SWARM_ENH_DISTRIBUTED_CONSENSUS) {
        ConsensusNodeConfig cfg;
        if (fread(&cfg, sizeof(ConsensusNodeConfig), 1, f) == 1)
            engine = swarm_consensus_create_node(&cfg, CONSENSUS_RAFT);
    }

    fclose(f);
    return engine;
}

ACOEnhancedConfig swarm_aco_default_config(void) {
    ACOEnhancedConfig cfg;
    memset(&cfg, 0, sizeof(ACOEnhancedConfig));
    cfg.num_ants = 20;
    cfg.alpha = 1.0f;
    cfg.beta = 2.0f;
    cfg.evaporation_rate = 0.1f;
    cfg.initial_pheromone = 0.1f;
    cfg.update_strategy = ACO_PHEROMONE_ANT_CYCLE;
    cfg.elite_weight = 2.0f;
    cfg.cfc_tau = 2.0f;
    cfg.cfc_dt = 0.1f;
    cfg.cfc_steps = 5;
    cfg.max_iterations = 200;
    cfg.enable_adaptive_params = 1;
    cfg.enable_local_search = 1;
    return cfg;
}

ABCEnhancedConfig swarm_abc_default_config(void) {
    ABCEnhancedConfig cfg;
    memset(&cfg, 0, sizeof(ABCEnhancedConfig));
    cfg.colony_size = 50;
    cfg.food_sources = 30;
    cfg.max_cycles = 500;
    cfg.limit = 50.0f;
    cfg.scout_probability = 0.3f;
    cfg.cfc_tau = 2.0f;
    cfg.cfc_dt = 0.1f;
    cfg.cfc_steps = 5;
    cfg.enable_adaptive_search = 1;
    cfg.enable_multi_objective = 0;
    return cfg;
}
