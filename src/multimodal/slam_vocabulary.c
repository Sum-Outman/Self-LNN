/**
 * @file slam_vocabulary.c
 * @brief SLAM视觉词汇表与共视图模块
 *
 * 实现视觉词袋（BoW）模型：
 * - 词汇表树构建（K-means++聚类）
 * - TF-IDF权重计算
 * - BoW向量计算与相似度
 * - 共视图（Covisibility Graph）管理
 * - 本质图（Essential Graph）构建
 */

#include "selflnn/multimodal/slam_internal.h"

/* ==================== 词汇表节点操作 ==================== */

VocabTreeNode* slam_vocab_node_create(int descriptor_length) {
    VocabTreeNode* node = (VocabTreeNode*)slam_malloc(sizeof(VocabTreeNode));
    if (!node) return NULL;
    memset(node, 0, sizeof(VocabTreeNode));
    node->descriptor_length = descriptor_length;
    node->weight = 1.0f;
    node->num_children = 0;
    node->children = NULL;
    node->feature_indices = NULL;
    node->num_features_assigned = 0;
    node->image_count = 0;
    return node;
}

void slam_vocab_node_free(VocabTreeNode* node) {
    if (!node) return;
    for (int i = 0; i < node->num_children; i++) {
        slam_vocab_node_free(node->children[i]);
    }
    slam_free(node->children);
    slam_free(node->feature_indices);
    slam_free(node);
}

int slam_vocab_node_train(VocabTreeNode* node, const float* descriptors,
                          int num_descriptors, int descriptor_length,
                          int depth, int max_depth, int branch_factor) {
    if (!node || !descriptors || num_descriptors < 1) return -1;

    node->descriptor_length = descriptor_length;

    if (depth >= max_depth || num_descriptors <= branch_factor) {
        node->is_leaf = 1;
        float avg_desc[SLAM_FEATURE_DESC_LENGTH];
        memset(avg_desc, 0, descriptor_length * sizeof(float));
        for (int i = 0; i < num_descriptors; i++) {
            for (int j = 0; j < descriptor_length; j++) {
                avg_desc[j] += descriptors[i * descriptor_length + j];
            }
        }
        float inv_n = 1.0f / num_descriptors;
        for (int j = 0; j < descriptor_length; j++) {
            node->descriptor[j] = avg_desc[j] * inv_n;
            node->descriptor[j + descriptor_length] = node->descriptor[j];
        }
        node->num_features_assigned = num_descriptors;
        return 0;
    }

    node->is_leaf = 0;
    node->num_children = branch_factor;

    node->children = (VocabTreeNode**)slam_malloc((size_t)branch_factor * sizeof(VocabTreeNode*));
    if (!node->children) return -1;

    int k = branch_factor;
    float* centers = (float*)slam_malloc((size_t)k * descriptor_length * sizeof(float));
    if (!centers) return -1;

    int* assignments = (int*)slam_malloc((size_t)num_descriptors * sizeof(int));
    if (!assignments) { slam_free(centers); return -1; }

    int* counts = (int*)slam_malloc((size_t)k * sizeof(int));
    if (!counts) { slam_free(centers); slam_free(assignments); return -1; }

    slam_kmeans_plus_plus(descriptors, num_descriptors, descriptor_length, k, centers);

    for (int iter = 0; iter < 20; iter++) {
        for (int i = 0; i < num_descriptors; i++) {
            float best_dist = 1e30f;
            int best_k = 0;
            for (int j = 0; j < k; j++) {
                float dist = 0;
                for (int d = 0; d < descriptor_length; d++) {
                    float diff = descriptors[i*descriptor_length+d] - centers[j*descriptor_length+d];
                    dist += diff * diff;
                }
                if (dist < best_dist) { best_dist = dist; best_k = j; }
            }
            assignments[i] = best_k;
        }

        memset(counts, 0, (size_t)k * sizeof(int));
        memset(centers, 0, (size_t)k * descriptor_length * sizeof(float));

        for (int i = 0; i < num_descriptors; i++) {
            int c = assignments[i];
            counts[c]++;
            for (int d = 0; d < descriptor_length; d++) {
                centers[c*descriptor_length+d] += descriptors[i*descriptor_length+d];
            }
        }

        for (int j = 0; j < k; j++) {
            if (counts[j] > 0) {
                float inv = 1.0f / counts[j];
                for (int d = 0; d < descriptor_length; d++) {
                    centers[j*descriptor_length+d] *= inv;
                }
            }
        }
    }

    for (int j = 0; j < k; j++) {
        node->children[j] = slam_vocab_node_create(descriptor_length);
        if (!node->children[j]) { slam_free(centers); slam_free(assignments); slam_free(counts); return -1; }
    }

    int* cluster_sizes = (int*)slam_calloc((size_t)k, sizeof(int));
    if (!cluster_sizes) { slam_free(centers); slam_free(assignments); slam_free(counts); return -1; }

    for (int i = 0; i < num_descriptors; i++) cluster_sizes[assignments[i]]++;

    for (int j = 0; j < k; j++) {
        if (cluster_sizes[j] == 0) continue;
        float* cluster_data = (float*)slam_malloc((size_t)cluster_sizes[j] * descriptor_length * sizeof(float));
        if (!cluster_data) continue;
        int idx = 0;
        for (int i = 0; i < num_descriptors; i++) {
            if (assignments[i] == j) {
                memcpy(&cluster_data[idx * descriptor_length], &descriptors[i * descriptor_length],
                       descriptor_length * sizeof(float));
                idx++;
            }
        }
        slam_vocab_node_train(node->children[j], cluster_data, cluster_sizes[j],
                              descriptor_length, depth + 1, max_depth, branch_factor);
        memcpy(node->children[j]->descriptor, &centers[j*descriptor_length],
               descriptor_length * sizeof(float));
        slam_free(cluster_data);
    }

    slam_free(centers);
    slam_free(assignments);
    slam_free(counts);
    slam_free(cluster_sizes);
    return 0;
}

/* ==================== 词汇表初始化 ==================== */

int slam_vocabulary_init(InternalVocabulary* vocab, const VisualVocabularyConfig* config) {
    if (!vocab) return -1;

    memset(vocab, 0, sizeof(InternalVocabulary));
    vocab->vocabulary_size = SLAM_VOCABULARY_SIZE;
    vocab->vocabulary_depth = SLAM_VOCABULARY_DEPTH;
    vocab->branching_factor = SLAM_VOCABULARY_BRANCHING;
    vocab->descriptor_length = SLAM_FEATURE_DESC_LENGTH;
    vocab->enable_tf_idf = 1;
    vocab->incremental_update_enabled = (config) ? config->enable_incremental_update : 0;

    vocab->root = slam_vocab_node_create(vocab->descriptor_length);
    if (!vocab->root) return -1;

    vocab->is_built = 0;
    return 0;
}

/* ==================== 词汇表构建 ==================== */

int slam_vocabulary_build(InternalVocabulary* vocab, const float* all_descriptors,
                          int num_descriptors, int descriptor_length) {
    if (!vocab || !all_descriptors || num_descriptors < 1) return -1;

    if (vocab->root) slam_vocab_node_free(vocab->root);
    vocab->root = slam_vocab_node_create(descriptor_length);
    if (!vocab->root) return -1;

    int result = slam_vocab_node_train(vocab->root, all_descriptors, num_descriptors,
                                       descriptor_length, 0, vocab->vocabulary_depth,
                                       vocab->branching_factor);
    if (result != 0) return result;

    int leaf_count = 0;
    int total_features = 0;
    if (vocab->root->is_leaf) {
        leaf_count = 1;
    } else {
        VocabTreeNode** stack = (VocabTreeNode**)slam_malloc(1024 * sizeof(VocabTreeNode*));
        if (!stack) return -1;
        int stack_ptr = 0;
        stack[stack_ptr++] = vocab->root;

        while (stack_ptr > 0) {
            VocabTreeNode* node = stack[--stack_ptr];
            if (node->is_leaf) {
                leaf_count++;
                total_features += node->num_features_assigned;
            } else {
                for (int i = 0; i < node->num_children; i++) {
                    if (node->children[i] && stack_ptr < 1024) {
                        stack[stack_ptr++] = node->children[i];
                    }
                }
            }
        }
        slam_free(stack);
    }

    vocab->num_leaf_nodes = leaf_count;
    vocab->total_trained_features = num_descriptors;

    if (vocab->enable_tf_idf) {
        slam_vocabulary_update_tfidf(vocab);
    }

    vocab->is_built = 1;
    return 0;
}

/* ==================== BoW向量计算 ==================== */

int slam_vocabulary_compute_bow(InternalVocabulary* vocab, const float* descriptors,
                               int num_descriptors, int descriptor_length,
                               float* bow_vector, int* bow_vector_size) {
    if (!vocab || !vocab->root || !descriptors || !bow_vector || !bow_vector_size) return -1;
    if (!vocab->is_built) return -1;

    memset(bow_vector, 0, SLAM_VOCABULARY_SIZE * sizeof(float));

    for (int i = 0; i < num_descriptors; i++) {
        int leaf_idx = slam_vocab_node_assign(vocab->root, &descriptors[i * descriptor_length],
                                               descriptor_length);
        if (leaf_idx >= 0 && leaf_idx < SLAM_VOCABULARY_SIZE) {
            bow_vector[leaf_idx] += 1.0f;
        }
    }

    float total = 0;
    for (int i = 0; i < SLAM_VOCABULARY_SIZE; i++) total += bow_vector[i];

    if (total > 0) {
        float inv_total = 1.0f / total;
        for (int i = 0; i < SLAM_VOCABULARY_SIZE; i++) {
            bow_vector[i] *= inv_total;
        }
    }

    if (vocab->enable_tf_idf && vocab->num_trained_frames > 0) {
        for (int i = 0; i < vocab->num_leaf_nodes && i < SLAM_VOCABULARY_SIZE; i++) {
            if (vocab->leaf_words) {
                bow_vector[i] *= vocab->leaf_words[i].weight;
            }
        }
    }

    *bow_vector_size = vocab->num_leaf_nodes;
    return 0;
}

/* ==================== BoW相似度计算 ==================== */

float slam_vocabulary_compute_similarity(const float* bow1, int size1,
                                        const float* bow2, int size2) {
    if (!bow1 || !bow2 || size1 < 1 || size2 < 1) return 0;

    int max_size = (size1 < size2) ? size1 : size2;
    float dot = 0, norm1 = 0, norm2 = 0;

    for (int i = 0; i < max_size; i++) {
        dot += bow1[i] * bow2[i];
        norm1 += bow1[i] * bow1[i];
        norm2 += bow2[i] * bow2[i];
    }

    if (norm1 < SLAM_EPSILON || norm2 < SLAM_EPSILON) return 0;
    return dot / (sqrtf(norm1) * sqrtf(norm2));
}

/* ==================== 添加帧到词汇表 ==================== */

int slam_vocabulary_add_frame(InternalVocabulary* vocab, const float* descriptors,
                             int num_descriptors, int descriptor_length) {
    if (!vocab || !descriptors || num_descriptors < 1) return -1;
    if (!vocab->is_built) return -1;

    float* bow = (float*)slam_malloc(SLAM_VOCABULARY_SIZE * sizeof(float));
    if (!bow) return -1;

    int bow_size = 0;
    if (slam_vocabulary_compute_bow(vocab, descriptors, num_descriptors,
                                    descriptor_length, bow, &bow_size) != 0) {
        slam_free(bow);
        return -1;
    }

    vocab->num_trained_frames++;

    if (vocab->enable_tf_idf) {
        slam_vocabulary_update_tfidf(vocab);
    }

    slam_free(bow);
    return 0;
}

/* ==================== TF-IDF更新 ==================== */

int slam_vocabulary_update_tfidf(InternalVocabulary* vocab) {
    if (!vocab || !vocab->root) return -1;

    int total_frames = vocab->num_trained_frames;
    if (total_frames < 1) total_frames = 1;

    slam_free(vocab->leaf_words);
    vocab->leaf_words = (VisualWord*)slam_malloc((size_t)vocab->num_leaf_nodes * sizeof(VisualWord));
    if (!vocab->leaf_words) return -1;

    int leaf_idx = 0;
    VocabTreeNode** stack = (VocabTreeNode**)slam_malloc(1024 * sizeof(VocabTreeNode*));
    if (!stack) return -1;
    int stack_ptr = 0;
    stack[stack_ptr++] = vocab->root;

    while (stack_ptr > 0) {
        VocabTreeNode* node = stack[--stack_ptr];
        if (node->is_leaf) {
            if (leaf_idx < vocab->num_leaf_nodes) {
                vocab->leaf_words[leaf_idx].descriptor[0] = 0; /* id replaced by index */
                memcpy(vocab->leaf_words[leaf_idx].descriptor, node->descriptor,
                       vocab->descriptor_length * sizeof(float));
                int ni = node->image_count > 0 ? node->image_count : 1;
                vocab->leaf_words[leaf_idx].weight = logf((float)total_frames / ni);
                if (vocab->leaf_words[leaf_idx].weight < 0) vocab->leaf_words[leaf_idx].weight = 0;
                leaf_idx++;
            }
        } else {
            for (int i = 0; i < node->num_children; i++) {
                if (node->children[i] && stack_ptr < 1024) {
                    stack[stack_ptr++] = node->children[i];
                }
            }
        }
    }

    slam_free(stack);
    return 0;
}

/* ==================== 词汇表节点分配 ==================== */

int slam_vocab_node_assign(VocabTreeNode* node, const float* descriptor, int descriptor_length) {
    if (!node || !descriptor) return -1;

    if (node->is_leaf) return node->leaf_index;

    float best_dist = 1e30f;
    int best_child = 0;

    for (int i = 0; i < node->num_children; i++) {
        if (!node->children[i]) continue;
        float dist = 0;
        for (int d = 0; d < descriptor_length; d++) {
            float diff = descriptor[d] - node->children[i]->descriptor[d];
            dist += diff * diff;
        }
        if (dist < best_dist) { best_dist = dist; best_child = i; }
    }

    if (best_child >= 0 && best_child < node->num_children && node->children[best_child]) {
        return slam_vocab_node_assign(node->children[best_child], descriptor, descriptor_length);
    }

    return -1;
}

/* ==================== K-means++聚类 ==================== */

int slam_kmeans_plus_plus(const float* descriptors, int num_descriptors,
                          int descriptor_length, int k, float* centers) {
    if (!descriptors || !centers || k < 1 || num_descriptors < k) return -1;

    /* K-012修复：安全随机数 */
    int first_idx = (int)(secure_random_int((uint32_t)(num_descriptors - 1)));
    memcpy(centers, &descriptors[first_idx * descriptor_length],
           descriptor_length * sizeof(float));

    float* min_dists = (float*)slam_malloc((size_t)num_descriptors * sizeof(float));
    if (!min_dists) return -1;

    for (int c = 1; c < k; c++) {
        float total_dist = 0;
        for (int i = 0; i < num_descriptors; i++) {
            float min_d = 1e30f;
            for (int j = 0; j < c; j++) {
                float d = 0;
                for (int l = 0; l < descriptor_length; l++) {
                    float diff = descriptors[i*descriptor_length+l] - centers[j*descriptor_length+l];
                    d += diff*diff;
                }
                if (d < min_d) min_d = d;
            }
            min_dists[i] = min_d;
            total_dist += min_d;
        }

        /* K-012修复：安全随机数 */
        float r = secure_random_float() * total_dist;
        float accum = 0;
        int chosen = 0;
        for (int i = 0; i < num_descriptors; i++) {
            accum += min_dists[i];
            if (accum >= r) { chosen = i; break; }
        }
        memcpy(&centers[c*descriptor_length], &descriptors[chosen*descriptor_length],
               descriptor_length * sizeof(float));
    }

    slam_free(min_dists);
    return 0;
}

/* ==================== 词汇表释放 ==================== */

void slam_vocabulary_free(InternalVocabulary* vocab) {
    if (!vocab) return;
    if (vocab->root) {
        slam_vocab_node_free(vocab->root);
        vocab->root = NULL;
    }
    slam_free(vocab->leaf_words);
    memset(vocab, 0, sizeof(InternalVocabulary));
}

/* ==================== 共视图初始化 ==================== */

int slam_covisibility_init(InternalCovisibility* cov, int max_frames) {
    if (!cov) return -1;

    cov->max_frames = max_frames;
    cov->num_frames = 0;

    cov->adjacency_matrix = (int*)slam_calloc((size_t)max_frames * max_frames, sizeof(int));
    cov->connected_frames = (int*)slam_calloc((size_t)max_frames * max_frames, sizeof(int));
    cov->connected_frame_counts = (int*)slam_calloc((size_t)max_frames, sizeof(int));

    if (!cov->adjacency_matrix || !cov->connected_frames || !cov->connected_frame_counts) {
        slam_free(cov->adjacency_matrix);
        slam_free(cov->connected_frames);
        slam_free(cov->connected_frame_counts);
        return -1;
    }

    cov->essential_graph_edges_from = NULL;
    cov->essential_graph_edges_to = NULL;
    cov->essential_graph_weights = NULL;
    cov->num_essential_edges = 0;
    cov->essential_graph_built = 0;

    return 0;
}

/* ==================== 共视图释放 ==================== */

void slam_covisibility_free(InternalCovisibility* cov) {
    if (!cov) return;
    slam_free(cov->adjacency_matrix);
    slam_free(cov->connected_frames);
    slam_free(cov->connected_frame_counts);
    slam_free(cov->essential_graph_edges_from);
    slam_free(cov->essential_graph_edges_to);
    slam_free(cov->essential_graph_weights);
    memset(cov, 0, sizeof(InternalCovisibility));
}

/* ==================== 共视图更新 ==================== */

int slam_covisibility_update(InternalCovisibility* cov, int frame_id,
                             int* landmark_ids, int num_landmarks,
                             const KeyFrame* keyframes, int num_keyframes) {
    if (!cov || !landmark_ids || !keyframes) return -1;
    if (frame_id < 0 || frame_id >= cov->max_frames) return -1;

    if (frame_id >= cov->num_frames) {
        cov->num_frames = frame_id + 1;
    }

    for (int i = 0; i < num_landmarks; i++) {
        int lm_id = landmark_ids[i];
        if (lm_id < 0) continue;

        for (int j = 0; j < num_keyframes; j++) {
            if (j == frame_id) continue;
            if (j >= cov->max_frames) continue;

            KeyFrame* kf = (KeyFrame*)&keyframes[j];
            if (!kf->landmark_ids) continue;

            for (int k = 0; k < kf->num_landmarks; k++) {
                if (kf->landmark_ids[k] == lm_id) {
                    cov->adjacency_matrix[frame_id * cov->max_frames + j] += 1;
                    cov->adjacency_matrix[j * cov->max_frames + frame_id] += 1;
                    break;
                }
            }
        }
    }

    for (int j = 0; j < cov->max_frames; j++) {
        cov->connected_frame_counts[frame_id] = 0;
    }
    int count = 0;
    for (int j = 0; j < cov->max_frames && j < num_keyframes; j++) {
        if (cov->adjacency_matrix[frame_id * cov->max_frames + j] > 0) {
            cov->connected_frames[frame_id * cov->max_frames + count] = j;
            cov->connected_frame_counts[frame_id]++;
            count++;
        }
    }

    return 0;
}

/* ==================== 获取共视关键帧 ==================== */

int slam_covisibility_get_connected(InternalCovisibility* cov, int frame_id,
                                    int* connected_ids, int max_count) {
    if (!cov || !connected_ids) return -1;
    if (frame_id < 0 || frame_id >= cov->num_frames) return -1;

    int count = cov->connected_frame_counts[frame_id];
    if (count > max_count) count = max_count;

    for (int i = 0; i < count; i++) {
        connected_ids[i] = cov->connected_frames[frame_id * cov->max_frames + i];
    }

    return count;
}

/* ==================== 获取共视权重 ==================== */

int slam_covisibility_get_weight(InternalCovisibility* cov, int frame_id1, int frame_id2) {
    if (!cov) return 0;
    if (frame_id1 < 0 || frame_id1 >= cov->num_frames) return 0;
    if (frame_id2 < 0 || frame_id2 >= cov->num_frames) return 0;

    return cov->adjacency_matrix[frame_id1 * cov->max_frames + frame_id2];
}

/* ==================== 构建本质图 ==================== */

int slam_covisibility_build_essential_graph(InternalCovisibility* cov,
                                            const KeyFrame* keyframes,
                                            int num_keyframes) {
    if (!cov || !keyframes || num_keyframes < 1) return -1;

    slam_free(cov->essential_graph_edges_from);
    slam_free(cov->essential_graph_edges_to);
    slam_free(cov->essential_graph_weights);

    int max_edges = cov->max_frames * 10;
    cov->essential_graph_edges_from = (int*)slam_malloc((size_t)max_edges * sizeof(int));
    cov->essential_graph_edges_to = (int*)slam_malloc((size_t)max_edges * sizeof(int));
    cov->essential_graph_weights = (float*)slam_malloc((size_t)max_edges * sizeof(float));

    if (!cov->essential_graph_edges_from || !cov->essential_graph_edges_to ||
        !cov->essential_graph_weights) {
        slam_free(cov->essential_graph_edges_from);
        slam_free(cov->essential_graph_edges_to);
        slam_free(cov->essential_graph_weights);
        return -1;
    }

    int edge_count = 0;
    for (int i = 0; i < cov->num_frames && i < num_keyframes; i++) {
        int num_connected = cov->connected_frame_counts[i];
        for (int j = 0; j < num_connected && j < cov->max_frames; j++) {
            int kf_j = cov->connected_frames[i * cov->max_frames + j];
            if (kf_j <= i) continue;

            int weight = cov->adjacency_matrix[i * cov->max_frames + kf_j];
            if (weight >= SLAM_COVISIBILITY_MIN_WEIGHT && edge_count < max_edges) {
                cov->essential_graph_edges_from[edge_count] = i;
                cov->essential_graph_edges_to[edge_count] = kf_j;
                cov->essential_graph_weights[edge_count] = (float)weight;
                edge_count++;
            }
        }
    }

    cov->num_essential_edges = edge_count;
    cov->essential_graph_built = 1;
    return 0;
}
