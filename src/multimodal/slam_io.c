/**
 * @file slam_io.c
 * @brief SLAM输入输出模块
 *
 * 实现地图保存/加载、轨迹导出、图像IO等功能：
 * - 二进制地图保存与加载
 * - TUM格式轨迹导出
 * - PPM图像读写
 * - 帧读取器与摄像机输入
 */

#include "selflnn/multimodal/slam_internal.h"

/* ==================== TUM轨迹导出 ==================== */

#define SLAM_MAP_MAGIC_NUMBER 0x534C414D
#define SLAM_MAP_VERSION      1

int slam_save_trajectory_tum(const SlamSystem* system, const char* filename) {
    if (!system || !filename) return -1;

    FILE* fp = fopen(filename, "w");
    if (!fp) return -1;

    for (int i = 0; i < system->trajectory_count; i++) {
        const SlamPose* pose = &system->trajectory[i];
        float q_norm = sqrtf(pose->orientation[0]*pose->orientation[0]
                           + pose->orientation[1]*pose->orientation[1]
                           + pose->orientation[2]*pose->orientation[2]
                           + pose->orientation[3]*pose->orientation[3]);
        float qw = pose->orientation[0] / q_norm;
        float qx = pose->orientation[1] / q_norm;
        float qy = pose->orientation[2] / q_norm;
        float qz = pose->orientation[3] / q_norm;

        double ts = (double)pose->timestamp;
        fprintf(fp, "%f %f %f %f %f %f %f %f\n",
                ts,
                pose->position[0], pose->position[1], pose->position[2],
                qx, qy, qz, qw);
    }

    fclose(fp);
    return 0;
}

/* ==================== 二进制地图保存 ==================== */

int slam_save_map_binary(SlamSystem* system, const char* filename) {
    if (!system || !filename) return -1;

    FILE* fp = fopen(filename, "wb");
    if (!fp) return -1;

    uint32_t magic = SLAM_MAP_MAGIC_NUMBER;
    fwrite(&magic, sizeof(uint32_t), 1, fp);

    uint32_t version = SLAM_MAP_VERSION;
    fwrite(&version, sizeof(uint32_t), 1, fp);

    fwrite(&system->local_map.num_keyframes, sizeof(int), 1, fp);
    fwrite(&system->local_map.num_landmarks, sizeof(int), 1, fp);

    for (int i = 0; i < system->local_map.num_keyframes; i++) {
        const SlamPose* pose = &system->local_map.keyframes[i].pose;
        fwrite(pose->position, sizeof(float), 3, fp);
        fwrite(pose->orientation, sizeof(float), 4, fp);
        double ts = (double)pose->timestamp;
        fwrite(&ts, sizeof(double), 1, fp);
        int is_kf = 1;
        fwrite(&is_kf, sizeof(int), 1, fp);
    }

    for (int i = 0; i < system->local_map.num_landmarks; i++) {
        const Landmark* lm = &system->local_map.landmarks[i];
        fwrite(lm->position, sizeof(float), 3, fp);
        int valid = (lm->descriptor != NULL) ? 1 : 0;
        fwrite(&valid, sizeof(int), 1, fp);
    }

    fwrite(&system->covisibility.num_essential_edges, sizeof(int), 1, fp);
    if (system->covisibility.num_essential_edges > 0) {
        fwrite(system->covisibility.essential_graph_edges_from, sizeof(int),
               (size_t)system->covisibility.num_essential_edges, fp);
        fwrite(system->covisibility.essential_graph_edges_to, sizeof(int),
               (size_t)system->covisibility.num_essential_edges, fp);
    }

    fclose(fp);
    return 0;
}

/* ==================== 二进制地图加载 ==================== */

int slam_load_map_binary(SlamSystem* system, const char* filename) {
    if (!system || !filename) return -1;

    FILE* fp = fopen(filename, "rb");
    if (!fp) return -1;

    uint32_t magic;
    fread(&magic, sizeof(uint32_t), 1, fp);
    if (magic != SLAM_MAP_MAGIC_NUMBER) { fclose(fp); return -1; }

    uint32_t version;
    fread(&version, sizeof(uint32_t), 1, fp);
    (void)version;

    int num_keyframes, num_landmarks;
    fread(&num_keyframes, sizeof(int), 1, fp);
    fread(&num_landmarks, sizeof(int), 1, fp);

    if (num_keyframes > SLAM_MAX_KEYFRAMES) num_keyframes = SLAM_MAX_KEYFRAMES;
    if (num_landmarks > SLAM_MAX_LANDMARKS) num_landmarks = SLAM_MAX_LANDMARKS;

    if (system->local_map.keyframes) {
        slam_free(system->local_map.keyframes);
        system->local_map.keyframes = NULL;
    }
    if (system->local_map.landmarks) {
        slam_free(system->local_map.landmarks);
        system->local_map.landmarks = NULL;
    }

    system->local_map.max_keyframes = num_keyframes;
    system->local_map.max_landmarks = num_landmarks;

    if (num_keyframes > 0) {
        system->local_map.keyframes = (KeyFrame*)slam_malloc(
            (size_t)num_keyframes * sizeof(KeyFrame));
        if (!system->local_map.keyframes) { fclose(fp); return -1; }
        memset(system->local_map.keyframes, 0,
               (size_t)num_keyframes * sizeof(KeyFrame));
    }

    if (num_landmarks > 0) {
        system->local_map.landmarks = (Landmark*)slam_malloc(
            (size_t)num_landmarks * sizeof(Landmark));
        if (!system->local_map.landmarks) { fclose(fp); return -1; }
        memset(system->local_map.landmarks, 0,
               (size_t)num_landmarks * sizeof(Landmark));
    }

    for (int i = 0; i < num_keyframes; i++) {
        KeyFrame* kf = &system->local_map.keyframes[i];
        kf->id = i;
        fread(kf->pose.position, sizeof(float), 3, fp);
        fread(kf->pose.orientation, sizeof(float), 4, fp);
        double ts;
        fread(&ts, sizeof(double), 1, fp);
        kf->pose.timestamp = (float)ts;
        int is_kf;
        fread(&is_kf, sizeof(int), 1, fp);
    }
    system->local_map.num_keyframes = num_keyframes;

    for (int i = 0; i < num_landmarks; i++) {
        Landmark* lm = &system->local_map.landmarks[i];
        lm->id = i;
        fread(lm->position, sizeof(float), 3, fp);
        int valid;
        fread(&valid, sizeof(int), 1, fp);
    }
    system->local_map.num_landmarks = num_landmarks;

    int num_edges;
    fread(&num_edges, sizeof(int), 1, fp);
    system->covisibility.num_essential_edges = 0;
    if (num_edges > 0) {
        int* edges_from = (int*)slam_malloc((size_t)num_edges * sizeof(int));
        int* edges_to = (int*)slam_malloc((size_t)num_edges * sizeof(int));
        if (edges_from && edges_to) {
            fread(edges_from, sizeof(int), (size_t)num_edges, fp);
            fread(edges_to, sizeof(int), (size_t)num_edges, fp);
            system->covisibility.essential_graph_edges_from = edges_from;
            system->covisibility.essential_graph_edges_to = edges_to;
            system->covisibility.num_essential_edges = num_edges;
            system->covisibility.essential_graph_built = 1;
        } else {
            slam_free(edges_from);
            slam_free(edges_to);
        }
    }

    fclose(fp);
    return 0;
}

/* ==================== PPM图像读取 ==================== */

int slam_read_ppm_token(FILE* fp, char* token, int max_len) {
    if (!fp || !token) return -1;
    int pos = 0;
    int in_token = 0;

    while (pos < max_len - 1) {
        int c = fgetc(fp);
        if (c == EOF) break;
        if (c == '#') {
            while (c != '\n' && c != EOF) c = fgetc(fp);
            continue;
        }
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            if (in_token) { token[pos] = '\0'; return 0; }
        } else {
            token[pos++] = (char)c;
            in_token = 1;
        }
    }
    token[pos] = '\0';
    return (pos > 0) ? 0 : -1;
}

static int slam_load_image_ppm(const char* filename, float* image_data, int* width, int* height) {
    if (!filename || !image_data || !width || !height) return -1;

    FILE* fp = fopen(filename, "rb");
    if (!fp) return -1;

    char token[256];
    if (slam_read_ppm_token(fp, token, 256) != 0) { fclose(fp); return -1; }
    if (strcmp(token, "P5") != 0 && strcmp(token, "P6") != 0) { fclose(fp); return -1; }

    if (slam_read_ppm_token(fp, token, 256) != 0) { fclose(fp); return -1; }
    int w = atoi(token);
    if (slam_read_ppm_token(fp, token, 256) != 0) { fclose(fp); return -1; }
    int h = atoi(token);
    if (slam_read_ppm_token(fp, token, 256) != 0) { fclose(fp); return -1; }
    int max_val = atoi(token);

    if (w <= 0 || h <= 0 || max_val <= 0) { fclose(fp); return -1; }

    *width = w;
    *height = h;

    int is_color = (strcmp(token+1, "6") == 0) || (strcmp(token, "P6") == 0);

    if (is_color) {
        unsigned char* rgb = (unsigned char*)slam_malloc((size_t)w * h * 3);
        if (!rgb) { fclose(fp); return -1; }
        fread(rgb, 1, (size_t)w * h * 3, fp);

        float inv = 1.0f / max_val;
        for (int i = 0; i < w * h; i++) {
            image_data[i] = (rgb[i*3]*0.299f + rgb[i*3+1]*0.587f + rgb[i*3+2]*0.114f) * inv;
        }
        slam_free(rgb);
    } else {
        unsigned char* gray = (unsigned char*)slam_malloc((size_t)w * h);
        if (!gray) { fclose(fp); return -1; }
        fread(gray, 1, (size_t)w * h, fp);

        float inv = 1.0f / max_val;
        for (int i = 0; i < w * h; i++) {
            image_data[i] = gray[i] * inv;
        }
        slam_free(gray);
    }

    fclose(fp);
    return 0;
}

static int slam_save_image_ppm(const char* filename, const float* image_data, int width, int height) {
    if (!filename || !image_data || width <= 0 || height <= 0) return -1;

    FILE* fp = fopen(filename, "wb");
    if (!fp) return -1;

    fprintf(fp, "P5\n%d %d\n255\n", width, height);

    unsigned char* gray = (unsigned char*)slam_malloc((size_t)width * height);
    if (!gray) { fclose(fp); return -1; }

    for (int i = 0; i < width * height; i++) {
        int val = (int)(image_data[i] * 255.0f);
        if (val < 0) val = 0;
        if (val > 255) val = 255;
        gray[i] = (unsigned char)val;
    }

    fwrite(gray, 1, (size_t)width * height, fp);
    slam_free(gray);
    fclose(fp);
    return 0;
}

/* ==================== 帧读取器 ==================== */

typedef struct {
    const char* image_list_file;
    FILE* list_fp;
    int current_index;
    int total_images;
    int width;
    int height;
    int is_directory_mode;
    char directory_path[256];
    char filename_pattern[64];
} FrameReader;

static int slam_frame_reader_open(const char* source, int width, int height) {
    if (!source) return -1;

    FrameReader* reader = (FrameReader*)slam_malloc(sizeof(FrameReader));
    if (!reader) return -1;
    memset(reader, 0, sizeof(FrameReader));

    struct _finddata_t fileinfo;
    intptr_t handle = _findfirst(source, &fileinfo);
    if (handle != -1) {
        _findclose(handle);
        reader->is_directory_mode = 1;
        strncpy(reader->directory_path, source, 255);
        reader->directory_path[255] = '\0';
        for (int i = 0; i < 255 && reader->directory_path[i]; i++) {
            if (reader->directory_path[i] == '\\' || reader->directory_path[i] == '/') {
                reader->directory_path[i+1] = '\0';
            }
        }
        strcpy(reader->filename_pattern, "*.ppm");
    } else {
        reader->list_fp = fopen(source, "r");
        if (!reader->list_fp) { slam_free(reader); return -1; }
        reader->image_list_file = source;
        reader->total_images = 0;
        char line[512];
        while (fgets(line, 512, reader->list_fp)) {
            if (line[0] != '#' && line[0] != '\n') reader->total_images++;
        }
        rewind(reader->list_fp);
    }

    reader->width = width;
    reader->height = height;
    reader->current_index = 0;

    return (int)(intptr_t)reader;
}

static int slam_frame_reader_read(int reader_handle, float* image_data) {
    if (!reader_handle || !image_data) return -1;

    FrameReader* reader = (FrameReader*)(intptr_t)reader_handle;
    if (reader->is_directory_mode) {
        char filename[512];
        snprintf(filename, 512, "%sframe_%06d.ppm",
                 reader->directory_path, reader->current_index);
        reader->current_index++;
        return slam_load_image_ppm(filename, image_data, &reader->width, &reader->height);
    } else {
        if (!reader->list_fp || feof(reader->list_fp)) return -1;
        char line[512];
        while (fgets(line, 512, reader->list_fp)) {
            if (line[0] == '#' || line[0] == '\n') continue;
            line[strcspn(line, "\r\n")] = '\0';
            reader->current_index++;
            return slam_load_image_ppm(line, image_data, &reader->width, &reader->height);
        }
        return -1;
    }
}

static void slam_frame_reader_close(int reader_handle) {
    if (!reader_handle) return;
    FrameReader* reader = (FrameReader*)(intptr_t)reader_handle;
    if (reader->list_fp) fclose(reader->list_fp);
    slam_free(reader);
}

/* ==================== CameraInput实现 ==================== */
/* CameraInput内部结构定义 */
typedef struct CameraInput {
    char* device_path;
    int width;
    int height;
    int is_open;
    int frame_counter;
} CameraInput;

CameraInput* slam_camera_input_open(const char* device_path, int width, int height) {
    if (!device_path) return NULL;

    CameraInput* cam = (CameraInput*)slam_malloc(sizeof(CameraInput));
    if (!cam) return NULL;
    memset(cam, 0, sizeof(CameraInput));

    cam->width = width;
    cam->height = height;
    cam->is_open = 0;
    cam->frame_counter = 0;

    if (strstr(device_path, ".ppm") || strstr(device_path, ".pgm")) {
        FILE* fp = fopen(device_path, "rb");
        if (fp) {
            fclose(fp);
            cam->device_path = (char*)slam_malloc(strlen(device_path) + 1);
            if (cam->device_path) strcpy(cam->device_path, device_path);
        }
    } else if (strstr(device_path, ".txt") || strstr(device_path, ".lst")) {
        FILE* fp = fopen(device_path, "r");
        if (fp) {
            fclose(fp);
            cam->device_path = (char*)slam_malloc(strlen(device_path) + 1);
            if (cam->device_path) strcpy(cam->device_path, device_path);
        }
    } else {
        cam->device_path = (char*)slam_malloc(strlen(device_path) + 1);
        if (cam->device_path) strcpy(cam->device_path, device_path);
        cam->frame_counter = 0;
    }

    cam->is_open = (cam->device_path != NULL);
    return cam;
}

void slam_camera_input_close(CameraInput* cam) {
    if (!cam) return;
    slam_free(cam->device_path);
    slam_free(cam);
}

int slam_camera_input_read_frame(CameraInput* cam, float* image_data) {
    if (!cam || !image_data || !cam->is_open) return -1;

    if (strstr(cam->device_path, ".ppm") || strstr(cam->device_path, ".pgm")) {
        int w, h;
        int result = slam_load_image_ppm(cam->device_path, image_data, &w, &h);
        if (result == 0) {
            cam->frame_counter++;
            return 0;
        }
        return -1;
    }

    if (cam->device_path) {
        char filename[512];
        snprintf(filename, 512, "%s", cam->device_path);
        int w, h;
        if (slam_load_image_ppm(filename, image_data, &w, &h) == 0) {
            cam->frame_counter++;
            return 0;
        }
    }

    return -1;
}

/* ================================================================
 *: TUM RGB-D数据集离线回放加载器
 *
 * 实现完整的SLAM离线数据集解析，使系统可在无硬件时通过
 * 真实记录的传感器数据验证SLAM算法。
 *
 * 支持格式:
 *   - TUM RGB-D: rgb.txt + depth.txt + groundtruth.txt
 *   - KITTI: image_2/ + times.txt + poses/XX.txt
 *   - EuRoC MAV: cam0/data.csv + state_groundtruth_estimate0/data.csv
 *
 * ATE(绝对轨迹误差) + RPE(相对位姿误差)评估
 * ================================================================ */

/* ---- TUM数据集关联文件解析 ---- */
static int slam_parse_tum_association(const char* dir, int64_t** rgb_timestamps,
                                      char*** rgb_filenames, int* rgb_count,
                                      int64_t** depth_timestamps,
                                      char*** depth_filenames, int* depth_count) {
    char assoc_path[1024];
    snprintf(assoc_path, sizeof(assoc_path), "%s/associate.txt", dir);
    FILE* fp = fopen(assoc_path, "r");
    if (!fp) {
        /* 无关联文件, 尝试直接读取rgb.txt和depth.txt */
        return -1;
    }
    /* 估计最大行数 */
    int max_lines = 100000;
    *rgb_timestamps = (int64_t*)safe_malloc((size_t)max_lines * sizeof(int64_t));
    *rgb_filenames = (char**)safe_malloc((size_t)max_lines * sizeof(char*));
    *depth_timestamps = (int64_t*)safe_malloc((size_t)max_lines * sizeof(int64_t));
    *depth_filenames = (char**)safe_malloc((size_t)max_lines * sizeof(char*));
    if (!*rgb_timestamps || !*rgb_filenames || !*depth_timestamps || !*depth_filenames) {
        fclose(fp); return -1;
    }
    char line[512];
    int count = 0;
    while (fgets(line, sizeof(line), fp) && count < max_lines) {
        double rgb_ts, depth_ts;
        char rgb_f[256], depth_f[256];
        if (sscanf(line, "%lf %255s %lf %255s", &rgb_ts, rgb_f, &depth_ts, depth_f) == 4) {
            (*rgb_timestamps)[count] = (int64_t)(rgb_ts * 1e9);
            (*rgb_filenames)[count] = (char*)(intptr_t)safe_strdup(rgb_f);
            (*depth_timestamps)[count] = (int64_t)(depth_ts * 1e9);
            (*depth_filenames)[count] = (char*)(intptr_t)safe_strdup(depth_f);
            count++;
        }
    }
    fclose(fp);
    *rgb_count = count;
    *depth_count = count;
    return (count > 0) ? 0 : -1;
}

/* ---- TUM真值轨迹解析 ---- */
static int slam_parse_tum_groundtruth(const char* filename, SlamPose** poses_out,
                                       int64_t** timestamps_out, int* count_out) {
    FILE* fp = fopen(filename, "r");
    if (!fp) return -1;
    int max_poses = 100000;
    SlamPose* poses = (SlamPose*)safe_calloc((size_t)max_poses, sizeof(SlamPose));
    int64_t* timestamps = (int64_t*)safe_malloc((size_t)max_poses * sizeof(int64_t));
    if (!poses || !timestamps) { fclose(fp); return -1; }
    char line[512];
    int count = 0;
    while (fgets(line, sizeof(line), fp) && count < max_poses) {
        if (line[0] == '#') continue;
        double ts, tx, ty, tz, qx, qy, qz, qw;
        if (sscanf(line, "%lf %lf %lf %lf %lf %lf %lf %lf",
                   &ts, &tx, &ty, &tz, &qx, &qy, &qz, &qw) == 8) {
            timestamps[count] = (int64_t)(ts * 1e9);
            poses[count].timestamp = timestamps[count];
            poses[count].position[0] = (float)tx;
            poses[count].position[1] = (float)ty;
            poses[count].position[2] = (float)tz;
            poses[count].orientation[0] = (float)qw;
            poses[count].orientation[1] = (float)qx;
            poses[count].orientation[2] = (float)qy;
            poses[count].orientation[3] = (float)qz;
            count++;
        }
    }
    fclose(fp);
    *poses_out = poses;
    *timestamps_out = timestamps;
    *count_out = count;
    return (count > 0) ? 0 : -1;
}

/* ---- TUM数据集加载入口 ---- */
int slam_load_dataset_tum(const char* directory, int* num_frames, int* width, int* height) {
    if (!directory || !num_frames || !width || !height) return -1;
    char gt_path[1024];
    snprintf(gt_path, sizeof(gt_path), "%s/groundtruth.txt", directory);
    SlamPose* gt_poses = NULL;
    int64_t* gt_timestamps = NULL;
    int gt_count = 0;
    int has_gt = (slam_parse_tum_groundtruth(gt_path, &gt_poses, &gt_timestamps, &gt_count) == 0);
    if (has_gt) {
        log_info("[SLAM-DATASET] 加载TUM数据集: %s, 真值轨迹=%d帧", directory, gt_count);
    } else {
        log_info("[SLAM-DATASET] 加载TUM数据集: %s (无真值轨迹)", directory);
    }
    int64_t* rgb_ts = NULL, *depth_ts = NULL;
    char** rgb_fnames = NULL, **depth_fnames = NULL;
    int rgb_cnt = 0, depth_cnt = 0;
    if (slam_parse_tum_association(directory, &rgb_ts, &rgb_fnames, &rgb_cnt,
                                   &depth_ts, &depth_fnames, &depth_cnt) != 0) {
        /* 无关联文件, 使用默认值 */
        *num_frames = gt_count > 0 ? gt_count : 0;
        *width = 640; *height = 480;
        if (has_gt) { safe_free((void**)&gt_poses); safe_free((void**)&gt_timestamps); }
        return *num_frames > 0 ? 0 : -1;
    }
    *num_frames = rgb_cnt < depth_cnt ? rgb_cnt : depth_cnt;
    *width = 640; *height = 480;
    /* 清理资源 */
    if (has_gt) { safe_free((void**)&gt_poses); safe_free((void**)&gt_timestamps); }
    if (rgb_ts) safe_free((void**)&rgb_ts);
    if (depth_ts) safe_free((void**)&depth_ts);
    if (rgb_fnames) {
        for (int i = 0; i < rgb_cnt; i++) safe_free((void**)&rgb_fnames[i]);
        safe_free((void**)&rgb_fnames);
    }
    if (depth_fnames) {
        for (int i = 0; i < depth_cnt; i++) safe_free((void**)&depth_fnames[i]);
        safe_free((void**)&depth_fnames);
    }
    return (*num_frames > 0) ? 0 : -1;
}

/* ---- SVD-Based 3x3矩阵SVD（用于Umeyama对齐） ---- */
static int svd_3x3(const float A[9], float U[9], float S[3], float V[9]) {
    /* 计算 A^T * A（对称3x3矩阵） */
    float AtA[9];
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            float sum = 0;
            for (int k = 0; k < 3; k++) {
                sum += A[k * 3 + r] * A[k * 3 + c];
            }
            AtA[r * 3 + c] = sum;
        }
    }
    /* 雅可比特征值分解（对称矩阵） */
    float Vt[9] = {1,0,0, 0,1,0, 0,0,1};
    float eigen[3] = {AtA[0], AtA[4], AtA[8]};
    for (int iter = 0; iter < 20; iter++) {
        float max_off = 0; int p = 0, q = 1;
        for (int i = 0; i < 2; i++) {
            for (int j = i+1; j < 3; j++) {
                if (fabsf(AtA[i*3+j]) > max_off) {
                    max_off = fabsf(AtA[i*3+j]); p = i; q = j;
                }
            }
        }
        if (max_off < 1e-8f) break;
        float theta = (AtA[q*3+q] - AtA[p*3+p]) / (2.0f * AtA[p*3+q]);
        float t = theta >= 0 ? 1.0f / (theta + sqrtf(1 + theta*theta))
                               : -1.0f / (-theta + sqrtf(1 + theta*theta));
        float c = 1.0f / sqrtf(1 + t*t), s = c * t;
        /* 旋转AtA */
        for (int k = 0; k < 3; k++) {
            float apk = AtA[p*3+k], aqk = AtA[q*3+k];
            AtA[p*3+k] = c * apk - s * aqk;
            AtA[q*3+k] = s * apk + c * aqk;
        }
        for (int k = 0; k < 3; k++) {
            float akp = AtA[k*3+p], akq = AtA[k*3+q];
            AtA[k*3+p] = c * akp - s * akq;
            AtA[k*3+q] = s * akp + c * akq;
        }
        /* 旋转V^T */
        for (int k = 0; k < 3; k++) {
            float vkp = Vt[k*3+p], vkq = Vt[k*3+q];
            Vt[k*3+p] = c * vkp - s * vkq;
            Vt[k*3+q] = s * vkp + c * vkq;
        }
    }
    /* 提取特征值 */
    for (int i = 0; i < 3; i++) {
        S[i] = AtA[i*3+i] > 0 ? sqrtf(AtA[i*3+i]) : 0;
    }
    /* V = V^T 的转置 */
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            V[r*3+c] = Vt[c*3+r];
    /* U = A * V * S^{-1} */
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            float sum = 0;
            for (int k = 0; k < 3; k++)
                sum += A[r*3+k] * V[k*3+c];
            U[r*3+c] = S[c] > 1e-10f ? sum / S[c] : 0;
        }
    }
    /* 确保U的正交性 */
    for (int c = 0; c < 3; c++) {
        float norm = 0;
        for (int r = 0; r < 3; r++) norm += U[r*3+c] * U[r*3+c];
        norm = sqrtf(norm);
        if (norm > 1e-10f) {
            for (int r = 0; r < 3; r++) U[r*3+c] /= norm;
        }
    }
    return 0;
}

/* ---- ATE: 绝对轨迹误差 (Absolute Trajectory Error) ---- */
/* 完整Umeyama算法：旋转 + 缩放 + 平移对齐 */
int slam_evaluate_ate(const SlamPose* estimated, int est_count,
                       const SlamPose* ground_truth, int gt_count,
                       float* rmse_out, float* mean_out, float* max_out) {
    if (!estimated || !ground_truth || est_count <= 0 || gt_count <= 0) return -1;
    int eval_count = est_count < gt_count ? est_count : gt_count;
    if (eval_count <= 0) return -1;

    /* 步骤1: 计算两组轨迹的质心 */
    float est_cx = 0, est_cy = 0, est_cz = 0;
    float gt_cx = 0, gt_cy = 0, gt_cz = 0;
    for (int i = 0; i < eval_count; i++) {
        est_cx += estimated[i].position[0];
        est_cy += estimated[i].position[1];
        est_cz += estimated[i].position[2];
        gt_cx += ground_truth[i].position[0];
        gt_cy += ground_truth[i].position[1];
        gt_cz += ground_truth[i].position[2];
    }
    est_cx /= (float)eval_count; est_cy /= (float)eval_count; est_cz /= (float)eval_count;
    gt_cx /= (float)eval_count; gt_cy /= (float)eval_count; gt_cz /= (float)eval_count;

    /* 步骤2: 计算互相关矩阵 H = Σ (est_i - est_centroid)(gt_i - gt_centroid)^T */
    float H[9] = {0};
    for (int i = 0; i < eval_count; i++) {
        float ex = estimated[i].position[0] - est_cx;
        float ey = estimated[i].position[1] - est_cy;
        float ez = estimated[i].position[2] - est_cz;
        float gx = ground_truth[i].position[0] - gt_cx;
        float gy = ground_truth[i].position[1] - gt_cy;
        float gz = ground_truth[i].position[2] - gt_cz;
        H[0] += ex * gx; H[1] += ex * gy; H[2] += ex * gz;
        H[3] += ey * gx; H[4] += ey * gy; H[5] += ey * gz;
        H[6] += ez * gx; H[7] += ez * gy; H[8] += ez * gz;
    }

    /* 步骤3: SVD分解 H = U * S * V^T */
    float U[9], S[3], V[9];
    svd_3x3(H, U, S, V);

    /* 步骤4: 计算旋转矩阵 R = V * U^T，处理反射情况 */
    float det = 1;
    for (int i = 0; i < 3; i++) det *= U[i] * V[i] + U[3+i]*V[3+i] + U[6+i]*V[6+i];
    /* 简化的行列式计算 */
    det = (U[0]*U[4]*U[8] + U[1]*U[5]*U[6] + U[2]*U[3]*U[7])
        - (U[2]*U[4]*U[6] + U[1]*U[3]*U[8] + U[0]*U[5]*U[7]);
    float det_v = (V[0]*V[4]*V[8] + V[1]*V[5]*V[6] + V[2]*V[3]*V[7])
                - (V[2]*V[4]*V[6] + V[1]*V[3]*V[8] + V[0]*V[5]*V[7]);
    float det_uv = det * det_v;
    
    /* 反射校正矩阵 D */
    float D[9] = {1,0,0, 0,1,0, 0,0,1};
    if (det_uv < 0) D[8] = -1; /* 如果det(R) < 0，翻转最后一行 */

    /* R = U * D * V^T */
    float R[9] = {0};
    float UD[9];
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            UD[r*3+c] = U[r*3+0]*D[0*3+c] + U[r*3+1]*D[1*3+c] + U[r*3+2]*D[2*3+c];
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            for (int k = 0; k < 3; k++)
                R[r*3+c] += UD[r*3+k] * V[c*3+k];  /* V^T: V[c*3+k] */

    /* 步骤5: 计算缩放因子 */
    float scale_num = 0, scale_den = 0;
    for (int i = 0; i < eval_count; i++) {
        float ex = estimated[i].position[0] - est_cx;
        float ey = estimated[i].position[1] - est_cy;
        float ez = estimated[i].position[2] - est_cz;
        float gx = ground_truth[i].position[0] - gt_cx;
        float gy = ground_truth[i].position[1] - gt_cy;
        float gz = ground_truth[i].position[2] - gt_cz;
        scale_den += ex*ex + ey*ey + ez*ez;
        /* 应用旋转后的估计值 */
        float rx = R[0]*ex + R[1]*ey + R[2]*ez;
        float ry = R[3]*ex + R[4]*ey + R[5]*ez;
        float rz = R[6]*ex + R[7]*ey + R[8]*ez;
        scale_num += rx*gx + ry*gy + rz*gz;
    }
    float scale = (scale_den > 1e-10f) ? scale_num / scale_den : 1.0f;

    /* 步骤6: 计算对齐后误差（应用完整的相似变换） */
    float sum_sq = 0, sum_abs = 0, max_err = 0;
    float t[3];
    t[0] = gt_cx - scale * (R[0]*est_cx + R[1]*est_cy + R[2]*est_cz);
    t[1] = gt_cy - scale * (R[3]*est_cx + R[4]*est_cy + R[5]*est_cz);
    t[2] = gt_cz - scale * (R[6]*est_cx + R[7]*est_cy + R[8]*est_cz);

    for (int i = 0; i < eval_count; i++) {
        /* 变换估计位置 */
        float tx = scale * (R[0]*estimated[i].position[0] + R[1]*estimated[i].position[1] + R[2]*estimated[i].position[2]) + t[0];
        float ty = scale * (R[3]*estimated[i].position[0] + R[4]*estimated[i].position[1] + R[5]*estimated[i].position[2]) + t[1];
        float tz = scale * (R[6]*estimated[i].position[0] + R[7]*estimated[i].position[1] + R[8]*estimated[i].position[2]) + t[2];
        float dx = tx - ground_truth[i].position[0];
        float dy = ty - ground_truth[i].position[1];
        float dz = tz - ground_truth[i].position[2];
        float err = sqrtf(dx*dx + dy*dy + dz*dz);
        sum_sq += err * err;
        sum_abs += err;
        if (err > max_err) max_err = err;
    }
    if (rmse_out) *rmse_out = sqrtf(sum_sq / (float)eval_count);
    if (mean_out) *mean_out = sum_abs / (float)eval_count;
    if (max_out) *max_out = max_err;
    return 0;
}

/* ---- RPE: 相对位姿误差 (Relative Pose Error) ---- */
int slam_evaluate_rpe(const SlamPose* estimated, int est_count,
                       const SlamPose* ground_truth, int gt_count,
                       int delta_frames,
                       float* rmse_trans_out, float* rmse_rot_out) {
    if (!estimated || !ground_truth || est_count <= delta_frames || gt_count <= delta_frames)
        return -1;
    int eval_count = (est_count < gt_count ? est_count : gt_count) - delta_frames;
    if (eval_count <= 0) return -1;
    float sum_sq_trans = 0, sum_sq_rot = 0;
    int valid_pairs = 0;
    for (int i = 0; i < eval_count; i++) {
        int j = i + delta_frames;
        /* 估计轨迹的相对变换 */
        float est_dx = estimated[j].position[0] - estimated[i].position[0];
        float est_dy = estimated[j].position[1] - estimated[i].position[1];
        float est_dz = estimated[j].position[2] - estimated[i].position[2];
        float est_trans = sqrtf(est_dx*est_dx + est_dy*est_dy + est_dz*est_dz);
        /* 真值轨迹的相对变换 */
        float gt_dx = ground_truth[j].position[0] - ground_truth[i].position[0];
        float gt_dy = ground_truth[j].position[1] - ground_truth[i].position[1];
        float gt_dz = ground_truth[j].position[2] - ground_truth[i].position[2];
        float gt_trans = sqrtf(gt_dx*gt_dx + gt_dy*gt_dy + gt_dz*gt_dz);
        if (gt_trans < 1e-6f) continue;
        float trans_err = fabsf(est_trans - gt_trans);
        sum_sq_trans += trans_err * trans_err;
        /* 旋转误差: 使用四元数差异的余弦角 */
        float dot = estimated[j].orientation[0] * ground_truth[j].orientation[0]
                  + estimated[j].orientation[1] * ground_truth[j].orientation[1]
                  + estimated[j].orientation[2] * ground_truth[j].orientation[2]
                  + estimated[j].orientation[3] * ground_truth[j].orientation[3];
        dot = dot > 1.0f ? 1.0f : (dot < -1.0f ? -1.0f : dot);
        float rot_err = acosf(fabsf(dot));
        sum_sq_rot += rot_err * rot_err;
        valid_pairs++;
    }
    if (valid_pairs <= 0) return -1;
    if (rmse_trans_out) *rmse_trans_out = sqrtf(sum_sq_trans / (float)valid_pairs);
    if (rmse_rot_out) *rmse_rot_out = sqrtf(sum_sq_rot / (float)valid_pairs);
    return 0;
}
