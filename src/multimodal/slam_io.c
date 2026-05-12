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
