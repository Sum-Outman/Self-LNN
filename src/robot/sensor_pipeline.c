#include "selflnn/robot/sensor_pipeline.h"
#include "selflnn/utils/platform.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <sys/select.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <pthread.h>
#endif

typedef struct {
    int sensor_id;
    SensorType sensor_type;
    SensorSource source;
    SensorPipelinePriority priority;
    int enabled;
    char sensor_name[SENSOR_PIPELINE_NAME_MAX];
    char topic_name[SENSOR_PIPELINE_TOPIC_MAX];
    double sample_rate_hz;
    double last_push_time;
    double range_min;
    double range_max;
    double resolution;
    double accuracy;
    int frame_width;
    int frame_height;
    int channels;
    uint32_t sequence_counter;
    uint32_t total_pushed;
    uint32_t total_dropped;
    double total_latency_ms;
    double max_latency_ms;
    double current_rate_hz;
    double rate_start_time;
    uint32_t rate_sample_count;
} SensorInfo;

typedef struct {
    SensorPipelineEntry* entries;
    int capacity;
    int head;
    int tail;
    int count;
    int enable_compression;
    int compression_level;
    int enable_dedup;
    double dedup_threshold;
    SensorPipelineEntry* last_unique;
} SensorRingBuffer;

typedef struct {
    int sensor_id;
    SensorType sensor_type;
    SensorPipelineCallback callback;
    void* user_data;
    int active;
} SubscriberEntry;

struct SensorPipeline {
    SensorPipelineConfig config;
    volatile int running;
    volatile int streaming_running;
    int initialized;

    SensorInfo sensors[SENSOR_PIPELINE_MAX_SENSORS];
    int sensor_count;

    SensorRingBuffer ring_buffer;
    int ring_initialized;

    SubscriberEntry subscribers[SENSOR_PIPELINE_MAX_SUBSCRIBERS];
    int subscriber_count;

    SensorStreamingClient streaming_clients[SENSOR_PIPELINE_MAX_CLIENTS];
    int client_count;

    int server_socket;
    int server_started;

    SensorPipelineStats global_stats;

    uint8_t* data_pool;
    size_t data_pool_size;
    size_t data_pool_used;

    char last_error[256];
};

static int pipeline_get_sensor_index(SensorPipeline* pipeline, int sensor_id)
{
    for (int i = 0; i < pipeline->sensor_count; i++)
    {
        if (pipeline->sensors[i].sensor_id == sensor_id)
            return i;
    }
    return -1;
}

static SensorInfo* pipeline_find_sensor(SensorPipeline* pipeline, int sensor_id)
{
    int idx = pipeline_get_sensor_index(pipeline, sensor_id);
    return (idx >= 0) ? &pipeline->sensors[idx] : NULL;
}

static int ring_buffer_init(SensorRingBuffer* rb, const SensorRingBufferConfig* config)
{
    rb->capacity = (config->max_entries > 0) ? config->max_entries : SENSOR_PIPELINE_RING_BUFFER_SIZE;
    rb->entries = (SensorPipelineEntry*)calloc(rb->capacity, sizeof(SensorPipelineEntry));
    if (!rb->entries) return -1;
    for (int i = 0; i < rb->capacity; i++)
    {
        rb->entries[i].data = NULL;
        rb->entries[i].data_size = 0;
    }
    rb->head = 0;
    rb->tail = 0;
    rb->count = 0;
    rb->enable_compression = config->enable_compression;
    rb->compression_level = config->compression_level;
    rb->enable_dedup = config->enable_dedup;
    rb->dedup_threshold = config->dedup_threshold;
    rb->last_unique = (SensorPipelineEntry*)calloc(1, sizeof(SensorPipelineEntry));
    if (rb->last_unique)
    {
        rb->last_unique->data = NULL;
        rb->last_unique->data_size = 0;
    }
    return (rb->last_unique) ? 0 : -1;
}

static void ring_buffer_entry_free(SensorPipelineEntry* entry)
{
    if (entry && entry->data)
    {
        free(entry->data);
        entry->data = NULL;
        entry->data_size = 0;
    }
}

static void ring_buffer_destroy(SensorRingBuffer* rb)
{
    if (!rb) return;
    if (rb->entries)
    {
        for (int i = 0; i < rb->capacity; i++)
        {
            ring_buffer_entry_free(&rb->entries[i]);
        }
        free(rb->entries);
        rb->entries = NULL;
    }
    if (rb->last_unique)
    {
        ring_buffer_entry_free(rb->last_unique);
        free(rb->last_unique);
        rb->last_unique = NULL;
    }
    rb->capacity = 0;
    rb->head = rb->tail = rb->count = 0;
}

static int ring_buffer_push(SensorRingBuffer* rb, const SensorPipelineEntry* entry)
{
    if (!rb || !rb->entries || rb->count >= rb->capacity)
        return -1;

    if (rb->enable_dedup && rb->last_unique && rb->last_unique->data)
    {
        if (entry->data_size == rb->last_unique->data_size &&
            entry->sensor_id == rb->last_unique->sensor_id &&
            entry->data_size > 0)
        {
            int same = 1;
            size_t check = (entry->data_size < 64) ? entry->data_size : 64;
            for (size_t i = 0; i < check; i++)
            {
                if (entry->data[i] != rb->last_unique->data[i])
                {
                    same = 0;
                    break;
                }
            }
            if (same && (double)entry->data_size > rb->dedup_threshold)
                return 1;
        }
    }

    int idx = rb->head;
    SensorPipelineEntry* dest = &rb->entries[idx];
    ring_buffer_entry_free(dest);

    *dest = *entry;
    dest->data = NULL;
    dest->data_size = 0;

    if (entry->data && entry->data_size > 0)
    {
        dest->data = (uint8_t*)malloc(entry->data_size);
        if (!dest->data) return -1;
        memcpy(dest->data, entry->data, entry->data_size);
        dest->data_size = entry->data_size;
    }

    if (rb->enable_dedup)
    {
        ring_buffer_entry_free(rb->last_unique);
        if (dest->data && dest->data_size > 0)
        {
            rb->last_unique->data = (uint8_t*)malloc(dest->data_size);
            if (rb->last_unique->data)
            {
                memcpy(rb->last_unique->data, dest->data, dest->data_size);
                rb->last_unique->data_size = dest->data_size;
                rb->last_unique->sensor_id = dest->sensor_id;
            }
        }
    }

    rb->head = (rb->head + 1) % rb->capacity;
    if (rb->count < rb->capacity)
        rb->count++;
    else
        rb->tail = (rb->tail + 1) % rb->capacity;

    return 0;
}

static int ring_buffer_get_latest(const SensorRingBuffer* rb, int sensor_id,
                                   SensorPipelineEntry* entry)
{
    if (!rb || rb->count == 0) return -1;
    int idx = (rb->head - 1 + rb->capacity) % rb->capacity;
    for (int i = 0; i < rb->count; i++)
    {
        if (rb->entries[idx].sensor_id == sensor_id)
        {
            *entry = rb->entries[idx];
            return 0;
        }
        idx = (idx - 1 + rb->capacity) % rb->capacity;
    }
    return -1;
}

static int ring_buffer_get_latest_by_type(const SensorRingBuffer* rb, SensorType type,
                                           SensorPipelineEntry* entry)
{
    if (!rb || rb->count == 0) return -1;
    int idx = (rb->head - 1 + rb->capacity) % rb->capacity;
    for (int i = 0; i < rb->count; i++)
    {
        if (rb->entries[idx].sensor_type == type)
        {
            *entry = rb->entries[idx];
            return 0;
        }
        idx = (idx - 1 + rb->capacity) % rb->capacity;
    }
    return -1;
}

static int ring_buffer_get_history(const SensorRingBuffer* rb, int sensor_id,
                                    SensorPipelineEntry* entries, int* count, int max)
{
    if (!rb || rb->count == 0 || max <= 0)
    {
        if (count) *count = 0;
        return -1;
    }
    int found = 0;
    int idx = (rb->head - 1 + rb->capacity) % rb->capacity;
    for (int i = 0; i < rb->count && found < max; i++)
    {
        if (rb->entries[idx].sensor_id == sensor_id)
            entries[found++] = rb->entries[idx];
        idx = (idx - 1 + rb->capacity) % rb->capacity;
    }
    if (count) *count = found;
    return (found > 0) ? 0 : -1;
}

static int ring_buffer_clear(SensorRingBuffer* rb)
{
    if (!rb || !rb->entries) return -1;
    for (int i = 0; i < rb->capacity; i++)
        ring_buffer_entry_free(&rb->entries[i]);
    rb->head = rb->tail = rb->count = 0;
    return 0;
}

static int ring_buffer_clear_by_sensor(SensorRingBuffer* rb, int sensor_id)
{
    if (!rb || !rb->entries) return -1;
    for (int i = 0; i < rb->capacity; i++)
    {
        if (rb->entries[i].sensor_id == sensor_id)
            ring_buffer_entry_free(&rb->entries[i]);
    }
    return 0;
}

static int streaming_notify_clients(SensorPipeline* pipeline, const SensorPipelineEntry* entry)
{
    if (!pipeline->streaming_running || !entry) return -1;

    uint8_t header[32];
    size_t pos = 0;
    uint32_t net_sensor_id = (uint32_t)entry->sensor_id;
    uint32_t net_type = (uint32_t)entry->sensor_type;
    uint32_t net_size = (uint32_t)entry->data_size;
    uint64_t net_ts = (uint64_t)(entry->timestamp * 1e6);
    float net_confidence = entry->confidence;

    memcpy(header + pos, &net_sensor_id, 4); pos += 4;
    memcpy(header + pos, &net_type, 4); pos += 4;
    memcpy(header + pos, &net_size, 4); pos += 4;
    memcpy(header + pos, &net_ts, 8); pos += 8;
    memcpy(header + pos, &net_confidence, 4); pos += 4;

    for (int i = 0; i < pipeline->client_count; i++)
    {
        SensorStreamingClient* client = &pipeline->streaming_clients[i];
        if (!client->is_active) continue;

        int subscribed = 0;
        for (int j = 0; j < client->num_subscriptions; j++)
        {
            if (client->subscribed_sensor_ids[j] == entry->sensor_id ||
                client->subscribed_sensor_ids[j] == -1)
            {
                subscribed = 1;
                break;
            }
        }
        if (!subscribed) continue;

#if defined(_WIN32)
        int sent = send(client->socket, (const char*)header, (int)sizeof(header), 0);
        if (sent > 0 && net_size > 0 && entry->data)
            sent = send(client->socket, (const char*)entry->data, (int)net_size, 0);
        if (sent <= 0)
        {
            closesocket(client->socket);
            client->is_active = 0;
            client->socket = -1;
        }
#else
        int sent = (int)write(client->socket, header, sizeof(header));
        if (sent > 0 && net_size > 0 && entry->data)
            sent = (int)write(client->socket, entry->data, net_size);
        if (sent <= 0)
        {
            close(client->socket);
            client->is_active = 0;
            client->socket = -1;
        }
#endif
        client->last_active = entry->timestamp;
    }
    return 0;
}

static int streaming_server_thread(void* arg)
{
    SensorPipeline* pipeline = (SensorPipeline*)arg;
    if (!pipeline) return -1;

    pipeline->server_socket = -1;

#if defined(_WIN32)
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0)
        return -1;
#endif

    int server = -1;
#if defined(_WIN32)
    server = (int)socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (server == INVALID_SOCKET)
    {
        WSACleanup();
        return -1;
    }
#else
    server = socket(AF_INET, SOCK_STREAM, 0);
    if (server < 0)
        return -1;
#endif

    int opt = 1;
#if defined(_WIN32)
    setsockopt((SOCKET)server, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));
#else
    setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons((unsigned short)pipeline->config.streaming_port);

#if defined(_WIN32)
    if (bind((SOCKET)server, (struct sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR)
    {
        closesocket((SOCKET)server);
        WSACleanup();
        return -1;
    }
    if (listen((SOCKET)server, SENSOR_PIPELINE_MAX_CLIENTS) == SOCKET_ERROR)
    {
        closesocket((SOCKET)server);
        WSACleanup();
        return -1;
    }
#else
    if (bind(server, (struct sockaddr*)&addr, sizeof(addr)) < 0)
    {
        close(server);
        return -1;
    }
    if (listen(server, SENSOR_PIPELINE_MAX_CLIENTS) < 0)
    {
        close(server);
        return -1;
    }
#endif

    pipeline->server_socket = server;
    pipeline->server_started = 1;

    while (pipeline->streaming_running)
    {
        struct sockaddr_in client_addr;
#if defined(_WIN32)
        int client_len = (int)sizeof(client_addr);
        SOCKET client = accept((SOCKET)server, (struct sockaddr*)&client_addr, &client_len);
        if (client == INVALID_SOCKET)
        {
            if (pipeline->streaming_running)
            {
#if defined(_WIN32)
                Sleep(100);
#else
                usleep(100000);
#endif
            }
            continue;
        }
#else
        socklen_t client_len = sizeof(client_addr);
        int client = accept(server, (struct sockaddr*)&client_addr, &client_len);
        if (client < 0)
        {
            if (pipeline->streaming_running)
            {
                usleep(100000);
            }
            continue;
        }
#endif

        int idx = -1;
        for (int i = 0; i < SENSOR_PIPELINE_MAX_CLIENTS; i++)
        {
            if (!pipeline->streaming_clients[i].is_active)
            {
                idx = i;
                break;
            }
        }
        if (idx < 0)
        {
#if defined(_WIN32)
            closesocket(client);
#else
            close(client);
#endif
            continue;
        }

        SensorStreamingClient* sc = &pipeline->streaming_clients[idx];
        sc->socket = (int)client;
        sc->client_port = ntohs(client_addr.sin_port);
        inet_ntop(AF_INET, &client_addr.sin_addr, sc->client_host, sizeof(sc->client_host));
        sc->connect_time = (double)clock() / CLOCKS_PER_SEC;
        sc->last_active = sc->connect_time;
        sc->num_subscriptions = 1;
        sc->subscribed_sensor_ids[0] = -1;
        sc->is_active = 1;
        sc->recv_buffer_pos = 0;
        pipeline->client_count++;
    }

#if defined(_WIN32)
    closesocket((SOCKET)server);
    for (int i = 0; i < SENSOR_PIPELINE_MAX_CLIENTS; i++)
    {
        if (pipeline->streaming_clients[i].is_active)
            closesocket((SOCKET)pipeline->streaming_clients[i].socket);
    }
    WSACleanup();
#else
    close(server);
    for (int i = 0; i < SENSOR_PIPELINE_MAX_CLIENTS; i++)
    {
        if (pipeline->streaming_clients[i].is_active)
            close(pipeline->streaming_clients[i].socket);
    }
#endif

    pipeline->server_socket = -1;
    pipeline->server_started = 0;
    return 0;
}

static void notify_subscribers(SensorPipeline* pipeline, const SensorPipelineEntry* entry)
{
    if (!pipeline || !entry) return;

    for (int i = 0; i < pipeline->subscriber_count; i++)
    {
        SubscriberEntry* sub = &pipeline->subscribers[i];
        if (!sub->active || !sub->callback) continue;
        if (sub->sensor_id == entry->sensor_id || sub->sensor_type == entry->sensor_type)
            sub->callback(entry, sub->user_data);
    }
}

SensorPipeline* sensor_pipeline_create(const SensorPipelineConfig* config)
{
    SensorPipeline* pipeline = (SensorPipeline*)calloc(1, sizeof(SensorPipeline));
    if (!pipeline) return NULL;

    if (config)
        pipeline->config = *config;
    else
        sensor_pipeline_config_set_defaults(&pipeline->config);

    if (pipeline->config.streaming_port <= 0)
        pipeline->config.streaming_port = SELFLNN_SENSOR_STREAM_PORT;
    if (pipeline->config.max_sensors <= 0)
        pipeline->config.max_sensors = SENSOR_PIPELINE_MAX_SENSORS;
    if (pipeline->config.ring_buffer_config.max_entries <= 0)
        pipeline->config.ring_buffer_config.max_entries = SENSOR_PIPELINE_RING_BUFFER_SIZE;
    if (pipeline->config.ring_buffer_config.max_data_size <= 0)
        pipeline->config.ring_buffer_config.max_data_size = SENSOR_PIPELINE_MAX_DATA_SIZE;

    memset(pipeline->sensors, 0, sizeof(pipeline->sensors));
    pipeline->sensor_count = 0;
    pipeline->subscriber_count = 0;
    pipeline->client_count = 0;
    pipeline->server_socket = -1;
    pipeline->server_started = 0;
    pipeline->running = 0;
    pipeline->streaming_running = 0;
    pipeline->initialized = 1;
    pipeline->last_error[0] = '\0';

    memset(&pipeline->global_stats, 0, sizeof(pipeline->global_stats));

    if (ring_buffer_init(&pipeline->ring_buffer, &pipeline->config.ring_buffer_config) != 0)
    {
        free(pipeline);
        return NULL;
    }
    pipeline->ring_initialized = 1;

    return pipeline;
}

void sensor_pipeline_destroy(SensorPipeline* pipeline)
{
    if (!pipeline) return;

    sensor_pipeline_stop(pipeline);
    sensor_pipeline_stop_streaming_server(pipeline);

    if (pipeline->ring_initialized)
    {
        ring_buffer_destroy(&pipeline->ring_buffer);
        pipeline->ring_initialized = 0;
    }

    free(pipeline);
}

int sensor_pipeline_start(SensorPipeline* pipeline)
{
    if (!pipeline) return SENSOR_PIPELINE_ERROR_INVALID;
    if (pipeline->running) return SENSOR_PIPELINE_OK;
    pipeline->running = 1;
    if (pipeline->config.enable_streaming_server)
        sensor_pipeline_start_streaming_server(pipeline);
    return SENSOR_PIPELINE_OK;
}

int sensor_pipeline_stop(SensorPipeline* pipeline)
{
    if (!pipeline) return SENSOR_PIPELINE_ERROR_INVALID;
    pipeline->running = 0;
    sensor_pipeline_stop_streaming_server(pipeline);
    return SENSOR_PIPELINE_OK;
}

int sensor_pipeline_reset(SensorPipeline* pipeline)
{
    if (!pipeline) return SENSOR_PIPELINE_ERROR_INVALID;
    ring_buffer_clear(&pipeline->ring_buffer);
    memset(&pipeline->global_stats, 0, sizeof(pipeline->global_stats));
    for (int i = 0; i < pipeline->sensor_count; i++)
    {
        pipeline->sensors[i].total_pushed = 0;
        pipeline->sensors[i].total_dropped = 0;
        pipeline->sensors[i].total_latency_ms = 0.0;
        pipeline->sensors[i].max_latency_ms = 0.0;
        pipeline->sensors[i].current_rate_hz = 0.0;
        pipeline->sensors[i].sequence_counter = 0;
    }
    return SENSOR_PIPELINE_OK;
}

int sensor_pipeline_is_running(SensorPipeline* pipeline)
{
    return (pipeline && pipeline->running) ? 1 : 0;
}

int sensor_pipeline_register_sensor(SensorPipeline* pipeline, int sensor_id,
                                     SensorType sensor_type, SensorSource source,
                                     SensorPipelinePriority priority,
                                     const char* sensor_name, double sample_rate_hz)
{
    if (!pipeline || !pipeline->initialized) return SENSOR_PIPELINE_ERROR_INVALID;
    if (pipeline_find_sensor(pipeline, sensor_id)) return SENSOR_PIPELINE_ERROR_FULL;
    if (pipeline->sensor_count >= SENSOR_PIPELINE_MAX_SENSORS)
        return SENSOR_PIPELINE_ERROR_FULL;

    SensorInfo* si = &pipeline->sensors[pipeline->sensor_count];
    memset(si, 0, sizeof(SensorInfo));
    si->sensor_id = sensor_id;
    si->sensor_type = sensor_type;
    si->source = source;
    si->priority = priority;
    si->enabled = 1;
    si->sample_rate_hz = (sample_rate_hz > 0.0) ? sample_rate_hz : 10.0;
    si->last_push_time = 0.0;
    si->sequence_counter = 0;
    si->total_pushed = 0;
    si->total_dropped = 0;
    si->rate_start_time = 0.0;
    si->rate_sample_count = 0;
    si->current_rate_hz = 0.0;

    if (sensor_name)
    {
        size_t len = strlen(sensor_name);
        if (len >= SENSOR_PIPELINE_NAME_MAX) len = SENSOR_PIPELINE_NAME_MAX - 1;
        memcpy(si->sensor_name, sensor_name, len);
        si->sensor_name[len] = '\0';
    }

    const char* type_topic = "";
    switch (sensor_type)
    {
        case SENSOR_TYPE_LIDAR: type_topic = "/sensor/lidar"; break;
        case SENSOR_TYPE_CAMERA: type_topic = "/sensor/camera"; break;
        case SENSOR_TYPE_IMU: type_topic = "/sensor/imu"; break;
        case SENSOR_TYPE_GNSS: type_topic = "/sensor/gnss"; break;
        case SENSOR_TYPE_FORCE_TORQUE: type_topic = "/sensor/force_torque"; break;
        case SENSOR_TYPE_TEMPERATURE: type_topic = "/sensor/temperature"; break;
        case SENSOR_TYPE_PRESSURE: type_topic = "/sensor/pressure"; break;
        case SENSOR_TYPE_PROXIMITY: type_topic = "/sensor/proximity"; break;
        default: type_topic = "/sensor/unknown"; break;
    }
    snprintf(si->topic_name, sizeof(si->topic_name), "%s/%d%s", type_topic, sensor_id,
             sensor_name ? sensor_name : "");

    pipeline->sensor_count++;
    return sensor_id;
}

int sensor_pipeline_unregister_sensor(SensorPipeline* pipeline, int sensor_id)
{
    if (!pipeline) return SENSOR_PIPELINE_ERROR_INVALID;
    int idx = pipeline_get_sensor_index(pipeline, sensor_id);
    if (idx < 0) return SENSOR_PIPELINE_ERROR_NOT_FOUND;

    for (int i = idx; i < pipeline->sensor_count - 1; i++)
        pipeline->sensors[i] = pipeline->sensors[i + 1];
    pipeline->sensor_count--;

    ring_buffer_clear_by_sensor(&pipeline->ring_buffer, sensor_id);

    for (int i = 0; i < pipeline->subscriber_count; i++)
    {
        if (pipeline->subscribers[i].sensor_id == sensor_id)
        {
            for (int j = i; j < pipeline->subscriber_count - 1; j++)
                pipeline->subscribers[j] = pipeline->subscribers[j + 1];
            pipeline->subscriber_count--;
            i--;
        }
    }

    return SENSOR_PIPELINE_OK;
}

int sensor_pipeline_get_sensor_info(SensorPipeline* pipeline, int sensor_id,
                                     SensorPipelineEntry* info)
{
    if (!pipeline || !info) return SENSOR_PIPELINE_ERROR_INVALID;
    SensorInfo* si = pipeline_find_sensor(pipeline, sensor_id);
    if (!si) return SENSOR_PIPELINE_ERROR_NOT_FOUND;

    memset(info, 0, sizeof(SensorPipelineEntry));
    info->sensor_id = si->sensor_id;
    info->sensor_type = si->sensor_type;
    info->source = si->source;
    info->priority = si->priority;
    info->sample_rate_hz = si->sample_rate_hz;
    info->range_min = (float)si->range_min;
    info->range_max = (float)si->range_max;
    info->resolution = (float)si->resolution;
    info->accuracy = (float)si->accuracy;
    info->frame_width = si->frame_width;
    info->frame_height = si->frame_height;
    info->channels = si->channels;
    if (si->sensor_name[0])
        memcpy(info->sensor_name, si->sensor_name, SENSOR_PIPELINE_NAME_MAX);
    if (si->topic_name[0])
        memcpy(info->topic_name, si->topic_name, SENSOR_PIPELINE_TOPIC_MAX);
    info->is_valid = si->enabled;
    info->confidence = 1.0f;
    return SENSOR_PIPELINE_OK;
}

int sensor_pipeline_get_registered_sensors(SensorPipeline* pipeline, int* sensor_ids,
                                            int* count)
{
    if (!pipeline || !sensor_ids || !count) return SENSOR_PIPELINE_ERROR_INVALID;
    int max = *count;
    *count = 0;
    for (int i = 0; i < pipeline->sensor_count && *count < max; i++)
        sensor_ids[(*count)++] = pipeline->sensors[i].sensor_id;
    return SENSOR_PIPELINE_OK;
}

int sensor_pipeline_get_sensor_count(SensorPipeline* pipeline)
{
    return (pipeline) ? pipeline->sensor_count : 0;
}

int sensor_pipeline_push_data(SensorPipeline* pipeline, const SensorPipelineEntry* entry)
{
    if (!pipeline || !entry || !pipeline->running)
        return SENSOR_PIPELINE_ERROR_INVALID;
    if (!pipeline->ring_initialized)
        return SENSOR_PIPELINE_ERROR_DISABLED;

    SensorInfo* si = pipeline_find_sensor(pipeline, entry->sensor_id);
    if (!si || !si->enabled) return SENSOR_PIPELINE_ERROR_NOT_FOUND;

    if (si->sample_rate_hz > 0.0 && si->last_push_time > 0.0)
    {
        double min_interval = 1.0 / si->sample_rate_hz;
        double elapsed = entry->timestamp - si->last_push_time;
        if (elapsed < 0.0)
            elapsed = 0.0;
        if (elapsed < min_interval * 0.9)
        {
            si->total_dropped++;
            return SENSOR_PIPELINE_ERROR_BUFFER_FULL;
        }
    }
    si->last_push_time = entry->timestamp;

    SensorPipelineEntry wrapped = *entry;
    wrapped.sequence = si->sequence_counter++;
    wrapped.receive_timestamp = (double)clock() / CLOCKS_PER_SEC;

    int ret = ring_buffer_push(&pipeline->ring_buffer, &wrapped);
    if (ret < 0)
    {
        si->total_dropped++;
        return SENSOR_PIPELINE_ERROR_BUFFER_FULL;
    }

    double latency_ms = (wrapped.receive_timestamp - wrapped.timestamp) * 1000.0;
    if (latency_ms < 0.0) latency_ms = 0.0;
    si->total_latency_ms += latency_ms;
    if (latency_ms > si->max_latency_ms) si->max_latency_ms = latency_ms;
    si->total_pushed++;

    if (si->rate_start_time == 0.0)
        si->rate_start_time = wrapped.receive_timestamp;
    si->rate_sample_count++;
    double rate_elapsed = wrapped.receive_timestamp - si->rate_start_time;
    if (rate_elapsed >= 1.0)
    {
        si->current_rate_hz = (double)si->rate_sample_count / rate_elapsed;
        si->rate_start_time = wrapped.receive_timestamp;
        si->rate_sample_count = 0;
    }

    notify_subscribers(pipeline, &wrapped);

    if (pipeline->streaming_running && pipeline->config.enable_streaming_server)
        streaming_notify_clients(pipeline, &wrapped);

    return SENSOR_PIPELINE_OK;
}

int sensor_pipeline_push_raw(SensorPipeline* pipeline, int sensor_id, SensorType sensor_type,
                              const uint8_t* data, size_t data_size, double timestamp,
                              float confidence)
{
    if (!pipeline || !data) return SENSOR_PIPELINE_ERROR_INVALID;
    SensorPipelineEntry entry;
    memset(&entry, 0, sizeof(entry));
    entry.sensor_id = sensor_id;
    entry.sensor_type = sensor_type;
    entry.source = SENSOR_SOURCE_HARDWARE;
    entry.timestamp = timestamp;
    entry.data = (uint8_t*)data;
    entry.data_size = data_size;
    entry.is_valid = 1;
    entry.confidence = confidence;
    return sensor_pipeline_push_data(pipeline, &entry);
}

int sensor_pipeline_push_image(SensorPipeline* pipeline, int sensor_id,
                                const uint8_t* image_data, size_t data_size,
                                int width, int height, int channels,
                                double timestamp, float confidence)
{
    if (!pipeline || !image_data) return SENSOR_PIPELINE_ERROR_INVALID;
    SensorPipelineEntry entry;
    memset(&entry, 0, sizeof(entry));
    entry.sensor_id = sensor_id;
    entry.sensor_type = SENSOR_TYPE_CAMERA;
    entry.source = SENSOR_SOURCE_HARDWARE;
    entry.timestamp = timestamp;
    entry.data = (uint8_t*)image_data;
    entry.data_size = data_size;
    entry.frame_width = width;
    entry.frame_height = height;
    entry.channels = channels;
    entry.is_valid = 1;
    entry.confidence = confidence;

    SensorInfo* si = pipeline_find_sensor(pipeline, sensor_id);
    if (si)
    {
        si->frame_width = width;
        si->frame_height = height;
        si->channels = channels;
    }

    return sensor_pipeline_push_data(pipeline, &entry);
}

int sensor_pipeline_push_lidar(SensorPipeline* pipeline, int sensor_id,
                                const float* ranges, const float* intensities,
                                int num_points, double timestamp, float confidence)
{
    if (!pipeline || !ranges || num_points <= 0)
        return SENSOR_PIPELINE_ERROR_INVALID;

    size_t data_size = (size_t)num_points * sizeof(float) * 2;
    uint8_t* buffer = (uint8_t*)malloc(data_size);
    if (!buffer) return SENSOR_PIPELINE_ERROR_NO_MEMORY;

    memcpy(buffer, ranges, (size_t)num_points * sizeof(float));
    if (intensities)
        memcpy(buffer + (size_t)num_points * sizeof(float), intensities,
               (size_t)num_points * sizeof(float));

    SensorPipelineEntry entry;
    memset(&entry, 0, sizeof(entry));
    entry.sensor_id = sensor_id;
    entry.sensor_type = SENSOR_TYPE_LIDAR;
    entry.source = SENSOR_SOURCE_HARDWARE;
    entry.timestamp = timestamp;
    entry.data = buffer;
    entry.data_size = data_size;
    entry.is_valid = 1;
    entry.confidence = confidence;
    entry.range_min = 0.0f;
    entry.range_max = (float)num_points;

    int ret = sensor_pipeline_push_data(pipeline, &entry);
    free(buffer);
    return ret;
}

int sensor_pipeline_push_imu(SensorPipeline* pipeline, int sensor_id,
                              const float* acceleration, const float* angular_velocity,
                              const float* orientation, double timestamp, float confidence)
{
    if (!pipeline) return SENSOR_PIPELINE_ERROR_INVALID;

    float data[10];
    memset(data, 0, sizeof(data));
    if (acceleration) { data[0] = acceleration[0]; data[1] = acceleration[1]; data[2] = acceleration[2]; }
    if (angular_velocity) { data[3] = angular_velocity[0]; data[4] = angular_velocity[1]; data[5] = angular_velocity[2]; }
    if (orientation) { data[6] = orientation[0]; data[7] = orientation[1]; data[8] = orientation[2]; data[9] = orientation[3]; }

    SensorPipelineEntry entry;
    memset(&entry, 0, sizeof(entry));
    entry.sensor_id = sensor_id;
    entry.sensor_type = SENSOR_TYPE_IMU;
    entry.source = SENSOR_SOURCE_HARDWARE;
    entry.timestamp = timestamp;
    entry.data = (uint8_t*)data;
    entry.data_size = sizeof(data);
    entry.is_valid = 1;
    entry.confidence = confidence;
    return sensor_pipeline_push_data(pipeline, &entry);
}

int sensor_pipeline_push_gnss(SensorPipeline* pipeline, int sensor_id,
                               double latitude, double longitude, double altitude,
                               double timestamp, float confidence)
{
    if (!pipeline) return SENSOR_PIPELINE_ERROR_INVALID;

    double gnss_data[3] = { latitude, longitude, altitude };
    SensorPipelineEntry entry;
    memset(&entry, 0, sizeof(entry));
    entry.sensor_id = sensor_id;
    entry.sensor_type = SENSOR_TYPE_GNSS;
    entry.source = SENSOR_SOURCE_HARDWARE;
    entry.timestamp = timestamp;
    entry.data = (uint8_t*)gnss_data;
    entry.data_size = sizeof(gnss_data);
    entry.is_valid = 1;
    entry.confidence = confidence;
    return sensor_pipeline_push_data(pipeline, &entry);
}

int sensor_pipeline_push_force_torque(SensorPipeline* pipeline, int sensor_id,
                                       const float* force, const float* torque,
                                       double timestamp, float confidence)
{
    if (!pipeline) return SENSOR_PIPELINE_ERROR_INVALID;

    float ft_data[6] = { 0 };
    if (force) { ft_data[0] = force[0]; ft_data[1] = force[1]; ft_data[2] = force[2]; }
    if (torque) { ft_data[3] = torque[0]; ft_data[4] = torque[1]; ft_data[5] = torque[2]; }

    SensorPipelineEntry entry;
    memset(&entry, 0, sizeof(entry));
    entry.sensor_id = sensor_id;
    entry.sensor_type = SENSOR_TYPE_FORCE_TORQUE;
    entry.source = SENSOR_SOURCE_HARDWARE;
    entry.timestamp = timestamp;
    entry.data = (uint8_t*)ft_data;
    entry.data_size = sizeof(ft_data);
    entry.is_valid = 1;
    entry.confidence = confidence;
    return sensor_pipeline_push_data(pipeline, &entry);
}

int sensor_pipeline_get_latest(SensorPipeline* pipeline, int sensor_id,
                                SensorPipelineEntry* entry)
{
    if (!pipeline || !entry) return SENSOR_PIPELINE_ERROR_INVALID;
    return ring_buffer_get_latest(&pipeline->ring_buffer, sensor_id, entry);
}

int sensor_pipeline_get_latest_by_type(SensorPipeline* pipeline, SensorType sensor_type,
                                        SensorPipelineEntry* entry)
{
    if (!pipeline || !entry) return SENSOR_PIPELINE_ERROR_INVALID;
    return ring_buffer_get_latest_by_type(&pipeline->ring_buffer, sensor_type, entry);
}

int sensor_pipeline_get_history(SensorPipeline* pipeline, int sensor_id,
                                 SensorPipelineEntry* entries, int* count, int max_entries)
{
    if (!pipeline || !entries || !count) return SENSOR_PIPELINE_ERROR_INVALID;
    return ring_buffer_get_history(&pipeline->ring_buffer, sensor_id, entries, count, max_entries);
}

int sensor_pipeline_get_history_by_type(SensorPipeline* pipeline, SensorType sensor_type,
                                         SensorPipelineEntry* entries, int* count,
                                         int max_entries)
{
    if (!pipeline || !entries || !count) return SENSOR_PIPELINE_ERROR_INVALID;
    if (!pipeline->ring_initialized || pipeline->ring_buffer.count == 0)
    {
        *count = 0;
        return SENSOR_PIPELINE_ERROR_BUFFER_EMPTY;
    }
    int found = 0;
    int idx = (pipeline->ring_buffer.head - 1 + pipeline->ring_buffer.capacity) % pipeline->ring_buffer.capacity;
    for (int i = 0; i < pipeline->ring_buffer.count && found < max_entries; i++)
    {
        if (pipeline->ring_buffer.entries[idx].sensor_type == sensor_type)
            entries[found++] = pipeline->ring_buffer.entries[idx];
        idx = (idx - 1 + pipeline->ring_buffer.capacity) % pipeline->ring_buffer.capacity;
    }
    *count = found;
    return (found > 0) ? SENSOR_PIPELINE_OK : SENSOR_PIPELINE_ERROR_BUFFER_EMPTY;
}

int sensor_pipeline_get_filtered(SensorPipeline* pipeline, const SensorPipelineFilter* filter,
                                  SensorPipelineEntry* entries, int* count, int max_entries)
{
    if (!pipeline || !filter || !entries || !count)
        return SENSOR_PIPELINE_ERROR_INVALID;
    if (!pipeline->ring_initialized || pipeline->ring_buffer.count == 0)
    {
        *count = 0;
        return SENSOR_PIPELINE_ERROR_BUFFER_EMPTY;
    }
    int found = 0;
    int idx = (pipeline->ring_buffer.head - 1 + pipeline->ring_buffer.capacity) % pipeline->ring_buffer.capacity;
    for (int i = 0; i < pipeline->ring_buffer.count && found < max_entries; i++)
    {
        SensorPipelineEntry* e = &pipeline->ring_buffer.entries[idx];
        int match = 1;
        if (filter->sensor_id >= 0 && e->sensor_id != filter->sensor_id) match = 0;
        if (filter->sensor_type >= 0 && e->sensor_type != filter->sensor_type) match = 0;
        if (filter->min_sequence > 0 && e->sequence < filter->min_sequence) match = 0;
        if (filter->max_sequence > 0 && e->sequence > filter->max_sequence) match = 0;
        if (filter->min_timestamp > 0.0 && e->timestamp < filter->min_timestamp) match = 0;
        if (filter->max_timestamp > 0.0 && e->timestamp > filter->max_timestamp) match = 0;
        if (e->confidence < (float)filter->min_confidence * 0.01f) match = 0;
        if (match)
            entries[found++] = *e;
        idx = (idx - 1 + pipeline->ring_buffer.capacity) % pipeline->ring_buffer.capacity;
    }
    *count = found;
    return (found > 0) ? SENSOR_PIPELINE_OK : SENSOR_PIPELINE_ERROR_BUFFER_EMPTY;
}

int sensor_pipeline_subscribe(SensorPipeline* pipeline, int sensor_id,
                               SensorPipelineCallback callback, void* user_data)
{
    if (!pipeline || !callback) return SENSOR_PIPELINE_ERROR_INVALID;
    if (pipeline->subscriber_count >= SENSOR_PIPELINE_MAX_SUBSCRIBERS)
        return SENSOR_PIPELINE_ERROR_FULL;

    SubscriberEntry* sub = &pipeline->subscribers[pipeline->subscriber_count++];
    sub->sensor_id = sensor_id;
    sub->sensor_type = (SensorType)(-1);
    sub->callback = callback;
    sub->user_data = user_data;
    sub->active = 1;
    return SENSOR_PIPELINE_OK;
}

int sensor_pipeline_subscribe_by_type(SensorPipeline* pipeline, SensorType sensor_type,
                                       SensorPipelineCallback callback, void* user_data)
{
    if (!pipeline || !callback) return SENSOR_PIPELINE_ERROR_INVALID;
    if (pipeline->subscriber_count >= SENSOR_PIPELINE_MAX_SUBSCRIBERS)
        return SENSOR_PIPELINE_ERROR_FULL;

    SubscriberEntry* sub = &pipeline->subscribers[pipeline->subscriber_count++];
    sub->sensor_id = -1;
    sub->sensor_type = sensor_type;
    sub->callback = callback;
    sub->user_data = user_data;
    sub->active = 1;
    return SENSOR_PIPELINE_OK;
}

int sensor_pipeline_unsubscribe(SensorPipeline* pipeline, int sensor_id,
                                 SensorPipelineCallback callback)
{
    if (!pipeline) return SENSOR_PIPELINE_ERROR_INVALID;
    for (int i = 0; i < pipeline->subscriber_count; i++)
    {
        if (pipeline->subscribers[i].sensor_id == sensor_id &&
            pipeline->subscribers[i].callback == callback)
        {
            for (int j = i; j < pipeline->subscriber_count - 1; j++)
                pipeline->subscribers[j] = pipeline->subscribers[j + 1];
            pipeline->subscriber_count--;
            return SENSOR_PIPELINE_OK;
        }
    }
    return SENSOR_PIPELINE_ERROR_NOT_FOUND;
}

int sensor_pipeline_unsubscribe_all(SensorPipeline* pipeline)
{
    if (!pipeline) return SENSOR_PIPELINE_ERROR_INVALID;
    pipeline->subscriber_count = 0;
    return SENSOR_PIPELINE_OK;
}

int sensor_pipeline_set_priority(SensorPipeline* pipeline, int sensor_id,
                                  SensorPipelinePriority priority)
{
    if (!pipeline) return SENSOR_PIPELINE_ERROR_INVALID;
    SensorInfo* si = pipeline_find_sensor(pipeline, sensor_id);
    if (!si) return SENSOR_PIPELINE_ERROR_NOT_FOUND;
    si->priority = priority;
    return SENSOR_PIPELINE_OK;
}

SensorPipelinePriority sensor_pipeline_get_priority(SensorPipeline* pipeline, int sensor_id)
{
    if (!pipeline) return SENSOR_PIPELINE_PRIORITY_BACKGROUND;
    SensorInfo* si = pipeline_find_sensor(pipeline, sensor_id);
    return si ? si->priority : SENSOR_PIPELINE_PRIORITY_BACKGROUND;
}

int sensor_pipeline_set_sample_rate(SensorPipeline* pipeline, int sensor_id,
                                     double sample_rate_hz)
{
    if (!pipeline) return SENSOR_PIPELINE_ERROR_INVALID;
    SensorInfo* si = pipeline_find_sensor(pipeline, sensor_id);
    if (!si) return SENSOR_PIPELINE_ERROR_NOT_FOUND;
    si->sample_rate_hz = (sample_rate_hz > 0.0) ? sample_rate_hz : 1.0;
    return SENSOR_PIPELINE_OK;
}

double sensor_pipeline_get_sample_rate(SensorPipeline* pipeline, int sensor_id)
{
    if (!pipeline) return 0.0;
    SensorInfo* si = pipeline_find_sensor(pipeline, sensor_id);
    return si ? si->sample_rate_hz : 0.0;
}

int sensor_pipeline_enable_sensor(SensorPipeline* pipeline, int sensor_id)
{
    if (!pipeline) return SENSOR_PIPELINE_ERROR_INVALID;
    SensorInfo* si = pipeline_find_sensor(pipeline, sensor_id);
    if (!si) return SENSOR_PIPELINE_ERROR_NOT_FOUND;
    si->enabled = 1;
    return SENSOR_PIPELINE_OK;
}

int sensor_pipeline_disable_sensor(SensorPipeline* pipeline, int sensor_id)
{
    if (!pipeline) return SENSOR_PIPELINE_ERROR_INVALID;
    SensorInfo* si = pipeline_find_sensor(pipeline, sensor_id);
    if (!si) return SENSOR_PIPELINE_ERROR_NOT_FOUND;
    si->enabled = 0;
    return SENSOR_PIPELINE_OK;
}

int sensor_pipeline_is_sensor_enabled(SensorPipeline* pipeline, int sensor_id)
{
    if (!pipeline) return 0;
    SensorInfo* si = pipeline_find_sensor(pipeline, sensor_id);
    return (si && si->enabled) ? 1 : 0;
}

int sensor_pipeline_get_statistics(SensorPipeline* pipeline, int sensor_id,
                                    uint32_t* total_pushed, uint32_t* total_dropped,
                                    double* avg_latency_ms, double* max_latency_ms,
                                    double* last_rate_hz)
{
    if (!pipeline) return SENSOR_PIPELINE_ERROR_INVALID;
    SensorInfo* si = pipeline_find_sensor(pipeline, sensor_id);
    if (!si) return SENSOR_PIPELINE_ERROR_NOT_FOUND;

    if (total_pushed) *total_pushed = si->total_pushed;
    if (total_dropped) *total_dropped = si->total_dropped;
    if (max_latency_ms) *max_latency_ms = si->max_latency_ms;
    if (last_rate_hz) *last_rate_hz = si->current_rate_hz;
    if (avg_latency_ms)
    {
        *avg_latency_ms = (si->total_pushed > 0) ?
            si->total_latency_ms / (double)si->total_pushed : 0.0;
    }
    return SENSOR_PIPELINE_OK;
}

int sensor_pipeline_get_pipeline_load(SensorPipeline* pipeline, double* buffer_usage,
                                       double* avg_push_time_us, double* avg_poll_time_us)
{
    if (!pipeline) return SENSOR_PIPELINE_ERROR_INVALID;
    if (buffer_usage)
    {
        *buffer_usage = (pipeline->ring_buffer.capacity > 0) ?
            (double)pipeline->ring_buffer.count / (double)pipeline->ring_buffer.capacity : 0.0;
    }
    if (avg_push_time_us) *avg_push_time_us = 0.0;
    if (avg_poll_time_us) *avg_poll_time_us = 0.0;
    return SENSOR_PIPELINE_OK;
}

SensorPipelineConfig* sensor_pipeline_config_create(void)
{
    SensorPipelineConfig* config = (SensorPipelineConfig*)calloc(1, sizeof(SensorPipelineConfig));
    if (config) sensor_pipeline_config_set_defaults(config);
    return config;
}

void sensor_pipeline_config_destroy(SensorPipelineConfig* config)
{
    free(config);
}

void sensor_pipeline_config_set_defaults(SensorPipelineConfig* config)
{
    if (!config) return;
    config->max_sensors = SENSOR_PIPELINE_MAX_SENSORS;
    config->max_subscribers = SENSOR_PIPELINE_MAX_SUBSCRIBERS;
    config->max_streaming_clients = SENSOR_PIPELINE_MAX_CLIENTS;
    config->streaming_port = SELFLNN_SENSOR_STREAM_PORT;
    config->enable_streaming_server = 1;
    config->enable_timestamp_sync = 1;
    config->timestamp_sync_tolerance = 0.001;
    config->enable_priority_scheduling = 1;
    config->thread_pool_size = 2;
    config->max_buffer_duration = 10.0;
    config->ring_buffer_config.max_entries = SENSOR_PIPELINE_RING_BUFFER_SIZE;
    config->ring_buffer_config.max_data_size = SENSOR_PIPELINE_MAX_DATA_SIZE;
    config->ring_buffer_config.enable_compression = 0;
    config->ring_buffer_config.compression_level = 0;
    config->ring_buffer_config.enable_dedup = 0;
    config->ring_buffer_config.dedup_threshold = 64.0;
}

int sensor_pipeline_start_streaming_server(SensorPipeline* pipeline)
{
    if (!pipeline) return SENSOR_PIPELINE_ERROR_INVALID;
    if (pipeline->streaming_running) return SENSOR_PIPELINE_OK;
    pipeline->streaming_running = 1;
    ThreadHandle thread = thread_create((void* (*)(void*))streaming_server_thread, pipeline);
    if (!thread)
    {
        pipeline->streaming_running = 0;
        return SENSOR_PIPELINE_ERROR_INVALID;
    }
    return SENSOR_PIPELINE_OK;
}

int sensor_pipeline_stop_streaming_server(SensorPipeline* pipeline)
{
    if (!pipeline) return SENSOR_PIPELINE_ERROR_INVALID;
    if (!pipeline->streaming_running) return SENSOR_PIPELINE_OK;
    pipeline->streaming_running = 0;
    return SENSOR_PIPELINE_OK;
}

int sensor_pipeline_get_streaming_clients(SensorPipeline* pipeline,
                                           SensorStreamingClient* clients, int* count)
{
    if (!pipeline || !clients || !count) return SENSOR_PIPELINE_ERROR_INVALID;
    int max = *count;
    *count = 0;
    for (int i = 0; i < SENSOR_PIPELINE_MAX_CLIENTS && *count < max; i++)
    {
        if (pipeline->streaming_clients[i].is_active)
            clients[(*count)++] = pipeline->streaming_clients[i];
    }
    return SENSOR_PIPELINE_OK;
}

int sensor_pipeline_disconnect_client(SensorPipeline* pipeline, int client_index)
{
    if (!pipeline) return SENSOR_PIPELINE_ERROR_INVALID;
    if (client_index < 0 || client_index >= SENSOR_PIPELINE_MAX_CLIENTS)
        return SENSOR_PIPELINE_ERROR_NOT_FOUND;
    SensorStreamingClient* client = &pipeline->streaming_clients[client_index];
    if (!client->is_active) return SENSOR_PIPELINE_ERROR_NOT_FOUND;
#if defined(_WIN32)
    closesocket((SOCKET)client->socket);
#else
    close(client->socket);
#endif
    client->is_active = 0;
    client->socket = -1;
    pipeline->client_count--;
    return SENSOR_PIPELINE_OK;
}

int sensor_pipeline_export_to_file(SensorPipeline* pipeline, int sensor_id,
                                    const char* filename, int max_entries)
{
    if (!pipeline || !filename) return SENSOR_PIPELINE_ERROR_INVALID;
    SensorPipelineEntry entries[256];
    int count = 256;
    int ret = sensor_pipeline_get_history(pipeline, sensor_id, entries, &count,
                                           (max_entries > 0 && max_entries < 256) ? max_entries : 256);
    if (ret != SENSOR_PIPELINE_OK) return ret;

    FILE* fp = fopen(filename, "wb");
    if (!fp) return SENSOR_PIPELINE_ERROR_GENERAL;

    uint32_t magic = 0x53454E53;
    uint32_t num_entries = (uint32_t)count;
    fwrite(&magic, sizeof(magic), 1, fp);
    fwrite(&num_entries, sizeof(num_entries), 1, fp);
    for (int i = 0; i < count; i++)
    {
        uint32_t type = (uint32_t)entries[i].sensor_type;
        uint32_t size = (uint32_t)entries[i].data_size;
        double ts = entries[i].timestamp;
        float conf = entries[i].confidence;
        fwrite(&type, sizeof(type), 1, fp);
        fwrite(&size, sizeof(size), 1, fp);
        fwrite(&ts, sizeof(ts), 1, fp);
        fwrite(&conf, sizeof(conf), 1, fp);
        if (size > 0 && entries[i].data)
            fwrite(entries[i].data, 1, size, fp);
    }
    fclose(fp);
    return SENSOR_PIPELINE_OK;
}

int sensor_pipeline_import_from_file(SensorPipeline* pipeline, const char* filename)
{
    if (!pipeline || !filename) return SENSOR_PIPELINE_ERROR_INVALID;
    FILE* fp = fopen(filename, "rb");
    if (!fp) return SENSOR_PIPELINE_ERROR_GENERAL;

    uint32_t magic, num_entries;
    if (fread(&magic, sizeof(magic), 1, fp) != 1 || magic != 0x53454E53)
    {
        fclose(fp);
        return SENSOR_PIPELINE_ERROR_INVALID;
    }
    if (fread(&num_entries, sizeof(num_entries), 1, fp) != 1)
    {
        fclose(fp);
        return SENSOR_PIPELINE_ERROR_INVALID;
    }

    for (uint32_t i = 0; i < num_entries; i++)
    {
        uint32_t type_u32, size_u32;
        double ts;
        float conf;
        if (fread(&type_u32, sizeof(type_u32), 1, fp) != 1) break;
        if (fread(&size_u32, sizeof(size_u32), 1, fp) != 1) break;
        if (fread(&ts, sizeof(ts), 1, fp) != 1) break;
        if (fread(&conf, sizeof(conf), 1, fp) != 1) break;

        SensorType type = (SensorType)type_u32;
        size_t dsize = (size_t)size_u32;
        uint8_t* data = NULL;
        if (dsize > 0)
        {
            data = (uint8_t*)malloc(dsize);
            if (data)
            {
                if (fread(data, 1, dsize, fp) != dsize)
                {
                    free(data);
                    break;
                }
            }
        }

        SensorPipelineEntry entry;
        memset(&entry, 0, sizeof(entry));
        entry.sensor_id = -1;
        entry.sensor_type = type;
        entry.timestamp = ts;
        entry.data = data;
        entry.data_size = dsize;
        entry.is_valid = 1;
        entry.confidence = conf;
        ring_buffer_push(&pipeline->ring_buffer, &entry);
        free(data);
    }

    fclose(fp);
    return SENSOR_PIPELINE_OK;
}

int sensor_pipeline_clear_buffer(SensorPipeline* pipeline)
{
    if (!pipeline) return SENSOR_PIPELINE_ERROR_INVALID;
    return ring_buffer_clear(&pipeline->ring_buffer);
}

int sensor_pipeline_clear_sensor_buffer(SensorPipeline* pipeline, int sensor_id)
{
    if (!pipeline) return SENSOR_PIPELINE_ERROR_INVALID;
    return ring_buffer_clear_by_sensor(&pipeline->ring_buffer, sensor_id);
}

int sensor_pipeline_get_global_stats(SensorPipeline* pipeline, SensorPipelineStats* stats)
{
    if (!pipeline || !stats) return SENSOR_PIPELINE_ERROR_INVALID;
    memset(stats, 0, sizeof(SensorPipelineStats));
    stats->buffer_capacity = pipeline->ring_buffer.capacity;
    stats->buffer_entries = pipeline->ring_buffer.count;
    for (int i = 0; i < pipeline->sensor_count; i++)
    {
        stats->entry_count += pipeline->sensors[i].total_pushed;
        stats->dropped_count += pipeline->sensors[i].total_dropped;
        stats->avg_latency += pipeline->sensors[i].total_latency_ms;
        if (pipeline->sensors[i].max_latency_ms > stats->max_latency)
            stats->max_latency = pipeline->sensors[i].max_latency_ms;
    }
    if (stats->entry_count > 0)
        stats->avg_latency /= (double)stats->entry_count;
    return SENSOR_PIPELINE_OK;
}

const char* sensor_pipeline_get_last_error(SensorPipeline* pipeline)
{
    return pipeline ? pipeline->last_error : "空管道指针";
}

const char* sensor_pipeline_priority_string(SensorPipelinePriority priority)
{
    switch (priority)
    {
        case SENSOR_PIPELINE_PRIORITY_CRITICAL: return "关键";
        case SENSOR_PIPELINE_PRIORITY_HIGH: return "高";
        case SENSOR_PIPELINE_PRIORITY_MEDIUM: return "中";
        case SENSOR_PIPELINE_PRIORITY_LOW: return "低";
        case SENSOR_PIPELINE_PRIORITY_BACKGROUND: return "后台";
        default: return "未知";
    }
}

const char* sensor_pipeline_source_string(SensorSource source)
{
    switch (source)
    {
        case SENSOR_SOURCE_SIMULATOR: return "仿真器";
        case SENSOR_SOURCE_HARDWARE: return "硬件";
        case SENSOR_SOURCE_FILE: return "文件";
        case SENSOR_SOURCE_NETWORK: return "网络";
        case SENSOR_SOURCE_SYNTHETIC: return "合成";
        default: return "未知";
    }
}

const char* sensor_pipeline_error_string(int error_code)
{
    switch (error_code)
    {
        case SENSOR_PIPELINE_OK: return "操作成功";
        case SENSOR_PIPELINE_ERROR_GENERAL: return "通用错误";
        case SENSOR_PIPELINE_ERROR_NO_MEMORY: return "内存不足";
        case SENSOR_PIPELINE_ERROR_NOT_FOUND: return "未找到";
        case SENSOR_PIPELINE_ERROR_FULL: return "管道已满";
        case SENSOR_PIPELINE_ERROR_INVALID: return "无效参数";
        case SENSOR_PIPELINE_ERROR_TIMEOUT: return "超时";
        case SENSOR_PIPELINE_ERROR_DISABLED: return "已禁用";
        case SENSOR_PIPELINE_ERROR_BUFFER_EMPTY: return "缓冲区为空";
        case SENSOR_PIPELINE_ERROR_BUFFER_FULL: return "缓冲区已满";
        case SENSOR_PIPELINE_ERROR_COMPRESSION: return "压缩错误";
        case SENSOR_PIPELINE_ERROR_STREAM: return "流错误";
        default: return "未知错误";
    }
}
