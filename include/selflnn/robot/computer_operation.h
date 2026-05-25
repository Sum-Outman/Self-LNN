#ifndef SELFLNN_COMPUTER_OPERATION_H
#define SELFLNN_COMPUTER_OPERATION_H

#include "selflnn/core/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CO_MAX_ACTIONS 512
#define CO_SCREEN_WIDTH 1920
#define CO_SCREEN_HEIGHT 1080
#define CO_SCREEN_CHANNELS 3
#define CO_MAX_ELEMENTS 256
#define CO_LABEL_LEN 128
#define CO_MAX_OCR_RESULTS 128
#define CO_OCR_TEXT_LEN 256
/* ZSFX-011: OCR字符集定义 — 从36类(0-9,A-Z)扩展到全ASCII可打印 */
#define CO_OCR_NUM_CLASSES 96          /* 95个可打印ASCII(32-126) + 1个未知类(索引0) */
#define CO_OCR_CONFIDENCE_THRESHOLD 0.4f  /* 字符识别置信度阈值 */
extern const char CO_OCR_CHARSET[CO_OCR_NUM_CLASSES];  /* OCR识别字符集映射表 */
#define CO_MAX_DEMO_FRAMES 10000
#define CO_MAX_WINDOWS 64
#define CO_MAX_FILES 256
#define CO_FILE_PATH_LEN 512
#define CO_MAX_BROWSER_TABS 32
#define CO_SAFETY_RULES 64

typedef enum {
    CO_ACTION_CLICK = 0,
    CO_ACTION_DOUBLE_CLICK = 1,
    CO_ACTION_RIGHT_CLICK = 2,
    CO_ACTION_DRAG = 3,
    CO_ACTION_SCROLL = 4,
    CO_ACTION_KEYPRESS = 5,
    CO_ACTION_KEY_COMBO = 6,
    CO_ACTION_TYPE_TEXT = 7,
    CO_ACTION_MOVE = 8,
    CO_ACTION_HOVER = 9,
    CO_ACTION_LAUNCH_APP = 10,
    CO_ACTION_CLOSE_WINDOW = 11,
    CO_ACTION_SWITCH_WINDOW = 12,
    CO_ACTION_FILE_OPEN = 13,
    CO_ACTION_FILE_SAVE = 14,
    CO_ACTION_FILE_DELETE = 15,
    CO_ACTION_BROWSER_NAV = 16,
    CO_ACTION_SCREENSHOT = 17,
    CO_ACTION_MAX = 20
} COActionType;

typedef enum {
    CO_UI_BUTTON = 0,
    CO_UI_TEXT_INPUT = 1,
    CO_UI_LABEL = 2,
    CO_UI_LINK = 3,
    CO_UI_CHECKBOX = 4,
    CO_UI_RADIO = 5,
    CO_UI_DROPDOWN = 6,
    CO_UI_SLIDER = 7,
    CO_UI_ICON = 8,
    CO_UI_IMAGE = 9,
    CO_UI_TABLE = 10,
    CO_UI_MENU = 11,
    CO_UI_SCROLLBAR = 12,
    CO_UI_DIALOG = 13,
    CO_UI_WINDOW = 14,
    CO_UI_UNKNOWN = 15
} COUIElementType;

typedef struct {
    char text[CO_OCR_TEXT_LEN];
    float bbox[4];
    float confidence;
    int is_recognized;
} COOCRResult;

typedef struct {
    char label[CO_LABEL_LEN];
    COUIElementType ui_type;
    float bbox[4];
    float confidence;
    int is_interactive;
    int is_visible;
    char associated_text[CO_OCR_TEXT_LEN];
} COUIElement;

typedef struct {
    char title[CO_LABEL_LEN];
    char process_name[CO_FILE_PATH_LEN];
    int pid;
    float position[4];
    int is_active;
    int is_minimized;
    float last_active_time;
} COWindowInfo;

typedef struct {
    char path[CO_FILE_PATH_LEN];
    char name[CO_LABEL_LEN];
    char extension[32];
    size_t size_bytes;
    int is_directory;
    int is_readonly;
    float last_modified;
} COFileInfo;

typedef struct {
    char url[1024];
    char title[CO_LABEL_LEN];
    int tab_id;
    int is_active;
    float load_progress;
    char status[64];
} COBrowserTab;

typedef struct {
    COActionType action_type;
    int x;
    int y;
    int target_x;
    int target_y;
    int key_code;
    int modifier_flags;
    char text[256];
    char window_title[128];
    char file_path[CO_FILE_PATH_LEN];
    char url[1024];
    float duration;
    float confidence;
    int safety_checked;
} COAction;

typedef struct {
    int rule_id;
    char description[256];
    int (*check_func)(const COAction*, const void* context);
    int is_enabled;
    int severity;
} COSafetyRule;

typedef struct {
    COAction actions[CO_MAX_ACTIONS];
    size_t num_actions;
    float screen_embedding[1024];
    COUIElement detected_elements[CO_MAX_ELEMENTS];
    float element_positions[CO_MAX_ELEMENTS][4];
    COOCRResult ocr_results[CO_MAX_OCR_RESULTS];
    size_t num_elements;
    size_t num_ocr_results;
    float task_embedding[512];
    char task_description[256];
    int is_plan_complete;
} COPlan;

typedef struct {
    size_t action_history_size;
    float screen_region_of_interest[4];
    int use_vision_guidance;
    int use_plan_execution;
    float execution_speed;
    float error_tolerance;
    int max_retries;
    int enable_element_detection;
    int enable_safety_check;
    int enable_ocr;
    int enable_browser_control;
    int enable_file_operations;
    int enable_app_control;
    int enable_demo_recording;
    float safety_confirm_threshold;
} COConfig;

#define CO_CONFIG_DEFAULT { \
    100, {0,0,1920,1080}, 1, 1, 1.0f, 0.1f, 3, 1, 1, 1, 1, 1, 1, 1, 0.8f \
}

typedef struct COSystem COSystem;

COSystem* co_system_create(COConfig config);
void co_system_destroy(COSystem* system);

int co_analyze_screen(COSystem* system, const float* screen_data, size_t width, size_t height, size_t channels);
int co_plan_task(COSystem* system, const char* task_description, COPlan* plan_out);
int co_execute_action(COSystem* system, const COAction* action);
int co_execute_plan(COSystem* system, COPlan* plan, int (*progress_callback)(float progress, const char* status, void* user_data), void* user_data);

int co_recognize_ui_element(COSystem* system, const float* screen_region, size_t region_width, size_t region_height, char* element_label, size_t label_buf_size, float* confidence);
int co_ocr_recognize(COSystem* system, const float* screen_data, size_t width, size_t height, COOCRResult* results, size_t* num_results);
int co_classify_ui_element(COSystem* system, const float* region_data, size_t rw, size_t rh, COUIElement* element_out);
int co_detect_text_region(COSystem* system, const float* screen_data, size_t width, size_t height, float* regions_out, size_t* num_regions);

int co_browser_navigate(COSystem* system, const char* url);
int co_browser_click_link(COSystem* system, const char* link_text);
int co_browser_fill_form(COSystem* system, const char* field_name, const char* value);
int co_browser_get_tabs(COSystem* system, COBrowserTab* tabs, size_t* num_tabs);
int co_browser_switch_tab(COSystem* system, int tab_id);
int co_browser_close_tab(COSystem* system, int tab_id);
int co_browser_scroll(COSystem* system, int dx, int dy);
int co_browser_execute_js(COSystem* system, const char* js_code, char* result_out, size_t result_size);

int co_file_open(COSystem* system, const char* path);
int co_file_save(COSystem* system, const char* path, const char* content, size_t content_len);
int co_file_delete(COSystem* system, const char* path);
int co_file_list_directory(COSystem* system, const char* dir_path, COFileInfo* files, size_t* num_files);
int co_file_copy(COSystem* system, const char* src, const char* dst);
int co_file_move(COSystem* system, const char* src, const char* dst);
int co_file_create_directory(COSystem* system, const char* path);

int co_launch_app(COSystem* system, const char* app_path, const char* args);
int co_close_app(COSystem* system, const char* app_name);
int co_list_windows(COSystem* system, COWindowInfo* windows, size_t* num_windows);
int co_switch_window(COSystem* system, const char* window_title);
int co_get_active_window(COSystem* system, COWindowInfo* window_out);
int co_minimize_window(COSystem* system, const char* window_title);
int co_maximize_window(COSystem* system, const char* window_title);

int co_find_windows_by_title(COSystem* system, const char* pattern,
                              COWindowInfo* windows, size_t* num_windows);
int co_set_window_info(COSystem* system, const COWindowInfo* info);

int co_simulate_keypress(COSystem* system, int key_code, int modifier_flags);
int co_simulate_key_combo(COSystem* system, const int* keys, size_t num_keys);
int co_simulate_type_text(COSystem* system, const char* text, float speed);
int co_simulate_mouse_move(COSystem* system, int x, int y);
int co_simulate_mouse_click(COSystem* system, int x, int y, int button);
int co_simulate_mouse_drag(COSystem* system, int x1, int y1, int x2, int y2);
int co_simulate_mouse_scroll(COSystem* system, int clicks);
int co_simulate_mouse_double_click(COSystem* system, int x, int y);

int co_screenshot_capture(COSystem* system, float* data_out, size_t* width, size_t* height);
int co_screenshot_compare(COSystem* system, const float* before, const float* after, size_t w, size_t h, float* diff_out);
int co_screenshot_find_change(COSystem* system, const float* before, const float* after, size_t w, size_t h, float* changed_region);

int co_safety_validate_action(COSystem* system, const COAction* action, char* reason_out, size_t reason_size);
int co_safety_add_rule(COSystem* system, COSafetyRule rule);
int co_safety_remove_rule(COSystem* system, int rule_id);
int co_safety_check_destructive(const COAction* action, const void* context);

int co_record_demo_start(COSystem* system);
int co_record_demo_frame(COSystem* system, const float* screen_data, size_t w, size_t h, const COAction* action);
int co_record_demo_stop(COSystem* system);
int co_record_demo_save(COSystem* system, const char* filepath);
int co_record_demo_load(COSystem* system, const char* filepath);
int co_record_demo_replay(COSystem* system, int (*progress_cb)(float, const char*, void*), void* user_data);

int co_get_task_stats(COSystem* system, size_t* completed, size_t* failed, float* success_rate);
int co_reset_stats(COSystem* system);
int co_get_system_status(COSystem* system, char* status_out, size_t status_size);

int co_learn_from_demo(COSystem* system, const float* screen_sequence, const COAction* action_sequence, size_t num_frames, size_t width, size_t height, const char* task_label);

#ifdef __cplusplus
}
#endif

#endif
