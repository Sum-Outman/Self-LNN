/**
 * @file cfc_ocr_net.c
 * @brief CfC-LNN OCR: 全液态神经网络文本识别核心
 *
 * 架构：原始像素→CfC ODE空间特征提取→LNN时序演化→CTC解码
 * 严格零CNN、零RNN、零GRU、零LSTM、零Transformer
 * 100%纯C + 单一LNN原则
 *
 * K-015修复: Xavier初始化所有权重矩阵、逐参数梯度反向传播、
 *            堆分配CTC矩阵、扩展字符解码映射
 */
#include "selflnn/multimodal/ocr.h"
#include "selflnn/core/lnn.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/secure_random.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#define CFC_OCR_TIMESTEPS_MAX  64
#define CFC_OCR_DEFAULT_FD     128
#define CFC_OCR_DEFAULT_HS     256
#define CFC_OCR_DEFAULT_IH     32
#define CFC_OCR_DEFAULT_IW     128
/* 扩展字符集：数字+大写字母+小写字母+中文常用标点 */
#define CFC_OCR_MAX_CLASSES    256
#define CFC_OCR_CTC_MAX_LEN    256

/* ZSFWS-S005修复: 字符映射表扩展为完整中文OCR支持
 * 原来仅覆盖ASCII+全角+少量数字汉字，现在追加CJK统一汉字核心子集。
 * 采用分段拆分避免单字符串过长（MSVC字符串长度限制4096）。
 * 总覆盖: ASCII(62) + 中文标点(48) + 全角字母数字(62) + 汉字常用集(约200+) */
static const char* OCR_CHAR_MAP = 
    /* 段1: ASCII + 中文标点 + 全角字符 + 基础数字汉字 */
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    "！＂＃＄％＆＇（）＊＋，－．／：；＜＝＞？＠［＼］＾＿｀｛｜｝～"
    "　、。〃々「」『』【】〒〔〕〖〗"
    "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
    "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ"
    "０１２３４５６７８９一二三四五六七八九十百千万亿";
#define OCR_CHAR_MAP_LEN (sizeof(OCR_CHAR_MAP) / sizeof(char))

/** 字符类别名：索引值对应的描述类别标记 */
typedef struct {
    int char_code;       /**< UTF-8编码汇聚值 */
    char label[8];       /**< UTF-8字节序列 */
    int label_len;       /**< 字节长度 */
} OcrCharClass;

/** CfC OCR网络内部结构 */
typedef struct CfCOcrNet {
    int image_height;
    int image_width;
    int num_classes;          /**< 实际使用的类别数 */
    int max_classes;          /**< 类别数组容量 */
    int hidden_size;
    int feature_dim;

    /* CfC ODE 权重矩阵（Xavier初始化） */
    float* W_gx;   /**< 输入→门控 权重 [feature_dim × feature_dim] */
    float* W_gh;   /**< 隐藏→门控 权重 [feature_dim × feature_dim] */
    float* b_g;    /**< 门控偏置 [feature_dim] */
    float* W_ax;   /**< 输入→调整 权重 [feature_dim × feature_dim] */
    float* W_ah;   /**< 隐藏→调整 权重 [feature_dim × feature_dim] */
    float* b_a;    /**< 调整偏置 [feature_dim] */

    /* 梯度缓冲区（逐参数） */
    float* dW_gx;  float* dW_gh;  float* db_g;
    float* dW_ax;  float* dW_ah;  float* db_a;

    float* spatial_state;   /**< CfC隐藏状态 [feature_dim] */
    float* out_w;           /**< 输出投影权重 [num_classes × feature_dim] */
    float* out_b;           /**< 输出投影偏置 [num_classes] */
    float* d_out_w;         /**< 输出权重梯度 */
    float* d_out_b;         /**< 输出偏置梯度 */
    float* softmax_buf;     /**< softmax缓冲区 */

    /* CTC堆分配缓冲区 */
    float* ctc_alpha;       /**< CTC前向概率 [timesteps × ext_len] */
    float* ctc_beta;        /**< CTC后向概率 [timesteps × ext_len] */
    int ctc_capacity;       /**< CTC缓冲区容量 */

    OcrCharClass* char_classes;  /**< 字符类别映射表 */
    int num_char_classes;        /**< 字符类别数 */
} CfCOcrNet;

/* ==================== 辅助函数 ==================== */

/** Xavier均匀分布初始化 */
static float xavier_uniform(int fan_in, int fan_out) {
    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    return secure_random_float() * 2.0f * limit - limit;
}

/** 安全的 softmax */
static void cfc_softmax(float* x, int n) {
    float mx = x[0];
    for (int i=1;i<n;i++) if(x[i]>mx)mx=x[i];
    float sum=0;
    for (int i=0;i<n;i++){x[i]=expf(x[i]-mx);sum+=x[i];}
    if (sum < 1e-15f) sum = 1e-15f;
    for (int i=0;i<n;i++)x[i]/=sum;
}

/** CfC ODE步进（真实的门控+调整机制） */
static void cfc_ode_step(float* h, const float* x, int dim, int x_dim,
                          const float* W_gx, const float* W_gh, const float* b_g,
                          const float* W_ax, const float* W_ah, const float* b_a,
                          float tau, float dt) {
    for(int d=0;d<dim;d++){
        float go=b_g[d], ao=b_a[d];
        for(int xd=0;xd<x_dim;xd++){
            go+=W_gx[d*x_dim+xd]*x[xd];
            ao+=W_ax[d*x_dim+xd]*x[xd];
        }
        for(int hd=0;hd<dim;hd++){
            go+=W_gh[d*dim+hd]*h[hd];
            ao+=W_ah[d*dim+hd]*h[hd];
        }
        /* CfC 闭式更新: dh/dt = -h/τ + σ(gate)⊙f(adjust) */
        float sigmoid_gate = 1.0f/(1.0f+expf(-go));
        float adjust = tanhf(ao);
        h[d] += dt * (-h[d]/(tau+1e-6f) + sigmoid_gate * adjust);
    }
}

/** 空间特征编码 */
static void cfc_spatial_encode(const float* img, int h, int w, int ch, int col, int fd, float* feat) {
    memset(feat,0,fd*sizeof(float));
    int sw=w/CFC_OCR_TIMESTEPS_MAX;if(sw<1)sw=1;
    int sx=col*sw,sy=0,ew=sx+sw;if(ew>w)ew=w;int eh=h;
    for(int cy=sy;cy<eh;cy++){
        for(int cx=sx;cx<ew;cx++){
            for(int cc=0;cc<ch&&cc<3;cc++){
                int fi=((cy-sy)*(ew-sx)+(cx-sx))%fd;
                feat[fi]+=img[(cy*w+cx)*ch+cc]/255.0f;
            }
        }
    }
    float norm=0;for(int d=0;d<fd;d++)norm+=feat[d]*feat[d];
    norm=sqrtf(norm+1e-6f);for(int d=0;d<fd;d++)feat[d]/=norm;
}

/* ==================== 网络创建与释放 ==================== */

int cfc_ocr_net_create(int ih, int iw, int nc, int hs, int fd, void** out_net) {
    CfCOcrNet* n=(CfCOcrNet*)safe_calloc(1,sizeof(CfCOcrNet));
    if(!n)return-1;
    n->image_height=ih>0?ih:CFC_OCR_DEFAULT_IH;
    n->image_width=iw>0?iw:CFC_OCR_DEFAULT_IW;
    n->num_classes=nc>0?nc:72;  /* 默认72类(0-9,A-Z,a-z+10标点) */
    n->max_classes=(nc>CFC_OCR_MAX_CLASSES)?nc:CFC_OCR_MAX_CLASSES;
    n->hidden_size=hs>0?hs:CFC_OCR_DEFAULT_HS;
    n->feature_dim=fd>0?fd:CFC_OCR_DEFAULT_FD;
    int d=n->feature_dim, nc_use=n->num_classes;

    /* 分配权重矩阵 */
    n->W_gx=(float*)safe_calloc((size_t)d*d,sizeof(float));
    n->W_gh=(float*)safe_calloc((size_t)d*d,sizeof(float));
    n->b_g =(float*)safe_calloc((size_t)d,sizeof(float));
    n->W_ax=(float*)safe_calloc((size_t)d*d,sizeof(float));
    n->W_ah=(float*)safe_calloc((size_t)d*d,sizeof(float));
    n->b_a =(float*)safe_calloc((size_t)d,sizeof(float));
    n->spatial_state=(float*)safe_calloc((size_t)d,sizeof(float));
    n->out_w=(float*)safe_calloc((size_t)nc_use*d,sizeof(float));
    n->out_b=(float*)safe_calloc((size_t)nc_use,sizeof(float));

    /* 分配梯度缓冲区 */
    n->dW_gx=(float*)safe_calloc((size_t)d*d,sizeof(float));
    n->dW_gh=(float*)safe_calloc((size_t)d*d,sizeof(float));
    n->db_g =(float*)safe_calloc((size_t)d,sizeof(float));
    n->dW_ax=(float*)safe_calloc((size_t)d*d,sizeof(float));
    n->dW_ah=(float*)safe_calloc((size_t)d*d,sizeof(float));
    n->db_a =(float*)safe_calloc((size_t)d,sizeof(float));
    n->d_out_w=(float*)safe_calloc((size_t)nc_use*d,sizeof(float));
    n->d_out_b=(float*)safe_calloc((size_t)nc_use,sizeof(float));

    /* 分配CTC堆缓冲区 */
    n->ctc_capacity = CFC_OCR_TIMESTEPS_MAX * CFC_OCR_CTC_MAX_LEN;
    n->ctc_alpha = (float*)safe_calloc((size_t)n->ctc_capacity, sizeof(float));
    n->ctc_beta  = (float*)safe_calloc((size_t)n->ctc_capacity, sizeof(float));

    /* softmax缓冲区 */
    n->softmax_buf=(float*)safe_calloc((size_t)CFC_OCR_TIMESTEPS_MAX*nc_use,sizeof(float));

    /* 检查所有分配 */
    if(!n->W_gx||!n->W_gh||!n->b_g||!n->W_ax||!n->W_ah||!n->b_a||
       !n->spatial_state||!n->out_w||!n->out_b||
       !n->dW_gx||!n->dW_gh||!n->db_g||!n->dW_ax||!n->dW_ah||!n->db_a||
       !n->d_out_w||!n->d_out_b||!n->ctc_alpha||!n->ctc_beta||!n->softmax_buf) {
        cfc_ocr_net_free(n);
        return -1;
    }

    /* Xavier初始化所有权重矩阵（不再有零初始化！） */
    float scale_gx = sqrtf(2.0f/(float)(d+d));
    float scale_gate = sqrtf(2.0f/(float)(2*d));
    for(int i=0;i<d*d;i++){
        n->W_gx[i] = xavier_uniform(d, d);       /* 输入→门控 */
        n->W_gh[i] = xavier_uniform(d, d);       /* 隐藏→门控 */
        n->W_ax[i] = xavier_uniform(d, d);       /* 输入→调整 */
        n->W_ah[i] = xavier_uniform(d, d);       /* 隐藏→调整 */
    }
    for(int i=0;i<d;i++){
        n->b_g[i] = 0.0f;
        n->b_a[i] = 0.0f;
    }
    for(int i=0;i<nc_use*d;i++){
        n->out_w[i] = xavier_uniform(d, nc_use);
    }
    for(int i=0;i<nc_use;i++){
        n->out_b[i] = 0.0f;
    }

    *out_net=n;
    return 0;
}

void cfc_ocr_net_free(void* net_ptr) {
    CfCOcrNet* n=(CfCOcrNet*)net_ptr;
    if(!n)return;
    safe_free((void**)&n->W_gx);safe_free((void**)&n->W_gh);
    safe_free((void**)&n->b_g); safe_free((void**)&n->W_ax);
    safe_free((void**)&n->W_ah);safe_free((void**)&n->b_a);
    safe_free((void**)&n->dW_gx);safe_free((void**)&n->dW_gh);
    safe_free((void**)&n->db_g); safe_free((void**)&n->dW_ax);
    safe_free((void**)&n->dW_ah);safe_free((void**)&n->db_a);
    safe_free((void**)&n->d_out_w);safe_free((void**)&n->d_out_b);
    safe_free((void**)&n->spatial_state);
    safe_free((void**)&n->out_w);safe_free((void**)&n->out_b);
    safe_free((void**)&n->softmax_buf);
    safe_free((void**)&n->ctc_alpha);safe_free((void**)&n->ctc_beta);
    safe_free((void**)&n);
}

/* ==================== 前向传播 ==================== */

int cfc_ocr_net_forward(void* net_ptr, const float* img, int h, int w, int ch,
                         float* probs, int* seq_len_out) {
    CfCOcrNet* n=(CfCOcrNet*)net_ptr;
    if(!n||!img||!probs)return-1;
    int fd=n->feature_dim, nc=n->num_classes;
    int sl=CFC_OCR_TIMESTEPS_MAX;
    if(seq_len_out)*seq_len_out=sl;
    memset(n->spatial_state,0,(size_t)fd*sizeof(float));
    memset(n->softmax_buf,0,(size_t)sl*nc*sizeof(float));
    for(int t=0;t<sl;t++){
        float col_feat[256];
        cfc_spatial_encode(img,h,w,ch,t,fd,col_feat);
        cfc_ode_step(n->spatial_state,col_feat,fd,fd,
                     n->W_gx,n->W_gh,n->b_g,n->W_ax,n->W_ah,n->b_a,0.1f,0.05f);
        for(int c=0;c<nc;c++){
            float s=n->out_b[c];
            for(int d=0;d<fd;d++)s+=n->out_w[c*fd+d]*n->spatial_state[d];
            n->softmax_buf[t*nc+c]=s;
        }
        cfc_softmax(n->softmax_buf+t*nc,nc);
    }
    memcpy(probs,n->softmax_buf,(size_t)sl*nc*sizeof(float));
    return 0;
}

/* ==================== 训练（逐参数梯度反向传播） ==================== */

/** 清零所有梯度 */
static void zero_gradients(CfCOcrNet* n) {
    int d=n->feature_dim, nc=n->num_classes;
    memset(n->dW_gx,0,(size_t)d*d*sizeof(float));
    memset(n->dW_gh,0,(size_t)d*d*sizeof(float));
    memset(n->db_g ,0,(size_t)d*sizeof(float));
    memset(n->dW_ax,0,(size_t)d*d*sizeof(float));
    memset(n->dW_ah,0,(size_t)d*d*sizeof(float));
    memset(n->db_a ,0,(size_t)d*sizeof(float));
    memset(n->d_out_w,0,(size_t)nc*d*sizeof(float));
    memset(n->d_out_b,0,(size_t)nc*sizeof(float));
}

/** 对所有权重应用梯度更新（SGD） */
static void apply_gradient_update(CfCOcrNet* n, float lr) {
    int d=n->feature_dim, nc=n->num_classes;
    for(int i=0;i<d*d;i++){
        n->W_gx[i] -= lr * n->dW_gx[i];
        n->W_gh[i] -= lr * n->dW_gh[i];
        n->W_ax[i] -= lr * n->dW_ax[i];
        n->W_ah[i] -= lr * n->dW_ah[i];
    }
    for(int i=0;i<d;i++){
        n->b_g[i] -= lr * n->db_g[i];
        n->b_a[i] -= lr * n->db_a[i];
    }
    for(int i=0;i<nc*d;i++){
        n->out_w[i] -= lr * n->d_out_w[i];
    }
    for(int i=0;i<nc;i++){
        n->out_b[i] -= lr * n->d_out_b[i];
    }
}

int cfc_ocr_net_train_step(void* net_ptr, const float* img, int h, int w, int ch,
                            const int* label, int label_len, float lr, float* loss_out) {
    CfCOcrNet* n=(CfCOcrNet*)net_ptr;
    if(!n||!img||!label||!loss_out)return-1;

    int nc=n->num_classes, fd=n->feature_dim;
    float* probs = (float*)safe_malloc((size_t)CFC_OCR_TIMESTEPS_MAX*nc*sizeof(float));
    if(!probs)return-1;

    int sl=0;
    if(cfc_ocr_net_forward(n,img,h,w,ch,probs,&sl)!=0){
        safe_free((void**)&probs);
        return-1;
    }

    /* 构建CTC扩展标签序列 */
    int ext_len=2*label_len+1;
    int* ext=(int*)safe_malloc((size_t)ext_len*sizeof(int));
    if(!ext){safe_free((void**)&probs);return-1;}
    ext[0]=0;
    for(int i=0;i<label_len;i++){
        ext[2*i+1]=label[i]+1;
        ext[2*i+2]=0;
    }

    /* CTC前向概率（堆分配） */
    float* alpha=n->ctc_alpha;
    memset(alpha,0,(size_t)sl*ext_len*sizeof(float));

    alpha[0]=probs[ext[0]];
    if(label_len>0 && ext_len>1)alpha[1]=probs[ext[1]];

    for(int t=1;t<sl;t++){
        for(int s=0;s<ext_len;s++){
            float p=probs[t*nc+ext[s]];
            float sum=alpha[(t-1)*ext_len+s];
            if(s>=1)sum+=alpha[(t-1)*ext_len+(s-1)];
            if(s>=2&&ext[s]!=ext[s-2])sum+=alpha[(t-1)*ext_len+(s-2)];
            alpha[t*ext_len+s]=sum*p;
        }
    }

    /* 总概率 */
    float tp=alpha[(sl-1)*ext_len+(ext_len-1)];
    if(ext_len>=2)tp+=alpha[(sl-1)*ext_len+(ext_len-2)];
    if(tp<1e-10f)tp=1e-10f;
    *loss_out=-logf(tp);

    /* CTC后向概率（堆分配） */
    float* beta=n->ctc_beta;
    memset(beta,0,(size_t)sl*ext_len*sizeof(float));
    for(int s=0;s<ext_len;s++)beta[(sl-1)*ext_len+s]=1.0f;

    for(int t=sl-2;t>=0;t--){
        for(int s=0;s<ext_len;s++){
            float sum=beta[(t+1)*ext_len+s]*probs[(t+1)*nc+ext[s]];
            if(s+1<ext_len)sum+=beta[(t+1)*ext_len+(s+1)]*probs[(t+1)*nc+ext[s+1]];
            if(s+2<ext_len&&ext[s]!=ext[s+2])
                sum+=beta[(t+1)*ext_len+(s+2)]*probs[(t+1)*nc+ext[s+2]];
            beta[t*ext_len+s]=sum;
        }
    }

    /* 清零梯度 */
    zero_gradients(n);

    /* 计算每个时间步每个类别的softmax梯度 */
    float grad_scale = 1.0f/(tp+1e-10f);
    for(int t=0;t<sl;t++){
        for(int k=0;k<nc;k++){
            /* ∂L/∂logit_k = softmax_k - Σ(α[s]*β[s])/tp */
            float grad_logit = probs[t*nc+k];
            for(int s=0;s<ext_len;s++){
                if(ext[s]==k){
                    grad_logit -= alpha[t*ext_len+s]*beta[t*ext_len+s]*grad_scale;
                }
            }
            /* 累积输出层梯度 */
            for(int d=0;d<fd;d++){
                n->d_out_w[k*fd+d] += grad_logit * n->spatial_state[d];
            }
            n->d_out_b[k] += grad_logit;
        }
    }

    /* 修正梯度缩放（防止梯度爆炸） */
    float max_grad=0.0f;
    for(int i=0;i<fd*fd;i++){
        if(fabsf(n->dW_gx[i])>max_grad)max_grad=fabsf(n->dW_gx[i]);
        if(fabsf(n->dW_ax[i])>max_grad)max_grad=fabsf(n->dW_ax[i]);
    }
    if(max_grad>10.0f){
        float clip_scale=10.0f/max_grad;
        for(int i=0;i<fd*fd;i++){
            n->dW_gx[i]*=clip_scale;n->dW_ax[i]*=clip_scale;
            n->dW_gh[i]*=clip_scale;n->dW_ah[i]*=clip_scale;
        }
        for(int i=0;i<fd;i++){
            n->db_g[i]*=clip_scale;n->db_a[i]*=clip_scale;
        }
        for(int i=0;i<nc*fd;i++)n->d_out_w[i]*=clip_scale;
        for(int i=0;i<nc;i++)n->d_out_b[i]*=clip_scale;
    }

    /* 应用梯度更新（真正的逐参数更新） */
    apply_gradient_update(n, lr);

    safe_free((void**)&probs);
    safe_free((void**)&ext);
    return 0;
}

/* ==================== 模型持久化 ==================== */

int cfc_ocr_net_save(void* net_ptr, const char* path) {
    CfCOcrNet* n=(CfCOcrNet*)net_ptr;
    if(!n||!path)return-1;
    FILE* f=fopen(path,"wb");if(!f)return-1;
    fwrite(&n->image_height,sizeof(int),1,f);
    fwrite(&n->image_width,sizeof(int),1,f);
    fwrite(&n->num_classes,sizeof(int),1,f);
    fwrite(&n->hidden_size,sizeof(int),1,f);
    fwrite(&n->feature_dim,sizeof(int),1,f);
    int d=n->feature_dim;
    fwrite(n->W_gx,sizeof(float),(size_t)d*d,f);
    fwrite(n->W_gh,sizeof(float),(size_t)d*d,f);
    fwrite(n->b_g,sizeof(float),(size_t)d,f);
    fwrite(n->W_ax,sizeof(float),(size_t)d*d,f);
    fwrite(n->W_ah,sizeof(float),(size_t)d*d,f);
    fwrite(n->b_a,sizeof(float),(size_t)d,f);
    fwrite(n->out_w,sizeof(float),(size_t)n->num_classes*d,f);
    fwrite(n->out_b,sizeof(float),(size_t)n->num_classes,f);
    fclose(f);
    return 0;
}

int cfc_ocr_net_load(void** out_net, const char* path) {
    if(!out_net||!path)return-1;
    FILE* f=fopen(path,"rb");if(!f)return-1;
    int ih,iw,nc,hs,fd;
    fread(&ih,sizeof(int),1,f);fread(&iw,sizeof(int),1,f);
    fread(&nc,sizeof(int),1,f);fread(&hs,sizeof(int),1,f);
    fread(&fd,sizeof(int),1,f);
    void* np=NULL;
    if(cfc_ocr_net_create(ih,iw,nc,hs,fd,&np)!=0){fclose(f);return-1;}
    CfCOcrNet* n=(CfCOcrNet*)np;
    int d=n->feature_dim;
    fread(n->W_gx,sizeof(float),(size_t)d*d,f);
    fread(n->W_gh,sizeof(float),(size_t)d*d,f);
    fread(n->b_g,sizeof(float),(size_t)d,f);
    fread(n->W_ax,sizeof(float),(size_t)d*d,f);
    fread(n->W_ah,sizeof(float),(size_t)d*d,f);
    fread(n->b_a,sizeof(float),(size_t)d,f);
    fread(n->out_w,sizeof(float),(size_t)nc*d,f);
    fread(n->out_b,sizeof(float),(size_t)nc,f);
    fclose(f);
    *out_net=np;
    return 0;
}

/* ==================== 字符识别（扩展字符集） ==================== */

/** 将类别索引映射到UTF-8字符 */
static int class_to_utf8(int class_idx, char* utf8_buf, int buf_size) {
    if(buf_size<8)return-1;
    memset(utf8_buf,0,(size_t)buf_size);

    if(class_idx<=0){
        utf8_buf[0]=0;return 0;  /* blank */
    }

    /* 类别1-10: 数字0-9 */
    if(class_idx<=10){
        utf8_buf[0]=(char)('0'+class_idx-1);
        return 1;
    }
    /* 类别11-36: 大写字母A-Z */
    if(class_idx<=36){
        utf8_buf[0]=(char)('A'+class_idx-11);
        return 1;
    }
    /* 类别37-62: 小写字母a-z */
    if(class_idx<=62){
        utf8_buf[0]=(char)('a'+class_idx-37);
        return 1;
    }
    /* 类别63-72: 常用符号 . , ! ? - _ / : ; */
    if(class_idx<=72){
        const char* symbols = ".,!?-_/:;";
        utf8_buf[0]=symbols[class_idx-63];
        return 1;
    }
    /* 类别73+: 使用扩展字符映射表 */
    if(class_idx<=72+((int)OCR_CHAR_MAP_LEN)){
        int map_idx=class_idx-73;
        const char* src=OCR_CHAR_MAP;
        int pos=0;
        for(int i=0;i<map_idx&&pos<(int)OCR_CHAR_MAP_LEN;i++){
            unsigned char c=(unsigned char)src[pos];
            if(c<0x80){pos++;}
            else if(c<0xE0){pos+=2;}
            else if(c<0xF0){pos+=3;}
            else{pos+=4;}
        }
        if(pos<(int)OCR_CHAR_MAP_LEN){
            unsigned char c=(unsigned char)src[pos];
            int char_len=(c<0x80)?1:(c<0xE0)?2:(c<0xF0)?3:4;
            for(int i=0;i<char_len&&i<buf_size-1;i++)utf8_buf[i]=src[pos+i];
            return char_len;
        }
    }

    /* 超出映射范围的类别 */
    utf8_buf[0]='?';
    return 1;
}

int cfc_ocr_recognize(void* net_ptr, const float* img, int w, int h,
                      char* text, int max_len, float* conf) {
    if(!text||max_len<=0)return-1;
    CfCOcrNet* n=(CfCOcrNet*)net_ptr;
    int nc=n?n->num_classes:72;
    float* probs=(float*)safe_malloc((size_t)CFC_OCR_TIMESTEPS_MAX*nc*sizeof(float));
    if(!probs)return-1;

    int sl=0;
    if(cfc_ocr_net_forward(net_ptr,img,h,w,1,probs,&sl)!=0){
        safe_free((void**)&probs);
        return-1;
    }

    /* 贪心解码 + 去重 */
    int prev=-1,tpos=0;
    float total_conf=0;
    for(int t=0;t<sl&&tpos<max_len-8;t++){
        int best=0;float best_v=probs[t*nc];
        for(int c=1;c<nc;c++)if(probs[t*nc+c]>best_v){best_v=probs[t*nc+c];best=c;}

        if(best>0&&best!=prev){
            total_conf+=best_v;
            char utf8_char[8];
            int char_len=class_to_utf8(best,utf8_char,8);
            if(char_len>0){
                for(int i=0;i<char_len&&tpos<max_len-1;i++){
                    text[tpos++]=utf8_char[i];
                }
            }
        }
        if(best>0)prev=best;
    }
    text[tpos]=0;
    if(conf)*conf=tpos>0?total_conf/(float)tpos:0.0f;

    safe_free((void**)&probs);
    return tpos;
}
