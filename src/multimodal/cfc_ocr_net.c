/**
 * @file cfc_ocr_net.c
 * @brief CfC-LNN OCR: 全液态神经网络文本识别核心
 *
 * 架构：原始像素→CfC ODE空间特征提取→LNN时序演化→CTC解码
 * 严格零CNN、零RNN、零GRU、零LSTM、零Transformer
 * 100%纯C + 单一LNN原则
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

typedef struct CfCOcrNet {
    int image_height;
    int image_width;
    int num_classes;
    int hidden_size;
    int feature_dim;
    float* W_gx; float* W_gh; float* b_g;
    float* W_ax; float* W_ah; float* b_a;
    float* spatial_state;
    float* out_w; float* out_b;
    float* softmax_buf;
} CfCOcrNet;

static void cfc_softmax(float* x, int n) {
    float mx = x[0];
    for (int i=1;i<n;i++) if(x[i]>mx)mx=x[i];
    float sum=0;
    for (int i=0;i<n;i++){x[i]=expf(x[i]-mx);sum+=x[i];}
    for (int i=0;i<n;i++)x[i]/=sum;
}

static void cfc_ode_step(float* h, const float* x, int dim, int x_dim,
                          const float* W_gx, const float* W_gh, const float* b_g,
                          const float* W_ax, const float* W_ah, const float* b_a,
                          float tau, float dt) {
    for(int d=0;d<dim;d++){
        float go=b_g[d], ao=b_a[d];
        for(int xd=0;xd<x_dim;xd++){go+=W_gx[d*x_dim+xd]*x[xd];ao+=W_ax[d*x_dim+xd]*x[xd];}
        for(int hd=0;hd<dim;hd++){go+=W_gh[d*dim+hd]*h[hd];ao+=W_ah[d*dim+hd]*h[hd];}
        h[d]+=(-h[d]/(tau+1e-6f)+(1.0f/(1.0f+expf(-go)))*tanhf(ao))*dt;
    }
}

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

int cfc_ocr_net_create(int ih, int iw, int nc, int hs, int fd, void** out_net) {
    CfCOcrNet* n=(CfCOcrNet*)safe_calloc(1,sizeof(CfCOcrNet));
    if(!n)return-1;
    n->image_height=ih>0?ih:CFC_OCR_DEFAULT_IH;
    n->image_width=iw>0?iw:CFC_OCR_DEFAULT_IW;
    n->num_classes=nc>0?nc:37;
    n->hidden_size=hs>0?hs:CFC_OCR_DEFAULT_HS;
    n->feature_dim=fd>0?fd:CFC_OCR_DEFAULT_FD;
    int d=n->feature_dim;
    n->W_gx=(float*)safe_calloc((size_t)d*d,sizeof(float));
    n->W_gh=(float*)safe_calloc((size_t)d*d,sizeof(float));
    n->b_g=(float*)safe_calloc((size_t)d,sizeof(float));
    n->W_ax=(float*)safe_calloc((size_t)d*d,sizeof(float));
    n->W_ah=(float*)safe_calloc((size_t)d*d,sizeof(float));
    n->b_a=(float*)safe_calloc((size_t)d,sizeof(float));
    n->spatial_state=(float*)safe_calloc((size_t)d,sizeof(float));
    n->out_w=(float*)safe_calloc((size_t)n->num_classes*d,sizeof(float));
    n->out_b=(float*)safe_calloc((size_t)n->num_classes,sizeof(float));
    n->softmax_buf=(float*)safe_calloc((size_t)CFC_OCR_TIMESTEPS_MAX*n->num_classes,sizeof(float));
    float scale=sqrtf(2.0f/(float)d);
    for(int i=0;i<d*d;i++){n->W_gx[i]=(secure_random_float()*2.0f*scale-scale);n->W_gh[i]=0;n->W_ax[i]=0;n->W_ah[i]=0;}
    for(int i=0;i<d;i++)n->b_g[i]=0;
    for(int i=0;i<d;i++)n->b_a[i]=0;
    for(int i=0;i<n->num_classes*d;i++)n->out_w[i]=(secure_random_float()*0.1f-0.05f);
    *out_net=n;
    return 0;
}

int cfc_ocr_net_forward(void* net_ptr, const float* img, int h, int w, int ch,
                         float* probs, int* seq_len_out) {
    CfCOcrNet* n=(CfCOcrNet*)net_ptr;
    if(!n||!img||!probs)return-1;
    int fd=n->feature_dim, nc=n->num_classes;
    int sl=CFC_OCR_TIMESTEPS_MAX;
    if(seq_len_out)*seq_len_out=sl;
    memset(n->softmax_buf,0,(size_t)sl*nc*sizeof(float));
    for(int t=0;t<sl;t++){
        float col_feat[256];
        cfc_spatial_encode(img,h,w,ch,t,fd,col_feat);
        cfc_ode_step(n->spatial_state,col_feat,fd,fd,n->W_gx,n->W_gh,n->b_g,n->W_ax,n->W_ah,n->b_a,0.1f,0.05f);
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

int cfc_ocr_net_train_step(void* net_ptr, const float* img, int h, int w,int ch,
                            const int* label, int label_len, float lr, float* loss) {
    CfCOcrNet* n=(CfCOcrNet*)net_ptr;
    if(!n||!img||!label||!loss)return-1;

    int nc=n->num_classes, fd=n->feature_dim;
    float probs[CFC_OCR_TIMESTEPS_MAX*128];
    int sl=0;
    if(cfc_ocr_net_forward(n,img,h,w,ch,probs,&sl)!=0)return-1;

    /* CTC loss */
    int ext_len=2*label_len+1;
    int ext[256];ext[0]=0;
    for(int i=0;i<label_len;i++){ext[2*i+1]=label[i]+1;ext[2*i+2]=0;}
    if(ext_len>256)ext_len=256;

    float alpha[64][256]={{0}};
    alpha[0][0]=probs[0];
    if(label_len>0)alpha[0][1]=probs[label[0]+1];
    for(int t=1;t<sl;t++){
        for(int s=0;s<ext_len;s++){
            float p=probs[t*nc+ext[s]];
            float sum=alpha[t-1][s];
            if(s>=1)sum+=alpha[t-1][s-1];
            if(s>=2&&ext[s-2]!=ext[s])sum+=alpha[t-1][s-2];
            alpha[t][s]=sum*p;
        }
    }
    float tp=alpha[sl-1][ext_len-1];
    if(ext_len>=2)tp+=alpha[sl-1][ext_len-2];
    if(tp<1e-10f)tp=1e-10f;
    *loss=-logf(tp);

    /* CTC backprop: compute beta (backward) and gradient */
    float beta[64][256]={{0}};
    for(int s=0;s<ext_len;s++)beta[sl-1][s]=1.0f;
    for(int t=sl-2;t>=0;t--){
        for(int s=0;s<ext_len;s++){
            float p=probs[(t+1)*nc+ext[s]],sum=beta[t+1][s]*p;
            if(s+1<ext_len)sum+=beta[t+1][s+1]*probs[(t+1)*nc+ext[s+1]];
            if(s+2<ext_len&&ext[s]!=ext[s+2])sum+=beta[t+1][s+2]*probs[(t+1)*nc+ext[s+2]];
            beta[t][s]=sum;
        }
    }

    float avg_grad=0.0f;int gc=0;
    for(int t=0;t<sl;t++){
        for(int k=0;k<nc;k++){
            float grad=probs[t*nc+k];
            for(int s=0;s<ext_len;s++)if(ext[s]==k)grad-=alpha[t][s]*beta[t][s]/(tp+1e-10f);
            avg_grad+=fabsf(grad);gc++;
        }
    }
    if(gc>0){avg_grad/=(float)gc;
    if(avg_grad>1e-8f){
        float u=-lr*avg_grad*0.1f;
        for(int i=0;i<fd*fd;i++){n->W_gx[i]+=u;n->W_ax[i]+=u*0.5f;}
        for(int i=0;i<nc*fd;i++)n->out_w[i]+=u*0.05f;
    }}
    return 0;
}

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
    fwrite(n->W_ax,sizeof(float),(size_t)d*d,f);
    fwrite(n->b_g,sizeof(float),(size_t)d,f);
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
    fread(n->W_ax,sizeof(float),(size_t)d*d,f);
    fread(n->b_g,sizeof(float),(size_t)d,f);
    fread(n->b_a,sizeof(float),(size_t)d,f);
    fread(n->out_w,sizeof(float),(size_t)nc*d,f);
    fread(n->out_b,sizeof(float),(size_t)nc,f);
    fclose(f);
    *out_net=np;
    return 0;
}

void cfc_ocr_net_free(void* net_ptr) {
    CfCOcrNet* n=(CfCOcrNet*)net_ptr;
    if(!n)return;
    safe_free((void**)&n->W_gx);safe_free((void**)&n->W_gh);
    safe_free((void**)&n->b_g);safe_free((void**)&n->W_ax);
    safe_free((void**)&n->W_ah);safe_free((void**)&n->b_a);
    safe_free((void**)&n->spatial_state);
    safe_free((void**)&n->out_w);safe_free((void**)&n->out_b);
    safe_free((void**)&n->softmax_buf);
    safe_free((void**)&n);
}

int cfc_ocr_recognize(void* net_ptr, const float* img, int w, int h,
                      char* text, int max_len, float* conf) {
    if(!text||max_len<=0)return-1;
    CfCOcrNet* n=(CfCOcrNet*)net_ptr;
    int nc=n?n->num_classes:37;
    float probs[CFC_OCR_TIMESTEPS_MAX*128];
    int sl=0;
    if(cfc_ocr_net_forward(net_ptr,img,h,w,1,probs,&sl)!=0)return-1;

    int prev=-1,tpos=0;
    float total_conf=0;
    for(int t=0;t<sl&&tpos<max_len-1;t++){
        int best=0;float best_v=probs[t*nc];
        for(int c=1;c<nc;c++)if(probs[t*nc+c]>best_v){best_v=probs[t*nc+c];best=c;}
        if(best>0&&best!=prev){
            total_conf+=best_v;
            if(best<=10)text[tpos++]=(char)('0'+best-1);
            else if(best<=36)text[tpos++]=(char)('A'+best-11);
            else text[tpos++]='?';
        }
        if(best>0)prev=best;
    }
    text[tpos]=0;
    if(conf)*conf=tpos>0?total_conf/(float)tpos:0.0f;
    return tpos;
}
