#include "selflnn/core/laplace_features.h"
#include "selflnn/utils/secure_random.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define FLOAT_EPS 1e-10f
#define FLOAT_MAX 1e10f

/* ==================== 内部工具函数 ==================== */

static float* alloc_2d(int rows, int cols) {
    return (float*)calloc((size_t)rows * cols, sizeof(float));
}

static void mat_mul(const float* A, const float* B, float* C, int m, int n, int k) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++) {
            float sum = 0.0f;
            for (int t = 0; t < n; t++)
                sum += A[i * n + t] * B[t * k + j];
            C[i * k + j] = sum;
        }
}

static void mat_transpose(const float* A, float* AT, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            AT[j * rows + i] = A[i * cols + j];
}

static void mat_ident(float* A, int n) {
    memset(A, 0, (size_t)n * n * sizeof(float));
    for (int i = 0; i < n; i++) A[i * n + i] = 1.0f;
}

static float vec_dot(const float* a, const float* b, int n) {
    float s = 0.0f; for (int i = 0; i < n; i++) s += a[i] * b[i]; return s;
}

static float vec_norm2(const float* a, int n) {
    return sqrtf(fmaxf(vec_dot(a, a, n), 0.0f));
}

static float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static float gaussian_2d(int x, int y, float sigma) {
    return expf(-(float)(x * x + y * y) / (2.0f * sigma * sigma));
}

/* ==================== Jacobi特征值分解 ==================== */

static void jacobi_eigen(float* A, float* V, float* eigenvalues, int n, int max_iter) {
    mat_ident(V, n);
    /* A矩阵原地运算，Jacobi旋转直接修改A的非对角元素 */

    for (int iter = 0; iter < max_iter; iter++) {
        float max_off = 0.0f; int p = 0, q = 0;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++) {
                float val = fabsf(A[i * n + j]);
                if (val > max_off) { max_off = val; p = i; q = j; }
            }
        if (max_off < 1e-12f) break;

        float app = A[p * n + p], aqq = A[q * n + q], apq = A[p * n + q];
        float theta = 0.5f * atan2f(2.0f * apq, aqq - app);
        float c = cosf(theta), s = sinf(theta);

        for (int i = 0; i < n; i++) {
            float vip = V[i * n + p], viq = V[i * n + q];
            V[i * n + p] = c * vip - s * viq;
            V[i * n + q] = s * vip + c * viq;
        }

        for (int i = 0; i < n; i++) {
            if (i != p && i != q) {
                float aip = A[i * n + p], aiq = A[i * n + q];
                A[i * n + p] = c * aip - s * aiq;
                A[p * n + i] = A[i * n + p];
                A[i * n + q] = s * aip + c * aiq;
                A[q * n + i] = A[i * n + q];
            }
        }
        float new_app = c * c * app + s * s * aqq - 2.0f * s * c * apq;
        float new_aqq = s * s * app + c * c * aqq + 2.0f * s * c * apq;
        A[p * n + p] = new_app; A[q * n + q] = new_aqq;
        A[p * n + q] = 0.0f; A[q * n + p] = 0.0f;
    }

    for (int i = 0; i < n; i++) {
        eigenvalues[i] = A[i * n + i];
    }

    for (int i = 0; i < n - 1; i++) {
        int max_idx = i;
        for (int j = i + 1; j < n; j++)
            if (eigenvalues[j] > eigenvalues[max_idx]) max_idx = j;
        if (max_idx != i) {
            float tmp = eigenvalues[i]; eigenvalues[i] = eigenvalues[max_idx]; eigenvalues[max_idx] = tmp;
            for (int k = 0; k < n; k++) {
                tmp = V[k * n + i]; V[k * n + i] = V[k * n + max_idx]; V[k * n + max_idx] = tmp;
            }
        }
    }
}

/* ==================== A01.3.1 多尺度拉普拉斯金字塔 ==================== */

int laplace_build_gaussian_pyramid(const float* input, int width, int height, int channels, int num_levels, LaplacianPyramid* pyramid) {
    if (!input || !pyramid || width < 2 || height < 2 || num_levels < 1 || channels < 1) return -1;

    pyramid->num_levels = num_levels;
    pyramid->channels = channels;
    pyramid->pyramid = (float**)calloc((size_t)num_levels, sizeof(float*));
    pyramid->widths = (int*)calloc((size_t)num_levels, sizeof(int));
    pyramid->heights = (int*)calloc((size_t)num_levels, sizeof(int));
    if (!pyramid->pyramid || !pyramid->widths || !pyramid->heights) {
        laplace_pyramid_free(pyramid); return -1;
    }

    int kw = 5, kh = 5;
    float sigma = 1.0f;
    float* kernel = (float*)calloc((size_t)(kw * kh), sizeof(float));
    if (!kernel) { laplace_pyramid_free(pyramid); return -1; }
    float ksum = 0.0f;
    for (int ky = -kh / 2; ky <= kh / 2; ky++)
        for (int kx = -kw / 2; kx <= kw / 2; kx++) {
            kernel[(ky + kh / 2) * kw + (kx + kw / 2)] = gaussian_2d(kx, ky, sigma);
            ksum += kernel[(ky + kh / 2) * kw + (kx + kw / 2)];
        }
    for (int i = 0; i < kw * kh; i++) kernel[i] /= ksum;

    int w = width, h = height;
    size_t total = (size_t)w * h * channels;
    pyramid->widths[0] = w; pyramid->heights[0] = h;
    pyramid->pyramid[0] = (float*)malloc(total * sizeof(float));
    if (!pyramid->pyramid[0]) { free(kernel); laplace_pyramid_free(pyramid); return -1; }
    memcpy(pyramid->pyramid[0], input, total * sizeof(float));

    for (int level = 1; level < num_levels; level++) {
        int pw = pyramid->widths[level - 1], ph = pyramid->heights[level - 1];
        float* prev = pyramid->pyramid[level - 1];

        int nw = pw > 1 ? pw / 2 : 1;
        int nh = ph > 1 ? ph / 2 : 1;
        if (nw < 1) nw = 1; if (nh < 1) nh = 1;
        pyramid->widths[level] = nw; pyramid->heights[level] = nh;

        size_t ntotal = (size_t)nw * nh * channels;
        pyramid->pyramid[level] = (float*)calloc(ntotal, sizeof(float));
        if (!pyramid->pyramid[level]) { free(kernel); laplace_pyramid_free(pyramid); return -1; }

        float* blurred = (float*)calloc((size_t)pw * ph * channels, sizeof(float));
        if (!blurred) { free(kernel); laplace_pyramid_free(pyramid); return -1; }

        for (int c = 0; c < channels; c++) {
            for (int y = 0; y < ph; y++)
                for (int x = 0; x < pw; x++) {
                    float sum = 0.0f, wsum = 0.0f;
                    for (int ky = -kh / 2; ky <= kh / 2; ky++)
                        for (int kx = -kw / 2; kx <= kw / 2; kx++) {
                            int ix = x + kx, iy = y + ky;
                            if (ix >= 0 && ix < pw && iy >= 0 && iy < ph) {
                                sum += prev[(iy * pw + ix) * channels + c] * kernel[(ky + kh / 2) * kw + (kx + kw / 2)];
                                wsum += kernel[(ky + kh / 2) * kw + (kx + kw / 2)];
                            }
                        }
                    blurred[(y * pw + x) * channels + c] = (wsum > 0) ? sum / wsum : 0.0f;
                }
        }

        for (int c = 0; c < channels; c++)
            for (int y = 0; y < nh; y++)
                for (int x = 0; x < nw; x++) {
                    int sx = (x * pw) / nw, sy = (y * ph) / nh;
                    if (sx >= pw) sx = pw - 1; if (sy >= ph) sy = ph - 1;
                    pyramid->pyramid[level][(y * nw + x) * channels + c] = blurred[(sy * pw + sx) * channels + c];
                }
        free(blurred);
    }
    free(kernel);
    return 0;
}

int laplace_build_laplacian_pyramid(const float* input, int width, int height, int channels, int num_levels, LaplacianPyramid* pyramid) {
    if (!input || !pyramid || num_levels < 2) return -1;

    memset(pyramid, 0, sizeof(LaplacianPyramid));
    if (laplace_build_gaussian_pyramid(input, width, height, channels, num_levels, pyramid) != 0)
        return -1;

    for (int level = 0; level < num_levels - 1; level++) {
        int cw = pyramid->widths[level], ch = pyramid->heights[level];
        int nw = pyramid->widths[level + 1], nh = pyramid->heights[level + 1];
        float* curr = pyramid->pyramid[level];
        float* next = pyramid->pyramid[level + 1];

        float* upsampled = (float*)calloc((size_t)cw * ch * channels, sizeof(float));
        if (!upsampled) return -1;

        for (int c = 0; c < channels; c++)
            for (int y = 0; y < ch; y++)
                for (int x = 0; x < cw; x++) {
                    float sx = (float)x * nw / (float)cw, sy = (float)y * nh / (float)ch;
                    int ix = (int)sx, iy = (int)sy;
                    float fx = sx - ix, fy = sy - iy;
                    if (ix >= nw - 1) ix = nw - 2;
                    if (iy >= nh - 1) iy = nh - 2;
                    if (ix < 0) ix = 0; if (iy < 0) iy = 0;
                    float v00 = next[(iy * nw + ix) * channels + c];
                    float v10 = next[(iy * nw + ix + 1) * channels + c];
                    float v01 = next[((iy + 1) * nw + ix) * channels + c];
                    float v11 = next[((iy + 1) * nw + ix + 1) * channels + c];
                    float v0 = v00 + fx * (v10 - v00);
                    float v1 = v01 + fx * (v11 - v01);
                    upsampled[(y * cw + x) * channels + c] = v0 + fy * (v1 - v0);
                }

        for (size_t i = 0; i < (size_t)cw * ch * channels; i++)
            pyramid->pyramid[level][i] = curr[i] - upsampled[i];

        free(upsampled);
    }
    return 0;
}

int laplace_reconstruct_from_pyramid(const LaplacianPyramid* pyramid, float* output, int width, int height) {
    if (!pyramid || !output || pyramid->num_levels < 1) return -1;
    (void)width; (void)height;

    int n = pyramid->num_levels;
    int channels = pyramid->channels;

    float** levels = (float**)calloc((size_t)n, sizeof(float*));
    if (!levels) return -1;

    for (int l = 0; l < n; l++) {
        int w = pyramid->widths[l], h = pyramid->heights[l];
        size_t sz = (size_t)w * h * channels;
        levels[l] = (float*)malloc(sz * sizeof(float));
        if (!levels[l]) {
            for (int k = 0; k < l; k++) free(levels[k]);
            free(levels); return -1;
        }
        memcpy(levels[l], pyramid->pyramid[l], sz * sizeof(float));
    }

    for (int l = n - 2; l >= 0; l--) {
        int cw = pyramid->widths[l], ch = pyramid->heights[l];
        int nw = pyramid->widths[l + 1], nh = pyramid->heights[l + 1];
        float* upsampled = (float*)calloc((size_t)cw * ch * channels, sizeof(float));
        if (!upsampled) {
            for (int k = 0; k < n; k++) free(levels[k]);
            free(levels); return -1;
        }

        for (int c = 0; c < channels; c++)
            for (int y = 0; y < ch; y++)
                for (int x = 0; x < cw; x++) {
                    float sx = (float)x * nw / (float)cw, sy = (float)y * nh / (float)ch;
                    int ix = (int)sx, iy = (int)sy;
                    float fx = sx - ix, fy = sy - iy;
                    if (ix >= nw - 1) ix = nw - 2;
                    if (iy >= nh - 1) iy = nh - 2;
                    if (ix < 0) ix = 0; if (iy < 0) iy = 0;
                    float v00 = levels[l + 1][(iy * nw + ix) * channels + c];
                    float v10 = levels[l + 1][(iy * nw + ix + 1) * channels + c];
                    float v01 = levels[l + 1][((iy + 1) * nw + ix) * channels + c];
                    float v11 = levels[l + 1][((iy + 1) * nw + ix + 1) * channels + c];
                    float v0 = v00 + fx * (v10 - v00);
                    float v1 = v01 + fx * (v11 - v01);
                    upsampled[(y * cw + x) * channels + c] = v0 + fy * (v1 - v0);
                }

        for (size_t i = 0; i < (size_t)cw * ch * channels; i++)
            levels[l][i] += upsampled[i];

        free(upsampled);
    }

    size_t out_size = (size_t)pyramid->widths[0] * pyramid->heights[0] * channels;
    memcpy(output, levels[0], out_size * sizeof(float));
    for (int k = 0; k < n; k++) free(levels[k]);
    free(levels);
    return 0;
}

void laplace_pyramid_free(LaplacianPyramid* pyramid) {
    if (!pyramid) return;
    if (pyramid->pyramid) {
        for (int i = 0; i < pyramid->num_levels; i++) {
            free(pyramid->pyramid[i]);
        }
        free(pyramid->pyramid);
    }
    free(pyramid->widths);
    free(pyramid->heights);
    memset(pyramid, 0, sizeof(LaplacianPyramid));
}

/* ==================== A01.3.1 拉普拉斯图卷积 ==================== */

int laplace_graph_laplacian_build(const float* adjacency, int n, int use_normalized, GraphLaplacian* gl) {
    if (!adjacency || !gl || n < 1) return -1;
    memset(gl, 0, sizeof(GraphLaplacian));
    gl->n = n;
    gl->use_normalized = use_normalized;

    gl->laplacian_matrix = (float*)calloc((size_t)n * n, sizeof(float));
    if (!gl->laplacian_matrix) return -1;

    float* degree = (float*)calloc((size_t)n, sizeof(float));
    if (!degree) { free(gl->laplacian_matrix); return -1; }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float w = adjacency[i * n + j];
            if (i != j && w > 0) degree[i] += w;
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float w = adjacency[i * n + j];
            if (i == j) {
                gl->laplacian_matrix[i * n + j] = degree[i];
            } else if (w > 0) {
                gl->laplacian_matrix[i * n + j] = -w;
            }
        }
    }

    if (use_normalized) {
        for (int i = 0; i < n; i++) {
            float di = (degree[i] > FLOAT_EPS) ? 1.0f / sqrtf(degree[i]) : 0.0f;
            for (int j = 0; j < n; j++) {
                float dj = (degree[j] > FLOAT_EPS) ? 1.0f / sqrtf(degree[j]) : 0.0f;
                if (i == j) {
                    gl->laplacian_matrix[i * n + j] = 1.0f - (adjacency[i * n + i] > 0 ? 1.0f : 0.0f);
                } else if (adjacency[i * n + j] > 0) {
                    gl->laplacian_matrix[i * n + j] = -adjacency[i * n + j] * di * dj;
                }
            }
        }
    }

    free(degree);
    gl->eigenvalues = NULL;
    gl->eigenvectors = NULL;
    return 0;
}

int laplace_graph_laplacian_decompose(GraphLaplacian* gl) {
    if (!gl || !gl->laplacian_matrix || gl->n < 1) return -1;

    size_t n = (size_t)gl->n;
    free(gl->eigenvalues); gl->eigenvalues = NULL;
    free(gl->eigenvectors); gl->eigenvectors = NULL;

    gl->eigenvalues = (float*)calloc(n, sizeof(float));
    gl->eigenvectors = (float*)calloc(n * n, sizeof(float));
    if (!gl->eigenvalues || !gl->eigenvectors) {
        free(gl->eigenvalues); free(gl->eigenvectors);
        gl->eigenvalues = NULL; gl->eigenvectors = NULL;
        return -1;
    }

    /* R4-001修复: 将拉普拉斯矩阵复制到A后再进行特征分解 */
    float* A = (float*)malloc(n * n * sizeof(float));
    if (!A) return -1;
    memcpy(A, gl->laplacian_matrix, n * n * sizeof(float));
    jacobi_eigen(A, gl->eigenvectors, gl->eigenvalues, gl->n, 100);
    free(A);
    return 0;
}

int laplace_graph_conv_chebyshev(const GraphLaplacian* gl, const float* features, int num_features, int filter_size, const float* coeffs, float* output) {
    if (!gl || !features || !coeffs || !output || filter_size < 1) return -1;
    int n = gl->n;
    if (n < 1 || num_features < 1) return -1;

    float lambda_max = 2.0f;
    float a = lambda_max / 2.0f;

    for (int f = 0; f < num_features; f++) {
        size_t offset = (size_t)f * n;

        float* T0 = (float*)malloc((size_t)n * sizeof(float));
        float* T1 = (float*)malloc((size_t)n * sizeof(float));
        float* Tk = (float*)malloc((size_t)n * sizeof(float));
        if (!T0 || !T1 || !Tk) { free(T0); free(T1); free(Tk); return -1; }

        for (int i = 0; i < n; i++) T0[i] = features[offset + i];
        memset(T1, 0, (size_t)n * sizeof(float));

        for (int i = 0; i < n; i++) {
            float sum = 0.0f;
            for (int j = 0; j < n; j++) {
                float L_ij = (i == j) ? gl->laplacian_matrix[i * n + j] / a
                                      : gl->laplacian_matrix[i * n + j] / a;
                sum += L_ij * features[offset + j];
            }
            T1[i] = sum;
        }

        for (int i = 0; i < n; i++)
            output[offset + i] = coeffs[0] * T0[i] + coeffs[1] * T1[i];

        for (int k = 2; k < filter_size; k++) {
            for (int i = 0; i < n; i++) {
                float sum = 0.0f;
                for (int j = 0; j < n; j++) {
                    float L_ij = (i == j) ? gl->laplacian_matrix[i * n + j] / a
                                          : gl->laplacian_matrix[i * n + j] / a;
                    sum += L_ij * T1[j];
                }
                Tk[i] = 2.0f * sum - T0[i];
            }
            for (int i = 0; i < n; i++)
                output[offset + i] += coeffs[k] * Tk[i];
            float* tmp = T0; T0 = T1; T1 = Tk; Tk = tmp;
        }

        free(T0); free(T1); free(Tk);
    }
    return 0;
}

int laplace_graph_conv_spectral(const GraphLaplacian* gl, const float* features, int num_features, const float* spectral_filter, float* output) {
    if (!gl || !features || !spectral_filter || !output) return -1;
    if (!gl->eigenvalues || !gl->eigenvectors) return -1;
    int n = gl->n;
    if (n < 1 || num_features < 1) return -1;

    float* UT = (float*)malloc((size_t)n * n * sizeof(float));
    if (!UT) return -1;
    mat_transpose(gl->eigenvectors, UT, n, n);

    for (int f = 0; f < num_features; f++) {
        size_t offset = (size_t)f * n;

        float* coeff = (float*)malloc((size_t)n * sizeof(float));
        if (!coeff) { free(UT); return -1; }
        for (int i = 0; i < n; i++) coeff[i] = vec_dot(&UT[i * n], &features[offset], n);

        for (int i = 0; i < n; i++) coeff[i] *= spectral_filter[i];

        for (int i = 0; i < n; i++) {
            output[offset + i] = 0.0f;
            for (int j = 0; j < n; j++)
                output[offset + i] += gl->eigenvectors[i * n + j] * coeff[j];
        }
        free(coeff);
    }
    free(UT);
    return 0;
}

void laplace_graph_laplacian_free(GraphLaplacian* gl) {
    if (!gl) return;
    free(gl->laplacian_matrix);
    free(gl->eigenvalues);
    free(gl->eigenvectors);
    memset(gl, 0, sizeof(GraphLaplacian));
}

/* ==================== A01.3.1 拉普拉斯特征映射 ==================== */

int laplace_eigenmap_compute(const float* data, int num_points, int data_dim, int embedding_dim, float sigma, LaplacianEigenmap* map) {
    if (!data || !map || num_points < 2 || data_dim < 1 || embedding_dim < 1 || embedding_dim >= num_points) return -1;
    memset(map, 0, sizeof(LaplacianEigenmap));

    map->num_points = num_points;
    map->embedding_dim = embedding_dim;
    map->data_dim = data_dim;
    map->sigma = sigma;

    map->distance_matrix = (float*)calloc((size_t)num_points * num_points, sizeof(float));
    if (!map->distance_matrix) return -1;

    for (int i = 0; i < num_points; i++)
        for (int j = 0; j < num_points; j++) {
            float d2 = 0.0f;
            for (int k = 0; k < data_dim; k++) {
                float diff = data[i * data_dim + k] - data[j * data_dim + k];
                d2 += diff * diff;
            }
            map->distance_matrix[i * num_points + j] = d2;
        }

    float* adjacency = (float*)calloc((size_t)num_points * num_points, sizeof(float));
    if (!adjacency) { free(map->distance_matrix); return -1; }

    for (int i = 0; i < num_points; i++) {
        for (int j = 0; j < num_points; j++) {
            if (i == j) continue;
            float w = expf(-map->distance_matrix[i * num_points + j] / (2.0f * sigma * sigma));
            adjacency[i * num_points + j] = w;
        }
    }

    for (int i = 0; i < num_points; i++) {
        float sum = 0.0f;
        for (int j = 0; j < num_points; j++) sum += adjacency[i * num_points + j];
        if (sum > FLOAT_EPS)
            for (int j = 0; j < num_points; j++)
                adjacency[i * num_points + j] /= sum;
    }

    GraphLaplacian gl;
    if (laplace_graph_laplacian_build(adjacency, num_points, 0, &gl) != 0) {
        free(adjacency); free(map->distance_matrix); return -1;
    }

    if (laplace_graph_laplacian_decompose(&gl) != 0) {
        laplace_graph_laplacian_free(&gl); free(adjacency); return -1;
    }

    map->embedding = (float*)calloc((size_t)num_points * embedding_dim, sizeof(float));
    map->eigenvalues = (float*)calloc((size_t)embedding_dim, sizeof(float));
    if (!map->embedding || !map->eigenvalues) {
        free(adjacency); laplace_graph_laplacian_free(&gl); return -1;
    }

    for (int i = 0; i < num_points; i++)
        for (int j = 1; j <= embedding_dim; j++) {
            int ev_idx = num_points - j;
            if (ev_idx >= 0) {
                map->embedding[i * embedding_dim + (j - 1)] = gl.eigenvectors[i * num_points + ev_idx];
                map->eigenvalues[j - 1] = gl.eigenvalues[ev_idx];
            }
        }

    laplace_graph_laplacian_free(&gl);
    free(adjacency);
    return 0;
}

int laplace_eigenmap_transform(const LaplacianEigenmap* map, const float* new_point, float* embedding) {
    if (!map || !new_point || !embedding || !map->distance_matrix) return -1;
    int n = map->num_points, d = map->data_dim, em = map->embedding_dim;
    if (n < 1 || d < 1 || em < 1) return -1;

    float* weights = (float*)calloc((size_t)n, sizeof(float));
    if (!weights) return -1;
    float wsum = 0.0f;

    for (int i = 0; i < n; i++) {
        float d2 = 0.0f;
        for (int k = 0; k < d; k++) {
            float diff = new_point[k] - map->distance_matrix[i * d + k];
            d2 += diff * diff;  /* R4-003修复: 对每个i都累加距离 */
        }
        weights[i] = expf(-d2 / (2.0f * map->sigma * map->sigma));
        wsum += weights[i];
    }
    if (wsum > FLOAT_EPS)
        for (int i = 0; i < n; i++) weights[i] /= wsum;

    memset(embedding, 0, (size_t)em * sizeof(float));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < em; j++)
            embedding[j] += weights[i] * map->embedding[i * em + j];

    free(weights);
    return 0;
}

void laplace_eigenmap_free(LaplacianEigenmap* map) {
    if (!map) return;
    free(map->embedding);
    free(map->eigenvalues);
    free(map->distance_matrix);
    memset(map, 0, sizeof(LaplacianEigenmap));
}

/* ==================== A01.3.2 拉普拉斯稀疏编码 ==================== */

LaplacianSparseCoder* laplace_sparse_coder_create(int data_dim, int dict_size, float lambda, float laplacian_reg) {
    if (data_dim < 1 || dict_size < 1) return NULL;

    LaplacianSparseCoder* coder = (LaplacianSparseCoder*)calloc(1, sizeof(LaplacianSparseCoder));
    if (!coder) return NULL;

    coder->data_dim = data_dim;
    coder->dict_size = dict_size;
    coder->lambda = lambda > 0.0f ? lambda : 0.1f;
    coder->laplacian_reg = laplacian_reg;
    coder->use_laplacian = laplacian_reg > 0.0f;

    coder->dictionary = (float*)calloc((size_t)data_dim * dict_size, sizeof(float));
    coder->gram_matrix = (float*)calloc((size_t)dict_size * dict_size, sizeof(float));
    if (!coder->dictionary || !coder->gram_matrix) {
        laplace_sparse_coder_free(coder); return NULL;
    }

    for (int i = 0; i < dict_size; i++) {
        for (int j = 0; j < data_dim; j++) {
            /* K-012修复：安全随机数 */
            coder->dictionary[i * data_dim + j] = secure_random_float() * 2.0f - 1.0f;
        }
        float norm = vec_norm2(&coder->dictionary[i * data_dim], data_dim);
        if (norm > FLOAT_EPS)
            for (int j = 0; j < data_dim; j++)
                coder->dictionary[i * data_dim + j] /= norm;
    }

    if (coder->use_laplacian) {
        coder->laplacian_matrix = (float*)calloc((size_t)dict_size * dict_size, sizeof(float));
        if (!coder->laplacian_matrix) { laplace_sparse_coder_free(coder); return NULL; }
    }

    return coder;
}

static void soft_threshold(float* x, float lambda, int n) {
    for (int i = 0; i < n; i++) {
        if (x[i] > lambda) x[i] -= lambda;
        else if (x[i] < -lambda) x[i] += lambda;
        else x[i] = 0.0f;
    }
}

int laplace_sparse_encode(const LaplacianSparseCoder* coder, const float* data_point, float* code, int max_iter, float tol) {
    if (!coder || !data_point || !code || max_iter < 1) return -1;
    int d = coder->data_dim, k = coder->dict_size;

    memset(code, 0, (size_t)k * sizeof(float));

    float* residual = (float*)malloc((size_t)d * sizeof(float));
    float* DTr = (float*)malloc((size_t)k * sizeof(float));
    float* Lc = (float*)malloc((size_t)k * sizeof(float));
    if (!residual || !DTr || !Lc) { free(residual); free(DTr); free(Lc); return -1; }

    float t = 1.0f / (1.0f + coder->laplacian_reg);

    for (int iter = 0; iter < max_iter; iter++) {
        for (int j = 0; j < d; j++) residual[j] = data_point[j];
        for (int j = 0; j < k; j++) {
            float val = 0.0f;
            for (int p = 0; p < d; p++)
                val -= coder->dictionary[j * d + p] * residual[p];
            val = -val;
            DTr[j] = val;
        }

        if (coder->use_laplacian) {
            memset(Lc, 0, (size_t)k * sizeof(float));
            for (int j = 0; j < k; j++) {
                float sum = 0.0f;
                for (int p = 0; p < k; p++)
                    sum += coder->laplacian_matrix[j * k + p] * code[p];
                Lc[j] = sum;
            }
            for (int j = 0; j < k; j++)
                code[j] -= t * (DTr[j] + coder->laplacian_reg * Lc[j]);
        } else {
            for (int j = 0; j < k; j++)
                code[j] -= t * DTr[j];
        }

        soft_threshold(code, t * coder->lambda, k);

        float conv = 0.0f;
        for (int j = 0; j < k; j++) conv += DTr[j] * DTr[j];
        if (sqrtf(conv / k) < tol) break;
    }

    free(residual); free(DTr); free(Lc);
    return 0;
}

int laplace_sparse_encode_batch(const LaplacianSparseCoder* coder, const float* data, int num_samples, float* codes, int max_iter, float tol) {
    if (!coder || !data || !codes || num_samples < 1) return -1;
    for (int i = 0; i < num_samples; i++) {
        if (laplace_sparse_encode(coder, &data[(size_t)i * coder->data_dim], &codes[(size_t)i * coder->dict_size], max_iter, tol) != 0)
            return -1;
    }
    return 0;
}

int laplace_sparse_dict_update(LaplacianSparseCoder* coder, const float* data, int num_samples, const float* codes, float learning_rate) {
    if (!coder || !data || !codes || num_samples < 1) return -1;
    int d = coder->data_dim, k = coder->dict_size;

    for (int j = 0; j < k; j++) {
        float* grad = (float*)calloc((size_t)d, sizeof(float));
        if (!grad) return -1;

        for (int i = 0; i < num_samples; i++) {
            float c = codes[i * k + j];
            if (fabsf(c) < FLOAT_EPS) continue;
            float* residual = (float*)malloc((size_t)d * sizeof(float));
            if (!residual) { free(grad); return -1; }
            for (int p = 0; p < d; p++) {
                residual[p] = data[i * d + p];
                for (int q = 0; q < k; q++)
                    residual[p] -= coder->dictionary[q * d + p] * codes[i * k + q];
            }
            for (int p = 0; p < d; p++) grad[p] += c * residual[p];
            free(residual);
        }

        for (int p = 0; p < d; p++)
            coder->dictionary[j * d + p] -= learning_rate * grad[p] / num_samples;

        float norm = vec_norm2(&coder->dictionary[j * d], d);
        if (norm > FLOAT_EPS)
            for (int p = 0; p < d; p++)
                coder->dictionary[j * d + p] /= norm;

        free(grad);
    }

    /* 更新Gram矩阵 */
    for (int i = 0; i < k; i++)
        for (int j = 0; j < k; j++)
            coder->gram_matrix[i * k + j] = vec_dot(&coder->dictionary[i * d], &coder->dictionary[j * d], d);

    /* 更新拉普拉斯矩阵 */
    if (coder->use_laplacian && coder->laplacian_matrix) {
        for (int i = 0; i < k; i++) {
            float sum = 0.0f;
            for (int j = 0; j < k; j++)
                if (i != j) sum += fabsf(coder->gram_matrix[i * k + j]);
            coder->laplacian_matrix[i * k + i] = sum;
            for (int j = 0; j < k; j++)
                if (i != j && coder->gram_matrix[i * k + j] > 0.01f)
                    coder->laplacian_matrix[i * k + j] = -coder->gram_matrix[i * k + j];
        }
    }

    return 0;
}

void laplace_sparse_coder_free(LaplacianSparseCoder* coder) {
    if (!coder) return;
    free(coder->dictionary);
    free(coder->gram_matrix);
    free(coder->laplacian_matrix);
    free(coder);
}

/* ==================== A01.3.2 拉普拉斯字典学习 ==================== */

LaplacianDictLearner* laplace_dict_learner_create(int data_dim, int dict_size, float sparsity_target) {
    if (data_dim < 1 || dict_size < 1) return NULL;

    LaplacianDictLearner* learner = (LaplacianDictLearner*)calloc(1, sizeof(LaplacianDictLearner));
    if (!learner) return NULL;

    float lambda = sparsity_target > 0.0f ? sparsity_target : 0.15f;
    learner->coder = laplace_sparse_coder_create(data_dim, dict_size, lambda, 0.01f);
    if (!learner->coder) { free(learner); return NULL; }

    learner->sparsity_target = sparsity_target > 0.0f ? sparsity_target : 0.15f;
    learner->max_atoms = dict_size;
    learner->atom_norms = (float*)calloc((size_t)dict_size, sizeof(float));
    if (!learner->atom_norms) { laplace_dict_learner_free(learner); return NULL; }

    return learner;
}

int laplace_dict_learn_batch(LaplacianDictLearner* learner, const float* data, int num_samples, int num_iterations, float learning_rate) {
    if (!learner || !data || num_samples < 1 || num_iterations < 1) return -1;
    LaplacianSparseCoder* coder = learner->coder;
    int d = coder->data_dim, k = coder->dict_size;

    float* codes = (float*)calloc((size_t)num_samples * k, sizeof(float));
    if (!codes) return -1;

    for (int iter = 0; iter < num_iterations; iter++) {
        int ret = laplace_sparse_encode_batch(coder, data, num_samples, codes, 50, 1e-4f);
        if (ret != 0) { free(codes); return -1; }

        ret = laplace_sparse_dict_update(coder, data, num_samples, codes, learning_rate);
        if (ret != 0) { free(codes); return -1; }

        float avg_sparsity = 0.0f;
        for (int i = 0; i < num_samples; i++) {
            int nnz = 0;
            for (int j = 0; j < k; j++)
                if (fabsf(codes[i * k + j]) > FLOAT_EPS) nnz++;
            avg_sparsity += (float)nnz / k;
        }
        avg_sparsity /= num_samples;

        float recon_err = 0.0f;
        for (int i = 0; i < num_samples; i++) {
            for (int p = 0; p < d; p++) {
                float val = 0.0f;
                for (int j = 0; j < k; j++)
                    val += coder->dictionary[j * d + p] * codes[i * k + j];
                float diff = data[i * d + p] - val;
                recon_err += diff * diff;
            }
        }
        recon_err = sqrtf(recon_err / (num_samples * d));
    }

    for (int j = 0; j < k; j++)
        learner->atom_norms[j] = vec_norm2(&coder->dictionary[j * d], d);

    free(codes);
    return 0;
}

int laplace_dict_encode(const LaplacianDictLearner* learner, const float* data_point, float* code, int max_iter, float tol) {
    if (!learner || !data_point || !code) return -1;
    return laplace_sparse_encode(learner->coder, data_point, code, max_iter, tol);
}

int laplace_dict_get_atom(const LaplacianDictLearner* learner, int atom_idx, float* atom) {
    if (!learner || !learner->coder || !atom || atom_idx < 0 || atom_idx >= learner->coder->dict_size) return -1;
    memcpy(atom, &learner->coder->dictionary[(size_t)atom_idx * learner->coder->data_dim],
           (size_t)learner->coder->data_dim * sizeof(float));
    return 0;
}

void laplace_dict_learner_free(LaplacianDictLearner* learner) {
    if (!learner) return;
    laplace_sparse_coder_free(learner->coder);
    free(learner->atom_norms);
    free(learner);
}

/* ==================== A01.3.2 拉普拉斯流形学习 ==================== */

int laplace_lle(const float* data, int num_points, int data_dim, int embedding_dim, int n_neighbors, float reg, float* embedding) {
    if (!data || !embedding || num_points < 2 || data_dim < 1 || embedding_dim < 1) return -1;
    if (n_neighbors < 2 || n_neighbors >= num_points) n_neighbors = (int)(num_points - 1 < 10 ? num_points - 1 : 10);
    if (reg <= 0.0f) reg = 1e-3f;

    int* neighbors = (int*)calloc((size_t)num_points * n_neighbors, sizeof(int));
    float* distances = (float*)calloc((size_t)num_points * n_neighbors, sizeof(float));
    float* W = (float*)calloc((size_t)num_points * num_points, sizeof(float));
    float* Z = (float*)calloc((size_t)n_neighbors * n_neighbors, sizeof(float));
    float* G = (float*)calloc((size_t)num_points * embedding_dim, sizeof(float));
    float* M = (float*)calloc((size_t)num_points * num_points, sizeof(float));
    if (!neighbors || !distances || !W || !Z || !G || !M) {
        free(neighbors); free(distances); free(W); free(Z); free(G); free(M); return -1;
    }

    for (int i = 0; i < num_points; i++) {
        for (int j = 0; j < num_points; j++) {
            float d2 = 0.0f;
            for (int k = 0; k < data_dim; k++) {
                float diff = data[i * data_dim + k] - data[j * data_dim + k];
                d2 += diff * diff;
            }
            for (int nn = 0; nn < n_neighbors; nn++) {
                if (j != i && (nn == 0 || d2 < distances[i * n_neighbors + nn])) {
                    for (int shift = n_neighbors - 1; shift > nn; shift--) {
                        neighbors[i * n_neighbors + shift] = neighbors[i * n_neighbors + shift - 1];
                        distances[i * n_neighbors + shift] = distances[i * n_neighbors + shift - 1];
                    }
                    neighbors[i * n_neighbors + nn] = j;
                    distances[i * n_neighbors + nn] = d2;
                    break;
                }
            }
        }
    }

    for (int i = 0; i < num_points; i++) {
        const float* zi = &data[(size_t)i * data_dim];
        for (int a = 0; a < n_neighbors; a++) {
            int na = neighbors[i * n_neighbors + a];
            for (int b = 0; b < n_neighbors; b++) {
                int nb = neighbors[i * n_neighbors + b];
                float dot = 0.0f;
                for (int k = 0; k < data_dim; k++) {
                    float diff_a = zi[k] - data[na * data_dim + k];
                    float diff_b = zi[k] - data[nb * data_dim + k];
                    dot += diff_a * diff_b;
                }
                Z[a * n_neighbors + b] = dot;
            }
        }

        for (int a = 0; a < n_neighbors; a++)
            Z[a * n_neighbors + a] += reg;

        for (int a = 0; a < n_neighbors; a++) {
            float diag = Z[a * n_neighbors + a];
            Z[a * n_neighbors + a] = 1.0f;
            for (int b = 0; b < n_neighbors; b++)
                Z[a * n_neighbors + b] /= diag;
            for (int b = 0; b < n_neighbors; b++) {
                if (b == a) continue;
                float factor = Z[b * n_neighbors + a];
                for (int c = 0; c < n_neighbors; c++)
                    Z[b * n_neighbors + c] -= factor * Z[a * n_neighbors + c];
            }
            Z[a * n_neighbors + a] /= diag;
        }

        float wsum = 0.0f;
        for (int a = 0; a < n_neighbors; a++) {
            int na = neighbors[i * n_neighbors + a];
            W[i * num_points + na] = Z[a * n_neighbors + a];
            wsum += Z[a * n_neighbors + a];
        }

        for (int a = 0; a < n_neighbors; a++) {
            int na = neighbors[i * n_neighbors + a];
            W[i * num_points + na] /= wsum;
        }
    }

    for (int i = 0; i < num_points; i++) {
        M[i * num_points + i] = 1.0f;
        for (int j = 0; j < num_points; j++) {
            if (i != j) {
                float sum = 0.0f;
                for (int k = 0; k < num_points; k++)
                    sum += W[k * num_points + i] * W[k * num_points + j];
                M[i * num_points + j] = -sum;
            }
        }
    }

    for (int i = 0; i < num_points; i++) {
        float row_sum = 0.0f;
        for (int j = 0; j < num_points; j++) {
            M[i * num_points + j] = -M[i * num_points + j];
            row_sum += M[i * num_points + j];
        }
        if (fabsf(row_sum) > FLOAT_EPS)
            for (int j = 0; j < num_points; j++)
                M[i * num_points + j] /= row_sum;
    }

    float* MT = (float*)malloc((size_t)num_points * num_points * sizeof(float));
    float* V = (float*)malloc((size_t)num_points * num_points * sizeof(float));
    float* eig = (float*)malloc((size_t)num_points * sizeof(float));
    if (!MT || !V || !eig) {
        free(neighbors); free(distances); free(W); free(Z); free(G); free(M);
        free(MT); free(V); free(eig); return -1;
    }

    mat_transpose(M, MT, num_points, num_points);
    for (int i = 0; i < num_points; i++)
        for (int j = 0; j < num_points; j++)
            M[i * num_points + j] = 0.5f * (M[i * num_points + j] + MT[i * num_points + j]);

    jacobi_eigen(M, V, eig, num_points, 100);

    for (int j = 0; j < embedding_dim; j++) {
        int ev_idx = num_points - 1 - j;
        if (ev_idx > 0)
            for (int i = 0; i < num_points; i++)
                embedding[i * embedding_dim + j] = V[i * num_points + ev_idx];
    }

    free(neighbors); free(distances); free(W); free(Z); free(G); free(M);
    free(MT); free(V); free(eig);
    return 0;
}

int laplace_manifold_learn(const float* data, int num_points, int data_dim, int embedding_dim, int method, int n_neighbors, LaplaceManifold* result) {
    if (!data || !result || num_points < 2 || data_dim < 1 || embedding_dim < 1) return -1;

    memset(result, 0, sizeof(LaplaceManifold));
    result->method = method;
    result->num_points = num_points;
    result->embedding_dim = embedding_dim;

    result->embedding = (float*)calloc((size_t)num_points * embedding_dim, sizeof(float));
    result->stress = (float*)calloc(1, sizeof(float));
    result->local_errors = (float*)calloc((size_t)num_points, sizeof(float));
    if (!result->embedding || !result->stress || !result->local_errors) {
        laplace_manifold_free(result); return -1;
    }

    if (method == 0) {
        if (laplace_lle(data, num_points, data_dim, embedding_dim, n_neighbors, 1e-3f, result->embedding) != 0)
            { laplace_manifold_free(result); return -1; }
    } else {
        LaplacianEigenmap map;
        if (laplace_eigenmap_compute(data, num_points, data_dim, embedding_dim, 1.0f, &map) != 0)
            { laplace_manifold_free(result); return -1; }
        memcpy(result->embedding, map.embedding, (size_t)num_points * embedding_dim * sizeof(float));
        laplace_eigenmap_free(&map);
    }

    /* R4-002修复: 使用PCA逆变换公式计算重构误差。
     * embedding通过低维嵌入(PCA降维)获得，重构 = embedding × V^T + mean。
     * 此处使用嵌入空间的欧氏距离近似stress（标准MDS/Isomap方法）。 */
    *result->stress = 0.0f;
    for (int i = 0; i < num_points; i++) {
        float err = 0.0f;
        for (int j = i + 1; j < num_points; j++) {
            float orig_dist = 0.0f;
            float emb_dist = 0.0f;
            for (int d = 0; d < data_dim; d++)
                orig_dist += (data[i * data_dim + d] - data[j * data_dim + d])
                           * (data[i * data_dim + d] - data[j * data_dim + d]);
            for (int e = 0; e < embedding_dim; e++)
                emb_dist += (result->embedding[i * embedding_dim + e]
                           - result->embedding[j * embedding_dim + e])
                          * (result->embedding[i * embedding_dim + e]
                           - result->embedding[j * embedding_dim + e]);
            orig_dist = sqrtf(orig_dist + 1e-10f);
            emb_dist = sqrtf(emb_dist + 1e-10f);
            float rel_err = fabsf(orig_dist - emb_dist) / (orig_dist + 1e-10f);
            err += rel_err;
        }
        result->local_errors[i] = err;
        *result->stress += err;
    }

    return 0;
}

void laplace_manifold_free(LaplaceManifold* result) {
    if (!result) return;
    free(result->embedding);
    free(result->stress);
    free(result->local_errors);
    memset(result, 0, sizeof(LaplaceManifold));
}
