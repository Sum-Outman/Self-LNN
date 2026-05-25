#include "selflnn/robot/sensor_simulation.h"
#include "selflnn/utils/memory_utils.h"
#include "selflnn/utils/math_utils.h"
#include "selflnn/robot/kinematics.h"
#include <math.h>
#include <string.h>
#include <float.h>

/* quat_multiply/quat_conjugate/quat_rotate: math_utils.h inline versions */

void sensor_sim_init(SensorSimContext* ctx) {
    if (!ctx) return;
#ifdef SELFLNN_STRICT_REAL_DATA
    (void)ctx; return;
#else
    memset(ctx, 0, sizeof(SensorSimContext));
    ctx->lidar_config.fov_horizontal = 1.57f;
    ctx->lidar_config.max_range = 100.0f;
    ctx->lidar_config.num_rays = 360;
    ctx->depth_config.width = 640;
    ctx->depth_config.height = 480;
    ctx->depth_config.max_depth = 20.0f;
    ctx->imu_config.gyro_noise_stddev = 0.01f;
    ctx->imu_config.accel_noise_stddev = 0.1f;
    ctx->rng_state = 12345u;
    ctx->initialized = 1;
#endif
}

void sensor_sim_set_lidar_config(SensorSimContext* ctx, const SensorSimLidarConfig* config) {
    if (!ctx || !config) return;
    ctx->lidar_config = *config;
}

void sensor_sim_set_depth_config(SensorSimContext* ctx, const SensorSimDepthCameraConfig* config) {
    if (!ctx || !config) return;
    ctx->depth_config = *config;
}

void sensor_sim_set_imu_config(SensorSimContext* ctx, const SensorSimImuConfig* config) {
    if (!ctx || !config) return;
    ctx->imu_config = *config;
}

int sensor_sim_lidar_scan(SensorSimContext* ctx,
                          const float* mount_pos, const float* mount_quat,
                          const SimPhysicsPipeline* pipeline,
                          SensorSimLidarPoint* points, int max_points) {
#ifdef SELFLNN_STRICT_REAL_DATA
    (void)ctx; (void)mount_pos; (void)mount_quat; (void)pipeline; (void)points; (void)max_points;
    return -1;
#else
    if (!mount_pos || !mount_quat || !points || max_points <= 0) return -1;
    int num_rays = ctx->lidar_config.num_rays;
    if (num_rays > max_points) num_rays = max_points;
    int count = 0;
    /* 扫描每个射线，与物理管道中的所有碰撞体求交点 */
    const SimPhysicsPipeline* pp = pipeline;
    for (int i = 0; i < num_rays && count < max_points; i++) {
        float angle = ctx->lidar_config.angle_min +
                      (float)i / (float)num_rays * (ctx->lidar_config.angle_max - ctx->lidar_config.angle_min);
        float ray_dir[3] = {cosf(angle), sinf(angle), 0};
        float world_dir[3];
        quat_rotate(mount_quat, ray_dir, world_dir);
        float best_t = ctx->lidar_config.max_range;
        int hit = 0;
        /* ZSFABC-S010修复: 严格真实数据模式下禁用地面回退检测
         * 仅在有真实物理管线碰撞体时计算交点 */
        if (pp) {
            for (int oi = 0; oi < pp->object_count; oi++) {
                float obj_t;
                if (sensor_sim_ray_intersect(mount_pos, world_dir,
                    &pp->objects[oi], &obj_t, NULL, NULL) == 0) {
                    if (obj_t > ctx->lidar_config.min_range && obj_t < best_t) {
                        best_t = obj_t; hit = 1;
                    }
                }
            }
        }
        if (hit) {
            points[count].point[0] = mount_pos[0] + world_dir[0] * best_t;
            points[count].point[1] = mount_pos[1] + world_dir[1] * best_t;
            points[count].point[2] = mount_pos[2] + world_dir[2] * best_t;
            points[count].range = best_t;
            points[count].intensity = 0.8f;
            points[count].valid = 1;
            count++;
        }
    }
    return count;
#endif
}

int sensor_sim_depth_camera(SensorSimContext* ctx,
                            const float* mount_pos, const float* mount_quat,
                            const SimPhysicsPipeline* pipeline,
                            SensorSimDepthPixel* pixels, int max_pixels) {
#ifdef SELFLNN_STRICT_REAL_DATA
    (void)ctx; (void)mount_pos; (void)mount_quat; (void)pipeline; (void)pixels; (void)max_pixels;
    return -1;
#else
    if (!mount_pos || !mount_quat || !pixels || max_pixels <= 0) return -1;

    int w = ctx->depth_config.width, h = ctx->depth_config.height;
    int total = w * h;
    if (total > max_pixels) total = max_pixels;
    float max_d = ctx->depth_config.max_depth;

    const SimPhysicsPipeline* pp = pipeline;

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = y * w + x;
            if (idx >= max_pixels) break;

            float u = ((float)x / (float)w - 0.5f) * 2.0f * tanf(ctx->depth_config.fov * 0.5f);
            float v = ((float)y / (float)h - 0.5f) * 2.0f * tanf(ctx->depth_config.fov * 0.5f);
            float dir[3] = {u, v, -1};
            float len = sqrtf(u*u + v*v + 1.0f);
            if (len > 0) { dir[0] /= len; dir[1] /= len; dir[2] /= len; }
            float world_dir[3];
            quat_rotate(mount_quat, dir, world_dir);

            /* 对物理管线中的碰撞体进行射线-球体相交检测 */
            float min_t = max_d;
            int hit = 0;

            if (pp && pp->object_count > 0) {
                for (int bi = 0; bi < pp->object_count && bi < 16; bi++) {
                    float bc[3] = {
                        pp->objects[bi].world_transform[0],
                        pp->objects[bi].world_transform[1],
                        pp->objects[bi].world_transform[2]
                    };
                    float br = pp->objects[bi].radius > 0.0f ? pp->objects[bi].radius : 0.3f;

                    float oc[3] = {
                        mount_pos[0] - bc[0],
                        mount_pos[1] - bc[1],
                        mount_pos[2] - bc[2]
                    };
                    float a_val = world_dir[0]*world_dir[0] + world_dir[1]*world_dir[1] + world_dir[2]*world_dir[2];
                    float b_val = 2.0f * (oc[0]*world_dir[0] + oc[1]*world_dir[1] + oc[2]*world_dir[2]);
                    float c_val = oc[0]*oc[0] + oc[1]*oc[1] + oc[2]*oc[2] - br*br;
                    float disc = b_val*b_val - 4.0f*a_val*c_val;

                    if (disc >= 0) {
                        float t = (-b_val - sqrtf(disc)) / (2.0f * a_val);
                        if (t > 0.001f && t < min_t) {
                            min_t = t;
                            hit = 1;
                        }
                    }
                }
            }

            /* 如果物理管线没有交点，回退到地平面检测 */
            if (!hit && world_dir[2] < -0.01f) {
                float t = -mount_pos[2] / world_dir[2];
                if (t > 0 && t < max_d) min_t = t;
            }

            if (min_t < max_d) {
                pixels[idx].point[0] = mount_pos[0] + world_dir[0] * min_t;
                pixels[idx].point[1] = mount_pos[1] + world_dir[1] * min_t;
                pixels[idx].point[2] = mount_pos[2] + world_dir[2] * min_t;
                pixels[idx].depth = min_t;
                pixels[idx].valid = 1;
            }
        }
    }
    return total;
#endif
}

int sensor_sim_imu_update(SensorSimContext* ctx,
                          const float* linear_accel, const float* angular_vel,
                          const float* orientation, float dt,
                          SensorSimImuFrame* frame) {
#ifdef SELFLNN_STRICT_REAL_DATA
    (void)ctx; (void)linear_accel; (void)angular_vel; (void)orientation; (void)dt; (void)frame;
    return -1;
#else
    if (!frame) return -1;
    (void)dt;
    memset(frame, 0, sizeof(SensorSimImuFrame));
    if (angular_vel) {
        frame->angular_velocity[0] = angular_vel[0];
        frame->angular_velocity[1] = angular_vel[1];
        frame->angular_velocity[2] = angular_vel[2];
    }
    if (linear_accel && orientation) {
        float g_world[3] = {0, 0, -9.81f};
        float inv_q[4], g_body[3];
        quat_conjugate(orientation, inv_q);
        quat_rotate(inv_q, g_world, g_body);
        frame->acceleration[0] = linear_accel[0] + g_body[0];
        frame->acceleration[1] = linear_accel[1] + g_body[1];
        frame->acceleration[2] = linear_accel[2] + g_body[2];
    } else if (linear_accel) {
        frame->acceleration[0] = linear_accel[0];
        frame->acceleration[1] = linear_accel[1];
        frame->acceleration[2] = linear_accel[2];
    }
    if (orientation) memcpy(frame->orientation, orientation, 4 * sizeof(float));
    frame->valid = 1;
    return 0;
#endif
}

/* ============================================================================
 * 传感器仿真内部辅助函数 — 编译时受 SELFLNN_STRICT_REAL_DATA 保护
 * 严格模式下这些函数不参与编译，彻底消除仿真数据生成代码路径
 * ============================================================================ */
#ifndef SELFLNN_STRICT_REAL_DATA

/* 射线-球体相交：二次方程判别式 */
static int ray_sphere_intersect(const float* ro, const float* rd,
                                const float* center, float radius,
                                float* t_out) {
    float oc[3] = {ro[0]-center[0], ro[1]-center[1], ro[2]-center[2]};
    float a = rd[0]*rd[0]+rd[1]*rd[1]+rd[2]*rd[2];
    float b = 2.0f*(oc[0]*rd[0]+oc[1]*rd[1]+oc[2]*rd[2]);
    float c = oc[0]*oc[0]+oc[1]*oc[1]+oc[2]*oc[2]-radius*radius;
    float d = b*b-4.0f*a*c;
    if (d < 0) return 0;
    float sqrt_d = sqrtf(d);
    float t0 = (-b-sqrt_d)/(2.0f*a);
    float t1 = (-b+sqrt_d)/(2.0f*a);
    if (t0 > 1e-6f) { *t_out=t0; return 1; }
    if (t1 > 1e-6f) { *t_out=t1; return 1; }
    return 0;
}

/* 射线-AABB盒相交：Slab方法 */
static int ray_aabb_intersect(const float* ro, const float* rd,
                              const float* bmin, const float* bmax,
                              float* t_out) {
    float tmin=-1e9f, tmax=1e9f;
    for (int i=0;i<3;i++) {
        if (fabsf(rd[i])<1e-8f) {
            if (ro[i]<bmin[i]||ro[i]>bmax[i]) return 0;
            continue;
        }
        float inv_d = 1.0f/rd[i];
        float t0=(bmin[i]-ro[i])*inv_d;
        float t1=(bmax[i]-ro[i])*inv_d;
        if (t0>t1) { float tmp=t0; t0=t1; t1=tmp; }
        if (t0>tmin) tmin=t0;
        if (t1<tmax) tmax=t1;
        if (tmin>tmax) return 0;
    }
    if (tmin>1e-6f) { *t_out=tmin; return 1; }
    if (tmax>1e-6f) { *t_out=tmax; return 1; }
    return 0;
}

/* 射线-胶囊相交：分段圆柱+两个半球 */
static int ray_capsule_intersect(const float* ro, const float* rd,
                                 const float* c, float r, float h,
                                 float* t_out) {
    float best_t=1e9f; int hit=0;
    float top[3]={c[0],c[1]+h*0.5f,c[2]};
    float bot[3]={c[0],c[1]-h*0.5f,c[2]};
    /* 上半球 */
    float t;
    if (ray_sphere_intersect(ro,rd,top,r,&t)&&t<best_t) { best_t=t; hit=1; }
    /* 下半球 */
    if (ray_sphere_intersect(ro,rd,bot,r,&t)&&t<best_t) { best_t=t; hit=1; }
    /* 圆柱体：射线到线段距离 */
    float d[3]={top[0]-bot[0],top[1]-bot[1],top[2]-bot[2]};
    float len_h = sqrtf(d[0]*d[0]+d[1]*d[1]+d[2]*d[2])+1e-10f;
    float ax[3]={d[0]/len_h,d[1]/len_h,d[2]/len_h};
    float wo[3]={ro[0]-bot[0],ro[1]-bot[1],ro[2]-bot[2]};
    float a = 1.0f-(rd[0]*ax[0]+rd[1]*ax[1]+rd[2]*ax[2]);
    a*=a;
    float b = 2.0f*((wo[0]-(rd[0]*ax[0]+rd[1]*ax[1]+rd[2]*ax[2])*ax[0])*
                    (rd[0]-(rd[0]*ax[0]+rd[1]*ax[1]+rd[2]*ax[2])*ax[0]));
    float cc = wo[0]*wo[0]+wo[1]*wo[1]+wo[2]*wo[2]
               -(wo[0]*ax[0]+wo[1]*ax[1]+wo[2]*ax[2]);
    cc=cc-r*r;
    float disc=b*b-4.0f*a*cc;
    if (disc>=0) {
        float sqrt_d=sqrtf(disc);
        float t0=(-b-sqrt_d)/(2.0f*a);
        float t1=(-b+sqrt_d)/(2.0f*a);
        for (int k=0;k<2;k++) {
            float tc=(k==0)?t0:t1;
            if (tc>1e-6f&&tc<best_t) {
                float p[3]={ro[0]+rd[0]*tc,ro[1]+rd[1]*tc,ro[2]+rd[2]*tc};
                float proj=(p[0]-bot[0])*ax[0]+(p[1]-bot[1])*ax[1]+(p[2]-bot[2])*ax[2];
                if (proj>=-1e-4f&&proj<=len_h+1e-4f) { best_t=tc; hit=1; }
            }
        }
    }
    if (hit) { *t_out=best_t; return 1; }
    return 0;
}

#endif /* !SELFLNN_STRICT_REAL_DATA */

int sensor_sim_ray_intersect(const float* ray_origin, const float* ray_dir,
                             const SimCollisionObject* object,
                             float* hit_distance, float* hit_point, float* hit_normal) {
#ifdef SELFLNN_STRICT_REAL_DATA
    (void)ray_origin; (void)ray_dir; (void)object; (void)hit_distance; (void)hit_point; (void)hit_normal;
    return -1;
#else
    if (!ray_origin || !ray_dir || !object || !hit_distance) return -1;
    float t = 1e9f; int found = 0;
    float pos[3] = {object->world_transform[0],object->world_transform[1],object->world_transform[2]};

    switch (object->shape.type) {
     case COLLISION_SHAPE_SPHERE: { /* SPHERE */
        Vec3 sc = object->shape.data.sphere.center;
        float r = object->shape.data.sphere.radius;
        float sp_c[3]={pos[0]+sc.x,pos[1]+sc.y,pos[2]+sc.z};
        if (ray_sphere_intersect(ray_origin,ray_dir,sp_c,r,&t)) found=1;
        break;
    }
    case COLLISION_SHAPE_BOX: {
        Vec3 he = object->shape.data.box.half_extents;
        Vec3 bc = object->shape.data.box.center;
        float bmin[3]={pos[0]+bc.x-he.x,pos[1]+bc.y-he.y,pos[2]+bc.z-he.z};
        float bmax[3]={pos[0]+bc.x+he.x,pos[1]+bc.y+he.y,pos[2]+bc.z+he.z};
        if (ray_aabb_intersect(ray_origin,ray_dir,bmin,bmax,&t)) found=1;
        break;
    }
    case COLLISION_SHAPE_CAPSULE: {
        Vec3 cc = object->shape.data.capsule.center;
        float cr = object->shape.data.capsule.radius;
        float ch = object->shape.data.capsule.height;
        float cp[3]={pos[0]+cc.x,pos[1]+cc.y,pos[2]+cc.z};
        if (ray_capsule_intersect(ray_origin,ray_dir,cp,cr,ch,&t)) found=1;
        break;
    }
    case COLLISION_SHAPE_CYLINDER: {
        Vec3 cc = object->shape.data.cylinder.center;
        float cr = object->shape.data.cylinder.radius;
        float ch = object->shape.data.cylinder.height;
        float cp[3]={pos[0]+cc.x,pos[1]+cc.y,pos[2]+cc.z};
        if (ray_capsule_intersect(ray_origin,ray_dir,cp,cr,ch,&t)) found=1;
        break;
    }
    case COLLISION_SHAPE_PLANE: {
        Vec3 n = object->shape.data.plane.normal;
        float d = object->shape.data.plane.d;
        float denom = n.x*ray_dir[0]+n.y*ray_dir[1]+n.z*ray_dir[2];
        if (fabsf(denom)>1e-8f) {
            t = -(n.x*ray_origin[0]+n.y*ray_origin[1]+n.z*ray_origin[2]+d)/denom;
            if (t>1e-6f) found=1;
        }
        break;
    }
    default: break;
    }

    if (found && t > 1e-6f && t < 1000.0f) {
        *hit_distance = t;
        if (hit_point) {
            hit_point[0]=ray_origin[0]+ray_dir[0]*t;
            hit_point[1]=ray_origin[1]+ray_dir[1]*t;
            hit_point[2]=ray_origin[2]+ray_dir[2]*t;
        }
        if (hit_normal) {
            /* 近似法线：从命中点指向物体中心 */
            float cx=pos[0], cy=pos[1], cz=pos[2];
            float hx=hit_point?hit_point[0]:(ray_origin[0]+ray_dir[0]*t);
            float hy=hit_point?hit_point[1]:(ray_origin[1]+ray_dir[1]*t);
            float hz=hit_point?hit_point[2]:(ray_origin[2]+ray_dir[2]*t);
            float nx=hx-cx, ny=hy-cy, nz=hz-cz;
            float nl=sqrtf(nx*nx+ny*ny+nz*nz)+1e-10f;
            hit_normal[0]=nx/nl; hit_normal[1]=ny/nl; hit_normal[2]=nz/nl;
        }
        return 0;
    }
    return -1;
#endif
}
